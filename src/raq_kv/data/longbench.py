"""LongBench QA rows as a hard-task vehicle: the one regime where Qwen3-1.7B does not answer at p = 1.

D62 answered why Llama-3.2-3B is more fault-fragile than Qwen3-1.7B — Qwen holds these answers at top-1 probability
**1.0000** and Llama at 0.9880, so a fault must do far more work to unseat the first — and stated its own limit:
saturation is a property of model × task. D63 tested that limit and it is real. On the probe no manipulation reached
Qwen: a tenfold increase in competing distractors did not move its accuracy (Fisher p = 1.000), and eight times the
context left it at 13 nats of margin. On `2wikimqa` it **desaturates completely**: top-1 probability **0.9284**
(min 0.5445), margin **2.875** against 16.5–19.5, and 25% of samples below 1 nat where the probe had none.

**What is still unmeasured, and what this vehicle is for: whether the fault-hazard ordering follows.** D62's account
predicts it should — if fragility is distance from the decision boundary, then a desaturated Qwen should become
*more* fragile, and the family ordering should compress or invert where the confidence ordering does. Nothing has
tested that, and it is what would close the line D59 opened.

**Read from the pinned `data.zip`, not the dataset script.** LongBench ships as `LongBench.py` plus `data.zip`, and
executing a remote script is neither reproducible under this project's provenance rules nor necessary: the zip is
pinned by commit SHA and the subsets inside are plain JSONL.

**Two hardware facts bound this vehicle and are enforced rather than noted.** Measured on a 16 GB card in the D63
pilot: **eager attention asks for 14.45 GiB at ~7k cached tokens** and OOMs, so the attention-scorer path (D5) — and
with it every retention and attention-ranking experiment — is unavailable here at any length; and **sdpa asks 19.36
GiB at ~15k** on the 3B model and OOMs. So rows are skipped above `max_cache_tokens`, the bound is a config field and
therefore part of the experiment's identity hash, and the skip count is recorded in every sample's `meta` so a result
states the population it ran on.
"""

from __future__ import annotations

import json
import zipfile
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from raq_kv.config.schema import LongBenchConfig
from raq_kv.data.prompts import PromptSample, Span, render_chat, split_index, tokenize_with_offsets

# LongBench's own prompt for the multi-hop QA subsets, verbatim, so this measures the benchmark's task and not one
# rewritten to suit the vehicle. The instruction is repeated before and after the passages in the original too.
TEMPLATE = (
    "Answer the question based on the given passages. Only give me the answer and do not output any other words."
    "\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give "
    "me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
)
QUESTION_MARKER = "\n\nQuestion:"


def _rows(cfg: LongBenchConfig) -> list[dict[str, Any]]:
    """The subset's rows, from the pinned archive. Downloads once and caches under the HF hub cache."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(cfg.hf_id, "data.zip", repo_type="dataset", revision=cfg.revision)
    with zipfile.ZipFile(path) as archive:
        raw = archive.open(f"data/{cfg.subset}.jsonl").read().decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def build_longbench(cfg: LongBenchConfig, tokenizer: PreTrainedTokenizerBase,
                    template_kwargs: dict[str, Any]) -> list[PromptSample]:
    """Samples in file order, skipping rows whose cached context exceeds the configured bound.

    File order rather than a random subset: the rows are already in no meaningful order, and a seeded sample would add
    a knob whose only effect is to make two configs that look alike select differently. The bound does the selecting
    and it is stated in the config.
    """
    samples: list[PromptSample] = []
    skipped = 0
    for row in _rows(cfg):
        if len(samples) >= cfg.n_samples:
            break
        body = TEMPLATE.format(context=row["context"], input=row["input"])
        prompt = render_chat(tokenizer, body, template_kwargs)
        ids, offsets = tokenize_with_offsets(tokenizer, prompt)
        # The passages are cached and the question is the query chunk, the same split every arm here uses. `rindex`
        # because the instruction text itself contains the word "question" -- the marker is the last occurrence.
        cut = prompt.rindex(QUESTION_MARKER)
        cache_len = split_index(offsets, cut)
        if cache_len > cfg.max_cache_tokens:
            skipped += 1
            continue
        answers = [a for a in row.get("answers", []) if a]
        if not answers:
            skipped += 1
            continue
        samples.append(PromptSample(
            sample_id=f"{cfg.subset}-{row['_id']}",
            prompt_text=prompt,
            prompt_ids=torch.tensor(ids, dtype=torch.long),
            cache_len=cache_len,
            scorer="longbench_qa_f1",
            reference={"answers": answers},
            # No gold continuation is scored: a free-form phrase has no single tokenization, so a teacher-forced NLL
            # over it would measure the reference's wording as much as the model. F1 over the generation is the
            # benchmark's own metric and is what this reports.
            gold_text=None,
            gold_scored_chars=None,
            max_new_tokens=cfg.max_new_tokens,
            spans=(
                Span("document", "context", 0, cut, 0, cache_len, prompt[:cut]),
                Span("query", "query", cut, len(prompt), cache_len, len(ids), prompt[cut:]),
            ),
            meta={"subset": cfg.subset, "row_id": row["_id"], "cache_len": cache_len,
                  "n_answers": len(answers), "context_chars": len(row["context"]),
                  "length_field": row.get("length")},
        ))
    if not samples:
        raise ValueError(f"no {cfg.subset} row fits max_cache_tokens={cfg.max_cache_tokens}; every row was skipped")
    for sample in samples:
        sample.meta["n_skipped_for_length"] = skipped
    return samples


# No `longbench_fingerprint` here, deliberately. The tasks hash their samples with `prompts.samples_fingerprint`, and
# the two knobs that change which rows are selected -- `max_cache_tokens` and `revision` -- are both in the config, so
# they already change the experiment's identity hash. A second fingerprint that folded in the skip count would be
# redundant protection that *looks* load-bearing, which is its own hazard: a reader would assume the ids alone are not
# enough. The skip count is still recorded in every sample's `meta` so a result can state the population it ran on.
