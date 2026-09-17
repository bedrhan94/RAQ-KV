"""GSM8K subset for reasoning accuracy through the cache.

Phase split: the cache holds the chat-template prefix and the problem statement; the answer-format instruction and
the assistant header are phase 2. Generated reasoning tokens are produced in phase 2 (their own KV is not
transformed in M0-M2). Gold answers are taken from the ``#### <number>`` marker of the dataset.
"""

from __future__ import annotations

import random
from typing import Any

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from raq_kv.config.schema import GSM8KConfig
from raq_kv.data.prompts import PromptSample, Span, make_span, render_chat, split_index, tokenize_with_offsets
from raq_kv.evaluation.scoring import gsm8k_gold

INSTRUCTION = (
    "Solve the problem step by step. At the end, write the final numeric answer on its own line in the form "
    "'#### <number>'."
)


def build_gsm8k(
    cfg: GSM8KConfig, tokenizer: PreTrainedTokenizerBase, template_kwargs: dict[str, Any]
) -> list[PromptSample]:
    ds = load_dataset(cfg.hf_id, cfg.subset, split=cfg.split, revision=cfg.revision)
    if cfg.n_samples > len(ds):
        raise ValueError(f"requested {cfg.n_samples} samples, split has {len(ds)}")
    indices = sorted(random.Random(cfg.subset_seed).sample(range(len(ds)), cfg.n_samples))
    samples = []
    for idx in indices:
        row = ds[idx]
        problem = row["question"].strip()
        prompt = render_chat(tokenizer, problem + "\n\n" + INSTRUCTION, template_kwargs)
        doc_start = prompt.index(problem)
        cache_end_char = doc_start + len(problem) + 2
        ids, offsets = tokenize_with_offsets(tokenizer, prompt)
        cache_len = split_index(offsets, cache_end_char)
        spans = (
            Span("template_prefix", "context", 0, doc_start, 0, split_index(offsets, doc_start), prompt[:doc_start]),
            make_span(tokenizer, ids, offsets, prompt, "document", "context", doc_start, doc_start + len(problem)),
            Span("query", "query", cache_end_char, len(prompt), cache_len, len(ids), prompt[cache_end_char:]),
        )
        samples.append(PromptSample(
            sample_id=f"gsm8k-{idx:04d}",
            prompt_text=prompt,
            prompt_ids=torch.tensor(ids, dtype=torch.long),
            cache_len=cache_len,
            scorer="gsm8k",
            reference={"gold": gsm8k_gold(row["answer"])},
            gold_text=None,
            gold_scored_chars=None,
            max_new_tokens=cfg.max_new_tokens,
            spans=spans,
            meta={"dataset_index": idx, "cache_len": cache_len},
        ))
    return samples
