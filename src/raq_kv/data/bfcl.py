"""BFCL as a long-cache agent vehicle: a real tool call inside a realistic tool catalogue.

M6 asks whether token-level metrics under-estimate *agent* failures (C4). The plan named BFCL's AST categories
(simple / multiple / parallel) as the vehicle. **Measured first, they do not work**, for the same reason GSM8K
does not: the cache is far too short to compress.

    simple                   median  166 cache tokens, max  262
    multiple                 median  333               max  639
    parallel                 median  181               max  351
    multi_turn_long_context  median  157               max  363   <- despite the name

Against a probe that runs at 512-4096 tokens and a joint policy that prunes to 16-25% of it, pruning a 166-token
cache leaves ~27 tokens, of which the attention sink alone pins 4. There is nothing to measure. (BFCL's
"long_context" category earns its name from tool *responses* accumulated while executing, which needs the
stateful harness the plan flagged as its own decision point -- not a static prompt.)

**What does work is the scenario the benchmark leaves out: the tool catalogue.** A real agent carries dozens to
hundreds of function schemas in its prompt, and BFCL's own files supply **769 distinct schemas** at a median of
122 tokens each. Embedding a question's gold schema in a catalogue of N real distractors gives exactly the
range this project needs --

    10 schemas -> ~1374 cache tokens    25 -> ~3293    50 -> ~6357

-- while keeping two properties nothing else here has. First, **AST scoring is family-agnostic**: it asks what
was called with which arguments, not what syntax was used to say it. That is precisely the defect D46 found in
RAQ-Probe's own `tool_arg` type, where Llama scored 0 of 120 for emitting `parameters` instead of `arguments`.
Second, the retention question becomes the agent-shaped version of D34: can the compressor keep the *one relevant
schema* while discarding 75-90% of the catalogue? The gold schema is labelled as the target span and every
distractor schema as a distractor span, so `kept_fraction_target` means exactly what it means in M4.

The catalogue is rendered through the model's own chat template, so each family serialises tools in its own
convention. That is deliberate: the KV cache is what gets compressed, and its contents should be what that model
would really have cached.
"""

from __future__ import annotations

import json
import random
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from raq_kv.config.schema import BFCLConfig
from raq_kv.data.prompts import PromptSample, Span, char_span_to_tokens, render_chat, tokenize_with_offsets

HF_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
# The AST categories. `exec_*` needs real execution and `live_*` is user-submitted with unstable schemas; both
# are out of scope for a cache-compression vehicle, and neither is longer.
AST_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")


def _load_category(category: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    questions = [json.loads(line) for line in
                 open(hf_hub_download(HF_REPO, f"BFCL_v3_{category}.json", repo_type="dataset"),
                      encoding="utf-8") if line.strip()]
    answers = {a["id"]: a["ground_truth"] for a in
               (json.loads(line) for line in
                open(hf_hub_download(HF_REPO, f"possible_answer/BFCL_v3_{category}.json", repo_type="dataset"),
                     encoding="utf-8") if line.strip())}
    return questions, answers


def _schema_pool(categories: tuple[str, ...]) -> list[dict[str, Any]]:
    """Every distinct schema across the AST categories, in a deterministic order.

    Sorted by name rather than left in file order: the pool is sampled from, so a stable order is what makes the
    sample reproducible from the config's seed alone.
    """
    seen: dict[str, dict[str, Any]] = {}
    for category in categories:
        questions, _ = _load_category(category)
        for row in questions:
            functions = row.get("function")
            for fn in (functions if isinstance(functions, list) else [functions]):
                if isinstance(fn, dict) and fn.get("name") and fn["name"] not in seen:
                    seen[fn["name"]] = fn
    return [seen[name] for name in sorted(seen)]


def _first_user_turn(question: Any) -> str:
    """BFCL nests turns as list[list[{role, content}]]; the AST categories have exactly one user turn."""
    turns = question[0] if isinstance(question, list) and question and isinstance(question[0], list) else question
    for message in turns:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    raise ValueError("no user turn in BFCL question")


def build_bfcl(
    cfg: BFCLConfig, tokenizer: PreTrainedTokenizerBase, template_kwargs: dict[str, Any]
) -> list[PromptSample]:
    """One sample per (question, catalogue size), with the gold schema hidden among real distractors."""
    pool = _schema_pool(AST_CATEGORIES)
    samples: list[PromptSample] = []

    for category in cfg.categories:
        questions, answers = _load_category(category)
        questions = sorted(questions, key=lambda r: r["id"])
        for row in questions[: cfg.questions_per_category]:
            gold = row.get("function")
            gold_list = gold if isinstance(gold, list) else [gold]
            gold_names = {fn["name"] for fn in gold_list if isinstance(fn, dict)}
            ground_truth = answers.get(row["id"])
            if ground_truth is None or not gold_names:
                continue
            user_text = _first_user_turn(row["question"])

            for n_catalogue in cfg.catalogue_sizes:
                rng = random.Random(f"{cfg.generator_seed}|{row['id']}|{n_catalogue}")
                distractors = [fn for fn in pool if fn["name"] not in gold_names]
                # Deterministic per (seed, question, size): the same catalogue every run, and a different one
                # per size so the comparison is not one catalogue truncated.
                chosen = rng.sample(distractors, min(n_catalogue, len(distractors)))
                catalogue = chosen + gold_list
                rng.shuffle(catalogue)

                text = render_chat(tokenizer, user_text, template_kwargs, tools=catalogue)
                ids, offsets = tokenize_with_offsets(tokenizer, text)
                # The cache ends where the user's question begins. Setting `cache_len = len(ids)` prefilled the
                # whole prompt and left the decode nothing to consume -- the two-phase protocol's own guard
                # refused it, which is what that guard is for.
                #
                # This split is not a workaround, it is the deployment shape: a serving system caches the tool
                # catalogue once and reuses it across requests, while the question is per-request. So M6's
                # vehicle is a **shared prefix cache**, which is exactly the setting D36 named as this project's
                # largest open problem -- query-aware scoring cannot be computed before the query arrives.
                question_at = text.rfind(user_text[:60] if len(user_text) > 60 else user_text)
                if question_at <= 0:
                    continue
                cache_len, _ = char_span_to_tokens(offsets, question_at, question_at + 1)
                if not 0 < cache_len < len(ids):
                    continue
                spans: list[Span] = []
                for fn in catalogue:
                    # Locate each schema by its name in the rendered text. The template decides how the schema is
                    # serialised, so the name is the one anchor that survives every family's convention.
                    needle = f'"{fn["name"]}"'
                    start = text.find(needle)
                    if start < 0:
                        start = text.find(fn["name"])
                    if start < 0:
                        continue
                    # A schema's span runs from its name to the start of the next one, which is close enough to
                    # "the tokens describing this tool" for a retention diagnostic and never overlaps two golds.
                    end = min((p for p in (text.find(f'"{o["name"]}"', start + len(needle)) for o in catalogue)
                               if p and p > 0), default=-1)
                    if end < 0:
                        end = min(start + cfg.schema_span_chars, len(text))
                    is_gold = fn["name"] in gold_names
                    token_start, token_end = char_span_to_tokens(offsets, start, end)
                    spans.append(Span(label="tool_schema", role="target" if is_gold else "distractor",
                                      char_start=start, char_end=end, token_start=token_start,
                                      token_end=token_end, text=fn["name"], fact_type="tool_schema"))

                samples.append(PromptSample(
                    sample_id=f"{category}-n{n_catalogue:03d}-{row['id']}",
                    prompt_text=text,
                    prompt_ids=torch.tensor(ids, dtype=torch.int64),
                    cache_len=cache_len,
                    scorer="bfcl_ast",
                    reference={"ground_truth": ground_truth, "gold_names": sorted(gold_names)},
                    gold_text=None,
                    gold_scored_chars=None,
                    max_new_tokens=cfg.max_new_tokens,
                    spans=tuple(spans),
                    meta={"category": category, "bfcl_id": row["id"], "n_catalogue": n_catalogue,
                          "n_gold_schemas": len(gold_list), "cache_len": cache_len,
                          "query_tokens": len(ids) - cache_len,
                          "fact_type": "tool_schema", "nominal_len": n_catalogue},
                ))
    return samples


def pool_size() -> int:
    """Exposed so a test can pin the catalogue's supply without rebuilding every sample."""
    return len(_schema_pool(AST_CATEGORIES))
