"""Experiment tasks: each turns an :class:`ExperimentConfig` into an ordered list of units and runs one unit.

The runner owns identity, resume and persistence; tasks own the measurement. A task must be able to list its
units *before* the model is loaded (tokenizer/data only), so a fully finished experiment never loads weights.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch

from raq_kv.budget.bit_accounting import full_precision_kv_budget
from raq_kv.config.schema import (
    CachePPLTask,
    ExperimentConfig,
    GSM8KConfig,
    LatencyTask,
    QATask,
    RAQProbeConfig,
)
from raq_kv.data.gsm8k import build_gsm8k
from raq_kv.data.prompts import PromptSample, char_span_to_tokens, samples_fingerprint
from raq_kv.data.raq_probe import build_raq_probe
from raq_kv.data.wikitext import PPLChunk, chunks_fingerprint, make_chunks, tokenize_wikitext2
from raq_kv.evaluation.latency import measure_generation_latency
from raq_kv.evaluation.memory import allocator_delta, cuda_memory, reset_peak
from raq_kv.evaluation.metrics import nll_stats
from raq_kv.evaluation.protocol import greedy_decode, prefill, score_answer, score_through_cache, token_char_spans
from raq_kv.evaluation.scoring import SCORERS
from raq_kv.kv.snapshot import KVSnapshot
from raq_kv.models.loader import LoadedModel, load_model, load_tokenizer


class Task(Protocol):
    def unit_keys(self) -> list[str]: ...
    def data_fingerprint(self) -> dict[str, Any]: ...
    def setup(self) -> dict[str, Any]: ...
    def run_unit(self, unit_key: str) -> dict[str, Any]: ...


def kv_record_fields(lm: LoadedModel, snapshot: KVSnapshot, original_tokens: int) -> dict[str, Any]:
    """Token/retention and KV memory fields shared by all cache-reading tasks."""
    budget = full_precision_kv_budget(lm.geometry, snapshot.seq_len)
    return {
        "tokens_original": original_tokens,
        "tokens_retained": snapshot.seq_len,
        "tokens_pruned": original_tokens - snapshot.seq_len,
        "compression_ratio": snapshot.seq_len / original_tokens,
        "kv_bytes_actual": snapshot.nbytes,
        "kv_bytes_analytic": budget.total_bytes,
        "kv_budget": budget.to_dict(),
        "avg_kv_bits": float(lm.geometry.dtype_bits),
    }


class CachePPL:
    """Golden cache-conditioned perplexity on WikiText-2 chunks (M0: identity transform between phases)."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        assert isinstance(cfg.task, CachePPLTask)
        self.cfg = cfg
        self.tokenizer = load_tokenizer(cfg.model)
        self.chunks: dict[str, PPLChunk] = {
            c.sample_id: c for c in make_chunks(tokenize_wikitext2(cfg.dataset, self.tokenizer), cfg.dataset)
        }
        self.lm: LoadedModel | None = None

    def unit_keys(self) -> list[str]:
        return list(self.chunks)

    def data_fingerprint(self) -> dict[str, Any]:
        return {"wikitext2_chunks_sha256": chunks_fingerprint(list(self.chunks.values()))}

    def setup(self) -> dict[str, Any]:
        self.lm = load_model(self.cfg.model)
        return {"model_memory": self.lm.memory, "kv_geometry": self.lm.geometry.to_dict()}

    def run_unit(self, unit_key: str) -> dict[str, Any]:
        assert self.lm is not None, "setup() must be called first"
        chunk = self.chunks[unit_key]
        model, dev = self.lm.model, self.lm.device

        reset_peak(dev)
        mem0 = cuda_memory(dev)
        snapshot = prefill(model, chunk.context_ids.to(dev).unsqueeze(0))
        mem1 = cuda_memory(dev)

        # M0: identity transform. M1+ inserts quantize/encode/inject/decode/prune here.
        result = score_through_cache(
            model,
            snapshot,
            chunk.read_input_ids.to(dev).unsqueeze(0),
            chunk.read_target_ids.to(dev).unsqueeze(0),
        )
        mem2 = cuda_memory(dev)

        record: dict[str, Any] = {
            "sample_id": unit_key,
            **kv_record_fields(self.lm, snapshot, chunk.prefix_len),
            **nll_stats(result.token_nll),
            "prefill_cuda_allocated_delta_bytes": allocator_delta(mem0, mem1)["cuda_allocated_delta_bytes"],
            "unit_cuda_peak_over_start_bytes": allocator_delta(mem0, mem2)["cuda_peak_over_start_bytes"],
        }
        if self.cfg.task.store_token_nll:
            record["token_nll"] = [round(float(x), 6) for x in result.token_nll.cpu()]
        del snapshot, result
        return record


class Latency:
    """TTFT/TPOT/E2E of the full-precision serving path for several prompt lengths."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        assert isinstance(cfg.task, LatencyTask)
        self.cfg = cfg
        self.task: LatencyTask = cfg.task
        self.tokenizer = load_tokenizer(cfg.model)
        chunks = make_chunks(tokenize_wikitext2(cfg.dataset, self.tokenizer), cfg.dataset)
        self.corpus = torch.cat([c.token_ids for c in chunks])
        self._fingerprint = chunks_fingerprint(chunks)
        self.lm: LoadedModel | None = None

    def unit_keys(self) -> list[str]:
        return [f"len{n:05d}-rep{r:02d}" for n in self.task.prompt_lens for r in range(self.task.repeats)]

    def data_fingerprint(self) -> dict[str, Any]:
        return {"wikitext2_chunks_sha256": self._fingerprint}

    def setup(self) -> dict[str, Any]:
        self.lm = load_model(self.cfg.model)
        for n in self.task.prompt_lens:  # warm kernels/allocator for every shape before timing
            for _ in range(self.task.warmup):
                measure_generation_latency(self.lm.model, self._prompt(n), self.task.gen_len)
        return {"model_memory": self.lm.memory, "kv_geometry": self.lm.geometry.to_dict()}

    def _prompt(self, n: int) -> torch.Tensor:
        assert self.lm is not None
        return self.corpus[:n].to(self.lm.device).unsqueeze(0)

    def run_unit(self, unit_key: str) -> dict[str, Any]:
        assert self.lm is not None, "setup() must be called first"
        n = int(unit_key.split("-")[0][3:])
        dev = self.lm.device
        reset_peak(dev)
        mem0 = cuda_memory(dev)
        timing = measure_generation_latency(self.lm.model, self._prompt(n), self.task.gen_len)
        mem1 = cuda_memory(dev)
        final_tokens = n + self.task.gen_len - 1  # last generated token is never fed back
        return {
            "sample_id": unit_key,
            **timing,
            "kv_bytes_analytic_final": full_precision_kv_budget(self.lm.geometry, final_tokens).total_bytes,
            "unit_cuda_peak_over_start_bytes": allocator_delta(mem0, mem1)["cuda_peak_over_start_bytes"],
        }


class QAGenerate:
    """Answer generation through the cache (RAQ-Probe, GSM8K).

    Per sample: prefill the cached part -> (M0: identity transform) -> greedy decode the query through the snapshot ->
    score the decoded text; and, when a gold continuation exists, teacher-force it through the same snapshot and
    record the NLL of its scored sub-span (the primary continuous retrieval metric).
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        assert isinstance(cfg.task, QATask)
        self.cfg = cfg
        self.task: QATask = cfg.task
        self.tokenizer = load_tokenizer(cfg.model)
        template_kwargs = dict(cfg.model.chat_template_kwargs)
        if isinstance(cfg.dataset, RAQProbeConfig):
            samples = build_raq_probe(cfg.dataset, self.tokenizer, template_kwargs)
        elif isinstance(cfg.dataset, GSM8KConfig):
            samples = build_gsm8k(cfg.dataset, self.tokenizer, template_kwargs)
        else:
            raise NotImplementedError(cfg.dataset.name)
        self.samples: dict[str, PromptSample] = {s.sample_id: s for s in samples}
        if len(self.samples) != len(samples):
            raise ValueError("duplicate sample ids")
        self._fingerprint = samples_fingerprint(samples)
        self.lm: LoadedModel | None = None

    def unit_keys(self) -> list[str]:
        return list(self.samples)

    def data_fingerprint(self) -> dict[str, Any]:
        return {f"{self.cfg.dataset.name}_samples_sha256": self._fingerprint}

    def setup(self) -> dict[str, Any]:
        self.lm = load_model(self.cfg.model)
        return {"model_memory": self.lm.memory, "kv_geometry": self.lm.geometry.to_dict(),
                "eos_token_ids": sorted(self.lm.eos_token_ids)}

    @staticmethod
    def _generation_margins(tok: Any, gen: Any, prediction: str, gold: str | None) -> dict[str, Any]:
        """Decode-path margins: distance to a different token at each generated step.

        Unlike the teacher-forced margin these do not depend on our tokenization of the gold string, which matters
        for multi-token answers the model may split differently (observed on invented entity names).
        """
        out: dict[str, Any] = {
            "gen_margin_min_all": min(gen.margins) if gen.margins else None,
            "gen_margin_mean_all": sum(gen.margins) / len(gen.margins) if gen.margins else None,
            "gen_margin_min_value": None,
            "gen_value_tokens": None,
        }
        if not gold or gold not in prediction:
            return out
        spans, text = token_char_spans(tok, gen.tokens)
        start = text.find(gold)
        if start < 0:
            return out
        end = start + len(gold)
        inside = [m for (a, b), m in zip(spans, gen.margins, strict=True) if a < end and b > start]
        if inside:
            out["gen_margin_min_value"] = min(inside)
            out["gen_value_tokens"] = len(inside)
        return out

    def run_unit(self, unit_key: str) -> dict[str, Any]:
        assert self.lm is not None, "setup() must be called first"
        s = self.samples[unit_key]
        model, dev, tok = self.lm.model, self.lm.device, self.lm.tokenizer
        ids = s.prompt_ids.to(dev).unsqueeze(0)
        query_ids = ids[:, s.cache_len :]

        reset_peak(dev)
        mem0 = cuda_memory(dev)
        snapshot = prefill(model, ids[:, : s.cache_len])
        mem1 = cuda_memory(dev)

        # M0: identity transform. M1+ inserts quantize/encode/inject/decode/prune here.
        gen = greedy_decode(model, query_ids, s.max_new_tokens, snapshot=snapshot,
                            eos_token_ids=set(self.lm.eos_token_ids))
        prediction = tok.decode(gen.tokens, skip_special_tokens=True)
        gen_margins = self._generation_margins(tok, gen, prediction, s.reference.get("gold"))
        record: dict[str, Any] = {
            "sample_id": unit_key,
            "meta": s.meta,
            **kv_record_fields(self.lm, snapshot, s.cache_len),
            "query_tokens": int(query_ids.shape[1]),
            "prediction": prediction,
            "generated_tokens": len(gen.tokens),
            "stopped_on_eos": gen.stopped_on_eos,
            "reference": s.reference,
            **gen_margins,
            **SCORERS[s.scorer](prediction, s.reference),
        }

        if self.task.score_answer_nll and s.gold_text is not None and s.gold_scored_chars is not None:
            enc = tok(s.gold_text, add_special_tokens=False, return_offsets_mapping=True)
            gold_ids = torch.tensor(enc["input_ids"], device=dev).unsqueeze(0)
            res = score_answer(model, snapshot, query_ids, gold_ids)
            nll, margin = res.token_nll.cpu(), res.token_margin.cpu()
            t0, t1 = char_span_to_tokens([tuple(o) for o in enc["offset_mapping"]], *s.gold_scored_chars)
            record.update({
                # Primary continuous metric: distance to an argmax flip on the answer value tokens (does not saturate).
                "answer_margin_min": float(margin[t0:t1].min()),
                "answer_margin_mean": float(margin[t0:t1].mean()),
                "answer_first_token_margin": float(margin[t0]),
                # float64 NLL of the answer value tokens (float32 rounds confident tokens to exactly 0).
                "answer_nll_sum": float(nll[t0:t1].sum()),
                "answer_nll_mean": float(nll[t0:t1].mean()),
                "answer_first_token_nll": float(nll[t0]),
                "answer_n_tokens": t1 - t0,
                "gold_continuation_nll_mean": float(nll.mean()),
                "gold_continuation_margin_min": float(margin.min()),
            })

        mem2 = cuda_memory(dev)
        record["prefill_cuda_allocated_delta_bytes"] = allocator_delta(mem0, mem1)["cuda_allocated_delta_bytes"]
        record["unit_cuda_peak_over_start_bytes"] = allocator_delta(mem0, mem2)["cuda_peak_over_start_bytes"]
        # Token ranges of every labelled region; surface text only for facts (template/query text is large and static).
        record["spans"] = [
            {**span.to_dict(), "text": span.text if span.label.startswith("fact_") else None} for span in s.spans
        ]
        del snapshot
        return record


def build_task(cfg: ExperimentConfig) -> Task:
    if isinstance(cfg.task, CachePPLTask):
        return CachePPL(cfg)
    if isinstance(cfg.task, LatencyTask):
        return Latency(cfg)
    if isinstance(cfg.task, QATask):
        return QAGenerate(cfg)
    raise NotImplementedError(f"no task for kind '{cfg.task.kind}'")
