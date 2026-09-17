"""Experiment tasks: each turns an :class:`ExperimentConfig` into an ordered list of units and runs one unit.

The runner owns identity, resume and persistence; tasks own the measurement. A task must be able to list its
units *before* the model is loaded (tokenizer/data only), so a fully finished experiment never loads weights.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch

from raq_kv.budget.bit_accounting import full_precision_kv_budget
from raq_kv.config.schema import CachePPLTask, ExperimentConfig, LatencyTask
from raq_kv.data.wikitext import PPLChunk, chunks_fingerprint, make_chunks, tokenize_wikitext2
from raq_kv.evaluation.latency import measure_generation_latency
from raq_kv.evaluation.memory import allocator_delta, cuda_memory, reset_peak
from raq_kv.evaluation.metrics import nll_stats
from raq_kv.evaluation.protocol import prefill, score_through_cache
from raq_kv.models.loader import LoadedModel, load_model, load_tokenizer


class Task(Protocol):
    def unit_keys(self) -> list[str]: ...
    def data_fingerprint(self) -> dict[str, Any]: ...
    def setup(self) -> dict[str, Any]: ...
    def run_unit(self, unit_key: str) -> dict[str, Any]: ...


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

        budget = full_precision_kv_budget(self.lm.geometry, snapshot.seq_len)
        record: dict[str, Any] = {
            "sample_id": unit_key,
            "tokens_original": chunk.prefix_len,
            "tokens_retained": snapshot.seq_len,
            "tokens_pruned": chunk.prefix_len - snapshot.seq_len,
            "compression_ratio": snapshot.seq_len / chunk.prefix_len,
            **nll_stats(result.token_nll),
            "kv_bytes_actual": snapshot.nbytes,
            "kv_bytes_analytic": budget.total_bytes,
            "kv_budget": budget.to_dict(),
            "avg_kv_bits": float(self.lm.geometry.dtype_bits),
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


def build_task(cfg: ExperimentConfig) -> Task:
    if isinstance(cfg.task, CachePPLTask):
        return CachePPL(cfg)
    if isinstance(cfg.task, LatencyTask):
        return Latency(cfg)
    raise NotImplementedError(f"no task for kind '{cfg.task.kind}'")
