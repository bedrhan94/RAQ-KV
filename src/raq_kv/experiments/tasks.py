"""Experiment tasks: each turns an :class:`ExperimentConfig` into an ordered list of units and runs one unit.

The runner owns identity, resume and persistence; tasks own the measurement. A task must be able to list its
units *before* the model is loaded (tokenizer/data only), so a fully finished experiment never loads weights.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Protocol

import torch

from raq_kv.budget.bit_accounting import full_precision_kv_budget
from raq_kv.compression.retain import apply_retention, select_keep
from raq_kv.compression.spans import select_keep_spans, sentence_spans
from raq_kv.config.schema import (
    BFCLConfig,
    CachePPLTask,
    ExperimentConfig,
    FaultInjectionTask,
    FaultQATask,
    GSM8KConfig,
    LatencyTask,
    ProtectionArmTask,
    QATask,
    QuantSensitivityTask,
    QuantSpec,
    RAQProbeConfig,
    RetentionTask,
)
from raq_kv.data.bfcl import build_bfcl
from raq_kv.data.gsm8k import build_gsm8k
from raq_kv.data.prompts import PromptSample, char_span_to_tokens, samples_fingerprint
from raq_kv.data.raq_probe import build_raq_probe
from raq_kv.data.wikitext import PPLChunk, chunks_fingerprint, make_chunks, tokenize_wikitext2
from raq_kv.ecc.protect import build_protection
from raq_kv.evaluation.latency import measure_generation_latency
from raq_kv.evaluation.memory import allocator_delta, cuda_memory, reset_peak
from raq_kv.evaluation.metrics import nll_stats
from raq_kv.evaluation.protocol import greedy_decode, prefill, score_answer, score_through_cache, token_char_spans
from raq_kv.evaluation.scoring import SCORERS
from raq_kv.experiments.ids import experiment_id
from raq_kv.faults.injector import injected_burst, injected_snapshot
from raq_kv.faults.plan import (
    describe_site,
    generate_bursts_in_tokens,
    generate_plan,
    generate_plan_in_tokens,
    save_plan,
)
from raq_kv.faults.storage import (
    flips_for_rate,
    generate_storage_plan,
    plan_composition,
    storage_regions,
    total_stored_bits,
)
from raq_kv.importance.scorers import ScoreContext, build_scorer
from raq_kv.kv.snapshot import KVSnapshot
from raq_kv.models.loader import LoadedModel, load_model, load_tokenizer
from raq_kv.quantization.apply import QuantReport, quantize_region, quantize_snapshot, quantize_to_codes
from raq_kv.utils.paths import resolve_output_root
from raq_kv.utils.seed import derive_seed


class Task(Protocol):
    def unit_keys(self) -> list[str]: ...
    def data_fingerprint(self) -> dict[str, Any]: ...
    def setup(self) -> dict[str, Any]: ...
    def run_unit(self, unit_key: str) -> dict[str, Any]: ...


def transform_snapshot(snapshot: KVSnapshot, cfg: ExperimentConfig) -> tuple[KVSnapshot, QuantReport | None]:
    """The single place where the cache is modified between the two phases (M1: quantization; M2+: faults, ECC)."""
    if cfg.quantization is None:
        return snapshot, None
    return quantize_snapshot(snapshot, cfg.quantization)


def kv_record_fields(
    lm: LoadedModel, snapshot: KVSnapshot, original_tokens: int, report: QuantReport | None = None,
    ecc_bits: int = 0,
) -> dict[str, Any]:
    """Token/retention and KV memory fields shared by all cache-reading tasks.

    Memory comes from the bit budget: with fake quantization the stored tensors stay bf16, so
    ``kv_bytes_actual`` (what the simulation holds) and ``kv_bytes_analytic`` (what a real system would hold)
    diverge on purpose - see decisions D6.
    """
    budget = report.budget if report is not None else full_precision_kv_budget(lm.geometry, snapshot.seq_len)
    if ecc_bits:
        # Check bits are part of what the system must store, so every downstream memory number and every
        # iso-budget comparison sees them (plan §21, decisions D6).
        budget = replace(budget, ecc_bits=ecc_bits)
    fields = {
        "tokens_original": original_tokens,
        "tokens_retained": snapshot.seq_len,
        "tokens_pruned": original_tokens - snapshot.seq_len,
        "compression_ratio": snapshot.seq_len / original_tokens,
        "kv_bytes_actual": snapshot.nbytes,
        "kv_bytes_analytic": budget.total_bytes,
        "kv_budget": budget.to_dict(),
        "avg_kv_bits": budget.total_bits / max(1, snapshot.seq_len * lm.geometry.kv_values_per_token),
    }
    if report is not None:
        fields["quantization"] = report.stats
    return fields


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

        snapshot, report = transform_snapshot(snapshot, self.cfg)
        result = score_through_cache(
            model,
            snapshot,
            chunk.read_input_ids.to(dev).unsqueeze(0),
            chunk.read_target_ids.to(dev).unsqueeze(0),
        )
        mem2 = cuda_memory(dev)

        record: dict[str, Any] = {
            "sample_id": unit_key,
            **kv_record_fields(self.lm, snapshot, chunk.prefix_len, report),
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
        elif isinstance(cfg.dataset, BFCLConfig):
            samples = build_bfcl(cfg.dataset, self.tokenizer, template_kwargs)
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

        snapshot, report = transform_snapshot(snapshot, self.cfg)
        gen = greedy_decode(model, query_ids, s.max_new_tokens, snapshot=snapshot,
                            eos_token_ids=set(self.lm.eos_token_ids))
        prediction = tok.decode(gen.tokens, skip_special_tokens=True)
        gen_margins = self._generation_margins(tok, gen, prediction, s.reference.get("gold"))
        record: dict[str, Any] = {
            "sample_id": unit_key,
            "meta": s.meta,
            **kv_record_fields(self.lm, snapshot, s.cache_len, report),
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


class QuantSensitivity:
    """Per-region quantization sensitivity on WikiText-2 chunks.

    One unit = one chunk. The chunk is prefilled once; then the golden continuation is scored, and for every region
    (layer × {keys, values}, optionally × KV head) the same continuation is scored with only that region quantized.
    Pairing is exact by construction: identical prefill, identical target tokens, same process and kernels.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        assert isinstance(cfg.task, QuantSensitivityTask)
        self.cfg = cfg
        self.task: QuantSensitivityTask = cfg.task
        self.tokenizer = load_tokenizer(cfg.model)
        chunks = make_chunks(tokenize_wikitext2(cfg.dataset, self.tokenizer), cfg.dataset)
        limit = self.task.n_chunks or len(chunks)
        self.chunks = {c.sample_id: c for c in chunks[:limit]}
        self._fingerprint = chunks_fingerprint(list(self.chunks.values()))
        self.lm: LoadedModel | None = None

    def unit_keys(self) -> list[str]:
        return list(self.chunks)

    def data_fingerprint(self) -> dict[str, Any]:
        return {"wikitext2_chunks_sha256": self._fingerprint}

    def setup(self) -> dict[str, Any]:
        self.lm = load_model(self.cfg.model)
        return {"model_memory": self.lm.memory, "kv_geometry": self.lm.geometry.to_dict(),
                "n_regions": len(self._regions())}

    def _regions(self) -> list[tuple[int, str, int | None]]:
        """(layer, part, head) triples; ``head=None`` means the whole part of that layer."""
        assert self.lm is not None, "setup() must be called first"
        layers = range(self.lm.geometry.num_layers)
        parts = ("keys", "values")
        if self.task.granularity == "layer_part":
            return [(layer, part, None) for layer in layers for part in parts]
        return [(layer, part, head) for layer in layers for part in parts
                for head in range(self.lm.geometry.num_kv_heads)]

    def run_unit(self, unit_key: str) -> dict[str, Any]:
        assert self.lm is not None, "setup() must be called first"
        chunk = self.chunks[unit_key]
        model, dev = self.lm.model, self.lm.device
        inputs = chunk.read_input_ids.to(dev).unsqueeze(0)
        targets = chunk.read_target_ids.to(dev).unsqueeze(0)

        reset_peak(dev)
        snapshot = prefill(model, chunk.context_ids.to(dev).unsqueeze(0))
        golden = score_through_cache(model, snapshot, inputs, targets).token_nll.double().cpu()

        regions = self._regions()
        conditions: list[tuple[int, str, int | None, QuantSpec, bool]] = [
            (layer, part, head, self.task.key_spec if part == "keys" else self.task.value_spec, False)
            for layer, part, head in regions
        ]
        if self.task.null_bits and self.task.null_regions:
            rng = random.Random(derive_seed(self.cfg.seed, unit_key, "null"))
            for layer, part, head in rng.sample(regions, k=min(self.task.null_regions, len(regions))):
                base = self.task.key_spec if part == "keys" else self.task.value_spec
                conditions.append((layer, part, head, base.model_copy(update={"bits": self.task.null_bits}), True))

        rows = []
        for layer, part, head, spec, is_null in conditions:
            perturbed, stats = quantize_region(snapshot, layer, part, spec, head)
            nll = score_through_cache(model, perturbed, inputs, targets).token_nll.double().cpu()
            delta = nll - golden
            rows.append({
                "layer": layer, "part": part, "head": head, "is_null": is_null, "bits": spec.bits,
                "delta_nll_mean": float(delta.mean()), "delta_nll_max": float(delta.max()),
                "delta_nll_frac_worse": float((delta > 0).double().mean()),
                "sqnr_db": stats["sqnr_db"], "mse": stats["mse"],
            })
            del perturbed, nll

        return {
            "sample_id": unit_key,
            **kv_record_fields(self.lm, snapshot, chunk.prefix_len),
            "golden_nll_mean": float(golden.mean()),
            "n_scored_tokens": int(golden.numel()),
            "key_spec": self.task.key_spec.model_dump(),
            "value_spec": self.task.value_spec.model_dump(),
            "granularity": self.task.granularity,
            "regions": rows,
        }


class FaultInjection:
    """Single-bit fault campaign on the quantized KV cache.

    One unit = one chunk. The chunk is prefilled and quantized once; the fault-free quantized read is the golden
    reference for that chunk, and every site in the chunk's plan is injected, scored and restored. Pairing is
    exact (same prefill, same codes, same scored tokens), and the plan is reproducible from the seed so M3 can
    replay identical faults under each protection scheme.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        assert isinstance(cfg.task, FaultInjectionTask) and cfg.quantization is not None
        self.cfg = cfg
        self.task: FaultInjectionTask = cfg.task
        self.tokenizer = load_tokenizer(cfg.model)
        chunks = make_chunks(tokenize_wikitext2(cfg.dataset, self.tokenizer), cfg.dataset)
        self.chunks = {c.sample_id: c for c in chunks[: self.task.n_chunks or len(chunks)]}
        self._fingerprint = chunks_fingerprint(list(self.chunks.values()))
        self.lm: LoadedModel | None = None

    def unit_keys(self) -> list[str]:
        return list(self.chunks)

    def data_fingerprint(self) -> dict[str, Any]:
        return {"wikitext2_chunks_sha256": self._fingerprint}

    def setup(self) -> dict[str, Any]:
        self.lm = load_model(self.cfg.model)
        return {"model_memory": self.lm.memory, "kv_geometry": self.lm.geometry.to_dict()}

    def run_unit(self, unit_key: str) -> dict[str, Any]:
        assert self.lm is not None and self.cfg.quantization is not None
        chunk = self.chunks[unit_key]
        model, dev = self.lm.model, self.lm.device
        inputs = chunk.read_input_ids.to(dev).unsqueeze(0)
        targets = chunk.read_target_ids.to(dev).unsqueeze(0)

        reset_peak(dev)
        snapshot = prefill(model, chunk.context_ids.to(dev).unsqueeze(0))
        qkv = quantize_to_codes(snapshot, self.cfg.quantization)
        base = qkv.to_snapshot()
        golden = score_through_cache(model, base, inputs, targets)
        golden_nll = golden.token_nll.double().cpu()
        golden_argmax = golden.token_argmax.cpu()

        plan = generate_plan(
            qkv,
            seed=derive_seed(self.cfg.seed, unit_key, "fault_plan"),
            n_uniform=self.task.n_uniform, n_msb=self.task.n_msb,
            n_metadata=self.task.n_metadata, n_lsb=self.task.n_lsb,
        )
        plan_path = (resolve_output_root(self.cfg.output.root) / "fault_plans"
                     / experiment_id(self.cfg) / f"{unit_key}.jsonl")
        if not plan_path.exists():
            save_plan(plan_path, plan, {"exp_id": experiment_id(self.cfg), "unit": unit_key,
                                        "seed": self.cfg.seed, "n_sites": len(plan)})

        def outcome(result: Any) -> dict[str, Any]:
            delta = result.token_nll.double().cpu() - golden_nll
            changed = result.token_argmax.cpu() != golden_argmax
            return {
                # A single stored bit perturbs a few positions strongly rather than all of them slightly, so the
                # maximum per-token change carries the signal and the mean mostly dilutes it (D17).
                "delta_nll_max": float(delta.max()), "delta_nll_abs_max": float(delta.abs().max()),
                "delta_nll_mean": float(delta.mean()),
                "tokens_changed": int(changed.sum()), "token_change_rate": float(changed.double().mean()),
                "any_token_changed": bool(changed.any()),
            }

        rows = []
        for _ in range(self.task.n_control):
            # No fault at all: proves the measurement floor is exactly zero before any effect size is quoted.
            rows.append({"stratum": "no_fault", "target": "none", "layer": -1, "part": "none", "bit": -1,
                         **outcome(score_through_cache(model, base, inputs, targets))})
        for site in plan:
            with injected_snapshot(qkv, site, base) as (faulty, info):
                result = score_through_cache(model, faulty, inputs, targets)
            rows.append({**describe_site(site, qkv), **info, **outcome(result)})

        return {
            "sample_id": unit_key,
            **kv_record_fields(self.lm, base, chunk.prefix_len, qkv.report),
            "golden_nll_mean": float(golden_nll.mean()),
            "n_scored_tokens": int(golden_nll.numel()),
            "n_sites": len(plan),
            "faults": rows,
        }


class FaultQA:
    """Role-targeted faults on the probe: the Q49 vehicle.

    For each sample the cached document is quantized once, the fault-free answer is generated as that sample's
    golden, and then faults are aimed at the tokens holding the answer, at same-type distractor tokens, at filler
    tokens and at quantization metadata. Outcomes are task-level: did the answer stop being correct?
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        assert isinstance(cfg.task, FaultQATask) and cfg.quantization is not None
        self.cfg = cfg
        self.task: FaultQATask = cfg.task
        self.tokenizer = load_tokenizer(cfg.model)
        samples = build_raq_probe(cfg.dataset, self.tokenizer, dict(cfg.model.chat_template_kwargs))
        if self.task.n_samples is not None and self.task.n_samples < len(samples):
            # A fixed stride hit only one depth (the generator emits a group's depths consecutively), so the
            # subset is drawn at random from a fixed seed instead.
            picked = random.Random(derive_seed(cfg.seed, "probe_subset")).sample(samples, self.task.n_samples)
            samples = sorted(picked, key=lambda s: s.sample_id)
        self.samples = {s.sample_id: s for s in samples}
        self._fingerprint = samples_fingerprint(samples)
        self.lm: LoadedModel | None = None

    def unit_keys(self) -> list[str]:
        return list(self.samples)

    def data_fingerprint(self) -> dict[str, Any]:
        return {"raq_probe_samples_sha256": self._fingerprint, "n_samples": len(self.samples)}

    def setup(self) -> dict[str, Any]:
        self.lm = load_model(self.cfg.model)
        return {"model_memory": self.lm.memory, "kv_geometry": self.lm.geometry.to_dict()}

    @staticmethod
    def _role_tokens(sample: PromptSample, cache_len: int) -> dict[str, list[int]]:
        """Cached token indices per role: the answer's own tokens, distractor values, and everything else."""
        target, distractor = [], []
        for span in sample.spans:
            if span.label != "fact_value":
                continue
            bucket = target if span.role == "target" else distractor
            bucket.extend(range(span.token_start, min(span.token_end, cache_len)))
        special = set(target) | set(distractor)
        filler = [t for t in range(cache_len) if t not in special]
        return {"target_value": target, "distractor_value": distractor, "filler": filler}

    def run_unit(self, unit_key: str) -> dict[str, Any]:
        assert self.lm is not None and self.cfg.quantization is not None
        sample = self.samples[unit_key]
        model, dev, tok = self.lm.model, self.lm.device, self.lm.tokenizer
        ids = sample.prompt_ids.to(dev).unsqueeze(0)
        query_ids = ids[:, sample.cache_len :]
        eos = set(self.lm.eos_token_ids)

        reset_peak(dev)
        snapshot = prefill(model, ids[:, : sample.cache_len])
        qkv = quantize_to_codes(snapshot, self.cfg.quantization)
        guard = build_protection(qkv, self.cfg.protection)
        base = qkv.to_snapshot() if guard is None else guard.read_or_raise()
        golden_gen = greedy_decode(model, query_ids, sample.max_new_tokens, snapshot=base, eos_token_ids=eos)
        golden_text = tok.decode(golden_gen.tokens, skip_special_tokens=True)
        golden_score = SCORERS[sample.scorer](golden_text, sample.reference)

        roles = self._role_tokens(sample, sample.cache_len)
        seed = derive_seed(self.cfg.seed, unit_key, "fault_qa")
        burst_bits = self.task.burst_bits or 1
        bursts: list[list[Any]] = []
        for role, count in (("target_value", self.task.n_target), ("distractor_value", self.task.n_distractor),
                            ("filler", self.task.n_filler)):
            if burst_bits > 1:
                bursts += generate_bursts_in_tokens(qkv, derive_seed(seed, role), roles[role], role, count,
                                                    burst_bits, self.task.burst_scope or "head")
            else:
                bursts += [[s] for s in generate_plan_in_tokens(qkv, derive_seed(seed, role), roles[role], role,
                                                                count)]
        bursts += [[s] for s in generate_plan(qkv, seed=derive_seed(seed, "meta"), n_uniform=0,
                                              n_metadata=self.task.n_metadata)]

        rows = []
        for _ in range(self.task.n_control):
            text = tok.decode(greedy_decode(model, query_ids, sample.max_new_tokens, snapshot=base,
                                            eos_token_ids=eos).tokens, skip_special_tokens=True)
            rows.append({"stratum": "no_fault", "target": "none", "layer": -1, "part": "none", "bit": -1,
                         "prediction_changed": text != golden_text,
                         **{f"faulty_{k}": v for k, v in SCORERS[sample.scorer](text, sample.reference).items()}})
        for burst in bursts:
            # With protection configured the read goes through the decoder, which is the point of the arm.
            reader = injected_burst(qkv, burst, base) if guard is None else guard.faulted_read(burst, base)
            with reader as (faulty, info):
                gen = greedy_decode(model, query_ids, sample.max_new_tokens, snapshot=faulty, eos_token_ids=eos)
            text = tok.decode(gen.tokens, skip_special_tokens=True)
            score = SCORERS[sample.scorer](text, sample.reference)
            rows.append({
                **describe_site(burst[0], qkv), **info,
                "prediction_changed": text != golden_text,
                "prediction": text if text != golden_text else None,
                "gen_margin_min_all": min(gen.margins) if gen.margins else None,
                **{f"faulty_{k}": v for k, v in score.items()},
            })

        return {
            "sample_id": unit_key,
            "meta": sample.meta,
            **kv_record_fields(self.lm, base, sample.cache_len, qkv.report,
                               ecc_bits=0 if guard is None else guard.ecc_bits),
            "golden_correct": bool(golden_score["correct"]),
            "golden_prediction": golden_text,
            "role_token_counts": {role: len(tokens) for role, tokens in roles.items()},
            "burst_bits": burst_bits,
            "n_sites": len(bursts),
            "faults": rows,
        }


def schema_spans(sample: PromptSample, cache_len: int) -> list[tuple[int, ...]]:
    """The catalogue's own tool schemas as retention blocks, covering [0, cache_len) without gaps.

    `select_keep_spans` needs a partition it can spend the whole budget on, so the template text between and
    around schemas is emitted as its own blocks rather than dropped -- otherwise the tokens the chat template
    contributes would be unprunable and the realised ratio would drift from the requested one.
    """
    bounds = sorted({0, cache_len} | {b for span in sample.spans if span.label == "tool_schema"
                                      for b in (span.token_start, min(span.token_end, cache_len))
                                      if 0 <= b <= cache_len})
    return [(lo, hi) for lo, hi in zip(bounds, bounds[1:], strict=False) if hi > lo]


def build_probe_samples(cfg: ExperimentConfig, tokenizer: Any) -> list[PromptSample]:
    """The sample builder for tasks that read cached facts, dispatched on the configured dataset.

    Added when M6 gave the retention task a second vehicle (`bfcl`). Before that every such task called
    `build_raq_probe` directly, which is why a second dataset could not be used without touching each one.
    """
    kwargs = dict(cfg.model.chat_template_kwargs)
    if isinstance(cfg.dataset, BFCLConfig):
        return build_bfcl(cfg.dataset, tokenizer, kwargs)
    return build_raq_probe(cfg.dataset, tokenizer, kwargs)


def role_tokens(sample: PromptSample, cache_len: int) -> dict[str, tuple[int, ...]]:
    """The vehicle's own span labels, as cache positions: which tokens carry the answer, a distractor, a span.

    Two vehicles, two labellings, one meaning. RAQ-Probe marks a fact's value tokens (`fact_value`) and the
    sentence around it (`fact_sentence`). BFCL's catalogue marks whole tool schemas (`tool_schema`), where the
    gold schema is the target and the other 10-50 are distractors -- and there the schema *is* the span, so it
    fills both roles. That is what makes `kept_fraction_target` mean the same thing in M6 as in M4: the share of
    the tokens the answer actually needs that survived pruning.
    """
    target, distractor, sentence = [], [], []
    for span in sample.spans:
        span_tokens = range(span.token_start, min(span.token_end, cache_len))
        if span.label == "fact_value":
            (target if span.role == "target" else distractor).extend(span_tokens)
        elif span.label == "fact_sentence" and span.role == "target":
            sentence.extend(span_tokens)
        elif span.label == "tool_schema":
            if span.role == "target":
                target.extend(span_tokens)
                sentence.extend(span_tokens)
            else:
                distractor.extend(span_tokens)
    return {"target": tuple(target), "distractor": tuple(distractor), "sentence": tuple(sentence)}


def control_span(seed: int, sample: PromptSample, cache_len: int, sentence: tuple[int, ...]) -> tuple[int, ...]:
    """A contiguous window the size of the fact's sentence, placed somewhere else in the document.

    Deterministic per sample, and deliberately kept clear of the sentence itself so the control cannot
    accidentally include the context it is meant to exclude.
    """
    width = len(sentence)
    if width == 0 or cache_len <= width:
        return ()
    rng = random.Random(derive_seed(seed, sample.sample_id, "control_span"))
    lo, hi = (min(sentence), max(sentence)) if sentence else (0, 0)
    for _ in range(64):
        start = rng.randrange(0, cache_len - width)
        if start + width <= lo or start > hi:
            return tuple(range(start, start + width))
    return ()


def build_retention_plan(
    cfg: ExperimentConfig, scorer: Any, tok: Any, model: Any, snapshot: KVSnapshot, context_ids: Any,
    query_ids: Any, sample: PromptSample,
) -> tuple[Any, list[Any], dict[str, tuple[int, ...]], tuple[int, ...]]:
    """Score the cached tokens and decide which to keep. One implementation, two tasks.

    Extracted from ``Retention`` when ``ProtectionArm`` gained a pruning step (D43), because the alternative was
    two copies of the span/token branch drifting apart -- and a difference between how the fault experiment
    prunes and how the retention experiment prunes would make the two milestones incomparable in exactly the
    place they are meant to be compared.
    """
    c = cfg.compression
    assert c is not None
    roles = role_tokens(sample, sample.cache_len)
    control = control_span(cfg.seed, sample, sample.cache_len, roles["sentence"])
    scores = scorer.score(ScoreContext(
        snapshot=snapshot, context_ids=context_ids, model=model, query_ids=query_ids,
        target_tokens=roles["target"], distractor_tokens=roles["distractor"],
        sentence_tokens=roles["sentence"], control_span_tokens=control))
    if c.unit in ("span", "schema"):
        # "span" infers blocks from punctuation; "schema" takes them from the sample's own labelled tool schemas,
        # which is the only sensible block structure inside a JSON tool catalogue (D50). Both then go through the
        # same selector, so the two vehicles' span results are produced by one code path.
        spans = (schema_spans(sample, sample.cache_len) if c.unit == "schema"
                 else sentence_spans(tok, context_ids))
        plan = select_keep_spans(scores, spans, c.ratio, c.protect_recent, c.protect_initial,
                                 c.span_aggregate or "mean")
    else:
        spans = []
        plan = select_keep(scores, c.ratio, c.protect_recent, c.protect_initial)
    return plan, spans, roles, control


class ProtectionArm:
    """One M3 arm at one bit error rate: the H2 vehicle.

    Per sample the cache is quantized once, the arm's protection is built over it, the fault-free answer is
    generated through the arm's own read path (so the protected and unprotected arms differ only in faults), and
    then ``n_trials`` independent storage-wide fault draws are applied and read back through the codec.

    Each trial is classified on two independent axes, because collapsing them is how reliability results get
    overstated: **did the answer change** (the only thing the task cares about) and **did the codec notice**
    (corrected words, or detected-uncorrectable ones). A changed answer with nothing detected is silent data
    corruption; a changed answer with a DUE is a failure the system could have caught and recomputed.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        assert isinstance(cfg.task, ProtectionArmTask) and cfg.quantization is not None
        self.cfg = cfg
        self.task: ProtectionArmTask = cfg.task
        self.tokenizer = load_tokenizer(cfg.model)
        samples = build_raq_probe(cfg.dataset, self.tokenizer, dict(cfg.model.chat_template_kwargs))
        if self.task.n_samples is not None and self.task.n_samples < len(samples):
            picked = random.Random(derive_seed(cfg.seed, "probe_subset")).sample(samples, self.task.n_samples)
            samples = sorted(picked, key=lambda s: s.sample_id)
        self.samples = {s.sample_id: s for s in samples}
        self._fingerprint = samples_fingerprint(samples)
        self.lm: LoadedModel | None = None
        # Only the fault x retention arms carry a scorer; a plain M3 arm prunes nothing and builds none.
        self.scorer = (build_scorer(cfg.compression.scorer, cfg.compression.window)
                       if cfg.compression is not None else None)

    def unit_keys(self) -> list[str]:
        return list(self.samples)

    def data_fingerprint(self) -> dict[str, Any]:
        return {"raq_probe_samples_sha256": self._fingerprint, "n_samples": len(self.samples)}

    def setup(self) -> dict[str, Any]:
        self.lm = load_model(self.cfg.model)
        if self.scorer is not None and self.scorer.needs_model and self.cfg.model.attn_implementation != "eager":
            raise ValueError(f"scorer '{self.scorer.name}' needs attention weights, which require eager "
                             f"attention (D5); this config asks for '{self.cfg.model.attn_implementation}'")
        info: dict[str, Any] = {"model_memory": self.lm.memory, "kv_geometry": self.lm.geometry.to_dict()}
        if self.scorer is not None and self.cfg.compression is not None:
            info |= {"scorer": self.scorer.name, "scorer_is_oracle": self.scorer.is_oracle,
                     "unit": self.cfg.compression.unit or "token"}
        return info

    def run_unit(self, unit_key: str) -> dict[str, Any]:
        assert self.lm is not None and self.cfg.quantization is not None
        sample = self.samples[unit_key]
        model, dev, tok = self.lm.model, self.lm.device, self.lm.tokenizer
        ids = sample.prompt_ids.to(dev).unsqueeze(0)
        query_ids = ids[:, sample.cache_len :]
        eos = set(self.lm.eos_token_ids)

        reset_peak(dev)
        context_ids = ids[:, : sample.cache_len]
        snapshot = prefill(model, context_ids)

        # Optional pruning, for the arms that vary retention and faults together (D43). Prune BEFORE quantizing,
        # the same order `Retention` uses and the order a system that compresses before storing would use:
        # quantizing first would build groups over tokens about to be discarded, and `group_token` groups span
        # tokens, so the two orders do not produce the same metadata or the same bit budget.
        # Named `retention_plan`, not `plan`: the fault-trial loop below binds `plan` to a storage fault plan,
        # and the first version of this method let that shadow the retention one. The smoke run caught it
        # immediately, but only because it ran -- two objects called `plan` in one method is not a naming
        # preference, it is a live bug waiting for the branch that reads the wrong one.
        retention_plan = roles = control = None
        spans: list[Any] = []
        unpruned_correct = None
        unpruned_snapshot = snapshot
        if self.cfg.compression is not None:
            # Whether the model could answer from the FULL cache, recorded separately. Without it a sample that
            # pruning alone breaks is indistinguishable from one the faults break, and the arm's clean failures
            # would be charged to the fault model.
            unpruned_text = tok.decode(greedy_decode(model, query_ids, sample.max_new_tokens, snapshot=snapshot,
                                                     eos_token_ids=eos).tokens, skip_special_tokens=True)
            unpruned_correct = bool(SCORERS[sample.scorer](unpruned_text, sample.reference)["correct"])
            retention_plan, spans, roles, control = build_retention_plan(
                self.cfg, self.scorer, tok, model, snapshot, context_ids, query_ids, sample)
            snapshot = apply_retention(snapshot, retention_plan)

        qkv = quantize_to_codes(snapshot, self.cfg.quantization)
        guard = build_protection(qkv, self.cfg.protection)
        base = qkv.to_snapshot() if guard is None else guard.read_or_raise()
        # The golden runs through THIS arm's cache -- pruned, quantized, protected, unfaulted. It is the arm's
        # own no-fault reference, so `task_failure` isolates fault damage; a golden taken from the full cache
        # would fold pruning damage into every trial and nothing would be attributable.
        golden_gen = greedy_decode(model, query_ids, sample.max_new_tokens, snapshot=base, eos_token_ids=eos)
        golden_text = tok.decode(golden_gen.tokens, skip_special_tokens=True)
        golden_score = SCORERS[sample.scorer](golden_text, sample.reference)

        regions = storage_regions(qkv, guard)
        stored_bits = total_stored_bits(regions)
        # Exactly one of `n_flips` and `ber` is set (the task schema enforces it).
        #
        # `ber_basis: unpruned` is the attribution arm (D43). A pruned cache stores about a quarter of the bits,
        # so at a matched *rate* it takes about a quarter of the flips -- and then "pruning improved
        # reliability" is arithmetic, not a finding about importance. This basis applies the rate to what the
        # arm WOULD store unpruned and injects that many flips into the pruned cache, so exposure is held equal
        # and only the concentration of risk varies. The count is derived per sample, because a fixed number
        # would give a 512-token and a 4096-token sample the same exposure when their storage differs eightfold.
        if self.task.n_flips_per_token is not None:
            # Per ORIGINAL token, not per retained one: the point is an exposure two arms of different precision
            # (or different retention) can be held to, and only the original length is common to both.
            n_flips = max(1, round(self.task.n_flips_per_token * sample.cache_len))
        elif self.task.n_flips is not None:
            n_flips = self.task.n_flips
        elif self.task.ber_basis == "unpruned" and self.cfg.compression is not None:
            # Quantized and protected for real rather than predicted, so this cannot drift from the encoders the
            # way a closed-form predictor could; the throwaway cache is dropped immediately.
            full_qkv = quantize_to_codes(unpruned_snapshot, self.cfg.quantization)
            full_guard = build_protection(full_qkv, self.cfg.protection)
            n_flips = flips_for_rate(full_qkv, full_guard, self.task.ber)
            del full_qkv, full_guard
        else:
            n_flips = flips_for_rate(qkv, guard, self.task.ber)
        seed = derive_seed(self.cfg.seed, unit_key, "protection_arm")

        rows = []
        for i in range(self.task.n_control):
            text = tok.decode(greedy_decode(model, query_ids, sample.max_new_tokens, snapshot=base,
                                            eos_token_ids=eos).tokens, skip_special_tokens=True)
            rows.append({"trial": -1 - i, "n_flips": 0, "answer_changed": text != golden_text,
                         "corrected_words": 0, "due_words": 0, "detected": False,
                         **{f"faulty_{k}": v for k, v in SCORERS[sample.scorer](text, sample.reference).items()}})
        for trial in range(self.task.n_trials):
            plan = generate_storage_plan(qkv, guard, derive_seed(seed, f"trial{trial}"), n_flips)
            reader = (injected_burst(qkv, plan, base) if guard is None
                      else guard.faulted_read(plan, base))
            with reader as (faulty, info):
                gen = greedy_decode(model, query_ids, sample.max_new_tokens, snapshot=faulty, eos_token_ids=eos)
            text = tok.decode(gen.tokens, skip_special_tokens=True)
            score = SCORERS[sample.scorer](text, sample.reference)
            changed = text != golden_text
            # An unprotected arm has no decoder, so these are zero by construction rather than missing: a blank
            # here would read as "not measured" in the report instead of "there was nothing to detect with".
            due = int(info.get("due_words", 0))
            corrected = int(info.get("corrected_words", 0))
            rows.append({
                "trial": trial, **info, **plan_composition(plan),
                "corrected_words": corrected, "due_words": due, "words_read": int(info.get("words_read", 0)),
                "detected": bool(due or corrected),
                "answer_changed": changed,
                "silent_corruption": bool(changed and due == 0),
                "detected_failure": bool(changed and due > 0),
                "prediction": text if changed else None,
                "gen_margin_min_all": min(gen.margins) if gen.margins else None,
                **{f"faulty_{k}": v for k, v in score.items()},
            })

        return {
            "sample_id": unit_key,
            "meta": sample.meta,
            **kv_record_fields(self.lm, base, sample.cache_len, qkv.report,
                               ecc_bits=0 if guard is None else guard.ecc_bits),
            "golden_correct": bool(golden_score["correct"]),
            "golden_prediction": golden_text,
            "ber": self.task.ber,
            "exposure_mode": ("per_token" if self.task.n_flips_per_token is not None
                              else "count" if self.task.n_flips is not None
                              else f"rate:{self.task.ber_basis or 'stored'}"),
            "stored_bits": stored_bits,
            "n_flips_per_trial": n_flips,
            "storage_composition": _storage_shares(regions),
            # Present only on the fault x retention arms (D43). `unpruned_correct` separates a sample that
            # pruning alone broke from one the faults broke; `kept_fraction_target` is the D32 diagnostic that
            # says whether the arm kept the tokens the answer needs at all.
            **({} if retention_plan is None else {
                **retention_plan.to_dict(),
                "scorer": self.scorer.name if self.scorer else None,
                "unit": (self.cfg.compression.unit or "token") if self.cfg.compression else None,
                "n_spans": len(spans),
                "unpruned_correct": unpruned_correct,
                "kept_fraction_target": retention_plan.kept_fraction_of(roles["target"]),
                "kept_fraction_sentence": retention_plan.kept_fraction_of(roles["sentence"]),
                "kept_fraction_control_span": retention_plan.kept_fraction_of(control),
            }),
            "faults": rows,
        }


def _storage_shares(regions: list[Any]) -> dict[str, int]:
    """Bits per target family, so a reader can see what the rate was actually applied to."""
    shares: dict[str, int] = {}
    for region in regions:
        shares[region.target] = shares.get(region.target, 0) + region.n_bits
    return shares


class Retention:
    """Token retention on the probe: how much of the cache can go, and does the choice of *which* tokens matter?

    Per sample the context is prefilled once. The golden answer is generated from the unpruned cache in the same
    run — the eager attention that ``window_attention`` needs changes the experiment identity, so no earlier
    golden applies — then the scorer ranks the cached tokens, the plan keeps its share, and the answer is
    regenerated from the pruned cache.

    The diagnostic that makes the table readable is ``kept_fraction_target``: a scorer that holds accuracy while
    discarding the answer-bearing tokens has not found importance, it has found that the model does not need
    them. That is a different and more interesting result, and it is the D19-D21 redundancy question.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        assert isinstance(cfg.task, RetentionTask) and cfg.compression is not None
        self.cfg = cfg
        self.task: RetentionTask = cfg.task
        self.compression = cfg.compression
        self.tokenizer = load_tokenizer(cfg.model)
        samples = build_raq_probe(cfg.dataset, self.tokenizer, dict(cfg.model.chat_template_kwargs))
        if self.task.n_samples is not None and self.task.n_samples < len(samples):
            picked = random.Random(derive_seed(cfg.seed, "probe_subset")).sample(samples, self.task.n_samples)
            samples = sorted(picked, key=lambda s: s.sample_id)
        self.samples = {s.sample_id: s for s in samples}
        self._fingerprint = samples_fingerprint(samples)
        self.scorer = build_scorer(self.compression.scorer, self.compression.window)
        self.lm: LoadedModel | None = None

    def unit_keys(self) -> list[str]:
        return list(self.samples)

    def data_fingerprint(self) -> dict[str, Any]:
        return {"raq_probe_samples_sha256": self._fingerprint, "n_samples": len(self.samples)}

    def setup(self) -> dict[str, Any]:
        self.lm = load_model(self.cfg.model)
        if self.scorer.needs_model and self.cfg.model.attn_implementation != "eager":
            raise ValueError(f"scorer '{self.scorer.name}' needs attention weights, so the model must use eager "
                             f"attention (D5); this config asks for '{self.cfg.model.attn_implementation}'")
        return {"model_memory": self.lm.memory, "kv_geometry": self.lm.geometry.to_dict(),
                "scorer": self.scorer.name, "scorer_is_oracle": self.scorer.is_oracle,
                "unit": self.compression.unit or "token"}

    def run_unit(self, unit_key: str) -> dict[str, Any]:
        assert self.lm is not None and self.cfg.compression is not None
        sample = self.samples[unit_key]
        model, dev, tok = self.lm.model, self.lm.device, self.lm.tokenizer
        ids = sample.prompt_ids.to(dev).unsqueeze(0)
        context_ids, query_ids = ids[:, : sample.cache_len], ids[:, sample.cache_len :]
        eos = set(self.lm.eos_token_ids)

        reset_peak(dev)
        snapshot = prefill(model, context_ids)
        golden_gen = greedy_decode(model, query_ids, sample.max_new_tokens, snapshot=snapshot, eos_token_ids=eos)
        golden_text = tok.decode(golden_gen.tokens, skip_special_tokens=True)
        golden_score = SCORERS[sample.scorer](golden_text, sample.reference)

        c = self.cfg.compression
        plan, spans, roles, control = build_retention_plan(self.cfg, self.scorer, tok, model, snapshot,
                                                           context_ids, query_ids, sample)
        pruned = apply_retention(snapshot, plan)

        # Prune first, then quantize: that is the order a system that compresses before storing would use, and
        # it is the order the budget assumes - quantizing first would build groups over tokens about to be
        # discarded, and group_token groups span tokens, so the two orders do not give the same metadata.
        stored, report = transform_snapshot(pruned, self.cfg)

        gen = greedy_decode(model, query_ids, sample.max_new_tokens, snapshot=stored, eos_token_ids=eos)
        text = tok.decode(gen.tokens, skip_special_tokens=True)
        score = SCORERS[sample.scorer](text, sample.reference)

        return {
            "sample_id": unit_key,
            "meta": sample.meta,
            **kv_record_fields(self.lm, stored, sample.cache_len, report),
            **plan.to_dict(),
            "scorer": self.scorer.name,
            "unit": c.unit or "token",
            "n_spans": len(spans),
            "golden_correct": bool(golden_score["correct"]),
            "golden_prediction": golden_text,
            "prediction": text,
            "answer_changed": text != golden_text,
            "task_failure": bool(golden_score["correct"] and not score["correct"]),
            "kept_fraction_target": plan.kept_fraction_of(roles["target"]),
            "kept_fraction_distractor": plan.kept_fraction_of(roles["distractor"]),
            "kept_fraction_sentence": plan.kept_fraction_of(roles["sentence"]),
            "n_sentence_tokens": len(roles["sentence"]),
            "kept_fraction_control_span": plan.kept_fraction_of(control),
            "n_control_span_tokens": len(control),
            "n_target_tokens": len(roles["target"]),
            "gen_margin_min_all": min(gen.margins) if gen.margins else None,
            **{f"pruned_{k}": v for k, v in score.items()},
        }


def build_task(cfg: ExperimentConfig) -> Task:
    if isinstance(cfg.task, CachePPLTask):
        return CachePPL(cfg)
    if isinstance(cfg.task, LatencyTask):
        return Latency(cfg)
    if isinstance(cfg.task, QATask):
        return QAGenerate(cfg)
    if isinstance(cfg.task, QuantSensitivityTask):
        return QuantSensitivity(cfg)
    if isinstance(cfg.task, FaultInjectionTask):
        return FaultInjection(cfg)
    if isinstance(cfg.task, FaultQATask):
        return FaultQA(cfg)
    if isinstance(cfg.task, ProtectionArmTask):
        return ProtectionArm(cfg)
    if isinstance(cfg.task, RetentionTask):
        return Retention(cfg)
    raise NotImplementedError(f"no task for kind '{cfg.task.kind}'")
