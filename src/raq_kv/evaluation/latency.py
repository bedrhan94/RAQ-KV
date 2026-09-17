"""Latency of the unmodified serving path (full-precision baseline).

Only wall-clock numbers that correspond to a real execution path are reported here. Quantization/ECC latency
of the Python simulation is NOT a hardware latency and is never reported through this module.

Definitions (greedy, fixed ``gen_len`` new tokens, EOS ignored):
* TTFT: prompt submission -> first new token materialized on host (includes prefill);
* TPOT: mean interval between consecutive new tokens after the first;
* E2E:  prompt submission -> last new token.
"""

from __future__ import annotations

import statistics
import time

import torch
from torch import nn

from raq_kv.evaluation.protocol import greedy_decode


def measure_generation_latency(model: nn.Module, prompt_ids: torch.Tensor, gen_len: int) -> dict[str, float]:
    device = prompt_ids.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stamps: list[float] = []
    start = time.perf_counter()
    result = greedy_decode(
        model, prompt_ids, max_new_tokens=gen_len, on_token=lambda _t: stamps.append(time.perf_counter())
    )
    if len(stamps) != gen_len:
        raise RuntimeError(f"expected {gen_len} tokens, got {len(stamps)}")
    ttft = stamps[0] - start
    intervals = [b - a for a, b in zip(stamps[:-1], stamps[1:], strict=True)]
    tpot_mean = statistics.fmean(intervals)
    return {
        "prompt_tokens": prompt_ids.shape[1],
        "new_tokens": len(result.tokens),
        "ttft_s": ttft,
        "tpot_mean_s": tpot_mean,
        "tpot_median_s": statistics.median(intervals),
        "e2e_s": stamps[-1] - start,
        "prefill_tokens_per_s": prompt_ids.shape[1] / ttft,
        "decode_tokens_per_s": 1.0 / tpot_mean,
    }
