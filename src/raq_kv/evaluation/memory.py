"""Memory measurement: model bytes, actual KV tensor bytes, analytic budget, and CUDA allocator deltas.

The analytic KV number (``budget.bit_accounting``) is what later quantized/protected configurations report,
because fake-quantized tensors do not shrink real memory. M0 validates that formula against the real
allocator at full precision, where both are measurable.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def module_bytes(model: nn.Module) -> dict[str, int]:
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return {"param_bytes": params, "buffer_bytes": buffers, "param_count": sum(p.numel() for p in model.parameters())}


def cuda_memory(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "max_allocated": torch.cuda.max_memory_allocated(device),
    }


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def allocator_delta(before: dict[str, int] | None, after: dict[str, int] | None) -> dict[str, Any]:
    """Live-tensor growth and peak during a measured region (None on CPU)."""
    if before is None or after is None:
        return {"cuda_allocated_delta_bytes": None, "cuda_peak_over_start_bytes": None}
    return {
        "cuda_allocated_delta_bytes": after["allocated"] - before["allocated"],
        "cuda_peak_over_start_bytes": after["max_allocated"] - before["allocated"],
    }
