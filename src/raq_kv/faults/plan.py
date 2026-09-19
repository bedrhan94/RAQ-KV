"""Fault plans: the list of sites to corrupt, generated once and replayed.

A plan is generated from a seed and the geometry of a quantized snapshot, written to disk, and replayed by every
protection configuration (M3), so NONE / PARITY / SECDED see *identical* faults rather than statistically similar
ones. Each site addresses one bit of one stored element:

* ``code``  — (layer, part, head, token, channel, bit) of an integer code;
* ``scale`` / ``zero`` — (layer, part, flat group index, bit) of a float16 quantization parameter.

Sampling is stratified so that rare-but-important classes (MSBs, metadata) are represented without making the
plan unbalanced by accident.

There is deliberately **no "inert bit" stratum**: measurement showed that at 4 bits the least significant bit of
a code still moves the stored value by one quantization step, which is about half the tensor's mean magnitude
(D17). The campaign's floor is instead established by no-fault control scorings, which are exactly zero because
the pipeline is deterministic. The ``lsb_midrange`` stratum therefore measures *the smallest possible* storage
error, not a harmless one.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from raq_kv.faults.bit_utils import bit_significance, float16_field
from raq_kv.quantization.apply import QuantizedKV
from raq_kv.quantization.base import QuantizedTensor
from raq_kv.utils.io import JsonlAppender, read_jsonl

Target = Literal["code", "scale", "zero"]


@dataclass(frozen=True)
class FaultSite:
    layer: int
    part: str  # keys | values
    target: Target
    index: tuple[int, ...]  # index into the codes tensor, or into the scale/zero tensor
    bit: int
    stratum: str  # why this site was drawn: uniform | msb | metadata | lsb_midrange

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "index": list(self.index)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaultSite:
        return cls(layer=data["layer"], part=data["part"], target=data["target"],
                   index=tuple(data["index"]), bit=data["bit"], stratum=data["stratum"])


def _quantized_parts(qkv: QuantizedKV) -> list[tuple[int, str, QuantizedTensor]]:
    parts = []
    for layer in range(qkv.num_layers):
        for part in ("keys", "values"):
            tensor = qkv.part(layer, part)
            if isinstance(tensor, QuantizedTensor):
                parts.append((layer, part, tensor))
    return parts


def _random_index(rng: random.Random, shape: torch.Size) -> tuple[int, ...]:
    return tuple(rng.randrange(int(dim)) for dim in shape)


def _small_code_index(rng: random.Random, qt: QuantizedTensor, tries: int = 64) -> tuple[int, ...] | None:
    """An index whose code is far from the extremes, so flipping its LSB changes the value by one step only."""
    mid = (1 << qt.bits) - 1
    best = None
    for _ in range(tries):
        idx = _random_index(rng, qt.codes.shape)
        code = int(qt.codes[idx])
        distance = min(code, mid - code)
        if best is None or distance > best[0]:
            best = (distance, idx)
    return None if best is None else best[1]


def generate_plan(qkv: QuantizedKV, seed: int, n_uniform: int, n_msb: int = 0, n_metadata: int = 0,
                  n_lsb: int = 0) -> list[FaultSite]:
    """Draw fault sites. Strata are drawn independently; every site is a single bit of a single stored element."""
    rng = random.Random(seed)
    parts = _quantized_parts(qkv)
    if not parts:
        raise ValueError("no quantized parts to inject into")
    sites: list[FaultSite] = []

    def draw_code(stratum: str, bit_chooser) -> FaultSite:
        layer, part, qt = parts[rng.randrange(len(parts))]
        idx = _random_index(rng, qt.codes.shape)
        return FaultSite(layer, part, "code", idx, bit_chooser(qt), stratum)

    for _ in range(n_uniform):
        sites.append(draw_code("uniform", lambda qt: rng.randrange(qt.bits)))
    for _ in range(n_msb):
        sites.append(draw_code("msb", lambda qt: qt.bits - 1))
    for _ in range(n_metadata):
        layer, part, qt = parts[rng.randrange(len(parts))]
        target: Target = "scale" if qt.zeros is None or rng.random() < 0.5 else "zero"
        tensor = qt.scales if target == "scale" else qt.zeros
        assert tensor is not None
        sites.append(FaultSite(layer, part, target, _random_index(rng, tensor.shape), rng.randrange(16), "metadata"))
    for _ in range(n_lsb):
        layer, part, qt = parts[rng.randrange(len(parts))]
        idx = _small_code_index(rng, qt)
        if idx is not None:
            sites.append(FaultSite(layer, part, "code", idx, 0, "lsb_midrange"))
    return sites


def generate_bursts_in_tokens(
    qkv: QuantizedKV, seed: int, tokens: list[int], stratum: str, n: int, burst_bits: int
) -> list[list[FaultSite]]:
    """``n`` bursts of ``burst_bits`` flips, each burst confined to one (layer, part, token, head).

    A burst models what a memory error actually looks like: the stored bits of one token's head sit together, so
    a corrupted word damages several of them at once. Single-bit faults at a usable operating point turned out to
    be harmless at the task level (D19), which is why the token-role question needs this unit.
    """
    rng = random.Random(seed)
    parts = _quantized_parts(qkv)
    if not parts or not tokens:
        return []
    bursts = []
    for _ in range(n):
        layer, part, qt = parts[rng.randrange(len(parts))]
        batch, heads, n_tokens, channels = qt.codes.shape
        token = tokens[rng.randrange(len(tokens))]
        if token >= n_tokens:
            continue
        b, head = rng.randrange(batch), rng.randrange(heads)
        picks = rng.sample([(c, bit) for c in range(channels) for bit in range(qt.bits)],
                           k=min(burst_bits, channels * qt.bits))
        bursts.append([FaultSite(layer, part, "code", (b, head, token, channel), bit, stratum)
                       for channel, bit in picks])
    return bursts


def generate_plan_in_tokens(qkv: QuantizedKV, seed: int, tokens: list[int], stratum: str, n: int) -> list[FaultSite]:
    """Draw ``n`` code sites whose token index is restricted to ``tokens``.

    This is what makes Q49 answerable: the probe knows which cached tokens hold the target fact, which hold
    same-type distractors and which are filler, so faults can be aimed at a role instead of at random storage.
    """
    rng = random.Random(seed)
    parts = _quantized_parts(qkv)
    if not parts or not tokens:
        return []
    sites = []
    for _ in range(n):
        layer, part, qt = parts[rng.randrange(len(parts))]
        batch, heads, n_tokens, channels = qt.codes.shape
        token = tokens[rng.randrange(len(tokens))]
        if token >= n_tokens:
            continue
        index = (rng.randrange(batch), rng.randrange(heads), token, rng.randrange(channels))
        sites.append(FaultSite(layer, part, "code", index, rng.randrange(qt.bits), stratum))
    return sites


def describe_site(site: FaultSite, qkv: QuantizedKV) -> dict[str, Any]:
    """Human- and analysis-readable labels for one site (bit significance, token/head position, group)."""
    tensor = qkv.part(site.layer, site.part)
    info: dict[str, Any] = {"layer": site.layer, "part": site.part, "target": site.target, "bit": site.bit,
                            "stratum": site.stratum}
    if site.target == "code" and isinstance(tensor, QuantizedTensor):
        _, head, token, channel = site.index
        info |= {"head": head, "token": token, "channel": channel,
                 "significance": bit_significance(site.bit, tensor.bits), "code": int(tensor.codes[site.index])}
    else:
        info |= {"significance": float16_field(site.bit), "group_index": list(site.index)}
    return info


def save_plan(path: Path, sites: list[FaultSite], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with JsonlAppender(path) as writer:
        writer.append({"__plan_meta__": meta, "n_sites": len(sites)})
        for site in sites:
            writer.append(site.to_dict())


def load_plan(path: Path) -> tuple[list[FaultSite], dict[str, Any]]:
    records = read_jsonl(path)
    meta = records[0].get("__plan_meta__", {}) if records else {}
    return [FaultSite.from_dict(r) for r in records if "__plan_meta__" not in r], meta
