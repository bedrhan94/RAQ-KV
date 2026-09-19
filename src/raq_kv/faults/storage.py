"""Fault plans defined over everything an arm actually stores.

Why this module exists: :mod:`raq_kv.faults.plan` draws sites from the *quantized* store -- codes, scales, zero
points. That is the right unit for M2, where nothing else exists. It becomes the wrong unit the moment arms are
compared at equal storage, because a protected arm spends part of its budget on check bits, and a fault model
that cannot reach those bits grants the protected arm an immunity no physical memory grants it. The comparison
would then measure the model's blind spot rather than the policy.

So an M3 fault model is a **bit error rate over the declared representation**: codes + quantization metadata +
ECC check bits, each region drawn in proportion to its real size. A check-bit flip is included and is not a
curiosity -- it makes the decoder compute a syndrome for an error that did not happen, i.e. corrupt intact data.

One consequence, stated here because it contradicts the M2 rule it inherits: **plans are not replayable across
arms.** Two arms with different group sizes or different protection have different layouts, so site (layer,
part, index, bit) does not denote the same physical bit. Replay stays valid between a protected and an
unprotected arm over the *same* quantization -- that is the paired comparison that isolates the codec -- while
arms with different layouts are matched on rate, with enough trials that the rate is what is being compared.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from raq_kv.ecc.protect import ProtectedKV
from raq_kv.faults.plan import FaultSite
from raq_kv.quantization.apply import QuantizedKV
from raq_kv.quantization.base import SCALE_BITS, QuantizedTensor


@dataclass(frozen=True)
class Region:
    """One addressable span of stored bits: what it is, how big it is, and how to turn an offset into a site."""

    layer: int
    part: str
    target: str  # code | scale | zero | check:code | check:scale | check:zero
    n_bits: int
    shape: tuple[int, ...]  # element grid, excluding the bit axis
    bits_per_element: int

    def site(self, offset: int, stratum: str) -> FaultSite:
        element, bit = divmod(offset, self.bits_per_element)
        index = []
        for dim in reversed(self.shape):
            element, rem = divmod(element, dim)
            index.append(rem)
        return FaultSite(self.layer, self.part, self.target, tuple(reversed(index)), bit, stratum)


def storage_regions(qkv: QuantizedKV, guard: ProtectedKV | None = None) -> list[Region]:
    """Inventory of the stored representation, in the same terms the bit budget counts it."""
    regions: list[Region] = []
    for layer in range(qkv.num_layers):
        for part in ("keys", "values"):
            stored = qkv.part(layer, part)
            if not isinstance(stored, QuantizedTensor):
                continue  # an unquantized part has no codes to address; M3 arms quantize both
            regions.append(Region(layer, part, "code", stored.codes.numel() * stored.bits,
                                  tuple(stored.codes.shape), stored.bits))
            for target, tensor in (("scale", stored.scales), ("zero", stored.zeros)):
                if tensor is not None:
                    regions.append(Region(layer, part, target, tensor.numel() * SCALE_BITS,
                                          tuple(tensor.shape), SCALE_BITS))
    if guard is not None:
        for (layer, part, target), code in guard.codes.items():
            check = code.check
            if check.numel() == 0:
                continue
            regions.append(Region(layer, part, f"check:{target}", check.numel(),
                                  (int(check.shape[0]),), int(check.shape[1])))
        for (layer, part, target), bound in guard.bounds.items():
            # Two float16 limits. Tiny, and included for exactly the reason the check bits are: protection that
            # faults cannot reach is protection measured with an advantage no memory grants it.
            regions.append(Region(layer, part, f"bound:{target}", bound.numel() * SCALE_BITS,
                                  (int(bound.numel()),), SCALE_BITS))
    return regions


def total_stored_bits(regions: list[Region]) -> int:
    return sum(r.n_bits for r in regions)


def generate_storage_plan(
    qkv: QuantizedKV,
    guard: ProtectedKV | None,
    seed: int,
    n_flips: int,
    stratum: str = "storage_seu",
) -> list[FaultSite]:
    """``n_flips`` independent single-bit upsets, uniform over every stored bit of this arm.

    Uniform over *bits*, so a region's share of the faults equals its share of the storage -- which is what
    makes "the same bit error rate" mean the same thing in two arms of different shape. Sites are drawn without
    replacement, so a plan never flips one bit twice (which would silently cancel).
    """
    regions = storage_regions(qkv, guard)
    total = total_stored_bits(regions)
    if total == 0:
        return []
    if n_flips > total:
        raise ValueError(f"cannot draw {n_flips} distinct bits from {total}")
    rng = random.Random(seed)
    offsets = sorted(rng.sample(range(total), n_flips))
    sites, start, r = [], 0, 0
    for offset in offsets:
        while offset >= start + regions[r].n_bits:
            start += regions[r].n_bits
            r += 1
        sites.append(regions[r].site(offset - start, stratum))
    return sites


def flips_for_rate(qkv: QuantizedKV, guard: ProtectedKV | None, rate: float) -> int:
    """Number of flips realising a bit error rate on this arm's storage (at least one, so a rate is never a no-op)."""
    if not 0.0 < rate < 1.0:
        raise ValueError("rate must be in (0, 1)")
    return max(1, round(total_stored_bits(storage_regions(qkv, guard)) * rate))


def plan_composition(sites: list[FaultSite]) -> dict[str, Any]:
    """Where a plan's flips landed, by target -- reported so a reader can check the draw against the layout."""
    counts: dict[str, int] = {}
    for site in sites:
        counts[site.target] = counts.get(site.target, 0) + 1
    return {"n_flips": len(sites), "by_target": counts}
