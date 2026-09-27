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


def governed_values(qkv: QuantizedKV, site: FaultSite) -> int:
    """How many stored values the corrupted element actually reaches. A code bit reaches one; a scale reaches its group.

    D76 needs this and the nominal ``group_size`` will not do, because **the trailing group of a sequence is short and
    it is not rare.** On the RAQ probe the median cache is 2052 tokens, so at ``group_size`` 512 there are 5 groups and
    the last holds **4** tokens: one draw in five lands on an element governing 4 values rather than 512, a 128×
    discrepancy, and at the shortest cache (514 tokens) there are 2 groups and it is one draw in two. Regressing damage
    on the nominal group size would compress the realised 32-512 span from 16× to **13×** and bias the exponent
    **downward by a factor of 0.925** — toward the conclusion that larger groups are strictly better, which is the most
    consequential of the three answers D76 names.

    Recorded per trial, it turns that sweep from five contaminated rungs into a regression on the realised span, where
    the short groups are data rather than contamination.

    The group size comes from the ``QuantizedTensor``'s own spec rather than being inferred from the grids, because
    ``ceil(C/S)`` is not the group size when the last group is partial: at 2052 tokens in 5 groups it gives 411, not
    512.
    """
    stored = qkv.part(site.layer, site.part)
    if not isinstance(stored, QuantizedTensor):
        raise ValueError(f"layer {site.layer} {site.part} is not quantized, so it has no governed values")
    if site.target not in ("scale", "zero"):
        return 1
    grid = stored.scales if site.target == "scale" else stored.zeros
    if grid is None:
        raise ValueError(f"layer {site.layer} {site.part} has no {site.target} grid (symmetric quantization?)")
    codes, scales = tuple(stored.codes.shape), tuple(grid.shape)
    if len(codes) != len(scales):
        raise ValueError(f"grid rank {scales} does not match code rank {codes}")
    total, grouped = 1, 0
    for axis, (c, s) in enumerate(zip(codes, scales, strict=True)):
        if s == c:
            continue  # one metadata element per value along this axis
        if s == 1:
            total *= c  # this element spans the whole axis
            continue
        grouped += 1
        if stored.group_size is None:
            raise ValueError(f"axis {axis} is grouped ({s} against {c}) but the tensor records no group_size")
        start = site.index[axis] * stored.group_size
        total *= max(0, min(stored.group_size, c - start))
    if grouped > 1:
        raise ValueError(f"grid {scales} against codes {codes} groups {grouped} axes; this helper assumes at most one")
    return total


def generate_metadata_plan(
    qkv: QuantizedKV,
    seed: int,
    n_flips: int,
    stratum: str = "metadata_uniform",
    part: str | None = None,
) -> list[FaultSite]:
    """``n_flips`` single-bit upsets uniform over the arm's **quantization metadata bits**.

    Deliberately not ``generate_plan(..., n_metadata=n)``. That helper picks a layer-part uniformly and then splits
    scale/zero 50/50, which induces a distribution over metadata *bits* quite different from the storage-wide
    uniform draw: keys are ``group_token`` and values ``per_token``, so their scale grids differ by orders of
    magnitude in element count, and a uniform layer-part pick over-samples the small ones.

    This draws uniformly over bits, exactly as ``generate_storage_plan`` does, with the population narrowed to
    scale and zero regions. That is what makes a per-metadata-flip rate measured here comparable to one derived
    from a storage-wide campaign by conditioning on the flips that landed in metadata (D59) -- the same population,
    reached from the other direction, rather than a second population with a similar name.

    ``part`` narrows the draw to ``"keys"`` or ``"values"``, and it exists because a sweep over the **key** group size
    is otherwise uninterpretable. Keys are ``group_token`` and values ``per_token``, so varying the key group size
    changes the number of key scales by the same factor while leaving the value scales untouched: on the probe at
    ``group_size`` 32 there are roughly 66,600 key scales against 16,500 value scales, and at 512 roughly 5,100
    against the same 16,500. **The mixture of which part a uniform draw lands on therefore moves with the group
    size**, and a value scale governs a different number of values than a key scale, so damage per flip would change
    for a reason that has nothing to do with the quantity under test. Default ``None`` keeps the storage-wide draw, so
    no existing arm's population changes.
    """
    if part not in (None, "keys", "values"):
        raise ValueError(f"part must be None, 'keys' or 'values', not {part!r}")
    regions = [r for r in storage_regions(qkv, None)
               if r.target in ("scale", "zero") and (part is None or r.part == part)]
    total = total_stored_bits(regions)
    if total == 0:
        raise ValueError(f"no metadata regions for part={part!r}; a symmetric unquantized arm has no scales "
                         f"to corrupt")
    if n_flips > total:
        raise ValueError(f"cannot draw {n_flips} distinct metadata bits from {total}")
    rng = random.Random(seed)
    offsets = sorted(rng.sample(range(total), n_flips))
    sites, start, r = [], 0, 0
    for offset in offsets:
        while offset >= start + regions[r].n_bits:
            start += regions[r].n_bits
            r += 1
        sites.append(regions[r].site(offset - start, stratum))
    return sites


def generate_group_targeted_plan(
    qkv: QuantizedKV,
    seed: int,
    token_group: int,
    n_token_groups: int,
    n_flips: int = 1,
    part: str = "keys",
    stratum: str = "group_targeted",
) -> list[FaultSite]:
    """Single-bit upsets confined to the **key scales covering one token group**.

    Built for D60's open question. The uniform plan above cannot answer it: two candidate mechanisms for why one
    model sits at the metadata hazard ceiling -- "more groups are consequential" and "the groups that matter matter
    more" -- predict the same *average* hazard and differ only in P(break | which group was hit). Measuring that
    needs the group chosen rather than drawn.

    Restricted to ``group_token`` scale grids on purpose, and the restriction is **checked** rather than described:
    ``n_token_groups`` is the group count the caller derives from its own config, and a grid whose axis 2 is not
    that length is refused. Without that check the guard would be ``len(shape) == 4``, which a ``per_token`` grid
    ``(1, H, T, 1)`` also satisfies -- ``token_group`` would then be silently reinterpreted as a token index and the
    function would return sites for a different unit than its name. That is D24's and D42's failure mode: a
    manifest internally consistent with itself and wrong about what ran.

    One flip corrupts one (layer, head, channel) scale within the group, not the group as a whole: at H_kv = 8 and
    head_dim = 128 there are 1024 such scales per group per layer-part. That dilution is exactly why the
    conditional break probability D60 derived sits near 20-40% rather than near 1, and it is preserved here so the
    measured conditional is comparable to the derived one.
    """
    if part not in ("keys", "values"):
        raise ValueError(f"part must be 'keys' or 'values', got {part!r}")
    if not 0 <= token_group < n_token_groups:
        raise ValueError(f"token_group {token_group} outside the {n_token_groups} groups requested")
    candidates: list[Region] = []
    for region in storage_regions(qkv, None):
        if region.part != part or region.target not in ("scale", "zero"):
            continue
        if len(region.shape) != 4 or region.shape[2] != n_token_groups:
            raise ValueError(
                f"layer {region.layer} {region.part} {region.target} has grid {region.shape}, which is not a "
                f"group_token grid of {n_token_groups} groups -- this helper is defined only for those (a "
                f"per_token grid is (1, H, T, 1) and would reindex token_group as a token)")
        candidates.append(region)
    if not candidates:
        raise ValueError(f"no {part} scale regions found; the arm may be unquantized on that part")

    rng = random.Random(seed)
    sites: list[FaultSite] = []
    seen: set[tuple[Any, ...]] = set()
    # Drawn without replacement over (region, head, channel, bit), matching the uniform generator's guarantee:
    # two flips in one scale would partially cancel and the trial would understate its own exposure.
    attempts, budget = 0, 1000 * max(1, n_flips)
    while len(sites) < n_flips and attempts < budget:
        attempts += 1
        region = rng.choice(candidates)
        head = rng.randrange(region.shape[1])
        channel = rng.randrange(region.shape[3])
        bit = rng.randrange(region.bits_per_element)
        key = (region.layer, region.part, region.target, head, channel, bit)
        if key in seen:
            continue
        seen.add(key)
        sites.append(FaultSite(region.layer, region.part, region.target,
                               (0, head, token_group, channel), bit, stratum))
    if len(sites) < n_flips:
        raise ValueError(f"could not draw {n_flips} distinct sites in group {token_group} of {len(candidates)} "
                         f"regions after {attempts} attempts")
    return sites


def flips_for_rate(qkv: QuantizedKV, guard: ProtectedKV | None, rate: float) -> int:
    """Number of flips realising a bit error rate on this arm's storage (at least one, so a rate is never a no-op)."""
    if not 0.0 < rate < 1.0:
        raise ValueError("rate must be in (0, 1)")
    return max(1, round(total_stored_bits(storage_regions(qkv, guard)) * rate))


def plan_composition(sites: list[FaultSite]) -> dict[str, Any]:
    """Where a plan's flips landed, by target -- reported so a reader can check the draw against the layout."""
    counts: dict[str, int] = {}
    parts: dict[str, int] = {}
    for site in sites:
        counts[site.target] = counts.get(site.target, 0) + 1
        # D76 wanted to filter an existing arm's flips to keys only and could not: the part was never recorded, so a
        # storage-wide draw's records cannot be split after the fact even though its population is 94% value scales on
        # this geometry. One dict, and the next arm does not have the same gap.
        parts[site.part] = parts.get(site.part, 0) + 1
    return {"n_flips": len(sites), "by_target": counts, "by_part": parts}
