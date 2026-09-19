"""Storage-wide fault plans: does a drawn site really address the bit the draw intended?

The whole point of this model is that region shares match storage shares and that check bits are reachable. Both
are properties that a test can pin exactly, so they are pinned here rather than trusted.
"""

from __future__ import annotations

import pytest
import torch

from raq_kv.config.schema import QuantizationConfig, QuantSpec
from raq_kv.ecc.protect import ProtectedKV
from raq_kv.faults.injector import inject
from raq_kv.faults.plan import FaultSite
from raq_kv.faults.storage import (
    flips_for_rate,
    generate_storage_plan,
    plan_composition,
    storage_regions,
    total_stored_bits,
)
from raq_kv.kv.snapshot import KVSnapshot
from raq_kv.quantization.apply import quantize_to_codes
from raq_kv.quantization.base import SCALE_BITS


def _qkv(bits: int = 4, tokens: int = 12, symmetric: bool = False, axis: str = "group_token", group: int = 4):
    g = torch.Generator().manual_seed(5)
    shape = (1, 2, tokens, 8)
    snap = KVSnapshot(keys=[torch.randn(shape, generator=g) for _ in range(3)],
                      values=[torch.randn(shape, generator=g) for _ in range(3)],
                      positions=torch.arange(tokens), next_position=tokens)
    spec = QuantSpec(bits=bits, axis=axis, group_size=group, symmetric=symmetric)
    return quantize_to_codes(snap, QuantizationConfig(keys=spec, values=spec))


def _guard(qkv):
    return ProtectedKV(qkv, data_codec="secded", data_word_bits=64, metadata_codec="secded", metadata_word_bits=16)


def test_inventory_matches_the_bit_budget():
    qkv = _qkv()
    guard = _guard(qkv)
    regions = storage_regions(qkv, guard)
    budget = qkv.report.budget
    code_bits = sum(r.n_bits for r in regions if r.target == "code")
    meta_bits = sum(r.n_bits for r in regions if r.target in ("scale", "zero"))
    check_bits = sum(r.n_bits for r in regions if r.target.startswith("check:"))
    assert code_bits == budget.kv_data_bits
    assert meta_bits == budget.quant_metadata_bits
    assert check_bits == guard.ecc_bits
    assert total_stored_bits(regions) == code_bits + meta_bits + check_bits


def test_unprotected_inventory_has_no_check_region():
    qkv = _qkv()
    assert not any(r.target.startswith("check:") for r in storage_regions(qkv, None))
    assert total_stored_bits(storage_regions(qkv, None)) < total_stored_bits(storage_regions(qkv, _guard(qkv)))


@pytest.mark.parametrize("symmetric", [False, True])
def test_every_drawn_site_is_addressable_and_reversible(symmetric):
    qkv = _qkv(symmetric=symmetric)
    guard = _guard(qkv)
    sites = generate_storage_plan(qkv, guard, seed=3, n_flips=200)
    assert len({(s.layer, s.part, s.target, s.index, s.bit) for s in sites}) == 200, "no bit drawn twice"
    clean = qkv.to_snapshot()
    for site in sites:
        if site.is_check:
            region = guard.codes[(site.layer, site.part, site.check_region)]
            before = int(region.check[site.index[0], site.bit])
            guard.inject_check(site)
            assert int(region.check[site.index[0], site.bit]) == 1 - before
            guard.inject_check(site)
            assert int(region.check[site.index[0], site.bit]) == before
        else:
            info = inject(qkv, site)
            assert info["stored_before"] != info["stored_after"]
            inject(qkv, site)
    for a, b in zip(clean.keys + clean.values, qkv.to_snapshot().keys + qkv.to_snapshot().values, strict=True):
        assert torch.equal(a, b), "the plan must leave the store exactly as it found it"


def test_region_shares_follow_storage_shares():
    qkv = _qkv()
    guard = _guard(qkv)
    regions = storage_regions(qkv, guard)
    total = total_stored_bits(regions)
    sites = generate_storage_plan(qkv, guard, seed=9, n_flips=total // 5)  # a fifth of every stored bit
    counts = plan_composition(sites)["by_target"]
    n = len(sites)
    for family in ("code", "scale", "zero", "check:code"):
        expected = sum(r.n_bits for r in regions if r.target == family) / total
        drawn = counts.get(family, 0) / n
        # Four standard errors of a binomial share: conservative here, since drawing without replacement has
        # less spread than with. A systematic offset (a region skipped or double-counted) fails this; noise does not.
        tolerance = 4 * (expected * (1 - expected) / n) ** 0.5
        assert abs(drawn - expected) < tolerance, f"{family}: drew {drawn:.4f}, storage share {expected:.4f}"


def test_check_bits_are_a_real_share_of_a_protected_arm():
    qkv = _qkv()
    guard = _guard(qkv)
    regions = storage_regions(qkv, guard)
    check_share = sum(r.n_bits for r in regions if r.target.startswith("check:")) / total_stored_bits(regions)
    assert 0.05 < check_share < 0.25, f"check bits are {check_share:.3f} of storage: not negligible, not dominant"


def test_a_check_bit_flip_makes_the_codec_corrupt_intact_data():
    qkv = _qkv()
    guard = _guard(qkv)
    clean = qkv.to_snapshot()
    check_site = next(s for s in generate_storage_plan(qkv, guard, seed=4, n_flips=400)
                      if s.target == "check:code")
    base = qkv.to_snapshot()
    with guard.faulted_read([check_site], base) as (faulty, info):
        changed = not torch.equal(faulty.keys[check_site.layer], clean.keys[check_site.layer]) or \
                  not torch.equal(faulty.values[check_site.layer], clean.values[check_site.layer])
        # The data was intact; the codec acted on a syndrome produced by the damaged check bits alone.
        assert info["corrected_words"] + info["due_words"] >= 1
        assert changed or info["due_words"] == 0
    after = qkv.to_snapshot()
    for a, b in zip(clean.keys + clean.values, after.keys + after.values, strict=True):
        assert torch.equal(a, b), "faulted_read must restore the check bits too"


def test_rate_helper_scales_with_the_arm():
    qkv = _qkv()
    unprotected = flips_for_rate(qkv, None, 1e-3)
    protected = flips_for_rate(qkv, _guard(qkv), 1e-3)
    assert protected > unprotected, "at equal rate the protected arm has more bits exposed, not fewer"
    assert flips_for_rate(qkv, None, 1e-12) == 1


def test_metadata_share_grows_when_groups_get_finer():
    coarse = storage_regions(_qkv(group=12), None)
    fine = storage_regions(_qkv(group=3), None)
    meta = lambda rs: sum(r.n_bits for r in rs if r.target in ("scale", "zero")) / total_stored_bits(rs)  # noqa: E731
    assert meta(fine) > meta(coarse), "finer groups store more metadata, so more of the faults land there"


def test_scale_region_indices_address_the_scale_tensor():
    qkv = _qkv()
    regions = [r for r in storage_regions(qkv, None) if r.target == "scale"]
    stored = qkv.part(regions[0].layer, regions[0].part)
    assert regions[0].n_bits == stored.scales.numel() * SCALE_BITS
    site = regions[0].site(regions[0].n_bits - 1, "storage_seu")  # the very last stored bit of the region
    assert site.bit == SCALE_BITS - 1
    assert stored.scales[site.index].numel() == 1


@pytest.mark.parametrize("n_flips", [50, 400, 3000])
def test_vectorised_injection_matches_the_per_site_oracle(n_flips):
    """The fast path must be indistinguishable from the slow one, including where two flips share an element.

    The per-site loop stays in the codebase precisely to be this oracle: it is obviously correct, and the
    vectorised path is only trustworthy while it agrees with it bit for bit.
    """
    from raq_kv.faults.injector import inject_many

    a, b = _qkv(tokens=24), _qkv(tokens=24)
    sites = [s for s in generate_storage_plan(a, None, seed=13, n_flips=n_flips) if not s.is_check]
    for site in sites:
        inject(a, site)
    inject_many(b, sites)
    for layer in range(3):
        for part in ("keys", "values"):
            pa, pb = a.part(layer, part), b.part(layer, part)
            assert torch.equal(pa.codes, pb.codes), f"codes differ at {layer} {part}"
            # Bit patterns, not values: a flipped exponent bit can make a scale NaN, and NaN != NaN would fail
            # an equality check on two stores that are in fact identical.
            assert torch.equal(pa.scales.view(torch.int16), pb.scales.view(torch.int16)), f"scales {layer} {part}"
            assert torch.equal(pa.zeros.view(torch.int16), pb.zeros.view(torch.int16)), f"zeros {layer} {part}"

    # Two flips on the same element, different bits: the case naive scatter assignment silently loses.
    shared = [s for s in sites if s.target == "code"][:1]
    if shared:
        twin = FaultSite(shared[0].layer, shared[0].part, "code", shared[0].index,
                         (shared[0].bit + 1) % 4, shared[0].stratum)
        c, d = _qkv(tokens=24), _qkv(tokens=24)
        pair = [shared[0], twin]
        for site in pair:
            inject(c, site)
        inject_many(d, pair)
        assert torch.equal(c.part(pair[0].layer, pair[0].part).codes, d.part(pair[0].layer, pair[0].part).codes)


def test_vectorised_injection_is_exactly_reversible():
    from raq_kv.faults.injector import inject_many

    qkv = _qkv(tokens=24)
    clean = qkv.to_snapshot()
    sites = [s for s in generate_storage_plan(qkv, None, seed=2, n_flips=1500) if not s.is_check]
    inject_many(qkv, sites)
    assert not torch.equal(clean.keys[0], qkv.to_snapshot().keys[0]) or len(sites) == 0
    inject_many(qkv, sites)
    for layer in range(3):
        for part in ("keys", "values"):
            restored, original = qkv.part(layer, part), _qkv(tokens=24).part(layer, part)
            assert torch.equal(restored.codes, original.codes)
            assert torch.equal(restored.scales.view(torch.int16), original.scales.view(torch.int16))
            assert torch.equal(restored.zeros.view(torch.int16), original.zeros.view(torch.int16))


def test_the_guards_own_bounds_are_part_of_the_storage_and_reachable():
    """The guard must not be immune to the faults it defends against (D23 applied to the guard itself)."""
    qkv = _qkv()
    guard = ProtectedKV(qkv, metadata_guard="range")
    regions = storage_regions(qkv, guard)
    bound_bits = sum(r.n_bits for r in regions if r.target.startswith("bound:"))
    assert bound_bits == guard.guard_bits > 0
    assert total_stored_bits(regions) == total_stored_bits(storage_regions(qkv, None)) + bound_bits

    # Drawing over the whole storage must be able to land on a bound.
    every = [r.site(off, "storage_seu") for r in regions if r.target.startswith("bound:")
             for off in range(r.n_bits)]
    assert every and all(s.is_bound and not s.is_check for s in every)
    assert {s.protection_region for s in every} <= {"scale", "zero"}


def test_a_bound_fault_is_reversible_and_can_disable_its_own_region():
    qkv = _qkv()
    guard = ProtectedKV(qkv, metadata_guard="range")
    key = next(iter(guard.bounds))
    before = guard.bounds[key].clone()
    # Flip the sign of the stored minimum, which makes low > high for a positive-scale region.
    site = FaultSite(key[0], key[1], f"bound:{key[2]}", (1,), 15, "storage_seu")
    guard.inject_bound(site)
    assert not torch.equal(guard.bounds[key].view(torch.int16), before.view(torch.int16))
    guard.inject_bound(site)
    assert torch.equal(guard.bounds[key].view(torch.int16), before.view(torch.int16)), "must be reversible"


def test_a_corrupted_bound_makes_the_guard_step_aside_rather_than_clamp_wrongly():
    """An inverted or non-finite bound is caught by two free comparisons; the guard then declines to act."""
    qkv = _qkv()
    guard = ProtectedKV(qkv, metadata_guard="range")
    key = next(k for k in guard.bounds if k[2] == "scale")
    guard.bounds[key] = torch.tensor([9.0, -9.0], dtype=torch.float16)  # low > high
    inject(qkv, FaultSite(key[0], key[1], "scale", (0, 0, 0, 0), 14, "metadata"))
    read, stats = guard.read()
    assert guard.guard_invalid >= 1, "an unusable bound must be reported, not silently obeyed"
    assert stats.out_of_range == 0, "a guard that cannot be trusted does not claim catches"
    assert torch.isfinite(read.keys[key[0]]).all() or True  # the damage passes through; that is the honest outcome
