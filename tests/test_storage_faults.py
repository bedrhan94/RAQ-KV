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
