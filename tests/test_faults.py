from __future__ import annotations

from pathlib import Path

import pytest
import torch

from raq_kv.config.schema import QuantizationConfig, QuantSpec
from raq_kv.faults.bit_utils import bit_significance, flip_code_bit, flip_float16_bit, float16_field
from raq_kv.faults.injector import inject, injected
from raq_kv.faults.plan import FaultSite, describe_site, generate_plan, load_plan, save_plan
from raq_kv.kv.snapshot import KVSnapshot
from raq_kv.quantization.apply import quantize_to_codes


def _qkv(bits: int = 4, tokens: int = 12):
    g = torch.Generator().manual_seed(4)
    keys = [torch.randn(1, 4, tokens, 16, generator=g) for _ in range(3)]
    values = [torch.randn(1, 4, tokens, 16, generator=g) for _ in range(3)]
    snap = KVSnapshot(keys=keys, values=values, positions=torch.arange(tokens), next_position=tokens)
    cfg = QuantizationConfig(keys=QuantSpec(bits=bits, axis="per_channel"),
                             values=QuantSpec(bits=bits, axis="per_token"))
    return quantize_to_codes(snap, cfg)


def test_bit_convention_lsb_is_bit_zero():
    codes = torch.tensor([[0b0101]], dtype=torch.uint8)
    assert flip_code_bit(codes, (0, 0), 0, 4) == 0b0100  # LSB
    assert flip_code_bit(codes, (0, 0), 3, 4) == 0b1100  # MSB of a 4-bit code
    with pytest.raises(ValueError):
        flip_code_bit(codes, (0, 0), 4, 4)
    assert bit_significance(3, 4) == "msb" and bit_significance(0, 4) == "lsb"
    assert bit_significance(2, 4) == "upper" and bit_significance(1, 4) == "lower"


def test_float16_bit_flips_hit_the_documented_fields():
    x = torch.tensor([1.0], dtype=torch.float16)
    assert flip_float16_bit(x.clone(), (0,), 15) == -1.0  # sign
    exponent_flipped = flip_float16_bit(x.clone(), (0,), 14)
    assert abs(exponent_flipped) > 100  # an exponent flip rescales the whole group
    mantissa_flipped = flip_float16_bit(x.clone(), (0,), 0)
    assert 1.0 < mantissa_flipped < 1.01
    assert float16_field(15) == "sign" and float16_field(12) == "exponent" and float16_field(3) == "mantissa"


def test_injection_is_exactly_reversible():
    qkv = _qkv()
    before = qkv.to_snapshot()
    site = FaultSite(1, "keys", "code", (0, 2, 3, 4), 3, "uniform")
    with injected(qkv, site) as info:
        during = qkv.to_snapshot()
        assert info["stored_after"] != info["stored_before"]
        assert not torch.equal(during.keys[1], before.keys[1])
    after = qkv.to_snapshot()
    for a, b in zip(before.keys + before.values, after.keys + after.values, strict=True):
        assert torch.equal(a, b)


def test_injection_only_touches_its_own_layer_and_part():
    qkv = _qkv()
    before = qkv.to_snapshot()
    with injected(qkv, FaultSite(0, "values", "code", (0, 1, 2, 3), 3, "uniform")):
        during = qkv.to_snapshot()
        assert not torch.equal(during.values[0], before.values[0])
        assert torch.equal(during.keys[0], before.keys[0])
        for layer in (1, 2):
            assert torch.equal(during.values[layer], before.values[layer])


def test_metadata_faults_change_a_whole_group():
    qkv = _qkv()
    before = qkv.to_snapshot()
    site = FaultSite(0, "keys", "scale", (0, 1, 0, 5), 14, "metadata")  # exponent bit of one channel's scale
    with injected(qkv, site):
        during = qkv.to_snapshot()
    changed = (during.keys[0] != before.keys[0]).sum(dim=(0, 1, 2))
    assert int(changed[5]) > 0 and int(changed.sum()) == int(changed[5])  # exactly that channel, all its tokens


def test_symmetric_tensors_have_no_zero_points_to_corrupt():
    g = torch.Generator().manual_seed(9)
    snap = KVSnapshot(keys=[torch.randn(1, 2, 8, 16, generator=g)], values=[torch.randn(1, 2, 8, 16, generator=g)],
                      positions=torch.arange(8), next_position=8)
    qkv = quantize_to_codes(snap, QuantizationConfig(keys=QuantSpec(bits=4, symmetric=True),
                                                     values=QuantSpec(bits=4, symmetric=True)))
    with pytest.raises(ValueError, match="symmetric"):
        inject(qkv, FaultSite(0, "keys", "zero", (0, 0, 0, 0), 3, "metadata"))


def test_plan_is_deterministic_stratified_and_round_trips(tmp_path: Path):
    qkv = _qkv()
    kwargs = {"n_uniform": 20, "n_msb": 5, "n_metadata": 5, "n_lsb": 5}
    plan = generate_plan(qkv, seed=7, **kwargs)
    assert plan == generate_plan(qkv, seed=7, **kwargs)
    assert plan != generate_plan(qkv, seed=8, **kwargs)
    strata = {s.stratum for s in plan}
    assert strata == {"uniform", "msb", "metadata", "lsb_midrange"}
    assert all(s.bit == 3 for s in plan if s.stratum == "msb")  # 4-bit codes -> MSB is bit 3
    assert all(s.bit == 0 and s.target == "code" for s in plan if s.stratum == "lsb_midrange")
    assert all(s.target in ("scale", "zero") for s in plan if s.stratum == "metadata")

    path = tmp_path / "plan.jsonl"
    save_plan(path, plan, {"seed": 7})
    loaded, meta = load_plan(path)
    assert loaded == plan and meta["seed"] == 7


def test_lsb_sites_move_the_stored_value_least_but_not_negligibly():
    """The smallest possible storage error is still one quantization step — at 4 bits that is 1/15 of the range,
    which is why there is no 'inert bit' stratum (D17)."""
    qkv = _qkv(bits=4)
    plan = generate_plan(qkv, seed=3, n_uniform=40, n_lsb=40)

    def mean_change(stratum: str) -> float:
        changes = []
        for site in (s for s in plan if s.stratum == stratum):
            with injected(qkv, site) as info:
                changes.append(info["relative_change"])
        return sum(changes) / len(changes)

    lsb = mean_change("lsb_midrange")
    assert lsb < mean_change("uniform") / 2
    assert lsb >= 1 / ((1 << 4) - 1) - 1e-9  # one code step, not "nothing"


def test_role_targeted_sites_stay_inside_their_tokens():
    qkv = _qkv(tokens=20)
    from raq_kv.faults.plan import generate_plan_in_tokens

    target_tokens = [5, 6, 7]
    sites = generate_plan_in_tokens(qkv, seed=2, tokens=target_tokens, stratum="target_value", n=30)
    assert len(sites) == 30
    assert all(s.stratum == "target_value" and s.target == "code" for s in sites)
    assert {s.index[2] for s in sites} <= set(target_tokens)
    assert len({s.index[2] for s in sites}) > 1  # not stuck on one token
    assert {s.part for s in sites} == {"keys", "values"}  # both parts reachable


def test_role_tokens_split_target_distractor_and_filler():
    from raq_kv.data.prompts import PromptSample, Span
    from raq_kv.experiments.tasks import FaultQA

    spans = (
        Span("fact_value", "target", 0, 1, 10, 13, "4821"),
        Span("fact_value", "distractor", 0, 1, 30, 32, "7777"),
        Span("fact_sentence", "target", 0, 1, 5, 20, "..."),  # not a value span: ignored
    )
    sample = PromptSample(sample_id="s", prompt_text="", prompt_ids=torch.zeros(40, dtype=torch.long),
                          cache_len=35, scorer="exact_value", reference={}, gold_text=None,
                          gold_scored_chars=None, max_new_tokens=8, spans=spans)
    roles = FaultQA._role_tokens(sample, sample.cache_len)
    assert roles["target_value"] == [10, 11, 12]
    assert roles["distractor_value"] == [30, 31]
    assert set(roles["filler"]).isdisjoint({10, 11, 12, 30, 31})
    assert len(roles["filler"]) == 35 - 5


def test_describe_site_labels_codes_and_metadata():
    qkv = _qkv()
    code = describe_site(FaultSite(0, "keys", "code", (0, 1, 2, 3), 3, "uniform"), qkv)
    assert code["significance"] == "msb" and code["head"] == 1 and code["token"] == 2 and code["channel"] == 3
    meta = describe_site(FaultSite(0, "keys", "scale", (0, 1, 0, 5), 14, "metadata"), qkv)
    assert meta["significance"] == "exponent"
