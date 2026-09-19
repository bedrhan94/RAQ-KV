from __future__ import annotations

import pytest
import torch

from raq_kv.config.schema import QuantizationConfig, QuantSpec
from raq_kv.kv.snapshot import KVSnapshot
from raq_kv.quantization.apply import quantize_region, quantize_snapshot
from raq_kv.quantization.base import SCALE_BITS, error_stats
from raq_kv.quantization.uniform import dequantize, quantize, quantize_dequantize

SHAPE = (1, 4, 16, 32)  # [batch, kv_heads, tokens, head_dim]


@pytest.fixture
def kv() -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    x = torch.randn(SHAPE, generator=g)
    x[0, 1, :, 5] *= 30  # an outlier channel, as real key tensors have
    return x


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("axis", ["per_tensor", "per_head", "per_channel", "per_token", "group_channel",
                                  "group_token"])
@pytest.mark.parametrize("symmetric", [False, True])
def test_codes_are_in_range_and_shapes_survive(kv, bits, axis, symmetric):
    deq, qt = quantize_dequantize(kv, bits, axis, group_size=8, symmetric=symmetric)
    assert qt.codes.dtype == torch.uint8 and qt.codes.shape == kv.shape
    assert int(qt.codes.max()) <= (1 << bits) - 1
    assert deq.shape == kv.shape and deq.dtype == kv.dtype
    assert (qt.zeros is None) == symmetric


def test_error_decreases_with_bit_width(kv):
    mses = [error_stats(kv, quantize_dequantize(kv, b, "per_token")[0])["mse"] for b in (2, 3, 4, 8)]
    assert mses == sorted(mses, reverse=True)
    assert error_stats(kv, quantize_dequantize(kv, 8, "per_token")[0])["sqnr_db"] > 35


def test_finer_granularity_is_not_worse(kv):
    per_tensor = error_stats(kv, quantize_dequantize(kv, 4, "per_tensor")[0])["mse"]
    per_token = error_stats(kv, quantize_dequantize(kv, 4, "per_token")[0])["mse"]
    per_channel = error_stats(kv, quantize_dequantize(kv, 4, "per_channel")[0])["mse"]
    grouped = error_stats(kv, quantize_dequantize(kv, 4, "group_channel", group_size=8)[0])["mse"]
    assert per_token < per_tensor and per_channel < per_tensor and grouped < per_token


def test_persistent_channel_scale_favours_per_channel():
    """Mechanism behind KIVI's per-channel keys: when each channel has its own magnitude that persists across
    tokens, per-channel scales fit it and per-token scales are set by the largest channel.

    (A single outlier channel is *not* enough: per-token then represents the outlier exactly at the range endpoint,
    so MSE can favour per-token - measured on random data. Which granularity wins for real Qwen3 keys is an
    empirical question answered by the M1 sweep, not an assumption.)
    """
    g = torch.Generator().manual_seed(5)
    magnitudes = torch.logspace(-1, 2, SHAPE[3]).reshape(1, 1, 1, -1)  # channel scale spans 0.1 .. 100
    x = torch.randn(SHAPE, generator=g) * magnitudes

    def worst_relative_error(deq: torch.Tensor) -> float:
        err = ((deq - x) ** 2).mean(dim=(0, 1, 2))
        return float((err / (x**2).mean(dim=(0, 1, 2))).max())

    per_channel = quantize_dequantize(x, 2, "per_channel")[0]
    per_token = quantize_dequantize(x, 2, "per_token")[0]
    assert error_stats(x, per_channel)["mse"] < error_stats(x, per_token)["mse"]
    # Total MSE is dominated by the largest channels; the per-channel gain shows up in relative error.
    assert worst_relative_error(per_channel) < worst_relative_error(per_token) / 100


def _causal(axis: str, group_size: int = 8) -> bool:
    """Does the reconstruction of early tokens depend on tokens that arrive later?

    A streaming decoder can only use schemes where it does not. per_channel computes one scale over every token,
    so it is an offline scheme; group_token closes a block once it is full.
    """
    g = torch.Generator().manual_seed(17)
    x = torch.randn(1, 2, 24, 16, generator=g)
    later = x.clone()
    later[0, :, 20:, :] *= 50  # a violent change confined to the last block
    early = slice(0, 16)  # tokens in earlier, already-closed blocks
    a = quantize_dequantize(x, 4, axis, group_size=group_size)[0][:, :, early]
    b = quantize_dequantize(later, 4, axis, group_size=group_size)[0][:, :, early]
    return torch.equal(a, b)


def test_group_token_is_causal_and_per_channel_is_not():
    assert _causal("group_token"), "group_token must not let later tokens change earlier reconstructions"
    assert not _causal("per_channel"), "control: per_channel is an offline scheme (scales span all tokens)"
    assert _causal("per_token"), "per_token scales are per position, so they are causal too"


def test_group_token_handles_context_lengths_that_are_not_a_multiple_of_the_block():
    g = torch.Generator().manual_seed(23)
    x = torch.randn(1, 2, 17, 16, generator=g)  # 17 tokens, blocks of 8 -> padded to 24
    deq, qt = quantize_dequantize(x, 4, "group_token", group_size=8)
    assert deq.shape == x.shape and qt.codes.shape == x.shape
    assert qt.n_groups == 1 * 2 * 3 * 16  # 3 blocks x 16 channels
    assert error_stats(x, deq)["sqnr_db"] > 20


def test_values_on_the_quantization_grid_round_trip_exactly():
    scale = torch.tensor(0.25, dtype=torch.float16)
    codes = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    x = codes * scale.to(torch.float32)  # exactly representable with bits=4, zero=0
    deq, qt = quantize_dequantize(x, 4, "per_tensor")
    assert torch.equal(deq, x)
    assert error_stats(x, deq)["mse"] == 0.0


def test_bit_accounting_matches_stored_objects(kv):
    qt = quantize(kv, 4, "per_token")
    n = kv.numel()
    assert qt.data_bits == n * 4
    assert qt.n_groups == SHAPE[0] * SHAPE[1] * SHAPE[2]  # one scale per (head, token)
    assert qt.metadata_bits == qt.n_groups * SCALE_BITS * 2  # scale + zero point
    assert quantize(kv, 4, "per_token", symmetric=True).metadata_bits == qt.n_groups * SCALE_BITS
    grouped = quantize(kv, 4, "group_channel", group_size=8)
    assert grouped.n_groups == n // 8
    budget = qt.budget(retained_tokens=SHAPE[2])
    assert budget.total_bits == qt.data_bits + qt.metadata_bits


def test_group_size_must_divide_head_dim(kv):
    with pytest.raises(ValueError, match="divisible"):
        quantize(kv, 4, "group_channel", group_size=7)
    with pytest.raises(ValueError, match="axis"):
        quantize(kv, 4, "nonsense")


def test_symmetric_dequantization_is_centred(kv):
    zeros = torch.zeros(SHAPE)
    deq, qt = quantize_dequantize(zeros, 4, "per_token", symmetric=True)
    assert torch.equal(deq, zeros) and qt.zeros is None
    assert torch.equal(dequantize(qt), zeros)


def _snapshot(n_layers: int = 3, tokens: int = 16) -> KVSnapshot:
    g = torch.Generator().manual_seed(11)
    keys = [torch.randn(1, 4, tokens, 32, generator=g) for _ in range(n_layers)]
    values = [torch.randn(1, 4, tokens, 32, generator=g) for _ in range(n_layers)]
    return KVSnapshot(keys=keys, values=values, positions=torch.arange(tokens), next_position=tokens)


def test_quantize_snapshot_budget_and_shapes():
    snap = _snapshot()
    cfg = QuantizationConfig(keys=QuantSpec(bits=4, axis="per_channel"), values=QuantSpec(bits=4, axis="per_token"))
    out, report = quantize_snapshot(snap, cfg)
    assert out.seq_len == snap.seq_len and out.num_layers == snap.num_layers
    assert torch.equal(out.positions, snap.positions) and out.next_position == snap.next_position
    assert not torch.equal(out.keys[0], snap.keys[0])  # actually quantized
    values_per_layer = snap.keys[0].numel()
    assert report.budget.kv_data_bits == 2 * 3 * values_per_layer * 4
    assert report.budget.quant_metadata_bits > 0
    assert report.stats["keys"]["sqnr_db_mean"] > 10


def test_residual_window_keeps_recent_tokens_exact():
    snap = _snapshot(n_layers=2, tokens=16)
    cfg = QuantizationConfig(keys=QuantSpec(bits=2), values=QuantSpec(bits=2), residual_window=4)
    out, report = quantize_snapshot(snap, cfg)
    assert torch.equal(out.keys[0][:, :, -4:, :], snap.keys[0][:, :, -4:, :])
    assert not torch.equal(out.keys[0][:, :, :-4, :], snap.keys[0][:, :, :-4, :])
    quantized = 2 * 2 * snap.keys[0][:, :, :-4, :].numel() * 2
    residual = 2 * 2 * snap.keys[0][:, :, -4:, :].numel() * 32  # float32 tensors in this fixture
    assert report.budget.kv_data_bits == quantized + residual


def test_keys_only_quantization_leaves_values_untouched():
    snap = _snapshot(n_layers=2)
    out, report = quantize_snapshot(snap, QuantizationConfig(keys=QuantSpec(bits=8)))
    assert torch.equal(out.values[0], snap.values[0])
    assert report.stats["values"]["spec"] is None


def test_quantize_region_touches_only_its_region():
    snap = _snapshot(n_layers=3, tokens=12)
    spec = QuantSpec(bits=2, axis="per_token")
    out, stats = quantize_region(snap, layer=1, part="keys", spec=spec)
    assert stats["layer"] == 1 and stats["part"] == "keys" and stats["head"] is None
    assert not torch.equal(out.keys[1], snap.keys[1])
    for i in (0, 2):
        assert out.keys[i] is snap.keys[i]  # other layers are shared, not copied
    for i in range(3):
        assert out.values[i] is snap.values[i]
    assert torch.equal(out.positions, snap.positions) and out.next_position == snap.next_position


def test_quantize_region_can_target_a_single_head():
    snap = _snapshot(n_layers=2, tokens=12)
    out, stats = quantize_region(snap, layer=0, part="values", spec=QuantSpec(bits=2), head=2)
    assert stats["head"] == 2
    changed = [h for h in range(snap.values[0].shape[1]) if not torch.equal(out.values[0][:, h], snap.values[0][:, h])]
    assert changed == [2]


def test_quantize_region_rejects_unknown_part():
    with pytest.raises(ValueError, match="part must be"):
        quantize_region(_snapshot(n_layers=1), layer=0, part="queries", spec=QuantSpec(bits=4))


def test_empty_quantization_section_is_rejected():
    with pytest.raises(ValueError, match="omit the section"):
        QuantizationConfig()
