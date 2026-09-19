"""Protected storage: does the codec actually repair what the injector breaks, and is the cost counted?"""

from __future__ import annotations

import pytest
import torch

from raq_kv.config.schema import QuantizationConfig, QuantSpec
from raq_kv.ecc.protect import ProtectedKV
from raq_kv.faults.injector import inject
from raq_kv.faults.plan import FaultSite
from raq_kv.kv.snapshot import KVSnapshot
from raq_kv.quantization.apply import quantize_to_codes


def _qkv(bits: int = 4, tokens: int = 16, symmetric: bool = False):
    g = torch.Generator().manual_seed(21)
    keys = [torch.randn(1, 2, tokens, 32, generator=g) for _ in range(2)]
    values = [torch.randn(1, 2, tokens, 32, generator=g) for _ in range(2)]
    snap = KVSnapshot(keys=keys, values=values, positions=torch.arange(tokens), next_position=tokens)
    cfg = QuantizationConfig(keys=QuantSpec(bits=bits, axis="per_channel", symmetric=symmetric),
                             values=QuantSpec(bits=bits, axis="per_token", symmetric=symmetric))
    return quantize_to_codes(snap, cfg)


def test_unprotected_read_matches_plain_dequantization():
    qkv = _qkv()
    plain = qkv.to_snapshot()
    protected, stats = ProtectedKV(qkv).read()
    assert stats.words_read == 0 and ProtectedKV(qkv).ecc_bits == 0
    for a, b in zip(plain.keys + plain.values, protected.keys + protected.values, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("symmetric", [False, True])
def test_protected_read_is_lossless_without_faults(symmetric):
    qkv = _qkv(symmetric=symmetric)
    plain = qkv.to_snapshot()
    guard = ProtectedKV(qkv, data_codec="secded", data_word_bits=64, metadata_codec="secded", metadata_word_bits=16)
    read, stats = guard.read()
    assert stats.corrected == 0 and stats.uncorrectable == 0
    for a, b in zip(plain.keys + plain.values, read.keys + read.values, strict=True):
        assert torch.equal(a, b)


def test_ecc_bits_match_the_arithmetic():
    qkv = _qkv(bits=4, tokens=16)
    guard = ProtectedKV(qkv, data_codec="secded", data_word_bits=64)
    parts = (qkv.part(layer, part) for layer in range(2) for part in ("keys", "values"))
    code_bits = sum(t.codes.numel() * t.bits for t in parts)
    words = code_bits // 64  # 4 layers-parts x 2x16x32 values x 4 bits, divisible by 64
    assert guard.ecc_bits == words * 8  # SECDED over 64 data bits costs 8 check bits
    assert guard.ecc_bits / code_bits == pytest.approx(0.125)


def test_secded_repairs_a_single_code_fault_and_parity_only_flags_it():
    site = FaultSite(1, "keys", "code", (0, 1, 5, 7), 2, "uniform")
    clean = _qkv().to_snapshot()

    unprotected = _qkv()
    inject(unprotected, site)
    assert not torch.equal(unprotected.to_snapshot().keys[1], clean.keys[1])  # the fault is real

    protected = _qkv()
    guard = ProtectedKV(protected, data_codec="secded", data_word_bits=64)
    inject(protected, site)
    read, stats = guard.read()
    assert stats.corrected == 1 and stats.uncorrectable == 0
    assert torch.equal(read.keys[1], clean.keys[1]), "SECDED must restore the exact pre-fault values"

    detected = _qkv()
    parity_guard = ProtectedKV(detected, data_codec="parity", data_word_bits=64)
    inject(detected, site)
    read_p, stats_p = parity_guard.read()
    assert stats_p.corrected == 0 and stats_p.uncorrectable == 1  # detected, not repaired
    assert not torch.equal(read_p.keys[1], clean.keys[1])


def test_metadata_protection_repairs_a_scale_fault():
    clean = _qkv().to_snapshot()
    qkv = _qkv()
    guard = ProtectedKV(qkv, metadata_codec="secded", metadata_word_bits=16)
    inject(qkv, FaultSite(0, "values", "scale", (0, 1, 3, 0), 14, "metadata"))
    read, stats = guard.read()
    assert stats.corrected == 1
    assert torch.equal(read.values[0], clean.values[0])


def test_layer_selective_protection_covers_only_its_layers():
    qkv = _qkv()
    guard = ProtectedKV(qkv, data_codec="secded", data_word_bits=64, data_layers=(0,))
    assert {key[0] for key in guard.codes} == {0}
    inject(qkv, FaultSite(1, "keys", "code", (0, 0, 2, 3), 1, "uniform"))
    _, stats = guard.read()
    assert stats.corrected == 0, "layer 1 is unprotected, so nothing should be repaired"


def test_double_fault_in_one_word_becomes_a_reported_erasure():
    qkv = _qkv()
    guard = ProtectedKV(qkv, data_codec="secded", data_word_bits=64, due_policy="erasure_mid")
    # two flips inside the same 64-bit word: bits 0 and 2 of neighbouring codes of the same head/token
    inject(qkv, FaultSite(0, "keys", "code", (0, 0, 0, 0), 0, "uniform"))
    inject(qkv, FaultSite(0, "keys", "code", (0, 0, 0, 1), 2, "uniform"))
    read, stats = guard.read()
    assert stats.uncorrectable == 1 and stats.corrected == 0
    assert stats.regions == {"keys_code": 1}
    assert not torch.equal(read.keys[0], qkv.to_snapshot().keys[0]), "the erasure replaces the damaged word"
