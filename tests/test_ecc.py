"""Codec correctness is proved by exhaustion on small words, not by spot checks.

For SECDED over 8 and 16 data bits every single-bit error pattern must be corrected and every double-bit
pattern must be detected-but-not-corrected — over the whole codeword, check bits included, because a fault does
not respect the payload/parity boundary.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from raq_kv.ecc.bits import (
    bits_to_codes,
    bits_to_float16,
    codes_to_bits,
    float16_to_bits,
    pack_words,
    unpack_words,
)
from raq_kv.ecc.codecs import SecdedCodec, make_codec, overhead_ratio


def _random_payload(words: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (words, k), generator=g, dtype=torch.uint8)


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_code_bit_roundtrip_is_lsb_first(bits):
    codes = torch.arange(1 << bits, dtype=torch.uint8)
    bit_matrix = codes_to_bits(codes, bits)
    assert bit_matrix.shape == (1 << bits, bits)
    assert torch.equal(bit_matrix[:, 0], (codes & 1))  # bit 0 is the LSB
    assert torch.equal(bits_to_codes(bit_matrix), codes)


def test_float16_bit_roundtrip_and_field_positions():
    values = torch.tensor([1.0, -1.0, 0.5, 65504.0, 6e-8], dtype=torch.float16)
    bit_matrix = float16_to_bits(values)
    assert bit_matrix.shape == (5, 16)
    assert torch.equal(bit_matrix[:, 15], torch.tensor([0, 1, 0, 0, 0], dtype=torch.uint8))  # sign
    assert torch.equal(bits_to_float16(bit_matrix), values)


def test_word_packing_pads_and_strips():
    stream = torch.randint(0, 2, (37,), dtype=torch.uint8)
    words, pad = pack_words(stream, 8)
    assert words.shape == (5, 8) and pad == 3
    assert torch.equal(unpack_words(words, pad), stream)


@pytest.mark.parametrize("name,k,expected", [("none", 8, 0), ("parity", 8, 1), ("secded", 8, 5),
                                             ("secded", 16, 6), ("secded", 32, 7), ("secded", 64, 8)])
def test_check_bit_counts_match_the_textbook(name, k, expected):
    codec = make_codec(name, k)
    assert codec.check_bits == expected
    assert codec.encode(_random_payload(4, k)).shape == (4, expected)


def test_overhead_ratios():
    assert overhead_ratio("secded", 8) == pytest.approx(0.625)
    assert overhead_ratio("secded", 64) == pytest.approx(0.125)
    assert overhead_ratio("parity", 16) == pytest.approx(0.0625)


@pytest.mark.parametrize("k", [8, 16])
def test_clean_words_decode_unchanged(k):
    codec = make_codec("secded", k)
    payload = _random_payload(64, k, seed=k)
    result = codec.decode(payload, codec.encode(payload))
    assert torch.equal(result.data, payload)
    assert not result.corrected.any() and not result.uncorrectable.any()


@pytest.mark.parametrize("k", [8, 16])
def test_secded_corrects_every_single_bit_error(k):
    codec: SecdedCodec = make_codec("secded", k)  # type: ignore[assignment]
    n = k + codec.check_bits  # payload + hamming + overall parity
    payload = _random_payload(1, k, seed=7).repeat(n, 1)
    check = codec.encode(payload)
    full = torch.cat([payload, check], dim=1)
    for position in range(n):  # one flipped bit per word, sweeping the entire codeword
        full[position, position] ^= 1
    result = codec.decode(full[:, :k], full[:, k:])
    assert result.corrected.all(), "every single-bit error must be reported as corrected (check bits included)"
    assert not result.uncorrectable.any()
    assert torch.equal(result.data, payload), "correction must restore the original payload"


@pytest.mark.parametrize("k", [8, 16])
def test_secded_detects_every_double_bit_error(k):
    codec: SecdedCodec = make_codec("secded", k)  # type: ignore[assignment]
    n = k + codec.check_bits
    pairs = list(itertools.combinations(range(n), 2))
    payload = _random_payload(1, k, seed=11).repeat(len(pairs), 1)
    check = codec.encode(payload)
    full = torch.cat([payload, check], dim=1)
    for row, (i, j) in enumerate(pairs):
        full[row, i] ^= 1
        full[row, j] ^= 1
    result = codec.decode(full[:, :k], full[:, k:])
    assert result.uncorrectable.all(), f"all {len(pairs)} double-bit patterns must be detected"
    assert not result.corrected.any(), "a double-bit error must never be 'corrected' into a third error"


def test_parity_detects_odd_errors_and_misses_even_ones():
    codec = make_codec("parity", 8)
    payload = _random_payload(8, 8, seed=3)
    check = codec.encode(payload)
    assert not codec.decode(payload, check).detected.any()

    single = payload.clone()
    single[:, 2] ^= 1
    assert codec.decode(single, check).uncorrectable.all()
    assert not codec.decode(single, check).corrected.any()  # parity cannot repair

    double = payload.clone()
    double[:, 2] ^= 1
    double[:, 5] ^= 1
    assert not codec.decode(double, check).detected.any()  # the classic parity blind spot


def test_no_codec_is_transparent():
    codec = make_codec("none", 4)
    payload = _random_payload(5, 4)
    result = codec.decode(payload, codec.encode(payload))
    assert torch.equal(result.data, payload) and not result.detected.any()


def test_triple_bit_errors_expose_miscorrection_and_silent_corruption():
    """SECDED promises nothing beyond two errors. This records what it actually does on all triples, because
    'corrected' flags that hide a wrong payload are precisely how silent data corruption arises in M3."""
    codec: SecdedCodec = make_codec("secded", 8)  # type: ignore[assignment]
    n = 8 + codec.check_bits
    triples = list(itertools.combinations(range(n), 3))
    payload = _random_payload(1, 8, seed=5).repeat(len(triples), 1)
    check = codec.encode(payload)
    full = torch.cat([payload, check], dim=1)
    for row, (i, j, m) in enumerate(triples):
        full[row, [i, j, m]] ^= 1
    result = codec.decode(full[:, :8], full[:, 8:])

    payload_wrong = ~(result.data == payload).all(dim=1)
    miscorrected = result.corrected & payload_wrong
    silent = payload_wrong & ~result.detected
    assert int(miscorrected.sum()) > 0, "some triples are mistaken for single-bit errors and 'repaired' wrongly"
    # Whatever the split, the code must never claim a clean decode of a payload it actually damaged without
    # flagging anything AND being counted as safe: those cases are the SDC the campaign must attribute.
    assert int(miscorrected.sum()) + int(silent.sum()) + int((result.uncorrectable & payload_wrong).sum()) \
        == int(payload_wrong.sum())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to compare devices")
@pytest.mark.parametrize("word_bits", [16, 64])
def test_encode_and_decode_are_bit_identical_on_cpu_and_gpu(word_bits):
    """The parity kernel runs in float32 because CUDA has no integer matmul; that must not change a single bit.

    Sums here are at most the codeword length, so float32 is exact -- but 'should be exact' is the kind of claim
    that silently stops being true, so it is measured on both devices instead.
    """
    codec: SecdedCodec = make_codec("secded", word_bits)  # type: ignore[assignment]
    payload = _random_payload(512, word_bits, seed=17)
    check = codec.encode(payload)
    faulty = payload.clone()
    faulty[torch.arange(512), torch.randint(0, word_bits, (512,), generator=torch.Generator().manual_seed(2))] ^= 1

    gpu = torch.device("cuda")
    check_gpu = make_codec("secded", word_bits).encode(payload.to(gpu))
    assert torch.equal(check, check_gpu.cpu()), "check bits differ between devices"

    cpu_result = codec.decode(faulty, check)
    gpu_result = make_codec("secded", word_bits).decode(faulty.to(gpu), check.to(gpu))
    assert torch.equal(cpu_result.data, gpu_result.data.cpu())
    assert torch.equal(cpu_result.corrected, gpu_result.corrected.cpu())
    assert torch.equal(cpu_result.uncorrectable, gpu_result.uncorrectable.cpu())
    assert torch.equal(cpu_result.data, payload), "every single-bit error must be repaired on both devices"
