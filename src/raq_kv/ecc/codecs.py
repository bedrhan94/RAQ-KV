"""Codecs: no protection, even parity, and single-error-correcting double-error-detecting Hamming (SECDED).

All three share one interface so a protection policy can swap them per region, and all three operate on bit
matrices ``[n_words, k]`` so the same logic can be re-expressed in RTL. Decoding reports what happened per word:

* ``corrected``     — a single-bit error was located and repaired;
* ``uncorrectable`` — an error was detected but cannot be repaired (a detected uncorrectable error, DUE);
* neither          — the word came back clean, which may mean no fault *or* a fault the code cannot see (that
  case is what silent data corruption means, and it is why the campaign compares outputs, not flags).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class DecodeResult:
    data: torch.Tensor  # [n_words, k] payload after correction
    corrected: torch.Tensor  # bool [n_words]
    uncorrectable: torch.Tensor  # bool [n_words]

    @property
    def detected(self) -> torch.Tensor:
        return self.corrected | self.uncorrectable

    def summary(self) -> dict[str, int]:
        return {"words": int(self.data.shape[0]), "corrected": int(self.corrected.sum()),
                "uncorrectable": int(self.uncorrectable.sum())}


class Codec(Protocol):
    name: str
    data_bits: int
    check_bits: int

    def encode(self, payload: torch.Tensor) -> torch.Tensor: ...
    def decode(self, payload: torch.Tensor, check: torch.Tensor) -> DecodeResult: ...


class NoCodec:
    """No protection: zero check bits, nothing detected. The reference arm."""

    name = "none"

    def __init__(self, data_bits: int) -> None:
        self.data_bits = data_bits
        self.check_bits = 0

    def encode(self, payload: torch.Tensor) -> torch.Tensor:
        return payload.new_zeros((payload.shape[0], 0))

    def decode(self, payload: torch.Tensor, check: torch.Tensor) -> DecodeResult:
        flags = torch.zeros(payload.shape[0], dtype=torch.bool, device=payload.device)
        return DecodeResult(payload, flags, flags.clone())


class ParityCodec:
    """One even-parity bit per word: detects any odd number of bit errors, corrects nothing."""

    name = "parity"

    def __init__(self, data_bits: int) -> None:
        self.data_bits = data_bits
        self.check_bits = 1

    def encode(self, payload: torch.Tensor) -> torch.Tensor:
        return (payload.to(torch.int32).sum(dim=-1) % 2).to(torch.uint8).unsqueeze(-1)

    def decode(self, payload: torch.Tensor, check: torch.Tensor) -> DecodeResult:
        recomputed = (payload.to(torch.int32).sum(dim=-1) % 2).to(torch.uint8)
        mismatch = recomputed != check[:, 0]
        no_correction = torch.zeros_like(mismatch)
        return DecodeResult(payload, no_correction, mismatch)


def _hamming_layout(data_bits: int) -> tuple[int, torch.Tensor]:
    """Number of Hamming parity bits and the parity-check matrix over the interleaved codeword.

    Codeword positions are 1-based; positions that are powers of two hold parity bits, the rest hold data. The
    check matrix column for position p is the binary expansion of p, which is what makes the syndrome equal the
    index of a single-bit error.
    """
    r = 1
    while (1 << r) < data_bits + r + 1:
        r += 1
    n = data_bits + r
    positions = torch.arange(1, n + 1, dtype=torch.int32)
    h = torch.stack([(positions >> i) & 1 for i in range(r)]).to(torch.uint8)  # [r, n]
    return r, h


class SecdedCodec:
    """Extended Hamming: corrects any single-bit error, detects any double-bit error.

    Check bits are the ``r`` Hamming parities plus one overall parity over the whole codeword, so
    ``check_bits = r + 1`` (8 data bits -> 5 check bits, 16 -> 6, 32 -> 7, 64 -> 8).
    """

    name = "secded"

    def __init__(self, data_bits: int) -> None:
        self.data_bits = data_bits
        self._r, self._h = _hamming_layout(data_bits)
        self.check_bits = self._r + 1
        n = data_bits + self._r
        self._parity_pos = torch.tensor([(1 << i) - 1 for i in range(self._r)], dtype=torch.long)
        mask = torch.ones(n, dtype=torch.bool)
        mask[self._parity_pos] = False
        self._data_pos = torch.nonzero(mask, as_tuple=False).squeeze(-1)

    def _to(self, device: torch.device) -> None:
        if self._h.device != device:
            self._h = self._h.to(device)
            self._parity_pos = self._parity_pos.to(device)
            self._data_pos = self._data_pos.to(device)

    def _codeword(self, payload: torch.Tensor, hamming: torch.Tensor | None = None) -> torch.Tensor:
        words = payload.shape[0]
        n = self.data_bits + self._r
        code = payload.new_zeros((words, n))
        code[:, self._data_pos] = payload
        if hamming is not None:
            code[:, self._parity_pos] = hamming
        return code

    def encode(self, payload: torch.Tensor) -> torch.Tensor:
        self._to(payload.device)
        code = self._codeword(payload)
        # Each Hamming parity is the XOR of the data bits its check row covers.
        hamming = (code.to(torch.int32) @ self._h.to(torch.int32).T) % 2
        code[:, self._parity_pos] = hamming.to(torch.uint8)
        overall = (code.to(torch.int32).sum(dim=-1) % 2).to(torch.uint8).unsqueeze(-1)
        return torch.cat([hamming.to(torch.uint8), overall], dim=-1)

    def decode(self, payload: torch.Tensor, check: torch.Tensor) -> DecodeResult:
        self._to(payload.device)
        hamming, overall = check[:, : self._r], check[:, self._r]
        code = self._codeword(payload, hamming)
        syndrome_bits = (code.to(torch.int32) @ self._h.to(torch.int32).T) % 2
        weights = (1 << torch.arange(self._r, device=payload.device, dtype=torch.int32))
        syndrome = (syndrome_bits * weights).sum(dim=-1)  # 0 = no detected Hamming error, else 1-based position
        overall_mismatch = ((code.to(torch.int32).sum(dim=-1) % 2).to(torch.uint8) != overall)

        # A syndrome beyond the codeword cannot name a bit position; that only happens with 3+ errors, which
        # this code makes no promise about, so it is reported as detected-uncorrectable rather than "corrected"
        # at an invalid index.
        in_range = syndrome <= code.shape[1]
        # syndrome == 0 with a parity mismatch: the overall parity bit itself flipped. SECDED corrects errors in
        # its own check bits too, and the payload needs no change, so it counts as corrected.
        parity_bit_only = (syndrome == 0) & overall_mismatch
        single = ((syndrome != 0) & overall_mismatch & in_range) | parity_bit_only
        double = (syndrome != 0) & (~overall_mismatch | ~in_range)

        fixed = code.clone()
        repairable = single & (syndrome != 0)
        if bool(repairable.any()):
            rows = torch.nonzero(repairable, as_tuple=False).squeeze(-1)
            cols = (syndrome[rows] - 1).to(torch.long)
            fixed[rows, cols] ^= 1
        return DecodeResult(fixed[:, self._data_pos], single, double)


def make_codec(name: str, data_bits: int) -> Codec:
    if name == "none":
        return NoCodec(data_bits)
    if name == "parity":
        return ParityCodec(data_bits)
    if name == "secded":
        return SecdedCodec(data_bits)
    raise ValueError(f"unknown codec {name!r}")


def overhead_ratio(name: str, data_bits: int) -> float:
    codec = make_codec(name, data_bits)
    return codec.check_bits / codec.data_bits
