"""Protected KV storage: check bits over the codes and over the quantization metadata.

The stored form is what M2 corrupts, so protection sits exactly there: at write time the bit stream of each
region is cut into words and encoded; at read time every word is decoded, single-bit errors are repaired, and
detected-uncorrectable words are handled by an explicit policy instead of being passed through silently.

Three outcomes are counted separately, because conflating them is how reliability papers overstate their case:
* **corrected** — the code repaired the word;
* **DUE** (detected, uncorrectable) — the code noticed and the erasure policy took over;
* **silent** — nothing was noticed, which only shows up as a wrong answer downstream.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from raq_kv.config.schema import ProtectionConfig
from raq_kv.ecc.bits import bits_to_codes, bits_to_float16, codes_to_bits, float16_to_bits, pack_words, unpack_words
from raq_kv.ecc.codecs import make_codec
from raq_kv.kv.snapshot import KVSnapshot
from raq_kv.quantization.apply import QuantizedKV
from raq_kv.quantization.base import QuantizedTensor
from raq_kv.quantization.uniform import dequantize


@dataclass
class RegionCode:
    """Check bits for one (layer, part, target) region, plus what is needed to rebuild its values."""

    codec_name: str
    word_bits: int
    check: torch.Tensor  # [n_words, check_bits]
    pad: int
    shape: torch.Size
    value_bits: int  # bits per stored value (code width, or 16 for float16 metadata)

    @property
    def ecc_bits(self) -> int:
        return int(self.check.numel())


@dataclass
class ProtectionStats:
    corrected: int = 0
    uncorrectable: int = 0
    words_read: int = 0
    regions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"corrected_words": self.corrected, "due_words": self.uncorrectable,
                "words_read": self.words_read, **{f"due_{k}": v for k, v in self.regions.items()}}


class ProtectedKV:
    """Adds check bits to a :class:`QuantizedKV` and reads it back through the decoder.

    ``data_layers=None`` protects every layer's codes; a subset expresses a layer-selective policy. Metadata
    (scales and zero points) is protected independently, which is the policy M2 predicts should matter most.
    """

    def __init__(
        self,
        qkv: QuantizedKV,
        data_codec: str = "none",
        data_word_bits: int = 64,
        metadata_codec: str = "none",
        metadata_word_bits: int = 16,
        data_layers: tuple[int, ...] | None = None,
        due_policy: str = "erasure_mid",
    ) -> None:
        self.qkv = qkv
        self.due_policy = due_policy
        self.codes: dict[tuple[int, str, str], RegionCode] = {}
        for layer in range(qkv.num_layers):
            for part in ("keys", "values"):
                stored = qkv.part(layer, part)
                if not isinstance(stored, QuantizedTensor):
                    continue
                if data_codec != "none" and (data_layers is None or layer in data_layers):
                    self.codes[(layer, part, "code")] = self._encode_codes(stored, data_codec, data_word_bits)
                if metadata_codec != "none":
                    for target, tensor in (("scale", stored.scales), ("zero", stored.zeros)):
                        if tensor is not None:
                            self.codes[(layer, part, target)] = self._encode_float16(
                                tensor, metadata_codec, metadata_word_bits)

    @staticmethod
    def _encode_codes(stored: QuantizedTensor, codec_name: str, word_bits: int) -> RegionCode:
        stream = codes_to_bits(stored.codes, stored.bits).reshape(-1)
        words, pad = pack_words(stream, word_bits)
        codec = make_codec(codec_name, word_bits)
        return RegionCode(codec_name, word_bits, codec.encode(words), pad, stored.codes.shape, stored.bits)

    @staticmethod
    def _encode_float16(tensor: torch.Tensor, codec_name: str, word_bits: int) -> RegionCode:
        stream = float16_to_bits(tensor).reshape(-1)
        words, pad = pack_words(stream, word_bits)
        codec = make_codec(codec_name, word_bits)
        return RegionCode(codec_name, word_bits, codec.encode(words), pad, tensor.shape, 16)

    @property
    def ecc_bits(self) -> int:
        return sum(region.ecc_bits for region in self.codes.values())

    def _decode_region(self, key: tuple[int, str, str], stream: torch.Tensor, stats: ProtectionStats) -> torch.Tensor:
        region = self.codes.get(key)
        if region is None:
            return stream
        words, pad = pack_words(stream, region.word_bits)
        result = make_codec(region.codec_name, region.word_bits).decode(words, region.check)
        data = result.data
        if bool(result.uncorrectable.any()):
            # An erasure, not a guess: the words the code flagged are replaced by the mid-code, which is the
            # value that minimises the damage without pretending the data survived.
            if self.due_policy == "erasure_mid":
                mid = (1 << (region.value_bits - 1)) - 1 if region.value_bits <= 8 else 0
                fill = codes_to_bits(torch.full((1,), mid, dtype=torch.uint8), region.value_bits).reshape(-1)
                repeat = (region.word_bits + fill.numel() - 1) // fill.numel()
                data = data.clone()
                data[result.uncorrectable] = fill.repeat(repeat)[: region.word_bits].to(data.device)
            elif self.due_policy != "pass_through":
                raise ValueError(f"unknown DUE policy {self.due_policy!r}")
        stats.corrected += int(result.corrected.sum())
        stats.uncorrectable += int(result.uncorrectable.sum())
        stats.words_read += int(words.shape[0])
        if int(result.uncorrectable.sum()):
            label = f"{key[1]}_{key[2]}"
            stats.regions[label] = stats.regions.get(label, 0) + int(result.uncorrectable.sum())
        return unpack_words(data, pad)

    def read_part(self, layer: int, part: str, stats: ProtectionStats) -> torch.Tensor:
        """Decode one layer-part through its protection and return the dequantized tensor."""
        stored = self.qkv.part(layer, part)
        if not isinstance(stored, QuantizedTensor):
            return stored
        codes = stored.codes
        if (layer, part, "code") in self.codes:
            stream = codes_to_bits(codes, stored.bits).reshape(-1)
            repaired = self._decode_region((layer, part, "code"), stream, stats)
            codes = bits_to_codes(repaired.reshape(-1, stored.bits)).reshape(stored.codes.shape)
        scales, zeros = stored.scales, stored.zeros
        for target in ("scale", "zero"):
            key = (layer, part, target)
            tensor = scales if target == "scale" else zeros
            if key in self.codes and tensor is not None:
                repaired = self._decode_region(key, float16_to_bits(tensor).reshape(-1), stats)
                fixed = bits_to_float16(repaired.reshape(-1, 16)).reshape(tensor.shape)
                if target == "scale":
                    scales = fixed
                else:
                    zeros = fixed
        # type(stored) keeps the symmetric subclass, whose dequantization differs from the asymmetric one.
        repaired_tensor = type(stored)(codes, scales, zeros, stored.bits, stored.axis, stored.group_size,
                                       stored.orig_dtype)
        return dequantize(repaired_tensor)

    def inject_check(self, site: Any) -> dict[str, Any]:
        """Flip one check bit in place, reversibly.

        A corrupted check bit is a real failure mode, not a technicality: the codec then computes a syndrome for
        an error that never happened and "repairs" intact data, which is a miscorrection.
        """
        key = (site.layer, site.part, site.check_region)
        region = self.codes.get(key)
        if region is None:
            raise ValueError(f"no check bits stored for {key}; this site cannot exist in this arm")
        word, bit = site.index[0], site.bit
        before = int(region.check[word, bit])
        region.check[word, bit] = 1 - before
        return {"stored_before": before, "stored_after": 1 - before, "relative_change": 1.0}

    @contextmanager
    def faulted_read(self, sites: list[Any], base: KVSnapshot):
        """Inject a burst, re-read only the affected layer-parts *through the codec*, then restore.

        Without this the campaign would compare protected storage against an unprotected read path; the whole
        point is that the decoder sits between the corrupted bits and the model. Sites aimed at check bits are
        applied to this object's own check tensors, so an arm is exposed on all of the storage it declares.
        """
        from raq_kv.faults.injector import inject  # local import keeps ecc independent of the fault package

        def apply(site: Any) -> dict[str, Any]:
            return self.inject_check(site) if site.is_check else inject(self.qkv, site)

        infos = [apply(site) for site in sites]
        touched = {(s.layer, s.part) for s in sites}
        saved: dict[tuple[int, str], torch.Tensor] = {}
        stats = ProtectionStats()
        try:
            for layer, part in touched:
                tensors = base.keys if part == "keys" else base.values
                saved[(layer, part)] = tensors[layer]
                tensors[layer] = self.read_part(layer, part, stats)
            yield base, {"n_flips": len(sites), **stats.to_dict(),
                         "max_relative_change": max(i["relative_change"] for i in infos)}
        finally:
            for (layer, part), tensor in saved.items():
                (base.keys if part == "keys" else base.values)[layer] = tensor
            for site in sites:
                apply(site)  # flipping the same bit again restores it, check bits included

    def read_or_raise(self) -> KVSnapshot:
        """Fault-free read: nothing may be corrected or flagged, so a non-zero count is a bug, not a result."""
        snapshot, stats = self.read()
        if stats.corrected or stats.uncorrectable:
            raise RuntimeError(f"protected read of an uncorrupted cache reported {stats.to_dict()}")
        return snapshot

    def read(self) -> tuple[KVSnapshot, ProtectionStats]:
        stats = ProtectionStats()
        keys = [self.read_part(layer, "keys", stats) for layer in range(self.qkv.num_layers)]
        values = [self.read_part(layer, "values", stats) for layer in range(self.qkv.num_layers)]
        return KVSnapshot(keys=keys, values=values, positions=self.qkv.positions,
                          next_position=self.qkv.next_position), stats


def build_protection(qkv: QuantizedKV, cfg: ProtectionConfig | None) -> ProtectedKV | None:
    """Config -> guard, or ``None`` when the arm stores the quantized cache unprotected."""
    if cfg is None:
        return None
    return ProtectedKV(qkv, data_codec=cfg.data_codec, data_word_bits=cfg.data_word_bits,
                       metadata_codec=cfg.metadata_codec, metadata_word_bits=cfg.metadata_word_bits,
                       data_layers=cfg.data_layers, due_policy=cfg.due_policy)
