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

import math
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
    out_of_range: int = 0
    regions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"corrected_words": self.corrected, "due_words": self.uncorrectable,
                "words_read": self.words_read, "out_of_range_values": self.out_of_range,
                **{f"due_{k}": v for k, v in self.regions.items()}}


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
        metadata_guard: str | None = None,
    ) -> None:
        self.qkv = qkv
        self.due_policy = due_policy
        self.metadata_guard = metadata_guard
        self.codes: dict[tuple[int, str, str], RegionCode] = {}
        # Plausibility bounds, recorded from the clean tensor exactly as a real system would at quantization time.
        # Stored as a float16 tensor, not a Python pair, because these bits are storage like any other and the
        # fault model must be able to land on them (D23's principle applied to the guard itself).
        self.bounds: dict[tuple[int, str, str], torch.Tensor] = {}
        self.guard_invalid = 0  # regions whose own bounds a fault made unusable
        for layer in range(qkv.num_layers):
            for part in ("keys", "values"):
                stored = qkv.part(layer, part)
                if not isinstance(stored, QuantizedTensor):
                    continue
                if data_codec != "none" and (data_layers is None or layer in data_layers):
                    self.codes[(layer, part, "code")] = self._encode_codes(stored, data_codec, data_word_bits)
                for target, tensor in (("scale", stored.scales), ("zero", stored.zeros)):
                    if tensor is None:
                        continue
                    if metadata_codec != "none":
                        self.codes[(layer, part, target)] = self._encode_float16(
                            tensor, metadata_codec, metadata_word_bits)
                    if metadata_guard == "range":
                        self.bounds[(layer, part, target)] = torch.stack(
                            [tensor.min(), tensor.max()]).to(torch.float16)

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
        """Every bit the protection itself costs: check bits, plus the guard's stored bounds."""
        return sum(region.ecc_bits for region in self.codes.values()) + self.guard_bits

    @property
    def guard_bits(self) -> int:
        """Two float16 bounds per guarded region. Tiny, but counted: an uncounted bit is a free lunch."""
        return len(self.bounds) * 2 * 16

    def _apply_guard(self, key: tuple[int, str, str], tensor: torch.Tensor,
                     stats: ProtectionStats) -> torch.Tensor:
        """Replace implausible metadata with the nearest stored bound.

        This is detection plus a bounded repair, not correction: a scale that a fault multiplied by 65536 is
        clamped back to the largest value the clean tensor held, which is wrong but wrong by a factor the model
        can absorb. It cannot see a flip that leaves the value inside the observed range, which is most mantissa
        bits — and those are the flips that move a scale by at most 36% (D27).
        """
        low, high = (float(v) for v in self.bounds[key].to(torch.float32))
        # The bounds are stored bits too, so a fault can reach them. Two comparisons the guard can make about
        # itself for nothing -- ordered and finite -- catch a corrupted bound; when they fail the guard cannot be
        # trusted for this region and steps aside rather than clamping against a wrong limit.
        if not (low <= high) or not (math.isfinite(low) and math.isfinite(high)):
            self.guard_invalid += 1
            return tensor
        bad = (tensor < low) | (tensor > high) | ~torch.isfinite(tensor)
        n_bad = int(bad.sum())
        if not n_bad:
            return tensor
        stats.out_of_range += n_bad
        fixed = torch.where(torch.isfinite(tensor), tensor, torch.full_like(tensor, low))
        return fixed.clamp(min=low, max=high)

    @staticmethod
    def _erasure_fill(region: RegionCode) -> torch.Tensor:
        """The bit pattern an erased word is replaced with, one repeat unit wide.

        Codes get the mid-code, which is the least-wrong integer when the real one is unknown. **Metadata is
        float16, not an 8-bit code**, so it gets float16 zero: an erased scale collapses its group to the zero
        point (the group's offset) and an erased zero point centres the group at 0. Both are visibly neutral
        rather than a plausible-looking wrong number, which is the point of an erasure.

        This path was unreachable until metadata DUEs started occurring at higher fault rates: 16 is not a legal
        code width, so the previous code raised instead of erasing (found by the BER 1e-5 campaign, D27).
        """
        if region.value_bits == 16:
            return float16_to_bits(torch.zeros(1, dtype=torch.float16)).reshape(-1)
        if region.value_bits > 8:
            raise ValueError(f"no erasure value defined for {region.value_bits}-bit stored values")
        mid = (1 << (region.value_bits - 1)) - 1
        return codes_to_bits(torch.full((1,), mid, dtype=torch.uint8), region.value_bits).reshape(-1)

    def _decode_region(self, key: tuple[int, str, str], stream: torch.Tensor, stats: ProtectionStats) -> torch.Tensor:
        region = self.codes.get(key)
        if region is None:
            return stream
        words, pad = pack_words(stream, region.word_bits)
        result = make_codec(region.codec_name, region.word_bits).decode(words, region.check)
        data = result.data
        if bool(result.uncorrectable.any()):
            # An erasure, not a guess: the words the code flagged are replaced by a neutral value, so the read
            # does not pretend the data survived. What "neutral" means depends on what the region stores.
            if self.due_policy == "erasure_mid":
                fill = self._erasure_fill(region)
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
            if tensor is None:
                continue
            if key in self.codes:
                repaired = self._decode_region(key, float16_to_bits(tensor).reshape(-1), stats)
                tensor = bits_to_float16(repaired.reshape(-1, 16)).reshape(tensor.shape)
            if key in self.bounds:
                # After the codec, so the guard only sees what the code could not repair.
                tensor = self._apply_guard(key, tensor, stats)
            if target == "scale":
                scales = tensor
            else:
                zeros = tensor
        # type(stored) keeps the symmetric subclass, whose dequantization differs from the asymmetric one.
        repaired_tensor = type(stored)(codes, scales, zeros, stored.bits, stored.axis, stored.group_size,
                                       stored.orig_dtype)
        return dequantize(repaired_tensor)

    def inject_bound(self, site: Any) -> dict[str, Any]:
        """Flip one bit of a stored plausibility bound, reversibly.

        A corrupted bound is the guard's own single point of failure: inflate a region's stored maximum and the
        guard stops catching explosions there for the rest of the read. Cheap as the bounds are, they are part of
        what the arm stores, so they are part of what faults can reach.
        """
        key = (site.layer, site.part, site.protection_region)
        bound = self.bounds.get(key)
        if bound is None:
            raise ValueError(f"no bounds stored for {key}; this site cannot exist in this arm")
        view = bound.view(torch.int16)
        element = site.index[0]
        before = float(bound[element])
        view[element] ^= torch.tensor(1 << site.bit if site.bit < 15 else -32768, dtype=torch.int16)
        return {"stored_before": before, "stored_after": float(bound[element]), "relative_change": 1.0}

    def inject_check(self, site: Any) -> dict[str, Any]:
        """Flip one check bit in place, reversibly.

        A corrupted check bit is a real failure mode, not a technicality: the codec then computes a syndrome for
        an error that never happened and "repairs" intact data, which is a miscorrection.
        """
        key = (site.layer, site.part, site.protection_region)
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
        # Local imports keep the ecc package independent of the fault package.
        from raq_kv.faults.injector import PER_SITE_LIMIT, inject, inject_many

        check_sites = [s for s in sites if s.is_check]
        bound_sites = [s for s in sites if s.is_bound]
        store_sites = [s for s in sites if not (s.is_check or s.is_bound)]
        fast = len(store_sites) > PER_SITE_LIMIT

        def apply() -> dict[str, Any]:
            """Flip everything once; returns the per-site summary when it is cheap enough to measure."""
            for site in check_sites:
                self.inject_check(site)
            for site in bound_sites:
                self.inject_bound(site)
            if fast:
                inject_many(self.qkv, store_sites)
                return {"max_relative_change": None, "relative_change_measured": False}
            infos = [inject(self.qkv, s) for s in store_sites]
            return {"max_relative_change": max((i["relative_change"] for i in infos), default=0.0),
                    "relative_change_measured": True}

        summary = apply()
        touched = {(s.layer, s.part) for s in sites}
        saved: dict[tuple[int, str], torch.Tensor] = {}
        stats = ProtectionStats()
        self.guard_invalid = 0
        try:
            for layer, part in touched:
                tensors = base.keys if part == "keys" else base.values
                saved[(layer, part)] = tensors[layer]
                tensors[layer] = self.read_part(layer, part, stats)
            yield base, {"n_flips": len(sites), **stats.to_dict(), **summary,
                         "guard_invalid_regions": self.guard_invalid}
        finally:
            for (layer, part), tensor in saved.items():
                (base.keys if part == "keys" else base.values)[layer] = tensor
            apply()  # flipping the same bits again restores them, check bits included

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
                       data_layers=cfg.data_layers, due_policy=cfg.due_policy,
                       metadata_guard=cfg.metadata_guard)
