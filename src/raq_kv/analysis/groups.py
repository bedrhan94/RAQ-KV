"""Attention mass folded into `group_token` quantization groups, and the ranking over them.

Shared because two callers need the identical fold and this project has already had to repair three cases of one
quantity computed in two places (the budget predictor against the real encoders, the span selector across two
tasks, the Wilson interval across two report scripts). `scripts/measure_group_attention.py` measures the
*distribution* of this mass; `ProtectionArm`'s group-targeted mode needs the *rank order* of the same quantity to
choose which group a fault lands in. If those two disagreed about what "group 3" means, the conditional
probability the second one exists to measure would not be about the thing the first one described.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def group_shares(scores: torch.Tensor, group_size: int) -> np.ndarray:
    """Attention mass per `group_token` group, normalised to sum to 1.

    Scores are non-negative attention weights summed over heads and layers, so they are already a mass; the only
    work here is folding them into the groups a scale actually covers and renormalising so two samples of
    different length are comparable. The trailing group is short whenever the length is not a multiple of the
    group size, which is exactly how the quantizer pads it.
    """
    v = scores.detach().float().reshape(-1).clamp_min(0.0).cpu().numpy()
    if v.sum() <= 0:
        return np.array([])
    n_groups = -(-len(v) // group_size)
    padded = np.zeros(n_groups * group_size, dtype=np.float64)
    padded[: len(v)] = v
    per_group = padded.reshape(n_groups, group_size).sum(axis=1)
    return per_group / per_group.sum()


def concentration(shares: np.ndarray) -> dict[str, float]:
    """Four views of the same distribution, because no single number is obviously the right one.

    `effective_groups` is exp(Shannon entropy) -- the number of equally-weighted groups that would carry the same
    entropy, which is the quantity a "fraction of groups that matter" story is about. The top-k shares and the
    90%-mass fraction are reported beside it so a reader can see whether they agree rather than trusting the
    entropy alone.
    """
    if shares.size == 0:
        return {}
    p = shares[shares > 0]
    entropy = float(-(p * np.log(p)).sum())
    order = np.sort(shares)[::-1]
    cumulative = np.cumsum(order)
    n90 = int(np.searchsorted(cumulative, 0.90) + 1)
    # WHERE the extremes sit, not just how big they are. D61 targets faults by attention rank, and if the
    # top-ranked group is always group 0 -- which the attention-sink literature and this probe's first two samples
    # both suggest -- then a rank-conditioned measurement is confounded with position and cannot be read as
    # "attention-mediated". Reporting the index is what makes that confound visible instead of assumed away.
    by_mass = np.argsort(-shares, kind="stable")
    last = max(1, shares.size - 1)
    # Measured at 1.000 over 80 samples on both families: rank 0 IS group 0, always. So a rank-0 cell is a
    # position cell, and the ranks below it have to be checked the same way before any of them is called
    # attention-mediated -- which is why each rank's index and relative position is reported, not just rank 0's.
    per_rank = {}
    for r in (0, 1, 2):
        if r < shares.size:
            per_rank |= {f"rank{r}_index": float(by_mass[r]),
                         f"rank{r}_is_first_group": float(by_mass[r] == 0),
                         f"rank{r}_is_last_group": float(by_mass[r] == shares.size - 1),
                         f"rank{r}_position": float(by_mass[r] / last),
                         f"rank{r}_share": float(shares[by_mass[r]])}
    return {
        "n_groups": float(shares.size),
        **per_rank,
        "top1_index": float(by_mass[0]),
        "top1_is_first_group": float(by_mass[0] == 0),
        "top1_position": float(by_mass[0] / last),
        "min_index": float(by_mass[-1]),
        "min_is_last_group": float(by_mass[-1] == shares.size - 1),
        "min_position": float(by_mass[-1] / last),
        "min_share": float(shares[by_mass[-1]]),
        "effective_groups": float(math.exp(entropy)),
        "effective_fraction": float(math.exp(entropy) / shares.size),
        "top1_share": float(order[0]),
        "top5_share": float(order[:5].sum()),
        "groups_for_90pc": float(n90),
        "fraction_for_90pc": float(n90 / shares.size),
    }


def group_at_rank(shares: np.ndarray, rank: int) -> int:
    """Index of the group holding the ``rank``-th largest attention mass, 0 = the largest.

    Ties are broken by group index via a stable sort, so the choice is reproducible across runs rather than
    dependent on numpy's partitioning: a fault campaign that picked a different group on a replay would break the
    replay guarantee M2's criterion 3 pins for every other plan.

    A rank past the end wraps to the **last** group rather than raising, because samples differ in length and a
    campaign asking for rank 15 would otherwise die on the first 512-token sample. The caller records the realised
    rank, so a wrapped cell is visible in the analysis instead of silently pretending to be rank 15.
    """
    if shares.size == 0:
        raise ValueError("no attention mass to rank; the scorer returned an all-zero or empty score vector")
    order = np.argsort(-shares, kind="stable")
    return int(order[min(rank, shares.size - 1)])
