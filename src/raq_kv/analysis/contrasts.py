"""The paired two-arm contrast, in one place.

D31's rule is that a difference needs p < 0.05 **and** a bootstrap CI excluding zero, and this is the only function
in the project allowed to decide that. It lived in `scripts/report_fault_retention.py` until D61 needed the same
contrast on a different set of arms; copying it would have been the fourth instance of one quantity computed twice
in this project, after the budget predictor, the span selector and the Wilson interval -- two of which actually
drifted and had to be repaired.

The bootstrap is keyed to the *pair of arm names* rather than drawn from a shared generator. That is not a style
choice: with one shared generator, asking for three contrasts in one invocation gave the third a different draw
than asking for it alone, so the same comparison had two intervals depending on the command line. A published
interval that depends on invocation order is not a reproducible number, and one had already been committed before
this was found.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy import stats

BOOTSTRAP = 10_000


def contrast_rng(x_name: str, y_name: str, seed: int) -> np.random.Generator:
    """A generator keyed by the pair, so a contrast's interval is a function of the data and ``seed`` alone."""
    digest = hashlib.sha256(f"{seed}|{x_name}|{y_name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def contrast(x_name: str, y_name: str, xv: np.ndarray, yv: np.ndarray, seed: int,
             n_bootstrap: int = BOOTSTRAP) -> dict:
    """Paired contrast of two arms over the samples they share, y against x.

    Pairing is at the **sample** level and never over pooled trials: two arms' trials are not exchangeable, and
    pooling them would treat 8 draws on one sample as 8 independent observations. ``discordant`` is reported
    because Wilcoxon on an all-zero difference vector is undefined -- that case returns p = 1 rather than raising,
    which is the honest reading of "no sample moved" and is also how a floor-to-floor comparison arrives here.
    """
    d = yv - xv
    discordant = int((d != 0).sum())
    p = 1.0 if discordant == 0 else float(stats.wilcoxon(xv, yv).pvalue)
    rng = contrast_rng(x_name, y_name, seed)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_bootstrap)])
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    return {"x": x_name, "y": y_name, "x_mean": float(xv.mean()), "y_mean": float(yv.mean()),
            "difference": float(d.mean()), "ci": (lo, hi), "p": p, "discordant": discordant,
            "n_pairs": int(len(d)),
            # D31's rule, both halves. A difference meeting one and not the other is a direction, not a result.
            "meets_rule": bool(p < 0.05 and lo * hi > 0)}
