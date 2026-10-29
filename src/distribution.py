"""Turn a predicted strikeout mean into probabilities like "chance of 6+ K".

The models predict one number per start. Strikeouts are a count, so a full
distribution around that number answers questions like P(6 or more). Three
ways to get one are compared in ``src/compare_features.py``:

* ``poisson``: Poisson with the predicted mean (variance = mean)
* ``negbin``: negative binomial with the predicted mean and one dispersion
  value fit on out-of-sample predictions (variance = mean + alpha * mean^2)
* ``empirical``: the predicted mean plus the model's actual misses on an
  earlier season, rounded to whole strikeouts

All need only numpy.
"""

from __future__ import annotations

import numpy as np

MAX_K = 25
# The method the app and predict_upcoming use (picked with src/compare_features.py).
PROB_METHOD = "negbin"
THRESHOLDS = list(range(3, 10))
ALPHA_GRID = np.r_[0.0, np.logspace(-3, 0, 46)]


def negbin_pmf(mu, alpha: float, max_k: int = MAX_K) -> np.ndarray:
    """P(K = k) for k = 0..max_k, one row per mean (Poisson when alpha is 0)."""
    mu = np.clip(np.asarray(mu, float), 1e-6, None)[:, None]
    pmf = np.empty((len(mu), max_k + 1))
    if alpha <= 0:
        pmf[:, :1] = np.exp(-mu)
        for k in range(max_k):
            pmf[:, k + 1] = pmf[:, k] * mu[:, 0] / (k + 1)
    else:
        r = 1.0 / alpha
        q = mu[:, 0] / (r + mu[:, 0])
        pmf[:, 0] = (r / (r + mu[:, 0])) ** r
        for k in range(max_k):
            pmf[:, k + 1] = pmf[:, k] * (k + r) / (k + 1) * q
    pmf[:, -1] += np.clip(1 - pmf.sum(axis=1), 0, None)  # fold the far tail into max_k
    return pmf


def empirical_pmf(mu, residuals, max_k: int = MAX_K) -> np.ndarray:
    """P(K = k) from mean + past misses, binned to whole strikeouts."""
    mu = np.asarray(mu, float)[:, None]
    res = np.sort(np.asarray(residuals, float))
    edges = np.arange(max_k + 2) - 0.5  # bin k covers [k - 0.5, k + 0.5)
    cdf = np.searchsorted(res, edges[None, :] - mu, side="left") / len(res)
    cdf[:, 0], cdf[:, -1] = 0.0, 1.0  # everything below 0.5 is 0 K, everything above is max_k
    pmf = np.diff(cdf, axis=1)
    pmf = (pmf + 1e-4) / (1 + 1e-4 * pmf.shape[1])  # no exact zeros, for log loss
    return pmf


class Distribution:
    """A fitted way of turning predicted means into strikeout probabilities."""

    def __init__(self, method: str):
        self.method = method
        self.alpha = 0.0
        self.residuals = None

    def fit(self, y, mu) -> Distribution:
        """Fit on actual strikeouts ``y`` and out-of-sample predicted means ``mu``."""
        y = np.asarray(y, int)
        if self.method == "negbin":
            scores = [log_loss(y, negbin_pmf(mu, a)) for a in ALPHA_GRID]
            self.alpha = float(ALPHA_GRID[int(np.argmin(scores))])
        elif self.method == "empirical":
            self.residuals = y - np.asarray(mu, float)
        elif self.method != "poisson":
            raise ValueError(self.method)
        return self

    def pmf(self, mu) -> np.ndarray:
        if self.method == "empirical":
            return empirical_pmf(mu, self.residuals)
        return negbin_pmf(mu, self.alpha)


def prob_at_least(pmf: np.ndarray, t: int) -> np.ndarray:
    return pmf[:, t:].sum(axis=1)


def log_loss(y, pmf: np.ndarray) -> float:
    """Mean negative log probability of the actual strikeout count."""
    y = np.clip(np.asarray(y, int), 0, pmf.shape[1] - 1)
    return float(-np.log(np.clip(pmf[np.arange(len(y)), y], 1e-9, None)).mean())


def brier(y, pmf: np.ndarray, thresholds=THRESHOLDS) -> float:
    """Mean Brier score of P(K >= t) over the thresholds."""
    y = np.asarray(y, int)
    return float(np.mean([((prob_at_least(pmf, t) - (y >= t)) ** 2).mean() for t in thresholds]))


def calibration(y, pmf: np.ndarray, t: int = 6, bins: int = 10) -> list[tuple[float, float, int]]:
    """(mean predicted P(K >= t), observed share with K >= t, starts) per probability decile."""
    p = prob_at_least(pmf, t)
    hit = (np.asarray(y, int) >= t).astype(float)
    order = np.argsort(p)
    return [(float(p[i].mean()), float(hit[i].mean()), len(i)) for i in np.array_split(order, bins)]


def calibration_error(y, pmf: np.ndarray, t: int = 6, bins: int = 10) -> float:
    """Start-weighted mean gap between predicted and observed P(K >= t) across deciles."""
    rows = calibration(y, pmf, t, bins)
    n = sum(r[2] for r in rows)
    return float(sum(abs(p - o) * k for p, o, k in rows) / n)
