"""Categorical verification metrics for nowcast evaluation (plan §9.3).

Standard set used in radar nowcasting literature:

- POD: Probability of Detection (recall)
- FAR: False Alarm Ratio
- CSI: Critical Success Index — the workhorse single number
- HSS: Heidke Skill Score
- ETS: Equitable Threat Score (Gilbert Skill Score)
- Frequency bias
- F1

For the probabilistic / spatial extensions (Brier, reliability, FSS, ROC)
see Phase 4. ETA error stats are here as a small companion module.

References:
- Roberts & Lean 2008 (FSS) — not yet implemented.
- pysteps.verification has reference implementations of all of these, but the
  arithmetic is simple enough that depending on pysteps for this would be
  silly (and is impossible until the macOS install is unblocked).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContingencyTable:
    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int

    @property
    def total(self) -> int:
        return self.hits + self.misses + self.false_alarms + self.correct_negatives


@dataclass(frozen=True)
class EtaStats:
    mae: float
    median_abs_err: float
    rmse: float
    bias: float  # mean(predicted - actual). Positive = we predict later than truth.
    n: int


def contingency(predicted: np.ndarray, actual: np.ndarray) -> ContingencyTable:
    """Build a 2×2 contingency table from boolean arrays."""
    p = np.asarray(predicted, dtype=bool)
    a = np.asarray(actual, dtype=bool)
    if p.shape != a.shape:
        raise ValueError(f"shape mismatch: predicted {p.shape} vs actual {a.shape}")
    return ContingencyTable(
        hits=int(np.sum(p & a)),
        misses=int(np.sum(~p & a)),
        false_alarms=int(np.sum(p & ~a)),
        correct_negatives=int(np.sum(~p & ~a)),
    )


def pod(ct: ContingencyTable) -> float:
    """Probability of detection (recall)."""
    return _safe_div(ct.hits, ct.hits + ct.misses)


def far(ct: ContingencyTable) -> float:
    """False alarm ratio = false alarms / total predictions of yes."""
    return _safe_div(ct.false_alarms, ct.hits + ct.false_alarms)


def csi(ct: ContingencyTable) -> float:
    """Critical Success Index = hits / (hits + misses + false_alarms)."""
    return _safe_div(ct.hits, ct.hits + ct.misses + ct.false_alarms)


def hss(ct: ContingencyTable) -> float:
    """Heidke skill score: 1 = perfect, 0 = no skill (climatology), <0 worse than random."""
    denom = (
        (ct.hits + ct.misses) * (ct.misses + ct.correct_negatives)
        + (ct.hits + ct.false_alarms) * (ct.false_alarms + ct.correct_negatives)
    )
    if denom == 0:
        return float("nan")
    return 2.0 * (ct.hits * ct.correct_negatives - ct.misses * ct.false_alarms) / denom


def ets(ct: ContingencyTable) -> float:
    """Equitable Threat Score / Gilbert Skill Score."""
    if ct.total == 0:
        return float("nan")
    hits_random = (ct.hits + ct.misses) * (ct.hits + ct.false_alarms) / ct.total
    denom = ct.hits + ct.misses + ct.false_alarms - hits_random
    if denom == 0:
        return float("nan")
    return (ct.hits - hits_random) / denom


def frequency_bias(ct: ContingencyTable) -> float:
    """Predicted-yes frequency / observed-yes frequency. 1.0 = unbiased."""
    return _safe_div(ct.hits + ct.false_alarms, ct.hits + ct.misses)


def f1(ct: ContingencyTable) -> float:
    return _safe_div(2 * ct.hits, 2 * ct.hits + ct.false_alarms + ct.misses)


def eta_stats(predicted_min: np.ndarray, actual_min: np.ndarray) -> EtaStats:
    """ETA error stats over pairs where both predicted and actual are finite."""
    p = np.asarray(predicted_min, dtype=float)
    a = np.asarray(actual_min, dtype=float)
    if p.shape != a.shape:
        raise ValueError(f"shape mismatch: predicted {p.shape} vs actual {a.shape}")
    mask = np.isfinite(p) & np.isfinite(a)
    if not mask.any():
        return EtaStats(np.nan, np.nan, np.nan, np.nan, 0)
    err = p[mask] - a[mask]
    return EtaStats(
        mae=float(np.mean(np.abs(err))),
        median_abs_err=float(np.median(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        bias=float(np.mean(err)),
        n=int(mask.sum()),
    )


def _safe_div(num: float | int, denom: float | int) -> float:
    return float(num) / denom if denom > 0 else float("nan")


def fss(
    predicted: np.ndarray,
    actual: np.ndarray,
    *,
    threshold: float,
    neighborhood_px: int,
) -> float:
    """Fractions Skill Score (Roberts & Lean 2008).

    Both fields are binarised at ``threshold``. Fractions are computed in a
    square neighbourhood of side ``neighborhood_px`` (so a 5-km neighbourhood
    on a 500 m grid is ``neighborhood_px=10``). NaN pixels are treated as 0
    (below threshold) so they neither contribute hits nor poison sums.

    Returns 1.0 for a perfect forecast, 0.0 for no skill (e.g. all-zero
    prediction against any non-zero observation), and approaches the random
    base rate as the prediction is decorrelated.
    """
    from scipy.ndimage import uniform_filter

    if predicted.shape != actual.shape:
        raise ValueError(f"shape mismatch: predicted {predicted.shape} vs actual {actual.shape}")

    pred_bin = (np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0) >= threshold).astype(np.float32)
    actual_bin = (np.nan_to_num(actual, nan=0.0, posinf=0.0, neginf=0.0) >= threshold).astype(np.float32)

    pred_frac = uniform_filter(pred_bin, size=neighborhood_px, mode="constant", cval=0.0)
    actual_frac = uniform_filter(actual_bin, size=neighborhood_px, mode="constant", cval=0.0)

    mse = float(np.mean((pred_frac - actual_frac) ** 2))
    mse_ref = float(np.mean(pred_frac ** 2 + actual_frac ** 2))
    if mse_ref == 0:
        return float("nan")
    return 1.0 - mse / mse_ref
