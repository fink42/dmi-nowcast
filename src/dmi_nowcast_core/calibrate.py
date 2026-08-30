"""Isotonic regression calibration for probabilistic forecasts (plan §7.1).

The pysteps STEPS ensemble produces raw probabilities; these tend to be
over- or under-confident depending on flow regime and rain type. Isotonic
regression maps raw → calibrated probabilities so that "predicted 0.7 → observed
frequency 0.7" on backtest data.

Implemented in 25 lines via Pool Adjacent Violators (PAVA) — no scikit-learn
dependency.

Calibration parameters are saved as JSON and refit **monthly** on the sidecar
VM, on the 1st of each month at 03:00 local time via ``dmi-calibrate.timer``
(see ``sidecar/deploy/``). Plan §14 "calibrated probabilities".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Monotone non-decreasing mapping ``raw_prob → calibrated_prob``.

    Stored as sorted breakpoint arrays; lookup is linear interpolation
    (with values clamped at the endpoints).
    """

    raw_breakpoints: tuple[float, ...]
    calibrated_values: tuple[float, ...]

    def predict(self, raw_prob: np.ndarray | float) -> np.ndarray | float:
        arr = np.atleast_1d(np.asarray(raw_prob, dtype=np.float32))
        bp = np.asarray(self.raw_breakpoints, dtype=np.float32)
        cv = np.asarray(self.calibrated_values, dtype=np.float32)
        out = np.interp(arr, bp, cv).astype(np.float32)
        if np.ndim(raw_prob) == 0:
            return float(out[0])
        return out

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({
                "raw_breakpoints": list(self.raw_breakpoints),
                "calibrated_values": list(self.calibrated_values),
            })
        )

    @classmethod
    def load(cls, path: Path) -> "IsotonicCalibrator":
        data = json.loads(path.read_text())
        return cls(
            raw_breakpoints=tuple(data["raw_breakpoints"]),
            calibrated_values=tuple(data["calibrated_values"]),
        )


def load_calibration_curves(path: Path) -> dict[int, IsotonicCalibrator]:
    """Load the per-lead curve file produced by ``scripts/fit_calibration.py``.

    Returns ``{lead_min: IsotonicCalibrator}``. The on-disk JSON format::

        {
          "metadata": {... fitted_at, n_samples, base_rate, brier scores ...},
          "curves": {
            "10": {"raw_breakpoints": [...], "calibrated_values": [...]},
            "30": {...},
            "60": {...}
          }
        }

    Returns an empty dict if the file is missing — coordinator treats
    that as "no calibration available, expose raw probs as-is".
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    curves = data.get("curves", {})
    return {
        int(lead): IsotonicCalibrator(
            raw_breakpoints=tuple(c["raw_breakpoints"]),
            calibrated_values=tuple(c["calibrated_values"]),
        )
        for lead, c in curves.items()
    }


def calibration_metadata(path: Path) -> dict:
    """Read just the metadata block (fitted_at, n_samples, …) without
    instantiating the calibrators. Used by the coordinator to expose the
    ``calibration_built_at`` sensor attribute."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()).get("metadata", {})
    except (OSError, json.JSONDecodeError):
        return {}


def pava(y: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators. Given ``y`` (sorted by some x), return the
    monotone non-decreasing least-squares fit. Same length as input."""
    if y.size == 0:
        return y.astype(np.float64)
    # Each "block" is represented as (sum_of_y_in_block, count_of_items_in_block,
    # index_of_first_element_in_original).
    blocks_sum: list[float] = [float(y[0])]
    blocks_count: list[int] = [1]
    blocks_start: list[int] = [0]

    for i in range(1, y.size):
        blocks_sum.append(float(y[i]))
        blocks_count.append(1)
        blocks_start.append(i)
        # Merge backwards while monotonicity is violated.
        while (
            len(blocks_sum) >= 2
            and blocks_sum[-2] / blocks_count[-2]
            > blocks_sum[-1] / blocks_count[-1]
        ):
            blocks_sum[-2] += blocks_sum[-1]
            blocks_count[-2] += blocks_count[-1]
            blocks_sum.pop()
            blocks_count.pop()
            blocks_start.pop()

    result = np.empty(y.size, dtype=np.float64)
    for s, c, start in zip(blocks_sum, blocks_count, blocks_start):
        result[start : start + c] = s / c
    return result


def fit_isotonic(
    raw_probs: np.ndarray,
    outcomes: np.ndarray,
) -> IsotonicCalibrator:
    """Fit an isotonic calibrator from (raw_prob, 0/1 outcome) pairs."""
    raw = np.asarray(raw_probs, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if raw.shape != y.shape:
        raise ValueError("shape mismatch")
    if raw.size == 0:
        raise ValueError("empty inputs")
    # Filter NaN pairs.
    mask = np.isfinite(raw) & np.isfinite(y)
    raw = raw[mask]
    y = y[mask]
    if raw.size == 0:
        raise ValueError("no finite samples after NaN filter")
    # Group by unique raw probability and aggregate outcomes. Doing this
    # *before* PAVA avoids two problems: (1) np.argsort's tie ordering is
    # implementation-dependent, which would change the PAVA result for
    # duplicate raw values; (2) np.interp later in predict() chokes on
    # duplicate xp values.
    unique_raw, inverse = np.unique(raw, return_inverse=True)
    sums = np.zeros(len(unique_raw), dtype=np.float64)
    counts = np.zeros(len(unique_raw), dtype=np.int64)
    np.add.at(sums, inverse, y)
    np.add.at(counts, inverse, 1)
    mean_outcomes = sums / counts
    fitted = pava(mean_outcomes)
    return IsotonicCalibrator(
        raw_breakpoints=tuple(float(v) for v in unique_raw),
        calibrated_values=tuple(float(v) for v in fitted),
    )


def pava_weighted(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted Pool Adjacent Violators. Given ``y`` (sorted by some x) and
    per-element weights ``w`` (> 0), return the monotone non-decreasing
    weighted-least-squares fit. Same length as input.

    Identical to :func:`pava` when all weights are equal: block means become
    ``Σ(w·y)/Σw`` instead of ``Σy/n``, which reduces to the plain mean for a
    constant ``w``.
    """
    if y.size == 0:
        return y.astype(np.float64)
    if y.shape != w.shape:
        raise ValueError("shape mismatch between y and weights")
    # Blocks as (sum_of_w*y, sum_of_w, index_of_first_element).
    blocks_wy: list[float] = [float(w[0] * y[0])]
    blocks_w: list[float] = [float(w[0])]
    blocks_start: list[int] = [0]

    for i in range(1, y.size):
        blocks_wy.append(float(w[i] * y[i]))
        blocks_w.append(float(w[i]))
        blocks_start.append(i)
        # Merge backwards while monotonicity is violated (weighted means).
        while (
            len(blocks_wy) >= 2
            and blocks_wy[-2] / blocks_w[-2] > blocks_wy[-1] / blocks_w[-1]
        ):
            blocks_wy[-2] += blocks_wy[-1]
            blocks_w[-2] += blocks_w[-1]
            blocks_wy.pop()
            blocks_w.pop()
            blocks_start.pop()

    result = np.empty(y.size, dtype=np.float64)
    n_blocks = len(blocks_wy)
    for b in range(n_blocks):
        start = blocks_start[b]
        stop = blocks_start[b + 1] if b + 1 < n_blocks else y.size
        result[start:stop] = blocks_wy[b] / blocks_w[b]
    return result


def fit_isotonic_weighted(
    raw_probs: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
) -> IsotonicCalibrator:
    """Fit an isotonic calibrator from weighted (raw_prob, 0/1 outcome) pairs.

    ``weights`` are relative sample weights — e.g. the unnormalised
    inverse-inclusion-probability weights of a wet-biased calibration
    corpus (website Phase B, Finding 3). The fit is invariant to a global
    weight scale. With all weights equal the result is identical to
    :func:`fit_isotonic`.

    Mirrors :func:`fit_isotonic`'s degenerate-input handling: shape
    mismatch and empty input raise ``ValueError``; pairs where the raw
    probability, outcome, or weight is non-finite are dropped, as are
    zero-weight pairs (they contribute nothing). Negative weights are an
    input error and raise.
    """
    raw = np.asarray(raw_probs, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if raw.shape != y.shape or raw.shape != w.shape:
        raise ValueError("shape mismatch")
    if raw.size == 0:
        raise ValueError("empty inputs")
    if np.any(w[np.isfinite(w)] < 0):
        raise ValueError("negative weights")
    # Filter non-finite pairs (and weight-zero pairs — no contribution).
    mask = np.isfinite(raw) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    raw, y, w = raw[mask], y[mask], w[mask]
    if raw.size == 0:
        raise ValueError("no finite samples after NaN filter")
    # Group by unique raw probability; per group keep the weighted mean
    # outcome and the total weight (same rationale as fit_isotonic: tie
    # ordering + duplicate breakpoints).
    unique_raw, inverse = np.unique(raw, return_inverse=True)
    wy_sums = np.zeros(len(unique_raw), dtype=np.float64)
    w_sums = np.zeros(len(unique_raw), dtype=np.float64)
    np.add.at(wy_sums, inverse, w * y)
    np.add.at(w_sums, inverse, w)
    mean_outcomes = wy_sums / w_sums
    fitted = pava_weighted(mean_outcomes, w_sums)
    return IsotonicCalibrator(
        raw_breakpoints=tuple(float(v) for v in unique_raw),
        calibrated_values=tuple(float(v) for v in fitted),
    )


def brier_score_weighted(
    raw_probs: np.ndarray, outcomes: np.ndarray, weights: np.ndarray
) -> float:
    """Weighted Brier score ``Σ w·(p − y)² / Σ w``.

    Non-finite (or non-positive-weight) samples are dropped, mirroring
    :func:`brier_score`; returns NaN when nothing survives the filter.
    """
    raw = np.asarray(raw_probs, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(raw) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    raw, y, w = raw[mask], y[mask], w[mask]
    if raw.size == 0:
        return float("nan")
    return float(np.sum(w * (raw - y) ** 2) / np.sum(w))


def reliability_curve(
    raw_probs: np.ndarray,
    outcomes: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, observed_frequency, count_per_bin).

    Used to plot the reliability diagram (plan §9.3 probabilistic verification).
    """
    raw = np.asarray(raw_probs, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    mask = np.isfinite(raw) & np.isfinite(y)
    raw, y = raw[mask], y[mask]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    freq = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        in_bin = (raw >= edges[i]) & (raw < edges[i + 1] if i < n_bins - 1 else raw <= edges[i + 1])
        counts[i] = int(in_bin.sum())
        if counts[i] > 0:
            freq[i] = float(y[in_bin].mean())
    return centers, freq, counts


def brier_score(raw_probs: np.ndarray, outcomes: np.ndarray) -> float:
    raw = np.asarray(raw_probs, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    mask = np.isfinite(raw) & np.isfinite(y)
    raw, y = raw[mask], y[mask]
    if raw.size == 0:
        return float("nan")
    return float(np.mean((raw - y) ** 2))
