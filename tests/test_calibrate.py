"""Tests for isotonic calibration."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dmi_nowcast_core.calibrate import (
    IsotonicCalibrator,
    brier_score,
    fit_isotonic,
    pava,
    reliability_curve,
)


def test_pava_already_monotone_is_identity():
    y = np.array([0.1, 0.3, 0.4, 0.7, 0.9])
    np.testing.assert_allclose(pava(y), y)


def test_pava_pools_decreasing_violations():
    # 0.4, 0.6, 0.2 → the last two violate. Pool: 0.4, (0.6+0.2)/2=0.4 → 0.4, 0.4, 0.4
    y = np.array([0.4, 0.6, 0.2])
    out = pava(y)
    assert out[0] == pytest.approx(0.4)
    assert out[1] == pytest.approx(0.4)
    assert out[2] == pytest.approx(0.4)
    # Result must be monotone non-decreasing
    assert np.all(np.diff(out) >= -1e-12)


def test_pava_more_complex_pooling():
    # 1, 0, 0, 1, 0 → standard textbook example: pool to 0.4 throughout? Let's compute:
    # Start: [1] → blocks=[1]/[1]
    # Add 0: blocks=[1, 0] → violation; merge to [1/2 of 2 items]
    # Add 0: blocks=[1/2, 0] → violation; merge to [1/3 of 3 items]
    # Add 1: blocks=[1/3, 1] → ok
    # Add 0: blocks=[1/3, 1, 0] → violation between 1 and 0; merge → [1/3, 1/2]
    #   Now check 1/3 vs 1/2 → ok
    y = np.array([1, 0, 0, 1, 0], dtype=float)
    out = pava(y)
    assert out[0] == pytest.approx(1/3)
    assert out[1] == pytest.approx(1/3)
    assert out[2] == pytest.approx(1/3)
    assert out[3] == pytest.approx(1/2)
    assert out[4] == pytest.approx(1/2)
    assert np.all(np.diff(out) >= -1e-12)


def test_fit_isotonic_perfect_calibration_passes_through():
    # If raw probs already match observed frequencies, calibrator should be ~identity.
    rng = np.random.default_rng(0)
    raw = np.linspace(0.05, 0.95, 100)
    # Sample binary outcomes with P=raw → expected frequency per bin matches raw.
    outcomes = (rng.random(100) < raw).astype(np.float64)
    cal = fit_isotonic(raw, outcomes)
    # Predictions in the mid-range should be roughly the input. Allow noise.
    test_inputs = np.array([0.2, 0.5, 0.8])
    calibrated = cal.predict(test_inputs)
    assert np.all(np.abs(calibrated - test_inputs) < 0.3)


def test_fit_isotonic_corrects_overconfidence():
    """If the model predicts 0.9 but only 0.5 of those events happen, calibration should pull down."""
    raw = np.array([0.9] * 50)
    outcomes = np.array([1] * 25 + [0] * 25, dtype=np.float64)
    cal = fit_isotonic(raw, outcomes)
    assert cal.predict(0.9) == pytest.approx(0.5, abs=1e-6)


def test_calibrator_output_is_monotone():
    rng = np.random.default_rng(1)
    raw = rng.random(200)
    outcomes = (rng.random(200) < raw).astype(np.float64)
    cal = fit_isotonic(raw, outcomes)
    xs = np.linspace(0.0, 1.0, 100)
    ys = cal.predict(xs)
    assert np.all(np.diff(ys) >= -1e-6)


def test_calibrator_roundtrip_via_json(tmp_path: Path):
    cal = IsotonicCalibrator(
        raw_breakpoints=(0.0, 0.3, 0.6, 1.0),
        calibrated_values=(0.0, 0.2, 0.5, 0.95),
    )
    p = tmp_path / "cal.json"
    cal.save(p)
    loaded = IsotonicCalibrator.load(p)
    assert loaded.raw_breakpoints == cal.raw_breakpoints
    assert loaded.calibrated_values == cal.calibrated_values


def test_reliability_curve_perfect_calibration():
    rng = np.random.default_rng(2)
    raw = np.linspace(0.05, 0.95, 500)
    outcomes = (rng.random(500) < raw).astype(np.float64)
    centers, freq, counts = reliability_curve(raw, outcomes, n_bins=10)
    # Observed frequencies should track bin centers within noise.
    valid = counts > 10
    err = np.abs(freq[valid] - centers[valid])
    assert np.nanmean(err) < 0.1


def test_brier_score_zero_for_perfect_prediction():
    raw = np.array([1.0, 0.0, 1.0, 0.0])
    outcomes = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(raw, outcomes) == 0.0


def test_brier_score_quarter_for_always_half():
    raw = np.full(100, 0.5)
    outcomes = np.array([1.0, 0.0] * 50)
    assert brier_score(raw, outcomes) == pytest.approx(0.25)
