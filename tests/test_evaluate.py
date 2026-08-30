"""Tests for the verification metrics. Hand-computed examples."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dmi_nowcast_core.evaluate import (
    ContingencyTable,
    contingency,
    csi,
    eta_stats,
    ets,
    f1,
    far,
    frequency_bias,
    fss,
    hss,
    pod,
)


def test_contingency_table_hand_example():
    # Predicted, Actual:
    #   T T → hit       T F → false alarm
    #   F T → miss      F F → correct neg
    pred = np.array([True, True, False, False, True, True, False])
    actl = np.array([True, False, True, False, True, True, False])
    ct = contingency(pred, actl)
    assert ct.hits == 3
    assert ct.misses == 1
    assert ct.false_alarms == 1
    assert ct.correct_negatives == 2
    assert ct.total == 7


def test_perfect_forecast():
    ct = ContingencyTable(hits=10, misses=0, false_alarms=0, correct_negatives=10)
    assert pod(ct) == 1.0
    assert far(ct) == 0.0
    assert csi(ct) == 1.0
    assert hss(ct) == 1.0
    assert ets(ct) == 1.0
    assert frequency_bias(ct) == 1.0
    assert f1(ct) == 1.0


def test_always_no_forecast():
    """If we never predict yes, POD=0; FAR is undefined (0/0)."""
    ct = ContingencyTable(hits=0, misses=10, false_alarms=0, correct_negatives=10)
    assert pod(ct) == 0.0
    assert math.isnan(far(ct)), "FAR undefined when no positive predictions"
    assert csi(ct) == 0.0
    assert frequency_bias(ct) == 0.0
    assert f1(ct) == 0.0


def test_always_yes_forecast():
    """Always predicting yes: POD=1, lots of FA."""
    ct = ContingencyTable(hits=10, misses=0, false_alarms=10, correct_negatives=0)
    assert pod(ct) == 1.0
    assert far(ct) == 0.5
    assert csi(ct) == 0.5
    assert frequency_bias(ct) == 2.0
    assert f1(ct) == pytest.approx(2 / 3)


def test_hss_zero_for_no_skill():
    """HSS = 0 when the forecast performs no better than random / climatology."""
    # Construct a case where hits·cn == misses·fa
    ct = ContingencyTable(hits=2, misses=2, false_alarms=2, correct_negatives=2)
    assert hss(ct) == 0.0


def test_hss_negative_when_worse_than_random():
    # More misses+FA than hits+CN
    ct = ContingencyTable(hits=1, misses=4, false_alarms=4, correct_negatives=1)
    assert hss(ct) < 0


def test_ets_zero_for_no_skill():
    """ETS = 0 when hits == hits_random."""
    # For a balanced contingency table with hits = expected by chance
    ct = ContingencyTable(hits=25, misses=25, false_alarms=25, correct_negatives=25)
    # hits_random = (50 * 50) / 100 = 25, hits = 25 → ETS = 0.
    assert ets(ct) == 0.0


def test_csi_invariant_under_correct_negatives():
    """CSI doesn't depend on correct negatives — known property of the metric."""
    ct1 = ContingencyTable(hits=5, misses=3, false_alarms=2, correct_negatives=10)
    ct2 = ContingencyTable(hits=5, misses=3, false_alarms=2, correct_negatives=10_000)
    assert csi(ct1) == csi(ct2)


def test_contingency_shape_mismatch_raises():
    with pytest.raises(ValueError):
        contingency(np.array([True, False]), np.array([True]))


def test_eta_stats_basic():
    pred = np.array([10.0, 12.0, 8.0, 15.0])
    actl = np.array([12.0, 12.0, 10.0, 14.0])
    s = eta_stats(pred, actl)
    # errors: -2, 0, -2, +1 → mae=1.25, bias=-0.75
    assert s.mae == pytest.approx(1.25)
    assert s.bias == pytest.approx(-0.75)
    assert s.median_abs_err == pytest.approx(1.5)
    assert s.n == 4


def test_eta_stats_ignores_nan_pairs():
    pred = np.array([10.0, np.nan, 8.0, 15.0])
    actl = np.array([12.0, 12.0, np.nan, 14.0])
    s = eta_stats(pred, actl)
    # Only two valid pairs: (10,12) and (15,14) → mae=1.5, bias=-0.5
    assert s.n == 2
    assert s.mae == pytest.approx(1.5)
    assert s.bias == pytest.approx(-0.5)


def test_eta_stats_all_nan_returns_zero_n():
    pred = np.full(5, np.nan)
    actl = np.full(5, np.nan)
    s = eta_stats(pred, actl)
    assert s.n == 0
    assert math.isnan(s.mae)


def test_fss_perfect_forecast_is_one():
    rng = np.random.default_rng(0)
    field = (rng.standard_normal((64, 64)) > 0.5).astype(np.float32) * 2.0
    assert fss(field, field, threshold=1.0, neighborhood_px=3) == pytest.approx(1.0)


def test_fss_all_zero_prediction_with_rain_actual_is_zero():
    pred = np.zeros((32, 32), dtype=np.float32)
    actual = np.zeros((32, 32), dtype=np.float32)
    actual[10:20, 10:20] = 5.0
    score = fss(pred, actual, threshold=1.0, neighborhood_px=5)
    assert score == 0.0


def test_fss_small_shift_recovered_at_neighborhood_at_least_2x_shift():
    """A 3-pixel shift should give high FSS at neighbourhood ≥ ~6 px and low at 1 px."""
    rng = np.random.default_rng(1)
    field = (rng.standard_normal((96, 96)) > 0.8).astype(np.float32) * 2.0
    shifted = np.roll(field, shift=(3, 3), axis=(0, 1))
    fss_small = fss(field, shifted, threshold=1.0, neighborhood_px=1)
    fss_large = fss(field, shifted, threshold=1.0, neighborhood_px=15)
    assert fss_large > fss_small
    assert fss_large > 0.8


def test_fss_handles_nan_pixels():
    pred = np.zeros((16, 16), dtype=np.float32)
    pred[5:10, 5:10] = 3.0
    pred[0, 0] = np.nan
    actual = pred.copy()
    actual[15, 15] = np.nan
    # NaN treated as 0 → still essentially identical, should be high FSS.
    assert fss(pred, actual, threshold=1.0, neighborhood_px=3) > 0.99


def test_fss_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fss(np.zeros((10, 10)), np.zeros((10, 11)), threshold=0.5, neighborhood_px=1)
