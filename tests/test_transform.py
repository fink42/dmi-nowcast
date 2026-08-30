"""Tests for the Marshall–Palmer Z–R conversion.

Reference benchmarks are from plan §6.2:
    ~20 dBZ → <1 mm/h (light)
    ~35 dBZ → ~5 mm/h (moderate)
    ≥45 dBZ → ≥20 mm/h (heavy or hail)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from dmi_nowcast_core.transform import (
    DBZ_HAIL_CAP,
    RAIN_RATE_CAP_MM_H,
    dbz_to_rain_rate,
)


def _scalar(dbz: float) -> float:
    return float(dbz_to_rain_rate(np.array([dbz], dtype=np.float32))[0])


@pytest.mark.parametrize(
    "dbz, expected_min, expected_max",
    [
        (20.0, 0.5, 1.0),    # light precip per plan §6.2
        (35.0, 4.0, 7.0),    # moderate
        (45.0, 18.0, 30.0),  # heavy
    ],
)
def test_canonical_dbz_bands_match_plan_guidance(dbz, expected_min, expected_max):
    rate = _scalar(dbz)
    assert expected_min <= rate <= expected_max, f"dBZ={dbz} → {rate:.2f} mm/h"


def test_undetect_negative_infinity_maps_to_zero():
    """Plan §6.2: undetect → 0 mm/h (observed dry, not missing)."""
    rate = _scalar(-math.inf)
    assert rate == 0.0


def test_nodata_nan_propagates():
    """Plan §6.2: nodata → NaN (missing, not zero)."""
    rate = _scalar(math.nan)
    assert math.isnan(rate)


def test_hail_cap_dominates_at_extreme_reflectivity():
    """A 60 dBZ hail core must not produce a hail-implied 600 mm/h. §6.2."""
    rate_at_60 = _scalar(60.0)
    rate_at_cap = _scalar(DBZ_HAIL_CAP)
    assert rate_at_60 == rate_at_cap, "values above 53 dBZ should be clamped"


def test_rain_rate_hard_cap_applied():
    """Even with the dBZ cap, the final rate cap prevents pathological values."""
    rate = _scalar(DBZ_HAIL_CAP)
    assert rate <= RAIN_RATE_CAP_MM_H


def test_custom_zr_coefficients_change_output():
    """zr_a/zr_b must be honored — the plan §6.2 reads them from /how."""
    arr = np.array([35.0], dtype=np.float32)
    mp = float(dbz_to_rain_rate(arr, zr_a=200, zr_b=1.6)[0])
    convective = float(dbz_to_rain_rate(arr, zr_a=300, zr_b=1.4)[0])
    # A drier (zr_a=300) convective relation gives lower R for the same Z.
    assert convective < mp


def test_handles_2d_array_with_mixed_sentinels():
    arr = np.array([
        [math.nan, -math.inf, 20.0],
        [35.0, 45.0, 60.0],
    ], dtype=np.float32)
    out = dbz_to_rain_rate(arr)
    assert math.isnan(out[0, 0])
    assert out[0, 1] == 0.0
    assert 0.5 <= out[0, 2] <= 1.0
    # Last column row 1: 60 dBZ → clamped to 53 dBZ rate
    assert out[1, 2] == dbz_to_rain_rate(np.array([53.0], dtype=np.float32))[0]


def test_zero_dbz_gives_tiny_rate():
    """0 dBZ → Z=1 → R = (1/200)^(1/1.6) ≈ 0.046 mm/h. Should not be filtered to 0."""
    rate = _scalar(0.0)
    assert 0.01 < rate < 0.1
