"""Tests for the confidence heuristic."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dmi_nowcast_core.confidence import (
    compute_confidence,
    intensity_volatility_from_disc,
    motion_divergence,
)


def test_ideal_conditions_yield_high_confidence():
    r = compute_confidence(
        horizon_minutes=5.0,
        frame_age_seconds=30.0,
        intensity_volatility=0.0,
        motion_divergence=0.0,
        n_frames=6,
    )
    assert 0.85 <= r.score <= 1.0


def test_high_volatility_drags_confidence_down():
    """Plan §7.2: intensity volatility is the most important single factor.
    Tuning note: a linear 1× multiplier means volatility=0.7 halves and
    volatility=1.0 zeros the factor, which keeps confidence usable on
    dry-to-wet transitions instead of always reporting 0."""
    stable = compute_confidence(
        horizon_minutes=10.0, frame_age_seconds=30.0,
        intensity_volatility=0.0, n_frames=3,
    ).score
    volatile = compute_confidence(
        horizon_minutes=10.0, frame_age_seconds=30.0,
        intensity_volatility=0.7, n_frames=3,
    ).score
    assert volatile < stable * 0.5


def test_long_horizon_reduces_confidence():
    short = compute_confidence(
        horizon_minutes=5.0, frame_age_seconds=30.0, intensity_volatility=0.0,
    ).score
    long_ = compute_confidence(
        horizon_minutes=60.0, frame_age_seconds=30.0, intensity_volatility=0.0,
    ).score
    assert long_ < short


def test_stale_frames_reduce_confidence():
    fresh = compute_confidence(
        horizon_minutes=10.0, frame_age_seconds=30.0, intensity_volatility=0.0,
    ).score
    stale = compute_confidence(
        horizon_minutes=10.0, frame_age_seconds=1500.0, intensity_volatility=0.0,
    ).score
    assert stale < fresh


def test_score_is_zero_when_any_factor_collapses():
    """Multiplicative: if any factor is 0, the score is 0 (or near it)."""
    r = compute_confidence(
        horizon_minutes=10.0, frame_age_seconds=2400.0,  # very stale (40+ min)
        intensity_volatility=0.0,
    )
    assert r.score == 0.0


def test_factors_dict_keys():
    r = compute_confidence(horizon_minutes=10.0, frame_age_seconds=0.0, intensity_volatility=0.0)
    assert set(r.factors.keys()) >= {
        "horizon", "frame_age", "intensity_volatility", "motion_divergence", "frame_count"
    }
    assert all(0.0 <= v <= 1.0 for v in r.factors.values())


def test_intensity_volatility_zero_for_identical_values():
    assert intensity_volatility_from_disc(2.0, 2.0) == 0.0


def test_intensity_volatility_caps_at_one_for_large_changes():
    """Big absolute changes (≥ 5 mm/h) saturate at 1.0."""
    assert intensity_volatility_from_disc(0.0, 5.0) == 1.0
    assert intensity_volatility_from_disc(5.0, 0.0) == 1.0
    assert intensity_volatility_from_disc(0.0, 20.0) == 1.0


def test_intensity_volatility_small_for_dry_to_light_rain():
    """A 0 → 0.5 mm/h transition should NOT be max volatility — the confidence
    heuristic would zero out the moment rain starts otherwise."""
    v = intensity_volatility_from_disc(0.0, 0.5)
    assert v < 0.2, f"dry→0.5mm/h should be mild volatility, got {v}"


def test_intensity_volatility_nan_inputs_treated_as_worst_case():
    assert intensity_volatility_from_disc(math.nan, 2.0) == 1.0
    assert intensity_volatility_from_disc(2.0, math.nan) == 1.0


def test_motion_divergence_uniform_flow_is_near_zero():
    h, w = 32, 32
    vy = np.full((h, w), 3.0, dtype=np.float32)
    vx = np.full((h, w), 2.0, dtype=np.float32)
    assert motion_divergence(vy, vx) < 1e-5


def test_motion_divergence_diverging_flow_is_positive():
    h, w = 32, 32
    ys, xs = np.indices((h, w), dtype=np.float32)
    # Outward-radial flow: divergence ≈ 2.0 everywhere
    cy, cx = h / 2, w / 2
    vy = (ys - cy) * 0.1
    vx = (xs - cx) * 0.1
    assert motion_divergence(vy, vx) > 0.1
