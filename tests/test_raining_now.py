"""Tests for the raining_now state machine with hysteresis."""
from __future__ import annotations

import math

import pytest

from dmi_nowcast_core.raining_now import (
    RainingNow,
    RainingNowConfig,
    RainingNowResult,
)


def test_defaults_match_plan_section_14():
    cfg = RainingNowConfig()
    assert cfg.detection_threshold_mm_h == 0.1
    assert cfg.hysteresis_offset_mm_h == 0.05
    assert cfg.off_threshold_mm_h == pytest.approx(0.05)


def test_off_to_on_when_crossing_detection_threshold():
    rn = RainingNow()
    result = rn.update(0.15)
    assert result == RainingNowResult(state=True, changed=True)


def test_off_stays_off_below_detection_threshold():
    rn = RainingNow()
    result = rn.update(0.08)  # below 0.1 threshold
    assert result == RainingNowResult(state=False, changed=False)


def test_on_stays_on_within_hysteresis_band():
    """Once on, staying above the off threshold (0.05) keeps the sensor on."""
    rn = RainingNow(initial_state=True)
    result = rn.update(0.06)  # below detection (0.1) but above off (0.05)
    assert result == RainingNowResult(state=True, changed=False)


def test_on_to_off_when_below_off_threshold():
    rn = RainingNow(initial_state=True)
    result = rn.update(0.04)  # below 0.05 off threshold
    assert result == RainingNowResult(state=False, changed=True)


def test_nan_preserves_previous_state():
    """A stale or missing measurement must not flip the sensor (plan §6.1)."""
    rn = RainingNow(initial_state=True)
    result = rn.update(math.nan)
    assert result.state is True
    assert result.changed is False
    rn_off = RainingNow(initial_state=False)
    result_off = rn_off.update(math.nan)
    assert result_off.state is False


def test_hysteresis_prevents_flapping():
    """Walk the value up and down across the band — should not oscillate."""
    rn = RainingNow()
    # Climb to on
    assert rn.update(0.5).state is True
    # Hold inside hysteresis band → stays on
    assert rn.update(0.07).state is True
    assert rn.update(0.06).state is True
    # Drop below off threshold → off
    assert rn.update(0.04).state is False
    # Tiny tick back up within band → stays off (still below detection)
    assert rn.update(0.07).state is False
    # Cross detection → on again
    assert rn.update(0.12).state is True


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        RainingNowConfig(hysteresis_offset_mm_h=-0.1)
    with pytest.raises(ValueError):
        # Hysteresis ≥ threshold would make off-threshold ≤ 0 → latches on forever
        RainingNowConfig(detection_threshold_mm_h=0.1, hysteresis_offset_mm_h=0.1)


def test_custom_thresholds():
    rn = RainingNow(RainingNowConfig(detection_threshold_mm_h=1.0, hysteresis_offset_mm_h=0.2))
    assert rn.update(0.9).state is False
    assert rn.update(1.0).state is True
    assert rn.update(0.8).state is True  # within hysteresis (off=0.8)
    assert rn.update(0.79).state is False
