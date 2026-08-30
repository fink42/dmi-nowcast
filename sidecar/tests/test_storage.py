"""StateStore atomic-write + last-good rollback."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from dmi_nowcast_sidecar.state_schema import (
    CalibrationBlock,
    DiagnosticsBlock,
    ForecastBlock,
    HomeBlock,
    MotionBlock,
    NowBlock,
    PerLeadEntry,
    RadarBlock,
    State,
)
from dmi_nowcast_sidecar.storage import StateStore


def _state(label: str = "first") -> State:
    """Helper to build a minimal State distinguishable across invocations
    via the generated_at minute field — keeps the test asserting which
    version came back without relying on every field being unique."""
    # Map label letter to a minute in [0..59] deterministically.
    minute = (hash(label) % 50) + 1
    return State(
        generated_at=datetime(2026, 5, 20, 7, minute, tzinfo=timezone.utc),
        radar=RadarBlock(
            latest_ts=datetime(2026, 5, 20, 7, max(0, minute - 5), tzinfo=timezone.utc),
            data_age_minutes=5.0,
        ),
        home=HomeBlock(lat=55.33, lon=10.32, radius_km=1.0),
        now=NowBlock(
            rain_rate_mm_h=0.0, rain_rate_p90_mm_h=0.0,
            raining=False, raining_hysteresis_state="dry",
        ),
        forecast=ForecastBlock(
            method="farneback",
            rain_incoming=False,
            eta_minutes=None,
            eta_p50_window_min=None,
            peak_intensity_mm_h=0.0,
            peak_lead_min=0,
            per_lead=[PerLeadEntry(lead_min=5, rain_rate_mm_h=0.0, p_rain=0.0, p_calibrated=0.0)],
        ),
        motion=MotionBlock(
            dy_px_per_min=0.0, dx_px_per_min=0.0,
            speed_km_per_h=0.0, bearing_deg_from=0.0,
        ),
        confidence=1.0,
        calibration=CalibrationBlock(
            fitted_at=None, n_events=None, brier_before=None, brier_after=None,
        ),
        diagnostics=DiagnosticsBlock(
            cycle_ms=100, fetch_ms=10, compute_ms=80, render_ms=10,
        ),
    )


def test_load_missing_returns_none(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    assert store.load() is None


def test_write_and_load_roundtrip(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    s = _state()
    store.write(s)
    loaded = store.load()
    assert loaded is not None
    assert loaded == s


def test_write_creates_prev_after_second_write(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    store.write(_state("a"))
    store.write(_state("b"))
    assert store.state_path.exists()
    assert store.prev_state_path.exists()
    # state.json contains the latest; .prev contains the older one.
    current = store.load()
    assert current is not None
    assert current.generated_at == _state("b").generated_at


def test_corrupt_state_falls_back_to_prev(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    store.write(_state("a"))
    store.write(_state("b"))  # promotes "a" to .prev, writes "b"
    # Corrupt the current file.
    store.state_path.write_text("not valid json {")
    loaded = store.load()
    assert loaded is not None
    assert loaded.generated_at == _state("a").generated_at  # fell back to .prev


def test_corrupt_state_no_prev_returns_none(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    store.write(_state("a"))
    store.state_path.write_text("garbage")
    # .prev didn't exist before the write (this was the first write)
    assert not store.prev_state_path.exists()
    assert store.load() is None


def test_data_dir_created_if_missing(tmp_path: Path) -> None:
    """The data_dir is created on StateStore construction (not lazily)."""
    target = tmp_path / "deep" / "nested" / "dir"
    assert not target.exists()
    StateStore(target)
    assert target.is_dir()
