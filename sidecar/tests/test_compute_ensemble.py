"""A0 — STEPS ensemble wiring in the cycle (website Phase A plan §A0).

Fully synthetic: tiny ODIM HDF5 composites (64×64 @ 500 m centred on the
test home) drive ``_compute_sync`` end-to-end, with ``run_ensemble``
monkeypatched to a deterministic fake — real STEPS on national-size grids
is far too slow for tests and is already covered on small grids by
``tests/test_probabilistic.py`` in the core suite.

Since package C0 (Phase B addendum 2026-08-29, fullRange-only frames) the
fixtures run at the 10-min fullRange cadence: the STEPS timestep is
derived from the measured inter-frame spacing, and the cycle takes a
no-new-frame fast path when the newest frame is unchanged from the
previous cycle. The horizon is ``forecast.steps.horizon_min`` and counts
from RADAR-FRAME time, so at the 10-min cadence the default 90 gives 9
timesteps (60 would give 6) — the extra steps are what the live 14–18 min
frame age eats before the served leads begin.

Covers:
- the ``frame_age_corrected_leads`` pure helper (incl. horizon clamping,
  at both the legacy 5-min and the live 10-min timestep),
- fallback paths (``steps.enabled=false``, ``EnsembleUnavailable``,
  fewer than 3 frames) leaving the deterministic state untouched,
- the happy path populating ``p_ensemble``, the ``probabilistic`` block,
  ``eta_p50_window_min`` and ``ensemble_ms``, with the dt-derived
  timestep reaching ``run_ensemble``,
- the no-new-frame fast path: previous state re-emitted with refreshed
  clock fields, no recompute, no streak/hysteresis stepping.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
from pyproj import CRS, Transformer

from dmi_nowcast_core.national import national_products
from dmi_nowcast_core.probabilistic import (
    EnsembleUnavailable,
    frame_age_corrected_leads,
)
from dmi_nowcast_sidecar import compute as compute_mod
from dmi_nowcast_sidecar.compute import CycleEngine
from dmi_nowcast_sidecar.config import Config, StepsConfig

# Same home as ``minimal_config`` (conftest.py); the synthetic grid is
# built so this point lands exactly at its centre.
HOME_LON, HOME_LAT = 10.32, 55.33
DMI_PROJ = "+proj=stere +lat_0=56 +lon_0=10.5666 +lat_ts=56 +ellps=WGS84 +units=m +no_defs"
GRID_PX = 64          # native synthetic grid — 64×64 @ 500 m = 32×32 km
PIXEL_M = 500.0
GAIN, OFFSET = 0.5, -32.0
NODATA, UNDETECT = 255, 0


# ---------------------------------------------------------------------------
# frame_age_corrected_leads — pure helper
# ---------------------------------------------------------------------------

def test_leads_identity_at_zero_frame_age() -> None:
    out = frame_age_corrected_leads(
        [5, 10, 30, 60], 0.0, n_timesteps=12, timestep_min=5.0,
    )
    assert out == (5.0, 10.0, 30.0, 60.0)


def test_leads_add_frame_age_and_snap_to_whole_timesteps() -> None:
    # 5+2.5=7.5 and 10+2.5=12.5 snap UP to the next whole timestep so
    # aggregate_at_home's round() bucketing matches the national products'
    # ceil() bucketing at any fractional frame age.
    out = frame_age_corrected_leads(
        [5.0, 10.0], 2.5, n_timesteps=12, timestep_min=5.0,
    )
    assert out == (10.0, 15.0)


def test_leads_snap_covers_round_vs_ceil_divergence() -> None:
    # age 2.0, lead 10 → effective 12.0: unsnapped, aggregate_at_home would
    # round(12/5)=2 steps while national ceil gives 3 — the snapped lead
    # (15.0) forces both into the 3-step bucket.
    out = frame_age_corrected_leads([10.0], 2.0, n_timesteps=12, timestep_min=5.0)
    assert out == (15.0,)


def test_leads_exact_multiples_do_not_overshoot() -> None:
    # Float fuzz at an exact timestep multiple must not ceil into the next
    # bucket (epsilon guard).
    out = frame_age_corrected_leads([10.0], 5.0, n_timesteps=12, timestep_min=5.0)
    assert out == (15.0,)


def test_leads_clamp_at_horizon_end() -> None:
    # Horizon is 12 × 5 = 60 min; 45+10 fits, 60+10 must clamp to 60.
    out = frame_age_corrected_leads(
        [45.0, 60.0], 10.0, n_timesteps=12, timestep_min=5.0,
    )
    assert out == (55.0, 60.0)


def test_leads_clamp_can_collapse_to_duplicates() -> None:
    out = frame_age_corrected_leads(
        [55.0, 60.0], 20.0, n_timesteps=12, timestep_min=5.0,
    )
    assert out == (60.0, 60.0)


def test_leads_floor_at_first_timestep() -> None:
    # A sub-timestep lead still needs at least one ensemble step —
    # mirrors aggregate_at_home's max(1, steps_in_lead).
    out = frame_age_corrected_leads([1.0], 0.0, n_timesteps=12, timestep_min=5.0)
    assert out == (5.0,)


def test_leads_negative_frame_age_treated_as_zero() -> None:
    out = frame_age_corrected_leads([10.0], -3.0, n_timesteps=12, timestep_min=5.0)
    assert out == (10.0,)


def test_leads_invalid_args_raise() -> None:
    with pytest.raises(ValueError):
        frame_age_corrected_leads([10.0], 0.0, n_timesteps=0, timestep_min=5.0)
    with pytest.raises(ValueError):
        frame_age_corrected_leads([10.0], 0.0, n_timesteps=12, timestep_min=0.0)


def test_leads_snap_at_10_min_buckets() -> None:
    # The live C0 configuration: 10-min fullRange timestep, 6-step horizon.
    # The helper is timestep-agnostic — snapping works at 10-min buckets
    # exactly as at 5: every corrected lead is a whole timestep multiple,
    # keeping aggregate_at_home's round() in the national ceil() bucket.
    out = frame_age_corrected_leads(
        [5.0, 10.0, 20.0, 30.0, 60.0], 4.0, n_timesteps=6, timestep_min=10.0,
    )
    assert out == (10.0, 20.0, 30.0, 40.0, 60.0)
    # Exact-multiple epsilon guard at the 10-min step too.
    out = frame_age_corrected_leads([10.0], 10.0, n_timesteps=6, timestep_min=10.0)
    assert out == (20.0,)
    # Horizon clamp at 6 × 10 = 60 min.
    out = frame_age_corrected_leads([60.0], 4.0, n_timesteps=6, timestep_min=10.0)
    assert out == (60.0,)


# ---------------------------------------------------------------------------
# Synthetic ODIM fixtures
# ---------------------------------------------------------------------------

def _corners_lonlat() -> dict[str, tuple[float, float]]:
    """Grid corners such that home sits exactly at the grid centre."""
    crs = CRS.from_proj4(DMI_PROJ)
    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x_home, y_home = to_proj.transform(HOME_LON, HOME_LAT)
    half = GRID_PX / 2 * PIXEL_M
    return {
        "UL": to_wgs.transform(x_home - half, y_home + half),
        "UR": to_wgs.transform(x_home + half, y_home + half),
        "LL": to_wgs.transform(x_home - half, y_home - half),
        "LR": to_wgs.transform(x_home + half, y_home - half),
    }


def _write_composite(path: Path, ts: datetime, dbz_value: float) -> None:
    """Write a minimal DMI-style ODIM HDF5 composite (uniform ``dbz_value``)."""
    raw_value = int(round((dbz_value - OFFSET) / GAIN))
    assert UNDETECT < raw_value < NODATA
    raw = np.full((GRID_PX, GRID_PX), raw_value, dtype=np.uint8)
    corners = _corners_lonlat()
    with h5py.File(path, "w") as h5:
        what = h5.create_group("what")
        # DMI keeps scaling + product in the ROOT /what (see parse.py).
        what.attrs["gain"] = GAIN
        what.attrs["offset"] = OFFSET
        what.attrs["nodata"] = NODATA
        what.attrs["undetect"] = UNDETECT
        what.attrs["date"] = ts.strftime("%Y%m%d").encode()
        what.attrs["time"] = ts.strftime("%H%M%S").encode()
        what.attrs["product"] = b"DBZH"
        where = h5.create_group("where")
        where.attrs["projdef"] = DMI_PROJ.encode()
        where.attrs["xscale"] = PIXEL_M
        where.attrs["yscale"] = PIXEL_M
        for name, (lon, lat) in corners.items():
            where.attrs[f"{name}_lon"] = lon
            where.attrs[f"{name}_lat"] = lat
        how = h5.create_group("how")
        how.attrs["zr-a"] = 200.0
        how.attrs["zr-b"] = 1.6
        h5.create_group("dataset1").create_group("data1").create_dataset(
            "data", data=raw,
        )


@pytest.fixture
def synthetic_paths(tmp_path: Path) -> list[Path]:
    """Three uniformly wet frames, 10 min cadence (fullRange-style, C0),
    newest ~4 min old.

    30–31 dBZ ≈ 2.7–3.0 mm/h — decisively above the 0.5 mm/h threshold,
    so the deterministic path is stably wet at every lead. Distinct dBZ
    per frame lets tests verify chronological (oldest → newest) ordering.
    A ~4 min frame age keeps the corrected lead 5 + age inside ONE
    10-min ensemble timestep (ceil(9/10) = 1) while lead 10 + age needs
    two (ceil(14/10) = 2) — the happy-path test relies on that split.
    """
    newest = datetime.now(timezone.utc) - timedelta(minutes=4)
    paths: list[Path] = []
    for i, dbz in enumerate((30.0, 30.5, 31.0)):
        ts = newest - timedelta(minutes=10 * (2 - i))
        p = tmp_path / f"synthetic_{i}.h5"
        _write_composite(p, ts, dbz)
        paths.append(p)
    return paths


@pytest.fixture
def engine(
    minimal_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> CycleEngine:
    """CycleEngine on the synthetic grid — no network, no rendering."""
    minimal_config.forecast.leads_min = [5, 10, 20, 30, 60]
    eng = CycleEngine(minimal_config)
    # Skip the OSM basemap fetch (network) and PIL rendering (irrelevant here).
    eng._basemap_attempted = True
    monkeypatch.setattr(compute_mod, "render_frames", lambda **kw: (b"", 0.0))
    return eng


def _make_fake_run_ensemble(calls: list[dict]):
    """A deterministic stand-in for ``run_ensemble``.

    8 members on the downsampled grid: members 0–3 wet from timestep 0,
    member 4 wet ONLY at timestep index 1 (20 min from frame time at the
    10-min fullRange timestep), members 5–7 dry. Known exceedance
    fractions everywhere on the grid.
    """
    def fake_run_ensemble(dbz_frames, vy, vx, **kwargs):
        calls.append({"dbz_frames": dbz_frames, "vy": vy, "vx": vx, **kwargs})
        f = kwargs["downsample_factor"]
        h, w = GRID_PX // f, GRID_PX // f
        out = np.zeros((8, kwargs["n_timesteps"], h, w), dtype=np.float32)
        out[0:4] = 5.0
        out[4, 1] = 5.0
        return out

    return fake_run_ensemble


# ---------------------------------------------------------------------------
# Fallback paths — state identical to the deterministic-only cycle
# ---------------------------------------------------------------------------

def _assert_deterministic_only(state) -> None:
    """New fields at their absent-defaults; existing semantics intact."""
    assert state.probabilistic is None
    assert state.forecast.eta_p50_window_min is None
    assert state.diagnostics.ensemble_ms == 0.0
    assert state.diagnostics.national_ms == 0.0
    assert state.diagnostics.artifact_bytes == 0
    assert [e.lead_min for e in state.forecast.per_lead] == [5, 10, 20, 30, 60]
    for entry in state.forecast.per_lead:
        assert entry.p_ensemble is None
        # Uniform ~2.7 mm/h frames → deterministically wet at every lead.
        assert entry.p_rain == 1.0
        assert entry.p_calibrated == 1.0
        assert entry.rain_rate_mm_h > 0.5
    assert state.forecast.eta_minutes == 5.0
    assert state.now.raining is True


def _stable_view(state) -> dict:
    """State dump minus timing-volatile fields, floats rounded.

    ``rain_rate``/``confidence`` depend (weakly) on the wall-clock frame
    age, which differs by the seconds between two ``_compute_sync`` calls;
    rounding removes that jitter while still catching semantic drift.
    """
    d = state.model_dump(mode="json")
    d.pop("generated_at")
    d.pop("diagnostics")
    d["radar"].pop("data_age_minutes")
    d["confidence"] = round(d["confidence"], 2)
    for e in d["forecast"]["per_lead"]:
        e["rain_rate_mm_h"] = round(e["rain_rate_mm_h"], 3)
    return d


def test_steps_disabled_falls_back_and_never_calls_run_ensemble(
    engine: CycleEngine,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine.config.forecast.steps.enabled = False

    def _boom(*args, **kwargs):  # pragma: no cover - failing is the assertion
        raise AssertionError("run_ensemble must not be called when disabled")

    monkeypatch.setattr(compute_mod, "run_ensemble", _boom)
    state = engine._compute_sync(synthetic_paths, fetch_ms=0.0)
    _assert_deterministic_only(state)


def test_ensemble_unavailable_falls_back_to_deterministic_state(
    minimal_config: Config,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EnsembleUnavailable → warning + exactly today's deterministic output."""
    minimal_config.forecast.leads_min = [5, 10, 20, 30, 60]
    monkeypatch.setattr(compute_mod, "render_frames", lambda **kw: (b"", 0.0))

    def _unavailable(*args, **kwargs):
        raise EnsembleUnavailable("vendored pysteps missing (test)")

    monkeypatch.setattr(compute_mod, "run_ensemble", _unavailable)
    eng_failing = CycleEngine(minimal_config)
    eng_failing._basemap_attempted = True
    state_failing = eng_failing._compute_sync(synthetic_paths, fetch_ms=0.0)
    _assert_deterministic_only(state_failing)

    # Cross-check: identical (modulo timing jitter) to a steps.enabled=false
    # run — the fallback is exactly today's behaviour, not a third variant.
    disabled_config = minimal_config.model_copy(deep=True)
    disabled_config.forecast.steps.enabled = False
    eng_disabled = CycleEngine(disabled_config)
    eng_disabled._basemap_attempted = True
    state_disabled = eng_disabled._compute_sync(synthetic_paths, fetch_ms=0.0)
    assert _stable_view(state_failing) == _stable_view(state_disabled)


def test_fewer_than_three_frames_skips_ensemble(
    engine: CycleEngine,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STEPS needs 3 frames; a 2-frame cycle must skip it, not crash."""

    def _boom(*args, **kwargs):  # pragma: no cover - failing is the assertion
        raise AssertionError("run_ensemble must not be called with < 3 frames")

    monkeypatch.setattr(compute_mod, "run_ensemble", _boom)
    state = engine._compute_sync(synthetic_paths[-2:], fetch_ms=0.0)
    _assert_deterministic_only(state)


# ---------------------------------------------------------------------------
# Happy path — mocked ensemble with known exceedance fractions
# ---------------------------------------------------------------------------

def test_happy_path_populates_ensemble_fields(
    engine: CycleEngine,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble(calls))

    state = engine._compute_sync(synthetic_paths, fetch_ms=0.0)
    age_min = state.radar.data_age_minutes
    assert 3.5 <= age_min < 5.0  # the frame-age premise the fixture sets up

    # Frame-age correction at the dt-derived 10-min timestep: member 4
    # exceeds only at timestep index 1 (20 min from FRAME time). Nominal
    # lead 5 + ~4 min age → ceil(9/10) = 1 timestep → member 4 does NOT
    # count: 4/8 = 0.5. Lead 10 + ~4 → ceil(14/10) = 2 timesteps → it
    # does: 5/8 = 0.625, as at every later lead. The 5-vs-10 split is
    # direct evidence the 10-min bucketing (not the old hardcoded 5-min
    # one) reached the ensemble reduction.
    expected_p = {5: 0.5, 10: 0.625, 20: 0.625, 30: 0.625, 60: 0.625}
    for entry in state.forecast.per_lead:
        assert entry.p_ensemble == pytest.approx(expected_p[entry.lead_min])
        # Deterministic fields keep their exact semantics alongside.
        assert entry.p_rain == 1.0
        assert entry.p_calibrated == 1.0

    # Probabilistic block: raw member fractions, explicitly uncalibrated.
    assert state.probabilistic is not None
    assert state.probabilistic.n_members == 8
    assert state.probabilistic.calibrated is False

    # ETA window: wet members first exceed at 10 min from frame time →
    # P25 = P50 = P75 = 10 min from frame = (10 − age) min from now, and
    # the forecast block mirrors the probabilistic block exactly.
    window = state.probabilistic.eta_p50_window_min
    assert window is not None
    assert state.forecast.eta_p50_window_min == window
    expected_eta = max(0.0, 10.0 - age_min)
    assert window[0] == pytest.approx(expected_eta, abs=0.02)
    assert window[1] == pytest.approx(expected_eta, abs=0.02)

    assert state.diagnostics.ensemble_ms > 0.0


def test_happy_path_wires_config_and_frames_into_run_ensemble(
    engine: CycleEngine,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble(calls))

    engine._compute_sync(synthetic_paths, fetch_ms=0.0)
    assert len(calls) == 1
    call = calls[0]

    steps_cfg = engine.config.forecast.steps
    assert call["n_ens_members"] == steps_cfg.ensemble_size
    assert call["n_cascade_levels"] == steps_cfg.n_cascade_levels
    assert call["downsample_factor"] == steps_cfg.downsample_factor
    # C0: the timestep is DERIVED from the measured 10-min inter-frame
    # spacing, and the step count keeps the configured horizon:
    # ceil(90/10) = 9 at the default horizon_min.
    assert call["n_timesteps"] == 9
    assert call["timestep_min"] == 10.0
    assert call["threshold_mm_h"] == engine.config.forecast.rain_threshold_mm_h
    assert call["pixel_scale_m"] == PIXEL_M
    # Z–R coefficients from the NEWEST composite, per the parse contract.
    assert call["zr_a"] == 200.0
    assert call["zr_b"] == 1.6

    # Exactly 3 dBZ frames, oldest → newest (fixture wrote 30.0/30.5/31.0).
    frames = call["dbz_frames"]
    assert len(frames) == 3
    maxima = [float(np.nanmax(f)) for f in frames]
    assert maxima == sorted(maxima)
    assert maxima[-1] == pytest.approx(31.0, abs=0.01)
    # Native-resolution flow, matching the frame grid.
    assert call["vy"].shape == (GRID_PX, GRID_PX)
    assert call["vx"].shape == (GRID_PX, GRID_PX)


# ---------------------------------------------------------------------------
# STEPS horizon — config-driven, measured from radar-frame time
# ---------------------------------------------------------------------------

def test_horizon_config_drives_the_step_count(
    engine: CycleEngine,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``n_timesteps = ceil(horizon_min / timestep)`` at the measured 10-min
    fullRange spacing: 9 steps at the default 90-min horizon, 6 at the old
    fixed 60. Nothing else about the cycle changes."""
    calls: list[dict] = []
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble(calls))

    engine._compute_sync(synthetic_paths, fetch_ms=0.0)
    assert calls[-1]["timestep_min"] == 10.0
    assert calls[-1]["n_timesteps"] == 9        # ceil(90 / 10)

    # Same frames, shorter horizon. ``_last_state`` is cleared so the cycle
    # recomputes instead of taking the no-new-frame fast path.
    engine.config.forecast.steps.horizon_min = 60
    engine._last_state = None
    engine._compute_sync(synthetic_paths, fetch_ms=0.0)
    assert calls[-1]["timestep_min"] == 10.0
    assert calls[-1]["n_timesteps"] == 6        # ceil(60 / 10)

    # A horizon shorter than one frame interval still buys one timestep.
    engine.config.forecast.steps.horizon_min = 30
    engine._last_state = None
    engine._compute_sync(synthetic_paths, fetch_ms=0.0)
    assert calls[-1]["n_timesteps"] == 3


def _crossing_staircase(n_timesteps: int) -> np.ndarray:
    """8 members, member ``m`` wet from timestep ``m`` onward.

    The cumulative member-exceedance fraction at step count ``k`` is
    ``min(k, 8) / 8`` at every pixel — so which step a lead lands on is
    directly readable off the probability.
    """
    ens = np.zeros((8, n_timesteps, 4, 4), dtype=np.float32)
    for m in range(8):
        ens[m, m:] = np.float32(m + 1.0)
    return ens


def test_late_leads_stop_clamping_onto_one_timestep_at_the_new_horizon() -> None:
    """The bug the horizon change fixes, in the reduction that serves it.

    At the live frame age (17 min) and the 10-min fullRange spacing, the
    45- and 60-min leads need ceil(62/10) = 7 and ceil(77/10) = 8 ensemble
    steps. A 6-step ensemble (the old fixed 60-min horizon) has neither, so
    both clamp onto its final step and the two leads publish the SAME
    probability. A 9-step ensemble (90-min horizon) has both.
    """
    old = national_products(
        _crossing_staircase(6), leads_min=(45, 60),
        timestep_min=10.0, frame_age_min=17.0, threshold_mm_h=0.5,
    )
    new = national_products(
        _crossing_staircase(9), leads_min=(45, 60),
        timestep_min=10.0, frame_age_min=17.0, threshold_mm_h=0.5,
    )

    # Old: both clamped to step 6 → 6/8, indistinguishable.
    assert float(old.p_rain[45][0, 0]) == pytest.approx(0.75)
    assert float(old.p_rain[60][0, 0]) == pytest.approx(0.75)
    np.testing.assert_array_equal(old.p_rain[45], old.p_rain[60])

    # New: step 7 vs step 8 → 7/8 vs 8/8, a real difference between the leads.
    assert float(new.p_rain[45][0, 0]) == pytest.approx(0.875)
    assert float(new.p_rain[60][0, 0]) == pytest.approx(1.0)
    assert float(new.p_rain[60][0, 0]) > float(new.p_rain[45][0, 0])
    assert not np.array_equal(new.p_rain[45], new.p_rain[60])


def test_cycle_horizon_reaches_the_manifest(
    engine: CycleEngine,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The site needs the horizon to know which leads are honest, so the
    configured value rides the manifest next to ``frame_age_min``."""
    calls: list[dict] = []
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble(calls))
    engine.config.forecast.steps.horizon_min = 90

    engine._compute_sync(synthetic_paths, fetch_ms=0.0)
    manifest = json.loads((_nowcast_dir(engine) / "manifest.json").read_text())
    assert manifest["ensemble_horizon_min"] == pytest.approx(90.0)
    # The honest horizon from now, the number the site actually shows.
    honest = manifest["ensemble_horizon_min"] - manifest["frame_age_min"]
    assert honest > 60.0  # every served lead is inside it


def test_aggregate_failure_falls_back(
    engine: CycleEngine,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run_ensemble result the aggregation can't use (wrong shape) must
    degrade to the deterministic state, not kill the cycle."""

    def _bad_shape(dbz_frames, vy, vx, **kwargs):
        return np.zeros((3, 2), dtype=np.float32)  # not (m, t, h, w)

    monkeypatch.setattr(compute_mod, "run_ensemble", _bad_shape)
    state = engine._compute_sync(synthetic_paths, fetch_ms=0.0)
    _assert_deterministic_only(state)


# ---------------------------------------------------------------------------
# No-new-frame fast path (Phase B addendum, package C0): fullRange frames
# land every ~10 min while the cycle polls every 5, so every other cycle
# re-sees the frame set it already computed. It must re-emit the previous
# state with refreshed clock fields and touch NO cross-cycle machinery.
# ---------------------------------------------------------------------------

def _boom_named(name: str):
    def _boom(*args, **kwargs):  # pragma: no cover - failing is the assertion
        raise AssertionError(f"{name} must not run on the no-new-frame fast path")
    return _boom


def test_no_new_frame_fast_path_reemits_state_without_recompute(
    engine: CycleEngine,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble(calls))

    state1 = engine._compute_sync(synthetic_paths, fetch_ms=0.0)
    assert engine._rain_incoming_streak == 1  # wet fixture → streak started
    assert state1.forecast.rain_incoming is False

    # Second cycle, same frames. Everything expensive — and every
    # cross-cycle state machine — must stay untouched: raining_now
    # hysteresis, STEPS, national reduction, artifacts, rendering.
    engine._raining_now.update = _boom_named("raining_now.update")  # type: ignore[method-assign]
    monkeypatch.setattr(compute_mod, "run_ensemble", _boom_named("run_ensemble"))
    monkeypatch.setattr(compute_mod, "render_frames", _boom_named("render_frames"))
    monkeypatch.setattr(
        compute_mod, "national_products", _boom_named("national_products"),
    )
    monkeypatch.setattr(
        compute_mod, "write_national_artifacts",
        _boom_named("write_national_artifacts"),
    )
    state2 = engine._compute_sync(synthetic_paths, fetch_ms=7.5)

    # run_ensemble ran exactly once ACROSS both cycles (the fake recorded
    # one call; the second cycle's stub would have raised).
    assert len(calls) == 1

    # Semantically the same state — only clock-derived fields refreshed.
    assert _stable_view(state2) == _stable_view(state1)
    assert state2.generated_at >= state1.generated_at
    assert state2.radar.latest_ts == state1.radar.latest_ts
    assert state2.radar.data_age_minutes >= state1.radar.data_age_minutes

    # Streak NOT double-advanced: two poll firings saw ONE radar
    # observation, so the two-cycle persistence must not fire.
    assert engine._rain_incoming_streak == 1
    assert state2.forecast.rain_incoming is False

    # Distinguishable skipped-cycle diagnostics: compute_ms 0 marker,
    # nothing re-run, but the real fetch_ms passed through.
    d = state2.diagnostics
    assert d.compute_ms == 0.0
    assert d.ensemble_ms == 0.0
    assert d.render_ms == 0.0
    assert d.national_ms == 0.0
    assert d.artifact_bytes == 0
    assert d.fetch_ms == 7.5


def test_fast_path_streak_advances_once_per_radar_observation(
    engine: CycleEngine,
    synthetic_paths: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rain_incoming streak counts radar observations, not poll
    firings: wet frame → skip (same frame) → NEW wet frame must yield
    streak 2 (rain_incoming True), exactly as if the skipped cycle had
    never fired."""
    calls: list[dict] = []
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble(calls))

    state1 = engine._compute_sync(synthetic_paths, fetch_ms=0.0)
    state2 = engine._compute_sync(synthetic_paths, fetch_ms=0.0)
    assert state1.forecast.rain_incoming is False
    assert state2.forecast.rain_incoming is False
    assert len(calls) == 1

    # Next fullRange frame arrives 10 min after the previous newest.
    new_path = tmp_path / "synthetic_3.h5"
    _write_composite(
        new_path, state1.radar.latest_ts + timedelta(minutes=10), 31.5,
    )
    state3 = engine._compute_sync(synthetic_paths[1:] + [new_path], fetch_ms=0.0)

    assert len(calls) == 2  # full compute resumed
    assert engine._rain_incoming_streak == 2
    assert state3.forecast.rain_incoming is True
    assert state3.radar.latest_ts == state1.radar.latest_ts + timedelta(minutes=10)
    assert state3.diagnostics.ensemble_ms > 0.0


def test_first_cycle_never_takes_fast_path(
    engine: CycleEngine,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no previous cycle there is nothing to re-emit — the first
    compute must always run in full, whatever the frame timestamps."""
    calls: list[dict] = []
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble(calls))
    assert engine._last_state is None
    state = engine._compute_sync(synthetic_paths, fetch_ms=0.0)
    assert len(calls) == 1
    assert state.diagnostics.ensemble_ms > 0.0


# ---------------------------------------------------------------------------
# fullRange-only fetch filter (Phase B addendum, package C0)
# ---------------------------------------------------------------------------

async def test_fetch_requests_fullrange_only(engine: CycleEngine) -> None:
    captured: dict = {}

    async def fake_list_latest(*, limit: int, scan_type: str | None = None):
        captured["limit"] = limit
        captured["scan_type"] = scan_type
        return []

    engine._client.list_latest = fake_list_latest  # type: ignore[method-assign]
    paths = await engine._fetch_latest_frames()
    assert paths == []
    # The server-side filter keeps doppler (:x5) composites out entirely.
    assert captured["scan_type"] == "fullRange"


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

def test_dmi_scan_type_locked_to_fullrange() -> None:
    """C0 decision: the runtime is fullRange-only. The Literal type keeps
    the setting visible (corpus-parity derives from it, §B0/§C1) while
    rejecting any other value."""
    from dmi_nowcast_sidecar.config import DmiConfig

    assert DmiConfig().scan_type == "fullRange"
    with pytest.raises(ValueError):
        DmiConfig(scan_type="doppler")


def test_steps_config_defaults_and_bounds() -> None:
    cfg = StepsConfig()
    assert cfg.enabled is True
    assert cfg.ensemble_size == 24
    assert cfg.n_cascade_levels == 6
    assert cfg.downsample_factor == 4
    # Horizon from radar-frame time: 60 min of served lead + ~30 of frame age.
    assert cfg.horizon_min == 90
    with pytest.raises(ValueError):
        StepsConfig(downsample_factor=0)
    with pytest.raises(ValueError):
        StepsConfig(downsample_factor=9)
    with pytest.raises(ValueError):
        StepsConfig(horizon_min=29)
    with pytest.raises(ValueError):
        StepsConfig(horizon_min=181)


# ---------------------------------------------------------------------------
# National products + artifacts in the cycle (integration of A1/A2, plan §A2)
# ---------------------------------------------------------------------------

def _nowcast_dir(engine: CycleEngine) -> Path:
    return engine.config.storage.data_dir / "nowcast"


def test_cycle_computes_national_products_and_writes_artifacts(
    engine: CycleEngine,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble(calls))

    state = engine._compute_sync(synthetic_paths, fetch_ms=0.0)

    # In-memory products published for the /forecast lookup (plan §A3).
    latest = engine.national_latest
    assert latest is not None
    products, radar_ts = latest
    assert radar_ts.tzinfo is not None
    assert products.n_members == 8
    assert products.leads_min == (10, 20, 30, 45, 60)
    f = engine.config.forecast.steps.downsample_factor
    assert products.eta_min.shape == (GRID_PX // f, GRID_PX // f)

    # Same fake ensemble as the home path: 4/8 members wet from timestep 0,
    # one more from timestep 1 → cumulative fraction 0.625 from step 1 on.
    # Every national lead (≥ 10 min + ~4 min age → ≥ 2 of the 10-min
    # timesteps) sees 0.625.
    centre = (GRID_PX // 2) // f
    for lead in products.leads_min:
        assert products.p_rain[lead][centre, centre] == pytest.approx(0.625)
    # Fraction reaches 0.5 at step 0 → ETA = 10 min from frame ≈ 6 min from now.
    eta = float(products.eta_min[centre, centre])
    assert 5.0 <= eta <= 6.5
    # Median raw rate at the ETA step: [5,5,5,5,0,0,0,0] → 2.5 mm/h.
    assert float(products.intensity_mm_h[centre, centre]) == pytest.approx(2.5)

    # §A4 in-cycle agreement: the state's p_ensemble equals the national
    # grid sampled at the home pixel, for every lead both sides serve.
    for entry in state.forecast.per_lead:
        if entry.lead_min in products.p_rain:
            grid_val = float(products.p_rain[entry.lead_min][centre, centre])
            assert entry.p_ensemble == pytest.approx(grid_val)

    # Artifacts on disk: 5 p_rain + eta + intensity + 2 motion grayscale,
    # overlays for every deterministic lead + the now frame, stamped +
    # stable manifests.
    out = _nowcast_dir(engine)
    names = sorted(p.name for p in out.iterdir())
    n_overlays = len(engine.config.forecast.leads_min) + 1
    assert len([n for n in names if n.startswith("p_rain_")]) == 5
    assert len([n for n in names if n.startswith("eta_")]) == 1
    assert len([n for n in names if n.startswith("intensity_")]) == 1
    assert len([n for n in names if n.startswith("motion_")]) == 2
    assert len([n for n in names if n.startswith("overlay_")]) == n_overlays
    assert "manifest.json" in names
    assert len([n for n in names if n.startswith("manifest_")]) == 1

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_members"] == 8
    assert manifest["leads_min"] == [10, 20, 30, 45, 60]
    assert len(manifest["artifacts"]) == 9 + n_overlays

    # R2: the motion grids ride the product grid the ensemble produced.
    assert manifest["motion"]["grid"] == "product"
    motion_entries = [e for e in manifest["artifacts"]
                      if str(e["product"]).startswith("motion_")]
    assert {e["product"] for e in motion_entries} == {
        "motion_east_kmh", "motion_north_kmh",
    }
    assert all(e["units"] == "km/h" for e in motion_entries)
    assert all(e["shape"] == manifest["grid"]["shape"] for e in motion_entries)

    # R1: schema v2 — every overlay says what it depicts and when.
    assert manifest["schema_version"] == 2
    overlays = [e for e in manifest["artifacts"] if e["product"] == "overlay"]
    assert len(overlays) == n_overlays          # cold start → no history yet
    radar_ts = datetime.fromisoformat(manifest["radar_ts_utc"])
    for entry in overlays:
        valid = datetime.fromisoformat(entry["valid_ts_utc"])
        if entry["lead_min"] == 0:
            assert entry["kind"] == "observation"
            assert valid == radar_ts
        else:
            assert entry["kind"] == "forecast"
            assert valid == radar_ts + timedelta(
                minutes=manifest["frame_age_min"] + entry["lead_min"],
            )

    assert state.diagnostics.national_ms > 0.0
    assert state.diagnostics.artifact_bytes > 0


def test_national_disabled_keeps_home_forecast_and_writes_nothing(
    engine: CycleEngine,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine.config.forecast.national.enabled = False
    calls: list[dict] = []
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble(calls))

    state = engine._compute_sync(synthetic_paths, fetch_ms=0.0)

    # Home ensemble output unaffected by the national switch (lead 5 →
    # one 10-min timestep → 4/8 members wet → 0.5, as on the happy path).
    assert state.probabilistic is not None
    assert state.forecast.per_lead[0].p_ensemble == pytest.approx(0.5)
    # No products held, nothing written, diagnostics at zero.
    assert engine.national_latest is None
    assert list(_nowcast_dir(engine).iterdir()) == []
    assert state.diagnostics.national_ms == 0.0
    assert state.diagnostics.artifact_bytes == 0


def test_national_products_failure_degrades_to_home_only(
    engine: CycleEngine,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A national reduction failure must not cost the home ensemble fields."""
    calls: list[dict] = []
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble(calls))

    def _boom(*args, **kwargs):
        raise RuntimeError("national reduction exploded")

    monkeypatch.setattr(compute_mod, "national_products", _boom)
    state = engine._compute_sync(synthetic_paths, fetch_ms=0.0)

    assert state.probabilistic is not None
    assert state.forecast.per_lead[0].p_ensemble == pytest.approx(0.5)
    assert engine.national_latest is None
    assert list(_nowcast_dir(engine).iterdir()) == []
    assert state.diagnostics.artifact_bytes == 0


def test_national_config_defaults() -> None:
    from dmi_nowcast_sidecar.config import NationalConfig

    cfg = NationalConfig()
    assert cfg.enabled is True
    assert cfg.leads_min == [10, 20, 30, 45, 60]
    assert cfg.keep_cycles == 24
    with pytest.raises(ValueError):
        NationalConfig(leads_min=[])
    with pytest.raises(ValueError):
        NationalConfig(leads_min=[20, 10])
    with pytest.raises(ValueError):
        NationalConfig(keep_cycles=0)
