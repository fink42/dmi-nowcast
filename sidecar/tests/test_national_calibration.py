"""B4 — national calibration wiring (website Phase B plan §B4).

Fully synthetic, mirroring ``test_compute_ensemble.py``'s conventions:
tiny ODIM HDF5 composites (64×64 @ 500 m centred on the test home) drive
``_compute_sync`` end-to-end with ``run_ensemble`` monkeypatched to the
same deterministic fake (raw exceedance fraction 0.625 at every lead).

Covers:
- national curves applied to home ``p_ensemble`` and the national
  ``p_rain`` grids, exactly as ``np.interp`` over the breakpoints
  predicts (float32, NaN pass-through),
- missing/corrupt curve file → today's raw behaviour (``calibrated:
  false``, raw fractions, null metadata everywhere),
- partial lead coverage → covered leads calibrated, uncovered leads RAW
  (never interpolated between leads' curves), flags/metadata truthful,
- the artifact manifest's ``calibration`` block and the ``/forecast``
  flags staying consistent with the grids actually served,
- the legacy binary ``p_calibrated`` path bit-for-bit untouched with
  both curve files present (the HA contract).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pyproj import CRS, Transformer

from dmi_nowcast_core.national import NationalProducts
from dmi_nowcast_sidecar import compute as compute_mod
from dmi_nowcast_sidecar.app import create_app
from dmi_nowcast_sidecar.compute import CycleEngine
from dmi_nowcast_sidecar.config import CalibrationConfig, Config

# Same home as ``minimal_config`` (conftest.py); the synthetic grid is
# built so this point lands exactly at its centre (adapted from
# test_compute_ensemble.py).
HOME_LON, HOME_LAT = 10.32, 55.33
DMI_PROJ = "+proj=stere +lat_0=56 +lon_0=10.5666 +lat_ts=56 +ellps=WGS84 +units=m +no_defs"
GRID_PX = 64          # native synthetic grid — 64×64 @ 500 m = 32×32 km
PIXEL_M = 500.0
GAIN, OFFSET = 0.5, -32.0
NODATA, UNDETECT = 255, 0

HOME_LEADS = [5, 10, 20, 30, 60]          # engine leads_min in these tests
NATIONAL_LEADS = (10, 20, 30, 45, 60)     # NationalConfig default
RAW_FRACTION = 0.625                      # fake ensemble: 5/8 members wet

FITTED_AT = "2026-08-15T03:00:00+00:00"
NATIONAL_METADATA = {
    "fitted_at": FITTED_AT,
    "n_samples": 3_800_000,
    "brier_before": 0.191,
    "brier_after": 0.163,
}

LEGACY_FITTED_AT = "2026-07-01T03:00:00+00:00"


def _curve(kink_value: float) -> dict:
    """Identity-ish curve with a known kink at raw 0.5."""
    return {
        "raw_breakpoints": [0.0, 0.5, 1.0],
        "calibrated_values": [0.0, kink_value, 0.95],
    }


# Distinct kink per lead so a curve applied to the wrong lead is caught.
FULL_CURVES = {
    "5": _curve(0.30), "10": _curve(0.35), "20": _curve(0.40),
    "30": _curve(0.45), "45": _curve(0.50), "60": _curve(0.55),
}


def _interp32(x, curve: dict) -> np.ndarray:
    """The test oracle: the exact float32 np.interp the calibrator runs."""
    bp = np.asarray(curve["raw_breakpoints"], dtype=np.float32)
    cv = np.asarray(curve["calibrated_values"], dtype=np.float32)
    return np.interp(np.asarray(x, dtype=np.float32), bp, cv).astype(np.float32)


def _expected(lead: int, curves: dict = FULL_CURVES, raw: float = RAW_FRACTION) -> float:
    return float(_interp32(raw, curves[str(lead)]))


def _write_curves_file(path: Path, curves: dict, metadata: dict = NATIONAL_METADATA) -> None:
    path.write_text(json.dumps({"metadata": metadata, "curves": curves}))


# ---------------------------------------------------------------------------
# Synthetic ODIM helpers (adapted from test_compute_ensemble.py)
# ---------------------------------------------------------------------------

def _corners_lonlat() -> dict[str, tuple[float, float]]:
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
    raw_value = int(round((dbz_value - OFFSET) / GAIN))
    assert UNDETECT < raw_value < NODATA
    raw = np.full((GRID_PX, GRID_PX), raw_value, dtype=np.uint8)
    corners = _corners_lonlat()
    with h5py.File(path, "w") as h5:
        what = h5.create_group("what")
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
    """Three uniformly wet frames, 5 min cadence, newest ~4 min old.

    Same premise as test_compute_ensemble.py: the ~4 min frame age pushes
    every corrected home lead to ≥ 2 ensemble timesteps, so the fake
    ensemble's exceedance fraction is exactly 0.625 at EVERY served lead.
    """
    newest = datetime.now(timezone.utc) - timedelta(minutes=4)
    paths: list[Path] = []
    for i, dbz in enumerate((30.0, 30.5, 31.0)):
        ts = newest - timedelta(minutes=5 * (2 - i))
        p = tmp_path / f"synthetic_{i}.h5"
        _write_composite(p, ts, dbz)
        paths.append(p)
    return paths


def _make_fake_run_ensemble(calls: list[dict]):
    """Deterministic run_ensemble stand-in: 8 members, fraction 0.625
    everywhere from timestep index 1 on (see test_compute_ensemble.py)."""
    def fake_run_ensemble(dbz_frames, vy, vx, **kwargs):
        calls.append({"dbz_frames": dbz_frames, "vy": vy, "vx": vx, **kwargs})
        f = kwargs["downsample_factor"]
        h, w = GRID_PX // f, GRID_PX // f
        out = np.zeros((8, kwargs["n_timesteps"], h, w), dtype=np.float32)
        out[0:4] = 5.0
        out[4, 1] = 5.0
        return out

    return fake_run_ensemble


def _make_engine(
    minimal_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> CycleEngine:
    """CycleEngine on the synthetic grid — no network, no rendering.

    Curve files must be written to the config paths BEFORE calling this:
    curves load once at engine init (the calibrate.sh restart contract).
    """
    minimal_config.forecast.leads_min = list(HOME_LEADS)
    monkeypatch.setattr(
        compute_mod, "run_ensemble", _make_fake_run_ensemble([]),
    )
    monkeypatch.setattr(compute_mod, "render_frames", lambda **kw: (b"", 0.0))
    eng = CycleEngine(minimal_config)
    eng._basemap_attempted = True
    return eng


def _centre_ds(engine: CycleEngine) -> int:
    f = engine.config.forecast.steps.downsample_factor
    return (GRID_PX // 2) // f


def _manifest(engine: CycleEngine) -> dict:
    path = engine.config.storage.data_dir / "nowcast" / "manifest.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

def test_calibration_config_gains_national_path_additively() -> None:
    cfg = CalibrationConfig()
    assert cfg.national_curves_path == Path("/var/lib/dmi-nowcast/national_curves.json")
    # Legacy default untouched.
    assert cfg.curves_path == Path("./calibration_curves.json")


# ---------------------------------------------------------------------------
# Full coverage — every served lead calibrated, exactly as np.interp says
# ---------------------------------------------------------------------------

def test_full_coverage_calibrates_home_p_ensemble_and_flags(
    minimal_config: Config,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_curves_file(minimal_config.calibration.national_curves_path, FULL_CURVES)
    engine = _make_engine(minimal_config, monkeypatch)
    state = engine._compute_sync(synthetic_paths, fetch_ms=0.0)

    for entry in state.forecast.per_lead:
        # Exact float32 np.interp of the raw 0.625 fraction, per lead.
        assert entry.p_ensemble == _expected(entry.lead_min)
        # Legacy deterministic fields keep their exact semantics: no
        # legacy curve file → p_calibrated == p_rain == binary forecast.
        assert entry.p_rain == 1.0
        assert entry.p_calibrated == 1.0

    prob = state.probabilistic
    assert prob is not None
    assert prob.calibrated is True
    assert prob.calibrated_leads == HOME_LEADS
    assert prob.calibration_fitted_at == datetime(2026, 8, 15, 3, 0, tzinfo=timezone.utc)
    # The legacy calibration block reflects the LEGACY file only (absent).
    assert state.calibration.fitted_at is None
    assert state.calibration.n_events is None


def test_full_coverage_calibrated_grid_replaces_raw_and_agrees_at_home(
    minimal_config: Config,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_curves_file(minimal_config.calibration.national_curves_path, FULL_CURVES)
    engine = _make_engine(minimal_config, monkeypatch)
    state = engine._compute_sync(synthetic_paths, fetch_ms=0.0)

    latest = engine.national_latest
    assert latest is not None
    products, _ = latest
    centre = _centre_ds(engine)
    for lead in products.leads_min:
        # One grid set served: the calibrated grid REPLACED the raw one.
        assert float(products.p_rain[lead][centre, centre]) == _expected(lead)

    # §A4 agreement survives calibration: same curve, same float32 math on
    # both sides, for every lead both surfaces serve.
    for entry in state.forecast.per_lead:
        if entry.lead_min in products.p_rain:
            grid_val = float(products.p_rain[entry.lead_min][centre, centre])
            assert entry.p_ensemble == grid_val


def test_manifest_carries_calibration_block(
    minimal_config: Config,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_curves_file(minimal_config.calibration.national_curves_path, FULL_CURVES)
    engine = _make_engine(minimal_config, monkeypatch)
    engine._compute_sync(synthetic_paths, fetch_ms=0.0)

    manifest = _manifest(engine)
    assert manifest["calibration"] == {
        "fitted_at": FITTED_AT,
        "calibrated_leads": list(NATIONAL_LEADS),
        "n_samples": 3_800_000,
        "brier_before": 0.191,
        "brier_after": 0.163,
    }


def test_forecast_endpoint_flags_calibrated_with_full_coverage(
    minimal_config: Config,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_curves_file(minimal_config.calibration.national_curves_path, FULL_CURVES)
    engine = _make_engine(minimal_config, monkeypatch)
    engine._compute_sync(synthetic_paths, fetch_ms=0.0)

    app = create_app(minimal_config, engine=engine, auto_start_scheduler=False)
    with TestClient(app) as c:
        r = c.get("/forecast", params={"lat": HOME_LAT, "lon": HOME_LON})
    assert r.status_code == 200
    body = r.json()
    assert body["calibrated"] is True
    fitted = datetime.fromisoformat(body["calibration_fitted_at"].replace("Z", "+00:00"))
    assert fitted == datetime(2026, 8, 15, 3, 0, tzinfo=timezone.utc)
    # The served values ARE the calibrated grid samples.
    assert body["per_lead"] == [
        {"lead_min": lead, "p_rain": pytest.approx(_expected(lead), abs=1e-9)}
        for lead in NATIONAL_LEADS
    ]


# ---------------------------------------------------------------------------
# Missing / corrupt file — today's behaviour, bit for bit
# ---------------------------------------------------------------------------

def test_missing_file_serves_raw_fractions_and_null_metadata(
    minimal_config: Config,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not minimal_config.calibration.national_curves_path.exists()
    engine = _make_engine(minimal_config, monkeypatch)
    state = engine._compute_sync(synthetic_paths, fetch_ms=0.0)

    for entry in state.forecast.per_lead:
        assert entry.p_ensemble == pytest.approx(RAW_FRACTION)
        assert entry.p_rain == 1.0
        assert entry.p_calibrated == 1.0
    prob = state.probabilistic
    assert prob is not None
    assert prob.calibrated is False
    assert prob.calibration_fitted_at is None
    assert prob.calibrated_leads is None

    products, _ = engine.national_latest
    centre = _centre_ds(engine)
    for lead in products.leads_min:
        assert float(products.p_rain[lead][centre, centre]) == pytest.approx(RAW_FRACTION)
    assert _manifest(engine)["calibration"] is None

    app = create_app(minimal_config, engine=engine, auto_start_scheduler=False)
    with TestClient(app) as c:
        body = c.get("/forecast", params={"lat": HOME_LAT, "lon": HOME_LON}).json()
    assert body["calibrated"] is False
    assert body["calibration_fitted_at"] is None
    assert body["per_lead"][0]["p_rain"] == pytest.approx(RAW_FRACTION)


def test_corrupt_file_behaves_exactly_like_missing(
    minimal_config: Config,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    minimal_config.calibration.national_curves_path.write_text("{ this is not json")
    engine = _make_engine(minimal_config, monkeypatch)
    assert engine.national_curve_leads == frozenset()
    assert engine.national_calibration_fitted_at is None

    state = engine._compute_sync(synthetic_paths, fetch_ms=0.0)
    assert state.probabilistic.calibrated is False
    assert state.probabilistic.calibrated_leads is None
    assert state.forecast.per_lead[0].p_ensemble == pytest.approx(RAW_FRACTION)
    assert _manifest(engine)["calibration"] is None


# ---------------------------------------------------------------------------
# Partial coverage — covered leads calibrated, uncovered leads RAW
# ---------------------------------------------------------------------------

PARTIAL_CURVES = {"10": FULL_CURVES["10"], "45": FULL_CURVES["45"]}


def test_partial_coverage_serves_raw_for_uncovered_leads(
    minimal_config: Config,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_curves_file(minimal_config.calibration.national_curves_path, PARTIAL_CURVES)
    engine = _make_engine(minimal_config, monkeypatch)
    state = engine._compute_sync(synthetic_paths, fetch_ms=0.0)

    # Home: only lead 10 has a curve; the rest stay raw — lead 5 is NOT
    # interpolated from the lead-10 curve.
    by_lead = {e.lead_min: e for e in state.forecast.per_lead}
    assert by_lead[10].p_ensemble == _expected(10)
    for lead in (5, 20, 30, 60):
        assert by_lead[lead].p_ensemble == pytest.approx(RAW_FRACTION)

    prob = state.probabilistic
    assert prob.calibrated is False               # not every served lead
    assert prob.calibrated_leads == [10]          # truthful subset
    assert prob.calibration_fitted_at == datetime(2026, 8, 15, 3, 0, tzinfo=timezone.utc)

    # National grids: 10 and 45 calibrated, the others raw.
    products, _ = engine.national_latest
    centre = _centre_ds(engine)
    for lead in (10, 45):
        assert float(products.p_rain[lead][centre, centre]) == _expected(lead)
    for lead in (20, 30, 60):
        assert float(products.p_rain[lead][centre, centre]) == pytest.approx(RAW_FRACTION)

    # Manifest names exactly the calibrated national leads.
    cal = _manifest(engine)["calibration"]
    assert cal["calibrated_leads"] == [10, 45]
    assert cal["fitted_at"] == FITTED_AT

    # /forecast: served grids are mixed → calibrated false, but fitted_at
    # is set (curves WERE applied to part of what's served).
    app = create_app(minimal_config, engine=engine, auto_start_scheduler=False)
    with TestClient(app) as c:
        body = c.get("/forecast", params={"lat": HOME_LAT, "lon": HOME_LON}).json()
    assert body["calibrated"] is False
    assert body["calibration_fitted_at"] is not None
    by_lead_fc = {e["lead_min"]: e["p_rain"] for e in body["per_lead"]}
    assert by_lead_fc[10] == pytest.approx(_expected(10), abs=1e-9)
    assert by_lead_fc[20] == pytest.approx(RAW_FRACTION)


# ---------------------------------------------------------------------------
# Grid application unit-level: exact np.interp, NaN pass-through, identity
# ---------------------------------------------------------------------------

def test_calibrate_national_matches_np_interp_and_preserves_nan(
    minimal_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_curves_file(minimal_config.calibration.national_curves_path, PARTIAL_CURVES)
    engine = _make_engine(minimal_config, monkeypatch)

    rng = np.random.default_rng(42)
    grid10 = rng.random((16, 16), dtype=np.float32)
    grid10[2, 3] = np.nan
    grid20 = rng.random((16, 16), dtype=np.float32)
    products = NationalProducts(
        p_rain={10: grid10, 20: grid20},
        eta_min=np.full((16, 16), 6.0, dtype=np.float32),
        intensity_mm_h=np.full((16, 16), 2.5, dtype=np.float32),
        leads_min=(10, 20),
        threshold_mm_h=0.5,
        timestep_min=5.0,
        frame_age_min=2.0,
        downsample_factor=4,
        n_members=8,
    )
    out = engine._calibrate_national(products)

    # Covered lead: exactly the float32 np.interp of every pixel; the NaN
    # pixel passes through as NaN.
    expected10 = _interp32(grid10, FULL_CURVES["10"])
    np.testing.assert_array_equal(out.p_rain[10], expected10)
    assert np.isnan(out.p_rain[10][2, 3])
    assert not np.isnan(out.p_rain[10][0, 0])
    # Uncovered lead: the RAW grid object itself, untouched.
    assert out.p_rain[20] is grid20
    # Non-probability products never go through probability curves.
    assert out.eta_min is products.eta_min
    assert out.intensity_mm_h is products.intensity_mm_h


def test_calibrate_national_is_noop_without_curves(
    minimal_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _make_engine(minimal_config, monkeypatch)
    products = NationalProducts(
        p_rain={10: np.full((4, 4), 0.5, dtype=np.float32)},
        eta_min=np.zeros((4, 4), dtype=np.float32),
        intensity_mm_h=np.zeros((4, 4), dtype=np.float32),
        leads_min=(10,),
        threshold_mm_h=0.5,
        timestep_min=5.0,
        frame_age_min=0.0,
        downsample_factor=4,
        n_members=8,
    )
    assert engine._calibrate_national(products) is products
    assert engine._calibrate_national(None) is None


# ---------------------------------------------------------------------------
# Legacy binary p_calibrated path — untouched with both curve files present
# ---------------------------------------------------------------------------

def _legacy_curves_file(path: Path) -> dict:
    """Legacy home-point curves: 1.0 → a distinctive per-lead value."""
    curves = {
        str(lead): {
            "raw_breakpoints": [0.0, 1.0],
            "calibrated_values": [0.05, 0.60 + i * 0.05],
        }
        for i, lead in enumerate(HOME_LEADS)
    }
    payload = {
        "metadata": {
            "fitted_at": LEGACY_FITTED_AT,
            "n_samples": 4000,
            "brier_before": 0.21,
            "brier_after": 0.17,
        },
        "curves": curves,
    }
    path.write_text(json.dumps(payload))
    return curves


def test_legacy_binary_path_unchanged_with_both_curve_files(
    minimal_config: Config,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_curves = _legacy_curves_file(minimal_config.calibration.curves_path)
    _write_curves_file(minimal_config.calibration.national_curves_path, FULL_CURVES)
    engine = _make_engine(minimal_config, monkeypatch)
    state = engine._compute_sync(synthetic_paths, fetch_ms=0.0)

    for entry in state.forecast.per_lead:
        # Binary forecast is wet (raw 1.0) → the LEGACY curve's endpoint,
        # exactly — never the national curve (0.95 at raw 1.0).
        assert entry.p_rain == 1.0
        assert entry.p_calibrated == float(
            _interp32(1.0, legacy_curves[str(entry.lead_min)])
        )
        # ...while p_ensemble goes through the NATIONAL curves.
        assert entry.p_ensemble == _expected(entry.lead_min)

    # The state's legacy calibration block still echoes the LEGACY file.
    assert state.calibration.fitted_at == datetime(2026, 7, 1, 3, 0, tzinfo=timezone.utc)
    assert state.calibration.n_events == 4000
    assert state.calibration.brier_before == pytest.approx(0.21)
    assert state.calibration.brier_after == pytest.approx(0.17)


def test_legacy_values_identical_with_and_without_national_file(
    minimal_config: Config,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proof of invariance: the exact legacy tuple with the national
    file present equals the one without it."""
    _legacy_curves_file(minimal_config.calibration.curves_path)

    def _legacy_view(state) -> list[tuple]:
        return [
            (e.lead_min, e.rain_rate_mm_h, e.p_rain, e.p_calibrated)
            for e in state.forecast.per_lead
        ] + [state.calibration.model_dump_json()]

    engine_without = _make_engine(minimal_config, monkeypatch)
    state_without = engine_without._compute_sync(synthetic_paths, fetch_ms=0.0)

    _write_curves_file(minimal_config.calibration.national_curves_path, FULL_CURVES)
    engine_with = _make_engine(minimal_config, monkeypatch)
    state_with = engine_with._compute_sync(synthetic_paths, fetch_ms=0.0)

    # rain_rate_mm_h drifts with wall-clock frame age between the two runs;
    # compare the calibration-relevant fields exactly.
    strip = lambda view: [
        (t[0], t[2], t[3]) if isinstance(t, tuple) else t for t in view
    ]
    assert strip(_legacy_view(state_with)) == strip(_legacy_view(state_without))
    # And the national file did change the ensemble path (sanity that the
    # invariance above isn't vacuous).
    assert state_with.forecast.per_lead[0].p_ensemble != \
        state_without.forecast.per_lead[0].p_ensemble
