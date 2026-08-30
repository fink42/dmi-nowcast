"""A4 — validation: national must agree with home (website Phase A plan §A4).

The phase's CI acceptance test, on REAL DMI radar composites from
``radar_archive/`` with the REAL vendored STEPS ensemble — nothing about
``run_ensemble`` or the national reduction is mocked (only rendering and
the OSM basemap fetch are skipped: PIL frame rendering is irrelevant here
and the basemap is a network call).

Fixture frames (three CONSECUTIVE **fullRange** composites each, 10-min
spacing — Phase B addendum / package C0: the runtime is fullRange-only,
minutes :x0, and the STEPS timestep follows the 10-min frame spacing; the
previously used 5-min triples mixed the interleaved doppler product in):

- **Approaching triple** ``2026-02-28 07:50 / 08:00 / 08:10`` — the home
  disc is dry (max ~0.19 mm/h, under the 0.5 threshold) but a rain band
  sits within 20 km and is moving in; the fullRange verification frames
  at 08:30–09:10 show it arriving (disc p90 rising 0.25 → 1.15 → 2.05
  mm/h). Ensemble members disagree on the arrival time, so the home
  probabilities are genuinely FRACTIONAL (0 at 10 min, ~0.75 at 20,
  ~0.875 at 30, 1.0 by 45 on dev hardware) and the ETA window is
  non-degenerate (~26–29 min). This is the main agreement test: exact
  equality of mid-range member fractions is much harder to pass by
  accident than 0.0 or 1.0. Bonus realism: the two newest frames are
  10 min 1 s apart in their ODIM timestamps, so the derived STEPS
  timestep is the genuinely non-round 10.016̅ min — any residual 5-min or
  10.0 hardcoding downstream would break the agreement.
- **Dry triple** ``2025-11-22 17:50 / 18:00 / 18:10`` — the fullRange
  frames around the triple named by the plan. It parses fine, but the
  nearest rain is > 60 km from home, which makes every home probability
  0.0 — kept as the None-window / NaN-ETA branch of the §A4 semantics
  rather than the headline test.

**Clock control** (the vacuity trap called out in the plan): the archive
frames are months old, so computing frame age against the real clock would
clamp every corrected lead to the horizon end and the agreement would hold
trivially. ``compute_mod.datetime`` is monkeypatched with a subclass whose
``now(tz)`` returns ``newest_frame_ts + 4 min``; everything else
(``fromisoformat``, arithmetic) passes through to the real ``datetime``.
The produced state must then report ``radar.data_age_minutes ≈ 4``.

Asserted per plan §A4:

(a) per-lead ``p_ensemble`` (state) == national ``p_rain[lead]`` at the
    home pixel (native row/col ÷ downsample, rounded) — exact float
    equality expected (8 members ⇒ all fractions are dyadic n/8),
(b) national ``eta_min`` at the home pixel vs the state's
    ``eta_p50_window_min``: finite ETA ⟹ inside the window ± one
    timestep; window None ⟹ pixel exceedance never reaches 0.5 and the
    ETA is NaN (the minority-member caveat in ``national.py``'s docstring
    is honoured, not forced into agreement),
(c) PNG round-trip: the quantised artifacts the cycle wrote, dequantised
    with the manifest's scale/offset, match the in-memory grids within
    half a quantisation step; NaN masks round-trip exactly,
(d) end-to-end HTTP: ``GET /forecast?lat=&lon=`` at home returns the same
    per-lead probabilities / ETA / intensity as the in-memory grids.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dmi_nowcast_core.parse import parse_composite
from dmi_nowcast_sidecar import compute as compute_mod
from dmi_nowcast_sidecar.app import create_app
from dmi_nowcast_sidecar.compute import CycleEngine
from dmi_nowcast_sidecar.config import Config
from dmi_nowcast_sidecar.national_artifacts import dequantise

# radar_archive/ lives at the repo root, two levels above sidecar/tests/.
ARCHIVE_DIR = Path(__file__).resolve().parents[2] / "radar_archive"

# Rain band approaching the (still dry) home disc — the main agreement
# fixture; yields fractional home probabilities and a finite ETA. All
# fullRange (minute % 10 == 0) at 10-min spacing, per the C0 decision;
# fullRange verification frames 202602280830–0910 in the archive show the
# band reaching the home disc within the hour.
APPROACHING_TRIPLE = (
    "dk.com.202602280750.500_max.h5",
    "dk.com.202602280800.500_max.h5",
    "dk.com.202602280810.500_max.h5",
)
# The fullRange frames around the plan-named triple: parses fine but home
# (and 60 km around it) is bone dry — exercises the None-window / NaN-ETA
# branch.
DRY_TRIPLE = (
    "dk.com.202511221750.500_max.h5",
    "dk.com.202511221800.500_max.h5",
    "dk.com.202511221810.500_max.h5",
)

# Frozen frame age, minutes. Small and realistic (radar latency is a few
# minutes); keeps corrected leads well inside the 60-min STEPS horizon.
FRAME_AGE_MIN = 4.0


def _archive_paths(names: tuple[str, ...]) -> list[Path]:
    paths = [ARCHIVE_DIR / n for n in names]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        pytest.skip(f"radar_archive fixture frames not present: {missing}")
    return paths


def _freeze_compute_clock(
    monkeypatch: pytest.MonkeyPatch, instant: datetime,
) -> None:
    """Freeze ``compute_mod.datetime.now`` at ``instant``; pass the rest through.

    A ``datetime`` subclass keeps ``fromisoformat`` / arithmetic /
    isinstance checks working — only ``now`` is overridden.
    """
    assert instant.tzinfo is not None

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001 — mirror datetime.now's signature
            if tz is None:
                return instant.replace(tzinfo=None)
            return instant.astimezone(tz)

    monkeypatch.setattr(compute_mod, "datetime", _FrozenDatetime)


def _build_engine(
    minimal_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> CycleEngine:
    """Real engine, real STEPS — only rendering + basemap skipped (no network)."""
    minimal_config.forecast.steps.ensemble_size = 8  # real ensemble, CI-sized
    engine = CycleEngine(minimal_config)
    engine._basemap_attempted = True  # never fetch the OSM basemap
    monkeypatch.setattr(compute_mod, "render_frames", lambda **kw: (b"", 0.0))
    return engine


def _run_cycle(engine: CycleEngine, paths: list[Path],
               monkeypatch: pytest.MonkeyPatch):
    """Freeze the clock at newest_ts + 4 min, run the real compute path."""
    newest_ts = parse_composite(paths[-1]).timestamp_utc
    _freeze_compute_clock(monkeypatch, newest_ts + timedelta(minutes=FRAME_AGE_MIN))
    state = engine._compute_sync(paths, fetch_ms=0.0)
    return state, newest_ts


def _home_product_pixel(engine: CycleEngine) -> tuple[int, int]:
    """Home pixel on the product grid: native row/col ÷ downsample, rounded —
    exactly the convention ``/forecast`` and ``aggregate_at_home`` use."""
    products, _ = engine.national_latest
    idx = engine.geo.lonlat_to_grid(
        engine.config.home.lon, engine.config.home.lat,
    )
    f = products.downsample_factor
    return int(round(idx.row / f)), int(round(idx.col / f))


def _assert_png_roundtrip(
    nowcast_dir: Path, entry: dict, grid: np.ndarray, pixel: tuple[int, int],
) -> None:
    """§A4(c) for one grayscale artifact: NaN mask exact, values within half
    a quantisation step of the (range-clipped) in-memory grid."""
    levels = np.asarray(Image.open(nowcast_dir / entry["filename"]))
    assert levels.dtype == np.uint8
    assert levels.shape == tuple(entry["shape"])
    deq = dequantise(levels, scale=entry["scale"], offset=entry["offset"])

    # NaN mask must round-trip exactly (255 = nodata ↔ NaN).
    assert np.array_equal(np.isnan(deq), np.isnan(grid)), entry["filename"]

    # Quantisation clamps to [offset, offset + 254·scale] before rounding, so
    # the round-trip guarantee is against the clipped grid.
    half_step = entry["scale"] / 2.0
    lo = entry["offset"]
    hi = entry["offset"] + entry["scale"] * 254
    finite = np.isfinite(grid)
    err = np.abs(deq[finite] - np.clip(grid[finite], lo, hi))
    assert float(err.max(initial=0.0)) <= half_step + 1e-6, entry["filename"]

    # And the home pixel specifically (the §A4 wording).
    r, c = pixel
    if np.isfinite(grid[r, c]):
        assert abs(float(deq[r, c]) - float(np.clip(grid[r, c], lo, hi))) <= (
            half_step + 1e-6
        ), entry["filename"]
    else:
        assert math.isnan(float(deq[r, c])), entry["filename"]


def _manifest_grid_entries(manifest: dict) -> dict[tuple[str, int | None], dict]:
    return {
        (a["product"], a["lead_min"]): a
        for a in manifest["artifacts"]
        if a.get("encoding") == "grayscale8"
    }


# ---------------------------------------------------------------------------
# Approaching triple — the headline agreement test, §A4 (a)–(d) in one cycle
# ---------------------------------------------------------------------------

def test_approaching_rain_national_agrees_with_home(
    minimal_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _archive_paths(APPROACHING_TRIPLE)
    engine = _build_engine(minimal_config, monkeypatch)
    state, newest_ts = _run_cycle(engine, paths, monkeypatch)

    # Clock control worked: months-old frames, yet the age is the frozen 4 min.
    assert state.radar.data_age_minutes == pytest.approx(FRAME_AGE_MIN, abs=0.01)

    # The real ensemble + national reduction must have run — a silent
    # fallback to deterministic-only would make everything below vacuous.
    assert state.probabilistic is not None, "STEPS ensemble did not run"
    assert state.probabilistic.n_members == 8
    assert state.probabilistic.calibrated is False
    latest = engine.national_latest
    assert latest is not None, "national products were not computed"
    products, radar_ts = latest
    assert radar_ts == newest_ts
    assert products.n_members == 8
    row, col = _home_product_pixel(engine)

    # Premises (guard against a silently-dry or saturated fixture; STEPS is
    # seeded — seed=42 — so these are deterministic for a given platform):
    # the approaching band must produce a decisive signal by 60 min AND at
    # least one genuinely fractional member split — mid-range fractions are
    # the strong version of the (a) equality below.
    p60 = float(products.p_rain[60][row, col])
    assert p60 >= 0.5, f"fixture premise broken: p_rain[60] at home = {p60}"
    home_ps = [float(products.p_rain[lead][row, col]) for lead in products.leads_min]
    assert any(0.0 < p < 1.0 for p in home_ps), (
        f"fixture premise broken: no fractional home probability in {home_ps}"
    )

    # --- (a) per-lead p_ensemble == national p_rain at the home pixel ------
    shared = [e for e in state.forecast.per_lead if e.lead_min in products.p_rain]
    assert [e.lead_min for e in shared] == [10, 20, 30, 45, 60]
    for entry in shared:
        assert entry.p_ensemble is not None
        grid_val = float(products.p_rain[entry.lead_min][row, col])
        # Exact equality expected: same ensemble, same threshold, same
        # frame-age-corrected timestep buckets, and the 1-km home disc on
        # the ×4 grid is exactly the rounded pixel. n/8 fractions are
        # dyadic, so float32 vs float64 cannot differ.
        assert entry.p_ensemble == pytest.approx(grid_val, abs=1e-7), (
            f"lead {entry.lead_min}: home {entry.p_ensemble} != grid {grid_val}"
        )

    # --- (b) national ETA at home vs the home ETA window -------------------
    eta_home = float(products.eta_min[row, col])
    window = state.forecast.eta_p50_window_min
    assert state.probabilistic.eta_p50_window_min == window
    if math.isfinite(eta_home):
        # ≥ half the members cross at home ⇒ the P25–P75 window exists and
        # the step-median ETA lies within it, ± one timestep of slack for
        # np.percentile's interpolation between timestep boundaries.
        assert window is not None
        lo, hi = window
        assert lo <= hi
        assert lo - products.timestep_min <= eta_home <= hi + products.timestep_min, (
            f"eta {eta_home} outside window {window} ± {products.timestep_min}"
        )
    else:
        # Minority-member caveat (national.py docstring): the national map
        # is conservatively NaN when fewer than half the members ever
        # cross, while the home window may still summarise that minority.
        assert p60 < 0.5
    # This fixture was chosen to exercise the finite branch — make the
    # choice load-bearing rather than silently falling into the other one.
    assert math.isfinite(eta_home)
    assert 0.0 <= eta_home <= 60.0

    # Intensity is defined exactly where ETA is.
    intensity_home = float(products.intensity_mm_h[row, col])
    assert math.isfinite(intensity_home)
    assert 0.0 <= intensity_home <= 100.0  # Z–R cap contract

    # --- (c) PNG round-trip through the written artifacts ------------------
    nowcast_dir = minimal_config.storage.data_dir / "nowcast"
    manifest = json.loads((nowcast_dir / "manifest.json").read_text())
    assert manifest["n_members"] == 8
    assert manifest["frame_age_min"] == pytest.approx(FRAME_AGE_MIN, abs=0.01)
    assert datetime.fromisoformat(manifest["radar_ts_utc"]) == newest_ts
    grids = _manifest_grid_entries(manifest)
    for lead in products.leads_min:
        _assert_png_roundtrip(
            nowcast_dir, grids[("p_rain", lead)], products.p_rain[lead], (row, col),
        )
    _assert_png_roundtrip(nowcast_dir, grids[("eta", None)],
                          products.eta_min, (row, col))
    _assert_png_roundtrip(nowcast_dir, grids[("intensity", None)],
                          products.intensity_mm_h, (row, col))
    # The ETA/intensity grids on real data have both finite and NaN pixels,
    # so the NaN-mask assertions above were not vacuous.
    assert np.isnan(products.eta_min).any()
    assert np.isfinite(products.eta_min).any()

    # --- (d) end-to-end HTTP: /forecast at home == the state's home block --
    engine.store.write(state)  # what run_cycle would do; feeds confidence
    app = create_app(minimal_config, engine=engine, auto_start_scheduler=False)
    with TestClient(app) as client:
        r = client.get(
            "/forecast",
            params={"lat": minimal_config.home.lat, "lon": minimal_config.home.lon},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n_members"] == 8
        assert body["calibrated"] is False
        assert datetime.fromisoformat(body["radar_ts_utc"]) == newest_ts

        by_lead = {e.lead_min: e.p_ensemble for e in state.forecast.per_lead}
        assert [pl["lead_min"] for pl in body["per_lead"]] == [10, 20, 30, 45, 60]
        for pl in body["per_lead"]:
            assert pl["p_rain"] == pytest.approx(
                by_lead[pl["lead_min"]], abs=1e-7,
            ), f"/forecast lead {pl['lead_min']} disagrees with state.json"

        assert body["eta_min"] == pytest.approx(eta_home, abs=1e-6)
        assert body["intensity_mm_h"] == pytest.approx(intensity_home, abs=1e-6)
        assert body["confidence"] == pytest.approx(state.confidence)

        # And the served manifest is the one we validated in (c).
        rm = client.get("/nowcast/manifest.json")
        assert rm.status_code == 200
        assert rm.json() == manifest


# ---------------------------------------------------------------------------
# Dry triple (the plan-named frames) — None-window / NaN-ETA semantics
# ---------------------------------------------------------------------------

def test_dry_archive_cycle_agrees_at_zero(
    minimal_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Home is dry and stays dry: agreement must hold at p = 0 with a None
    window and NaN ETA — the branch §A4(b) defines for 'no rain in horizon'."""
    paths = _archive_paths(DRY_TRIPLE)
    # Runtime trim: each deterministic lead costs a native-grid advect_field
    # + an overlay PNG, and the wet test already runs the full lead list.
    # The ensemble / national path under test here is unaffected; {10, 60}
    # keeps a shared lead at each end of the horizon for the agreement.
    minimal_config.forecast.leads_min = [10, 60]
    engine = _build_engine(minimal_config, monkeypatch)
    state, newest_ts = _run_cycle(engine, paths, monkeypatch)

    assert state.radar.data_age_minutes == pytest.approx(FRAME_AGE_MIN, abs=0.01)
    assert state.probabilistic is not None, "STEPS ensemble did not run"
    latest = engine.national_latest
    assert latest is not None, "national products were not computed"
    products, radar_ts = latest
    assert radar_ts == newest_ts
    row, col = _home_product_pixel(engine)

    # (a) at zero: no member brings rain to the home pixel within the horizon
    # (nearest echo is > 60 km out), on both sides of the agreement.
    for entry in state.forecast.per_lead:
        if entry.lead_min in products.p_rain:
            grid_val = float(products.p_rain[entry.lead_min][row, col])
            assert entry.p_ensemble == pytest.approx(grid_val, abs=1e-7)
            assert entry.p_ensemble == 0.0
        # Deterministic fields keep their exact semantics alongside.
        assert entry.p_rain == 0.0

    # (b) None-window branch: no member crossed in the home disc, so the
    # window is None AND the pixel's exceedance never reaches 0.5 (ETA NaN).
    assert state.forecast.eta_p50_window_min is None
    assert state.probabilistic.eta_p50_window_min is None
    assert math.isnan(float(products.eta_min[row, col]))
    assert math.isnan(float(products.intensity_mm_h[row, col]))

    # (d)-light: the HTTP surface reports the same nothing-coming forecast.
    engine.store.write(state)
    app = create_app(minimal_config, engine=engine, auto_start_scheduler=False)
    with TestClient(app) as client:
        r = client.get(
            "/forecast",
            params={"lat": minimal_config.home.lat, "lon": minimal_config.home.lon},
        )
        assert r.status_code == 200
        body = r.json()
        assert [pl["p_rain"] for pl in body["per_lead"]] == [0.0] * 5
        assert body["eta_min"] is None
        assert body["intensity_mm_h"] is None


# ---------------------------------------------------------------------------
# Live-validation script — offline smoke only
# ---------------------------------------------------------------------------

def test_validate_national_script_help_runs_offline() -> None:
    """The post-deploy script must at least parse + show --help without any
    network access (the real run happens against a deployed instance — see
    the script docstring). Guards against import/syntax rot in CI."""
    import subprocess
    import sys

    script = ARCHIVE_DIR.parent / "scripts" / "validate_national.py"
    assert script.is_file()
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--base-url" in result.stdout
    assert "--lat" in result.stdout and "--lon" in result.stdout
    assert "--token" in result.stdout
