"""H-P serving, part 2 — ``/forecast`` and ``state.json`` carry the model.

The push path started deciding on the gauge-trained post-processed
probability on 2026-09-11 (``test_push_postprocess.py``). What a
subscriber is *told* and what the website *shows* were then two different
numbers, which is the one thing this project is not allowed to do: the
panel would draw the curve-calibrated probability under a notification
fired on the model's.

So both public read paths carry it now, and the seams are here:

1. **A point the cycle already scored reads the cycle's own number.** Not
   a recomputation of it — the very array the fan-out decided on. A
   subscriber comparing the notification with the panel must find them
   equal, and "equal because it is the same object" is the only version
   of that which cannot drift.
2. **A point the cycle never heard of gets the same answer anyway.** The
   website serves whatever pixel a visitor clicks. The on-demand path
   assembles the feature row off the cycle's retained grids, and this
   suite pins it against the cycle's table for a point that appears in
   both — the on-demand row and the published row must agree to the bit.
3. **The raw fractions, not the served ones.** The model was fitted on
   the UNcalibrated ensemble fraction. After ``_calibrate_national`` the
   served grid is a different number at the same pixel, and reading it by
   mistake is invisible in the output — so the fixture makes the two
   differ and the test would catch it.
4. **No model is the old behaviour, exactly.** ``p_post`` null,
   ``probability_source`` ``curve``, ``p_rain`` untouched.
5. **``state.json`` is additive.** ``p_calibrated`` keeps its value and
   its meaning, because a Home Assistant install is pinned to it.

Synthetic throughout: no radar, no STEPS, no network.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("pyarrow")

from dmi_nowcast_core import postprocess as pp
from dmi_nowcast_core.geo import GridIndex
from dmi_nowcast_core.national import NationalProducts
from dmi_nowcast_core.product_pairs import nearest_radar_km
from dmi_nowcast_sidecar import compute as compute_mod
from dmi_nowcast_sidecar.app import create_app
from dmi_nowcast_sidecar.compute import CycleEngine, NationalSnapshot
from dmi_nowcast_sidecar.config import Config
from dmi_nowcast_sidecar.push.paths import resolved_postprocess_path
from dmi_nowcast_sidecar.push.postprocess import (
    PostprocessContext,
    PostprocessTable,
    build_cycle_postprocess,
    point_key,
)

RADAR_TS = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)
GENERATED_AT = RADAR_TS + timedelta(minutes=14)
FRAME_AGE_MIN = 14.0
LEADS = (10, 20, 30, 45, 60)

GRID_PX = 64
PIXEL_KM = 0.5
DT_MIN = 10.0
DOWNSAMPLE = 4
GRID_DS = GRID_PX // DOWNSAMPLE

#: The linear geo below is centred here; every test point is an offset.
LAT0, LON0 = 56.0, 10.0
#: Degrees per native pixel. Small enough that the whole 64-px grid is one
#: neighbourhood, which is all the projection has to be for these tests.
DEG_PER_PX = 0.004

#: Two points the cycle scores, and one it does not. The third is what a
#: website visitor clicking an arbitrary pixel looks like.
IN_CYCLE: tuple[tuple[float, float], ...] = (
    (LAT0 - 20 * DEG_PER_PX, LON0 + 20 * DEG_PER_PX),   # "home"
    (LAT0 - 32 * DEG_PER_PX, LON0 + 36 * DEG_PER_PX),   # a subscription
)
OFF_CYCLE = (LAT0 - 26 * DEG_PER_PX, LON0 + 28 * DEG_PER_PX)


class LinearGeo:
    """``(lon, lat)`` → a fractional native index, linear and total.

    Total is the point: ``/forecast`` is asked about coordinates nobody
    enumerated in advance, so a fixture geo that only knows three places
    would pass the on-demand test for the wrong reason.
    """

    def lonlat_to_grid(self, lon: float, lat: float) -> GridIndex:
        return GridIndex(
            row=(LAT0 - float(lat)) / DEG_PER_PX,
            col=(float(lon) - LON0) / DEG_PER_PX,
        )


# ---------------------------------------------------------------------------
# One cycle, by hand
# ---------------------------------------------------------------------------


def _rain_field() -> np.ndarray:
    """A diagonal band of rain, dry elsewhere, NaN off the composite."""
    field = np.zeros((GRID_PX, GRID_PX), dtype=np.float32)
    rows, cols = np.mgrid[0:GRID_PX, 0:GRID_PX]
    field[(rows + cols > 20) & (rows + cols < 44)] = 3.5
    field[rows > 58] = np.nan
    return field


def _flow() -> tuple[np.ndarray, np.ndarray]:
    vy = np.full((GRID_PX, GRID_PX), 2.0, dtype=np.float32)
    vx = np.full((GRID_PX, GRID_PX), 1.5, dtype=np.float32)
    return vy, vx


def _ds_grid(value: float) -> np.ndarray:
    return np.full((GRID_DS, GRID_DS), value, dtype=np.float32)


def _raw_grids() -> dict[int, np.ndarray]:
    """The UNcalibrated fractions, one distinct value per lead."""
    return {lead: _ds_grid(0.30 + 0.07 * i) for i, lead in enumerate(LEADS)}


def _products() -> NationalProducts:
    """The SERVED products — ``p_rain`` deliberately unlike the raw grids.

    A calibrated grid is what the endpoint publishes as ``p_rain`` and it
    is NOT what the model reads. Making the two differ by a wide margin is
    what turns "the on-demand path read the wrong grid" from an invisible
    bug into a failing assertion.
    """
    return NationalProducts(
        p_rain={lead: _ds_grid(0.90 - 0.05 * i) for i, lead in enumerate(LEADS)},
        eta_min=_ds_grid(25.0),
        intensity_mm_h=_ds_grid(1.8),
        leads_min=LEADS,
        threshold_mm_h=0.5,
        timestep_min=DT_MIN,
        frame_age_min=FRAME_AGE_MIN,
        downsample_factor=DOWNSAMPLE,
        n_members=16,
    )


def _observed() -> np.ndarray:
    return _ds_grid(0.2)


def _model() -> pp.PostprocessModel:
    """A model fitted on noise: only its SHAPE matters to the serving path."""
    rng = np.random.default_rng(11)
    n = 400
    signal = rng.random(n)
    rows: dict[str, object] = {
        name: rng.random(n) * 5.0
        for name in pp.DESIGN_SOURCE_COLUMNS if name != "hour_utc"
    }
    for lead in LEADS:
        rows[pp.raw_fraction_column(lead)] = signal
    rows["hour_utc"] = rng.integers(0, 24, n).astype(float)
    rows["season"] = np.where(signal > 0.5, "summer", "winter").astype("<U8")
    truth = {
        lead: ((signal > 0.5).astype(float), np.ones(n, dtype=bool))
        for lead in LEADS
    }
    return pp.fit_postprocess(
        rows, truth, LEADS, l2=1.0, design_leads=LEADS,
        fitted_at=datetime(2026, 9, 11, 3, 40, tzinfo=timezone.utc),
    )


def _table(config: Config, *, fitted: bool = True) -> PostprocessTable:
    path = resolved_postprocess_path(config)
    if fitted:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_model().dumps())
    table = PostprocessTable(path)
    table.load()
    return table


def _cycle_objects(
    table: PostprocessTable, keys: tuple[tuple[float, float], ...],
):
    """``(CyclePostprocess, PostprocessContext)`` for ``keys``.

    Assembled exactly as ``CycleEngine._publish_postprocess`` does, from
    the same helpers — including ``_read_points``, so the raw fractions
    come off the raw grids at the product pixel the endpoint would read.
    """
    products = _products()
    raw = _raw_grids()
    geo = LinearGeo()
    field = _rain_field()
    vy, vx = _flow()
    native = [geo.lonlat_to_grid(lon, lat) for lat, lon in keys]
    # The raw products, as they exist for the one moment before
    # ``_calibrate_national`` replaces the grids.
    raw_products = NationalProducts(
        p_rain=raw,
        eta_min=products.eta_min,
        intensity_mm_h=products.intensity_mm_h,
        leads_min=products.leads_min,
        threshold_mm_h=products.threshold_mm_h,
        timestep_min=products.timestep_min,
        frame_age_min=products.frame_age_min,
        downsample_factor=products.downsample_factor,
        n_members=products.n_members,
    )
    point_products = compute_mod._read_points(
        raw_products, native, keep_grids=True,
    )
    assert point_products is not None
    observed = _observed()
    shared = []
    for pixel in point_products.pixels:
        assert pixel is not None
        row, col = pixel
        shared.append({
            "observed_mm_h": float(observed[row, col]),
            "eta_min": float(products.eta_min[row, col]),
            "intensity_mm_h": float(products.intensity_mm_h[row, col]),
        })
    grid_features = pp.station_features(
        field, vy, vx,
        np.array([idx.row for idx in native], dtype=np.float64),
        np.array([idx.col for idx in native], dtype=np.float64),
        pixel_km=PIXEL_KM, dt_min=DT_MIN,
        bulk_vy=2.0, bulk_vx=1.5, stalled_share=0.011,
    )
    cycle = build_cycle_postprocess(
        table,
        radar_ts_utc=RADAR_TS,
        generated_at_utc=GENERATED_AT,
        keys=[point_key(lat, lon) for lat, lon in keys],
        grid_features=grid_features,
        raw_fractions=point_products.raw_fractions,
        shared=shared,
        station_radar_km=[
            float(nearest_radar_km(lat, lon)) for lat, lon in keys
        ],
        leads=products.leads_min,
        season=pp.season_of_month(GENERATED_AT.month),
        hour_utc=GENERATED_AT.hour,
        frame_age_min=FRAME_AGE_MIN,
    )
    context = PostprocessContext(
        radar_ts_utc=RADAR_TS,
        generated_at_utc=GENERATED_AT,
        rain_mm_h=field,
        vy=vy,
        vx=vx,
        raw_fraction_grids=dict(point_products.raw_grids or {}),
        leads=tuple(products.leads_min),
        pixel_km=PIXEL_KM,
        dt_min=DT_MIN,
        bulk_vy=2.0,
        bulk_vx=1.5,
        stalled_share=0.011,
        season=pp.season_of_month(GENERATED_AT.month),
        hour_utc=GENERATED_AT.hour,
        frame_age_min=FRAME_AGE_MIN,
    )
    return cycle, context, products


def _engine(
    config: Config,
    *,
    fitted: bool = True,
    keys: tuple[tuple[float, float], ...] = IN_CYCLE,
) -> CycleEngine:
    engine = CycleEngine(config)
    engine._basemap_attempted = True
    engine._geo = LinearGeo()  # type: ignore[assignment]
    table = _table(config, fitted=fitted)
    engine._postprocess = table
    cycle, context, products = _cycle_objects(table, keys)
    engine._national_latest = NationalSnapshot(
        products, RADAR_TS, _observed(), None, GENERATED_AT,
    )
    engine._postprocess_latest = cycle
    engine._postprocess_context = context
    return engine


def _client(config: Config, engine: CycleEngine) -> TestClient:
    return TestClient(
        create_app(config, engine=engine, auto_start_scheduler=False),
    )


def _per_lead(body: dict) -> dict[int, float | None]:
    return {int(e["lead_min"]): e["p_post"] for e in body["per_lead"]}


# ---------------------------------------------------------------------------
# 1. A point the cycle scored
# ---------------------------------------------------------------------------


class TestPointTheCycleScored:
    def test_forecast_serves_the_cycles_own_number(
        self, minimal_config: Config,
    ) -> None:
        """The panel and the notification read one array, not two."""
        engine = _engine(minimal_config)
        lat, lon = IN_CYCLE[1]
        with _client(minimal_config, engine) as client:
            body = client.get(
                "/forecast", params={"lat": lat, "lon": lon},
            ).json()
        served = _per_lead(body)
        assert body["probability_source"] == "postprocess"
        assert body["postprocess_fitted_at_utc"] is not None
        assert set(served) == set(LEADS)
        for lead in LEADS:
            expected = engine._postprocess_latest.probability(lat, lon, lead)
            assert expected is not None
            assert served[lead] == pytest.approx(expected)

    def test_the_reuse_is_a_lookup_not_a_recomputation(
        self, minimal_config: Config,
    ) -> None:
        """Dropping the grids must not change a served point's answer.

        The grids are ~48 MB the process holds only for the on-demand
        path. A point the cycle already scored has to be answerable
        without them, or a deployment that failed to retain them would
        quietly start serving the curve to its own subscribers.
        """
        engine = _engine(minimal_config)
        lat, lon = IN_CYCLE[0]
        before = engine.postprocess_point(lat, lon)
        engine._postprocess_context = None
        after = engine.postprocess_point(lat, lon)
        assert before is not None and after is not None
        assert before.reused and after.reused
        assert after.p_post == before.p_post

    def test_p_rain_is_untouched_by_the_model(
        self, minimal_config: Config,
    ) -> None:
        """``p_rain`` stays the served, curve-calibrated grid sample."""
        engine = _engine(minimal_config)
        lat, lon = IN_CYCLE[0]
        with _client(minimal_config, engine) as client:
            body = client.get(
                "/forecast", params={"lat": lat, "lon": lon},
            ).json()
        served = {int(e["lead_min"]): e["p_rain"] for e in body["per_lead"]}
        for index, lead in enumerate(LEADS):
            assert served[lead] == pytest.approx(0.90 - 0.05 * index)


# ---------------------------------------------------------------------------
# 2. A point the cycle never heard of
# ---------------------------------------------------------------------------


class TestPointComputedOnDemand:
    def test_an_arbitrary_point_is_scored_off_the_retained_grids(
        self, minimal_config: Config,
    ) -> None:
        engine = _engine(minimal_config)
        lat, lon = OFF_CYCLE
        assert engine._postprocess_latest.index_of(lat, lon) is None
        with _client(minimal_config, engine) as client:
            body = client.get(
                "/forecast", params={"lat": lat, "lon": lon},
            ).json()
        served = _per_lead(body)
        assert body["probability_source"] == "postprocess"
        assert set(served) == set(LEADS)
        assert all(value is not None for value in served.values())
        assert all(0.0 <= float(value) <= 1.0 for value in served.values())

    def test_the_on_demand_answer_equals_the_cycles_for_the_same_point(
        self, minimal_config: Config,
    ) -> None:
        """The seam that matters: two assemblies, one number.

        ``OFF_CYCLE`` is scored twice — once by a cycle that happens to
        carry it, once on demand by a cycle that does not. Same grids,
        same flow, same instant, so the same feature row and the same
        probability. Any drift between the two code paths lands here.
        """
        table = _table(minimal_config)
        published, _ctx, _products = _cycle_objects(
            table, IN_CYCLE + (OFF_CYCLE,),
        )
        engine = _engine(minimal_config)  # a cycle WITHOUT that point
        lat, lon = OFF_CYCLE
        on_demand = engine.postprocess_point(lat, lon)
        assert on_demand is not None and not on_demand.reused
        for lead in LEADS:
            expected = published.probability(lat, lon, lead)
            assert expected is not None
            assert on_demand.p_post[lead] == pytest.approx(expected)

    def test_a_stale_context_is_refused(self, minimal_config: Config) -> None:
        """A context from another frame is not evidence about this one."""
        engine = _engine(minimal_config)
        engine._postprocess_context = PostprocessContext(
            **{
                **engine._postprocess_context.__dict__,
                "radar_ts_utc": RADAR_TS - timedelta(minutes=10),
            },
        )
        assert engine.postprocess_point(*OFF_CYCLE) is None

    def test_a_scoring_failure_costs_the_field_not_the_response(
        self, minimal_config: Config,
    ) -> None:
        """A broken grid must leave ``/forecast`` serving the curve."""
        engine = _engine(minimal_config)
        engine._postprocess_context = PostprocessContext(
            **{**engine._postprocess_context.__dict__, "rain_mm_h": np.zeros(3)},
        )
        lat, lon = OFF_CYCLE
        with _client(minimal_config, engine) as client:
            response = client.get("/forecast", params={"lat": lat, "lon": lon})
        assert response.status_code == 200
        body = response.json()
        assert body["probability_source"] == "curve"
        assert all(value is None for value in _per_lead(body).values())
        assert all(e["p_rain"] is not None for e in body["per_lead"])


# ---------------------------------------------------------------------------
# 3. No model at all
# ---------------------------------------------------------------------------


class TestWithoutAModel:
    def test_forecast_is_exactly_what_it_was_before_phase_h(
        self, minimal_config: Config,
    ) -> None:
        engine = _engine(minimal_config, fitted=False)
        lat, lon = IN_CYCLE[0]
        with _client(minimal_config, engine) as client:
            body = client.get(
                "/forecast", params={"lat": lat, "lon": lon},
            ).json()
        assert body["probability_source"] == "curve"
        assert body["postprocess_fitted_at_utc"] is None
        assert all(value is None for value in _per_lead(body).values())
        assert all(e["p_rain"] is not None for e in body["per_lead"])

    def test_the_engine_holds_no_grids_it_cannot_use(
        self, minimal_config: Config,
    ) -> None:
        """``keep_grids`` follows the model, not the cycle.

        Without a model the raw grids are ~7 MB and the native fields
        ~41 MB held for a question nobody can ask.
        """
        products = _products()
        native = [LinearGeo().lonlat_to_grid(lon, lat) for lat, lon in IN_CYCLE]
        assert compute_mod._read_points(
            products, native, keep_grids=False,
        ).raw_grids is None


# ---------------------------------------------------------------------------
# 4. state.json
# ---------------------------------------------------------------------------


class TestStateSchema:
    def test_p_post_rides_beside_p_calibrated_without_touching_it(self) -> None:
        from dmi_nowcast_sidecar.state_schema import ForecastBlock, PerLeadEntry

        entry = PerLeadEntry(
            lead_min=20, rain_rate_mm_h=1.2, p_rain=1.0, p_calibrated=0.61,
        )
        # Additive: a caller that knows nothing about the model still
        # builds a valid entry, and the HA contract field is unchanged.
        assert entry.p_post is None
        assert entry.p_calibrated == 0.61
        scored = entry.model_copy(update={"p_post": 0.42})
        assert scored.p_calibrated == 0.61 and scored.p_post == 0.42

        block = ForecastBlock(
            method="farneback", rain_incoming=True, eta_minutes=12.0,
            eta_p50_window_min=None, peak_intensity_mm_h=2.4, peak_lead_min=20,
            per_lead=[scored],
        )
        assert block.probability_source == "curve"
        assert block.model_dump()["per_lead"][0]["p_post"] == 0.42

    def test_probability_source_is_a_closed_set(self) -> None:
        from pydantic import ValidationError

        from dmi_nowcast_sidecar.state_schema import ForecastBlock

        with pytest.raises(ValidationError):
            ForecastBlock(
                method="farneback", rain_incoming=False, eta_minutes=None,
                eta_p50_window_min=None, peak_intensity_mm_h=0.0,
                peak_lead_min=0, per_lead=[],
                probability_source="gauge",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# 5. A whole cycle, end to end
# ---------------------------------------------------------------------------


class TestAWholeCycle:
    """The wiring, exercised by a real ``_compute_sync``.

    Everything above builds the published objects by hand, which pins the
    two read paths but says nothing about whether the cycle actually fills
    them. This runs the cycle — a synthetic composite, a stand-in ensemble,
    no rendering — and reads the state and the endpoint the way a consumer
    would.
    """

    @pytest.fixture
    def cycled(self, minimal_config: Config, monkeypatch: pytest.MonkeyPatch):
        from tests.test_national_calibration import (
            HOME_LAT,
            HOME_LEADS,
            HOME_LON,
            _make_engine,
            _write_composite,
        )

        resolved = resolved_postprocess_path(minimal_config)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(_model().dumps())
        engine = _make_engine(minimal_config, monkeypatch)
        newest = datetime.now(timezone.utc) - timedelta(minutes=4)
        paths = []
        for index, dbz in enumerate((30.0, 30.5, 31.0)):
            path = minimal_config.storage.data_dir / f"cycle_{index}.h5"
            _write_composite(path, newest - timedelta(minutes=5 * (2 - index)), dbz)
            paths.append(path)
        state = engine._compute_sync(paths, fetch_ms=0.0)
        assert state is not None
        return engine, state, (HOME_LAT, HOME_LON), HOME_LEADS

    def test_state_json_carries_the_home_points_model_probability(
        self, cycled,
    ) -> None:
        _engine_, state, _home, home_leads = cycled
        assert state.forecast.probability_source == "postprocess"
        scored = {
            entry.lead_min: entry.p_post for entry in state.forecast.per_lead
        }
        # The national products publish (10, 20, 30, 45, 60); the home leads
        # are (5, 10, 20, 30, 60). A lead the model was not fitted for is
        # null rather than borrowed from its neighbour.
        assert scored[10] is not None and scored[5] is None
        assert set(scored) == set(home_leads)
        for value in scored.values():
            assert value is None or 0.0 <= value <= 1.0

    def test_p_calibrated_is_exactly_what_it_was(self, cycled) -> None:
        """The HA contract field does not move because a model appeared."""
        _engine_, state, _home, _leads = cycled
        for entry in state.forecast.per_lead:
            # These composites are uniformly wet, so the deterministic
            # Yes/No probability is 1.0 and, with no curves loaded, the
            # calibrated twin is the same number — as it was pre-Phase-H.
            assert entry.p_rain == pytest.approx(1.0)
            assert entry.p_calibrated == pytest.approx(1.0)

    def test_forecast_at_home_reuses_that_very_number(self, cycled) -> None:
        """One cycle, one answer, whichever door it is read through."""
        engine, state, (lat, lon), _leads = cycled
        point = engine.postprocess_point(lat, lon)
        assert point is not None and point.reused
        for entry in state.forecast.per_lead:
            if entry.p_post is None:
                continue
            assert point.p_post[entry.lead_min] == pytest.approx(entry.p_post)
