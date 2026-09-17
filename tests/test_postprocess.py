"""Gauge-trained post-processing (Phase H, H-P) — features, model, LOMO.

Three things are pinned here, and they fail for different reasons:

* **the geometry** — the upwind corridor and the observation disc, on
  hand-planted fields where the right answer is arithmetic, not a
  regression baseline;
* **the transform** — every log, cap, indicator and one-hot, including
  what happens to a column that is missing outright;
* **the model** — that a fit on a known generating process recovers the
  signs and rough magnitudes, that the leave-one-month-out evaluation
  beats a *generously* fitted curve baseline when the baseline's single
  predictor saturates, and that the JSON artefact round-trips to the same
  predictions.

Everything is synthetic and offline. No radar, no STEPS, no gauges.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from dmi_nowcast_core import postprocess as pp
from dmi_nowcast_core.benchmark import reliability_bins
from dmi_nowcast_core.calibrate import fit_isotonic

PIXEL_KM = 0.5
GRID = 200


# ---------------------------------------------------------------------------
# Season / hour / month keys
# ---------------------------------------------------------------------------


def test_the_season_split_partitions_the_year() -> None:
    months = sorted(m for name in pp.SEASONS for m in pp.SEASON_MONTHS[name])
    assert months == list(range(1, 13))
    assert pp.season_of_month(7) == "summer"
    assert pp.season_of_month(1) == "winter"
    assert pp.season_of_month(4) == "shoulder"
    with pytest.raises(ValueError, match="belongs to no season"):
        pp.season_of_month(13)


def test_season_hour_and_month_come_off_the_decision_instant() -> None:
    # 2026-04-15T23:30Z and 2026-12-01T00:05Z.
    t = np.array([1776295800, 1796083500], dtype=np.int64)
    assert list(pp.seasons_from_epoch(t)) == ["shoulder", "winter"]
    assert list(pp.hours_from_epoch(t)) == [23, 0]
    assert list(pp.months_from_epoch(t)) == [4, 12]
    # Folds are (year, month) pairs, so two Decembers are two folds.
    keys = pp.year_months_from_epoch(
        np.array([1796083500, 1796083500 - 365 * 86400], dtype=np.int64)
    )
    assert keys[0] != keys[1]
    assert pp.month_labels(keys) == ["2026-12", "2025-12"]


# ---------------------------------------------------------------------------
# Grid features
# ---------------------------------------------------------------------------


def _empty_field() -> np.ndarray:
    return np.zeros((GRID, GRID), dtype=np.float32)


def _blob(field: np.ndarray, row: int, col: int, value: float, half: int = 2) -> None:
    field[row - half:row + half + 1, col - half:col + half + 1] = value


def test_disc_max_reads_a_radius_not_a_pixel() -> None:
    field = _empty_field()
    # 4 km east of the station: inside a 5 km disc, outside a 3 km one.
    _blob(field, 100, 100 + int(4.0 / PIXEL_KM), 7.0, half=0)
    inside = pp.disc_max(
        field, np.array([100.0]), np.array([100.0]),
        radius_km=5.0, pixel_km=PIXEL_KM,
    )
    outside = pp.disc_max(
        field, np.array([100.0]), np.array([100.0]),
        radius_km=3.0, pixel_km=PIXEL_KM,
    )
    assert inside[0] == pytest.approx(7.0)
    assert outside[0] == pytest.approx(0.0)


def test_disc_max_off_the_grid_is_nan_not_zero() -> None:
    field = _empty_field()
    out = pp.disc_max(
        field, np.array([-500.0]), np.array([-500.0]),
        radius_km=5.0, pixel_km=PIXEL_KM,
    )
    assert math.isnan(float(out[0]))


def _features_with_flow(
    field: np.ndarray, vy: float, vx: float, row: float = 100.0, col: float = 100.0,
) -> dict[str, np.ndarray]:
    """``station_features`` for one station under a uniform flow."""
    return pp.station_features(
        field,
        np.full(field.shape, vy, dtype=np.float32),
        np.full(field.shape, vx, dtype=np.float32),
        np.array([row]), np.array([col]),
        pixel_km=PIXEL_KM, dt_min=10.0,
        bulk_vy=vy, bulk_vx=vx, stalled_share=0.25,
    )


class TestUpstreamCorridor:
    """The corridor is upwind, 6 km wide, and stops at 40 km."""

    #: 2 px per frame eastward on a 500 m grid at 10-minute spacing is
    #: 1 km per 10 min = 6 km/h, and the rain is heading toward 090°.
    EAST = (0.0, 2.0)

    def test_it_looks_upwind_and_ignores_downwind(self) -> None:
        field = _empty_field()
        # 15 km upwind (west) of the station, and a bigger cell downwind.
        _blob(field, 100, 100 - int(15.0 / PIXEL_KM), 4.0)
        _blob(field, 100, 100 + int(15.0 / PIXEL_KM), 40.0)
        out = _features_with_flow(field, *self.EAST)
        assert out["up_max_40km_mm_h"][0] == pytest.approx(4.0)
        assert out["up_max_20km_mm_h"][0] == pytest.approx(4.0)
        assert out["up_dist_km"][0] == pytest.approx(14.0, abs=0.6)

    def test_the_near_window_stops_at_twenty_kilometres(self) -> None:
        field = _empty_field()
        _blob(field, 100, 100 - int(30.0 / PIXEL_KM), 9.0)
        out = _features_with_flow(field, *self.EAST)
        assert out["up_max_40km_mm_h"][0] == pytest.approx(9.0)
        assert out["up_max_20km_mm_h"][0] == pytest.approx(0.0)
        assert out["up_dist_km"][0] == pytest.approx(29.0, abs=0.6)

    def test_it_stops_at_forty_kilometres(self) -> None:
        field = _empty_field()
        _blob(field, 100, 100 - int(45.0 / PIXEL_KM), 9.0)
        out = _features_with_flow(field, *self.EAST)
        assert out["up_max_40km_mm_h"][0] == pytest.approx(0.0)
        # Nothing wet inside the corridor: "no echo upwind", not a big number.
        assert math.isnan(float(out["up_dist_km"][0]))

    def test_it_is_six_kilometres_wide(self) -> None:
        field = _empty_field()
        # 15 km upwind but 5 km across the flow — outside the ± 3 km corridor.
        _blob(field, 100 - int(5.0 / PIXEL_KM), 100 - int(15.0 / PIXEL_KM), 9.0, half=1)
        assert _features_with_flow(field, *self.EAST)["up_max_40km_mm_h"][0] == (
            pytest.approx(0.0)
        )
        # 2 km across is inside it.
        field = _empty_field()
        _blob(field, 100 - int(2.0 / PIXEL_KM), 100 - int(15.0 / PIXEL_KM), 9.0, half=1)
        assert _features_with_flow(field, *self.EAST)["up_max_40km_mm_h"][0] == (
            pytest.approx(9.0)
        )

    def test_it_follows_the_flow_direction(self) -> None:
        field = _empty_field()
        # A cell to the SOUTH; only a northward flow puts it upwind.
        _blob(field, 100 + int(15.0 / PIXEL_KM), 100, 6.0)
        northward = _features_with_flow(field, -2.0, 0.0)
        eastward = _features_with_flow(field, *self.EAST)
        assert northward["up_max_40km_mm_h"][0] == pytest.approx(6.0)
        assert eastward["up_max_40km_mm_h"][0] == pytest.approx(0.0)

    def test_a_station_under_rain_reads_zero_distance(self) -> None:
        field = _empty_field()
        _blob(field, 100, 100, 3.0)
        out = _features_with_flow(field, *self.EAST)
        assert out["up_dist_km"][0] == pytest.approx(0.0)

    def test_the_wet_fraction_counts_the_whole_corridor(self) -> None:
        field = np.full((GRID, GRID), 2.0, dtype=np.float32)
        soaked = _features_with_flow(field, *self.EAST)
        assert soaked["up_wet_frac_40km"][0] == pytest.approx(1.0)
        assert _features_with_flow(_empty_field(), *self.EAST)[
            "up_wet_frac_40km"
        ][0] == pytest.approx(0.0)

    def test_no_usable_flow_leaves_the_corridor_undefined(self) -> None:
        field = np.full((GRID, GRID), 5.0, dtype=np.float32)
        out = _features_with_flow(field, 0.0, 0.0)
        for name in (
            "up_max_20km_mm_h", "up_max_40km_mm_h", "up_dist_km",
            "up_wet_frac_40km",
        ):
            assert math.isnan(float(out[name][0])), name
        # The observation disc still works — it needs no direction.
        assert out["obs_max_5km_mm_h"][0] == pytest.approx(5.0)

    def test_a_stalled_local_flow_falls_back_to_the_bulk(self) -> None:
        field = _empty_field()
        _blob(field, 100, 100 - int(15.0 / PIXEL_KM), 4.0)
        out = pp.station_features(
            field,
            np.zeros(field.shape, dtype=np.float32),
            np.zeros(field.shape, dtype=np.float32),
            np.array([100.0]), np.array([100.0]),
            pixel_km=PIXEL_KM, dt_min=10.0,
            bulk_vy=0.0, bulk_vx=2.0, stalled_share=1.0,
        )
        assert out["up_max_40km_mm_h"][0] == pytest.approx(4.0)
        assert out["local_speed_kmh"][0] == pytest.approx(0.0)


class TestMotionFeatures:
    """Speeds in km/h and a compass bearing the rain heads toward."""

    def test_speed_uses_the_frames_own_pixel_and_spacing(self) -> None:
        out = _features_with_flow(_empty_field(), 0.0, 2.0)
        # 2 px × 0.5 km per 10 min = 6 km/h.
        assert out["bulk_kmh"][0] == pytest.approx(6.0)
        assert out["local_speed_kmh"][0] == pytest.approx(6.0)

    @pytest.mark.parametrize(
        "vy,vx,bearing",
        [(-1.0, 0.0, 0.0), (0.0, 1.0, 90.0), (1.0, 0.0, 180.0), (0.0, -1.0, 270.0)],
    )
    def test_the_bearing_is_where_the_rain_is_going(
        self, vy: float, vx: float, bearing: float,
    ) -> None:
        out = _features_with_flow(_empty_field(), vy, vx)
        assert out["bulk_dir_deg"][0] == pytest.approx(bearing, abs=1e-3)

    def test_a_still_bulk_has_no_bearing(self) -> None:
        out = _features_with_flow(_empty_field(), 0.0, 0.0)
        assert math.isnan(float(out["bulk_dir_deg"][0]))

    def test_the_stall_share_rides_along_unchanged(self) -> None:
        out = _features_with_flow(_empty_field(), 0.0, 2.0)
        assert out["stalled_share"][0] == pytest.approx(0.25)


def test_station_features_rejects_nonsense() -> None:
    field = _empty_field()
    with pytest.raises(ValueError, match="2-D"):
        pp.station_features(
            field[0], field, field, np.array([1.0]), np.array([1.0]),
            pixel_km=PIXEL_KM, dt_min=10.0,
            bulk_vy=0.0, bulk_vx=0.0, stalled_share=0.0,
        )
    with pytest.raises(ValueError, match="pixel_km"):
        pp.station_features(
            field, field, field, np.array([1.0]), np.array([1.0]),
            pixel_km=0.0, dt_min=10.0,
            bulk_vy=0.0, bulk_vx=0.0, stalled_share=0.0,
        )


def test_the_feature_catalogue_documents_every_column() -> None:
    columns = pp.feature_columns((10, 30))
    names = [name for name, _ in columns]
    assert names[:2] == ["raw_frac_10", "raw_frac_30"]
    assert set(names) >= {name for name, _ in pp.SCALAR_FEATURE_COLUMNS}
    assert all(definition.strip() for _name, definition in columns)
    # The schema's own columns are not duplicated as features.
    assert "observed_mm_h" not in names
    assert "eta_min" not in names


def test_the_catalogue_is_append_only() -> None:
    """The v1 block keeps its exact place; the v2 columns come after it.

    A stored row is aligned by name, so the order binds only the writers —
    but the writers are the parity contract, and a column that moves is a
    column one writer can fill and the other cannot.
    """
    import pyarrow as pa

    names = [name for name, _ in pp.feature_columns((10, 30))]
    v1 = ["raw_frac_10", "raw_frac_30"] + [
        name for name, _ in pp.SCALAR_FEATURE_COLUMNS
    ]
    assert names[:len(v1)] == v1
    tail = names[len(v1):]
    assert tail[:4] == ["ens_mean_10", "ens_p90_10", "ens_mean_30", "ens_p90_30"]
    # The v2 block keeps its place, and the neighbour-gauge block appended
    # in 2026-09 comes after it — never inside it.
    assert tail[4:] == [
        name for name, _ in
        pp.SCALAR_FEATURE_COLUMNS_V2 + pp.SCALAR_FEATURE_COLUMNS_NG
    ]
    # Every appended column is documented, and every one of them is a
    # nullable float — "unknown" has to be expressible, and it is never a 0.
    schema = pp.feature_schema((10, 30))
    for name, _definition in (
        pp.SCALAR_FEATURE_COLUMNS_V2 + pp.SCALAR_FEATURE_COLUMNS_NG
    ):
        assert name in pp.FEATURE_DOC
        assert schema.field(name).type == pa.float32()


# ---------------------------------------------------------------------------
# v2/F2 — radar history at the point
# ---------------------------------------------------------------------------


class TestRadarHistory:
    def test_the_previous_frames_are_read_at_the_point(self) -> None:
        now, prev10, prev20 = _empty_field(), _empty_field(), _empty_field()
        _blob(prev10, 100, 100, 4.0, half=0)
        _blob(prev20, 100, 100, 9.0, half=0)
        out = pp.station_features(
            now,
            np.zeros(now.shape, dtype=np.float32),
            np.zeros(now.shape, dtype=np.float32),
            np.array([100.0]), np.array([100.0]),
            pixel_km=PIXEL_KM, dt_min=10.0,
            bulk_vy=0.0, bulk_vx=2.0, stalled_share=0.0,
            rain_prev10_mm_h=prev10, rain_prev20_mm_h=prev20,
        )
        assert out["obs_prev10_mm_h"][0] == pytest.approx(4.0)
        assert out["obs_prev20_mm_h"][0] == pytest.approx(9.0)
        assert out["obs_max_5km_prev10_mm_h"][0] == pytest.approx(4.0)

    def test_without_previous_frames_the_columns_are_absent_not_zero(self) -> None:
        out = _features_with_flow(_empty_field(), 0.0, 2.0)
        for name in (
            "obs_prev10_mm_h", "obs_prev20_mm_h", "obs_max_5km_prev10_mm_h",
        ):
            assert name not in out
        # The row still carries the column — as a null, never a 0 mm/h.
        row = pp.feature_row(
            out, 0, raw_fractions={}, leads=(10,),
            season="summer", hour_utc=12,
            frame_age_min=15.0, station_radar_km=40.0,
        )
        assert row["obs_prev10_mm_h"] is None

    def test_a_previous_frame_of_the_wrong_shape_is_refused(self) -> None:
        field = _empty_field()
        with pytest.raises(ValueError, match="rain_prev10_mm_h"):
            pp.station_features(
                field,
                np.zeros(field.shape, dtype=np.float32),
                np.zeros(field.shape, dtype=np.float32),
                np.array([100.0]), np.array([100.0]),
                pixel_km=PIXEL_KM, dt_min=10.0,
                bulk_vy=0.0, bulk_vx=2.0, stalled_share=0.0,
                rain_prev10_mm_h=field[:10, :10],
            )

    def test_the_wet_fraction_is_of_the_finite_pixels(self) -> None:
        field = _empty_field()
        rows, cols = np.mgrid[0:GRID, 0:GRID]
        field[((rows - 100) ** 2 + (cols - 100) ** 2) * PIXEL_KM ** 2 <= 25.0] = 2.0
        out = _features_with_flow(field, 0.0, 2.0)
        assert out["wet_frac_5km"][0] == pytest.approx(1.0)
        # The 10 km disc has four times the area of the wet part of it.
        assert out["wet_frac_10km"][0] == pytest.approx(0.25, abs=0.02)

    def test_off_the_composite_the_wet_fraction_is_unknown(self) -> None:
        field = np.full((GRID, GRID), np.nan, dtype=np.float32)
        out = _features_with_flow(field, 0.0, 2.0)
        assert math.isnan(float(out["wet_frac_5km"][0]))
        assert math.isnan(float(out["wet_frac_10km"][0]))


# ---------------------------------------------------------------------------
# v2/F3 — the corridor, resolved
# ---------------------------------------------------------------------------


class TestCorridorProfile:
    EAST = (0.0, 2.0)

    def test_each_bin_holds_its_own_five_kilometres(self) -> None:
        field = _empty_field()
        # One cell 17 km upwind (west): bin 3 is 15-20 km.
        _blob(field, 100, 100 - int(17.0 / PIXEL_KM), 8.0, half=0)
        out = _features_with_flow(field, *self.EAST)
        assert out["up_max_b3"][0] == pytest.approx(8.0)
        assert [
            float(out[f"up_max_b{i}"][0]) for i in range(8) if i != 3
        ] == pytest.approx([0.0] * 7)
        assert out["up_max_40km_mm_h"][0] == pytest.approx(8.0)

    def test_the_bins_tile_the_whole_forty_kilometres(self) -> None:
        field = np.full((GRID, GRID), 3.0, dtype=np.float32)
        out = _features_with_flow(field, *self.EAST)
        assert [float(out[f"up_max_b{i}"][0]) for i in range(8)] == (
            pytest.approx([3.0] * 8)
        )
        assert out["up_mean_40km_mm_h"][0] == pytest.approx(3.0)

    def test_the_mean_separates_a_band_from_a_core(self) -> None:
        band = np.full((GRID, GRID), 5.0, dtype=np.float32)
        core = _empty_field()
        _blob(core, 100, 100 - int(10.0 / PIXEL_KM), 5.0)
        wide = _features_with_flow(band, *self.EAST)
        narrow = _features_with_flow(core, *self.EAST)
        assert wide["up_max_40km_mm_h"][0] == pytest.approx(
            narrow["up_max_40km_mm_h"][0],
        )
        assert wide["up_mean_40km_mm_h"][0] > narrow["up_mean_40km_mm_h"][0]

    def test_no_usable_flow_leaves_every_bin_undefined(self) -> None:
        out = _features_with_flow(
            np.full((GRID, GRID), 5.0, dtype=np.float32), 0.0, 0.0,
        )
        assert math.isnan(float(out["up_mean_40km_mm_h"][0]))
        for index in range(8):
            assert math.isnan(float(out[f"up_max_b{index}"][0])), index


# ---------------------------------------------------------------------------
# v2/F4 — the shape of the ensemble
# ---------------------------------------------------------------------------


def _ensemble(members: list[list[float]]) -> np.ndarray:
    """``(n_members, n_timesteps, 1, 1)`` from one rate series per member."""
    return np.array(members, dtype=np.float32)[:, :, np.newaxis, np.newaxis]


class TestEnsembleShape:
    #: Ten-minute timesteps read with no frame age, so timestep ``t``
    #: answers for lead ``(t + 1) * 10``.
    KWARGS = dict(threshold_mm_h=0.5, timestep_min=10.0, frame_age_min=0.0)

    def test_the_mean_and_p90_are_of_the_cumulative_maximum(self) -> None:
        # Member 0 peaks at 4 and dries; member 1 holds 1 throughout. By
        # lead 30 their cumulative maxima are 4 and 1.
        ensemble = _ensemble([[0.0, 4.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        out = pp.ensemble_point_features(
            ensemble, [(0, 0)], leads_min=(10, 30), **self.KWARGS,
        )
        assert out["ens_mean_10"][0] == pytest.approx(0.5)
        assert out["ens_mean_30"][0] == pytest.approx(2.5)
        assert out["ens_p90_30"][0] == pytest.approx(3.7, abs=0.05)

    def test_the_spread_is_the_iqr_of_the_arrival_times(self) -> None:
        # Four members arriving at timesteps 0…3 → 10, 20, 30, 40 min.
        ensemble = _ensemble([
            [2.0, 2.0, 2.0, 2.0],
            [0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0],
        ])
        out = pp.ensemble_point_features(
            ensemble, [(0, 0)], leads_min=(30,), **self.KWARGS,
        )
        assert out["ens_eta_spread_min"][0] == pytest.approx(15.0)

    def test_agreement_reads_as_no_spread(self) -> None:
        ensemble = _ensemble([[0.0, 2.0, 2.0, 2.0]] * 5)
        out = pp.ensemble_point_features(
            ensemble, [(0, 0)], leads_min=(30,), **self.KWARGS,
        )
        assert out["ens_eta_spread_min"][0] == pytest.approx(0.0)

    def test_under_four_arrivals_there_is_no_spread_to_report(self) -> None:
        ensemble = _ensemble([
            [2.0, 2.0, 2.0, 2.0],
            [0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        out = pp.ensemble_point_features(
            ensemble, [(0, 0)], leads_min=(30,), **self.KWARGS,
        )
        assert math.isnan(float(out["ens_eta_spread_min"][0]))

    def test_a_point_off_the_product_grid_reads_nothing(self) -> None:
        ensemble = _ensemble([[2.0, 2.0, 2.0, 2.0]] * 5)
        out = pp.ensemble_point_features(
            ensemble, [None], leads_min=(10, 30), **self.KWARGS,
        )
        for name in ("ens_mean_10", "ens_p90_30", "ens_eta_spread_min"):
            assert math.isnan(float(out[name][0])), name

    def test_a_missing_timestep_does_not_poison_the_members_tail(self) -> None:
        """``fmax`` carries the running maximum past a NaN; ``maximum`` would not."""
        ensemble = _ensemble([[2.0, np.nan, 0.0, 0.0]] * 4)
        out = pp.ensemble_point_features(
            ensemble, [(0, 0)], leads_min=(40,), **self.KWARGS,
        )
        assert out["ens_mean_40"][0] == pytest.approx(2.0)

    def test_the_lead_picks_the_timestep_the_fraction_did(self) -> None:
        """One bucket rule, shared with ``national_products``."""
        from dmi_nowcast_core.national import national_products

        ensemble = _ensemble([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 0.0, 2.0]])
        products = national_products(
            ensemble, leads_min=(30, 40), threshold_mm_h=0.5,
            timestep_min=10.0, frame_age_min=0.0, downsample_factor=1,
        )
        out = pp.ensemble_point_features(
            ensemble, [(0, 0)], leads_min=(30, 40), **self.KWARGS,
        )
        assert products.p_rain[30][0, 0] == pytest.approx(0.5)
        assert products.p_rain[40][0, 0] == pytest.approx(1.0)
        assert out["ens_mean_30"][0] == pytest.approx(1.0)
        assert out["ens_mean_40"][0] == pytest.approx(2.0)

    def test_it_refuses_an_array_that_is_not_an_ensemble(self) -> None:
        with pytest.raises(ValueError, match="n_members"):
            pp.ensemble_point_features(
                np.zeros((3, 3)), [(0, 0)], leads_min=(10,), **self.KWARGS,
            )


# ---------------------------------------------------------------------------
# v2/F1 — the station's own gauge, and the availability rule
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


def _slots(*entries: tuple[float, bool | None, float | None]) -> list[tuple]:
    """``(minutes before noon, wet, mm)`` → the triples the core reads."""
    return [
        (_T0 - timedelta(minutes=back), wet, mm) for back, wet, mm in entries
    ]


class TestGaugeFeatures:
    LAG = 10.0

    def test_the_windows_are_measured_from_the_visibility_horizon(self) -> None:
        out = pp.station_gauge_features(
            [_slots((10, True, 0.4), (20, True, 0.3), (30, False, 0.0))],
            now_utc=_T0, lag_min=self.LAG,
        )
        assert out["g_mm_10"][0] == pytest.approx(0.4)
        assert out["g_mm_30"][0] == pytest.approx(0.7)
        assert out["g_mm_60"][0] == pytest.approx(0.7)
        assert out["g_known"][0] == pytest.approx(1.0)
        assert out["g_dry_60"][0] == pytest.approx(0.0)
        # ...but the age of the rain is measured from the decision
        # instant, so the lag is part of it.
        assert out["g_min_since_wet"][0] == pytest.approx(10.0)

    def test_a_slot_inside_the_lag_is_invisible(self) -> None:
        """The leakage guard: one minute too fresh is not there at all.

        A slot ending at ``t - lag + 1`` had not reached the store when
        the cycle ran. If the replay could see it, every coefficient on
        the gauge block would be worth less in production than it looks
        offline — and nothing would fail.
        """
        out = pp.station_gauge_features(
            [_slots((self.LAG - 1, True, 5.0))],
            now_utc=_T0, lag_min=self.LAG,
        )
        assert out["g_known"][0] == pytest.approx(0.0)
        for name in ("g_mm_10", "g_mm_30", "g_mm_60", "g_min_since_wet",
                     "g_dry_60"):
            assert math.isnan(float(out[name][0])), name
        # A minute older and it is visible — the boundary from both sides.
        seen = pp.station_gauge_features(
            [_slots((self.LAG, True, 5.0))], now_utc=_T0, lag_min=self.LAG,
        )
        assert seen["g_mm_10"][0] == pytest.approx(5.0)

    def test_a_dry_hour_is_a_measurement(self) -> None:
        out = pp.station_gauge_features(
            [_slots(*[(10 * k, False, 0.0) for k in range(1, 8)])],
            now_utc=_T0, lag_min=self.LAG,
        )
        assert out["g_dry_60"][0] == pytest.approx(1.0)
        assert out["g_known"][0] == pytest.approx(1.0)
        assert out["g_mm_60"][0] == pytest.approx(0.0)
        # Known, and dry as far back as anything can see: the cap, not null.
        assert out["g_min_since_wet"][0] == pytest.approx(pp.GAUGE_SINCE_CAP_MIN)

    def test_an_unknown_slot_is_not_a_dry_one(self) -> None:
        out = pp.station_gauge_features(
            [_slots(*[(10 * k, None, None) for k in range(1, 8)])],
            now_utc=_T0, lag_min=self.LAG,
        )
        assert out["g_known"][0] == pytest.approx(0.0)
        assert math.isnan(float(out["g_dry_60"][0]))
        assert math.isnan(float(out["g_mm_60"][0]))

    def test_rain_before_the_hour_still_dates_it(self) -> None:
        """``g_dry_60`` and ``g_min_since_wet`` answer different questions."""
        out = pp.station_gauge_features(
            [_slots((10, False, 0.0), (20, False, 0.0), (130, True, 2.0))],
            now_utc=_T0, lag_min=self.LAG,
        )
        assert out["g_dry_60"][0] == pytest.approx(1.0)
        assert out["g_min_since_wet"][0] == pytest.approx(130.0)

    def test_a_point_with_no_gauge_is_a_null_block_and_a_zero_flag(self) -> None:
        out = pp.station_gauge_features([None, []], now_utc=_T0, lag_min=self.LAG)
        assert list(out["g_known"]) == [0.0, 0.0]
        for index in (0, 1):
            assert math.isnan(float(out["g_mm_60"][index]))
            assert math.isnan(float(out["g_min_since_wet"][index]))

    def test_the_cap_holds_however_old_the_rain_is(self) -> None:
        out = pp.station_gauge_features(
            [_slots((10, False, 0.0), (5000, True, 9.0))],
            now_utc=_T0, lag_min=self.LAG,
        )
        assert out["g_min_since_wet"][0] == pytest.approx(pp.GAUGE_SINCE_CAP_MIN)

    def test_a_naive_slot_end_is_a_bug_not_a_zone(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            pp.station_gauge_features(
                [[(datetime(2026, 9, 16, 11, 50), True, 1.0)]],
                now_utc=_T0, lag_min=self.LAG,
            )

    def test_the_wet_rule_is_the_projects_one(self) -> None:
        """``wet`` arrives already decided — by ``warning_score``, once."""
        from dmi_nowcast_core import warning_score as ws

        table = [{
            "station_id": "06180",
            "observed_utc": _T0 - timedelta(minutes=20),
            "parameter_id": ws.PRECIP_PARAM,
            "value": 0.1,
        }]
        slots = ws.gauge_slot_amounts(
            table, "06180",
            start_utc=_T0 - timedelta(minutes=60), end_utc=_T0,
        )
        out = pp.station_gauge_features([slots], now_utc=_T0, lag_min=self.LAG)
        assert out["g_mm_30"][0] == pytest.approx(0.1)
        assert out["g_dry_60"][0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The design matrix
# ---------------------------------------------------------------------------


def _one_row(**over) -> dict[str, np.ndarray]:
    row = {
        "raw_frac_20": 0.5, "raw_frac_30": 0.75,
        "observed_mm_h": 0.0, "obs_max_5km_mm_h": 1.0,
        "up_max_20km_mm_h": 3.0, "up_max_40km_mm_h": 3.0,
        "up_dist_km": 12.0, "up_wet_frac_40km": 0.2,
        "eta_min": 25.0, "intensity_mm_h": 2.0,
        "bulk_kmh": 30.0, "bulk_dir_deg": 270.0,
        "local_speed_kmh": 28.0, "stalled_share": 0.02,
        "frame_age_min": 15.0, "station_radar_km": 40.0,
        "hour_utc": 6.0, "season": "summer",
    }
    row.update(over)
    return {
        key: (np.array([value]) if not isinstance(value, str)
              else np.array([value], dtype="<U8"))
        for key, value in row.items()
    }


def _design_value(features, name: str, leads=(20, 30)) -> float:
    names = pp.design_columns(leads)
    return float(pp.build_design(features, leads)[0, names.index(name)])


def test_the_design_names_line_up_with_its_columns() -> None:
    names = pp.design_columns((20, 30))
    design = pp.build_design(_one_row(), (20, 30))
    assert design.shape == (1, len(names))
    assert names[0] == "raw_frac_20"
    assert _design_value(_one_row(), "raw_frac_30") == pytest.approx(0.75)


def test_rain_rates_and_distances_go_through_log1p() -> None:
    features = _one_row()
    assert _design_value(features, "log1p_up_max_40km_mm_h") == pytest.approx(
        math.log1p(3.0)
    )
    assert _design_value(features, "log1p_station_radar_km") == pytest.approx(
        math.log1p(40.0)
    )
    assert _design_value(features, "log1p_up_dist_km") == pytest.approx(
        math.log1p(12.0)
    )
    # A dry observation stays exactly dry under log1p.
    assert _design_value(features, "log1p_observed_mm_h") == pytest.approx(0.0)


def test_a_missing_upstream_distance_becomes_an_indicator_and_a_cap() -> None:
    present = _one_row()
    absent = _one_row(up_dist_km=float("nan"))
    assert _design_value(present, "up_no_echo") == 0.0
    assert _design_value(absent, "up_no_echo") == 1.0
    assert _design_value(absent, "log1p_up_dist_km") == pytest.approx(
        math.log1p(pp.UP_DIST_CAP_KM)
    )


def test_a_missing_eta_caps_the_time_and_zeroes_the_intensity() -> None:
    absent = _one_row(eta_min=float("nan"), intensity_mm_h=float("nan"))
    assert _design_value(absent, "eta_missing") == 1.0
    assert _design_value(absent, "eta_min_capped") == pytest.approx(pp.ETA_CAP_MIN)
    assert _design_value(absent, "log1p_intensity_mm_h") == pytest.approx(0.0)
    assert _design_value(_one_row(), "eta_missing") == 0.0


def test_angles_become_sin_cos_so_the_wrap_is_not_a_cliff() -> None:
    north = _one_row(bulk_dir_deg=0.0, hour_utc=0.0)
    assert _design_value(north, "bulk_dir_sin") == pytest.approx(0.0, abs=1e-9)
    assert _design_value(north, "bulk_dir_cos") == pytest.approx(1.0)
    assert _design_value(north, "hour_sin") == pytest.approx(0.0, abs=1e-9)
    assert _design_value(north, "hour_cos") == pytest.approx(1.0)
    # 23:00 and 00:00 are one hour apart on the circle.
    late = _design_value(_one_row(hour_utc=23.0), "hour_sin")
    assert late == pytest.approx(math.sin(23 * 2 * math.pi / 24))


def test_a_missing_bearing_contributes_nothing_rather_than_a_direction() -> None:
    features = _one_row(bulk_dir_deg=float("nan"))
    assert _design_value(features, "bulk_dir_sin") == 0.0
    assert _design_value(features, "bulk_dir_cos") == 0.0


def test_the_season_is_one_hot_over_all_three_levels() -> None:
    for season in pp.SEASONS:
        features = _one_row(season=season)
        hot = [
            _design_value(features, f"season_{name}") for name in pp.SEASONS
        ]
        assert sum(hot) == 1.0
        assert hot[pp.SEASONS.index(season)] == 1.0


def test_a_column_the_run_never_wrote_reads_as_missing() -> None:
    features = _one_row()
    del features["up_max_40km_mm_h"]
    # NaN in the design, which the standardiser then imputes.
    assert math.isnan(_design_value(features, "log1p_up_max_40km_mm_h"))


def test_a_row_from_before_the_v2_columns_still_builds_a_design() -> None:
    """The additive rule, from the direction that would break a refit.

    Every replay row written before 2026-09-16 carries the v1 block and
    nothing else. The nightly fit reads them by name off parquet, so the
    v2 columns arrive as nulls — and a design that refused them, or that
    changed shape because of them, would silently drop ten months of
    training rows.
    """
    v1_only = _one_row()
    assert not set(v1_only) & {
        name for name, _ in pp.SCALAR_FEATURE_COLUMNS_V2
    }
    design = pp.build_design(v1_only, (20, 30))
    assert design.shape == (1, len(pp.design_columns((20, 30))))
    # And it is scoreable: the standardiser imputes what is missing, so
    # the transformed row is finite whatever the writer could not compute.
    scaler = pp.Standardiser.fit(design)
    assert np.all(np.isfinite(scaler.transform(design)))


class TestStandardiser:
    def test_it_centres_and_scales_and_imputes_with_the_mean(self) -> None:
        design = np.array([[1.0, 5.0], [3.0, np.nan], [5.0, 5.0]])
        scaler = pp.Standardiser.fit(design)
        out = scaler.transform(design)
        assert out[:, 0] == pytest.approx([-1.2247449, 0.0, 1.2247449])
        # A constant column has no scale, so it standardises to zero...
        assert out[:, 1] == pytest.approx([0.0, 0.0, 0.0])
        # ...and the missing entry took the column mean, i.e. zero after.
        assert np.all(np.isfinite(out))

    def test_an_all_missing_column_is_finite_and_carries_nothing(self) -> None:
        design = np.array([[1.0, np.nan], [2.0, np.nan]])
        out = pp.Standardiser.fit(design).transform(design)
        assert np.all(np.isfinite(out))
        assert out[:, 1] == pytest.approx([0.0, 0.0])

    def test_a_width_mismatch_is_an_error_not_a_broadcast(self) -> None:
        scaler = pp.Standardiser.fit(np.zeros((3, 2)))
        with pytest.raises(ValueError, match="standardiser has"):
            scaler.transform(np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# The logistic fit
# ---------------------------------------------------------------------------


def test_the_logistic_recovers_the_signs_and_rough_magnitudes() -> None:
    rng = np.random.default_rng(11)
    n = 20_000
    x = rng.normal(size=(n, 3))
    truth = np.array([1.5, -0.8, 0.0])
    z = -0.5 + x @ truth
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-z))).astype(float)
    fit = pp.fit_logistic(x, y, l2=1.0)
    assert fit["converged"]
    coefficients = np.array(fit["coefficients"])
    assert coefficients[0] == pytest.approx(1.5, abs=0.1)
    assert coefficients[1] == pytest.approx(-0.8, abs=0.1)
    assert abs(coefficients[2]) < 0.1
    assert fit["intercept"] == pytest.approx(-0.5, abs=0.1)
    assert fit["base_rate"] == pytest.approx(float(y.mean()))


def test_a_stronger_ridge_shrinks_the_slopes_but_not_the_intercept() -> None:
    rng = np.random.default_rng(3)
    n = 2_000
    x = rng.normal(size=(n, 2))
    z = 0.4 + x @ np.array([2.0, -2.0])
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-z))).astype(float)
    loose = pp.fit_logistic(x, y, l2=1.0)
    tight = pp.fit_logistic(x, y, l2=5_000.0)
    assert np.linalg.norm(tight["coefficients"]) < np.linalg.norm(
        loose["coefficients"]
    )
    # The intercept is never penalised: it still tracks the base rate.
    assert tight["intercept"] == pytest.approx(
        math.log(y.mean() / (1 - y.mean())), abs=0.15
    )


def test_a_single_class_fold_returns_an_intercept_and_says_so() -> None:
    fit = pp.fit_logistic(np.zeros((10, 2)), np.zeros(10), l2=1.0)
    assert fit["coefficients"] == [0.0, 0.0]
    assert fit["converged"] is False
    assert "single-class" in fit["message"]
    assert fit["intercept"] < 0


def test_the_logistic_refuses_inputs_it_cannot_fit() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        pp.fit_logistic(np.array([[np.nan]]), np.array([1.0]))
    with pytest.raises(ValueError, match="0/1"):
        pp.fit_logistic(np.zeros((2, 1)), np.array([0.0, 0.5]))
    with pytest.raises(ValueError, match="rows"):
        pp.fit_logistic(np.zeros((2, 1)), np.array([0.0]))


# ---------------------------------------------------------------------------
# Isotonic recalibration
# ---------------------------------------------------------------------------


def test_the_isotonic_step_is_monotone_and_compact() -> None:
    rng = np.random.default_rng(5)
    n = 50_000
    score = rng.uniform(size=n)
    # Systematically over-confident: the truth is half the score.
    y = (rng.uniform(size=n) < score / 2.0).astype(float)
    curve = pp.fit_isotonic_binned(score, y, n_bins=100)
    values = np.asarray(curve.calibrated_values)
    assert np.all(np.diff(values) >= -1e-9)
    assert len(curve.raw_breakpoints) <= 100
    assert np.all(np.diff(curve.raw_breakpoints) > 0)
    assert float(np.asarray(curve.predict(np.array([0.8]))[0])) == pytest.approx(
        0.4, abs=0.05
    )


def test_a_constant_score_calibrates_to_its_base_rate() -> None:
    curve = pp.fit_isotonic_binned(np.full(100, 0.3), np.r_[np.ones(20), np.zeros(80)])
    assert float(np.asarray(curve.predict(np.array([0.3]))[0])) == pytest.approx(0.2)


def test_reliability_bins_follow_the_project_wide_rule() -> None:
    p = np.array([0.0, 0.05, 0.15, 1.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    rows = reliability_bins(p, y, n_bins=10)
    assert [row["bin"] for row in rows] == [0, 1, 9]
    # p == 1 folds into the last bin rather than falling off the end.
    assert rows[-1]["n"] == 1
    assert rows[0]["n"] == 2


# ---------------------------------------------------------------------------
# End to end: a known generating process
# ---------------------------------------------------------------------------

#: The synthetic archive: four months, twelve days each, twenty decision
#: instants a day at six stations. Small enough for a unit test, wide
#: enough that a day-block bootstrap over 48 days says something.
N_MONTHS, N_DAYS, N_INSTANTS, N_STATIONS = 4, 12, 20, 6
LEADS = (20, 30)


def _synthetic(seed: int = 17) -> dict:
    """Rows whose outcome rises with upstream rain and falls with distance.

    The fixture is built around the **baseline's blind spot**, which is the
    defect this whole work package exists for. The outcome depends on two
    things: how much rain is upstream and how far away it is. The ensemble
    fraction is a function of the first alone — because sixteen members
    sharing one motion field all agree about *whether* there is rain
    upstream — and it saturates at 1.0 once there is plenty of it. So the
    curve baseline, which sees only the fraction, pools a band 5 km away
    with one 40 km away and cannot tell them apart at any threshold. The
    distance is what separates them, and only the post-processor is given
    it.
    """
    rng = np.random.default_rng(seed)
    n = N_MONTHS * N_DAYS * N_INSTANTS * N_STATIONS

    # Timestamps: consecutive months from 2026-01, twelve days each, ten
    # minutes apart, every station sharing the instant.
    month_index = np.repeat(np.arange(N_MONTHS), N_DAYS * N_INSTANTS * N_STATIONS)
    day_in_month = np.tile(
        np.repeat(np.arange(N_DAYS), N_INSTANTS * N_STATIONS), N_MONTHS,
    )
    instant = np.tile(np.repeat(np.arange(N_INSTANTS), N_STATIONS), N_MONTHS * N_DAYS)
    station = np.tile(np.arange(N_STATIONS), N_MONTHS * N_DAYS * N_INSTANTS)
    base = np.datetime64("2026-01-01T06:00:00")
    t = (
        base.astype("datetime64[s]").astype(np.int64)
        + month_index.astype(np.int64) * 31 * 86400
        + day_in_month.astype(np.int64) * 86400
        + instant.astype(np.int64) * 600
    )

    up_max = np.exp(rng.normal(0.2, 0.9, size=n))          # mm/h, log-normal
    up_dist = rng.uniform(0.0, 45.0, size=n)               # km
    no_echo = up_dist > 40.0
    up_dist = np.where(no_echo, np.nan, up_dist)
    up_max = np.where(no_echo, 0.0, up_max)

    log_max = np.log1p(up_max)
    log_dist = np.log1p(np.nan_to_num(up_dist, nan=pp.UP_DIST_CAP_KM))
    logit = 1.4 + 1.2 * log_max - 1.3 * log_dist
    probability = 1.0 / (1.0 + np.exp(-logit))
    y = {
        lead: (rng.uniform(size=n) < np.clip(probability * (lead / 30.0), 0, 1)
               ).astype(np.float64)
        for lead in LEADS
    }

    # The ensemble fraction: sixteen members, blind to the distance and
    # saturating once there is plenty of rain upstream. Roughly the top
    # quarter of the rows read a unanimous 1.0.
    agreement = 1.0 / (1.0 + np.exp(-(-1.4 + 1.3 * log_max)))
    fraction = np.clip(
        np.round(agreement * 1.6 * 16.0) / 16.0 + rng.normal(0, 0.03, size=n), 0, 1,
    )
    features = {
        "up_max_40km_mm_h": up_max,
        "up_max_20km_mm_h": np.where(np.nan_to_num(up_dist, nan=99) <= 20, up_max, 0.0),
        "up_dist_km": up_dist,
        "up_wet_frac_40km": np.clip(up_max / 10.0, 0, 1),
        "season": pp.seasons_from_epoch(t),
        "hour_utc": pp.hours_from_epoch(t).astype(np.float64),
    }
    for lead in LEADS:
        features[pp.raw_fraction_column(lead)] = fraction
    truth = {lead: (y[lead], np.ones(n, dtype=bool)) for lead in LEADS}
    # The baseline is generous on purpose: the curve is fitted IN SAMPLE on
    # all the rows it is then scored on, so any win below is a win against
    # the best a fraction-only calibration could possibly do.
    baseline = {
        lead: np.asarray(
            fit_isotonic(fraction, y[lead]).predict(fraction), dtype=np.float64,
        )
        for lead in LEADS
    }
    return {
        "features": features,
        "truth": truth,
        "baseline": baseline,
        "t": t,
        "month": pp.year_months_from_epoch(t),
        "day": t // 86_400,
        "station": station,
        "n": n,
    }


@pytest.fixture(scope="module")
def synthetic() -> dict:
    return _synthetic()


def test_the_fit_recovers_the_generating_signs_and_scale(synthetic: dict) -> None:
    """Signs first, then magnitude — on the predictor the baseline lacks.

    The generating process is linear in ``log1p(distance)`` with a
    coefficient of −1.3, and the design standardises, so the fitted weight
    should land near ``−1.3 × sd(log1p(distance))``. Upstream rain is
    checked for sign only: it is collinear with the ensemble fraction by
    construction, and the two split its weight between them in a way no
    single number pins down.
    """
    model = pp.fit_postprocess(
        synthetic["features"], synthetic["truth"], LEADS, l2=1.0,
        design_leads=LEADS,
    )
    names = list(model.feature_names)
    coefficients = np.array(model.models[30].coefficients)
    up_max = coefficients[names.index("log1p_up_max_40km_mm_h")]
    index = names.index("log1p_up_dist_km")
    up_dist = coefficients[index]
    expected = -1.3 * model.standardiser.scale[index]

    assert up_max > 0.0
    assert up_dist < 0.0
    assert up_dist == pytest.approx(expected, rel=0.35)


def test_the_predictions_are_probabilities(synthetic: dict) -> None:
    model = pp.fit_postprocess(
        synthetic["features"], synthetic["truth"], LEADS, design_leads=LEADS,
    )
    predictions = model.predict(synthetic["features"])
    assert set(predictions) == set(LEADS)
    for lead, values in predictions.items():
        assert values.shape == (synthetic["n"],)
        assert np.all((values >= 0.0) & (values <= 1.0))
    # A single lead comes back as its own array.
    assert np.allclose(
        model.predict(synthetic["features"], 30), predictions[30],
    )
    with pytest.raises(KeyError):
        model.predict(synthetic["features"], 45)


def test_the_json_round_trip_reproduces_the_predictions(synthetic: dict) -> None:
    model = pp.fit_postprocess(
        synthetic["features"], synthetic["truth"], LEADS, l2=2.0,
        design_leads=LEADS, training={"rows": synthetic["n"], "note": "unit test"},
    )
    document = json.loads(model.dumps())
    assert document["schema_version"] == pp.SCHEMA_VERSION
    assert document["leads"] == list(LEADS)
    assert document["l2"] == 2.0
    assert document["training"]["note"] == "unit test"
    assert document["features"]["names"] == list(model.feature_names)
    assert len(document["scaling"]["mean"]) == len(model.feature_names)
    assert document["models"]["30"]["isotonic"]["raw_breakpoints"]
    assert document["fitted_at_utc"].endswith("+00:00")

    restored = pp.PostprocessModel.loads(model.dumps())
    for lead in LEADS:
        assert np.allclose(
            restored.predict(synthetic["features"], lead),
            model.predict(synthetic["features"], lead),
        )


def test_a_document_from_another_schema_version_is_refused(synthetic: dict) -> None:
    model = pp.fit_postprocess(
        synthetic["features"], synthetic["truth"], LEADS, design_leads=LEADS,
    )
    document = model.to_json()
    document["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        pp.PostprocessModel.from_json(document)


def test_fit_postprocess_refuses_a_lead_with_no_truth(synthetic: dict) -> None:
    with pytest.raises(ValueError, match="no truth"):
        pp.fit_postprocess(synthetic["features"], synthetic["truth"], (60,))


class TestLeaveOneMonthOut:
    """The honest test: nothing is ever scored on a month it was fitted on."""

    @pytest.fixture(scope="class")
    def evaluation(self, synthetic: dict) -> dict:
        return pp.leave_one_month_out(
            synthetic["features"], synthetic["truth"], LEADS,
            month=synthetic["month"], day=synthetic["day"],
            baseline=synthetic["baseline"],
            l2=1.0, design_leads=LEADS,
            n_resamples=100, seed=0,
        )

    def test_every_month_is_held_out_exactly_once(
        self, evaluation: dict, synthetic: dict,
    ) -> None:
        assert len(evaluation["folds"]) == N_MONTHS
        assert sum(f["n_test"] for f in evaluation["folds"]) == synthetic["n"]
        for fold in evaluation["folds"]:
            assert fold["n_train"] == synthetic["n"] - fold["n_test"]
        # Every row got a prediction from the fold that held it out.
        for lead in LEADS:
            assert np.all(np.isfinite(evaluation["out_of_fold"][lead]))

    def test_the_post_processor_beats_the_curve_baseline(
        self, evaluation: dict,
    ) -> None:
        for lead in LEADS:
            pooled = evaluation["leads"][str(lead)][pp.POOLED]
            assert pooled["postprocess"]["bss"] > pooled["baseline"]["bss"]
            assert pooled["postprocess"]["pr_auc"] > pooled["baseline"]["pr_auc"]

    def test_the_paired_interval_excludes_zero(self, evaluation: dict) -> None:
        difference = evaluation["leads"]["30"][pp.POOLED]["difference"]
        point, lo, hi = difference["bss"]
        assert point > 0
        assert lo > 0 and hi > lo
        assert difference["bss_excludes_zero"] is True
        assert difference["days"] == N_MONTHS * N_DAYS

    def test_it_reports_the_strata_it_has_rows_for(self, evaluation: dict) -> None:
        strata = evaluation["leads"]["30"]
        assert strata[pp.POOLED] is not None
        # Jan-Apr 2026: three winter months and one shoulder, no summer.
        assert strata["winter"] is not None
        assert strata["shoulder"] is not None
        assert strata["summer"] is None

    def test_both_forecasts_are_scored_on_the_very_same_rows(
        self, evaluation: dict,
    ) -> None:
        for lead in LEADS:
            block = evaluation["leads"][str(lead)][pp.POOLED]
            assert block["baseline"]["n"] == block["postprocess"]["n"]
            assert block["baseline"]["base_rate"] == pytest.approx(
                block["postprocess"]["base_rate"]
            )

    def test_the_reliability_table_rides_along(self, evaluation: dict) -> None:
        table = evaluation["leads"]["30"][pp.POOLED]["postprocess"][
            "reliability_table"
        ]
        assert table
        assert all(0 <= row["bin"] <= 9 and row["n"] > 0 for row in table)

    def test_no_bootstrap_means_no_interval_rather_than_a_fake_one(
        self, synthetic: dict,
    ) -> None:
        quick = pp.leave_one_month_out(
            synthetic["features"], synthetic["truth"], (30,),
            month=synthetic["month"], day=synthetic["day"],
            baseline=synthetic["baseline"], design_leads=LEADS,
            n_resamples=0,
        )
        assert quick["leads"]["30"][pp.POOLED]["difference"] is None


def test_out_of_fold_predictions_differ_from_the_in_sample_ones(
    synthetic: dict,
) -> None:
    """A held-out prediction is not the training fit read back."""
    evaluation = pp.leave_one_month_out(
        synthetic["features"], synthetic["truth"], (30,),
        month=synthetic["month"], day=synthetic["day"],
        baseline=synthetic["baseline"], design_leads=LEADS, n_resamples=0,
    )
    in_sample = pp.fit_postprocess(
        synthetic["features"], synthetic["truth"], (30,), design_leads=LEADS,
    ).predict(synthetic["features"], 30)
    out_of_fold = evaluation["out_of_fold"][30]
    assert not np.allclose(in_sample, out_of_fold)
    # Close, though — the folds share most of their training data.
    assert float(np.mean(np.abs(in_sample - out_of_fold))) < 0.05
