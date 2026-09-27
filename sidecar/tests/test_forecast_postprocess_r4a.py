"""Review R4a — a clicked pixel's row carries the history and ensemble blocks.

Before R4a the on-demand ``/forecast`` row left ``obs_prev10_mm_h``,
``obs_prev20_mm_h``, ``obs_max_5km_prev10_mm_h`` and the whole ``ens_*``
block NaN. The shipped tree models have ``missing_type`` None on those
nodes and read a NaN as 0 — "no rain ten minutes ago, no ensemble rain" —
so a clicked wet pixel was scored low. The cycle now keeps the previous
frames and per-pixel ensemble grids in its context, and these tests pin:

1. the on-demand row equals the cycle's own row for the same point, bit
   for bit, in all 14 columns (and the probability with it);
2. a context that lacks a family says so (``missing_families``), serves
   that family null exactly as before, and the engine counts it;
3. the per-cycle counter of NaN cells in columns a tree model never saw a
   NaN in (``PostprocessTable.nan_blind_columns``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from dmi_nowcast_core import postprocess as pp
from dmi_nowcast_core import postprocess_trees as pt
from dmi_nowcast_core.national import NationalProducts
from dmi_nowcast_core.product_pairs import nearest_radar_km
from dmi_nowcast_sidecar import compute as compute_mod
from dmi_nowcast_sidecar.config import Config
from dmi_nowcast_sidecar.national_sample import product_pixel_of
from dmi_nowcast_sidecar.push.postprocess import (
    PostprocessContext,
    PostprocessTable,
    build_cycle_postprocess,
    point_key,
    score_point,
)
from tests import test_forecast_postprocess as base

LEADS = base.LEADS
R4A_COLUMNS = (
    ["obs_prev10_mm_h", "obs_prev20_mm_h", "obs_max_5km_prev10_mm_h"]
    + [pp.ens_mean_column(lead) for lead in LEADS]
    + [pp.ens_p90_column(lead) for lead in LEADS]
    + ["ens_eta_spread_min"]
)


def _ensemble() -> np.ndarray:
    """16 members × 9 steps on the product grid, wet enough to arrive."""
    rng = np.random.default_rng(4)
    ens = rng.gamma(0.8, 1.5, size=(16, 9, base.GRID_DS, base.GRID_DS))
    ens = ens.astype(np.float32)
    ens[ens < 0.6] = 0.0
    return ens


def _prev_fields() -> tuple[np.ndarray, np.ndarray]:
    """The anchor band one and two frames earlier: shifted up-flow."""
    field = base._rain_field()
    prev10 = np.roll(field, (-2, -2), axis=(0, 1)) * np.float32(0.8)
    prev20 = np.roll(field, (-4, -3), axis=(0, 1)) * np.float32(0.6)
    return prev10.astype(np.float32), prev20.astype(np.float32)


def _raw_products() -> NationalProducts:
    products = base._products()
    return NationalProducts(
        p_rain=base._raw_grids(),
        eta_min=products.eta_min,
        intensity_mm_h=products.intensity_mm_h,
        leads_min=products.leads_min,
        threshold_mm_h=products.threshold_mm_h,
        timestep_min=products.timestep_min,
        frame_age_min=products.frame_age_min,
        downsample_factor=products.downsample_factor,
        n_members=products.n_members,
    )


def _cycle(table: PostprocessTable, keys, *, with_blocks: bool = True):
    """``(CyclePostprocess, PostprocessContext)`` with the R4a blocks.

    Assembled as ``CycleEngine._compute_sync`` / ``_publish_postprocess``
    do: ``station_features`` with the previous frames, ``_read_points``
    with the ensemble (which yields both the per-point ``ens_*`` columns
    and, with ``keep_grids``, the per-pixel grids).
    """
    geo = base.LinearGeo()
    field = base._rain_field()
    vy, vx = base._flow()
    prev10, prev20 = _prev_fields()
    native = [geo.lonlat_to_grid(lon, lat) for lat, lon in keys]
    points = compute_mod._read_points(
        _raw_products(), native, keep_grids=True,
        ensemble=_ensemble(), threshold_mm_h=0.5,
    )
    assert points is not None and points.ens_grids
    grid_features = pp.station_features(
        field, vy, vx,
        np.array([idx.row for idx in native], dtype=np.float64),
        np.array([idx.col for idx in native], dtype=np.float64),
        pixel_km=base.PIXEL_KM, dt_min=base.DT_MIN,
        bulk_vy=2.0, bulk_vx=1.5, stalled_share=0.011,
        rain_prev10_mm_h=prev10, rain_prev20_mm_h=prev20,
    )
    grid_features = {**grid_features, **points.ens_features}
    products = base._products()
    observed = base._observed()
    shared = [
        {
            "observed_mm_h": float(observed[r, c]),
            "eta_min": float(products.eta_min[r, c]),
            "intensity_mm_h": float(products.intensity_mm_h[r, c]),
        }
        for r, c in points.pixels
    ]
    cycle = build_cycle_postprocess(
        table,
        radar_ts_utc=base.RADAR_TS,
        generated_at_utc=base.GENERATED_AT,
        keys=[point_key(lat, lon) for lat, lon in keys],
        grid_features=grid_features,
        raw_fractions=points.raw_fractions,
        shared=shared,
        station_radar_km=[float(nearest_radar_km(la, lo)) for la, lo in keys],
        leads=LEADS,
        season=pp.season_of_month(base.GENERATED_AT.month),
        hour_utc=base.GENERATED_AT.hour,
        frame_age_min=base.FRAME_AGE_MIN,
    )
    context = PostprocessContext(
        radar_ts_utc=base.RADAR_TS,
        generated_at_utc=base.GENERATED_AT,
        rain_mm_h=field,
        vy=vy,
        vx=vx,
        raw_fraction_grids=dict(points.raw_grids or {}),
        leads=tuple(LEADS),
        pixel_km=base.PIXEL_KM,
        dt_min=base.DT_MIN,
        bulk_vy=2.0,
        bulk_vx=1.5,
        stalled_share=0.011,
        season=pp.season_of_month(base.GENERATED_AT.month),
        hour_utc=base.GENERATED_AT.hour,
        frame_age_min=base.FRAME_AGE_MIN,
        rain_prev10_mm_h=prev10 if with_blocks else None,
        rain_prev20_mm_h=prev20 if with_blocks else None,
        ens_grids=dict(points.ens_grids) if with_blocks else {},
    )
    return cycle, context, products


def _on_demand(table, context, products, lat, lon):
    idx = base.LinearGeo().lonlat_to_grid(lon, lat)
    pixel = product_pixel_of(products, idx.row, idx.col)
    assert pixel is not None
    r, c = pixel
    observed = base._observed()
    return score_point(
        table, context,
        row=idx.row, col=idx.col,
        raw_fractions={
            int(lead): float(grid[r, c])
            for lead, grid in context.raw_fraction_grids.items()
        },
        shared={
            "observed_mm_h": float(observed[r, c]),
            "eta_min": float(products.eta_min[r, c]),
            "intensity_mm_h": float(products.intensity_mm_h[r, c]),
        },
        station_radar_km=float(nearest_radar_km(lat, lon)),
        lat=lat, lon=lon,
        product_pixel=(r, c),
    )


class TestTheClickedRowCarriesTheBlocks:
    def test_on_demand_equals_the_cycle_row_in_all_14_columns(
        self, minimal_config: Config,
    ) -> None:
        table = base._table(minimal_config)
        keys = base.IN_CYCLE + (base.OFF_CYCLE,)
        cycle, context, products = _cycle(table, keys)
        for index, (lat, lon) in enumerate(keys):
            scored, features = _on_demand(table, context, products, lat, lon)
            row = cycle.rows[index]
            for name in R4A_COLUMNS:
                assert features[name] == row[name], name
            for lead, values in cycle.p_post.items():
                assert scored[lead] == values[index]

    def test_the_blocks_are_finite_where_the_inputs_exist(
        self, minimal_config: Config,
    ) -> None:
        table = base._table(minimal_config)
        _cycle_obj, context, products = _cycle(table, base.IN_CYCLE)
        assert context.missing_families() == ()
        _scored, features = _on_demand(table, context, products, *base.OFF_CYCLE)
        for name in R4A_COLUMNS:
            assert features[name] is not None, name
            assert np.isfinite(features[name]), name

    def test_a_context_without_the_blocks_serves_them_null(
        self, minimal_config: Config,
    ) -> None:
        """Pre-R4a behaviour, and now it says so."""
        table = base._table(minimal_config)
        _c, context, products = _cycle(table, base.IN_CYCLE, with_blocks=False)
        assert context.missing_families() == ("prev10", "prev20", "ensemble")
        _scored, features = _on_demand(table, context, products, *base.OFF_CYCLE)
        for name in R4A_COLUMNS:
            assert features[name] is None, name

    def test_the_engine_counts_incomplete_on_demand_rows(
        self, minimal_config: Config,
    ) -> None:
        engine = base._engine(minimal_config)  # the old fixture: no blocks
        assert engine._postprocess_context.missing_families()
        assert engine._pp_incomplete_served == 0
        result = engine.postprocess_point(*base.OFF_CYCLE)
        assert result is not None and not result.reused
        assert engine._pp_incomplete_served == 1

    def test_read_points_keeps_no_grids_without_a_model(self) -> None:
        geo = base.LinearGeo()
        native = [geo.lonlat_to_grid(lon, lat) for lat, lon in base.IN_CYCLE]
        points = compute_mod._read_points(
            _raw_products(), native, keep_grids=False,
            ensemble=_ensemble(), threshold_mm_h=0.5,
        )
        assert points is not None
        assert points.ens_grids is None
        assert points.ens_features  # the served points still get theirs


# ---------------------------------------------------------------------------
# 3. The never-NaN-in-training counter
# ---------------------------------------------------------------------------


def _stump_model(names_missing: dict[str, int]) -> pp.PostprocessModel:
    """``base._model()`` re-dressed as trees: one stump per named column."""
    model = base._model()
    names = list(model.feature_names)
    width = len(names)
    trees = []
    for name, missing in names_missing.items():
        trees.append(pt.Tree(
            feature=np.array([names.index(name), -1, -1], dtype=np.int32),
            threshold=np.array([0.5, 0.0, 0.0]),
            left=np.array([1, -1, -1], dtype=np.int32),
            right=np.array([2, -1, -1], dtype=np.int32),
            value=np.array([0.0, -0.3, 0.3]),
            default_left=np.array([True, False, False]),
            missing=np.array([missing, 0, 0], dtype=np.int8),
        ))
    ensemble = pt.TreeEnsemble(
        trees=tuple(trees), n_features=width, feature_names=tuple(names),
    )
    models = {
        lead: pp.LeadModel(
            lead_min=lead, intercept=0.0, coefficients=(),
            isotonic=lm.isotonic, n=lm.n, base_rate=lm.base_rate,
            converged=True, iterations=1, trees=ensemble,
        )
        for lead, lm in model.models.items()
    }
    return pp.PostprocessModel(
        leads=model.leads, design_leads=model.design_leads,
        feature_names=model.feature_names, standardiser=model.standardiser,
        models=models, l2=model.l2, fitted_at_utc=model.fitted_at_utc,
        training=model.training, kind=pp.KIND_TREES, spec=model.spec,
    )


class TestTheBlindNanCounter:
    def _table(self, tmp_path: Path) -> PostprocessTable:
        path = tmp_path / "postprocess.json"
        path.write_text(_stump_model({
            "log1p_observed_mm_h": pt.MISSING_NONE,
            "log1p_obs_max_5km_mm_h": pt.MISSING_NAN,
        }).dumps())
        table = PostprocessTable(path)
        table.load()
        assert table.active
        return table

    def test_blind_columns_come_from_the_node_missing_types(
        self, tmp_path: Path,
    ) -> None:
        table = self._table(tmp_path)
        assert table.nan_blind_columns == frozenset({"log1p_observed_mm_h"})

    def test_a_logistic_has_no_blind_columns(
        self, minimal_config: Config,
    ) -> None:
        assert base._table(minimal_config).nan_blind_columns == frozenset()

    def test_the_cycle_counts_nan_cells_in_blind_columns(
        self, tmp_path: Path,
    ) -> None:
        table = self._table(tmp_path)
        keys = [(55.0, 10.0), (55.1, 10.1), (55.2, 10.2)]
        grid = {
            "obs_max_5km_mm_h": np.array([np.nan, 1.0, 2.0], dtype=np.float32),
        }
        cycle = build_cycle_postprocess(
            table,
            radar_ts_utc=base.RADAR_TS,
            generated_at_utc=base.GENERATED_AT,
            keys=keys,
            grid_features=grid,
            raw_fractions={lead: (0.1, 0.2, 0.3) for lead in LEADS},
            shared=[
                {"observed_mm_h": None, "eta_min": 5.0, "intensity_mm_h": 1.0},
                {"observed_mm_h": 1.0, "eta_min": 5.0, "intensity_mm_h": 1.0},
                {"observed_mm_h": None, "eta_min": 5.0, "intensity_mm_h": 1.0},
            ],
            station_radar_km=[50.0, 50.0, 50.0],
            leads=LEADS,
            season="summer",
            hour_utc=12,
            frame_age_min=14.0,
        )
        # Two NaN observed rows in the blind column; the NaN in the
        # NaN-aware column is not counted.
        assert cycle.blind_nan == {"log1p_observed_mm_h": 2}
        assert cycle.onset_blind_nan == {}
