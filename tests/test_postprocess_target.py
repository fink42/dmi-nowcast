"""The fit target travels with the model, the way the protocol does.

``scripts/fit_postprocess.py --target onset`` fits the same design on a
different outcome — a gauge onset inside the push scorer's window instead
of "wet within L". The document has to say which one its probabilities
are OF: a threshold fitted on one means nothing on the other. The field
is additive (default ``wet``), the schema version does not move, and a
document from before it existed loads as the wet model it is.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from dmi_nowcast_core import postprocess as pp

LEADS = (20, 30)
DESIGN_LEADS = (10, 20, 30)


def _rows(n: int = 1200, seed: int = 3) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    t = (
        np.datetime64("2026-02-01T00:00:00").astype("datetime64[s]").astype(np.int64)
        + np.arange(n) * 600
    )
    signal = rng.uniform(size=n)
    rows: dict = {
        "season": pp.seasons_from_epoch(t),
        "hour_utc": pp.hours_from_epoch(t).astype(np.float64),
        "up_dist_km": np.where(signal > 0.2, 40.0 * (1.0 - signal), np.nan),
        "up_max_20km_mm_h": signal * 6.0,
        "bulk_kmh": rng.uniform(5.0, 60.0, size=n),
        "station_radar_km": rng.uniform(10.0, 90.0, size=n),
    }
    for lead in DESIGN_LEADS:
        rows[pp.raw_fraction_column(lead)] = np.clip(
            signal * (lead / 60.0) + rng.normal(0, 0.03, size=n), 0.0, 1.0,
        )
    # A rare target, like the onset one: a few per cent positives.
    p = np.clip(0.005 + 0.06 * signal, 0.0, 1.0)
    truth = {
        lead: (
            (rng.uniform(size=n) < p * lead / 20.0).astype(float),
            np.ones(n, dtype=bool),
        )
        for lead in LEADS
    }
    return rows, truth


@pytest.fixture(scope="module")
def dataset() -> tuple[dict, dict]:
    return _rows()


def test_the_default_target_is_wet(dataset) -> None:
    rows, truth = dataset
    model = pp.fit_postprocess(rows, truth, LEADS, design_leads=DESIGN_LEADS)
    assert model.target == pp.TARGET_WET
    payload = json.loads(model.dumps())
    assert payload["target"] == pp.TARGET_WET
    assert payload["training"]["target"] == pp.TARGET_WET


def test_onset_is_top_level_and_survives_a_round_trip(dataset) -> None:
    rows, truth = dataset
    model = pp.fit_postprocess(
        rows, truth, LEADS, design_leads=DESIGN_LEADS, target=pp.TARGET_ONSET,
    )
    payload = json.loads(model.dumps())
    assert payload["target"] == pp.TARGET_ONSET
    assert payload["training"]["target"] == pp.TARGET_ONSET
    # Additive: the schema version does not move for a key with a default.
    assert payload["schema_version"] == pp.SCHEMA_VERSION
    restored = pp.PostprocessModel.loads(model.dumps())
    assert restored.target == pp.TARGET_ONSET
    # The target is provenance for serving: same predictions either way.
    again = restored.predict(rows)
    for lead in LEADS:
        np.testing.assert_allclose(again[lead], model.predict(rows)[lead])


def test_a_document_from_before_the_field_loads_as_wet(dataset) -> None:
    rows, truth = dataset
    model = pp.fit_postprocess(rows, truth, LEADS, design_leads=DESIGN_LEADS)
    payload = json.loads(model.dumps())
    del payload["target"]
    payload["training"].pop("target", None)
    assert pp.PostprocessModel.from_json(payload).target == pp.TARGET_WET


def test_the_provenance_block_is_the_fallback(dataset) -> None:
    rows, truth = dataset
    model = pp.fit_postprocess(
        rows, truth, LEADS, design_leads=DESIGN_LEADS, target=pp.TARGET_ONSET,
    )
    payload = json.loads(model.dumps())
    del payload["target"]
    assert pp.PostprocessModel.from_json(payload).target == pp.TARGET_ONSET


def test_an_explicit_training_target_is_not_overwritten(dataset) -> None:
    rows, truth = dataset
    model = pp.fit_postprocess(
        rows, truth, LEADS, design_leads=DESIGN_LEADS, target=pp.TARGET_ONSET,
        training={"rows": 1, "target": pp.TARGET_ONSET, "onset_rows": "dry"},
    )
    assert model.training["onset_rows"] == "dry"
    assert model.training["target"] == pp.TARGET_ONSET


def test_an_unknown_target_is_refused_at_the_fit(dataset) -> None:
    rows, truth = dataset
    with pytest.raises(ValueError, match="unknown target"):
        pp.fit_postprocess(
            rows, truth, LEADS, design_leads=DESIGN_LEADS, target="moist",
        )


def test_the_write_back_column_follows_the_target() -> None:
    assert pp.target_column_template(pp.TARGET_WET) == "p_post_{lead}"
    assert pp.target_column_template(pp.TARGET_ONSET) == "p_onset_{lead}"
    assert pp.ONSET_COLUMN_TEMPLATE.format(lead=30) == "p_onset_30"
    with pytest.raises(ValueError, match="unknown target"):
        pp.target_column_template("moist")


def test_per_season_isotonic_copes_with_a_rare_target() -> None:
    """A few per cent positives: every curve is monotone and in [0, 1].

    The per-season floor is a ROW count; a season at the floor with a
    1.8 % base rate has ~90 positives over 200 quantile bins. PAVA pools
    the bins into a coarse, monotone step curve — noisy, but valid.
    """
    rng = np.random.default_rng(0)
    n = 3 * pp.MIN_SEASON_ISOTONIC_ROWS
    score = rng.uniform(size=n)
    y = (rng.uniform(size=n) < 0.036 * score).astype(float)
    season = np.repeat(np.array(pp.SEASONS), n // 3)
    pooled, curves = pp.fit_isotonic_by_season(score, y, season)
    assert set(curves) == set(pp.SEASONS)
    for curve in (pooled, *curves.values()):
        values = np.asarray(curve.calibrated_values)
        assert np.all(np.diff(values) >= -1e-12)
        assert values.min() >= 0.0 and values.max() <= 1.0
