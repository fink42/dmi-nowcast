"""Post-processing v2 on the SERVING side: what the sidecar has to swallow.

``sidecar/tests/test_push_postprocess.py`` owns the cycle — the features,
the points, the fan-out. This file owns the narrow question the v2 track
opens up: the artefact can now describe four model families and two
designs, the running process has to load any of them without knowing
which, and the nightly refit has to keep defaulting to exactly the model
that has been in service since 2026-09-11.

Nothing here fits a tree. LightGBM is not in this image and never will be
(``dmi_nowcast_core.postprocess_trees``); what IS in the image is the
numpy evaluator, so the tree half is exercised by handing the table a
document with a hand-built ensemble in it — which is precisely the
situation on the public instance, where the model arrives over the sync
and nothing local ever fitted it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from dmi_nowcast_core import postprocess as pp
from dmi_nowcast_core import postprocess_trees as pt
from dmi_nowcast_sidecar import quality_job
from dmi_nowcast_sidecar.config import FitPostprocessConfig
from dmi_nowcast_sidecar.postprocess_fit import (
    PostprocessFitOptions,
    options_from_json,
    options_to_json,
)
from dmi_nowcast_sidecar.push.postprocess import PostprocessTable

LEADS = (20, 30, 45, 60)
DESIGN_LEADS = (10, 20, 30, 45, 60)


def _rows(n: int = 3000, seed: int = 5) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    t = (
        np.datetime64("2026-03-01T06:00:00").astype("datetime64[s]").astype(np.int64)
        + np.arange(n) * 600
    )
    signal = rng.uniform(size=n)
    rows: dict = {
        "season": pp.seasons_from_epoch(t),
        "hour_utc": pp.hours_from_epoch(t).astype(np.float64),
        "up_dist_km": np.where(signal > 0.2, 40.0 * (1.0 - signal), np.nan),
        "bulk_kmh": rng.uniform(5.0, 60.0, size=n),
        "observed_mm_h": np.clip(rng.normal(0.4, 0.6, size=n), 0.0, None),
        "station_id": np.array([f"S{i % 5:02d}" for i in range(n)]),
    }
    for lead in DESIGN_LEADS:
        rows[pp.raw_fraction_column(lead)] = np.clip(
            signal * (lead / 60.0) + rng.normal(0, 0.02, size=n), 0.0, 1.0,
        )
    truth = {
        lead: (
            (rng.uniform(size=n) < np.clip(signal * lead / 60.0, 0, 1)).astype(float),
            np.ones(n, dtype=bool),
        )
        for lead in LEADS
    }
    return rows, truth


def _fitted(kind: str = pp.KIND_LOGISTIC, design: str = pp.DESIGN_V1):
    rows, truth = _rows()
    model = pp.fit_postprocess(
        rows, truth, LEADS, design_leads=DESIGN_LEADS,
        settings=pp.FitSettings(kind=kind, design=design),
        fitted_at=datetime(2026, 9, 16, 3, 40, tzinfo=timezone.utc),
    )
    return model, rows


def _as_trees(model: pp.PostprocessModel) -> pp.PostprocessModel:
    """Turn a fitted logistic into a TREE document with the same shape.

    A stump per lead, so the numbers are arithmetic and the point of the
    test is the LOADING and the EVALUATION path, not the fit. This is what
    a synced-in tree model looks like to a process that has no LightGBM.
    """
    width = len(model.feature_names)
    models = {}
    for index, (lead, lead_model) in enumerate(sorted(model.models.items())):
        ensemble = pt.TreeEnsemble(
            trees=(
                pt.Tree(
                    feature=np.array([0, -1, -1], dtype=np.int32),
                    threshold=np.array([0.0, 0.0, 0.0]),
                    left=np.array([1, -1, -1], dtype=np.int32),
                    right=np.array([2, -1, -1], dtype=np.int32),
                    value=np.array([0.0, -1.0, 1.0 + index]),
                    default_left=np.array([True, False, False]),
                    missing=np.array([pt.MISSING_NAN, 0, 0], dtype=np.int8),
                ),
            ),
            n_features=width,
            feature_names=model.feature_names,
        )
        models[lead] = pp.LeadModel(
            lead_min=lead, intercept=0.0, coefficients=(),
            isotonic=lead_model.isotonic, n=lead_model.n,
            base_rate=lead_model.base_rate, converged=True, iterations=1,
            trees=ensemble,
        )
    return pp.PostprocessModel(
        leads=model.leads, design_leads=model.design_leads,
        feature_names=model.feature_names, standardiser=model.standardiser,
        models=models, l2=model.l2, fitted_at_utc=model.fitted_at_utc,
        training=model.training, kind=pp.KIND_TREES, spec=model.spec,
    )


# ---------------------------------------------------------------------------
# The table loads whatever the artefact says
# ---------------------------------------------------------------------------


class TestTheTableLoadsEitherKind:
    def test_a_logistic_document_loads_as_it_always_did(self, tmp_path: Path) -> None:
        model, rows = _fitted()
        path = tmp_path / "postprocess.json"
        path.write_text(model.dumps())
        table = PostprocessTable(path)
        table.load()
        assert table.active
        assert table.leads == list(LEADS)
        assert table.design_leads == list(DESIGN_LEADS)
        assert table.fitted_at_utc == model.fitted_at_utc
        served = table.predict_table(rows)
        assert set(served) == set(LEADS)

    def test_a_tree_document_loads_and_scores(self, tmp_path: Path) -> None:
        """No LightGBM in this image; the numpy evaluator is all it needs."""
        model, rows = _fitted()
        path = tmp_path / "postprocess.json"
        path.write_text(_as_trees(model).dumps())
        table = PostprocessTable(path)
        table.load()
        assert table.active
        assert table.model.kind == pp.KIND_TREES
        assert table.leads == list(LEADS)
        served = table.predict_table(rows)
        assert set(served) == set(LEADS)
        for values in served.values():
            assert values.shape == (rows["bulk_kmh"].size,)
            assert np.all((values >= 0.0) & (values <= 1.0))

    def test_a_v2_design_document_loads_and_scores(self, tmp_path: Path) -> None:
        model, rows = _fitted(design=pp.DESIGN_V2)
        path = tmp_path / "postprocess.json"
        path.write_text(model.dumps())
        table = PostprocessTable(path)
        table.load()
        assert table.active
        assert table.model.spec.version == pp.DESIGN_V2
        assert set(table.predict_table(rows)) == set(LEADS)

    def test_a_shared_logistic_document_loads_and_scores(
        self, tmp_path: Path,
    ) -> None:
        model, rows = _fitted(kind=pp.KIND_LOGISTIC_SHARED)
        path = tmp_path / "postprocess.json"
        path.write_text(model.dumps())
        table = PostprocessTable(path)
        table.load()
        assert table.model.kind == pp.KIND_LOGISTIC_SHARED
        assert table.model.is_shared
        served = table.predict_table(rows)
        assert set(served) == set(LEADS)

    @pytest.mark.parametrize(
        "kind", [pp.KIND_LOGISTIC, pp.KIND_LOGISTIC_SHARED],
    )
    def test_the_served_probabilities_never_fall_with_the_lead(
        self, tmp_path: Path, kind: str,
    ) -> None:
        """A 60-minute window contains the 45-minute one. Both APIs agree."""
        model, rows = _fitted(kind=kind)
        path = tmp_path / "postprocess.json"
        path.write_text(model.dumps())
        table = PostprocessTable(path)
        table.load()
        served = table.predict_table(rows)
        for shorter, longer in zip(LEADS, LEADS[1:]):
            assert np.all(served[longer] >= served[shorter] - 1e-12)
        # The single-point API is the one a subscriber's forecast goes
        # through, and it must not disagree with the table.
        row = {name: values[0] for name, values in rows.items()}
        single = table.predict_row(row)
        ordered = [single[lead] for lead in LEADS]
        assert ordered == sorted(ordered)
        assert table.predict(30, row) == pytest.approx(single[30])

    def test_an_unusable_document_still_degrades_to_the_curve(
        self, tmp_path: Path,
    ) -> None:
        """Every new field is additive; junk is still junk, and still safe."""
        path = tmp_path / "postprocess.json"
        payload = json.loads(_fitted()[0].dumps())
        payload["kind"] = "a family this build has never heard of"
        payload["models"]["20"]["trees"] = {"schema_version": 99, "trees": []}
        path.write_text(json.dumps(payload))
        table = PostprocessTable(path)
        table.load()
        assert not table.active
        assert table.model is None


# ---------------------------------------------------------------------------
# The nightly refit keeps its defaults
# ---------------------------------------------------------------------------


def test_the_nightly_options_default_to_the_model_in_service() -> None:
    options = PostprocessFitOptions(
        decisions_dirs=[], corpus_dir=Path("/nowhere"), out=Path("/nowhere/m.json"),
    )
    assert options.model == "logistic"
    assert options.design == "v1"
    assert options.station_offsets is False
    assert options.isotonic == "pooled"


def test_the_config_defaults_to_the_model_in_service() -> None:
    settings = FitPostprocessConfig()
    assert settings.model == "logistic"
    assert settings.design == "v1"
    assert settings.station_offsets is False
    assert settings.isotonic == "pooled"


def test_the_config_refuses_a_family_this_build_cannot_name() -> None:
    with pytest.raises(ValueError):
        FitPostprocessConfig(model="randomforest")
    with pytest.raises(ValueError):
        FitPostprocessConfig(design="v3")
    with pytest.raises(ValueError):
        FitPostprocessConfig(isotonic="loess")


def test_the_new_options_cross_the_process_boundary() -> None:
    options = PostprocessFitOptions(
        decisions_dirs=[Path("/a"), Path("/b")],
        corpus_dir=Path("/corpus"), out=Path("/out/m.json"),
        model="trees-shared", design="v2", station_offsets=True,
        isotonic="per-season",
    )
    restored = options_from_json(json.loads(json.dumps(options_to_json(options))))
    assert restored == options


def test_an_older_child_ignores_a_key_it_does_not_know() -> None:
    """A deploy between a parent restart and a container restart."""
    payload = {
        "decisions_dirs": ["/a"], "corpus_dir": "/c", "out": "/o/m.json",
        "model": "logistic", "something_from_the_future": 7,
    }
    restored = options_from_json(payload)
    assert restored.model == "logistic"


# ---------------------------------------------------------------------------
# The rollback
# ---------------------------------------------------------------------------


def _options(out: Path) -> dict:
    """The smallest options block ``options_from_json`` accepts."""
    return {
        "decisions_dirs": [str(out.parent)],
        "corpus_dir": str(out.parent),
        "out": str(out),
    }


class TestThePreviousCopy:
    def test_the_previous_model_is_kept_beside_the_new_one(
        self, tmp_path: Path,
    ) -> None:
        target = tmp_path / "postprocess.json"
        target.write_text('{"yesterday": true}\n')
        kept = quality_job.keep_previous(target)
        assert kept == tmp_path / "postprocess.prev.json"
        assert kept.read_text() == '{"yesterday": true}\n'

    def test_the_first_ever_fit_has_nothing_to_keep(self, tmp_path: Path) -> None:
        assert quality_job.keep_previous(tmp_path / "postprocess.json") is None

    def test_the_refit_copies_before_it_overwrites(self, tmp_path: Path) -> None:
        """The whole point: rolling back is a `cp`, not an eight-hour refit."""
        target = tmp_path / "postprocess.json"
        target.write_text('{"yesterday": true}\n')
        model, _rows = _fitted()
        summary = quality_job.run_postprocess_fit(
            {"out": str(target), "options": _options(target)},
            fitter=lambda _opts: {"model": model, "summary": {"rows": 1}},
        )
        assert summary["postprocess_path"] == str(target)
        assert summary["postprocess_previous_path"] == str(
            tmp_path / "postprocess.prev.json"
        )
        assert (tmp_path / "postprocess.prev.json").read_text() == (
            '{"yesterday": true}\n'
        )
        assert json.loads(target.read_text())["kind"] == "logistic"

    def test_a_failed_refit_leaves_both_files_alone(self, tmp_path: Path) -> None:
        target = tmp_path / "postprocess.json"
        target.write_text('{"yesterday": true}\n')

        def boom(_options):
            raise RuntimeError("no rows")

        summary = quality_job.run_postprocess_fit(
            {"out": str(target), "options": _options(target)}, fitter=boom,
        )
        assert "postprocess_error" in summary
        assert target.read_text() == '{"yesterday": true}\n'
        assert not (tmp_path / "postprocess.prev.json").exists()
