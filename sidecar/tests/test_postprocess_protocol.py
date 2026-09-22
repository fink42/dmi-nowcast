"""The serving side of the random-point protocol, and the loud failure.

Two things the running process has to get right about a model fitted with
the point's own gauge masked away (``--protocol random-point``):

1. **It has to keep masking.** The model itself does that now
   (``PostprocessModel.design``), so the job of this file is to prove that
   the SIDECAR's paths inherit it — the cycle's table and the single-row
   twin the on-demand ``/forecast`` lookup goes through — because those
   are the two places a live ``g_min_since_wet`` could reach a model that
   was taught the column is always unknown.
2. **It has to refuse a document it cannot score.** Until the load-time
   check existed, a document whose design width did not match its own
   models raised inside every predict, was caught, and turned into a
   curve-calibrated answer plus one ``push_postprocess_predict_failed``
   line per cycle — a service that had silently stopped using its model
   and mentioned it only in a stream of identical five-minute warnings.
   One ``push_postprocess_model_invalid`` at load, and inactive, is the
   difference between a fault and a fault somebody sees.

Nothing here fits a tree. LightGBM is not in this image; the documents
below carry a hand-built stump that splits on ``g_min_since_wet`` and
sends a missing value the OTHER way, so "was the gauge masked?" is a
question with two visibly different answers. That is also exactly the
shape the public instance sees: a tree model that arrived over the sync
from a fit nothing local ever ran.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import structlog

from dmi_nowcast_core import postprocess as pp
from dmi_nowcast_core import postprocess_trees as pt
from dmi_nowcast_core.calibrate import IsotonicCalibrator
from dmi_nowcast_sidecar.push.postprocess import PostprocessTable

LEADS = (20, 30)
DESIGN_LEADS = (10, 20, 30)

#: The own-gauge reading the protocol says the model may not see: rain in
#: the last ten minutes and a gauge that went wet two minutes ago.
LIVE_GAUGE = {
    "g_mm_10": 1.6, "g_mm_30": 3.1, "g_mm_60": 4.4,
    "g_min_since_wet": 2.0, "g_dry_60": 0.0, "g_known": 1.0,
}

#: An identity recalibration, so the served number IS the ensemble's
#: probability and a reader can check it with a sigmoid by hand.
IDENTITY = IsotonicCalibrator(
    raw_breakpoints=(0.0, 1.0), calibrated_values=(0.0, 1.0),
)

#: The stump's split on ``g_min_since_wet``. A live gauge that was wet two
#: minutes ago is BELOW it and goes left; a masked NaN goes right, because
#: the node's default is right. Hence two different probabilities.
_SINCE_WET_SPLIT = 30.0


def _rows(n: int = 400, seed: int = 3) -> dict:
    """Live-shaped feature columns, own gauge included and reading wet."""
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
        "observed_mm_h": rng.uniform(0.0, 3.0, size=n),
        "station_radar_km": rng.uniform(10.0, 90.0, size=n),
        "station_id": np.array([f"S{i % 6:02d}" for i in range(n)]),
        "ng_near_km": rng.uniform(2.0, 45.0, size=n),
        "ng_mm_10_idw": rng.uniform(0.0, 2.0, size=n),
    }
    for name, value in LIVE_GAUGE.items():
        rows[name] = np.full(n, float(value), dtype=np.float64)
    for lead in DESIGN_LEADS:
        rows[pp.raw_fraction_column(lead)] = np.clip(
            signal * (lead / 60.0), 0.0, 1.0,
        )
    return rows


def _stump(feature: int, *, left: float, right: float) -> pt.Tree:
    """One split. NaN goes RIGHT — the direction a masked gauge takes."""
    return pt.Tree(
        feature=np.array([feature, -1, -1], dtype=np.int32),
        threshold=np.array([_SINCE_WET_SPLIT, 0.0, 0.0], dtype=np.float64),
        left=np.array([1, -1, -1], dtype=np.int32),
        right=np.array([2, -1, -1], dtype=np.int32),
        value=np.array([0.0, left, right], dtype=np.float64),
        default_left=np.array([False, False, False]),
        missing=np.array([pt.MISSING_NAN, 0, 0], dtype=np.int8),
    )


def _document(protocol: str) -> pp.PostprocessModel:
    """A v2 tree document under ``protocol``, split on the own gauge.

    The two arms differ ONLY in the protocol string, which is the point:
    every number below that changes between them changed because of the
    mask and for no other reason.
    """
    rows = _rows()
    spec = pp.design_spec(rows, DESIGN_LEADS, version=pp.DESIGN_V2)
    names = pp.design_columns(DESIGN_LEADS, spec)
    width = len(names)
    index = list(names).index("g_min_since_wet")
    ensemble = pt.TreeEnsemble(
        trees=(_stump(index, left=-2.0, right=2.0),),
        n_features=width,
        feature_names=tuple(names),
    )
    models = {
        lead: pp.LeadModel(
            lead_min=lead, intercept=0.0, coefficients=(),
            isotonic=IDENTITY, n=1000, base_rate=0.3,
            converged=True, iterations=1, trees=ensemble,
        )
        for lead in LEADS
    }
    return pp.PostprocessModel(
        leads=LEADS, design_leads=DESIGN_LEADS, feature_names=tuple(names),
        standardiser=pp.Standardiser(
            mean=tuple([0.0] * width), scale=tuple([1.0] * width),
        ),
        models=models, l2=1.0,
        fitted_at_utc=datetime(
            2026, 9, 22, 3, 40, tzinfo=timezone.utc,
        ).isoformat(timespec="seconds"),
        training={"rows": 1000, "protocol": protocol},
        kind=pp.KIND_TREES, spec=spec, protocol=protocol,
    )


def _write(path: Path, model: pp.PostprocessModel) -> Path:
    path.write_text(model.dumps())
    return path


#: sigmoid(-2) and sigmoid(2): what the stump returns for a live gauge and
#: for a masked one. The isotonic is the identity, so these ARE the served
#: probabilities, and no assertion below has to trust the fit to produce
#: a spread.
LIVE_P = 1.0 / (1.0 + np.exp(2.0))
MASKED_P = 1.0 / (1.0 + np.exp(-2.0))


# ---------------------------------------------------------------------------
# The table serves the protocol it was handed
# ---------------------------------------------------------------------------


class TestTheTableHonoursTheProtocol:
    def test_it_loads_a_random_point_document_and_says_so(
        self, tmp_path: Path,
    ) -> None:
        path = _write(tmp_path / "m.json", _document(pp.PROTOCOL_RANDOM_POINT))
        table = PostprocessTable(path)
        with structlog.testing.capture_logs() as logs:
            assert table.load() is not None
        assert table.active
        assert table.model.protocol == pp.PROTOCOL_RANDOM_POINT
        assert table.model.masks_own_gauge is True
        # One line, and it names the protocol beside the kind and the
        # design: the three facts that decide what the served numbers mean.
        assert [e["event"] for e in logs] == ["push_postprocess_loaded"]
        loaded = logs[0]
        assert loaded["protocol"] == pp.PROTOCOL_RANDOM_POINT
        assert loaded["masks_own_gauge"] is True
        assert loaded["kind"] == pp.KIND_TREES
        assert loaded["design"] == pp.DESIGN_V2

    def test_an_at_gauge_document_says_the_other_thing(
        self, tmp_path: Path,
    ) -> None:
        path = _write(tmp_path / "m.json", _document(pp.PROTOCOL_AT_GAUGE))
        table = PostprocessTable(path)
        with structlog.testing.capture_logs() as logs:
            table.load()
        assert table.active
        assert logs[0]["protocol"] == pp.PROTOCOL_AT_GAUGE
        assert logs[0]["masks_own_gauge"] is False

    def test_the_cycle_table_is_masked(self, tmp_path: Path) -> None:
        """``predict_table`` — the path every served point goes through."""
        rows = _rows()
        masked = PostprocessTable(
            _write(tmp_path / "rp.json", _document(pp.PROTOCOL_RANDOM_POINT)),
        )
        masked.load()
        served = masked.predict_table(rows)
        assert set(served) == set(LEADS)
        for lead in LEADS:
            np.testing.assert_allclose(
                served[lead], np.full(rows["bulk_kmh"].size, MASKED_P),
                atol=1e-6,
            )

    def test_the_at_gauge_twin_reads_the_gauge(self, tmp_path: Path) -> None:
        """The control. Same document, other protocol, other number.

        If this came out at ``MASKED_P`` too, the test above would be
        proving nothing — the stump would simply never have seen the gauge.
        """
        rows = _rows()
        plain = PostprocessTable(
            _write(tmp_path / "ag.json", _document(pp.PROTOCOL_AT_GAUGE)),
        )
        plain.load()
        served = plain.predict_table(rows)
        for lead in LEADS:
            np.testing.assert_allclose(
                served[lead], np.full(rows["bulk_kmh"].size, LIVE_P),
                atol=1e-6,
            )
        assert abs(LIVE_P - MASKED_P) > 0.5

    def test_the_single_row_paths_are_masked_too(self, tmp_path: Path) -> None:
        """``predict`` and ``predict_row``: the on-demand forecast lookup.

        A subscriber's ``/forecast`` goes through ``predict_row``, and the
        gauge-eval scoreboard scores stations through it as well. A mask
        that held for the table and not for the row would put the two
        numbers for the same point at the same instant a long way apart.
        """
        row = {name: values[0] for name, values in _rows().items()}
        masked = PostprocessTable(
            _write(tmp_path / "rp.json", _document(pp.PROTOCOL_RANDOM_POINT)),
        )
        masked.load()
        plain = PostprocessTable(
            _write(tmp_path / "ag.json", _document(pp.PROTOCOL_AT_GAUGE)),
        )
        plain.load()
        for lead in LEADS:
            assert masked.predict(lead, row) == pytest.approx(MASKED_P, abs=1e-6)
            assert plain.predict(lead, row) == pytest.approx(LIVE_P, abs=1e-6)
        assert masked.predict_row(row) == pytest.approx(
            {lead: MASKED_P for lead in LEADS}, abs=1e-6,
        )
        assert plain.predict_row(row) == pytest.approx(
            {lead: LIVE_P for lead in LEADS}, abs=1e-6,
        )

    def test_a_row_with_no_gauge_at_all_scores_the_masked_way(
        self, tmp_path: Path,
    ) -> None:
        """Most served points are addresses: no own gauge, ever.

        They already looked to the model exactly like a masked gauge
        station, and they still do — the mask did not invent a third shape.
        """
        row = {
            name: values[0] for name, values in _rows().items()
            if name not in LIVE_GAUGE
        }
        table = PostprocessTable(
            _write(tmp_path / "rp.json", _document(pp.PROTOCOL_RANDOM_POINT)),
        )
        table.load()
        for lead in LEADS:
            assert table.predict(lead, row) == pytest.approx(MASKED_P, abs=1e-6)

    def test_the_protocol_survives_a_document_that_only_says_it_inside_training(
        self, tmp_path: Path,
    ) -> None:
        """The artefacts already fitted: masked, no top-level key."""
        payload = json.loads(_document(pp.PROTOCOL_RANDOM_POINT).dumps())
        del payload["protocol"]
        path = tmp_path / "m.json"
        path.write_text(json.dumps(payload))
        table = PostprocessTable(path)
        table.load()
        assert table.model.protocol == pp.PROTOCOL_RANDOM_POINT
        for lead in LEADS:
            assert table.predict_table(_rows())[lead][0] == pytest.approx(
                MASKED_P, abs=1e-6,
            )


# ---------------------------------------------------------------------------
# A document it cannot score is refused once, not per cycle
# ---------------------------------------------------------------------------


class TestAnUnscorableDocumentIsRefusedAtLoad:
    def test_a_width_mismatch_is_inactive_and_one_log_line(
        self, tmp_path: Path,
    ) -> None:
        """The skew that actually happens: the design moved, the fit did not.

        A column leaves the catalogue, or a document is synced from a build
        with a different design, and ``features.names`` no longer counts
        what the ensembles were fitted on. Before the load-time check this
        was a permanent fallback to the curve announced only by a warning
        per cycle.
        """
        payload = json.loads(_document(pp.PROTOCOL_RANDOM_POINT).dumps())
        width = len(payload["features"]["names"])
        payload["features"]["names"] = payload["features"]["names"][:-1]
        path = tmp_path / "m.json"
        path.write_text(json.dumps(payload))
        table = PostprocessTable(path)
        with structlog.testing.capture_logs() as logs:
            assert table.load() is None
        assert table.active is False
        assert table.model is None
        assert [e["event"] for e in logs] == ["push_postprocess_model_invalid"]
        reason = logs[0]["reason"]
        assert f"fitted on {width} column(s)" in reason
        assert f"design names {width - 1}" in reason
        assert logs[0]["protocol"] == pp.PROTOCOL_RANDOM_POINT
        # And the fallback is silent from here on: the engine asks, gets
        # nothing, and uses the curve — no per-cycle predict warning.
        with structlog.testing.capture_logs() as quiet:
            assert table.predict_table(_rows()) == {}
            assert table.predict(20, {"observed_mm_h": 1.0}) is None
            assert table.predict_row({"observed_mm_h": 1.0}) == {}
        assert quiet == []

    def test_a_standardiser_that_does_not_fit_the_design_is_refused(
        self, tmp_path: Path,
    ) -> None:
        """A logistic scores ``design @ coefficients`` after standardising.

        Both arrays have to be the design's width, and a document where
        they are not is not a model that can be served.
        """
        payload = json.loads(_document(pp.PROTOCOL_AT_GAUGE).dumps())
        payload["kind"] = pp.KIND_LOGISTIC
        for entry in payload["models"].values():
            entry.pop("trees", None)
            entry["coefficients"] = [0.0] * len(payload["features"]["names"])
        payload["scaling"]["mean"] = payload["scaling"]["mean"][:-1]
        path = tmp_path / "m.json"
        path.write_text(json.dumps(payload))
        table = PostprocessTable(path)
        with structlog.testing.capture_logs() as logs:
            table.load()
        assert table.active is False
        assert [e["event"] for e in logs] == ["push_postprocess_model_invalid"]
        assert "standardiser" in logs[0]["reason"]

    def test_a_logistic_whose_coefficients_are_the_wrong_length_is_refused(
        self, tmp_path: Path,
    ) -> None:
        payload = json.loads(_document(pp.PROTOCOL_AT_GAUGE).dumps())
        payload["kind"] = pp.KIND_LOGISTIC
        width = len(payload["features"]["names"])
        for entry in payload["models"].values():
            entry.pop("trees", None)
            entry["coefficients"] = [0.0] * (width - 2)
        path = tmp_path / "m.json"
        path.write_text(json.dumps(payload))
        table = PostprocessTable(path)
        with structlog.testing.capture_logs() as logs:
            table.load()
        assert table.active is False
        assert [e["event"] for e in logs] == ["push_postprocess_model_invalid"]
        assert f"fitted on {width - 2} column(s)" in logs[0]["reason"]

    def test_a_protocol_this_build_does_not_know_is_refused(
        self, tmp_path: Path,
    ) -> None:
        """Forward skew, and the safe direction to fail in.

        A protocol invented after this build was cut would be served as if
        it were at-gauge — which for any future masked protocol means
        feeding a model rows it was never fitted on. Better inactive.
        """
        payload = json.loads(_document(pp.PROTOCOL_AT_GAUGE).dumps())
        payload["protocol"] = "random-point-v3"
        path = tmp_path / "m.json"
        path.write_text(json.dumps(payload))
        table = PostprocessTable(path)
        with structlog.testing.capture_logs() as logs:
            table.load()
        assert table.active is False
        assert [e["event"] for e in logs] == ["push_postprocess_model_invalid"]
        assert "random-point-v3" in logs[0]["reason"]

    def test_a_document_with_no_design_column_is_refused(
        self, tmp_path: Path,
    ) -> None:
        payload = json.loads(_document(pp.PROTOCOL_AT_GAUGE).dumps())
        payload["features"]["names"] = []
        path = tmp_path / "m.json"
        path.write_text(json.dumps(payload))
        table = PostprocessTable(path)
        with structlog.testing.capture_logs() as logs:
            table.load()
        assert table.active is False
        assert [e["event"] for e in logs] == ["push_postprocess_model_invalid"]
        assert "names no design column" in logs[0]["reason"]

    def test_a_good_document_is_not_refused(self, tmp_path: Path) -> None:
        """The check has to pass the thing it exists to guard."""
        for protocol in pp.PROTOCOLS:
            table = PostprocessTable(
                _write(tmp_path / f"{protocol}.json", _document(protocol)),
            )
            with structlog.testing.capture_logs() as logs:
                assert table.load() is not None
            assert [e["event"] for e in logs] == ["push_postprocess_loaded"]

    def test_a_tree_document_that_never_recorded_its_width_still_loads(
        self, tmp_path: Path,
    ) -> None:
        """``n_features: 0`` means 'unknown', not 'zero columns'.

        The ensemble's own guard treats it that way, and the load-time
        check must not turn an older tree block into a refusal.
        """
        model = _document(pp.PROTOCOL_RANDOM_POINT)
        stripped = dataclasses.replace(
            model,
            models={
                lead: dataclasses.replace(
                    entry,
                    trees=dataclasses.replace(entry.trees, n_features=0),
                )
                for lead, entry in model.models.items()
            },
        )
        table = PostprocessTable(_write(tmp_path / "m.json", stripped))
        with structlog.testing.capture_logs() as logs:
            assert table.load() is not None
        assert table.active
        assert [e["event"] for e in logs] == ["push_postprocess_loaded"]
