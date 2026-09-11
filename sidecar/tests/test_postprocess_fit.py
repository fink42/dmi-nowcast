"""H-P — the nightly refit, and the thresholds fitted on top of it.

Two claims are under test, and both are about consistency rather than
about skill (the skill is in ``archive/l3_and_postprocess_20260911/``):

1. **The model the service reads is fitted on the rows the service
   wrote.** Same loader, same gauge outcome, same dead-gauge rule as the
   offline study — and rows that carry no features are skipped and
   counted rather than imputed into a prediction that looks like a
   forecast and is actually the base rate.
2. **A threshold is a percent ON a probability.** Once the engine decides
   on ``p_post``, the nightly sweep has to be fitted on ``p_post`` too,
   filling it from the freshly fitted model wherever a row has features
   but no stored value — and the thresholds document has to say which
   probability that was.

Synthetic and offline: a two-day decision tree and a hand-planted gauge
store, the same shape ``test_sweep_thresholds.py`` uses.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from dmi_nowcast_core import postprocess as pp
from dmi_nowcast_core.metobs import Observation
from dmi_nowcast_core.station_store import StationObsStore
from dmi_nowcast_core.warning_score import decision_table
from dmi_nowcast_sidecar import quality_job
from dmi_nowcast_sidecar.config import Config
from dmi_nowcast_sidecar.postprocess_fit import (
    PostprocessFitOptions,
    ProbabilityFiller,
    featured_mask,
    options_from_json,
    options_to_json,
    run_postprocess_fit,
)
from dmi_nowcast_sidecar.threshold_sweep import SweepError

STATIONS = ("06180", "06120")
LEADS = (10, 30)
DAYS = (1, 2, 3, 4)
FIRST_FRAME_MIN = 6 * 60
LAST_FRAME_MIN = 11 * 60
FIRST_SLOT_MIN = 5 * 60
LAST_SLOT_MIN = 12 * 60


def _at(day: int, minute_of_day: int) -> datetime:
    return datetime(2026, 6, day, tzinfo=timezone.utc) + timedelta(
        minutes=minute_of_day,
    )


def _wet(station: str, day: int, stamp: datetime) -> bool:
    """Rain at station A on the even days, from 08:00 to 08:30."""
    if station != STATIONS[0] or day % 2:
        return False
    return 8 * 60 <= stamp.hour * 60 + stamp.minute <= 8 * 60 + 30


def _decision_rows(day: int, *, features: bool = True) -> list[dict]:
    """One frame per station per ten minutes, with a usable signal.

    ``raw_frac_<lead>`` and the upstream corridor carry the signal the
    model is supposed to learn; everything else is constant, so a fitted
    coefficient is attributable.
    """
    rows: list[dict] = []
    for minute in range(FIRST_FRAME_MIN, LAST_FRAME_MIN + 1, 10):
        radar_ts = _at(day, minute)
        for station in STATIONS:
            coming = _wet(station, day, radar_ts + timedelta(minutes=30))
            row: dict = {
                "radar_ts": radar_ts,
                "generated_at": radar_ts,
                "station_id": station,
                "p_rain": 0.5,
                "action": "none",
                "p_rain_10": 0.55 if coming else 0.45,
                "p_rain_30": 0.55 if coming else 0.45,
                "eta_min": 25.0,
                "intensity_mm_h": 1.2,
                "observed_mm_h": 0.0,
                "forecast_now_mm_h": 0.0,
                "armed_after": True,
                "streak_after": 0,
            }
            if features:
                row.update({
                    "raw_frac_10": 0.9 if coming else 0.05,
                    "raw_frac_30": 0.95 if coming else 0.05,
                    "obs_max_5km_mm_h": 0.0,
                    "up_max_20km_mm_h": 8.0 if coming else 0.0,
                    "up_max_40km_mm_h": 8.0 if coming else 0.0,
                    "up_dist_km": 6.0 if coming else None,
                    "up_wet_frac_40km": 0.4 if coming else 0.0,
                    "bulk_kmh": 45.0,
                    "bulk_dir_deg": 90.0,
                    "local_speed_kmh": 44.0,
                    "stalled_share": 0.01,
                    "season": "summer",
                    "hour_utc": radar_ts.hour,
                    "frame_age_min": 14.0,
                    "station_radar_km": 40.0,
                })
            rows.append(row)
    return rows


def _write_decisions(
    directory: Path, day: int, *, features: bool = True,
) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = _decision_rows(day, features=features)
    table = decision_table(rows, LEADS)
    if features:
        for field in pp.feature_schema(LEADS):
            table = table.append_column(field, pa.array(
                [row.get(field.name) for row in rows], type=field.type,
            ))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"2026-06-{day:02d}.parquet"
    pq.write_table(table, path)
    return path


def _write_gauge(corpus_dir: Path) -> None:
    store = StationObsStore(corpus_dir)
    observations: list[Observation] = []
    for day in DAYS:
        for station in STATIONS:
            for minute in range(FIRST_SLOT_MIN, LAST_SLOT_MIN + 1, 10):
                stamp = _at(day, minute)
                observations.append(Observation(
                    station_id=station,
                    observed_utc=stamp,
                    parameter_id="precip_past10min",
                    value=0.5 if _wet(station, day, stamp) else 0.0,
                ))
    store.append(observations)


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """A gauge store plus a featured ``decisions/`` tree."""
    _write_gauge(tmp_path / "corpus")
    for day in DAYS:
        _write_decisions(tmp_path / "decisions", day)
    return tmp_path


def _options(root: Path, **over) -> PostprocessFitOptions:
    kwargs = {
        "decisions_dirs": [root / "decisions"],
        "corpus_dir": root / "corpus",
        "out": root / "postprocess.json",
        "leads": LEADS,
        "design_leads": LEADS,
        "min_rows": 1,
        "min_known_slots": 0,
    }
    kwargs.update(over)
    return PostprocessFitOptions(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


class TestTheNightlyFit:
    def test_it_fits_every_lead_and_records_what_it_stood_on(
        self, corpus: Path,
    ) -> None:
        result = run_postprocess_fit(_options(corpus))
        model = result["model"]
        assert sorted(model.models) == list(LEADS)
        assert model.design_leads == LEADS
        assert model.feature_names == pp.design_columns(LEADS)

        summary = result["summary"]
        assert summary["rows"] == summary["rows_on_disk"]
        assert summary["rows_without_features"] == 0
        assert summary["stations"] == 2
        assert summary["days"] == len(DAYS)
        # In-sample by construction; the artefact says so rather than
        # leaving a reader to assume the base rates are scores.
        assert summary["held_out"] is False
        assert model.training["held_out"] is False
        assert model.training["corpus_dir"] == str(corpus / "corpus")

    def test_it_learned_the_signal_the_curve_cannot_see(
        self, corpus: Path,
    ) -> None:
        """The served p_rain barely separates; the upstream echo does.

        Not a skill claim — the fixture is noiseless — but it does pin
        that the design reaches the feature columns at all, which a wiring
        mistake would silently break.
        """
        model = run_postprocess_fit(_options(corpus))["model"]
        wet = {
            "raw_frac_10": [0.9], "raw_frac_30": [0.95],
            "up_max_20km_mm_h": [8.0], "up_dist_km": [6.0],
            "up_wet_frac_40km": [0.4], "season": np.array(["summer"]),
            "hour_utc": [8.0],
        }
        dry = {
            "raw_frac_10": [0.05], "raw_frac_30": [0.05],
            "up_max_20km_mm_h": [0.0], "up_dist_km": [np.nan],
            "up_wet_frac_40km": [0.0], "season": np.array(["summer"]),
            "hour_utc": [8.0],
        }
        assert model.predict(wet, 30)[0] > model.predict(dry, 30)[0]

    def test_rows_without_features_are_skipped_and_counted(
        self, corpus: Path,
    ) -> None:
        """A pre-H-P replay day beside the featured ones."""
        _write_decisions(corpus / "decisions", 5, features=False)
        _write_gauge(corpus / "corpus")
        summary = run_postprocess_fit(_options(corpus))["summary"]
        assert summary["rows_without_features"] > 0
        assert summary["rows"] == (
            summary["rows_on_disk"] - summary["rows_without_features"]
        )

    def test_nothing_to_fit_on_is_a_sweep_error_not_a_crash(
        self, tmp_path: Path,
    ) -> None:
        _write_gauge(tmp_path / "corpus")
        for day in DAYS:
            _write_decisions(tmp_path / "decisions", day, features=False)
        with pytest.raises(SweepError, match="post-processing features"):
            run_postprocess_fit(_options(tmp_path))

    def test_a_fit_below_the_row_floor_is_refused(self, corpus: Path) -> None:
        with pytest.raises(SweepError, match="row floor"):
            run_postprocess_fit(_options(corpus, min_rows=10_000_000))

    def test_the_options_survive_the_process_boundary(self, corpus: Path) -> None:
        options = _options(corpus)
        assert options_from_json(options_to_json(options)) == options
        assert json.loads(json.dumps(options_to_json(options)))

    def test_featured_mask_ignores_the_shared_columns(self) -> None:
        """``observed_mm_h`` has always been there; it proves nothing."""
        rows = {
            "t": np.zeros(3, dtype=np.int64),
            "extra": {
                "observed_mm_h": np.array([1.0, 2.0, 3.0]),
                "up_max_20km_mm_h": np.array([np.nan, 5.0, np.nan]),
            },
        }
        assert featured_mask(rows, LEADS).tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# Filling the served probability for the sweep
# ---------------------------------------------------------------------------


class TestProbabilityFiller:
    def _table(self, corpus: Path, *, features: bool = True):
        import pyarrow.parquet as pq

        return pq.read_table(_write_decisions(
            corpus / "other", 1, features=features,
        ))

    def test_it_computes_from_features_where_nothing_is_stored(
        self, corpus: Path,
    ) -> None:
        model = run_postprocess_fit(_options(corpus))["model"]
        filler = ProbabilityFiller(model, LEADS, LEADS, pp.post_column)
        out = filler(self._table(corpus))
        assert out.num_rows == self._table(corpus).num_rows
        assert filler.counts["stored"] == 0
        assert filler.counts["computed"] == out.num_rows * len(LEADS)
        assert filler.counts["dropped"] == 0
        values = out.column("p_post_30").to_pylist()
        assert all(v is not None and 0.0 <= v <= 1.0 for v in values)

    def test_a_stored_value_wins_over_the_model(self, corpus: Path) -> None:
        """It is what the engine actually decided on."""
        import pyarrow as pa

        model = run_postprocess_fit(_options(corpus))["model"]
        table = self._table(corpus)
        table = table.append_column(
            "p_post_30", pa.array([0.123] * table.num_rows, pa.float32()),
        )
        filler = ProbabilityFiller(model, LEADS, LEADS, pp.post_column)
        out = filler(table)
        assert out.column("p_post_30").to_pylist() == [
            pytest.approx(0.123, rel=1e-5),
        ] * table.num_rows
        assert filler.counts["stored"] == table.num_rows

    def test_rows_with_neither_are_dropped_and_counted(
        self, corpus: Path,
    ) -> None:
        """Scoring them would call every frame below threshold — which
        measures the archive's depth, not the rule."""
        model = run_postprocess_fit(_options(corpus))["model"]
        table = self._table(corpus, features=False)
        filler = ProbabilityFiller(model, LEADS, LEADS, pp.post_column)
        out = filler(table)
        assert out.num_rows == 0
        assert filler.counts["dropped"] == table.num_rows

    def test_without_a_model_only_stored_rows_survive(
        self, corpus: Path,
    ) -> None:
        filler = ProbabilityFiller(None, LEADS, LEADS, pp.post_column)
        out = filler(self._table(corpus))
        # Featured rows are kept — the model simply had nothing to add.
        assert out.num_rows == self._table(corpus).num_rows
        assert filler.counts["computed"] == 0
        assert all(v is None for v in out.column("p_post_30").to_pylist())


# ---------------------------------------------------------------------------
# The sweep, fitted on the served probability
# ---------------------------------------------------------------------------


class TestTheSweepFollowsTheEngine:
    def _sweep(self, corpus: Path, **over):
        from dmi_nowcast_sidecar.threshold_sweep import SweepOptions, run_fit

        kwargs = {
            "decisions_dirs": [corpus / "decisions"],
            "corpus_dir": corpus / "corpus",
            "leads": LEADS,
            "thresholds": (40, 50, 60),
            "min_warnings": 1,
            "min_known_slots": 0,
        }
        kwargs.update(over)
        return run_fit(SweepOptions(**kwargs))  # type: ignore[arg-type]

    def test_the_default_is_still_the_served_probability(
        self, corpus: Path,
    ) -> None:
        payload = self._sweep(corpus)
        assert payload["settings"]["probability_column"] == "p_rain_{lead}"
        assert payload["settings"]["probability_rows"] is None
        assert payload["thresholds"]["objective"]["probability_column"] == (
            "p_rain_{lead}"
        )

    def test_it_replays_the_rule_on_the_post_processed_column(
        self, corpus: Path,
    ) -> None:
        model_path = corpus / "postprocess.json"
        model = run_postprocess_fit(_options(corpus))["model"]
        model_path.write_text(model.dumps())

        payload = self._sweep(
            corpus,
            probability_column=pp.POST_COLUMN_TEMPLATE,
            postprocess_model=model_path,
            design_leads=LEADS,
        )
        settings = payload["settings"]
        assert settings["probability_column"] == "p_post_{lead}"
        # Every row had features and none had a stored value, so the
        # model spoke for all of them and nothing was excluded.
        assert settings["probability_rows"]["computed"] > 0
        assert settings["probability_rows"]["dropped"] == 0
        assert payload["thresholds"]["objective"]["probability_column"] == (
            "p_post_{lead}"
        )
        # The document still validates: the new key is optional, so an
        # older table keeps loading and this one is not a schema break.
        from dmi_nowcast_core.push_thresholds import validate_thresholds

        assert validate_thresholds(payload["thresholds"]) == []

    def test_rows_without_features_are_excluded_and_counted(
        self, corpus: Path,
    ) -> None:
        _write_decisions(corpus / "decisions", 5, features=False)
        model_path = corpus / "postprocess.json"
        model_path.write_text(run_postprocess_fit(
            _options(corpus),
        )["model"].dumps())
        payload = self._sweep(
            corpus,
            probability_column=pp.POST_COLUMN_TEMPLATE,
            postprocess_model=model_path,
            design_leads=LEADS,
        )
        assert payload["settings"]["probability_rows"]["dropped"] > 0

    def test_a_template_that_does_not_vary_is_refused(
        self, corpus: Path,
    ) -> None:
        with pytest.raises(SweepError, match="does not vary"):
            self._sweep(corpus, probability_column="p_post")


# ---------------------------------------------------------------------------
# The nightly job
# ---------------------------------------------------------------------------


class TestTheNightlyJob:
    def _job_config(self, corpus: Path, tmp_path: Path) -> dict:
        return {
            "quality": {
                "out_json": str(tmp_path / "quality.json"),
                "markdown_dir": None,
                "inputs": {},
            },
            "fit": {"enabled": False},
            "fit_postprocess": {
                "enabled": True,
                "out": str(tmp_path / "postprocess.json"),
                "options": options_to_json(_options(corpus)),
            },
        }

    def test_it_refits_and_publishes_atomically(
        self, corpus: Path, tmp_path: Path,
    ) -> None:
        config = self._job_config(corpus, tmp_path)
        summary = quality_job.run_job(
            config,
            builder=lambda inputs, **kw: {"built_at_utc": "2026-06-05"},
            renderer=lambda report: "# quality\n",
        )
        out = tmp_path / "postprocess.json"
        assert summary["postprocess_path"] == str(out)
        assert summary["postprocess"]["rows"] > 0
        assert summary["postprocess_error"] is None
        assert summary["postprocess_fitted_at"]
        # And what landed is a model the serving path can read.
        from dmi_nowcast_sidecar.push.postprocess import PostprocessTable

        table = PostprocessTable(out)
        table.load()
        assert table.active is True
        assert table.leads == list(LEADS)

    def test_a_failed_refit_leaves_the_model_in_service(
        self, corpus: Path, tmp_path: Path,
    ) -> None:
        out = tmp_path / "postprocess.json"
        out.write_text("the model already in service\n")
        config = self._job_config(corpus, tmp_path)

        def boom(_options):
            raise RuntimeError("the corpus volume is gone")

        summary = quality_job.run_job(
            config,
            builder=lambda inputs, **kw: {},
            renderer=lambda report: "",
            postprocess_fitter=boom,
        )
        assert "the corpus volume is gone" in summary["postprocess_error"]
        assert summary["postprocess_path"] is None
        assert out.read_text() == "the model already in service\n"

    def test_the_live_refresh_never_refits(
        self, corpus: Path, tmp_path: Path,
    ) -> None:
        """The model is a once-a-day decision, like the thresholds."""
        (tmp_path / "quality.json").write_text(json.dumps({"a": 1}))
        summary = quality_job.run_job(
            self._job_config(corpus, tmp_path),
            live_only=True,
            builder=lambda inputs, **kw: {},
            renderer=lambda report: "",
        )
        assert summary["mode"] == "live"
        assert summary["postprocess_path"] is None
        assert not (tmp_path / "postprocess.json").exists()

    def test_the_model_is_fitted_before_the_thresholds_are(
        self, corpus: Path, tmp_path: Path,
    ) -> None:
        """Order is the contract: a threshold is a percent ON a probability.

        A sweep fitted before the refit would be a threshold on last
        night's scale, applied to tonight's model.
        """
        order: list[str] = []
        config = self._job_config(corpus, tmp_path)
        config["fit"] = {
            "enabled": True,
            "thresholds_out": str(tmp_path / "push_thresholds.json"),
            "options": {
                "decisions_dirs": [str(corpus / "decisions")],
                "corpus_dir": str(corpus / "corpus"),
            },
        }
        quality_job.run_job(
            config,
            builder=lambda inputs, **kw: {},
            renderer=lambda report: "",
            postprocess_fitter=lambda o: (
                order.append("postprocess")
                or {"model": _stub_model(), "summary": {}}
            ),
            fitter=lambda o: (
                order.append("thresholds")
                or {"thresholds": _stub_thresholds()}
            ),
        )
        assert order == ["postprocess", "thresholds"]


def _stub_model():
    class _M:
        fitted_at_utc = "2026-06-05T03:00:00+00:00"

        def dumps(self) -> str:
            return "{}\n"

    return _M()


def _stub_thresholds() -> dict:
    return {
        "schema_version": 1,
        "fitted_at_utc": "2026-06-05T03:00:00+00:00",
        "objective": {},
        "window": {},
        "fallback_threshold_pct": 40,
        "leads": {},
    }


# ---------------------------------------------------------------------------
# Wiring: what the task hands the child
# ---------------------------------------------------------------------------


class TestTheTaskWiring:
    def _config(self, tmp_path: Path) -> Config:
        return Config(
            home={"lat": 55.33, "lon": 10.32},  # type: ignore[arg-type]
            calibration={  # type: ignore[arg-type]
                "curves_path": tmp_path / "curves.json",
                "national_curves_path": tmp_path / "national_curves.json",
            },
            storage={  # type: ignore[arg-type]
                "data_dir": tmp_path / "data",
                "corpus_dir": tmp_path / "corpus",
            },
            lightning={"archive_dir": tmp_path / "strikes"},  # type: ignore[arg-type]
            quality_report={  # type: ignore[arg-type]
                "enabled": True,
                "fit_thresholds": {
                    "enabled": True,
                    "decisions_dirs": [str(tmp_path / "decisions")],
                },
            },
        )

    def test_the_refit_rides_on_the_threshold_fits_rows_by_default(
        self, tmp_path: Path,
    ) -> None:
        from dmi_nowcast_sidecar.quality_report import QualityReportTask

        task = QualityReportTask(self._config(tmp_path))
        payload = task.job_config()
        assert payload["fit_postprocess"]["enabled"] is True
        assert payload["fit_postprocess"]["options"]["decisions_dirs"] == [
            str(tmp_path / "decisions"),
        ]
        # And it lands where the cycle reads it.
        assert payload["fit_postprocess"]["out"] == str(
            tmp_path / "data" / "postprocess.json",
        )
        assert json.loads(json.dumps(payload))

    def test_the_sweep_follows_push_probability_source(
        self, tmp_path: Path,
    ) -> None:
        config = self._config(tmp_path)
        from dmi_nowcast_sidecar.quality_report import QualityReportTask

        task = QualityReportTask(config)
        options = task._fit_options()
        assert options.probability_column == "p_post_{lead}"
        assert options.postprocess_model == tmp_path / "data" / "postprocess.json"

        config.push.probability_source = "curve"
        options = QualityReportTask(config)._fit_options()
        assert options.probability_column is None
        assert options.postprocess_model is None

    def test_an_empty_decisions_list_switches_the_refit_off(
        self, tmp_path: Path,
    ) -> None:
        config = self._config(tmp_path)
        config.quality_report.fit_thresholds.decisions_dirs = []
        from dmi_nowcast_sidecar.quality_report import QualityReportTask

        payload = QualityReportTask(config).job_config()
        assert payload["fit_postprocess"]["enabled"] is False
