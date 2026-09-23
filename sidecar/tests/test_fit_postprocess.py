"""The H-P post-processing fit, end to end (``scripts/fit_postprocess.py``).

Tested from the sidecar suite for the reason ``test_benchmark_report`` is:
the script leans on the core library for the model AND on the sidecar's
threshold sweep for the row reader and the Layer C replay, and this
environment is the only one that has both.

Fully offline and fully synthetic — no radar, no STEPS, no network. The
fixture is three single-week months over two stations, planted so the
answer is arithmetic rather than luck:

* every (day, station) has one hour of rain at a known instant;
* every decision row carries ``m``, the minutes from that row's instant to
  the first wet gauge slot, expressed the way the cycle would see it: as
  ``up_dist_km`` under a storm travelling one kilometre a minute, and as
  ``up_max_40km_mm_h`` when there is any echo in the corridor at all;
* the ensemble fraction is **distance-blind and saturating** — 1.0 for
  every row with echo upwind, whether the rain is five minutes away or
  forty — which is exactly the defect this work package exists for.

So the curve baseline cannot beat a coin toss among the rows with echo,
and the post-processor should separate them almost perfectly. A test that
merely showed "some improvement" would pass on a bug; this one has a right
answer.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
numpy = pytest.importorskip("numpy")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import benchmark_report as bench  # noqa: E402  (after the sys.path edit)
import fit_postprocess as fit  # noqa: E402
import replay_warnings as rw  # noqa: E402

from dmi_nowcast_core import postprocess as pp  # noqa: E402
from dmi_nowcast_core.metobs import Observation  # noqa: E402
from dmi_nowcast_core.station_store import StationObsStore  # noqa: E402
from dmi_nowcast_sidecar.threshold_sweep import SEASON_MONTHS  # noqa: E402

STATION_A = "06180"
STATION_B = "06120"
STATIONS = (STATION_A, STATION_B)
LEADS = (20, 30)

#: Three months, one week each: winter, shoulder, summer — so the LOMO has
#: three folds and every stratum has a month of its own.
MONTHS: tuple[tuple[int, int], ...] = ((2026, 1), (2026, 4), (2026, 6))
DAYS_PER_MONTH = 7

#: Decision instants 06:00–16:00 every 10 min, and gauge slots 04:00–20:00,
#: so every outcome window closes well inside the reported record.
FIRST_FRAME_MIN, LAST_FRAME_MIN = 6 * 60, 16 * 60
FIRST_SLOT_MIN, LAST_SLOT_MIN = 4 * 60, 20 * 60

#: One hour of rain per (day, station). The first WET SLOT END is here;
#: the two stations rain at different times, and the time walks through the
#: week so no single clock reading gives the answer away.
EVENT_SLOTS = 6


def _days() -> list[tuple[int, int, int]]:
    return [
        (year, month, 1 + offset)
        for year, month in MONTHS
        for offset in range(DAYS_PER_MONTH)
    ]


def _at(day: tuple[int, int, int], minute: int) -> datetime:
    return datetime(*day, tzinfo=timezone.utc) + timedelta(minutes=minute)


def _first_wet_slot_min(day: tuple[int, int, int], station: str) -> int:
    """Minute of day at which the first wet gauge slot ENDS."""
    base = 9 * 60 if station == STATION_A else 13 * 60
    return base + 30 * (day[2] % 5)


def _minutes_to_onset(day: tuple[int, int, int], station: str, minute: int) -> float:
    return float(_first_wet_slot_min(day, station) - minute)


#: The fixture's storm speed.
STORM_KMH = 48.0
KM_PER_MIN = STORM_KMH / 60.0


def _feature_row(m: float) -> dict:
    """The cycle's view of a storm ``m`` minutes away at ``STORM_KMH``.

    ``m < 0`` means it is already raining at the station, which is what the
    zero upwind distance says; beyond 40 km the corridor is empty and the
    distance is NaN, written as a null — "no echo upwind", never a large
    number.

    One deliberate simplification: the near and far corridor maxima are the
    SAME here. A real ``up_max_20km_mm_h`` carries some distance
    information of its own — it drops to zero once the rain is more than
    20 km out — and on a fixture whose truth is a step function of ``m``
    that would make it a near-perfect classifier by itself, leaving
    ``up_dist_km`` nothing to explain and the coefficient assertion below
    meaningless. Here every column except the distance says only "there is
    rain upstream".
    """
    distance = max(0.0, m * KM_PER_MIN)
    has_echo = -60.0 <= m and distance <= 40.0
    return {
        "up_dist_km": (None if distance > 40.0 else min(distance, 40.0)),
        "up_max_40km_mm_h": 5.0 if has_echo else 0.0,
        "up_max_20km_mm_h": 5.0 if has_echo else 0.0,
        "up_wet_frac_40km": 0.4 if has_echo else 0.0,
        "obs_max_5km_mm_h": 5.0 if m <= 0.0 else 0.0,
        # Distance-blind and saturating: the defect, in one column.
        **{pp.raw_fraction_column(lead): (1.0 if has_echo else 0.05)
           for lead in (10, 20, 30, 45, 60)},
        "bulk_kmh": STORM_KMH,
        "bulk_dir_deg": 270.0,
        "local_speed_kmh": STORM_KMH,
        "stalled_share": 0.01,
        "frame_age_min": 0.0,
    }


def _decision_rows(day: tuple[int, int, int]) -> list[dict]:
    rows: list[dict] = []
    for minute in range(FIRST_FRAME_MIN, LAST_FRAME_MIN + 1, 10):
        radar_ts = _at(day, minute)
        for station in STATIONS:
            m = _minutes_to_onset(day, station, minute)
            features = _feature_row(m)
            has_echo = features["up_max_40km_mm_h"] > 0
            rows.append({
                "radar_ts": radar_ts,
                # Zero frame age: the decision instant IS the frame instant,
                # which makes the outcome-window arithmetic checkable.
                "generated_at": radar_ts,
                "station_id": station,
                "p_rain": 0.99,             # a trap, as in the sweep fixture
                "action": "none",
                # The served curve on a saturating fraction: one number for
                # every row with echo, whatever the distance.
                **{f"p_rain_{lead}": (0.6 if has_echo else 0.02)
                   for lead in LEADS},
                # The ensemble ETA is coarse in the same way the fraction
                # is: it says "soon" for everything with echo upwind. If it
                # carried the distance the model would have two copies of
                # the answer and the coefficient test below would mean
                # nothing.
                "eta_min": 15.0 if has_echo else None,
                "intensity_mm_h": 2.0 if has_echo else None,
                "observed_mm_h": 5.0 if m <= 0.0 else 0.0,
                "forecast_now_mm_h": 0.0,
                "armed_after": True,
                "streak_after": 0,
                "season": pp.season_of_month(day[1]),
                "hour_utc": radar_ts.hour,
                "station_radar_km": 40.0 if station == STATION_A else 25.0,
                **features,
            })
    return rows


def _write_run(directory: Path) -> Path:
    for day in _days():
        rw.write_decisions(
            directory / "decisions" / f"{day[0]:04d}-{day[1]:02d}-{day[2]:02d}.parquet",
            _decision_rows(day), LEADS, features=True,
        )
    (directory / "summary.json").write_text(json.dumps({
        "run": {
            "n_days": len(_days()), "n_stations": len(STATIONS),
            "frame_age_min": 0,
            "steps": {
                "ensemble_size": 16, "n_cascade_levels": 6,
                "downsample_factor": 4, "horizon_min": 90,
                "leads_min": list(LEADS), "threshold_mm_h": 0.5,
            },
            "flow": {"completion": "confidence"},
            "rules": {"lead_min": 30, "threshold_pct": 40},
            "national_curves": "/x/national_curves.json",
            "features": {"enabled": True},
        },
    }))
    return directory


def _write_gauge(corpus_dir: Path) -> None:
    store = StationObsStore(corpus_dir)
    observations: list[Observation] = []
    for day in _days():
        for station in STATIONS:
            first = _first_wet_slot_min(day, station)
            wet = {first + 10 * k for k in range(EVENT_SLOTS)}
            for minute in range(FIRST_SLOT_MIN, LAST_SLOT_MIN + 1, 10):
                observations.append(Observation(
                    station_id=station,
                    observed_utc=_at(day, minute),
                    parameter_id="precip_past10min",
                    value=0.5 if minute in wet else 0.0,
                ))
    store.append(observations)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("postprocess")
    _write_gauge(root / "corpus")
    _write_run(root / "replay")
    return root


def _run(root: Path, out: Path, *extra: str) -> dict:
    argv = [
        "--run", str(root / "replay"),
        "--corpus-dir", str(root / "corpus"),
        "--out-dir", str(out),
        "--leads", "20,30",
        "--design-leads", "10,20,30,45,60",
        "--resamples", "50",
        *extra,
    ]
    assert fit.main(argv) == 0
    return json.loads((out / "postprocess_report.json").read_text())


@pytest.fixture(scope="module")
def fitted(corpus: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple:
    out = tmp_path_factory.mktemp("fit-out")
    back = tmp_path_factory.mktemp("fit-back") / "replay_postprocess"
    report = _run(corpus, out, "--write-back", str(back))
    return report, out, back, corpus


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


class TestTheFit:
    def test_it_writes_the_artefact_and_both_report_files(self, fitted) -> None:
        _report, out, _back, _corpus = fitted
        assert (out / "postprocess.json").is_file()
        assert (out / "postprocess_report.md").is_file()
        assert (out / "postprocess_report.json").is_file()

    def test_the_artefact_round_trips_into_a_usable_model(self, fitted) -> None:
        _report, out, _back, _corpus = fitted
        model = pp.PostprocessModel.loads((out / "postprocess.json").read_text())
        assert model.leads == LEADS
        assert model.design_leads == (10, 20, 30, 45, 60)
        assert len(model.standardiser.mean) == len(model.feature_names)
        assert set(model.models) == set(LEADS)
        assert model.training["stations"] == len(STATIONS)
        assert model.training["months"] == ["2026-01", "2026-04", "2026-06"]

    def test_it_beats_the_saturating_curve_out_of_fold(self, fitted) -> None:
        report, _out, _back, _corpus = fitted
        for lead in LEADS:
            pooled = report["evaluation"]["leads"][str(lead)]["all"]
            assert pooled["postprocess"]["bss"] > pooled["baseline"]["bss"] + 0.1
            assert pooled["postprocess"]["pr_auc"] > pooled["baseline"]["pr_auc"]
            # Both forecasts are scored on exactly the same rows.
            assert pooled["baseline"]["n"] == pooled["postprocess"]["n"]

    def test_the_paired_interval_is_reported_and_excludes_zero(
        self, fitted,
    ) -> None:
        report, _out, _back, _corpus = fitted
        difference = report["evaluation"]["leads"]["30"]["all"]["difference"]
        assert difference["bss"][0] > 0
        assert difference["bss_excludes_zero"] is True
        assert difference["days"] == len(_days())

    def test_every_month_is_a_fold_and_every_season_a_stratum(
        self, fitted,
    ) -> None:
        report, _out, _back, _corpus = fitted
        assert [f["fold"] for f in report["evaluation"]["folds"]] == [
            "2026-01", "2026-04", "2026-06",
        ]
        strata = report["evaluation"]["leads"]["30"]
        for season in pp.SEASONS:
            assert strata[season] is not None, season

    def test_the_distance_carries_the_weight(self, fitted) -> None:
        """The predictor the baseline lacks is the one the fit leans on.

        Nothing else in the fixture encodes how far away the rain is — the
        fraction saturates and the ETA says "soon" for every row with echo
        — so a large negative weight here is the model finding the one
        thing that separates the rows the curve pools.
        """
        _report, out, _back, _corpus = fitted
        model = pp.PostprocessModel.loads((out / "postprocess.json").read_text())
        names = list(model.feature_names)
        for lead in LEADS:
            coefficients = model.models[lead].coefficients
            distance = coefficients[names.index("log1p_up_dist_km")]
            assert distance < -0.5, lead

    def test_the_markdown_carries_the_tables_a_reader_needs(
        self, fitted,
    ) -> None:
        _report, out, _back, _corpus = fitted
        text = (out / "postprocess_report.md").read_text()
        assert "## Out-of-fold skill" in text
        assert "## Reliability, pooled" in text
        assert "## Standardised coefficients" in text
        assert "## Folds" in text
        assert "## Feature definitions" in text
        assert "`log1p_up_dist_km`" in text
        # Every stratum named, and the shoulder caveat said out loud.
        for stratum in ("all",) + pp.SEASONS:
            assert f"| {stratum} |" in text
        assert "Shoulder is April alone" in text

    def test_the_headline_json_is_valid(self, corpus: Path, tmp_path: Path) -> None:
        report = _run(corpus, tmp_path / "again")
        assert report["rows"] == len(_days()) * len(STATIONS) * (
            (LAST_FRAME_MIN - FIRST_FRAME_MIN) // 10 + 1
        )
        assert report["stations"] == len(STATIONS)
        assert report["n_months"] == len(MONTHS)
        assert report["run_settings"]["available"] is True
        assert report["run_settings"]["ensemble_size"] == 16


# ---------------------------------------------------------------------------
# Write-back and re-scoring
# ---------------------------------------------------------------------------


class TestWriteBack:
    def test_the_copy_keeps_the_original_columns_and_adds_p_post(
        self, fitted,
    ) -> None:
        import pyarrow.parquet as pq

        _report, _out, back, corpus = fitted
        source = sorted((corpus / "replay" / "decisions").glob("*.parquet"))
        copied = sorted((back / "decisions").glob("*.parquet"))
        assert [p.name for p in copied] == [p.name for p in source]
        original = pq.read_table(source[0])
        after = pq.read_table(copied[0])
        assert set(original.schema.names) < set(after.schema.names)
        assert {"p_post_20", "p_post_30"} <= set(after.schema.names)
        assert after.num_rows == original.num_rows
        values = after.column("p_post_30").to_pylist()
        assert all(v is not None and 0.0 <= v <= 1.0 for v in values)

    @pytest.mark.parametrize("where", ["replay", "replay/decisions", ""])
    def test_it_never_writes_into_the_run(
        self, corpus: Path, tmp_path: Path, where: str,
    ) -> None:
        """The run itself, a directory inside it, and its parent are refused."""
        target = corpus / where if where else corpus
        assert fit.main([
            "--run", str(corpus / "replay"),
            "--corpus-dir", str(corpus / "corpus"),
            "--out-dir", str(tmp_path / "out"),
            "--leads", "30", "--resamples", "0",
            "--write-back", str(target),
        ]) == 2

    def test_the_summary_records_where_p_post_came_from(self, fitted) -> None:
        _report, out, back, _corpus = fitted
        summary = json.loads((back / "summary.json").read_text())
        # The original settings survive, so the benchmark's parity check
        # still has something to check.
        assert summary["run"]["steps"]["ensemble_size"] == 16
        block = summary["postprocess"]
        assert block["column_template"] == "p_post_{lead}"
        assert block["kind"].startswith("out-of-fold")
        assert block["model"].endswith("postprocess.json")
        assert block["files"] == len(_days())

    def test_the_benchmark_scores_the_copy_through_the_same_report(
        self, fitted, tmp_path: Path,
    ) -> None:
        _report, _out, back, corpus = fitted
        common = [
            "--corpus-dir", str(corpus / "corpus"),
            "--leads", "20,30", "--layers", "b", "--resamples", "20",
        ]
        base_out = tmp_path / "layer-b-base"
        assert bench.main(
            ["--baseline", str(corpus / "replay"), "--out-dir", str(base_out)]
            + common
        ) == 0
        post_out = tmp_path / "layer-b-post"
        assert bench.main(
            ["--baseline", str(back), "--out-dir", str(post_out),
             "--probability-column", "p_post_{lead}"] + common
        ) == 0
        baseline = json.loads((base_out / "benchmark.json").read_text())
        posted = json.loads((post_out / "benchmark.json").read_text())
        assert posted["settings"]["probability_column"] == "p_post_{lead}"
        for lead in ("20", "30"):
            a = baseline["layer_b"]["leads"][lead]["all"]["baseline"]
            b = posted["layer_b"]["leads"][lead]["all"]["baseline"]
            # Same rows, same outcome, a better probability.
            assert a["n"] == b["n"]
            assert b["bss"] > a["bss"]

    def test_layer_c_replays_the_rule_on_the_new_column(
        self, fitted, tmp_path: Path,
    ) -> None:
        _report, _out, back, corpus = fitted
        out = tmp_path / "layer-c-post"
        assert bench.main([
            "--baseline", str(back),
            "--corpus-dir", str(corpus / "corpus"),
            "--out-dir", str(out),
            "--leads", "30", "--layers", "c",
            "--thresholds", "30:70:20", "--min-warnings", "1",
            "--resamples", "0",
            "--probability-column", "p_post_{lead}",
        ]) == 0
        report = json.loads((out / "benchmark.json").read_text())
        entry = report["layer_c"]["leads"]["30"]
        # The sweep saw real probabilities, not a column of nulls.
        assert entry["out_of_fold"]["warnings"] > 0


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def test_the_column_template_must_name_a_lead() -> None:
    assert bench.column_template("p_post_{lead}")(30) == "p_post_30"
    assert bench.column_template("p_rain_{lead}")(45) == "p_rain_45"
    with pytest.raises(ValueError, match=r"\{lead\}"):
        bench.column_template("p_post")


def test_a_bad_template_is_a_usage_error_not_a_traceback(
    corpus: Path, tmp_path: Path,
) -> None:
    assert bench.main([
        "--baseline", str(corpus / "replay"),
        "--corpus-dir", str(corpus / "corpus"),
        "--out-dir", str(tmp_path / "nope"),
        "--probability-column", "nonsense",
    ]) == 2


def test_the_core_season_split_matches_the_sweeps() -> None:
    """One seasonal split project-wide, restated in core to keep it clean."""
    assert pp.SEASON_MONTHS == SEASON_MONTHS


def test_a_run_without_features_is_refused_with_a_hint(tmp_path: Path) -> None:
    """Fitting on a feature-less run must say what is missing, not crash."""
    run = tmp_path / "bare"
    for day in _days()[:2]:
        rw.write_decisions(
            run / "decisions" / f"{day[0]:04d}-{day[1]:02d}-{day[2]:02d}.parquet",
            _decision_rows(day), LEADS, features=False,
        )
    _write_gauge(tmp_path / "corpus")
    assert fit.main([
        "--run", str(run),
        "--corpus-dir", str(tmp_path / "corpus"),
        "--out-dir", str(tmp_path / "out"),
        "--leads", "30", "--resamples", "0",
    ]) == 2


def test_the_loader_carries_features_through_the_same_deduplication(
    corpus: Path,
) -> None:
    rows = fit.load_rows(
        [corpus / "replay"], LEADS, (10, 20, 30, 45, 60),
        stations=None, log=None,
    )
    assert set(rows["extra"]) == set(fit.feature_source_columns((10, 20, 30, 45, 60)))
    assert rows["radar_ts"].shape == rows["t"].shape
    assert numpy.all(numpy.isfinite(rows["extra"]["up_max_40km_mm_h"]))
    # "No echo upwind" survives as a NaN, never as a distance of zero.
    assert numpy.any(numpy.isnan(rows["extra"]["up_dist_km"]))
    features = fit.build_features(rows)
    assert set(features["season"]) <= set(pp.SEASONS)
    assert features["hour_utc"].min() >= 0


# ---------------------------------------------------------------------------
# Post-processing v2: the arms, and the evaluation protocol around them
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_v2(corpus: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple:
    """One run with every v2 arm on, against a REFITTED v1 baseline.

    The point is the protocol, not the skill: this fixture is three
    synthetic weeks, so nothing in it is evidence about Denmark. What it
    does prove is that the candidate and the baseline are both fitted
    inside the same folds, that every comparison comes back on the dry
    subset as well as on all rows, and that the learning curve and the
    ablation produce the rows the report renders.
    """
    out = tmp_path_factory.mktemp("fit-out-v2")
    report = _run(
        corpus, out,
        "--design", "v2",
        "--station-offsets",
        "--isotonic", "per-season",
        "--baseline", "refit-v1",
        "--learning-curve", "3,7",
        "--ablate",
    )
    return report, out


class TestTheV2Arms:
    def test_the_settings_block_records_what_was_fitted(self, fitted_v2) -> None:
        report, _out = fitted_v2
        settings = report["settings"]
        assert settings["design"] == "v2"
        assert settings["model"] == "logistic"
        assert settings["isotonic"] == "per-season"
        assert settings["station_offsets"] is True
        assert settings["baseline"] == "refit-v1"
        assert settings["fit"]["design"] == "v2"
        assert settings["baseline_fit"]["design"] == "v1"

    def test_the_v2_design_is_wider_than_v1(self, fitted_v2) -> None:
        _report, out = fitted_v2
        model = pp.PostprocessModel.loads((out / "postprocess.json").read_text())
        assert model.spec.version == "v2"
        assert len(model.feature_names) > len(pp.design_columns(model.design_leads))
        assert model.spec.interactions
        assert set(model.spec.log_columns) <= set(model.spec.extra_columns)

    def test_the_artefact_carries_the_knots_and_the_offsets(self, fitted_v2) -> None:
        _report, out = fitted_v2
        payload = json.loads((out / "postprocess.json").read_text())
        design = payload["design"]
        assert design["version"] == "v2"
        for entry in design["splines"]:
            assert len(entry["knots"]) >= 3
            assert entry["knots"] == sorted(entry["knots"])
        offsets = payload["models"]["20"]["station_offsets"]
        assert set(offsets) == set(STATIONS)

    def test_a_v2_model_still_scores_rows_through_the_public_api(
        self, fitted_v2, corpus: Path,
    ) -> None:
        """``predict`` is unchanged for callers — the spec does the work."""
        _report, out = fitted_v2
        model = pp.PostprocessModel.loads((out / "postprocess.json").read_text())
        rows = fit.load_rows(
            [corpus / "replay"], LEADS, model.design_leads, stations=None, log=None,
        )
        predicted = model.predict(fit.build_features(rows))
        assert set(predicted) == set(LEADS)
        for values in predicted.values():
            assert values.shape == rows["t"].shape
            assert numpy.all((values >= 0.0) & (values <= 1.0))


class TestTheDrySubset:
    def test_every_lead_is_scored_on_the_dry_rows_as_well(self, fitted_v2) -> None:
        report, _out = fitted_v2
        assert report["dry_rows"] > 0
        for lead in LEADS:
            block = report["evaluation"]["leads"][str(lead)][pp.DRY]
            assert 0 < block["n"] <= report["evaluation"]["leads"][str(lead)]["all"]["n"]
            assert "bss" in block["baseline"]
            assert "bss" in block["postprocess"]

    def test_the_headline_carries_the_dry_numbers(self, fitted_v2) -> None:
        report, _out = fitted_v2
        headline = fit._headline(report)
        for lead in LEADS:
            assert "dry" in headline["leads"][str(lead)]
            assert headline["leads"][str(lead)]["dry"]["n"] > 0

    def test_the_subset_says_where_it_came_from(self, fitted_v2) -> None:
        report, _out = fitted_v2
        assert report["dry_source"] in {"column", "derived from the gauge store"}


class TestTheLearningCurveAndAblation:
    def test_one_learning_curve_row_per_budget(self, fitted_v2) -> None:
        report, _out = fitted_v2
        rows = report["learning_curve"]
        assert [row["days"] for row in rows] == [3, 7]
        assert report["settings"]["learning_curve_order"] == "random"
        assert report["settings"]["learning_curve_seed"] == 0
        for row in rows:
            assert row["train_rows"] > 0
            assert set(row["leads"]) == {str(lead) for lead in LEADS}
            assert "all" in row["leads"]["20"]
            # The draw, so a reader of the JSON can see what it stands on.
            assert row["order"] == "random" and row["seed"] == 0
            assert row["train_days"] == row["days"]
            assert sum(row["months"].values()) == row["days"] * len(MONTHS)
            for fold in row["folds"]:
                # Two of the three fixture months per fold, both drawn from.
                assert len(fold["months"]) == len(MONTHS) - 1
                assert sum(fold["months"].values()) == row["days"]
        # The curve is scored on the rows the out-of-fold table scored.
        for lead in LEADS:
            table = report["evaluation"]["leads"][str(lead)]
            for subset in ("all", pp.DRY):
                assert rows[-1]["leads"][str(lead)][subset]["n"] == (
                    table[subset]["n"]
                )

    def test_one_ablation_row_per_family(self, fitted_v2) -> None:
        report, _out = fitted_v2
        block = report["ablation"]
        families = [entry["family"] for entry in block["dropped"]]
        assert "raw_frac" in families and "up" in families
        assert families == sorted(families, key=pp.FAMILY_NAMES.index)
        for entry in block["dropped"]:
            assert entry["columns"] > 0
            assert "delta" in entry["leads"]["20"]["all"]

    def test_the_markdown_grows_the_new_sections(self, fitted_v2) -> None:
        _report, out = fitted_v2
        text = (out / "postprocess_report.md").read_text()
        assert "## Candidate vs baseline" in text
        assert "## Learning curve" in text
        assert "| days | train days | train rows | subset | rows |" in text
        assert "Draw order `random`, seed 0." in text
        assert "## Ablation" in text
        assert "| lead | subset | n |" in text
        assert "refit-v1" in text

    def test_a_default_run_does_not_grow_them(self, fitted) -> None:
        """The extra passes cost minutes; they only happen when asked for."""
        report, out, _back, _corpus = fitted
        assert report["learning_curve"] == []
        assert report["ablation"] is None
        text = (out / "postprocess_report.md").read_text()
        assert "## Learning curve" not in text
        assert "## Ablation" not in text
        # ...but the summary table and the dry subset are always there.
        assert "## Candidate vs baseline" in text


def test_the_default_run_is_still_the_shipped_configuration(fitted) -> None:
    """No flag, no change: the same family, design, calibration and baseline."""
    report, out, _back, _corpus = fitted
    assert report["settings"]["model"] == "logistic"
    assert report["settings"]["design"] == "v1"
    assert report["settings"]["isotonic"] == "pooled"
    assert report["settings"]["station_offsets"] is False
    assert report["settings"]["baseline"] == "curve"
    model = pp.PostprocessModel.loads((out / "postprocess.json").read_text())
    assert model.kind == pp.KIND_LOGISTIC
    assert model.spec == pp.DesignSpec()
    assert model.feature_names == pp.design_columns(model.design_leads)


def test_tree_params_parse_into_the_types_lightgbm_wants() -> None:
    parsed = fit.parse_tree_params("n_estimators=500,max_depth=6,learning_rate=0.03")
    assert parsed == {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.03,
    }
    assert fit.parse_tree_params("") == {}
    with pytest.raises(ValueError, match="key=value"):
        fit.parse_tree_params("n_estimators")


def test_a_baseline_spelling_the_script_does_not_know_is_a_usage_error(
    corpus: Path, tmp_path: Path,
) -> None:
    assert fit.main([
        "--run", str(corpus / "replay"),
        "--corpus-dir", str(corpus / "corpus"),
        "--out-dir", str(tmp_path / "out"),
        "--leads", "30", "--resamples", "0",
        "--baseline", "whatever",
    ]) == 2


def test_the_baseline_can_be_read_off_an_existing_model(
    fitted, corpus: Path, tmp_path: Path,
) -> None:
    """``--baseline model:<path>`` refits what is in service, in fold."""
    _report, out, _back, _corpus = fitted
    report = _run(
        corpus, tmp_path / "out",
        "--design", "v2",
        "--baseline", f"model:{out / 'postprocess.json'}",
        "--resamples", "0",
    )
    assert report["settings"]["baseline_fit"]["design"] == "v1"
    assert report["settings"]["baseline_fit"]["kind"] == "logistic"


def test_the_four_arms_can_be_scored_side_by_side(
    corpus: Path, tmp_path: Path,
) -> None:
    """``--compare`` puts every family in one table, on one set of folds.

    LightGBM is not in this environment, so the two tree arms are not
    exercised here — ``tests/test_postprocess_trees.py`` under
    ``.venv-fit`` is where those live. What this pins is that the arms
    share the folds, share the baseline, and only ``--model`` reaches
    ``postprocess.json``.
    """
    report = _run(
        corpus, tmp_path / "out",
        "--model", "logistic",
        "--compare", "logistic,logistic-shared",
        "--baseline", "refit-v1",
        "--resamples", "0",
    )
    assert [arm["model"] for arm in report["arms"]] == ["logistic-shared"]
    for arm in report["arms"]:
        for lead in LEADS:
            assert arm["leads"][str(lead)]["all"]["n"] > 0
    text = (tmp_path / "out" / "postprocess_report.md").read_text()
    assert "| `logistic-shared` |" in text
    assert "| `logistic` |" in text
    model = pp.PostprocessModel.loads(
        (tmp_path / "out" / "postprocess.json").read_text()
    )
    assert model.kind == "logistic"


def test_a_shared_fit_ships_one_coefficient_vector_and_stays_ordered(
    corpus: Path, tmp_path: Path,
) -> None:
    _report = _run(
        corpus, tmp_path / "out",
        "--model", "logistic-shared", "--design", "v2", "--resamples", "0",
    )
    model = pp.PostprocessModel.loads(
        (tmp_path / "out" / "postprocess.json").read_text()
    )
    assert model.kind == "logistic-shared"
    assert model.is_shared
    assert model.spec.lead_column == "lead_min"
    assert [name for name, _s in model.spec.own_columns][0] == "raw_frac_own"
    first = model.models[LEADS[0]].coefficients
    assert all(model.models[lead].coefficients == first for lead in LEADS)

    rows = fit.load_rows(
        [corpus / "replay"], LEADS, model.design_leads, stations=None, log=None,
    )
    predicted = model.predict(fit.build_features(rows))
    for shorter, longer in zip(LEADS, LEADS[1:]):
        assert numpy.all(predicted[longer] >= predicted[shorter] - 1e-12)


def test_an_unknown_comparison_arm_is_a_usage_error(
    corpus: Path, tmp_path: Path,
) -> None:
    assert fit.main([
        "--run", str(corpus / "replay"),
        "--corpus-dir", str(corpus / "corpus"),
        "--out-dir", str(tmp_path / "out"),
        "--leads", "30", "--resamples", "0",
        "--compare", "logistic,randomforest",
    ]) == 2


# ---------------------------------------------------------------------------
# --target onset: the push's own event as the label
# ---------------------------------------------------------------------------


def _onset_inputs(corpus: Path) -> tuple:
    """``(grid, onsets, scored)`` over the fixture archive, the script's way."""
    from dmi_nowcast_sidecar.threshold_sweep import gauge_truth

    days = _days()
    window = (_at(days[0], 0), _at(days[-1], 23 * 60))
    grid, _dead, scored = bench.build_gauge_grid(
        corpus / "corpus", list(STATIONS), window,
        dry_min=60, onset_min_mm=0.2, min_known_slots=500,
    )
    onsets, _known_until, _slots, _dead = gauge_truth(
        corpus / "corpus", list(STATIONS), window,
    )
    return grid, onsets, scored


class TestTheOnsetLabel:
    """Hand-placed rows against the fixture's planted events.

    Day 2026-01-01, station A: the first wet slot ENDS at 09:30 and the
    rain lasts six slots (to 10:20). Gauge slots are reported 04:00–20:00
    only, so a window reaching past 20:00 has unknown slots.
    """

    DAY = (2026, 1, 1)

    def _rows(self, scored, cases) -> dict:
        t = numpy.array(
            [int(_at(self.DAY, minute).timestamp()) for minute, _s in cases],
            dtype=numpy.int64,
        )
        station = numpy.array(
            [scored.index(s) for _m, s in cases], dtype=numpy.int64,
        )
        return {"t": t, "station": station,
                "dropped": numpy.zeros(t.size, dtype=bool)}

    def test_the_planted_cases(self, corpus: Path) -> None:
        grid, onsets, scored = _onset_inputs(corpus)
        first = _first_wet_slot_min(self.DAY, STATION_A)
        assert first == 9 * 60 + 30
        cases = [
            (first - 20, STATION_A),    # onset inside (t, t+20+10]  -> 1
            (first - 30, STATION_A),    # onset at t+30: only the tolerance -> 1
            (7 * 60, STATION_A),        # a fully reported dry window -> 0
            (19 * 60 + 50, STATION_A),  # window runs into unreported slots -> excluded
            (first + 30, STATION_A),    # wet in the hour before, no onset -> 0, kept
        ]
        rows = self._rows(scored, cases)
        truth = fit.build_onset_truth(
            rows, grid, onsets, scored, [20], tolerance_min=10,
        )
        y, usable = truth[20]
        assert list(usable) == [True, True, True, False, True]
        assert list(y[usable]) == [1.0, 1.0, 0.0, 0.0]

        # Without the tolerance the onset at t+30 is outside a 20-min window.
        strict = fit.build_onset_truth(
            rows, grid, onsets, scored, [20], tolerance_min=0,
        )
        assert strict[20][0][1] == 0.0 and strict[20][1][1]

        # The dry-rows rule drops the wet-before row and nothing else.
        dry = pp.dry_subset(
            {"x": numpy.zeros(rows["t"].size)},
            grid=grid, t=rows["t"], station=rows["station"],
        )
        assert not dry[4] and dry[0] and dry[2]
        restricted = fit.build_onset_truth(
            rows, grid, onsets, scored, [20], tolerance_min=10, restrict=dry,
        )
        assert not restricted[20][1][4]
        assert restricted[20][1][0] and restricted[20][1][2]

    def test_a_dropped_row_is_excluded(self, corpus: Path) -> None:
        grid, onsets, scored = _onset_inputs(corpus)
        rows = self._rows(scored, [(7 * 60, STATION_A)])
        rows["dropped"] = numpy.array([True])
        truth = fit.build_onset_truth(rows, grid, onsets, scored, [20])
        assert not truth[20][1][0]

    def test_the_onsets_are_the_sweeps(self, corpus: Path) -> None:
        """One onset per (day, station): the first wet slot of the event."""
        _grid, onsets, _scored = _onset_inputs(corpus)
        for station in STATIONS:
            got = sorted(onsets[station])
            want = sorted(
                _at(day, _first_wet_slot_min(day, station)) for day in _days()
            )
            assert got == want


@pytest.fixture(scope="module")
def fitted_onset(corpus: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple:
    out = tmp_path_factory.mktemp("fit-out-onset")
    back = tmp_path_factory.mktemp("fit-back-onset") / "replay_onset"
    report = _run(
        corpus, out, "--target", "onset", "--baseline", "refit-v1",
        "--write-back", str(back),
    )
    return report, out, back


class TestTheOnsetFit:
    def test_the_report_names_the_target(self, fitted_onset) -> None:
        report, out, _back = fitted_onset
        settings = report["settings"]
        assert settings["target"] == "onset"
        assert settings["onset_tolerance_min"] == 10
        assert settings["onset_rows"] == "all"
        assert settings["column_template"] == "p_onset_{lead}"
        text = (out / "postprocess_report.md").read_text()
        assert "**Target `onset`**" in text
        assert "`(t, t + L + 10]`" in text
        assert "Rows `all`" in text
        assert "| target | onset in `(t, t + L + 10]`, rows `all` |" in text
        assert fit._headline(report)["target"] == "onset"

    def test_every_metric_is_scored_on_the_onset_label(self, fitted_onset) -> None:
        """One onset per (day, station) → a base rate far below the wet one."""
        report, _out, _back = fitted_onset
        for lead in LEADS:
            counts = report["target_counts"][str(lead)]
            block = report["evaluation"]["leads"][str(lead)]["all"]
            assert block["n"] == counts["rows"]
            assert block["baseline"]["base_rate"] == pytest.approx(
                counts["base_rate"],
            )
            assert block["postprocess"]["base_rate"] == pytest.approx(
                counts["base_rate"],
            )
            # 14 days x 2 stations, one onset each, is at most a handful
            # of rows per event inside a 30–40 min window.
            assert 0 < counts["positives"] < 0.1 * counts["rows"]

    def test_the_artefact_says_onset(self, fitted_onset) -> None:
        _report, out, _back = fitted_onset
        payload = json.loads((out / "postprocess.json").read_text())
        assert payload["target"] == "onset"
        assert payload["training"]["target"] == "onset"
        assert payload["training"]["onset_tolerance_min"] == 10
        assert payload["training"]["onset_rows"] == "all"
        model = pp.PostprocessModel.loads((out / "postprocess.json").read_text())
        assert model.target == pp.TARGET_ONSET

    def test_the_write_back_column_and_summary(self, fitted_onset) -> None:
        import pyarrow.parquet as pq

        _report, _out, back = fitted_onset
        table = pq.read_table(sorted((back / "decisions").glob("*.parquet"))[0])
        names = set(table.schema.names)
        assert {"p_onset_20", "p_onset_30"} <= names
        assert not {"p_post_20", "p_post_30"} & names
        block = json.loads((back / "summary.json").read_text())["postprocess"]
        assert block["column_template"] == "p_onset_{lead}"
        assert block["target"] == "onset"
        assert block["onset_tolerance_min"] == 10
        assert block["onset_rows"] == "all"

    def test_the_benchmark_reads_the_onset_column(
        self, fitted_onset, corpus: Path, tmp_path: Path,
    ) -> None:
        _report, _out, back = fitted_onset
        out = tmp_path / "layer-b-onset"
        assert bench.main([
            "--baseline", str(back), "--out-dir", str(out),
            "--corpus-dir", str(corpus / "corpus"),
            "--leads", "20,30", "--layers", "b", "--resamples", "0",
            "--probability-column", "p_onset_{lead}",
        ]) == 0
        posted = json.loads((out / "benchmark.json").read_text())
        assert posted["settings"]["probability_column"] == "p_onset_{lead}"
        assert posted["layer_b"]["leads"]["30"]["all"]["baseline"]["n"] > 0

    def test_dry_rows_restrict_training_and_scoring(
        self, fitted_onset, corpus: Path, tmp_path: Path,
    ) -> None:
        report_all, _out, _back = fitted_onset
        report = _run(
            corpus, tmp_path / "dry", "--target", "onset",
            "--onset-rows", "dry", "--resamples", "0",
        )
        assert report["settings"]["onset_rows"] == "dry"
        for lead in LEADS:
            every = report["evaluation"]["leads"][str(lead)]
            assert every["all"]["n"] == every[pp.DRY]["n"]
            assert every["all"]["n"] < (
                report_all["evaluation"]["leads"][str(lead)]["all"]["n"]
            )
            # ...and the dry subset itself is the same rows either way.
            assert every[pp.DRY]["n"] == (
                report_all["evaluation"]["leads"][str(lead)][pp.DRY]["n"]
            )
        assert report["dry_rows"] == report_all["dry_rows"]
        assert "Rows `dry`" in (tmp_path / "dry" / "postprocess_report.md").read_text()

    def test_onset_rows_without_the_onset_target_is_a_usage_error(
        self, corpus: Path, tmp_path: Path,
    ) -> None:
        assert fit.main([
            "--run", str(corpus / "replay"),
            "--corpus-dir", str(corpus / "corpus"),
            "--out-dir", str(tmp_path / "out"),
            "--leads", "30", "--resamples", "0",
            "--onset-rows", "dry",
        ]) == 2


def test_target_wet_is_the_default_to_the_number(
    fitted, corpus: Path, tmp_path: Path,
) -> None:
    """``--target wet`` spelled out changes nothing about the fit or its scores."""
    report_default, out_default, _back, _corpus = fitted
    report = _run(corpus, tmp_path / "wet", "--target", "wet")
    assert report["settings"]["target"] == "wet"
    assert report_default["settings"]["target"] == "wet"
    assert report["evaluation"]["leads"] == report_default["evaluation"]["leads"]
    a = json.loads((out_default / "postprocess.json").read_text())
    b = json.loads((tmp_path / "wet" / "postprocess.json").read_text())
    assert a["models"] == b["models"]
    assert a["target"] == b["target"] == "wet"
    summary = json.loads((_back / "summary.json").read_text())["postprocess"]
    assert summary["column_template"] == "p_post_{lead}"
    assert summary["target"] == "wet"
