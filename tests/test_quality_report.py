"""The quality.json producer (Phase F, F4).

Three properties are under test, and they are the three ways this document
can lie:

1. **A missing input nulls its section.** Never a zero, never a stale
   number borrowed from a neighbour. The page renders null as "not
   measured yet", so a section that quietly became 0 would put a false
   claim on screen with a decimal point on it.
2. **The numbers are the ones the SQL would give.** The reliability bins
   here and ``sql/reliability_pooled.sql`` are the same ten bins with the
   same weights, checked against DuckDB on a table small enough to work
   by hand.
3. **The document the client would parse is the document we wrote.** The
   assertions ``frontend/src/lib/quality/load.test.ts`` makes on its
   fixture are mirrored here against a freshly built report, so a producer
   change that silently drops a section on the browser side fails on this
   side first.

Everything is synthetic: a hand-built corpus, a hand-built gauge archive
and a hand-built replay directory, all under ``tmp_path``. No network, no
DMI, no real STEPS run.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from dmi_nowcast_core.quality_report import (
    N_BINS,
    SCHEMA_VERSION,
    QualityInputs,
    bin_statistics,
    build_quality_report,
    render_markdown,
    validate_report,
)
from dmi_nowcast_core.warning_score import decision_schema

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

UTC = timezone.utc
#: A fixed "now" so every window in the fixtures is deterministic.
NOW = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
#: The day the replay and the live scoreboard both cover.
DAY = datetime(2026, 9, 3, tzinfo=UTC)
LEADS = (10, 20, 30, 45, 60)
STATIONS = ("06180", "06181", "06182")

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _corpus_table(rows: list[dict], *, gauge: bool) -> "pa.Table":
    """A calibration-corpus table, optionally widened with gauge truth."""
    fields = [
        ("event_time", pa.string()),
        ("point_id", pa.string()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("region", pa.string()),
        ("lead_min", pa.int32()),
        ("raw_prob", pa.float32()),
        ("outcome", pa.int8()),
        ("sample_weight", pa.float64()),
        ("frame_age_min", pa.float32()),
        ("threshold_mm_h", pa.float64()),
    ]
    if gauge:
        fields += [
            ("gauge_mm", pa.float32()),
            ("gauge_dur_min", pa.float32()),
            ("gauge_outcome", pa.int8()),
        ]
    schema = pa.schema(fields)
    return pa.table(
        {
            name: pa.array([r.get(name) for r in rows], type=schema.field(name).type)
            for name in schema.names
        },
        schema=schema,
    )


def _corpus_rows(
    *, gauge: bool, n_events: int = 60, step_hours: int = 24,
) -> list[dict]:
    """A corpus with every bin populated at every lead, and a real signal.

    ``raw_prob`` walks the whole [0, 1] range so no bin is empty at the
    headline lead, and the outcome is drawn deterministically from the
    probability so the reliability diagram is close to the diagonal rather
    than nonsense.

    One event a day by default, so 60 events span two calendar months and
    the radar reliability's leave-one-month-out folds are real months —
    the production path, not the day-level fallback.
    """
    rows: list[dict] = []
    start = datetime(2026, 3, 1, tzinfo=UTC)
    for event in range(n_events):
        stamp = (start + timedelta(hours=event * step_hours)).isoformat()
        for point_index, station in enumerate(STATIONS):
            for lead in LEADS:
                # A deterministic spread over [0, 1), one step per row.
                step = (event * 7 + point_index * 3 + lead) % 100
                prob = step / 100.0
                # Outcome tracks the probability with a lead-dependent
                # miscalibration, so longer leads are visibly worse.
                threshold = 0.5 + (lead - 30) / 400.0
                outcome = 1 if prob > threshold else 0
                row = {
                    "event_time": stamp,
                    "point_id": station,
                    "lat": 55.0 + point_index * 0.5,
                    "lon": 10.0 + point_index * 0.5,
                    "region": "Denmark",
                    "lead_min": lead,
                    "raw_prob": prob,
                    "outcome": outcome,
                    "sample_weight": 1.0 + (step % 5) * 0.25,
                    "frame_age_min": 14.0 + (event % 11),
                    "threshold_mm_h": 0.5,
                }
                if gauge:
                    row["gauge_outcome"] = 1 if prob > threshold + 0.05 else 0
                    row["gauge_mm"] = 0.2 if row["gauge_outcome"] else 0.0
                    row["gauge_dur_min"] = 3.0 if row["gauge_outcome"] else 0.0
                rows.append(row)
    return rows


def write_radar_corpus(path: Path) -> Path:
    pq.write_table(_corpus_table(_corpus_rows(gauge=False), gauge=False), path)
    return path


def write_station_corpus(path: Path) -> Path:
    pq.write_table(_corpus_table(_corpus_rows(gauge=True), gauge=True), path)
    return path


def write_curves(path: Path) -> Path:
    """Per-lead isotonic curves that visibly shrink the raw probability.

    Two breakpoints is enough to make "calibrated" and "raw" different
    numbers, which is the only thing the tests need from them.
    """
    path.write_text(json.dumps({
        "metadata": {"fitted_at": "2026-09-01T00:00:00+00:00"},
        "curves": {
            str(lead): {
                "raw_breakpoints": [0.0, 1.0],
                "calibrated_values": [0.0, 1.0 - lead / 200.0],
            }
            for lead in LEADS
        },
    }))
    return path


def write_and_load_curves(path: Path) -> dict:
    """The curve file, already parsed into calibrators."""
    from dmi_nowcast_core.calibrate import load_calibration_curves

    return load_calibration_curves(write_curves(path))


def _obs_rows(station: str, wet_slots: set[datetime]) -> list[dict]:
    """A whole day of 10-min gauge slots for one station."""
    rows: list[dict] = []
    slot = DAY
    end = DAY + timedelta(days=1)
    while slot <= end:
        wet = slot in wet_slots
        rows.append({
            "station_id": station,
            "observed_utc": slot,
            "parameter_id": "precip_past10min",
            "value": 0.4 if wet else 0.0,
        })
        rows.append({
            "station_id": station,
            "observed_utc": slot,
            "parameter_id": "precip_dur_past10min",
            "value": 5.0 if wet else 0.0,
        })
        slot += timedelta(minutes=10)
    return rows


#: Rain arrives at 06180 three times, each after a long dry spell, so each
#: is an onset the warning before it can claim.
ONSET_TIMES = (
    DAY + timedelta(hours=2, minutes=10),
    DAY + timedelta(hours=4, minutes=10),
    DAY + timedelta(hours=6, minutes=10),
)
#: 06181 gets one, so its two warnings can be one hit and one false alarm.
ONSET_TIMES_2 = (DAY + timedelta(hours=3, minutes=0),)


#: Onsets in the BACKFILLED archive, on days before any decision row
#: exists. This is the shape of the defect: the gauge store is read a
#: month at a time (it has to be — the onset rule needs the dry slots
#: before an event), while the decisions cover a single day, and every
#: event in the difference was being counted as a miss.
ARCHIVE_ONSETS = (
    DAY - timedelta(days=2) + timedelta(hours=6),    # 09-01 06:00
    DAY - timedelta(days=2) + timedelta(hours=18),   # 09-01 18:00
    DAY - timedelta(days=1) + timedelta(hours=6),    # 09-02 06:00
    DAY - timedelta(days=1) + timedelta(hours=18),   # 09-02 18:00
)
#: An unclaimed onset ON the decision day, inside coverage: a real miss,
#: and the control that proves the fix did not simply stop counting.
COVERED_MISS = DAY + timedelta(hours=9)


def _archive_rows() -> list[dict]:
    """Gauge slots from the start of the month up to the fixture day.

    All dry except around :data:`ARCHIVE_ONSETS`, so the onset rule finds
    them — the point of the test is that they ARE detected and then
    deliberately left unscored, not that they are never seen.
    """
    wet = {
        onset + timedelta(minutes=10 * k)
        for onset in ARCHIVE_ONSETS for k in range(3)
    }
    rows: list[dict] = []
    slot = DAY.replace(day=1)
    while slot < DAY:
        is_wet = slot in wet
        rows.append({
            "station_id": "06180", "observed_utc": slot,
            "parameter_id": "precip_past10min", "value": 0.4 if is_wet else 0.0,
        })
        slot += timedelta(minutes=10)
    # …and one wet spell at a station that never warns, mid-decision-day.
    rows += [
        {
            "station_id": "06182",
            "observed_utc": COVERED_MISS + timedelta(minutes=10 * k),
            "parameter_id": "precip_past10min", "value": 0.4,
        }
        for k in range(3)
    ]
    return rows


def write_gauge_store(corpus_dir: Path, *, with_archive: bool = False) -> None:
    from dmi_nowcast_core.station_store import obs_schema

    wet_by_station = {
        "06180": {t + timedelta(minutes=k * 10) for t in ONSET_TIMES for k in range(3)},
        "06181": {t + timedelta(minutes=k * 10) for t in ONSET_TIMES_2 for k in range(3)},
        "06182": set(),
    }
    rows = [r for s in STATIONS for r in _obs_rows(s, wet_by_station[s])]
    if with_archive:
        rows += _archive_rows()
    schema = obs_schema()
    table = pa.table(
        {
            name: pa.array([r[name] for r in rows], type=schema.field(name).type)
            for name in schema.names
        },
        schema=schema,
    )
    out = corpus_dir / "stations" / "obs" / f"{DAY.year:04d}"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / f"{DAY.month:02d}.parquet")


def write_catalogue(corpus_dir: Path) -> None:
    from dmi_nowcast_core.station_store import catalogue_schema

    names = {"06180": "Køge", "06181": "Aarhus Syd", "06182": "Skagen Fyr"}
    kinds = {"06180": "Synop", "06181": "Pluvio", "06182": "Synop"}
    rows = [
        {
            "station_id": s, "name": names[s], "kind": kinds[s],
            "lat": 55.0 + i * 0.5, "lon": 10.0 + i * 0.5, "country": "DNK",
            "operation_from": None, "operation_to": None, "status": "Active",
            "parameter_ids": ["precip_past10min"], "region_id": "Denmark",
        }
        for i, s in enumerate(STATIONS)
    ]
    schema = catalogue_schema()
    table = pa.table(
        {
            name: pa.array([r[name] for r in rows], type=schema.field(name).type)
            for name in schema.names
        },
        schema=schema,
    )
    (corpus_dir / "stations").mkdir(parents=True, exist_ok=True)
    pq.write_table(table, corpus_dir / "stations" / "catalogue.parquet")


def write_points(corpus_dir: Path) -> None:
    (corpus_dir / "stations").mkdir(parents=True, exist_ok=True)
    (corpus_dir / "stations" / "station_points.json").write_text(json.dumps({
        "version": 2,
        "points": [
            {
                "id": s, "lat": 55.0 + i * 0.5, "lon": 10.0 + i * 0.5,
                "region": "Denmark",
                "strata": {"station_kind": "Synop", "country_region": "DK"},
            }
            for i, s in enumerate(STATIONS)
        ],
    }))


def _decision(
    stamp: datetime, station: str, *, action: str, eta: float | None,
    p_rain: float, observed: float | None, forecast: float | None,
) -> dict:
    return {
        "radar_ts": stamp - timedelta(minutes=14),
        "generated_at": stamp,
        "station_id": station,
        "p_rain": p_rain,
        "eta_min": eta,
        "intensity_mm_h": 1.2,
        "observed_mm_h": observed,
        "forecast_now_mm_h": forecast,
        "action": action,
        "armed_after": action != "notify",
        "streak_after": 1,
    }


def _iso_z(stamp: datetime) -> str:
    """The producer's wire format for a timestamp."""
    return stamp.astimezone(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z",
    )


def _wet_at(station: str, stamp: datetime) -> bool:
    """Whether the gauge fixture has that station wet in that slot."""
    onsets = {"06180": ONSET_TIMES, "06181": ONSET_TIMES_2}.get(station, ())
    return any(
        onset <= stamp < onset + timedelta(minutes=30) for onset in onsets
    )


#: (station, minutes-into-DAY, eta) for every warning the fixtures send.
#: The first three at 06180 land inside their onset windows; the fourth is
#: a deliberate false alarm hours after the last rain.
WARNINGS = [
    ("06180", 110, 20.0),   # 01:50 → onset 02:10, lead error 0
    ("06180", 220, 30.0),   # 03:40 → onset 04:10, lead error 0
    ("06180", 350, 15.0),   # 05:50 → onset 06:10, lead error −5
    ("06180", 700, 25.0),   # 11:40 → nothing follows: false alarm
    ("06181", 160, 20.0),   # 02:40 → onset 03:00, hit
    ("06181", 690, 20.0),   # 11:30 → false alarm
]


def decision_rows(*, live: bool) -> list[dict]:
    """One decision per station per 10-min frame across the fixture day.

    ``live`` shifts the forecast column so a live row and a replay row for
    the same key are distinguishable — that is how the dedupe test proves
    which one survived.
    """
    warned = {(s, DAY + timedelta(minutes=m)): eta for s, m, eta in WARNINGS}
    rows: list[dict] = []
    stamp = DAY
    end = DAY + timedelta(hours=12)
    while stamp <= end:
        for station in STATIONS:
            wet = _wet_at(station, stamp)
            eta = warned.get((station, stamp))
            rows.append(_decision(
                stamp, station,
                action="notify" if eta is not None else "hold",
                eta=eta,
                p_rain=0.82 if eta is not None else 0.1,
                observed=(2.0 if wet else 0.0),
                # The live rows claim rain a notch harder, so a dedupe that
                # kept the replay row instead would change the numbers.
                forecast=(2.5 if wet else 0.0) + (0.5 if live else 0.0),
            ))
        stamp += timedelta(minutes=10)
    return rows


def _write_decisions(path: Path, rows: list[dict]) -> None:
    schema = decision_schema()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                name: pa.array(
                    [r.get(name) for r in rows], type=schema.field(name).type,
                )
                for name in schema.names
            },
            schema=schema,
        ),
        path,
    )


def write_replay(replay_dir: Path, rows: list[dict] | None = None) -> Path:
    _write_decisions(
        replay_dir / "decisions" / f"{DAY.date().isoformat()}.parquet",
        decision_rows(live=False) if rows is None else rows,
    )
    (replay_dir / "summary.json").write_text(json.dumps({
        "generated_at": "2026-09-04T02:00:00+00:00",
        "run": {
            "days": [DAY.date().isoformat()],
            "n_days": 1,
            "frame_age_min": 14.0,
            "rules": {
                "threshold_pct": 40, "lead_min": 30, "rearm_after_min": 60,
                "persistence_obs": 1, "raining_now_eta_min": 1.5,
                "raining_now_mm_h": 0.5,
            },
        },
    }))
    return replay_dir


def write_live_eval(corpus_dir: Path, rows: list[dict] | None = None) -> None:
    _write_decisions(
        corpus_dir / "stations" / "eval" / f"{DAY.year:04d}"
        / f"{DAY.month:02d}.parquet",
        decision_rows(live=True) if rows is None else rows,
    )


def write_persistence(path: Path) -> Path:
    path.write_text(json.dumps({
        "meta": {
            "days": 30,
            "cases (frames)": 3841,
            "day_list": ["2026-08-06", "2026-09-04"],
        },
        "aggregate": {
            "pooled": {
                "10": {
                    "advection": {"CSI": 0.6737, "POD": 0.79, "FAR": 0.18},
                    "persistence": {"CSI": 0.4976, "POD": 0.66, "FAR": 0.34},
                },
                "20": {
                    "advection": {"CSI": 0.55},
                    "persistence": {"CSI": 0.38},
                },
            },
        },
    }))
    return path


@pytest.fixture
def full_inputs(tmp_path: Path) -> QualityInputs:
    """Every input present — the report with nothing missing."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    write_gauge_store(corpus_dir)
    write_catalogue(corpus_dir)
    write_points(corpus_dir)
    write_live_eval(corpus_dir)
    return QualityInputs(
        radar_corpus=write_radar_corpus(tmp_path / "radar.parquet"),
        station_corpus=write_station_corpus(tmp_path / "station.parquet"),
        replay_dir=write_replay(tmp_path / "replay"),
        corpus_dir=corpus_dir,
        persistence_json=write_persistence(tmp_path / "pva.json"),
        national_curves=write_curves(tmp_path / "curves.json"),
        now=NOW,
    )


# ---------------------------------------------------------------------------
# Every section present
# ---------------------------------------------------------------------------


class TestFullReport:
    def test_every_section_is_populated(self, full_inputs: QualityInputs) -> None:
        report = build_quality_report(full_inputs)
        assert report["schema_version"] == SCHEMA_VERSION
        for key in ("windows", "headline", "reliability", "raining_now",
                    "stations", "events", "methods"):
            assert report[key] is not None, f"{key} should be measured here"
        assert report["windows"]["radar"] is not None
        assert report["windows"]["gauge"] is not None
        assert report["windows"]["live"] is not None
        assert report["headline"]["reliability"]["radar"] is not None
        assert report["headline"]["reliability"]["gauge"] is not None
        assert report["headline"]["warnings"] is not None
        assert report["headline"]["persistence_margin"] is not None

    def test_the_document_passes_its_own_schema_check(
        self, full_inputs: QualityInputs,
    ) -> None:
        assert validate_report(build_quality_report(full_inputs)) == []

    def test_windows_describe_the_two_corpora(
        self, full_inputs: QualityInputs,
    ) -> None:
        windows = build_quality_report(full_inputs)["windows"]
        assert windows["radar"]["events"] == 60
        # 60 events × 3 points × 5 leads, all of them valid.
        assert windows["radar"]["points"] == 60 * 3 * 5
        assert windows["gauge"]["stations"] == len(STATIONS)
        assert windows["radar"]["from"] == "2026-03-01T00:00:00Z"
        assert windows["live"]["days"] == 1

    def test_headline_reads_the_highest_well_populated_bin(
        self, full_inputs: QualityInputs,
    ) -> None:
        """The card quotes the most confident thing we say often enough.

        Not "the bin containing 0.7": calibration shrinks probabilities,
        and on a real corpus that bin can be empty while every number
        behind it is fine — which is exactly how the first live report
        came out with a null headline card.
        """
        report = build_quality_report(full_inputs)
        for truth in ("radar", "gauge"):
            headline = report["headline"]["reliability"][truth]
            assert headline is not None
            curve = next(
                c for c in report["reliability"][truth]
                if c["lead_min"] == headline["lead_min"]
            )
            bins = curve["bins"]
            binned = next(b for b in bins if [b["lo"], b["hi"]] == headline["bin"])
            confident = [b for b in bins if b["n"] >= 200 and b["forecast_mean"]
                         is not None]
            if confident:
                # The highest bin we say often enough to mean something.
                assert binned is confident[-1]
            else:
                # Fallback: the most populated bin above 0.3, never below.
                assert binned["lo"] >= 0.3
                upper = [b for b in bins if b["lo"] >= 0.3
                         and b["forecast_mean"] is not None]
                assert binned["n"] == max(b["n"] for b in upper)
            # said_pct/happened_pct are percentages rounded for the wire,
            # so they match the bin to a thousandth of a percent.
            assert headline["said_pct"] == pytest.approx(
                binned["forecast_mean"] * 100, abs=1e-3,
            )
            assert headline["happened_pct"] == pytest.approx(
                binned["observed_freq"] * 100, abs=1e-3,
            )
            assert headline["n"] == binned["n"]

    def test_persistence_margin_comes_from_the_study(
        self, full_inputs: QualityInputs,
    ) -> None:
        margin = build_quality_report(full_inputs)["headline"]["persistence_margin"]
        assert margin == {
            "horizon_min": 10,
            "csi_advection": 0.6737,
            "csi_persistence": 0.4976,
            "frames": 3841,
            "from": "2026-08-06T00:00:00Z",
            "to": "2026-09-04T00:00:00Z",
        }

    def test_warnings_are_scored_against_the_gauge(
        self, full_inputs: QualityInputs,
    ) -> None:
        warnings = build_quality_report(full_inputs)["headline"]["warnings"]
        assert warnings["warnings"] == len(WARNINGS)
        assert warnings["hits"] == 4          # three at 06180, one at 06181
        assert warnings["false_alarms"] == 2
        assert warnings["hits"] + warnings["false_alarms"] == warnings["warnings"]
        assert warnings["window_days"] == 1
        assert warnings["n_stations"] == len(STATIONS)
        assert 0.0 <= warnings["pod"] <= 1.0
        assert 0.0 <= warnings["far"] <= 1.0
        # Positive means the rain beat the ETA. The fixture's warnings are
        # on time or five minutes early, so the median must not be late.
        assert warnings["lead_error_min"]["p50"] <= 0.0

    def test_raining_now_scores_both_series(
        self, full_inputs: QualityInputs,
    ) -> None:
        raining = build_quality_report(full_inputs)["raining_now"]
        assert raining["n_slots"] > 0
        for key in ("agreement", "pod", "far", "observation_agreement"):
            assert 0.0 <= raining[key] <= 1.0

    def test_methods_records_the_rules_actually_used(
        self, full_inputs: QualityInputs,
    ) -> None:
        methods = build_quality_report(full_inputs)["methods"]
        assert methods["sources"] == {
            "radar": "DMI radar composites (CC BY 4.0)",
            "gauges": "DMI meteorological observations, metObs (CC BY 4.0)",
        }
        assert methods["threshold_mm_h"] == 0.5
        assert methods["frame_age_range_min"] == [14.0, 24.0]
        assert methods["subscriber_rule"]["threshold_pct"] == 40
        assert methods["subscriber_rule"]["lead_min"] == 30
        assert "0.1 mm" in methods["gauge_wet_rule"]
        assert "30 minutes of known-dry" in methods["onset_rule"]
        # The two truths are two different claims and must be named as such.
        assert methods["reliability_probability"] == (
            "radar: calibrated out-of-sample, leave-one-month-out CV; "
            "gauges: the served curves against gauge truth"
        )
        assert "raw →" in methods["reliability_brier_improvement"]
        assert "n ≥ 200" in methods["headline_bin_rule"]
        assert "pending" in methods["pending_rule"]


# ---------------------------------------------------------------------------
# Missing inputs
# ---------------------------------------------------------------------------


class TestMissingInputs:
    def test_no_inputs_at_all_nulls_every_section(self) -> None:
        report = build_quality_report(QualityInputs(now=NOW))
        assert report["schema_version"] == SCHEMA_VERSION
        assert report["generated_at_utc"] == "2026-09-05T12:00:00Z"
        assert report["windows"] == {"radar": None, "gauge": None, "live": None}
        assert report["headline"]["reliability"] == {"radar": None, "gauge": None}
        assert report["headline"]["warnings"] is None
        assert report["headline"]["persistence_margin"] is None
        assert report["reliability"] == {"radar": None, "gauge": None}
        assert report["raining_now"] is None
        assert report["stations"] is None
        assert report["events"] is None
        assert report["methods"] is None
        assert validate_report(report) == []

    def test_a_path_that_does_not_exist_is_a_missing_input(
        self, tmp_path: Path,
    ) -> None:
        report = build_quality_report(QualityInputs(
            radar_corpus=tmp_path / "nope.parquet",
            replay_dir=tmp_path / "nowhere",
            persistence_json=tmp_path / "nothing.json",
            now=NOW,
        ))
        assert report["windows"]["radar"] is None
        assert report["headline"]["persistence_margin"] is None
        assert report["reliability"]["radar"] is None

    @pytest.mark.parametrize(
        ("drop", "nulled"),
        [
            ("radar_corpus", ["windows.radar", "reliability.radar",
                              "headline.reliability.radar"]),
            ("station_corpus", ["windows.gauge", "reliability.gauge",
                                "headline.reliability.gauge"]),
            ("persistence_json", ["headline.persistence_margin"]),
        ],
    )
    def test_dropping_one_input_nulls_only_its_sections(
        self, full_inputs: QualityInputs, drop: str, nulled: list[str],
    ) -> None:
        from dataclasses import replace

        report = build_quality_report(replace(full_inputs, **{drop: None}))
        for path in nulled:
            node = report
            for part in path.split("."):
                node = node[part]
            assert node is None, f"{path} should be null without {drop}"
        # …and the untouched neighbours are still there.
        assert report["headline"]["warnings"] is not None
        assert report["stations"] is not None
        assert validate_report(report) == []

    def test_without_a_gauge_store_there_is_no_warning_scoreboard(
        self, tmp_path: Path,
    ) -> None:
        """Decisions with no gauge behind them grade nothing.

        The alternative — scoring the radar against itself — would produce
        a POD and a FAR that look like measurements and are not.
        """
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        write_points(corpus_dir)
        write_live_eval(corpus_dir)
        report = build_quality_report(QualityInputs(
            corpus_dir=corpus_dir, now=NOW,
        ))
        assert report["headline"]["warnings"] is None
        assert report["raining_now"] is None
        assert report["events"] is None
        # The live window is still honest about what rows exist.
        assert report["windows"]["live"] is not None


# ---------------------------------------------------------------------------
# Bins, weights, and the SQL
# ---------------------------------------------------------------------------


class TestBinning:
    def test_bins_are_the_ten_fixed_bins_with_one_upper_fold(self) -> None:
        bins = bin_statistics([0.0, 0.05, 0.15, 0.999, 1.0], [0, 0, 1, 1, 1],
                              [1.0, 1.0, 1.0, 1.0, 1.0])
        assert len(bins) == N_BINS
        assert bins[0]["n"] == 2
        assert bins[1]["n"] == 1
        # 0.999 AND 1.0 both land in bin 9 — LEAST(..., 9) in the SQL.
        assert bins[9]["n"] == 2
        assert bins[5]["n"] == 0
        assert bins[5]["forecast_mean"] is None
        assert bins[5]["observed_freq"] is None
        assert bins[5]["eff_n"] == 0.0

    def test_bins_are_weighted_not_counted(self) -> None:
        """A heavy row moves the bin's means; an unweighted mean would not."""
        heavy = bin_statistics([0.11, 0.19], [0.0, 1.0], [1.0, 9.0])[1]
        assert heavy["observed_freq"] == pytest.approx(0.9)
        assert heavy["forecast_mean"] == pytest.approx((0.11 + 9 * 0.19) / 10)
        assert heavy["n"] == 2
        # Kish: (Σw)²/Σw² = 100/82 — two rows, but not two rows' worth.
        assert heavy["eff_n"] == pytest.approx(100 / 82, rel=1e-6)

    def test_bins_agree_with_reliability_pooled_sql(self, tmp_path: Path) -> None:
        """The producer and the SQL must be the same ten bins.

        Same table, same weights, same fold at 1.0 — checked through
        DuckDB rather than by restating the arithmetic, so a change to
        either side has to break this test to diverge.
        """
        duckdb = pytest.importorskip("duckdb")
        corpus = tmp_path / "small.parquet"
        rows = _corpus_rows(gauge=False, n_events=8)
        pq.write_table(_corpus_table(rows, gauge=False), corpus)

        sql = (REPO_ROOT / "sql" / "reliability_pooled.sql").read_text().replace(
            "{corpus}", f"'{corpus}'",
        )
        expected = {
            (int(r[0]), int(r[1])): {
                "n": int(r[2]), "eff_n": float(r[4]),
                "forecast_mean": float(r[5]), "observed_freq": float(r[6]),
            }
            for r in duckdb.connect().execute(sql).fetchall()
        }
        assert expected, "the SQL returned nothing to compare against"

        by_lead: dict[int, list[dict]] = {}
        for row in rows:
            by_lead.setdefault(int(row["lead_min"]), []).append(row)
        seen = 0
        for lead, lead_rows in by_lead.items():
            bins = bin_statistics(
                [r["raw_prob"] for r in lead_rows],
                [r["outcome"] for r in lead_rows],
                [r["sample_weight"] for r in lead_rows],
            )
            for index, binned in enumerate(bins):
                want = expected.get((lead, index))
                if want is None:
                    assert binned["n"] == 0
                    continue
                seen += 1
                assert binned["n"] == want["n"]
                for key in ("eff_n", "forecast_mean", "observed_freq"):
                    # The producer rounds to six places for the wire; the
                    # comparison is against the SQL's answer rounded the
                    # same way, so "agree" means agree exactly.
                    assert binned[key] == pytest.approx(
                        round(want[key], 6), rel=1e-12, abs=1e-12,
                    )
        assert seen == len(expected)


def _noise_rows(base_rate_by_month: dict[str, float], per_month: int = 900) -> list[dict]:
    """A corpus where ``raw_prob`` carries NO information about the outcome.

    The probability is deterministic pseudo-noise; the outcome is
    different pseudo-noise whose base rate depends only on the month. A
    calibration fitted on these rows can do exactly one useful thing —
    learn the pooled base rate — so:

    - fitted and applied IN SAMPLE it lands every row on that base rate,
      and the bin's forecast mean equals its observed frequency by
      construction. A perfect diagonal that measures nothing.
    - fitted out-of-fold it lands each month's rows on the OTHER month's
      base rate, which is not that month's frequency. The diagram is
      visibly off the diagonal, which is the truth.

    That gap is the whole of defect (1), in a table small enough to check.
    """
    rows: list[dict] = []
    for month_index, (month, base) in enumerate(sorted(base_rate_by_month.items())):
        year, mm = (int(x) for x in month.split("-"))
        for i in range(per_month):
            # Two independent-looking deterministic sequences.
            prob = ((i * 37 + month_index * 11) % 1000) / 1000.0
            outcome = 1 if ((i * 61 + 7) % 1000) / 1000.0 < base else 0
            stamp = datetime(year, mm, 1 + (i % 27), tzinfo=UTC).isoformat()
            rows.append({
                "event_time": stamp, "point_id": "06180",
                "lat": 55.0, "lon": 10.0, "region": "Denmark",
                "lead_min": 30, "raw_prob": prob, "outcome": outcome,
                "sample_weight": 1.0, "frame_age_min": 15.0,
                "threshold_mm_h": 0.5,
            })
    return rows


def _diagonal_gap(curve: dict) -> float:
    """The largest |forecast_mean − observed_freq| over the populated bins."""
    return max(
        abs(b["forecast_mean"] - b["observed_freq"])
        for b in curve["bins"]
        if b["forecast_mean"] is not None and b["n"] >= 50
    )


class TestOutOfSampleRadarReliability:
    """Defect (1): the radar diagram must not grade a fit on its own data."""

    @pytest.fixture
    def noise_corpus(self, tmp_path: Path) -> Path:
        path = tmp_path / "noise.parquet"
        rows = _noise_rows({"2026-03": 0.2, "2026-04": 0.6})
        pq.write_table(_corpus_table(rows, gauge=False), path)
        return path

    def test_in_sample_curves_would_draw_a_perfect_diagonal(
        self, noise_corpus: Path,
    ) -> None:
        """The bug, pinned: this is what the first real report did.

        Fit on the corpus, apply to the corpus, bin — and every bin lands
        on the diagonal no matter how worthless the forecast is.
        """
        from dmi_nowcast_core.calibrate import fit_isotonic_weighted
        from dmi_nowcast_core.quality_report import reliability_from_corpus

        rows = _noise_rows({"2026-03": 0.2, "2026-04": 0.6})
        in_sample = fit_isotonic_weighted(
            np.array([r["raw_prob"] for r in rows]),
            np.array([r["outcome"] for r in rows], dtype=float),
            np.ones(len(rows)),
        )
        served = reliability_from_corpus(
            noise_corpus, curves={30: in_sample},
            inputs=QualityInputs(served_leads=(30,)), calibration="served",
        )
        assert served["mode"] == "served"
        # …to three decimals, exactly the symptom that was reported.
        assert _diagonal_gap(served["curves"][0]) < 1e-3

    def test_cross_validation_refuses_to_flatter_the_fit(
        self, noise_corpus: Path,
    ) -> None:
        from dmi_nowcast_core.quality_report import reliability_from_corpus

        cv = reliability_from_corpus(
            noise_corpus, inputs=QualityInputs(served_leads=(30,)),
            calibration="cv",
        )
        assert cv["mode"] == "cv"
        assert cv["fold"] == "month"
        assert cv["cv_folds"] == 2
        # Each month is graded by the other month's base rate, which is
        # 0.4 away. A diagonal here would mean the CV leaked.
        assert _diagonal_gap(cv["curves"][0]) > 0.3

    def test_the_report_uses_cv_for_radar_and_served_curves_for_gauges(
        self, full_inputs: QualityInputs,
    ) -> None:
        """Two truths, two different and correctly-labelled claims."""
        report = build_quality_report(full_inputs)
        assert report["methods"]["reliability_probability"] == (
            "radar: calibrated out-of-sample, leave-one-month-out CV; "
            "gauges: the served curves against gauge truth"
        )

    def test_every_curve_carries_the_paired_raw_brier(
        self, full_inputs: QualityInputs,
    ) -> None:
        report = build_quality_report(full_inputs)
        for truth in ("radar", "gauge"):
            for curve in report["reliability"][truth]:
                assert isinstance(curve["brier_raw"], float)
                assert 0.0 <= curve["brier_raw"] <= 1.0
        improvement = report["methods"]["reliability_brier_improvement"]
        assert "raw →" in improvement and "out-of-sample calibrated" in improvement

    def test_a_single_month_falls_back_to_day_folds(self, tmp_path: Path) -> None:
        """A short corpus still gets cross-validated, just more finely."""
        from dmi_nowcast_core.quality_report import reliability_from_corpus

        path = tmp_path / "short.parquet"
        # Twelve events eight hours apart: four days inside one month.
        rows = _corpus_rows(gauge=False, n_events=12, step_hours=8)
        pq.write_table(_corpus_table(rows, gauge=False), path)
        out = reliability_from_corpus(path, calibration="cv")
        assert out["fold"] == "day"
        assert out["mode"] == "cv"

    def test_one_fold_means_no_cross_validation_and_says_so(
        self, tmp_path: Path,
    ) -> None:
        """Raw, honestly labelled, rather than a fit grading itself."""
        from dmi_nowcast_core.quality_report import reliability_from_corpus

        path = tmp_path / "oneday.parquet"
        rows = [
            {**r, "event_time": datetime(2026, 3, 1, tzinfo=UTC).isoformat()}
            for r in _corpus_rows(gauge=False, n_events=12, step_hours=1)
        ]
        pq.write_table(_corpus_table(rows, gauge=False), path)
        out = reliability_from_corpus(path, calibration="cv")
        assert out["mode"] == "raw"
        assert out["cv_folds"] == 0 and out["fold"] is None

        report = build_quality_report(QualityInputs(radar_corpus=path, now=NOW))
        assert report["methods"]["reliability_probability"].startswith(
            "radar: the raw ensemble exceedance fraction",
        )


class TestHeadlineBinRule:
    """Defect (2): the card must not go null because 0.7 is never said."""

    @staticmethod
    def _curve(counts: list[int]) -> dict:
        return {
            "lead_min": 30,
            "bins": [
                {
                    "lo": round(k / 10, 6), "hi": round((k + 1) / 10, 6),
                    "forecast_mean": None if n == 0 else round(k / 10 + 0.05, 3),
                    "observed_freq": None if n == 0 else round(k / 10 + 0.02, 3),
                    "n": n, "eff_n": float(n),
                }
                for k, n in enumerate(counts)
            ],
        }

    def test_it_takes_the_highest_bin_over_the_threshold(self) -> None:
        from dmi_nowcast_core.quality_report import _headline_bin

        # Calibrated probability tops out in bin 5 — exactly the shape of
        # the real corpus, where bins 7-9 are empty.
        curve = self._curve([9000, 4000, 2000, 900, 400, 250, 0, 0, 0, 0])
        chosen = _headline_bin(curve, QualityInputs())
        assert [chosen["lo"], chosen["hi"]] == [0.5, 0.6]

    def test_a_thin_bin_above_the_top_one_is_not_chosen(self) -> None:
        from dmi_nowcast_core.quality_report import _headline_bin

        curve = self._curve([9000, 4000, 2000, 900, 400, 250, 12, 0, 0, 0])
        chosen = _headline_bin(curve, QualityInputs())
        assert [chosen["lo"], chosen["hi"]] == [0.5, 0.6]
        assert chosen["n"] == 250

    def test_the_fallback_is_the_fullest_bin_above_a_third(self) -> None:
        from dmi_nowcast_core.quality_report import _headline_bin

        # Nothing clears n >= 200 anywhere, so the primary rule is out.
        curve = self._curve([190, 180, 190, 150, 40, 12, 0, 0, 0, 0])
        chosen = _headline_bin(curve, QualityInputs())
        assert [chosen["lo"], chosen["hi"]] == [0.3, 0.4]
        assert chosen["n"] == 150

    def test_nothing_confident_enough_is_no_sentence_at_all(self) -> None:
        from dmi_nowcast_core.quality_report import _headline_bin

        # Nothing over the threshold, and nothing at all above 0.3:
        # there is no "when we say X %" worth making.
        curve = self._curve([190, 100, 190, 0, 0, 0, 0, 0, 0, 0])
        assert _headline_bin(curve, QualityInputs()) is None

    def test_the_card_survives_a_corpus_that_never_says_seventy_percent(
        self, tmp_path: Path,
    ) -> None:
        """The live symptom, end to end.

        Curves that cap every lead at 0.55 empty bins 6-9 completely. The
        old rule read bin 7 and returned null; the new one reads the
        highest bin that exists.
        """
        curves = tmp_path / "shrunk.json"
        curves.write_text(json.dumps({
            "curves": {
                str(lead): {"raw_breakpoints": [0.0, 1.0],
                            "calibrated_values": [0.0, 0.55]}
                for lead in LEADS
            },
        }))
        report = build_quality_report(QualityInputs(
            station_corpus=write_station_corpus(tmp_path / "station.parquet"),
            national_curves=curves, now=NOW,
        ))
        headline = report["headline"]["reliability"]["gauge"]
        assert headline is not None
        curve = next(
            c for c in report["reliability"]["gauge"]
            if c["lead_min"] == headline["lead_min"]
        )
        assert curve["bins"][7]["n"] == 0      # the old rule's bin: empty
        assert headline["bin"][1] <= 0.6       # …and we quoted a real one
        assert headline["n"] > 0


class TestGaugeCurves:
    def test_the_gauge_diagram_uses_the_served_curves(
        self, tmp_path: Path,
    ) -> None:
        """Legitimate: the fit never saw ``gauge_outcome``.

        With curves the probabilities shrink and the top bin empties;
        without them the raw values stay where the corpus put them.
        """
        from dmi_nowcast_core.quality_report import reliability_from_corpus

        corpus = write_station_corpus(tmp_path / "station.parquet")
        raw = reliability_from_corpus(
            corpus, outcome_column="gauge_outcome", calibration="raw",
        )
        served = reliability_from_corpus(
            corpus, outcome_column="gauge_outcome",
            curves=write_and_load_curves(tmp_path / "c.json"),
            calibration="served",
        )
        raw_60 = next(c for c in raw["curves"] if c["lead_min"] == 60)
        cal_60 = next(c for c in served["curves"] if c["lead_min"] == 60)
        assert raw_60["bins"][9]["n"] > 0
        # The curve caps lead 60 at 0.7, so the top bin must empty out.
        assert cal_60["bins"][9]["n"] == 0
        assert cal_60["brier"] != raw_60["brier"]
        assert served["mode"] == "served"

    def test_the_served_leads_are_the_leads_with_curves(
        self, tmp_path: Path,
    ) -> None:
        curves = tmp_path / "partial.json"
        curves.write_text(json.dumps({
            "curves": {
                "20": {"raw_breakpoints": [0.0, 1.0],
                       "calibrated_values": [0.0, 1.0]},
                "30": {"raw_breakpoints": [0.0, 1.0],
                       "calibrated_values": [0.0, 1.0]},
            },
        }))
        report = build_quality_report(QualityInputs(
            radar_corpus=write_radar_corpus(tmp_path / "radar.parquet"),
            national_curves=curves, now=NOW,
        ))
        assert [c["lead_min"] for c in report["reliability"]["radar"]] == [20, 30]


# ---------------------------------------------------------------------------
# Decision rows: the dedupe, the stations, the events
# ---------------------------------------------------------------------------


class TestDecisions:
    def test_live_rows_win_over_replay_rows_for_the_same_frame(
        self, tmp_path: Path,
    ) -> None:
        """Replay and live cover the same day; the live decision is the record.

        The fixtures make the two distinguishable by their
        ``forecast_now_mm_h``: the live rows claim rain half a millimetre
        harder, so if the replay row survived the dedupe the raining-now
        agreement would come out different.
        """
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        write_gauge_store(corpus_dir)
        write_points(corpus_dir)
        write_catalogue(corpus_dir)

        # Replay says "no rain forecast anywhere" for the same keys the
        # live rows cover; live says the truth.
        blind = [
            {**row, "forecast_now_mm_h": 0.0, "observed_mm_h": 0.0}
            for row in decision_rows(live=False)
        ]
        write_replay(tmp_path / "replay", blind)
        write_live_eval(corpus_dir)

        both = build_quality_report(QualityInputs(
            replay_dir=tmp_path / "replay", corpus_dir=corpus_dir, now=NOW,
        ))
        replay_only = build_quality_report(QualityInputs(
            replay_dir=tmp_path / "replay", now=NOW,
            corpus_dir=corpus_dir,
        ))
        # Sanity: the same number of decision rows either way, because the
        # dedupe collapsed the two copies rather than concatenating them.
        assert both["headline"]["warnings"]["warnings"] == len(WARNINGS)
        assert replay_only["headline"]["warnings"]["warnings"] == len(WARNINGS)
        # …but the live forecast column is the one that got scored.
        assert both["raining_now"]["pod"] > 0.0

    def test_dedupe_does_not_double_count_warnings(
        self, full_inputs: QualityInputs,
    ) -> None:
        report = build_quality_report(full_inputs)
        # Both the replay parquet and the live parquet carry every warning.
        # Without the dedupe there would be twice as many.
        assert report["headline"]["warnings"]["warnings"] == len(WARNINGS)

    def test_a_station_under_three_warnings_reports_null_rates(
        self, full_inputs: QualityInputs,
    ) -> None:
        stations = build_quality_report(full_inputs)["stations"]["features"]
        by_id = {f["properties"]["station_id"]: f["properties"] for f in stations}
        assert by_id["06180"]["warnings"] == 4
        assert by_id["06180"]["warn_pod"] is not None
        assert by_id["06180"]["warn_far"] is not None
        # Two warnings is not a rate.
        assert by_id["06181"]["warnings"] == 2
        assert by_id["06181"]["warn_pod"] is None
        assert by_id["06181"]["warn_far"] is None
        # …but it still has a Brier score from the corpus, which is what
        # the map's colour falls back to.
        assert by_id["06181"]["brier_gauge"] is not None
        assert by_id["06182"]["warnings"] == 0
        assert by_id["06182"]["n_events"] > 0

    def test_stations_carry_names_and_coordinates(
        self, full_inputs: QualityInputs,
    ) -> None:
        features = build_quality_report(full_inputs)["stations"]["features"]
        assert len(features) == len(STATIONS)
        first = next(f for f in features if f["properties"]["station_id"] == "06180")
        assert first["properties"]["name"] == "Køge"
        assert first["properties"]["kind"] == "Synop"
        lon, lat = first["geometry"]["coordinates"]
        assert (lon, lat) == (10.0, 55.0)

    def test_events_are_newest_first_and_capped(
        self, full_inputs: QualityInputs,
    ) -> None:
        events = build_quality_report(full_inputs)["events"]
        assert 0 < len(events) <= 20
        stamps = [e["warned_at_utc"] for e in events]
        assert stamps == sorted(stamps, reverse=True)
        assert {e["outcome"] for e in events} <= {"hit", "false_alarm"}
        false_alarm = next(e for e in events if e["outcome"] == "false_alarm")
        assert false_alarm["gauge_onset_utc"] is None
        assert false_alarm["lead_error_min"] is None
        hit = next(e for e in events if e["outcome"] == "hit")
        assert hit["gauge_onset_utc"] is not None
        assert hit["lead_error_min"] is not None
        assert hit["name"] in ("Køge", "Aarhus Syd", "Skagen Fyr")

    def test_events_are_capped_at_twenty(self, tmp_path: Path) -> None:
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        write_gauge_store(corpus_dir)
        write_points(corpus_dir)
        # 40 warnings at 06180, one per frame across the dry afternoon.
        rows = []
        stamp = DAY + timedelta(hours=8)
        for _ in range(40):
            rows.append(_decision(
                stamp, "06180", action="notify", eta=20.0, p_rain=0.7,
                observed=0.0, forecast=0.0,
            ))
            stamp += timedelta(minutes=10)
        write_replay(tmp_path / "replay", rows)
        report = build_quality_report(QualityInputs(
            replay_dir=tmp_path / "replay", corpus_dir=corpus_dir, now=NOW,
        ))
        assert len(report["events"]) == 20
        assert validate_report(report) == []


class TestOnsetsOutsideCoverage:
    """Defect (4): a backfilled archive is not a list of missed warnings.

    The served report had five warnings, two hits and 2 088 misses, POD
    0.001 — the misses were gauge onsets going back to the December
    backfill, scored against decision rows that existed only for that day.
    """

    @pytest.fixture
    def corpus_dir(self, tmp_path: Path) -> Path:
        out = tmp_path / "corpus"
        out.mkdir()
        write_gauge_store(out, with_archive=True)
        write_points(out)
        write_catalogue(out)
        return out

    def _report(self, tmp_path: Path, corpus_dir: Path) -> dict:
        write_replay(tmp_path / "replay")
        return build_quality_report(QualityInputs(
            replay_dir=tmp_path / "replay", corpus_dir=corpus_dir, now=NOW,
        ))

    def test_the_archive_onsets_are_found_and_then_left_unscored(
        self, tmp_path: Path, corpus_dir: Path,
    ) -> None:
        warnings = self._report(tmp_path, corpus_dir)["headline"]["warnings"]
        # Found — the onset rule sees every one of them…
        assert warnings["uncovered_onsets"] == len(ARCHIVE_ONSETS)
        # …and not one reached the misses, where the served report put
        # 2 088 of them.
        assert warnings["misses"] == 1
        assert warnings["hits"] == 4
        assert warnings["pod"] == pytest.approx(0.8)

    def test_an_onset_inside_coverage_with_no_warning_is_still_a_miss(
        self, tmp_path: Path, corpus_dir: Path,
    ) -> None:
        """The control: the fix must not stop counting the real misses.

        06182 never warns, and rain starts there mid-morning while the
        service is demonstrably watching — decision rows either side, ten
        minutes apart. That is a miss and must stay one.
        """
        warnings = self._report(tmp_path, corpus_dir)["headline"]["warnings"]
        assert warnings["misses"] == 1
        assert warnings["pod"] < 1.0

    def test_a_three_hour_hole_in_the_decisions_does_not_bridge(
        self, tmp_path: Path, corpus_dir: Path,
    ) -> None:
        """The same onset, scored or not, purely on whether anyone watched.

        With decisions all day, the 09:00 rain at 06182 is a miss (the
        test above). Drop three hours of decisions around it — an outage,
        a restart, a resumed replay skipping a day — and the very same
        onset is uncovered instead. Nothing about the weather changed;
        what changed is whether the service was there to see it.
        """
        hole = (DAY + timedelta(hours=8), DAY + timedelta(hours=11))
        rows = [
            row for row in decision_rows(live=False)
            if not (hole[0] <= row["generated_at"] < hole[1])
        ]
        write_replay(tmp_path / "replay", rows)
        warnings = build_quality_report(QualityInputs(
            replay_dir=tmp_path / "replay", corpus_dir=corpus_dir, now=NOW,
        ))["headline"]["warnings"]
        assert warnings["misses"] == 0
        assert warnings["uncovered_onsets"] == len(ARCHIVE_ONSETS) + 1

    def test_methods_states_the_coverage_rule(
        self, tmp_path: Path, corpus_dir: Path,
    ) -> None:
        methods = self._report(tmp_path, corpus_dir)["methods"]
        assert "20 min apart" in methods["coverage_rule"]
        assert "not watching" in methods["coverage_rule"]


class TestFreshWarningsArePending:
    """Defect (3): a promise is not graded before it comes due.

    The gauge fixture stops reporting at the end of the fixture day. A
    warning sent minutes before that edge promises rain over the next
    forty minutes, and the gauge has said nothing about them — calling it
    a false alarm measures only how recently the report was built.
    """

    @pytest.fixture
    def corpus_dir(self, tmp_path: Path) -> Path:
        out = tmp_path / "corpus"
        out.mkdir()
        write_gauge_store(out)
        write_points(out)
        write_catalogue(out)
        return out

    #: The gauge store's last slot is exactly here.
    LAST_SLOT = DAY + timedelta(days=1)

    def _report(self, tmp_path: Path, corpus_dir: Path, rows: list[dict]) -> dict:
        write_replay(tmp_path / "replay", decision_rows(live=False) + rows)
        return build_quality_report(QualityInputs(
            replay_dir=tmp_path / "replay", corpus_dir=corpus_dir, now=NOW,
        ))

    def test_a_warning_past_the_gauges_last_word_is_not_a_false_alarm(
        self, tmp_path: Path, corpus_dir: Path,
    ) -> None:
        # Sent ten minutes before the last known slot: its window runs to
        # +30 min past it, and no gauge data covers that.
        fresh = self.LAST_SLOT - timedelta(minutes=10)
        report = self._report(tmp_path, corpus_dir, [
            _decision(fresh, "06182", action="notify", eta=20.0, p_rain=0.85,
                      observed=0.0, forecast=2.0),
        ])
        warnings = report["headline"]["warnings"]
        assert warnings["pending"] == 1
        assert warnings["n_sent"] == len(WARNINGS) + 1
        # Not counted anywhere it could look like a failure.
        assert warnings["warnings"] == len(WARNINGS)
        assert warnings["hits"] + warnings["false_alarms"] == warnings["warnings"]
        assert warnings["false_alarms"] == 2

    def test_a_pending_warning_never_reaches_the_events_table(
        self, tmp_path: Path, corpus_dir: Path,
    ) -> None:
        """The schema's outcome enum is hit|false_alarm — and stays that way."""
        fresh = self.LAST_SLOT - timedelta(minutes=10)
        report = self._report(tmp_path, corpus_dir, [
            _decision(fresh, "06182", action="notify", eta=20.0, p_rain=0.85,
                      observed=0.0, forecast=2.0),
        ])
        stamps = {e["warned_at_utc"] for e in report["events"]}
        assert _iso_z(fresh) not in stamps
        assert {e["outcome"] for e in report["events"]} <= {"hit", "false_alarm"}
        assert validate_report(report) == []

    def test_a_warning_whose_window_closed_is_still_graded(
        self, tmp_path: Path, corpus_dir: Path,
    ) -> None:
        """The fix must not swallow real false alarms near the edge."""
        settled = self.LAST_SLOT - timedelta(minutes=120)
        report = self._report(tmp_path, corpus_dir, [
            _decision(settled, "06182", action="notify", eta=20.0, p_rain=0.85,
                      observed=0.0, forecast=2.0),
        ])
        warnings = report["headline"]["warnings"]
        assert warnings["pending"] == 0
        assert warnings["false_alarms"] == 3
        assert _iso_z(settled) in {e["warned_at_utc"] for e in report["events"]}

    def test_the_station_scores_exclude_pending_too(
        self, tmp_path: Path, corpus_dir: Path,
    ) -> None:
        fresh = self.LAST_SLOT - timedelta(minutes=10)
        report = self._report(tmp_path, corpus_dir, [
            _decision(fresh - timedelta(minutes=10 * k), "06182",
                      action="notify", eta=20.0, p_rain=0.85,
                      observed=0.0, forecast=2.0)
            for k in range(3)
        ])
        props = {
            f["properties"]["station_id"]: f["properties"]
            for f in report["stations"]["features"]
        }["06182"]
        # All three are still open, so the station has no scored warnings
        # and therefore no rate to report — null, never 100 % false.
        assert props["warnings"] == 0
        assert props["warn_far"] is None


# ---------------------------------------------------------------------------
# The schema checker, and the client's own assertions
# ---------------------------------------------------------------------------


class TestSchemaChecker:
    def test_it_catches_a_section_written_as_a_half_block(
        self, full_inputs: QualityInputs,
    ) -> None:
        report = build_quality_report(full_inputs)
        del report["headline"]["warnings"]["hits"]
        problems = validate_report(report)
        assert any("headline.warnings.hits" in p for p in problems)

    def test_it_catches_a_wrong_type_and_a_bad_timestamp(
        self, full_inputs: QualityInputs,
    ) -> None:
        report = build_quality_report(full_inputs)
        report["raining_now"]["pod"] = "quite good"
        report["windows"]["radar"]["from"] = "yesterday"
        problems = validate_report(report)
        assert any("raining_now.pod" in p for p in problems)
        assert any("windows.radar.from" in p for p in problems)

    def test_it_catches_events_out_of_order_and_a_bad_outcome(
        self, full_inputs: QualityInputs,
    ) -> None:
        report = build_quality_report(full_inputs)
        report["events"].reverse()
        report["events"][0]["outcome"] = "probably"
        problems = validate_report(report)
        assert any("newest first" in p for p in problems)
        assert any("outcome" in p for p in problems)

    def test_it_catches_a_curve_with_the_wrong_number_of_bins(
        self, full_inputs: QualityInputs,
    ) -> None:
        report = build_quality_report(full_inputs)
        report["reliability"]["radar"][0]["bins"] = []
        assert any("bins" in p for p in validate_report(report))

    def test_it_catches_a_missing_top_level_key(
        self, full_inputs: QualityInputs,
    ) -> None:
        report = build_quality_report(full_inputs)
        del report["methods"]
        assert any("methods: missing" in p for p in validate_report(report))


class TestFrontendLoaderAssertions:
    """The same checks ``frontend/src/lib/quality/load.test.ts`` makes.

    Mirrored in Python against a freshly built document rather than the
    committed fixture, so a producer that starts writing something the
    loader would drop fails here first — on the side that can still be
    fixed before a deploy.
    """

    @pytest.fixture
    def report(self, full_inputs: QualityInputs) -> dict:
        return build_quality_report(full_inputs)

    def test_it_parses_every_section(self, report: dict) -> None:
        assert report["schema_version"] == SCHEMA_VERSION
        assert report["windows"]["radar"] is not None
        assert report["windows"]["gauge"] is not None
        assert report["windows"]["live"] is not None
        assert report["headline"]["reliability"]["radar"] is not None
        assert report["headline"]["reliability"]["gauge"] is not None
        assert report["headline"]["warnings"] is not None
        assert report["headline"]["persistence_margin"] is not None
        assert report["reliability"]["radar"] is not None
        assert report["reliability"]["gauge"] is not None
        assert report["raining_now"] is not None
        assert report["stations"] is not None
        assert report["events"] is not None
        assert report["methods"] is not None

    def test_reliability_curves_are_ordered_by_lead(self, report: dict) -> None:
        for key in ("radar", "gauge"):
            leads = [c["lead_min"] for c in report["reliability"][key]]
            assert leads == sorted(leads)
            assert leads == list(LEADS)

    def test_every_curve_has_ten_bins_and_empty_ones_are_null(
        self, report: dict,
    ) -> None:
        for key in ("radar", "gauge"):
            for curve in report["reliability"][key]:
                assert len(curve["bins"]) == N_BINS
                for binned in curve["bins"]:
                    if binned["n"] == 0:
                        assert binned["forecast_mean"] is None
                        assert binned["observed_freq"] is None
                    else:
                        assert isinstance(binned["forecast_mean"], float)
                        assert isinstance(binned["observed_freq"], float)

    def test_the_station_colour_fallback_is_exercised(self, report: dict) -> None:
        without_pod = [
            f for f in report["stations"]["features"]
            if f["properties"]["warn_pod"] is None
        ]
        assert without_pod
        assert all(f["properties"]["brier_gauge"] is not None for f in without_pod)

    def test_a_false_alarm_keeps_its_legitimate_nulls(self, report: dict) -> None:
        false_alarm = next(
            e for e in report["events"] if e["outcome"] == "false_alarm"
        )
        assert false_alarm["gauge_onset_utc"] is None
        assert false_alarm["lead_error_min"] is None

    def test_methods_frame_age_is_a_pair_of_numbers(self, report: dict) -> None:
        ages = report["methods"]["frame_age_range_min"]
        assert len(ages) == 2
        assert all(isinstance(a, (int, float)) for a in ages)
        assert ages[0] <= ages[1]

    def test_the_document_round_trips_through_json(self, report: dict) -> None:
        """Whatever the builder produced has to survive ``json.dumps``.

        numpy scalars serialise nowhere, and a float32 that leaked out of
        pyarrow would fail here rather than at 03:30 on the VM.
        """
        again = json.loads(json.dumps(report))
        assert validate_report(again) == []
        assert again["headline"]["warnings"] == report["headline"]["warnings"]


# ---------------------------------------------------------------------------
# The markdown twin
# ---------------------------------------------------------------------------


class TestMarkdown:
    def test_it_renders_every_section(self, full_inputs: QualityInputs) -> None:
        text = render_markdown(build_quality_report(full_inputs))
        for heading in ("# How good are we?", "## Windows", "## Headline",
                        "## Reliability", "## Is it raining now?",
                        "## Stations", "## Recent warnings", "## Methods"):
            assert heading in text
        assert "measured days" in text
        assert "DMI radar composites (CC BY 4.0)" in text
        assert "Køge" in text

    def test_a_null_section_says_not_measured_rather_than_vanishing(self) -> None:
        text = render_markdown(build_quality_report(QualityInputs(now=NOW)))
        assert text.count("Not measured.") >= 4
        assert "not measured" in text
        assert "## Stations" in text
