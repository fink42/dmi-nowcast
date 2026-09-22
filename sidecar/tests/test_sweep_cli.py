"""The manual threshold fit's CLI (``scripts/sweep_thresholds.py``), S3.

The nightly fit reads ``push.probability_source`` and fits the table on
the probability the engine decides with (``quality_report._fit_options``).
The manual fit could not, until this work package: the flags did not
exist, so a hand-run refit — the one an operator does right after
installing a model rather than waiting for 03:40 — silently fitted the
percents on the curve-calibrated ``p_rain`` and served them against
``p_post``. Every horizon would then warn at the wrong percent, and
nothing in the output would say so.

Two claims, therefore:

1. The three flags land on :class:`SweepOptions` unchanged, and their
   absence leaves the shipped fit exactly as it was.
2. A real run over rows that carry ``p_post_<lead>`` is fitted on THOSE.
   The fixture makes the two answers incompatible on purpose: ``p_rain``
   is flat and never crosses any threshold in the grid, ``p_post`` is high
   at the onsets and low elsewhere. A fit on the wrong column produces no
   warnings at all.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import sweep_thresholds as cli  # noqa: E402  (after the sys.path edit)

from dmi_nowcast_core import postprocess as pp  # noqa: E402
from dmi_nowcast_core.metobs import Observation  # noqa: E402
from dmi_nowcast_core.station_store import StationObsStore  # noqa: E402
from dmi_nowcast_core.warning_score import (  # noqa: E402
    DEFAULT_PRODUCT_LEADS_MIN,
    decision_table,
)

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
    """Rain at the first station on the even days, 08:00 to 08:30."""
    if station != STATIONS[0] or day % 2:
        return False
    return 8 * 60 <= stamp.hour * 60 + stamp.minute <= 8 * 60 + 30


def _write_decisions(directory: Path, day: int) -> Path:
    """One frame per station per ten minutes, with the two columns disagreeing."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for minute in range(FIRST_FRAME_MIN, LAST_FRAME_MIN + 1, 10):
        radar_ts = _at(day, minute)
        for station in STATIONS:
            coming = _wet(station, day, radar_ts + timedelta(minutes=30))
            rows.append({
                "radar_ts": radar_ts,
                "generated_at": radar_ts,
                "station_id": station,
                "p_rain": 0.1,
                "action": "none",
                # Flat and below every threshold in the grid: a fit on this
                # column can never issue a warning.
                "p_rain_10": 0.1,
                "p_rain_30": 0.1,
                # The post-processed answer, as the engine would have
                # stored it: high where rain is coming.
                "p_post_10": 0.9 if coming else 0.1,
                "p_post_30": 0.9 if coming else 0.1,
                "eta_min": 25.0,
                "intensity_mm_h": 1.2,
                "observed_mm_h": 0.0,
                "forecast_now_mm_h": 0.0,
                "armed_after": True,
                "streak_after": 0,
            })
    table = decision_table(rows, LEADS)
    for field in pp.post_schema(LEADS):
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
    """A gauge store plus a ``decisions/`` tree carrying both columns."""
    _write_gauge(tmp_path / "corpus")
    for day in DAYS:
        _write_decisions(tmp_path / "decisions", day)
    return tmp_path


def _argv(corpus: Path, *extra: str) -> list[str]:
    return [
        "--decisions-dir", str(corpus / "decisions"),
        "--corpus-dir", str(corpus / "corpus"),
        "--leads", "10,30",
        "--thresholds", "40,50,60",
        "--min-warnings", "1",
        "--min-known-slots", "0",
        *extra,
    ]


# ---------------------------------------------------------------------------
# 1. the flags reach the options
# ---------------------------------------------------------------------------


class TestTheFlagsLandOnTheOptions:
    def _captured(self, monkeypatch, argv: list[str]):
        seen: dict = {}

        def fake_run_fit(options, log=None):
            seen["options"] = options
            # Nothing to sweep: the options are the subject here.
            raise cli.SweepError("stop after the options")

        monkeypatch.setattr(cli, "run_fit", fake_run_fit)
        rc = cli.main(argv)
        return rc, seen.get("options")

    def test_all_three_are_passed_through(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        model = tmp_path / "postprocess.json"
        model.write_text("{}\n")
        rc, options = self._captured(monkeypatch, [
            "--decisions-dir", str(tmp_path),
            "--corpus-dir", str(tmp_path),
            "--probability-column", "p_post_{lead}",
            "--postprocess-model", str(model),
            "--design-leads", "10,20,30,45,60",
        ])
        assert rc == 2  # the stubbed SweepError, not a parse failure
        assert options is not None
        assert options.probability_column == "p_post_{lead}"
        assert options.postprocess_model == model
        assert tuple(options.design_leads) == (10, 20, 30, 45, 60)

    def test_without_them_the_shipped_fit_is_unchanged(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        rc, options = self._captured(monkeypatch, [
            "--decisions-dir", str(tmp_path), "--corpus-dir", str(tmp_path),
        ])
        assert rc == 2
        assert options is not None
        assert options.probability_column is None
        assert options.postprocess_model is None
        assert tuple(options.design_leads) == tuple(DEFAULT_PRODUCT_LEADS_MIN)

    def test_a_model_without_a_column_is_refused(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Otherwise the model is never read and the fit is on p_rain."""
        model = tmp_path / "postprocess.json"
        model.write_text("{}\n")
        rc, options = self._captured(monkeypatch, [
            "--decisions-dir", str(tmp_path), "--corpus-dir", str(tmp_path),
            "--postprocess-model", str(model),
        ])
        assert rc == 2
        assert options is None  # the sweep never started

    def test_a_missing_model_file_is_refused(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        rc, options = self._captured(monkeypatch, [
            "--decisions-dir", str(tmp_path), "--corpus-dir", str(tmp_path),
            "--probability-column", "p_post_{lead}",
            "--postprocess-model", str(tmp_path / "nothing.json"),
        ])
        assert rc == 2
        assert options is None


# ---------------------------------------------------------------------------
# 2. a real run is fitted on the column it was told to fit on
# ---------------------------------------------------------------------------


class TestItFitsOnThePostProcessedColumn:
    def test_the_table_is_fitted_on_p_post_and_says_so(
        self, corpus: Path,
    ) -> None:
        out = corpus / "push_thresholds.json"
        rc = cli.main(_argv(
            corpus,
            "--probability-column", "p_post_{lead}",
            "--out-thresholds", str(out),
            "--out-json", str(corpus / "sweep.json"),
        ))
        assert rc == 0
        doc = json.loads(out.read_text())
        assert doc["objective"]["probability_column"] == "p_post_{lead}"
        payload = json.loads((corpus / "sweep.json").read_text())
        assert payload["settings"]["probability_column"] == "p_post_{lead}"
        # Stored values, so nothing had to be computed and nothing dropped.
        assert payload["settings"]["probability_rows"]["stored"] > 0
        assert payload["settings"]["probability_rows"]["computed"] == 0
        # The fit stands on warnings that only the p_post column can raise.
        assert any(
            (cell.get("warnings") or 0) > 0 for cell in payload["cells"]
        )
        assert any(
            entry.get("threshold_pct") is not None
            for entry in doc["leads"].values()
        )

    def test_the_same_rows_on_p_rain_raise_nothing(self, corpus: Path) -> None:
        """The control: the flat column cannot produce the table above."""
        out = corpus / "push_thresholds_curve.json"
        rc = cli.main(_argv(corpus, "--out-thresholds", str(out)))
        assert rc == 0
        doc = json.loads(out.read_text())
        assert doc["objective"]["probability_column"] == "p_rain_{lead}"
        assert all(
            entry.get("threshold_pct") is None
            for entry in doc["leads"].values()
        )
