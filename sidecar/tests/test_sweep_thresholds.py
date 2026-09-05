"""The (lead, threshold) sweep (``scripts/sweep_thresholds.py``).

Tested from the sidecar suite for the same reason
``test_replay_warnings.py`` is: the script imports ``push.engine`` for the
decision and the core library for the scoring, and this environment is the
only one that has both.

Fully offline and fully synthetic — no radar, no STEPS, no network. The
fixture is a hand-planted grid of probabilities over two stations and two
days, with a gauge store whose wet slots were chosen so that every count
below can be worked out on paper:

* **06180, day 1** — the probability crosses at 07:30 (lead 30) and, one
  frame later, at 07:40 (lead 10, because the 07:30 row's ``p_rain_10``
  is null and a null is skipped). The gauge starts raining at 08:00, so
  both warnings claim that onset: one HIT each.
* **06180, day 2** — one crossing at 08:00 and a gauge that stays dry all
  day: one FALSE ALARM at lead 30, nothing at lead 10.
* **06120, both days** — the probability never crosses, but the gauge
  starts raining at 08:30 on day 1: one MISS, at every lead.

So lead 30 at 40 % scores 2 warnings / 1 hit / 1 false alarm / 1 miss
(POD ½, FAR ½, CSI ⅓) and lead 10 at 40 % scores 1 / 1 / 0 / 1 (POD ½,
FAR 0, CSI ½). Every assertion below is one of those hand-worked numbers.

The rows also carry two traps: ``p_rain`` is 0.99 on every row and
``action`` is ``"none"`` on every row. A sweep that read either — the
rule's own probability instead of the per-lead column, or the stored
decision instead of a fresh replay — would produce far too many warnings
or none at all, never the two that are asserted.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import sweep_thresholds as sweep  # noqa: E402  (after the sys.path edit)

from dmi_nowcast_core.metobs import Observation  # noqa: E402
from dmi_nowcast_core.station_store import StationObsStore  # noqa: E402
from dmi_nowcast_core.warning_score import decision_table  # noqa: E402

STATION_A = "06180"
STATION_B = "06120"
STATIONS = (STATION_A, STATION_B)
LEADS = (10, 30)
#: 2026-06-01 and 2026-06-02, one month partition, no month seam to worry about.
DAYS = (1, 2)
#: Decision frames: 07:00–09:00 UTC every 10 minutes.
FIRST_FRAME_MIN = 7 * 60
LAST_FRAME_MIN = 9 * 60
#: Gauge slots: 06:00–10:00 UTC every 10 minutes, so every onset has its
#: three dry slots behind it and every warning window closes inside the
#: reported record (nothing is ever "pending").
FIRST_SLOT_MIN = 6 * 60
LAST_SLOT_MIN = 10 * 60


def _at(day: int, minute_of_day: int) -> datetime:
    return datetime(2026, 6, day, tzinfo=timezone.utc) + timedelta(
        minutes=minute_of_day
    )


def _hhmm(ts: datetime) -> str:
    return ts.strftime("%H:%M")


def _p_values(station: str, day: int, ts: datetime) -> tuple[float | None, float]:
    """``(p_rain_10, p_rain_30)`` for one frame of the fixture."""
    hhmm = _hhmm(ts)
    if station == STATION_A and day == 1:
        p_30 = 0.60 if hhmm in ("07:30", "07:40") else 0.10
        if hhmm == "07:30":
            p_10 = None  # the null the sweep must skip without firing
        elif hhmm == "07:40":
            p_10 = 0.60
        else:
            p_10 = 0.10
        return p_10, p_30
    if station == STATION_A and day == 2:
        return 0.10, (0.60 if hhmm == "08:00" else 0.10)
    return 0.10, 0.10


def _wet_slots(station: str, day: int) -> set[str]:
    """The gauge slot ends (HH:MM) this station reports as wet."""
    if station == STATION_A and day == 1:
        return {"08:00", "08:10"}
    if station == STATION_B and day == 1:
        return {"08:30"}
    return set()


def _decision_rows(day: int) -> list[dict]:
    rows: list[dict] = []
    for minute in range(FIRST_FRAME_MIN, LAST_FRAME_MIN + 1, 10):
        radar_ts = _at(day, minute)
        for station in STATIONS:
            p_10, p_30 = _p_values(station, day, radar_ts)
            rows.append({
                "radar_ts": radar_ts,
                # Zero frame age: the fixture's arithmetic is easier to
                # check when the send instant IS the radar instant.
                "generated_at": radar_ts,
                "station_id": station,
                # Traps: a sweep that read either of these would not
                # produce the hand-worked counts.
                "p_rain": 0.99,
                "action": "none",
                "p_rain_10": p_10,
                "p_rain_30": p_30,
                "eta_min": 25.0,
                "intensity_mm_h": 1.2,
                # Below Rules.raining_now_mm_h, so nothing is silenced.
                "observed_mm_h": 0.0,
                "forecast_now_mm_h": 0.0,
                "armed_after": True,
                "streak_after": 0,
            })
    return rows


def _write_decisions(directory: Path, rows: list[dict], name: str) -> Path:
    import pyarrow.parquet as pq

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    pq.write_table(decision_table(rows, LEADS), path)
    return path


def _write_gauge(corpus_dir: Path) -> None:
    store = StationObsStore(corpus_dir)
    observations: list[Observation] = []
    for day in DAYS:
        for station in STATIONS:
            wet = _wet_slots(station, day)
            for minute in range(FIRST_SLOT_MIN, LAST_SLOT_MIN + 1, 10):
                stamp = _at(day, minute)
                observations.append(Observation(
                    station_id=station,
                    observed_utc=stamp,
                    parameter_id="precip_past10min",
                    value=0.5 if _hhmm(stamp) in wet else 0.0,
                ))
    store.append(observations)


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """A gauge store plus a ``decisions/`` tree, both hand-planted."""
    _write_gauge(tmp_path / "corpus")
    decisions = tmp_path / "decisions"
    for day in DAYS:
        _write_decisions(decisions, _decision_rows(day), f"2026-06-{day:02d}.parquet")
    return tmp_path


def _run(tmp_path: Path, corpus_root: Path, *extra: str) -> dict:
    """Run the CLI over the fixture and return the JSON document."""
    out_json = tmp_path / "sweep.json"
    argv = [
        "--decisions-dir", str(corpus_root / "decisions"),
        "--corpus-dir", str(corpus_root / "corpus"),
        "--leads", "10,30",
        "--thresholds", "40,50,70",
        "--out-json", str(out_json),
        *extra,
    ]
    assert sweep.main(argv) == 0
    return json.loads(out_json.read_text())


def _cell(payload: dict, lead: int, threshold: int) -> dict:
    for cell in payload["cells"]:
        if cell["lead_min"] == lead and cell["threshold_pct"] == threshold:
            return cell
    raise AssertionError(f"no cell for lead {lead} at {threshold}%")


# ---------------------------------------------------------------------------
# Grid parsing
# ---------------------------------------------------------------------------


def test_threshold_range_is_inclusive_at_both_ends() -> None:
    assert sweep.parse_thresholds("20:80:5") == (
        20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
    )
    assert sweep.parse_thresholds("40:60:10") == (40, 50, 60)


def test_threshold_spec_accepts_a_list_and_a_default_step() -> None:
    assert sweep.parse_thresholds("30, 40 ,55") == (30, 40, 55)
    assert sweep.parse_thresholds("40:50") == (40, 45, 50)


@pytest.mark.parametrize("spec", ["", "0:80:5", "20:120:5", "20:80:0", "80:20:5"])
def test_bad_threshold_specs_are_rejected(spec: str) -> None:
    with pytest.raises(ValueError):
        sweep.parse_thresholds(spec)


def test_leads_parse_and_deduplicate() -> None:
    assert sweep.parse_leads("30,10,30") == (10, 30)
    assert sweep.parse_leads(None) == (10, 20, 30, 45, 60)
    with pytest.raises(ValueError):
        sweep.parse_leads("0")


# ---------------------------------------------------------------------------
# Loading and deduplication
# ---------------------------------------------------------------------------


def test_later_decisions_dir_wins_the_duplicate_key(tmp_path: Path) -> None:
    """Two directories, one shared key: the last one on the command line wins."""
    radar_ts = _at(1, 7 * 60)
    base = {
        "radar_ts": radar_ts,
        "generated_at": radar_ts,
        "station_id": STATION_A,
        "action": "none",
        "eta_min": 25.0,
    }
    _write_decisions(tmp_path / "old", [{**base, "p_rain_30": 0.10}], "a.parquet")
    _write_decisions(tmp_path / "new", [{**base, "p_rain_30": 0.90}], "b.parquet")

    rows, leads, counts = sweep.load_decisions(
        [tmp_path / "old", tmp_path / "new"], leads_min=(),
    )
    assert leads == LEADS
    assert counts["duplicates"] == 1
    assert len(rows) == 1
    assert rows[0]["p_rain_30"] == pytest.approx(0.90)

    reversed_rows, _, _ = sweep.load_decisions(
        [tmp_path / "new", tmp_path / "old"], leads_min=(),
    )
    assert reversed_rows[0]["p_rain_30"] == pytest.approx(0.10)


def test_a_non_decision_parquet_under_the_tree_is_skipped(tmp_path: Path) -> None:
    """``events.parquet`` sits beside ``decisions/`` in a replay directory."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = tmp_path / "replay"
    directory.mkdir()
    pq.write_table(pa.table({"nonsense": [1, 2, 3]}), directory / "events.parquet")
    radar_ts = _at(1, 7 * 60)
    _write_decisions(directory, [{
        "radar_ts": radar_ts, "generated_at": radar_ts,
        "station_id": STATION_A, "action": "none", "p_rain_30": 0.5,
    }], "day.parquet")

    rows, _, counts = sweep.load_decisions([directory], leads_min=())
    assert counts["skipped"] == 1
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# The replay itself
# ---------------------------------------------------------------------------


def _track(values: list[float | None], *, run: int = 0) -> list[tuple]:
    """A one-lead track of ``values`` at 10-minute spacing."""
    out = []
    for i, value in enumerate(values):
        stamp = _at(1, 7 * 60 + 10 * i)
        out.append((stamp, stamp, 25.0, 1.2, 0.0, 0.0, run, (value,)))
    return out


def test_a_null_probability_neither_warns_nor_breaks_the_streak() -> None:
    """A null is "no reading", not "dry": it must not reset the persistence."""
    warnings = sweep.replay_station(
        _track([0.10, 0.60, None, 0.60]), 0, 40,
        persistence_obs=2, rearm_after_min=60,
    )
    assert [sent for sent, _ in warnings] == [_at(1, 7 * 60 + 30)]

    # Same rows with the null replaced by a genuinely low reading: the
    # streak breaks and nothing fires at all.
    assert sweep.replay_station(
        _track([0.10, 0.60, 0.05, 0.60]), 0, 40,
        persistence_obs=2, rearm_after_min=60,
    ) == []


def test_a_coverage_gap_restarts_the_subscription_armed() -> None:
    """A new run starts armed, so the second crossing fires despite the first."""
    first = _track([0.60], run=0)
    second = _track([0.60], run=1)
    second = [(t + timedelta(hours=6), g + timedelta(hours=6), *rest)
              for t, g, *rest in second]
    warnings = sweep.replay_station(
        first + second, 0, 40, persistence_obs=1, rearm_after_min=60,
    )
    assert len(warnings) == 2


def test_already_raining_silences_the_notification() -> None:
    """An observed rate at or over the threshold consumes the arm quietly."""
    track = _track([0.60])
    wet = [(t, g, eta, i, 2.0, f, run, p) for t, g, eta, i, _, f, run, p in track]
    assert sweep.replay_station(
        wet, 0, 40, persistence_obs=1, rearm_after_min=60,
    ) == []


# ---------------------------------------------------------------------------
# End to end: the hand-worked counts
# ---------------------------------------------------------------------------


def test_lead_30_at_40_percent_scores_the_planted_events(
    tmp_path: Path, corpus: Path,
) -> None:
    payload = _run(tmp_path, corpus)
    cell = _cell(payload, 30, 40)
    assert cell["n_sent"] == 2
    assert cell["hits"] == 1
    assert cell["false_alarms"] == 1
    assert cell["misses"] == 1
    assert cell["pending"] == 0
    assert cell["uncovered_onsets"] == 0
    assert cell["n_onsets"] == 2
    assert cell["pod"] == pytest.approx(0.5)
    assert cell["far"] == pytest.approx(0.5)
    assert cell["csi"] == pytest.approx(1 / 3)
    # eta 25 min against a 30-minute wait: the warning was 5 minutes early.
    assert cell["lead_error_min"]["p50"] == pytest.approx(-5.0)


def test_lead_10_skips_the_null_row_and_fires_one_frame_later(
    tmp_path: Path, corpus: Path,
) -> None:
    payload = _run(tmp_path, corpus)
    cell = _cell(payload, 10, 40)
    assert cell["n_sent"] == 1
    assert cell["hits"] == 1
    assert cell["false_alarms"] == 0
    assert cell["misses"] == 1
    assert cell["far"] == pytest.approx(0.0)
    assert cell["csi"] == pytest.approx(0.5)


def test_a_threshold_above_every_probability_sends_nothing(
    tmp_path: Path, corpus: Path,
) -> None:
    payload = _run(tmp_path, corpus)
    cell = _cell(payload, 30, 70)
    assert cell["n_sent"] == 0
    assert cell["hits"] == 0
    assert cell["far"] is None
    assert cell["misses"] == 2
    # ...which is exactly the do-nothing floor.
    assert payload["do_nothing"]["30"]["misses"] == 2
    assert payload["do_nothing"]["30"]["csi"] == pytest.approx(0.0)


def test_the_stored_action_and_base_p_rain_are_ignored(
    tmp_path: Path, corpus: Path,
) -> None:
    """Every row says ``action="none"`` and ``p_rain=0.99``; neither is read."""
    payload = _run(tmp_path, corpus)
    assert payload["window"]["rows"] == 2 * 2 * 13
    assert _cell(payload, 30, 40)["n_sent"] == 2


def test_the_picks_prefer_the_higher_threshold_on_a_tie(
    tmp_path: Path, corpus: Path,
) -> None:
    """40 % and 50 % score identically here; the quieter rule wins."""
    payload = _run(tmp_path, corpus)
    assert payload["picks"]["30"]["max_csi"]["threshold_pct"] == 50
    assert payload["picks"]["10"]["max_csi"]["threshold_pct"] == 50


def test_the_far_cap_pick_reports_none_when_no_cell_clears_it(
    tmp_path: Path, corpus: Path,
) -> None:
    payload = _run(tmp_path, corpus)
    # Lead 30's best FAR on this grid is 0.5, over the 0.30 cap.
    assert payload["picks"]["30"]["max_pod_far_capped"] is None
    # Lead 10 sends one warning and it is a hit: FAR 0.
    assert payload["picks"]["10"]["max_pod_far_capped"]["threshold_pct"] == 50


def test_a_lead_with_no_column_in_the_rows_is_skipped(
    tmp_path: Path, corpus: Path,
) -> None:
    payload = _run(tmp_path, corpus, "--leads", "10,30,45")
    assert payload["leads"] == [10, 30]
    assert payload["settings"]["leads_requested"] == [10, 30, 45]


@pytest.mark.filterwarnings(
    # The pool forks so the loaded rows are shared copy-on-write instead of
    # pickled per cell. Python warns because *pytest* is multi-threaded, not
    # because the sweep is: the cells are pure functions of an inherited
    # payload and take no locks.
    "ignore:This process .* is multi-threaded:DeprecationWarning"
)
def test_workers_greater_than_one_gives_the_same_grid(
    tmp_path: Path, corpus: Path,
) -> None:
    serial = _run(tmp_path / "serial", corpus)
    parallel = _run(tmp_path / "parallel", corpus, "--workers", "2")
    assert serial["cells"] == parallel["cells"]
    assert serial["do_nothing"] == parallel["do_nothing"]


# ---------------------------------------------------------------------------
# Picking, on a hand-made grid
# ---------------------------------------------------------------------------


def _grid_cell(threshold: int, pod: float | None, far: float | None,
               csi: float | None) -> dict:
    return {
        "lead_min": 30, "threshold_pct": threshold,
        "pod": pod, "far": far, "csi": csi,
    }


def test_pick_max_csi_takes_the_best_and_breaks_ties_upward() -> None:
    cells = [
        _grid_cell(20, 0.9, 0.9, 0.10),
        _grid_cell(30, 0.6, 0.5, 0.40),
        _grid_cell(40, 0.4, 0.3, 0.40),
        _grid_cell(50, 0.2, 0.2, 0.18),
    ]
    assert sweep.pick_max_csi(cells)["threshold_pct"] == 40
    assert sweep.pick_max_csi([_grid_cell(20, None, None, None)]) is None


def test_pick_max_pod_under_far_respects_the_cap() -> None:
    cells = [
        _grid_cell(20, 0.90, 0.55, 0.30),   # too many false alarms
        _grid_cell(30, 0.55, 0.30, 0.35),   # exactly at the cap: eligible
        _grid_cell(40, 0.40, 0.20, 0.32),
        _grid_cell(50, 0.55, 0.10, 0.34),   # ties on POD, higher threshold
        _grid_cell(60, 0.10, None, 0.05),   # sent nothing: no FAR, not eligible
    ]
    assert sweep.pick_max_pod_under_far(cells, 0.30)["threshold_pct"] == 50
    assert sweep.pick_max_pod_under_far(cells, 0.05) is None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_markdown_renders_and_names_the_shipping_rule(
    tmp_path: Path, corpus: Path,
) -> None:
    out_md = tmp_path / "sweep.md"
    out_csv = tmp_path / "sweep.csv"
    payload = _run(tmp_path, corpus, "--out-md", str(out_md), "--out-csv", str(out_csv))
    text = out_md.read_text()

    assert "## Lead 30 min" in text
    assert "## Lead 10 min" in text
    # The cell the site ships today is marked in the table and explained
    # in the paragraph underneath it.
    assert "**40 %** (shipping today)" in text
    assert "30 min at 40 %" in text
    assert "no rule" in text  # the do-nothing floor row
    assert "Best CSI:" in text
    # Nothing that could identify a subscriber or an endpoint.
    assert "http" not in text
    assert str(corpus) not in text

    assert payload["current_rule"]["cell"]["hits"] == 1

    csv_lines = out_csv.read_text().splitlines()
    assert csv_lines[0].startswith("lead_min,threshold_pct,")
    # One row per cell plus one do-nothing row per lead.
    assert len(csv_lines) == 1 + len(payload["cells"]) + len(payload["leads"])


def test_settings_and_window_are_recorded(tmp_path: Path, corpus: Path) -> None:
    payload = _run(tmp_path, corpus)
    settings = payload["settings"]
    assert settings["persistence_obs"] == 1
    assert settings["rearm_after_min"] == 60
    assert settings["quiet_hours"] is False
    assert settings["thresholds"] == [40, 50, 70]
    assert payload["window"]["days"] == 2
    assert payload["window"]["stations_scored"] == 2
    assert payload["window"]["station_days"] == 4


def test_an_empty_decisions_tree_is_an_error(tmp_path: Path, corpus: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert sweep.main([
        "--decisions-dir", str(empty),
        "--corpus-dir", str(corpus / "corpus"),
    ]) == 2
