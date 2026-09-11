"""The (lead, threshold) sweep.

The implementation is ``dmi_nowcast_sidecar.threshold_sweep`` (Phase G,
G4: the nightly quality-report task runs the same fit, so it cannot live
in a script); ``scripts/sweep_thresholds.py`` is the CLI over it, and
every test below drives it through that CLI — which is also how the module
and the script are held to the same behaviour.

Tested from the sidecar suite for the same reason
``test_replay_warnings.py`` is: the fit replays ``push.engine`` for the
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
from dmi_nowcast_core.push_thresholds import (  # noqa: E402
    effective_threshold,
    validate_thresholds,
)
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
#: six dry slots behind it and every warning window closes inside the
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
               csi: float | None, f1: float | None = None,
               warnings: int = 100, lead: int = 30) -> dict:
    return {
        "lead_min": lead, "threshold_pct": threshold,
        "pod": pod, "far": far, "csi": csi, "f1": f1,
        "warnings": warnings,
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


# ---------------------------------------------------------------------------
# The plateau pick, on a hand-made F1 curve
# ---------------------------------------------------------------------------


def _f1_curve(values: dict[int, float | None], warnings: int = 100) -> list[dict]:
    """Cells carrying only what the picks read: threshold, F1, warnings."""
    return [
        _grid_cell(threshold, None, None, None, f1=f1, warnings=warnings)
        for threshold, f1 in sorted(values.items())
    ]


def test_the_plateau_is_every_threshold_within_the_fraction() -> None:
    """0.95 of the best F1 is on the plateau; 0.94 is not."""
    cells = _f1_curve({20: 0.50, 30: 0.95, 40: 1.00, 50: 0.96, 60: 0.94})
    pick = sweep.pick_plateau(cells, 0.95)
    assert pick["plateau"] == [30, 50]
    assert pick["n_thresholds"] == 3
    assert pick["max_f1"] == pytest.approx(1.0)
    # The midpoint of [30, 50], not the argmax at 40 — which happens to be
    # the same cell here, and the next test separates them.
    assert pick["threshold_pct"] == 40


def test_the_pick_is_the_midpoint_not_the_argmax() -> None:
    cells = _f1_curve({30: 1.00, 40: 0.97, 50: 0.96, 60: 0.50})
    pick = sweep.pick_plateau(cells, 0.95)
    assert pick["plateau"] == [30, 50]
    assert pick["threshold_pct"] == 40          # argmax would say 30
    assert pick["cell"]["threshold_pct"] == 40  # a measured cell, not a fit


def test_a_half_way_midpoint_rounds_to_the_higher_threshold() -> None:
    """[30, 45] has midpoint 37.5: ties go up, to the quieter rule."""
    cells = _f1_curve({30: 1.00, 45: 0.99, 60: 0.10})
    pick = sweep.pick_plateau(cells, 0.95)
    assert pick["plateau"] == [30, 45]
    assert pick["threshold_pct"] == 40
    # 40 is not on this grid, so the metrics come from the nearest cell on
    # the plateau rather than from an interpolation.
    assert pick["cell"]["threshold_pct"] == 45


def test_a_grid_with_no_f1_at_all_has_no_plateau() -> None:
    assert sweep.pick_plateau(_f1_curve({20: None, 30: None}), 0.95) is None
    # Everything scored zero: there is no flat top to stand in the middle of.
    assert sweep.pick_plateau(_f1_curve({20: 0.0, 30: 0.0}), 0.95) is None


def test_a_lead_with_too_few_warnings_gets_no_pick() -> None:
    thin = _f1_curve({30: 0.8, 40: 0.9, 50: 0.85}, warnings=9)
    picks = sweep.build_picks(thin, [30], min_warnings=30)["30"]
    assert picks["insufficient"] is True
    assert picks["plateau"] is None
    assert picks["scored_warnings"] == 27
    # One more warning and the same grid is fitted.
    ok = sweep.build_picks(thin, [30], min_warnings=27)["30"]
    assert ok["insufficient"] is False
    assert ok["plateau"]["threshold_pct"] == 40


def test_agrees_with_radar_says_yes_no_and_not_compared() -> None:
    gauge = _f1_curve({30: 1.00, 40: 0.99, 50: 0.20})
    inside = _f1_curve({30: 0.96, 40: 1.00, 50: 0.99})   # plateau [30, 50]
    outside = _f1_curve({50: 1.00, 60: 0.99, 30: 0.10})  # plateau [50, 60]

    agreeing = sweep.build_picks(gauge, [30], min_warnings=1, radar_cells=inside)
    assert agreeing["30"]["plateau"]["threshold_pct"] == 35
    assert agreeing["30"]["radar_plateau"]["plateau"] == [30, 50]
    assert agreeing["30"]["agrees_with_radar"] is True

    disagreeing = sweep.build_picks(
        gauge, [30], min_warnings=1, radar_cells=outside,
    )
    assert disagreeing["30"]["agrees_with_radar"] is False

    alone = sweep.build_picks(gauge, [30], min_warnings=1)
    assert alone["30"]["radar_plateau"] is None
    assert alone["30"]["agrees_with_radar"] is None


# ---------------------------------------------------------------------------
# Radar truth: onsets from the composite's own observed rate
# ---------------------------------------------------------------------------


def _radar_track(rates: list[float | None]) -> list[tuple]:
    """A track at 10-minute spacing carrying only ``observed_mm_h``."""
    out = []
    for i, rate in enumerate(rates):
        stamp = _at(1, 7 * 60 + 10 * i)
        out.append((stamp, stamp, 25.0, 1.2, rate, 0.0, 0, (0.1,)))
    return out


def test_radar_onsets_need_the_same_dry_run_the_gauges_do() -> None:
    track = _radar_track([0.0, 0.0, 0.0, 1.2, 1.2, 0.0])
    onsets, known_until = sweep.radar_truth({"P1": track}, dry_min=30)
    assert onsets["P1"] == [_at(1, 7 * 60 + 30)]
    assert known_until["P1"] == _at(1, 7 * 60 + 50)

    # Two dry slots are not enough, exactly as at a gauge.
    too_soon, _ = sweep.radar_truth(
        {"P1": _radar_track([0.0, 0.0, 1.2])}, dry_min=30,
    )
    assert too_soon["P1"] == []

    # And the shipped default asks for six, exactly as at a gauge.
    default, _ = sweep.radar_truth({"P1": track})
    assert default["P1"] == []
    six_dry, _ = sweep.radar_truth(
        {"P1": _radar_track([0.0] * 6 + [1.2, 1.2])},
    )
    assert six_dry["P1"] == [_at(1, 7 * 60 + 60)]


def test_the_radar_amount_rule_reads_the_rate_as_millimetres() -> None:
    """0.2 mm over two 10-minute slots is 1.2 mm/h between them.

    The gauge reports a depth and the radar a rate, so the slot's rate is
    read as sustained across the slot: ``mm = rate * 10 / 60``. Drizzle
    that never adds up is not an event on either instrument.
    """
    drizzle, _ = sweep.radar_truth(
        {"P1": _radar_track([0.0] * 6 + [0.5, 0.5])},
    )
    assert drizzle["P1"] == []          # 0.083 + 0.083 mm
    # …and it was the amount that dropped it, not the dry run.
    loose, _ = sweep.radar_truth(
        {"P1": _radar_track([0.0] * 6 + [0.5, 0.5])}, onset_min_mm=0.0,
    )
    assert loose["P1"] == [_at(1, 7 * 60 + 60)]
    # One slot can carry the floor alone: 1.5 mm/h is 0.25 mm, 1.1 is
    # 0.18. (The bar sits at 1.2 mm/h, which is 0.2 mm to within a float.)
    enough, _ = sweep.radar_truth(
        {"P1": _radar_track([0.0] * 6 + [1.5, 0.0])},
    )
    assert enough["P1"] == [_at(1, 7 * 60 + 60)]
    short, _ = sweep.radar_truth(
        {"P1": _radar_track([0.0] * 6 + [1.1, 0.0])},
    )
    assert short["P1"] == []


def test_the_detection_threshold_is_inclusive_and_a_null_is_unknown() -> None:
    at_threshold, _ = sweep.radar_truth(
        {"P1": _radar_track([0.0, 0.0, 0.0, 0.5])},
        dry_min=30, onset_min_mm=0.0,
    )
    assert at_threshold["P1"] == [_at(1, 7 * 60 + 30)]
    just_under, _ = sweep.radar_truth(
        {"P1": _radar_track([0.0, 0.0, 0.0, 0.49])},
        dry_min=30, onset_min_mm=0.0,
    )
    assert just_under["P1"] == []
    # A null is nodata, not a dry slot: it resets the dry run, so the wet
    # slot behind it cannot be an onset.
    with_hole, _ = sweep.radar_truth(
        {"P1": _radar_track([0.0, 0.0, None, 0.0, 1.2])},
        dry_min=30, onset_min_mm=0.0,
    )
    assert with_hole["P1"] == []


# ---------------------------------------------------------------------------
# The new columns, and the late outcome
# ---------------------------------------------------------------------------


def test_every_cell_carries_the_objective_columns(
    tmp_path: Path, corpus: Path,
) -> None:
    out_csv = tmp_path / "sweep.csv"
    payload = _run(tmp_path, corpus, "--out-csv", str(out_csv))
    cell = _cell(payload, 30, 40)
    # 1 hit, 1 false alarm, 1 miss, no lates at the 5-minute default.
    assert cell["late"] == 0
    assert cell["precision"] == pytest.approx(0.5)
    assert cell["recall"] == pytest.approx(0.5)
    assert cell["f1"] == pytest.approx(0.5)
    assert cell["f_beta_0.5"] == pytest.approx(0.5)
    assert cell["f_beta_2"] == pytest.approx(0.5)
    assert cell["recall"] == cell["pod"]
    header = out_csv.read_text().splitlines()[0].split(",")
    for column in ("late", "precision", "recall", "f1", "f_beta_0.5", "f_beta_2"):
        assert column in header


def test_a_lead_shorter_than_the_useful_minimum_is_late_not_a_hit(
    tmp_path: Path, corpus: Path,
) -> None:
    """The 07:30 warning gets 30 minutes of lead; demand 35 and it is late."""
    payload = _run(tmp_path, corpus, "--min-useful-lead-min", "35")
    cell = _cell(payload, 30, 40)
    assert cell["hits"] == 0
    assert cell["late"] == 1
    assert cell["false_alarms"] == 1     # the day-2 warning is still wrong
    assert cell["misses"] == 1
    assert cell["precision"] == 0.0
    assert cell["recall"] == 0.0
    assert cell["f1"] is None            # no harmonic mean of two zeroes
    assert payload["settings"]["min_useful_lead_min"] == 35.0


# ---------------------------------------------------------------------------
# The thresholds document
# ---------------------------------------------------------------------------


def test_a_thin_grid_gets_no_pick_and_says_so(
    tmp_path: Path, corpus: Path,
) -> None:
    """Two warnings is not evidence; the lead keeps the fallback."""
    out = tmp_path / "push_thresholds.json"
    payload = _run(tmp_path, corpus, "--out-thresholds", str(out))
    assert payload["picks"]["30"]["insufficient"] is True
    assert payload["picks"]["30"]["plateau"] is None

    doc = json.loads(out.read_text())
    assert validate_thresholds(doc) == []
    assert doc["leads"]["30"]["insufficient"] is True
    assert doc["leads"]["30"]["threshold_pct"] is None
    assert effective_threshold(doc, 30) == doc["fallback_threshold_pct"] == 40


def test_the_thresholds_document_validates_and_round_trips(
    tmp_path: Path, corpus: Path,
) -> None:
    out = tmp_path / "push_thresholds.json"
    payload = _run(
        tmp_path, corpus, "--min-warnings", "1", "--out-thresholds", str(out),
    )
    doc = json.loads(out.read_text())
    assert validate_thresholds(doc) == []
    assert doc["schema_version"] == 1
    assert doc["objective"] == {
        "metric": "f1", "min_useful_lead_min": 5.0, "plateau_frac": 0.95,
        "min_warnings": 1, "rearm_after_min": 60, "persistence_obs": 1,
        "tolerance_min": 10, "dry_min": 60, "onset_min_mm": 0.2,
        # A percent is a threshold ON something; the default is the
        # served probability, and the document says so.
        "probability_column": "p_rain_{lead}",
    }
    assert doc["window"]["days"] == 2
    assert doc["window"]["stations"] == 2

    # 40 % and 50 % tie on F1 and 70 % sends nothing: the plateau is
    # [40, 50] and its midpoint is 45.
    lead30 = doc["leads"]["30"]
    assert lead30["plateau"] == [40, 50]
    assert lead30["threshold_pct"] == 45
    assert lead30["insufficient"] is False
    assert lead30["hits"] == 1 and lead30["false_alarms"] == 1
    assert lead30["late"] == 0
    assert lead30["f1"] == pytest.approx(0.5)
    assert lead30["radar_plateau"] is None
    assert lead30["agrees_with_radar"] is None
    assert effective_threshold(doc, 30) == 45
    # A lead nobody fitted is not an error, it is the fallback.
    assert effective_threshold(doc, 45) == 40
    assert payload["picks"]["30"]["max_csi"]["threshold_pct"] == 50


def test_the_markdown_reports_the_pick_and_the_plateau(
    tmp_path: Path, corpus: Path,
) -> None:
    out_md = tmp_path / "sweep.md"
    _run(tmp_path, corpus, "--min-warnings", "1", "--out-md", str(out_md))
    text = out_md.read_text()
    assert "**Pick: 45 %**" in text
    assert "F1 plateau [40 %, 50 %]" in text
    assert "late" in text
    assert str(corpus) not in text


# ---------------------------------------------------------------------------
# The radar cross-check, end to end
# ---------------------------------------------------------------------------

#: Radar points, and the frames their probability crosses at.
RADAR_POINTS = ("P1", "P2")


def _radar_rows() -> list[dict]:
    """Two calibration points; truth is the rows' own ``observed_mm_h``.

    P1 crosses at 07:30 and the composite starts raining there at 08:00 —
    a hit at every lead. P2 never crosses and starts raining at 08:30 — a
    miss. The same shape as the gauge fixture, with the gauge swapped for
    the radar's own observation.
    """
    rows: list[dict] = []
    for minute in range(FIRST_FRAME_MIN, LAST_FRAME_MIN + 1, 10):
        radar_ts = _at(1, minute)
        hhmm = _hhmm(radar_ts)
        for point in RADAR_POINTS:
            if point == "P1":
                p = 0.60 if hhmm in ("07:30", "07:40") else 0.10
                observed = 1.2 if hhmm in ("08:00", "08:10") else 0.0
            else:
                p = 0.10
                observed = 1.2 if hhmm in ("08:30", "08:40") else 0.0
            rows.append({
                "radar_ts": radar_ts,
                "generated_at": radar_ts,
                "station_id": point,
                "p_rain": 0.99,
                "action": "none",
                "p_rain_10": p,
                "p_rain_30": p,
                "eta_min": 25.0,
                "intensity_mm_h": 1.2,
                "observed_mm_h": observed,
                "forecast_now_mm_h": 0.0,
                "armed_after": True,
                "streak_after": 0,
            })
    return rows


def test_the_radar_set_is_swept_and_reported_beside_the_gauge_pick(
    tmp_path: Path, corpus: Path,
) -> None:
    radar_dir = tmp_path / "radar"
    _write_decisions(radar_dir, _radar_rows(), "2026-06-01.parquet")
    out = tmp_path / "push_thresholds.json"
    out_md = tmp_path / "with_radar.md"
    payload = _run(
        tmp_path, corpus, "--min-warnings", "1",
        "--radar-decisions-dir", str(radar_dir),
        "--out-thresholds", str(out), "--out-md", str(out_md),
    )
    assert payload["radar"]["points"] == 2
    assert payload["radar"]["onsets"] == 2

    doc = json.loads(out.read_text())
    assert validate_thresholds(doc) == []
    lead30 = doc["leads"]["30"]
    # The radar sees the same crossing and the same rain: 40 % and 50 %
    # both score, 70 % sends nothing, so the plateau matches the gauges'
    # and the gauge pick of 45 % sits inside it.
    assert lead30["radar_plateau"] == [40, 50]
    assert lead30["agrees_with_radar"] is True

    text = out_md.read_text()
    assert "Radar cross-check" in text
    assert "is inside it" in text
    assert "NOT independent truth" in text


def test_a_radar_pick_that_disagrees_is_reported_as_a_disagreement(
    tmp_path: Path, corpus: Path,
) -> None:
    """The radar set is allowed to disagree, and then it says so.

    Its probabilities are shifted down so only the 40 % column fires: the
    radar plateau collapses to [40, 40] and the gauge pick of 45 % is
    outside it.
    """
    rows = []
    for row in _radar_rows():
        shifted = dict(row)
        for column in ("p_rain_10", "p_rain_30"):
            if shifted[column] > 0.5:
                shifted[column] = 0.45
        rows.append(shifted)
    radar_dir = tmp_path / "radar_low"
    _write_decisions(radar_dir, rows, "2026-06-01.parquet")
    out = tmp_path / "push_thresholds.json"
    _run(
        tmp_path, corpus, "--min-warnings", "1",
        "--radar-decisions-dir", str(radar_dir), "--out-thresholds", str(out),
    )
    doc = json.loads(out.read_text())
    assert doc["leads"]["30"]["radar_plateau"] == [40, 40]
    assert doc["leads"]["30"]["agrees_with_radar"] is False
    # The gauge pick still ships: the radar set never overrides it.
    assert doc["leads"]["30"]["threshold_pct"] == 45



# ---------------------------------------------------------------------------
# The importable module (Phase G, G4)
# ---------------------------------------------------------------------------


def test_the_module_and_the_cli_fit_the_same_table(
    tmp_path: Path, corpus: Path,
) -> None:
    """``run_fit`` is what the CLI runs — the nightly task gets the same fit.

    The two paths differ only in how the options arrive (argparse versus a
    dataclass) and in what is done with the result, so the fitted table
    must be identical apart from its timestamp.
    """
    from dmi_nowcast_sidecar.threshold_sweep import SweepOptions, run_fit

    out = tmp_path / "push_thresholds.json"
    cli = _run(
        tmp_path, corpus, "--min-warnings", "1", "--out-thresholds", str(out),
    )
    direct = run_fit(SweepOptions(
        decisions_dirs=[corpus / "decisions"],
        corpus_dir=corpus / "corpus",
        leads=(10, 30),
        thresholds=(40, 50, 70),
        min_warnings=1,
    ))

    # The CLI runs the stability guard over the fit; with no --previous
    # that stamps every lead "first_fit" and changes nothing else.
    assert cli["thresholds"]["leads"] == {
        lead: {**row, "guard": "first_fit" if row["threshold_pct"] else
               "insufficient"}
        for lead, row in direct["thresholds"]["leads"].items()
    }
    assert direct["window"] == cli["window"]
    assert direct["cells"] == cli["cells"]
    assert validate_thresholds(direct["thresholds"]) == []
    assert direct["thresholds_schema_problems"] == []


def test_nothing_to_fit_on_raises_rather_than_exits(
    tmp_path: Path, corpus: Path,
) -> None:
    """The CLI's exit code 2 is this exception, caught."""
    from dmi_nowcast_sidecar.threshold_sweep import SweepError, SweepOptions, run_fit

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SweepError, match="no decision rows"):
        run_fit(SweepOptions(
            decisions_dirs=[empty], corpus_dir=corpus / "corpus",
        ))


# ---------------------------------------------------------------------------
# The stability guard, through the CLI
# ---------------------------------------------------------------------------


def test_previous_keeps_a_served_threshold_the_refit_barely_moves(
    tmp_path: Path, corpus: Path,
) -> None:
    """The fixture fits 45 %; a served 44 % is one point away and stays."""
    out = tmp_path / "push_thresholds.json"
    _run(tmp_path, corpus, "--min-warnings", "1", "--out-thresholds", str(out))
    assert json.loads(out.read_text())["leads"]["30"]["threshold_pct"] == 45

    served = tmp_path / "served.json"
    doc = json.loads(out.read_text())
    doc["leads"]["30"]["threshold_pct"] = 44
    doc["leads"]["30"]["plateau"] = [40, 50]
    served.write_text(json.dumps(doc))

    _run(
        tmp_path, corpus, "--min-warnings", "1",
        "--out-thresholds", str(out), "--previous", str(served),
    )
    refitted = json.loads(out.read_text())["leads"]["30"]
    assert refitted["threshold_pct"] == 44
    assert refitted["guard"] == "kept_previous"
    assert refitted["candidate_threshold_pct"] == 45
    assert validate_thresholds(json.loads(out.read_text())) == []


def test_an_absent_previous_file_is_a_first_fit(
    tmp_path: Path, corpus: Path,
) -> None:
    out = tmp_path / "push_thresholds.json"
    _run(
        tmp_path, corpus, "--min-warnings", "1",
        "--out-thresholds", str(out),
        "--previous", str(tmp_path / "never_written.json"),
    )
    row = json.loads(out.read_text())["leads"]["30"]
    assert row["threshold_pct"] == 45
    assert row["guard"] == "first_fit"


# ---------------------------------------------------------------------------
# Seasons: --strata season
# ---------------------------------------------------------------------------
#
# A second, deliberately two-regime fixture. Four stations, three days, one
# crossing each at 07:30 and (on a wet day) rain at 08:00:
#
#   2026-03-02  winter, p = 0.75, rain      → 4 hits wherever 75 % fires
#   2026-03-03  winter, p = 0.45, no rain   → 4 false alarms below 45 %
#   2026-05-04  summer, p = 0.55, rain      → 4 hits wherever 55 % fires
#
# so on a 30/40/50/60/70 grid the F1 columns are, by hand:
#
#   summer  30–50: 1.00        60,70: nothing sent   → plateau [30, 50] → 40 %
#   winter  30,40: 0.67 (4 FA) 50–70: 1.00           → plateau [50, 70] → 60 %
#   pooled  30,40: 0.80        50: 1.00  60,70: 0.67 → plateau [50, 50] → 50 %
#
# The pooled pick sits exactly between the two seasonal ones, twenty points
# apart — which is the case the flag exists to name.

SEASON_STATIONS = ("06180", "06120", "06110", "06060")

#: ``(day, probability at the 07:30 crossing, does it rain at 08:00)``.
SPLIT_DAYS = (
    (datetime(2026, 3, 2, tzinfo=timezone.utc), 0.75, True),
    (datetime(2026, 3, 3, tzinfo=timezone.utc), 0.45, False),
    (datetime(2026, 5, 4, tzinfo=timezone.utc), 0.55, True),
)
#: The same shape with the winter noise removed: both seasons then want the
#: same threshold, and the flag must stay off.
AGREEING_DAYS = (
    (datetime(2026, 3, 2, tzinfo=timezone.utc), 0.55, True),
    (datetime(2026, 5, 4, tzinfo=timezone.utc), 0.55, True),
)
#: SPLIT_DAYS plus an April day — a shoulder month, reported on its own.
SHOULDER_DAYS = SPLIT_DAYS + (
    (datetime(2026, 4, 6, tzinfo=timezone.utc), 0.55, True),
)


def _episode_rows(date_utc: datetime, p_crossing: float) -> list[dict]:
    """One day of frames for every season station, crossing at 07:30."""
    rows: list[dict] = []
    for minute in range(FIRST_FRAME_MIN, LAST_FRAME_MIN + 1, 10):
        radar_ts = date_utc + timedelta(minutes=minute)
        p = p_crossing if _hhmm(radar_ts) == "07:30" else 0.10
        for station in SEASON_STATIONS:
            rows.append({
                "radar_ts": radar_ts,
                "generated_at": radar_ts,
                "station_id": station,
                "p_rain": 0.99,
                "action": "none",
                "p_rain_10": p,
                "p_rain_30": p,
                "eta_min": 25.0,
                "intensity_mm_h": 1.2,
                "observed_mm_h": 0.0,
                "forecast_now_mm_h": 0.0,
                "armed_after": True,
                "streak_after": 0,
            })
    return rows


def _season_corpus(root: Path, days) -> Path:
    """Decisions and gauge for a list of ``(day, p, wet)`` episodes."""
    observations: list[Observation] = []
    for date_utc, p_crossing, wet in days:
        _write_decisions(
            root / "decisions", _episode_rows(date_utc, p_crossing),
            f"{date_utc:%Y-%m-%d}.parquet",
        )
        for station in SEASON_STATIONS:
            for minute in range(FIRST_SLOT_MIN, LAST_SLOT_MIN + 1, 10):
                stamp = date_utc + timedelta(minutes=minute)
                rainy = wet and _hhmm(stamp) in ("08:00", "08:10")
                observations.append(Observation(
                    station_id=station,
                    observed_utc=stamp,
                    parameter_id="precip_past10min",
                    value=0.5 if rainy else 0.0,
                ))
    StationObsStore(root / "corpus").append(observations)
    return root


def _run_seasons(out_dir: Path, root: Path, *extra: str) -> dict:
    out_json = out_dir / "sweep.json"
    argv = [
        "--decisions-dir", str(root / "decisions"),
        "--corpus-dir", str(root / "corpus"),
        "--leads", "30",
        "--thresholds", "30,40,50,60,70",
        "--min-warnings", "1",
        "--out-json", str(out_json),
        *extra,
    ]
    assert sweep.main(argv) == 0
    return json.loads(out_json.read_text())


def _season_pick(payload: dict, season: str, lead: int = 30) -> int | None:
    entry = payload["strata"]["season"][season]["picks"][str(lead)]
    return (entry["plateau"] or {}).get("threshold_pct")


# -- the month → stratum mapping --------------------------------------------


@pytest.mark.parametrize(("month", "season"), [
    (1, "winter"), (2, "winter"), (3, "winter"),    # March is still winter
    (4, "shoulder"),                                # April is neither
    (5, "summer"), (6, "summer"), (7, "summer"),
    (8, "summer"), (9, "summer"),                   # September is still summer
    (10, "shoulder"), (11, "shoulder"),             # October is neither
    (12, "winter"),
])
def test_every_month_maps_to_exactly_one_season(month: int, season: str) -> None:
    assert sweep.season_of(datetime(2026, month, 15, tzinfo=timezone.utc)) == season
    assert month in sweep.SEASON_MONTHS[season]
    others = [name for name in sweep.SEASON_ORDER if name != season]
    assert all(month not in sweep.SEASON_MONTHS[name] for name in others)


def test_the_seasons_partition_the_year_without_overlapping() -> None:
    """Twelve months, three disjoint buckets — nothing lost, nothing double."""
    buckets = [set(sweep.SEASON_MONTHS[name]) for name in sweep.SEASON_ORDER]
    assert set().union(*buckets) == set(range(1, 13))
    assert sum(len(bucket) for bucket in buckets) == 12
    # The shoulder is its own stratum, never folded into a season.
    assert set(sweep.SEASON_MONTHS["shoulder"]) == {4, 10, 11}


def test_only_known_strata_are_accepted() -> None:
    assert sweep.parse_strata(["season", "season"]) == ("season",)
    assert sweep.parse_strata(None) == ()
    with pytest.raises(ValueError, match="unknown stratum"):
        sweep.parse_strata(["region"])


# -- slicing the tracks ------------------------------------------------------


def _stamped_track(stamps: list[datetime]) -> list[tuple]:
    return [(ts, ts, 25.0, 1.2, 0.0, 0.0, 0, (0.6,)) for ts in stamps]


def test_a_slice_keeps_only_its_months_and_re_runs_the_coverage_index() -> None:
    """The seam where the other season was cut out is a gap, not a carry."""
    march = datetime(2026, 3, 2, 7, tzinfo=timezone.utc)
    may = datetime(2026, 5, 4, 7, tzinfo=timezone.utc)
    track = _stamped_track([
        march, march + timedelta(minutes=10),
        may, may + timedelta(minutes=10),
        may + timedelta(days=1),  # a second summer run, a day later
    ])

    summer = sweep.filter_tracks_by_months({"A": track}, (5, 6, 7, 8, 9))
    assert [record[0] for record in summer["A"]] == [
        may, may + timedelta(minutes=10), may + timedelta(days=1),
    ]
    # Re-derived from what survived: two frames in run 0, the next day in 1.
    assert [record[6] for record in summer["A"]] == [0, 0, 1]

    winter = sweep.filter_tracks_by_months({"A": track}, (12, 1, 2, 3))
    assert [record[6] for record in winter["A"]] == [0, 0]
    # A station with nothing in the slice drops out rather than arriving empty.
    assert sweep.filter_tracks_by_months({"A": track}, (10,)) == {}


# -- the two-regime fixture, end to end -------------------------------------


def test_the_seasons_pick_apart_and_the_pool_lands_between(tmp_path: Path) -> None:
    root = _season_corpus(tmp_path / "split", SPLIT_DAYS)
    payload = _run_seasons(tmp_path, root, "--strata", "season")

    # No April, October or November in the window: two strata, not three.
    assert set(payload["strata"]["season"]) == {"summer", "winter"}
    assert payload["settings"]["strata"] == ["season"]

    assert _season_pick(payload, "summer") == 40
    assert _season_pick(payload, "winter") == 60
    assert payload["picks"]["30"]["plateau"]["threshold_pct"] == 50

    summer = payload["strata"]["season"]["summer"]
    winter = payload["strata"]["season"]["winter"]
    assert summer["months"] == [5, 6, 7, 8, 9]
    assert winter["months"] == [12, 1, 2, 3]
    assert summer["window"]["onsets"] == 4
    assert winter["window"]["onsets"] == 4
    # A stratum is scored on its own rows: one summer day, two winter days.
    assert summer["window"]["days"] == 1
    assert winter["window"]["days"] == 2

    def cell(group: dict, threshold: int) -> dict:
        for entry in group["cells"]:
            if entry["threshold_pct"] == threshold:
                return entry
        raise AssertionError(threshold)

    # Summer: four warnings, four hits, nothing false, at 30 through 50.
    assert cell(summer, 40)["hits"] == 4
    assert cell(summer, 40)["false_alarms"] == 0
    assert cell(summer, 60)["n_sent"] == 0
    assert cell(summer, 60)["misses"] == 4
    # Winter: the 45 % noise day is four false alarms below 50 % and silent
    # above it — which is the whole reason the two seasons disagree.
    assert cell(winter, 40)["false_alarms"] == 4
    assert cell(winter, 50)["false_alarms"] == 0
    assert cell(winter, 50)["hits"] == 4
    # A stratum's picks carry the pooled shape, radar fields included.
    entry = winter["picks"]["30"]
    assert entry["insufficient"] is False
    assert entry["radar_plateau"] is None and entry["agrees_with_radar"] is None
    assert entry["min_warnings"] == 1
    assert entry["plateau"]["plateau"] == [50, 70]


def test_a_thin_stratum_is_marked_insufficient_like_the_pool(
    tmp_path: Path,
) -> None:
    """The evidence test is per replay, so a season can fail it alone."""
    root = _season_corpus(tmp_path / "thin", SPLIT_DAYS)
    payload = _run_seasons(
        tmp_path, root, "--strata", "season", "--min-warnings", "16",
    )
    summer = payload["strata"]["season"]["summer"]["picks"]["30"]
    winter = payload["strata"]["season"]["winter"]["picks"]["30"]
    # Summer grades 12 warnings across the grid, winter 28.
    assert summer["scored_warnings"] == 12
    assert summer["insufficient"] is True
    assert summer["plateau"] is None
    assert winter["scored_warnings"] == 28
    assert winter["insufficient"] is False


def test_a_shoulder_month_is_its_own_stratum(tmp_path: Path) -> None:
    """April is reported beside the seasons, never folded into one."""
    root = _season_corpus(tmp_path / "shoulder", SHOULDER_DAYS)
    payload = _run_seasons(tmp_path, root, "--strata", "season")
    seasons = payload["strata"]["season"]
    assert set(seasons) == {"summer", "winter", "shoulder"}
    assert seasons["shoulder"]["months"] == [4, 10, 11]
    assert seasons["shoulder"]["window"]["onsets"] == 4
    # The April day did not join the summer slice: one summer day still.
    assert seasons["summer"]["window"]["days"] == 1
    assert seasons["summer"]["window"]["onsets"] == 4


def test_the_strata_do_not_touch_the_pooled_fit(tmp_path: Path) -> None:
    """Analysis output: same cells, same picks, same document."""
    root = _season_corpus(tmp_path / "same", SPLIT_DAYS)
    out_plain = tmp_path / "plain.json"
    out_split = tmp_path / "split.json"
    plain = _run_seasons(
        tmp_path / "a", root, "--out-thresholds", str(out_plain),
    )
    with_strata = _run_seasons(
        tmp_path / "b", root, "--strata", "season",
        "--out-thresholds", str(out_split),
    )
    assert plain["strata"] == {}
    assert plain["cells"] == with_strata["cells"]
    assert plain["picks"] == with_strata["picks"]
    assert plain["do_nothing"] == with_strata["do_nothing"]
    assert plain["window"] == with_strata["window"]

    def without_stamp(path: Path) -> dict:
        doc = json.loads(path.read_text())
        doc.pop("fitted_at_utc")
        return doc

    assert without_stamp(out_plain) == without_stamp(out_split)
    assert validate_thresholds(json.loads(out_split.read_text())) == []
    assert json.loads(out_split.read_text())["leads"]["30"]["threshold_pct"] == 50


# -- the CSV's stratum column ------------------------------------------------


def test_the_csv_labels_the_pooled_rows_all_and_the_slices_by_name(
    tmp_path: Path,
) -> None:
    import csv as csv_module

    root = _season_corpus(tmp_path / "csv", SPLIT_DAYS)
    out_csv = tmp_path / "sweep.csv"
    payload = _run_seasons(
        tmp_path, root, "--strata", "season", "--out-csv", str(out_csv),
    )
    text = out_csv.read_text()
    # Appended, not prepended: a reader pinned to the leading columns lives.
    assert text.splitlines()[0].startswith("lead_min,threshold_pct,")
    assert text.splitlines()[0].endswith(",stratum")

    rows = list(csv_module.DictReader(text.splitlines()))
    by_stratum: dict[str, list[dict]] = {}
    for row in rows:
        by_stratum.setdefault(row["stratum"], []).append(row)
    assert set(by_stratum) == {"all", "summer", "winter"}

    leads = len(payload["leads"])
    assert len(by_stratum["all"]) == len(payload["cells"]) + leads
    for season in ("summer", "winter"):
        group = payload["strata"]["season"][season]
        assert len(by_stratum[season]) == len(group["cells"]) + leads
    # The pooled block is written first and is the fit; the slices follow.
    assert rows[0]["stratum"] == "all"
    assert {row["stratum"] for row in rows[:len(by_stratum["all"])]} == {"all"}
    winter_40 = next(
        row for row in by_stratum["winter"]
        if row["threshold_pct"] == "40" and row["lead_min"] == "30"
    )
    assert winter_40["false_alarms"] == "4"


def test_without_strata_the_csv_still_says_all(tmp_path: Path, corpus: Path) -> None:
    out_csv = tmp_path / "plain.csv"
    payload = _run(tmp_path, corpus, "--out-csv", str(out_csv))
    rows = out_csv.read_text().splitlines()
    assert len(rows) == 1 + len(payload["cells"]) + len(payload["leads"])
    assert all(row.endswith(",all") for row in rows[1:])


# -- the markdown comparison and its flag ------------------------------------


def test_the_markdown_compares_the_seasons_and_raises_the_flag(
    tmp_path: Path,
) -> None:
    root = _season_corpus(tmp_path / "flag", SPLIT_DAYS)
    out_md = tmp_path / "sweep.md"
    _run_seasons(tmp_path, root, "--strata", "season", "--out-md", str(out_md))
    text = out_md.read_text()

    assert "| threshold | F1 all | F1 summer | F1 winter |" in text
    assert "**Seasons at lead 30 min.**" in text
    # Pooled 50 %, summer ten points under it, winter ten points over.
    assert (
        "- **Seasonal picks:** pooled 50 %; summer 40 % (-10 points), "
        "winter 60 % (+10 points)." in text
    )
    assert "20 points apart — **seasonal split worth considering**" in text
    assert "Summer is May–September, winter December–March" in text
    assert str(root) not in text


def test_the_flag_stays_down_when_the_seasons_agree(tmp_path: Path) -> None:
    root = _season_corpus(tmp_path / "agree", AGREEING_DAYS)
    out_md = tmp_path / "sweep.md"
    payload = _run_seasons(
        tmp_path, root, "--strata", "season", "--out-md", str(out_md),
    )
    assert _season_pick(payload, "summer") == _season_pick(payload, "winter") == 40
    text = out_md.read_text()
    assert "seasonal split worth considering" not in text
    assert "0 points apart, under the 10-point mark" in text
    assert "| F1 summer | F1 winter |" in text


def test_no_strata_means_no_season_block_in_the_markdown(
    tmp_path: Path, corpus: Path,
) -> None:
    out_md = tmp_path / "plain.md"
    _run(tmp_path, corpus, "--out-md", str(out_md))
    text = out_md.read_text()
    assert "F1 summer" not in text
    assert "Seasonal picks" not in text
