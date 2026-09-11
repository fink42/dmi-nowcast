"""The Layer B / Layer C benchmark report (Phase H, H0b).

Tested from the sidecar suite for the same reason ``test_sweep_thresholds``
is: the report replays ``push.engine`` for Layer C and the core library for
Layer B, and this environment is the only one that has both.

Fully offline and fully synthetic — no radar, no STEPS, no network. The
fixture is three single-day months over two stations, planted so that every
number below can be worked out on paper:

* **06180** — the probability rises to 0.875 two frames before the rain,
  and the gauge starts raining at 08:00 (0.5 mm in that slot and 0.5 in
  the next, so the onset's amount test passes with room to spare).
* **06120** — the gauge never once rains, and the probability rises to
  exactly 0.5 at 08:00, so a threshold of 30 % or 50 % buys a false alarm
  there and a threshold of 70 % does not. That is what makes the
  leave-one-month-out refit have an *obviously* right answer.

Every probability in the fixture — 0.125, 0.5, 0.875 — is a binary
fraction, so it survives the decision schema's float32 column unchanged
and the reliability bin it lands in is not an accident of rounding. A
"0.9" would come back as 0.899999976 and fall in bin **8**, which is
correct behaviour and a terrible thing to assert against.

The months are January (winter), April (shoulder) and June (summer), one
day each, so the seasonal strata and the three-fold LOMO both have
something to chew on.

The rows carry the same two traps the sweep fixture uses: ``p_rain`` is
0.99 on every row and ``action`` is ``"none"`` on every row. A report that
read either — the rule's own probability instead of the per-lead column,
or the stored decision instead of a fresh replay — would produce numbers
nothing like the ones asserted here.
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

import benchmark_report as report_module  # noqa: E402  (after the sys.path edit)

from dmi_nowcast_core.metobs import Observation  # noqa: E402
from dmi_nowcast_core.station_store import StationObsStore  # noqa: E402
from dmi_nowcast_core.warning_score import (  # noqa: E402
    GaugeTruth,
    StationSlots,
    decision_table,
)

STATION_A = "06180"
STATION_B = "06120"
STATIONS = (STATION_A, STATION_B)
LEADS = (20, 30)

#: One day per month: winter, shoulder, summer.
DAYS: tuple[tuple[int, int, int], ...] = (
    (2026, 1, 15), (2026, 4, 15), (2026, 6, 15),
)
#: Decision frames 07:00–09:00 UTC every 10 min (13 per station per day).
FIRST_FRAME_MIN = 7 * 60
LAST_FRAME_MIN = 9 * 60
FRAMES_PER_DAY = (LAST_FRAME_MIN - FIRST_FRAME_MIN) // 10 + 1

#: Gauge slots 06:00–10:00, so every onset has its six dry slots behind it
#: and every warning window closes inside the reported record.
FIRST_SLOT_MIN = 6 * 60
LAST_SLOT_MIN = 10 * 60

#: The slots station A reports as wet, on every day.
WET_SLOTS = ("08:00", "08:10")

SLOT_SEC = 600


def _at(day: tuple[int, int, int], minute_of_day: int) -> datetime:
    return datetime(*day, tzinfo=timezone.utc) + timedelta(minutes=minute_of_day)


def _hhmm(ts: datetime) -> str:
    return ts.strftime("%H:%M")


def _p_values(station: str, ts: datetime) -> tuple[float, float]:
    """``(p_rain_20, p_rain_30)`` for one frame of the fixture."""
    hhmm = _hhmm(ts)
    if station == STATION_A:
        p_20 = 0.875 if hhmm in ("07:40", "07:50") else 0.125
        p_30 = 0.875 if hhmm in ("07:30", "07:40", "07:50") else 0.125
        return p_20, p_30
    # Station B: one medium spike over a gauge that stays dry all day.
    spike = 0.5 if hhmm == "08:00" else 0.125
    return spike, spike


def _decision_rows(day: tuple[int, int, int]) -> list[dict]:
    rows: list[dict] = []
    for minute in range(FIRST_FRAME_MIN, LAST_FRAME_MIN + 1, 10):
        radar_ts = _at(day, minute)
        for station in STATIONS:
            p_20, p_30 = _p_values(station, radar_ts)
            rows.append({
                "radar_ts": radar_ts,
                # Zero frame age: the decision instant IS the frame instant,
                # which makes the outcome-window arithmetic checkable.
                "generated_at": radar_ts,
                "station_id": station,
                # Traps — see the module docstring.
                "p_rain": 0.99,
                "action": "none",
                "p_rain_20": p_20,
                "p_rain_30": p_30,
                "eta_min": 20.0,
                "intensity_mm_h": 1.2,
                # Below Rules.raining_now_mm_h, so nothing is silenced.
                "observed_mm_h": 0.0,
                "forecast_now_mm_h": 0.0,
                "armed_after": True,
                "streak_after": 0,
            })
    return rows


def _write_decisions(directory: Path) -> Path:
    import pyarrow.parquet as pq

    decisions = directory / "decisions"
    decisions.mkdir(parents=True, exist_ok=True)
    for day in DAYS:
        pq.write_table(
            decision_table(_decision_rows(day), LEADS),
            decisions / f"{day[0]:04d}-{day[1]:02d}-{day[2]:02d}.parquet",
        )
    return directory


def _anchor_block(
    policy: str = "fullRange", *, harmonisation_sha: str | None = None,
) -> dict:
    """The replay's L3 provenance block, as ``summary.json`` carries it."""
    return {
        "policy": policy,
        "lag_min": {"fullRange": 13.1, "doppler": 8.1, "flat": None},
        "poll_interval_min": 5.0,
        "frame_age_override_min": None,
        "max_anchor_age_min": 60.0,
        "history_mode": "same-type",
        "harmonisation": (
            {"path": None} if harmonisation_sha is None
            else {"path": "/x/doppler_harmonisation.json",
                  "sha256": harmonisation_sha, "schema_version": 1,
                  "fitted_at": "2026-09-08T18:44:53+00:00"}
        ),
        "counts": {"instants": 10, "doppler_anchored": 0,
                   "fullrange_anchored": 10, "degraded": 0,
                   "history_fallback": 0, "no_anchor": 0, "no_history": 0},
        "frame_age_min": {"n": 10, "mean": 15.0, "p50": 15.0,
                          "min": 15.0, "max": 15.0},
    }


def _write_summary(
    directory: Path, *, ensemble_size: int = 16, anchor: dict | None = None,
) -> None:
    (directory / "summary.json").write_text(json.dumps({
        "run": {
            "n_days": len(DAYS),
            "n_frames": FRAMES_PER_DAY * len(DAYS),
            "n_stations": len(STATIONS),
            "n_decision_rows": FRAMES_PER_DAY * len(STATIONS) * len(DAYS),
            "frame_age_min": 0,
            "anchor": anchor if anchor is not None else _anchor_block(),
            "steps": {
                "ensemble_size": ensemble_size,
                "n_cascade_levels": 6,
                "downsample_factor": 4,
                "horizon_min": 90,
                "leads_min": list(LEADS),
                "threshold_mm_h": 0.5,
            },
            "flow": {"completion": "gated"},
            "rules": {"lead_min": 30, "threshold_pct": 40},
        },
    }))


def _write_gauge(corpus_dir: Path) -> None:
    store = StationObsStore(corpus_dir)
    observations: list[Observation] = []
    for day in DAYS:
        for station in STATIONS:
            for minute in range(FIRST_SLOT_MIN, LAST_SLOT_MIN + 1, 10):
                stamp = _at(day, minute)
                wet = station == STATION_A and _hhmm(stamp) in WET_SLOTS
                observations.append(Observation(
                    station_id=station,
                    observed_utc=stamp,
                    parameter_id="precip_past10min",
                    value=0.5 if wet else 0.0,
                ))
    store.append(observations)


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """A gauge store plus a replay run directory, both hand-planted."""
    _write_gauge(tmp_path / "corpus")
    run = _write_decisions(tmp_path / "replay")
    _write_summary(run)
    return tmp_path


def _run(tmp_path: Path, *extra: str) -> dict:
    """Run the CLI over the fixture and return ``benchmark.json``."""
    out = tmp_path / "out"
    argv = [
        "--baseline", str(tmp_path / "replay"),
        "--corpus-dir", str(tmp_path / "corpus"),
        "--out-dir", str(out),
        "--leads", "20,30",
        "--thresholds", "30:70:20",
        "--min-warnings", "1",
        "--resamples", "25",
        *extra,
    ]
    assert report_module.main(argv) == 0
    assert (out / "benchmark.md").is_file()
    return json.loads((out / "benchmark.json").read_text())


# ---------------------------------------------------------------------------
# The outcome window, straight on the grid
# ---------------------------------------------------------------------------


def _grid(pattern: str, *, base: datetime) -> report_module.GaugeGrid:
    """A one-station grid from ``.`` dry / ``W`` wet / ``?`` unknown."""
    mm = numpy.array(
        [{".": 0.0, "W": 0.5, "?": float("nan")}[s] for s in pattern],
        dtype=numpy.float32,
    )
    slots = StationSlots(
        station_id="X",
        slot_end=int(base.timestamp())
        + numpy.arange(mm.size, dtype=numpy.int64) * SLOT_SEC,
        mm=mm,
        dur=numpy.full(mm.size, numpy.nan, dtype=numpy.float32),
        known=~numpy.isnan(mm),
        wet=numpy.nan_to_num(mm, nan=0.0) >= 0.1,
    )
    return report_module.GaugeGrid(GaugeTruth(series={"X": slots}), ["X"])


def _ask(grid, base: datetime, offset_min: int, lead: int):
    t = numpy.array([int((base + timedelta(minutes=offset_min)).timestamp())])
    y, usable = grid.outcome(t, numpy.zeros(1, dtype=numpy.int64), lead)
    return (None if not bool(usable[0]) else int(y[0]))


class TestOutcomeWindow:
    """`(t, t + lead]` on slot ENDS — the one boundary everything hangs on."""

    BASE = datetime(2026, 6, 15, 6, 0, tzinfo=timezone.utc)
    # Slot ends 06:00, 06:10, … ; the wet one ends at 07:00 (index 6).
    PATTERN = "." * 6 + "W" + "." * 12

    def test_a_slot_ending_inside_the_window_makes_it_wet(self) -> None:
        grid = _grid(self.PATTERN, base=self.BASE)
        # t = 06:40, lead 20 → slots ending 06:50 and 07:00. 07:00 is wet.
        assert _ask(grid, self.BASE, 40, 20) == 1

    def test_a_slot_ending_exactly_at_t_is_not_in_the_window(self) -> None:
        """Half-open at the start: the window is (t, t + lead], never [t, …]."""
        grid = _grid(self.PATTERN, base=self.BASE)
        # t = 07:00, lead 20 → slots ending 07:10 and 07:20, both dry.
        assert _ask(grid, self.BASE, 60, 20) == 0

    def test_a_slot_ending_exactly_at_t_plus_lead_is_in_the_window(self) -> None:
        """Closed at the end: the last slot of the promise still counts."""
        grid = _grid(self.PATTERN, base=self.BASE)
        # t = 06:50, lead 10 → the single slot ending 07:00, which is wet.
        assert _ask(grid, self.BASE, 50, 10) == 1
        # One frame earlier the same lead reaches only 06:50, which is dry.
        assert _ask(grid, self.BASE, 40, 10) == 0

    def test_an_unknown_slot_excludes_a_dry_window_but_not_a_wet_one(self) -> None:
        """Silence cannot certify a dry half hour; one wet slot needs no help."""
        # 06:50 unknown, 07:00 wet.
        grid = _grid("." * 5 + "?W" + "." * 12, base=self.BASE)
        assert _ask(grid, self.BASE, 40, 20) == 1        # wet despite the gap
        assert _ask(grid, self.BASE, 30, 20) is None     # 06:40 + 06:50: unknown

    def test_a_window_running_off_the_grid_is_excluded(self) -> None:
        grid = _grid(self.PATTERN, base=self.BASE)
        # The grid ends at 09:00; a 60-minute window from 08:30 runs past it.
        assert _ask(grid, self.BASE, 150, 60) is None
        assert _ask(grid, self.BASE, -30, 20) is None


# ---------------------------------------------------------------------------
# Layer B, against arithmetic done on paper
# ---------------------------------------------------------------------------


class TestLayerB:
    """78 rows, 9 of them wet — every score below is worked out in the test."""

    def test_the_pooled_scores_match_the_hand_computation(self, corpus: Path) -> None:
        payload = _run(corpus, "--layers", "b")
        pooled = payload["layer_b"]["leads"]["20"]["all"]["baseline"]

        # 13 frames × 2 stations × 3 days, every window inside the grid.
        assert pooled["n"] == FRAMES_PER_DAY * len(STATIONS) * len(DAYS) == 78
        # Every score is reported to six decimals, so the tolerance below
        # is the report's rounding and nothing else.
        close = lambda value, expected: value == pytest.approx(expected, abs=1e-6)
        # Station A is wet-within-20 at 07:40, 07:50 and 08:00; B never is.
        assert close(pooled["base_rate"], 9 / 78)
        # Σ(p − y)² per day, station A then station B:
        #   2·0.125² + 0.875² + 10·0.125²  +  0.5² + 12·0.125²  = 1.390625
        per_day = 2 * 0.125 ** 2 + 0.875 ** 2 + 10 * 0.125 ** 2 + (
            0.5 ** 2 + 12 * 0.125 ** 2
        )
        assert per_day == 1.390625
        assert close(pooled["brier"], 3 * per_day / 78)
        assert close(pooled["uncertainty"], (9 * 69) / 78 ** 2)
        assert close(
            pooled["bss"], 1 - (3 * per_day / 78) / ((9 * 69) / 78 ** 2),
        )
        # Ranked pairs: 6 positives at 0.875 beat all 69 negatives; the 3
        # positives at 0.125 tie with 66 negatives and lose to 3 at 0.5.
        assert close(pooled["roc_auc"], (6 * 69 + 0.5 * 3 * 66) / (9 * 69))
        # Average precision: 6/9 recall at precision 1, then the 0.125
        # step adds the last 3/9 of recall at precision 9/78.
        assert close(pooled["pr_auc"], 2 / 3 + (1 / 3) * (9 / 78))
        # REL − RES + UNC is the Brier of the BINNED forecast, and the
        # fixture's three probabilities each sit alone in a bin, so it
        # equals the raw Brier exactly here.
        assert close(pooled["brier_binned"], pooled["brier"])

    def test_the_reliability_table_carries_one_row_per_occupied_bin(
        self, corpus: Path,
    ) -> None:
        payload = _run(corpus, "--layers", "b")
        table = payload["layer_b"]["leads"]["20"]["all"]["baseline"]["reliability_table"]
        # 0.125 → bin 1, 0.5 → bin 5, 0.875 → bin 8. Nothing else is
        # occupied, and none of the three sits on a bin edge.
        assert [row["bin"] for row in table] == [1, 5, 8]
        by_bin = {row["bin"]: row for row in table}
        assert by_bin[8]["n"] == 6 and by_bin[8]["observed"] == pytest.approx(1.0)
        assert by_bin[5]["n"] == 3 and by_bin[5]["observed"] == pytest.approx(0.0)
        assert by_bin[1]["n"] == 69
        assert by_bin[1]["observed"] == pytest.approx(3 / 69, abs=1e-6)

    def test_every_season_is_scored_on_its_own_month(self, corpus: Path) -> None:
        payload = _run(corpus, "--layers", "b")
        strata = payload["layer_b"]["leads"]["20"]
        for season in ("summer", "winter", "shoulder"):
            scores = strata[season]["baseline"]
            # One day each, so a third of the pool — and the same shape of
            # day, so the same BSS.
            assert scores["n"] == 26
            assert scores["days"] == 1
            assert scores["bss"] == pytest.approx(
                strata["all"]["baseline"]["bss"], abs=1e-6,
            )

    def test_a_candidate_identical_to_the_baseline_moves_nothing(
        self, corpus: Path,
    ) -> None:
        """The null experiment: same rows twice, zero difference, zero width."""
        payload = _run(
            corpus, "--layers", "b", "--candidate", str(corpus / "replay"),
        )
        for lead in ("20", "30"):
            entry = payload["layer_b"]["leads"][lead]["all"]
            assert entry["candidate"] == entry["baseline"]
            difference = entry["difference"]
            assert difference["bss"] == [0.0, 0.0, 0.0]
            assert difference["pr_auc"] == [0.0, 0.0, 0.0]
            # A zero-width interval straddles zero: no evidence of a change,
            # which is exactly right for a run against itself.
            assert difference["bss_excludes_zero"] is False

    def test_the_candidate_arm_can_score_its_own_probability_column(
        self, corpus: Path,
    ) -> None:
        """A post-processed copy (p_post) against the original's p_rain, in one A/B.

        ``--probability-column`` applies to both arms, so scoring a
        written-back copy through it would compare the copy with itself.
        ``--candidate-probability-column`` names the candidate's column
        alone; the baseline keeps reading ``p_rain_{lead}``.
        """
        import shutil

        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        post = corpus / "replay_post"
        shutil.copytree(corpus / "replay", post)
        for path in sorted((post / "decisions").glob("*.parquet")):
            table = pq.read_table(path)
            for lead in ("20", "30"):
                shifted = pc.min_element_wise(
                    pc.add(table.column(f"p_rain_{lead}"), 0.3), 1.0,
                )
                table = table.append_column(f"p_post_{lead}", shifted)
            pq.write_table(table, path)

        payload = _run(
            corpus, "--layers", "b", "--candidate", str(post),
            "--candidate-probability-column", "p_post_{lead}",
        )
        assert payload["settings"]["probability_column"] == "p_rain_{lead}"
        assert payload["settings"]["candidate_probability_column"] == "p_post_{lead}"
        entry = payload["layer_b"]["leads"]["20"]["all"]
        assert entry["candidate"]["brier"] != entry["baseline"]["brier"]

    def test_a_dead_gauge_leaves_the_pool_and_is_named(self, corpus: Path) -> None:
        """06120 reports 75 slots and is never wet — the 06080 pattern."""
        payload = _run(corpus, "--layers", "b", "--min-known-slots", "40")
        assert [row["station_id"] for row in payload["dead_gauges"]] == [STATION_B]
        assert payload["dead_gauges"][0]["known_slots"] == 75
        assert payload["dead_gauges"][0]["wet_slots"] == 0
        pooled = payload["layer_b"]["leads"]["20"]["all"]["baseline"]
        # Station A alone: 13 frames × 3 days, still 9 of them wet.
        assert pooled["n"] == FRAMES_PER_DAY * len(DAYS) == 39
        assert pooled["base_rate"] == pytest.approx(9 / 39, abs=1e-6)
        markdown = (corpus / "out" / "benchmark.md").read_text()
        assert f"{STATION_B} (75 known slots)" in markdown

    def test_the_default_floor_keeps_a_short_dry_record(self, corpus: Path) -> None:
        """75 known slots is a quiet week, not a broken bucket."""
        payload = _run(corpus, "--layers", "b")
        assert payload["dead_gauges"] == []


# ---------------------------------------------------------------------------
# Layer C, and the leave-one-month-out refit
# ---------------------------------------------------------------------------


class TestLayerC:
    def test_every_fold_refits_on_the_other_months_and_picks_70(
        self, corpus: Path,
    ) -> None:
        """The constructed answer: 70 % buys the hit without the false alarm.

        At 30 % or 50 % the rule also fires on 06120's 0.5 spike over a
        gauge that never rains — one false alarm per training day, F1 ⅔.
        At 70 % it fires only on 06180, F1 1.0. The plateau is the single
        cell at 70 and its midpoint is 70.
        """
        payload = _run(corpus, "--layers", "c")
        folds = payload["layer_c"]["leads"]["20"]["folds"]
        assert [fold["month"] for fold in folds] == ["2026-01", "2026-04", "2026-06"]
        assert [fold["season"] for fold in folds] == ["winter", "shoulder", "summer"]
        assert [fold["threshold_pct"] for fold in folds] == [70, 70, 70]
        assert all(fold["fitted"] for fold in folds)
        assert all(fold["plateau"] == [70, 70] for fold in folds)
        # Two months trained on, one held out — never the same month.
        for fold in folds:
            assert fold["train_rows"] == 2 * FRAMES_PER_DAY * len(STATIONS)
            assert fold["test_rows"] == FRAMES_PER_DAY * len(STATIONS)

    def test_the_out_of_fold_counts_pool_rather_than_average(
        self, corpus: Path,
    ) -> None:
        payload = _run(corpus, "--layers", "c")
        pooled = payload["layer_c"]["leads"]["20"]["out_of_fold"]
        # One warning and one onset per held-out day, claimed each time.
        assert pooled["hits"] == 3
        assert pooled["false_alarms"] == 0
        assert pooled["misses"] == 0
        assert pooled["late"] == 0
        assert pooled["precision"] == pytest.approx(1.0)
        assert pooled["recall"] == pytest.approx(1.0)
        assert pooled["f1"] == pytest.approx(1.0)
        assert pooled["csi"] == pytest.approx(1.0)
        # ETA 20 min, rain 20 min later: the promise was exactly kept.
        assert pooled["lead_error_p50"] == pytest.approx(0.0)
        # Six station-days of warnings, three warnings.
        assert pooled["warnings_per_station_day"] == pytest.approx(3 / 6)

    def test_each_season_carries_its_own_held_out_month(self, corpus: Path) -> None:
        payload = _run(corpus, "--layers", "c")
        seasons = payload["layer_c"]["leads"]["20"]["by_season"]
        assert sorted(seasons) == ["shoulder", "summer", "winter"]
        assert all(block["hits"] == 1 for block in seasons.values())

    def test_a_candidate_identical_to_the_baseline_gives_a_zero_width_ci(
        self, corpus: Path,
    ) -> None:
        payload = _run(
            corpus, "--layers", "c", "--candidate", str(corpus / "replay"),
        )
        for lead in ("20", "30"):
            base = payload["layer_c"]["leads"][lead]["out_of_fold"]
            cand = payload["layer_c_candidate"]["leads"][lead]["out_of_fold"]
            assert base == cand
            difference = payload["layer_c_difference"][lead]
            assert difference["f1"] == [0.0, 0.0, 0.0]
            assert difference["f1_excludes_zero"] is False

    def test_a_forced_low_threshold_shows_the_false_alarms_it_buys(
        self, corpus: Path,
    ) -> None:
        """A one-cell grid at 30 % — every fold must pick it, warts and all.

        06120's 0.5 spike now crosses, once per day over a gauge that never
        rains. The pooled out-of-fold numbers must show three hits AND
        three false alarms: precision ½, recall 1, F1 ⅔, CSI ½. If the
        accounting pooled per-fold rates instead of counts, or lost the
        false alarms to the pending bucket, none of these would hold.
        """
        payload = _run(corpus, "--layers", "c", "--thresholds", "30")
        assert [
            fold["threshold_pct"]
            for fold in payload["layer_c"]["leads"]["20"]["folds"]
        ] == [30, 30, 30]
        pooled = payload["layer_c"]["leads"]["20"]["out_of_fold"]
        assert (pooled["hits"], pooled["false_alarms"], pooled["misses"]) == (3, 3, 0)
        assert pooled["pending"] == 0
        assert pooled["precision"] == pytest.approx(0.5)
        assert pooled["recall"] == pytest.approx(1.0)
        assert pooled["f1"] == pytest.approx(2 / 3)
        assert pooled["csi"] == pytest.approx(0.5)
        # Six warnings over six station-days (2 stations × 3 held-out days).
        assert pooled["warnings_per_station_day"] == pytest.approx(1.0)

    def test_the_per_day_blocks_are_not_serialised(self, corpus: Path) -> None:
        """They exist for the bootstrap; a reader wants the scores."""
        payload = _run(corpus, "--layers", "c")
        assert "per_day" not in payload["layer_c"]["leads"]["20"]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_the_report_carries_the_settings_the_rows_were_made_under(
        self, corpus: Path,
    ) -> None:
        payload = _run(corpus, "--layers", "b")
        settings = payload["baseline"]["settings"]
        assert settings["available"] is True
        assert settings["ensemble_size"] == 16
        assert settings["n_cascade_levels"] == 6
        assert settings["flow"] == {"completion": "gated"}
        markdown = (corpus / "out" / "benchmark.md").read_text()
        assert "| baseline |" in markdown

    def test_a_settings_mismatch_between_two_runs_is_reported_loudly(
        self, tmp_path: Path,
    ) -> None:
        """A 16-member run against a 24-member one is not an A/B."""
        _write_gauge(tmp_path / "corpus")
        _write_summary(_write_decisions(tmp_path / "replay"), ensemble_size=16)
        _write_summary(_write_decisions(tmp_path / "other"), ensemble_size=24)
        payload = _run(
            tmp_path, "--layers", "b", "--candidate", str(tmp_path / "other"),
        )
        assert payload["parity_problems"] == [
            "ensemble_size: baseline 16 vs candidate 24",
        ]
        markdown = (tmp_path / "out" / "benchmark.md").read_text()
        assert "The two runs are not at parity" in markdown

    def test_a_different_anchor_policy_fails_parity_unless_it_is_allowed(
        self, tmp_path: Path,
    ) -> None:
        """L3's whole point is a different anchor — say so, or it is a bug.

        A fresher frame changes every probability in the table. Left
        unnamed it is an unexplained difference; named with
        ``--allow-differing anchor`` it is the experiment.
        """
        _write_gauge(tmp_path / "corpus")
        _write_summary(_write_decisions(tmp_path / "replay"))
        _write_summary(
            _write_decisions(tmp_path / "other"),
            anchor=_anchor_block(
                "freshest", harmonisation_sha="deadbeefcafe0001",
            ),
        )
        payload = _run(
            tmp_path, "--layers", "b", "--candidate", str(tmp_path / "other"),
        )
        assert payload["parity_problems"] == [
            "anchor.policy: baseline 'fullRange' vs candidate 'freshest'",
            "anchor.harmonisation: baseline None vs candidate "
            "'deadbeefcafe0001'",
        ]
        assert "The two runs are not at parity" in (
            tmp_path / "out" / "benchmark.md"
        ).read_text()

        allowed = _run(
            tmp_path, "--layers", "b", "--candidate", str(tmp_path / "other"),
            "--allow-differing", "anchor",
        )
        assert allowed["parity_problems"] == []
        assert allowed["deliberate_differences"] == [
            "anchor.policy: baseline 'fullRange' vs candidate 'freshest'",
            "anchor.harmonisation: baseline None vs candidate "
            "'deadbeefcafe0001'",
        ]
        markdown = (tmp_path / "out" / "benchmark.md").read_text()
        assert "The candidate difference under test" in markdown
        assert "The two runs are not at parity" not in markdown
        # The runs table names the policy each side stood on.
        assert "| fullRange |" in markdown
        assert "| freshest |" in markdown

    def test_allowing_the_anchor_does_not_excuse_anything_else(
        self, tmp_path: Path,
    ) -> None:
        _write_gauge(tmp_path / "corpus")
        _write_summary(_write_decisions(tmp_path / "replay"), ensemble_size=16)
        _write_summary(
            _write_decisions(tmp_path / "other"), ensemble_size=24,
            anchor=_anchor_block("freshest"),
        )
        payload = _run(
            tmp_path, "--layers", "b", "--candidate", str(tmp_path / "other"),
            "--allow-differing", "anchor",
        )
        assert payload["parity_problems"] == [
            "ensemble_size: baseline 16 vs candidate 24",
        ]

    def test_an_unknown_allow_differing_name_is_refused(
        self, tmp_path: Path,
    ) -> None:
        """A typo must not silently disable the parity check."""
        with pytest.raises(ValueError, match="unknown"):
            report_module.parse_allow_differing("anchr")
        assert report_module.parse_allow_differing("") == ()
        assert report_module.parse_allow_differing("anchor") == ("anchor",)

    def test_rows_with_no_summary_beside_them_are_not_assumed_at_parity(
        self, tmp_path: Path,
    ) -> None:
        _write_gauge(tmp_path / "corpus")
        _write_decisions(tmp_path / "replay")  # no summary.json written
        payload = _run(tmp_path, "--layers", "b")
        assert payload["baseline"]["settings"]["available"] is False


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------


def test_reliability_table_follows_the_project_wide_binning_rule() -> None:
    """Bin k is [k/10, (k+1)/10), and p == 1 folds into the last bin."""
    p = numpy.array([0.0, 0.0999, 0.1, 0.95, 1.0])
    y = numpy.array([0.0, 1.0, 0.0, 1.0, 1.0])
    table = report_module.reliability_table(p, y)
    assert [row["bin"] for row in table] == [0, 1, 9]
    assert {row["bin"]: row["n"] for row in table} == {0: 2, 1: 1, 9: 2}


def test_excludes_zero_reads_an_interval_the_way_the_gate_does() -> None:
    assert report_module._excludes_zero((0.02, 0.01, 0.03)) is True
    assert report_module._excludes_zero((-0.02, -0.03, -0.01)) is True
    assert report_module._excludes_zero((0.02, -0.01, 0.05)) is False
    assert report_module._excludes_zero((0.0, float("nan"), 0.1)) is None
    assert report_module._excludes_zero(None) is None


def test_f1_of_pools_the_day_counts_instead_of_averaging_them() -> None:
    """A quiet day must not weigh as much as a frontal one."""
    quiet = {"hits": 1, "late": 0, "false_alarms": 0, "misses": 0}      # F1 1.0
    busy = {"hits": 10, "late": 0, "false_alarms": 30, "misses": 10}    # F1 0.33…
    pooled = report_module._f1_of([quiet, busy])
    # 11 hits, 30 false alarms, 10 misses → P = 11/41, R = 11/21.
    precision, recall = 11 / 41, 11 / 21
    assert pooled == pytest.approx(
        2 * precision * recall / (precision + recall)
    )
    assert pooled != pytest.approx((1.0 + report_module._f1_of([busy])) / 2)


def test_layer_c_day_blocks_count_a_missed_onset(corpus: Path) -> None:
    """A missed onset lands in its day's `misses` (the 2026-09-11 KeyError).

    Every onset the sweep leaves unwarned is bumped with the outcome name
    `miss`, which the block map translates to the `misses` count; passing
    the count name instead raised KeyError on the first real fold.
    """
    payload = _run(corpus, "--layers", "c", "--thresholds", "90:90:10")
    pooled = payload["layer_c"]["leads"]["20"]["out_of_fold"]
    assert pooled["misses"] > 0
