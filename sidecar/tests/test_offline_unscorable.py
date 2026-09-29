"""Review R5 in the offline studies: a warning that claimed nothing over a
gauge hole is ``unscorable`` — out of every rate, counted on its own — in
every script that grades warnings, exactly as the page and the nightly
sweep grade them.

One shared fixture: ``test_eta_revision_study``'s station A, whose second
push (sent 04:25, lead 30, window (04:25, 05:05], dry lead-in (03:25,
04:25]) claims nothing. With a complete gauge record it is a false alarm;
with one unreported slot at 04:40 it is unscorable. The hit is untouched
either way — hits cannot move by design.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import combined_rule_study as crs  # noqa: E402
import eta_revision_study as ers  # noqa: E402
import no_eta_study as nes  # noqa: E402
import test_eta_revision_study as fx  # noqa: E402

from dmi_nowcast_core.warning_score import KnownGrid, pooled_summary  # noqa: E402

LEAD = fx.LEAD
HOLE = fx._at(4, 40)


def _grid(hole: datetime | None) -> KnownGrid:
    """Every 10-minute slot from T0 − 2 h to T0 + 9 h reported, but ``hole``."""
    first = fx.T0 - timedelta(hours=2)
    n = 11 * 6 + 1
    known = np.ones(n, bool)
    if hole is not None:
        known[int((hole - first).total_seconds()) // 600] = False
    return KnownGrid(int(first.timestamp()), 600, known)


def _eta(grid):
    return ers.analyse_station(
        fx.STATION_A, fx.ONSETS_A, fx.KNOWN_UNTIL, [LEAD], fx.THR,
        gauge_known=grid,
    )[LEAD]


def test_eta_revision_study_grades_the_hole_unscorable():
    whole, holed = _eta(_grid(None)), _eta(_grid(HOLE))
    assert [w["outcome"] for w in whole["warnings"]] == ["hit", "false_alarm"]
    assert [w["outcome"] for w in holed["warnings"]] == ["hit", "unscorable"]
    a, b = whole["score"].summary, holed["score"].summary
    assert (a["hits"], a["false_alarms"], a["unscorable"], a["warnings"]) == (1, 1, 0, 2)
    assert (b["hits"], b["false_alarms"], b["unscorable"], b["warnings"]) == (1, 0, 1, 1)
    # Without a grid nothing is unscorable: the pre-R5 grading.
    legacy = ers.analyse_station(fx.STATION_A, fx.ONSETS_A, fx.KNOWN_UNTIL, [LEAD], fx.THR)
    assert legacy[LEAD]["score"].summary["false_alarms"] == 1

    report = ers.aggregate({"A": {LEAD: holed}}, [LEAD], fx.THR, 10)
    gate = report["leads"][str(LEAD)]["gate"]
    assert gate["unscorable"] == 1 and gate["false_alarms"] == 0
    # The markdown carries the new column.
    report.update(
        generated_at_utc="x", runtime_s=0.0,
        sanity_gate={"passed": False, "rows": []},
    )
    assert "| unscorable |" in ers.render_markdown(report)


def _crs_arrays():
    a = dict(fx.STATION_A)
    a[f"o{LEAD}"] = np.full(fx.N, np.nan)
    return a


def test_combined_rule_study_counts_unscorable_in_its_own_slot():
    arrays = _crs_arrays()
    day0 = ers.to_us(fx.T0)
    cell = (LEAD, "served", 0, fx.THR[LEAD])
    out = {}
    for name, grid in (("whole", _grid(None)), ("holed", _grid(HOLE)), ("none", None)):
        out[name] = crs.score_station(
            arrays, fx.ONSETS_A, fx.KNOWN_UNTIL, [cell], day0, 1, gauge_known=grid,
        )[cell].sum(0)
    assert out["whole"].shape == (crs.N_SLOTS,) == (6,)
    m_whole, m_holed = crs.metrics(out["whole"]), crs.metrics(out["holed"])
    assert (m_whole["false_alarms"], m_whole["unscorable"]) == (1, 0)
    assert (m_holed["false_alarms"], m_holed["unscorable"]) == (0, 1)
    assert m_holed["hits"] == m_whole["hits"] == 1
    assert m_holed["warnings"] == m_whole["warnings"] - 1
    assert out["none"].tolist() == out["whole"].tolist()
    # The five-slot shape older callers (and tests) hand in still reads.
    assert crs.metrics(np.array([1, 0, 2, 0, 3]))["unscorable"] == 0


def test_combined_rule_study_gate_values_are_the_r5_ones():
    # Pre-R5 minus the measured unscorable counts (see the constants).
    assert crs.SWEEP_EXPECTED[20]["warnings"] == 15501 - 405
    assert crs.SWEEP_EXPECTED[30]["warnings"] == 16282 - 429
    assert crs.ONSET_EXPECTED["false_alarms"] == 6638 - 267
    assert crs.ONSET_EXPECTED["hits"] == 3111


def test_no_eta_study_counts_unscorable():
    arrays = dict(fx.STATION_A)
    runs = ers.run_ids(arrays["radar"])
    radar_dt = [ers.to_dt(v) for v in arrays["radar"]]
    gen_dt = [ers.to_dt(v) for v in arrays["gen"]]
    got = {}
    for name, grid in (("whole", _grid(None)), ("holed", _grid(HOLE))):
        res = nes.score_cell_station(
            arrays, fx.ONSETS_A, fx.KNOWN_UNTIL, LEAD, fx.THR[LEAD], "served", 0,
            runs=runs, radar_dt=radar_dt, gen_dt=gen_dt, want_pushes=True,
            gauge_known=grid,
        )
        got[name] = nes.pooled_counts({"A": res})
        if name == "holed":
            assert [o for _i, o, _on in res["pushes"]] == ["hit", "unscorable"]
    assert (got["whole"]["false_alarms"], got["whole"]["unscorable"]) == (1, 0)
    assert (got["holed"]["false_alarms"], got["holed"]["unscorable"]) == (0, 1)
    # A five-slot count vector (a stored result from before R5) still pools.
    assert nes.pooled_counts({"x": {"counts": {0: [1, 0, 2, 0, 3]}}})["false_alarms"] == 2


# ---------------------------------------------------------------------------
# replay_warnings: the grid is built from the merged slot list
# ---------------------------------------------------------------------------


def test_replay_warnings_known_grid_and_unscorable():
    import replay_warnings as rw

    base = datetime(2026, 9, 5, 6, 0, tzinfo=timezone.utc)
    slots = [(base + timedelta(minutes=10 * k), False, 0.0) for k in range(-12, 13)]
    grid = rw.known_grid_of(slots)
    assert grid.all_known(base - timedelta(hours=1), base + timedelta(hours=1))
    assert rw.known_grid_of([]) is None
    holed = [
        (ts, None if ts == base + timedelta(minutes=30) else wet, mm)
        for ts, wet, mm in slots
    ]
    assert not rw.known_grid_of(holed).all_known(base, base + timedelta(hours=1))

    decisions = [{
        "radar_ts": base - timedelta(minutes=14), "generated_at": base,
        "station_id": "06180", "p_rain": 0.8, "eta_min": 25.0,
        "action": "notify", "observed_mm_h": 0.0,
    }]
    point = [rw.StationPoint("06180", 55.33, 10.32)]
    kwargs = dict(lead_min=30, tolerance_min=10, dry_min=60, threshold_mm_h=0.5)
    whole, _, _ = rw.score(decisions, [{"06180": slots}], point, **kwargs)
    hole, _, _ = rw.score(decisions, [{"06180": holed}], point, **kwargs)
    assert (whole["06180"].summary["false_alarms"], whole["06180"].summary["unscorable"]) == (1, 0)
    assert (hole["06180"].summary["false_alarms"], hole["06180"].summary["unscorable"]) == (0, 1)


# ---------------------------------------------------------------------------
# benchmark_report Layer C and onset_sensitivity: the shared payload's grids
# ---------------------------------------------------------------------------


def _shared(grid):
    return {
        "leads": [LEAD], "stations": ["A"], "tracks": {"A": object()},
        "persistence_obs": 1, "rearm_after_min": 60, "raining_now_mm_h": 0.5,
        "onsets": {"A": list(fx.ONSETS_A)}, "tolerance_min": 10, "dry_min": 60,
        "onset_min_mm": 0.2, "known_until": {"A": fx.KNOWN_UNTIL},
        "coverage": {LEAD: {"A": [(fx.T0, fx.T0 + timedelta(hours=8))]}},
        "min_useful_lead_min": 5.0, "station_days": 1,
        "known_grids": {} if grid is None else {"A": grid},
    }


PUSHES = [(fx._at(0, 45), 20.0), (fx._at(4, 25), 30.0)]


def test_benchmark_score_fold_holds_unscorable_out_of_every_block(monkeypatch):
    import benchmark_report as br

    monkeypatch.setattr(br, "replay_station", lambda *a, **k: list(PUSHES))
    for grid, fa, unsc in ((_grid(None), 1, 0), (_grid(HOLE), 0, 1), (None, 1, 0)):
        results, per_day, n_sent = br.score_fold(_shared(grid), LEAD, 45)
        pooled = pooled_summary(results)
        assert (pooled["false_alarms"], pooled["unscorable"], pooled["hits"]) == (fa, unsc, 1)
        assert sum(d["false_alarms"] for d in per_day.values()) == fa
        assert n_sent == 2
        row = br._summary_row(pooled, n_sent, 1)
        assert row["unscorable"] == unsc


def test_onset_sensitivity_scores_with_the_grid():
    import onset_sensitivity as sens

    variant = sens._variant("V2")
    for grid, fa, unsc in ((_grid(None), 1, 0), (_grid(HOLE), 0, 1)):
        shared = _shared(grid)
        shared["thresholds"] = {LEAD: 45}
        cell = sens.score_variant_lead(
            shared, variant, LEAD, {"A": PUSHES},
            {"A": {o: 1.0 for o in fx.ONSETS_A}},
        )
        assert (cell["false_alarms"], cell["unscorable"], cell["hits"]) == (fa, unsc, 1)
