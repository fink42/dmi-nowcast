"""The ETA revision study (``scripts/eta_revision_study.py``).

Synthetic and offline. Two stations, six hours at the 10-minute radar
cadence, ``generated_at`` 15 minutes after ``radar_ts``, lead 30 at 45 %:

* **A** — push at row 3 (sent 00:45, ETA 20) and the gauge onset at 01:10:
  a HIT with push error (00:45 + 20) − 01:10 = −5 min. Row 4 dips below
  the threshold (a wrong all-clear) with ETA 12 (error −3); row 5 is back
  over with ETA 6 (error +1) and is the last row before the onset, so the
  arrival estimate moved 6 min and got closer. Rows 6–8 are raining.
  Re-armed at row 12; row 13 is over threshold with 2 mm/h observed —
  already raining, consumed silently. Row 16 goes over while disarmed and
  restarts the dry clock, so the arm returns at row 23, and row 25 (sent
  04:25, ETA 30) pushes into a dry gauge: a FALSE ALARM, below threshold
  on rows 26 and 27 (a right all-clear, single at +10, strict at +20).
* **B** — push at row 2 with NO ETA (sent 00:35), onset 00:50: a hit
  without a push ETA; row 3 carries ETA 3 (error −2). A second onset at
  04:00 is never warned about: a miss.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import eta_revision_study as study  # noqa: E402

from dmi_nowcast_sidecar.threshold_sweep import (  # noqa: E402
    build_tracks,
    replay_station,
)
from dmi_nowcast_core.warning_score import pooled_summary  # noqa: E402

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
N = 36
LEAD = 30
THR = {LEAD: 45}
NAN = float("nan")


def _arrays(p: dict[int, float], eta: dict[int, float], obs: dict[int, float],
            base_p: float = 0.1) -> dict:
    radar = np.array([study.to_us(T0 + timedelta(minutes=10 * i)) for i in range(N)], np.int64)
    pp = np.full(N, base_p)
    for i, v in p.items():
        pp[i] = v
    ee = np.full(N, NAN)
    for i, v in eta.items():
        ee[i] = v
    oo = np.zeros(N)
    for i, v in obs.items():
        oo[i] = v
    return {
        "radar": radar,
        "gen": radar + 15 * study.US_PER_MIN,
        "eta": ee,
        "intensity": np.full(N, NAN),
        "observed": oo,
        "forecast": np.full(N, NAN),
        f"p{LEAD}": pp,
    }


def _at(h: int, m: int) -> datetime:
    return T0 + timedelta(hours=h, minutes=m)


STATION_A = _arrays(
    p={3: 0.6, 4: 0.3, 5: 0.6, 13: 0.7, 16: 0.9,
       25: 0.8, 26: 0.2, 27: 0.2},
    eta={3: 20.0, 4: 12.0, 5: 6.0, 25: 30.0},
    obs={6: 1.0, 7: 1.0, 8: 1.0, 13: 2.0},
)
ONSETS_A = [_at(1, 10)]
STATION_B = _arrays(p={2: 0.5, 3: 0.5}, eta={3: 3.0}, obs={})
ONSETS_B = [_at(0, 50), _at(4, 0)]
KNOWN_UNTIL = _at(7, 0)


def _run():
    return {
        "A": study.analyse_station(STATION_A, ONSETS_A, KNOWN_UNTIL, [LEAD], THR),
        "B": study.analyse_station(STATION_B, ONSETS_B, KNOWN_UNTIL, [LEAD], THR),
    }


def test_pushes_match_the_sweep_replay():
    """The row indices that fire are the sweep's own warnings."""
    for arrays in (STATION_A, STATION_B):
        rows = [
            {
                "radar_ts": study.to_dt(arrays["radar"][i]),
                "generated_at": study.to_dt(arrays["gen"][i]),
                "station_id": "X",
                "eta_min": None if np.isnan(arrays["eta"][i]) else float(arrays["eta"][i]),
                "observed_mm_h": float(arrays["observed"][i]),
                f"p_post_{LEAD}": float(arrays[f"p{LEAD}"][i]),
            }
            for i in range(N)
        ]
        tracks, _ = build_tracks(rows, [LEAD], column_for=lambda lead: f"p_post_{lead}")
        expected = replay_station(tracks["X"], 0, THR[LEAD], persistence_obs=1, rearm_after_min=60)
        pushes, _ = study.replay_pushes(arrays, LEAD, THR[LEAD])
        assert [study.to_dt(arrays["gen"][i]) for i in pushes] == [w[0] for w in expected]


def test_push_rearm_and_already_raining():
    pushes, consumed = study.replay_pushes(STATION_A, LEAD, THR[LEAD])
    # Row 13 would fire (re-armed at row 12) but is already raining; row 16
    # is disarmed and restarts the dry clock, so row 25 is the next push.
    assert pushes == [3, 25]
    assert consumed == 1
    pushes_b, consumed_b = study.replay_pushes(STATION_B, LEAD, THR[LEAD])
    assert pushes_b == [2] and consumed_b == 0


def test_counts_errors_and_reversals():
    res = _run()
    pooled = pooled_summary([res[s][LEAD]["score"] for s in res])
    assert (pooled["n_sent"], pooled["hits"], pooled["false_alarms"],
            pooled["late"], pooled["misses"]) == (3, 2, 1, 0, 1)

    a = res["A"][LEAD]
    hit, fa = a["warnings"]
    assert hit["outcome"] == "hit" and fa["outcome"] == "false_alarm"
    # The error value: (00:45 + 20 min) − 01:10 = −5 min at the push row.
    assert hit["err0"] == pytest.approx(-5.0)
    assert hit["err_last"] == pytest.approx(1.0)
    assert a["error_rows"]["error"].tolist() == pytest.approx([-5.0, -3.0, 1.0])
    assert a["error_rows"]["since_push"].tolist() == pytest.approx([0.0, 10.0, 20.0])
    # A wrong all-clear: row 4, 10 min after the push, 15 min before onset.
    assert hit["first_below_min"] == pytest.approx(10.0)
    assert hit["first_below_to_onset_min"] == pytest.approx(15.0)
    assert hit["first_below2_min"] is None
    # A right all-clear on the false alarm, single at +10, strict at +20.
    assert fa["first_below_min"] == pytest.approx(10.0)
    assert fa["first_below2_min"] == pytest.approx(20.0)

    b_hit = res["B"][LEAD]["warnings"][0]
    assert b_hit["outcome"] == "hit" and b_hit["eta0"] is None
    assert b_hit["err0"] is None and b_hit["err_last"] == pytest.approx(-2.0)
    assert b_hit["first_below_min"] is None


def test_aggregate_and_gate():
    res = _run()
    report = study.aggregate(res, [LEAD], THR, 10)
    e = report["leads"][str(LEAD)]
    assert e["gate"]["hits"] == 2 and e["gate"]["warnings"] == 3
    assert e["already_raining_consumed"] == 1
    rv = e["A_revision"]
    assert rv["hits_with_push_eta"] == 1
    assert rv["hits_without_push_eta"] == 1
    assert rv["hits_without_push_eta_later_eta"] == 1
    assert rv["n_diff_ge5"] == 1 and rv["n_diff_ge10"] == 0
    assert rv["share_last_closer"] == 1.0
    since = {r["bucket"]: r for r in e["A_since_push"]}
    assert since["0 (push row)"]["n"] == 1
    assert since["(5, 10]"]["n"] == 2          # A row 4 and B row 3
    assert since["(15, 20]"]["median_err"] == pytest.approx(1.0)
    single = e["B_reversals"]["single"]
    assert (single["hits"]["reversed"], single["false_alarms"]["reversed"]) == (1, 1)
    assert single["hits"]["reversed_before_slot_start"] == 1
    strict = e["B_reversals"]["two_consecutive"]
    assert (strict["hits"]["reversed"], strict["false_alarms"]["reversed"]) == (0, 1)
    assert e["C_no_eta"]["all_sent"]["no_eta"] == 1
    fa_rows = {r["bucket"]: r["n"] for r in e["D_eta_false_alarms"]["rows"]}
    assert fa_rows["(20, 30]"] == 1

    ok, rows = study.sanity_gate(report, {LEAD: {"warnings": 3, "hits": 2}})
    assert ok and all(r["pass"] for r in rows)
    bad, _ = study.sanity_gate(report, {LEAD: {"warnings": 300, "hits": 2}})
    assert not bad

    report.update(
        sanity_gate={"passed": True, "rows": rows}, generated_at_utc="x", runtime_s=1.0,
        settings={"decisions_dir": "d"}, load={"rows": 72, "stations": 2, "duplicates": 0},
        stations_scored=2, dead_gauges=[], reading=["one line"],
    )
    md = study.render_markdown(report)
    assert "PASSED" in md and "Lead 30 min" in md and "one line" in md
