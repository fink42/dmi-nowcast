#!/usr/bin/env python3
"""Would revising a push's ETA be worth it? A measurement, not a feature.

The push rule sends ONE notification per rain event, carrying the ETA of
the cycle that fired, and never revises it. This study replays that rule
over stored decision rows, scores every warning against the rain gauges
exactly as the threshold sweep does, and then looks at what the LATER
cycles of the same station said between the push and the rain:

A. ETA accuracy vs time since the push (hits only): for the push row and
   every later row with a finite ETA before the matched onset,
   ``error = (t_k + eta_k) - onset`` in minutes. POSITIVE = the rain
   arrived before the predicted instant (the scorer's sign convention,
   ``warning_score.lead_error_min``).
B. Reversals: does the decision probability fall back below the
   threshold after the push, before the onset (hits) or before the
   scorer's window closes (false alarms)?
C. How often a push carries no ETA at all.
D. The push ETA of false alarms.

Nothing here re-implements the rule or the scoring:

* the state machine is ``push.engine.evaluate`` itself, driven exactly as
  ``threshold_sweep.replay_station`` drives it (fresh armed state at the
  head of every coverage run, rows with a null probability skipped, quiet
  hours off, already-raining at ETA <= 1.5 min or observed >= 0.5 mm/h);
  the only difference is that this loop remembers WHICH row fired;
* gauge truth is ``threshold_sweep.gauge_truth`` (the function the sweep
  and the benchmark's Layer C both use: dry 60 min, >= 0.2 mm over the
  onset slot and the next, dead gauges with >= 500 known slots and never
  wet excluded);
* warnings are graded by ``warning_score.score_warnings`` with the sweep's
  settings (tolerance 10, minimum useful lead 5, ``known_until`` and
  ``coverage`` per station) and pooled with ``pooled_summary``.

The pooled counts must reproduce the sweep's within 1 % (the sanity gate)
before any other number is written.

Offline and read-only: parquet in, ``report.md`` / ``report.json`` out.

Usage (from ``sidecar/``)::

    PYTHONPATH=../src:. .venv/bin/python ../scripts/eta_revision_study.py \\
        --decisions-dir ~/dmi-nowcast-corpus-local/stations/replay_rp_trees_all_oof/decisions \\
        --corpus-dir ~/dmi-nowcast-corpus-local \\
        --thresholds ~/dmi-nowcast-corpus-local/stations/thresholds_rp_trees_all/push_thresholds.json \\
        --out-dir ~/dmi-nowcast-corpus-local/stations/eta_revision_study \\
        --leads 20,30,45,60 --workers 10
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "sidecar"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dmi_nowcast_core.warning_score import (  # noqa: E402
    DEFAULT_COVERAGE_GAP_MIN,
    DEFAULT_DRY_MIN,
    DEFAULT_MIN_KNOWN_SLOTS,
    DEFAULT_ONSET_MIN_MM,
    DEFAULT_TOLERANCE_MIN,
    coverage_runs,
    pooled_summary,
    score_warnings,
)
from dmi_nowcast_sidecar.push import engine as decision_engine  # noqa: E402
from dmi_nowcast_sidecar.threshold_sweep import (  # noqa: E402
    FIT_MIN_USEFUL_LEAD_MIN,
    RAIN_THRESHOLD_MM_H,
)

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
US_PER_MIN = 60_000_000
#: The gauge slot length: an onset is stamped at its slot END, so the
#: first drop fell somewhere in the 10 minutes before it.
SLOT_MIN = 10

#: Expected pooled counts from the sweep at the shipped thresholds
#: (thresholds_rp_trees_all/sweep.md). ``warnings`` is the SCORED count.
SWEEP_EXPECTED = {
    20: {"warnings": 15501, "hits": 3074},
    30: {"warnings": 16282, "hits": 4274},
    45: {"warnings": 15139, "hits": 4817},
    60: {"warnings": 15131, "hits": 5165},
}
GATE_TOLERANCE = 0.01

SINCE_PUSH_EDGES = (0, 5, 10, 15, 20, 30, 45)
BEFORE_ONSET_EDGES = (0, 5, 10, 15, 20, 30, 45, 60)
ETA_EDGES = (0, 5, 10, 15, 20, 30, 45, 60, 90)

COLUMNS = {
    "decision_instant": "generated_at",
    "radar_clock": "radar_ts",
    "station": "station_id",
    "probability": "p_post_{lead}",
    "eta": "eta_min",
    "observed_rate": "observed_mm_h",
    "carried_to_engine": ["intensity_mm_h", "forecast_now_mm_h"],
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_station_arrays(
    decisions_dir: Path, leads: Sequence[int], log=print,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    """Per-station numpy arrays, sorted by ``radar_ts``.

    Duplicate ``(radar_ts, station_id)`` keys keep the LAST file's row, as
    ``threshold_sweep.load_decisions`` does.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    wanted = [
        "radar_ts", "generated_at", "station_id", "eta_min",
        "intensity_mm_h", "observed_mm_h", "forecast_now_mm_h",
    ] + [f"p_post_{lead}" for lead in leads]
    files = sorted(Path(decisions_dir).rglob("*.parquet"))
    tables = []
    for order, path in enumerate(files):
        names = set(pq.read_schema(path).names)
        cols = [c for c in wanted if c in names]
        t = pq.read_table(path, columns=cols)
        for c in wanted:
            if c not in names:
                t = t.append_column(c, pa.nulls(t.num_rows, pa.float64()))
        t = t.select(wanted)
        t = t.append_column("_file", pa.array(np.full(t.num_rows, order, np.int32)))
        tables.append(t)
    table = pa.concat_tables(tables, promote_options="default")
    n_rows_raw = table.num_rows

    def ts_us(name: str) -> np.ndarray:
        col = table.column(name).cast(pa.timestamp("us", tz="UTC"))
        arr = col.to_numpy(zero_copy_only=False)
        out = arr.astype("datetime64[us]").astype(np.int64)
        return out, np.isnat(arr)

    radar, radar_nat = ts_us("radar_ts")
    gen, gen_nat = ts_us("generated_at")
    gen = np.where(gen_nat, radar, gen)
    station = np.asarray(table.column("station_id").to_pylist(), dtype=object)

    def f64(name: str) -> np.ndarray:
        return table.column(name).cast(pa.float64()).to_numpy(zero_copy_only=False)

    fields = {
        "eta": f64("eta_min"),
        "intensity": f64("intensity_mm_h"),
        "observed": f64("observed_mm_h"),
        "forecast": f64("forecast_now_mm_h"),
    }
    for lead in leads:
        fields[f"p{lead}"] = f64(f"p_post_{lead}")
    file_order = table.column("_file").to_numpy()
    del table

    keep = ~radar_nat
    codes, stations = _encode(station)
    order = np.lexsort((file_order, radar, codes))
    order = order[keep[order]]
    # Dedup (station, radar_ts): the last file wins.
    c_s, r_s = codes[order], radar[order]
    last = np.ones(order.size, bool)
    last[:-1] = (c_s[1:] != c_s[:-1]) | (r_s[1:] != r_s[:-1])
    duplicates = int(order.size - last.sum())
    order = order[last]

    out: dict[str, dict[str, np.ndarray]] = {}
    c_o = codes[order]
    bounds = np.flatnonzero(np.diff(c_o)) + 1
    for chunk in np.split(order, bounds):
        if chunk.size == 0:
            continue
        sid = stations[codes[chunk[0]]]
        arrays = {"radar": radar[chunk], "gen": gen[chunk]}
        for name, values in fields.items():
            arrays[name] = values[chunk]
        out[sid] = arrays
    counts = {
        "files": len(files), "rows": n_rows_raw,
        "duplicates": duplicates, "stations": len(out),
    }
    log(f"loaded {n_rows_raw} rows from {len(files)} files, "
        f"{len(out)} stations, {duplicates} duplicate keys")
    return out, counts


def _encode(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    uniq = sorted({str(v) for v in values})
    lookup = {v: i for i, v in enumerate(uniq)}
    return np.fromiter((lookup[str(v)] for v in values), np.int64, values.size), uniq


def run_ids(radar_us: np.ndarray, gap_min: int = DEFAULT_COVERAGE_GAP_MIN) -> np.ndarray:
    """Coverage-run index per row: a gap > ``gap_min`` starts a new run."""
    if radar_us.size == 0:
        return np.zeros(0, np.int64)
    breaks = np.diff(radar_us) > gap_min * US_PER_MIN
    return np.concatenate([[0], np.cumsum(breaks)]).astype(np.int64)


def to_dt(us: int) -> datetime:
    return EPOCH + timedelta(microseconds=int(us))


def to_us(when: datetime) -> int:
    return (when - EPOCH) // timedelta(microseconds=1)


def _opt(value: float) -> float | None:
    return None if value != value else float(value)


# ---------------------------------------------------------------------------
# The rule — push.engine.evaluate, remembering which row fired
# ---------------------------------------------------------------------------


def replay_pushes(
    arrays: Mapping[str, np.ndarray],
    lead: int,
    threshold_pct: int,
    *,
    runs: np.ndarray | None = None,
    radar_dt: Sequence[datetime] | None = None,
    gen_dt: Sequence[datetime] | None = None,
    persistence_obs: int = decision_engine.DEFAULT_PERSISTENCE_OBS,
    rearm_after_min: int = decision_engine.DEFAULT_REARM_AFTER_MIN,
    raining_now_mm_h: float = RAIN_THRESHOLD_MM_H,
) -> tuple[list[int], int]:
    """``(row indices that notified, already-raining consumptions)``.

    Mirrors ``threshold_sweep.replay_station`` line for line: fresh
    ``INITIAL_STATE`` at the head of every coverage run, a row whose
    probability is null skipped for this lead, quiet hours off.
    """
    eng = decision_engine
    rules = eng.Rules(
        persistence_obs=int(persistence_obs),
        rearm_after_min=int(rearm_after_min),
        raining_now_mm_h=float(raining_now_mm_h),
    )
    radar = arrays["radar"]
    if runs is None:
        runs = run_ids(radar)
    if radar_dt is None:
        radar_dt = [to_dt(v) for v in radar]
    if gen_dt is None:
        gen_dt = [to_dt(v) for v in arrays["gen"]]
    p = arrays[f"p{lead}"].tolist()
    eta = arrays["eta"].tolist()
    inten = arrays["intensity"].tolist()
    obs = arrays["observed"].tolist()
    fc = arrays["forecast"].tolist()
    run_list = runs.tolist()
    state = eng.INITIAL_STATE
    run = None
    pushes: list[int] = []
    consumed = 0
    for i in range(len(p)):
        if run_list[i] != run:
            run = run_list[i]
            state = eng.INITIAL_STATE
        pi = p[i]
        if pi != pi:
            continue
        decision = eng.evaluate(
            state,
            eng.Observation(
                radar_ts_utc=radar_dt[i],
                p_rain=pi,
                eta_min=_opt(eta[i]),
                intensity_mm_h=_opt(inten[i]),
                observed_mm_h=_opt(obs[i]),
                forecast_now_mm_h=_opt(fc[i]),
            ),
            threshold_pct=int(threshold_pct),
            quiet=None,
            tz="UTC",
            now_utc=gen_dt[i],
            rules=rules,
        )
        state = decision.state
        if decision.action == "notify":
            pushes.append(i)
        elif decision.action == "already_raining":
            consumed += 1
    return pushes, consumed


# ---------------------------------------------------------------------------
# Per-station analysis
# ---------------------------------------------------------------------------


def analyse_station(
    arrays: Mapping[str, np.ndarray],
    onsets: Sequence[datetime],
    known_until: datetime | None,
    leads: Sequence[int],
    thresholds: Mapping[int, int],
    *,
    tolerance_min: int = DEFAULT_TOLERANCE_MIN,
    dry_min: int = DEFAULT_DRY_MIN,
    onset_min_mm: float = DEFAULT_ONSET_MIN_MM,
    min_useful_lead_min: float = FIT_MIN_USEFUL_LEAD_MIN,
    coverage_gap_min: int = DEFAULT_COVERAGE_GAP_MIN,
) -> dict[int, dict]:
    """Everything the report needs from one station, per lead.

    Returns ``{lead: {"score": ScoreResult, "warnings": [...],
    "error_rows": {...arrays...}, "consumed": int}}``. Each warning record
    carries its outcome, push ETA, the hit's per-row errors summary and the
    reversal findings.
    """
    radar = arrays["radar"]
    gen = arrays["gen"]
    eta = arrays["eta"]
    runs = run_ids(radar, coverage_gap_min)
    radar_dt = [to_dt(v) for v in radar]
    gen_dt = [to_dt(v) for v in gen]
    out: dict[int, dict] = {}
    for lead in leads:
        thr = int(thresholds[lead])
        pushes, consumed = replay_pushes(
            arrays, lead, thr, runs=runs, radar_dt=radar_dt, gen_dt=gen_dt,
        )
        pushes.sort(key=lambda i: (gen[i], i))
        coverage = coverage_runs(
            radar_dt, max_gap_min=coverage_gap_min,
            extend_min=lead + tolerance_min,
        )
        result = score_warnings(
            [(gen_dt[i], _opt(eta[i])) for i in pushes],
            list(onsets),
            lead_min=int(lead), tolerance_min=int(tolerance_min),
            dry_min=int(dry_min), onset_min_mm=float(onset_min_mm),
            known_until=known_until, coverage=coverage,
            min_useful_lead_min=float(min_useful_lead_min),
        )
        p = arrays[f"p{lead}"]
        valid_p = ~np.isnan(p)
        below = valid_p & (p < thr / 100.0)
        records: list[dict] = []
        err_cols: dict[str, list] = {
            "since_push": [], "before_onset": [], "error": [], "is_push": [],
        }
        for i0, w in zip(pushes, result.warnings):
            t0 = int(gen[i0])
            rec: dict[str, Any] = {
                "outcome": w.outcome,
                "eta0": _opt(eta[i0]),
            }
            if w.outcome == "pending":
                records.append(rec)
                continue
            if w.onset_utc is not None:
                onset = to_us(w.onset_utc)
                end = onset
                inclusive = False
            else:
                end = t0 + (lead + tolerance_min) * US_PER_MIN
                onset = None
                inclusive = True
            # Rows strictly after the push, before the interval end.
            later = np.arange(i0 + 1, gen.size)
            later = later[(gen[later] <= end) if inclusive else (gen[later] < end)]
            later = later[gen[later] > t0]
            # --- B: reversals --------------------------------------------
            lv = later[valid_p[later]]
            rec["n_later_rows"] = int(lv.size)
            sub = below[lv]
            if sub.any():
                j = int(np.argmax(sub))
                rec["first_below_min"] = (int(gen[lv[j]]) - t0) / US_PER_MIN
                if onset is not None:
                    rec["first_below_to_onset_min"] = (onset - int(gen[lv[j]])) / US_PER_MIN
            else:
                rec["first_below_min"] = None
            two = sub[:-1] & sub[1:] if sub.size >= 2 else np.zeros(0, bool)
            if two.any():
                j = int(np.argmax(two)) + 1
                rec["first_below2_min"] = (int(gen[lv[j]]) - t0) / US_PER_MIN
                if onset is not None:
                    rec["first_below2_to_onset_min"] = (onset - int(gen[lv[j]])) / US_PER_MIN
            else:
                rec["first_below2_min"] = None
            # --- A: ETA errors, hits only ----------------------------------
            if w.outcome == "hit":
                rows = np.concatenate([[i0], lv])
                rows = rows[~np.isnan(eta[rows])]
                errs = (gen[rows] + eta[rows] * US_PER_MIN - onset) / US_PER_MIN
                err_cols["since_push"].extend(((gen[rows] - t0) / US_PER_MIN).tolist())
                err_cols["before_onset"].extend(((onset - gen[rows]) / US_PER_MIN).tolist())
                err_cols["error"].extend(errs.tolist())
                err_cols["is_push"].extend((rows == i0).tolist())
                rec["err0"] = float(errs[0]) if rows.size and rows[0] == i0 else None
                rec["err_last"] = float(errs[-1]) if rows.size else None
                rec["last_is_push"] = bool(rows.size and rows[-1] == i0)
                rec["last_before_onset_min"] = (
                    (onset - int(gen[rows[-1]])) / US_PER_MIN if rows.size else None
                )
                rec["n_later_eta"] = int((rows != i0).sum())
            records.append(rec)
        out[lead] = {
            "score": result,
            "warnings": records,
            "error_rows": {k: np.asarray(v) for k, v in err_cols.items()},
            "consumed": consumed,
            "pushes": len(pushes),
        }
    return out


# ---------------------------------------------------------------------------
# Parallel driver
# ---------------------------------------------------------------------------

_SHARED: dict | None = None


def _work(station: str) -> tuple[str, dict]:
    s = _SHARED
    assert s is not None
    return station, analyse_station(
        s["arrays"][station],
        s["onsets"].get(station, ()),
        s["known_until"].get(station),
        s["leads"],
        s["thresholds"],
    )


def run_all(shared: dict, stations: Sequence[str], workers: int, log=print) -> dict:
    global _SHARED
    _SHARED = shared
    results: dict[str, dict] = {}
    if workers <= 1:
        for st in stations:
            results[st] = _work(st)[1]
    else:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for n, (st, res) in enumerate(pool.map(_work, stations, chunksize=1), 1):
                results[st] = res
                if n % 10 == 0:
                    log(f"  {n}/{len(stations)} stations done")
    _SHARED = None
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _q(values: np.ndarray, q: float) -> float | None:
    return None if values.size == 0 else float(np.percentile(values, q))


def _bucket_label(edges: Sequence[int], k: int, *, zero_label: str | None) -> str:
    if k == 0 and zero_label is not None:
        return zero_label
    if k == len(edges):
        return f"> {edges[-1]}"
    return f"({edges[k - 1]}, {edges[k]}]"


def error_table(values: np.ndarray, errors: np.ndarray, edges: Sequence[int],
                *, zero_label: str | None) -> list[dict]:
    """Rows of n / median signed / median |e| / p90 |e| per bucket.

    With ``zero_label`` the bucket ``values == 0`` is its own row (the push
    row itself); the others are right-closed ``(e_{k-1}, e_k]`` and a final
    open ``> e_last``.
    """
    rows = []
    idx = np.searchsorted(np.asarray(edges, float), values, side="left")
    # side=left: value == edge lands in the bucket that edge closes.
    for k in range(len(edges) + 1):
        if k == 0 and zero_label is None:
            continue
        m = idx == k
        e = errors[m]
        mid = e + SLOT_MIN / 2  # onset at slot midpoint instead of slot end
        rows.append({
            "bucket": _bucket_label(edges, k, zero_label=zero_label),
            "n": int(m.sum()),
            "median_err": _q(e, 50),
            "median_abs_err": _q(np.abs(e), 50),
            "p90_abs_err": _q(np.abs(e), 90),
            "median_err_mid": _q(mid, 50),
            "median_abs_err_mid": _q(np.abs(mid), 50),
            # The ETA lands on the ensemble's 10-min steps and the onset on
            # the gauge's 10-min slot ends, so errors are (nearly) multiples
            # of 10 and these shares say more than an interpolated median.
            "share_exact": (float((np.abs(e) < 1).mean()) if e.size else None),
            "share_within_10": (float((np.abs(e) <= 10.5).mean()) if e.size else None),
            "share_early_ge20": (float((e <= -19.5).mean()) if e.size else None),
            "share_late_ge10": (float((e >= 9.5).mean()) if e.size else None),
        })
    return rows


def aggregate(results: Mapping[str, dict], leads: Sequence[int],
              thresholds: Mapping[int, int], tolerance_min: int) -> dict:
    report: dict[str, Any] = {"leads": {}}
    for lead in leads:
        per = [results[s][lead] for s in results]
        pooled = pooled_summary([r["score"] for r in per])
        warnings = [w for r in per for w in r["warnings"]]
        er = {
            k: np.concatenate([r["error_rows"][k] for r in per]).astype(float)
            for k in ("since_push", "before_onset", "error")
        }
        is_push = np.concatenate([r["error_rows"]["is_push"] for r in per]).astype(bool)
        # The push row is its own bucket; later rows by time since push.
        since = np.where(is_push, 0.0, np.maximum(er["since_push"], 1e-9))
        table_since = error_table(since, er["error"], SINCE_PUSH_EDGES, zero_label="0 (push row)")
        table_before = error_table(er["before_onset"], er["error"], BEFORE_ONSET_EDGES, zero_label=None)

        hits = [w for w in warnings if w["outcome"] == "hit"]
        fas = [w for w in warnings if w["outcome"] == "false_alarm"]
        lates = [w for w in warnings if w["outcome"] == "late"]
        sent = warnings

        # --- A: revision summary ---------------------------------------
        with_eta0 = [h for h in hits if h.get("err0") is not None]
        revised = [h for h in with_eta0 if not h["last_is_push"]]
        diff = np.array([abs(h["err_last"] - h["err0"]) for h in with_eta0])
        e0 = np.array([abs(h["err0"]) for h in revised])
        el = np.array([abs(h["err_last"]) for h in revised])
        n0 = len(with_eta0)
        revision = {
            "hits": len(hits),
            "hits_with_push_eta": n0,
            "hits_without_push_eta": len(hits) - n0,
            "hits_without_push_eta_later_eta": sum(
                1 for h in hits if h.get("err0") is None and h.get("err_last") is not None
            ),
            "hits_with_later_eta_row": len(revised),
            "share_diff_ge5": (float((diff >= 5).mean()) if n0 else None),
            "share_diff_ge10": (float((diff >= 10).mean()) if n0 else None),
            "n_diff_ge5": int((diff >= 5).sum()),
            "n_diff_ge10": int((diff >= 10).sum()),
            "share_last_closer": (float((el < e0).mean()) if revised else None),
            "share_last_same": (float((el == e0).mean()) if revised else None),
            "share_last_worse": (float((el > e0).mean()) if revised else None),
            "median_abs_err_push": _q(e0, 50),
            "median_abs_err_last": _q(el, 50),
            "share_exact_push": (float((e0 < 1).mean()) if revised else None),
            "share_exact_last": (float((el < 1).mean()) if revised else None),
            "median_min_last_before_onset": _q(
                np.array([h["last_before_onset_min"] for h in revised], float), 50),
            "median_abs_err_push_all": _q(np.array([abs(h["err0"]) for h in with_eta0]), 50),
            "median_abs_err_last_all": _q(np.array([abs(h["err_last"]) for h in with_eta0]), 50),
        }

        # --- B: reversals ---------------------------------------------
        def reversal(group: list[dict], key: str, to_onset: str | None) -> dict:
            n = len(group)
            with_later = [g for g in group if g.get("n_later_rows", 0) > 0]
            flagged = [g for g in group if g.get(key) is not None]
            times = np.array([g[key] for g in flagged], float)
            out = {
                "n": n,
                "with_later_row": len(with_later),
                "reversed": len(flagged),
                "share": (len(flagged) / n) if n else None,
                "median_min_after_push": _q(times, 50),
            }
            if to_onset is not None:
                lead_left = np.array([g[to_onset] for g in flagged], float)
                out["median_min_before_onset"] = _q(lead_left, 50)
                # Before the onset SLOT began: certainly before the rain.
                out["reversed_before_slot_start"] = int((lead_left > SLOT_MIN).sum())
            return out

        reversals = {
            "single": {
                "hits": reversal(hits, "first_below_min", "first_below_to_onset_min"),
                "false_alarms": reversal(fas, "first_below_min", None),
                "late": reversal(lates, "first_below_min", "first_below_to_onset_min"),
            },
            "two_consecutive": {
                "hits": reversal(hits, "first_below2_min", "first_below2_to_onset_min"),
                "false_alarms": reversal(fas, "first_below2_min", None),
                "late": reversal(lates, "first_below2_min", "first_below2_to_onset_min"),
            },
        }

        # --- C: availability -------------------------------------------
        def no_eta(group):
            n = len(group)
            k = sum(1 for g in group if g["eta0"] is None)
            return {"n": n, "no_eta": k, "share": (k / n) if n else None}

        availability = {
            "all_sent": no_eta(sent),
            "hits": no_eta(hits),
            "late": no_eta(lates),
            "false_alarms": no_eta(fas),
        }

        # --- D: push ETA distribution ----------------------------------
        def eta_dist(group):
            etas = np.array([g["eta0"] for g in group if g["eta0"] is not None], float)
            idx = np.searchsorted(np.asarray(ETA_EDGES, float), etas, side="left")
            rows = []
            for k in range(1, len(ETA_EDGES) + 1):
                m = int((idx == k).sum())
                rows.append({"bucket": _bucket_label(ETA_EDGES, k, zero_label=None),
                             "n": m, "share": (m / len(group)) if group else None})
            none = sum(1 for g in group if g["eta0"] is None)
            rows.append({"bucket": "no ETA", "n": none,
                         "share": (none / len(group)) if group else None})
            return {"rows": rows, "p25": _q(etas, 25), "p50": _q(etas, 50),
                    "p75": _q(etas, 75), "n": len(group)}

        report["leads"][str(lead)] = {
            "threshold_pct": int(thresholds[lead]),
            "gate": {k: pooled[k] for k in (
                "warnings", "n_sent", "pending", "hits", "late",
                "false_alarms", "misses", "pending_onsets", "uncovered_onsets",
            )},
            "already_raining_consumed": sum(r["consumed"] for r in per),
            "pooled_lead_error_p50": pooled["lead_error_min"]["p50"],
            "A_since_push": table_since,
            "A_before_onset": table_before,
            "A_revision": revision,
            "B_reversals": reversals,
            "C_no_eta": availability,
            "D_eta_false_alarms": eta_dist(fas),
            "D_eta_hits": eta_dist(hits),
        }
    return report


def sanity_gate(report: dict, expected: Mapping[int, Mapping[str, int]]) -> tuple[bool, list[dict]]:
    ok = True
    rows = []
    for lead, exp in expected.items():
        got = report["leads"].get(str(lead))
        if got is None:
            continue
        for key, want in exp.items():
            have = got["gate"][key]
            rel = abs(have - want) / want
            passed = rel <= GATE_TOLERANCE
            ok &= passed
            rows.append({"lead": lead, "count": key, "expected": want,
                         "got": have, "rel_diff": rel, "pass": passed})
    return ok, rows


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _f(v: Any, d: int = 1) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:.1f} %"


def render_markdown(report: dict) -> str:
    L: list[str] = ["# ETA revision study", ""]
    L.append(f"Generated {report['generated_at_utc']}. Runtime {report['runtime_s']:.0f} s.")
    L.append("")
    L.append("## Sanity gate")
    L.append("")
    gate = report["sanity_gate"]
    L.append(f"**{'PASSED' if gate['passed'] else 'FAILED'}** — pooled counts vs "
             "`thresholds_rp_trees_all/sweep.md` at the shipped thresholds, tolerance 1 %.")
    L.append("")
    L.append("| lead | thr | warnings (scored) | hits | late | false alarms | pending | misses | sweep warnings | sweep hits |")
    L.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for lead, e in report["leads"].items():
        g = e["gate"]
        exp = SWEEP_EXPECTED.get(int(lead), {})
        L.append(f"| {lead} | {e['threshold_pct']} % | {g['warnings']} | {g['hits']} | {g['late']} | "
                 f"{g['false_alarms']} | {g['pending']} | {g['misses']} | "
                 f"{exp.get('warnings', '–')} | {exp.get('hits', '–')} |")
    L.append("")
    if not gate["passed"]:
        L.append("The gate failed; no other number is reported.")
        return "\n".join(L) + "\n"

    L += ["## Columns and conventions", ""]
    L.append("- Decision rows: `" + report["settings"]["decisions_dir"] + "`; "
             f"{report['load']['rows']} rows, {report['load']['stations']} stations, "
             f"{report['load']['duplicates']} duplicate keys; {report['stations_scored']} stations scored "
             f"(dead gauges excluded: {', '.join(report['dead_gauges']) or 'none'}).")
    L.append("- Decision instant `t` = `generated_at` (the scorer's `sent`); rule clock = `radar_ts`; "
             "probability = `p_post_<lead>`; ETA = `eta_min` (null = no rain within the ensemble horizon, "
             "max seen ≈ 87 min); observed rate at the point = `observed_mm_h` (≥ 0.5 mm/h = already raining). "
             "`intensity_mm_h` / `forecast_now_mm_h` are passed to the engine but take no part in the decision.")
    L.append("- Rule: `push.engine.evaluate` itself, driven as `threshold_sweep.replay_station` drives it "
             "(persistence 1, re-arm after 60 min below threshold on the radar clock, already-raining at "
             "ETA ≤ 1.5 min or observed ≥ 0.5 mm/h, fresh armed state at every coverage run, gap > 20 min).")
    L.append("- Truth and grading: `threshold_sweep.gauge_truth` (dry 60 min, ≥ 0.2 mm over the onset slot and the next, "
             "dead gauge = ≥ 500 known slots never wet) and `warning_score.score_warnings` "
             "(window `(t0, t0 + lead + 10]`, minimum useful lead 5 min → `late`, `known_until` + coverage).")
    L.append("- **Onset resolution.** A gauge slot is 10 min stamped at its END; the onset used here is the slot "
             "end the scorer matched (the scorer's convention). The true first drop lies up to 10 min earlier, so "
             "every error is resolved to no better than ±5 min and is biased up to 10 min NEGATIVE. "
             "Columns marked *(mid)* move the onset to the slot midpoint (−5 min): the signed median shifts by "
             "exactly +5.")
    L.append("- Error sign: `error = (t_k + eta_k) − onset`, minutes. POSITIVE = the rain came before the "
             "predicted instant (the scorer's `lead_error_min` sign). Later rows are rows of the same station "
             "with `t_k` in `(t0, onset)`, a finite ETA and a non-null probability at that lead.")
    L.append("- **ETA resolution.** In these rows `generated_at + eta_min` always lands on the forecast's 10-min "
             "timesteps (`radar_ts + 10k`, k ≥ 2; checked on 28k rows: 100 % for 0 < ETA ≤ 20, 85 % above), and onsets are "
             "10-min slot ends on the same grid, so errors are multiples of 10 min (±1 s). The exception is "
             "ETA = 0 (rain forecast at the point now; about a third of all finite ETAs), whose arrival instant "
             "is `t` itself, off the grid. Because ETA ≥ 0, a row `m` minutes before the onset cannot score an "
             "error below `−m`: the later rows' early-side errors are bounded by construction. Medians and p90s that are "
             "not multiples of 10 are numpy's linear interpolation between two grid values; the share columns "
             "(err = 0, |err| ≤ 10, err ≤ −20 i.e. ≥ 2 steps early, err ≥ +10 i.e. rain ≥ 1 step before the "
             "predicted instant) are the honest reading.")
    L.append("- Revision (A, last table): the predicted ARRIVAL instant `t + eta` at the last row before the onset "
             "vs at the push row; `|Δ| = |error_last − error_0|`.")
    L.append("- Reversal (B): a later row with `p_post_<lead>` < threshold, before the onset (hits, late) or at "
             "`t ≤ t0 + lead + 10` (false alarms: the scorer's window end). Strict = two consecutive evaluated "
             "rows both below, timed at the second.")
    L.append("")

    for lead, e in report["leads"].items():
        L.append(f"## Lead {lead} min — threshold {e['threshold_pct']} %")
        L.append("")
        L.append(f"Already-raining silent consumptions: {e['already_raining_consumed']}. "
                 f"Scorer's pooled lead-error p50 over hits: {_f(e['pooled_lead_error_p50'])} min.")
        L.append("")
        for title, key in (("A1. ETA error by minutes since the push (hits)", "A_since_push"),
                           ("A2. ETA error by minutes before the onset (hits)", "A_before_onset")):
            L.append(f"**{title}**")
            L.append("")
            L.append("| bucket (min) | n rows | median err | median \\|err\\| | p90 \\|err\\| | median err (mid) | median \\|err\\| (mid) | err = 0 | \\|err\\| ≤ 10 | err ≤ −20 | err ≥ +10 |")
            L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for r in e[key]:
                L.append(f"| {r['bucket']} | {r['n']} | {_f(r['median_err'])} | {_f(r['median_abs_err'])} | "
                         f"{_f(r['p90_abs_err'])} | {_f(r['median_err_mid'])} | {_f(r['median_abs_err_mid'])} | "
                         f"{_pct(r['share_exact'])} | {_pct(r['share_within_10'])} | "
                         f"{_pct(r['share_early_ge20'])} | {_pct(r['share_late_ge10'])} |")
            L.append("")
        rv = e["A_revision"]
        L.append("**A3. Revision at the last cycle before the onset (hits)**")
        L.append("")
        L.append("| quantity | value |")
        L.append("| --- | ---: |")
        L.append(f"| hits | {rv['hits']} |")
        L.append(f"| hits with a push ETA | {rv['hits_with_push_eta']} |")
        L.append(f"| hits with no push ETA (of which a later row had one) | {rv['hits_without_push_eta']} ({rv['hits_without_push_eta_later_eta']}) |")
        L.append(f"| hits with ≥ 1 later row carrying an ETA | {rv['hits_with_later_eta_row']} |")
        L.append(f"| arrival estimate moved ≥ 5 min (share of hits with a push ETA) | {rv['n_diff_ge5']} ({_pct(rv['share_diff_ge5'])}) |")
        L.append(f"| arrival estimate moved ≥ 10 min | {rv['n_diff_ge10']} ({_pct(rv['share_diff_ge10'])}) |")
        L.append(f"| last-row \\|err\\| < push \\|err\\| (of hits with a later ETA row) | {_pct(rv['share_last_closer'])} |")
        L.append(f"| equal / worse | {_pct(rv['share_last_same'])} / {_pct(rv['share_last_worse'])} |")
        L.append(f"| median \\|err\\| push → last (hits with a later ETA row) | {_f(rv['median_abs_err_push'])} → {_f(rv['median_abs_err_last'])} |")
        L.append(f"| median min from the last ETA row to the onset | {_f(rv['median_min_last_before_onset'])} |")
        L.append(f"| err = 0 (on the onset slot end) push → last | {_pct(rv['share_exact_push'])} → {_pct(rv['share_exact_last'])} |")
        L.append("")
        L.append("**B. Reversals after the push**")
        L.append("")
        L.append("| rule | group | n | with a later row | reversed | share | median min push → reversal | median min reversal → onset | reversed before onset slot began |")
        L.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for rule, label in (("single", "1 row below"), ("two_consecutive", "2 rows below")):
            for grp, glabel in (("hits", "hits (all-clear would be WRONG)"),
                                ("false_alarms", "false alarms (all-clear would be RIGHT)"),
                                ("late", "late")):
                r = e["B_reversals"][rule][grp]
                L.append(f"| {label} | {glabel} | {r['n']} | {r['with_later_row']} | {r['reversed']} | "
                         f"{_pct(r['share'])} | {_f(r['median_min_after_push'])} | "
                         f"{_f(r.get('median_min_before_onset'))} | {_f(r.get('reversed_before_slot_start'))} |")
        L.append("")
        c = e["C_no_eta"]
        L.append("**C. Pushes with no ETA (\"within N min\" title)**")
        L.append("")
        L.append("| group | pushes | no ETA | share |")
        L.append("| --- | ---: | ---: | ---: |")
        for grp, glabel in (("all_sent", "all sent (incl. pending)"), ("hits", "hits"),
                            ("late", "late"), ("false_alarms", "false alarms")):
            L.append(f"| {glabel} | {c[grp]['n']} | {c[grp]['no_eta']} | {_pct(c[grp]['share'])} |")
        L.append("")
        d, dh = e["D_eta_false_alarms"], e["D_eta_hits"]
        L.append("**D. Push ETA of false alarms (hits beside it for comparison)**")
        L.append("")
        L.append("| push ETA (min) | false alarms | share | hits | share |")
        L.append("| --- | ---: | ---: | ---: | ---: |")
        for rf, rh in zip(d["rows"], dh["rows"]):
            L.append(f"| {rf['bucket']} | {rf['n']} | {_pct(rf['share'])} | {rh['n']} | {_pct(rh['share'])} |")
        L.append(f"| p25 / p50 / p75 | {_f(d['p25'])} / {_f(d['p50'])} / {_f(d['p75'])} | | "
                 f"{_f(dh['p25'])} / {_f(dh['p50'])} / {_f(dh['p75'])} | |")
        L.append("")
    if report.get("reading"):
        L += ["## Reading", ""] + report["reading"] + [""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decisions-dir", type=Path, required=True)
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--thresholds", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--leads", default="20,30,45,60")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--render-only", action="store_true",
        help="re-render report.md from report.json (+ reading.md) in --out-dir",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    from dmi_nowcast_sidecar.threshold_sweep import gauge_truth

    args = build_parser().parse_args(argv)
    started = time.time()
    out = Path(args.out_dir)
    reading_file = out / "reading.md"
    if args.render_only:
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        if report["sanity_gate"]["passed"] and reading_file.is_file():
            report["reading"] = [
                line for line in reading_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
        (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
        return 0

    def log(msg: str) -> None:
        print(f"[{time.time() - started:7.1f}s] {msg}", file=sys.stderr, flush=True)

    leads = [int(x) for x in args.leads.split(",") if x.strip()]
    doc = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    thresholds = {lead: int(doc["leads"][str(lead)]["threshold_pct"]) for lead in leads}
    log(f"thresholds: {thresholds} on {doc['objective']['probability_column']}")

    arrays, load_counts = load_station_arrays(args.decisions_dir, leads, log=log)
    stations = sorted(arrays)
    lo = min(int(a["radar"].min()) for a in arrays.values())
    hi = max(int(a["radar"].max()) for a in arrays.values())
    onsets, known_until, _known_slots, dead = gauge_truth(
        Path(args.corpus_dir), stations, (to_dt(lo), to_dt(hi)),
        dry_min=DEFAULT_DRY_MIN, onset_min_mm=DEFAULT_ONSET_MIN_MM,
        min_known_slots=DEFAULT_MIN_KNOWN_SLOTS, log=log,
    )
    scored = [s for s in stations if s in known_until]
    log(f"{len(scored)} scored stations; replaying with {args.workers} workers")
    shared = {"arrays": arrays, "onsets": onsets, "known_until": known_until,
              "leads": leads, "thresholds": thresholds}
    results = run_all(shared, scored, int(args.workers), log=log)
    log("aggregating")
    report = aggregate(results, leads, thresholds, DEFAULT_TOLERANCE_MIN)
    passed, gate_rows = sanity_gate(report, {k: v for k, v in SWEEP_EXPECTED.items() if k in leads})
    report["sanity_gate"] = {"passed": bool(passed), "tolerance": GATE_TOLERANCE, "rows": gate_rows}
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["settings"] = {
        "decisions_dir": str(args.decisions_dir), "corpus_dir": str(args.corpus_dir),
        "thresholds_file": str(args.thresholds), "thresholds": thresholds,
        "tolerance_min": DEFAULT_TOLERANCE_MIN, "min_useful_lead_min": FIT_MIN_USEFUL_LEAD_MIN,
        "dry_min": DEFAULT_DRY_MIN, "onset_min_mm": DEFAULT_ONSET_MIN_MM,
        "min_known_slots": DEFAULT_MIN_KNOWN_SLOTS, "coverage_gap_min": DEFAULT_COVERAGE_GAP_MIN,
        "onset_convention": "slot end (scorer); '_mid' fields shift to slot midpoint",
    }
    report["columns"] = COLUMNS
    report["load"] = load_counts
    report["dead_gauges"] = list(dead)
    report["stations_scored"] = len(scored)
    report["reading"] = []
    report["runtime_s"] = time.time() - started

    out.mkdir(parents=True, exist_ok=True)
    if passed and reading_file.is_file():
        report["reading"] = [
            line for line in reading_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    log(f"sanity gate {'PASSED' if passed else 'FAILED'}; wrote {out}/report.md and report.json")
    for r in gate_rows:
        log(f"  lead {r['lead']} {r['count']}: {r['got']} vs {r['expected']} ({100 * r['rel_diff']:.2f} %)")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
