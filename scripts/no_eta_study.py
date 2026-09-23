#!/usr/bin/env python3
"""Pushes without an ETA: why the post-processor fires on them, and what to do.

A read-only diagnosis (S8). ``eta_revision_study.py`` found that about half
of the served pushes carry no ETA (the ensemble brings no rain to the point
within its horizon) and that those pushes are most of the false alarms. The
tree post-processor already sees ``eta_missing``; this script asks why it
still assigns >= threshold to those rows, and whether the information can
lift precision AND recall.

Nothing re-implements the rule or the truth:

* the push rule is ``push.engine.evaluate``, driven by
  ``eta_revision_study.replay_pushes`` (itself a line-for-line mirror of
  ``threshold_sweep.replay_station``); a rule VARIANT only changes the
  inputs the engine sees — a suppressed row has its probability capped
  just below the threshold (it is "not over": it resets a streak and runs
  the re-arm clock, exactly as a row under the threshold does), and a
  "consumed" row has its observed rate set above the raining-now cut so the
  engine's own already-raining branch consumes the arm silently;
* warning truth is ``threshold_sweep.gauge_truth`` + ``warning_score.
  score_warnings`` with the sweep's settings, pooled by ``pooled_summary``
  / ``skill_scores``;
* probability truth (a) is Layer B's: ``decision_rows.build_gauge_grid``
  and ``GaugeGrid.outcome`` (wet within ``(t, t + L]``), reliability by
  ``benchmark_report.reliability_table``;
* threshold picks use ``threshold_sweep.pick_plateau`` (the sweep's F1
  plateau midpoint), leave-one-(year, month)-out.

Usage (from ``sidecar/``)::

    PYTHONPATH=../src:. .venv/bin/python ../scripts/no_eta_study.py \\
        --decisions-dir ~/dmi-nowcast-corpus-local/stations/replay_rp_trees_all_oof/decisions \\
        --corpus-dir ~/dmi-nowcast-corpus-local \\
        --thresholds ~/dmi-nowcast-corpus-local/stations/thresholds_rp_trees_all/push_thresholds.json \\
        --out-dir ~/dmi-nowcast-corpus-local/stations/no_eta_study --workers 10
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "sidecar", _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eta_revision_study as ers  # noqa: E402
from benchmark_report import reliability_table  # noqa: E402
from dmi_nowcast_core.benchmark import roc_auc  # noqa: E402
from dmi_nowcast_core.warning_score import (  # noqa: E402
    DEFAULT_COVERAGE_GAP_MIN,
    DEFAULT_DRY_MIN,
    DEFAULT_MIN_KNOWN_SLOTS,
    DEFAULT_ONSET_MIN_MM,
    DEFAULT_TOLERANCE_MIN,
    coverage_runs,
    score_warnings,
    skill_scores,
)
from dmi_nowcast_sidecar.decision_rows import build_gauge_grid  # noqa: E402
from dmi_nowcast_sidecar.threshold_sweep import (  # noqa: E402
    FIT_MIN_USEFUL_LEAD_MIN,
    RAIN_THRESHOLD_MM_H,
    gauge_truth,
    pick_plateau,
)

US_PER_MIN = 60_000_000
LEADS = (20, 30, 45, 60)
SWEEP_EXPECTED = ers.SWEEP_EXPECTED
GATE_TOLERANCE = 0.0  # this study must reproduce the counts EXACTLY

#: Feature columns carried for (c) and (e). Per-lead ones are expanded.
FEATURES = [
    "observed_mm_h", "obs_prev10_mm_h", "obs_prev20_mm_h", "obs_max_5km_mm_h",
    "obs_max_5km_prev10_mm_h", "wet_frac_5km", "wet_frac_10km",
    "up_max_20km_mm_h", "up_max_40km_mm_h", "up_mean_40km_mm_h",
    "up_wet_frac_40km", "up_dist_km",
    "ng_upwet_tau_min", "ng_upwet_cross_km", "ng_upwet_mm_30",
    "ng_up_wet_share_t30", "ng_up_wet_share_t60", "ng_up_mm_max_t60",
    "ng_near_km", "ng_near_mm_60", "ng_near_min_since_wet",
    "ng_wet_share_20km", "ng_count_20km",
    "ens_eta_spread_min", "station_radar_km", "frame_age_min", "bulk_kmh",
    "local_speed_kmh", "stalled_share", "hour_utc",
    "g_mm_10", "g_mm_60", "g_min_since_wet", "g_dry_60", "g_known",
]
PER_LEAD = ["ens_mean_{lead}", "ens_p90_{lead}", "raw_frac_{lead}"]
#: Own-gauge columns: a measurement AT the station, which an address does
#: not have (``postprocess.OWN_GAUGE_COLUMNS``). Reported, never proposed.
OWN_GAUGE = {"g_mm_10", "g_mm_60", "g_min_since_wet", "g_dry_60", "g_known"}

SEASON_CODE = {"winter": 0, "shoulder": 1, "summer": 2}


# ---------------------------------------------------------------------------
# Loading — eta_revision_study.load_station_arrays plus feature columns
# ---------------------------------------------------------------------------


def load_arrays(decisions_dir: Path, leads: Sequence[int], log=print):
    """Per-station arrays sorted by radar_ts, dedup last-file-wins.

    Same selection and dedup as ``eta_revision_study.load_station_arrays``
    (which is itself ``threshold_sweep.load_decisions``'), with the feature
    columns carried along.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    feats = list(FEATURES) + [c.format(lead=l) for c in PER_LEAD for l in leads]
    core = ["radar_ts", "generated_at", "station_id", "eta_min",
            "intensity_mm_h", "observed_mm_h", "forecast_now_mm_h", "season"]
    wanted = core + [f"p_post_{lead}" for lead in leads] + [f for f in feats if f not in core]
    files = sorted(Path(decisions_dir).rglob("*.parquet"))
    tables = []
    for order, path in enumerate(files):
        names = set(pq.read_schema(path).names)
        t = pq.read_table(path, columns=[c for c in wanted if c in names])
        for c in wanted:
            if c not in names:
                t = t.append_column(c, pa.nulls(t.num_rows, pa.string() if c == "season" else pa.float64()))
        t = t.select(wanted)
        t = t.append_column("_file", pa.array(np.full(t.num_rows, order, np.int32)))
        tables.append(t)
    table = pa.concat_tables(tables, promote_options="default")
    n_rows_raw = table.num_rows

    def ts_us(name):
        arr = table.column(name).cast(pa.timestamp("us", tz="UTC")).to_numpy(zero_copy_only=False)
        return arr.astype("datetime64[us]").astype(np.int64), np.isnat(arr)

    radar, radar_nat = ts_us("radar_ts")
    gen, gen_nat = ts_us("generated_at")
    gen = np.where(gen_nat, radar, gen)
    station = np.asarray(table.column("station_id").to_pylist(), dtype=object)

    def f64(name):
        return table.column(name).cast(pa.float64()).to_numpy(zero_copy_only=False)

    fields = {"eta": f64("eta_min"), "intensity": f64("intensity_mm_h"),
              "observed": f64("observed_mm_h"), "forecast": f64("forecast_now_mm_h")}
    for lead in leads:
        fields[f"p{lead}"] = f64(f"p_post_{lead}")
    for f in feats:
        if f not in ("observed_mm_h",):
            fields["f:" + f] = f64(f)
    season = np.asarray(table.column("season").to_pylist(), dtype=object)
    fields["season"] = np.array([SEASON_CODE.get(s, -1) for s in season], np.int8)
    file_order = table.column("_file").to_numpy()
    del table

    keep = ~radar_nat
    codes, stations = ers._encode(station)
    order = np.lexsort((file_order, radar, codes))
    order = order[keep[order]]
    c_s, r_s = codes[order], radar[order]
    last = np.ones(order.size, bool)
    last[:-1] = (c_s[1:] != c_s[:-1]) | (r_s[1:] != r_s[:-1])
    duplicates = int(order.size - last.sum())
    order = order[last]
    out = {}
    c_o = codes[order]
    for chunk in np.split(order, np.flatnonzero(np.diff(c_o)) + 1):
        if chunk.size == 0:
            continue
        sid = stations[codes[chunk[0]]]
        arrays = {"radar": radar[chunk], "gen": gen[chunk]}
        for name, values in fields.items():
            arrays[name] = values[chunk]
        arrays["f:observed_mm_h"] = arrays["observed"]
        out[sid] = arrays
    log(f"loaded {n_rows_raw} rows from {len(files)} files, {len(out)} stations, "
        f"{duplicates} duplicate keys")
    return out, {"files": len(files), "rows": n_rows_raw, "duplicates": duplicates,
                 "stations": len(out)}


# ---------------------------------------------------------------------------
# Rule variants
# ---------------------------------------------------------------------------
#
# A cell is (lead, t1, kind, param):
#   kind "served"   : the served rule at threshold t1.
#   kind "cond"     : t1 with an ETA, p2 = param without one (param 101 = never).
#   kind "consume"  : served rule, but a no-ETA row where cond(param) holds
#                     counts as already raining (arm consumed silently).
#   kind "gatex"    : served rule, but a no-ETA row where cond(param) holds is
#                     "not over" (suppressed, re-arm clock runs).
# ``param`` for consume/gatex is a key into CONDITIONS.

CONDITIONS: dict[str, tuple[str, str, float]] = {}


def _cond_mask(arrays: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    col, op, cut = CONDITIONS[key]
    v = arrays["f:" + col]
    with np.errstate(invalid="ignore"):
        if op == ">":
            return np.isfinite(v) & (v > cut)
        if op == ">=":
            return np.isfinite(v) & (v >= cut)
        if op == "<":
            return np.isfinite(v) & (v < cut)
        if op == "<=":
            return np.isfinite(v) & (v <= cut)
    raise ValueError(op)


def variant_arrays(arrays, lead, t1, kind, param):
    """The engine inputs a variant sees; everything else is untouched."""
    if kind == "served":
        return arrays
    p = arrays[f"p{lead}"]
    no_eta = ~np.isfinite(arrays["eta"])
    below = t1 / 100.0 - 1e-9
    out = dict(arrays)
    if kind == "cond":
        p2 = float(param) / 100.0
        sup = no_eta & np.isfinite(p) & (p < p2)
        out[f"p{lead}"] = np.where(sup, np.minimum(p, below), p)
    elif kind == "gatex":
        sup = no_eta & _cond_mask(arrays, param)
        out[f"p{lead}"] = np.where(sup & np.isfinite(p), np.minimum(p, below), p)
    elif kind == "consume":
        sup = no_eta & _cond_mask(arrays, param)
        out["observed"] = np.where(sup, 1e3, arrays["observed"])
    else:
        raise ValueError(kind)
    return out


def _month_key_us(us: int) -> int:
    d = ers.to_dt(us)
    return d.year * 12 + d.month - 1


def score_cell_station(arrays, onsets, known_until, lead, t1, kind, param, *,
                       runs, radar_dt, gen_dt, want_pushes=False):
    """Per-month counts ``{month: [hits, late, fa, pending, misses]}``."""
    va = variant_arrays(arrays, lead, t1, kind, param)
    pushes, consumed = ers.replay_pushes(va, lead, int(t1), runs=runs,
                                         radar_dt=radar_dt, gen_dt=gen_dt)
    gen = arrays["gen"]
    eta = arrays["eta"]
    pushes.sort(key=lambda i: (gen[i], i))
    coverage = coverage_runs(radar_dt, max_gap_min=DEFAULT_COVERAGE_GAP_MIN,
                             extend_min=lead + DEFAULT_TOLERANCE_MIN)
    res = score_warnings(
        [(gen_dt[i], ers._opt(eta[i])) for i in pushes], list(onsets),
        lead_min=int(lead), tolerance_min=DEFAULT_TOLERANCE_MIN,
        dry_min=DEFAULT_DRY_MIN, onset_min_mm=DEFAULT_ONSET_MIN_MM,
        known_until=known_until, coverage=coverage,
        min_useful_lead_min=FIT_MIN_USEFUL_LEAD_MIN,
    )
    counts: dict[int, list[int]] = {}
    slot = {"hit": 0, "late": 1, "false_alarm": 2, "pending": 3}
    for i, w in zip(pushes, res.warnings):
        m = _month_key_us(int(gen[i]))
        counts.setdefault(m, [0, 0, 0, 0, 0])[slot[w.outcome]] += 1
    for o in res.onsets:
        if o.outcome == "miss":
            d = o.onset_utc
            counts.setdefault(d.year * 12 + d.month - 1, [0, 0, 0, 0, 0])[4] += 1
    out = {"counts": counts, "consumed": consumed}
    if want_pushes:
        out["pushes"] = [
            (int(i), w.outcome, None if w.onset_utc is None else ers.to_us(w.onset_utc))
            for i, w in zip(pushes, res.warnings)
        ]
    return out


_SHARED: dict | None = None


def _work(task):
    station, cells, want = task
    s = _SHARED
    arrays = s["arrays"][station]
    runs = ers.run_ids(arrays["radar"])
    radar_dt = [ers.to_dt(v) for v in arrays["radar"]]
    gen_dt = [ers.to_dt(v) for v in arrays["gen"]]
    out = {}
    for cell in cells:
        out[cell] = score_cell_station(
            arrays, s["onsets"].get(station, ()), s["known_until"].get(station),
            *cell, runs=runs, radar_dt=radar_dt, gen_dt=gen_dt,
            want_pushes=(cell in want),
        )
    return station, out


def run_cells(shared, stations, cells, want, workers, log):
    """``{cell: {station: result}}`` for every cell, parallel over stations."""
    global _SHARED
    _SHARED = shared
    by_cell: dict = {c: {} for c in cells}
    tasks = [(st, list(cells), set(want)) for st in stations]
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for n, (st, res) in enumerate(pool.map(_work, tasks, chunksize=1), 1):
            for c, r in res.items():
                by_cell[c][st] = r
            if n % 20 == 0:
                log(f"  {n}/{len(stations)} stations × {len(cells)} cells")
    _SHARED = None
    return by_cell


def pooled_counts(per_station: Mapping[str, dict], exclude_month: int | None = None,
                  only_month: int | None = None) -> dict:
    tot = np.zeros(5, np.int64)
    for r in per_station.values():
        for m, c in r["counts"].items():
            if exclude_month is not None and m == exclude_month:
                continue
            if only_month is not None and m != only_month:
                continue
            tot += np.asarray(c)
    h, late, fa, pend, miss = (int(x) for x in tot)
    sk = skill_scores(h, fa, miss, late)
    return {"warnings": h + late + fa, "hits": h, "late": late, "false_alarms": fa,
            "pending": pend, "misses": miss, "precision": sk["precision"],
            "recall": sk["recall"], "f1": sk["f1"], "csi": sk["csi"]}


def months_of(by_cell) -> list[int]:
    ms = set()
    for per in by_cell.values():
        for r in per.values():
            ms |= set(r["counts"])
    return sorted(ms)


def lomo(by_cell, cells: Sequence[tuple], param_of, months) -> dict:
    """Leave-one-month-out pick over ``cells`` with the sweep's plateau rule.

    ``param_of(cell)`` is the 1-D parameter the plateau is read on. Returns
    the pooled out-of-fold counts plus the pick per month.
    """
    tot = np.zeros(5, np.int64)
    picks = {}
    for m in months:
        cand = []
        for c in cells:
            pc = pooled_counts(by_cell[c], exclude_month=m)
            cand.append({"threshold_pct": param_of(c), "f1": pc["f1"], "_cell": c})
        pick = pick_plateau(cand)
        if pick is None:
            continue
        cell = pick["cell"]["_cell"]
        picks[m] = param_of(cell)
        held = pooled_counts(by_cell[cell], only_month=m)
        tot += np.array([held["hits"], held["late"], held["false_alarms"],
                         held["pending"], held["misses"]])
    h, late, fa, pend, miss = (int(x) for x in tot)
    sk = skill_scores(h, fa, miss, late)
    return {"warnings": h + late + fa, "hits": h, "late": late, "false_alarms": fa,
            "pending": pend, "misses": miss, "precision": sk["precision"],
            "recall": sk["recall"], "f1": sk["f1"], "csi": sk["csi"],
            "picks": {f"{m // 12}-{m % 12 + 1:02d}": v for m, v in picks.items()}}


# ---------------------------------------------------------------------------
# Row-level tables
# ---------------------------------------------------------------------------


def flat_rows(arrays: Mapping[str, dict], scored: Sequence[str]) -> dict:
    """All stations' rows concatenated, with a grid station code."""
    code = {s: i for i, s in enumerate(scored)}
    keys = [k for k in next(iter(arrays.values())) if k not in ("radar",)]
    out = {k: [] for k in keys}
    out["code"] = []
    out["local"] = []
    out["sid"] = []
    for s in scored:
        a = arrays[s]
        for k in keys:
            out[k].append(a[k])
        out["code"].append(np.full(a["gen"].size, code[s], np.int64))
        out["local"].append(np.arange(a["gen"].size))
        out["sid"].append(np.full(a["gen"].size, code[s], np.int64))
    return {k: np.concatenate(v) for k, v in out.items()}


def onset_in_window(rows, scored, onsets, lead):
    """1 where a gauge onset falls in the scorer window ``(t, t + L + 10]``."""
    out = np.zeros(rows["gen"].size, bool)
    w = (lead + DEFAULT_TOLERANCE_MIN) * US_PER_MIN
    for i, s in enumerate(scored):
        m = rows["code"] == i
        on = np.array(sorted(ers.to_us(d) for d in onsets.get(s, ())), np.int64)
        if on.size == 0:
            continue
        t = rows["gen"][m]
        idx = np.searchsorted(on, t, side="right")
        ok = idx < on.size
        hit = np.zeros(t.size, bool)
        hit[ok] = on[idx[ok]] <= t[ok] + w
        out[m] = hit
    return out


def gauge_state(grid, t_sec, code, back_min=60):
    """'wet' / 'dry' / 'unknown' for slots ending in ``(t - back, t]``."""
    wet, usable = grid.outcome(t_sec - back_min * 60, code, back_min)
    st = np.full(t_sec.size, 2, np.int8)  # unknown
    st[usable & (wet > 0)] = 0  # wet
    st[usable & (wet == 0)] = 1  # dry
    return st


STATE_NAMES = ("wet in prev 60 min", "dry ≥ 60 min", "unknown")


def q(v, p):
    v = v[np.isfinite(v)]
    return None if v.size == 0 else float(np.percentile(v, p))


def best_split(x, y):
    """Best single cut on x separating y (1 = FA, 0 = hit), by Gini gain.

    NaN goes to its own side and is tried on both sides. Returns
    ``(direction, cut, nan_side, share of y=1 removed, share of y=0 removed,
    gain)`` where "removed" means the side flagged as FA-rich.
    """
    fin = np.isfinite(x)
    n = y.size
    if n == 0 or y.min() == y.max():
        return None

    def gini(pos, tot):
        if tot == 0:
            return 0.0
        pr = pos / tot
        return 2 * pr * (1 - pr)

    g0 = gini(y.sum(), n)
    xs = x[fin]
    ys = y[fin]
    order = np.argsort(xs, kind="mergesort")
    xs, ys = xs[order], ys[order]
    cpos = np.cumsum(ys)
    cnt = np.arange(1, xs.size + 1)
    last = np.r_[np.flatnonzero(np.diff(xs)), xs.size - 1] if xs.size else np.array([], int)
    nan_pos, nan_n = int(y[~fin].sum()), int((~fin).sum())
    best = None
    tot_pos, tot_n = int(ys.sum()), xs.size
    for j in last:
        lp, ln = int(cpos[j]), int(cnt[j])
        rp, rn = tot_pos - lp, tot_n - ln
        for nan_left in (True, False):
            Lp, Ln = lp + (nan_pos if nan_left else 0), ln + (nan_n if nan_left else 0)
            Rp, Rn = rp + (0 if nan_left else nan_pos), rn + (0 if nan_left else nan_n)
            if Ln == 0 or Rn == 0:
                continue
            gain = g0 - (Ln / n) * gini(Lp, Ln) - (Rn / n) * gini(Rp, Rn)
            if best is None or gain > best[-1]:
                cut = float(xs[j])
                # flag the FA-richer side
                left_rich = (Lp / Ln) > (Rp / Rn)
                fp, fn = (Lp, Ln) if left_rich else (Rp, Rn)
                best = ("<=" if left_rich else ">", cut,
                        "flagged" if (nan_left == left_rich) else "kept",
                        fp / max(1, y.sum()), (fn - fp) / max(1, n - y.sum()), gain)
    return best


# ---------------------------------------------------------------------------
# Candidate new columns: the point's own radar history over the last hour
# ---------------------------------------------------------------------------

#: Derived from the station's own decision rows (the point's radar history),
#: which an address has as well as a gauge station does. Not in the design.
HISTORY = ("rad_prev60_max_mm_h", "rad_min_since_rain", "rad_min_since_echo")
HISTORY_CAP_MIN = 180.0


def derive_history(a: dict) -> None:
    """Adds the HISTORY columns to one station's arrays, in place.

    * ``rad_prev60_max_mm_h`` — max ``observed_mm_h`` over the rows with
      radar_ts in ``[t - 60, t)`` (NaN when none);
    * ``rad_min_since_rain`` — minutes since the last row (this one included)
      with ``observed_mm_h >= 0.5`` (the raining-now cut), capped at 180;
    * ``rad_min_since_echo`` — the same with ``observed_mm_h >= 0.1``.
    """
    radar = a["radar"]
    obs = a["observed"]
    n = radar.size
    prev_max = np.full(n, np.nan)
    for k in range(1, 16):
        if k >= n:
            break
        dt = radar[k:] - radar[:-k]
        ok = dt <= 60 * US_PER_MIN
        v = np.where(ok, obs[:-k], np.nan)
        cur = prev_max[k:]
        with np.errstate(invalid="ignore"):
            prev_max[k:] = np.fmax(cur, v)
    a["f:rad_prev60_max_mm_h"] = prev_max
    for name, cut in (("rad_min_since_rain", 0.5), ("rad_min_since_echo", 0.1)):
        wet = np.nan_to_num(obs, nan=0.0) >= cut
        last = np.where(wet, radar, np.iinfo(np.int64).min)
        last = np.maximum.accumulate(last)
        since = np.where(last == np.iinfo(np.int64).min, np.inf,
                         (radar - last) / US_PER_MIN)
        a["f:" + name] = np.minimum(since, HISTORY_CAP_MIN)


# ---------------------------------------------------------------------------
# (e) onset-target logistic re-fit on the no-ETA rows
# ---------------------------------------------------------------------------

#: Only rows at or above this p are fitted: below it the rule never fires.
E_MIN_P = 0.15

E_BASE = ["log1p:observed_mm_h", "log1p:obs_prev10_mm_h", "log1p:obs_prev20_mm_h",
          "log1p:ens_mean_L", "log1p:ens_p90_L", "raw_frac_L", "up_wet_frac_40km",
          "log1p:up_max_40km_mm_h", "log1p:up_mean_40km_mm_h", "ng_near_min_since_wet",
          "log1p:ng_near_mm_60", "ng_wet_share_20km", "stalled_share", "bulk_kmh",
          "season:0", "season:2"]
E_NEW = ["log1p:rad_prev60_max_mm_h", "rad_min_since_rain", "rad_min_since_echo"]
E_OWN = ["g_min_since_wet", "log1p:g_mm_60"]
E_SETS = {
    "M0 logit(p_post)": [],
    "M1 + existing design columns": E_BASE,
    "M2 + radar history (new)": E_BASE + E_NEW,
    "M3 + own gauge (NOT live at a random point)": E_BASE + E_OWN,
}
#: The OOF predictions that become rule-variant inputs.
E_EXPORT = {"M1 + existing design columns": "onset_m1", "M2 + radar history (new)": "onset_m2"}


def _design(rows, mask, lead, names):
    p = np.clip(rows[f"p{lead}"][mask], 1e-4, 1 - 1e-4)
    cols = [np.log(p / (1 - p))]
    for nm in names:
        if nm.startswith("season:"):
            cols.append((rows["season"][mask] == int(nm.split(":")[1])).astype(float))
            continue
        log1p = nm.startswith("log1p:")
        base = nm.split(":", 1)[1] if log1p else nm
        base = base.replace("_L", f"_{lead}") if base.endswith("_L") else base
        v = rows["f:" + base][mask].astype(float)
        v = np.where(np.isfinite(v), v, np.nan)
        if log1p:
            v = np.log1p(np.clip(v, 0, None))
        cols.append(v)
        if np.isnan(v).any():
            cols.append(np.isnan(v).astype(float))
    return np.column_stack(cols)


def refit_onset(rows, fit_mask, onset_w, lead, log):
    """Out-of-fold (leave-one-month-out) logistic models of ONSET in window."""
    from dmi_nowcast_core.benchmark import brier_decomposition, pr_auc
    from dmi_nowcast_core.postprocess import Standardiser, _sigmoid, fit_logistic

    y = onset_w[fit_mask].astype(float)
    gen = rows["gen"][fit_mask]
    month = (gen.astype("datetime64[us]").astype("datetime64[M]").astype(np.int64))
    out = {"n": int(fit_mask.sum()), "positives": int(y.sum()),
           "base_rate": float(y.mean()), "min_p": E_MIN_P, "models": {}}
    preds = {}
    p_raw = rows[f"p{lead}"][fit_mask]
    for label, names in E_SETS.items():
        x = _design(rows, fit_mask, lead, names)
        oof = np.full(y.size, np.nan)
        for m in np.unique(month):
            tr, te = month != m, month == m
            st = Standardiser.fit(x[tr])
            fit = fit_logistic(st.transform(x[tr]), y[tr], l2=1.0)
            z = fit["intercept"] + st.transform(x[te]) @ np.asarray(fit["coefficients"])
            oof[te] = _sigmoid(z)
        eps = 1e-6
        ll = float(-np.mean(y * np.log(oof + eps) + (1 - y) * np.log(1 - oof + eps)))
        bd = brier_decomposition(oof, y, n_bins=10)
        out["models"][label] = {"log_loss": ll, "brier": float(bd["brier"]),
                                "bss": float(bd["bss"]), "roc_auc": float(roc_auc(oof, y)),
                                "pr_auc": float(pr_auc(oof, y)), "n_columns": int(x.shape[1])}
        if label in E_EXPORT:
            preds[E_EXPORT[label]] = oof
        log(f"(e) lead {lead} {label}: PR-AUC {out['models'][label]['pr_auc']:.4f} "
            f"ROC {out['models'][label]['roc_auc']:.4f} BSS {out['models'][label]['bss']:.4f}")
    # p_post itself, read as a probability of ONSET (it is not one)
    bd = brier_decomposition(p_raw, y, n_bins=10)
    out["p_post_as_onset"] = {"mean_p": float(p_raw.mean()), "brier": float(bd["brier"]),
                              "bss": float(bd["bss"]), "roc_auc": float(roc_auc(p_raw, y)),
                              "pr_auc": float(pr_auc(p_raw, y))}
    return out, preds


# ---------------------------------------------------------------------------
# (d) the variant grid and its scoring
# ---------------------------------------------------------------------------

T1_GRID = tuple(range(20, 75, 5))
Q_GRID = (0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20)


def build_cells(leads, thr):
    """``(cells, families)``; a family is ``{name: [cells]}`` per lead."""
    CONDITIONS.clear()
    for c in (0.05, 0.1, 0.2, 0.3):
        CONDITIONS[f"observed_mm_h>={c}"] = ("observed_mm_h", ">=", c)
    for c in (10, 20, 30, 60):
        CONDITIONS[f"rad_min_since_rain<={c}"] = ("rad_min_since_rain", "<=", c)
    for c in (10, 30, 60):
        CONDITIONS[f"rad_min_since_echo<={c}"] = ("rad_min_since_echo", "<=", c)
    for c in (0.1, 0.3):
        CONDITIONS[f"ng_near_mm_60>={c}"] = ("ng_near_mm_60", ">=", c)
    for c in (10, 30):
        CONDITIONS[f"ng_near_min_since_wet<={c}"] = ("ng_near_min_since_wet", "<=", c)
    for c in (30, 50, 60):
        CONDITIONS[f"g_min_since_wet<={c}"] = ("g_min_since_wet", "<=", c)
    for lead in leads:
        for m in ("onset_m1", "onset_m2"):
            for qq in Q_GRID:
                CONDITIONS[f"{m}_{lead}<{qq}"] = (f"{m}_{lead}", "<", qq)
    cells: list = []
    families: dict = {}
    for lead in leads:
        t = thr[lead]
        fam = {}
        fam["served (t1 refit)"] = [(lead, x, "served", 0) for x in T1_GRID]
        fam["no ETA, no push (t1 refit)"] = [(lead, x, "cond", 101) for x in T1_GRID]
        fam["conditional p2 (t1 served)"] = [(lead, t, "cond", p2) for p2 in range(t + 5, 100, 5)]
        fam["conditional p2 (t1 served - 5)"] = [(lead, t - 5, "cond", p2) for p2 in range(t, 100, 5)]
        for key, (col, op, cut) in CONDITIONS.items():
            if col.startswith("onset_"):
                if not col.endswith(f"_{lead}"):
                    continue
                m = col.rsplit("_", 1)[0]
                fam.setdefault(f"gate on {m} (t1 served)", []).append((lead, t, "gatex", key))
                fam.setdefault(f"gate on {m} (t1 served - 5)", []).append((lead, t - 5, "gatex", key))
                fam.setdefault(f"gate on {m} (t1 served - 10)", []).append((lead, t - 10, "gatex", key))
            else:
                fam.setdefault(f"consume: {key}", []).append((lead, t, "consume", key))
        families[lead] = fam
        for cs in fam.values():
            for c in cs:
                if c not in cells:
                    cells.append(c)
    return cells, families


def _param(cell):
    lead, t1, kind, param = cell
    if kind == "served" or (kind == "cond" and param == 101):
        return t1
    if kind == "cond":
        return int(param)
    if kind == "gatex":
        return int(round(100 * CONDITIONS[param][2]))
    return 0


def score_families(by_cell, families, months, thr, leads):
    out = {}
    for lead in leads:
        base = pooled_counts(by_cell[(lead, thr[lead], "served", 0)])
        ent = {"baseline": base, "families": {}}
        for name, cells in families[lead].items():
            rows = []
            for c in cells:
                pc = pooled_counts(by_cell[c])
                rows.append({"cell": list(c), "t1": c[1], "param": c[3], **pc,
                             "d_hits": pc["hits"] - base["hits"],
                             "d_fa": pc["false_alarms"] - base["false_alarms"],
                             "both_up": bool(pc["precision"] is not None and pc["recall"] is not None
                                             and pc["precision"] > base["precision"]
                                             and pc["recall"] > base["recall"])})
            fam = {"cells": rows}
            if len(cells) > 1 and not name.startswith("consume"):
                fam["lomo"] = lomo(by_cell, cells, _param, months)
            ent["families"][name] = fam
        out[str(lead)] = ent
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _p(v, d=1):
    return "–" if v is None else f"{100 * v:.{d}f} %"


def _f(v, d=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def render_markdown(r) -> str:
    L = ["# No-ETA push study (S8)", ""]
    L.append(f"Generated {r.get('generated_at_utc')}. Runtime {r.get('runtime_s', 0):.0f} s. "
             f"Script `scripts/no_eta_study.py`.")
    L += ["", "## Sanity gate", ""]
    g = r["sanity_gate"]
    L.append(f"**{'PASSED' if g['passed'] else 'FAILED'}** — the served rule replayed at the shipped "
             "thresholds must reproduce the sweep's scored warnings and hits exactly.")
    L += ["", "| lead | thr | warnings | hits | late | false alarms | misses | pending | sweep warnings | sweep hits |",
          "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for x in g["rows"]:
        L.append(f"| {x['lead']} | {x['threshold']} % | {x['warnings']} | {x['hits']} | {x['late']} | "
                 f"{x['false_alarms']} | {x['misses']} | {x['pending']} | {x['expected_warnings']} | {x['expected_hits']} |")
    L.append("")
    if not g["passed"]:
        return "\n".join(L) + "\n"
    L += ["## Conventions", ""]
    L.append(f"- Rows: `{r['settings']['decisions_dir']}` — {r['load']['rows']} rows, "
             f"{r['stations_scored']} scored stations (dead gauge excluded: {', '.join(r['dead_gauges'])}).")
    L.append("- **No ETA** = `eta_min` null. **Eligible** = a row the engine could notify on: "
             "`observed_mm_h` < 0.5 (or null) and not `eta_min` ≤ 1.5.")
    L.append("- **Wet within L** (Layer B, the post-processor's training target): a wet gauge slot ending in "
             "`(t, t + L]`, `t` = `generated_at`; unknown windows excluded (`GaugeGrid.outcome`).")
    L.append("- **Onset in window** (the push's target): a `gauge_truth` onset (dry 60 min, ≥ 0.2 mm) in the "
             "scorer's window `(t, t + L + 10]`.")
    L.append("- **Gauge before the push**: slots ending in `(t − 60, t]` on the same grid: *wet* = any wet slot; "
             "*dry* = all known and none wet; else *unknown*. An onset needs 60 dry minutes before it, so a gauge "
             "wet at `t` cannot produce an onset for the first part of the window (none at all at 20/30 min "
             "unless it stops at once).")
    L.append("")

    # (a)
    L += ["## (a) Calibration of `p_post_<lead>` on the no-ETA stratum", ""]
    L.append("Mean p vs observed rate per stratum (all rows with a known Layer-B outcome), and the rate of the "
             "event the push is graded on (onset in window).")
    L += ["", "| lead | stratum | rows | mean p | wet within L | onset in window |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    names = {"no_eta": "no ETA, all", "eta": "ETA, all", "no_eta_eligible": "no ETA, eligible",
             "eta_eligible": "ETA, eligible", "no_eta_eligible_dry": "no ETA, eligible, gauge dry 60",
             "eta_eligible_dry": "ETA, eligible, gauge dry 60"}
    for lead, e in r["a_reliability"].items():
        for k, lab in names.items():
            x = e[k]
            L.append(f"| {lead} | {lab} | {x['n']} | {_f(x['mean_p'])} | {_f(x['observed'])} | {_f(x['onset_rate'], 4)} |")
    L.append("")
    for lead, e in r["a_reliability"].items():
        L.append(f"**Lead {lead} — reliability, ten bins** (obs = wet within L; onset = onset in window; "
                 "prev wet = share of rows whose gauge was wet in the 60 min before)")
        L += ["", "| bin | no ETA n | mean p | obs | onset | prev wet | no ETA dry-60 n | obs | onset | ETA n | obs | onset | ETA dry-60 obs | onset |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        tabs = {k: {row["bin"]: row for row in e[k]["table"]} for k in ("no_eta", "no_eta_eligible_dry", "eta", "eta_eligible_dry")}
        for b in range(10):
            a, d, c, f = (tabs[k].get(b, {}) for k in ("no_eta", "no_eta_eligible_dry", "eta", "eta_eligible_dry"))
            L.append(f"| {b / 10:.1f}–{(b + 1) / 10:.1f} | {a.get('n', 0)} | {_f(a.get('mean_p'))} | {_f(a.get('observed'))} | "
                     f"{_f(a.get('onset_rate'))} | {_p(a.get('share_prev_wet'), 0)} | {d.get('n', 0)} | {_f(d.get('observed'))} | "
                     f"{_f(d.get('onset_rate'))} | {c.get('n', 0)} | {_f(c.get('observed'))} | {_f(c.get('onset_rate'))} | "
                     f"{_f(f.get('observed'))} | {_f(f.get('onset_rate'))} |")
        L.append("")

    # (b)
    L += ["## (b) The station's own gauge at the push", ""]
    L.append("Pushes of the served rule. Columns: gauge state over the 60 min before `t`; then, inside that, whether "
             "the gauge was wet anywhere in the scorer's window `(t, t + L + 10]` (a false alarm with a wet window = "
             "rain did fall, but not as an onset).")
    L += ["", "| lead | group | pushes | wet before | dry ≥ 60 | unknown | wet before & wet window | dry before & wet window | dry before & dry window |",
          "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    glab = {"no_eta_fa": "no-ETA false alarms", "eta_fa": "with-ETA false alarms", "no_eta_hit": "no-ETA hits",
            "eta_hit": "with-ETA hits", "no_eta_late": "no-ETA late", "eta_late": "with-ETA late"}
    for lead, e in r["b_gauge_state"].items():
        for k, lab in glab.items():
            x = e[k]
            n = max(1, x["n"])
            w = x["window_wet_by_prev"]
            L.append(f"| {lead} | {lab} | {x['n']} | {x['prev'][0]} ({_p(x['prev'][0] / n, 0)}) | "
                     f"{x['prev'][1]} ({_p(x['prev'][1] / n, 0)}) | {x['prev'][2]} ({_p(x['prev'][2] / n, 0)}) | "
                     f"{w[0][0]} | {w[1][0]} | {w[1][1]} |")
    L.append("")

    # (c)
    L += ["## (c) What the no-ETA pushes look like", ""]
    L.append("Medians (share > 0 in brackets) at the push row. AUC = how well the variable alone ranks no-ETA "
             "false alarms above no-ETA hits (0.5 = no separation; < 0.5 = false alarms have LOWER values). "
             "Best split = the single Gini-optimal cut flagging the false-alarm-rich side; "
             "`FA flagged` / `hits flagged` = share of each group on that side. **own** = own-gauge column, "
             "not available at a random point.")
    for lead in r["c_splits"]:
        sp = r["c_splits"][lead]
        di = r["c_distributions"][lead]
        hist = r.get("c_history", {}).get(lead, {})
        L += ["", f"**Lead {lead}**", "",
              "| variable | no-ETA FA p50 (>0) | no-ETA hit p50 (>0) | ETA hit p50 (>0) | ETA FA p50 | AUC FA vs hit | best split | FA flagged | hits flagged |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
        items = [(k, v, di.get(k)) for k, v in sp.items()]
        items += [(k, {"auc_fa_vs_hit": v["auc_fa_vs_hit"], "split": v["split"]}, None) for k, v in hist.items()]
        items.sort(key=lambda kv: -(kv[1]["split"][-1] if kv[1]["split"] else 0))
        for k, v, d in items:
            s = v["split"]
            tag = " (own)" if k in OWN_GAUGE else (" (new)" if k in HISTORY else "")
            if d is not None:
                c = lambda gname: f"{_f(d[gname]['p50'])} ({_p(d[gname]['gt0'], 0)})"
                cells = f"{c('no_eta_fa')} | {c('no_eta_hit')} | {c('eta_hit')} | {_f(d['eta_fa']['p50'])}"
            else:
                h = hist[k]
                cells = (f"{_f(h['no_eta_fa']['p50'])} | {_f(h['no_eta_hit']['p50'])} | "
                         f"{_f(h['eta_hit']['p50'])} | {_f(h['eta_fa']['p50'])}")
            split = "–" if not s else f"{s[0]} {s[1]:.3g} (NaN {s[2]})"
            L.append(f"| `{k}`{tag} | {cells} | {_f(v['auc_fa_vs_hit'])} | {split} | "
                     f"{_p(s[3], 0) if s else '–'} | {_p(s[4], 0) if s else '–'} |")
        L.append("")

    # (d)
    L += ["## (d) Rule variants", ""]
    L.append("Every variant is the shipped engine on modified inputs (see the script docstring): *suppressed* = "
             "the row counts as under the threshold; *consume* = the row counts as already raining (arm consumed, "
             "no push). **Pooled** rows use the probabilities out of fold but pick nothing; **LOMO** rows pick the "
             "parameter on the other months with the sweep's F1-plateau rule (`pick_plateau`) and score the held-out "
             "month, pooled over months. ↑↑ = precision AND recall above the served baseline.")
    for lead, e in r["d_variants"].items():
        b = e["baseline"]
        L += ["", f"### Lead {lead}", "",
              f"Served baseline (t1 = {r['thresholds'][lead] if lead in r['thresholds'] else r['thresholds'][int(lead)]} %): "
              f"warnings {b['warnings']}, hits {b['hits']}, FA {b['false_alarms']}, misses {b['misses']}, late {b['late']}, "
              f"P {_p(b['precision'])}, R {_p(b['recall'])}, F1 {_f(b['f1'])}.", "",
              "| variant | t1 | param | warnings | hits | FA | misses | late | precision | recall | F1 | Δhits | ΔFA | ↑↑ |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for name, fam in e["families"].items():
            for c in fam["cells"]:
                L.append(f"| {name} | {c['t1']} | {c['param']} | {c['warnings']} | {c['hits']} | {c['false_alarms']} | "
                         f"{c['misses']} | {c['late']} | {_p(c['precision'])} | {_p(c['recall'])} | {_f(c['f1'])} | "
                         f"{c['d_hits']:+d} | {c['d_fa']:+d} | {'↑↑' if c['both_up'] else ''} |")
        L += ["", f"**Lead {lead} — LOMO (parameter refit out of month)**", "",
              "| family | warnings | hits | FA | misses | late | precision | recall | F1 | picks by month |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for name, fam in e["families"].items():
            lo = fam.get("lomo")
            if not lo:
                continue
            picks = ", ".join(f"{k[2:]}:{v}" for k, v in lo["picks"].items())
            L.append(f"| {name} | {lo['warnings']} | {lo['hits']} | {lo['false_alarms']} | {lo['misses']} | {lo['late']} | "
                     f"{_p(lo['precision'])} | {_p(lo['recall'])} | {_f(lo['f1'])} | {picks} |")
        L.append("")

    # (e)
    L += ["## (e) Is a feature missing? (estimate)", ""]
    L.append(f"ESTIMATE, not a model proposal. Rows: no ETA, eligible, `p_post` ≥ {E_MIN_P}, Layer-B window known. "
             "Target: onset in the scorer window (the push's event). Logistic (L2 = 1, `postprocess.fit_logistic`), "
             "leave-one-month-out; inputs standardised with NaN → mean plus a missing indicator. All models start from "
             "logit(`p_post`), so M1 − M0 = what re-targeting on columns the design ALREADY has buys, M2 − M1 = what "
             "the new radar-history columns add, M3 = own-gauge upper bound (not live at an address).")
    L += ["", "| lead | rows | onsets | base rate | model | log loss | Brier | BSS | ROC-AUC | PR-AUC |",
          "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for lead, e in r["e_refit"].items():
        pp = e["p_post_as_onset"]
        L.append(f"| {lead} | {e['n']} | {e['positives']} | {_f(e['base_rate'], 4)} | `p_post` read as P(onset) "
                 f"(mean {_f(pp['mean_p'])}) | – | {_f(pp['brier'], 4)} | {_f(pp['bss'], 3)} | {_f(pp['roc_auc'])} | {_f(pp['pr_auc'])} |")
        for lab, m in e["models"].items():
            L.append(f"| {lead} | | | | {lab} | {_f(m['log_loss'], 4)} | {_f(m['brier'], 4)} | {_f(m['bss'], 3)} | "
                     f"{_f(m['roc_auc'])} | {_f(m['pr_auc'])} |")
    L.append("")
    if r.get("reading"):
        L += ["## Reading", ""] + r["reading"] + [""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decisions-dir", type=Path, required=True)
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--thresholds", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--stage", default="all", help="'explore' = (a)-(c) only")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    started = time.time()

    def log(msg):
        print(f"[{time.time() - started:7.1f}s] {msg}", file=sys.stderr, flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    leads = list(LEADS)
    doc = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    thr = {lead: int(doc["leads"][str(lead)]["threshold_pct"]) for lead in leads}
    arrays, load_counts = load_arrays(args.decisions_dir, leads, log=log)
    stations = sorted(arrays)
    lo = min(int(a["radar"].min()) for a in arrays.values())
    hi = max(int(a["radar"].max()) for a in arrays.values())
    onsets, known_until, _slots, dead = gauge_truth(
        Path(args.corpus_dir), stations, (ers.to_dt(lo), ers.to_dt(hi)),
        dry_min=DEFAULT_DRY_MIN, onset_min_mm=DEFAULT_ONSET_MIN_MM,
        min_known_slots=DEFAULT_MIN_KNOWN_SLOTS, log=log,
    )
    scored = [s for s in stations if s in known_until]
    log(f"{len(scored)} scored stations")
    shared = {"arrays": arrays, "onsets": onsets, "known_until": known_until}

    # ---------------- baseline pushes -------------------------------------
    base_cells = [(lead, thr[lead], "served", 0) for lead in leads]
    by_cell = run_cells(shared, scored, base_cells, base_cells, args.workers, log)
    gate_rows = []
    passed = True
    for lead in leads:
        pc = pooled_counts(by_cell[(lead, thr[lead], "served", 0)])
        exp = SWEEP_EXPECTED[lead]
        ok = pc["warnings"] == exp["warnings"] and pc["hits"] == exp["hits"]
        passed &= ok
        gate_rows.append({"lead": lead, "threshold": thr[lead], **{k: pc[k] for k in (
            "warnings", "hits", "late", "false_alarms", "misses", "pending")},
            "expected_warnings": exp["warnings"], "expected_hits": exp["hits"], "pass": ok})
        log(f"gate lead {lead}: {pc['warnings']}/{pc['hits']} vs {exp['warnings']}/{exp['hits']}")
    report: dict[str, Any] = {"sanity_gate": {"passed": bool(passed), "rows": gate_rows},
                              "load": load_counts, "dead_gauges": list(dead),
                              "stations_scored": len(scored), "thresholds": thr}
    if not passed:
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
        log("SANITY GATE FAILED")
        return 3

    # ---------------- row table + Layer B truth ----------------------------
    rows = flat_rows(arrays, scored)
    t_sec = rows["gen"] // 1_000_000
    from dmi_nowcast_sidecar.decision_rows import decision_window
    grid, dead_rows, grid_scored = build_gauge_grid(
        Path(args.corpus_dir), stations, decision_window(t_sec),
        dry_min=DEFAULT_DRY_MIN, onset_min_mm=DEFAULT_ONSET_MIN_MM,
        min_known_slots=DEFAULT_MIN_KNOWN_SLOTS, log=log,
    )
    gcode = {s: i for i, s in enumerate(grid_scored)}
    remap = np.array([gcode.get(s, -1) for s in scored], np.int64)
    gc = remap[rows["code"]]
    assert (gc >= 0).all()
    prev_state = gauge_state(grid, t_sec, gc, 60)
    no_eta = ~np.isfinite(rows["eta"])
    eligible = ~(np.nan_to_num(rows["observed"], nan=0.0) >= RAIN_THRESHOLD_MM_H) & ~(
        np.isfinite(rows["eta"]) & (rows["eta"] <= 1.5))
    offset = {}
    pos = 0
    for s in scored:
        offset[s] = pos
        pos += arrays[s]["gen"].size

    # (a) reliability ------------------------------------------------------
    rel: dict[str, Any] = {}
    for lead in leads:
        p = rows[f"p{lead}"]
        y, usable = grid.outcome(t_sec, gc, lead)
        onset_w = onset_in_window(rows, scored, onsets, lead)
        base = usable & np.isfinite(p)
        ent = {}
        for name, m in (("no_eta", base & no_eta), ("eta", base & ~no_eta),
                        ("no_eta_eligible_dry", base & no_eta & eligible & (prev_state == 1)),
                        ("eta_eligible_dry", base & ~no_eta & eligible & (prev_state == 1)),
                        ("no_eta_eligible", base & no_eta & eligible),
                        ("eta_eligible", base & ~no_eta & eligible)):
            tab = reliability_table(p[m], y[m])
            # onset-in-window rate per bin (the push's target), same bins
            idx = np.clip(np.floor(p[m] * 10).astype(int), 0, 9)
            ow = onset_w[m]
            pv = prev_state[m]
            for r in tab:
                k = idx == r["bin"]
                r["onset_rate"] = float(ow[k].mean()) if k.any() else None
                r["share_prev_wet"] = float((pv[k] == 0).mean()) if k.any() else None
            ent[name] = {"n": int(m.sum()), "mean_p": float(p[m].mean()) if m.any() else None,
                         "observed": float(y[m].mean()) if m.any() else None,
                         "onset_rate": float(ow.mean()) if m.any() else None,
                         "table": tab}
        rel[str(lead)] = ent
        log(f"(a) lead {lead}: no-ETA mean p {ent['no_eta']['mean_p']:.3f} obs {ent['no_eta']['observed']:.3f}")
    report["a_reliability"] = rel

    # push table -------------------------------------------------------------
    push_tab: dict[str, Any] = {}
    push_idx = {}
    for lead in leads:
        cell = (lead, thr[lead], "served", 0)
        idx_l, outc, ons = [], [], []
        for s in scored:
            for i, o, on in by_cell[cell][s]["pushes"]:
                idx_l.append(offset[s] + i)
                outc.append(o)
                ons.append(-1 if on is None else on)
        idx_l = np.array(idx_l, np.int64)
        outc = np.array(outc)
        push_idx[lead] = (idx_l, outc)
        # scorer-window wetness
        ywin, uwin = grid.outcome(t_sec[idx_l], gc[idx_l], lead + DEFAULT_TOLERANCE_MIN)
        win_state = np.where(uwin & (ywin > 0), 0, np.where(uwin, 1, 2))
        ps = prev_state[idx_l]
        ne = no_eta[idx_l]
        g = {}
        for gname, gm in (("no_eta_fa", ne & (outc == "false_alarm")),
                          ("eta_fa", ~ne & (outc == "false_alarm")),
                          ("no_eta_hit", ne & (outc == "hit")),
                          ("eta_hit", ~ne & (outc == "hit")),
                          ("no_eta_late", ne & (outc == "late")),
                          ("eta_late", ~ne & (outc == "late"))):
            n = int(gm.sum())
            g[gname] = {"n": n,
                        "prev": [int(((ps == k) & gm).sum()) for k in range(3)],
                        "window_wet_by_prev": [
                            [int(((ps == k) & (win_state == w) & gm).sum()) for w in range(3)]
                            for k in range(3)],
                        "g_dry_60_stored": [int((gm & (rows["f:g_dry_60"][idx_l] == 0)).sum()),
                                            int((gm & (rows["f:g_dry_60"][idx_l] == 1)).sum()),
                                            int((gm & ~np.isfinite(rows["f:g_dry_60"][idx_l])).sum())]}
        push_tab[str(lead)] = g
    report["b_gauge_state"] = push_tab

    # (c) distributions and splits --------------------------------------------
    cvars = [f for f in FEATURES] + ["season"] + [c.split("_{")[0] + "_L" for c in PER_LEAD] + ["p_post_L"]
    dist: dict[str, Any] = {}
    splits: dict[str, Any] = {}
    for lead in leads:
        idx_l, outc = push_idx[lead]
        ne = no_eta[idx_l]
        groups = {"no_eta_fa": ne & (outc == "false_alarm"), "no_eta_hit": ne & (outc == "hit"),
                  "eta_fa": ~ne & (outc == "false_alarm"), "eta_hit": ~ne & (outc == "hit")}

        def col(name):
            if name == "season":
                return rows["season"][idx_l].astype(float)
            if name.endswith("_L"):
                base = name[:-2]
                if base == "p_post":
                    return rows[f"p{lead}"][idx_l]
                return rows[f"f:{base}_{lead}"][idx_l]
            return rows["f:" + name][idx_l]

        d = {}
        sp = {}
        mask2 = groups["no_eta_fa"] | groups["no_eta_hit"]
        yfa = groups["no_eta_fa"][mask2].astype(float)
        for v in cvars:
            x = col(v)
            d[v] = {}
            for gname, gm in groups.items():
                xv = x[gm]
                d[v][gname] = {"n": int(gm.sum()), "finite": float(np.isfinite(xv).mean()) if xv.size else None,
                               "p25": q(xv, 25), "p50": q(xv, 50), "p75": q(xv, 75),
                               "gt0": float((np.nan_to_num(xv, nan=0) > 0).mean()) if xv.size else None}
                if v == "season":
                    d[v][gname]["shares"] = [float((xv == k).mean()) if xv.size else None for k in range(3)]
            xs = x[mask2]
            auc = roc_auc(np.where(np.isfinite(xs), xs, np.nanmin(xs) - 1 if np.isfinite(xs).any() else 0), yfa)
            sp[v] = {"auc_fa_vs_hit": float(auc), "split": best_split(xs, yfa.astype(int))}
        dist[str(lead)] = d
        splits[str(lead)] = sp
        top = sorted(sp.items(), key=lambda kv: -(kv[1]["split"][-1] if kv[1]["split"] else 0))[:5]
        log(f"(c) lead {lead} top splits: " + "; ".join(f"{k} {v['split'][0]}{v['split'][1]:.3g} gain {v['split'][-1]:.4f}" for k, v in top))
    report["c_distributions"] = dist
    report["c_splits"] = splits

    if args.stage == "explore":
        report["runtime_s"] = time.time() - started
        (out_dir / "explore.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
        log("explore stage done")
        return 0

    # ---------------- history features (candidate new columns) -------------
    for s in scored:
        derive_history(arrays[s])
    rows = flat_rows(arrays, scored)
    report["c_history"] = {}
    for lead in leads:
        idx_l, outc = push_idx[lead]
        ne = no_eta[idx_l]
        groups = {"no_eta_fa": ne & (outc == "false_alarm"), "no_eta_hit": ne & (outc == "hit"),
                  "eta_fa": ~ne & (outc == "false_alarm"), "eta_hit": ~ne & (outc == "hit")}
        ent = {}
        mask2 = groups["no_eta_fa"] | groups["no_eta_hit"]
        yfa = groups["no_eta_fa"][mask2].astype(int)
        for v in HISTORY:
            x = rows["f:" + v][idx_l]
            ent[v] = {g: {"p25": q(x[gm], 25), "p50": q(x[gm], 50), "p75": q(x[gm], 75),
                          "le30": float((x[gm] <= 30).mean()) if gm.any() else None}
                      for g, gm in groups.items()}
            xs = x[mask2]
            ent[v]["auc_fa_vs_hit"] = float(roc_auc(np.nan_to_num(xs, nan=-1.0), yfa.astype(float)))
            ent[v]["split"] = best_split(xs, yfa)
        report["c_history"][str(lead)] = ent

    # ---------------- (e) onset-target re-fit on no-ETA rows ----------------
    report["e_refit"] = {}
    for lead in leads:
        onset_w = onset_in_window(rows, scored, onsets, lead)
        ywin, uwin = grid.outcome(t_sec, gc, lead + DEFAULT_TOLERANCE_MIN)
        p = rows[f"p{lead}"]
        fit_mask = no_eta & eligible & np.isfinite(p) & (p >= E_MIN_P) & uwin
        res, preds = refit_onset(rows, fit_mask, onset_w, lead, log)
        report["e_refit"][str(lead)] = res
        for name, pr in preds.items():
            col = np.full(p.size, np.nan)
            col[fit_mask] = pr
            for s in scored:
                o = offset[s]
                arrays[s][f"f:{name}_{lead}"] = col[o:o + arrays[s]["gen"].size]

    # ---------------- (d) rule variants ------------------------------------
    cells, families = build_cells(leads, thr)
    log(f"(d) {len(cells)} cells")
    by_cell = run_cells(shared, scored, cells, [], args.workers, log)
    months = months_of(by_cell)
    report["d_variants"] = score_families(by_cell, families, months, thr, leads)
    report["runtime_s"] = time.time() - started
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["settings"] = {
        "decisions_dir": str(args.decisions_dir), "corpus_dir": str(args.corpus_dir),
        "thresholds_file": str(args.thresholds), "e_min_p": E_MIN_P,
        "conditions": CONDITIONS,
    }
    reading = out_dir / "reading.md"
    report["reading"] = ([l for l in reading.read_text().splitlines() if l.strip()]
                         if reading.is_file() else [])
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    log(f"wrote {out_dir}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
