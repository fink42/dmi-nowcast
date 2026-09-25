#!/usr/bin/env python3
"""Can a simple rule over BOTH probabilities beat the served push rule?

Two out-of-fold probabilities exist per decision row: ``p_post_{L}`` (the
served model, "gauge wet within L") and ``p_onset_{L}`` (the onset model,
"rain STARTS within L"). This study replays a few simple rules over both:

* ``served`` — ``p_post >= b`` at the shipped threshold (the reference);
* ``onset``  — ``p_onset >= a``;
* ``or``     — ``p_onset >= a  OR  p_post >= b``;
* ``and``    — ``p_onset >= a  AND p_post >= b`` (does p_post filter the
  early low-p onset fires?);
* ``blend``  — ``w * p_onset + (1 - w) * p_post >= t`` (the smooth middle
  between the two; only with ``--blend``).

How a rule reaches the engine: each row's fire decision is computed with
the engine's own comparison (``p >= pct / 100``) and handed to
``push.engine.evaluate`` as an indicator probability (1.0 = fire, 0.0 =
not, NaN = a row with a missing input, skipped exactly as a null
probability is) at threshold 50 %. The state machine only ever reads
``over``, so this is the rule, not an approximation of it: the served cell
run this way must reproduce the sweep's counts EXACTLY (the sanity gate).
Replay is ``eta_revision_study.replay_pushes`` (a line-for-line mirror of
``threshold_sweep.replay_station``); truth is ``threshold_sweep.
gauge_truth``; grading is ``warning_score.score_warnings`` with the
sweep's settings.

Selection is leave-one-month-out: for each held-out month a family's
parameters are chosen on the other months by "max F1 subject to precision
>= served AND recall >= served (both on the training months); if no cell
qualifies, max F1", and the chosen cell scores the held-out month. The
out-of-fold counts pool over months. CIs are a paired day-block bootstrap
of the OOF rule minus served.

Offline and read-only: parquet in, ``report.md`` / ``results.json`` out.

Usage (from ``sidecar/``)::

    PYTHONPATH=../src:. .venv/bin/python ../scripts/combined_rule_study.py \\
        --post-dir ~/dmi-nowcast-corpus-local/stations/replay_rp_trees_all_oof/decisions \\
        --onset-dir ~/dmi-nowcast-corpus-local/stations/replay_onset_trees_all_oof/decisions \\
        --corpus-dir ~/dmi-nowcast-corpus-local \\
        --out-dir ~/dmi-nowcast-corpus-local/stations/combined_rule_study --workers 8
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
from typing import Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "sidecar", _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eta_revision_study as ers  # noqa: E402
from dmi_nowcast_core.postprocess import season_of_month  # noqa: E402
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
from dmi_nowcast_sidecar.threshold_sweep import FIT_MIN_USEFUL_LEAD_MIN  # noqa: E402

LEADS = (20, 30, 45, 60)
#: The shipped thresholds on p_post (thresholds_rp_trees_all).
SERVED_PCT = {20: 35, 30: 45, 45: 60, 60: 65}
SWEEP_EXPECTED = ers.SWEEP_EXPECTED
#: thresholds_onset_trees_all_low/sweep.md, lead 20 at 26 %.
ONSET_EXPECTED = {"lead": 20, "pct": 26, "hits": 3111, "false_alarms": 6638,
                  "late": 281, "misses": 8660}

ONSET_GRID = tuple(range(4, 41, 2))
A_GRID = tuple(range(10, 41, 2))
B_GRID = tuple(range(35, 86, 5))
BLEND_W = (0.25, 0.5, 0.75)
BLEND_T = tuple(range(10, 71, 2))

#: Count slots per (cell, day): hits, late, false alarms, pending, misses.
N_SLOTS = 5
BOOTSTRAP_N = 500
US_PER_DAY = 86_400_000_000


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def build_cells(leads: Sequence[int], blend: bool = False) -> list[tuple]:
    """Cells ``(lead, family, x, y)``; x/y are percent (blend: x = w)."""
    cells = []
    for lead in leads:
        cells.append((lead, "served", 0, SERVED_PCT[lead]))
        cells += [(lead, "onset", a, 0) for a in ONSET_GRID]
        cells += [(lead, "or", a, b) for a in A_GRID for b in B_GRID]
        cells += [(lead, "and", a, b) for a in A_GRID for b in B_GRID]
        if blend:
            cells += [(lead, "blend", w, t) for w in BLEND_W for t in BLEND_T]
    return cells


def _over(p: np.ndarray, pct: float) -> np.ndarray:
    """The engine's comparison, verbatim: ``p >= threshold_pct / 100``."""
    with np.errstate(invalid="ignore"):
        return p >= pct / 100


def rule_column(p_onset: np.ndarray, p_post: np.ndarray, family: str,
                x: float, y: float) -> np.ndarray:
    """Indicator probability the engine sees at 50 %: 1 fire, 0 not, NaN skip.

    A row with a missing input the rule reads is NaN (the engine's replay
    skips it, as it skips a null probability).
    """
    if family == "served":
        fire, missing = _over(p_post, y), np.isnan(p_post)
    elif family == "onset":
        fire, missing = _over(p_onset, x), np.isnan(p_onset)
    else:
        missing = np.isnan(p_onset) | np.isnan(p_post)
        if family == "or":
            fire = _over(p_onset, x) | _over(p_post, y)
        elif family == "and":
            fire = _over(p_onset, x) & _over(p_post, y)
        elif family == "blend":
            fire = _over(x * p_onset + (1 - x) * p_post, y)
        else:
            raise ValueError(family)
    return np.where(missing, np.nan, fire.astype(float))


# ---------------------------------------------------------------------------
# Loading and the 1:1 join
# ---------------------------------------------------------------------------


def load_onset(onset_dir: Path, leads: Sequence[int]) -> tuple[dict, dict]:
    """Per-station ``{radar, gen, o<lead>, eta, observed}``, dedup last-file-wins."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    wanted = ["radar_ts", "generated_at", "station_id", "eta_min", "observed_mm_h"] + [
        f"p_onset_{lead}" for lead in leads]
    files = sorted(Path(onset_dir).rglob("*.parquet"))
    tables = []
    for order, path in enumerate(files):
        t = pq.read_table(path, columns=wanted)
        tables.append(t.append_column("_file", pa.array(np.full(t.num_rows, order, np.int32))))
    table = pa.concat_tables(tables)

    def ts(name):
        arr = table.column(name).cast(pa.timestamp("us", tz="UTC")).to_numpy(zero_copy_only=False)
        return arr.astype("datetime64[us]").astype(np.int64)

    def f64(name):
        return table.column(name).cast(pa.float64()).to_numpy(zero_copy_only=False)

    radar, gen = ts("radar_ts"), ts("generated_at")
    codes, stations = ers._encode(np.asarray(table.column("station_id").to_pylist(), dtype=object))
    fields = {"eta": f64("eta_min"), "observed": f64("observed_mm_h")}
    for lead in leads:
        fields[f"o{lead}"] = f64(f"p_onset_{lead}")
    order = np.lexsort((table.column("_file").to_numpy(), radar, codes))
    c_s, r_s = codes[order], radar[order]
    last = np.ones(order.size, bool)
    last[:-1] = (c_s[1:] != c_s[:-1]) | (r_s[1:] != r_s[:-1])
    counts = {"files": len(files), "rows": table.num_rows,
              "duplicates": int(order.size - last.sum())}
    order = order[last]
    out = {}
    for chunk in np.split(order, np.flatnonzero(np.diff(codes[order])) + 1):
        if chunk.size:
            a = {"radar": radar[chunk], "gen": gen[chunk]}
            a.update({k: v[chunk] for k, v in fields.items()})
            out[stations[codes[chunk[0]]]] = a
    return out, counts


def join_onset(post: dict, onset: dict, leads: Sequence[int]) -> dict:
    """Attach ``o<lead>`` to each post station's arrays; count the join.

    Matches on (station, radar_ts). Unmatched post rows get NaN (skipped by
    any rule that reads p_onset). ``generated_at`` / ETA / observed of the
    matched pairs are compared, since the rule inputs must be the same row.
    """
    j = {"post_rows": 0, "onset_rows": 0, "matched": 0, "post_only": 0,
         "onset_only": 0, "gen_mismatch": 0, "eta_mismatch": 0, "observed_mismatch": 0}
    for sid, a in post.items():
        n = a["radar"].size
        j["post_rows"] += n
        for lead in leads:
            a[f"o{lead}"] = np.full(n, np.nan)
        b = onset.get(sid)
        if b is None:
            j["post_only"] += n
            continue
        _, ia, ib = np.intersect1d(a["radar"], b["radar"], assume_unique=True, return_indices=True)
        j["matched"] += ia.size
        j["post_only"] += n - ia.size
        for lead in leads:
            a[f"o{lead}"][ia] = b[f"o{lead}"][ib]
        j["gen_mismatch"] += int((a["gen"][ia] != b["gen"][ib]).sum())
        for key, name in (("eta", "eta_mismatch"), ("observed", "observed_mismatch")):
            x, y = a[key][ia], b[key][ib]
            j[name] += int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum())
    j["onset_rows"] = sum(b["radar"].size for b in onset.values())
    j["onset_only"] = j["onset_rows"] - j["matched"]
    return j


# ---------------------------------------------------------------------------
# Replay and grading, per station
# ---------------------------------------------------------------------------


def score_station(arrays, onsets, known_until, cells, day0_us, n_days) -> dict:
    """``{cell: int array (n_days, 5)}`` of hits/late/fa/pending/misses by UTC day.

    Warnings are dated by their send instant, misses by the onset.
    """
    runs = ers.run_ids(arrays["radar"])
    radar_dt = [ers.to_dt(v) for v in arrays["radar"]]
    gen_dt = [ers.to_dt(v) for v in arrays["gen"]]
    gen, eta = arrays["gen"], arrays["eta"]
    slot = {"hit": 0, "late": 1, "false_alarm": 2, "pending": 3}
    out = {}
    coverage_by_lead = {}
    for cell in cells:
        lead, family, x, y = cell
        col = rule_column(arrays[f"o{lead}"], arrays[f"p{lead}"], family, x, y)
        pushes, _ = ers.replay_pushes({**arrays, f"p{lead}": col}, lead, 50,
                                      runs=runs, radar_dt=radar_dt, gen_dt=gen_dt)
        pushes.sort(key=lambda i: (gen[i], i))
        if lead not in coverage_by_lead:
            coverage_by_lead[lead] = coverage_runs(
                radar_dt, max_gap_min=DEFAULT_COVERAGE_GAP_MIN,
                extend_min=lead + DEFAULT_TOLERANCE_MIN)
        res = score_warnings(
            [(gen_dt[i], ers._opt(eta[i])) for i in pushes], list(onsets),
            lead_min=int(lead), tolerance_min=DEFAULT_TOLERANCE_MIN,
            dry_min=DEFAULT_DRY_MIN, onset_min_mm=DEFAULT_ONSET_MIN_MM,
            known_until=known_until, coverage=coverage_by_lead[lead],
            min_useful_lead_min=FIT_MIN_USEFUL_LEAD_MIN,
        )
        counts = np.zeros((n_days, N_SLOTS), np.int32)
        for i, w in zip(pushes, res.warnings):
            counts[(int(gen[i]) - day0_us) // US_PER_DAY, slot[w.outcome]] += 1
        for o in res.onsets:
            if o.outcome == "miss":
                d = (ers.to_us(o.onset_utc) - day0_us) // US_PER_DAY
                if 0 <= d < n_days:
                    counts[d, 4] += 1
        out[cell] = counts
    return out


_SHARED: dict | None = None


def _work(task):
    station, lead = task
    s = _SHARED
    cells = [c for c in s["cells"] if c[0] == lead]
    return score_station(s["arrays"][station], s["onsets"].get(station, ()),
                         s["known_until"].get(station), cells, s["day0"], s["n_days"])


def run_cells(shared: dict, stations, leads, workers: int, log) -> dict:
    """``{cell: (n_days, 5)}`` summed over stations."""
    global _SHARED
    _SHARED = shared
    total = {c: np.zeros((shared["n_days"], N_SLOTS), np.int64) for c in shared["cells"]}
    tasks = [(st, lead) for lead in leads for st in stations]
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for n, res in enumerate(pool.map(_work, tasks, chunksize=1), 1):
            for c, v in res.items():
                total[c] += v
            if n % 25 == 0:
                log(f"  {n}/{len(tasks)} station-leads done")
    _SHARED = None
    return total


# ---------------------------------------------------------------------------
# Metrics, selection, bootstrap
# ---------------------------------------------------------------------------


def metrics(c: np.ndarray) -> dict:
    """Pooled metrics from a 5-slot count vector."""
    h, late, fa, pend, miss = (int(x) for x in c)
    sk = skill_scores(h, fa, miss, late)
    return {"warnings": h + late + fa, "hits": h, "late": late, "false_alarms": fa,
            "pending": pend, "misses": miss, "precision": sk["precision"],
            "recall": sk["recall"], "f1": sk["f1"]}


def _pr(c: np.ndarray) -> tuple[float, float]:
    """Vectorised precision / recall over the last axis (NaN on empty)."""
    h, late, fa, miss = c[..., 0], c[..., 1], c[..., 2], c[..., 4]
    with np.errstate(invalid="ignore", divide="ignore"):
        return h / (h + fa), h / (h + late + miss)


def pick_cell(train: Mapping[tuple, np.ndarray], served: np.ndarray) -> tuple:
    """Max F1 among cells beating served on both precision and recall (>=).

    ``train`` maps cell -> pooled 5-slot counts on the training months;
    ``served`` is the served rule's counts on the same months. Falls back
    to plain max F1 when no cell qualifies. Ties go to the first cell.
    """
    sp, sr = _pr(served)
    best, best_f1, best_ok = None, -1.0, False
    for cell, c in train.items():
        p, r = _pr(c)
        if not (p > 0 and r > 0):
            continue
        f1 = 2 * p * r / (p + r)
        ok = bool(p >= sp and r >= sr)
        if (ok, f1) > (best_ok, best_f1):
            best, best_f1, best_ok = cell, f1, ok
    return best, best_ok


def lomo(daily: Mapping[tuple, np.ndarray], served_cell: tuple, family_cells: Sequence[tuple],
         day_month: np.ndarray) -> dict:
    """Leave-one-month-out: pick on the other months, score the held-out one.

    Returns the OOF per-day counts (n_days, 5) and the pick per month.
    """
    months = sorted(set(day_month.tolist()))
    oof = np.zeros_like(daily[served_cell])
    picks = {}
    for m in months:
        train_days = day_month != m
        train = {c: daily[c][train_days].sum(0) for c in family_cells}
        cell, ok = pick_cell(train, daily[served_cell][train_days].sum(0))
        if cell is None:
            continue
        held = day_month == m
        oof[held] = daily[cell][held]
        picks[int(m)] = {"cell": list(cell[1:]), "beats_served_in_train": ok}
    return {"daily": oof, "picks": picks}


def bootstrap_delta(rule: np.ndarray, served: np.ndarray, n: int = BOOTSTRAP_N,
                    seed: int = 0) -> dict:
    """Paired day-block bootstrap 95 % CI on Δprecision and Δrecall."""
    rng = np.random.default_rng(seed)
    n_days = rule.shape[0]
    w = np.stack([np.bincount(rng.integers(0, n_days, n_days), minlength=n_days)
                  for _ in range(n)]).astype(float)
    pr, rr = _pr(w @ rule)
    ps, rs = _pr(w @ served)
    dp, dr = pr - ps, rr - rs
    return {"d_precision_ci": [float(np.nanpercentile(dp, 2.5)), float(np.nanpercentile(dp, 97.5))],
            "d_recall_ci": [float(np.nanpercentile(dr, 2.5)), float(np.nanpercentile(dr, 97.5))]}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _f(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def _cell_label(family: str, x, y) -> str:
    if family == "served":
        return f"p_post ≥ {y}"
    if family == "onset":
        return f"p_onset ≥ {x}"
    if family == "or":
        return f"p_onset ≥ {x} OR p_post ≥ {y}"
    if family == "and":
        return f"p_onset ≥ {x} AND p_post ≥ {y}"
    return f"{x}·p_onset + {1 - x:g}·p_post ≥ {y}"


def render_markdown(r: dict) -> str:
    L = ["# Combined-rule study (p_onset × p_post)", "",
         f"Generated {r['generated_at_utc']}. Runtime {r['runtime_s']:.0f} s. "
         "Script `scripts/combined_rule_study.py`.", ""]
    j = r["join"]
    L += ["## Join and sanity gate", "",
          f"Join on (station_id, radar_ts), each side deduped last-file-wins: post rows {j['post_rows']}, "
          f"onset rows {j['onset_rows']}, matched {j['matched']}, post-only {j['post_only']}, "
          f"onset-only {j['onset_only']}; generated_at / ETA / observed mismatches on matched rows: "
          f"{j['gen_mismatch']} / {j['eta_mismatch']} / {j['observed_mismatch']}. "
          f"Stations scored: {r['stations_scored']}; station-days: {r['station_days']}.", "",
          f"**{'PASSED' if r['sanity_gate']['passed'] else 'FAILED'}** (exact match required).", "",
          "| check | expected | got |", "| --- | ---: | ---: |"]
    for g in r["sanity_gate"]["rows"]:
        L.append(f"| {g['check']} | {g['expected']} | {g['got']} |")
    L.append("")
    if not r["sanity_gate"]["passed"]:
        return "\n".join(L + ["The gate failed; no other number is reported."]) + "\n"

    L += ["## Out-of-sample (leave-one-month-out), pooled", "",
          "Per held-out month the family's cell is chosen on the other months by max F1 subject to "
          "precision ≥ served AND recall ≥ served on those months (else max F1). Δ = rule − served; "
          f"95 % CI from a paired day-block bootstrap ({BOOTSTRAP_N} resamples). "
          "Served thresholds were themselves fitted on the full window (in-sample), which favours served.", ""]
    for lead, e in r["leads"].items():
        L += [f"### Lead {lead} min", "",
              "| rule | warnings | hits | late | false alarms | misses | precision | recall | F1 | warnings / station-day | Δprecision [95 % CI] | Δrecall [95 % CI] | months beating served in training |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |"]
        for fam, x in e["oof"].items():
            m = x["pooled"]
            if fam == "served":
                d = "–", "–"
                ok = "–"
            else:
                ci = x["bootstrap"]
                d = (f"{m['precision'] - e['oof']['served']['pooled']['precision']:+.3f} "
                     f"[{ci['d_precision_ci'][0]:+.3f}, {ci['d_precision_ci'][1]:+.3f}]",
                     f"{m['recall'] - e['oof']['served']['pooled']['recall']:+.3f} "
                     f"[{ci['d_recall_ci'][0]:+.3f}, {ci['d_recall_ci'][1]:+.3f}]")
                ok = f"{sum(p['beats_served_in_train'] for p in x['picks'].values())}/{len(x['picks'])}"
            L.append(f"| {x['label']} | {m['warnings']} | {m['hits']} | {m['late']} | {m['false_alarms']} | "
                     f"{m['misses']} | {_f(m['precision'])} | {_f(m['recall'])} | {_f(m['f1'])} | "
                     f"{_f(x['per_station_day'], 2)} | {d[0]} | {d[1]} | {ok} |")
        L.append("")
        for fam, x in e["oof"].items():
            if fam != "served":
                picks = ", ".join(f"{k}: {_cell_label(*v['cell'])}" for k, v in x["picks"].items())
                L.append(f"- {fam} picks by held-out month: {picks}")
        L.append("")

    L += ["## In-sample Pareto: cells beating served on BOTH precision and recall (full window)", "",
          "Top 5 by F1 per family and lead; the count is all such cells.", ""]
    for lead, e in r["leads"].items():
        s = e["served_in_sample"]
        L.append(f"**Lead {lead}** — served {_f(s['precision'])} / {_f(s['recall'])} (F1 {_f(s['f1'])})")
        L.append("")
        L += ["| family | cells beating served | top cells (precision / recall / F1 / warnings) |",
              "| --- | ---: | --- |"]
        for fam, cells in e["pareto"].items():
            top = "; ".join(f"{_cell_label(fam, *c['cell'])}: {_f(c['precision'])} / {_f(c['recall'])} / "
                            f"{_f(c['f1'])} / {c['warnings']}" for c in cells[:5])
            L.append(f"| {fam} | {len(cells)} | {top or '–'} |")
        L.append("")

    L += ["## Season split of the recommended family (OOF)", ""]
    for lead, e in r["leads"].items():
        fam = e["recommended"]["family"]
        L += [f"**Lead {lead}** — {fam}", "",
              "| season | served P / R | rule P / R | ΔP | ΔR | rule warnings |", "| --- | --- | --- | ---: | ---: | ---: |"]
        for season, v in e["season"].items():
            s, x = v["served"], v["rule"]
            dp = None if x["precision"] is None or s["precision"] is None else x["precision"] - s["precision"]
            dr = None if x["recall"] is None or s["recall"] is None else x["recall"] - s["recall"]
            L.append(f"| {season} | {_f(s['precision'])} / {_f(s['recall'])} | {_f(x['precision'])} / "
                     f"{_f(x['recall'])} | {_f(dp)} | {_f(dr)} | {x['warnings']} |")
        L.append("")

    L += ["## Recommendation", ""]
    for lead, e in r["leads"].items():
        rec = e["recommended"]
        L.append(f"- Lead {lead}: **{rec['family']}** — full-window cell `{rec['label']}`; "
                 f"beats served on both out of sample: {'yes' if rec['beats_both_oof'] else 'no'}"
                 f"{' (both CIs exclude 0)' if rec['both_ci_positive'] else ''}.")
    L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--post-dir", type=Path, required=True)
    ap.add_argument("--onset-dir", type=Path, required=True)
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--leads", default="20,30,45,60")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--blend", action="store_true", help="add the blend family")
    ap.add_argument("--gate-only", action="store_true", help="run only the sanity-gate cells")
    ap.add_argument("--render-only", action="store_true",
                    help="re-render report.md from results.json in --out-dir")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    from dmi_nowcast_sidecar.threshold_sweep import gauge_truth

    args = build_parser().parse_args(argv)
    started = time.time()
    out = Path(args.out_dir)
    if args.render_only:
        report = json.loads((out / "results.json").read_text(encoding="utf-8"))
        (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
        return 0

    def log(msg: str) -> None:
        print(f"[{time.time() - started:7.1f}s] {msg}", file=sys.stderr, flush=True)

    leads = [int(x) for x in args.leads.split(",") if x.strip()]
    post, _ = ers.load_station_arrays(args.post_dir, leads, log=log)
    onset, onset_counts = load_onset(args.onset_dir, leads)
    log(f"onset: {onset_counts}")
    join = join_onset(post, onset, leads)
    del onset
    log(f"join: {join}")

    stations = sorted(post)
    lo = min(int(a["radar"].min()) for a in post.values())
    hi = max(int(a["radar"].max()) for a in post.values())
    onsets, known_until, _slots, dead = gauge_truth(
        Path(args.corpus_dir), stations, (ers.to_dt(lo), ers.to_dt(hi)),
        dry_min=DEFAULT_DRY_MIN, onset_min_mm=DEFAULT_ONSET_MIN_MM,
        min_known_slots=DEFAULT_MIN_KNOWN_SLOTS, log=log)
    scored = [s for s in stations if s in known_until]
    day0 = lo - lo % US_PER_DAY
    n_days = int((hi + (60 + DEFAULT_TOLERANCE_MIN) * ers.US_PER_MIN - day0) // US_PER_DAY) + 1
    station_days = sum(np.unique((post[s]["gen"] - day0) // US_PER_DAY).size for s in scored)
    day_dt = [ers.to_dt(day0 + d * US_PER_DAY) for d in range(n_days)]
    day_month = np.array([d.year * 12 + d.month - 1 for d in day_dt])
    day_season = np.array([season_of_month(d.month) for d in day_dt])

    cells = build_cells(leads, blend=args.blend)
    if args.gate_only:
        cells = [c for c in cells if c[1] == "served"
                 or (c[1] == "onset" and c[0] == ONSET_EXPECTED["lead"] and c[2] == ONSET_EXPECTED["pct"])]
    log(f"{len(scored)} scored stations, {len(cells)} cells, {n_days} days; {args.workers} workers")
    shared = {"arrays": post, "onsets": onsets, "known_until": known_until,
              "cells": cells, "day0": day0, "n_days": n_days}
    daily = run_cells(shared, scored, leads, int(args.workers), log)

    # --- sanity gate: exact ------------------------------------------------
    gate = []
    for lead in leads:
        m = metrics(daily[(lead, "served", 0, SERVED_PCT[lead])].sum(0))
        for key in ("warnings", "hits"):
            gate.append({"check": f"served {lead} min @ {SERVED_PCT[lead]} % {key}",
                         "expected": SWEEP_EXPECTED[lead][key], "got": m[key]})
    oc = (ONSET_EXPECTED["lead"], "onset", ONSET_EXPECTED["pct"], 0)
    if oc in daily:
        m = metrics(daily[oc].sum(0))
        for key in ("hits", "false_alarms", "late", "misses"):
            gate.append({"check": f"onset 20 min @ 26 % {key}", "expected": ONSET_EXPECTED[key],
                         "got": m[key]})
    passed = all(g["expected"] == g["got"] for g in gate)
    log(f"sanity gate {'PASSED' if passed else 'FAILED'}: {gate}")

    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "join": join, "onset_load": onset_counts, "dead_gauges": list(dead),
              "stations_scored": len(scored), "station_days": int(station_days),
              "sanity_gate": {"passed": passed, "rows": gate}, "leads": {},
              "settings": {"served_pct": SERVED_PCT, "onset_grid": ONSET_GRID, "a_grid": A_GRID,
                           "b_grid": B_GRID, "blend": bool(args.blend), "bootstrap_n": BOOTSTRAP_N}}
    if passed and not args.gate_only:
        families = ["onset", "or", "and"] + (["blend"] if args.blend else [])
        for lead in leads:
            served_cell = (lead, "served", 0, SERVED_PCT[lead])
            sd = daily[served_cell]
            served_m = metrics(sd.sum(0))
            e = {"served_in_sample": served_m, "oof": {}, "pareto": {}}
            e["oof"]["served"] = {"label": _cell_label("served", 0, SERVED_PCT[lead]),
                                  "pooled": served_m,
                                  "per_station_day": served_m["warnings"] / station_days}
            oof_daily = {}
            for fam in families:
                fcells = [c for c in cells if c[0] == lead and c[1] == fam]
                res = lomo(daily, served_cell, fcells, day_month)
                pm = metrics(res["daily"].sum(0))
                oof_daily[fam] = res["daily"]
                e["oof"][fam] = {
                    "label": f"{fam} (LOMO)", "pooled": pm,
                    "per_station_day": pm["warnings"] / station_days,
                    "bootstrap": bootstrap_delta(res["daily"], sd),
                    "picks": {f"{k // 12}-{k % 12 + 1:02d}": v for k, v in res["picks"].items()},
                }
                beat = []
                for c in fcells:
                    cm = metrics(daily[c].sum(0))
                    if (cm["precision"] or 0) > served_m["precision"] and (cm["recall"] or 0) > served_m["recall"]:
                        beat.append({"cell": list(c[2:]), **cm})
                e["pareto"][fam] = sorted(beat, key=lambda c: -c["f1"])
            # Recommended family: both bootstrap CIs above 0, then both OOF
            # point deltas above 0, then the highest OOF F1.
            def rank(fam):
                pm, ci = e["oof"][fam]["pooled"], e["oof"][fam]["bootstrap"]
                both = pm["precision"] > served_m["precision"] and pm["recall"] > served_m["recall"]
                sig = ci["d_precision_ci"][0] > 0 and ci["d_recall_ci"][0] > 0
                return (sig, both, pm["f1"])
            best = max(families, key=rank)
            bx = e["oof"][best]
            full_cell, _ = pick_cell({c: daily[c].sum(0) for c in cells if c[0] == lead and c[1] == best},
                                     sd.sum(0))
            ci = bx["bootstrap"]
            e["recommended"] = {
                "family": best, "cell": list(full_cell[2:]), "label": _cell_label(best, *full_cell[2:]),
                "beats_both_oof": bool(rank(best)[1]),
                "both_ci_positive": bool(ci["d_precision_ci"][0] > 0 and ci["d_recall_ci"][0] > 0),
            }
            e["season"] = {
                s: {"served": metrics(sd[day_season == s].sum(0)),
                    "rule": metrics(oof_daily[best][day_season == s].sum(0))}
                for s in ("summer", "winter", "shoulder")
            }
            report["leads"][str(lead)] = e
    report["runtime_s"] = time.time() - started
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    log(f"wrote {out}/report.md and results.json")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
