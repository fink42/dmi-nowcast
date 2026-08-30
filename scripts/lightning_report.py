"""Generate the lightning-evaluation report: calibration reliability diagrams,
ROC / PR curves, confusion matrices, and the data story (corpus growth +
probability separation). Reads a probabilistic backtest parquet (written by
``lightning_backtest.py --probabilistic --output``) and the strike archive.

    .venv/bin/python scripts/lightning_report.py \\
        --parquet reports/lightning_prob_grid_v2.parquet \\
        --archive-dir data/strikes --out-dir reports/figures
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import lightning_backtest as lb  # noqa: E402
from dmi_nowcast_core.calibrate import fit_isotonic  # noqa: E402

LEADS = {3.0: 15.0, 10.0: 30.0}
SLICES = [("Denmark", 3.0), ("Denmark", 10.0), ("Alps", 3.0), ("Alps", 10.0)]
COLORS = {("Denmark", 3.0): "#2a9d8f", ("Denmark", 10.0): "#1d6fb8",
          ("Alps", 3.0): "#e76f51", ("Alps", 10.0): "#9b2226"}


def slice_xy(rows, region, ring):
    lead = LEADS[ring]
    rs = [r for r in rows if r["region"] == region and r["ring_km"] == ring
          and not r["inside_at_t"] and r.get("prob") is not None]
    rs.sort(key=lambda r: r["t_utc"])
    p = np.array([r["prob"] for r in rs])
    y = np.array([1.0 if lb._event(r, lead) else 0.0 for r in rs])
    return rs, p, y


def holdout_calibrated(rs, p, y, frac=0.66):
    """Return (test_raw, test_cal, test_y) with isotonic fit on the earlier frac."""
    n = len(rs)
    s = int(n * frac)
    cal = fit_isotonic(p[:s], y[:s])
    return p[s:], cal.predict(p[s:]), y[s:]


def reliability_bins(pred, y, nbins=10):
    edges = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(pred, edges) - 1, 0, nbins - 1)
    out = []
    for b in range(nbins):
        m = idx == b
        if m.sum() >= 5:
            out.append((pred[m].mean(), y[m].mean(), int(m.sum())))
    return np.array(out) if out else np.empty((0, 3))


def roc_points(scores, y):
    order = np.argsort(-scores, kind="mergesort")
    ys = y[order]
    P, N = ys.sum(), len(ys) - ys.sum()
    tpr = np.concatenate([[0], np.cumsum(ys) / P])
    fpr = np.concatenate([[0], np.cumsum(1 - ys) / N])
    return fpr, tpr


def pr_points(scores, y):
    order = np.argsort(-scores, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    return tp / ys.sum(), tp / (tp + fp)  # recall, precision


# --------------------------------------------------------------------------- #
def fig_reliability(rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (region, ring) in zip(axes.flat, SLICES):
        rs, p, y = slice_xy(rows, region, ring)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5, label="perfect")
        if len(rs) > 40 and y.sum() >= lb.MIN_EVENTS:
            traw, tcal, ty = holdout_calibrated(rs, p, y)
            rb = reliability_bins(traw, ty)
            cb = reliability_bins(tcal, ty)
            if len(rb):
                ax.plot(rb[:, 0], rb[:, 1], "o-", color="#bbb", label="raw")
            if len(cb):
                ax.plot(cb[:, 0], cb[:, 1], "o-", color=COLORS[(region, ring)], label="calibrated")
        ax.set_title(f"{region} · {ring:.0f} km", fontsize=11)
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle("Calibration reliability (temporal holdout) — points on the diagonal = honest probabilities",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=130); plt.close(fig)


def fig_roc_pr(rows, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    a1.plot([0, 1], [0, 1], "k--", lw=1, alpha=.4)
    for region, ring in SLICES:
        rs, p, y = slice_xy(rows, region, ring)
        if y.sum() < lb.MIN_EVENTS:
            continue
        c = COLORS[(region, ring)]
        fpr, tpr = roc_points(p, y)
        auc = lb._roc_auc(p, y)
        a1.plot(fpr, tpr, color=c, lw=1.8, label=f"{region[:3]} {ring:.0f}km  AUC={auc:.2f}")
        rec, prec = pr_points(p, y)
        ap = lb._pr_auc(p, y)
        a2.plot(rec, prec, color=c, lw=1.8, label=f"{region[:3]} {ring:.0f}km  AP={ap:.2f}")
        a2.axhline(y.mean(), color=c, ls=":", lw=.8, alpha=.5)
    a1.set_title("ROC"); a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate (recall)")
    a1.legend(fontsize=8); a1.grid(alpha=.2)
    a2.set_title("Precision–Recall (dotted = base rate)"); a2.set_xlabel("Recall"); a2.set_ylabel("Precision")
    a2.legend(fontsize=8); a2.grid(alpha=.2); a2.set_ylim(0, 1)
    fig.suptitle("Discrimination — how well the probability ranks arrivals vs non-arrivals", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=130); plt.close(fig)


def fig_confusion(rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    for col, region in enumerate(["Denmark", "Alps"]):
        for row, ring in enumerate([3.0, 10.0]):
            ax = axes[row][col]
            rs, p, y = slice_xy(rows, region, ring)
            if y.sum() < lb.MIN_EVENTS:
                ax.text(.5, .5, f"{region} {ring:.0f}km\n(too few events)", ha="center", va="center")
                ax.axis("off"); continue
            f1 = lb._best_f1(p, y); thr = f1["threshold"]
            pred = (p >= thr).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
            cm = np.array([[tp, fp], [fn, tn]], float)
            coln = cm / cm.sum(axis=0, keepdims=True)
            ax.imshow(coln, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            lab = [[f"TP\n{tp}", f"FP\n{fp}"], [f"FN\n{fn}", f"TN\n{tn}"]]
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, lab[i][j], ha="center", va="center", fontsize=12,
                            color="white" if coln[i][j] > .55 else "black", fontweight="bold")
            rec = tp / (tp + fn); prec = tp / (tp + fp) if tp + fp else 0
            spec = tn / (tn + fp); f1v = 2 * prec * rec / (prec + rec) if prec + rec else 0
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Arrival", "No arr."])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["Warn", "Quiet"])
            ax.set_title(f"{region} · {ring:.0f} km (thr={thr:.2f})\n"
                         f"recall {rec:.2f} · prec {prec:.2f} · spec {spec:.2f} · F1 {f1v:.2f}", fontsize=10)
    fig.suptitle("Confusion matrices at best-F1 (column-normalised: diagonal = recall & specificity)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=130); plt.close(fig)


def fig_data(rows, archive_dir, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    # corpus growth per day, DK vs Alps
    per_day = defaultdict(lambda: defaultdict(int))
    for f in sorted(glob.glob(str(Path(archive_dir) / "strikes_*.ndjson"))):
        day = Path(f).stem.replace("strikes_", "")
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            per_day[day][lb.region_of(float(d["lat"]), float(d["lon"]))] += 1
    days = sorted(per_day)
    dk = [per_day[d].get("Denmark", 0) for d in days]
    al = [per_day[d].get("Alps", 0) for d in days]
    x = np.arange(len(days))
    a1.bar(x, al, color="#e76f51", label="Alps")
    a1.bar(x, dk, bottom=al, color="#1d6fb8", label="Denmark")
    a1.set_xticks(x); a1.set_xticklabels([d[5:] for d in days], rotation=90, fontsize=7)
    a1.set_title("Corpus growth — strikes per day"); a1.set_ylabel("strikes"); a1.legend(fontsize=9)
    # probability separation (DK-10km): histogram of calibrated P by outcome
    rs, p, y = slice_xy(rows, "Denmark", 10.0)
    bins = np.linspace(0, 1, 21)
    a2.hist(p[y == 0], bins=bins, color="#888", alpha=.6, label="no arrival", density=True)
    a2.hist(p[y == 1], bins=bins, color="#1d6fb8", alpha=.7, label="arrival", density=True)
    a2.set_title("Probability separation — Denmark 10 km")
    a2.set_xlabel("predicted P"); a2.set_ylabel("density"); a2.legend(fontsize=9)
    fig.suptitle("The data", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=130); plt.close(fig)


def print_stats(rows):
    print(f"\n{'='*92}\nSKILL SUMMARY (post-storm corpus)\n{'='*92}")
    h = (f"{'region':<9}{'ring':>5}{'n':>7}{'evt':>5}{'base%':>7}{'ROC':>6}{'PR':>6}"
         f"{'F1':>6}{'prec':>6}{'rec':>6}{'Brier_r':>9}{'Brier_c':>9}{'BSS_r':>7}{'BSS_c':>7}")
    print(h); print("-" * len(h))
    for region, ring in SLICES:
        sl = [r for r in rows if r["region"] == region and r["ring_km"] == ring]
        m = lb._prob_metrics(sl, LEADS[ring])
        c = lb._calibration_holdout(sl, LEADS[ring])
        if m["n"] == 0:
            continue
        def f(x, w=6, p=2):
            return f"{x:>{w}.{p}f}" if isinstance(x, float) else f"{'—':>{w}}"
        print(f"{region:<9}{ring:>5.0f}{m['n']:>7}{m['events']:>5}{100*m['base_rate']:>6.1f}%"
              f"{f(m['auc'])}{f(m['pr_auc'])}{f(m['f1'])}{f(m['f1_prec'])}{f(m['f1_rec'])}"
              f"{f(c.get('brier_raw'),9,3)}{f(c.get('brier_cal'),9,3)}{f(c.get('bss_raw'),7,3)}{f(c.get('bss_cal'),7,3)}")
    print("-" * len(h))
    print("⚠ n counts pseudo-replicated grid rows (5-min stride × overlapping targets);\n"
          "  effective sample size ≈ independent storm episodes ≪ n — treat the metrics\n"
          "  as point estimates, not n-backed confidence.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--archive-dir", default=str(ROOT / "data" / "strikes"))
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "figures"))
    args = ap.parse_args()
    rows = pq.read_table(args.parquet).to_pylist()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(rows)} rows from {args.parquet}")
    fig_reliability(rows, out / "1_reliability.png")
    fig_roc_pr(rows, out / "2_roc_pr.png")
    fig_confusion(rows, out / "3_confusion.png")
    fig_data(rows, args.archive_dir, out / "4_data.png")
    print(f"Wrote 4 figures → {out}")
    print_stats(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
