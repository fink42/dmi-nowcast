"""Lightning ETA backtest: replay the self-archived strike stream, score the
ETA against what actually happened, stratified by region (Denmark vs Alps).

For lightning the strikes are *both* the input and the ground truth. At
each sampled time ``T`` we predict the
ETA of a storm cell's leading edge to each ring (3 / 10 km) around each target,
using only strikes ``<= T`` (reuse :func:`dmi_nowcast_core.lightning.compute_eta`);
the "truth" is whether — and when — a strike actually entered that ring in
``(T, T+H]``.

One Parquet row per ``(T, target, ring)`` with a ``region`` column, mirroring
``phase4_backtest.py``. Metrics are reported **per (region, ring)** — never only
pooled — because the corpus is ~87 % Alps / 13 % Denmark and a pooled number is
really an Alpine number.

Targets default to one Danish anchor plus the two Alpine anchors that
produced the archived Alpine sample; moving targets can't be replayed. The
Alpine anchors stopped *collecting* on 2026-08-14 but stay here: their
season is banked in the archive and remains the only Alpine sample there
is, so dropping them as targets would silently delete the Alps rows from
every skill table. Add your own with ``--extra-target``. Strikes are read
from the service's NDJSON archive — copy it into ``data/strikes/`` first.

Examples:
    # As-deployed evaluation (default EtaParams == live LightningConfig):
    .venv/bin/python scripts/lightning_backtest.py \\
        --archive-dir data/strikes --output reports/lightning_backtest.parquet

    # Parameter-tuning sweep (per-region skill table, no Parquet):
    .venv/bin/python scripts/lightning_backtest.py --archive-dir data/strikes --grid
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dmi_nowcast_core.calibrate import brier_score, fit_isotonic, reliability_curve  # noqa: E402
from dmi_nowcast_core.lightning import (  # noqa: E402
    STATE_APPROACHING,
    STATE_INSIDE,
    EtaParams,
    LightningStrike,
    ProbParams,
    compute_eta,
    estimate_velocity,
    haversine_km,
    smooth_eta,
    strike_probability,
)
from dmi_nowcast_core.regions import region_of  # noqa: E402

# Equirectangular km-per-degree (matches lightning.py's local-ENU constants).
_KM_PER_DEG_LAT = 110.574
_KM_PER_DEG_LON = 111.320  # × cos(lat)


def _enu_km(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """East/north offset (km) of (lat, lon) relative to (lat0, lon0)."""
    east = (lon - lon0) * _KM_PER_DEG_LON * math.cos(math.radians(lat0))
    north = (lat - lat0) * _KM_PER_DEG_LAT
    return east, north


def _lagrangian_eta(offsets: list[tuple[float, float]], ve: float, vn: float,
                    ring_km: float, horizon_min: float) -> float | None:
    """Earliest time (min, ≤ horizon) any of the recent strikes at ENU ``offsets``
    (km from target), advected by bulk velocity (``ve``, ``vn``) km/min, first
    enters ``ring_km`` — the Lagrangian-persistence baseline's predicted ETA, or
    ``None`` if none reach the ring within the horizon."""
    a = ve * ve + vn * vn
    best: float | None = None
    for e0, n0 in offsets:
        c = e0 * e0 + n0 * n0 - ring_km * ring_km
        if c <= 0:          # already inside (shouldn't occur for non-inside rows)
            t = 0.0
        elif a < 1e-9:      # stationary field, outside ring → never enters
            continue
        else:               # smallest non-negative root of |p + v·t|² = r²
            b = 2.0 * (e0 * ve + n0 * vn)
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            sq = math.sqrt(disc)
            roots = [r for r in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)) if r >= 0]
            if not roots:
                continue
            t = min(roots)
        if t <= horizon_min and (best is None or t < best):
            best = t
    return best

# Default evaluation targets. (name, lat, lon). The Fyn anchor is a live
# collection anchor; the two Alpine ones are retired from collection
# (2026-08-14) but kept here to keep scoring the archived Alpine season. The
# four other Danish anchors (Aalborg, Silkeborg, Esbjerg, Ringsted) are
# deliberately NOT here — adding them would change every published metric,
# so add them with --extra-target when you want the wider Danish sample.
DEFAULT_TARGETS: list[tuple[str, float, float]] = [
    ("Fyn", 55.33, 10.32),
    ("Mont Blanc", 45.8326, 6.8652),
    ("Matterhorn", 45.9763, 7.6586),
]


def make_grid_targets(
    anchors: list[tuple[str, float, float]], grid_km: float, radius_km: float,
) -> list[tuple[str, float, float]]:
    """Quasi-independent grid of evaluation targets around each anchor.

    Points are spaced ``grid_km`` apart (≈ storm decorrelation scale, so each is
    a near-independent "what if the target were here") and kept within ``radius_km``
    of
    the anchor. ``radius_km`` must be ≤ (collection_radius − relevance_radius) so
    every target's relevance neighbourhood is fully archived (default 40 km =
    100 − 60). Overlapping discs (the two Alps anchors) are de-duplicated."""
    seen: set[tuple[float, float]] = set()
    out: list[tuple[str, float, float]] = []
    n = int(radius_km / grid_km)
    for aname, alat, alon in anchors:
        dlat = grid_km / _KM_PER_DEG_LAT
        for i in range(-n, n + 1):
            tlat = alat + i * dlat
            dlon = grid_km / (_KM_PER_DEG_LON * math.cos(math.radians(tlat)))
            for j in range(-n, n + 1):
                tlon = alon + j * dlon
                if haversine_km(alat, alon, tlat, tlon) > radius_km:
                    continue
                key = (round(tlat, 3), round(tlon, 3))
                if key in seen:
                    continue
                seen.add(key)
                out.append((f"{aname[:3]}_{i:+d}{j:+d}", tlat, tlon))
    return out


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_strikes(archive_dir: Path) -> list[LightningStrike]:
    """Load every archived strike, sorted ascending by time. Drops dup lines."""
    seen: set[tuple] = set()
    out: list[LightningStrike] = []
    for f in sorted(archive_dir.glob("strikes_*.ndjson")):
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    lat, lon, t = float(d["lat"]), float(d["lon"]), str(d["t"])
                except (ValueError, KeyError):
                    continue
                key = (lat, lon, t)
                if key in seen:
                    continue
                seen.add(key)
                ts = datetime.fromisoformat(t)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                out.append(LightningStrike(lat=lat, lon=lon, t=ts.astimezone(timezone.utc)))
    out.sort(key=lambda s: s.t)
    return out


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def run_backtest(
    strikes: list[LightningStrike],
    targets: list[tuple[str, float, float]],
    params: EtaParams,
    rings_km: list[float],
    *,
    stride_min: float,
    horizon_min: float,
    max_buffer: int,
    smoothing: bool = False,
    tau_min: float = 3.0,
    max_gap_min: float = 10.0,
    multicell: bool = False,
    prob_leads: dict[float, float] | None = None,
    prob_params: ProbParams = ProbParams(),
) -> list[dict]:
    """Replay the archive and return one row per ``(T, target, ring)`` where a
    storm was either detectable near the target or actually arrived in-ring. When
    ``prob_leads`` (ring→lead) is given, also attach the Phase-2 probabilistic
    forecast ``prob`` per ring row."""
    if not strikes:
        return []
    rings_sorted = sorted(rings_km)
    largest_ring = rings_sorted[-1]
    epoch = np.array([s.t.timestamp() for s in strikes], dtype=float)
    t0, t1 = epoch[0], epoch[-1]
    buffer_s = params.buffer_window_min * 60.0
    horizon_s = horizon_min * 60.0
    stride_s = stride_min * 60.0

    # Stop a full horizon before the archive end: the last `horizon` minutes have a
    # truncated (T, T+H] truth window, which would right-censor late arrivals and
    # bias absolute POD/FAR (review LOW finding).
    grid = np.arange(t0 + buffer_s, t1 - horizon_s, stride_s)
    rows: list[dict] = []

    for name, tlat, tlon in targets:
        region = region_of(tlat, tlon)
        # Per-strike distance to this target — computed once.
        dist = np.array([haversine_km(tlat, tlon, s.lat, s.lon) for s in strikes])
        # Per-ring sorted strike times within the ring (epoch is already time-sorted,
        # so the boolean slice stays sorted) — fast searchsorted for arrivals + the
        # persistence baseline, instead of a full-array mask per (T, ring).
        ring_epochs = {r: epoch[dist <= r] for r in rings_sorted}
        prior_c: float | None = None
        prior_e: float | None = None
        prior_t: float | None = None

        for T in grid:
            # Buffer-window strikes within the relevance radius (compute_eta input).
            in_win = (epoch > T - buffer_s) & (epoch <= T) & (dist <= params.relevance_radius_km)
            win_idx = np.nonzero(in_win)[0]
            # Any future in-(largest)ring arrival? (so surprise events still score).
            re_big = ring_epochs[largest_ring]
            has_arrival = (np.searchsorted(re_big, T + horizon_s, "right")
                           > np.searchsorted(re_big, T, "right"))
            if win_idx.size == 0 and not has_arrival:
                continue

            T_dt = datetime.fromtimestamp(float(T), tz=timezone.utc)
            lagr_eta = {r: None for r in rings_sorted}
            ring_prob = {r: 0.0 for r in rings_sorted}
            if win_idx.size:
                if win_idx.size > max_buffer:
                    win_idx = win_idx[-max_buffer:]
                wstrikes = [strikes[i] for i in win_idx]
                res = compute_eta(wstrikes, tlat, tlon, rings_sorted, params,
                                  now=T_dt, multicell=multicell)
                if prob_leads is not None:
                    for rp in strike_probability(
                            wstrikes, tlat, tlon,
                            [(r, prob_leads[r]) for r in rings_sorted],
                            params, prob_params, now=T_dt):
                        ring_prob[rp.ring_km] = rp.prob
                # Lagrangian-persistence baseline: bulk velocity of the relevant
                # strikes, advect the most-recent ones, earliest ring entry.
                vel = estimate_velocity(wstrikes, tlat, tlon)
                if vel is not None and vel.span_min >= params.min_fit_span_min:
                    latest = max(s.t for s in wstrikes)
                    recent = [s for s in wstrikes
                              if (latest - s.t).total_seconds() / 60.0
                              <= params.leading_edge_recent_min] or wstrikes
                    offs = [_enu_km(s.lat, s.lon, tlat, tlon) for s in recent]
                    for ring in rings_sorted:
                        lagr_eta[ring] = _lagrangian_eta(
                            offs, vel.east_kmmin, vel.north_kmmin, ring, horizon_min)
            else:
                res = compute_eta([], tlat, tlon, rings_sorted, params, now=T_dt)

            if smoothing:
                gap = None if prior_t is None else (float(T) - prior_t) / 60.0
                if gap is not None and gap > max_gap_min:
                    prior_c = prior_e = None
                alpha = 1.0 if gap is None else 1.0 - math.exp(-gap / tau_min)
                res, prior_c, prior_e = smooth_eta(
                    res, prior_c, prior_e, alpha, params.min_closing_kmh
                )
                prior_t = float(T)

            for ring in rings_sorted:
                ring_pred = next(r for r in res.rings if r.ring_km == ring)
                re = ring_epochs[ring]
                pos = np.searchsorted(re, T, "right")  # # strikes within ring at ≤ T
                hi = np.searchsorted(re, T + horizon_s, "right")
                arrived = hi > pos
                actual_eta = (re[pos] - T) / 60.0 if arrived else None
                # Eulerian persistence baseline: minutes since the last in-ring strike
                # (None if never). The metrics treat "in-ring within the last L min" as
                # the persistence warning for lead L.
                last_in_ring_age = (T - re[pos - 1]) / 60.0 if pos > 0 else None
                rows.append({
                    "t_utc": T_dt.isoformat(),
                    "target": name,
                    "region": region,
                    "ring_km": float(ring),
                    "state": res.state,
                    "predicted_eta_min": ring_pred.eta_min,
                    "inside_at_t": bool(ring_pred.inside),
                    "leading_edge_km": res.leading_edge_km,
                    "closing_kmh": res.closing_kmh,
                    "cell_speed_kmh": res.cell_speed_kmh,
                    "confidence": float(res.confidence),
                    "n_strikes": int(res.n_strikes),
                    "n_cells": int(res.n_cells),
                    "arrived": bool(arrived),
                    "actual_eta_min": actual_eta,
                    "last_in_ring_age_min": last_in_ring_age,
                    "lagr_eta_min": lagr_eta[ring],
                    "prob": ring_prob[ring],
                    "horizon_min": float(horizon_min),
                })
    return rows


# --------------------------------------------------------------------------- #
# Metrics (stratified by region + ring)
# --------------------------------------------------------------------------- #
# Minimum in-window events for a (region, ring) slice's POD/FAR to be reported.
MIN_EVENTS = 8


def _event(r: dict, lead: float) -> bool:
    """A strike actually entered the ring within ``lead`` min of T."""
    return (r["arrived"] and r["actual_eta_min"] is not None
            and r["actual_eta_min"] <= lead)


def _model_warned(r: dict, lead: float) -> bool:
    """Our forecast: approaching with predicted ETA within ``lead``."""
    return (r["state"] == STATE_APPROACHING
            and r["predicted_eta_min"] is not None
            and r["predicted_eta_min"] <= lead)


def _persist_warned(r: dict, lead: float) -> bool:
    """Eulerian-persistence baseline: a strike was in-ring within the last ``lead``
    min, so persistence predicts one in the next ``lead`` min."""
    age = r.get("last_in_ring_age_min")
    return age is not None and age <= lead


def _lagr_warned(r: dict, lead: float) -> bool:
    """Lagrangian-persistence baseline: bulk-advected recent strikes reach the ring
    within ``lead`` min."""
    eta = r.get("lagr_eta_min")
    return eta is not None and eta <= lead


def _clear_at_t(r: dict, lead: float) -> bool:
    """Target had no in-ring strike in the last ``lead`` min (so it's a genuine
    arrival situation, not ongoing activity — and Eulerian persistence is silent)."""
    age = r.get("last_in_ring_age_min")
    return age is None or age > lead


def _confusion(rows: list[dict], warned_fn, lead: float) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for r in rows:
        w, e = warned_fn(r, lead), _event(r, lead)
        if w and e:
            tp += 1
        elif w and not e:
            fp += 1
        elif e:
            fn += 1
    return tp, fp, fn


def _skill(tp: int, fp: int, fn: int, enough: bool) -> dict:
    if not enough:
        return {"pod": None, "far": None, "csi": None}
    return {
        "pod": tp / (tp + fn) if (tp + fn) else None,
        "far": fp / (tp + fp) if (tp + fp) else None,
        "csi": tp / (tp + fp + fn) if (tp + fp + fn) else None,
    }


# --- probabilistic scoring (Phase 2b) --------------------------------------- #
def _roc_auc(scores: np.ndarray, y: np.ndarray) -> float | None:
    """ROC-AUC via the rank-sum identity (ties-averaged via scipy rankdata)."""
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata(scores)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _tie_grouped(scores: np.ndarray, y: np.ndarray):
    """Sort by descending score and return cumulative (tp, fp, score) at the END of
    each tie block — i.e. only at *realizable* ``score >= thr`` cut points. The
    ensemble probability is quantised to n_members+1 values, so per-item prefixes
    land inside tie blocks at thresholds no real cut can produce, which would
    optimistically bias F1 and make AP order-dependent (review finding M7)."""
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]  # last index per distinct score
    return tp[last], fp[last], s[last]


def _pr_auc(scores: np.ndarray, y: np.ndarray) -> float | None:
    """Average precision (PR-AUC), evaluated at distinct thresholds (ties grouped,
    matching sklearn.metrics.average_precision_score)."""
    if y.sum() == 0:
        return None
    tp, fp, _ = _tie_grouped(scores, y)
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rec_prev) * precision))


def _best_f1(scores: np.ndarray, y: np.ndarray) -> dict | None:
    """Max F1 over *realizable* thresholds (ties grouped) + its precision/recall/
    threshold — so it matches a real ``p >= threshold`` cut (and the confusion
    matrix in lightning_report)."""
    P = float(y.sum())
    if P == 0:
        return None
    tp, fp, s = _tie_grouped(scores, y)
    prec = tp / (tp + fp)
    rec = tp / P
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    i = int(np.argmax(f1))
    return {"f1": float(f1[i]), "precision": float(prec[i]),
            "recall": float(rec[i]), "threshold": float(s[i])}


def _prob_metrics(rows: list[dict], lead_min: float, clear_only: bool = False) -> dict:
    """Probabilistic skill for one (region, ring) slice: Brier, Brier skill score
    (vs climatology), ROC-AUC, PR-AUC, best-F1, and a persistence-recency AUC
    baseline. Scored against the binary 'arrival within lead' outcome."""
    rows = [r for r in rows if not r["inside_at_t"] and r.get("prob") is not None]
    if clear_only:
        rows = [r for r in rows if _clear_at_t(r, lead_min)]
    n = len(rows)
    if n == 0:
        return {"n": 0, "events": 0}
    p = np.array([r["prob"] for r in rows])
    y = np.array([1.0 if _event(r, lead_min) else 0.0 for r in rows])
    events = int(y.sum())
    base = events / n
    enough = events >= MIN_EVENTS
    brier = float(np.mean((p - y) ** 2))
    brier_clim = base * (1.0 - base)
    f1 = _best_f1(p, y) if enough else None
    age = np.array([r["last_in_ring_age_min"] if r["last_in_ring_age_min"] is not None
                    else 1e6 for r in rows])
    return {
        "n": n, "events": events, "base_rate": base,
        "brier": brier,
        "bss": (1.0 - brier / brier_clim) if (enough and brier_clim > 0) else None,
        "auc": _roc_auc(p, y) if enough else None,
        "pr_auc": _pr_auc(p, y) if enough else None,
        "pers_auc": _roc_auc(1.0 / (1.0 + age), y) if enough else None,
        "f1": f1["f1"] if f1 else None,
        "f1_prec": f1["precision"] if f1 else None,
        "f1_rec": f1["recall"] if f1 else None,
        "f1_thr": f1["threshold"] if f1 else None,
    }


def _calibration_holdout(rows: list[dict], lead_min: float, frac: float = 0.66) -> dict:
    """Isotonic calibration with a TEMPORAL holdout (fit on the earlier ``frac``,
    evaluate on the later rows — no leakage). Reports test-set Brier + BSS raw vs
    calibrated. Calibration is monotonic, so ROC/PR are unchanged; only the
    reliability/Brier improve."""
    rows = sorted((r for r in rows if not r["inside_at_t"] and r.get("prob") is not None),
                  key=lambda r: r["t_utc"])
    n = len(rows)
    if n < 30:
        return {"n": n, "events": 0}
    split = int(n * frac)
    tr, te = rows[:split], rows[split:]

    def xy(rs):
        return (np.array([r["prob"] for r in rs]),
                np.array([1.0 if _event(r, lead_min) else 0.0 for r in rs]))

    ptr, ytr = xy(tr)
    pte, yte = xy(te)
    if int(ytr.sum()) < MIN_EVENTS or int(yte.sum()) < MIN_EVENTS:
        return {"n": len(te), "events": int(yte.sum()), "thin": True}
    cal = fit_isotonic(ptr, ytr)
    pcal = cal.predict(pte)
    base = float(yte.mean())
    bc = base * (1.0 - base)
    braw = float(np.mean((pte - yte) ** 2))
    bcal = float(np.mean((pcal - yte) ** 2))
    return {
        "n": len(te), "events": int(yte.sum()), "base_rate": base,
        "brier_raw": braw, "brier_cal": bcal,
        "bss_raw": (1.0 - braw / bc) if bc > 0 else None,
        "bss_cal": (1.0 - bcal / bc) if bc > 0 else None,
    }


def _metrics_for(rows: list[dict], lead_min: float, clear_only: bool = False) -> dict:
    """Confusion + ETA/lead metrics for one (region, ring) slice, for the model and
    both baselines (Eulerian + Lagrangian persistence; CLAUDE.md contract: a nowcast
    must beat persistence). Operationally matched: a *warning* = approaching with
    predicted ETA ≤ ``lead``; an *event* = a strike entered the ring within the same
    ``lead``. Rows already inside the ring at T are excluded (nothing to forecast).

    ``clear_only`` restricts to rows where the target was clear at T (no in-ring
    strike in the last ``lead`` min) — the genuine *arrival* situations, where
    Eulerian persistence is silent and the honest baseline is Lagrangian.

    Caveat: rows are T-samples (5-min stride), so one storm passage contributes
    several event-rows — metrics are sample-weighted, fine for A/B comparison but
    not an absolute episode rate."""
    rows = [r for r in rows if not r["inside_at_t"]]
    if clear_only:
        rows = [r for r in rows if _clear_at_t(r, lead_min)]
    n = len(rows)
    if n == 0:
        return {"n": 0, "events": 0}

    tp, fp, fn = _confusion(rows, _model_warned, lead_min)
    events = tp + fn
    enough = events >= MIN_EVENTS
    model = _skill(tp, fp, fn, enough)
    persist = _skill(*_confusion(rows, _persist_warned, lead_min), enough)
    lagr = _skill(*_confusion(rows, _lagr_warned, lead_min), enough)

    # ETA error + lead time on model hits.
    errs, leads = [], []
    for r in rows:
        if _model_warned(r, lead_min) and _event(r, lead_min):
            errs.append(r["predicted_eta_min"] - r["actual_eta_min"])
            leads.append(r["actual_eta_min"])

    # Brier of confidence -> P(arrival within lead), over approaching rows.
    appr = [r for r in rows if r["state"] == STATE_APPROACHING]
    brier = None
    if len(appr) >= 10:
        conf = np.array([r["confidence"] for r in appr])
        out = np.array([1.0 if _event(r, lead_min) else 0.0 for r in appr])
        brier = float(brier_score(conf, out))

    return {
        "n": n,
        "events": events,
        "warnings": tp + fp,
        "base_rate": events / n,
        "pod": model["pod"],
        "far": model["far"],
        "csi": model["csi"],
        "p_pod": persist["pod"],
        "p_far": persist["far"],
        "p_csi": persist["csi"],
        "lagr_pod": lagr["pod"],
        "lagr_far": lagr["far"],
        "lagr_csi": lagr["csi"],
        "eta_mae": float(np.mean(np.abs(errs))) if errs else None,
        "eta_bias": float(np.mean(errs)) if errs else None,
        "lead_median": float(np.median(leads)) if leads else None,
        "n_hits": len(errs),
        "brier_conf": brier,
    }


def _fmt(x, w, p=2):
    return f"{x:>{w}.{p}f}" if isinstance(x, float) else f"{'—':>{w}}"


def _print_table(rows, leads, rings_km, present, *, clear_only, base_key, base_label):
    """One per-(region,ring) table comparing the model CSI against a baseline CSI."""
    hdr = (f"{'region':<9}{'ring':>5}{'n':>7}{'evt':>5}{'base%':>7}"
           f"{'POD':>6}{'FAR':>6}{'CSI':>6}{base_label:>9}{'beat':>5}{'lead50':>8}")
    print(hdr)
    print("-" * len(hdr))
    for reg in present:
        for ring in sorted(rings_km):
            sl = [r for r in rows if r["region"] == reg and r["ring_km"] == ring]
            m = _metrics_for(sl, leads[ring], clear_only=clear_only)
            if m["n"] == 0:
                continue
            base = m.get(base_key)
            beat = ("✓" if m["csi"] > base else "✗") if (m["csi"] is not None and base is not None) else ""
            note = "  ⚠ thin" if m["events"] < MIN_EVENTS else ""
            print(f"{reg:<9}{ring:>5.0f}{m['n']:>7}{m['events']:>5}{100*m['base_rate']:>6.1f}%"
                  f"{_fmt(m['pod'],6)}{_fmt(m['far'],6)}{_fmt(m['csi'],6)}{_fmt(base,9)}"
                  f"{beat:>5}{_fmt(m['lead_median'],8,1)}{note}")
    print("-" * len(hdr))


def summarize(rows: list[dict], leads: dict[float, float], rings_km: list[float]) -> None:
    regions = ["Denmark", "Alps", "Other"]
    present = [reg for reg in regions if any(r["region"] == reg for r in rows)]
    leadstr = ", ".join(f"{r:.0f}km→{leads[r]:.0f}min" for r in sorted(rings_km))
    print(f"\n{'='*80}\nLightning ETA backtest — {len(rows)} rows "
          f"(operational leads: {leadstr})\n{'='*80}")
    print("\n[A] All storm-present situations — model vs Eulerian persistence "
          "(continuation-inclusive)")
    _print_table(rows, leads, rings_km, present, clear_only=False,
                 base_key="p_csi", base_label="persCSI")
    print("\n[B] Arrival skill — target CLEAR at T — model vs Lagrangian persistence "
          "(the early-warning metric)")
    _print_table(rows, leads, rings_km, present, clear_only=True,
                 base_key="lagr_csi", base_label="lagrCSI")
    print(f"\nCSI/POD/FAR matched at the per-ring lead; suppressed when events < {MIN_EVENTS}.\n"
          "beat ✓ = model CSI > baseline CSI. [A] persistence = 'in-ring within last L min';\n"
          "[B] is the honest early-warning test (Eulerian persistence is silent on clear→arrival,\n"
          "so the baseline is Lagrangian = bulk-advect recent strikes). lead50 = median real\n"
          "warning min on hits; base% = event base rate.")

    if any(r.get("prob") for r in rows):
        print("\n[C] PROBABILISTIC forecast — P(strike within r within lead), all "
              "storm-present situations")
        phdr = (f"{'region':<9}{'ring':>5}{'n':>7}{'evt':>5}{'base%':>7}"
                f"{'Brier':>7}{'BSS':>7}{'ROC':>6}{'PR':>6}{'persROC':>8}"
                f"{'bestF1':>7}{'prec':>6}{'rec':>6}")
        print(phdr)
        print("-" * len(phdr))
        for reg in present:
            for ring in sorted(rings_km):
                sl = [r for r in rows if r["region"] == reg and r["ring_km"] == ring]
                m = _prob_metrics(sl, leads[ring])
                if m["n"] == 0:
                    continue
                note = "  ⚠ thin" if m["events"] < MIN_EVENTS else ""
                print(f"{reg:<9}{ring:>5.0f}{m['n']:>7}{m['events']:>5}{100*m['base_rate']:>6.1f}%"
                      f"{_fmt(m['brier'],7,3)}{_fmt(m['bss'],7,3)}{_fmt(m['auc'],6)}"
                      f"{_fmt(m['pr_auc'],6)}{_fmt(m['pers_auc'],8)}{_fmt(m['f1'],7)}"
                      f"{_fmt(m['f1_prec'],6)}{_fmt(m['f1_rec'],6)}{note}")
        print("-" * len(phdr))
        print("ROC/PR = AUC of the probability vs arrival; persROC = persistence-recency AUC\n"
              "baseline (beat it = the physics adds ranking skill). BSS vs climatology;\n"
              "bestF1 = max F1 over thresholds with its precision/recall.\n"
              "⚠ rows are pseudo-replicates (5-min stride; in --target-grid mode also\n"
              "  overlapping targets sharing strikes) — effective sample size ≈ the number\n"
              "  of independent storm episodes, far below n. Treat AUC/Brier/BSS as point\n"
              "  estimates; do NOT read n as i.i.d. confidence.")

        print("\n[D] Calibration — isotonic on a temporal holdout (fit earlier 2/3, "
              "test later 1/3)")
        dhdr = (f"{'region':<9}{'ring':>5}{'test_n':>8}{'evt':>5}{'base%':>7}"
                f"{'Brier_raw':>10}{'Brier_cal':>10}{'BSS_raw':>9}{'BSS_cal':>9}")
        print(dhdr)
        print("-" * len(dhdr))
        for reg in present:
            for ring in sorted(rings_km):
                sl = [r for r in rows if r["region"] == reg and r["ring_km"] == ring]
                m = _calibration_holdout(sl, leads[ring])
                if m["n"] == 0 or m.get("thin"):
                    continue
                print(f"{reg:<9}{ring:>5.0f}{m['n']:>8}{m['events']:>5}{100*m['base_rate']:>6.1f}%"
                      f"{_fmt(m['brier_raw'],10,3)}{_fmt(m['brier_cal'],10,3)}"
                      f"{_fmt(m['bss_raw'],9,3)}{_fmt(m['bss_cal'],9,3)}")
        print("-" * len(dhdr))
        print("Isotonic is monotonic → ROC/PR unchanged; this shows the Brier/BSS lift from\n"
              "calibration on unseen later data (BSS_cal > 0 ⇒ beats climatology when calibrated).")


# --------------------------------------------------------------------------- #
# Tuning grid
# --------------------------------------------------------------------------- #
def grid_search(strikes, targets, base: EtaParams, rings_km, leads, *,
                stride_min, horizon_min, max_buffer) -> None:
    """Small per-region sweep over the most impactful params (future-plan §4).

    Reports only regions/rings with enough events; in practice that's the Alps —
    Denmark is too event-poor in the current corpus to tune on."""
    grid = {
        "cluster_eps_km": [10.0, 15.0, 20.0],
        "relevance_radius_km": [40.0, 60.0, 80.0],
        "leading_edge_recent_min": [5.0, 10.0],
        "min_closing_kmh": [5.0, 10.0, 15.0],
    }
    print("Tuning sweep — one param varied at a time around the deployed defaults.")
    for field, values in grid.items():
        print(f"\n### {field} (default {getattr(base, field)})")
        for v in values:
            params = replace(base, **{field: v})
            rows = run_backtest(strikes, targets, params, rings_km,
                                stride_min=stride_min, horizon_min=horizon_min,
                                max_buffer=max_buffer)
            line = f"  {field}={v:<6}"
            for reg in ("Denmark", "Alps"):
                for ring in sorted(rings_km):
                    sl = [r for r in rows if r["region"] == reg and r["ring_km"] == ring]
                    m = _metrics_for(sl, leads[ring])
                    pod = f"{m.get('pod'):.2f}" if m.get("pod") is not None else "—"
                    far = f"{m.get('far'):.2f}" if m.get("far") is not None else "—"
                    lead = f"{m.get('lead_median'):.0f}" if m.get("lead_median") is not None else "—"
                    line += f" | {reg[:3]}{ring:.0f} POD={pod} FAR={far} lead={lead}"
            print(line)


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-dir", default=str(ROOT / "data" / "strikes"))
    p.add_argument("--rings", default="3,10", help="Comma-separated ring radii km")
    p.add_argument("--leads", default="3:15,10:30",
                   help="Per-ring operational lead, ring:lead_min comma-separated")
    p.add_argument("--stride-min", type=float, default=5.0)
    p.add_argument("--horizon-min", type=float, default=60.0)
    p.add_argument("--max-buffer", type=int, default=2000)
    p.add_argument("--extra-target", action="append", default=[],
                   help="lat,lon,name — repeatable")
    p.add_argument("--target-grid", action="store_true",
                   help="Evaluate over a grid of targets around each anchor (vastly more events)")
    p.add_argument("--grid-km", type=float, default=10.0,
                   help="Grid spacing km (≈ decorrelation scale; finer = more autocorrelated)")
    p.add_argument("--grid-radius-km", type=float, default=40.0,
                   help="Grid disc radius km; must be ≤ collection_radius − relevance_radius")
    p.add_argument("--multicell", action="store_true",
                   help="Phase 1: evaluate all cells (soonest ETA across cells) not just one")
    p.add_argument("--probabilistic", action="store_true",
                   help="Phase 2: also score the probabilistic forecast (Brier/ROC/PR/F1)")
    p.add_argument("--smoothing", action="store_true", help="Replay the EMA-smoothed pipeline")
    p.add_argument("--tau-min", type=float, default=3.0)
    p.add_argument("--max-gap-min", type=float, default=10.0)
    p.add_argument("--grid", action="store_true", help="Run the tuning sweep instead of a single eval")
    p.add_argument("--output", default=None, help="Parquet output path (single-run mode)")
    p.add_argument("--calibration-output", default=None,
                   help="Fit + save isotonic confidence→P(arrival) here")
    args = p.parse_args()

    rings_km = [float(x) for x in args.rings.split(",")]
    leads = {float(k): float(v) for k, v in (kv.split(":") for kv in args.leads.split(","))}
    for r in rings_km:
        leads.setdefault(r, args.horizon_min)
    if args.target_grid:
        targets = make_grid_targets(DEFAULT_TARGETS, args.grid_km, args.grid_radius_km)
    else:
        targets = list(DEFAULT_TARGETS)
    for spec in args.extra_target:
        lat, lon, name = spec.split(",")
        targets.append((name, float(lat), float(lon)))

    archive = Path(args.archive_dir)
    strikes = load_strikes(archive)
    if not strikes:
        print(f"No strikes found in {archive}", file=sys.stderr)
        return 1
    span_h = (strikes[-1].t - strikes[0].t).total_seconds() / 3600.0
    from collections import Counter
    by_region = Counter(region_of(la, lo) for _, la, lo in targets)
    tdesc = (", ".join(f"{n}[{region_of(la, lo)}]" for n, la, lo in targets)
             if len(targets) <= 6 else
             f"{len(targets)} grid points ({dict(by_region)})")
    print(f"Loaded {len(strikes)} strikes over {span_h:.1f} h "
          f"({strikes[0].t.date()} → {strikes[-1].t.date()}); targets: {tdesc}")

    base = EtaParams()

    if args.grid:
        grid_search(strikes, targets, base, rings_km, leads,
                    stride_min=args.stride_min, horizon_min=args.horizon_min,
                    max_buffer=args.max_buffer)
        return 0

    rows = run_backtest(
        strikes, targets, base, rings_km,
        stride_min=args.stride_min, horizon_min=args.horizon_min,
        max_buffer=args.max_buffer, smoothing=args.smoothing,
        tau_min=args.tau_min, max_gap_min=args.max_gap_min,
        multicell=args.multicell,
        prob_leads=leads if args.probabilistic else None,
    )
    print(f"(mode: {'MULTI-CELL' if args.multicell else 'single-cell'}"
          f"{', PROBABILISTIC' if args.probabilistic else ''})")
    summarize(rows, leads, rings_km)

    if args.output:
        import pyarrow as pa
        import pyarrow.parquet as pq
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), out)
        print(f"\nWrote {out} ({len(rows)} rows)")

    if args.calibration_output:
        appr = [r for r in rows if r["state"] == STATE_APPROACHING]
        if len(appr) >= 10:
            conf = np.array([r["confidence"] for r in appr])
            out = np.array([1.0 if r["arrived"] else 0.0 for r in appr])
            cal = fit_isotonic(conf, out)
            cal.save(Path(args.calibration_output))
            print(f"Saved confidence→P(arrival) calibrator → {args.calibration_output} "
                  f"(N={len(appr)}, base rate {100*out.mean():.1f}%)")
        else:
            print(f"Only {len(appr)} approaching rows — too few to calibrate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
