"""Fit isotonic calibration curves from the corpus Parquet.

Reads ``reports/calibration_corpus.parquet`` (built by
``build_calibration_corpus.py``), fits one ``IsotonicCalibrator`` per
lead time, writes the curves to a JSON file the integration loads at
startup, and prints reliability statistics so the user can sanity-check
the fit.

Usage::

    python scripts/fit_calibration.py \\
        --corpus reports/calibration_corpus.parquet \\
        --output src/dmi_nowcast_core/calibration_curves.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmi_nowcast_core.calibrate import (  # noqa: E402
    brier_score,
    fit_isotonic,
    reliability_curve,
)


def _load_corpus(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read the Parquet; return ``{lead_min: (raw_probs, outcomes)}`` with
    NaN raw and missing outcomes filtered out."""
    import pyarrow.parquet as pq

    tbl = pq.read_table(path)
    leads = tbl.column("lead_min").to_pylist()
    raws = tbl.column("raw_prob").to_pylist()
    outs = tbl.column("outcome").to_pylist()

    by_lead: dict[int, list[tuple[float, int]]] = {}
    for lead, raw, out in zip(leads, raws, outs):
        if raw is None or raw != raw:  # NaN
            continue
        if out not in (0, 1):
            continue
        by_lead.setdefault(int(lead), []).append((float(raw), int(out)))

    result = {}
    for lead, pairs in by_lead.items():
        rp = np.array([p[0] for p in pairs], dtype=np.float64)
        oc = np.array([p[1] for p in pairs], dtype=np.float64)
        result[lead] = (rp, oc)
    return result


def _print_summary(lead: int, raw: np.ndarray, obs: np.ndarray, calibrator) -> None:
    """Tabulate raw vs calibrated reliability for the user."""
    cal = calibrator.predict(raw)
    raw_brier = brier_score(raw.astype(np.float32), obs.astype(np.int8))
    cal_brier = brier_score(cal.astype(np.float32), obs.astype(np.int8))
    base_rate = obs.mean()

    print(f"\n=== Lead +{lead} min  ({len(raw)} samples)  base rate {base_rate*100:.1f}% ===")
    print(f"  Brier score: raw={raw_brier:.4f}  →  calibrated={cal_brier:.4f}  "
          f"({'improved' if cal_brier < raw_brier else 'WORSE'})")

    # Reliability curve in 10 quantile bins.
    print(f"\n  {'bin (raw)':<14} {'n':>6}  {'mean raw':>9}  {'observed':>9}  {'calibrated':>11}")
    bins = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.85, 0.95, 1.0001])
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (raw >= lo) & (raw < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mr = float(raw[mask].mean())
        mo = float(obs[mask].mean())
        mc = float(calibrator.predict(np.array([mr])).item())
        print(f"  {lo*100:3.0f}-{hi*100:3.0f}%      {n:>6}  {mr*100:>7.1f}%  "
              f"{mo*100:>7.1f}%  {mc*100:>9.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("reports/calibration_corpus.parquet"))
    ap.add_argument("--output", type=Path, default=Path("src/dmi_nowcast_core/calibration_curves.json"))
    ap.add_argument("--min-samples", type=int, default=50,
                    help="Skip leads with fewer than this many valid samples")
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"Corpus not found: {args.corpus}", file=sys.stderr)
        return 1

    by_lead = _load_corpus(args.corpus)
    print(f"Loaded {sum(len(v[0]) for v in by_lead.values())} samples across "
          f"{len(by_lead)} leads from {args.corpus}")

    curves = {}
    metadata = {
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(args.corpus),
        "n_samples_total": int(sum(len(v[0]) for v in by_lead.values())),
        "leads": {},
    }

    for lead in sorted(by_lead.keys()):
        raw, obs = by_lead[lead]
        if len(raw) < args.min_samples:
            print(f"\nLead +{lead} min: only {len(raw)} samples — skipping")
            continue
        calibrator = fit_isotonic(raw, obs)
        _print_summary(lead, raw, obs, calibrator)
        curves[str(lead)] = {
            "raw_breakpoints": list(calibrator.raw_breakpoints),
            "calibrated_values": list(calibrator.calibrated_values),
        }
        metadata["leads"][str(lead)] = {
            "n_samples": int(len(raw)),
            "base_rate": float(obs.mean()),
            "raw_brier": float(brier_score(raw.astype(np.float32), obs.astype(np.int8))),
            "calibrated_brier": float(brier_score(
                calibrator.predict(raw).astype(np.float32), obs.astype(np.int8),
            )),
        }

    if not curves:
        print("\nNo curves fit — corpus has too little signal.", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "metadata": metadata,
        "curves": curves,
    }, indent=2))
    print(f"\nWrote {len(curves)} curves to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
