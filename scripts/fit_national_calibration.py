"""Fit weighted national isotonic calibration curves (website Phase B, B3).

Consumes the **v2 multi-point corpus Parquet** written by
``scripts/build_calibration_corpus.py`` (one row per event × point × lead,
with ``sample_weight`` = unnormalised inverse-inclusion-probability from
the wet-biased event sampler — Finding 3) and fits one **weight-aware**
isotonic calibrator per lead on the pooled national sample via
:func:`dmi_nowcast_core.calibrate.fit_isotonic_weighted`.

This supersedes ``scripts/fit_calibration.py`` for the national path (the
legacy fitter remains for the legacy home-only binary curves). Key
differences: sample weights correct the wet-bias base-rate shift, the fit
refuses mixed-settings and non-v2 corpora outright, and thin leads are
refused rather than fitted on noise.

Dependency contract: **pyarrow + numpy only** (plus the repo's own
``dmi_nowcast_core``). No duckdb / pandas — this script must run inside
the sidecar container in Phase B5 without new runtime dependencies. The
exploratory DuckDB analysis lives in ``sql/`` +
``scripts/national_calibration_report.py`` instead.

Output format is identical to the legacy curves file, so the sidecar's
``load_calibration_curves`` reads it verbatim::

    {
      "metadata": {
        "fitted_at": "<UTC ISO>",
        "settings_hash": "...", "settings": {...},
        "n_events": ..., "n_points": ...,
        "n_samples": ...,                # total fitted rows (B4 manifest echo)
        "brier_before": ..., "brier_after": ...,   # weighted, pooled (B4)
        "leads": {"<lead>": {"n_samples", "effective_n", "base_rate",
                              "brier_before", "brier_after"}, ...},
        ...
      },
      "curves": {
        "<lead>": {"raw_breakpoints": [...], "calibrated_values": [...]},
        ...
      }
    }

``effective_n`` is the Kish effective sample size ``(Σw)² / Σw²`` — the
honest sample count once the design weights are accounted for (plan §2:
"reports state effective sample sizes"). ``brier_*`` are weighted Brier
scores ``Σw(p−y)²/Σw``.

Usage::

    python scripts/fit_national_calibration.py \\
        --corpus reports/calibration_corpus.parquet \\
        --output data/national_curves.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmi_nowcast_core.calibrate import (  # noqa: E402
    IsotonicCalibrator,
    brier_score_weighted,
    fit_isotonic_weighted,
)

EXPECTED_SCHEMA_VERSION = 2

#: Columns the fitter needs beyond the settings echo.
_CORE_COLUMNS = (
    "event_time",
    "point_id",
    "lead_min",
    "raw_prob",
    "outcome",
    "sample_weight",
    "settings_hash",
    "schema_version",
)

#: Settings columns echoed into the output metadata (identical on every
#: row of a valid corpus, enforced via the settings hash at build time).
_SETTINGS_COLUMNS = (
    "ensemble_size",
    "n_cascade_levels",
    "downsample_factor",
    "threshold_mm_h",
    "disc_radius_m",
    "detection_stat",
    "scan_type",
    # R5: identifies the motion field STEPS was driven with. A corpus built
    # before motion-field completion has no such column at all, so the
    # missing-column check rejects it before the hash comparison runs.
    "motion_method",
    "timestep_min",
    "n_timesteps",
    "leads_min_csv",
)


class CorpusError(ValueError):
    """The corpus cannot be fitted as-is (wrong schema, mixed settings, …)."""


@dataclass
class Corpus:
    """The fit-relevant slice of a validated v2 corpus."""

    lead_min: np.ndarray  # int
    raw_prob: np.ndarray  # float (NaN = failed forecast / out of grid)
    outcome: np.ndarray  # int, -1 = missing verification (Parquet null)
    sample_weight: np.ndarray  # float
    settings_hash: str
    settings: dict
    n_events: int
    n_points: int


def kish_effective_n(weights: np.ndarray) -> float:
    """Kish effective sample size ``(Σw)² / Σw²`` of a weighted sample."""
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return 0.0
    sw = float(np.sum(w))
    sw2 = float(np.sum(w * w))
    return (sw * sw) / sw2 if sw2 > 0 else 0.0


def load_v2_corpus(path: Path) -> Corpus:
    """Read + validate the corpus Parquet. Raises :class:`CorpusError` on a
    legacy / mixed / weight-less corpus — never fits through those."""
    import pyarrow.parquet as pq

    if not path.exists():
        raise CorpusError(f"corpus not found: {path}")

    schema = pq.read_schema(path)
    if "schema_version" not in schema.names:
        raise CorpusError(
            f"{path} has no schema_version column — this is a legacy (v1) "
            "single-point corpus. Rebuild it with the multi-point "
            "build_calibration_corpus.py; the national fit refuses v1 corpora."
        )
    if "sample_weight" not in schema.names:
        raise CorpusError(
            f"{path} has no sample_weight column — cannot run the weighted "
            "fit (Finding 3: unweighted fits on a wet-biased sample are "
            "themselves miscalibrated). Rebuild the corpus."
        )
    missing = [c for c in (*_CORE_COLUMNS, *_SETTINGS_COLUMNS) if c not in schema.names]
    if missing:
        raise CorpusError(f"{path} is missing corpus columns: {missing}")

    tbl = pq.read_table(path, columns=list(_CORE_COLUMNS) + list(_SETTINGS_COLUMNS))
    if tbl.num_rows == 0:
        raise CorpusError(f"{path} is empty")

    versions = sorted(set(tbl.column("schema_version").to_pylist()))
    if versions != [EXPECTED_SCHEMA_VERSION]:
        raise CorpusError(
            f"{path} has schema_version values {versions}; this fitter "
            f"requires schema_version == {EXPECTED_SCHEMA_VERSION} exactly."
        )
    hashes = sorted(set(tbl.column("settings_hash").to_pylist()))
    if len(hashes) != 1:
        raise CorpusError(
            f"{path} mixes settings hashes {hashes} — a fit across different "
            "STEPS/sampling settings calibrates nothing. Rebuild a clean corpus."
        )

    settings = {c: tbl.column(c)[0].as_py() for c in _SETTINGS_COLUMNS}
    return Corpus(
        lead_min=tbl.column("lead_min").to_numpy(zero_copy_only=False).astype(np.int64),
        raw_prob=tbl.column("raw_prob").to_numpy(zero_copy_only=False).astype(np.float64),
        outcome=(
            tbl.column("outcome")
            .fill_null(-1)
            .to_numpy(zero_copy_only=False)
            .astype(np.int64)
        ),
        sample_weight=tbl.column("sample_weight")
        .to_numpy(zero_copy_only=False)
        .astype(np.float64),
        settings_hash=hashes[0],
        settings=settings,
        n_events=len(set(tbl.column("event_time").to_pylist())),
        n_points=len(set(tbl.column("point_id").to_pylist())),
    )


def valid_mask(corpus: Corpus) -> np.ndarray:
    """Rows usable for fitting: finite raw_prob, 0/1 outcome, positive
    finite weight. (NaN raw = failed forecast; -1 outcome = Parquet null =
    missing verification frame; NaN weight = degenerate stratum.)"""
    return (
        np.isfinite(corpus.raw_prob)
        & ((corpus.outcome == 0) | (corpus.outcome == 1))
        & np.isfinite(corpus.sample_weight)
        & (corpus.sample_weight > 0)
    )


@dataclass
class LeadFit:
    lead: int
    calibrator: IsotonicCalibrator
    n_samples: int
    effective_n: float
    base_rate: float  # weighted
    brier_before: float  # weighted
    brier_after: float  # weighted
    # Pooled-aggregate accumulators (Σw·(p−y)², Σw·(cal−y)², Σw).
    sum_w_sq_before: float
    sum_w_sq_after: float
    sum_w: float


def fit_lead(
    lead: int, raw: np.ndarray, out: np.ndarray, w: np.ndarray
) -> LeadFit:
    """Weighted isotonic fit + weighted diagnostics for one lead."""
    calibrator = fit_isotonic_weighted(raw, out, w)
    cal = np.asarray(calibrator.predict(raw), dtype=np.float64)
    y = out.astype(np.float64)
    sum_w = float(np.sum(w))
    return LeadFit(
        lead=lead,
        calibrator=calibrator,
        n_samples=int(raw.size),
        effective_n=kish_effective_n(w),
        base_rate=float(np.sum(w * y) / sum_w),
        brier_before=brier_score_weighted(raw, y, w),
        brier_after=brier_score_weighted(cal, y, w),
        sum_w_sq_before=float(np.sum(w * (raw - y) ** 2)),
        sum_w_sq_after=float(np.sum(w * (cal - y) ** 2)),
        sum_w=sum_w,
    )


def _print_table(fits: list[LeadFit], skipped: dict[int, int], min_samples: int) -> None:
    print(
        f"\n{'lead':>5}  {'n':>8}  {'eff N':>9}  {'base rate':>9}  "
        f"{'Brier raw':>9}  {'Brier cal':>9}  {'':>8}"
    )
    for f in sorted(fits, key=lambda f: f.lead):
        verdict = "improved" if f.brier_after < f.brier_before else "WORSE"
        print(
            f"{f.lead:>4}m  {f.n_samples:>8}  {f.effective_n:>9.1f}  "
            f"{f.base_rate * 100:>8.2f}%  {f.brier_before:>9.4f}  "
            f"{f.brier_after:>9.4f}  {verdict:>8}"
        )
    for lead in sorted(skipped):
        print(
            f"{lead:>4}m  {skipped[lead]:>8}  "
            f"— refused: fewer than {min_samples} valid samples"
        )


def build_output(
    fits: list[LeadFit], corpus: Corpus, corpus_path: Path
) -> dict:
    """Assemble the curves JSON — format-identical to the legacy file.

    B4's manifest echo reads ``metadata.fitted_at`` / ``n_samples`` /
    ``brier_before`` / ``brier_after`` — those exact keys, top-level in
    metadata.
    """
    total_w = sum(f.sum_w for f in fits)
    metadata = {
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(corpus_path),
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "settings_hash": corpus.settings_hash,
        "settings": corpus.settings,
        "weighted": True,
        "n_events": int(corpus.n_events),
        "n_points": int(corpus.n_points),
        "n_samples": int(sum(f.n_samples for f in fits)),
        "brier_before": float(sum(f.sum_w_sq_before for f in fits) / total_w),
        "brier_after": float(sum(f.sum_w_sq_after for f in fits) / total_w),
        "leads": {
            str(f.lead): {
                "n_samples": int(f.n_samples),
                "effective_n": float(f.effective_n),
                "base_rate": float(f.base_rate),
                "brier_before": float(f.brier_before),
                "brier_after": float(f.brier_after),
            }
            for f in sorted(fits, key=lambda f: f.lead)
        },
    }
    curves = {
        str(f.lead): {
            "raw_breakpoints": list(f.calibrator.raw_breakpoints),
            "calibrated_values": list(f.calibrator.calibrated_values),
        }
        for f in sorted(fits, key=lambda f: f.lead)
    }
    return {"metadata": metadata, "curves": curves}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--corpus", type=Path, required=True,
        help="v2 multi-point corpus Parquet (build_calibration_corpus.py output)",
    )
    ap.add_argument(
        "--output", type=Path, required=True,
        help="national_curves.json destination (load_calibration_curves format)",
    )
    ap.add_argument(
        "--min-samples-per-lead", type=int, default=500,
        help="Refuse to fit a lead with fewer valid samples than this — a "
             "thin lead produces a noise curve, not a calibration (default 500).",
    )
    args = ap.parse_args(argv)

    try:
        corpus = load_v2_corpus(args.corpus)
    except CorpusError as exc:
        print(f"refusing to fit: {exc}", file=sys.stderr)
        return 2

    mask = valid_mask(corpus)
    leads = sorted(set(int(x) for x in corpus.lead_min[mask]))
    print(
        f"Corpus {args.corpus}: settings_hash={corpus.settings_hash}, "
        f"{corpus.n_events} events × {corpus.n_points} points, "
        f"{int(mask.sum())} valid rows across leads {leads}"
    )

    fits: list[LeadFit] = []
    skipped: dict[int, int] = {}
    for lead in leads:
        sel = mask & (corpus.lead_min == lead)
        n = int(sel.sum())
        if n < args.min_samples_per_lead:
            skipped[lead] = n
            print(
                f"lead +{lead} min: only {n} valid samples "
                f"(< {args.min_samples_per_lead}) — refusing to fit noise; "
                "raise --n-events on the corpus build or lower "
                "--min-samples-per-lead deliberately.",
                file=sys.stderr,
            )
            continue
        fits.append(
            fit_lead(
                lead,
                corpus.raw_prob[sel],
                corpus.outcome[sel],
                corpus.sample_weight[sel],
            )
        )

    _print_table(fits, skipped, args.min_samples_per_lead)

    if not fits:
        print("\nno lead had enough valid samples — nothing written", file=sys.stderr)
        return 2

    payload = build_output(fits, corpus, args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(
        f"\nWrote {len(fits)} weighted curves to {args.output} "
        f"(pooled Brier {payload['metadata']['brier_before']:.4f} → "
        f"{payload['metadata']['brier_after']:.4f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
