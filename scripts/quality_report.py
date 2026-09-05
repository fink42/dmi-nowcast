#!/usr/bin/env python3
"""Build ``quality.json`` — the document behind the site's /quality page.

Thin CLI over :func:`dmi_nowcast_core.quality_report.build_quality_report`.
Every input is optional and a missing one nulls its section rather than
faking it, so this is safe to run before the whole corpus exists: the first
report can be a window and a persistence margin, and it grows as the
evidence does.

Usage — the nightly build on the private VM, every input::

    python scripts/quality_report.py \\
        --radar-corpus /var/lib/dmi-nowcast-corpus/calibration/latest.parquet \\
        --station-corpus /var/lib/dmi-nowcast-corpus/stations/station_corpus_gauge.parquet \\
        --replay-dir /var/lib/dmi-nowcast-corpus/stations/replay \\
        --corpus-dir /var/lib/dmi-nowcast-corpus \\
        --persistence-json /var/lib/dmi-nowcast-corpus/pva/results.json \\
        --national-curves /var/lib/dmi-nowcast/national_curves.json \\
        --thresholds /var/lib/dmi-nowcast/push_thresholds.json \\
        --out-json /var/lib/dmi-nowcast/nowcast/quality.json \\
        --out-md /var/lib/dmi-nowcast-corpus/quality_reports/$(date -u +%Y%m%d).md

The JSON is written atomically (tmp + rename in the target directory), so
the sidecar route can never serve a half-written document.

Exit codes: 0 on success, 2 on a CLI error, 1 when ``--strict`` is given
and the produced document does not satisfy the schema checker.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from dmi_nowcast_core.quality_report import (  # noqa: E402
    QualityInputs,
    build_quality_report,
    render_markdown,
    validate_report,
)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the quality.json report (and its markdown twin).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--radar-corpus", type=Path, default=None,
                   help="national calibration corpus Parquet (radar truth)")
    p.add_argument("--station-corpus", type=Path, default=None,
                   help="station corpus widened by join_gauge_truth.py (gauge truth)")
    p.add_argument("--replay-dir", type=Path, default=None,
                   help="replay_warnings.py output directory")
    p.add_argument("--corpus-dir", type=Path, default=None,
                   help="corpus root holding stations/{obs,eval,catalogue,points}")
    p.add_argument("--persistence-json", type=Path, default=None,
                   help="persistence_vs_advection.py results.json")
    p.add_argument("--national-curves", type=Path, default=None,
                   help="the served national isotonic curves; without them "
                        "reliability is of the RAW ensemble fraction")
    p.add_argument("--thresholds", type=Path, default=None,
                   help="push_thresholds.json from sweep_thresholds.py: the "
                        "fitted horizon->threshold table the push rule serves")
    p.add_argument("--live-days", type=int, default=90,
                   help="how far back live stations/eval rows are read")
    p.add_argument("--live-days-secondary", type=int, default=30,
                   help="recency window the 'recent warnings' list is drawn from")
    p.add_argument("--headline-lead-min", type=int, default=30,
                   help="the lead the headline reliability sentence quotes")
    p.add_argument("--out-json", type=Path, default=Path("quality.json"))
    p.add_argument("--out-md", type=Path, default=None,
                   help="markdown twin for the archive; skipped when omitted")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 when the produced document fails the schema check")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = QualityInputs(
        radar_corpus=args.radar_corpus,
        station_corpus=args.station_corpus,
        replay_dir=args.replay_dir,
        corpus_dir=args.corpus_dir,
        persistence_json=args.persistence_json,
        national_curves=args.national_curves,
        thresholds_path=args.thresholds,
        live_days=args.live_days,
        live_days_secondary=args.live_days_secondary,
        headline_lead_min=args.headline_lead_min,
    )
    report = build_quality_report(inputs)
    problems = validate_report(report)
    for problem in problems:
        print(f"schema: {problem}", file=sys.stderr)

    _write_atomic(args.out_json, json.dumps(report, indent=1, sort_keys=False))
    print(f"wrote {args.out_json}", file=sys.stderr)
    if args.out_md is not None:
        _write_atomic(args.out_md, render_markdown(report))
        print(f"wrote {args.out_md}", file=sys.stderr)

    filled = [
        key for key in ("windows", "headline", "reliability", "raining_now",
                        "stations", "events", "methods", "thresholds")
        if report.get(key) is not None
    ]
    print(json.dumps({
        "generated_at_utc": report["generated_at_utc"],
        "sections": filled,
        "schema_problems": len(problems),
    }, indent=2))
    return 1 if (problems and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
