#!/usr/bin/env python3
"""CLI over the push-threshold fit.

The sweep itself — the replay, the scoring, the picks, the document —
lives in ``dmi_nowcast_sidecar.threshold_sweep``, because the nightly
quality-report task runs the same fit on the same rows and the two must
not be able to drift apart. Read that module's docstring for what is being
measured and why; this file only turns flags into a
:class:`~dmi_nowcast_sidecar.threshold_sweep.SweepOptions`, writes the
outputs, and prints the summary.

One thing is the CLI's own: ``--previous``. A refit is damped against the
table already in service
(``dmi_nowcast_core.push_thresholds.apply_stability_guard``) so that a
lead's threshold moves only when the new pick is at least
``--min-delta-pct`` away AND stands on ``--min-warnings`` scored warnings.
Pass the file the service is reading now; omit it for the first fit, which
then ships every pick as ``"first_fit"``.

Usage (on the VM that holds the corpus)::

    python scripts/sweep_thresholds.py \\
        --decisions-dir /var/lib/dmi-nowcast-corpus/stations/replay/decisions \\
        --decisions-dir /var/lib/dmi-nowcast-corpus/stations/eval \\
        --corpus-dir /var/lib/dmi-nowcast-corpus \\
        --leads 10,20,30,45,60 --thresholds 20:80:5 --workers 8 \\
        --out-json sweep.json --out-md sweep.md --out-csv sweep.csv \\
        --out-thresholds push_thresholds.json \\
        --previous /var/lib/dmi-nowcast/push_thresholds.json

Offline and read-only: parquet in, files out. No DMI calls, no network.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
_SIDECAR = _REPO_ROOT / "sidecar"
if str(_SIDECAR) not in sys.path:
    sys.path.insert(0, str(_SIDECAR))

from dmi_nowcast_core.push_thresholds import (  # noqa: E402
    DEFAULT_FALLBACK_THRESHOLD_PCT,
    DEFAULT_MIN_DELTA_PCT,
    apply_stability_guard,
    load_thresholds,
    validate_thresholds,
)
from dmi_nowcast_core.warning_score import (  # noqa: E402
    DEFAULT_COVERAGE_GAP_MIN,
    DEFAULT_DRY_MIN,
    DEFAULT_PRODUCT_LEADS_MIN,
    DEFAULT_TOLERANCE_MIN,
)

# Re-exported wholesale so this module stays the entry point it always was
# (``sweep_thresholds.pick_plateau``, ``sweep_thresholds.load_decisions``,
# … all still resolve here) while the implementation lives in the package.
from dmi_nowcast_sidecar.threshold_sweep import *  # noqa: E402,F403
from dmi_nowcast_sidecar.threshold_sweep import (  # noqa: E402
    DEFAULT_FAR_CAP,
    DEFAULT_MIN_WARNINGS,
    DEFAULT_PLATEAU_FRAC,
    FIT_MIN_USEFUL_LEAD_MIN,
    SweepError,
    SweepOptions,
    _threshold_of,
    parse_leads,
    parse_thresholds,
    render_markdown,
    run_fit,
    write_atomic,
    write_csv,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sweep (lead, threshold) for the push rule against gauges.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--decisions-dir", type=Path, nargs="+", required=True,
                   action="extend", dest="decisions_dirs",
                   help="directory tree of decision parquet files; repeatable, "
                        "later directories win a (radar_ts, station_id) tie")
    p.add_argument("--radar-decisions-dir", type=Path, nargs="+",
                   action="extend", dest="radar_decisions_dirs", default=None,
                   help="optional second decision set over the radar "
                        "calibration points, scored against the radar's own "
                        "observed_mm_h; a self-consistency check, not truth")
    p.add_argument("--corpus-dir", type=Path, required=True,
                   help="gauge store root (the directory holding stations/)")
    p.add_argument("--leads", default=",".join(
        str(lead) for lead in DEFAULT_PRODUCT_LEADS_MIN),
        help="comma-separated lead times; leads with no column are skipped")
    p.add_argument("--thresholds", default="20:80:5",
                   help="lo:hi[:step] (inclusive) or a comma-separated list")
    p.add_argument("--rearm-after-min", type=int, default=60)
    p.add_argument("--persistence-obs", type=int, default=1)
    p.add_argument("--tolerance-min", type=int, default=DEFAULT_TOLERANCE_MIN)
    p.add_argument("--dry-min", type=int, default=DEFAULT_DRY_MIN)
    p.add_argument("--coverage-gap-min", type=int,
                   default=DEFAULT_COVERAGE_GAP_MIN,
                   help="a longer gap between frames ends a coverage run and "
                        "restarts every subscription armed")
    p.add_argument("--far-cap", type=float, default=DEFAULT_FAR_CAP,
                   help="the ceiling the second pick's false-alarm rate obeys")
    p.add_argument("--min-useful-lead-min", type=float,
                   default=FIT_MIN_USEFUL_LEAD_MIN,
                   help="rain arriving sooner than this after a warning "
                        "makes it LATE: not a hit, not a false alarm, still "
                        "counted against recall")
    p.add_argument("--plateau-frac", type=float, default=DEFAULT_PLATEAU_FRAC,
                   help="share of the lead's best F1 a threshold must reach "
                        "to be on the plateau the pick is the midpoint of")
    p.add_argument("--min-warnings", type=int, default=DEFAULT_MIN_WARNINGS,
                   help="a lead with fewer scored warnings than this across "
                        "the whole grid gets no pick at all")
    p.add_argument("--fallback-threshold-pct", type=int,
                   default=DEFAULT_FALLBACK_THRESHOLD_PCT,
                   help="the threshold a lead with no pick falls back to")
    p.add_argument("--previous", type=Path, default=None,
                   help="the table currently in service; the stability guard "
                        "keeps its value for any lead the new fit does not "
                        "clearly improve on. Omit for the first fit.")
    p.add_argument("--min-delta-pct", type=int, default=DEFAULT_MIN_DELTA_PCT,
                   help="a new pick must differ from the served one by at "
                        "least this many points to replace it")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--out-thresholds", type=Path, default=None,
                   help="the fitted push_thresholds.json the service reads")
    p.add_argument("--workers", type=int, default=1,
                   help="processes over (lead, threshold) cells")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        requested_leads = parse_leads(args.leads)
        thresholds = parse_thresholds(args.thresholds)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.persistence_obs < 1:
        print("error: --persistence-obs must be >= 1", file=sys.stderr)
        return 2
    if not 0.0 < args.far_cap <= 1.0:
        print("error: --far-cap must be in (0, 1]", file=sys.stderr)
        return 2
    if not 0.0 < args.plateau_frac <= 1.0:
        print("error: --plateau-frac must be in (0, 1]", file=sys.stderr)
        return 2
    if args.min_useful_lead_min < 0:
        print("error: --min-useful-lead-min must not be negative", file=sys.stderr)
        return 2
    if args.min_warnings < 0:
        print("error: --min-warnings must not be negative", file=sys.stderr)
        return 2
    if not 0 < args.fallback_threshold_pct < 100:
        print(
            "error: --fallback-threshold-pct must be in (0, 100)",
            file=sys.stderr,
        )
        return 2
    if args.min_delta_pct < 0:
        print("error: --min-delta-pct must not be negative", file=sys.stderr)
        return 2

    options = SweepOptions(
        decisions_dirs=list(args.decisions_dirs),
        corpus_dir=Path(args.corpus_dir),
        radar_decisions_dirs=(
            list(args.radar_decisions_dirs) if args.radar_decisions_dirs else None
        ),
        leads=requested_leads,
        thresholds=thresholds,
        rearm_after_min=int(args.rearm_after_min),
        persistence_obs=int(args.persistence_obs),
        tolerance_min=int(args.tolerance_min),
        dry_min=int(args.dry_min),
        coverage_gap_min=int(args.coverage_gap_min),
        far_cap=float(args.far_cap),
        min_useful_lead_min=float(args.min_useful_lead_min),
        plateau_frac=float(args.plateau_frac),
        min_warnings=int(args.min_warnings),
        fallback_threshold_pct=int(args.fallback_threshold_pct),
        workers=int(args.workers),
    )
    try:
        payload = run_fit(options, log=log)
    except SweepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    cells = payload["cells"]
    # The stability guard, against the table the service is reading now. A
    # missing or unusable --previous is a first fit, not an error: that is
    # exactly the state this file is in before it has ever been written.
    previous = load_thresholds(args.previous) if args.previous else None
    if args.previous and previous is None:
        log(f"no usable previous table at {args.previous}; treating as a first fit")
    thresholds_doc = apply_stability_guard(
        payload["thresholds"], previous,
        min_delta_pct=int(args.min_delta_pct),
        min_warnings=int(args.min_warnings),
    )
    problems = validate_thresholds(thresholds_doc)
    for problem in problems:
        log(f"thresholds schema: {problem}")
    payload["thresholds"] = thresholds_doc

    if args.out_thresholds:
        write_atomic(
            args.out_thresholds, json.dumps(thresholds_doc, indent=1) + "\n",
        )
        log(f"wrote {args.out_thresholds}")
    if args.out_json:
        write_atomic(
            args.out_json, json.dumps(payload, indent=1, default=str) + "\n",
        )
        log(f"wrote {args.out_json}")
    if args.out_md:
        write_atomic(args.out_md, render_markdown(payload))
        log(f"wrote {args.out_md}")
    if args.out_csv:
        write_csv(Path(args.out_csv), cells, payload["do_nothing"])
        log(f"wrote {args.out_csv}")

    log(f"swept {len(cells)} cell(s) in {time.time() - started:.1f}s")
    picks = payload["picks"]
    print(json.dumps({
        "window": payload["window"],
        "picks": {
            lead: {
                # The fitted answer first, then the two secondary picks as
                # bare thresholds — the whole cell is in the JSON output.
                **{
                    key: entry.get(key) for key in (
                        "threshold_pct", "insufficient", "f1", "precision",
                        "recall", "far", "csi", "late", "plateau",
                        "radar_plateau", "agrees_with_radar", "guard",
                    )
                },
                "max_csi_pct": _threshold_of(picks[lead].get("max_csi")),
                "far_capped_pct": _threshold_of(
                    picks[lead].get("max_pod_far_capped"),
                ),
            }
            for lead, entry in thresholds_doc["leads"].items()
        },
        "thresholds_schema_problems": len(problems),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
