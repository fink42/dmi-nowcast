#!/usr/bin/env python3
"""Fit the gauge-trained post-processor over replay decision rows (H-P).

The served probability is a per-lead isotonic map of the STEPS ensemble
fraction and nothing else. With 16 members sharing one motion field that
fraction saturates, so a large cell 13 km upstream and a drizzle edge land
in the same top bin and come out at the same calibrated number
(``archive/calibration_target_20260909/``). This script fits a small model
per lead on predictors the cycle already recorded — the raw fraction at
every lead, how much rain sits upstream along the flow and how far away it
is, how fast it moves, how strong the echo overhead is, the season, the
hour, the frame age and the station's distance to a radar — against the
**same** gauge outcome Layer B scores.

What it reads and what it writes
--------------------------------
In: one or more replay run directories written by ``replay_warnings.py``
with ``--features`` (the feature columns are additive, so the same run is
still scored by ``benchmark_report.py`` unchanged), plus the gauge store.

Out, under ``--out-dir``:

``postprocess.json``
    The stable artefact: schema version, fitted-at stamp, the design's
    column names in order, the standardiser, one logistic + isotonic model
    per lead, and the training window it stands on. Fitted on **all**
    months — it is the model that would ship.
``postprocess_report.md`` / ``postprocess_report.json``
    The leave-one-(year, month)-out evidence: per lead and per stratum,
    the curve baseline's and the post-processor's BSS / ROC-AUC / PR-AUC
    on the same out-of-fold rows, the paired day-block bootstrap CI on the
    differences, the reliability bins of both, and the standardised
    coefficients per lead so a reader can see which predictors carry the
    weight.

Nothing in ``postprocess.json`` is evidence. The number that counts is in
the report, and it comes only from folds the model never saw.

The honest test
---------------
Every fold refits the WHOLE two-stage model — the logistic and the
isotonic recalibration on top of it — on the other months, then predicts
the held-out one. The baseline is the served ``p_rain_<lead>`` on exactly
the same rows, so the difference is attributable to the model rather than
to a different sample, and the bootstrap is paired by calendar day
because the two forecasts saw identical weather.

``--write-back DIR`` copies the run into ``DIR`` with the out-of-fold
probabilities added as ``p_post_<lead>`` columns — never in place, and
never the in-sample predictions. Scoring that copy with
``benchmark_report.py --probability-column 'p_post_{lead}'`` runs Layers B
and C over it with the same outcome window, dead-gauge rule, strata and
statistics as the baseline report.

Usage (on the VM that holds the corpus)::

    python scripts/fit_postprocess.py \\
        --run /var/lib/dmi-nowcast-corpus/stations/replay \\
        --corpus-dir /var/lib/dmi-nowcast-corpus \\
        --points /var/lib/dmi-nowcast-corpus/stations/station_points.json \\
        --leads 20,30,45,60 \\
        --out-dir /var/lib/dmi-nowcast-corpus/stations/postprocess \\
        --write-back /var/lib/dmi-nowcast-corpus/stations/replay_postprocess

Offline and read-only apart from its own outputs: parquet in, three files
out (plus the optional copy). Memory-lean — Arrow columns throughout, no
row dicts; the ten-month replay is around 460,000 rows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
_SIDECAR = _REPO_ROOT / "sidecar"
if str(_SIDECAR) not in sys.path:
    sys.path.insert(0, str(_SIDECAR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dmi_nowcast_core import postprocess as pp  # noqa: E402
from dmi_nowcast_core.warning_score import (  # noqa: E402
    DEFAULT_DRY_MIN,
    DEFAULT_MIN_KNOWN_SLOTS,
    DEFAULT_ONSET_MIN_MM,
)
from dmi_nowcast_sidecar.threshold_sweep import (  # noqa: E402
    SweepError,
    decision_parquets,
    parse_leads,
    write_atomic,
)

# The Layer B machinery, imported rather than restated: the outcome
# window, the dead-gauge rule, the row deduplication and the run's
# provenance block must be the report's, or "beats the baseline" is a
# claim about two different samples.
import benchmark_report as bench  # noqa: E402

#: The leads the acceptance gate is written in terms of (plan §2).
DEFAULT_LEADS: tuple[int, ...] = (20, 30, 45, 60)

#: Leads whose raw ensemble fraction goes into the design. Every lead the
#: national products publish: the SHAPE of the fraction against lead is
#: what separates a distant band from an overhead drizzle edge.
DEFAULT_DESIGN_LEADS: tuple[int, ...] = (10, 20, 30, 45, 60)

#: Column the write-back adds, per lead.
POST_COLUMN_TEMPLATE = "p_post_{lead}"


def post_column(lead: int) -> str:
    return POST_COLUMN_TEMPLATE.format(lead=int(lead))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def feature_source_columns(design_leads: Sequence[int]) -> list[str]:
    """Every stored column the design reads, in a stable order.

    ``season`` and ``hour_utc`` are absent on purpose: both are functions
    of the decision instant, which is already loaded, and deriving them
    keeps the loader numeric-only. The replay writes them to the parquet
    for a human or a DuckDB query, and
    ``tests/test_postprocess.py`` pins the derivation against the column.
    """
    names = [pp.raw_fraction_column(lead) for lead in sorted(set(design_leads))]
    names += [
        name for name in pp.DESIGN_SOURCE_COLUMNS
        if name not in ("hour_utc",)
    ]
    return list(dict.fromkeys(names))


#: Columns the decision schema already carries, so their presence says
#: nothing about whether the replay was run with ``--features``.
SHARED_SOURCE_COLUMNS: frozenset[str] = frozenset(
    {"observed_mm_h", "eta_min", "intensity_mm_h"}
)


def feature_only_columns(design_leads: Sequence[int]) -> list[str]:
    """The columns that exist ONLY when the replay wrote features."""
    return [
        name for name in feature_source_columns(design_leads)
        if name not in SHARED_SOURCE_COLUMNS
    ]


def load_rows(
    run_dirs: Sequence[Path],
    leads: Sequence[int],
    design_leads: Sequence[int],
    *,
    stations: Sequence[str] | None,
    log,
) -> dict:
    """Decision rows as numpy columns, deduplicated the report's way."""
    return bench.load_probabilities(
        run_dirs, leads, stations=stations,
        extra_columns=feature_source_columns(design_leads),
        log=log,
    )


def build_features(rows: Mapping[str, Any]) -> dict[str, Any]:
    """The model's input columns: the stored ones plus season and hour.

    Both derived columns come from ``t`` — ``generated_at``, the decision
    instant — which is the same stamp Layer B anchors its outcome window
    on and the same one the replay stamped the ``season`` column from.
    """
    features: dict[str, Any] = dict(rows["extra"])
    features["season"] = pp.seasons_from_epoch(rows["t"])
    features["hour_utc"] = pp.hours_from_epoch(rows["t"]).astype(np.float64)
    return features


def build_truth(
    rows: Mapping[str, Any], grid: Any, leads: Sequence[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{lead: (outcome, usable)}`` from the Layer B gauge grid.

    Rows at a station the grid does not carry — one the dead-gauge rule
    excluded, or one outside ``--points`` — were pointed at station 0 by
    ``_recode_stations``; they are forced unusable here rather than
    silently graded against another station's slots.
    """
    dropped = np.asarray(
        rows.get("dropped", np.zeros(rows["t"].size, dtype=bool)), dtype=bool,
    )
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for lead in leads:
        y, usable = grid.outcome(rows["t"], rows["station"], int(lead))
        out[int(lead)] = (y, usable & ~dropped)
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _round(value: Any, places: int = 6) -> Any:
    return bench._round(value, places)


def _fmt(value: Any, digits: int = 3) -> str:
    return bench._fmt(value, digits)


def _ci(triple: Sequence[float] | None, digits: int = 4) -> str:
    if not triple:
        return "–"
    return bench._ci(tuple(triple), digits)


def _strip_arrays(evaluation: Mapping[str, Any]) -> dict:
    """The LOMO payload without the per-row prediction arrays."""
    return {k: v for k, v in evaluation.items() if k != "out_of_fold"}


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = ["# Gauge-trained post-processing (Phase H, H-P)", ""]
    settings = report["settings"]
    lines += [
        f"Generated {report['generated_at_utc']}.",
        "",
        "Leave-one-(year, month)-out over the replay's decision rows. The "
        "baseline is the served `p_rain_<lead>` — the per-lead isotonic "
        "curve on the ensemble fraction — and the candidate is this "
        "model's out-of-fold prediction on **the same rows**. Every "
        "confidence interval is a paired day-block bootstrap "
        f"({settings['resamples']} resamples, "
        f"{settings['ci']:.0%}); an interval excluding zero is the plan's "
        "evidence that the difference is real.",
        "",
        "Shoulder is April alone until October and November 2026 are "
        "archived — read every shoulder row with that in mind.",
        "",
        "## The run",
        "",
        "| | |",
        "|---|---|",
        f"| runs | {', '.join(settings['run_dirs'])} |",
        f"| corpus | {settings['corpus_dir']} |",
        f"| rows | {report['rows']} |",
        f"| stations scored | {report['stations']} |",
        f"| days | {report['days']} |",
        f"| window | {report['window']['from']} → {report['window']['to']} |",
        f"| months | {report['n_months']} |",
        f"| leads | {', '.join(str(x) for x in settings['leads'])} |",
        f"| design leads | {', '.join(str(x) for x in settings['design_leads'])} |",
        f"| L2 | {settings['l2']} |",
        f"| dead gauges | "
        f"{', '.join(bench._dead_label(r) for r in report['dead_gauges']) or 'none'} |",
        "",
    ]
    provenance = report.get("run_settings") or {}
    if provenance.get("available"):
        lines += [
            "Replay provenance: "
            f"ensemble {provenance.get('ensemble_size')}, "
            f"cascade {provenance.get('n_cascade_levels')}, "
            f"downsample {provenance.get('downsample_factor')}, "
            f"flow {(provenance.get('flow') or {}).get('completion')}, "
            f"curves `{provenance.get('national_curves')}`.",
            "",
        ]
    else:
        lines += [
            "**No `summary.json` beside the rows** — the settings these "
            "features were produced under could not be checked.",
            "",
        ]

    lines += _skill_section(report)
    lines += _reliability_section(report)
    lines += _coefficient_section(report)
    lines += _fold_section(report)
    lines += _feature_section(report)
    return "\n".join(lines) + "\n"


def _skill_section(report: Mapping[str, Any]) -> list[str]:
    lines = ["## Out-of-fold skill", ""]
    for lead in report["settings"]["leads"]:
        entry = report["evaluation"]["leads"].get(str(lead))
        lines += [f"### Lead {lead} min", ""]
        if not entry:
            lines += ["No scored rows at this lead.", ""]
            continue
        lines += [
            "| stratum | n | days | base rate | BSS base | BSS post | "
            "ΔBSS [CI] | PR-AUC base | PR-AUC post | ΔPR-AUC [CI] | "
            "ROC base | ROC post |",
            "|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|",
        ]
        for stratum in (pp.POOLED,) + pp.SEASONS:
            block = entry.get(stratum)
            if not block:
                lines.append(f"| {stratum} | – | – | – | – | – | – | – | – | – | – | – |")
                continue
            base = block["baseline"]
            post = block["postprocess"]
            diff = block.get("difference") or {}
            lines.append(
                f"| {stratum} | {block['n']} | {block['days']} | "
                f"{_fmt(base['base_rate'])} | "
                f"{_fmt(base['bss'], 4)} | {_fmt(post['bss'], 4)} | "
                f"{_ci(diff.get('bss'))} | "
                f"{_fmt(base['pr_auc'], 4)} | {_fmt(post['pr_auc'], 4)} | "
                f"{_ci(diff.get('pr_auc'))} | "
                f"{_fmt(base['roc_auc'], 4)} | {_fmt(post['roc_auc'], 4)} |"
            )
        lines.append("")
    return lines


def _reliability_section(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Reliability, pooled",
        "",
        "Ten bins, `[k/10, (k+1)/10)` with `p == 1` in the last — the "
        "project-wide binning. `observed` is the share of those rows whose "
        "gauge was wet inside the lead.",
        "",
    ]
    for lead in report["settings"]["leads"]:
        entry = (report["evaluation"]["leads"].get(str(lead)) or {}).get(pp.POOLED)
        if not entry:
            continue
        lines += [
            f"### Lead {lead} min", "",
            "| bin | n base | mean p base | observed base | n post | "
            "mean p post | observed post |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        base = {row["bin"]: row for row in entry["baseline"]["reliability_table"]}
        post = {row["bin"]: row for row in entry["postprocess"]["reliability_table"]}
        for k in sorted(set(base) | set(post)):
            b, q = base.get(k), post.get(k)
            lines.append(
                f"| {k / 10:.1f}–{(k + 1) / 10:.1f} | "
                f"{'–' if b is None else b['n']} | "
                f"{'–' if b is None else _fmt(b['mean_p'])} | "
                f"{'–' if b is None else _fmt(b['observed'])} | "
                f"{'–' if q is None else q['n']} | "
                f"{'–' if q is None else _fmt(q['mean_p'])} | "
                f"{'–' if q is None else _fmt(q['observed'])} |"
            )
        lines.append("")
    return lines


def _coefficient_section(report: Mapping[str, Any]) -> list[str]:
    model = report["model"]
    leads = [int(x) for x in model["leads"]]
    names = list(model["features"]["names"])
    lines = [
        "## Standardised coefficients",
        "",
        "From the model fitted on **all** months (`postprocess.json`), on "
        "standardised features — so a coefficient is the change in log-"
        "odds for a one-standard-deviation move in that predictor, and the "
        "numbers are comparable across rows of this table. Sorted by mean "
        "absolute weight.",
        "",
        "| feature | " + " | ".join(f"{lead} min" for lead in leads) + " |",
        "|---|" + "---:|" * len(leads),
    ]
    coefficients = {
        lead: list(model["models"][str(lead)]["coefficients"]) for lead in leads
    }
    order = sorted(
        range(len(names)),
        key=lambda i: -float(np.mean([abs(coefficients[l][i]) for l in leads])),
    )
    for i in order:
        lines.append(
            f"| `{names[i]}` | "
            + " | ".join(f"{coefficients[lead][i]:+.3f}" for lead in leads)
            + " |"
        )
    lines += [
        "",
        "| | " + " | ".join(f"{lead} min" for lead in leads) + " |",
        "|---|" + "---:|" * len(leads),
        "| intercept | "
        + " | ".join(
            f"{model['models'][str(lead)]['intercept']:+.3f}" for lead in leads
        )
        + " |",
        "| base rate | "
        + " | ".join(
            _fmt(model["models"][str(lead)]["base_rate"], 4) for lead in leads
        )
        + " |",
        "| training rows | "
        + " | ".join(str(model["models"][str(lead)]["n"]) for lead in leads)
        + " |",
        "| converged | "
        + " | ".join(
            "yes" if model["models"][str(lead)]["converged"] else "no"
            for lead in leads
        )
        + " |",
        "",
    ]
    return lines


def _fold_section(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Folds",
        "",
        "One fold per archived `(year, month)`: the whole two-stage fit is "
        "redone on the other months and applied to this one.",
        "",
        "| fold | train rows | test rows | base rate (first lead) |",
        "|---|---:|---:|---:|",
    ]
    first = str(report["settings"]["leads"][0])
    for fold in report["evaluation"]["folds"]:
        lead_entry = (fold.get("leads") or {}).get(first) or {}
        lines.append(
            f"| {fold['fold']} | {fold['n_train']} | {fold['n_test']} | "
            f"{_fmt(lead_entry.get('base_rate'), 4)} |"
        )
    lines.append("")
    return lines


def _feature_section(report: Mapping[str, Any]) -> list[str]:
    lines = ["## Feature definitions", "", "| column | definition |", "|---|---|"]
    for name, definition in report["model"]["features"]["definitions"].items():
        lines.append(f"| `{name}` | {definition} |")
    lines += [
        "",
        "`observed_mm_h`, `eta_min` and `intensity_mm_h` are decision "
        "columns the model also reads; `season` and `hour_utc` are derived "
        "from the decision instant, and the replay writes them beside the "
        "rows for readers.",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------


def check_write_back(run_dirs: Sequence[Path], out_dir: Path) -> Path:
    """Refuse a write-back that overlaps the run, and return the target.

    Either nesting is refused: the copy must not land in the run, and the
    run must not land in the copy. Overwriting the rows the model was
    fitted on would destroy the baseline arm of the comparison. Checked
    before the fit runs, not after it — the fit is minutes of work.
    """
    resolved_out = Path(out_dir).resolve()
    for directory in run_dirs:
        resolved = Path(directory).resolve()
        if (
            resolved_out == resolved
            or resolved in resolved_out.parents
            or resolved_out in resolved.parents
        ):
            raise SweepError(
                f"--write-back {resolved_out} overlaps the run {resolved}; "
                "the copy must be a separate directory"
            )
    return resolved_out


def write_back(
    run_dirs: Sequence[Path],
    out_dir: Path,
    *,
    radar_ts: np.ndarray,
    station_codes: np.ndarray,
    station_ids: Sequence[str],
    predictions: Mapping[int, np.ndarray],
    log,
) -> dict:
    """Copy the run, adding ``p_post_<lead>`` from the out-of-fold rows.

    Never in place — see :func:`check_write_back`. A row the fit never saw
    (a station outside ``--points``, a file the loader skipped) gets a
    null, not a guess, and a null in that column is exactly what the
    benchmark's ``np.isfinite`` filter drops.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = check_write_back(run_dirs, out_dir)
    lookup = {station: index for index, station in enumerate(station_ids)}
    # One int64 key per scored row. The stride leaves ``len(station_ids)``
    # free as the "not one of ours" code, so an unmatched row cannot
    # collide with a real one.
    stride = len(station_ids) + 1
    keys = radar_ts.astype(np.int64) * stride + station_codes.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    if sorted_keys.size == 0:
        raise SweepError("no scored rows to write back")
    columns = {lead: np.asarray(values)[order] for lead, values in predictions.items()}

    (out_dir / "decisions").mkdir(parents=True, exist_ok=True)
    written = {"files": 0, "rows": 0, "matched": 0}
    for directory in run_dirs:
        for path in decision_parquets(Path(directory)):
            table = pq.read_table(path)
            names = set(table.schema.names)
            if not {"radar_ts", "station_id"} <= names:
                continue
            stamps = np.asarray(
                table.column("radar_ts").combine_chunks()
                .cast(pa.int64()).to_numpy(zero_copy_only=False), dtype=np.int64,
            ) // 1_000_000
            codes = np.array(
                [lookup.get(s, -1) for s in table.column("station_id").to_pylist()],
                dtype=np.int64,
            )
            probe = stamps * stride + np.where(codes < 0, len(station_ids), codes)
            index = np.searchsorted(sorted_keys, probe)
            safe = np.clip(index, 0, sorted_keys.size - 1)
            hit = (sorted_keys[safe] == probe) & (codes >= 0)
            for lead, values in columns.items():
                taken = np.where(hit, values[safe], np.nan)
                name = post_column(lead)
                if name in names:
                    table = table.drop_columns([name])
                table = table.append_column(
                    name, pa.array(taken, type=pa.float32()),
                )
            target = out_dir / "decisions" / path.name
            pq.write_table(table, target, compression="zstd")
            written["files"] += 1
            written["rows"] += table.num_rows
            written["matched"] += int(hit.sum())
            if log:
                log(f"wrote {target.name}: {int(hit.sum())}/{table.num_rows} matched")
    return written


def copy_summary(
    run_dirs: Sequence[Path], out_dir: Path, provenance: Mapping[str, Any],
) -> None:
    """The run's ``summary.json`` plus a ``postprocess`` block.

    The benchmark reads that file for its parity check, so the copy has to
    carry the original settings unchanged; the extra block records where
    ``p_post_<lead>`` came from.
    """
    for directory in run_dirs:
        path = bench.find_summary(Path(directory))
        if path is None:
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001 — an unreadable summary is not fatal
            continue
        payload["postprocess"] = dict(provenance)
        write_atomic(
            Path(out_dir) / "summary.json",
            json.dumps(payload, indent=1, default=str) + "\n",
        )
        return


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit the H-P gauge-trained probability post-processor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run", type=Path, nargs="+", required=True,
                   action="extend", dest="run_dirs",
                   help="replay run directory (or its decisions/ directory) "
                        "written with --features; repeatable, later "
                        "directories win a (radar_ts, station_id) tie")
    p.add_argument("--corpus-dir", type=Path, required=True,
                   help="gauge store root (the directory holding stations/)")
    p.add_argument("--points", type=Path, default=None,
                   help="station points file; restricts the fit to its "
                        "stations")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="postprocess.json and the two report files go here")
    p.add_argument("--leads", default=",".join(str(x) for x in DEFAULT_LEADS),
                   help="comma-separated leads to fit and score")
    p.add_argument(
        "--design-leads",
        default=",".join(str(x) for x in DEFAULT_DESIGN_LEADS),
        help="leads whose raw ensemble fraction enters the design matrix; "
             "every served lead by default, because the shape of the "
             "fraction against lead is itself a predictor",
    )
    p.add_argument("--l2", type=float, default=1.0,
                   help="ridge strength on the slopes (scikit-learn's 1/C); "
                        "the intercept is never penalised")
    p.add_argument("--isotonic-bins", type=int, default=pp.DEFAULT_ISOTONIC_BINS,
                   help="quantile bins for the isotonic recalibration knots")
    p.add_argument("--min-known-slots", type=int, default=DEFAULT_MIN_KNOWN_SLOTS,
                   help="dead-gauge rule, as in the benchmark report")
    p.add_argument("--dry-min", type=int, default=DEFAULT_DRY_MIN)
    p.add_argument("--onset-min-mm", type=float, default=DEFAULT_ONSET_MIN_MM)
    p.add_argument("--resamples", type=int, default=bench.DEFAULT_RESAMPLES,
                   help="paired day-block bootstrap resamples; 0 skips CIs")
    p.add_argument("--ci", type=float, default=bench.DEFAULT_CI)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write-back", type=Path, default=None,
                   help="copy the run here with the OUT-OF-FOLD "
                        "probabilities added as p_post_<lead>; score it "
                        "with benchmark_report.py --probability-column "
                        "'p_post_{lead}'. Never writes into the run.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    log = bench._log_to(sys.stderr)

    try:
        leads = parse_leads(args.leads)
        design_leads = parse_leads(args.design_leads)
        stations = bench.load_points(args.points)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.resamples < 0:
        print("error: --resamples must not be negative", file=sys.stderr)
        return 2
    if not 0.0 < args.ci < 1.0:
        print("error: --ci must be in (0, 1)", file=sys.stderr)
        return 2
    if args.l2 < 0:
        print("error: --l2 must not be negative", file=sys.stderr)
        return 2

    run_dirs = [Path(d) for d in args.run_dirs]
    try:
        # Checked before the fit, not after it: the LOMO is minutes of work
        # and a refused write-back would throw all of it away.
        if args.write_back is not None:
            check_write_back(run_dirs, Path(args.write_back))
        rows = load_rows(
            run_dirs, leads, design_leads, stations=stations, log=log,
        )
    except SweepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    missing = [
        name for name, values in rows["extra"].items()
        if not np.any(np.isfinite(values))
    ]
    feature_only = feature_only_columns(design_leads)
    if set(feature_only) <= set(missing):
        print(
            "error: not one of the H-P feature columns carries a value — "
            "was the replay run with --features?",
            file=sys.stderr,
        )
        return 2
    if missing:
        log(f"feature column(s) with no data, imputed throughout: {', '.join(missing)}")

    # Station identity BEFORE the grid re-codes it, so the write-back can
    # put a prediction back on the row it came from.
    original_ids = list(rows["stations"])
    original_codes = np.array(rows["station"], dtype=np.int64, copy=True)

    grid, dead_rows, scored = bench.build_gauge_grid(
        Path(args.corpus_dir), sorted(rows["stations"]),
        bench.decision_window(rows["t"]),
        dry_min=int(args.dry_min), onset_min_mm=float(args.onset_min_mm),
        min_known_slots=int(args.min_known_slots), log=log,
    )
    bench._recode_stations(rows, scored)
    truth = build_truth(rows, grid, leads)
    del grid

    features = build_features(rows)
    baseline = {int(lead): rows["p"][int(lead)] for lead in leads}
    month = pp.year_months_from_epoch(rows["t"])
    day = rows["t"] // bench.DAY_SEC

    log(f"fitting {len(leads)} lead(s) over {rows['rows']} row(s)")
    evaluation = pp.leave_one_month_out(
        features, truth, leads,
        month=month, day=day, baseline=baseline,
        l2=float(args.l2), design_leads=design_leads,
        n_resamples=int(args.resamples), seed=int(args.seed), ci=float(args.ci),
        isotonic_bins=int(args.isotonic_bins), log=log,
    )
    window = bench.decision_window(rows["t"])
    model = pp.fit_postprocess(
        features, truth, leads,
        l2=float(args.l2), design_leads=design_leads,
        isotonic_bins=int(args.isotonic_bins),
        training={
            "from": window[0].isoformat(),
            "to": window[1].isoformat(),
            "rows": int(rows["rows"]),
            "days": int(np.unique(day).size),
            "stations": len(scored),
            "months": pp.month_labels(sorted(set(int(m) for m in np.unique(month)))),
            "run_dirs": [str(d) for d in run_dirs],
            "corpus_dir": str(args.corpus_dir),
            "dead_gauges": [row["station_id"] for row in dead_rows],
        },
    )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {
            "leads": list(leads),
            "design_leads": list(design_leads),
            "l2": float(args.l2),
            "isotonic_bins": int(args.isotonic_bins),
            "min_known_slots": int(args.min_known_slots),
            "dry_min": int(args.dry_min),
            "onset_min_mm": float(args.onset_min_mm),
            "resamples": int(args.resamples),
            "ci": float(args.ci),
            "seed": int(args.seed),
            "run_dirs": [str(d) for d in run_dirs],
            "corpus_dir": str(args.corpus_dir),
            "points_file": None if args.points is None else str(args.points),
            "baseline_column": "p_rain_<lead>",
        },
        "rows": int(rows["rows"]),
        "stations": len(scored),
        "days": int(np.unique(day).size),
        "n_months": len(set(int(m) for m in np.unique(month))),
        "window": {"from": window[0].isoformat(), "to": window[1].isoformat()},
        "dead_gauges": dead_rows,
        "run_settings": bench.run_settings(run_dirs),
        "evaluation": _strip_arrays(evaluation),
        "model": model.to_json(),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(out_dir / "postprocess.json", model.dumps())
    write_atomic(
        out_dir / "postprocess_report.json",
        json.dumps(report, indent=1, default=str) + "\n",
    )
    write_atomic(out_dir / "postprocess_report.md", render_markdown(report))
    log(f"wrote {out_dir / 'postprocess.json'} and the two report files")

    if args.write_back is not None:
        try:
            written = write_back(
                run_dirs, Path(args.write_back),
                radar_ts=rows["radar_ts"],
                station_codes=original_codes,
                station_ids=original_ids,
                predictions=evaluation["out_of_fold"],
                log=log,
            )
        except SweepError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        copy_summary(run_dirs, Path(args.write_back), {
            "source": [str(d) for d in run_dirs],
            "column_template": POST_COLUMN_TEMPLATE,
            "leads": list(leads),
            "kind": "out-of-fold (leave-one-month-out)",
            "model": str(out_dir / "postprocess.json"),
            "fitted_at_utc": model.fitted_at_utc,
            **written,
        })
        log(
            f"wrote {written['files']} file(s) to {args.write_back}; score "
            "them with benchmark_report.py --probability-column "
            "'p_post_{lead}'"
        )

    log(f"done in {time.time() - started:.1f}s")
    print(json.dumps(_headline(report), indent=2, default=str))
    return 0


def _headline(report: Mapping[str, Any]) -> dict:
    out: dict[str, Any] = {
        "rows": report["rows"],
        "stations": report["stations"],
        "months": report["n_months"],
        "dead_gauges": [row["station_id"] for row in report["dead_gauges"]],
        "leads": {},
    }
    for lead in report["settings"]["leads"]:
        block = (report["evaluation"]["leads"].get(str(lead)) or {}).get(pp.POOLED)
        if not block:
            continue
        out["leads"][str(lead)] = {
            "n": block["n"],
            "bss_baseline": _round(block["baseline"]["bss"]),
            "bss_postprocess": _round(block["postprocess"]["bss"]),
            "bss_difference": (block.get("difference") or {}).get("bss"),
            "pr_auc_baseline": _round(block["baseline"]["pr_auc"]),
            "pr_auc_postprocess": _round(block["postprocess"]["pr_auc"]),
            "pr_auc_difference": (block.get("difference") or {}).get("pr_auc"),
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
