"""Paired day-block bootstrap between two Layer A runs.

Phase H's acceptance gate (``forecast_skill_plan.md`` §2, DECIDE-6) is
written in terms of a confidence interval on a *difference*: a candidate
counts as better only when the 95 % interval on candidate-minus-baseline
excludes zero, per stratum and per lead. This script computes that
interval from two ``persistence_vs_advection.py`` JSON outputs.

Why it has to be paired and blocked by day:

* **Blocked.** Frames ten minutes apart show the same rain. A bootstrap
  over frames or pixels would treat 3,800 highly correlated cases as
  3,800 independent ones and report an interval an order of magnitude too
  narrow — it would call every candidate significant.
* **Paired.** Both runs scored the identical case list, so each resample
  draws one set of days and applies it to both. The day-to-day variance
  that dominates either score cancels in the difference, which is what
  makes a 0.005 CSI move detectable at all.
* **Re-pooled, never averaged.** Each resample sums the contingency
  tables (and the FSS numerator / denominator) of the drawn days and only
  then forms the ratio. Averaging per-day CSIs would weight a drizzly day
  the same as a frontal one.

Both runs must have been produced with the same ``--days``, ``--horizons``
and ``--thresholds``; the script intersects on day, stratum, horizon,
threshold and scale, and reports anything it had to drop.

One stratum is not strictly like-for-like: the ``speed < / >= 20 km/h``
split is read off each run's OWN bulk motion, so a borderline case can
land on different sides in the two runs. Pooled, monthly and seasonal
rows compare identical case sets; the speed rows are diagnostic.

Usage::

    python scripts/compare_layer_a.py \\
        --baseline production.json --candidate gated.json \\
        --out-md compare.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

# The repo layout puts the algorithm library under src/ and this script's
# siblings under scripts/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from dmi_nowcast_core.benchmark import (  # noqa: E402
    contingency_sum, paired_block_bootstrap,
)
from dmi_nowcast_core.evaluate import csi  # noqa: E402
from persistence_vs_advection import METHODS, table_of  # noqa: E402

#: FSS component layout in the JSON: ``[p_mse, p_ref, a_mse, a_ref]``.
_FSS_SLOTS = {"persistence": (0, 1), "advection": (2, 3)}


def csi_statistic(blocks: Sequence[Sequence[int]]) -> float:
    """Pooled CSI over a resampled list of per-day count vectors."""
    return csi(contingency_sum([table_of(b) for b in blocks]))


def fss_statistic(blocks: Sequence[Sequence[float]]) -> float:
    """Pooled FSS over a resampled list of per-day ``(mse, ref)`` pairs."""
    mse = math.fsum(b[0] for b in blocks)
    ref = math.fsum(b[1] for b in blocks)
    if ref <= 0:
        return float("nan")
    return 1.0 - mse / ref


def day_blocks(payload: dict[str, Any]) -> dict[str, Any]:
    """The ``days`` section, or a clear error if the run predates it."""
    days = payload.get("days")
    if not isinstance(days, dict):
        raise SystemExit(
            "input JSON has no 'days' block — rerun persistence_vs_advection.py "
            "(the per-day tables the bootstrap needs were added in Phase H0a)"
        )
    return days


def _collect(
    days: dict[str, Any],
    day_list: Sequence[str],
    stratum: str,
    horizon: str,
    pick: Callable[[dict[str, Any]], Sequence[float] | None],
) -> list[Sequence[float]] | None:
    """One block per day, or None if any day is missing the entry."""
    out: list[Sequence[float]] = []
    for day in day_list:
        block = days.get(day, {}).get(stratum, {}).get(horizon)
        if block is None:
            return None
        value = pick(block)
        if value is None:
            return None
        out.append(value)
    return out


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    method: str = "advection",
    n_resamples: int = 500,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Every (stratum, horizon, threshold) difference with its interval.

    Returns ``{"rows": [...], "skipped": [...], "days": [...]}``. A row
    carries the two point scores, the difference and the interval; the
    caller decides how to render it.
    """
    base_days = day_blocks(baseline)
    cand_days = day_blocks(candidate)
    shared_days = sorted(set(base_days) & set(cand_days))
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    if not shared_days:
        return {"rows": rows, "skipped": ["no day is present in both runs"],
                "days": shared_days}

    strata = sorted(
        {s for d in shared_days for s in base_days[d]}
        & {s for d in shared_days for s in cand_days[d]}
    )
    for stratum in strata:
        horizons = sorted(
            {h for d in shared_days for h in base_days[d].get(stratum, {})}
            & {h for d in shared_days for h in cand_days[d].get(stratum, {})},
            key=int,
        )
        for horizon in horizons:
            # Days that carry this stratum × horizon in BOTH runs. A day of
            # only fast-moving rain contributes to one speed stratum, so
            # the usable day list is per cell, not global.
            usable = [
                d for d in shared_days
                if horizon in base_days[d].get(stratum, {})
                and horizon in cand_days[d].get(stratum, {})
            ]
            if not usable:
                continue
            sample = base_days[usable[0]][stratum][horizon]
            for threshold in sorted(sample["thresholds"], key=float):
                rows.extend(_threshold_rows(
                    base_days, cand_days, usable, stratum, horizon, threshold,
                    method=method, n_resamples=n_resamples, seed=seed, ci=ci,
                    skipped=skipped,
                ))
    return {"rows": rows, "skipped": skipped, "days": shared_days}


def _threshold_rows(
    base_days: dict[str, Any],
    cand_days: dict[str, Any],
    usable: Sequence[str],
    stratum: str,
    horizon: str,
    threshold: str,
    *,
    method: str,
    n_resamples: int,
    seed: int,
    ci: float,
    skipped: list[str],
) -> list[dict[str, Any]]:
    """CSI plus one FSS row per neighbourhood, for one threshold."""
    where = f"{stratum} / +{horizon} min / {threshold} mm/h"
    rows: list[dict[str, Any]] = []

    base_ct = _collect(
        base_days, usable, stratum, horizon,
        lambda b: b["thresholds"].get(threshold, {}).get(method),
    )
    cand_ct = _collect(
        cand_days, usable, stratum, horizon,
        lambda b: b["thresholds"].get(threshold, {}).get(method),
    )
    if base_ct is None or cand_ct is None:
        skipped.append(f"{where}: CSI missing in one run")
    else:
        point, lo, hi = paired_block_bootstrap(
            cand_ct, base_ct, csi_statistic, n_resamples, seed, ci,
        )
        rows.append({
            "stratum": stratum, "horizon": horizon, "threshold": threshold,
            "metric": "CSI", "scale_km": None,
            "baseline": csi_statistic(base_ct),
            "candidate": csi_statistic(cand_ct),
            "diff": point, "lo": lo, "hi": hi, "n_days": len(usable),
        })

    slots = _FSS_SLOTS[method]
    scales = sorted(
        base_days[usable[0]][stratum][horizon]["fss"].get(threshold, {}),
        key=float,
    )
    for scale in scales:
        def pick(block: dict[str, Any], _s: str = scale) -> Sequence[float] | None:
            comp = block["fss"].get(threshold, {}).get(_s)
            return None if comp is None else (comp[slots[0]], comp[slots[1]])

        base_f = _collect(base_days, usable, stratum, horizon, pick)
        cand_f = _collect(cand_days, usable, stratum, horizon, pick)
        if base_f is None or cand_f is None:
            skipped.append(f"{where}: FSS {scale} km missing in one run")
            continue
        point, lo, hi = paired_block_bootstrap(
            cand_f, base_f, fss_statistic, n_resamples, seed, ci,
        )
        rows.append({
            "stratum": stratum, "horizon": horizon, "threshold": threshold,
            "metric": f"FSS {scale} km", "scale_km": float(scale),
            "baseline": fss_statistic(base_f),
            "candidate": fss_statistic(cand_f),
            "diff": point, "lo": lo, "hi": hi, "n_days": len(usable),
        })
    return rows


def _fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:+.4f}" if abs(value) < 1 else f"{value:.4f}"


def _plain(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def verdict(row: dict[str, Any]) -> str:
    """Whether the interval clears zero, and in which direction."""
    lo, hi = row["lo"], row["hi"]
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "?"
    if lo > 0:
        return "better"
    if hi < 0:
        return "worse"
    return "tie"


def markdown_report(result: dict[str, Any], meta: dict[str, Any]) -> str:
    lines = ["# Layer A comparison — paired day-block bootstrap", ""]
    for key, value in meta.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append(
        "`diff` is candidate minus baseline. `verdict` is **better** when the "
        "interval lies entirely above zero, **worse** when entirely below, "
        "**tie** when it straddles zero."
    )
    lines.append("")
    lines.append(
        "The `speed <` / `speed >=` strata are assigned from each run's own "
        "bulk motion, so a borderline case can sit on different sides in the "
        "two runs; those rows are diagnostic, not like-for-like."
    )
    lines.append("")

    rows = result["rows"]
    strata = ["pooled"] + sorted({r["stratum"] for r in rows} - {"pooled"})
    for stratum in strata:
        subset = [r for r in rows if r["stratum"] == stratum]
        if not subset:
            continue
        lines += [f"## {stratum}", ""]
        lines.append(
            "| horizon | threshold | metric | baseline | candidate | diff | "
            "95% CI | days | verdict |"
        )
        lines.append("|" + "---|" * 9)
        for row in sorted(
            subset, key=lambda r: (int(r["horizon"]), float(r["threshold"]),
                                   r["scale_km"] if r["scale_km"] is not None else -1.0)
        ):
            lines.append(
                f"| +{row['horizon']} min | {row['threshold']} mm/h | "
                f"{row['metric']} | {_plain(row['baseline'])} | "
                f"{_plain(row['candidate'])} | {_fmt(row['diff'])} | "
                f"[{_fmt(row['lo'])}, {_fmt(row['hi'])}] | {row['n_days']} | "
                f"{verdict(row)} |"
            )
        lines.append("")

    if result["skipped"]:
        lines += ["## skipped", ""]
        lines += [f"- {s}" for s in result["skipped"][:50]]
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--baseline", required=True, type=Path,
                   help="persistence_vs_advection.py JSON for the baseline run")
    p.add_argument("--candidate", required=True, type=Path,
                   help="persistence_vs_advection.py JSON for the candidate run")
    p.add_argument("--method", default="advection", choices=list(METHODS),
                   help="which forecast to compare (default advection; "
                        "persistence is a sanity check and should tie)")
    p.add_argument("--resamples", type=int, default=500,
                   help="bootstrap resamples (plan fixes 500)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--out-md", type=Path)
    p.add_argument("--out-json", type=Path)
    args = p.parse_args(argv)

    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    result = compare(
        baseline, candidate,
        method=args.method, n_resamples=args.resamples,
        seed=args.seed, ci=args.ci,
    )
    meta = {
        "baseline": f"{args.baseline.name} "
                    f"(variant {baseline.get('meta', {}).get('variant', '?')})",
        "candidate": f"{args.candidate.name} "
                     f"(variant {candidate.get('meta', {}).get('variant', '?')})",
        "method": args.method,
        "days in both runs": len(result["days"]),
        "resamples": args.resamples,
        "confidence": f"{args.ci:.0%}",
        "seed": args.seed,
    }
    report = markdown_report(result, meta)
    if args.out_md:
        args.out_md.write_text(report)
    if args.out_json:
        args.out_json.write_text(
            json.dumps({"meta": meta, **result}, indent=2)
        )
    if not args.out_md and not args.out_json:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
