#!/usr/bin/env python3
"""Layer B and Layer C of the Phase H benchmark, over replay decision rows.

The plan (``forecast_skill_plan.md`` §2) fixes three layers. Layer A asks
whether the physics improved and screens candidates against the radar's
own field. **Layer B** asks whether ``p_rain`` improved, threshold-free,
against an instrument the nowcast never sees — the rain gauges — and it is
the primary gate. **Layer C** asks whether a user would feel it: the same
gauges, but through the shipped push rule at a threshold refit out of
sample.

This script is Layers B and C. It reads what the replay already wrote —
one decision row per (frame, station), carrying ``p_rain_<lead>`` for
every served lead — plus the gauge archive, and writes ``benchmark.md``
and ``benchmark.json``. It runs no STEPS, fetches nothing, and needs no
network.

Layer B: the probability, scored directly
-----------------------------------------
For every decision instant *t* and every lead *L*, the forecast is the
row's ``p_rain_<L>`` and the outcome is **"the gauge at that station was
wet at some point in the next L minutes"**.

*The boundary, stated once.* Gauge slots are ten minutes long and stamped
at their END: ``precip_past10min`` at 07:20Z covers (07:10Z, 07:20Z]. The
outcome window is the slots whose END falls in ``(t, t + L]``. So at
t = 07:00Z and L = 20 the window is the slots ending 07:10Z and 07:20Z,
covering (06:50Z, 07:20Z] of real time — the ten minutes before *t* ride
along, unavoidably, because the gauge cannot be asked about a finer grid
than it reports on. A window is:

* **1** when any slot in it is wet under the canonical rule
  (``warning_score``: ≥ 0.1 mm, or ≥ 1 min with precipitation). One wet
  slot proves rain whatever its neighbours did or did not report.
* **0** when no slot is wet AND every slot in the window is known.
* **excluded** otherwise — an unknown slot cannot certify a dry window,
  and grading it as dry would score the archive's gaps as forecast errors.
  Windows running off either end of the gauge grid are excluded for the
  same reason.

*t* is the row's ``generated_at``, not its ``radar_ts``: the served
probability at lead L is a claim about the L minutes after the user is
told, and ``generated_at = radar_ts + frame_age`` is when that would have
been. It is also the instant ``warning_score`` matches warnings at, so
Layers B and C are anchored identically.

Reported per lead and per stratum (pooled, summer, winter, shoulder):
Brier with Murphy's reliability / resolution / uncertainty decomposition
and BSS against the sample climatology, ROC-AUC, PR-AUC, and a ten-bin
reliability table with counts. With a candidate, the paired day-block
bootstrap CI on the BSS and PR-AUC **differences**.

Layer C: the decision, at an out-of-sample threshold
----------------------------------------------------
Leave-one-month-out. For each archived month, the F1-plateau threshold is
fitted on the OTHER months with the nightly fit's own objective and its
own code (``threshold_sweep.run_sweep`` / ``build_picks``), and the held-
out month is then scored at that threshold with the warning scorer. The
out-of-fold hits / false alarms / misses / late / pending pool across
folds into precision, recall, F1, CSI, warnings per station-day and the
lead-error median — never scoring a fit on its own training data, which
is the whole reason F1 is reported here and never used as the gate.

Each fold is a self-contained replay: the months are cut out of the tracks
BEFORE the state machine runs, so a fold's coverage runs, arming and
re-arm clock start fresh inside it, exactly as ``run_strata`` does for the
seasons. Folds are ``(year, month)`` pairs, not month numbers: once the
archive passes a year there are two Decembers.

Seasons follow the sweep's own split — summer May–September, winter
December–March, and April / October / November as shoulder. **Until
October and November 2026 are archived, "shoulder" is April alone**, and
every shoulder number below says so.

Statistics
----------
Every confidence interval is a day-block bootstrap: 500 resamples of whole
calendar days, 95 % percentile interval, and the pooled metric recomputed
from the resampled days rather than averaged over per-day scores (frames
five minutes apart share the same rain; a per-row bootstrap would report
intervals an order of magnitude too narrow). Against a candidate the draw
is PAIRED — one set of day indices used for both pipelines — because the
two saw identical weather and that variance cancels in the difference.

Provenance
----------
The report carries each run's ``summary.json`` settings — ensemble size,
cascade levels, downsample, frame age, flow completion, the rules — and
says loudly when the baseline and the candidate disagree about any of
them. A benchmark that silently compared a 16-member run against a
24-member one would be measuring the wrong thing with great precision.
It also names every station the dead-gauge rule excluded.

Usage (on the VM that holds the corpus)::

    python scripts/benchmark_report.py \\
        --baseline /var/lib/dmi-nowcast-corpus/stations/replay \\
        --corpus-dir /var/lib/dmi-nowcast-corpus \\
        --points /var/lib/dmi-nowcast-corpus/stations/station_points.json \\
        --leads 20,30,45,60 --workers 4 \\
        --out-dir /var/lib/dmi-nowcast-corpus/layer_bc

Add ``--candidate <dir>`` for an A/B. ``--layers b`` skips the expensive
half; ``--layers c`` skips Layer B.

Offline and read-only: parquet in, two files out.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
_SIDECAR = _REPO_ROOT / "sidecar"
if str(_SIDECAR) not in sys.path:
    sys.path.insert(0, str(_SIDECAR))

from dmi_nowcast_core.benchmark import (  # noqa: E402
    brier_decomposition,
    paired_block_bootstrap,
    pr_auc,
    reliability_bins,
    roc_auc,
)
from dmi_nowcast_core.push_thresholds import (  # noqa: E402
    DEFAULT_FALLBACK_THRESHOLD_PCT,
)
from dmi_nowcast_core.warning_score import (  # noqa: E402
    DEFAULT_DRY_MIN,
    DEFAULT_MIN_KNOWN_SLOTS,
    DEFAULT_ONSET_MIN_MM,
    DEFAULT_TOLERANCE_MIN,
    SLOT_MIN,
    dead_gauge_scan,
    gauge_truth_vectorised,
    p_rain_column,
    pooled_summary,
    skill_scores,
)
from dmi_nowcast_sidecar.decision_rows import (  # noqa: E402
    DAY_SEC,
    DEFAULT_PROBABILITY_TEMPLATE,
    GaugeGrid,
    build_gauge_grid,
    column_template,
    decision_window,
    load_probabilities,
    recode_stations,
)
from dmi_nowcast_sidecar.threshold_sweep import (  # noqa: E402
    DEFAULT_FAR_CAP,
    DEFAULT_MIN_WARNINGS,
    DEFAULT_PLATEAU_FRAC,
    FIT_MIN_USEFUL_LEAD_MIN,
    GAUGE_PAD_MIN,
    RAIN_THRESHOLD_MM_H,
    SEASON_MONTHS,
    SEASON_ORDER,
    SweepError,
    build_picks,
    build_shared,
    build_tracks,
    decision_parquets,
    filter_tracks_by_months,
    gauge_truth,
    load_decisions,
    month_filter,
    parse_leads,
    parse_thresholds,
    release_shared,
    replay_station,
    run_sweep,
    write_atomic,
)
from dmi_nowcast_core.warning_score import score_warnings  # noqa: E402

#: The leads the acceptance gate is written in terms of (plan §2 and
#: DECIDE-6). 10 is served but never subscribed to, so it is not a default.
DEFAULT_LEADS: tuple[int, ...] = (20, 30, 45, 60)

#: Reliability bins. Ten, matching ``quality_report._bin_index`` and
#: ``sql/reliability_pooled.sql``: bin *k* covers ``[k/K, (k+1)/K)`` with
#: ``p == 1`` folded into the last one. One binning convention project-wide.
N_BINS = 10

#: Day-block bootstrap, as the plan fixes it.
DEFAULT_RESAMPLES = 500
DEFAULT_CI = 0.95

#: The pooled stratum's label, matching the sweep's CSV.
POOLED = "all"

#: Strata reported, in order. "shoulder" is April alone until October and
#: November 2026 are archived — said next to every shoulder number.
STRATA: tuple[str, ...] = (POOLED,) + SEASON_ORDER

#: Seconds in a gauge slot. The day (the bootstrap's block key) is
#: ``decision_rows.DAY_SEC``, re-exported above.
SLOT_SEC = SLOT_MIN * 60


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _log_to(stream) -> Callable[[str], None]:
    def log(message: str) -> None:
        print(message, file=stream, flush=True)

    return log


def _round(value: Any, places: int = 6) -> Any:
    """JSON-safe rounding: NaN and None become null, not 'NaN'."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return None
    return round(number, places)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "–" if not math.isfinite(number) else f"{number:.{digits}f}"


def _ci(triple: tuple[float, float, float] | None, digits: int = 4) -> str:
    if triple is None:
        return "–"
    point, lo, hi = triple
    return f"{_fmt(point, digits)} [{_fmt(lo, digits)}, {_fmt(hi, digits)}]"


def _excludes_zero(triple: tuple[float, float, float] | None) -> bool | None:
    """Whether a CI excludes zero — the plan's evidence test, or null."""
    if triple is None:
        return None
    _point, lo, hi = triple
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return lo > 0.0 or hi < 0.0


def _months_of(stamps: Iterable[datetime]) -> list[tuple[int, int]]:
    return sorted({(ts.year, ts.month) for ts in stamps})


def _season_of_month(month: int) -> str:
    for name in SEASON_ORDER:
        if month in SEASON_MONTHS[name]:
            return name
    raise ValueError(f"month {month} belongs to no season")  # pragma: no cover


# ---------------------------------------------------------------------------
# Run provenance: what settings produced these rows
# ---------------------------------------------------------------------------

#: ``summary.json`` fields that MUST match between two runs being
#: compared. Everything else is either the thing under test or noise.
PARITY_KEYS: tuple[str, ...] = (
    "ensemble_size", "n_cascade_levels", "downsample_factor", "horizon_min",
    "threshold_mm_h", "frame_age_min",
)

#: Fields of the replay's ``anchor`` block that make two runs different
#: experiments rather than two samples of one. The observed frame-age
#: statistics are deliberately NOT here: they are an *outcome* of the
#: policy and the lags, and they vary with the archive's gaps.
ANCHOR_PARITY_KEYS: tuple[str, ...] = (
    "policy", "lag_min", "poll_interval_min", "frame_age_override_min",
    "history_mode",
)

#: Differences ``--allow-differing`` will accept as the candidate's whole
#: point rather than as a broken comparison.
ALLOWED_DIFFERENCES: tuple[str, ...] = ("anchor",)


def find_summary(directory: Path) -> Path | None:
    """The replay's ``summary.json`` for a decisions directory, if any.

    Accepts either the run root (``.../replay``) or the decisions
    directory inside it (``.../replay/decisions``), because both are
    reasonable things to point at and only one of them holds the file.
    """
    for candidate in (directory / "summary.json", directory.parent / "summary.json"):
        if candidate.is_file():
            return candidate
    return None


def run_settings(directories: Sequence[Path]) -> dict:
    """The STEPS / flow / rule settings behind a set of decision rows.

    Flattened out of ``replay_warnings``' ``summary.json``. Missing is
    reported as missing: a run with no summary beside it gets
    ``{"available": false}`` rather than production's defaults, because
    assuming parity is exactly the failure this block exists to prevent.
    """
    for directory in directories:
        path = find_summary(Path(directory))
        if path is None:
            continue
        try:
            raw = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 — an unreadable summary
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        run = raw.get("run") or {}
        steps = run.get("steps") or {}
        return {
            "available": True,
            "summary_path": str(path),
            "days": run.get("n_days"),
            "frames": run.get("n_frames"),
            "stations": run.get("n_stations"),
            "rows": run.get("n_decision_rows"),
            "frame_age_min": run.get("frame_age_min"),
            "ensemble_size": steps.get("ensemble_size"),
            "n_cascade_levels": steps.get("n_cascade_levels"),
            "downsample_factor": steps.get("downsample_factor"),
            "horizon_min": steps.get("horizon_min"),
            "threshold_mm_h": steps.get("threshold_mm_h"),
            "leads_min": steps.get("leads_min"),
            "flow": run.get("flow"),
            # L3 (2026-09-09): which frame each cycle stood on. Absent from
            # runs made before the anchor policy existed, which is itself
            # informative — those are fullRange runs at a flat frame age.
            "anchor": run.get("anchor"),
            "rules": run.get("rules"),
            "national_curves": run.get("national_curves"),
        }
    return {"available": False, "error": "no summary.json beside the rows"}


def _harmonisation_id(anchor: Mapping[str, Any] | None) -> Any:
    """The harmonisation map's identity: its digest, not its path.

    Two runs on two machines name the same file differently; a run that
    applied a re-fitted table is a different experiment even under the
    same name. The digest is what decides.
    """
    stamp = (anchor or {}).get("harmonisation") or {}
    return stamp.get("sha256") or stamp.get("path")


def anchor_differences(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any],
) -> list[str]:
    """How the two runs' anchor policies differ, in the report's words."""
    a = baseline.get("anchor") or {}
    b = candidate.get("anchor") or {}
    if not a and not b:
        return []
    out = []
    if bool(a) != bool(b):
        out.append(
            "anchor: one run records an anchor policy and the other does "
            "not (a run made before 2026-09-09 is fullRange at a flat "
            "frame age)"
        )
        return out
    for key in ANCHOR_PARITY_KEYS:
        if a.get(key) != b.get(key):
            out.append(
                f"anchor.{key}: baseline {a.get(key)!r} vs "
                f"candidate {b.get(key)!r}"
            )
    if _harmonisation_id(a) != _harmonisation_id(b):
        out.append(
            f"anchor.harmonisation: baseline {_harmonisation_id(a)!r} vs "
            f"candidate {_harmonisation_id(b)!r}"
        )
    return out


def parity_problems(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    allow: Sequence[str] = (),
) -> list[str]:
    """Settings the two runs disagree about, in the report's words.

    ``allow`` names differences that are the candidate's whole point —
    ``"anchor"`` for an L3 run, where the fresher frame IS the change
    under test. Anything not named is still a parity failure: a
    difference measured across two changes is attributable to neither.
    """
    if not baseline.get("available") or not candidate.get("available"):
        return [
            "one of the two runs has no summary.json, so parity could not "
            "be checked at all — treat every difference below as unexplained"
        ]
    problems = []
    for key in PARITY_KEYS:
        a, b = baseline.get(key), candidate.get(key)
        if a != b:
            problems.append(f"{key}: baseline {a!r} vs candidate {b!r}")
    if baseline.get("flow") != candidate.get("flow"):
        problems.append(
            f"flow: baseline {baseline.get('flow')!r} vs "
            f"candidate {candidate.get('flow')!r}"
        )
    if "anchor" not in allow:
        problems += anchor_differences(baseline, candidate)
    return problems



def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    """Ten-bin reliability curve with counts, one row per occupied bin.

    ``benchmark.reliability_bins`` does the binning — bin *k* is
    ``[k/K, (k+1)/K)`` with ``p == 1`` folded into the last, which is
    ``quality_report._bin_index``' convention and
    ``brier_decomposition``'s, so a number here, a number from the nightly
    report and a number from DuckDB all agree. This wrapper only rounds
    for JSON.
    """
    return [
        {**row, "mean_p": _round(row["mean_p"]), "observed": _round(row["observed"])}
        for row in reliability_bins(p, y, n_bins=n_bins)
    ]


def probability_scores(p: np.ndarray, y: np.ndarray) -> dict:
    """Brier decomposition + BSS + ROC-AUC + PR-AUC + the reliability table."""
    scores = dict(brier_decomposition(p, y, n_bins=N_BINS))
    scores["roc_auc"] = roc_auc(p, y)
    scores["pr_auc"] = pr_auc(p, y)
    scores["reliability_table"] = reliability_table(p, y)
    return scores


def _day_blocks(
    day: np.ndarray, p: np.ndarray, y: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{epoch day: (p, y)}`` as views into one sorted copy.

    Sorted once and sliced, rather than boolean-masked per day: the
    bootstrap draws 500 samples of every day, and a mask per draw would
    copy the whole array 500 times over.
    """
    order = np.argsort(day, kind="stable")
    days = day[order]
    ps = p[order]
    ys = y[order]
    edges = np.flatnonzero(np.diff(days)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, days.size]
    return {
        int(days[start]): (ps[start:end], ys[start:end])
        for start, end in zip(starts, ends)
    }


def _bss_of(blocks: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    if not blocks:
        return float("nan")
    p = np.concatenate([b[0] for b in blocks])
    y = np.concatenate([b[1] for b in blocks])
    return float(brier_decomposition(p, y, n_bins=N_BINS)["bss"])


def _pr_auc_of(blocks: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    if not blocks:
        return float("nan")
    p = np.concatenate([b[0] for b in blocks])
    y = np.concatenate([b[1] for b in blocks])
    return float(pr_auc(p, y))


def paired_probability_ci(
    base: dict[int, tuple[np.ndarray, np.ndarray]],
    cand: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    n_resamples: int,
    seed: int,
    ci: float,
) -> dict:
    """Paired day-block CIs on the BSS and PR-AUC differences.

    Only days BOTH runs scored take part; a day one of them is missing is
    not a difference, it is an absence, and pairing is the whole reason
    the interval on a difference is tighter than the difference of two
    intervals.
    """
    shared = sorted(set(base) & set(cand))
    if not shared:
        return {"days": 0, "bss": None, "pr_auc": None}
    blocks_a = [cand[d] for d in shared]
    blocks_b = [base[d] for d in shared]
    bss = paired_block_bootstrap(
        blocks_a, blocks_b, _bss_of,
        n_resamples=n_resamples, seed=seed, ci=ci,
    )
    area = paired_block_bootstrap(
        blocks_a, blocks_b, _pr_auc_of,
        n_resamples=n_resamples, seed=seed + 1, ci=ci,
    )
    return {
        "days": len(shared),
        "bss": [_round(v) for v in bss],
        "bss_excludes_zero": _excludes_zero(bss),
        "pr_auc": [_round(v) for v in area],
        "pr_auc_excludes_zero": _excludes_zero(area),
    }


def layer_b(
    baseline: dict,
    candidate: dict | None,
    grid: GaugeGrid,
    leads: Sequence[int],
    *,
    n_resamples: int,
    seed: int,
    ci: float,
    log=None,
) -> dict:
    """The whole of Layer B: per lead, per stratum, plus the paired CIs.

    ``n_resamples <= 0`` reports the point differences' inputs and no
    interval at all — the escape hatch for a quick look, never for a
    number that goes in front of the acceptance gate.
    """
    out: dict[str, Any] = {"leads": {}}
    for lead in leads:
        by_stratum: dict[str, Any] = {}
        base_cols = _scored_columns(baseline, grid, lead)
        cand_cols = (
            None if candidate is None else _scored_columns(candidate, grid, lead)
        )
        for stratum in STRATA:
            base_slice = _stratum_slice(base_cols, stratum)
            entry: dict[str, Any] = {
                "baseline": _scores_or_none(base_slice),
            }
            if cand_cols is not None:
                cand_slice = _stratum_slice(cand_cols, stratum)
                entry["candidate"] = _scores_or_none(cand_slice)
                entry["difference"] = (
                    None if n_resamples <= 0 else paired_probability_ci(
                        _day_blocks(*base_slice),
                        _day_blocks(*cand_slice),
                        n_resamples=n_resamples, seed=seed, ci=ci,
                    )
                )
            by_stratum[stratum] = entry
            if log:
                scored = entry["baseline"]
                log(
                    f"layer B lead {lead} {stratum}: "
                    f"n={0 if scored is None else scored['n']}, "
                    f"BSS {_fmt(None if scored is None else scored['bss'], 4)}"
                )
        out["leads"][str(lead)] = by_stratum
    return out


def _scored_columns(
    rows: dict, grid: GaugeGrid, lead: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(day, month, p, y)`` for the rows this lead can be scored on."""
    p = rows["p"][lead]
    t = rows["t"]
    station = rows["station"]
    y, usable = grid.outcome(t, station, lead)
    keep = usable & np.isfinite(p)
    t_keep = t[keep]
    month = (
        t_keep.astype("datetime64[s]").astype("datetime64[M]").astype(np.int64) % 12
    ) + 1
    return t_keep // DAY_SEC, month, p[keep], y[keep]


def _stratum_slice(
    columns: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], stratum: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(day, p, y)`` for one stratum. Pooled is everything."""
    day, month, p, y = columns
    if stratum == POOLED:
        return day, p, y
    keep = np.isin(month, np.array(SEASON_MONTHS[stratum], dtype=np.int64))
    return day[keep], p[keep], y[keep]


def _scores_or_none(
    sliced: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict | None:
    day, p, y = sliced
    if p.size == 0:
        return None
    scores = probability_scores(p, y)
    scores["days"] = int(np.unique(day).size)
    return {
        key: (_round(value) if isinstance(value, float) else value)
        for key, value in scores.items()
    }


# ---------------------------------------------------------------------------
# Layer C: leave-one-month-out threshold refit
# ---------------------------------------------------------------------------


class FoldSet:
    """One run's tracks, gauge truth and months — the input to every fold."""

    def __init__(
        self,
        directories: Sequence[Path],
        corpus_dir: Path,
        leads: Sequence[int],
        *,
        stations: Sequence[str] | None,
        dry_min: int,
        onset_min_mm: float,
        min_known_slots: int,
        coverage_gap_min: int,
        column_for: Callable[[int], str] = p_rain_column,
        log=None,
    ) -> None:
        # A non-default probability column is not in the decision schema,
        # so the aligning read would drop it; it is asked for explicitly.
        extra = [
            column_for(lead) for lead in leads
            if column_for(lead) != p_rain_column(lead)
        ]
        rows, file_leads, counts = load_decisions(
            directories, leads_min=(), extra_columns=extra, log=log,
        )
        if not rows:
            raise SweepError("no decision rows found")
        usable = [lead for lead in leads if lead in file_leads]
        if not usable:
            raise SweepError("none of the requested leads has a p_rain column")
        if stations is not None:
            allowed = {str(s) for s in stations}
            rows = [r for r in rows if str(r.get("station_id")) in allowed]
        tracks, frames = build_tracks(
            rows, usable, coverage_gap_min=coverage_gap_min,
            column_for=column_for,
        )
        self.n_rows = len(rows)
        del rows
        stamps = [stamp for series in frames.values() for stamp in series]
        del frames
        if not stamps:
            raise SweepError("no decision row carries a radar_ts")
        window = (min(stamps), max(stamps))
        self.months = _months_of(stamps)
        self.n_days = len({stamp.date() for stamp in stamps})
        del stamps
        station_ids = sorted(tracks)
        onsets, known_until, known_slots, dead = gauge_truth(
            Path(corpus_dir), station_ids, window,
            dry_min=dry_min, onset_min_mm=onset_min_mm,
            min_known_slots=min_known_slots, log=log,
        )
        if known_slots == 0:
            raise SweepError("the gauge store has no observations over this window")
        self.stations = [s for s in station_ids if s in known_until]
        if not self.stations:
            raise SweepError("no station has gauge observations")
        self.tracks = {s: tracks[s] for s in self.stations}
        del tracks
        self.onsets = onsets
        self.known_until = known_until
        self.dead = dead
        self.leads = usable
        self.window = window


def _fold_shared(
    folds: FoldSet,
    months: Sequence[tuple[int, int]],
    *,
    coverage_gap_min: int,
    tolerance_min: int,
    dry_min: int,
    onset_min_mm: float,
    min_useful_lead_min: float,
    persistence_obs: int,
    rearm_after_min: int,
) -> tuple[dict, dict, list[str]] | None:
    """One slice of the tracks, ready to sweep — ``run_strata``'s recipe.

    The months are cut BEFORE the state machine runs, so the fold's
    coverage runs, arming and re-arm clock start fresh inside it and no
    subscription carries an arm across the months that were cut away. The
    onsets are cut by the same predicate; ``known_until`` is NOT, because
    the gauge record covering a warning sent on the fold's last day runs
    into the next month whatever fold that month belongs to.
    """
    sliced = filter_tracks_by_months(
        folds.tracks, months, coverage_gap_min=coverage_gap_min,
    )
    sliced = {s: t for s, t in sliced.items() if s in set(folds.stations)}
    if not sliced:
        return None
    stations = sorted(sliced)
    inside = month_filter(months)
    onsets = {
        station: [o for o in folds.onsets.get(station, ()) if inside(o)]
        for station in stations
    }
    shared = build_shared(
        sliced, stations, folds.leads,
        onsets=onsets,
        known_until=folds.known_until,
        coverage_gap_min=coverage_gap_min,
        tolerance_min=tolerance_min,
        dry_min=dry_min,
        onset_min_mm=onset_min_mm,
        min_useful_lead_min=min_useful_lead_min,
        persistence_obs=persistence_obs,
        rearm_after_min=rearm_after_min,
    )
    return shared, onsets, stations


def score_fold(
    shared: dict, lead: int, threshold_pct: int,
) -> tuple[list[Any], dict[int, dict[str, int]], int]:
    """Replay and score one held-out fold at one threshold.

    Returns ``(per-station ScoreResults, per-day counts, warnings sent)``.
    ``score_cell`` would pool the stations for us, but the day-block
    bootstrap needs the outcomes broken out by calendar day and a pooled
    cell cannot be taken apart again. The replay and the scoring are the
    sweep's own functions, so the rule is shared even though the
    accounting is not.
    """
    lead_index = shared["leads"].index(int(lead))
    results = []
    per_day: dict[int, dict[str, int]] = {}
    n_sent = 0

    #: Outcome name → the count it belongs to. Pending and uncovered are
    #: in neither: they are out of every rate, so out of every block.
    into = {
        "hit": "hits", "late": "late", "false_alarm": "false_alarms",
        "miss": "misses",
    }

    def bump(day: date, outcome: str) -> None:
        counts = per_day.setdefault(
            day.toordinal(),
            {"hits": 0, "late": 0, "false_alarms": 0, "misses": 0},
        )
        counts[into[outcome]] += 1

    for station in shared["stations"]:
        warnings = replay_station(
            shared["tracks"][station], lead_index, int(threshold_pct),
            persistence_obs=shared["persistence_obs"],
            rearm_after_min=shared["rearm_after_min"],
            raining_now_mm_h=shared["raining_now_mm_h"],
        )
        n_sent += len(warnings)
        result = score_warnings(
            warnings,
            shared["onsets"].get(station, ()),
            lead_min=int(lead),
            tolerance_min=shared["tolerance_min"],
            dry_min=shared["dry_min"],
            onset_min_mm=shared["onset_min_mm"],
            known_until=shared["known_until"].get(station),
            coverage=shared["coverage"][int(lead)].get(station, ()),
            min_useful_lead_min=shared["min_useful_lead_min"],
        )
        results.append(result)
        # A warning is stamped by when it was sent, an unclaimed onset by
        # when the rain arrived. Pending and uncovered outcomes are in
        # neither: they are held out of every rate, so they are held out
        # of every block too.
        for warning in result.warnings:
            if warning.outcome in into:
                bump(warning.sent_utc.date(), warning.outcome)
        for onset in result.onsets:
            if onset.outcome == "miss":
                bump(onset.onset_utc.date(), "miss")
    return results, per_day, n_sent


def _f1_of(blocks: Sequence[Mapping[str, int]]) -> float:
    """F1 over pooled day counts — never the mean of per-day F1s.

    Late warnings sit on the miss side of recall and out of precision
    altogether, exactly as ``warning_score.skill_scores`` puts them, so
    the number here and the number in the sweep's table are the same
    quantity.
    """
    hits = sum(int(b["hits"]) for b in blocks)
    late = sum(int(b["late"]) for b in blocks)
    false_alarms = sum(int(b["false_alarms"]) for b in blocks)
    misses = sum(int(b["misses"]) for b in blocks)
    score = skill_scores(hits, false_alarms, misses, late)
    return float("nan") if score["f1"] is None else float(score["f1"])


def layer_c(
    folds: FoldSet,
    leads: Sequence[int],
    thresholds: Sequence[int],
    *,
    far_cap: float,
    plateau_frac: float,
    min_warnings: int,
    fallback_threshold_pct: int,
    coverage_gap_min: int,
    tolerance_min: int,
    dry_min: int,
    onset_min_mm: float,
    min_useful_lead_min: float,
    persistence_obs: int,
    rearm_after_min: int,
    workers: int,
    log=None,
) -> dict:
    """Leave-one-month-out refit and out-of-fold scoring, per lead.

    For each archived month: fit on the others with the nightly fit's
    objective (``run_sweep`` + ``build_picks``), then score that month at
    the fitted threshold. Pooling the out-of-fold counts — never averaging
    per-fold rates — is what makes the result a single honest number
    rather than an average over folds of wildly different size.
    """
    months = folds.months
    if len(months) < 2:
        raise SweepError(
            f"leave-one-month-out needs at least two months, got {len(months)}"
        )
    slice_kwargs = dict(
        coverage_gap_min=coverage_gap_min,
        tolerance_min=tolerance_min,
        dry_min=dry_min,
        onset_min_mm=onset_min_mm,
        min_useful_lead_min=min_useful_lead_min,
        persistence_obs=persistence_obs,
        rearm_after_min=rearm_after_min,
    )
    per_lead: dict[str, dict] = {
        str(lead): {
            "folds": [],
            "results": [],
            "per_day": {},
            "by_season": {},
            "n_sent": 0,
            "station_days": 0,
        }
        for lead in leads
    }
    for held_out in months:
        train_months = [m for m in months if m != held_out]
        train = _fold_shared(folds, train_months, **slice_kwargs)
        test = _fold_shared(folds, [held_out], **slice_kwargs)
        if train is None or test is None:
            if log:
                log(f"fold {held_out[0]}-{held_out[1]:02d}: empty slice, skipped")
            continue
        train_shared, _train_onsets, _train_stations = train
        test_shared, test_onsets, _test_stations = test
        train_rows = int(train_shared["n_rows"])
        if log:
            log(
                f"fold {held_out[0]}-{held_out[1]:02d}: fitting on "
                f"{train_shared['n_rows']} row(s), testing on "
                f"{test_shared['n_rows']} row(s), "
                f"{sum(len(v) for v in test_onsets.values())} onset(s)"
            )
        cells, _do_nothing = run_sweep(
            train_shared, leads, thresholds, workers=int(workers),
            log=None,
        )
        del train_shared, train
        release_shared()
        picks = build_picks(
            cells, leads,
            far_cap=float(far_cap),
            plateau_frac=float(plateau_frac),
            min_warnings=int(min_warnings),
        )
        season = _season_of_month(held_out[1])
        for lead in leads:
            pick = picks[str(lead)].get("plateau")
            insufficient = bool(picks[str(lead)].get("insufficient"))
            threshold = (
                int(fallback_threshold_pct) if pick is None
                else int(pick["threshold_pct"])
            )
            results, per_day, n_sent = score_fold(test_shared, lead, threshold)
            bucket = per_lead[str(lead)]
            bucket["folds"].append({
                "month": f"{held_out[0]}-{held_out[1]:02d}",
                "season": season,
                "threshold_pct": threshold,
                "fitted": pick is not None,
                "insufficient": insufficient,
                "plateau": None if pick is None else list(pick["plateau"]),
                "train_rows": train_rows,
                "test_rows": int(test_shared["n_rows"]),
                "summary": _fold_summary(results, test_shared, n_sent),
            })
            bucket["results"].extend(results)
            bucket["n_sent"] += n_sent
            bucket["station_days"] += int(test_shared["station_days"])
            for day, counts in per_day.items():
                target = bucket["per_day"].setdefault(
                    day, {"hits": 0, "late": 0, "false_alarms": 0, "misses": 0},
                )
                for key, value in counts.items():
                    target[key] += value
            bucket["by_season"].setdefault(season, []).extend(results)
            if log:
                log(
                    f"  lead {lead}: threshold {threshold} % "
                    f"({'fitted' if pick is not None else 'fallback'}), "
                    f"{n_sent} warning(s) out of fold"
                )
        del test_shared, test

    out: dict[str, Any] = {"months": [f"{y}-{m:02d}" for y, m in months], "leads": {}}
    for lead in leads:
        bucket = per_lead[str(lead)]
        station_days = max(1, bucket["station_days"])
        pooled = pooled_summary(bucket["results"])
        entry = {
            "folds": bucket["folds"],
            "out_of_fold": _summary_row(pooled, bucket["n_sent"], station_days),
            "by_season": {
                season: _summary_row(
                    pooled_summary(results),
                    sum(
                        f["summary"]["n_sent"] for f in bucket["folds"]
                        if f["season"] == season
                    ),
                    max(1, sum(
                        f["summary"]["station_days"] for f in bucket["folds"]
                        if f["season"] == season
                    )),
                )
                for season, results in sorted(bucket["by_season"].items())
            },
            "per_day": bucket["per_day"],
        }
        out["leads"][str(lead)] = entry
    return out


def _fold_summary(results: Sequence[Any], shared: dict, n_sent: int) -> dict:
    pooled = pooled_summary(results)
    return {
        **_summary_row(pooled, n_sent, max(1, int(shared["station_days"]))),
        "station_days": int(shared["station_days"]),
    }


def _summary_row(pooled: Mapping[str, Any], n_sent: int, station_days: int) -> dict:
    spread = pooled["lead_error_min"]
    return {
        "warnings": int(pooled["warnings"]),
        "n_sent": int(pooled["n_sent"]),
        "pending": int(pooled["pending"]),
        "hits": int(pooled["hits"]),
        "late": int(pooled["late"]),
        "false_alarms": int(pooled["false_alarms"]),
        "misses": int(pooled["misses"]),
        "uncovered_onsets": int(pooled["uncovered_onsets"]),
        "n_onsets": int(pooled["n_onsets"]),
        "precision": _round(pooled["precision"]),
        "recall": _round(pooled["recall"]),
        "f1": _round(pooled["f1"]),
        "csi": _round(pooled["csi"]),
        "far": _round(pooled["far"]),
        "warnings_per_station_day": _round(n_sent / station_days),
        "lead_error_p50": _round(spread.get("p50"), 3),
        "lead_error_p25": _round(spread.get("p25"), 3),
        "lead_error_p75": _round(spread.get("p75"), 3),
        "station_days": int(station_days),
    }


def paired_f1_ci(
    base: Mapping[int, Mapping[str, int]],
    cand: Mapping[int, Mapping[str, int]],
    *,
    n_resamples: int,
    seed: int,
    ci: float,
) -> dict | None:
    """Paired day-block CI on the out-of-fold F1 difference.

    Days both runs produced out-of-fold outcomes on. The blocks are
    COUNTS, pooled and only then turned into an F1, because the mean of
    per-day F1s would weight a day with two warnings like a frontal day
    with two hundred.
    """
    shared = sorted(set(base) & set(cand))
    if not shared:
        return None
    triple = paired_block_bootstrap(
        [cand[d] for d in shared], [base[d] for d in shared], _f1_of,
        n_resamples=n_resamples, seed=seed, ci=ci,
    )
    return {
        "days": len(shared),
        "f1": [_round(v) for v in triple],
        "f1_excludes_zero": _excludes_zero(triple),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

SHOULDER_NOTE = (
    "**Shoulder is April alone** until October and November 2026 are "
    "archived (plan §4a, DECIDE-6): every shoulder number below stands on "
    "one month of one regime and should be read as provisional."
)


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    settings = report["settings"]
    lines.append("# Phase H benchmark — Layers B and C")
    lines.append("")
    lines.append(
        f"Generated {report['generated_at_utc']}. Layer B scores the served "
        "probability against the rain gauges, threshold-free; Layer C scores "
        "the shipped push rule at a leave-one-month-out refitted threshold. "
        "Neither runs STEPS: both replay rows the nowcast already wrote."
    )
    lines.append("")
    if settings.get("probability_column", DEFAULT_PROBABILITY_TEMPLATE) != (
        DEFAULT_PROBABILITY_TEMPLATE
    ):
        lines.append(
            "**Scored column: "
            f"`{settings['probability_column']}`**, not the served "
            f"`{DEFAULT_PROBABILITY_TEMPLATE}`. Every number below is "
            "about that column."
        )
        lines.append("")
    lines.append("## Runs compared")
    lines.append("")
    lines.append(_runs_table(report))
    lines.append("")
    problems = report.get("parity_problems") or []
    deliberate = report.get("deliberate_differences") or []
    if report.get("candidate") is None:
        lines.append("Baseline only — no candidate given, so no differences below.")
    elif problems:
        lines.append(
            "**The two runs are not at parity.** "
            + "; ".join(problems)
            + ". A difference measured across a settings change is not "
            "attributable to the change under test."
        )
    else:
        lines.append(
            "The two runs agree on every parity setting "
            f"({', '.join(PARITY_KEYS)}) and on the flow configuration."
        )
    if deliberate:
        lines.append("")
        lines.append(
            "**The candidate difference under test** (accepted via "
            "`--allow-differing "
            + ",".join(settings.get("allow_differing") or ())
            + "`): "
            + "; ".join(deliberate)
            + ". Everything below is the effect of that change and of "
            "nothing else the parity check can see."
        )
    elif (
        report.get("candidate") is not None
        and not problems
        and (settings.get("allow_differing") or ())
    ):
        lines.append("")
        lines.append(
            "`--allow-differing "
            + ",".join(settings.get("allow_differing") or ())
            + "` was passed, but the two runs do not differ there — the "
            "candidate is a repeat of the baseline, not a variant."
        )
    lines.append("")
    dead = report.get("dead_gauges") or []
    if dead:
        lines.append(
            f"**Dead gauges excluded** ({settings['min_known_slots']} known "
            "slots minimum, never once wet): "
            + ", ".join(_dead_label(row) for row in dead)
            + ". A bucket stuck at zero makes every wet slot at that station "
            "a false alarm nothing could have avoided."
        )
    else:
        lines.append(
            "No station met the dead-gauge rule "
            f"(≥ {settings['min_known_slots']} known slots and never wet)."
        )
    lines.append("")
    lines.append(SHOULDER_NOTE)
    lines.append("")

    if report.get("layer_b"):
        lines.extend(_layer_b_markdown(report))
    if report.get("layer_c"):
        lines.extend(_layer_c_markdown(report))
    lines.append("## Definitions")
    lines.append("")
    lines.append(
        "**Layer B outcome.** Gauge slots are 10 minutes long and stamped at "
        "their end. At a decision instant `t` and lead `L` the outcome window "
        "is the slots whose END falls in `(t, t + L]`: 1 if any is wet "
        "(≥ 0.1 mm, or ≥ 1 min with precipitation), 0 if none is wet and all "
        "are known, and EXCLUDED otherwise — an unknown slot cannot certify a "
        "dry window. `t` is `generated_at`, the instant a subscriber would "
        "have been told, not the frame stamp."
    )
    lines.append("")
    lines.append(
        "**BSS** is `1 − Brier / uncertainty`, the skill against the sample's "
        "own climatology; `reliability − resolution + uncertainty` is the "
        "Brier of the binned forecast, so it differs from the raw Brier by "
        "the within-bin spread and both are printed. **PR-AUC** is average "
        "precision, the step-wise area — not the interpolated one, which "
        f"flatters a rare-event forecast. **Every CI** is a "
        f"{settings['resamples']}-resample day-block bootstrap at "
        f"{settings['ci'] * 100:.0f} %, paired between the two runs."
    )
    lines.append("")
    lines.append(
        "**Layer C** never scores a threshold on its own training months. "
        "The pooled numbers are computed from the summed out-of-fold counts, "
        "not averaged over folds. A **late** warning — rain arriving less "
        f"than {settings['min_useful_lead_min']:.0f} min after it was sent — "
        "is not a hit and not a false alarm: it counts against recall only."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _runs_table(report: Mapping[str, Any]) -> str:
    rows = [("baseline", report["baseline"])]
    if report.get("candidate") is not None:
        rows.append(("candidate", report["candidate"]))
    header = (
        "| run | rows | days | stations | ensemble | cascade | downsample "
        "| frame age | flow completion | anchor | lags (fR/dop) | history "
        "| harmonisation |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|"
    lines = [header, sep]
    for name, block in rows:
        settings = block["settings"]
        flow = (settings.get("flow") or {}).get("completion")
        anchor = settings.get("anchor") or {}
        lag = anchor.get("lag_min") or {}
        observed = (anchor.get("frame_age_min") or {}).get("p50")
        age = settings.get("frame_age_min")
        # The flat override when a run used one; otherwise the age the run
        # actually got, which is the number the policy is judged on.
        age_cell = (
            f"{age} (flat)" if age is not None
            else (f"{observed} (p50)" if observed is not None else "–")
        )
        lags = (
            f"{lag.get('fullRange', '–')}/{lag.get('doppler', '–')}"
            if lag else "–"
        )
        stamp = (anchor.get("harmonisation") or {}).get("sha256")
        lines.append(
            f"| {name} | {block.get('rows', '–')} | "
            f"{block.get('days', '–')} | {block.get('stations', '–')} | "
            f"{settings.get('ensemble_size', '–')} | "
            f"{settings.get('n_cascade_levels', '–')} | "
            f"{settings.get('downsample_factor', '–')} | "
            f"{age_cell} | {flow or '–'} | "
            f"{anchor.get('policy', '–')} | {lags} | "
            f"{anchor.get('history_mode', '–')} | {stamp or '–'} |"
        )
    return "\n".join(lines)


def _layer_b_markdown(report: Mapping[str, Any]) -> list[str]:
    lines = ["## Layer B — probability skill at the gauges", ""]
    lines.append(
        "The primary gate (plan §2, DECIDE-1): BSS per lead on held-out "
        "weather, PR-AUC secondary. Higher BSS and higher PR-AUC are better; "
        "lower reliability is better."
    )
    lines.append("")
    has_candidate = report.get("candidate") is not None
    for lead, strata in sorted(
        report["layer_b"]["leads"].items(), key=lambda kv: int(kv[0]),
    ):
        lines.append(f"### Lead {lead} min")
        lines.append("")
        head = (
            "| stratum | n | base rate | Brier | REL | RES | UNC | BSS | "
            "ROC-AUC | PR-AUC |"
        )
        lines.append(head)
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for stratum in STRATA:
            entry = strata.get(stratum) or {}
            for which in ("baseline", "candidate"):
                scores = entry.get(which)
                if scores is None:
                    continue
                label = stratum if which == "baseline" else f"{stratum} (cand.)"
                lines.append(
                    f"| {label} | {scores['n']} | "
                    f"{_fmt(scores['base_rate'], 4)} | "
                    f"{_fmt(scores['brier'], 4)} | "
                    f"{_fmt(scores['reliability'], 4)} | "
                    f"{_fmt(scores['resolution'], 4)} | "
                    f"{_fmt(scores['uncertainty'], 4)} | "
                    f"{_fmt(scores['bss'], 4)} | "
                    f"{_fmt(scores['roc_auc'], 4)} | "
                    f"{_fmt(scores['pr_auc'], 4)} |"
                )
        lines.append("")
        if has_candidate:
            lines.append(
                "| stratum | days | ΔBSS [95 % CI] | excludes 0 | "
                "ΔPR-AUC [95 % CI] | excludes 0 |"
            )
            lines.append("|---|---:|---|---|---|---|")
            for stratum in STRATA:
                diff = (strata.get(stratum) or {}).get("difference")
                if not diff or not diff.get("bss"):
                    continue
                lines.append(
                    f"| {stratum} | {diff['days']} | "
                    f"{_ci(tuple(diff['bss']))} | "
                    f"{_yes_no(diff.get('bss_excludes_zero'))} | "
                    f"{_ci(tuple(diff['pr_auc']))} | "
                    f"{_yes_no(diff.get('pr_auc_excludes_zero'))} |"
                )
            lines.append("")
            lines.append(
                "Differences are candidate − baseline, so positive is better "
                "for both columns."
            )
            lines.append("")
        base = (strata.get(POOLED) or {}).get("baseline")
        if base and base.get("reliability_table"):
            lines.append("<details><summary>Reliability, pooled</summary>")
            lines.append("")
            lines.append("| bin | n | mean p | observed |")
            lines.append("|---|---:|---:|---:|")
            for row in base["reliability_table"]:
                lines.append(
                    f"| {row['p_lo']:.1f}–{row['p_hi']:.1f} | {row['n']} | "
                    f"{_fmt(row['mean_p'], 4)} | {_fmt(row['observed'], 4)} |"
                )
            lines.append("")
            lines.append("</details>")
            lines.append("")
    return lines


def _layer_c_markdown(report: Mapping[str, Any]) -> list[str]:
    lines = ["## Layer C — decision skill at an out-of-fold threshold", ""]
    lines.append(
        "Reported for the record, never the gate (plan §2). Every threshold "
        "below was fitted on the months OTHER than the one it scored: "
        + ", ".join(report["layer_c"]["months"])
        + "."
    )
    lines.append("")
    has_candidate = report.get("candidate") is not None
    for lead, entry in sorted(
        report["layer_c"]["leads"].items(), key=lambda kv: int(kv[0]),
    ):
        lines.append(f"### Lead {lead} min")
        lines.append("")
        lines.append(
            "| slice | warnings | hits | late | false alarms | misses | "
            "precision | recall | F1 | CSI | warn/station-day | lead err p50 |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        rows: list[tuple[str, Mapping[str, Any]]] = [
            ("pooled out-of-fold", entry["out_of_fold"]),
        ]
        for season in SEASON_ORDER:
            if season in entry["by_season"]:
                rows.append((season, entry["by_season"][season]))
        if has_candidate:
            candidate = report["layer_c_candidate"]["leads"][lead]
            rows.append(("pooled (cand.)", candidate["out_of_fold"]))
            for season in SEASON_ORDER:
                if season in candidate["by_season"]:
                    rows.append(
                        (f"{season} (cand.)", candidate["by_season"][season]),
                    )
        for label, row in rows:
            lines.append(
                f"| {label} | {row['warnings']} | {row['hits']} | "
                f"{row['late']} | {row['false_alarms']} | {row['misses']} | "
                f"{_fmt(row['precision'])} | {_fmt(row['recall'])} | "
                f"{_fmt(row['f1'])} | {_fmt(row['csi'])} | "
                f"{_fmt(row['warnings_per_station_day'])} | "
                f"{_fmt(row['lead_error_p50'], 1)} |"
            )
        lines.append("")
        difference = (report.get("layer_c_difference") or {}).get(lead)
        if difference:
            lines.append(
                f"ΔF1 (candidate − baseline): {_ci(tuple(difference['f1']))} "
                f"over {difference['days']} paired day(s); excludes zero: "
                f"{_yes_no(difference.get('f1_excludes_zero'))}."
            )
            lines.append("")
        lines.append("| fold | season | threshold | fitted | warnings | F1 |")
        lines.append("|---|---|---:|---|---:|---:|")
        for fold in entry["folds"]:
            lines.append(
                f"| {fold['month']} | {fold['season']} | "
                f"{fold['threshold_pct']} % | "
                f"{'yes' if fold['fitted'] else 'fallback'} | "
                f"{fold['summary']['warnings']} | "
                f"{_fmt(fold['summary']['f1'])} |"
            )
        lines.append("")
    return lines


def _dead_label(row: Mapping[str, Any]) -> str:
    """``06080 (3876 known slots)``, or the bare id when Layer B did not run."""
    known = row.get("known_slots")
    if known is None:
        return str(row.get("station_id"))
    return f"{row['station_id']} ({known} known slots)"


def _yes_no(value: Any) -> str:
    if value is None:
        return "–"
    return "yes" if value else "no"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_points(path: Path | None) -> list[str] | None:
    """Station ids from a v2 station points file, or None for 'all'."""
    if path is None:
        return None
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or int(raw.get("version", 0)) != 2:
        raise ValueError(f"{path}: expected a version-2 station points file")
    ids = [str(entry["id"]) for entry in raw.get("points", ())]
    if not ids:
        raise ValueError(f"{path}: no points")
    return ids


def parse_allow_differing(spec: str | None) -> tuple[str, ...]:
    """``--allow-differing`` as a validated tuple; unknown names are an error.

    Deliberately not a free-form list: silently accepting a typo would
    turn a parity failure into no warning at all, which is the one
    outcome this whole block exists to prevent.
    """
    names = tuple(x.strip() for x in (spec or "").split(",") if x.strip())
    unknown = [n for n in names if n not in ALLOWED_DIFFERENCES]
    if unknown:
        raise ValueError(
            f"--allow-differing: unknown {', '.join(unknown)}; "
            f"known: {', '.join(ALLOWED_DIFFERENCES)}"
        )
    return names


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Layer B / Layer C benchmark over replay decision rows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline", type=Path, nargs="+", required=True,
                   action="extend", dest="baseline_dirs",
                   help="replay run directory (or its decisions/ directory); "
                        "repeatable, later directories win a "
                        "(radar_ts, station_id) tie")
    p.add_argument("--candidate", type=Path, nargs="+", default=None,
                   action="extend", dest="candidate_dirs",
                   help="the run under test; omit for a baseline-only report")
    p.add_argument("--corpus-dir", type=Path, required=True,
                   help="gauge store root (the directory holding stations/)")
    p.add_argument("--points", type=Path, default=None,
                   help="station points file; restricts scoring to its "
                        "stations, so a run over the radar calibration "
                        "points cannot leak into a gauge benchmark")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="benchmark.md and benchmark.json are written here")
    p.add_argument("--leads", default=",".join(str(x) for x in DEFAULT_LEADS),
                   help="comma-separated lead times")
    p.add_argument("--layers", default="bc", choices=("b", "c", "bc"),
                   help="which layers to run; Layer C is the expensive one")
    p.add_argument(
        "--probability-column", default=DEFAULT_PROBABILITY_TEMPLATE,
        help="template naming the probability column to score, with "
             "'{lead}' standing for the lead in minutes. The default is "
             "the served p_rain_<lead>; 'p_post_{lead}' scores a run "
             "written back by scripts/fit_postprocess.py through this "
             "same report, so the two are comparable line for line.",
    )
    p.add_argument(
        "--candidate-probability-column", default=None, metavar="TEMPLATE",
        help="probability column template for the CANDIDATE arm only "
             "(default: the same as --probability-column). Lets a "
             "post-processed copy (p_post_{lead}) be scored against the "
             "original run's p_rain_{lead} in one A/B.",
    )
    p.add_argument("--thresholds", default="20:80:5",
                   help="Layer C refit grid: lo:hi[:step] or a list")
    p.add_argument("--min-known-slots", type=int, default=DEFAULT_MIN_KNOWN_SLOTS,
                   help="a station reporting at least this many gauge slots "
                        "over the window and never once wet is excluded as a "
                        "dead gauge; 0 disables the rule")
    p.add_argument("--dry-min", type=int, default=DEFAULT_DRY_MIN)
    p.add_argument("--onset-min-mm", type=float, default=DEFAULT_ONSET_MIN_MM)
    p.add_argument("--tolerance-min", type=int, default=DEFAULT_TOLERANCE_MIN)
    p.add_argument("--coverage-gap-min", type=int, default=20)
    p.add_argument("--min-useful-lead-min", type=float,
                   default=FIT_MIN_USEFUL_LEAD_MIN)
    p.add_argument("--plateau-frac", type=float, default=DEFAULT_PLATEAU_FRAC)
    p.add_argument("--min-warnings", type=int, default=DEFAULT_MIN_WARNINGS)
    p.add_argument("--far-cap", type=float, default=DEFAULT_FAR_CAP)
    p.add_argument("--fallback-threshold-pct", type=int,
                   default=DEFAULT_FALLBACK_THRESHOLD_PCT)
    p.add_argument("--persistence-obs", type=int, default=1)
    p.add_argument("--rearm-after-min", type=int, default=60)
    p.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES,
                   help="day-block bootstrap resamples; 0 skips every CI")
    p.add_argument("--ci", type=float, default=DEFAULT_CI)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1,
                   help="processes over (lead, threshold) cells in Layer C")
    p.add_argument(
        "--allow-differing", default="", dest="allow_differing",
        help="comma-separated settings the candidate is ALLOWED to differ "
             f"on, from {{{', '.join(ALLOWED_DIFFERENCES)}}}. 'anchor' is "
             "the L3 case: the candidate stands on a fresher frame, and "
             "that difference is the experiment rather than a parity "
             "failure. Everything not named here still fails parity.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    log = _log_to(sys.stderr)

    try:
        leads = parse_leads(args.leads)
        thresholds = parse_thresholds(args.thresholds)
        stations = load_points(args.points)
        allow_differing = parse_allow_differing(args.allow_differing)
        column_template(args.probability_column)
        if args.candidate_probability_column:
            column_template(args.candidate_probability_column)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.resamples < 0:
        print("error: --resamples must not be negative", file=sys.stderr)
        return 2
    if not 0.0 < args.ci < 1.0:
        print("error: --ci must be in (0, 1)", file=sys.stderr)
        return 2

    baseline_dirs = [Path(d) for d in args.baseline_dirs]
    candidate_dirs = (
        [Path(d) for d in args.candidate_dirs] if args.candidate_dirs else None
    )
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {
            "leads": list(leads),
            "layers": args.layers,
            "probability_column": args.probability_column,
            "candidate_probability_column": (
                args.candidate_probability_column or args.probability_column
            ),
            "thresholds": list(thresholds),
            "min_known_slots": int(args.min_known_slots),
            "dry_min": int(args.dry_min),
            "onset_min_mm": float(args.onset_min_mm),
            "tolerance_min": int(args.tolerance_min),
            "coverage_gap_min": int(args.coverage_gap_min),
            "min_useful_lead_min": float(args.min_useful_lead_min),
            "plateau_frac": float(args.plateau_frac),
            "min_warnings": int(args.min_warnings),
            "far_cap": float(args.far_cap),
            "fallback_threshold_pct": int(args.fallback_threshold_pct),
            "persistence_obs": int(args.persistence_obs),
            "rearm_after_min": int(args.rearm_after_min),
            "raining_now_mm_h": RAIN_THRESHOLD_MM_H,
            "resamples": int(args.resamples),
            "ci": float(args.ci),
            "seed": int(args.seed),
            "n_bins": N_BINS,
            "slot_min": SLOT_MIN,
            "points_file": None if args.points is None else str(args.points),
            "baseline_dirs": [str(d) for d in baseline_dirs],
            "candidate_dirs": (
                None if candidate_dirs is None else [str(d) for d in candidate_dirs]
            ),
            "corpus_dir": str(args.corpus_dir),
            "allow_differing": list(allow_differing),
        },
        "baseline": {"settings": run_settings(baseline_dirs)},
        "candidate": (
            None if candidate_dirs is None
            else {"settings": run_settings(candidate_dirs)}
        ),
    }
    if candidate_dirs is not None:
        report["parity_problems"] = parity_problems(
            report["baseline"]["settings"], report["candidate"]["settings"],
            allow=allow_differing,
        )
        report["deliberate_differences"] = (
            anchor_differences(
                report["baseline"]["settings"], report["candidate"]["settings"],
            ) if "anchor" in allow_differing else []
        )

    try:
        if "b" in args.layers:
            _run_layer_b(report, args, leads, stations, baseline_dirs,
                         candidate_dirs, log)
        if "c" in args.layers:
            _run_layer_c(report, args, leads, thresholds, stations,
                         baseline_dirs, candidate_dirs, log)
    except SweepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(
        out_dir / "benchmark.json", json.dumps(report, indent=1, default=str) + "\n",
    )
    write_atomic(out_dir / "benchmark.md", render_markdown(report))
    log(f"wrote {out_dir / 'benchmark.json'} and {out_dir / 'benchmark.md'}")
    log(f"done in {time.time() - started:.1f}s")
    print(json.dumps(_headline(report), indent=2, default=str))
    return 0



#: The old private name, kept because ``scripts/fit_postprocess.py`` calls
#: it through this module.
_recode_stations = recode_stations


def _candidate_column_for(args: Any) -> Callable[[int], str]:
    """The candidate arm's probability column: its own template, else the baseline's."""
    return column_template(args.candidate_probability_column or args.probability_column)


def _run_layer_b(
    report: dict, args: Any, leads: Sequence[int], stations: Sequence[str] | None,
    baseline_dirs: Sequence[Path], candidate_dirs: Sequence[Path] | None, log,
) -> None:
    column_for = column_template(args.probability_column)
    baseline = load_probabilities(
        baseline_dirs, leads, stations=stations, column_for=column_for, log=log,
    )
    candidate = (
        None if candidate_dirs is None
        else load_probabilities(
            candidate_dirs, leads, stations=stations,
            column_for=_candidate_column_for(args), log=log,
        )
    )
    ids = sorted(set(baseline["stations"]) | set(
        candidate["stations"] if candidate else []
    ))
    grid, dead_rows, scored = build_gauge_grid(
        Path(args.corpus_dir), ids, decision_window(baseline["t"]),
        dry_min=int(args.dry_min), onset_min_mm=float(args.onset_min_mm),
        min_known_slots=int(args.min_known_slots), log=log,
    )
    report["dead_gauges"] = dead_rows

    # Re-code the station column against the scored set; a row at an
    # excluded station gets code -1 and is dropped with the rest.
    for rows in (baseline, candidate):
        if rows is not None:
            _recode_stations(rows, scored)
    report["baseline"].update({
        "rows": baseline["rows"], "stations": len(baseline["stations"]),
        "days": int(np.unique(baseline["t"] // DAY_SEC).size),
        "duplicate_keys": baseline["duplicates"], "files": baseline["files"],
    })
    if candidate is not None:
        report["candidate"].update({
            "rows": candidate["rows"], "stations": len(candidate["stations"]),
            "days": int(np.unique(candidate["t"] // DAY_SEC).size),
            "duplicate_keys": candidate["duplicates"], "files": candidate["files"],
        })
    report["layer_b"] = layer_b(
        baseline, candidate, grid, leads,
        n_resamples=int(args.resamples),
        seed=int(args.seed),
        ci=float(args.ci),
        log=log,
    )



def _run_layer_c(
    report: dict, args: Any, leads: Sequence[int], thresholds: Sequence[int],
    stations: Sequence[str] | None, baseline_dirs: Sequence[Path],
    candidate_dirs: Sequence[Path] | None, log,
) -> None:
    column_for = column_template(args.probability_column)
    common = dict(
        far_cap=float(args.far_cap),
        plateau_frac=float(args.plateau_frac),
        min_warnings=int(args.min_warnings),
        fallback_threshold_pct=int(args.fallback_threshold_pct),
        coverage_gap_min=int(args.coverage_gap_min),
        tolerance_min=int(args.tolerance_min),
        dry_min=int(args.dry_min),
        onset_min_mm=float(args.onset_min_mm),
        min_useful_lead_min=float(args.min_useful_lead_min),
        persistence_obs=int(args.persistence_obs),
        rearm_after_min=int(args.rearm_after_min),
        workers=int(args.workers),
    )
    log("layer C: loading the baseline rows")
    base_folds = FoldSet(
        baseline_dirs, Path(args.corpus_dir), leads,
        stations=stations,
        dry_min=int(args.dry_min), onset_min_mm=float(args.onset_min_mm),
        min_known_slots=int(args.min_known_slots),
        coverage_gap_min=int(args.coverage_gap_min),
        column_for=column_for, log=log,
    )
    if "dead_gauges" not in report:
        report["dead_gauges"] = [{"station_id": s} for s in base_folds.dead]
    report["baseline"].setdefault("rows", base_folds.n_rows)
    report["baseline"].setdefault("stations", len(base_folds.stations))
    report["baseline"].setdefault("days", base_folds.n_days)
    base = layer_c(base_folds, base_folds.leads, thresholds, log=log, **common)
    report["layer_c"] = _strip_per_day(base)
    del base_folds

    if candidate_dirs is None:
        return
    log("layer C: loading the candidate rows")
    cand_folds = FoldSet(
        candidate_dirs, Path(args.corpus_dir), leads,
        stations=stations,
        dry_min=int(args.dry_min), onset_min_mm=float(args.onset_min_mm),
        min_known_slots=int(args.min_known_slots),
        coverage_gap_min=int(args.coverage_gap_min),
        column_for=_candidate_column_for(args), log=log,
    )
    report["candidate"].setdefault("rows", cand_folds.n_rows)
    report["candidate"].setdefault("stations", len(cand_folds.stations))
    report["candidate"].setdefault("days", cand_folds.n_days)
    cand = layer_c(cand_folds, cand_folds.leads, thresholds, log=log, **common)
    report["layer_c_candidate"] = _strip_per_day(cand)
    del cand_folds
    report["layer_c_difference"] = {} if int(args.resamples) <= 0 else {
        lead: paired_f1_ci(
            base["leads"][lead]["per_day"], cand["leads"][lead]["per_day"],
            n_resamples=max(1, int(args.resamples)),
            seed=int(args.seed) + 2,
            ci=float(args.ci),
        )
        for lead in base["leads"]
        if lead in cand["leads"]
    }


def _strip_per_day(payload: dict) -> dict:
    """Drop the per-day count blocks before the payload is serialised.

    They exist for the bootstrap; a year of them per lead is thousands of
    rows of intermediate arithmetic in a document meant to be read.
    """
    return {
        **payload,
        "leads": {
            lead: {k: v for k, v in entry.items() if k != "per_day"}
            for lead, entry in payload["leads"].items()
        },
    }


def _headline(report: Mapping[str, Any]) -> dict:
    """The few numbers worth printing to stdout at the end of a run."""
    out: dict[str, Any] = {
        "dead_gauges": [
            row.get("station_id") for row in (report.get("dead_gauges") or [])
        ],
        "parity_problems": report.get("parity_problems") or [],
    }
    layer_b_block = report.get("layer_b")
    if layer_b_block:
        out["layer_b"] = {
            lead: {
                "n": (strata[POOLED].get("baseline") or {}).get("n"),
                "bss": (strata[POOLED].get("baseline") or {}).get("bss"),
                "pr_auc": (strata[POOLED].get("baseline") or {}).get("pr_auc"),
                "difference": (strata[POOLED].get("difference") or {}).get("bss"),
            }
            for lead, strata in layer_b_block["leads"].items()
        }
    layer_c_block = report.get("layer_c")
    if layer_c_block:
        out["layer_c"] = {
            lead: {
                "f1": entry["out_of_fold"]["f1"],
                "precision": entry["out_of_fold"]["precision"],
                "recall": entry["out_of_fold"]["recall"],
                "thresholds": [f["threshold_pct"] for f in entry["folds"]],
            }
            for lead, entry in layer_c_block["leads"].items()
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
