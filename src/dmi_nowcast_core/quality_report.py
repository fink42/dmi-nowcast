"""The ``quality.json`` producer — "How good are we?" in one document.

The website's `/quality` page renders exactly one artifact: the document
this module builds. Its contract is
``frontend/src/lib/quality/schema.ts``, which is canonical — the producer
writes those fields and no others, the client invents nothing, and every
top-level section is nullable.

The rule that governs the whole module: **a missing input nulls its
section, it never fakes it**. There is no zero-filling, no "assume dry",
no carrying a number over from a neighbouring window. A page about
honesty that renders 0 % where it means "not measured" would be worse
than no page, so a number that cannot be computed from evidence present
on disk comes out as ``None`` and the page says so in words.

What goes in
------------

Five optional inputs, each nulling only what it feeds:

``radar_corpus``
    The national calibration corpus Parquet (``event_time``, ``point_id``,
    ``lead_min``, ``raw_prob``, ``outcome``, ``sample_weight``,
    ``frame_age_min``, …). Feeds ``windows.radar``, ``reliability.radar``
    and ``headline.reliability.radar``.
``station_corpus``
    The same shape widened by ``scripts/join_gauge_truth.py`` with
    ``gauge_outcome`` / ``gauge_mm`` / ``gauge_dur_min``. Feeds
    ``windows.gauge``, ``reliability.gauge``,
    ``headline.reliability.gauge`` and the per-station ``brier_gauge``.
``replay_dir``
    ``scripts/replay_warnings.py`` output — ``decisions/YYYY-MM-DD.parquet``
    plus ``summary.json``. The historical half of the warning scoreboard.
``corpus_dir``
    The corpus root: the gauge store under ``stations/obs/``, the live
    ``station_eval`` rows under ``stations/eval/``, the station catalogue
    and the station points file. The live half of the scoreboard, and the
    only source of gauge truth.
``persistence_json``
    ``scripts/persistence_vs_advection.py``'s ``results.json``. Feeds
    ``headline.persistence_margin`` and nothing else.

Reliability, and *whose* probability it is
------------------------------------------

The corpus stores ``raw_prob``, the RAW ensemble exceedance fraction. The
site does not serve that: it serves the isotonic-calibrated probability.
So when ``national_curves`` is given, every ``raw_prob`` is pushed through
the curve for its lead **before** binning, and the resulting curves
describe what the site actually said. Without curves the raw value is
binned and ``methods.reliability_probability`` says so, because a
reliability diagram of a number nobody was shown is a different claim.

The bins themselves are the ten fixed-width bins of
``sql/reliability_pooled.sql`` — ``bin = min(floor(p * 10), 9)`` — with
the same validity filter (finite probability, non-null outcome, finite
positive weight) and the same ``sample_weight`` weighting, so a number
here and a number from the SQL are the same number. ``tests/`` pins that
agreement against DuckDB on a small table.

Warnings, and one continuous table
----------------------------------

``scripts/replay_warnings.py`` (history) and the sidecar's
``station_eval`` step (live) append decision rows of exactly the same
shape (``warning_score.DECISION_COLUMNS``). This module concatenates both,
deduplicates on ``(radar_ts, station_id)`` with the LIVE row winning — the
live row is the decision the service really took, the replay's is a
reconstruction — and scores the union against the gauge store with
``warning_score.score_warnings``. ``window_days`` is therefore the number
of distinct UTC days that carry a decision row, which is a union of replay
days and live days and not a rolling "last N days"; the page's sentence
says "over D measured days" for that reason.

Everything is timezone-aware UTC end to end; conversion to a viewer's
clock is the browser's business.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .warning_score import (
    DEFAULT_COVERAGE_GAP_MIN,
    DEFAULT_DRY_MIN,
    DEFAULT_LEAD_MIN,
    DEFAULT_TOLERANCE_MIN,
    SLOT_MIN,
    WET_DUR_MIN,
    WET_PRECIP_MM,
    coverage_runs,
    gauge_slots,
    onsets,
    pooled_summary,
    raining_now_agreement,
    score_warnings,
    slot_end_of,
)

__all__ = [
    "SCHEMA_VERSION",
    "N_BINS",
    "DEFAULT_SERVED_LEADS",
    "RADAR_SOURCE",
    "GAUGE_SOURCE",
    "QualityInputs",
    "build_quality_report",
    "render_markdown",
    "validate_report",
    "reliability_from_corpus",
    "bin_statistics",
]

#: The contract version ``frontend/src/lib/quality/schema.ts`` pins.
SCHEMA_VERSION = 1

#: Ten fixed-width probability bins, as in ``sql/reliability_pooled.sql``.
N_BINS = 10

#: The leads the national products publish (``NationalConfig.leads_min``).
#: Only a fallback: when calibration curves are supplied, the leads they
#: cover ARE the served leads, because an uncalibrated lead is not one the
#: site quotes a probability for.
DEFAULT_SERVED_LEADS: tuple[int, ...] = (10, 20, 30, 45, 60)

#: Attribution, verbatim. Both datasets are DMI Open Data under CC BY 4.0.
RADAR_SOURCE = "DMI radar composites (CC BY 4.0)"
GAUGE_SOURCE = "DMI meteorological observations, metObs (CC BY 4.0)"

#: The gauge window is padded this far either side of a decision window so
#: a warning at the edge can still find its onset, and so the first slots
#: have the dry evidence the onset rule needs behind them.
GAUGE_PAD_MIN = 120

#: Per-station scores below this many warnings are reported as null: POD
#: and FAR over one or two warnings are noise with a decimal point.
MIN_STATION_WARNINGS = 3


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityInputs:
    """Everything the producer reads, and the rules it reads it under.

    Every path is optional. ``live_days`` bounds how far back the live
    ``stations/eval`` partitions are read; ``live_days_secondary`` is the
    shorter recency window the ``events`` list is drawn from, so the page's
    "recent warnings" table stays recent even when the replay covers a
    year.
    """

    radar_corpus: Path | None = None
    station_corpus: Path | None = None
    replay_dir: Path | None = None
    corpus_dir: Path | None = None
    persistence_json: Path | None = None
    national_curves: Path | None = None
    live_days: int = 90
    live_days_secondary: int = 30
    #: Overrides ``datetime.now(timezone.utc)``; tests pin it.
    now: datetime | None = None
    #: The lead the headline sentence quotes.
    headline_lead_min: int = 30
    #: The headline reads the highest bin at that lead carrying at least
    #: this many forecasts; failing that, the most populated bin whose
    #: lower edge is at or above ``headline_min_prob``.
    headline_min_n: int = 200
    headline_min_prob: float = 0.3
    #: The horizon ``headline.persistence_margin`` is taken at.
    persistence_horizon_min: int = 10
    #: Scoring rules. ``lead_min`` is the subscriber's promise; the rest
    #: are the onset definition, carried into ``methods``.
    lead_min: int = DEFAULT_LEAD_MIN
    tolerance_min: int = DEFAULT_TOLERANCE_MIN
    dry_min: int = DEFAULT_DRY_MIN
    raining_now_mm_h: float = 0.5
    #: The longest gap between consecutive decision rows that still counts
    #: as continuous coverage (two radar cycles).
    coverage_gap_min: int = DEFAULT_COVERAGE_GAP_MIN
    served_leads: tuple[int, ...] | None = None
    max_events: int = 20
    min_station_warnings: int = MIN_STATION_WARNINGS


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now(inputs: QualityInputs) -> datetime:
    return inputs.now or datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    """A UTC ISO-8601 stamp with a ``Z``, or None."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_ts(value: Any) -> datetime | None:
    """A timestamp from a corpus/Parquet/JSON field, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _round(value: float | None, places: int = 6) -> float | None:
    """Round for the wire; None and non-finite both become None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return round(out, places) if math.isfinite(out) else None


def _months_between(start: datetime, end: datetime) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        out.append((year, month))
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return out


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


def _bin_index(prob: "np.ndarray") -> "np.ndarray":
    """``LEAST(CAST(floor(p * 10) AS INTEGER), 9)``, in numpy.

    One function so there is exactly one place the bin edges live. The
    clamp at both ends matters: DuckDB's ``LEAST`` folds ``p == 1.0`` into
    bin 9, and a probability that arrives fractionally negative from a
    float round-trip must land in bin 0 rather than index ``-1``.
    """
    return np.clip(
        np.floor(prob * N_BINS).astype(np.int64), 0, N_BINS - 1,
    )


def bin_statistics(
    prob: Sequence[float] | "np.ndarray",
    outcome: Sequence[float] | "np.ndarray",
    weight: Sequence[float] | "np.ndarray",
) -> list[dict]:
    """The ten fixed bins of ``sql/reliability_pooled.sql``, weighted.

    ``bin k`` covers ``[k/10, (k+1)/10)`` with ``p == 1.0`` folded into
    bin 9 — the same arithmetic as the SQL, so a number here and a number
    from DuckDB are the same number (``tests/test_quality_report.py`` pins
    that). An empty bin keeps its edges and reports ``None`` for both
    means: a gap in the curve, not a point at zero.

    Caller filters validity; this function assumes finite inputs and
    positive weights.
    """
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    index = _bin_index(p)
    counts = np.bincount(index, minlength=N_BINS)
    sum_w = np.bincount(index, weights=w, minlength=N_BINS)
    sum_w2 = np.bincount(index, weights=w * w, minlength=N_BINS)
    sum_wp = np.bincount(index, weights=w * p, minlength=N_BINS)
    sum_wy = np.bincount(index, weights=w * y, minlength=N_BINS)
    out: list[dict] = []
    for k in range(N_BINS):
        empty = counts[k] == 0 or sum_w[k] <= 0.0
        out.append({
            "lo": round(k / N_BINS, 6),
            "hi": round((k + 1) / N_BINS, 6),
            "forecast_mean": None if empty else _round(sum_wp[k] / sum_w[k]),
            "observed_freq": None if empty else _round(sum_wy[k] / sum_w[k]),
            "n": int(counts[k]),
            "eff_n": 0.0 if empty else _round(
                (sum_w[k] * sum_w[k]) / sum_w2[k],
            ),
        })
    return out


def _weighted_brier(
    prob: "np.ndarray", outcome: "np.ndarray", weight: "np.ndarray",
) -> float | None:
    """``Σw(p−y)² / Σw`` — ``brier_exact`` in ``sql/brier_decomposition.sql``."""
    total = float(weight.sum())
    if total <= 0.0:
        return None
    diff = prob - outcome
    return float((weight * diff * diff).sum() / total)


def _kish(weight: "np.ndarray") -> float:
    """Kish effective sample size ``(Σw)²/Σw²``."""
    total = float(weight.sum())
    squares = float((weight * weight).sum())
    return (total * total / squares) if squares > 0 else 0.0


def _read_corpus_valid(path: Path, outcome_column: str):
    """Read a corpus Parquet, keeping only the SQL's valid rows.

    The filter is ``sql/reliability_pooled.sql``'s ``valid`` CTE exactly:
    finite ``raw_prob``, non-null outcome, finite positive
    ``sample_weight``. Applying it in Arrow rather than in Python is what
    keeps a multi-million-row corpus a few seconds rather than a few
    minutes.

    pyarrow is imported here rather than at module scope: the core package
    lists it as a dev dependency, and this module must stay importable
    without it so ``render_markdown`` and ``validate_report`` work
    anywhere.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    wanted = [
        "event_time", "point_id", "lead_min", "raw_prob", outcome_column,
        "sample_weight", "frame_age_min", "threshold_mm_h",
    ]
    available = set(pq.ParquetFile(path).schema_arrow.names)
    table = pq.read_table(path, columns=[c for c in wanted if c in available])
    if outcome_column not in available or "raw_prob" not in available:
        return None
    prob = pc.cast(table.column("raw_prob"), "float64")
    weight = pc.cast(table.column("sample_weight"), "float64")
    mask = pc.and_kleene(
        pc.and_kleene(pc.is_finite(prob), pc.is_valid(table.column(outcome_column))),
        pc.and_kleene(pc.is_finite(weight), pc.greater(weight, 0.0)),
    )
    return table.filter(pc.fill_null(mask, False))


def _load_curves(path: Path | None) -> dict[int, Any]:
    """``{lead: IsotonicCalibrator}`` from the served curve file, or ``{}``."""
    if path is None:
        return {}
    from .calibrate import load_calibration_curves

    return load_calibration_curves(Path(path))


def _served_leads(
    inputs: QualityInputs, corpus_leads: Iterable[int], curves: Mapping[int, Any],
) -> list[int]:
    """The leads to publish a curve for.

    The calibration curves define the served set when present: a lead with
    no curve is a lead the site quotes no calibrated probability for. The
    national default is the fallback, and if neither intersects the corpus
    (a corpus built at other leads) every corpus lead is published rather
    than nothing.
    """
    present = sorted({int(lead) for lead in corpus_leads})
    if inputs.served_leads is not None:
        wanted = [lead for lead in sorted(inputs.served_leads) if lead in present]
        return wanted or present
    if curves:
        wanted = [lead for lead in present if lead in curves]
        if wanted:
            return wanted
    wanted = [lead for lead in present if lead in DEFAULT_SERVED_LEADS]
    return wanted or present


def _fold_labels(values: Sequence[Any]) -> tuple[list[str], list[str]]:
    """``(month, day)`` labels for a column of event timestamps.

    String slicing rather than datetime parsing: the corpus stores
    ``event_time`` as an ISO string, and over a few million rows the
    difference is seconds against minutes. Anything that is not an ISO
    string falls back to parsing, and anything unreadable gets the empty
    label — which groups with nothing and is dropped from the folds.
    """
    months: list[str] = []
    days: list[str] = []
    for value in values:
        if isinstance(value, str) and len(value) >= 10 and value[4] == "-":
            months.append(value[:7])
            days.append(value[:10])
            continue
        stamp = _parse_ts(value)
        if stamp is None:
            months.append("")
            days.append("")
            continue
        months.append(f"{stamp.year:04d}-{stamp.month:02d}")
        days.append(stamp.date().isoformat())
    return months, days


def _out_of_fold_calibrated(
    prob: "np.ndarray",
    outcome: "np.ndarray",
    weight: "np.ndarray",
    folds: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray", int] | None:
    """Leave-one-fold-out isotonic calibration: ``(values, mask, n_folds)``.

    For each fold, the calibrator is fitted on **every other fold's** rows
    with the same weighted fitter the served curves come from
    (:func:`dmi_nowcast_core.calibrate.fit_isotonic_weighted`) and applied
    to the held-out fold. Pooling the held-out predictions gives a
    reliability diagram of a calibration that never saw the row it is
    grading.

    This is not a refinement, it is the difference between a measurement
    and a tautology. The served curves were fitted ON this corpus, so
    binning their output against it draws a perfect diagonal by
    construction — the first real report had ``forecast_mean ==
    observed_freq`` to three decimals in every bin, which says only that
    isotonic regression can describe its own training data.

    ``mask`` marks the rows that got a prediction; a fold whose training
    set is degenerate (one distinct probability, no positive weight) is
    dropped rather than silently left raw, because a diagram mixing
    calibrated and uncalibrated values is neither. ``None`` when there is
    only one fold and cross-validation is impossible.
    """
    from .calibrate import fit_isotonic_weighted

    labels = [f for f in dict.fromkeys(folds.tolist()) if f]
    if len(labels) < 2:
        return None
    out = np.array(prob, dtype=np.float64, copy=True)
    ok = np.zeros(prob.shape, dtype=bool)
    used = 0
    for label in labels:
        test = folds == label
        train = (~test) & (folds != "")
        if not test.any() or not train.any():
            continue
        try:
            calibrator = fit_isotonic_weighted(
                prob[train], outcome[train], weight[train],
            )
        except ValueError:
            continue
        out[test] = np.asarray(calibrator.predict(prob[test]), dtype=np.float64)
        ok |= test
        used += 1
    if used < 2 or not ok.any():
        return None
    return out, ok, used


def reliability_from_corpus(
    path: Path,
    *,
    outcome_column: str = "outcome",
    curves: Mapping[int, Any] | None = None,
    inputs: QualityInputs | None = None,
    calibration: str = "served",
    per_point: bool = False,
) -> dict:
    """Reliability curves plus the window and totals of one corpus.

    ``calibration`` picks which probability the diagram is OF, and the
    choice is different for the two truths:

    ``"cv"``
        Leave-one-month-out cross-validation (:func:`_out_of_fold_calibrated`).
        This is what the RADAR corpus needs: the served curves were fitted
        on it, so grading them against it measures nothing.
    ``"served"``
        The live curves, applied as the service applies them. Honest for
        the GAUGE corpus, whose truth column the fit never saw — the
        gauge is an independent instrument, so "the probability we
        published against what the ground recorded" is a real claim.
    ``"raw"``
        The raw ensemble exceedance fraction, uncalibrated.

    Returns ``{"curves": [...], "window": {...}, "frame_age": (lo, hi),
    "threshold_mm_h": float|None, "per_point_brier": {...}, "mode": str,
    "cv_folds": int, "fold": str|None}``. ``curves`` is empty when nothing
    in the corpus is usable, which the caller reads as "null this
    section". Each curve carries ``brier_raw`` alongside ``brier`` — the
    same rows scored without any calibration, so the improvement is a
    paired comparison rather than two numbers over different samples.
    """
    inputs = inputs or QualityInputs()
    curves = dict(curves or {})
    empty = {
        "curves": [], "window": None, "frame_age": None,
        "threshold_mm_h": None, "per_point_brier": {},
        "mode": "none", "cv_folds": 0, "fold": None,
    }
    table = _read_corpus_valid(Path(path), outcome_column)
    if table is None or table.num_rows == 0:
        return empty

    names = set(table.schema.names)
    prob_all = np.asarray(table.column("raw_prob").to_numpy(zero_copy_only=False),
                          dtype=np.float64)
    y_all = np.asarray(table.column(outcome_column).to_numpy(zero_copy_only=False),
                       dtype=np.float64)
    w_all = np.asarray(table.column("sample_weight").to_numpy(zero_copy_only=False),
                       dtype=np.float64)
    lead_all = np.asarray(table.column("lead_min").to_numpy(zero_copy_only=False),
                          dtype=np.int64)
    point_all = (
        table.column("point_id").to_pylist() if "point_id" in names
        else [""] * table.num_rows
    )

    # --- window, folds, frame age, threshold — over the valid rows --------
    stamps: list[datetime] = []
    events: set[str] = set()
    fold_all = np.array([""] * table.num_rows, dtype=object)
    fold_name: str | None = None
    if "event_time" in names:
        raw_times = table.column("event_time").to_pylist()
        for raw_stamp in raw_times:
            stamp = _parse_ts(raw_stamp)
            if stamp is not None:
                stamps.append(stamp)
                events.add(stamp.isoformat())
        months, days = _fold_labels(raw_times)
        # Months are the natural fold: a month of radar is many weather
        # regimes, and holding one out leaves plenty to fit on. A corpus
        # spanning less than two months falls back to days rather than
        # giving up on cross-validation entirely.
        if len({m for m in months if m}) >= 2:
            fold_all = np.array(months, dtype=object)
            fold_name = "month"
        elif len({d for d in days if d}) >= 2:
            fold_all = np.array(days, dtype=object)
            fold_name = "day"
    frame_age: tuple[float, float] | None = None
    if "frame_age_min" in names:
        ages = np.asarray(
            table.column("frame_age_min").to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        ages = ages[np.isfinite(ages)]
        if ages.size:
            frame_age = (float(ages.min()), float(ages.max()))
    threshold: float | None = None
    if "threshold_mm_h" in names and table.num_rows:
        first = table.column("threshold_mm_h")[0].as_py()
        if first is not None:
            threshold = float(first)

    leads = _served_leads(inputs, np.unique(lead_all).tolist(), curves)
    out_curves: list[dict] = []
    per_point_brier: dict[int, dict[str, float]] = {}
    modes: set[str] = set()
    cv_folds = 0
    for lead in leads:
        keep = lead_all == lead
        if not keep.any():
            continue
        raw_prob = prob_all[keep]
        y = y_all[keep]
        w = w_all[keep]
        rows = np.flatnonzero(keep)
        mode = calibration
        if calibration == "cv":
            folded = _out_of_fold_calibrated(raw_prob, y, w, fold_all[keep])
            if folded is None:
                # One fold is no fold. Raw is the only thing left that is
                # not a claim about the fit's own training data.
                probs, mode = raw_prob, "raw"
            else:
                probs, ok, used = folded
                probs, raw_prob = probs[ok], raw_prob[ok]
                y, w, rows = y[ok], w[ok], rows[ok]
                cv_folds = max(cv_folds, used)
        elif calibration == "served":
            curve = curves.get(lead)
            if curve is None:
                probs, mode = raw_prob, "raw"
            else:
                probs = np.asarray(curve.predict(raw_prob), dtype=np.float64)
        else:
            probs, mode = raw_prob, "raw"
        brier = _weighted_brier(probs, y, w)
        if brier is None:
            continue
        modes.add(mode)
        out_curves.append({
            "lead_min": int(lead),
            "brier": _round(brier),
            # Additive: the same rows with no calibration at all, so the
            # methods block can state the improvement as a paired number.
            "brier_raw": _round(_weighted_brier(raw_prob, y, w)),
            "n": int(probs.size),
            "eff_n": _round(_kish(w)),
            "bins": bin_statistics(probs, y, w),
        })
        if not per_point:
            continue
        # Per-point Brier, for the station map. Same weights, same probs.
        acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        squares = w * (probs - y) * (probs - y)
        for offset, row in enumerate(rows):
            cell = acc[str(point_all[row])]
            cell[0] += float(w[offset])
            cell[1] += float(squares[offset])
        per_point_brier[int(lead)] = {
            pid: (v[1] / v[0]) for pid, v in acc.items() if v[0] > 0
        }

    return {
        "curves": out_curves,
        "window": {
            "from": _iso(min(stamps)) if stamps else None,
            "to": _iso(max(stamps)) if stamps else None,
            "events": len(events),
            "rows": int(table.num_rows),
            "points": len({str(p) for p in point_all}),
        },
        "frame_age": frame_age,
        "threshold_mm_h": threshold,
        "per_point_brier": per_point_brier,
        "mode": "mixed" if len(modes) > 1 else (modes.pop() if modes else "none"),
        "cv_folds": cv_folds,
        "fold": fold_name if cv_folds else None,
    }
def _closest_lead(curves: Sequence[dict], wanted: int) -> dict | None:
    """The served curve nearest ``wanted`` (ties go to the lower lead)."""
    if not curves:
        return None
    return min(curves, key=lambda c: (abs(int(c["lead_min"]) - wanted), int(c["lead_min"])))


def _headline_bin(curve: Mapping[str, Any], inputs: QualityInputs) -> dict | None:
    """The bin the headline sentence should be read from.

    The obvious choice — "the bin containing 70 %" — assumes the service
    ever says 70 %. Calibration is a shrinking operation, and on this
    corpus the calibrated probability tops out near 0.62 at lead 20 and
    0.56 at lead 30: the 0.7 bin is EMPTY, and the card came out null
    while the numbers behind it were perfectly good. So the rule is
    "the most confident thing we actually say, often enough to mean
    something": the HIGHEST bin carrying at least ``headline_min_n``
    forecasts.

    The fallback, when no bin clears that bar, is the most populated bin
    above 0.3 — still a claim about confident forecasts, just one whose
    sample size the page will have to caveat. Below 0.3 there is no
    sentence worth writing, so that returns None and the card says "not
    measured".
    """
    bins = curve.get("bins") or []
    usable = [
        (index, b) for index, b in enumerate(bins)
        if b.get("forecast_mean") is not None and b.get("observed_freq") is not None
    ]
    confident = [(i, b) for i, b in usable if b["n"] >= inputs.headline_min_n]
    if confident:
        return max(confident, key=lambda pair: pair[0])[1]
    upper = [(i, b) for i, b in usable if b["lo"] >= inputs.headline_min_prob]
    if upper:
        return max(upper, key=lambda pair: (pair[1]["n"], pair[0]))[1]
    return None


def _headline_reliability(curves: Sequence[dict], inputs: QualityInputs) -> dict | None:
    """"When we say X %, it rains Y % of the time", at one lead.

    Null when no bin at that lead is both populated and confident enough
    to make a sentence out of — see :func:`_headline_bin`.
    """
    curve = _closest_lead(curves, inputs.headline_lead_min)
    if curve is None:
        return None
    binned = _headline_bin(curve, inputs)
    if binned is None:
        return None
    return {
        "lead_min": int(curve["lead_min"]),
        "said_pct": _round(binned["forecast_mean"] * 100.0, 3),
        "happened_pct": _round(binned["observed_freq"] * 100.0, 3),
        "n": int(binned["n"]),
        # Additive: which bin the two percentages came from. The page
        # renders the sentence; the archive should be able to check it.
        "bin": [binned["lo"], binned["hi"]],
    }


# ---------------------------------------------------------------------------
# Decision rows: replay history + live evaluation, one table
# ---------------------------------------------------------------------------


def _read_decision_parquet(path: Path) -> list[dict]:
    from .warning_score import decision_schema
    import pyarrow.parquet as pq

    return pq.read_table(path, schema=decision_schema()).to_pylist()


def _load_decisions(inputs: QualityInputs) -> tuple[list[dict], dict[str, int]]:
    """Replay + live decision rows, deduplicated on ``(radar_ts, station_id)``.

    The live row wins: it is the decision the running service actually
    took, while the replay's is a reconstruction of what it would have
    taken. Where both cover a frame the live one is the record.
    """
    counts = {"replay": 0, "live": 0, "duplicates": 0}
    merged: dict[tuple[Any, str], dict] = {}

    if inputs.replay_dir is not None:
        decisions_dir = Path(inputs.replay_dir) / "decisions"
        if decisions_dir.is_dir():
            for path in sorted(decisions_dir.glob("*.parquet")):
                try:
                    rows = _read_decision_parquet(path)
                except Exception:  # noqa: BLE001 — one unreadable day
                    continue
                counts["replay"] += len(rows)
                for row in rows:
                    key = (row.get("radar_ts"), str(row.get("station_id")))
                    if key in merged:
                        counts["duplicates"] += 1
                    merged[key] = row

    if inputs.corpus_dir is not None:
        eval_dir = Path(inputs.corpus_dir) / "stations" / "eval"
        cutoff = _now(inputs) - timedelta(days=max(1, inputs.live_days))
        if eval_dir.is_dir():
            for (year, month) in _months_between(cutoff, _now(inputs)):
                path = eval_dir / f"{year:04d}" / f"{month:02d}.parquet"
                if not path.is_file():
                    continue
                try:
                    rows = _read_decision_parquet(path)
                except Exception:  # noqa: BLE001
                    continue
                for row in rows:
                    stamp = _parse_ts(row.get("generated_at"))
                    if stamp is not None and stamp < cutoff:
                        continue
                    counts["live"] += 1
                    key = (row.get("radar_ts"), str(row.get("station_id")))
                    if key in merged:
                        counts["duplicates"] += 1
                    merged[key] = row

    rows = sorted(
        merged.values(),
        key=lambda r: (_parse_ts(r.get("generated_at")) or datetime.min.replace(
            tzinfo=timezone.utc), str(r.get("station_id"))),
    )
    return rows, counts


def _live_window(inputs: QualityInputs) -> dict | None:
    """``windows.live`` — what the live ``stations/eval`` rows actually cover.

    Not the requested lookback: the window the rows are in. Null when
    there are none, because "0 days of live evidence" and "we did not
    look" must not render the same.
    """
    if inputs.corpus_dir is None:
        return None
    eval_dir = Path(inputs.corpus_dir) / "stations" / "eval"
    if not eval_dir.is_dir():
        return None
    cutoff = _now(inputs) - timedelta(days=max(1, inputs.live_days))
    stamps: list[datetime] = []
    days: set[date] = set()
    for (year, month) in _months_between(cutoff, _now(inputs)):
        path = eval_dir / f"{year:04d}" / f"{month:02d}.parquet"
        if not path.is_file():
            continue
        try:
            rows = _read_decision_parquet(path)
        except Exception:  # noqa: BLE001
            continue
        for row in rows:
            stamp = _parse_ts(row.get("generated_at"))
            if stamp is None or stamp < cutoff:
                continue
            stamps.append(stamp)
            days.add(stamp.date())
    if not stamps:
        return None
    return {
        "days": len(days),
        "from": _iso(min(stamps)),
        "to": _iso(max(stamps)),
    }


# ---------------------------------------------------------------------------
# Gauge truth: onsets and the wet/dry slot behind every decision
# ---------------------------------------------------------------------------


@dataclass
class _GaugeTruth:
    """Onsets per station, the wet flag of each decision's slot, and the edge.

    ``known_until`` is the last slot end each station actually reported.
    It is what keeps a warning sent ninety seconds ago from being graded
    as a false alarm by a report built ninety seconds later — see
    ``warning_score.score_warnings``.
    """

    onsets: dict[str, list[datetime]] = field(default_factory=dict)
    wet_at: dict[tuple[str, datetime], bool | None] = field(default_factory=dict)
    known_until: dict[str, datetime] = field(default_factory=dict)
    known_slots: int = 0


def _gauge_truth(
    corpus_dir: Path,
    station_ids: Sequence[str],
    window: tuple[datetime, datetime],
    needed: set[tuple[str, datetime]],
    *,
    dry_min: int,
) -> _GaugeTruth:
    """Read the gauge store month by month; return onsets and slot flags.

    Month-sized windows, padded either side, are what keeps this bounded:
    the full slot grid for ~100 stations over a year would be tens of
    millions of tuples, and only two things are ever needed from it —
    the onsets, and the wet flag at each decision's own instant. Onsets
    found in overlapping windows are deduplicated by instant, exactly as
    ``replay_warnings.score`` does across its day windows.
    """
    from .station_store import StationObsStore
    from .warning_score import PRECIP_DUR_PARAM, PRECIP_PARAM

    store = StationObsStore(Path(corpus_dir))
    truth = _GaugeTruth()
    onset_sets: dict[str, set[datetime]] = defaultdict(set)
    last_known: dict[str, datetime] = {}
    pad = timedelta(minutes=GAUGE_PAD_MIN)
    start, end = window
    for (year, month) in _months_between(start, end):
        month_start = datetime(year, month, 1, tzinfo=timezone.utc) - pad
        if month == 12:
            month_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) + pad
        else:
            month_end = datetime(year, month + 1, 1, tzinfo=timezone.utc) + pad
        try:
            table = store.read(
                month_start, month_end,
                [PRECIP_PARAM, PRECIP_DUR_PARAM], list(station_ids),
            )
        except Exception:  # noqa: BLE001 — one unreadable month
            continue
        for station in station_ids:
            slots = gauge_slots(
                table, station, start_utc=month_start, end_utc=month_end,
            )
            if not slots:
                continue
            onset_sets[station].update(onsets(slots, dry_min))
            for stamp, wet in slots:
                if wet is None:
                    continue
                truth.known_slots += 1
                previous = last_known.get(station)
                if previous is None or stamp > previous:
                    last_known[station] = stamp
                key = (station, stamp)
                if key in needed:
                    truth.wet_at[key] = wet
    truth.onsets = {sid: sorted(values) for sid, values in onset_sets.items()}
    truth.known_until = last_known
    return truth


# ---------------------------------------------------------------------------
# The warning scoreboard
# ---------------------------------------------------------------------------


@dataclass
class _Scoreboard:
    """Everything derived from the decision rows and the gauge behind them."""

    headline: dict | None = None
    raining_now: dict | None = None
    per_station: dict[str, dict] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    window_days: int = 0


def _score_decisions(
    rows: Sequence[dict], inputs: QualityInputs,
) -> _Scoreboard:
    """Score every decision row against the gauge store.

    Null in, null out: without a corpus directory there is no gauge, and
    without a gauge there is no truth to score against, so the whole
    scoreboard stays empty rather than grading the radar against itself.
    """
    board = _Scoreboard()
    if not rows or inputs.corpus_dir is None:
        return board

    stamps = [s for s in (_parse_ts(r.get("generated_at")) for r in rows) if s]
    if not stamps:
        return board
    board.window_days = len({s.date() for s in stamps})
    station_ids = sorted({str(r.get("station_id")) for r in rows})

    needed = {
        (str(r.get("station_id")), slot_end_of(stamp))
        for r, stamp in (
            (r, _parse_ts(r.get("generated_at"))) for r in rows
        )
        if stamp is not None
    }
    truth = _gauge_truth(
        Path(inputs.corpus_dir), station_ids, (min(stamps), max(stamps)),
        needed, dry_min=inputs.dry_min,
    )
    if truth.known_slots == 0:
        return board

    warnings_by_station: dict[str, list[tuple[datetime, float | None]]] = defaultdict(list)
    frames_by_station: dict[str, list[datetime]] = defaultdict(list)
    p_rain_at: dict[tuple[str, datetime], float | None] = {}
    for row in rows:
        stamp = _parse_ts(row.get("generated_at"))
        if stamp is None:
            continue
        station = str(row.get("station_id"))
        frame = _parse_ts(row.get("radar_ts")) or stamp
        frames_by_station[station].append(frame)
        if row.get("action") != "notify":
            continue
        eta = row.get("eta_min")
        warnings_by_station[station].append(
            (stamp, None if eta is None else float(eta)),
        )
        p_rain_at[(station, stamp)] = (
            None if row.get("p_rain") is None else float(row["p_rain"])
        )

    # An onset is only a miss where a decision could have caught it. The
    # gauge archive is backfilled months deep; the decision rows cover the
    # frames the service actually evaluated, which on a fresh install is
    # one day. Without this the first live report counted every rain event
    # since the backfill as a miss — 2 088 of them against five warnings.
    coverage_by_station = {
        station: coverage_runs(
            stamps,
            max_gap_min=inputs.coverage_gap_min,
            extend_min=inputs.lead_min + inputs.tolerance_min,
        )
        for station, stamps in frames_by_station.items()
    }
    results = {
        station: score_warnings(
            warnings_by_station.get(station, []),
            truth.onsets.get(station, []),
            lead_min=inputs.lead_min,
            tolerance_min=inputs.tolerance_min,
            dry_min=inputs.dry_min,
            known_until=truth.known_until.get(station),
            coverage=coverage_by_station.get(station, []),
        )
        for station in station_ids
    }
    pooled = pooled_summary(
        results.values(),
        lead_min=inputs.lead_min,
        tolerance_min=inputs.tolerance_min,
        dry_min=inputs.dry_min,
    )
    spread = pooled["lead_error_min"]
    if (
        pooled["pod"] is not None
        and pooled["far"] is not None
        and spread.get("p50") is not None
    ):
        board.headline = {
            "window_days": int(board.window_days),
            "n_stations": len(station_ids),
            # ``warnings`` is the SCORED count: hits + false_alarms adds up
            # to it in the page's sentence. Warnings whose window has not
            # closed yet are counted separately (additive keys; the client
            # ignores them) rather than being graded early.
            "warnings": int(pooled["warnings"]),
            "pending": int(pooled["pending"]),
            "n_sent": int(pooled["n_sent"]),
            # Onsets outside every decision run: rain that fell while the
            # service was not watching. Reported, never scored.
            "uncovered_onsets": int(pooled["uncovered_onsets"]),
            "hits": int(pooled["hits"]),
            "false_alarms": int(pooled["false_alarms"]),
            "misses": int(pooled["misses"]),
            "pod": _round(pooled["pod"]),
            "far": _round(pooled["far"]),
            "lead_error_min": {
                "p25": _round(spread["p25"], 3),
                "p50": _round(spread["p50"], 3),
                "p75": _round(spread["p75"], 3),
            },
        }

    # --- "is it raining now" ---------------------------------------------
    resolved = []
    for row in rows:
        stamp = _parse_ts(row.get("generated_at"))
        if stamp is None:
            continue
        key = (str(row.get("station_id")), slot_end_of(stamp))
        if key not in truth.wet_at:
            continue
        resolved.append({**row, "gauge_wet": truth.wet_at[key]})
    agreement = raining_now_agreement(
        resolved, threshold_mm_h=inputs.raining_now_mm_h,
    )
    forecast = agreement["forecast_now"]
    observed = agreement["observed"]
    if (
        forecast["agreement"] is not None
        and forecast["pod"] is not None
        and forecast["far"] is not None
        and observed["agreement"] is not None
    ):
        board.raining_now = {
            "n_slots": int(agreement["n_scored"]),
            "agreement": _round(forecast["agreement"]),
            "pod": _round(forecast["pod"]),
            "far": _round(forecast["far"]),
            "observation_agreement": _round(observed["agreement"]),
            "from": _iso(min(stamps)),
            "to": _iso(max(stamps)),
        }

    # --- per station ------------------------------------------------------
    rows_by_station: dict[str, list[dict]] = defaultdict(list)
    for row in resolved:
        rows_by_station[str(row.get("station_id"))].append(row)
    counts_by_station: dict[str, int] = defaultdict(int)
    for row in rows:
        counts_by_station[str(row.get("station_id"))] += 1
    for station in station_ids:
        result = results[station]
        summary = result.summary
        enough = summary["warnings"] >= inputs.min_station_warnings
        station_now = raining_now_agreement(
            rows_by_station.get(station, []),
            threshold_mm_h=inputs.raining_now_mm_h,
        )
        board.per_station[station] = {
            "n_events": counts_by_station[station],
            "warnings": int(summary["warnings"]),
            "warn_pod": _round(summary["pod"]) if enough else None,
            "warn_far": _round(summary["far"]) if enough else None,
            "raining_now_agreement": _round(
                station_now["forecast_now"]["agreement"],
            ),
        }

    # --- the newest warnings ---------------------------------------------
    cutoff = _now(inputs) - timedelta(days=max(1, inputs.live_days_secondary))
    events: list[dict] = []
    for station, result in results.items():
        for warning in result.warnings:
            if warning.sent_utc < cutoff:
                continue
            if warning.outcome == "pending":
                # Its window has not closed. The schema's outcome enum is
                # hit|false_alarm, and there is no honest third answer to
                # put in the table yet.
                continue
            probability = p_rain_at.get((station, warning.sent_utc))
            if warning.eta_min is None or probability is None:
                # The client drops an event missing either number, so
                # emitting one would only make the list look longer than
                # the page renders. Off-coverage rows land here.
                continue
            events.append({
                "station_id": station,
                "warned_at_utc": _iso(warning.sent_utc),
                "eta_min": _round(warning.eta_min, 3),
                "p_rain": _round(probability),
                "gauge_onset_utc": _iso(warning.onset_utc),
                "outcome": warning.outcome,
                "lead_error_min": _round(warning.lead_error_min, 3),
                "_sent": warning.sent_utc,
            })
    events.sort(key=lambda e: (e["_sent"], e["station_id"]), reverse=True)
    board.events = events[: inputs.max_events]
    return board


# ---------------------------------------------------------------------------
# Stations: geometry, names, and the numbers hung on them
# ---------------------------------------------------------------------------


def _station_geometry(inputs: QualityInputs) -> dict[str, dict]:
    """``{station_id: {lat, lon, name, kind}}`` from whatever is on disk.

    The points file is the authority on which stations are in the
    benchmark; the catalogue supplies the human name and DMI's station
    class. Either can be missing — a station with coordinates and no name
    keeps its id as its name, which the map can still label.
    """
    out: dict[str, dict] = {}
    if inputs.corpus_dir is None:
        return out
    points_file = Path(inputs.corpus_dir) / "stations" / "station_points.json"
    if points_file.is_file():
        try:
            raw = json.loads(points_file.read_text())
        except (OSError, json.JSONDecodeError):
            raw = {}
        for entry in (raw.get("points") or ()):
            try:
                out[str(entry["id"])] = {
                    "lat": float(entry["lat"]),
                    "lon": float(entry["lon"]),
                    "name": str(entry["id"]),
                    "kind": str((entry.get("strata") or {}).get("station_kind") or ""),
                }
            except (KeyError, TypeError, ValueError):
                continue
    catalogue = Path(inputs.corpus_dir) / "stations" / "catalogue.parquet"
    if catalogue.is_file():
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(
                catalogue, columns=["station_id", "name", "kind", "lat", "lon"],
            )
            for row in table.to_pylist():
                sid = str(row["station_id"])
                entry = out.setdefault(sid, {
                    "lat": row.get("lat"), "lon": row.get("lon"),
                    "name": sid, "kind": "",
                })
                if row.get("name"):
                    entry["name"] = str(row["name"])
                if row.get("kind"):
                    entry["kind"] = str(row["kind"])
                if entry.get("lat") is None and row.get("lat") is not None:
                    entry["lat"] = float(row["lat"])
                if entry.get("lon") is None and row.get("lon") is not None:
                    entry["lon"] = float(row["lon"])
        except Exception:  # noqa: BLE001 — a catalogue is a nicety, not a need
            pass
    return out


def _stations_section(
    board: _Scoreboard,
    geometry: Mapping[str, dict],
    gauge_brier: Mapping[str, float],
) -> dict | None:
    """GeoJSON of the benchmark stations with their scores.

    A station without coordinates is dropped: the map is the only place
    these features are rendered, and a point with no position is not a
    feature. A station with coordinates but no scores is kept — "measured
    here, nothing conclusive yet" is a fact the map should show.
    """
    features: list[dict] = []
    for station in sorted(set(board.per_station) | set(gauge_brier)):
        geo = geometry.get(station)
        if geo is None or geo.get("lat") is None or geo.get("lon") is None:
            continue
        scores = board.per_station.get(station, {})
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    _round(geo["lon"], 5), _round(geo["lat"], 5),
                ],
            },
            "properties": {
                "station_id": station,
                "name": geo.get("name") or station,
                "kind": geo.get("kind") or "",
                "n_events": int(scores.get("n_events", 0)),
                "brier_gauge": _round(gauge_brier.get(station)),
                "warn_pod": scores.get("warn_pod"),
                "warn_far": scores.get("warn_far"),
                "warnings": int(scores.get("warnings", 0)),
                "raining_now_agreement": scores.get("raining_now_agreement"),
            },
        })
    if not features:
        return None
    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Persistence margin
# ---------------------------------------------------------------------------


def _persistence_margin(inputs: QualityInputs) -> dict | None:
    """``headline.persistence_margin`` from the advection study's results.

    Reads the pooled block at the requested horizon. Anything missing —
    an absent file, a horizon the study did not run, a CSI that failed to
    compute — nulls the section; "we are 0.00 better than nothing" is not
    a claim this file should be able to make by accident.
    """
    if inputs.persistence_json is None:
        return None
    path = Path(inputs.persistence_json)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    pooled = ((raw.get("aggregate") or {}).get("pooled") or {})
    horizon = str(inputs.persistence_horizon_min)
    block = pooled.get(horizon) or pooled.get(int(horizon))  # type: ignore[arg-type]
    if not isinstance(block, dict):
        return None
    advection = (block.get("advection") or {}).get("CSI")
    persistence = (block.get("persistence") or {}).get("CSI")
    if advection is None or persistence is None:
        return None
    meta = raw.get("meta") or {}
    days = sorted(str(d) for d in (meta.get("day_list") or ()))
    frames = meta.get("cases (frames)")
    if frames is None:
        frames = block.get("n_cases")
    if frames is None or not days:
        return None
    return {
        "horizon_min": int(inputs.persistence_horizon_min),
        "csi_advection": _round(advection),
        "csi_persistence": _round(persistence),
        "frames": int(frames),
        "from": _iso(_parse_ts(days[0])),
        "to": _iso(_parse_ts(days[-1])),
    }


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def _replay_summary(inputs: QualityInputs) -> dict:
    if inputs.replay_dir is None:
        return {}
    path = Path(inputs.replay_dir) / "summary.json"
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


#: How each truth's probability was produced, in one clause per mode.
_MODE_WORDS = {
    "cv": "calibrated out-of-sample, leave-one-{fold}-out CV",
    "served": "the served curves",
    "raw": "the raw ensemble exceedance fraction, uncalibrated",
    "mixed": "a mix of calibrated and raw leads",
    "none": "not measured",
}


def _mode_phrase(block: Mapping[str, Any] | None) -> str | None:
    """The clause describing one truth's probability, or None if absent."""
    if not block or not block.get("curves"):
        return None
    mode = str(block.get("mode") or "none")
    return _MODE_WORDS.get(mode, mode).format(fold=block.get("fold") or "month")


def _brier_improvement(
    curves: Sequence[Mapping[str, Any]], lead: int,
) -> str | None:
    """"0.0249 raw -> 0.0241 calibrated (-3.2 %)" at the headline lead.

    Paired: ``brier_raw`` is the same rows scored without calibration, so
    the difference is what the calibration did rather than what two
    different samples happened to contain.
    """
    curve = _closest_lead(list(curves), lead)
    if curve is None:
        return None
    after = curve.get("brier")
    before = curve.get("brier_raw")
    if after is None or before is None or before <= 0:
        return None
    change = (after - before) / before * 100.0
    return (
        f"radar Brier at {int(curve['lead_min'])} min: {before:.4f} raw → "
        f"{after:.4f} out-of-sample calibrated ({change:+.1f} %)"
    )


def _methods_section(
    inputs: QualityInputs,
    radar: Mapping[str, Any] | None,
    gauge: Mapping[str, Any] | None,
    summary: Mapping[str, Any],
    radar_curves: Sequence[Mapping[str, Any]],
    gauge_curves: Sequence[Mapping[str, Any]],
) -> dict | None:
    """The rules the numbers were produced under, in the producer's words.

    Null when the frame-age range is unknown, which is the case only when
    neither corpus nor a replay summary was read: without it the page
    cannot say how old the radar picture behind a forecast was, and the
    rest of the block would be a definition of a measurement nobody made.
    """
    frame_age: tuple[float, float] | None = None
    for source in (radar, gauge):
        if source and source.get("frame_age"):
            frame_age = source["frame_age"]
            break
    if frame_age is None:
        age = ((summary.get("run") or {}).get("frame_age_min"))
        if age is not None:
            frame_age = (float(age), float(age))
    if frame_age is None:
        return None

    threshold: float | None = None
    for source in (radar, gauge):
        if source and source.get("threshold_mm_h") is not None:
            threshold = float(source["threshold_mm_h"])
            break
    if threshold is None:
        threshold = float(inputs.raining_now_mm_h)

    rules = (summary.get("run") or {}).get("rules") or {}
    subscriber = {
        "threshold_pct": float(rules.get("threshold_pct", 40)),
        "lead_min": float(rules.get("lead_min", inputs.lead_min)),
        "rearm_after_min": float(rules.get("rearm_after_min", 60)),
        "persistence_obs": float(rules.get("persistence_obs", 1)),
    }
    return {
        "gauge_wet_rule": (
            f"≥ {WET_PRECIP_MM:g} mm, or ≥ {WET_DUR_MIN:g} min with "
            f"precipitation, in a {SLOT_MIN}-minute gauge slot"
        ),
        "onset_rule": (
            f"the first wet slot after {inputs.dry_min} minutes of known-dry "
            f"slots; a warning is a hit when that onset falls within "
            f"{inputs.lead_min} + {inputs.tolerance_min} minutes of it"
        ),
        "threshold_mm_h": _round(threshold, 3),
        "frame_age_range_min": [
            _round(frame_age[0], 1), _round(frame_age[1], 1),
        ],
        "subscriber_rule": subscriber,
        "sources": {"radar": RADAR_SOURCE, "gauges": GAUGE_SOURCE},
        # Additive: the client ignores both, the archive does not. Which
        # probability each diagram is OF is the difference between a claim
        # about the service and a claim about a fit's own training data —
        # the radar curves and the gauge curves are not the same claim and
        # must not be described as though they were.
        "reliability_probability": _reliability_sentence(radar, gauge),
        "reliability_brier_improvement": _brier_improvement(
            radar_curves, inputs.headline_lead_min,
        ),
        "headline_bin_rule": (
            f"the highest bin at the headline lead with n ≥ "
            f"{inputs.headline_min_n}; failing that, the most populated "
            f"bin above {inputs.headline_min_prob:g}"
        ),
        "coverage_rule": (
            f"an onset counts only inside a run of decision rows — "
            f"consecutive frames no more than {inputs.coverage_gap_min} min "
            f"apart, extended by {inputs.lead_min} + {inputs.tolerance_min} "
            f"min at the end. Rain outside those runs fell while the "
            f"service was not watching and is neither a hit nor a miss"
        ),
        "pending_rule": (
            "a warning whose window (sent + lead + tolerance) reaches past "
            "the last gauge slot the station reported, and which has "
            "claimed no onset, is pending: excluded from hits, false "
            "alarms, POD and FAR until the gauge can answer"
        ),
    }


def _reliability_sentence(
    radar: Mapping[str, Any] | None, gauge: Mapping[str, Any] | None,
) -> str:
    """Which probability each reliability diagram is of, named separately."""
    parts = []
    radar_phrase = _mode_phrase(radar)
    if radar_phrase is not None:
        parts.append(f"radar: {radar_phrase}")
    gauge_phrase = _mode_phrase(gauge)
    if gauge_phrase is not None:
        parts.append(f"gauges: {gauge_phrase} against gauge truth")
    return "; ".join(parts) if parts else "not measured"


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


def build_quality_report(inputs: QualityInputs) -> dict:
    """Produce the ``quality.json`` document for these inputs.

    Never raises on a missing or unreadable input: each section is built
    independently and nulls itself when its evidence is absent. The one
    guaranteed pair of fields is ``schema_version`` and
    ``generated_at_utc`` — without those the client renders nothing at
    all, and a document that says only "here is when I was built, and I
    know nothing" is still an honest document.
    """
    curves = _load_curves(inputs.national_curves)

    radar: dict | None = None
    if inputs.radar_corpus is not None and Path(inputs.radar_corpus).is_file():
        # OUT-OF-SAMPLE. The served curves were fitted on this corpus, so
        # applying them here would draw a perfect diagonal and call it a
        # measurement. Leave-one-month-out CV grades a calibration that
        # never saw the row.
        radar = reliability_from_corpus(
            Path(inputs.radar_corpus), outcome_column="outcome",
            curves=curves, inputs=inputs, calibration="cv",
        )
    gauge: dict | None = None
    if inputs.station_corpus is not None and Path(inputs.station_corpus).is_file():
        # The SERVED curves, and legitimately so: the fit never saw
        # ``gauge_outcome``. A rain gauge is an independent instrument, so
        # "what we published against what the ground recorded" is already
        # an out-of-sample claim.
        gauge = reliability_from_corpus(
            Path(inputs.station_corpus), outcome_column="gauge_outcome",
            curves=curves, inputs=inputs, calibration="served",
            per_point=True,
        )

    radar_curves = list(radar["curves"]) if radar else []
    gauge_curves = list(gauge["curves"]) if gauge else []

    decisions, _counts = _load_decisions(inputs)
    board = _score_decisions(decisions, inputs)

    geometry = _station_geometry(inputs)
    names = {sid: geo.get("name") or sid for sid, geo in geometry.items()}

    gauge_brier: dict[str, float] = {}
    if gauge and gauge["per_point_brier"]:
        headline_curve = _closest_lead(gauge_curves, inputs.headline_lead_min)
        if headline_curve is not None:
            gauge_brier = gauge["per_point_brier"].get(
                int(headline_curve["lead_min"]), {},
            )

    summary = _replay_summary(inputs)

    windows_radar = None
    if radar and radar["window"] and radar["window"]["from"]:
        windows_radar = {
            "from": radar["window"]["from"],
            "to": radar["window"]["to"],
            "events": radar["window"]["events"],
            "points": radar["window"]["rows"],
        }
    windows_gauge = None
    if gauge and gauge["window"] and gauge["window"]["from"]:
        windows_gauge = {
            "from": gauge["window"]["from"],
            "to": gauge["window"]["to"],
            "events": gauge["window"]["events"],
            "stations": gauge["window"]["points"],
        }

    events = [
        {
            "station_id": event["station_id"],
            "name": names.get(event["station_id"], event["station_id"]),
            "warned_at_utc": event["warned_at_utc"],
            "eta_min": event["eta_min"],
            "p_rain": event["p_rain"],
            "gauge_onset_utc": event["gauge_onset_utc"],
            "outcome": event["outcome"],
            "lead_error_min": event["lead_error_min"],
        }
        for event in board.events
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _iso(_now(inputs)),
        "windows": {
            "radar": windows_radar,
            "gauge": windows_gauge,
            "live": _live_window(inputs),
        },
        "headline": {
            "reliability": {
                "radar": _headline_reliability(radar_curves, inputs),
                "gauge": _headline_reliability(gauge_curves, inputs),
            },
            "warnings": board.headline,
            "persistence_margin": _persistence_margin(inputs),
        },
        "reliability": {
            "radar": radar_curves or None,
            "gauge": gauge_curves or None,
        },
        "raining_now": board.raining_now,
        "stations": _stations_section(board, geometry, gauge_brier),
        "events": events or None,
        "methods": _methods_section(
            inputs, radar, gauge, summary, radar_curves, gauge_curves,
        ),
    }


# ---------------------------------------------------------------------------
# The schema checker
# ---------------------------------------------------------------------------

_WINDOW_RADAR = {"from": str, "to": str, "events": int, "points": int}
_WINDOW_GAUGE = {"from": str, "to": str, "events": int, "stations": int}
_WINDOW_LIVE = {"days": int, "from": str, "to": str}
_HEADLINE_REL = {"lead_min": float, "said_pct": float, "happened_pct": float, "n": int}
_WARNINGS = {
    "window_days": int, "n_stations": int, "warnings": int, "hits": int,
    "false_alarms": int, "misses": int, "pod": float, "far": float,
}
_MARGIN = {
    "horizon_min": float, "csi_advection": float, "csi_persistence": float,
    "frames": int, "from": str, "to": str,
}
_RAINING_NOW = {
    "n_slots": int, "agreement": float, "pod": float, "far": float,
    "observation_agreement": float, "from": str, "to": str,
}
_STATION_PROPS = {
    "station_id": str, "name": str, "kind": str, "n_events": int,
    "warnings": int,
}
_STATION_NULLABLE = ("brier_gauge", "warn_pod", "warn_far", "raining_now_agreement")


def _check_block(
    block: Any, spec: Mapping[str, type], where: str, problems: list[str],
) -> None:
    if not isinstance(block, dict):
        problems.append(f"{where}: expected an object, got {type(block).__name__}")
        return
    for key, kind in spec.items():
        if key not in block:
            problems.append(f"{where}.{key}: missing")
            continue
        value = block[key]
        if kind is str:
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{where}.{key}: expected a non-empty string")
            elif key in ("from", "to", "warned_at_utc", "gauge_onset_utc",
                         "generated_at_utc"):
                if _parse_ts(value) is None:
                    problems.append(f"{where}.{key}: not an ISO timestamp")
        elif kind is int:
            if isinstance(value, bool) or not isinstance(value, int):
                problems.append(f"{where}.{key}: expected an integer")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"{where}.{key}: expected a number")
            elif not math.isfinite(float(value)):
                problems.append(f"{where}.{key}: not finite")


def validate_report(report: Any) -> list[str]:
    """Structural problems with a report, as a list of human-readable lines.

    An empty list means the document satisfies the ``schema.ts`` contract:
    the right keys, the right types, and nullability only where the schema
    allows it. This is the Python side of the same assertions
    ``frontend/src/lib/quality/load.test.ts`` makes on the fixture — the
    producer must not be able to ship a document the client would silently
    drop half of.
    """
    problems: list[str] = []
    if not isinstance(report, dict):
        return [f"report: expected an object, got {type(report).__name__}"]
    if report.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version: expected {SCHEMA_VERSION}, "
            f"got {report.get('schema_version')!r}",
        )
    if _parse_ts(report.get("generated_at_utc")) is None:
        problems.append("generated_at_utc: missing or not an ISO timestamp")

    for key in ("windows", "headline", "reliability", "raining_now",
                "stations", "events", "methods"):
        if key not in report:
            problems.append(f"{key}: missing (use null, never absent)")

    windows = report.get("windows")
    if not isinstance(windows, dict):
        problems.append("windows: expected an object")
    else:
        for key, spec in (("radar", _WINDOW_RADAR), ("gauge", _WINDOW_GAUGE),
                          ("live", _WINDOW_LIVE)):
            if windows.get(key) is not None:
                _check_block(windows[key], spec, f"windows.{key}", problems)

    headline = report.get("headline")
    if not isinstance(headline, dict):
        problems.append("headline: expected an object")
    else:
        rel = headline.get("reliability")
        if not isinstance(rel, dict):
            problems.append("headline.reliability: expected an object")
        else:
            for key in ("radar", "gauge"):
                if rel.get(key) is not None:
                    _check_block(
                        rel[key], _HEADLINE_REL,
                        f"headline.reliability.{key}", problems,
                    )
        if headline.get("warnings") is not None:
            _check_block(
                headline["warnings"], _WARNINGS, "headline.warnings", problems,
            )
            spread = headline["warnings"].get("lead_error_min")
            _check_block(
                spread, {"p25": float, "p50": float, "p75": float},
                "headline.warnings.lead_error_min", problems,
            )
        if headline.get("persistence_margin") is not None:
            _check_block(
                headline["persistence_margin"], _MARGIN,
                "headline.persistence_margin", problems,
            )

    reliability = report.get("reliability")
    if not isinstance(reliability, dict):
        problems.append("reliability: expected an object")
    else:
        for key in ("radar", "gauge"):
            curves = reliability.get(key)
            if curves is None:
                continue
            if not isinstance(curves, list) or not curves:
                problems.append(f"reliability.{key}: expected a non-empty array")
                continue
            leads = [c.get("lead_min") if isinstance(c, dict) else None for c in curves]
            if leads != sorted(x for x in leads if x is not None):
                problems.append(f"reliability.{key}: leads must be ascending")
            for curve in curves:
                where = f"reliability.{key}[{curve.get('lead_min') if isinstance(curve, dict) else '?'}]"
                _check_block(
                    curve,
                    {"lead_min": float, "brier": float, "n": int, "eff_n": float},
                    where, problems,
                )
                bins = curve.get("bins") if isinstance(curve, dict) else None
                if not isinstance(bins, list) or len(bins) != N_BINS:
                    problems.append(f"{where}.bins: expected {N_BINS} bins")
                    continue
                for index, binned in enumerate(bins):
                    _check_block(
                        binned, {"lo": float, "hi": float, "n": int, "eff_n": float},
                        f"{where}.bins[{index}]", problems,
                    )
                    if not isinstance(binned, dict):
                        continue
                    for field_name in ("forecast_mean", "observed_freq"):
                        value = binned.get(field_name)
                        if value is None:
                            continue
                        if not isinstance(value, (int, float)) or isinstance(value, bool):
                            problems.append(
                                f"{where}.bins[{index}].{field_name}: expected a number or null",
                            )

    if report.get("raining_now") is not None:
        _check_block(report["raining_now"], _RAINING_NOW, "raining_now", problems)

    stations = report.get("stations")
    if stations is not None:
        if not isinstance(stations, dict) or stations.get("type") != "FeatureCollection":
            problems.append("stations: expected a GeoJSON FeatureCollection")
        else:
            features = stations.get("features")
            if not isinstance(features, list) or not features:
                problems.append("stations.features: expected a non-empty array")
            else:
                for index, feature in enumerate(features):
                    where = f"stations.features[{index}]"
                    if not isinstance(feature, dict) or feature.get("type") != "Feature":
                        problems.append(f"{where}: expected a GeoJSON Feature")
                        continue
                    geometry = feature.get("geometry")
                    coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
                    if (
                        not isinstance(geometry, dict)
                        or geometry.get("type") != "Point"
                        or not isinstance(coords, list)
                        or len(coords) != 2
                        or any(
                            not isinstance(c, (int, float)) or isinstance(c, bool)
                            for c in coords
                        )
                    ):
                        problems.append(f"{where}.geometry: expected a [lon, lat] Point")
                    _check_block(
                        feature.get("properties"), _STATION_PROPS,
                        f"{where}.properties", problems,
                    )
                    props = feature.get("properties")
                    if isinstance(props, dict):
                        for name in _STATION_NULLABLE:
                            if name not in props:
                                problems.append(f"{where}.properties.{name}: missing")
                            elif props[name] is not None and (
                                isinstance(props[name], bool)
                                or not isinstance(props[name], (int, float))
                            ):
                                problems.append(
                                    f"{where}.properties.{name}: expected a number or null",
                                )

    events = report.get("events")
    if events is not None:
        if not isinstance(events, list) or not events:
            problems.append("events: expected a non-empty array")
        elif len(events) > 20:
            problems.append(f"events: at most 20, got {len(events)}")
        else:
            stamps = []
            for index, event in enumerate(events):
                where = f"events[{index}]"
                _check_block(
                    event,
                    {"station_id": str, "name": str, "warned_at_utc": str,
                     "eta_min": float, "p_rain": float},
                    where, problems,
                )
                if not isinstance(event, dict):
                    continue
                if event.get("outcome") not in ("hit", "false_alarm"):
                    problems.append(f"{where}.outcome: expected 'hit' or 'false_alarm'")
                onset = event.get("gauge_onset_utc")
                error = event.get("lead_error_min")
                if onset is not None and _parse_ts(onset) is None:
                    problems.append(f"{where}.gauge_onset_utc: not an ISO timestamp")
                if error is not None and (
                    isinstance(error, bool) or not isinstance(error, (int, float))
                ):
                    problems.append(f"{where}.lead_error_min: expected a number or null")
                if event.get("outcome") == "false_alarm" and onset is not None:
                    problems.append(f"{where}: a false alarm cannot carry an onset")
                stamp = _parse_ts(event.get("warned_at_utc"))
                if stamp is not None:
                    stamps.append(stamp)
            if stamps != sorted(stamps, reverse=True):
                problems.append("events: must be newest first")

    methods = report.get("methods")
    if methods is not None:
        _check_block(
            methods,
            {"gauge_wet_rule": str, "onset_rule": str, "threshold_mm_h": float},
            "methods", problems,
        )
        ages = methods.get("frame_age_range_min") if isinstance(methods, dict) else None
        if (
            not isinstance(ages, list) or len(ages) != 2
            or any(
                not isinstance(a, (int, float)) or isinstance(a, bool) for a in ages
            )
        ):
            problems.append("methods.frame_age_range_min: expected [min, max] numbers")
        _check_block(
            (methods or {}).get("subscriber_rule"),
            {"threshold_pct": float, "lead_min": float,
             "rearm_after_min": float, "persistence_obs": float},
            "methods.subscriber_rule", problems,
        )
        _check_block(
            (methods or {}).get("sources"), {"radar": str, "gauges": str},
            "methods.sources", problems,
        )
    return problems


# ---------------------------------------------------------------------------
# The markdown twin
# ---------------------------------------------------------------------------


def _pct(value: float | None, places: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{places}f} %"


def _num(value: float | None, places: int = 3) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """The archive copy: the same numbers, readable without a browser.

    Deliberately a twin rather than a summary — every section of the JSON
    appears, and a null section is printed as "not measured" rather than
    omitted, so a reader can tell the difference between a number that is
    zero and a number that does not exist.
    """
    lines: list[str] = []
    add = lines.append
    add("# How good are we?")
    add("")
    add(f"Generated {report.get('generated_at_utc', '—')} · "
        f"schema version {report.get('schema_version', '—')}.")
    add("")

    windows = report.get("windows") or {}
    add("## Windows")
    add("")
    radar_window = windows.get("radar")
    if radar_window:
        add(f"- **Radar corpus**: {radar_window['from']} → {radar_window['to']}, "
            f"{radar_window['events']:,} events, "
            f"{radar_window['points']:,} verified forecast/observation pairs.")
    else:
        add("- **Radar corpus**: not measured.")
    gauge_window = windows.get("gauge")
    if gauge_window:
        add(f"- **Gauge corpus**: {gauge_window['from']} → {gauge_window['to']}, "
            f"{gauge_window['events']:,} events, "
            f"{gauge_window['stations']:,} stations.")
    else:
        add("- **Gauge corpus**: not measured.")
    live = windows.get("live")
    if live:
        add(f"- **Live scoreboard**: {live['from']} → {live['to']}, "
            f"{live['days']} day(s) covered.")
    else:
        add("- **Live scoreboard**: not measured.")
    add("")

    headline = report.get("headline") or {}
    reliability_headline = headline.get("reliability") or {}
    add("## Headline")
    add("")
    for label, key in (("Radar truth", "radar"), ("Gauge truth", "gauge")):
        block = reliability_headline.get(key)
        if block:
            add(f"- **{label}**: at {block['lead_min']} min we said "
                f"{block['said_pct']:.0f} %, it rained "
                f"{block['happened_pct']:.1f} % of the time "
                f"(n = {block['n']:,}).")
        else:
            add(f"- **{label}**: not measured.")
    warnings = headline.get("warnings")
    if warnings:
        spread = warnings["lead_error_min"]
        pending = warnings.get("pending") or 0
        add(f"- **Warnings**: {warnings['warnings']:,} scored over "
            f"{warnings['window_days']} measured days at "
            f"{warnings['n_stations']} stations — {warnings['hits']:,} hits, "
            f"{warnings['false_alarms']:,} false alarms, "
            f"{warnings['misses']:,} misses"
            + (f", {pending:,} still pending (window not closed)"
               if pending else "")
            + (f", {warnings['uncovered_onsets']:,} gauge onsets outside "
               f"any decision run (not scored)"
               if warnings.get("uncovered_onsets") else "")
            + f". POD {_num(warnings['pod'])}, FAR {_num(warnings['far'])}. "
            f"Lead error p25/p50/p75 = {_num(spread['p25'], 1)} / "
            f"{_num(spread['p50'], 1)} / {_num(spread['p75'], 1)} min "
            f"(positive = late).")
    else:
        add("- **Warnings**: not measured.")
    margin = headline.get("persistence_margin")
    if margin:
        add(f"- **Against persistence** at +{margin['horizon_min']} min: CSI "
            f"{_num(margin['csi_advection'])} vs "
            f"{_num(margin['csi_persistence'])} over "
            f"{margin['frames']:,} frames ({margin['from']} → {margin['to']}).")
    else:
        add("- **Against persistence**: not measured.")
    add("")

    add("## Reliability")
    add("")
    for label, key in (("Radar truth", "radar"), ("Gauge truth", "gauge")):
        curves = (report.get("reliability") or {}).get(key)
        add(f"### {label}")
        add("")
        if not curves:
            add("Not measured.")
            add("")
            continue
        add("| lead | Brier | Brier (raw) | n | eff n |")
        add("|---:|---:|---:|---:|---:|")
        for curve in curves:
            add(f"| {curve['lead_min']} min | {_num(curve['brier'], 4)} | "
                f"{_num(curve.get('brier_raw'), 4)} | "
                f"{curve['n']:,} | {curve['eff_n']:,.0f} |")
        add("")
        headline_curve = _closest_lead(curves, 30)
        if headline_curve is not None:
            add(f"Bins at {headline_curve['lead_min']} min:")
            add("")
            add("| bin | said | happened | n | eff n |")
            add("|---|---:|---:|---:|---:|")
            for binned in headline_curve["bins"]:
                add(f"| {binned['lo']:.1f}–{binned['hi']:.1f} | "
                    f"{_pct(binned['forecast_mean'])} | "
                    f"{_pct(binned['observed_freq'])} | {binned['n']:,} | "
                    f"{binned['eff_n']:,.0f} |")
            add("")

    add("## Is it raining now?")
    add("")
    raining = report.get("raining_now")
    if raining:
        add(f"Over {raining['n_slots']:,} gauge slots "
            f"({raining['from']} → {raining['to']}): the served answer agreed "
            f"{_pct(raining['agreement'])} of the time "
            f"(POD {_num(raining['pod'])}, FAR {_num(raining['far'])}); the "
            f"raw radar observation alone agreed "
            f"{_pct(raining['observation_agreement'])}.")
    else:
        add("Not measured.")
    add("")

    add("## Stations")
    add("")
    stations = report.get("stations")
    if stations and stations.get("features"):
        add("| station | name | kind | rows | Brier (gauge) | warnings | POD | FAR | now agrees |")
        add("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for feature in stations["features"]:
            p = feature["properties"]
            add(f"| `{p['station_id']}` | {p['name']} | {p['kind'] or '—'} | "
                f"{p['n_events']:,} | {_num(p['brier_gauge'], 4)} | "
                f"{p['warnings']:,} | {_num(p['warn_pod'])} | "
                f"{_num(p['warn_far'])} | "
                f"{_pct(p['raining_now_agreement'])} |")
    else:
        add("Not measured.")
    add("")

    add("## Recent warnings")
    add("")
    events = report.get("events")
    if events:
        add("| sent (UTC) | station | ETA | P(rain) | onset | outcome | lead error |")
        add("|---|---|---:|---:|---|---|---:|")
        for event in events:
            add(f"| {event['warned_at_utc']} | {event['name']} "
                f"(`{event['station_id']}`) | {_num(event['eta_min'], 0)} min | "
                f"{_pct(event['p_rain'], 0)} | "
                f"{event['gauge_onset_utc'] or '—'} | {event['outcome']} | "
                f"{_num(event['lead_error_min'], 1)} min |")
    else:
        add("Not measured.")
    add("")

    add("## Methods")
    add("")
    methods = report.get("methods")
    if methods:
        rule = methods["subscriber_rule"]
        ages = methods["frame_age_range_min"]
        add(f"- Wet gauge slot: {methods['gauge_wet_rule']}.")
        add(f"- Onset: {methods['onset_rule']}.")
        add(f"- Forecast threshold: {methods['threshold_mm_h']} mm/h.")
        add(f"- Radar frame age behind a forecast: {ages[0]}–{ages[1]} min.")
        add(f"- Subscriber rule: warn at {rule['threshold_pct']:.0f} % of rain "
            f"within {rule['lead_min']:.0f} min, "
            f"{rule['persistence_obs']:.0f} observation(s) of persistence, "
            f"{rule['rearm_after_min']:.0f} min disarmed after a warning.")
        if methods.get("reliability_probability"):
            add(f"- Reliability is of — {methods['reliability_probability']}.")
        if methods.get("reliability_brier_improvement"):
            add(f"- {methods['reliability_brier_improvement']}.")
        if methods.get("headline_bin_rule"):
            add(f"- Headline bin: {methods['headline_bin_rule']}.")
        if methods.get("coverage_rule"):
            add(f"- Coverage: {methods['coverage_rule']}.")
        if methods.get("pending_rule"):
            add(f"- Pending: {methods['pending_rule']}.")
        add(f"- Sources: {methods['sources']['radar']}; "
            f"{methods['sources']['gauges']}.")
    else:
        add("Not measured.")
    add("")
    return "\n".join(lines)
