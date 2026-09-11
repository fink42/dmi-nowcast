"""The gauge reliability diagram, scored on the probability the site serves.

Until 2026-09-11 the ``/quality`` page's gauge curve came from the station
calibration corpus: ``raw_prob`` pushed through the served isotonic curve,
graded against that corpus's ``gauge_outcome``. That was an honest claim
about the *curve*, and the site stopped serving the curve — the panel and
the push notifications both decide on the gauge-trained post-processed
model (``push/postprocess.py``). The page was therefore drawing a
reliability diagram of a probability nobody is shown, and it showed: the
curve sat far below the diagonal while the model verifies on it at the
gauges (0.75 said → 0.73 observed).

So the source changes. This module scores the **decision rows** — the
replay's reconstruction plus the live ``station_eval`` scoreboard, the
same table the threshold sweep and the nightly refit read — taking
``p_post_<lead>`` where a row carries it, against "the gauge was wet
within L" as :class:`~dmi_nowcast_sidecar.decision_rows.GaugeGrid` defines
it. Every piece of that is borrowed rather than restated:

* :func:`~dmi_nowcast_sidecar.decision_rows.load_probabilities` decides
  which rows a run contributes, and the live row wins a tie with the
  replay's reconstruction;
* :class:`~dmi_nowcast_sidecar.postprocess_fit.ProbabilityFiller` decides
  what a row without a stored probability is worth — so the page, the
  fitted thresholds and the benchmark all score the same rows;
* :func:`~dmi_nowcast_sidecar.decision_rows.build_gauge_grid` decides what
  wet means, how far the window is padded and which broken buckets are
  excluded;
* :func:`~dmi_nowcast_core.benchmark.reliability_bins` and
  :func:`~dmi_nowcast_core.benchmark.brier_decomposition` do the binning
  and the score.

* :func:`~dmi_nowcast_core.postprocess.leave_one_month_out` decides what
  "out-of-sample" means, so the page's diagram and the offline evidence
  are honest in the same way.

Those are exactly the calls ``scripts/benchmark_report.py``'s Layer B and
``scripts/fit_postprocess.py`` make, on the same rows — which is the
point. The page
and the benchmark now cannot disagree about how well the served
probability verifies at the gauges, because they are one computation
invoked twice. ``sidecar/tests/test_gauge_reliability.py`` pins the two
against each other on synthetic rows.

Out-of-fold, or it measures nothing
-----------------------------------

The nightly refit fits on ALL rows (``training.held_out: false``) — no
folds, because the out-of-sample evidence is the offline script's job.
Scoring that model's predictions on the very rows it was fitted on draws
a perfect diagonal and calls it a measurement: the first live build came
back 0.149 → 0.149, 0.754 → 0.753, 0.949 → 0.949 over hundreds of
thousands of rows, which says only that isotonic regression can describe
its own training data. It is the same trap the radar curve had, removed
there by leave-one-month-out CV.

So the probability this module bins is an OUT-OF-FOLD one. For each
``(year, month)`` the whole two-stage fit — logistic then isotonic — is
redone on the other months and applied to the held-out one, through
:func:`~dmi_nowcast_core.postprocess.leave_one_month_out` with the
bootstrap and the strata switched off: this wants the fold predictions,
not a report. A fold that cannot be fitted at all falls back to the
served model's own prediction for those rows and is counted, because a
gap in the curve would be a quieter lie than a labelled fallback.

That costs one logistic fit per fold per lead — about ten folds by four
leads on the current archive — which is why it runs in the nightly
child process and never on the hourly refresh.

The output is shaped like
:func:`~dmi_nowcast_core.quality_report.reliability_from_corpus`'s, so it
drops into the report builder where the corpus fit used to sit and feeds
the same three things: ``reliability.gauge``, ``windows.gauge`` and the
per-station ``brier_gauge`` behind the station map's colours. A run that
cannot be scored at all returns ``None``, and the builder falls back to
the corpus fit rather than nulling the section.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from dmi_nowcast_core import postprocess as core_postprocess
from dmi_nowcast_core.warning_score import (
    DEFAULT_DRY_MIN,
    DEFAULT_MIN_KNOWN_SLOTS,
    DEFAULT_ONSET_MIN_MM,
    DEFAULT_PRODUCT_LEADS_MIN,
    p_rain_column,
)

#: Ten fixed-width bins, the convention shared by ``quality_report``,
#: ``benchmark.reliability_bins`` and ``sql/reliability_pooled.sql``.
N_BINS = 10

#: Seconds in a day — the fold machinery's block key. Restated from
#: ``decision_rows`` rather than imported at module scope so this module's
#: import graph stays free of pyarrow until something actually reads.
DAY_SEC = 86_400

#: What ``methods.reliability_probability`` calls each probability. The
#: report turns these into the sentence; they are named here because this
#: module is the only thing that knows which column it actually read.
MODE_POSTPROCESS = "postprocess"
#: The same model, refitted per fold and scored on the month it never saw.
MODE_POSTPROCESS_CV = "postprocess_cv"
MODE_CURVE = "served"


@dataclass(frozen=True)
class GaugeReliabilityOptions:
    """Where the rows are, what to score, and under which gauge rule.

    Deliberately mirrors :class:`~dmi_nowcast_sidecar.postprocess_fit.
    PostprocessFitOptions` and the threshold sweep's options: the page's
    curve, the model's fit and the served thresholds have to stand on one
    set of rows under one definition of wet, or the three numbers on the
    page are about three different samples.
    """

    #: Decision-row trees, LATER directories winning a ``(radar_ts,
    #: station_id)`` tie — the replay first, the live scoreboard second.
    decisions_dirs: list[Path]
    #: The corpus root holding ``stations/obs/``: the gauge truth.
    corpus_dir: Path
    #: The horizons to publish a curve for.
    leads: tuple[int, ...]
    #: ``"{lead}"``-templated probability column. ``None`` reads the served
    #: curve column ``p_rain_<lead>``, which is what the page showed before
    #: the model existed and remains the honest answer on a deployment
    #: that serves the curve.
    probability_column: str | None = None
    #: The served model, used to FILL ``probability_column`` on rows that
    #: carry features but no stored value — the replay tree, and every
    #: live partition written before the engine started writing it. Rows
    #: with neither are excluded and counted. ``None`` scores only what is
    #: stored, which on this archive is a few hours of it.
    postprocess_model: Path | None = None
    #: The leads whose raw fraction the model's design reads. Must match
    #: the model's own ``design_leads``; the national product leads, as
    #: the sweep and the nightly fit both pass.
    design_leads: tuple[int, ...] = DEFAULT_PRODUCT_LEADS_MIN
    #: The onset/dead-gauge rule, from ``fit_thresholds`` so the page and
    #: the fitted table exclude the same broken buckets.
    dry_min: int = DEFAULT_DRY_MIN
    onset_min_mm: float = DEFAULT_ONSET_MIN_MM
    min_known_slots: int = DEFAULT_MIN_KNOWN_SLOTS
    #: Score OUT-OF-FOLD predictions (leave-one-month-out refit) rather
    #: than the served model's own. On by default and effectively
    #: mandatory: the nightly model is fitted on all rows, so its in-sample
    #: diagram is a tautology. Off only to reproduce that tautology
    #: deliberately, or to cut ~40 logistic fits out of a build.
    out_of_fold: bool = True
    #: Ridge strength of the out-of-fold refits. The served fit's, so a
    #: fold model differs from the shipped one only in what it saw.
    l2: float = 1.0
    #: Stations to restrict to; ``None`` scores every station in the rows.
    stations: Sequence[str] | None = None
    #: Per-lead Brier for the station map (``reliability.gauge`` feeds the
    #: same map today, through ``per_point_brier``).
    per_point: bool = True
    #: Extra columns to log nothing about — reserved, kept so the
    #: dataclass can grow without a signature change.
    extra: dict[str, Any] = field(default_factory=dict)


def _column_for(template: str | None) -> Callable[[int], str]:
    """The probability column namer for a template, or the served default."""
    if template is None:
        return p_rain_column
    from .decision_rows import column_template

    return column_template(template)


def _filler(
    options: GaugeReliabilityOptions,
    column_for: Callable[[int], str],
    log: Callable[[str], None] | None,
) -> Any:
    """The ``ProbabilityFiller`` for these options, or ``None``.

    ``None`` whenever there is nothing to fill WITH — the curve column is
    being scored, or no model path was given — in which case the read
    falls back to whatever the rows already carry, which is the behaviour
    that shipped.

    A model file that cannot be read is not fatal: the sweep treats that
    the same way, and for the same reason. A diagram drawn from the stored
    rows alone is a narrower claim, never a wrong one, and it beats a
    nightly build that dies over one bad file.
    """
    if options.probability_column is None or options.postprocess_model is None:
        return None
    from dmi_nowcast_core.postprocess import PostprocessModel

    from .postprocess_fit import ProbabilityFiller

    model = None
    try:
        model = PostprocessModel.loads(
            Path(options.postprocess_model).read_text(encoding="utf-8"),
        )
    except Exception as exc:  # noqa: BLE001 — every way a file can be junk
        if log:
            log(
                f"gauge reliability: cannot read {options.postprocess_model} "
                f"({type(exc).__name__}: {exc}); scoring only the rows that "
                "already carry the column"
            )
    return ProbabilityFiller(
        model,
        tuple(sorted({int(lead) for lead in options.leads})),
        tuple(int(lead) for lead in options.design_leads),
        column_for,
    )


def _out_of_fold(
    options: GaugeReliabilityOptions,
    filler: Any,
    rows: dict,
    outcomes: dict[int, tuple[np.ndarray, np.ndarray]],
    leads: Sequence[int],
    *,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[int, np.ndarray], int] | None:
    """``({lead: held-out prediction}, n_folds)``, or ``None``.

    One fold per ``(year, month)``, the whole two-stage fit redone on the
    other months — :func:`~dmi_nowcast_core.postprocess.leave_one_month_out`
    does it, and does it the way the offline evidence was produced. Two of
    its knobs are turned off here on purpose:

    ``n_resamples=0``
        No bootstrap. The confidence interval on a difference is what the
        shipping decision is written in; this is a reliability diagram,
        and the interval would cost far more than the fits.
    ``strata={}``
        Pooled only. The per-season split is the report's business, and
        computing it would score every lead four times over for numbers
        nothing here publishes.

    ``None`` when the refit cannot happen at all: no model to refit, no
    feature columns to refit on, or fewer than two months in the rows —
    one fold is no fold, and pretending otherwise would be the tautology
    this exists to remove.
    """
    if filler is None or getattr(filler, "model", None) is None:
        return None
    t = rows["t"]
    months = core_postprocess.year_months_from_epoch(t)
    n_folds = int(np.unique(months).size)
    if n_folds < 2:
        if log:
            log(
                f"gauge reliability: {n_folds} month(s) of rows — too few to "
                "hold one out, scoring the served model in-sample"
            )
        return None
    design_leads = tuple(int(lead) for lead in options.design_leads)
    features: dict[str, Any] = {}
    for name in core_postprocess.feature_source_columns(design_leads):
        values = rows["extra"].get(name)
        if values is not None:
            features[name] = values
    if not features:
        return None
    # ``season`` and ``hour_utc`` from the decision instant, exactly as
    # ``ProbabilityFiller._predict`` derives them — never a stored column,
    # so a row written by a writer that had one scores identically to a
    # row written by one that did not.
    features["season"] = core_postprocess.seasons_from_epoch(t)
    features["hour_utc"] = core_postprocess.hours_from_epoch(t).astype(
        np.float64,
    )
    baseline = {
        int(lead): np.asarray(rows["p"][int(lead)], dtype=np.float64)
        for lead in leads
    }
    if log:
        log(
            f"gauge reliability: refitting out-of-fold over {n_folds} "
            f"month(s) x {len(leads)} lead(s)"
        )
    evaluation = core_postprocess.leave_one_month_out(
        features,
        {int(lead): outcomes[int(lead)] for lead in leads},
        [int(lead) for lead in leads],
        month=months,
        day=t // DAY_SEC,
        baseline=baseline,
        l2=float(options.l2),
        design_leads=design_leads,
        strata={},
        n_resamples=0,
        log=log,
    )
    return {
        int(lead): np.asarray(values, dtype=np.float64)
        for lead, values in evaluation["out_of_fold"].items()
    }, n_folds


def _mode(
    options: GaugeReliabilityOptions, held_out: dict | None,
) -> str:
    """Which probability the diagram is OF, for the methods sentence."""
    if options.probability_column is None:
        return MODE_CURVE
    return MODE_POSTPROCESS if held_out is None else MODE_POSTPROCESS_CV


def _calibration(
    options: GaugeReliabilityOptions, held_out: dict | None,
) -> str:
    """Whether the diagram is a measurement or a description of a fit."""
    if options.probability_column is None:
        return "served"
    return "in-sample" if held_out is None else "out-of-fold"


def _bins(prob: np.ndarray, outcome: np.ndarray) -> list[dict]:
    """``benchmark.reliability_bins`` output, widened to all ten bins.

    The benchmark omits an empty bin because 0.0 is a value and an
    unoccupied bin has no observed frequency. ``quality.json``'s contract
    keeps every bin and nulls the two means instead, so the page can draw
    a gap rather than a point at the origin. Same numbers, two audiences.
    """
    from dmi_nowcast_core.benchmark import reliability_bins

    occupied = {
        int(row["bin"]): row for row in reliability_bins(
            prob, outcome, n_bins=N_BINS,
        )
    }
    out: list[dict] = []
    for k in range(N_BINS):
        row = occupied.get(k)
        out.append({
            "lo": round(k / N_BINS, 6),
            "hi": round((k + 1) / N_BINS, 6),
            "forecast_mean": None if row is None else _round(row["mean_p"]),
            "observed_freq": None if row is None else _round(row["observed"]),
            "n": 0 if row is None else int(row["n"]),
            # Every decision row is one forecast at one station: there is
            # no sample weight to discount, so the effective count IS the
            # count. (The corpus path weights pixels, which is why it
            # carries a Kish size at all.)
            "eff_n": 0.0 if row is None else float(int(row["n"])),
        })
    return out


def _round(value: Any, places: int = 6) -> float | None:
    """JSON-safe rounding; non-finite becomes null, never a zero."""
    if value is None:
        return None
    out = float(value)
    return round(out, places) if math.isfinite(out) else None


def _brier(prob: np.ndarray, outcome: np.ndarray) -> float | None:
    from dmi_nowcast_core.benchmark import brier_decomposition

    if prob.size == 0:
        return None
    score = float(brier_decomposition(prob, outcome, n_bins=N_BINS)["brier"])
    return score if math.isfinite(score) else None


def gauge_reliability_from_decisions(
    options: GaugeReliabilityOptions, *, log: Callable[[str], None] | None = None,
) -> dict | None:
    """Reliability of the SERVED probability at the gauges, per lead.

    Returns the shape
    :func:`~dmi_nowcast_core.quality_report.reliability_from_corpus`
    returns — ``curves`` / ``window`` / ``frame_age`` / ``threshold_mm_h``
    / ``per_point_brier`` / ``mode`` — so the report builder can use it
    wherever the corpus fit used to go. ``None`` when there is nothing to
    score: no decision rows, no gauge store, no station with observations,
    or no lead whose probability column any row carries.

    Never raises on bad evidence. It runs inside the nightly build, which
    must produce a document even when one of its inputs is missing, and a
    null gauge section is a worse answer than the curve it replaces only
    if it is silent about why — so failures log and fall back rather than
    propagating.
    """
    from .decision_rows import (
        build_gauge_grid,
        decision_window,
        load_probabilities,
        recode_stations,
    )
    from .threshold_sweep import SweepError

    dirs = [Path(d) for d in options.decisions_dirs]
    leads = tuple(sorted({int(lead) for lead in options.leads}))
    if not dirs or not leads or options.corpus_dir is None:
        return None
    column_for = _column_for(options.probability_column)
    # The served curve column travels alongside as an extra, so the
    # improvement the model made is a PAIRED number on the very rows the
    # curve was replaced on rather than two scores over two samples.
    baseline_names = {
        lead: p_rain_column(lead) for lead in leads
    }
    # Most of the archive predates the serving path and carries features
    # instead of a stored probability. Filling happens inside the read,
    # per file, through the very class the nightly sweep uses.
    filler = _filler(options, column_for, log)
    feature_columns = (
        [] if filler is None
        else core_postprocess.feature_source_columns(options.design_leads)
    )
    # The out-of-fold refit needs the features to SURVIVE the read, not
    # just to pass through it — it fits on them. Without it they are read
    # for the filler and dropped again, which is twenty columns of the
    # merged table saved on a build that will not refit.
    carried = sorted(set(baseline_names.values()))
    if options.out_of_fold and feature_columns:
        carried = sorted(set(carried) | set(feature_columns))
    try:
        rows = load_probabilities(
            dirs, leads,
            stations=options.stations,
            column_for=column_for,
            extra_columns=carried,
            derive=filler,
            derive_columns=feature_columns,
            log=log,
        )
    except SweepError as exc:
        if log:
            log(f"gauge reliability: no rows to score ({exc})")
        return None
    except Exception as exc:  # noqa: BLE001 — an unreadable tree is evidence
        # we do not have, not a build failure.
        if log:
            log(f"gauge reliability: reading the decision rows failed: {exc}")
        return None

    if log and filler is not None:
        log(
            f"gauge reliability: {options.probability_column} — "
            f"{filler.counts['stored']} stored, "
            f"{filler.counts['computed']} computed from features, "
            f"{filler.counts['dropped']} row(s) excluded (no features)"
        )

    try:
        window = decision_window(rows["t"])
        grid, _dead, scored = build_gauge_grid(
            Path(options.corpus_dir),
            sorted(set(rows["stations"])),
            window,
            dry_min=int(options.dry_min),
            onset_min_mm=float(options.onset_min_mm),
            min_known_slots=int(options.min_known_slots),
            log=log,
        )
        recode_stations(rows, scored)
    except SweepError as exc:
        if log:
            log(f"gauge reliability: no gauge truth to score against ({exc})")
        return None
    except Exception as exc:  # noqa: BLE001 — see above.
        if log:
            log(f"gauge reliability: building the gauge grid failed: {exc}")
        return None

    dropped = rows.get("dropped")
    station_ids = list(scored)
    t = rows["t"]
    station = rows["station"]

    # The gauge outcome per lead, once: the refit below is fitted on it
    # and the diagram is scored against it, and a second opinion about
    # which rows are gradable would make those two different claims.
    outcomes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for lead in leads:
        outcome, usable = grid.outcome(t, station, lead)
        if dropped is not None:
            usable = usable & ~dropped
        outcomes[int(lead)] = (outcome, usable)

    held_out: dict[int, np.ndarray] | None = None
    n_folds = 0
    if options.out_of_fold:
        try:
            folded = _out_of_fold(
                options, filler, rows, outcomes, leads, log=log,
            )
        except Exception as exc:  # noqa: BLE001 — a refit failure costs the
            # out-of-sample claim, not the build. The block then says
            # ``in-sample`` rather than quietly labelling a tautology
            # out-of-fold, which is the failure that matters.
            if log:
                log(
                    "gauge reliability: the out-of-fold refit failed "
                    f"({type(exc).__name__}: {exc}); falling back to the "
                    "served model's own predictions, labelled in-sample"
                )
            folded = None
        if folded is not None:
            held_out, n_folds = folded

    curves: list[dict] = []
    per_point_brier: dict[int, dict[str, float]] = {}
    fallbacks = 0
    for lead in leads:
        served = np.asarray(rows["p"][lead], dtype=np.float64)
        outcome, usable = outcomes[int(lead)]
        if held_out is None:
            prob = served
        else:
            # A fold the refit could not fit leaves NaN. Those rows fall
            # back to the served model — in-sample for them, and counted,
            # because a hole in the curve would be the quieter lie.
            fold_prob = held_out.get(int(lead))
            if fold_prob is None:
                fold_prob = np.full(served.shape, np.nan, dtype=np.float64)
            gap = usable & ~np.isfinite(fold_prob) & np.isfinite(served)
            fallbacks += int(gap.sum())
            prob = np.where(np.isfinite(fold_prob), fold_prob, served)
        keep = usable & np.isfinite(prob)
        # A row the gauge CAN answer for but whose probability column the
        # file never carried is excluded and counted. Scoring it would
        # mean inventing the number the service is being graded on.
        excluded = int((usable & ~np.isfinite(prob)).sum())
        if not keep.any():
            continue
        p = prob[keep]
        y = outcome[keep]
        brier = _brier(p, y)
        if brier is None:
            continue
        baseline = rows["extra"].get(baseline_names[lead])
        brier_raw: float | None = None
        if baseline is not None:
            paired = keep & np.isfinite(baseline)
            if paired.any():
                brier_raw = _brier(baseline[paired], outcome[paired])
        curves.append({
            "lead_min": int(lead),
            "brier": _round(brier),
            # The same rows on the probability this one replaced — the
            # served curve. On the corpus path this key means "no
            # calibration at all"; here it means "the number we used to
            # show", which is the comparison that matters now.
            "brier_raw": _round(brier_raw),
            "n": int(p.size),
            "eff_n": float(int(p.size)),
            # Additive: how many scoreable rows had no probability in the
            # column this curve is of. A large number here means the
            # archive predates the model, not that the model is silent.
            "n_excluded": excluded,
            "bins": _bins(p, y),
        })
        if not options.per_point:
            continue
        codes = station[keep]
        squares = (p - y) * (p - y)
        totals = np.bincount(codes, minlength=len(station_ids))
        sums = np.bincount(codes, weights=squares, minlength=len(station_ids))
        per_point_brier[int(lead)] = {
            station_ids[index]: float(sums[index] / totals[index])
            for index in range(len(station_ids))
            if totals[index] > 0
        }

    if not curves:
        if log:
            log("gauge reliability: no lead could be scored")
        return None
    return {
        "curves": curves,
        "window": {
            "from": _iso(window[0]),
            "to": _iso(window[1]),
            # One "event" is one decision instant the service took a
            # decision at — the decision-row analogue of the corpus's
            # radar event, and what the page's "over N events" reads.
            "events": int(np.unique(t).size),
            "rows": int(rows["rows"]),
            "points": len(station_ids),
        },
        # The decision rows carry no frame age in this narrow read and no
        # rain threshold at all; both stay null so the methods block falls
        # through to the radar corpus's, which is where they are measured.
        "frame_age": None,
        "threshold_mm_h": None,
        "per_point_brier": per_point_brier,
        "mode": _mode(options, held_out),
        # Which claim the diagram is: ``out-of-fold`` means every
        # prediction was made by a model refitted without the month it
        # was scored on; ``in-sample`` means it was the served model's own
        # output on its own training rows, which is a tautology and is
        # labelled as one rather than quietly shown. ``served`` is the
        # curve path, where the fit never saw a gauge at all.
        "calibration": _calibration(options, held_out),
        "cv_folds": int(n_folds),
        "fold": "month" if n_folds else None,
        # Rows a fold could not be fitted for, which fell back to the
        # served model's own prediction. Non-zero means that many rows of
        # the diagram are in-sample.
        "in_sample_fallbacks": int(fallbacks),
        # Additive provenance: which column was scored, and over how many
        # files. The page ignores both; a human reading the archived
        # document should not have to guess.
        "probability_column": (
            options.probability_column or "p_rain_{lead}"
        ),
        "files": int(rows["files"]),
        # Where each scored row's probability came from, in the filler's
        # own units — ``rows``, ``stored`` and ``dropped`` count ROWS,
        # ``computed`` counts the (row, lead) cells it wrote. The same
        # four numbers the nightly sweep logs, so the two can be read
        # against each other. On this archive the interesting one is
        # ``computed``: ten months of replay rows that carry features and
        # no stored value, i.e. the difference between a diagram of the
        # whole record and a diagram of the hours since the serving path
        # shipped. ``dropped`` is rows with neither — the only exclusions
        # the read itself makes.
        "fill": (
            None if filler is None else {
                key: int(value) for key, value in filler.counts.items()
            }
        ),
    }


def _iso(value: datetime) -> str:
    """A UTC ISO-8601 stamp with a ``Z``, as ``quality_report._iso`` writes."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


__all__ = [
    "GaugeReliabilityOptions",
    "MODE_CURVE",
    "MODE_POSTPROCESS",
    "MODE_POSTPROCESS_CV",
    "N_BINS",
    "gauge_reliability_from_decisions",
]
