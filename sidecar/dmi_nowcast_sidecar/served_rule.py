"""The quality page's warning scoreboard, re-decided under the served rule.

Until this existed, ``/quality``'s scoreboard, station map and recent
warnings counted the ``action == "notify"`` rows stored in the two
decision trees. Both trees decide with a FIXED subscriber row — 40 % at
30 min, one observation of persistence, 60 min disarmed — while the push
service that actually notifies people decides with the nightly fitted
threshold table (``push_thresholds``; 50/60/70/75 % at 20/30/45/60 min on
2026-09-11) applied to the gauge-trained ``p_post``. So the page measured
a rule nobody is subscribed to: 40 % on the old curve scale over-fires,
and nationally that read 6 912 warnings at FAR 0.86 and POD 0.22, where
the served rule measured properly comes out at POD 0.35 and precision
0.31 at 30 min.

Rather than regenerate a season of parquet under a new threshold — which
would have to be redone at every refit — the rows are re-decided at
report build time. The rows already carry every served lead's probability
and the post-processing features; what they do not carry is a decision
under tonight's rule, and that is a replay of a state machine over
columns already on disk.

One definition of the rule, borrowed not restated
-------------------------------------------------

Every piece of this is the same call the nightly threshold fit and
``scripts/benchmark_report.py`` make, on the same rows:

* :func:`~dmi_nowcast_sidecar.threshold_sweep.load_decisions` reads the
  trees and decides which row wins a ``(radar_ts, station_id)`` tie;
* :class:`~dmi_nowcast_sidecar.postprocess_fit.ProbabilityFiller` fills
  ``p_post_<lead>`` from the feature columns for the rows that predate the
  serving path (which is most of the archive);
* :func:`~dmi_nowcast_sidecar.threshold_sweep.build_tracks` assigns the
  coverage-run index every replay resets its state on;
* :func:`~dmi_nowcast_sidecar.threshold_sweep.replay_station` runs
  ``push.engine.evaluate`` itself.

Two things are this module's own, and both exist to make the page's two
halves one measurement:

**The row set is the report's.** The core builder has already decided
which rows it scores — its live-day cutoff, its "the live row wins the
replay's reconstruction" dedup — and hands them over. This restricts
itself to those ``(radar_ts, station_id)`` keys, so the scoreboard's
warnings and the coverage runs they are graded inside come from exactly
one population. A row this module reads that the report did not is
dropped; a row the report has that this cannot fill is simply never
warned on.

**The engine's per-row fallback is applied before the replay.** The push
engine reads ``p_post`` where it exists and ``p_rain`` where it does not
(``push.engine.Observation.p_decision``) — one point off coverage for the
model must not silence it. ``replay_station`` sees one column, so the
fallback happens here: the post-processed column is filled from the
curve column wherever it is null, and the rows that took it are counted.

The output is :attr:`~dmi_nowcast_core.quality_report.QualityInputs.
decide_warnings`' contract — ``{station_id: [(sent_utc, eta_min,
probability)]}`` — plus a stats dict the report publishes under
``methods.subscriber_rule`` and the job logs. The core report cannot
import this module (it must import nothing from the sidecar), which is
why it takes the decision as a callable rather than a path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dmi_nowcast_core.push_thresholds import (
    effective_threshold,
    lead_pick,
    load_thresholds,
)
from dmi_nowcast_core.quality_report import SCORED_RE_DECIDED
from dmi_nowcast_core.warning_score import (
    DEFAULT_COVERAGE_GAP_MIN,
    DEFAULT_LEAD_MIN,
    DEFAULT_PRODUCT_LEADS_MIN,
    p_rain_column,
)

#: What ``subscriber_rule.probability`` says, mirroring
#: ``push.probability_source`` exactly — the page must not invent a third
#: word for a choice the config already has two for.
PROBABILITY_POSTPROCESS = "postprocess"
PROBABILITY_CURVE = "curve"

#: Where the threshold came from. ``table`` and ``fallback`` are
#: ``push.thresholds.ThresholdTable.effective``'s own two answers;
#: ``config`` is this module's third, for a build with no table file at
#: all, which then uses the station-eval rule's configured percent.
SOURCE_TABLE = "table"
SOURCE_FALLBACK = "fallback"
SOURCE_CONFIG = "config"


@dataclass(frozen=True)
class ServedRuleOptions:
    """Where the rows are, and the rule to re-decide them under.

    Deliberately mirrors :class:`~dmi_nowcast_sidecar.gauge_reliability.
    GaugeReliabilityOptions`: the page's scoreboard, the page's gauge
    curve and the served thresholds have to stand on one set of rows read
    one way, or the numbers beside each other on the page are about
    different samples.
    """

    #: Decision-row trees, LATER directories winning a ``(radar_ts,
    #: station_id)`` tie — the replay first, the live scoreboard second,
    #: exactly as the core builder merges them.
    decisions_dirs: list[Path] = field(default_factory=list)
    #: The served ``push_thresholds.json``. ``None``, missing or unusable
    #: falls back to :attr:`fallback_threshold_pct` and says so.
    thresholds_path: Path | None = None
    #: The horizon the scoreboard is about — ``station_eval.rules.lead_min``.
    lead_min: int = DEFAULT_LEAD_MIN
    #: ``push.probability_source``. ``"postprocess"`` decides on
    #: ``p_post_<lead>`` with a per-row fallback to ``p_rain_<lead>``;
    #: ``"curve"`` decides on ``p_rain_<lead>`` and fills nothing.
    probability_source: str = PROBABILITY_CURVE
    #: The served post-processing model, used to FILL ``p_post_<lead>``
    #: on rows that carry features but no stored value. Without it only
    #: the hours since the serving path shipped carry a probability, and
    #: the page would measure the archive's depth.
    postprocess_model: Path | None = None
    #: The leads the model's design reads. Must match the model's own.
    design_leads: tuple[int, ...] = DEFAULT_PRODUCT_LEADS_MIN
    #: The rest of the live subscriber row (``station_eval.rules`` and
    #: ``forecast.rain_threshold_mm_h``), so the replay is the service's
    #: rule and not the sweep's defaults.
    persistence_obs: int = 1
    rearm_after_min: int = 60
    raining_now_mm_h: float = 0.5
    raining_now_eta_min: float = 1.5
    #: The same coverage-gap the report scores inside, so the state
    #: machine resets at the instants the coverage runs break.
    coverage_gap_min: int = DEFAULT_COVERAGE_GAP_MIN
    #: The percent to warn at when there is no usable table document:
    #: ``station_eval.rules.threshold_pct``, i.e. what the live job would
    #: itself fall back to.
    fallback_threshold_pct: int = 40


def post_column(lead: int) -> str:
    """``p_post_<lead>`` — the engine's own column, named once."""
    from dmi_nowcast_core import postprocess as core_postprocess

    return core_postprocess.post_column(int(lead))


def resolve_threshold(options: ServedRuleOptions) -> tuple[int, str]:
    """``(percent, source)`` for this run's lead. Total, never raises.

    The document is read through the core's own helpers, so the page
    picks the number the running service would pick for the same horizon
    — including its fallback for a lead the fit could not speak for.
    """
    doc = None
    if options.thresholds_path is not None:
        doc = load_thresholds(options.thresholds_path)
    if doc is None:
        return int(options.fallback_threshold_pct), SOURCE_CONFIG
    threshold = int(effective_threshold(doc, options.lead_min))
    source = (
        SOURCE_TABLE
        if lead_pick(doc, str(int(options.lead_min))) is not None
        else SOURCE_FALLBACK
    )
    return threshold, source


class ServedRuleDecider:
    """The ``decide_warnings`` hook: rows in, warnings under the served rule out.

    Construct it, hand :meth:`__call__` to
    :class:`~dmi_nowcast_core.quality_report.QualityInputs` as
    ``decide_warnings`` and :attr:`stats` as ``served_rule``. The stats
    dict is filled in during the call and read afterwards, which is safe
    because the builder scores the decisions before it writes the methods
    block — see ``QualityInputs.served_rule``.

    Never raises. The whole point of the page is that a missing input
    nulls its own section; a scoreboard that could not be re-decided
    should come back empty and say why, not take the nightly build down
    with it. A failure leaves ``stats["error"]`` set and returns no
    warnings, which the builder renders as a scoreboard of pure misses —
    honest, and loud.
    """

    def __init__(
        self,
        options: ServedRuleOptions,
        *,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.options = options
        self._log = log
        threshold, source = resolve_threshold(options)
        self.threshold_pct = threshold
        self.threshold_source = source
        post = options.probability_source == PROBABILITY_POSTPROCESS
        self.probability = (
            PROBABILITY_POSTPROCESS if post else PROBABILITY_CURVE
        )
        self.column = (
            post_column(options.lead_min) if post
            else p_rain_column(options.lead_min)
        )
        #: Published under ``methods.subscriber_rule`` and logged. Mutated
        #: by :meth:`__call__`; see the class docstring for why that is
        #: the contract rather than a surprise.
        self.stats: dict[str, Any] = {
            "threshold_pct": int(threshold),
            "threshold_source": source,
            "lead_min": float(options.lead_min),
            "rearm_after_min": float(options.rearm_after_min),
            "persistence_obs": float(options.persistence_obs),
            "probability": self.probability,
            "probability_column": self.column,
            # The core report's own word for it, imported rather than
            # typed out: the page's methods block and this module must
            # not be able to disagree about what happened.
            "scored": SCORED_RE_DECIDED,
            "rows_loaded": 0,
            "rows_matched": 0,
            "rows_fallback": 0,
            "warnings": 0,
            "stations": 0,
            "error": None,
        }

    # -- the hook ----------------------------------------------------------

    def __call__(
        self, rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, list[tuple[datetime, float | None, float | None]]]:
        try:
            return self._decide(rows)
        except Exception as exc:  # noqa: BLE001 — see the class docstring
            self.stats["error"] = f"{type(exc).__name__}: {exc}"
            self._say(
                "served rule: re-deciding the scoreboard failed "
                f"({self.stats['error']}); no warnings scored",
            )
            return {}

    # -- the work ----------------------------------------------------------

    def _decide(
        self, rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, list[tuple[datetime, float | None, float | None]]]:
        from .threshold_sweep import build_tracks, load_decisions, replay_station

        keys = {
            (row.get("radar_ts"), str(row.get("station_id"))) for row in rows
        }
        if not keys or not self.options.decisions_dirs:
            return {}

        lead = int(self.options.lead_min)
        curve_column = p_rain_column(lead)
        filler = self._filler()
        wide, _leads, counts = load_decisions(
            [Path(d) for d in self.options.decisions_dirs],
            leads_min=(lead,),
            # The engine's column travels beside the shared schema, which
            # would otherwise drop a name it does not know — the same
            # arrangement ``threshold_sweep.run_fit`` uses. Asked for
            # whenever it is not the curve's own, filler or no filler: a
            # live partition written since the serving path shipped
            # STORES it, and dropping that would push those rows onto the
            # fallback for no reason.
            extra_columns=() if self.column == curve_column else (self.column,),
            derive=filler,
            log=self._log,
        )
        self.stats["rows_loaded"] = len(wide)
        if filler is not None:
            self.stats["fill"] = {
                key: int(value) for key, value in filler.counts.items()
            }

        # The report's row set, and only it: both halves of the page must
        # be about one population.
        kept = [
            row for row in wide
            if (row.get("radar_ts"), str(row.get("station_id"))) in keys
        ]
        del wide
        self.stats["rows_matched"] = len(kept)
        if not kept:
            self._say(
                "served rule: none of the loaded rows is in the report's "
                f"set of {len(keys)} — nothing to re-decide"
            )
            return {}

        # ``Observation.p_decision``, applied as a column: a row the model
        # could not speak for decides on the curve, exactly as the service
        # does, instead of being skipped as "no probability at this lead".
        if self.column != curve_column:
            fallbacks = 0
            for row in kept:
                if row.get(self.column) is None and row.get(curve_column) is not None:
                    row[self.column] = row[curve_column]
                    fallbacks += 1
            self.stats["rows_fallback"] = fallbacks

        tracks, _frames = build_tracks(
            kept, [lead],
            coverage_gap_min=int(self.options.coverage_gap_min),
            column_for=lambda _lead: self.column,
        )
        del kept
        out: dict[str, list[tuple[datetime, float | None, float | None]]] = {}
        total = 0
        for station, track in tracks.items():
            warnings = replay_station(
                track, 0, self.threshold_pct,
                persistence_obs=int(self.options.persistence_obs),
                rearm_after_min=int(self.options.rearm_after_min),
                raining_now_mm_h=float(self.options.raining_now_mm_h),
                raining_now_eta_min=float(self.options.raining_now_eta_min),
                with_probability=True,
            )
            if warnings:
                out[str(station)] = warnings
                total += len(warnings)
        self.stats["warnings"] = total
        self.stats["stations"] = len(tracks)
        self._say(
            f"served rule: {self.threshold_pct} % ({self.threshold_source}) "
            f"at {lead} min on {self.column}; "
            f"{counts['files']} file(s), {self.stats['rows_loaded']} row(s) "
            f"loaded, {self.stats['rows_matched']} matched the report, "
            f"{self.stats['rows_fallback']} fell back to the curve, "
            f"{total} warning(s) over {len(tracks)} station(s)"
        )
        return out

    def _filler(self) -> Any:
        """The ``ProbabilityFiller`` for this run, or ``None``.

        ``None`` whenever there is nothing to fill WITH — the curve column
        is what the service decides on, or no model path was given — in
        which case the read carries whatever the rows already hold, which
        is what the page showed before the model existed.

        Built exactly as ``gauge_reliability._filler`` builds it, with one
        difference stated in :class:`~dmi_nowcast_sidecar.postprocess_fit.
        ProbabilityFiller`: rows it cannot fill are KEPT, because the
        engine's per-row fallback has an answer for them and dropping them
        would silence warnings the service would have sent.
        """
        if (
            self.probability != PROBABILITY_POSTPROCESS
            or self.options.postprocess_model is None
        ):
            return None
        from dmi_nowcast_core.postprocess import PostprocessModel

        from .postprocess_fit import ProbabilityFiller

        model = None
        try:
            model = PostprocessModel.loads(
                Path(self.options.postprocess_model).read_text(
                    encoding="utf-8",
                ),
            )
        except Exception as exc:  # noqa: BLE001 — every way a file can be junk
            self._say(
                f"served rule: cannot read {self.options.postprocess_model} "
                f"({type(exc).__name__}: {exc}); re-deciding on the rows that "
                "already carry the column, with the curve behind them"
            )
        return ProbabilityFiller(
            model,
            (int(self.options.lead_min),),
            tuple(int(lead) for lead in self.options.design_leads),
            lambda _lead: self.column,
            drop_unfilled=False,
        )

    def _say(self, message: str) -> None:
        if self._log:
            self._log(message)


def decider_for(
    options: ServedRuleOptions, *, log: Callable[[str], None] | None = None,
) -> ServedRuleDecider:
    """A :class:`ServedRuleDecider`, for callers that want one expression."""
    return ServedRuleDecider(options, log=log)


__all__ = [
    "PROBABILITY_CURVE",
    "PROBABILITY_POSTPROCESS",
    "SOURCE_CONFIG",
    "SOURCE_FALLBACK",
    "SOURCE_TABLE",
    "ServedRuleDecider",
    "ServedRuleOptions",
    "decider_for",
    "post_column",
    "resolve_threshold",
]
