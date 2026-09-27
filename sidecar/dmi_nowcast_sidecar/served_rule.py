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

* the dedup of ``threshold_sweep.load_decisions`` — one row per
  ``(radar_ts, station_id)``, the later directory winning — applied here
  to numpy columns rather than to a dict per row;
* :class:`~dmi_nowcast_sidecar.postprocess_fit.ProbabilityFiller` fills
  ``p_post_<lead>`` (and ``p_onset_<lead>``) ONLY on rows that store none
  and carry every column the model's design reads; a row that lacks one
  (the v1 replay tree has no ``g_*`` / ``ng_*`` block) keeps a null and
  follows the engine's own fallback, exactly as live;
* the coverage-run index ``threshold_sweep.build_tracks`` assigns;
* :func:`~dmi_nowcast_sidecar.threshold_sweep.replay_station` runs
  ``push.engine.evaluate`` itself.

The rule's timing is borrowed too: persistence is ONE observation and the
re-arm 60 minutes, read from ``push.persistence_obs`` /
``push.rearm_after_min`` by ``quality_report._served_rule_options`` and
defaulted in :mod:`dmi_nowcast_core.push_rules`. Nothing here is allowed
to have its own opinion about either — the page scoring two observations
while the service fired on one is exactly the divergence DECIDE-14 closed
on 2026-09-13.

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
probability, all_clear_utc)]}``, the last element the instant the engine
retracted that warning with an all-clear (``None`` when it did not), which
the report grades right / wrong beside the scoreboard — plus a stats dict the report publishes under
``methods.subscriber_rule`` and the job logs. The core report cannot
import this module (it must import nothing from the sidecar), which is
why it takes the decision as a callable rather than a path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dmi_nowcast_core.push_rules import (
    DEFAULT_ALLCLEAR_ENABLED,
    DEFAULT_ALLCLEAR_READINGS,
    DEFAULT_PERSISTENCE_OBS,
    DEFAULT_REARM_AFTER_MIN,
)
from dmi_nowcast_core.push_thresholds import (
    effective_threshold,
    lead_pick,
    load_thresholds,
    onset_rule,
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
    #: The push-only ONSET model (S11, ``push.onset_model_path``), used to
    #: fill ``p_onset_<lead>`` the same way when the table puts the lead
    #: on the onset AND rule. None: only rows that stored it carry it, and
    #: the rest fall back to the single threshold, as the service does.
    onset_model: Path | None = None
    #: The leads the model's design reads. Must match the model's own.
    design_leads: tuple[int, ...] = DEFAULT_PRODUCT_LEADS_MIN
    #: The rest of the live subscriber row — the rule's timing from
    #: ``push.persistence_obs`` / ``push.rearm_after_min`` and the
    #: detection threshold from ``forecast.rain_threshold_mm_h`` — so the
    #: replay is the service's rule and not this module's idea of it. The
    #: defaults are the shipped numbers, in one place
    #: (:mod:`dmi_nowcast_core.push_rules`).
    persistence_obs: int = DEFAULT_PERSISTENCE_OBS
    rearm_after_min: int = DEFAULT_REARM_AFTER_MIN
    raining_now_mm_h: float = 0.5
    raining_now_eta_min: float = 1.5
    #: The all-clear, from ``push.allclear_enabled`` /
    #: ``push.allclear_readings`` — so the retractions the page grades are
    #: the ones the service sends.
    allclear_enabled: bool = DEFAULT_ALLCLEAR_ENABLED
    allclear_readings: int = DEFAULT_ALLCLEAR_READINGS
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


def resolve_onset_rule(options: ServedRuleOptions) -> tuple[int, int] | None:
    """The onset AND rule for this run's lead (S11), or None.

    Only under the post-processed source: the curve rollback never reads
    ``p_onset`` in the service, so every observation there is judged on the
    single threshold — which :func:`resolve_threshold` then returns.
    """
    if (
        options.thresholds_path is None
        or options.probability_source != PROBABILITY_POSTPROCESS
    ):
        return None
    return onset_rule(load_thresholds(options.thresholds_path), options.lead_min)


def resolve_threshold(options: ServedRuleOptions) -> tuple[int, str]:
    """``(percent, source)`` for this run's lead. Total, never raises.

    The document is read through the core's own helpers, so the page
    picks the number the running service would pick for the same horizon
    — including its fallback for a lead the fit could not speak for.

    Under the curve rollback a lead on the onset AND rule is graded at its
    ``single_threshold_pct``: the service passes the rule to the engine
    with no ``p_onset`` (``push/service.py``, ``station_eval``), and the
    engine then judges every observation on the single threshold. The
    table's own ``threshold_pct`` there is the p_post HALF of the AND rule
    (0 at 20 min) and was never a rule on its own.
    """
    doc = None
    if options.thresholds_path is not None:
        doc = load_thresholds(options.thresholds_path)
    if doc is None:
        return int(options.fallback_threshold_pct), SOURCE_CONFIG
    threshold = int(effective_threshold(doc, options.lead_min))
    if options.probability_source != PROBABILITY_POSTPROCESS:
        rule = onset_rule(doc, options.lead_min)
        if rule is not None:
            threshold = int(rule[1])
    source = (
        SOURCE_TABLE
        if lead_pick(doc, str(int(options.lead_min))) is not None
        else SOURCE_FALLBACK
    )
    return threshold, source


#: The columns every decision row contributes to the replay besides its
#: probabilities — exactly what ``build_tracks`` reads.
_TRACK_COLUMNS = ("eta_min", "intensity_mm_h", "observed_mm_h", "forecast_now_mm_h")

#: Where a row's model probability came from.
_STORED, _COMPUTED, _UNFILLED = 0, 1, 2


def _timestamps_us(column: Any) -> np.ndarray:
    """An Arrow timestamp column as int64 UTC microseconds (nulls → min)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    values = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    values = values.cast(pa.timestamp("us", tz="UTC")).cast(pa.int64())
    return np.asarray(
        pc.fill_null(values, np.iinfo(np.int64).min)
        .to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )


def _floats(table: Any, name: str) -> np.ndarray:
    import pyarrow as pa

    if name not in table.schema.names:
        return np.full(table.num_rows, np.nan, dtype=np.float64)
    return np.asarray(
        table.column(name).combine_chunks().cast(pa.float64())
        .to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )


def _optional(values: np.ndarray) -> list[float | None]:
    """NaN → ``None``, as ``threshold_sweep._opt_float`` reads a null."""
    return [None if v != v else v for v in values.tolist()]


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

    The read is columnar: per file only the columns the replay needs, cut
    to the report's keys before anything else happens, the feature
    columns read only for rows whose probability is not stored and only
    from a file that carries every column the model reads. Rows become
    Python objects once, as the replay's track tuples.
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
        #: S11: ``(onset_threshold_pct, single_threshold_pct)`` when the
        #: served table puts this lead on the onset AND rule — the page
        #: then grades that rule, never ``p_post`` alone against its half.
        self.onset = resolve_onset_rule(options)
        from dmi_nowcast_core.postprocess import ONSET_COLUMN_TEMPLATE

        self.onset_column = ONSET_COLUMN_TEMPLATE.format(
            lead=int(options.lead_min),
        )
        #: Published under ``methods.subscriber_rule`` and logged. Mutated
        #: by :meth:`__call__`; see the class docstring for why that is
        #: the contract rather than a surprise.
        self.stats: dict[str, Any] = {
            "threshold_pct": int(
                threshold if self.onset is None else self.onset[0]
            ),
            "threshold_source": source,
            "lead_min": float(options.lead_min),
            "rearm_after_min": float(options.rearm_after_min),
            "persistence_obs": float(options.persistence_obs),
            # 0 = the all-clear is off; otherwise the readings it takes.
            "allclear_readings": float(
                options.allclear_readings if options.allclear_enabled else 0
            ),
            "probability": self.probability,
            "probability_column": self.column,
            # S11: null on the single-threshold rule. On the onset AND
            # rule the page's headline ``threshold_pct`` is the onset
            # threshold — the same number ``/api/push/options`` shows
            # (``ThresholdTable.headline``) — and the p_post half, which
            # may be 0, travels as ``post_threshold_pct``.
            "onset_threshold_pct": (
                None if self.onset is None else int(self.onset[0])
            ),
            "post_threshold_pct": (
                None if self.onset is None else int(threshold)
            ),
            "rows_onset_fallback": 0,
            # The core report's own word for it, imported rather than
            # typed out: the page's methods block and this module must
            # not be able to disagree about what happened.
            "scored": SCORED_RE_DECIDED,
            "rows_loaded": 0,
            "rows_matched": 0,
            "rows_fallback": 0,
            # Where each scored row's model probability came from: stored
            # by the writer, computed here by the served model from the
            # row's features, or unfillable (the row lacks a column the
            # model reads) and therefore on the engine's fallback. Null
            # when the rule reads no such column.
            "rows_post_stored": None,
            "rows_post_computed": None,
            "rows_post_unfillable": None,
            "rows_onset_stored": None,
            "rows_onset_computed": None,
            "rows_onset_unfillable": None,
            "warnings": 0,
            "stations": 0,
            "error": None,
        }

    # -- the hook ----------------------------------------------------------

    def __call__(
        self, rows: Sequence[Mapping[str, Any]],
    ) -> dict[
        str, list[tuple[datetime, float | None, float | None, datetime | None]]
    ]:
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
    ) -> dict[
        str, list[tuple[datetime, float | None, float | None, datetime | None]]
    ]:
        from .threshold_sweep import replay_station

        if not rows or not self.options.decisions_dirs:
            return {}
        stations, keys = _report_keys(rows)
        if keys.size == 0:
            return {}

        lead = int(self.options.lead_min)
        post = self.column != p_rain_column(lead)
        filler = self._filler()
        onset_filler = self._onset_filler()
        table = self._read(stations, keys, filler, onset_filler)
        if filler is not None:
            self.stats["fill"] = {
                key: int(value) for key, value in filler.counts.items()
            }
        if onset_filler is not None:
            self.stats["onset_fill"] = {
                key: int(value) for key, value in onset_filler.counts.items()
            }
        n = int(table["code"].size)
        self.stats["rows_matched"] = n
        if n == 0:
            self._say(
                "served rule: none of the loaded rows is in the report's "
                f"set of {keys.size} — nothing to re-decide"
            )
            return {}

        curve = table["p_rain"]
        if post:
            for label, name in (("post", "p_post"), ("onset", "p_onset")):
                if name == "p_onset" and self.onset is None:
                    continue
                source = table[f"{name}_source"]
                self.stats[f"rows_{label}_stored"] = int((source == _STORED).sum())
                self.stats[f"rows_{label}_computed"] = int(
                    (source == _COMPUTED).sum(),
                )
                self.stats[f"rows_{label}_unfillable"] = int(
                    (source == _UNFILLED).sum(),
                )
            # ``Observation.p_decision``, applied as a column: a row the
            # model could not speak for decides on the curve, exactly as
            # the service does, instead of being skipped as "no
            # probability at this lead".
            fallback = ~np.isfinite(table["p_post"]) & np.isfinite(curve)
            self.stats["rows_fallback"] = int(fallback.sum())
            decision = np.where(fallback, curve, table["p_post"])
        else:
            decision = curve
        onset = None
        if self.onset is not None:
            onset = table["p_onset"]
            self.stats["rows_onset_fallback"] = int(
                (~np.isfinite(onset) & np.isfinite(decision)).sum(),
            )

        out: dict[
            str,
            list[tuple[datetime, float | None, float | None, datetime | None]],
        ] = {}
        total = 0
        tracks = 0
        for code, track in self._tracks(table, decision, onset):
            tracks += 1
            warnings = replay_station(
                track, 0, self.threshold_pct,
                persistence_obs=int(self.options.persistence_obs),
                rearm_after_min=int(self.options.rearm_after_min),
                raining_now_mm_h=float(self.options.raining_now_mm_h),
                raining_now_eta_min=float(self.options.raining_now_eta_min),
                with_probability=True,
                with_all_clear=True,
                allclear_enabled=bool(self.options.allclear_enabled),
                allclear_readings=int(self.options.allclear_readings),
                onset_threshold_pct=(
                    None if self.onset is None else int(self.onset[0])
                ),
                single_threshold_pct=(
                    None if self.onset is None else int(self.onset[1])
                ),
            )
            if warnings:
                out[stations[code]] = warnings
                total += len(warnings)
        self.stats["warnings"] = total
        self.stats["stations"] = tracks
        rule = (
            "" if self.onset is None
            else f" AND {self.onset_column} >= {self.onset[0]} % "
            f"(single {self.onset[1]} % without it, "
            f"{self.stats['rows_onset_fallback']} row(s))"
        )
        sources = ""
        if post:
            sources = (
                f"; {self.column}: {self.stats['rows_post_stored']} stored, "
                f"{self.stats['rows_post_computed']} computed, "
                f"{self.stats['rows_post_unfillable']} unfillable"
            )
            if self.onset is not None:
                sources += (
                    f"; {self.onset_column}: "
                    f"{self.stats['rows_onset_stored']} stored, "
                    f"{self.stats['rows_onset_computed']} computed, "
                    f"{self.stats['rows_onset_unfillable']} unfillable"
                )
        self._say(
            f"served rule: {self.threshold_pct} % ({self.threshold_source}) "
            f"at {lead} min on {self.column}{rule}; "
            f"{self.stats['files']} file(s), {self.stats['rows_loaded']} row(s) "
            f"loaded, {self.stats['rows_matched']} matched the report, "
            f"{self.stats['rows_fallback']} fell back to the curve{sources}, "
            f"{total} warning(s) over {tracks} station(s)"
        )
        return out

    def _read(
        self,
        stations: list[str],
        keys: np.ndarray,
        filler: Any,
        onset_filler: Any,
    ) -> dict[str, np.ndarray]:
        """The decision rows in the report's key set, as numpy columns.

        One row per ``(radar_ts, station_id)``, the LATER directory (and,
        within one, the later file) winning a tie — the rule
        ``threshold_sweep.load_decisions`` applies — sorted by station and
        frame, which is the order the replay walks.
        """
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        from .threshold_sweep import decision_parquets

        lead = int(self.options.lead_min)
        curve = p_rain_column(lead)
        post = self.column != curve
        onset = self.onset is not None
        value_set = pa.array(stations, pa.string())
        stride = len(stations) + 1
        wanted = ["radar_ts", "generated_at", "station_id", *_TRACK_COLUMNS, curve]
        if post:
            wanted.append(self.column)
        if onset:
            wanted.append(self.onset_column)
        fillers = [
            (f, name) for f, name in (
                (filler, self.column), (onset_filler, self.onset_column),
            ) if f is not None
        ]
        parts: dict[str, list[np.ndarray]] = {}
        files = rows_loaded = 0
        for directory in self.options.decisions_dirs:
            for path in decision_parquets(Path(directory)):
                try:
                    names = set(pq.read_schema(path).names)
                except Exception as exc:  # noqa: BLE001 — one unreadable file
                    self._say(f"served rule: skipping {path.name}: {exc}")
                    continue
                if not {"radar_ts", "station_id", "action"} <= names:
                    continue
                try:
                    table = pq.read_table(
                        path, columns=[c for c in wanted if c in names],
                    )
                except Exception as exc:  # noqa: BLE001 — one unreadable file
                    self._say(f"served rule: skipping {path.name}: {exc}")
                    continue
                files += 1
                rows_loaded += table.num_rows
                codes = np.asarray(
                    pc.fill_null(
                        pc.index_in(
                            table.column("station_id").cast(pa.string()),
                            value_set=value_set,
                        ),
                        -1,
                    ).to_numpy(zero_copy_only=False),
                    dtype=np.int64,
                )
                radar = _timestamps_us(table.column("radar_ts"))
                probe = radar * stride + np.where(codes < 0, stride - 1, codes)
                where = np.clip(np.searchsorted(keys, probe), 0, keys.size - 1)
                hit = (codes >= 0) & (keys[where] == probe)
                if not hit.any():
                    continue
                index = np.flatnonzero(hit)
                table = table.take(pa.array(index))
                # The feature columns, for the rows the model may have to
                # speak for — only from a file that has every one of them.
                extra: list[str] = []
                for f, name in fillers:
                    if (
                        f.model is not None
                        and set(f.source_columns) <= names
                        and not np.isfinite(_floats(table, name)).all()
                    ):
                        extra += [
                            c for c in f.source_columns
                            if c not in table.schema.names and c not in extra
                        ]
                if extra:
                    features = pq.read_table(path, columns=extra).take(
                        pa.array(index),
                    )
                    for column in extra:
                        table = table.append_column(
                            column, features.column(column),
                        )
                    del features
                block: dict[str, np.ndarray] = {
                    "code": codes[index],
                    "radar": radar[index],
                }
                generated = (
                    _timestamps_us(table.column("generated_at"))
                    if "generated_at" in table.schema.names
                    else np.full(index.size, np.iinfo(np.int64).min)
                )
                block["generated"] = np.where(
                    generated == np.iinfo(np.int64).min, block["radar"], generated,
                )
                for column in _TRACK_COLUMNS:
                    block[column] = _floats(table, column)
                block["p_rain"] = _floats(table, curve)
                for f, name, key in (
                    (filler, self.column, "p_post"),
                    (onset_filler, self.onset_column, "p_onset"),
                ):
                    if key == "p_post" and not post:
                        continue
                    if key == "p_onset" and not onset:
                        continue
                    before = np.isfinite(_floats(table, name))
                    if f is not None:
                        table = f(table)
                    values = _floats(table, name)
                    block[key] = values
                    block[f"{key}_source"] = np.where(
                        before, _STORED,
                        np.where(np.isfinite(values), _COMPUTED, _UNFILLED),
                    ).astype(np.int8)
                for key, values in block.items():
                    parts.setdefault(key, []).append(values)
                del table
        self.stats["rows_loaded"] = rows_loaded
        self.stats["files"] = files
        if not parts:
            return {"code": np.zeros(0, dtype=np.int64)}
        merged = {key: np.concatenate(values) for key, values in parts.items()}
        del parts
        # Last read wins a key, then (station, frame) order: a stable sort
        # on the arrival index inside each key keeps the latest last.
        arrival = np.arange(merged["code"].size)
        order = np.lexsort((arrival, merged["radar"], merged["code"]))
        code, radar = merged["code"][order], merged["radar"][order]
        last = np.ones(order.size, dtype=bool)
        last[:-1] = (code[1:] != code[:-1]) | (radar[1:] != radar[:-1])
        order = order[last]
        # Column by column, so only one extra column is alive at a time.
        for key in list(merged):
            merged[key] = merged[key][order]
        return merged

    def _tracks(
        self,
        table: Mapping[str, np.ndarray],
        decision: np.ndarray,
        onset: np.ndarray | None,
    ):
        """``(station code, track)`` per station, in the replay's record shape.

        The coverage-run index is assigned over EVERY frame, as
        ``threshold_sweep.build_tracks`` does; the rows with no decision
        probability are then left out, because ``replay_station`` passes
        over them without touching the state anyway.
        """
        import pyarrow as pa

        gap_us = int(self.options.coverage_gap_min) * 60_000_000
        code = table["code"]
        bounds = np.flatnonzero(np.diff(code)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [code.size]))
        stamp = pa.timestamp("us", tz="UTC")
        for start, end in zip(starts.tolist(), ends.tolist()):
            radar = table["radar"][start:end]
            run = np.concatenate(
                ([0], np.cumsum(np.diff(radar) > gap_us)),
            )
            keep = np.flatnonzero(np.isfinite(decision[start:end])) + start
            if keep.size == 0:
                yield int(code[start]), []
                continue
            radar_dt = pa.array(table["radar"][keep], stamp).to_pylist()
            generated_dt = pa.array(table["generated"][keep], stamp).to_pylist()
            columns = [
                _optional(table[column][keep]) for column in _TRACK_COLUMNS
            ]
            runs = run[keep - start].tolist()
            probabilities = [(p,) for p in decision[keep].tolist()]
            parts = [
                radar_dt, generated_dt, *columns, runs, probabilities,
            ]
            if onset is not None:
                parts.append([(o,) for o in _optional(onset[keep])])
            yield int(code[start]), list(zip(*parts))

    def _load_model(self, path: Path | None, what: str, fallback: str) -> Any:
        from dmi_nowcast_core.postprocess import PostprocessModel

        try:
            return PostprocessModel.loads(
                Path(path).read_text(encoding="utf-8"),
            )
        except Exception as exc:  # noqa: BLE001 — every way a file can be junk
            self._say(
                f"served rule: cannot read {what} {path} "
                f"({type(exc).__name__}: {exc}); {fallback}"
            )
            return None

    def _filler(self) -> Any:
        """The ``ProbabilityFiller`` for ``p_post_<lead>``, or ``None``.

        ``None`` whenever there is nothing to fill WITH — the curve column
        is what the service decides on, or no model path was given — in
        which case the read carries whatever the rows already hold.

        Rows it cannot fill are KEPT with a null: the engine's per-row
        fallback has an answer for them (the curve), and dropping them
        would silence warnings the service would have sent.
        """
        if (
            self.probability != PROBABILITY_POSTPROCESS
            or self.options.postprocess_model is None
        ):
            return None
        from .postprocess_fit import ProbabilityFiller

        model = self._load_model(
            self.options.postprocess_model, "model",
            "re-deciding on the rows that already carry the column, with "
            "the curve behind them",
        )
        return ProbabilityFiller(
            model,
            (int(self.options.lead_min),),
            tuple(int(lead) for lead in self.options.design_leads),
            lambda _lead: self.column,
            drop_unfilled=False,
        )

    def _onset_filler(self) -> Any:
        """The ``ProbabilityFiller`` for ``p_onset_<lead>``, or ``None``.

        Only when this lead is on the onset AND rule and an onset model
        path was given; rows it cannot fill keep a null and fall back to
        the single threshold in the replay, as the service does.
        """
        if self.onset is None or self.options.onset_model is None:
            return None
        from .postprocess_fit import ProbabilityFiller

        model = self._load_model(
            self.options.onset_model, "onset model",
            f"rows without a stored {self.onset_column} fall back to the "
            "single threshold",
        )
        return ProbabilityFiller(
            model,
            (int(self.options.lead_min),),
            tuple(int(lead) for lead in self.options.design_leads),
            lambda _lead: self.onset_column,
            drop_unfilled=False,
        )

    def _say(self, message: str) -> None:
        if self._log:
            self._log(message)


def _report_keys(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], np.ndarray]:
    """``(station ids, sorted unique int64 keys)`` of the report's rows.

    A key is ``radar_ts`` in UTC microseconds times ``len(stations) + 1``
    plus the station's index — one integer per ``(radar_ts, station_id)``,
    so membership is a binary search rather than a set of tuples.
    """
    import pyarrow as pa

    columns = getattr(rows, "columns", None)
    strings = getattr(rows, "strings", None)
    if (
        isinstance(columns, Mapping) and isinstance(strings, Mapping)
        and "radar_ts" in columns and "station_id" in strings
    ):
        # The core report's ``DecisionRows``: the key columns are already
        # numpy — int64 UTC microseconds (null = int64 min) and station
        # codes into a list of distinct ids.
        codes, values = strings["station_id"]
        names = [str(v) for v in values]
        used = np.unique(codes).tolist()
        stations = sorted({names[c] for c in used})
        position = {name: i for i, name in enumerate(stations)}
        index = np.array(
            [position.get(name, 0) for name in names], dtype=np.int64,
        )
        radar = np.asarray(columns["radar_ts"], dtype=np.int64)
        valid = radar != np.iinfo(np.int64).min
        keys = radar[valid] * (len(stations) + 1) + index[codes[valid]]
        return stations, np.unique(keys)

    ids = [str(row.get("station_id")) for row in rows]
    stations = sorted(set(ids))
    lookup = {station: index for index, station in enumerate(stations)}
    codes = np.fromiter(
        (lookup[s] for s in ids), dtype=np.int64, count=len(ids),
    )
    del ids
    radar = _timestamps_us(pa.array(
        [row.get("radar_ts") for row in rows], pa.timestamp("us", tz="UTC"),
    ))
    valid = radar != np.iinfo(np.int64).min
    keys = radar[valid] * (len(stations) + 1) + codes[valid]
    return stations, np.unique(keys)


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
    "resolve_onset_rule",
    "resolve_threshold",
]
