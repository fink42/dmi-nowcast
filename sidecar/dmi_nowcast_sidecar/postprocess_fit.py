"""The nightly post-processing refit (Phase H, H-P).

``scripts/fit_postprocess.py`` is the *study*: it fits the model, scores it
leave-one-(year, month)-out against the served curve on the same rows, and
writes the report that argued for shipping it. This module is the
*production* half of the same fit — the part that has to run on the VM,
inside the nightly quality job, in a container that does not ship
``scripts/``.

What is deliberately the same
-----------------------------
Everything that decides *what is being fitted*: the rows
(:func:`decision_rows.load_probabilities` — one row per
``(radar_ts, station_id)``, the later directory winning), the outcome
(:class:`decision_rows.GaugeGrid` — the gauge wet at some point in
``(t, t + L]``, ungradable where the gauge said nothing), the dead-gauge
exclusion, and the transform and the fit itself
(``dmi_nowcast_core.postprocess``). A second opinion about any of those
would mean the model in service was not the model the report measured.

What is deliberately different
------------------------------
**No leave-one-month-out.** The fit here is on all rows. The out-of-fold
evidence is the study's job and it is already on the record
(``archive/l3_and_postprocess_20260911/``): ΔBSS +0.14…+0.19 at every
lead and every season. Running ten folds nightly would cost ten times the
fit to produce a number nothing in the service gates on, on a VM whose
batch budget is already shared with the threshold sweep that runs
immediately after this. What the nightly job needs is a model that has
seen the most recent month; what it must not do is spend an hour proving
again what April's fold already proved.

The order matters. This runs BEFORE the threshold sweep, because the
sweep has to be fitted on the probability the engine will decide with —
see ``threshold_sweep.SweepOptions.probability_column``.

Failure policy, as everywhere else in the nightly job: every way this can
fail leaves the model already in service exactly where it is, and costs
one log line.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dmi_nowcast_core import postprocess as pp
from dmi_nowcast_core.warning_score import (
    DEFAULT_DRY_MIN,
    DEFAULT_MIN_KNOWN_SLOTS,
    DEFAULT_ONSET_MIN_MM,
)

from .decision_rows import (
    DAY_SEC,
    build_gauge_grid,
    decision_window,
    load_probabilities,
    recode_stations,
)
from .threshold_sweep import SweepError

#: Fields of :class:`PostprocessFitOptions` that carry a list of paths.
_PATH_LIST_FIELDS = frozenset({"decisions_dirs"})
#: Fields that carry a single path.
_PATH_FIELDS = frozenset({"corpus_dir", "out"})
#: Fields that are tuples of ints on the way in and lists on the way out.
_TUPLE_FIELDS = frozenset({"leads", "design_leads"})


@dataclass(frozen=True)
class PostprocessFitOptions:
    """Everything :func:`run_postprocess_fit` needs. The config, as a value.

    Crosses a process boundary as JSON (the nightly build runs in a child),
    so every field is a plain type — see :func:`options_to_json`.
    """

    decisions_dirs: Sequence[Path]
    corpus_dir: Path
    out: Path
    #: The horizons to fit. The subscribable ones: fitting a lead nobody
    #: can choose spends CPU on a model nothing reads.
    leads: Sequence[int] = (20, 30, 45, 60)
    #: The leads whose raw ensemble fraction enters the design — every
    #: lead the national products publish, because the SHAPE of the
    #: fraction against lead is itself a predictor.
    design_leads: Sequence[int] = (10, 20, 30, 45, 60)
    #: Ridge strength on the slopes; the intercept is never penalised.
    l2: float = 1.0
    isotonic_bins: int = pp.DEFAULT_ISOTONIC_BINS
    #: The gauge onset/wet definition and the dead-gauge rule, matching
    #: ``fit_thresholds`` so the model and the thresholds stand on the
    #: same rows.
    dry_min: int = DEFAULT_DRY_MIN
    onset_min_mm: float = DEFAULT_ONSET_MIN_MM
    min_known_slots: int = DEFAULT_MIN_KNOWN_SLOTS
    #: Refuse to publish a fit that stands on less than this. A model
    #: fitted on a handful of rows would be served to every subscriber.
    min_rows: int = 1000
    #: Extra provenance to put in the document's ``training`` block.
    training: dict[str, Any] = field(default_factory=dict)


def options_to_json(options: PostprocessFitOptions) -> dict:
    """A :class:`PostprocessFitOptions` as plain JSON types."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(options):
        value = getattr(options, f.name)
        if f.name in _PATH_LIST_FIELDS:
            out[f.name] = [str(p) for p in value]
        elif f.name in _PATH_FIELDS:
            out[f.name] = None if value is None else str(value)
        elif isinstance(value, (list, tuple)):
            out[f.name] = list(value)
        else:
            out[f.name] = value
    return out


def options_from_json(payload: dict) -> PostprocessFitOptions:
    """The :class:`PostprocessFitOptions` :func:`options_to_json` encoded.

    Unknown keys are ignored rather than raised on: a newer parent and an
    older child must not take the nightly build down between a deploy and
    a container restart.
    """
    known = {f.name for f in dataclasses.fields(PostprocessFitOptions)}
    kwargs: dict[str, Any] = {}
    for name, value in (payload or {}).items():
        if name not in known:
            continue
        if name in _PATH_LIST_FIELDS:
            kwargs[name] = [Path(p) for p in (value or ())]
        elif name in _PATH_FIELDS:
            kwargs[name] = None if value in (None, "") else Path(value)
        elif name in _TUPLE_FIELDS:
            kwargs[name] = tuple(int(v) for v in (value or ()))
        else:
            kwargs[name] = value
    return PostprocessFitOptions(**kwargs)


def featured_mask(
    rows: dict, design_leads: Sequence[int],
) -> np.ndarray:
    """Which rows carry post-processing features at all.

    A row written before the features existed — the pre-H-P replay tree,
    a live partition from before this deploy — has nulls in every one of
    them. The design would happily impute the training mean for each and
    produce a number, which is the failure mode this guards: a prediction
    that looks like a forecast and is actually the base rate.

    Shared columns (``observed_mm_h``, ``eta_min``, ``intensity_mm_h``)
    are not evidence either way — every decision row has always had them.
    """
    names = [
        name for name in pp.feature_only_columns(design_leads)
        if name in rows["extra"]
    ]
    if not names:
        return np.zeros(int(rows["t"].size), dtype=bool)
    keep = np.zeros(int(rows["t"].size), dtype=bool)
    for name in names:
        keep |= np.isfinite(rows["extra"][name])
    return keep


def filter_rows(rows: dict, keep: np.ndarray) -> dict:
    """Subset every column of a loaded row set in place, and return it.

    A boolean subset rather than a filter at read time: the dedup, the
    station coding and the window all happen over the full read, and
    narrowing afterwards keeps the counts this reports honest about what
    was on disk versus what was fitted on.
    """
    rows["t"] = rows["t"][keep]
    rows["radar_ts"] = rows["radar_ts"][keep]
    rows["station"] = rows["station"][keep]
    rows["p"] = {lead: values[keep] for lead, values in rows["p"].items()}
    rows["extra"] = {
        name: values[keep] for name, values in rows["extra"].items()
    }
    rows["rows"] = int(keep.sum())
    return rows


def build_features(rows: dict) -> dict[str, Any]:
    """The model's input columns: the stored ones plus season and hour.

    Both derived columns come from ``t`` — ``generated_at``, the decision
    instant — which is the stamp the outcome window is anchored on and the
    one the writers stamped their ``season`` column from.
    """
    features: dict[str, Any] = dict(rows["extra"])
    features["season"] = pp.seasons_from_epoch(rows["t"])
    features["hour_utc"] = pp.hours_from_epoch(rows["t"]).astype(np.float64)
    return features


def run_postprocess_fit(
    options: PostprocessFitOptions, *, log=None,
) -> dict:
    """Fit the model on the configured rows and return it with its counts.

    Returns ``{"model": PostprocessModel, "summary": {...}}``. Writing is
    the caller's job (:func:`dmi_nowcast_sidecar.quality_job.run_postprocess_fit`
    does it atomically), so a test can exercise the fit without a file.

    Raises :class:`~dmi_nowcast_sidecar.threshold_sweep.SweepError` when
    there is nothing to fit on — no rows, no featured rows, no gauge
    truth. That is not a bug and not worth a warning every night while a
    corpus is still filling up.
    """
    leads = tuple(sorted({int(lead) for lead in options.leads}))
    design_leads = tuple(sorted({int(lead) for lead in options.design_leads}))
    if not leads:
        raise SweepError("no leads to fit")

    rows = load_probabilities(
        options.decisions_dirs, leads,
        extra_columns=pp.feature_source_columns(design_leads),
        log=log,
    )
    on_disk = int(rows["rows"])
    keep = featured_mask(rows, design_leads)
    skipped = on_disk - int(keep.sum())
    if not keep.any():
        raise SweepError(
            "no decision row carries post-processing features — the rows "
            "predate them, or the cycle never wrote them",
        )
    rows = filter_rows(rows, keep)
    if log:
        log(
            f"{rows['rows']} featured row(s) of {on_disk} "
            f"({skipped} without features, skipped)"
        )
    if rows["rows"] < int(options.min_rows):
        raise SweepError(
            f"only {rows['rows']} featured row(s), below the "
            f"{options.min_rows}-row floor",
        )

    window = decision_window(rows["t"])
    grid, dead_rows, scored = build_gauge_grid(
        Path(options.corpus_dir), sorted(rows["stations"]), window,
        dry_min=int(options.dry_min),
        onset_min_mm=float(options.onset_min_mm),
        min_known_slots=int(options.min_known_slots),
        log=log,
    )
    # A station the grid does not carry cannot verify anything; its rows
    # are marked and then forced ungradable below rather than silently
    # scored against another station's slots.
    recode_stations(rows, scored)
    dropped = np.asarray(
        rows.get("dropped", np.zeros(rows["t"].size, dtype=bool)), dtype=bool,
    )
    truth: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    gradable: dict[int, int] = {}
    for lead in leads:
        y, usable = grid.outcome(rows["t"], rows["station"], int(lead))
        usable = usable & ~dropped
        truth[int(lead)] = (y, usable)
        gradable[int(lead)] = int(usable.sum())
    del grid
    if not any(gradable.values()):
        raise SweepError("no row is gradable against the gauge store")
    unfittable = [lead for lead in leads if gradable[lead] == 0]
    if unfittable:
        # A lead with no gradable row would raise inside ``fit_postprocess``
        # and cost the whole model; drop it and say so.
        if log:
            log(
                "no gradable row at lead(s) "
                + ", ".join(str(lead) for lead in unfittable)
                + " — not fitted"
            )
        leads = tuple(lead for lead in leads if gradable[lead] > 0)
        truth = {lead: truth[lead] for lead in leads}

    day = rows["t"] // DAY_SEC
    features = build_features(rows)
    if log:
        log(f"fitting {len(leads)} lead(s) over {rows['rows']} row(s)")
    model = pp.fit_postprocess(
        features, truth, leads,
        l2=float(options.l2),
        design_leads=design_leads,
        isotonic_bins=int(options.isotonic_bins),
        training={
            "from": window[0].isoformat(),
            "to": window[1].isoformat(),
            "rows": int(rows["rows"]),
            "rows_on_disk": on_disk,
            "rows_without_features": skipped,
            "days": int(np.unique(day).size),
            "stations": len(scored),
            "decisions_dirs": [str(d) for d in options.decisions_dirs],
            "corpus_dir": str(options.corpus_dir),
            "dead_gauges": [row["station_id"] for row in dead_rows],
            # In-sample by construction — see the module docstring. Said
            # in the artefact so nobody reads a base rate here as a score.
            "held_out": False,
            "fitted_by": "dmi_nowcast_sidecar.postprocess_fit",
            **dict(options.training),
        },
    )
    summary = {
        "leads": list(leads),
        "design_leads": list(design_leads),
        "rows": int(rows["rows"]),
        "rows_on_disk": on_disk,
        "rows_without_features": skipped,
        "stations": len(scored),
        "days": int(np.unique(day).size),
        "gradable": {str(lead): gradable[lead] for lead in leads},
        "base_rate": {
            str(lead): round(float(model.models[lead].base_rate), 6)
            for lead in leads if lead in model.models
        },
        "dead_gauges": [row["station_id"] for row in dead_rows],
        "window": {"from": window[0].isoformat(), "to": window[1].isoformat()},
        "fitted_at_utc": model.fitted_at_utc,
        "held_out": False,
    }
    return {"model": model, "summary": summary}


class ProbabilityFiller:
    """Fills ``p_post_<lead>`` on a decision table, and drops what it cannot.

    The nightly threshold sweep has to be fitted on the SAME probability
    the engine decides with, and the rows it reads are a mix: the replay
    tree (features, no ``p_post``), live partitions written since this
    shipped (both), and anything older (neither). This class makes one
    table of them, applied per file inside
    :func:`threshold_sweep.load_decisions` — before the rows become Python
    dicts, so the twenty feature columns are numpy for their whole life
    and never a hundred million float objects.

    The rule, per row:

    * a stored value wins — it is what the engine actually used;
    * otherwise, if the row carries features, the freshly fitted model
      speaks for it (and may honestly answer "unknown" for a point off
      coverage, exactly as a null ``p_rain`` does);
    * otherwise the row is DROPPED and counted. Scoring it would mean
      calling every frame of it below threshold, which is not a
      measurement of the rule, it is a measurement of the archive's depth.
    """

    def __init__(
        self,
        model: Any,
        leads: Sequence[int],
        design_leads: Sequence[int],
        column_for: Any,
    ) -> None:
        self.model = model
        self.leads = tuple(int(lead) for lead in leads)
        self.design_leads = tuple(int(lead) for lead in design_leads)
        self.column_for = column_for
        self.counts: dict[str, int] = {
            "rows": 0, "stored": 0, "computed": 0, "dropped": 0,
        }

    def __call__(self, table: Any) -> Any:
        import pyarrow as pa

        names = set(table.schema.names)
        n = table.num_rows
        self.counts["rows"] += n
        stored: dict[int, np.ndarray] = {}
        for lead in self.leads:
            name = self.column_for(lead)
            stored[lead] = (
                np.asarray(
                    table.column(name).combine_chunks()
                    .cast(pa.float64()).to_numpy(zero_copy_only=False),
                    dtype=np.float64,
                )
                if name in names
                else np.full(n, np.nan, dtype=np.float64)
            )
        featured = self._featured(table, names)
        has_stored = np.zeros(n, dtype=bool)
        for values in stored.values():
            has_stored |= np.isfinite(values)
        self.counts["stored"] += int(has_stored.sum())

        if self.model is not None and featured.any():
            predicted = self._predict(table, names)
            for lead, values in predicted.items():
                if lead not in stored:
                    continue
                fill = featured & ~np.isfinite(stored[lead])
                stored[lead] = np.where(fill, values, stored[lead])
                self.counts["computed"] += int(
                    (fill & np.isfinite(values)).sum(),
                )

        keep = featured | has_stored
        self.counts["dropped"] += int(n - keep.sum())
        for lead in self.leads:
            name = self.column_for(lead)
            if name in names:
                table = table.drop_columns([name])
            values = stored[lead]
            # A null, not a NaN: "the model could not speak for this row"
            # has one spelling in a decision parquet, and every reader
            # already knows it.
            table = table.append_column(name, pa.array(
                values, type=pa.float64(), mask=~np.isfinite(values),
            ))
        if not keep.all():
            table = table.filter(pa.array(keep))
        return table

    def _featured(self, table: Any, names: set) -> np.ndarray:
        import pyarrow as pa

        n = table.num_rows
        out = np.zeros(n, dtype=bool)
        for name in pp.feature_only_columns(self.design_leads):
            if name not in names:
                continue
            values = np.asarray(
                table.column(name).combine_chunks()
                .cast(pa.float64()).to_numpy(zero_copy_only=False),
                dtype=np.float64,
            )
            out |= np.isfinite(values)
        return out

    def _predict(self, table: Any, names: set) -> dict[int, np.ndarray]:
        import pyarrow as pa
        import pyarrow.compute as pc

        n = table.num_rows
        features: dict[str, Any] = {}
        for name in pp.feature_source_columns(self.design_leads):
            features[name] = (
                np.asarray(
                    table.column(name).combine_chunks()
                    .cast(pa.float64()).to_numpy(zero_copy_only=False),
                    dtype=np.float64,
                )
                if name in names
                else np.full(n, np.nan, dtype=np.float64)
            )
        # ``season`` and ``hour_utc`` from the decision instant, the same
        # stamp the fit derived them from — never the stored ``season``
        # column, so a row written by a writer that did not have one
        # scores identically to one that did. A row written before
        # ``generated_at`` existed falls back to its frame stamp, exactly
        # as ``decision_rows.load_probabilities`` does.
        radar_ts = table.column("radar_ts").combine_chunks()
        stamp = (
            pc.if_else(
                pc.is_valid(table.column("generated_at").combine_chunks()),
                table.column("generated_at").combine_chunks(),
                radar_ts,
            )
            if "generated_at" in names else radar_ts
        )
        seconds = np.asarray(
            stamp.cast(pa.int64()).to_numpy(zero_copy_only=False),
            dtype=np.int64,
        ) // 1_000_000
        features["season"] = pp.seasons_from_epoch(seconds)
        features["hour_utc"] = pp.hours_from_epoch(seconds).astype(np.float64)
        try:
            return {
                int(lead): np.asarray(values, dtype=np.float64)
                for lead, values in self.model.predict(features).items()
            }
        except Exception:  # noqa: BLE001 — a model that cannot score these
            # rows leaves them on whatever they already carried.
            return {}


__all__ = [
    "PostprocessFitOptions",
    "ProbabilityFiller",
    "build_features",
    "featured_mask",
    "filter_rows",
    "options_from_json",
    "options_to_json",
    "run_postprocess_fit",
]
