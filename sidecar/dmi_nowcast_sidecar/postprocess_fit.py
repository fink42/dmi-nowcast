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

What it refuses to do
---------------------
Overwrite a model it could not have produced. Once an artefact fitted on
the workstation — a tree model, a v2 design, a random-point fit — is
installed into the served file, a nightly refit would quietly replace it
with tonight's pooled logistic and then refit the thresholds on the
replacement. :func:`refit_skip_reason` compares the served document's own
statement of what it is against this job's configuration and skips the
refit when the two cannot be the same model. The threshold sweep still
runs, on the probabilities the SERVED model produces — which is the whole
point of the order.

Failure policy, as everywhere else in the nightly job: every way this can
fail leaves the model already in service exactly where it is, and costs
one log line.
"""
from __future__ import annotations

import dataclasses
import json
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
    #: Which of ``postprocess.MODEL_KINDS`` to fit — ``"logistic"``,
    #: ``"logistic-shared"``, ``"trees"`` or ``"trees-shared"``.
    #: ``"logistic"`` is what shipped, and it is the default here for the
    #: reason it is the default everywhere: changing what the service
    #: serves is a decision, not a default.
    #:
    #: Either tree kind needs LightGBM, which the sidecar image
    #: deliberately does NOT carry (see
    #: ``dmi_nowcast_core.postprocess_trees``). Asking for one here fails
    #: this step with a clear ImportError and leaves last night's model in
    #: service — the failure policy of every other step in this job. A
    #: tree model reaches production by being fitted offline in
    #: ``.venv-fit`` and synced in, never by a nightly refit on the VM.
    model: str = pp.KIND_LOGISTIC
    #: ``"v1"`` (the 27 shipped columns) or ``"v2"`` (every catalogue
    #: column, the spline bases, the interactions).
    design: str = pp.DESIGN_V1
    #: Learn a per-station intercept offset under its own stronger ridge.
    station_offsets: bool = False
    #: ``"pooled"`` (one curve per lead) or ``"per-season"``.
    isotonic: str = pp.ISOTONIC_POOLED
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
    #: Manual override: never refit, whatever is on disk. The operator's
    #: switch for "this model was installed deliberately, leave it alone"
    #: — the automatic guard below (:func:`refit_skip_reason`) already
    #: covers the cases it can recognise from the document itself.
    hold: bool = False


#: The validation protocol a nightly refit runs under, always. It has no
#: option for anything else: the rows it fits on are gauges, scored at
#: those gauges, with the point's own gauge in the design. A document
#: fitted under :data:`~dmi_nowcast_core.postprocess.PROTOCOL_RANDOM_POINT`
#: is a different model of a different question and cannot be reproduced
#: here — see :func:`refit_skip_reason`.
NIGHTLY_PROTOCOL = pp.PROTOCOL_AT_GAUGE


@dataclass(frozen=True)
class ServedModel:
    """The header of the ``postprocess.json`` currently in service.

    Three fields, read without loading the model: what family it is, what
    design it was built on, and which protocol fitted it. Enough to answer
    the only question the nightly job asks of it — "could I have produced
    this?" — and nothing more.
    """

    kind: str
    design: str
    protocol: str
    #: The fit target (S11): the nightly refit only ever produces ``wet``.
    target: str = pp.TARGET_WET

    def describe(self) -> str:
        text = f"kind={self.kind} design={self.design} protocol={self.protocol}"
        if self.target != pp.TARGET_WET:
            text += f" target={self.target}"
        return text


def describe_served_model(path: Path) -> ServedModel | None:
    """The served document's kind, design version and protocol.

    ``None`` when there is nothing to protect: no file, unreadable, not
    JSON, not an object. A junk file is not a model, and the nightly fit
    should overwrite it exactly as it always has.

    The protocol is read from the top-level ``protocol`` key when the
    document carries one, and otherwise from ``training.protocol``, which
    is where the offline study has been recording it. Neither: the
    document predates the distinction and is therefore
    :data:`NIGHTLY_PROTOCOL` — every model fitted before the random-point
    work was an at-gauge fit.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    design = raw.get("design")
    version = design.get("version") if isinstance(design, dict) else None
    training = raw.get("training")
    protocol = raw.get("protocol")
    if not protocol and isinstance(training, dict):
        protocol = training.get("protocol")
    target = raw.get("target")
    if not target and isinstance(training, dict):
        target = training.get("target")
    return ServedModel(
        kind=str(raw.get("kind") or pp.KIND_LOGISTIC),
        design=str(version or pp.DESIGN_V1),
        protocol=str(protocol or NIGHTLY_PROTOCOL),
        target=str(target or pp.TARGET_WET),
    )


def refit_skip_reason(
    out: Path, options: PostprocessFitOptions, *, hold: bool = False,
) -> str | None:
    """Why tonight's refit must NOT run, or ``None`` to go ahead.

    The nightly fit publishes into the file the cycle reads, one
    generation of rollback deep. That is exactly right while the served
    model is one this job produced — and exactly wrong once an artefact
    fitted somewhere else is installed there: the refit would replace a
    tree model trained on the full archive under the random-point
    protocol with tonight's pooled logistic, the thresholds would be
    refitted on the replacement, and the only trace would be a
    ``postprocess.prev.json`` nobody was watching.

    So: if the configuration in this job could not have produced the
    document in service, leave it alone and say so. The comparison is on
    the three things the document states about itself — its kind, its
    design version, and the protocol it was fitted under — plus the flat
    rule that a tree model never comes from here at all (the image has no
    LightGBM; see :mod:`dmi_nowcast_core.postprocess_trees`).

    ``hold`` is the manual override, for a model this cannot tell apart
    from a nightly one but which the operator installed deliberately.
    """
    served = describe_served_model(out)
    if hold:
        if served is None:
            return (
                "fit_postprocess.hold is set; no refit (there is no readable "
                f"model at {out} to protect, and none will be written)"
            )
        return (
            f"fit_postprocess.hold is set; served model {served.describe()} "
            "left in place"
        )
    if served is None:
        return None
    configured = ServedModel(
        kind=str(options.model),
        design=str(options.design),
        protocol=NIGHTLY_PROTOCOL,
    )
    reproducible = (
        # S11: an onset-target model is never the nightly fit's output.
        served.target == pp.TARGET_WET
        and not pp.is_tree_kind(served.kind)
        and served.protocol == configured.protocol
        and served.design == configured.design
        and served.kind == configured.kind
    )
    if reproducible:
        return None
    return (
        f"served model {served.describe()} is not reproducible by the "
        f"nightly fit (model={configured.kind} design={configured.design} "
        f"protocol={configured.protocol}); left in place"
    )


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
    # The station each row was graded at, for the learned offsets. After
    # the recode, so a row the grid dropped carries the station it was
    # actually scored against rather than the one it claimed.
    scored_ids = np.asarray(list(scored) + [""], dtype=object)
    features["station_id"] = scored_ids[
        np.clip(np.asarray(rows["station"], dtype=np.int64), 0, len(scored))
    ].astype(str)
    settings = pp.FitSettings(
        kind=str(options.model),
        design=str(options.design),
        l2=float(options.l2),
        isotonic=str(options.isotonic),
        isotonic_bins=int(options.isotonic_bins),
        station_offsets=bool(options.station_offsets),
    ).validate()
    if log:
        log(
            f"fitting {len(leads)} lead(s) over {rows['rows']} row(s) "
            f"({settings.kind}, design {settings.design}, "
            f"{settings.isotonic} isotonic"
            + (", station offsets" if settings.station_offsets else "")
            + ")"
        )
    model = pp.fit_postprocess(
        features, truth, leads,
        design_leads=design_leads,
        settings=settings,
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
        # Which configuration tonight's model is, so the summary line and
        # the quality page say what is in service rather than what the
        # defaults used to be.
        "kind": model.kind,
        "design": model.spec.version,
        "isotonic": settings.isotonic,
        "station_offsets": settings.station_offsets,
        "columns": len(model.feature_names),
    }
    return {"model": model, "summary": summary}


#: Columns the filler derives from the decision instant rather than reading.
_DERIVED_COLUMNS = frozenset({"season", "hour_utc"})


def required_source_columns(model: Any) -> tuple[str, ...]:
    """The stored columns ``model``'s design reads, derived from the design.

    Not a hand list: :func:`dmi_nowcast_core.postprocess.build_design` is
    run once on a one-row probe that records every column it asks for, so
    a v1 logistic needs the v1 block, a v2 tree ensemble needs its
    ``g_*`` / ``ng_*`` / ensemble-shape columns too, and a design that grows
    next week is covered without anyone editing this.

    Two kinds of column are left out because no row has to carry them:
    ``season`` / ``hour_utc`` (derived from the decision instant) and,
    for a model that masks the point's own gauge, the columns the mask
    overwrites anyway.
    """
    names = list(dict.fromkeys(
        list(pp.feature_source_columns(model.design_leads))
        + list(model.spec.extra_columns)
    ))
    seen: list[str] = []

    class _Probe(dict):
        def get(self, key: Any, default: Any = None) -> Any:
            seen.append(str(key))
            return super().get(key, default)

        def __getitem__(self, key: Any) -> Any:
            seen.append(str(key))
            return super().__getitem__(key)

    probe = _Probe({name: np.zeros(1, dtype=np.float64) for name in names})
    probe["season"] = np.array(["summer"])
    try:
        pp.build_design(probe, model.design_leads, model.spec)
        read = [name for name in dict.fromkeys(seen)]
    except Exception:  # noqa: BLE001 — a design the probe cannot run:
        # require everything it could read, which only ever makes MORE rows
        # unfillable, never fills one on a guess.
        read = names
    skip = set(_DERIVED_COLUMNS)
    if model.masks_own_gauge:
        skip |= set(pp.OWN_GAUGE_COLUMNS) | {pp.GAUGE_KNOWN_COLUMN}
    return tuple(name for name in read if name not in skip)


def _writer_blocks(design_leads: Sequence[int]) -> tuple[tuple[str, ...], ...]:
    """The feature catalogue's append-only blocks, one per writer generation.

    A writer either computed a block or did not: a row written before the
    v2 block existed has nulls in ALL of it, while a row whose writer had
    it carries at least its never-null indicator (``g_known``,
    ``ng_frame_ok``). That is the per-row test :class:`ProbabilityFiller`
    applies — any finite value in each block the model reads.
    """
    leads = sorted({int(lead) for lead in design_leads})
    return (
        tuple(pp.raw_fraction_column(lead) for lead in leads),
        tuple(name for name, _doc in pp.SCALAR_FEATURE_COLUMNS),
        tuple(
            name for lead in leads
            for name in (pp.ens_mean_column(lead), pp.ens_p90_column(lead))
        ),
        tuple(name for name, _doc in pp.SCALAR_FEATURE_COLUMNS_V2),
        tuple(name for name, _doc in pp.SCALAR_FEATURE_COLUMNS_NG),
    )


def _float_column(table: Any, name: str) -> np.ndarray:
    import pyarrow as pa

    return np.asarray(
        table.column(name).combine_chunks()
        .cast(pa.float64()).to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )


class ProbabilityFiller:
    """Fills ``p_post_<lead>`` (or ``p_onset_<lead>``) on a decision table.

    The readers that score the served probability — the nightly threshold
    sweep, the quality page's scoreboard and gauge curve — read a mix of
    rows: live partitions that STORE the value the engine used, replay
    trees that carry the model's feature columns and no value, and older
    trees that carry neither (or only the v1 block of the features). This
    class makes one table of them, per file, while the columns are still
    numpy.

    The rule, per row:

    * a stored value wins — it is what the engine actually used — and is
      never re-predicted;
    * otherwise the served model speaks for the row ONLY when the row
      carries every column the model's design reads
      (:func:`required_source_columns`): the file has the column, and the
      row has a finite value in every feature block the model reads
      (:func:`_writer_blocks`). A v2 tree model handed a v1 row would read
      NaN for its forty gauge and neighbour columns and answer close to
      zero, which is a number about the archive's depth, not a forecast;
    * otherwise the row is UNFILLABLE: its value stays null and it is
      counted. With ``drop_unfilled`` (the default) a row that has no
      value at all is also removed; ``drop_unfilled=False`` keeps it, for
      the caller that has a second answer for such a row — the served-rule
      scorer, where the push engine's own fallback takes over exactly as
      it does live (``p_post`` null → the curve's ``p_rain``; ``p_onset``
      null → the single threshold).

    Only the rows that need a value are predicted. ``counts``: ``rows``
    read, ``stored`` rows carrying a value, ``computed`` (row, lead) cells
    the model filled, ``unfillable`` rows that needed a value the model
    could not give, and ``dropped`` (kept for older readers) rows left
    with no value at all.
    """

    def __init__(
        self,
        model: Any,
        leads: Sequence[int],
        design_leads: Sequence[int],
        column_for: Any,
        *,
        drop_unfilled: bool = True,
        mark_column: str | None = None,
    ) -> None:
        self.model = model
        #: When set, each table leaves with this float column: 1.0 on a
        #: row whose value the model computed here, 0.0 otherwise — the
        #: provenance a reader needs to call a curve in-sample.
        self.mark_column = mark_column
        self.leads = tuple(int(lead) for lead in leads)
        self.design_leads = tuple(int(lead) for lead in design_leads)
        self.column_for = column_for
        self.drop_unfilled = bool(drop_unfilled)
        #: What a row must carry for the model to speak for it; empty
        #: without a model (nothing can be filled then).
        self.required: tuple[str, ...] = (
            () if model is None else required_source_columns(model)
        )
        wanted = set(self.required) - pp.SHARED_SOURCE_COLUMNS
        self._blocks = tuple(
            block for block in (
                tuple(name for name in names if name in wanted)
                for names in _writer_blocks(self.design_leads)
            ) if block
        )
        self.counts: dict[str, int] = {
            "rows": 0, "stored": 0, "computed": 0, "unfillable": 0,
            "dropped": 0,
        }

    @property
    def source_columns(self) -> tuple[str, ...]:
        """The columns a caller must hand over for this filler to predict."""
        return self.required

    def fillable(self, table: Any) -> np.ndarray:
        """Rows the model can speak for — see the class docstring."""
        n = table.num_rows
        names = set(table.schema.names)
        if self.model is None or not set(self.required) <= names:
            return np.zeros(n, dtype=bool)
        out = np.ones(n, dtype=bool)
        for block in self._blocks:
            present = np.zeros(n, dtype=bool)
            for name in block:
                present |= np.isfinite(_float_column(table, name))
            out &= present
        return out

    def __call__(self, table: Any) -> Any:
        import pyarrow as pa

        names = set(table.schema.names)
        n = table.num_rows
        self.counts["rows"] += n
        stored: dict[int, np.ndarray] = {}
        for lead in self.leads:
            name = self.column_for(lead)
            stored[lead] = (
                _float_column(table, name) if name in names
                else np.full(n, np.nan, dtype=np.float64)
            )
        has_stored = np.zeros(n, dtype=bool)
        need = np.zeros(n, dtype=bool)
        for values in stored.values():
            finite = np.isfinite(values)
            has_stored |= finite
            need |= ~finite
        self.counts["stored"] += int(has_stored.sum())

        fillable = self.fillable(table) if need.any() else np.zeros(n, bool)
        predict = need & fillable
        computed_any = np.zeros(n, dtype=bool)
        if predict.any():
            index = np.flatnonzero(predict)
            predicted = self._predict(table.take(pa.array(index)))
            for lead, values in predicted.items():
                if lead not in stored:
                    continue
                fill = np.zeros(n, dtype=bool)
                fill[index] = np.isfinite(values)
                fill &= ~np.isfinite(stored[lead])
                column = np.array(stored[lead], dtype=np.float64)
                full = np.full(n, np.nan, dtype=np.float64)
                full[index] = values
                column[fill] = full[fill]
                stored[lead] = column
                computed_any |= fill
                self.counts["computed"] += int(fill.sum())
        self.counts["unfillable"] += int((need & ~fillable).sum())

        keep = has_stored | fillable
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
        if self.mark_column is not None:
            if self.mark_column in names:
                table = table.drop_columns([self.mark_column])
            table = table.append_column(
                self.mark_column, pa.array(computed_any.astype(np.float64)),
            )
        if self.drop_unfilled and not keep.all():
            table = table.filter(pa.array(keep))
        return table

    def _predict(self, table: Any) -> dict[int, np.ndarray]:
        import pyarrow as pa
        import pyarrow.compute as pc

        names = set(table.schema.names)
        n = table.num_rows
        features: dict[str, Any] = {}
        # Every name the design could read gets a key (NaN when absent):
        # the own-gauge mask only overwrites keys that exist, and a model
        # that masks must see ``g_known`` = 0, not a missing column.
        for name in dict.fromkeys(
            list(pp.feature_source_columns(self.design_leads))
            + list(self.model.spec.extra_columns)
        ):
            if name in _DERIVED_COLUMNS:
                continue
            features[name] = (
                _float_column(table, name) if name in names
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
    "NIGHTLY_PROTOCOL",
    "PostprocessFitOptions",
    "ProbabilityFiller",
    "ServedModel",
    "build_features",
    "describe_served_model",
    "featured_mask",
    "filter_rows",
    "options_from_json",
    "options_to_json",
    "refit_skip_reason",
    "required_source_columns",
    "run_postprocess_fit",
]
