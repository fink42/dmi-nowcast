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

The arms this can compare (post-processing v2)
----------------------------------------------
``--model logistic|logistic-shared|trees|trees-shared``, ``--design
v1|v2``, ``--station-offsets``, ``--isotonic pooled|per-season``. Every
default is the model in service, so a run with no new flag reproduces the
shipped fit.

A ``-shared`` kind fits ONE model over the rows of every served lead
stacked, with the lead in the design: ``lead_min`` and ``log_lead_min``,
plus ``raw_frac_own`` (and ``ens_mean_own`` / ``ens_p90_own`` under
``--design v2``), which carry the row's OWN lead's values. Every other
coefficient is shared, and only the isotonic recalibration stays per lead
— the base rate at 60 minutes is not the base rate at 20. The four
kinds are scored side by side in one run with
``--compare logistic,logistic-shared,trees,trees-shared``; only
``--model`` is written to ``postprocess.json``.

P(rain within L) is non-decreasing in L, and the served answer is made to
respect that: ``trees-shared`` carries a LightGBM monotone constraint on
the lead columns, and every kind goes through the same running max across
leads (``national.enforce_lead_monotonic``) that guards the served
``p_rain`` — at scoring time AND on the out-of-fold predictions this
writes back, so the number that is measured is the number that is served.

``--baseline`` decides what the candidate is measured against.
``curve`` is the served ``p_rain_<lead>``, which is the comparison that
argued for shipping the post-processor at all. ``refit-v1`` is the v1
design plus the shipped logistic, **refitted inside every fold** on the
same rows and the same folds — the baseline of record for v2, because
"beats the curve" is settled and the open question is whether a new
design beats the old one. ``model:<path>`` refits whatever configuration
a ``postprocess.json`` records.

``--target wet|onset`` decides what the probability is OF. ``wet`` (the
default, the shipped model) is Layer B's "gauge wet within (t, t+L]".
``onset`` is the event the push is graded on: a gauge onset from
``threshold_sweep.gauge_truth`` inside the scorer's window
``(t, t+L+--onset-tolerance-min]`` (10 min, ``warning_score``'s), 0 only
where every slot of that window is known, excluded otherwise. Every
out-of-fold number — the baseline arm's too — is then scored on that
label, ``--write-back`` writes ``p_onset_<lead>`` instead of
``p_post_<lead>``, and the artefact says ``"target": "onset"``.
``--onset-rows dry`` restricts training AND scoring to the dry-60 rows.

Everything is reported twice: on all rows, and on the **dry** subset —
rows whose gauge was dry for the hour before the decision instant. A
model that wins only on rows where it was already raining has not won.

``--learning-curve 10,20,30,60,90`` adds one LOMO pass per N, each fold
trained on N of its training DAYS drawn at random and **stratified by
month** (``--learning-curve-seed``, default 0), so the N-day point
differs from the all-days point in volume and not in season.
``--learning-curve-order date`` reverts to the first N calendar days,
which is the seasonal experiment rather than the data-volume one; the
report says which order produced the table. Both are scored by the same
function the out-of-fold table above them uses, on the same rows, so the
curve's last point IS the table. ``--ablate`` adds one pass per feature
family, with the family dropped.

LightGBM, and why it is not in this project's venv
--------------------------------------------------
``--model trees`` fits with LightGBM and exports a JSON description of
the ensemble that ``dmi_nowcast_core.postprocess_trees`` evaluates in
pure numpy. The sidecar image has numpy, scipy and pyarrow and will not
be growing LightGBM or scikit-learn: the serving container is what a
radar cycle blocks on. So the fit runs in a separate venv beside the
project's::

    uv venv --python .venv/bin/python .venv-fit
    uv pip install --python .venv-fit/bin/python lightgbm scikit-learn \\
        numpy scipy pytest pyarrow
    .venv-fit/bin/python scripts/fit_postprocess.py --model trees ...

``.venv`` and the sidecar's venv are left alone, and a tree fit attempted
from either says so with that command in the error.
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
from dmi_nowcast_core.postprocess_trees import (  # noqa: E402
    DEFAULT_TREE_PARAMS,
)
from dmi_nowcast_core.warning_score import (  # noqa: E402
    DEFAULT_DRY_MIN,
    DEFAULT_MIN_KNOWN_SLOTS,
    DEFAULT_ONSET_MIN_MM,
    DEFAULT_TOLERANCE_MIN,
)
from dmi_nowcast_sidecar.threshold_sweep import (  # noqa: E402
    SweepError,
    decision_parquets,
    gauge_truth,
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

#: Column the write-back adds, per lead. The core module's name, so the
#: study, the nightly refit, the live writer and the sweep all mean one
#: column.
POST_COLUMN_TEMPLATE = pp.POST_COLUMN_TEMPLATE
post_column = pp.post_column

#: ``--model`` values — the core module's model kinds, unchanged, so the
#: flag, the artefact's ``kind`` and the config key all spell the four
#: arms the same way.
MODEL_LOGISTIC = pp.KIND_LOGISTIC
MODEL_LOGISTIC_SHARED = pp.KIND_LOGISTIC_SHARED
MODEL_TREES = pp.KIND_TREES
MODEL_TREES_SHARED = pp.KIND_TREES_SHARED
MODEL_CHOICES: tuple[str, ...] = pp.MODEL_KINDS

#: ``--baseline`` values that are not ``model:<path>``.
BASELINE_CURVE = "curve"
BASELINE_REFIT_V1 = "refit-v1"

#: ``--target`` values: the core module's, so the flag and the artefact's
#: ``target`` spell the same two words.
TARGET_WET = pp.TARGET_WET
TARGET_ONSET = pp.TARGET_ONSET
TARGET_CHOICES: tuple[str, ...] = pp.TARGETS

#: ``--onset-rows`` values. ``all`` keeps every gradable row, including the
#: ones whose gauge was wet in the hour before — the model has to learn
#: that rain already at the point means no onset, and under random-point
#: it must learn it from radar and neighbours because its own gauge is
#: masked. ``dry`` restricts training AND scoring to the dry-60 subset.
ONSET_ROWS_ALL = "all"
ONSET_ROWS_DRY = "dry"
ONSET_ROWS_CHOICES: tuple[str, ...] = (ONSET_ROWS_ALL, ONSET_ROWS_DRY)


def parse_tree_params(text: str) -> dict:
    """``"n_estimators=500,max_depth=6"`` → a LightGBM override dict.

    Numbers are parsed as numbers so LightGBM sees an int where it wants
    one; anything else is left as a string and LightGBM complains about it
    in its own words, which are better than any this script could add.
    """
    out: dict = {}
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--trees expects key=value, got {item!r}")
        key, _, value = item.partition("=")
        value = value.strip()
        try:
            out[key.strip()] = int(value)
        except ValueError:
            try:
                out[key.strip()] = float(value)
            except ValueError:
                out[key.strip()] = value
    return out


def settings_from_args(args) -> Any:
    """The :class:`postprocess.FitSettings` the command line describes."""
    return settings_for(args, str(args.model), parse_families(args.drop_families))


def settings_for(args, kind: str, drop_families: Sequence[str] = ()):
    """The same settings under a different model kind — one arm of a sweep.

    ``drop_families`` is the arm's own ablation: the families whose design
    columns are removed before the fit. It is a per-ARM choice rather than a
    run-wide one so that "v2 with the gauges" and "v2 with no gauge
    information at all" can be scored in one pass, on the same folds and
    against the same baseline.
    """
    return pp.FitSettings(
        kind=kind,
        design=args.design,
        l2=float(args.l2),
        isotonic=args.isotonic,
        isotonic_bins=int(args.isotonic_bins),
        station_offsets=bool(args.station_offsets),
        station_l2_multiple=float(args.station_l2_multiple),
        tree_params=parse_tree_params(args.trees) or None,
        seed=int(args.seed),
        drop_families=tuple(drop_families),
    ).validate()


def baseline_settings_from_args(args) -> Any:
    """The baseline configuration, or None for the served curve.

    ``refit-v1`` is the shipped model refitted IN FOLD: the v1 design, the
    logistic, no station offsets, one pooled curve — whatever ``--l2`` and
    ``--isotonic-bins`` say, because those are properties of the fit and
    not of the design under test. ``model:<path>`` reads the settings a
    ``postprocess.json`` recorded and refits THAT, which is how "did this
    change beat what is in service?" gets asked of the thing in service.
    """
    choice = str(args.baseline or BASELINE_CURVE)
    if choice == BASELINE_CURVE:
        return None
    if choice == BASELINE_REFIT_V1:
        return pp.FitSettings(
            l2=float(args.l2), isotonic_bins=int(args.isotonic_bins),
            seed=int(args.seed),
        ).validate()
    if not choice.startswith("model:"):
        raise ValueError(
            f"--baseline must be '{BASELINE_CURVE}', '{BASELINE_REFIT_V1}' "
            f"or 'model:<path>', got {choice!r}"
        )
    path = Path(choice.split(":", 1)[1])
    document = json.loads(path.read_text())
    stored = ((document.get("training") or {}).get("settings")) or {}
    return pp.FitSettings(
        kind=str(document.get("kind", stored.get("kind", pp.KIND_LOGISTIC))),
        # A pre-v2 document has no ``kind`` and is a per-lead logistic; a
        # v2 one names its own arm, shared or not.
        design=str(
            (document.get("design") or {}).get(
                "version", stored.get("design", pp.DESIGN_V1),
            )
        ),
        l2=float(document.get("l2", stored.get("l2", args.l2))),
        isotonic=str(stored.get("isotonic", pp.ISOTONIC_POOLED)),
        isotonic_bins=int(stored.get("isotonic_bins", args.isotonic_bins)),
        station_offsets=bool(stored.get("station_offsets", False)),
        station_l2_multiple=float(
            stored.get("station_l2_multiple", pp.STATION_L2_MULTIPLE),
        ),
        tree_params=dict(stored.get("tree_params") or {}) or None,
        seed=int(args.seed),
    ).validate()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


#: The design's source columns, and the subset that exists only when a run
#: wrote features. Both live in the core module now: the nightly refit in
#: the sidecar asks the same two questions of the same rows, and a second
#: opinion would silently change which rows are trained on.
feature_source_columns = pp.feature_source_columns
feature_only_columns = pp.feature_only_columns
SHARED_SOURCE_COLUMNS = pp.SHARED_SOURCE_COLUMNS


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


def _onset_seconds(onsets: Sequence[Any]) -> np.ndarray:
    """Sorted epoch seconds of a station's onset instants."""
    return np.array(
        sorted(int(o.timestamp()) for o in onsets), dtype=np.int64,
    )


def window_all_known(
    grid: Any, t: np.ndarray, station: np.ndarray, window_min: int,
) -> np.ndarray:
    """True where every gauge slot ending in ``(t, t + window]`` is known.

    The same slot arithmetic as ``GaugeGrid.outcome`` — the window is the
    slots whose END falls in the half-open interval, and a window running
    off either end of the grid is not known — read off the grid's known-
    slot prefix sum. ``outcome`` cannot answer this itself: it certifies
    "no WET slot", and one wet slot makes its window usable whatever the
    others said, while "no ONSET" needs every slot, wet or dry.
    """
    step = int(grid.step_sec)
    t = np.asarray(t, dtype=np.int64)
    station = np.asarray(station, dtype=np.int64)
    lo = (t - grid.first_sec) // step + 1
    hi = (t + int(window_min) * 60 - grid.first_sec) // step
    inside = (lo >= 0) & (hi < grid.n_slots) & (hi >= lo)
    safe_lo = np.where(inside, lo, 0)
    safe_hi = np.where(inside, hi, 0)
    # ``_known_flat`` is the grid's own cumulative count of known slots,
    # one leading zero per station row; ``_stride`` is that row length.
    known = grid._known_flat
    base = station * grid._stride
    n_known = known[base + safe_hi + 1] - known[base + safe_lo]
    return inside & (n_known == safe_hi - safe_lo + 1)


def build_onset_truth(
    rows: Mapping[str, Any],
    grid: Any,
    onsets: Mapping[str, Sequence[Any]],
    stations: Sequence[str],
    leads: Sequence[int],
    *,
    tolerance_min: int = DEFAULT_TOLERANCE_MIN,
    restrict: np.ndarray | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{lead: (onset in window, usable)}`` — the push's own event.

    A row at decision instant ``t`` (``generated_at``) and lead ``L`` is:

    * **1** when a gauge onset from ``threshold_sweep.gauge_truth`` — the
      list the push is scored against, same dry spell, same amount rule,
      same dead-gauge exclusions — falls in ``(t, t + L + tolerance]``,
      the scorer's window (``warning_score.score_warnings``). An onset is
      its own evidence, whatever the rest of the window reported.
    * **0** when no onset falls there AND every gauge slot ending in that
      window is known.
    * **excluded** otherwise: an unknown slot cannot certify "no onset".

    Onsets are stamped at their slot's END, so "onset in the window" and
    "slot ending in the window" are the same interval.

    ``stations`` are the grid's scored stations, indexed by
    ``rows["station"]`` after ``_recode_stations``. Rows the recode
    dropped are excluded, as in :func:`build_truth`. ``restrict`` narrows
    the usable rows further (``--onset-rows dry``).

    Rows whose gauge was wet in the hour before ``t`` keep their label —
    0 by construction, except where the wet spell ended early enough in
    that hour for a new onset to fit inside the window.
    """
    t = np.asarray(rows["t"], dtype=np.int64)
    code = np.asarray(rows["station"], dtype=np.int64)
    dropped = np.asarray(
        rows.get("dropped", np.zeros(t.size, dtype=bool)), dtype=bool,
    )
    keep = ~dropped
    if restrict is not None:
        keep = keep & np.asarray(restrict, dtype=bool)
    by_code = [_onset_seconds(onsets.get(s, ())) for s in stations]
    order = np.argsort(code, kind="stable")
    bounds = np.searchsorted(code[order], np.arange(len(stations) + 1))
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for lead in leads:
        window = int(lead) + int(tolerance_min)
        hit = np.zeros(t.size, dtype=bool)
        for index, times in enumerate(by_code):
            if times.size == 0:
                continue
            members = order[bounds[index]:bounds[index + 1]]
            if members.size == 0:
                continue
            at = t[members]
            nxt = np.searchsorted(times, at, side="right")
            ok = nxt < times.size
            found = np.zeros(at.size, dtype=bool)
            found[ok] = times[nxt[ok]] <= at[ok] + window * 60
            hit[members] = found
        known = window_all_known(grid, t, code, window)
        usable = keep & (hit | known)
        out[int(lead)] = (hit.astype(np.float64), usable)
    return out


def target_counts(
    truth: Mapping[int, tuple[np.ndarray, np.ndarray]], n: int,
) -> dict[str, dict[str, Any]]:
    """Per lead: gradable rows, positives, excluded rows and the base rate."""
    out: dict[str, dict[str, Any]] = {}
    for lead, (y, usable) in sorted(truth.items()):
        graded = int(np.count_nonzero(usable))
        positive = int(np.count_nonzero(usable & (y > 0)))
        out[str(lead)] = {
            "rows": graded,
            "positives": positive,
            "excluded": int(n - graded),
            "base_rate": (positive / graded) if graded else None,
        }
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


#: Keys of the LOMO payload that hold one value per row. They are the
#: write-back's input and have no business in a JSON report.
_ARRAY_KEYS = frozenset({"out_of_fold", "baseline_out_of_fold"})


def _strip_arrays(evaluation: Mapping[str, Any]) -> dict:
    """The LOMO payload without the per-row prediction arrays."""
    return {k: v for k, v in evaluation.items() if k not in _ARRAY_KEYS}


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = ["# Gauge-trained post-processing (Phase H, H-P)", ""]
    settings = report["settings"]
    baseline_name = str(settings.get("baseline", BASELINE_CURVE))
    lines += [
        f"Generated {report['generated_at_utc']}.",
        "",
    ]
    lines += _target_section(report)
    lines += [
        "Leave-one-(year, month)-out over the replay's decision rows. The "
        + (
            "baseline is the served `p_rain_<lead>` — the per-lead "
            "isotonic curve on the ensemble fraction — and the candidate "
            "is this model's out-of-fold prediction on **the same rows**."
            if baseline_name == BASELINE_CURVE else
            f"baseline is `{baseline_name}`, refitted **inside every "
            "fold** on the same rows and the same folds as the candidate, "
            "so the difference is attributable to the configuration and "
            "not to a different sample or a different split."
        )
        + " Every confidence interval is a paired day-block bootstrap "
        f"({settings['resamples']} resamples, "
        f"{settings['ci']:.0%}); an interval excluding zero is the plan's "
        "evidence that the difference is real.",
        "",
        "Every comparison is reported twice: on **all** rows, and on the "
        f"**dry** subset — the rows whose gauge was dry for the "
        f"{pp.DRY_BEFORE_MIN} minutes before the decision instant. Those "
        "are the onset-relevant ones: a row whose gauge was already wet "
        "is one nobody needed a warning for, and pooling them in flatters "
        "every arm equally while hiding which one is better at the thing "
        "a subscriber notices.",
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
        f"| dry (onset-relevant) rows | {report.get('dry_rows', 0)} "
        f"({report.get('dry_source', '–')}) |",
        f"| stations scored | {report['stations']} |",
        f"| days | {report['days']} |",
        f"| window | {report['window']['from']} → {report['window']['to']} |",
        f"| months | {report['n_months']} |",
        f"| leads | {', '.join(str(x) for x in settings['leads'])} |",
        f"| design leads | {', '.join(str(x) for x in settings['design_leads'])} |",
        f"| model | {settings.get('model', MODEL_LOGISTIC)} |",
        f"| design | {settings.get('design', pp.DESIGN_V1)} |",
        f"| isotonic | {settings.get('isotonic', pp.ISOTONIC_POOLED)} |",
        f"| station offsets | "
        f"{'yes' if settings.get('station_offsets') else 'no'} |",
        f"| baseline | {baseline_name} |",
        f"| target | {_target_label(settings)} |",
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

    lines += _summary_section(report)
    lines += _skill_section(report)
    lines += _distance_section(report)
    lines += _random_point_section(report)
    lines += _learning_curve_section(report)
    lines += _ablation_section(report)
    lines += _reliability_section(report)
    lines += _coefficient_section(report)
    lines += _fold_section(report)
    lines += _feature_section(report)
    return "\n".join(lines) + "\n"


def _target_label(settings: Mapping[str, Any]) -> str:
    """One cell: the outcome, and under onset its window and its rows."""
    if settings.get("target", TARGET_WET) != TARGET_ONSET:
        return "wet within `(t, t + L]`"
    return (
        f"onset in `(t, t + L + {settings.get('onset_tolerance_min')}]`, "
        f"rows `{settings.get('onset_rows', ONSET_ROWS_ALL)}`"
    )


def _target_section(report: Mapping[str, Any]) -> list[str]:
    """What every number below is a probability OF."""
    settings = report["settings"]
    if settings.get("target", TARGET_WET) != TARGET_ONSET:
        return [
            "**Target `wet`**: the gauge was wet at some point in "
            "`(t, t + L]` — Layer B's outcome.",
            "",
        ]
    tolerance = settings.get("onset_tolerance_min")
    rows_rule = settings.get("onset_rows", ONSET_ROWS_ALL)
    lines = [
        f"**Target `onset`**: a gauge onset (rain after "
        f"{settings.get('dry_min')} dry minutes delivering ≥ "
        f"{settings.get('onset_min_mm')} mm, `threshold_sweep.gauge_truth`) "
        f"inside the scorer's window `(t, t + L + {tolerance}]` — the event "
        "the push is graded on. 1 when an onset falls there; 0 when none "
        "does and every gauge slot ending in the window is known; excluded "
        "otherwise. **Every** out-of-fold metric below — the baseline "
        "arm's included — is scored on this label.",
        "",
        (
            "Rows `all`: rows whose gauge was wet in the hour before `t` are "
            "kept, with their label (0 unless the wet spell ended early "
            "enough for a new onset to fit the window)."
            if rows_rule == ONSET_ROWS_ALL else
            f"Rows `dry`: training AND scoring are restricted to the dry-"
            f"{pp.DRY_BEFORE_MIN} subset, so the `all` stratum below IS the "
            "dry subset."
        ),
        "",
    ]
    counts = report.get("target_counts") or {}
    if counts:
        lines += [
            "| lead | window | gradable rows | onsets | base rate | excluded |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for lead in settings["leads"]:
            entry = counts.get(str(lead)) or {}
            lines.append(
                f"| {lead} | `(t, t + {int(lead) + int(tolerance or 0)}]` | "
                f"{entry.get('rows', '–')} | {entry.get('positives', '–')} | "
                f"{_fmt(entry.get('base_rate'), 4)} | "
                f"{entry.get('excluded', '–')} |"
            )
        lines.append("")
    return lines


def _summary_section(report: Mapping[str, Any]) -> list[str]:
    """The one table a reader should be able to stop at.

    Candidate against baseline, ΔBSS with its interval, on all rows and
    on the dry subset, per lead. Everything below this is the working.
    """
    settings = report["settings"]
    protocol = str(settings.get("protocol", PROTOCOL_AT_GAUGE))
    lines = [
        f"## Candidate vs baseline — protocol `{protocol}`",
        "",
        (
            "Rows are gauges and are allowed to be: the model reads the "
            "gauge under the point, and the folds hold out a month."
            if protocol == PROTOCOL_AT_GAUGE else
            "**Rows are treated as the addresses they stand in for.** The "
            "point's own gauge block is masked to 'unknown' in training AND "
            "in scoring, per-station intercepts are refused, and every fold "
            "holds out a month AND a group of stations together — so no "
            "prediction comes from a model that saw that month or that "
            "place. The `ng_*` block is not masked: it never contained the "
            "point's own gauge."
        ),
        "",
        f"`{settings.get('model', MODEL_LOGISTIC)}` on design "
        f"`{settings.get('design', pp.DESIGN_V1)}`"
        + (" with station offsets" if settings.get("station_offsets") else "")
        + f", `{settings.get('isotonic', pp.ISOTONIC_POOLED)}` recalibration, "
        f"against `{settings.get('baseline', BASELINE_CURVE)}`. ΔBSS is "
        "candidate minus baseline on the same out-of-fold rows; an "
        "interval that excludes zero is the evidence.",
        "",
        "| model | lead | subset | n | BSS base | BSS cand. | ΔBSS [CI] | real? |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    arms = [
        (str(settings.get("arm") or settings.get("model", MODEL_LOGISTIC)),
         report["evaluation"]["leads"]),
    ] + [(str(arm["model"]), arm["leads"]) for arm in report.get("arms") or ()]
    for model, per_lead in arms:
        for lead in settings["leads"]:
            entry = per_lead.get(str(lead)) or {}
            for subset in (pp.POOLED, pp.DRY):
                block = entry.get(subset)
                if not block:
                    lines.append(
                        f"| `{model}` | {lead} | {subset} | – | – | – | – | – |"
                    )
                    continue
                diff = block.get("difference") or {}
                verdict = diff.get("bss_excludes_zero")
                lines.append(
                    f"| `{model}` | {lead} | {subset} | {block['n']} | "
                    f"{_fmt(block['baseline']['bss'], 4)} | "
                    f"{_fmt(block['postprocess']['bss'], 4)} | "
                    f"{_ci(diff.get('bss'))} | "
                    + ("yes" if verdict else ("no" if verdict is False else "–"))
                    + " |"
                )
    lines.append("")
    if len(arms) > 1:
        lines += [
            "Only the first model is written to `postprocess.json`; the "
            "rest are `--compare` arms, scored on the same folds and "
            "against the same baseline, so a column means one thing all "
            "the way down the table.",
            "",
        ]
    return lines


def _learning_curve_section(report: Mapping[str, Any]) -> list[str]:
    """How much archive the configuration needs before it is worth having."""
    rows = report.get("learning_curve") or []
    if not rows:
        return []
    leads = [str(x) for x in report["settings"]["leads"]]
    subsets = sorted({
        name for row in rows for block in row["leads"].values() for name in block
    })
    order = str(rows[0].get("order", pp.CURVE_ORDER_RANDOM))
    seed = rows[0].get("seed", pp.DEFAULT_CURVE_SEED)
    how = (
        "Each fold trains on **N of its training days, drawn at random "
        "and stratified by month**: every training month "
        "contributes a share of the draw proportional to the days it "
        "has, so an N-day point differs from the all-days point in "
        "volume and not in season. The draw is nested \u2014 a larger "
        "budget's days contain a smaller one's."
        if order == pp.CURVE_ORDER_RANDOM else
        "Each fold trains on **the first N calendar days** of its "
        "training rows (`--learning-curve-order date`). That is the "
        "seasonal experiment: on a winter-first archive the small N "
        "points are winter-only fits scored on every month, so read a "
        "non-monotone curve as a statement about season, not about how "
        "much archive the model wants."
    )
    lines = [
        "## Learning curve",
        "",
        f"Draw order `{order}`, seed {seed}. " + how,
        "",
        "BSS is out of fold, pooled over the folds, and computed by the "
        "same function and on the same rows as the out-of-fold table "
        "above \u2014 so the last point of this curve, where the budget "
        "covers every training day, is that table's number.",
        "",
        "`train days` is the largest number of distinct days any fold "
        "actually got (a fold with fewer available gives what it has); "
        "`rows` is the out-of-fold rows scored in that subset, at "
        "whichever lead grades the most of them.",
        "",
        "| days | train days | train rows | subset | rows | "
        + " | ".join(f"BSS {lead} min" for lead in leads) + " |",
        "|---:|---:|---:|---|---:|" + "---:|" * len(leads),
    ]
    for row in rows:
        for subset in subsets:
            cells = []
            scored = 0
            for lead in leads:
                block = (row["leads"].get(lead) or {}).get(subset) or {}
                cells.append(_fmt(block.get("bss"), 4))
                scored = max(scored, int(block.get("n", 0)))
            used = row.get("train_days", "\u2013")
            lines.append(
                f"| {row['days']} | {used} | "
                f"{row['train_rows']} | {subset} | {scored} | "
                + " | ".join(cells) + " |"
            )
    lines.append("")
    months = rows[-1].get("months") or {}
    if months:
        lines += [
            "Month mix of the draw at N = "
            f"{rows[-1]['days']}, summed over the folds (a day drawn for "
            "three folds is counted three times): "
            + ", ".join(f"{label} {count}" for label, count in months.items())
            + ".",
            "",
        ]
    return lines


def _ablation_section(report: Mapping[str, Any]) -> list[str]:
    """What each feature family is worth, by taking it away."""
    block = report.get("ablation")
    if not block:
        return []
    leads = [str(x) for x in report["settings"]["leads"]]
    subsets = sorted({
        name for lead in block["full"].values() for name in lead
    })
    lines = [
        "## Ablation",
        "",
        "One re-run of the whole leave-one-month-out per family, with that "
        "family's columns — and every interaction either parent appears in "
        "— removed from the design. ΔBSS is *dropped minus full*, so a "
        "**negative** number means the family was carrying something and "
        "a positive one means it was costing.",
        "",
        "| family | columns dropped | subset | "
        + " | ".join(f"ΔBSS {lead} min" for lead in leads) + " |",
        "|---|---:|---|" + "---:|" * len(leads),
    ]
    for entry in block["dropped"]:
        for subset in subsets:
            cells = []
            for lead in leads:
                cell = (entry["leads"].get(lead) or {}).get(subset) or {}
                cells.append(_fmt(cell.get("delta"), 4))
            lines.append(
                f"| `{entry['family']}` | {entry['columns']} | {subset} | "
                + " | ".join(cells) + " |"
            )
    lines.append("")
    return lines


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
        for stratum in (pp.POOLED, pp.DRY) + pp.SEASONS:
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
        "project-wide binning. `observed` is the share of those rows "
        + (
            "with a gauge onset inside the scorer window `(t, t + L + "
            f"{report['settings'].get('onset_tolerance_min')}]`."
            if report["settings"].get("target") == TARGET_ONSET else
            "whose gauge was wet inside the lead."
        ),
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
    if model.get("kind") == pp.KIND_TREES:
        return _tree_section(report)
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


def _tree_section(report: Mapping[str, Any]) -> list[str]:
    """What a tree model has instead of coefficients.

    Not a coefficient table and not pretending to be one: a boosted
    ensemble's answer to "which predictor carries the weight" is the
    ablation above, which measures what happens when a family is taken
    away rather than how large a number next to it is.
    """
    model = report["model"]
    leads = [int(x) for x in model["leads"]]
    shared = model.get("shared_trees")
    lines = [
        "## The ensemble",
        "",
        "A boosted ensemble has no standardised coefficients to read. "
        "What it is worth per feature family is in the ablation section; "
        "what is below is only the size of the artefact the sidecar has "
        "to evaluate every cycle.",
        "",
        "| | " + " | ".join(f"{lead} min" for lead in leads) + " |",
        "|---|" + "---:|" * len(leads),
    ]

    def block(lead: int) -> dict:
        entry = model["models"][str(lead)]
        return entry.get("trees") or shared or {}

    lines += [
        "| trees | "
        + " | ".join(str((block(lead) or {}).get("n_trees", "–")) for lead in leads)
        + " |",
        "| base rate | "
        + " | ".join(
            _fmt(model["models"][str(lead)]["base_rate"], 4) for lead in leads
        )
        + " |",
        "| training rows | "
        + " | ".join(str(model["models"][str(lead)]["n"]) for lead in leads)
        + " |",
        "",
        (
            "One ensemble is shared across every lead, with the lead as a "
            "feature." if shared else "One ensemble per lead."
        ),
        "",
        "LightGBM parameters: "
        + ", ".join(
            f"`{k}={v}`"
            for k, v in sorted(((block(leads[0]) or {}).get("params") or {}).items())
        )
        + ".",
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
    column_template: str = POST_COLUMN_TEMPLATE,
) -> dict:
    """Copy the run, adding ``p_post_<lead>`` from the out-of-fold rows.

    ``column_template`` names the column: ``p_post_{lead}`` for a ``wet``
    fit, ``p_onset_{lead}`` for an ``onset`` one, so the two can never be
    read as each other.

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
                name = column_template.format(lead=int(lead))
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
# The validation protocol: at a gauge, or at a random point
# ---------------------------------------------------------------------------

PROTOCOL_AT_GAUGE = pp.PROTOCOL_AT_GAUGE
PROTOCOL_RANDOM_POINT = pp.PROTOCOL_RANDOM_POINT
PROTOCOL_CHOICES: tuple[str, ...] = pp.PROTOCOLS

#: Separator between an arm's model kind and its per-arm options in
#: ``--compare``, e.g. ``logistic/drop=gauge+neighbour``. A slash rather
#: than a comma because the flag itself is comma-separated, and a plus
#: inside ``drop=`` for the same reason.
ARM_SEPARATOR = "/"
ARM_DROP = "drop="
ARM_FAMILY_SEPARATOR = "+"


def parse_families(text: str) -> tuple[str, ...]:
    """``"gauge,neighbour"`` or ``"gauge+neighbour"`` → validated family names.

    Unknown names are an error rather than a no-op: a typo'd ``--drop-families
    neigbour`` that silently dropped nothing would produce a comparison
    between two identical arms and a ΔBSS of zero, which reads exactly like
    a feature family that does not earn its keep.
    """
    names = tuple(
        item.strip()
        for chunk in str(text or "").split(",")
        for item in chunk.split(ARM_FAMILY_SEPARATOR)
        if item.strip()
    )
    known = set(pp.FAMILY_NAMES) | {"other"}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(
            f"unknown feature famil(ies) {', '.join(unknown)}; expected from "
            + ", ".join(pp.FAMILY_NAMES)
        )
    return names


def arm_label(kind: str, drop_families: Sequence[str]) -> str:
    """How an arm is named in the summary table: ``logistic -gauge,neighbour``."""
    if not drop_families:
        return str(kind)
    return f"{kind} −{','.join(drop_families)}"


def parse_compare(text: str, args) -> list[tuple[str, Any]]:
    """``--compare`` → ``[(label, FitSettings), ...]``.

    An entry is a model kind, optionally followed by ``/drop=<families>``:
    ``logistic``, ``trees-shared``, ``logistic/drop=gauge+neighbour``. The
    drop suffix is what lets one run score "v2 with the gauges" against "v2
    without any gauge information" on the same folds and against the same
    baseline — the comparison the random-point protocol exists to make —
    without a second invocation whose folds a reader would have to take on
    trust.
    """
    out: list[tuple[str, Any]] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        kind, _, options = item.partition(ARM_SEPARATOR)
        kind = kind.strip() or str(args.model)
        if kind not in MODEL_CHOICES:
            raise ValueError(
                f"--compare: unknown model kind {kind!r}; expected from "
                + ", ".join(MODEL_CHOICES)
            )
        options = options.strip()
        if options and not options.startswith(ARM_DROP):
            raise ValueError(
                f"--compare: unknown arm option {options!r}; the only one is "
                f"'{ARM_DROP}<family>[{ARM_FAMILY_SEPARATOR}<family>...]'"
            )
        families = parse_families(options[len(ARM_DROP):]) if options else ()
        out.append((
            arm_label(kind, families), settings_for(args, kind, families),
        ))
    return out


def station_coordinates(path: Path | None) -> dict[str, tuple[float, float]]:
    """``{station_id: (lat, lon)}`` — the geometry the protocol needs.

    Read through the neighbour-feature builder's loader, so the coordinates
    the ``ng_*`` columns were computed from and the coordinates the spatial
    split and the random-point weights are computed from cannot come apart.
    """
    if path is None:
        raise ValueError(
            f"--protocol {PROTOCOL_RANDOM_POINT} needs --points: the spatial "
            "hold-out and the random-point weights are both geometry"
        )
    from add_neighbour_gauge_features import load_station_coords

    return load_station_coords(Path(path))


def protocol_folds(args, features: Mapping[str, Any], month: np.ndarray):
    """``(FoldPlan, block for the report)`` for the chosen protocol.

    ``at-gauge`` is one fold per (year, month) — leave-one-month-out, the
    historical evaluation, unchanged to the row.

    ``random-point`` crosses that with a station group, so a prediction
    comes only from a model that saw neither the month nor the place. The
    groups are :func:`~dmi_nowcast_core.postprocess.station_groups`'
    longitude-interleaved comb, which is deterministic and needs no seed.
    """
    if str(args.protocol) == PROTOCOL_AT_GAUGE:
        return pp.FoldPlan.by_month(month), {"kind": "month"}
    coords = station_coordinates(args.points)
    present = {
        str(sid) for sid in np.unique(np.asarray(features["station_id"]).astype(str))
        if str(sid) in coords
    }
    groups = pp.station_groups(
        {sid: coords[sid] for sid in sorted(present)},
        groups=int(args.station_groups),
    )
    codes = pp.group_codes(features["station_id"], groups)
    plan = pp.FoldPlan.by_month_and_group(month, codes)
    members: dict[str, list[str]] = {}
    for sid, index in sorted(groups.items()):
        members.setdefault(f"g{index}", []).append(sid)
    return plan, {
        "kind": "month x station group",
        "station_groups": int(args.station_groups),
        "split": (
            "stations sorted by longitude and dealt round-robin, so every "
            "group spans the country"
        ),
        "members": members,
        "unmatched_rows": int((codes < 0).sum()),
        "n_folds": len(plan.folds()),
    }


def protocol_strata(
    features: Mapping[str, Any], dry: np.ndarray, season: Any,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """The strata every arm is scored on, plus the distance-bin names.

    Beside the seasons and the onset-relevant ``dry`` subset, the rows are
    cut by ``ng_near_km`` — how far the point is from the nearest gauge that
    is not itself. That is the axis a gauge row and a subscriber's address
    differ on, and binning on it is what lets the skill be re-weighted from
    the one population to the other.
    """
    strata: dict[str, np.ndarray] = {
        name: np.asarray(season).astype("<U8") == name for name in pp.SEASONS
    }
    if dry.any():
        strata[pp.DRY] = dry
    distance: list[str] = []
    values = features.get(pp.DISTANCE_COLUMN)
    if values is not None and np.any(np.isfinite(np.asarray(values, dtype=float))):
        for name, mask in pp.distance_bins(values).items():
            if mask.any():
                strata[name] = mask
            distance.append(name)
    return strata, distance


def distance_weights(args, features: Mapping[str, Any]) -> dict[str, Any] | None:
    """The two nearest-gauge distributions the expectation stands on.

    ``random_point`` is where people live; ``at_gauge`` is where the
    archive's rows are. Both are reported, because the re-weighting is only
    as good as the overlap between them and a reader should be able to see
    the overlap rather than be told about it.
    """
    if str(args.protocol) != PROTOCOL_RANDOM_POINT:
        return None
    coords = station_coordinates(args.points)
    present = sorted({
        str(sid) for sid in np.unique(np.asarray(features["station_id"]).astype(str))
        if str(sid) in coords
    })
    scored = {sid: coords[sid] for sid in present}
    return {
        "random_point": pp.random_point_distance_weights(
            scored, n=int(args.random_point_samples),
            seed=int(args.random_point_seed),
        ),
        "at_gauge": pp.gauge_distance_weights(scored),
        "column": pp.DISTANCE_COLUMN,
        "stations": len(scored),
    }


def _bin_values(
    per_lead: Mapping[str, Any], lead: int, names: Sequence[str], key: str,
) -> dict[str, float | None]:
    """One statistic per distance bin, for the re-weighting."""
    entry = per_lead.get(str(lead)) or {}
    out: dict[str, float | None] = {}
    for name in names:
        block = entry.get(name)
        if not block:
            out[name] = None
        elif key == "delta":
            difference = block.get("difference") or {}
            interval = difference.get("bss")
            # ``(point, lo, hi)`` — ``benchmark.paired_block_bootstrap``'s
            # order, which ``benchmark_report._ci`` unpacks the same way.
            # Taking [1] here would re-weight the interval's LOWER BOUND
            # and call it the difference.
            out[name] = None if not interval else float(interval[0])
        else:
            out[name] = float(block[key]["bss"])
    return out


def random_point_rows(
    report_leads: Mapping[str, Any],
    leads: Sequence[int],
    names: Sequence[str],
    weights: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Per lead, the distance-binned skill re-weighted to a random point.

    The point estimate only. A confidence interval on a re-weighted mean of
    four paired bootstraps is a harder object than it looks — the bins are
    disjoint so the day blocks are not, and the weights are themselves
    sampled — and inventing one would say more than the evidence does. The
    per-bin intervals are printed above it; this row says where the weight
    of the country sits.
    """
    shares = {
        name: float(weight) for name, weight in zip(
            weights.get("labels") or (), weights.get("weights") or (),
        )
        if name in names
    }
    out: list[dict[str, Any]] = []
    for lead in leads:
        row: dict[str, Any] = {"lead": int(lead)}
        for key, label in (
            ("baseline", "bss_baseline"),
            ("postprocess", "bss_candidate"),
            ("delta", "bss_difference"),
        ):
            got = pp.random_point_expectation(
                _bin_values(report_leads, lead, names, key), shares,
            )
            row[label] = _round(got["value"])
            # Two coverages, because they can differ: a bin with rows
            # always has a BSS, but its paired interval is absent when
            # --resamples is 0, and a ΔBSS standing on fewer bins than
            # the levels beside it must say so rather than borrow their
            # reach.
            row["covered" if key != "delta" else "delta_covered"] = _round(
                got["covered"], 4,
            )
        out.append(row)
    return out


def _distance_section(report: Mapping[str, Any]) -> list[str]:
    """Skill against the one thing a gauge row and an address differ on."""
    names = report["settings"].get("distance_bins") or []
    if not names:
        return []
    lines = [
        "## By distance to the nearest gauge",
        "",
        f"Rows binned on `{pp.DISTANCE_COLUMN}` — the distance from the "
        "point to the nearest gauge that is **not** the point's own. At a "
        "training row that is the nearest OTHER station; at a subscriber's "
        "address it is simply the nearest station, and the two are the same "
        "quantity, which is why a model fitted here can be read there. Skill "
        "should fall with distance; how fast it falls is what the next "
        "section re-weights.",
        "",
        "| lead | bin | n | BSS base | BSS cand. | ΔBSS [CI] | real? |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for lead in report["settings"]["leads"]:
        entry = report["evaluation"]["leads"].get(str(lead)) or {}
        for name in names:
            block = entry.get(name)
            if not block:
                lines.append(f"| {lead} | {name} | – | – | – | – | – |")
                continue
            diff = block.get("difference") or {}
            verdict = diff.get("bss_excludes_zero")
            lines.append(
                f"| {lead} | {name} | {block['n']} | "
                f"{_fmt(block['baseline']['bss'], 4)} | "
                f"{_fmt(block['postprocess']['bss'], 4)} | "
                f"{_ci(diff.get('bss'))} | "
                + ("yes" if verdict else ("no" if verdict is False else "–"))
                + " |"
            )
    lines.append("")
    return lines


def _random_point_section(report: Mapping[str, Any]) -> list[str]:
    """What the distance-binned table is worth to somebody who is not a gauge."""
    block = report.get("distance_weights")
    if not block:
        return []
    random_point = block["random_point"]
    at_gauge = block["at_gauge"]
    labels = random_point["labels"]
    lines = [
        "## Expected at a random point",
        "",
        f"{random_point['n']} points drawn uniformly **by area** inside "
        "Denmark's coastline (`dmi_nowcast_core.denmark_outline`, Natural "
        "Earth 1:10m simplified to ~1 km), each one's distance to the "
        f"nearest of the {block['stations']} scored gauges measured on the "
        "same kilometre grid the `ng_*` features use "
        f"(seed {random_point['seed']}).",
        "",
        "| bin | random point | the archive's rows |",
        "|---|---:|---:|",
    ]
    for index, name in enumerate(labels):
        lines.append(
            f"| {name} | {random_point['weights'][index]:.3f} | "
            f"{at_gauge['weights'][index]:.3f} |"
        )
    closer = float(random_point["km"]["p50"]) < float(at_gauge["km"]["p50"])
    lines += [
        f"| **median km** | **{random_point['km']['p50']:.1f}** | "
        f"**{at_gauge['km']['p50']:.1f}** |",
        "",
        "The two columns are the whole reason this section exists — they "
        "are not the same distribution, and every number above them was "
        "measured on the right-hand one while every number a subscriber "
        "sees is drawn from the left.",
        "",
        (
            "Here a random place in Denmark is CLOSER to a gauge than a "
            "gauge is to its nearest neighbour, which is what a roughly "
            "regular network gives you: a point sits about half a step "
            "from a node while a node sits a whole step from the next "
            "one. The archive is therefore trained and scored under "
            "conditions slightly HARSHER than a subscriber's, and the "
            "re-weighting moves the answer toward the close bins rather "
            "than away from them. The extrapolation is in the near bins, "
            "which carry most of the country and only the handful of "
            "gauges that happen to sit near another one — read their row "
            "counts in the table above before leaning on this number."
            if closer else
            "Here a random place in Denmark is FARTHER from a gauge than "
            "a gauge is from its nearest neighbour, so the archive is "
            "easier than service and the re-weighting moves the answer "
            "toward the far bins. Read those bins' row counts in the "
            "table above before leaning on this number: they are where "
            "the extrapolation lives."
        ),
        "",
        "| lead | BSS base | BSS cand. | ΔBSS | weight covered |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report.get("random_point") or ():
        lines.append(
            f"| {row['lead']} | {_fmt(row.get('bss_baseline'), 4)} | "
            f"{_fmt(row.get('bss_candidate'), 4)} | "
            f"{_fmt(row.get('bss_difference'), 4)} | "
            f"{_fmt(row.get('covered'), 3)}"
            + (
                "" if row.get("delta_covered") == row.get("covered")
                else f" (Δ {_fmt(row.get('delta_covered'), 3)})"
            )
            + " |"
        )
    lines += [
        "",
        "Each number is the per-bin value above, weighted by the random-"
        "point column. `weight covered` is the share of the country the "
        "filled bins account for — a ΔBSS standing on 0.96 of the weight is "
        "a different claim from one standing on 0.5. No interval: the bins "
        "share day blocks and the weights are themselves sampled, so a "
        "bootstrap over the re-weighted mean would claim more than the "
        "evidence supports. The per-bin intervals are the evidence.",
        "",
    ]
    return lines


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
    p.add_argument(
        "--model", default=MODEL_LOGISTIC, choices=list(MODEL_CHOICES),
        help="model family. 'logistic' is the shipped one, one fit per "
             "lead. A '-shared' kind is ONE fit over every served lead "
             "stacked, with the lead in the design (lead_min, "
             "log_lead_min, raw_frac_own and friends) and every other "
             "coefficient shared; only the isotonic recalibration stays "
             "per lead. Both tree options need LightGBM, which lives only "
             "in .venv-fit (see the module docstring); the sidecar "
             "evaluates the exported JSON in numpy and never imports it.",
    )
    p.add_argument(
        "--design", default=pp.DESIGN_V1, choices=list(pp.DESIGN_VERSIONS),
        help="design matrix. v1 is the 27 shipped columns; v2 adds every "
             "feature column the catalogue has grown, a spline basis on "
             "the columns that bend, and the season interactions",
    )
    p.add_argument(
        "--station-offsets", action="store_true",
        help="learn a per-station intercept offset under its own, "
             "stronger ridge; stored as a {station_id: offset} map and "
             "applied only at stations the fit saw (a subscriber gets 0)",
    )
    p.add_argument(
        "--station-l2-multiple", type=float, default=pp.STATION_L2_MULTIPLE,
        help="how much harder the station offsets are shrunk than the slopes",
    )
    p.add_argument(
        "--isotonic", default=pp.ISOTONIC_POOLED,
        choices=list(pp.ISOTONIC_MODES),
        help="recalibration: one curve per lead, or one per season with a "
             f"fallback to the pooled curve below "
             f"{pp.MIN_SEASON_ISOTONIC_ROWS} rows",
    )
    p.add_argument(
        "--trees", default="", metavar="K=V,...",
        help="LightGBM overrides, e.g. "
             "'n_estimators=500,max_depth=6,learning_rate=0.03'. Defaults: "
             + ", ".join(f"{k}={v}" for k, v in DEFAULT_TREE_PARAMS.items()),
    )
    p.add_argument(
        "--baseline", default=BASELINE_CURVE,
        help="what the candidate is compared against: 'curve' (the served "
             "p_rain_<lead>), 'refit-v1' (the v1 design + logistic, "
             "refitted IN FOLD on the same rows — the baseline of record "
             "for post-processing v2), or 'model:<path>' to refit the "
             "configuration a postprocess.json was fitted under",
    )
    p.add_argument(
        "--learning-curve", default="", metavar="N,N,...",
        help="also fit each fold on N of its training days and score the "
             "held-out month, one row per N, e.g. '10,20,30,60,90'. "
             "Costs one full LOMO pass per N.",
    )
    p.add_argument(
        "--learning-curve-order", default=pp.CURVE_ORDER_RANDOM,
        choices=list(pp.CURVE_ORDERS),
        help="how those N days are chosen: 'random' draws them "
             "stratified by month, so less data does not also mean a "
             "different season; 'date' takes the first N calendar days, "
             "which is the seasonal experiment the curve used to run by "
             "accident",
    )
    p.add_argument(
        "--learning-curve-seed", type=int, default=pp.DEFAULT_CURVE_SEED,
        help="seed for the stratified draw, so a curve is reproducible",
    )
    p.add_argument(
        "--compare", default="", metavar="MODEL,...",
        help="also score these arms out of fold on the SAME folds and "
             "against the same baseline, and put them beside the candidate "
             "in the summary table — e.g. "
             "'logistic,logistic-shared,trees,trees-shared'. An arm may "
             "carry its own ablation with a '/drop=' suffix, e.g. "
             "'logistic/drop=gauge+neighbour' for the same family with no "
             "gauge information. Each one costs a full pass; none of them "
             "is written to postprocess.json, which is always --model.",
    )
    p.add_argument(
        "--protocol", default=PROTOCOL_AT_GAUGE, choices=list(PROTOCOL_CHOICES),
        help="what a row is taken to BE. 'at-gauge' is the historical "
             "evaluation: the row is a DMI station, it may read its own "
             "gauge, and the folds hold out a month. 'random-point' treats "
             "it as the address it stands in for — the point's own g_* "
             "block is masked to unknown in training and in scoring, "
             "--station-offsets is refused, and each fold holds out a month "
             "AND a station group, so no prediction comes from a model that "
             "saw that place. The ng_* block is never masked: it is "
             "leave-self-out by construction.",
    )
    p.add_argument(
        "--station-groups", type=int, default=pp.DEFAULT_STATION_GROUPS,
        help="how many station groups the random-point hold-out cuts the "
             "country into. Stations are sorted by longitude and dealt "
             "round-robin, so every group spans the country; no seed, "
             "because the split is a function of the catalogue. K groups x "
             "M months is K*M fits per model arm.",
    )
    p.add_argument(
        "--random-point-samples", type=int, default=200_000,
        help="points drawn inside Denmark's outline to weigh the "
             "distance-binned skill by where people actually live",
    )
    p.add_argument(
        "--random-point-seed", type=int, default=0,
        help="seed for that draw, so the weights are reproducible",
    )
    p.add_argument(
        "--drop-families", default="", metavar="FAMILY,...",
        help="drop these feature families from the candidate's design "
             "before fitting — the ablation's knob as a first-class arm. "
             "e.g. 'gauge,neighbour' for a model with no gauge information "
             "at all. Known families: " + ", ".join(pp.FAMILY_NAMES) + ".",
    )
    p.add_argument(
        "--ablate", action="store_true",
        help="also re-run the LOMO once per feature family with that "
             "family dropped, and report the ΔBSS. Costs one LOMO pass "
             "per family (" + ", ".join(pp.FAMILY_NAMES) + ").",
    )
    p.add_argument(
        "--target", default=TARGET_WET, choices=list(TARGET_CHOICES),
        help="the outcome the model is fitted AND scored on. 'wet' is "
             "Layer B's 'gauge wet within (t, t+L]' — the shipped model. "
             "'onset' is the push's own event: a gauge onset "
             "(threshold_sweep.gauge_truth) in the scorer window "
             "(t, t+L+tolerance]; 0 needs every slot of that window known",
    )
    p.add_argument(
        "--onset-tolerance-min", type=int, default=DEFAULT_TOLERANCE_MIN,
        help="the scorer's tolerance added to the lead under --target onset",
    )
    p.add_argument(
        "--onset-rows", default=ONSET_ROWS_ALL, choices=list(ONSET_ROWS_CHOICES),
        help="under --target onset: 'all' keeps rows whose gauge was wet "
             "in the hour before (label 0 by construction, so the model "
             "learns it); 'dry' restricts training AND scoring to the "
             f"dry-{pp.DRY_BEFORE_MIN} subset",
    )
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
                        "probabilities added as p_post_<lead> (p_onset_<lead> "
                        "under --target onset); score it with "
                        "benchmark_report.py --probability-column "
                        "'p_post_{lead}' (or 'p_onset_{lead}'). Never writes "
                        "into the run.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    log = bench._log_to(sys.stderr)

    try:
        leads = parse_leads(args.leads)
        design_leads = parse_leads(args.design_leads)
        stations = bench.load_points(args.points)
        settings = settings_from_args(args)
        baseline_config = baseline_settings_from_args(args)
        curve_days = [
            int(v) for v in str(args.learning_curve or "").split(",") if v.strip()
        ]
        compare = parse_compare(args.compare, args)
        if str(args.protocol) == PROTOCOL_RANDOM_POINT:
            if args.station_offsets:
                raise ValueError(
                    "--station-offsets cannot be used with --protocol "
                    f"{PROTOCOL_RANDOM_POINT}: a per-station intercept is "
                    "learnable at a gauge and meaningless at an address, so "
                    "a number measured with it is not a number a subscriber "
                    "would get. Drop the flag, or use --protocol "
                    f"{PROTOCOL_AT_GAUGE}."
                )
            # Read here so a missing or unreadable points file is a message
            # before the loader spends minutes on parquet.
            station_coordinates(args.points)
    except (ValueError, OSError) as exc:
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
    target = str(args.target)
    if target != TARGET_ONSET and str(args.onset_rows) != ONSET_ROWS_ALL:
        print(
            "error: --onset-rows applies to --target onset only",
            file=sys.stderr,
        )
        return 2
    if args.onset_tolerance_min < 0:
        print("error: --onset-tolerance-min must not be negative", file=sys.stderr)
        return 2
    column_template = pp.target_column_template(target)

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

    features = build_features(rows)
    # The station id every row was scored at, for the learned offsets and
    # for a reader of the artefact. Taken AFTER the recode, so a row the
    # grid dropped carries the station it was actually graded against.
    station_ids = np.asarray(scored + [""], dtype=object)[
        np.clip(np.asarray(rows["station"], dtype=np.int64), 0, len(scored))
    ]
    features["station_id"] = station_ids.astype(str)
    baseline = {int(lead): rows["p"][int(lead)] for lead in leads}
    month = pp.year_months_from_epoch(rows["t"])
    day = rows["t"] // bench.DAY_SEC

    # The onset-relevant subset. The stored column when the replay wrote
    # it, otherwise derived from the gauge grid through the SAME function
    # the writer's definition comes from — never a third opinion.
    dry = pp.dry_subset(
        features, grid=grid, t=rows["t"], station=rows["station"],
    )
    if target == TARGET_ONSET:
        # The push's event, from the push scorer's own onset list: the same
        # window, pad, onset rule and dead-gauge rule the threshold sweep
        # grades a warning with. The grid above is still needed — it is
        # what says whether a window without an onset was fully reported.
        onsets, _known_until, _slots, _dead = gauge_truth(
            Path(args.corpus_dir), sorted(rows["stations"]),
            bench.decision_window(rows["t"]),
            dry_min=int(args.dry_min), onset_min_mm=float(args.onset_min_mm),
            min_known_slots=int(args.min_known_slots), log=log,
        )
        truth = build_onset_truth(
            rows, grid, onsets, scored, leads,
            tolerance_min=int(args.onset_tolerance_min),
            restrict=dry if str(args.onset_rows) == ONSET_ROWS_DRY else None,
        )
        del onsets
    del grid
    counts = target_counts(truth, int(rows["rows"]))
    for lead, entry in counts.items():
        log(
            f"target {target} lead {lead}: {entry['positives']} positive(s) "
            f"of {entry['rows']} gradable row(s), {entry['excluded']} excluded"
        )
    log(
        f"onset-relevant (gauge dry for {pp.DRY_BEFORE_MIN} min before the "
        f"decision): {int(dry.sum())} of {rows['rows']} row(s)"
        + ("" if pp.DRY_COLUMN in features else f" — derived, no {pp.DRY_COLUMN} column")
    )
    season = features.get("season")

    # The protocol, applied in this order and no other: the onset-relevant
    # subset is TRUTH about the point and is derived from the gauge above,
    # before the same gauge is taken away from the model as a PREDICTOR.
    # Reversing the two would empty the dry subset under the mask.
    if str(args.protocol) == PROTOCOL_RANDOM_POINT:
        features = pp.mask_own_gauge(features)
        log(
            "protocol random-point: masked the point's own gauge "
            f"({', '.join(pp.OWN_GAUGE_COLUMNS)}, {pp.GAUGE_KNOWN_COLUMN}=0) "
            "in training and in scoring; the dry subset above was derived "
            "before the mask"
        )
    strata, distance_names = protocol_strata(features, dry, season)
    folds, fold_block = protocol_folds(args, features, month)
    weights_block = distance_weights(args, features)
    log(
        f"protocol {args.protocol}: {len(folds.folds())} fold(s) over "
        f"{' x '.join(folds.names)}"
    )
    if distance_names:
        log(
            "distance strata: "
            + ", ".join(
                f"{name} n={int(strata[name].sum())}"
                for name in distance_names if name in strata
            )
        )

    log(f"fitting {len(leads)} lead(s) over {rows['rows']} row(s)")
    evaluation = pp.leave_one_month_out(
        features, truth, leads,
        month=month, day=day, baseline=baseline, folds=folds,
        design_leads=design_leads, strata=strata,
        n_resamples=int(args.resamples), seed=int(args.seed), ci=float(args.ci),
        settings=settings, baseline_settings=baseline_config,
        baseline_label=str(args.baseline), log=log,
    )
    # The rows the table above scored: whichever baseline it was paired
    # against decides them, and the curve and the ablation are read
    # beside that table, so they grade the same rows and not a wider set
    # of their own.
    scored_against = evaluation["baseline_out_of_fold"] or baseline
    subsets = {pp.POOLED: np.ones(dry.size, dtype=bool), **(
        {pp.DRY: dry} if dry.any() else {}
    )}
    curve_rows: list[dict] = []
    if curve_days:
        if str(args.protocol) != PROTOCOL_AT_GAUGE:
            log(
                "learning curve: month folds only — it answers 'how much "
                "archive', which the spatial hold-out does not change"
            )
        curve_rows = pp.learning_curve(
            features, truth, leads,
            month=month, day=day, settings=settings, days=curve_days,
            design_leads=design_leads, subsets=subsets,
            also_finite=scored_against,
            order=str(args.learning_curve_order),
            seed=int(args.learning_curve_seed),
            log=log,
        )
    arms: list[dict] = []
    candidate = arm_label(str(args.model), settings.drop_families)
    for label, arm_settings in compare:
        if label == candidate:
            continue
        log(f"comparison arm: {label}")
        arm = pp.leave_one_month_out(
            features, truth, leads,
            month=month, day=day, baseline=baseline, folds=folds,
            design_leads=design_leads, strata=strata,
            n_resamples=int(args.resamples), seed=int(args.seed),
            ci=float(args.ci),
            settings=arm_settings,
            baseline_settings=baseline_config,
            baseline_label=str(args.baseline), log=None,
        )
        arms.append({
            "model": label, "fit": arm_settings.to_json(),
            "leads": _strip_arrays(arm)["leads"],
        })
    ablation_block: dict | None = None
    if args.ablate:
        ablation_block = pp.ablation(
            features, truth, leads,
            month=month, settings=settings, design_leads=design_leads,
            subsets=subsets, folds=folds, also_finite=scored_against,
            log=log,
        )
    window = bench.decision_window(rows["t"])
    model = pp.fit_postprocess(
        features, truth, leads,
        design_leads=design_leads, settings=settings,
        # The protocol as a FIELD of the model, not only a note in the
        # provenance below: `features` above was masked in place for
        # random-point, and the model has to keep masking when it is
        # served or the rows it scores stop matching the rows it saw.
        protocol=str(args.protocol),
        target=target,
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
            "settings": settings.to_json(),
            # The protocol belongs in the ARTEFACT, not only in the
            # report: a model fitted under 'random-point' was trained
            # with the point's own gauge masked away and cannot use it in
            # service either, and somebody reading postprocess.json a
            # month from now has to be able to see that without going
            # back to the run that made it.
            "protocol": str(args.protocol),
            "folds": fold_block,
            # What the probabilities are OF, and under onset the window
            # and the rows the labels were drawn on.
            "target": target,
            "onset_tolerance_min": (
                int(args.onset_tolerance_min) if target == TARGET_ONSET else None
            ),
            "onset_rows": (
                str(args.onset_rows) if target == TARGET_ONSET else None
            ),
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
            "baseline_column": (
                "p_rain_<lead>" if baseline_config is None
                else f"{args.baseline} (refitted in fold)"
            ),
            "model": str(args.model),
            "target": target,
            "onset_tolerance_min": (
                int(args.onset_tolerance_min) if target == TARGET_ONSET else None
            ),
            "onset_rows": (
                str(args.onset_rows) if target == TARGET_ONSET else None
            ),
            "column_template": column_template,
            "arm": candidate,
            "protocol": str(args.protocol),
            "folds": fold_block,
            "drop_families": list(settings.drop_families),
            "distance_bins": distance_names,
            "design": str(args.design),
            "isotonic": str(args.isotonic),
            "station_offsets": bool(args.station_offsets),
            "baseline": str(args.baseline),
            "fit": settings.to_json(),
            "baseline_fit": (
                None if baseline_config is None else baseline_config.to_json()
            ),
            "learning_curve_days": list(curve_days),
            "learning_curve_order": str(args.learning_curve_order),
            "learning_curve_seed": int(args.learning_curve_seed),
            "ablate": bool(args.ablate),
            "compare": [label for label, _settings in compare],
        },
        "rows": int(rows["rows"]),
        "stations": len(scored),
        "days": int(np.unique(day).size),
        "n_months": len(set(int(m) for m in np.unique(month))),
        "window": {"from": window[0].isoformat(), "to": window[1].isoformat()},
        "target_counts": counts,
        "dry_rows": int(dry.sum()),
        "dry_source": (
            "column" if pp.DRY_COLUMN in features else "derived from the gauge store"
        ),
        "dead_gauges": dead_rows,
        "run_settings": bench.run_settings(run_dirs),
        "evaluation": _strip_arrays(evaluation),
        "distance_weights": weights_block,
        "random_point": (
            random_point_rows(
                evaluation["leads"], leads, distance_names,
                weights_block["random_point"],
            )
            if weights_block and distance_names else []
        ),
        "arms": arms,
        "learning_curve": curve_rows,
        "ablation": ablation_block,
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
                column_template=column_template,
            )
        except SweepError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        copy_summary(run_dirs, Path(args.write_back), {
            "source": [str(d) for d in run_dirs],
            "column_template": column_template,
            "target": target,
            "onset_tolerance_min": (
                int(args.onset_tolerance_min) if target == TARGET_ONSET else None
            ),
            "onset_rows": (
                str(args.onset_rows) if target == TARGET_ONSET else None
            ),
            "leads": list(leads),
            "kind": "out-of-fold (leave-one-month-out)",
            "model": str(out_dir / "postprocess.json"),
            "fitted_at_utc": model.fitted_at_utc,
            **written,
        })
        log(
            f"wrote {written['files']} file(s) to {args.write_back}; score "
            "them with benchmark_report.py --probability-column "
            f"'{column_template}'"
        )

    log(f"done in {time.time() - started:.1f}s")
    print(json.dumps(_headline(report), indent=2, default=str))
    return 0


def _headline(report: Mapping[str, Any]) -> dict:
    out: dict[str, Any] = {
        "protocol": report["settings"].get("protocol", PROTOCOL_AT_GAUGE),
        "target": report["settings"].get("target", TARGET_WET),
        "arm": report["settings"].get("arm"),
        "folds": (report["settings"].get("folds") or {}).get("kind"),
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
        # The onset-relevant half of the gate, beside the pooled one: a
        # win on all rows that is not a win here is a win on rows nobody
        # was going to be warned about.
        dry = (report["evaluation"]["leads"].get(str(lead)) or {}).get(pp.DRY)
        if dry:
            out["leads"][str(lead)]["dry"] = {
                "n": dry["n"],
                "bss_baseline": _round(dry["baseline"]["bss"]),
                "bss_postprocess": _round(dry["postprocess"]["bss"]),
                "bss_difference": (dry.get("difference") or {}).get("bss"),
            }
    if report.get("random_point"):
        # The number the protocol exists to produce, in the one place a
        # reader of stdout will look.
        out["random_point"] = report["random_point"]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
