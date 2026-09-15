"""Build an offline event-review bundle from the corpus (Phase: review tool).

The served push rule scores POD 0.38 / FAR 0.67 against gauge onsets, and
nobody has ever looked at an individual bad warning. A pooled F1 cannot
tell a forecast that invented rain from a shower that passed four
kilometres north of the gauge, and those two need opposite fixes. This
script draws a few hundred of those events, assembles everything needed to
judge each one by eye, renders the radar loop around it, and writes the lot
as a self-contained directory.

Run it on the VM, where the corpus lives; copy the directory to a laptop;
serve it with ``scripts/review_server.py`` and review at ``/review/`` in the
dev frontend. Nothing in the bundle points back at the VM, because the
corporate VPN blocks it and the composite archive is 7 GB.

The work is split three ways and this script is only the seam:
``dmi_nowcast_sidecar.review`` produces the population, the sample and the
per-event documents; ``dmi_nowcast_sidecar.review_frames`` plans and renders
the imagery; ``dmi_nowcast_core.review_schema`` is the contract all of them
and the browser agree on. What is genuinely this script's own is the
manifest — the provenance that lets a reader work out, months later,
whether a finding was real or an artefact of how the bundle was drawn.

It needs core AND sidecar on the path (``push.engine`` for the decision,
``national_artifacts`` for the quantisation), so it runs out of the sidecar
environment — the same one ``scripts/replay_warnings.py`` needs, and the one
the VM has::

    sidecar/.venv/bin/python scripts/build_review_bundle.py ...

Typical run::

    python scripts/build_review_bundle.py \\
        --corpus-dir /var/lib/dmi-nowcast-corpus \\
        --decisions-dir stations/replay_postprocess/decisions \\
        --thresholds /var/lib/dmi-nowcast/push_thresholds.json \\
        --from 2025-12-01 --to 2026-08-31 \\
        --events 300 --seed 20260916 \\
        --out-dir ~/review/bundle-20260916

And, with no VM and no HDF5 at all, a tiny synthetic bundle for developing
the UI against::

    python scripts/build_review_bundle.py --fixture --out-dir /tmp/review-fixture

Two flags decide whether the numbers mean anything
--------------------------------------------------

``--decisions-dir`` is repeatable and is passed to ``load_decisions`` in the
order given, which dedupes ``(radar_ts, station_id)`` with the LAST
directory winning. So the MOST authoritative tree goes last. Getting this
backwards lets a live row whose feature columns were nulled beat the
out-of-fold replay row for the same instant, which silently changes the
probability scale the rule was applied on. The effective order is recorded
in the manifest.

``--thresholds`` and the probability column decide whether the sample is
honest. The post-processing model was fitted on all months, so re-deciding
an event with it means the probability was partly memorised — which biases
the sample toward the model's residual failures and makes false alarms look
rarer and stranger than they are. Point ``--decisions-dir`` at a tree
written by ``fit_postprocess.py --write-back`` (out-of-fold ``p_post``) and
pass no ``--postprocess-model``; the manifest then records
``probability_provenance: out_of_fold``. Anything else is recorded as
in-sample and carries a caveat, because a bundle that cannot say which it
was is a bundle whose findings cannot be defended.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "sidecar") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "sidecar"))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from dmi_nowcast_core import review_schema  # noqa: E402
from dmi_nowcast_sidecar import review, review_frames  # noqa: E402


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {message}", flush=True)


def _utc_day(text: str) -> datetime:
    """``YYYY-MM-DD`` or a full ISO instant → an aware UTC datetime."""
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover - argparse reports it
        raise argparse.ArgumentTypeError(f"not a date/instant: {text!r}") from exc
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _json_default(value: Any) -> str:
    """Datetimes as strict ISO 8601 with an explicit offset.

    ``str(datetime)`` produces a SPACE separator, not a ``T``. The contract
    promises ISO 8601 and some JS engines refuse the space, so this is not
    cosmetic: it is the difference between a timestamp the browser parses
    and an Invalid Date rendered as "NaN minutes before the onset".
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"naive datetime in the bundle: {value!r}")
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json_atomic(path: Path, payload: Any) -> int:
    """Write JSON through a temp file and rename. Returns bytes written.

    Atomic because a bundle build runs for hours over thousands of frames
    and may be interrupted: a half-written ``events.json`` that still
    parses is far worse than one that is missing, since the reviewer would
    judge a truncated sample without knowing it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=1, sort_keys=False, default=_json_default).encode()
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)
    return len(data)


def _git_provenance() -> dict:
    """Commit and dirty flag, so a finding can be traced to the code.

    A dirty tree does not stop the build — a reviewer often wants a bundle
    from a work in progress — but it is recorded, because "reproducible
    from this commit" is then not true and nobody should later assume it.
    """
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                args, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _run("git", "rev-parse", "HEAD")
    status = _run("git", "status", "--porcelain")
    return {
        "script": "scripts/build_review_bundle.py",
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "argv": sys.argv[1:],
        "host": platform.node(),
        "python": platform.python_version(),
    }


def _bundle_id(seed: int, population_hash: str, built_at: datetime) -> str:
    """A short, stable name for this draw.

    It keys the annotation database, so two bundles must never collide and
    one bundle must keep its id across a resumed build. Derived from the
    seed and the population it drew from rather than from the wall clock
    alone, so re-running the same draw over the same corpus reproduces the
    id and the reviewer's judgements still attach.
    """
    digest = hashlib.sha256(
        f"{seed}:{population_hash}".encode(),
    ).hexdigest()[:6]
    return f"review-{built_at:%Y%m%d}-{digest}"


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------

def load_stations(points_file: Path, catalogue: Path | None) -> list:
    """``station_points.json`` (+ optional catalogue) → ``StationMeta`` list.

    The points file carries the id, position and region the replay scored
    on; the catalogue adds the human name. The name is cosmetic — but a
    reviewer looking at three hundred events remembers "Årslev", not
    "06074", and a bundle that only has numbers is materially harder to
    find patterns in.
    """
    if not points_file.is_file():
        raise SystemExit(
            f"station points file not found: {points_file}\n"
            "Pass --points, or --corpus-dir pointing at the corpus root "
            "(it is expected at <corpus>/stations/station_points.json)."
        )
    document = json.loads(points_file.read_text(encoding="utf-8"))
    points = document.get("points") or document.get("stations") or []
    names: dict[str, str] = {}
    if catalogue is not None and catalogue.is_file():
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(catalogue, columns=["station_id", "name"])
            names = {
                str(sid): str(name)
                for sid, name in zip(
                    table.column("station_id").to_pylist(),
                    table.column("name").to_pylist(),
                )
                if name
            }
        except Exception as exc:  # noqa: BLE001 - names are cosmetic
            _log(f"station catalogue unreadable ({exc}); using ids as names")

    stations = []
    for point in points:
        station_id = str(point["id"] if "id" in point else point["station_id"])
        stations.append(review.station_meta(
            station_id=station_id,
            name=names.get(station_id, station_id),
            lat=float(point["lat"]),
            lon=float(point["lon"]),
            region=point.get("region"),
        ))
    if not stations:
        raise SystemExit(f"no stations in {points_file}")
    return stations


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from station 1 to station 2, degrees true.

    Computed here rather than carried on ``review.Neighbour`` because it is
    presentation, not truth. It earns its place because "the wet neighbour
    was UPWIND" and "the wet neighbour was downwind" separate a cell that
    diverted around this gauge from one that died before reaching it —
    which are two different tags and two different fixes.
    """
    import math

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def stations_document(stations: Sequence, population) -> dict:
    """``stations.json`` — metadata plus the neighbour graph.

    The graph is shared rather than repeated inside each event document:
    a station's neighbours do not depend on the event, and at ~100 stations
    the adjacency is a few tens of kilobytes against a few tens of
    megabytes if every event carried its own copy.
    """
    by_id = {station.station_id: station for station in stations}

    def _neighbours(station) -> list[dict]:
        out = []
        for neighbour in population.neighbours.get(station.station_id, ()):
            other = by_id.get(neighbour.station_id)
            out.append({
                "station_id": neighbour.station_id,
                "name": neighbour.name,
                "lat": other.lat if other else None,
                "lon": other.lon if other else None,
                "distance_km": neighbour.distance_km,
                "bearing_deg": (
                    _bearing_deg(station.lat, station.lon, other.lat, other.lon)
                    if other else None
                ),
            })
        return out

    return {
        "schema_version": review_schema.REVIEW_SCHEMA_VERSION,
        "neighbour_radius_km": review_schema.NEIGHBOUR_RADIUS_KM,
        "stations": [
            {
                "station_id": station.station_id,
                "name": station.name,
                "lat": station.lat,
                "lon": station.lon,
                "region": station.region,
                "neighbours": _neighbours(station),
            }
            for station in stations
        ],
    }


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------

def build(args: argparse.Namespace) -> dict:
    """Draw, assemble, render and write. Returns the manifest."""
    built_at = datetime.now(timezone.utc)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_dir = Path(args.corpus_dir).expanduser() if args.corpus_dir else None

    # -- population --------------------------------------------------------
    if args.fixture:
        _log("fixture mode: synthesising a population (no corpus, no HDF5)")
        population = review.synthetic_population(seed=args.seed)
        stations = list(population.stations.values())
        corpus_dir = Path(
            args.fixture_corpus
            or Path(tempfile.mkdtemp(prefix="review-fixture-corpus-")),
        )
    else:
        if corpus_dir is None:
            raise SystemExit("--corpus-dir is required unless --fixture")
        points_file = (
            Path(args.points).expanduser() if args.points
            else corpus_dir / "stations" / "station_points.json"
        )
        catalogue = (
            Path(args.catalogue).expanduser() if args.catalogue
            else corpus_dir / "stations" / "catalogue.parquet"
        )
        rule = _resolve_rule(args)
        stations = load_stations(points_file, catalogue)
        _log(f"{len(stations)} station(s) from {points_file}")

        decisions_dirs = [
            (corpus_dir / d).expanduser() if not Path(d).is_absolute()
            else Path(d).expanduser()
            for d in args.decisions_dir
        ]
        for directory in decisions_dirs:
            if not directory.is_dir():
                raise SystemExit(f"decisions dir not found: {directory}")
        _log(
            "decision trees, least authoritative first: "
            + " -> ".join(str(d) for d in decisions_dirs)
        )
        population = review.load_population(
            decisions_dirs=decisions_dirs,
            decisions_labels=[_tree_label(d) for d in decisions_dirs],
            corpus_dir=corpus_dir,
            stations=stations,
            rule=rule,
            window=(args.window_from, args.window_to),
            window_min=args.window_min,
            allow_feature_gap=args.allow_feature_gap,
            postprocess_model=(
                Path(args.postprocess_model).expanduser()
                if args.postprocess_model else None
            ),
            log=_log,
        )

    _log(
        f"population: {len(population.records)} candidate event(s); "
        f"excluded {population.excluded}"
    )

    # -- sample ------------------------------------------------------------
    class_targets = _class_targets(args)
    sample = review.stratify(
        population.records,
        seed=args.seed,
        target=None if class_targets else args.events,
        class_targets=class_targets,
        floor_per_cell=args.floor_per_cell,
    )
    _log(
        f"sampled {len(sample.records)} event(s) across {len(sample.cells)} cell(s); "
        f"population_hash {sample.population_hash[:16]}…"
    )
    bundle_id = args.bundle_id or _bundle_id(
        args.seed, sample.population_hash, built_at,
    )

    # -- frames ------------------------------------------------------------
    plan = review_frames.frame_plan(
        [(record.event_id, record.anchor_utc) for record in sample.records],
        window_min=args.window_min,
        pad_min=args.frame_pad_min,
        include_doppler=args.include_doppler,
    )
    _log(
        f"frame plan: {plan.stamps_total} slot(s) -> {plan.stamps_unique} unique "
        f"({plan.dedup_saving_pct:.0f}% deduped)"
    )
    if args.fixture:
        corpus_dir.mkdir(parents=True, exist_ok=True)
        review_frames.write_fixture_archive(corpus_dir, plan.stamp_strings())

    if args.no_frames:
        _log("--no-frames: skipping rendering")
        frame_result = None
    else:
        started = time.monotonic()
        frame_result = review_frames.write_frames(
            plan, corpus_dir=corpus_dir, bundle_dir=out_dir,
        )
        _log(
            f"frames: {len(frame_result.written)} written, "
            f"{len(frame_result.skipped)} already present, "
            f"{len(frame_result.missing)} missing, "
            f"{frame_result.bytes_total / 1e6:.0f} MB total "
            f"in {time.monotonic() - started:.0f}s"
        )

    # -- per-event documents ----------------------------------------------
    events_dir = out_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    by_event = plan.by_event_id()
    # An EMPTY list and None mean different things downstream: [] is "the
    # renderer ran and nothing was missing", so every frame is present;
    # None is "nobody rendered", so presence is unknown and stays null.
    # Collapsing them would claim imagery a --no-frames bundle has not got.
    missing_stamps = sorted(frame_result.missing) if frame_result else None
    station_by_id = {station.station_id: station for station in stations}

    planned_stamps = {stamp.stamp: stamp for stamp in plan.stamps}
    station_pixel = _station_pixel_mapper(
        frame_result.grid if frame_result else None,
    )

    index_rows: list[dict] = []
    detail_bytes = 0
    for record in sample.records:
        event_frames = by_event.get(record.event_id)
        detail = review.build_event(
            record, population,
            bundle_id=bundle_id,
            window_min=args.window_min,
            frame_pad_min=args.frame_pad_min,
            event_frames=event_frames,
            planned_stamps=planned_stamps,
            missing_stamps=missing_stamps,
            station_pixel=station_pixel,
        )
        detail_bytes += _write_json_atomic(
            events_dir / f"{record.event_id}.json", detail,
        )
        index_rows.append(detail["index"])
    _log(
        f"{len(index_rows)} event document(s), "
        f"{detail_bytes / 1e6:.1f} MB ({detail_bytes // max(len(index_rows), 1)} B each)"
    )

    # -- the shared documents ---------------------------------------------
    _write_json_atomic(out_dir / "events.json", {
        "schema_version": review_schema.REVIEW_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "events": index_rows,
    })
    # Only the stations the sample actually reached, plus every neighbour
    # they reference — a bundle that shipped all ~100 would carry adjacency
    # for stations no event mentions.
    referenced = {r.station_id for r in sample.records}
    for station_id in list(referenced):
        for neighbour in population.neighbours.get(station_id, ()):
            referenced.add(neighbour.station_id)
    _write_json_atomic(out_dir / "stations.json", stations_document(
        [s for s in stations if s.station_id in referenced] or list(stations),
        population,
    ))
    _write_json_atomic(out_dir / "tags.json", review_schema.tags_document())

    manifest = _manifest(
        args, population, sample, plan, frame_result,
        bundle_id=bundle_id, built_at=built_at, n_events=len(index_rows),
    )
    _write_json_atomic(out_dir / "manifest.json", manifest)
    return manifest


def _load_fold_thresholds(path: str | None) -> dict[tuple[int, int], int] | None:
    """``{"2026-03": 60, ...}`` → ``{(2026, 3): 60}``.

    Leave-one-month-out thresholds are an INPUT, not something this script
    can derive: fitting one fold is a full threshold sweep over every other
    month, which is hours of work and belongs to the caller that can afford
    it. Passing ``--rule-source lomo`` without them is refused up front
    rather than deep inside the rule's own validation.
    """
    if path is None:
        return None
    document = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    table: dict[tuple[int, int], int] = {}
    for key, value in document.items():
        year, _, month = str(key).partition("-")
        table[(int(year), int(month))] = int(
            value if isinstance(value, (int, float)) else value["threshold_pct"]
        )
    if not table:
        raise SystemExit(f"no folds in {path}")
    return table


def _resolve_rule(args: argparse.Namespace):
    """The rule the events are re-decided under, with its provenance.

    ``probability_provenance`` is the field a later reader needs most: an
    out-of-fold tree means the post-processing model never saw the event it
    is being judged on, and anything else means it partly did.
    """
    provenance = (
        "in_sample_fill" if args.postprocess_model
        else args.probability_provenance
    )
    thresholds = {}
    if args.thresholds:
        document = json.loads(
            Path(args.thresholds).expanduser().read_text(encoding="utf-8"),
        )
        thresholds = document
    folds = _load_fold_thresholds(args.fold_thresholds)
    if args.rule_source == review.RULE_LOMO and folds is None:
        raise SystemExit(
            "--rule-source lomo needs --fold-thresholds: one threshold per "
            "(year, month), each fitted WITHOUT that month. Fitting them is a "
            "full sweep per fold, so it is an input here, not a side effect."
        )
    if args.rule_source != review.RULE_LOMO and folds is not None:
        raise SystemExit("--fold-thresholds only applies to --rule-source lomo")

    return review.ReviewRule(
        lead_min=args.lead_min,
        threshold_pct=_threshold_for_lead(thresholds, args.lead_min, args.threshold_pct),
        threshold_source="table" if thresholds else "explicit",
        probability=args.probability,
        probability_provenance=provenance,
        rule_source=args.rule_source,
        min_useful_lead_min=args.min_useful_lead_min,
        fold_thresholds=review.fold_thresholds_of(folds),
    )


def _threshold_for_lead(document: dict, lead_min: int, fallback: int | None) -> int:
    """Pull one lead's percent out of a served threshold table."""
    if not document:
        if fallback is None:
            raise SystemExit("--thresholds or --threshold-pct is required")
        return int(fallback)
    leads = document.get("leads") or document.get("by_lead_pct") or {}
    if isinstance(leads, dict):
        for key in (str(lead_min), lead_min):
            if key in leads:
                entry = leads[key]
                return int(
                    entry if isinstance(entry, (int, float))
                    else entry.get("threshold_pct")
                )
    if isinstance(leads, list):
        for entry in leads:
            if int(entry.get("lead_min", -1)) == lead_min:
                return int(entry["threshold_pct"])
    if fallback is not None:
        return int(fallback)
    raise SystemExit(f"no threshold for lead {lead_min} in the table")


def _tree_label(directory: Path) -> str:
    """A short, readable name for one decision tree.

    ``row_source`` on every decision entry answers "did this row come from
    the replay or from the live service?", which matters because the live
    rows are the ones that lost their feature columns. The raw directory
    name is the honest answer and reads better than a full path in a UI
    badge, so ``.../stations/eval`` becomes ``live`` and
    ``.../stations/replay_postprocess/decisions`` becomes
    ``replay_postprocess``.
    """
    parts = [part for part in directory.parts if part not in ("decisions",)]
    tail = parts[-1] if parts else str(directory)
    return "live" if tail == "eval" else tail


def _class_targets(args: argparse.Namespace) -> dict[str, int] | None:
    if not args.class_target:
        return None if args.events else dict(review.DEFAULT_CLASS_TARGETS)
    targets = dict(review.DEFAULT_CLASS_TARGETS)
    for item in args.class_target:
        group, _, count = item.partition("=")
        targets[group.strip()] = int(count)
    return targets


def _manifest(
    args, population, sample, plan, frame_result, *,
    bundle_id: str, built_at: datetime, n_events: int,
) -> dict:
    """Everything a reader needs to decide whether to trust the findings.

    The caveats are not decoration. Each one names a way this bundle could
    support a confident wrong conclusion — an in-sample probability, a
    threshold table fitted on the months it scores, live rows whose feature
    columns were nulled — and they travel with the data so a reviewer
    coming back in six months does not have to remember.
    """
    provenance = dict(population.provenance)
    rule_block = dict(provenance.get("rule") or {})

    # The builder emits its standing method caveats as prose — facts about
    # how this kind of bundle is made, true of every one of them (the radar
    # is not independent, the composite is column-max, an onset is a slot
    # end). The ones added below are different in kind: they are defects of
    # THIS draw, and a reviewer must be able to filter on them, so they
    # carry real codes and a severity.
    caveats: list[dict] = [
        entry if isinstance(entry, dict)
        else {"code": "method_note", "severity": "low", "detail": str(entry)}
        for entry in (provenance.get("caveats") or [])
    ]

    if not rule_block.get("held_out", False) and args.thresholds:
        caveats.append({
            "code": "in_sample_thresholds",
            "severity": "medium",
            "detail": (
                "The threshold table was fitted over the same months these "
                "events come from. Four numbers over ten months is low "
                "capacity, so the leak is small, but the F1 that justifies "
                "the rule is in-sample. Re-run with --rule-source lomo for "
                "the held-out variant."
            ),
        })
    if args.postprocess_model:
        caveats.append({
            "code": "in_sample_probability",
            "severity": "high",
            "detail": (
                "p_post was filled from a model fitted on all months, so the "
                "probability at each event is partly memorised. That biases "
                "the sample toward the model's residual failures and makes "
                "false alarms look rarer and stranger than they are. Prefer a "
                "tree written by fit_postprocess.py --write-back."
            ),
        })
    manifest = {
        "schema_version": review_schema.REVIEW_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "built_at_utc": built_at,
        "builder": _git_provenance(),
        "corpus": provenance.get("corpus", {}),
        "window": provenance.get("window", {}),
        "rule": rule_block,
        "truth": provenance.get("truth", {}),
        "features": provenance.get("features", {}),
        "sampling": sample.document(),
        "events": {"count": n_events},
        "feature_doc": _feature_doc(population),
        "caveats": caveats,
    }
    if frame_result is not None:
        manifest["grid"] = frame_result.grid
        manifest["frames"] = review_frames.frames_manifest_block(plan, frame_result)
    else:
        manifest["grid"] = None
        manifest["frames"] = {
            "count": 0, "bytes": 0, "missing": [],
            "note": "built with --no-frames; the scrubber has no imagery",
        }
    return manifest


def _station_pixel_mapper(grid: dict | None):
    """``(lat, lon) -> (row, col)`` on the product grid, or None.

    The exact inverse the manifest documents for the browser:
    ``col = (x - x_ul_m) / pixel_scale_x_m`` and
    ``row = (y_ul_m - y) / pixel_scale_y_m``. Fractional on purpose — a
    station snapped to a pixel centre can sit up to a kilometre from where
    it really is, which at a 1 km verification disc is the whole quantity.

    Returns None when there is no grid (``--no-frames``), and the bundle
    then carries an honest null rather than a plausible position.
    """
    if not grid:
        return None
    try:
        from pyproj import CRS, Transformer
    except ImportError:  # pragma: no cover - pyproj is a core runtime dep
        return None

    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_proj4(grid["proj4"]), always_xy=True,
    )
    x_ul, y_ul = float(grid["x_ul_m"]), float(grid["y_ul_m"])
    sx, sy = float(grid["pixel_scale_x_m"]), float(grid["pixel_scale_y_m"])

    def mapper(lat: float, lon: float) -> tuple[float, float]:
        x, y = transformer.transform(lon, lat)
        return ((y_ul - y) / sy, (x - x_ul) / sx)

    return mapper


def _feature_doc(population) -> dict:
    """Column → prose, for the feature panel's tooltips."""
    try:
        from dmi_nowcast_core import postprocess

        return postprocess.feature_documentation(population.design_leads)
    except Exception:  # noqa: BLE001 - tooltips are not worth failing a build
        return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", required=True, help="bundle directory to write")
    p.add_argument("--corpus-dir", help="e.g. /var/lib/dmi-nowcast-corpus")
    p.add_argument(
        "--decisions-dir", action="append", default=[],
        help=(
            "decision parquet tree, relative to --corpus-dir or absolute. "
            "Repeatable; pass LEAST authoritative first, because dedup on "
            "(radar_ts, station_id) lets the last one win."
        ),
    )
    p.add_argument("--points", help="station_points.json (default: under the corpus)")
    p.add_argument("--catalogue", help="catalogue.parquet, for station names")
    p.add_argument("--thresholds", help="push_thresholds.json")
    p.add_argument("--threshold-pct", type=int, help="explicit threshold, overrides the table")
    p.add_argument("--lead-min", type=int, default=30)
    p.add_argument("--probability", default="postprocess", choices=["postprocess", "curve"])
    p.add_argument(
        "--probability-provenance", default="out_of_fold",
        choices=["out_of_fold", "in_sample_fill", "stored", "mixed"],
        help="what the decision tree's p_post column actually is",
    )
    p.add_argument("--postprocess-model", help="fill p_post from this model (IN-SAMPLE)")
    p.add_argument("--rule-source", default="served", choices=["served", "lomo"])
    p.add_argument(
        "--fold-thresholds",
        help='{"YYYY-MM": pct} JSON, required by --rule-source lomo',
    )
    p.add_argument("--min-useful-lead-min", type=float, default=5.0)
    p.add_argument("--from", dest="window_from", type=_utc_day, default=_utc_day("2025-12-01"))
    p.add_argument("--to", dest="window_to", type=_utc_day, default=None)
    p.add_argument("--events", type=int, default=None, help="total target; default: per-class")
    p.add_argument(
        "--class-target", action="append", default=[], metavar="GROUP=N",
        help=f"override one group's target; groups: {', '.join(review.SAMPLE_GROUPS)}",
    )
    p.add_argument("--floor-per-cell", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--bundle-id", help="override the derived id (annotations key on it)")
    p.add_argument("--window-min", type=int, default=review_schema.DEFAULT_WINDOW_MIN)
    p.add_argument("--frame-pad-min", type=int, default=review_schema.DEFAULT_FRAME_PAD_MIN)
    p.add_argument("--include-doppler", action="store_true", help="also plan :x5 frames")
    p.add_argument("--allow-feature-gap", action="store_true")
    p.add_argument("--no-frames", action="store_true", help="documents only, no imagery")
    p.add_argument("--fixture", action="store_true", help="synthetic bundle, no corpus")
    p.add_argument(
        "--fixture-corpus",
        help="where --fixture writes its synthetic archive (default: a temp dir). "
             "It is an INPUT, so it never goes inside the bundle.",
    )
    args = p.parse_args(argv)
    if args.window_to is None:
        args.window_to = datetime.now(timezone.utc)
    if not args.fixture and not args.decisions_dir:
        args.decisions_dir = ["stations/replay/decisions"]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build(args)
    out_dir = Path(args.out_dir).expanduser()
    frames = manifest.get("frames") or {}
    print(json.dumps({
        "bundle_id": manifest["bundle_id"],
        "out_dir": str(out_dir),
        "events": manifest["events"]["count"],
        "frames": frames.get("count", 0),
        "frame_bytes": frames.get("bytes", 0),
        "caveats": [c["code"] for c in manifest.get("caveats", [])],
    }, indent=2))
    print(
        f"\nServe it:\n"
        f"  python scripts/review_server.py --bundle {out_dir}\n"
        f"  cd frontend && npm run dev   # then open http://localhost:5173/review/",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
