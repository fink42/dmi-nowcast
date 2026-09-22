#!/usr/bin/env python3
"""Add the neighbour-gauge feature block (``ng_*``) to a replayed run.

Why this exists rather than a re-replay
---------------------------------------
The ``ng_*`` columns are computed from three things a decision row already
carries — the point, the decision instant, and the cycle's bulk motion —
plus the gauge archive, which is on disk. Nothing about them needs the
radar composite, the flow field or STEPS. Re-running the replay to get them
would cost days of CPU to recompute 4 million rows of things that are
already in the parquet; this walks the rows instead and costs minutes.

It writes a COPY of each run (``<run><suffix>/``) rather than editing it:
the run without the block is the control arm of the comparison the fit
script is about to run, and an in-place edit would destroy it.

Leave-self-out, by construction
-------------------------------
Every row is a DMI gauge, and the whole point of the block is to describe
what a point learns from the gauges around it — so the row's OWN station is
excluded from its own features, by id
(:func:`~dmi_nowcast_core.postprocess.neighbour_gauge_features`'s
``exclude_self``) and again by the half-kilometre self radius. A column
here therefore means the same thing at a training row and at a subscriber's
address, which is the only reason it is worth fitting on.

One gauge read per day
----------------------
The obs store is read once per input file, over that file's own decision
instants padded backwards far enough for ``min_since_wet`` to reach its
six-hour cap, through the same pushdown scan and the same
``slots_by_station`` bucketing the live cycle and the replay use. The slots
are then digested once into a
:class:`~dmi_nowcast_core.postprocess.GaugeSlotTable` and reused for every
one of the day's ~145 decision instants, because re-walking a hundred
stations' Python lists 145 times costs more than the features do.

Usage::

    python scripts/add_neighbour_gauge_features.py \\
        --run  /var/lib/dmi-nowcast-corpus/stations/replay_v2_base \\
        --corpus-dir /var/lib/dmi-nowcast-corpus \\
        --points /var/lib/dmi-nowcast-corpus/stations/station_points.json \\
        --out-suffix _ng --workers 8

Superseded for NEW runs (v2/S1, 2026-09-22)
-------------------------------------------
``scripts/replay_warnings.py`` now writes the block itself, in the same
pass and through the same core call, so a run produced after that date
already carries it. This script stays for the runs written before —
re-replaying a month of archive to add columns that are minutes of
arithmetic would still be the wrong trade. It therefore refuses a source
run whose ``ng_*`` columns are already filled unless ``--force``: adding
the block to a run that has it can only mean the two disagree about which
gauges or which lag, and a silent second opinion is worse than a message.

Offline and read-only apart from its own outputs. Idempotent: it refuses to
overwrite an existing copy unless ``--force``, and a forced re-run of the
same inputs produces the same bytes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "sidecar") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "sidecar"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dmi_nowcast_core import postprocess as pp  # noqa: E402
from dmi_nowcast_core.station_store import StationObsStore  # noqa: E402
from dmi_nowcast_core.warning_score import (  # noqa: E402
    PRECIP_DUR_PARAM,
    PRECIP_PARAM,
    SLOT_MIN,
)

#: The default name of the copy: ``replay_v2_base`` -> ``replay_v2_base_ng``.
DEFAULT_SUFFIX = "_ng"

#: How far apart two rows' bulk motion may be inside one decision instant
#: before it stops being "the cycle's bulk motion". It is a per-cycle
#: constant in every run written so far (checked over 5 749 instants), and
#: the features take one value per instant; this is the assertion that says
#: so out loud rather than silently averaging something that varies.
BULK_TOLERANCE = 1e-3


class BuildError(RuntimeError):
    """A refusal the caller should see as a message, not a traceback."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_station_coords(path: Path) -> dict[str, tuple[float, float]]:
    """``{station_id: (lat, lon)}`` from a v2 station points file."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or int(raw.get("version", 0)) != 2:
        raise BuildError(f"{path}: expected a version-2 station points file")
    coords = {
        str(entry["id"]): (float(entry["lat"]), float(entry["lon"]))
        for entry in raw.get("points") or ()
    }
    if not coords:
        raise BuildError(f"{path}: no points")
    return coords


def decision_files(run_dir: Path) -> list[Path]:
    """The run's decision parquets, in name order.

    ``--run`` may name the run directory or its ``decisions/`` directory,
    matching every other script in this tree.
    """
    directory = Path(run_dir)
    if directory.name == "decisions":
        directory = directory.parent
    found = sorted((directory / "decisions").glob("*.parquet"))
    if not found:
        raise BuildError(f"{run_dir}: no decisions/*.parquet")
    return found


def filled_neighbour_rows(path: Path) -> int:
    """How many ``ng_*`` values a decision file ALREADY carries.

    Metadata only — the per-row-group null counts the parquet footer
    already holds — so the whole refusal below costs one footer read per
    file rather than a decode. A column with no statistics is counted as
    filled: "I cannot tell" has to refuse, or the guard is decoration.
    """
    import pyarrow.parquet as pq

    metadata = pq.read_metadata(str(path))
    names = list(metadata.schema.to_arrow_schema().names)
    wanted = [
        names.index(name)
        for name, _definition in pp.SCALAR_FEATURE_COLUMNS_NG
        if name in names
    ]
    if not wanted:
        return 0
    filled = 0
    for group in range(metadata.num_row_groups):
        row_group = metadata.row_group(group)
        for index in wanted:
            column = row_group.column(index)
            stats = column.statistics
            nulls = None if stats is None else stats.null_count
            if nulls is None:
                filled += row_group.num_rows
            else:
                filled += max(int(row_group.num_rows) - int(nulls), 0)
    return filled


def output_dir(run_dir: Path, suffix: str) -> Path:
    """Where this run's copy goes: a sibling with ``suffix`` appended."""
    directory = Path(run_dir)
    if directory.name == "decisions":
        directory = directory.parent
    if not suffix:
        raise BuildError("--out-suffix must not be empty")
    return directory.parent / f"{directory.name}{suffix}"


# ---------------------------------------------------------------------------
# One day
# ---------------------------------------------------------------------------


def day_slot_table(
    store: Any,
    station_ids: Sequence[str],
    *,
    first_instant: datetime,
    last_instant: datetime,
    lag_min: float,
) -> pp.GaugeSlotTable:
    """The gauge slots one file's rows can see, digested once.

    The twin of ``replay_warnings.day_feature_slots``, and the same two
    calls — :meth:`StationObsStore.read_recent` for the pushdown scan, then
    the sidecar's ``slots_by_station`` to bucket the rows in one pass — but
    the window comes from the FILE's own decision instants rather than from
    the calendar day. A replayed day's last instants land after midnight
    (the anchor lag pushes ``generated_at`` past the frame), and a window
    that stopped at midnight would give those rows a gauge archive that
    ends before their visibility horizon.

    It reaches back ``GAUGE_SINCE_CAP_MIN`` plus the lag plus one slot
    behind the first instant, which is exactly far enough that
    ``ng_near_min_since_wet`` can reach its cap: a wet slot older than that
    is capped to the same number a caller who never read it would produce.
    """
    from dmi_nowcast_sidecar.gauge_history import slots_by_station

    start = first_instant - timedelta(
        minutes=pp.GAUGE_SINCE_CAP_MIN + float(lag_min) + float(SLOT_MIN),
    )
    end = last_instant - timedelta(minutes=float(lag_min))
    table = store.read_recent(
        start, end, [PRECIP_PARAM, PRECIP_DUR_PARAM], list(station_ids),
    )
    return pp.GaugeSlotTable.from_slots(
        slots_by_station(table, list(station_ids), start_utc=start, end_utc=end),
    )


def _instant_bulk(
    values: np.ndarray, rows: np.ndarray,
) -> tuple[float, int]:
    """One instant's bulk value, and 1 when the rows disagreed about it."""
    finite = values[rows]
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan"), 0
    spread = float(finite.max() - finite.min())
    if spread <= BULK_TOLERANCE:
        return float(finite[0]), 0
    return float(np.median(finite)), 1


def neighbour_columns_for_table(
    table: Any,
    slots: pp.GaugeSlotTable,
    station_coords: Mapping[str, tuple[float, float]],
    *,
    lag_min: float,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """``{column: float32 array}`` for every row of one decision table.

    Vectorised per decision instant: the rows of one instant share a motion
    and a gauge horizon, so they are one call to
    :func:`~dmi_nowcast_core.postprocess.neighbour_gauge_features` over
    ~100 points, not 100 calls over one.
    """
    stamps = (
        np.asarray(
            table.column("generated_at").combine_chunks()
            .cast("int64").to_numpy(zero_copy_only=False), dtype=np.int64,
        ) // 1_000_000
    )
    stations = np.asarray(table.column("station_id").to_pylist(), dtype=object)
    bulk_kmh = np.asarray(
        table.column("bulk_kmh").to_numpy(zero_copy_only=False), dtype=np.float64,
    )
    bulk_dir = np.asarray(
        table.column("bulk_dir_deg").to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    n = len(stamps)
    names = [name for name, _definition in pp.SCALAR_FEATURE_COLUMNS_NG]
    out = {name: np.full(n, np.nan, dtype=np.float32) for name in names}
    counts = {
        "rows": n, "instants": 0, "rows_without_coordinates": 0,
        "instants_with_varying_bulk": 0, "rows_frame_ok": 0,
    }
    if not n:
        return out, counts

    known = np.array(
        [str(sid) in station_coords for sid in stations], dtype=bool,
    )
    counts["rows_without_coordinates"] = int((~known).sum())
    instants, inverse = np.unique(stamps, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    counts["instants"] = int(instants.size)
    for index, stamp in enumerate(instants.tolist()):
        rows = np.flatnonzero((inverse == index) & known)
        if not rows.size:
            continue
        speed, varied = _instant_bulk(bulk_kmh, rows)
        bearing, _varied_dir = _instant_bulk(bulk_dir, rows)
        counts["instants_with_varying_bulk"] += int(bool(varied))
        ids = [str(sid) for sid in stations[rows]]
        points = [station_coords[sid] for sid in ids]
        block = pp.neighbour_gauge_features(
            points, slots, station_coords,
            now_utc=datetime.fromtimestamp(int(stamp), tz=timezone.utc),
            bulk_kmh=speed, bulk_dir_deg=bearing,
            lag_min=lag_min, exclude_self=ids,
        )
        for name, values in block.items():
            out[name][rows] = values
    counts["rows_frame_ok"] = int(np.nansum(out["ng_frame_ok"]))
    return out, counts


def process_file(
    path: str,
    target: str,
    corpus_dir: str,
    station_coords: Mapping[str, tuple[float, float]],
    lag_min: float,
) -> dict[str, Any]:
    """Copy one decision parquet with the ``ng_*`` columns added.

    A module-level function taking only picklable arguments, because it is
    what the worker pool calls.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.time()
    table = pq.read_table(path)
    required = {"generated_at", "station_id", "bulk_kmh", "bulk_dir_deg"}
    missing = sorted(required - set(table.schema.names))
    if missing:
        raise BuildError(f"{path}: no {', '.join(missing)} column(s)")
    stamps = (
        np.asarray(
            table.column("generated_at").combine_chunks()
            .cast("int64").to_numpy(zero_copy_only=False), dtype=np.int64,
        ) // 1_000_000
    )
    wanted = sorted({
        str(sid) for sid in table.column("station_id").to_pylist()
        if str(sid) in station_coords
    })
    if table.num_rows and wanted:
        slots = day_slot_table(
            StationObsStore(Path(corpus_dir)), wanted,
            first_instant=datetime.fromtimestamp(
                int(stamps.min()), tz=timezone.utc,
            ),
            last_instant=datetime.fromtimestamp(
                int(stamps.max()), tz=timezone.utc,
            ),
            lag_min=lag_min,
        )
    else:
        slots = pp.GaugeSlotTable.from_slots({})
    columns, counts = neighbour_columns_for_table(
        table, slots, station_coords, lag_min=lag_min,
    )
    dropped: list[str] = []
    for name, values in columns.items():
        if name in table.schema.names:
            # Replaced, not appended beside: two columns of one name is not
            # a schema a reader can align. Reported rather than silent —
            # under --force this is exactly the destructive half of the job.
            dropped.append(name)
            table = table.drop_columns([name])
        table = table.append_column(name, pa.array(values, type=pa.float32()))
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target, compression="zstd")
    counts["file"] = Path(path).name
    counts["seconds"] = round(time.time() - started, 2)
    counts["gauges_read"] = len(wanted)
    counts["columns_replaced"] = len(dropped)
    return counts


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def neighbour_settings_block(lag_min: float) -> dict[str, Any]:
    """What went in the copy's ``summary.json``, under ``run.features``.

    Every constant the block's meaning depends on, so a reader of the copy
    can tell which corridor and which lag the columns in front of them were
    computed under without going back to the source.
    """
    return {
        "enabled": True,
        "columns": [name for name, _definition in pp.SCALAR_FEATURE_COLUMNS_NG],
        "radius_km": pp.NG_RADIUS_KM,
        "cross_km": pp.NG_CROSS_KM,
        "cross_weight_km": pp.NG_CROSS_WEIGHT_KM,
        "tau_edges_min": list(pp.NG_TAU_EDGES_MIN),
        "vicinity_km": pp.NG_VICINITY_KM,
        "self_km": pp.NG_SELF_KM,
        "min_speed_kmh": pp.NG_MIN_SPEED_KMH,
        "wet_window_min": pp.NG_WET_WINDOW_MIN,
        "near_window_min": pp.NG_NEAR_WINDOW_MIN,
        "gauge_lag_min": float(lag_min),
        "reference_lat_deg": pp.NG_REF_LAT_DEG,
        "geometry": (
            "equirectangular km, linearised at "
            f"{pp.NG_REF_LAT_DEG:.0f} N"
        ),
        "leave_self_out": (
            "the row's own station id is excluded, and so is any gauge "
            f"within {pp.NG_SELF_KM} km of the point"
        ),
    }


def copy_summary(run_dir: Path, out_dir: Path, lag_min: float) -> bool:
    """The run's ``summary.json`` with a ``features.neighbour`` block added."""
    directory = Path(run_dir)
    if directory.name == "decisions":
        directory = directory.parent
    source = directory / "summary.json"
    if not source.is_file():
        return False
    payload = json.loads(source.read_text())
    run = payload.setdefault("run", {})
    features = run.setdefault("features", {})
    features["neighbour"] = neighbour_settings_block(lag_min)
    (Path(out_dir) / "summary.json").write_text(
        json.dumps(payload, indent=1, default=str) + "\n",
    )
    return True


def plan(
    run_dirs: Sequence[Path], suffix: str, *, force: bool,
) -> list[tuple[Path, Path, list[tuple[Path, Path]]]]:
    """``[(run, out, [(source, target), ...]), ...]``, refusing what it must.

    Every refusal happens here, before a single row is read: the job is
    minutes of work per run and a copy that lands on top of a previous one
    halfway through is worse than one that never started.
    """
    out: list[tuple[Path, Path, list[tuple[Path, Path]]]] = []
    for run_dir in run_dirs:
        files = decision_files(run_dir)
        target_dir = output_dir(run_dir, suffix)
        existing = sorted((target_dir / "decisions").glob("*.parquet"))
        if existing and not force:
            raise BuildError(
                f"{target_dir} already holds {len(existing)} decision file(s); "
                "pass --force to overwrite it"
            )
        if not force:
            # A run written by replay_warnings since 2026-09-22 has the
            # block already. Recomputing it would replace columns with a
            # second opinion about the same rows, so say so instead.
            for path in files:
                filled = filled_neighbour_rows(path)
                if filled:
                    raise BuildError(
                        f"{path} already carries {filled} non-null ng_* "
                        "value(s) — the replay writes the block itself "
                        "since 2026-09-22. Pass --force to recompute and "
                        "replace them."
                    )
        out.append((
            Path(run_dir), target_dir,
            [(path, target_dir / "decisions" / path.name) for path in files],
        ))
    return out


def build(
    run_dirs: Sequence[Path],
    *,
    corpus_dir: Path,
    station_coords: Mapping[str, tuple[float, float]],
    suffix: str,
    lag_min: float,
    workers: int,
    force: bool,
    log,
) -> dict[str, Any]:
    jobs = plan(run_dirs, suffix, force=force)
    totals = {
        "runs": 0, "files": 0, "rows": 0, "instants": 0,
        "rows_without_coordinates": 0, "instants_with_varying_bulk": 0,
        "rows_frame_ok": 0, "columns_replaced": 0,
    }
    outputs: list[str] = []
    for run_dir, target_dir, pairs in jobs:
        (target_dir / "decisions").mkdir(parents=True, exist_ok=True)
        log(f"{run_dir} -> {target_dir}: {len(pairs)} file(s)")
        results: list[dict[str, Any]] = []
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        process_file, str(source), str(target), str(corpus_dir),
                        dict(station_coords), float(lag_min),
                    ): source
                    for source, target in pairs
                }
                for future in as_completed(futures):
                    results.append(future.result())
                    if len(results) % 20 == 0:
                        log(f"  {len(results)}/{len(pairs)} done")
        else:
            for source, target in pairs:
                results.append(process_file(
                    str(source), str(target), str(corpus_dir),
                    dict(station_coords), float(lag_min),
                ))
        for entry in results:
            totals["files"] += 1
            for key in (
                "rows", "instants", "rows_without_coordinates",
                "instants_with_varying_bulk", "rows_frame_ok",
                "columns_replaced",
            ):
                totals[key] += int(entry.get(key, 0))
        copied = copy_summary(run_dir, target_dir, lag_min)
        replaced = sum(int(e.get("columns_replaced", 0)) for e in results)
        log(
            f"  {len(results)} file(s), "
            f"{sum(int(e['rows']) for e in results)} row(s)"
            + (
                f" — replaced {replaced} pre-existing ng_* column(s)"
                if replaced else ""
            )
            + ("" if copied else " — no summary.json to copy")
        )
        totals["runs"] += 1
        outputs.append(str(target_dir))
    totals["outputs"] = outputs
    return totals


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Copy a replay run with the neighbour-gauge feature block added."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run", type=Path, nargs="+", required=True,
                   action="extend", dest="run_dirs",
                   help="replay run directory (or its decisions/ directory); "
                        "repeatable, each gets its own copy")
    p.add_argument("--corpus-dir", type=Path, required=True,
                   help="gauge store root (the directory holding stations/)")
    p.add_argument("--points", type=Path, required=True,
                   help="v2 station points file — the gauge catalogue AND "
                        "the coordinates the geometry is computed from")
    p.add_argument("--out-suffix", default=DEFAULT_SUFFIX,
                   help="appended to each run's directory name for the copy")
    p.add_argument("--gauge-lag-min", type=float,
                   default=pp.DEFAULT_GAUGE_LAG_MIN,
                   help="availability lag: a slot is visible only once it "
                        "ended this many minutes before the decision instant. "
                        "Must match the run's own g_* lag, or the two gauge "
                        "blocks are reading different archives.")
    p.add_argument("--workers", type=int, default=1,
                   help="worker processes; one file per worker at a time")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing copy")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        coords = load_station_coords(Path(args.points))
        totals = build(
            [Path(d) for d in args.run_dirs],
            corpus_dir=Path(args.corpus_dir),
            station_coords=coords,
            suffix=str(args.out_suffix),
            lag_min=float(args.gauge_lag_min),
            workers=max(int(args.workers), 1),
            force=bool(args.force),
            log=log,
        )
    except (BuildError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    totals["seconds"] = round(time.time() - started, 1)
    totals["stations"] = len(coords)
    totals["gauge_lag_min"] = float(args.gauge_lag_min)
    log(f"done in {totals['seconds']}s")
    print(json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
