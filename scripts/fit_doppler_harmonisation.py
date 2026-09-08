#!/usr/bin/env python3
"""L2 — the doppler → fullRange harmonisation map (Phase H, H-L).

L1 says how the two products compare at the gauges. This fits the map that
would let the cycle actually *use* the fresher doppler frame: an empirical
quantile mapping of doppler dBZ onto fullRange dBZ, stratified by distance
to the nearest radar and by season.

Why quantile mapping and not a constant offset: the two products differ by
0–2.5 dB where both see ≥ 15 dBZ, but by much more in the tail — doppler's
echo area at 30 dBZ is a quarter to a half of fullRange's (plan §0.3). One
number cannot fix a difference that changes sign with intensity. A quantile
map matches the whole distribution, and because it is fitted with the dry
pixels in the denominator it matches the **area at every threshold** by
construction, which is exactly the property the wet mask needs.

The fit, per (season, band):

* triples of ``fullRange(T)``, ``doppler(T+5)``, ``fullRange(T+10)``;
* restricted to the joint coverage — a pixel all three frames report;
* the two bracketing fullRange frames pooled as the target distribution,
  so the map is fitted against where the echo was *before and after* the
  doppler frame rather than against one side of five minutes of advection;
* accumulated as fixed-size 0.5 dB histograms. Pixel pairs are never
  stored — a season of them would be hundreds of gigabytes.

Evaluation is on held-out days (every other day, alternating), before and
after the map: the wet-area ratio doppler/fullRange at 0.5 mm/h (target
1.00) and the overlap CSI at 0.5 mm/h, per band and season.

Read-only over the archive. Outputs:
  ``--out-json``  ``doppler_harmonisation.json`` — the table the runtime
                  would load and :func:`product_pairs.apply_harmonisation`
                  would apply
  ``--out-md``    the report, including the before/after evaluation

Usage (container paths on the VM)::

    python scripts/fit_doppler_harmonisation.py \\
        --corpus-dir /var/lib/dmi-nowcast-corpus \\
        --days-file /tmp/study_days.txt \\
        --workers 2 \\
        --out-json /var/lib/dmi-nowcast-corpus/stations/product_study/doppler_harmonisation.json \\
        --out-md   /var/lib/dmi-nowcast-corpus/stations/product_study/doppler_harmonisation.md

Data licence: CC BY 4.0 (DMI Open Data).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmi_nowcast_core.corpus import (  # noqa: E402
    SCAN_TYPE_DOPPLER,
    SCAN_TYPE_FULL_RANGE,
    ArchiveIndex,
)
from dmi_nowcast_core.parse import RadarComposite, parse_composite  # noqa: E402
from dmi_nowcast_core.product_pairs import (  # noqa: E402
    DOPPLER_BANDS,
    DISTANCE_BANDS,
    HARM_MIN_DBZ,
    HARMONISATION_SCHEMA_VERSION,
    POOLED_KEY,
    SEASON_ORDER,
    apply_harmonisation,
    band_index_grid,
    digitize_dbz,
    fit_quantile_map,
    frame_map,
    hist_edges,
    mapped_edges,
    radar_distance_grid,
    rain_rate_to_dbz,
    require_same_grid,
    season_of,
    table_key,
)

#: Spacing of the two fullRange frames that bracket a doppler frame.
PAIR_STEP_MIN = 10

#: Where the doppler frame sits between them.
DOPPLER_OFFSET_MIN = 5

#: The threshold the evaluation is run at. 0.5 mm/h is the cycle's own
#: rain threshold, so "does the mapped wet mask match" is asked about the
#: mask the service actually draws.
DEFAULT_EVAL_MM_H = 0.5

#: A (season, band) cell needs this many doppler echo pixels (>= 7 dBZ)
#: before it gets its own table; below it, the pooled table for that band
#: is the honest answer and the cell is listed as unfitted.
DEFAULT_MIN_ECHO_PIXELS = 200_000

#: dBZ values the markdown reports the map at — 18.2 dBZ is 0.5 mm/h.
REPORT_DBZ = (10.0, 15.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0)


# ---------------------------------------------------------------------------
# Days and triples
# ---------------------------------------------------------------------------


def parse_day(text: str) -> date:
    return datetime.strptime(text.strip(), "%Y-%m-%d").date()


def load_days(days_csv: str | None, days_file: Path | None) -> list[date]:
    """Days from ``--days`` and/or ``--days-file``, deduped and sorted."""
    raw: list[str] = []
    if days_csv:
        raw += [part for part in days_csv.split(",") if part.strip()]
    if days_file:
        raw += [
            line for line in days_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return sorted({parse_day(item) for item in raw})


def split_days(days: Sequence[date]) -> tuple[list[date], list[date]]:
    """Alternating fit / hold-out split.

    Alternating rather than a block split on purpose: consecutive days
    share weather, so a trailing block would hold out one synoptic
    situation and call it a season. Alternating still shares weather
    across midnight, which the report says out loud — it is a check
    against overfitting the *map*, not an independent sample.
    """
    fit = [day for i, day in enumerate(days) if i % 2 == 0]
    holdout = [day for i, day in enumerate(days) if i % 2 == 1]
    return fit, holdout


def triples_for_day(
    frames: dict[str, dict[datetime, Path]], stride: int = 1,
) -> list[tuple[datetime, datetime, datetime]]:
    """``(T, T+5, T+10)`` timestamps where all three frames are archived."""
    full = frames.get(SCAN_TYPE_FULL_RANGE, {})
    doppler = frames.get(SCAN_TYPE_DOPPLER, {})
    step = timedelta(minutes=PAIR_STEP_MIN)
    offset = timedelta(minutes=DOPPLER_OFFSET_MIN)
    out = []
    for t0 in sorted(full):
        if (t0 + step) in full and (t0 + offset) in doppler:
            out.append((t0, t0 + offset, t0 + step))
    return out[::max(1, int(stride))]


# ---------------------------------------------------------------------------
# One day, in a worker
# ---------------------------------------------------------------------------


class _Grid:
    """Per-worker grid constants: distance to radar, band index.

    Built from the first composite the worker parses and reused for every
    frame after it — the grid never moves, and rebuilding a 3.4M-pixel
    geodesic distance field per frame would dominate the run.
    """

    def __init__(self, composite: RadarComposite) -> None:
        self.shape = composite.reflectivity_dbz.shape
        self.distance_km = radar_distance_grid(composite)
        self.band_index = band_index_grid(self.distance_km)
        self.band_masks = {
            band: (self.band_index == i)
            for i, band in enumerate(DISTANCE_BANDS)
            if band in DOPPLER_BANDS
        }


def _accumulate_histograms(
    hist: dict[str, np.ndarray],
    grid: _Grid,
    season: str,
    doppler: np.ndarray,
    full_frames: Sequence[np.ndarray],
    joint: np.ndarray,
) -> int:
    """Fold one triple's pixels into the (season, band) histograms."""
    n_bins = hist_edges().size
    counted = 0
    for band, band_mask in grid.band_masks.items():
        mask = joint & band_mask
        n = int(np.count_nonzero(mask))
        if n == 0:
            continue
        counted += n
        for key in (table_key(season, band), table_key(POOLED_KEY, band)):
            entry = hist.setdefault(
                key, np.zeros((2, n_bins), dtype=np.int64),
            )
            entry[0] += np.bincount(
                digitize_dbz(doppler[mask]), minlength=n_bins,
            ).astype(np.int64)
            for field in full_frames:
                entry[1] += np.bincount(
                    digitize_dbz(field[mask]), minlength=n_bins,
                ).astype(np.int64)
    return counted


def _wet_counts(
    doppler_wet: np.ndarray,
    full_wet: Sequence[np.ndarray],
    mask: np.ndarray,
) -> tuple[int, int, int, int]:
    """``(dop_wet, fr_wet, hits, misses+false alarms split)`` over ``mask``.

    Returns ``(n_doppler_wet, n_fullrange_wet, hits, false_alarms)`` with
    the fullRange side counted once per bracketing frame; misses follow
    from ``n_fullrange_wet - hits``, so the caller stores four integers
    instead of five.
    """
    dop = doppler_wet & mask
    n_dop = int(np.count_nonzero(dop)) * len(full_wet)
    n_full = 0
    hits = 0
    for field in full_wet:
        wet = field & mask
        n_full += int(np.count_nonzero(wet))
        hits += int(np.count_nonzero(wet & dop))
    return n_dop, n_full, hits, n_dop - hits


def run_day(
    day_iso: str,
    frames: dict[str, dict[datetime, Path]],
    stride: int,
    mode: str,
    table: dict[str, Any] | None,
    eval_mm_h: float,
) -> tuple[str, dict[str, Any]]:
    """Accumulate one day: histograms (``mode="fit"``) or evaluation counts.

    Frames are parsed once each, not once per triple: consecutive triples
    share their fullRange bracket, so a day of 144 triples costs 145
    fullRange parses and 144 doppler ones. Resident arrays are three
    parsed composites, the two per-worker grids and a handful of masks —
    roughly 100 MB, and about 290 MB of RSS once the interpreter, numpy
    and pyproj are counted.
    """
    day = parse_day(day_iso)
    season = season_of(datetime(day.year, day.month, day.day, tzinfo=timezone.utc))
    triples = triples_for_day(frames, stride)
    full_paths = frames.get(SCAN_TYPE_FULL_RANGE, {})
    doppler_paths = frames.get(SCAN_TYPE_DOPPLER, {})

    grid: _Grid | None = None
    cache: dict[datetime, RadarComposite] = {}
    hist: dict[str, np.ndarray] = {}
    evaluation: dict[str, np.ndarray] = {}
    errors: list[str] = []
    n_triples = 0
    n_pixels = 0

    def load(path: Path) -> RadarComposite:
        return parse_composite(Path(path))

    for t0, t_dop, t1 in triples:
        try:
            for ts, source in ((t0, full_paths), (t1, full_paths),
                               (t_dop, doppler_paths)):
                if ts not in cache:
                    cache[ts] = load(source[ts])
            fr0, fr1, dop = cache[t0], cache[t1], cache[t_dop]
            require_same_grid(fr0, dop)
            require_same_grid(fr0, fr1)
            if grid is None:
                grid = _Grid(fr0)

            fields = [fr0.reflectivity_dbz, fr1.reflectivity_dbz]
            dop_dbz = dop.reflectivity_dbz
            joint = ~np.isnan(dop_dbz)
            for field in fields:
                joint &= ~np.isnan(field)

            if mode == "fit":
                n_pixels += _accumulate_histograms(
                    hist, grid, season, dop_dbz, fields, joint,
                )
            else:
                threshold_dbz = rain_rate_to_dbz(
                    eval_mm_h, zr_a=dop.zr_a, zr_b=dop.zr_b,
                )
                mapped = (
                    apply_harmonisation(dop_dbz, grid.distance_km, season, table)
                    if table else dop_dbz
                )
                full_wet = [field >= threshold_dbz for field in fields]
                raw_wet = dop_dbz >= threshold_dbz
                map_wet = mapped >= threshold_dbz
                for band, band_mask in grid.band_masks.items():
                    mask = joint & band_mask
                    if not mask.any():
                        continue
                    for key in (table_key(season, band),
                                table_key(POOLED_KEY, band)):
                        cell = evaluation.setdefault(
                            key, np.zeros((2, 4), dtype=np.int64),
                        )
                        cell[0] += _wet_counts(raw_wet, full_wet, mask)
                        cell[1] += _wet_counts(map_wet, full_wet, mask)
            n_triples += 1
        except Exception as exc:  # noqa: BLE001 — one bad triple is not fatal
            errors.append(f"{t0:%Y%m%d%H%M}: {exc}")
        finally:
            # Only the frames a later triple can still need stay resident.
            for ts in [ts for ts in cache if ts < t0 + timedelta(minutes=PAIR_STEP_MIN)]:
                cache.pop(ts, None)

    result: dict[str, Any] = {
        "day": day_iso,
        "season": season,
        "n_triples": n_triples,
        "n_triples_available": len(triples),
        "n_pixels": int(n_pixels),
        "errors": errors[:10],
        "n_errors": len(errors),
        "histograms": {k: v for k, v in hist.items()},
        "evaluation": {k: v for k, v in evaluation.items()},
    }
    return day_iso, result


# ---------------------------------------------------------------------------
# Fitting and reporting
# ---------------------------------------------------------------------------


def merge_counts(
    into: dict[str, np.ndarray], extra: dict[str, np.ndarray],
) -> None:
    for key, value in extra.items():
        if key in into:
            into[key] += value
        else:
            into[key] = value.copy()


def build_tables(
    histograms: dict[str, np.ndarray], min_echo_pixels: int,
) -> tuple[dict[str, Any], list[str]]:
    """Fit one quantile map per (season, band) with enough echo behind it."""
    edges = hist_edges()
    echo_from = int(np.searchsorted(edges, HARM_MIN_DBZ))
    tables: dict[str, Any] = {}
    skipped: list[str] = []
    for key, hist in sorted(histograms.items()):
        n_echo = int(hist[0, echo_from:].sum())
        season = key.split("|", 1)[0]
        if n_echo < min_echo_pixels and season != POOLED_KEY:
            skipped.append(f"{key} ({n_echo:,} doppler echo px)")
            continue
        mapped = fit_quantile_map(hist[0], hist[1])
        tables[key] = {
            "mapped_dbz": [round(float(v), 3) for v in mapped],
            "n_doppler_px": int(hist[0].sum()),
            "n_fullrange_px": int(hist[1].sum()),
            "n_doppler_echo_px": n_echo,
            "n_fullrange_echo_px": int(hist[1, echo_from:].sum()),
        }
    return tables, skipped


def evaluation_rows(counts: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Area ratio and CSI, before and after, per (season, band)."""
    rows = []
    for key, cell in sorted(counts.items()):
        season, band = key.split("|", 1)
        row: dict[str, Any] = {"season": season, "band": band}
        for i, label in enumerate(("before", "after")):
            n_dop, n_full, hits, false_alarms = (int(v) for v in cell[i])
            misses = n_full - hits
            denominator = hits + misses + false_alarms
            row[label] = {
                "doppler_wet_px": n_dop,
                "fullrange_wet_px": n_full,
                "area_ratio": (n_dop / n_full) if n_full else None,
                "csi": (hits / denominator) if denominator else None,
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
            }
        rows.append(row)
    return rows


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def markdown_report(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    edges = np.asarray(payload["bin_edges_dbz"], dtype=np.float64)
    lines = [
        "# L2 — doppler → fullRange harmonisation map",
        "",
        f"Generated {meta['generated']} · fit on {len(meta['fit_days'])} day(s), "
        f"{meta['n_fit_triples']:,} triples, {meta['n_fit_pixels']:,} pixel "
        f"samples · evaluated on {len(meta['holdout_days'])} held-out day(s), "
        f"{meta['n_eval_triples']:,} triples.",
        "",
        "## The map",
        "",
        "Empirical quantile mapping: a doppler dBZ value is replaced by the "
        "fullRange dBZ value with the same exceedance fraction over the joint "
        "coverage, so the mapped field covers the same area as fullRange at "
        "every threshold. Fitted per distance band (beam height is what "
        "differs between a 120 km and a 240 km scan) and per season. Pixels "
        f"below {HARM_MIN_DBZ:g} dBZ are left alone — dry stays dry.",
        "",
        "Target distribution: the two fullRange frames bracketing each "
        "doppler frame, pooled. Fitted from fixed-size 0.5 dB histograms; no "
        "pixel pairs are stored.",
        "",
        "| season | band | doppler px | fullRange px | "
        + " | ".join(f"{d:g} dBZ" for d in REPORT_DBZ) + " |",
        "|---|---|---|---|" + "---|" * len(REPORT_DBZ),
    ]
    for key, table in sorted(payload["tables"].items()):
        season, band = key.split("|", 1)
        mapped = np.asarray(table["mapped_dbz"], dtype=np.float64)
        cells = []
        for value in REPORT_DBZ:
            out = float(np.interp(value, edges, mapped))
            cells.append(f"{out:.1f} ({out - value:+.1f})")
        lines.append(
            f"| {season} | {band} | {table['n_doppler_px']:,} | "
            f"{table['n_fullrange_px']:,} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "Cells are the mapped fullRange dBZ and, in brackets, the shift "
        "applied. A positive shift means doppler reads low there and the map "
        "lifts it.",
        "",
    ]
    if payload.get("unfitted"):
        lines += [
            "Cells with too little echo to fit their own table (they fall "
            f"back to the pooled table for their band, ≥ "
            f"{meta['min_echo_pixels']:,} doppler echo pixels required):",
            "",
        ] + [f"- {item}" for item in payload["unfitted"]] + [""]

    lines += [
        f"## Held-out evaluation (wet at {meta['eval_mm_h']:g} mm/h)",
        "",
        "`area ratio` is the doppler wet area over the fullRange wet area — "
        "1.00 is the target, and the map is fitted to deliver it. `CSI` is "
        "the pixel overlap of the doppler wet mask with each bracketing "
        "fullRange mask; it is limited by five minutes of advection, so it "
        "cannot reach 1.00 and the question is only whether the map moves it.",
        "",
        "| season | band | area ratio before | after | CSI before | after |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["evaluation"]:
        lines.append(
            f"| {row['season']} | {row['band']} | "
            f"{_fmt(row['before']['area_ratio'])} | "
            f"{_fmt(row['after']['area_ratio'])} | "
            f"{_fmt(row['before']['csi'])} | {_fmt(row['after']['csi'])} |"
        )
    lines.append("")

    if meta["holdout_days"] and meta["fit_days"]:
        lines += [
            "Hold-out is every other day of the input list. Adjacent days "
            "share weather, so this checks that the map is not fitted to "
            "individual frames — it is not an independent sample of "
            "Danish weather.",
            "",
        ]
    if meta.get("errors"):
        lines += ["## Frame errors", "", "```"] + meta["errors"][:20] + ["```", ""]
    lines += [
        "---",
        "",
        "Radar data: DMI Open Data, licence CC BY 4.0.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _dispatch(
    days: Sequence[date],
    per_day_frames: dict[date, dict[str, dict[datetime, Path]]],
    stride: int,
    mode: str,
    table: dict[str, Any] | None,
    eval_mm_h: float,
    workers: int,
) -> list[dict[str, Any]]:
    """Run one pass over the days, in parallel or (``workers <= 1``) serially."""
    args = [
        (day.isoformat(), per_day_frames[day], stride, mode, table, eval_mm_h)
        for day in days
    ]
    results: list[dict[str, Any]] = []

    def note(i: int, day_iso: str, result: dict[str, Any]) -> None:
        print(
            f"[{mode} {i}/{len(days)}] {day_iso}: {result['n_triples']}"
            f"/{result['n_triples_available']} triples"
            + (f", {result['n_errors']} error(s)" if result["n_errors"] else ""),
            file=sys.stderr, flush=True,
        )

    if workers <= 1:
        for i, call in enumerate(args, 1):
            day_iso, result = run_day(*call)
            results.append(result)
            note(i, day_iso, result)
        return results

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_day, *call) for call in args]
        for i, future in enumerate(as_completed(futures), 1):
            day_iso, result = future.result()
            results.append(result)
            note(i, day_iso, result)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus-dir", type=Path,
                        default=Path("/var/lib/dmi-nowcast-corpus"),
                        help="corpus root (holds composites/)")
    parser.add_argument("--archive-dir", type=Path, default=None,
                        help="corpus root that HOLDS composites/ "
                             "(default: --corpus-dir)")
    parser.add_argument("--days", help="comma-separated YYYY-MM-DD")
    parser.add_argument("--days-file", type=Path,
                        help="file with one YYYY-MM-DD per line")
    parser.add_argument("--stride", type=int, default=1,
                        help="use every Nth eligible triple within a day")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval-mm-h", type=float, default=DEFAULT_EVAL_MM_H)
    parser.add_argument("--min-echo-pixels", type=int,
                        default=DEFAULT_MIN_ECHO_PIXELS,
                        help="doppler echo pixels a (season, band) needs "
                             "before it gets its own table")
    parser.add_argument("--no-holdout", action="store_true",
                        help="fit on every day and evaluate on the same days "
                             "(says so in the report; for a short run)")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args(argv)

    days = load_days(args.days, args.days_file)
    if not days:
        parser.error("need --days or --days-file")
    fit_days, holdout_days = (days, days) if args.no_holdout else split_days(days)
    if not holdout_days:
        print("!! only one day: evaluating on the fit day itself",
              file=sys.stderr)
        holdout_days = list(fit_days)

    archive_dir = args.archive_dir or args.corpus_dir
    started = time.time()
    index = ArchiveIndex(archive_dir)
    print(f"==> archive index: {len(index)} frame(s) "
          f"({index.count(SCAN_TYPE_FULL_RANGE)} fullRange, "
          f"{index.count(SCAN_TYPE_DOPPLER)} doppler)", file=sys.stderr)
    per_day_frames: dict[date, dict[str, dict[datetime, Path]]] = {}
    for day in days:
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        per_day_frames[day] = frame_map(
            index.list_in_window(start, start + timedelta(days=1))
        )
    del index

    histograms: dict[str, np.ndarray] = {}
    errors: list[str] = []
    n_fit_triples = 0
    n_fit_pixels = 0
    for result in _dispatch(fit_days, per_day_frames, args.stride,
                            "fit", None, args.eval_mm_h, args.workers):
        merge_counts(histograms, result["histograms"])
        n_fit_triples += result["n_triples"]
        n_fit_pixels += result["n_pixels"]
        errors += [f"{result['day']}: {e}" for e in result["errors"]]

    if not histograms:
        print("FATAL: no triples fitted — check the days and the archive",
              file=sys.stderr)
        return 2

    tables, unfitted = build_tables(histograms, args.min_echo_pixels)
    payload: dict[str, Any] = {
        "schema_version": HARMONISATION_SCHEMA_VERSION,
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "DMI Open Data radar composites (CC BY 4.0)",
            "fit_days": [d.isoformat() for d in fit_days],
            "holdout_days": [d.isoformat() for d in holdout_days],
            "stride": args.stride,
            "n_fit_triples": n_fit_triples,
            "n_fit_pixels": n_fit_pixels,
            "n_eval_triples": 0,
            "eval_mm_h": args.eval_mm_h,
            "min_echo_pixels": args.min_echo_pixels,
            "min_dbz": HARM_MIN_DBZ,
            "bands": list(DOPPLER_BANDS),
            "seasons": list(SEASON_ORDER),
            "errors": errors[:50],
        },
        "bin_edges_dbz": [round(float(v), 3) for v in mapped_edges()],
        "tables": tables,
        "unfitted": unfitted,
        "evaluation": [],
    }

    evaluation: dict[str, np.ndarray] = {}
    n_eval_triples = 0
    for result in _dispatch(holdout_days, per_day_frames, args.stride,
                            "eval", payload, args.eval_mm_h, args.workers):
        merge_counts(evaluation, result["evaluation"])
        n_eval_triples += result["n_triples"]
        errors += [f"{result['day']}: {e}" for e in result["errors"]]
    payload["meta"]["n_eval_triples"] = n_eval_triples
    payload["meta"]["errors"] = errors[:50]
    payload["evaluation"] = evaluation_rows(evaluation)
    payload["meta"]["elapsed_s"] = round(time.time() - started, 1)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"    json -> {args.out_json}", file=sys.stderr)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown_report(payload))
        print(f"    markdown -> {args.out_md}", file=sys.stderr)
    if not (args.out_json or args.out_md):
        print(json.dumps(
            {"tables": sorted(payload["tables"]), "evaluation": payload["evaluation"]},
            indent=2,
        ))
    print(f"==> done in {payload['meta']['elapsed_s']} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
