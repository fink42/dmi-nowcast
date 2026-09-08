#!/usr/bin/env python3
"""L1 — gauge agreement per radar product (Phase H, H-L).

The question: **which of DMI's two composites is closer to the ground?**
The cycle anchors on ``fullRange`` (:x0 minutes, 240 km) and ignores
``doppler`` (:x5, 120 km), which publishes five minutes fresher. Before
any of that freshness can be spent, we need to know whether doppler agrees
with the rain gauges as well as fullRange does — and where it does not.

The measurement, for every 10-minute gauge slot ending at T, at every
station with coordinates:

* ``fullRange`` at **T-10** — the frame at the START of the slot;
* ``doppler``   at **T-5**  — the frame in the MIDDLE of the slot;
* ``fullRange`` at **T**    — the frame at the END of the slot;

each sampled with the production "raining now" statistic: p90 (and max)
of the rain rate inside a 1 km disc, Z-R with the file's own coefficients
(:class:`dmi_nowcast_core.product_pairs.DiscSampler`, which is
``sample.sample_disc`` with the disc indices computed once). Scored against
the canonical wet rule — ``precip_past10min >= 0.1 mm`` OR
``precip_dur_past10min >= 1 min``, imported from
:mod:`dmi_nowcast_core.warning_score`, never re-implemented.

Three products are reported, plus two combinations of the two that
overlap in time (fullRange at the end of the slot and doppler in the
middle of it): ``consensus`` (wet in BOTH — the "cleaner raining now" the
plan speculates about) and ``either`` (wet in EITHER — its counterpart,
which is what a per-pixel freshest-frame anchor would produce if the
products were used raw).

Read-only. It never writes into the corpus tree the monthly routine owns;
its outputs go where ``--out-*`` says.

Outputs:
  ``--out-parquet``  one row per (station, slot) with every sampled value,
                     so L2 and later analyses never touch the archive again
  ``--out-json``     every table in this report, machine-readable
  ``--out-md``       the report

Usage (container paths on the VM)::

    python scripts/gauge_agreement_study.py \\
        --corpus-dir /var/lib/dmi-nowcast-corpus \\
        --days-file /tmp/study_days.txt \\
        --workers 2 \\
        --out-parquet /var/lib/dmi-nowcast-corpus/stations/product_study/slots.parquet \\
        --out-json    /var/lib/dmi-nowcast-corpus/stations/product_study/gauge_agreement.json \\
        --out-md      /var/lib/dmi-nowcast-corpus/stations/product_study/gauge_agreement.md

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
from dmi_nowcast_core.parse import parse_composite  # noqa: E402
from dmi_nowcast_core.product_pairs import (  # noqa: E402
    DISTANCE_BANDS,
    SEASON_ORDER,
    DiscSampler,
    band_of_km,
    frame_map,
    nearest_radar_km,
    season_of,
)
from dmi_nowcast_core.warning_score import (  # noqa: E402
    SLOT_MIN,
    WET_DUR_MIN,
    WET_PRECIP_MM,
    gauge_truth_vectorised,
)

#: Slots in a day at the 10-minute gauge cadence. A day owns the slots
#: ENDING in ``(00:00, 24:00]``, i.e. 00:10 through the next midnight, so
#: every slot lies wholly inside the day and every frame it needs
#: (T-10, T-5, T) is one the day either contains or bounds.
SLOTS_PER_DAY = 24 * 60 // SLOT_MIN

#: A disc counts as covered when at least half its pixels carry data.
#: Below that the p90 is a statistic over the corner of a disc that
#: happens to be inside the product's range — doppler's edge, mostly —
#: and it is neither wet nor dry but unmeasured.
DEFAULT_MIN_VALID_FRACTION = 0.5

#: Radar-wet thresholds on the disc p90, mm/h. 0.1 is "the radar sees
#: anything", 0.5 is the cycle's own rain threshold, 1.0 is "actually
#: raining" — three points on the same curve, because which one the
#: anchor decision should be judged at is exactly what is unknown.
DEFAULT_THRESHOLDS = (0.1, 0.5, 1.0)

#: The threshold the per-station and headline tables use.
DEFAULT_PRIMARY_THRESHOLD = 0.5

#: Quantiles of the disc p90 reported conditional on gauge wet / dry.
REPORT_QUANTILES = (10, 25, 50, 75, 90, 95, 99)

#: A station is called "doppler-covered" when this share of its slots had
#: a usable doppler disc. The six NW-Jutland stations outside the 120 km
#: range should land at ~0.
DOPPLER_COVERED_FRACTION = 0.5

#: The three sampled products, ``name -> (p90 column, valid-fraction column)``.
PRODUCTS: dict[str, tuple[str, str]] = {
    "fullRange_start": ("fr_start_p90", "fr_start_valid"),
    "fullRange_end": ("fr_end_p90", "fr_end_valid"),
    "doppler_mid": ("dop_p90", "dop_valid"),
}

#: Combinations of the two products that overlap at the end of the slot.
#: ``consensus`` is the plan's "wet in both"; ``either`` is its union, the
#: shape a raw per-pixel freshest anchor would have.
COMBINATIONS = ("consensus", "either")

#: Report order for the five series.
SERIES_ORDER = tuple(PRODUCTS) + COMBINATIONS

#: Offset of each sampled frame from the slot END, in minutes.
FRAME_OFFSETS_MIN = {
    "fr_start": -SLOT_MIN,
    "dop": -SLOT_MIN // 2,
    "fr_end": 0,
}

#: Float columns the worker returns and the parquet carries, in order.
SAMPLE_COLUMNS = (
    "fr_start_p90", "fr_start_max", "fr_start_valid",
    "dop_p90", "dop_max", "dop_valid",
    "fr_end_p90", "fr_end_max", "fr_end_valid",
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def parse_day(text: str) -> date:
    return datetime.strptime(text.strip(), "%Y-%m-%d").date()


def load_days(days_csv: str | None, days_file: Path | None) -> list[date]:
    """Days from ``--days`` and/or ``--days-file`` — deduped, sorted.

    Same file format ``scripts/persistence_vs_advection.py`` and
    ``sidecar/deploy/replay.sh`` use: one ``YYYY-MM-DD`` per line, blank
    lines and ``#`` comments ignored.
    """
    raw: list[str] = []
    if days_csv:
        raw += [part for part in days_csv.split(",") if part.strip()]
    if days_file:
        raw += [
            line for line in days_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return sorted({parse_day(item) for item in raw})


def load_points(path: Path) -> list[dict[str, Any]]:
    """Station points from the ``build_station_points.py`` v2 points file."""
    payload = json.loads(Path(path).read_text())
    version = int(payload.get("version", 0))
    if version != 2:
        raise ValueError(f"{path}: points schema version {version}, expected 2")
    points: list[dict[str, Any]] = []
    for item in payload.get("points", []):
        lat, lon = float(item["lat"]), float(item["lon"])
        km = nearest_radar_km(lat, lon)
        points.append({
            "station_id": str(item["id"]),
            "lat": lat,
            "lon": lon,
            "region": str(item.get("region") or ""),
            "radar_km": km,
            "band": band_of_km(km),
        })
    points.sort(key=lambda p: p["station_id"])
    if not points:
        raise ValueError(f"{path}: no points")
    return points


def slot_ends(day: date) -> np.ndarray:
    """Epoch seconds of the day's slot ends: 00:10 through next midnight."""
    midnight = int(
        datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
    )
    step = SLOT_MIN * 60
    return midnight + (np.arange(SLOTS_PER_DAY, dtype=np.int64) + 1) * step


def load_gauge_days(
    corpus_dir: Path,
    days: Sequence[date],
    station_ids: Sequence[str],
    *,
    log=None,
) -> dict[date, dict[str, np.ndarray]]:
    """Per-day gauge truth arrays, read one month partition at a time.

    Returns ``{day: {"known": bool[station, slot], "wet": ..., "mm": ...,
    "dur": ...}}`` — arrays aligned to :func:`slot_ends`, station rows in
    the order of ``station_ids``.

    Read in the parent, not the workers: the vectorised reader works a
    month at a time, and thirty workers each re-reading their day's month
    partition would read the same 1.1M rows thirty times over. The result
    is ~270 kB per day, which pickles into a worker for free.
    """
    by_month: dict[tuple[int, int], list[date]] = {}
    for day in days:
        by_month.setdefault((day.year, day.month), []).append(day)

    out: dict[date, dict[str, np.ndarray]] = {}
    step = SLOT_MIN * 60
    for (year, month), month_days in sorted(by_month.items()):
        anchor = datetime(year, month, 1, tzinfo=timezone.utc)
        truth = gauge_truth_vectorised(
            corpus_dir, anchor, anchor, station_ids, log=log,
        )
        series = [truth.series.get(sid) for sid in station_ids]
        base = None
        for entry in series:
            if entry is not None and len(entry):
                base = int(entry.slot_end[0])
                n_grid = len(entry)
                break
        for day in month_days:
            wanted = slot_ends(day)
            known = np.zeros((len(station_ids), SLOTS_PER_DAY), dtype=bool)
            wet = np.zeros_like(known)
            mm = np.full(known.shape, np.nan, dtype=np.float32)
            dur = np.full(known.shape, np.nan, dtype=np.float32)
            if base is not None:
                idx = (wanted - base) // step
                inside = (idx >= 0) & (idx < n_grid)
                take = idx[inside]
                for row, entry in enumerate(series):
                    if entry is None or not len(entry):
                        continue
                    known[row, inside] = entry.known[take]
                    wet[row, inside] = entry.wet[take]
                    mm[row, inside] = entry.mm[take]
                    dur[row, inside] = entry.dur[take]
            out[day] = {"known": known, "wet": wet, "mm": mm, "dur": dur}
    return out


# ---------------------------------------------------------------------------
# One day, in a worker
# ---------------------------------------------------------------------------


def run_day(
    day_iso: str,
    points: list[tuple[float, float]],
    frames: dict[str, dict[datetime, Path]],
    radius_m: float,
) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    """Sample every frame of one day at every station.

    The loop is over FRAMES, not over stations: each of the day's ~289
    composites is read and parsed once, sampled at all ~118 stations, and
    dropped. Only the samples survive (four float32 per station per frame,
    ~550 kB for a whole day), so peak memory is one parsed composite —
    about 20 MB — however many stations there are.
    """
    day = parse_day(day_iso)
    sampler: DiscSampler | None = None
    samples: dict[str, dict[datetime, Any]] = {
        SCAN_TYPE_FULL_RANGE: {}, SCAN_TYPE_DOPPLER: {},
    }
    n_read = 0
    errors: list[str] = []

    for scan_type in (SCAN_TYPE_FULL_RANGE, SCAN_TYPE_DOPPLER):
        for ts in sorted(frames.get(scan_type, {})):
            path = frames[scan_type][ts]
            try:
                composite = parse_composite(Path(path))
                if sampler is None:
                    sampler = DiscSampler(composite, points, radius_m=radius_m)
                samples[scan_type][ts] = sampler.sample(composite)
                n_read += 1
            except Exception as exc:  # noqa: BLE001 — one bad frame is not fatal
                errors.append(f"{Path(path).name}: {exc}")
            finally:
                composite = None

    n_points = len(points)
    n_rows = n_points * SLOTS_PER_DAY
    columns = {name: np.full(n_rows, np.nan, dtype=np.float32)
               for name in SAMPLE_COLUMNS}
    ends = slot_ends(day)
    for slot, end_epoch in enumerate(ends):
        end = datetime.fromtimestamp(int(end_epoch), timezone.utc)
        for prefix, offset in FRAME_OFFSETS_MIN.items():
            scan_type = SCAN_TYPE_DOPPLER if prefix == "dop" else SCAN_TYPE_FULL_RANGE
            sample = samples[scan_type].get(end + timedelta(minutes=offset))
            if sample is None:
                continue
            lo = slot * n_points
            hi = lo + n_points
            columns[f"{prefix}_p90"][lo:hi] = sample.p90
            columns[f"{prefix}_max"][lo:hi] = sample.max_
            columns[f"{prefix}_valid"][lo:hi] = sample.valid_frac

    meta = {
        "day": day_iso,
        "frames_read": n_read,
        # Only the two products count: an off-grid minute in the archive
        # is a frame this study never asked for, not one it failed to read.
        "frames_expected": (
            len(frames.get(SCAN_TYPE_FULL_RANGE, {}))
            + len(frames.get(SCAN_TYPE_DOPPLER, {}))
        ),
        "errors": errors[:10],
        "n_errors": len(errors),
    }
    return day_iso, columns, meta


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _ratio(num: float, den: float) -> float | None:
    return float(num) / float(den) if den else None


def contingency(hits: int, misses: int, false_alarms: int, correct: int) -> dict:
    """POD / FAR / CSI / frequency bias from one 2x2 table."""
    return {
        "hits": int(hits),
        "misses": int(misses),
        "false_alarms": int(false_alarms),
        "correct_negatives": int(correct),
        "n": int(hits + misses + false_alarms + correct),
        "pod": _ratio(hits, hits + misses),
        "far": _ratio(false_alarms, hits + false_alarms),
        "csi": _ratio(hits, hits + misses + false_alarms),
        "bias": _ratio(hits + false_alarms, hits + misses),
        "base_rate": _ratio(hits + misses, hits + misses + false_alarms + correct),
    }


def series_masks(
    table: dict[str, np.ndarray], name: str, threshold: float, min_valid: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(scored, radar_wet)`` for one series at one threshold.

    ``scored`` marks the rows the series can be judged on at all: the
    gauge said something AND the product covered the disc. A row where
    doppler does not reach is not a doppler miss — it is not a doppler
    measurement.
    """
    known = table["gauge_known"]
    if name in PRODUCTS:
        p90_col, valid_col = PRODUCTS[name]
        covered = table[valid_col] >= min_valid
        return known & covered, covered & (table[p90_col] >= threshold)
    # Combinations need BOTH overlapping products to have covered the disc.
    fr_p90, fr_valid = PRODUCTS["fullRange_end"]
    dp_p90, dp_valid = PRODUCTS["doppler_mid"]
    covered = (table[fr_valid] >= min_valid) & (table[dp_valid] >= min_valid)
    fr_wet = covered & (table[fr_p90] >= threshold)
    dp_wet = covered & (table[dp_p90] >= threshold)
    both = fr_wet & dp_wet if name == "consensus" else fr_wet | dp_wet
    return known & covered, both


def score_groups(
    table: dict[str, np.ndarray],
    groups: list[tuple[str, np.ndarray]],
    thresholds: Sequence[float],
    min_valid: float,
) -> dict[str, dict[str, dict[str, dict]]]:
    """``{series: {threshold: {group: contingency}}}``."""
    gauge_wet = table["gauge_wet"]
    out: dict[str, dict[str, dict[str, dict]]] = {}
    for name in SERIES_ORDER:
        per_threshold: dict[str, dict[str, dict]] = {}
        for threshold in thresholds:
            scored, radar_wet = series_masks(table, name, threshold, min_valid)
            hit = scored & radar_wet & gauge_wet
            miss = scored & ~radar_wet & gauge_wet
            false_alarm = scored & radar_wet & ~gauge_wet
            correct = scored & ~radar_wet & ~gauge_wet
            per_group: dict[str, dict] = {}
            for group_name, mask in groups:
                per_group[group_name] = contingency(
                    np.count_nonzero(hit & mask),
                    np.count_nonzero(miss & mask),
                    np.count_nonzero(false_alarm & mask),
                    np.count_nonzero(correct & mask),
                )
            per_threshold[f"{threshold:g}"] = per_group
        out[name] = per_threshold
    return out


def score_stations(
    table: dict[str, np.ndarray],
    n_stations: int,
    threshold: float,
    min_valid: float,
) -> dict[str, dict[str, list]]:
    """Per-station contingency counts at one threshold, via ``bincount``."""
    gauge_wet = table["gauge_wet"]
    station = table["station_idx"]
    out: dict[str, dict[str, list]] = {}
    for name in SERIES_ORDER:
        scored, radar_wet = series_masks(table, name, threshold, min_valid)
        counts = {}
        for label, mask in (
            ("hits", scored & radar_wet & gauge_wet),
            ("misses", scored & ~radar_wet & gauge_wet),
            ("false_alarms", scored & radar_wet & ~gauge_wet),
            ("correct_negatives", scored & ~radar_wet & ~gauge_wet),
        ):
            counts[label] = np.bincount(
                station[mask], minlength=n_stations,
            ).astype(np.int64)
        out[name] = counts
    return out


def p90_distributions(
    table: dict[str, np.ndarray], min_valid: float,
) -> dict[str, dict[str, Any]]:
    """Quantiles of the disc p90 conditional on the gauge being wet / dry.

    The scores answer "does the radar fire where the gauge is wet"; this
    answers "what does the radar actually read there" — which is what a
    threshold refit would need, and what says whether doppler is low by a
    fixed offset or by a distribution.
    """
    known = table["gauge_known"]
    gauge_wet = table["gauge_wet"]
    out: dict[str, dict[str, Any]] = {}
    for name, (p90_col, valid_col) in PRODUCTS.items():
        scored = known & (table[valid_col] >= min_valid)
        entry: dict[str, Any] = {}
        for label, mask in (("wet", scored & gauge_wet), ("dry", scored & ~gauge_wet)):
            values = table[p90_col][mask]
            values = values[np.isfinite(values)]
            entry[label] = {
                "n": int(values.size),
                "mean_mm_h": float(values.mean()) if values.size else None,
                "quantiles_mm_h": {
                    str(q): float(np.percentile(values, q))
                    for q in REPORT_QUANTILES
                } if values.size else {},
            }
        out[name] = entry
    return out


def coverage_summary(
    table: dict[str, np.ndarray],
    points: list[dict[str, Any]],
    min_valid: float,
) -> dict[str, Any]:
    """Who doppler reaches, and how often both products are available."""
    n_stations = len(points)
    station = table["station_idx"]
    slots_per_station = np.bincount(station, minlength=n_stations).astype(np.int64)
    dop_ok = table[PRODUCTS["doppler_mid"][1]] >= min_valid
    fr_ok = table[PRODUCTS["fullRange_end"][1]] >= min_valid
    dop_per_station = np.bincount(
        station[dop_ok], minlength=n_stations,
    ).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(slots_per_station > 0,
                         dop_per_station / np.maximum(slots_per_station, 1), 0.0)
    covered = share >= DOPPLER_COVERED_FRACTION
    return {
        "min_valid_fraction": min_valid,
        "stations_total": n_stations,
        "stations_doppler_covered": int(np.count_nonzero(covered)),
        "stations_outside_doppler": [
            {
                "station_id": points[i]["station_id"],
                "lat": points[i]["lat"],
                "lon": points[i]["lon"],
                "radar_km": round(points[i]["radar_km"], 1),
                "doppler_slot_share": round(float(share[i]), 4),
            }
            for i in np.flatnonzero(~covered)
        ],
        "slots_total": int(table["station_idx"].size),
        "slots_gauge_known": int(np.count_nonzero(table["gauge_known"])),
        "slots_gauge_wet": int(np.count_nonzero(table["gauge_wet"])),
        "slots_fullrange_end_covered": int(np.count_nonzero(fr_ok)),
        "slots_doppler_covered": int(np.count_nonzero(dop_ok)),
        "slots_both_covered": int(np.count_nonzero(fr_ok & dop_ok)),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _score_rows(scores: dict, threshold: str, group: str) -> list[str]:
    lines = []
    for name in SERIES_ORDER:
        cell = scores[name][threshold][group]
        lines.append(
            f"| `{name}` | {cell['n']} | {_fmt(cell['base_rate'])} | "
            f"{_fmt(cell['pod'])} | {_fmt(cell['far'])} | {_fmt(cell['csi'])} | "
            f"{_fmt(cell['bias'])} |"
        )
    return lines


_SCORE_HEADER = (
    "| series | slots scored | gauge wet rate | POD | FAR | CSI | bias |\n"
    "|---|---|---|---|---|---|---|"
)


def markdown_report(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    scores = payload["scores"]
    primary = f"{meta['primary_threshold']:g}"
    thresholds = [f"{t:g}" for t in meta["thresholds"]]

    lines = [
        "# L1 — gauge agreement per radar product",
        "",
        f"Generated {meta['generated']} · "
        f"{meta['n_days']} day(s), {meta['n_stations']} station(s), "
        f"{meta['n_slots']:,} station-slots, "
        f"{meta['frames_read']:,} composite frames read.",
        "",
        "## What is measured",
        "",
        "For each 10-minute gauge slot ending at **T**, at each station, the "
        "1 km disc p90 rain rate (Z–R with the file's own coefficients, the "
        "production *raining now* statistic) is read from three frames:",
        "",
        "| series | frame | position in the slot |",
        "|---|---|---|",
        "| `fullRange_start` | fullRange at T−10 | start |",
        "| `doppler_mid` | doppler at T−5 | middle |",
        "| `fullRange_end` | fullRange at T | end |",
        "| `consensus` | both of the two above | wet in BOTH |",
        "| `either` | both of the two above | wet in EITHER |",
        "",
        f"A slot is **gauge wet** when `precip_past10min ≥ {WET_PRECIP_MM:g} mm` "
        f"OR `precip_dur_past10min ≥ {WET_DUR_MIN:g} min` (the project's shipped "
        "rule, `warning_score.py`). A series is **radar wet** when its disc p90 "
        "reaches the threshold. A slot is scored for a series only where the "
        "gauge reported AND that product covered at least "
        f"{meta['min_valid_fraction']:.0%} of the disc — outside doppler's range "
        "there is no doppler measurement to be wrong.",
        "",
        "`bias` is the frequency bias: radar-wet slots / gauge-wet slots. "
        "1.00 means the radar fires as often as the gauge is wet, whether or "
        "not on the same slots.",
        "",
        "## Coverage",
        "",
    ]
    cov = payload["coverage"]
    lines += [
        f"- stations reached by doppler in ≥ {DOPPLER_COVERED_FRACTION:.0%} of "
        f"slots: **{cov['stations_doppler_covered']} / {cov['stations_total']}**",
        f"- station-slots with a gauge reading: {cov['slots_gauge_known']:,} "
        f"({cov['slots_gauge_wet']:,} wet)",
        f"- station-slots with fullRange (end): {cov['slots_fullrange_end_covered']:,}"
        f" · with doppler: {cov['slots_doppler_covered']:,}"
        f" · with both: {cov['slots_both_covered']:,}",
        "",
    ]
    if cov["stations_outside_doppler"]:
        lines += [
            "Stations outside doppler's range:",
            "",
            "| station | lat | lon | km to nearest radar | doppler slot share |",
            "|---|---|---|---|---|",
        ]
        for row in cov["stations_outside_doppler"]:
            lines.append(
                f"| `{row['station_id']}` | {row['lat']:.3f} | {row['lon']:.3f} | "
                f"{row['radar_km']:.1f} | {row['doppler_slot_share']:.3f} |"
            )
        lines.append("")

    lines += ["## Pooled scores", ""]
    for threshold in thresholds:
        lines += [f"### radar wet at p90 ≥ {threshold} mm/h", "", _SCORE_HEADER]
        lines += _score_rows(scores, threshold, "pooled")
        lines.append("")

    lines += [f"## By season (p90 ≥ {primary} mm/h)", ""]
    for season in SEASON_ORDER:
        group = f"season:{season}"
        if scores[SERIES_ORDER[0]][primary][group]["n"] == 0:
            continue
        lines += [f"### {season}", "", _SCORE_HEADER]
        lines += _score_rows(scores, primary, group)
        lines.append("")

    lines += [
        f"## By distance to the nearest radar (p90 ≥ {primary} mm/h)",
        "",
        "The covariate the harmonisation map (L2) is stratified by: what "
        "differs between the products is beam height, and beam height is a "
        "function of range.",
        "",
    ]
    for band in DISTANCE_BANDS:
        group = f"band:{band}"
        if scores[SERIES_ORDER[0]][primary][group]["n"] == 0:
            continue
        lines += [f"### {band}", "", _SCORE_HEADER]
        lines += _score_rows(scores, primary, group)
        lines.append("")

    lines += [
        "## Disc p90 conditional on the gauge (mm/h)",
        "",
        "| product | gauge | n | mean | "
        + " | ".join(f"p{q}" for q in REPORT_QUANTILES) + " |",
        "|---|---|---|---|" + "---|" * len(REPORT_QUANTILES),
    ]
    for name, entry in payload["p90_distribution"].items():
        for label in ("wet", "dry"):
            cell = entry[label]
            quantiles = " | ".join(
                _fmt(cell["quantiles_mm_h"].get(str(q)), 3) for q in REPORT_QUANTILES
            )
            lines.append(
                f"| `{name}` | {label} | {cell['n']:,} | "
                f"{_fmt(cell['mean_mm_h'])} | {quantiles} |"
            )
    lines.append("")

    lines += [
        f"## Per station (p90 ≥ {primary} mm/h, top and bottom 10 by CSI)",
        "",
    ]
    for name in SERIES_ORDER:
        rows = payload["stations"][name]
        ranked = [r for r in rows if r["csi"] is not None]
        if not ranked:
            continue
        ranked.sort(key=lambda r: r["csi"], reverse=True)
        lines += [
            f"### `{name}`",
            "",
            "| rank | station | region | km to radar | n | POD | FAR | CSI | bias |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        head = ranked[:10]
        tail = [r for r in ranked[-10:] if r not in head]
        for label, chunk in (("top", head), ("bottom", tail)):
            for i, row in enumerate(chunk, 1):
                lines.append(
                    f"| {label} {i} | `{row['station_id']}` | {row['region']} | "
                    f"{row['radar_km']:.0f} | {row['n']} | {_fmt(row['pod'])} | "
                    f"{_fmt(row['far'])} | {_fmt(row['csi'])} | {_fmt(row['bias'])} |"
                )
        lines.append("")

    if meta.get("errors"):
        lines += ["## Frame errors", "", "```"] + meta["errors"][:20] + ["```", ""]

    lines += [
        "---",
        "",
        "Radar and gauge data: DMI Open Data, licence CC BY 4.0. "
        "Column-max reflectivity biases rain rate high (tall convective "
        "cores, virga, bright band) — every intensity here is an upper-bound "
        "proxy, which affects the products equally and so leaves the "
        "comparison intact.",
        "",
    ]
    return "\n".join(lines)


def write_parquet(
    path: Path,
    table: dict[str, np.ndarray],
    points: list[dict[str, Any]],
) -> int:
    """One row per (station, slot). Dictionary-encoded strings."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    def dictionary(values: np.ndarray, levels: Sequence[str]) -> pa.Array:
        return pa.DictionaryArray.from_arrays(
            pa.array(values.astype(np.int32)), pa.array(list(levels), pa.string()),
        )

    station_idx = table["station_idx"]
    seasons = list(SEASON_ORDER)
    regions = list(dict.fromkeys(p["region"] for p in points))
    region_of_point = np.array(
        [regions.index(p["region"]) for p in points], dtype=np.int32,
    )
    columns = {
        "station_id": dictionary(
            station_idx, [p["station_id"] for p in points]),
        "region": dictionary(region_of_point[station_idx], regions),
        "band": dictionary(table["band_idx"], DISTANCE_BANDS),
        "season": dictionary(table["season_idx"], seasons),
        "slot_end_utc": pa.array(
            (table["slot_end"] * 1_000_000).astype(np.int64),
            pa.timestamp("us", tz="UTC")),
        "radar_km": pa.array(table["radar_km"], pa.float32()),
        "gauge_known": pa.array(table["gauge_known"], pa.bool_()),
        "gauge_wet": pa.array(table["gauge_wet"], pa.bool_()),
        "gauge_mm": pa.array(table["gauge_mm"], pa.float32()),
        "gauge_dur_min": pa.array(table["gauge_dur"], pa.float32()),
    }
    for name in SAMPLE_COLUMNS:
        columns[name] = pa.array(table[name], pa.float32())
    arrow = pa.table(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(arrow, path, compression="zstd")
    return arrow.num_rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_table(
    days: Sequence[date],
    points: list[dict[str, Any]],
    per_day_samples: dict[date, dict[str, np.ndarray]],
    gauge: dict[date, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Stitch the per-day worker output into one columnar table.

    Row order is (slot-major, station-minor) within a day, days in order —
    the order the workers produce, so nothing is re-sorted.
    """
    n_points = len(points)
    seasons = list(SEASON_ORDER)
    station_idx, slot_end, season_idx = [], [], []
    known, wet, mm, dur = [], [], [], []
    samples: dict[str, list[np.ndarray]] = {name: [] for name in SAMPLE_COLUMNS}
    for day in days:
        if day not in per_day_samples:
            continue
        ends = slot_ends(day)
        station_idx.append(np.tile(np.arange(n_points, dtype=np.int32), SLOTS_PER_DAY))
        slot_end.append(np.repeat(ends, n_points))
        season_idx.append(np.full(
            n_points * SLOTS_PER_DAY,
            seasons.index(season_of(datetime(day.year, day.month, day.day,
                                             tzinfo=timezone.utc))),
            dtype=np.int32,
        ))
        truth = gauge.get(day)
        if truth is None:
            shape = (n_points, SLOTS_PER_DAY)
            truth = {
                "known": np.zeros(shape, dtype=bool),
                "wet": np.zeros(shape, dtype=bool),
                "mm": np.full(shape, np.nan, dtype=np.float32),
                "dur": np.full(shape, np.nan, dtype=np.float32),
            }
        # Gauge arrays are [station, slot]; the rows are slot-major.
        known.append(truth["known"].T.reshape(-1))
        wet.append(truth["wet"].T.reshape(-1))
        mm.append(truth["mm"].T.reshape(-1))
        dur.append(truth["dur"].T.reshape(-1))
        for name in SAMPLE_COLUMNS:
            samples[name].append(per_day_samples[day][name])

    table = {
        "station_idx": np.concatenate(station_idx),
        "slot_end": np.concatenate(slot_end),
        "season_idx": np.concatenate(season_idx),
        "gauge_known": np.concatenate(known),
        "gauge_wet": np.concatenate(wet),
        "gauge_mm": np.concatenate(mm),
        "gauge_dur": np.concatenate(dur),
    }
    for name in SAMPLE_COLUMNS:
        table[name] = np.concatenate(samples[name])
    # A slot the gauge never reported is unknown, never dry.
    table["gauge_wet"] = table["gauge_wet"] & table["gauge_known"]
    bands = np.array([DISTANCE_BANDS.index(p["band"]) for p in points], dtype=np.int32)
    kms = np.array([p["radar_km"] for p in points], dtype=np.float32)
    table["band_idx"] = bands[table["station_idx"]]
    table["radar_km"] = kms[table["station_idx"]]
    return table


def group_masks(table: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    """Pooled, per season and per distance band."""
    n = table["station_idx"].size
    groups: list[tuple[str, np.ndarray]] = [("pooled", np.ones(n, dtype=bool))]
    for i, season in enumerate(SEASON_ORDER):
        groups.append((f"season:{season}", table["season_idx"] == i))
    for i, band in enumerate(DISTANCE_BANDS):
        groups.append((f"band:{band}", table["band_idx"] == i))
    return groups


def station_rows(
    counts: dict[str, dict[str, np.ndarray]],
    points: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for name, per_label in counts.items():
        rows = []
        for i, point in enumerate(points):
            cell = contingency(
                per_label["hits"][i], per_label["misses"][i],
                per_label["false_alarms"][i], per_label["correct_negatives"][i],
            )
            rows.append({
                "station_id": point["station_id"],
                "region": point["region"],
                "radar_km": round(point["radar_km"], 1),
                **cell,
            })
        out[name] = rows
    return out


def map_days(
    days: Sequence[date],
    coords: list[tuple[float, float]],
    per_day_frames: dict[date, dict[str, dict[datetime, Path]]],
    radius_m: float,
    workers: int,
):
    """Run :func:`run_day` over every day, in parallel or serially.

    ``--workers 1`` stays in this process on purpose: it is the mode that
    can be profiled, stepped through and tested, and a pool of one buys
    nothing but a fork.
    """
    if workers <= 1:
        for day in days:
            yield run_day(
                day.isoformat(), coords, per_day_frames[day], radius_m,
            )
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_day, day.isoformat(), coords, per_day_frames[day], radius_m,
            )
            for day in days
        ]
        for future in as_completed(futures):
            yield future.result()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus-dir", type=Path,
                        default=Path("/var/lib/dmi-nowcast-corpus"),
                        help="corpus root (holds composites/ and stations/)")
    parser.add_argument("--archive-dir", type=Path, default=None,
                        help="corpus root that HOLDS composites/ "
                             "(default: --corpus-dir); not the composites "
                             "directory itself")
    parser.add_argument("--points", type=Path, default=None,
                        help="station points JSON "
                             "(default <corpus>/stations/station_points.json)")
    parser.add_argument("--days", help="comma-separated YYYY-MM-DD")
    parser.add_argument("--days-file", type=Path,
                        help="file with one YYYY-MM-DD per line")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--radius-m", type=float, default=1000.0,
                        help="disc radius, metres (production default 1000)")
    parser.add_argument("--min-valid-fraction", type=float,
                        default=DEFAULT_MIN_VALID_FRACTION,
                        help="share of a disc that must carry data to count")
    parser.add_argument("--thresholds", default=",".join(
        f"{t:g}" for t in DEFAULT_THRESHOLDS),
        help="radar-wet thresholds on the disc p90, mm/h")
    parser.add_argument("--primary-threshold", type=float,
                        default=DEFAULT_PRIMARY_THRESHOLD)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--out-parquet", type=Path)
    args = parser.parse_args(argv)

    days = load_days(args.days, args.days_file)
    if not days:
        parser.error("need --days or --days-file")
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    if args.primary_threshold not in thresholds:
        thresholds.append(args.primary_threshold)
    thresholds.sort()

    corpus_dir = args.corpus_dir
    archive_dir = args.archive_dir or corpus_dir
    points_path = args.points or (corpus_dir / "stations" / "station_points.json")
    points = load_points(points_path)
    station_ids = [p["station_id"] for p in points]
    coords = [(p["lat"], p["lon"]) for p in points]

    started = time.time()
    print(f"==> {len(days)} day(s), {len(points)} station(s)", file=sys.stderr)
    print(f"    archive {archive_dir} · points {points_path}", file=sys.stderr)

    index = ArchiveIndex(archive_dir)
    print(f"    archive index: {len(index)} frame(s) "
          f"({index.count(SCAN_TYPE_FULL_RANGE)} fullRange, "
          f"{index.count(SCAN_TYPE_DOPPLER)} doppler)", file=sys.stderr)

    print("    reading gauge truth...", file=sys.stderr)
    gauge = load_gauge_days(
        corpus_dir, days, station_ids,
        log=lambda msg: print(f"      {msg}", file=sys.stderr),
    )

    per_day_frames: dict[date, dict[str, dict[datetime, Path]]] = {}
    for day in days:
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        per_day_frames[day] = frame_map(index.list_in_window(start, end))
    del index  # the workers get paths, never the index

    per_day_samples: dict[date, dict[str, np.ndarray]] = {}
    metas: list[dict[str, Any]] = []
    for i, (day_iso, columns, meta) in enumerate(
        map_days(days, coords, per_day_frames, args.radius_m, args.workers), 1
    ):
        per_day_samples[parse_day(day_iso)] = columns
        metas.append(meta)
        print(
            f"[{i}/{len(days)}] {day_iso}: {meta['frames_read']}"
            f"/{meta['frames_expected']} frames"
            + (f", {meta['n_errors']} error(s)" if meta["n_errors"] else ""),
            file=sys.stderr, flush=True,
        )

    table = build_table(days, points, per_day_samples, gauge)
    del per_day_samples

    groups = group_masks(table)
    scores = score_groups(table, groups, thresholds, args.min_valid_fraction)
    stations = station_rows(
        score_stations(table, len(points), args.primary_threshold,
                       args.min_valid_fraction),
        points,
    )
    errors = [f"{m['day']}: {e}" for m in metas for e in m["errors"]]
    payload: dict[str, Any] = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "days": [d.isoformat() for d in days],
            "n_days": len(days),
            "n_stations": len(points),
            "n_slots": int(table["station_idx"].size),
            "frames_read": sum(m["frames_read"] for m in metas),
            "frames_expected": sum(m["frames_expected"] for m in metas),
            "n_frame_errors": sum(m["n_errors"] for m in metas),
            "errors": errors[:50],
            "thresholds": thresholds,
            "primary_threshold": args.primary_threshold,
            "min_valid_fraction": args.min_valid_fraction,
            "radius_m": args.radius_m,
            "wet_rule": {"precip_mm": WET_PRECIP_MM, "dur_min": WET_DUR_MIN},
            "slot_min": SLOT_MIN,
            "elapsed_s": round(time.time() - started, 1),
        },
        "coverage": coverage_summary(table, points, args.min_valid_fraction),
        "scores": scores,
        "p90_distribution": p90_distributions(table, args.min_valid_fraction),
        "stations": stations,
    }

    if args.out_parquet:
        n_rows = write_parquet(args.out_parquet, table, points)
        payload["meta"]["parquet_rows"] = n_rows
        print(f"    {n_rows:,} row(s) -> {args.out_parquet}", file=sys.stderr)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"    json -> {args.out_json}", file=sys.stderr)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown_report(payload))
        print(f"    markdown -> {args.out_md}", file=sys.stderr)
    if not (args.out_json or args.out_md or args.out_parquet):
        print(json.dumps(
            {"coverage": payload["coverage"],
             "pooled": {
                 name: payload["scores"][name][f"{args.primary_threshold:g}"]["pooled"]
                 for name in SERIES_ORDER
             }},
            indent=2,
        ))
    print(f"==> done in {payload['meta']['elapsed_s']} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
