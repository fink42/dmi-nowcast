"""Build ``manifest.parquet`` over the persistent corpus archive.

Builds the corpus frame manifest. Walks
``<corpus>/composites/YYYY/MM/*.h5``, parses each frame, computes the
frame-level statistics that downstream splitting and stratification
(Phase B) need, and writes one row per frame to ``manifest.parquet``.

Columns:

- ``ts_utc``           timestamp parsed from the filename (authoritative)
- ``ts_file_utc``      timestamp parsed from the HDF5 ``/what`` group
- ``path``             relative path under ``<corpus>/``
- ``size_bytes``       file size on disk
- ``height``, ``width``  grid shape
- ``wet_fraction``     fraction of pixels with rain rate ≥ 0.1 mm/h
- ``heavy_fraction``   fraction of pixels with rain rate ≥ 5 mm/h (convective proxy)
- ``max_rain_mm_h``    max rain rate after Z–R + 100 mm/h cap
- ``mean_rain_mm_h``   mean over finite pixels (mm/h)
- ``nodata_fraction``  fraction of NaN pixels (radar reported missing)
- ``undetect_fraction`` fraction of -inf pixels (radar reported below-detection)

Notes on the convective proxy: Steiner-Houze-Yuter 1995 is the canonical
classifier but needs CAPPI plus background subtraction; we use a rain-rate
threshold (≥ 5 mm/h) on column-max reflectivity as a cheap proxy and
document the deviation alongside the manifest.

Idempotent and incremental: existing manifest rows are reused unless
``--rebuild`` is passed.

Usage::

    python scripts/build_corpus_manifest.py \\
        --corpus-dir /var/lib/dmi-nowcast-corpus
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmi_nowcast_core.corpus import (  # noqa: E402
    CorpusArchiver,
    parse_filename_ts,
)
from dmi_nowcast_core.parse import parse_composite  # noqa: E402
from dmi_nowcast_core.transform import dbz_to_rain_rate  # noqa: E402


_LOGGER = logging.getLogger("manifest")

# Thresholds for stratification keys (mm/h). The "wet" floor matches
# DB_THRESHOLD_MM_H from probabilistic.py — keep them in sync.
_WET_THRESHOLD_MM_H = 0.1
_HEAVY_THRESHOLD_MM_H = 5.0


@dataclass(frozen=True)
class FrameStats:
    ts_utc: object              # datetime — kept as object to avoid pyarrow type ambiguity
    ts_file_utc: object         # datetime
    path: str
    size_bytes: int
    height: int
    width: int
    wet_fraction: float
    heavy_fraction: float
    max_rain_mm_h: float
    mean_rain_mm_h: float
    nodata_fraction: float
    undetect_fraction: float


def compute_frame_stats(file_path: Path, corpus_dir: Path) -> FrameStats:
    """Compute one manifest row for a single archived HDF5 frame."""
    composite = parse_composite(file_path)
    dbz = composite.reflectivity_dbz
    h, w = int(dbz.shape[0]), int(dbz.shape[1])
    total = float(h * w)

    nodata_mask = np.isnan(dbz)
    undetect_mask = np.isneginf(dbz)
    finite_mask = ~(nodata_mask | undetect_mask)

    rain = dbz_to_rain_rate(dbz, zr_a=composite.zr_a, zr_b=composite.zr_b)
    # ``rain`` has NaN where dbz was NaN (nodata) and 0 where dbz was -inf.
    # For aggregate stats treat NaN as "missing" and the rest as observed.
    finite_rain = rain[finite_mask]
    if finite_rain.size:
        max_rain = float(np.nanmax(finite_rain))
        mean_rain = float(np.nanmean(finite_rain))
        wet_pixels = int(np.count_nonzero(finite_rain >= _WET_THRESHOLD_MM_H))
        heavy_pixels = int(np.count_nonzero(finite_rain >= _HEAVY_THRESHOLD_MM_H))
    else:
        max_rain = 0.0
        mean_rain = 0.0
        wet_pixels = 0
        heavy_pixels = 0

    return FrameStats(
        ts_utc=parse_filename_ts(file_path.name),
        ts_file_utc=composite.timestamp_utc,
        path=str(file_path.relative_to(corpus_dir)),
        size_bytes=int(file_path.stat().st_size),
        height=h,
        width=w,
        wet_fraction=wet_pixels / total,
        heavy_fraction=heavy_pixels / total,
        max_rain_mm_h=max_rain,
        mean_rain_mm_h=mean_rain,
        nodata_fraction=float(np.count_nonzero(nodata_mask)) / total,
        undetect_fraction=float(np.count_nonzero(undetect_mask)) / total,
    )


def _existing_rows(manifest_path: Path) -> dict[str, dict]:
    """Read the existing manifest into ``{path: row_dict}`` for resume."""
    if not manifest_path.is_file():
        return {}
    import pyarrow.parquet as pq

    table = pq.read_table(manifest_path)
    rows: dict[str, dict] = {}
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    n = table.num_rows
    for i in range(n):
        row = {name: cols[name][i] for name in cols}
        rows[row["path"]] = row
    return rows


def build_manifest(
    corpus_dir: Path,
    manifest_path: Path | None = None,
    *,
    rebuild: bool = False,
    log_every: int = 200,
) -> Path:
    """Walk the corpus, compute frame stats, write ``manifest.parquet``.

    Returns the manifest path.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus_dir = Path(corpus_dir)
    if manifest_path is None:
        manifest_path = corpus_dir / "manifest.parquet"

    archiver = CorpusArchiver(corpus_dir)
    files = [p for _, p in archiver.iter_files()]
    if not files:
        raise SystemExit(f"no composites found under {corpus_dir / 'composites'}")

    existing = {} if rebuild else _existing_rows(manifest_path)
    _LOGGER.info(
        "manifest_start",
        extra={
            "corpus_dir": str(corpus_dir),
            "n_files": len(files),
            "existing_rows": len(existing),
            "rebuild": rebuild,
        },
    )

    rows: list[dict] = []
    n_new = 0
    n_skipped = 0
    errors: list[tuple[Path, str]] = []
    t0 = time.perf_counter()
    for i, path in enumerate(files, start=1):
        rel = str(path.relative_to(corpus_dir))
        if rel in existing and not rebuild:
            rows.append(existing[rel])
            n_skipped += 1
            continue
        try:
            stats = compute_frame_stats(path, corpus_dir)
        except Exception as exc:  # noqa: BLE001
            errors.append((path, str(exc)))
            _LOGGER.warning("parse_failed", extra={"path": rel, "error": str(exc)})
            continue
        rows.append(_stats_to_dict(stats))
        n_new += 1
        if i % log_every == 0:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed if elapsed else 0.0
            _LOGGER.info(
                "progress",
                extra={
                    "processed": i,
                    "total": len(files),
                    "new": n_new,
                    "skipped": n_skipped,
                    "rate_per_s": round(rate, 1),
                },
            )

    table = pa.Table.from_pylist(rows, schema=_arrow_schema())
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: write to a tempfile in the same dir and rename.
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(manifest_path)

    _LOGGER.info(
        "manifest_done",
        extra={
            "path": str(manifest_path),
            "total_rows": len(rows),
            "new_rows": n_new,
            "skipped_rows": n_skipped,
            "errors": len(errors),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        },
    )
    return manifest_path


def _stats_to_dict(s: FrameStats) -> dict:
    return {
        "ts_utc": s.ts_utc,
        "ts_file_utc": s.ts_file_utc,
        "path": s.path,
        "size_bytes": s.size_bytes,
        "height": s.height,
        "width": s.width,
        "wet_fraction": float(s.wet_fraction),
        "heavy_fraction": float(s.heavy_fraction),
        "max_rain_mm_h": float(s.max_rain_mm_h),
        "mean_rain_mm_h": float(s.mean_rain_mm_h),
        "nodata_fraction": float(s.nodata_fraction),
        "undetect_fraction": float(s.undetect_fraction),
    }


def _arrow_schema():
    import pyarrow as pa

    return pa.schema([
        ("ts_utc", pa.timestamp("us", tz="UTC")),
        ("ts_file_utc", pa.timestamp("us", tz="UTC")),
        ("path", pa.string()),
        ("size_bytes", pa.int64()),
        ("height", pa.int32()),
        ("width", pa.int32()),
        ("wet_fraction", pa.float32()),
        ("heavy_fraction", pa.float32()),
        ("max_rain_mm_h", pa.float32()),
        ("mean_rain_mm_h", pa.float32()),
        ("nodata_fraction", pa.float32()),
        ("undetect_fraction", pa.float32()),
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("/var/lib/dmi-nowcast-corpus"),
        help="Root of the corpus archive. Default /var/lib/dmi-nowcast-corpus.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Manifest path. Default <corpus-dir>/manifest.parquet.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-parse every frame; default is to reuse cached rows by path.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    build_manifest(args.corpus_dir, args.output, rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
