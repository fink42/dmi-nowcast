"""Bulk-backfill the persistent corpus from DMI's 180-day archive.

One-time backfill of the persistent corpus archive.
Walks the DMI radar composite collection backwards from now over the
requested window, downloads any frame not already in the corpus, and
archives it into ``<corpus>/composites/YYYY/MM/``.

Resumable: every successful archive is idempotent, so re-running picks up
where a previous run left off. Designed to be run once on the sidecar
host (where ``/var/lib/dmi-nowcast-corpus`` is bind-mounted into the
container) but works locally too — pass an explicit ``--corpus-dir``.

Typical use::

    python scripts/backfill_corpus.py \\
        --corpus-dir /var/lib/dmi-nowcast-corpus \\
        --days-back 180

DMI rate-limits at 500 req / 5 s; the underlying ``AsyncDMIClient``
respects this with its own sliding-window throttle.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmi_nowcast_core.bulk_fetch import list_window  # noqa: E402
from dmi_nowcast_core.corpus import (  # noqa: E402
    CorpusArchiver,
    archive_path_for,
)
from dmi_nowcast_core.fetch import AsyncDMIClient  # noqa: E402


_LOGGER = logging.getLogger("backfill")

# DMI's STAC ``/items`` endpoint paginates on ``limit``; we ask for hour-long
# slices so a single page request stays well under that. 180 days × 24 h ≈
# 4320 list requests — at the 500 req / 5 s ceiling that's ~45 s of listing
# even before any download.
SLICE_HOURS = 6


async def _list_slices(
    client: AsyncDMIClient,
    start: datetime,
    end: datetime,
    slice_hours: int,
) -> list:
    """List every composite feature in [start, end), in slice_hours chunks."""
    out: list = []
    cursor = start
    delta = timedelta(hours=slice_hours)
    n_slices = 0
    while cursor < end:
        slice_end = min(cursor + delta, end)
        feats = await list_window(client, cursor, slice_end)
        out.extend(feats)
        n_slices += 1
        if n_slices % 20 == 0:
            _LOGGER.info(
                "list_progress",
                extra={"slice": n_slices, "found": len(out), "cursor": cursor.isoformat()},
            )
        cursor = slice_end
    # Deduplicate by filename (a feature can appear at a slice boundary if
    # the API treats both bounds as inclusive).
    seen = set()
    unique = []
    for f in out:
        if f.filename in seen:
            continue
        seen.add(f.filename)
        unique.append(f)
    return unique


async def run_backfill(
    corpus_dir: Path,
    days_back: int,
    *,
    concurrency: int = 4,
    slice_hours: int = SLICE_HOURS,
) -> None:
    archiver = CorpusArchiver(corpus_dir)
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days_back)
    _LOGGER.info(
        "backfill_start",
        extra={
            "corpus_dir": str(corpus_dir),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "days_back": days_back,
        },
    )

    # Use a tempdir as a transient download buffer. Files are immediately
    # archived (atomic move via copyfile + os.replace) into the corpus,
    # then removed from the buffer to keep peak disk usage tiny. This
    # also keeps the script's I/O outside the working cache so we don't
    # disturb a running sidecar's LRU eviction.
    with tempfile.TemporaryDirectory(prefix="dmi-backfill-") as buf_str:
        buffer = Path(buf_str)

        async with AsyncDMIClient() as client:
            t_list = time.perf_counter()
            features = await _list_slices(client, start, end, slice_hours)
            _LOGGER.info(
                "listing_done",
                extra={
                    "n_features": len(features),
                    "elapsed_s": round(time.perf_counter() - t_list, 1),
                },
            )

            # Filter to features that aren't already in the corpus. The
            # archive layout is filename-deterministic, so this is a stat()
            # per candidate — fast enough to do upfront.
            todo = []
            already = 0
            for f in features:
                try:
                    dst = archive_path_for(corpus_dir, f.filename)
                except ValueError:
                    # Unexpected filename — let download attempt it and surface
                    # the error there.
                    todo.append(f)
                    continue
                if dst.is_file():
                    already += 1
                else:
                    todo.append(f)
            _LOGGER.info(
                "filter_done",
                extra={"total": len(features), "already": already, "todo": len(todo)},
            )

            sem = asyncio.Semaphore(concurrency)
            done = {"n": 0, "bytes": 0, "errors": 0}
            t_start = time.perf_counter()

            async def fetch_one(feat) -> None:
                async with sem:
                    try:
                        local = await client.download(feat, buffer)
                    except Exception as exc:  # noqa: BLE001
                        done["errors"] += 1
                        _LOGGER.warning(
                            "download_failed",
                            extra={"filename": feat.filename, "error": str(exc)},
                        )
                        return
                    try:
                        res = await asyncio.to_thread(archiver.archive, local)
                        done["n"] += 1
                        done["bytes"] += local.stat().st_size if res.archived else 0
                    except Exception as exc:  # noqa: BLE001
                        done["errors"] += 1
                        _LOGGER.warning(
                            "archive_failed",
                            extra={"filename": feat.filename, "error": str(exc)},
                        )
                        return
                    # Drop from the transient buffer immediately — we already
                    # have the persistent copy.
                    try:
                        local.unlink()
                    except FileNotFoundError:
                        pass

                    if done["n"] % 100 == 0:
                        elapsed = time.perf_counter() - t_start
                        rate = done["n"] / elapsed if elapsed else 0.0
                        _LOGGER.info(
                            "progress",
                            extra={
                                "archived": done["n"],
                                "errors": done["errors"],
                                "rate_per_s": round(rate, 1),
                                "remaining": len(todo) - done["n"] - done["errors"],
                            },
                        )

            await asyncio.gather(*(fetch_one(f) for f in todo))

    _LOGGER.info(
        "backfill_done",
        extra={
            "archived": done["n"],
            "errors": done["errors"],
            "already_present": already,
            "total_seen": len(features),
            "elapsed_s": round(time.perf_counter() - t_start, 1),
            "corpus_total_files": archiver.count(),
            "corpus_total_bytes": archiver.total_bytes(),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("/var/lib/dmi-nowcast-corpus"),
        help="Root of the corpus archive (composites/YYYY/MM written below). "
        "Default: /var/lib/dmi-nowcast-corpus (the sidecar bind-mount path).",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=180,
        help="How far back to backfill (default 180 = DMI archive ceiling).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Parallel downloads (default 4; honors DMI's 500 req / 5 s limit).",
    )
    parser.add_argument(
        "--slice-hours",
        type=int,
        default=SLICE_HOURS,
        help=f"Listing slice size in hours (default {SLICE_HOURS}).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    asyncio.run(
        run_backfill(
            corpus_dir=args.corpus_dir,
            days_back=args.days_back,
            concurrency=args.concurrency,
            slice_hours=args.slice_hours,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
