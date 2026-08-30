"""Tests for the corpus manifest builder.

Uses the two DMI HDF5 fixtures (committed at ``tests/fixtures/``), renamed
into the DMI ``dk.com.YYYYMMDDhhmm.500_max.h5`` convention and placed in a
corpus tree so the builder sees them.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

pyarrow = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

# Manifest builder is a script, not a package — add scripts/ to the path.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_corpus_manifest import (  # noqa: E402  (after sys.path edit)
    build_manifest,
    compute_frame_stats,
)


FIXTURES = Path(__file__).parent / "fixtures"
FULLRANGE = FIXTURES / "composite_fullrange.h5"


@pytest.fixture
def corpus_with_two_frames(tmp_path: Path) -> Path:
    """Stage two copies of the fullrange fixture in the corpus YYYY/MM layout."""
    corpus = tmp_path / "corpus"
    base = corpus / "composites" / "2026" / "05"
    base.mkdir(parents=True)
    # Copy the fixture twice with valid DMI filenames 5 min apart.
    shutil.copy(FULLRANGE, base / "dk.com.202605210500.500_max.h5")
    shutil.copy(FULLRANGE, base / "dk.com.202605210505.500_max.h5")
    return corpus


def test_compute_frame_stats_returns_expected_schema(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    base = corpus / "composites" / "2026" / "05"
    base.mkdir(parents=True)
    frame = base / "dk.com.202605210500.500_max.h5"
    shutil.copy(FULLRANGE, frame)

    stats = compute_frame_stats(frame, corpus)

    # Fields populated and dimensionally sensible.
    assert stats.path == "composites/2026/05/dk.com.202605210500.500_max.h5"
    assert stats.size_bytes > 0
    assert stats.height > 0 and stats.width > 0
    assert 0.0 <= stats.wet_fraction <= 1.0
    assert 0.0 <= stats.heavy_fraction <= stats.wet_fraction + 1e-6
    assert 0.0 <= stats.nodata_fraction <= 1.0
    assert 0.0 <= stats.undetect_fraction <= 1.0
    # Max rain must respect the 100 mm/h cap from transform.py.
    assert 0.0 <= stats.max_rain_mm_h <= 100.0
    # Both timestamps are tz-aware UTC. (They only agree on real DMI files,
    # not on fixtures renamed into a different YYYY/MM bucket.)
    assert stats.ts_utc.tzinfo is not None
    assert stats.ts_file_utc.tzinfo is not None


def test_build_manifest_writes_one_row_per_frame(
    corpus_with_two_frames: Path,
) -> None:
    manifest_path = build_manifest(corpus_with_two_frames)
    assert manifest_path == corpus_with_two_frames / "manifest.parquet"
    table = pq.read_table(manifest_path)
    assert table.num_rows == 2
    expected_cols = {
        "ts_utc", "ts_file_utc", "path", "size_bytes", "height", "width",
        "wet_fraction", "heavy_fraction", "max_rain_mm_h", "mean_rain_mm_h",
        "nodata_fraction", "undetect_fraction",
    }
    assert set(table.column_names) == expected_cols


def test_build_manifest_is_incremental(corpus_with_two_frames: Path) -> None:
    """Second build with no new files should be a no-op on row count, and
    it should not re-parse — we verify by mutating the manifest's
    ``size_bytes`` value and confirming the rebuild-flag-off run preserves it."""
    manifest_path = build_manifest(corpus_with_two_frames)
    table = pq.read_table(manifest_path)
    # Sentinel: rewrite the manifest with a deliberately wrong size_bytes
    # so we can detect whether the second build re-parsed.
    import pyarrow as pa

    cols = {c: table.column(c).to_pylist() for c in table.column_names}
    cols["size_bytes"] = [-1 for _ in cols["size_bytes"]]
    new_table = pa.Table.from_pylist(
        [
            {k: cols[k][i] for k in cols}
            for i in range(table.num_rows)
        ],
        schema=table.schema,
    )
    pq.write_table(new_table, manifest_path)

    # Without --rebuild, the second pass should reuse our sentinel rows.
    build_manifest(corpus_with_two_frames)
    second = pq.read_table(manifest_path)
    assert set(second.column("size_bytes").to_pylist()) == {-1}


def test_build_manifest_rebuild_re_parses(corpus_with_two_frames: Path) -> None:
    manifest_path = build_manifest(corpus_with_two_frames)
    table = pq.read_table(manifest_path)
    import pyarrow as pa

    cols = {c: table.column(c).to_pylist() for c in table.column_names}
    cols["size_bytes"] = [-1 for _ in cols["size_bytes"]]
    new_table = pa.Table.from_pylist(
        [{k: cols[k][i] for k in cols} for i in range(table.num_rows)],
        schema=table.schema,
    )
    pq.write_table(new_table, manifest_path)

    build_manifest(corpus_with_two_frames, rebuild=True)
    rebuilt = pq.read_table(manifest_path)
    # Rebuild must have replaced the sentinel with the real size.
    assert all(sz > 0 for sz in rebuilt.column("size_bytes").to_pylist())


def test_build_manifest_raises_when_corpus_empty(tmp_path: Path) -> None:
    corpus = tmp_path / "empty"
    corpus.mkdir()
    with pytest.raises(SystemExit):
        build_manifest(corpus)
