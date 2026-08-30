"""Tests for the backtest harness using the existing recon-cache fixtures.

We seed the cache with the two Phase 0 fixtures (they're consecutive 5-min
frames of the same scan type? No — one is fullRange, one is doppler. So we
use only one scan-type and build a synthetic third frame to make a window of
≥ 3 frames. For real verification we run the CLI; this test just shape-checks.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dmi_nowcast_core.backtest import (
    DEFAULT_HORIZONS_MIN,
    run_backtest,
    summarize,
    write_parquet,
)
from dmi_nowcast_core.cache import CacheConfig, DiskCache
from dmi_nowcast_core.fetch import RadarFeature

FIXTURES = Path(__file__).parent / "fixtures"
FULLRANGE = FIXTURES / "composite_fullrange.h5"  # 19:40 UTC
DOPPLER = FIXTURES / "composite_doppler.h5"      # 19:45 UTC


def _feature(filename: str, ts: datetime, scan_type: str) -> RadarFeature:
    return RadarFeature(
        feature_id=filename,
        datetime_utc=ts,
        scan_type=scan_type,
        download_url=f"https://example/{filename}",
        filename=filename,
    )


@pytest.fixture
def populated_cache(tmp_path: Path) -> tuple[DiskCache, list[RadarFeature]]:
    """Stage three fullRange-named frames by copying the fullRange fixture under
    three different filenames. The contents are identical, so:
    - Mean motion is zero (phase correlation of identical fields)
    - Persistence and Lagrangian must predict identical rates
    - Actual = current → CSI per timestamp/horizon should be 1.0 on the wet pixels.
    """
    cache = DiskCache(CacheConfig(root=tmp_path, max_bytes=10**9))
    features = []
    for minute in (30, 40, 50):
        fname = f"dk.com.202605171{9:01d}{minute:02d}.500_max.h5"
        shutil.copy(FULLRANGE, cache.path(fname))
        ts = datetime(2026, 5, 17, 19, minute, tzinfo=timezone.utc)
        features.append(_feature(fname, ts, "fullRange"))
    return cache, features


def test_backtest_returns_one_row_per_method_horizon_timestamp(populated_cache):
    cache, features = populated_cache
    rows = run_backtest(
        features, cache,
        lat=55.6726, lon=12.5645,
        radius_m=1000.0,
        horizons_min=(10,),  # only short horizon — needs two frames apart
        methods=("persistence", "lagrangian_mean"),
    )
    # 3 frames → 2 prediction timestamps × 1 horizon × 2 methods, but horizon=10 with
    # 10-min cadence requires future_ts to exist → only the first prediction has it.
    # Specifically: ts[1] (19:40) predicts 19:50 (present). ts[2] (19:50) predicts 20:00 (missing).
    # So we get 1 timestamp × 1 horizon × 2 methods = 2 rows.
    assert len(rows) == 2
    methods = {r["method"] for r in rows}
    assert methods == {"persistence", "lagrangian_mean"}


def test_identical_frames_yield_perfect_predictions(populated_cache):
    cache, features = populated_cache
    rows = run_backtest(
        features, cache,
        lat=55.6726, lon=12.5645,
        horizons_min=(10,),
    )
    # With identical frames + zero motion, predicted == actual exactly.
    for r in rows:
        if r["actual_rain"] is None or r["predicted_rain"] is None:
            continue
        assert r["predicted_rain"] == r["actual_rain"]
        assert r["predicted_intensity_mm_h"] == r["actual_intensity_mm_h"]


def test_write_parquet_roundtrip(populated_cache, tmp_path):
    cache, features = populated_cache
    rows = run_backtest(features, cache, lat=55.6726, lon=12.5645, horizons_min=(10,))
    output = tmp_path / "out.parquet"
    write_parquet(rows, output)
    assert output.is_file()
    # Read back and check schema columns we care about.
    import pyarrow.parquet as pq
    table = pq.read_table(output)
    expected = {
        "timestamp_utc", "method", "horizon_minutes",
        "predicted_intensity_mm_h", "predicted_rain",
        "actual_intensity_mm_h", "actual_rain",
        "n_valid_pred", "n_valid_actual",
        "threshold_mm_h", "radius_m",
    }
    assert expected.issubset(set(table.column_names))


def test_write_parquet_handles_empty_rows(tmp_path):
    out = tmp_path / "empty.parquet"
    write_parquet([], out)
    assert out.is_file()


def test_summarize_emits_per_method_per_horizon_lines(populated_cache):
    cache, features = populated_cache
    rows = run_backtest(features, cache, lat=55.6726, lon=12.5645, horizons_min=(10,))
    text = summarize(rows)
    assert "method" in text  # header
    assert "persistence" in text or len(rows) == 0


def test_backtest_skips_when_fewer_than_two_frames(tmp_path: Path):
    cache = DiskCache(CacheConfig(root=tmp_path, max_bytes=10**9))
    shutil.copy(FULLRANGE, cache.path("only.h5"))
    feature = _feature("only.h5", datetime(2026, 5, 17, 19, 40, tzinfo=timezone.utc), "fullRange")
    rows = run_backtest([feature], cache, lat=55.6726, lon=12.5645, horizons_min=(10,))
    assert rows == []
