"""CycleEngine — end-to-end test against real cached radar frames.

We mock only the DMI client surface (``list_latest``, ``download``) so the
test exercises the full algorithm: parse HDF5 → Z–R → Farnebäck →
advect → disc stats → state.json schema. Inputs are two real frames
from ``radar_archive/``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from dmi_nowcast_core.fetch import RadarFeature
from dmi_nowcast_sidecar.compute import CycleEngine
from dmi_nowcast_sidecar.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = REPO_ROOT / "radar_archive"

# Two real consecutive frames (10 min apart, both on 2026-03-11).
FRAME_PREV = ARCHIVE / "dk.com.202603110050.500_max.h5"
FRAME_CURR = ARCHIVE / "dk.com.202603110100.500_max.h5"


pytestmark = pytest.mark.skipif(
    not FRAME_PREV.exists() or not FRAME_CURR.exists(),
    reason="radar_archive frames not available — run from a checkout with the archive",
)


def _feature(path: Path) -> RadarFeature:
    """Build a RadarFeature that points at a local file path."""
    name = path.name
    # filename pattern: dk.com.YYYYMMDDHHMM.500_max.h5
    ts_str = name.split(".")[2]
    ts = datetime(
        int(ts_str[0:4]), int(ts_str[4:6]), int(ts_str[6:8]),
        int(ts_str[8:10]), int(ts_str[10:12]),
        tzinfo=timezone.utc,
    )
    return RadarFeature(
        feature_id=name,
        filename=name,
        datetime_utc=ts,
        download_url=f"file://{path}",
        scan_type="composite",
    )


@pytest.fixture
def engine(minimal_config: Config) -> CycleEngine:
    """Build a CycleEngine with the DMI client mocked to serve archive frames."""
    minimal_config.forecast.leads_min = [5, 10, 20, 30]
    engine = CycleEngine(minimal_config)

    # Mock list_latest to return our two frames.
    engine._client.list_latest = AsyncMock(  # type: ignore[method-assign]
        return_value=[_feature(FRAME_PREV), _feature(FRAME_CURR)],
    )

    async def _download(feature: RadarFeature, dest_dir: Path) -> Path:
        # Don't actually copy — return the source path directly. Compute
        # only reads the file, so this is fine.
        return ARCHIVE / feature.filename

    engine._client.download = AsyncMock(side_effect=_download)  # type: ignore[method-assign]
    return engine


@pytest.mark.asyncio
async def test_cycle_produces_state_with_all_required_fields(engine: CycleEngine) -> None:
    result = await engine.run_cycle()
    assert result.error is None, f"cycle errored: {result.error}"
    assert result.state is not None
    s = result.state

    # Schema completeness — every block must populate.
    assert s.schema_version == 1
    assert s.radar.latest_ts.tzinfo is not None
    assert s.radar.data_age_minutes >= 0
    assert s.home.lat == 55.33
    assert s.home.lon == 10.32
    assert s.now.raining_hysteresis_state in ("wet", "dry")

    # Per-lead matches what we configured.
    leads = [e.lead_min for e in s.forecast.per_lead]
    assert leads == [5, 10, 20, 30]
    for entry in s.forecast.per_lead:
        assert entry.rain_rate_mm_h >= 0
        assert 0 <= entry.p_rain <= 1
        assert 0 <= entry.p_calibrated <= 1

    # Diagnostics populated.
    assert s.diagnostics.cycle_ms > 0
    assert s.diagnostics.compute_ms > 0
    # fetch_ms is small here because we're mocking the network, but still positive.

    # On a sample of real radar that contains some scattered echoes,
    # Farnebäck should produce a non-zero motion field most of the time.
    # We only assert finiteness because for a dry frame motion can be zero.
    assert s.motion.speed_km_per_h >= 0


@pytest.mark.asyncio
async def test_cycle_writes_state_json_to_disk(engine: CycleEngine) -> None:
    state_path = engine.store.state_path
    assert not state_path.exists()
    result = await engine.run_cycle()
    assert result.error is None
    assert state_path.exists()
    # Reloadable.
    loaded = engine.store.load()
    assert loaded is not None
    assert loaded.schema_version == 1


@pytest.mark.asyncio
async def test_cycle_archives_each_frame_into_persistent_corpus(
    engine: CycleEngine,
) -> None:
    """Verifies Phase A1 wiring: every fetched frame ends up in the corpus
    tree so the long-term archive survives ``docker compose down -v``."""
    await engine.run_cycle()

    assert engine._corpus is not None
    archived = list(engine._corpus.iter_files())
    archived_names = sorted(p.name for _, p in archived)
    assert archived_names == sorted([FRAME_PREV.name, FRAME_CURR.name])

    # Files must be inside the year/month tree, not at the corpus root.
    for _, p in archived:
        # Path component layout: composites/YYYY/MM/file.h5
        rel = p.relative_to(engine._corpus.corpus_dir)
        parts = rel.parts
        assert parts[0] == "composites"
        assert parts[1].isdigit() and len(parts[1]) == 4   # year
        assert parts[2].isdigit() and len(parts[2]) == 2   # month


@pytest.mark.asyncio
async def test_cycle_with_corpus_disabled_runs_cleanly(minimal_config) -> None:
    """``corpus_dir=None`` is the explicit opt-out; cycle must still work."""
    minimal_config.forecast.leads_min = [5, 10]
    minimal_config.storage.corpus_dir = None
    engine = CycleEngine(minimal_config)
    engine._client.list_latest = AsyncMock(  # type: ignore[method-assign]
        return_value=[_feature(FRAME_PREV), _feature(FRAME_CURR)],
    )

    async def _download(feature: RadarFeature, dest_dir: Path) -> Path:
        return ARCHIVE / feature.filename

    engine._client.download = AsyncMock(side_effect=_download)  # type: ignore[method-assign]

    result = await engine.run_cycle()
    assert result.error is None
    assert engine._corpus is None


@pytest.mark.asyncio
async def test_cycle_evicts_working_cache_to_stay_under_max_bytes(minimal_config) -> None:
    """LRU eviction must run each cycle so the working cache stays bounded."""
    minimal_config.forecast.leads_min = [5, 10]
    minimal_config.storage.working_cache_max_bytes = 10 * 1024 * 1024  # 10 MB
    engine = CycleEngine(minimal_config)
    engine._client.list_latest = AsyncMock(  # type: ignore[method-assign]
        return_value=[_feature(FRAME_PREV), _feature(FRAME_CURR)],
    )

    # Plant 12 MB of stale files in the working cache. LRU should evict them
    # after the cycle runs, since the just-downloaded frames have newer atime.
    import os
    import time as _time

    cache_dir = engine._cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    stale_files = []
    for i in range(12):
        f = cache_dir / f"dk.com.202501010{i:03d}.500_max.h5"
        f.write_bytes(b"\0" * (1024 * 1024))  # 1 MB
        stale = _time.time() - 86400.0
        os.utime(f, (stale, stale))
        stale_files.append(f)

    async def _download(feature: RadarFeature, dest_dir: Path) -> Path:
        # Copy the fixture into the cache so its atime is "now".
        dest = dest_dir / feature.filename
        dest.write_bytes((ARCHIVE / feature.filename).read_bytes())
        return dest

    engine._client.download = AsyncMock(side_effect=_download)  # type: ignore[method-assign]

    await engine.run_cycle()

    # Stale files should be evicted; recent downloads should survive.
    survivors = list(cache_dir.glob("dk.com.*.h5"))
    assert FRAME_PREV.name in [p.name for p in survivors]
    assert FRAME_CURR.name in [p.name for p in survivors]
    remaining_bytes = sum(p.stat().st_size for p in survivors)
    assert remaining_bytes <= minimal_config.storage.working_cache_max_bytes


@pytest.mark.asyncio
async def test_rain_incoming_requires_two_consecutive_cycles(engine: CycleEngine) -> None:
    """Per plan §6.4 / §14: rain_incoming only flips True after the
    second cycle that predicts rain. A single positive cycle isn't
    enough — clutter shouldn't trigger notifications."""
    # Cycle 1: wet_predicted may or may not be True; if true, streak=1, rain_incoming=False.
    r1 = await engine.run_cycle()
    assert r1.state is not None
    # If our test frames yielded wet_predicted=False, streak is still 0 and
    # rain_incoming is False regardless. Either way, after ONE cycle
    # rain_incoming cannot be True.
    assert r1.state.forecast.rain_incoming is False


@pytest.mark.asyncio
async def test_cycle_fails_gracefully_with_one_frame(minimal_config: Config) -> None:
    """list_latest returning only 1 frame must not crash the cycle —
    it should produce a CycleResult with an error message."""
    engine = CycleEngine(minimal_config)
    engine._client.list_latest = AsyncMock(  # type: ignore[method-assign]
        return_value=[_feature(FRAME_CURR)],
    )

    async def _download(feature: RadarFeature, dest_dir: Path) -> Path:
        return ARCHIVE / feature.filename

    engine._client.download = AsyncMock(side_effect=_download)  # type: ignore[method-assign]

    result = await engine.run_cycle()
    assert result.state is None
    assert result.error is not None
    assert "not enough frames" in result.error.lower()
