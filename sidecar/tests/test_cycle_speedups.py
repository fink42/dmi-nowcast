"""R3 (2026-09-27 code review): cycle speed-ups and the small fixes that
rode along.

- the private loop render reuses the cycle's advected fields (bit-identical
  to re-advecting them);
- artifact PNGs are zlib level 6, not ``optimize=True`` — lossless, so the
  decoded pixels are the same;
- ``bearing_deg_from`` is the direction the rain comes FROM (was 180° off);
- the STEPS step count follows the nominal cadence (no 9 <-> 10 flip on
  DMI's :01 s timestamps);
- a corrupt cached frame costs that frame, not the cycle, and a corrupt
  download is never archived;
- ``state.json`` is never missing between two writes and is written off
  the event loop.
"""
from __future__ import annotations

import io
import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from dmi_nowcast_core.advect import advect_field_series
from dmi_nowcast_core.fetch import RadarFeature
from dmi_nowcast_sidecar import compute as compute_mod
from dmi_nowcast_sidecar import national_artifacts as na_mod
from dmi_nowcast_sidecar import render as render_mod
from dmi_nowcast_sidecar.compute import (
    CycleEngine,
    _bearing_compass_label,
    _bearing_from_deg,
    _nominal_timestep_min,
)
from dmi_nowcast_sidecar.config import Config
from dmi_nowcast_sidecar.storage import StateStore
from tests.test_compute_ensemble import (  # noqa: F401 — fixtures + helpers
    GRID_PX,
    _make_fake_run_ensemble,
    _write_composite,
    engine,
    synthetic_paths,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_textured(path: Path, ts: datetime, shift: int) -> None:
    """A composite with real texture (blobs), shifted ``shift`` px east per
    frame, so advection has something to move."""
    _write_composite(path, ts, 30.0)
    rng = np.random.default_rng(1)
    yy, xx = np.indices((GRID_PX, GRID_PX))
    dbz = np.zeros((GRID_PX, GRID_PX), np.float32)
    for _ in range(6):
        cy, cx = rng.uniform(8, GRID_PX - 8, 2)
        dbz += rng.uniform(15, 35) * np.exp(
            -((yy - cy) ** 2 + (xx - cx - shift) ** 2) / (2 * rng.uniform(3, 8) ** 2)
        )
    raw = np.clip(np.round((dbz - (-32.0)) / 0.5), 1, 254).astype(np.uint8)
    with h5py.File(path, "r+") as h5:
        h5["dataset1/data1/data"][...] = raw


@pytest.fixture
def textured_paths(tmp_path: Path) -> list[Path]:
    newest = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=4)
    paths = []
    for i in range(3):
        p = tmp_path / f"textured_{i}.h5"
        _write_textured(p, newest - timedelta(minutes=10 * (2 - i)), shift=2 * i)
        paths.append(p)
    return paths


class _FixedNow(datetime):
    fixed = datetime(2026, 9, 8, 18, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.fixed


# ---------------------------------------------------------------------------
# Render reuse (item 2)
# ---------------------------------------------------------------------------


def _loop_engine(minimal_config: Config, leads: list[int]) -> CycleEngine:
    minimal_config.forecast.leads_min = leads
    eng = CycleEngine(minimal_config)
    eng._basemap_attempted = True  # no OSM fetch
    return eng


def _decoded_frames(out_dir: Path) -> dict[str, bytes]:
    return {
        p.name: np.asarray(Image.open(p)).tobytes()
        for p in sorted(out_dir.glob("frame_*.png"))
    }


def test_render_reuses_the_cycle_fields_bit_identically(
    minimal_config: Config, textured_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    eng = _loop_engine(minimal_config, [5, 10, 15, 20, 25, 30, 45, 60])
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble([]))
    monkeypatch.setattr(render_mod, "datetime", _FixedNow)
    captured: dict = {}
    real = render_mod.render_frames

    def spy(**kw):
        captured.update(kw)
        return real(**kw)

    monkeypatch.setattr(compute_mod, "render_frames", spy)
    eng._compute_sync(textured_paths, fetch_ms=0.0)

    fields = captured["forecast_fields"]
    assert sorted(fields) == list(render_mod.LOOP_FORECAST_LEADS_MIN)
    # The reused fields ARE the loop's own advection, element for element.
    expected = list(advect_field_series(
        captured["rain_now"], captured["vy"], captured["vx"],
        horizons_minutes=render_mod.loop_horizons_minutes(captured["frame_age_min"]),
        dt_minutes=captured["dt_min"],
    ))
    for lead, exp in zip(render_mod.LOOP_FORECAST_LEADS_MIN, expected):
        assert np.array_equal(fields[lead], exp, equal_nan=True), lead
    # And rendering without them (the fallback path) gives the same loop.
    reused_apng = (eng._frames_dir / "loop.png").read_bytes()
    reused = _decoded_frames(eng._frames_dir)
    fallback_dir = tmp_path / "fallback_frames"
    kw = {k: v for k, v in captured.items() if k != "forecast_fields"}
    kw["out_dir"] = fallback_dir
    apng, _ = real(**kw)
    assert apng == reused_apng
    assert _decoded_frames(fallback_dir) == reused


def test_render_advects_for_itself_when_the_leads_do_not_line_up(
    minimal_config: Config, textured_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = _loop_engine(minimal_config, [10, 20, 30, 60])
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble([]))
    captured: dict = {}
    monkeypatch.setattr(
        compute_mod, "render_frames",
        lambda **kw: (captured.update(kw), (b"", 0.0))[1],
    )
    eng._compute_sync(textured_paths, fetch_ms=0.0)
    assert captured["forecast_fields"] is None


def test_render_frames_falls_back_on_incomplete_fields(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        render_mod, "advect_field_series",
        lambda *a, **k: calls.append(k) or iter([np.zeros((2, 2))] * 7),
    )
    monkeypatch.setattr(render_mod, "render_loop_png", lambda frames, **k: b"x")

    class _C:
        timestamp_utc = datetime(2026, 9, 8, 17, 40, tzinfo=timezone.utc)
        reflectivity_dbz = np.zeros((2, 2), np.float32)
        zr_a, zr_b = 200.0, 1.6

    c = _C()
    render_mod.render_frames(
        composites=[c], rain_now=np.zeros((2, 2)), vy=np.zeros((2, 2)),
        vx=np.zeros((2, 2)), dt_min=10.0, frame_age_min=15.0, geo=None,
        home_lat=0, home_lon=0, radius_km=1, out_dir=Path("."),
        now_stats_subline=None, disc_motion_dy_per_min=0,
        disc_motion_dx_per_min=0, disc_motion_speed_kmh=0,
        disc_motion_bearing_from="",
        forecast_fields={0: np.zeros((2, 2)), 5: np.zeros((2, 2))},
    )
    assert len(calls) == 1
    assert calls[0]["horizons_minutes"] == [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]


# ---------------------------------------------------------------------------
# PNG encoding (item 1): lossless at any level
# ---------------------------------------------------------------------------


def _decode(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)))


def test_artifact_png_pixels_identical_to_optimize_encoding() -> None:
    rng = np.random.default_rng(0)
    levels = rng.integers(0, 256, (97, 113), dtype=np.uint8)
    rgba = rng.integers(0, 256, (61, 89, 4), dtype=np.uint8)
    for enc, arr, mode in (
        (na_mod._encode_gray_png, levels, "L"),
        (na_mod._encode_rgba_png, rgba, "RGBA"),
    ):
        buf = io.BytesIO()
        Image.fromarray(arr, mode).save(buf, format="PNG", optimize=True)
        assert np.array_equal(_decode(enc(arr)), _decode(buf.getvalue()))
        assert np.array_equal(_decode(enc(arr)), arr)
    assert na_mod.PNG_COMPRESS_LEVEL == 6


def test_overlays_written_in_parallel_match_serial_encoding() -> None:
    """The overlay PNG bytes do not depend on the pool (each is its own
    field's colour map + zlib)."""
    from dmi_nowcast_core.render import _apply_colormap

    rng = np.random.default_rng(4)
    field = rng.gamma(0.5, 4.0, (120, 140)).astype(np.float32)
    field[:10] = np.nan
    one = na_mod._encode_rgba_png(_apply_colormap(field))
    many = na_mod.run_each(lambda i: na_mod._encode_rgba_png(_apply_colormap(field)), 6)
    assert all(m == one for m in many)


# ---------------------------------------------------------------------------
# Bearing (item 5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("dy", "dx", "expected"), [
    (0.0, 1.0, 270.0),   # moving east  -> from the west
    (1.0, 0.0, 0.0),     # moving south -> from the north
    (0.0, -1.0, 90.0),   # moving west  -> from the east
    (-1.0, 0.0, 180.0),  # moving north -> from the south
    (1.0, 1.0, 315.0),   # moving south-east -> from the north-west
])
def test_bearing_from_is_where_the_rain_comes_from(dy, dx, expected) -> None:
    assert _bearing_from_deg(dy, dx) == pytest.approx(expected)


def test_bearing_degrees_agree_with_the_compass_label() -> None:
    labels = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    for k in range(64):
        a = 2 * np.pi * k / 64 + 0.01
        dy, dx = float(np.sin(a)), float(np.cos(a))
        deg = _bearing_from_deg(dy, dx)
        assert labels[int(round(deg / 45.0)) % 8] == _bearing_compass_label(dy, dx)
    assert _bearing_from_deg(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Nominal step count (item 6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("dt", "nominal"), [
    (9.983333, 10.0), (10.016667, 10.0), (10.0, 10.0), (5.0167, 5.0),
    (4.9833, 5.0), (0.1, 0.1),
])
def test_nominal_timestep(dt, nominal) -> None:
    assert _nominal_timestep_min(dt) == pytest.approx(nominal)


@pytest.mark.parametrize("second_offsets", [(0, 1, 0), (0, 0, 1), (1, 1, 1)])
def test_step_count_no_longer_flips_on_one_second_timestamps(
    engine: CycleEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    second_offsets,
) -> None:
    """DMI stamps some frames :01 s. The measured dt (9.983 / 10.017) stays
    the STEPS timestep; the COUNT is ceil(90 / 10) = 9 every time (it used
    to be 10 whenever dt came out at 9.983)."""
    calls: list[dict] = []
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble(calls))
    newest = (datetime.now(timezone.utc) - timedelta(minutes=4)).replace(second=0, microsecond=0)
    paths = []
    for i, (dbz, sec) in enumerate(zip((30.0, 30.5, 31.0), second_offsets)):
        ts = newest - timedelta(minutes=10 * (2 - i)) + timedelta(seconds=sec)
        p = tmp_path / f"jitter_{i}.h5"
        _write_composite(p, ts, dbz)
        paths.append(p)
    # parse.py reads HHMMSS, so the second survives into timestamp_utc
    engine._compute_sync(paths, fetch_ms=0.0)
    call = calls[-1]
    dt = (second_offsets[2] - second_offsets[1]) / 60.0 + 10.0
    assert call["timestep_min"] == pytest.approx(dt)
    assert call["n_timesteps"] == 9


# ---------------------------------------------------------------------------
# Corrupt frames (item 7)
# ---------------------------------------------------------------------------


def _into_cache(eng: CycleEngine, paths: list[Path]) -> list[Path]:
    eng._cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for p in paths:
        dst = eng._cache_dir / p.name
        dst.write_bytes(p.read_bytes())
        out.append(dst)
    return out


def test_corrupt_cached_frame_is_deleted_and_the_cycle_continues(
    engine: CycleEngine, synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble([]))
    good = _into_cache(engine, synthetic_paths)
    bad = engine._cache_dir / "dk.com.202609081700.500_max.h5"
    bad.write_bytes(b"\x89HDF truncated")
    state = engine._compute_sync([bad, *good], fetch_ms=0.0)
    assert state is not None
    assert not bad.exists()
    assert all(p.exists() for p in good)


def test_corrupt_frame_outside_the_cache_is_never_deleted(
    engine: CycleEngine, synthetic_paths: list[Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble([]))
    bad = tmp_path / "elsewhere.h5"
    bad.write_bytes(b"not hdf5")
    engine._compute_sync([bad, *synthetic_paths], fetch_ms=0.0)
    assert bad.exists()


def test_too_few_readable_frames_fails_the_cycle(
    engine: CycleEngine, synthetic_paths: list[Path],
) -> None:
    good = _into_cache(engine, synthetic_paths[-1:])
    bad = engine._cache_dir / "dk.com.202609081700.500_max.h5"
    bad.write_bytes(b"junk")
    with pytest.raises(RuntimeError, match="not enough readable frames"):
        engine._compute_sync([bad, *good], fetch_ms=0.0)
    assert not bad.exists()


async def test_corrupt_download_is_not_archived(
    engine: CycleEngine, synthetic_paths: list[Path],
) -> None:
    good_src = synthetic_paths[-1]

    def feature(name: str, minute: int) -> RadarFeature:
        return RadarFeature(
            feature_id=name, filename=name,
            datetime_utc=datetime(2026, 9, 8, 17, minute, tzinfo=timezone.utc),
            download_url="file://x", scan_type="fullRange",
        )

    feats = [feature("dk.com.202609081730.500_max.h5", 30),
             feature("dk.com.202609081740.500_max.h5", 40)]

    async def list_latest(*, limit, scan_type=None):
        return feats

    async def download(feat, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / feat.filename
        if feat.filename.endswith("1730.500_max.h5"):
            dest.write_bytes(b"truncated")
        else:
            dest.write_bytes(good_src.read_bytes())
        return dest

    archived: list[str] = []

    class _Corpus:
        def archive(self, path):
            archived.append(Path(path).name)

            class R:
                archived = True
            return R()

    engine._client.list_latest = list_latest  # type: ignore[method-assign]
    engine._client.download = download  # type: ignore[method-assign]
    engine._corpus = _Corpus()
    paths = await engine._fetch_latest_frames()
    assert [p.name for p in paths] == ["dk.com.202609081740.500_max.h5"]
    assert archived == ["dk.com.202609081740.500_max.h5"]
    assert not (engine._cache_dir / "dk.com.202609081730.500_max.h5").exists()


# ---------------------------------------------------------------------------
# state.json write order (item 4)
# ---------------------------------------------------------------------------


def _state(engine: CycleEngine, paths: list[Path], monkeypatch) -> object:
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble([]))
    return engine._compute_sync(paths, fetch_ms=0.0)


def test_state_json_exists_at_every_rename(
    engine: CycleEngine, synthetic_paths: list[Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os as _os

    import dmi_nowcast_sidecar.storage as storage_mod

    state = _state(engine, synthetic_paths, monkeypatch)
    store = StateStore(tmp_path / "store")
    store.write(state)
    seen: list[bool] = []
    real_replace = _os.replace

    def spying_replace(src, dst):
        seen.append(store.state_path.is_file())
        return real_replace(src, dst)

    monkeypatch.setattr(storage_mod.os, "replace", spying_replace)
    store.write(state.model_copy(update={"confidence": 0.123}))
    assert seen and all(seen)
    assert store.load().confidence == pytest.approx(0.123)
    assert json.loads(store.prev_state_path.read_text())["confidence"] == state.confidence


def test_failed_write_leaves_both_files_untouched(
    engine: CycleEngine, synthetic_paths: list[Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dmi_nowcast_sidecar.storage as storage_mod

    state = _state(engine, synthetic_paths, monkeypatch)
    store = StateStore(tmp_path / "store")
    store.write(state.model_copy(update={"confidence": 0.1}))
    store.write(state.model_copy(update={"confidence": 0.2}))
    cur, prev = store.state_path.read_bytes(), store.prev_state_path.read_bytes()

    def boom(fd):
        raise OSError("disk full")

    monkeypatch.setattr(storage_mod.os, "fsync", boom)
    with pytest.raises(OSError):
        store.write(state.model_copy(update={"confidence": 0.3}))
    assert store.state_path.read_bytes() == cur
    assert store.prev_state_path.read_bytes() == prev
    assert not list(store.data_dir.glob(".state-*"))


def test_load_falls_back_to_prev_when_state_json_is_missing(
    engine: CycleEngine, synthetic_paths: list[Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(engine, synthetic_paths, monkeypatch)
    store = StateStore(tmp_path / "store")
    store.write(state.model_copy(update={"confidence": 0.4}))
    store.write(state.model_copy(update={"confidence": 0.5}))
    store.state_path.unlink()
    assert store.load().confidence == pytest.approx(0.4)


async def test_run_cycle_writes_state_off_the_event_loop(
    engine: CycleEngine, synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compute_mod, "run_ensemble", _make_fake_run_ensemble([]))

    async def fetch():
        return synthetic_paths

    engine._fetch_latest_frames = fetch  # type: ignore[method-assign]
    loop_thread = threading.get_ident()
    writer_threads: list[int] = []
    real_write = engine._store.write

    def spy(state):
        writer_threads.append(threading.get_ident())
        real_write(state)

    engine._store.write = spy  # type: ignore[method-assign]
    result = await engine.run_cycle()
    assert result.error is None, result.error
    assert writer_threads and writer_threads[0] != loop_thread
    assert engine.store.load() is not None
