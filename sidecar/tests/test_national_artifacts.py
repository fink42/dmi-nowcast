"""A2 — national artifact writer (website Phase A plan §A2).

Fully synthetic: a tiny ODIM HDF5 composite (64×64 @ 500 m, same builder
pattern as ``test_compute_ensemble.py``) provides the geometry the way
``tests/test_geo.py`` does in the core suite (``parse_composite`` →
``CompositeGeo``); ``NationalProducts`` grids are hand-built numpy arrays.

Covers:
- quantisation round-trip ≤ half a step per product, NaN→255→NaN,
- the manifest schema incl. the ``grid`` geometry block verified against
  the synthetic composite,
- overlay RGBA size + NaN transparency (render.py conventions),
- retention pruning (>24 cycles → newest 24 survive, foreign files kept),
- write atomicity (temp-then-replace; manifest written last).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image
from pyproj import CRS, Transformer

from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.national import NationalProducts
from dmi_nowcast_core.parse import parse_composite
from dmi_nowcast_sidecar import national_artifacts as na
from dmi_nowcast_sidecar.national_artifacts import (
    LATEST_MANIFEST_NAME,
    MANIFEST_SCHEMA_VERSION,
    NODATA_LEVEL,
    QUANT_SPECS,
    dequantise,
    quantise,
    write_national_artifacts,
)

HOME_LON, HOME_LAT = 10.32, 55.33
DMI_PROJ = "+proj=stere +lat_0=56 +lon_0=10.5666 +lat_ts=56 +ellps=WGS84 +units=m +no_defs"
GRID_PX = 64          # native synthetic grid — 64×64 @ 500 m
PIXEL_M = 500.0
DOWNSAMPLE = 4
PRODUCT_PX = GRID_PX // DOWNSAMPLE  # 16×16 product grid
GAIN, OFFSET = 0.5, -32.0
NODATA_RAW, UNDETECT_RAW = 255, 0

RADAR_TS = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
GENERATED_AT = datetime(2026, 8, 28, 12, 3, 21, tzinfo=timezone.utc)
STAMP = "202608281200"

_STAMP_IN_NAME = re.compile(r"_(\d{12})\.(?:png|json)$")


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _corners_lonlat() -> dict[str, tuple[float, float]]:
    """Grid corners such that home sits exactly at the grid centre."""
    crs = CRS.from_proj4(DMI_PROJ)
    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x_home, y_home = to_proj.transform(HOME_LON, HOME_LAT)
    half = GRID_PX / 2 * PIXEL_M
    return {
        "UL": to_wgs.transform(x_home - half, y_home + half),
        "UR": to_wgs.transform(x_home + half, y_home + half),
        "LL": to_wgs.transform(x_home - half, y_home - half),
        "LR": to_wgs.transform(x_home + half, y_home - half),
    }


def _write_composite(path: Path, ts: datetime) -> None:
    """Write a minimal DMI-style ODIM HDF5 composite (uniform 30 dBZ)."""
    raw_value = int(round((30.0 - OFFSET) / GAIN))
    raw = np.full((GRID_PX, GRID_PX), raw_value, dtype=np.uint8)
    corners = _corners_lonlat()
    with h5py.File(path, "w") as h5:
        what = h5.create_group("what")
        what.attrs["gain"] = GAIN
        what.attrs["offset"] = OFFSET
        what.attrs["nodata"] = NODATA_RAW
        what.attrs["undetect"] = UNDETECT_RAW
        what.attrs["date"] = ts.strftime("%Y%m%d").encode()
        what.attrs["time"] = ts.strftime("%H%M%S").encode()
        what.attrs["product"] = b"DBZH"
        where = h5.create_group("where")
        where.attrs["projdef"] = DMI_PROJ.encode()
        where.attrs["xscale"] = PIXEL_M
        where.attrs["yscale"] = PIXEL_M
        for name, (lon, lat) in corners.items():
            where.attrs[f"{name}_lon"] = lon
            where.attrs[f"{name}_lat"] = lat
        how = h5.create_group("how")
        how.attrs["zr-a"] = 200.0
        how.attrs["zr-b"] = 1.6
        h5.create_group("dataset1").create_group("data1").create_dataset(
            "data", data=raw,
        )


@pytest.fixture(scope="module")
def geo(tmp_path_factory: pytest.TempPathFactory) -> CompositeGeo:
    path = tmp_path_factory.mktemp("composite") / "synthetic_composite.h5"
    _write_composite(path, RADAR_TS)
    return CompositeGeo(parse_composite(path))


def _make_products(
    *, leads: tuple[int, ...] = (10, 20), h: int = PRODUCT_PX, w: int = PRODUCT_PX,
) -> NationalProducts:
    """Deterministic product grids spanning each quantisation range, with a
    NaN (nodata) pixel at [0, 0]."""
    ramp = np.linspace(0.0, 1.0, h * w, dtype=np.float32).reshape(h, w)

    def grid(hi: float) -> np.ndarray:
        g = (ramp * hi).astype(np.float32)
        g[0, 0] = np.nan
        return g

    return NationalProducts(
        p_rain={lead: grid(1.0) for lead in leads},
        eta_min=grid(60.0),
        intensity_mm_h=grid(100.0),
        leads_min=leads,
        threshold_mm_h=0.1,
        timestep_min=5.0,
        frame_age_min=2.5,
        downsample_factor=DOWNSAMPLE,
        n_members=8,
    )


def _overlay_field() -> np.ndarray:
    """Native 64×64 rain field: NaN block, dry area, one heavy-rain pixel,
    one below-render-floor pixel."""
    field = np.zeros((GRID_PX, GRID_PX), dtype=np.float32)
    field[:4, :4] = np.nan       # off-composite
    field[32, 32] = 10.0         # heavy rain → opaque
    field[10, 10] = 0.1          # below the 0.3 mm/h render floor → transparent
    return field


def _write(geo: CompositeGeo, out_dir: Path, **overrides):
    kwargs = dict(
        geo=geo,
        radar_ts_utc=RADAR_TS,
        generated_at_utc=GENERATED_AT,
        overlay_fields_mm_h={0: _overlay_field(), 10: _overlay_field()},
        out_dir=out_dir,
    )
    kwargs.update(overrides)
    products = kwargs.pop("products", _make_products())
    return write_national_artifacts(products, **kwargs)


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("product", sorted(QUANT_SPECS))
def test_quantise_round_trip_within_half_step(product: str) -> None:
    spec = QUANT_SPECS[product]
    values = np.linspace(spec.lo, spec.hi, 1001, dtype=np.float32).reshape(7, 143)
    levels = quantise(values, spec)
    assert levels.dtype == np.uint8
    assert NODATA_LEVEL not in levels          # all values are finite
    back = dequantise(levels, scale=spec.scale, offset=spec.offset)
    err = np.abs(back - values)
    assert float(err.max()) <= spec.scale / 2 + 1e-4, (
        f"{product}: max round-trip error {err.max()} > half step {spec.scale / 2}"
    )
    # Range endpoints hit the exact end levels.
    assert quantise(np.array([spec.lo]), spec)[0] == 0
    assert quantise(np.array([spec.hi]), spec)[0] == 254


def test_quantise_nan_round_trips_via_255() -> None:
    spec = QUANT_SPECS["p_rain"]
    values = np.array([[np.nan, 0.5], [np.inf, -np.inf]], dtype=np.float32)
    levels = quantise(values, spec)
    assert levels[0, 0] == NODATA_LEVEL
    assert levels[1, 0] == NODATA_LEVEL and levels[1, 1] == NODATA_LEVEL
    assert levels[0, 1] != NODATA_LEVEL
    back = dequantise(levels, scale=spec.scale, offset=spec.offset)
    assert np.isnan(back[0, 0]) and np.isnan(back[1, 0]) and np.isnan(back[1, 1])
    assert abs(back[0, 1] - 0.5) <= spec.scale / 2


def test_quantise_clamps_out_of_range_finite_values() -> None:
    spec = QUANT_SPECS["intensity"]
    levels = quantise(np.array([-5.0, 250.0], dtype=np.float32), spec)
    assert levels[0] == 0        # below range → lo, not nodata
    assert levels[1] == 254      # above range → hi, not nodata


# ---------------------------------------------------------------------------
# Writer end-to-end
# ---------------------------------------------------------------------------

def test_writer_emits_expected_files(geo: CompositeGeo, tmp_path: Path) -> None:
    result = _write(geo, tmp_path)
    expected = {
        f"p_rain_10min_{STAMP}.png",
        f"p_rain_20min_{STAMP}.png",
        f"eta_{STAMP}.png",
        f"intensity_{STAMP}.png",
        f"overlay_now_{STAMP}.png",
        f"overlay_10min_{STAMP}.png",
        f"manifest_{STAMP}.json",
        LATEST_MANIFEST_NAME,
    }
    on_disk = {p.name for p in tmp_path.iterdir()}
    assert on_disk == expected
    assert {p.name for p in result.files_written} == expected
    assert result.manifest_path == tmp_path / f"manifest_{STAMP}.json"
    assert result.latest_manifest_path == tmp_path / LATEST_MANIFEST_NAME
    assert result.bytes_written == sum(p.stat().st_size for p in result.files_written)
    assert result.bytes_written > 0
    assert not list(tmp_path.glob("*.tmp")), "no temp files may survive a write"


def test_manifest_schema_and_geometry(geo: CompositeGeo, tmp_path: Path) -> None:
    result = _write(geo, tmp_path)
    manifest = json.loads(result.manifest_path.read_text())
    # Stable alias is byte-identical to the stamped manifest.
    assert result.latest_manifest_path.read_bytes() == result.manifest_path.read_bytes()

    assert set(manifest) == {
        "schema_version", "cycle", "radar_ts_utc", "generated_at_utc",
        "threshold_mm_h", "timestep_min", "frame_age_min", "n_members",
        "leads_min", "grid", "overlay_grid", "calibration", "artifacts",
    }
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    # §B4: no calibration block passed → served grids are raw → null.
    assert manifest["calibration"] is None
    assert manifest["cycle"] == STAMP
    assert manifest["threshold_mm_h"] == pytest.approx(0.1)
    assert manifest["timestep_min"] == pytest.approx(5.0)
    assert manifest["frame_age_min"] == pytest.approx(2.5)
    assert manifest["n_members"] == 8
    assert manifest["leads_min"] == [10, 20]

    # Timestamps: ISO 8601, explicit UTC offset, matching the inputs.
    for key, expected_dt in (("radar_ts_utc", RADAR_TS),
                             ("generated_at_utc", GENERATED_AT)):
        raw = manifest[key]
        assert raw.endswith("+00:00"), f"{key} must carry an explicit UTC offset"
        parsed = datetime.fromisoformat(raw)
        assert parsed.utcoffset() == timedelta(0)
        assert parsed == expected_dt

    # Grid geometry block, verified against the synthetic composite the way
    # tests/test_geo.py verifies CompositeGeo: independently re-project the
    # UL corner with pyproj and compare.
    grid = manifest["grid"]
    composite = geo.composite
    assert grid["proj4"] == composite.projection
    to_proj = Transformer.from_crs(
        "EPSG:4326", CRS.from_proj4(composite.projection), always_xy=True,
    )
    ul_x, ul_y = to_proj.transform(*composite.corners_lonlat["UL"])
    assert grid["x_ul_m"] == pytest.approx(ul_x, abs=1e-6)
    assert grid["y_ul_m"] == pytest.approx(ul_y, abs=1e-6)
    assert grid["pixel_scale_x_m"] == pytest.approx(PIXEL_M * DOWNSAMPLE)
    assert grid["pixel_scale_y_m"] == pytest.approx(PIXEL_M * DOWNSAMPLE)
    assert grid["shape"] == [PRODUCT_PX, PRODUCT_PX]
    assert grid["downsample_factor"] == DOWNSAMPLE

    # Browser-side sampling with the manifest numbers must agree with
    # CompositeGeo (aggregate_at_home's native-row / factor convention).
    x_home, y_home = to_proj.transform(HOME_LON, HOME_LAT)
    manifest_col = (x_home - grid["x_ul_m"]) / grid["pixel_scale_x_m"]
    manifest_row = (grid["y_ul_m"] - y_home) / grid["pixel_scale_y_m"]
    idx = geo.lonlat_to_grid(HOME_LON, HOME_LAT)
    assert manifest_col == pytest.approx(idx.col / DOWNSAMPLE, abs=1e-6)
    assert manifest_row == pytest.approx(idx.row / DOWNSAMPLE, abs=1e-6)

    # Overlay grid: native resolution, same UL corner.
    overlay_grid = manifest["overlay_grid"]
    assert overlay_grid["proj4"] == composite.projection
    assert overlay_grid["x_ul_m"] == grid["x_ul_m"]
    assert overlay_grid["y_ul_m"] == grid["y_ul_m"]
    assert overlay_grid["pixel_scale_x_m"] == pytest.approx(PIXEL_M)
    assert overlay_grid["shape"] == [GRID_PX, GRID_PX]
    assert overlay_grid["downsample_factor"] == 1

    # Artifact entries: every documented key, per encoding.
    entries = {e["filename"]: e for e in manifest["artifacts"]}
    assert set(entries) == {p.name for p in result.files_written} - {
        f"manifest_{STAMP}.json", LATEST_MANIFEST_NAME,
    }
    gray_keys = {"filename", "product", "lead_min", "encoding",
                 "scale", "offset", "nodata", "units", "shape"}
    for name, entry in entries.items():
        if entry["product"] == "overlay":
            assert set(entry) == {"filename", "product", "lead_min",
                                  "encoding", "shape"}
            assert entry["encoding"] == "rgba8"
        else:
            assert set(entry) == gray_keys
            assert entry["encoding"] == "grayscale8"
            assert entry["nodata"] == NODATA_LEVEL
    assert entries[f"p_rain_10min_{STAMP}.png"]["lead_min"] == 10
    assert entries[f"p_rain_20min_{STAMP}.png"]["lead_min"] == 20
    assert entries[f"eta_{STAMP}.png"]["lead_min"] is None
    assert entries[f"intensity_{STAMP}.png"]["lead_min"] is None
    assert entries[f"overlay_now_{STAMP}.png"]["lead_min"] == 0
    assert entries[f"overlay_10min_{STAMP}.png"]["lead_min"] == 10


def test_timestamps_are_normalised_to_utc(geo: CompositeGeo, tmp_path: Path) -> None:
    """Non-UTC (but tz-aware) inputs land as the same instant in UTC — the
    cycle stamp comes from the UTC wall clock."""
    cest = timezone(timedelta(hours=2))
    result = _write(
        geo, tmp_path,
        radar_ts_utc=RADAR_TS.astimezone(cest),
        generated_at_utc=GENERATED_AT.astimezone(cest),
        overlay_fields_mm_h=None,
    )
    assert result.manifest_path.name == f"manifest_{STAMP}.json"
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["radar_ts_utc"] == "2026-08-28T12:00:00+00:00"
    assert manifest["generated_at_utc"] == "2026-08-28T12:03:21+00:00"


def test_naive_datetimes_are_rejected(geo: CompositeGeo, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _write(geo, tmp_path, radar_ts_utc=RADAR_TS.replace(tzinfo=None))
    with pytest.raises(ValueError, match="timezone-aware"):
        _write(geo, tmp_path, generated_at_utc=GENERATED_AT.replace(tzinfo=None))


def test_keep_cycles_must_be_positive(geo: CompositeGeo, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_cycles"):
        _write(geo, tmp_path, keep_cycles=0)


def test_png_round_trip_matches_in_memory_grids(
    geo: CompositeGeo, tmp_path: Path,
) -> None:
    """Decode every grayscale artifact with the manifest's scale/offset and
    compare against the in-memory product grids — ≤ half a quantisation
    step, NaN masks identical (the A4(c) criterion at unit level)."""
    products = _make_products()
    result = _write(geo, tmp_path, products=products)
    manifest = json.loads(result.manifest_path.read_text())
    in_memory = {
        f"p_rain_10min_{STAMP}.png": products.p_rain[10],
        f"p_rain_20min_{STAMP}.png": products.p_rain[20],
        f"eta_{STAMP}.png": products.eta_min,
        f"intensity_{STAMP}.png": products.intensity_mm_h,
    }
    checked = 0
    for entry in manifest["artifacts"]:
        if entry["product"] == "overlay":
            continue
        with Image.open(tmp_path / entry["filename"]) as img:
            assert img.mode == "L"
            levels = np.asarray(img)
        decoded = dequantise(levels, scale=entry["scale"], offset=entry["offset"])
        expected = in_memory[entry["filename"]]
        assert decoded.shape == expected.shape
        assert np.array_equal(np.isnan(decoded), np.isnan(expected))
        finite = ~np.isnan(expected)
        err = np.abs(decoded[finite] - expected[finite])
        assert float(err.max()) <= entry["scale"] / 2 + 1e-4
        checked += 1
    assert checked == 4


def test_overlay_size_and_nan_transparency(geo: CompositeGeo, tmp_path: Path) -> None:
    _write(geo, tmp_path)
    with Image.open(tmp_path / f"overlay_now_{STAMP}.png") as img:
        assert img.mode == "RGBA"
        assert img.size == (GRID_PX, GRID_PX)
        rgba = np.asarray(img)
    alpha = rgba[..., 3]
    assert (alpha[:4, :4] == 0).all(), "NaN pixels must be fully transparent"
    assert alpha[32, 32] == 255, "heavy rain must be opaque"
    assert alpha[10, 10] == 0, "sub-floor drizzle fades out (render.py convention)"
    assert alpha[50, 50] == 0, "dry pixels are transparent"


def test_no_overlays_is_allowed(geo: CompositeGeo, tmp_path: Path) -> None:
    result = _write(geo, tmp_path, overlay_fields_mm_h=None)
    assert not list(tmp_path.glob("overlay_*.png"))
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["overlay_grid"] is None
    assert all(e["product"] != "overlay" for e in manifest["artifacts"])


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def test_pruner_keeps_newest_24_cycles_and_foreign_files(
    geo: CompositeGeo, tmp_path: Path,
) -> None:
    foreign = [tmp_path / "readme.txt", tmp_path / "frame_01.png",
               tmp_path / "p_rain_10min_2026082800.png"]  # 10-digit stamp: foreign
    tmp_path.mkdir(exist_ok=True)
    for f in foreign:
        f.write_bytes(b"leave me alone")
    (tmp_path / "subdir").mkdir()

    base = datetime(2026, 8, 28, 0, 0, tzinfo=timezone.utc)
    n_cycles = 27
    for i in range(n_cycles):
        ts = base + timedelta(minutes=5 * i)
        result = _write(
            geo, tmp_path,
            radar_ts_utc=ts, generated_at_utc=ts + timedelta(minutes=3),
            overlay_fields_mm_h=None, keep_cycles=24,
        )

    stamps_on_disk = {
        m.group(1)
        for p in tmp_path.iterdir() if p.is_file()
        if (m := _STAMP_IN_NAME.search(p.name))
    }
    expected = {
        (base + timedelta(minutes=5 * i)).strftime("%Y%m%d%H%M")
        for i in range(n_cycles - 24, n_cycles)
    }
    assert stamps_on_disk == expected, "only the newest 24 cycles may remain"
    for f in foreign:
        assert f.exists(), f"foreign file {f.name} must survive pruning"
    assert (tmp_path / "subdir").is_dir()
    assert (tmp_path / LATEST_MANIFEST_NAME).is_file()
    # The 25th..27th write each pruned exactly one whole cycle:
    # 2 p_rain + eta + intensity + stamped manifest = 5 files.
    assert result.pruned_files == 5
    assert result.pruned_bytes > 0


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------

def test_atomic_write_never_exposes_partial_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A torn write (crash mid-``write_bytes``) must leave the published
    filename untouched — partial bytes only ever live in the ``.tmp``."""
    target = tmp_path / "manifest.json"
    na._atomic_write_bytes(target, b"old-good-manifest")

    def torn_write(self: Path, data: bytes) -> int:
        with open(self, "wb") as fh:
            fh.write(data[: len(data) // 2])
        raise OSError("disk full mid-write")

    monkeypatch.setattr(Path, "write_bytes", torn_write)
    with pytest.raises(OSError, match="disk full"):
        na._atomic_write_bytes(target, b"new-manifest-content")
    monkeypatch.undo()

    assert target.read_bytes() == b"old-good-manifest"
    tmp = tmp_path / "manifest.json.tmp"
    assert tmp.exists() and tmp.read_bytes() != b"new-manifest-content"


def test_manifest_is_written_last(
    geo: CompositeGeo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every PNG must be fully on disk before the manifest that references
    it appears; the stable alias lands after the stamped manifest."""
    order: list[str] = []
    real = na._atomic_write_bytes

    def recorder(target: Path, data: bytes) -> None:
        order.append(target.name)
        real(target, data)

    monkeypatch.setattr(na, "_atomic_write_bytes", recorder)
    _write(geo, tmp_path)
    assert order[-2:] == [f"manifest_{STAMP}.json", LATEST_MANIFEST_NAME]
    assert all(name.endswith(".png") for name in order[:-2])
    assert len(order) == 8
