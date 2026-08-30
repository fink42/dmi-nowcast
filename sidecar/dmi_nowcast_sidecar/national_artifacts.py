"""National artifact writer — quantised PNGs + JSON manifest (Phase A §A2).

Turns one cycle's :class:`~dmi_nowcast_core.national.NationalProducts` into
static, immutable-cacheable artifacts served by the ``/nowcast/`` endpoints
(package A3):

- each product grid → an 8-bit **grayscale** PNG with linear scale/offset
  quantisation (value 255 reserved for NaN/nodata),
- the supplied native-500 m advected rain fields → **colormapped RGBA**
  overlay PNGs (render.py's colormap + NaN-transparency conventions),
- one JSON manifest per cycle describing every artifact, written **last**
  and atomically, plus a stable ``manifest.json`` alias pointing at the
  newest cycle (what ``GET /nowcast/manifest.json`` serves).

Quantisation ranges (fixed, documented, stable across cycles so browser
consumers can hard-code expectations; the authoritative per-artifact
scale/offset still travels in the manifest):

===============  ==========  ============================================
product          range       rationale
===============  ==========  ============================================
``p_rain``       0 – 1       probability is naturally [0, 1]
``eta``          0 – 120 min covers any ETA the 60-min STEPS horizon can
                             produce, with headroom for longer horizons
``intensity``    0 – 100     mm/h — the Z–R rain-rate cap (CLAUDE.md
                             contract), values above are clamped upstream
===============  ==========  ============================================

Levels 0…254 span the range linearly (``value = level * scale + offset``),
so the worst-case round-trip error is half a step: ½ · range/254.

Filenames are cycle-stamped from the **radar** timestamp in UTC
(``p_rain_20min_202608281200.png``) so they never change content under a
given name — safe for ``Cache-Control: immutable``. All writes go through a
local atomic helper (write ``.tmp`` → ``os.replace``), modelled on the
sidecar's existing pattern but deliberately not imported from
``compute.py`` (compute will call *this* module; importing the other way
would create a cycle).

A pruner runs after each write and keeps the newest ``keep_cycles`` cycles
(decided: 24 ≈ 2 h at the 5-min cadence), parsing stamps out of filenames
and ignoring anything that doesn't carry one — foreign files survive.

Grid geometry: the manifest's ``grid`` block serialises everything a Phase C
browser needs to sample the product grids client-side. The ×4 downsample in
``probabilistic.run_ensemble`` is plain stride slicing (``[::f, ::f]``), so
the downsampled grid keeps the native UL projection corner and simply has an
effective pixel scale of native × ``downsample_factor`` — the same
convention ``aggregate_at_home`` uses (``home.row / f``). Browser-side:
``col = (x - x_ul_m) / pixel_scale_x_m``, ``row = (y_ul_m - y) /
pixel_scale_y_m``.

All timestamps are UTC ISO 8601 with an explicit offset (CLAUDE.md UTC
contract): conversion to local time happens at the consumer, never here.
"""
from __future__ import annotations

import io
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import structlog

from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.national import NationalProducts
from dmi_nowcast_core.render import _apply_colormap

_log = structlog.get_logger(__name__)

# Manifest schema version — bump on any breaking change to the JSON layout.
MANIFEST_SCHEMA_VERSION = 1

# Stable alias for the newest cycle's manifest; served by A3's
# ``GET /nowcast/manifest.json``. Never pruned (carries no cycle stamp).
LATEST_MANIFEST_NAME = "manifest.json"

# 8-bit quantisation: levels 0..254 span the value range; 255 is nodata.
NODATA_LEVEL = 255
_MAX_LEVEL = 254

# Cycle stamp: UTC %Y%m%d%H%M — lexicographic order == chronological order.
_STAMP_FMT = "%Y%m%d%H%M"
_STAMP_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*_(\d{12})\.(?:png|json)$")


@dataclass(frozen=True)
class QuantSpec:
    """Linear 8-bit quantisation over ``[lo, hi]`` with 255 = nodata."""

    lo: float
    hi: float

    @property
    def scale(self) -> float:
        return (self.hi - self.lo) / _MAX_LEVEL

    @property
    def offset(self) -> float:
        return self.lo


# The documented, fixed quantisation ranges (see module docstring).
QUANT_SPECS: dict[str, QuantSpec] = {
    "p_rain": QuantSpec(0.0, 1.0),
    "eta": QuantSpec(0.0, 120.0),
    "intensity": QuantSpec(0.0, 100.0),
}


@dataclass(frozen=True)
class NationalArtifactsResult:
    """Outcome of one artifact-writing cycle.

    ``manifest_path`` is the cycle-stamped manifest (the immutable one);
    ``latest_manifest_path`` the stable ``manifest.json`` alias.
    ``files_written`` lists every file written this cycle, manifests
    included; ``bytes_written`` is their total payload size. ``pruned_*``
    report what the retention pass deleted.
    """

    manifest_path: Path
    latest_manifest_path: Path
    files_written: tuple[Path, ...]
    bytes_written: int
    pruned_files: int
    pruned_bytes: int


def quantise(field: np.ndarray, spec: QuantSpec) -> np.ndarray:
    """Quantise a float grid to uint8: 0..254 linear over the spec's range,
    255 for any non-finite value. In-range values round-trip through
    :func:`dequantise` within half a quantisation step; out-of-range finite
    values are clamped to the range first."""
    field = np.asarray(field, dtype=np.float32)
    finite = np.isfinite(field)
    with np.errstate(invalid="ignore"):
        clipped = np.clip(field, spec.lo, spec.hi)
        levels = np.rint((clipped - spec.offset) / spec.scale)
    return np.where(finite, levels, NODATA_LEVEL).astype(np.uint8)


def dequantise(levels: np.ndarray, *, scale: float, offset: float) -> np.ndarray:
    """Invert :func:`quantise` given the manifest's scale/offset; 255 → NaN."""
    levels = np.asarray(levels)
    out = levels.astype(np.float32) * np.float32(scale) + np.float32(offset)
    out[levels == NODATA_LEVEL] = np.nan
    return out


def write_national_artifacts(
    products: NationalProducts,
    *,
    geo: CompositeGeo,
    radar_ts_utc: datetime,
    generated_at_utc: datetime,
    overlay_fields_mm_h: dict[int, np.ndarray] | None,
    out_dir: Path,
    keep_cycles: int = 24,
    calibration: dict | None = None,
) -> NationalArtifactsResult:
    """Write one cycle's national artifacts + manifest into ``out_dir``.

    ``overlay_fields_mm_h`` maps lead minutes → **native-500 m** advected
    rain-rate grids (mm/h), computed by the caller (this module never
    advects); key ``0`` is rendered as the "now" frame. ``None`` or ``{}``
    skips the overlays. ``geo`` must be the native composite's geometry —
    the product grids' geometry is derived from it via
    ``products.downsample_factor``.

    ``calibration`` is the §B4 calibration-metadata block echoed verbatim
    into the manifest (fitted_at, curve-file metadata, ``calibrated_leads``)
    — pass ``None`` when the served grids are uncalibrated and the manifest
    carries ``"calibration": null``. This module never applies curves; the
    caller calibrates the grids before handing them over.

    Both datetimes must be timezone-aware; they are converted to UTC for
    the manifest and the filename cycle stamp. Ordering guarantee: every
    PNG is fully (and atomically) on disk before the manifest that
    references it is written; the stamped manifest lands before the stable
    ``manifest.json`` alias; pruning runs last.
    """
    if radar_ts_utc.tzinfo is None or generated_at_utc.tzinfo is None:
        raise ValueError("radar_ts_utc and generated_at_utc must be timezone-aware")
    if keep_cycles < 1:
        raise ValueError(f"keep_cycles must be >= 1, got {keep_cycles}")
    radar_utc = radar_ts_utc.astimezone(timezone.utc)
    generated_utc = generated_at_utc.astimezone(timezone.utc)
    stamp = radar_utc.strftime(_STAMP_FMT)

    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    files_written: list[Path] = []
    bytes_written = 0
    artifacts: list[dict] = []

    def _emit(name: str, data: bytes, entry: dict) -> None:
        nonlocal bytes_written
        path = out_dir / name
        _atomic_write_bytes(path, data)
        files_written.append(path)
        bytes_written += len(data)
        artifacts.append(entry)

    # --- product grids → grayscale PNGs -----------------------------------
    p_spec = QUANT_SPECS["p_rain"]
    for lead in products.leads_min:
        grid = products.p_rain[lead]
        _emit(
            f"p_rain_{lead}min_{stamp}.png",
            _encode_gray_png(quantise(grid, p_spec)),
            _grid_entry(f"p_rain_{lead}min_{stamp}.png", "p_rain", lead,
                        p_spec, grid.shape, units="probability"),
        )

    eta_spec = QUANT_SPECS["eta"]
    _emit(
        f"eta_{stamp}.png",
        _encode_gray_png(quantise(products.eta_min, eta_spec)),
        _grid_entry(f"eta_{stamp}.png", "eta", None, eta_spec,
                    products.eta_min.shape, units="min"),
    )

    int_spec = QUANT_SPECS["intensity"]
    _emit(
        f"intensity_{stamp}.png",
        _encode_gray_png(quantise(products.intensity_mm_h, int_spec)),
        _grid_entry(f"intensity_{stamp}.png", "intensity", None, int_spec,
                    products.intensity_mm_h.shape, units="mm/h"),
    )

    # --- 500 m deterministic overlays → RGBA PNGs --------------------------
    overlay_shape: tuple[int, int] | None = None
    for lead in sorted((overlay_fields_mm_h or {})):
        field = np.asarray(overlay_fields_mm_h[lead])
        if field.ndim != 2:
            raise ValueError(
                f"overlay field for lead {lead} must be 2-D, got shape {field.shape}"
            )
        if overlay_shape is not None and field.shape != overlay_shape:
            raise ValueError(
                f"overlay fields must share one shape; lead {lead} has "
                f"{field.shape}, earlier leads had {overlay_shape}"
            )
        overlay_shape = field.shape
        name = (f"overlay_now_{stamp}.png" if lead == 0
                else f"overlay_{lead}min_{stamp}.png")
        _emit(
            name,
            _encode_rgba_png(_apply_colormap(field)),
            {
                "filename": name,
                "product": "overlay",
                "lead_min": int(lead),
                "encoding": "rgba8",
                "shape": [int(field.shape[0]), int(field.shape[1])],
            },
        )

    # --- manifest (written last, atomically) -------------------------------
    manifest = _build_manifest(
        products=products,
        geo=geo,
        radar_utc=radar_utc,
        generated_utc=generated_utc,
        stamp=stamp,
        artifacts=artifacts,
        overlay_shape=overlay_shape,
        calibration=calibration,
    )
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    manifest_path = out_dir / f"manifest_{stamp}.json"
    _atomic_write_bytes(manifest_path, manifest_bytes)
    files_written.append(manifest_path)
    bytes_written += len(manifest_bytes)
    latest_path = out_dir / LATEST_MANIFEST_NAME
    _atomic_write_bytes(latest_path, manifest_bytes)
    files_written.append(latest_path)
    bytes_written += len(manifest_bytes)

    # --- retention ---------------------------------------------------------
    pruned_files, pruned_bytes = _prune_old_cycles(out_dir, keep_cycles)

    _log.info(
        "national_artifacts_written",
        cycle=stamp,
        files=len(files_written),
        bytes=bytes_written,
        overlays=len(overlay_fields_mm_h or {}),
        pruned_files=pruned_files,
        pruned_bytes=pruned_bytes,
        write_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    return NationalArtifactsResult(
        manifest_path=manifest_path,
        latest_manifest_path=latest_path,
        files_written=tuple(files_written),
        bytes_written=bytes_written,
        pruned_files=pruned_files,
        pruned_bytes=pruned_bytes,
    )


def _grid_entry(
    filename: str,
    product: str,
    lead_min: int | None,
    spec: QuantSpec,
    shape: tuple[int, ...],
    *,
    units: str,
) -> dict:
    return {
        "filename": filename,
        "product": product,
        "lead_min": None if lead_min is None else int(lead_min),
        "encoding": "grayscale8",
        "scale": spec.scale,
        "offset": spec.offset,
        "nodata": NODATA_LEVEL,
        "units": units,
        "shape": [int(shape[0]), int(shape[1])],
    }


def _build_manifest(
    *,
    products: NationalProducts,
    geo: CompositeGeo,
    radar_utc: datetime,
    generated_utc: datetime,
    stamp: str,
    artifacts: list[dict],
    overlay_shape: tuple[int, int] | None,
    calibration: dict | None = None,
) -> dict:
    composite = geo.composite
    x_ul, y_ul = geo.projection_origin_m
    h, w = products.eta_min.shape
    f = products.downsample_factor
    grid = {
        "proj4": composite.projection,
        "x_ul_m": x_ul,
        "y_ul_m": y_ul,
        "pixel_scale_x_m": float(composite.xscale_m) * f,
        "pixel_scale_y_m": float(composite.yscale_m) * f,
        "shape": [int(h), int(w)],
        "downsample_factor": f,
    }
    overlay_grid = None
    if overlay_shape is not None:
        overlay_grid = {
            "proj4": composite.projection,
            "x_ul_m": x_ul,
            "y_ul_m": y_ul,
            "pixel_scale_x_m": float(composite.xscale_m),
            "pixel_scale_y_m": float(composite.yscale_m),
            "shape": [int(overlay_shape[0]), int(overlay_shape[1])],
            "downsample_factor": 1,
        }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "cycle": stamp,
        "radar_ts_utc": radar_utc.isoformat(),
        "generated_at_utc": generated_utc.isoformat(),
        "threshold_mm_h": products.threshold_mm_h,
        "timestep_min": products.timestep_min,
        "frame_age_min": products.frame_age_min,
        "n_members": products.n_members,
        "leads_min": [int(lead) for lead in products.leads_min],
        "grid": grid,
        "overlay_grid": overlay_grid,
        # §B4 calibration metadata: null when the served grids are raw;
        # otherwise fitted_at + curve-file echo + the exact leads whose
        # p_rain grids went through a curve (raw is recoverable by
        # inverting the published breakpoints).
        "calibration": calibration,
        "artifacts": artifacts,
    }


def _encode_gray_png(levels: np.ndarray) -> bytes:
    """uint8 (h, w) → grayscale PNG bytes."""
    from PIL import Image  # lazy, matching render.py's convention

    buf = io.BytesIO()
    Image.fromarray(levels, "L").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _encode_rgba_png(rgba: np.ndarray) -> bytes:
    """uint8 (h, w, 4) → RGBA PNG bytes (NaN pixels already alpha-0)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Write ``data`` to ``target`` atomically (tempfile in same dir → replace).

    Local copy of the sidecar's established pattern (see ``compute.py``);
    not imported from there because compute will call this module — the
    import must not point back.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)


def _prune_old_cycles(out_dir: Path, keep_cycles: int) -> tuple[int, int]:
    """Delete artifacts/manifests of all but the newest ``keep_cycles`` cycles.

    Cycle membership is parsed from the ``_YYYYMMDDHHMM.png|json`` filename
    stamp; files that don't match (foreign files, the stable
    ``manifest.json`` alias, ``*.tmp`` leftovers) are left alone. Returns
    ``(files_deleted, bytes_deleted)``.
    """
    by_stamp: dict[str, list[Path]] = {}
    for path in out_dir.iterdir():
        if not path.is_file():
            continue
        match = _STAMP_RE.match(path.name)
        if match is None:
            continue
        by_stamp.setdefault(match.group(1), []).append(path)

    doomed_stamps = sorted(by_stamp, reverse=True)[keep_cycles:]
    files_deleted = 0
    bytes_deleted = 0
    for doomed_stamp in doomed_stamps:
        for path in by_stamp[doomed_stamp]:
            try:
                size = path.stat().st_size
                path.unlink()
            except OSError as exc:
                _log.warning("national_prune_failed", file=path.name, error=str(exc))
                continue
            files_deleted += 1
            bytes_deleted += size
    if files_deleted:
        _log.info(
            "national_artifacts_pruned",
            cycles=len(doomed_stamps),
            files=files_deleted,
            bytes=bytes_deleted,
        )
    return files_deleted, bytes_deleted
