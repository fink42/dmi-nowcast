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

=================  ============  ==========================================
product            range         rationale
=================  ============  ==========================================
``p_rain``         0 – 1         probability is naturally [0, 1]
``eta``            0 – 120 min   covers any ETA the 60-min STEPS horizon can
                                 produce, with headroom for longer horizons
``intensity``      0 – 100       mm/h — the Z–R rain-rate cap (CLAUDE.md
                                 contract), values above are clamped upstream
``motion_*_kmh``   −120 – +120   km/h — the flow is clipped to 30 px/frame
                                 upstream, i.e. ±90 km/h at 500 m / 10 min;
                                 ±120 leaves headroom for a shorter cadence
=================  ============  ==========================================

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
and ignoring anything that doesn't carry one — foreign files survive. Any
stamp the manifest just written *references* is protected on top of that,
so the observation history below can never outlive its own files.

Schema v2 — animation history + frame validity (R1). Every overlay entry
carries ``valid_ts_utc`` (the instant the frame depicts) and ``kind``
(``"observation"`` | ``"forecast"``), and the manifest additionally lists
the up-to-``history_frames`` most recent PRIOR cycles' ``overlay_now``
PNGs so the loop can start 30 min in the past. History is pure
reference — those files were written by earlier cycles and are neither
re-encoded nor re-counted here — and only files present on disk at write
time are listed, so a cold start simply has a shorter (or empty) history.

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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import structlog

from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.national import (
    DEFAULT_MOTION_SUPPORT_RADIUS_KM,
    NationalProducts,
)
from dmi_nowcast_core.render import _apply_colormap

_log = structlog.get_logger(__name__)

# Manifest schema version — bump on any breaking change to the JSON layout.
# v2 (R1/R2): overlay entries gained ``valid_ts_utc`` + ``kind``, the
# manifest gained trailing observation-history entries, a ``motion`` block
# and the two ``motion_*_kmh`` product grids.
MANIFEST_SCHEMA_VERSION = 2

# Trailing observed frames referenced by each manifest — 3 prior cycles is
# ~30 min at the 10-min fullRange cadence.
DEFAULT_HISTORY_FRAMES = 3

# Prior cycles' "now" overlays, the only artifact the history references.
_OVERLAY_NOW_RE = re.compile(r"^overlay_now_(\d{12})\.png$")

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


# Motion grids are symmetric about zero; ±120 km/h (see module docstring).
MOTION_MAX_ABS_KMH = 120.0

# The documented, fixed quantisation ranges (see module docstring).
QUANT_SPECS: dict[str, QuantSpec] = {
    "p_rain": QuantSpec(0.0, 1.0),
    "eta": QuantSpec(0.0, 120.0),
    "intensity": QuantSpec(0.0, 100.0),
    "motion_east_kmh": QuantSpec(-MOTION_MAX_ABS_KMH, MOTION_MAX_ABS_KMH),
    "motion_north_kmh": QuantSpec(-MOTION_MAX_ABS_KMH, MOTION_MAX_ABS_KMH),
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
    motion_east_kmh: np.ndarray | None = None,
    motion_north_kmh: np.ndarray | None = None,
    motion_support_radius_km: float = DEFAULT_MOTION_SUPPORT_RADIUS_KM,
    history_frames: int = DEFAULT_HISTORY_FRAMES,
) -> NationalArtifactsResult:
    """Write one cycle's national artifacts + manifest into ``out_dir``.

    ``overlay_fields_mm_h`` maps lead minutes → **native-500 m** advected
    rain-rate grids (mm/h), computed by the caller (this module never
    advects); key ``0`` is rendered as the "now" frame. ``None`` or ``{}``
    skips the overlays. ``geo`` must be the native composite's geometry —
    the product grids' geometry is derived from it via
    ``products.downsample_factor``.

    ``motion_east_kmh`` / ``motion_north_kmh`` are the R2 cell-motion grids
    on the **product** grid (same shape as ``products.eta_min``), in km/h,
    east- and north-positive, NaN where no honest estimate exists — as
    returned by ``dmi_nowcast_core.national.motion_grids_kmh``. Pass both or
    neither; ``motion_support_radius_km`` is echoed into the manifest's
    ``motion`` block so the client can explain the nodata region.

    ``history_frames`` caps how many prior cycles' ``overlay_now`` PNGs the
    manifest references as observation history (0 disables). Only files
    still on disk are referenced, and their stamps are protected from this
    cycle's pruning.

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
    if history_frames < 0:
        raise ValueError(f"history_frames must be >= 0, got {history_frames}")
    if (motion_east_kmh is None) != (motion_north_kmh is None):
        raise ValueError(
            "motion_east_kmh and motion_north_kmh must be passed together"
        )
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

    # --- R2 cell-motion grids → grayscale PNGs -----------------------------
    # Same product grid, same quantisation machinery as everything above, so
    # the browser samples an arrow exactly the way it samples a probability.
    if motion_east_kmh is not None and motion_north_kmh is not None:
        product_shape = products.eta_min.shape
        for product, field in (
            ("motion_east_kmh", motion_east_kmh),
            ("motion_north_kmh", motion_north_kmh),
        ):
            grid = np.asarray(field)
            if grid.shape != product_shape:
                raise ValueError(
                    f"{product} must be on the product grid {product_shape}, "
                    f"got {grid.shape}"
                )
            spec = QUANT_SPECS[product]
            _emit(
                f"{product}_{stamp}.png",
                _encode_gray_png(quantise(grid, spec)),
                _grid_entry(f"{product}_{stamp}.png", product, None, spec,
                            grid.shape, units="km/h"),
            )

    # --- 500 m deterministic overlays → RGBA PNGs --------------------------
    # Frame validity (schema v2). Lead 0 IS the radar observation, so it is
    # valid at ``radar_utc``. A forecast lead is "minutes from now", and the
    # caller advects it by ``lead + frame_age_min`` from radar-frame time
    # (compute.py's ``advect_field_series`` horizons) — so its validity is
    # ``radar_utc + frame_age_min + lead``, NOT ``radar_utc + lead``. At the
    # 10-min fullRange cadence the frame age is a whole animation step, so
    # the difference is not cosmetic.
    overlay_shape: tuple[int, int] | None = None
    overlay_start = len(artifacts)
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
        observed = int(lead) == 0
        valid_utc = radar_utc if observed else radar_utc + timedelta(
            minutes=float(products.frame_age_min) + float(lead),
        )
        _emit(
            name,
            _encode_rgba_png(_apply_colormap(field)),
            {
                "filename": name,
                "product": "overlay",
                "lead_min": int(lead),
                "kind": "observation" if observed else "forecast",
                "valid_ts_utc": valid_utc.isoformat(),
                "encoding": "rgba8",
                "shape": [int(field.shape[0]), int(field.shape[1])],
            },
        )

    # --- observation history: reference prior cycles' "now" overlays -------
    # Zero re-encode, zero new bytes — the animation just gets a past. Only
    # meaningful alongside a "now" frame, so it rides with the overlays.
    history: list[dict] = []
    if overlay_shape is not None and history_frames > 0:
        history = _history_entries(
            out_dir,
            stamp=stamp,
            radar_utc=radar_utc,
            shape=overlay_shape,
            limit=history_frames,
        )
        artifacts[overlay_start:overlay_start] = history

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
        motion=(
            {
                "grid": "product",
                "support_radius_km": float(motion_support_radius_km),
                "max_abs_kmh": MOTION_MAX_ABS_KMH,
                "convention": (
                    "motion_east_kmh / motion_north_kmh are the cell motion "
                    "in km/h on the product grid (see \"grid\"), east- and "
                    "north-positive. nodata (255) outside radar coverage and "
                    "farther than support_radius_km from any echo — there is "
                    "no motion estimate there, do not draw an arrow."
                ),
            }
            if motion_east_kmh is not None else None
        ),
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
    # Whatever the just-published manifest points at is off limits, however
    # small ``keep_cycles`` gets: a manifest referencing a file the same call
    # deleted is the one failure mode the history feature can introduce.
    pruned_files, pruned_bytes = _prune_old_cycles(
        out_dir, keep_cycles,
        protected_stamps=_referenced_stamps(artifacts) | {stamp},
    )

    _log.info(
        "national_artifacts_written",
        cycle=stamp,
        files=len(files_written),
        bytes=bytes_written,
        overlays=len(overlay_fields_mm_h or {}),
        history=len(history),
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


def _history_entries(
    out_dir: Path,
    *,
    stamp: str,
    radar_utc: datetime,
    shape: tuple[int, int],
    limit: int,
) -> list[dict]:
    """Manifest entries for the newest prior cycles' ``overlay_now`` PNGs.

    Reference-only: these files were written (and their bytes counted) by
    earlier cycles. Discovery is a directory listing, so a file that has
    been pruned, never written, or removed by hand simply isn't referenced —
    which is also what makes a cold start valid rather than special-cased.

    Validity time: the cycle stamp IS the radar observation time (the caller
    passes ``radar_ts_utc=composite.timestamp_utc``), but the stamp is
    minute-resolution, so the sibling ``manifest_<stamp>.json``'s
    ``radar_ts_utc`` is preferred when readable — it is the same instant
    with its seconds intact. The stamp is the fallback, never a guess.

    Entries are oldest-first and carry a NEGATIVE ``lead_min``: minutes of
    the past relative to this cycle's radar frame. The filename is an
    existing one, so the ``_STAMP_RE`` / ``_NOWCAST_NAME_RE`` naming contract
    is untouched.
    """
    stamps: list[str] = []
    try:
        for path in out_dir.iterdir():
            if not path.is_file():
                continue
            match = _OVERLAY_NOW_RE.match(path.name)
            if match is not None and match.group(1) < stamp:
                stamps.append(match.group(1))
    except OSError as exc:  # pragma: no cover - unreadable dir
        _log.warning("national_history_scan_failed", error=str(exc))
        return []

    entries: list[dict] = []
    for past_stamp in sorted(stamps, reverse=True)[:limit]:
        valid = _history_valid_ts(out_dir, past_stamp)
        entries.append({
            "filename": f"overlay_now_{past_stamp}.png",
            "product": "overlay",
            "lead_min": int(round((valid - radar_utc).total_seconds() / 60.0)),
            "kind": "observation",
            "valid_ts_utc": valid.isoformat(),
            "encoding": "rgba8",
            "shape": [int(shape[0]), int(shape[1])],
        })
    entries.reverse()  # oldest first — animation order
    return entries


def _history_valid_ts(out_dir: Path, past_stamp: str) -> datetime:
    """Radar validity of a past cycle: its own manifest, else its stamp."""
    manifest_path = out_dir / f"manifest_{past_stamp}.json"
    try:
        raw = json.loads(manifest_path.read_text())["radar_ts_utc"]
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return datetime.strptime(past_stamp, _STAMP_FMT).replace(tzinfo=timezone.utc)


def _referenced_stamps(artifacts: list[dict]) -> set[str]:
    """Cycle stamps of every file the manifest points at."""
    stamps = set()
    for entry in artifacts:
        match = _STAMP_RE.match(str(entry.get("filename", "")))
        if match is not None:
            stamps.add(match.group(1))
    return stamps


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
    motion: dict | None = None,
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
        # R2 motion grids: geometry is the product ``grid`` block above (same
        # shape, same UL corner, same pixel scale) — not duplicated here.
        # This block carries only what geometry can't say: the nodata rule.
        # null when no motion grids were written this cycle.
        "motion": motion,
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


def _prune_old_cycles(
    out_dir: Path,
    keep_cycles: int,
    *,
    protected_stamps: set[str] | frozenset[str] = frozenset(),
) -> tuple[int, int]:
    """Delete artifacts/manifests of all but the newest ``keep_cycles`` cycles.

    Cycle membership is parsed from the ``_YYYYMMDDHHMM.png|json`` filename
    stamp; files that don't match (foreign files, the stable
    ``manifest.json`` alias, ``*.tmp`` leftovers) are left alone.
    ``protected_stamps`` survive regardless of age — the caller passes every
    stamp the manifest it just wrote references, which makes "the newest
    manifest never points at a deleted file" an invariant rather than a
    consequence of ``keep_cycles`` happening to exceed the history depth.
    Returns ``(files_deleted, bytes_deleted)``.
    """
    by_stamp: dict[str, list[Path]] = {}
    for path in out_dir.iterdir():
        if not path.is_file():
            continue
        match = _STAMP_RE.match(path.name)
        if match is None:
            continue
        by_stamp.setdefault(match.group(1), []).append(path)

    doomed_stamps = [
        s for s in sorted(by_stamp, reverse=True)[keep_cycles:]
        if s not in protected_stamps
    ]
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
