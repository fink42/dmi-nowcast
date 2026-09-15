"""Review-bundle radar frames: which stamps the bundle needs, and their PNGs.

Two jobs, deliberately split so the expensive one can be interrupted:

:func:`frame_plan` decides **which radar stamps** a bundle of events needs.
Pure arithmetic over the anchors, no I/O, so the builder can print the plan
(and its byte estimate) before committing an eight-hour render.

:func:`write_frames` renders each planned stamp from the corpus archive into
the bundle's ``frames/`` directory as a pair of PNGs on the ×4 product grid:

``<stamp>.overlay.png``
    RGBA, ``render._apply_colormap`` over the observed rain rate — what the
    reviewer looks at.
``<stamp>.observed.png``
    grayscale8, ``quantise(field, QUANT_SPECS["observed_mm_h"])`` — what the
    browser samples under the cursor, so the number in the read-out is the
    same mm/h the colour was drawn from rather than a colour inverted back
    into a rate.

Both are the *observation*, never a forecast: see the plan's §4.10. The
bundle cannot show what the model expected at +30 min without re-running
STEPS for every cycle, and advecting the observation with ``bulk_kmh`` to
fake it would show a movie the model never produced.

Why the frame window is wider than the decision window
------------------------------------------------------

The decision window is ``anchor ± window_min``. The FRAME window is
``anchor − (window_min + pad_min + cadence_min)`` → ``anchor + window_min``.

A composite is 13–24 minutes old by the time a cycle stands on it
(:data:`~dmi_nowcast_core.review_schema.DEFAULT_FRAME_PAD_MIN` documents
this), so the decision at the left edge of the decision window refers to a
frame from *before* that edge. ``cadence_min`` is in the subtraction on top
of the pad because stamps sit on the product's own 10-minute grid: flooring
an arbitrary anchor onto that grid can cost up to one cadence, and paying
for it up front keeps the guarantee "every frame a decision in the window
stood on is in the bundle" true for every anchor, not just the on-grid ones.
Without it the reviewer scrubs to the exact moment that matters and finds a
blank map.

The two edges stay separate values all the way into the manifest
(``window.from_utc`` vs ``window.frames_from_utc``) so nobody later reads a
frame edge as a decision edge.

Why dedup is the whole economy
------------------------------

300 events × ~22 stamps is ~6,600 slots, but a stratified sample clusters
on rain days, so the real dedup is 20–40 %. Frames are therefore named by
stamp alone and shared by every event that needs them — which is also why
they are rendered on the national product grid rather than cropped per
event: a per-event crop cannot dedup at all, and comes out *larger* for a
*smaller* field of view.

fullRange only, by default
--------------------------

``include_doppler`` is off because the decision path ran on fullRange: at
:x0 minutes, 10 min IS the decision cadence. Doppler (:x5) covers ~40 % of
the area on a different intensity distribution, so interleaving raw :x5
frames puts a coverage edge and an intensity step into the movie that a
reviewer would read as weather — a tool that manufactures its own
false-alarm causes. When it is switched on, every doppler stamp is tagged
``product: "doppler"`` so the UI can badge it.

Geometry
--------

The ``grid`` block this module emits is built exactly as
``national_artifacts._grid_entry`` / ``_build_manifest`` build theirs: same
field names (:data:`~dmi_nowcast_core.review_schema.GRID_BLOCK_FIELDS`),
same values, same downsample convention — stride slicing keeps the native
UL projection corner and multiplies the pixel scale by ``f``. That is
load-bearing: the review page reuses ``lib/map/warp.ts`` and
``lib/nowcast/sampler.ts`` unchanged, and they parse that exact block.

The atomic-write and PNG-encoding helpers are imported from
``national_artifacts`` rather than copied: this module is a batch tool that
nothing in the live service imports, so there is no cycle to avoid, and two
copies of an atomic write is how they drift.

All timestamps here are UTC and tz-aware; naive datetimes are rejected at
the door rather than silently assumed to be UTC.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import structlog
from pyproj import CRS, Transformer

from dmi_nowcast_core.corpus import (
    SCAN_TYPE_DOPPLER,
    SCAN_TYPE_FULL_RANGE,
    SCAN_TYPE_UNKNOWN,
    archive_path_for,
    scan_type_from_filename,
)
from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.national import observed_rain_grid
from dmi_nowcast_core.parse import RadarComposite, parse_composite
from dmi_nowcast_core.render import (
    _COLORMAP_STOPS,
    _RENDER_FLOOR_MM_H,
    _RENDER_MIN_ALPHA,
    _RENDER_SOLID_MM_H,
    _apply_colormap,
)
from dmi_nowcast_core.review_schema import (
    DEFAULT_FRAME_PAD_MIN,
    DEFAULT_WINDOW_MIN,
    GRID_BLOCK_FIELDS,
)
from dmi_nowcast_core.transform import dbz_to_rain_rate

from .national_artifacts import (
    NODATA_LEVEL,
    QUANT_SPECS,
    _atomic_write_bytes,
    _encode_gray_png,
    _encode_rgba_png,
    quantise,
)

_log = structlog.get_logger(__name__)

#: Frames live in one flat directory under the bundle root, named by stamp
#: alone — that IS the dedup: two events 30 minutes apart share every frame
#: their windows have in common, without either knowing about the other.
FRAMES_DIRNAME = "frames"

OVERLAY_SUFFIX = ".overlay.png"
OBSERVED_SUFFIX = ".observed.png"

#: ``%Y%m%d%H%M`` in UTC — the composite filename's own stamp, so
#: lexicographic order is chronological order and the frames directory
#: sorts into animation order with ``sorted()``.
STAMP_FMT = "%Y%m%d%H%M"

#: The fullRange cadence. DMI interleaves fullRange at :x0 and doppler at
#: :x5 (``corpus._SCAN_TYPE_BY_MINUTE_MOD_10``), so a cadence that is not a
#: multiple of 10 would plan stamps that cannot exist as fullRange frames.
DEFAULT_CADENCE_MIN = 10

#: Minutes from a fullRange stamp to the doppler frame that follows it.
DOPPLER_OFFSET_MIN = 5

#: Product-grid stride. 4 at 500 m = 2 km pixels, the same grid the national
#: artifacts and the website's sampler already use.
DEFAULT_DOWNSAMPLE_FACTOR = 4

#: The quantisation spec for the grayscale frame. Shared object with
#: ``intensity`` and ``forecast_mm_h`` (same quantity, same cap), so a
#: browser that decodes one decodes all of them.
OBSERVED_QUANT_KEY = "observed_mm_h"

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedStamp:
    """One radar frame the bundle needs, named once for all of its events."""

    stamp: str            # "202606121140"
    ts_utc: datetime      # the composite's nominal time
    product: str          # SCAN_TYPE_FULL_RANGE | SCAN_TYPE_DOPPLER
    filename: str         # dk.com.202606121140.500_max.h5

    @property
    def overlay_name(self) -> str:
        return f"{self.stamp}{OVERLAY_SUFFIX}"

    @property
    def observed_name(self) -> str:
        return f"{self.stamp}{OBSERVED_SUFFIX}"


@dataclass(frozen=True)
class EventFrames:
    """One event's own scrubber: its two windows and its ordered stamps.

    ``window_*`` are the DECISION edges and ``frames_*`` the wider frame
    edges; they are separate fields because the bundle keeps them separate
    (``window.from_utc`` vs ``window.frames_from_utc``) and a reader that
    conflates them will place a decision against the wrong frame.
    """

    event_id: str
    anchor_utc: datetime
    window_from_utc: datetime
    window_to_utc: datetime
    frames_from_utc: datetime
    frames_to_utc: datetime
    stamps: tuple[str, ...]      # chronological, oldest first

    @property
    def n_frames(self) -> int:
        return len(self.stamps)


@dataclass(frozen=True)
class FramePlan:
    """The deduplicated frame set for a whole bundle, plus each event's list.

    ``stamps`` is chronological and unique; ``events`` keeps the caller's
    order. ``stamps_total`` (pre-dedup) and ``stamps_unique`` are both
    reported because their ratio is what decides whether a bundle fits the
    ~250 MB budget — a number worth printing before a long render, not
    discovering afterwards.
    """

    stamps: tuple[PlannedStamp, ...]
    events: tuple[EventFrames, ...]
    window_min: int
    pad_min: int
    cadence_min: int
    include_doppler: bool

    @property
    def stamps_unique(self) -> int:
        return len(self.stamps)

    @property
    def stamps_total(self) -> int:
        """Slots before dedup: what a per-event bundle would have rendered."""
        return sum(event.n_frames for event in self.events)

    @property
    def dedup_saving_pct(self) -> float:
        """Percent of pre-dedup slots the shared naming removes. 0 when empty."""
        total = self.stamps_total
        if total == 0:
            return 0.0
        return 100.0 * (1.0 - self.stamps_unique / total)

    @property
    def frames_from_utc(self) -> datetime | None:
        """Earliest frame edge over all events; ``None`` for an empty plan."""
        return min((e.frames_from_utc for e in self.events), default=None)

    @property
    def frames_to_utc(self) -> datetime | None:
        return max((e.frames_to_utc for e in self.events), default=None)

    @property
    def products(self) -> tuple[str, ...]:
        """Which DMI products the plan draws on, in cadence order."""
        seen = {stamp.product for stamp in self.stamps}
        return tuple(p for p in (SCAN_TYPE_FULL_RANGE, SCAN_TYPE_DOPPLER) if p in seen)

    def stamp_strings(self) -> tuple[str, ...]:
        return tuple(stamp.stamp for stamp in self.stamps)

    def by_event_id(self) -> dict[str, EventFrames]:
        return {event.event_id: event for event in self.events}


def composite_filename(ts_utc: datetime) -> str:
    """``dk.com.YYYYMMDDhhmm.500_max.h5`` — the corpus archive's own name."""
    return f"dk.com.{stamp_of(ts_utc)}.500_max.h5"


def stamp_of(ts_utc: datetime) -> str:
    """UTC ``YYYYMMDDhhmm`` for a tz-aware instant."""
    return _as_utc(ts_utc, "timestamp").strftime(STAMP_FMT)


def stamp_to_datetime(stamp: str) -> datetime:
    """Inverse of :func:`stamp_of`."""
    return datetime.strptime(stamp, STAMP_FMT).replace(tzinfo=timezone.utc)


def frame_plan(
    anchors: Sequence[tuple[str, datetime]],
    *,
    window_min: int = DEFAULT_WINDOW_MIN,
    pad_min: int = DEFAULT_FRAME_PAD_MIN,
    cadence_min: int = DEFAULT_CADENCE_MIN,
    include_doppler: bool = False,
) -> FramePlan:
    """Plan the radar stamps a bundle of ``(event_id, anchor_utc)`` needs.

    Pure: no filesystem, no archive, no network. Two things it exists to
    get right, both explained at length in the module docstring — the past
    edge is padded by ``window_min + pad_min + cadence_min`` (NOT ±90), and
    the returned stamp set is deduplicated across events.

    Raises ``ValueError`` on a naive anchor (UTC discipline), on a duplicate
    ``event_id`` (annotations are keyed on it, so two anchors under one id
    would silently collide), on a negative window or pad, and on a cadence
    that is not a positive multiple of 10 — fullRange composites exist only
    on :x0 minutes, so any other cadence plans frames that cannot exist.
    """
    if window_min < 0 or pad_min < 0:
        raise ValueError(f"window_min/pad_min must be >= 0, got {window_min}/{pad_min}")
    if cadence_min <= 0 or cadence_min % 10 != 0:
        raise ValueError(
            "cadence_min must be a positive multiple of 10 (fullRange frames "
            f"exist only on :x0 minutes), got {cadence_min}"
        )

    events: list[EventFrames] = []
    unique: dict[str, PlannedStamp] = {}
    seen_ids: set[str] = set()

    for event_id, anchor in anchors:
        if event_id in seen_ids:
            raise ValueError(f"duplicate event_id in anchors: {event_id!r}")
        seen_ids.add(event_id)
        anchor_utc = _as_utc(anchor, f"anchor for {event_id}")

        window_from = anchor_utc - timedelta(minutes=window_min)
        window_to = anchor_utc + timedelta(minutes=window_min)
        frames_to = window_to
        frames_from = anchor_utc - timedelta(
            minutes=window_min + pad_min + cadence_min
        )

        planned = _stamps_in_window(
            frames_from, frames_to,
            cadence_min=cadence_min,
            include_doppler=include_doppler,
        )
        for stamp in planned:
            unique.setdefault(stamp.stamp, stamp)
        events.append(EventFrames(
            event_id=event_id,
            anchor_utc=anchor_utc,
            window_from_utc=window_from,
            window_to_utc=window_to,
            frames_from_utc=frames_from,
            frames_to_utc=frames_to,
            stamps=tuple(stamp.stamp for stamp in planned),
        ))

    ordered = tuple(sorted(unique.values(), key=lambda s: s.ts_utc))
    return FramePlan(
        stamps=ordered,
        events=tuple(events),
        window_min=int(window_min),
        pad_min=int(pad_min),
        cadence_min=int(cadence_min),
        include_doppler=bool(include_doppler),
    )


def _stamps_in_window(
    frames_from: datetime,
    frames_to: datetime,
    *,
    cadence_min: int,
    include_doppler: bool,
) -> tuple[PlannedStamp, ...]:
    """The stamps on the product grid inside ``[frames_from, frames_to]``.

    fullRange stamps sit on the ``cadence_min`` grid anchored at the epoch
    (which, for any multiple of 10, lands them on :x0 minutes). The first
    one is the grid point at or AFTER ``frames_from``: flooring instead
    would widen the window by up to a cadence for no gain, because
    ``frames_from`` already carries a whole cadence of slack for exactly
    this rounding.

    Doppler stamps, when asked for, are each fullRange stamp + 5 min, so
    the series interleaves and every :x5 frame is adjacent to the :x0 frame
    the decision path actually used.
    """
    step = timedelta(minutes=cadence_min)
    out: list[PlannedStamp] = []
    ts = _ceil_to_cadence(frames_from, cadence_min)
    while ts <= frames_to:
        out.append(_planned_stamp(ts))
        if include_doppler:
            doppler_ts = ts + timedelta(minutes=DOPPLER_OFFSET_MIN)
            if doppler_ts <= frames_to:
                out.append(_planned_stamp(doppler_ts))
        ts += step
    return tuple(out)


def _planned_stamp(ts_utc: datetime) -> PlannedStamp:
    """Build a :class:`PlannedStamp`, reading the product off the filename.

    The product is not asserted from the arithmetic that produced ``ts_utc``
    but read back through :func:`~dmi_nowcast_core.corpus.scan_type_from_filename`,
    so the bundle's product labels come from the same rule the archive index
    uses. ``unknown`` is raised on rather than passed through: it must never
    reach a consumer that would read it as fullRange.
    """
    filename = composite_filename(ts_utc)
    product = scan_type_from_filename(filename)
    if product == SCAN_TYPE_UNKNOWN:  # pragma: no cover - guarded by cadence check
        raise ValueError(
            f"planned stamp {filename} is on neither product's minute grid"
        )
    return PlannedStamp(
        stamp=stamp_of(ts_utc), ts_utc=ts_utc, product=product, filename=filename,
    )


def _ceil_to_cadence(ts: datetime, cadence_min: int) -> datetime:
    step = cadence_min * 60
    seconds = (ts - _EPOCH).total_seconds()
    return _EPOCH + timedelta(seconds=math.ceil(seconds / step) * step)


def _as_utc(ts: datetime, what: str) -> datetime:
    """Reject naive datetimes; normalise aware ones to UTC.

    The bundle states once that every instant is UTC with an explicit
    offset. A naive datetime here would be *assumed* UTC and would be wrong
    by an hour or two for exactly the summer events the review cares about.
    """
    if ts.tzinfo is None:
        raise ValueError(f"{what} must be timezone-aware UTC, got naive {ts!r}")
    return ts.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameWriteResult:
    """What one :func:`write_frames` run did, in the manifest's own terms.

    ``bytes_written`` counts only this run's payload; ``bytes_total`` is
    every frame file of the plan present on disk afterwards, which is the
    number to compare against the ~250 MB bundle budget after a resumed
    run (where ``bytes_written`` is near zero by design).
    """

    frames_dir: Path
    written: tuple[str, ...]           # stamps rendered this run
    skipped: tuple[str, ...]           # stamps already complete on disk
    missing: tuple[str, ...]           # stamps with no usable composite
    missing_reasons: dict[str, str]    # stamp → why, for the builder's notes
    bytes_written: int
    bytes_total: int
    grid: dict                         # GRID_BLOCK_FIELDS, manifest["grid"]
    encodings: dict                    # per-encoding metadata, manifest["frames"]
    downsample_factor: int
    elapsed_s: float

    @property
    def n_present(self) -> int:
        """Stamps with a complete PNG pair on disk (written + skipped)."""
        return len(self.written) + len(self.skipped)

    @property
    def mean_bytes_per_frame(self) -> float:
        return self.bytes_total / self.n_present if self.n_present else 0.0


def write_frames(
    plan: FramePlan,
    *,
    corpus_dir: Path,
    bundle_dir: Path,
    downsample_factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
    log_every: int = 100,
    on_frame: Callable[[str, int], None] | None = None,
) -> FrameWriteResult:
    """Render every planned stamp into ``<bundle_dir>/frames/``.

    Resumable by construction: a stamp whose two PNGs are both present is
    skipped without opening the HDF5 at all, and every write is atomic
    (temp file → ``os.replace``), so an interrupted run leaves only whole
    files behind. A pair that is half there — the run was killed between
    the two writes — is re-rendered rather than counted as done.

    Because a frame is named by its stamp alone (that IS the dedup), a
    resumed run must use the same ``downsample_factor`` as the run it
    resumes; a different factor is a different bundle, not a continuation.

    A composite that is absent or unreadable is recorded in ``missing`` and
    the run continues. A 7 GB archive with a handful of gaps (and DMI does
    have gaps) must not lose an eight-hour render, and a missing frame is
    honest information the manifest carries into the UI. A composite on a
    different geometry is treated the same way: the bundle publishes ONE
    grid block, so a frame on another grid would place rain kilometres from
    where it fell.

    ``on_frame(stamp, bytes)`` is called after each rendered frame for
    callers that want their own progress bar; the module also logs a
    running total every ``log_every`` frames and the byte total at the end,
    so a run can be compared against the ~250 MB budget from the log alone.
    """
    if downsample_factor < 1:
        raise ValueError(f"downsample_factor must be >= 1, got {downsample_factor}")

    frames_dir = Path(bundle_dir) / FRAMES_DIRNAME
    frames_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = Path(corpus_dir)

    started = time.monotonic()
    written: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []
    reasons: dict[str, str] = {}
    bytes_written = 0
    bytes_total = 0
    geo: CompositeGeo | None = None
    product_shape: tuple[int, int] | None = None

    for i, planned in enumerate(plan.stamps, start=1):
        overlay_path = frames_dir / planned.overlay_name
        observed_path = frames_dir / planned.observed_name
        if overlay_path.is_file() and observed_path.is_file():
            skipped.append(planned.stamp)
            bytes_total += overlay_path.stat().st_size + observed_path.stat().st_size
            continue

        source = archive_path_for(corpus_dir, planned.filename)
        if not source.is_file():
            # The ordinary case — DMI's archive has gaps — and worth
            # distinguishing from a file that is there but broken, which is
            # a corpus problem somebody should go and look at.
            missing.append(planned.stamp)
            reasons[planned.stamp] = "absent from the corpus archive"
            continue
        try:
            composite = parse_composite(source)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            missing.append(planned.stamp)
            reasons[planned.stamp] = _short_reason("unreadable", exc)
            _log.warning(
                "review_frames.unreadable", stamp=planned.stamp, error=str(exc)[:200],
            )
            continue

        if geo is None:
            geo = CompositeGeo(composite)
        elif not _same_geometry(geo.composite, composite):
            missing.append(planned.stamp)
            reasons[planned.stamp] = "geometry differs from the bundle's grid"
            _log.warning(
                "review_frames.geometry_mismatch",
                stamp=planned.stamp,
                expected=geo.composite.reflectivity_dbz.shape,
                got=composite.reflectivity_dbz.shape,
            )
            continue

        field = _observed_field(composite, downsample_factor=downsample_factor)
        if product_shape is None:
            product_shape = (int(field.shape[0]), int(field.shape[1]))

        overlay_bytes = _encode_rgba_png(_apply_colormap(field))
        observed_bytes = _encode_gray_png(
            quantise(field, QUANT_SPECS[OBSERVED_QUANT_KEY])
        )
        # Grayscale first: the overlay is what a reader checks for
        # existence, so writing it last means "overlay present" never
        # promises a sampler grid that isn't there yet.
        _atomic_write_bytes(observed_path, observed_bytes)
        _atomic_write_bytes(overlay_path, overlay_bytes)

        frame_bytes = len(overlay_bytes) + len(observed_bytes)
        written.append(planned.stamp)
        bytes_written += frame_bytes
        bytes_total += frame_bytes
        if on_frame is not None:
            on_frame(planned.stamp, frame_bytes)
        if log_every and i % log_every == 0:
            _log.info(
                "review_frames.progress",
                done=i,
                of=len(plan.stamps),
                written=len(written),
                skipped=len(skipped),
                missing=len(missing),
                mb_total=round(bytes_total / 1e6, 1),
            )

    if geo is None:
        # Everything was skipped (a resumed run) or missing: the grid block
        # still has to be published, so parse one composite for its
        # geometry alone.
        geo, product_shape = _geometry_from_any(
            plan, corpus_dir=corpus_dir, downsample_factor=downsample_factor,
        )

    grid = (
        grid_block(geo, downsample_factor=downsample_factor, shape=product_shape)
        if geo is not None and product_shape is not None
        else {}
    )
    result = FrameWriteResult(
        frames_dir=frames_dir,
        written=tuple(written),
        skipped=tuple(skipped),
        missing=tuple(missing),
        missing_reasons=reasons,
        bytes_written=bytes_written,
        bytes_total=bytes_total,
        grid=grid,
        encodings=encoding_metadata(downsample_factor=downsample_factor),
        downsample_factor=int(downsample_factor),
        elapsed_s=time.monotonic() - started,
    )
    _log.info(
        "review_frames.done",
        stamps=len(plan.stamps),
        written=len(written),
        skipped=len(skipped),
        missing=len(missing),
        mb_total=round(bytes_total / 1e6, 1),
        mb_written=round(bytes_written / 1e6, 1),
        kb_per_frame=round(result.mean_bytes_per_frame / 1e3, 1),
        elapsed_s=round(result.elapsed_s, 1),
    )
    return result


def _observed_field(
    composite: RadarComposite, *, downsample_factor: int,
) -> np.ndarray:
    """Composite → observed rain rate on the product grid (mm/h).

    Z–R with the file's OWN ``zr-a`` / ``zr-b`` (``parse_composite`` reads
    them from ``/how``; the plan's 200/1.6 is a default, not a constant),
    the 53 dBZ / 100 mm/h caps from ``transform``, then the same block-wise
    p90 reduction the live ``observed_mm_h`` product uses — so a pixel in
    the bundle means what the same pixel means on the website.

    The three-way distinction from the HDF5 survives all of it: ``nodata``
    is NaN in and NaN out (level 255, "we did not measure here"),
    ``undetect`` is −inf in and 0.0 mm/h out ("we measured, nothing fell").
    """
    rain = dbz_to_rain_rate(
        composite.reflectivity_dbz, zr_a=composite.zr_a, zr_b=composite.zr_b,
    )
    return observed_rain_grid(rain, downsample_factor=downsample_factor)


def _short_reason(label: str, exc: Exception) -> str:
    """A one-line reason for the manifest. Truncated on purpose: h5py's
    errors embed the absolute path, and the manifest is copied to a laptop
    and read by a human, not a stack-trace parser. The full text is logged."""
    detail = " ".join(str(exc).split())[:120]
    return f"{label}: {type(exc).__name__}: {detail}"


def _same_geometry(a: RadarComposite, b: RadarComposite) -> bool:
    return (
        a.reflectivity_dbz.shape == b.reflectivity_dbz.shape
        and a.projection == b.projection
        and a.xscale_m == b.xscale_m
        and a.yscale_m == b.yscale_m
        and a.corners_lonlat == b.corners_lonlat
    )


def _geometry_from_any(
    plan: FramePlan, *, corpus_dir: Path, downsample_factor: int,
) -> tuple[CompositeGeo | None, tuple[int, int] | None]:
    """Geometry from the first readable composite in the plan, or ``None``."""
    for planned in plan.stamps:
        try:
            composite = parse_composite(archive_path_for(corpus_dir, planned.filename))
        except (OSError, KeyError, ValueError, RuntimeError):
            continue
        shape = composite.reflectivity_dbz[::downsample_factor, ::downsample_factor]
        return CompositeGeo(composite), (int(shape.shape[0]), int(shape.shape[1]))
    return None, None


def grid_block(
    geo: CompositeGeo, *, downsample_factor: int, shape: tuple[int, int],
) -> dict:
    """The manifest's ``grid`` block — byte-identical in shape to A2's.

    Built the way ``national_artifacts._build_manifest`` builds it: the ×f
    reduction is stride slicing, so the grid keeps the composite's native UL
    projection corner and simply has ``native × f`` pixel scales. Browser
    inverse: ``col = (x - x_ul_m) / pixel_scale_x_m``,
    ``row = (y_ul_m - y) / pixel_scale_y_m``.
    """
    composite = geo.composite
    x_ul, y_ul = geo.projection_origin_m
    block = {
        "proj4": composite.projection,
        "x_ul_m": x_ul,
        "y_ul_m": y_ul,
        "pixel_scale_x_m": float(composite.xscale_m) * downsample_factor,
        "pixel_scale_y_m": float(composite.yscale_m) * downsample_factor,
        "shape": [int(shape[0]), int(shape[1])],
        "downsample_factor": int(downsample_factor),
    }
    assert tuple(block) == GRID_BLOCK_FIELDS, "grid block drifted from the schema"
    return block


def encoding_metadata(*, downsample_factor: int) -> dict:
    """How to read the two PNGs — enough that the browser needs no constants.

    The grayscale entry carries the exact ``scale`` / ``offset`` / ``nodata``
    the sampler inverts (``value = level * scale + offset``, 255 → null) and
    names the reduction, because "2 km pixel" and "p90 of the 16 native
    pixels under it" are different claims and only the second is true.

    The overlay entry names the colormap and the alpha ramp rather than
    implying a legend can be reconstructed from the image: light rain is
    deliberately faded (the composite is column-max and over-reads faint
    echo), so a reviewer reading opacity as intensity would mis-rank every
    drizzle event. Colours are advisory; the grayscale grid is the number.
    """
    spec = QUANT_SPECS[OBSERVED_QUANT_KEY]
    return {
        "observed": {
            "suffix": OBSERVED_SUFFIX,
            "encoding": "grayscale8",
            "product": OBSERVED_QUANT_KEY,
            "units": "mm/h",
            "scale": spec.scale,
            "offset": spec.offset,
            "nodata": NODATA_LEVEL,
            "reduction": (
                f"block p90 over {downsample_factor}x{downsample_factor} native "
                "500 m pixels (national.observed_rain_grid); a block is nodata "
                "only when every native pixel in it is"
            ),
            "source": (
                "DMI column-max composite, Marshall-Palmer Z-R with the file's "
                "own zr-a/zr-b, capped at 53 dBZ and 100 mm/h "
                "(transform.dbz_to_rain_rate)"
            ),
            "caveat": (
                "column-max reflectivity biases the rate HIGH (tall cores, "
                "virga, bright band) — an upper-bound proxy for surface rain"
            ),
        },
        "overlay": {
            "suffix": OVERLAY_SUFFIX,
            "encoding": "rgba8",
            "colormap": "dmi_nowcast_core.render._COLORMAP_STOPS",
            "colormap_stops": [
                {
                    "mm_h": float(row[0]),
                    "rgb": [int(row[1]), int(row[2]), int(row[3])],
                }
                for row in _COLORMAP_STOPS
            ],
            "interpolation": "linear in log10(mm/h) per channel",
            "alpha_ramp": (
                f"transparent below {_RENDER_FLOOR_MM_H} mm/h; alpha "
                f"{int(_RENDER_MIN_ALPHA)} -> 255 linear in log10(mm/h) from "
                f"{_RENDER_FLOOR_MM_H} to {_RENDER_SOLID_MM_H} mm/h; opaque above"
            ),
            "floor_mm_h": _RENDER_FLOOR_MM_H,
            "solid_mm_h": _RENDER_SOLID_MM_H,
            "min_alpha": int(_RENDER_MIN_ALPHA),
        },
    }


def frames_manifest_block(plan: FramePlan, result: FrameWriteResult) -> dict:
    """The manifest's ``frames`` block, for the builder to embed as-is.

    The bundle's ``grid`` block is NOT in here: it is a top-level manifest
    key (:data:`~dmi_nowcast_core.review_schema.GRID_BLOCK_FIELDS`), and the
    builder takes it from ``result.grid``.

    Both window edges travel: ``from_utc`` / ``to_utc`` are the FRAME edges,
    with ``window_min`` and ``frame_pad_min`` alongside so the decision
    edges stay derivable and visibly different.
    """
    return {
        "products": list(plan.products),
        "include_doppler": plan.include_doppler,
        "cadence_min": plan.cadence_min,
        "window_min": plan.window_min,
        "frame_pad_min": plan.pad_min,
        "from_utc": (
            plan.frames_from_utc.isoformat() if plan.frames_from_utc else None
        ),
        "to_utc": plan.frames_to_utc.isoformat() if plan.frames_to_utc else None,
        "count": result.n_present,
        "stamps_total": plan.stamps_total,
        "stamps_unique": plan.stamps_unique,
        "dedup_saving_pct": round(plan.dedup_saving_pct, 1),
        "bytes": result.bytes_total,
        "bytes_written_this_run": result.bytes_written,
        "mean_bytes_per_frame": round(result.mean_bytes_per_frame, 1),
        "downsample_factor": result.downsample_factor,
        "missing": list(result.missing),
        "missing_reasons": dict(result.missing_reasons),
        "encodings": result.encodings,
    }


# ---------------------------------------------------------------------------
# Synthetic fixtures — used by the tests AND by the builder's --fixture mode
# ---------------------------------------------------------------------------

#: Odense, the home point the rest of the project uses for synthetic grids.
FIXTURE_CENTRE_LONLAT = (10.32, 55.33)
FIXTURE_PROJ = (
    "+proj=stere +lat_0=56 +lon_0=10.5666 +lat_ts=56 +ellps=WGS84 +units=m +no_defs"
)
FIXTURE_PIXEL_M = 500.0
FIXTURE_GRID_PX = 64
FIXTURE_GAIN, FIXTURE_OFFSET = 0.5, -32.0
FIXTURE_NODATA_RAW, FIXTURE_UNDETECT_RAW = 255, 0


def write_fixture_composite(
    path: Path,
    ts_utc: datetime,
    *,
    dbz: float | np.ndarray = 30.0,
    nodata_mask: np.ndarray | None = None,
    undetect_mask: np.ndarray | None = None,
    shape: tuple[int, int] = (FIXTURE_GRID_PX, FIXTURE_GRID_PX),
    pixel_m: float = FIXTURE_PIXEL_M,
    centre_lonlat: tuple[float, float] = FIXTURE_CENTRE_LONLAT,
) -> Path:
    """Write a minimal DMI-style ODIM composite: the fixture, not a mock.

    It is a real HDF5 in DMI's own dialect — scaling and ``product`` in the
    ROOT ``/what`` (see ``parse.py``), ``zr-a`` / ``zr-b`` in ``/how`` — so
    everything downstream runs the production code path, including the
    scaling read that ``CLAUDE.md`` forbids hardcoding. A mock composite
    would let exactly that contract rot undetected.

    ``dbz`` may be a scalar or a full array. ``nodata_mask`` marks pixels the
    radar did not measure and ``undetect_mask`` pixels it measured as below
    detection: the two are encoded as DIFFERENT raw values, because the
    whole review tool exists to tell "no data" from "no rain".
    """
    rows, cols = shape
    if np.ndim(dbz) == 0:
        field = np.full((rows, cols), float(dbz), dtype=np.float64)
    else:
        field = np.asarray(dbz, dtype=np.float64).reshape(rows, cols)
    raw = np.rint((field - FIXTURE_OFFSET) / FIXTURE_GAIN).astype(np.int32)
    # Clipped clear of both sentinels: a legitimate dBZ value that happened
    # to land on the nodata or undetect raw level would come back out of
    # the parser as "missing" and quietly poison the fixture.
    raw = np.clip(raw, FIXTURE_UNDETECT_RAW + 1, FIXTURE_NODATA_RAW - 1)
    if undetect_mask is not None:
        raw[np.asarray(undetect_mask, dtype=bool)] = FIXTURE_UNDETECT_RAW
    if nodata_mask is not None:
        raw[np.asarray(nodata_mask, dtype=bool)] = FIXTURE_NODATA_RAW
    raw = raw.astype(np.uint8)

    ts = _as_utc(ts_utc, "fixture timestamp")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    corners = _fixture_corners(shape=shape, pixel_m=pixel_m, centre=centre_lonlat)
    with h5py.File(path, "w") as h5:
        what = h5.create_group("what")
        what.attrs["gain"] = FIXTURE_GAIN
        what.attrs["offset"] = FIXTURE_OFFSET
        what.attrs["nodata"] = FIXTURE_NODATA_RAW
        what.attrs["undetect"] = FIXTURE_UNDETECT_RAW
        what.attrs["date"] = ts.strftime("%Y%m%d").encode()
        what.attrs["time"] = ts.strftime("%H%M%S").encode()
        what.attrs["product"] = b"DBZH"
        where = h5.create_group("where")
        where.attrs["projdef"] = FIXTURE_PROJ.encode()
        where.attrs["xscale"] = pixel_m
        where.attrs["yscale"] = pixel_m
        for name, (lon, lat) in corners.items():
            where.attrs[f"{name}_lon"] = lon
            where.attrs[f"{name}_lat"] = lat
        how = h5.create_group("how")
        how.attrs["zr-a"] = 200.0
        how.attrs["zr-b"] = 1.6
        h5.create_group("dataset1").create_group("data1").create_dataset(
            "data", data=raw,
        )
    return path


def write_fixture_archive(
    corpus_dir: Path,
    stamps: Sequence[str] | Sequence[datetime],
    **composite_kwargs,
) -> list[Path]:
    """Write synthetic composites into a corpus-shaped ``YYYY/MM`` tree.

    Paths come from :func:`~dmi_nowcast_core.corpus.archive_path_for`, so
    ``write_frames`` finds them exactly as it finds the real 7 GB archive —
    the ``--fixture`` path exercises the same lookup, not a stub.
    """
    out: list[Path] = []
    for stamp in stamps:
        ts = stamp_to_datetime(stamp) if isinstance(stamp, str) else _as_utc(
            stamp, "fixture stamp",
        )
        path = archive_path_for(Path(corpus_dir), composite_filename(ts))
        out.append(write_fixture_composite(path, ts, **composite_kwargs))
    return out


def _fixture_corners(
    *, shape: tuple[int, int], pixel_m: float, centre: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    """Corner lon/lats placing ``centre`` at the grid's centre."""
    rows, cols = shape
    crs = CRS.from_proj4(FIXTURE_PROJ)
    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x_c, y_c = to_proj.transform(centre[0], centre[1])
    half_x = cols / 2 * pixel_m
    half_y = rows / 2 * pixel_m
    return {
        "UL": to_wgs.transform(x_c - half_x, y_c + half_y),
        "UR": to_wgs.transform(x_c + half_x, y_c + half_y),
        "LL": to_wgs.transform(x_c - half_x, y_c - half_y),
        "LR": to_wgs.transform(x_c + half_x, y_c - half_y),
    }
