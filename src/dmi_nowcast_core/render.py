"""Diagnostic radar overlay rendering using Pillow.

Used by:
- ``custom_components/dmi_rain_incoming/image.py`` (HA image entity)
- ``scripts/phase1_render.py`` (visual verification)
- backtest galleries (plan §10.2)

Output is a PNG ``bytes`` object. Pillow is loaded lazily so this module can
be imported without it; ``render_overlay`` and ``render_loop_png`` raise
``RenderUnavailable`` if Pillow is missing.

Why Pillow rather than matplotlib:
- matplotlib has no wheels for musllinux Python 3.14 (HA OS), so it cannot be
  installed there;
- our overlay needs only a colormap, two circles, and a text label — Pillow
  handles that with a fraction of matplotlib's footprint;
- import time is ~10× faster, which matters at the polling cadence.

``render_loop_png`` emits an animated PNG (APNG) covering past observations
plus Lagrangian forecast frames. Modern browsers play APNG natively, so the
HA image entity needs no special handling — the same ``image/png`` content
type is served and the dashboard tile animates.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .geo import CompositeGeo
from .parse import RadarComposite


class RenderUnavailable(RuntimeError):
    """Raised when Pillow is not installed.

    Caught by ``image.py`` so the image entity returns no payload when
    Pillow isn't available.
    """


@dataclass(frozen=True)
class OverlayInputs:
    """Everything ``render_overlay`` needs apart from Pillow."""

    composite: RadarComposite
    rain_rate_mm_h: np.ndarray
    geo: CompositeGeo
    home_lat: float
    home_lon: float
    radius_km: float
    # Optional annotations
    disc_max_mm_h: float | None = None
    eta_p50_min: float | None = None
    probability_30min: float | None = None
    confidence: float | None = None
    # Disc-area motion (radar-grid pixels per minute) — drives the
    # "rain coming from" arrow. None or (0, 0) suppresses the arrow.
    motion_dy_per_min: float | None = None
    motion_dx_per_min: float | None = None
    # Same motion as a human caption. Drawn in the top-right of the
    # overlay regardless of whether the arrow itself was rendered.
    motion_speed_kmh: float | None = None
    motion_bearing_from: str | None = None


@dataclass(frozen=True)
class LoopFrame:
    """One frame of an animated radar loop.

    ``timestamp_utc`` is the time the field represents (past for observed,
    future for forecast). ``label`` is a short human caption such as
    ``"−10 min"`` / ``"now"`` / ``"+15 min"``. ``kind`` drives the band
    colour at the top of the frame (gray for observed, amber for forecast).
    """

    rain_rate_mm_h: np.ndarray
    timestamp_utc: datetime
    label: str
    kind: str  # "observed" | "forecast"
    duration_ms: int


# Simple radar-style colormap: control points in (mm/h, R, G, B).
# Matches the spirit of DMI's own radar viewer — blue for light, green/yellow
# for moderate, red/magenta for heavy.
# Major Danish cities for visual reference on the overlay. Lon, lat.
_LANDMARKS: tuple[tuple[str, float, float], ...] = (
    ("Copenhagen", 12.5645, 55.6726),
    ("Aarhus",     10.2107, 56.1572),
    ("Odense",     10.3886, 55.4038),
    ("Aalborg",     9.9217, 57.0488),
    ("Esbjerg",     8.4500, 55.4667),
    ("Kolding",     9.4920, 55.4904),
    ("Vejle",       9.5350, 55.7058),
    ("Roskilde",   12.0833, 55.6500),
    ("Slagelse",   11.3500, 55.4000),
    ("Svendborg",  10.6100, 55.0600),
)


_COLORMAP_STOPS = np.array([
    [0.05,   0,   0, 255],   # very light: deep blue
    [0.20,   0, 150, 255],   # light:      sky blue
    [0.50,   0, 220, 220],   # moderate:   cyan
    [1.00,   0, 200,  60],   # noticeable: green
    [2.50, 220, 220,   0],   # firm:       yellow
    [5.00, 255, 140,   0],   # heavy:      orange
    [15.0, 230,  30,  30],   # very heavy: red
    [50.0, 200,  20, 220],   # extreme:    magenta
], dtype=np.float32)

# Rendering opacity: transparent below the floor, a log-space alpha ramp from
# _RENDER_MIN_ALPHA at the floor to fully opaque at _RENDER_SOLID_MM_H. This fades
# very light / clutter / virga echoes — the DMI composite is column-max and
# over-reads faint returns — so the overlay reads like DMI's QC'd display instead
# of a solid blue wash. (The rain-at-home *detection* threshold lives separately
# in the sidecar config, default p90 @ 0.5 mm/h.)
_RENDER_FLOOR_MM_H = 0.3
_RENDER_SOLID_MM_H = 2.0
_RENDER_MIN_ALPHA = 70.0


# Top band colours (RGBA) signalling whether the frame is an observation
# or a Lagrangian forecast.
_BAND_COLOURS = {
    "observed": (90, 90, 90, 230),
    "forecast": (220, 140, 0, 230),
}
_BAND_HEIGHT_PX = 4


# F1/Ubimet-style distance reference: rings at these distances (km) around
# home, plus a horizontal+vertical crosshair through home and labels on the
# left-of-home horizontal axis. 10 km spacing for the close-in rings is
# dense enough to give a quick "X km away" read without crowding the image.
# 100 km sits at the cardinal edges of the 100-km-zoom crop, framing the
# visible area; corners go out to ~141 km.
_DEFAULT_RING_RADII_KM: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0)
# Only every-other ring gets labelled to avoid text collisions on the W
# axis. At 100 km zoom with 600 px output, ~3 px/km, so 20 km between
# labels = ~60 px — comfortable for 28-px-wide "60 km" text.
_LABELLED_RING_RADII_KM: tuple[float, ...] = (20.0, 40.0, 60.0, 80.0, 100.0)
_RING_COLOUR_RGBA = (80, 80, 80, 153)  # ~60 % opaque dark gray
_CROSSHAIR_COLOUR_RGBA = (80, 80, 80, 120)


def _apply_colormap(rain_mm_h: np.ndarray) -> np.ndarray:
    """Map rain rate (mm/h) → RGBA uint8. Transparent below ``_RENDER_FLOOR_MM_H``;
    a log-space alpha ramp fades light rain so the overlay isn't a solid wash."""
    stops = _COLORMAP_STOPS
    log_levels = np.log10(stops[:, 0])
    rgb_at_levels = stops[:, 1:4]
    finite = np.isfinite(rain_mm_h) & (rain_mm_h > 0)
    log_r = np.where(finite, np.log10(np.maximum(rain_mm_h, 1e-3)), log_levels[0] - 1)
    # Linear interpolate R, G, B independently in log-space
    rgba = np.zeros((*rain_mm_h.shape, 4), dtype=np.float32)
    for ch in range(3):
        rgba[..., ch] = np.interp(log_r, log_levels, rgb_at_levels[:, ch])
    # Alpha: 0 below the floor; ramps _RENDER_MIN_ALPHA → 255 (in log-rate) from
    # the floor to _RENDER_SOLID_MM_H; opaque above.
    lf, ls = float(np.log10(_RENDER_FLOOR_MM_H)), float(np.log10(_RENDER_SOLID_MM_H))
    ramp = np.clip((log_r - lf) / (ls - lf), 0.0, 1.0)
    alpha = _RENDER_MIN_ALPHA + ramp * (255.0 - _RENDER_MIN_ALPHA)
    visible = finite & (rain_mm_h >= _RENDER_FLOOR_MM_H)
    rgba[..., 3] = np.where(visible, alpha, 0.0)
    return rgba.astype(np.uint8)


def _draw_distance_rings(
    draw,
    *,
    home_x: float,
    home_y: float,
    pixel_scale_m: float,
    scale: float,
    image_w: int,
    image_h: int,
    radii_km: tuple[float, ...] = _DEFAULT_RING_RADII_KM,
    colour=_RING_COLOUR_RGBA,
    width: int = 1,
) -> None:
    """Concentric distance circles centred on home.

    Drawn semi-transparent so the underlying radar field still reads. Rings
    that fall entirely outside the visible frame are skipped (keeps the
    100 km ring from being drawn when home sits near the edge of the crop).
    """
    for r_km in radii_km:
        r_px = (r_km * 1000.0) / pixel_scale_m * scale
        if (home_x + r_px < 0 or home_x - r_px > image_w
                or home_y + r_px < 0 or home_y - r_px > image_h):
            continue
        draw.ellipse(
            (home_x - r_px, home_y - r_px, home_x + r_px, home_y + r_px),
            outline=colour, width=width,
        )


def _draw_crosshair(
    draw,
    *,
    home_x: float,
    home_y: float,
    image_w: int,
    image_h: int,
    colour=_CROSSHAIR_COLOUR_RGBA,
    width: int = 1,
) -> None:
    """Horizontal + vertical lines through home, spanning the full image.

    Together with the rings this gives the F1/Ubimet "distance from venue"
    look; without the crosshair the rings are visually adrift.
    """
    draw.line([(0, home_y), (image_w, home_y)], fill=colour, width=width)
    draw.line([(home_x, 0), (home_x, image_h)], fill=colour, width=width)


def _draw_motion_arrow(
    draw,
    *,
    home_x: float,
    home_y: float,
    motion_dy_per_min: float,
    motion_dx_per_min: float,
    pixel_scale_m: float,
    scale: float,
    image_w: int,
    image_h: int,
    minute_ticks: tuple[int, ...] = (10, 20, 30, 40, 50, 60),
    arrow_colour=(220, 30, 30),
    halo_colour=(255, 255, 255),
    line_width: int = 3,
    font=None,
    min_60min_km: float = 20.0,
) -> bool:
    """FIA/Ubimet-style "rain coming from" arrow.

    Origin is at home. Direction is the **reverse** of the motion vector,
    so the arrow points toward where the rain currently sits — i.e., toward
    where the rain about to hit home is coming *from*.

    Tick marks at +10/+20/.../+60 min along the line indicate where on
    that trajectory the rain was 10/20/.../60 minutes ago (equivalently:
    when the cell currently at that tick will reach home, if motion holds).

    Returns ``True`` if an arrow was drawn, ``False`` if motion was below
    ``min_60min_km`` (which corresponds to a 60-minute projection shorter
    than 20 km by default, i.e. wind speeds under ~20 km/h). Below that
    the arrow either disappears under the home dot or visibly bunches its
    tick labels — the top-right caption keeps the actual speed/bearing
    readable in either case.
    """
    import math

    # Motion magnitude in km per minute, derived from pixels-per-minute.
    speed_km_per_min = math.hypot(
        motion_dy_per_min * pixel_scale_m / 1000.0,
        motion_dx_per_min * pixel_scale_m / 1000.0,
    )
    if speed_km_per_min * 60.0 < min_60min_km:
        return False

    # Image-pixel offsets per minute (reverse of motion). Image y grows down
    # in the same direction as radar grid rows, so we negate both axes to
    # point from home back along the inverse of the motion vector.
    img_dy_per_min = -motion_dy_per_min * scale
    img_dx_per_min = -motion_dx_per_min * scale

    # Don't draw past the image edge — clip ticks beyond the visible area.
    visible_ticks: list[int] = []
    for m in minute_ticks:
        tx = home_x + img_dx_per_min * m
        ty = home_y + img_dy_per_min * m
        if 0 <= tx <= image_w and 0 <= ty <= image_h:
            visible_ticks.append(m)
    if not visible_ticks:
        # Arrow heads off the visible frame even at +10 min — silly to draw.
        return False
    tip_min = visible_ticks[-1]
    tip_x = home_x + img_dx_per_min * tip_min
    tip_y = home_y + img_dy_per_min * tip_min

    # Main line — drawn with a white halo so it stays readable over both
    # land and rain pixels. Halo is a slightly thicker white line painted
    # first; the red line goes on top.
    draw.line([(home_x, home_y), (tip_x, tip_y)],
              fill=halo_colour + (220,), width=line_width + 2)
    draw.line([(home_x, home_y), (tip_x, tip_y)],
              fill=arrow_colour + (255,), width=line_width)

    # Arrowhead: small triangle at the tip pointing along the line.
    head_size = max(8.0, line_width * 3.0)
    angle = math.atan2(img_dy_per_min, img_dx_per_min)
    # Two base points 150° / -150° from the tip's direction of travel.
    base_a = (
        tip_x - head_size * math.cos(angle - math.radians(25)),
        tip_y - head_size * math.sin(angle - math.radians(25)),
    )
    base_b = (
        tip_x - head_size * math.cos(angle + math.radians(25)),
        tip_y - head_size * math.sin(angle + math.radians(25)),
    )
    # White halo arrowhead first, then red on top.
    draw.polygon([(tip_x, tip_y), base_a, base_b],
                 fill=halo_colour + (220,), outline=halo_colour + (220,))
    draw.polygon([(tip_x, tip_y), base_a, base_b],
                 fill=arrow_colour + (255,), outline=arrow_colour + (255,))

    # Tick marks + minute labels at each visible tick.
    font = font or _load_font(10)
    tick_half = max(4.0, line_width * 1.5)
    # Unit perpendicular vector to the arrow direction (for ticks).
    px = -math.sin(angle)
    py = math.cos(angle)
    for m in visible_ticks:
        tx = home_x + img_dx_per_min * m
        ty = home_y + img_dy_per_min * m
        tk_a = (tx + tick_half * px, ty + tick_half * py)
        tk_b = (tx - tick_half * px, ty - tick_half * py)
        # Halo then arrow colour
        draw.line([tk_a, tk_b], fill=halo_colour + (220,), width=line_width + 1)
        draw.line([tk_a, tk_b], fill=arrow_colour + (255,), width=line_width - 1)
        # Label offset slightly along the perpendicular, on the same side
        # for every tick so they read left-to-right (or top-to-bottom).
        lbl_x = tx + 1.6 * tick_half * px
        lbl_y = ty + 1.6 * tick_half * py - 6
        _text_with_shadow(draw, (lbl_x, lbl_y), str(m), font, fill=arrow_colour)
    return True


def _draw_axis_labels(
    draw,
    *,
    home_x: float,
    home_y: float,
    pixel_scale_m: float,
    scale: float,
    image_w: int,
    image_h: int,
    radii_km: tuple[float, ...] = _LABELLED_RING_RADII_KM,
    font=None,
) -> None:
    """Label each ring on the west-of-home horizontal axis, FIA-style.

    Placed just below the horizontal crosshair so labels don't overlap the
    line itself. Skipped if the label would render outside the image.
    """
    font = font or _load_font(11)
    for r_km in radii_km:
        r_px = (r_km * 1000.0) / pixel_scale_m * scale
        lx = home_x - r_px
        ly = home_y + 4
        # Tight rejection: 6 px of left margin, 16 px below.
        if not (6 <= lx <= image_w - 40 and 0 <= ly <= image_h - 16):
            continue
        _text_with_shadow(draw, (lx + 2, ly), f"{int(r_km)} km", font, fill=(40, 40, 40))


def _render_frame(
    *,
    rain_rate_mm_h: np.ndarray,
    composite: RadarComposite,
    geo: CompositeGeo,
    home_lat: float,
    home_lon: float,
    radius_km: float,
    zoom_km: float,
    output_px: int,
    header: str,
    subline: str | None,
    band_kind: str | None,
    show_rings: bool = True,
    motion_dy_per_min: float | None = None,
    motion_dx_per_min: float | None = None,
    motion_speed_kmh: float | None = None,
    motion_bearing_from: str | None = None,
    basemap=None,  # PIL.Image | None — OSM basemap underlay
):
    """Render one frame and return a PIL Image (RGB).

    Shared by ``render_overlay`` (single frame) and ``render_loop_png`` (many).
    ``header`` is the top text line; ``subline`` is the optional second line
    of stats below it. ``band_kind`` is ``"observed"`` / ``"forecast"`` /
    ``None`` — a coloured stripe at the top edge that signals whether the
    field is an observation or a Lagrangian forecast.

    ``show_rings`` toggles the F1/Ubimet-style distance rings + crosshair +
    axis labels. On by default; tests can opt out for the bare overlay.
    """
    from PIL import Image, ImageDraw  # local import: Pillow availability check

    h, w = rain_rate_mm_h.shape
    idx = geo.lonlat_to_grid(home_lon, home_lat)
    pixel_scale_m = composite.xscale_m
    zoom_px = (zoom_km * 1000.0) / pixel_scale_m
    radius_px = (radius_km * 1000.0) / pixel_scale_m

    # Crop to a square around home, clipped to the grid.
    half = int(round(zoom_px))
    col_c = int(round(idx.col))
    row_c = int(round(idx.row))
    c0 = max(0, col_c - half)
    c1 = min(w, col_c + half)
    r0 = max(0, row_c - half)
    r1 = min(h, row_c + half)
    crop = rain_rate_mm_h[r0:r1, c0:c1]

    rgba = _apply_colormap(crop)
    img = Image.fromarray(rgba, "RGBA")

    # Upscale to ``output_px`` so the home marker is visible.
    crop_h, crop_w = crop.shape
    scale = output_px / max(crop_h, crop_w)
    new_w = int(round(crop_w * scale))
    new_h = int(round(crop_h * scale))
    img = img.resize((new_w, new_h), Image.NEAREST)

    # Bottom layer: OSM basemap if provided, otherwise solid light gray.
    # Basemap is RGBA at (new_w, new_h) from build_basemap; we slightly
    # desaturate by overlaying a 25 % white veil so the radar colours read
    # clearly on top without making the map disappear.
    if basemap is not None:
        bg = basemap.resize((new_w, new_h), Image.LANCZOS).convert("RGBA")
        veil = Image.new("RGBA", (new_w, new_h), (255, 255, 255, 64))
        bg.alpha_composite(veil)
    else:
        bg = Image.new("RGBA", (new_w, new_h), (245, 245, 245, 255))
    bg.alpha_composite(img)
    img = bg

    draw = ImageDraw.Draw(img)

    # Distance rings + crosshair + axis labels — F1/Ubimet style. Drawn
    # before the city markers so cities overlay the rings (a city dot
    # sitting on a faint ring should still be readable).
    home_x = (idx.col - c0) * scale
    home_y = (idx.row - r0) * scale
    small = _load_font(11)
    if show_rings:
        _draw_crosshair(draw, home_x=home_x, home_y=home_y,
                        image_w=new_w, image_h=new_h)
        _draw_distance_rings(draw, home_x=home_x, home_y=home_y,
                             pixel_scale_m=pixel_scale_m, scale=scale,
                             image_w=new_w, image_h=new_h)
        _draw_axis_labels(draw, home_x=home_x, home_y=home_y,
                          pixel_scale_m=pixel_scale_m, scale=scale,
                          image_w=new_w, image_h=new_h, font=small)

    # "Rain coming from" arrow — drawn after the rings (so the arrow line
    # overlays them) but before the city markers and home dot (so those
    # remain readable on top).
    if motion_dy_per_min is not None and motion_dx_per_min is not None:
        _draw_motion_arrow(
            draw,
            home_x=home_x, home_y=home_y,
            motion_dy_per_min=motion_dy_per_min,
            motion_dx_per_min=motion_dx_per_min,
            pixel_scale_m=pixel_scale_m, scale=scale,
            image_w=new_w, image_h=new_h,
        )

    # City markers for geographic reference. Only draw the ones that fall
    # inside the visible crop after scaling. Suppressed when a basemap is
    # present — OSM already labels Danish cities, doubling up would be
    # noisy.
    for name, lon, lat in (() if basemap is not None else _LANDMARKS):
        lidx = geo.lonlat_to_grid(lon, lat)
        if not (c0 <= lidx.col < c1 and r0 <= lidx.row < r1):
            continue
        lx = (lidx.col - c0) * scale
        ly = (lidx.row - r0) * scale
        draw.ellipse((lx - 3, ly - 3, lx + 3, ly + 3),
                     fill=(255, 255, 255, 200), outline=(0, 0, 0, 255), width=1)
        _text_with_shadow(draw, (lx + 5, ly - 7), name, small, fill=(20, 20, 20))

    # Home marker + disc — home_x/home_y were computed up-front for the rings.
    rpx = radius_px * scale

    # 1-px thick black dashed-ish outer ring (Pillow lacks true dashes), draw
    # two concentric rings to make the disc readable on any background.
    draw.ellipse((home_x - rpx, home_y - rpx, home_x + rpx, home_y + rpx),
                 outline=(255, 255, 255, 255), width=3)
    draw.ellipse((home_x - rpx, home_y - rpx, home_x + rpx, home_y + rpx),
                 outline=(0, 0, 0, 255), width=1)
    # Home dot
    draw.ellipse((home_x - 5, home_y - 5, home_x + 5, home_y + 5),
                 fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=1)

    # Header text
    font = _load_font(14)
    _text_with_shadow(draw, (8, 6), header, font, fill=(20, 20, 20))
    if subline:
        _text_with_shadow(draw, (8, 26), subline, small, fill=(40, 40, 40))

    # Motion caption (top-right): "13 km/h from SW" / "calm". Stays
    # readable even on calm days when the arrow itself is suppressed —
    # the user still sees the actual measurement.
    if motion_speed_kmh is not None:
        caption_font = _load_font(13)
        if motion_speed_kmh < 0.5 or not motion_bearing_from:
            caption = "calm"
        else:
            caption = f"{motion_speed_kmh:.0f} km/h from {motion_bearing_from}"
        cbbox = draw.textbbox((0, 0), caption, font=caption_font)
        cw, ch = cbbox[2] - cbbox[0], cbbox[3] - cbbox[1]
        _text_with_shadow(
            draw, (new_w - cw - 8, 6),
            caption, caption_font, fill=(20, 20, 20),
        )

    # Top-edge band (forecast / observed indicator). Drawn last so it overlays
    # everything else at the very top of the frame.
    if band_kind in _BAND_COLOURS:
        draw.rectangle((0, 0, new_w, _BAND_HEIGHT_PX), fill=_BAND_COLOURS[band_kind])

    # Attribution (plan §16) — OSM contributors are required when the
    # basemap is present per the tile usage policy.
    attribution = (
        "Data: DMI Open Data  ·  Map: © OpenStreetMap contributors"
        if basemap is not None else "Data: DMI Open Data"
    )
    bbox = draw.textbbox((0, 0), attribution, font=small)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    _text_with_shadow(
        draw, (new_w - text_w - 8, new_h - text_h - 8),
        attribution, small, fill=(40, 40, 40),
    )

    return img.convert("RGB")


def render_overlay(
    inputs: OverlayInputs,
    *,
    zoom_km: float = 100.0,
    output_px: int = 600,
    basemap=None,  # PIL.Image | None
) -> bytes:
    """Render the radar field with a home marker + disc and return PNG bytes."""
    try:
        from PIL import Image  # noqa: F401 — availability probe
    except ImportError as exc:
        raise RenderUnavailable(
            "Pillow is not installed; render_overlay needs it."
        ) from exc

    composite = inputs.composite
    header = f"DMI radar  {composite.timestamp_utc.strftime('%Y-%m-%d %H:%M UTC')}  ({composite.quantity})"
    stats: list[str] = []
    if inputs.disc_max_mm_h is not None and np.isfinite(inputs.disc_max_mm_h):
        stats.append(f"now: {inputs.disc_max_mm_h:.2f} mm/h")
    if inputs.eta_p50_min is not None and np.isfinite(inputs.eta_p50_min):
        stats.append(f"ETA ~{inputs.eta_p50_min:.0f} min")
    if inputs.probability_30min is not None and np.isfinite(inputs.probability_30min):
        stats.append(f"P(30min) {100 * inputs.probability_30min:.0f}%")
    if inputs.confidence is not None and np.isfinite(inputs.confidence):
        stats.append(f"conf {100 * inputs.confidence:.0f}%")
    subline = "  •  ".join(stats) if stats else None

    img = _render_frame(
        rain_rate_mm_h=inputs.rain_rate_mm_h,
        composite=composite,
        geo=inputs.geo,
        home_lat=inputs.home_lat, home_lon=inputs.home_lon,
        radius_km=inputs.radius_km,
        zoom_km=zoom_km, output_px=output_px,
        header=header, subline=subline, band_kind=None,
        motion_dy_per_min=inputs.motion_dy_per_min,
        motion_dx_per_min=inputs.motion_dx_per_min,
        motion_speed_kmh=inputs.motion_speed_kmh,
        motion_bearing_from=inputs.motion_bearing_from,
        basemap=basemap,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_loop_png(
    frames: list[LoopFrame],
    *,
    composite: RadarComposite,
    geo: CompositeGeo,
    home_lat: float,
    home_lon: float,
    radius_km: float,
    zoom_km: float = 100.0,
    output_px: int = 600,
    now_stats_subline: str | None = None,
    motion_dy_per_min: float | None = None,
    motion_dx_per_min: float | None = None,
    motion_speed_kmh: float | None = None,
    motion_bearing_from: str | None = None,
    basemap=None,  # PIL.Image | None
    individual_frames_dir: Path | None = None,
) -> bytes:
    """Render a list of frames as an animated PNG (APNG).

    Frame order is the playback order: typically oldest observed → now (held
    longer) → forecast +5 → ... → +30 min (last frame held). Pillow writes
    APNG natively; the resulting bytes can be served as ``image/png`` and
    every modern browser plays the loop on its own.

    ``now_stats_subline`` is shown only on frames labelled ``"now"`` so the
    user sees current disc-max + ETA on the held frame without cluttering
    every observation/forecast frame with the same numbers.

    When ``individual_frames_dir`` is provided, each frame is *also* written
    out as a standalone PNG file in that directory plus a ``frames.json``
    manifest. This is how the custom Lovelace slider card gets its frames —
    the APNG stays the canonical autoplay artifact, but the slider needs
    individually-addressable images so it can scrub without seeking into an
    APNG (which browsers don't expose).
    """
    if not frames:
        raise ValueError("render_loop_png needs at least one frame")
    try:
        from PIL import Image  # noqa: F401 — availability probe
    except ImportError as exc:
        raise RenderUnavailable(
            "Pillow is not installed; render_loop_png needs it."
        ) from exc

    pil_frames = []
    durations: list[int] = []
    for f in frames:
        ts = f.timestamp_utc.strftime("%Y-%m-%d %H:%M UTC")
        suffix = "observed" if f.kind == "observed" else "forecast"
        header = f"{f.label}  {ts}  ({suffix})"
        subline = now_stats_subline if f.label.lower() == "now" else None
        pil_frames.append(_render_frame(
            rain_rate_mm_h=f.rain_rate_mm_h,
            composite=composite,  # geo + scale; same projection across frames
            geo=geo,
            home_lat=home_lat, home_lon=home_lon, radius_km=radius_km,
            zoom_km=zoom_km, output_px=output_px,
            header=header, subline=subline, band_kind=f.kind,
            motion_dy_per_min=motion_dy_per_min,
            motion_dx_per_min=motion_dx_per_min,
            motion_speed_kmh=motion_speed_kmh,
            motion_bearing_from=motion_bearing_from,
            basemap=basemap,
        ))
        durations.append(max(50, int(f.duration_ms)))

    buf = io.BytesIO()
    # APNG dispose/blend, picked deliberately:
    # - ``disposal=0`` (DISPOSE_OP_NONE) keeps each frame's pixels on the
    #   canvas. Without this Pillow's diff-aware encoder produces tiny
    #   sub-rectangles (e.g. just the header text changed) that, under
    #   ``disposal=2`` (DISPOSE_OP_PREVIOUS), render onto a freshly-blanked
    #   canvas and lose the radar field — leading to "mostly white" frames
    #   interleaved with the full ones. Source-over with persistent canvas
    #   keeps the radar visible while sub-rects overwrite just what changed.
    # - ``blend=0`` (BLEND_OP_SOURCE) means each sub-rect's pixels fully
    #   replace the destination (no alpha blending against the prior frame).
    #   The frames are RGB so this is equivalent to OVER for fully-opaque
    #   content, but it avoids surprises if a future frame ever introduces
    #   semi-transparency.
    pil_frames[0].save(
        buf,
        format="PNG",
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
        disposal=0,
        blend=0,
        optimize=True,
    )
    apng_bytes = buf.getvalue()

    if individual_frames_dir is not None:
        _write_individual_frames(
            pil_frames, frames, individual_frames_dir, now_stats_subline,
        )

    return apng_bytes


def _write_individual_frames(
    pil_frames: list,
    metadata: list[LoopFrame],
    out_dir: Path,
    now_stats_subline: str | None,
) -> None:
    """Write per-frame PNGs + a frames.json manifest the slider card reads.

    Atomic per file: writes to ``.tmp`` then ``os.replace``. Stale files
    from a previous longer loop are removed up front so the manifest
    always describes exactly what's on disk.
    """
    import json
    import os

    out_dir.mkdir(parents=True, exist_ok=True)
    # Prune any frame_NN.png that won't be overwritten by this batch.
    for existing in out_dir.glob("frame_*.png"):
        try:
            idx_part = existing.stem.split("_", 1)[1]
            idx = int(idx_part)
            if idx >= len(pil_frames):
                existing.unlink()
        except (ValueError, IndexError):
            continue

    manifest_frames = []
    for i, (pil, meta) in enumerate(zip(pil_frames, metadata)):
        fname = f"frame_{i:02d}.png"
        target = out_dir / fname
        tmp = target.with_suffix(target.suffix + ".tmp")
        pil.save(tmp, format="PNG", optimize=True)
        os.replace(tmp, target)
        manifest_frames.append({
            "index": i,
            "filename": fname,
            "timestamp_utc": meta.timestamp_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kind": meta.kind,
            "label": meta.label,
            "duration_ms": max(50, int(meta.duration_ms)),
        })

    manifest = {
        "version": 1,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frame_count": len(pil_frames),
        "frames": manifest_frames,
        # Used by the slider card to show stats on the "now" frame.
        "now_stats_subline": now_stats_subline,
    }
    manifest_tmp = out_dir / "frames.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(manifest_tmp, out_dir / "frames.json")


def _load_font(size: int):
    """Try a real TTF; fall back to PIL's bitmap default if none are bundled.

    The default bitmap font ignores size — that's fine for an emergency
    fallback; the radar overlay is mainly visual.
    """
    from PIL import ImageFont
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _text_with_shadow(draw, xy, text, font, *, fill):
    """Draw text with a 1px white halo so it's readable on any radar colour."""
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=(255, 255, 255))
    draw.text((x, y), text, font=font, fill=fill)
