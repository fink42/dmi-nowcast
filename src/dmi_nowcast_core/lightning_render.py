"""Render a lightning-nowcast map PNG (Pillow), matching the radar overlay.

Reuses the radar renderer's projection + basemap + ring/crosshair/font helpers
(:mod:`dmi_nowcast_core.render`) so the lightning map overlays the same
home-centred crop pixel-for-pixel. Draws:

- individual strikes as dots, coloured by age (recent = bright, old = faded);
- each cluster as a circle at its centroid (threatening cell in red);
- a forward motion arrow per moving cluster (direction of travel, length ∝ a
  ~20-min projection);
- distance rings + crosshair around home (and the Pixel, if given);
- a header line of stats.

Pure Python + Pillow (lazy import); no HA / FastAPI. Output is PNG ``bytes``.
"""
from __future__ import annotations

import io
import math
from datetime import datetime, timezone

from .geo import CompositeGeo
from .lightning import ClusterSummary, LightningStrike
from .render import (
    RenderUnavailable,
    _draw_crosshair,
    _draw_distance_rings,
    _load_font,
    _text_with_shadow,
)

# Strike dot colours by age bucket (minutes) — bright/recent → faint/old.
_AGE_COLOURS = (
    (5.0, (255, 240, 60)),    # ≤5 min: bright yellow (leading edge)
    (15.0, (255, 150, 0)),    # ≤15 min: amber
    (30.0, (150, 120, 90)),   # ≤30 min: faded brown-grey
)
_AGE_COLOUR_OLD = (110, 110, 110)

_CLUSTER_THREAT_RGBA = (230, 30, 30, 255)
_CLUSTER_OTHER_RGBA = (235, 170, 0, 255)
_ARROW_PROJECT_MIN = 20.0  # arrow length represents this many minutes of motion
_ARROW_MAX_PX = 140.0


def _strike_colour(age_min: float) -> tuple[int, int, int]:
    for cutoff, colour in _AGE_COLOURS:
        if age_min <= cutoff:
            return colour
    return _AGE_COLOUR_OLD


def _draw_vector_arrow(draw, x0, y0, dx, dy, *, colour, halo=(255, 255, 255)):
    """Forward arrow from (x0,y0) along (dx,dy) image px. dx/dy already scaled."""
    length = math.hypot(dx, dy)
    if length < 6.0:
        return  # too short to read
    if length > _ARROW_MAX_PX:
        dx *= _ARROW_MAX_PX / length
        dy *= _ARROW_MAX_PX / length
    x1, y1 = x0 + dx, y0 + dy
    draw.line([(x0, y0), (x1, y1)], fill=halo + (220,), width=5)
    draw.line([(x0, y0), (x1, y1)], fill=colour, width=3)
    ang = math.atan2(dy, dx)
    hs = 11.0
    for sgn in (-1, 1):
        bx = x1 - hs * math.cos(ang + sgn * math.radians(25))
        by = y1 - hs * math.sin(ang + sgn * math.radians(25))
        draw.line([(x1, y1), (bx, by)], fill=halo + (220,), width=5)
        draw.line([(x1, y1), (bx, by)], fill=colour, width=3)


def render_lightning_map(
    *,
    geo: CompositeGeo,
    pixel_scale_m: float,
    home_lat: float,
    home_lon: float,
    strikes: list[LightningStrike],
    clusters: list[ClusterSummary],
    rings_km: tuple[float, ...] = (3.0, 10.0, 60.0),
    pixel_lat: float | None = None,
    pixel_lon: float | None = None,
    zoom_km: float = 100.0,
    output_px: int = 600,
    basemap=None,
    now: datetime | None = None,
    header: str | None = None,
) -> bytes:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise RenderUnavailable("Pillow is not installed") from exc

    if now is None:
        now = datetime.now(timezone.utc)

    h, w = geo.composite.reflectivity_dbz.shape
    idx = geo.lonlat_to_grid(home_lon, home_lat)
    half = int(round((zoom_km * 1000.0) / pixel_scale_m))
    col_c, row_c = int(round(idx.col)), int(round(idx.row))
    c0, c1 = max(0, col_c - half), min(w, col_c + half)
    r0, r1 = max(0, row_c - half), min(h, row_c + half)
    crop_h, crop_w = max(1, r1 - r0), max(1, c1 - c0)
    scale = output_px / max(crop_h, crop_w)
    new_w, new_h = int(round(crop_w * scale)), int(round(crop_h * scale))

    if basemap is not None:
        bg = basemap.resize((new_w, new_h), Image.LANCZOS).convert("RGBA")
        bg.alpha_composite(Image.new("RGBA", (new_w, new_h), (255, 255, 255, 70)))
    else:
        bg = Image.new("RGBA", (new_w, new_h), (245, 245, 245, 255))
    img = bg
    draw = ImageDraw.Draw(img)

    def to_xy(lat: float, lon: float) -> tuple[float, float]:
        li = geo.lonlat_to_grid(lon, lat)
        return (li.col - c0) * scale, (li.row - r0) * scale

    def vec_px_per_min(ve: float, vn: float) -> tuple[float, float]:
        # ENU km/min → image px/min. North = decreasing grid row.
        dcol = ve * 1000.0 / pixel_scale_m
        drow = -vn * 1000.0 / pixel_scale_m
        return dcol * scale, drow * scale

    home_x, home_y = to_xy(home_lat, home_lon)
    _draw_crosshair(draw, home_x=home_x, home_y=home_y, image_w=new_w, image_h=new_h)
    _draw_distance_rings(draw, home_x=home_x, home_y=home_y,
                         pixel_scale_m=pixel_scale_m, scale=scale,
                         image_w=new_w, image_h=new_h, radii_km=tuple(rings_km))

    # Strikes (oldest first so recent dots land on top).
    for s in sorted(strikes, key=lambda x: x.t):
        x, y = to_xy(s.lat, s.lon)
        if not (-5 <= x <= new_w + 5 and -5 <= y <= new_h + 5):
            continue
        age = (now - s.t).total_seconds() / 60.0
        col = _strike_colour(age)
        r = 3 if age <= 5 else 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=col + (255,),
                     outline=(40, 40, 40, 180))

    # Clusters: circle + motion arrow + label.
    font = _load_font(11)
    for c in clusters:
        cx, cy = to_xy(c.centroid_lat, c.centroid_lon)
        if not (-20 <= cx <= new_w + 20 and -20 <= cy <= new_h + 20):
            continue
        col = _CLUSTER_THREAT_RGBA if c.threatening else _CLUSTER_OTHER_RGBA
        rpx = max(8.0, (max(c.spread_km, 1.5) * 1000.0 / pixel_scale_m) * scale)
        draw.ellipse((cx - rpx, cy - rpx, cx + rpx, cy + rpx),
                     outline=col, width=3 if c.threatening else 2)
        if c.vel_east_kmmin is not None and c.vel_north_kmmin is not None:
            dpx, dpy = vec_px_per_min(c.vel_east_kmmin, c.vel_north_kmmin)
            _draw_vector_arrow(draw, cx, cy, dpx * _ARROW_PROJECT_MIN,
                               dpy * _ARROW_PROJECT_MIN, colour=col)
        bits = []
        if c.speed_kmh is not None:
            bits.append(f"{c.speed_kmh:.0f}km/h")
        if c.bearing_deg is not None:
            bits.append(f"{c.bearing_deg:.0f}°")
        if c.eta_min is not None:
            bits.append(f"ETA {c.eta_min:.0f}m")
        if bits:
            _text_with_shadow(draw, (cx + rpx + 3, cy - 7), " ".join(bits), font,
                              fill=(255, 80, 80) if c.threatening else (40, 40, 40))

    # Pixel target marker + its own 3/10 km rings (when distinct from home).
    if pixel_lat is not None and pixel_lon is not None:
        px, py = to_xy(pixel_lat, pixel_lon)
        if 0 <= px <= new_w and 0 <= py <= new_h:
            _draw_distance_rings(draw, home_x=px, home_y=py,
                                 pixel_scale_m=pixel_scale_m, scale=scale,
                                 image_w=new_w, image_h=new_h,
                                 radii_km=(3.0, 10.0), colour=(0, 120, 255, 150))
            draw.ellipse((px - 5, py - 5, px + 5, py + 5),
                         fill=(0, 140, 255, 255), outline=(255, 255, 255, 255), width=2)
            _text_with_shadow(draw, (px + 7, py - 7), "Pixel", font, fill=(0, 90, 200))

    # Home marker on top.
    draw.ellipse((home_x - 5, home_y - 5, home_x + 5, home_y + 5),
                 fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=2)

    if header:
        _text_with_shadow(draw, (6, 4), header, _load_font(13), fill=(20, 20, 20))

    out = io.BytesIO()
    img.convert("RGB").save(out, format="PNG")
    return out.getvalue()


__all__ = ["render_lightning_map"]
