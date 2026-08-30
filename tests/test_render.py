"""Smoke test for render_overlay — checks that the function produces a valid PNG."""
from __future__ import annotations

import io
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.parse import parse_composite
from dmi_nowcast_core.render import (
    LoopFrame,
    OverlayInputs,
    _apply_colormap,
    render_loop_png,
    render_overlay,
)
from dmi_nowcast_core.transform import dbz_to_rain_rate

FIXTURE = Path(__file__).parent / "fixtures" / "composite_fullrange.h5"


def test_colormap_alpha_ramp_fades_light_rain():
    # Faint/clutter echoes (the column-max product over-reads these) must be
    # transparent or near-transparent, not a solid wash; heavy rain stays opaque.
    alpha = lambda r: int(_apply_colormap(np.array([[r]]))[0, 0, 3])
    assert alpha(0.0) == 0 and alpha(0.1) == 0 and alpha(0.2) == 0  # below floor
    assert 0 < alpha(0.3) <= 90        # light: faint
    assert alpha(0.5) < alpha(1.0) < alpha(2.0)   # monotone ramp
    assert alpha(2.0) == 255 and alpha(20.0) == 255  # heavy: solid


def test_render_overlay_produces_png():
    composite = parse_composite(FIXTURE)
    geo = CompositeGeo(composite)
    rain = dbz_to_rain_rate(composite.reflectivity_dbz, zr_a=composite.zr_a, zr_b=composite.zr_b)
    png = render_overlay(OverlayInputs(
        composite=composite,
        rain_rate_mm_h=rain,
        geo=geo,
        home_lat=55.33, home_lon=10.32,
        radius_km=1.0,
        disc_max_mm_h=0.5,
        eta_p50_min=12.0,
        probability_30min=0.7,
        confidence=0.85,
    ))
    # PNG signature: 0x89 P N G \r \n 0x1a \n
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 5_000  # reasonable lower bound for a 8×7.5 inch dpi=120 figure


def test_render_overlay_handles_missing_annotations():
    composite = parse_composite(FIXTURE)
    geo = CompositeGeo(composite)
    rain = dbz_to_rain_rate(composite.reflectivity_dbz)
    png = render_overlay(OverlayInputs(
        composite=composite, rain_rate_mm_h=rain, geo=geo,
        home_lat=55.33, home_lon=10.32, radius_km=1.0,
    ))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def _loop_frame(rain, composite, *, offset_min: int, label: str, kind: str, duration_ms: int):
    """Build a LoopFrame timestamped relative to the fixture's UTC time."""
    return LoopFrame(
        rain_rate_mm_h=rain,
        timestamp_utc=composite.timestamp_utc + timedelta(minutes=offset_min),
        label=label, kind=kind, duration_ms=duration_ms,
    )


def test_render_loop_png_produces_animated_png():
    """The loop renderer must produce a multi-frame PNG that Pillow re-loads
    as an animation with the expected frame count."""
    from PIL import Image
    composite = parse_composite(FIXTURE)
    geo = CompositeGeo(composite)
    rain = dbz_to_rain_rate(composite.reflectivity_dbz, zr_a=composite.zr_a, zr_b=composite.zr_b)
    frames = [
        _loop_frame(rain, composite, offset_min=-10, label="−10 min", kind="observed", duration_ms=300),
        _loop_frame(rain, composite, offset_min=-5,  label="−5 min",  kind="observed", duration_ms=300),
        _loop_frame(rain, composite, offset_min=0,   label="obs",     kind="observed", duration_ms=1000),
        _loop_frame(rain, composite, offset_min=0,   label="now",     kind="forecast", duration_ms=1500),
        _loop_frame(rain, composite, offset_min=10,  label="+10 min", kind="forecast", duration_ms=400),
        _loop_frame(rain, composite, offset_min=20,  label="+20 min", kind="forecast", duration_ms=400),
        _loop_frame(rain, composite, offset_min=30,  label="+30 min", kind="forecast", duration_ms=1000),
    ]
    png = render_loop_png(
        frames, composite=composite, geo=geo,
        home_lat=55.33, home_lon=10.32, radius_km=1.0,
        now_stats_subline="now: 0.42 mm/h  •  ETA ~12 min  •  conf 78%",
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # APNGs are larger than single PNGs even with diff encoding.
    assert len(png) > 50_000

    # Re-parse with Pillow and check n_frames.
    img = Image.open(io.BytesIO(png))
    assert getattr(img, "is_animated", False), "expected an animated PNG"
    assert img.n_frames == len(frames), f"expected {len(frames)} frames, got {img.n_frames}"


def test_render_loop_png_empty_input_raises():
    """Empty frame list is a programmer error, not an empty PNG."""
    composite = parse_composite(FIXTURE)
    geo = CompositeGeo(composite)
    with pytest.raises(ValueError, match="at least one frame"):
        render_loop_png([], composite=composite, geo=geo,
                        home_lat=55.33, home_lon=10.32, radius_km=1.0)


def test_render_loop_png_minimum_one_frame():
    """A single-frame loop is degenerate but valid (e.g., first poll with
    only one observation and forecast disabled)."""
    composite = parse_composite(FIXTURE)
    geo = CompositeGeo(composite)
    rain = dbz_to_rain_rate(composite.reflectivity_dbz)
    png = render_loop_png(
        [_loop_frame(rain, composite, offset_min=0, label="obs",
                     kind="observed", duration_ms=1000)],
        composite=composite, geo=geo,
        home_lat=55.33, home_lon=10.32, radius_km=1.0,
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_loop_png_uses_persistent_canvas_disposal():
    """Every frame's fcTL must declare dispose_op=0 (DISPOSE_OP_NONE).

    Regression for a bug where ``disposal=2`` (DISPOSE_OP_PREVIOUS) combined
    with Pillow's frame-diff encoder produced tiny sub-rectangle frames that
    rendered onto a blanked canvas — the radar field disappeared whenever
    only the header text had changed between frames.
    """
    import struct
    composite = parse_composite(FIXTURE)
    geo = CompositeGeo(composite)
    rain = dbz_to_rain_rate(composite.reflectivity_dbz, zr_a=composite.zr_a, zr_b=composite.zr_b)
    # Construct a sequence where most frames are visually identical so the
    # encoder is tempted to use sub-rectangles.
    frames = [
        _loop_frame(rain, composite, offset_min=i, label=f"+{i} min",
                    kind="forecast", duration_ms=400)
        for i in range(0, 30, 5)
    ]
    png = render_loop_png(frames, composite=composite, geo=geo,
                          home_lat=55.33, home_lon=10.32, radius_km=1.0)
    # Walk APNG chunks and check every fcTL's dispose_op byte.
    dispose_ops: list[int] = []
    i = 8  # skip PNG signature
    while i < len(png):
        length = struct.unpack(">I", png[i:i + 4])[0]
        ctype = png[i + 4:i + 8].decode("ascii")
        if ctype == "fcTL":
            payload = png[i + 8:i + 8 + length]
            (dispose_op,) = struct.unpack(">B", payload[24:25])
            dispose_ops.append(dispose_op)
        elif ctype == "IEND":
            break
        i += 8 + length + 4
    assert dispose_ops, "no fcTL chunks found — not an APNG?"
    assert all(d == 0 for d in dispose_ops), (
        f"every frame must use DISPOSE_OP_NONE (0); got {dispose_ops}"
    )


def _count_red_pixels(png_bytes: bytes) -> int:
    """Count red-channel-dominant pixels — proxy for "is the red arrow drawn?"."""
    from PIL import Image
    arr = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    return int(((arr[..., 0] > 200) & (arr[..., 1] < 80) & (arr[..., 2] < 80)).sum())


def test_motion_arrow_drawn_for_moderate_motion():
    """A motion of ~1 px/min (radar grid) → 60-min projection > 1 km → arrow."""
    composite = parse_composite(FIXTURE)
    geo = CompositeGeo(composite)
    rain = dbz_to_rain_rate(composite.reflectivity_dbz)
    png = render_overlay(OverlayInputs(
        composite=composite, rain_rate_mm_h=rain, geo=geo,
        home_lat=55.33, home_lon=10.32, radius_km=1.0,
        motion_dy_per_min=-1.0, motion_dx_per_min=0.5,
    ))
    assert _count_red_pixels(png) > 50, "expected a visible red arrow"


def test_motion_arrow_suppressed_when_calm():
    """Motion magnitudes below the 1-km-per-hour threshold must NOT draw."""
    composite = parse_composite(FIXTURE)
    geo = CompositeGeo(composite)
    rain = dbz_to_rain_rate(composite.reflectivity_dbz)
    # Zero motion → no arrow.
    png_zero = render_overlay(OverlayInputs(
        composite=composite, rain_rate_mm_h=rain, geo=geo,
        home_lat=55.33, home_lon=10.32, radius_km=1.0,
        motion_dy_per_min=0.0, motion_dx_per_min=0.0,
    ))
    assert _count_red_pixels(png_zero) == 0
    # Tiny motion → 60-min projected distance < 1 km → suppressed.
    png_tiny = render_overlay(OverlayInputs(
        composite=composite, rain_rate_mm_h=rain, geo=geo,
        home_lat=55.33, home_lon=10.32, radius_km=1.0,
        motion_dy_per_min=0.005, motion_dx_per_min=0.005,
    ))
    assert _count_red_pixels(png_tiny) == 0


def test_motion_arrow_suppressed_below_20_kmh_threshold():
    """The arrow should only draw when 60-min projection ≥ 20 km
    (~20 km/h sustained). Below that the arrow either disappears under
    the home dot or its tick labels visibly bunch — the top-right
    caption keeps the actual numbers readable instead.

    Test inputs:
      - 0.15 px/min on a 500 m/pixel grid = 0.075 km/min = 4.5 km/h →
        well below 20 km/h threshold → no arrow.
      - 1.0 px/min on the same grid ≈ 30 km/h → above threshold → arrow.
    """
    composite = parse_composite(FIXTURE)
    geo = CompositeGeo(composite)
    rain = dbz_to_rain_rate(composite.reflectivity_dbz)
    png_below = render_overlay(OverlayInputs(
        composite=composite, rain_rate_mm_h=rain, geo=geo,
        home_lat=55.33, home_lon=10.32, radius_km=1.0,
        motion_dy_per_min=0.15, motion_dx_per_min=0.0,
    ))
    assert _count_red_pixels(png_below) == 0, "expected no arrow at 4.5 km/h"
    png_above = render_overlay(OverlayInputs(
        composite=composite, rain_rate_mm_h=rain, geo=geo,
        home_lat=55.33, home_lon=10.32, radius_km=1.0,
        motion_dy_per_min=1.0, motion_dx_per_min=0.0,
    ))
    assert _count_red_pixels(png_above) > 50, "expected arrow at ~30 km/h"


def test_motion_arrow_suppressed_when_no_motion_supplied():
    """If motion isn't passed, no arrow draws (back-compat default)."""
    composite = parse_composite(FIXTURE)
    geo = CompositeGeo(composite)
    rain = dbz_to_rain_rate(composite.reflectivity_dbz)
    png = render_overlay(OverlayInputs(
        composite=composite, rain_rate_mm_h=rain, geo=geo,
        home_lat=55.33, home_lon=10.32, radius_km=1.0,
    ))
    assert _count_red_pixels(png) == 0


def test_motion_arrow_direction_is_reversed():
    """The arrow points OPPOSITE to the motion vector — it should show where
    the rain is coming FROM, not where it's going.

    With motion = pure-east (vx > 0, vy = 0) the arrow must extend WEST of
    home. We check this by counting red pixels in each half of the image.
    """
    from PIL import Image
    composite = parse_composite(FIXTURE)
    geo = CompositeGeo(composite)
    rain = dbz_to_rain_rate(composite.reflectivity_dbz)
    png = render_overlay(OverlayInputs(
        composite=composite, rain_rate_mm_h=rain, geo=geo,
        home_lat=55.33, home_lon=10.32, radius_km=1.0,
        motion_dy_per_min=0.0, motion_dx_per_min=4.0,  # pure east motion
    ))
    arr = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    red = (arr[..., 0] > 200) & (arr[..., 1] < 80) & (arr[..., 2] < 80)
    h, w = red.shape
    # Home is in the middle of the image. With east motion the arrow points
    # west: most red pixels should be in the LEFT half of the image.
    left_red = red[:, : w // 2].sum()
    right_red = red[:, w // 2:].sum()
    assert left_red > 5 * right_red, (
        f"arrow should point west (left), got left={left_red} right={right_red}"
    )


def test_render_loop_png_clamps_too_short_duration():
    """duration_ms below the APNG safe floor (50 ms) must be clamped, not
    rejected — protects against bad per-frame timing from upstream."""
    from PIL import Image
    composite = parse_composite(FIXTURE)
    geo = CompositeGeo(composite)
    rain = dbz_to_rain_rate(composite.reflectivity_dbz)
    frames = [
        _loop_frame(rain, composite, offset_min=0, label="obs", kind="observed", duration_ms=0),
        _loop_frame(rain, composite, offset_min=5, label="+5 min", kind="forecast", duration_ms=-100),
    ]
    png = render_loop_png(frames, composite=composite, geo=geo,
                          home_lat=55.33, home_lon=10.32, radius_km=1.0)
    img = Image.open(io.BytesIO(png))
    assert img.n_frames == 2
