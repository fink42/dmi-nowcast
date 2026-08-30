"""OSM basemap underlay for the radar overlay.

Renders a static OpenStreetMap raster (roads, water, town blocks) sized
to match the radar crop, so the radar field has geographic context like
the FIA Weather paddock display. Tiles are fetched once per (home,
zoom_km, output_px) combination and cached on disk — subsequent renders
just load the cached PNG.

OSM tile-server policy notes:
- One fetch per configuration is well within personal-use limits.
- A descriptive User-Agent is required.
- We must include "© OpenStreetMap contributors" on the rendered output.
  ``render.py`` handles that in the attribution line.

The projection mismatch (stereographic radar vs Web Mercator tiles) is
~1 km at the corners of a 150 km crop at 55 °N — negligible for visual
context, far smaller than the resolution of the upscaled radar (~187 m
per output pixel).
"""
from __future__ import annotations

import hashlib
import io
import logging
import math
import os
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

# OSM tile server. The standard endpoint is fine for one-off personal fetches;
# heavy usage would need a self-hosted Renderd or a commercial provider.
_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
# OSM's tile usage policy asks for a User-Agent identifying the *deployment*
# with a way to get in touch. Set DMI_NOWCAST_USER_AGENT to something like
# "myapp/1.0 (contact@example.com)" before pointing this at the public tile
# servers; the bare default suits only a handful of one-off local fetches.
_USER_AGENT = os.environ.get("DMI_NOWCAST_USER_AGENT", "dmi-nowcast/0.1")


def _lonlat_to_tile_xy(lon: float, lat: float, z: int) -> tuple[float, float]:
    """Slippy-map tile coordinates (fractional) for the given lon/lat."""
    lat_rad = math.radians(lat)
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _bbox_for_crop(
    home_lat: float, home_lon: float, zoom_km: float,
) -> tuple[float, float, float, float]:
    """Approximate lat/lon bounding box for a square crop of side 2*zoom_km
    centred on home. Uses a flat-earth approximation (good to ~0.1 % at
    Denmark latitudes for ~150 km half-side)."""
    lat_deg_per_km = 1.0 / 111.0
    lon_deg_per_km = 1.0 / (111.0 * max(math.cos(math.radians(home_lat)), 0.1))
    half_lat = zoom_km * lat_deg_per_km
    half_lon = zoom_km * lon_deg_per_km
    return (
        home_lon - half_lon,  # west
        home_lat - half_lat,  # south
        home_lon + half_lon,  # east
        home_lat + half_lat,  # north
    )


def cache_path_for(
    cache_dir: Path, home_lat: float, home_lon: float,
    zoom_km: float, output_px: int, tile_zoom: int,
) -> Path:
    """Stable cache path for a given configuration.

    Hashed inputs so file names stay short; collisions are vanishingly
    unlikely (5-char base16 truncation of SHA-256 over five floats)."""
    h = hashlib.sha256(
        f"{home_lat:.4f}_{home_lon:.4f}_{zoom_km:.0f}_{output_px}_{tile_zoom}".encode()
    ).hexdigest()[:10]
    return cache_dir / f"basemap_{h}.png"


def build_basemap(
    *,
    home_lat: float,
    home_lon: float,
    zoom_km: float,
    output_px: tuple[int, int],
    cache_dir: Path,
    tile_zoom: int = 8,
    timeout_s: float = 15.0,
):
    """Return a PIL Image of the basemap sized to ``output_px``.

    Result is cached on disk; subsequent calls with the same arguments
    just load the file. Returns ``None`` on failure (network error,
    Pillow missing, etc.) — caller should fall back to the plain white
    background.

    ``tile_zoom`` picks OSM tile detail; 8 gives one tile per ~150 km at
    lat 55°, which is right for our 150 km crop with 9 tiles in a 3×3
    mosaic. Bump to 9 for sharper text at the cost of 4× more tiles.
    """
    try:
        from PIL import Image
    except ImportError:
        _LOGGER.info("Pillow not installed; basemap unavailable")
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path_for(cache_dir, home_lat, home_lon, zoom_km, output_px[0], tile_zoom)
    if cache_file.exists():
        try:
            return Image.open(cache_file).convert("RGBA")
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("basemap cache read failed (%s); refetching", exc)

    try:
        img = _fetch_and_stitch(
            home_lat=home_lat, home_lon=home_lon,
            zoom_km=zoom_km, output_px=output_px,
            tile_zoom=tile_zoom, timeout_s=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("basemap fetch failed: %s — radar will render without it", exc)
        return None

    try:
        img.save(cache_file, format="PNG", optimize=True)
        _LOGGER.info("basemap cached at %s (%d bytes)",
                     cache_file, cache_file.stat().st_size)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("basemap cache write failed: %s", exc)
    return img


def _fetch_and_stitch(
    *,
    home_lat: float,
    home_lon: float,
    zoom_km: float,
    output_px: tuple[int, int],
    tile_zoom: int,
    timeout_s: float,
):
    """Heavy lifting for build_basemap.

    Importing httpx lazily so this module is cheap to import on test
    paths that never need the basemap.
    """
    import httpx
    from PIL import Image

    west, south, east, north = _bbox_for_crop(home_lat, home_lon, zoom_km)
    # Tile range covering the bbox (tile_y goes north→south so swap).
    x_nw_f, y_nw_f = _lonlat_to_tile_xy(west, north, tile_zoom)
    x_se_f, y_se_f = _lonlat_to_tile_xy(east, south, tile_zoom)
    x_min, x_max = int(math.floor(x_nw_f)), int(math.floor(x_se_f))
    y_min, y_max = int(math.floor(y_nw_f)), int(math.floor(y_se_f))
    n_x = x_max - x_min + 1
    n_y = y_max - y_min + 1
    _LOGGER.info(
        "fetching basemap tiles z=%d, x=%d..%d, y=%d..%d (%d tiles)",
        tile_zoom, x_min, x_max, y_min, y_max, n_x * n_y,
    )

    with httpx.Client(
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout_s,
        http2=False,
    ) as client:
        tile_imgs: dict[tuple[int, int], Image.Image] = {}
        for tx in range(x_min, x_max + 1):
            for ty in range(y_min, y_max + 1):
                url = _TILE_URL.format(z=tile_zoom, x=tx, y=ty)
                resp = client.get(url)
                resp.raise_for_status()
                tile_imgs[(tx, ty)] = Image.open(io.BytesIO(resp.content)).convert("RGBA")

    # Stitch into a single (n_x*256) × (n_y*256) mosaic.
    mosaic = Image.new("RGBA", (n_x * 256, n_y * 256))
    for (tx, ty), tile in tile_imgs.items():
        mosaic.paste(tile, ((tx - x_min) * 256, (ty - y_min) * 256))

    # Crop the mosaic to the precise bbox in tile-pixel space.
    crop_left = int(round((x_nw_f - x_min) * 256))
    crop_top = int(round((y_nw_f - y_min) * 256))
    crop_right = int(round((x_se_f - x_min) * 256))
    crop_bottom = int(round((y_se_f - y_min) * 256))
    cropped = mosaic.crop((crop_left, crop_top, crop_right, crop_bottom))

    # Resize to match the radar output. LANCZOS gives clean roads/text
    # at the cost of ~5 % more CPU than BILINEAR — basemap renders once
    # so the cost is paid once.
    return cropped.resize(output_px, Image.LANCZOS)
