"""Project lat/lon ↔ radar grid index using a composite's stereographic projection.

DMI's composites use ``+proj=stere +lat_0=56 +lon_0=10.5666 +lat_ts=56`` on WGS84.
The grid origin is the geographic ``UL`` corner from ``/where``, which corresponds
to the minimum-x / maximum-y point of the projection rectangle.

Row/column convention:
- row 0 is the top of the array (north); row increases southward.
- col 0 is the left edge (west); col increases eastward.

Plan §13 Phase 0 calls for a verified lat/lon → grid round-trip against a known
landmark; the test in ``tests/test_geo.py`` does exactly that for Copenhagen
Central Station.
"""
from __future__ import annotations

from dataclasses import dataclass

from pyproj import CRS, Transformer

from .parse import RadarComposite


@dataclass(frozen=True)
class GridIndex:
    row: float
    col: float


class CompositeGeo:
    """Pre-built coordinate transforms for one composite frame."""

    def __init__(self, composite: RadarComposite):
        self.composite = composite
        crs = CRS.from_proj4(composite.projection)
        self._to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self._to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        ul_lon, ul_lat = composite.corners_lonlat["UL"]
        # UL in geographic terms = (min_x, max_y) in projection coordinates.
        self._x_min, self._y_max = self._to_proj.transform(ul_lon, ul_lat)

    @property
    def projection_origin_m(self) -> tuple[float, float]:
        """UL projection-plane corner ``(x_min, y_max)`` in metres.

        Read-only accessor for serialising the grid geometry (website Phase A
        §A2 artifact manifests): together with the composite's proj4 string
        and pixel scale it lets a remote consumer reproduce
        ``lonlat_to_grid`` — ``col = (x - x_min) / xscale_m`` and
        ``row = (y_max - y) / yscale_m`` — without parsing the HDF5.
        """
        return (float(self._x_min), float(self._y_max))

    def lonlat_to_grid(self, lon: float, lat: float) -> GridIndex:
        x, y = self._to_proj.transform(lon, lat)
        col = (x - self._x_min) / self.composite.xscale_m
        row = (self._y_max - y) / self.composite.yscale_m
        return GridIndex(row=row, col=col)

    def grid_to_lonlat(self, row: float, col: float) -> tuple[float, float]:
        x = self._x_min + col * self.composite.xscale_m
        y = self._y_max - row * self.composite.yscale_m
        lon, lat = self._to_wgs.transform(x, y)
        return float(lon), float(lat)

    def is_in_grid(self, lon: float, lat: float) -> bool:
        idx = self.lonlat_to_grid(lon, lat)
        height, width = self.composite.reflectivity_dbz.shape
        return 0 <= idx.row < height and 0 <= idx.col < width
