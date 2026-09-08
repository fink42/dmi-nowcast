"""Shared helpers for the two-product (fullRange / doppler) studies — Phase H, H-L.

DMI interleaves two composites in one collection: ``fullRange`` on the
``:x0`` minutes (240 km range) and ``doppler`` on the ``:x5`` minutes
(120 km range). They share the grid, the projection and the scaling, but
they are **not** the same view: doppler covers ~40 % of fullRange's area
and, where both see an echo, reads 0–2.5 dB lower at the intensities the
wet threshold lives at (plan §0.3). Anything that wants to use the fresher
doppler frame therefore needs three things this module provides:

* a way to say **where** a pixel or a station is relative to the radars
  (:func:`nearest_radar_km`, :func:`radar_distance_grid`, :func:`band_of_km`) —
  the distance to the nearest radar is the one covariate that plausibly
  explains the difference, since it is the beam height that differs;
* a **fast disc sampler** (:class:`DiscSampler`) that reproduces
  :func:`dmi_nowcast_core.sample.sample_disc` exactly while reading a
  hundred stations out of one frame instead of converting the whole grid
  to rain rate once per station;
* the **harmonisation map** (:func:`fit_quantile_map`,
  :func:`apply_harmonisation`) that L2 fits and the runtime would later
  apply: an empirical quantile mapping doppler dBZ → fullRange dBZ.

Nothing here fetches, and nothing here knows about HA or FastAPI. The two
study scripts (``scripts/gauge_agreement_study.py``,
``scripts/fit_doppler_harmonisation.py``) are the only callers today; the
runtime becomes one if L3 says the freshest-frame anchor wins.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .lightning import EARTH_RADIUS_KM, haversine_km
from .parse import RadarComposite
from .sample import disc_pixel_indices
from .transform import dbz_to_rain_rate

__all__ = [
    "RADAR_SITES",
    "DISTANCE_BAND_EDGES_KM",
    "DISTANCE_BANDS",
    "DOPPLER_BANDS",
    "SEASON_MONTHS",
    "SEASON_ORDER",
    "POOLED_KEY",
    "band_of_km",
    "band_index_grid",
    "season_of",
    "nearest_radar_km",
    "radar_distance_grid",
    "rain_rate_to_dbz",
    "same_grid",
    "require_same_grid",
    "DiscSample",
    "DiscSampler",
    "HARMONISATION_SCHEMA_VERSION",
    "HARM_BIN_WIDTH_DBZ",
    "HARM_MIN_DBZ",
    "HARM_HIST_LO_DBZ",
    "HARM_HIST_HI_DBZ",
    "HARM_MIN_TAIL_FRACTION",
    "HARM_MIN_TAIL_PIXELS",
    "hist_edges",
    "digitize_dbz",
    "mapped_edges",
    "fit_quantile_map",
    "table_key",
    "apply_harmonisation",
    "frame_map",
]


# ---------------------------------------------------------------------------
# Geometry: the radars, the distance bands, the seasons
# ---------------------------------------------------------------------------

#: DMI's five operational C-band radar sites, ``name -> (lat, lon)``.
#:
#: Copied deliberately rather than imported: ``scripts/build_calibration_points.py``
#: owns the same table but is a script, and a core module must not import
#: from ``scripts/``. ``tests/test_gauge_agreement_study.py`` asserts the two
#: are identical, so they cannot drift.
#:
#: Provenance: publicly known site locations from DMI's radar network
#: documentation and the EUMETNET OPERA radar database; ~1 km accuracy,
#: which is ample for distance *banding* (nothing here aims a radar).
RADAR_SITES: dict[str, tuple[float, float]] = {
    "Rømø (Juvre)": (55.1731, 8.5520),
    "Sindal": (57.4893, 10.1361),
    "Stevns": (55.3262, 12.4493),
    "Virring": (56.0240, 10.0246),
    "Bornholm (Rø)": (55.1127, 14.8875),
}

#: Band edges in km from the NEAREST radar. 120 km is doppler's range, so
#: the last band is by construction fullRange-only; 60 and 90 split the
#: covered part into "beam still low", "beam climbing", "beam over the
#: freezing level in winter".
DISTANCE_BAND_EDGES_KM: tuple[float, float, float] = (60.0, 90.0, 120.0)

#: Band names, in order. Index i is the band for
#: ``digitize(km, DISTANCE_BAND_EDGES_KM)``.
DISTANCE_BANDS: tuple[str, ...] = ("0-60km", "60-90km", "90-120km", ">120km")

#: The bands doppler can be harmonised in — beyond 120 km it does not see.
DOPPLER_BANDS: tuple[str, ...] = DISTANCE_BANDS[:3]

#: Months per season. Identical to
#: ``dmi_nowcast_sidecar.threshold_sweep.SEASON_MONTHS`` — the project's
#: shipped cut — repeated here because a core module must not import the
#: sidecar package. Summer is the convective half, winter the stratiform
#: one, and the shoulder months are neither.
SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "summer": (5, 6, 7, 8, 9),
    "winter": (12, 1, 2, 3),
    "shoulder": (4, 10, 11),
}

#: Reporting order: the two seasons the contrast is about, then the rest.
SEASON_ORDER: tuple[str, ...] = ("summer", "winter", "shoulder")

#: Stratum name for "every season at once" — a fitted table under this key
#: is the fallback when a season has too little data of its own.
POOLED_KEY = "pooled"


def season_of(when: datetime) -> str:
    """``"summer"`` / ``"winter"`` / ``"shoulder"`` for one instant."""
    month = when.month
    for name in SEASON_ORDER:
        if month in SEASON_MONTHS[name]:
            return name
    raise ValueError(f"month {month} belongs to no season")  # pragma: no cover


def band_of_km(km: float) -> str:
    """Distance band name for a distance to the nearest radar.

    NaN — a point with no distance, which should not happen — falls in the
    last band rather than raising: an unbanded row would silently vanish
    from every stratified table.
    """
    if not math.isfinite(km):
        return DISTANCE_BANDS[-1]
    for edge, name in zip(DISTANCE_BAND_EDGES_KM, DISTANCE_BANDS):
        if km < edge:
            return name
    return DISTANCE_BANDS[-1]


def band_index_grid(distance_km: np.ndarray) -> np.ndarray:
    """Band index (uint8, matching :data:`DISTANCE_BANDS`) per pixel."""
    return np.digitize(
        np.asarray(distance_km, dtype=np.float32), DISTANCE_BAND_EDGES_KM,
    ).astype(np.uint8)


def nearest_radar_km(lat: float, lon: float) -> float:
    """Great-circle km from ``(lat, lon)`` to the closest DMI radar."""
    return min(
        haversine_km(lat, lon, site_lat, site_lon)
        for site_lat, site_lon in RADAR_SITES.values()
    )


def _haversine_km_arrays(
    lat: np.ndarray, lon: np.ndarray, site_lat: float, site_lon: float,
) -> np.ndarray:
    """Vectorised twin of :func:`dmi_nowcast_core.lightning.haversine_km`.

    Same formula, same Earth radius; the scalar version is math-module
    only and cannot take arrays. ``tests/test_doppler_harmonisation.py``
    pins the two against each other.
    """
    p1 = np.radians(site_lat)
    p2 = np.radians(lat)
    dphi = np.radians(lat - site_lat)
    dlmb = np.radians(lon - site_lon)
    a = np.sin(dphi / 2.0) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def radar_distance_grid(
    composite: RadarComposite, *, chunk_rows: int = 128,
) -> np.ndarray:
    """Km from every pixel to the nearest radar, as a float32 grid.

    Computed on the geoid (the same haversine the station distances use),
    not in the projection plane, so a pixel's band and a station's band
    mean the same thing. Built in row blocks: the intermediate lon/lat and
    per-site distance arrays for a whole 1728x1984 grid are ~27 MB each in
    float64, and this runs inside a batch worker under a 5 GB cgroup cap.
    """
    from pyproj import CRS, Transformer  # local: keeps module import cheap

    height, width = composite.reflectivity_dbz.shape
    crs = CRS.from_proj4(composite.projection)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    ul_lon, ul_lat = composite.corners_lonlat["UL"]
    x_min, y_max = to_proj.transform(ul_lon, ul_lat)

    # Same index convention as CompositeGeo: col c sits at x_min + c*xscale.
    xs = x_min + np.arange(width, dtype=np.float64) * composite.xscale_m
    out = np.empty((height, width), dtype=np.float32)
    for row0 in range(0, height, int(chunk_rows)):
        row1 = min(height, row0 + int(chunk_rows))
        ys = y_max - np.arange(row0, row1, dtype=np.float64) * composite.yscale_m
        gx, gy = np.meshgrid(xs, ys)
        lon, lat = to_wgs.transform(gx, gy)
        best = None
        for site_lat, site_lon in RADAR_SITES.values():
            d = _haversine_km_arrays(lat, lon, site_lat, site_lon)
            best = d if best is None else np.minimum(best, d, out=best)
        out[row0:row1] = best.astype(np.float32)
    return out


def rain_rate_to_dbz(mm_h: float, *, zr_a: float = 200.0, zr_b: float = 1.6) -> float:
    """Invert Marshall-Palmer: the reflectivity a rain rate corresponds to.

    The exact inverse of :func:`dmi_nowcast_core.transform.dbz_to_rain_rate`
    below its caps, so "wet at 0.5 mm/h" can be expressed as a dBZ
    threshold and applied to a reflectivity field without a full Z-R pass
    over the grid.
    """
    if mm_h <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(zr_a * mm_h ** zr_b)


def same_grid(a: RadarComposite, b: RadarComposite) -> bool:
    """True when two composites share shape, projection and pixel scale.

    The premise of every dual-product comparison: DMI publishes both
    products on one grid, so pixel *(r, c)* is the same place in both. It
    is checked rather than assumed — a silently reprojected product would
    otherwise produce plausible, wrong numbers.
    """
    return (
        a.reflectivity_dbz.shape == b.reflectivity_dbz.shape
        and a.projection == b.projection
        and a.xscale_m == b.xscale_m
        and a.yscale_m == b.yscale_m
        and a.corners_lonlat == b.corners_lonlat
    )


def require_same_grid(a: RadarComposite, b: RadarComposite) -> None:
    """:func:`same_grid`, raising ``ValueError`` with both source paths."""
    if not same_grid(a, b):
        raise ValueError(
            f"composites are not on the same grid: {a.source_path} vs {b.source_path}"
        )


# ---------------------------------------------------------------------------
# Disc sampling, many stations at a time
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscSample:
    """One frame sampled at every station, as parallel arrays.

    All arrays are ``float32`` of length ``n_points``, in the order the
    points were given. ``p90`` / ``max_`` / ``mean`` are mm/h and are NaN
    where the disc holds no valid pixel at all. ``valid_frac`` is the
    share of the disc's pixels that carry data (``nodata`` excluded,
    ``undetect`` counted — an observed dry pixel is data), and is NaN for
    a point that falls outside the grid entirely.
    """

    p90: np.ndarray
    max_: np.ndarray
    mean: np.ndarray
    valid_frac: np.ndarray


class DiscSampler:
    """Pre-computed 1 km disc indices for a fixed set of points.

    :func:`dmi_nowcast_core.sample.sample_disc` is the production "raining
    now" statistic and this must agree with it exactly — the test asserts
    it does, on the committed fixture, for points on land, at the grid
    edge and off the grid. The difference is only *when* the work happens:
    ``sample_disc`` takes a rain-rate grid, so sampling 118 stations from
    one frame means converting 3.4M pixels to mm/h to read ~1500 of them.
    Here the disc indices are computed once (the grid never moves), and
    each frame converts only the pixels inside the discs.

    The indices depend on the grid, not the frame, so a sampler built from
    one composite is valid for every composite of both products — checked
    per frame with :func:`same_grid` via :meth:`sample`.
    """

    def __init__(
        self,
        composite: RadarComposite,
        points: Sequence[tuple[float, float]],
        *,
        radius_m: float = 1000.0,
    ) -> None:
        from .geo import CompositeGeo  # local: pyproj import is not free

        self.reference = composite
        self.radius_m = float(radius_m)
        self.shape = tuple(composite.reflectivity_dbz.shape)
        geo = CompositeGeo(composite)
        pixel_scale_m = (composite.xscale_m + composite.yscale_m) / 2.0
        radius_px = self.radius_m / pixel_scale_m
        self._idx: list[tuple[np.ndarray, np.ndarray]] = []
        self._n_pixels = np.zeros(len(points), dtype=np.int32)
        for i, (lat, lon) in enumerate(points):
            idx = geo.lonlat_to_grid(lon, lat)
            rows, cols = disc_pixel_indices(
                self.shape, idx.row, idx.col, radius_px,
            )
            self._idx.append((rows, cols))
            self._n_pixels[i] = rows.size

    def __len__(self) -> int:
        return len(self._idx)

    @property
    def n_pixels(self) -> np.ndarray:
        """Pixels in each point's disc (0 for a point off the grid)."""
        return self._n_pixels.copy()

    def sample(self, composite: RadarComposite) -> DiscSample:
        """Sample ``composite`` at every point. Rain rates in mm/h."""
        require_same_grid(self.reference, composite)
        dbz = composite.reflectivity_dbz
        n = len(self._idx)
        p90 = np.full(n, np.nan, dtype=np.float32)
        mx = np.full(n, np.nan, dtype=np.float32)
        mean = np.full(n, np.nan, dtype=np.float32)
        frac = np.full(n, np.nan, dtype=np.float32)
        for i, (rows, cols) in enumerate(self._idx):
            n_pixels = int(self._n_pixels[i])
            if n_pixels == 0:
                continue  # off the grid: nothing to say, not even "dry"
            rate = dbz_to_rain_rate(
                dbz[rows, cols], zr_a=composite.zr_a, zr_b=composite.zr_b,
            )
            finite = rate[np.isfinite(rate)]
            frac[i] = finite.size / n_pixels
            if finite.size == 0:
                continue  # inside the grid but all nodata: not covered
            p90[i] = np.percentile(finite, 90)
            mx[i] = finite.max()
            mean[i] = finite.mean()
        return DiscSample(p90=p90, max_=mx, mean=mean, valid_frac=frac)


# ---------------------------------------------------------------------------
# The harmonisation map: doppler dBZ -> fullRange dBZ
# ---------------------------------------------------------------------------

#: Bump when the on-disk shape of ``doppler_harmonisation.json`` changes
#: incompatibly. A consumer must refuse a version it does not know.
HARMONISATION_SCHEMA_VERSION = 1

#: Resolution of the mapping table. DMI publishes dBZ with ``gain=0.5``,
#: so half a dB is the data's own quantisation — a finer bin would be
#: interpolating between values that cannot occur.
HARM_BIN_WIDTH_DBZ = 0.5

#: Below this, a pixel is not echo and the map leaves it alone. 7 dBZ is
#: the plan's own "is there anything there at all" level (§0.3), well
#: under the 18.2 dBZ that 0.5 mm/h converts to.
HARM_MIN_DBZ = 7.0

#: A quantile is only fitted where BOTH products still have real pixels
#: above the edge: at least :data:`HARM_MIN_TAIL_PIXELS`, and at least
#: this share of the sample. Higher up the two tails are a handful of
#: convective cores and the inverse CDF is noise — worse, doppler's tail
#: thins out faster than fullRange's (it sees the same storms from a
#: lower beam and a shorter range), so matching the two tails there
#: stretches doppler's last few pixels across fullRange's, turning a
#: 40 dBZ doppler echo into a fictional 59.5 dBZ core. Measured on the
#: committed fixture pair: 14 doppler pixels above 40 dBZ against 174
#: fullRange ones.
HARM_MIN_TAIL_FRACTION = 1e-5

#: Absolute floor for the same test. A quantile fitted on fewer than a
#: thousand pixels moves by whole dB when one storm cell comes or goes.
HARM_MIN_TAIL_PIXELS = 1000.0

#: Histogram support. The low end is DMI's own floor (``offset = -32``);
#: ``undetect`` (-inf in the parsed field) is folded into the first bin,
#: which is what makes the exceedance fractions share one denominator with
#: the dry pixels included — the property that makes the map area-matching.
HARM_HIST_LO_DBZ = -32.0
HARM_HIST_HI_DBZ = 60.0


def hist_edges() -> np.ndarray:
    """Left edge of every histogram bin, ascending (float64)."""
    n = int(round((HARM_HIST_HI_DBZ - HARM_HIST_LO_DBZ) / HARM_BIN_WIDTH_DBZ))
    return HARM_HIST_LO_DBZ + np.arange(n, dtype=np.float64) * HARM_BIN_WIDTH_DBZ


def mapped_edges() -> np.ndarray:
    """The doppler bin edges the fitted table is indexed by (>= 7 dBZ)."""
    edges = hist_edges()
    return edges[edges >= HARM_MIN_DBZ]


def digitize_dbz(dbz: np.ndarray) -> np.ndarray:
    """Histogram bin index per value, clipped into the support.

    ``-inf`` (undetect) lands in bin 0 and NaN must be filtered by the
    caller — a NaN index would be meaningless, and silently dropping
    nodata here would hide a coverage bug.
    """
    edges = hist_edges()
    idx = np.floor(
        (np.asarray(dbz, dtype=np.float64) - HARM_HIST_LO_DBZ) / HARM_BIN_WIDTH_DBZ
    )
    return np.clip(idx, 0, edges.size - 1).astype(np.int32)


def _exceedance(hist: np.ndarray) -> np.ndarray:
    """Fraction of the sample at or above each bin's left edge."""
    total = float(hist.sum())
    if total <= 0.0:
        return np.zeros(hist.size, dtype=np.float64)
    return np.cumsum(hist[::-1])[::-1] / total


def fit_quantile_map(
    hist_doppler: np.ndarray, hist_fullrange: np.ndarray,
) -> np.ndarray:
    """Empirical quantile map, doppler dBZ -> fullRange dBZ.

    Both histograms are over :func:`hist_edges` and over the SAME pixel
    domain (joint coverage, one band, one season), each including its dry
    pixels. That shared denominator is the point: the map sends a doppler
    value ``x`` to the fullRange value ``y`` with the same exceedance
    fraction, so at every threshold the mapped doppler field covers the
    same area as fullRange does. Area bias -> 1.00 is then a property of
    the fit rather than something to hope for.

    Returns one value per edge in :func:`mapped_edges` — the table is
    emitted only from 7 dBZ up, because below that a pixel is not echo and
    the runtime leaves it exactly as it found it (dry stays dry).

    In the far tail the map stops fitting and starts extrapolating. Above
    the highest value either product ever showed, the exceedance fractions
    are zero or single pixels, and the naive inverse CDF there is not just
    noisy: where fullRange has run out of pixels entirely it sends a rare
    doppler echo to the top of the histogram (59.5 dBZ, ~1000 mm/h before
    the hail cap). So the last bin where both tails still hold
    :data:`HARM_MIN_TAIL_PIXELS` pixels is the anchor, and everything
    above it carries that bin's offset — a rare 55 dBZ pixel keeps the
    correction its neighbours got instead of inventing one of its own.

    The result is forced non-decreasing: it is monotone by construction
    (both exceedance curves are), and ``np.interp`` across a plateau of
    empty fullRange bins can otherwise emit a flat step in the wrong
    order.
    """
    edges = hist_edges()
    counts_d = np.asarray(hist_doppler, dtype=np.float64)
    counts_f = np.asarray(hist_fullrange, dtype=np.float64)
    frac_d = _exceedance(counts_d)
    frac_f = _exceedance(counts_f)
    keep = edges >= HARM_MIN_DBZ
    kept_edges = edges[keep]
    # np.interp wants ascending x: exceedance falls as dBZ rises, so both
    # axes are reversed together.
    mapped = np.asarray(
        np.interp(frac_d[keep], frac_f[::-1], edges[::-1]), dtype=np.float64,
    )
    # Pixels still above each edge, in each product's own sample.
    tail_d = np.cumsum(counts_d[::-1])[::-1][keep]
    tail_f = np.cumsum(counts_f[::-1])[::-1][keep]
    floor_d = max(HARM_MIN_TAIL_PIXELS, HARM_MIN_TAIL_FRACTION * counts_d.sum())
    floor_f = max(HARM_MIN_TAIL_PIXELS, HARM_MIN_TAIL_FRACTION * counts_f.sum())
    supported = np.flatnonzero((tail_d >= floor_d) & (tail_f >= floor_f))
    if supported.size == 0:
        return kept_edges.copy()  # nothing observed: the identity is honest
    last = int(supported[-1])
    if last + 1 < mapped.size:
        offset = mapped[last] - kept_edges[last]
        mapped[last + 1:] = kept_edges[last + 1:] + offset
    return np.maximum.accumulate(mapped)


def table_key(season: str, band: str) -> str:
    """Key of one fitted table inside the JSON payload."""
    return f"{season}|{band}"


def _lookup_table(
    payload: Mapping[str, Any], season: str, band: str,
) -> Mapping[str, Any] | None:
    """The table for ``(season, band)``, falling back to the pooled one.

    A season the fit never saw enough of has no table of its own; the
    pooled table is a better answer than the identity, and returning
    ``None`` (identity) is only correct when there is no table at all.
    """
    tables = payload.get("tables") or {}
    for key in (table_key(season, band), table_key(POOLED_KEY, band)):
        table = tables.get(key)
        if table and table.get("mapped_dbz"):
            return table
    return None


def apply_harmonisation(
    dbz_doppler: np.ndarray,
    distance_km_grid: np.ndarray,
    season: str,
    table: Mapping[str, Any],
) -> np.ndarray:
    """Map a doppler reflectivity field onto fullRange's distribution.

    ``table`` is the whole parsed ``doppler_harmonisation.json`` payload,
    not one of its entries: the band a pixel belongs to is decided here,
    from ``distance_km_grid`` (:func:`radar_distance_grid`), so one call
    harmonises a whole frame.

    Untouched, deliberately: ``nodata`` (NaN) and ``undetect`` (-inf) stay
    exactly what they were, pixels below :data:`HARM_MIN_DBZ` stay dry,
    and pixels beyond doppler's 120 km range are left alone because no
    table is fitted there. The returned array is a new float32 grid; the
    input is not modified.
    """
    version = int(table.get("schema_version", 0))
    if version != HARMONISATION_SCHEMA_VERSION:
        raise ValueError(
            f"harmonisation schema version {version} != "
            f"{HARMONISATION_SCHEMA_VERSION}; refusing to apply"
        )
    src = np.asarray(dbz_doppler, dtype=np.float32)
    out = src.astype(np.float32, copy=True)
    dist = np.asarray(distance_km_grid, dtype=np.float32)
    if dist.shape != src.shape:
        raise ValueError(
            f"distance grid {dist.shape} does not match field {src.shape}"
        )
    edges = np.asarray(table.get("bin_edges_dbz") or mapped_edges(), dtype=np.float64)
    bands = band_index_grid(dist)
    echo = np.isfinite(src) & (src >= HARM_MIN_DBZ)
    for i, band in enumerate(DISTANCE_BANDS):
        if band not in DOPPLER_BANDS:
            continue  # beyond doppler's range: nothing was fitted
        entry = _lookup_table(table, season, band)
        if entry is None:
            continue  # no table, no change — the identity is the honest map
        mask = echo & (bands == i)
        if not mask.any():
            continue
        mapped = np.asarray(entry["mapped_dbz"], dtype=np.float64)
        out[mask] = np.interp(src[mask], edges, mapped).astype(np.float32)
    return out


def frame_map(
    frames: Iterable[Any],
) -> dict[str, dict[datetime, Path]]:
    """``{scan_type: {timestamp: path}}`` for archived frames.

    Takes what :meth:`dmi_nowcast_core.corpus.ArchiveIndex.list_in_window`
    returns. Both study scripts build this ONCE in the parent process and
    hand each worker the handful of paths its day needs — an
    ``ArchiveIndex`` per worker per day would re-scan 77k filenames thirty
    times over.
    """
    out: dict[str, dict[datetime, Path]] = {}
    for frame in frames:
        out.setdefault(frame.scan_type, {})[frame.datetime_utc] = Path(frame.path)
    return out
