"""Build the fixed national calibration point set.

Produces ``src/dmi_nowcast_core/calibration_points_v2.json``: ~120 points across
Denmark, stratified by sub-region × coast-distance band × distance-to-nearest-
DMI-radar band, selected once and committed so month-over-month calibration
fits stay comparable. One fixed reference point — the geometric centroid of
Fyn — is always included, so there is a stable, named point in the middle of
the country to plot curves for.

Stratification runs on **DuckDB with the ``spatial`` extension** — the geo
prep engine the idea doc (§8) commits to: land containment against the
committed Denmark boundary polygon, and metric coast / radar distances via
``ST_Transform`` to ETRS89 / UTM 32N.

Reproducibility contract
------------------------
- The boundary polygon is committed at ``data/denmark_boundary.geojson``
  (Natural Earth 10m admin-0 Denmark, public domain — see ``data/README.md``),
  so regeneration never needs the network. ``main()`` can re-download the
  Natural Earth source only if the committed file is somehow missing.
- Everything except the ``generated`` timestamp is a pure function of
  ``--seed`` (default fixed): candidate cloud, stratum areas (Monte-Carlo
  estimated from the same seeded cloud), allocation, and selection.
- Every point is verified inside the radar composite grid with
  ``CompositeGeo`` against a real archived composite; generation fails loudly
  otherwise, and the provenance line records which composite was used.

Sub-region boxes
----------------
``regions.py``'s Denmark box stays the coarse authority for region
classification project-wide. The boxes below are **calibration strata only**:
8 named lat/lon boxes, priority-ordered (first match wins) because rectangles
cannot follow the Little Belt or the Kattegat coast exactly. Known, accepted
slivers of that compromise: Fredericia's shore strip falls in "Fyn", Samsø in
"Sydjylland", Haderslev in "Sydjylland", Anholt in "Nordjylland". Harmless —
strata only need to spread points across the country, not match municipal law.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_SEED = 20260829
DEFAULT_N_POINTS = 120
N_CANDIDATES = 40_000

BOUNDARY_PATH = REPO_ROOT / "data" / "denmark_boundary.geojson"
OUTPUT_PATH = REPO_ROOT / "src" / "dmi_nowcast_core" / "calibration_points_v2.json"
DEFAULT_COMPOSITE = REPO_ROOT / "radar_archive" / "dk.com.202511221750.500_max.h5"

# Natural Earth source for the committed boundary (public domain). Only used
# by main() if data/denmark_boundary.geojson is missing.
NE_ADMIN0_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_admin_0_countries.geojson"
)

# The fixed reference point — always included in the set (replaces one sample
# of its stratum); its id is load-bearing downstream. It is the geometric
# (area-weighted) centroid of the Fyn polygon of the committed Natural Earth
# boundary, rounded to 2 decimals: a neutral, reproducible "middle of the
# country" point rather than anyone's address.
REFERENCE_ID = "fyn-centroid"
REFERENCE_LAT = 55.33
REFERENCE_LON = 10.32

# Calibration strata boxes: name -> (lat_min, lat_max, lon_min, lon_max).
# PRIORITY-ORDERED — first box containing the point wins (dict order).
# These are strata for the national calibration fit, NOT a replacement for
# regions.py (whose Denmark box remains the coarse classification authority).
CALIBRATION_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "Bornholm": (54.90, 55.35, 14.50, 15.30),
    "Hovedstaden": (55.50, 56.15, 11.90, 12.75),
    "Sjælland": (54.50, 56.05, 10.85, 12.70),  # incl. Lolland, Falster, Møn
    "Nordjylland": (56.60, 57.80, 8.00, 11.65),  # incl. Læsø (+ Anholt sliver)
    "Sønderjylland": (54.55, 55.15, 8.00, 10.10),  # incl. Als, Rømø
    "Fyn": (54.60, 55.65, 9.72, 10.90),  # incl. Langeland, Ærø
    "Sydjylland": (55.15, 55.95, 8.00, 10.90),  # incl. Samsø sliver
    "Midtjylland": (55.95, 56.60, 8.00, 11.60),
}

# Denmark bounding box for candidate sampling (superset of all region boxes).
DK_BBOX = (54.40, 57.85, 8.00, 15.35)  # lat_min, lat_max, lon_min, lon_max

# DMI's five operational C-band radar sites, approximate coordinates
# (lat, lon). Provenance: publicly known site locations from DMI's radar
# network documentation and the EUMETNET OPERA radar database; accuracy of
# ~1 km is ample for distance *banding* (nothing here aims them).
DMI_RADARS: dict[str, tuple[float, float]] = {
    "Rømø (Juvre)": (55.1731, 8.5520),
    "Sindal": (57.4893, 10.1361),
    "Stevns": (55.3262, 12.4493),
    "Virring": (56.0240, 10.0246),
    "Bornholm (Rø)": (55.1127, 14.8875),
}

COAST_BANDS = ["<10km", ">=10km"]  # split at 10 km from the coastline
RADAR_BANDS = ["<50km", "50-100km", ">=100km"]  # to the nearest DMI radar

# Per-region minimum point counts: enough for a per-region reliability curve
# to at least exist. Bornholm is one radar-near stratum of ~590 km² — a pure
# area-proportional share would give it a single point.
REGION_MIN_POINTS_DEFAULT = 3
REGION_MIN_POINTS = {"Bornholm": 2}


def _coast_band(km: float) -> str:
    return COAST_BANDS[0] if km < 10.0 else COAST_BANDS[1]


def _radar_band(km: float) -> str:
    if km < 50.0:
        return RADAR_BANDS[0]
    if km < 100.0:
        return RADAR_BANDS[1]
    return RADAR_BANDS[2]


def _region_of(lat: float, lon: float) -> str | None:
    """First calibration-region box containing (lat, lon) — priority order."""
    for name, (a0, a1, o0, o1) in CALIBRATION_REGIONS.items():
        if a0 <= lat <= a1 and o0 <= lon <= o1:
            return name
    return None


def _slug(name: str) -> str:
    """ASCII id fragment: 'Sønderjylland' -> 'soenderjylland'."""
    out = []
    for ch in name.lower():
        if ch == "æ":
            out.append("ae")
        elif ch == "ø":
            out.append("oe")
        elif ch == "å":
            out.append("aa")
        else:
            out.append(unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode())
    return "".join(out)


def _classify_candidates(
    boundary_path: Path, candidates: list[tuple[int, float, float]]
) -> dict[int, tuple[float, float]]:
    """DuckDB spatial: keep candidates on Danish land; return metric distances.

    Returns {candidate_idx: (coast_km, radar_km)} for the on-land subset.
    Land containment, coast distance (to the boundary polygon's outline) and
    nearest-radar distance are all computed in SQL — the idea-doc §8 pattern.
    Distances are metric via ETRS89 / UTM 32N (EPSG:25832); the slight
    distortion at Bornholm (zone 33) is irrelevant for 10/50/100 km banding.
    """
    import duckdb
    import pyarrow as pa

    fc = json.loads(boundary_path.read_text())
    geom_json = json.dumps(fc["features"][0]["geometry"])

    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute(
        "CREATE TABLE dk AS SELECT ST_GeomFromGeoJSON(?) AS geom", [geom_json]
    )
    radars = pa.table(
        {
            "rlat": [lat for lat, _ in DMI_RADARS.values()],
            "rlon": [lon for _, lon in DMI_RADARS.values()],
        }
    )
    cand = pa.table(
        {
            "idx": [i for i, _, _ in candidates],
            "lat": [la for _, la, _ in candidates],
            "lon": [lo for _, _, lo in candidates],
        }
    )
    con.register("radars_arrow", radars)
    con.register("cand_arrow", cand)

    rows = con.execute(
        """
        WITH coast AS (
            SELECT ST_Transform(ST_Boundary(geom),
                                'EPSG:4326', 'EPSG:25832', always_xy := true)
                   AS geom_m
            FROM dk
        ),
        radar_m AS (
            SELECT ST_Transform(ST_Point(rlon, rlat),
                                'EPSG:4326', 'EPSG:25832', always_xy := true)
                   AS geom_m
            FROM radars_arrow
        ),
        land AS (
            SELECT c.idx, ST_Point(c.lon, c.lat) AS geom
            FROM cand_arrow c, dk
            WHERE ST_Contains(dk.geom, ST_Point(c.lon, c.lat))
        ),
        land_m AS (
            SELECT idx,
                   ST_Transform(geom, 'EPSG:4326', 'EPSG:25832',
                                always_xy := true) AS geom_m
            FROM land
        )
        SELECT l.idx,
               (SELECT ST_Distance(l.geom_m, coast.geom_m) FROM coast) / 1000.0
                   AS coast_km,
               (SELECT MIN(ST_Distance(l.geom_m, r.geom_m)) FROM radar_m r)
                   / 1000.0 AS radar_km
        FROM land_m l
        ORDER BY l.idx
        """
    ).fetchall()
    con.close()
    return {idx: (coast_km, radar_km) for idx, coast_km, radar_km in rows}


def _allocate(total: int, weights: dict, floors: dict | int = 1) -> dict:
    """Largest-remainder allocation proportional to ``weights``, with floors.

    Deterministic: keys are processed in a fixed sorted order. ``floors`` is
    a per-key dict or a single int. If the floors alone exceed ``total``
    (many tiny strata), the largest-weight keys get 1 point each instead.
    """
    keys = sorted(weights)
    floor_of = (lambda k: floors.get(k, 1)) if isinstance(floors, dict) else (lambda k: floors)
    if sum(floor_of(k) for k in keys) > total:
        top = sorted(keys, key=lambda k: (-weights[k], k))[:total]
        return {k: 1 for k in top}
    w_sum = sum(weights[k] for k in keys)
    raw = {k: total * weights[k] / w_sum for k in keys}
    alloc = {k: max(floor_of(k), int(raw[k])) for k in keys}
    # Distribute any shortfall by largest fractional remainder.
    while sum(alloc.values()) < total:
        rem = sorted(keys, key=lambda k: (-(raw[k] - int(raw[k])), k))
        for k in rem:
            if sum(alloc.values()) >= total:
                break
            alloc[k] += 1
    # Trim any oversubscription (floors on many tiny keys) from the largest,
    # never below a key's floor.
    while sum(alloc.values()) > total:
        trimmable = [k for k in keys if alloc[k] > floor_of(k)]
        if not trimmable:
            break  # floors saturated — accept the overshoot
        big = sorted(trimmable, key=lambda k: (-alloc[k], k))[0]
        alloc[big] -= 1
    return alloc


def generate(
    seed: int = DEFAULT_SEED,
    boundary_path: Path = BOUNDARY_PATH,
    composite_path: Path = DEFAULT_COMPOSITE,
    n_points: int = DEFAULT_N_POINTS,
    generated: str | None = None,
) -> dict:
    """Build the point-set payload. Pure function of ``seed`` (except
    ``generated``, which can be pinned for determinism tests)."""
    from dmi_nowcast_core.geo import CompositeGeo
    from dmi_nowcast_core.parse import parse_composite

    rng = random.Random(seed)
    lat_min, lat_max, lon_min, lon_max = DK_BBOX
    # 5 decimals ≈ 1.1 m: round *before* classification so the committed
    # coordinates are exactly the classified ones (no band/land flips later).
    candidates = [
        (
            i,
            round(rng.uniform(lat_min, lat_max), 5),
            round(rng.uniform(lon_min, lon_max), 5),
        )
        for i in range(N_CANDIDATES)
    ]
    # The reference point rides through the same classification pipeline
    # as idx -1.
    candidates.append((-1, REFERENCE_LAT, REFERENCE_LON))

    dists = _classify_candidates(boundary_path, candidates)
    if -1 not in dists:
        raise SystemExit(
            f"FATAL: reference point ({REFERENCE_LAT}, {REFERENCE_LON}) is not "
            f"inside the Denmark boundary polygon {boundary_path}"
        )

    # Stratum key: (region, coast_band, radar_band); membership in candidate
    # (idx) order so selection below is deterministic.
    strata: dict[tuple, list[tuple[int, float, float]]] = {}
    reference_key = None
    by_idx = {i: (la, lo) for i, la, lo in candidates}
    for idx in sorted(dists):
        lat, lon = by_idx[idx]
        region = _region_of(lat, lon)
        if region is None:
            continue  # on land but outside every calibration box — skip
        coast_km, radar_km = dists[idx]
        key = (region, _coast_band(coast_km), _radar_band(radar_km))
        if idx == -1:
            reference_key = key
            continue
        strata.setdefault(key, []).append((idx, lat, lon))

    assert reference_key is not None
    if reference_key not in strata:
        # Degenerate but possible: the reference point alone fills the stratum.
        strata[reference_key] = []

    # Two-stage allocation: regions first (proportional to Monte-Carlo land
    # area, with per-region floors so every region can carry a reliability
    # curve), then strata within each region (proportional, floor 1).
    by_region: dict[str, dict[tuple, list]] = {}
    for key, members in strata.items():
        by_region.setdefault(key[0], {})[key] = members
    region_weights = {
        # max(1, ...) keeps a degenerate reference-only region allocatable.
        r: max(1, sum(len(m) for m in s.values()))
        for r, s in by_region.items()
    }
    region_floors = {
        r: REGION_MIN_POINTS.get(r, REGION_MIN_POINTS_DEFAULT)
        for r in region_weights
    }
    region_alloc = _allocate(n_points, region_weights, region_floors)
    alloc: dict[tuple, int] = {}
    for region, n_region in region_alloc.items():
        stratum_weights = {
            k: max(1, len(v)) if k == reference_key else len(v)
            for k, v in by_region[region].items()
        }
        alloc.update(_allocate(n_region, stratum_weights, 1))

    # Selection: first-k candidates per stratum; the reference point replaces
    # one slot of its own stratum. Ids are region-sequenced in stratum-sorted
    # order.
    points: list[dict] = []
    region_seq: dict[str, int] = {}
    for key in sorted(alloc):
        region = key[0]
        n_take = alloc[key]
        members = strata[key]
        if key == reference_key:
            points.append(
                {
                    "id": REFERENCE_ID,
                    "lat": REFERENCE_LAT,
                    "lon": REFERENCE_LON,
                    "region": region,
                    "strata": {"coast_km_band": key[1], "radar_km_band": key[2]},
                }
            )
            n_take -= 1
        if n_take > len(members):
            n_take = len(members)  # tiny stratum: take what exists
        for idx, lat, lon in members[:n_take]:
            region_seq[region] = region_seq.get(region, 0) + 1
            points.append(
                {
                    "id": f"{_slug(region)}-{region_seq[region]:02d}",
                    "lat": lat,
                    "lon": lon,
                    "region": region,
                    "strata": {"coast_km_band": key[1], "radar_km_band": key[2]},
                }
            )

    if not (115 <= len(points) <= 125):
        raise SystemExit(
            f"FATAL: produced {len(points)} points, outside the 115–125 band"
        )

    # In-grid verification against a real composite — fail loudly (plan B1).
    geo = CompositeGeo(parse_composite(composite_path))
    outside = [p["id"] for p in points if not geo.is_in_grid(p["lon"], p["lat"])]
    if outside:
        raise SystemExit(
            f"FATAL: {len(outside)} points outside the composite grid of "
            f"{composite_path.name}: {outside}"
        )

    if generated is None:
        generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provenance = (
        "Stratified sample: Natural Earth 10m admin-0 Denmark boundary "
        "(public domain, data/denmark_boundary.geojson) via DuckDB spatial; "
        "strata = 8 region boxes x coast band (10 km) x nearest-DMI-radar "
        f"band (50/100 km); all points in-grid vs {composite_path.name}"
    )
    return {
        "version": 2,
        "generated": generated,
        "seed": seed,
        "provenance": provenance,
        "points": points,
    }


def _ensure_boundary(boundary_path: Path) -> None:
    """Re-create the committed boundary from Natural Earth if missing."""
    if boundary_path.exists():
        return
    import urllib.request

    print(f"boundary missing — downloading Natural Earth admin-0 ({NE_ADMIN0_URL})")
    with urllib.request.urlopen(NE_ADMIN0_URL) as resp:
        fc = json.load(resp)
    feature = next(
        f for f in fc["features"] if f["properties"].get("ADMIN") == "Denmark"
    )
    out = {
        "type": "FeatureCollection",
        "name": "denmark_boundary",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "admin": "Denmark",
                    "source": (
                        "Natural Earth 1:10m Cultural Vectors, Admin 0 - "
                        "Countries (ne_10m_admin_0_countries), extracted "
                        "feature ADMIN='Denmark', geometry unmodified"
                    ),
                    "source_url": NE_ADMIN0_URL,
                    "license": "Public domain (Natural Earth terms of use)",
                    "retrieved": datetime.now(timezone.utc).date().isoformat(),
                },
                "geometry": feature["geometry"],
            }
        ],
    }
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    boundary_path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {boundary_path}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--boundary", type=Path, default=BOUNDARY_PATH)
    ap.add_argument("--composite", type=Path, default=DEFAULT_COMPOSITE)
    ap.add_argument("--out", type=Path, default=OUTPUT_PATH)
    ap.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS)
    args = ap.parse_args(argv)

    _ensure_boundary(args.boundary)
    payload = generate(
        seed=args.seed,
        boundary_path=args.boundary,
        composite_path=args.composite,
        n_points=args.n_points,
    )
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")

    per_region: dict[str, int] = {}
    for p in payload["points"]:
        per_region[p["region"]] = per_region.get(p["region"], 0) + 1
    print(f"wrote {args.out}: {len(payload['points'])} points")
    for region in CALIBRATION_REGIONS:
        print(f"  {region:>15}: {per_region.get(region, 0)}")


if __name__ == "__main__":
    main()
