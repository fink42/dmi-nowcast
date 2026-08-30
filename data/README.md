# data/

## denmark_boundary.geojson

Denmark land boundary used by `scripts/build_calibration_points.py` to
stratify the national calibration point set (website Phase B, package B1).

- **Source:** Natural Earth, 1:10m Cultural Vectors, "Admin 0 – Countries"
  (`ne_10m_admin_0_countries`), feature `ADMIN = 'Denmark'`, geometry
  committed unmodified (2,180 vertices, 15 polygons, ~47 KB).
  Retrieved 2026-08-29 from
  <https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson>.
- **License:** Public domain. Natural Earth's terms of use place all
  versions of Natural Earth vector and raster data in the public domain
  (<https://www.naturalearthdata.com/about/terms-of-use/>).
- **Why committed:** the calibration point generator must be reproducible
  offline forever; the one-time download happened once, here. The 1:10m
  coastline is generalised (~1 km scale error) — fine for the <10 km /
  ≥10 km coast-distance banding it feeds, not a survey-grade coastline.

## strikes/

Lightning strike archive (see the lightning pipeline docs) — not related
to the calibration point set.
