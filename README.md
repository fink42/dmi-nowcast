# dmi-nowcast

Short-range rain nowcasting for Denmark from **DMI Open Data radar
composites**: dense optical flow + Lagrangian advection + a STEPS
ensemble, reduced to national grids of *calibrated* rain probability,
expected time-of-arrival and intensity, and served over HTTP.

Two pieces live here:

- **`src/dmi_nowcast_core/`** — the algorithm library. Fetch, ODIM HDF5
  parsing, Z–R conversion, motion estimation, advection, STEPS ensemble,
  national products, isotonic calibration, verification metrics,
  rendering. Pure Python + numpy; no web framework, no Home Assistant.
- **`sidecar/dmi_nowcast_sidecar/`** — a small FastAPI service that runs
  the pipeline on a schedule (a new DMI composite arrives every ~5 min)
  and serves the results: `/healthz`, `/state.json`, `/nowcast/*`
  artifacts, `/forecast?lat=&lon=`, and rendered radar frames.

> Rain expected in ~12 minutes (8–15 min window), moderate intensity
> (~2.4 mm/h), P(rain within 20 min) = 0.78.

## What this is, honestly

This is a **hobby project**. It is built in spare evenings, with heavy AI
assistance — vibe-coded, because that is the time budget that exists. It
runs on a single small server. There is **no SLA**, no support promise,
and no guarantee that any endpoint keeps working tomorrow. It is shared
because the algorithm work and the validation are real and might be
useful to someone else, not because it is a product.

**This is not a warning service.** Official Danish weather warnings come
from [DMI](https://www.dmi.dk/). Two limits are worth stating up front:

- DMI's composite is **column-max reflectivity**, not surface rain rate.
  It reads high over tall convective cores, virga and the melting-layer
  bright band, so predicted intensity is an upper-bound proxy, not a
  rain-gauge estimate.
- **Advection nowcasting decorrelates fast.** Useful skill is in the
  0–30 min band, degrading through ~60 min; beyond that a numerical
  weather model beats this decisively. Summer convection is roughly 3×
  harder than winter stratiform.

Probabilities served by the pipeline are isotonic-calibrated against a
backtest corpus built from DMI's own 180-day archive, sampled at ~120
fixed points across the country, so "0.7" is meant to be worth 0.7.

## Data source and attribution

Radar data: **DMI** — [DMI Open Data](https://opendatadocs.dmi.govcloud.dk/).
Attribution is required by DMI's free-data terms, and any deployment or
UI built on this must credit DMI as the source.

The committed Denmark boundary polygon (`data/denmark_boundary.geojson`)
comes from [Natural Earth](https://www.naturalearthdata.com/) (public
domain) — see `data/README.md`.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# core library + dev tooling
uv sync --all-groups
uv run python -m pytest tests/ -q

# the sidecar service
cd sidecar
uv sync --all-groups
uv run python -m pytest tests/ -q

# run it (copy the example config first and edit the reference point)
cp config.example.yaml config.yaml
uv run python -m dmi_nowcast_sidecar
# → http://localhost:8081/healthz
```

The first cycle needs three consecutive radar frames, so the service
serves `503` from `/state.json` until it has fetched them (~a few
minutes after start).

Container deployment lives in `sidecar/deploy/` (Dockerfile + compose +
an SSH deploy script). Everything host-specific there is driven by
environment variables — copy `.env.example` to `.env` and fill it in.

## Architecture

```
DMI Open Data  ──►  fetch + parse (ODIM HDF5)  ──►  dBZ field
                                                      │
                                    Farnebäck dense optical flow
                                                      │
                              ┌───────────────────────┴─────────────────┐
                     Lagrangian advection                    STEPS ensemble
                     (deterministic frames)          (probabilistic, cascade + noise)
                              │                                          │
                              └──────────────► national products ◄───────┘
                                     p_rain / ETA / intensity grids
                                                      │
                                        isotonic calibration curves
                                                      │
                                     FastAPI: /state.json, /nowcast/*, /forecast
```

Supporting pieces:

- `scripts/` — corpus builders, calibration fitting, backtests, reports,
  validation. `scripts/build_calibration_corpus.py` samples events from
  DMI's archive and writes a Parquet corpus (one row per
  `(event_time, point, lead)`); `scripts/fit_national_calibration.py`
  fits the pooled national curves from it.
- `sql/` — DuckDB queries over the corpus (reliability, Brier
  decomposition, per-region and per-season breakdowns).
- `tests/` — the core suite; `sidecar/tests/` — the service suite.

## Method, and why it is this method

Optical flow plus extrapolation, not deep learning. Ayzel et al. (2020)
found RainNet losing to plain optical flow at the ≥10 mm/h intensities
that actually matter, and models of the DGMR class need multi-year
national archives and TPU-scale training. The STEPS scheme (Bowler et
al. 2006; Pulkkinen et al. 2019) supplies the probabilistic part: a
spectral cascade where small scales — which lose predictability first —
are replaced by stochastic noise as the lead time grows.

The STEPS implementation is a **vendored subset of pysteps 1.21.1**
under `src/dmi_nowcast_core/_vendor/pysteps_steps/` (BSD-3-Clause,
notices preserved). Upstream pysteps has no musl wheels and its source
build wants an OpenMP-capable compiler, which made it unusable in the
target environment; the vendored subset is the same code, trimmed to
what is called.

For calibration sanity: Imhoff et al. (2020) report pysteps
decorrelation times of 25 / 40 / 56 / 116 min for 1 / 3 / 6 / 24-hour
events over Dutch lowland catchments — the closest published analogue to
Denmark. Numbers much worse than that suggest a bug; much better
suggests a verification mistake.

## License

MIT — see [LICENSE](LICENSE).

Third-party notices:

- **pysteps** — the vendored STEPS subset is BSD-3-Clause. See
  `src/dmi_nowcast_core/_vendor/pysteps_steps/LICENSE-pysteps` and
  `NOTICE`.
- **Natural Earth** — the Denmark boundary polygon is public domain.
- **DMI Open Data** — radar composites, used under DMI's free-data terms
  with attribution.

## Acknowledgements

- **DMI** for publishing radar composites as open data.
- **pysteps** (Pulkkinen et al. 2019, *Geosci. Model Dev.* 12,
  4185–4219) — the nowcast engine this builds on.
- **Imhoff et al. 2020** (*Water Resour. Res.* 56, e2019WR026723) — the
  lowland benchmark the skill numbers are checked against.
