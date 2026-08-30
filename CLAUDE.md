# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

Radar nowcasting for Denmark from DMI Open Data composites: optical flow
+ Lagrangian advection + a STEPS ensemble, reduced to national grids of
calibrated rain probability / ETA / intensity, served over HTTP.

```
src/dmi_nowcast_core/          algorithm library (numpy only; no web framework)
sidecar/dmi_nowcast_sidecar/   FastAPI service that runs the pipeline on a timer
scripts/                       corpus builders, calibration fits, backtests, reports
sql/                           DuckDB queries over the calibration corpus
tests/  sidecar/tests/         the two suites
```

- **`src/dmi_nowcast_core/`** — fetch, ODIM HDF5 parse, Z–R conversion,
  dense flow, advection, STEPS (vendored subset under `_vendor/`),
  national products, isotonic calibration, verification metrics,
  rendering. No FastAPI, no home-automation imports — this package must
  stay importable on its own.
- **`sidecar/dmi_nowcast_sidecar/`** — scheduler + FastAPI routes:
  `/healthz`, `/state.json`, `/forecast?lat=&lon=`,
  `/nowcast/manifest.json` + quantised PNG artifacts, `/frames/*`.
  Deployment assets in `sidecar/deploy/`.

## Non-negotiable implementation contracts

Easy to get wrong, expensive to fix later:

- **No blocking work on an event loop.** Network I/O is async; every
  numpy / HDF5 / optical-flow call goes to an executor. This holds for
  the sidecar's scheduler and for any client integration built on it.
- **UTC internally; convert only at the presentation boundary.** Radar
  timestamps, the service and the viewer are in three different frames;
  DST bugs are otherwise guaranteed.
- **Never hardcode HDF5 scaling.** Read `quantity`, `gain`, `offset`,
  `nodata`, `undetect` from the file. Physical value =
  `raw * gain + offset`; `raw == nodata` is missing, `raw == undetect`
  is below detection.
- **Cap reflectivity before Z–R.** `dBZ_eff = min(dBZ, 53)`;
  `R = min(R, 100)` mm/h. Without the cap a 60 dBZ hail core implies
  ~600 mm/h — nonsense.
- **Radius-based detection, not nearest pixel.** Default 1 km disc,
  tracking max / mean / P90 inside it. ~500 m pixels and fuzzy rain
  edges make a single pixel brittle.
- **Hysteresis on "raining now"; persistence on "rain incoming".** A
  two-cycle requirement keeps one-frame clutter from firing alerts.
- **`state.json` changes are additive.** Clients pin field paths; adding
  keys is free, renaming or removing them breaks deployments in the
  field. Same for `/nowcast/manifest.json`.
- **Derived data only in the served state.** Never inline raw rain-rate
  fields — reference the cached artifact path instead.
- **Poll on the DMI cadence with jitter** (5 min ± 30 s). DMI's rate
  limit is 500 req / 5 s; a 429 means back off.
- **Confidence ≠ probability.** Probability comes from the ensemble and
  is isotonic-calibrated against backtest data. Confidence is a separate
  0–1 heuristic (frame count, motion divergence, intensity volatility,
  frame age, horizon). Expose both, never conflate them.

## Committed algorithm choices

Don't relitigate these without new evidence:

- **Advection first, not deep learning.** Ayzel et al. 2020: RainNet
  loses to optical flow at ≥10 mm/h; DGMR-class models need multi-year
  national archives and TPU-scale training.
- **STEPS is vendored, not installed.** `src/dmi_nowcast_core/_vendor/
  pysteps_steps/` is a subset of pysteps 1.21.1 (BSD-3, notices kept).
  Upstream has no musl wheels and its source build wants OpenMP. Don't
  add `pysteps` as a dependency; don't reimplement what's vendored.
- **DMI's composite is column-max reflectivity, not surface rain.** It
  biases intensity high (tall convective cores, virga, bright band).
  Say so wherever intensity is surfaced.
- **`https://opendataapi.dmi.dk/v1/radardata/`** is the API host (no
  auth). The legacy `dmigw.govcloud.dk` host is retired; keep it only as
  an optional override.

## Verification discipline

A nowcast that doesn't beat persistence and Eulerian baselines on
CSI / FSS at 10–45 min isn't a nowcast. Backtest output is Parquet, one
row per `(timestamp, method, horizon)` — DuckDB reads it directly, and
`sql/` holds the standard queries.

Calibration sanity check: Imhoff et al. 2020 report pysteps
decorrelation times of 25 / 40 / 56 / 116 min for 1 / 3 / 6 / 24-hour
events on Dutch lowland catchments. Numbers substantially worse suggest
a bug; substantially better suggests a verification mistake. Expect ~3×
lower skill on summer convection than winter stratiform.

## Commands

```bash
# core library
uv sync --all-groups
uv run python -m pytest tests/ -q

# sidecar service
cd sidecar && uv sync --all-groups && uv run python -m pytest tests/ -q

# run the service locally
cp sidecar/config.example.yaml sidecar/config.yaml
cd sidecar && uv run python -m dmi_nowcast_sidecar
```

Both suites run in CI on every push (`.github/workflows/ci.yml`) and are
fully offline — no network, no DMI calls, no real STEPS runs.

## Conventions

- Configuration is YAML plus `DMI_NOWCAST_`-prefixed env vars (`__` as
  the nesting separator).
- Nothing host-specific is committed: deployment reads hosts, ports,
  paths and keys from the environment (`.env`, gitignored — see
  `.env.example`).
- The fixed calibration point set lives at
  `src/dmi_nowcast_core/calibration_points_v2.json` and is regenerated,
  not hand-edited, by `scripts/build_calibration_points.py`.
