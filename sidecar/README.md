# dmi-nowcast-sidecar

Standalone HTTP service that runs the radar nowcasting compute
(`dmi_nowcast_core`) on a schedule and serves the results as JSON and
PNG artifacts. Clients — a web frontend, a home-automation integration,
anything that speaks HTTP — stay thin.

It runs on a Linux glibc host (a small VM is plenty) because
`opencv-python-headless` installs cleanly from PyPI there; musl-based
images (Alpine) have no wheels for it.

## Layout

- `dmi_nowcast_sidecar/` — application package
  - `app.py` — FastAPI app + routes
  - `config.py` — pydantic-settings + YAML loader
  - `logging_setup.py` — structlog (pretty in dev, JSON in prod)
  - `__main__.py` — uvicorn entrypoint (`python -m dmi_nowcast_sidecar`)
- `config.example.yaml` — copy to `config.yaml` and edit
- `deploy/` — Docker Compose + systemd assets
- `tests/` — pytest

## Quick start (dev)

From the repo root:

```bash
uv sync                                # installs both core lib and sidecar
cp sidecar/config.example.yaml sidecar/config.yaml
uv run --package dmi-nowcast-sidecar python -m dmi_nowcast_sidecar
# → service listens on http://0.0.0.0:8081
curl http://localhost:8081/healthz
```

Run tests:

```bash
uv run --package dmi-nowcast-sidecar pytest sidecar/tests
```

## Config

YAML file, path picked up from the `DMI_NOWCAST_CONFIG` env var
(defaults to `./config.yaml`). Any field can also be set via env vars
with the prefix `DMI_NOWCAST_` and `__` as nested separator — e.g.
`DMI_NOWCAST_SERVER__PORT=9000`. See `config.example.yaml` for the
full schema.

## Auth

`server.api_key` is optional. When unset, all endpoints are open —
only safe on a network you trust. When set, requests to **write**
endpoints (currently `/v1/trigger-refresh`) must include
`Authorization: Bearer <key>`. Read endpoints (`/healthz`,
`/state.json`, `/frames/*`) stay open so clients can poll them
unauthenticated.

## Deployment

`deploy/` ships a Dockerfile, a Compose file and an SSH deploy script.
Everything host-specific is read from environment variables — copy
`.env.example` at the repo root to `.env` and fill in your own host:

```bash
DEPLOY_SSH_HOST=...        # the host running docker
DEPLOY_SSH_PORT=22
DEPLOY_SSH_USER=...
DEPLOY_SSH_KEY=~/.ssh/your_key
```

Then, from the repo root:

```bash
sidecar/deploy/deploy.sh             # build image + restart the container
sidecar/deploy/deploy.sh --no-build  # config-only restart, skip the build
sidecar/deploy/deploy.sh --logs      # tail container logs after deploy
```

The service listens on port 8081. Publish it only where you mean to:
the Compose file binds the published port on the host, and
`server.api_key` (below) is the only auth there is.

### Persistent corpus archive

The sidecar archives every fetched composite into a host-bind-mounted
directory that survives `docker compose down -v`. Layout on the host
(the path comes from `CORPUS_HOST_DIR`, default `~/dmi-nowcast-corpus`):

```
$CORPUS_HOST_DIR/                         host bind mount, uid 10001
  composites/YYYY/MM/dk.com.YYYYMMDDhhmm.500_max.h5
  manifest.parquet                       built by scripts/build_corpus_manifest.py
```

The host side defaults to the deploy user's home
(`~/dmi-nowcast-corpus`, overridable with `CORPUS_HOST_DIR`) and is
bind-mounted to `/var/lib/dmi-nowcast-corpus` *inside* the container —
the container-internal path that `storage.corpus_dir` points at. Home
placement means provisioning needs no host `sudo`.

The working cache at `/var/lib/dmi-nowcast/composites/` (named volume) is
LRU-evicted each cycle so disk stays bounded by `storage.working_cache_max_bytes`
(default 500 MB). The corpus is the source of truth; the cache is just a
short-lived buffer.

`deploy.sh` creates the host directory and chowns it to uid 10001 (the
unprivileged `dmi` user inside the container) using a throwaway root
`busybox` container — the docker daemon is root, so no host `sudo` is
needed and the deploy stays fully unattended. To disable archiving set
`storage.corpus_dir: null` in `config.yaml`.

Scripts are not baked into the runtime image (the image stays lean) — they're
mounted in on demand the same way `calibrate.sh` does it.

#### One-time backfill from DMI's 180-day archive

```bash
sidecar/deploy/backfill_corpus.sh                  # default 180-day window
DAYS_BACK=30 sidecar/deploy/backfill_corpus.sh     # last 30 days only
```

Resumable — re-running picks up where a previous run left off. At
~80 KB/frame × 288 frames/day × 180 days ≈ 4 GB total; expect ~10–20
minutes on a decent connection.

#### Build the frame manifest

```bash
sidecar/deploy/build_corpus_manifest.sh            # incremental
sidecar/deploy/build_corpus_manifest.sh --rebuild  # re-parse every frame
```

Writes `<corpus>/manifest.parquet` — one row per frame with `wet_fraction`,
`heavy_fraction`, `max_rain_mm_h`, etc. Incremental by default (reuses
existing rows by path); pass `--rebuild` to re-parse every frame.

### Monthly recalibration

A systemd timer rebuilds the calibration corpus and refits curves once
a month. The new curves land in the data volume at
`/var/lib/dmi-nowcast/calibration_curves.json`, where the running
sidecar picks them up immediately.

Install on the deploy host (one-time). The unit ships with
`__DEPLOY_USER__` / `__DEPLOY_DIR__` placeholders — substitute your own
user and checkout path:

```bash
# as root on the deploy host, with USER/DIR set to your own:
sed -e "s|__DEPLOY_USER__|$USER|g" -e "s|__DEPLOY_DIR__|$DIR|g" \
    "$DIR/sidecar/deploy/dmi-calibrate.service" \
    > /etc/systemd/system/dmi-calibrate.service
cp "$DIR/sidecar/deploy/dmi-calibrate.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now dmi-calibrate.timer
systemctl list-timers dmi-calibrate.timer    # confirm next-run
```

Run on demand (e.g. after a code change):

```bash
sidecar/deploy/calibrate.sh                       # default 12-month window
CALIBRATION_INPUT_MONTHS=6 sidecar/deploy/calibrate.sh
```

Notes:

- The job uses `scripts/build_calibration_corpus.py` +
  `scripts/fit_national_calibration.py` from the repo. The deploy script
  copies those onto the host; `calibrate.sh` mounts the repo into the
  sidecar container read-only and runs the scripts there.
- Wet/dry stratification uses the corpus builder's five spread national
  reference points by default; override with `CALIBRATION_WET_REFS`.
- Archived raw frames aren't pruned by the timer — only the corpus
  Parquet is regenerated each month.
- A new corpus Parquet lands at
  `/var/lib/dmi-nowcast/calibration_corpus_<YYYYMMDD_HHMMSS>.parquet` so
  past runs aren't overwritten.
