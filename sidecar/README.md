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

## Public mode

`server.public_mode: true` turns the process into the internet-facing
instance: only the static frontend (`server.frontend_dir`), `/healthz`,
`/nowcast/*` and `/forecast` are served. Everything else — `/state.json`
(the configured point's block), `/frames/*`, `/lightning/*`, `/docs` —
answers `404`, indistinguishable from a route that was never registered,
unless the request carries the `api_key` bearer. The cycle also skips the
home-crop rendering and the OSM basemap fetch, which only feed the hidden
`/frames/*`. The default (`false`) leaves this LAN service unchanged.

Deployment assets for that mode live in `deploy/public/` — see its README.

## Web Push

`push.enabled: true` turns on browser notifications: a visitor subscribes
with a point, a probability threshold and a lead time, and the service
pushes them once when the calibrated probability at that point holds over
the threshold for two consecutive radar observations. Off by default — the
LAN instance has Home Assistant for alerting.

Enable it in `config.yaml`:

```yaml
push:
  enabled: true
  vapid_subject: mailto:you@example.com   # required when enabled
```

or through the environment, which is what the compose stacks do:

```bash
DMI_NOWCAST_PUSH__ENABLED=true
DMI_NOWCAST_PUSH__VAPID_SUBJECT=mailto:you@example.com
```

`vapid_subject` must be a `mailto:` or `https:` operator contact; the
service refuses to start without one when push is enabled.

### Where the state lives

Both files sit in the data volume under `storage.data_dir/push/`
(override with `push.vapid_private_key_file` / `push.db_path`):

```
<data_dir>/push/vapid_private.pem        0600, generated on first start
<data_dir>/push/subscriptions.sqlite     endpoint, keys, point, preferences
```

The PEM is the service's identity. Rotate it and every existing
subscription's `applicationServerKey` stops matching, so every subscriber
has to re-subscribe — back it up together with the SQLite file. Neither is
ever committed (see the repo `.gitignore`). The table holds no email, no
name and no IP: an endpoint, its two keys, the coordinate the subscriber
asked about, their preferences and the per-subscription state machine.

To create the key ahead of the first boot (provisioning a volume, or
pinning one identity across two instances):

```bash
uv run --package dmi-nowcast-sidecar \
  python -m dmi_nowcast_sidecar.push.keygen ./vapid_private.pem
# prints the public key; refuses to overwrite without --force
```

### Routes

| Route | Auth | Purpose |
| --- | --- | --- |
| `GET /api/push/config` | open | feature flag, VAPID public key, option lists |
| `POST /api/push/subscribe` | open | store/replace one subscription |
| `POST /api/push/unsubscribe` | open | forget one endpoint (idempotent) |
| `POST /api/push/test` | **bearer** | send the canned test notification |
| `GET /api/push/stats` | **bearer** | counts + the last fan-out summary |

The first three are on the public-mode allow-list; the last two are not,
so on a public instance they answer `404` without the bearer. All five
answer `503` while the feature is disabled.

```bash
# API_KEY is server.api_key (env: DMI_NOWCAST_SERVER__API_KEY).
# Name one endpoint, or omit the body to reach every subscription.
curl -fsS -X POST http://localhost:8081/api/push/test \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"endpoint": "https://fcm.googleapis.com/fcm/send/..."}'
# {"sent":1,"failed":0,"removed":0}

curl -fsS http://localhost:8081/api/push/stats \
  -H "Authorization: Bearer $API_KEY"
# {"subscriptions":12,"armed":11,"last_evaluated_radar_ts":"...","last_fanout":{...}}
```

Subscriptions the push service reports as gone (404/410) are deleted
during the fan-out — that is the only garbage collection the store has.

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
sidecar/deploy/calibrate.sh                       # default: the whole archive
CALIBRATION_INPUT_MONTHS=6 sidecar/deploy/calibrate.sh   # fixed 6-month window
```

Notes:

- The window defaults to `CALIBRATION_INPUT_MONTHS=all`, which passes
  `--days-back 0`: events are sampled from the oldest archived fullRange
  frame onwards. The corpus builder lists frames from the persistent
  archive first and only falls back to DMI's items API for windows the
  archive does not hold, so the calibration window is bounded by the
  archive's depth, not by DMI's 180-day listing horizon — the archive
  gains a month every month. Set a number of months for a shorter, fixed
  window. The actual window is printed by the job and recorded in the
  progress JSON (`event_window`); it is deliberately not part of the
  corpus `settings_hash`, so a corpus can be extended backwards across
  runs.
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
