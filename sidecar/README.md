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
  - `workers.py` — one thread per heavy blocking job (see below)
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

## Observed rain (`observed_mm_h`)

Every other national product is a forecast, and none of them can answer
"is it raining here *right now*": the STEPS ensemble's first timestep is
already ~10 minutes ahead of the radar image, so a point under a shower
that clears within those 10 minutes and gets the next cell at +30 reads an
ETA of ~16 min — a fresh arrival, while it is raining on you.

So each cycle also publishes the OBSERVED rain rate from the newest
composite, reduced onto the same ×4 product grid by a block-wise 90th
percentile (`dmi_nowcast_core.national.observed_rain_grid`). p90 over a
2 × 2 km block mirrors the Home Assistant `raining_now` rule and, on a
column-max composite, keeps one clutter or virga pixel from making a block
look wet.

It appears in three places, all additive:

- `/nowcast/observed_mm_h_<cycle>.png` — grayscale8, quantised with the
  *same* scale/offset as `intensity` (0–100 mm/h, 255 = nodata), with a
  manifest entry carrying `"product": "observed_mm_h"`, `"lead_min": 0`
  (it depicts `radar_ts_utc` itself) and `"units": "mm/h"`. Manifest
  schema stays at v2 — a client that doesn't know the product ignores the
  entry.
- `/forecast?lat=&lon=` — a new `observed_mm_h` field, null when the pixel
  is nodata or the cycle published no observed grid.
- the Web Push decision engine — a subscription whose point is measured at
  or above `forecast.rain_threshold_mm_h` is treated as "already raining":
  the arm is consumed silently instead of sending "rain incoming" into
  falling rain.

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

## The quality report (`/nowcast/quality.json`)

The document behind the website's /quality page: reliability against
radar and gauge truth, the warning scoreboard, the margin over
persistence, per-station scores. Its contract is
`frontend/src/lib/quality/schema.ts`, and every top-level section is
nullable — a section whose evidence is not on disk comes out `null`, and
the page says "not measured yet" rather than showing a zero.

Two instances, two different jobs:

| | private (this config) | public (`deploy/public/`) |
|---|---|---|
| `quality_report.enabled` | `true` — builds it nightly at `at_utc` | refused by config load |
| `sync.enabled` | `false` — nothing to pull | `true` — pulls it, and the curves |
| `GET /nowcast/quality.json` | serves the file it built | serves the file it pulled |
| `GET /calibration/national_curves.json` | serves the live curves | hidden by the public gate |

The builder is `dmi_nowcast_core.quality_report`; every input path is
optional and a missing one nulls only its own section, so the feature can
be turned on before the whole corpus exists.

**It runs out of process.** The nightly task spawns
`python -m dmi_nowcast_sidecar.quality_job` with the resolved paths as
one JSON argument, waits for it under `quality_report.timeout_s` (default
1800 s) and reads a one-line JSON summary from its stdout; the manual
`deploy/quality_report.sh` runs the same module in a throwaway container,
so the two paths cannot drift. The reason is memory: the build pulls both
calibration corpora, every replay and live decision row and the gauge
store behind them into Arrow and numpy, and neither Arrow's pool nor
CPython's allocator returns that to the kernel — built in a thread of the
service it took RSS from ~0.9 GB to 5.5 GB and the host's OOM killer took
the service down (exit 137) while a batch replay ran beside it. A child
process gives the memory back by exiting, and a build that crashes,
hangs, or is itself OOM-killed now costs one log line and a day-old
report. The compose files add `oom_score_adj: -500` and a
`mem_reservation` so that, if the host runs short anyway, the kernel
picks the batch jobs first. Only one thing stays in the parent: after a
successful threshold fit the running service is told to re-read its
table.

Build the FIRST report by hand — the route 503s until a document is on
disk, and waiting until 03:30 to find out a path was wrong is a poor way
to learn it:

```bash
sidecar/deploy/quality_report.sh
curl -fs http://localhost:8081/nowcast/quality.json | head -c 400
```

Nothing needs restarting: the route reads the file on every request, and
the public instance picks up both files on its next `sync` interval. A
freshly synced `national_curves.json` is re-read at the start of the next
radar cycle, so the monthly fit now reaches the public instance without a
restart too.

When the private peer is unreachable the public instance keeps the last
good copy of each file and logs one line per failure. A stale report is
honest — it carries its own `generated_at_utc` — where a blanked one
would not be.

## Where the blocking work runs

Every heavy job in this service is off the event loop. `workers.py` says
*which thread* it goes to, and that turns out to matter as much.

`asyncio.to_thread` submits to the loop's shared default executor, which
grows a worker whenever a job arrives and none is idle. glibc gives each
allocating thread its own malloc arena and never returns an arena's
high-water mark to the kernel — so a job with a gigabyte-scale transient
costs that much RSS **for every thread it has ever run on**.

Until Phase F the radar cycle was this process's only `to_thread` caller,
so STEPS always ran on thread #1 and the service sat flat at ~0.9 GB for
days. Phase F added two more callers on unrelated cadences — the gauge
poller (10 min, its own scheduler) and the station scoreboard (after each
cycle) — which overlap the 5-min cycle often enough that the pool grows,
and each new worker that takes a turn at the cycle inflates another arena
to the STEPS high-water. The private instance started climbing ~1 GB an
hour and needed an hourly restart; the public one, which runs neither new
task, never moved.

So the three heavy jobs each get a **named single-worker pool** and always
run there: `cycle` (`CycleEngine._compute_sync`), `station_eval` and
`station_obs` (the two month-partition rewrites). Small jobs — a file
copy, a SQLite read, a PNG, a JSON write — stay on `asyncio.to_thread`,
where the pool's elasticity is worth having and costs nothing.

Measured on `python:3.12-slim`, the live wiring driven 34 cycles:

| | compute ran on | RSS |
|---|---|---|
| shared default executor | 3 threads | 2.5 GB, still stepping |
| per-job pools | 1 thread | 1.19 GB, flat from cycle 13 |

Two things ride along with the same discipline. `release_arrow_pool()`
is called after each partition rewrite in both tasks, because Arrow's
pool otherwise keeps the largest month it ever built. And the compose
files set `MALLOC_ARENA_MAX=2` as the belt-and-braces underneath the
pools, so a future thread nobody thought about cannot quietly buy another
arena.

`DMI_NOWCAST_WORKER_DEBUG=1` logs the pool, the worker thread count and
the process RSS after every pooled job — off by default, and the
instrument to reach for first if RSS ever climbs again.

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

### Rain-gauge truth (Phase F)

The corpus's `outcome` column is verified against the radar composite —
the same instrument the forecast is made from. `station_obs` adds an
independent truth: DMI's metObs rain gauges, mirrored into the same
corpus volume.

```
$CORPUS_HOST_DIR/
  stations/
    catalogue.parquet                    294 stations, current version of each
    obs/YYYY/MM.parquet                  one row per (station, parameter, stamp)
```

Turn it on in `config.yaml` (off by default, and **refused** together with
`server.public_mode` — the public stack has no corpus mount):

```yaml
station_obs:
  enabled: true
  interval_min: 10       # poll cadence
  lookback_min: 40       # re-read window; must be >= interval_min
  parameters: [precip_past10min, precip_dur_past10min]
```

The 40-minute lookback deliberately overlaps four polls: DMI backfills
late station reports into slots that already passed, and the store dedupes
on `(station_id, observed_utc, parameter_id)`, so a missed poll heals
itself on the next one.

History (the poller only collects from the moment it starts) comes from a
one-time backfill — one request per day per parameter, at 2 req/s:

```bash
python scripts/backfill_station_obs.py \
    --corpus-dir /var/lib/dmi-nowcast-corpus \
    --from 2026-06-01 --to 2026-09-05 \
    --progress /var/lib/dmi-nowcast-corpus/stations/backfill_progress.json
```

Then pick the stations with usable coverage, and join their gauges onto a
corpus built over those same points:

```bash
python scripts/build_station_points.py \
    --corpus-dir /var/lib/dmi-nowcast-corpus \
    --from 2026-06-01 --to 2026-09-05 --min-coverage 0.8 \
    --out station_points.json --availability-md station_availability.md

python scripts/join_gauge_truth.py \
    --corpus reports/station_corpus.parquet \
    --corpus-dir /var/lib/dmi-nowcast-corpus \
    --out reports/station_corpus_gauge.parquet
```

`sql/reliability_gauge.sql` and `sql/reliability_radar_vs_gauge.sql` read
the result. Gauge data is DMI Open Data, licence **CC BY 4.0** — attribute
DMI in anything published from it. API keys are not required on
`opendataapi.dmi.dk` (dropped 2025-12-02); fair use still applies.

### Monthly routine

`calibrate.sh` **is** the routine, and the existing `dmi-calibrate`
systemd timer already runs it: **1st of the month, 03:00 UTC** (the VM's
clock is UTC) plus up to 15 minutes of `RandomizedDelaySec`. There is no
second scheduler to install — one timer, one entry point.

**One corpus build serves both point sets.** `--points` is repeatable and
a single STEPS run per event already feeds every point in the union, so
the ~120 radar calibration points and the DMI gauge stations come out of
one ~3.5 h pass rather than two:

```
build   --points src/dmi_nowcast_core/calibration_points_v2.json \
        --points <corpus>/stations/station_points.json
fit     --point-set calibration_points_v2      → national_curves.json
report  --point-set calibration_points_v2      → calibration_reports/<stamp>/
join    --point-set station_points             → station_corpus_gauge.parquet
```

Every row carries a `point_set` column naming the points file it came
from, which is what lets one corpus be split back into the two it
replaces. `calibration/latest.parquet` is the **whole union**; the quality
report's radar section filters it to `calibration_points_v2` itself
(`RADAR_POINT_SET` in `src/dmi_nowcast_core/quality_report.py`), because
the station rows belong to the gauge section and are read there through
the gauge-joined file.

The gauge join runs **after** the restart, so the new curves are in
service within minutes, and it is never fatal: the curves are the product,
the gauge corpus is a report input. A missing
`stations/station_points.json` logs one line and the run continues with
radar points only; `CALIBRATION_STATIONS=0` drops the station points and
the join deliberately; `CALIBRATION_UNION=0` falls back to the old
two-build path (radar corpus, then `station_corpus.sh`) if a union run
ever misbehaves. The job's exit code reflects one thing only — whether the
sidecar came back serving a newer `calibration_fitted_at` — so
`systemctl --failed` means "the curves are not live", not "the gauge
column is stale".

Wall time is therefore ~3.5 h plus a join of minutes, finishing around
06:30 UTC.

The run overlaps the 03:30 UTC nightly quality report, which is harmless:
the report is a child process of the sidecar, the batch work is in a
separate capped container, and both stable filenames are published
atomically (temp name, then rename) so the report can never read a
half-written parquet.

Nothing needs restarting afterwards.

#### Memory rules (read before raising any worker count)

The VM has 12 GB shared between the live sidecar, the public stack and
whatever batch job is running. A STEPS worker (16 members, 432×496) holds
1.3–2.0 GB of anonymous RSS. Uncapped `docker compose run` containers with
3–5 workers got the **live** sidecar chosen by the kernel's global OOM
killer 18 times over 2026-09-05/06 — `oom_score_adj: -500` biases that
choice but cannot survive a machine that is genuinely out of memory.

`sidecar/deploy/lib/batch.sh` encodes what works, and every batch script
sources it:

- **`BATCH_WORKERS=2`.** Not five. Three is the next thing to try — peak
  RSS per worker is ~1.65 GB after the September 2026 STEPS work, so
  3 × 1.65 GB sits right at the cap — but 2 ships until a real monthly run
  confirms it, because the measurement that matters is a 3.5 h build
  beside the live sidecar, not a benchmark.
- **`BATCH_MEM_CAP=5000m`**, applied with `docker update` to the
  `deploy-sidecar-run-*` container ~25 s after it starts. Under the cap the
  *cgroup's* OOM killer fires first and takes a batch worker (the pool
  restarts it) instead of the global one taking the service. The cap cannot
  live in `docker-compose.yml`: it must land on the throwaway container,
  and compose gives it and the service the same definition.
- **One batch job at a time.** Every script refuses to start beside a
  running `deploy-sidecar-run-*` container. `BATCH_FORCE=1` overrides it;
  don't.

The nightly quality report at 03:30 UTC is a *child process of the
sidecar*, not a batch container, so the guard does not see it and it will
land in the middle of step 1. That is survivable exactly because the batch
container is capped — the report is the reason the cap exists.

#### Where the outputs go

```
/var/lib/dmi-nowcast-corpus/
  calibration/national_corpus_<stamp>.parquet   this run's UNION corpus (both point sets)
  calibration/latest.parquet                    a COPY of it — quality_report.radar_corpus
  calibration/latest.md                         which run, which point sets, which report
  calibration_reports/<stamp>/                  reliability report (radar point set)
  stations/station_corpus_<stamp>_gauge.parquet the station rows, joined to gauge truth
  stations/station_corpus_gauge.parquet         a COPY of it — quality_report.station_corpus
/var/lib/dmi-nowcast/national_curves.json       the served curves (radar point set)
```

`latest.parquet` and `station_corpus_gauge.parquet` are **copies, not
symlinks**, published **atomically** (copy to a temp name in the same
directory, then rename). Both are read through a docker bind-mount by a
different container than the one that writes them, and by the report's
child process, which resolves the configured path itself; a symlink into a
stamped sibling breaks silently when the target moves or is pruned, and a
plain `cp` onto a live path lets the 03:30 UTC report open a truncated
parquet. A parquet is tens of MB — a copy once a month is the boring
option.

#### By hand

The timer's output goes to journald; a hand-started run should keep its
own log, because it is seven hours you do not want to re-run blind.

```bash
mkdir -p ~/dmi-nowcast-logs
sidecar/deploy/calibrate.sh 2>&1 \
    | tee ~/dmi-nowcast-logs/calibrate-$(date -u +%Y%m%d_%H%M%S).log

CALIBRATION_STATIONS=0 sidecar/deploy/calibrate.sh        # curves only
CALIBRATION_UNION=0 sidecar/deploy/calibrate.sh           # two separate builds
CALIBRATION_INPUT_MONTHS=6 sidecar/deploy/calibrate.sh    # fixed 6-month window

sidecar/deploy/station_corpus.sh                          # redo the gauge half
STATION_REUSE_CORPUS=0 sidecar/deploy/station_corpus.sh   # force a rebuild
```

`station_corpus.sh` is now the *gauge half on its own* — run it after a
gauge backfill, or when the join failed and the curves did not. It joins
`calibration/latest.parquet` when that corpus holds `station_points` rows
(seconds), and only builds a station-only corpus (~3.5 h) when it does
not.

Every script refuses to start beside a running batch container, so a hand
run on the 1st cannot collide with the timer's.

#### How to check it worked

```bash
journalctl -u dmi-calibrate.service -n 40 --no-pager      # or the tee'd log
docker exec dmi-nowcast-sidecar cat /var/lib/dmi-nowcast-corpus/calibration/latest.md
curl -fs http://localhost:8081/state.json | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["probabilistic"]["calibration_fitted_at"])'
curl -fs http://localhost:8081/nowcast/quality.json | head -c 400   # after 03:30 UTC
```

The job's last lines summarise both steps. It fails loudly if the
restarted sidecar does not serve a newer `calibration_fitted_at`, so a
green run means the curves are live. The quality report picks the new
corpora up on its next nightly run; `sidecar/deploy/quality_report.sh`
builds one immediately.

### Warning replays (only after a pipeline change)

`replay.sh` (gauge stations) and `radar_replay.sh` (the 120 fixed radar
points) re-derive warning decisions over a list of past days:

```bash
sidecar/deploy/replay.sh                                   # ~/replay_days.txt
REPLAY_DAYS_FILE=~/august.txt sidecar/deploy/replay.sh
sidecar/deploy/radar_replay.sh                             # the cross-check
```

These are **not** part of the monthly routine. The live gauge scoreboard
(`station_eval`, one row per gauge per cycle) accumulates the same decision
rows continuously and is strictly better evidence, because it is what the
service actually served. Reach for a replay only when the history has to be
re-derived under changed code: the push decision rule / thresholds / onset
definition moved, the ensemble settings changed, the curves were refit in a
way that shifts served probability, or a season of evidence is needed now
and the live scoreboard is young.

Output goes to `<corpus>/stations/replay` (`quality_report.replay_dir`) and
`<corpus>/points/replay`; the replayed and live rows are scored as one
table, deduplicated on `(radar_ts, station_id)` with the live row winning.
Both honour `BATCH_WORKERS` / `BATCH_MEM_CAP` and the one-at-a-time guard.

### Installing the monthly timer

The schedule for the routine above: `dmi-calibrate.timer` runs
`calibrate.sh` — both steps — on the 1st at 03:00 UTC, `Persistent=true`
so a host that was off catches up. Already installed and active on the
VM; this is for a rebuild or a second host.

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

Running it by hand is covered under "Monthly routine" above.

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
  Parquets are regenerated each month.
- The unit's `ExecStart` is `calibrate.sh`, which now covers both steps of
  the routine. Nothing in the unit file changed when step 2 was added, and
  nothing needs to.
- A new corpus Parquet lands at
  `<corpus>/calibration/national_corpus_<YYYYMMDD_HHMMSS>.parquet` so past
  runs aren't overwritten, and `calibration/latest.parquet` is refreshed to
  a copy of it — that stable name is what `quality_report.radar_corpus`
  reads.
