# Public instance stack

The internet-facing sidecar: national nowcast products plus the static
site, published through a Cloudflare Tunnel. A second, fully independent
stack next to `../docker-compose.yml` — its own container, volume, config
and DMI polling. The two share only the image.

## What "public" means

`server.public_mode: true` installs a default-deny gate over the whole
route table (see `dmi_nowcast_sidecar/app.py`'s docstring):

| Surface | Reachable by anyone | Notes |
| --- | --- | --- |
| `/` + static frontend | yes | SPA fallback, cache headers per asset |
| `/healthz` | yes | also the container healthcheck |
| `/nowcast/manifest.json`, `/nowcast/*` | yes | the site's whole payload |
| `/forecast?lat=&lon=` | yes | fallback for browsers that can't decode PNGs |
| `/api/push/config`, `/api/push/subscribe`, `/api/push/unsubscribe` | yes | Web Push, subscriber-facing (see below) |
| `/api/push/test`, `/api/push/stats` | **no** | operator routes — exact paths, not the `/api/push/` prefix |
| `/state.json`, `/frames/*`, `/lightning/*`, `/docs`, `/openapi.json` | **no** | `404 {"detail":"Not Found"}`, identical to a nonexistent path |
| `/calibration/*`, `/stations/station_points.json` | **no** | the artefacts this stack *pulls* from the private one, not republishes |

The hidden routes answer normally to a request carrying
`Authorization: Bearer <server.api_key>`. With `api_key: null` (the
example's default) they are simply unreachable.

The cycle also skips the home-crop rendering and the OSM basemap fetch in
public mode — both only feed `/frames/*`, which is hidden here.

## Bring-up

Build order matters: the frontend is built by its own toolchain and the
image copies whatever `frontend/build/` contains at build time.

```bash
# 1. build the site (skip only if you want an API-only instance)
cd frontend && npm ci && npm run build && cd ..

# 2. stage the config
cp sidecar/deploy/public/config.public.example.yaml \
   sidecar/deploy/public/config.public.yaml
$EDITOR sidecar/deploy/public/config.public.yaml

# 3. build + start (HOST_PORT defaults to 8082)
docker compose -f sidecar/deploy/public/docker-compose.yml up -d --build

# 4. verify the gate, not just the health
curl -fs   http://localhost:8082/healthz            # 200
curl -s -o /dev/null -w '%{http_code}\n' \
           http://localhost:8082/state.json         # 404
curl -fs   http://localhost:8082/nowcast/manifest.json | head -c 200
```

Publishing on one interface only: set `HOST_PORT` to an explicit
`address:port` in a `.env` next to the compose file, e.g.
`HOST_PORT=10.0.0.5:8082`, so only the tunnel container can reach it.
The tunnel maps the public hostname to that address; nothing else should
talk to this container.

Logs, restart, teardown:

```bash
docker compose -f sidecar/deploy/public/docker-compose.yml logs -f
docker compose -f sidecar/deploy/public/docker-compose.yml restart
docker compose -f sidecar/deploy/public/docker-compose.yml down          # keeps the volume
```

## Calibration curves

This instance doesn't fit curves — the monthly job runs where the corpus
lives (the private instance). To publish calibrated probabilities, copy a
fitted `national_curves.json` into this stack's volume and restart:

```bash
docker cp national_curves.json dmi-nowcast-public:/var/lib/dmi-nowcast/
docker compose -f sidecar/deploy/public/docker-compose.yml restart
```

Without it the grids are served raw and reported honestly as
`calibrated: false` — never silently presented as calibrated.

## Rain gauges on the public stack

This instance decides its notifications with `p_post`, the output of a
post-processing model whose features include 21 neighbour-gauge (`ng_*`)
columns: what the rain gauges *around* a subscriber's point measured in
the last hours, placed in the cycle's own motion frame. At an address that
is not itself a gauge — which is every subscriber — those columns are the
only gauge signal in the row, and served as nulls they make a model that
was trained on them decide on 21 imputed means. So this stack computes
them, which means it needs gauge readings locally and *fresh*: the
features look back from `now - postprocess.gauge_lag_min` (10 min), so an
hourly `sync` of the private instance's parquet would be the wrong number
rather than a late one.

Two config blocks, both in `config.public.example.yaml`:

```yaml
station_obs:
  enabled: true
  store_dir: /var/lib/dmi-nowcast/gauges   # bounded — NOT an archive
  retention_days: 7
  interval_min: 10
  lookback_min: 40
  parameters: [precip_past10min, precip_dur_past10min]
  base_url: https://opendataapi.dmi.dk/v2/metObs
  api_key: null

postprocess:
  gauge_points_file: /var/lib/dmi-nowcast/stations/station_points.json
  gauge_lag_min: 10.0

sync:
  files:
    - ...                                  # the three fitted artefacts
    - stations/station_points.json
```

- `store_dir` is what makes polling legal here: config load refuses
  `station_obs.enabled` under `server.public_mode` unless it names a
  bounded store on this stack's own volume, and it may never be
  `storage.corpus_dir`. After every poll, month partitions that ended more
  than `retention_days` ago are deleted — real gauge data is ~0.3 MiB per
  month (1.4M rows at ~0.2 bytes/row), so the volume never notices. The
  readings the cycle actually reads are the last six hours of it.
- `gauge_store_dir` is left unset: it falls back to `station_obs.store_dir`,
  so the reader follows the writer and the directory is named once.
- The catalogue is the one piece this instance cannot derive. It is built
  on the private instance and served there at
  `GET /stations/station_points.json` — which resolves that instance's
  `station_eval.points_file`, so it must be set there (the scoreboard
  itself may stay off); 503 until it exists, behind `server.api_key` if
  the private instance sets one, and 404 on any public instance. `sync`
  copies it to `postprocess.gauge_points_file`. Before the first sync the
  cycle logs one `gauge_history_points_unreadable` line and publishes a
  null `ng_*` block; the cycle after the sync picks the file up with no
  restart. A catalogue *replaced* later takes a restart.
- Cost to DMI: one request per parameter per poll — 12 an hour — against a
  limit of 500 per 5 s. No API key: DMI dropped that requirement on
  `opendataapi.dmi.dk` on 2025-12-02.

Check it in the logs:

```bash
docker compose -f sidecar/deploy/public/docker-compose.yml logs \
  | grep -E 'station_obs_poll|gauge_history_points_loaded|postprocess_cycle'
```

`postprocess_cycle` carries `ng_frame_ok`, `ng_near_km` and
`ng_upwet_tau_min`: how many points the neighbour block answered for. All
three at zero with a loaded catalogue means the store is empty or
unreadable, not that it is dry outside.

## Web Push

`config.public.example.yaml` ships with `push.enabled: true`. Before
deploying, change the one placeholder — `push.vapid_subject`, the operator
contact push services see in the VAPID JWT (`mailto:` or `https:`). The
service refuses to start with push enabled and no subject.

Either edit the staged config or set it in the environment (env beats
YAML):

```bash
DMI_NOWCAST_PUSH__ENABLED=true
DMI_NOWCAST_PUSH__VAPID_SUBJECT=mailto:operator@example.com
```

The VAPID private key and the subscription database live in this stack's
named volume:

```
/var/lib/dmi-nowcast/push/vapid_private.pem       0600, generated on first start
/var/lib/dmi-nowcast/push/subscriptions.sqlite
```

Losing the volume loses both, and losing the key forces every subscriber
to re-subscribe (their `applicationServerKey` no longer matches). Back the
two up together:

```bash
docker compose -f sidecar/deploy/public/docker-compose.yml \
  exec dmi-nowcast-public tar -C /var/lib/dmi-nowcast -cf - push > push-backup.tar
```

To pin a key you already have, generate it first and copy it in before the
first boot:

```bash
uv run --package dmi-nowcast-sidecar \
  python -m dmi_nowcast_sidecar.push.keygen ./vapid_private.pem
```

`GET /api/push/config`, `POST /api/push/subscribe` and
`POST /api/push/unsubscribe` are on the public allow-list. The two
operator routes are not, so they need `server.api_key` set and the bearer
presented — without it they are `404`, like every other hidden route:

```bash
curl -fsS -X POST http://localhost:8082/api/push/test \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' -d '{}'     # all subscriptions
curl -fsS http://localhost:8082/api/push/stats \
  -H "Authorization: Bearer $API_KEY"
```

Set `push.enabled: false` to turn the feature off; the routes then answer
`503` and the cycle skips the evaluation entirely.

## Notes

- No corpus bind-mount: `storage.corpus_dir: null`. The public instance
  archives nothing and writes only inside its own volume — the bounded
  gauge store under `station_obs.store_dir` included, which is why
  retention keeps it a working set and not an archive.
- Resource budget: a second 10-min-cadence ensemble is roughly +17 s CPU
  per 10 min and ~1 GB transient.
- Deploying to a remote host: `sidecar/deploy/deploy.sh` ships the service
  tree for the LAN stack; the public stack additionally needs
  `frontend/build/` present in the build context on that host.
