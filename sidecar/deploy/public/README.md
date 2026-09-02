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
  archives nothing and writes only inside its own volume.
- Resource budget: a second 10-min-cadence ensemble is roughly +17 s CPU
  per 10 min and ~1 GB transient.
- Deploying to a remote host: `sidecar/deploy/deploy.sh` ships the service
  tree for the LAN stack; the public stack additionally needs
  `frontend/build/` present in the build context on that host.
