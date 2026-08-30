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

## Notes

- No corpus bind-mount: `storage.corpus_dir: null`. The public instance
  archives nothing and writes only inside its own volume.
- Resource budget: a second 10-min-cadence ensemble is roughly +17 s CPU
  per 10 min and ~1 GB transient.
- Deploying to a remote host: `sidecar/deploy/deploy.sh` ships the service
  tree for the LAN stack; the public stack additionally needs
  `frontend/build/` present in the build context on that host.
