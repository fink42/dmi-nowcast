# frontend — the public web app

A SvelteKit (static adapter) PWA: a map of Denmark with the animated
radar/nowcast overlay, and a click-anywhere forecast panel that samples the
sidecar's national grids **in the browser**. Danish and English. The build
output is a folder of static files that the sidecar serves itself, so the whole
site — app, basemap, glyphs, forecast data — comes from one origin.

```bash
npm ci
node scripts/fetch-basemap.mjs          # one-time: the self-hosted basemap
npm run dev                             # http://localhost:5173
npm run test                            # vitest units
npm run build                           # → build/
```

## Where the data comes from

The app only ever uses same-origin paths: `/nowcast/manifest.json`,
`/nowcast/<artifact>.png` and `/forecast?lat=&lon=`. In production that is
literally true — the sidecar serves both the app and the data. For local
development, point the dev server at a sidecar and it proxies those paths:

```bash
VITE_SIDECAR_URL=http://192.0.2.10:8081 npm run dev
```

No sidecar at hand? A real one needs three consecutive DMI radar frames before
it serves anything, so there is a stand-in with synthetic weather — two rain
blobs over Copenhagen and Aarhus drifting east-north-east, quantised exactly
the way the sidecar quantises:

```bash
node scripts/mock-sidecar.mjs --port 8099
VITE_SIDECAR_URL=http://localhost:8099 npm run dev
```

Because the blobs sit at known coordinates, the mock also makes
georeferencing mistakes obvious: a blob that is not over Copenhagen means the
overlay is misplaced.

It mirrors the sidecar's two lead lists rather than collapsing them into one:
`OVERLAY_LEADS` (10/20/30/40/50/60, the sidecar's `forecast.leads_min`) is the
forecast overlay frames the timeline scrubs through, on the radar's own 10 min
cadence, and `PROB_LEADS` (10/20/30/45/60, `forecast.national.leads_min`) is
the calibrated probability grids, `leads_min`, `calibrated_leads` and
`/forecast`'s `per_lead`. The ETA and intensity grids scan a dense 5 min
`ARRIVAL_SCAN` instead — that is a search for when rain first reaches a pixel,
not a published lead list.

`VITE_API_BASE` exists as a build-time escape hatch for pointing a deployed
build at a *different* origin (that origin then has to send CORS headers).
Leave it unset for the normal one-origin deployment.

## The basemap (deploy step)

`static/basemap.pmtiles` is **not committed** — it is a build input, fetched by
a script and gitignored:

```bash
node scripts/fetch-basemap.mjs               # ~147 MB at maxzoom 12 (default)
node scripts/fetch-basemap.mjs --maxzoom 11  # ~73 MB, less street detail
node scripts/fetch-basemap.mjs --force       # refetch (monthly-ish is plenty)
```

**Deployment must run this script before `npm run build`.** It does two things:

1. slices a Denmark bbox (7.4–15.5 E, 54.3–58.1 N — Bornholm and Skagen
   included) straight out of the Protomaps daily planet build over HTTP range
   requests, using the `pmtiles` CLI (downloaded from the go-pmtiles releases
   into `.cache/` unless one is on `PATH`; the `pmtiles` npm package is a
   browser library, not a CLI);
2. downloads the Noto Sans glyph PBFs and the Protomaps v4 sprites into
   `static/basemap-assets/` (~11 MB), so the style needs nothing off-origin at
   runtime.

Both are gitignored. Nothing in the app talks to a third-party tile server.

## Layout

```
src/lib/i18n/          da.ts + en.ts catalogs, types.ts, state.svelte.ts
src/lib/nowcast/       manifest types, PNG decode, grid sampling, timeline,
                       cell-motion bearings, store
src/lib/push/          support detection, prefs + storage, VAPID keys, API,
                       payload parsing, store
src/lib/map/           MapLibre style, Mercator resampling, overlay frames
src/lib/components/    MapView, LoopControls, ForecastPanel + panel.ts (its
                       fold rule), NotifyPanel, LangToggle, footer
src/routes/            map (/) + about, data, privacy, support
scripts/               fetch-basemap, make-icons, mock-sidecar
```

Three pieces deserve a note:

- **Sampling (`src/lib/nowcast/sampler.ts`)** mirrors `/forecast` exactly:
  `col = (x - x_ul) / pixel_scale`, `row = (y_ul - y) / pixel_scale`, round to
  the nearest pixel, `value = level * scale + offset`, level 255 → *null*, and
  outside the grid is **off-coverage**, never a 0 % probability. `sampler.test.ts`
  pins this against numbers produced by pyproj on the server side.
  One of the sampled grids, `observed_mm_h`, is a measurement rather than a
  forecast — the radar's own rain field, block-p90'd onto the product grid —
  and the panel leads with it: at or above 0.5 mm/h (the sidecar's
  `forecast.rain_threshold_mm_h`) the headline is "it is raining here now",
  whatever the ETA product says about the next cell behind this one.
- **The overlay is resampled, not stretched (`src/lib/map/warp.ts`).** MapLibre
  interpolates an image source's four corners linearly in Mercator; the radar
  grid is polar stereographic and a thousand kilometres across, and the two do
  not differ by an affine map. Measured on the real composite geometry, the
  four-corner shortcut misplaces rain by up to **15 km inside Denmark**. Each
  frame is therefore resampled through a control mesh (≈ 40 px cells, error
  under 100 m) into a Mercator-aligned canvas, which MapLibre can then place
  exactly. `warp.test.ts` holds both numbers.
- **The PNG decoder (`src/lib/nowcast/png.ts`)** is ~120 lines instead of a
  canvas readback, because canvas colour management can shift a level and a
  level is a physical value. It also runs in Node, so the sampling maths is
  testable for real.

## PWA

`static/manifest.webmanifest` plus `src/service-worker.ts`. The worker
precaches the app shell and **never** caches `/nowcast/*`, `/forecast` or
`/api/*` (a stale rain map is worse than none), nor the basemap archive (too
big, and it is range-requested). It also carries the push and
`notificationclick` handlers — see below. Icons are generated from one raindrop
path: `node scripts/make-icons.mjs`.

## Notifications

Opt-in Web Push, per point. The pure half lives in `src/lib/push/`: `support.ts`
(can this browser do it), `prefs.ts` (threshold / lead / quiet hours, the
localStorage copy, the POST body), `keys.ts` (VAPID base64url → bytes),
`api.ts` (`/api/push/config`, `subscribe`, `unsubscribe`) and `notification.ts`
(payload → notification, notification URL → map point). `store.svelte.ts` holds
the state the UI reads, and `src/lib/components/NotifyPanel.svelte` sits at the
bottom of the forecast panel for a point that has a forecast.

The server owns the options: which probability thresholds and lead windows may
be chosen comes from `/api/push/config`, and the constants in `prefs.ts` only
cover the moment before that response lands. A sidecar without the feature
answers that path with the app shell, which reads as "push disabled" rather
than an error.

The service worker gains a `push` handler (`showNotification`) and a
`notificationclick` handler that focuses an existing tab and messages it
`{ type: 'open-point', lat, lon }`, or opens `/?lat=&lon=` if there is none —
which the map page also handles on a cold start.

`scripts/mock-sidecar.mjs` serves `/api/push/config`, `/api/push/subscribe` and
`/api/push/unsubscribe` against an in-memory table and a structurally valid
(but meaningless) VAPID key, so `npm run dev` drives the whole UI. The browser
subscription is real; nothing is ever delivered.

**iOS needs the app installed.** Safari exposes `PushManager` only to a site
added to the home screen, so an iPhone in a normal tab gets an explanation of
the Share → Add to Home Screen step instead of a button that cannot work.
`detectPushSupport()` covers that, an insecure origin, a browser without push,
and a permission already refused.

## Tests

`npm run test` runs vitest in Node:

- `src/lib/nowcast/sampler.test.ts` — projection parity with pyproj,
  quantisation round-trip, the 255 → null rule, nearest-pixel rounding,
  off-coverage;
- `src/lib/map/warp.test.ts` — the four-corner error and the mesh's accuracy;
- `src/lib/i18n/catalog.test.ts` — every key, argument count and list length
  present in both locales;
- `src/lib/components/panel.test.ts` — when a new point unfolds a minimised
  forecast panel: a *different* place does, the same place re-sampled by the
  next cycle does not;
- `src/lib/nowcast/forecast.test.ts` — the server fallback's mapping, including
  an additive field an older sidecar omits;
- `src/lib/pwa.test.ts` — the web manifest and its icons;
- `src/lib/push/*.test.ts` — support detection per branch, preference
  normalisation and the storage copy (including storage that throws), the VAPID
  conversion against fixed vectors, the API layer against a stubbed `fetch`,
  and push-payload parsing.

CI additionally runs `npm run check` (svelte-check) and `npm run build`.
