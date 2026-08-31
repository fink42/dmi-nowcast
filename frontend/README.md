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
src/lib/map/           MapLibre style, Mercator resampling, overlay frames
src/lib/components/    MapView, LoopControls, ForecastPanel, LangToggle, footer
src/routes/            map (/) + about, data, privacy, support
scripts/               fetch-basemap, make-icons, mock-sidecar
```

Three pieces deserve a note:

- **Sampling (`src/lib/nowcast/sampler.ts`)** mirrors `/forecast` exactly:
  `col = (x - x_ul) / pixel_scale`, `row = (y_ul - y) / pixel_scale`, round to
  the nearest pixel, `value = level * scale + offset`, level 255 → *null*, and
  outside the grid is **off-coverage**, never a 0 % probability. `sampler.test.ts`
  pins this against numbers produced by pyproj on the server side.
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
precaches the app shell and **never** caches `/nowcast/*` or `/forecast` (a
stale rain map is worse than none) nor the basemap archive (too big, and it is
range-requested). Icons are generated from one raindrop path:
`node scripts/make-icons.mjs`.

## Tests

`npm run test` runs vitest in Node:

- `src/lib/nowcast/sampler.test.ts` — projection parity with pyproj,
  quantisation round-trip, the 255 → null rule, nearest-pixel rounding,
  off-coverage;
- `src/lib/map/warp.test.ts` — the four-corner error and the mesh's accuracy;
- `src/lib/i18n/catalog.test.ts` — every key, argument count and list length
  present in both locales;
- `src/lib/pwa.test.ts` — the web manifest and its icons.

CI additionally runs `npm run check` (svelte-check) and `npm run build`.
