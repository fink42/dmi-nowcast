/// <reference types="@sveltejs/kit" />
/// <reference lib="webworker" />
/**
 * A deliberately small service worker: it precaches the app shell so the site
 * opens instantly (and at all) on a bad mobile connection, and it stays out
 * of the way of everything else.
 *
 * What it never touches:
 *  - `/nowcast/*` and `/forecast` — forecast data has its own HTTP caching
 *    (immutable cycle-stamped artifacts, 30 s on the manifest). A stale rain
 *    map served from a cache would be worse than no map.
 *  - the basemap archive and its glyphs — `basemap.pmtiles` is well over a
 *    hundred megabytes and is fetched in ranges; the HTTP cache handles it.
 *
 * Installability is the point: this is the groundwork Phase D's push
 * notifications need.
 */
import { build, files, version } from '$service-worker';

const sw = self as unknown as ServiceWorkerGlobalScope;
const CACHE = `app-shell-${version}`;

/** Static assets that are too big or too range-y to precache. */
const EXCLUDED = /^\/(basemap\.pmtiles|basemap-assets\/)/;

const SHELL = [...build, ...files.filter((f) => !EXCLUDED.test(f))];

sw.addEventListener('install', (event) => {
	event.waitUntil(
		caches
			.open(CACHE)
			.then((cache) => cache.addAll(SHELL))
			.then(() => sw.skipWaiting())
	);
});

sw.addEventListener('activate', (event) => {
	event.waitUntil(
		caches
			.keys()
			.then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
			.then(() => sw.clients.claim())
	);
});

sw.addEventListener('fetch', (event) => {
	const request = event.request;
	if (request.method !== 'GET') return;

	const url = new URL(request.url);
	if (url.origin !== location.origin) return;
	// Live data: never cached here, never intercepted.
	if (url.pathname.startsWith('/nowcast/') || url.pathname.startsWith('/forecast')) return;
	if (EXCLUDED.test(url.pathname)) return;

	// Hashed build assets are immutable: cache-first is safe and fast.
	if (SHELL.includes(url.pathname)) {
		event.respondWith(
			caches
				.open(CACHE)
				.then((cache) => cache.match(request))
				.then((cached) => cached ?? fetch(request))
		);
		return;
	}

	// Navigations: network first, falling back to the cached shell offline.
	if (request.mode === 'navigate') {
		event.respondWith(
			fetch(request).catch(async () => {
				const cache = await caches.open(CACHE);
				return (await cache.match('/')) ?? (await cache.match('/index.html')) ?? Response.error();
			})
		);
	}
});
