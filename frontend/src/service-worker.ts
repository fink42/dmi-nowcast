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
 *  - `/api/*` — a cached push config, or a replayed subscribe, would both be
 *    wrong in ways that are hard to see.
 *  - the basemap archive and its glyphs — `basemap.pmtiles` is well over a
 *    hundred megabytes and is fetched in ranges; the HTTP cache handles it.
 *
 * It does two other things, both push: show the notification a message
 * carries, and put the map on the right point when that notification is
 * tapped. Both keep their logic in `$lib/push/notification`, which has no
 * imports of its own — a worker that pulls in the app's reactive state or its
 * DOM helpers is a worker that fails to install.
 */
import { build, files, version } from '$service-worker';
import { da } from '$lib/i18n/da';
import { en } from '$lib/i18n/en';
import {
	notificationFromPayload,
	parsePushPayload,
	payloadLang,
	pointFromUrl
} from '$lib/push/notification';

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
	// Live data and the push API: never cached here, never intercepted. The
	// API matters especially — a cached `/api/push/config`, or a replayed
	// subscribe, would both be wrong in ways that are hard to see.
	if (
		url.pathname.startsWith('/nowcast/') ||
		url.pathname.startsWith('/forecast') ||
		url.pathname.startsWith('/api/')
	)
		return;
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

// --- push --------------------------------------------------------------------

const ICON = '/icons/icon-192.png';

/**
 * Show what the message asked for. A payload this version cannot parse still
 * gets a notification — the alternative is a silent push, which browsers
 * punish by revoking the permission — in whatever language survived.
 */
async function showFromPush(data: PushMessageData | null): Promise<void> {
	let raw: unknown = null;
	try {
		raw = data?.json() ?? null;
	} catch {
		raw = null;
	}

	const payload = parsePushPayload(raw);
	if (payload) {
		const { title, options } = notificationFromPayload(payload);
		await sw.registration.showNotification(title, options);
		return;
	}

	const lang = payloadLang(raw);
	const catalog = lang === 'en' ? en : da;
	await sw.registration.showNotification(catalog.push.fallbackTitle, {
		body: catalog.push.fallbackBody,
		icon: ICON,
		badge: ICON,
		lang,
		data: { url: '/' }
	});
}

sw.addEventListener('push', (event) => {
	event.waitUntil(showFromPush(event.data));
});

/**
 * Open the point the notification is about.
 *
 * Focusing the tab the user already has open beats opening a second one, so a
 * window of this origin is preferred and told where to go by message; only
 * with no window at all is a new one opened at the URL.
 */
async function openFromNotification(url: string): Promise<void> {
	let target: URL;
	try {
		target = new URL(url, sw.location.origin);
	} catch {
		target = new URL('/', sw.location.origin);
	}
	// The payload decides where a tap lands, so it does not get to send the
	// user off this origin.
	if (target.origin !== sw.location.origin) target = new URL('/', sw.location.origin);

	const point = pointFromUrl(target.search);
	const clients = (await sw.clients.matchAll({
		type: 'window',
		includeUncontrolled: true
	})) as readonly WindowClient[];
	const existing = clients.find((client) => new URL(client.url).origin === sw.location.origin);
	if (existing) {
		try {
			await existing.focus();
		} catch {
			// Focus can be refused; the message below is still worth sending.
		}
		if (point) existing.postMessage({ type: 'open-point', lat: point.lat, lon: point.lon });
		return;
	}
	await sw.clients.openWindow(target.href);
}

sw.addEventListener('notificationclick', (event) => {
	event.notification.close();
	const data = event.notification.data as { url?: string } | null;
	event.waitUntil(openFromNotification(typeof data?.url === 'string' ? data.url : '/'));
});
