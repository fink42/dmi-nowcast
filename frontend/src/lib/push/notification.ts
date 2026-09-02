/**
 * The push payload contract, and the two conversions the service worker
 * needs: payload → notification, and notification URL → map point.
 *
 * This module is deliberately free of imports. The service worker imports it,
 * and a worker that pulls in the app's reactive state or its DOM helpers is a
 * worker that fails to install.
 */

export type PushPayloadType = 'rain_incoming' | 'test';

/** The JSON the sidecar encrypts into a push message. */
export interface PushPayload {
	type: PushPayloadType;
	title: string;
	body: string;
	lang: 'da' | 'en';
	/** The point the alert is about; null when the payload carried none. */
	lat: number | null;
	lon: number | null;
	/** Where a click should land, same-origin and relative. */
	url: string;
	tag: string;
	sentUtc: string;
	etaMin: number | null;
	pPct: number | null;
	leadMin: number | null;
	intensityMmH: number | null;
}

/** Notification options plus the one field the click handler reads back. */
export interface PushNotificationOptions extends NotificationOptions {
	renotify?: boolean;
	data: { url: string };
}

const TYPES: readonly string[] = ['rain_incoming', 'test'];
const ICON = '/icons/icon-192.png';

const isRecord = (value: unknown): value is Record<string, unknown> =>
	typeof value === 'object' && value !== null && !Array.isArray(value);

const str = (value: unknown): string | null =>
	typeof value === 'string' && value.trim() !== '' ? value : null;

const num = (value: unknown): number | null =>
	typeof value === 'number' && Number.isFinite(value) ? value : null;

/**
 * The language a payload asks for — Danish unless it explicitly says English.
 * Exported because the fallback notification (for a payload that would not
 * parse at all) still wants to be in the right language if that one field
 * survived.
 */
export function payloadLang(raw: unknown): 'da' | 'en' {
	return isRecord(raw) && raw.lang === 'en' ? 'en' : 'da';
}

/**
 * Validate and camel-case a push payload. Returns null for anything this
 * client does not understand — an unknown `type` included, because a future
 * payload kind may mean something this version would render misleadingly.
 */
export function parsePushPayload(data: unknown): PushPayload | null {
	if (!isRecord(data)) return null;
	const type = typeof data.type === 'string' && TYPES.includes(data.type) ? data.type : null;
	const title = str(data.title);
	const body = str(data.body);
	if (!type || !title || !body) return null;

	const lat = num(data.lat);
	const lon = num(data.lon);
	const inRange = lat !== null && lon !== null && Math.abs(lat) <= 90 && Math.abs(lon) <= 180;

	return {
		type: type as PushPayloadType,
		title,
		body,
		lang: payloadLang(data),
		lat: inRange ? lat : null,
		lon: inRange ? lon : null,
		url: str(data.url) ?? '/',
		tag: str(data.tag) ?? 'rain-incoming',
		sentUtc: str(data.sent_utc) ?? '',
		etaMin: num(data.eta_min),
		pPct: num(data.p_pct),
		leadMin: num(data.lead_min),
		intensityMmH: num(data.intensity_mm_h)
	};
}

/** Payload → the arguments of `registration.showNotification()`. */
export function notificationFromPayload(p: PushPayload): {
	title: string;
	options: PushNotificationOptions;
} {
	return {
		title: p.title,
		options: {
			body: p.body,
			icon: ICON,
			badge: ICON,
			tag: p.tag,
			lang: p.lang,
			// With a tag, a second alert replaces the first silently unless
			// this says otherwise — and "rain in 10 min" superseding "rain in
			// 25 min" is exactly the update worth buzzing for.
			renotify: p.tag !== '',
			data: { url: p.url }
		}
	};
}

/**
 * `?lat=&lon=` → a point, or null. Accepts a bare query string or a whole
 * URL, since one caller has `location.search` and the other has the
 * notification's `data.url`.
 */
export function pointFromUrl(search: string): { lat: number; lon: number } | null {
	if (typeof search !== 'string' || search === '') return null;
	const query = search.includes('?') ? search.slice(search.indexOf('?') + 1) : search;
	let params: URLSearchParams;
	try {
		params = new URLSearchParams(query);
	} catch {
		return null;
	}
	const rawLat = params.get('lat');
	const rawLon = params.get('lon');
	// `Number('')` is 0, and 0/0 is a real point in the Gulf of Guinea — an
	// empty parameter has to be rejected before it is parsed.
	if (!rawLat?.trim() || !rawLon?.trim()) return null;
	const lat = Number(rawLat);
	const lon = Number(rawLon);
	if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
	if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
	return { lat, lon };
}
