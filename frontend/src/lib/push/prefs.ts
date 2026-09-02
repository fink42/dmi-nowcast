/**
 * Notification preferences: the shapes, the server config they are validated
 * against, and the small amount of localStorage that remembers what this
 * browser subscribed to.
 *
 * Two rules run through the whole file:
 *
 *  - **The server owns the options.** Which thresholds and lead windows may
 *    be chosen comes from `/api/push/config`; the constants below are what
 *    the UI renders in the moment before that response lands, and what a
 *    stored preference is measured against if the config never arrives.
 *  - **Storage is allowed to fail.** Private windows, disabled site data and
 *    iOS quirks all throw on `localStorage`. Every access is guarded: a
 *    browser that refuses storage still gets working notifications, it just
 *    cannot show the summary line after a reload.
 */
import { LOCALES, type Locale } from '$lib/i18n/types';

export interface QuietHours {
	enabled: boolean;
	/** `HH:MM`, 24-hour, in the subscriber's own time zone. */
	start: string;
	end: string;
}

export interface PushPrefs {
	thresholdPct: number;
	leadMin: number;
	quietHours: QuietHours;
}

/** `/api/push/config`, camel-cased. */
export interface PushConfig {
	enabled: boolean;
	/** VAPID application server key, base64url. Null when the feature is off. */
	vapidPublicKey: string | null;
	thresholdOptionsPct: number[];
	leadOptionsMin: number[];
	defaults: PushPrefs;
	/** The server will accept no further devices right now. */
	capacityReached: boolean;
}

/** What this browser believes it is subscribed to. */
export interface StoredSubscription {
	endpoint: string;
	lat: number;
	lon: number;
	prefs: PushPrefs;
	lang: Locale;
	tz: string;
	subscribedAt: string;
}

export const STORAGE_KEY = 'dmi-nowcast.push';

/** Rendered before `/api/push/config` answers — never a substitute for it. */
export const FALLBACK_THRESHOLD_OPTIONS_PCT = [40, 60, 80];
export const FALLBACK_LEAD_OPTIONS_MIN = [20, 30, 45, 60];
export const FALLBACK_PREFS: PushPrefs = {
	thresholdPct: 60,
	leadMin: 30,
	quietHours: { enabled: false, start: '22:00', end: '07:00' }
};

/** The config a disabled — or unreachable — server implies. */
export const DISABLED_CONFIG: PushConfig = {
	enabled: false,
	vapidPublicKey: null,
	thresholdOptionsPct: FALLBACK_THRESHOLD_OPTIONS_PCT,
	leadOptionsMin: FALLBACK_LEAD_OPTIONS_MIN,
	defaults: FALLBACK_PREFS,
	capacityReached: false
};

const DEFAULT_TZ = 'Europe/Copenhagen';

const isRecord = (value: unknown): value is Record<string, unknown> =>
	typeof value === 'object' && value !== null && !Array.isArray(value);

/** `HH:MM`, 24-hour, zero-padded — exactly what `<input type="time">` emits. */
export function isValidHhMm(s: unknown): s is string {
	return typeof s === 'string' && /^([01]\d|2[0-3]):[0-5]\d$/.test(s);
}

/** A finite number from a number or a numeric string; null for anything else. */
function toNumber(value: unknown): number | null {
	if (typeof value === 'number') return Number.isFinite(value) ? value : null;
	if (typeof value === 'string' && value.trim() !== '') {
		const parsed = Number(value);
		return Number.isFinite(parsed) ? parsed : null;
	}
	return null;
}

function numberList(value: unknown, fallback: number[]): number[] {
	if (!Array.isArray(value)) return fallback;
	const out = value.map(toNumber).filter((n): n is number => n !== null);
	return out.length > 0 ? out : fallback;
}

/**
 * The nearest option the server offers.
 *
 * A stored 60 % survives the server changing its options from
 * `[40, 60, 80]` to `[50, 70]` as 70, rather than snapping back to whatever
 * the default happens to be: the choice the user made is better evidence of
 * what they want than the server's opinion of a typical user.
 */
function nearestOption(value: number, options: number[], fallback: number): number {
	if (options.length === 0) return fallback;
	let best = options[0];
	let bestDistance = Math.abs(options[0] - value);
	for (const option of options.slice(1)) {
		const distance = Math.abs(option - value);
		if (distance < bestDistance) {
			best = option;
			bestDistance = distance;
		}
	}
	return best;
}

export function defaultPrefs(config: PushConfig | null): PushPrefs {
	const defaults = config?.defaults ?? FALLBACK_PREFS;
	return { ...defaults, quietHours: { ...defaults.quietHours } };
}

/**
 * Anything → a preference set the server will accept: values coerced to the
 * offered options, garbage replaced by the defaults.
 */
export function normalisePrefs(raw: unknown, config: PushConfig | null): PushPrefs {
	const defaults = defaultPrefs(config);
	const thresholds = config?.thresholdOptionsPct ?? FALLBACK_THRESHOLD_OPTIONS_PCT;
	const leads = config?.leadOptionsMin ?? FALLBACK_LEAD_OPTIONS_MIN;
	if (!isRecord(raw)) return defaults;

	const threshold = toNumber(raw.thresholdPct);
	const lead = toNumber(raw.leadMin);
	const quiet = isRecord(raw.quietHours) ? raw.quietHours : {};

	return {
		thresholdPct:
			threshold === null
				? nearestOption(defaults.thresholdPct, thresholds, defaults.thresholdPct)
				: nearestOption(threshold, thresholds, defaults.thresholdPct),
		leadMin:
			lead === null
				? nearestOption(defaults.leadMin, leads, defaults.leadMin)
				: nearestOption(lead, leads, defaults.leadMin),
		quietHours: {
			enabled: quiet.enabled === true,
			start: isValidHhMm(quiet.start) ? quiet.start : defaults.quietHours.start,
			end: isValidHhMm(quiet.end) ? quiet.end : defaults.quietHours.end
		}
	};
}

/** `/api/push/config` → `PushConfig`. Anything unexpected reads as disabled. */
export function parsePushConfig(raw: unknown): PushConfig {
	if (!isRecord(raw) || raw.enabled !== true) return DISABLED_CONFIG;
	const key = typeof raw.vapid_public_key === 'string' ? raw.vapid_public_key : null;
	// No key, no subscription: the browser cannot call `subscribe()` without
	// an application server key, so a keyless "enabled" is really disabled.
	if (!key) return DISABLED_CONFIG;
	const thresholds = numberList(raw.threshold_options_pct, FALLBACK_THRESHOLD_OPTIONS_PCT);
	const leads = numberList(raw.lead_options_min, FALLBACK_LEAD_OPTIONS_MIN);
	const rawDefaults = isRecord(raw.defaults) ? raw.defaults : {};
	const rawQuiet = isRecord(rawDefaults.quiet_hours) ? rawDefaults.quiet_hours : {};
	const shell: PushConfig = {
		enabled: true,
		vapidPublicKey: key,
		thresholdOptionsPct: thresholds,
		leadOptionsMin: leads,
		defaults: FALLBACK_PREFS,
		capacityReached: raw.capacity_reached === true
	};
	shell.defaults = normalisePrefs(
		{
			thresholdPct: rawDefaults.threshold_pct,
			leadMin: rawDefaults.lead_min,
			quietHours: { enabled: rawQuiet.enabled, start: rawQuiet.start, end: rawQuiet.end }
		},
		// Validate the server's own defaults against the server's own options,
		// but with the fallback defaults underneath — otherwise this recurses.
		{ ...shell, defaults: FALLBACK_PREFS }
	);
	return shell;
}

// --- storage -----------------------------------------------------------------

/** The storage to use, or null when there is none (server, or a browser that refuses). */
function storage(): Storage | null {
	try {
		return typeof localStorage === 'undefined' ? null : localStorage;
	} catch {
		return null;
	}
}

const isLocale = (value: unknown): value is Locale =>
	typeof value === 'string' && (LOCALES as readonly string[]).includes(value);

/** The stored copy, or null if there is none, it is unreadable, or it is junk. */
export function loadStored(config: PushConfig | null = null): StoredSubscription | null {
	const store = storage();
	if (!store) return null;
	let raw: unknown;
	try {
		const text = store.getItem(STORAGE_KEY);
		if (!text) return null;
		raw = JSON.parse(text);
	} catch {
		return null;
	}
	if (!isRecord(raw)) return null;
	const endpoint = typeof raw.endpoint === 'string' ? raw.endpoint : '';
	const lat = toNumber(raw.lat);
	const lon = toNumber(raw.lon);
	if (!endpoint || lat === null || lon === null) return null;
	return {
		endpoint,
		lat,
		lon,
		prefs: normalisePrefs(raw.prefs, config),
		lang: isLocale(raw.lang) ? raw.lang : 'da',
		tz: typeof raw.tz === 'string' && raw.tz ? raw.tz : DEFAULT_TZ,
		subscribedAt: typeof raw.subscribedAt === 'string' ? raw.subscribedAt : ''
	};
}

/** Persist the copy. Returns false when storage refused — never throws. */
export function saveStored(sub: StoredSubscription): boolean {
	const store = storage();
	if (!store) return false;
	try {
		store.setItem(STORAGE_KEY, JSON.stringify(sub));
		return true;
	} catch {
		return false;
	}
}

export function clearStored(): void {
	const store = storage();
	if (!store) return;
	try {
		store.removeItem(STORAGE_KEY);
	} catch {
		/* nothing to do: the copy is advisory, the server holds the truth */
	}
}

/** The subscriber's IANA time zone — the server needs it for quiet hours. */
export function resolveTimeZone(): string {
	try {
		return Intl.DateTimeFormat().resolvedOptions().timeZone || DEFAULT_TZ;
	} catch {
		return DEFAULT_TZ;
	}
}

/** The `POST /api/push/subscribe` body. */
export interface SubscribeBody {
	subscription: PushSubscriptionJSON;
	lat: number;
	lon: number;
	threshold_pct: number;
	lead_min: number;
	quiet_hours: QuietHours;
	tz: string;
	lang: Locale;
}

export function subscribeBody(
	subscriptionJson: PushSubscriptionJSON,
	lat: number,
	lon: number,
	prefs: PushPrefs,
	lang: Locale,
	tz: string
): SubscribeBody {
	return {
		subscription: subscriptionJson,
		lat,
		lon,
		threshold_pct: prefs.thresholdPct,
		lead_min: prefs.leadMin,
		quiet_hours: {
			enabled: prefs.quietHours.enabled,
			start: prefs.quietHours.start,
			end: prefs.quietHours.end
		},
		tz,
		lang
	};
}
