/**
 * Notification preferences: the shapes, the server config they are validated
 * against, and the small amount of localStorage that remembers what this
 * browser subscribed to.
 *
 * Three rules run through the whole file:
 *
 *  - **One knob.** A subscriber chooses a horizon and nothing else. The
 *    probability threshold that horizon warns at is fitted per lead and
 *    served by `/api/push/options` (see `./thresholds`), so it is not a
 *    preference and is not stored here. `thresholdPct` below is the hidden
 *    `?threshold=NN` override, null in every ordinary case.
 *  - **The server owns the options.** Which horizons may be chosen comes
 *    from `/api/push/options`; `/api/push/config` only says whether push
 *    works at all and what the initial horizon is.
 *  - **Storage is allowed to fail.** Private windows, disabled site data and
 *    iOS quirks all throw on `localStorage`. Every access is guarded: a
 *    browser that refuses storage still gets working notifications, it just
 *    cannot show the summary line after a reload.
 */
import { LOCALES, type Locale } from '$lib/i18n/types';
import { isOverridePct, type ThresholdSource } from './thresholds';

export interface QuietHours {
	enabled: boolean;
	/** `HH:MM`, 24-hour, in the subscriber's own time zone. */
	start: string;
	end: string;
}

export interface PushPrefs {
	/**
	 * The `?threshold=NN` override, or null — which is what a subscriber who
	 * has not gone looking for the address bar always has. Null means "use
	 * the threshold fitted for this horizon", and the server is left to look
	 * it up at send time.
	 */
	thresholdPct: number | null;
	leadMin: number;
	quietHours: QuietHours;
}

/** `/api/push/config`, camel-cased. */
export interface PushConfig {
	enabled: boolean;
	/** VAPID application server key, base64url. Null when the feature is off. */
	vapidPublicKey: string | null;
	defaults: PushPrefs;
	/** The server will accept no further devices right now. */
	capacityReached: boolean;
}

/** The threshold the server said it would actually warn this device at. */
export interface EffectiveThreshold {
	thresholdPct: number;
	source: ThresholdSource;
	/** When the table behind it was fitted; null when there is no table. */
	fittedAtUtc: string | null;
}

/** What this browser believes it is subscribed to. */
export interface StoredSubscription {
	endpoint: string;
	lat: number;
	lon: number;
	prefs: PushPrefs;
	/** The subscribe response's own answer; null for a copy written before it existed. */
	effective: EffectiveThreshold | null;
	lang: Locale;
	tz: string;
	subscribedAt: string;
}

export const STORAGE_KEY = 'dmi-nowcast.push';

/** Rendered before `/api/push/config` answers — never a substitute for it. */
export const FALLBACK_PREFS: PushPrefs = {
	thresholdPct: null,
	leadMin: 30,
	quietHours: { enabled: false, start: '22:00', end: '07:00' }
};

/** The config a disabled — or unreachable — server implies. */
export const DISABLED_CONFIG: PushConfig = {
	enabled: false,
	vapidPublicKey: null,
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

export function defaultPrefs(config: PushConfig | null): PushPrefs {
	const defaults = config?.defaults ?? FALLBACK_PREFS;
	return { ...defaults, quietHours: { ...defaults.quietHours } };
}

/**
 * Anything → a preference set the server will accept: garbage replaced by
 * the defaults.
 *
 * The horizon is *not* snapped to the offered list here. Which horizons are
 * offered is `/api/push/options`' business and this module does not read it;
 * a stored 35 min survives as 35 min, and the panel is what preselects the
 * nearest offered option and says that it did (see `nearestLead`). An
 * out-of-range threshold override is dropped rather than clamped, because a
 * `?threshold=999` is a typo, not a wish for 80 %.
 */
export function normalisePrefs(raw: unknown, config: PushConfig | null): PushPrefs {
	const defaults = defaultPrefs(config);
	if (!isRecord(raw)) return defaults;

	const threshold = toNumber(raw.thresholdPct);
	const lead = toNumber(raw.leadMin);
	const quiet = isRecord(raw.quietHours) ? raw.quietHours : {};

	return {
		thresholdPct: threshold !== null && isOverridePct(threshold) ? Math.round(threshold) : null,
		leadMin: lead === null ? defaults.leadMin : lead,
		quietHours: {
			enabled: quiet.enabled === true,
			start: isValidHhMm(quiet.start) ? quiet.start : defaults.quietHours.start,
			end: isValidHhMm(quiet.end) ? quiet.end : defaults.quietHours.end
		}
	};
}

/**
 * `/api/push/config` → `PushConfig`. Anything unexpected reads as disabled.
 *
 * `threshold_options_pct` and `defaults.threshold_pct` are deliberately not
 * read: since the fit exists, no threshold the config offers is a choice the
 * UI may make. The served default horizon still is one.
 */
export function parsePushConfig(raw: unknown): PushConfig {
	if (!isRecord(raw) || raw.enabled !== true) return DISABLED_CONFIG;
	const key = typeof raw.vapid_public_key === 'string' ? raw.vapid_public_key : null;
	// No key, no subscription: the browser cannot call `subscribe()` without
	// an application server key, so a keyless "enabled" is really disabled.
	if (!key) return DISABLED_CONFIG;
	const rawDefaults = isRecord(raw.defaults) ? raw.defaults : {};
	const rawQuiet = isRecord(rawDefaults.quiet_hours) ? rawDefaults.quiet_hours : {};
	return {
		enabled: true,
		vapidPublicKey: key,
		defaults: normalisePrefs(
			{
				// Never an override: the config's threshold is not a choice.
				thresholdPct: null,
				leadMin: rawDefaults.lead_min,
				quietHours: { enabled: rawQuiet.enabled, start: rawQuiet.start, end: rawQuiet.end }
			},
			null
		),
		capacityReached: raw.capacity_reached === true
	};
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
		effective: parseEffective(raw.effective),
		lang: isLocale(raw.lang) ? raw.lang : 'da',
		tz: typeof raw.tz === 'string' && raw.tz ? raw.tz : DEFAULT_TZ,
		subscribedAt: typeof raw.subscribedAt === 'string' ? raw.subscribedAt : ''
	};
}

/**
 * The stored copy of what the server said it would warn at. Null for a copy
 * written before this field existed, which is not a problem: the panel falls
 * back to reading the same number out of `/api/push/options`.
 */
function parseEffective(raw: unknown): EffectiveThreshold | null {
	if (!isRecord(raw)) return null;
	const pct = toNumber(raw.thresholdPct);
	if (pct === null) return null;
	const source = raw.source;
	if (source !== 'table' && source !== 'override' && source !== 'fallback') return null;
	return {
		thresholdPct: Math.round(pct),
		source,
		fittedAtUtc: typeof raw.fittedAtUtc === 'string' && raw.fittedAtUtc ? raw.fittedAtUtc : null
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

/**
 * The `POST /api/push/subscribe` body.
 *
 * `threshold_pct` is optional and is sent *only* for the hidden override.
 * Omitting it is the normal case and is what asks the server to use the
 * threshold fitted for `lead_min` — sending a number here instead would
 * freeze this device at today's value and quietly opt it out of every
 * refit.
 */
export interface SubscribeBody {
	subscription: PushSubscriptionJSON;
	lat: number;
	lon: number;
	threshold_pct?: number;
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
		...(prefs.thresholdPct === null ? {} : { threshold_pct: prefs.thresholdPct }),
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
