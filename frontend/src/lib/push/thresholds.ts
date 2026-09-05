/**
 * One knob: the horizon. Everything in this file exists so that the
 * probability threshold behind it never becomes a second one.
 *
 * `/api/push/options` says which horizons may be chosen and, for each of
 * them, the threshold it warns at — fitted per lead against DMI's rain
 * gauges rather than picked by anybody. The page shows that number as a
 * fact, not as a control, and the subscribe request leaves it out entirely
 * so the server looks it up at send time: a device that pinned today's value
 * would silently opt out of every refit.
 *
 * The one exception is `?threshold=NN` in the URL. It is not advertised, it
 * is only shown when it is present, and it is the only thing that puts a
 * `threshold_pct` in the request body.
 *
 * Nothing here throws, and nothing here holds prose — the catalog owns every
 * sentence, and the builders below only decide which one and what goes in it.
 */
import type { Catalog, Locale } from '$lib/i18n';

/** Which of the three ways a threshold can be arrived at is in force. */
export type ThresholdSource = 'table' | 'fallback' | 'override';

/** One horizon's fitted threshold, as the options endpoint reports it. */
export interface LeadThreshold {
	thresholdPct: number;
	/** `fallback` when the fit has nothing to say about this horizon yet. */
	source: 'table' | 'fallback';
}

/** `/api/push/options`, camel-cased. */
export interface PushOptions {
	leadOptionsMin: number[];
	fallbackThresholdPct: number;
	/** When the table was fitted; null when there is no table. */
	fittedAtUtc: string | null;
	/** Keyed by horizon in minutes. A missing horizon reads as the fallback. */
	thresholds: Record<number, LeadThreshold>;
}

/** The threshold in force for one horizon, and where it came from. */
export interface EffectiveThresholdPick {
	pct: number;
	source: ThresholdSource;
}

/** The quiet line under the selector, and the override line when there is one. */
export interface ThresholdFact extends EffectiveThresholdPick {
	fact: string;
	/** Null unless `?threshold=NN` (or a stored override) is in force. */
	override: string | null;
}

/** Rendered before `/api/push/options` answers — never a substitute for it. */
export const FALLBACK_LEAD_OPTIONS_MIN = [20, 30, 45, 60];

/** The rule that shipped before any fit existed, and the answer when none applies. */
export const DEFAULT_FALLBACK_THRESHOLD_PCT = 40;

/**
 * The override's accepted range. Narrower than 0–100 on purpose: below 20 %
 * the rule fires on almost every cycle and above 80 % it fires on almost
 * none, and neither is a setting anyone wants by accident.
 */
export const OVERRIDE_MIN_PCT = 20;
export const OVERRIDE_MAX_PCT = 80;

/** The options a server that never answered implies. */
export const FALLBACK_OPTIONS: PushOptions = {
	leadOptionsMin: FALLBACK_LEAD_OPTIONS_MIN,
	fallbackThresholdPct: DEFAULT_FALLBACK_THRESHOLD_PCT,
	fittedAtUtc: null,
	thresholds: {}
};

type Obj = Record<string, unknown>;

const isRecord = (value: unknown): value is Obj =>
	typeof value === 'object' && value !== null && !Array.isArray(value);

/** A finite number from a number or a numeric string; null for anything else. */
function toNumber(value: unknown): number | null {
	if (typeof value === 'number') return Number.isFinite(value) ? value : null;
	if (typeof value === 'string' && value.trim() !== '') {
		const parsed = Number(value);
		return Number.isFinite(parsed) ? parsed : null;
	}
	return null;
}

/** Whether `value` is a threshold the hidden override is allowed to name. */
export function isOverridePct(value: number): boolean {
	return Number.isFinite(value) && value >= OVERRIDE_MIN_PCT && value <= OVERRIDE_MAX_PCT;
}

/**
 * `?threshold=NN` → the override, or null.
 *
 * Takes the query string (or a whole URL) rather than reading `location`, so
 * the service worker and the page can both call it and so it is testable.
 * Out of range is null, not clamped: a `?threshold=8` is a mistake, and
 * quietly turning it into 20 % would hide the mistake behind a number.
 */
export function thresholdOverrideFromUrl(search: string): number | null {
	if (typeof search !== 'string' || search === '') return null;
	const query = search.includes('?') ? search.slice(search.indexOf('?')) : `?${search}`;
	let raw: string | null;
	try {
		raw = new URLSearchParams(query).get('threshold');
	} catch {
		return null;
	}
	if (raw === null || raw.trim() === '') return null;
	const value = toNumber(raw);
	if (value === null || !Number.isInteger(value) || !isOverridePct(value)) return null;
	return value;
}

/**
 * The nearest horizon the server offers.
 *
 * A stored 45 min survives the server dropping 45 from its list as 60 rather
 * than snapping back to a default: the choice the user made is better
 * evidence of what they want than the server's opinion of a typical user.
 * Ties go to the earlier — shorter — option.
 */
export function nearestLead(value: number, options: number[]): number {
	if (options.length === 0) return value;
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

/** `/api/push/options` → `PushOptions`. Anything unreadable falls back. */
export function parsePushOptions(raw: unknown): PushOptions {
	if (!isRecord(raw)) return FALLBACK_OPTIONS;

	const leads = Array.isArray(raw.lead_options)
		? raw.lead_options.map(toNumber).filter((n): n is number => n !== null && n > 0)
		: [];
	const fallback = toNumber(raw.fallback_threshold_pct);
	const fittedAt = typeof raw.fitted_at_utc === 'string' && raw.fitted_at_utc ? raw.fitted_at_utc : null;

	const thresholds: Record<number, LeadThreshold> = {};
	if (isRecord(raw.thresholds)) {
		for (const [key, value] of Object.entries(raw.thresholds)) {
			const lead = toNumber(key);
			if (lead === null || !isRecord(value)) continue;
			const pct = toNumber(value.threshold_pct);
			// A row with no number in it is a row that says nothing; it falls
			// back like an absent one rather than becoming a threshold of 0.
			if (pct === null) continue;
			thresholds[lead] = {
				thresholdPct: Math.round(pct),
				source: value.source === 'table' ? 'table' : 'fallback'
			};
		}
	}

	return {
		leadOptionsMin: leads.length > 0 ? leads : FALLBACK_LEAD_OPTIONS_MIN,
		fallbackThresholdPct:
			fallback === null ? DEFAULT_FALLBACK_THRESHOLD_PCT : Math.round(fallback),
		fittedAtUtc: fittedAt === null || !Number.isFinite(Date.parse(fittedAt)) ? null : fittedAt,
		thresholds
	};
}

/**
 * What this device will actually be warned at, and why.
 *
 * The override wins over everything, then the table, then the fallback —
 * which is also what the server does, so the sentence on screen and the rule
 * that fires agree.
 */
export function effectiveThreshold(
	options: PushOptions,
	leadMin: number,
	override: number | null = null
): EffectiveThresholdPick {
	if (override !== null && isOverridePct(override)) {
		return { pct: Math.round(override), source: 'override' };
	}
	const entry = options.thresholds[leadMin];
	if (entry && entry.source === 'table') return { pct: entry.thresholdPct, source: 'table' };
	return {
		pct: entry ? entry.thresholdPct : options.fallbackThresholdPct,
		source: 'fallback'
	};
}

/** "3 Sep" / "3. sep." — the day the table was fitted, no year, no clock. */
export function fittedDate(isoUtc: string, locale: Locale): string | null {
	const ms = Date.parse(isoUtc);
	if (!Number.isFinite(ms)) return null;
	try {
		return new Intl.DateTimeFormat(locale === 'da' ? 'da-DK' : 'en-GB', {
			day: 'numeric',
			month: 'short'
		}).format(new Date(ms));
	} catch {
		return isoUtc.slice(0, 10);
	}
}

/**
 * The quiet fact under the horizon selector.
 *
 * Three shapes, because they say three different things: a fitted threshold
 * with the day it was fitted on, the same without a usable date, and the
 * default that stands in until a horizon has enough evidence of its own. The
 * override, when in force, is an extra line rather than a replacement — the
 * fitted value is still the thing everyone else gets, and hiding it would
 * make the override look like the site's own answer.
 */
export function thresholdFact(
	t: Catalog,
	locale: Locale,
	options: PushOptions,
	leadMin: number,
	override: number | null = null
): ThresholdFact {
	const table = effectiveThreshold(options, leadMin, null);
	const pick = effectiveThreshold(options, leadMin, override);
	const date = options.fittedAtUtc === null ? null : fittedDate(options.fittedAtUtc, locale);

	let fact: string;
	if (table.source === 'fallback') {
		fact = t.push.factFallback(table.pct);
	} else if (date === null) {
		fact = t.push.factTableUndated(table.pct);
	} else {
		fact = t.push.factTable(table.pct, date);
	}

	return {
		...pick,
		fact,
		override: pick.source === 'override' ? t.push.factOverride(pick.pct) : null
	};
}
