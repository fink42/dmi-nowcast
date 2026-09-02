/**
 * Turning numbers into the sentences the panel tells.
 *
 * Every string comes from the catalog; this module only decides *which*
 * string, and does the one conversion the UTC-everywhere rule allows at the
 * boundary: radar timestamps into the viewer's local clock.
 */
import type { Catalog, Locale } from '$lib/i18n';
import type { PointForecast } from '$lib/nowcast/sampler';

/** Below this many minutes an ETA means "it is already raining here". */
export const RAINING_NOW_MIN = 1.5;

/** Minutes between `iso` and `nowMs`, or null when `iso` is unusable. */
function minutesSince(iso: string | undefined, nowMs: number): number | null {
	if (typeof iso !== 'string' || iso.trim() === '') return null;
	const t = Date.parse(iso);
	if (!Number.isFinite(t) || !Number.isFinite(nowMs)) return null;
	// Clocks disagree; a cycle stamped slightly in the future has simply not
	// aged yet — it is never negatively old.
	return Math.max(0, (nowMs - t) / 60000);
}

/**
 * A cycle's ETA, counted down to `nowMs`.
 *
 * The sidecar's point ETA is minutes from the instant the cycle was *computed*
 * (`generated_at_utc`), already corrected for the radar image's own age. But
 * cycles land 5–10 min apart while the page re-reads the clock every 15 s, so
 * printing that number unchanged leaves "rain in 12 min" on screen minutes
 * after the rain was due, and delays the "raining now" headline by as much.
 * Subtracting the cycle's age makes the same forecast tick.
 *
 * Deliberately pure: the caller supplies the clock, so nothing here depends on
 * `Date.now()`. A null ETA (no rain within the horizon) stays null, and a
 * manifest with no usable `generated_at_utc` — an older sidecar, a garbled
 * stamp — yields the ETA unchanged rather than a wrong countdown. It never
 * throws.
 */
export function countdownEtaMin(
	etaMin: number | null,
	generatedAtUtc: string | undefined,
	nowMs: number
): number | null {
	if (etaMin === null) return null;
	const elapsed = minutesSince(generatedAtUtc, nowMs);
	if (elapsed === null) return etaMin;
	// The floor matters: rain that was due two minutes ago is "now", not a
	// negative ETA counting up into the past.
	return Math.max(0, etaMin - elapsed);
}

export type Headline = 'raining-now' | 'eta' | 'no-rain';

/**
 * Which sentence the panel leads with, from the ETA *as of now* — pass the
 * `countdownEtaMin` value, not the forecast's own field, or the headline
 * flips to "raining here" a cycle late.
 */
export function headlineKind(etaMin: number | null): Headline {
	if (etaMin === null) return 'no-rain';
	return etaMin <= RAINING_NOW_MIN ? 'raining-now' : 'eta';
}

export function headline(t: Catalog, etaMin: number | null): string {
	switch (headlineKind(etaMin)) {
		case 'raining-now':
			return t.panel.headlineRainingNow;
		case 'eta':
			return t.panel.headlineEta(Math.round(etaMin ?? 0));
		default:
			return t.panel.headlineNoRain;
	}
}

/** WMO-ish rain-rate bands, in the words of the catalog. */
export function intensityWord(t: Catalog, mmH: number | null): string {
	if (mmH === null || mmH <= 0.05) return t.panel.intensityNone;
	if (mmH < 2.5) return t.panel.intensityLight;
	if (mmH < 10) return t.panel.intensityModerate;
	if (mmH < 50) return t.panel.intensityHeavy;
	return t.panel.intensityViolent;
}

export function confidenceWord(t: Catalog, confidence: number): string {
	if (confidence >= 0.66) return t.panel.confidenceHigh;
	if (confidence >= 0.33) return t.panel.confidenceMedium;
	return t.panel.confidenceLow;
}

export const percent = (p: number): number => Math.round(p * 100);

/** The lead closest to `minutes` that the cycle actually served. */
export function probabilityWithin(
	forecast: PointForecast,
	minutes: number
): { leadMin: number; pRain: number } | null {
	let best: { leadMin: number; pRain: number } | null = null;
	let bestDistance = Infinity;
	for (const lead of forecast.perLead) {
		if (lead.pRain === null) continue;
		const distance = Math.abs(lead.leadMin - minutes);
		if (distance < bestDistance) {
			bestDistance = distance;
			best = { leadMin: lead.leadMin, pRain: lead.pRain };
		}
	}
	return best;
}

const localeTag = (locale: Locale): string => (locale === 'da' ? 'da-DK' : 'en-GB');

/** UTC ISO timestamp → the viewer's local clock time (hh:mm). */
export function clockTime(iso: string, locale: Locale): string {
	try {
		return new Intl.DateTimeFormat(localeTag(locale), {
			hour: '2-digit',
			minute: '2-digit'
		}).format(new Date(iso));
	} catch {
		return iso.slice(11, 16);
	}
}
