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
const RAINING_NOW_MIN = 1.5;

export type Headline = 'raining-now' | 'eta' | 'no-rain';

export function headlineKind(forecast: PointForecast): Headline {
	if (forecast.etaMin === null) return 'no-rain';
	return forecast.etaMin <= RAINING_NOW_MIN ? 'raining-now' : 'eta';
}

export function headline(t: Catalog, forecast: PointForecast): string {
	switch (headlineKind(forecast)) {
		case 'raining-now':
			return t.panel.headlineRainingNow;
		case 'eta':
			return t.panel.headlineEta(Math.round(forecast.etaMin ?? 0));
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
