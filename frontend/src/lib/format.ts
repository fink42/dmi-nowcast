/**
 * Turning numbers into the sentences the panel tells.
 *
 * Every string comes from the catalog; this module only decides *which*
 * string, and does the one conversion the UTC-everywhere rule allows at the
 * boundary: radar timestamps into the viewer's local clock.
 */
import type { Catalog, Locale } from '$lib/i18n';
import type { PointForecast, RainSample } from '$lib/nowcast/sampler';

/**
 * An ensemble ETA at or below this many minutes has already run out: the
 * cycle's first-arrival product is saying the rain reached this point at or
 * before the moment we are asking about. It is *not* a claim that it is
 * raining now — the ETA product is cumulative first arrival, so it sticks at
 * zero for the rest of the cycle once a cell has passed over — which is why
 * the decision below hands that case to the rain field rather than to a
 * headline.
 */
export const RAINING_NOW_MIN = 1.5;

/**
 * Rain rate (mm/h) at or above which we say it is raining at a point.
 * Mirrors the sidecar's `forecast.rain_threshold_mm_h` (default 0.5 ≈ 18 dBZ,
 * genuine light rain) and is applied to the same statistic: the rain grids
 * are the native field reduced by block-wise p90, so this is the
 * `detection_stat: p90` test the sidecar's own `raining_now` performs.
 * Change it only together with that setting.
 */
export const RAINING_NOW_MM_H = 0.5;

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

/** The instants of a rain series, or null when any of them is unusable. */
function seriesTimes(series: readonly RainSample[]): number[] | null {
	const times = series.map((s) => Date.parse(s.validTsUtc));
	// One frame we cannot place in time makes the ordering of all of them a
	// guess, and this series is only useful as a function of time. Better to
	// say nothing than to interpolate across a hole of unknown width.
	return times.every((t) => Number.isFinite(t)) ? times : null;
}

/**
 * The rain rate over the point at wall-clock `nowMs`, in mm/h, read off the
 * deterministic advected field — the same field the loop draws.
 *
 * The series is sampled at discrete leads (0, 10 … 60 min from the cycle's
 * generation), and "now" almost never lands on one of them, so the value is
 * interpolated linearly in time between the two entries bracketing it. Before
 * the first entry and after the last the nearest one is used unchanged rather
 * than extrapolated.
 *
 * Null means "we do not know" and must never be shown as dry: an empty series
 * (a cycle that served no field), a nodata pixel at either end of the bracket,
 * or timestamps that will not parse all land there. Pure, and it never throws.
 */
export function rainNowMmH(series: readonly RainSample[], nowMs: number): number | null {
	if (series.length === 0 || !Number.isFinite(nowMs)) return null;
	const times = seriesTimes(series);
	if (!times) return null;
	// Ascending in time, whatever order the caller had it in.
	const order = Array.from({ length: series.length }, (_, i) => i).sort(
		(a, b) => times[a] - times[b]
	);
	const at = (k: number) => ({ t: times[order[k]], mmH: series[order[k]].mmH });

	const first = at(0);
	if (nowMs <= first.t) return first.mmH;
	const last = at(order.length - 1);
	if (nowMs >= last.t) return last.mmH;

	for (let k = 0; k < order.length - 1; k++) {
		const a = at(k);
		const b = at(k + 1);
		if (nowMs < a.t || nowMs > b.t) continue;
		// A nodata end of the bracket poisons the interpolation: there is no
		// honest number between "2 mm/h" and "no idea".
		if (a.mmH === null || b.mmH === null) return null;
		if (b.t === a.t) return a.mmH;
		return a.mmH + ((nowMs - a.t) / (b.t - a.t)) * (b.mmH - a.mmH);
	}
	return null;
}

/**
 * Minutes from `nowMs` until the field is next wet at this point, or null when
 * it never is within the series.
 *
 * Only entries *after* now count: a wet entry in the past is what already
 * happened, and one exactly at now is the business of `rainNowMmH`. Entries
 * with no readable value or timestamp are skipped rather than treated as dry.
 */
export function nextWetMinutes(
	series: readonly RainSample[],
	nowMs: number,
	threshold: number = RAINING_NOW_MM_H
): number | null {
	if (!Number.isFinite(nowMs)) return null;
	let best: number | null = null;
	for (const sample of series) {
		if (sample.mmH === null || sample.mmH < threshold) continue;
		const t = Date.parse(sample.validTsUtc);
		if (!Number.isFinite(t) || t <= nowMs) continue;
		const minutes = (t - nowMs) / 60000;
		if (best === null || minutes < best) best = minutes;
	}
	return best;
}

/** The headline, and the arrival time the same decision implies. */
export interface HeadlineDecision {
	kind: Headline;
	/** Minutes until rain arrives — only ever set on the `eta` kind. */
	etaMin: number | null;
}

/**
 * Which sentence the panel leads with, and the arrival it implies.
 *
 * The rule is that the headline is read from the same field the loop draws,
 * sampled at the same wall-clock instant the loop's clock marker sits on. Text
 * and picture then agree by construction, which is the only way they can be
 * made to agree at all.
 *
 * Two tempting inputs are deliberately *not* used:
 *
 *  - **The raw observation.** The newest composite is 14–24 min old at any
 *    moment a viewer looks — DMI's own delay plus a 10 min composite cadence
 *    plus a cycle that serves for up to 10 min — so it is a measurement of the
 *    recent past, not of now. Letting it win outright is how the panel came to
 *    say "it is raining here now" while the loop's frame for that same minute
 *    showed the point dry. It is still sampled and still served; it is simply
 *    not evidence about *now*.
 *  - **A zero ETA on its own.** The ensemble ETA is a cumulative first-arrival
 *    product: once a cell was over the point at the first ensemble step it
 *    stays at 0 for the whole cycle, long after that cell has gone. A zero
 *    therefore means "the ensemble says rain reached you by now", which the
 *    field can and does contradict — and when it does, the field wins and the
 *    next wet step of the same field becomes the arrival time.
 *
 * `etaNow` must be the counted-down value from `countdownEtaMin`, not the
 * cycle's own field, or the decision flips a cycle late.
 *
 * A cycle that served no field at all (an older sidecar, or a download that
 * failed) leaves the series empty, and the decision falls back to the ETA rule
 * as it stood before any of this existed: absence of evidence is not evidence
 * of a dry sky.
 */
export function headlineDecision(
	etaNow: number | null,
	series: readonly RainSample[],
	nowMs: number
): HeadlineDecision {
	if (series.length === 0) {
		// Legacy path: no field to read, so the ensemble is all there is.
		if (etaNow === null) return { kind: 'no-rain', etaMin: null };
		if (etaNow <= RAINING_NOW_MIN) return { kind: 'raining-now', etaMin: null };
		return { kind: 'eta', etaMin: etaNow };
	}

	const rainNow = rainNowMmH(series, nowMs);
	if (rainNow !== null && rainNow >= RAINING_NOW_MM_H) return { kind: 'raining-now', etaMin: null };
	if (etaNow === null) return { kind: 'no-rain', etaMin: null };
	if (etaNow > RAINING_NOW_MIN) return { kind: 'eta', etaMin: etaNow };

	// The sticky first-arrival case: the ensemble says the rain arrived by
	// now, the field for now says it is not here. Answer with the field's own
	// next wet step, and if it has none, with no rain.
	const next = nextWetMinutes(series, nowMs);
	return next === null ? { kind: 'no-rain', etaMin: null } : { kind: 'eta', etaMin: next };
}

export function headline(t: Catalog, decision: HeadlineDecision): string {
	switch (decision.kind) {
		case 'raining-now':
			return t.panel.headlineRainingNow;
		case 'eta':
			return t.panel.headlineEta(Math.round(decision.etaMin ?? 0));
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
