/**
 * The three headline sentences, and the numbers inside them.
 *
 * Same division of labour as `$lib/format`: nothing user-visible is written
 * here, the catalog owns every phrase, and this module only decides which
 * phrase and what goes into it. Two rules it exists to enforce:
 *
 *  - **A missing measurement is a sentence, not a zero.** Every builder
 *    returns the catalog's "not measured yet" line when its section is null.
 *    "0 warnings" and "not measured yet" look the same in a number and mean
 *    opposite things.
 *  - **The number is bold, the sentence is a sentence.** The catalog holds
 *    whole sentences (a translator needs the whole sentence), and
 *    `emphasise` marks the value tokens inside the finished string, so the
 *    markup never has to be built out of translated fragments.
 */
import type { Catalog, Locale } from '$lib/i18n';
import type {
	HeadlineWarnings,
	PersistenceMargin,
	QualityHeadline,
	RainingNowCheck
} from './schema';

/** A run of a sentence, bold or not. */
export interface Segment {
	text: string;
	strong: boolean;
}

/** One headline card: the sentence, the smaller line under it, and whether it says anything. */
export interface HeadlineCard {
	segments: Segment[];
	detail: string | null;
	measured: boolean;
}

const tag = (locale: Locale): string => (locale === 'da' ? 'da-DK' : 'en-GB');

/** An integer with the locale's grouping — 41 250, 41,250. */
export function countText(value: number, locale: Locale): string {
	try {
		return new Intl.NumberFormat(tag(locale)).format(Math.round(value));
	} catch {
		return String(Math.round(value));
	}
}

/** A decimal with a fixed number of places, in the locale's notation. */
export function decimalText(value: number, locale: Locale, digits = 2): string {
	try {
		return new Intl.NumberFormat(tag(locale), {
			minimumFractionDigits: digits,
			maximumFractionDigits: digits
		}).format(value);
	} catch {
		return value.toFixed(digits);
	}
}

/** A 0–100 percentage, rounded, with the locale's percent spacing. */
export const percentText = (t: Catalog, value: number): string =>
	t.quality.percent(Math.round(value));

/** A 0–1 fraction as a percentage. */
export const fractionText = (t: Catalog, fraction: number): string =>
	percentText(t, fraction * 100);

/** True when `text[index]` continues a number that started before it. */
const isDigit = (text: string, index: number): boolean =>
	index >= 0 && index < text.length && text[index] >= '0' && text[index] <= '9';

/**
 * Split a finished sentence into bold and plain runs.
 *
 * `tokens` are the rendered values, in the order they appear in the sentence;
 * each is looked for after the previous match, so a value that also occurs
 * earlier in the sentence (the "70 %" we said, before the "68 %" that
 * happened) cannot steal the emphasis. A token that is not found is silently
 * left unemphasised — a translation that phrases a number differently loses
 * the bold, not the sentence. Matches are rejected when a digit sits directly
 * against them, so 103 does not match inside 1030.
 */
export function emphasise(sentence: string, tokens: readonly string[]): Segment[] {
	const segments: Segment[] = [];
	let cursor = 0;
	let plainFrom = 0;
	for (const token of tokens) {
		if (token === '') continue;
		let at = sentence.indexOf(token, cursor);
		while (at !== -1 && (isDigit(sentence, at - 1) || isDigit(sentence, at + token.length))) {
			at = sentence.indexOf(token, at + 1);
		}
		if (at === -1) continue;
		if (at > plainFrom) segments.push({ text: sentence.slice(plainFrom, at), strong: false });
		segments.push({ text: token, strong: true });
		cursor = at + token.length;
		plainFrom = cursor;
	}
	if (plainFrom < sentence.length) {
		segments.push({ text: sentence.slice(plainFrom), strong: false });
	}
	return segments;
}

const notMeasured = (t: Catalog): HeadlineCard => ({
	segments: [{ text: t.quality.notMeasured, strong: false }],
	detail: null,
	measured: false
});

/**
 * "When we say 70 %, it rains 68 % of the time (radar) / 64 % (gauges)."
 *
 * Either truth may be missing on its own — the gauge comparison needs
 * stations that reported, the radar one needs a backtest window — so there is
 * a sentence for each, and the "not measured" line only when both are absent.
 */
export function reliabilityCard(
	t: Catalog,
	locale: Locale,
	reliability: QualityHeadline['reliability']
): HeadlineCard {
	const { radar, gauge } = reliability;
	if (!radar && !gauge) return notMeasured(t);

	const said = percentText(t, (radar ?? gauge!).said_pct);
	const radarPct = radar ? percentText(t, radar.happened_pct) : null;
	const gaugePct = gauge ? percentText(t, gauge.happened_pct) : null;

	const sentence =
		radar && gauge
			? t.quality.headline.reliabilityBoth(said, radarPct!, gaugePct!)
			: radar
				? t.quality.headline.reliabilityRadar(said, radarPct!)
				: t.quality.headline.reliabilityGauge(said, gaugePct!);
	const tokens = [radarPct, gaugePct].filter((token): token is string => token !== null);

	const detail =
		radar && gauge
			? radar.lead_min === gauge.lead_min
				? t.quality.headline.reliabilityLead(radar.lead_min, countText(radar.n + gauge.n, locale))
				: t.quality.headline.reliabilityLeadPair(radar.lead_min, gauge.lead_min)
			: t.quality.headline.reliabilityLead(
					(radar ?? gauge!).lead_min,
					countText((radar ?? gauge!).n, locale)
				);

	return { segments: emphasise(sentence, tokens), detail, measured: true };
}

/**
 * "Of 148 warnings in the last 30 days, 103 were followed by rain at the
 * gauge; 45 were false alarms; the median warning came 4 minutes late."
 *
 * The sign convention is the schema's: a positive `p50` means the rain had
 * already started when the warning said it would arrive — the warning was
 * late. Zero is neither, and gets its own phrase rather than "0 minutes
 * early", which reads as a measurement of nothing.
 */
export function warningsCard(
	t: Catalog,
	locale: Locale,
	warnings: HeadlineWarnings | null
): HeadlineCard {
	if (!warnings) return notMeasured(t);

	const total = countText(warnings.warnings, locale);
	const hits = countText(warnings.hits, locale);
	const falseAlarms = countText(warnings.false_alarms, locale);
	const p50 = warnings.lead_error_min.p50;
	const minutes = countText(Math.abs(p50), locale);

	const sentence =
		Math.round(p50) === 0
			? t.quality.headline.warningsOnTime(total, warnings.window_days, hits, falseAlarms)
			: p50 > 0
				? t.quality.headline.warningsLate(
						total,
						warnings.window_days,
						hits,
						falseAlarms,
						minutes
					)
				: t.quality.headline.warningsEarly(
						total,
						warnings.window_days,
						hits,
						falseAlarms,
						minutes
					);

	const tokens = Math.round(p50) === 0 ? [hits, falseAlarms] : [hits, falseAlarms, minutes];
	const detail = t.quality.headline.warningsRates(
		fractionText(t, warnings.pod),
		fractionText(t, warnings.far),
		countText(warnings.n_stations, locale)
	);

	return { segments: emphasise(sentence, tokens), detail, measured: true };
}

/**
 * "At +10 min we beat 'assume nothing moves' by 6 points of CSI."
 *
 * A margin that is zero or negative is said out loud in the same place. A
 * page about how good we are that can only phrase good news is not a
 * measurement, it is an advertisement.
 */
export function marginCard(
	t: Catalog,
	locale: Locale,
	margin: PersistenceMargin | null
): HeadlineCard {
	if (!margin) return notMeasured(t);

	const points = (margin.csi_advection - margin.csi_persistence) * 100;
	const rounded = Math.round(points);
	const pointsText = countText(Math.abs(points), locale);

	const sentence =
		rounded === 0
			? t.quality.headline.marginTied(margin.horizon_min)
			: rounded > 0
				? t.quality.headline.marginBeats(margin.horizon_min, pointsText)
				: t.quality.headline.marginBehind(margin.horizon_min, pointsText);

	const detail = t.quality.headline.marginDetail(
		decimalText(margin.csi_advection, locale),
		decimalText(margin.csi_persistence, locale),
		countText(margin.frames, locale)
	);

	return {
		segments: emphasise(sentence, rounded === 0 ? [] : [pointsText]),
		detail,
		measured: true
	};
}

/** The "raining now" check: one sentence, plus the raw-radar comparison. */
export interface RainingNowLines {
	sentence: string;
	comparison: string;
	detail: string;
}

export function rainingNowLines(
	t: Catalog,
	locale: Locale,
	check: RainingNowCheck | null
): RainingNowLines | null {
	if (!check) return null;
	return {
		sentence: t.quality.rainingNow.sentence(
			fractionText(t, check.agreement),
			fractionText(t, check.pod),
			fractionText(t, check.far)
		),
		comparison: t.quality.rainingNow.comparison(fractionText(t, check.observation_agreement)),
		detail: t.quality.rainingNow.detail(countText(check.n_slots, locale))
	};
}
