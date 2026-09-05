/**
 * The "how we choose when to warn" table: one row per horizon.
 *
 * Same division of labour as `./sentences` and `$lib/format` — the catalog
 * owns every phrase, this module only decides which phrase and what goes
 * into it. Two rules it exists to enforce:
 *
 *  - **A null rate is a dash, not a zero.** Precision over zero warnings is
 *    undefined, and printing 0 % there would read as "it was wrong every
 *    time" rather than "it never warned".
 *  - **A horizon the fit cannot speak for says so.** `insufficient` is not
 *    hidden and it is not rendered as a threshold: the row keeps its place,
 *    shows the default it falls back to, and carries the note.
 */
import type { Catalog, Locale } from '$lib/i18n';
import { countText } from './sentences';
import type { LeadThresholdRow, QualityThresholds } from './schema';

/** One rendered row of the table. Every field is display-ready text. */
export interface ThresholdRow {
	key: number;
	horizon: string;
	threshold: string;
	precision: string;
	recall: string;
	f1: string;
	warnings: string;
	/** The note under the row, or null when there is nothing to flag. */
	note: string | null;
	/** True while the fit has too little evidence at this horizon. */
	insufficient: boolean;
}

/** A 0–1 fraction as a whole percent, or the dash for "not measured". */
const rate = (t: Catalog, value: number | null): string =>
	value === null ? t.quality.thresholds.empty : t.quality.percent(Math.round(value * 100));

/** A 0–1 score to two places, or the dash. */
const score = (t: Catalog, value: number | null, locale: Locale): string => {
	if (value === null) return t.quality.thresholds.empty;
	try {
		return new Intl.NumberFormat(locale === 'da' ? 'da-DK' : 'en-GB', {
			minimumFractionDigits: 2,
			maximumFractionDigits: 2
		}).format(value);
	} catch {
		return value.toFixed(2);
	}
};

/**
 * The note under one row.
 *
 * Insufficient evidence comes first: when the fit could not pick a threshold
 * at all, whether the radar would have agreed with a pick it never made is
 * not worth a sentence. The disagreement note is deliberately blunt about
 * which truth won, because that is the decision a reader is entitled to
 * disagree with.
 */
function noteFor(t: Catalog, row: LeadThresholdRow): string | null {
	if (row.insufficient) return t.quality.thresholds.insufficient;
	if (row.agrees_with_radar === false) return t.quality.thresholds.disagrees;
	return null;
}

/**
 * `QualityThresholds` → the rows the page renders, in horizon order. An
 * absent section is an empty list, which the page renders as "not fitted
 * yet" rather than as an empty table.
 */
export function thresholdRows(
	t: Catalog,
	locale: Locale,
	thresholds: QualityThresholds | null
): ThresholdRow[] {
	if (thresholds === null) return [];
	return thresholds.leads.map((row) => ({
		key: row.lead_min,
		horizon: t.quality.thresholds.horizon(row.lead_min),
		// An insufficient horizon shows the default it actually warns at, not
		// a blank: the rule still fires, it is just not fitted for this lead.
		threshold: t.quality.percent(row.threshold_pct ?? thresholds.fallback_threshold_pct),
		precision: rate(t, row.precision),
		recall: rate(t, row.recall),
		f1: score(t, row.f1, locale),
		warnings: countText(row.warnings, locale),
		note: noteFor(t, row),
		insufficient: row.insufficient
	}));
}

/**
 * The sentence explaining the rule, with the minimum useful lead in it when
 * the producer stated one.
 */
export function thresholdIntro(t: Catalog, thresholds: QualityThresholds | null): string {
	const lead = thresholds?.objective?.min_useful_lead_min ?? null;
	return lead === null
		? t.quality.thresholds.introNoLead
		: t.quality.thresholds.intro(Math.round(lead));
}
