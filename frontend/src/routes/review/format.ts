/**
 * Words and numbers for the review page — the render boundary, and nothing
 * behind it.
 *
 * Every figure the components print comes out of `$lib/review/**`; this file
 * only decides how it reads. It exists so the six panels spell the same
 * thing the same way: one sentence for "the gauge did not report", one
 * spelling of a stamp, one rule for what a null looks like.
 *
 * Two conventions run through all of it:
 *
 *  - **UTC, stated.** The bundle is UTC end to end — radar stamps, slot
 *    ends, `known_until`, the judgement's own cursor — and a reviewer
 *    comparing "the warning went at 13:45" against "the gauge's slot ends
 *    13:50" must be comparing the same clock. So stamps render in UTC with
 *    the letters on them, and the viewer's own zone appears only as a
 *    tooltip on the event header.
 *  - **A null is a sentence, not a dash.** `NOT_MEASURED` is the word for
 *    every unmeasured quantity in the tool, because a dash in a column of
 *    numbers reads as a zero and a zero here is evidence the bundle does not
 *    have.
 *
 * Strings are English literals. `i18n/catalog.test.ts` enforces exact key
 * parity between the `da` and `en` catalogs, and this route is dev-only —
 * putting its vocabulary in the catalog would ship it to the public site and
 * demand a Danish translation of `fa_drizzle_below_gauge_floor`.
 */
import { percent } from '$lib/format';
import type { Decision, EngineAction } from '$lib/review/schema';
import type { SlotState, UnknownReason } from '$lib/review/truth';

/** What an unmeasured quantity says. Never a dash, never a zero. */
export const NOT_MEASURED = 'not measured';

/**
 * An ISO instant as epoch milliseconds, or null.
 *
 * A private copy of `timeline.msOf`'s contract rather than an import of it,
 * because everything here is formatting and nothing may place a marker: a
 * stamp that will not parse renders as `NOT_MEASURED` and stops there.
 */
function parsed(iso: string | null | undefined): Date | null {
	if (typeof iso !== 'string' || iso.trim() === '') return null;
	const ms = Date.parse(iso);
	return Number.isFinite(ms) ? new Date(ms) : null;
}

/** `13:45` — UTC, always, whatever offset the bundle wrote. */
export function utcClock(iso: string | null | undefined): string | null {
	const date = parsed(iso);
	return date === null ? null : date.toISOString().slice(11, 16);
}

/** `13:45 UTC` — the same instant with the frame of reference on it. */
export function utcTime(iso: string | null | undefined): string | null {
	const clock = utcClock(iso);
	return clock === null ? null : `${clock} UTC`;
}

/** `2026-06-12` — the day, for a header that has to place the event. */
export function utcDay(iso: string | null | undefined): string | null {
	const date = parsed(iso);
	return date === null ? null : date.toISOString().slice(0, 10);
}

/** `2026-06-12 13:45 UTC` — a stamp that can stand on its own in a tooltip. */
export function utcStamp(iso: string | null | undefined): string | null {
	const date = parsed(iso);
	return date === null ? null : `${date.toISOString().slice(0, 10)} ${date.toISOString().slice(11, 16)} UTC`;
}

/**
 * The same instant on the viewer's own clock, for a `title` beside a UTC
 * stamp. Secondary on purpose: the tool reasons in UTC, and a local time in
 * the middle of a comparison is how a reviewer loses an hour to DST.
 */
export function localStamp(iso: string | null | undefined): string | null {
	const date = parsed(iso);
	if (date === null) return null;
	try {
		return `${new Intl.DateTimeFormat(undefined, {
			dateStyle: 'medium',
			timeStyle: 'short'
		}).format(date)} local`;
	} catch {
		return null;
	}
}

/** `62 %` from a 0–1 fraction, rounded the way the rule rounds it. */
export function pctText(fraction: number | null | undefined): string | null {
	if (typeof fraction !== 'number' || !Number.isFinite(fraction)) return null;
	return `${percent(fraction)} %`;
}

/**
 * How far the probability sat above (positive) or below (negative) its
 * threshold, in percentage POINTS.
 *
 * Both sides are integers before the subtraction — the probability rounded
 * exactly as `pctText` shows it — so the margin on screen always equals the
 * two numbers beside it. A margin computed on the fractions would print
 * "62 % against 45 %, +16 points" and look like a bug.
 */
export function marginPoints(
	fraction: number | null | undefined,
	thresholdPct: number | null | undefined
): number | null {
	if (typeof fraction !== 'number' || !Number.isFinite(fraction)) return null;
	if (typeof thresholdPct !== 'number' || !Number.isFinite(thresholdPct)) return null;
	return percent(fraction) - Math.round(thresholdPct);
}

/** `+16` / `−3` — a signed count of points, with a real minus sign. */
export function signedPoints(points: number | null): string | null {
	if (points === null || !Number.isFinite(points)) return null;
	return points < 0 ? `−${Math.abs(points)}` : `+${points}`;
}

/** A number at a fixed precision, or null. Null is the caller's to name. */
export function numberText(
	value: number | null | undefined,
	digits = 1
): string | null {
	if (typeof value !== 'number' || !Number.isFinite(value)) return null;
	return value.toFixed(digits);
}

/** `12 min`, rounded to the minute the reviewer is judging. */
export function minutesText(value: number | null | undefined): string | null {
	if (typeof value !== 'number' || !Number.isFinite(value)) return null;
	return `${Math.round(value)} min`;
}

/** `2.4 mm/h`, or null. The unit rides with the number so it cannot drift. */
export function mmHText(value: number | null | undefined, digits = 1): string | null {
	const text = numberText(value, digits);
	return text === null ? null : `${text} mm/h`;
}

/** `0.4 mm`, the depth in a 10-minute slot. */
export function mmText(value: number | null | undefined, digits = 1): string | null {
	const text = numberText(value, digits);
	return text === null ? null : `${text} mm`;
}

/**
 * Why an instrument had nothing to say. Each reason is a different fact and
 * they are worth telling apart: "the gauge went quiet" and "the record has
 * not reached here yet" lead a reviewer to different tags.
 */
export function unknownReasonText(reason: UnknownReason): string {
	switch (reason) {
		case 'no_series':
			return 'this bundle carries no series for it';
		case 'not_reported':
			return 'it did not report for this slot';
		case 'beyond_known_until':
			return 'past known_until — the record does not reach this far yet';
		case 'before_series':
			return 'before the series starts';
		case 'after_series':
			return 'after the series ends';
		default:
			return 'it reported no value for this slot';
	}
}

/**
 * The three states in one word each.
 *
 * `unknown` gets "not known" rather than a blank or a dash, and the
 * components draw it hatched: a silent gauge read as a dry one is a gap in
 * the record presented as evidence of a false alarm, which is the single
 * mistake this tool exists to avoid.
 */
export function slotWord(state: SlotState | null): string {
	if (state === null) return 'not known';
	switch (state.state) {
		case 'wet':
			return 'wet';
		case 'dry':
			return 'dry';
		default:
			return 'not known';
	}
}

/** The sentence under the word: the depth, or the reason there is none. */
export function slotDetail(state: SlotState | null, unit: 'mm' | 'mm/h'): string {
	if (state === null) return 'no reading at the cursor';
	if (state.state === 'unknown') return unknownReasonText(state.reason);
	const value = unit === 'mm' ? mmText(state.mm, 1) : mmHText(state.mm, 1);
	return value ?? 'reported, with no value';
}

export interface ActionWords {
	label: string;
	reading: string;
}

/**
 * What the engine did, and what it means. `notify` and `already_raining`
 * both disarm the station, and telling them apart is the whole of the
 * `miss_arm_consumed_already_raining` tag — so the difference is spelled out
 * rather than left to the code word.
 */
export function actionText(action: EngineAction | null): ActionWords {
	switch (action) {
		case 'notify':
			return { label: 'notify', reading: 'a push went out — and the arm was spent' };
		case 'already_raining':
			return {
				label: 'already raining',
				reading: 'the engine judged rain to be falling already: no push, and the arm was spent silently'
			};
		case 'deferred_quiet':
			return { label: 'deferred (quiet hours)', reading: 'over threshold, held back by quiet hours' };
		case 'skipped':
			return { label: 'skipped', reading: 'the engine passed over this row without comparing it' };
		case 'none':
			return { label: 'none', reading: 'compared, and under the threshold' };
		default:
			return { label: NOT_MEASURED, reading: 'no traced action at this instant' };
	}
}

export interface LeadRow {
	leadMin: number;
	/** The curve's probability at this lead, or null where it was unserved. */
	pRain: number | null;
	/** The post-processed probability, or null. */
	pPost: number | null;
	/** True for the lead the rule actually decided on. */
	isDecisionLead: boolean;
}

/**
 * The per-lead probabilities as rows, shortest lead first.
 *
 * The keys are minutes written as strings, so they sort as text into 20, 30,
 * 45, 60 by luck and into 100, 20, 30 by arithmetic — hence the numeric
 * sort. Leads present in either map get a row: a lead the post-processor
 * could not serve but the curve could is exactly the case
 * `miss_no_probability` is about, and dropping it would hide it.
 */
export function leadRows(decision: Decision | null): LeadRow[] {
	if (decision === null) return [];
	const leads = new Set<string>([...Object.keys(decision.p_rain), ...Object.keys(decision.p_post)]);
	return [...leads]
		.map((key) => ({
			leadMin: Number(key),
			pRain: decision.p_rain[key] ?? null,
			pPost: decision.p_post[key] ?? null,
			isDecisionLead: Number(key) === decision.p_decision_lead_min
		}))
		.filter((row) => Number.isFinite(row.leadMin))
		.sort((a, b) => a.leadMin - b.leadMin);
}

export interface Provenance {
	/** True when the probabilities were fitted on the events they score. */
	contaminated: boolean;
	label: string;
	reading: string;
}

/**
 * What `rule.probability_provenance` means for the sample in front of the
 * reviewer.
 *
 * This is the one manifest field that decides whether the review is honest,
 * so it is rendered where the reviewer will see it rather than filed under
 * provenance: `in_sample_fill` means the model had already seen these very
 * events, which makes false alarms look rarer and stranger than they are and
 * quietly biases every mechanism count drawn from the tally.
 */
export function provenanceText(value: string | null | undefined): Provenance {
	switch (value) {
		case 'out_of_fold':
			return {
				contaminated: false,
				label: 'out of fold',
				reading:
					'Probabilities come from a leave-one-month-out fit: the model never saw the event it is being judged on.'
			};
		case 'in_sample_fill':
			return {
				contaminated: true,
				label: 'in sample',
				reading:
					'The post-processor was fitted on months that include these events, so it had already seen them. False alarms will look rarer and stranger than they are.'
			};
		case 'stored':
			return {
				contaminated: true,
				label: 'stored',
				reading:
					'Probabilities are the ones the archive stored, under whatever model was live at the time. Their provenance is not recoverable from the bundle.'
			};
		case 'mixed':
			return {
				contaminated: true,
				label: 'mixed',
				reading:
					'Some probabilities are out of fold and some are not. Treat any tally drawn from this bundle as contaminated until the split is known.'
			};
		default:
			return {
				contaminated: true,
				label: NOT_MEASURED,
				reading:
					'This bundle does not say where its probabilities came from, so it cannot be ruled out that they were fitted on these events.'
			};
	}
}

/** Which of the two frames the map is showing, in words for a button. */
export function frameModeText(mode: 'truth' | 'service'): string {
	return mode === 'truth' ? 'Truth — the newest picture at the cursor' : 'Service — the picture the estimate stood on';
}
