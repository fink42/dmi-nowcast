/**
 * The headline sentences.
 *
 * Two things are pinned here above all: a missing measurement produces the
 * "not measured yet" sentence and never a zero, and the early/late wording
 * follows the schema's sign convention (positive = the rain was already
 * falling when we said it would arrive, so the warning was late).
 */
import { describe, expect, it } from 'vitest';
import { da } from '$lib/i18n/da';
import { en } from '$lib/i18n/en';
import {
	countText,
	decimalText,
	emphasise,
	marginCard,
	rainingNowLines,
	reliabilityCard,
	warningsCard,
	type HeadlineCard
} from './sentences';
import type { HeadlineWarnings, PersistenceMargin, QualityHeadline } from './schema';

const text = (card: HeadlineCard): string => card.segments.map((s) => s.text).join('');
const bold = (card: HeadlineCard): string[] =>
	card.segments.filter((s) => s.strong).map((s) => s.text);

const reliability = (
	radar: { said: number; happened: number; lead: number; n: number } | null,
	gauge: { said: number; happened: number; lead: number; n: number } | null
): QualityHeadline['reliability'] => ({
	radar: radar && {
		lead_min: radar.lead,
		said_pct: radar.said,
		happened_pct: radar.happened,
		n: radar.n
	},
	gauge: gauge && {
		lead_min: gauge.lead,
		said_pct: gauge.said,
		happened_pct: gauge.happened,
		n: gauge.n
	}
});

const warnings = (p50: number, extra: Partial<HeadlineWarnings> = {}): HeadlineWarnings => ({
	window_days: 30,
	n_stations: 96,
	warnings: 148,
	hits: 103,
	false_alarms: 45,
	misses: 37,
	pod: 0.736,
	far: 0.304,
	lead_error_min: { p25: p50 - 12, p50, p75: p50 + 12 },
	...extra
});

const margin = (advection: number, persistence: number): PersistenceMargin => ({
	horizon_min: 10,
	csi_advection: advection,
	csi_persistence: persistence,
	frames: 4032,
	from: '2026-06-01T00:00:00Z',
	to: '2026-09-01T00:00:00Z'
});

describe('emphasise', () => {
	it('marks the tokens and leaves the rest plain', () => {
		const segments = emphasise('It rains 68 % of the time.', ['68 %']);
		expect(segments).toEqual([
			{ text: 'It rains ', strong: false },
			{ text: '68 %', strong: true },
			{ text: ' of the time.', strong: false }
		]);
	});

	it('takes the tokens in the order they appear, not the first match', () => {
		// "70 %" is what we said and must not steal the emphasis from "68 %".
		const segments = emphasise('When we say 70 %, it rains 68 % of the time.', ['68 %']);
		expect(segments.filter((s) => s.strong)).toEqual([{ text: '68 %', strong: true }]);
		expect(segments.map((s) => s.text).join('')).toBe(
			'When we say 70 %, it rains 68 % of the time.'
		);
	});

	it('never matches inside a longer number', () => {
		const segments = emphasise('Of 1030 warnings, 103 were followed by rain.', ['103']);
		const strong = segments.filter((s) => s.strong);
		expect(strong).toHaveLength(1);
		expect(segments[0].text).toBe('Of 1030 warnings, ');
	});

	it('leaves a token it cannot find unemphasised rather than throwing', () => {
		const segments = emphasise('Nothing to see here.', ['42', '']);
		expect(segments).toEqual([{ text: 'Nothing to see here.', strong: false }]);
	});

	it('returns the whole sentence when there are no tokens', () => {
		expect(emphasise('Plain.', [])).toEqual([{ text: 'Plain.', strong: false }]);
	});
});

describe('the reliability sentence', () => {
	it('names both truths when both were measured', () => {
		const card = reliabilityCard(
			da,
			'da',
			reliability(
				{ said: 70, happened: 67.8, lead: 30, n: 41250 },
				{ said: 70, happened: 63.5, lead: 30, n: 1180 }
			)
		);
		expect(card.measured).toBe(true);
		expect(text(card)).toContain('70 %');
		expect(bold(card)).toEqual(['68 %', '64 %']);
		expect(card.detail).toContain('30');
	});

	it('says only what it has when one truth is missing', () => {
		const radarOnly = reliabilityCard(
			en,
			'en',
			reliability({ said: 70, happened: 67.8, lead: 30, n: 41250 }, null)
		);
		expect(bold(radarOnly)).toEqual(['68%']);
		expect(text(radarOnly)).toContain('radar');
		expect(text(radarOnly)).not.toContain('gauges,');

		const gaugeOnly = reliabilityCard(
			en,
			'en',
			reliability(null, { said: 70, happened: 63.5, lead: 45, n: 1180 })
		);
		expect(bold(gaugeOnly)).toEqual(['64%']);
		expect(gaugeOnly.detail).toContain('45');
	});

	it('says both leads when the two truths were measured at different ones', () => {
		const card = reliabilityCard(
			en,
			'en',
			reliability(
				{ said: 70, happened: 68, lead: 30, n: 100 },
				{ said: 70, happened: 64, lead: 60, n: 50 }
			)
		);
		expect(card.detail).toContain('30');
		expect(card.detail).toContain('60');
	});

	it('is "not measured yet" when neither truth is there', () => {
		const card = reliabilityCard(da, 'da', reliability(null, null));
		expect(card.measured).toBe(false);
		expect(text(card)).toBe(da.quality.notMeasured);
		expect(card.detail).toBeNull();
		expect(text(card)).not.toContain('0');
	});
});

describe('the warnings sentence', () => {
	it('says "late" when the rain had already started', () => {
		const card = warningsCard(en, 'en', warnings(4));
		expect(text(card)).toContain('late');
		expect(text(card)).not.toContain('early');
		expect(bold(card)).toEqual(['103', '45', '4']);
	});

	it('says "early" when the rain came after we said', () => {
		const card = warningsCard(en, 'en', warnings(-11));
		expect(text(card)).toContain('early');
		expect(text(card)).not.toContain('late');
		// The minutes are said as a magnitude; the sign lives in the wording.
		expect(text(card)).toContain('11 minutes');
		expect(text(card)).not.toContain('-11');
	});

	it('has its own phrase for a median that is neither', () => {
		const card = warningsCard(da, 'da', warnings(0));
		expect(text(card)).toContain('ramte tiden');
		expect(bold(card)).toEqual(['103', '45']);
	});

	it('puts the rates in the smaller line', () => {
		const card = warningsCard(en, 'en', warnings(4));
		expect(card.detail).toContain('74%');
		expect(card.detail).toContain('30%');
		expect(card.detail).toContain('96');
	});

	it('is "not measured yet" rather than zero warnings', () => {
		const card = warningsCard(en, 'en', null);
		expect(card.measured).toBe(false);
		expect(text(card)).toBe(en.quality.notMeasured);
	});
});

describe('the persistence-margin sentence', () => {
	it('states the margin in points of CSI', () => {
		const card = marginCard(en, 'en', margin(0.612, 0.548));
		expect(text(card)).toContain('beat');
		expect(bold(card)).toEqual(['6']);
		expect(card.detail).toContain(decimalText(0.612, 'en'));
		expect(card.detail).toContain(decimalText(0.548, 'en'));
	});

	it('says so out loud when advection is behind the baseline', () => {
		const card = marginCard(en, 'en', margin(0.51, 0.58));
		expect(text(card)).toContain('behind');
		expect(bold(card)).toEqual(['7']);
	});

	it('has a phrase for a dead heat, with nothing bold', () => {
		const card = marginCard(da, 'da', margin(0.6, 0.602));
		expect(text(card)).toContain('lige med');
		expect(bold(card)).toEqual([]);
	});

	it('never matches the margin inside the horizon', () => {
		// Horizon 60, margin 6: the "6" of "60" must not be the bold one.
		const sixty = { ...margin(0.612, 0.548), horizon_min: 60 };
		const card = marginCard(en, 'en', sixty);
		const first = card.segments[0];
		expect(first.strong).toBe(false);
		expect(first.text.startsWith('60')).toBe(true);
		expect(bold(card)).toEqual(['6']);
	});

	it('is "not measured yet" when the comparison has not run', () => {
		expect(marginCard(da, 'da', null).measured).toBe(false);
	});
});

describe('the raining-now lines', () => {
	it('carries the agreement, the rates and the raw-radar comparison', () => {
		const lines = rainingNowLines(en, 'en', {
			n_slots: 8640,
			agreement: 0.871,
			pod: 0.824,
			far: 0.187,
			observation_agreement: 0.793,
			from: '2026-08-06T00:00:00Z',
			to: '2026-09-05T00:00:00Z'
		});
		expect(lines!.sentence).toContain('87%');
		expect(lines!.sentence).toContain('82%');
		expect(lines!.sentence).toContain('19%');
		expect(lines!.comparison).toContain('79%');
		expect(lines!.detail).toContain(countText(8640, 'en'));
	});

	it('is null when the check has not run', () => {
		expect(rainingNowLines(da, 'da', null)).toBeNull();
	});
});

describe('number formatting', () => {
	it('groups counts the way the locale does', () => {
		expect(countText(41250, 'en')).toBe('41,250');
		expect(countText(41250, 'da').replace(/ /g, ' ')).toBe('41.250');
	});

	it('writes decimals the way the locale does', () => {
		expect(decimalText(0.612, 'en')).toBe('0.61');
		expect(decimalText(0.612, 'da')).toBe('0,61');
	});
});
