/**
 * The one-knob machinery: what the options endpoint says, what the hidden
 * override may say, and which sentence the panel ends up printing.
 *
 * The property under test throughout is that no answer here is ever a
 * *choice* — the threshold comes from the served table, falls back when the
 * table cannot speak for a horizon, and is only overruled by something the
 * user typed into the address bar on purpose.
 */
import { describe, expect, it } from 'vitest';
import { da } from '$lib/i18n/da';
import { en } from '$lib/i18n/en';
import {
	DEFAULT_FALLBACK_THRESHOLD_PCT,
	FALLBACK_LEAD_OPTIONS_MIN,
	effectiveThreshold,
	fittedDate,
	isOverridePct,
	nearestLead,
	parsePushOptions,
	thresholdFact,
	thresholdOverrideFromUrl,
	type PushOptions
} from './thresholds';

const SERVER = {
	lead_options: [20, 30, 45, 60],
	fallback_threshold_pct: 40,
	fitted_at_utc: '2026-09-03T02:11:07Z',
	thresholds: {
		20: { threshold_pct: 35, source: 'table' },
		30: { threshold_pct: 35, source: 'table' },
		45: { threshold_pct: 30, source: 'table' },
		60: { threshold_pct: 40, source: 'fallback' }
	}
};

describe('parsePushOptions', () => {
	it('camel-cases a full answer and keys the table by horizon', () => {
		const options = parsePushOptions(SERVER);
		expect(options.leadOptionsMin).toEqual([20, 30, 45, 60]);
		expect(options.fallbackThresholdPct).toBe(40);
		expect(options.fittedAtUtc).toBe('2026-09-03T02:11:07Z');
		expect(options.thresholds[45]).toEqual({ thresholdPct: 30, source: 'table' });
		expect(options.thresholds[60]).toEqual({ thresholdPct: 40, source: 'fallback' });
	});

	it('accepts an answer with no thresholds at all', () => {
		const options = parsePushOptions({
			lead_options: [20, 30, 45, 60],
			fallback_threshold_pct: 40,
			fitted_at_utc: null
		});
		expect(options.thresholds).toEqual({});
		expect(options.fittedAtUtc).toBeNull();
		expect(options.fallbackThresholdPct).toBe(40);
	});

	it('falls back for anything that is not a well-formed answer', () => {
		for (const value of [null, undefined, 'html', [], 42]) {
			const options = parsePushOptions(value);
			expect(options.leadOptionsMin, String(value)).toEqual(FALLBACK_LEAD_OPTIONS_MIN);
			expect(options.fallbackThresholdPct).toBe(DEFAULT_FALLBACK_THRESHOLD_PCT);
			expect(options.thresholds).toEqual({});
		}
	});

	it('repairs the parts it cannot read and keeps the parts it can', () => {
		const options = parsePushOptions({
			lead_options: 'nope',
			fallback_threshold_pct: null,
			fitted_at_utc: 'the third of September',
			thresholds: {
				20: { threshold_pct: 35, source: 'table' },
				// No number in it: says nothing, so it falls back like an
				// absent row rather than becoming a threshold of zero.
				30: { source: 'table' },
				45: 'nope',
				notALead: { threshold_pct: 50, source: 'table' }
			}
		});
		expect(options.leadOptionsMin).toEqual(FALLBACK_LEAD_OPTIONS_MIN);
		expect(options.fallbackThresholdPct).toBe(DEFAULT_FALLBACK_THRESHOLD_PCT);
		expect(options.fittedAtUtc).toBeNull();
		expect(Object.keys(options.thresholds)).toEqual(['20']);
	});

	it('drops a horizon list that has nothing usable in it', () => {
		expect(parsePushOptions({ lead_options: [0, -5, 'x'] }).leadOptionsMin).toEqual(
			FALLBACK_LEAD_OPTIONS_MIN
		);
	});
});

describe('thresholdOverrideFromUrl', () => {
	const parse = thresholdOverrideFromUrl;

	it('reads a whole number in range, with or without the leading ?', () => {
		expect(parse('?threshold=55')).toBe(55);
		expect(parse('threshold=20')).toBe(20);
		expect(parse('?lat=55.6&lon=12.5&threshold=80')).toBe(80);
		expect(parse('https://example.org/?threshold=45')).toBe(45);
	});

	it('refuses anything outside 20–80 rather than clamping it', () => {
		for (const search of ['?threshold=19', '?threshold=81', '?threshold=0', '?threshold=999']) {
			expect(parse(search), search).toBeNull();
		}
	});

	it('refuses anything that is not a whole number', () => {
		for (const search of ['?threshold=', '?threshold=abc', '?threshold=35.5', '?threshold=NaN']) {
			expect(parse(search), search).toBeNull();
		}
	});

	it('is null when the parameter is simply not there', () => {
		expect(parse('')).toBeNull();
		expect(parse('?lat=55.6&lon=12.5')).toBeNull();
	});
});

describe('isOverridePct', () => {
	it('accepts the range the override is allowed to name', () => {
		expect(isOverridePct(20)).toBe(true);
		expect(isOverridePct(80)).toBe(true);
		expect(isOverridePct(19.9)).toBe(false);
		expect(isOverridePct(Number.NaN)).toBe(false);
	});
});

describe('nearestLead', () => {
	const options = [20, 30, 45, 60];

	it('keeps a horizon that is on the list', () => {
		for (const lead of options) expect(nearestLead(lead, options)).toBe(lead);
	});

	it('picks the nearest for one that is not', () => {
		expect(nearestLead(33, options)).toBe(30);
		expect(nearestLead(50, options)).toBe(45);
		expect(nearestLead(5, options)).toBe(20);
		expect(nearestLead(120, options)).toBe(60);
	});

	it('gives a tie to the shorter horizon', () => {
		// 25 is 5 from both 20 and 30; warning earlier is the safer default.
		expect(nearestLead(25, options)).toBe(20);
	});

	it('hands the value back when there is nothing to choose from', () => {
		expect(nearestLead(33, [])).toBe(33);
	});
});

describe('effectiveThreshold', () => {
	const options = parsePushOptions(SERVER);

	it('uses the fitted value for a horizon the table covers', () => {
		expect(effectiveThreshold(options, 45)).toEqual({ pct: 30, source: 'table' });
	});

	it('falls back for a horizon the table marks fallback', () => {
		expect(effectiveThreshold(options, 60)).toEqual({ pct: 40, source: 'fallback' });
	});

	it('falls back for a horizon the table has never heard of', () => {
		expect(effectiveThreshold(options, 90)).toEqual({ pct: 40, source: 'fallback' });
	});

	it('lets a valid override win over everything', () => {
		expect(effectiveThreshold(options, 45, 55)).toEqual({ pct: 55, source: 'override' });
		expect(effectiveThreshold(options, 60, 25)).toEqual({ pct: 25, source: 'override' });
	});

	it('ignores an override outside the range', () => {
		expect(effectiveThreshold(options, 45, 5)).toEqual({ pct: 30, source: 'table' });
	});
});

describe('fittedDate', () => {
	it('renders the day without a year or a clock', () => {
		// The month abbreviation is ICU's ("Sept" on current CLDR, "Sep" on
		// older); the contract here is the shape, not the exact spelling.
		expect(fittedDate('2026-09-03T02:11:07Z', 'en')).toMatch(/^3 Sep/);
		expect(fittedDate('2026-09-03T02:11:07Z', 'da')).toMatch(/^3\. sep/);
		expect(fittedDate('2026-09-03T02:11:07Z', 'en')).not.toContain('2026');
	});

	it('is null for a stamp that will not parse', () => {
		expect(fittedDate('the third of September', 'en')).toBeNull();
	});
});

describe('thresholdFact', () => {
	const options = parsePushOptions(SERVER);

	it('names the fitted threshold and the day it was fitted', () => {
		const fact = thresholdFact(en, 'en', options, 20);
		expect(fact).toMatchObject({ pct: 35, source: 'table', override: null });
		expect(fact.fact).toContain('35 %');
		expect(fact.fact).toMatch(/3 Sep/);
		expect(thresholdFact(da, 'da', options, 20).fact).toContain('regnmålere');
	});

	it('drops the day when the table carries no usable one', () => {
		const undated: PushOptions = { ...options, fittedAtUtc: null };
		const fact = thresholdFact(en, 'en', undated, 20);
		expect(fact.fact).toContain('35 %');
		expect(fact.fact).not.toContain('on ');
	});

	it('says the default is standing in for a horizon with no fit', () => {
		const fact = thresholdFact(en, 'en', options, 60);
		expect(fact).toMatchObject({ pct: 40, source: 'fallback' });
		expect(fact.fact).toContain('40 %');
		expect(fact.fact).toContain('default');
		expect(thresholdFact(da, 'da', options, 60).fact).toContain('standard');
	});

	it('keeps the fitted line and adds the override on top of it', () => {
		const fact = thresholdFact(en, 'en', options, 20, 55);
		// The fitted value is what everyone else gets, so it stays on screen:
		// hiding it would make the override look like the site's own answer.
		expect(fact.fact).toContain('35 %');
		expect(fact.override).toContain('55 %');
		expect(fact).toMatchObject({ pct: 55, source: 'override' });
	});

	it('has no override line without one in force', () => {
		expect(thresholdFact(en, 'en', options, 20, null).override).toBeNull();
		expect(thresholdFact(en, 'en', options, 20, 5).override).toBeNull();
	});
});
