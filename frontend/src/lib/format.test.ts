/**
 * The live ETA countdown, and the headline that follows it.
 *
 * The bug behind this: the sidecar's point ETA is minutes from the instant the
 * cycle was computed, cycles land 5–10 min apart, and the panel printed that
 * number unchanged — so "rain in 12 min" could still say 12 nine minutes later,
 * and "it is raining here now" arrived a whole cycle late.
 */
import { describe, expect, it } from 'vitest';
import { countdownEtaMin, headline, headlineKind, RAINING_NOW_MIN } from './format';
import { da } from './i18n/da';
import { en } from './i18n/en';

const GENERATED = '2026-08-28T12:00:00Z';
const AT = Date.parse(GENERATED);

/** The wall clock this many minutes after the cycle was computed. */
const after = (minutes: number): number => AT + minutes * 60_000;

describe('countdownEtaMin', () => {
	it('counts the cycle ETA down by the cycle age', () => {
		expect(countdownEtaMin(12, GENERATED, after(0))).toBe(12);
		expect(countdownEtaMin(12, GENERATED, after(7))).toBe(5);
	});

	it('counts down continuously, not in whole minutes', () => {
		expect(countdownEtaMin(12, GENERATED, after(1.5))).toBeCloseTo(10.5, 10);
	});

	it('floors at zero instead of counting into the past', () => {
		// Rain that was due eight minutes ago is "now", never a negative ETA.
		expect(countdownEtaMin(12, GENERATED, after(20))).toBe(0);
		expect(countdownEtaMin(0, GENERATED, after(5))).toBe(0);
	});

	it('keeps a null ETA null — no rain within the horizon stays that way', () => {
		expect(countdownEtaMin(null, GENERATED, after(7))).toBeNull();
		expect(countdownEtaMin(null, undefined, after(7))).toBeNull();
	});

	it('returns the ETA unchanged when the cycle has no usable timestamp', () => {
		// An older sidecar, a garbled stamp: no countdown beats a wrong one.
		expect(countdownEtaMin(12, undefined, after(7))).toBe(12);
		expect(countdownEtaMin(12, '', after(7))).toBe(12);
		expect(countdownEtaMin(12, '   ', after(7))).toBe(12);
		expect(countdownEtaMin(12, 'not-a-timestamp', after(7))).toBe(12);
	});

	it('never throws on absurd input', () => {
		expect(() => countdownEtaMin(12, 'not-a-timestamp', Number.NaN)).not.toThrow();
		expect(countdownEtaMin(12, GENERATED, Number.NaN)).toBe(12);
	});

	it('treats a cycle stamped in the future as brand new, not as a longer wait', () => {
		// Viewer and sidecar clocks disagree; that must not inflate an ETA.
		expect(countdownEtaMin(12, GENERATED, after(-3))).toBe(12);
	});
});

describe('headlineKind', () => {
	it('reads the ETA it is given, not a forecast field', () => {
		expect(headlineKind(null)).toBe('no-rain');
		expect(headlineKind(12)).toBe('eta');
		expect(headlineKind(RAINING_NOW_MIN)).toBe('raining-now');
		expect(headlineKind(RAINING_NOW_MIN + 0.01)).toBe('eta');
		expect(headlineKind(0)).toBe('raining-now');
	});

	it('crosses into "raining now" as the cycle ages, without a new cycle', () => {
		// One six-minute ETA, computed once, watched over five minutes.
		const at = (minutes: number) => headlineKind(countdownEtaMin(6, GENERATED, after(minutes)));
		expect(at(0)).toBe('eta');
		expect(at(4)).toBe('eta'); // 2 min out
		expect(at(4.5)).toBe('raining-now'); // 1.5 min out — the boundary
		expect(at(5)).toBe('raining-now');
		expect(at(30)).toBe('raining-now');
	});
});

describe('headline', () => {
	it('says the counted-down minutes, not the ones the cycle computed', () => {
		const etaNow = countdownEtaMin(12, GENERATED, after(7));
		expect(headline(da, etaNow)).toBe('Regn om ca. 5 min');
		expect(headline(en, etaNow)).toBe('Rain in about 5 min');
		// What the panel used to print seven minutes into the cycle.
		expect(headline(da, 12)).toBe('Regn om ca. 12 min');
	});

	it('switches to the raining-now and no-rain sentences', () => {
		expect(headline(da, countdownEtaMin(6, GENERATED, after(5)))).toBe(da.panel.headlineRainingNow);
		expect(headline(en, countdownEtaMin(6, GENERATED, after(5)))).toBe(en.panel.headlineRainingNow);
		expect(headline(da, null)).toBe(da.panel.headlineNoRain);
		expect(headline(en, null)).toBe(en.panel.headlineNoRain);
	});
});
