/**
 * The live ETA countdown, and the headline that follows it.
 *
 * The bug behind this: the sidecar's point ETA is minutes from the instant the
 * cycle was computed, cycles land 5–10 min apart, and the panel printed that
 * number unchanged — so "rain in 12 min" could still say 12 nine minutes later,
 * and "it is raining here now" arrived a whole cycle late.
 */
import { describe, expect, it } from 'vitest';
import {
	countdownEtaMin,
	headline,
	headlineKind,
	RAINING_NOW_MIN,
	RAINING_NOW_MM_H
} from './format';
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
		expect(headlineKind(null, null)).toBe('no-rain');
		expect(headlineKind(12, null)).toBe('eta');
		expect(headlineKind(RAINING_NOW_MIN, null)).toBe('raining-now');
		expect(headlineKind(RAINING_NOW_MIN + 0.01, null)).toBe('eta');
		expect(headlineKind(0, null)).toBe('raining-now');
	});

	it('crosses into "raining now" as the cycle ages, without a new cycle', () => {
		// One six-minute ETA, computed once, watched over five minutes.
		const at = (minutes: number) =>
			headlineKind(countdownEtaMin(6, GENERATED, after(minutes)), null);
		expect(at(0)).toBe('eta');
		expect(at(4)).toBe('eta'); // 2 min out
		expect(at(4.5)).toBe('raining-now'); // 1.5 min out — the boundary
		expect(at(5)).toBe('raining-now');
		expect(at(30)).toBe('raining-now');
	});

	/**
	 * The bug this half exists for: the ETA product answers "when does the
	 * next shower reach this pixel", which on the trailing edge of rain is a
	 * later cell 16 min out — while the radar shows rain over the point now.
	 */
	describe('with an observation', () => {
		const OBSERVED = [
			['dry', 0],
			['a trace below the threshold', RAINING_NOW_MM_H - 0.01],
			['exactly the threshold', RAINING_NOW_MM_H],
			['a downpour', 12]
		] as const;
		const ETAS = [
			['no rain within the horizon', null],
			['imminent', 0.5],
			['a quarter of an hour out', 16]
		] as const;

		it('lets a measurement at or above the threshold outrank any ETA', () => {
			for (const [etaName, eta] of ETAS) {
				for (const [obsName, mmH] of OBSERVED) {
					const raining = mmH >= RAINING_NOW_MM_H;
					const expected = raining
						? 'raining-now'
						: eta === null
							? 'no-rain'
							: eta <= RAINING_NOW_MIN
								? 'raining-now'
								: 'eta';
					expect(headlineKind(eta, mmH), `${obsName} / ${etaName}`).toBe(expected);
				}
			}
			// Spelled out, because this row is the whole point of the feature.
			expect(headlineKind(16, 1.2)).toBe('raining-now');
			expect(headlineKind(16, 0.4)).toBe('eta');
			expect(headlineKind(null, 1.2)).toBe('raining-now');
		});

		it('treats a missing observation as unknown, never as dry', () => {
			// No observation grid this cycle, a nodata pixel, or the server
			// path: all null, and all fall back to the ETA rule untouched.
			expect(headlineKind(16, null)).toBe('eta');
			expect(headlineKind(null, null)).toBe('no-rain');
			expect(headlineKind(0.5, null)).toBe('raining-now');
		});
	});
});

describe('headline', () => {
	it('says the counted-down minutes, not the ones the cycle computed', () => {
		const etaNow = countdownEtaMin(12, GENERATED, after(7));
		expect(headline(da, etaNow, null)).toBe('Regn om ca. 5 min');
		expect(headline(en, etaNow, null)).toBe('Rain in about 5 min');
		// What the panel used to print seven minutes into the cycle.
		expect(headline(da, 12, null)).toBe('Regn om ca. 12 min');
	});

	it('switches to the raining-now and no-rain sentences', () => {
		expect(headline(da, countdownEtaMin(6, GENERATED, after(5)), null)).toBe(
			da.panel.headlineRainingNow
		);
		expect(headline(en, countdownEtaMin(6, GENERATED, after(5)), null)).toBe(
			en.panel.headlineRainingNow
		);
		expect(headline(da, null, null)).toBe(da.panel.headlineNoRain);
		expect(headline(en, null, null)).toBe(en.panel.headlineNoRain);
	});

	it('says it is raining here now when the radar says so, whatever the ETA', () => {
		expect(headline(da, 16, 1.2)).toBe(da.panel.headlineRainingNow);
		expect(headline(en, 16, 1.2)).toBe(en.panel.headlineRainingNow);
		// Without the observation the same cycle still promises rain in 16 min.
		expect(headline(en, 16, null)).toBe('Rain in about 16 min');
		expect(headline(en, 16, 0.2)).toBe('Rain in about 16 min');
		// And an observation cannot invent rain out of a dry no-rain cycle.
		expect(headline(en, null, 0.2)).toBe(en.panel.headlineNoRain);
	});
});
