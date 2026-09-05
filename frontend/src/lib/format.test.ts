/**
 * The live ETA countdown, the rain field read at wall-clock now, and the
 * headline that follows from both.
 *
 * Two bugs are pinned here, and the second is the reason the first half of
 * this file is not enough:
 *
 *  1. The sidecar's point ETA is minutes from the instant the cycle was
 *     computed, cycles land 5–10 min apart, and the panel printed that number
 *     unchanged — "rain in 12 min" still said 12 nine minutes later.
 *  2. At 09:43 the panel said "It is raining here now" while the loop's frame
 *     for 09:43 showed the point dry. The newest composite was 23 min old and
 *     won the headline outright, and the cycle's ETA product — a cumulative
 *     first arrival — had stuck at 0 since the cell passed over. The fix reads
 *     the headline from the same advected field the loop draws, at the same
 *     instant, so the sentence and the picture cannot disagree.
 */
import { describe, expect, it } from 'vitest';
import {
	countdownEtaMin,
	headline,
	headlineDecision,
	nextWetMinutes,
	rainNowMmH,
	RAINING_NOW_MIN,
	RAINING_NOW_MM_H
} from './format';
import type { RainSample } from './nowcast/sampler';
import { da } from './i18n/da';
import { en } from './i18n/en';

const GENERATED = '2026-08-28T12:00:00Z';
const AT = Date.parse(GENERATED);

/** The wall clock this many minutes after the cycle was computed. */
const after = (minutes: number): number => AT + minutes * 60_000;

/** One step of the advected field: lead in minutes from `GENERATED`, mm/h. */
const step = (leadMin: number, mmH: number | null): RainSample => ({
	leadMin,
	validTsUtc: new Date(after(leadMin)).toISOString(),
	mmH
});

/** The overlay leads the sidecar serves the field on. */
const series = (...mmH: (number | null)[]): RainSample[] =>
	mmH.map((value, i) => step(i * 10, value));

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


describe('rainNowMmH', () => {
	it('interpolates linearly between the two entries bracketing now', () => {
		// 0 mm/h at +0, 2 mm/h at +10: halfway through, 1 mm/h.
		const field = series(0, 2);
		expect(rainNowMmH(field, after(5))).toBeCloseTo(1, 12);
		expect(rainNowMmH(field, after(2.5))).toBeCloseTo(0.5, 12);
		// And exactly on a step it is that step's own value.
		expect(rainNowMmH(field, after(0))).toBe(0);
		expect(rainNowMmH(field, after(10))).toBe(2);
	});

	it('walks the right bracket of a longer series', () => {
		const field = series(0, 0, 4, 0);
		// Between +10 (0) and +20 (4), a quarter of the way along.
		expect(rainNowMmH(field, after(12.5))).toBeCloseTo(1, 12);
		// And between +20 (4) and +30 (0), on the way back down.
		expect(rainNowMmH(field, after(25))).toBeCloseTo(2, 12);
	});

	it('holds the nearest value before the first entry and after the last', () => {
		const field = series(1.5, 3);
		expect(rainNowMmH(field, after(-30))).toBe(1.5);
		expect(rainNowMmH(field, after(90))).toBe(3);
	});

	it('propagates a nodata end of the bracket instead of inventing a number', () => {
		// Null is "we do not know", and there is nothing between that and 2.
		expect(rainNowMmH(series(null, 2), after(5))).toBeNull();
		expect(rainNowMmH(series(2, null), after(5))).toBeNull();
		expect(rainNowMmH(series(null, null), after(5))).toBeNull();
		// Outside the bracket the clamped value carries its own null.
		expect(rainNowMmH(series(null, 2), after(-5))).toBeNull();
		expect(rainNowMmH(series(null, 2), after(50))).toBe(2);
	});

	it('is null for an empty series — no field is not a dry field', () => {
		expect(rainNowMmH([], after(5))).toBeNull();
	});

	it('is null on an unparsable stamp, and never throws', () => {
		const broken = [step(0, 0), { leadMin: 10, validTsUtc: 'not-a-timestamp', mmH: 4 }];
		expect(() => rainNowMmH(broken, after(5))).not.toThrow();
		expect(rainNowMmH(broken, after(5))).toBeNull();
		expect(rainNowMmH(series(0, 2), Number.NaN)).toBeNull();
	});

	it('reads a single-entry series as that entry, everywhere', () => {
		const one = [step(0, 1.2)];
		expect(rainNowMmH(one, after(-5))).toBe(1.2);
		expect(rainNowMmH(one, after(0))).toBe(1.2);
		expect(rainNowMmH(one, after(40))).toBe(1.2);
	});
});

describe('nextWetMinutes', () => {
	it('counts to the first wet entry strictly after now', () => {
		// Wet at +30 only; the clock is at +12.
		expect(nextWetMinutes(series(0, 0, 0, 4), after(12))).toBeCloseTo(18, 12);
	});

	it('skips entries at or before now, however wet they were', () => {
		// The shower was overhead at +0 and +10 and has gone; nothing ahead.
		expect(nextWetMinutes(series(6, 6, 0, 0), after(10))).toBeNull();
		// One later cell does count.
		expect(nextWetMinutes(series(6, 6, 0, 3), after(10))).toBeCloseTo(20, 12);
	});

	it('picks the first wet entry, not the wettest', () => {
		expect(nextWetMinutes(series(0, 1, 9), after(0))).toBeCloseTo(10, 12);
	});

	it('is null when nothing reaches the threshold, and treats nulls as unknown', () => {
		expect(nextWetMinutes(series(0, 0.2, 0.4), after(0))).toBeNull();
		expect(nextWetMinutes(series(0, null, null), after(0))).toBeNull();
		expect(nextWetMinutes([], after(0))).toBeNull();
		// Exactly at the threshold counts — the same boundary as raining-now.
		expect(nextWetMinutes(series(0, RAINING_NOW_MM_H), after(0))).toBeCloseTo(10, 12);
	});

	it('honours an explicit threshold and survives a bad stamp', () => {
		expect(nextWetMinutes(series(0, 1, 5), after(0), 4)).toBeCloseTo(20, 12);
		const broken = [step(0, 0), { leadMin: 10, validTsUtc: '', mmH: 9 }];
		expect(nextWetMinutes(broken, after(0))).toBeNull();
		expect(nextWetMinutes(series(0, 4), Number.NaN)).toBeNull();
	});
});

describe('headlineDecision', () => {
	/**
	 * The incident, in numbers. The cell was over the point when the cycle was
	 * computed, so the cumulative first-arrival ETA reads 0 and stays there;
	 * the observation grid (23 min old) shows rain; and the field for the
	 * minute the viewer is actually looking at — the frame the loop is drawing
	 * — is dry. The panel said "It is raining here now". It must not.
	 */
	it('does not claim rain now when the field for now is dry, whatever the ETA says', () => {
		const passed = series(3, 0.1, 0, 0);
		const decision = headlineDecision(0, passed, after(12));
		expect(decision.kind).toBe('no-rain');
		expect(decision.etaMin).toBeNull();
	});

	it('answers the sticky ETA with the field’s own next wet step', () => {
		// Same stuck ETA, but a second cell arrives at +30 (18 min from now).
		const decision = headlineDecision(0, series(3, 0.1, 0, 4), after(12));
		expect(decision.kind).toBe('eta');
		expect(decision.etaMin).toBeCloseTo(18, 12);
	});

	it('says it is raining when the field says so, whatever the ETA', () => {
		// A 40 min ETA (the next cell behind this one) does not get to speak.
		expect(headlineDecision(40, series(2, 2, 0), after(5)).kind).toBe('raining-now');
		expect(headlineDecision(null, series(2, 2, 0), after(5)).kind).toBe('raining-now');
		// Exactly at the threshold is raining; a hair under it is not.
		expect(headlineDecision(null, series(RAINING_NOW_MM_H, RAINING_NOW_MM_H), after(5)).kind).toBe(
			'raining-now'
		);
		expect(
			headlineDecision(null, series(RAINING_NOW_MM_H - 0.01, RAINING_NOW_MM_H - 0.01), after(5))
				.kind
		).toBe('no-rain');
	});

	it('leads with the ensemble countdown when the field for now is dry', () => {
		const decision = headlineDecision(12, series(0, 0, 3), after(0));
		expect(decision.kind).toBe('eta');
		expect(decision.etaMin).toBe(12);
	});

	it('says no rain when there is no ETA and the field is dry', () => {
		const decision = headlineDecision(null, series(0, 0, 0), after(5));
		expect(decision.kind).toBe('no-rain');
		expect(decision.etaMin).toBeNull();
	});

	it('crosses into raining-now as the clock moves through the field', () => {
		// Dry now, 3 mm/h at +20: the same cycle, watched for twenty minutes.
		const field = series(0, 0, 3);
		const at = (minutes: number) => headlineDecision(20, field, after(minutes)).kind;
		expect(at(0)).toBe('eta');
		expect(at(10)).toBe('eta');
		// Interpolation crosses 0.5 mm/h a third of the way from +10 to +20.
		expect(at(14)).toBe('raining-now');
		expect(at(20)).toBe('raining-now');
	});

	describe('with no field at all', () => {
		/**
		 * An older sidecar, or a download that failed. The series is empty,
		 * which is not evidence of a dry sky — so the decision falls back to
		 * the ETA rule exactly as it read before any of this existed.
		 */
		it('falls back to the old ETA rule', () => {
			expect(headlineDecision(null, [], after(5))).toEqual({ kind: 'no-rain', etaMin: null });
			expect(headlineDecision(0, [], after(5))).toEqual({ kind: 'raining-now', etaMin: null });
			expect(headlineDecision(RAINING_NOW_MIN, [], after(5))).toEqual({
				kind: 'raining-now',
				etaMin: null
			});
			expect(headlineDecision(RAINING_NOW_MIN + 0.01, [], after(5))).toEqual({
				kind: 'eta',
				etaMin: RAINING_NOW_MIN + 0.01
			});
			expect(headlineDecision(12, [], after(5))).toEqual({ kind: 'eta', etaMin: 12 });
		});

		it('still crosses into raining-now as the cycle ages', () => {
			// One six-minute ETA, computed once, watched over five minutes.
			const at = (minutes: number) =>
				headlineDecision(countdownEtaMin(6, GENERATED, after(minutes)), [], after(minutes)).kind;
			expect(at(0)).toBe('eta');
			expect(at(4)).toBe('eta'); // 2 min out
			expect(at(4.5)).toBe('raining-now'); // 1.5 min out — the boundary
			expect(at(30)).toBe('raining-now');
		});
	});

	it('never lets an unreadable field assert anything on its own', () => {
		// Every step nodata: the field says nothing, so the ensemble answers.
		const blind = series(null, null, null);
		expect(headlineDecision(12, blind, after(0))).toEqual({ kind: 'eta', etaMin: 12 });
		expect(headlineDecision(null, blind, after(0)).kind).toBe('no-rain');
	});
});

describe('headline', () => {
	it('says the counted-down minutes, not the ones the cycle computed', () => {
		const etaNow = countdownEtaMin(12, GENERATED, after(7));
		expect(headline(da, headlineDecision(etaNow, [], after(7)))).toBe('Regn om ca. 5 min');
		expect(headline(en, headlineDecision(etaNow, [], after(7)))).toBe('Rain in about 5 min');
		// What the panel used to print seven minutes into the cycle.
		expect(headline(da, headlineDecision(12, [], after(7)))).toBe('Regn om ca. 12 min');
	});

	it('switches to the raining-now and no-rain sentences', () => {
		const raining = headlineDecision(countdownEtaMin(6, GENERATED, after(5)), [], after(5));
		expect(headline(da, raining)).toBe(da.panel.headlineRainingNow);
		expect(headline(en, raining)).toBe(en.panel.headlineRainingNow);
		const dry = headlineDecision(null, [], after(5));
		expect(headline(da, dry)).toBe(da.panel.headlineNoRain);
		expect(headline(en, dry)).toBe(en.panel.headlineNoRain);
	});

	it('leads with the field, not with the ensemble’s stale first arrival', () => {
		// The incident sentence, in both languages: dry now, ETA stuck at 0.
		const passed = headlineDecision(0, series(3, 0.1, 0, 0), after(12));
		expect(headline(da, passed)).toBe(da.panel.headlineNoRain);
		expect(headline(en, passed)).toBe(en.panel.headlineNoRain);
		// Wet now: the sentence the panel was right to want, for a real reason.
		const wet = headlineDecision(40, series(2, 2, 0), after(5));
		expect(headline(da, wet)).toBe(da.panel.headlineRainingNow);
		expect(headline(en, wet)).toBe(en.panel.headlineRainingNow);
	});

	it('rounds the arrival the decision carries', () => {
		// 18 min to the next wet step, from the sticky-ETA branch.
		expect(headline(en, headlineDecision(0, series(3, 0.1, 0, 4), after(12)))).toBe(
			'Rain in about 18 min'
		);
	});
});
