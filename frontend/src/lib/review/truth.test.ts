/**
 * The truth side. One property dominates this file: **an unreported slot is
 * never reported as dry.** It is checked against every slot the fixture
 * carries, at several instants inside each one, because it is the mistake
 * that would quietly turn a gauge outage into evidence for a false alarm —
 * and the reviewer, looking at a blank strip, would have no way to tell.
 *
 * The same rule governs the dual-truth badge: a null verdict is "not
 * known", and must never fall through to `both_dry`, which is the claim
 * that the forecast invented rain.
 */
import { describe, expect, it } from 'vitest';
import fixture from './fixture.json';
import { parseEvent } from './load';
import type { DualTruthClass, EventDetail, RadarDiscBlock, Slot } from './schema';
import {
	dualTruthLabel,
	gaugeStateAt,
	neighbourStatesAt,
	radarStateAt,
	slotAt,
	wetRunSegments
} from './truth';

const at = (iso: string) => Date.parse(iso);
const details = fixture.details as Record<string, any>;
const eventIds = Object.keys(details);
const parsed = (eventId: string): EventDetail =>
	parseEvent(JSON.parse(JSON.stringify(details[eventId])))!;

const slot = (end: string, known: boolean, wet: boolean, mm: number | null = null): Slot => ({
	slot_end_utc: end,
	mm,
	known,
	wet
});

describe('slotAt', () => {
	const slots = [
		slot('2026-06-12T13:00:00Z', true, false, 0),
		slot('2026-06-12T13:10:00Z', true, true, 0.4),
		slot('2026-06-12T13:20:00Z', false, false)
	];

	it('covers (end - slot, end]: the end instant belongs to its own slot', () => {
		expect(slotAt(slots, at('2026-06-12T13:10:00Z')).slotEndUtc).toBe('2026-06-12T13:10:00Z');
		expect(slotAt(slots, at('2026-06-12T13:10:00.001Z')).slotEndUtc).toBe(
			'2026-06-12T13:20:00Z'
		);
		expect(slotAt(slots, at('2026-06-12T13:00:00.001Z')).slotEndUtc).toBe(
			'2026-06-12T13:10:00Z'
		);
	});

	it('reads a wet slot as wet and a reported dry slot as dry', () => {
		expect(slotAt(slots, at('2026-06-12T13:05:00Z')).state).toBe('wet');
		expect(slotAt(slots, at('2026-06-12T12:55:00Z')).state).toBe('dry');
	});

	it('reads an unreported slot as unknown, never as dry', () => {
		const state = slotAt(slots, at('2026-06-12T13:15:00Z'));
		expect(state.state).toBe('unknown');
		expect(state.state === 'unknown' && state.reason).toBe('not_reported');
	});

	it('says which side of the record it fell off', () => {
		const before = slotAt(slots, at('2026-06-12T11:00:00Z'));
		const after = slotAt(slots, at('2026-06-12T15:00:00Z'));
		expect(before.state === 'unknown' && before.reason).toBe('before_series');
		expect(after.state === 'unknown' && after.reason).toBe('after_series');
	});

	it('calls a hole in the middle of a series unknown, not dry', () => {
		const holed = [slots[0], slot('2026-06-12T13:40:00Z', true, false, 0)];
		const state = slotAt(holed, at('2026-06-12T13:15:00Z'));
		expect(state.state).toBe('unknown');
		expect(state.state === 'unknown' && state.reason).toBe('not_reported');
	});

	it('has nothing to say about an empty series', () => {
		const state = slotAt([], at('2026-06-12T13:00:00Z'));
		expect(state.state === 'unknown' && state.reason).toBe('no_series');
	});
});

describe('an unreported slot is never dry — over the whole fixture', () => {
	it('answers unknown at every instant inside every known:false slot', () => {
		let checked = 0;
		for (const eventId of eventIds) {
			const event = parsed(eventId);
			const series: Array<{ slots: readonly Slot[]; slotMin: number }> = [
				{ slots: event.gauge?.slots ?? [], slotMin: event.gauge?.slot_min ?? 10 },
				{ slots: event.radar_disc?.slots ?? [], slotMin: 10 },
				...(event.neighbours?.stations ?? []).map((station) => ({
					slots: station.slots,
					slotMin: 10
				}))
			];
			for (const { slots, slotMin } of series) {
				for (const entry of slots) {
					if (entry.known) continue;
					const end = at(entry.slot_end_utc);
					for (const offset of [1, slotMin * 30_000, slotMin * 60_000]) {
						const state = slotAt(slots, end - slotMin * 60_000 + offset, slotMin);
						expect(state.state, `${eventId} ${entry.slot_end_utc}`).not.toBe('dry');
						expect(state.state).toBe('unknown');
						checked += 1;
					}
				}
			}
		}
		// The fixture must actually contain some silence, or this proves nothing.
		expect(checked).toBeGreaterThan(0);
	});
});

describe('gaugeStateAt', () => {
	const event = parsed('fa-06104-20260421T0255Z');

	it('is unknown past the pinned gauge horizon, whatever the slots say', () => {
		const knownUntil = at(event.gauge!.known_until_utc!);
		const state = gaugeStateAt(event.gauge, knownUntil + 60_000);
		expect(state.state).toBe('unknown');
		expect(state.state === 'unknown' && state.reason).toBe('beyond_known_until');
	});

	it('still reads the slots before the horizon', () => {
		// The last slot the gauge actually reported, read at its own end
		// instant. Derived rather than guessed: the horizon falls inside a
		// slot, and that slot is already unknown.
		const lastReported = [...event.gauge!.slots].reverse().find((slot) => slot.known)!;
		expect(gaugeStateAt(event.gauge, at(lastReported.slot_end_utc)).state).toBe('dry');
	});

	it('has nothing to say without a gauge block', () => {
		expect(gaugeStateAt(null, at('2026-04-21T02:00:00Z')).state).toBe('unknown');
	});
});

describe('radarStateAt', () => {
	const event = parsed('fa-06126-20260612T1345Z');

	it('reads the disc’s slot series when the bundle carries one', () => {
		const firstWet = event.radar_disc!.first_wet_utc!;
		expect(radarStateAt(event.radar_disc, at(firstWet)).state).toBe('wet');
	});

	it('falls back to the p90 series, applying the bundle’s own threshold', () => {
		const disc: RadarDiscBlock = {
			disc_radius_m: 1000,
			statistic: 'p90',
			threshold_mm_h: 0.5,
			series: [
				{
					radar_ts_utc: '2026-06-12T13:00:00Z',
					p90_mm_h: 0.4,
					max_mm_h: 1,
					mean_mm_h: 0.2,
					n_pixels: 5,
					n_valid: 5
				},
				{
					radar_ts_utc: '2026-06-12T13:10:00Z',
					p90_mm_h: 0.9,
					max_mm_h: 2,
					mean_mm_h: 0.5,
					n_pixels: 5,
					n_valid: 5
				},
				{
					radar_ts_utc: '2026-06-12T13:20:00Z',
					p90_mm_h: null,
					max_mm_h: null,
					mean_mm_h: null,
					n_pixels: null,
					n_valid: null
				}
			],
			slots: [],
			wet_in_window: true,
			first_wet_utc: '2026-06-12T13:10:00Z'
		};
		expect(radarStateAt(disc, at('2026-06-12T13:05:00Z')).state).toBe('dry');
		expect(radarStateAt(disc, at('2026-06-12T13:15:00Z')).state).toBe('wet');
	});

	it('reads a null p90 as no value, never as zero', () => {
		const disc: RadarDiscBlock = {
			disc_radius_m: 1000,
			statistic: 'p90',
			threshold_mm_h: 0.5,
			series: [
				{
					radar_ts_utc: '2026-06-12T13:20:00Z',
					p90_mm_h: null,
					max_mm_h: null,
					mean_mm_h: null,
					n_pixels: null,
					n_valid: null
				}
			],
			slots: [],
			wet_in_window: null,
			first_wet_utc: null
		};
		const state = radarStateAt(disc, at('2026-06-12T13:25:00Z'));
		expect(state.state).toBe('unknown');
		expect(state.state === 'unknown' && state.reason).toBe('no_value');
	});

	it('does not carry a stale composite forward over a gap', () => {
		const disc: RadarDiscBlock = {
			disc_radius_m: 1000,
			statistic: 'p90',
			threshold_mm_h: 0.5,
			series: [
				{
					radar_ts_utc: '2026-06-12T13:00:00Z',
					p90_mm_h: 4,
					max_mm_h: 6,
					mean_mm_h: 2,
					n_pixels: 5,
					n_valid: 5
				}
			],
			slots: [],
			wet_in_window: true,
			first_wet_utc: '2026-06-12T13:00:00Z'
		};
		// Forty minutes later that image is not a reading for now.
		expect(radarStateAt(disc, at('2026-06-12T13:40:00Z')).state).toBe('unknown');
	});
});

describe('neighbourStatesAt', () => {
	const event = parsed('fa-06126-20260612T1345Z');

	it('answers nearest first', () => {
		const states = neighbourStatesAt(event.neighbours, at('2026-06-12T13:45:00Z'));
		const distances = states.map((state) => state.station.distance_km);
		expect(distances).toEqual([...distances].sort((a, b) => a - b));
	});

	it('reads a neighbour that never reported as unknown', () => {
		const silent = parsed('fa-06104-20260421T0255Z');
		const states = neighbourStatesAt(silent.neighbours, at('2026-04-21T02:55:00Z'));
		expect(states.length).toBeGreaterThan(0);
		for (const state of states) {
			expect(state.state.state).toBe('unknown');
			expect(state.wetInWindow).toBeNull();
		}
	});

	it('has nothing to say without a neighbours block', () => {
		expect(neighbourStatesAt(null, at('2026-06-12T13:45:00Z'))).toEqual([]);
	});
});

describe('dualTruthLabel', () => {
	const classes: DualTruthClass[] = [
		'both_wet',
		'radar_wet_gauge_dry',
		'gauge_wet_radar_dry',
		'both_dry'
	];

	it('names every quadrant and carries the same-instrument caveat with each', () => {
		for (const cls of classes) {
			const label = dualTruthLabel(cls);
			expect(label.code).toBe(cls);
			expect(label.label.length).toBeGreaterThan(0);
			expect(label.reading.length).toBeGreaterThan(0);
			expect(label.caveat).toMatch(/consistency check/);
			expect(label.caveat).toMatch(/same instrument|instrument the forecast was made from/);
		}
	});

	it('gives a null verdict its own words and never falls through to both_dry', () => {
		const label = dualTruthLabel(null);
		expect(label.code).toBeNull();
		expect(label.label).not.toBe(dualTruthLabel('both_dry').label);
		expect(label.reading).not.toMatch(/invented/);
		expect(label.reading).toMatch(/not evidence that the sky was dry/);
	});

	it('matches the verdict every fixture event actually carries', () => {
		for (const eventId of eventIds) {
			const event = parsed(eventId);
			const label = dualTruthLabel(event.dual_truth?.class ?? null);
			expect(label.code).toBe(event.dual_truth?.class ?? null);
		}
	});
});

describe('wetRunSegments', () => {
	it('collapses a series into wet, dry and unknown runs', () => {
		const runs = wetRunSegments([
			slot('2026-06-12T13:00:00Z', true, false, 0),
			slot('2026-06-12T13:10:00Z', true, false, 0),
			slot('2026-06-12T13:20:00Z', true, true, 0.4),
			slot('2026-06-12T13:30:00Z', true, true, 0.6),
			slot('2026-06-12T13:40:00Z', false, false)
		]);
		expect(runs.map((run) => run.state)).toEqual(['dry', 'wet', 'unknown']);
		expect(runs[0].slots).toBe(2);
		expect(runs[1].mm).toBeCloseTo(1.0, 6);
		// An unknown run carries no depth: there is no number to add up.
		expect(runs[2].mm).toBeNull();
	});

	it('never merges an unknown run into the dry runs on either side', () => {
		const runs = wetRunSegments([
			slot('2026-06-12T13:00:00Z', true, false, 0),
			slot('2026-06-12T13:10:00Z', false, false),
			slot('2026-06-12T13:20:00Z', true, false, 0)
		]);
		expect(runs.map((run) => run.state)).toEqual(['dry', 'unknown', 'dry']);
	});

	it('splits a run where the series has a hole', () => {
		const runs = wetRunSegments([
			slot('2026-06-12T13:00:00Z', true, false, 0),
			slot('2026-06-12T14:00:00Z', true, false, 0)
		]);
		expect(runs).toHaveLength(2);
	});

	it('spans slot widths, so the strip lines up with the track', () => {
		const runs = wetRunSegments([slot('2026-06-12T13:10:00Z', true, true, 0.4)], 10);
		expect(runs[0].fromMs).toBe(at('2026-06-12T13:00:00Z'));
		expect(runs[0].toMs).toBe(at('2026-06-12T13:10:00Z'));
	});

	it('orders a series that arrived shuffled', () => {
		const runs = wetRunSegments([
			slot('2026-06-12T13:20:00Z', true, true, 0.4),
			slot('2026-06-12T13:00:00Z', true, false, 0),
			slot('2026-06-12T13:10:00Z', true, false, 0)
		]);
		expect(runs.map((run) => run.state)).toEqual(['dry', 'wet']);
	});
});
