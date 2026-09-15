/**
 * The planning half of the bitmap pipeline.
 *
 * Small functions, but the two failure modes they prevent are both
 * expensive: a prefetch order that fetches the wrong frames first stutters
 * exactly when the reviewer is looking at the interesting minute, and an
 * unbounded cache holds a few hundred megabytes of decoded RGBA — an event
 * carries about two dozen 432×496 frames and a reviewer goes through
 * hundreds of events in a sitting.
 */
import { describe, expect, it } from 'vitest';
import fixture from './fixture.json';
import {
	DEFAULT_CACHE_CAPACITY,
	frameUrl,
	lruEvict,
	prefetchPlan,
	presentStamps
} from './frames';
import { parseEvent } from './load';

const STAMPS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

describe('frameUrl', () => {
	it('addresses a frame by its content stamp', () => {
		expect(frameUrl('202606121340', 'overlay')).toBe(
			'/review-data/frames/202606121340.overlay.png'
		);
		expect(frameUrl('202606121340', 'observed')).toBe(
			'/review-data/frames/202606121340.observed.png'
		);
	});
});

describe('prefetchPlan', () => {
	it('starts with the frame on screen, then alternates forward and back', () => {
		expect(prefetchPlan(STAMPS, 5, 2)).toEqual(['5', '6', '4', '7', '3']);
	});

	it('fetches forward before backward — playback only goes one way', () => {
		const plan = prefetchPlan(STAMPS, 5, 3);
		expect(plan[0]).toBe('5');
		expect(plan[1]).toBe('6');
		expect(plan.indexOf('6')).toBeLessThan(plan.indexOf('4'));
	});

	it('skips past the ends instead of clamping onto them', () => {
		expect(prefetchPlan(STAMPS, 0, 2)).toEqual(['0', '1', '2']);
		expect(prefetchPlan(STAMPS, 9, 2)).toEqual(['9', '8', '7']);
		// No repeats: a clamped plan would ask for frame 0 three times.
		const plan = prefetchPlan(STAMPS, 1, 4);
		expect(new Set(plan).size).toBe(plan.length);
	});

	it('plans only the current frame at radius zero', () => {
		expect(prefetchPlan(STAMPS, 4, 0)).toEqual(['4']);
	});

	it('plans nothing for a cursor that is not on the track', () => {
		// Between events the index means nothing, and fetching frame 0 because
		// the index happened to be 0 is how a scrub lands on the wrong day.
		expect(prefetchPlan(STAMPS, -1)).toEqual([]);
		expect(prefetchPlan(STAMPS, 10)).toEqual([]);
		expect(prefetchPlan(STAMPS, 1.5)).toEqual([]);
		expect(prefetchPlan([], 0)).toEqual([]);
	});

	it('never plans more than the track holds', () => {
		expect(prefetchPlan(STAMPS, 4, 99)).toHaveLength(STAMPS.length);
	});
});

describe('lruEvict', () => {
	const cacheOf = (n: number) =>
		new Map(Array.from({ length: n }, (_, i) => [`k${i}`, i] as const));

	it('evicts nothing while there is room', () => {
		expect(lruEvict(cacheOf(3), 5)).toEqual([]);
		expect(lruEvict(cacheOf(5), 5)).toEqual([]);
		expect(lruEvict(new Map(), 5)).toEqual([]);
	});

	it('evicts the least recently inserted first', () => {
		expect(lruEvict(cacheOf(8), 5)).toEqual(['k0', 'k1', 'k2']);
	});

	it('follows re-insertion, which is how the store marks a hit', () => {
		const cache = cacheOf(6);
		// The store's touch protocol: delete, then set, moving the key to the
		// end of the insertion order.
		const touched = cache.get('k0')!;
		cache.delete('k0');
		cache.set('k0', touched);
		expect(lruEvict(cache, 4)).toEqual(['k1', 'k2']);
	});

	it('empties the cache when the capacity is zero or nonsense', () => {
		expect(lruEvict(cacheOf(3), 0)).toEqual(['k0', 'k1', 'k2']);
		expect(lruEvict(cacheOf(3), Number.NaN)).toEqual(['k0', 'k1', 'k2']);
	});

	it('defaults to a capacity that holds one event’s loop', () => {
		expect(lruEvict(cacheOf(DEFAULT_CACHE_CAPACITY))).toEqual([]);
		expect(lruEvict(cacheOf(DEFAULT_CACHE_CAPACITY + 1))).toHaveLength(1);
	});
});

describe('presentStamps', () => {
	it('drops the frames the builder said it could not write', () => {
		const event = parseEvent(
			JSON.parse(
				JSON.stringify((fixture.details as Record<string, unknown>)['fa-06104-20260421T0255Z'])
			)
		)!;
		const stamps = presentStamps(event.frames);
		expect(stamps).toHaveLength(event.frames.length - event.index.frames_missing);
		const missing = event.frames.filter((frame) => !frame.present).map((frame) => frame.stamp);
		expect(missing.length).toBeGreaterThan(0);
		for (const stamp of missing) expect(stamps).not.toContain(stamp);
	});

	it('still tries a frame whose presence the bundle did not state', () => {
		// An untried frame is a hole we made ourselves; a failed fetch costs
		// one picture the track already marks as uncertain.
		const frames = [
			{ stamp: 'a', present: true },
			{ stamp: 'b', present: null },
			{ stamp: 'c', present: false }
		];
		expect(presentStamps(frames)).toEqual(['a', 'b']);
	});
});
