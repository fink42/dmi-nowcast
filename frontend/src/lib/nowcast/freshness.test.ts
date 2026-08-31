/**
 * Freshness classification. The case that matters most is the first one: a
 * perfectly healthy pipeline sits at a radar age of ~28 min just before the
 * next fullRange composite lands, and calling that an outage is exactly the
 * bug this replaced.
 */
import { describe, expect, it } from 'vitest';
import {
	freshness,
	PIPELINE_STALE_AFTER_MIN,
	RADAR_AGE_WARN_MIN,
	type FreshnessState
} from './freshness';
import type { Manifest } from './manifest';

const NOW = Date.parse('2026-08-28T12:00:00Z');
const MIN = 60_000;

const iso = (minAgo: number) => new Date(NOW - minAgo * MIN).toISOString();

/** A manifest carrying only the fields freshness looks at. */
function manifest(radarMinAgo: number, generatedMinAgo: number | null | 'garbage'): Manifest {
	return {
		radar_ts_utc: iso(radarMinAgo),
		generated_at_utc:
			generatedMinAgo === 'garbage'
				? 'not-a-timestamp'
				: generatedMinAgo === null
					? (undefined as unknown as string)
					: iso(generatedMinAgo)
	} as Manifest;
}

const stateOf = (radarMinAgo: number, generatedMinAgo: number | null | 'garbage'): FreshnessState =>
	freshness(manifest(radarMinAgo, generatedMinAgo), NOW).state;

describe('freshness thresholds', () => {
	it('separates pipeline liveness from radar age', () => {
		// Two poll intervals plus margin vs. beyond the worst sawtooth peak.
		expect(PIPELINE_STALE_AFTER_MIN).toBe(15);
		expect(RADAR_AGE_WARN_MIN).toBe(35);
	});
});

describe('freshness', () => {
	it('reports no manifest as unknown ages and no alarm', () => {
		expect(freshness(null, NOW)).toEqual({
			radarAgeMin: null,
			pipelineAgeMin: null,
			state: 'ok'
		});
	});

	it('stays quiet at the healthy sawtooth peak', () => {
		// 10 min radar cadence + DMI publication delay + a 5 min sidecar poll:
		// 28 min of radar age with a cycle computed 2 min ago is normal.
		const f = freshness(manifest(28, 2), NOW);
		expect(f.state).toBe('ok');
		expect(f.radarAgeMin).toBeCloseTo(28, 6);
		expect(f.pipelineAgeMin).toBeCloseTo(2, 6);
	});

	it('keeps reporting radar age even when nothing is wrong', () => {
		expect(freshness(manifest(12, 3), NOW).radarAgeMin).toBeCloseTo(12, 6);
	});

	it('raises the pipeline alarm when no cycle has been computed for 20 min', () => {
		const f = freshness(manifest(28, 20), NOW);
		expect(f.state).toBe('pipeline-stale');
		expect(f.pipelineAgeMin).toBeCloseTo(20, 6);
	});

	it('warns about radar age only, when the pipeline is still computing', () => {
		const f = freshness(manifest(40, 2), NOW);
		expect(f.state).toBe('radar-old');
		expect(f.radarAgeMin).toBeCloseTo(40, 6);
	});

	it('lets pipeline liveness take precedence over radar age', () => {
		expect(stateOf(50, 20)).toBe('pipeline-stale');
	});

	it('treats the thresholds as strict, so a value sitting on them is fine', () => {
		expect(stateOf(RADAR_AGE_WARN_MIN, 2)).toBe('ok');
		expect(stateOf(5, PIPELINE_STALE_AFTER_MIN)).toBe('ok');
		expect(stateOf(RADAR_AGE_WARN_MIN + 0.5, 2)).toBe('radar-old');
		expect(stateOf(5, PIPELINE_STALE_AFTER_MIN + 0.5)).toBe('pipeline-stale');
	});

	describe('manifests without a usable generated_at_utc', () => {
		for (const [name, generated] of [
			['missing', null],
			['unparseable', 'garbage']
		] as const) {
			it(`falls back to radar age alone when it is ${name}`, () => {
				expect(freshness(manifest(28, generated), NOW).pipelineAgeMin).toBeNull();
				// No false alarm: an unknown pipeline age is not an outage …
				expect(stateOf(28, generated)).toBe('ok');
				// … but radar age still speaks for itself.
				expect(stateOf(40, generated)).toBe('radar-old');
			});
		}
	});

	it('never reports a negative age when the server clock runs ahead', () => {
		const f = freshness(manifest(-3, -1), NOW);
		expect(f.radarAgeMin).toBe(0);
		expect(f.pipelineAgeMin).toBe(0);
		expect(f.state).toBe('ok');
	});

	it('reports an unparseable radar timestamp as unknown rather than NaN', () => {
		const f = freshness({ radar_ts_utc: 'nope', generated_at_utc: iso(2) } as Manifest, NOW);
		expect(f.radarAgeMin).toBeNull();
		expect(f.state).toBe('ok');
	});

	it('defaults `now` to the wall clock', () => {
		const fresh = {
			radar_ts_utc: new Date().toISOString(),
			generated_at_utc: new Date().toISOString()
		} as Manifest;
		expect(freshness(fresh).state).toBe('ok');
	});
});
