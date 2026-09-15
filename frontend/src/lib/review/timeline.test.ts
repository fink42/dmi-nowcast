/**
 * The review track.
 *
 * The property that matters is that the track is a chart of TIME, not of
 * frame count: a 40-minute coverage gap has to be four times the width of a
 * 10-minute step, because the hole is the evidence a reviewer is being
 * asked to judge. An index-spaced track — which is the right answer for the
 * public loop and is imported unchanged for playback — would draw that hole
 * as a straight line and quietly delete it.
 *
 * The width test is built inline rather than from the fixture, because it
 * is arithmetic and must stay true whatever a regenerated fixture contains.
 */
import { describe, expect, it } from 'vitest';
import fixture from './fixture.json';
import * as loop from '$lib/nowcast/timeline';
import { parseEvent } from './load';
import type { EventDetail } from './schema';
import {
	clampIndex,
	cursorForIndex,
	frameDelayMs,
	frameTicks,
	markers,
	msAtPosition,
	msOf,
	nearestFrameIndex,
	nextFrameIndex,
	positionAt,
	trackBounds
} from './timeline';

const at = (iso: string) => Date.parse(iso);
const details = fixture.details as Record<string, any>;
const parsed = (eventId: string): EventDetail =>
	parseEvent(JSON.parse(JSON.stringify(details[eventId])))!;

/** A stamp as the builder writes it: `%Y%m%d%H%M` in UTC. */
const stamp = (iso: string) => iso.slice(0, 16).replace(/[-:T]/g, '');

/**
 * A minimal event carrying exactly the frames given, built through the real
 * parser so the test cannot drift from what a bundle actually produces.
 */
function eventWithFrames(frames: string[], extra: Record<string, unknown> = {}): EventDetail {
	const row = JSON.parse(JSON.stringify(fixture.index.events[0]));
	return parseEvent({
		schema_version: 1,
		bundle_id: 'inline',
		event_id: row.event_id,
		index: row,
		window: {
			anchor_utc: frames[0],
			from_utc: frames[0],
			to_utc: frames[frames.length - 1],
			frames_from_utc: frames[0]
		},
		frames: frames.map((iso) => ({
			radar_ts_utc: iso,
			stamp: stamp(iso),
			product: 'fullRange',
			overlay: '',
			observed: '',
			present: true
		})),
		decisions: [],
		...extra
	})!;
}

describe('msOf', () => {
	it('parses both ISO spellings and refuses everything else', () => {
		expect(msOf('2026-06-12T13:45:00Z')).toBe(at('2026-06-12T13:45:00Z'));
		expect(msOf('2026-06-12T13:45:00+00:00')).toBe(at('2026-06-12T13:45:00Z'));
		expect(msOf('yesterday')).toBeNull();
		expect(msOf('')).toBeNull();
		expect(msOf(null)).toBeNull();
		expect(msOf(undefined)).toBeNull();
	});
});

describe('trackBounds', () => {
	it('spans the FRAME window, which starts before the decision window', () => {
		const event = parsed('fa-06126-20260612T1345Z');
		const bounds = trackBounds(event)!;
		expect(bounds.fromMs).toBe(at(event.window.frames_from_utc));
		expect(bounds.fromMs).toBeLessThan(at(event.window.from_utc));
		expect(bounds.spanMs).toBe(bounds.toMs - bounds.fromMs);
	});

	it('absorbs anything the bundle carries outside its own edges', () => {
		const event = parsed('fa-06126-20260612T1345Z');
		const late = { ...event, frames: [...event.frames] };
		late.frames.push({ ...event.frames[0], radar_ts_utc: '2026-06-12T23:00:00Z' });
		expect(trackBounds(late)!.toMs).toBe(at('2026-06-12T23:00:00Z'));
	});

	it('is null on a track with no width to divide by', () => {
		const instant = eventWithFrames(['2026-06-12T13:00:00Z']);
		expect(trackBounds(instant)).toBeNull();
	});
});

describe('positionAt', () => {
	/** 12:00 … 13:00, three 10-minute steps and then a 40-minute hole. */
	const event = eventWithFrames([
		'2026-06-12T12:00:00Z',
		'2026-06-12T12:10:00Z',
		'2026-06-12T12:20:00Z',
		'2026-06-12T13:00:00Z'
	]);
	const bounds = trackBounds(event)!;

	it('is proportional to time, so a 40-minute gap is four 10-minute steps wide', () => {
		const ticks = frameTicks(event);
		const step = ticks[2].position - ticks[1].position;
		const gap = ticks[3].position - ticks[2].position;
		expect(gap / step).toBeCloseTo(4, 10);
	});

	it('puts every instant on its own place, to the millisecond', () => {
		for (const iso of [
			'2026-06-12T12:00:00Z',
			'2026-06-12T12:07:30Z',
			'2026-06-12T12:30:00Z',
			'2026-06-12T13:00:00Z'
		]) {
			const expected = (at(iso) - bounds.fromMs) / bounds.spanMs;
			expect(positionAt(bounds, at(iso))).toBeCloseTo(expected, 12);
		}
	});

	it('clamps at both ends and says nothing about an unusable instant', () => {
		expect(positionAt(bounds, at('2026-06-12T09:00:00Z'))).toBe(0);
		expect(positionAt(bounds, at('2026-06-12T18:00:00Z'))).toBe(1);
		expect(positionAt(bounds, null)).toBeNull();
		expect(positionAt(bounds, Number.NaN)).toBeNull();
	});

	it('round-trips through msAtPosition', () => {
		const instant = at('2026-06-12T12:25:00Z');
		expect(msAtPosition(bounds, positionAt(bounds, instant)!)).toBe(instant);
		expect(msAtPosition(bounds, -1)).toBe(bounds.fromMs);
		expect(msAtPosition(bounds, 2)).toBe(bounds.toMs);
	});
});

describe('frameTicks', () => {
	const event = parsed('fa-06104-20260421T0255Z');

	it('places every frame of the event, in order', () => {
		const ticks = frameTicks(event);
		expect(ticks).toHaveLength(event.frames.length);
		const positions = ticks.map((tick) => tick.position);
		expect(positions).toEqual([...positions].sort((a, b) => a - b));
		expect(positions[0]).toBeGreaterThanOrEqual(0);
		expect(positions[positions.length - 1]).toBeLessThanOrEqual(1);
	});

	it('keeps the frames the builder could not write, marked absent', () => {
		const missing = frameTicks(event).filter((tick) => tick.present === false);
		expect(missing.length).toBe(event.index.frames_missing);
		// The hole stays on the track: dropping it would close the gap it makes.
		expect(missing.length).toBeGreaterThan(0);
	});

	it('carries the index the scrubber needs to seek by', () => {
		frameTicks(event).forEach((tick) => {
			expect(event.frames[tick.index].stamp).toBe(tick.stamp);
		});
	});
});

describe('markers', () => {
	it('marks the anchor, the warning, the onset and the gauge horizon', () => {
		const event = parsed('hit-06181-20260805T1715Z');
		const { points } = markers(event);
		const kinds = points.map((point) => point.kind);
		expect(kinds).toContain('anchor');
		expect(kinds).toContain('warning');
		expect(kinds).toContain('onset');
		expect(kinds).toContain('known_until');

		const anchor = points.find((point) => point.kind === 'anchor')!;
		expect(anchor.utc).toBe(event.window.anchor_utc);
		const warning = points.find((point) => point.kind === 'warning')!;
		expect(warning.isEventWarning).toBe(true);
	});

	it('sorts points by time and lands each on its own instant', () => {
		const event = parsed('hit-06181-20260805T1715Z');
		const bounds = trackBounds(event)!;
		const { points } = markers(event);
		expect(points.map((p) => p.ms)).toEqual([...points.map((p) => p.ms)].sort((a, b) => a - b));
		for (const point of points) {
			expect(point.position).toBeCloseTo(positionAt(bounds, point.ms)!, 12);
		}
	});

	it('draws a coverage gap as a band of its real width', () => {
		const event = parsed('fa-06104-20260421T0255Z');
		const { bands } = markers(event);
		const gap = bands.find((band) => band.kind === 'coverage_gap')!;
		expect(gap.minutes).toBe(event.decision_gaps[0].minutes);
		const bounds = trackBounds(event)!;
		const width = gap.to - gap.from;
		expect(width).toBeCloseTo((gap.minutes * 60_000) / bounds.spanMs, 12);
	});

	it('hatches everything past the gauge’s last report', () => {
		const event = parsed('fa-06104-20260421T0255Z');
		const { bands } = markers(event);
		const beyond = bands.find((band) => band.kind === 'beyond_known')!;
		expect(beyond.fromUtc).toBe(event.window.known_until_utc);
		// It runs to the end of the track, because that is how far the
		// ignorance reaches.
		expect(beyond.to).toBe(1);
	});

	it('marks the replay’s free re-arm, so a notify after it is not read as a bug', () => {
		const event = parsed('fa-06104-20260421T0255Z');
		const rearms = markers(event).points.filter((p) => p.kind === 'run_boundary_rearm');
		expect(rearms.map((point) => point.utc)).toEqual(
			event.prologue!.run_boundary_rearms_utc
		);
	});

	it('draws no band for an event with no gap and no horizon', () => {
		const event = eventWithFrames(['2026-06-12T12:00:00Z', '2026-06-12T13:00:00Z']);
		expect(markers(event).bands).toEqual([]);
	});
});

describe('nearestFrameIndex and cursorForIndex', () => {
	const frames = [
		{ radar_ts_utc: '2026-06-12T13:00:00Z' },
		{ radar_ts_utc: '2026-06-12T13:10:00Z' },
		{ radar_ts_utc: '2026-06-12T13:20:00Z' }
	];

	it('finds the frame the thumb is over', () => {
		expect(nearestFrameIndex(frames, at('2026-06-12T13:09:00Z'))).toBe(1);
		expect(nearestFrameIndex(frames, at('2026-06-12T13:19:00Z'))).toBe(2);
		expect(nearestFrameIndex(frames, at('2026-06-12T09:00:00Z'))).toBe(0);
		expect(nearestFrameIndex(frames, at('2026-06-12T20:00:00Z'))).toBe(2);
	});

	it('breaks a tie towards the frame that had already happened', () => {
		expect(nearestFrameIndex(frames, at('2026-06-12T13:05:00Z'))).toBe(0);
	});

	it('has no frame to be on when there are none', () => {
		expect(nearestFrameIndex([], at('2026-06-12T13:00:00Z'))).toBe(-1);
		expect(nearestFrameIndex(frames, Number.NaN)).toBe(-1);
	});

	it('turns an index back into the cursor instant', () => {
		expect(cursorForIndex(frames, 1)).toBe(at('2026-06-12T13:10:00Z'));
		expect(cursorForIndex(frames, 9)).toBeNull();
		expect(cursorForIndex(frames, -1)).toBeNull();
	});
});

describe('playback', () => {
	it('re-exports the public loop’s own helpers, not copies of them', () => {
		// Both loops must behave identically; one implementation is the only
		// way to guarantee that.
		expect(clampIndex).toBe(loop.clampIndex);
		expect(nextFrameIndex).toBe(loop.nextFrameIndex);
		expect(frameDelayMs).toBe(loop.frameDelayMs);
	});
});
