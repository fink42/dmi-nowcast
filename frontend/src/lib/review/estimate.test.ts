/**
 * "What was known at the cursor" — the contract the whole review rests on.
 *
 * Three properties are defended here, and each of them is a way the tool
 * could lie without anyone noticing:
 *
 *  1. No estimate from the future. A decision shown before it existed makes
 *     a service that said nothing look as though it had been warning all
 *     along.
 *  2. The truth frame and the service frame are different pictures and both
 *     stamps survive the call. Showing one and labelling it the other
 *     flatters (or damns) every false alarm.
 *  3. The arm state follows the engine's own dry clock, INCLUDING the reset
 *     that any over-threshold observation performs while disarmed. That
 *     reset is why a station can sit disarmed through a whole showery
 *     afternoon, and why some misses were structurally unreachable — the
 *     finding that would be a rule change rather than a model change.
 *
 * The arm sequence is built inline rather than read from the fixture: it is
 * hand-worked, and a hand-worked answer must not move when the fixture is
 * regenerated.
 */
import { describe, expect, it } from 'vitest';
import {
	armStateAt,
	decisionsBetween,
	etaCountdownAt,
	frameForCursor,
	latestDecisionAt
} from './estimate';
import fixture from './fixture.json';
import { parseEvent } from './load';
import type { EngineAction, FrameRef, Prologue } from './schema';

const at = (iso: string) => Date.parse(iso);

describe('latestDecisionAt', () => {
	const decisions = [
		{ generated_at_utc: '2026-06-12T13:05:00Z', p: 0.2 },
		{ generated_at_utc: '2026-06-12T13:15:00Z', p: 0.4 },
		{ generated_at_utc: '2026-06-12T13:25:00Z', p: 0.6 }
	];

	it('returns null before the first decision, not the oldest one', () => {
		expect(latestDecisionAt(decisions, at('2026-06-12T12:00:00Z'))).toBeNull();
		expect(latestDecisionAt(decisions, at('2026-06-12T13:04:59Z'))).toBeNull();
	});

	it('is exact at the boundary instant', () => {
		// The decision exists AT the instant it was generated.
		expect(latestDecisionAt(decisions, at('2026-06-12T13:05:00Z'))!.p).toBe(0.2);
		expect(latestDecisionAt(decisions, at('2026-06-12T13:04:59.999Z'))).toBeNull();
	});

	it('never returns a decision from the future', () => {
		for (let minute = 0; minute <= 60; minute++) {
			const cursor = at('2026-06-12T13:00:00Z') + minute * 60_000;
			const answer = latestDecisionAt(decisions, cursor);
			if (answer === null) continue;
			expect(Date.parse(answer.generated_at_utc)).toBeLessThanOrEqual(cursor);
		}
	});

	it('does not care what order the rows arrived in', () => {
		const shuffled = [decisions[2], decisions[0], decisions[1]];
		expect(latestDecisionAt(shuffled, at('2026-06-12T13:20:00Z'))!.p).toBe(0.4);
	});

	it('skips rows it cannot date rather than ordering them arbitrarily', () => {
		const withJunk = [...decisions, { generated_at_utc: 'never', p: 9 }];
		expect(latestDecisionAt(withJunk, at('2026-06-12T14:00:00Z'))!.p).toBe(0.6);
	});

	it('answers null for a cursor that is not a time', () => {
		expect(latestDecisionAt(decisions, Number.NaN)).toBeNull();
	});
});

describe('decisionsBetween', () => {
	it('returns the window’s rows, oldest first', () => {
		const rows = [
			{ generated_at_utc: '2026-06-12T13:25:00Z' },
			{ generated_at_utc: '2026-06-12T13:05:00Z' },
			{ generated_at_utc: '2026-06-12T13:15:00Z' }
		];
		const between = decisionsBetween(
			rows,
			at('2026-06-12T13:05:00Z'),
			at('2026-06-12T13:15:00Z')
		);
		expect(between.map((row) => row.generated_at_utc)).toEqual([
			'2026-06-12T13:05:00Z',
			'2026-06-12T13:15:00Z'
		]);
	});
});

describe('etaCountdownAt', () => {
	const decision = { eta_min: 30, generated_at_utc: '2026-06-12T13:00:00Z' };

	it('counts the cycle’s own ETA down as the cursor moves', () => {
		expect(etaCountdownAt(decision, at('2026-06-12T13:00:00Z'))).toBeCloseTo(30, 6);
		expect(etaCountdownAt(decision, at('2026-06-12T13:12:00Z'))).toBeCloseTo(18, 6);
	});

	it('floors at zero rather than counting up into the past', () => {
		expect(etaCountdownAt(decision, at('2026-06-12T14:00:00Z'))).toBe(0);
	});

	it('keeps a null ETA null — no rain within the horizon is not "now"', () => {
		expect(etaCountdownAt({ ...decision, eta_min: null }, at('2026-06-12T13:10:00Z'))).toBeNull();
	});

	it('says nothing before the decision existed, and nothing with no decision', () => {
		expect(etaCountdownAt(decision, at('2026-06-12T12:50:00Z'))).toBeNull();
		expect(etaCountdownAt(null, at('2026-06-12T13:10:00Z'))).toBeNull();
	});
});

// ---------------------------------------------------------------------------
// The hand-worked arm sequence
// ---------------------------------------------------------------------------

interface Trace {
	generated_at_utc: string;
	replay: {
		armed_after: boolean;
		streak_after: number;
		below_since_utc: string | null;
		action: EngineAction;
		run_boundary_rearm: boolean;
	} | null;
}

const traced = (
	generated: string,
	armed: boolean,
	belowSince: string | null,
	action: EngineAction = 'none',
	streak = 0
): Trace => ({
	generated_at_utc: generated,
	replay: {
		armed_after: armed,
		streak_after: streak,
		below_since_utc: belowSince,
		action,
		run_boundary_rearm: false
	}
});

/**
 * One station's afternoon, worked out by hand against the engine's rule:
 * disarm on notify, re-arm only after 60 minutes below threshold measured
 * from `below_since_utc`, and **any** over-threshold observation resets that
 * clock while disarmed.
 *
 *   12:00  notify            → disarmed, dry clock not started
 *   12:10  still over        → dry clock still not started
 *   12:20  below             → dry clock starts at 12:20
 *   12:40  OVER again        → dry clock RESET: back to not started
 *   12:50  below             → dry clock restarts at 12:50
 *   13:50  60 min of dry     → re-armed
 *
 * The station is therefore disarmed from 12:00 to 13:50 — an hour and fifty
 * minutes off the air after a single warning, because one shower crossed it
 * halfway through.
 */
const SHOWERY_AFTERNOON: Trace[] = [
	traced('2026-07-04T12:00:00Z', false, null, 'notify', 2),
	traced('2026-07-04T12:10:00Z', false, null, 'none', 3),
	traced('2026-07-04T12:20:00Z', false, '2026-07-04T12:20:00Z'),
	traced('2026-07-04T12:30:00Z', false, '2026-07-04T12:20:00Z'),
	traced('2026-07-04T12:40:00Z', false, null, 'none', 1),
	traced('2026-07-04T12:50:00Z', false, '2026-07-04T12:50:00Z'),
	traced('2026-07-04T13:00:00Z', false, '2026-07-04T12:50:00Z'),
	traced('2026-07-04T13:40:00Z', false, '2026-07-04T12:50:00Z'),
	traced('2026-07-04T13:50:00Z', true, null)
];

const PROLOGUE: Prologue = {
	run_id: 1,
	run_start_utc: '2026-07-04T00:00:00Z',
	armed_at_window_start: true,
	streak_at_window_start: 0,
	below_since_utc: '2026-07-04T10:00:00Z',
	minutes_to_rearm_at_window_start: null,
	last_notify_utc: '2026-07-04T09:15:00Z',
	last_already_raining_utc: null,
	recent_actions: [],
	rearm_after_min: 60,
	at_anchor: { armed: false, streak: 0, minutes_to_rearm: null },
	run_boundary_rearms_utc: [],
	note: null
};

describe('armStateAt', () => {
	it('answers from the prologue before the first traced decision', () => {
		const state = armStateAt(SHOWERY_AFTERNOON, PROLOGUE, at('2026-07-04T11:30:00Z'));
		expect(state.source).toBe('prologue');
		expect(state.armed).toBe(true);
		expect(state.lastNotifyUtc).toBe('2026-07-04T09:15:00Z');
	});

	it('disarms on the notify, at the notify’s own instant', () => {
		const state = armStateAt(SHOWERY_AFTERNOON, PROLOGUE, at('2026-07-04T12:00:00Z'));
		expect(state.armed).toBe(false);
		expect(state.lastAction).toBe('notify');
		expect(state.lastNotifyUtc).toBe('2026-07-04T12:00:00Z');
	});

	it('reports no countdown while the dry clock has not started', () => {
		const state = armStateAt(SHOWERY_AFTERNOON, PROLOGUE, at('2026-07-04T12:05:00Z'));
		expect(state.armed).toBe(false);
		// Not "about to re-arm" and not a number: the clock has not started,
		// which `belowSinceUtc === null` is how the UI is told.
		expect(state.belowSinceUtc).toBeNull();
		expect(state.minutesToRearm).toBeNull();
	});

	it('counts down from the start of the dry run', () => {
		const state = armStateAt(SHOWERY_AFTERNOON, PROLOGUE, at('2026-07-04T12:30:00Z'));
		expect(state.belowSinceUtc).toBe('2026-07-04T12:20:00Z');
		expect(state.minutesToRearm).toBeCloseTo(50, 6);
	});

	it('RESETS the dry clock on an over-threshold observation while disarmed', () => {
		// 12:40 is over threshold. Ten minutes of dry time are thrown away and
		// the station is no closer to re-arming than it was at 12:00.
		const before = armStateAt(SHOWERY_AFTERNOON, PROLOGUE, at('2026-07-04T12:39:00Z'));
		const after = armStateAt(SHOWERY_AFTERNOON, PROLOGUE, at('2026-07-04T12:45:00Z'));
		expect(before.minutesToRearm).toBeCloseTo(41, 6);
		expect(after.belowSinceUtc).toBeNull();
		expect(after.minutesToRearm).toBeNull();
		expect(after.armed).toBe(false);

		// And the restarted clock runs the full hour again from 12:50.
		const restarted = armStateAt(SHOWERY_AFTERNOON, PROLOGUE, at('2026-07-04T13:00:00Z'));
		expect(restarted.belowSinceUtc).toBe('2026-07-04T12:50:00Z');
		expect(restarted.minutesToRearm).toBeCloseTo(50, 6);
	});

	it('stays disarmed for an hour and fifty minutes after one warning', () => {
		for (const instant of [
			'2026-07-04T12:00:00Z',
			'2026-07-04T12:45:00Z',
			'2026-07-04T13:30:00Z',
			'2026-07-04T13:49:59Z'
		]) {
			expect(armStateAt(SHOWERY_AFTERNOON, PROLOGUE, at(instant)).armed, instant).toBe(false);
		}
		expect(armStateAt(SHOWERY_AFTERNOON, PROLOGUE, at('2026-07-04T13:50:00Z')).armed).toBe(true);
	});

	it('clamps the countdown at zero and drops it once re-armed', () => {
		const rearmed = armStateAt(SHOWERY_AFTERNOON, PROLOGUE, at('2026-07-04T14:30:00Z'));
		expect(rearmed.armed).toBe(true);
		expect(rearmed.minutesToRearm).toBeNull();

		const overdue = armStateAt(
			[traced('2026-07-04T12:00:00Z', false, '2026-07-04T10:00:00Z')],
			PROLOGUE,
			at('2026-07-04T13:00:00Z')
		);
		expect(overdue.minutesToRearm).toBe(0);
	});

	it('measures against the bundle’s own re-arm constant, not a hardcoded 60', () => {
		const shortRule: Prologue = { ...PROLOGUE, rearm_after_min: 45 };
		const state = armStateAt(SHOWERY_AFTERNOON, shortRule, at('2026-07-04T13:00:00Z'));
		expect(state.rearmAfterMin).toBe(45);
		expect(state.minutesToRearm).toBeCloseTo(35, 6);
	});

	it('ignores rows that did not trace rather than treating them as state', () => {
		const withGap = [
			...SHOWERY_AFTERNOON.slice(0, 3),
			{ generated_at_utc: '2026-07-04T12:25:00Z', replay: null },
			...SHOWERY_AFTERNOON.slice(3)
		];
		const state = armStateAt(withGap, PROLOGUE, at('2026-07-04T12:25:00Z'));
		expect(state.belowSinceUtc).toBe('2026-07-04T12:20:00Z');
		expect(state.source).toBe('decision');
	});

	it('flags the replay’s free re-arm from the prologue’s own list', () => {
		const withBoundary: Prologue = {
			...PROLOGUE,
			run_boundary_rearms_utc: ['2026-07-04T13:20:00Z']
		};
		expect(
			armStateAt(SHOWERY_AFTERNOON, withBoundary, at('2026-07-04T13:10:00Z')).runBoundaryRearm
		).toBe(false);
		expect(
			armStateAt(SHOWERY_AFTERNOON, withBoundary, at('2026-07-04T13:25:00Z')).runBoundaryRearm
		).toBe(true);
	});

	it('assumes armed when nothing said, rather than excusing a miss', () => {
		const state = armStateAt([], null, at('2026-07-04T12:00:00Z'));
		expect(state.armed).toBe(true);
		expect(state.source).toBe('unknown');
	});
});

// ---------------------------------------------------------------------------
// Truth vs service
// ---------------------------------------------------------------------------

const frame = (iso: string, present = true): FrameRef => ({
	radar_ts_utc: iso,
	stamp: iso.slice(0, 16).replace(/[-:T]/g, ''),
	product: 'fullRange',
	overlay: '',
	observed: '',
	present,
	has_decision_row: true
});

const FRAMES = [
	frame('2026-06-12T13:00:00Z'),
	frame('2026-06-12T13:10:00Z'),
	frame('2026-06-12T13:20:00Z'),
	frame('2026-06-12T13:30:00Z')
];

const standing = (generated: string, radar: string) => ({
	generated_at_utc: generated,
	radar_ts_utc: radar,
	frame_ref: radar.slice(0, 16).replace(/[-:T]/g, '')
});

describe('frameForCursor', () => {
	const decisions = [
		standing('2026-06-12T13:05:00Z', '2026-06-12T12:50:00Z'),
		standing('2026-06-12T13:25:00Z', '2026-06-12T13:10:00Z')
	];
	const cursor = at('2026-06-12T13:33:00Z');

	it('shows a different frame in each mode at the same cursor', () => {
		const truth = frameForCursor(FRAMES, decisions, cursor, 'truth');
		const service = frameForCursor(FRAMES, decisions, cursor, 'service');
		expect(truth.frame!.radar_ts_utc).toBe('2026-06-12T13:30:00Z');
		expect(service.frame!.radar_ts_utc).toBe('2026-06-12T13:10:00Z');
		expect(truth.frame).not.toBe(service.frame);
	});

	it('carries BOTH stamps whichever mode was asked for', () => {
		for (const mode of ['truth', 'service'] as const) {
			const answer = frameForCursor(FRAMES, decisions, cursor, mode);
			expect(answer.truthTsUtc).toBe('2026-06-12T13:30:00Z');
			expect(answer.serviceTsUtc).toBe('2026-06-12T13:10:00Z');
			expect(answer.serviceLagMin).toBeCloseTo(20, 6);
		}
	});

	it('never shows a frame from after the cursor', () => {
		const answer = frameForCursor(FRAMES, decisions, at('2026-06-12T13:29:59Z'), 'truth');
		expect(answer.truth!.radar_ts_utc).toBe('2026-06-12T13:20:00Z');
	});

	it('keeps the service stamp when the bundle has no picture of that frame', () => {
		// The estimate stood on 12:50, which is outside this event's frames.
		const answer = frameForCursor(FRAMES, decisions, at('2026-06-12T13:06:00Z'), 'service');
		expect(answer.service).toBeNull();
		expect(answer.frame).toBeNull();
		expect(answer.serviceTsUtc).toBe('2026-06-12T12:50:00Z');
		expect(answer.truthTsUtc).toBe('2026-06-12T13:00:00Z');
	});

	it('has no estimate before the first decision, and no service frame either', () => {
		const answer = frameForCursor(FRAMES, decisions, at('2026-06-12T13:01:00Z'), 'service');
		expect(answer.decision).toBeNull();
		expect(answer.serviceTsUtc).toBeNull();
		expect(answer.truth!.radar_ts_utc).toBe('2026-06-12T13:00:00Z');
	});

	it('matches on the radar instant when the frame reference does not', () => {
		const odd = [{ ...standing('2026-06-12T13:25:00Z', '2026-06-12T13:10:00Z'), frame_ref: '' }];
		const answer = frameForCursor(FRAMES, odd, cursor, 'service');
		expect(answer.service!.radar_ts_utc).toBe('2026-06-12T13:10:00Z');
	});
});

describe('against the fixture', () => {
	const event = parseEvent(
		JSON.parse(JSON.stringify((fixture.details as Record<string, unknown>)['fa-06126-20260612T1345Z']))
	)!;

	it('reads the event’s own warning as the estimate at its anchor', () => {
		const anchor = Date.parse(event.window.anchor_utc);
		const decision = latestDecisionAt(event.decisions, anchor)!;
		expect(decision.generated_at_utc).toBe(event.window.anchor_utc);
		expect(decision.replay!.action).toBe('notify');
	});

	it('shows a service frame older than the truth frame at the anchor', () => {
		const anchor = Date.parse(event.window.anchor_utc);
		const answer = frameForCursor(event.frames, event.decisions, anchor, 'truth');
		expect(answer.serviceLagMin).toBeGreaterThan(0);
		expect(Date.parse(answer.serviceTsUtc!)).toBeLessThan(Date.parse(answer.truthTsUtc!));
	});
});
