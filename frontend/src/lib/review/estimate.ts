/**
 * What was known at the cursor — and nothing that was not.
 *
 * This is the module the whole tool stands on. A reviewer scrubbing to
 * 13:40 is asking "what did the service have at 13:40, and was it
 * defensible?", and every function here exists to stop the answer being
 * contaminated by evidence that arrived later:
 *
 *  - `latestDecisionAt` never returns a decision from the future, and
 *    returns null before the first one rather than the oldest. A window
 *    opens 90 minutes before the anchor and the first cycle inside it
 *    lands minutes later; showing that cycle's numbers for the empty
 *    stretch before it would make the service look as if it had been
 *    warning all along.
 *  - `frameForCursor` keeps the two frames apart by name. The newest
 *    composite at the cursor (`truth`) is 13–18 minutes newer than the one
 *    the current estimate stood on (`service`). Defaulting to `service`
 *    flatters every false alarm; defaulting to `truth` and hiding the
 *    difference shows rain the service could not have known about. Both
 *    stamps come back from the same call so the UI can print them side by
 *    side, which is the only presentation that is not misleading.
 *  - `armStateAt` reads the engine state the replay traced, rather than
 *    re-deriving it. A second engine in TypeScript would eventually
 *    disagree with the Python one, and then the tool would be arguing with
 *    the thing it exists to audit. The one piece of arithmetic here is the
 *    countdown to re-arm, which the trace does not carry per instant.
 *
 * Every function takes the cursor as epoch milliseconds and is pure. Inputs
 * are typed structurally, so a test can hand them three-field objects and
 * the store can hand them parsed bundle documents.
 */
import { countdownEtaMin } from '$lib/format';
import { msOf } from './timeline';
import type { Decision, EngineAction, FrameMode, FrameRef, Prologue } from './schema';

/**
 * The re-arm window the service shipped with, in minutes.
 *
 * A last resort only. The bundle states its own — `prologue.rearm_after_min`
 * first, then the manifest's `rule.rearm_after_min` — because the rule is
 * configurable, and a countdown drawn against 60 when the bundle was
 * decided at 45 is wrong by a quarter of an hour with nothing on screen to
 * say so.
 */
export const DEFAULT_REARM_AFTER_MIN = 60;

/** The minimum a decision needs to be placed on the clock. */
interface Dated {
	generated_at_utc: string;
}

/**
 * The newest decision at or before the cursor, or null before the first.
 *
 * Linear rather than binary-searched on purpose: a window holds at most
 * ~19 decisions, the input is not guaranteed sorted (a bundle rebuilt from
 * two decision trees can interleave), and a wrong answer here is invisible
 * in the UI — it looks like a forecast, not like a bug. Rows whose stamp
 * will not parse are skipped rather than ordered arbitrarily.
 *
 * The comparison is `<=`: a cursor sitting exactly on a decision's own
 * instant sees that decision. That is the instant it came into existence.
 */
export function latestDecisionAt<T extends Dated>(
	decisions: readonly T[],
	cursorMs: number
): T | null {
	if (!Number.isFinite(cursorMs)) return null;
	let best: T | null = null;
	let bestMs = -Infinity;
	for (const decision of decisions) {
		const ms = msOf(decision.generated_at_utc);
		if (ms === null || ms > cursorMs) continue;
		if (ms > bestMs) {
			bestMs = ms;
			best = decision;
		}
	}
	return best;
}

/** Every decision in `[fromMs, toMs]`, oldest first — the read-out's series. */
export function decisionsBetween<T extends Dated>(
	decisions: readonly T[],
	fromMs: number,
	toMs: number
): T[] {
	return decisions
		.map((decision) => ({ decision, ms: msOf(decision.generated_at_utc) }))
		.filter(
			(entry): entry is { decision: T; ms: number } =>
				entry.ms !== null && entry.ms >= fromMs && entry.ms <= toMs
		)
		.sort((a, b) => a.ms - b.ms)
		.map((entry) => entry.decision);
}

/**
 * The estimate's ETA, counted down to the cursor — `format.ts`'s rule,
 * applied to a replayed cycle instead of a live one.
 *
 * The sidecar's ETA is minutes from `generated_at_utc`, so printing it
 * unchanged while the cursor moves through the cycle leaves "rain in 12
 * min" on screen twelve minutes after the rain was due. Subtracting the
 * cycle's age at the cursor makes the same forecast tick, exactly as the
 * live panel does — which matters here because the reviewer is judging
 * whether the warning was *timed* well.
 *
 * Null stays null (no rain within the horizon), and a cursor before the
 * decision existed yields null rather than a countdown running backwards.
 */
export function etaCountdownAt(
	decision: { eta_min: number | null; generated_at_utc: string } | null,
	cursorMs: number
): number | null {
	if (decision === null) return null;
	const generatedMs = msOf(decision.generated_at_utc);
	if (generatedMs === null || !Number.isFinite(cursorMs) || cursorMs < generatedMs) return null;
	return countdownEtaMin(decision.eta_min, decision.generated_at_utc, cursorMs);
}

export interface ArmState {
	/** False means a notification could not have been sent at this instant. */
	armed: boolean;
	/** Consecutive over-threshold observations behind the arm test. */
	streak: number;
	/**
	 * When the dry clock started. **Any** over-threshold observation while
	 * disarmed resets this, which is why a station can sit disarmed for hours
	 * in a showery spell and every onset in it is a miss the rule could not
	 * have caught. Null when nothing has said.
	 */
	belowSinceUtc: string | null;
	/**
	 * Minutes still to run before the station re-arms, or null when it is
	 * armed already or the dry clock's start is unknown. Counted from
	 * `belowSinceUtc` to the cursor, so it moves as the reviewer scrubs —
	 * and jumps back to the full window the moment a reset lands.
	 */
	minutesToRearm: number | null;
	/** The last notification at or before the cursor, from the trace or the prologue. */
	lastNotifyUtc: string | null;
	/**
	 * The last action the engine took at or before the cursor. `notify` and
	 * `already_raining` both disarm, and telling them apart is the whole of
	 * the `miss_arm_consumed_already_raining` tag.
	 */
	lastAction: EngineAction | null;
	/**
	 * The replay reset engine state at a coverage-run boundary at or before
	 * the cursor, handing this station a re-arm the live service never had.
	 * Without this the resulting notify reads as a bug in the tool.
	 */
	runBoundaryRearm: boolean;
	/** Whether the answer came from a traced decision or from the prologue. */
	source: 'decision' | 'prologue' | 'unknown';
	/** The re-arm window this countdown was measured against, in minutes. */
	rearmAfterMin: number;
}

/** The traced fields `armStateAt` reads off a decision. */
interface Traced extends Dated {
	replay: {
		armed_after: boolean;
		streak_after: number;
		below_since_utc: string | null;
		action: EngineAction;
		run_boundary_rearm: boolean;
	} | null;
}

/**
 * Engine state at the cursor.
 *
 * Before the first traced decision the answer is the prologue's, which is
 * mandatory for exactly this reason: the action that disarmed a station is
 * usually *outside* the ±90-minute window, and without the prologue the
 * reviewer sees a station that inexplicably never fires and tags the wrong
 * mechanism.
 *
 * Decisions whose `replay` block is missing are skipped rather than treated
 * as state changes: a row that did not trace says nothing about the engine,
 * and the last row that did is still the best knowledge there is.
 */
export function armStateAt(
	decisions: readonly Traced[],
	prologue: Prologue | null,
	cursorMs: number,
	rearmAfterMin?: number
): ArmState {
	// The bundle's own constant wins over the caller's, and both win over the
	// shipped default — this event was decided under one specific rule.
	const rearmMin =
		prologue?.rearm_after_min ?? rearmAfterMin ?? DEFAULT_REARM_AFTER_MIN;
	const traced = decisions.filter((decision) => decision.replay !== null);
	const latest = latestDecisionAt(traced, cursorMs);

	const lastNotify = latestDecisionAt(
		traced.filter((decision) => decision.replay?.action === 'notify'),
		cursorMs
	);
	const runBoundaryRearm =
		traced.some((decision) => {
			const ms = msOf(decision.generated_at_utc);
			return ms !== null && ms <= cursorMs && decision.replay?.run_boundary_rearm === true;
		}) ||
		(prologue?.run_boundary_rearms_utc ?? []).some((at) => {
			const ms = msOf(at);
			return ms !== null && ms <= cursorMs;
		});

	const state: ArmState =
		latest !== null && latest.replay !== null
			? {
					armed: latest.replay.armed_after,
					streak: latest.replay.streak_after,
					belowSinceUtc: latest.replay.below_since_utc,
					minutesToRearm: null,
					lastNotifyUtc: lastNotify?.generated_at_utc ?? prologue?.last_notify_utc ?? null,
					lastAction: latest.replay.action,
					runBoundaryRearm,
					source: 'decision',
					rearmAfterMin: rearmMin
				}
			: prologue !== null
				? {
						armed: prologue.armed_at_window_start,
						streak: prologue.streak_at_window_start,
						belowSinceUtc: prologue.below_since_utc,
						minutesToRearm: null,
						lastNotifyUtc: prologue.last_notify_utc,
						lastAction: null,
						runBoundaryRearm,
						source: 'prologue',
						rearmAfterMin: rearmMin
					}
				: {
						// No trace and no prologue. Armed is the state that lets a
						// warning happen, and claiming the engine was blocked when
						// nothing said so would excuse every miss in the window.
						armed: true,
						streak: 0,
						belowSinceUtc: null,
						minutesToRearm: null,
						lastNotifyUtc: null,
						lastAction: null,
						runBoundaryRearm,
						source: 'unknown',
						rearmAfterMin: rearmMin
					};

	state.minutesToRearm = minutesToRearm(state.armed, state.belowSinceUtc, cursorMs, rearmMin);
	state.rearmAfterMin = rearmMin;
	return state;
}

/**
 * Minutes left on the dry clock. Armed stations have none; a disarmed one
 * with no known start has an unknown countdown, which is null and not zero —
 * zero would read as "about to re-arm", the opposite of ignorance.
 */
function minutesToRearm(
	armed: boolean,
	belowSinceUtc: string | null,
	cursorMs: number,
	rearmAfterMin: number
): number | null {
	if (armed) return null;
	const belowSinceMs = msOf(belowSinceUtc);
	if (belowSinceMs === null || !Number.isFinite(cursorMs)) return null;
	const elapsedMin = (cursorMs - belowSinceMs) / 60_000;
	return Math.max(0, rearmAfterMin - elapsedMin);
}

/** What `frameForCursor` needs of a decision: two instants and a frame key. */
interface Standing extends Dated {
	radar_ts_utc: string;
	frame_ref: string;
}

export interface CursorFrame<T extends Standing = Decision> {
	/** Which of the two the caller asked for. */
	mode: FrameMode;
	/** The frame to draw. Null when the chosen mode has none at this cursor. */
	frame: FrameRef | null;
	/** Newest composite at or before the cursor: what was actually happening. */
	truth: FrameRef | null;
	/** The composite the current estimate stood on. */
	service: FrameRef | null;
	/** The estimate at the cursor, or null before the first one. */
	decision: T | null;
	/** Both stamps, always — the UI prints them together. */
	truthTsUtc: string | null;
	/**
	 * The service frame's instant. Retrievable even when the frame itself is
	 * missing from the bundle, because the decision carries its own
	 * `radar_ts_utc`: "the estimate stood on 13:20, which this bundle does
	 * not have a picture of" is a true and useful sentence.
	 */
	serviceTsUtc: string | null;
	/**
	 * How far behind the truth frame the service frame was, in minutes. This
	 * is the number that explains most false alarms and it is null only when
	 * one of the two stamps is missing.
	 */
	serviceLagMin: number | null;
}

/**
 * The frame the map should show at the cursor, plus everything needed to
 * say honestly which frame it is.
 *
 * `truth` is the newest composite at or before the cursor — including one
 * the builder could not write, because a hole in the record is evidence and
 * silently falling back to an older picture would hide it.
 *
 * `service` is resolved from the estimate's own `frame_ref` (a stamp), and
 * failing that by matching radar instants. The estimate's stamp is carried
 * out separately from the frame, so a missing PNG costs the picture and not
 * the fact.
 */
export function frameForCursor<T extends Standing = Decision>(
	frames: readonly FrameRef[],
	decisions: readonly T[],
	cursorMs: number,
	mode: FrameMode
): CursorFrame<T> {
	const decision = latestDecisionAt(decisions, cursorMs);

	let truth: FrameRef | null = null;
	let truthMs = -Infinity;
	for (const frame of frames) {
		const ms = msOf(frame.radar_ts_utc);
		if (ms === null || ms > cursorMs) continue;
		if (ms > truthMs) {
			truthMs = ms;
			truth = frame;
		}
	}

	const serviceTsUtc = decision?.radar_ts_utc ?? null;
	const service =
		decision === null
			? null
			: (frames.find((frame) => frame.stamp === decision.frame_ref) ??
				frames.find((frame) => frame.radar_ts_utc === decision.radar_ts_utc) ??
				null);

	const serviceMs = msOf(serviceTsUtc);
	const truthTsUtc = truth?.radar_ts_utc ?? null;
	return {
		mode,
		frame: mode === 'service' ? service : truth,
		truth,
		service,
		decision,
		truthTsUtc,
		serviceTsUtc,
		serviceLagMin:
			serviceMs === null || truthMs === -Infinity ? null : (truthMs - serviceMs) / 60_000
	};
}
