/**
 * Laying the evidence out along the track.
 *
 * Everything on the review page that is drawn *over time* — the gauge strip,
 * the radar strip, the neighbour strips, the arm-state band — is the same
 * shape: a run of some state, placed at `(t − from) / span` so a 40-minute
 * hole is four times the width of a 10-minute step. `review/timeline.ts`
 * owns that arithmetic and `review/truth.ts` owns what the states are; this
 * file is the join between them, and it decides nothing of its own.
 *
 * That is the point of it: the arm band does NOT re-derive engine state.
 * It asks `armStateAt` at the instants the trace changed, exactly as the
 * estimate panel asks it at the cursor, so the band and the panel cannot
 * disagree — a second engine in TypeScript would eventually argue with the
 * Python one, and then the tool would be arguing with the thing it exists to
 * audit.
 *
 * It lives beside the components rather than in `$lib/review/` because it is
 * layout, not logic: positions in a bar, not facts about weather.
 */
import { armStateAt, type ArmState } from '$lib/review/estimate';
import { msOf, positionAt, type TrackBounds } from '$lib/review/timeline';
import { radarStateAt, wetRunSegments, type SlotState } from '$lib/review/truth';
import type { Decision, Prologue, RadarDiscBlock, Slot } from '$lib/review/schema';

export interface StripSegment {
	/** Three states, never two: `unknown` is drawn hatched and never merges. */
	state: SlotState['state'];
	/** Track positions, 0 … 1. */
	from: number;
	to: number;
	fromUtc: string;
	toUtc: string;
	/** Depth over the run, or null when any slot in it had no value. */
	mm: number | null;
	slots: number;
}

/**
 * A slot series as blocks on the track.
 *
 * Runs whose ends will not place are dropped rather than clamped to the
 * edge: a block at the wrong instant is worse than a block missing, and the
 * strip is evidence.
 */
export function stripSegments(
	slots: readonly Slot[],
	slotMin: number,
	bounds: TrackBounds | null
): StripSegment[] {
	if (bounds === null) return [];
	const out: StripSegment[] = [];
	for (const run of wetRunSegments(slots, slotMin)) {
		const from = positionAt(bounds, run.fromMs);
		const to = positionAt(bounds, run.toMs);
		if (from === null || to === null || to <= from) continue;
		out.push({
			state: run.state,
			from,
			to,
			fromUtc: run.fromUtc,
			toUtc: run.toUtc,
			mm: run.mm,
			slots: run.slots
		});
	}
	return out;
}

/**
 * The radar disc as blocks, from the slot series when the bundle carries one
 * and from the p90 series otherwise.
 *
 * The fallback asks `radarStateAt` at each series instant rather than
 * comparing p90 against the threshold here, so the strip applies the
 * bundle's own statistic and the bundle's own threshold — and a null p90
 * comes back as `unknown`, which is what "off coverage" or "an all-nodata
 * disc" actually is. Each block is one cadence wide, which is as far as a
 * composite speaks for.
 */
export function discSegments(
	disc: RadarDiscBlock | null,
	bounds: TrackBounds | null,
	slotMin: number,
	cadenceMin: number
): StripSegment[] {
	if (disc === null || bounds === null) return [];
	if (disc.slots.length > 0) return stripSegments(disc.slots, slotMin, bounds);

	const spanMs = Math.max(1, cadenceMin) * 60_000;
	const out: StripSegment[] = [];
	for (const entry of disc.series) {
		const endMs = msOf(entry.radar_ts_utc);
		if (endMs === null) continue;
		const state = radarStateAt(disc, endMs, cadenceMin);
		const from = positionAt(bounds, endMs - spanMs);
		const to = positionAt(bounds, endMs);
		if (from === null || to === null || to <= from) continue;
		out.push({
			state: state.state,
			from,
			to,
			fromUtc: new Date(endMs - spanMs).toISOString(),
			toUtc: entry.radar_ts_utc,
			mm: entry.p90_mm_h,
			slots: 1
		});
	}
	return out;
}

export interface ArmSegment {
	/** False means a notification could not have been sent in this stretch. */
	armed: boolean;
	/** Whether the answer came from a traced decision or from the prologue. */
	source: ArmState['source'];
	streak: number;
	from: number;
	to: number;
	fromUtc: string;
	toUtc: string;
}

/**
 * The arm-state band.
 *
 * Sampled at the start of the track and at every traced decision, because
 * those are the only instants the engine's state could have changed, and
 * `armStateAt` is asked for each one — the same call the estimate panel
 * makes at the cursor. The stretch before the first decision is the
 * prologue's answer, which is the whole reason the prologue is mandatory:
 * the action that disarmed a station is usually outside the ±90-minute
 * window, and without it the reviewer sees a station that inexplicably never
 * fires.
 *
 * Adjacent stretches in the same state are merged so the band reads as one
 * fact rather than nineteen. Coverage gaps are NOT subtracted here — the
 * track hatches them over the top, so a stretch with no evidence under it
 * stays visibly a stretch with no evidence under it.
 */
export function armSegments(
	decisions: readonly Decision[],
	prologue: Prologue | null,
	bounds: TrackBounds | null,
	rearmAfterMin: number | undefined
): ArmSegment[] {
	if (bounds === null) return [];
	const instants = [bounds.fromMs];
	for (const decision of decisions) {
		const ms = msOf(decision.generated_at_utc);
		if (ms === null || ms <= bounds.fromMs || ms > bounds.toMs) continue;
		instants.push(ms);
	}
	instants.sort((a, b) => a - b);

	const out: ArmSegment[] = [];
	instants.forEach((ms, index) => {
		const endMs = index + 1 < instants.length ? instants[index + 1] : bounds.toMs;
		if (endMs <= ms) return;
		const state = armStateAt(decisions, prologue, ms, rearmAfterMin);
		const previous = out[out.length - 1];
		if (previous !== undefined && previous.armed === state.armed && previous.source === state.source) {
			previous.to = positionAt(bounds, endMs) ?? previous.to;
			previous.toUtc = new Date(endMs).toISOString();
			previous.streak = state.streak;
			return;
		}
		const from = positionAt(bounds, ms);
		const to = positionAt(bounds, endMs);
		if (from === null || to === null || to <= from) return;
		out.push({
			armed: state.armed,
			source: state.source,
			streak: state.streak,
			from,
			to,
			fromUtc: new Date(ms).toISOString(),
			toUtc: new Date(endMs).toISOString()
		});
	});
	return out;
}
