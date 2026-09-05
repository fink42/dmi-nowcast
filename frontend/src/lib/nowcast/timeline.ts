/**
 * The loop as a timeline: what the scrubber draws, and where playback goes
 * next.
 *
 * Everything here is pure — a manifest in, plain data out — because these are
 * the decisions that are easy to get subtly wrong and impossible to eyeball:
 * which frames are measurements and which are extrapolation, and what a seek
 * past a frame that is still downloading should do. The store owns the timers;
 * this module owns the arithmetic.
 *
 * Two positions on the track are different things and are kept apart by name:
 * the *latest observation* is the hinge between measured and extrapolated, and
 * the *clock* is wall-clock now. They are minutes apart — the newest composite
 * is 14–24 min old whenever anyone looks — and conflating them is what made
 * the panel and the picture disagree.
 */
import {
	frameKind,
	frameValidTs,
	overlayFrames,
	type FrameKind,
	type Manifest
} from './manifest';

export interface TimelineFrame {
	filename: string;
	/** Minutes relative to the radar frame: negative for observation history. */
	leadMin: number;
	kind: FrameKind;
	/** The instant this frame depicts (ISO 8601 UTC). */
	validTsUtc: string;
	/**
	 * The newest observation: the hinge between what was measured and what is
	 * extrapolated. It is emphatically *not* "now" — the composite behind it is
	 * 14–24 min old by the time anyone reads it, and wall-clock now sits well
	 * to the right of it, among the forecast frames (see `clockPosition`).
	 */
	isLatest: boolean;
}

/**
 * Every overlay frame of a cycle, oldest first: the schema-v2 observation
 * history (0–3 frames, fewer on a cold start), then the latest observation,
 * then the forecast leads — including a forecast at lead 0, the radar field
 * advected forward by its own age. A manifest with no overlays at all yields
 * an empty timeline, which the UI renders as "no frames" rather than as an
 * empty track.
 */
export function buildTimeline(manifest: Manifest | null): TimelineFrame[] {
	if (!manifest) return [];
	const entries = overlayFrames(manifest);
	const frames = entries.map((entry) => ({
		filename: entry.filename,
		leadMin: entry.lead_min ?? 0,
		kind: frameKind(entry),
		validTsUtc: frameValidTs(manifest, entry),
		isLatest: false
	}));
	const latest = latestIndex(frames);
	if (latest >= 0) frames[latest].isLatest = true;
	return frames;
}

/**
 * Index of the latest observation: the lead-0 *measurement* by preference,
 * otherwise the newest observation of any lead. −1 when a cycle served only
 * forecasts, which the track then draws with no history segment and no hinge
 * marker rather than inventing one.
 *
 * The kind test is load-bearing since schema v2: two frames carry lead 0, the
 * radar image and the deterministic field advected forward by the frame age.
 * Only the first of them is a measurement, and calling the extrapolation the
 * hinge would hatch a forecast frame as history.
 */
export function latestIndex(frames: readonly { leadMin: number; kind: FrameKind }[]): number {
	const zero = frames.findIndex((f) => f.kind === 'observation' && f.leadMin === 0);
	if (zero >= 0) return zero;
	let last = -1;
	frames.forEach((frame, i) => {
		if (frame.kind === 'observation') last = i;
	});
	return last;
}

export interface TimelineGeometry {
	count: number;
	latestIndex: number;
	/** Position of every frame along the track, 0 … 1. One tick per frame. */
	positions: number[];
	/**
	 * Where the latest-observation hinge sits, or null when a cycle served no
	 * observation at all.
	 */
	latestPosition: number | null;
	/** How the track splits. `historyCount` is 0 on a cold-start manifest. */
	historyCount: number;
	forecastCount: number;
}

/**
 * Track geometry. Frames are spaced evenly rather than by time, because the
 * scrubber is an index slider: a tick has to sit exactly where dragging the
 * thumb to it lands, and the leads are not evenly spaced in minutes.
 */
export function timelineGeometry(frames: readonly TimelineFrame[]): TimelineGeometry {
	const count = frames.length;
	const positions = frames.map((_, i) => (count > 1 ? i / (count - 1) : 0));
	const latest = latestIndex(frames);
	return {
		count,
		latestIndex: latest,
		positions,
		latestPosition: latest >= 0 ? positions[latest] : null,
		historyCount: frames.filter((f) => f.kind === 'observation' && !f.isLatest).length,
		forecastCount: frames.filter((f) => f.kind === 'forecast').length
	};
}

/**
 * Where wall-clock now sits on the track, 0 … 1, or null when it cannot be
 * placed.
 *
 * The track is index-spaced while the frames are not evenly spaced in time, so
 * this walks the two frames bracketing `nowMs` and interpolates between *their
 * positions*, not between their indices — the marker then sits under whatever
 * the loop would be drawing for this minute. Clamped at both ends: before the
 * first frame it pins to 0, after the last to 1, because a marker off the end
 * of the track is worse than one at the end of it.
 *
 * Null for a track too short to interpolate on (fewer than two frames) or any
 * stamp that will not parse — no marker beats a marker in the wrong place.
 * Pure; the caller supplies the clock.
 */
export function clockPosition(
	frames: readonly { validTsUtc: string }[],
	nowMs: number
): number | null {
	if (frames.length < 2 || !Number.isFinite(nowMs)) return null;
	const times = frames.map((f) => Date.parse(f.validTsUtc));
	if (!times.every((t) => Number.isFinite(t))) return null;
	const positions = frames.map((_, i) => i / (frames.length - 1));

	if (nowMs <= times[0]) return 0;
	if (nowMs >= times[times.length - 1]) return 1;
	for (let i = 0; i < times.length - 1; i++) {
		const a = times[i];
		const b = times[i + 1];
		if (nowMs < a || nowMs > b) continue;
		if (b === a) return positions[i];
		return positions[i] + ((nowMs - a) / (b - a)) * (positions[i + 1] - positions[i]);
	}
	// Frames out of chronological order: nothing brackets now, so say nothing.
	return null;
}

/**
 * Frames stream in sequentially and in timeline order, so the first
 * `loadedCount` of them are the ones with a bitmap. An index at or past that
 * is a frame the viewer has scrubbed to before it finished downloading —
 * which the UI must announce, because the map is still showing the previous
 * frame and nothing else would explain why.
 */
export const isBuffering = (index: number, loadedCount: number): boolean => index >= loadedCount;

/** Snap an arbitrary index (a range input's value, a stale one) into range. */
export function clampIndex(index: number, count: number): number {
	if (count <= 0) return 0;
	if (!Number.isFinite(index)) return 0;
	return Math.max(0, Math.min(count - 1, Math.round(index)));
}

/**
 * Where playback goes on the next tick.
 *
 * Buffering holds position: the viewer asked for that frame, and jumping
 * somewhere else to fill the silence is worse than waiting a tick for it.
 * Otherwise it is the next loaded frame, wrapping at the end of what has
 * arrived — during a cycle load that means the loop plays the frames it has
 * and grows into the rest.
 */
export function nextFrameIndex(index: number, loadedCount: number): number {
	if (loadedCount <= 0) return index;
	if (isBuffering(index, loadedCount)) return index;
	return index + 1 >= loadedCount ? 0 : index + 1;
}

/**
 * How long the current frame stays up: the last one gets an extra beat so the
 * loop reads as a loop instead of a stutter. A buffering frame is timed like
 * any other — the tick that finds it loaded is the one that moves on.
 */
export function frameDelayMs(
	index: number,
	loadedCount: number,
	frameMs: number,
	lastFrameHoldMs: number
): number {
	return loadedCount > 0 && index === loadedCount - 1 ? lastFrameHoldMs : frameMs;
}
