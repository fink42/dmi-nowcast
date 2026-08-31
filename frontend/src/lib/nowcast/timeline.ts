/**
 * The loop as a timeline: what the scrubber draws, and where playback goes
 * next.
 *
 * Everything here is pure — a manifest in, plain data out — because these are
 * the decisions that are easy to get subtly wrong and impossible to eyeball:
 * which frames are measurements and which are extrapolation, where "now" sits
 * on the track, and what a seek past a frame that is still downloading should
 * do. The store owns the timers; this module owns the arithmetic.
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
	/** The newest observation — the hinge the past and the future turn on. */
	isNow: boolean;
}

/**
 * Every overlay frame of a cycle, oldest first: the schema-v2 observation
 * history (0–3 frames, fewer on a cold start), then "now", then the forecast
 * leads. A manifest with no overlays at all yields an empty timeline, which
 * the UI renders as "no frames" rather than as an empty track.
 */
export function buildTimeline(manifest: Manifest | null): TimelineFrame[] {
	if (!manifest) return [];
	const entries = overlayFrames(manifest);
	const frames = entries.map((entry) => ({
		filename: entry.filename,
		leadMin: entry.lead_min ?? 0,
		kind: frameKind(entry),
		validTsUtc: frameValidTs(manifest, entry),
		isNow: false
	}));
	const now = nowIndex(frames);
	if (now >= 0) frames[now].isNow = true;
	return frames;
}

/**
 * Index of the "now" frame: lead 0 by preference, otherwise the newest
 * observation. −1 when a cycle served only forecasts, which the track then
 * draws with no history segment and no marker rather than inventing one.
 */
export function nowIndex(frames: readonly { leadMin: number; kind: FrameKind }[]): number {
	const zero = frames.findIndex((f) => f.leadMin === 0);
	if (zero >= 0) return zero;
	let last = -1;
	frames.forEach((frame, i) => {
		if (frame.kind === 'observation') last = i;
	});
	return last;
}

export interface TimelineGeometry {
	count: number;
	nowIndex: number;
	/** Position of every frame along the track, 0 … 1. One tick per frame. */
	positions: number[];
	/** Where the "now" marker sits, or null when there is no observation. */
	nowPosition: number | null;
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
	const now = nowIndex(frames);
	return {
		count,
		nowIndex: now,
		positions,
		nowPosition: now >= 0 ? positions[now] : null,
		historyCount: frames.filter((f) => f.kind === 'observation' && !f.isNow).length,
		forecastCount: frames.filter((f) => f.kind === 'forecast').length
	};
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
