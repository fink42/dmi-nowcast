/**
 * The review track: one event's ±90 minutes, in ABSOLUTE time.
 *
 * This is the one place the review loop deliberately parts company with the
 * public site's scrubber. `nowcast/timeline.ts` spaces frames by index,
 * because there the track *is* a range input and a tick has to sit where
 * dragging the thumb to it lands. Here the track is a chart of what was
 * known when, and the single most important thing it has to show is a
 * **hole**: a coverage gap, a run of missing composites, the edge past
 * which the gauge has not reported. Index spacing draws a 40-minute gap the
 * same width as a 10-minute step, which turns the evidence the reviewer is
 * judging into a straight line. So every position here is
 * `(t − from) / span`, and a 40-minute gap is four times the width of a
 * 10-minute step by construction.
 *
 * Playback is the opposite: it must behave *identically* to the public
 * loop, so `clampIndex`, `nextFrameIndex` and `frameDelayMs` are imported
 * and re-exported rather than rewritten. One implementation, two tracks.
 *
 * Everything is pure and takes the cursor as a number of milliseconds since
 * the epoch. UTC in, positions out; the viewer's clock is applied by the
 * component at the render boundary and nowhere earlier.
 */
import { clampIndex, frameDelayMs, nextFrameIndex } from '$lib/nowcast/timeline';
import type { EventDetail, FrameRef } from './schema';

export { clampIndex, frameDelayMs, nextFrameIndex };

/**
 * An ISO stamp as epoch milliseconds, or null when it will not parse.
 *
 * Null is "we cannot place this", never "now" and never zero: a marker
 * dropped is a marker missing, while a marker at the wrong instant is a
 * reviewer judging the wrong minute.
 */
export function msOf(value: string | null | undefined): number | null {
	if (typeof value !== 'string' || value.trim() === '') return null;
	const ms = Date.parse(value);
	return Number.isFinite(ms) ? ms : null;
}

export interface TrackBounds {
	fromUtc: string;
	toUtc: string;
	fromMs: number;
	toMs: number;
	/** Milliseconds the whole track spans. Always > 0. */
	spanMs: number;
}

/**
 * The instants the track runs between.
 *
 * The frame window is the outer one — composites are 13–24 min old, so the
 * builder starts the frames two cadences before the decision window (see
 * `window.frames_from_utc`) — and the track has to show all of it or the
 * earliest decision would stand on a frame off the left edge. Frames and
 * decisions outside even that are absorbed rather than clipped: a bundle
 * whose edges disagree with its own contents should draw everything it
 * carries, and the alternative is a marker pinned to the end of the track
 * with no way to tell it apart from one that genuinely landed there.
 *
 * Null when the event carries no usable instants at all, or when the span
 * collapses to zero — there is no proportional position on a track of no
 * width, and the caller must render "no timeline" rather than divide by it.
 */
export function trackBounds(event: EventDetail): TrackBounds | null {
	const candidates: number[] = [];
	const push = (value: string | null | undefined) => {
		const ms = msOf(value);
		if (ms !== null) candidates.push(ms);
	};
	push(event.window.frames_from_utc);
	push(event.window.from_utc);
	push(event.window.to_utc);
	for (const frame of event.frames) push(frame.radar_ts_utc);
	for (const decision of event.decisions) push(decision.generated_at_utc);
	if (candidates.length === 0) return null;
	const fromMs = Math.min(...candidates);
	const toMs = Math.max(...candidates);
	if (!(toMs > fromMs)) return null;
	return {
		fromUtc: new Date(fromMs).toISOString(),
		toUtc: new Date(toMs).toISOString(),
		fromMs,
		toMs,
		spanMs: toMs - fromMs
	};
}

/**
 * Where an instant sits on the track, 0 … 1, proportional to TIME.
 *
 * Clamped at both ends, as `clockPosition` is: a marker off the end of the
 * track is worse than one at the end of it. Null for an instant that will
 * not parse, which draws nothing.
 */
export function positionAt(bounds: TrackBounds, ms: number | null): number | null {
	if (ms === null || !Number.isFinite(ms)) return null;
	const position = (ms - bounds.fromMs) / bounds.spanMs;
	return Math.max(0, Math.min(1, position));
}

/** The inverse, for click-to-seek: a position on the track → an instant. */
export function msAtPosition(bounds: TrackBounds, position: number): number {
	if (!Number.isFinite(position)) return bounds.fromMs;
	const clamped = Math.max(0, Math.min(1, position));
	return Math.round(bounds.fromMs + clamped * bounds.spanMs);
}

export interface FrameTick {
	/** Index into `event.frames` — what the scrubber's thumb carries. */
	index: number;
	stamp: string;
	radarTsUtc: string;
	ms: number;
	position: number;
	/**
	 * False when the builder could not write this composite, null when it did
	 * not say. The tick is drawn either way — hollow for a known hole, plain
	 * for an unknown one — because a frame the bundle knows is missing is
	 * evidence, and dropping it would close the hole it makes.
	 */
	present: boolean | null;
}

/** Every composite of the event, placed in time. Frames whose stamp will
 * not parse are dropped — they cannot be placed and cannot be fetched. */
export function frameTicks(event: EventDetail): FrameTick[] {
	const bounds = trackBounds(event);
	if (bounds === null) return [];
	const ticks: FrameTick[] = [];
	event.frames.forEach((frame, index) => {
		const ms = msOf(frame.radar_ts_utc);
		const position = positionAt(bounds, ms);
		if (ms === null || position === null) return;
		ticks.push({
			index,
			stamp: frame.stamp,
			radarTsUtc: frame.radar_ts_utc,
			ms,
			position,
			present: frame.present
		});
	});
	return ticks;
}

export type TrackMarkerKind =
	| 'anchor'
	| 'onset'
	| 'warning'
	| 'already_raining'
	| 'known_until'
	/**
	 * An instant where the REPLAY reset engine state at a coverage-run
	 * boundary, handing this station a re-arm the live service never had. A
	 * notify just after one of these is an artefact of replaying, not
	 * something the service would have sent — and a reviewer looking at an
	 * unmarked track reads it as a bug in the tool and tags a mechanism that
	 * does not exist.
	 */
	| 'run_boundary_rearm';

export interface TrackMarker {
	kind: TrackMarkerKind;
	utc: string;
	ms: number;
	position: number;
	/** The event's own warning, as opposed to another one in the window. */
	isEventWarning: boolean;
	/** Whatever the marker needs beside it — a probability, an ETA, a reason. */
	detail: string | null;
}

export type TrackBandKind = 'coverage_gap' | 'beyond_known';

export interface TrackBand {
	kind: TrackBandKind;
	fromUtc: string;
	toUtc: string;
	fromMs: number;
	toMs: number;
	/** Track positions, 0 … 1. `to` may equal `from` on a clamped band. */
	from: number;
	to: number;
	minutes: number;
	reason: string;
}

export interface TrackMarkers {
	points: TrackMarker[];
	bands: TrackBand[];
}

/**
 * Everything drawn *on* the track besides the frames.
 *
 * Points and bands are separated because they are read differently: a point
 * is an instant a reviewer can seek to, a band is a span where the evidence
 * is absent and the reviewer must not read the blank as "nothing happened".
 * The two bands are the reason this module exists — a coverage gap, and
 * everything past `known_until`, where the gauge has not reported yet and
 * "silent" is emphatically not "dry".
 *
 * The anchor comes first and the event's own warning is flagged, so the UI
 * can draw the instant the event is *about* differently from the other
 * warnings that happened to fall in the same window.
 */
export function markers(event: EventDetail): TrackMarkers {
	const bounds = trackBounds(event);
	if (bounds === null) return { points: [], bands: [] };

	const points: TrackMarker[] = [];
	const add = (
		kind: TrackMarkerKind,
		utc: string | null,
		detail: string | null = null,
		isEventWarning = false
	) => {
		const ms = msOf(utc);
		const position = positionAt(bounds, ms);
		if (utc === null || ms === null || position === null) return;
		points.push({ kind, utc, ms, position, isEventWarning, detail });
	};

	add('anchor', event.window.anchor_utc);
	// Every onset the gauge recorded in the window, not only the event's:
	// a miss whose window holds three onsets is a different event from one
	// that holds a single drop.
	for (const onset of event.gauge?.onsets ?? []) {
		add(
			'onset',
			onset.onset_utc,
			onset.two_slot_mm === null ? null : `${onset.two_slot_mm.toFixed(1)} mm`
		);
	}
	for (const notification of event.notifications) {
		const detail =
			notification.p_decision === null
				? null
				: `p=${Math.round(notification.p_decision * 100)}%`;
		add(
			notification.action === 'already_raining' ? 'already_raining' : 'warning',
			notification.generated_at_utc,
			detail,
			notification.is_event_warning
		);
	}
	for (const at of event.prologue?.run_boundary_rearms_utc ?? []) {
		add('run_boundary_rearm', at, 'the replay re-armed here; the live service did not');
	}
	add('known_until', event.window.known_until_utc, 'gauge has not reported past here');

	const bands: TrackBand[] = [];
	const addBand = (
		kind: TrackBandKind,
		fromUtc: string | null,
		toUtc: string | null,
		reason: string,
		minutes?: number
	) => {
		const fromMs = msOf(fromUtc);
		const toMs = msOf(toUtc);
		const from = positionAt(bounds, fromMs);
		const to = positionAt(bounds, toMs);
		if (fromUtc === null || toUtc === null || fromMs === null || toMs === null) return;
		if (from === null || to === null || toMs <= fromMs) return;
		bands.push({
			kind,
			fromUtc,
			toUtc,
			fromMs,
			toMs,
			from,
			to,
			minutes: minutes ?? (toMs - fromMs) / 60_000,
			reason
		});
	};

	for (const gap of event.decision_gaps) {
		addBand('coverage_gap', gap.from_utc, gap.to_utc, gap.reason, gap.minutes);
	}
	// Past the last reported slot the gauge is silent, not dry. The band runs
	// to the end of the track because that is exactly how far the ignorance
	// reaches.
	addBand(
		'beyond_known',
		event.window.known_until_utc,
		bounds.toUtc,
		'no gauge report past this instant'
	);

	return {
		points: points.sort((a, b) => a.ms - b.ms),
		bands: bands.sort((a, b) => a.fromMs - b.fromMs)
	};
}

/**
 * The frame the cursor is on: nearest in ABSOLUTE time, not the newest at or
 * before it. This answers "which frame is the thumb over" — `estimate.ts`
 * answers the different and stricter question of which frame the map may
 * show, where anything after the cursor is a frame the service could not
 * have had.
 *
 * Ties resolve to the earlier frame, so a cursor exactly between two
 * composites shows the one that had already happened. −1 when there is no
 * frame to be on, which the scrubber renders as no thumb rather than as
 * frame zero.
 */
export function nearestFrameIndex(
	frames: readonly Pick<FrameRef, 'radar_ts_utc'>[],
	cursorMs: number
): number {
	if (frames.length === 0 || !Number.isFinite(cursorMs)) return -1;
	let best = -1;
	let bestDistance = Infinity;
	frames.forEach((frame, index) => {
		const ms = msOf(frame.radar_ts_utc);
		if (ms === null) return;
		const distance = Math.abs(ms - cursorMs);
		if (distance < bestDistance) {
			bestDistance = distance;
			best = index;
		}
	});
	return best;
}

/**
 * Where the cursor goes when playback steps to a frame: that frame's own
 * radar instant. Null for an index off the end or a stamp that will not
 * parse — the caller leaves the cursor where it was rather than jumping to
 * the epoch.
 */
export function cursorForIndex(
	frames: readonly Pick<FrameRef, 'radar_ts_utc'>[],
	index: number
): number | null {
	const frame = frames[index];
	return frame === undefined ? null : msOf(frame.radar_ts_utc);
}
