/**
 * The planning half of the bitmap pipeline: which composites to fetch next,
 * which to throw away, and where they live.
 *
 * The fetching, decoding and warping half is the store's, because it owns
 * the network and an `ImageBitmap` is not something a test can assert
 * about. What *is* worth pinning down is the arithmetic: a scrubber that
 * prefetches in the wrong order stutters at exactly the moment a reviewer
 * is looking at the interesting frame, and a cache with no bound holds a
 * few hundred megabytes of decoded RGBA — the product grid is 432×496×4
 * bytes a frame, and an event carries around two dozen of them.
 *
 * Prefetch runs FORWARD-first, interleaved with backward. Playback moves
 * forward, so the next frame must already be in hand on the next tick; but
 * a reviewer who drags the thumb backwards to look at something again
 * would otherwise wait for every frame twice. Interleaving costs one slot
 * of latency on playback and removes the whole stall on a reverse scrub.
 */
import { bundleUrl } from './load';

/** Which of the two PNGs a stamp has. */
export type FrameKind = 'overlay' | 'observed';

/**
 * How many decoded frames to keep. Twenty-four is one event's worth of
 * composites plus a little slack, which is the working set while a
 * reviewer scrubs; beyond that the next event's frames evict this one's
 * instead of accumulating.
 */
export const DEFAULT_CACHE_CAPACITY = 24;

/** How far either side of the cursor to fetch ahead, in frames. */
export const DEFAULT_PREFETCH_RADIUS = 4;

/**
 * The URL of one composite.
 *
 * Frame names are content-stamped (`202606121340.overlay.png`) and a
 * stamp's pixels never change, so the server marks them immutable for a
 * year and the scrubber never re-fetches one. That is also why the stamp,
 * not the index, is the cache key.
 */
export function frameUrl(stamp: string, kind: FrameKind): string {
	return bundleUrl(`frames/${stamp}.${kind}.png`);
}

/**
 * The stamps to fetch, in the order to fetch them.
 *
 * The cursor's own frame first — it is the one on screen — then one
 * forward, one back, two forward, two back, out to `radius`. Indices off
 * either end are skipped rather than clamped, so the plan never asks for
 * the same frame twice and never pads the end of an event with repeats of
 * its last frame.
 *
 * An out-of-range cursor yields an empty plan rather than a clamped one: it
 * means the caller is between events, and fetching the first event's frames
 * because the index happened to be zero is how a scrub lands on the wrong
 * day's weather.
 */
export function prefetchPlan(
	stamps: readonly string[],
	cursorIndex: number,
	radius: number = DEFAULT_PREFETCH_RADIUS
): string[] {
	if (stamps.length === 0) return [];
	if (!Number.isInteger(cursorIndex) || cursorIndex < 0 || cursorIndex >= stamps.length) {
		return [];
	}
	const span = Math.max(0, Math.round(Number.isFinite(radius) ? radius : 0));
	const plan: string[] = [stamps[cursorIndex]];
	const seen = new Set(plan);
	for (let step = 1; step <= span; step++) {
		for (const index of [cursorIndex + step, cursorIndex - step]) {
			const stamp = stamps[index];
			if (index < 0 || index >= stamps.length || stamp === undefined) continue;
			if (seen.has(stamp)) continue;
			seen.add(stamp);
			plan.push(stamp);
		}
	}
	return plan;
}

/**
 * The keys to evict from a least-recently-used cache, oldest first.
 *
 * `Map` iterates in insertion order, so this is an LRU only if the store
 * re-inserts on every hit (`cache.delete(key); cache.set(key, value)`).
 * That protocol lives with the caller because touching a map is a
 * mutation, and everything in this module is pure; the eviction *decision*
 * is the part worth testing, so it is the part that is here.
 *
 * A non-positive capacity evicts everything, which is what "cache off"
 * should do rather than an error.
 */
export function lruEvict(
	cache: ReadonlyMap<string, unknown>,
	capacity: number = DEFAULT_CACHE_CAPACITY
): string[] {
	const limit = Number.isFinite(capacity) ? Math.max(0, Math.floor(capacity)) : 0;
	const excess = cache.size - limit;
	if (excess <= 0) return [];
	return [...cache.keys()].slice(0, excess);
}

/**
 * The stamps an event's frames are keyed by, in track order, minus the ones
 * the builder said it could not write.
 *
 * A frame the bundle KNOWS is missing still belongs on the *track* — the
 * hole is evidence — but it must never enter a fetch plan, because a 404
 * mid-playback is indistinguishable to the eye from a slow network. A frame
 * whose presence the bundle did not state (`present: null`) IS planned: an
 * untried frame is a hole we made ourselves, and the fetch failing costs
 * one picture that the track already marks as uncertain.
 */
export function presentStamps(
	frames: readonly { stamp: string; present: boolean | null }[]
): string[] {
	return frames.filter((frame) => frame.present !== false).map((frame) => frame.stamp);
}
