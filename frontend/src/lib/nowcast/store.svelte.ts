/**
 * The single source of truth for what the map page shows: the current cycle's
 * manifest, its reprojected overlay frames, the decoded product grids, the
 * animation position, and the forecast for whatever point the user picked.
 *
 * Polling is once a minute. The sidecar polls DMI every ~5 min for radar
 * composites that themselves arrive on a 10 min cadence (fullRange), and
 * artifact URLs are cycle-stamped and immutably cacheable, so a poll that
 * finds the same cycle costs one small conditional request and nothing else.
 */
import { browser } from '$app/environment';
import { loadOverlayFrames, overlayGeometry, type OverlayFrame } from '$lib/map/overlay';
import type { Corners } from '$lib/map/warp';
import { fetchPointForecast, withServedProbabilities } from './forecast';
import { freshness, type Freshness } from './freshness';
import { loadGrids } from './grids';
import { fetchManifest, isCalibrated, NoDataError, type Manifest } from './manifest';
import { gridToLonLat, samplePoint, type DecodedGrids, type PointForecast } from './sampler';
import {
	buildTimeline,
	clampIndex,
	frameDelayMs,
	isBuffering,
	nextFrameIndex,
	type TimelineFrame
} from './timeline';

const POLL_MS = 60_000;
const CLOCK_MS = 15_000;
/**
 * Background tabs get their timers throttled, so a phone coming out of a
 * pocket can be showing an age that is minutes wrong. Becoming visible
 * re-polls at once — but not more often than this, or flicking between tabs
 * turns into a request stream.
 */
const VISIBILITY_POLL_GAP_MS = 10_000;
/** Time each frame is shown, and the extra pause on the last one. */
const FRAME_MS = 550;
const LAST_FRAME_HOLD_MS = 1400;

export type Status = 'loading' | 'ready' | 'nodata' | 'error';

export interface PointState {
	lat: number;
	lon: number;
	status: 'loading' | 'ready' | 'off-coverage' | 'error';
	forecast: PointForecast | null;
	/**
	 * True while the server's `/forecast` answer for this point is still in
	 * flight. The client-side sample lands first and carries everything that
	 * comes off the grids — the deterministic series, the ETA, the motion
	 * arrow — but NOT the probability the site serves, which only the server
	 * can compute. The panel therefore withholds the probability bars and the
	 * "within 20 min" line until this clears, rather than drawing the
	 * curve-calibrated numbers for a moment and swapping them under the
	 * reader's eye.
	 */
	probabilitiesPending: boolean;
}

class NowcastStore {
	manifest = $state<Manifest | null>(null);
	status = $state<Status>('loading');
	/** The frames whose bitmaps have arrived — a prefix of `timeline`. */
	frames = $state<OverlayFrame[]>([]);
	/**
	 * Every frame of the cycle, known from the manifest before any of them has
	 * downloaded. The scrubber is built from this, not from `frames`: the
	 * track must be its full length from the start, or it grows under the
	 * viewer's thumb while they are dragging it.
	 */
	timeline = $state<TimelineFrame[]>([]);
	geometry = $state<{ corners: Corners } | null>(null);
	frameIndex = $state(0);
	playing = $state(true);
	point = $state<PointState | null>(null);
	/**
	 * The pipeline's confidence scalar. It is global for the whole cycle
	 * (Phase A keeps confidence national, not per-pixel) and does not travel
	 * in the artifacts, so it is picked up once per cycle with a single
	 * `/forecast` call at the grid centre and reused for every point.
	 */
	confidence = $state<number | null>(null);
	/** Ticks so "x min ago" stays honest without re-fetching anything. */
	now = $state(Date.now());

	#grids: DecodedGrids | null = null;
	#cycle: string | null = null;
	#cycleAbort: AbortController | null = null;
	#pollTimer: ReturnType<typeof setInterval> | null = null;
	#clockTimer: ReturnType<typeof setInterval> | null = null;
	#frameTimer: ReturnType<typeof setTimeout> | null = null;
	#onVisible: (() => void) | null = null;
	#lastPollAt = 0;
	/** Bumped per `selectPoint`, so a superseded answer is dropped on arrival. */
	#pointToken = 0;

	/** Radar age and pipeline liveness, kept apart on purpose — see freshness.ts. */
	get freshness(): Freshness {
		return freshness(this.manifest, this.now);
	}

	get radarAgeMin(): number | null {
		return this.freshness.radarAgeMin;
	}

	/**
	 * True when the manifest on screen is one we could not refresh: the poll is
	 * failing, so what is displayed is the last cycle that did arrive. Worth
	 * saying out loud — otherwise the age simply climbs with no explanation.
	 */
	get offlineWithCachedCycle(): boolean {
		return this.status === 'error' && this.manifest !== null;
	}

	get calibrated(): boolean {
		return this.manifest ? isCalibrated(this.manifest) : false;
	}

	/** The bitmap on the map, or null while the active frame is still loading. */
	get currentFrame(): OverlayFrame | null {
		return this.frames[this.frameIndex] ?? null;
	}

	/** What the active frame *is* — known even before its bitmap arrives. */
	get activeFrame(): TimelineFrame | null {
		return this.timeline[this.frameIndex] ?? null;
	}

	/** Length of the scrubber: every frame of the cycle, loaded or not. */
	get frameCount(): number {
		return Math.max(this.timeline.length, this.frames.length);
	}

	get loadedCount(): number {
		return this.frames.length;
	}

	/**
	 * True when the active frame has no bitmap yet. The map keeps showing the
	 * previous frame — the alternative is a blank hole in the middle of the
	 * country — so the controls have to say what is going on.
	 */
	get buffering(): boolean {
		return this.frameCount > 0 && isBuffering(this.frameIndex, this.frames.length);
	}

	/** Start polling and animating. Returns the matching teardown. */
	start(): () => void {
		if (!browser) return () => {};
		void this.refresh();
		this.#pollTimer = setInterval(() => void this.refresh(), POLL_MS);
		this.#clockTimer = setInterval(() => (this.now = Date.now()), CLOCK_MS);
		this.#onVisible = () => this.#onBecameVisible();
		document.addEventListener('visibilitychange', this.#onVisible);
		this.#scheduleFrame();
		return () => this.stop();
	}

	stop(): void {
		if (this.#pollTimer) clearInterval(this.#pollTimer);
		if (this.#clockTimer) clearInterval(this.#clockTimer);
		if (this.#frameTimer) clearTimeout(this.#frameTimer);
		if (this.#onVisible) document.removeEventListener('visibilitychange', this.#onVisible);
		this.#pollTimer = this.#clockTimer = null;
		this.#frameTimer = null;
		this.#onVisible = null;
		this.#cycleAbort?.abort();
	}

	/**
	 * A tab that was in the background has a throttled clock and a poll that
	 * may not have run for minutes. Re-stamp `now` first, so the age on screen
	 * is honest within the same frame, then refresh.
	 */
	#onBecameVisible(): void {
		if (document.visibilityState !== 'visible') return;
		this.now = Date.now();
		if (this.now - this.#lastPollAt < VISIBILITY_POLL_GAP_MS) return;
		void this.refresh();
	}

	async refresh(): Promise<void> {
		this.#lastPollAt = Date.now();
		try {
			const manifest = await fetchManifest();
			this.now = Date.now();
			if (manifest.cycle !== this.#cycle) {
				await this.#loadCycle(manifest);
			} else {
				this.manifest = manifest;
			}
			this.status = 'ready';
		} catch (err) {
			// The old manifest stays on screen — there is nothing better to show —
			// but `status` flips, and the page says so rather than letting the age
			// climb unexplained.
			this.now = Date.now();
			this.status = err instanceof NoDataError ? 'nodata' : 'error';
			if (!(err instanceof NoDataError)) console.warn('manifest poll failed', err);
		}
	}

	async #loadCycle(manifest: Manifest): Promise<void> {
		this.#cycleAbort?.abort();
		const abort = new AbortController();
		this.#cycleAbort = abort;
		this.#cycle = manifest.cycle;
		this.manifest = manifest;
		this.geometry = overlayGeometry(manifest);
		this.timeline = buildTimeline(manifest);
		// A new cycle replaces every bitmap. A playing loop restarts at the
		// oldest frame rather than stalling on an index whose image is a whole
		// download away; a paused viewer keeps the frame they were reading,
		// and it buffers visibly until it arrives.
		this.frameIndex = this.playing ? 0 : clampIndex(this.frameIndex, this.timeline.length);

		const previous = this.frames;
		const collected: OverlayFrame[] = [];
		void this.#loadConfidence(manifest, abort.signal);

		// Product grids and overlay frames are independent; a grid failure only
		// costs us the client-side sampling path, not the radar loop.
		const gridsPromise = loadGrids(manifest)
			.then((grids) => {
				if (!abort.signal.aborted) this.#grids = grids;
			})
			.catch((err) => {
				console.warn('product grids unavailable, using /forecast', err);
				this.#grids = null;
			});

		try {
			for await (const frame of loadOverlayFrames(manifest, abort.signal)) {
				collected.push(frame);
				// Swap the array in as it grows so the loop can start early.
				// An index past the end is not corrected here: that is the
				// buffering state, and it resolves itself as frames arrive.
				this.frames = [...collected];
			}
		} catch (err) {
			if (!abort.signal.aborted) console.warn('overlay frames failed', err);
		}
		await gridsPromise;

		if (abort.signal.aborted) {
			for (const frame of collected) frame.bitmap.close();
			return;
		}
		for (const frame of previous) frame.bitmap.close();
		// Loading is over, so whatever arrived is all there is: an index past
		// it would buffer for ever (a cycle whose frames failed part-way).
		if (this.frames.length > 0 && this.frameIndex >= this.frames.length) {
			this.frameIndex = this.frames.length - 1;
		}
		// Re-sample the selected point against the new cycle.
		if (this.point) void this.selectPoint(this.point.lat, this.point.lon);
	}

	/** One `/forecast` call per cycle, purely for the confidence scalar. */
	async #loadConfidence(manifest: Manifest, signal: AbortSignal): Promise<void> {
		const [rows, cols] = manifest.grid.shape;
		const [lon, lat] = gridToLonLat(manifest.grid, rows / 2, cols / 2);
		try {
			const centre = await fetchPointForecast(lat, lon, signal);
			if (!signal.aborted) this.confidence = centre?.confidence ?? null;
		} catch {
			if (!signal.aborted) this.confidence = null;
		}
	}

	// --- animation ---------------------------------------------------------

	#scheduleFrame(): void {
		if (this.#frameTimer) clearTimeout(this.#frameTimer);
		this.#frameTimer = setTimeout(
			() => {
				if (this.playing) {
					this.frameIndex = nextFrameIndex(this.frameIndex, this.frames.length);
				}
				this.#scheduleFrame();
			},
			frameDelayMs(this.frameIndex, this.frames.length, FRAME_MS, LAST_FRAME_HOLD_MS)
		);
	}

	/**
	 * Restart the frame timer so the frame just landed on gets a full interval
	 * instead of whatever was left of the previous one. Only meaningful while
	 * the loop is running — otherwise it would start an orphan timer that
	 * `stop()` never sees.
	 */
	#resyncFrameTimer(): void {
		if (this.#frameTimer) this.#scheduleFrame();
	}

	togglePlay(): void {
		this.playing = !this.playing;
		this.#resyncFrameTimer();
	}

	/**
	 * Move the loop to one frame. Seeking says *where*, not *whether*: a
	 * playing loop keeps playing from the frame you dropped it on, a paused
	 * one stays paused. (It used to pause on every scrub, which made the play
	 * button feel broken.)
	 */
	seek(index: number): void {
		if (this.frameCount === 0) return;
		this.frameIndex = clampIndex(index, this.frameCount);
		this.#resyncFrameTimer();
	}

	// --- point forecast ----------------------------------------------------

	/**
	 * Forecast for one point, in two arrivals.
	 *
	 * The decoded grids answer instantly and answer almost everything: the
	 * pixel, the deterministic rain series the headline is read from, the
	 * ETA, the intensity, the motion arrow. What they cannot answer is the
	 * probability the site actually serves — the gauge-trained model needs
	 * the cycle's flow field and its raw ensemble fractions, and neither is
	 * published as a grid. So `/forecast` is fetched for EVERY selected
	 * point, not only as a fallback, and its probabilities are merged in when
	 * they land.
	 *
	 * The two-arrival shape is deliberate. Making the whole panel wait on the
	 * network would cost the instant answer that is the reason the grids are
	 * decoded at all; drawing the sampled probabilities and then replacing
	 * them would show the reader a number the site does not serve. So the
	 * panel gets everything except the bars at once, and the bars when the
	 * server answers.
	 *
	 * A selection that is superseded — a second click, or the next cycle's
	 * re-sample — is dropped on arrival rather than racing the newer one.
	 */
	async selectPoint(lat: number, lon: number): Promise<void> {
		const token = ++this.#pointToken;
		const current = () => this.#pointToken === token;
		this.point = { lat, lon, status: 'loading', forecast: null, probabilitiesPending: true };
		const manifest = this.manifest;
		let sampled: PointForecast | null = null;
		if (manifest && this.#grids) {
			try {
				sampled = samplePoint(manifest, this.#grids, lat, lon);
				this.point = sampled
					? { lat, lon, status: 'ready', forecast: sampled, probabilitiesPending: true }
					: { lat, lon, status: 'off-coverage', forecast: null, probabilitiesPending: false };
				// Off coverage on the client is off coverage: the sampler read
				// the same grid the endpoint would have, and asking again
				// cannot change the answer.
				if (!sampled) return;
			} catch (err) {
				console.warn('client-side sampling failed, using /forecast alone', err);
				sampled = null;
			}
		}
		try {
			const served = await fetchPointForecast(lat, lon);
			if (!current()) return;
			if (served) {
				const forecast = sampled ? withServedProbabilities(sampled, served) : served;
				this.point = { lat, lon, status: 'ready', forecast, probabilitiesPending: false };
			} else if (sampled) {
				// The server says off-coverage where the grids answered. Trust
				// the sampled answer and leave the bars on the curve-calibrated
				// numbers rather than blanking a panel that has real content.
				this.point = { lat, lon, status: 'ready', forecast: sampled, probabilitiesPending: false };
			} else {
				this.point = { lat, lon, status: 'off-coverage', forecast: null, probabilitiesPending: false };
			}
		} catch (err) {
			if (!current()) return;
			console.warn('/forecast failed', err);
			// A failed fetch costs the served probability, not the panel: the
			// sampled numbers are the honest fallback and the source label
			// stays null, so nothing claims to be the model.
			this.point = sampled
				? { lat, lon, status: 'ready', forecast: sampled, probabilitiesPending: false }
				: { lat, lon, status: 'error', forecast: null, probabilitiesPending: false };
		}
	}

	clearPoint(): void {
		this.point = null;
	}
}

export const nowcast = new NowcastStore();
