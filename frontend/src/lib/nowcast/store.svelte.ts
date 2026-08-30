/**
 * The single source of truth for what the map page shows: the current cycle's
 * manifest, its reprojected overlay frames, the decoded product grids, the
 * animation position, and the forecast for whatever point the user picked.
 *
 * Polling is once a minute. The sidecar produces a cycle every ~5 min, and
 * artifact URLs are cycle-stamped and immutably cacheable, so a poll that
 * finds the same cycle costs one small conditional request and nothing else.
 */
import { browser } from '$app/environment';
import { loadOverlayFrames, overlayGeometry, type OverlayFrame } from '$lib/map/overlay';
import type { Corners } from '$lib/map/warp';
import { fetchPointForecast } from './forecast';
import { loadGrids } from './grids';
import { fetchManifest, isCalibrated, NoDataError, radarAgeMin, type Manifest } from './manifest';
import { gridToLonLat, samplePoint, type DecodedGrids, type PointForecast } from './sampler';

const POLL_MS = 60_000;
const CLOCK_MS = 15_000;
/** Time each frame is shown, and the extra pause on the last one. */
const FRAME_MS = 550;
const LAST_FRAME_HOLD_MS = 1400;
/** A cycle older than this means the pipeline has stopped. */
export const STALE_AFTER_MIN = 20;

export type Status = 'loading' | 'ready' | 'nodata' | 'error';

export interface PointState {
	lat: number;
	lon: number;
	status: 'loading' | 'ready' | 'off-coverage' | 'error';
	forecast: PointForecast | null;
}

class NowcastStore {
	manifest = $state<Manifest | null>(null);
	status = $state<Status>('loading');
	frames = $state<OverlayFrame[]>([]);
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

	get radarAgeMin(): number | null {
		return this.manifest ? Math.max(0, radarAgeMin(this.manifest, this.now)) : null;
	}

	get stale(): boolean {
		const age = this.radarAgeMin;
		return age !== null && age > STALE_AFTER_MIN;
	}

	get calibrated(): boolean {
		return this.manifest ? isCalibrated(this.manifest) : false;
	}

	get currentFrame(): OverlayFrame | null {
		return this.frames[this.frameIndex] ?? null;
	}

	/** Start polling and animating. Returns the matching teardown. */
	start(): () => void {
		if (!browser) return () => {};
		void this.refresh();
		this.#pollTimer = setInterval(() => void this.refresh(), POLL_MS);
		this.#clockTimer = setInterval(() => (this.now = Date.now()), CLOCK_MS);
		this.#scheduleFrame();
		return () => this.stop();
	}

	stop(): void {
		if (this.#pollTimer) clearInterval(this.#pollTimer);
		if (this.#clockTimer) clearInterval(this.#clockTimer);
		if (this.#frameTimer) clearTimeout(this.#frameTimer);
		this.#pollTimer = this.#clockTimer = null;
		this.#frameTimer = null;
		this.#cycleAbort?.abort();
	}

	async refresh(): Promise<void> {
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
				this.frames = [...collected];
				if (this.frameIndex >= collected.length) this.frameIndex = 0;
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
		if (this.frameIndex >= this.frames.length) this.frameIndex = 0;
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
		const last = this.frames.length > 0 && this.frameIndex === this.frames.length - 1;
		this.#frameTimer = setTimeout(
			() => {
				if (this.playing && this.frames.length > 0) {
					this.frameIndex = (this.frameIndex + 1) % this.frames.length;
				}
				this.#scheduleFrame();
			},
			last ? LAST_FRAME_HOLD_MS : FRAME_MS
		);
	}

	togglePlay(): void {
		this.playing = !this.playing;
	}

	seek(index: number): void {
		if (this.frames.length === 0) return;
		this.playing = false;
		this.frameIndex = Math.max(0, Math.min(this.frames.length - 1, index));
	}

	// --- point forecast ----------------------------------------------------

	/**
	 * Forecast for one point: sampled from the decoded grids when they are
	 * available, otherwise from the server. The two paths use the same
	 * conventions and produce the same shape.
	 */
	async selectPoint(lat: number, lon: number): Promise<void> {
		this.point = { lat, lon, status: 'loading', forecast: null };
		const manifest = this.manifest;
		if (manifest && this.#grids) {
			try {
				const forecast = samplePoint(manifest, this.#grids, lat, lon);
				this.point = forecast
					? { lat, lon, status: 'ready', forecast }
					: { lat, lon, status: 'off-coverage', forecast: null };
				return;
			} catch (err) {
				console.warn('client-side sampling failed, falling back to /forecast', err);
			}
		}
		try {
			const forecast = await fetchPointForecast(lat, lon);
			this.point = forecast
				? { lat, lon, status: 'ready', forecast }
				: { lat, lon, status: 'off-coverage', forecast: null };
		} catch (err) {
			console.warn('/forecast failed', err);
			this.point = { lat, lon, status: 'error', forecast: null };
		}
	}

	clearPoint(): void {
		this.point = null;
	}
}

export const nowcast = new NowcastStore();
