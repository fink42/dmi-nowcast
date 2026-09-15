/**
 * The review page's single source of truth: which bundle is open, which
 * event is selected, where the cursor is, what the reviewer has typed, and
 * which bitmaps are in hand.
 *
 * This is the only runes file in `lib/review`, and deliberately the only
 * one with no test beside it — every number it shows comes from a pure
 * module that has one. What lives here is state, timers, the network and
 * the canvas: the things a test would have to fake, and the things whose
 * bugs are visible on screen rather than invisible in a figure.
 *
 * The cursor is the spine. It is an absolute instant in UTC milliseconds,
 * not a frame index, because the whole tool is about what was known *when*:
 * the frame, the estimate, the arm state, the gauge slot and the neighbour
 * dots are all read from the same instant by the modules above, so they
 * cannot disagree with each other. Playback moves the cursor by stepping
 * frames; dragging the track moves it directly.
 */
import { browser } from '$app/environment';
import { decodeGray8Png, type Gray8Image } from '$lib/nowcast/png';
import { buildMesh, targetCorners, warpImage, warpTarget, type Corners } from '$lib/map/warp';
import {
	deleteAnnotation,
	exportBundle,
	fetchAnnotations,
	health,
	saveAnnotation,
	type ApiFailure,
	type ExportFormat,
	type ReviewHealth
} from './api';
import { armStateAt, etaCountdownAt, frameForCursor, latestDecisionAt } from './estimate';
import {
	applyFilters,
	facetCounts,
	FilterState,
	nextUnreviewed,
	reviewProgress
} from './filter';
import { frameUrl, lruEvict, prefetchPlan, presentStamps } from './frames';
import {
	annotationsByEvent,
	fetchEvent,
	fetchIndex,
	fetchManifest,
	fetchVocabulary
} from './load';
import { discPolygon, neighbourFeatures, upwindVector } from './geometry';
import type {
	Annotation,
	AnnotationDraft,
	EventDetail,
	FrameMode,
	IndexRow,
	Manifest,
	Vocabulary
} from './schema';
import { BUILTIN_VOCABULARY, isComplete, isDirty, validateAnnotation } from './tags';
import {
	clampIndex,
	cursorForIndex,
	frameDelayMs,
	frameTicks,
	markers,
	msOf,
	nearestFrameIndex,
	nextFrameIndex,
	positionAt,
	trackBounds
} from './timeline';
import { sampleDisc, sampleObservedAt, type DiscSample } from './sample';
import { gaugeStateAt, neighbourStatesAt, radarStateAt } from './truth';

/** Frame dwell, matching the public loop so both read at the same speed. */
const FRAME_MS = 550;
const LAST_FRAME_HOLD_MS = 1400;

export type BundleStatus = 'idle' | 'loading' | 'ready' | 'error';
export type DetailStatus = 'idle' | 'loading' | 'ready' | 'error';
export type SaveStatus = 'idle' | 'saving' | 'saved' | 'error';

/**
 * A 2D surface to warp one frame into — `OffscreenCanvas` where there is
 * one, a detached `<canvas>` otherwise, which is the same fallback
 * `map/overlay.ts` carries. Null when the browser will not give a context,
 * which costs the picture and nothing else.
 */
function createSurface(
	width: number,
	height: number
): { ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D; finish: () => Promise<ImageBitmap> } | null {
	if (typeof OffscreenCanvas !== 'undefined') {
		const canvas = new OffscreenCanvas(width, height);
		const ctx = canvas.getContext('2d');
		// transferToImageBitmap hands over the pixels and resets the canvas,
		// which is exactly the per-frame lifecycle wanted here.
		return ctx === null ? null : { ctx, finish: async () => canvas.transferToImageBitmap() };
	}
	const canvas = document.createElement('canvas');
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext('2d');
	return ctx === null ? null : { ctx, finish: () => createImageBitmap(canvas) };
}

/** An empty judgement — what an unjudged event's form starts from. */
const emptyDraft = (vocabVersion: number): AnnotationDraft => ({
	verdict: null,
	tags: [],
	confidence: null,
	needs_second_look: false,
	note: '',
	cursor_utc: null,
	vocab_version: vocabVersion
});

class ReviewStore {
	// --- the bundle ---------------------------------------------------------
	status = $state<BundleStatus>('idle');
	error = $state<string | null>(null);
	manifest = $state<Manifest | null>(null);
	/** The bundle's own vocabulary, or the client's built-in fallback. */
	vocabulary = $state<Vocabulary>(BUILTIN_VOCABULARY);
	/** True while the page is tagging under the compiled-in vocabulary. */
	vocabularyIsFallback = $state(false);
	rows = $state<IndexRow[]>([]);
	serverHealth = $state<ReviewHealth | null>(null);

	// --- the list -----------------------------------------------------------
	filter = $state(new FilterState());
	annotations = $state<Map<string, Annotation>>(new Map());

	// --- the selected event -------------------------------------------------
	selectedId = $state<string | null>(null);
	detail = $state<EventDetail | null>(null);
	detailStatus = $state<DetailStatus>('idle');

	// --- the cursor ---------------------------------------------------------
	/** Absolute UTC milliseconds. Null before an event is open. */
	cursorMs = $state<number | null>(null);
	mode = $state<FrameMode>('truth');
	playing = $state(false);

	// --- the judgement ------------------------------------------------------
	draft = $state<AnnotationDraft>(emptyDraft(BUILTIN_VOCABULARY.vocab_version));
	saveStatus = $state<SaveStatus>('idle');
	saveError = $state<ApiFailure | null>(null);

	/**
	 * Decoded, reprojected frames by stamp. A plain Map behind a version
	 * counter rather than `$state`: `ImageBitmap`s must not be wrapped in a
	 * proxy (the canvas wants the real object), and the only reactive fact
	 * about this cache is "something changed".
	 */
	#bitmaps = new Map<string, ImageBitmap>();
	bitmapVersion = $state(0);
	/**
	 * The grayscale observation grids, decoded byte-exactly by
	 * `nowcast/png.ts`. Kept apart from the bitmaps because they are a
	 * different thing: the overlay is a picture and this is the NUMBER. The
	 * colours are advisory — light rain is deliberately faded — so every
	 * mm/h the read-out quotes comes from here.
	 */
	#observed = new Map<string, Gray8Image>();
	observedVersion = $state(0);
	#loading = new Set<string>();
	#frameTimer: ReturnType<typeof setTimeout> | null = null;
	#detailAbort: AbortController | null = null;
	#frameAbort: AbortController | null = null;
	/** Bumped per `selectEvent`, so a superseded detail is dropped on arrival. */
	#detailToken = 0;

	// --- derived: the list --------------------------------------------------

	get visibleRows(): IndexRow[] {
		return applyFilters(this.rows, this.filter, this.annotations);
	}

	get facets() {
		return facetCounts(this.rows, this.filter, this.annotations);
	}

	get progress() {
		return reviewProgress(this.rows, this.annotations);
	}

	get selectedRow(): IndexRow | null {
		return this.rows.find((row) => row.event_id === this.selectedId) ?? null;
	}

	// --- derived: the track -------------------------------------------------

	get bounds() {
		return this.detail === null ? null : trackBounds(this.detail);
	}

	get ticks() {
		return this.detail === null ? [] : frameTicks(this.detail);
	}

	get trackMarkers() {
		return this.detail === null ? { points: [], bands: [] } : markers(this.detail);
	}

	/** Where the cursor sits on the track, 0 … 1, for the playhead. */
	get cursorPosition(): number | null {
		const bounds = this.bounds;
		return bounds === null ? null : positionAt(bounds, this.cursorMs);
	}

	get frameIndex(): number {
		if (this.detail === null || this.cursorMs === null) return -1;
		return nearestFrameIndex(this.detail.frames, this.cursorMs);
	}

	// --- derived: what was known at the cursor ------------------------------

	get cursorFrame() {
		if (this.detail === null || this.cursorMs === null) return null;
		return frameForCursor(this.detail.frames, this.detail.decisions, this.cursorMs, this.mode);
	}

	get decision() {
		if (this.detail === null || this.cursorMs === null) return null;
		return latestDecisionAt(this.detail.decisions, this.cursorMs);
	}

	get etaMin(): number | null {
		return this.cursorMs === null ? null : etaCountdownAt(this.decision, this.cursorMs);
	}

	get armState() {
		if (this.detail === null || this.cursorMs === null) return null;
		return armStateAt(
			this.detail.decisions,
			this.detail.prologue,
			this.cursorMs,
			this.manifest?.rule?.rearm_after_min
		);
	}

	get gaugeState() {
		if (this.detail === null || this.cursorMs === null) return null;
		return gaugeStateAt(this.detail.gauge, this.cursorMs);
	}

	get radarState() {
		if (this.detail === null || this.cursorMs === null) return null;
		return radarStateAt(this.detail.radar_disc, this.cursorMs);
	}

	get neighbourStates() {
		if (this.detail === null || this.cursorMs === null) return [];
		return neighbourStatesAt(this.detail.neighbours, this.cursorMs);
	}

	/**
	 * Station + neighbours as one GeoJSON source, coloured by the cursor and
	 * labelled against the flow.
	 *
	 * Fed from `neighbours.stations[]`, not from `station.neighbours[]`: the
	 * builder inlines the coordinates on the former, and the latter carries
	 * ids and distances only. The upwind bearing rides along so each dot
	 * knows whether it is upwind or downwind of the cell — which is the
	 * difference between a shower that went round this gauge and one that
	 * had already crossed it.
	 */
	get stationFeatures() {
		return neighbourFeatures(
			this.detail?.station ?? null,
			this.detail?.neighbours?.stations ?? [],
			this.neighbourStates,
			this.upwind?.bearingFromDeg ?? null
		);
	}

	/** The 1 km verdict disc, so the reviewer sees what was sampled. */
	get discFeature() {
		const station = this.detail?.station;
		const radius = this.manifest?.truth?.radar_disc_radius_m;
		if (!station || radius === undefined) return null;
		return discPolygon(station.lat, station.lon, radius);
	}

	/** The upwind arrow's input, from the estimate standing at the cursor. */
	get upwind() {
		return upwindVector(this.decision?.features ?? null);
	}

	// --- derived: the picture ----------------------------------------------

	/** Mercator corners for the map's image source, from the bundle's grid. */
	get corners(): Corners | null {
		const grid = this.manifest?.grid ?? null;
		return grid === null ? null : targetCorners(warpTarget(grid));
	}

	get currentBitmap(): ImageBitmap | null {
		// Read the version so the getter re-runs when a frame lands.
		void this.bitmapVersion;
		const stamp = this.cursorFrame?.frame?.stamp;
		return stamp === undefined ? null : (this.#bitmaps.get(stamp) ?? null);
	}

	/** True when the frame the cursor is on has not been decoded yet. */
	get buffering(): boolean {
		void this.bitmapVersion;
		const frame = this.cursorFrame?.frame;
		if (frame === null || frame === undefined || frame.present === false) return false;
		return !this.#bitmaps.has(frame.stamp);
	}

	/**
	 * The observation grid behind the frame on screen, or null while it is
	 * still decoding. Null is "we do not know yet", and the read-out says so
	 * rather than showing a dash that reads as dry.
	 */
	get observedGrid(): Gray8Image | null {
		void this.observedVersion;
		const stamp = this.cursorFrame?.frame?.stamp;
		return stamp === undefined ? null : (this.#observed.get(stamp) ?? null);
	}

	/**
	 * What the radar measured at one point, on the frame the cursor is on.
	 *
	 * Two numbers, deliberately: the pixel under the pointer, and the p90
	 * over the verdict DISC — which is the statistic the service actually
	 * acted on. A reviewer comparing a warning against whatever single pixel
	 * sits under the station marker is comparing against a different number
	 * from the one that fired it.
	 *
	 * Null throughout means "not measured": off the grid, a nodata pixel, a
	 * frame still decoding, or a bundle with no quantisation. None of them
	 * is zero.
	 */
	sampleAt(lat: number, lon: number): {
		stamp: string | null;
		radarTsUtc: string | null;
		pixelMmH: number | null;
		disc: DiscSample | null;
	} {
		const frame = this.cursorFrame?.frame ?? null;
		const grid = this.observedGrid;
		const manifest = this.manifest;
		const radius = manifest?.truth?.radar_disc_radius_m ?? null;
		if (grid === null || manifest === null) {
			return {
				stamp: frame?.stamp ?? null,
				radarTsUtc: frame?.radar_ts_utc ?? null,
				pixelMmH: null,
				disc: null
			};
		}
		return {
			stamp: frame?.stamp ?? null,
			radarTsUtc: frame?.radar_ts_utc ?? null,
			pixelMmH: sampleObservedAt(grid, manifest, lat, lon),
			disc: radius === null ? null : sampleDisc(grid, manifest, lat, lon, radius)
		};
	}

	// --- derived: the judgement --------------------------------------------

	get savedAnnotation(): Annotation | null {
		return this.selectedId === null ? null : (this.annotations.get(this.selectedId) ?? null);
	}

	get dirty(): boolean {
		return isDirty(this.draft, this.savedAnnotation);
	}

	get draftComplete(): boolean {
		return isComplete(this.draft);
	}

	get draftProblems() {
		return validateAnnotation(this.draft, this.vocabulary);
	}

	// --- loading ------------------------------------------------------------

	/**
	 * Open the bundle. The four documents are fetched together because the
	 * page cannot show anything useful without the index and the manifest,
	 * and the annotations decide what the list looks like.
	 *
	 * A failing annotation server is NOT a failed load: the bundle is static
	 * and reviewable, and telling the reviewer "the server is down, your
	 * existing judgements are not shown" is better than a blank page. The
	 * save path reports the same failure again if they try to write.
	 */
	async load(): Promise<void> {
		if (!browser) return;
		this.status = 'loading';
		this.error = null;
		try {
			const [manifest, index] = await Promise.all([fetchManifest(), fetchIndex()]);
			this.manifest = manifest;
			this.rows = index.events;
			try {
				this.vocabulary = await fetchVocabulary();
				this.vocabularyIsFallback = false;
			} catch {
				// A bundle older than the vocabulary, or a tags.json that did
				// not parse. Tagging still works; the page says which copy.
				this.vocabulary = BUILTIN_VOCABULARY;
				this.vocabularyIsFallback = true;
			}
			this.draft = emptyDraft(this.vocabulary.vocab_version);
			await this.refreshAnnotations();
			const up = await health();
			this.serverHealth = up.ok ? up.value : null;
			this.status = 'ready';
		} catch (err) {
			this.status = 'error';
			this.error = String(err);
		}
	}

	async refreshAnnotations(): Promise<void> {
		const result = await fetchAnnotations();
		if (!result.ok) {
			this.saveError = result.error;
			return;
		}
		this.annotations = annotationsByEvent(result.value);
	}

	/**
	 * Open one event: its detail document, its cursor, and its judgement.
	 *
	 * The cursor starts on the saved `cursor_utc` when there is one — a
	 * reviewer returning to an event lands where they left off, which is
	 * usually the instant that decided it — and on the anchor otherwise.
	 */
	async selectEvent(eventId: string): Promise<void> {
		if (!browser) return;
		const row = this.rows.find((candidate) => candidate.event_id === eventId);
		if (row === undefined) return;
		const token = ++this.#detailToken;
		this.#detailAbort?.abort();
		this.#detailAbort = new AbortController();
		this.pause();
		this.selectedId = eventId;
		this.detail = null;
		this.detailStatus = 'loading';
		this.#resetDraft(eventId);

		try {
			const detail = await fetchEvent(row.detail, this.#detailAbort.signal);
			if (token !== this.#detailToken) return;
			this.detail = detail;
			this.detailStatus = 'ready';
			const saved = msOf(this.annotations.get(eventId)?.cursor_utc ?? null);
			this.cursorMs = saved ?? msOf(detail.window.anchor_utc) ?? trackBounds(detail)?.fromMs ?? null;
			this.#evictForeignFrames();
			void this.#prefetch();
		} catch (err) {
			if (token !== this.#detailToken) return;
			this.detailStatus = 'error';
			this.error = String(err);
		}
	}

	/** Open whichever visible event is still unjudged, after this one. */
	selectNextUnreviewed(): void {
		const next = nextUnreviewed(this.visibleRows, this.annotations, this.selectedId);
		if (next !== null) void this.selectEvent(next.event_id);
	}

	// --- the cursor ---------------------------------------------------------

	setCursor(ms: number): void {
		const bounds = this.bounds;
		if (bounds === null || !Number.isFinite(ms)) return;
		this.cursorMs = Math.max(bounds.fromMs, Math.min(bounds.toMs, Math.round(ms)));
		void this.#prefetch();
	}

	/** Move to one frame of the loop, by index. */
	seekToFrame(index: number): void {
		if (this.detail === null) return;
		const bounded = clampIndex(index, this.detail.frames.length);
		const ms = cursorForIndex(this.detail.frames, bounded);
		if (ms !== null) this.setCursor(ms);
	}

	/** One frame forward or back, for the arrow keys. */
	step(delta: number): void {
		if (this.detail === null) return;
		const index = this.frameIndex;
		if (index < 0) return;
		this.seekToFrame(index + delta);
	}

	setMode(mode: FrameMode): void {
		this.mode = mode;
	}

	play(): void {
		if (this.playing || this.detail === null) return;
		this.playing = true;
		this.#scheduleFrame();
	}

	pause(): void {
		this.playing = false;
		if (this.#frameTimer !== null) {
			clearTimeout(this.#frameTimer);
			this.#frameTimer = null;
		}
	}

	toggle(): void {
		if (this.playing) this.pause();
		else this.play();
	}

	#scheduleFrame(): void {
		if (this.#frameTimer !== null) clearTimeout(this.#frameTimer);
		const count = this.detail?.frames.length ?? 0;
		const delay = frameDelayMs(Math.max(0, this.frameIndex), count, FRAME_MS, LAST_FRAME_HOLD_MS);
		this.#frameTimer = setTimeout(() => {
			if (!this.playing) return;
			// The public loop's own advance rule, so the two behave alike.
			this.seekToFrame(nextFrameIndex(Math.max(0, this.frameIndex), count));
			this.#scheduleFrame();
		}, delay);
	}

	// --- the judgement ------------------------------------------------------

	#resetDraft(eventId: string): void {
		const saved = this.annotations.get(eventId) ?? null;
		this.saveStatus = 'idle';
		this.saveError = null;
		this.draft =
			saved === null
				? emptyDraft(this.vocabulary.vocab_version)
				: {
						verdict: saved.verdict,
						tags: [...saved.tags],
						confidence: saved.confidence,
						needs_second_look: saved.needs_second_look,
						note: saved.note,
						cursor_utc: saved.cursor_utc,
						// The vocabulary the reviewer is looking at NOW, not the
						// one the row was first saved under: an edit is a new
						// judgement under the current codes.
						vocab_version: this.vocabulary.vocab_version
					};
	}

	updateDraft(patch: Partial<AnnotationDraft>): void {
		this.draft = { ...this.draft, ...patch };
		if (this.saveStatus === 'saved') this.saveStatus = 'idle';
	}

	toggleTag(code: string): void {
		const tags = this.draft.tags.includes(code)
			? this.draft.tags.filter((tag) => tag !== code)
			: [...this.draft.tags, code];
		this.updateDraft({ tags });
	}

	/**
	 * Save the judgement, stamping it with the cursor the reviewer was
	 * looking at. That stamp is not decoration: it is how a later reader
	 * finds the instant the judgement was made from, and it is what makes a
	 * disputed tag checkable.
	 */
	async save(): Promise<void> {
		if (this.selectedId === null) return;
		const cursor = this.cursorMs === null ? null : new Date(this.cursorMs).toISOString();
		const draft: AnnotationDraft = { ...this.draft, cursor_utc: cursor };
		this.draft = draft;
		this.saveStatus = 'saving';
		this.saveError = null;
		const result = await saveAnnotation(this.selectedId, draft, {
			ifMatch: this.savedAnnotation?.revision ?? null
		});
		if (!result.ok) {
			this.saveStatus = 'error';
			this.saveError = result.error;
			return;
		}
		const next = new Map(this.annotations);
		next.set(result.value.event_id, result.value);
		this.annotations = next;
		this.saveStatus = 'saved';
	}

	async discard(): Promise<void> {
		if (this.selectedId === null) return;
		const result = await deleteAnnotation(this.selectedId, {
			ifMatch: this.savedAnnotation?.revision ?? null
		});
		if (!result.ok) {
			this.saveStatus = 'error';
			this.saveError = result.error;
			return;
		}
		const next = new Map(this.annotations);
		next.delete(this.selectedId);
		this.annotations = next;
		this.#resetDraft(this.selectedId);
	}

	async exportRows(format: ExportFormat = 'parquet'): Promise<string | null> {
		const result = await exportBundle(format);
		if (result.ok) return result.value.path;
		this.saveError = result.error;
		return null;
	}

	// --- frames -------------------------------------------------------------

	/**
	 * Fetch, decode and reproject the frames around the cursor.
	 *
	 * Reprojection is not optional: the composite is polar stereographic and
	 * the naive four-corner placement misplaces rain by ~15 km over Denmark,
	 * which at this zoom is the difference between a cell over the station
	 * and a cell over the next town. `map/warp.ts` does it for the public
	 * map and does it here.
	 */
	async #prefetch(): Promise<void> {
		const detail = this.detail;
		const grid = this.manifest?.grid ?? null;
		if (!browser || detail === null || grid === null) return;
		// Centred on the nearest frame that exists, by TIME. Falling back to
		// index 0 when the cursor sits in a gap would fetch the start of the
		// window while the reviewer is looking at the end of it.
		const present = detail.frames.filter((frame) => frame.present !== false);
		const stamps = presentStamps(present);
		const index = this.cursorMs === null ? 0 : nearestFrameIndex(present, this.cursorMs);
		const plan = prefetchPlan(stamps, index < 0 ? 0 : index);
		const target = warpTarget(grid);
		const mesh = buildMesh(grid, target);

		for (const stamp of plan) {
			if (this.#bitmaps.has(stamp) || this.#loading.has(stamp)) continue;
			this.#loading.add(stamp);
			try {
				this.#frameAbort ??= new AbortController();
				const res = await fetch(frameUrl(stamp, 'overlay'), {
					signal: this.#frameAbort.signal
				});
				if (!res.ok) continue;
				const source = await createImageBitmap(await res.blob());
				try {
					const surface = createSurface(target.width, target.height);
					if (surface === null) continue;
					warpImage(surface.ctx, source, mesh);
					this.#bitmaps.set(stamp, await surface.finish());
					this.bitmapVersion += 1;
				} finally {
					source.close();
				}
			} catch {
				// A frame that will not load costs one picture. The track still
				// shows it, and the read-out still has its numbers.
			} finally {
				this.#loading.delete(stamp);
			}
		}
		await this.#loadObserved(this.cursorFrame?.frame?.stamp ?? null);
		this.#evict();
	}

	/**
	 * The grayscale grid for one stamp — the cursor's own frame only, not
	 * the whole prefetch plan. It is ~6 KB against the overlay's ~45 KB, but
	 * the read-out only ever asks about the frame on screen, and bandwidth
	 * spent here is bandwidth not spent keeping playback smooth.
	 */
	async #loadObserved(stamp: string | null): Promise<void> {
		if (!browser || stamp === null) return;
		const key = `observed:${stamp}`;
		if (this.#observed.has(stamp) || this.#loading.has(key)) return;
		this.#loading.add(key);
		try {
			this.#frameAbort ??= new AbortController();
			const res = await fetch(frameUrl(stamp, 'observed'), { signal: this.#frameAbort.signal });
			if (!res.ok) return;
			const bytes = new Uint8Array(await res.arrayBuffer());
			this.#observed.set(stamp, await decodeGray8Png(bytes));
			this.observedVersion += 1;
		} catch {
			// A grid that will not load or decode costs the read-out's number,
			// which then says "not measured" — true, and visibly different
			// from a measured zero.
		} finally {
			this.#loading.delete(key);
		}
	}

	#evict(): void {
		for (const stamp of lruEvict(this.#bitmaps)) {
			this.#bitmaps.get(stamp)?.close();
			this.#bitmaps.delete(stamp);
		}
		// The grids are plain typed arrays with nothing to close, and one is
		// a fiftieth of a bitmap, so they ride on the same capacity.
		for (const stamp of lruEvict(this.#observed)) this.#observed.delete(stamp);
	}

	/** Drop every frame that does not belong to the open event. */
	#evictForeignFrames(): void {
		const keep = new Set(this.detail?.frames.map((frame) => frame.stamp) ?? []);
		for (const [stamp, bitmap] of this.#bitmaps) {
			if (keep.has(stamp)) continue;
			bitmap.close();
			this.#bitmaps.delete(stamp);
		}
		for (const stamp of this.#observed.keys()) {
			if (!keep.has(stamp)) this.#observed.delete(stamp);
		}
		this.bitmapVersion += 1;
		this.observedVersion += 1;
	}

	/** Stop every timer and release every bitmap. The page's teardown. */
	stop(): void {
		this.pause();
		this.#detailAbort?.abort();
		this.#frameAbort?.abort();
		this.#frameAbort = null;
		for (const bitmap of this.#bitmaps.values()) bitmap.close();
		this.#bitmaps.clear();
		this.#observed.clear();
		this.bitmapVersion += 1;
		this.observedVersion += 1;
	}
}

export const review = new ReviewStore();
