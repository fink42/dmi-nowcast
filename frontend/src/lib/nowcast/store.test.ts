/**
 * The nowcast store's lifecycle, driven with fakes: which bitmaps get closed
 * when a cycle load is cut short or superseded, that a stopped load is
 * reloaded on return, that the selected point is re-sampled as soon as the
 * grids arrive (not after the frame loop), and that a hidden tab neither
 * polls nor animates.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { OverlayFrame } from '$lib/map/overlay';
import type { Manifest } from './manifest';
import type { DecodedGrids, PointForecast } from './sampler';

vi.mock('$app/environment', () => ({ browser: true }));

const mocks = vi.hoisted(() => ({
	fetchManifest: vi.fn(),
	loadOverlayFrames: vi.fn(),
	loadGrids: vi.fn(),
	fetchPointForecast: vi.fn(),
	samplePoint: vi.fn()
}));

vi.mock('$lib/map/overlay', () => ({
	loadOverlayFrames: mocks.loadOverlayFrames,
	overlayGeometry: () => null
}));
vi.mock('./grids', () => ({ loadGrids: mocks.loadGrids }));
vi.mock('./forecast', async (importOriginal) => ({
	...(await importOriginal<typeof import('./forecast')>()),
	fetchPointForecast: mocks.fetchPointForecast
}));
vi.mock('./manifest', async (importOriginal) => ({
	...(await importOriginal<typeof import('./manifest')>()),
	fetchManifest: mocks.fetchManifest
}));
vi.mock('./sampler', async (importOriginal) => ({
	...(await importOriginal<typeof import('./sampler')>()),
	gridToLonLat: () => [10, 56],
	samplePoint: mocks.samplePoint
}));
vi.mock('./timeline', async (importOriginal) => ({
	...(await importOriginal<typeof import('./timeline')>()),
	buildTimeline: () =>
		Array.from({ length: 4 }, (_, i) => ({ leadMin: i * 10, kind: 'forecast', validTsUtc: '' }))
}));

const { NowcastStore } = await import('./store.svelte');

// --- fakes -------------------------------------------------------------------

interface FakeBitmap {
	name: string;
	close: ReturnType<typeof vi.fn>;
}

function bitmap(name: string): FakeBitmap {
	return { name, close: vi.fn() };
}

function frame(name: string, leadMin = 0): OverlayFrame & { bitmap: FakeBitmap } {
	return { leadMin, filename: `${name}.png`, bitmap: bitmap(name) as unknown as ImageBitmap & FakeBitmap };
}

function manifest(cycle: string): Manifest {
	return {
		schema_version: 2,
		cycle,
		radar_ts_utc: `${cycle}:00Z`,
		generated_at_utc: `${cycle}:30Z`,
		threshold_mm_h: 0.5,
		timestep_min: 10,
		frame_age_min: 10,
		n_members: 16,
		leads_min: [10],
		grid: { shape: [10, 10] } as unknown as Manifest['grid'],
		overlay_grid: null,
		calibration: null,
		artifacts: []
	};
}

interface Gate {
	promise: Promise<void>;
	open: () => void;
}

function gate(): Gate {
	let open!: () => void;
	const promise = new Promise<void>((resolve) => (open = resolve));
	return { promise, open };
}

/**
 * A frame source that yields `frames`, pausing before index `pauseAt` until
 * the gate opens, and failing like `fetch` does once its signal aborts.
 */
function source(frames: OverlayFrame[], pauseAt = -1, pause: Gate = gate()) {
	return async function* (_m: Manifest, signal?: AbortSignal): AsyncGenerator<OverlayFrame> {
		for (const [i, f] of frames.entries()) {
			if (i === pauseAt) {
				await new Promise<void>((resolve, reject) => {
					pause.promise.then(resolve);
					signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
				});
			}
			if (signal?.aborted) throw new DOMException('aborted', 'AbortError');
			yield f;
		}
	};
}

const settle = async () => {
	for (let i = 0; i < 10; i++) await Promise.resolve();
	await new Promise((resolve) => setTimeout(resolve, 0));
};

function forecast(cycle: string, etaMin: number): PointForecast {
	return {
		lat: 56,
		lon: 10,
		radarTsUtc: `${cycle}:00Z`,
		generatedAtUtc: `${cycle}:30Z`,
		perLead: [{ leadMin: 10, pRain: 0.5 }],
		etaMin,
		intensityMmH: 1,
		observedMmH: null,
		rainSeries: [],
		motion: null,
		confidence: null,
		calibrated: false,
		probabilitySource: null,
		source: 'client'
	};
}

beforeEach(() => {
	for (const fn of Object.values(mocks)) fn.mockReset();
	mocks.loadGrids.mockResolvedValue({} as DecodedGrids);
	mocks.fetchPointForecast.mockResolvedValue(null);
	mocks.samplePoint.mockReturnValue(null);
});

afterEach(() => {
	vi.useRealTimers();
	vi.unstubAllGlobals();
});

// --- bitmap ownership --------------------------------------------------------

describe('cycle loads and bitmap ownership', () => {
	it('closes the previous cycle once the new one has loaded, and nothing on screen', async () => {
		const store = new NowcastStore();
		const a = [frame('a0'), frame('a1')];
		const b = [frame('b0'), frame('b1')];
		mocks.fetchManifest.mockResolvedValueOnce(manifest('A')).mockResolvedValueOnce(manifest('B'));
		mocks.loadOverlayFrames.mockImplementationOnce(source(a)).mockImplementationOnce(source(b));

		await store.refresh();
		expect(store.frames).toEqual(a);
		await store.refresh();
		expect(store.frames).toEqual(b);
		for (const f of a) expect(f.bitmap.close).toHaveBeenCalledTimes(1);
		for (const f of b) expect(f.bitmap.close).not.toHaveBeenCalled();
	});

	it('stop() mid-load keeps the frames on screen open and reloads the same cycle next time', async () => {
		const store = new NowcastStore();
		const a = [frame('a0'), frame('a1')];
		const partial = [frame('b0'), frame('b1'), frame('b2')];
		const full = [frame('B0'), frame('B1'), frame('B2')];
		mocks.fetchManifest.mockResolvedValue(manifest('A'));
		mocks.loadOverlayFrames.mockImplementationOnce(source(a));
		await store.refresh();

		// Cycle B starts loading, gets one frame on screen, then the page goes.
		mocks.fetchManifest.mockResolvedValue(manifest('B'));
		mocks.loadOverlayFrames.mockImplementationOnce(source(partial, 1));
		const loading = store.refresh();
		await settle();
		expect(store.frames.map((f) => f.filename)).toEqual(['b0.png']);
		store.stop();
		await loading;

		// What is on screen is still drawable; what it replaced is released.
		expect(partial[0].bitmap.close).not.toHaveBeenCalled();
		for (const f of a) expect(f.bitmap.close).toHaveBeenCalledTimes(1);

		// Back on the page: the same cycle is loaded again, in full.
		mocks.loadOverlayFrames.mockImplementationOnce(source(full));
		await store.refresh();
		expect(mocks.loadOverlayFrames).toHaveBeenCalledTimes(3);
		expect(store.frames).toEqual(full);
		expect(partial[0].bitmap.close).toHaveBeenCalledTimes(1);
		for (const f of full) expect(f.bitmap.close).not.toHaveBeenCalled();
		// Never touched: b1, b2 were not produced before the abort.
		expect(partial[1].bitmap.close).not.toHaveBeenCalled();
	});

	it('a superseded load releases its own frames and its previous set exactly once', async () => {
		const store = new NowcastStore();
		const a = [frame('a0'), frame('a1')];
		const b = [frame('b0'), frame('b1')];
		const c = [frame('c0'), frame('c1')];
		mocks.fetchManifest.mockResolvedValueOnce(manifest('A'));
		mocks.loadOverlayFrames.mockImplementationOnce(source(a));
		await store.refresh();

		// B shows one frame and stalls; C arrives and supersedes it.
		const bGate = gate();
		mocks.fetchManifest.mockResolvedValueOnce(manifest('B'));
		mocks.loadOverlayFrames.mockImplementationOnce(source(b, 1, bGate));
		const loadingB = store.refresh();
		await settle();
		expect(store.frames.map((f) => f.filename)).toEqual(['b0.png']);

		const cGate = gate();
		mocks.fetchManifest.mockResolvedValueOnce(manifest('C'));
		mocks.loadOverlayFrames.mockImplementationOnce(source(c, 0, cGate));
		const loadingC = store.refresh();
		await loadingB;
		// B was cut short while its frame was still the one on screen: it
		// stays open for C to release, but A (which B had replaced) is gone.
		expect(b[0].bitmap.close).not.toHaveBeenCalled();
		for (const f of a) expect(f.bitmap.close).toHaveBeenCalledTimes(1);

		cGate.open();
		await loadingC;
		expect(store.frames).toEqual(c);
		expect(b[0].bitmap.close).toHaveBeenCalledTimes(1);
		for (const f of a) expect(f.bitmap.close).toHaveBeenCalledTimes(1);
		for (const f of c) expect(f.bitmap.close).not.toHaveBeenCalled();
	});

	it('a load aborted before its first frame leaves the old frames on screen, open', async () => {
		const store = new NowcastStore();
		const a = [frame('a0')];
		mocks.fetchManifest.mockResolvedValueOnce(manifest('A'));
		mocks.loadOverlayFrames.mockImplementationOnce(source(a));
		await store.refresh();

		mocks.fetchManifest.mockResolvedValueOnce(manifest('B'));
		mocks.loadOverlayFrames.mockImplementationOnce(source([frame('b0')], 0));
		const loading = store.refresh();
		await settle();
		store.stop();
		await loading;
		expect(store.frames).toEqual(a);
		expect(a[0].bitmap.close).not.toHaveBeenCalled();
	});
});

// --- point re-sampling -------------------------------------------------------

describe('selected point across a cycle change', () => {
	it('is re-sampled from the new cycle as soon as the grids land, before the frames finish', async () => {
		const store = new NowcastStore();
		const gridsA = { tag: 'A' } as unknown as DecodedGrids;
		const gridsB = { tag: 'B' } as unknown as DecodedGrids;
		mocks.samplePoint.mockImplementation((m: Manifest, g: { tag: string }) =>
			forecast(m.cycle, g.tag === 'A' ? 20 : 12)
		);
		mocks.fetchManifest.mockResolvedValueOnce(manifest('A'));
		mocks.loadGrids.mockResolvedValueOnce(gridsA);
		mocks.loadOverlayFrames.mockImplementationOnce(source([frame('a0')]));
		await store.refresh();
		await store.selectPoint(56, 10);
		expect(store.point?.forecast?.generatedAtUtc).toBe('A:30Z');

		// Cycle B: the manifest is adopted, the grids resolve later, and the
		// frames later still.
		let resolveGrids!: (g: DecodedGrids) => void;
		mocks.loadGrids.mockReturnValueOnce(new Promise((r) => (resolveGrids = r)));
		const frames = gate();
		mocks.fetchManifest.mockResolvedValueOnce(manifest('B'));
		mocks.loadOverlayFrames.mockImplementationOnce(source([frame('b0'), frame('b1')], 0, frames));
		const loading = store.refresh();
		await settle();
		expect(store.manifest?.cycle).toBe('B');

		// A click in the gap samples the OLD grids with the OLD manifest, and
		// the forecast says which cycle it belongs to.
		await store.selectPoint(56, 10);
		expect(mocks.samplePoint).toHaveBeenLastCalledWith(
			expect.objectContaining({ cycle: 'A' }),
			gridsA,
			56,
			10
		);
		expect(store.point?.forecast?.generatedAtUtc).toBe('A:30Z');

		resolveGrids(gridsB);
		await settle();
		// Frames are still pending, and the point is already on cycle B.
		expect(store.frames.map((f) => f.filename)).toEqual(['a0.png']);
		expect(store.point?.forecast?.generatedAtUtc).toBe('B:30Z');
		expect(store.point?.forecast?.etaMin).toBe(12);

		frames.open();
		await loading;
	});
});

// --- hidden tab --------------------------------------------------------------

describe('hidden tab', () => {
	function fakeDocument(state: 'visible' | 'hidden') {
		const listeners = new Set<() => void>();
		const doc = {
			visibilityState: state,
			addEventListener: (_: string, fn: () => void) => listeners.add(fn),
			removeEventListener: (_: string, fn: () => void) => listeners.delete(fn),
			fire(next: 'visible' | 'hidden') {
				doc.visibilityState = next;
				for (const fn of listeners) fn();
			}
		};
		vi.stubGlobal('document', doc);
		return doc;
	}

	it('neither polls nor advances frames while hidden, and resumes when visible', async () => {
		vi.useFakeTimers();
		const doc = fakeDocument('visible');
		const store = new NowcastStore();
		mocks.fetchManifest.mockResolvedValue(manifest('A'));
		mocks.loadOverlayFrames.mockImplementation(source([frame('a0'), frame('a1'), frame('a2')]));
		const stop = store.start();
		await vi.advanceTimersByTimeAsync(0);
		expect(store.frames).toHaveLength(3);
		expect(mocks.fetchManifest).toHaveBeenCalledTimes(1);

		// Visible: the loop moves.
		await vi.advanceTimersByTimeAsync(600);
		expect(store.frameIndex).toBe(1);

		doc.fire('hidden');
		const index = store.frameIndex;
		await vi.advanceTimersByTimeAsync(5 * 60_000);
		expect(store.frameIndex).toBe(index);
		expect(mocks.fetchManifest).toHaveBeenCalledTimes(1);

		// Back: one immediate refresh, and the loop runs again.
		doc.fire('visible');
		await vi.advanceTimersByTimeAsync(0);
		expect(mocks.fetchManifest).toHaveBeenCalledTimes(2);
		await vi.advanceTimersByTimeAsync(600);
		expect(store.frameIndex).not.toBe(index);
		stop();
	});

	it('does not start the loop when started in a hidden tab', async () => {
		vi.useFakeTimers();
		const doc = fakeDocument('hidden');
		const store = new NowcastStore();
		mocks.fetchManifest.mockResolvedValue(manifest('A'));
		mocks.loadOverlayFrames.mockImplementation(source([frame('a0'), frame('a1')]));
		const stop = store.start();
		await vi.advanceTimersByTimeAsync(3_000);
		expect(store.frameIndex).toBe(0);
		doc.fire('visible');
		await vi.advanceTimersByTimeAsync(600);
		expect(store.frameIndex).toBe(1);
		stop();
	});
});
