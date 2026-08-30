/**
 * Turning a cycle's 500 m overlay PNGs into map-ready frames.
 *
 * Each frame is fetched, decoded, resampled into Mercator (see ./warp.ts) and
 * kept as one `ImageBitmap`. That costs a single warp per frame per cycle and
 * leaves the animation itself free: playing the loop is just handing MapLibre
 * a different bitmap. Frames are yielded as they become ready so the "now"
 * frame is on screen while the rest are still downloading.
 */
import { artifactUrl, overlayFrames, type Manifest } from '$lib/nowcast/manifest';
import { buildMesh, targetCorners, warpImage, warpTarget, type Corners } from './warp';

export interface OverlayFrame {
	leadMin: number;
	filename: string;
	bitmap: ImageBitmap;
}

export interface OverlayGeometry {
	corners: Corners;
	width: number;
	height: number;
}

interface Canvas2D {
	canvas: OffscreenCanvas | HTMLCanvasElement;
	ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
	finish: () => Promise<ImageBitmap>;
}

function createCanvas(width: number, height: number): Canvas2D {
	if (typeof OffscreenCanvas !== 'undefined') {
		const canvas = new OffscreenCanvas(width, height);
		const ctx = canvas.getContext('2d', { willReadFrequently: false });
		if (!ctx) throw new Error('no 2d context');
		// transferToImageBitmap hands over the pixels and resets the canvas,
		// which is exactly the per-frame lifecycle we want.
		return { canvas, ctx, finish: async () => canvas.transferToImageBitmap() };
	}
	const canvas = document.createElement('canvas');
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext('2d');
	if (!ctx) throw new Error('no 2d context');
	return { canvas, ctx, finish: () => createImageBitmap(canvas) };
}

/** Geometry of the warped frames — what the image source's corners must be. */
export function overlayGeometry(manifest: Manifest): OverlayGeometry | null {
	const grid = manifest.overlay_grid;
	if (!grid) return null;
	const target = warpTarget(grid);
	return { corners: targetCorners(target), width: target.width, height: target.height };
}

/**
 * Load and reproject every overlay frame of a cycle, yielding each as soon as
 * it is ready. Aborting the signal stops between frames.
 */
export async function* loadOverlayFrames(
	manifest: Manifest,
	signal?: AbortSignal
): AsyncGenerator<OverlayFrame> {
	const grid = manifest.overlay_grid;
	if (!grid) return;
	const entries = overlayFrames(manifest);
	if (entries.length === 0) return;

	const target = warpTarget(grid);
	const mesh = buildMesh(grid, target);
	const surface = createCanvas(target.width, target.height);

	for (const entry of entries) {
		if (signal?.aborted) return;
		const res = await fetch(artifactUrl(entry.filename), { signal });
		if (!res.ok) throw new Error(`${entry.filename}: HTTP ${res.status}`);
		const source = await createImageBitmap(await res.blob());
		try {
			warpImage(surface.ctx, source, mesh);
			yield {
				leadMin: entry.lead_min ?? 0,
				filename: entry.filename,
				bitmap: await surface.finish()
			};
		} finally {
			source.close();
		}
	}
}
