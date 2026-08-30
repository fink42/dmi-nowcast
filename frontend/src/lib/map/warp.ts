/**
 * Reprojecting the radar overlay from the composite's polar-stereographic
 * grid into the Web Mercator space MapLibre draws in.
 *
 * Why this exists at all: MapLibre's image source stretches an image across
 * four corner coordinates, interpolating linearly in Mercator. The radar grid
 * is polar stereographic and ~1000 km across, and the two projections do not
 * differ by an affine map. Measured against the real composite geometry
 * (1984 × 1728 px at 500 m), the four-corner shortcut misplaces rain by up to
 * **15 km inside Denmark** and 32 km over the full grid — a shower on the
 * wrong side of a city. So the frame is resampled instead:
 *
 *   dest canvas (linear in Mercator) → lon/lat → proj4 → grid col/row
 *
 * evaluated on a control mesh of ~64 px cells and applied with one clipped,
 * affine `drawImage` per cell. Residual error scales with the square of the
 * cell size: at these cell sizes it is well under a tenth of a radar pixel.
 * The resulting canvas *is* an axis-aligned Mercator rectangle, so handing
 * MapLibre its lon/lat corners is then exact.
 */
import type { GridBlock } from '$lib/nowcast/manifest';
import { gridToLonLat, lonLatToGrid } from '$lib/nowcast/sampler';

/** Mercator y for a latitude, in the same units as longitude degrees. */
export const mercatorY = (lat: number): number =>
	(Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)) * 180) / Math.PI;

/** Inverse of {@link mercatorY}. */
export const inverseMercatorY = (y: number): number =>
	(Math.atan(Math.exp((y * Math.PI) / 180)) - Math.PI / 4) * (360 / Math.PI);

export interface WarpTarget {
	west: number;
	south: number;
	east: number;
	north: number;
	width: number;
	height: number;
}

/** MapLibre image-source corners: TL, TR, BR, BL as [lon, lat]. */
export type Corners = [[number, number], [number, number], [number, number], [number, number]];

export const targetCorners = (t: WarpTarget): Corners => [
	[t.west, t.north],
	[t.east, t.north],
	[t.east, t.south],
	[t.west, t.south]
];

/** Denmark, with sea margin: the only part of the composite worth drawing. */
const WINDOW = { west: 7.0, south: 54.0, east: 15.9, north: 58.4 };

/**
 * Choose the Mercator rectangle and canvas size to resample a grid into.
 * `metresPerPixel` is ground resolution at the window's centre latitude; the
 * default slightly oversamples the 500 m composite so the loop stays sharp
 * at city zoom levels without turning a phone's memory into a texture farm.
 */
export function warpTarget(grid: GridBlock, metresPerPixel = 400, maxDimension = 1600): WarpTarget {
	// The grid's own lon/lat footprint: its edges bow in geographic space, so
	// sample along them rather than trusting the four corners.
	const [rows, cols] = grid.shape;
	let west = 180;
	let east = -180;
	let south = 90;
	let north = -90;
	const steps = 16;
	for (let i = 0; i <= steps; i++) {
		const f = i / steps;
		const edges: [number, number][] = [
			[0, f * cols],
			[rows, f * cols],
			[f * rows, 0],
			[f * rows, cols]
		];
		for (const [row, col] of edges) {
			const [lon, lat] = gridToLonLat(grid, row, col);
			west = Math.min(west, lon);
			east = Math.max(east, lon);
			south = Math.min(south, lat);
			north = Math.max(north, lat);
		}
	}
	// Clip to the Denmark window — the composite reaches well into the North
	// Sea and southern Sweden, which nobody opens this site for.
	west = Math.max(west, WINDOW.west);
	east = Math.min(east, WINDOW.east);
	south = Math.max(south, WINDOW.south);
	north = Math.min(north, WINDOW.north);

	const midLat = (north + south) / 2;
	const metresPerDegreeLon = 111_320 * Math.cos((midLat * Math.PI) / 180);
	let width = Math.round(((east - west) * metresPerDegreeLon) / metresPerPixel);
	const mercSpan = mercatorY(north) - mercatorY(south);
	let height = Math.round((width * mercSpan) / (east - west));
	const scale = Math.min(1, maxDimension / Math.max(width, height));
	width = Math.max(1, Math.round(width * scale));
	height = Math.max(1, Math.round(height * scale));
	return { west, south, east, north, width, height };
}

export interface WarpMesh {
	target: WarpTarget;
	/** Cells along x and y. */
	nx: number;
	ny: number;
	/** Dest canvas coordinates of the (nx+1)×(ny+1) control points. */
	dx: Float64Array;
	dy: Float64Array;
	/** Matching source-image coordinates (grid pixel centres + 0.5). */
	sx: Float64Array;
	sy: Float64Array;
}

/**
 * Build the control mesh mapping the destination canvas back into the source
 * image. Depends only on the manifest's grid block, so it is computed once
 * per cycle and reused for every frame.
 */
export function buildMesh(grid: GridBlock, target: WarpTarget, cellPx = 40): WarpMesh {
	const nx = Math.max(2, Math.ceil(target.width / cellPx));
	const ny = Math.max(2, Math.ceil(target.height / cellPx));
	const n = (nx + 1) * (ny + 1);
	const dx = new Float64Array(n);
	const dy = new Float64Array(n);
	const sx = new Float64Array(n);
	const sy = new Float64Array(n);
	const yNorth = mercatorY(target.north);
	const ySouth = mercatorY(target.south);

	for (let j = 0; j <= ny; j++) {
		const py = (j / ny) * target.height;
		const lat = inverseMercatorY(yNorth + (py / target.height) * (ySouth - yNorth));
		for (let i = 0; i <= nx; i++) {
			const px = (i / nx) * target.width;
			const lon = target.west + (px / target.width) * (target.east - target.west);
			const { row, col } = lonLatToGrid(grid, lon, lat);
			const k = j * (nx + 1) + i;
			dx[k] = px;
			dy[k] = py;
			// Grid index → image coordinates: index 0 is the centre of the
			// first pixel, which sits half a pixel into the image.
			sx[k] = col + 0.5;
			sy[k] = row + 0.5;
		}
	}
	return { target, nx, ny, dx, dy, sx, sy };
}

type AnyContext = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

/**
 * Draw one source frame into a Mercator canvas through the mesh. The canvas
 * must be `mesh.target.width × mesh.target.height`.
 */
export function warpImage(
	ctx: AnyContext,
	source: CanvasImageSource,
	mesh: WarpMesh,
	opacity = 1
): void {
	const { nx, ny, dx, dy, sx, sy, target } = mesh;
	ctx.setTransform(1, 0, 0, 1, 0, 0);
	ctx.clearRect(0, 0, target.width, target.height);
	ctx.imageSmoothingEnabled = true;
	ctx.imageSmoothingQuality = 'high';
	ctx.globalAlpha = opacity;

	for (let j = 0; j < ny; j++) {
		for (let i = 0; i < nx; i++) {
			const k00 = j * (nx + 1) + i;
			const k10 = k00 + 1;
			const k01 = k00 + (nx + 1);

			// Affine from three correspondences: source → destination.
			const s1x = sx[k10] - sx[k00];
			const s1y = sy[k10] - sy[k00];
			const s2x = sx[k01] - sx[k00];
			const s2y = sy[k01] - sy[k00];
			const det = s1x * s2y - s2x * s1y;
			if (!Number.isFinite(det) || Math.abs(det) < 1e-9) continue;
			const d1x = dx[k10] - dx[k00];
			const d1y = dy[k10] - dy[k00];
			const d2x = dx[k01] - dx[k00];
			const d2y = dy[k01] - dy[k00];
			const a = (d1x * s2y - d2x * s1y) / det;
			const c = (d2x * s1x - d1x * s2x) / det;
			const b = (d1y * s2y - d2y * s1y) / det;
			const d = (d2y * s1x - d1y * s2x) / det;
			const e = dx[k00] - (a * sx[k00] + c * sy[k00]);
			const f = dy[k00] - (b * sx[k00] + d * sy[k00]);

			ctx.save();
			ctx.beginPath();
			// Half-pixel bleed, so neighbouring cells overlap instead of
			// leaving hairline seams where their affines disagree.
			ctx.rect(dx[k00] - 0.5, dy[k00] - 0.5, d1x + 1, d2y + 1);
			ctx.clip();
			ctx.setTransform(a, b, c, d, e, f);
			ctx.drawImage(source, 0, 0);
			ctx.restore();
		}
	}
	ctx.setTransform(1, 0, 0, 1, 0, 0);
	ctx.globalAlpha = 1;
}
