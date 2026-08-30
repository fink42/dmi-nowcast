/**
 * Why the overlay is resampled instead of stretched across four corners, in
 * numbers — and proof that the mesh actually fixes it.
 *
 * The first test reproduces the failure: interpolating the composite's four
 * corners linearly in Mercator (what a plain MapLibre image source does)
 * misplaces rain by kilometres over Denmark. The second measures the mesh's
 * piecewise-affine approximation against the exact projection and holds it to
 * a fraction of a radar pixel.
 */
import { describe, expect, it } from 'vitest';
import type { GridBlock } from '$lib/nowcast/manifest';
import { gridToLonLat, lonLatToGrid } from '$lib/nowcast/sampler';
import {
	buildMesh,
	inverseMercatorY,
	mercatorY,
	targetCorners,
	warpTarget,
	type WarpMesh
} from './warp';

/** The real 500 m composite geometry: 1728 × 1984 px, centred on the origin. */
const OVERLAY_GRID: GridBlock = {
	proj4: '+proj=stere +lat_0=56 +lon_0=10.5666 +lat_ts=56 +ellps=WGS84 +units=m +no_defs',
	x_ul_m: -496000,
	y_ul_m: 432000,
	pixel_scale_x_m: 500,
	pixel_scale_y_m: 500,
	shape: [1728, 1984],
	downsample_factor: 1
};

const R = 6378137;
const toMercator = (lon: number, lat: number): [number, number] => [
	(lon * Math.PI * R) / 180,
	(mercatorY(lat) * Math.PI * R) / 180
];

describe('Mercator helpers', () => {
	it('round-trips latitudes across Denmark', () => {
		for (const lat of [54.3, 55.0, 56.05, 57.8, 58.1]) {
			expect(inverseMercatorY(mercatorY(lat))).toBeCloseTo(lat, 9);
		}
	});
});

describe('four-corner image source', () => {
	it('would misplace rain by kilometres — which is why the mesh exists', () => {
		const [rows, cols] = OVERLAY_GRID.shape;
		const corner = (row: number, col: number) => {
			const [lon, lat] = gridToLonLat(OVERLAY_GRID, row, col);
			return toMercator(lon, lat);
		};
		const tl = corner(0, 0);
		const tr = corner(0, cols);
		const br = corner(rows, cols);
		const bl = corner(rows, 0);

		let worstDenmark = 0;
		for (let v = 0; v <= 1.0001; v += 0.025) {
			for (let u = 0; u <= 1.0001; u += 0.025) {
				const [lon, lat] = gridToLonLat(OVERLAY_GRID, v * rows, u * cols);
				if (lon < 7.5 || lon > 15.5 || lat < 54.3 || lat > 58) continue;
				const truth = toMercator(lon, lat);
				const guess = [0, 1].map(
					(i) =>
						(1 - u) * (1 - v) * tl[i] + u * (1 - v) * tr[i] + u * v * br[i] + (1 - u) * v * bl[i]
				);
				// Mercator metres → ground metres at this latitude.
				const error =
					Math.hypot(guess[0] - truth[0], guess[1] - truth[1]) * Math.cos((lat * Math.PI) / 180);
				worstDenmark = Math.max(worstDenmark, error);
			}
		}
		// Measured ≈ 14.8 km inside the Denmark window (32 km over the full grid).
		expect(worstDenmark).toBeGreaterThan(10_000);
	});
});

describe('warp target and mesh', () => {
	const target = warpTarget(OVERLAY_GRID);

	it('clips the composite to a Denmark-sized Mercator rectangle', () => {
		expect(target.west).toBeGreaterThanOrEqual(7);
		expect(target.east).toBeLessThanOrEqual(15.9);
		expect(target.south).toBeGreaterThanOrEqual(54);
		expect(target.north).toBeLessThanOrEqual(58.4);
		expect(target.width).toBeGreaterThan(500);
		expect(Math.max(target.width, target.height)).toBeLessThanOrEqual(1600);
	});

	it('gives MapLibre an axis-aligned lon/lat box, which maps exactly', () => {
		const [tl, tr, br, bl] = targetCorners(target);
		expect(tl[1]).toBe(tr[1]); // same latitude along the top
		expect(bl[1]).toBe(br[1]);
		expect(tl[0]).toBe(bl[0]); // same longitude down the left
		expect(tr[0]).toBe(br[0]);
	});

	it('approximates the true projection to a fraction of a radar pixel', () => {
		const mesh = buildMesh(OVERLAY_GRID, target);
		const groundMetresPerPixel =
			((target.east - target.west) * 111_320 * Math.cos(((target.north + target.south) / 2) * (Math.PI / 180))) /
			target.width;

		let worst = 0;
		for (let j = 0; j < mesh.ny; j++) {
			for (let i = 0; i < mesh.nx; i++) {
				// Sample inside the cell, where a piecewise fit is at its worst.
				for (const [fx, fy] of [
					[0.5, 0.5],
					[0.85, 0.85],
					[1, 1]
				]) {
					worst = Math.max(worst, cellErrorPx(mesh, i, j, fx, fy));
				}
			}
		}
		const worstGroundMetres = worst * groundMetresPerPixel;
		// A radar pixel is 500 m; the default 40 px cells land near 60 m —
		// a fifth of a pixel, against 15 km for the four-corner shortcut.
		expect(worstGroundMetres).toBeLessThan(100);
	});
});

/**
 * Error, in destination pixels, of the cell's affine map at (fx, fy) inside
 * it — the same three-corner affine `warpImage` hands to the canvas.
 */
function cellErrorPx(mesh: WarpMesh, i: number, j: number, fx: number, fy: number): number {
	const { nx, dx, dy, sx, sy, target } = mesh;
	const k00 = j * (nx + 1) + i;
	const k10 = k00 + 1;
	const k01 = k00 + (nx + 1);

	const s1x = sx[k10] - sx[k00];
	const s1y = sy[k10] - sy[k00];
	const s2x = sx[k01] - sx[k00];
	const s2y = sy[k01] - sy[k00];
	const det = s1x * s2y - s2x * s1y;
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

	// The exact destination point and its exact source position.
	const px = dx[k00] + fx * d1x;
	const py = dy[k00] + fy * d2y;
	const yNorth = mercatorY(target.north);
	const ySouth = mercatorY(target.south);
	const lat = inverseMercatorY(yNorth + (py / target.height) * (ySouth - yNorth));
	const lon = target.west + (px / target.width) * (target.east - target.west);
	const { row, col } = lonLatToGrid(OVERLAY_GRID, lon, lat);
	const sxExact = col + 0.5;
	const syExact = row + 0.5;

	// Where the affine would put that source sample.
	const mappedX = a * sxExact + c * syExact + e;
	const mappedY = b * sxExact + d * syExact + f;
	return Math.hypot(mappedX - px, mappedY - py);
}
