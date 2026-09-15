/**
 * Reading mm/h off the bundle's observation grids.
 *
 * The bundle ships two PNGs per composite: an RGBA overlay to look at, and
 * a grayscale-8 `observed.png` that *is the number*. The colours are
 * advisory — light rain is deliberately faded because a column-max
 * composite over-reads faint echo — so every figure a reviewer quotes has
 * to come from the grayscale grid, decoded byte-exactly by
 * `nowcast/png.ts` and dequantised with the manifest's own scale and
 * offset. None of that is re-implemented here: the same decoder and the
 * same sampler the public site uses are called with an artifact entry
 * assembled from `frames.encodings.observed`, so the review page and the
 * site cannot drift apart about what a level means.
 *
 * The disc statistic is the one place this is an approximation, and it is
 * flagged in `sampleDisc` rather than hidden: the service computed its p90
 * over the 500 m native grid, while the bundle ships the 2 km product grid
 * whose pixels are already block-p90 reductions. A disc p90 taken here is
 * therefore a p90 of p90s. It is the right order of magnitude and the right
 * shape, and it is not the service's number.
 */
import type { ArtifactEntry } from '$lib/nowcast/manifest';
import type { Gray8Image } from '$lib/nowcast/png';
import { lonLatToGrid, nearestPixel, sampleArtifact, type PixelIndex } from '$lib/nowcast/sampler';
import type { GridBlock, Manifest } from './schema';

/**
 * The manifest's observation encoding as an `ArtifactEntry`, so the
 * existing sampler can be used unchanged.
 *
 * Null when the bundle does not state the quantisation. There is no default
 * worth having: `level * scale + offset` with a guessed scale produces a
 * plausible, wrong mm/h — the one error a reviewer could never catch by
 * eye, since the picture would look exactly the same.
 */
export function observedArtifact(manifest: Manifest): ArtifactEntry | null {
	const frames = manifest.frames;
	const grid = manifest.grid;
	if (frames === null || grid === null) return null;
	const { scale, offset, nodata, encoding, units } = frames.encodings.observed;
	if (encoding !== 'grayscale8') return null;
	return {
		// Only ever quoted in the sampler's own error messages.
		filename: 'frames/<stamp>.observed.png',
		product: 'observed_mm_h',
		lead_min: 0,
		encoding: 'grayscale8',
		scale,
		offset,
		nodata,
		units,
		shape: grid.shape
	};
}

/** True when a decoded PNG matches the grid the manifest describes. */
function fits(image: Gray8Image, grid: GridBlock): boolean {
	const [rows, cols] = grid.shape;
	if (image.height === rows && image.width === cols) return true;
	// A builder bug, not a dry pixel. Say so once, loudly, and read as
	// unknown — a silently misaligned grid would put a real rain rate from
	// the wrong place under the cursor.
	console.warn(
		`review observed grid is ${image.height}×${image.width}, manifest says ${rows}×${cols}`
	);
	return false;
}

/**
 * mm/h at one point, or null.
 *
 * Null covers three different things the UI must distinguish in words:
 * outside the grid, a nodata pixel (level 255), and a bundle whose
 * quantisation could not be read. All three are "we do not know" and none
 * of them is zero — the difference between a dry pixel and an unmeasured
 * one is the difference between a false alarm and an unreviewable event.
 */
export function sampleObservedAt(
	image: Gray8Image,
	manifest: Manifest,
	lat: number,
	lon: number
): number | null {
	const entry = observedArtifact(manifest);
	const grid = manifest.grid;
	if (entry === null || grid === null || !fits(image, grid)) return null;
	const pixel = nearestPixel(grid, lon, lat);
	if (pixel === null) return null;
	return sampleArtifact(image, entry, pixel);
}

export interface DiscSample {
	/**
	 * The 90th percentile over the disc — the statistic the service's
	 * `raining_now` acts on. Null when no pixel in the disc carried a value.
	 */
	p90MmH: number | null;
	maxMmH: number | null;
	meanMmH: number | null;
	/** Pixels inside the disc, including nodata ones. */
	nPixels: number;
	/** Of those, how many carried a value. */
	nValid: number;
	radiusM: number;
	/**
	 * True when the disc is a single pixel — at a 1 km radius on the 2 km
	 * product grid it usually is, and then "p90 over the disc" is just that
	 * pixel's own block-p90. The read-out must say so rather than implying a
	 * distribution was summarised.
	 */
	singlePixel: boolean;
}

/**
 * The disc statistics at a point, computed the way `core/sample.py` does.
 *
 * Conventions copied from the Python rather than re-derived, because the
 * point of showing this number is to compare it with the one the service
 * acted on:
 *
 *   radius_px = radius_m / ((pixel_scale_x + pixel_scale_y) / 2)
 *   a pixel is in the disc when (r − row)² + (c − col)² <= radius_px²
 *   nodata pixels are dropped, not zeroed
 *   p90 is numpy's linear-interpolation percentile
 *
 * Null when the point is off the grid, when the manifest has no
 * quantisation, or when the PNG does not match the grid. An empty disc
 * comes back as a `DiscSample` with null statistics and `nValid: 0`, which
 * is a different statement from "no disc" and is rendered as one.
 */
export function sampleDisc(
	image: Gray8Image,
	manifest: Manifest,
	lat: number,
	lon: number,
	radiusM: number
): DiscSample | null {
	const entry = observedArtifact(manifest);
	const grid = manifest.grid;
	if (entry === null || grid === null || !fits(image, grid)) return null;
	if (!Number.isFinite(radiusM) || radiusM <= 0) return null;

	const centre = lonLatToGrid(grid, lon, lat);
	if (!Number.isFinite(centre.row) || !Number.isFinite(centre.col)) return null;
	const pixelScaleM = (grid.pixel_scale_x_m + grid.pixel_scale_y_m) / 2;
	const radiusPx = radiusM / pixelScaleM;
	const [rows, cols] = grid.shape;

	const rowLo = Math.max(0, Math.floor(centre.row - radiusPx));
	const rowHi = Math.min(rows - 1, Math.ceil(centre.row + radiusPx));
	const colLo = Math.max(0, Math.floor(centre.col - radiusPx));
	const colHi = Math.min(cols - 1, Math.ceil(centre.col + radiusPx));
	if (rowHi < rowLo || colHi < colLo) return null;

	const values: number[] = [];
	let nPixels = 0;
	for (let row = rowLo; row <= rowHi; row++) {
		for (let col = colLo; col <= colHi; col++) {
			const dr = row - centre.row;
			const dc = col - centre.col;
			if (dr * dr + dc * dc > radiusPx * radiusPx) continue;
			nPixels += 1;
			const value = sampleArtifact(image, entry, { row, col } as PixelIndex);
			if (value !== null) values.push(value);
		}
	}
	if (nPixels === 0) return null;
	if (values.length === 0) {
		return {
			p90MmH: null,
			maxMmH: null,
			meanMmH: null,
			nPixels,
			nValid: 0,
			radiusM,
			singlePixel: nPixels === 1
		};
	}
	const sum = values.reduce((total, value) => total + value, 0);
	return {
		p90MmH: percentile(values, 90),
		maxMmH: Math.max(...values),
		meanMmH: sum / values.length,
		nPixels,
		nValid: values.length,
		radiusM,
		singlePixel: nPixels === 1
	};
}

/** Backwards-compatible alias: the disc statistic the read-out leads with. */
export const sampleDiscP90 = (
	image: Gray8Image,
	manifest: Manifest,
	lat: number,
	lon: number,
	radiusM: number
): number | null => sampleDisc(image, manifest, lat, lon, radiusM)?.p90MmH ?? null;

/**
 * `numpy.percentile(values, p)` with the default linear interpolation —
 * the exact function `core/sample.py` calls, so the browser's disc p90 and
 * the sidecar's agree to the last decimal on the same pixels.
 *
 * Sorts a copy: the caller's array is its own.
 */
export function percentile(values: readonly number[], p: number): number | null {
	if (values.length === 0) return null;
	const sorted = [...values].sort((a, b) => a - b);
	if (sorted.length === 1) return sorted[0];
	const position = ((sorted.length - 1) * p) / 100;
	const lower = Math.floor(position);
	const upper = Math.ceil(position);
	if (lower === upper) return sorted[lower];
	return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}
