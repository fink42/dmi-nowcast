/**
 * Client-side point sampling of the national product grids.
 *
 * This is the zero-server-cost path the Phase A artifacts were designed for:
 * the browser already downloaded the quantised PNGs to draw them, so looking
 * up one point costs a projection and an array index. The server's
 * `/forecast?lat=&lon=` endpoint is the fallback, and the two must agree —
 * so the conventions below are copied from the sidecar, not re-derived:
 *
 *   col = (x - x_ul_m) / pixel_scale_x_m        [national_artifacts.py]
 *   row = (y_ul_m - y) / pixel_scale_y_m
 *   nearest pixel = round(row), round(col)      [app.py /forecast]
 *   outside [0, rows) × [0, cols) → off coverage (the endpoint's 400)
 *   value = level * scale + offset              [dequantise()]
 *   level == nodata (255) → null                [NODATA_LEVEL]
 *
 * The manifest's `grid` block already carries the *effective* pixel scale
 * (native × downsample_factor), so there is no second division by the
 * downsample factor here — that is exactly what `/forecast` does when it
 * divides the native index by `f`.
 */
import proj4 from 'proj4';
import type { ArtifactEntry, GridBlock, Manifest } from './manifest';
import { findArtifact, isCalibrated } from './manifest';
import { cellMotion, type CellMotion } from './motion';
import type { Gray8Image } from './png';

export interface GridIndex {
	/** Fractional grid position, before rounding to a pixel. */
	row: number;
	col: number;
}

export interface PixelIndex {
	row: number;
	col: number;
}

/** Cache the proj4 converters — building one parses the proj string. */
const converters = new Map<string, proj4.Converter>();

function converter(proj: string): proj4.Converter {
	let c = converters.get(proj);
	if (!c) {
		c = proj4('WGS84', proj);
		converters.set(proj, c);
	}
	return c;
}

/** lon/lat (degrees) → fractional grid position in the manifest's grid. */
export function lonLatToGrid(grid: GridBlock, lon: number, lat: number): GridIndex {
	const [x, y] = converter(grid.proj4).forward([lon, lat]);
	return {
		col: (x - grid.x_ul_m) / grid.pixel_scale_x_m,
		row: (grid.y_ul_m - y) / grid.pixel_scale_y_m
	};
}

/** Fractional grid position → lon/lat, for placing the grid on the map. */
export function gridToLonLat(grid: GridBlock, row: number, col: number): [number, number] {
	const x = grid.x_ul_m + col * grid.pixel_scale_x_m;
	const y = grid.y_ul_m - row * grid.pixel_scale_y_m;
	const [lon, lat] = converter(grid.proj4).inverse([x, y]);
	return [lon, lat];
}

/**
 * Nearest pixel, or null when the point falls outside the grid — the case
 * `/forecast` answers with 400 and the UI must render as "outside radar
 * coverage" rather than a 0 % probability.
 *
 * Note: Python's `round()` breaks ties to even and `Math.round` breaks them
 * upward. Half-pixel-exact coordinates are a measure-zero case that only
 * moves the sample one pixel (≈ 2 km) when it happens; everything else is
 * bit-identical.
 */
export function nearestPixel(grid: GridBlock, lon: number, lat: number): PixelIndex | null {
	const { row, col } = lonLatToGrid(grid, lon, lat);
	const r = Math.round(row);
	const c = Math.round(col);
	const [rows, cols] = grid.shape;
	if (r < 0 || r >= rows || c < 0 || c >= cols) return null;
	return { row: r, col: c };
}

/** True when the point is inside the product grid at all. */
export const inCoverage = (grid: GridBlock, lon: number, lat: number): boolean =>
	nearestPixel(grid, lon, lat) !== null;

/**
 * Dequantise one pixel of a grayscale product: `level * scale + offset`,
 * with the nodata level (255) becoming null.
 */
export function sampleArtifact(
	image: Gray8Image,
	entry: ArtifactEntry,
	pixel: PixelIndex
): number | null {
	if (image.width !== entry.shape[1] || image.height !== entry.shape[0]) {
		throw new Error(
			`artifact ${entry.filename}: PNG is ${image.height}×${image.width}, manifest says ${entry.shape[0]}×${entry.shape[1]}`
		);
	}
	if (entry.scale === undefined || entry.offset === undefined) {
		throw new Error(`artifact ${entry.filename} carries no scale/offset`);
	}
	const level = image.levels[pixel.row * image.width + pixel.col];
	if (level === (entry.nodata ?? 255)) return null;
	return level * entry.scale + entry.offset;
}

export interface LeadProbability {
	leadMin: number;
	/** Probability of rain by this lead, or null where the grid has no value. */
	pRain: number | null;
}

export interface PointForecast {
	lat: number;
	lon: number;
	/** Radar timestamp the products were computed from (ISO 8601, UTC). */
	radarTsUtc: string;
	perLead: LeadProbability[];
	/** Minutes until rain arrives; null when no rain within the horizon. */
	etaMin: number | null;
	/** Ensemble-median rain rate at the ETA step (mm/h); null without an ETA. */
	intensityMmH: number | null;
	/**
	 * Rain the radar is measuring at this point *right now* (mm/h), from the
	 * observation grid — not a forecast. Null when the cycle served no such
	 * grid (any manifest older than the product) or the pixel is nodata, and
	 * those two must stay indistinguishable from "we don't know": a null here
	 * may never be read as "it is dry here".
	 */
	observedMmH: number | null;
	/**
	 * Which way the echo over this point is moving. Null whenever there is no
	 * estimate — a cycle without motion grids, a nodata pixel, or the server
	 * path, which does not serve motion at all. Never a fabricated arrow.
	 */
	motion: CellMotion | null;
	/** Global confidence scalar; only the server path can supply it. */
	confidence: number | null;
	/** True only when every served lead went through a calibration curve. */
	calibrated: boolean;
	/** Where the numbers came from — shown in the panel, honestly. */
	source: 'client' | 'server';
}

/** One decoded grayscale product: the manifest entry plus its pixels. */
export interface DecodedGrid {
	entry: ArtifactEntry;
	image: Gray8Image;
}

/** Product grids for one cycle, decoded once and reused for every click. */
export interface DecodedGrids {
	pRain: Map<number, DecodedGrid>;
	eta?: DecodedGrid;
	intensity?: DecodedGrid;
	/**
	 * This cycle's observed rain field. Optional for the same reason motion is:
	 * it postdates the manifest schema, and a cycle without it loses the
	 * "it is raining here now" headline and nothing else.
	 */
	observed?: DecodedGrid;
	/**
	 * Cell motion, both components or neither — the sidecar writes them as a
	 * pair and a single component says nothing. Absent on a cycle that served
	 * no motion grids, which costs the arrow and nothing else.
	 */
	motion?: { east: DecodedGrid; north: DecodedGrid };
}

/**
 * Sample every product at one point. Returns null when the point lies outside
 * the grid, which the caller renders as the off-coverage state.
 */
export function samplePoint(
	manifest: Manifest,
	grids: DecodedGrids,
	lat: number,
	lon: number
): PointForecast | null {
	const pixel = nearestPixel(manifest.grid, lon, lat);
	if (!pixel) return null;

	const perLead: LeadProbability[] = [];
	for (const lead of manifest.leads_min) {
		const grid = grids.pRain.get(lead);
		perLead.push({
			leadMin: lead,
			pRain: grid ? sampleArtifact(grid.image, grid.entry, pixel) : null
		});
	}
	return {
		lat,
		lon,
		radarTsUtc: manifest.radar_ts_utc,
		perLead,
		etaMin: grids.eta ? sampleArtifact(grids.eta.image, grids.eta.entry, pixel) : null,
		intensityMmH: grids.intensity
			? sampleArtifact(grids.intensity.image, grids.intensity.entry, pixel)
			: null,
		observedMmH: grids.observed
			? sampleArtifact(grids.observed.image, grids.observed.entry, pixel)
			: null,
		motion: sampleMotion(grids, pixel),
		confidence: null,
		calibrated: isCalibrated(manifest),
		source: 'client'
	};
}

/**
 * Cell motion at one pixel. Either component reading nodata means the pixel
 * has no estimate — outside coverage, or too far from any echo — and the
 * answer is "we don't know", never an arrow pointing at nothing.
 */
export function sampleMotion(grids: DecodedGrids, pixel: PixelIndex): CellMotion | null {
	if (!grids.motion) return null;
	const { east, north } = grids.motion;
	return cellMotion(
		sampleArtifact(east.image, east.entry, pixel),
		sampleArtifact(north.image, north.entry, pixel)
	);
}

/** The grayscale artifacts one cycle needs, in fetch order. */
export function productArtifacts(manifest: Manifest): ArtifactEntry[] {
	const wanted: (ArtifactEntry | undefined)[] = [
		...manifest.leads_min.map((lead) => findArtifact(manifest, 'p_rain', lead)),
		findArtifact(manifest, 'eta'),
		findArtifact(manifest, 'intensity')
	];
	return wanted.filter((a): a is ArtifactEntry => a !== undefined);
}

/**
 * The observation grid of this cycle, or null when it served none. Kept out of
 * `productArtifacts` for the same reason the motion pair is: those are what
 * the client-side path needs to work at all, whereas a missing observation
 * only costs one headline. The sidecar stamps it `lead_min: 0` — it is a
 * measurement of now, not a lead — and a manifest that leaves the lead out is
 * read the same way.
 */
export function observedArtifact(manifest: Manifest): ArtifactEntry | null {
	return (
		manifest.artifacts.find((a) => a.product === 'observed_mm_h' && (a.lead_min ?? 0) === 0) ?? null
	);
}

/**
 * The optional cell-motion pair, or null when this cycle served neither. Kept
 * out of `productArtifacts` on purpose: those are required for the client-side
 * path to work at all, while a missing arrow is a missing row in a panel.
 */
export function motionArtifacts(manifest: Manifest): [ArtifactEntry, ArtifactEntry] | null {
	const east = findArtifact(manifest, 'motion_east_kmh');
	const north = findArtifact(manifest, 'motion_north_kmh');
	return east && north ? [east, north] : null;
}
