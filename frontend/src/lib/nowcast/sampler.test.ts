/**
 * The contract this file defends: sampling a point in the browser must give
 * the same answer as `GET /forecast?lat=&lon=` on the server.
 *
 * Three things can drift, so all three are pinned here:
 *  1. the quantisation round-trip (level → physical value, 255 → null),
 *  2. the projection (proj4js against pyproj's numbers for the same grid),
 *  3. the pixel convention (round to nearest; outside the grid is *unknown*,
 *     not zero — the endpoint answers 400 there and the UI must say so).
 *
 * The PNGs are built here rather than checked in, with every PNG row filter
 * exercised, because the decoder is ours too and a decoder bug would look
 * exactly like a sampling bug.
 */
import { deflateSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { decodeGray8Png } from './png';
import type { ArtifactEntry, Manifest } from './manifest';
import {
	forecastSeriesArtifacts,
	lonLatToGrid,
	nearestPixel,
	observedArtifact,
	sampleArtifact,
	samplePoint
} from './sampler';

// --- fixtures ---------------------------------------------------------------

/** The DMI composite's projection, verbatim from a real manifest. */
const PROJ4 = '+proj=stere +lat_0=56 +lon_0=10.5666 +lat_ts=56 +ellps=WGS84 +units=m +no_defs';

/**
 * A grid block shaped like the real one: the native 500 m composite
 * downsampled ×4, so the manifest carries a 2 km effective pixel scale.
 */
const GRID = {
	proj4: PROJ4,
	x_ul_m: -496000,
	y_ul_m: 432000,
	pixel_scale_x_m: 2000,
	pixel_scale_y_m: 2000,
	shape: [432, 496] as [number, number],
	downsample_factor: 4
};

/**
 * Ground truth from pyproj (the library the sidecar projects with), for the
 * grid above:
 *   Transformer.from_crs("EPSG:4326", CRS.from_proj4(PROJ4), always_xy=True)
 *   col = (x - x_ul) / 2000 ; row = (y_ul - y) / 2000
 */
const PYPROJ_POINTS = [
	{ name: 'Copenhagen', lon: 12.5683, lat: 55.6761, col: 310.959755, row: 233.123033 },
	{ name: 'Aarhus', lon: 10.2039, lat: 56.1629, col: 236.732628, row: 206.901509 },
	{ name: 'Bornholm', lon: 14.9, lat: 55.1, col: 386.244759, row: 261.796029 },
	{ name: 'Skagen', lon: 10.5833, lat: 57.7167, col: 248.497811, row: 120.404784 },
	{ name: 'projection origin', lon: 10.5666, lat: 56.0, col: 248, row: 216 }
];

/** The sidecar's fixed quantisation ranges (national_artifacts.QUANT_SPECS). */
const P_RAIN = { lo: 0, hi: 1 };
const ETA = { lo: 0, hi: 120 };
const INTENSITY = { lo: 0, hi: 100 };
/** The observation grid: mm/h, quantised exactly like intensity. */
const OBSERVED = { lo: 0, hi: 100 };
/** The advected rain field: same units, same quantisation as the observation. */
const FORECAST = { lo: 0, hi: 100 };
/** Cell motion: symmetric about zero, ±MOTION_MAX_ABS_KMH. */
const MOTION = { lo: -120, hi: 120 };
const MAX_LEVEL = 254;
const NODATA_LEVEL = 255;

const scaleOf = (spec: { lo: number; hi: number }) => (spec.hi - spec.lo) / MAX_LEVEL;

/** `national_artifacts.quantise` — the exact server-side rounding. */
function quantise(value: number | null, spec: { lo: number; hi: number }): number {
	if (value === null || !Number.isFinite(value)) return NODATA_LEVEL;
	const clipped = Math.min(Math.max(value, spec.lo), spec.hi);
	return Math.round((clipped - spec.lo) / scaleOf(spec));
}

// --- a minimal PNG encoder, so the decoder is tested on real bytes ----------

const CRC_TABLE = (() => {
	const table = new Uint32Array(256);
	for (let n = 0; n < 256; n++) {
		let c = n;
		for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
		table[n] = c >>> 0;
	}
	return table;
})();

const crc32 = (buf: Uint8Array) => {
	let c = 0xffffffff;
	for (const byte of buf) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8);
	return (c ^ 0xffffffff) >>> 0;
};

function chunk(type: string, body: Uint8Array): Uint8Array {
	const out = new Uint8Array(body.length + 12);
	const view = new DataView(out.buffer);
	view.setUint32(0, body.length);
	for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
	out.set(body, 8);
	view.setUint32(8 + body.length, crc32(out.subarray(4, 8 + body.length)));
	return out;
}

const paeth = (a: number, b: number, c: number) => {
	const p = a + b - c;
	const pa = Math.abs(p - a);
	const pb = Math.abs(p - b);
	const pc = Math.abs(p - c);
	return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
};

/**
 * Encode a grayscale-8 PNG, cycling through all five row filters so the
 * decoder's unfilter paths are all exercised (PIL picks adaptively too).
 */
function encodeGray8Png(levels: Uint8Array, width: number, height: number): Uint8Array {
	const raw = new Uint8Array((width + 1) * height);
	for (let y = 0; y < height; y++) {
		const filter = y % 5;
		raw[y * (width + 1)] = filter;
		for (let x = 0; x < width; x++) {
			const value = levels[y * width + x];
			const a = x > 0 ? levels[y * width + x - 1] : 0;
			const b = y > 0 ? levels[(y - 1) * width + x] : 0;
			const c = x > 0 && y > 0 ? levels[(y - 1) * width + x - 1] : 0;
			let encoded: number;
			switch (filter) {
				case 1:
					encoded = value - a;
					break;
				case 2:
					encoded = value - b;
					break;
				case 3:
					encoded = value - ((a + b) >> 1);
					break;
				case 4:
					encoded = value - paeth(a, b, c);
					break;
				default:
					encoded = value;
			}
			raw[y * (width + 1) + 1 + x] = encoded & 0xff;
		}
	}
	const ihdr = new Uint8Array(13);
	const view = new DataView(ihdr.buffer);
	view.setUint32(0, width);
	view.setUint32(4, height);
	ihdr[8] = 8; // bit depth
	ihdr[9] = 0; // grayscale
	const parts = [
		new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
		chunk('IHDR', ihdr),
		chunk('IDAT', new Uint8Array(deflateSync(raw))),
		chunk('IEND', new Uint8Array(0))
	];
	const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
	let at = 0;
	for (const part of parts) {
		out.set(part, at);
		at += part.length;
	}
	return out;
}

const gridEntry = (
	filename: string,
	product: ArtifactEntry['product'],
	leadMin: number | null,
	spec: { lo: number; hi: number },
	shape: [number, number]
): ArtifactEntry => ({
	filename,
	product,
	lead_min: leadMin,
	encoding: 'grayscale8',
	scale: scaleOf(spec),
	offset: spec.lo,
	nodata: NODATA_LEVEL,
	shape
});

// --- tests ------------------------------------------------------------------

describe('grid projection', () => {
	it('matches pyproj on the sidecar grid, to well under a metre', () => {
		for (const point of PYPROJ_POINTS) {
			const index = lonLatToGrid(GRID, point.lon, point.lat);
			// 1e-4 px at a 2 km pixel is 20 cm on the ground.
			expect(index.col, point.name).toBeCloseTo(point.col, 4);
			expect(index.row, point.name).toBeCloseTo(point.row, 4);
		}
	});

	it('rounds to the nearest pixel, the way /forecast does', () => {
		// Copenhagen sits at row 233.12, col 310.96 → (233, 311), not (233, 310).
		expect(nearestPixel(GRID, 12.5683, 55.6761)).toEqual({ row: 233, col: 311 });
		// Skagen at row 120.40, col 248.50.
		expect(nearestPixel(GRID, 10.5833, 57.7167)).toEqual({ row: 120, col: 248 });
	});

	it('reports off-coverage outside the grid instead of a value', () => {
		// Well west of the composite, in the North Sea.
		expect(nearestPixel(GRID, -2.0, 56.0)).toBeNull();
		// And south of it.
		expect(nearestPixel(GRID, 10.5, 48.0)).toBeNull();
	});
});

describe('quantised PNG sampling', () => {
	const width = 5;
	const height = 4;
	/** Physical probabilities laid out on a tiny grid; null = nodata. */
	const values: (number | null)[] = [
		0, 0.25, 0.5, 0.75, 1,
		0.01, 0.123, 0.456, 0.789, 0.999,
		null, 0.5, null, 0.5, null,
		0.2, 0.4, 0.6, 0.8, 0.95
	];
	const levels = new Uint8Array(values.map((v) => quantise(v, P_RAIN)));
	const entry = gridEntry('p_rain_20min_202608281200.png', 'p_rain', 20, P_RAIN, [height, width]);

	it('round-trips every value within half a quantisation step', async () => {
		const image = await decodeGray8Png(encodeGray8Png(levels, width, height));
		expect(image.width).toBe(width);
		expect(image.height).toBe(height);
		const halfStep = scaleOf(P_RAIN) / 2;
		for (let row = 0; row < height; row++) {
			for (let col = 0; col < width; col++) {
				const expected = values[row * width + col];
				const sampled = sampleArtifact(image, entry, { row, col });
				if (expected === null) {
					expect(sampled).toBeNull();
				} else {
					expect(sampled).not.toBeNull();
					expect(Math.abs((sampled as number) - expected)).toBeLessThanOrEqual(halfStep + 1e-12);
					// And it is exactly level * scale + offset, the documented rule.
					expect(sampled).toBeCloseTo(
						image.levels[row * width + col] * (entry.scale as number) + (entry.offset as number),
						12
					);
				}
			}
		}
	});

	it('maps level 255 to null rather than to the top of the range', async () => {
		const image = await decodeGray8Png(encodeGray8Png(levels, width, height));
		expect(image.levels[2 * width + 0]).toBe(255);
		expect(sampleArtifact(image, entry, { row: 2, col: 0 })).toBeNull();
		// 254 — the level just below nodata — is a real value: probability 1.
		expect(sampleArtifact(image, entry, { row: 0, col: 4 })).toBeCloseTo(1, 12);
	});

	it('refuses a PNG whose shape disagrees with the manifest', async () => {
		const image = await decodeGray8Png(encodeGray8Png(levels, width, height));
		const wrong = { ...entry, shape: [height + 1, width] as [number, number] };
		expect(() => sampleArtifact(image, wrong, { row: 0, col: 0 })).toThrow(/manifest says/);
	});

	it('honours the eta and intensity ranges, not just probability', async () => {
		const etaLevels = new Uint8Array([quantise(12.5, ETA), quantise(0, ETA), NODATA_LEVEL]);
		const etaImage = await decodeGray8Png(encodeGray8Png(etaLevels, 3, 1));
		const etaEntry = gridEntry('eta_202608281200.png', 'eta', null, ETA, [1, 3]);
		// 0–120 min over 254 levels: half a step is 14 seconds.
		const etaHalfStep = scaleOf(ETA) / 2;
		expect(
			Math.abs((sampleArtifact(etaImage, etaEntry, { row: 0, col: 0 }) as number) - 12.5)
		).toBeLessThanOrEqual(etaHalfStep);
		expect(sampleArtifact(etaImage, etaEntry, { row: 0, col: 1 })).toBe(0);
		expect(sampleArtifact(etaImage, etaEntry, { row: 0, col: 2 })).toBeNull();

		const mmLevels = new Uint8Array([quantise(2.4, INTENSITY), quantise(100, INTENSITY)]);
		const mmImage = await decodeGray8Png(encodeGray8Png(mmLevels, 2, 1));
		const mmEntry = gridEntry('intensity_202608281200.png', 'intensity', null, INTENSITY, [1, 2]);
		expect(sampleArtifact(mmImage, mmEntry, { row: 0, col: 0 })).toBeCloseTo(2.4, 1);
		expect(sampleArtifact(mmImage, mmEntry, { row: 0, col: 1 })).toBeCloseTo(100, 12);
	});
});

describe('samplePoint', () => {
	/**
	 * A 3×3 fixture grid whose centre pixel *is* the projection origin
	 * (56.0 N, 10.5666 E), 2 km pixels — so the expected indices are exact.
	 */
	const smallGrid = { ...GRID, shape: [3, 3] as [number, number], x_ul_m: -2000, y_ul_m: 2000 };
	const manifest = {
		schema_version: 1,
		cycle: '202608281200',
		radar_ts_utc: '2026-08-28T12:00:00+00:00',
		generated_at_utc: '2026-08-28T12:01:30+00:00',
		threshold_mm_h: 0.1,
		timestep_min: 5,
		frame_age_min: 1.5,
		n_members: 24,
		leads_min: [10, 20],
		grid: smallGrid,
		overlay_grid: null,
		calibration: { fitted_at: '2026-08-01T00:00:00+00:00', calibrated_leads: [10, 20] },
		artifacts: []
	} satisfies Manifest;

	async function decoded(values: (number | null)[], spec: { lo: number; hi: number }) {
		const levels = new Uint8Array(values.map((v) => quantise(v, spec)));
		return decodeGray8Png(encodeGray8Png(levels, 3, 3));
	}

	it('assembles per-lead probabilities, ETA and intensity for one pixel', async () => {
		const centre = [null, null, null, null, 0.62, null, null, null, null];
		const grids = {
			pRain: new Map([
				[
					10,
					{
						entry: gridEntry('p_rain_10min_x.png', 'p_rain', 10, P_RAIN, [3, 3]),
						image: await decoded(centre, P_RAIN)
					}
				],
				[
					20,
					{
						entry: gridEntry('p_rain_20min_x.png', 'p_rain', 20, P_RAIN, [3, 3]),
						image: await decoded([...Array(4).fill(null), 0.81, ...Array(4).fill(null)], P_RAIN)
					}
				]
			]),
			eta: {
				entry: gridEntry('eta_x.png', 'eta', null, ETA, [3, 3]),
				image: await decoded([...Array(4).fill(null), 14, ...Array(4).fill(null)], ETA)
			},
			intensity: {
				entry: gridEntry('intensity_x.png', 'intensity', null, INTENSITY, [3, 3]),
				image: await decoded([...Array(4).fill(null), 2.4, ...Array(4).fill(null)], INTENSITY)
			}
		};

		// The projection origin lands on the centre pixel of this grid.
		const forecast = samplePoint(manifest, grids, 56.0, 10.5666);
		expect(forecast).not.toBeNull();
		expect(forecast!.perLead.map((l) => l.leadMin)).toEqual([10, 20]);
		expect(forecast!.perLead[0].pRain).toBeCloseTo(0.62, 2);
		expect(forecast!.perLead[1].pRain).toBeCloseTo(0.81, 2);
		expect(forecast!.etaMin).toBeCloseTo(14, 0);
		expect(forecast!.intensityMmH).toBeCloseTo(2.4, 1);
		// This cycle served no observation grid, which is null — "we don't
		// know whether it is raining here", never "it is dry here".
		expect(forecast!.observedMmH).toBeNull();
		expect(forecast!.calibrated).toBe(true);
		expect(forecast!.source).toBe('client');
		expect(forecast!.radarTsUtc).toBe(manifest.radar_ts_utc);

		// A neighbouring pixel (one row north) is nodata everywhere: the
		// answer is null, never a zero probability.
		const neighbour = samplePoint(manifest, grids, 56.017, 10.5666);
		expect(neighbour!.perLead.every((l) => l.pRain === null)).toBe(true);
		expect(neighbour!.etaMin).toBeNull();

		// And a point outside the grid is off-coverage, not p = 0.
		expect(samplePoint(manifest, grids, 55.0, 12.0)).toBeNull();
	});

	/**
	 * Cell motion is the one product that may simply not be there — an older
	 * cycle, a failed download, or a pixel too far from any echo for an honest
	 * estimate. Each of those has to reach the panel as "no motion", never as
	 * an arrow pointing somewhere plausible.
	 */
	describe('cell motion', () => {
		/** p_rain/eta/intensity, all nodata: this block is about motion only. */
		async function baseGrids() {
			const empty = Array(9).fill(null);
			return {
				pRain: new Map([
					[
						10,
						{
							entry: gridEntry('p_rain_10min_x.png', 'p_rain', 10, P_RAIN, [3, 3]),
							image: await decoded(empty, P_RAIN)
						}
					]
				]),
				eta: {
					entry: gridEntry('eta_x.png', 'eta', null, ETA, [3, 3]),
					image: await decoded(empty, ETA)
				}
			};
		}

		/** A motion pair whose centre pixel holds (east, north); rest nodata. */
		async function motionPair(east: number | null, north: number | null) {
			const centre = (value: number | null) => [...Array(4).fill(null), value, ...Array(4).fill(null)];
			return {
				east: {
					entry: gridEntry('motion_east_kmh_x.png', 'motion_east_kmh', null, MOTION, [3, 3]),
					image: await decoded(centre(east), MOTION)
				},
				north: {
					entry: gridEntry('motion_north_kmh_x.png', 'motion_north_kmh', null, MOTION, [3, 3]),
					image: await decoded(centre(north), MOTION)
				}
			};
		}

		it('turns the two component grids into a bearing and a speed', async () => {
			// Moving 24 km/h east and 24 km/h north: out of the south-west.
			const grids = { ...(await baseGrids()), motion: await motionPair(24, 24) };
			const forecast = samplePoint(manifest, grids, 56.0, 10.5666);
			expect(forecast!.motion).not.toBeNull();
			expect(forecast!.motion!.compass).toBe('sw');
			expect(forecast!.motion!.bearingFromDeg).toBeCloseTo(225, 1);
			// Each component lands within half a quantisation step (0.47 km/h),
			// so the magnitude is inside one step of the truth.
			expect(Math.abs(forecast!.motion!.speedKmh - Math.sqrt(2 * 24 * 24))).toBeLessThanOrEqual(
				scaleOf(MOTION)
			);
		});

		it('reads a negative component correctly through the offset', async () => {
			// Due west at 40 km/h ⇒ coming from the east.
			const grids = { ...(await baseGrids()), motion: await motionPair(-40, 0) };
			const forecast = samplePoint(manifest, grids, 56.0, 10.5666);
			expect(forecast!.motion!.compass).toBe('e');
			expect(Math.abs(forecast!.motion!.speedKmh - 40)).toBeLessThanOrEqual(scaleOf(MOTION));
		});

		it('reports no motion where either component is nodata', async () => {
			for (const [east, north] of [
				[null, 24],
				[24, null],
				[null, null]
			] as [number | null, number | null][]) {
				const grids = { ...(await baseGrids()), motion: await motionPair(east, north) };
				expect(samplePoint(manifest, grids, 56.0, 10.5666)!.motion).toBeNull();
			}
		});

		it('reports no motion at a pixel the grids leave as nodata', async () => {
			// The neighbouring pixel is nodata in both components — what the
			// sidecar writes outside radar coverage, and everywhere on a cycle
			// whose composite holds no echo at all.
			const grids = { ...(await baseGrids()), motion: await motionPair(24, 24) };
			expect(samplePoint(manifest, grids, 56.017, 10.5666)!.motion).toBeNull();
		});

		it('degrades to no motion when the cycle served no motion grids', async () => {
			const forecast = samplePoint(manifest, await baseGrids(), 56.0, 10.5666);
			expect(forecast).not.toBeNull();
			expect(forecast!.motion).toBeNull();
			// And the rest of the forecast is unaffected.
			expect(forecast!.perLead).toHaveLength(2);
		});
	});

	/**
	 * The observed rain field: what the radar measured over the point, as
	 * opposed to what the ensemble predicts for it. The panel leads with it,
	 * so the three ways it can be absent all have to read as "unknown" — a
	 * null that says "dry" would put "no rain expected" over a downpour.
	 */
	describe('observed rain', () => {
		/** p_rain and eta all nodata; this block is about the observation. */
		async function baseGrids() {
			const empty = Array(9).fill(null);
			return {
				pRain: new Map([
					[
						10,
						{
							entry: gridEntry('p_rain_10min_x.png', 'p_rain', 10, P_RAIN, [3, 3]),
							image: await decoded(empty, P_RAIN)
						}
					]
				]),
				eta: {
					entry: gridEntry('eta_x.png', 'eta', null, ETA, [3, 3]),
					image: await decoded(empty, ETA)
				}
			};
		}

		/** An observation grid whose centre pixel holds `mmH`; rest nodata. */
		async function observedGrid(mmH: number | null) {
			return {
				entry: gridEntry('observed_mm_h_x.png', 'observed_mm_h', 0, OBSERVED, [3, 3]),
				image: await decoded([...Array(4).fill(null), mmH, ...Array(4).fill(null)], OBSERVED)
			};
		}

		it('reads the measured rain rate at a wet pixel', async () => {
			const grids = { ...(await baseGrids()), observed: await observedGrid(1.8) };
			const forecast = samplePoint(manifest, grids, 56.0, 10.5666);
			// 0–100 mm/h over 254 levels: half a step is 0.2 mm/h.
			expect(forecast!.observedMmH).not.toBeNull();
			expect(Math.abs((forecast!.observedMmH as number) - 1.8)).toBeLessThanOrEqual(
				scaleOf(OBSERVED) / 2
			);
			// A measured zero is a real answer — dry, not unknown.
			const dry = { ...(await baseGrids()), observed: await observedGrid(0) };
			expect(samplePoint(manifest, dry, 56.0, 10.5666)!.observedMmH).toBe(0);
		});

		it('reports null at a nodata pixel rather than a rain rate', async () => {
			const grids = { ...(await baseGrids()), observed: await observedGrid(null) };
			expect(samplePoint(manifest, grids, 56.0, 10.5666)!.observedMmH).toBeNull();
			// And at a pixel the grid leaves nodata while its centre is wet.
			const wet = { ...(await baseGrids()), observed: await observedGrid(1.8) };
			expect(samplePoint(manifest, wet, 56.017, 10.5666)!.observedMmH).toBeNull();
		});

		it('degrades to null when the cycle served no observation grid', async () => {
			const forecast = samplePoint(manifest, await baseGrids(), 56.0, 10.5666);
			expect(forecast).not.toBeNull();
			expect(forecast!.observedMmH).toBeNull();
			// A manifest predating the product costs this one field, no more.
			expect(forecast!.perLead).toHaveLength(2);
			expect(forecast!.source).toBe('client');
		});

		it('finds the artifact at lead 0, and nothing on a manifest without one', () => {
			const entry = gridEntry('observed_mm_h_x.png', 'observed_mm_h', 0, OBSERVED, [3, 3]);
			expect(observedArtifact({ ...manifest, artifacts: [entry] })).toBe(entry);
			expect(observedArtifact(manifest)).toBeNull();
			// Everything else in a cycle is left alone by the lookup.
			const others = [gridEntry('eta_x.png', 'eta', null, ETA, [3, 3])];
			expect(observedArtifact({ ...manifest, artifacts: others })).toBeNull();
		});
	});

	/**
	 * The advected rain field: what the loop draws, sampled at the point. The
	 * headline is read from this series, so two things have to hold — it is
	 * sampled at the *same pixel* as everything else (a one-pixel drift would
	 * put the sentence and the picture on different places), and a cycle
	 * without it yields an empty series rather than a dry one.
	 */
	describe('advected rain series', () => {
		/** p_rain and eta all nodata; this block is about the series. */
		async function baseGrids() {
			const empty = Array(9).fill(null);
			return {
				pRain: new Map([
					[
						10,
						{
							entry: gridEntry('p_rain_10min_x.png', 'p_rain', 10, P_RAIN, [3, 3]),
							image: await decoded(empty, P_RAIN)
						}
					]
				]),
				eta: {
					entry: gridEntry('eta_x.png', 'eta', null, ETA, [3, 3]),
					image: await decoded(empty, ETA)
				}
			};
		}

		const GENERATED = Date.parse('2026-08-28T12:01:30+00:00');
		const validAt = (leadMin: number) => new Date(GENERATED + leadMin * 60_000).toISOString();

		/** A forecast-field entry for one lead, stamped the way the sidecar does. */
		const seriesEntry = (leadMin: number): ArtifactEntry => ({
			...gridEntry(`forecast_mm_h_${leadMin}min_x.png`, 'forecast_mm_h', leadMin, FORECAST, [
				3, 3
			]),
			kind: 'forecast',
			valid_ts_utc: validAt(leadMin)
		});

		/**
		 * A series whose centre pixel walks `centreMmH` lead by lead, with the
		 * ring around it left nodata — so a sample from the wrong pixel is a
		 * null, not a plausible number.
		 */
		async function forecastSeries(centreMmH: (number | null)[]) {
			return Promise.all(
				centreMmH.map(async (mmH, i) => ({
					entry: seriesEntry(i * 10),
					image: await decoded([...Array(4).fill(null), mmH, ...Array(4).fill(null)], FORECAST)
				}))
			);
		}

		it('samples the series at the same pixel as everything else', async () => {
			const grids = {
				...(await baseGrids()),
				observed: {
					entry: gridEntry('observed_mm_h_x.png', 'observed_mm_h', 0, OBSERVED, [3, 3]),
					image: await decoded([...Array(4).fill(null), 1.8, ...Array(4).fill(null)], OBSERVED)
				},
				forecastSeries: await forecastSeries([1.6, 0.9, 0, null])
			};
			const forecast = samplePoint(manifest, grids, 56.0, 10.5666);
			expect(forecast!.rainSeries.map((s) => s.leadMin)).toEqual([0, 10, 20, 30]);
			expect(forecast!.rainSeries.map((s) => s.validTsUtc)).toEqual([
				validAt(0),
				validAt(10),
				validAt(20),
				validAt(30)
			]);
			const halfStep = scaleOf(FORECAST) / 2;
			expect(Math.abs((forecast!.rainSeries[0].mmH as number) - 1.6)).toBeLessThanOrEqual(halfStep);
			expect(Math.abs((forecast!.rainSeries[1].mmH as number) - 0.9)).toBeLessThanOrEqual(halfStep);
			expect(forecast!.rainSeries[2].mmH).toBe(0);
			// Nodata at a lead stays null: unknown, never dry.
			expect(forecast!.rainSeries[3].mmH).toBeNull();
			// The same pixel the observation came from — this is the whole point.
			expect(forecast!.observedMmH).not.toBeNull();

			// One pixel north, everything is nodata in both products alike.
			const neighbour = samplePoint(manifest, grids, 56.017, 10.5666);
			expect(neighbour!.rainSeries.every((s) => s.mmH === null)).toBe(true);
			expect(neighbour!.observedMmH).toBeNull();
		});

		it('is empty when the cycle served no such grids', async () => {
			const forecast = samplePoint(manifest, await baseGrids(), 56.0, 10.5666);
			expect(forecast!.rainSeries).toEqual([]);
			// An older sidecar costs the series and nothing else.
			expect(forecast!.perLead).toHaveLength(2);
			expect(forecast!.source).toBe('client');

			// An explicitly empty array reads the same way.
			const empty = { ...(await baseGrids()), forecastSeries: [] };
			expect(samplePoint(manifest, empty, 56.0, 10.5666)!.rainSeries).toEqual([]);
		});

		it('collects the artifacts in ascending lead, dropping unstamped ones', () => {
			const stamped = [seriesEntry(20), seriesEntry(0), seriesEntry(10)];
			const unstamped = { ...seriesEntry(30), valid_ts_utc: undefined };
			const blank = { ...seriesEntry(40), valid_ts_utc: '  ' };
			const found = forecastSeriesArtifacts({
				...manifest,
				artifacts: [...stamped, unstamped, blank, gridEntry('eta_x.png', 'eta', null, ETA, [3, 3])]
			});
			expect(found.map((a) => a.lead_min)).toEqual([0, 10, 20]);
		});

		it('finds nothing on a manifest written before the product existed', () => {
			expect(forecastSeriesArtifacts(manifest)).toEqual([]);
		});
	});
});
