/**
 * Reading mm/h off the bundle's observation grid.
 *
 * The PNG is built here, byte by byte, rather than checked in: the decoder
 * is ours too, and a decoder bug would look exactly like a sampling bug.
 * The expected values are hand-computed from the manifest's own
 * quantisation — `value = level * scale + offset` — because that
 * arithmetic is the whole contract between this page and the sidecar, and
 * one wrong constant produces a plausible, wrong rain rate that nobody
 * could catch by eye.
 *
 * The other half of the contract is that **level 255 is not zero**. A
 * nodata pixel means "no measurement here"; rendering it as 0.0 mm/h would
 * turn a hole in the composite into evidence that it was dry, which is the
 * mistake this whole tool is built to avoid.
 */
import { deflateSync } from 'node:zlib';
import { describe, expect, it, vi } from 'vitest';
import { decodeGray8Png, type Gray8Image } from '$lib/nowcast/png';
import { parseManifest } from './load';
import type { Manifest } from './schema';
import fixture from './fixture.json';
import { observedArtifact, percentile, sampleDisc, sampleDiscP90, sampleObservedAt } from './sample';

// --- a minimal grayscale-8 PNG encoder -------------------------------------

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

/** Encode levels as the sidecar does: grayscale-8, non-interlaced. */
function encodeGray8Png(levels: Uint8Array, width: number, height: number): Uint8Array {
	const raw = new Uint8Array((width + 1) * height);
	for (let y = 0; y < height; y++) {
		raw[y * (width + 1)] = 0; // no filter
		for (let x = 0; x < width; x++) raw[y * (width + 1) + 1 + x] = levels[y * width + x];
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

// --- a 5×5 grid centred on the projection origin ---------------------------

const PROJ4 = fixture.manifest.grid.proj4;
/** The projection's own origin projects to (0, 0) — so the maths is checkable. */
const ORIGIN = { lat: 56, lon: 10.5666 };
const PIXEL_M = 2000;
const SCALE = fixture.manifest.frames.encodings.observed.scale;
const NODATA = 255;

/**
 * Levels of the 5×5 test grid. The centre pixel (row 2, col 2) is the
 * projection origin; the four pixels around it are the rest of a 2 km disc.
 *
 *        col 1   col 2   col 3
 * row 1          5
 * row 2    0     10      20
 * row 3          255  (nodata)
 */
const LEVELS = (() => {
	const levels = new Uint8Array(25).fill(NODATA);
	levels[2 * 5 + 2] = 10;
	levels[1 * 5 + 2] = 5;
	levels[2 * 5 + 1] = 0;
	levels[2 * 5 + 3] = 20;
	levels[3 * 5 + 2] = NODATA;
	return levels;
})();

const manifestFor = (shape: [number, number] = [5, 5]): Manifest =>
	parseManifest({
		...JSON.parse(JSON.stringify(fixture.manifest)),
		grid: {
			proj4: PROJ4,
			x_ul_m: -2 * PIXEL_M,
			y_ul_m: 2 * PIXEL_M,
			pixel_scale_x_m: PIXEL_M,
			pixel_scale_y_m: PIXEL_M,
			shape,
			downsample_factor: 4
		}
	})!;

let image: Gray8Image;

const decoded = async (): Promise<Gray8Image> => {
	image ??= await decodeGray8Png(encodeGray8Png(LEVELS, 5, 5));
	return image;
};

describe('observedArtifact', () => {
	it('builds an entry the existing sampler can use', () => {
		const entry = observedArtifact(manifestFor())!;
		expect(entry.encoding).toBe('grayscale8');
		expect(entry.scale).toBe(SCALE);
		expect(entry.offset).toBe(fixture.manifest.frames.encodings.observed.offset);
		expect(entry.nodata).toBe(NODATA);
		expect(entry.shape).toEqual([5, 5]);
	});

	it('is null when the bundle does not state the quantisation', () => {
		const raw = JSON.parse(JSON.stringify(fixture.manifest));
		delete raw.frames.encodings.observed.offset;
		expect(observedArtifact(parseManifest(raw)!)).toBeNull();
	});
});

describe('sampleObservedAt', () => {
	it('dequantises a pixel to a hand-computed mm/h', async () => {
		// level 10 × (100 / 254) mm/h per level = 3.937007874015748 mm/h
		const expected = 10 * (100 / 254);
		expect(expected).toBeCloseTo(3.937007874, 9);
		expect(await sampleAt(ORIGIN.lat, ORIGIN.lon)).toBeCloseTo(expected, 12);
	});

	it('reads the four neighbours as their own levels', async () => {
		// One pixel north is level 5; one pixel east is level 20.
		expect(await sampleAt(ORIGIN.lat + degLat(PIXEL_M), ORIGIN.lon)).toBeCloseTo(
			5 * SCALE,
			9
		);
		expect(await sampleAt(ORIGIN.lat, ORIGIN.lon + degLon(PIXEL_M, ORIGIN.lat))).toBeCloseTo(
			20 * SCALE,
			9
		);
	});

	it('reads level 255 as null — NOT as zero', async () => {
		const south = await sampleAt(ORIGIN.lat - degLat(PIXEL_M), ORIGIN.lon);
		expect(south).toBeNull();
		expect(south).not.toBe(0);
	});

	it('reads a real zero as zero', async () => {
		const west = await sampleAt(ORIGIN.lat, ORIGIN.lon - degLon(PIXEL_M, ORIGIN.lat));
		expect(west).toBe(0);
	});

	it('is null off the grid, which the UI must say in words', async () => {
		expect(await sampleAt(ORIGIN.lat + 5, ORIGIN.lon)).toBeNull();
	});

	it('warns and reads unknown when the PNG does not match the grid', async () => {
		const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
		const value = sampleObservedAt(await decoded(), manifestFor([9, 9]), ORIGIN.lat, ORIGIN.lon);
		expect(value).toBeNull();
		expect(warn).toHaveBeenCalled();
		warn.mockRestore();
	});
});

describe('sampleDisc', () => {
	it('computes the statistics the service computes, over the same disc', async () => {
		const disc = sampleDisc(await decoded(), manifestFor(), ORIGIN.lat, ORIGIN.lon, PIXEL_M)!;
		// radius_px = 2000 / 2000 = 1 → the five-pixel plus shape.
		expect(disc.nPixels).toBe(5);
		// Four of them carry a value; the fifth is nodata and is DROPPED,
		// never counted as a zero.
		expect(disc.nValid).toBe(4);

		const values = [0, 5, 10, 20].map((level) => level * SCALE);
		expect(disc.maxMmH).toBeCloseTo(Math.max(...values), 12);
		expect(disc.meanMmH).toBeCloseTo(values.reduce((a, b) => a + b, 0) / 4, 12);
		// numpy's linear percentile: position (4-1) × 0.9 = 2.7, between the
		// third and fourth sorted values.
		expect(disc.p90MmH).toBeCloseTo(10 * SCALE + 0.7 * (20 * SCALE - 10 * SCALE), 12);
		expect(disc.singlePixel).toBe(false);
	});

	it('says so when the disc is one pixel wide', async () => {
		const disc = sampleDisc(await decoded(), manifestFor(), ORIGIN.lat, ORIGIN.lon, 100)!;
		expect(disc.nPixels).toBe(1);
		expect(disc.singlePixel).toBe(true);
		expect(disc.p90MmH).toBeCloseTo(10 * SCALE, 12);
	});

	it('reports an all-nodata disc as measured-nothing, not as dry', async () => {
		const away = { lat: ORIGIN.lat - degLat(2 * PIXEL_M), lon: ORIGIN.lon };
		const disc = sampleDisc(await decoded(), manifestFor(), away.lat, away.lon, 100)!;
		expect(disc.nPixels).toBe(1);
		expect(disc.nValid).toBe(0);
		expect(disc.p90MmH).toBeNull();
		expect(disc.maxMmH).toBeNull();
	});

	it('is null off the grid and for a radius that is not one', async () => {
		expect(sampleDisc(await decoded(), manifestFor(), ORIGIN.lat + 5, ORIGIN.lon, 1000)).toBeNull();
		expect(sampleDisc(await decoded(), manifestFor(), ORIGIN.lat, ORIGIN.lon, 0)).toBeNull();
	});

	it('has a shorthand for the statistic the read-out leads with', async () => {
		const image = await decoded();
		expect(sampleDiscP90(image, manifestFor(), ORIGIN.lat, ORIGIN.lon, PIXEL_M)).toBeCloseTo(
			sampleDisc(image, manifestFor(), ORIGIN.lat, ORIGIN.lon, PIXEL_M)!.p90MmH!,
			12
		);
	});
});

describe('percentile', () => {
	it('matches numpy’s default linear interpolation', () => {
		// np.percentile([1, 2, 3, 4], 90) == 3.7
		expect(percentile([1, 2, 3, 4], 90)).toBeCloseTo(3.7, 12);
		// np.percentile([1, 2, 3, 4], 50) == 2.5
		expect(percentile([1, 2, 3, 4], 50)).toBeCloseTo(2.5, 12);
		expect(percentile([4, 1, 3, 2], 90)).toBeCloseTo(3.7, 12);
		expect(percentile([7], 90)).toBe(7);
		expect(percentile([], 90)).toBeNull();
	});

	it('does not reorder the caller’s array', () => {
		const values = [4, 1, 3, 2];
		percentile(values, 90);
		expect(values).toEqual([4, 1, 3, 2]);
	});
});

// --- helpers ---------------------------------------------------------------

const EARTH_KM = 6371.0088;
const KM_PER_DEG_LAT = (Math.PI * EARTH_KM) / 180;
const degLat = (metres: number) => metres / 1000 / KM_PER_DEG_LAT;
const degLon = (metres: number, lat: number) =>
	metres / 1000 / (KM_PER_DEG_LAT * Math.cos((lat * Math.PI) / 180));

async function sampleAt(lat: number, lon: number): Promise<number | null> {
	return sampleObservedAt(await decoded(), manifestFor(), lat, lon);
}
