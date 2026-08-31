/**
 * The loop's timeline, built from a schema-v2 manifest.
 *
 * Three things are pinned here, all of them things a viewer would be misled
 * by if they drifted:
 *
 *  1. Order and identity. History frames arrive with negative leads and must
 *     land before "now"; forecast frames after it. Nothing may be labelled a
 *     measurement that is an extrapolation, or the other way round.
 *  2. Times. Every displayed time comes from `valid_ts_utc`, which is already
 *     frame-age corrected — `radar_ts + lead` is *not* the same instant, and
 *     is only the fallback for a manifest that omits the field.
 *  3. Cold start. 0 history frames is a normal manifest, not an error: the
 *     track then simply starts at "now".
 */
import { describe, expect, it } from 'vitest';
import type { ArtifactEntry, Manifest } from './manifest';
import {
	buildTimeline,
	clampIndex,
	frameDelayMs,
	isBuffering,
	nextFrameIndex,
	nowIndex,
	timelineGeometry
} from './timeline';

const RADAR_TS = '2026-08-28T12:30:00+00:00';
/** The cycle's frame age: a forecast's validity is radar + this + lead. */
const FRAME_AGE_MIN = 2.5;

const at = (minutesFromRadar: number) =>
	new Date(Date.parse(RADAR_TS) + minutesFromRadar * 60_000).toISOString();

const overlay = (
	filename: string,
	leadMin: number,
	kind: 'observation' | 'forecast',
	validTsUtc: string
): ArtifactEntry => ({
	filename,
	product: 'overlay',
	lead_min: leadMin,
	kind,
	valid_ts_utc: validTsUtc,
	encoding: 'rgba8',
	shape: [1728, 1984]
});

/** A grayscale product entry, to prove the filter ignores non-overlays. */
const grid = (filename: string, product: ArtifactEntry['product']): ArtifactEntry => ({
	filename,
	product,
	lead_min: null,
	encoding: 'grayscale8',
	scale: 0.1,
	offset: 0,
	nodata: 255,
	shape: [432, 496]
});

/**
 * A v2 manifest with two history frames, "now", and two forecast leads —
 * artifacts deliberately out of order, because the manifest's own ordering is
 * not something the client may rely on.
 */
function manifestV2(history: number[] = [-20, -10]): Manifest {
	const artifacts: ArtifactEntry[] = [
		grid('p_rain_10min_202608281230.png', 'p_rain'),
		overlay(
			'overlay_10min_202608281230.png',
			10,
			'forecast',
			at(FRAME_AGE_MIN + 10)
		),
		overlay('overlay_now_202608281230.png', 0, 'observation', RADAR_TS),
		grid('motion_east_kmh_202608281230.png', 'motion_east_kmh'),
		overlay(
			'overlay_20min_202608281230.png',
			20,
			'forecast',
			at(FRAME_AGE_MIN + 20)
		),
		...history.map((lead) =>
			overlay(`overlay_now_history${-lead}.png`, lead, 'observation', at(lead))
		)
	];
	return {
		schema_version: 2,
		cycle: '202608281230',
		radar_ts_utc: RADAR_TS,
		generated_at_utc: '2026-08-28T12:33:00+00:00',
		threshold_mm_h: 0.1,
		timestep_min: 5,
		frame_age_min: FRAME_AGE_MIN,
		n_members: 24,
		leads_min: [10, 20],
		grid: {
			proj4: '+proj=stere +lat_0=56 +lon_0=10.5666 +lat_ts=56 +ellps=WGS84 +units=m +no_defs',
			x_ul_m: -496000,
			y_ul_m: 432000,
			pixel_scale_x_m: 2000,
			pixel_scale_y_m: 2000,
			shape: [432, 496],
			downsample_factor: 4
		},
		overlay_grid: null,
		motion: {
			grid: 'product',
			support_radius_km: 20,
			max_abs_kmh: 120,
			convention: 'east- and north-positive; nodata means no estimate'
		},
		calibration: null,
		artifacts
	};
}

describe('buildTimeline', () => {
	it('runs oldest observation → now → forecast, whatever order the manifest lists', () => {
		const frames = buildTimeline(manifestV2());
		expect(frames.map((f) => f.leadMin)).toEqual([-20, -10, 0, 10, 20]);
		expect(frames.map((f) => f.kind)).toEqual([
			'observation',
			'observation',
			'observation',
			'forecast',
			'forecast'
		]);
		expect(frames.map((f) => f.isNow)).toEqual([false, false, true, false, false]);
	});

	it('takes every displayed time from valid_ts_utc, frame age included', () => {
		const frames = buildTimeline(manifestV2());
		expect(frames.map((f) => f.validTsUtc)).toEqual([
			at(-20),
			at(-10),
			RADAR_TS,
			at(FRAME_AGE_MIN + 10),
			at(FRAME_AGE_MIN + 20)
		]);
		// The +10 frame is valid at 12:42:30, not at 12:40 — the difference the
		// frame-age correction exists for.
		expect(frames[3].validTsUtc).toBe('2026-08-28T12:42:30.000Z');
	});

	it('ignores the grayscale product grids entirely', () => {
		const frames = buildTimeline(manifestV2());
		expect(frames.every((f) => f.filename.startsWith('overlay'))).toBe(true);
	});

	it('handles a cold start with no history at all', () => {
		const frames = buildTimeline(manifestV2([]));
		expect(frames.map((f) => f.leadMin)).toEqual([0, 10, 20]);
		expect(frames[0].isNow).toBe(true);
		const geometry = timelineGeometry(frames);
		expect(geometry.historyCount).toBe(0);
		expect(geometry.nowIndex).toBe(0);
		expect(geometry.nowPosition).toBe(0);
	});

	it('handles a single history frame (a cycle after one prior cycle)', () => {
		const frames = buildTimeline(manifestV2([-10]));
		expect(frames.map((f) => f.leadMin)).toEqual([-10, 0, 10, 20]);
		expect(timelineGeometry(frames).historyCount).toBe(1);
	});

	it('is empty for no manifest and for a manifest with no overlays', () => {
		expect(buildTimeline(null)).toEqual([]);
		const noOverlays = { ...manifestV2(), artifacts: [grid('eta_x.png', 'eta')] };
		expect(buildTimeline(noOverlays)).toEqual([]);
	});

	it('falls back to radar_ts + lead only where v2 fields are missing', () => {
		// A v1-shaped manifest: no `kind`, no `valid_ts_utc`, no history.
		const legacy: Manifest = {
			...manifestV2([]),
			schema_version: 1,
			motion: null,
			artifacts: [
				{
					filename: 'overlay_10min_x.png',
					product: 'overlay',
					lead_min: 10,
					encoding: 'rgba8',
					shape: [1728, 1984]
				},
				{
					filename: 'overlay_now_x.png',
					product: 'overlay',
					lead_min: 0,
					encoding: 'rgba8',
					shape: [1728, 1984]
				}
			]
		};
		const frames = buildTimeline(legacy);
		expect(frames.map((f) => f.leadMin)).toEqual([0, 10]);
		// Kind inferred from the sign of the lead — the only evidence there is.
		expect(frames.map((f) => f.kind)).toEqual(['observation', 'forecast']);
		expect(frames[0].validTsUtc).toBe(new Date(RADAR_TS).toISOString());
		// Uncorrected on purpose: this path has no frame age to apply.
		expect(frames[1].validTsUtc).toBe(at(10));
	});
});

describe('nowIndex', () => {
	it('is the lead-0 frame when there is one', () => {
		expect(nowIndex(buildTimeline(manifestV2()))).toBe(2);
	});

	it('falls back to the newest observation', () => {
		const frames = [
			{ leadMin: -20, kind: 'observation' as const },
			{ leadMin: -10, kind: 'observation' as const },
			{ leadMin: 10, kind: 'forecast' as const }
		];
		expect(nowIndex(frames)).toBe(1);
	});

	it('is −1 when a cycle served nothing observed', () => {
		expect(nowIndex([{ leadMin: 10, kind: 'forecast' as const }])).toBe(-1);
		expect(nowIndex([])).toBe(-1);
	});
});

describe('timelineGeometry', () => {
	it('splits the track into history, now and forecast', () => {
		const geometry = timelineGeometry(buildTimeline(manifestV2()));
		expect(geometry.count).toBe(5);
		expect(geometry.historyCount).toBe(2);
		expect(geometry.forecastCount).toBe(2);
		expect(geometry.nowIndex).toBe(2);
		expect(geometry.nowPosition).toBeCloseTo(0.5, 12);
	});

	it('spaces one tick per frame across the whole track', () => {
		const geometry = timelineGeometry(buildTimeline(manifestV2()));
		expect(geometry.positions).toEqual([0, 0.25, 0.5, 0.75, 1]);
	});

	it('does not divide by zero on a one-frame or empty timeline', () => {
		const one = timelineGeometry(buildTimeline(manifestV2([])).slice(0, 1));
		expect(one.positions).toEqual([0]);
		const none = timelineGeometry([]);
		expect(none.positions).toEqual([]);
		expect(none.nowPosition).toBeNull();
		expect(none.count).toBe(0);
	});
});

describe('loaded flags and buffering', () => {
	it('calls every frame past the loaded prefix a buffering frame', () => {
		// Three of five frames have arrived.
		expect([0, 1, 2, 3, 4].map((i) => isBuffering(i, 3))).toEqual([
			false,
			false,
			false,
			true,
			true
		]);
	});

	it('treats a cycle with nothing loaded yet as buffering at every index', () => {
		expect(isBuffering(0, 0)).toBe(true);
	});
});

describe('clampIndex', () => {
	it('keeps a seek inside the track', () => {
		expect(clampIndex(-3, 5)).toBe(0);
		expect(clampIndex(9, 5)).toBe(4);
		expect(clampIndex(2, 5)).toBe(2);
	});

	it('rounds a fractional index and survives an empty track', () => {
		expect(clampIndex(2.6, 5)).toBe(3);
		expect(clampIndex(3, 0)).toBe(0);
		expect(clampIndex(NaN, 5)).toBe(0);
	});
});

describe('nextFrameIndex', () => {
	it('walks the loaded frames and wraps at the end of what has arrived', () => {
		expect(nextFrameIndex(0, 5)).toBe(1);
		expect(nextFrameIndex(3, 5)).toBe(4);
		expect(nextFrameIndex(4, 5)).toBe(0);
	});

	it('wraps at the loaded count while a cycle is still streaming in', () => {
		// Five frames in the manifest, two on screen so far.
		expect(nextFrameIndex(1, 2)).toBe(0);
	});

	it('holds position on a buffering frame instead of jumping away', () => {
		expect(nextFrameIndex(4, 2)).toBe(4);
		expect(nextFrameIndex(0, 0)).toBe(0);
	});
});

describe('frameDelayMs', () => {
	it('holds the last loaded frame longer than the rest', () => {
		expect(frameDelayMs(0, 5, 550, 1400)).toBe(550);
		expect(frameDelayMs(4, 5, 550, 1400)).toBe(1400);
		// Mid-load, the end of the loop is wherever loading has got to.
		expect(frameDelayMs(1, 2, 550, 1400)).toBe(1400);
	});

	it('uses the plain interval when nothing is loaded', () => {
		expect(frameDelayMs(0, 0, 550, 1400)).toBe(550);
	});
});
