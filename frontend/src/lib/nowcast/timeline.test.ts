/**
 * The loop's timeline, built from a schema-v2 manifest.
 *
 * Four things are pinned here, all of them things a viewer would be misled
 * by if they drifted:
 *
 *  1. Order and identity. History frames arrive with negative leads and must
 *     land before the latest observation; forecast frames after it, including
 *     the second lead-0 frame schema v2 adds — the radar field advected
 *     forward by its own age. Nothing may be labelled a measurement that is an
 *     extrapolation, or the other way round.
 *  2. Times. Every displayed time comes from `valid_ts_utc`, which is already
 *     frame-age corrected — `radar_ts + lead` is *not* the same instant, and
 *     is only the fallback for a manifest that omits the field.
 *  3. Cold start. 0 history frames is a normal manifest, not an error: the
 *     track then simply starts at the latest observation.
 *  4. Two marks, not one. The latest observation is the hinge between measured
 *     and extrapolated; wall-clock now is somewhere to the right of it, 14–24
 *     min later, and has its own position on the track.
 */
import { describe, expect, it } from 'vitest';
import { overlayFrames, type ArtifactEntry, type Manifest } from './manifest';
import {
	buildTimeline,
	clampIndex,
	clockPosition,
	frameDelayMs,
	isBuffering,
	latestIndex,
	nextFrameIndex,
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
 * A v2 manifest with two history frames, the latest observation, and two
 * forecast leads — artifacts deliberately out of order, because the manifest's
 * own ordering is not something the client may rely on. `lead0Forecast` adds
 * the second lead-0 frame: the radar field advected forward by its own age,
 * valid at `generated_at_utc`, and listed *before* the observation it ties
 * with so the tie-break has something to do.
 */
function manifestV2(
	history: number[] = [-20, -10],
	{ lead0Forecast = false }: { lead0Forecast?: boolean } = {}
): Manifest {
	const artifacts: ArtifactEntry[] = [
		grid('p_rain_10min_202608281230.png', 'p_rain'),
		overlay(
			'overlay_10min_202608281230.png',
			10,
			'forecast',
			at(FRAME_AGE_MIN + 10)
		),
		...(lead0Forecast
			? [overlay('overlay_0min_202608281230.png', 0, 'forecast', at(FRAME_AGE_MIN))]
			: []),
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
			support_radius_km: null,
			fill: 'nearest-cells-v1',
			fill_scales_km: [25, 50, 100],
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
		expect(frames.map((f) => f.isLatest)).toEqual([false, false, true, false, false]);
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
		expect(frames[0].isLatest).toBe(true);
		const geometry = timelineGeometry(frames);
		expect(geometry.historyCount).toBe(0);
		expect(geometry.latestIndex).toBe(0);
		expect(geometry.latestPosition).toBe(0);
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

describe('latestIndex', () => {
	it('is the lead-0 observation when there is one', () => {
		expect(latestIndex(buildTimeline(manifestV2()))).toBe(2);
	});

	/**
	 * Schema v2 puts two frames at lead 0: the radar image, and the same field
	 * advected forward by its own age. Only the first is a measurement, and
	 * only the first may be the hinge — calling the extrapolation "the latest
	 * observation" would hatch a forecast frame as history.
	 */
	it('never picks the lead-0 forecast over the lead-0 observation', () => {
		const frames = buildTimeline(manifestV2([-10], { lead0Forecast: true }));
		expect(frames.map((f) => f.leadMin)).toEqual([-10, 0, 0, 10, 20]);
		expect(frames.map((f) => f.kind)).toEqual([
			'observation',
			'observation',
			'forecast',
			'forecast',
			'forecast'
		]);
		expect(latestIndex(frames)).toBe(1);
		expect(frames.map((f) => f.isLatest)).toEqual([false, true, false, false, false]);
		// And the hatched past therefore stops at the measurement.
		expect(timelineGeometry(frames).latestPosition).toBeCloseTo(0.25, 12);
	});

	it('takes the lead-0 forecast as the hinge under no circumstances', () => {
		// A pathological cycle: a lead-0 forecast and no observation at all.
		expect(
			latestIndex([
				{ leadMin: 0, kind: 'forecast' as const },
				{ leadMin: 10, kind: 'forecast' as const }
			])
		).toBe(-1);
	});

	it('falls back to the newest observation', () => {
		const frames = [
			{ leadMin: -20, kind: 'observation' as const },
			{ leadMin: -10, kind: 'observation' as const },
			{ leadMin: 10, kind: 'forecast' as const }
		];
		expect(latestIndex(frames)).toBe(1);
	});

	it('is −1 when a cycle served nothing observed', () => {
		expect(latestIndex([{ leadMin: 10, kind: 'forecast' as const }])).toBe(-1);
		expect(latestIndex([])).toBe(-1);
	});
});

describe('timelineGeometry', () => {
	it('splits the track into history, the latest observation and forecast', () => {
		const geometry = timelineGeometry(buildTimeline(manifestV2()));
		expect(geometry.count).toBe(5);
		expect(geometry.historyCount).toBe(2);
		expect(geometry.forecastCount).toBe(2);
		expect(geometry.latestIndex).toBe(2);
		expect(geometry.latestPosition).toBeCloseTo(0.5, 12);
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
		expect(none.latestPosition).toBeNull();
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

/**
 * The tie-break lives in manifest.ts but is only observable here, where the
 * frames become a timeline: two entries share lead 0 since schema v2, and
 * which of them comes first decides where the hatched past stops.
 */
describe('overlayFrames', () => {
	it('puts the observation before the forecast at the same lead', () => {
		const entries = overlayFrames(manifestV2([-10], { lead0Forecast: true }));
		expect(entries.map((e) => [e.lead_min, e.kind])).toEqual([
			[-10, 'observation'],
			[0, 'observation'],
			[0, 'forecast'],
			[10, 'forecast'],
			[20, 'forecast']
		]);
	});

	it('does not depend on the order the manifest happened to list them in', () => {
		const manifest = manifestV2([-10], { lead0Forecast: true });
		const reversed = { ...manifest, artifacts: [...manifest.artifacts].reverse() };
		expect(overlayFrames(reversed).map((e) => [e.lead_min, e.kind])).toEqual(
			overlayFrames(manifest).map((e) => [e.lead_min, e.kind])
		);
	});
});

/**
 * Wall-clock now on the track. It is a different mark from the latest
 * observation and usually minutes to the right of it: the newest composite is
 * 14–24 min old whenever anyone looks.
 */
describe('clockPosition', () => {
	/** Frames every ten minutes from the radar time: five of them, 0 … 1. */
	const evenFrames = [-20, -10, 0, 10, 20].map((lead) => ({ validTsUtc: at(lead) }));
	const clock = (minutesFromRadar: number) => Date.parse(RADAR_TS) + minutesFromRadar * 60_000;

	it('interpolates between the frames bracketing now', () => {
		// Exactly on a frame: that frame's own tick position.
		expect(clockPosition(evenFrames, clock(0))).toBeCloseTo(0.5, 12);
		expect(clockPosition(evenFrames, clock(-10))).toBeCloseTo(0.25, 12);
		// Halfway between the 0 and +10 frames, which are 0.5 and 0.75 apart.
		expect(clockPosition(evenFrames, clock(5))).toBeCloseTo(0.625, 12);
		// A tenth of the way along the last interval.
		expect(clockPosition(evenFrames, clock(11))).toBeCloseTo(0.775, 12);
	});

	it('interpolates in time, not in index, on unevenly spaced frames', () => {
		// Two five-minute steps then a twenty-minute one: three intervals, so
		// each is a third of the track however many minutes it spans.
		const uneven = [0, 5, 10, 30].map((lead) => ({ validTsUtc: at(lead) }));
		expect(clockPosition(uneven, clock(5))).toBeCloseTo(1 / 3, 12);
		// Quarter of the way through the long last interval.
		expect(clockPosition(uneven, clock(15))).toBeCloseTo(2 / 3 + 0.25 / 3, 12);
	});

	it('pins to the ends rather than running off the track', () => {
		expect(clockPosition(evenFrames, clock(-90))).toBe(0);
		expect(clockPosition(evenFrames, clock(90))).toBe(1);
		// The realistic case: a cycle whose forecast frames end before now.
		expect(clockPosition(evenFrames, clock(20))).toBe(1);
	});

	it('has nothing to say about a track it cannot interpolate on', () => {
		expect(clockPosition([{ validTsUtc: at(0) }], clock(0))).toBeNull();
		expect(clockPosition([], clock(0))).toBeNull();
		expect(clockPosition(evenFrames, Number.NaN)).toBeNull();
	});

	it('is null when any stamp will not parse, rather than guessing', () => {
		const broken = [{ validTsUtc: at(-10) }, { validTsUtc: 'not-a-timestamp' }];
		expect(clockPosition(broken, clock(0))).toBeNull();
	});

	it('sits well to the right of the latest observation on a real cycle', () => {
		// The incident's geometry: the newest composite 23 min old, the loop
		// running to +20. The two marks must not land on the same place.
		const frames = buildTimeline(manifestV2());
		const geometry = timelineGeometry(frames);
		const now = clockPosition(frames, Date.parse(RADAR_TS) + 23 * 60_000);
		expect(now).not.toBeNull();
		expect(now as number).toBeGreaterThan(geometry.latestPosition as number);
	});
});
