/**
 * The station scatter's two pieces of logic: where a station lands on the
 * map, and what its colour is actually measuring. The second is the one that
 * can mislead — a dot coloured by a Brier score is not saying the same thing
 * as a dot coloured by a hit rate, and the code has to keep track of which is
 * which so the legend can say so.
 */
import { describe, expect, it } from 'vitest';
import {
	DENMARK_PATH,
	MAP_HEIGHT,
	MAP_WIDTH,
	plotStations,
	project,
	stationScore
} from './stations';
import type { StationFeature, StationProperties } from './schema';

const props = (overrides: Partial<StationProperties> = {}): StationProperties => ({
	station_id: '06181',
	name: 'Københavns Lufthavn',
	kind: 'Synop',
	n_events: 199,
	brier_gauge: 0.106,
	warn_pod: 0.79,
	warn_far: 0.23,
	warnings: 19,
	raining_now_agreement: 0.9,
	...overrides
});

const station = (
	lon: number,
	lat: number,
	overrides: Partial<StationProperties> = {}
): StationFeature => ({
	type: 'Feature',
	geometry: { type: 'Point', coordinates: [lon, lat] },
	properties: props(overrides)
});

describe('the projection', () => {
	it('puts Skagen above Gedser and Bornholm to the right of Blåvand', () => {
		const skagen = project(10.63, 57.74);
		const gedser = project(11.97, 54.57);
		const bornholm = project(14.78, 55.3);
		const blaavand = project(8.08, 55.56);
		expect(skagen.y).toBeLessThan(gedser.y);
		expect(bornholm.x).toBeGreaterThan(blaavand.x);
	});

	it('fills the frame it declares', () => {
		const topLeft = project(7.7, 57.9);
		const bottomRight = project(15.35, 54.45);
		expect(topLeft.x).toBeCloseTo(0, 6);
		expect(topLeft.y).toBeCloseTo(0, 6);
		expect(bottomRight.x).toBeCloseTo(MAP_WIDTH, 6);
		expect(bottomRight.y).toBeCloseTo(MAP_HEIGHT, 0);
	});

	it('has an outline to draw the stations on', () => {
		expect(DENMARK_PATH.startsWith('M')).toBe(true);
		expect(DENMARK_PATH).toContain('Z');
	});
});

describe('the colour scale', () => {
	it('bands a station by how much of the rain it warned about', () => {
		expect(stationScore(props({ warn_pod: 0.86 }))).toMatchObject({ basis: 'pod', band: 'best' });
		expect(stationScore(props({ warn_pod: 0.72 }))).toMatchObject({ basis: 'pod', band: 'good' });
		expect(stationScore(props({ warn_pod: 0.58 }))).toMatchObject({ basis: 'pod', band: 'fair' });
		expect(stationScore(props({ warn_pod: 0.31 }))).toMatchObject({ basis: 'pod', band: 'poor' });
	});

	it('falls back to the Brier score, and says that it did', () => {
		const good = stationScore(props({ warn_pod: null, brier_gauge: 0.07 }));
		expect(good.basis).toBe('brier');
		expect(good.band).toBe('best');
		const poor = stationScore(props({ warn_pod: null, brier_gauge: 0.22 }));
		expect(poor.basis).toBe('brier');
		expect(poor.band).toBe('poor');
		// Lower is better for a Brier score; the scale must not read it as a rate.
		expect(good.value!).toBeGreaterThan(poor.value!);
	});

	it('has no colour at all for a station with no score', () => {
		const none = stationScore(props({ warn_pod: null, brier_gauge: null }));
		expect(none).toEqual({ value: null, basis: null, band: 'unknown' });
	});

	it('clamps rather than trusting an out-of-range number', () => {
		expect(stationScore(props({ warn_pod: 1.4 })).value).toBe(1);
		expect(stationScore(props({ warn_pod: null, brier_gauge: 0.9 })).value).toBe(0);
	});
});

describe('plotStations', () => {
	it('projects and scores each station', () => {
		const plotted = plotStations([station(12.64, 55.61), station(8.13, 56.0, { warn_pod: 0.4 })]);
		expect(plotted).toHaveLength(2);
		expect(plotted[0].x).toBeGreaterThan(plotted[1].x);
		expect(plotted[1].score.band).toBe('poor');
	});

	it('drops a station that is not in the frame', () => {
		expect(plotStations([station(-51.7, 64.2)])).toEqual([]);
		expect(plotStations([])).toEqual([]);
	});
});
