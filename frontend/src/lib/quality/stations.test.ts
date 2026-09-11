/**
 * The station scatter's two pieces of logic: where a station lands on the
 * map, and what its colour is actually measuring. The second is the one that
 * can mislead — a dot coloured by a Brier score is not saying the same thing
 * as a dot coloured by a hit rate, and the code has to keep track of which is
 * which so the legend can say so.
 *
 * The hit rate is banded against the measured national rate, so the band
 * edges move with the data; the Brier fallback is banded on a fixed scale and
 * must not move with it. Both are pinned here, and so are the legend lines
 * the edges are printed into, because a legend that disagrees with the
 * colours is worse than no legend.
 */
import { describe, expect, it } from 'vitest';
import { da } from '$lib/i18n/da';
import { en } from '$lib/i18n/en';
import { countText, fractionText } from './sentences';
import {
	DENMARK_PATH,
	MAP_HEIGHT,
	MAP_WIDTH,
	plotStations,
	podBandEdges,
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

/** The measured national hit rate the map is banded against. */
const NATIONAL = 0.35;

describe('the band edges', () => {
	it('puts the middle two bands ten points either side of the national rate', () => {
		const [poor, middle, best] = podBandEdges(NATIONAL);
		expect(poor).toBeCloseTo(0.25, 10);
		expect(middle).toBeCloseTo(0.35, 10);
		expect(best).toBeCloseTo(0.45, 10);
	});

	it('clamps an edge that would fall off either end of the scale', () => {
		const [low, , high] = podBandEdges(0.04);
		expect(low).toBe(0);
		expect(high).toBeCloseTo(0.14, 10);
		const [, , top] = podBandEdges(0.95);
		expect(top).toBe(1);
		expect(podBandEdges(1.4)).toEqual([0.9, 1, 1]);
	});

	it('falls back to the fixed bands when there is no national rate', () => {
		expect(podBandEdges(null)).toEqual([0.5, 0.65, 0.8]);
		expect(podBandEdges(Number.NaN)).toEqual([0.5, 0.65, 0.8]);
	});
});

describe('the colour scale', () => {
	it('bands a station against the national hit rate, not against an aspiration', () => {
		// 0.58 was "fair" under the old fixed bands; against a national 35 % it
		// is one of the best stations in the country, and must read as one.
		expect(stationScore(props({ warn_pod: 0.58 }), NATIONAL)).toMatchObject({
			basis: 'pod',
			band: 'best'
		});
		expect(stationScore(props({ warn_pod: 0.4 }), NATIONAL)).toMatchObject({
			basis: 'pod',
			band: 'good'
		});
		expect(stationScore(props({ warn_pod: 0.31 }), NATIONAL)).toMatchObject({
			basis: 'pod',
			band: 'fair'
		});
		expect(stationScore(props({ warn_pod: 0.18 }), NATIONAL)).toMatchObject({
			basis: 'pod',
			band: 'poor'
		});
	});

	it('puts a band edge itself in the upper of the two bands it divides', () => {
		expect(stationScore(props({ warn_pod: 0.25 }), NATIONAL).band).toBe('fair');
		expect(stationScore(props({ warn_pod: 0.35 }), NATIONAL).band).toBe('good');
		expect(stationScore(props({ warn_pod: 0.45 }), NATIONAL).band).toBe('best');
	});

	it('leaves a band empty rather than cutting outside the scale', () => {
		// The poor edge clamps to 0, and no hit rate is below zero.
		expect(stationScore(props({ warn_pod: 0 }), 0.04).band).toBe('fair');
		// The best edge clamps to 1, which only a perfect station reaches.
		expect(stationScore(props({ warn_pod: 0.99 }), 0.95).band).toBe('good');
		expect(stationScore(props({ warn_pod: 1 }), 0.95).band).toBe('best');
	});

	it('uses the fixed bands when the national rate is missing', () => {
		expect(stationScore(props({ warn_pod: 0.86 }), null)).toMatchObject({
			basis: 'pod',
			band: 'best'
		});
		expect(stationScore(props({ warn_pod: 0.72 }), null)).toMatchObject({
			basis: 'pod',
			band: 'good'
		});
		expect(stationScore(props({ warn_pod: 0.58 }), null)).toMatchObject({
			basis: 'pod',
			band: 'fair'
		});
		expect(stationScore(props({ warn_pod: 0.31 }), null)).toMatchObject({
			basis: 'pod',
			band: 'poor'
		});
	});

	it('falls back to the Brier score, and says that it did', () => {
		const good = stationScore(props({ warn_pod: null, brier_gauge: 0.07 }), NATIONAL);
		expect(good.basis).toBe('brier');
		expect(good.band).toBe('best');
		const poor = stationScore(props({ warn_pod: null, brier_gauge: 0.22 }), NATIONAL);
		expect(poor.basis).toBe('brier');
		expect(poor.band).toBe('poor');
		// Lower is better for a Brier score; the scale must not read it as a rate.
		expect(good.value!).toBeGreaterThan(poor.value!);
	});

	it('never bands a Brier score against the hit rate — it is not a hit rate', () => {
		// 0.15 rescales to 0.5: "fair" on the fixed Brier scale, and it must
		// stay there whatever the national POD happens to be.
		const brier = props({ warn_pod: null, brier_gauge: 0.15 });
		expect(stationScore(brier, NATIONAL).value).toBeCloseTo(0.5, 10);
		expect(stationScore(brier, NATIONAL).band).toBe('fair');
		expect(stationScore(brier, null).band).toBe('fair');
		expect(stationScore(brier, 0.9).band).toBe('fair');
	});

	it('has no colour at all for a station with no score', () => {
		const none = stationScore(props({ warn_pod: null, brier_gauge: null }), NATIONAL);
		expect(none).toEqual({ value: null, basis: null, band: 'unknown' });
	});

	it('clamps rather than trusting an out-of-range number', () => {
		expect(stationScore(props({ warn_pod: 1.4 }), NATIONAL).value).toBe(1);
		expect(stationScore(props({ warn_pod: null, brier_gauge: 0.9 }), NATIONAL).value).toBe(0);
	});
});

describe('plotStations', () => {
	it('projects and scores each station', () => {
		const plotted = plotStations(
			[station(12.64, 55.61), station(8.13, 56.0, { warn_pod: 0.4 })],
			null
		);
		expect(plotted).toHaveLength(2);
		expect(plotted[0].x).toBeGreaterThan(plotted[1].x);
		expect(plotted[1].score.band).toBe('poor');
	});

	it('scores every dot against the same national rate', () => {
		const plotted = plotStations([station(8.13, 56.0, { warn_pod: 0.4 })], NATIONAL);
		expect(plotted[0].score.band).toBe('good');
	});

	it('drops a station that is not in the frame', () => {
		expect(plotStations([station(-51.7, 64.2)], NATIONAL)).toEqual([]);
		expect(plotStations([], NATIONAL)).toEqual([]);
	});
});

/**
 * The legend the way `QualityStationMap` assembles it: the low end of a range
 * is a bare number and the high end carries the unit, so the line reads
 * "25–35 %" and not "25 %–35 %".
 */
const legendLines = (
	catalog: typeof en,
	locale: 'da' | 'en',
	reference: number | null
): string[] => {
	const edges = podBandEdges(reference);
	const bare = (edge: number) => countText(edge * 100, locale);
	const withUnit = (edge: number) => fractionText(catalog, edge);
	const { stations } = catalog.quality;
	return [
		reference === null
			? stations.legendTitleAbsolute
			: stations.legendTitle(withUnit(reference)),
		stations.legendPoor(withUnit(edges[0])),
		stations.legendFair(bare(edges[0]), withUnit(edges[1])),
		stations.legendGood(bare(edges[1]), withUnit(edges[2])),
		stations.legendBest(withUnit(edges[2]))
	];
};

describe('the legend', () => {
	it('prints the national rate and the edges the dots were cut at', () => {
		expect(legendLines(en, 'en', NATIONAL)).toEqual([
			'Share of the rain we warned about, against the national 35%',
			'under 25%',
			'25–35%',
			'35–45%',
			'over 45%'
		]);
	});

	it('spaces the Danish percent the way Danish does', () => {
		expect(legendLines(da, 'da', NATIONAL)).toEqual([
			'Andel af regnen vi varslede, målt mod landsgennemsnittet på 35 %',
			'under 25 %',
			'25–35 %',
			'35–45 %',
			'over 45 %'
		]);
	});

	it('says the bands are absolute when there is no national rate to compare with', () => {
		const [title, poor, , , best] = legendLines(en, 'en', null);
		expect(title).toBe('Share of the rain we warned about');
		expect(title).not.toContain('national');
		expect(poor).toBe('under 50%');
		expect(best).toBe('over 80%');
	});

	it('keeps the line that says a ring is not a hit rate', () => {
		expect(en.quality.stations.legendBrier).toContain('Brier');
		expect(da.quality.stations.legendBrier).toContain('Brier');
	});
});
