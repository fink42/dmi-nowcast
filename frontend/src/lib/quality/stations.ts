/**
 * The station scatter: where the gauges are, and how well the warnings did at
 * each of them.
 *
 * The map is a plain equirectangular projection of lon/lat, drawn over the
 * coarse Denmark outline in `./denmark.ts`. It is deliberately not a second
 * MapLibre instance: this is a content page, the picture is twelve dots on a
 * country, and a basemap here would cost a megabyte of tiles and a WebGL
 * context to say nothing extra. The projection is one function, so the
 * outline and the dots cannot drift apart.
 *
 * The colour is `warn_pod` — of the rain that arrived at this station, how
 * much we warned about — falling back to `brier_gauge` where a station has
 * too few warnings for a POD, and to "no score" where it has neither. The
 * fallback is stated in the legend rather than hidden: a dot coloured by a
 * different measurement is not the same dot.
 */
import { DENMARK_OUTLINE } from './denmark';
import type { StationFeature, StationProperties } from './schema';

/** The frame the map is drawn in, in degrees. Denmark plus a little air. */
export const MAP_BOUNDS = { minLon: 7.7, maxLon: 15.35, minLat: 54.45, maxLat: 57.9 } as const;

/** Latitude the longitude scale is taken at — the middle of the country. */
const LAT0_RAD = (56 * Math.PI) / 180;
const LON_SCALE = Math.cos(LAT0_RAD);

const SPAN_X = (MAP_BOUNDS.maxLon - MAP_BOUNDS.minLon) * LON_SCALE;
const SPAN_Y = MAP_BOUNDS.maxLat - MAP_BOUNDS.minLat;

/** SVG user units. Width is arbitrary; the height follows from the aspect. */
export const MAP_WIDTH = 400;
export const MAP_HEIGHT = Math.round((MAP_WIDTH * SPAN_Y) / SPAN_X);

/** Longitude/latitude in degrees → SVG user units. */
export function project(lon: number, lat: number): { x: number; y: number } {
	const scale = MAP_WIDTH / SPAN_X;
	return {
		x: (lon - MAP_BOUNDS.minLon) * LON_SCALE * scale,
		y: (MAP_BOUNDS.maxLat - lat) * scale
	};
}

/** The outline as SVG path data, computed once — it never changes. */
export const DENMARK_PATH: string = DENMARK_OUTLINE.map(
	(ring) =>
		ring
			.map(([lon, lat], i) => {
				const { x, y } = project(lon, lat);
				return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
			})
			.join('') + 'Z'
).join(' ');

/** Colour bands. Deliberately four, so the legend is readable at a glance. */
export type QualityBand = 'poor' | 'fair' | 'good' | 'best' | 'unknown';

export interface StationScore {
	/** 0–1, higher is better, whatever it was derived from. Null when unknown. */
	value: number | null;
	/** Which measurement the value came from. */
	basis: 'pod' | 'brier' | null;
	band: QualityBand;
}

/**
 * Brier scores worth colouring run from about 0.05 (very good) to 0.25
 * (barely better than climatology) on this kind of event, and lower is
 * better — so it is turned upside down before it is banded, and a station
 * coloured by it is never presented as if it were a POD.
 */
const BRIER_GOOD = 0.05;
const BRIER_POOR = 0.25;

const bandOf = (value: number): QualityBand =>
	value < 0.5 ? 'poor' : value < 0.65 ? 'fair' : value < 0.8 ? 'good' : 'best';

export function stationScore(properties: StationProperties): StationScore {
	const pod = properties.warn_pod;
	if (pod !== null && Number.isFinite(pod)) {
		const value = Math.min(1, Math.max(0, pod));
		return { value, basis: 'pod', band: bandOf(value) };
	}
	const brier = properties.brier_gauge;
	if (brier !== null && Number.isFinite(brier)) {
		const value = Math.min(
			1,
			Math.max(0, 1 - (brier - BRIER_GOOD) / (BRIER_POOR - BRIER_GOOD))
		);
		return { value, basis: 'brier', band: bandOf(value) };
	}
	return { value: null, basis: null, band: 'unknown' };
}

/** A station ready to draw. */
export interface PlottedStation {
	feature: StationFeature;
	x: number;
	y: number;
	score: StationScore;
}

/**
 * Project and score the stations, dropping any that fall outside the frame —
 * a station in Greenland is a producer bug, not a reason to stretch the map.
 */
export function plotStations(features: readonly StationFeature[]): PlottedStation[] {
	const plotted: PlottedStation[] = [];
	for (const feature of features) {
		const [lon, lat] = feature.geometry.coordinates;
		if (
			lon < MAP_BOUNDS.minLon ||
			lon > MAP_BOUNDS.maxLon ||
			lat < MAP_BOUNDS.minLat ||
			lat > MAP_BOUNDS.maxLat
		) {
			continue;
		}
		const { x, y } = project(lon, lat);
		plotted.push({ feature, x, y, score: stationScore(feature.properties) });
	}
	return plotted;
}
