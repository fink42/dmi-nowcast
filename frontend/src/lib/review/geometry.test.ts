/**
 * Map geometry: the verdict disc, the station dots, and the upwind vector.
 *
 * The disc is on the map because "radius-based detection, not nearest
 * pixel" is a project-wide contract, and a reviewer judging a warning
 * against whatever pixel sits under the station marker is judging a
 * different number from the one the service acted on. So the circle has to
 * be the right size, and that is measured here against an independent
 * distance function rather than trusted.
 *
 * The arrow's direction is the other easy mistake: `bulk_dir_deg` is the
 * bearing the flow heads TOWARD, and the arrow points at where the rain
 * comes FROM. Getting it backwards puts the arrow over rain that has
 * already passed — which a reviewer would then tag as `fa_cell_diverted`,
 * a mechanism that did not happen.
 */
import { describe, expect, it } from 'vitest';
import fixture from './fixture.json';
import { discPolygon, distanceM, neighbourFeatures, upwindVector } from './geometry';
import { parseEvent } from './load';
import type { EventDetail } from './schema';
import { neighbourStatesAt } from './truth';

const details = fixture.details as Record<string, any>;
const parsed = (eventId: string): EventDetail =>
	parseEvent(JSON.parse(JSON.stringify(details[eventId])))!;

const ODENSE = { lat: 55.4757, lon: 10.3308 };

describe('discPolygon', () => {
	it('draws a closed ring of the requested resolution', () => {
		const disc = discPolygon(ODENSE.lat, ODENSE.lon, 1000, 32)!;
		const ring = disc.geometry.coordinates[0];
		expect(ring).toHaveLength(33);
		expect(ring[0]).toEqual(ring[ring.length - 1]);
		expect(disc.properties.radius_m).toBe(1000);
	});

	it('puts every vertex at the radius, measured independently', () => {
		for (const radiusM of [1000, 5000, 20000]) {
			const ring = discPolygon(ODENSE.lat, ODENSE.lon, radiusM)!.geometry.coordinates[0];
			for (const [lon, lat] of ring) {
				const measured = distanceM(ODENSE.lat, ODENSE.lon, lat, lon);
				// Half a metre at 20 km: three orders under one radar pixel.
				expect(Math.abs(measured - radiusM)).toBeLessThan(1);
			}
		}
	});

	it('refuses to draw a disc it cannot place or size', () => {
		expect(discPolygon(Number.NaN, 10, 1000)).toBeNull();
		expect(discPolygon(55, Number.NaN, 1000)).toBeNull();
		expect(discPolygon(55, 10, 0)).toBeNull();
		expect(discPolygon(55, 10, -1)).toBeNull();
	});

	it('never degenerates below eight segments', () => {
		const ring = discPolygon(55, 10, 1000, 2)!.geometry.coordinates[0];
		expect(ring).toHaveLength(9);
	});
});

describe('distanceM', () => {
	it('agrees with a degree of latitude', () => {
		// One degree of latitude on the mean-Earth sphere: 111.195 km.
		expect(distanceM(55, 10, 56, 10)).toBeCloseTo(111195, -1);
	});

	it('is zero at the same point', () => {
		expect(distanceM(55.4757, 10.3308, 55.4757, 10.3308)).toBe(0);
	});
});

describe('neighbourFeatures', () => {
	const event = parsed('fa-06126-20260612T1345Z');
	const cursor = Date.parse(event.window.anchor_utc);
	const states = neighbourStatesAt(event.neighbours, cursor);
	const collection = neighbourFeatures(event.station, event.station!.neighbours, states);

	it('carries the station and every neighbour', () => {
		expect(collection.features).toHaveLength(1 + event.station!.neighbours.length);
		expect(collection.features[0].properties.role).toBe('station');
		expect(collection.features.slice(1).every((f) => f.properties.role === 'neighbour')).toBe(
			true
		);
	});

	it('places each dot at its own coordinates', () => {
		const station = collection.features[0];
		expect(station.geometry.coordinates).toEqual([event.station!.lon, event.station!.lat]);
		for (const neighbour of event.station!.neighbours) {
			const feature = collection.features.find(
				(f) => f.properties.station_id === neighbour.station_id
			)!;
			expect(feature.geometry.coordinates).toEqual([neighbour.lon, neighbour.lat]);
			expect(feature.properties.distance_km).toBe(neighbour.distance_km);
		}
	});

	it('paints the reading at the cursor, with unknown as its own state', () => {
		const silent = parsed('fa-06104-20260421T0255Z');
		const silentStates = neighbourStatesAt(silent.neighbours, Date.parse(silent.window.anchor_utc));
		const features = neighbourFeatures(
			silent.station,
			silent.station!.neighbours,
			silentStates
		);
		for (const feature of features.features.slice(1)) {
			// A neighbour that did not report is not a dry neighbour.
			expect(feature.properties.state).toBe('unknown');
			expect(feature.properties.unknown_reason).not.toBeNull();
			expect(feature.properties.mm).toBeNull();
		}
	});

	it('keeps a neighbour with no state at all rather than dropping the dot', () => {
		const features = neighbourFeatures(event.station, event.station!.neighbours, []);
		expect(features.features).toHaveLength(1 + event.station!.neighbours.length);
		expect(features.features[1].properties.state).toBe('unknown');
		expect(features.features[1].properties.unknown_reason).toBe('no_series');
	});

	it('draws nothing at all without a station block', () => {
		expect(neighbourFeatures(null, [], []).features).toEqual([]);
	});
});

describe('upwindVector', () => {
	it('turns the flow’s heading into the bearing the rain comes FROM', () => {
		const vector = upwindVector({ bulk_dir_deg: 70, bulk_kmh: 32 })!;
		expect(vector.bearingFromDeg).toBe(250);
		expect(vector.speedKmh).toBe(32);

		// And wraps rather than running past 360.
		expect(upwindVector({ bulk_dir_deg: 250, bulk_kmh: 10 })!.bearingFromDeg).toBe(70);
		expect(upwindVector({ bulk_dir_deg: 0, bulk_kmh: 10 })!.bearingFromDeg).toBe(180);
	});

	it('draws no arrow where the pipeline had no motion', () => {
		expect(upwindVector(null)).toBeNull();
		expect(upwindVector({})).toBeNull();
		expect(upwindVector({ bulk_dir_deg: null, bulk_kmh: 30 })).toBeNull();
		expect(upwindVector({ bulk_dir_deg: 250, bulk_kmh: null })).toBeNull();
		expect(upwindVector({ bulk_dir_deg: 250, bulk_kmh: 0 })).toBeNull();
	});

	it('keeps an empty upwind corridor null, never zero', () => {
		// NaN there means "no echo upwind at all" — the signature of
		// convective initiation, which advection cannot see. Zero would mean
		// "echo right on top of us", the opposite claim.
		const vector = upwindVector({ bulk_dir_deg: 250, bulk_kmh: 30, up_dist_km: null })!;
		expect(vector.upstreamDistanceKm).toBeNull();
		expect(upwindVector({ bulk_dir_deg: 250, bulk_kmh: 30, up_dist_km: 0 })!.upstreamDistanceKm).toBe(
			0
		);
	});

	it('reads the vector off a real decision in the bundle', () => {
		const event = parsed('fa-06126-20260612T1345Z');
		const decision = event.decisions.find((d) => d.features_present)!;
		const vector = upwindVector(decision.features)!;
		expect(vector.bearingFromDeg).toBeCloseTo(
			((decision.features.bulk_dir_deg as number) + 180) % 360,
			6
		);
		expect(vector.localSpeedKmh).toBe(decision.features.local_speed_kmh);
	});

	it('has nothing to draw on a row whose features are missing', () => {
		const event = parsed('fa-06104-20260421T0255Z');
		const decision = event.decisions.find((d) => !d.features_present)!;
		expect(upwindVector(decision.features)).toBeNull();
	});
});
