/**
 * The map geometry of one event: the disc the verdict was taken over, the
 * neighbours that voted, and the upwind arrow that explains the decision.
 *
 * Pure GeoJSON with no MapLibre import, like `map/arrow.ts`, so all of it
 * is testable without a canvas and the component is left with sources,
 * layers and paint.
 *
 * The disc is drawn because "radius-based detection, not nearest pixel" is
 * a project-wide contract: the radar verdict is the p90 over a 1 km disc,
 * and a reviewer judging a warning against a single pixel under the station
 * marker is judging a different number from the one the service acted on.
 * The circle has to be on the map for that to be visible.
 */
import { destinationPoint } from '$lib/map/arrow';
import type { Feature, FeatureCollection, Point, Polygon } from 'geojson';
import type { NeighbourState, SlotState } from './truth';
import type { NeighbourRef, StationBlock } from './schema';

/** Metres per degree of latitude — the mean-Earth sphere `arrow.ts` uses. */
const EARTH_RADIUS_M = 6371008.8;

export type DiscProperties = { radius_m: number };

/**
 * The verdict disc as a polygon.
 *
 * Built from the same equirectangular step `arrow.ts` uses, so the circle
 * and the motion arrow cannot disagree about where 1 km is. At Danish
 * latitudes over a 1 km radius the error against an exact geodesic circle
 * is millimetres — three orders under a 500 m radar pixel — and the shape
 * is closed explicitly because a GeoJSON ring that does not repeat its
 * first point is invalid and renderers disagree about what to do with it.
 *
 * `steps` is the number of segments, floored at 8: below that it is a
 * polygon a reviewer would measure with their eye and get wrong.
 */
export function discPolygon(
	lat: number,
	lon: number,
	radiusM: number,
	steps: number = 64
): Feature<Polygon, DiscProperties> | null {
	if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
	if (!Number.isFinite(radiusM) || radiusM <= 0) return null;
	const segments = Math.max(8, Math.round(Number.isFinite(steps) ? steps : 64));
	const radiusKm = radiusM / 1000;
	const ring = Array.from({ length: segments }, (_, i) =>
		destinationPoint(lat, lon, (i * 360) / segments, radiusKm)
	);
	ring.push(ring[0]);
	return {
		type: 'Feature',
		properties: { radius_m: radiusM },
		geometry: { type: 'Polygon', coordinates: [ring] }
	};
}

export type StationRole = 'station' | 'neighbour';

/**
 * A type alias rather than an interface: GeoJSON's `properties` slot wants
 * an implicit index signature, which interfaces do not get.
 */
export type StationProperties = {
	role: StationRole;
	station_id: string;
	name: string;
	/** Kilometres from the event's station; zero for the station itself. */
	distance_km: number;
	bearing_deg: number;
	/**
	 * The reading at the cursor. `unknown` is its own value and must be
	 * styled as its own thing — a neighbour that did not report is not a dry
	 * neighbour, and a map that paints the two alike is the `known`/`wet`
	 * mistake in colour form.
	 */
	state: SlotState['state'];
	/** Why, when the state is unknown. Null otherwise. */
	unknown_reason: string | null;
	/** The neighbour's verdict over the whole window, or null when unknown. */
	wet_in_window: boolean | null;
	mm: number | null;
};

export type StationCollection = FeatureCollection<Point, StationProperties>;

const pointFeature = (
	lon: number,
	lat: number,
	properties: StationProperties
): Feature<Point, StationProperties> => ({
	type: 'Feature',
	properties,
	geometry: { type: 'Point', coordinates: [lon, lat] }
});

/**
 * The event's station and its neighbours as one source, each carrying its
 * reading at the cursor.
 *
 * The station is included so the map has a single source to style by
 * `role`, and so the station marker moves with the same data as the dots
 * around it. A neighbour with no state at the cursor is emitted as
 * `unknown` rather than dropped: a dot that disappears while scrubbing
 * reads as a station that does not exist, and the absence of a reading is
 * itself the thing a reviewer is judging.
 */
export function neighbourFeatures(
	station: StationBlock | null,
	neighbours: readonly NeighbourRef[],
	statesAtCursor: readonly NeighbourState[]
): StationCollection {
	const byId = new Map(statesAtCursor.map((entry) => [entry.station.station_id, entry]));
	const features: Feature<Point, StationProperties>[] = [];

	if (station !== null) {
		const own = byId.get(station.station_id);
		features.push(
			pointFeature(station.lon, station.lat, {
				role: 'station',
				station_id: station.station_id,
				name: station.name,
				distance_km: 0,
				bearing_deg: 0,
				state: own?.state.state ?? 'unknown',
				unknown_reason: own?.state.state === 'unknown' ? own.state.reason : null,
				wet_in_window: own?.wetInWindow ?? null,
				mm: own !== undefined && own.state.state !== 'unknown' ? own.state.mm : null
			})
		);
	}

	for (const neighbour of neighbours) {
		const entry = byId.get(neighbour.station_id);
		features.push(
			pointFeature(neighbour.lon, neighbour.lat, {
				role: 'neighbour',
				station_id: neighbour.station_id,
				name: neighbour.name,
				distance_km: neighbour.distance_km,
				bearing_deg: neighbour.bearing_deg,
				state: entry?.state.state ?? 'unknown',
				unknown_reason:
					entry === undefined
						? 'no_series'
						: entry.state.state === 'unknown'
							? entry.state.reason
							: null,
				wet_in_window: entry?.wetInWindow ?? null,
				mm: entry !== undefined && entry.state.state !== 'unknown' ? entry.state.mm : null
			})
		);
	}

	return { type: 'FeatureCollection', features };
}

export interface UpwindVector {
	/**
	 * Bearing the rain comes FROM, degrees clockwise from north — what
	 * `motionArrow` wants. The feature column is the bearing the flow heads
	 * TOWARD (`postprocess._motion_features`: `atan2(vx, -vy)` on a north-up
	 * grid), so this is that value plus 180. Getting it backwards points the
	 * arrow at the rain that has already passed, which is exactly the
	 * mistake a reviewer would then tag as `fa_cell_diverted`.
	 */
	bearingFromDeg: number;
	/** Bulk flow speed over the ground, km/h. */
	speedKmh: number;
	/** The completed flow's speed at the station itself, km/h, or null. */
	localSpeedKmh: number | null;
	/**
	 * Distance to the nearest echo in the 40 km upwind corridor, km. Null
	 * means the corridor was EMPTY — the pipeline writes NaN for "no echo
	 * upwind", and that is the signature of convective initiation, which
	 * advection cannot see. It is emphatically not zero.
	 */
	upstreamDistanceKm: number | null;
	/** Max rain rate in the 40 km upwind corridor, mm/h, or null. */
	upstreamMaxMmH: number | null;
}

/** One feature column as a finite number, or null for anything else. */
function feature(
	features: Record<string, number | string | null>,
	name: string
): number | null {
	const value = features[name];
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/**
 * The upwind vector behind one decision, or null when the cycle had no
 * usable motion.
 *
 * Null is a real answer and the caller must draw nothing for it: below
 * `MIN_FLOW_PX_PER_FRAME` the pipeline writes NaN for the bearing rather
 * than a direction, and inventing an arrow there would be the tool making
 * up the evidence it exists to check. A zero speed is the same case.
 */
export function upwindVector(
	features: Record<string, number | string | null> | null | undefined
): UpwindVector | null {
	if (!features) return null;
	const heading = feature(features, 'bulk_dir_deg');
	const speed = feature(features, 'bulk_kmh');
	if (heading === null || speed === null || speed <= 0) return null;
	return {
		bearingFromDeg: (heading + 180) % 360,
		speedKmh: speed,
		localSpeedKmh: feature(features, 'local_speed_kmh'),
		upstreamDistanceKm: feature(features, 'up_dist_km'),
		upstreamMaxMmH: feature(features, 'up_max_40km_mm_h')
	};
}

/**
 * Metres between two points on the mean-Earth sphere (haversine).
 *
 * Used to check what the cursor is actually sampling — "you are 1.4 km from
 * the station, outside the verdict disc" — which a reviewer cannot judge
 * from a zoomed map by eye.
 */
export function distanceM(lat1: number, lon1: number, lat2: number, lon2: number): number {
	const toRad = Math.PI / 180;
	const dLat = (lat2 - lat1) * toRad;
	const dLon = (lon2 - lon1) * toRad;
	const a =
		Math.sin(dLat / 2) ** 2 +
		Math.cos(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.sin(dLon / 2) ** 2;
	return 2 * EARTH_RADIUS_M * Math.asin(Math.min(1, Math.sqrt(a)));
}
