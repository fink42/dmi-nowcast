/**
 * The motion arrow's geometry, held to what the arrow *claims*.
 *
 * The claim is specific and falsifiable: the shaft is one hour of travel long,
 * it points at where the rain is coming from, and every tick is one radar
 * timestep of travel further out. So the tests do not compare against the
 * implementation's own arithmetic — they measure the built GeoJSON back with
 * an independent haversine and an independent bearing, and check the metres.
 *
 * `destinationPoint` is checked against the exact rhumb-line destination,
 * because a cell advected at a constant bearing follows a rhumb line; the
 * great circle would be the wrong yardstick and would "fail" a correct arrow.
 */
import { describe, expect, it } from 'vitest';
import {
	ARROW_HORIZON_MIN,
	destinationPoint,
	emptyArrow,
	motionArrow,
	type ArrowCollection,
	type ArrowRole
} from './arrow';

const R_KM = 6371.0088;
const rad = (deg: number): number => (deg * Math.PI) / 180;

/** Great-circle distance in km — independent of the module under test. */
function haversineKm([lon1, lat1]: number[], [lon2, lat2]: number[]): number {
	const p1 = rad(lat1);
	const p2 = rad(lat2);
	const h =
		Math.sin((p2 - p1) / 2) ** 2 +
		Math.cos(p1) * Math.cos(p2) * Math.sin(rad(lon2 - lon1) / 2) ** 2;
	return 2 * R_KM * Math.asin(Math.sqrt(h));
}

/**
 * Constant (rhumb) bearing from a to b, degrees clockwise from north — the
 * bearing a cell holds while it is advected, and therefore the one the arrow
 * claims. The great circle's *initial* bearing is a different quantity: over a
 * 36 km shaft at 56°N the two differ by ~0.17°, which is meridian convergence,
 * not an error in the arrow.
 */
function rhumbBearing([lon1, lat1]: number[], [lon2, lat2]: number[]): number {
	const p1 = rad(lat1);
	const p2 = rad(lat2);
	const dPsi = Math.log(Math.tan(Math.PI / 4 + p2 / 2) / Math.tan(Math.PI / 4 + p1 / 2));
	let dLon = rad(lon2 - lon1);
	if (Math.abs(dLon) > Math.PI) dLon -= Math.sign(dLon) * 2 * Math.PI;
	return ((Math.atan2(dLon, dPsi) * 180) / Math.PI + 360) % 360;
}

/** Signed difference between two bearings, in (−180, 180]. */
const angleDiff = (a: number, b: number): number => (((a - b + 540) % 360) - 180);

/** Exact rhumb-line destination — the reference the approximation must match. */
function rhumbDestination(lat: number, lon: number, bearingDeg: number, distKm: number): number[] {
	const theta = rad(bearingDeg);
	const delta = distKm / R_KM;
	const p1 = rad(lat);
	const dPhi = delta * Math.cos(theta);
	const p2 = p1 + dPhi;
	const dPsi = Math.log(Math.tan(Math.PI / 4 + p2 / 2) / Math.tan(Math.PI / 4 + p1 / 2));
	const q = Math.abs(dPsi) > 1e-12 ? dPhi / dPsi : Math.cos(p1);
	const dLambda = (delta * Math.sin(theta)) / q;
	return [lon + (dLambda * 180) / Math.PI, (p2 * 180) / Math.PI];
}

/** Aarhus-ish: middle of the country, middle of the radar composite. */
const HOME = { lat: 56.15, lon: 10.2 };

const byRole = (fc: ArrowCollection, role: ArrowRole) =>
	fc.features.filter((f) => f.properties.role === role);

describe('destinationPoint', () => {
	it('walks due north along the meridian', () => {
		const [lon, lat] = destinationPoint(HOME.lat, HOME.lon, 0, 50);
		expect(lon).toBeCloseTo(HOME.lon, 12);
		expect(lat).toBeGreaterThan(HOME.lat);
		// 50 km / 111.195 km per degree.
		expect(lat - HOME.lat).toBeCloseTo(0.4496, 3);
	});

	it('walks due south along the meridian', () => {
		const [lon, lat] = destinationPoint(HOME.lat, HOME.lon, 180, 50);
		expect(lon).toBeCloseTo(HOME.lon, 12);
		expect(HOME.lat - lat).toBeCloseTo(0.4496, 3);
	});

	it('walks due east and west without changing latitude', () => {
		const east = destinationPoint(HOME.lat, HOME.lon, 90, 50);
		const west = destinationPoint(HOME.lat, HOME.lon, 270, 50);
		expect(east[1]).toBeCloseTo(HOME.lat, 12);
		expect(west[1]).toBeCloseTo(HOME.lat, 12);
		expect(east[0]).toBeGreaterThan(HOME.lon);
		expect(west[0]).toBeLessThan(HOME.lon);
		// Symmetric about the meridian, and one degree of longitude at 56.15°N
		// is 111.195·cos(56.15°) ≈ 61.94 km, so 50 km is ≈ 0.8073°.
		expect(east[0] - HOME.lon).toBeCloseTo(HOME.lon - west[0], 12);
		expect(east[0] - HOME.lon).toBeCloseTo(0.8073, 3);
	});

	it('places the endpoint at the requested great-circle range', () => {
		for (const distKm of [5, 20, 60, 120]) {
			for (let b = 0; b < 360; b += 15) {
				const p = destinationPoint(HOME.lat, HOME.lon, b, distKm);
				// Within 10 m: far under one 500 m radar pixel.
				expect(haversineKm([HOME.lon, HOME.lat], p)).toBeCloseTo(distKm, 2);
			}
		}
	});

	it('matches the exact rhumb destination to a few metres over Denmark', () => {
		let worstKm = 0;
		for (const lat of [54.0, 56.15, 58.4]) {
			for (const distKm of [20, 60, 120]) {
				for (let b = 0; b < 360; b += 5) {
					const got = destinationPoint(lat, 10.2, b, distKm);
					const want = rhumbDestination(lat, 10.2, b, distKm);
					worstKm = Math.max(worstKm, haversineKm(got, want));
				}
			}
		}
		// Documented bound in arrow.ts: 4.4 m at 120 km.
		expect(worstKm).toBeLessThan(0.005);
	});

	it('is its own inverse under a reversed bearing', () => {
		const out = destinationPoint(HOME.lat, HOME.lon, 305, 80);
		const back = destinationPoint(out[1], out[0], 305 - 180, 80);
		expect(back[0]).toBeCloseTo(HOME.lon, 6);
		expect(back[1]).toBeCloseTo(HOME.lat, 6);
	});
});

describe('motionArrow', () => {
	const BEARING = 315; // coming from the north-west
	const SPEED = 36; // km/h → a 36 km shaft
	const build = (over: Partial<Parameters<typeof motionArrow>[0]> = {}) =>
		motionArrow({
			lat: HOME.lat,
			lon: HOME.lon,
			bearingFromDeg: BEARING,
			speedKmh: SPEED,
			timestepMin: 10,
			...over
		});

	it('is a shaft, a head and one tick per timestep short of the tip', () => {
		const fc = build();
		// 10 min steps: ticks at 10/20/30/40/50; 60 is the tip, which gets the head.
		expect(fc.features).toHaveLength(1 + 1 + 5);
		expect(byRole(fc, 'shaft')).toHaveLength(1);
		expect(byRole(fc, 'head')).toHaveLength(1);
		expect(byRole(fc, 'tick').map((f) => f.properties.minute)).toEqual([10, 20, 30, 40, 50]);
	});

	it('adapts the tick count to the radar cadence', () => {
		expect(byRole(build({ timestepMin: 15 }), 'tick').map((f) => f.properties.minute)).toEqual([
			15, 30, 45
		]);
		expect(byRole(build({ timestepMin: 5 }), 'tick')).toHaveLength(11);
		// A cadence as long as the horizon leaves the shaft bare — no tick may
		// land on the tip, which is the head's.
		expect(byRole(build({ timestepMin: 60 }), 'tick')).toHaveLength(0);
		expect(build({ timestepMin: 60 }).features).toHaveLength(2);
	});

	it('never loops on a nonsense cadence', () => {
		for (const timestepMin of [0, -10, Number.NaN, 0.5]) {
			const fc = build({ timestepMin });
			expect(byRole(fc, 'tick')).toHaveLength(0);
			// The arrow itself survives: the cadence only costs the ruler marks.
			expect(fc.features).toHaveLength(2);
		}
	});

	it('makes the shaft exactly one hour of travel long', () => {
		for (const speedKmh of [3, 18, 36, 90]) {
			const shaft = byRole(build({ speedKmh }), 'shaft')[0];
			const coords = (shaft.geometry as { coordinates: number[][] }).coordinates;
			expect(coords[0]).toEqual([HOME.lon, HOME.lat]);
			// speedKmh × 1 h, in km.
			expect(haversineKm(coords[0], coords[1])).toBeCloseTo(speedKmh, 2);
		}
		expect(ARROW_HORIZON_MIN).toBe(60);
	});

	it('points upstream, along the bearing the rain comes from', () => {
		for (const bearingFromDeg of [0, 45, 90, 180, 271, 315]) {
			const shaft = byRole(build({ bearingFromDeg }), 'shaft')[0];
			const coords = (shaft.geometry as { coordinates: number[][] }).coordinates;
			// The arrow lies over the rain that is on its way, not over where it
			// is headed: reverse this and the map contradicts the panel.
			expect(angleDiff(rhumbBearing(coords[0], coords[1]), bearingFromDeg)).toBeCloseTo(0, 2);
		}
	});

	it('spaces the ticks evenly, at the distance travelled by that minute', () => {
		const fc = build();
		const tail = [HOME.lon, HOME.lat];
		const ranges = byRole(fc, 'tick').map((f) => {
			const [a, b] = (f.geometry as { coordinates: number[][] }).coordinates;
			// Range to the tick's centre, not to either of its ends.
			const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
			return haversineKm(tail, mid);
		});
		// 36 km/h ⇒ 6 km every 10 min.
		expect(ranges).toHaveLength(5);
		ranges.forEach((km, i) => expect(km).toBeCloseTo(SPEED * ((i + 1) * 10) / 60, 2));
		const gaps = ranges.slice(1).map((km, i) => km - ranges[i]);
		for (const gap of gaps) expect(gap).toBeCloseTo(6, 2);
	});

	it('draws ticks square across the shaft', () => {
		const tick = byRole(build(), 'tick')[0];
		const [a, b] = (tick.geometry as { coordinates: number[][] }).coordinates;
		// The tick runs from the left of the shaft to its right: bearing + 90.
		expect(angleDiff(rhumbBearing(a, b), BEARING + 90)).toBeCloseTo(0, 2);
		// Its full width is the head length (12 % of a 36 km shaft = 4.32 km),
		// which is the ruler-tick proportion the HA card uses.
		expect(haversineKm(a, b)).toBeCloseTo(4.32, 2);
	});

	it('caps the shaft with a closed triangle whose apex is the tip', () => {
		const fc = build();
		const shaft = (byRole(fc, 'shaft')[0].geometry as { coordinates: number[][] }).coordinates;
		const tip = shaft[1];
		const head = byRole(fc, 'head')[0];
		expect(head.geometry.type).toBe('Polygon');
		const ring = (head.geometry as { coordinates: number[][][] }).coordinates[0];
		expect(ring).toHaveLength(4);
		expect(ring[0]).toEqual(tip);
		expect(ring[3]).toEqual(ring[0]);
		// Both base corners sit one head length back from the tip, symmetric
		// about the shaft at ±25°.
		const headKm = 36 * 0.12;
		expect(haversineKm(tip, ring[1])).toBeCloseTo(headKm, 2);
		expect(haversineKm(tip, ring[2])).toBeCloseTo(headKm, 2);
		// …and behind it: nearer the tail than the tip is.
		const tail = [HOME.lon, HOME.lat];
		expect(haversineKm(tail, ring[1])).toBeLessThan(haversineKm(tail, tip));
		expect(haversineKm(tail, ring[2])).toBeLessThan(haversineKm(tail, tip));
	});

	it('keeps the head visible on a crawling cell and modest on a racing one', () => {
		const headLen = (speedKmh: number) => {
			const fc = motionArrow({ ...HOME, bearingFromDeg: 270, speedKmh, timestepMin: 10 });
			const ring = (byRole(fc, 'head')[0].geometry as { coordinates: number[][][] })
				.coordinates[0];
			return haversineKm(ring[0], ring[1]);
		};
		expect(headLen(2)).toBeCloseTo(1.5, 2); // floor
		expect(headLen(36)).toBeCloseTo(4.32, 2); // 12 %
		expect(headLen(200)).toBeCloseTo(8, 2); // ceiling
	});

	it('refuses a speed that is not a length', () => {
		for (const speedKmh of [0, -5, Number.NaN, Number.POSITIVE_INFINITY]) {
			expect(build({ speedKmh }).features).toHaveLength(0);
		}
		expect(build({ lat: Number.NaN }).features).toHaveLength(0);
	});

	it('has an empty collection for "no arrow"', () => {
		const fc = emptyArrow();
		expect(fc.type).toBe('FeatureCollection');
		expect(fc.features).toHaveLength(0);
	});
});
