/**
 * Bearing arithmetic, pinned in both directions.
 *
 * The trap this file exists for is the half turn. The grids give the vector
 * the cell is *travelling along*; the panel says where it is *coming from*,
 * and getting that backwards produces a sentence that is confidently wrong in
 * a way no test of "does it render" would catch. Every axis and every one of
 * the eight sectors is checked against a hand-worked answer.
 *
 * The second contract is the refusal: nodata is not a direction, and neither
 * is a stationary cell. Both must come back null so the UI says "no measured
 * cell motion here" instead of drawing an arrow.
 */
import { describe, expect, it } from 'vitest';
import {
	bearingFromDeg,
	cellMotion,
	compassPoint,
	COMPASS_POINTS,
	MIN_SPEED_KMH,
	type CompassPoint
} from './motion';

/** (east, north) in km/h → the bearing the cell comes from, by hand. */
const AXES: [number, number, number, CompassPoint][] = [
	// Travelling north ⇒ comes from the south.
	[0, 20, 180, 's'],
	// Travelling east ⇒ comes from the west.
	[20, 0, 270, 'w'],
	// Travelling south ⇒ comes from the north.
	[0, -20, 0, 'n'],
	// Travelling west ⇒ comes from the east.
	[-20, 0, 90, 'e']
];

/** The four diagonals, at equal components so the bearing is exact. */
const DIAGONALS: [number, number, number, CompassPoint][] = [
	[10, 10, 225, 'sw'], // north-east bound ⇒ from the south-west
	[10, -10, 315, 'nw'], // south-east bound ⇒ from the north-west
	[-10, -10, 45, 'ne'], // south-west bound ⇒ from the north-east
	[-10, 10, 135, 'se'] // north-west bound ⇒ from the south-east
];

describe('bearingFromDeg', () => {
	it('names the direction the cell comes from, on every axis', () => {
		for (const [east, north, bearing] of AXES) {
			expect(bearingFromDeg(east, north), `${east},${north}`).toBeCloseTo(bearing, 10);
		}
	});

	it('and on every diagonal', () => {
		for (const [east, north, bearing] of DIAGONALS) {
			expect(bearingFromDeg(east, north), `${east},${north}`).toBeCloseTo(bearing, 10);
		}
	});

	it('stays in [0, 360) whatever the sign of the components', () => {
		/** Shortest angular distance, so a result of 359.999 ≈ 0 passes. */
		const apart = (a: number, b: number) => Math.abs((((a - b) % 360) + 540) % 360 - 180);
		for (let deg = 0; deg < 360; deg += 7) {
			const rad = (deg * Math.PI) / 180;
			const bearing = bearingFromDeg(Math.sin(rad) * 30, Math.cos(rad) * 30);
			expect(bearing, `heading ${deg}`).toBeGreaterThanOrEqual(0);
			expect(bearing, `heading ${deg}`).toBeLessThan(360);
			// Travelling towards `deg` means coming from `deg + 180`.
			expect(apart(bearing, deg + 180), `heading ${deg}`).toBeLessThan(1e-6);
		}
	});
});

describe('compassPoint', () => {
	it('returns each of the eight sectors at its centre', () => {
		COMPASS_POINTS.forEach((name, i) => {
			expect(compassPoint(i * 45)).toBe(name);
		});
	});

	it('snaps to the nearer point inside a sector, both ways', () => {
		expect(compassPoint(22)).toBe('n');
		expect(compassPoint(23)).toBe('ne');
		expect(compassPoint(67)).toBe('ne');
		expect(compassPoint(68)).toBe('e');
		expect(compassPoint(337)).toBe('nw');
		expect(compassPoint(338)).toBe('n');
	});

	it('wraps around north rather than falling off the end', () => {
		expect(compassPoint(360)).toBe('n');
		expect(compassPoint(359)).toBe('n');
		expect(compassPoint(-45)).toBe('nw');
		expect(compassPoint(720 + 90)).toBe('e');
	});
});

describe('cellMotion', () => {
	it('carries bearing, speed and compass point together', () => {
		// 20 km/h east + 20 km/h north: 28.3 km/h out of the south-west.
		const motion = cellMotion(20, 20);
		expect(motion).not.toBeNull();
		expect(motion!.bearingFromDeg).toBeCloseTo(225, 10);
		expect(motion!.speedKmh).toBeCloseTo(Math.sqrt(800), 10);
		expect(motion!.compass).toBe('sw');
	});

	it('agrees with the axis and diagonal tables end to end', () => {
		for (const [east, north, bearing, compass] of [...AXES, ...DIAGONALS]) {
			const motion = cellMotion(east, north);
			expect(motion, `${east},${north}`).not.toBeNull();
			expect(motion!.bearingFromDeg).toBeCloseTo(bearing, 10);
			expect(motion!.compass).toBe(compass);
		}
	});

	it('refuses nodata in either component — never half an arrow', () => {
		expect(cellMotion(null, 12)).toBeNull();
		expect(cellMotion(12, null)).toBeNull();
		expect(cellMotion(null, null)).toBeNull();
		expect(cellMotion(undefined, undefined)).toBeNull();
	});

	it('refuses non-finite components', () => {
		expect(cellMotion(NaN, 10)).toBeNull();
		expect(cellMotion(10, Infinity)).toBeNull();
	});

	it('refuses a speed below the noise floor instead of inventing a heading', () => {
		// atan2(0, 0) is 0, which would render as a confident "coming from S".
		expect(cellMotion(0, 0)).toBeNull();
		expect(cellMotion(0.1, -0.2)).toBeNull();
		// Just above the floor is a real, if slow, measurement.
		const slow = cellMotion(MIN_SPEED_KMH + 0.01, 0);
		expect(slow).not.toBeNull();
		expect(slow!.compass).toBe('w');
	});
});
