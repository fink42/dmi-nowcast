/**
 * Cell motion at one point, in the words people actually use: "coming from
 * the north-west at 32 km/h".
 *
 * The sidecar serves two quantised grids on the product geometry —
 * `motion_east_kmh` and `motion_north_kmh`, east- and north-positive, in km/h
 * — with nodata wherever there is no honest estimate (outside radar coverage,
 * or too far from any echo to have one). The manifest's `motion` block says so
 * outright: *there is no motion estimate there, do not draw an arrow.* That is
 * the rule this module enforces, and the reason every entry point returns null
 * rather than a plausible-looking zero.
 *
 * Bearings are meteorological: degrees clockwise from true north, naming the
 * direction the weather comes FROM. A cell travelling east comes from the
 * west — bearing 270.
 */

/** The eight points, in bearing order from north. Catalog keys, not labels. */
export const COMPASS_POINTS = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'] as const;

export type CompassPoint = (typeof COMPASS_POINTS)[number];

export interface CellMotion {
	/** Bearing the cell comes from: degrees clockwise from north, [0, 360). */
	bearingFromDeg: number;
	/** Speed over the ground, km/h — unrounded; the UI rounds for display. */
	speedKmh: number;
	compass: CompassPoint;
}

/**
 * Below this, the direction is quantisation noise rather than a measurement:
 * the grids' own step is ~0.94 km/h, and a speed that rounds to zero has no
 * meaningful bearing to draw an arrow from.
 */
export const MIN_SPEED_KMH = 0.5;

/**
 * Motion vector → the bearing it comes from.
 *
 *   bearing_from = (atan2(east, north) · 180/π + 180) mod 360
 *
 * `atan2(east, north)` — arguments in that order — is the compass bearing the
 * cell is heading *towards*; the half turn flips it to where it came from.
 */
export function bearingFromDeg(eastKmh: number, northKmh: number): number {
	return (((Math.atan2(eastKmh, northKmh) * 180) / Math.PI + 180) % 360 + 360) % 360;
}

/** Nearest of the eight points; sector boundaries fall on the 22.5° halves. */
export function compassPoint(bearingDeg: number): CompassPoint {
	const wrapped = ((bearingDeg % 360) + 360) % 360;
	return COMPASS_POINTS[Math.round(wrapped / 45) % 8];
}

/**
 * The cell motion at a sampled point, or null when there is nothing honest to
 * say: either component missing (nodata, or a cycle that served no motion
 * grids), a non-finite value, or a speed below the noise floor.
 */
export function cellMotion(
	eastKmh: number | null | undefined,
	northKmh: number | null | undefined
): CellMotion | null {
	if (typeof eastKmh !== 'number' || typeof northKmh !== 'number') return null;
	if (!Number.isFinite(eastKmh) || !Number.isFinite(northKmh)) return null;
	const speedKmh = Math.hypot(eastKmh, northKmh);
	if (speedKmh < MIN_SPEED_KMH) return null;
	const bearing = bearingFromDeg(eastKmh, northKmh);
	return { bearingFromDeg: bearing, speedKmh, compass: compassPoint(bearing) };
}
