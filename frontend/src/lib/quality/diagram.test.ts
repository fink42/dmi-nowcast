/**
 * Bins to coordinates. The case worth pinning is the empty bin: it is a hole
 * in the curve, and a diagram that plots it at (0, 0) invents a claim that
 * the forecast was catastrophically wrong at a probability nobody ever
 * forecast.
 */
import { describe, expect, it } from 'vitest';
import {
	curveAt,
	curvePoints,
	leadsOf,
	markerRadius,
	maxBinN,
	PLOT,
	plotX,
	plotY,
	polyline,
	tableRows
} from './diagram';
import type { ReliabilityBin, ReliabilityCurve } from './schema';

const bin = (
	lo: number,
	forecast: number | null,
	observed: number | null,
	n = 100
): ReliabilityBin => ({
	lo,
	hi: Math.round((lo + 0.1) * 10) / 10,
	forecast_mean: forecast,
	observed_freq: observed,
	n: forecast === null ? 0 : n,
	eff_n: forecast === null ? 0 : Math.round(n / 8)
});

const curve = (lead: number, bins: ReliabilityBin[]): ReliabilityCurve => ({
	lead_min: lead,
	brier: 0.08,
	n: bins.reduce((sum, b) => sum + b.n, 0),
	eff_n: 1000,
	bins
});

describe('curvePoints', () => {
	it('maps a bin to the point the diagram draws', () => {
		const points = curvePoints(curve(30, [bin(0.7, 0.74, 0.68, 4200)]));
		expect(points).toEqual([{ x: 0.74, y: 0.68, n: 4200, effN: 525, lo: 0.7, hi: 0.8 }]);
	});

	it('skips bins with a null value on either axis', () => {
		const points = curvePoints(
			curve(60, [bin(0.1, 0.14, 0.13), bin(0.8, null, null), bin(0.9, 0.94, null)])
		);
		expect(points.map((p) => p.x)).toEqual([0.14]);
	});

	it('skips an empty bin even when it carries values', () => {
		// n = 0 with numbers in it is a producer bug; it is still not evidence.
		const points = curvePoints(curve(10, [{ ...bin(0.5, 0.55, 0.0), n: 0 }]));
		expect(points).toEqual([]);
	});

	it('sorts by forecast probability, whatever order the bins arrived in', () => {
		const points = curvePoints(curve(10, [bin(0.8, 0.84, 0.79), bin(0.1, 0.14, 0.13)]));
		expect(points.map((p) => p.x)).toEqual([0.14, 0.84]);
	});

	it('is empty for a missing curve rather than throwing', () => {
		expect(curvePoints(null)).toEqual([]);
		expect(curvePoints(undefined)).toEqual([]);
	});
});

describe('panel geometry', () => {
	it('puts 0 at the left and bottom, 1 at the right and top', () => {
		expect(plotX(0)).toBe(PLOT.pad.left);
		expect(plotX(1)).toBe(PLOT.width - PLOT.pad.right);
		expect(plotY(0)).toBe(PLOT.height - PLOT.pad.bottom);
		expect(plotY(1)).toBe(PLOT.pad.top);
	});

	it('places the diagonal on equal probability and frequency', () => {
		// Perfect calibration is a 45° line in the panel's own units.
		expect(plotX(0.5) - plotX(0)).toBeCloseTo(plotY(0) - plotY(0.5), 6);
	});

	it('builds a polyline only when there is a line to draw', () => {
		const points = curvePoints(curve(10, [bin(0.1, 0.14, 0.13), bin(0.8, 0.84, 0.79)]));
		expect(polyline(points).split(' ')).toHaveLength(2);
		expect(polyline(points.slice(0, 1))).toBe('');
		expect(polyline([])).toBe('');
	});

	it('scales markers by sample size, with a visible floor', () => {
		expect(markerRadius(0, 1000)).toBeLessThan(markerRadius(500, 1000));
		expect(markerRadius(500, 1000)).toBeLessThan(markerRadius(1000, 1000));
		// A bin larger than the reference is clamped, not drawn off the panel.
		expect(markerRadius(4000, 1000)).toBe(markerRadius(1000, 1000));
		expect(markerRadius(10, 0)).toBeGreaterThan(0);
	});

	it('takes the marker reference from every truth in the panel', () => {
		const radar = curve(10, [bin(0.1, 0.14, 0.13, 8000)]);
		const gauge = curve(10, [bin(0.1, 0.14, 0.2, 120)]);
		expect(maxBinN(radar, gauge)).toBe(8000);
		expect(maxBinN(null, gauge)).toBe(120);
		expect(maxBinN(null, null)).toBe(0);
	});
});

describe('panels and curves', () => {
	const radar = [curve(10, [bin(0.1, 0.14, 0.13)]), curve(30, [bin(0.1, 0.14, 0.12)])];
	const gauge = [curve(30, [bin(0.1, 0.14, 0.18)]), curve(60, [bin(0.1, 0.14, 0.1)])];

	it('draws a panel for every lead either truth has', () => {
		expect(leadsOf({ radar, gauge })).toEqual([10, 30, 60]);
		expect(leadsOf({ radar, gauge: null })).toEqual([10, 30]);
		expect(leadsOf({ radar: null, gauge: null })).toEqual([]);
	});

	it('finds the curve for a lead, or null', () => {
		expect(curveAt(radar, 30)!.lead_min).toBe(30);
		expect(curveAt(radar, 45)).toBeNull();
		expect(curveAt(null, 30)).toBeNull();
	});
});

describe('the table alternative', () => {
	it('pairs the two truths bin by bin', () => {
		const radar = curve(30, [bin(0.0, 0.03, 0.02), bin(0.7, 0.74, 0.68)]);
		const gauge = curve(30, [bin(0.7, 0.75, 0.62), bin(0.9, null, null)]);
		const rows = tableRows(radar, gauge);
		expect(rows.map((r) => r.lo)).toEqual([0, 0.7, 0.9]);
		expect(rows[0].gauge).toBeNull();
		expect(rows[1].radar!.observed_freq).toBe(0.68);
		expect(rows[1].gauge!.observed_freq).toBe(0.62);
		// The empty bin survives into the table, where "no data" can be said.
		expect(rows[2].gauge!.forecast_mean).toBeNull();
	});

	it('works with only one truth, or none', () => {
		expect(tableRows(curve(10, [bin(0.1, 0.14, 0.13)]), null)).toHaveLength(1);
		expect(tableRows(null, null)).toEqual([]);
	});
});
