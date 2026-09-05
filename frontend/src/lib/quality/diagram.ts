/**
 * The reliability diagram, as data.
 *
 * One panel per lead, x = the probability we forecast, y = how often it then
 * rained, and the diagonal is where a calibrated forecast sits. Everything
 * geometric lives here rather than in the markup so the mapping from bins to
 * coordinates can be tested without a DOM, and so the SVG and the table
 * alternative are built from exactly the same numbers.
 *
 * The one rule that matters: an empty bin is a hole in the curve. It is
 * skipped, never drawn at zero and never interpolated across — a bin with no
 * forecasts in it is not evidence that the forecast was wrong there.
 */
import type { QualityReliability, ReliabilityBin, ReliabilityCurve } from './schema';

/** A plotted bin. `x`/`y` are 0–1 fractions; `n` sizes the marker. */
export interface ReliabilityPoint {
	x: number;
	y: number;
	n: number;
	effN: number;
	lo: number;
	hi: number;
}

/**
 * Panel geometry, in SVG user units. The plot area is deliberately square:
 * perfect calibration has to be a 45° diagonal, or the eye reads a bias off a
 * diagram that has none.
 */
export const PLOT = {
	width: 220,
	height: 214,
	pad: { left: 36, right: 10, top: 10, bottom: 30 }
} as const;

const inner = {
	width: PLOT.width - PLOT.pad.left - PLOT.pad.right,
	height: PLOT.height - PLOT.pad.top - PLOT.pad.bottom
};

/** Probability (0–1) → SVG x. */
export const plotX = (value: number): number => PLOT.pad.left + value * inner.width;

/** Frequency (0–1) → SVG y. Zero is at the bottom, which SVG does not think. */
export const plotY = (value: number): number => PLOT.pad.top + (1 - value) * inner.height;

/** Marker radius in user units, area ∝ sample size, floored so a rare bin stays visible. */
export function markerRadius(n: number, maxN: number): number {
	const MIN = 2.2;
	const MAX = 6;
	if (!Number.isFinite(n) || n <= 0 || !Number.isFinite(maxN) || maxN <= 0) return MIN;
	return MIN + (MAX - MIN) * Math.sqrt(Math.min(1, n / maxN));
}

/** The plottable bins of a curve: both values present and at least one sample. */
export function curvePoints(curve: ReliabilityCurve | null | undefined): ReliabilityPoint[] {
	if (!curve) return [];
	const points: ReliabilityPoint[] = [];
	for (const bin of curve.bins) {
		if (bin.forecast_mean === null || bin.observed_freq === null || bin.n <= 0) continue;
		points.push({
			x: bin.forecast_mean,
			y: bin.observed_freq,
			n: bin.n,
			effN: bin.eff_n,
			lo: bin.lo,
			hi: bin.hi
		});
	}
	return points.sort((a, b) => a.x - b.x);
}

/** An SVG polyline `points` attribute for a curve, or '' when it has under two points. */
export function polyline(points: readonly ReliabilityPoint[]): string {
	if (points.length < 2) return '';
	return points.map((p) => `${plotX(p.x).toFixed(2)},${plotY(p.y).toFixed(2)}`).join(' ');
}

/** The leads to draw panels for: whatever the document actually carries, ascending. */
export function leadsOf(reliability: QualityReliability): number[] {
	const leads = new Set<number>();
	for (const curve of reliability.radar ?? []) leads.add(curve.lead_min);
	for (const curve of reliability.gauge ?? []) leads.add(curve.lead_min);
	return [...leads].sort((a, b) => a - b);
}

export const curveAt = (
	curves: ReliabilityCurve[] | null,
	lead: number
): ReliabilityCurve | null => curves?.find((curve) => curve.lead_min === lead) ?? null;

/** The largest bin population in a panel — the reference the markers scale against. */
export function maxBinN(...curves: (ReliabilityCurve | null)[]): number {
	let max = 0;
	for (const curve of curves) {
		for (const bin of curve?.bins ?? []) if (bin.n > max) max = bin.n;
	}
	return max;
}

/** One row of the table alternative: the same bin, in both truths. */
export interface TableRow {
	lo: number;
	hi: number;
	radar: ReliabilityBin | null;
	gauge: ReliabilityBin | null;
}

/**
 * The table behind a panel. Bins are matched between the two truths by their
 * lower edge, so a producer that bins the gauges differently still lines up
 * where it can and shows a gap where it cannot.
 */
export function tableRows(
	radar: ReliabilityCurve | null,
	gauge: ReliabilityCurve | null
): TableRow[] {
	const rows = new Map<string, TableRow>();
	const key = (bin: ReliabilityBin) => bin.lo.toFixed(3);
	for (const bin of radar?.bins ?? []) {
		rows.set(key(bin), { lo: bin.lo, hi: bin.hi, radar: bin, gauge: null });
	}
	for (const bin of gauge?.bins ?? []) {
		const existing = rows.get(key(bin));
		if (existing) existing.gauge = bin;
		else rows.set(key(bin), { lo: bin.lo, hi: bin.hi, radar: null, gauge: bin });
	}
	return [...rows.values()].sort((a, b) => a.lo - b.lo);
}
