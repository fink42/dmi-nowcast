/**
 * The quality loader. The property under test throughout is that a document
 * which is wrong somewhere still renders everywhere else, and that a section
 * we cannot read comes back as null — never as a zero, which on this page
 * would be a lie with a decimal point.
 */
import { describe, expect, it, vi } from 'vitest';
import fixture from './fixture.json';
import { parseQuality } from './load';
import { QUALITY_SCHEMA_VERSION } from './schema';

/** A fresh deep copy, so a test that mutates cannot reach the next one. */
const doc = (): Record<string, unknown> => JSON.parse(JSON.stringify(fixture));

describe('parseQuality on the fixture', () => {
	const report = parseQuality(doc());

	it('parses every section', () => {
		expect(report).not.toBeNull();
		expect(report!.schema_version).toBe(QUALITY_SCHEMA_VERSION);
		expect(report!.windows.radar).not.toBeNull();
		expect(report!.windows.gauge).not.toBeNull();
		expect(report!.windows.live).not.toBeNull();
		expect(report!.headline.reliability.radar).not.toBeNull();
		expect(report!.headline.reliability.gauge).not.toBeNull();
		expect(report!.headline.warnings).not.toBeNull();
		expect(report!.headline.persistence_margin).not.toBeNull();
		expect(report!.reliability.radar).not.toBeNull();
		expect(report!.reliability.gauge).not.toBeNull();
		expect(report!.raining_now).not.toBeNull();
		expect(report!.stations).not.toBeNull();
		expect(report!.events).not.toBeNull();
		expect(report!.methods).not.toBeNull();
	});

	it('keeps the numbers it was given', () => {
		expect(report!.headline.warnings!.hits).toBe(103);
		expect(report!.headline.warnings!.lead_error_min.p50).toBe(4);
		expect(report!.raining_now!.observation_agreement).toBeCloseTo(0.793, 6);
		expect(report!.stations!.features).toHaveLength(12);
		expect(report!.events!).toHaveLength(8);
		expect(report!.methods!.frame_age_range_min).toEqual([14, 24]);
	});

	it('orders reliability curves by lead', () => {
		expect(report!.reliability.radar!.map((c) => c.lead_min)).toEqual([10, 20, 30, 45, 60]);
		expect(report!.reliability.gauge!.map((c) => c.lead_min)).toEqual([10, 20, 30, 45, 60]);
	});

	it('keeps empty bins as bins with null values', () => {
		const lead60 = report!.reliability.gauge!.find((c) => c.lead_min === 60)!;
		expect(lead60.bins).toHaveLength(10);
		const empty = lead60.bins.filter((b) => b.forecast_mean === null);
		expect(empty).toHaveLength(2);
		expect(empty[0].observed_freq).toBeNull();
		expect(empty[0].n).toBe(0);
	});

	it('exercises the fallback the station colouring needs', () => {
		const withoutPod = report!.stations!.features.filter((f) => f.properties.warn_pod === null);
		expect(withoutPod).toHaveLength(1);
		expect(withoutPod[0].properties.brier_gauge).not.toBeNull();
	});
});

describe('parseQuality with sections missing', () => {
	it('nulls every top-level section without losing the rest', () => {
		const raw = doc();
		for (const key of [
			'windows',
			'headline',
			'reliability',
			'raining_now',
			'stations',
			'events',
			'methods'
		]) {
			raw[key] = null;
		}
		const report = parseQuality(raw);
		expect(report).not.toBeNull();
		expect(report!.generated_at_utc).toBe(fixture.generated_at_utc);
		expect(report!.windows).toEqual({ radar: null, gauge: null, live: null });
		expect(report!.headline.reliability).toEqual({ radar: null, gauge: null });
		expect(report!.headline.warnings).toBeNull();
		expect(report!.headline.persistence_margin).toBeNull();
		expect(report!.reliability).toEqual({ radar: null, gauge: null });
		expect(report!.raining_now).toBeNull();
		expect(report!.stations).toBeNull();
		expect(report!.events).toBeNull();
		expect(report!.methods).toBeNull();
	});

	it('survives sections that are missing entirely', () => {
		const report = parseQuality({
			schema_version: QUALITY_SCHEMA_VERSION,
			generated_at_utc: '2026-09-05T02:14:37Z'
		});
		expect(report).not.toBeNull();
		expect(report!.windows.radar).toBeNull();
		expect(report!.headline.warnings).toBeNull();
		expect(report!.reliability.radar).toBeNull();
	});

	it('nulls one truth without touching the other', () => {
		const raw = doc();
		(raw.reliability as Record<string, unknown>).gauge = null;
		(raw.headline as { reliability: Record<string, unknown> }).reliability.gauge = null;
		const report = parseQuality(raw);
		expect(report!.reliability.radar).not.toBeNull();
		expect(report!.reliability.gauge).toBeNull();
		expect(report!.headline.reliability.radar).not.toBeNull();
		expect(report!.headline.reliability.gauge).toBeNull();
	});

	it('drops a half-written block rather than filling it with zeros', () => {
		const raw = doc();
		delete (raw.headline as { warnings: Record<string, unknown> }).warnings.hits;
		(raw.raining_now as Record<string, unknown>).pod = null;
		(raw.windows as Record<string, unknown>).live = { days: 30 };
		const report = parseQuality(raw);
		expect(report!.headline.warnings).toBeNull();
		expect(report!.raining_now).toBeNull();
		expect(report!.windows.live).toBeNull();
		// …and the untouched neighbours are still there.
		expect(report!.headline.persistence_margin).not.toBeNull();
		expect(report!.windows.radar).not.toBeNull();
	});

	it('returns null only when there is nothing to render', () => {
		expect(parseQuality(null)).toBeNull();
		expect(parseQuality('{}')).toBeNull();
		expect(parseQuality([])).toBeNull();
		expect(parseQuality({ schema_version: 1 })).toBeNull();
		expect(parseQuality({ schema_version: 1, generated_at_utc: 'yesterday' })).toBeNull();
	});
});

describe('parseQuality on a schema it was not written against', () => {
	it('warns and still renders what parses', () => {
		const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
		const raw = doc();
		raw.schema_version = 99;
		const report = parseQuality(raw);
		expect(warn).toHaveBeenCalledOnce();
		expect(String(warn.mock.calls[0][0])).toContain('99');
		expect(report).not.toBeNull();
		expect(report!.schema_version).toBe(99);
		expect(report!.headline.warnings).not.toBeNull();
		warn.mockRestore();
	});

	it('warns when the version is missing or not a number', () => {
		const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
		const raw = doc();
		delete raw.schema_version;
		const report = parseQuality(raw);
		expect(warn).toHaveBeenCalledOnce();
		expect(report!.schema_version).toBe(0);
		expect(report!.stations).not.toBeNull();
		warn.mockRestore();
	});
});

describe('parseQuality on malformed pieces', () => {
	it('drops unreadable bins and keeps the rest of the curve', () => {
		const raw = doc();
		const curve = (raw.reliability as { radar: { bins: unknown[] }[] }).radar[0];
		curve.bins[2] = { lo: 0.2, hi: 0.3, forecast_mean: 'quite likely', observed_freq: 0.25 };
		curve.bins[5] = 'not a bin';
		const report = parseQuality(raw);
		const parsed = report!.reliability.radar![0];
		expect(parsed.bins).toHaveLength(8);
		expect(parsed.bins.map((b) => b.lo)).not.toContain(0.2);
	});

	it('drops a curve whose bins are unusable', () => {
		const raw = doc();
		const radar = (raw.reliability as { radar: Record<string, unknown>[] }).radar;
		radar[0].bins = [];
		radar[1].bins = 'ten bins';
		const report = parseQuality(raw);
		expect(report!.reliability.radar!.map((c) => c.lead_min)).toEqual([30, 45, 60]);
	});

	it('drops a station without coordinates and keeps its neighbours', () => {
		const raw = doc();
		const features = (raw.stations as { features: Record<string, unknown>[] }).features;
		features[0].geometry = { type: 'Point', coordinates: ['east', 57] };
		features[1] = { type: 'Feature' };
		const report = parseQuality(raw);
		expect(report!.stations!.features).toHaveLength(10);
	});

	it('drops an event with an outcome it does not know', () => {
		const raw = doc();
		const events = (raw.events as Record<string, unknown>[])!;
		events[0].outcome = 'probably';
		events[1].warned_at_utc = 'this morning';
		const report = parseQuality(raw);
		expect(report!.events).toHaveLength(6);
	});

	it('keeps a false alarm, whose onset and error are legitimately null', () => {
		const report = parseQuality(doc());
		const falseAlarm = report!.events!.find((e) => e.outcome === 'false_alarm')!;
		expect(falseAlarm.gauge_onset_utc).toBeNull();
		expect(falseAlarm.lead_error_min).toBeNull();
	});

	it('drops methods when a rule is missing rather than inventing one', () => {
		const raw = doc();
		delete (raw.methods as Record<string, unknown>).onset_rule;
		expect(parseQuality(raw)!.methods).toBeNull();

		const partial = doc();
		(partial.methods as Record<string, unknown>).frame_age_range_min = [14];
		expect(parseQuality(partial)!.methods).toBeNull();
	});
});
