/**
 * The fitted-threshold section: the loader that has to tolerate a producer
 * which has never run the fit, and the table builder that must not turn an
 * undefined rate into a confident 0 %.
 */
import { describe, expect, it } from 'vitest';
import { da } from '$lib/i18n/da';
import { en } from '$lib/i18n/en';
import fixture from './fixture.json';
import { parseQuality } from './load';
import { thresholdIntro, thresholdRows } from './thresholds';
import type { QualityThresholds } from './schema';

/** A fresh deep copy, so a test that mutates cannot reach the next one. */
const doc = (): Record<string, unknown> => JSON.parse(JSON.stringify(fixture));

const parsed = (thresholds: unknown): QualityThresholds | null =>
	parseQuality({ ...doc(), thresholds })?.thresholds ?? null;

describe('the thresholds section of the fixture', () => {
	const section = parseQuality(doc())!.thresholds!;

	it('parses, in horizon order', () => {
		expect(section).not.toBeNull();
		expect(section.leads.map((row) => row.lead_min)).toEqual([20, 30, 45, 60]);
		expect(section.fallback_threshold_pct).toBe(40);
		expect(section.fitted_at_utc).toBe('2026-09-03T02:11:07Z');
		expect(section.objective).toMatchObject({ metric: 'f1', min_useful_lead_min: 5 });
	});

	it('keeps the numbers it was given', () => {
		const lead20 = section.leads[0];
		expect(lead20.threshold_pct).toBe(35);
		expect(lead20.insufficient).toBe(false);
		expect(lead20.warnings).toBe(214);
		expect(lead20.plateau).toEqual([30, 40]);
		expect(lead20.agrees_with_radar).toBe(true);
	});

	it('carries an insufficient horizon as null scores, not zeros', () => {
		const lead60 = section.leads[3];
		expect(lead60.insufficient).toBe(true);
		expect(lead60.threshold_pct).toBeNull();
		expect(lead60.f1).toBeNull();
		expect(lead60.precision).toBeNull();
		// The counts behind the verdict are still real numbers.
		expect(lead60.warnings).toBe(18);
	});

	it('carries the horizon where the two truths disagree', () => {
		expect(section.leads[2].agrees_with_radar).toBe(false);
		expect(section.leads[2].radar_plateau).toEqual([40, 55]);
	});
});

describe('parsing a thresholds section that is missing or wrong', () => {
	it('is null when the producer has never fitted them', () => {
		const without = doc();
		delete without.thresholds;
		expect(parseQuality(without)!.thresholds).toBeNull();
		expect(parsed(null)).toBeNull();
	});

	it('is null for anything that is not a section with horizons in it', () => {
		for (const value of ['soon', 42, [], {}, { leads: {} }, { leads: 'nope' }]) {
			expect(parsed(value), JSON.stringify(value)).toBeNull();
		}
	});

	it('leaves the rest of the document readable', () => {
		const broken = { ...doc(), thresholds: 'nope' };
		const report = parseQuality(broken)!;
		expect(report.thresholds).toBeNull();
		expect(report.methods).not.toBeNull();
		expect(report.headline.warnings).not.toBeNull();
	});

	it('drops a horizon whose counts cannot be read, and keeps the others', () => {
		const section = parsed({
			fitted_at_utc: '2026-09-03T02:11:07Z',
			fallback_threshold_pct: 40,
			leads: {
				20: { threshold_pct: 35, insufficient: false, warnings: 10, hits: 6, false_alarms: 4, misses: 3, late: 1 },
				30: { threshold_pct: 35, insufficient: false, warnings: 'many' },
				notALead: { threshold_pct: 35, warnings: 10, hits: 6, false_alarms: 4, misses: 3, late: 1 }
			}
		});
		expect(section!.leads.map((row) => row.lead_min)).toEqual([20]);
	});

	it('treats a horizon with no threshold as insufficient whatever the flag says', () => {
		const section = parsed({
			leads: {
				45: {
					threshold_pct: null,
					insufficient: false,
					warnings: 3,
					hits: 1,
					false_alarms: 2,
					misses: 9,
					late: 0
				}
			}
		});
		expect(section!.leads[0].insufficient).toBe(true);
	});

	it('falls back for a section that states no default, and drops an unreadable objective', () => {
		const section = parsed({
			objective: { min_useful_lead_min: 5 },
			leads: { 20: { threshold_pct: 35, warnings: 10, hits: 6, false_alarms: 4, misses: 3, late: 1 } }
		});
		expect(section!.fallback_threshold_pct).toBe(40);
		expect(section!.objective).toBeNull();
		expect(section!.fitted_at_utc).toBeNull();
	});
});

describe('thresholdRows', () => {
	const section = parseQuality(doc())!.thresholds!;
	const rows = thresholdRows(en, 'en', section);

	it('renders one row per horizon, in order', () => {
		expect(rows.map((row) => row.key)).toEqual([20, 30, 45, 60]);
		expect(rows[0].horizon).toBe('20 min');
	});

	it('renders the fitted threshold and the rates as whole percent', () => {
		expect(rows[0].threshold).toBe('35%');
		expect(rows[0].precision).toBe('66%');
		expect(rows[0].recall).toBe('57%');
		expect(rows[0].f1).toBe('0.61');
		expect(rows[0].warnings).toBe('214');
	});

	it('shows the default an insufficient horizon actually warns at, and says so', () => {
		const lead60 = rows[3];
		expect(lead60.insufficient).toBe(true);
		// The rule still fires there; it is just not fitted for that lead.
		expect(lead60.threshold).toBe('40%');
		expect(lead60.precision).toBe('—');
		expect(lead60.f1).toBe('—');
		expect(lead60.note).toBe(en.quality.thresholds.insufficient);
	});

	it('notes the horizon where radar and gauges disagree', () => {
		expect(rows[2].note).toBe(en.quality.thresholds.disagrees);
		expect(rows[2].note).toContain('gauges win');
	});

	it('leaves a sound horizon without a note', () => {
		expect(rows[0].note).toBeNull();
		expect(rows[1].note).toBeNull();
	});

	it('renders in Danish too', () => {
		const daRows = thresholdRows(da, 'da', section);
		expect(daRows[0].f1).toBe('0,61');
		expect(daRows[3].note).toBe(da.quality.thresholds.insufficient);
	});

	it('is an empty list with no section, which the page renders as a sentence', () => {
		expect(thresholdRows(en, 'en', null)).toEqual([]);
	});
});

describe('thresholdIntro', () => {
	const section = parseQuality(doc())!.thresholds!;

	it('names the minimum useful lead the fit ran under', () => {
		expect(thresholdIntro(en, section)).toContain('5 minutes');
		expect(thresholdIntro(da, section)).toContain('5 minutter');
	});

	it('drops the clause when no objective survived', () => {
		const without: QualityThresholds = { ...section, objective: null };
		expect(thresholdIntro(en, without)).toBe(en.quality.thresholds.introNoLead);
		expect(thresholdIntro(en, null)).toBe(en.quality.thresholds.introNoLead);
	});
});
