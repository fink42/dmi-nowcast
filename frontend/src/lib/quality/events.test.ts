/**
 * The verified-events table. The row that matters is the false alarm: it has
 * no onset and no timing error, and both cells have to stay empty rather than
 * become "on time", which is what a zero would read as.
 */
import { describe, expect, it } from 'vitest';
import { clockTime } from '$lib/format';
import { da } from '$lib/i18n/da';
import { en } from '$lib/i18n/en';
import { localDate, localDateTime } from './dates';
import { eventRows } from './events';
import type { VerifiedEvent } from './schema';

const event = (overrides: Partial<VerifiedEvent> = {}): VerifiedEvent => ({
	station_id: '06181',
	name: 'Københavns Lufthavn',
	warned_at_utc: '2026-09-04T18:35:00Z',
	eta_min: 17,
	p_rain: 0.81,
	gauge_onset_utc: '2026-09-04T18:50:00Z',
	outcome: 'hit',
	lead_error_min: 2,
	...overrides
});

describe('eventRows', () => {
	it('says what we said and what happened, in local time', () => {
		const [row] = eventRows(en, 'en', [event()]);
		expect(row.name).toBe('Københavns Lufthavn');
		expect(row.warnedAt).toBe(localDateTime('2026-09-04T18:35:00Z', 'en'));
		expect(row.said).toContain('17');
		expect(row.said).toContain('81%');
		expect(row.happened).toBe(en.quality.events.onset(clockTime('2026-09-04T18:50:00Z', 'en')));
		expect(row.hit).toBe(true);
	});

	it('phrases the timing error by its sign', () => {
		const [late] = eventRows(en, 'en', [event({ lead_error_min: 12 })]);
		expect(late.error).toBe(en.quality.events.late(12));
		const [early] = eventRows(en, 'en', [event({ lead_error_min: -13 })]);
		expect(early.error).toBe(en.quality.events.early(13));
		const [onTime] = eventRows(da, 'da', [event({ lead_error_min: 0 })]);
		expect(onTime.error).toBe(da.quality.events.onTime);
	});

	it('leaves a false alarm without an onset or an error', () => {
		const [row] = eventRows(da, 'da', [
			event({ outcome: 'false_alarm', gauge_onset_utc: null, lead_error_min: null })
		]);
		expect(row.happened).toBe(da.quality.events.noRain);
		expect(row.error).toBeNull();
		expect(row.hit).toBe(false);
	});

	it('says "no rain" for a hit whose onset went missing, rather than a wrong time', () => {
		const [row] = eventRows(en, 'en', [event({ gauge_onset_utc: null })]);
		expect(row.happened).toBe(en.quality.events.noRain);
	});

	it('keeps the order it was given and makes a key per row', () => {
		const rows = eventRows(en, 'en', [
			event(),
			event({ station_id: '06074', name: 'Aarhus Syd', warned_at_utc: '2026-09-04T14:05:00Z' })
		]);
		expect(rows.map((r) => r.name)).toEqual(['Københavns Lufthavn', 'Aarhus Syd']);
		expect(new Set(rows.map((r) => r.key)).size).toBe(2);
	});

	it('is empty when nothing has been verified', () => {
		expect(eventRows(en, 'en', null)).toEqual([]);
		expect(eventRows(en, 'en', [])).toEqual([]);
	});
});

describe('dates at the boundary', () => {
	it('formats a window edge as a date and an event as a date and time', () => {
		expect(localDate('2026-03-01T00:00:00Z', 'en')).toMatch(/2026/);
		expect(localDateTime('2026-09-04T18:35:00Z', 'en')).toMatch(/\d{2}[:.]\d{2}/);
	});

	it('never throws on a timestamp it cannot read', () => {
		expect(() => localDate('not a date', 'da')).not.toThrow();
		expect(() => localDateTime('not a date', 'da')).not.toThrow();
	});
});
