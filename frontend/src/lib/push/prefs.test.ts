/**
 * Preferences: normalisation, the localStorage copy, and the POST body.
 *
 * The storage half is written against a fake `localStorage` — including one
 * that throws on every call, which is what Safari does in a locked-down
 * private window and what iOS has done at various times for no stated reason.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
	clearStored,
	defaultPrefs,
	DISABLED_CONFIG,
	FALLBACK_PREFS,
	isValidHhMm,
	loadStored,
	normalisePrefs,
	parsePushConfig,
	resolveTimeZone,
	saveStored,
	STORAGE_KEY,
	subscribeBody,
	type PushConfig,
	type StoredSubscription
} from './prefs';

const CONFIG: PushConfig = {
	enabled: true,
	vapidPublicKey: 'BAaaaa',
	thresholdOptionsPct: [40, 60, 80],
	leadOptionsMin: [20, 30, 45, 60],
	defaults: {
		thresholdPct: 60,
		leadMin: 30,
		quietHours: { enabled: false, start: '22:00', end: '07:00' }
	},
	capacityReached: false
};

/** A minimal in-memory Storage; `throwing` reproduces a browser that refuses. */
function fakeStorage(throwing = false, seed: Record<string, string> = {}) {
	const map = new Map(Object.entries(seed));
	const boom = () => {
		throw new DOMException('storage disabled', 'SecurityError');
	};
	return {
		get length() {
			return map.size;
		},
		key: (i: number) => [...map.keys()][i] ?? null,
		getItem: (k: string) => (throwing ? boom() : (map.get(k) ?? null)),
		setItem: (k: string, v: string) => (throwing ? boom() : void map.set(k, v)),
		removeItem: (k: string) => (throwing ? boom() : void map.delete(k)),
		clear: () => (throwing ? boom() : map.clear()),
		_map: map
	} as unknown as Storage & { _map: Map<string, string> };
}

afterEach(() => vi.unstubAllGlobals());

describe('isValidHhMm', () => {
	it('accepts the 24-hour zero-padded form an <input type=time> emits', () => {
		for (const value of ['00:00', '07:00', '09:05', '22:30', '23:59']) {
			expect(isValidHhMm(value), value).toBe(true);
		}
	});

	it('rejects everything else', () => {
		for (const value of ['7:00', '24:00', '22:60', '22.30', '2230', '', ' 22:30', 7, null, {}]) {
			expect(isValidHhMm(value), String(value)).toBe(false);
		}
	});
});

describe('defaultPrefs', () => {
	it('uses the server defaults, and the fallbacks with no config', () => {
		expect(defaultPrefs(CONFIG)).toEqual(CONFIG.defaults);
		expect(defaultPrefs(null)).toEqual(FALLBACK_PREFS);
	});

	it('hands back a copy, so the caller cannot edit the config', () => {
		const prefs = defaultPrefs(CONFIG);
		prefs.thresholdPct = 80;
		prefs.quietHours.enabled = true;
		expect(CONFIG.defaults.thresholdPct).toBe(60);
		expect(CONFIG.defaults.quietHours.enabled).toBe(false);
	});
});

describe('normalisePrefs', () => {
	it('passes a valid set through unchanged', () => {
		const prefs = {
			thresholdPct: 80,
			leadMin: 45,
			quietHours: { enabled: true, start: '23:00', end: '06:30' }
		};
		expect(normalisePrefs(prefs, CONFIG)).toEqual(prefs);
	});

	it('accepts numeric strings, which is what a <select> gives back', () => {
		expect(normalisePrefs({ thresholdPct: '80', leadMin: '20' }, CONFIG)).toMatchObject({
			thresholdPct: 80,
			leadMin: 20
		});
	});

	it('snaps an out-of-set value to the nearest option the server offers', () => {
		expect(normalisePrefs({ thresholdPct: 55, leadMin: 33 }, CONFIG)).toMatchObject({
			thresholdPct: 60,
			leadMin: 30
		});
		expect(normalisePrefs({ thresholdPct: 100, leadMin: 5 }, CONFIG)).toMatchObject({
			thresholdPct: 80,
			leadMin: 20
		});
	});

	it('falls back to the defaults for missing keys and outright garbage', () => {
		expect(normalisePrefs({}, CONFIG)).toEqual(CONFIG.defaults);
		expect(normalisePrefs(null, CONFIG)).toEqual(CONFIG.defaults);
		expect(normalisePrefs('60', CONFIG)).toEqual(CONFIG.defaults);
		expect(normalisePrefs([60, 30], CONFIG)).toEqual(CONFIG.defaults);
		expect(normalisePrefs({ thresholdPct: 'high', leadMin: Number.NaN }, CONFIG)).toEqual(
			CONFIG.defaults
		);
	});

	it('repairs quiet hours one field at a time', () => {
		expect(
			normalisePrefs({ quietHours: { enabled: 'yes', start: '25:00', end: '06:30' } }, CONFIG)
		).toMatchObject({
			quietHours: { enabled: false, start: '22:00', end: '06:30' }
		});
		expect(normalisePrefs({ quietHours: 'nope' }, CONFIG)).toEqual(CONFIG.defaults);
	});

	it('works with no config at all, against the fallback options', () => {
		expect(normalisePrefs({ thresholdPct: 41, leadMin: 61 }, null)).toMatchObject({
			thresholdPct: 40,
			leadMin: 60
		});
	});
});

describe('parsePushConfig', () => {
	it('reads anything that is not an enabled, keyed config as disabled', () => {
		for (const value of [null, undefined, 'html', {}, { enabled: false }, { enabled: true }]) {
			expect(parsePushConfig(value), String(value)).toEqual(DISABLED_CONFIG);
		}
	});

	it('repairs a config whose options or defaults are unusable', () => {
		const config = parsePushConfig({
			enabled: true,
			vapid_public_key: 'BAaaaa',
			threshold_options_pct: 'nope',
			lead_options_min: [],
			defaults: { threshold_pct: 'x', lead_min: null, quiet_hours: { start: '99:99' } }
		});
		expect(config.thresholdOptionsPct).toEqual([40, 60, 80]);
		expect(config.leadOptionsMin).toEqual([20, 30, 45, 60]);
		expect(config.defaults).toEqual(FALLBACK_PREFS);
	});

	it('validates the server defaults against the server options', () => {
		const config = parsePushConfig({
			enabled: true,
			vapid_public_key: 'BAaaaa',
			threshold_options_pct: [50, 70],
			lead_options_min: [15, 45],
			defaults: { threshold_pct: 60, lead_min: 30 }
		});
		// Both defaults sit exactly between two options; the earlier one wins.
		expect(config.defaults.thresholdPct).toBe(50);
		expect(config.defaults.leadMin).toBe(15);
	});
});

const SUB: StoredSubscription = {
	endpoint: 'https://push.example/abc',
	lat: 55.6761,
	lon: 12.5683,
	prefs: { thresholdPct: 80, leadMin: 45, quietHours: { enabled: true, start: '23:00', end: '06:30' } },
	lang: 'da',
	tz: 'Europe/Copenhagen',
	subscribedAt: '2026-09-02T13:05:00.000Z'
};

describe('stored subscription', () => {
	it('round-trips through storage', () => {
		const store = fakeStorage();
		vi.stubGlobal('localStorage', store);
		expect(saveStored(SUB)).toBe(true);
		expect(store._map.has(STORAGE_KEY)).toBe(true);
		expect(loadStored(CONFIG)).toEqual(SUB);
		clearStored();
		expect(loadStored(CONFIG)).toBeNull();
	});

	it('survives storage that throws on every call', () => {
		vi.stubGlobal('localStorage', fakeStorage(true));
		expect(() => loadStored(CONFIG)).not.toThrow();
		expect(loadStored(CONFIG)).toBeNull();
		expect(saveStored(SUB)).toBe(false);
		expect(() => clearStored()).not.toThrow();
	});

	it('survives no storage at all', () => {
		vi.stubGlobal('localStorage', undefined);
		expect(loadStored(CONFIG)).toBeNull();
		expect(saveStored(SUB)).toBe(false);
		expect(() => clearStored()).not.toThrow();
	});

	it('rejects a stored copy that is not usable', () => {
		for (const text of [
			'not json',
			'null',
			'[]',
			'{}',
			JSON.stringify({ ...SUB, endpoint: '' }),
			JSON.stringify({ ...SUB, lat: 'north' })
		]) {
			vi.stubGlobal('localStorage', fakeStorage(false, { [STORAGE_KEY]: text }));
			expect(loadStored(CONFIG), text).toBeNull();
		}
	});

	it('repairs a stored copy whose preferences drifted out of range', () => {
		const stale = JSON.stringify({
			...SUB,
			lang: 'de',
			tz: '',
			prefs: {
				thresholdPct: 95,
				leadMin: 'x',
				quietHours: { enabled: true, start: 'bad', end: '06:30' }
			}
		});
		vi.stubGlobal('localStorage', fakeStorage(false, { [STORAGE_KEY]: stale }));
		const loaded = loadStored(CONFIG);
		expect(loaded).toMatchObject({
			lang: 'da',
			tz: 'Europe/Copenhagen',
			prefs: {
				thresholdPct: 80,
				leadMin: 30,
				quietHours: { enabled: true, start: '22:00', end: '06:30' }
			}
		});
	});
});

describe('resolveTimeZone', () => {
	it('returns an IANA zone', () => {
		expect(resolveTimeZone()).toMatch(/^[A-Za-z]+\/[A-Za-z_+-]/);
	});
});

describe('subscribeBody', () => {
	it('builds exactly the shape the endpoint documents', () => {
		const json = { endpoint: 'https://push.example/abc', keys: { p256dh: 'k', auth: 'a' } };
		expect(subscribeBody(json, 55.6761, 12.5683, SUB.prefs, 'en', 'Europe/Copenhagen')).toEqual({
			subscription: json,
			lat: 55.6761,
			lon: 12.5683,
			threshold_pct: 80,
			lead_min: 45,
			quiet_hours: { enabled: true, start: '23:00', end: '06:30' },
			tz: 'Europe/Copenhagen',
			lang: 'en'
		});
	});
});
