/**
 * Payload handling. Everything here runs inside the service worker, where a
 * thrown exception means a push message that silently shows nothing — so the
 * rule is "return null, never throw", and these tests hold it to that.
 */
import { describe, expect, it } from 'vitest';
import {
	notificationFromPayload,
	parsePushPayload,
	payloadLang,
	pointFromUrl
} from './notification';

const raw = {
	type: 'rain_incoming',
	title: 'Regn på vej',
	body: 'Regn om ca. 12 min (78 %)',
	lang: 'da',
	lat: 55.6761,
	lon: 12.5683,
	url: '/?lat=55.6761&lon=12.5683',
	tag: 'rain-incoming',
	sent_utc: '2026-09-02T13:05:00Z',
	eta_min: 12,
	p_pct: 78,
	lead_min: 30,
	intensity_mm_h: 2.4
};

describe('parsePushPayload', () => {
	it('camel-cases a complete payload', () => {
		const p = parsePushPayload(raw);
		expect(p).not.toBeNull();
		expect(p).toMatchObject({
			type: 'rain_incoming',
			title: 'Regn på vej',
			lang: 'da',
			lat: 55.6761,
			lon: 12.5683,
			url: '/?lat=55.6761&lon=12.5683',
			tag: 'rain-incoming',
			sentUtc: '2026-09-02T13:05:00Z',
			etaMin: 12,
			pPct: 78,
			leadMin: 30,
			intensityMmH: 2.4
		});
	});

	it('accepts a test payload', () => {
		expect(parsePushPayload({ ...raw, type: 'test' })?.type).toBe('test');
	});

	it('rejects an unknown type', () => {
		expect(parsePushPayload({ ...raw, type: 'hail_warning' })).toBeNull();
		expect(parsePushPayload({ ...raw, type: 42 })).toBeNull();
	});

	it('rejects a payload with no title or no body', () => {
		expect(parsePushPayload({ ...raw, title: '' })).toBeNull();
		expect(parsePushPayload({ ...raw, body: undefined })).toBeNull();
	});

	it('rejects things that are not objects', () => {
		for (const value of [null, undefined, 'text', 7, [], true]) {
			expect(parsePushPayload(value)).toBeNull();
		}
	});

	it('keeps the optional numbers null rather than guessing', () => {
		const p = parsePushPayload({ type: 'test', title: 'T', body: 'B' });
		expect(p).toMatchObject({
			lang: 'da',
			lat: null,
			lon: null,
			url: '/',
			tag: 'rain-incoming',
			etaMin: null,
			pPct: null,
			leadMin: null,
			intensityMmH: null
		});
	});

	it('drops coordinates that are out of range or not numbers', () => {
		expect(parsePushPayload({ ...raw, lat: 955 })?.lat).toBeNull();
		expect(parsePushPayload({ ...raw, lon: '12.5' })?.lon).toBeNull();
		expect(parsePushPayload({ ...raw, lat: Number.NaN })?.lat).toBeNull();
	});

	it('falls back to Danish for an unknown language', () => {
		expect(parsePushPayload({ ...raw, lang: 'de' })?.lang).toBe('da');
		expect(parsePushPayload({ ...raw, lang: 'en' })?.lang).toBe('en');
	});
});

describe('payloadLang', () => {
	it('reads a language out of even an otherwise unusable payload', () => {
		expect(payloadLang({ lang: 'en' })).toBe('en');
		expect(payloadLang({ lang: 'da' })).toBe('da');
		expect(payloadLang({ nothing: true })).toBe('da');
		expect(payloadLang(null)).toBe('da');
	});
});

describe('notificationFromPayload', () => {
	it('builds the options a rain alert needs', () => {
		const p = parsePushPayload(raw)!;
		const { title, options } = notificationFromPayload(p);
		expect(title).toBe('Regn på vej');
		expect(options.body).toBe('Regn om ca. 12 min (78 %)');
		expect(options.icon).toBe('/icons/icon-192.png');
		expect(options.badge).toBe('/icons/icon-192.png');
		expect(options.tag).toBe('rain-incoming');
		expect(options.lang).toBe('da');
		expect(options.renotify).toBe(true);
		expect(options.data).toEqual({ url: '/?lat=55.6761&lon=12.5683' });
	});

	it('does not renotify when there is no tag to replace', () => {
		const p = parsePushPayload({ ...raw, tag: '' })!;
		// An empty tag falls back to the default one, which is still a tag.
		expect(notificationFromPayload(p).options.tag).toBe('rain-incoming');
		expect(notificationFromPayload({ ...p, tag: '' }).options.renotify).toBe(false);
	});
});

describe('notificationFromPayload: all-clear and the quiet fields', () => {
	const allClear = {
		type: 'all_clear',
		title: 'Regn alligevel ikke på vej',
		body: 'Radaren viser ikke længere regn ved dit punkt.',
		lang: 'da',
		lat: 55.6761,
		lon: 12.5683,
		url: '/?lat=55.6761&lon=12.5683',
		tag: 'rain-incoming',
		sent_utc: '2026-09-02T13:15:00Z',
		silent: true,
		renotify: false
	};

	it('parses an all_clear payload and reads the booleans', () => {
		const p = parsePushPayload(allClear);
		expect(p).toMatchObject({ type: 'all_clear', silent: true, renotify: false });
	});

	it('replaces the warning quietly: same tag, silent, no renotify', () => {
		const { title, options } = notificationFromPayload(parsePushPayload(allClear)!);
		expect(title).toBe('Regn alligevel ikke på vej');
		expect(options).toEqual({
			body: 'Radaren viser ikke længere regn ved dit punkt.',
			icon: '/icons/icon-192.png',
			badge: '/icons/icon-192.png',
			tag: 'rain-incoming',
			lang: 'da',
			renotify: false,
			silent: true,
			data: { url: '/?lat=55.6761&lon=12.5683' }
		});
		// Platform default: no vibration pattern, no sticky notification.
		expect(options).not.toHaveProperty('vibrate');
		expect(options).not.toHaveProperty('requireInteraction');
	});

	it('uses the tag and English strings the payload carries', () => {
		const { title, options } = notificationFromPayload(
			parsePushPayload({
				...allClear,
				lang: 'en',
				title: 'Rain no longer expected',
				body: 'No rain at your point after all.',
				tag: 'rain-incoming-2'
			})!
		);
		expect(title).toBe('Rain no longer expected');
		expect(options.body).toBe('No rain at your point after all.');
		expect(options.tag).toBe('rain-incoming-2');
		expect(options.lang).toBe('en');
	});

	it('keeps an all_clear quiet even when the fields are missing', () => {
		const { silent: _s, renotify: _r, ...bare } = allClear;
		const { options } = notificationFromPayload(parsePushPayload(bare)!);
		expect(options.silent).toBe(true);
		expect(options.renotify).toBe(false);
	});

	it('honours explicit silent/renotify false on a warning', () => {
		const { options } = notificationFromPayload(
			parsePushPayload({ ...raw, silent: false, renotify: false })!
		);
		expect(options.silent).toBe(false);
		expect(options.renotify).toBe(false);
	});

	it('ignores non-boolean values for the quiet fields', () => {
		const p = parsePushPayload({ ...raw, silent: 'yes', renotify: 1 })!;
		expect(p.silent).toBeNull();
		expect(p.renotify).toBeNull();
	});

	it('leaves a legacy payload exactly as before', () => {
		const { title, options } = notificationFromPayload(parsePushPayload(raw)!);
		expect(title).toBe('Regn på vej');
		expect(options).toEqual({
			body: 'Regn om ca. 12 min (78 %)',
			icon: '/icons/icon-192.png',
			badge: '/icons/icon-192.png',
			tag: 'rain-incoming',
			lang: 'da',
			renotify: true,
			data: { url: '/?lat=55.6761&lon=12.5683' }
		});
		expect(options).not.toHaveProperty('silent');
	});
});

describe('pointFromUrl', () => {
	it('reads a point out of a query string', () => {
		expect(pointFromUrl('?lat=55.6761&lon=12.5683')).toEqual({ lat: 55.6761, lon: 12.5683 });
		expect(pointFromUrl('lat=56&lon=10')).toEqual({ lat: 56, lon: 10 });
	});

	it('reads a point out of a whole URL', () => {
		expect(pointFromUrl('https://example.org/?lat=57.05&lon=9.92')).toEqual({
			lat: 57.05,
			lon: 9.92
		});
		expect(pointFromUrl('/?lat=-33.9&lon=18.4')).toEqual({ lat: -33.9, lon: 18.4 });
	});

	it('returns null when either parameter is missing or empty', () => {
		expect(pointFromUrl('')).toBeNull();
		expect(pointFromUrl('?lat=55.6')).toBeNull();
		expect(pointFromUrl('?lon=12.5')).toBeNull();
		// The trap: Number('') is 0, and (0, 0) is a point in the ocean.
		expect(pointFromUrl('?lat=&lon=')).toBeNull();
		expect(pointFromUrl('?lat=%20&lon=%20')).toBeNull();
	});

	it('returns null for values that are not coordinates', () => {
		expect(pointFromUrl('?lat=here&lon=there')).toBeNull();
		expect(pointFromUrl('?lat=91&lon=12')).toBeNull();
		expect(pointFromUrl('?lat=55&lon=181')).toBeNull();
	});
});
