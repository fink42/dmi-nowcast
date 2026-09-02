/**
 * The API layer against a stubbed `fetch`. The case worth the most here is a
 * config endpoint that answers with the SPA shell: an older sidecar serves
 * `index.html` with HTTP 200 for any unknown path, and the app has to read
 * that as "no push here" rather than blowing up on the first HTML byte.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
	fetchPushConfig,
	OffCoverageError,
	postSubscribe,
	postUnsubscribe,
	PushUnavailableError
} from './api';
import { FALLBACK_PREFS, subscribeBody } from './prefs';

interface Call {
	url: string;
	init?: RequestInit;
}

const calls: Call[] = [];

function stubFetch(responder: (url: string, init?: RequestInit) => Response | Promise<Response>) {
	vi.stubGlobal('fetch', async (input: string | URL | Request, init?: RequestInit) => {
		const url = typeof input === 'string' ? input : String(input);
		calls.push({ url, init });
		return responder(url, init);
	});
}

const json = (body: unknown, status = 200) =>
	new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

afterEach(() => {
	calls.length = 0;
	vi.unstubAllGlobals();
});

const SERVER_CONFIG = {
	enabled: true,
	vapid_public_key: 'BAaaaa',
	threshold_options_pct: [40, 60, 80],
	lead_options_min: [20, 30, 45, 60],
	defaults: {
		threshold_pct: 60,
		lead_min: 30,
		quiet_hours: { enabled: false, start: '22:00', end: '07:00' }
	},
	capacity_reached: false
};

describe('fetchPushConfig', () => {
	it('camel-cases an enabled config', async () => {
		stubFetch(() => json(SERVER_CONFIG));
		const config = await fetchPushConfig();
		expect(calls[0].url).toBe('/api/push/config');
		expect(config).toEqual({
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
		});
	});

	it('carries capacity_reached through', async () => {
		stubFetch(() => json({ ...SERVER_CONFIG, capacity_reached: true }));
		expect((await fetchPushConfig()).capacityReached).toBe(true);
	});

	it('reads a disabled server as disabled', async () => {
		stubFetch(() => json({ enabled: false }));
		expect(await fetchPushConfig()).toMatchObject({ enabled: false, vapidPublicKey: null });
	});

	it('reads the SPA shell as disabled instead of throwing', async () => {
		stubFetch(
			() =>
				new Response('<!doctype html><html lang="da"><body>app</body></html>', {
					status: 200,
					headers: { 'content-type': 'text/html' }
				})
		);
		await expect(fetchPushConfig()).resolves.toMatchObject({ enabled: false });
	});

	it('reads a 404 or a dead network as disabled', async () => {
		stubFetch(() => json({ detail: 'not found' }, 404));
		expect((await fetchPushConfig()).enabled).toBe(false);
		vi.unstubAllGlobals();
		stubFetch(() => {
			throw new TypeError('Failed to fetch');
		});
		expect((await fetchPushConfig()).enabled).toBe(false);
	});

	it('treats an enabled config with no VAPID key as disabled', async () => {
		stubFetch(() => json({ ...SERVER_CONFIG, vapid_public_key: null }));
		expect((await fetchPushConfig()).enabled).toBe(false);
	});
});

const BODY = subscribeBody(
	{ endpoint: 'https://push.example/abc', keys: { p256dh: 'k', auth: 'a' } },
	55.6761,
	12.5683,
	FALLBACK_PREFS,
	'da',
	'Europe/Copenhagen'
);

describe('postSubscribe', () => {
	it('posts JSON and returns the server result', async () => {
		stubFetch(() => json({ ok: true, created: true }));
		await expect(postSubscribe(BODY)).resolves.toEqual({ ok: true, created: true });
		expect(calls[0].url).toBe('/api/push/subscribe');
		expect(calls[0].init?.method).toBe('POST');
		expect(JSON.parse(String(calls[0].init?.body))).toEqual(BODY);
	});

	it('maps 400 to OffCoverageError', async () => {
		stubFetch(() => json({ detail: 'coordinates outside the radar composite grid' }, 400));
		await expect(postSubscribe(BODY)).rejects.toBeInstanceOf(OffCoverageError);
	});

	it('maps 503 to PushUnavailableError', async () => {
		stubFetch(() => json({ detail: 'push disabled' }, 503));
		await expect(postSubscribe(BODY)).rejects.toBeInstanceOf(PushUnavailableError);
	});

	it('maps 422 and 500 to a plain Error', async () => {
		stubFetch(() => json({ detail: 'validation' }, 422));
		const err = await postSubscribe(BODY).catch((e: unknown) => e);
		expect(err).toBeInstanceOf(Error);
		expect(err).not.toBeInstanceOf(OffCoverageError);
		expect(err).not.toBeInstanceOf(PushUnavailableError);
		expect(String(err)).toContain('422');
	});
});

describe('postUnsubscribe', () => {
	it('posts the endpoint and reports whether a row went away', async () => {
		stubFetch(() => json({ ok: true, deleted: true }));
		await expect(postUnsubscribe('https://push.example/abc')).resolves.toEqual({
			ok: true,
			deleted: true
		});
		expect(JSON.parse(String(calls[0].init?.body))).toEqual({
			endpoint: 'https://push.example/abc'
		});
	});

	it('throws on a non-2xx', async () => {
		stubFetch(() => json({ detail: 'nope' }, 500));
		await expect(postUnsubscribe('https://push.example/abc')).rejects.toThrow('500');
	});
});
