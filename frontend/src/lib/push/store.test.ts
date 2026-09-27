/**
 * The push store's subscribe flow against a fake PushManager. The bug pinned
 * here: any `pushManager.subscribe()` failure used to be read as "the server
 * rotated its key", and the existing — working — subscription was thrown
 * away. Only a subscription bound to a different key may be replaced.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { RateLimitedError } from './api';
import { FALLBACK_PREFS, type PushConfig } from './prefs';

vi.mock('$app/environment', () => ({ browser: true }));

const mocks = vi.hoisted(() => ({ postSubscribe: vi.fn() }));

vi.mock('./api', async (importOriginal) => ({
	...(await importOriginal<typeof import('./api')>()),
	postSubscribe: mocks.postSubscribe
}));

const { PushStore, errorKey } = await import('./store.svelte');

/** A 65-byte uncompressed P-256 point, as the server would publish it. */
const SERVER_KEY = new Uint8Array(65).map((_, i) => (i === 0 ? 4 : i));
const OTHER_KEY = new Uint8Array(65).map((_, i) => (i === 0 ? 4 : 200 - i));

function b64url(bytes: Uint8Array): string {
	return btoa(String.fromCharCode(...bytes))
		.replace(/\+/g, '-')
		.replace(/\//g, '_')
		.replace(/=+$/, '');
}

interface FakeSubscription {
	endpoint: string;
	options: { applicationServerKey: ArrayBuffer | null };
	unsubscribe: ReturnType<typeof vi.fn>;
	toJSON: () => PushSubscriptionJSON;
}

function subscription(endpoint: string, key: Uint8Array | null): FakeSubscription {
	return {
		endpoint,
		options: { applicationServerKey: key ? key.slice().buffer : null },
		unsubscribe: vi.fn(async () => true),
		toJSON: () => ({ endpoint, keys: { p256dh: 'p', auth: 'a' } })
	};
}

function install(existing: FakeSubscription | null, subscribe: () => Promise<FakeSubscription>) {
	const pushManager = {
		getSubscription: vi.fn(async () => existing),
		subscribe: vi.fn(subscribe)
	};
	vi.stubGlobal('Notification', { permission: 'granted', requestPermission: async () => 'granted' });
	vi.stubGlobal('navigator', { serviceWorker: { ready: Promise.resolve({ pushManager }) } });
	// No storage: the store copes (it has to, for private windows), and Node's
	// experimental global would only print a warning.
	vi.stubGlobal('localStorage', undefined);
	return pushManager;
}

function store() {
	const s = new PushStore();
	s.config = {
		enabled: true,
		vapidPublicKey: b64url(SERVER_KEY),
		defaults: FALLBACK_PREFS
	} as unknown as PushConfig;
	return s;
}

const OK = {
	ok: true,
	created: true,
	effectiveThresholdPct: 40,
	thresholdSource: 'table',
	fittedAtUtc: null
};

beforeEach(() => {
	mocks.postSubscribe.mockReset();
	mocks.postSubscribe.mockResolvedValue(OK);
	vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
	vi.unstubAllGlobals();
	vi.restoreAllMocks();
});

describe('subscribe with an existing subscription', () => {
	it('keeps a working subscription when subscribe() fails for another reason', async () => {
		const existing = subscription('https://push.example/old', SERVER_KEY);
		const pm = install(existing, async () => {
			throw new DOMException('push service unreachable', 'AbortError');
		});
		const s = store();
		await s.subscribe(55.7, 12.6, FALLBACK_PREFS);
		expect(pm.subscribe).toHaveBeenCalledTimes(1);
		expect(existing.unsubscribe).not.toHaveBeenCalled();
		expect(s.status).toBe('error');
		expect(s.error).toBe('failed');
	});

	it('replaces a subscription bound to a rotated server key', async () => {
		const existing = subscription('https://push.example/old', OTHER_KEY);
		const fresh = subscription('https://push.example/new', SERVER_KEY);
		const pm = install(existing, async () => fresh);
		const s = store();
		await s.subscribe(55.7, 12.6, FALLBACK_PREFS);
		expect(existing.unsubscribe).toHaveBeenCalledTimes(1);
		expect(pm.subscribe).toHaveBeenCalledTimes(1);
		expect(s.status).toBe('subscribed');
		expect(s.stored?.endpoint).toBe('https://push.example/new');
	});

	it('reuses a subscription on the same key without touching it', async () => {
		const existing = subscription('https://push.example/same', SERVER_KEY);
		install(existing, async () => existing);
		const s = store();
		await s.subscribe(55.7, 12.6, FALLBACK_PREFS);
		expect(existing.unsubscribe).not.toHaveBeenCalled();
		expect(s.status).toBe('subscribed');
	});

	it('does not tear down a pre-existing subscription when the server refuses the POST', async () => {
		const existing = subscription('https://push.example/same', SERVER_KEY);
		install(existing, async () => existing);
		mocks.postSubscribe.mockRejectedValueOnce(new RateLimitedError('slow down', 90));
		const s = store();
		await s.subscribe(55.7, 12.6, FALLBACK_PREFS);
		expect(existing.unsubscribe).not.toHaveBeenCalled();
		expect(s.error).toBe('rateLimited');
		expect(s.retryAfterSec).toBe(90);
		s.clearError();
		expect(s.retryAfterSec).toBeNull();
	});

	it('leaves the subscription alone when the browser will not say which key it has', async () => {
		const existing = subscription('https://push.example/old', null);
		install(existing, async () => {
			throw new DOMException('different key', 'InvalidStateError');
		});
		const s = store();
		await s.subscribe(55.7, 12.6, FALLBACK_PREFS);
		expect(existing.unsubscribe).not.toHaveBeenCalled();
		expect(s.error).toBe('failed');
	});
});

describe('errorKey', () => {
	it('maps the new server answers to their own messages', async () => {
		const api = await import('./api');
		expect(errorKey(new api.InvalidSubscriptionError('bad key'))).toBe('invalid');
		expect(errorKey(new api.RequestTooLargeError('big'))).toBe('tooLarge');
		expect(errorKey(new api.RateLimitedError('slow', null))).toBe('rateLimited');
		expect(errorKey(new api.OffCoverageError('out'))).toBe('offCoverage');
		expect(errorKey(new Error('x'))).toBe('failed');
	});
});
