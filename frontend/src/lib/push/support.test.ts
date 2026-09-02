/**
 * Support detection, branch by branch. The iOS cases are the reason this
 * module exists, so they carry real user-agent strings.
 */
import { describe, expect, it } from 'vitest';
import { currentEnv, detectPushSupport, isIosLike, type PushEnv } from './support';

const UA = {
	iphone:
		'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1',
	ipadAsMac:
		'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15',
	mac: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36',
	android:
		'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Mobile Safari/537.36'
};

/** A capable, secure, non-iOS browser; overridden per case. */
const env = (over: Partial<PushEnv> = {}): PushEnv => ({
	hasServiceWorker: true,
	hasPushManager: true,
	hasNotification: true,
	isSecureContext: true,
	userAgent: UA.android,
	standalone: false,
	maxTouchPoints: 1,
	...over
});

describe('isIosLike', () => {
	it('recognises iPhone and iPad user agents', () => {
		expect(isIosLike(UA.iphone, 5)).toBe(true);
		expect(isIosLike('Mozilla/5.0 (iPad; CPU OS 17_5 like Mac OS X)', 5)).toBe(true);
	});

	it('recognises an iPad claiming to be a Mac by its touch points', () => {
		expect(isIosLike(UA.ipadAsMac, 5)).toBe(true);
	});

	it('does not mistake a real Mac for an iPad', () => {
		expect(isIosLike(UA.mac, 0)).toBe(false);
		expect(isIosLike(UA.ipadAsMac, 0)).toBe(false);
	});

	it('does not mistake Android for iOS', () => {
		expect(isIosLike(UA.android, 5)).toBe(false);
	});
});

describe('detectPushSupport', () => {
	it('reports a capable secure browser as supported', () => {
		expect(detectPushSupport(env())).toBe('supported');
	});

	it('reports an insecure context first, whatever else is true', () => {
		expect(detectPushSupport(env({ isSecureContext: false }))).toBe('insecure-context');
		expect(detectPushSupport(env({ isSecureContext: false, userAgent: UA.iphone }))).toBe(
			'insecure-context'
		);
	});

	it('reports a browser missing any of the three APIs as unsupported', () => {
		expect(detectPushSupport(env({ hasServiceWorker: false }))).toBe('unsupported');
		expect(detectPushSupport(env({ hasPushManager: false }))).toBe('unsupported');
		expect(detectPushSupport(env({ hasNotification: false }))).toBe('unsupported');
	});

	it('tells an iOS tab to install the app, even with no PushManager at all', () => {
		const ios = env({
			userAgent: UA.iphone,
			maxTouchPoints: 5,
			standalone: false,
			hasPushManager: false,
			hasNotification: false
		});
		expect(detectPushSupport(ios)).toBe('ios-not-installed');
	});

	it('says the same for an iPad reporting a Mac user agent', () => {
		expect(
			detectPushSupport(env({ userAgent: UA.ipadAsMac, maxTouchPoints: 5, hasPushManager: false }))
		).toBe('ios-not-installed');
	});

	it('supports an installed iOS app that exposes the APIs', () => {
		expect(
			detectPushSupport(env({ userAgent: UA.iphone, maxTouchPoints: 5, standalone: true }))
		).toBe('supported');
	});

	it('still reports unsupported for an installed iOS app on an old OS', () => {
		// iOS < 16.4 in standalone mode: installed, but no PushManager.
		expect(
			detectPushSupport(
				env({ userAgent: UA.iphone, maxTouchPoints: 5, standalone: true, hasPushManager: false })
			)
		).toBe('unsupported');
	});
});

describe('currentEnv', () => {
	it('answers "nothing available" off the browser instead of throwing', () => {
		const real = currentEnv();
		expect(real.hasServiceWorker).toBe(false);
		expect(real.isSecureContext).toBe(false);
		expect(real.userAgent).toBe('');
		expect(detectPushSupport(real)).toBe('insecure-context');
	});
});
