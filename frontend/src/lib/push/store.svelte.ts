/**
 * Everything the notification UI needs to know, and the four operations it
 * can start: subscribe, change preferences, move the alert to another point,
 * stop.
 *
 * Three sources of truth have to be kept in step, and only one of them is
 * ours:
 *
 *  1. the browser's own `PushSubscription`, which the user can revoke from
 *     site settings without telling the page;
 *  2. the server's row, which disappears when the push service reports the
 *     endpoint gone;
 *  3. our localStorage copy, which is what lets the panel say *where* the
 *     alert is set and at what threshold.
 *
 * The copy is advisory. `init()` reconciles it against the browser at every
 * start-up, and anything that fails leaves `status: 'error'` with a catalog
 * key — the component picks the sentence, this module never holds prose.
 */
import { browser } from '$app/environment';
import { locale } from '$lib/i18n';
import {
	fetchPushConfig,
	fetchPushOptions,
	InvalidSubscriptionError,
	OffCoverageError,
	postSubscribe,
	postUnsubscribe,
	PushUnavailableError,
	RateLimitedError,
	RequestTooLargeError,
	type SubscribeResult
} from './api';
import { sameApplicationServerKey, urlBase64ToUint8Array } from './keys';
import {
	clearStored,
	loadStored,
	saveStored,
	subscribeBody,
	resolveTimeZone,
	type EffectiveThreshold,
	type PushConfig,
	type PushPrefs,
	type StoredSubscription
} from './prefs';
import { currentEnv, detectPushSupport, type PushSupport } from './support';
import {
	effectiveThreshold,
	FALLBACK_OPTIONS,
	thresholdOverrideFromUrl,
	type PushOptions
} from './thresholds';

export type PushStatus = 'idle' | 'loading' | 'subscribing' | 'subscribed' | 'error';

/** Keys into `t().push.errors` — never a sentence. */
export type PushErrorKey =
	| 'permission'
	| 'offCoverage'
	| 'unavailable'
	| 'invalid'
	| 'tooLarge'
	| 'rateLimited'
	| 'failed';

/**
 * `navigator.serviceWorker.ready` never rejects: on a page whose worker
 * failed to register it simply never settles, and an `await` on it would hang
 * the panel in its loading state for the life of the tab.
 */
const REGISTRATION_TIMEOUT_MS = 5_000;

async function readyRegistration(): Promise<ServiceWorkerRegistration | null> {
	try {
		if (typeof navigator === 'undefined' || !('serviceWorker' in navigator)) return null;
		return await Promise.race([
			navigator.serviceWorker.ready,
			new Promise<null>((resolve) => setTimeout(() => resolve(null), REGISTRATION_TIMEOUT_MS))
		]);
	} catch {
		return null;
	}
}

function readPermission(): NotificationPermission | 'unknown' {
	try {
		return typeof Notification === 'undefined' ? 'unknown' : Notification.permission;
	} catch {
		return 'unknown';
	}
}

/** Exported for tests; the app uses the `push` singleton below. */
export class PushStore {
	config = $state<PushConfig | null>(null);
	/** The horizons on offer and the threshold fitted for each. */
	options = $state<PushOptions>(FALLBACK_OPTIONS);
	/**
	 * The hidden `?threshold=NN` override this tab was opened with, or null.
	 * Read once at start-up: it is a power-user escape hatch, not a control,
	 * and a value that appeared mid-session would have nothing to explain it.
	 */
	urlOverride = $state<number | null>(null);
	support = $state<PushSupport | 'unknown'>('unknown');
	permission = $state<NotificationPermission | 'unknown'>('unknown');
	status = $state<PushStatus>('idle');
	stored = $state<StoredSubscription | null>(null);
	error = $state<PushErrorKey | null>(null);
	/**
	 * With `error: 'rateLimited'`, how long the server asked us to wait, in
	 * seconds — null when it did not say. Cleared with the error.
	 */
	retryAfterSec = $state<number | null>(null);

	#initialised = false;

	/** True while an operation is in flight — the buttons disable on this. */
	get busy(): boolean {
		return this.status === 'loading' || this.status === 'subscribing';
	}

	clearError(): void {
		this.error = null;
		this.retryAfterSec = null;
		if (this.status === 'error') this.status = this.stored ? 'subscribed' : 'idle';
	}

	/**
	 * Read the environment, the server config and the stored copy, then make
	 * the copy agree with what the browser actually holds. Runs once.
	 */
	async init(): Promise<void> {
		if (!browser || this.#initialised) return;
		this.#initialised = true;
		this.status = 'loading';
		this.support = detectPushSupport(currentEnv());
		this.permission = readPermission();
		this.urlOverride = thresholdOverrideFromUrl(
			typeof location === 'undefined' ? '' : location.search
		);
		// Two independent endpoints, neither of which throws: the config says
		// whether push works at all, the options say what may be chosen.
		const [config, options] = await Promise.all([fetchPushConfig(), fetchPushOptions()]);
		this.config = config;
		this.options = options;
		this.stored = loadStored(this.config);
		if (this.support !== 'supported' || !this.config.enabled) {
			this.status = 'idle';
			return;
		}
		await this.#reconcile();
		this.status = this.stored ? 'subscribed' : 'idle';
	}

	/**
	 * The browser is the authority on whether a subscription exists. A copy
	 * without one behind it (permissions reset, site data cleared on another
	 * tab, the push service dropped it) is a lie the panel must not tell.
	 */
	async #reconcile(): Promise<void> {
		const subscription = await this.#currentSubscription();
		if (!subscription) {
			if (this.stored) {
				clearStored();
				this.stored = null;
			}
			return;
		}
		// A subscription with no copy stays as it is: we do not know which
		// point or thresholds it was created with, and inventing them would
		// put a wrong sentence on screen. The panel offers to subscribe, and
		// the server upserts on the same endpoint.
		if (this.stored && this.stored.endpoint !== subscription.endpoint) {
			clearStored();
			this.stored = null;
		}
	}

	async #currentSubscription(): Promise<PushSubscription | null> {
		const registration = await readyRegistration();
		if (!registration) return null;
		try {
			return await registration.pushManager.getSubscription();
		} catch {
			return null;
		}
	}

	/**
	 * Ask the browser for a subscription bound to the server's VAPID key.
	 *
	 * `created` says whether this call is what brought it into existence —
	 * which decides, on a failed POST, whether tearing it down again is a
	 * clean-up or the destruction of a subscription that already works.
	 */
	async #ensureSubscription(
		registration: ServiceWorkerRegistration,
		key: Uint8Array
	): Promise<{ subscription: PushSubscription; created: boolean }> {
		const options: PushSubscriptionOptionsInit = {
			userVisibleOnly: true,
			applicationServerKey: key as BufferSource
		};
		const existing = await registration.pushManager.getSubscription();
		// The server rotated its VAPID key: the old subscription can never
		// receive another message, so it goes. Decided on the key itself, not
		// on `subscribe()` failing — that also fails for reasons that have
		// nothing to do with the key (a flaky push service, a storage error),
		// and unsubscribing then would destroy a subscription that works.
		const bound = existing
			? sameApplicationServerKey(existing.options?.applicationServerKey, key)
			: null;
		if (existing && bound === false) {
			await existing.unsubscribe();
			return { subscription: await registration.pushManager.subscribe(options), created: true };
		}
		// Same key (or a browser that will not say): the browser hands the
		// existing subscription back. Any failure is rethrown with the
		// existing subscription left exactly as it was.
		const subscription = await registration.pushManager.subscribe(options);
		return { subscription, created: existing === null };
	}

	/** Turn notifications on for a point, with the given preferences. */
	async subscribe(lat: number, lon: number, prefs: PushPrefs): Promise<void> {
		if (!browser) return;
		const config = this.config;
		if (!config?.enabled || !config.vapidPublicKey) {
			this.#fail('unavailable');
			return;
		}
		this.status = 'subscribing';
		this.error = null;

		let permission: NotificationPermission = 'denied';
		try {
			permission = await Notification.requestPermission();
		} catch {
			permission = 'denied';
		}
		this.permission = permission;
		if (permission !== 'granted') {
			// Not an error state: the browser's own prompt said no, and the
			// panel has a sentence for that.
			this.status = this.stored ? 'subscribed' : 'idle';
			this.error = 'permission';
			return;
		}

		const registration = await readyRegistration();
		if (!registration) {
			this.#fail('failed');
			return;
		}

		let subscription: PushSubscription;
		let created: boolean;
		try {
			const result = await this.#ensureSubscription(
				registration,
				urlBase64ToUint8Array(config.vapidPublicKey)
			);
			subscription = result.subscription;
			created = result.created;
		} catch (err) {
			console.warn('pushManager.subscribe failed', err);
			this.#fail('failed');
			return;
		}

		const lang = locale();
		const tz = resolveTimeZone();
		let result: SubscribeResult;
		try {
			result = await postSubscribe(subscribeBody(subscription.toJSON(), lat, lon, prefs, lang, tz));
		} catch (err) {
			// Nothing must dangle: a subscription this call created, which the
			// server never recorded, would leave the browser thinking it is
			// subscribed to a service that has never heard of it.
			if (created) {
				try {
					await subscription.unsubscribe();
				} catch {
					/* best effort */
				}
			}
			this.#fail(errorKey(err), err);
			return;
		}

		const stored: StoredSubscription = {
			endpoint: subscription.endpoint,
			lat,
			lon,
			prefs,
			effective: this.#effectiveOf(result, prefs),
			lang,
			tz,
			subscribedAt: new Date().toISOString()
		};
		saveStored(stored);
		this.stored = stored;
		this.status = 'subscribed';
		this.error = null;
	}

	/** Re-post the existing subscription with new preferences, same point. */
	async updatePrefs(prefs: PushPrefs): Promise<void> {
		const stored = this.stored;
		if (!browser || !stored) return;
		this.status = 'subscribing';
		this.error = null;
		const subscription = await this.#currentSubscription();
		if (!subscription) {
			// The browser lost it under us. Subscribing again at the same point
			// is what the user asked for anyway.
			await this.subscribe(stored.lat, stored.lon, prefs);
			return;
		}
		const lang = locale();
		const tz = resolveTimeZone();
		let result: SubscribeResult;
		try {
			result = await postSubscribe(
				subscribeBody(subscription.toJSON(), stored.lat, stored.lon, prefs, lang, tz)
			);
		} catch (err) {
			this.#fail(errorKey(err), err);
			return;
		}
		const next: StoredSubscription = {
			...stored,
			endpoint: subscription.endpoint,
			prefs,
			effective: this.#effectiveOf(result, prefs),
			lang,
			tz
		};
		saveStored(next);
		this.stored = next;
		this.status = 'subscribed';
		this.error = null;
	}

	/**
	 * Turn notifications off. Both halves are attempted and neither is
	 * allowed to block the other: whatever happens, this browser stops
	 * claiming to be subscribed.
	 */
	async unsubscribe(): Promise<void> {
		if (!browser) return;
		this.status = 'subscribing';
		this.error = null;
		const subscription = await this.#currentSubscription();
		const endpoint = subscription?.endpoint ?? this.stored?.endpoint ?? null;
		if (endpoint) {
			try {
				await postUnsubscribe(endpoint);
			} catch (err) {
				// The row may outlive this, but it dies on the first push that
				// comes back 404/410 from the push service.
				console.warn('push unsubscribe failed on the server', err);
			}
		}
		if (subscription) {
			try {
				await subscription.unsubscribe();
			} catch (err) {
				console.warn('pushManager.unsubscribe failed', err);
			}
		}
		clearStored();
		this.stored = null;
		this.status = 'idle';
		this.error = null;
	}

	/**
	 * What the server answered it will warn this device at.
	 *
	 * A server built before the fit existed answers without the field, and
	 * the same number is then derived from the options table — the two agree
	 * whenever both exist, and the response wins when they do not, because it
	 * is the row that was actually written.
	 */
	#effectiveOf(result: SubscribeResult, prefs: PushPrefs): EffectiveThreshold {
		if (result.effectiveThresholdPct !== null) {
			return {
				thresholdPct: result.effectiveThresholdPct,
				source: result.thresholdSource ?? 'fallback',
				fittedAtUtc: result.fittedAtUtc ?? this.options.fittedAtUtc
			};
		}
		const pick = effectiveThreshold(this.options, prefs.leadMin, prefs.thresholdPct);
		return {
			thresholdPct: pick.pct,
			source: pick.source,
			fittedAtUtc: this.options.fittedAtUtc
		};
	}

	#fail(key: PushErrorKey, err?: unknown): void {
		this.status = 'error';
		this.error = key;
		this.retryAfterSec = err instanceof RateLimitedError ? err.retryAfterSec : null;
	}
}

/** Exported for tests. */
export function errorKey(err: unknown): PushErrorKey {
	if (err instanceof OffCoverageError) return 'offCoverage';
	if (err instanceof PushUnavailableError) return 'unavailable';
	if (err instanceof InvalidSubscriptionError) return 'invalid';
	if (err instanceof RequestTooLargeError) return 'tooLarge';
	if (err instanceof RateLimitedError) return 'rateLimited';
	return 'failed';
}

export const push = new PushStore();
