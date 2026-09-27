/**
 * The three `/api/push/*` calls, and the failures worth their own type.
 *
 * Same-origin like everything else the app fetches, so the Cloudflare Access
 * cookie rides along without being touched here.
 */
import { apiUrl } from '$lib/nowcast/manifest';
import { DISABLED_CONFIG, parsePushConfig, type PushConfig, type SubscribeBody } from './prefs';
import {
	FALLBACK_OPTIONS,
	parsePushOptions,
	type PushOptions,
	type ThresholdSource
} from './thresholds';

/** The chosen point is outside the radar composite: nothing could be sent. */
export class OffCoverageError extends Error {}

/** The server refuses subscriptions right now — switched off, or full. */
export class PushUnavailableError extends Error {}

/**
 * A 400 that is not about the point: the server could not use the
 * subscription itself (its keys, a push service it does not accept, the
 * time zone). Trying again with the same browser state will not help.
 */
export class InvalidSubscriptionError extends Error {}

/** 413: the request body was larger than the server accepts. */
export class RequestTooLargeError extends Error {}

/** 429: too many requests from this client; `retryAfterSec` when stated. */
export class RateLimitedError extends Error {
	constructor(
		message: string,
		readonly retryAfterSec: number | null
	) {
		super(message);
	}
}

/**
 * `Retry-After` in seconds: the header is either a number of seconds or an
 * HTTP date. Null when absent or unreadable — never a negative wait.
 */
export function parseRetryAfter(value: string | null, nowMs: number = Date.now()): number | null {
	if (value === null || value.trim() === '') return null;
	const trimmed = value.trim();
	if (/^\d+$/.test(trimmed)) return Number(trimmed);
	// An HTTP date always names its day and month; without letters this is
	// junk like "-5", which `Date.parse` would happily read as a year.
	if (!/[a-z]/i.test(trimmed)) return null;
	const at = Date.parse(trimmed);
	if (!Number.isFinite(at)) return null;
	return Math.max(0, Math.ceil((at - nowMs) / 1000));
}

/**
 * The sidecar answers 400 for an off-coverage point and for a subscription it
 * cannot use, and only the `detail` tells them apart. Its off-coverage detail
 * is "coordinates outside the radar composite grid".
 */
const OFF_COVERAGE_DETAIL = /outside|coverage/i;

export interface SubscribeResult {
	ok: boolean;
	created: boolean;
	/**
	 * The threshold the server says this device will be warned at, and where
	 * it got it. Null on a server built before the fit existed — the panel
	 * then reads the same number out of `/api/push/options` instead.
	 */
	effectiveThresholdPct: number | null;
	thresholdSource: ThresholdSource | null;
	fittedAtUtc: string | null;
}

export interface UnsubscribeResult {
	ok: boolean;
	deleted: boolean;
}

/** The server's `detail`, when it sent one worth logging. */
async function detail(res: Response, fallback: string): Promise<string> {
	try {
		const body = (await res.json()) as { detail?: unknown };
		return typeof body?.detail === 'string' ? body.detail : fallback;
	} catch {
		return fallback;
	}
}

/**
 * Push configuration, or "disabled" for anything that is not a well-formed
 * enabled config.
 *
 * This one never throws, on purpose. A sidecar built before the feature
 * existed answers `/api/push/config` with the SPA shell — HTML, HTTP 200 —
 * and the only sensible reading of that is "this deployment has no push", not
 * "the site is broken". Same for a network failure.
 */
export async function fetchPushConfig(signal?: AbortSignal): Promise<PushConfig> {
	try {
		const res = await fetch(apiUrl('/api/push/config'), { signal, cache: 'no-cache' });
		if (!res.ok) return DISABLED_CONFIG;
		return parsePushConfig(await res.json());
	} catch {
		return DISABLED_CONFIG;
	}
}

/**
 * The horizons on offer and the threshold fitted for each, or the fallback
 * options for anything that is not a well-formed answer.
 *
 * Like `fetchPushConfig`, this never throws: a sidecar built before the fit
 * existed answers with the SPA shell, and the honest reading of that is "no
 * table here yet", which is exactly what the fallback says.
 */
export async function fetchPushOptions(signal?: AbortSignal): Promise<PushOptions> {
	try {
		const res = await fetch(apiUrl('/api/push/options'), { signal, cache: 'no-cache' });
		if (!res.ok) return FALLBACK_OPTIONS;
		return parsePushOptions(await res.json());
	} catch {
		return FALLBACK_OPTIONS;
	}
}

async function postJson(path: string, body: unknown, signal?: AbortSignal): Promise<Response> {
	return fetch(apiUrl(path), {
		method: 'POST',
		headers: { 'content-type': 'application/json' },
		body: JSON.stringify(body),
		signal
	});
}

/**
 * Create or update this browser's subscription. Re-posting the same endpoint
 * is how a preference change is saved — the server upserts.
 */
export async function postSubscribe(
	body: SubscribeBody,
	signal?: AbortSignal
): Promise<SubscribeResult> {
	const res = await postJson('/api/push/subscribe', body, signal);
	if (res.status === 400) {
		const reason = await detail(res, 'bad request');
		if (OFF_COVERAGE_DETAIL.test(reason)) throw new OffCoverageError(reason);
		throw new InvalidSubscriptionError(reason);
	}
	if (res.status === 413) {
		throw new RequestTooLargeError(await detail(res, 'request too large'));
	}
	if (res.status === 429) {
		const retryAfter = parseRetryAfter(res.headers.get('retry-after'));
		throw new RateLimitedError(await detail(res, 'rate limited'), retryAfter);
	}
	if (res.status === 503) {
		throw new PushUnavailableError(await detail(res, 'push notifications unavailable'));
	}
	if (!res.ok) throw new Error(`push subscribe: HTTP ${res.status}`);
	const result = (await res.json()) as Record<string, unknown>;
	const pct = result?.effective_threshold_pct;
	const source = result?.threshold_source;
	const fittedAt = result?.fitted_at_utc;
	return {
		ok: result?.ok === true,
		created: result?.created === true,
		effectiveThresholdPct:
			typeof pct === 'number' && Number.isFinite(pct) ? Math.round(pct) : null,
		thresholdSource:
			source === 'table' || source === 'override' || source === 'fallback' ? source : null,
		fittedAtUtc: typeof fittedAt === 'string' && fittedAt ? fittedAt : null
	};
}

export async function postUnsubscribe(
	endpoint: string,
	signal?: AbortSignal
): Promise<UnsubscribeResult> {
	const res = await postJson('/api/push/unsubscribe', { endpoint }, signal);
	if (!res.ok) throw new Error(`push unsubscribe: HTTP ${res.status}`);
	const result = (await res.json()) as Partial<UnsubscribeResult>;
	return { ok: result?.ok === true, deleted: result?.deleted === true };
}
