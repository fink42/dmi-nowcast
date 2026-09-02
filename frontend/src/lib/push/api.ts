/**
 * The three `/api/push/*` calls, and the two failures worth their own type.
 *
 * Same-origin like everything else the app fetches, so the Cloudflare Access
 * cookie rides along without being touched here.
 */
import { apiUrl } from '$lib/nowcast/manifest';
import { DISABLED_CONFIG, parsePushConfig, type PushConfig, type SubscribeBody } from './prefs';

/** The chosen point is outside the radar composite: nothing could be sent. */
export class OffCoverageError extends Error {}

/** The server refuses subscriptions right now — switched off, or full. */
export class PushUnavailableError extends Error {}

export interface SubscribeResult {
	ok: boolean;
	created: boolean;
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
		throw new OffCoverageError(await detail(res, 'coordinates outside the radar composite grid'));
	}
	if (res.status === 503) {
		throw new PushUnavailableError(await detail(res, 'push notifications unavailable'));
	}
	if (!res.ok) throw new Error(`push subscribe: HTTP ${res.status}`);
	const result = (await res.json()) as Partial<SubscribeResult>;
	return { ok: result?.ok === true, created: result?.created === true };
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
