/**
 * The annotation server's client — `scripts/review_server.py` over
 * `/review-api/`.
 *
 * Nothing here throws. A reviewer three hours into a bundle has two
 * hundred judgements behind them and a laptop that may have suspended, the
 * loopback server may have been restarted, and the one outcome that must
 * never happen is a rejected promise taking the page down with the draft in
 * it. So every call answers with a tagged result and the UI decides what to
 * say — "server not running", "someone else edited this", "the server
 * refused this tag" are three different sentences and a thrown `Error`
 * flattens them into one.
 *
 * The failure kinds map to the server's own vocabulary:
 *
 *  - `offline`    — fetch itself failed, or answered with the SPA shell
 *                   because nothing is listening on the proxy target.
 *  - `not_found`  — 404. For an annotation that simply means "never saved",
 *                   which is not an error at all and is turned into a null
 *                   by `fetchAnnotation`.
 *  - `conflict`   — 409, an `If-Match` that no longer matches: the row was
 *                   edited elsewhere. The stored revision comes back so the
 *                   UI can offer to re-read rather than clobber.
 *  - `validation` — 422 with per-field problems, which the form shows
 *                   against the fields they name.
 *  - `unavailable`— 503, the export without pyarrow installed.
 *  - `server`     — anything else with a status.
 *  - `malformed`  — a 2xx whose body was not the shape we asked for.
 */
import type { Annotation, AnnotationDraft, DualTruthClass, OutcomeClass, Verdict } from './schema';

export const REVIEW_API_PREFIX = '/review-api/';

export const apiUrl = (route: string): string =>
	REVIEW_API_PREFIX + String(route).replace(/^\/+/, '');

export type FailureKind =
	| 'offline'
	| 'not_found'
	| 'conflict'
	| 'validation'
	| 'unavailable'
	| 'server'
	| 'malformed';

export interface FieldProblem {
	field: string;
	message: string;
	offending: string[];
}

export interface ApiFailure {
	kind: FailureKind;
	/** Null when the request never reached the server. */
	status: number | null;
	/** The server's own `error` code, or a local one. */
	code: string;
	message: string;
	problems: FieldProblem[];
	/** The revision the server holds, on a conflict. */
	stored: number | null;
}

export type ApiResult<T> = { ok: true; value: T } | { ok: false; error: ApiFailure };

const ok = <T>(value: T): ApiResult<T> => ({ ok: true, value });

const fail = (
	kind: FailureKind,
	status: number | null,
	code: string,
	message: string,
	extra: Partial<Pick<ApiFailure, 'problems' | 'stored'>> = {}
): ApiResult<never> => ({
	ok: false,
	error: { kind, status, code, message, problems: [], stored: null, ...extra }
});

export interface ReviewHealth {
	bundleId: string;
	bundleRoot: string;
	events: number;
	stored: number;
	annotated: number;
	schemaVersion: number;
	vocabVersion: number;
	dbPath: string;
}

export interface ExportResult {
	path: string;
	rows: number;
	format: ExportFormat;
}

export type ExportFormat = 'parquet' | 'md' | 'csv';

type Obj = Record<string, unknown>;

const isObject = (value: unknown): value is Obj =>
	typeof value === 'object' && value !== null && !Array.isArray(value);

const str = (value: unknown): string | null =>
	typeof value === 'string' && value.trim() !== '' ? value : null;

const int = (value: unknown): number | null =>
	typeof value === 'number' && Number.isFinite(value) ? Math.round(value) : null;

/** The server's error body, or a stand-in when it did not send one. */
async function failureFrom(res: Response): Promise<ApiResult<never>> {
	let body: unknown = null;
	try {
		body = await res.json();
	} catch {
		// A non-JSON body on an error status is the SPA shell or a proxy page.
		body = null;
	}
	const payload = isObject(body) ? body : {};
	const code = str(payload.error) ?? `http_${res.status}`;
	const detail = str(payload.detail) ?? str(payload.message);
	const problems = Array.isArray(payload.problems)
		? payload.problems
				.map((problem) => {
					if (!isObject(problem)) return null;
					return {
						field: str(problem.field) ?? '_body',
						message: str(problem.message) ?? '',
						offending: Array.isArray(problem.offending)
							? problem.offending.map((value) => String(value))
							: []
					};
				})
				.filter((problem): problem is FieldProblem => problem !== null)
		: [];

	if (res.status === 404) {
		return fail('not_found', 404, code, detail ?? 'not found');
	}
	if (res.status === 409) {
		return fail('conflict', 409, code, detail ?? 'this row was edited elsewhere', {
			stored: int(payload.stored)
		});
	}
	if (res.status === 422) {
		return fail('validation', 422, code, detail ?? 'the server refused this judgement', {
			problems
		});
	}
	if (res.status === 503) {
		return fail('unavailable', 503, code, detail ?? 'the server cannot do that right now');
	}
	return fail('server', res.status, code, detail ?? `HTTP ${res.status}`);
}

/**
 * One request, with every failure turned into a value.
 *
 * The `offline` case covers more than a dead socket: with the review server
 * not running, vite's proxy answers the SPA shell with HTTP 200, so a body
 * that will not parse as JSON is read the same way `push/api.ts` reads it —
 * "this deployment has no such server" rather than "the site is broken".
 */
async function request<T>(
	route: string,
	init: RequestInit,
	parse: (body: unknown) => T | null
): Promise<ApiResult<T>> {
	let res: Response;
	try {
		res = await fetch(apiUrl(route), { cache: 'no-store', ...init });
	} catch (error) {
		return fail('offline', null, 'network', String(error));
	}
	if (!res.ok) return failureFrom(res);
	let body: unknown;
	try {
		body = await res.json();
	} catch {
		return fail(
			'offline',
			res.status,
			'not_json',
			'the review server did not answer — is scripts/review_server.py running?'
		);
	}
	const parsed = parse(body);
	if (parsed === null) {
		return fail('malformed', res.status, 'malformed', `${route}: unexpected response shape`);
	}
	return ok(parsed);
}

const VERDICTS: readonly Verdict[] = ['real_failure', 'metric_artefact', 'unclear'];

const DUAL_TRUTH: readonly DualTruthClass[] = [
	'both_wet',
	'radar_wet_gauge_dry',
	'gauge_wet_radar_dry',
	'both_dry'
];

/**
 * One stored row.
 *
 * Defensive even though the rows come from our own SQLite: the database
 * survives schema migrations and bundles rebuilt underneath it, and a row
 * whose verdict column holds something this client has never heard of must
 * read as "no verdict" rather than be shown as one.
 */
function parseAnnotation(raw: unknown): Annotation | null {
	if (!isObject(raw)) return null;
	const eventId = str(raw.event_id);
	if (eventId === null) return null;
	const verdict = VERDICTS.find((known) => known === raw.verdict) ?? null;
	return {
		bundle_id: str(raw.bundle_id) ?? '',
		event_id: eventId,
		station_id: str(raw.station_id) ?? '',
		anchor_utc: str(raw.anchor_utc) ?? '',
		// Preserved as the server wrote it rather than dropped. A class this
		// client has not heard of is a bundle newer than the page, and losing
		// the row would show a judged event as unjudged — which ends with the
		// reviewer judging it twice and colliding with their own revision.
		event_class: (str(raw.event_class) ?? '') as OutcomeClass,
		dual_truth: DUAL_TRUTH.find((known) => known === raw.dual_truth) ?? null,
		season: str(raw.season) ?? '',
		region: str(raw.region) ?? '',
		reviewer: str(raw.reviewer) ?? '',
		verdict,
		tags: Array.isArray(raw.tags)
			? raw.tags.filter((tag): tag is string => typeof tag === 'string')
			: [],
		vocab_version: int(raw.vocab_version) ?? 0,
		confidence: int(raw.confidence),
		needs_second_look: raw.needs_second_look === true,
		note: typeof raw.note === 'string' ? raw.note : '',
		cursor_utc: str(raw.cursor_utc),
		review_seq: int(raw.review_seq) ?? 0,
		created_utc: str(raw.created_utc) ?? '',
		updated_utc: str(raw.updated_utc) ?? '',
		revision: int(raw.revision) ?? 0
	};
}

/** Is the server up, and which bundle is it serving? */
export function health(signal?: AbortSignal): Promise<ApiResult<ReviewHealth>> {
	return request('health', { method: 'GET', signal }, (body) => {
		if (!isObject(body) || body.ok !== true) return null;
		const bundleId = str(body.bundle_id);
		if (bundleId === null) return null;
		return {
			bundleId,
			bundleRoot: str(body.bundle_root) ?? '',
			events: int(body.events) ?? 0,
			stored: int(body.stored) ?? 0,
			annotated: int(body.annotated) ?? 0,
			schemaVersion: int(body.schema_version) ?? 0,
			vocabVersion: int(body.vocab_version) ?? 0,
			dbPath: str(body.db_path) ?? ''
		};
	});
}

/**
 * Every judgement for the served bundle, in the order they were first
 * saved. Rows that do not parse are dropped rather than failing the call:
 * one unreadable row must not cost the reviewer the other two hundred.
 */
export function fetchAnnotations(signal?: AbortSignal): Promise<ApiResult<Annotation[]>> {
	return request('annotations', { method: 'GET', signal }, (body) => {
		if (!isObject(body) || !Array.isArray(body.annotations)) return null;
		return body.annotations
			.map(parseAnnotation)
			.filter((row): row is Annotation => row !== null);
	});
}

/**
 * One judgement, or null when this event has never been saved.
 *
 * The server answers 404 `not_annotated` for that, which is an ordinary
 * state — most events are unjudged — so it comes back as a successful null
 * rather than as a failure the UI would have to special-case.
 */
export async function fetchAnnotation(
	eventId: string,
	signal?: AbortSignal
): Promise<ApiResult<Annotation | null>> {
	const result = await request(
		`annotations/${encodeURIComponent(eventId)}`,
		{ method: 'GET', signal },
		parseAnnotation
	);
	if (!result.ok && result.error.kind === 'not_found') return ok(null);
	return result;
}

export interface SaveOptions {
	/**
	 * The revision the draft was based on, sent as `If-Match`. Omitting it
	 * means "last write wins", which is right for the first save of an event
	 * and wrong for every later one: two tabs open on the same bundle is a
	 * normal way to work and the second save must be told, not silently win.
	 */
	ifMatch?: number | null;
	signal?: AbortSignal;
}

/** Create or update one judgement. */
export function saveAnnotation(
	eventId: string,
	draft: AnnotationDraft,
	options: SaveOptions = {}
): Promise<ApiResult<Annotation>> {
	const headers: Record<string, string> = { 'content-type': 'application/json' };
	if (options.ifMatch !== undefined && options.ifMatch !== null) {
		headers['If-Match'] = `"${options.ifMatch}"`;
	}
	return request(
		`annotations/${encodeURIComponent(eventId)}`,
		{ method: 'PUT', headers, body: JSON.stringify(draft), signal: options.signal },
		parseAnnotation
	);
}

/**
 * Remove one judgement. A row that was never there comes back as
 * `deleted: false` rather than as a failure — the end state the caller
 * asked for is the end state either way.
 */
export async function deleteAnnotation(
	eventId: string,
	options: SaveOptions = {}
): Promise<ApiResult<{ eventId: string; deleted: boolean }>> {
	const headers: Record<string, string> = {};
	if (options.ifMatch !== undefined && options.ifMatch !== null) {
		headers['If-Match'] = `"${options.ifMatch}"`;
	}
	const result = await request(
		`annotations/${encodeURIComponent(eventId)}`,
		{ method: 'DELETE', headers, signal: options.signal },
		(body) => {
			if (!isObject(body)) return null;
			return { eventId: str(body.event_id) ?? eventId, deleted: body.deleted === true };
		}
	);
	if (!result.ok && result.error.kind === 'not_found') return ok({ eventId, deleted: false });
	return result;
}

/**
 * Write the judgements out into `<bundle>/exports/`.
 *
 * Parquet is the default because the rest of the project's analysis is
 * DuckDB over Parquet; `md` and `csv` exist for the same numbers without
 * pyarrow, which the server says so explicitly when it is missing (503).
 */
export function exportBundle(
	format: ExportFormat = 'parquet',
	signal?: AbortSignal
): Promise<ApiResult<ExportResult>> {
	return request(
		`export?format=${encodeURIComponent(format)}`,
		{ method: 'POST', signal },
		(body) => {
			if (!isObject(body)) return null;
			const path = str(body.path);
			if (path === null) return null;
			const answered = str(body.format);
			return {
				path,
				rows: int(body.rows) ?? 0,
				format: (answered === 'parquet' || answered === 'md' || answered === 'csv'
					? answered
					: format) as ExportFormat
			};
		}
	);
}
