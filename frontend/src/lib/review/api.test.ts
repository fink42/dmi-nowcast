/**
 * The annotation client against a stubbed `fetch`.
 *
 * The case worth the most here is the review server not running at all: the
 * dev server proxies `/review-api` unconditionally, so a reviewer who
 * forgot to start `scripts/review_server.py` gets the SPA shell — HTML,
 * HTTP 200 — and the page has to say "the server is not running" rather
 * than throw on the first HTML byte and take a half-written judgement with
 * it.
 *
 * Nothing in this module may reject. Every test therefore asserts on a
 * returned value, and a rejected promise is a failure by construction.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
	deleteAnnotation,
	exportBundle,
	fetchAnnotation,
	fetchAnnotations,
	health,
	saveAnnotation
} from './api';
import type { AnnotationDraft } from './schema';

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

const shell = () =>
	new Response('<!doctype html><html lang="da"><body>app</body></html>', {
		status: 200,
		headers: { 'content-type': 'text/html' }
	});

afterEach(() => {
	calls.length = 0;
	vi.unstubAllGlobals();
});

const ROW = {
	bundle_id: 'rev-fixture-0001',
	event_id: 'fa-06126-20260612T1345Z',
	station_id: '06126',
	anchor_utc: '2026-06-12T13:45:00Z',
	event_class: 'false_alarm',
	dual_truth: 'radar_wet_gauge_dry',
	season: 'summer',
	region: 'Funen',
	verdict: 'metric_artefact',
	tags: ['fa_virga_or_aloft'],
	vocab_version: 1,
	confidence: 2,
	needs_second_look: false,
	note: 'column-max over a dry gauge',
	cursor_utc: '2026-06-12T13:45:00Z',
	review_seq: 7,
	reviewer: 'nsimonsen',
	created_utc: '2026-09-15T10:00:00Z',
	updated_utc: '2026-09-15T10:04:00Z',
	revision: 3
};

const DRAFT: AnnotationDraft = {
	verdict: 'metric_artefact',
	tags: ['fa_virga_or_aloft'],
	confidence: 2,
	needs_second_look: false,
	note: 'column-max over a dry gauge',
	cursor_utc: '2026-06-12T13:45:00Z',
	vocab_version: 1
};

describe('health', () => {
	it('reports the bundle the server is serving', async () => {
		stubFetch(() =>
			json({
				ok: true,
				bundle_id: 'rev-fixture-0001',
				bundle_root: '/srv/review/rev-fixture-0001',
				events: 300,
				stored: 12,
				annotated: 9,
				schema_version: 1,
				vocab_version: 1,
				db_path: '/srv/review/annotations.sqlite'
			})
		);
		const result = await health();
		expect(calls[0].url).toBe('/review-api/health');
		expect(result.ok && result.value.bundleId).toBe('rev-fixture-0001');
		expect(result.ok && result.value.annotated).toBe(9);
	});

	it('reads the SPA shell as "the server is not running"', async () => {
		stubFetch(shell);
		const result = await health();
		expect(result.ok).toBe(false);
		expect(!result.ok && result.error.kind).toBe('offline');
		expect(!result.ok && result.error.message).toMatch(/review_server/);
	});

	it('reads a dead socket as offline rather than throwing', async () => {
		stubFetch(() => {
			throw new TypeError('Failed to fetch');
		});
		const result = await health();
		expect(!result.ok && result.error.kind).toBe('offline');
		expect(!result.ok && result.error.status).toBeNull();
	});
});

describe('fetchAnnotations', () => {
	it('parses the stored rows, identity columns and all', async () => {
		stubFetch(() => json({ bundle_id: ROW.bundle_id, annotations: [ROW] }));
		const result = await fetchAnnotations();
		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.value).toHaveLength(1);
		expect(result.value[0].verdict).toBe('metric_artefact');
		expect(result.value[0].station_id).toBe('06126');
		expect(result.value[0].reviewer).toBe('nsimonsen');
		expect(result.value[0].revision).toBe(3);
	});

	it('drops a row it cannot read and keeps the others', async () => {
		stubFetch(() => json({ annotations: [ROW, { note: 'no event id' }] }));
		const result = await fetchAnnotations();
		expect(result.ok && result.value).toHaveLength(1);
	});

	it('reads a verdict it has never heard of as no verdict', async () => {
		stubFetch(() => json({ annotations: [{ ...ROW, verdict: 'probably_fine' }] }));
		const result = await fetchAnnotations();
		expect(result.ok && result.value[0].verdict).toBeNull();
	});

	it('reports a body of the wrong shape as malformed, not as empty', async () => {
		stubFetch(() => json({ annotations: 'lots' }));
		const result = await fetchAnnotations();
		expect(!result.ok && result.error.kind).toBe('malformed');
	});
});

describe('fetchAnnotation', () => {
	it('returns null for an event nobody has judged yet', async () => {
		stubFetch(() => json({ error: 'not_annotated', event_id: 'x' }, 404));
		const result = await fetchAnnotation('x');
		// Not an error: most events are unjudged.
		expect(result.ok).toBe(true);
		expect(result.ok && result.value).toBeNull();
	});
});

describe('saveAnnotation', () => {
	it('PUTs the draft and sends the revision it was based on', async () => {
		stubFetch(() => json(ROW));
		const result = await saveAnnotation(ROW.event_id, DRAFT, { ifMatch: 3 });
		expect(result.ok).toBe(true);
		expect(calls[0].url).toBe(`/review-api/annotations/${ROW.event_id}`);
		expect(calls[0].init?.method).toBe('PUT');
		expect(JSON.parse(String(calls[0].init?.body))).toEqual(DRAFT);
		expect((calls[0].init?.headers as Record<string, string>)['If-Match']).toBe('"3"');
	});

	it('sends no If-Match for an event that has never been saved', async () => {
		stubFetch(() => json(ROW));
		await saveAnnotation(ROW.event_id, DRAFT, { ifMatch: null });
		expect((calls[0].init?.headers as Record<string, string>)['If-Match']).toBeUndefined();
	});

	it('reports a revision conflict with the revision the server holds', async () => {
		stubFetch(() => json({ error: 'revision_conflict', expected: 3, stored: 5 }, 409));
		const result = await saveAnnotation(ROW.event_id, DRAFT, { ifMatch: 3 });
		expect(!result.ok && result.error.kind).toBe('conflict');
		expect(!result.ok && result.error.stored).toBe(5);
	});

	it('carries the server’s per-field problems back to the form', async () => {
		stubFetch(() =>
			json(
				{
					error: 'validation_failed',
					problems: [
						{ field: 'tags', offending: ['fa_made_up'], message: 'unknown tag code(s)' }
					]
				},
				422
			)
		);
		const result = await saveAnnotation(ROW.event_id, DRAFT);
		expect(!result.ok && result.error.kind).toBe('validation');
		expect(!result.ok && result.error.problems[0].field).toBe('tags');
		expect(!result.ok && result.error.problems[0].offending).toEqual(['fa_made_up']);
	});

	it('escapes an event id rather than building a broken path', async () => {
		stubFetch(() => json(ROW));
		await saveAnnotation('fa/06126 20260612', DRAFT);
		expect(calls[0].url).toBe('/review-api/annotations/fa%2F06126%2020260612');
	});

	it('never rejects, whatever the server does', async () => {
		for (const responder of [
			() => json({ error: 'internal_error' }, 500),
			() => shell(),
			() => {
				throw new TypeError('Failed to fetch');
			}
		]) {
			vi.unstubAllGlobals();
			stubFetch(responder);
			await expect(saveAnnotation(ROW.event_id, DRAFT)).resolves.toMatchObject({ ok: false });
		}
	});
});

describe('deleteAnnotation', () => {
	it('deletes and says whether a row went away', async () => {
		stubFetch(() => json({ event_id: ROW.event_id, deleted: true, was: ROW }));
		const result = await deleteAnnotation(ROW.event_id, { ifMatch: 3 });
		expect(result.ok && result.value.deleted).toBe(true);
		expect(calls[0].init?.method).toBe('DELETE');
		expect((calls[0].init?.headers as Record<string, string>)['If-Match']).toBe('"3"');
	});

	it('treats deleting what was never there as the end state asked for', async () => {
		stubFetch(() => json({ error: 'not_annotated' }, 404));
		const result = await deleteAnnotation('x');
		expect(result.ok && result.value).toEqual({ eventId: 'x', deleted: false });
	});
});

describe('exportBundle', () => {
	it('POSTs the format and returns where the file landed', async () => {
		stubFetch(() => json({ path: '/srv/review/exports/rev.parquet', rows: 212, format: 'parquet' }));
		const result = await exportBundle('parquet');
		expect(calls[0].url).toBe('/review-api/export?format=parquet');
		expect(calls[0].init?.method).toBe('POST');
		expect(result.ok && result.value.rows).toBe(212);
	});

	it('reports a missing pyarrow as something the server cannot do right now', async () => {
		stubFetch(() => json({ error: 'pyarrow_missing', detail: 'the parquet export needs pyarrow' }, 503));
		const result = await exportBundle('parquet');
		expect(!result.ok && result.error.kind).toBe('unavailable');
		expect(!result.ok && result.error.message).toMatch(/pyarrow/);
	});
});
