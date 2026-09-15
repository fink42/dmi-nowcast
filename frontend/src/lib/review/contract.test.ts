/**
 * What the producer writes, and what this client reads.
 *
 * `generated.test.ts` asserts that every block still PARSES — it catches a
 * rename or a restructure. It cannot catch the other half of drift: a field
 * the producer has been writing all along that nothing here ever reads.
 * Those are invisible by construction, because a parser that ignores a key
 * behaves exactly like one that does not know it exists, and the first
 * symptom is a reviewer asking why the page does not show something the
 * bundle plainly contains.
 *
 * So this walks the real bundle's OWN keys and asserts each one is either
 * parsed or on the ignore-list below. The list is the point of the file: it
 * turns "we do not read this" from an accident into a decision somebody
 * wrote down and can be argued with. Adding a field to the producer fails
 * this test until someone either reads it or says in one line why not.
 *
 * The same walk runs over a live `review_server.py` response, so the
 * annotation store gets the protection the builder has.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import generated from './generated-fixture.json';
import { fetchAnnotations, health } from './api';
import { parseEvent, parseIndex, parseManifest, parseVocabulary } from './load';

const raw = generated as unknown as {
	manifest: Record<string, unknown>;
	index: Record<string, unknown>;
	vocabulary: Record<string, unknown>;
	details: Record<string, Record<string, unknown>>;
	annotations: Record<string, unknown>;
	health: Record<string, unknown>;
};

/**
 * Paths whose VALUE is a map keyed by data — a station id, a date, a lead,
 * a feature name. Walking into one would report every key in the bundle as
 * an unread field, and the keys are content rather than contract.
 */
const DATA_KEYED: readonly string[] = [
	'manifest.truth.known_until',
	'manifest.feature_doc',
	'manifest.features.rows_by_day',
	'manifest.features.documentation',
	'manifest.frames.missing_reasons',
	'manifest.corpus.probability_fill',
	'manifest.sampling.targets',
	'manifest.sampling.drawn',
	'manifest.sampling.population',
	'index.events[].stratum',
	'details.decisions[].p_rain',
	'details.decisions[].p_post',
	'details.decisions[].features',
	'vocabulary.classes'
];

/**
 * Fields the producer writes that this client deliberately does not read.
 *
 * Every entry is a decision, not an oversight. Re-read the reason before
 * deleting one: several of these are numbers that would be actively
 * misleading in the UI, not merely unused.
 */
const NOT_READ: Record<string, string> = {
	// --- provenance for a human reading the JSON, not for the page ---------
	'manifest.frames.bytes_written_this_run':
		'how much this run wrote, as opposed to what the bundle holds. A build statistic; the page shows the bundle.',
	'manifest.frames.encodings.observed.product':
		'the source column name, already implied by the units and the reduction sentence beside it.',

	// --- the detail window's own copies of manifest-level geometry ---------
	'details.window.decision_min':
		'the manifest states the window geometry once; a per-event copy would be a second place for it to disagree.',
	'details.window.frame_pad_min': 'same as decision_min: manifest-level geometry.',
	'details.window.frame_cadence_min':
		'the cadence is in manifest.frames.cadence_min; the track is time-proportional and does not assume one.',
	'details.window.verdict':
		'the verdict window travels on dual_truth.window_used, which is what the badge is rendered from. Two copies, one read.',

	// --- restatements of numbers the page derives or shows elsewhere -------
	'details.gauge.station_id': 'the event already names its station.',
	'details.gauge.wet_rule':
		'the rule is stated once in manifest.truth and shown in the methods panel.',
	'details.gauge.onset_rule': 'as wet_rule: stated once, at manifest level.',
	'details.gauge.onset_utc': 'the index row carries the event onset; gauge.onsets carries them all.',
	'details.gauge.onset_two_slot_mm': 'on the index row, and on each gauge.onsets entry.',
	'details.gauge.known_slots_in_window':
		'derivable from the slots themselves, and the strip counts them as it draws.',
	'details.gauge.suspect_month':
		'surfaced as the suspect_gauge_month event FLAG, which is what the list filters on.',
	'details.radar_disc.source_column':
		'which column the disc was read from; the statistic sentence beside it says the same thing.',
	'details.radar_disc.same_instrument_as_the_forecast':
		'always true, and said in words by dualTruthLabel’s caveat rather than as a boolean.',
	'details.neighbours.caveat':
		'the "a wet neighbour says rain existed in the area" sentence, which the UI states in its own words.',
	'details.dual_truth.caveat': 'the same-instrument caveat, rendered from dualTruthLabel.',
	'details.dual_truth.neighbour_n_known': 'on the index row, which is what the list filters on.',
	'details.station.known_until_utc': 'the window block carries the horizon the track draws.',
	'details.station.dead_gauge':
		'a dead gauge is excluded upstream; manifest.truth.dead_gauges lists them for the methods panel.',
	'details.decisions[].frame_age_min_stored':
		'what the archive recorded, against the replayed frame_age_min. A builder cross-check, not a review one.',
	'details.decisions[].latest_estimate':
		'the producer’s own "newest row at this instant" marker; estimate.ts answers that from the cursor instead, which is the question the UI actually asks.',
	'details.frames[].source': 'which planner wrote the frame entry.',
	'details.frames[].present_source': 'which stage decided the frame was present.'
};

// ---------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------

const isObject = (value: unknown): value is Record<string, unknown> =>
	typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * Every key the producer wrote that the parsed object does not carry.
 *
 * Arrays are compared on their first element: the bundle's lists are
 * homogeneous, and one element is enough to see a field.
 */
function unreadKeys(rawValue: unknown, parsed: unknown, path: string): string[] {
	if (DATA_KEYED.includes(path)) return [];
	if (Array.isArray(rawValue)) {
		if (!Array.isArray(parsed) || rawValue.length === 0 || parsed.length === 0) return [];
		return unreadKeys(rawValue[0], parsed[0], `${path}[]`);
	}
	if (!isObject(rawValue) || !isObject(parsed)) return [];
	const out: string[] = [];
	for (const [key, value] of Object.entries(rawValue)) {
		const here = `${path}.${key}`;
		if (!(key in parsed)) {
			out.push(here);
			continue;
		}
		out.push(...unreadKeys(value, parsed[key], here));
	}
	return out;
}

const describeGap = (paths: readonly string[]) =>
	paths.map((path) => `${path} — not parsed and not on the ignore-list`).join('\n');

describe('every field the builder writes is read or explicitly ignored', () => {
	it('manifest', () => {
		const parsed = parseManifest(raw.manifest);
		const unread = unreadKeys(raw.manifest, parsed, 'manifest');
		expect(describeGap(unread.filter((path) => !(path in NOT_READ)))).toBe('');
	});

	it('index rows', () => {
		const parsed = parseIndex(raw.index);
		const unread = unreadKeys(raw.index, parsed, 'index');
		expect(describeGap(unread.filter((path) => !(path in NOT_READ)))).toBe('');
	});

	it('vocabulary', () => {
		const parsed = parseVocabulary(raw.vocabulary);
		const unread = unreadKeys(raw.vocabulary, parsed, 'vocabulary');
		expect(describeGap(unread.filter((path) => !(path in NOT_READ)))).toBe('');
	});

	it('every detail document', () => {
		for (const document of Object.values(raw.details)) {
			const parsed = parseEvent(document);
			const unread = unreadKeys(document, parsed, 'details');
			expect(describeGap(unread.filter((path) => !(path in NOT_READ)))).toBe('');
		}
	});

	it('has no stale entries on the ignore-list', () => {
		// An entry that no longer matches anything the producer writes is a
		// decision about a field that has gone away, and leaving it there
		// makes the list stop meaning what it says.
		const seen = new Set<string>();
		const collect = (rawValue: unknown, parsed: unknown, path: string) => {
			for (const key of unreadKeys(rawValue, parsed, path)) seen.add(key);
		};
		collect(raw.manifest, parseManifest(raw.manifest), 'manifest');
		collect(raw.index, parseIndex(raw.index), 'index');
		collect(raw.vocabulary, parseVocabulary(raw.vocabulary), 'vocabulary');
		for (const document of Object.values(raw.details)) {
			collect(document, parseEvent(document), 'details');
		}
		const stale = Object.keys(NOT_READ).filter((path) => !seen.has(path));
		expect(stale.join('\n')).toBe('');
	});
});

// ---------------------------------------------------------------------------
// The annotation server
// ---------------------------------------------------------------------------

const calls: string[] = [];

function stubFetch(body: unknown, status = 200) {
	vi.stubGlobal('fetch', async (input: string | URL | Request) => {
		calls.push(typeof input === 'string' ? input : String(input));
		return new Response(JSON.stringify(body), {
			status,
			headers: { 'content-type': 'application/json' }
		});
	});
}

afterEach(() => {
	calls.length = 0;
	vi.unstubAllGlobals();
});

describe('a real review_server response', () => {
	it('parses a stored annotation through the client the UI uses', async () => {
		stubFetch(raw.annotations);
		const result = await fetchAnnotations();
		expect(calls[0]).toBe('/review-api/annotations');
		expect(result.ok).toBe(true);
		if (!result.ok) return;

		const recorded = (raw.annotations.annotations as Record<string, unknown>[])[0];
		const parsed = result.value[0];
		expect(result.value).toHaveLength(1);
		expect(parsed.event_id).toBe(recorded.event_id);
		// The identity columns are the whole reason the export is readable
		// without the bundle beside it.
		expect(parsed.station_id).toBe(recorded.station_id);
		expect(parsed.anchor_utc).toBe(recorded.anchor_utc);
		expect(parsed.event_class).toBe(recorded.event_class);
		expect(parsed.dual_truth).toBe(recorded.dual_truth);
		expect(parsed.season).toBe(recorded.season);
		expect(parsed.region).toBe(recorded.region);
		expect(parsed.reviewer).toBe(recorded.reviewer);
		// And the judgement itself, including the revision the next save
		// sends back as If-Match.
		expect(parsed.verdict).toBe(recorded.verdict);
		expect(parsed.tags).toEqual(recorded.tags);
		expect(parsed.confidence).toBe(recorded.confidence);
		expect(parsed.needs_second_look).toBe(recorded.needs_second_look);
		expect(parsed.note).toBe(recorded.note);
		expect(parsed.cursor_utc).toBe(recorded.cursor_utc);
		expect(parsed.review_seq).toBe(recorded.review_seq);
		expect(parsed.revision).toBe(recorded.revision);
	});

	it('reads every field of that row, or ignores it on purpose', async () => {
		stubFetch(raw.annotations);
		const result = await fetchAnnotations();
		if (!result.ok) throw new Error('the recorded response did not parse');
		const recorded = (raw.annotations.annotations as Record<string, unknown>[])[0];
		const unread = unreadKeys(recorded, result.value[0], 'annotation');
		expect(describeGap(unread.filter((path) => !(path in NOT_READ)))).toBe('');
	});

	it('parses a real health response', async () => {
		stubFetch(raw.health);
		const result = await health();
		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.value.bundleId).toBe(raw.health.bundle_id);
		expect(result.value.events).toBe(raw.health.events);
		expect(result.value.stored).toBe(raw.health.stored);
		expect(result.value.annotated).toBe(raw.health.annotated);
		expect(result.value.schemaVersion).toBe(raw.health.schema_version);
		expect(result.value.vocabVersion).toBe(raw.health.vocab_version);
		expect(result.value.dbPath).toBe(raw.health.db_path);
		expect(result.value.bundleRoot).toBe(raw.health.bundle_root);
	});

	it('serves the same bundle the snapshot was built from', async () => {
		// A health response naming a different bundle than the manifest would
		// mean the annotations on screen belong to another draw entirely.
		stubFetch(raw.health);
		const result = await health();
		const manifest = parseManifest(raw.manifest)!;
		expect(result.ok && result.value.bundleId).toBe(manifest.bundle_id);
		expect(raw.annotations.bundle_id).toBe(manifest.bundle_id);
	});
});
