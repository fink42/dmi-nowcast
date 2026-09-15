/**
 * The bundle loader. The property under test throughout is the one the
 * review depends on: a document that is wrong somewhere still opens
 * everywhere else, a section we cannot read comes back as null, and nothing
 * ever throws into the UI — because the alternative is a reviewer judging a
 * gauge series that failed to parse and reading its absence as "dry".
 *
 * Expectations are derived from the fixture's own contents wherever they
 * can be, because `fixture.json` will later be regenerated from the Python
 * builder's `--fixture` mode and a test that hardcodes its numbers would
 * fail on a fixture that is still perfectly correct.
 */
import { describe, expect, it, vi } from 'vitest';
import fixture from './fixture.json';
import {
	annotationsByEvent,
	bundleUrl,
	parseEvent,
	parseIndex,
	parseManifest,
	parseVocabulary
} from './load';
import { REVIEW_SCHEMA_VERSION } from './schema';

/** A fresh deep copy, so a test that mutates cannot reach the next one. */
const doc = (value: unknown): Record<string, any> => JSON.parse(JSON.stringify(value));

const manifestDoc = () => doc(fixture.manifest);
const indexDoc = () => doc(fixture.index);
const details = fixture.details as Record<string, unknown>;
const detailDoc = (eventId: string) => doc(details[eventId]);
const eventIds = Object.keys(details);

describe('bundleUrl', () => {
	it('joins a bundle path onto the proxied prefix', () => {
		expect(bundleUrl('events.json')).toBe('/review-data/events.json');
		expect(bundleUrl('events/fa-1.json')).toBe('/review-data/events/fa-1.json');
	});

	it('refuses to walk out of the bundle', () => {
		// The server answers 403 for these; the client does not ask.
		expect(bundleUrl('../../etc/passwd')).toBe('/review-data/etc/passwd');
		expect(bundleUrl('/events.json')).toBe('/review-data/events.json');
		expect(bundleUrl('./frames/./x.png')).toBe('/review-data/frames/x.png');
	});
});

describe('parseManifest', () => {
	const manifest = parseManifest(manifestDoc());

	it('parses every block of a well-formed manifest', () => {
		expect(manifest).not.toBeNull();
		expect(manifest!.bundle_id).toBe(fixture.manifest.bundle_id);
		expect(manifest!.schema_version).toBe(REVIEW_SCHEMA_VERSION);
		for (const block of ['builder', 'corpus', 'window', 'rule', 'truth', 'sampling', 'grid', 'frames', 'events'] as const) {
			expect(manifest![block], block).not.toBeNull();
		}
		expect(manifest!.caveats!.length).toBe(fixture.manifest.caveats.length);
		expect(Object.keys(manifest!.feature_doc!)).toEqual(
			Object.keys(fixture.manifest.feature_doc)
		);
	});

	it('keeps the grid exactly as written — it places the rain', () => {
		expect(manifest!.grid).toEqual(fixture.manifest.grid);
	});

	it('keeps the observation quantisation, which nothing may guess', () => {
		const observed = manifest!.frames!.encodings.observed;
		expect(observed.scale).toBe(fixture.manifest.frames.encodings.observed.scale);
		expect(observed.offset).toBe(fixture.manifest.frames.encodings.observed.offset);
		expect(observed.nodata).toBe(fixture.manifest.frames.encodings.observed.nodata);
	});

	it('keeps the two window figures apart', () => {
		const frames = manifest!.frames!;
		// The decision half-window and the extra imagery before it are
		// different numbers; adding them anywhere would draw the track's left
		// edge in the wrong place.
		expect(frames.window_min).toBe(fixture.manifest.frames.window_min);
		expect(frames.frame_pad_min).toBe(fixture.manifest.frames.frame_pad_min);
		expect(frames.products).toEqual(fixture.manifest.frames.products);
	});

	it('drops the frames block when the quantisation is unreadable', () => {
		for (const field of ['scale', 'offset', 'nodata']) {
			const raw = manifestDoc();
			delete raw.frames.encodings.observed[field];
			const parsed = parseManifest(raw);
			// Null, never a default: a guessed scale puts a plausible, wrong
			// mm/h under the cursor, which no reviewer could catch by eye.
			expect(parsed!.frames, field).toBeNull();
			expect(parsed!.grid).not.toBeNull();
		}
	});

	it('keeps the frames block when only the colour legend is unreadable', () => {
		const raw = manifestDoc();
		raw.frames.encodings.overlay = 'gone';
		const frames = parseManifest(raw)!.frames!;
		// The overlay metadata is a legend, not a number.
		expect(frames.encodings.observed.scale).toBe(
			fixture.manifest.frames.encodings.observed.scale
		);
		expect(frames.encodings.overlay.colormap_stops).toEqual([]);
	});

	it('parses a --no-frames bundle, which has no grid and no imagery', () => {
		const raw = manifestDoc();
		raw.grid = null;
		raw.frames = null;
		const parsed = parseManifest(raw)!;
		expect(parsed.grid).toBeNull();
		expect(parsed.frames).toBeNull();
		// The review still opens: every number is still there.
		expect(parsed.rule).not.toBeNull();
		expect(parsed.truth).not.toBeNull();
	});

	it('nulls a section that does not parse without losing the rest', () => {
		const raw = manifestDoc();
		raw.rule = null;
		raw.sampling = 'not an object';
		raw.truth = { onset_rule: {} };
		const parsed = parseManifest(raw);
		expect(parsed!.rule).toBeNull();
		expect(parsed!.sampling).toBeNull();
		expect(parsed!.truth).toBeNull();
		expect(parsed!.grid).not.toBeNull();
		expect(parsed!.bundle_id).toBe(fixture.manifest.bundle_id);
	});

	it('warns about a schema it was not written against, and parses on', () => {
		const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
		const parsed = parseManifest(doc(fixture.broken.future_manifest));
		expect(warn).toHaveBeenCalled();
		// A producer that added a field must not blank the page.
		expect(parsed).not.toBeNull();
		expect(parsed!.grid).not.toBeNull();
		warn.mockRestore();
	});

	it('returns null for a truncated or renamed document, and never throws', () => {
		expect(parseManifest(doc(fixture.broken.truncated_manifest))).toBeNull();
		expect(parseManifest(doc(fixture.broken.renamed_manifest))).toBeNull();
	});
});

describe('parseIndex', () => {
	const index = parseIndex(indexDoc());

	it('parses every row the fixture carries', () => {
		expect(index).not.toBeNull();
		expect(index!.events).toHaveLength(fixture.index.events.length);
		expect(index!.events.map((row) => row.event_id)).toEqual(
			fixture.index.events.map((row) => row.event_id)
		);
	});

	it('keeps a null dual-truth verdict null', () => {
		const withoutVerdict = fixture.index.events.filter((row) => row.dual_truth === null);
		expect(withoutVerdict.length, 'the fixture needs one unjudgeable event').toBeGreaterThan(0);
		for (const row of withoutVerdict) {
			const parsed = index!.events.find((candidate) => candidate.event_id === row.event_id)!;
			// `both_dry` is the claim "the forecast invented rain". Falling
			// through to it here would be the tool fabricating its own worst
			// finding.
			expect(parsed.dual_truth).toBeNull();
			expect(parsed.gauge_wet_in_window).toBeNull();
		}
	});

	it('drops a row it cannot identify and keeps the others', () => {
		const raw = indexDoc();
		raw.events[1].class = 'something_new';
		delete raw.events[2].event_id;
		const parsed = parseIndex(raw);
		expect(parsed!.events).toHaveLength(fixture.index.events.length - 2);
	});

	it('returns null for a truncated document', () => {
		expect(parseIndex(doc(fixture.broken.truncated_index))).toBeNull();
		expect(parseIndex('not a document')).toBeNull();
	});
});

describe('parseEvent', () => {
	it('parses every detail document in the fixture', () => {
		for (const eventId of eventIds) {
			const event = parseEvent(detailDoc(eventId));
			expect(event, eventId).not.toBeNull();
			expect(event!.event_id).toBe(eventId);
			expect(event!.decisions.length).toBe(
				(details[eventId] as any).decisions.length
			);
			expect(event!.frames.length).toBe((details[eventId] as any).frames.length);
		}
	});

	it('orders decisions and frames by time whatever order they arrived in', () => {
		const raw = detailDoc(eventIds[0]);
		raw.decisions.reverse();
		raw.frames.reverse();
		const event = parseEvent(raw)!;
		const decisionTimes = event.decisions.map((d) => Date.parse(d.generated_at_utc));
		const frameTimes = event.frames.map((f) => Date.parse(f.radar_ts_utc));
		expect(decisionTimes).toEqual([...decisionTimes].sort((a, b) => a - b));
		expect(frameTimes).toEqual([...frameTimes].sort((a, b) => a - b));
	});

	it('reads both ISO spellings — Z and +00:00', () => {
		const event = parseEvent(detailDoc(eventIds[0]))!;
		for (const decision of event.decisions) {
			expect(Number.isFinite(Date.parse(decision.radar_ts_utc))).toBe(true);
		}
	});

	it('never lets an unreported slot be wet', () => {
		const raw = detailDoc(eventIds[0]);
		raw.gauge.slots[0] = { slot_end_utc: raw.gauge.slots[0].slot_end_utc, known: false, wet: true, mm: 3 };
		const event = parseEvent(raw)!;
		expect(event.gauge!.slots[0].known).toBe(false);
		expect(event.gauge!.slots[0].wet).toBe(false);
	});

	it('defaults an unreadable slot to unknown rather than to dry', () => {
		const raw = detailDoc(eventIds[0]);
		delete raw.gauge.slots[1].known;
		delete raw.gauge.slots[1].wet;
		const event = parseEvent(raw)!;
		expect(event.gauge!.slots[1].known).toBe(false);
		expect(event.gauge!.slots[1].wet).toBe(false);
	});

	it('drops a decision that cannot be dated and keeps the rest', () => {
		const raw = detailDoc(eventIds[0]);
		const total = raw.decisions.length;
		delete raw.decisions[3].generated_at_utc;
		raw.decisions[4].radar_ts_utc = 'not a time';
		const event = parseEvent(raw)!;
		expect(event.decisions).toHaveLength(total - 2);
	});

	it('nulls the truth sections it cannot read without losing the decisions', () => {
		const raw = detailDoc(eventIds[0]);
		raw.gauge = { slots: 'nonsense' };
		raw.radar_disc = null;
		raw.neighbours = {};
		raw.prologue = 'gone';
		const event = parseEvent(raw)!;
		expect(event.gauge).toBeNull();
		expect(event.radar_disc).toBeNull();
		expect(event.neighbours).toBeNull();
		expect(event.prologue).toBeNull();
		expect(event.decisions.length).toBeGreaterThan(0);
	});

	it('keeps a dual-truth block whose verdict is "not known"', () => {
		const raw = detailDoc(eventIds[0]);
		raw.dual_truth = { ...raw.dual_truth, class: null, gauge_wet: null };
		const event = parseEvent(raw)!;
		expect(event.dual_truth).not.toBeNull();
		expect(event.dual_truth!.class).toBeNull();
		expect(event.dual_truth!.gauge_wet).toBeNull();
	});

	it('carries the prologue’s own re-arm constant and run-boundary instants', () => {
		const withRearms = eventIds
			.map((id) => parseEvent(detailDoc(id))!)
			.filter((event) => (event.prologue?.run_boundary_rearms_utc.length ?? 0) > 0);
		expect(withRearms.length, 'the fixture needs one replay re-arm').toBeGreaterThan(0);
		for (const event of withRearms) {
			expect(event.prologue!.rearm_after_min).toBeGreaterThan(0);
			for (const at of event.prologue!.run_boundary_rearms_utc) {
				expect(Number.isFinite(Date.parse(at))).toBe(true);
			}
		}
	});

	it('returns null for a truncated or renamed document, and never throws', () => {
		expect(parseEvent(doc(fixture.broken.truncated_event))).toBeNull();
		expect(parseEvent(doc(fixture.broken.renamed_event))).toBeNull();
	});
});

describe('parseVocabulary', () => {
	it('parses the bundle’s own vocabulary', () => {
		const vocabulary = parseVocabulary(doc(fixture.vocabulary))!;
		expect(vocabulary.vocab_version).toBe(fixture.vocabulary.vocab_version);
		expect(vocabulary.verdicts).toHaveLength(fixture.vocabulary.verdicts.length);
		expect(vocabulary.tag_groups.map((group) => group.group)).toEqual(
			fixture.vocabulary.tag_groups.map((group) => group.group)
		);
	});

	it('returns null when there is no usable vocabulary, so the fallback wins', () => {
		expect(parseVocabulary({ vocab_version: 1 })).toBeNull();
		expect(parseVocabulary({ vocab_version: 1, verdicts: [], tag_groups: [] })).toBeNull();
		expect(parseVocabulary(null)).toBeNull();
	});
});

describe('the parsers never throw', () => {
	const junk = [
		null,
		undefined,
		0,
		'',
		'a string',
		[],
		[1, 2, 3],
		{},
		{ events: 4 },
		{ schema_version: {} },
		{ index: [], window: 7 },
		NaN
	];

	it('answers null for anything unreadable', () => {
		for (const value of junk) {
			expect(() => parseManifest(value)).not.toThrow();
			expect(() => parseIndex(value)).not.toThrow();
			expect(() => parseEvent(value)).not.toThrow();
			expect(() => parseVocabulary(value)).not.toThrow();
		}
	});
});

describe('annotationsByEvent', () => {
	it('keys judgements by the event they belong to', () => {
		const map = annotationsByEvent([
			{ event_id: 'a', verdict: 'real_failure' },
			{ event_id: 'b', verdict: null }
		]);
		expect(map.get('a')!.verdict).toBe('real_failure');
		expect(map.get('c')).toBeUndefined();
	});
});
