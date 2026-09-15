/**
 * The vocabulary and what makes one judgement valid.
 *
 * The load-bearing test is the first one: the codes compiled into this
 * client must be exactly the codes the bundle ships. They are the keys of
 * the whole exercise — a tag tally is the output — and a client offering a
 * code the server will reject, or missing one the bundle defines, produces
 * a review that cannot be aggregated. The fixture's vocabulary is a verbatim
 * dump of `review_schema.tags_document()`, so this is a test against the
 * Python.
 */
import { describe, expect, it } from 'vitest';
import fixture from './fixture.json';
import { parseVocabulary } from './load';
import type { Annotation, AnnotationDraft, Vocabulary } from './schema';
import {
	allTags,
	BUILTIN_VOCABULARY,
	isComplete,
	isDirty,
	isMechanismTag,
	NOTE_REQUIRED_TAGS,
	tagsForClass,
	validateAnnotation
} from './tags';

const bundleVocabulary = parseVocabulary(JSON.parse(JSON.stringify(fixture.vocabulary)))!;

const codes = (tags: { code: string }[]) => tags.map((tag) => tag.code).sort();

describe('the built-in vocabulary', () => {
	it('carries exactly the bundle’s codes, group for group', () => {
		expect(BUILTIN_VOCABULARY.vocab_version).toBe(bundleVocabulary.vocab_version);
		expect(codes(BUILTIN_VOCABULARY.verdicts)).toEqual(codes(bundleVocabulary.verdicts));
		expect(BUILTIN_VOCABULARY.tag_groups.map((group) => group.group)).toEqual(
			bundleVocabulary.tag_groups.map((group) => group.group)
		);
		for (const group of BUILTIN_VOCABULARY.tag_groups) {
			const theirs = bundleVocabulary.tag_groups.find((g) => g.group === group.group)!;
			expect(codes(group.tags), group.group).toEqual(codes(theirs.tags));
		}
	});

	it('agrees about which codes are not mechanisms', () => {
		expect([...BUILTIN_VOCABULARY.non_mechanism].sort()).toEqual(
			[...bundleVocabulary.non_mechanism].sort()
		);
	});

	it('agrees about which codes each outcome class offers', () => {
		for (const [eventClass, allowed] of Object.entries(bundleVocabulary.classes)) {
			expect([...(BUILTIN_VOCABULARY.classes[eventClass] ?? [])].sort(), eventClass).toEqual(
				[...allowed].sort()
			);
		}
	});

	it('describes every code it offers', () => {
		for (const tag of allTags(BUILTIN_VOCABULARY)) {
			expect(tag.description.length, tag.code).toBeGreaterThan(10);
		}
	});
});

describe('tagsForClass', () => {
	it('offers the false-alarm causes to a false alarm', () => {
		const groups = tagsForClass(bundleVocabulary, 'false_alarm');
		expect(groups.map((group) => group.group)).toEqual(['false_alarm', 'common']);
	});

	it('offers the miss causes to a miss and to an uncovered control', () => {
		for (const eventClass of ['miss', 'miss_late', 'uncovered'] as const) {
			expect(tagsForClass(bundleVocabulary, eventClass).map((g) => g.group)).toEqual([
				'miss',
				'common'
			]);
		}
	});

	it('offers BOTH cause lists to a hit — that is what makes it a base rate', () => {
		// If `fa_cell_died` describes 40 % of the hits too, it explains nothing
		// about the false alarms, and the only way to find out is to let a
		// reviewer tag it on a hit.
		expect(tagsForClass(bundleVocabulary, 'hit').map((g) => g.group)).toEqual([
			'false_alarm',
			'miss',
			'common'
		]);
	});

	it('falls back to every group for a class the vocabulary does not mention', () => {
		expect(tagsForClass(bundleVocabulary, 'something_new').map((g) => g.group)).toEqual([
			'false_alarm',
			'miss',
			'common'
		]);
	});

	it('knows which codes count as a mechanism', () => {
		expect(isMechanismTag(bundleVocabulary, 'fa_cell_died')).toBe(true);
		expect(isMechanismTag(bundleVocabulary, 'fa_threshold_marginal')).toBe(false);
		expect(isMechanismTag(bundleVocabulary, 'interesting')).toBe(false);
	});
});

const draft = (extra: Partial<AnnotationDraft> = {}): AnnotationDraft => ({
	verdict: null,
	tags: [],
	confidence: null,
	needs_second_look: false,
	note: '',
	cursor_utc: null,
	vocab_version: 1,
	...extra
});

describe('validateAnnotation', () => {
	const problems = (value: Partial<AnnotationDraft>, vocabulary: Vocabulary = bundleVocabulary) =>
		validateAnnotation(value, vocabulary).map((problem) => problem.field);

	it('accepts a well-formed judgement', () => {
		expect(
			problems(draft({ verdict: 'real_failure', tags: ['fa_cell_died'], confidence: 2 }))
		).toEqual([]);
	});

	it('refuses a verdict and a tag the vocabulary does not define', () => {
		expect(problems(draft({ verdict: 'probably' as never }))).toEqual(['verdict']);
		expect(problems(draft({ tags: ['fa_cell_died', 'fa_made_up'] }))).toEqual(['tags']);
	});

	it('accepts a false-alarm tag on any class, exactly as the server does', () => {
		// A reviewer reaching for one on an `uncovered` event is telling us
		// something about the coverage rule, not making a mistake.
		expect(problems(draft({ tags: ['fa_virga_or_aloft'] }))).toEqual([]);
	});

	it('takes confidence as an integer 1..3, or nothing', () => {
		expect(problems(draft({ confidence: null }))).toEqual([]);
		for (const confidence of [1, 2, 3]) {
			expect(problems(draft({ confidence }))).toEqual([]);
		}
		for (const confidence of [0, 4, 1.5, -1]) {
			expect(problems(draft({ confidence })), String(confidence)).toEqual(['confidence']);
		}
	});

	it('takes a cursor stamp only if it is an instant', () => {
		expect(problems(draft({ cursor_utc: '2026-06-12T13:45:00Z' }))).toEqual([]);
		expect(problems(draft({ cursor_utc: 'halfway' }))).toEqual(['cursor_utc']);
	});

	it('names every problem at once, so the form can point at all of them', () => {
		expect(
			problems(draft({ verdict: 'nope' as never, tags: ['nope'], confidence: 7 }))
		).toEqual(['verdict', 'tags', 'confidence']);
	});
});

describe('isComplete', () => {
	it('needs a verdict and at least one tag', () => {
		expect(isComplete(null)).toBe(false);
		expect(isComplete(draft())).toBe(false);
		expect(isComplete(draft({ verdict: 'unclear' }))).toBe(false);
		expect(isComplete(draft({ tags: ['fa_cell_died'] }))).toBe(false);
		expect(isComplete(draft({ verdict: 'unclear', tags: ['fa_cell_died'] }))).toBe(true);
	});

	it('demands the note that "something else" promises', () => {
		for (const code of NOTE_REQUIRED_TAGS) {
			expect(isComplete(draft({ verdict: 'real_failure', tags: [code] })), code).toBe(false);
			expect(
				isComplete(draft({ verdict: 'real_failure', tags: [code], note: 'the beam overshot' })),
				code
			).toBe(true);
		}
	});
});

const saved = (extra: Partial<Annotation> = {}): Annotation => ({
	bundle_id: 'b',
	event_id: 'e',
	station_id: '06126',
	anchor_utc: '2026-06-12T13:45:00Z',
	event_class: 'false_alarm',
	dual_truth: 'radar_wet_gauge_dry',
	season: 'summer',
	region: 'Funen',
	reviewer: 'nsimonsen',
	verdict: 'real_failure',
	tags: ['fa_cell_died', 'interesting'],
	vocab_version: 1,
	confidence: 2,
	needs_second_look: false,
	note: 'decayed on the way in',
	cursor_utc: '2026-06-12T13:45:00Z',
	review_seq: 3,
	created_utc: '2026-09-15T10:00:00Z',
	updated_utc: '2026-09-15T10:05:00Z',
	revision: 2,
	...extra
});

describe('isDirty', () => {
	const asDraft = (row: Annotation): AnnotationDraft => ({
		verdict: row.verdict,
		tags: [...row.tags],
		confidence: row.confidence,
		needs_second_look: row.needs_second_look,
		note: row.note,
		cursor_utc: row.cursor_utc,
		vocab_version: row.vocab_version
	});

	it('is clean when the draft still matches the stored row', () => {
		expect(isDirty(asDraft(saved()), saved())).toBe(false);
	});

	it('ignores the order the tags were toggled in', () => {
		// The server dedupes and the tally counts a set; an unsaved-changes
		// warning that fires on a reorder just trains people to ignore it.
		const reordered = asDraft(saved());
		reordered.tags = ['interesting', 'fa_cell_died'];
		expect(isDirty(reordered, saved())).toBe(false);
	});

	it('notices every field the reviewer can change', () => {
		const changes: Array<Partial<AnnotationDraft>> = [
			{ verdict: 'unclear' },
			{ tags: ['fa_cell_died'] },
			{ confidence: 3 },
			{ needs_second_look: true },
			{ note: 'something else' },
			{ cursor_utc: '2026-06-12T14:00:00Z' }
		];
		for (const change of changes) {
			expect(isDirty({ ...asDraft(saved()), ...change }, saved()), JSON.stringify(change)).toBe(
				true
			);
		}
	});

	it('treats an untouched blank draft on an unjudged event as clean', () => {
		expect(isDirty(draft(), null)).toBe(false);
		expect(isDirty(draft({ verdict: 'unclear' }), null)).toBe(true);
		expect(isDirty(draft({ tags: ['interesting'] }), null)).toBe(true);
		expect(isDirty(draft({ note: '   ' }), null)).toBe(false);
		expect(isDirty(draft({ confidence: 1 }), null)).toBe(true);
	});
});
