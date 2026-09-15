/**
 * The list model, over the whole fixture index.
 *
 * Two behaviours here are review methodology rather than UI convenience and
 * are pinned as such: facet counts are computed with the facet's own filter
 * lifted (so a reviewer can still see what else is in the bundle), and the
 * control group stays blind — a class filter must not empty it out of the
 * list and announce which rows it held, because the base rate it provides
 * is the only thing that makes a tag tally interpretable.
 */
import { describe, expect, it } from 'vitest';
import {
	applyFilters,
	displayClass,
	DUAL_TRUTH_UNKNOWN,
	facetCounts,
	FilterState,
	nextUnreviewed,
	reviewProgress,
	sortRows
} from './filter';
import fixture from './fixture.json';
import { annotationsByEvent, parseIndex } from './load';
import type { Annotation, IndexRow } from './schema';

const rows = parseIndex(JSON.parse(JSON.stringify(fixture.index)))!.events;

const row = (eventId: string): IndexRow => rows.find((r) => r.event_id === eventId)!;

const annotation = (eventId: string, extra: Partial<Annotation> = {}): Annotation => ({
	bundle_id: fixture.index.bundle_id,
	event_id: eventId,
	// The identity columns the server denormalises onto every row, copied
	// from the index so a fixture row is the shape a real one would be.
	station_id: rows.find((r) => r.event_id === eventId)?.station_id ?? '',
	anchor_utc: rows.find((r) => r.event_id === eventId)?.anchor_utc ?? '',
	event_class: rows.find((r) => r.event_id === eventId)?.class ?? 'false_alarm',
	dual_truth: rows.find((r) => r.event_id === eventId)?.dual_truth ?? null,
	season: rows.find((r) => r.event_id === eventId)?.season ?? '',
	region: rows.find((r) => r.event_id === eventId)?.region ?? '',
	reviewer: 'nsimonsen',
	verdict: null,
	tags: [],
	vocab_version: 1,
	confidence: null,
	needs_second_look: false,
	note: '',
	cursor_utc: null,
	review_seq: 1,
	created_utc: '2026-09-15T10:00:00Z',
	updated_utc: '2026-09-15T10:00:00Z',
	revision: 1,
	...extra
});

const ids = (result: readonly IndexRow[]) => result.map((r) => r.event_id);
const controlRow = rows.find((r) => r.control)!;
const missRow = rows.find((r) => r.class === 'miss' && !r.control)!;

describe('applyFilters', () => {
	it('returns everything when nothing is selected', () => {
		const filter = new FilterState();
		expect(filter.active).toBe(false);
		expect(applyFilters(rows, filter)).toHaveLength(rows.length);
	});

	it('filters by class, and keeps the blinded control group interleaved', () => {
		const filter = FilterState.from({ classes: [missRow.class] });
		const kept = applyFilters(rows, filter);
		expect(ids(kept)).toContain(missRow.event_id);
		// The control row is a different class and stays anyway: its absence
		// would be the reveal the toggle exists to prevent.
		expect(ids(kept)).toContain(controlRow.event_id);
	});

	it('honours the class filter on the control group once it is revealed', () => {
		const filter = FilterState.from({ classes: [missRow.class], revealControls: true });
		const kept = applyFilters(rows, filter);
		expect(ids(kept)).not.toContain(controlRow.event_id);
		expect(kept.every((r) => r.class === missRow.class)).toBe(true);
	});

	it('treats a missing dual-truth verdict as its own facet value', () => {
		const unjudgeable = rows.filter((r) => r.dual_truth === null);
		expect(unjudgeable.length).toBeGreaterThan(0);
		const kept = applyFilters(rows, FilterState.from({ dualTruth: [DUAL_TRUTH_UNKNOWN] }));
		expect(ids(kept)).toEqual(unjudgeable.map((r) => r.event_id));

		// And it is never swept into a quadrant.
		const bothDry = applyFilters(rows, FilterState.from({ dualTruth: ['both_dry'] }));
		for (const id of unjudgeable.map((r) => r.event_id)) {
			expect(ids(bothDry)).not.toContain(id);
		}
	});

	it('filters by season, region and flag', () => {
		const flagged = rows.filter((r) => r.flags.includes('coverage_gap'));
		expect(flagged.length).toBeGreaterThan(0);
		expect(ids(applyFilters(rows, FilterState.from({ flags: ['coverage_gap'] })))).toEqual(
			flagged.map((r) => r.event_id)
		);

		const season = missRow.season;
		const inSeason = rows.filter((r) => r.season === season);
		expect(ids(applyFilters(rows, FilterState.from({ seasons: [season] })))).toEqual(
			inSeason.map((r) => r.event_id)
		);

		const region = missRow.region;
		expect(
			applyFilters(rows, FilterState.from({ regions: [region] })).every(
				(r) => r.region === region
			)
		).toBe(true);
	});

	it('searches the identifying columns, the flags and the note', () => {
		expect(
			ids(applyFilters(rows, FilterState.from({ text: missRow.station_name.slice(0, 5) })))
		).toContain(missRow.event_id);
		expect(ids(applyFilters(rows, FilterState.from({ text: missRow.station_id })))).toEqual([
			missRow.event_id
		]);

		const annotations = annotationsByEvent([
			annotation(controlRow.event_id, { note: 'virga over a dry gauge', tags: ['fa_virga_or_aloft'] })
		]);
		expect(
			ids(applyFilters(rows, FilterState.from({ text: 'virga' }), annotations))
		).toEqual([controlRow.event_id]);
	});

	it('intersects the dimensions', () => {
		const filter = FilterState.from({
			classes: [missRow.class],
			seasons: ['a season no event has'],
			revealControls: true
		});
		expect(applyFilters(rows, filter)).toHaveLength(0);
	});

	it('filters on the judgement: tagged, untagged, and by verdict', () => {
		const annotations = annotationsByEvent([
			annotation(missRow.event_id, { verdict: 'metric_artefact' }),
			annotation(controlRow.event_id, { verdict: 'real_failure' })
		]);
		expect(ids(applyFilters(rows, FilterState.from({ verdict: 'tagged' }), annotations)).sort()).toEqual(
			[missRow.event_id, controlRow.event_id].sort()
		);
		expect(
			applyFilters(rows, FilterState.from({ verdict: 'untagged' }), annotations)
		).toHaveLength(rows.length - 2);
		expect(
			ids(applyFilters(rows, FilterState.from({ verdict: 'metric_artefact' }), annotations))
		).toEqual([missRow.event_id]);
	});

	it('treats a stored row with no verdict as untagged', () => {
		const annotations = annotationsByEvent([annotation(missRow.event_id, { note: 'looked at it' })]);
		expect(
			ids(applyFilters(rows, FilterState.from({ verdict: 'untagged' }), annotations))
		).toContain(missRow.event_id);
	});
});

describe('FilterState', () => {
	it('toggles a facet value on and off', () => {
		const filter = new FilterState();
		filter.toggle('class', 'miss');
		expect(filter.classes).toEqual(['miss']);
		filter.toggle('class', 'miss');
		expect(filter.classes).toEqual([]);
	});

	it('clears the facets but keeps how the reviewer is working', () => {
		const filter = FilterState.from({
			classes: ['miss'],
			text: 'x',
			sort: 'anchor',
			revealControls: true
		});
		filter.clear();
		expect(filter.active).toBe(false);
		expect(filter.sort).toBe('anchor');
		expect(filter.revealControls).toBe(true);
	});

	it('clones without sharing its arrays', () => {
		const filter = FilterState.from({ classes: ['miss'] });
		const copy = filter.clone();
		copy.toggle('class', 'hit');
		expect(filter.classes).toEqual(['miss']);
		expect(copy.classes).toEqual(['miss', 'hit']);
	});
});

describe('sortRows', () => {
	it('leaves the bundle’s own seeded order alone', () => {
		expect(ids(sortRows(rows, 'bundle'))).toEqual(ids(rows));
		expect(ids(sortRows(rows, 'bundle', true))).toEqual([...ids(rows)].reverse());
	});

	it('sorts by anchor in both directions', () => {
		const ascending = sortRows(rows, 'anchor').map((r) => Date.parse(r.anchor_utc));
		expect(ascending).toEqual([...ascending].sort((a, b) => a - b));
		const descending = sortRows(rows, 'anchor', true).map((r) => Date.parse(r.anchor_utc));
		expect(descending).toEqual([...ascending].reverse());
	});

	it('keeps rows with nothing to sort on at the bottom, both ways', () => {
		const withNulls: IndexRow[] = [
			{ ...row(rows[0].event_id), event_id: 'a', lead_error_min: null },
			{ ...row(rows[0].event_id), event_id: 'b', lead_error_min: 4 },
			{ ...row(rows[0].event_id), event_id: 'c', lead_error_min: -9 }
		];
		expect(ids(sortRows(withNulls, 'lead_error'))).toEqual(['c', 'b', 'a']);
		expect(ids(sortRows(withNulls, 'lead_error', true))).toEqual(['b', 'c', 'a']);
	});

	it('breaks ties on the event id, so the order is total', () => {
		const tied: IndexRow[] = [
			{ ...row(rows[0].event_id), event_id: 'z', p_decision: 0.5 },
			{ ...row(rows[0].event_id), event_id: 'a', p_decision: 0.5 }
		];
		expect(ids(sortRows(tied, 'probability'))).toEqual(['a', 'z']);
	});

	it('sorts by the reviewer’s own order when there is one', () => {
		const annotations = annotationsByEvent([
			annotation(controlRow.event_id, { review_seq: 2, verdict: 'unclear' }),
			annotation(missRow.event_id, { review_seq: 1, verdict: 'unclear' })
		]);
		const sorted = sortRows(rows, 'review_seq', false, annotations);
		expect(ids(sorted).slice(0, 2)).toEqual([missRow.event_id, controlRow.event_id]);
	});
});

describe('facetCounts', () => {
	it('counts every facet of an unfiltered bundle', () => {
		const counts = facetCounts(rows, new FilterState());
		const total = (facet: { count: number }[]) => facet.reduce((n, f) => n + f.count, 0);
		// Classes are counted over the non-control rows while blinded.
		expect(total(counts.class)).toBe(rows.length - 1);
		expect(counts.control).toBe(1);
		expect(total(counts.dual_truth)).toBe(rows.length);
		expect(counts.verdict.find((f) => f.value === 'untagged')!.count).toBe(rows.length);
	});

	it('counts a facet with its OWN filter lifted', () => {
		const filter = FilterState.from({ classes: ['false_alarm'], revealControls: true });
		const counts = facetCounts(rows, filter);
		const classCounts = Object.fromEntries(counts.class.map((f) => [f.value, f.count]));
		// Selecting false_alarm must not zero the other classes: the reviewer
		// still has to see what else is in the bundle.
		expect(classCounts.false_alarm).toBe(rows.filter((r) => r.class === 'false_alarm').length);
		expect(classCounts.miss).toBe(1);
		expect(counts.class.find((f) => f.value === 'false_alarm')!.selected).toBe(true);
	});

	it('narrows the other facets to the current selection', () => {
		const filter = FilterState.from({ classes: ['miss'], revealControls: true });
		const counts = facetCounts(rows, filter);
		expect(counts.season.map((f) => f.value)).toEqual([missRow.season]);
		expect(counts.region.map((f) => f.value)).toEqual([missRow.region]);
	});

	it('counts the control group as one anonymous bucket while blinded', () => {
		const blind = facetCounts(rows, new FilterState());
		expect(blind.class.map((f) => f.value)).not.toContain(controlRow.class);
		const revealed = facetCounts(rows, FilterState.from({ revealControls: true }));
		expect(revealed.class.map((f) => f.value)).toContain(controlRow.class);
	});

	it('offers "unknown" for the events with no dual-truth verdict', () => {
		const counts = facetCounts(rows, new FilterState());
		const unknown = counts.dual_truth.find((f) => f.value === DUAL_TRUTH_UNKNOWN)!;
		expect(unknown.count).toBe(rows.filter((r) => r.dual_truth === null).length);
	});

	it('orders by count and then alphabetically, so nothing jitters', () => {
		const counts = facetCounts(rows, FilterState.from({ revealControls: true })).class;
		for (let i = 1; i < counts.length; i++) {
			const previous = counts[i - 1];
			const current = counts[i];
			expect(
				previous.count > current.count ||
					(previous.count === current.count && previous.value < current.value)
			).toBe(true);
		}
	});
});

describe('reviewProgress', () => {
	it('is empty before anything is judged', () => {
		const progress = reviewProgress(rows);
		expect(progress.total).toBe(rows.length);
		expect(progress.withVerdict).toBe(0);
		expect(progress.remaining).toBe(rows.length);
		expect(progress.fraction).toBe(0);
	});

	it('separates stored, judged and finished', () => {
		const annotations = annotationsByEvent([
			// Stored, no verdict: the reviewer opened it and typed a note.
			annotation(rows[0].event_id, { note: 'come back to this' }),
			// Judged but not finished: a verdict with no mechanism behind it.
			annotation(rows[1].event_id, { verdict: 'real_failure' }),
			// Finished.
			annotation(rows[2].event_id, { verdict: 'metric_artefact', tags: ['fa_virga_or_aloft'] })
		]);
		const progress = reviewProgress(rows, annotations);
		expect(progress.stored).toBe(3);
		expect(progress.withVerdict).toBe(2);
		expect(progress.complete).toBe(1);
		expect(progress.byVerdict).toEqual({ real_failure: 1, metric_artefact: 1 });
		expect(progress.byClass[rows[1].class].withVerdict).toBeGreaterThan(0);
	});

	it('reports on whatever rows it is given, filtered or not', () => {
		expect(reviewProgress([]).fraction).toBeNull();
		expect(reviewProgress(rows.slice(0, 2)).total).toBe(2);
	});
});

describe('nextUnreviewed', () => {
	it('starts after the current event and wraps once', () => {
		const annotations = annotationsByEvent([
			annotation(rows[1].event_id, { verdict: 'unclear' })
		]);
		expect(nextUnreviewed(rows, annotations, rows[0].event_id)!.event_id).toBe(rows[2].event_id);
		expect(nextUnreviewed(rows, annotations, rows[rows.length - 1].event_id)!.event_id).toBe(
			rows[0].event_id
		);
	});

	it('is null once everything visible is judged', () => {
		const annotations = annotationsByEvent(
			rows.map((r) => annotation(r.event_id, { verdict: 'unclear' }))
		);
		expect(nextUnreviewed(rows, annotations, null)).toBeNull();
	});
});

describe('displayClass', () => {
	it('withholds the control group’s class until it is revealed', () => {
		expect(displayClass(controlRow, false)).toBeNull();
		expect(displayClass(controlRow, true)).toBe(controlRow.class);
		expect(displayClass(missRow, false)).toBe(missRow.class);
	});
});
