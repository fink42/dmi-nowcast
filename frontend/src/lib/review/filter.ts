/**
 * The list model: which of the ~300 events are on screen, in what order,
 * and how much of the review is done.
 *
 * All of it is array-pure over the parsed index, because that index is
 * small enough to filter from scratch on every keystroke (300 rows × a
 * dozen predicates is microseconds) and because a filter that is a pure
 * function of state is one a test can pin. The store holds a `FilterState`
 * and calls these; nothing here reads or writes anything outside its
 * arguments.
 *
 * Two decisions here are about the *review*, not about the UI:
 *
 *  - **Facet counts are computed against the other facets, not against
 *    themselves.** Selecting `miss` must not reduce the class counts to
 *    "miss: 118, everything else: 0", or the reviewer loses the only view
 *    of what else is in the bundle and cannot tell a filter that found
 *    nothing from a bundle that contains nothing.
 *  - **The control group stays blind by default.** ~50 hits and ~20 late
 *    events are drawn with `control: true` and their class hidden, because
 *    a tag tally with no base rate is uninterpretable: if `fa_cell_died`
 *    describes 40 % of the *hits* too, it explains nothing. Filtering by
 *    class would give that away by absence, so while the reveal toggle is
 *    off, control rows ignore the class filter and stay interleaved.
 */
import type { Annotation, DualTruthClass, IndexRow, OutcomeClass, Verdict } from './schema';
import { isComplete } from './tags';

/** `any` ignores annotations entirely; the rest filter on them. */
export type VerdictFilter = 'any' | 'tagged' | 'untagged' | Verdict;

/**
 * The facet value standing for "no dual-truth verdict".
 *
 * A row whose `dual_truth` is null is not in any quadrant: neither truth
 * could speak, so there is nothing to classify. It gets its own selectable
 * bucket rather than being silently absent — "which events could not be
 * judged at all?" is one of the questions this review exists to answer, and
 * a row that matches no facet value is invisible to it.
 */
export const DUAL_TRUTH_UNKNOWN = 'unknown';

export type SortKey =
	/** As the bundle delivered it — the builder's seeded shuffle. */
	| 'bundle'
	| 'anchor'
	| 'class'
	| 'station'
	| 'probability'
	| 'lead_error'
	| 'intensity'
	| 'review_seq';

/** The dimensions a facet count can hold constant. */
export type FilterDimension =
	| 'class'
	| 'dual_truth'
	| 'season'
	| 'region'
	| 'flag'
	| 'verdict'
	| 'text';

/**
 * What the list is filtered to.
 *
 * Array fields rather than `Set`s, because arrays of a dozen strings are
 * faster to scan than a Set is to build at this size.
 *
 * **Mutating an instance in place is not reactive, and callers must not do
 * it.** Svelte 5's `$state` proxies plain objects and arrays, not class
 * instances, so this whole object is held unwrapped: `filter.toggle(...)`
 * or `bind:value={filter.text}` changes the value and notifies nobody, and
 * the list silently stops responding to its own chips — a failure with no
 * error to notice it by. Callers go through `clone()` and reassign, which
 * is why `clone` exists and why every mutator returns a new instance
 * rather than `this`.
 *
 * Every dimension is a union — a row passes when it matches ANY selected
 * value — and the dimensions intersect. That is the behaviour every facet
 * UI has, and the one a reviewer will assume without being told.
 */
export class FilterState {
	classes: OutcomeClass[] = [];
	dualTruth: Array<DualTruthClass | typeof DUAL_TRUTH_UNKNOWN> = [];
	seasons: string[] = [];
	regions: string[] = [];
	flags: string[] = [];
	verdict: VerdictFilter = 'any';
	/** Free text over the identifying columns, the flags and the note. */
	text = '';
	/**
	 * Show the control group's true class. Off by default and deliberately
	 * awkward to turn on: once seen it cannot be unseen, and the base rate
	 * it protects is the reason the control group was drawn.
	 */
	revealControls = false;
	sort: SortKey = 'bundle';
	descending = false;

	/** A filter that is doing something — worth offering a "clear" button for. */
	get active(): boolean {
		return (
			this.classes.length > 0 ||
			this.dualTruth.length > 0 ||
			this.seasons.length > 0 ||
			this.regions.length > 0 ||
			this.flags.length > 0 ||
			this.verdict !== 'any' ||
			this.text.trim() !== ''
		);
	}

	static from(partial: Partial<FilterState> = {}): FilterState {
		const state = new FilterState();
		Object.assign(state, partial);
		return state;
	}

	clone(): FilterState {
		return FilterState.from({
			classes: [...this.classes],
			dualTruth: [...this.dualTruth],
			seasons: [...this.seasons],
			regions: [...this.regions],
			flags: [...this.flags],
			verdict: this.verdict,
			text: this.text,
			revealControls: this.revealControls,
			sort: this.sort,
			descending: this.descending
		});
	}

	/** Clears the facets and the text, keeping the sort and the reveal state —
	 * those are how the reviewer is working, not what they are looking at. */
	clear(): void {
		this.classes = [];
		this.dualTruth = [];
		this.seasons = [];
		this.regions = [];
		this.flags = [];
		this.verdict = 'any';
		this.text = '';
	}

	/** Add or remove one facet value, in place. */
	toggle(dimension: Exclude<FilterDimension, 'verdict' | 'text'>, value: string): void {
		const list = this.#list(dimension);
		const index = list.indexOf(value);
		if (index >= 0) list.splice(index, 1);
		else list.push(value);
	}

	#list(dimension: Exclude<FilterDimension, 'verdict' | 'text'>): string[] {
		switch (dimension) {
			case 'class':
				return this.classes;
			case 'dual_truth':
				return this.dualTruth;
			case 'season':
				return this.seasons;
			case 'region':
				return this.regions;
			default:
				return this.flags;
		}
	}
}

/**
 * The class to SHOW for a row. `null` means "withheld": a control-group
 * member while the reveal toggle is off. Returning a sentinel rather than
 * the true class keeps the blinding in one place — a component that forgets
 * to check gets a null it has to handle, not a hit it will render.
 */
export function displayClass(row: IndexRow, revealControls: boolean): OutcomeClass | null {
	return row.control && !revealControls ? null : row.class;
}

const haystack = (row: IndexRow, annotation: Annotation | undefined): string =>
	[
		row.event_id,
		row.station_id,
		row.station_name,
		row.region,
		row.season,
		row.intensity_band,
		row.dual_truth ?? DUAL_TRUTH_UNKNOWN,
		...row.flags,
		...(annotation?.tags ?? []),
		annotation?.note ?? ''
	]
		.join(' ')
		.toLowerCase();

/**
 * Does one row pass, ignoring one dimension?
 *
 * `skip` is what makes the facet counts honest: the count shown next to
 * "summer" is the number of rows that would appear if summer were selected,
 * with every *other* filter still applied.
 */
function passes(
	row: IndexRow,
	annotation: Annotation | undefined,
	filter: FilterState,
	skip?: FilterDimension
): boolean {
	// A blinded control row is exempt from the class filter, so filtering to
	// "false alarm" does not empty the control group out of the list and
	// announce which rows it held.
	const classFiltered = filter.classes.length > 0 && skip !== 'class';
	if (classFiltered && !(row.control && !filter.revealControls)) {
		if (!filter.classes.includes(row.class)) return false;
	}
	if (skip !== 'dual_truth' && filter.dualTruth.length > 0) {
		// Null is `unknown`, a value in its own right — never silently one of
		// the four quadrants, and `both_dry` least of all.
		if (!filter.dualTruth.includes(row.dual_truth ?? DUAL_TRUTH_UNKNOWN)) return false;
	}
	if (skip !== 'season' && filter.seasons.length > 0) {
		if (!filter.seasons.includes(row.season)) return false;
	}
	if (skip !== 'region' && filter.regions.length > 0) {
		if (!filter.regions.includes(row.region)) return false;
	}
	if (skip !== 'flag' && filter.flags.length > 0) {
		if (!filter.flags.some((flag) => row.flags.includes(flag))) return false;
	}
	if (skip !== 'verdict' && filter.verdict !== 'any') {
		const verdict = annotation?.verdict ?? null;
		if (filter.verdict === 'untagged' && verdict !== null) return false;
		if (filter.verdict === 'tagged' && verdict === null) return false;
		if (
			filter.verdict !== 'untagged' &&
			filter.verdict !== 'tagged' &&
			verdict !== filter.verdict
		) {
			return false;
		}
	}
	if (skip !== 'text') {
		const needle = filter.text.trim().toLowerCase();
		if (needle !== '' && !haystack(row, annotation).includes(needle)) return false;
	}
	return true;
}

/**
 * The rows the list shows, filtered and sorted.
 *
 * `annotations` is keyed by `event_id` (see `load.annotationsByEvent`) and
 * may be empty — before the annotation server has answered, the verdict
 * facet behaves as "nothing is judged yet", which is true rather than
 * convenient.
 */
export function applyFilters(
	rows: readonly IndexRow[],
	filter: FilterState,
	annotations: ReadonlyMap<string, Annotation> = new Map()
): IndexRow[] {
	const kept = rows.filter((row) => passes(row, annotations.get(row.event_id), filter));
	return sortRows(kept, filter.sort, filter.descending, annotations);
}

const NULL_LAST = Number.POSITIVE_INFINITY;

/**
 * Sort, with nulls last in both directions.
 *
 * A false alarm has no lead error and a miss has no probability, and those
 * rows belong at the end of the list whichever way it is pointing — sorting
 * them to the top as if they were zero would put the events with the least
 * information first, every time.
 *
 * The tie-break is always `event_id`, so the order is total and a re-render
 * cannot reshuffle rows the reviewer is halfway through.
 */
export function sortRows(
	rows: readonly IndexRow[],
	key: SortKey,
	descending = false,
	annotations: ReadonlyMap<string, Annotation> = new Map()
): IndexRow[] {
	const out = [...rows];
	if (key === 'bundle') {
		// The builder's own order is a seeded shuffle that interleaves the
		// classes; reversing it is still meaningful, re-sorting it is not.
		return descending ? out.reverse() : out;
	}
	const number = (row: IndexRow): number => {
		switch (key) {
			case 'anchor':
				return Date.parse(row.anchor_utc);
			case 'probability':
				return row.p_decision ?? NULL_LAST;
			case 'lead_error':
				return row.lead_error_min ?? NULL_LAST;
			case 'intensity':
				return row.intensity_mm_h ?? NULL_LAST;
			case 'review_seq':
				return annotations.get(row.event_id)?.review_seq ?? NULL_LAST;
			default:
				return NULL_LAST;
		}
	};
	const text = (row: IndexRow): string =>
		key === 'class' ? row.class : key === 'station' ? row.station_name : '';

	out.sort((a, b) => {
		let order = 0;
		if (key === 'class' || key === 'station') {
			order = text(a).localeCompare(text(b));
		} else {
			const left = number(a);
			const right = number(b);
			// Nulls stay at the bottom whichever way the arrow points.
			if (left === NULL_LAST && right !== NULL_LAST) return 1;
			if (right === NULL_LAST && left !== NULL_LAST) return -1;
			order = left === right ? 0 : left < right ? -1 : 1;
		}
		if (order === 0) return a.event_id.localeCompare(b.event_id);
		return descending ? -order : order;
	});
	return out;
}

export interface Facet {
	value: string;
	count: number;
	selected: boolean;
}

export interface FacetCounts {
	class: Facet[];
	dual_truth: Facet[];
	season: Facet[];
	region: Facet[];
	flag: Facet[];
	verdict: Facet[];
	/**
	 * How many control-group rows are in the current view. Counted as one
	 * anonymous bucket while the reveal toggle is off: a per-class breakdown
	 * of the control group is the same leak as filtering by class.
	 */
	control: number;
}

function tally(values: Iterable<string>, selected: readonly string[]): Facet[] {
	const counts = new Map<string, number>();
	for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
	// Count descending, then alphabetical: the biggest bucket is usually the
	// one a reviewer wants, and equal buckets must not swap places between
	// renders.
	return [...counts.entries()]
		.map(([value, count]) => ({ value, count, selected: selected.includes(value) }))
		.sort((a, b) => b.count - a.count || a.value.localeCompare(b.value));
}

/**
 * The counts beside each facet value.
 *
 * Each dimension is counted with its own filter lifted, so the numbers say
 * "select this and you will see N" rather than "you have selected this".
 * Values with a zero count do not appear at all: the bundle is a stratified
 * sample and an empty stratum is not a thing to click on.
 */
export function facetCounts(
	rows: readonly IndexRow[],
	filter: FilterState,
	annotations: ReadonlyMap<string, Annotation> = new Map()
): FacetCounts {
	const forDimension = (dimension: FilterDimension): IndexRow[] =>
		rows.filter((row) => passes(row, annotations.get(row.event_id), filter, dimension));

	const classRows = forDimension('class');
	const revealed = filter.revealControls;
	const classValues = classRows
		.filter((row) => revealed || !row.control)
		.map((row) => row.class);

	const verdictRows = forDimension('verdict');
	const verdictValues = verdictRows.map((row) => {
		const verdict = annotations.get(row.event_id)?.verdict ?? null;
		return verdict === null ? 'untagged' : verdict;
	});

	return {
		class: tally(classValues, filter.classes),
		dual_truth: tally(
			forDimension('dual_truth').map((row) => row.dual_truth ?? DUAL_TRUTH_UNKNOWN),
			filter.dualTruth
		),
		season: tally(
			forDimension('season').map((row) => row.season).filter((season) => season !== ''),
			filter.seasons
		),
		region: tally(
			forDimension('region').map((row) => row.region).filter((region) => region !== ''),
			filter.regions
		),
		flag: tally(
			forDimension('flag').flatMap((row) => row.flags),
			filter.flags
		),
		verdict: tally(verdictValues, filter.verdict === 'any' ? [] : [filter.verdict]),
		control: classRows.filter((row) => row.control).length
	};
}

export interface ReviewProgress {
	total: number;
	/** Rows the server holds anything for, judged or not. */
	stored: number;
	/** Rows carrying a verdict — the server's own definition of reviewed. */
	withVerdict: number;
	/** Rows that also carry a tag, and a note where the vocabulary demands one. */
	complete: number;
	remaining: number;
	/** 0 … 1, by verdict. Null when there is nothing to divide by. */
	fraction: number | null;
	byVerdict: Record<string, number>;
	/** Per outcome class, so a reviewer can see they have done every miss. */
	byClass: Record<string, { total: number; withVerdict: number }>;
}

/**
 * How far the review has got.
 *
 * `withVerdict` and `complete` are both reported because they answer
 * different questions: the server counts a row as reviewed when it carries
 * a verdict, while a reviewer wants to know how many events they have
 * actually finished — a verdict with no mechanism tag is a judgement that
 * records *that* something failed without recording *what*, which is the
 * aggregate this whole tool exists to replace.
 *
 * Counted over the rows handed in, so it reports on the filtered view when
 * given one and on the bundle when given the whole index.
 */
export function reviewProgress(
	rows: readonly IndexRow[],
	annotations: ReadonlyMap<string, Annotation> = new Map()
): ReviewProgress {
	const byVerdict: Record<string, number> = {};
	const byClass: Record<string, { total: number; withVerdict: number }> = {};
	let stored = 0;
	let withVerdict = 0;
	let complete = 0;

	for (const row of rows) {
		const bucket = (byClass[row.class] ??= { total: 0, withVerdict: 0 });
		bucket.total += 1;
		const annotation = annotations.get(row.event_id);
		if (annotation === undefined) continue;
		stored += 1;
		if (annotation.verdict === null) continue;
		withVerdict += 1;
		bucket.withVerdict += 1;
		byVerdict[annotation.verdict] = (byVerdict[annotation.verdict] ?? 0) + 1;
		if (isComplete(annotation)) complete += 1;
	}

	return {
		total: rows.length,
		stored,
		withVerdict,
		complete,
		remaining: rows.length - withVerdict,
		fraction: rows.length === 0 ? null : withVerdict / rows.length,
		byVerdict,
		byClass
	};
}

/**
 * The next event still to judge, for the keyboard shortcut that moves on
 * after a save.
 *
 * Search starts *after* the current event and wraps once, so a reviewer
 * working down the list is not thrown back to the top on every save, and
 * one working backwards is not stuck. Null when everything visible is
 * judged — which is the signal to say so rather than to jump somewhere.
 */
export function nextUnreviewed(
	rows: readonly IndexRow[],
	annotations: ReadonlyMap<string, Annotation>,
	afterEventId: string | null = null
): IndexRow | null {
	if (rows.length === 0) return null;
	const start = afterEventId === null ? -1 : rows.findIndex((row) => row.event_id === afterEventId);
	for (let step = 1; step <= rows.length; step++) {
		const row = rows[(start + step + rows.length) % rows.length];
		if ((annotations.get(row.event_id)?.verdict ?? null) === null) return row;
	}
	return null;
}
