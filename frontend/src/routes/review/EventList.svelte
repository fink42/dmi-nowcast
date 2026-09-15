<script lang="ts">
	/**
	 * The left pane: ~300 events, what is left to judge, and the facets that
	 * narrow them.
	 *
	 * No virtualisation — three hundred rows is nothing, and a virtualised
	 * list would break the one thing this pane has to get right, which is
	 * that `j`/`k` walk it in the order it is drawn and the selected row is
	 * always on screen.
	 *
	 * Two things here are about the review rather than the UI:
	 *
	 *  - **The control group stays blind.** `displayClass` returns null for a
	 *    control row while the reveal toggle is off, and the badge then says
	 *    "withheld" rather than borrowing a neighbour's class. Printing a
	 *    plausible class would be a lie; printing nothing at all would let
	 *    the row read as a bundle bug. The class filter deliberately does not
	 *    remove these rows (see `filter.ts`), so they stay interleaved.
	 *  - **Facet counts hold the other facets.** The number beside "summer"
	 *    is how many rows you would see if you selected summer, with every
	 *    other filter still applied — so a count of zero never appears, and a
	 *    filter that found nothing is visibly different from a bundle that
	 *    contains nothing.
	 */
	import {
		displayClass,
		FilterState,
		nextUnreviewed,
		type Facet,
		type SortKey,
		type VerdictFilter
	} from '$lib/review/filter';
	import { review } from '$lib/review/store.svelte';
	import { dualTruthLabel } from '$lib/review/truth';
	import { minutesText, pctText, utcDay, utcClock } from './format';

	interface Props {
		/** Open an event. The page owns the unsaved-changes guard. */
		select: (eventId: string) => void;
	}
	let { select }: Props = $props();

	const rows = $derived(review.visibleRows);
	const facets = $derived(review.facets);
	const progress = $derived(review.progress);
	const reveal = $derived(review.filter.revealControls);

	/**
	 * The facet groups, typed here rather than inline in the markup: Svelte's
	 * template expressions are JavaScript, so a `as 'class' | 'dual_truth'`
	 * next to the click handler would not compile.
	 */
	interface ChipGroup {
		dimension: 'class' | 'dual_truth' | 'season' | 'region' | 'flag';
		label: string;
		values: Facet[];
	}
	const leadGroups: ChipGroup[] = $derived([
		{ dimension: 'class', label: 'Class', values: facets.class },
		{ dimension: 'dual_truth', label: 'Dual truth', values: facets.dual_truth }
	]);
	const extraGroups: ChipGroup[] = $derived([
		{ dimension: 'season', label: 'Season', values: facets.season },
		{ dimension: 'region', label: 'Region', values: facets.region },
		{ dimension: 'flag', label: 'Flags', values: facets.flag }
	]);

	/** The sort columns, in the order the header offers them. */
	const SORTS: Array<{ key: SortKey; label: string; title: string }> = [
		{ key: 'bundle', label: 'Bundle', title: "the builder's seeded shuffle — classes interleaved" },
		{ key: 'anchor', label: 'When', title: 'the anchor instant' },
		{ key: 'class', label: 'Class', title: 'outcome class' },
		{ key: 'station', label: 'Station', title: 'station name' },
		{ key: 'probability', label: 'p', title: 'the deciding probability; rows without one sort last' },
		{ key: 'lead_error', label: 'Lead err', title: 'positive means the rain beat the ETA — the warning was late' },
		{ key: 'intensity', label: 'mm/h', title: 'intensity band source varies; see the event header' },
		{ key: 'review_seq', label: 'Judged', title: 'the order you first saved them — the criteria-drift check' }
	];

	/**
	 * Every filter change goes through a clone.
	 *
	 * `FilterState` is a class, and Svelte 5's `$state` proxies plain objects
	 * and arrays but hands a class instance back unwrapped — so
	 * `filter.classes.push(…)` would change the value and tell nobody, and the
	 * list would silently stop responding to its own chips. `clone()` is on
	 * `FilterState` for exactly this; the reassignment is the reactive event.
	 */
	function editFilter(change: (filter: FilterState) => void): void {
		const next = review.filter.clone();
		change(next);
		review.filter = next;
	}

	/** Toggling the column that is already sorted flips the direction. */
	function sortBy(key: SortKey): void {
		editFilter((filter) => {
			if (filter.sort === key) filter.descending = !filter.descending;
			else {
				filter.sort = key;
				filter.descending = false;
			}
		});
	}

	// Handlers rather than `bind:`, for the same reason: a binding would write
	// straight through to the unproxied instance. They live here rather than in
	// the markup because a Svelte template expression is JavaScript, and these
	// need a cast.
	const setText = (value: string) => editFilter((filter) => (filter.text = value));
	const setVerdict = (value: string) =>
		editFilter((filter) => (filter.verdict = value as VerdictFilter));
	const setReveal = (checked: boolean) =>
		editFilter((filter) => (filter.revealControls = checked));

	/**
	 * The next event in this view with no verdict on it.
	 *
	 * Routed through `select` rather than through `store.selectNextUnreviewed`
	 * so it goes past the page's unsaved-changes guard: the store's own
	 * shortcut would open the next event and drop a typed note on the way.
	 */
	function goNextUnjudged(): void {
		const next = nextUnreviewed(rows, review.annotations, review.selectedId);
		if (next !== null) select(next.event_id);
	}

	/** The judgement mark: what is stored for this row, at a glance. */
	function mark(eventId: string): { glyph: string; title: string } {
		const annotation = review.annotations.get(eventId);
		if (annotation === undefined) return { glyph: '', title: 'not judged' };
		if (annotation.verdict === null) return { glyph: '·', title: 'stored, with no verdict' };
		const second = annotation.needs_second_look ? ' · needs a second look' : '';
		const glyph = annotation.needs_second_look ? '!' : annotation.tags.length > 0 ? '✓' : '–';
		return {
			glyph,
			title: `${annotation.verdict}${annotation.tags.length > 0 ? ` · ${annotation.tags.join(', ')}` : ' · no mechanism tag'}${second}`
		};
	}

	/**
	 * Keep the selected row on screen. `j`/`k` are handled by the page, which
	 * has no idea where the list scrolled to, so the list follows the
	 * selection rather than the other way round.
	 */
	$effect(() => {
		const id = review.selectedId;
		if (id === null) return;
		document.getElementById(`review-row-${id}`)?.scrollIntoView({ block: 'nearest' });
	});
</script>

<div class="list">
	<header>
		<p class="progress">
			<strong>{progress.withVerdict} / {progress.total}</strong> reviewed
			{#if progress.byVerdict.unclear}· {progress.byVerdict.unclear} unclear{/if}
			{#if progress.complete !== progress.withVerdict}
				· {progress.complete} with a mechanism tag
			{/if}
		</p>
		<p class="quiet">
			{rows.length} shown{#if facets.control > 0}, {facets.control}
				{reveal ? 'control (revealed)' : 'in the control group'}{/if}
			<!-- Search starts after the current event and wraps once, so working
			     down the list is not thrown back to the top on every save. -->
			<button
				type="button"
				class="chip"
				onclick={goNextUnjudged}
				title="The next event in this view that carries no verdict yet.">next unjudged</button
			>
		</p>

		<input
			class="text"
			type="search"
			placeholder="station, id, flag, tag, note…"
			value={review.filter.text}
			oninput={(event) => setText(event.currentTarget.value)}
		/>

		<div class="facets">
			{#each leadGroups as group (group.dimension)}
				<div class="facet">
					<span class="facet-label">{group.label}</span>
					{#each group.values as facet (facet.value)}
						<button
							type="button"
							class="chip"
							class:on={facet.selected}
							onclick={() => editFilter((filter) => filter.toggle(group.dimension, facet.value))}
						>
							{facet.value}<span class="count">{facet.count}</span>
						</button>
					{/each}
				</div>
			{/each}

			<details>
				<summary>Season, region, flags, verdict</summary>
				{#each extraGroups as group (group.dimension)}
					{#if group.values.length > 0}
						<div class="facet">
							<span class="facet-label">{group.label}</span>
							{#each group.values as facet (facet.value)}
								<button
									type="button"
									class="chip"
									class:on={facet.selected}
									onclick={() => editFilter((filter) => filter.toggle(group.dimension, facet.value))}
								>
									{facet.value}<span class="count">{facet.count}</span>
								</button>
							{/each}
						</div>
					{/if}
				{/each}
				<div class="facet">
					<span class="facet-label">Judgement</span>
					<select
						value={review.filter.verdict}
						onchange={(event) => setVerdict(event.currentTarget.value)}
					>
						<option value="any">any</option>
						<option value="untagged">not judged</option>
						<option value="tagged">judged</option>
						<option value="real_failure">real_failure</option>
						<option value="metric_artefact">metric_artefact</option>
						<option value="unclear">unclear</option>
					</select>
				</div>
				<!-- Deliberately awkward to reach: once the control group is seen
				     it cannot be unseen, and the base rate it protects is the
				     reason it was drawn. -->
				<label class="reveal">
					<input
						type="checkbox"
						checked={review.filter.revealControls}
						onchange={(event) => setReveal(event.currentTarget.checked)}
					/>
					Reveal the control group's classes
				</label>
			</details>

			{#if review.filter.active}
				<button
					type="button"
					class="chip clear"
					onclick={() => editFilter((filter) => filter.clear())}
				>
					Clear filters
				</button>
			{/if}
		</div>

		<div class="sorts" role="group" aria-label="Sort the list">
			{#each SORTS as column (column.key)}
				<button
					type="button"
					class="sort"
					class:on={review.filter.sort === column.key}
					title={column.title}
					onclick={() => sortBy(column.key)}
				>
					{column.label}{#if review.filter.sort === column.key}<span aria-hidden="true"
							>{review.filter.descending ? '↓' : '↑'}</span
						>{/if}
				</button>
			{/each}
		</div>
	</header>

	<ol>
		{#each rows as row (row.event_id)}
			{@const shown = displayClass(row, reveal)}
			{@const judged = mark(row.event_id)}
			<li id="review-row-{row.event_id}">
				<button
					type="button"
					class="row"
					class:selected={row.event_id === review.selectedId}
					aria-current={row.event_id === review.selectedId ? 'true' : undefined}
					onclick={() => select(row.event_id)}
				>
					<span class="mark" title={judged.title}>{judged.glyph}</span>
					<span class="body">
						<span class="line">
							<!-- A control row shows "withheld", never a borrowed class:
							     the reviewer learns that the class is hidden and not
							     what it is. -->
							<span class="badge {shown ?? 'withheld'}"
								>{shown ?? 'withheld'}</span
							>
							<span class="station">{row.station_name}</span>
							<span class="quiet">{row.station_id}</span>
						</span>
						<span class="line quiet">
							<!-- `13:45Z`: UTC, compactly, because the row sits beside a
							     hundred others and every stamp in this tool is UTC. -->
							{utcDay(row.anchor_utc) ?? 'unplaceable'}
							{utcClock(row.anchor_utc) ?? '??:??'}Z · {dualTruthLabel(row.dual_truth).label}
							{#if row.p_decision !== null}
								· p {pctText(row.p_decision)}{#if row.threshold_pct !== null}
									/ {row.threshold_pct} %{/if}
							{/if}
							{#if row.lead_error_min !== null}
								· lead err {minutesText(row.lead_error_min)}
							{/if}
						</span>
						{#if row.flags.length > 0}
							<span class="line flags">{row.flags.join(' · ')}</span>
						{/if}
					</span>
				</button>
			</li>
		{/each}
		{#if rows.length === 0}
			<li class="empty">
				No event matches these filters. The bundle holds {progress.total}.
			</li>
		{/if}
	</ol>
</div>

<style>
	.list {
		display: flex;
		flex-direction: column;
		min-height: 0;
		height: 100%;
		background: var(--surface);
		border-right: 1px solid var(--border);
	}

	header {
		flex: 0 0 auto;
		padding: 0.5rem 0.6rem;
		border-bottom: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
	}

	.progress {
		margin: 0;
		font-size: 0.85rem;
	}

	.quiet {
		margin: 0;
		color: var(--muted);
		font-size: 0.75rem;
	}

	.text {
		width: 100%;
		padding: 0.3rem 0.45rem;
		border: 1px solid var(--border);
		border-radius: 8px;
		background: var(--bg);
		color: var(--ink);
		font-size: 0.8rem;
	}

	.facets {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
	}

	.facet {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.25rem;
	}

	.facet-label {
		font-size: 0.68rem;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: var(--muted);
		margin-right: 0.15rem;
	}

	.chip {
		border: 1px solid var(--border);
		background: var(--bg);
		color: var(--ink);
		border-radius: 999px;
		padding: 0.05rem 0.45rem;
		font-size: 0.72rem;
		cursor: pointer;
	}

	.chip.on {
		background: var(--accent);
		color: var(--accent-ink);
		border-color: var(--accent);
	}

	.chip .count {
		margin-left: 0.3rem;
		opacity: 0.7;
		font-variant-numeric: tabular-nums;
	}

	.chip.clear {
		align-self: flex-start;
	}

	details {
		font-size: 0.75rem;
	}

	summary {
		cursor: pointer;
		color: var(--muted);
		margin-bottom: 0.25rem;
	}

	select {
		font-size: 0.72rem;
		background: var(--bg);
		color: var(--ink);
		border: 1px solid var(--border);
		border-radius: 6px;
	}

	.reveal {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		font-size: 0.72rem;
		color: var(--warn);
		margin-top: 0.3rem;
	}

	.sorts {
		display: flex;
		flex-wrap: wrap;
		gap: 0.2rem;
	}

	.sort {
		border: none;
		background: none;
		color: var(--muted);
		font-size: 0.7rem;
		padding: 0.05rem 0.25rem;
		cursor: pointer;
		border-radius: 4px;
	}

	.sort.on {
		color: var(--ink);
		background: var(--track);
		font-weight: 600;
	}

	ol {
		flex: 1 1 auto;
		min-height: 0;
		overflow-y: auto;
		list-style: none;
		margin: 0;
		padding: 0;
	}

	li {
		border-bottom: 1px solid var(--border);
	}

	li.empty {
		padding: 0.8rem 0.6rem;
		color: var(--muted);
		font-size: 0.8rem;
	}

	.row {
		display: flex;
		gap: 0.4rem;
		width: 100%;
		text-align: left;
		border: none;
		background: none;
		color: inherit;
		padding: 0.35rem 0.5rem;
		cursor: pointer;
	}

	.row:hover {
		background: var(--bg);
	}

	.row.selected {
		background: var(--track);
		box-shadow: inset 3px 0 0 var(--accent);
	}

	.mark {
		flex: 0 0 0.9rem;
		font-size: 0.85rem;
		color: var(--accent);
		line-height: 1.3;
	}

	.body {
		flex: 1 1 auto;
		min-width: 0;
		display: flex;
		flex-direction: column;
		gap: 0.05rem;
	}

	.line {
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem;
		align-items: baseline;
		font-size: 0.78rem;
	}

	.line.quiet {
		font-size: 0.7rem;
		color: var(--muted);
	}

	.station {
		font-weight: 600;
	}

	.flags {
		font-size: 0.68rem;
		color: var(--warn);
	}

	.badge {
		font-size: 0.65rem;
		text-transform: uppercase;
		letter-spacing: 0.03em;
		border: 1px solid var(--border);
		border-radius: 4px;
		padding: 0 0.25rem;
		color: var(--muted);
		white-space: nowrap;
	}

	.badge.false_alarm,
	.badge.late {
		color: var(--warn);
		border-color: var(--warn);
	}

	.badge.miss,
	.badge.miss_late {
		color: var(--accent);
		border-color: var(--accent);
	}

	/* Withheld is its own look: not a class, and not an error either. */
	.badge.withheld {
		background-image: repeating-linear-gradient(-45deg, var(--muted) 0 2px, transparent 2px 5px);
		opacity: 0.8;
	}
</style>
