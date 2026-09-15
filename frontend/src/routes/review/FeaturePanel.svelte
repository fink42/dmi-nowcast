<script lang="ts">
	/**
	 * The Phase-H features behind the estimate at the cursor, each with the
	 * producer's own definition.
	 *
	 * These columns are what separates one mechanism from another — an empty
	 * upwind corridor is convective initiation, which advection cannot see; a
	 * high `stalled_share` is clutter; a large `frame_age_min` ate the lead —
	 * so the definitions travel with them rather than living in a doc nobody
	 * has open. `manifest.feature_doc` is the producer's copy, which is the
	 * only one that cannot drift from the column.
	 *
	 * `features_present: false` means the decision row's feature columns were
	 * null, which happened for real between 2026-09-05 and the fix on
	 * 2026-09-13. The panel greys out and says so, because a reviewer reading
	 * blanks as zeroes would tag `miss_convective_initiation` on every one of
	 * those rows: "nothing upstream" and "nobody wrote down what was upstream"
	 * look identical in an empty cell.
	 */
	import { review } from '$lib/review/store.svelte';
	import { NOT_MEASURED, numberText } from './format';

	const decision = $derived(review.decision);
	const docs = $derived(review.manifest?.feature_doc ?? null);
	const present = $derived(decision?.features_present ?? false);

	interface FeatureRow {
		name: string;
		text: string;
		doc: string | null;
		/** True when the producer wrote no value — never rendered as a zero. */
		missing: boolean;
	}

	const rows: FeatureRow[] = $derived(
		Object.entries(decision?.features ?? {}).map(([name, value]) => ({
			name,
			text:
				value === null
					? NOT_MEASURED
					: typeof value === 'number'
						? (numberText(value, 2) ?? NOT_MEASURED)
						: String(value),
			doc: docs?.[name] ?? null,
			missing: value === null
		}))
	);

	/**
	 * Definitions in line, rather than only in a `title`. Off by default
	 * because the list is fourteen rows long and a reviewer who knows the
	 * columns wants them dense; on, because a reviewer who does not cannot
	 * discover a tooltip they have no reason to hover.
	 */
	let showDocs = $state(false);

	/** The upwind corridor's emptiness is a finding, not a gap. */
	const emptyCorridor = $derived(
		decision !== null &&
			decision.features_present &&
			(decision.features.up_dist_km === null || decision.features.up_dist_km === undefined)
	);
</script>

<section class="panel" class:absent={decision !== null && !present}>
	<h2>
		Features at the cursor
		{#if docs !== null}
			<button type="button" class="docs-toggle" onclick={() => (showDocs = !showDocs)}>
				{showDocs ? 'hide definitions' : 'definitions'}
			</button>
		{/if}
	</h2>

	{#if decision === null}
		<p class="quiet">No estimate at this instant, so no features.</p>
	{:else if !present}
		<p class="quiet">
			Not computed for this row: the decision row's feature columns were null, so
			everything below is absent rather than zero. A row that fell back to the curve
			for its probability is judged on a different scale from the thresholds — see the
			estimate panel.
		</p>
	{/if}

	{#if decision !== null && rows.length > 0}
		<dl>
			{#each rows as row (row.name)}
				<div class="row" class:missing={row.missing}>
					<dt title={row.doc ?? 'no definition in this bundle'}>
						{row.name}
						{#if row.doc}<span class="mark" aria-hidden="true">?</span>{/if}
					</dt>
					<dd>{row.text}</dd>
					{#if row.doc && showDocs}
						<p class="doc">{row.doc}</p>
					{/if}
				</div>
			{/each}
		</dl>
	{/if}

	{#if emptyCorridor}
		<p class="quiet">
			<strong>up_dist_km is absent with the features present.</strong> The pipeline writes
			NaN when the 40 km upwind corridor held no echo at all — the signature of rain
			growing in place, which advection cannot see. It is emphatically not a distance of
			zero.
		</p>
	{/if}

	{#if review.upwind === null}
		<p class="quiet">
			No usable motion at this cycle, so no upwind arrow is drawn on the map. Below the
			flow floor the pipeline writes NaN for the bearing rather than a direction, and an
			arrow drawn there would be the tool inventing the evidence it exists to check.
		</p>
	{:else}
		<p class="quiet">
			Arrow: rain coming from {Math.round(review.upwind.bearingFromDeg)}° at
			{numberText(review.upwind.speedKmh, 0)} km/h (bulk);
			{#if review.upwind.localSpeedKmh !== null}
				{numberText(review.upwind.localSpeedKmh, 0)} km/h at the station;
			{/if}
			nearest upwind echo
			{review.upwind.upstreamDistanceKm === null
				? 'nowhere in the corridor'
				: `${numberText(review.upwind.upstreamDistanceKm, 0)} km`}.
		</p>
	{/if}
</section>

<style>
	.panel {
		border-bottom: 1px solid var(--border);
		padding: 0.55rem 0.7rem 0.7rem;
	}

	/* Greyed as a whole: the reader should see at a glance that this row's
	   features are absent, not read one blank at a time. */
	.panel.absent dl {
		opacity: 0.5;
	}

	h2 {
		font-size: 0.78rem;
		text-transform: uppercase;
		letter-spacing: 0.03em;
		color: var(--muted);
		margin: 0 0 0.35rem;
	}

	p {
		margin: 0 0 0.25rem;
		font-size: 0.8rem;
	}

	.quiet {
		color: var(--muted);
		font-size: 0.74rem;
	}

	dl {
		margin: 0;
		font-size: 0.76rem;
	}

	.row {
		display: grid;
		grid-template-columns: 9.5rem 1fr;
		gap: 0 0.4rem;
		border-bottom: 1px solid var(--border);
		padding: 0.1rem 0;
	}

	dt {
		color: var(--muted);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		cursor: help;
	}

	dd {
		margin: 0;
		font-variant-numeric: tabular-nums;
	}

	/* A column the producer left null. Not a zero, and not styled like one. */
	.row.missing dd {
		color: var(--warn);
		font-style: italic;
	}

	.mark {
		font-size: 0.62rem;
		border: 1px solid var(--border);
		border-radius: 50%;
		padding: 0 0.22rem;
		margin-left: 0.2rem;
	}

	/* The producer's own definition, in line. Also on the term's `title`, so
	   it is reachable with the list dense. */
	.doc {
		grid-column: 1 / -1;
		margin: 0 0 0.15rem;
		font-size: 0.68rem;
		color: var(--muted);
	}

	.docs-toggle {
		float: right;
		border: 1px solid var(--border);
		background: var(--bg);
		color: var(--muted);
		border-radius: 999px;
		font-size: 0.64rem;
		text-transform: none;
		letter-spacing: 0;
		padding: 0 0.4rem;
		cursor: pointer;
	}
</style>
