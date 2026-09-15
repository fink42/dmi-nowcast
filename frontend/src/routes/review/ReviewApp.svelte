<script lang="ts">
	/**
	 * The event review tool: three panes, one cursor, one judgement at a time.
	 *
	 * The push rule scores POD 0.38 / FAR 0.67 against gauge onsets and nobody
	 * has ever looked at an individual bad warning. This page exists to turn
	 * that pooled F1 into a ranked list of named failure mechanisms: scrub the
	 * radar loop around one event, read what the service had at every step,
	 * and tag the cause.
	 *
	 * It is dev-only. `+page.ts` refuses the route outside `npm run dev` and
	 * `+page.svelte` reaches this file through a dynamic import that Rollup
	 * drops from a production build, so neither this component nor anything
	 * under `$lib/review/**` is ever served to the public site.
	 *
	 * The layout is fixed to the viewport rather than flowing inside the
	 * site's shell: the three panes each scroll on their own, and a page that
	 * scrolls as a whole would put the scrubber below the fold exactly when a
	 * reviewer reaches for it.
	 *
	 * Strings are English literals. `i18n/catalog.test.ts` pins the `da` and
	 * `en` catalogs to identical keys, and this vocabulary — `fa_cell_died`,
	 * `run_boundary_rearm` — has no business in the public site's bundle.
	 */
	import { onMount } from 'svelte';
	import { displayClass } from '$lib/review/filter';
	import { REVIEW_SCHEMA_VERSION, type Caveat } from '$lib/review/schema';
	import { review } from '$lib/review/store.svelte';
	import { tagsForClass } from '$lib/review/tags';
	import AnnotationForm from './AnnotationForm.svelte';
	import BundleGate from './BundleGate.svelte';
	import EstimatePanel from './EstimatePanel.svelte';
	import EventList from './EventList.svelte';
	import FeaturePanel from './FeaturePanel.svelte';
	import ReviewMap from './ReviewMap.svelte';
	import Scrubber from './Scrubber.svelte';
	import TruthPanel from './TruthPanel.svelte';
	import { minutesText, NOT_MEASURED, pctText, provenanceText, utcStamp, localStamp } from './format';

	let helpOpen = $state(false);
	/** Read a bundle whose schema version this client does not know. */
	let schemaAccepted = $state(false);

	onMount(() => {
		void review.load();
		// `stop()` is not optional: it clears the frame timer, aborts the
		// in-flight fetches and closes every decoded `ImageBitmap`. A bitmap
		// that is never closed is memory the GC cannot reclaim on its own.
		return () => review.stop();
	});

	const manifest = $derived(review.manifest);
	const selected = $derived(review.selectedRow);
	const provenance = $derived(provenanceText(manifest?.rule?.probability_provenance));
	const schemaKnown = $derived(
		manifest === null || manifest.schema_version === REVIEW_SCHEMA_VERSION
	);

	/**
	 * Which gate to show, if any. Computed here rather than inside
	 * `BundleGate` so the page and the gate cannot disagree about whether
	 * there is anything to review.
	 */
	const gate = $derived(
		review.status === 'idle' || review.status === 'loading'
			? 'loading'
			: review.status === 'error'
				? 'failed'
				: !schemaKnown && !schemaAccepted
					? 'schema'
					: null
	);

	const SEVERITY_RANK: Record<string, number> = { high: 0, medium: 1, low: 2 };
	/** Caveats, worst first — the `severity` field is there to rank them. */
	const caveats: Caveat[] = $derived(
		[...(manifest?.caveats ?? [])].sort(
			(a, b) => (SEVERITY_RANK[a.severity] ?? 3) - (SEVERITY_RANK[b.severity] ?? 3)
		)
	);
	const loudCaveats = $derived(caveats.filter((caveat) => caveat.severity === 'high'));

	/** The class to show for the open event — null while a control is blind. */
	const shownClass = $derived(
		selected === null ? null : displayClass(selected, review.filter.revealControls)
	);

	/**
	 * The cause lists offered for this event.
	 *
	 * A blinded control row is asked for under an empty class, which
	 * `tagsForClass` answers with every group — the same superset every
	 * blinded row gets. Passing the true class would leak it: a `hit` is the
	 * only class offered both lists, so a row showing both would announce
	 * itself.
	 */
	const groups = $derived(
		selected === null ? [] : tagsForClass(review.vocabulary, shownClass ?? '')
	);
	/** What `1`–`9` reach, in the order the form lists them. */
	const quickTags = $derived(groups.flatMap((group) => group.tags).slice(0, 9));

	/**
	 * Opening another event throws away an unsaved judgement, so it asks
	 * first. A dev tool may use `confirm`; losing a typed note to a keystroke
	 * is worse than a modal.
	 */
	function openEvent(eventId: string): void {
		if (eventId === review.selectedId) return;
		if (review.dirty && !confirm('This event has an unsaved judgement. Leave it without saving?')) {
			return;
		}
		void review.selectEvent(eventId);
	}

	/** Walk the visible list, in the order it is drawn. */
	function moveSelection(delta: number): void {
		const list = review.visibleRows;
		if (list.length === 0) return;
		const at = review.selectedId === null ? -1 : list.findIndex((row) => row.event_id === review.selectedId);
		const next = list[Math.min(list.length - 1, Math.max(0, at + delta))];
		if (next !== undefined) openEvent(next.event_id);
	}

	/** Typing: every shortcut is off, including the letters. */
	function isTextEntry(target: EventTarget | null): boolean {
		if (!(target instanceof HTMLElement)) return false;
		if (target.isContentEditable) return true;
		if (target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return true;
		if (!(target instanceof HTMLInputElement)) return false;
		return !['checkbox', 'radio', 'range', 'button', 'submit'].includes(target.type);
	}

	/**
	 * A control that owns its own arrows and space bar — the scrubber's range
	 * input, a checkbox, a button. Those keys are left to it, so focusing the
	 * track and pressing ← still scrubs (a minute at a time, the input's own
	 * step) and space still toggles a checkbox.
	 */
	function ownsNavigationKeys(target: EventTarget | null): boolean {
		return (
			target instanceof HTMLInputElement ||
			target instanceof HTMLButtonElement ||
			target instanceof HTMLSelectElement ||
			target instanceof HTMLAnchorElement
		);
	}

	function onKeydown(event: KeyboardEvent): void {
		// A browser or OS shortcut is never one of ours.
		if (event.metaKey || event.ctrlKey || event.altKey) return;
		if (isTextEntry(event.target)) return;
		const control = ownsNavigationKeys(event.target);

		switch (event.key) {
			case 'j':
				moveSelection(1);
				return;
			case 'k':
				moveSelection(-1);
				return;
			case 'ArrowRight':
				if (control) return;
				event.preventDefault();
				review.step(1);
				return;
			case 'ArrowLeft':
				if (control) return;
				event.preventDefault();
				review.step(-1);
				return;
			case ' ':
				if (control) return;
				event.preventDefault();
				review.toggle();
				return;
			case 's':
				void review.save();
				return;
			case '?':
				helpOpen = !helpOpen;
				return;
			case 'Escape':
				helpOpen = false;
				return;
			default:
				break;
		}
		if (/^[1-9]$/.test(event.key)) {
			const tag = quickTags[Number(event.key) - 1];
			if (tag !== undefined) review.toggleTag(tag.code);
		}
	}
</script>

<svelte:window onkeydown={onKeydown} />

<div class="review" data-review-tool="dmi-nowcast-event-review">
	{#if gate !== null}
		<BundleGate
			kind={gate}
			onretry={() => void review.load()}
			onproceed={() => (schemaAccepted = true)}
		/>
	{:else}
		<header class="topbar">
			<span class="title">Event review</span>
			<span class="quiet">{manifest?.bundle_id ?? 'no bundle id'}</span>

			<!-- The one manifest field that decides whether the sample is
			     honest, next to the events it describes rather than filed under
			     provenance. -->
			<span class="chip" class:bad={provenance.contaminated} title={provenance.reading}>
				probabilities: {provenance.label}
			</span>
			{#if manifest?.rule}
				<span class="quiet">
					{manifest.rule.source} rule · {manifest.rule.lead_min} min lead ·
					{manifest.rule.persistence_obs} over threshold to fire ·
					{manifest.rule.rearm_after_min} min re-arm · {manifest.rule.probability_column}
				</span>
				{#if !manifest.rule.held_out}
					<span
						class="chip bad"
						title="The thresholds were fitted over the same months they score here — a small leak, and one the tally inherits."
					>
						thresholds in sample
					</span>
				{/if}
			{/if}
			{#if review.serverHealth === null}
				<span class="chip bad" title="Judgements cannot be stored until the server answers.">
					annotation server not answering
				</span>
			{:else}
				<span class="quiet">
					{review.serverHealth.annotated} judged of {review.serverHealth.events}
				</span>
			{/if}
			<button type="button" class="chip" onclick={() => (helpOpen = !helpOpen)}>keys ?</button>
			<button
				type="button"
				class="chip"
				onclick={() =>
					void review.exportRows('parquet').then((path) => {
						if (path !== null) alert(`Exported to ${path}`);
					})}>export</button
			>
		</header>

		{#if provenance.contaminated || loudCaveats.length > 0 || review.vocabularyIsFallback}
			<div class="banner">
				{#if provenance.contaminated}
					<p><strong>Probabilities: {provenance.label}.</strong> {provenance.reading}</p>
				{/if}
				{#each loudCaveats as caveat (caveat.code)}
					<p><strong>{caveat.code}</strong> · {caveat.detail}</p>
				{/each}
				{#if review.vocabularyIsFallback}
					<p>
						<strong>Fallback vocabulary.</strong> The bundle carries no readable
						<code>tags.json</code>; tagging is running on this page's compiled-in codes.
					</p>
				{/if}
			</div>
		{/if}

		{#if caveats.length > 0}
			<details class="caveats">
				<summary>{caveats.length} caveat(s) that change how this bundle reads</summary>
				<ul>
					{#each caveats as caveat (caveat.code)}
						<li>
							<span class="severity {caveat.severity}">{caveat.severity}</span>
							<strong>{caveat.code}</strong> — {caveat.detail}
						</li>
					{/each}
				</ul>
			</details>
		{/if}

		<div class="body">
			<div class="pane list-pane">
				<EventList select={openEvent} />
			</div>

			<div class="pane centre">
				{#if selected === null}
					<div class="empty">
						<p>Pick an event. <kbd>j</kbd> / <kbd>k</kbd> walk the list.</p>
					</div>
				{:else}
					<div class="event-head">
						<span class="badge {shownClass ?? 'withheld'}">{shownClass ?? 'class withheld'}</span>
						<strong>{selected.station_name}</strong>
						<span class="quiet">{selected.station_id} · {selected.region}</span>
						<span class="quiet" title={localStamp(selected.anchor_utc) ?? ''}>
							anchor {utcStamp(selected.anchor_utc)}
						</span>
						<span class="quiet">
							{selected.season} · {selected.intensity_band}
							{#if selected.intensity_mm_h !== null}
								({selected.intensity_mm_h} mm/h, from {selected.intensity_band_source})
							{/if}
						</span>
						{#if selected.p_decision !== null}
							<span class="quiet">
								p {pctText(selected.p_decision)}{#if selected.threshold_pct !== null}
									/ {selected.threshold_pct} %{/if}
							</span>
						{/if}
						{#if selected.lead_error_min !== null}
							<span class="quiet" title="Positive means the rain arrived sooner than the ETA said — the warning was late.">
								lead error {minutesText(selected.lead_error_min)}
							</span>
						{/if}
						<span class="quiet">
							armed at anchor: {selected.arm_state_at_anchor}
							{#if selected.minutes_to_rearm_at_anchor !== null}
								({minutesText(selected.minutes_to_rearm_at_anchor)} to re-arm)
							{/if}
						</span>
						{#if selected.frames_missing > 0}
							<span class="chip bad">{selected.frames_missing} frame(s) missing</span>
						{/if}
						{#each selected.flags as flag (flag)}
							<span class="chip">{flag}</span>
						{/each}
					</div>

					{#if review.detailStatus === 'loading'}
						<p class="quiet loading">Loading the event…</p>
					{/if}
					{#each review.detail?.builder_notes ?? [] as note, i (i)}
						<p class="quiet note">builder: {note}</p>
					{/each}

					<ReviewMap />
					<Scrubber />
				{/if}
			</div>

			<div class="pane right">
				{#if selected === null}
					<p class="quiet empty-right">
						No event open. The panels here show what the service had at the cursor, what
						the gauges said, and the judgement.
					</p>
				{:else}
					<EstimatePanel />
					<TruthPanel />
					<FeaturePanel />
					<AnnotationForm {groups} />
				{/if}
			</div>
		</div>

		<!-- The save state, where it is visible from any pane: the form is at
		     the bottom of a scrolling column and `s` can be pressed from
		     anywhere. -->
		{#if review.saveStatus !== 'idle' || review.dirty}
			<div
				class="savebar"
				class:error={review.saveStatus === 'error'}
				class:dirty={review.dirty}
				role="status"
			>
				{#if review.saveStatus === 'error'}
					could not save: {review.saveError?.message ?? NOT_MEASURED}
				{:else if review.saveStatus === 'saving'}
					saving…
				{:else if review.dirty}
					unsaved judgement — press s
				{:else}
					saved
				{/if}
			</div>
		{/if}

		{#if helpOpen}
			<div class="help" role="dialog" aria-label="Keyboard">
				<h2>Keyboard</h2>
				<dl>
					<dt><kbd>j</kbd> <kbd>k</kbd></dt>
					<dd>next / previous event in the list as it is filtered and sorted</dd>
					<dt><kbd>←</kbd> <kbd>→</kbd></dt>
					<dd>
						one composite back / forward. With the scrubber focused the input's own step
						takes over and moves the cursor a minute at a time.
					</dd>
					<dt><kbd>space</kbd></dt>
					<dd>play / pause the loop</dd>
					<dt><kbd>1</kbd>…<kbd>9</kbd></dt>
					<dd>toggle the first nine cause tags for this event's class</dd>
					<dt><kbd>s</kbd></dt>
					<dd>save the judgement</dd>
					<dt><kbd>?</kbd></dt>
					<dd>this panel · <kbd>esc</kbd> closes it</dd>
				</dl>
				<p class="quiet">
					Shortcuts are off while a text field has focus, and the arrows and space bar
					are left to whatever control has focus.
				</p>
				<button type="button" onclick={() => (helpOpen = false)}>Close</button>
			</div>
		{/if}
	{/if}
</div>

<style>
	/*
	 * Shared tokens for the whole tool. Declared on the root element rather
	 * than in `app.css`: custom properties inherit through the DOM, so every
	 * child component's scoped styles can use them, and nothing leaks into the
	 * public site's stylesheet.
	 */
	.review {
		--wet: #2f7fd6;
		--unknown: #8a6ea8;
		position: fixed;
		inset: 0;
		z-index: 5;
		display: flex;
		flex-direction: column;
		background: var(--bg);
		color: var(--ink);
		font-size: 0.85rem;
		overflow: hidden;
	}

	.topbar {
		flex: 0 0 auto;
		display: flex;
		align-items: center;
		gap: 0.5rem;
		flex-wrap: wrap;
		padding: 0.3rem 0.6rem;
		background: var(--surface);
		border-bottom: 1px solid var(--border);
		font-size: 0.75rem;
	}

	.title {
		font-weight: 650;
	}

	.quiet {
		color: var(--muted);
	}

	.chip {
		border: 1px solid var(--border);
		background: var(--bg);
		color: var(--ink);
		border-radius: 999px;
		padding: 0 0.4rem;
		font-size: 0.7rem;
		cursor: default;
	}

	button.chip {
		cursor: pointer;
	}

	.chip.bad {
		color: var(--warn);
		border-color: var(--warn);
	}

	/* Loud, and at the top: a contaminated probability or a high-severity
	   caveat changes what every tag in the export means. */
	.banner {
		flex: 0 0 auto;
		padding: 0.3rem 0.6rem;
		background: color-mix(in srgb, var(--warn) 12%, var(--surface));
		border-bottom: 1px solid var(--warn);
		font-size: 0.75rem;
	}

	.banner p {
		margin: 0 0 0.15rem;
	}

	.caveats {
		flex: 0 0 auto;
		padding: 0.15rem 0.6rem;
		background: var(--surface);
		border-bottom: 1px solid var(--border);
		font-size: 0.73rem;
	}

	.caveats summary {
		cursor: pointer;
		color: var(--muted);
	}

	.caveats ul {
		margin: 0.2rem 0;
		padding-left: 1rem;
	}

	.severity {
		font-size: 0.64rem;
		text-transform: uppercase;
		letter-spacing: 0.03em;
		border: 1px solid var(--border);
		border-radius: 3px;
		padding: 0 0.2rem;
		color: var(--muted);
	}

	.severity.high {
		color: var(--warn);
		border-color: var(--warn);
	}

	.body {
		flex: 1 1 auto;
		min-height: 0;
		display: grid;
		grid-template-columns: 21rem minmax(0, 1fr) 26rem;
	}

	.pane {
		min-width: 0;
		min-height: 0;
	}

	.centre {
		display: flex;
		flex-direction: column;
		min-height: 0;
		background: var(--bg);
	}

	.right {
		overflow-y: auto;
		background: var(--surface);
		border-left: 1px solid var(--border);
	}

	.empty,
	.empty-right {
		padding: 1rem;
		color: var(--muted);
	}

	.event-head {
		flex: 0 0 auto;
		display: flex;
		align-items: baseline;
		flex-wrap: wrap;
		gap: 0.4rem;
		padding: 0.3rem 0.6rem;
		background: var(--surface);
		border-bottom: 1px solid var(--border);
		font-size: 0.76rem;
	}

	.badge {
		font-size: 0.66rem;
		text-transform: uppercase;
		letter-spacing: 0.03em;
		border: 1px solid var(--border);
		border-radius: 4px;
		padding: 0 0.3rem;
		color: var(--muted);
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

	/* Withheld: the reviewer learns that the class is hidden, never what it
	   is. Borrowing a plausible class would be a lie. */
	.badge.withheld {
		background-image: repeating-linear-gradient(-45deg, var(--muted) 0 2px, transparent 2px 5px);
	}

	.loading,
	.note {
		margin: 0;
		padding: 0.15rem 0.6rem;
		font-size: 0.72rem;
	}

	.savebar {
		position: absolute;
		right: 0.8rem;
		bottom: 0.8rem;
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: 999px;
		padding: 0.15rem 0.7rem;
		font-size: 0.74rem;
		box-shadow: var(--shadow);
	}

	.savebar.dirty {
		border-color: var(--accent);
		color: var(--accent);
	}

	.savebar.error {
		border-color: var(--warn);
		color: var(--warn);
	}

	.help {
		position: absolute;
		left: 50%;
		top: 20%;
		transform: translateX(-50%);
		width: min(34rem, 90vw);
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		box-shadow: var(--shadow);
		padding: 0.8rem 1rem;
	}

	.help h2 {
		margin: 0 0 0.4rem;
		font-size: 1rem;
	}

	.help dl {
		display: grid;
		grid-template-columns: 7rem 1fr;
		gap: 0.15rem 0.6rem;
		margin: 0 0 0.5rem;
		font-size: 0.78rem;
	}

	.help dt {
		color: var(--muted);
	}

	.help dd {
		margin: 0;
	}

	.help button {
		border: 1px solid var(--border);
		background: var(--bg);
		color: var(--ink);
		border-radius: 8px;
		padding: 0.2rem 0.7rem;
		cursor: pointer;
	}

	kbd {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.7rem;
		border: 1px solid var(--border);
		border-radius: 3px;
		padding: 0 0.22rem;
		background: var(--bg);
	}

	code {
		font-size: 0.72rem;
	}

	@media (max-width: 80rem) {
		.body {
			grid-template-columns: 17rem minmax(0, 1fr) 21rem;
		}
	}
</style>
