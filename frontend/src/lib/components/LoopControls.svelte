<script lang="ts">
	/**
	 * The timeline: one track carrying the whole loop, docked to the bottom
	 * edge of the map.
	 *
	 * The past half is hatched and the future half is plain, split at the
	 * latest observation, because the single thing a viewer must never get
	 * wrong is which side of that hinge they are looking at — a forecast frame
	 * read as a measurement is a lie the styling told. The active frame's own
	 * clock time and state word are spelled out above the track, and they are
	 * the *only* place the site states a frame time.
	 *
	 * Two marks sit on the track and they are different instants. The hinge is
	 * the newest radar image, which is 14–24 min old whenever anyone looks:
	 * DMI's delay, a 10 min composite cadence, and a cycle that serves for up
	 * to 10 min. Wall-clock "now" is therefore some way to the right of it,
	 * among the forecast frames, and it gets its own moving marker — labelling
	 * the newest observation "Now" while the panel counted from the wall clock
	 * put two different "now"s on one screen.
	 *
	 * Underneath it stays a native `input[type=range]`: it owns the keyboard,
	 * the touch behaviour and the a11y semantics, and everything drawn here is
	 * decoration positioned in the same coordinate space as its thumb.
	 *
	 * The radar-age line below is R4's and keeps R4's semantics: the age is
	 * shown always and flagged rarely, and the loud line is pipeline liveness
	 * (see nowcast/freshness.ts), never the age itself.
	 */
	import { t, locale } from '$lib/i18n';
	import { clockTime } from '$lib/format';
	import { nowcast } from '$lib/nowcast/store.svelte';
	import { clockPosition, timelineGeometry, type TimelineFrame } from '$lib/nowcast/timeline';

	const frames = $derived(nowcast.timeline);
	const count = $derived(frames.length);
	const index = $derived(nowcast.frameIndex);
	const active = $derived(nowcast.activeFrame);
	const loaded = $derived(nowcast.loadedCount);
	const buffering = $derived(nowcast.buffering);
	const geometry = $derived(timelineGeometry(frames));
	const fresh = $derived(nowcast.freshness);
	const ageMin = $derived(fresh.radarAgeMin);

	/** Track coordinates: 0 … 1 along the bar, which the thumb also travels. */
	const pct = (position: number) => `${(position * 100).toFixed(2)}%`;

	const stateWord = (frame: TimelineFrame) =>
		frame.isLatest
			? t().loop.stateNow
			: frame.kind === 'observation'
				? t().loop.stateObserved
				: t().loop.stateForecast;

	/** Frame time, always from `valid_ts_utc`, in the viewer's own zone. */
	const frameTime = (frame: TimelineFrame) => clockTime(frame.validTsUtc, locale());

	/**
	 * The end labels say where a frame sits relative to the radar image, so
	 * lead 0 is "Latest", never "Now": the radar image is not now, and "Now"
	 * belongs to the clock marker alone.
	 */
	const edgeLabel = (frame: TimelineFrame | undefined) =>
		!frame
			? ''
			: frame.leadMin < 0
				? t().loop.lag(-frame.leadMin)
				: frame.leadMin === 0
					? t().loop.latest
					: t().loop.lead(frame.leadMin);

	// The screen-reader value of the slider: the same three facts the sighted
	// label carries, plus the position in the loop.
	const valueText = $derived(
		active
			? `${frameTime(active)} · ${stateWord(active)} · ${t().loop.frameOf(index + 1, count)}`
			: t().loop.noFrames
	);

	/** The hinge: the newest radar image, where hatched past meets forecast. */
	const latestAt = $derived(geometry.latestPosition);
	/**
	 * Wall-clock now, interpolated between the frames bracketing it. The store
	 * re-stamps `now` every 15 s and whenever the tab becomes visible, so the
	 * marker walks the track on its own — no timer of this component's.
	 */
	const clockAt = $derived(clockPosition(frames, nowcast.now));

	// Both ends carry a label; either is dropped when a marker's label is
	// sitting close enough to collide with it (a cold-start manifest has no
	// history at all, which puts the hinge exactly on the left edge).
	const clearOfEdge = (position: number | null, edge: number) =>
		position === null || Math.abs(position - edge) > 0.14;
	const showStart = $derived(count > 1 && clearOfEdge(latestAt, 0) && clearOfEdge(clockAt, 0));
	const showEnd = $derived(count > 1 && clearOfEdge(latestAt, 1) && clearOfEdge(clockAt, 1));
	/**
	 * "Now" is the label that must always be readable, so when the two land on
	 * top of each other it is "Latest" that gives way — its tick stays, since
	 * the hatching already says which side of it is measured.
	 */
	const showLatestLabel = $derived(
		latestAt !== null && (clockAt === null || Math.abs(clockAt - latestAt) >= 0.08)
	);
</script>

<div class="timeline" class:empty={count === 0}>
	{#if count === 0}
		<p class="muted">{t().loop.noFrames}</p>
	{:else}
		<div class="head">
			<button
				class="play"
				type="button"
				onclick={() => nowcast.togglePlay()}
				aria-label={nowcast.playing ? t().loop.pause : t().loop.play}
			>
				{#if nowcast.playing}
					<svg viewBox="0 0 24 24" aria-hidden="true"
						><rect x="6" y="5" width="4" height="14" rx="1" /><rect
							x="14"
							y="5"
							width="4"
							height="14"
							rx="1"
						/></svg
					>
				{:else}
					<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4l12 8-12 8z" /></svg>
				{/if}
			</button>

			<!-- No aria-live here: at 550 ms a frame it would talk over
			     everything. The slider's aria-valuetext carries the same
			     three facts, announced when the viewer actually moves it. -->
			<p class="stamp">
				<strong class:pending={buffering}>{active ? frameTime(active) : ''}</strong>
				<span class="state" class:forecast={active?.kind === 'forecast'}>
					{active ? stateWord(active) : ''}
				</span>
				{#if buffering}
					<span class="loading">{t().loop.buffering}</span>
				{/if}
			</p>
		</div>

		<div class="rail">
			<!-- Decoration only; the range input below owns the interaction. -->
			<!-- The hatched region ends at the latest observation: everything to
			     its right is extrapolation, including the frames the clock marker
			     has already walked past. -->
			<div class="bar" aria-hidden="true">
				{#if latestAt !== null && latestAt > 0}
					<div class="past" style:width={pct(latestAt)}></div>
				{/if}
			</div>

			<div class="marks" aria-hidden="true">
				{#each frames as frame, i (frame.filename)}
					<span
						class="tick"
						class:pending={i >= loaded}
						style:left={pct(geometry.positions[i])}
					></span>
				{/each}
				{#if latestAt !== null}
					<span class="latest" style:left={pct(latestAt)}></span>
				{/if}
				{#if clockAt !== null}
					<span class="clock" style:left={pct(clockAt)}></span>
				{/if}
			</div>

			<input
				type="range"
				min="0"
				max={count - 1}
				step="1"
				value={index}
				aria-label={t().loop.scrubber}
				aria-valuetext={valueText}
				oninput={(e) => nowcast.seek(Number(e.currentTarget.value))}
			/>
		</div>

		<div class="labels" aria-hidden="true">
			{#if showStart}
				<span class="edge start">{edgeLabel(frames[0])}</span>
			{/if}
			{#if latestAt !== null && showLatestLabel}
				<!-- Centred on the marker, except at the ends of the track — a
				     cold-start manifest puts the hinge on the left edge. -->
				<span
					class="edge at-latest"
					class:flush-start={latestAt <= 0.02}
					class:flush-end={latestAt >= 0.98}
					style:left={pct(latestAt)}>{t().loop.latest}</span
				>
			{/if}
			{#if clockAt !== null}
				<span
					class="edge at-clock"
					class:flush-start={clockAt <= 0.02}
					class:flush-end={clockAt >= 0.98}
					style:left={pct(clockAt)}>{t().loop.now}</span
				>
			{/if}
			{#if showEnd}
				<span class="edge end">{edgeLabel(frames[count - 1])}</span>
			{/if}
		</div>

		{#if ageMin !== null}
			<p class="meta">
				<span class:warn={fresh.state !== 'ok'}>{t().loop.radarAge(Math.round(ageMin))}</span>
			</p>
		{/if}
		{#if fresh.state === 'pipeline-stale'}
			<p class="warn small alert">{t().loop.pipelineStale}</p>
		{:else if fresh.state === 'radar-old'}
			<p class="warn small">{t().loop.radarOld}</p>
		{/if}
	{/if}
</div>

<style>
	/* Half the thumb: the inset that makes the bar, the ticks and the thumb
	   share one coordinate space, so a tick sits exactly where dragging to it
	   lands. */
	.timeline {
		--pad: 0.6rem;
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		padding: 0.5rem 0.7rem 0.55rem;
		background: var(--surface);
		border-radius: var(--radius) var(--radius) 0 0;
		box-shadow: var(--shadow);
	}

	.timeline.empty {
		padding: 0.9rem;
	}

	.head {
		display: flex;
		align-items: center;
		gap: 0.6rem;
		flex-wrap: wrap;
	}

	.play {
		flex: 0 0 auto;
		width: 2.5rem;
		height: 2.5rem;
		border-radius: 50%;
		border: none;
		background: var(--accent);
		color: var(--accent-ink);
		display: grid;
		place-items: center;
		cursor: pointer;
	}

	.play svg {
		width: 1.2rem;
		height: 1.2rem;
		fill: currentColor;
	}

	.stamp {
		flex: 1 1 auto;
		margin: 0;
		display: flex;
		align-items: baseline;
		gap: 0.4rem;
		flex-wrap: wrap;
		min-width: 0;
	}

	.stamp strong {
		font-size: 1.25rem;
		font-weight: 650;
		line-height: 1.1;
		font-variant-numeric: tabular-nums;
	}

	/* A time whose image has not arrived is still the honest answer to "what
	   am I looking at next" — it is just not on the map yet. */
	.stamp strong.pending {
		opacity: 0.55;
	}

	.state {
		font-size: 0.82rem;
		color: var(--muted);
	}

	.state.forecast {
		color: var(--accent);
	}

	.loading {
		font-size: 0.72rem;
		color: var(--muted);
		border: 1px solid var(--border);
		border-radius: 999px;
		padding: 0 0.45rem;
	}

	.rail {
		position: relative;
		height: 2.1rem;
	}

	.bar {
		position: absolute;
		left: var(--pad);
		right: var(--pad);
		top: 50%;
		transform: translateY(-50%);
		height: 0.5rem;
		border-radius: 999px;
		background: var(--track);
		overflow: hidden;
	}

	/* Observed history: hatched, so "already happened" is legible without
	   colour and survives both themes. */
	.past {
		position: absolute;
		inset: 0 auto 0 0;
		background-image: repeating-linear-gradient(
			-45deg,
			var(--muted) 0 2px,
			transparent 2px 5px
		);
		opacity: 0.55;
	}

	/* A zero-height strip on the bar's centre line, inset exactly like the bar:
	   ticks and the "now" marker then share the thumb's coordinate space. */
	.marks {
		position: absolute;
		left: var(--pad);
		right: var(--pad);
		top: 50%;
		height: 0;
	}

	/* The halo is what keeps a tick legible over the hatched past as well as
	   over the plain future track. */
	.tick {
		position: absolute;
		top: 0;
		width: 4px;
		height: 4px;
		border-radius: 50%;
		background: var(--muted);
		box-shadow: 0 0 0 1.5px var(--surface);
		opacity: 0.85;
		transform: translate(-50%, -50%);
	}

	/* Not downloaded yet: hollow, so scrubbing into it is not a surprise. */
	.tick.pending {
		background: var(--surface);
		box-shadow: 0 0 0 1px var(--muted);
		opacity: 0.7;
	}

	/* The hinge: a plain dark bar in the text colour, the end of the hatching. */
	.latest {
		position: absolute;
		top: 0;
		width: 2px;
		height: 1.05rem;
		border-radius: 1px;
		background: var(--ink);
		transform: translate(-50%, -50%);
	}

	/* Wall-clock now: a thinner, brighter line carrying a small triangle, so
	   the two marks are told apart by shape as well as by colour. */
	.clock {
		position: absolute;
		top: 0;
		width: 1.5px;
		height: 1.35rem;
		background: var(--accent);
		transform: translate(-50%, -50%);
	}

	.clock::before {
		content: '';
		position: absolute;
		left: 50%;
		top: -0.16rem;
		width: 0;
		height: 0;
		border-left: 0.28rem solid transparent;
		border-right: 0.28rem solid transparent;
		border-top: 0.34rem solid var(--accent);
		transform: translateX(-50%);
	}

	/* Transparent and the full height of the rail: the decoration underneath
	   is what you see, this is what you touch — a 28 px target on a phone. */
	input[type='range'] {
		position: absolute;
		left: 0;
		right: 0;
		top: 50%;
		transform: translateY(-50%);
		width: 100%;
		height: 1.75rem;
		margin: 0;
		background: transparent;
		-webkit-appearance: none;
		appearance: none;
		cursor: pointer;
	}

	input[type='range']::-webkit-slider-runnable-track {
		height: 1.75rem;
		background: transparent;
	}

	input[type='range']::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 1.2rem;
		height: 1.2rem;
		/* WebKit aligns the thumb's top to the track's: centre it by hand. */
		margin-top: 0.275rem;
		border-radius: 50%;
		background: var(--accent);
		border: 2px solid var(--surface);
		box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35);
	}

	input[type='range']::-moz-range-track {
		height: 1.75rem;
		background: transparent;
		border: none;
	}

	input[type='range']::-moz-range-thumb {
		width: 1.2rem;
		height: 1.2rem;
		border-radius: 50%;
		background: var(--accent);
		border: 2px solid var(--surface);
		box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35);
	}

	.labels {
		position: relative;
		height: 0.9rem;
		margin: 0 var(--pad);
	}

	.edge {
		position: absolute;
		top: 0;
		font-size: 0.62rem;
		line-height: 0.9rem;
		color: var(--muted);
		white-space: nowrap;
	}

	.edge.start {
		left: 0;
	}

	.edge.end {
		right: 0;
	}

	.edge.at-latest,
	.edge.at-clock {
		transform: translateX(-50%);
		font-weight: 600;
	}

	.edge.at-latest {
		color: var(--ink);
	}

	/* Same colour as its marker: the label and the line are one object. */
	.edge.at-clock {
		color: var(--accent);
	}

	.edge.flush-start {
		transform: none;
	}

	.edge.flush-end {
		transform: translateX(-100%);
	}

	.meta {
		margin: 0;
		font-size: 0.74rem;
		color: var(--muted);
	}

	.warn {
		color: var(--warn);
	}

	.small {
		margin: 0;
		font-size: 0.74rem;
	}

	/* The outage reads louder than "the newest image is a bit late": they are
	   different claims and must not look like the same one. */
	.alert {
		font-weight: 600;
	}

	.muted {
		margin: 0;
		color: var(--muted);
		font-size: 0.85rem;
	}
</style>
