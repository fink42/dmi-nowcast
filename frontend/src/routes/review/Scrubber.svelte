<script lang="ts">
	/**
	 * The track: one event's window in ABSOLUTE time.
	 *
	 * This is where the review loop parts company with the public site's
	 * scrubber. There the track is spaced by frame index, because the range
	 * input *is* the track and a tick has to sit where dragging to it lands.
	 * Here the single most important thing the track has to show is a
	 * **hole** — a coverage gap, a run of missing composites, the edge past
	 * which the gauge has not reported — and index spacing draws a 40-minute
	 * gap the same width as a 10-minute step. So every position comes from
	 * `review/timeline.ts`'s `(t − from) / span`, and the range input carries
	 * milliseconds rather than an index.
	 *
	 * The decorated-scrubber idiom is `LoopControls.svelte`'s: a native
	 * `input[type=range]` owns the keyboard, the touch behaviour and the a11y
	 * semantics, and everything drawn is decoration positioned in the same
	 * inset coordinate space as its thumb.
	 *
	 * Both stamps are printed, always. The composite on screen and the
	 * estimate beside it come from cycles 13–18 minutes apart: printing only
	 * the service's frame flatters every false alarm, and printing only the
	 * truth frame shows rain the service could not have known about.
	 */
	import { review } from '$lib/review/store.svelte';
	import type { TrackMarker } from '$lib/review/timeline';
	import { minutesText, utcTime, utcClock } from './format';
	import { armSegments } from './track';

	const bounds = $derived(review.bounds);
	const ticks = $derived(review.ticks);
	const marks = $derived(review.trackMarkers);
	const cursor = $derived(review.cursorFrame);
	const frames = $derived(review.detail?.frames ?? []);
	const arm = $derived(
		armSegments(
			review.detail?.decisions ?? [],
			review.detail?.prologue ?? null,
			bounds,
			review.manifest?.rule?.rearm_after_min
		)
	);

	/** Track coordinates: 0 … 1 along the bar, which the thumb also travels. */
	const pct = (position: number) => `${(position * 100).toFixed(3)}%`;

	/** The marker's own sentence, for its tooltip. */
	function markerTitle(marker: TrackMarker): string {
		const kind = marker.kind.replace(/_/g, ' ');
		const own = marker.isEventWarning ? " — this event's own warning" : '';
		return `${kind}${own} · ${utcTime(marker.utc)}${marker.detail === null ? '' : ` · ${marker.detail}`}`;
	}

	/** The instants worth a button: what a reviewer scrubs back to. */
	const jumps = $derived(
		marks.points.filter(
			(point) =>
				point.kind === 'anchor' ||
				point.kind === 'onset' ||
				point.kind === 'run_boundary_rearm' ||
				(point.kind === 'warning' && point.isEventWarning)
		)
	);
</script>

<div class="scrubber">
	{#if bounds === null}
		<p class="quiet">This event carries no usable instants, so there is no track to draw.</p>
	{:else}
		<div class="head">
			<button
				class="play"
				type="button"
				onclick={() => review.toggle()}
				aria-label={review.playing ? 'Pause' : 'Play'}
			>
				{#if review.playing}
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
			<button type="button" class="step" onclick={() => review.step(-1)} aria-label="Previous frame"
				>←</button
			>
			<button type="button" class="step" onclick={() => review.step(1)} aria-label="Next frame"
				>→</button
			>

			<!-- Both stamps, always. "13:20 picture, 13:35 estimate, from the
			     13:20 frame" is the only phrasing that is not misleading; one
			     stamp alone makes every false alarm look better or worse than
			     it was. -->
			<p class="stamps">
				<span class="cursor-time"
					>cursor {utcTime(
						review.cursorMs === null ? null : new Date(review.cursorMs).toISOString()
					)}</span
				>
				<span>
					picture <strong class:pending={review.buffering}
						>{utcClock(cursor?.truthTsUtc) ?? '—'}</strong
					>
					{#if review.buffering}<span class="tag">decoding</span>{/if}
				</span>
				<span>
					estimate issued <strong>{utcClock(cursor?.decision?.generated_at_utc) ?? '—'}</strong>,
					from the <strong>{utcClock(cursor?.serviceTsUtc) ?? '—'}</strong> frame
				</span>
				{#if cursor?.serviceLagMin !== null && cursor?.serviceLagMin !== undefined}
					<span class="lag">service {minutesText(cursor.serviceLagMin)} behind the picture</span>
				{/if}
				<span class="quiet">
					frame {review.frameIndex + 1} / {frames.length}
				</span>
			</p>

			<div class="modes" role="group" aria-label="Which frame the map shows">
				<button
					type="button"
					class:on={review.mode === 'truth'}
					title="The newest composite at or before the cursor — what was actually happening."
					onclick={() => review.setMode('truth')}>Truth</button
				>
				<button
					type="button"
					class:on={review.mode === 'service'}
					title="The composite the current estimate stood on — 13–18 minutes older."
					onclick={() => review.setMode('service')}>Service</button
				>
			</div>
		</div>

		<div class="rail">
			<!-- Decoration only; the range input below owns the interaction. -->
			<div class="bar" aria-hidden="true"></div>

			<!-- Keyed by index, here and below. A bundle can carry two markers on
			     the same instant — a replayed notification and the stored one it
			     came from, two coverage gaps meeting end to end, a doppler frame
			     beside a fullRange frame on the same stamp — and a keyed `each`
			     throws on a duplicate key. None of this decoration reorders, so
			     the index is the honest key. -->
			<div class="bands" aria-hidden="true">
				{#each marks.bands as band, i (i)}
					<!-- A coverage BREAK is louder than a missed cycle: past that
					     length the coverage rule stops counting and the replay hands
					     out a re-arm the live service never had. -->
					<span
						class="band {band.kind}"
						class:break={band.coverageBreak}
						style:left={pct(band.from)}
						style:width={pct(band.to - band.from)}
						title="{band.kind.replace(/_/g, ' ')}{band.coverageBreak
							? ' (coverage break)'
							: ''} · {Math.round(band.minutes)} min{band.edge === null
							? ''
							: ` · ${band.edge}`} · {band.reason}"
					></span>
				{/each}
			</div>

			<div class="marks" aria-hidden="true">
				{#each ticks as tick, i (i)}
					<!-- Two different holes, drawn differently: no composite (the
					     builder could not write it) and a composite with no decision
					     standing on it. One is a hole in the imagery, the other a
					     hole in the engine's attention, and they carry different
					     tags. -->
					<span
						class="tick"
						class:missing={tick.present === false}
						class:no-decision={tick.present !== false && tick.hasDecisionRow === false}
						style:left={pct(tick.position)}
						title="{tick.stamp} · {utcTime(tick.radarTsUtc)}{tick.present === false
							? ' · the builder could not write this composite'
							: ''}{tick.hasDecisionRow === false
							? ' · no decision row stood on this composite'
							: ''}"
					></span>
				{/each}
				{#each marks.points as point, i (i)}
					<span
						class="point {point.kind}"
						class:own={point.isEventWarning}
						style:left={pct(point.position)}
						title={markerTitle(point)}
					></span>
				{/each}
				{#if review.cursorPosition !== null}
					<span class="playhead" style:left={pct(review.cursorPosition)}></span>
				{/if}
			</div>

			<input
				type="range"
				min={bounds.fromMs}
				max={bounds.toMs}
				step={60000}
				value={review.cursorMs ?? bounds.fromMs}
				aria-label="Cursor, in absolute time"
				aria-valuetext="{utcTime(
					review.cursorMs === null ? null : new Date(review.cursorMs).toISOString()
				)} · picture {utcClock(cursor?.truthTsUtc) ?? 'none'} · estimate {utcClock(
					cursor?.decision?.generated_at_utc
				) ?? 'none'}"
				oninput={(event) => review.setCursor(Number(event.currentTarget.value))}
			/>
		</div>

		<!-- The arm band. A disarmed stretch is one where no notification could
		     have been sent at all, whatever the probability did — which is the
		     whole of `miss_disarmed_rearm`. Coverage-gap hatching is drawn over
		     the top of it, so a stretch with no evidence under it still reads
		     as one. -->
		<div class="armband" aria-hidden="true">
			{#each arm as segment, i (i)}
				<span
					class="arm"
					class:disarmed={!segment.armed}
					class:assumed={segment.source !== 'decision'}
					style:left={pct(segment.from)}
					style:width={pct(segment.to - segment.from)}
					title="{segment.armed ? 'armed' : 'disarmed'} · {utcTime(segment.fromUtc)} → {utcTime(
						segment.toUtc
					)} · streak {segment.streak} · from the {segment.source}"
				></span>
			{/each}
			{#each marks.bands as band, i (i)}
				<span
					class="band {band.kind}"
					style:left={pct(band.from)}
					style:width={pct(band.to - band.from)}
				></span>
			{/each}
		</div>

		<div class="labels">
			<span class="edge">{utcTime(bounds.fromUtc)}</span>
			<span class="jump">
				{#each jumps as point, i (i)}
					<button
						type="button"
						class="jump-button {point.kind}"
						title={markerTitle(point)}
						onclick={() => review.setCursor(point.ms)}
					>
						{point.kind === 'run_boundary_rearm' ? 'replay re-arm' : point.kind}
						{utcClock(point.utc)}
					</button>
				{/each}
			</span>
			<span class="edge">{utcTime(bounds.toUtc)}</span>
		</div>

		<p class="legend">
			<span><i class="key anchor"></i>anchor</span>
			<span><i class="key onset"></i>gauge onset</span>
			<span><i class="key warning"></i>warning</span>
			<span><i class="key already_raining"></i>already raining</span>
			<span><i class="key run_boundary_rearm"></i>replay re-arm</span>
			<span><i class="key known_until"></i>known_until</span>
			<span><i class="key nodecision"></i>composite with no decision on it</span>
			<span><i class="key hatch"></i>coverage gap / past known_until — no evidence, not "dry"</span>
		</p>
	{/if}
</div>

<style>
	/* Half the thumb: the inset that makes the bar, the ticks and the thumb
	   share one coordinate space, so a tick sits exactly where dragging to it
	   lands. */
	.scrubber {
		--pad: 0.6rem;
		flex: 0 0 auto;
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		padding: 0.4rem 0.6rem 0.5rem;
		background: var(--surface);
		border-top: 1px solid var(--border);
	}

	.head {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		flex-wrap: wrap;
	}

	.play {
		flex: 0 0 auto;
		width: 2rem;
		height: 2rem;
		border-radius: 50%;
		border: none;
		background: var(--accent);
		color: var(--accent-ink);
		display: grid;
		place-items: center;
		cursor: pointer;
	}

	.play svg {
		width: 1rem;
		height: 1rem;
		fill: currentColor;
	}

	.step {
		border: 1px solid var(--border);
		background: var(--bg);
		color: var(--ink);
		border-radius: 6px;
		padding: 0.1rem 0.4rem;
		cursor: pointer;
	}

	.stamps {
		flex: 1 1 auto;
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 0.1rem 0.7rem;
		font-size: 0.76rem;
		min-width: 0;
	}

	.stamps strong {
		font-variant-numeric: tabular-nums;
	}

	.stamps strong.pending {
		opacity: 0.55;
	}

	.cursor-time {
		font-weight: 600;
	}

	.lag {
		color: var(--warn);
	}

	.tag {
		font-size: 0.68rem;
		color: var(--muted);
		border: 1px solid var(--border);
		border-radius: 999px;
		padding: 0 0.35rem;
	}

	.modes {
		display: flex;
		gap: 0.2rem;
	}

	.modes button {
		border: 1px solid var(--border);
		background: var(--bg);
		color: var(--ink);
		font-size: 0.72rem;
		border-radius: 6px;
		padding: 0.1rem 0.45rem;
		cursor: pointer;
	}

	.modes button.on {
		background: var(--accent);
		color: var(--accent-ink);
		border-color: var(--accent);
	}

	.rail {
		position: relative;
		height: 2rem;
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
	}

	.bands {
		position: absolute;
		left: var(--pad);
		right: var(--pad);
		top: 50%;
		transform: translateY(-50%);
		height: 0.5rem;
		border-radius: 999px;
		overflow: hidden;
	}

	/* Absence of evidence, drawn as absence of evidence. Hatching survives
	   both themes and colour blindness, which a grey fill does not. */
	.band {
		position: absolute;
		top: 0;
		bottom: 0;
		background-image: repeating-linear-gradient(-45deg, var(--muted) 0 2px, transparent 2px 5px);
		opacity: 0.75;
	}

	.band.beyond_known {
		background-image: repeating-linear-gradient(45deg, var(--warn) 0 2px, transparent 2px 5px);
	}

	.band.break {
		opacity: 1;
	}

	.marks {
		position: absolute;
		left: var(--pad);
		right: var(--pad);
		top: 50%;
		height: 0;
	}

	.tick {
		position: absolute;
		top: 0;
		width: 4px;
		height: 4px;
		border-radius: 50%;
		background: var(--muted);
		box-shadow: 0 0 0 1.5px var(--surface);
		transform: translate(-50%, -50%);
	}

	/* A composite the builder could not write: hollow, because the hole is
	   evidence and closing it would hide it. */
	.tick.missing {
		background: var(--surface);
		box-shadow: 0 0 0 1px var(--warn);
	}

	/* The picture exists; no decision stood on it. A square, so the two holes
	   are told apart by shape and not only by colour. */
	.tick.no-decision {
		border-radius: 0;
		background: var(--warn);
	}

	.point {
		position: absolute;
		top: 0;
		width: 2px;
		height: 1.1rem;
		transform: translate(-50%, -50%);
		background: var(--ink);
	}

	.point.anchor {
		height: 1.5rem;
		width: 3px;
	}

	.point.onset {
		background: #2f7fd6;
		height: 1.3rem;
	}

	.point.warning {
		background: var(--accent);
	}

	.point.warning.own {
		width: 4px;
		height: 1.5rem;
	}

	.point.already_raining {
		background: var(--muted);
	}

	.point.run_boundary_rearm {
		background: var(--warn);
		height: 1.4rem;
	}

	.point.known_until {
		background: var(--warn);
		width: 1px;
	}

	.playhead {
		position: absolute;
		top: 0;
		width: 1.5px;
		height: 1.7rem;
		background: var(--accent);
		transform: translate(-50%, -50%);
	}

	/* Transparent and the full height of the rail: the decoration underneath
	   is what you see, this is what you touch. */
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

	.armband {
		position: relative;
		height: 0.55rem;
		margin: 0 var(--pad);
		border-radius: 3px;
		overflow: hidden;
		background: var(--bg);
	}

	.arm {
		position: absolute;
		top: 0;
		bottom: 0;
		background: color-mix(in srgb, var(--accent) 35%, transparent);
	}

	/* Disarmed: hatched in the warning colour, because "no push was possible
	   here" is a fact about the rule and not about the weather. */
	.arm.disarmed {
		background-image: repeating-linear-gradient(90deg, var(--warn) 0 3px, transparent 3px 6px);
		background-color: transparent;
	}

	/* State the trace did not witness — the prologue's answer, or none at
	   all. Faded, so it is not read as a traced fact. */
	.arm.assumed {
		opacity: 0.45;
	}

	.labels {
		display: flex;
		align-items: baseline;
		gap: 0.4rem;
		margin: 0 var(--pad);
		font-size: 0.68rem;
		color: var(--muted);
	}

	.jump {
		flex: 1 1 auto;
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem;
		justify-content: center;
	}

	.jump-button {
		border: 1px solid var(--border);
		background: var(--bg);
		color: var(--ink);
		border-radius: 999px;
		font-size: 0.66rem;
		padding: 0 0.35rem;
		cursor: pointer;
	}

	.jump-button.run_boundary_rearm {
		color: var(--warn);
		border-color: var(--warn);
	}

	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
		margin: 0.1rem var(--pad) 0;
		font-size: 0.66rem;
		color: var(--muted);
	}

	.key {
		display: inline-block;
		width: 3px;
		height: 0.7rem;
		margin-right: 0.25rem;
		vertical-align: -2px;
		background: var(--ink);
	}

	.key.onset {
		background: #2f7fd6;
	}

	.key.warning {
		background: var(--accent);
	}

	.key.already_raining {
		background: var(--muted);
	}

	.key.run_boundary_rearm,
	.key.known_until {
		background: var(--warn);
	}

	.key.nodecision {
		width: 0.4rem;
		height: 0.4rem;
		background: var(--warn);
	}

	.key.hatch {
		width: 1.4rem;
		background-image: repeating-linear-gradient(-45deg, var(--muted) 0 2px, transparent 2px 5px);
		background-color: transparent;
	}

	.quiet {
		margin: 0;
		color: var(--muted);
		font-size: 0.78rem;
	}
</style>
