<script lang="ts">
	/**
	 * Play/pause plus a scrubber over the loop: "now" and the +lead frames,
	 * with the age of the radar image the whole loop is built from. The age
	 * matters more than it looks — every frame after "now" is extrapolation
	 * from an image that is already a few minutes old.
	 */
	import { t, locale } from '$lib/i18n';
	import { clockTime } from '$lib/format';
	import { nowcast } from '$lib/nowcast/store.svelte';

	const frames = $derived(nowcast.frames);
	const index = $derived(nowcast.frameIndex);
	const current = $derived(frames[index]);
	const ageMin = $derived(nowcast.radarAgeMin);

	const label = (leadMin: number) => (leadMin === 0 ? t().loop.now : t().loop.lead(leadMin));
</script>

<div class="controls" class:empty={frames.length === 0}>
	{#if frames.length === 0}
		<p class="muted">{t().loop.noFrames}</p>
	{:else}
		<div class="row">
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

			<div class="track">
				<input
					type="range"
					min="0"
					max={frames.length - 1}
					step="1"
					value={index}
					aria-label={t().loop.scrubber}
					aria-valuetext={label(current?.leadMin ?? 0)}
					oninput={(e) => nowcast.seek(Number(e.currentTarget.value))}
				/>
				<div class="ticks" aria-hidden="true">
					{#each frames as frame, i (frame.filename)}
						<button
							type="button"
							class="tick"
							class:active={i === index}
							tabindex="-1"
							onclick={() => nowcast.seek(i)}>{label(frame.leadMin)}</button
						>
					{/each}
				</div>
			</div>
		</div>

		<p class="meta">
			<strong>{label(current?.leadMin ?? 0)}</strong>
			{#if nowcast.manifest}
				<span class="sep">·</span>
				<span>{t().loop.radarTime(clockTime(nowcast.manifest.radar_ts_utc, locale()))}</span>
			{/if}
			{#if ageMin !== null}
				<span class="sep">·</span>
				<span class:warn={nowcast.stale}>{t().loop.radarAge(Math.round(ageMin))}</span>
			{/if}
		</p>
		{#if nowcast.stale}
			<p class="warn small">{t().loop.stale}</p>
		{/if}
	{/if}
</div>

<style>
	.controls {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
		padding: 0.6rem 0.8rem 0.7rem;
		background: var(--surface);
		border-radius: var(--radius) var(--radius) 0 0;
		box-shadow: var(--shadow);
	}

	.controls.empty {
		padding: 0.9rem;
	}

	.row {
		display: flex;
		align-items: center;
		gap: 0.7rem;
	}

	.play {
		flex: 0 0 auto;
		width: 2.75rem;
		height: 2.75rem;
		border-radius: 50%;
		border: none;
		background: var(--accent);
		color: var(--accent-ink);
		display: grid;
		place-items: center;
		cursor: pointer;
	}

	.play svg {
		width: 1.3rem;
		height: 1.3rem;
		fill: currentColor;
	}

	.track {
		flex: 1 1 auto;
		min-width: 0;
	}

	input[type='range'] {
		width: 100%;
		accent-color: var(--accent);
	}

	.ticks {
		display: flex;
		justify-content: space-between;
		gap: 0.15rem;
		overflow: hidden;
	}

	.tick {
		border: none;
		background: none;
		padding: 0;
		font-size: 0.62rem;
		color: var(--muted);
		cursor: pointer;
		white-space: nowrap;
	}

	.tick.active {
		color: var(--accent);
		font-weight: 600;
	}

	.meta {
		margin: 0;
		font-size: 0.78rem;
		color: var(--muted);
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		align-items: baseline;
	}

	.meta strong {
		color: var(--ink);
	}

	.sep {
		opacity: 0.5;
	}

	.warn {
		color: var(--warn);
	}

	.small {
		margin: 0;
		font-size: 0.75rem;
	}

	.muted {
		margin: 0;
		color: var(--muted);
		font-size: 0.85rem;
	}

	/* Nine labels fit neither a phone nor the desktop side panel: show every
	   second one, and let the current frame's own label carry the rest (it is
	   spelled out in full underneath). */
	@media (max-width: 34rem), (min-width: 52rem) {
		.tick:nth-child(even) {
			display: none;
		}

		.ticks {
			justify-content: space-around;
		}
	}
</style>
