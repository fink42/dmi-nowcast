<script lang="ts">
	/**
	 * The forecast for one point, told the way the Home Assistant card tells
	 * it: a headline first, then the numbers behind it — ETA, the calibrated
	 * probability across leads, intensity, confidence, and how old the radar
	 * image is. Off-coverage is its own state on purpose: a point outside the
	 * composite is unknown, not dry.
	 *
	 * On a phone the panel can be folded down to a peek — the header and the
	 * one line that answers the question — so the map underneath is visible
	 * without giving up the point, which is what the × does.
	 */
	import { t, locale } from '$lib/i18n';
	import {
		clockTime,
		confidenceWord,
		headline,
		headlineKind,
		intensityWord,
		percent,
		probabilityWithin
	} from '$lib/format';
	import { nowcast } from '$lib/nowcast/store.svelte';
	import NotifyPanel from './NotifyPanel.svelte';
	import { shouldExpandOnPointChange, type PanelPoint } from './panel';

	/**
	 * May this panel be folded down? True on a phone, where the sheet sits on
	 * top of the map and hides most of it. False for the wide-screen side
	 * panel, which covers nothing and so has nothing to gain by folding.
	 */
	let { collapsible = false }: { collapsible?: boolean } = $props();

	/** The `aria-controls` target; one forecast panel exists per page. */
	const BODY_ID = 'forecast-panel-body';

	const point = $derived(nowcast.point);
	const forecast = $derived(point?.forecast ?? null);
	/**
	 * Cell motion is null wherever there is no honest estimate — no motion
	 * grids this cycle, a nodata pixel (outside coverage, or too far from any
	 * echo), or the server fallback, which serves none. The row then says so
	 * in words; it never draws an arrow it cannot back up.
	 */
	const motion = $derived(forecast?.motion ?? null);
	const confidence = $derived(forecast?.confidence ?? nowcast.confidence);
	const ageMin = $derived(nowcast.radarAgeMin);
	const highlight = $derived(forecast ? probabilityWithin(forecast, 20) : null);
	const bars = $derived(forecast?.perLead.filter((l) => l.pRain !== null) ?? []);
	const maxP = $derived(Math.max(0.05, ...bars.map((b) => b.pRain ?? 0)));

	let collapsed = $state(false);
	/** Folded only where folding is offered at all. */
	const expanded = $derived(!(collapsible && collapsed));

	/**
	 * Picking a *different* point unfolds the panel: the user asked about
	 * somewhere new and the answer should not arrive hidden. The comparison
	 * is on coordinates, not object identity — the store replaces the whole
	 * `point` object every cycle when it re-samples the same place, and that
	 * must not undo a minimise.
	 */
	let seen: PanelPoint | null = null;
	$effect(() => {
		const next = point ? { lat: point.lat, lon: point.lon } : null;
		if (shouldExpandOnPointChange(seen, next)) collapsed = false;
		seen = next;
	});
</script>

{#if point}
	<section class="panel" class:collapsed={!expanded} aria-live="polite">
		<header>
			{#if collapsible}
				<!-- The whole title/coordinates row is the toggle: one control,
				     one tab stop, and a tap target as wide as the sheet. The ×
				     keeps its own job — clearing the point. -->
				<h2 class="heading">
					<button
						type="button"
						class="summary"
						aria-expanded={expanded}
						aria-controls={BODY_ID}
						onclick={() => (collapsed = !collapsed)}
					>
						<span class="chevron" class:up={!expanded} aria-hidden="true">
							<svg viewBox="0 0 24 24">
								<path
									d="M6 9l6 6 6-6"
									fill="none"
									stroke="currentColor"
									stroke-width="2.5"
									stroke-linecap="round"
									stroke-linejoin="round"
								/>
							</svg>
						</span>
						<span class="summary-text">
							<span class="title">{t().panel.title}</span>
							<span class="coords">{t().panel.coordinates(point.lat, point.lon)}</span>
						</span>
						<span class="sr-only">{expanded ? t().panel.collapse : t().panel.expand}</span>
					</button>
				</h2>
			{:else}
				<div class="summary-text">
					<h2 class="title">{t().panel.title}</h2>
					<p class="coords">{t().panel.coordinates(point.lat, point.lon)}</p>
				</div>
			{/if}
			<button type="button" class="close" aria-label={t().panel.close} onclick={() => nowcast.clearPoint()}
				>×</button
			>
		</header>

		<!-- The line that survives a fold: what is going to happen here, or
		     why there is no answer to give yet. -->
		{#if point.status === 'loading'}
			<p class="muted peek">{t().panel.loading}</p>
		{:else if point.status === 'off-coverage'}
			<p class="headline peek">{t().panel.offCoverage}</p>
		{:else if point.status === 'error'}
			<p class="muted peek">{t().panel.error}</p>
		{:else if forecast}
			<p class="headline peek" class:rain={headlineKind(forecast) !== 'no-rain'}>
				{headline(t(), forecast)}
			</p>
		{/if}

		<!-- Everything the peek leaves out. Folded, it is not rendered at all
		     — not merely hidden — so nothing below keeps polling or holding a
		     draft the user cannot see. -->
		<div id={BODY_ID}>
			{#if expanded}
				{#if point.status === 'off-coverage'}
					<p class="muted">{t().panel.offCoverageBody}</p>
				{:else if forecast}
					{#if highlight}
						<p class="lede">
							{t().panel.probabilityWithin(highlight.leadMin, percent(highlight.pRain))}
						</p>
					{/if}

					{#if bars.length > 0}
						<div class="curve" role="img" aria-label={t().panel.probabilityLabel}>
							{#each bars as bar (bar.leadMin)}
								<div class="bar-column">
									<div class="bar-track">
										<div
											class="bar"
											style:height={`${Math.round(((bar.pRain ?? 0) / maxP) * 100)}%`}
										></div>
									</div>
									<span class="bar-value">{percent(bar.pRain ?? 0)}</span>
									<span class="bar-label">{bar.leadMin}</span>
								</div>
							{/each}
						</div>
						<p class="axis">{t().panel.leadAxis}</p>
					{/if}

					<dl class="facts">
						<div>
							<dt>{t().panel.etaLabel}</dt>
							<dd>
								{forecast.etaMin === null
									? t().panel.etaNone
									: t().panel.etaValue(Math.round(forecast.etaMin))}
							</dd>
						</div>
						<div>
							<dt>{t().panel.intensityLabel}</dt>
							<dd>
								{intensityWord(t(), forecast.intensityMmH)}
								{#if forecast.intensityMmH !== null && forecast.intensityMmH > 0.05}
									<span class="muted">({t().panel.intensityValue(forecast.intensityMmH)})</span>
								{/if}
							</dd>
						</div>
						<div>
							<dt>{t().panel.motionLabel}</dt>
							<dd>
								{#if motion}
									<!-- The glyph points the way the cell is travelling and the
									     rotation IS the "coming from" bearing: the arrow is
									     drawn pointing south at 0°, so "from N" sends it down
									     the compass rose, towards the viewer. -->
									<span
										class="arrow"
										aria-hidden="true"
										style:rotate={`${motion.bearingFromDeg}deg`}
									>
										<svg viewBox="0 0 24 24">
											<path
												d="M12 2v12"
												fill="none"
												stroke="currentColor"
												stroke-width="2.5"
												stroke-linecap="round"
											/>
											<path d="M12 22l-5.5-8h11z" fill="currentColor" />
										</svg>
									</span>
									{t().panel.motionValue(
										t().panel.compass[motion.compass],
										Math.round(motion.speedKmh)
									)}
								{:else}
									<span class="muted">{t().panel.motionNone}</span>
								{/if}
							</dd>
						</div>
						{#if confidence !== null}
							<div>
								<dt>{t().panel.confidenceLabel}</dt>
								<dd>
									{confidenceWord(t(), confidence)}
									<span class="muted">({t().panel.confidenceValue(percent(confidence))})</span>
								</dd>
							</div>
						{/if}
						{#if ageMin !== null}
							<div>
								<dt>{t().panel.radarAgeLabel}</dt>
								<dd>
									{t().panel.radarAgeValue(Math.round(ageMin))}
									<span class="muted">({clockTime(forecast.radarTsUtc, locale())})</span>
								</dd>
							</div>
						{/if}
					</dl>

					<p class="badges">
						<span
							class="badge"
							class:calibrated={forecast.calibrated}
							title={forecast.calibrated
								? t().panel.calibratedTooltip
								: t().panel.uncalibratedTooltip}
						>
							{forecast.calibrated ? t().panel.calibratedBadge : t().panel.uncalibratedBadge}
						</span>
						<span class="source"
							>{forecast.source === 'client' ? t().panel.sourceLocal : t().panel.sourceServer}</span
						>
					</p>

					<!-- Only under a forecast that exists: a point with one is a point
					     inside the composite, which is the same test the subscribe
					     endpoint applies. -->
					<NotifyPanel />
				{/if}
			{/if}
		</div>
	</section>
{/if}

<style>
	.panel {
		background: var(--surface);
		border-radius: var(--radius) var(--radius) 0 0;
		box-shadow: var(--shadow);
		padding: 0.85rem 1rem 1rem;
		max-height: 62vh;
		overflow-y: auto;
	}

	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.5rem;
		margin-bottom: 0.5rem;
	}

	h2 {
		margin: 0;
		font-size: inherit;
		font-weight: inherit;
	}

	.heading {
		flex: 1 1 auto;
		min-width: 0;
	}

	/* The fold toggle: a plain button wearing the header row's clothes. */
	.summary {
		width: 100%;
		display: flex;
		align-items: center;
		gap: 0.5rem;
		min-width: 0;
		margin: 0;
		padding: 0;
		border: none;
		background: none;
		font: inherit;
		color: inherit;
		text-align: left;
		cursor: pointer;
	}

	.summary-text {
		flex: 1 1 auto;
		min-width: 0;
	}

	.title {
		display: block;
		font-size: 0.8rem;
		font-weight: 600;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		color: var(--muted);
	}

	.chevron {
		flex: 0 0 auto;
		width: 1.1rem;
		height: 1.1rem;
		color: var(--muted);
		transition: rotate 150ms ease-out;
	}

	/* Down while the panel is open ("push me down"), up while it is a peek. */
	.chevron.up {
		rotate: 180deg;
	}

	.chevron svg {
		width: 100%;
		height: 100%;
		display: block;
	}

	.close {
		flex: 0 0 auto;
		border: none;
		background: none;
		font-size: 1.6rem;
		line-height: 1;
		color: var(--muted);
		cursor: pointer;
		padding: 0 0.25rem;
	}

	.coords {
		display: block;
		margin: 0.1rem 0 0;
		font-size: 0.78rem;
		color: var(--muted);
		font-variant-numeric: tabular-nums;
	}

	.headline {
		margin: 0 0 0.25rem;
		font-size: 1.35rem;
		font-weight: 650;
		line-height: 1.2;
	}

	.headline.rain {
		color: var(--accent);
	}

	.lede {
		margin: 0 0 0.75rem;
		font-size: 0.95rem;
	}

	.curve {
		display: flex;
		align-items: flex-end;
		gap: 0.3rem;
		height: 5.5rem;
		margin-bottom: 0.2rem;
	}

	.bar-column {
		flex: 1 1 0;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.15rem;
		height: 100%;
	}

	.bar-track {
		flex: 1 1 auto;
		width: 100%;
		display: flex;
		align-items: flex-end;
		background: var(--track);
		border-radius: 3px;
		overflow: hidden;
	}

	.bar {
		width: 100%;
		background: var(--accent);
		border-radius: 3px 3px 0 0;
		min-height: 2px;
		transition: height 150ms ease-out;
	}

	.bar-value,
	.bar-label {
		font-size: 0.62rem;
		color: var(--muted);
		font-variant-numeric: tabular-nums;
	}

	.axis {
		margin: 0 0 0.75rem;
		font-size: 0.7rem;
		color: var(--muted);
		text-align: center;
	}

	.facts {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr));
		gap: 0.5rem 1rem;
		margin: 0 0 0.75rem;
	}

	dt {
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: var(--muted);
	}

	dd {
		margin: 0.1rem 0 0;
		font-size: 0.95rem;
	}

	.arrow {
		display: inline-block;
		width: 0.95rem;
		height: 0.95rem;
		vertical-align: -0.1rem;
		color: var(--accent);
	}

	.arrow svg {
		width: 100%;
		height: 100%;
		display: block;
	}

	.badges {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin: 0;
		flex-wrap: wrap;
	}

	.badge {
		font-size: 0.68rem;
		padding: 0.15rem 0.5rem;
		border-radius: 999px;
		border: 1px solid var(--border);
		color: var(--muted);
		cursor: help;
	}

	.badge.calibrated {
		border-color: var(--accent);
		color: var(--accent);
	}

	.source {
		font-size: 0.68rem;
		color: var(--muted);
	}

	.muted {
		color: var(--muted);
	}

	/* The one line that survives a collapse. */
	.peek {
		margin: 0 0 0.25rem;
	}

	.panel.collapsed {
		padding-bottom: 0.7rem;
	}

	.panel.collapsed .peek {
		margin-bottom: 0;
	}

	/* Smaller headline while collapsed: on a narrow phone one line is worth
	   more here than the extra weight. */
	.panel.collapsed .headline {
		font-size: 1.1rem;
	}

	/* Screen-reader-only: the toggle's state in words, next to aria-expanded. */
	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip-path: inset(50%);
		white-space: nowrap;
		border: 0;
	}
</style>
