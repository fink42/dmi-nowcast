<script lang="ts">
	/**
	 * "How good are we?" — the page that has to be readable by someone who
	 * does not know what a Brier score is, and checkable by someone who does.
	 *
	 * Three rules run through it:
	 *
	 *  - **Two truths, always labelled.** The radar sees the whole country but
	 *    is blind to its own biases; the gauges are the ground but are a
	 *    hundred points on a 10-minute clock. Every number says which one it
	 *    was measured against.
	 *  - **Missing is missing.** Every section of `quality.json` may be null,
	 *    and a null renders as "not measured yet" — never as a zero, and never
	 *    as an empty chart that looks like a measurement of nothing.
	 *  - **Plain language first, detail after.** Three sentences at the top,
	 *    the diagrams below them, and the rules and their limits at the
	 *    bottom, where the reader who got that far wants them.
	 */
	import { onMount } from 'svelte';
	import { locale, t } from '$lib/i18n';
	import { DMI_OPEN_DATA_URL } from '$lib/links';
	import SiteFooter from '$lib/components/SiteFooter.svelte';
	import QualityDiagram from '$lib/components/QualityDiagram.svelte';
	import QualityStationMap from '$lib/components/QualityStationMap.svelte';
	import { localDate, localDateTime } from '$lib/quality/dates';
	import { eventRows } from '$lib/quality/events';
	import { fetchQuality } from '$lib/quality/load';
	import {
		countText,
		decimalText,
		marginCard,
		rainingNowLines,
		reliabilityCard,
		warningsCard
	} from '$lib/quality/sentences';
	import { thresholdIntro, thresholdRows } from '$lib/quality/thresholds';
	import type { QualityReport } from '$lib/quality/schema';

	let status = $state<'loading' | 'ready' | 'error'>('loading');
	let report = $state<QualityReport | null>(null);

	onMount(() => {
		const abort = new AbortController();
		fetchQuality(abort.signal)
			.then((parsed) => {
				report = parsed;
				status = 'ready';
			})
			.catch((error: unknown) => {
				if (abort.signal.aborted) return;
				// One line on screen and the detail on the console: a page about
				// honesty must not invent numbers when it cannot read any.
				console.warn('quality.json', error);
				status = 'error';
			});
		return () => abort.abort();
	});

	const cards = $derived(
		report
			? [
					{
						title: t().quality.headline.reliabilityTitle,
						card: reliabilityCard(t(), locale(), report.headline.reliability)
					},
					{
						title: t().quality.headline.warningsTitle,
						card: warningsCard(t(), locale(), report.headline.warnings)
					},
					{
						title: t().quality.headline.marginTitle,
						card: marginCard(t(), locale(), report.headline.persistence_margin)
					}
				]
			: []
	);

	const rainingNow = $derived(report ? rainingNowLines(t(), locale(), report.raining_now) : null);
	const rows = $derived(report ? eventRows(t(), locale(), report.events) : []);
	const methods = $derived(report?.methods ?? null);
	const windows = $derived(report?.windows ?? null);
	const thresholds = $derived(report?.thresholds ?? null);
	const thresholdTable = $derived(report ? thresholdRows(t(), locale(), thresholds) : []);
</script>

<article class="prose">
	<h1>{t().quality.title}</h1>
	<p class="lead">{t().quality.lead}</p>

	{#if status === 'loading'}
		<p class="quiet">{t().quality.loading}</p>
	{:else if status === 'error' || !report}
		<p class="quiet">{t().quality.error}</p>
	{:else}
		<section class="cards">
			{#each cards as entry (entry.title)}
				<div class="card" class:unmeasured={!entry.card.measured}>
					<h2>{entry.title}</h2>
					<p class="sentence">
						{#each entry.card.segments as segment, i (i)}
							{#if segment.strong}<strong>{segment.text}</strong>{:else}{segment.text}{/if}
						{/each}
					</p>
					{#if entry.card.detail}
						<p class="detail">{entry.card.detail}</p>
					{/if}
				</div>
			{/each}
		</section>

		<h2>{t().quality.reliability.title}</h2>
		{#if report.reliability.radar || report.reliability.gauge}
			<QualityDiagram reliability={report.reliability} />
		{:else}
			<p class="quiet">{t().quality.reliability.none}</p>
		{/if}

		<h2>{t().quality.stations.title}</h2>
		{#if report.stations}
			<QualityStationMap stations={report.stations} />
		{:else}
			<p class="quiet">{t().quality.stations.none}</p>
		{/if}

		<h2>{t().quality.rainingNow.title}</h2>
		{#if rainingNow}
			<p>{rainingNow.sentence}</p>
			<p>{rainingNow.comparison}</p>
			<p class="detail">{rainingNow.detail}</p>
		{:else}
			<p class="quiet">{t().quality.rainingNow.none}</p>
		{/if}

		<h2>{t().quality.events.title}</h2>
		{#if rows.length > 0}
			<p>{t().quality.events.intro}</p>
			<div class="scroll">
				<table class="events">
					<thead>
						<tr>
							<th scope="col">{t().quality.events.colStation}</th>
							<th scope="col">{t().quality.events.colWarned}</th>
							<th scope="col">{t().quality.events.colSaid}</th>
							<th scope="col">{t().quality.events.colHappened}</th>
							<th scope="col">{t().quality.events.colError}</th>
						</tr>
					</thead>
					<tbody>
						{#each rows as row (row.key)}
							<tr class:miss={!row.hit}>
								<th scope="row">{row.name}</th>
								<td>{row.warnedAt}</td>
								<td>{row.said}</td>
								<td>{row.happened}</td>
								<td>{row.error ?? t().quality.events.empty}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<p class="quiet">{t().quality.events.none}</p>
		{/if}

		<h2>{t().quality.thresholds.title}</h2>
		{#if thresholdTable.length > 0}
			<p>{thresholdIntro(t(), thresholds)}</p>
			<div class="scroll">
				<table class="thresholds">
					<thead>
						<tr>
							<th scope="col">{t().quality.thresholds.colHorizon}</th>
							<th scope="col">{t().quality.thresholds.colThreshold}</th>
							<th scope="col">{t().quality.thresholds.colPrecision}</th>
							<th scope="col">{t().quality.thresholds.colRecall}</th>
							<th scope="col">{t().quality.thresholds.colF1}</th>
							<th scope="col">{t().quality.thresholds.colWarnings}</th>
						</tr>
					</thead>
					<tbody>
						{#each thresholdTable as row (row.key)}
							<tr class:unfitted={row.insufficient}>
								<th scope="row">{row.horizon}</th>
								<td>{row.threshold}</td>
								<td>{row.precision}</td>
								<td>{row.recall}</td>
								<td>{row.f1}</td>
								<td>{row.warnings}</td>
							</tr>
							{#if row.note}
								<tr class="note">
									<td colspan="6">{row.note}</td>
								</tr>
							{/if}
						{/each}
					</tbody>
				</table>
			</div>
			<p class="detail">
				{t().quality.thresholds.fallbackNote(thresholds!.fallback_threshold_pct)}
				{#if thresholds?.fitted_at_utc}
					{t().quality.thresholds.fittedAt(localDate(thresholds.fitted_at_utc, locale()))}
				{/if}
			</p>
		{:else}
			<p class="quiet">{t().quality.thresholds.none}</p>
		{/if}

		<h2>{t().quality.methods.title}</h2>
		<h3>{t().quality.methods.radarTitle}</h3>
		<p>{t().quality.methods.radarBody}</p>
		<h3>{t().quality.methods.gaugeTitle}</h3>
		<p>{t().quality.methods.gaugeBody}</p>

		{#if methods}
			<h3>{t().quality.methods.rulesTitle}</h3>
			<dl>
				<dt>{t().quality.methods.wetRule}</dt>
				<dd>{methods.gauge_wet_rule}</dd>
				<dt>{t().quality.methods.onsetRule}</dt>
				<dd>{methods.onset_rule}</dd>
				<dt>{t().quality.methods.threshold}</dt>
				<dd>{t().quality.methods.thresholdValue(decimalText(methods.threshold_mm_h, locale(), 1))}</dd>
				<dt>{t().quality.methods.frameAge}</dt>
				<dd>
					{t().quality.methods.frameAgeValue(
						countText(methods.frame_age_range_min[0], locale()),
						countText(methods.frame_age_range_min[1], locale())
					)}
				</dd>
				<dt>{t().quality.methods.subscriberRule}</dt>
				<dd>
					{t().quality.methods.subscriberRuleValue(
						t().quality.percent(methods.subscriber_rule.threshold_pct),
						methods.subscriber_rule.lead_min,
						methods.subscriber_rule.persistence_obs,
						methods.subscriber_rule.rearm_after_min
					)}
				</dd>
				<dt>{t().quality.methods.sourceRadar}</dt>
				<dd>{methods.sources.radar}</dd>
				<dt>{t().quality.methods.sourceGauges}</dt>
				<dd>{methods.sources.gauges}</dd>
			</dl>
		{/if}
		<p class="detail">{t().quality.methods.leadErrorNote}</p>

		{#if windows && (windows.radar || windows.gauge || windows.live)}
			<h3>{t().quality.methods.windowsTitle}</h3>
			<ul>
				{#if windows.radar}
					<li>
						{t().quality.methods.radarWindow(
							localDate(windows.radar.from, locale()),
							localDate(windows.radar.to, locale()),
							countText(windows.radar.events, locale()),
							countText(windows.radar.points, locale())
						)}
					</li>
				{/if}
				{#if windows.gauge}
					<li>
						{t().quality.methods.gaugeWindow(
							localDate(windows.gauge.from, locale()),
							localDate(windows.gauge.to, locale()),
							countText(windows.gauge.events, locale()),
							countText(windows.gauge.stations, locale())
						)}
					</li>
				{/if}
				{#if windows.live}
					<li>
						{t().quality.methods.liveWindow(
							windows.live.days,
							localDate(windows.live.from, locale()),
							localDate(windows.live.to, locale())
						)}
					</li>
				{/if}
			</ul>
		{/if}

		<p>
			{t().quality.methods.cadence}
			{t().quality.generatedAt(localDateTime(report.generated_at_utc, locale()))}
		</p>
		<p>
			{t().quality.methods.attribution}
			<a href={DMI_OPEN_DATA_URL} target="_blank" rel="noreferrer">{t().data.radarLink}</a>
		</p>
	{/if}
</article>
<SiteFooter />

<style>
	.quiet {
		color: var(--muted);
	}

	.cards {
		display: grid;
		gap: 0.7rem;
		margin: 1rem 0 1.6rem;
	}

	.card {
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 0.75rem 0.9rem;
	}

	.card h2 {
		font-size: 0.78rem;
		text-transform: uppercase;
		letter-spacing: 0.03em;
		color: var(--muted);
		margin: 0 0 0.3rem;
	}

	.sentence {
		margin: 0;
		font-size: 1rem;
	}

	.sentence strong {
		font-size: 1.15rem;
	}

	.card.unmeasured .sentence {
		color: var(--muted);
	}

	.detail {
		margin: 0.3rem 0 0;
		font-size: 0.78rem;
		color: var(--muted);
	}

	dl {
		margin: 0.4rem 0 0.8rem;
		font-size: 0.9rem;
	}

	dt {
		font-weight: 600;
		margin-top: 0.5rem;
	}

	dd {
		margin: 0.1rem 0 0;
		color: var(--muted);
	}

	.scroll {
		overflow-x: auto;
	}

	table.events,
	table.thresholds {
		border-collapse: collapse;
		font-size: 0.82rem;
		width: 100%;
	}

	table.events th,
	table.events td,
	table.thresholds th,
	table.thresholds td {
		text-align: left;
		padding: 0.3rem 0.55rem 0.3rem 0;
		border-bottom: 1px solid var(--border);
		white-space: nowrap;
	}

	table.events thead th,
	table.thresholds thead th {
		color: var(--muted);
		font-weight: 500;
	}

	table.events tbody th,
	table.thresholds tbody th {
		font-weight: 600;
	}

	/* A horizon the fit cannot speak for keeps its row and loses its weight. */
	table.thresholds tr.unfitted td,
	table.thresholds tr.unfitted th {
		color: var(--muted);
	}

	table.thresholds tr.note td {
		color: var(--muted);
		font-size: 0.76rem;
		white-space: normal;
		padding-bottom: 0.45rem;
	}

	/* A false alarm is not a failure to hide; it is greyed, not deleted. */
	table.events tr.miss td,
	table.events tr.miss th {
		color: var(--muted);
	}
</style>
