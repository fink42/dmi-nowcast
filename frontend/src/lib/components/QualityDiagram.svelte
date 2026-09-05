<script lang="ts">
	/**
	 * Reliability diagrams, one panel per lead: across, the probability we
	 * said; up, how often it then rained. The diagonal is where a calibrated
	 * forecast sits, and the whole page hangs on being able to see the
	 * distance from it.
	 *
	 * Two truths on the same panel, told apart by shape as well as colour
	 * (filled circle = radar, diamond = gauges) — colour alone is not a
	 * distinction everyone can see. Marker area grows with the number of
	 * forecasts behind the point, because the gauge curve's top bins rest on a
	 * few dozen events and should not look as solid as the radar's tens of
	 * thousands.
	 *
	 * Every panel carries the same numbers as a table, one `<details>` away,
	 * which is both the accessible alternative to the picture and the only
	 * honest way to show `n`.
	 */
	import { locale, t } from '$lib/i18n';
	import {
		curveAt,
		curvePoints,
		leadsOf,
		markerRadius,
		maxBinN,
		PLOT,
		plotX,
		plotY,
		polyline,
		tableRows
	} from '$lib/quality/diagram';
	import { countText, decimalText } from '$lib/quality/sentences';
	import type { QualityReliability } from '$lib/quality/schema';

	let { reliability }: { reliability: QualityReliability } = $props();

	const panels = $derived(
		leadsOf(reliability).map((lead) => {
			const radar = curveAt(reliability.radar, lead);
			const gauge = curveAt(reliability.gauge, lead);
			return {
				lead,
				radar,
				gauge,
				radarPoints: curvePoints(radar),
				gaugePoints: curvePoints(gauge),
				maxN: maxBinN(radar, gauge),
				rows: tableRows(radar, gauge)
			};
		})
	);

	/** A diamond, so the gauge series is legible without colour. */
	const diamond = (x: number, y: number, r: number): string =>
		`M${x},${y - r} L${x + r},${y} L${x},${y + r} L${x - r},${y} Z`;

	const pct = (value: number): string => t().quality.percent(Math.round(value * 100));
	const binLabel = (lo: number, hi: number): string =>
		`${Math.round(lo * 100)}–${t().quality.percent(Math.round(hi * 100))}`;
	const freq = (value: number | null): string =>
		value === null ? t().quality.reliability.empty : pct(value);
	const count = (value: number): string => countText(value, locale());
</script>

<p class="intro">{t().quality.reliability.intro}</p>

<p class="legend">
	<span class="key">
		<svg viewBox="0 0 12 12" aria-hidden="true" class="swatch"
			><circle cx="6" cy="6" r="4" class="radar" /></svg
		>
		{t().quality.reliability.radar}
	</span>
	<span class="key">
		<svg viewBox="0 0 12 12" aria-hidden="true" class="swatch"
			><path d={diamond(6, 6, 4.6)} class="gauge" /></svg
		>
		{t().quality.reliability.gauge}
	</span>
	<span class="key">
		<svg viewBox="0 0 12 12" aria-hidden="true" class="swatch"
			><line x1="1" y1="11" x2="11" y2="1" class="diagonal" /></svg
		>
		{t().quality.reliability.perfect}
	</span>
	<span class="note">{t().quality.reliability.markerNote}</span>
</p>

<div class="panels">
	{#each panels as panel (panel.lead)}
		<figure>
			<figcaption>
				<strong>{t().quality.reliability.panel(panel.lead)}</strong>
				<span class="scores">
					{#if panel.radar}
						<span>{t().quality.reliability.brierRadar(decimalText(panel.radar.brier, locale(), 3))}</span>
					{/if}
					{#if panel.gauge}
						<span>{t().quality.reliability.brierGauge(decimalText(panel.gauge.brier, locale(), 3))}</span>
					{/if}
				</span>
			</figcaption>

			<svg
				viewBox={`0 0 ${PLOT.width} ${PLOT.height}`}
				role="img"
				aria-label={t().quality.reliability.tableCaption(panel.lead)}
			>
				<!-- Frame and the perfect-calibration diagonal. -->
				<rect
					x={plotX(0)}
					y={plotY(1)}
					width={plotX(1) - plotX(0)}
					height={plotY(0) - plotY(1)}
					class="frame"
				/>
				<line x1={plotX(0)} y1={plotY(0)} x2={plotX(1)} y2={plotY(1)} class="diagonal" />

				{#each [0, 0.5, 1] as tick (tick)}
					<line x1={plotX(tick)} y1={plotY(0)} x2={plotX(tick)} y2={plotY(0) + 4} class="tick" />
					<text x={plotX(tick)} y={plotY(0) + 14} class="tick-label" text-anchor="middle"
						>{pct(tick)}</text
					>
					<line x1={plotX(0) - 4} y1={plotY(tick)} x2={plotX(0)} y2={plotY(tick)} class="tick" />
					<text x={plotX(0) - 7} y={plotY(tick) + 3.5} class="tick-label" text-anchor="end"
						>{pct(tick)}</text
					>
				{/each}

				{#if polyline(panel.radarPoints)}
					<polyline points={polyline(panel.radarPoints)} class="line radar" />
				{/if}
				{#if polyline(panel.gaugePoints)}
					<polyline points={polyline(panel.gaugePoints)} class="line gauge" />
				{/if}

				{#each panel.radarPoints as point (point.lo)}
					<circle
						cx={plotX(point.x)}
						cy={plotY(point.y)}
						r={markerRadius(point.n, panel.maxN)}
						class="radar"
					/>
				{/each}
				{#each panel.gaugePoints as point (point.lo)}
					<path
						d={diamond(
							plotX(point.x),
							plotY(point.y),
							markerRadius(point.n, panel.maxN) * 1.15
						)}
						class="gauge"
					/>
				{/each}

				<text x={plotX(0.5)} y={PLOT.height - 3} class="axis" text-anchor="middle"
					>{t().quality.reliability.axisX}</text
				>
				<text
					x={-plotY(0.5)}
					y={9}
					class="axis"
					text-anchor="middle"
					transform="rotate(-90)"
					transform-origin="0 0">{t().quality.reliability.axisY}</text
				>
			</svg>

			<details>
				<summary>{t().quality.reliability.tableToggle}</summary>
				<div class="scroll">
					<table>
						<caption>{t().quality.reliability.tableCaption(panel.lead)}</caption>
						<thead>
							<tr>
								<th scope="col">{t().quality.reliability.colBin}</th>
								<th scope="col">{t().quality.reliability.colRadar}</th>
								<th scope="col">{t().quality.reliability.colRadarN}</th>
								<th scope="col">{t().quality.reliability.colGauge}</th>
								<th scope="col">{t().quality.reliability.colGaugeN}</th>
							</tr>
						</thead>
						<tbody>
							{#each panel.rows as row (row.lo)}
								<tr>
									<th scope="row">{binLabel(row.lo, row.hi)}</th>
									<td>{row.radar ? freq(row.radar.observed_freq) : t().quality.reliability.empty}</td>
									<td>{row.radar ? count(row.radar.n) : t().quality.reliability.empty}</td>
									<td>{row.gauge ? freq(row.gauge.observed_freq) : t().quality.reliability.empty}</td>
									<td>{row.gauge ? count(row.gauge.n) : t().quality.reliability.empty}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</details>
		</figure>
	{/each}
</div>

<style>
	.intro {
		margin: 0 0 0.6rem;
	}

	.legend {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.25rem 0.9rem;
		margin: 0 0 0.9rem;
		font-size: 0.8rem;
		color: var(--muted);
	}

	.key {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		color: var(--ink);
	}

	.swatch {
		width: 0.8rem;
		height: 0.8rem;
	}

	.note {
		flex-basis: 100%;
	}

	.panels {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
		gap: 1rem;
	}

	figure {
		margin: 0;
	}

	figcaption {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 0.15rem 0.6rem;
		font-size: 0.85rem;
		margin-bottom: 0.2rem;
	}

	.scores {
		display: flex;
		flex-wrap: wrap;
		gap: 0.6rem;
		font-size: 0.72rem;
		color: var(--muted);
	}

	svg {
		width: 100%;
		height: auto;
		display: block;
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: 8px;
	}

	.frame {
		fill: none;
		stroke: var(--border);
		stroke-width: 1;
	}

	.tick {
		stroke: var(--border);
		stroke-width: 1;
	}

	.tick-label,
	.axis {
		fill: var(--muted);
		font-size: 8px;
	}

	.axis {
		font-size: 8.5px;
	}

	.diagonal {
		stroke: var(--muted);
		stroke-width: 1;
		stroke-dasharray: 3 3;
		fill: none;
	}

	/* Two series, told apart by shape first and colour second. */
	circle.radar {
		fill: var(--accent);
	}

	path.gauge {
		fill: var(--warn);
	}

	.line {
		fill: none;
		stroke-width: 1.4;
		opacity: 0.75;
	}

	.line.radar {
		stroke: var(--accent);
	}

	.line.gauge {
		stroke: var(--warn);
	}

	details {
		margin-top: 0.35rem;
		font-size: 0.8rem;
	}

	summary {
		cursor: pointer;
		color: var(--accent);
	}

	.scroll {
		overflow-x: auto;
	}

	table {
		border-collapse: collapse;
		margin-top: 0.4rem;
		font-size: 0.75rem;
		width: 100%;
	}

	caption {
		text-align: left;
		color: var(--muted);
		padding-bottom: 0.25rem;
	}

	th,
	td {
		text-align: right;
		padding: 0.15rem 0.4rem;
		border-bottom: 1px solid var(--border);
		white-space: nowrap;
	}

	thead th,
	tbody th {
		text-align: left;
		font-weight: 600;
	}

	thead th {
		color: var(--muted);
		font-weight: 500;
	}
</style>
