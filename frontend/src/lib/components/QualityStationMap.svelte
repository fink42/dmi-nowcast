<script lang="ts">
	/**
	 * Where the gauges are, and how the warnings did at each of them.
	 *
	 * An inline SVG scatter over a coarse Denmark outline, not a second
	 * MapLibre map: this is a content page, the picture is a dozen dots on a
	 * country, and a basemap here would cost a WebGL context and a megabyte of
	 * tiles to say nothing the outline does not. The projection and the
	 * outline come from `$lib/quality/stations`, so the dots and the coast
	 * cannot drift apart.
	 *
	 * Colour is warm-to-cool rather than red-to-green (deuteranopia sees the
	 * latter as one colour), the band boundaries are written out in the
	 * legend, and every dot's numbers are also in the list underneath — so the
	 * colour is never the only place a number lives.
	 */
	import { locale, t } from '$lib/i18n';
	import {
		DENMARK_PATH,
		MAP_HEIGHT,
		MAP_WIDTH,
		plotStations,
		type PlottedStation
	} from '$lib/quality/stations';
	import { countText, decimalText, fractionText } from '$lib/quality/sentences';
	import type { StationCollection } from '$lib/quality/schema';

	let { stations }: { stations: StationCollection } = $props();

	const plotted = $derived(plotStations(stations.features));

	/** The line of numbers a station carries, in the tooltip and in the list. */
	function detail(station: PlottedStation): string {
		const { properties } = station.feature;
		const parts: string[] = [];
		if (properties.warn_pod !== null) {
			parts.push(t().quality.stations.stationPod(fractionText(t(), properties.warn_pod)));
		}
		if (properties.warn_far !== null) {
			parts.push(t().quality.stations.stationFar(fractionText(t(), properties.warn_far)));
		}
		if (properties.warn_pod === null) {
			parts.push(
				properties.brier_gauge === null
					? t().quality.stations.stationNoScore
					: t().quality.stations.stationBrier(decimalText(properties.brier_gauge, locale(), 3))
			);
		}
		parts.push(
			t().quality.stations.stationEvents(
				countText(properties.n_events, locale()),
				countText(properties.warnings, locale())
			)
		);
		return parts.join(' · ');
	}

	const label = (station: PlottedStation): string =>
		t().quality.stations.stationLabel(
			station.feature.properties.name,
			station.feature.properties.kind
		);
</script>

<p class="intro">{t().quality.stations.intro}</p>

<svg
	viewBox={`0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`}
	role="img"
	aria-label={t().quality.stations.mapLabel}
>
	<path d={DENMARK_PATH} class="land" />
	{#each plotted as station (station.feature.properties.station_id)}
		<circle
			cx={station.x}
			cy={station.y}
			r="6"
			class={`dot ${station.score.band}`}
			class:ring={station.score.basis === 'brier'}
		>
			<title>{label(station)} — {detail(station)}</title>
		</circle>
	{/each}
</svg>

<p class="hint">{t().quality.stations.hint}</p>

<ul class="legend">
	<li><span class="dot poor" aria-hidden="true"></span>{t().quality.stations.legendPoor}</li>
	<li><span class="dot fair" aria-hidden="true"></span>{t().quality.stations.legendFair}</li>
	<li><span class="dot good" aria-hidden="true"></span>{t().quality.stations.legendGood}</li>
	<li><span class="dot best" aria-hidden="true"></span>{t().quality.stations.legendBest}</li>
	<li>
		<span class="dot unknown" aria-hidden="true"></span>{t().quality.stations.legendUnknown}
	</li>
	<li class="wide">{t().quality.stations.legendBrier}</li>
</ul>

<ul class="stations">
	{#each plotted as station (station.feature.properties.station_id)}
		<li>
			<span class={`dot ${station.score.band}`} class:ring={station.score.basis === 'brier'}
			></span>
			<span class="name">{label(station)}</span>
			<span class="numbers">{detail(station)}</span>
		</li>
	{/each}
</ul>

<style>
	.intro {
		margin: 0 0 0.6rem;
	}

	svg {
		width: 100%;
		max-width: 26rem;
		height: auto;
		display: block;
		margin: 0 auto;
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: 8px;
	}

	.land {
		fill: var(--map-bg);
		stroke: var(--border);
		stroke-width: 1;
		stroke-linejoin: round;
	}

	/*
	 * Warm = little of the rain warned about, cool = most of it. Deliberately
	 * not red-to-green: orange against blue survives the common forms of
	 * colour blindness, and the legend spells the bands out anyway.
	 */
	.poor {
		color: #c2622b;
	}

	.fair {
		color: #d9a441;
	}

	.good {
		color: #5a9bd4;
	}

	.best {
		color: #1f6aa8;
	}

	.unknown {
		color: var(--muted);
	}

	@media (prefers-color-scheme: dark) {
		.poor {
			color: #e0813f;
		}

		.fair {
			color: #e8bf63;
		}

		.good {
			color: #7fb8e8;
		}

		.best {
			color: #4a95d6;
		}
	}

	circle.dot {
		fill: currentColor;
		stroke: var(--surface);
		stroke-width: 1.5;
	}

	/* An open ring: the colour came from the Brier score, not from a hit rate. */
	circle.dot.ring {
		fill: var(--surface);
		stroke: currentColor;
		stroke-width: 2.5;
	}

	.hint {
		margin: 0.4rem 0 0.6rem;
		font-size: 0.78rem;
		color: var(--muted);
		text-align: center;
	}

	.legend {
		list-style: none;
		padding: 0;
		margin: 0 0 0.8rem;
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem 0.9rem;
		font-size: 0.78rem;
		color: var(--muted);
	}

	.legend li {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		margin: 0;
	}

	.legend .wide {
		flex-basis: 100%;
	}

	span.dot {
		width: 0.7rem;
		height: 0.7rem;
		border-radius: 999px;
		background: currentColor;
		flex: 0 0 auto;
	}

	span.dot.ring {
		background: transparent;
		border: 2px solid currentColor;
	}

	.stations {
		list-style: none;
		padding: 0;
		margin: 0;
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
		gap: 0.3rem 1rem;
		font-size: 0.8rem;
	}

	.stations li {
		display: grid;
		grid-template-columns: auto 1fr;
		column-gap: 0.4rem;
		margin: 0;
	}

	.name {
		font-weight: 600;
	}

	.numbers {
		grid-column: 2;
		color: var(--muted);
		font-size: 0.75rem;
	}
</style>
