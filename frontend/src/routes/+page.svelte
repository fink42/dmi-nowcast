<script lang="ts">
	/** The map page: everything the site is actually for. */
	import { onMount } from 'svelte';
	import MapView from '$lib/components/MapView.svelte';
	import LoopControls from '$lib/components/LoopControls.svelte';
	import ForecastPanel from '$lib/components/ForecastPanel.svelte';
	import SiteFooter from '$lib/components/SiteFooter.svelte';
	import { t } from '$lib/i18n';
	import { nowcast } from '$lib/nowcast/store.svelte';
	import { inCoverage } from '$lib/nowcast/sampler';

	/** Matches the breakpoint where the sheet becomes a side panel (see below). */
	const SIDE_PANEL = '(min-width: 52rem)';

	let mapView: MapView;
	let locating = $state(false);
	let locationError = $state<string | null>(null);
	let sheet = $state<HTMLDivElement>();
	let sheetHeight = $state(0);
	let sidePanel = $state(false);

	onMount(() => nowcast.start());

	// The sheet only covers the map when it is *under* it; beside it, the map
	// needs no bottom padding.
	$effect(() => {
		const media = matchMedia(SIDE_PANEL);
		sidePanel = media.matches;
		const onChange = () => (sidePanel = media.matches);
		media.addEventListener('change', onChange);
		return () => media.removeEventListener('change', onChange);
	});

	// Keep the map's padding in sync with the sheet, so a tapped point never
	// ends up hidden behind it.
	$effect(() => {
		if (!sheet) return;
		const observer = new ResizeObserver(([entry]) => {
			sheetHeight = entry.contentRect.height;
		});
		observer.observe(sheet);
		return () => observer.disconnect();
	});

	function locate() {
		locationError = null;
		if (!('geolocation' in navigator)) {
			locationError = t().map.locateFailed;
			return;
		}
		locating = true;
		navigator.geolocation.getCurrentPosition(
			(position) => {
				locating = false;
				const { latitude, longitude } = position.coords;
				const grid = nowcast.manifest?.grid;
				if (grid && !inCoverage(grid, longitude, latitude)) {
					locationError = t().map.locateOutside;
				}
				void nowcast.selectPoint(latitude, longitude);
				mapView?.flyTo(latitude, longitude);
			},
			(err) => {
				locating = false;
				locationError = err.code === err.PERMISSION_DENIED ? t().map.locateDenied : t().map.locateFailed;
			},
			{ enableHighAccuracy: false, timeout: 10_000, maximumAge: 60_000 }
		);
	}
</script>

<div class="page">
	<div class="map-area">
		<MapView bind:this={mapView} bottomInset={sidePanel ? 0 : sheetHeight} />

		<div class="floating">
			{#if nowcast.status === 'nodata'}
				<p class="notice">{t().status.noData}</p>
			{:else if nowcast.offlineWithCachedCycle}
				<!-- Polls are failing but a cycle is still on the map: say which one
				     the map belongs to, rather than letting its age creep up in
				     silence. -->
				<p class="notice warn">{t().status.offlineCached}</p>
			{:else if nowcast.status === 'error'}
				<p class="notice">{t().status.offline}</p>
			{:else if !nowcast.point}
				<p class="notice subtle">{t().map.hint}</p>
			{/if}
			{#if locationError}
				<p class="notice">{locationError}</p>
			{/if}
		</div>

		<button class="locate" type="button" onclick={locate} aria-label={t().map.locate} disabled={locating}>
			<svg viewBox="0 0 24 24" aria-hidden="true">
				<path
					d="M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zm0-6v3m0 14v3m10-10h-3M5 12H2"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					stroke-linecap="round"
				/>
			</svg>
		</button>
	</div>

	<div class="sheet" bind:this={sheet}>
		<ForecastPanel />
		<LoopControls />
		<SiteFooter />
	</div>
</div>

<style>
	.page {
		display: flex;
		flex-direction: column;
		flex: 1 1 auto;
		min-height: 0;
	}

	.map-area {
		position: relative;
		flex: 1 1 auto;
		min-height: 12rem;
	}

	.sheet {
		flex: 0 0 auto;
		background: var(--surface);
		max-height: 80dvh;
		overflow-y: auto;
	}

	.floating {
		position: absolute;
		top: 0.6rem;
		left: 0.6rem;
		right: 3.6rem;
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
		pointer-events: none;
	}

	.notice {
		margin: 0;
		align-self: flex-start;
		background: var(--surface);
		color: var(--ink);
		border-radius: 999px;
		padding: 0.3rem 0.75rem;
		font-size: 0.78rem;
		box-shadow: var(--shadow);
	}

	.notice.subtle {
		color: var(--muted);
	}

	.notice.warn {
		color: var(--warn);
		font-weight: 600;
	}

	.locate {
		position: absolute;
		right: 0.6rem;
		/* Clear of MapLibre's attribution strip, which must stay readable. */
		bottom: 2.9rem;
		width: 2.75rem;
		height: 2.75rem;
		border-radius: 50%;
		border: 1px solid var(--border);
		background: var(--surface);
		color: var(--ink);
		display: grid;
		place-items: center;
		box-shadow: var(--shadow);
		cursor: pointer;
	}

	.locate:disabled {
		opacity: 0.6;
	}

	.locate svg {
		width: 1.35rem;
		height: 1.35rem;
	}

	/* Wide screens: the sheet becomes a side panel. */
	@media (min-width: 52rem) {
		.page {
			flex-direction: row;
		}

		.sheet {
			width: 24rem;
			max-height: none;
			border-left: 1px solid var(--border);
			display: flex;
			flex-direction: column;
		}
	}
</style>
