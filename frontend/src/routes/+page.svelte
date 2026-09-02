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
	import { pointFromUrl } from '$lib/push/notification';
	import { push } from '$lib/push/store.svelte';

	/** Matches the breakpoint where the sheet becomes a side panel (see below). */
	const SIDE_PANEL = '(min-width: 52rem)';

	let mapView: MapView;
	let locating = $state(false);
	let locationError = $state<string | null>(null);
	let sheet = $state<HTMLDivElement>();
	let sheetHeight = $state(0);
	let dock = $state<HTMLDivElement>();
	let dockHeight = $state(0);
	let sidePanel = $state(false);

	/**
	 * A point arriving from outside the page: `/?lat=&lon=` when a
	 * notification opened a cold tab, and the service worker's `open-point`
	 * message when it focused a tab that was already there. Both mean the same
	 * thing — show this point — and both may arrive before the manifest does.
	 */
	let deepLink: { lat: number; lon: number } | null = null;

	function openPoint(lat: number, lon: number) {
		void nowcast.selectPoint(lat, lon);
		mapView?.flyTo(lat, lon);
	}

	onMount(() => {
		const stopNowcast = nowcast.start();
		void push.init();

		const initial = pointFromUrl(location.search);
		if (initial) {
			// `selectPoint` works without grids — it falls back to /forecast —
			// so the panel fills in straight away and is re-sampled below once
			// the cycle's grids arrive.
			openPoint(initial.lat, initial.lon);
			if (!nowcast.manifest) deepLink = initial;
		}

		const onMessage = (event: MessageEvent) => {
			const data = event.data as { type?: string; lat?: number; lon?: number } | null;
			if (!data || data.type !== 'open-point') return;
			if (typeof data.lat !== 'number' || typeof data.lon !== 'number') return;
			openPoint(data.lat, data.lon);
		};
		const worker = 'serviceWorker' in navigator ? navigator.serviceWorker : null;
		worker?.addEventListener('message', onMessage);

		return () => {
			worker?.removeEventListener('message', onMessage);
			stopNowcast();
		};
	});

	// The deep link landed before the first manifest: re-select once the cycle
	// is there, so the point is sampled from the grids rather than left on the
	// server fallback, and the map flies to it now that it can.
	$effect(() => {
		if (!nowcast.manifest || !deepLink) return;
		const point = deepLink;
		deepLink = null;
		openPoint(point.lat, point.lon);
	});

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

	// The timeline sits *on* the map, so its height is map chrome: it pads the
	// map, lifts the locate button, and lifts MapLibre's attribution strip —
	// which has to stay visible whatever the timeline does (see the style
	// block below and $lib/map/style.ts).
	$effect(() => {
		if (!dock) return;
		const observer = new ResizeObserver(([entry]) => {
			dockHeight = entry.contentRect.height;
		});
		observer.observe(dock);
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
	<div class="map-area" style:--dock-h={`${dockHeight}px`}>
		<MapView bind:this={mapView} bottomInset={dockHeight + (sidePanel ? 0 : sheetHeight)} />

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

		<!-- The timeline belongs to the map on every viewport, phone included:
		     it is the map's own clock, not a panel item. -->
		<div class="dock" bind:this={dock}>
			<LoopControls />
		</div>
	</div>

	<!-- Only the phone sheet folds: it lies on top of the map, so a peek is
	     the only way to see both. The side panel covers nothing. -->
	<div class="sheet" bind:this={sheet}>
		<ForecastPanel collapsible={!sidePanel} />
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

	.dock {
		position: absolute;
		left: 0;
		right: 0;
		bottom: 0;
		/* Above MapLibre's own control containers (z-index 2). */
		z-index: 3;
	}

	/* The attribution rides above the timeline instead of under it: DMI and
	   OpenStreetMap credit is not optional chrome. */
	.map-area :global(.maplibregl-ctrl-bottom-left),
	.map-area :global(.maplibregl-ctrl-bottom-right) {
		bottom: var(--dock-h, 0px);
	}

	.locate {
		position: absolute;
		right: 0.6rem;
		/* Clear of MapLibre's attribution strip, which must stay readable. */
		bottom: calc(var(--dock-h, 0px) + 2.9rem);
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

	/* Wide screens: the sheet becomes a side panel, on the left. Only the
	   panel moves — the zoom control, the locate button, the notices and the
	   attribution all live inside .map-area and stay with the map. */
	@media (min-width: 52rem) {
		.page {
			flex-direction: row;
		}

		.sheet {
			order: -1;
			width: 24rem;
			max-height: none;
			border-right: 1px solid var(--border);
			display: flex;
			flex-direction: column;
		}
	}
</style>
