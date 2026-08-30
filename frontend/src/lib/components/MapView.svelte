<script lang="ts">
	/**
	 * The map: Protomaps basemap from our own pmtiles archive, the radar loop
	 * as a single image source whose bitmap swaps per frame, a marker for the
	 * selected point, and a click handler that turns any point in Denmark into
	 * a forecast.
	 */
	import { onMount } from 'svelte';
	import {
		addProtocol,
		removeProtocol,
		setWorkerUrl,
		Map as MapLibreMap,
		Marker,
		NavigationControl,
		type ImageSource
	} from 'maplibre-gl';
	// MapLibre 6 loads its worker by a runtime-built relative URL, which no
	// bundler can follow. Let Vite bundle the worker and hand us its URL.
	import maplibreWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';
	import { Protocol } from 'pmtiles';
	import 'maplibre-gl/dist/maplibre-gl.css';
	import { buildStyle, preferredTheme, type Theme } from '$lib/map/style';
	import { nowcast } from '$lib/nowcast/store.svelte';
	import { t } from '$lib/i18n';

	interface Props {
		/** Extra bottom padding so the forecast sheet does not cover the marker. */
		bottomInset?: number;
	}
	let { bottomInset = 0 }: Props = $props();

	const OVERLAY_SOURCE = 'nowcast-overlay';
	const OVERLAY_LAYER = 'nowcast-overlay-layer';
	/** 1×1 transparent PNG — the placeholder an image source must be born with. */
	const BLANK_PNG =
		'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
	/**
	 * Denmark, framed to whatever viewport we get — a phone in portrait and a
	 * desktop side-panel layout want very different zooms for the same country.
	 */
	const DENMARK: [[number, number], [number, number]] = [
		[7.9, 54.5],
		[15.3, 57.85]
	];

	/** Language the current style was built with (see the effect below). */
	const INITIAL_LANG = t().locale;

	let container: HTMLDivElement;
	let map: MapLibreMap | null = null;
	let theme = $state<Theme>('light');
	let marker: Marker | null = null;
	let styleReady = $state(false);

	onMount(() => {
		setWorkerUrl(maplibreWorkerUrl);
		const protocol = new Protocol();
		addProtocol('pmtiles', protocol.tile);
		theme = preferredTheme();

		map = new MapLibreMap({
			container,
			style: buildStyle(theme, t().locale, {
				radar: t().map.attributionRadar,
				osm: t().data.basemapLinkOsm
			}),
			bounds: DENMARK,
			fitBoundsOptions: { padding: 12 },
			minZoom: 5,
			maxZoom: 12,
			attributionControl: { compact: true },
			// Denmark only — no point letting people pan to the Pacific.
			maxBounds: [
				[3.5, 52.5],
				[19.5, 59.5]
			]
		});
		map.addControl(new NavigationControl({ showCompass: false }), 'top-right');
		map.on('style.load', () => {
			styleReady = true;
			syncOverlay();
		});
		map.on('click', (e) => {
			void nowcast.selectPoint(e.lngLat.lat, e.lngLat.lng);
		});

		const media = matchMedia('(prefers-color-scheme: dark)');
		const onScheme = () => {
			theme = media.matches ? 'dark' : 'light';
			styleReady = false;
			map?.setStyle(
				buildStyle(theme, t().locale, {
					radar: t().map.attributionRadar,
					osm: t().data.basemapLinkOsm
				})
			);
		};
		media.addEventListener('change', onScheme);

		return () => {
			media.removeEventListener('change', onScheme);
			marker?.remove();
			map?.remove();
			map = null;
			removeProtocol('pmtiles');
		};
	});

	/** Create or update the image source that carries the current frame. */
	function syncOverlay() {
		if (!map || !styleReady) return;
		const geometry = nowcast.geometry;
		const frame = nowcast.currentFrame;
		if (!geometry || !frame) return;
		const existing = map.getSource(OVERLAY_SOURCE) as ImageSource | undefined;
		if (existing) {
			existing.updateImage({ image: frame.bitmap, coordinates: geometry.corners });
			return;
		}
		// MapLibre's style validation insists on a `url`, so the source is born
		// holding a 1×1 transparent pixel and gets its real bitmap immediately
		// after. The DMI credit rides on the basemap source's attribution
		// string, which MapLibre always shows (see $lib/map/style.ts).
		map.addSource(OVERLAY_SOURCE, {
			type: 'image',
			url: BLANK_PNG,
			coordinates: geometry.corners
		});
		(map.getSource(OVERLAY_SOURCE) as ImageSource).updateImage({ image: frame.bitmap });
		// Under the labels: place names stay readable through the rain.
		const labelLayer = map
			.getStyle()
			.layers.find((l) => l.type === 'symbol')?.id;
		map.addLayer(
			{
				id: OVERLAY_LAYER,
				type: 'raster',
				source: OVERLAY_SOURCE,
				paint: { 'raster-opacity': 0.85, 'raster-fade-duration': 0, 'raster-resampling': 'linear' }
			},
			labelLayer
		);
	}

	// Frame changes, cycle changes and style reloads all land here.
	$effect(() => {
		void nowcast.frameIndex;
		void nowcast.frames.length;
		void nowcast.geometry;
		void styleReady;
		syncOverlay();
	});

	// Switching language re-labels the map too (Protomaps ships localised
	// name fields), which means a fresh style — the overlay re-attaches on
	// style.load like it does for a theme change.
	let styledLang = INITIAL_LANG;
	$effect(() => {
		const lang = t().locale;
		if (!map || lang === styledLang) return;
		styledLang = lang;
		styleReady = false;
		map.setStyle(
			buildStyle(theme, lang, { radar: t().map.attributionRadar, osm: t().data.basemapLinkOsm })
		);
	});

	// The marker follows the selected point.
	$effect(() => {
		const point = nowcast.point;
		if (!map) return;
		if (!point) {
			marker?.remove();
			marker = null;
			return;
		}
		if (!marker) {
			const element = document.createElement('div');
			element.className = 'point-marker';
			element.setAttribute('aria-label', t().map.selectedPoint);
			marker = new Marker({ element }).setLngLat([point.lon, point.lat]).addTo(map);
		} else {
			marker.setLngLat([point.lon, point.lat]);
		}
	});

	// Keep the selected point clear of the bottom sheet.
	$effect(() => {
		map?.setPadding({ top: 0, right: 0, bottom: bottomInset, left: 0 });
	});

	export function flyTo(lat: number, lon: number) {
		map?.easeTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), 8) });
	}
</script>

<div class="map" bind:this={container} role="application" aria-label={t().map.label}></div>

<style>
	.map {
		position: absolute;
		inset: 0;
		background: var(--map-bg);
	}

	:global(.point-marker) {
		width: 18px;
		height: 18px;
		border-radius: 50%;
		border: 3px solid var(--accent);
		background: var(--surface);
		box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.25);
	}

	/* MapLibre's own chrome, nudged to match the app. */
	:global(.maplibregl-ctrl-attrib) {
		font-size: 0.7rem;
	}
</style>
