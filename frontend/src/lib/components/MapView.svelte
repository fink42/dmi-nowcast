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
		type GeoJSONSource,
		type ImageSource,
		type LayerSpecification
	} from 'maplibre-gl';
	// MapLibre 6 loads its worker by a runtime-built relative URL, which no
	// bundler can follow. Let Vite bundle the worker and hand us its URL.
	import maplibreWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';
	import { Protocol } from 'pmtiles';
	import 'maplibre-gl/dist/maplibre-gl.css';
	import { emptyArrow, motionArrow, type ArrowCollection } from '$lib/map/arrow';
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
	const ARROW_SOURCE = 'nowcast-motion-arrow';
	/** Bottom to top: white halo, filled head, ink shaft, the now mark. See `arrowLayers`. */
	const ARROW_CASING_LAYER = 'nowcast-motion-arrow-casing';
	const ARROW_HEAD_LAYER = 'nowcast-motion-arrow-head';
	const ARROW_LINE_LAYER = 'nowcast-motion-arrow-line';
	const ARROW_NOW_LAYER = 'nowcast-motion-arrow-now';
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
			syncArrow();
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
			removeArrow();
			marker?.remove();
			map?.remove();
			map = null;
			removeProtocol('pmtiles');
		};
	});

	/**
	 * The lowest label layer. Everything we draw goes underneath it, so place
	 * names stay readable through the rain and under the arrow.
	 */
	function firstSymbolLayerId(): string | undefined {
		return map?.getStyle().layers.find((l) => l.type === 'symbol')?.id;
	}

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
		// Under the labels, and under the motion arrow if that got there first
		// (a point can be clicked before any frame has downloaded). Whichever
		// of the two is created first, the radar ends up below the arrow.
		map.addLayer(
			{
				id: OVERLAY_LAYER,
				type: 'raster',
				source: OVERLAY_SOURCE,
				paint: { 'raster-opacity': 0.85, 'raster-fade-duration': 0, 'raster-resampling': 'linear' }
			},
			map.getLayer(ARROW_CASING_LAYER) ? ARROW_CASING_LAYER : firstSymbolLayerId()
		);
	}

	/**
	 * The three layers of the arrow, bottom to top. The look is the Home
	 * Assistant card's (`_draw_motion_arrow`): a haloed shaft with ruler ticks
	 * and a filled head. The card's red becomes theme ink here — red is a rain
	 * intensity on the radar overlay, and an annotation must not be mistakable
	 * for data. The halo is what carries it over dark land, bright cells and
	 * the sea alike; the ink is what makes it legible against the halo.
	 *
	 * A `line` layer strokes polygon rings too, so the casing haloes the head
	 * without a layer of its own, and the ink line then crisps its edge.
	 *
	 * The one addition is the *now* mark, which the arrow carries when the radar
	 * image has aged (see `$lib/map/arrow`): the point on the shaft that
	 * wall-clock now has reached, everything behind it being rain that has
	 * already arrived. It is drawn in the same ink but heavier, so it separates
	 * from the ruler ticks it sits among without becoming a second colour.
	 */
	function arrowLayers(): LayerSpecification[] {
		const dark = theme === 'dark';
		const ink = dark ? '#f2f5f8' : '#101820';
		const casing = dark ? 'rgba(8, 12, 18, 0.85)' : 'rgba(255, 255, 255, 0.92)';
		const round = { 'line-cap': 'round', 'line-join': 'round' } as const;
		return [
			{
				id: ARROW_CASING_LAYER,
				type: 'line',
				source: ARROW_SOURCE,
				layout: round,
				paint: {
					'line-color': casing,
					'line-width': ['interpolate', ['linear'], ['zoom'], 5, 4.5, 12, 7.5]
				}
			},
			{
				id: ARROW_HEAD_LAYER,
				type: 'fill',
				source: ARROW_SOURCE,
				paint: { 'fill-color': ink, 'fill-opacity': 0.95 }
			},
			{
				id: ARROW_LINE_LAYER,
				type: 'line',
				source: ARROW_SOURCE,
				// Everything but the now mark, which the layer above draws
				// heavier — filtering here keeps it from being painted twice.
				filter: ['!=', ['get', 'role'], 'now'],
				layout: round,
				paint: {
					'line-color': ink,
					'line-width': ['interpolate', ['linear'], ['zoom'], 5, 2, 12, 3.5]
				}
			},
			{
				id: ARROW_NOW_LAYER,
				type: 'line',
				source: ARROW_SOURCE,
				filter: ['==', ['get', 'role'], 'now'],
				layout: round,
				paint: {
					'line-color': ink,
					// Same ink, roughly half again as thick: the mark is already
					// twice as wide as a tick, and the weight is what stops a
					// glance mistaking it for one.
					'line-width': ['interpolate', ['linear'], ['zoom'], 5, 3, 12, 5.5]
				}
			}
		];
	}

	/**
	 * Arrow geometry for the selected point, or the empty collection.
	 *
	 * The one rule for *whether* there is an arrow is `motion === null`, and it
	 * is not re-decided here: this reads the very same `cellMotion()` result the
	 * forecast panel prints, so the two cannot disagree. A manifest without a
	 * timestep costs the ticks and nothing else — the builder drops them.
	 *
	 * `radarAgeMin` is the same number the panel prints as "radar data: N min
	 * old", and it is what makes the arrow's marks wall-clock minutes rather
	 * than minutes after a scan that happened half an hour ago. An unknown age
	 * (no manifest yet) is zero: the arrow then means what it used to, which is
	 * the right thing to fall back on.
	 */
	function arrowData(): ArrowCollection {
		const point = nowcast.point;
		const motion = point?.forecast?.motion;
		if (!point || !motion) return emptyArrow();
		return motionArrow({
			lat: point.lat,
			lon: point.lon,
			bearingFromDeg: motion.bearingFromDeg,
			speedKmh: motion.speedKmh,
			timestepMin: nowcast.manifest?.timestep_min ?? 0,
			radarAgeMin: nowcast.radarAgeMin ?? 0
		});
	}

	/**
	 * Create the arrow's source and layers once per style, then keep them fed
	 * with `setData`. No point selected is an empty collection, not a teardown:
	 * clicking around the map should not be churning layers.
	 *
	 * Nothing here is announced to screen readers. The arrow restates what the
	 * forecast panel already says in words ("kommer fra NV · 32 km/t") in an
	 * `aria-live` region, and a second voice saying the same thing is noise.
	 */
	function syncArrow() {
		if (!map || !styleReady) return;
		const data = arrowData();
		const existing = map.getSource(ARROW_SOURCE) as GeoJSONSource | undefined;
		if (existing) {
			existing.setData(data);
			return;
		}
		map.addSource(ARROW_SOURCE, { type: 'geojson', data });
		// Above the radar, below the labels. Each layer is inserted before the
		// same id, which keeps them in the order `arrowLayers` returns them.
		const beforeId = firstSymbolLayerId();
		for (const layer of arrowLayers()) map.addLayer(layer, beforeId);
	}

	/** Drop the arrow's layers and source — the style outlives them otherwise. */
	function removeArrow() {
		if (!map) return;
		for (const id of [ARROW_NOW_LAYER, ARROW_LINE_LAYER, ARROW_HEAD_LAYER, ARROW_CASING_LAYER]) {
			if (map.getLayer(id)) map.removeLayer(id);
		}
		if (map.getSource(ARROW_SOURCE)) map.removeSource(ARROW_SOURCE);
	}

	// Frame changes, cycle changes and style reloads all land here.
	$effect(() => {
		void nowcast.frameIndex;
		void nowcast.frames.length;
		void nowcast.geometry;
		void styleReady;
		syncOverlay();
	});

	/**
	 * The motion arrow redraws whenever the thing it depicts changes: a new
	 * point (a click, or the store re-sampling the old point against a freshly
	 * arrived cycle — `point` is replaced wholesale either way), a manifest
	 * whose cadence changed the tick spacing, a style reload that wiped the
	 * layers, and a theme flip that changes the ink. Clearing the point empties
	 * the collection, so the arrow goes with the panel.
	 *
	 * Radar age is in there too, which means the arrow creeps outwards as the
	 * displayed image ages — every 15 s, on the store's clock tick. That is a
	 * `setData` with a handful of features and no layer churn — far cheaper
	 * than the bitmap swap the radar loop already does on every frame — and the
	 * alternative is a tip that quietly stops meaning "now + 60".
	 */
	$effect(() => {
		void nowcast.point;
		void nowcast.manifest?.timestep_min;
		void nowcast.radarAgeMin;
		void styleReady;
		void theme;
		syncArrow();
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
