<script lang="ts">
	/**
	 * The picture: the composite at the cursor, the 1 km disc the verdict was
	 * taken over, the gauges that voted, and the flow the estimate stood on.
	 *
	 * Same machinery as the public map (`$lib/components/MapView.svelte`) —
	 * one image source whose bitmap swaps per frame, the bundled MapLibre
	 * worker, our own pmtiles basemap — because the frames are reprojected by
	 * the same `map/warp.ts`, and the naive four-corner placement misplaces
	 * rain by ~15 km over Denmark. At this zoom that is the difference between
	 * a cell over the station and a cell over the next town.
	 *
	 * Three things here exist to stop the map misleading a reviewer:
	 *
	 *  - **Opacity is not intensity.** The overlay fades light rain on purpose
	 *    (the composite over-reads faint echo), so the colours carry no legend
	 *    they could support. Every number in the read-out comes from the
	 *    grayscale `observed.png` through `store.sampleAt`, and the read-out
	 *    says so.
	 *  - **A dot that did not report is not a dry dot.** Station and neighbour
	 *    markers are coloured from `SlotState` at the cursor, and `unknown`
	 *    gets its own fill AND a `?` over it — two channels, because colour
	 *    alone is the same mistake in a different medium.
	 *  - **The disc is drawn.** The service acted on the p90 over a 1 km disc,
	 *    not on the pixel under the marker; a reviewer judging the warning
	 *    against one pixel is judging a different number.
	 */
	import { onMount } from 'svelte';
	import {
		addProtocol,
		removeProtocol,
		setWorkerUrl,
		Map as MapLibreMap,
		NavigationControl,
		type GeoJSONSource,
		type ImageSource,
		type LayerSpecification
	} from 'maplibre-gl';
	// MapLibre 6 builds its worker URL at runtime, which no bundler can
	// follow. Let Vite bundle the worker and hand us its URL.
	import maplibreWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';
	import { Protocol } from 'pmtiles';
	import 'maplibre-gl/dist/maplibre-gl.css';
	import { emptyArrow, motionArrow, type ArrowCollection } from '$lib/map/arrow';
	import { buildStyle, preferredTheme, type Theme } from '$lib/map/style';
	import { distanceM } from '$lib/review/geometry';
	import { review } from '$lib/review/store.svelte';
	import type { DiscSample } from '$lib/review/sample';
	import { mmHText, NOT_MEASURED, numberText, utcTime } from './format';

	const OVERLAY_SOURCE = 'review-overlay';
	const OVERLAY_LAYER = 'review-overlay-layer';
	const DISC_SOURCE = 'review-disc';
	const DISC_FILL = 'review-disc-fill';
	const DISC_LINE = 'review-disc-line';
	const STATION_SOURCE = 'review-stations';
	const STATION_CIRCLE = 'review-station-circle';
	const STATION_QUERY = 'review-station-query';
	const STATION_LABEL = 'review-station-label';
	const ARROW_SOURCE = 'review-arrow';
	const ARROW_CASING_LAYER = 'review-arrow-casing';
	const ARROW_HEAD_LAYER = 'review-arrow-head';
	const ARROW_LINE_LAYER = 'review-arrow-line';
	const ARROW_NOW_LAYER = 'review-arrow-now';
	/** 1×1 transparent PNG — the placeholder an image source must be born with. */
	const BLANK_PNG =
		'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
	const DENMARK: [[number, number], [number, number]] = [
		[7.9, 54.5],
		[15.3, 57.85]
	];
	/** Close enough to read a 2 km pixel, wide enough to see where rain is. */
	const EVENT_ZOOM = 8.6;

	const EMPTY_GEOJSON = { type: 'FeatureCollection' as const, features: [] };

	let container: HTMLDivElement;
	let map: MapLibreMap | null = null;
	let styleReady = $state(false);
	/**
	 * The theme is read once. A dev tool does not need to follow a mid-session
	 * colour-scheme flip, and the style rebuild that would take costs the
	 * overlay, the disc and the dots a re-attach for nothing.
	 */
	let theme: Theme = 'light';

	interface Reading {
		lat: number;
		lon: number;
		stamp: string | null;
		radarTsUtc: string | null;
		pixelMmH: number | null;
		disc: DiscSample | null;
		/** Metres from the event's station — is this even in the disc? */
		fromStationM: number | null;
	}
	let reading = $state<Reading | null>(null);

	const discRadiusM = $derived(review.manifest?.truth?.radar_disc_radius_m ?? null);

	onMount(() => {
		setWorkerUrl(maplibreWorkerUrl);
		const protocol = new Protocol();
		addProtocol('pmtiles', protocol.tile);
		theme = preferredTheme();

		map = new MapLibreMap({
			container,
			// English labels: the tool is internal and its vocabulary is English
			// throughout, so a Danish basemap under an English read-out would be
			// the only bilingual thing on the page.
			style: buildStyle(theme, 'en', {
				radar: 'Radar data: DMI',
				osm: '© OpenStreetMap contributors'
			}),
			bounds: DENMARK,
			fitBoundsOptions: { padding: 12 },
			minZoom: 5,
			maxZoom: 13,
			attributionControl: { compact: true }
		});
		map.addControl(new NavigationControl({ showCompass: false }), 'top-right');
		map.on('style.load', () => {
			styleReady = true;
			syncOverlay();
			syncDisc();
			syncStations();
			syncArrow();
		});
		// Sampling follows the pointer rather than a click: the question
		// "what does the grid say *there*" gets asked a hundred times per
		// event, and a click would also fight the map's own drag.
		map.on('mousemove', (event) => sample(event.lngLat.lat, event.lngLat.lng));

		return () => {
			map?.remove();
			map = null;
			removeProtocol('pmtiles');
		};
	});

	/** The lowest label layer: everything we draw goes under the place names. */
	function firstSymbolLayerId(): string | undefined {
		return map?.getStyle().layers.find((layer) => layer.type === 'symbol')?.id;
	}

	/**
	 * The composite at the cursor. `store.currentBitmap` is already warped
	 * into Mercator and `store.corners` is that target's own footprint, so the
	 * two always belong together.
	 */
	function syncOverlay(): void {
		if (!map || !styleReady) return;
		const corners = review.corners;
		const bitmap = review.currentBitmap;
		if (corners === null) return;
		const existing = map.getSource(OVERLAY_SOURCE) as ImageSource | undefined;
		if (existing) {
			// No bitmap yet (the frame is still decoding, or the bundle has no
			// picture for this stamp): show the blank rather than the previous
			// frame, which would be a different minute's weather under this
			// minute's stamp.
			existing.updateImage(
				bitmap === null ? { url: BLANK_PNG, coordinates: corners } : { image: bitmap, coordinates: corners }
			);
			return;
		}
		map.addSource(OVERLAY_SOURCE, { type: 'image', url: BLANK_PNG, coordinates: corners });
		if (bitmap !== null) {
			(map.getSource(OVERLAY_SOURCE) as ImageSource).updateImage({ image: bitmap });
		}
		map.addLayer(
			{
				id: OVERLAY_LAYER,
				type: 'raster',
				source: OVERLAY_SOURCE,
				// `nearest`, where the public map smooths: the reviewer is judging
				// which pixels were wet, and interpolation invents a gradient
				// across a 2 km cell that the grid behind the read-out does not
				// have.
				paint: { 'raster-opacity': 0.85, 'raster-fade-duration': 0, 'raster-resampling': 'nearest' }
			},
			firstSymbolLayerId()
		);
	}

	/** The verdict disc: what the service's `raining_now` actually sampled. */
	function syncDisc(): void {
		if (!map || !styleReady) return;
		const disc = review.discFeature;
		const data = disc === null ? EMPTY_GEOJSON : disc;
		const existing = map.getSource(DISC_SOURCE) as GeoJSONSource | undefined;
		if (existing) {
			existing.setData(data);
			return;
		}
		map.addSource(DISC_SOURCE, { type: 'geojson', data });
		const ink = theme === 'dark' ? '#f2f5f8' : '#101820';
		map.addLayer(
			{
				id: DISC_FILL,
				type: 'fill',
				source: DISC_SOURCE,
				paint: { 'fill-color': ink, 'fill-opacity': 0.08 }
			},
			firstSymbolLayerId()
		);
		map.addLayer(
			{
				id: DISC_LINE,
				type: 'line',
				source: DISC_SOURCE,
				paint: { 'line-color': ink, 'line-width': 1.5, 'line-dasharray': [2, 2] }
			},
			firstSymbolLayerId()
		);
	}

	/**
	 * The station and its neighbours, coloured by what they said at the
	 * cursor.
	 *
	 * `unknown` is a fill of its own AND a `?` on top. A map that paints "did
	 * not report" the same as "reported nothing" is the `known`/`wet` mistake
	 * in colour, and it is the one that turns a hole in the gauge record into
	 * evidence for a false alarm.
	 */
	function syncStations(): void {
		if (!map || !styleReady) return;
		const data = review.stationFeatures;
		const existing = map.getSource(STATION_SOURCE) as GeoJSONSource | undefined;
		if (existing) {
			existing.setData(data);
			return;
		}
		map.addSource(STATION_SOURCE, { type: 'geojson', data });
		const ink = theme === 'dark' ? '#f2f5f8' : '#101820';
		const surface = theme === 'dark' ? '#141a21' : '#ffffff';
		map.addLayer({
			id: STATION_CIRCLE,
			type: 'circle',
			source: STATION_SOURCE,
			paint: {
				'circle-color': [
					'match',
					['get', 'state'],
					'wet',
					'#2f7fd6',
					'dry',
					surface,
					// Unknown: a colour that is neither of the other two, and never
					// the dry one.
					'#8a6ea8'
				],
				'circle-radius': ['case', ['==', ['get', 'role'], 'station'], 9, 6],
				'circle-stroke-width': ['case', ['==', ['get', 'role'], 'station'], 3, 1.5],
				'circle-stroke-color': ink
			}
		});
		map.addLayer({
			id: STATION_QUERY,
			type: 'symbol',
			source: STATION_SOURCE,
			filter: ['==', ['get', 'state'], 'unknown'],
			layout: {
				'text-field': '?',
				'text-font': ['Noto Sans Medium'],
				'text-size': 11,
				'text-allow-overlap': true
			},
			paint: { 'text-color': surface, 'text-halo-color': ink, 'text-halo-width': 0.6 }
		});
		map.addLayer({
			id: STATION_LABEL,
			type: 'symbol',
			source: STATION_SOURCE,
			layout: {
				'text-field': ['get', 'name'],
				'text-font': ['Noto Sans Regular'],
				'text-size': 11,
				'text-offset': [0, 1.1],
				'text-anchor': 'top'
			},
			paint: { 'text-color': ink, 'text-halo-color': surface, 'text-halo-width': 1.2 }
		});
	}

	/**
	 * The upwind arrow for the estimate standing at the cursor.
	 *
	 * `store.upwind` is null when the cycle had no usable motion — below the
	 * flow floor the pipeline writes NaN rather than a direction — and then
	 * nothing is drawn. Inventing an arrow there would be the tool making up
	 * the evidence it exists to check.
	 *
	 * The age fed to `motionArrow` is the decision's OWN `frame_age_min`: the
	 * arrow belongs to that estimate, and the marks then sit where the flow
	 * had carried the rain to by the time the estimate was made.
	 */
	function arrowData(): ArrowCollection {
		const station = review.detail?.station;
		const upwind = review.upwind;
		if (!station || upwind === null) return emptyArrow();
		return motionArrow({
			lat: station.lat,
			lon: station.lon,
			bearingFromDeg: upwind.bearingFromDeg,
			speedKmh: upwind.speedKmh,
			timestepMin: review.manifest?.frames?.cadence_min ?? 0,
			radarAgeMin: review.decision?.frame_age_min ?? 0
		});
	}

	/** The arrow's layers, bottom to top — `MapView.svelte`'s look. */
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
				paint: { 'line-color': casing, 'line-width': 5.5 }
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
				filter: ['!=', ['get', 'role'], 'now'],
				layout: round,
				paint: { 'line-color': ink, 'line-width': 2.5 }
			},
			{
				id: ARROW_NOW_LAYER,
				type: 'line',
				source: ARROW_SOURCE,
				filter: ['==', ['get', 'role'], 'now'],
				layout: round,
				paint: { 'line-color': ink, 'line-width': 4 }
			}
		];
	}

	function syncArrow(): void {
		if (!map || !styleReady) return;
		const data = arrowData();
		const existing = map.getSource(ARROW_SOURCE) as GeoJSONSource | undefined;
		if (existing) {
			existing.setData(data);
			return;
		}
		map.addSource(ARROW_SOURCE, { type: 'geojson', data });
		const beforeId = firstSymbolLayerId();
		for (const layer of arrowLayers()) map.addLayer(layer, beforeId);
	}

	/**
	 * What the grid says at a point — the pixel AND the disc p90, because the
	 * p90 is the statistic the service acted on and the pixel is what the
	 * pointer is over. Quoting one for the other is how a warning gets judged
	 * against a number that never existed.
	 */
	function sample(lat: number, lon: number): void {
		const station = review.detail?.station ?? null;
		const taken = review.sampleAt(lat, lon);
		reading = {
			lat,
			lon,
			stamp: taken.stamp,
			radarTsUtc: taken.radarTsUtc,
			pixelMmH: taken.pixelMmH,
			disc: taken.disc,
			fromStationM: station === null ? null : distanceM(station.lat, station.lon, lat, lon)
		};
	}

	// The frame on screen, the cursor's own dots, the disc and the arrow. Each
	// effect names what it watches, exactly as the public map does.
	$effect(() => {
		void review.bitmapVersion;
		void review.cursorFrame;
		void review.corners;
		void styleReady;
		syncOverlay();
	});

	$effect(() => {
		void review.discFeature;
		void styleReady;
		syncDisc();
	});

	$effect(() => {
		void review.stationFeatures;
		void styleReady;
		syncStations();
	});

	$effect(() => {
		void review.upwind;
		void review.decision;
		void styleReady;
		syncArrow();
	});

	/** A new event re-centres the map; scrubbing inside one never moves it. */
	$effect(() => {
		const station = review.detail?.station;
		// `styleReady` is in here so an event that loaded before the map did
		// still gets centred, rather than leaving the reviewer looking at all
		// of Denmark.
		void styleReady;
		if (!map || !station) return;
		map.easeTo({ center: [station.lon, station.lat], zoom: EVENT_ZOOM, duration: 400 });
	});
</script>

<div class="map-pane">
	<div class="map" bind:this={container} role="application" aria-label="Radar at the cursor"></div>

	<div class="readout">
		{#if reading === null}
			<p class="quiet">Move the pointer over the map to read the grid.</p>
		{:else}
			<p class="head">
				<strong>{mmHText(reading.pixelMmH, 2) ?? NOT_MEASURED}</strong>
				<span class="quiet">at the pointer</span>
			</p>
			<p>
				disc p90
				<strong>{mmHText(reading.disc?.p90MmH ?? null, 2) ?? NOT_MEASURED}</strong>
				{#if reading.disc !== null}
					<span class="quiet">
						· max {mmHText(reading.disc.maxMmH, 2) ?? NOT_MEASURED} · {reading.disc.nValid}/{reading
							.disc.nPixels} pixels with a value
					</span>
				{/if}
			</p>
			{#if reading.disc?.singlePixel}
				<p class="quiet">
					The disc is one pixel of the 2 km product grid, so this “p90” is that
					pixel's own block p90 — not a distribution over the disc, and not the
					500 m number the service computed.
				</p>
			{/if}
			{#if reading.fromStationM !== null && discRadiusM !== null}
				<p class="quiet">
					{numberText(reading.fromStationM / 1000, 2)} km from the station —
					{reading.fromStationM <= discRadiusM ? 'inside' : 'outside'} the {discRadiusM} m
					verdict disc.
				</p>
			{/if}
			<p class="quiet">
				from {reading.stamp ?? 'no frame'}
				{#if reading.radarTsUtc}({utcTime(reading.radarTsUtc)}){/if} · numbers come from
				the grayscale grid, never from the colours: light rain is faded on purpose.
			</p>
		{/if}
	</div>

	<div class="legend" aria-hidden="true">
		<span><i class="dot wet"></i>wet</span>
		<span><i class="dot dry"></i>dry</span>
		<span><i class="dot unknown"></i>did not report</span>
		<span class="quiet">dashed circle = the verdict disc</span>
	</div>
</div>

<style>
	.map-pane {
		position: relative;
		flex: 1 1 auto;
		min-height: 0;
	}

	.map {
		position: absolute;
		inset: 0;
		background: var(--map-bg);
	}

	.readout {
		position: absolute;
		left: 0.5rem;
		bottom: 0.5rem;
		max-width: 22rem;
		background: color-mix(in srgb, var(--surface) 92%, transparent);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 0.4rem 0.55rem;
		font-size: 0.75rem;
		box-shadow: var(--shadow);
	}

	.readout p {
		margin: 0 0 0.15rem;
	}

	.readout .head strong {
		font-size: 1rem;
		font-variant-numeric: tabular-nums;
	}

	.quiet {
		color: var(--muted);
	}

	.legend {
		position: absolute;
		left: 0.5rem;
		top: 0.5rem;
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
		align-items: center;
		background: color-mix(in srgb, var(--surface) 92%, transparent);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 0.2rem 0.5rem;
		font-size: 0.7rem;
	}

	.dot {
		display: inline-block;
		width: 0.6rem;
		height: 0.6rem;
		border-radius: 50%;
		border: 1px solid var(--ink);
		margin-right: 0.25rem;
		vertical-align: -1px;
	}

	.dot.wet {
		background: #2f7fd6;
	}

	.dot.dry {
		background: var(--surface);
	}

	.dot.unknown {
		background: #8a6ea8;
	}
</style>
