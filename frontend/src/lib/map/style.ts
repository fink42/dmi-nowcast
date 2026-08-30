/**
 * The MapLibre style: Protomaps basemap layers over our own self-hosted
 * pmtiles archive, with the glyphs and sprites served from the same origin.
 * Nothing here talks to a third-party tile server, which is the whole point
 * of the Protomaps decision — no usage policy, no external runtime
 * dependency, and the map keeps working as long as our origin does.
 */
import { layers, namedFlavor } from '@protomaps/basemaps';
import type { StyleSpecification } from 'maplibre-gl';
import { base } from '$app/paths';

export type Theme = 'light' | 'dark';

/**
 * Absolute URL for a file in `static/` — the pmtiles protocol needs one.
 * Deliberately string concatenation and not `new URL(...)`: the glyphs URL
 * carries `{fontstack}` / `{range}` placeholders that MapLibre substitutes
 * itself, and URL-encoding the braces makes it reject the style.
 */
const origin = (): string =>
	typeof location === 'undefined' ? base : `${location.origin}${base}`.replace(/\/$/, '');

const asset = (path: string): string => `${origin()}${path}`;

export const BASEMAP_URL = () => `pmtiles://${asset('/basemap.pmtiles')}`;

export interface StyleAttribution {
	/** "Radar data: DMI" — required by DMI's free-data terms. */
	radar: string;
	/** "© OpenStreetMap contributors". */
	osm: string;
}

/**
 * Build the style for a theme. `lang` picks the label language Protomaps
 * ships (Danish names where OSM has them, else the local name).
 */
export function buildStyle(
	theme: Theme,
	lang: string,
	attribution: StyleAttribution
): StyleSpecification {
	const credit =
		`${attribution.radar} · ` +
		`<a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">${attribution.osm}</a> · ` +
		`<a href="https://protomaps.com" target="_blank" rel="noreferrer">Protomaps</a>`;
	return {
		version: 8,
		glyphs: asset('/basemap-assets/fonts/{fontstack}/{range}.pbf'),
		sprite: asset(`/basemap-assets/sprites/v4/${theme}`),
		sources: {
			protomaps: {
				type: 'vector',
				url: BASEMAP_URL(),
				attribution: credit
			}
		},
		layers: layers('protomaps', namedFlavor(theme), { lang })
	};
}

/** The viewer's colour scheme, defaulting to light where it cannot be read. */
export function preferredTheme(): Theme {
	if (typeof matchMedia === 'undefined') return 'light';
	try {
		return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
	} catch {
		return 'light';
	}
}
