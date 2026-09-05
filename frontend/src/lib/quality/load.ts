/**
 * Reading `/nowcast/quality.json` without trusting it.
 *
 * The page's whole subject is honesty about the numbers, so a half-written,
 * half-stale or newer-than-us document must never turn into a confident zero
 * on screen. The rule everywhere below is the same: a section that does not
 * parse becomes null, the rest still renders, and null is rendered as "not
 * measured yet" — never as 0 %, 0 warnings or an empty diagram.
 *
 * A `schema_version` we were not written against is warned about on the
 * console and then parsed anyway, exactly as the manifest loader does: a
 * producer that added a field must not blank the page, and a producer that
 * renamed one loses the sections that no longer parse, which is the honest
 * outcome.
 */
import { apiUrl } from '$lib/nowcast/manifest';
import {
	QUALITY_SCHEMA_VERSION,
	type GaugeWindow,
	type HeadlineReliability,
	type HeadlineWarnings,
	type LiveWindow,
	type PersistenceMargin,
	type QualityHeadline,
	type QualityMethods,
	type QualityReliability,
	type QualityReport,
	type QualityWindows,
	type RadarWindow,
	type RainingNowCheck,
	type ReliabilityBin,
	type ReliabilityCurve,
	type StationCollection,
	type StationFeature,
	type StationProperties,
	type VerifiedEvent
} from './schema';

export const qualityUrl = (): string => apiUrl('/nowcast/quality.json');

type Obj = Record<string, unknown>;

const isObject = (value: unknown): value is Obj =>
	typeof value === 'object' && value !== null && !Array.isArray(value);

/** A finite number, or null for anything else — including null and NaN. */
const num = (value: unknown): number | null =>
	typeof value === 'number' && Number.isFinite(value) ? value : null;

/** A non-empty string, or null. */
const str = (value: unknown): string | null =>
	typeof value === 'string' && value.trim() !== '' ? value : null;

/** A parseable ISO timestamp, or null. Unparseable is missing, not "now". */
const iso = (value: unknown): string | null => {
	const s = str(value);
	return s !== null && Number.isFinite(Date.parse(s)) ? s : null;
};

/**
 * Read `keys` as finite numbers; null if any of them is missing. Used for the
 * blocks where a single absent number makes the whole sentence a guess.
 */
function numbers<K extends string>(source: Obj, keys: readonly K[]): Record<K, number> | null {
	const out = {} as Record<K, number>;
	for (const key of keys) {
		const value = num(source[key]);
		if (value === null) return null;
		out[key] = value;
	}
	return out;
}

function parseRadarWindow(raw: unknown): RadarWindow | null {
	if (!isObject(raw)) return null;
	const from = iso(raw.from);
	const to = iso(raw.to);
	const counts = numbers(raw, ['events', 'points'] as const);
	if (from === null || to === null || counts === null) return null;
	return { from, to, events: counts.events, points: counts.points };
}

function parseGaugeWindow(raw: unknown): GaugeWindow | null {
	if (!isObject(raw)) return null;
	const from = iso(raw.from);
	const to = iso(raw.to);
	const counts = numbers(raw, ['events', 'stations'] as const);
	if (from === null || to === null || counts === null) return null;
	return { from, to, events: counts.events, stations: counts.stations };
}

function parseLiveWindow(raw: unknown): LiveWindow | null {
	if (!isObject(raw)) return null;
	const from = iso(raw.from);
	const to = iso(raw.to);
	const days = num(raw.days);
	if (from === null || to === null || days === null) return null;
	return { days, from, to };
}

function parseWindows(raw: unknown): QualityWindows {
	const source = isObject(raw) ? raw : {};
	return {
		radar: parseRadarWindow(source.radar),
		gauge: parseGaugeWindow(source.gauge),
		live: parseLiveWindow(source.live)
	};
}

function parseHeadlineReliability(raw: unknown): HeadlineReliability | null {
	if (!isObject(raw)) return null;
	const values = numbers(raw, ['lead_min', 'said_pct', 'happened_pct', 'n'] as const);
	return values === null ? null : values;
}

function parseWarnings(raw: unknown): HeadlineWarnings | null {
	if (!isObject(raw)) return null;
	const values = numbers(
		raw,
		[
			'window_days',
			'n_stations',
			'warnings',
			'hits',
			'false_alarms',
			'misses',
			'pod',
			'far'
		] as const
	);
	if (values === null) return null;
	const spread = isObject(raw.lead_error_min)
		? numbers(raw.lead_error_min, ['p25', 'p50', 'p75'] as const)
		: null;
	if (spread === null) return null;
	return { ...values, lead_error_min: spread };
}

function parseMargin(raw: unknown): PersistenceMargin | null {
	if (!isObject(raw)) return null;
	const values = numbers(
		raw,
		['horizon_min', 'csi_advection', 'csi_persistence', 'frames'] as const
	);
	const from = iso(raw.from);
	const to = iso(raw.to);
	if (values === null || from === null || to === null) return null;
	return { ...values, from, to };
}

function parseHeadline(raw: unknown): QualityHeadline {
	const source = isObject(raw) ? raw : {};
	const reliability = isObject(source.reliability) ? source.reliability : {};
	return {
		reliability: {
			radar: parseHeadlineReliability(reliability.radar),
			gauge: parseHeadlineReliability(reliability.gauge)
		},
		warnings: parseWarnings(source.warnings),
		persistence_margin: parseMargin(source.persistence_margin)
	};
}

/**
 * One bin. `forecast_mean` / `observed_freq` stay null when the bin is empty;
 * a bin whose edges or counts are unreadable is dropped from the curve rather
 * than plotted somewhere arbitrary.
 */
function parseBin(raw: unknown): ReliabilityBin | null {
	if (!isObject(raw)) return null;
	const edges = numbers(raw, ['lo', 'hi', 'n', 'eff_n'] as const);
	if (edges === null) return null;
	return {
		lo: edges.lo,
		hi: edges.hi,
		forecast_mean: num(raw.forecast_mean),
		observed_freq: num(raw.observed_freq),
		n: edges.n,
		eff_n: edges.eff_n
	};
}

function parseCurve(raw: unknown): ReliabilityCurve | null {
	if (!isObject(raw)) return null;
	const values = numbers(raw, ['lead_min', 'brier', 'n', 'eff_n'] as const);
	if (values === null || !Array.isArray(raw.bins)) return null;
	const bins = raw.bins.map(parseBin).filter((bin): bin is ReliabilityBin => bin !== null);
	// A curve with nothing left to plot is not a curve.
	if (bins.length === 0) return null;
	return { ...values, bins };
}

function parseCurves(raw: unknown): ReliabilityCurve[] | null {
	if (!Array.isArray(raw)) return null;
	const curves = raw.map(parseCurve).filter((curve): curve is ReliabilityCurve => curve !== null);
	return curves.length === 0 ? null : curves.sort((a, b) => a.lead_min - b.lead_min);
}

function parseReliability(raw: unknown): QualityReliability {
	const source = isObject(raw) ? raw : {};
	return { radar: parseCurves(source.radar), gauge: parseCurves(source.gauge) };
}

function parseRainingNow(raw: unknown): RainingNowCheck | null {
	if (!isObject(raw)) return null;
	const values = numbers(
		raw,
		['n_slots', 'agreement', 'pod', 'far', 'observation_agreement'] as const
	);
	const from = iso(raw.from);
	const to = iso(raw.to);
	if (values === null || from === null || to === null) return null;
	return { ...values, from, to };
}

function parseStation(raw: unknown): StationFeature | null {
	if (!isObject(raw) || !isObject(raw.geometry) || !isObject(raw.properties)) return null;
	const coordinates = raw.geometry.coordinates;
	if (!Array.isArray(coordinates)) return null;
	const lon = num(coordinates[0]);
	const lat = num(coordinates[1]);
	const props = raw.properties;
	const id = str(props.station_id);
	const name = str(props.name);
	const counts = numbers(props, ['n_events', 'warnings'] as const);
	if (lon === null || lat === null || id === null || name === null || counts === null) return null;
	const properties: StationProperties = {
		station_id: id,
		name,
		kind: str(props.kind) ?? '',
		n_events: counts.n_events,
		brier_gauge: num(props.brier_gauge),
		warn_pod: num(props.warn_pod),
		warn_far: num(props.warn_far),
		warnings: counts.warnings,
		raining_now_agreement: num(props.raining_now_agreement)
	};
	return {
		type: 'Feature',
		geometry: { type: 'Point', coordinates: [lon, lat] },
		properties
	};
}

function parseStations(raw: unknown): StationCollection | null {
	if (!isObject(raw) || !Array.isArray(raw.features)) return null;
	const features = raw.features
		.map(parseStation)
		.filter((feature): feature is StationFeature => feature !== null);
	return features.length === 0 ? null : { type: 'FeatureCollection', features };
}

function parseEvent(raw: unknown): VerifiedEvent | null {
	if (!isObject(raw)) return null;
	const id = str(raw.station_id);
	const name = str(raw.name);
	const warnedAt = iso(raw.warned_at_utc);
	const values = numbers(raw, ['eta_min', 'p_rain'] as const);
	const outcome = raw.outcome === 'hit' || raw.outcome === 'false_alarm' ? raw.outcome : null;
	if (id === null || name === null || warnedAt === null || values === null || outcome === null) {
		return null;
	}
	return {
		station_id: id,
		name,
		warned_at_utc: warnedAt,
		eta_min: values.eta_min,
		p_rain: values.p_rain,
		gauge_onset_utc: iso(raw.gauge_onset_utc),
		outcome,
		lead_error_min: num(raw.lead_error_min)
	};
}

function parseEvents(raw: unknown): VerifiedEvent[] | null {
	if (!Array.isArray(raw)) return null;
	const events = raw.map(parseEvent).filter((event): event is VerifiedEvent => event !== null);
	return events.length === 0 ? null : events;
}

function parseMethods(raw: unknown): QualityMethods | null {
	if (!isObject(raw)) return null;
	const wetRule = str(raw.gauge_wet_rule);
	const onsetRule = str(raw.onset_rule);
	const threshold = num(raw.threshold_mm_h);
	const ages = Array.isArray(raw.frame_age_range_min) ? raw.frame_age_range_min : [];
	const ageMin = num(ages[0]);
	const ageMax = num(ages[1]);
	const rule = isObject(raw.subscriber_rule)
		? numbers(
				raw.subscriber_rule,
				['threshold_pct', 'lead_min', 'rearm_after_min', 'persistence_obs'] as const
			)
		: null;
	const sources = isObject(raw.sources) ? raw.sources : {};
	const radar = str(sources.radar);
	const gauges = str(sources.gauges);
	if (
		wetRule === null ||
		onsetRule === null ||
		threshold === null ||
		ageMin === null ||
		ageMax === null ||
		rule === null ||
		radar === null ||
		gauges === null
	) {
		return null;
	}
	return {
		gauge_wet_rule: wetRule,
		onset_rule: onsetRule,
		threshold_mm_h: threshold,
		frame_age_range_min: [ageMin, ageMax],
		subscriber_rule: rule,
		sources: { radar, gauges }
	};
}

/**
 * Parse a quality document. Null only when there is nothing to render at all —
 * not an object, or without a readable `generated_at_utc`, which is the one
 * field the page needs to say when the numbers are from.
 */
export function parseQuality(raw: unknown): QualityReport | null {
	if (!isObject(raw)) return null;
	const generatedAt = iso(raw.generated_at_utc);
	if (generatedAt === null) return null;

	const version = num(raw.schema_version);
	if (version !== QUALITY_SCHEMA_VERSION) {
		// Keep rendering whatever still parses rather than blanking the page.
		console.warn(
			`quality.json schema_version ${String(raw.schema_version)} != ${QUALITY_SCHEMA_VERSION}`
		);
	}

	return {
		schema_version: version ?? 0,
		generated_at_utc: generatedAt,
		windows: parseWindows(raw.windows),
		headline: parseHeadline(raw.headline),
		reliability: parseReliability(raw.reliability),
		raining_now: parseRainingNow(raw.raining_now),
		stations: parseStations(raw.stations),
		events: parseEvents(raw.events),
		methods: parseMethods(raw.methods)
	};
}

/** Fetch and parse the newest quality document. */
export async function fetchQuality(signal?: AbortSignal): Promise<QualityReport> {
	const res = await fetch(qualityUrl(), { signal, cache: 'no-cache' });
	if (!res.ok) throw new Error(`quality: HTTP ${res.status}`);
	const parsed = parseQuality(await res.json());
	if (parsed === null) throw new Error('quality: nothing readable in the document');
	return parsed;
}
