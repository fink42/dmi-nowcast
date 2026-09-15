/**
 * Reading a review bundle without trusting it.
 *
 * The bundle is built on a VM by a script that is still changing, carried to
 * a laptop by hand, and then served off disk. Every one of those steps can
 * hand the page a document that is half-written, half-stale, or newer than
 * this client — and the whole point of the tool is to judge evidence, so a
 * section we cannot read must come back as `null` and be rendered as "not in
 * this bundle". A confident zero here would be a reviewer tagging
 * `fa_gauge_missed_it` on a gauge series that simply failed to parse.
 *
 * The rule is `quality/load.ts`'s, applied to a bigger document: a section
 * that does not parse becomes null, the rest still renders, nothing throws
 * into the UI, and a `schema_version` we were not written against is warned
 * about on the console and then parsed anyway. A producer that *added* a
 * field must not blank the page; a producer that *renamed* one loses the
 * sections that no longer parse, which is the honest outcome.
 *
 * Every parser returns `schema.ts`'s own type. Each block it could not read
 * is null there, which is what lets this file be defensive without
 * inventing a widened shape of its own — one contract, one spelling of
 * every field.
 */
import {
	REVIEW_SCHEMA_VERSION,
	type Annotation,
	type BuilderBlock,
	type Caveat,
	type CorpusBlock,
	type Decision,
	type DecisionGap,
	type DualTruthBlock,
	type DualTruthClass,
	type EventDetail,
	type FeaturesBlock,
	type EventIndex,
	type FrameRef,
	type FramesBlock,
	type GaugeBlock,
	type GridBlock,
	type IndexRow,
	type Manifest,
	type NeighbourRef,
	type NeighboursBlock,
	type NotificationMarker,
	type ObservedEncoding,
	type OutcomeClass,
	type OverlayEncoding,
	type Prologue,
	type RadarDiscBlock,
	type ReplayTrace,
	type RuleBlock,
	type SamplingBlock,
	type Slot,
	type StationBlock,
	type StoredRow,
	type TruthBlock,
	type Vocabulary,
	type VocabularyTag,
	type WindowBlock,
	type WindowGeometry
} from './schema';

/**
 * Where the bundle is served. `scripts/review_server.py` binds to loopback
 * and vite proxies this prefix to it (vite.config.ts), so the page fetches
 * same-origin paths exactly as the rest of the app does.
 */
export const REVIEW_DATA_PREFIX = '/review-data/';

/**
 * The re-arm window the service shipped with, in minutes. Only ever used
 * when a bundle's prologue does not state its own — the rule is
 * configurable, and a countdown drawn against the wrong constant is wrong
 * silently.
 */
export const DEFAULT_REARM_AFTER_MIN = 60;

/**
 * A path inside the bundle → the URL to fetch it from.
 *
 * Paths come out of the bundle's own documents (`IndexRow.detail`,
 * `FrameRef.overlay`), so a `..` segment in one is either a builder bug or
 * something worse. The server resolves and refuses those with a 403; this
 * drops them client-side too, so a broken path fetches the bundle root
 * instead of walking out of it.
 */
export function bundleUrl(path: string): string {
	const parts = String(path)
		.split('/')
		.filter((part) => part !== '' && part !== '.' && part !== '..');
	return REVIEW_DATA_PREFIX + parts.join('/');
}

type Obj = Record<string, unknown>;

const isObject = (value: unknown): value is Obj =>
	typeof value === 'object' && value !== null && !Array.isArray(value);

/** A finite number, or null for anything else — including null and NaN. */
const num = (value: unknown): number | null =>
	typeof value === 'number' && Number.isFinite(value) ? value : null;

/** A non-empty string, or null. */
const str = (value: unknown): string | null =>
	typeof value === 'string' && value.trim() !== '' ? value : null;

/** A real boolean, or null. `0` and `"false"` are not booleans. */
const bool = (value: unknown): boolean | null =>
	typeof value === 'boolean' ? value : null;

/** A parseable ISO timestamp, or null. Unparseable is missing, not "now". */
const iso = (value: unknown): string | null => {
	const s = str(value);
	return s !== null && Number.isFinite(Date.parse(s)) ? s : null;
};

/** Every readable string of an array; a non-array is an empty list. */
const strings = (value: unknown): string[] =>
	Array.isArray(value) ? value.map(str).filter((s): s is string => s !== null) : [];

/** `Record<string, string>` over whatever was readable. Never null — an
 * empty documentation map and a missing one render identically. */
function stringMap(value: unknown): Record<string, string> {
	if (!isObject(value)) return {};
	const out: Record<string, string> = {};
	for (const [key, raw] of Object.entries(value)) {
		const text = str(raw);
		if (text !== null) out[key] = text;
	}
	return out;
}

function numberMap(value: unknown): Record<string, number> {
	if (!isObject(value)) return {};
	const out: Record<string, number> = {};
	for (const [key, raw] of Object.entries(value)) {
		const n = num(raw);
		if (n !== null) out[key] = n;
	}
	return out;
}

/** Lead (a string key) → probability, with unserved leads kept as null. */
function probabilityMap(value: unknown): Record<string, number | null> {
	if (!isObject(value)) return {};
	const out: Record<string, number | null> = {};
	for (const [key, raw] of Object.entries(value)) out[key] = num(raw);
	return out;
}

/**
 * Read `keys` as finite numbers; null if any is missing. For the blocks
 * where one absent number makes the whole block a guess — a grid with no
 * pixel scale is not a grid.
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

const OUTCOME_CLASSES: readonly OutcomeClass[] = [
	'false_alarm',
	'miss',
	'miss_late',
	'late',
	'hit',
	'uncovered'
];

const DUAL_TRUTH_CLASSES: readonly DualTruthClass[] = [
	'both_wet',
	'radar_wet_gauge_dry',
	'gauge_wet_radar_dry',
	'both_dry'
];

const outcomeClass = (value: unknown): OutcomeClass | null =>
	OUTCOME_CLASSES.find((known) => known === value) ?? null;

const dualTruthClass = (value: unknown): DualTruthClass | null =>
	DUAL_TRUTH_CLASSES.find((known) => known === value) ?? null;

/**
 * A `schema_version` that is not ours: warn once per document and carry on.
 * Refusing outright would blank the page over an added field.
 */
function checkVersion(what: string, raw: unknown): number {
	const version = num(raw);
	if (version !== REVIEW_SCHEMA_VERSION) {
		console.warn(`${what} schema_version ${String(raw)} != ${REVIEW_SCHEMA_VERSION}`);
	}
	return version ?? 0;
}

// ---------------------------------------------------------------------------
// manifest.json
// ---------------------------------------------------------------------------

function parseBuilder(raw: unknown): BuilderBlock | null {
	if (!isObject(raw)) return null;
	const script = str(raw.script);
	const commit = str(raw.git_commit);
	if (script === null || commit === null) return null;
	return {
		script,
		git_commit: commit,
		// A builder that did not say is not clean: provenance defaults to the
		// answer that makes a reviewer look twice.
		git_dirty: bool(raw.git_dirty) ?? true,
		argv: strings(raw.argv),
		host: str(raw.host) ?? '',
		python: str(raw.python) ?? ''
	};
}

/**
 * Where the rows came from. Null for the `{}` a `--fixture` bundle writes,
 * which is honest rather than broken: that bundle has no corpus behind it,
 * and the page says "synthetic" rather than quoting a provenance it does
 * not have.
 */
function parseCorpus(raw: unknown): CorpusBlock | null {
	if (!isObject(raw)) return null;
	const corpusDir = str(raw.corpus_dir);
	if (corpusDir === null) return null;
	return {
		decisions_dirs: strings(raw.decisions_dirs),
		decisions_labels: strings(raw.decisions_labels),
		decisions_precedence: str(raw.decisions_precedence) ?? '',
		corpus_dir: corpusDir,
		files: num(raw.files) ?? 0,
		files_skipped: num(raw.files_skipped) ?? 0,
		rows_read: num(raw.rows_read) ?? 0,
		rows_kept: num(raw.rows_kept) ?? 0,
		duplicates_dropped: num(raw.duplicates_dropped) ?? 0,
		leads: Array.isArray(raw.leads)
			? raw.leads.map(num).filter((lead): lead is number => lead !== null)
			: [],
		window_from_utc: iso(raw.window_from_utc),
		window_to_utc: iso(raw.window_to_utc),
		// Absent on the honest path: a filler that did not run has no counts,
		// and an empty object would read as "it ran and filled nothing".
		probability_fill: isObject(raw.probability_fill)
			? numberMap(raw.probability_fill)
			: null
	};
}

/**
 * The rule the events were re-decided under. The engine constants are
 * required: `armStateAt` counts minutes against `rearm_after_min`, and a
 * default in place of the bundle's own number would date the arm band
 * wrongly without saying so.
 */
function parseRule(raw: unknown): RuleBlock | null {
	if (!isObject(raw)) return null;
	const timings = numbers(
		raw,
		[
			'lead_min',
			'persistence_obs',
			'rearm_after_min',
			'raining_now_mm_h',
			'raining_now_eta_min',
			'coverage_gap_min',
			'tolerance_min',
			'min_useful_lead_min'
		] as const
	);
	if (timings === null) return null;
	const threshold = num(raw.threshold_pct);
	if (threshold === null) return null;
	const source = raw.source === 'served' || raw.source === 'lomo' ? raw.source : null;
	const probability = raw.probability === 'postprocess' || raw.probability === 'curve'
		? raw.probability
		: null;
	if (source === null || probability === null) return null;
	const provenance =
		raw.probability_provenance === 'out_of_fold' ||
		raw.probability_provenance === 'in_sample_fill' ||
		raw.probability_provenance === 'stored' ||
		raw.probability_provenance === 'mixed'
			? raw.probability_provenance
			: // Not stated is not "clean". `out_of_fold` is the one value that
				// makes the sample honest, so it is never the fallback.
				'mixed';
	return {
		source,
		// Absent means the producer never CLAIMED the thresholds were held
		// out, and an unclaimed hold-out is an in-sample one. Defaulting the
		// other way would let a bundle look honest by omission.
		held_out: bool(raw.held_out) ?? false,
		threshold_pct: threshold,
		threshold_source: str(raw.threshold_source) ?? '',
		fold_thresholds: Array.isArray(raw.fold_thresholds)
			? (raw.fold_thresholds as RuleBlock['fold_thresholds'])
			: null,
		probability,
		probability_column: str(raw.probability_column) ?? '',
		probability_provenance: provenance,
		...timings
	};
}

/**
 * The truth rule. The numbers are required together: the disc radius, the
 * threshold and the onset rule are what the reviewer is judging the
 * verdict against, and a block quoting three of them and defaulting the
 * fourth would put an unlabelled guess in the methods panel.
 */
function parseTruth(raw: unknown): TruthBlock | null {
	if (!isObject(raw)) return null;
	const values = numbers(
		raw,
		[
			'dry_min',
			'onset_min_mm',
			'tolerance_min',
			'lead_min',
			'min_useful_lead_min',
			'radar_disc_radius_m',
			'radar_threshold_mm_h',
			'neighbour_radius_km',
			'min_known_slots'
		] as const
	);
	if (values === null) return null;
	return {
		...values,
		radar_source: str(raw.radar_source) ?? '',
		dead_gauges: strings(raw.dead_gauges),
		known_until: stringMap(raw.known_until),
		suspect_gauge_months: strings(raw.suspect_gauge_months)
	};
}

function parseSampling(raw: unknown): SamplingBlock | null {
	if (!isObject(raw)) return null;
	const seed = num(raw.seed);
	const hash = str(raw.population_hash);
	// Without the seed and the hash the draw cannot be reproduced or
	// compared with a second bundle, which is the whole point of recording it.
	if (seed === null || hash === null) return null;
	const cells = Array.isArray(raw.cells)
		? raw.cells
				.map((cell) => {
					if (!isObject(cell)) return null;
					const counts = numbers(cell, ['population', 'drawn'] as const);
					const stratum = str(cell.stratum);
					if (counts === null || stratum === null) return null;
					return {
						stratum,
						keys: strings(cell.keys),
						group: str(cell.group) ?? '',
						...counts
					};
				})
				.filter((cell): cell is SamplingBlock['cells'][number] => cell !== null)
		: [];
	return {
		seed,
		strata: strings(raw.strata),
		floor_per_cell: num(raw.floor_per_cell) ?? 0,
		targets: numberMap(raw.targets),
		drawn: numberMap(raw.drawn),
		population: numberMap(raw.population),
		total_drawn: num(raw.total_drawn) ?? 0,
		population_hash: hash,
		collapsed_late_pairs: num(raw.collapsed_late_pairs) ?? 0,
		control_groups: strings(raw.control_groups),
		cells
	};
}

/** The window's geometry: minutes either side, and the imagery pad. */
function parseWindowGeometry(raw: unknown): WindowGeometry | null {
	if (!isObject(raw)) return null;
	const values = numbers(raw, ['decision_min', 'frame_pad_min'] as const);
	return values === null ? null : values;
}

/**
 * The feature-gap audit. Null when the producer did not run one, which is
 * itself worth saying: "we did not check" is not "there was no gap".
 *
 * Named apart from `parseFeatureColumns` deliberately: this is the
 * bundle-wide audit of whether the Phase-H columns were THERE, and that one
 * reads the columns themselves off a single decision row.
 */
function parseFeatureAudit(raw: unknown): FeaturesBlock | null {
	if (!isObject(raw)) return null;
	const probe = str(raw.probe_column);
	if (probe === null) return null;
	const rowsByDay: FeaturesBlock['rows_by_day'] = {};
	if (isObject(raw.rows_by_day)) {
		for (const [day, value] of Object.entries(raw.rows_by_day)) {
			if (!isObject(value)) continue;
			const counts = numbers(value, ['rows', 'with_features'] as const);
			if (counts !== null) rowsByDay[day] = counts;
		}
	}
	return {
		probe_column: probe,
		gap_share: num(raw.gap_share) ?? 0,
		gap_days: strings(raw.gap_days),
		// "Not stated" is not "excluded": a bundle that does not say whether
		// it kept the feature-gap days is one the reviewer must treat as
		// having kept them.
		allowed: bool(raw.allowed) ?? true,
		rows_by_day: rowsByDay,
		documentation: stringMap(raw.documentation)
	};
}

/**
 * The grid, all seven fields or nothing. A half-read grid does not misplace
 * the map by a little — it puts the rain in the wrong country (see
 * `nowcast/sampler.ts` for the inverse), so there is no partial version of
 * this block worth having.
 */
function parseGrid(raw: unknown): GridBlock | null {
	if (!isObject(raw)) return null;
	const values = numbers(
		raw,
		['x_ul_m', 'y_ul_m', 'pixel_scale_x_m', 'pixel_scale_y_m', 'downsample_factor'] as const
	);
	const proj4 = str(raw.proj4);
	const shape = Array.isArray(raw.shape) ? raw.shape : [];
	const rows = num(shape[0]);
	const cols = num(shape[1]);
	if (values === null || proj4 === null || rows === null || cols === null) return null;
	return { proj4, shape: [rows, cols], ...values };
}

/**
 * How to read the grayscale PNG — the block `sample.ts` inverts.
 *
 * `scale`, `offset` and `nodata` are required and the block is null
 * without them. There is no default worth having: a guessed quantisation
 * puts a plausible, WRONG mm/h under the cursor, which is the one failure
 * a reviewer could never catch by eye because the picture looks the same.
 * Everything else is prose for the read-out and defaults to empty.
 */
function parseObserved(raw: unknown): ObservedEncoding | null {
	if (!isObject(raw)) return null;
	const quant = numbers(raw, ['scale', 'offset', 'nodata'] as const);
	if (quant === null) return null;
	return {
		suffix: str(raw.suffix) ?? '',
		encoding: str(raw.encoding) ?? '',
		units: str(raw.units) ?? '',
		...quant,
		reduction: str(raw.reduction) ?? '',
		source: str(raw.source) ?? '',
		caveat: str(raw.caveat) ?? ''
	};
}

/**
 * How the display PNG was coloured. Every field is advisory — the legend,
 * not the number — so an unreadable one costs a caption rather than the
 * block, and the empty defaults render as "not stated".
 */
function parseOverlay(raw: unknown): OverlayEncoding {
	const source = isObject(raw) ? raw : {};
	const stops = Array.isArray(source.colormap_stops)
		? source.colormap_stops
				.map((stop) => {
					if (!isObject(stop) || !Array.isArray(stop.rgb)) return null;
					const mm = num(stop.mm_h);
					const rgb = stop.rgb.map(num);
					if (mm === null || rgb.length !== 3 || rgb.some((c) => c === null)) return null;
					return { mm_h: mm, rgb: rgb as [number, number, number] };
				})
				.filter((stop): stop is OverlayEncoding['colormap_stops'][number] => stop !== null)
		: [];
	return {
		suffix: str(source.suffix) ?? '',
		encoding: str(source.encoding) ?? '',
		colormap: str(source.colormap) ?? '',
		colormap_stops: stops,
		interpolation: str(source.interpolation) ?? '',
		alpha_ramp: str(source.alpha_ramp) ?? '',
		floor_mm_h: num(source.floor_mm_h) ?? 0,
		solid_mm_h: num(source.solid_mm_h) ?? 0,
		min_alpha: num(source.min_alpha) ?? 0
	};
}

/**
 * The frames block. Null when the grayscale encoding is unreadable, which
 * takes the cursor read-out with it — the alternative is a read-out that
 * quietly lies.
 *
 * Note which numbers are NOT interchangeable here: `window_min` is the
 * DECISION half-window and `frame_pad_min` is the extra imagery before it,
 * because a composite is 13-24 minutes old by the time a cycle stands on
 * it. Adding them together anywhere would draw the track's left edge in
 * the wrong place.
 */
function parseFrames(raw: unknown): FramesBlock | null {
	if (!isObject(raw)) return null;
	const encodings = isObject(raw.encodings) ? raw.encodings : {};
	const observed = parseObserved(encodings.observed);
	if (observed === null) return null;
	return {
		products: strings(raw.products),
		include_doppler: bool(raw.include_doppler) ?? false,
		cadence_min: num(raw.cadence_min) ?? 0,
		downsample_factor: num(raw.downsample_factor) ?? 0,
		count: num(raw.count) ?? 0,
		stamps_total: num(raw.stamps_total) ?? 0,
		stamps_unique: num(raw.stamps_unique) ?? 0,
		dedup_saving_pct: num(raw.dedup_saving_pct) ?? 0,
		bytes: num(raw.bytes) ?? 0,
		mean_bytes_per_frame: num(raw.mean_bytes_per_frame),
		window_min: num(raw.window_min) ?? 0,
		frame_pad_min: num(raw.frame_pad_min) ?? 0,
		from_utc: iso(raw.from_utc),
		to_utc: iso(raw.to_utc),
		encodings: { observed, overlay: parseOverlay(encodings.overlay) },
		missing: strings(raw.missing),
		missing_reasons: stringMap(raw.missing_reasons)
	};
}

function parseCaveats(raw: unknown): Caveat[] {
	if (!Array.isArray(raw)) return [];
	return raw
		.map((entry) => {
			if (!isObject(entry)) return null;
			const code = str(entry.code);
			const detail = str(entry.detail);
			if (code === null || detail === null) return null;
			const severity =
				entry.severity === 'low' || entry.severity === 'medium' || entry.severity === 'high'
					? entry.severity
					: // A caveat whose severity we cannot read is shown loudly. The
						// cost of over-warning here is a reviewer reading one line.
						'high';
			return { code, severity, detail };
		})
		.filter((caveat): caveat is Caveat => caveat !== null);
}

/**
 * Parse `manifest.json`. Null only when there is nothing to key on: not an
 * object, or without a `bundle_id`. Annotations are stored per bundle, so a
 * bundle that cannot name itself cannot be reviewed at all.
 */
export function parseManifest(raw: unknown): Manifest | null {
	if (!isObject(raw)) return null;
	const bundleId = str(raw.bundle_id);
	if (bundleId === null) return null;
	const version = checkVersion('review manifest', raw.schema_version);

	const events = isObject(raw.events) ? num(raw.events.count) : null;

	return {
		schema_version: version,
		bundle_id: bundleId,
		built_at_utc: iso(raw.built_at_utc),
		builder: parseBuilder(raw.builder),
		corpus: parseCorpus(raw.corpus),
		window: parseWindowGeometry(raw.window),
		rule: parseRule(raw.rule),
		truth: parseTruth(raw.truth),
		sampling: parseSampling(raw.sampling),
		features: parseFeatureAudit(raw.features),
		grid: parseGrid(raw.grid),
		frames: parseFrames(raw.frames),
		events: events === null ? null : { count: events },
		// Both null rather than empty when the producer wrote none: "this
		// bundle documents no features" and "the documentation did not parse"
		// read the same on screen, and neither is an empty tooltip.
		feature_doc: isObject(raw.feature_doc) ? stringMap(raw.feature_doc) : null,
		caveats: Array.isArray(raw.caveats) ? parseCaveats(raw.caveats) : null
	};
}

// ---------------------------------------------------------------------------
// events.json
// ---------------------------------------------------------------------------

/**
 * One index row.
 *
 * Four fields are required because without them the row cannot be placed,
 * filtered or annotated: the id (annotations are keyed on it), the outcome
 * class (it decides which tag vocabulary is offered), the station, and the
 * anchor instant. An unknown outcome class is dropped rather than passed
 * through — a row this client does not understand would be tagged under a
 * vocabulary chosen for a different question.
 *
 * The dual-truth verdict is NOT required and is never guessed. Null means
 * neither truth could speak in the verdict window, and the quadrant it
 * would otherwise fall into — `both_dry`, "the forecast invented rain" — is
 * the single most damaging thing this tool could fabricate.
 */
function parseIndexRow(raw: unknown): IndexRow | null {
	if (!isObject(raw)) return null;
	const eventId = str(raw.event_id);
	const eventClass = outcomeClass(raw.class);
	const stationId = str(raw.station_id);
	const anchor = iso(raw.anchor_utc);
	if (eventId === null || eventClass === null || stationId === null || anchor === null) {
		return null;
	}
	const source = raw.p_decision_source;
	return {
		event_id: eventId,
		class: eventClass,
		// Absent is not a control member: the control group only works while
		// its membership is deliberate, so "not stated" is an ordinary event.
		control: bool(raw.control) ?? false,
		station_id: stationId,
		station_name: str(raw.station_name) ?? stationId,
		region: str(raw.region) ?? '',
		lat: num(raw.lat) ?? 0,
		lon: num(raw.lon) ?? 0,
		anchor_utc: anchor,
		sent_utc: iso(raw.sent_utc),
		onset_utc: iso(raw.onset_utc),
		eta_min: num(raw.eta_min),
		eta_arrival_utc: iso(raw.eta_arrival_utc),
		p_decision: num(raw.p_decision),
		p_decision_source: source === 'postprocess' || source === 'curve' ? source : null,
		threshold_pct: num(raw.threshold_pct),
		lead_error_min: num(raw.lead_error_min),
		dual_truth: dualTruthClass(raw.dual_truth),
		// Never defaulted to false: "the gauge was dry" is a claim, and
		// inventing it is how a gauge that never reported reads as evidence
		// for a false alarm.
		gauge_wet_in_window: bool(raw.gauge_wet_in_window),
		radar_wet_in_window: bool(raw.radar_wet_in_window),
		neighbour_wet_in_window: bool(raw.neighbour_wet_in_window),
		neighbour_n_known: num(raw.neighbour_n_known) ?? 0,
		season: str(raw.season) ?? '',
		hour_utc: num(raw.hour_utc) ?? new Date(anchor).getUTCHours(),
		intensity_band: str(raw.intensity_band) ?? '',
		intensity_mm_h: num(raw.intensity_mm_h),
		intensity_band_source: str(raw.intensity_band_source) ?? '',
		onset_two_slot_mm: num(raw.onset_two_slot_mm),
		arm_state_at_anchor: raw.arm_state_at_anchor === 'disarmed' ? 'disarmed' : 'armed',
		minutes_to_rearm_at_anchor: num(raw.minutes_to_rearm_at_anchor),
		frames: num(raw.frames) ?? 0,
		frames_missing: num(raw.frames_missing) ?? 0,
		flags: strings(raw.flags),
		stratum: stringMap(raw.stratum),
		detail: str(raw.detail) ?? `events/${eventId}.json`
	};
}

/**
 * Parse `events.json`. A row that does not parse is dropped and the list
 * still opens — three hundred events with one bad row is a review of 299
 * events, not a blank page. Null only when the document has no readable
 * `events` array at all.
 */
export function parseIndex(raw: unknown): EventIndex | null {
	if (!isObject(raw) || !Array.isArray(raw.events)) return null;
	const version = checkVersion('review events.json', raw.schema_version);
	return {
		schema_version: version,
		bundle_id: str(raw.bundle_id) ?? '',
		events: raw.events.map(parseIndexRow).filter((row): row is IndexRow => row !== null)
	};
}

// ---------------------------------------------------------------------------
// events/<event_id>.json
// ---------------------------------------------------------------------------

function parseWindow(raw: unknown): WindowBlock | null {
	if (!isObject(raw)) return null;
	const anchor = iso(raw.anchor_utc);
	const from = iso(raw.from_utc);
	const to = iso(raw.to_utc);
	if (anchor === null || from === null || to === null) return null;
	return {
		anchor_utc: anchor,
		from_utc: from,
		to_utc: to,
		// The frame window starts earlier than the decision window; a bundle
		// that does not say falls back to the decision edge, which draws a
		// shorter track rather than an invented one.
		frames_from_utc: iso(raw.frames_from_utc) ?? from,
		frames_to_utc: iso(raw.frames_to_utc),
		slot_from_utc: iso(raw.slot_from_utc) ?? from,
		slot_to_utc: iso(raw.slot_to_utc) ?? to,
		known_until_utc: iso(raw.known_until_utc)
	};
}

function parseNeighbourRef(raw: unknown): NeighbourRef | null {
	if (!isObject(raw)) return null;
	const id = str(raw.station_id);
	const distance = num(raw.distance_km);
	// The distance is the whole claim a neighbour makes — "rain existed 11 km
	// away" — so a reference without one says nothing worth drawing.
	if (id === null || distance === null) return null;
	return { station_id: id, station_name: str(raw.station_name) ?? id, distance_km: distance };
}

function parseStation(raw: unknown): StationBlock | null {
	if (!isObject(raw)) return null;
	const id = str(raw.station_id);
	const position = numbers(raw, ['lat', 'lon'] as const);
	if (id === null || position === null) return null;
	const grid = isObject(raw.grid) ? raw.grid : {};
	return {
		station_id: id,
		name: str(raw.name) ?? id,
		...position,
		region: str(raw.region) ?? '',
		station_radar_km: num(raw.station_radar_km),
		// Fractional grid position is a convenience for the marker; null where
		// the builder could not place the station, and the map then works from
		// lat/lon like it does for everything else.
		grid: { row: num(grid.row), col: num(grid.col) },
		neighbours: Array.isArray(raw.neighbours)
			? raw.neighbours
					.map(parseNeighbourRef)
					.filter((ref): ref is NeighbourRef => ref !== null)
			: []
	};
}

const ENGINE_ACTIONS = ['none', 'notify', 'deferred_quiet', 'already_raining', 'skipped'] as const;

const engineAction = (value: unknown): ReplayTrace['action'] | null =>
	ENGINE_ACTIONS.find((known) => known === value) ?? null;

function parseReplay(raw: unknown): ReplayTrace | null {
	if (!isObject(raw)) return null;
	const action = engineAction(raw.action);
	const armedBefore = bool(raw.armed_before);
	const armedAfter = bool(raw.armed_after);
	if (action === null || armedBefore === null || armedAfter === null) return null;
	return {
		run_id: num(raw.run_id) ?? 0,
		run_boundary_rearm: bool(raw.run_boundary_rearm) ?? false,
		armed_before: armedBefore,
		streak_before: num(raw.streak_before) ?? 0,
		armed_after: armedAfter,
		streak_after: num(raw.streak_after) ?? 0,
		below_since_utc: iso(raw.below_since_utc),
		action,
		skipped_reason: str(raw.skipped_reason)
	};
}

function parseStored(raw: unknown): StoredRow | null {
	if (!isObject(raw)) return null;
	return {
		action: str(raw.action),
		armed_after: bool(raw.armed_after),
		streak_after: num(raw.streak_after),
		threshold_pct: num(raw.threshold_pct),
		p_rain_at_rule_lead: num(raw.p_rain_at_rule_lead)
	};
}

/** Feature columns: numbers, strings and nulls, kept as the producer wrote them. */
function parseFeatureColumns(raw: unknown): Record<string, number | string | null> {
	if (!isObject(raw)) return {};
	const out: Record<string, number | string | null> = {};
	for (const [key, value] of Object.entries(raw)) {
		out[key] =
			typeof value === 'number' && Number.isFinite(value)
				? value
				: typeof value === 'string'
					? value
					: null;
	}
	return out;
}

/**
 * One decision. The two instants are required and everything else is
 * nullable: `radar_ts_utc` places it on the track and `generated_at_utc` is
 * the instant the estimate existed at, which is the single fact
 * `latestDecisionAt` is built on. A row missing either is dropped, because
 * a decision that cannot be dated is one the tool would show at the wrong
 * moment.
 */
function parseDecision(raw: unknown): Decision | null {
	if (!isObject(raw)) return null;
	const radarTs = iso(raw.radar_ts_utc);
	const generatedAt = iso(raw.generated_at_utc);
	if (radarTs === null || generatedAt === null) return null;
	const source = raw.p_decision_source;
	return {
		radar_ts_utc: radarTs,
		generated_at_utc: generatedAt,
		frame_age_min: num(raw.frame_age_min),
		frame_age_source: raw.frame_age_source === 'feature' ? 'feature' : 'derived',
		frame_ref: str(raw.frame_ref) ?? '',
		row_source: raw.row_source === 'live' ? 'live' : 'replay',
		p_rain: probabilityMap(raw.p_rain),
		p_post: probabilityMap(raw.p_post),
		p_decision: num(raw.p_decision),
		p_decision_source: source === 'postprocess' || source === 'curve' ? source : null,
		p_decision_lead_min: num(raw.p_decision_lead_min) ?? 0,
		threshold_pct: num(raw.threshold_pct),
		// Null on a row the engine passed over: it was never compared to a
		// threshold, which is a different thing from comparing below it.
		over_threshold: bool(raw.over_threshold),
		eta_min: num(raw.eta_min),
		eta_arrival_utc: iso(raw.eta_arrival_utc),
		intensity_mm_h: num(raw.intensity_mm_h),
		observed_mm_h: num(raw.observed_mm_h),
		forecast_now_mm_h: num(raw.forecast_now_mm_h),
		features: parseFeatureColumns(raw.features),
		features_present: bool(raw.features_present) ?? false,
		replay: parseReplay(raw.replay),
		stored: parseStored(raw.stored)
	};
}

function parseGaps(raw: unknown): DecisionGap[] {
	if (!Array.isArray(raw)) return [];
	return raw
		.map((entry) => {
			if (!isObject(entry)) return null;
			const from = iso(entry.from_utc);
			const to = iso(entry.to_utc);
			if (from === null || to === null) return null;
			const edge = entry.edge;
			return {
				from_utc: from,
				to_utc: to,
				// Derivable from the edges, so a missing figure is computed
				// rather than dropping a gap the track must draw.
				minutes: num(entry.minutes) ?? (Date.parse(to) - Date.parse(from)) / 60_000,
				// Not stated is not "a mere hiccup": claiming the coverage rule
				// kept counting is what turns an unwatched station into an
				// apparent miss.
				coverage_break: bool(entry.coverage_break) ?? true,
				edge:
					edge === 'leading' || edge === 'trailing' || edge === 'whole_window'
						? edge
						: null,
				reason: str(entry.reason) ?? ''
			};
		})
		.filter((gap): gap is DecisionGap => gap !== null);
}

function parsePrologue(raw: unknown): Prologue | null {
	if (!isObject(raw)) return null;
	const armed = bool(raw.armed_at_window_start);
	if (armed === null) return null;
	const recent = Array.isArray(raw.recent_actions)
		? raw.recent_actions
				.map((entry) => {
					if (!isObject(entry)) return null;
					const at = iso(entry.generated_at_utc);
					const action = engineAction(entry.action);
					if (at === null || action === null) return null;
					return { generated_at_utc: at, action, p_decision: num(entry.p_decision) };
				})
				.filter((entry): entry is Prologue['recent_actions'][number] => entry !== null)
		: [];
	const atAnchor = isObject(raw.at_anchor) ? raw.at_anchor : {};
	return {
		run_id: num(raw.run_id) ?? 0,
		run_start_utc: iso(raw.run_start_utc) ?? '',
		armed_at_window_start: armed,
		streak_at_window_start: num(raw.streak_at_window_start) ?? 0,
		below_since_utc: iso(raw.below_since_utc),
		minutes_to_rearm_at_window_start: num(raw.minutes_to_rearm_at_window_start),
		last_notify_utc: iso(raw.last_notify_utc),
		last_already_raining_utc: iso(raw.last_already_raining_utc),
		recent_actions: recent,
		// The rule's own constant. A bundle that does not state it falls back
		// to the shipped 60, which is what every bundle so far was built with.
		rearm_after_min: num(raw.rearm_after_min) ?? DEFAULT_REARM_AFTER_MIN,
		at_anchor: {
			// Armed is the state that lets a warning happen; claiming the
			// engine was blocked when nothing said so would excuse a miss.
			armed: bool(atAnchor.armed) ?? true,
			streak: num(atAnchor.streak) ?? 0,
			minutes_to_rearm: num(atAnchor.minutes_to_rearm)
		},
		run_boundary_rearms_utc: strings(raw.run_boundary_rearms_utc).filter((at) =>
			Number.isFinite(Date.parse(at))
		),
		note: str(raw.note)
	};
}

/**
 * One gauge or radar slot.
 *
 * `known` defaults to **false** and `wet` to false, which together read as
 * "we have no word on this slot". The other default — assuming a slot we
 * could not read was known and dry — is the exact mistake the `known`/`wet`
 * split exists to prevent.
 */
function parseSlot(raw: unknown): Slot | null {
	if (!isObject(raw)) return null;
	const end = iso(raw.slot_end_utc);
	if (end === null) return null;
	const known = bool(raw.known) ?? false;
	return {
		slot_end_utc: end,
		mm: num(raw.mm),
		mm_h: num(raw.mm_h),
		dur_min: num(raw.dur_min),
		known,
		// A slot nobody reported cannot be wet, whatever the flag says.
		wet: known && (bool(raw.wet) ?? false)
	};
}

const parseSlots = (raw: unknown): Slot[] =>
	Array.isArray(raw) ? raw.map(parseSlot).filter((slot): slot is Slot => slot !== null) : [];

function parseGauge(raw: unknown): GaugeBlock | null {
	if (!isObject(raw)) return null;
	const slots = parseSlots(raw.slots);
	// A gauge block with no readable slot says nothing; null renders as "no
	// gauge series in this bundle", which is true and visible.
	if (slots.length === 0) return null;
	const onsets = Array.isArray(raw.onsets)
		? raw.onsets
				.map((entry) => {
					if (!isObject(entry)) return null;
					const at = iso(entry.onset_utc);
					if (at === null) return null;
					return {
						onset_utc: at,
						two_slot_mm: num(entry.two_slot_mm),
						in_window: bool(entry.in_window) ?? true,
						is_event: bool(entry.is_event) ?? false
					};
				})
				.filter((entry): entry is GaugeBlock['onsets'][number] => entry !== null)
		: [];
	return {
		slot_min: num(raw.slot_min) ?? 10,
		slots,
		known_until_utc: iso(raw.known_until_utc),
		wet_slots_in_window: num(raw.wet_slots_in_window) ?? slots.filter((s) => s.wet).length,
		onsets
	};
}

function parseRadarDisc(raw: unknown): RadarDiscBlock | null {
	if (!isObject(raw)) return null;
	const series = Array.isArray(raw.series)
		? raw.series
				.map((entry) => {
					if (!isObject(entry)) return null;
					const at = iso(entry.radar_ts_utc);
					if (at === null) return null;
					return {
						radar_ts_utc: at,
						p90_mm_h: num(entry.p90_mm_h),
						max_mm_h: num(entry.max_mm_h),
						mean_mm_h: num(entry.mean_mm_h),
						n_pixels: num(entry.n_pixels),
						n_valid: num(entry.n_valid)
					};
				})
				.filter((entry): entry is RadarDiscBlock['series'][number] => entry !== null)
		: [];
	const slots = parseSlots(raw.slots);
	if (series.length === 0 && slots.length === 0) return null;
	return {
		disc_radius_m: num(raw.disc_radius_m) ?? 0,
		statistic: str(raw.statistic) ?? '',
		threshold_mm_h: num(raw.threshold_mm_h) ?? 0,
		series,
		slots,
		wet_in_window: bool(raw.wet_in_window),
		first_wet_utc: iso(raw.first_wet_utc)
	};
}

function parseNeighbours(raw: unknown): NeighboursBlock | null {
	if (!isObject(raw) || !Array.isArray(raw.stations)) return null;
	const stations = raw.stations
		.map((entry) => {
			const ref = parseNeighbourRef(entry);
			if (ref === null || !isObject(entry)) return null;
			return {
				...ref,
				// Null-safe on purpose: a neighbour whose coordinates did not
				// parse still belongs in the panel with its distance and its
				// slots. It simply gets no dot — see `neighbourFeatures`.
				lat: num(entry.lat),
				lon: num(entry.lon),
				bearing_deg: num(entry.bearing_deg),
				wet_in_window: bool(entry.wet_in_window),
				first_wet_utc: iso(entry.first_wet_utc),
				known_slots: num(entry.known_slots) ?? 0,
				onsets_in_window_utc: strings(entry.onsets_in_window_utc).filter((at) =>
					Number.isFinite(Date.parse(at))
				),
				slots: parseSlots(entry.slots)
			};
		})
		.filter((entry): entry is NeighboursBlock['stations'][number] => entry !== null);
	if (stations.length === 0) return null;
	return {
		radius_km: num(raw.radius_km) ?? 0,
		any_wet_in_window: bool(raw.any_wet_in_window),
		n_known: num(raw.n_known) ?? stations.filter((s) => s.wet_in_window !== null).length,
		n_wet: num(raw.n_wet) ?? stations.filter((s) => s.wet_in_window === true).length,
		stations
	};
}

function parseDualTruth(raw: unknown): DualTruthBlock | null {
	if (!isObject(raw)) return null;
	const window = isObject(raw.window_used) ? raw.window_used : {};
	const from = iso(window.from_utc);
	const to = iso(window.to_utc);
	return {
		// Both nullable, and both kept null rather than resolved: a block that
		// says "not known" is a statement the page must render in words, while
		// a missing block reads as a bundle without the section at all.
		class: dualTruthClass(raw.class),
		gauge_wet: bool(raw.gauge_wet),
		radar_wet: bool(raw.radar_wet),
		neighbour_wet: bool(raw.neighbour_wet),
		// The window the verdict was taken over travels with it or the class
		// is uninterpretable; unreadable edges leave the instants empty rather
		// than borrowing the event's own window, which is a different span.
		window_used: {
			kind: window.kind === 'warning' || window.kind === 'onset' ? window.kind : null,
			from_utc: from ?? '',
			to_utc: to ?? '',
			definition: str(window.definition) ?? ''
		},
		rule: str(raw.rule) ?? ''
	};
}

function parseNotifications(raw: unknown): NotificationMarker[] {
	if (!Array.isArray(raw)) return [];
	return raw
		.map((entry) => {
			if (!isObject(entry)) return null;
			const at = iso(entry.generated_at_utc);
			const action = engineAction(entry.action);
			if (at === null || action === null) return null;
			return {
				kind: entry.kind === 'stored' ? ('stored' as const) : ('replayed' as const),
				generated_at_utc: at,
				radar_ts_utc: iso(entry.radar_ts_utc),
				action,
				p_decision: num(entry.p_decision),
				eta_min: num(entry.eta_min),
				eta_arrival_utc: iso(entry.eta_arrival_utc),
				threshold_pct: num(entry.threshold_pct),
				is_event_warning: bool(entry.is_event_warning) ?? false
			};
		})
		.filter((marker): marker is NotificationMarker => marker !== null);
}

function parseFrameRefs(raw: unknown): FrameRef[] {
	if (!Array.isArray(raw)) return [];
	return raw
		.map((entry) => {
			if (!isObject(entry)) return null;
			const at = iso(entry.radar_ts_utc);
			const stamp = str(entry.stamp);
			if (at === null || stamp === null) return null;
			return {
				radar_ts_utc: at,
				stamp,
				product: str(entry.product),
				overlay: str(entry.overlay) ?? '',
				observed: str(entry.observed) ?? '',
				// Null means the builder did not say. The track draws such a
				// frame as unknown rather than as a hole, and the fetcher still
				// tries it — an untried frame is a hole we made ourselves.
				present: bool(entry.present),
				has_decision_row: bool(entry.has_decision_row)
			};
		})
		.filter((ref): ref is FrameRef => ref !== null)
		.sort((a, b) => Date.parse(a.radar_ts_utc) - Date.parse(b.radar_ts_utc));
}

/**
 * Parse one `events/<event_id>.json`. Null when the document cannot be
 * identified (`event_id`), cannot be placed in time (`window`), or carries
 * no readable index row — the row is what the list already showed the
 * reviewer, and a detail page that disagreed with it about the class or the
 * station would be worse than one that refuses to open.
 */
export function parseEvent(raw: unknown): EventDetail | null {
	if (!isObject(raw)) return null;
	const eventId = str(raw.event_id);
	if (eventId === null) return null;
	const index = parseIndexRow(raw.index);
	const window = parseWindow(raw.window);
	if (index === null || window === null) return null;
	const version = checkVersion(`review ${eventId}`, raw.schema_version);
	return {
		schema_version: version,
		bundle_id: str(raw.bundle_id) ?? '',
		event_id: eventId,
		index,
		station: parseStation(raw.station),
		window,
		decisions: Array.isArray(raw.decisions)
			? raw.decisions
					.map(parseDecision)
					.filter((d): d is Decision => d !== null)
					.sort((a, b) => Date.parse(a.generated_at_utc) - Date.parse(b.generated_at_utc))
			: [],
		decision_gaps: parseGaps(raw.decision_gaps),
		prologue: parsePrologue(raw.prologue),
		gauge: parseGauge(raw.gauge),
		radar_disc: parseRadarDisc(raw.radar_disc),
		neighbours: parseNeighbours(raw.neighbours),
		dual_truth: parseDualTruth(raw.dual_truth),
		notifications: parseNotifications(raw.notifications),
		frames: parseFrameRefs(raw.frames),
		flags: strings(raw.flags),
		builder_notes: strings(raw.builder_notes)
	};
}

// ---------------------------------------------------------------------------
// tags.json
// ---------------------------------------------------------------------------

function parseTag(raw: unknown): VocabularyTag | null {
	if (!isObject(raw)) return null;
	const code = str(raw.code);
	if (code === null) return null;
	return { code, description: str(raw.description) ?? '' };
}

const parseTags = (raw: unknown): VocabularyTag[] =>
	Array.isArray(raw) ? raw.map(parseTag).filter((tag): tag is VocabularyTag => tag !== null) : [];

/**
 * Parse `tags.json` — the vocabulary the bundle was drawn under, which is
 * the one a reviewer must be shown. Null hands the page over to the
 * built-in copy in `tags.ts`; a vocabulary with no verdicts and no tags is
 * the same as no vocabulary at all.
 */
export function parseVocabulary(raw: unknown): Vocabulary | null {
	if (!isObject(raw)) return null;
	const verdicts = parseTags(raw.verdicts);
	const groups = Array.isArray(raw.tag_groups)
		? raw.tag_groups
				.map((entry) => {
					if (!isObject(entry)) return null;
					const group = str(entry.group);
					if (group === null) return null;
					const tags = parseTags(entry.tags);
					return tags.length === 0 ? null : { group, label: str(entry.label) ?? group, tags };
				})
				.filter((entry): entry is Vocabulary['tag_groups'][number] => entry !== null)
		: [];
	if (verdicts.length === 0 || groups.length === 0) return null;
	const classes: Record<string, string[]> = {};
	if (isObject(raw.classes)) {
		for (const [key, value] of Object.entries(raw.classes)) classes[key] = strings(value);
	}
	return {
		vocab_version: num(raw.vocab_version) ?? 0,
		verdicts,
		tag_groups: groups,
		non_mechanism: strings(raw.non_mechanism),
		classes
	};
}

// ---------------------------------------------------------------------------
// Fetching
// ---------------------------------------------------------------------------

/**
 * Fetch and parse one bundle document.
 *
 * `no-store`, not `no-cache`: the JSON documents are rewritten under the
 * same names by every rebuild (the server sends the same header), and a
 * cached `events.json` after a `--deepen` run is a reviewer judging events
 * the bundle no longer describes. Rejects with a plain `Error` the caller
 * turns into a state — the store owns that, not this module.
 */
async function fetchDocument<T>(
	path: string,
	parse: (raw: unknown) => T | null,
	signal?: AbortSignal
): Promise<T> {
	const url = bundleUrl(path);
	const res = await fetch(url, { signal, cache: 'no-store' });
	if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
	const parsed = parse(await res.json());
	if (parsed === null) throw new Error(`${path}: nothing readable in the document`);
	return parsed;
}

export const fetchManifest = (signal?: AbortSignal): Promise<Manifest> =>
	fetchDocument('manifest.json', parseManifest, signal);

export const fetchIndex = (signal?: AbortSignal): Promise<EventIndex> =>
	fetchDocument('events.json', parseIndex, signal);

export const fetchVocabulary = (signal?: AbortSignal): Promise<Vocabulary> =>
	fetchDocument('tags.json', parseVocabulary, signal);

/** One event's detail, by the index row's own relative path. */
export const fetchEvent = (detailPath: string, signal?: AbortSignal): Promise<EventDetail> =>
	fetchDocument(detailPath, parseEvent, signal);

/**
 * Annotations as the list model wants them: keyed by event. Exported here
 * rather than in `api.ts` because the index and the annotations are joined
 * on the same key, and that join belongs beside the documents it joins.
 */
export function annotationsByEvent<T extends Pick<Annotation, 'event_id'>>(
	annotations: readonly T[]
): Map<string, T> {
	const out = new Map<string, T>();
	for (const annotation of annotations) out.set(annotation.event_id, annotation);
	return out;
}
