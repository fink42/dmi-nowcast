/**
 * The `/nowcast/quality.json` contract, as the browser sees it.
 *
 * This file is the canonical definition of that document: the producer writes
 * exactly these fields, and nothing here may be invented by the client. Every
 * top-level section is nullable, because a section is null whenever the
 * evidence behind it does not exist yet — a truth we have not verified
 * against, a window with no events in it, a check that has not run. Null
 * means "not measured", never zero, and the page must say so in words.
 *
 * Units, once, for all of it:
 *
 *  - Probabilities, rates and scores (`p_rain`, `pod`, `far`, `brier`,
 *    `agreement`, `csi_*`, bin edges and frequencies) are 0–1 fractions.
 *  - `said_pct` and `happened_pct` are percentages (0–100) — they are the two
 *    numbers of the headline sentence, and rounding them to whole percent is
 *    the page's business, not the producer's.
 *  - Times are ISO 8601 UTC strings. The page converts to the viewer's clock
 *    at the boundary and nowhere else.
 *  - Durations are minutes.
 */

/** Schema version this client was written against. */
export const QUALITY_SCHEMA_VERSION = 1;

/** The radar verification window: gridded truth, deep and national. */
export interface RadarWindow {
	from: string;
	to: string;
	/** Rain events (contiguous wet episodes) the window covers. */
	events: number;
	/** Forecast/observation pairs verified inside it. */
	points: number;
}

/** The gauge verification window: ~100 ground points, 10-min slots. */
export interface GaugeWindow {
	from: string;
	to: string;
	events: number;
	stations: number;
}

/** The rolling window the live warning tally is taken over. */
export interface LiveWindow {
	days: number;
	from: string;
	to: string;
}

export interface QualityWindows {
	radar: RadarWindow | null;
	gauge: GaugeWindow | null;
	live: LiveWindow | null;
}

/**
 * "When we say X %, it rains Y % of the time", for one truth and one lead.
 * `n` is the number of forecasts in the probability bin the two numbers are
 * read from — small `n` is why the gauge line can be honest and still wobble.
 */
export interface HeadlineReliability {
	lead_min: number;
	said_pct: number;
	happened_pct: number;
	n: number;
}

/**
 * The live warning tally over `window_days`.
 *
 * `lead_error_min` is the spread of (predicted onset − observed onset) in
 * minutes over the hits: **positive means the rain had already started when
 * the warning said it would arrive — the warning was late**; negative means
 * the warning came early. `p50` is the median, `p25`/`p75` the quartiles.
 */
export interface HeadlineWarnings {
	window_days: number;
	n_stations: number;
	warnings: number;
	hits: number;
	false_alarms: number;
	misses: number;
	pod: number;
	far: number;
	lead_error_min: { p25: number; p50: number; p75: number };
}

/** Advection against the "assume nothing moves" baseline, at one horizon. */
export interface PersistenceMargin {
	horizon_min: number;
	csi_advection: number;
	csi_persistence: number;
	/** Radar frames the comparison was run over. */
	frames: number;
	from: string;
	to: string;
}

export interface QualityHeadline {
	reliability: {
		radar: HeadlineReliability | null;
		gauge: HeadlineReliability | null;
	};
	warnings: HeadlineWarnings | null;
	persistence_margin: PersistenceMargin | null;
}

/**
 * One bin of a reliability diagram. `forecast_mean` and `observed_freq` are
 * null when the bin is empty — an empty bin is a gap in the curve, not a
 * point at zero. `eff_n` is the effective sample size after accounting for
 * the correlation between neighbouring pixels and consecutive slots; it is
 * always ≤ `n` and it is what the uncertainty should be read from.
 */
export interface ReliabilityBin {
	lo: number;
	hi: number;
	forecast_mean: number | null;
	observed_freq: number | null;
	n: number;
	eff_n: number;
}

/** The reliability of one lead, against one truth. */
export interface ReliabilityCurve {
	lead_min: number;
	brier: number;
	n: number;
	eff_n: number;
	bins: ReliabilityBin[];
}

export interface QualityReliability {
	radar: ReliabilityCurve[] | null;
	gauge: ReliabilityCurve[] | null;
}

/**
 * The "is it raining now" check, against the gauges: how often the served
 * answer agreed with the ground. `observation_agreement` is the same test
 * applied to the raw radar image alone — the comparison that says whether the
 * processing earns its keep.
 */
export interface RainingNowCheck {
	n_slots: number;
	agreement: number;
	pod: number;
	far: number;
	observation_agreement: number;
	from: string;
	to: string;
}

/** Per-station scores. Any score is null where the station has too little of it. */
export interface StationProperties {
	station_id: string;
	name: string;
	/** DMI station class — "Pluvio", "Synop", or whatever else the API reports. */
	kind: 'Pluvio' | 'Synop' | string;
	n_events: number;
	brier_gauge: number | null;
	warn_pod: number | null;
	warn_far: number | null;
	warnings: number;
	raining_now_agreement: number | null;
}

export interface StationFeature {
	type: 'Feature';
	geometry: { type: 'Point'; coordinates: [number, number] };
	properties: StationProperties;
}

export interface StationCollection {
	type: 'FeatureCollection';
	features: StationFeature[];
}

/**
 * One verified warning. `gauge_onset_utc` and `lead_error_min` are null on a
 * false alarm — there is no onset to compare against. Sign of the error is as
 * in `HeadlineWarnings`: positive = the warning was late.
 */
export interface VerifiedEvent {
	station_id: string;
	name: string;
	warned_at_utc: string;
	eta_min: number;
	p_rain: number;
	gauge_onset_utc: string | null;
	outcome: 'hit' | 'false_alarm';
	lead_error_min: number | null;
}

/**
 * The rules the numbers were produced under, in the producer's own words.
 * These strings are shown verbatim next to translated labels: they are
 * technical definitions, and paraphrasing them in two languages is how a
 * definition quietly stops matching the code that implements it.
 */
export interface QualityMethods {
	/** What counts as a wet gauge slot, e.g. "≥ 0.1 mm in a 10-min slot". */
	gauge_wet_rule: string;
	/** What counts as the onset of rain at a station. */
	onset_rule: string;
	/** The rain-rate threshold the forecast probability is about, mm/h. */
	threshold_mm_h: number;
	/** [min, max] age of the radar frame behind a forecast, in minutes. */
	frame_age_range_min: [number, number];
	/** The rule a push subscriber's warning fires under. */
	subscriber_rule: {
		threshold_pct: number;
		lead_min: number;
		rearm_after_min: number;
		persistence_obs: number;
	};
	sources: { radar: string; gauges: string };
}

/**
 * What the threshold fit maximised, and the constraints it did so under.
 * Every number is nullable because a producer may report a subset — the page
 * drops the clause it cannot fill rather than the whole section.
 */
export interface ThresholdObjective {
	/** What was maximised, e.g. "f1". Verbatim from the producer. */
	metric: string;
	/** Warnings shorter than this are not useful, so they do not count. */
	min_useful_lead_min: number | null;
	/** How close to the best score a threshold may be and still be picked. */
	plateau_frac: number | null;
	/** Fewer scored warnings than this at a lead and the lead is `insufficient`. */
	min_warnings: number | null;
}

/**
 * One horizon's fitted threshold and the scores behind it.
 *
 * `threshold_pct` is null exactly when `insufficient` is true: a threshold
 * fitted on four warnings is a story about four warnings, and the served
 * rule falls back instead. Rates are null over an empty denominator, never
 * 0.0. `plateau` / `radar_plateau` are the [lo, hi] threshold ranges that
 * score within `plateau_frac` of the best, against the gauges and against
 * the radar; `agrees_with_radar` says whether the gauge pick lands inside
 * the radar's range — null when there was no radar range to compare with.
 */
export interface LeadThresholdRow {
	lead_min: number;
	threshold_pct: number | null;
	insufficient: boolean;
	f1: number | null;
	precision: number | null;
	recall: number | null;
	far: number | null;
	csi: number | null;
	warnings: number;
	hits: number;
	false_alarms: number;
	misses: number;
	late: number;
	plateau: [number, number] | null;
	radar_plateau: [number, number] | null;
	agrees_with_radar: boolean | null;
}

/** The fitted push thresholds: one row per horizon, plus what produced them. */
export interface QualityThresholds {
	fitted_at_utc: string | null;
	objective: ThresholdObjective | null;
	/** The threshold a horizon the fit cannot speak for warns at. */
	fallback_threshold_pct: number;
	/** Sorted by horizon, ascending. */
	leads: LeadThresholdRow[];
}

export interface QualityReport {
	schema_version: number;
	generated_at_utc: string;
	windows: QualityWindows;
	headline: QualityHeadline;
	reliability: QualityReliability;
	raining_now: RainingNowCheck | null;
	stations: StationCollection | null;
	/** Newest first, at most 20. */
	events: VerifiedEvent[] | null;
	methods: QualityMethods | null;
	/** Additive, and null on any producer that has not fitted them yet. */
	thresholds: QualityThresholds | null;
}
