/**
 * The review bundle's contract, as the browser sees it.
 *
 * This is a transcription of `src/dmi_nowcast_core/review_schema.py`, which
 * is authoritative: a field is renamed there first, here second, and nothing
 * in this file may be invented by the client. Every section is nullable,
 * because a section is null whenever the evidence behind it does not exist —
 * a truth not computed, a window with nothing in it, a feature the producer
 * could not fill. **Null means "not measured", never zero**, and the page
 * must say so in words rather than render a confident 0.
 *
 * What a bundle is: a self-contained directory describing a few hundred
 * hand-picked warning events — the false positives and false negatives
 * behind a POD of 0.38 and a FAR of 0.67 — plus everything needed to judge
 * each one and the radar imagery to look at while judging. It is built on
 * the VM, copied to a laptop, and served read-only by
 * `scripts/review_server.py`. The page is dev-only and never ships.
 *
 * Units, once, for all of it:
 *
 *  - Timestamps are ISO 8601 UTC strings with an explicit offset. The page
 *    converts to the viewer's clock at the render boundary and nowhere
 *    earlier.
 *  - `radar_ts_utc` is the composite's nominal time;
 *    `generated_at_utc = radar_ts_utc + frame_age_min` is the decision
 *    instant and the wall clock the engine ran on. A warning's `sent_utc`
 *    equals it exactly. These are genuinely different instants and the UI
 *    must show both — see `FrameMode` below.
 *  - Gauge and radar slots are stamped at their END: `slot_end_utc` names
 *    the slot covering `(slot_end − 10 min, slot_end]`. An onset instant is
 *    such a slot end, so the true first drop fell somewhere in the preceding
 *    ten minutes and every measured lead is biased that far negative.
 *  - `lead_error_min = eta_min − (onset − sent)`. **Positive means the rain
 *    arrived sooner than the ETA said — the warning was LATE.** The quality
 *    page is built on this convention; do not flip it here.
 *  - Probabilities are 0–1 fractions; `*_pct` are integers 0–100.
 */

/** Schema version this client was written against. */
export const REVIEW_SCHEMA_VERSION = 1;

export type OutcomeClass =
	| 'false_alarm'
	| 'miss'
	| 'miss_late'
	| 'late'
	| 'hit'
	| 'uncovered';

/**
 * The dual-truth verdict.
 *
 * Gauge onsets define the events; the radar disc at the same point and the
 * gauges within 20 km are second and third opinions. The radar is **not**
 * independent evidence — forecast and truth come from the same instrument —
 * so `both_wet` is a consistency check, not corroboration. That sentence
 * belongs next to the badge in the UI.
 */
export type DualTruthClass =
	| 'both_wet'
	| 'radar_wet_gauge_dry'
	| 'gauge_wet_radar_dry'
	| 'both_dry';

export type Verdict = 'real_failure' | 'metric_artefact' | 'unclear';

export type ProbabilitySource = 'postprocess' | 'curve';

/** What the engine did at one observation. */
export type EngineAction =
	| 'none'
	| 'notify'
	| 'deferred_quiet'
	| 'already_raining'
	| 'skipped';

/**
 * Which frame the map shows at the cursor.
 *
 * `truth` — the newest frame at or before the cursor: what was actually
 * happening. `service` — the frame the current estimate was computed from,
 * which is 13–18 minutes older. Defaulting to `service` would flatter every
 * false alarm, because the reviewer would see only what the service saw;
 * defaulting to `truth` and hiding the distinction would do the opposite,
 * showing rain the service could not have known about. Hence two modes and
 * both stamps always printed.
 */
export type FrameMode = 'truth' | 'service';

/** Grid geometry — the same block `nowcast/manifest.ts` already parses. */
export interface GridBlock {
	proj4: string;
	x_ul_m: number;
	y_ul_m: number;
	pixel_scale_x_m: number;
	pixel_scale_y_m: number;
	shape: [number, number];
	downsample_factor: number;
}

export interface BuilderBlock {
	script: string;
	git_commit: string;
	git_dirty: boolean;
	argv: string[];
	host: string;
	python: string;
}

export interface CorpusBlock {
	root: string;
	/** In EFFECTIVE precedence order — the later directory won a tie. */
	decisions_dirs: string[];
	decision_rows_loaded: number;
	decision_row_duplicates: number;
	/** sha256 over the sorted (station, anchor, class) list. */
	population_hash: string;
}

/**
 * The rule the events were re-decided under.
 *
 * `probability_provenance` is the one field that decides whether the sample
 * is honest: `out_of_fold` means the post-processed probability came from a
 * leave-one-month-out fit, so the model never saw the event it is being
 * judged on. `in_sample_fill` means it did, which biases the sample toward
 * the model's residual failures and makes false alarms look rarer and
 * stranger than they are. The UI surfaces this.
 */
export interface RuleBlock {
	source: 'served_rule' | 'lomo' | 'stored_action';
	probability: ProbabilitySource;
	probability_column: string;
	probability_provenance: 'out_of_fold' | 'in_sample_fill' | 'stored' | 'mixed';
	thresholds: {
		by_lead_pct: Record<string, number>;
		source: string;
		fallback_threshold_pct: number;
		fitted_on_months: string[];
		/** False when the table was fitted over the same months it scores. */
		held_out: boolean;
	};
	lead_min: number;
	persistence_obs: number;
	rearm_after_min: number;
	raining_now_mm_h: number;
	raining_now_eta_min: number;
	coverage_gap_min: number;
	tolerance_min: number;
	min_useful_lead_min: number;
	/** Rows where p_post was absent and the engine fell back to the curve. */
	rows_fallback_to_curve: number;
}

export interface TruthBlock {
	onset_rule: {
		dry_min: number;
		onset_min_mm: number;
		slot_min: number;
		wet_precip_mm: number;
		wet_dur_min: number;
	};
	radar_verdict: {
		disc_radius_m: number;
		statistic: string;
		threshold_mm_h: number;
	};
	neighbour: { radius_km: number; rule: string };
	dead_gauges_excluded: string[];
	/**
	 * Pinned per station on purpose: the gauge archive grows, so a later
	 * rebuild would otherwise silently re-label events that were `pending`
	 * when this bundle was drawn.
	 */
	known_until_utc_by_station: Record<string, string>;
}

export interface SamplingCell {
	outcome_class: string;
	season: string;
	region: string;
	intensity_band: string;
	population: number;
	drawn: number;
}

export interface SamplingBlock {
	seed: number;
	target_n: number;
	drawn_n: number;
	strata_keys: string[];
	allocation: string;
	floor_per_cell: number;
	excluded: Record<string, number>;
	population: Record<string, number>;
	drawn: Record<string, number>;
	cells: SamplingCell[];
}

export interface FramesBlock {
	product: string;
	cadence_min: number;
	count: number;
	bytes: number;
	/** The DECISION window; the frame window starts earlier. See `before_pad`. */
	window_min: { before: number; after: number };
	overlay: { encoding: string; colormap: string; alpha: string };
	observed: {
		encoding: string;
		scale: number;
		offset: number;
		nodata: number;
		units: string;
		reduction: string;
	};
	missing: string[];
}

/** A reason a reviewer's reading could be wrong if they did not know. */
export interface Caveat {
	code: string;
	severity: 'low' | 'medium' | 'high';
	detail: string;
}

export interface Manifest {
	schema_version: number;
	bundle_id: string;
	built_at_utc: string;
	builder: BuilderBlock;
	corpus: CorpusBlock;
	window: { from_utc: string; to_utc: string };
	rule: RuleBlock;
	truth: TruthBlock;
	sampling: SamplingBlock;
	grid: GridBlock;
	frames: FramesBlock;
	events: { count: number };
	/** Column → prose definition, for the feature panel's tooltips. */
	feature_doc: Record<string, string>;
	caveats: Caveat[];
}

/**
 * One row of `events.json` — everything the list can filter, sort and facet
 * on. A field that is not here can never be filtered on, only inspected.
 */
export interface IndexRow {
	event_id: string;
	class: OutcomeClass;
	/** Control-group member. The UI hides this behind a reveal toggle. */
	control: boolean;
	station_id: string;
	station_name: string;
	region: string;
	lat: number;
	lon: number;
	/** `sent_utc` for warning-side classes, `onset_utc` otherwise. */
	anchor_utc: string;
	sent_utc: string | null;
	onset_utc: string | null;
	eta_min: number | null;
	eta_arrival_utc: string | null;
	p_decision: number | null;
	p_decision_source: ProbabilitySource | null;
	threshold_pct: number | null;
	lead_error_min: number | null;
	dual_truth: DualTruthClass;
	gauge_wet_in_window: boolean;
	radar_wet_in_window: boolean | null;
	neighbour_wet_in_window: boolean | null;
	neighbour_n_known: number;
	season: string;
	hour_utc: number;
	intensity_band: string;
	intensity_mm_h: number | null;
	/** Which quantity the band was read from — the stratification is auditable. */
	intensity_band_source: string;
	onset_two_slot_mm: number | null;
	arm_state_at_anchor: 'armed' | 'disarmed';
	minutes_to_rearm_at_anchor: number | null;
	frames: number;
	frames_missing: number;
	flags: string[];
	stratum: Record<string, string>;
	/** Relative path of the detail document within the bundle. */
	detail: string;
}

export interface EventIndex {
	schema_version: number;
	bundle_id: string;
	events: IndexRow[];
}

export interface NeighbourRef {
	station_id: string;
	name: string;
	lat: number;
	lon: number;
	distance_km: number;
	bearing_deg: number;
}

export interface StationBlock {
	station_id: string;
	name: string;
	lat: number;
	lon: number;
	region: string;
	station_radar_km: number | null;
	/** Fractional product-grid position, so the marker is not pixel-snapped. */
	grid: { row: number; col: number };
	neighbours: NeighbourRef[];
}

export interface WindowBlock {
	anchor_utc: string;
	/** The DECISION window. */
	from_utc: string;
	to_utc: string;
	/** The FRAME window — starts earlier, because composites are 13–24 min old. */
	frames_from_utc: string;
	slot_from_utc: string;
	slot_to_utc: string;
	known_until_utc: string | null;
}

/** The traced engine state around one observation. */
export interface ReplayTrace {
	run_id: number;
	/**
	 * The replay reset engine state at a coverage-run boundary here, handing
	 * this station a re-arm the live service never had. Without this flag a
	 * reviewer reads the resulting notify as a bug in the tool.
	 */
	run_boundary_rearm: boolean;
	armed_before: boolean;
	streak_before: number;
	armed_after: boolean;
	streak_after: number;
	below_since_utc: string | null;
	action: EngineAction;
	skipped_reason: string | null;
}

/** What the archive recorded — shown for contrast, never used to classify. */
export interface StoredRow {
	action: string | null;
	armed_after: boolean | null;
	streak_after: number | null;
	/** Null on the oldest files, whose rule cannot be recovered. */
	threshold_pct: number | null;
	p_rain_at_rule_lead: number | null;
}

export interface Decision {
	radar_ts_utc: string;
	generated_at_utc: string;
	frame_age_min: number | null;
	frame_age_source: 'feature' | 'derived';
	frame_ref: string;
	row_source: 'replay' | 'live';
	/** Lead (minutes, as a string key) → probability. Null where unserved. */
	p_rain: Record<string, number | null>;
	p_post: Record<string, number | null>;
	p_decision: number | null;
	p_decision_source: ProbabilitySource | null;
	p_decision_lead_min: number;
	threshold_pct: number | null;
	over_threshold: boolean;
	eta_min: number | null;
	eta_arrival_utc: string | null;
	intensity_mm_h: number | null;
	observed_mm_h: number | null;
	forecast_now_mm_h: number | null;
	features: Record<string, number | string | null>;
	/** False when the Phase-H feature columns were null on this row. */
	features_present: boolean;
	replay: ReplayTrace;
	stored: StoredRow;
}

export interface DecisionGap {
	from_utc: string;
	to_utc: string;
	minutes: number;
	reason: string;
}

/**
 * Engine state entering the window. Mandatory, not decorative.
 *
 * `evaluate` disarms on every `notify` AND every `already_raining`, and
 * re-arms only after 60 minutes below threshold — measured from
 * `below_since_utc`, which any over-threshold observation resets while
 * disarmed. In a showery spell the dry clock never accumulates and a station
 * can stay disarmed for hours, so every onset in that spell is a miss the
 * rule could not have caught. The disarming action is usually outside the
 * ±90-minute window, so without this block the reviewer sees a station that
 * inexplicably never fires.
 */
export interface Prologue {
	run_id: number;
	run_start_utc: string;
	armed_at_window_start: boolean;
	streak_at_window_start: number;
	below_since_utc: string | null;
	minutes_to_rearm_at_window_start: number | null;
	last_notify_utc: string | null;
	last_already_raining_utc: string | null;
	recent_actions: Array<{
		generated_at_utc: string;
		action: EngineAction;
		p_decision: number | null;
	}>;
}

/**
 * One 10-minute slot. `known` is not decoration: `{wet: false, known: false}`
 * is UNKNOWN, not dry, and a reader that collapses the two turns a gap in
 * the gauge record into evidence of a false alarm.
 */
export interface Slot {
	slot_end_utc: string;
	mm: number | null;
	dur_min?: number | null;
	known: boolean;
	wet: boolean;
}

export interface GaugeBlock {
	slot_min: number;
	slots: Slot[];
	known_until_utc: string | null;
	wet_slots_in_window: number;
	onsets: Array<{
		onset_utc: string;
		two_slot_mm: number | null;
		in_window: boolean;
		is_event: boolean;
	}>;
}

export interface RadarDiscBlock {
	disc_radius_m: number;
	statistic: string;
	threshold_mm_h: number;
	series: Array<{
		radar_ts_utc: string;
		p90_mm_h: number | null;
		max_mm_h: number | null;
		mean_mm_h: number | null;
		n_pixels: number;
		n_valid: number;
	}>;
	slots: Slot[];
	wet_in_window: boolean | null;
	first_wet_utc: string | null;
}

export interface NeighboursBlock {
	radius_km: number;
	any_wet_in_window: boolean | null;
	n_known: number;
	n_wet: number;
	stations: Array<
		NeighbourRef & {
			wet_in_window: boolean | null;
			first_wet_utc: string | null;
			known_slots: number;
			slots: Slot[];
		}
	>;
}

export interface DualTruthBlock {
	class: DualTruthClass;
	gauge_wet: boolean;
	radar_wet: boolean | null;
	neighbour_wet: boolean | null;
	/**
	 * A warning-side event is judged over `(sent, sent + lead + tol]`; an
	 * onset-side event over a window anchored on the onset. Two different
	 * windows, so this travels with every event or the class is
	 * uninterpretable.
	 */
	window_used: { from_utc: string; to_utc: string; definition: string };
	rule: string;
}

export interface NotificationMarker {
	kind: 'replayed' | 'stored';
	generated_at_utc: string;
	radar_ts_utc: string | null;
	action: EngineAction;
	p_decision: number | null;
	eta_min: number | null;
	eta_arrival_utc: string | null;
	threshold_pct: number | null;
	/** True for the warning this event is anchored on. */
	is_event_warning: boolean;
}

export interface FrameRef {
	radar_ts_utc: string;
	stamp: string;
	product: string;
	overlay: string;
	observed: string;
	present: boolean;
}

export interface EventDetail {
	schema_version: number;
	bundle_id: string;
	event_id: string;
	/** The index row repeated, so a detail file is meaningful on its own. */
	index: IndexRow;
	station: StationBlock;
	window: WindowBlock;
	decisions: Decision[];
	decision_gaps: DecisionGap[];
	prologue: Prologue | null;
	gauge: GaugeBlock | null;
	radar_disc: RadarDiscBlock | null;
	neighbours: NeighboursBlock | null;
	dual_truth: DualTruthBlock;
	notifications: NotificationMarker[];
	frames: FrameRef[];
	flags: string[];
	builder_notes: string[];
}

export interface VocabularyTag {
	code: string;
	description: string;
}

export interface Vocabulary {
	vocab_version: number;
	verdicts: VocabularyTag[];
	tag_groups: Array<{ group: string; label: string; tags: VocabularyTag[] }>;
	/** Codes excluded from the mechanism ranking — "I could not decide". */
	non_mechanism: string[];
	classes: Record<string, string[]>;
}

/** One human judgement, as the server stores and returns it. */
export interface Annotation {
	bundle_id: string;
	event_id: string;
	verdict: Verdict | null;
	tags: string[];
	vocab_version: number;
	confidence: number | null;
	needs_second_look: boolean;
	note: string;
	cursor_utc: string | null;
	/** Order this event was FIRST saved — the criteria-drift check. */
	review_seq: number;
	created_utc: string;
	updated_utc: string;
	revision: number;
}

/** The PUT payload. Everything else on `Annotation` is the server's. */
export type AnnotationDraft = Pick<
	Annotation,
	'verdict' | 'tags' | 'confidence' | 'needs_second_look' | 'note' | 'cursor_utc'
> & { vocab_version: number };
