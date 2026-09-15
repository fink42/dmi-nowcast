/**
 * The controlled vocabulary, and what makes one judgement valid.
 *
 * The bundle ships its own `tags.json` and that copy is the one a reviewer
 * must be shown: renaming a tag under saved annotations silently rewrites
 * history, which is why every annotation carries the `vocab_version` it was
 * made under. The constant below is a **fallback** for a bundle too old to
 * carry a vocabulary, transcribed from
 * `src/dmi_nowcast_core/review_schema.py`, and a test pins its codes to the
 * fixture's so the two cannot drift apart unnoticed.
 *
 * Why a vocabulary at all: free-text notes on three hundred events cannot
 * be counted, and the output of this exercise is a *ranked list of named
 * failure modes*. Each code names a mechanism that exists in this pipeline
 * and points at a fix — a rule change, a model change, or a change to the
 * scoring definition. The `non_mechanism` codes are the ones that mean "I
 * could not decide", and they are excluded from that ranking rather than
 * counted as a cause.
 */
import type { Annotation, AnnotationDraft, OutcomeClass, Vocabulary, VocabularyTag } from './schema';

/** Vocabulary version this client's built-in copy was transcribed from. */
export const BUILTIN_VOCAB_VERSION = 1;

const VERDICTS: VocabularyTag[] = [
	{
		code: 'real_failure',
		description: 'The forecast was wrong in a way a better model or rule could fix.'
	},
	{
		code: 'metric_artefact',
		description:
			'The forecast was defensible; the label came from the scoring rule — the onset definition, the matching window, the re-arm, a low-catch gauge, or coverage.'
	},
	{ code: 'unclear', description: 'The evidence in the bundle does not settle it.' }
];

const FALSE_ALARM_TAGS: VocabularyTag[] = [
	{
		code: 'fa_cell_died',
		description:
			'Echo existed upstream (up_max_20/40km_mm_h > 0) and decayed before arrival. STEPS advects; it carries no deterministic growth or decay.'
	},
	{
		code: 'fa_cell_diverted',
		description:
			'The echo passed to one side: the completed flow at the station (local_speed_kmh, bulk_dir_deg) differed from the cell\'s own motion.'
	},
	{
		code: 'fa_arrived_late',
		description:
			'The rain did come, but after sent + lead + tolerance, so the greedy matching window would not let this warning claim the onset.'
	},
	{
		code: 'fa_arrived_early',
		description:
			'The rain came before the warning could be useful; a neighbouring claim carries a large positive lead_error_min.'
	},
	{
		code: 'fa_virga_or_aloft',
		description:
			'Column-max reflectivity over a dry gauge — the documented composite bias. Radar wet, gauge dry, neighbours dry.'
	},
	{
		code: 'fa_clutter_or_bright_band',
		description:
			'Stationary echo across frames, or a high stalled_share; small station_radar_km (near-radar clutter) or large (melting layer).'
	},
	{
		code: 'fa_drizzle_below_gauge_floor',
		description:
			'Radar 0.5-1 mm/h; the gauge stayed below 0.1 mm per slot or below the 0.2 mm two-slot onset floor. Real rain, not a countable onset.'
	},
	{
		code: 'fa_gauge_missed_it',
		description:
			'Neighbours wet, this gauge dry with known=true: wind loss, low catch, or a gauge not yet dead by the dead_gauges rule.'
	},
	{
		code: 'fa_gauge_unreported',
		description:
			'The window\'s slots are known=false. There is no truth here, and known_until / pending did not catch it.'
	},
	{
		code: 'fa_already_raining',
		description:
			'Rain was already falling before the warning, so the onset rule\'s 60 dry minutes was never satisfied. The engine\'s already-raining test reads only the current row.'
	},
	{
		code: 'fa_probability_saturated',
		description:
			'raw_frac_<lead> pinned at 1.0 — ensemble saturation; visible as p_rain much greater than p_post, or both pinned.'
	},
	{
		code: 'fa_threshold_marginal',
		description:
			'p_decision within 5 points of the threshold: a coin flip, not a mechanism. Tag it so it does not pollute the mechanism counts.'
	},
	{
		code: 'fa_stale_frame',
		description:
			'frame_age_min >= 20, or a coverage gap immediately before: the decision stood on an old composite.'
	},
	{
		code: 'fa_edge_of_coverage',
		description:
			'observed_mm_h null, few valid pixels in the disc, or the station near a composite or radar edge.'
	},
	{ code: 'fa_other', description: 'Something else. Requires a note.' }
];

const MISS_TAGS: VocabularyTag[] = [
	{
		code: 'miss_disarmed_rearm',
		description:
			'Disarmed at the onset with the 60-minute re-arm unexpired: structurally unreachable. If this is a large share of misses, the finding is a rule change, not a model change.'
	},
	{
		code: 'miss_arm_consumed_already_raining',
		description:
			'The arm was consumed silently by the already-raining branch before the onset — no push, still disarmed.'
	},
	{
		code: 'miss_below_threshold',
		description:
			'p_decision peaked below the threshold across the whole pre-onset window: sharpness or calibration, not plumbing.'
	},
	{
		code: 'miss_no_probability',
		description:
			'Both p_post and p_rain were null at the rule\'s lead, so the engine passed over those rows (off coverage, unserved lead, feature gap).'
	},
	{
		code: 'miss_convective_initiation',
		description:
			'Nothing upstream at -30 min (up_max_40km_mm_h near zero, up_dist_km NaN): the rain grew in place, which advection cannot see.'
	},
	{
		code: 'miss_too_fast',
		description:
			'Rain was upstream but arrived sooner than the lead could cover: small up_dist_km with high bulk_kmh.'
	},
	{
		code: 'miss_frame_age_ate_the_lead',
		description:
			'frame_age_min plus the arrival time exceeded the lead. A fresher anchor would have warned.'
	},
	{
		code: 'miss_coverage_gap',
		description:
			'Decision rows are missing around the onset even though the coverage rule did not call it uncovered.'
	},
	{
		code: 'miss_radar_saw_nothing',
		description:
			'The disc stayed dry through the onset: beam overshoot or shallow rain, typically at large station_radar_km.'
	},
	{
		code: 'miss_gauge_spurious_onset',
		description:
			'An isolated 0.2 mm with no radar and no neighbour: a suspect onset (heated gauge, dew, a bumped bucket).'
	},
	{
		code: 'miss_snow_or_sleet',
		description:
			'Winter, long precip duration with tiny depth, weak radar: the Z-R relation is rain-tuned.'
	},
	{
		code: 'miss_onset_definition_artefact',
		description:
			'The “new” onset is a continuation whose dry run was reset by an UNKNOWN slot rather than a dry one.'
	},
	{
		code: 'miss_threshold_marginal',
		description: 'Peak p_decision within 5 points below the threshold.'
	},
	{ code: 'miss_other', description: 'Something else. Requires a note.' }
];

const COMMON_TAGS: VocabularyTag[] = [
	{
		code: 'needs_better_imagery',
		description:
			'The +/-90 minute window or the 2 km product grid was not enough to judge this one. Candidate for a --deepen re-run.'
	},
	{ code: 'interesting', description: 'Worth coming back to, or worth showing someone.' }
];

/**
 * Codes whose meaning is "I could not decide". They are offered because a
 * reviewer needs somewhere honest to put a coin-flip event, and excluded
 * from the mechanism ranking because counting them as causes would put
 * "marginal" at the top of a list of things to fix.
 */
const NON_MECHANISM = [
	'fa_other',
	'fa_threshold_marginal',
	'interesting',
	'miss_other',
	'miss_threshold_marginal',
	'needs_better_imagery'
];

/** Codes whose description demands a note before the judgement is finished. */
export const NOTE_REQUIRED_TAGS: readonly string[] = ['fa_other', 'miss_other'];

const codes = (tags: readonly VocabularyTag[]): string[] => tags.map((tag) => tag.code);

/**
 * Which cause list each outcome class gets — `review_schema.tags_for_class`.
 *
 * A `hit` gets BOTH lists on purpose. The hit control group exists to
 * supply a base rate, and a base rate is only useful if the same vocabulary
 * was available: if `fa_cell_died` describes 40 % of the hits as well, it
 * explains nothing about the false alarms, and the only way to find that
 * out is to let a reviewer tag it on a hit.
 */
function builtinClasses(): Record<string, string[]> {
	const fa = [...codes(FALSE_ALARM_TAGS), ...codes(COMMON_TAGS)].sort();
	const miss = [...codes(MISS_TAGS), ...codes(COMMON_TAGS)].sort();
	const both = [
		...codes(FALSE_ALARM_TAGS),
		...codes(MISS_TAGS),
		...codes(COMMON_TAGS)
	].sort();
	return {
		false_alarm: fa,
		late: fa,
		miss,
		miss_late: miss,
		uncovered: miss,
		hit: both
	};
}

/**
 * The vocabulary this client was compiled with. Used only when the bundle
 * carries none — an older bundle, or a `tags.json` that did not parse —
 * and the page says which of the two it is showing, because a reviewer
 * tagging under a vocabulary the bundle does not know is producing rows the
 * export cannot interpret.
 */
export const BUILTIN_VOCABULARY: Vocabulary = {
	vocab_version: BUILTIN_VOCAB_VERSION,
	verdicts: VERDICTS,
	tag_groups: [
		{ group: 'false_alarm', label: 'Why did it warn with no rain?', tags: FALSE_ALARM_TAGS },
		{ group: 'miss', label: 'Why was there no warning?', tags: MISS_TAGS },
		{ group: 'common', label: 'Either way', tags: COMMON_TAGS }
	],
	non_mechanism: NON_MECHANISM,
	classes: builtinClasses()
};

/** Every tag the vocabulary offers, across all of its groups. */
export function allTags(vocabulary: Vocabulary): VocabularyTag[] {
	const out: VocabularyTag[] = [];
	const seen = new Set<string>();
	for (const group of vocabulary.tag_groups) {
		for (const tag of group.tags) {
			if (seen.has(tag.code)) continue;
			seen.add(tag.code);
			out.push(tag);
		}
	}
	return out;
}

/**
 * The tags offered for one outcome class, grouped as the vocabulary groups
 * them so the UI can keep the two questions ("why did it warn?", "why was
 * there no warning?") visually apart.
 *
 * The bundle's own `classes` map decides membership when it has one. A
 * class it does not mention falls back to every group, because offering too
 * many codes costs a reviewer a scroll while offering too few silently
 * makes a mechanism untaggable — and a mechanism nobody could tag looks in
 * the export exactly like a mechanism that never happened.
 */
export function tagsForClass(
	vocabulary: Vocabulary,
	eventClass: OutcomeClass | string
): Array<{ group: string; label: string; tags: VocabularyTag[] }> {
	const allowed = vocabulary.classes[eventClass];
	const filter = Array.isArray(allowed) && allowed.length > 0 ? new Set(allowed) : null;
	return vocabulary.tag_groups
		.map((group) => ({
			group: group.group,
			label: group.label,
			tags: filter === null ? group.tags : group.tags.filter((tag) => filter.has(tag.code))
		}))
		.filter((group) => group.tags.length > 0);
}

/** True for codes that must not be counted as a mechanism in the tally. */
export const isMechanismTag = (vocabulary: Vocabulary, code: string): boolean =>
	!vocabulary.non_mechanism.includes(code);

export interface AnnotationProblem {
	field: 'verdict' | 'tags' | 'confidence' | 'note' | 'cursor_utc' | 'vocab_version';
	message: string;
	/** The offending values, so the UI can point at them rather than describe them. */
	offending: string[];
}

/**
 * Check a draft the way `scripts/review_server.py` will.
 *
 * Client-side validation here is not a security measure — the server
 * re-checks everything and answers 422 — it is a way to keep a reviewer
 * from losing a judgement to a round trip. The rules are deliberately the
 * server's, including the awkward one: `confidence` is an integer 1…3, not
 * the 0–1 fraction `schema.ts` leaves room for.
 *
 * Tags are checked against the whole vocabulary rather than the class's
 * subset, exactly as the server does: a reviewer reaching for a false-alarm
 * tag on an `uncovered` event is telling us something about the coverage
 * rule, not making a mistake worth blocking.
 */
export function validateAnnotation(
	draft: Partial<AnnotationDraft>,
	vocabulary: Vocabulary
): AnnotationProblem[] {
	const problems: AnnotationProblem[] = [];

	const verdicts = new Set(vocabulary.verdicts.map((verdict) => verdict.code));
	if (draft.verdict !== null && draft.verdict !== undefined && !verdicts.has(draft.verdict)) {
		problems.push({
			field: 'verdict',
			message: `unknown verdict ${draft.verdict}`,
			offending: [String(draft.verdict)]
		});
	}

	const known = new Set(allTags(vocabulary).map((tag) => tag.code));
	const tags = draft.tags ?? [];
	if (!Array.isArray(tags) || tags.some((tag) => typeof tag !== 'string')) {
		problems.push({
			field: 'tags',
			message: 'tags must be a list of vocabulary codes',
			offending: []
		});
	} else {
		const bad = [...new Set(tags.filter((tag) => !known.has(tag)))].sort();
		if (bad.length > 0) {
			problems.push({
				field: 'tags',
				message: `unknown tag code(s): ${bad.join(', ')}`,
				offending: bad
			});
		}
	}

	const confidence = draft.confidence;
	if (
		confidence !== null &&
		confidence !== undefined &&
		(!Number.isInteger(confidence) || confidence < 1 || confidence > 3)
	) {
		problems.push({
			field: 'confidence',
			message: 'confidence must be null or an integer 1..3',
			offending: [String(confidence)]
		});
	}

	if (draft.note !== undefined && typeof draft.note !== 'string') {
		problems.push({ field: 'note', message: 'note must be a string', offending: [] });
	}

	const cursor = draft.cursor_utc;
	if (cursor !== null && cursor !== undefined && !Number.isFinite(Date.parse(cursor))) {
		problems.push({
			field: 'cursor_utc',
			message: 'cursor_utc must be null or an ISO 8601 instant',
			offending: [String(cursor)]
		});
	}

	if (draft.vocab_version !== undefined && !Number.isInteger(draft.vocab_version)) {
		problems.push({
			field: 'vocab_version',
			message: 'vocab_version must be an integer',
			offending: [String(draft.vocab_version)]
		});
	}

	return problems;
}

/**
 * Is this judgement finished?
 *
 * Stricter than the server's "has a verdict", on purpose: the server counts
 * what is stored, while the reviewer needs to know what is *done*. A
 * verdict with no tag records that something went wrong without recording
 * what, which is the aggregate the tool exists to replace; and the two
 * `_other` codes say in their own description that they require a note,
 * since "something else" with no sentence after it cannot be read six
 * months later.
 */
export function isComplete(draft: Partial<AnnotationDraft> | null): boolean {
	if (draft === null || draft === undefined) return false;
	if (!draft.verdict) return false;
	const tags = draft.tags ?? [];
	if (tags.length === 0) return false;
	const needsNote = tags.some((tag) => NOTE_REQUIRED_TAGS.includes(tag));
	return !needsNote || (draft.note ?? '').trim() !== '';
}

const sameTags = (a: readonly string[], b: readonly string[]): boolean => {
	const left = new Set(a);
	const right = new Set(b);
	if (left.size !== right.size) return false;
	for (const code of left) if (!right.has(code)) return false;
	return true;
};

/**
 * Has the draft moved away from what the server holds?
 *
 * Tags compare as SETS. The server dedupes them and the aggregation counts
 * them as a set, so a reviewer who toggled a tag off and straight back on
 * has changed nothing — and an unsaved-changes warning that fires on that
 * trains people to ignore it.
 *
 * A draft with no saved row behind it is dirty as soon as it carries
 * anything at all; an untouched blank draft is not, or every event a
 * reviewer merely looked at would block navigation.
 */
export function isDirty(
	draft: Partial<AnnotationDraft> | null,
	saved: Annotation | null
): boolean {
	if (draft === null || draft === undefined) return false;
	const tags = draft.tags ?? [];
	if (saved === null) {
		return (
			Boolean(draft.verdict) ||
			tags.length > 0 ||
			(draft.note ?? '').trim() !== '' ||
			(draft.confidence ?? null) !== null ||
			draft.needs_second_look === true
		);
	}
	return (
		(draft.verdict ?? null) !== saved.verdict ||
		!sameTags(tags, saved.tags) ||
		(draft.confidence ?? null) !== saved.confidence ||
		(draft.needs_second_look ?? false) !== saved.needs_second_look ||
		(draft.note ?? '') !== saved.note ||
		(draft.cursor_utc ?? null) !== saved.cursor_utc
	);
}
