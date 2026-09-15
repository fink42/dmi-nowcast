/**
 * The truth side of one event: what the gauge, the radar disc and the
 * neighbours said at the cursor.
 *
 * One rule shapes every type here: **a slot that was not reported is never
 * dry.** `{wet: false, known: false}` means the gauge said nothing — the
 * station was offline, the report has not landed yet, the window reaches
 * past `known_until` — and reading that as "no rain fell" turns a hole in
 * the record into evidence for a false alarm. That is the single mistake
 * this tool exists to avoid, so it is enforced in the *type*: `SlotState`
 * is a tagged union with no boolean on it, and a caller that wants "dry"
 * has to ask for `state === 'dry'` explicitly. There is no field to read
 * carelessly.
 *
 * Slots are stamped at their END: `slot_end_utc` names the slot covering
 * `(slot_end − slot_min, slot_end]`. A cursor lands in exactly one of them,
 * and an instant exactly on a slot end belongs to that slot rather than the
 * next.
 *
 * Strings are English and hard-coded rather than drawn from the i18n
 * catalog: this is a dev-only route that never ships, and putting its
 * vocabulary in the catalog would put it in the public site's bundle.
 */
import type {
	DualTruthClass,
	GaugeBlock,
	NeighbourRef,
	NeighboursBlock,
	RadarDiscBlock,
	Slot
} from './schema';
import { msOf } from './timeline';

/** Default gauge/radar slot length, in minutes. The bundle states its own. */
export const DEFAULT_SLOT_MIN = 10;

/** Why an instant has no reading. Each one renders as its own sentence. */
export type UnknownReason =
	| 'no_series'
	| 'not_reported'
	| 'beyond_known_until'
	| 'before_series'
	| 'after_series'
	| 'no_value';

/**
 * What one instrument said at one instant.
 *
 * Deliberately three states, not a nullable boolean: `unknown` carries a
 * reason and no slot value to be mistaken for zero.
 */
export type SlotState =
	| { state: 'wet'; slot: Slot; slotEndUtc: string; mm: number | null }
	| { state: 'dry'; slot: Slot; slotEndUtc: string; mm: number | null }
	| { state: 'unknown'; slot: Slot | null; slotEndUtc: string | null; reason: UnknownReason };

const unknown = (reason: UnknownReason, slot: Slot | null = null): SlotState => ({
	state: 'unknown',
	slot,
	slotEndUtc: slot?.slot_end_utc ?? null,
	reason
});

/**
 * The slot covering the cursor, as a state.
 *
 * `(end − slotMin, end]`, so 12:00:00 belongs to the slot ending 12:00 and
 * 12:00:01 to the one ending 12:10. Outside the series the answer is
 * unknown with a reason that says which side, because "before the record
 * starts" and "the gauge went quiet" are different things to a reviewer.
 */
export function slotAt(
	slots: readonly Slot[],
	cursorMs: number,
	slotMin: number = DEFAULT_SLOT_MIN
): SlotState {
	if (slots.length === 0) return unknown('no_series');
	if (!Number.isFinite(cursorMs)) return unknown('no_value');
	const spanMs = Math.max(1, slotMin) * 60_000;

	let earliestEnd = Infinity;
	let latestEnd = -Infinity;
	for (const slot of slots) {
		const endMs = msOf(slot.slot_end_utc);
		if (endMs === null) continue;
		earliestEnd = Math.min(earliestEnd, endMs);
		latestEnd = Math.max(latestEnd, endMs);
		if (cursorMs > endMs - spanMs && cursorMs <= endMs) {
			if (!slot.known) return unknown('not_reported', slot);
			return {
				state: slot.wet ? 'wet' : 'dry',
				slot,
				slotEndUtc: slot.slot_end_utc,
				mm: slot.mm
			};
		}
	}
	if (earliestEnd === Infinity) return unknown('no_series');
	if (cursorMs <= earliestEnd - spanMs) return unknown('before_series');
	if (cursorMs > latestEnd) return unknown('after_series');
	// Inside the series' span but in none of its slots: the series has a hole
	// in it, which is exactly the case that must not read as dry.
	return unknown('not_reported');
}

/**
 * The gauge at the cursor.
 *
 * `known_until` wins over the slots. The builder pins it per station
 * precisely because the gauge archive grows — a rebuild would otherwise
 * relabel events that were still pending — so an instant past it is
 * unknown even if some slot in the file happens to carry a value for it.
 */
export function gaugeStateAt(gauge: GaugeBlock | null, cursorMs: number): SlotState {
	if (gauge === null) return unknown('no_series');
	const knownUntilMs = msOf(gauge.known_until_utc);
	if (knownUntilMs !== null && cursorMs > knownUntilMs) return unknown('beyond_known_until');
	return slotAt(gauge.slots, cursorMs, gauge.slot_min);
}

/**
 * The radar disc at the cursor, from the slot series when the bundle
 * carries one and from the p90 series otherwise.
 *
 * The series path applies the bundle's own threshold to the bundle's own
 * statistic, and it only looks back one cadence: a composite from forty
 * minutes ago is not a reading for now, and carrying it forward would paint
 * a dry gap wet. A null p90 is `no_value` — off coverage or an all-nodata
 * disc — and never zero.
 */
export function radarStateAt(
	disc: RadarDiscBlock | null,
	cursorMs: number,
	slotMin: number = DEFAULT_SLOT_MIN
): SlotState {
	if (disc === null) return unknown('no_series');
	if (disc.slots.length > 0) return slotAt(disc.slots, cursorMs, slotMin);
	if (disc.series.length === 0) return unknown('no_series');
	if (!Number.isFinite(cursorMs)) return unknown('no_value');

	const spanMs = Math.max(1, slotMin) * 60_000;
	let newest: { entry: RadarDiscBlock['series'][number]; ms: number } | null = null;
	for (const entry of disc.series) {
		const ms = msOf(entry.radar_ts_utc);
		if (ms === null || ms > cursorMs || ms <= cursorMs - spanMs) continue;
		if (newest === null || ms > newest.ms) newest = { entry, ms };
	}
	if (newest === null) return unknown('not_reported');
	const value = newest.entry.p90_mm_h;
	// A synthetic slot, so the caller handles both paths identically.
	const slot: Slot = {
		slot_end_utc: newest.entry.radar_ts_utc,
		mm: value,
		known: value !== null,
		wet: value !== null && value >= disc.threshold_mm_h
	};
	if (value === null) return unknown('no_value', slot);
	return { state: slot.wet ? 'wet' : 'dry', slot, slotEndUtc: slot.slot_end_utc, mm: value };
}

export interface NeighbourState {
	station: NeighbourRef;
	state: SlotState;
	/** The neighbour's verdict over the whole window, for the map legend. */
	wetInWindow: boolean | null;
	firstWetUtc: string | null;
}

/**
 * Every neighbour's reading at the cursor, nearest first.
 *
 * A wet neighbour says rain existed in the area, NOT that it rained at the
 * event's station — 20 km comfortably exceeds a Danish summer shower — so
 * the caller must label it that way. The per-instant state is what
 * separates `fa_gauge_missed_it` (neighbours wet, this gauge known and dry)
 * from `fa_gauge_unreported` (this gauge silent).
 */
export function neighbourStatesAt(
	neighbours: NeighboursBlock | null,
	cursorMs: number,
	slotMin: number = DEFAULT_SLOT_MIN
): NeighbourState[] {
	if (neighbours === null) return [];
	return neighbours.stations
		.map((station) => ({
			station: {
				station_id: station.station_id,
				name: station.name,
				lat: station.lat,
				lon: station.lon,
				distance_km: station.distance_km,
				bearing_deg: station.bearing_deg
			},
			state: slotAt(station.slots, cursorMs, slotMin),
			wetInWindow: station.wet_in_window,
			firstWetUtc: station.first_wet_utc
		}))
		.sort((a, b) => a.station.distance_km - b.station.distance_km);
}

export interface DualTruthLabel {
	/** Null when neither truth could speak — there is no quadrant to name. */
	code: DualTruthClass | null;
	/** Two or three words for the badge. */
	label: string;
	/** What the combination means for this event. */
	reading: string;
	/**
	 * The sentence that must sit next to the badge whenever the radar has a
	 * vote: forecast and radar truth come from the same instrument, so
	 * agreement is a consistency check and not a second opinion.
	 */
	caveat: string;
}

/**
 * The 2×2 in words, plus the fifth case that is not in the 2×2 at all.
 *
 * **Null is not a quadrant.** When neither truth could speak — the gauge
 * silent, the disc with no valid pixel — there is no verdict, and the
 * quadrant it would fall into if the nulls were read as "dry" is
 * `both_dry`: "the forecast invented rain", the most damaging claim this
 * tool can make. So null gets its own words and its own branch, and the
 * switch below never sees it.
 *
 * The caveat is returned with every class rather than only with `both_wet`,
 * because the radar's *dry* verdict is no more independent than its wet
 * one: `gauge_wet_radar_dry` is the same instrument failing to see rain it
 * later forecast from, and a reviewer reading it as "the radar exonerates
 * the forecast" has the argument backwards.
 */
export function dualTruthLabel(cls: DualTruthClass | null): DualTruthLabel {
	const caveat =
		'The radar verdict comes from the instrument the forecast was made from, so it is a consistency check, not an independent opinion.';
	if (cls === null) {
		return {
			code: null,
			label: 'Not known',
			reading:
				'Neither truth could speak in the verdict window, so this event has no dual-truth verdict. That is not evidence that the sky was dry.',
			caveat
		};
	}
	switch (cls) {
		case 'both_wet':
			return {
				code: cls,
				label: 'Gauge wet, radar wet',
				reading:
					'Both instruments saw rain: a real hit or a real miss of the forecast chain, not a truth artefact.',
				caveat
			};
		case 'radar_wet_gauge_dry':
			return {
				code: cls,
				label: 'Radar wet, gauge dry',
				reading:
					'Representativeness: virga, bright band, a cell that passed beside the gauge, or rain below the gauge’s counting floor.',
				caveat
			};
		case 'gauge_wet_radar_dry':
			return {
				code: cls,
				label: 'Gauge wet, radar dry',
				reading:
					'The radar was blind here: low-level growth, beam overshoot at range, or clutter filtering.',
				caveat
			};
		case 'both_dry':
			return {
				code: cls,
				label: 'Both dry',
				reading: 'Neither instrument saw rain — the forecast invented it.',
				caveat
			};
		default:
			// A quadrant this client has never heard of. Unreachable through
			// the parser, and still not worth guessing a meaning for.
			return {
				code: null,
				label: 'Not known',
				reading: 'This bundle names a dual-truth class this page does not know.',
				caveat
			};
	}
}

export interface WetRun {
	state: 'wet' | 'dry' | 'unknown';
	fromUtc: string;
	toUtc: string;
	fromMs: number;
	toMs: number;
	/** How many slots the run covers — the strip's tooltip counts them. */
	slots: number;
	/** Total depth over the run, or null when any slot in it had no value. */
	mm: number | null;
}

/**
 * A slot series collapsed into contiguous runs, for the strip under the
 * track.
 *
 * Three states, never two: an unknown run is drawn hatched and must never
 * merge with the dry runs on either side of it. Runs are bounded by slot
 * spans — a run of three 10-minute slots is thirty minutes wide — so the
 * strip lines up with the time-proportional track above it.
 *
 * Slots are sorted by end instant first, because a series assembled from
 * two sources can arrive interleaved and a run computed over shuffled
 * slots would draw rain in the wrong half of the window.
 */
export function wetRunSegments(
	slots: readonly Slot[],
	slotMin: number = DEFAULT_SLOT_MIN
): WetRun[] {
	const spanMs = Math.max(1, slotMin) * 60_000;
	const dated = slots
		.map((slot) => ({ slot, endMs: msOf(slot.slot_end_utc) }))
		.filter((entry): entry is { slot: Slot; endMs: number } => entry.endMs !== null)
		.sort((a, b) => a.endMs - b.endMs);

	const runs: WetRun[] = [];
	for (const { slot, endMs } of dated) {
		const state = !slot.known ? 'unknown' : slot.wet ? 'wet' : 'dry';
		const startMs = endMs - spanMs;
		const last = runs[runs.length - 1];
		// Contiguous means "the previous run ends where this slot starts". A
		// series with a hole in it gets two runs, and the hole stays visible.
		if (last !== undefined && last.state === state && last.toMs === startMs) {
			last.toMs = endMs;
			last.toUtc = slot.slot_end_utc;
			last.slots += 1;
			last.mm = last.mm === null || slot.mm === null ? null : last.mm + slot.mm;
			continue;
		}
		runs.push({
			state,
			fromUtc: new Date(startMs).toISOString(),
			toUtc: slot.slot_end_utc,
			fromMs: startMs,
			toMs: endMs,
			slots: 1,
			mm: state === 'unknown' ? null : slot.mm
		});
	}
	return runs;
}
