/**
 * How fresh the displayed cycle is — two questions the UI must not conflate.
 *
 * The radar composites arrive on a 10 min cadence (fullRange), DMI publishes
 * each scan typically 5–10 min after it was taken, and the sidecar polls every
 * 5 min on top of that. The age of the newest radar *observation* is therefore
 * a sawtooth that legitimately peaks around 27–28 min on a perfectly healthy
 * pipeline. Alarming on radar age alarms nearly always, which is how the site
 * ended up permanently claiming to be broken.
 *
 * What actually says "the pipeline stopped" is `generated_at_utc`: the wall
 * clock time the last cycle was computed. It advances every sidecar poll no
 * matter what the radar is doing, so it goes stale only when the computation
 * really has stopped. Radar age stays on screen, but as information.
 */
import { radarAgeMin, type Manifest } from './manifest';

/**
 * No new cycle computed for this long means the pipeline itself has stopped:
 * roughly two sidecar poll intervals plus margin.
 */
export const PIPELINE_STALE_AFTER_MIN = 15;

/**
 * Radar age past this is worth flagging — it is beyond the worst legitimate
 * sawtooth peak, so the newest image really is later than it should be.
 */
export const RADAR_AGE_WARN_MIN = 35;

/**
 * `pipeline-stale` is the outage; `radar-old` only says DMI's newest image is
 * later than usual, which is not the same claim and must not read like it.
 */
export type FreshnessState = 'ok' | 'radar-old' | 'pipeline-stale';

export interface Freshness {
	/** Age of the newest radar observation, in minutes. Null when unknown. */
	radarAgeMin: number | null;
	/** Age of the last computed cycle, in minutes. Null when the manifest omits it. */
	pipelineAgeMin: number | null;
	state: FreshnessState;
}

/** Minutes since `iso`, or null when it is missing or unparseable. */
function ageOf(iso: string | null | undefined, now: number): number | null {
	if (typeof iso !== 'string' || iso.trim() === '') return null;
	const t = Date.parse(iso);
	if (!Number.isFinite(t)) return null;
	// Clocks disagree; a manifest stamped slightly in the future is age zero,
	// never a negative age.
	return Math.max(0, (now - t) / 60000);
}

/**
 * Classify a manifest's freshness. A manifest from an older sidecar with no
 * usable `generated_at_utc` falls back to radar age alone — an unknown
 * pipeline age is not evidence of an outage.
 */
export function freshness(manifest: Manifest | null, now: number = Date.now()): Freshness {
	if (!manifest) return { radarAgeMin: null, pipelineAgeMin: null, state: 'ok' };

	const radar = Math.max(0, radarAgeMin(manifest, now));
	const pipeline = ageOf(manifest.generated_at_utc, now);

	// Liveness wins: once the computation has stopped, the radar age is a
	// symptom of that and repeating it as a second warning helps nobody.
	const state: FreshnessState =
		pipeline !== null && pipeline > PIPELINE_STALE_AFTER_MIN
			? 'pipeline-stale'
			: radar > RADAR_AGE_WARN_MIN
				? 'radar-old'
				: 'ok';

	return {
		radarAgeMin: Number.isFinite(radar) ? radar : null,
		pipelineAgeMin: pipeline,
		state
	};
}
