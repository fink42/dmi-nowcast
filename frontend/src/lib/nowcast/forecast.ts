/**
 * `GET /forecast?lat=&lon=` — the server's answer for one point.
 *
 * It started as the fallback for a failed client-side decode and is now also
 * the *only* source of two things the artifacts cannot carry: the global
 * confidence scalar, and (since 2026-09-11) the gauge-trained post-processed
 * probability the site shows. Computing that needs the cycle's flow field and
 * its raw ensemble fractions, neither of which is published as a grid — so
 * the store fetches this for every selected point and merges the
 * probabilities into whatever the client-side sampler produced.
 *
 * It answers the same questions with the same conventions, so it also remains
 * a complete substitute when the grids are unavailable.
 */
import { apiUrl, NoDataError } from './manifest';
import type { PointForecast, RainSample } from './sampler';

interface ForecastPointLead {
	lead_min: number;
	p_rain: number | null;
	/**
	 * The gauge-trained post-processed probability. Additive: absent from a
	 * sidecar older than the model, null from a cycle or a point the model
	 * could not speak for. Both mean "no model number here", and the mapping
	 * below normalises one into the other.
	 */
	p_post?: number | null;
}

/** One step of the advected rain field, as the endpoint serves it. */
interface ForecastPointRain {
	lead_min: number;
	valid_ts_utc: string;
	mm_h: number | null;
}

interface ForecastPointResponse {
	lat: number;
	lon: number;
	radar_ts_utc: string;
	n_members: number;
	calibrated: boolean;
	calibration_fitted_at: string | null;
	per_lead: ForecastPointLead[];
	eta_min: number | null;
	intensity_mm_h: number | null;
	/**
	 * Additive: absent from a sidecar older than the observation product. An
	 * absent field and an explicit null both mean "no measurement here", which
	 * is why the mapping below normalises one into the other.
	 */
	observed_mm_h?: number | null;
	/**
	 * When the cycle behind this answer was computed. Additive, and read here
	 * only to document the contract: the countdown the panel applies uses the
	 * manifest's stamp, which is available on this path too (the fallback
	 * exists because the *grids* could not be read, not the manifest).
	 */
	generated_at_utc?: string | null;
	/**
	 * The advected rain field at this point, one entry per overlay lead —
	 * additive, so absent from an older sidecar and null from a cycle that
	 * produced none. Both map to an empty series, which the headline reads as
	 * "no field evidence", never as "dry".
	 */
	forecast_mm_h?: ForecastPointRain[] | null;
	/**
	 * Which probability this response's `p_post` came from. Additive, with
	 * the pre-model answer (`curve`) as the default an older sidecar implies
	 * by omitting it.
	 */
	probability_source?: 'postprocess' | 'curve' | null;
	/** When the model behind `p_post` was fitted; null without one. */
	postprocess_fitted_at_utc?: string | null;
	confidence: number | null;
}

/**
 * The served series as `RainSample`s: ascending in lead, and entries without
 * a usable timestamp dropped, exactly as `forecastSeriesArtifacts` does on the
 * client-side path. The two paths must answer the same question the same way.
 */
function rainSeriesOf(served: ForecastPointRain[] | null | undefined): RainSample[] {
	if (!Array.isArray(served)) return [];
	return served
		.filter((s) => typeof s.valid_ts_utc === 'string' && s.valid_ts_utc.trim() !== '')
		.map((s) => ({ leadMin: s.lead_min, validTsUtc: s.valid_ts_utc, mmH: s.mm_h ?? null }))
		.sort((a, b) => a.leadMin - b.leadMin);
}

/**
 * Point forecast from the server. Returns null for a point outside the radar
 * composite (the endpoint's 400) — the same off-coverage state the client-side
 * sampler reports with a null pixel.
 */
export async function fetchPointForecast(
	lat: number,
	lon: number,
	signal?: AbortSignal
): Promise<PointForecast | null> {
	const url = apiUrl(`/forecast?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`);
	const res = await fetch(url, { signal });
	if (res.status === 400) return null;
	if (res.status === 503) throw new NoDataError('sidecar has no national cycle yet');
	if (!res.ok) throw new Error(`forecast: HTTP ${res.status}`);
	const body = (await res.json()) as ForecastPointResponse;
	return {
		lat: body.lat,
		lon: body.lon,
		radarTsUtc: body.radar_ts_utc,
		perLead: body.per_lead.map((l) => ({
			leadMin: l.lead_min,
			pRain: l.p_rain,
			pPost: l.p_post ?? null
		})),
		etaMin: body.eta_min,
		intensityMmH: body.intensity_mm_h,
		observedMmH: body.observed_mm_h ?? null,
		rainSeries: rainSeriesOf(body.forecast_mm_h),
		// The endpoint serves no motion vector, and this path exists precisely
		// because the grids could not be read. Null is the honest answer.
		motion: null,
		confidence: body.confidence,
		calibrated: body.calibrated,
		probabilitySource: body.probability_source ?? 'curve',
		source: 'server'
	};
}

/**
 * The server's probabilities, folded into a forecast the client sampled.
 *
 * Everything positional stays the client's: it read the same pixel out of the
 * same cycle's grids, and it carries the motion vector and the deterministic
 * rain series the endpoint does not serve. Only the per-lead probabilities,
 * the source label, the calibration flag and the confidence scalar come
 * across — the things the server knows and the grids do not.
 *
 * Leads are matched by value, not by position: the two paths read the same
 * manifest, but a cycle that served a lead to one and not the other must lose
 * that lead's probability rather than silently take its neighbour's.
 */
export function withServedProbabilities(
	client: PointForecast,
	server: PointForecast
): PointForecast {
	const byLead = new Map(server.perLead.map((lead) => [lead.leadMin, lead]));
	return {
		...client,
		perLead: client.perLead.map((lead) => {
			const served = byLead.get(lead.leadMin);
			return served ? { ...lead, pRain: served.pRain, pPost: served.pPost ?? null } : lead;
		}),
		probabilitySource: server.probabilitySource ?? 'curve',
		confidence: server.confidence ?? client.confidence,
		calibrated: server.calibrated
	};
}
