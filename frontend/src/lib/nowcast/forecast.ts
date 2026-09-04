/**
 * The server fallback: `GET /forecast?lat=&lon=`.
 *
 * Used when the client-side decode or projection fails for any reason (an
 * exotic PNG, a browser without `DecompressionStream`, a grid that failed to
 * download). It answers the same questions with the same conventions, and it
 * carries one thing the grids do not: the global confidence scalar.
 */
import { apiUrl, NoDataError } from './manifest';
import type { PointForecast } from './sampler';

interface ForecastPointLead {
	lead_min: number;
	p_rain: number | null;
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
	confidence: number | null;
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
		perLead: body.per_lead.map((l) => ({ leadMin: l.lead_min, pRain: l.p_rain })),
		etaMin: body.eta_min,
		intensityMmH: body.intensity_mm_h,
		observedMmH: body.observed_mm_h ?? null,
		// The endpoint serves no motion vector, and this path exists precisely
		// because the grids could not be read. Null is the honest answer.
		motion: null,
		confidence: body.confidence,
		calibrated: body.calibrated,
		source: 'server'
	};
}
