/**
 * The server fallback's mapping, against a stubbed `fetch`.
 *
 * It has to produce the same shape as the client-side sampler, including the
 * fields it cannot answer — and the fields that matter most here are the
 * additive ones. `observed_mm_h` and `forecast_mm_h` are both absent from a
 * sidecar that predates them, and both absences have to arrive as something
 * the headline reads as "we do not know": null for the observation, an empty
 * series for the field, never `undefined` leaking through.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchPointForecast } from './forecast';
import { NoDataError } from './manifest';

function stubFetch(responder: (url: string) => Response) {
	vi.stubGlobal('fetch', async (input: string | URL | Request) =>
		responder(typeof input === 'string' ? input : String(input))
	);
}

const json = (body: unknown, status = 200) =>
	new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

afterEach(() => vi.unstubAllGlobals());

/** A `/forecast` body from a sidecar that serves the observation. */
const BODY = {
	lat: 55.6761,
	lon: 12.5683,
	radar_ts_utc: '2026-08-28T12:00:00+00:00',
	n_members: 24,
	calibrated: true,
	calibration_fitted_at: '2026-08-01T00:00:00+00:00',
	per_lead: [
		{ lead_min: 10, p_rain: 0.62 },
		{ lead_min: 20, p_rain: null }
	],
	eta_min: 16,
	intensity_mm_h: 2.4,
	observed_mm_h: 1.2,
	generated_at_utc: '2026-08-28T12:01:30+00:00',
	forecast_mm_h: [
		{ lead_min: 0, valid_ts_utc: '2026-08-28T12:01:30+00:00', mm_h: 0.1 },
		{ lead_min: 10, valid_ts_utc: '2026-08-28T12:11:30+00:00', mm_h: 2.2 },
		{ lead_min: 20, valid_ts_utc: '2026-08-28T12:21:30+00:00', mm_h: null }
	],
	confidence: 0.72
};

describe('fetchPointForecast', () => {
	it('camel-cases the body, observation and rain field included', async () => {
		stubFetch(() => json(BODY));
		const forecast = await fetchPointForecast(55.6761, 12.5683);
		expect(forecast).toEqual({
			lat: 55.6761,
			lon: 12.5683,
			radarTsUtc: '2026-08-28T12:00:00+00:00',
			perLead: [
				{ leadMin: 10, pRain: 0.62 },
				{ leadMin: 20, pRain: null }
			],
			etaMin: 16,
			intensityMmH: 2.4,
			observedMmH: 1.2,
			rainSeries: [
				{ leadMin: 0, validTsUtc: '2026-08-28T12:01:30+00:00', mmH: 0.1 },
				{ leadMin: 10, validTsUtc: '2026-08-28T12:11:30+00:00', mmH: 2.2 },
				{ leadMin: 20, validTsUtc: '2026-08-28T12:21:30+00:00', mmH: null }
			],
			motion: null,
			confidence: 0.72,
			calibrated: true,
			source: 'server'
		});
	});

	it('reads an older sidecar’s missing observation as null, not undefined', async () => {
		// The field is additive: a service deployed before it existed omits it
		// entirely, and the panel must fall back to the ETA rule there.
		const { observed_mm_h: _omitted, ...older } = BODY;
		stubFetch(() => json(older));
		const forecast = await fetchPointForecast(55.6761, 12.5683);
		expect(forecast!.observedMmH).toBeNull();
		expect('observedMmH' in forecast!).toBe(true);
		// An explicit null (nodata at this pixel) reads identically.
		stubFetch(() => json({ ...BODY, observed_mm_h: null }));
		expect((await fetchPointForecast(55.6761, 12.5683))!.observedMmH).toBeNull();
	});

	it('keeps a measured zero, which is dry rather than unknown', async () => {
		stubFetch(() => json({ ...BODY, observed_mm_h: 0 }));
		expect((await fetchPointForecast(55.6761, 12.5683))!.observedMmH).toBe(0);
	});

	it('reads a missing or null rain field as an empty series, not as dry', async () => {
		// A sidecar older than the product omits the key entirely.
		const { forecast_mm_h: _omitted, ...older } = BODY;
		stubFetch(() => json(older));
		expect((await fetchPointForecast(55.6761, 12.5683))!.rainSeries).toEqual([]);
		// A cycle that produced none sends an explicit null.
		stubFetch(() => json({ ...BODY, forecast_mm_h: null }));
		expect((await fetchPointForecast(55.6761, 12.5683))!.rainSeries).toEqual([]);
		// Empty is empty; the headline falls back to the ensemble ETA there.
		stubFetch(() => json({ ...BODY, forecast_mm_h: [] }));
		expect((await fetchPointForecast(55.6761, 12.5683))!.rainSeries).toEqual([]);
	});

	it('sorts the rain field by lead and drops entries with no instant', async () => {
		// The client-side path drops unstamped entries too: a step we cannot
		// place on the clock is worse than one fewer sample.
		stubFetch(() =>
			json({
				...BODY,
				forecast_mm_h: [
					{ lead_min: 20, valid_ts_utc: '2026-08-28T12:21:30+00:00', mm_h: 1 },
					{ lead_min: 10, valid_ts_utc: '', mm_h: 5 },
					{ lead_min: 0, valid_ts_utc: '2026-08-28T12:01:30+00:00', mm_h: 3 }
				]
			})
		);
		const forecast = await fetchPointForecast(55.6761, 12.5683);
		expect(forecast!.rainSeries.map((s) => s.leadMin)).toEqual([0, 20]);
	});

	it('asks the endpoint for the point it was given', async () => {
		let seen = '';
		stubFetch((url) => {
			seen = url;
			return json(BODY);
		});
		await fetchPointForecast(55.6761, 12.5683);
		expect(seen).toBe('/forecast?lat=55.6761&lon=12.5683');
	});

	it('maps the endpoint’s off-coverage and cold-start answers', async () => {
		stubFetch(() => json({ detail: 'outside the grid' }, 400));
		expect(await fetchPointForecast(0, 0)).toBeNull();
		stubFetch(() => json({ detail: 'no cycle yet' }, 503));
		await expect(fetchPointForecast(55.6, 12.5)).rejects.toBeInstanceOf(NoDataError);
		stubFetch(() => json({ detail: 'boom' }, 500));
		await expect(fetchPointForecast(55.6, 12.5)).rejects.toThrow(/HTTP 500/);
	});
});
