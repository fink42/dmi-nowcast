/**
 * The `/nowcast/manifest.json` contract, as the browser sees it.
 *
 * The sidecar writes one manifest per cycle (sidecar/dmi_nowcast_sidecar/
 * national_artifacts.py) describing every artifact plus the grid geometry
 * needed to sample the product grids client-side. Everything here is a
 * transcription of that schema — no invention, no defaults that hide a
 * missing field.
 */

export interface GridBlock {
	/** proj4 string of the radar composite's projection (polar stereographic). */
	proj4: string;
	/** Upper-left corner of the grid in projection metres. */
	x_ul_m: number;
	y_ul_m: number;
	/** Effective pixel size — native scale × downsample_factor, already applied. */
	pixel_scale_x_m: number;
	pixel_scale_y_m: number;
	/** [rows, cols]. */
	shape: [number, number];
	downsample_factor: number;
}

export type ProductName =
	| 'p_rain'
	| 'eta'
	| 'intensity'
	| 'overlay'
	/**
	 * The rain the radar measured *now*, on the product grid: the native field
	 * reduced by block-wise 90th percentile, which is the same "p90 over about
	 * a kilometre" rule the home-assistant integration calls raining_now.
	 * Additive in schema v2 — absent from every manifest written before it
	 * existed, so every reader of it must tolerate `undefined`.
	 */
	| 'observed_mm_h'
	| 'motion_east_kmh'
	| 'motion_north_kmh';

/**
 * What an overlay frame depicts. `observation` frames are radar measurements —
 * this cycle's "now" and, since schema v2, the prior cycles kept as history;
 * `forecast` frames are extrapolation. The distinction is the whole point of
 * the timeline: it must never be inferred from styling alone.
 */
export type FrameKind = 'observation' | 'forecast';

export interface ArtifactEntry {
	filename: string;
	product: ProductName;
	/** Negative on schema-v2 history frames: minutes of the past. */
	lead_min: number | null;
	/** Overlay frames, schema v2. Absent on a v1 manifest — see `frameKind`. */
	kind?: FrameKind;
	/**
	 * Overlay frames, schema v2: the instant the frame depicts (ISO 8601 UTC),
	 * already corrected for frame age on forecasts. Authoritative — never
	 * recompute it from `radar_ts_utc` + `lead_min` when it is present.
	 */
	valid_ts_utc?: string;
	encoding: 'grayscale8' | 'rgba8';
	/** Quantisation, grayscale artifacts only: value = level * scale + offset. */
	scale?: number;
	offset?: number;
	/** Level reserved for "no value" (255). */
	nodata?: number;
	units?: string;
	shape: [number, number];
}

/**
 * Schema v2's cell-motion block. The grids themselves live in `artifacts` as
 * two grayscale products on the *product* grid geometry (the `grid` block);
 * this only carries what geometry cannot say — chiefly that nodata means
 * "no estimate here", not "no motion here".
 */
export interface MotionBlock {
	grid: string;
	/**
	 * Null since issue #6: the sidecar no longer cuts the grids off at a
	 * radius from the echo, it fills coverage with the nearest cells' motion.
	 * The key stays so older manifests still parse.
	 */
	support_radius_km: number | null;
	/** How the off-echo vectors were filled, e.g. `nearest-cells-v1`. */
	fill?: string;
	/** Search scales of that fill, in km, smallest first. */
	fill_scales_km?: number[];
	max_abs_kmh: number;
	convention: string;
}

export interface CalibrationBlock {
	fitted_at: string | null;
	calibrated_leads: number[] | null;
	n_samples?: number | null;
	brier_before?: number | null;
	brier_after?: number | null;
}

export interface Manifest {
	schema_version: number;
	cycle: string;
	radar_ts_utc: string;
	generated_at_utc: string;
	threshold_mm_h: number;
	timestep_min: number;
	frame_age_min: number;
	/**
	 * STEPS horizon in minutes from RADAR-FRAME time. The honest horizon
	 * from now is `ensemble_horizon_min - frame_age_min`; leads beyond it
	 * are answered by the ensemble's final timestep rather than by a
	 * forecast for that lead. Absent on a manifest written before the field
	 * existed, null when the sidecar did not state one — treat both as
	 * "unknown", never as zero.
	 */
	ensemble_horizon_min?: number | null;
	n_members: number;
	leads_min: number[];
	grid: GridBlock;
	/** Null when the cycle produced no 500 m overlay frames. */
	overlay_grid: GridBlock | null;
	/**
	 * Null when this cycle wrote no cell-motion grids; absent entirely on a
	 * pre-v2 manifest. Both read the same way here: no arrow.
	 */
	motion?: MotionBlock | null;
	/** Null when the served probability grids are raw (uncalibrated). */
	calibration: CalibrationBlock | null;
	artifacts: ArtifactEntry[];
}

/** Schema version this client was written against. */
export const SUPPORTED_SCHEMA_VERSION = 2;

/**
 * Base URL for the data endpoints. Empty in production: the sidecar serves
 * this app, so `/nowcast/...` is same-origin. `VITE_API_BASE` exists as an
 * escape hatch for a build that must talk to a different origin (that origin
 * then has to send CORS headers); local development instead proxies through
 * the dev server — see vite.config.ts and README.md.
 */
const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/+$/, '');

export const apiUrl = (path: string): string => `${API_BASE}${path}`;

export const manifestUrl = (): string => apiUrl('/nowcast/manifest.json');

export const artifactUrl = (filename: string): string => apiUrl(`/nowcast/${filename}`);

export class NoDataError extends Error {}

/** Fetch the newest cycle manifest. 503 → NoDataError (first cycle pending). */
export async function fetchManifest(signal?: AbortSignal): Promise<Manifest> {
	const res = await fetch(manifestUrl(), { signal, cache: 'no-cache' });
	if (res.status === 503) throw new NoDataError('sidecar has no cycle yet');
	if (!res.ok) throw new Error(`manifest: HTTP ${res.status}`);
	const manifest = (await res.json()) as Manifest;
	if (manifest.schema_version !== SUPPORTED_SCHEMA_VERSION) {
		// Newer schema: keep rendering rather than blanking the site, but say so.
		console.warn(
			`manifest schema_version ${manifest.schema_version} != ${SUPPORTED_SCHEMA_VERSION}`
		);
	}
	return manifest;
}

/**
 * Overlay frames in animation order, ordered by lead (0 = "now").
 *
 * Schema v2's observation history carries negative leads, so ascending order
 * puts the past first for free: −20, −10, 0, +10 …
 */
export function overlayFrames(manifest: Manifest): ArtifactEntry[] {
	return manifest.artifacts
		.filter((a) => a.product === 'overlay')
		.sort((a, b) => (a.lead_min ?? 0) - (b.lead_min ?? 0));
}

/**
 * Measurement or extrapolation. v2 states it per entry; on an older manifest
 * the sign of the lead is the only evidence there is, and it says the same
 * thing (lead 0 is the radar image itself).
 */
export function frameKind(entry: ArtifactEntry): FrameKind {
	if (entry.kind === 'observation' || entry.kind === 'forecast') return entry.kind;
	return (entry.lead_min ?? 0) <= 0 ? 'observation' : 'forecast';
}

/**
 * The instant a frame depicts, as an ISO 8601 UTC string.
 *
 * v2 stamps every overlay entry with it, frame-age-corrected — a forecast is
 * valid at `radar_ts + frame_age + lead`, not `radar_ts + lead`, and at the
 * 10 min composite cadence that difference is a whole animation step. The
 * arithmetic below is therefore a *fallback* for a manifest that omits the
 * field, never a cross-check of one that has it.
 */
export function frameValidTs(manifest: Manifest, entry: ArtifactEntry): string {
	const stamped = entry.valid_ts_utc;
	if (typeof stamped === 'string' && Number.isFinite(Date.parse(stamped))) return stamped;
	const radar = Date.parse(manifest.radar_ts_utc);
	if (!Number.isFinite(radar)) return manifest.radar_ts_utc;
	return new Date(radar + (entry.lead_min ?? 0) * 60_000).toISOString();
}

/** The single grayscale artifact for a product (+ lead, for p_rain). */
export function findArtifact(
	manifest: Manifest,
	product: ProductName,
	leadMin: number | null = null
): ArtifactEntry | undefined {
	return manifest.artifacts.find((a) => a.product === product && a.lead_min === leadMin);
}

/** Age of the radar image behind this manifest, in minutes. */
export function radarAgeMin(manifest: Manifest, now: number = Date.now()): number {
	return (now - Date.parse(manifest.radar_ts_utc)) / 60000;
}

/** True when the manifest says every served lead went through a curve. */
export function isCalibrated(manifest: Manifest): boolean {
	const cal = manifest.calibration;
	if (!cal || !cal.calibrated_leads) return false;
	const leads = new Set(cal.calibrated_leads);
	return manifest.leads_min.length > 0 && manifest.leads_min.every((l) => leads.has(l));
}
