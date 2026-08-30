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

export type ProductName = 'p_rain' | 'eta' | 'intensity' | 'overlay';

export interface ArtifactEntry {
	filename: string;
	product: ProductName;
	lead_min: number | null;
	encoding: 'grayscale8' | 'rgba8';
	/** Quantisation, grayscale artifacts only: value = level * scale + offset. */
	scale?: number;
	offset?: number;
	/** Level reserved for "no value" (255). */
	nodata?: number;
	units?: string;
	shape: [number, number];
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
	n_members: number;
	leads_min: number[];
	grid: GridBlock;
	/** Null when the cycle produced no 500 m overlay frames. */
	overlay_grid: GridBlock | null;
	/** Null when the served probability grids are raw (uncalibrated). */
	calibration: CalibrationBlock | null;
	artifacts: ArtifactEntry[];
}

/** Schema version this client was written against. */
export const SUPPORTED_SCHEMA_VERSION = 1;

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

/** Overlay frames, ordered by lead (0 = "now"). */
export function overlayFrames(manifest: Manifest): ArtifactEntry[] {
	return manifest.artifacts
		.filter((a) => a.product === 'overlay')
		.sort((a, b) => (a.lead_min ?? 0) - (b.lead_min ?? 0));
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
