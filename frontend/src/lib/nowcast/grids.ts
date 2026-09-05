/**
 * Fetching and decoding one cycle's product grids (p_rain per lead, ETA,
 * intensity). They are small — 432×496 levels, a few tens of kB each — and
 * cycle-stamped, so the browser cache does the right thing across polls and
 * we only decode when the cycle actually changes.
 */
import { artifactUrl, type ArtifactEntry, type Manifest } from './manifest';
import { decodeGray8Png, type Gray8Image } from './png';
import {
	forecastSeriesArtifacts,
	motionArtifacts,
	observedArtifact,
	productArtifacts,
	type DecodedGrid,
	type DecodedGrids
} from './sampler';

async function loadArtifact(entry: ArtifactEntry): Promise<Gray8Image> {
	const res = await fetch(artifactUrl(entry.filename));
	if (!res.ok) throw new Error(`${entry.filename}: HTTP ${res.status}`);
	return decodeGray8Png(new Uint8Array(await res.arrayBuffer()));
}

/**
 * Decode every grayscale product of a cycle. Throws if a *required* one fails
 * — the caller then falls back to the server's `/forecast` for point lookups.
 * The cell-motion pair and the observation grid are not required and are
 * loaded separately.
 */
export async function loadGrids(manifest: Manifest): Promise<DecodedGrids> {
	const entries = productArtifacts(manifest);
	const images = await Promise.all(entries.map(loadArtifact));
	const grids: DecodedGrids = { pRain: new Map() };
	entries.forEach((entry, i) => {
		const image = images[i];
		if (entry.product === 'p_rain' && entry.lead_min !== null) {
			grids.pRain.set(entry.lead_min, { entry, image });
		} else if (entry.product === 'eta') {
			grids.eta = { entry, image };
		} else if (entry.product === 'intensity') {
			grids.intensity = { entry, image };
		}
	});
	// The extras are optional and independent, so none waits on another.
	const [motion, observed, forecastSeries] = await Promise.all([
		loadMotion(manifest),
		loadObserved(manifest),
		loadForecastSeries(manifest)
	]);
	grids.motion = motion;
	grids.observed = observed;
	grids.forecastSeries = forecastSeries;
	return grids;
}

/**
 * The cell-motion pair, when the cycle has one. Deliberately forgiving: a
 * manifest without motion grids (schema v1, or a cycle that could not
 * estimate one) and a pair that fails to download both end the same way —
 * no arrow in the panel, everything else untouched.
 */
async function loadMotion(manifest: Manifest): Promise<DecodedGrids['motion']> {
	const pair = motionArtifacts(manifest);
	if (!pair) return undefined;
	try {
		const [east, north] = await Promise.all(pair.map(loadArtifact));
		return { east: { entry: pair[0], image: east }, north: { entry: pair[1], image: north } };
	} catch (err) {
		console.warn('cell-motion grids unavailable', err);
		return undefined;
	}
}

/**
 * This cycle's observed rain field, when it has one. As forgiving as the
 * motion pair, and for the same reason: a manifest written before the product
 * existed and a download that fails both end as "no observation", which costs
 * the "it is raining here now" headline and leaves the forecast untouched.
 */
async function loadObserved(manifest: Manifest): Promise<DecodedGrids['observed']> {
	const entry = observedArtifact(manifest);
	if (!entry) return undefined;
	try {
		return { entry, image: await loadArtifact(entry) };
	} catch (err) {
		console.warn('observed rain grid unavailable', err);
		return undefined;
	}
}

/**
 * This cycle's advected rain field, one grid per overlay lead. As forgiving
 * as the two above and for a sharper reason: this series is what the headline
 * reads, and a partial series would be read as a shorter forecast rather than
 * as a failed download. So it is all of it or none of it — an empty array,
 * which the headline treats as "no field evidence" and answers from the
 * ensemble ETA instead. It never throws.
 */
export async function loadForecastSeries(manifest: Manifest): Promise<DecodedGrid[]> {
	const entries = forecastSeriesArtifacts(manifest);
	if (entries.length === 0) return [];
	try {
		const images = await Promise.all(entries.map(loadArtifact));
		return entries.map((entry, i) => ({ entry, image: images[i] }));
	} catch (err) {
		console.warn('advected rain grids unavailable', err);
		return [];
	}
}
