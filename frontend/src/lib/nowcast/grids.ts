/**
 * Fetching and decoding one cycle's product grids (p_rain per lead, ETA,
 * intensity). They are small — 432×496 levels, a few tens of kB each — and
 * cycle-stamped, so the browser cache does the right thing across polls and
 * we only decode when the cycle actually changes.
 */
import { artifactUrl, type ArtifactEntry, type Manifest } from './manifest';
import { decodeGray8Png, type Gray8Image } from './png';
import { productArtifacts, type DecodedGrids } from './sampler';

async function loadArtifact(entry: ArtifactEntry): Promise<Gray8Image> {
	const res = await fetch(artifactUrl(entry.filename));
	if (!res.ok) throw new Error(`${entry.filename}: HTTP ${res.status}`);
	return decodeGray8Png(new Uint8Array(await res.arrayBuffer()));
}

/**
 * Decode every grayscale product of a cycle. Throws if any of them fails —
 * the caller then falls back to the server's `/forecast` for point lookups.
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
	return grids;
}
