/**
 * The review tool must not ship.
 *
 * The bundle it reads is a few hundred hand-picked events with station
 * locations and gauge records, the judgements are private working notes, and
 * the vocabulary names the pipeline's own failure modes. None of it belongs
 * in the public site's JavaScript, and every mechanism that keeps it out
 * fails **silently**: a missing `prerender = false` fails the build loudly,
 * but a static `import ReviewApp from './ReviewApp.svelte'` at the top of
 * `+page.svelte` would succeed, ship the whole of `$lib/review/**`, and look
 * exactly like a working deployment.
 *
 * So this checks the artefact rather than the source. It skips when there is
 * no `build/` — `npx vitest run` on a clean checkout must not demand a build
 * — and when there is one it asserts that Rollup really did drop the dynamic
 * import: no `/review` page, and no marker string from the tool anywhere in
 * the emitted chunks or in the service worker's precache.
 *
 * Run `npm run build` before trusting a pass.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = path.resolve(import.meta.dirname, '..', '..', '..');
const BUILD = path.join(ROOT, 'build');
const IMMUTABLE = path.join(BUILD, '_app', 'immutable');
const SERVICE_WORKER = path.join(BUILD, 'service-worker.js');

/**
 * Strings that exist nowhere but the review tool. One per layer, so a
 * failure says *which* part leaked: the bundle fetcher, the annotation
 * client, the tag vocabulary, or the app shell itself.
 */
const MARKERS = [
	'/review-data/',
	'/review-api/',
	'fa_cell_died',
	'dmi-nowcast-event-review'
];

/** Every file under a directory, recursively. */
function filesUnder(dir: string): string[] {
	const out: string[] = [];
	for (const entry of readdirSync(dir)) {
		const full = path.join(dir, entry);
		if (statSync(full).isDirectory()) out.push(...filesUnder(full));
		else out.push(full);
	}
	return out;
}

/** Only the text the browser would execute; the images are not the risk. */
const isCode = (file: string): boolean => /\.(js|mjs|css|map)$/.test(file);

describe.skipIf(!existsSync(BUILD))('the production build', () => {
	it('emits something, so the assertions below mean something', () => {
		expect(existsSync(IMMUTABLE), '_app/immutable').toBe(true);
		expect(filesUnder(IMMUTABLE).filter(isCode).length).toBeGreaterThan(0);
	});

	it('has no /review route', () => {
		// `+page.ts` sets prerender = false and its load 404s outside dev, so
		// the prerenderer must never have written a page here.
		expect(existsSync(path.join(BUILD, 'review'))).toBe(false);
	});

	it('names no file after a review component', () => {
		// Vite names an emitted CSS asset after the component it came from, and
		// a stylesheet carries none of the marker strings below — so the file
		// names are checked as well as the contents.
		const named = filesUnder(IMMUTABLE)
			.map((file) => path.basename(file))
			.filter((name) => /review/i.test(name));
		expect(named).toEqual([]);
	});

	it('carries no chunk of the review tool', () => {
		const offenders: string[] = [];
		for (const file of filesUnder(IMMUTABLE).filter(isCode)) {
			const text = readFileSync(file, 'utf8');
			for (const marker of MARKERS) {
				if (text.includes(marker)) offenders.push(`${path.relative(BUILD, file)} :: ${marker}`);
			}
		}
		expect(offenders).toEqual([]);
	});

	it('keeps the review tool out of the service worker precache', () => {
		if (!existsSync(SERVICE_WORKER)) return;
		// The worker precaches `[...build, ...files]`. Anything of the tool's
		// that reached `static/` would be installed on every visitor's device.
		const text = readFileSync(SERVICE_WORKER, 'utf8');
		for (const marker of MARKERS) expect(text, marker).not.toContain(marker);
		// The precache is a list of file names, so a leaked chunk shows up here
		// by name rather than by content. This is how the first build of this
		// route was caught shipping an orphan `ReviewApp.css`.
		expect(text).not.toMatch(/Review[A-Za-z]*\.(?:js|css)/);
	});
});
