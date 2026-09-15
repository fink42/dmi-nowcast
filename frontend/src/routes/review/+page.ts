/**
 * The gate that keeps the event-review tool out of production.
 *
 * Three things are load-bearing here and each of them is checked by
 * `review-build.test.ts` against a real build, because every one of them
 * fails silently:
 *
 *  - **`prerender = false`.** `routes/+layout.ts` turns prerendering on for
 *    the whole site. Without this line the prerenderer walks into this
 *    route, runs the load below, is handed a 404 and fails the build — in
 *    production, where `dev` is false and the 404 is the correct answer.
 *  - **`ssr = false`.** The tool is a map, a canvas and a loopback HTTP
 *    server; there is nothing to render on a server and the whole app is
 *    client-only anyway.
 *  - **the `dev` check.** `dev` is `esm-env`'s `DEV`, which Vite replaces
 *    with the literal `false` in a build, so Rollup can see that the
 *    dynamic import in `+page.svelte` is unreachable and emit no chunk for
 *    the app or for anything under `$lib/review/**`. The bundle and the
 *    judgements are local, private and gitignored; the code that reads them
 *    has no business being served to the public site's visitors.
 */
import { dev } from '$app/environment';
import { error } from '@sveltejs/kit';

export const prerender = false;
export const ssr = false;

export function load(): void {
	if (!dev) {
		throw error(404, 'The event review tool only runs under `npm run dev`.');
	}
}
