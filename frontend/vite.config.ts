import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

// Local development runs the app on Vite's dev server while the data comes
// from a real sidecar. Point VITE_SIDECAR_URL at it (e.g.
// `VITE_SIDECAR_URL=http://192.0.2.10:8081 npm run dev`) and the dev server
// proxies the data endpoints, so the app always uses same-origin paths —
// exactly what it does in production, where the sidecar serves the built
// files itself. With no proxy configured the endpoints 404 and the UI shows
// its "no data yet" state.
const sidecar = process.env.VITE_SIDECAR_URL;

// The dev-only event-review tool (src/routes/review) reads its bundle and
// stores its annotations through scripts/review_server.py, which binds to
// loopback. These two prefixes are always proxied: they exist nowhere else
// in the app, nothing is listening unless the reviewer started the server,
// and /review/ 404s in a production build, so an unconditional entry costs
// nothing and saves setting a second env var to use the tool.
const review = process.env.VITE_REVIEW_URL ?? 'http://127.0.0.1:8770';
const proxy = {
	...(sidecar
		? {
				'/nowcast': { target: sidecar, changeOrigin: true },
				'/forecast': { target: sidecar, changeOrigin: true },
				'/api': { target: sidecar, changeOrigin: true }
			}
		: {}),
	'/review-api': { target: review, changeOrigin: true },
	'/review-data': { target: review, changeOrigin: true }
};

export default defineConfig({
	plugins: [
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries.
				runes: ({ filename }: { filename: string }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			adapter: adapter({
				// SPA fallback: unknown paths render the app shell and the client
				// router takes over. Every route is also prerendered into its own
				// directory (src/routes/+layout.ts), so a plain static file server
				// resolves /about/ without needing the fallback.
				fallback: 'index.html',
				pages: 'build',
				assets: 'build'
			})
		})
	],
	// MapLibre creates a *module* worker, so the bundled worker must be ESM.
	worker: { format: 'es' },
	server: { proxy },
	preview: { proxy },
	test: {
		include: ['src/**/*.test.ts'],
		environment: 'node'
	}
});
