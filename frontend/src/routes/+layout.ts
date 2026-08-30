/**
 * Static, client-rendered site. Every route is prerendered into its own
 * directory (`/about/index.html`) so a plain file server resolves it, and
 * adapter-static also emits an `index.html` SPA fallback for anything else.
 * No SSR: the whole app is a map that only exists in the browser.
 */
export const prerender = true;
export const ssr = false;
export const trailingSlash = 'always';
