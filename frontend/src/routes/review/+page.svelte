<script lang="ts">
	/**
	 * A shell, and deliberately nothing else. The tool itself is behind a
	 * dynamic import inside a dev-only branch, so a production build contains
	 * neither `ReviewApp.svelte` nor anything it pulls in — the whole of
	 * `$lib/review/**`, the bundle parser, the tag vocabulary, the annotation
	 * client. A static `import ReviewApp from './ReviewApp.svelte'` at the top
	 * of this file would ship all of it, and would look exactly like a working
	 * deployment while doing so.
	 *
	 * **The guard is `import.meta.env.DEV`, not `dev` from `$app/environment`,
	 * and the difference is measured rather than stylistic.** Vite replaces
	 * `import.meta.env.DEV` with the literal `false` during *transform*, so
	 * rolldown parses `if (false)`, never records the dynamic import, and
	 * emits no chunk. `$app/environment`'s `dev` is a re-export of `esm-env`'s
	 * `DEV`, which is a module binding: rolldown resolves the branch as dead
	 * only after the module graph exists, by which time the chunk has been
	 * created. The result on this repo, checked by building it, was an orphan
	 * `chunks/*.js` carrying the entire review tool plus a `ReviewApp.css` —
	 * unreferenced by any other chunk, and listed in the service worker's
	 * precache, which installs it on every visitor's device. See
	 * `review-build.test.ts`, which asserts against the built artefact.
	 *
	 * `+page.ts` still uses `$app/environment`'s `dev` for the 404: there the
	 * value is read at runtime, so the folding question does not arise.
	 */
	const DEV = import.meta.env.DEV;
</script>

{#if DEV}
	{#await import('./ReviewApp.svelte')}
		<p class="review-boot">Loading the review tool…</p>
	{:then { default: ReviewApp }}
		<ReviewApp />
	{:catch loadError}
		<p class="review-boot">The review tool did not load: {loadError}</p>
	{/await}
{/if}

<style>
	.review-boot {
		padding: 1rem;
		color: var(--muted);
	}
</style>
