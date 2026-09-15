<script lang="ts">
	/**
	 * What the page says when there is nothing to review.
	 *
	 * Three states, kept apart because they have three different fixes and a
	 * single "could not load" would send the reviewer looking in the wrong
	 * place:
	 *
	 *  - **loading** — the four documents are in flight.
	 *  - **failed** — the fetch did not come back with a bundle. That is
	 *    either "the server is not running" or "the server is running and has
	 *    no bundle", and the page cannot always tell which, so it prints the
	 *    error verbatim and lists both remedies rather than guessing one.
	 *  - **schema** — a bundle whose `schema_version` is not the one this
	 *    client was written against. The parser is defensive and will render
	 *    what it can, so this is a stop sign rather than a wall: a reviewer
	 *    who knows the bundle is newer can go on, having been told that
	 *    sections may be silently missing.
	 *
	 * Nothing here invents a diagnosis. `review_server.py` binds to loopback
	 * and vite proxies `/review-data/` and `/review-api/` to it
	 * unconditionally, so a dead server answers with the SPA shell and a
	 * missing bundle answers 404 — two failures that look alike from here.
	 */
	import { REVIEW_SCHEMA_VERSION } from '$lib/review/schema';
	import { review } from '$lib/review/store.svelte';

	interface Props {
		/** Which gate to draw. The page decides; see `ReviewApp.svelte`. */
		kind: 'loading' | 'failed' | 'schema';
		/** Re-run the whole load — the server was started since. */
		onretry: () => void;
		/** Read a bundle of an unknown schema version anyway. */
		onproceed: () => void;
	}
	let { kind, onretry, onproceed }: Props = $props();

	const manifest = $derived(review.manifest);
</script>

<div class="gate">
	{#if kind === 'loading'}
		<h1>Opening the bundle…</h1>
		<p>manifest.json, events.json, tags.json and the stored judgements.</p>
	{:else if kind === 'failed'}
		<h1>No bundle</h1>
		<p class="error">{review.error ?? 'the bundle did not load'}</p>
		<p>Two things produce that message, and the page cannot tell them apart:</p>
		<ol>
			<li>
				<strong>The review server is not reachable.</strong> Start it:
				<code>python scripts/review_server.py &lt;bundle-dir&gt;</code>. It binds to
				loopback on port 8770; vite proxies <code>/review-data/</code> and
				<code>/review-api/</code> to it, so with nothing listening the proxy answers
				this page's own HTML and the parse fails.
			</li>
			<li>
				<strong>The server is running with no bundle.</strong> Build one on the VM with
				<code>scripts/build_review_bundle.py</code>, or
				<code>--fixture</code> for a synthetic one that needs no VM and no HDF5, and
				point the server at the directory.
			</li>
		</ol>
		<button type="button" onclick={onretry}>Try again</button>
	{:else}
		<h1>This bundle is a different schema</h1>
		<p>
			The bundle says <strong>schema_version {manifest?.schema_version ?? 'unknown'}</strong>;
			this page was written against <strong>{REVIEW_SCHEMA_VERSION}</strong>.
		</p>
		<p>
			The parser drops whatever it cannot read and keeps the rest, so going on is
			safe for the page and unsafe for the reading: a block that was renamed is now
			<em>missing</em>, and a missing block looks on screen exactly like a bundle that
			never had one. Rebuild the bundle, or update this client, if a panel is empty
			that should not be.
		</p>
		<div class="row">
			<button type="button" onclick={onproceed}>Review it anyway</button>
			<button type="button" class="quiet" onclick={onretry}>Reload</button>
		</div>
	{/if}
</div>

<style>
	.gate {
		max-width: 38rem;
		margin: 0 auto;
		padding: 2rem 1.2rem;
	}

	h1 {
		font-size: 1.3rem;
		margin: 0 0 0.6rem;
	}

	p,
	li {
		font-size: 0.92rem;
	}

	/* The raw error, verbatim: a paraphrase of it has cost people hours. */
	.error {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.82rem;
		color: var(--warn);
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 0.5rem 0.7rem;
		overflow-x: auto;
	}

	code {
		font-size: 0.82rem;
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: 4px;
		padding: 0 0.25rem;
	}

	ol {
		padding-left: 1.2rem;
	}

	li {
		margin-bottom: 0.6rem;
	}

	.row {
		display: flex;
		gap: 0.5rem;
	}

	button {
		border: 1px solid var(--border);
		background: var(--accent);
		color: var(--accent-ink);
		border-radius: var(--radius);
		padding: 0.4rem 0.9rem;
		cursor: pointer;
	}

	button.quiet {
		background: var(--surface);
		color: var(--ink);
	}
</style>
