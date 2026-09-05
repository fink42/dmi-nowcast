<script lang="ts">
	import '../app.css';
	import { browser } from '$app/environment';
	import { page } from '$app/state';
	import { base } from '$app/paths';
	import { initLocale, t } from '$lib/i18n';
	import LangToggle from '$lib/components/LangToggle.svelte';

	let { children } = $props();

	// Before the first render of any child, so the map builds its style with
	// the right label language.
	if (browser) initLocale();

	const isMap = $derived(page.url.pathname === `${base}/` || page.url.pathname === base);
	/**
	 * The one content page that states its own title: it is the page people
	 * link to, and “Regnradar — kortsigtet regnvarsel” tells a reader nothing
	 * about what they are about to open. Set here rather than in the page's
	 * own <svelte:head>, because the browser takes the *first* <title> in the
	 * document, which is always the layout's.
	 */
	const isQuality = $derived(page.url.pathname === `${base}/quality/`);
</script>

<svelte:head>
	<title
		>{isQuality
			? `${t().quality.title} — ${t().site.title}`
			: `${t().site.title} — ${t().site.tagline}`}</title
	>
	<meta
		name="description"
		content={isQuality ? t().quality.description : t().site.description}
	/>
</svelte:head>

<div class="shell" class:map-page={isMap}>
	<header class="topbar">
		<a class="brand" href={`${base}/`}>
			<svg viewBox="0 0 24 24" aria-hidden="true" class="drop">
				<path
					d="M12 2.5c4 5.2 6.2 8.6 6.2 11.4A6.2 6.2 0 0 1 12 20.1a6.2 6.2 0 0 1-6.2-6.2C5.8 11.1 8 7.7 12 2.5z"
				/>
			</svg>
			<span>{t().site.title}</span>
		</a>
		<span class="tagline">{t().site.tagline}</span>
		<nav class="topnav" aria-label={t().nav.menu}>
			<a href={`${base}/quality/`} aria-current={isQuality ? 'page' : undefined}
				>{t().nav.quality}</a
			>
		</nav>
		<LangToggle />
	</header>

	<main>
		{@render children()}
	</main>
</div>

<style>
	.shell {
		display: flex;
		flex-direction: column;
		min-height: 100dvh;
	}

	.shell.map-page {
		height: 100dvh;
		overflow: hidden;
	}

	.topbar {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 0.75rem;
		background: var(--surface);
		border-bottom: 1px solid var(--border);
		flex: 0 0 auto;
		z-index: 2;
	}

	.brand {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		font-weight: 650;
		text-decoration: none;
		color: var(--ink);
		font-size: 0.95rem;
	}

	.drop {
		width: 1.1rem;
		height: 1.1rem;
		fill: var(--accent);
	}

	.topnav a {
		font-size: 0.8rem;
		color: var(--ink);
		text-decoration: none;
		white-space: nowrap;
	}

	.topnav a[aria-current='page'] {
		color: var(--accent);
		font-weight: 600;
	}

	.tagline {
		margin-left: auto;
		font-size: 0.75rem;
		color: var(--muted);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	main {
		flex: 1 1 auto;
		min-height: 0;
		display: flex;
		flex-direction: column;
	}

	@media (max-width: 26rem) {
		.tagline {
			display: none;
		}
	}
</style>
