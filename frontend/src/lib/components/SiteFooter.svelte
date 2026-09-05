<script lang="ts">
	/**
	 * The footer carries three things: the site's navigation (it lives here
	 * rather than in the header, because a phone header cannot hold five links
	 * and a language toggle), the disclaimer DMI's terms and plain honesty
	 * both call for, and the source link.
	 */
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import { t } from '$lib/i18n';
	import { GITHUB_URL } from '$lib/links';

	const links = $derived([
		{ href: `${base}/`, label: t().nav.map },
		{ href: `${base}/about/`, label: t().nav.about },
		{ href: `${base}/quality/`, label: t().nav.quality },
		{ href: `${base}/data/`, label: t().nav.data },
		{ href: `${base}/privacy/`, label: t().nav.privacy },
		{ href: `${base}/support/`, label: t().nav.support }
	]);
</script>

<footer>
	<nav aria-label={t().nav.menu}>
		{#each links as link (link.href)}
			<a href={link.href} aria-current={page.url.pathname === link.href ? 'page' : undefined}
				>{link.label}</a
			>
		{/each}
	</nav>
	<p class="disclaimer">
		<strong>{t().footer.disclaimer}</strong>
		<span>{t().footer.official}</span>
	</p>
	<p class="meta">
		<span>{t().footer.attribution}</span>
		<span class="sep">·</span>
		<a href={GITHUB_URL} target="_blank" rel="noreferrer">{t().footer.source}</a>
	</p>
</footer>

<style>
	footer {
		padding: 0.55rem 1rem 1rem;
		font-size: 0.75rem;
		color: var(--muted);
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
	}

	nav {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem 0.9rem;
		margin-bottom: 0.15rem;
	}

	nav a {
		color: var(--ink);
		text-decoration: none;
		font-size: 0.82rem;
	}

	nav a[aria-current='page'] {
		color: var(--accent);
		font-weight: 600;
	}

	.disclaimer strong {
		color: var(--ink);
	}

	.disclaimer span {
		margin-left: 0.35rem;
	}

	.meta {
		margin: 0;
		display: flex;
		gap: 0.35rem;
		flex-wrap: wrap;
	}

	.sep {
		opacity: 0.5;
	}

	p {
		margin: 0;
	}

	a {
		color: inherit;
	}
</style>
