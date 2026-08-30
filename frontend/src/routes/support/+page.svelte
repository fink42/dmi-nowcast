<script lang="ts">
	import { t } from '$lib/i18n';
	import { DONATION_LINKS } from '$lib/links';
	import SiteFooter from '$lib/components/SiteFooter.svelte';
</script>

<article class="prose">
	<h1>{t().support.title}</h1>
	<p class="lead">{t().support.lead}</p>

	{#if DONATION_LINKS.length > 0}
		<ul class="links">
			{#each DONATION_LINKS as link (link.url)}
				<li><a href={link.url} target="_blank" rel="noreferrer">{link.label}</a></li>
			{/each}
		</ul>
	{:else}
		<p>{t().support.body}</p>
		<h2>{t().support.soonTitle}</h2>
		<p>{t().support.soon}</p>
	{/if}

	<p>{t().support.thanks}</p>
</article>
<SiteFooter />

<style>
	.links {
		list-style: none;
		padding: 0;
		display: flex;
		gap: 0.75rem;
		flex-wrap: wrap;
	}

	.links a {
		display: inline-block;
		padding: 0.5rem 1rem;
		border: 1px solid var(--border);
		border-radius: 999px;
		text-decoration: none;
	}
</style>
