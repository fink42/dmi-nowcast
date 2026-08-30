<script lang="ts">
	import { t } from '$lib/i18n';
	import { GITHUB_ISSUES_URL } from '$lib/links';
	import SiteFooter from '$lib/components/SiteFooter.svelte';
</script>

<article class="prose">
	<h1>{t().privacy.title}</h1>
	<p class="lead">{t().privacy.lead}</p>

	<ul>
		{#each t().privacy.points as point, i (i)}
			<li>{point}</li>
		{/each}
	</ul>

	<h2>{t().privacy.contactTitle}</h2>
	<p>
		{t().privacy.contactBody}
		<a href={GITHUB_ISSUES_URL} target="_blank" rel="noreferrer">GitHub</a>
	</p>
</article>
<SiteFooter />
