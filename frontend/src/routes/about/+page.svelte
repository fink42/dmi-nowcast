<script lang="ts">
	import { base } from '$app/paths';
	import { t } from '$lib/i18n';
	import { GITHUB_URL } from '$lib/links';
	import SiteFooter from '$lib/components/SiteFooter.svelte';
</script>

<article class="prose">
	<h1>{t().about.title}</h1>
	<p class="lead">{t().about.lead}</p>

	<h2>{t().about.honestyTitle}</h2>
	<p>{t().about.honesty}</p>

	<h2>{t().about.howTitle}</h2>
	<ol>
		{#each t().about.how as sentence, i (i)}
			<li>{sentence}</li>
		{/each}
	</ol>

	<h2>{t().about.qualityTitle}</h2>
	<p>
		{t().about.qualityBody}
		<a href={`${base}/quality/`}>{t().about.qualityLink}</a>
	</p>

	<h2>{t().about.limitsTitle}</h2>
	<ul>
		{#each t().about.limits as limit, i (i)}
			<li>{limit}</li>
		{/each}
	</ul>

	<h2>{t().about.sourceTitle}</h2>
	<p>
		{t().about.sourceBody}
		<a href={GITHUB_URL} target="_blank" rel="noreferrer">{t().about.sourceLink}</a>
	</p>
</article>
<SiteFooter />
