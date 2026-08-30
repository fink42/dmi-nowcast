<script lang="ts">
	import { base } from '$app/paths';
	import { t } from '$lib/i18n';
	import SiteFooter from '$lib/components/SiteFooter.svelte';
</script>

<article class="prose">
	<h1>{t().error.title}</h1>
	<p>{t().error.body}</p>
	<p><a href={`${base}/`}>{t().error.backToMap}</a></p>
</article>
<SiteFooter />
