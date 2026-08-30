<script lang="ts">
	import { t } from '$lib/i18n';
	import {
		DMI_OPEN_DATA_URL,
		NATURAL_EARTH_URL,
		OSM_COPYRIGHT_URL,
		PROTOMAPS_URL
	} from '$lib/links';
	import SiteFooter from '$lib/components/SiteFooter.svelte';
</script>

<article class="prose">
	<h1>{t().data.title}</h1>

	<h2>{t().data.radarTitle}</h2>
	<p>
		{t().data.radarBody}
		<a href={DMI_OPEN_DATA_URL} target="_blank" rel="noreferrer">{t().data.radarLink}</a>
	</p>

	<h2>{t().data.basemapTitle}</h2>
	<p>
		{t().data.basemapBody}
		<a href={OSM_COPYRIGHT_URL} target="_blank" rel="noreferrer">{t().data.basemapLinkOsm}</a>
		·
		<a href={PROTOMAPS_URL} target="_blank" rel="noreferrer">{t().data.basemapLinkProtomaps}</a>
	</p>

	<h2>{t().data.boundaryTitle}</h2>
	<p>
		{t().data.boundaryBody}
		<a href={NATURAL_EARTH_URL} target="_blank" rel="noreferrer">{t().data.boundaryLink}</a>
	</p>

	<h2>{t().data.methodTitle}</h2>
	<p>{t().data.methodBody}</p>
</article>
<SiteFooter />
