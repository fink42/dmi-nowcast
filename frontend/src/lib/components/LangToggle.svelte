<script lang="ts">
	import { locale, setLocale, t } from '$lib/i18n';

	const next = $derived(locale() === 'da' ? 'en' : 'da');
	const label = $derived(next === 'en' ? t().lang.switchToEn : t().lang.switchToDa);
</script>

<button type="button" class="toggle" onclick={() => setLocale(next)} aria-label={label} title={label}>
	<span class:active={locale() === 'da'}>DA</span>
	<span aria-hidden="true">/</span>
	<span class:active={locale() === 'en'}>EN</span>
</button>

<style>
	.toggle {
		display: inline-flex;
		gap: 0.25rem;
		align-items: center;
		border: 1px solid var(--border);
		background: none;
		color: var(--muted);
		border-radius: 999px;
		padding: 0.2rem 0.6rem;
		font-size: 0.75rem;
		font-weight: 600;
		letter-spacing: 0.03em;
		cursor: pointer;
	}

	.active {
		color: var(--ink);
	}
</style>
