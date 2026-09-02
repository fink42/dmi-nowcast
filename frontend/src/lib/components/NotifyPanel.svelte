<script lang="ts">
	/**
	 * "Tell me when it starts raining here" — the notification controls, at the
	 * bottom of the forecast panel for one specific point.
	 *
	 * The panel is honest about the four ways this can be unavailable rather
	 * than hiding the button: an insecure origin, a browser without push, an
	 * iPhone that has not installed the site (one Share-sheet tap away from
	 * working), and a permission the user already refused. Each gets one muted
	 * line, because a button that silently does nothing is worse than a
	 * sentence explaining why there is no button.
	 */
	import { t } from '$lib/i18n';
	import { nowcast } from '$lib/nowcast/store.svelte';
	import { defaultPrefs, type PushPrefs } from '$lib/push/prefs';
	import { push } from '$lib/push/store.svelte';

	/** ~50 m: closer than this and it is the same place, not a new one. */
	const SAME_POINT_DEG = 0.0005;

	const config = $derived(push.config);
	const point = $derived(nowcast.point);
	const stored = $derived(push.stored);
	const busy = $derived(push.busy);
	const thresholds = $derived(config?.thresholdOptionsPct ?? []);
	const leads = $derived(config?.leadOptionsMin ?? []);

	/** The saved alert is at a different place than the one on screen. */
	const movedAway = $derived(
		!!stored &&
			!!point &&
			(Math.abs(stored.lat - point.lat) > SAME_POINT_DEG ||
				Math.abs(stored.lon - point.lon) > SAME_POINT_DEG)
	);

	let showForm = $state(false);
	/** The preferences being edited. */
	let draft = $state<PushPrefs>(defaultPrefs(null));

	/**
	 * While the form is closed, the draft tracks reality: the server defaults
	 * once the config lands, then whatever was actually saved. Opening the
	 * form freezes it, so an in-progress edit is never overwritten by a
	 * background poll — and closing it discards the edit, which is what
	 * "Fortryd" has to mean.
	 */
	$effect(() => {
		const base = push.stored?.prefs ?? defaultPrefs(push.config);
		if (!showForm) draft = { ...base, quietHours: { ...base.quietHours } };
	});

	/**
	 * Which operation "try again" repeats. Subscribing at the point on screen
	 * and saving preferences at the saved point are different requests, and
	 * after a failed "Flyt hertil" the retry has to be the move, not a
	 * re-save at the old coordinates.
	 */
	let lastAction: 'enable' | 'save' = $state('enable');

	/** A plain copy: the store keeps what it is given, and a reactive proxy
	 *  handed to it would keep changing under it as the form is edited. */
	const snapshot = (prefs: PushPrefs): PushPrefs => ({
		...prefs,
		quietHours: { ...prefs.quietHours }
	});

	async function enable() {
		if (!point) return;
		lastAction = 'enable';
		push.clearError();
		await push.subscribe(point.lat, point.lon, snapshot(draft));
		if (push.status === 'subscribed') showForm = false;
	}

	async function save() {
		lastAction = 'save';
		push.clearError();
		await push.updatePrefs(snapshot(draft));
		if (push.status === 'subscribed') showForm = false;
	}

	function retry() {
		void (lastAction === 'save' && stored ? save() : enable());
	}

	/**
	 * Nothing the user does here can succeed: the browser cannot take a
	 * subscription at all, or the permission is already refused. The state
	 * line above says why, and an error line with a retry underneath it would
	 * only offer a button that cannot work.
	 */
	const blocked = $derived(push.support !== 'supported' || push.permission === 'denied');
</script>

{#if config?.enabled && point && point.status === 'ready' && push.support !== 'unknown'}
	<section class="notify">
		<h3>{t().push.title}</h3>

		{#if push.support === 'insecure-context'}
			<p class="muted">{t().push.insecure}</p>
		{:else if push.support === 'ios-not-installed'}
			<p class="muted">{t().push.iosNotInstalled}</p>
		{:else if push.support === 'unsupported'}
			<p class="muted">{t().push.unsupported}</p>
		{:else if push.permission === 'denied'}
			<p class="muted">{t().push.denied}</p>
		{:else if stored}
			<p class="summary" aria-live="polite">
				{t().push.summary(
					t().panel.coordinates(stored.lat, stored.lon),
					stored.prefs.leadMin,
					stored.prefs.thresholdPct
				)}
				{#if stored.prefs.quietHours.enabled}
					<span class="muted"
						>{t().push.summaryQuiet(
							stored.prefs.quietHours.start,
							stored.prefs.quietHours.end
						)}</span
					>
				{/if}
			</p>

			{#if movedAway}
				<p class="muted">{t().push.movedAway}</p>
			{/if}

			{#if showForm}
				{@render prefsForm()}
				<div class="actions">
					<button type="button" class="primary" onclick={save} disabled={busy}>
						{busy ? t().push.working : t().push.save}
					</button>
					<button type="button" onclick={() => (showForm = false)} disabled={busy}>
						{t().push.cancel}
					</button>
				</div>
			{:else}
				<div class="actions">
					<button type="button" onclick={() => (showForm = true)} disabled={busy}>
						{t().push.edit}
					</button>
					{#if movedAway}
						<!-- Re-subscribing at the point on screen: same browser
						     subscription, new coordinates on the server row. -->
						<button type="button" class="primary" onclick={enable} disabled={busy}>
							{busy ? t().push.working : t().push.moveHere}
						</button>
					{/if}
					<button type="button" class="quiet" onclick={() => push.unsubscribe()} disabled={busy}>
						{t().push.stop}
					</button>
				</div>
			{/if}
		{:else if config.capacityReached}
			<p class="muted">{t().push.capacity}</p>
		{:else}
			<div class="actions">
				<button type="button" class="primary" onclick={enable} disabled={busy}>
					{busy ? t().push.working : t().push.enable}
				</button>
				<button
					type="button"
					aria-expanded={showForm}
					onclick={() => (showForm = !showForm)}
					disabled={busy}
				>
					{showForm ? t().push.hideSettings : t().push.showSettings}
				</button>
			</div>
			{#if showForm}
				{@render prefsForm()}
			{/if}
		{/if}

		{#if push.error && !blocked}
			<p class="error" aria-live="polite">
				{t().push.errors[push.error]}
				<button type="button" class="link" onclick={retry} disabled={busy}>{t().push.retry}</button>
			</p>
		{/if}
	</section>
{/if}

{#snippet prefsForm()}
	<div class="prefs">
		<div class="row">
			<span class="label" id="push-threshold-label">{t().push.thresholdLabel}</span>
			<div class="segmented" role="group" aria-labelledby="push-threshold-label">
				{#each thresholds as option (option)}
					<button
						type="button"
						aria-pressed={draft.thresholdPct === option}
						class:on={draft.thresholdPct === option}
						onclick={() => (draft.thresholdPct = option)}
					>
						{t().push.thresholdOption(option)}
					</button>
				{/each}
			</div>
		</div>

		<div class="row">
			<span class="label" id="push-lead-label">{t().push.leadLabel}</span>
			<div class="segmented" role="group" aria-labelledby="push-lead-label">
				{#each leads as option (option)}
					<button
						type="button"
						aria-pressed={draft.leadMin === option}
						class:on={draft.leadMin === option}
						onclick={() => (draft.leadMin = option)}
					>
						{t().push.leadOption(option)}
					</button>
				{/each}
			</div>
		</div>

		<div class="row">
			<label class="check">
				<input type="checkbox" bind:checked={draft.quietHours.enabled} />
				{t().push.quietLabel}
			</label>
			{#if draft.quietHours.enabled}
				<div class="times">
					<label>
						<span class="label">{t().push.quietFrom}</span>
						<input type="time" bind:value={draft.quietHours.start} required />
					</label>
					<label>
						<span class="label">{t().push.quietTo}</span>
						<input type="time" bind:value={draft.quietHours.end} required />
					</label>
				</div>
			{/if}
		</div>
	</div>
{/snippet}

<style>
	.notify {
		margin-top: 0.85rem;
		padding-top: 0.75rem;
		border-top: 1px solid var(--border);
	}

	h3 {
		margin: 0 0 0.4rem;
		font-size: 0.7rem;
		font-weight: 600;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		color: var(--muted);
	}

	p {
		margin: 0 0 0.5rem;
		font-size: 0.85rem;
	}

	.summary {
		font-size: 0.9rem;
	}

	.muted {
		color: var(--muted);
	}

	.error {
		color: var(--warn);
	}

	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
	}

	.actions button,
	.segmented button {
		border: 1px solid var(--border);
		background: var(--surface);
		color: var(--ink);
		border-radius: var(--radius);
		padding: 0.5rem 0.85rem;
		font-size: 0.85rem;
		cursor: pointer;
		min-height: 2.5rem;
	}

	.actions .primary {
		background: var(--accent);
		border-color: var(--accent);
		color: var(--accent-ink);
		font-weight: 600;
	}

	.actions .quiet {
		color: var(--muted);
	}

	button:disabled {
		opacity: 0.6;
		cursor: default;
	}

	.link {
		border: none;
		background: none;
		color: inherit;
		text-decoration: underline;
		padding: 0;
		font-size: inherit;
		cursor: pointer;
	}

	.prefs {
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
		margin: 0.6rem 0;
	}

	.row {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
	}

	.label {
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: var(--muted);
	}

	.segmented {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem;
	}

	.segmented button {
		flex: 1 1 auto;
		padding: 0.45rem 0.6rem;
		font-variant-numeric: tabular-nums;
	}

	.segmented button.on {
		border-color: var(--accent);
		color: var(--accent);
		font-weight: 600;
	}

	.check {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.85rem;
	}

	.check input {
		width: 1.1rem;
		height: 1.1rem;
		accent-color: var(--accent);
	}

	.times {
		display: flex;
		gap: 0.75rem;
		margin-top: 0.2rem;
	}

	.times label {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
	}

	.times input {
		border: 1px solid var(--border);
		background: var(--surface);
		color: var(--ink);
		border-radius: 8px;
		padding: 0.35rem 0.5rem;
		font: inherit;
		font-size: 0.85rem;
	}
</style>
