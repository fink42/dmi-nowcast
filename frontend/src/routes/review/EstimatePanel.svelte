<script lang="ts">
	/**
	 * What the service had at the cursor, and nothing that arrived later.
	 *
	 * Every number here comes from `latestDecisionAt` through the store, which
	 * never returns a decision from the future and returns null before the
	 * first one rather than the oldest. A window opens 90 minutes before the
	 * anchor and the first cycle inside it lands minutes later; showing that
	 * cycle's probability for the empty stretch before it would make the
	 * service look as if it had been warning all along.
	 *
	 * The ETA ticks. The sidecar's `eta_min` counts from the cycle's own
	 * instant, so printing it unchanged while the cursor moves leaves "rain in
	 * 12 min" on screen twelve minutes after the rain was due — and whether
	 * the warning was *timed* well is half of what the reviewer is judging.
	 */
	import { review } from '$lib/review/store.svelte';
	import {
		actionText,
		leadRows,
		marginPoints,
		minutesText,
		mmHText,
		NOT_MEASURED,
		pctText,
		signedPoints,
		utcTime
	} from './format';

	const decision = $derived(review.decision);
	const arm = $derived(review.armState);
	const leads = $derived(leadRows(decision));
	const margin = $derived(marginPoints(decision?.p_decision, decision?.threshold_pct));
	const action = $derived(actionText(decision?.replay?.action ?? null));
</script>

<section class="panel">
	<h2>The estimate at the cursor</h2>

	{#if decision === null}
		<p class="quiet">
			No estimate had been issued yet at this instant. The window opens before the
			first cycle inside it; the panel stays empty rather than borrowing a later
			cycle's numbers.
		</p>
	{:else}
		<p class="headline">
			<strong>{pctText(decision.p_decision) ?? NOT_MEASURED}</strong>
			<span class="quiet">against a threshold of</span>
			<strong>{decision.threshold_pct === null ? NOT_MEASURED : `${decision.threshold_pct} %`}</strong>
			{#if margin !== null}
				<span class="margin" class:over={margin >= 0}>{signedPoints(margin)} points</span>
			{/if}
		</p>
		<p class="quiet">
			{#if decision.over_threshold === null}
				The engine passed over this row: it was never compared against the threshold.
			{:else if decision.over_threshold}
				Over the threshold at this cycle.
			{:else}
				Under the threshold at this cycle.
			{/if}
			· decided on the <strong>{decision.p_decision_lead_min} min</strong> lead, from the
			<strong>{decision.p_decision_source ?? NOT_MEASURED}</strong> scale
			{#if decision.p_decision_source === 'curve'}
				— a fallback: the post-processed probability was absent, so this row was judged
				on the curve scale against a threshold fitted on the post-processed one.
			{/if}
		</p>

		<table>
			<thead>
				<tr>
					<th scope="col">lead</th>
					<th scope="col">p_rain (curve)</th>
					<th scope="col">p_post</th>
				</tr>
			</thead>
			<tbody>
				{#each leads as lead (lead.leadMin)}
					<tr class:decision-lead={lead.isDecisionLead}>
						<th scope="row">{lead.leadMin} min</th>
						<td>{pctText(lead.pRain) ?? 'unserved'}</td>
						<td>{pctText(lead.pPost) ?? 'unserved'}</td>
					</tr>
				{/each}
				{#if leads.length === 0}
					<tr><td colspan="3" class="quiet">no probabilities on this row</td></tr>
				{/if}
			</tbody>
		</table>

		<dl>
			<dt>ETA</dt>
			<dd>
				{#if review.etaMin === null}
					{NOT_MEASURED} — no arrival within the horizon at this cycle
				{:else}
					{minutesText(review.etaMin)} from the cursor
					<span class="quiet">
						(issued as {minutesText(decision.eta_min)}{#if decision.eta_arrival_utc}, arriving
							{utcTime(decision.eta_arrival_utc)}{/if})
					</span>
				{/if}
			</dd>

			<dt>Predicted intensity</dt>
			<dd>
				{mmHText(decision.intensity_mm_h, 2) ?? NOT_MEASURED}
				<span class="quiet">— column-max reflectivity reads high: an upper bound.</span>
			</dd>

			<dt>Observed at the station</dt>
			<dd>
				{mmHText(decision.observed_mm_h, 2) ?? NOT_MEASURED}
				{#if decision.forecast_now_mm_h !== null}
					<span class="quiet">· forecast now {mmHText(decision.forecast_now_mm_h, 2)}</span>
				{/if}
			</dd>

			<dt>Frame</dt>
			<dd>
				{utcTime(decision.radar_ts_utc)} · {minutesText(decision.frame_age_min) ?? NOT_MEASURED}
				old at the decision
				<span class="quiet">({decision.frame_age_source}) · row from the {decision.row_source}</span>
			</dd>
		</dl>

		<h3>What the engine did</h3>
		<p>
			<span class="badge {decision.replay?.action ?? 'untraced'}">{action.label}</span>
			<span class="quiet">{action.reading}</span>
			{#if decision.replay?.skipped_reason}
				<span class="quiet">· {decision.replay.skipped_reason}</span>
			{/if}
		</p>

		{#if arm !== null}
			<p class:warn={!arm.armed}>
				<strong>{arm.armed ? 'Armed' : 'Disarmed'}</strong>
				· streak {arm.streak} / {review.manifest?.rule?.persistence_obs ?? '?'}
				{#if !arm.armed}
					·
					{#if arm.minutesToRearm === null}
						re-arms in {NOT_MEASURED}: the dry clock's start is unknown
					{:else}
						re-arms in {minutesText(arm.minutesToRearm)} (of {arm.rearmAfterMin})
					{/if}
				{/if}
			</p>
			<p class="quiet">
				{#if arm.source === 'prologue'}
					From the prologue — before the first traced decision in this window.
				{:else if arm.source === 'unknown'}
					Neither traced nor stated by a prologue. Read as armed, because claiming the
					engine was blocked when nothing said so would excuse every miss in the window.
				{:else}
					From the replay trace at this cycle.
				{/if}
				{#if arm.belowSinceUtc}
					· dry clock since {utcTime(arm.belowSinceUtc)}; any over-threshold row resets it
					while disarmed.
				{/if}
				{#if arm.lastNotifyUtc}· last push {utcTime(arm.lastNotifyUtc)}{/if}
			</p>
			{#if arm.runBoundaryRearm}
				<p class="warn">
					The replay reset engine state at a coverage-run boundary at or before this
					instant, handing the station a re-arm the live service never had. A notify
					just after one of those is an artefact of replaying, not something the service
					would have sent.
				</p>
			{/if}
		{/if}

		{#if decision.stored !== null}
			<h3>What the archive recorded</h3>
			<p class="quiet">
				Shown for contrast, never used to classify: action
				{decision.stored.action ?? NOT_MEASURED} · armed after
				{decision.stored.armed_after === null ? NOT_MEASURED : String(decision.stored.armed_after)}
				· streak {decision.stored.streak_after ?? NOT_MEASURED} · threshold
				{decision.stored.threshold_pct === null ? NOT_MEASURED : `${decision.stored.threshold_pct} %`}
				· p_rain at the rule's lead {pctText(decision.stored.p_rain_at_rule_lead) ?? NOT_MEASURED}
			</p>
		{/if}
	{/if}
</section>

<style>
	.panel {
		border-bottom: 1px solid var(--border);
		padding: 0.55rem 0.7rem 0.7rem;
	}

	h2 {
		font-size: 0.78rem;
		text-transform: uppercase;
		letter-spacing: 0.03em;
		color: var(--muted);
		margin: 0 0 0.35rem;
	}

	h3 {
		font-size: 0.75rem;
		margin: 0.7rem 0 0.2rem;
	}

	p {
		margin: 0 0 0.25rem;
		font-size: 0.8rem;
	}

	.headline {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 0.35rem;
	}

	.headline strong {
		font-size: 1.15rem;
		font-variant-numeric: tabular-nums;
	}

	.margin {
		font-size: 0.78rem;
		border: 1px solid var(--border);
		border-radius: 999px;
		padding: 0 0.4rem;
		color: var(--muted);
	}

	.margin.over {
		color: var(--accent);
		border-color: var(--accent);
	}

	.quiet {
		color: var(--muted);
		font-size: 0.75rem;
	}

	.warn {
		color: var(--warn);
	}

	table {
		border-collapse: collapse;
		font-size: 0.76rem;
		width: 100%;
		margin: 0.3rem 0;
	}

	th,
	td {
		text-align: left;
		padding: 0.12rem 0.4rem 0.12rem 0;
		border-bottom: 1px solid var(--border);
		font-variant-numeric: tabular-nums;
	}

	thead th {
		color: var(--muted);
		font-weight: 500;
	}

	/* The lead the rule actually decided on, so the row that mattered is not
	   one of four that look alike. */
	tbody tr.decision-lead th,
	tbody tr.decision-lead td {
		font-weight: 650;
		background: var(--track);
	}

	dl {
		margin: 0.4rem 0 0;
		font-size: 0.78rem;
	}

	dt {
		font-weight: 600;
		margin-top: 0.3rem;
	}

	dd {
		margin: 0;
	}

	.badge {
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.03em;
		border: 1px solid var(--border);
		border-radius: 4px;
		padding: 0 0.3rem;
	}

	.badge.notify {
		color: var(--accent-ink);
		background: var(--accent);
		border-color: var(--accent);
	}

	.badge.already_raining,
	.badge.deferred_quiet {
		color: var(--warn);
		border-color: var(--warn);
	}
</style>
