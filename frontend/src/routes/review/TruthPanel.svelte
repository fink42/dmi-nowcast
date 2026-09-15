<script lang="ts">
	/**
	 * The truth side: what the gauge, the radar disc and the neighbours said —
	 * at the cursor, and across the window.
	 *
	 * One rule shapes every pixel of it: **a slot that was not reported is
	 * never dry.** `SlotState` is a tagged union with no boolean on it
	 * precisely so this component cannot read one carelessly, and every
	 * `unknown` here is drawn hatched and named in words. A silent gauge
	 * rendered as a dry one turns a hole in the record into evidence for a
	 * false alarm, which is the single mistake the tool exists to avoid.
	 *
	 * The dual-truth badge renders `null` as "Not known" and never as
	 * `both_dry`. "The forecast invented rain" is the most damaging claim this
	 * tool can make, and it must never appear because two instruments happened
	 * to be silent.
	 *
	 * The radar's vote is not independent evidence: it comes from the same
	 * instrument the forecast was made from. That sentence sits next to the
	 * badge, not in a footnote.
	 */
	import { compassPoint, windRelation } from '$lib/review/geometry';
	import { review } from '$lib/review/store.svelte';
	import { DEFAULT_SLOT_MIN, dualTruthLabel } from '$lib/review/truth';
	import { mmText, NOT_MEASURED, slotDetail, slotWord, utcTime } from './format';
	import { discSegments, stripSegments, type StripSegment } from './track';

	const detail = $derived(review.detail);
	const bounds = $derived(review.bounds);
	const dual = $derived(dualTruthLabel(detail?.dual_truth?.class ?? null));
	const slotMin = $derived(detail?.gauge?.slot_min ?? DEFAULT_SLOT_MIN);
	const cadenceMin = $derived(review.manifest?.frames?.cadence_min ?? DEFAULT_SLOT_MIN);

	const gaugeStrip = $derived(stripSegments(detail?.gauge?.slots ?? [], slotMin, bounds));
	const radarStrip = $derived(discSegments(detail?.radar_disc ?? null, bounds, slotMin, cadenceMin));
	/**
	 * A neighbour's strip, with its position relative to the flow.
	 *
	 * Upwind and downwind are different mechanisms: a wet neighbour UPWIND is
	 * a cell that diverted around this gauge, a wet one DOWNWIND is a cell
	 * that had already crossed it. Both come out of `geometry.ts` so the
	 * label and the map's own dots cannot disagree.
	 */
	const neighbourStrips = $derived(
		(detail?.neighbours?.stations ?? []).map((station) => ({
			station,
			relation: windRelation(station.bearing_deg, review.upwind?.bearingFromDeg ?? null),
			compass: compassPoint(station.bearing_deg),
			segments: stripSegments(station.slots, slotMin, bounds)
		}))
	);

	const pct = (position: number) => `${(position * 100).toFixed(3)}%`;

	const segmentTitle = (segment: StripSegment, unit: 'mm' | 'mm/h'): string => {
		const depth =
			segment.state === 'unknown'
				? 'no reading — this is not a dry stretch'
				: unit === 'mm'
					? (mmText(segment.mm, 1) ?? 'no value')
					: `${segment.mm === null ? 'no value' : segment.mm.toFixed(1)} mm/h`;
		return `${segment.state} · ${utcTime(segment.fromUtc)} → ${utcTime(segment.toUtc)} · ${segment.slots} slot(s) · ${depth}`;
	};

	/** Tri-state in words: null is "not known", never false. */
	const vote = (value: boolean | null | undefined): string =>
		value === true ? 'wet' : value === false ? 'dry' : 'not known';
</script>

<section class="panel">
	<h2>Truth</h2>

	<p class="badge-line">
		<span class="badge" class:unknown={dual.code === null}>{dual.label}</span>
		{#if detail?.index.neighbour_n_known !== undefined}
			<span class="quiet">{detail.index.neighbour_n_known} neighbours with a known verdict</span>
		{/if}
	</p>
	<p class="reading">{dual.reading}</p>
	<p class="quiet">{dual.caveat}</p>

	{#if detail?.dual_truth}
		<p class="quiet">
			gauge {vote(detail.dual_truth.gauge_wet)} · radar {vote(detail.dual_truth.radar_wet)} ·
			neighbours {vote(detail.dual_truth.neighbour_wet)} · judged over
			{utcTime(detail.dual_truth.window_used.from_utc)} →
			{utcTime(detail.dual_truth.window_used.to_utc)}
			({detail.dual_truth.window_used.definition})
		</p>
	{:else}
		<p class="quiet">This bundle carries no dual-truth block for the event.</p>
	{/if}

	<h3>At the cursor</h3>
	<ul class="states">
		<li>
			<span class="what">gauge</span>
			<span class="state {review.gaugeState?.state ?? 'unknown'}">{slotWord(review.gaugeState)}</span>
			<span class="quiet">{slotDetail(review.gaugeState, 'mm')}</span>
		</li>
		<li>
			<span class="what">radar disc</span>
			<span class="state {review.radarState?.state ?? 'unknown'}">{slotWord(review.radarState)}</span>
			<span class="quiet">
				{slotDetail(review.radarState, 'mm/h')}
				{#if detail?.radar_disc}
					· {detail.radar_disc.statistic} over {detail.radar_disc.disc_radius_m} m, wet at
					{detail.radar_disc.threshold_mm_h} mm/h
				{/if}
			</span>
		</li>
		<li>
			<span class="what">neighbours</span>
			<span class="quiet">
				{#if review.neighbourStates.length === 0}
					no neighbour series in this bundle
				{:else}
					{review.neighbourStates.filter((entry) => entry.state.state === 'wet').length} wet,
					{review.neighbourStates.filter((entry) => entry.state.state === 'dry').length} dry,
					{review.neighbourStates.filter((entry) => entry.state.state === 'unknown').length}
					not known — a wet neighbour says rain existed in the area, not that it rained here.
				{/if}
			</span>
		</li>
	</ul>

	<h3>Across the window</h3>
	<div class="strips">
		<div class="strip-row">
			<span class="label">gauge</span>
			<div class="strip">
				{#each gaugeStrip as segment, i (i)}
					<span
						class="seg {segment.state}"
						style:left={pct(segment.from)}
						style:width={pct(segment.to - segment.from)}
						title={segmentTitle(segment, 'mm')}
					></span>
				{/each}
				{#if review.cursorPosition !== null}
					<span class="cursor" style:left={pct(review.cursorPosition)}></span>
				{/if}
			</div>
		</div>

		<div class="strip-row">
			<span class="label">radar disc</span>
			<div class="strip">
				{#each radarStrip as segment, i (i)}
					<span
						class="seg {segment.state}"
						style:left={pct(segment.from)}
						style:width={pct(segment.to - segment.from)}
						title={segmentTitle(segment, 'mm/h')}
					></span>
				{/each}
				{#if review.cursorPosition !== null}
					<span class="cursor" style:left={pct(review.cursorPosition)}></span>
				{/if}
			</div>
		</div>

		{#each neighbourStrips as entry (entry.station.station_id)}
			<div class="strip-row">
				<span
					class="label"
					title="{entry.station.station_name} · {entry.station.distance_km.toFixed(1)} km{entry.compass
						? ` ${entry.compass}`
						: ''}{entry.relation ? ` · ${entry.relation} of the flow` : ''}{entry.station
						.wet_in_window === true
						? ' · wet somewhere in the window'
						: ''}"
					>{entry.station.station_name} · {entry.station.distance_km.toFixed(0)} km
					{#if entry.relation}<span class="relation">{entry.relation}</span>{/if}</span
				>
				<div class="strip">
					{#each entry.segments as segment, i (i)}
						<span
							class="seg {segment.state}"
							style:left={pct(segment.from)}
							style:width={pct(segment.to - segment.from)}
							title={segmentTitle(segment, 'mm')}
						></span>
					{/each}
					{#if review.cursorPosition !== null}
						<span class="cursor" style:left={pct(review.cursorPosition)}></span>
					{/if}
				</div>
			</div>
		{/each}
	</div>
	<p class="legend quiet">
		<i class="key wet"></i>wet <i class="key dry"></i>dry <i class="key unknown"></i>did not
		report — hatched, because it is not a dry stretch
	</p>

	{#if detail?.gauge}
		<p class="quiet">
			{detail.gauge.wet_slots_in_window} wet slot(s) in the window · gauge known until
			{utcTime(detail.gauge.known_until_utc) ?? NOT_MEASURED} — past that instant the gauge is
			silent, not dry.
		</p>
		{#if detail.gauge.onsets.length > 0}
			<ul class="onsets">
				{#each detail.gauge.onsets as onset, i (i)}
					<li>
						onset {utcTime(onset.onset_utc)} · {mmText(onset.two_slot_mm, 1) ?? NOT_MEASURED} over
						two slots
						{#if onset.is_event}<strong>· this event</strong>{/if}
						{#if !onset.in_window}<span class="quiet">· outside the window</span>{/if}
					</li>
				{/each}
			</ul>
		{/if}
		<p class="quiet">
			Slots are stamped at their END: a slot ending 13:50 covers 13:40–13:50, so the
			first drop fell somewhere in those ten minutes and every measured lead is biased
			that far negative.
		</p>
	{/if}

	{#if detail?.radar_disc && detail.radar_disc.first_wet_utc}
		<p class="quiet">Disc first wet {utcTime(detail.radar_disc.first_wet_utc)}.</p>
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

	.quiet {
		color: var(--muted);
		font-size: 0.74rem;
	}

	.badge-line {
		display: flex;
		align-items: baseline;
		gap: 0.4rem;
		flex-wrap: wrap;
	}

	.badge {
		font-size: 0.82rem;
		font-weight: 650;
		border: 1px solid var(--ink);
		border-radius: 4px;
		padding: 0 0.35rem;
	}

	/* "Not known" is its own look. It must never read as a quadrant, and
	   least of all as `both_dry`. */
	.badge.unknown {
		border-style: dashed;
		color: var(--muted);
		border-color: var(--muted);
	}

	.reading {
		font-size: 0.78rem;
	}

	.states {
		list-style: none;
		margin: 0;
		padding: 0;
		font-size: 0.78rem;
	}

	.states li {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		align-items: baseline;
		margin-bottom: 0.15rem;
	}

	.what {
		min-width: 5.5rem;
		color: var(--muted);
		font-size: 0.72rem;
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}

	.state {
		font-weight: 650;
	}

	.state.wet {
		color: #2f7fd6;
	}

	.state.unknown {
		color: #8a6ea8;
	}

	.strips {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
	}

	.strip-row {
		display: flex;
		align-items: center;
		gap: 0.35rem;
	}

	.label {
		flex: 0 0 9rem;
		font-size: 0.68rem;
		color: var(--muted);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	/* Upwind or downwind of the flow: the word that decides whether a wet
	   neighbour means "it went round this gauge" or "it had already been
	   here". */
	.relation {
		color: var(--ink);
	}

	.strip {
		position: relative;
		flex: 1 1 auto;
		height: 0.6rem;
		background: var(--bg);
		border-radius: 3px;
		overflow: hidden;
	}

	.seg {
		position: absolute;
		top: 0;
		bottom: 0;
	}

	.seg.wet {
		background: #2f7fd6;
	}

	.seg.dry {
		background: var(--track);
	}

	/* Hatched, and never the same as dry: a gauge that did not report is not
	   a gauge that reported nothing. */
	.seg.unknown {
		background-image: repeating-linear-gradient(-45deg, #8a6ea8 0 2px, transparent 2px 5px);
	}

	.cursor {
		position: absolute;
		top: 0;
		bottom: 0;
		width: 1.5px;
		background: var(--accent);
		transform: translateX(-50%);
	}

	.legend {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		margin-top: 0.25rem;
	}

	.key {
		display: inline-block;
		width: 0.9rem;
		height: 0.6rem;
		border-radius: 2px;
		vertical-align: -1px;
	}

	.key.wet {
		background: #2f7fd6;
	}

	.key.dry {
		background: var(--track);
	}

	.key.unknown {
		background-image: repeating-linear-gradient(-45deg, #8a6ea8 0 2px, transparent 2px 5px);
	}

	.onsets {
		list-style: none;
		margin: 0.2rem 0;
		padding: 0;
		font-size: 0.76rem;
	}
</style>
