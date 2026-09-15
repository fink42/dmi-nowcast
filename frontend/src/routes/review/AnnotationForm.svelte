<script lang="ts">
	/**
	 * The judgement: one verdict, the mechanisms behind it, and how sure the
	 * reviewer is.
	 *
	 * The output of this whole exercise is a *ranked list of named failure
	 * modes*, so the form is stricter than the server is. The server counts a
	 * row as reviewed once it carries a verdict; `isComplete` also wants a
	 * mechanism tag, because a verdict with no tag records *that* something
	 * failed without recording *what* — which is the aggregate the tool exists
	 * to replace — and it wants a note behind the two `_other` codes, since
	 * "something else" with no sentence after it cannot be read six months
	 * later.
	 *
	 * Nothing is written through `bind:` on the draft. Every change goes
	 * through `updateDraft`/`toggleTag`, which is what keeps "saved" from
	 * staying on screen over an edited draft.
	 */
	import { NOTE_REQUIRED_TAGS } from '$lib/review/tags';
	import { review } from '$lib/review/store.svelte';
	import type { Verdict, VocabularyTag } from '$lib/review/schema';
	import { utcStamp } from './format';

	interface Props {
		/**
		 * The cause lists for this event's class, in the order the keyboard's
		 * `1`–`9` reach them. The page computes them, because it owns the
		 * shortcut and the two must not drift apart.
		 */
		groups: Array<{ group: string; label: string; tags: VocabularyTag[] }>;
	}
	let { groups }: Props = $props();

	const saved = $derived(review.savedAnnotation);
	const draft = $derived(review.draft);
	const problems = $derived(review.draftProblems);
	const needsNote = $derived(draft.tags.some((tag) => NOTE_REQUIRED_TAGS.includes(tag)));

	/** The digit that toggles a tag, or null past the ninth. */
	const digits = $derived(new Map(groups.flatMap((group) => group.tags).slice(0, 9).map((tag, index) => [tag.code, index + 1])));

	/**
	 * The bundle's vocabulary may name a verdict this client's union does not.
	 * Storing it as written is right — the server and `validateAnnotation`
	 * both check the code against the vocabulary, and silently dropping it
	 * would lose a judgement that was made.
	 */
	const chooseVerdict = (code: string) => review.updateDraft({ verdict: code as Verdict });
</script>

<section class="panel">
	<h2>Judgement</h2>

	{#if review.vocabularyIsFallback}
		<p class="warn">
			This bundle carries no readable <code>tags.json</code>, so the codes below are the
			ones compiled into this page (vocab_version {review.vocabulary.vocab_version}). A
			judgement made under a vocabulary the bundle does not know is a row the export
			cannot interpret.
		</p>
	{/if}

	<fieldset>
		<legend>Verdict <span class="required">required</span></legend>
		{#each review.vocabulary.verdicts as verdict (verdict.code)}
			<label class="choice">
				<input
					type="radio"
					name="review-verdict"
					checked={draft.verdict === verdict.code}
					onchange={() => chooseVerdict(verdict.code)}
				/>
				<span>
					<strong>{verdict.code}</strong>
					<span class="quiet">{verdict.description}</span>
				</span>
			</label>
		{/each}
	</fieldset>

	{#each groups as group (group.group)}
		<fieldset>
			<legend>{group.label}</legend>
			{#each group.tags as tag (tag.code)}
				<label class="choice tag">
					<input
						type="checkbox"
						checked={draft.tags.includes(tag.code)}
						onchange={() => review.toggleTag(tag.code)}
					/>
					<span>
						<strong>{tag.code}</strong>
						{#if digits.get(tag.code)}<kbd>{digits.get(tag.code)}</kbd>{/if}
						<span class="quiet">{tag.description}</span>
					</span>
				</label>
			{/each}
		</fieldset>
	{/each}

	<fieldset class="inline">
		<legend>Confidence</legend>
		<div class="confidence">
			{#each [1, 2, 3] as level (level)}
				<button
					type="button"
					class:on={draft.confidence === level}
					onclick={() => review.updateDraft({ confidence: level })}
					title={level === 1 ? 'a guess' : level === 2 ? 'probable' : 'sure'}>{level}</button
				>
			{/each}
			<button
				type="button"
				class="clear"
				onclick={() => review.updateDraft({ confidence: null })}
				disabled={draft.confidence === null}>clear</button
			>
			<span class="quiet">1 a guess · 3 sure. An integer, as the server stores it.</span>
		</div>
	</fieldset>

	<label class="choice">
		<input
			type="checkbox"
			checked={draft.needs_second_look}
			onchange={(event) => review.updateDraft({ needs_second_look: event.currentTarget.checked })}
		/>
		<span>Needs a second look</span>
	</label>

	<label class="note">
		<span>Note {#if needsNote}<span class="required">required by an “other” tag</span>{/if}</span>
		<textarea
			rows="3"
			value={draft.note}
			placeholder="What actually happened, in a sentence."
			oninput={(event) => review.updateDraft({ note: event.currentTarget.value })}
		></textarea>
	</label>

	{#if problems.length > 0}
		<ul class="problems">
			{#each problems as problem (problem.field + problem.message)}
				<li><strong>{problem.field}</strong>: {problem.message}</li>
			{/each}
		</ul>
	{/if}

	{#if review.saveError !== null}
		<div class="problems">
			<p><strong>{review.saveError.code}</strong>: {review.saveError.message}</p>
			{#each review.saveError.problems as problem (problem.field + problem.message)}
				<p>{problem.field}: {problem.message}</p>
			{/each}
			{#if review.saveError.kind === 'conflict'}
				<p>
					This row was edited elsewhere (the server holds revision
					{review.saveError.stored ?? '?'}).
					<button type="button" class="link" onclick={() => void review.refreshAnnotations()}
						>Re-read the stored judgements</button
					>
					before saving over it.
				</p>
			{:else if review.saveError.kind === 'offline'}
				<p>
					The annotation server did not answer. The bundle is static and still
					reviewable; nothing will be stored until
					<code>scripts/review_server.py</code> is running again.
				</p>
			{/if}
		</div>
	{/if}

	<div class="actions">
		<button
			type="button"
			class="save"
			disabled={review.saveStatus === 'saving' || review.selectedId === null}
			onclick={() => void review.save()}
		>
			{review.saveStatus === 'saving' ? 'Saving…' : 'Save'} <kbd>s</kbd>
		</button>
		<button
			type="button"
			class="discard"
			disabled={saved === null}
			onclick={() => void review.discard()}>Delete the stored judgement</button
		>
		<span class="state">
			{#if review.dirty}
				unsaved changes
			{:else if review.saveStatus === 'saved'}
				saved
			{:else if saved !== null}
				stored
			{:else}
				not judged
			{/if}
			{#if !review.draftComplete}
				· incomplete: a verdict and at least one mechanism tag
			{/if}
		</span>
	</div>

	{#if saved !== null}
		<p class="quiet">
			Stored by {saved.reviewer || 'an unnamed reviewer'} · #{saved.review_seq} in review order
			· revision {saved.revision} · {utcStamp(saved.updated_utc)}
			{#if saved.cursor_utc}· judged from the cursor at {utcStamp(saved.cursor_utc)}{/if}
			{#if saved.vocab_version !== review.vocabulary.vocab_version}
				<strong class="warn">
					· saved under vocab_version {saved.vocab_version}; this page shows
					{review.vocabulary.vocab_version}. Saving again re-stamps it with the current
					codes.
				</strong>
			{/if}
		</p>
	{/if}
</section>

<style>
	.panel {
		padding: 0.55rem 0.7rem 1.2rem;
	}

	h2 {
		font-size: 0.78rem;
		text-transform: uppercase;
		letter-spacing: 0.03em;
		color: var(--muted);
		margin: 0 0 0.35rem;
	}

	fieldset {
		border: 1px solid var(--border);
		border-radius: 8px;
		margin: 0 0 0.45rem;
		padding: 0.3rem 0.5rem 0.4rem;
	}

	fieldset.inline {
		padding-bottom: 0.3rem;
	}

	legend {
		font-size: 0.72rem;
		color: var(--muted);
		padding: 0 0.25rem;
	}

	.required {
		color: var(--warn);
		font-size: 0.66rem;
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}

	.choice {
		display: flex;
		gap: 0.35rem;
		align-items: baseline;
		font-size: 0.76rem;
		margin-bottom: 0.15rem;
		cursor: pointer;
	}

	.choice input {
		flex: 0 0 auto;
		margin: 0;
	}

	.choice strong {
		font-weight: 600;
	}

	.quiet {
		color: var(--muted);
		font-size: 0.7rem;
	}

	kbd {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.62rem;
		border: 1px solid var(--border);
		border-radius: 3px;
		padding: 0 0.2rem;
		margin: 0 0.15rem;
		background: var(--bg);
	}

	.confidence {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		flex-wrap: wrap;
	}

	.confidence button {
		border: 1px solid var(--border);
		background: var(--bg);
		color: var(--ink);
		border-radius: 6px;
		width: 1.7rem;
		cursor: pointer;
	}

	.confidence button.on {
		background: var(--accent);
		color: var(--accent-ink);
		border-color: var(--accent);
	}

	.confidence button.clear {
		width: auto;
		padding: 0 0.4rem;
		font-size: 0.7rem;
	}

	.note {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		font-size: 0.74rem;
		margin: 0.35rem 0;
	}

	textarea {
		width: 100%;
		font: inherit;
		font-size: 0.78rem;
		padding: 0.3rem 0.4rem;
		background: var(--bg);
		color: var(--ink);
		border: 1px solid var(--border);
		border-radius: 8px;
		resize: vertical;
	}

	.problems {
		margin: 0.3rem 0;
		padding: 0.3rem 0.5rem;
		border: 1px solid var(--warn);
		border-radius: 8px;
		color: var(--warn);
		font-size: 0.74rem;
		list-style: none;
	}

	.problems p {
		margin: 0 0 0.15rem;
	}

	.actions {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		flex-wrap: wrap;
		margin-top: 0.4rem;
	}

	.save {
		border: none;
		background: var(--accent);
		color: var(--accent-ink);
		border-radius: 8px;
		padding: 0.25rem 0.8rem;
		cursor: pointer;
	}

	.save:disabled,
	.discard:disabled {
		opacity: 0.5;
		cursor: default;
	}

	.discard {
		border: 1px solid var(--border);
		background: var(--surface);
		color: var(--muted);
		border-radius: 8px;
		padding: 0.25rem 0.6rem;
		font-size: 0.74rem;
		cursor: pointer;
	}

	.link {
		border: none;
		background: none;
		color: inherit;
		text-decoration: underline;
		padding: 0;
		cursor: pointer;
		font: inherit;
	}

	.state {
		font-size: 0.72rem;
		color: var(--muted);
	}

	.warn {
		color: var(--warn);
	}

	code {
		font-size: 0.72rem;
	}
</style>
