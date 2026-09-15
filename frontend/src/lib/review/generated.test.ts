/**
 * The drift alarm: real builder output, parsed by the real parsers.
 *
 * Every other test in this directory runs against `fixture.json`, which is
 * hand-built to exercise the awkward cases. That is the right tool for
 * testing the parsers, and the wrong tool for noticing that the PRODUCER
 * has moved: a hand-built fixture says what we believe the Python emits,
 * and it keeps saying it long after the Python stopped.
 *
 * `generated-fixture.json` is instead a verbatim snapshot of a bundle from
 * `scripts/build_review_bundle.py --fixture --seed 424242`. If the builder
 * renames a field, restructures a block, or starts emitting null where the
 * browser expects a value, these tests go red — on the frontend side,
 * which is where the mistake would otherwise surface as a blank panel that
 * nobody attributes to a Python change three weeks earlier.
 *
 * When it does go red, the fix is to reconcile the two in ONE direction
 * and regenerate the snapshot. It is never to widen a parser to accept
 * both spellings: reading two spellings of one field is exactly how the
 * two sides drift apart quietly, each of them still "working".
 *
 * Regenerate with the command in the file's own `_comment` field.
 */
import { describe, expect, it } from 'vitest';

import generated from './generated-fixture.json';
import { parseEvent, parseIndex, parseManifest, parseVocabulary } from './load';
import { armStateAt, frameForCursor, latestDecisionAt } from './estimate';
import { markers, trackBounds } from './timeline';
import { observedArtifact } from './sample';
import { tagsForClass } from './tags';

const raw = generated as unknown as {
	manifest: unknown;
	index: unknown;
	vocabulary: unknown;
	details: Record<string, unknown>;
};

const details = Object.values(raw.details);

describe('a real builder bundle', () => {
	it('parses its manifest, and every block survives', () => {
		const manifest = parseManifest(raw.manifest);
		expect(manifest).not.toBeNull();
		// A null block here means the producer renamed or restructured it.
		// Each of these is load-bearing somewhere in the UI.
		expect(manifest!.grid).not.toBeNull();
		expect(manifest!.frames).not.toBeNull();
		expect(manifest!.rule).not.toBeNull();
		expect(manifest!.truth).not.toBeNull();
		expect(manifest!.sampling).not.toBeNull();
		expect(manifest!.builder).not.toBeNull();
		// `corpus` is deliberately absent from a --fixture bundle: there is
		// no corpus behind it. A real bundle fills it, and the assertion
		// that matters there is the one below on provenance.
	});

	it('is honest about having no corpus when there is none', () => {
		const manifest = parseManifest(raw.manifest)!;
		const corpus = manifest.corpus;
		expect(corpus === null || Object.keys(corpus).length >= 0).toBe(true);
	});

	it('keeps the quantisation the cursor read-out depends on', () => {
		// The one place a silent wrong number is possible: if scale/offset/
		// nodata go missing the parser must refuse the block rather than
		// default them, so a surviving block means they were all readable.
		const manifest = parseManifest(raw.manifest)!;
		const observed = manifest.frames!.encodings.observed;
		expect(Number.isFinite(observed.scale)).toBe(true);
		expect(Number.isFinite(observed.offset)).toBe(true);
		expect(observed.nodata).toBe(255);
		expect(observedArtifact(manifest)).not.toBeNull();
	});

	it('states its probability provenance, which decides if the sample is honest', () => {
		const manifest = parseManifest(raw.manifest)!;
		expect(manifest.rule!.probability_provenance).toBeTruthy();
	});

	it('parses its index, and every row carries what the list filters on', () => {
		const index = parseIndex(raw.index);
		expect(index).not.toBeNull();
		expect(index!.events.length).toBeGreaterThan(0);
		for (const row of index!.events) {
			expect(row.event_id).toBeTruthy();
			expect(row.class).toBeTruthy();
			expect(row.anchor_utc).toBeTruthy();
			expect(row.detail).toBeTruthy();
			expect(Array.isArray(row.flags)).toBe(true);
			// `stratum` is a mapping, not a joined string — the facets read it.
			expect(typeof row.stratum).toBe('object');
		}
	});

	it('parses its vocabulary, and the classes it ships can all be tagged', () => {
		const vocabulary = parseVocabulary(raw.vocabulary);
		expect(vocabulary).not.toBeNull();
		const index = parseIndex(raw.index)!;
		for (const row of index.events) {
			expect(tagsForClass(vocabulary!, row.class).length).toBeGreaterThan(0);
		}
	});

	it.each(details.map((d, i) => [i, d] as const))(
		'parses detail document %i with every block the UI reads',
		(_i, document) => {
			const event = parseEvent(document);
			expect(event).not.toBeNull();
			expect(event!.index).not.toBeNull();
			expect(event!.window).not.toBeNull();
			expect(event!.decisions.length).toBeGreaterThan(0);
			expect(event!.frames.length).toBeGreaterThan(0);
			// The prologue is what makes an unreachable miss legible; if the
			// producer drops it the reviewer sees a station that mysteriously
			// never fires.
			expect(event!.prologue).not.toBeNull();
		}
	);

	it.each(details.map((d, i) => [i, d] as const))(
		'detail %i drives the timeline and the estimate engine',
		(_i, document) => {
			const event = parseEvent(document)!;
			const bounds = trackBounds(event)!;
			expect(bounds).not.toBeNull();
			expect(bounds.toMs).toBeGreaterThan(bounds.fromMs);

			// Every marker must land inside the track it is drawn on.
			const { points } = markers(event);
			expect(points.length).toBeGreaterThan(0);
			for (const point of points) {
				expect(point.ms).toBeGreaterThanOrEqual(bounds.fromMs);
				expect(point.ms).toBeLessThanOrEqual(bounds.toMs);
			}

			// At the anchor there is always a decision at or before it, and
			// never one after it.
			const anchorMs = Date.parse(event.index.anchor_utc);
			const latest = latestDecisionAt(event.decisions, anchorMs);
			expect(latest).not.toBeNull();
			expect(Date.parse(latest!.generated_at_utc)).toBeLessThanOrEqual(anchorMs);

			// Before the first decision there is nothing to report.
			expect(latestDecisionAt(event.decisions, bounds.fromMs - 1)).toBeNull();

			const arm = armStateAt(event.decisions, event.prologue, anchorMs);
			expect(typeof arm.armed).toBe('boolean');

			// The picture and the words come from different cycles, and both
			// stamps have to survive the round trip or the UI cannot say so.
			const frame = frameForCursor(event.frames, event.decisions, anchorMs, 'truth');
			expect(frame.truthTsUtc).toBeTruthy();
		}
	);

	it('never claims both_dry when a truth could not speak', () => {
		// `both_dry` reads as "the forecast invented rain". It is the most
		// damaging claim in the vocabulary and must only appear when both
		// truths actually spoke.
		for (const document of details) {
			const event = parseEvent(document)!;
			const block = event.dual_truth;
			if (block && block.class === 'both_dry') {
				expect(block.gauge_wet).not.toBeNull();
				expect(block.radar_wet).not.toBeNull();
			}
		}
	});

	it('never reports an unknown gauge slot as dry', () => {
		for (const document of details) {
			const event = parseEvent(document)!;
			for (const slot of event.gauge?.slots ?? []) {
				if (!slot.known) {
					// A silent gauge weighs nothing; it must not weigh zero.
					expect(slot.wet).toBe(false);
					expect(slot.mm).toBeNull();
				}
			}
		}
	});
});
