/**
 * Start a bounded number of jobs at once, hand their results back strictly in
 * order.
 *
 * The overlay loop needs both halves: the frames' downloads should overlap
 * (fetched one after another, a cycle's ~15 PNGs cost ~15 round trips before
 * the loop is complete), but the frames must still arrive in timeline order —
 * the loop starts playing on the prefix it has, and a hole in that prefix
 * would be a hole in the animation.
 *
 * Jobs started ahead of the consumer own resources (decoded bitmaps), so a
 * consumer that stops early — an abort, a failed frame, a `break` — must not
 * leak them: every result that was produced but never handed out is passed to
 * `dispose` once it settles.
 */

/**
 * Yield `start(0) … start(count - 1)` in index order, with at most `limit` of
 * them in flight (started and not yet handed out) at any time.
 *
 * A job that rejects fails the generator at its own position, not before: the
 * frames ahead of it are still yielded first. Rejections of jobs started ahead
 * and never reached are swallowed — the consumer stopped caring about them.
 */
export async function* inOrder<T>(
	count: number,
	start: (index: number) => Promise<T>,
	limit: number,
	dispose: (value: T) => void = () => {}
): AsyncGenerator<T> {
	const window = Math.max(1, Math.floor(limit));
	const started: Promise<T>[] = [];
	const fill = (upTo: number) => {
		while (started.length < Math.min(upTo, count)) {
			const job = start(started.length);
			// Settled ahead of its turn with nobody awaiting it yet: without a
			// handler this is an "unhandled rejection" even though the loop
			// below will await it in order.
			job.catch(() => {});
			started.push(job);
		}
	};
	// Index of the first job whose result has not been handed to the consumer.
	let taken = 0;
	try {
		fill(window);
		while (taken < count) {
			const value = await started[taken];
			taken++;
			// Refill before handing the value out, so the next downloads run
			// while the consumer is busy with this one.
			fill(taken + window);
			yield value;
		}
	} finally {
		for (let i = taken; i < started.length; i++) {
			started[i].then(dispose, () => {});
		}
	}
}
