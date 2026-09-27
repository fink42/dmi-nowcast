import { describe, expect, it } from 'vitest';
import { inOrder } from './ordered';

interface Deferred<T> {
	promise: Promise<T>;
	resolve: (value: T) => void;
	reject: (err: unknown) => void;
}

function deferred<T>(): Deferred<T> {
	let resolve!: (value: T) => void;
	let reject!: (err: unknown) => void;
	const promise = new Promise<T>((res, rej) => {
		resolve = res;
		reject = rej;
	});
	return { promise, resolve, reject };
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

describe('inOrder', () => {
	it('starts up to the limit at once and yields in index order', async () => {
		const jobs = Array.from({ length: 6 }, () => deferred<number>());
		const startedAt: number[] = [];
		const gen = inOrder(
			jobs.length,
			(i) => {
				startedAt.push(i);
				return jobs[i].promise;
			},
			3
		);
		const first = gen.next();
		await tick();
		// Three downloads in flight before any of them has finished.
		expect(startedAt).toEqual([0, 1, 2]);

		// Later jobs finishing first changes nothing about the order.
		jobs[2].resolve(2);
		jobs[1].resolve(1);
		await tick();
		expect(startedAt).toEqual([0, 1, 2]);
		jobs[0].resolve(0);
		expect((await first).value).toBe(0);
		// Handing one out frees a slot.
		const second = gen.next();
		expect((await second).value).toBe(1);
		expect(startedAt).toEqual([0, 1, 2, 3, 4]);

		for (const [i, job] of jobs.entries()) job.resolve(i);
		const rest: number[] = [];
		for await (const value of gen) rest.push(value);
		expect(rest).toEqual([2, 3, 4, 5]);
	});

	it('fails at the failing job, after the ones before it', async () => {
		const gen = inOrder(
			3,
			(i) => (i === 1 ? Promise.reject(new Error('boom')) : Promise.resolve(i)),
			3
		);
		expect((await gen.next()).value).toBe(0);
		await expect(gen.next()).rejects.toThrow('boom');
	});

	it('disposes results that were produced but never handed out', async () => {
		const disposed: number[] = [];
		const gen = inOrder(5, (i) => Promise.resolve(i), 4, (v) => disposed.push(v));
		for await (const value of gen) {
			if (value === 1) break;
		}
		await tick();
		// 0 and 1 went to the consumer; 2..4 were started ahead and are released.
		expect(disposed.sort()).toEqual([2, 3, 4]);
	});

	it('does not raise unhandled rejections for jobs it never reached', async () => {
		const gen = inOrder(
			3,
			(i) => (i === 0 ? Promise.resolve(0) : Promise.reject(new Error(`late ${i}`))),
			3
		);
		expect((await gen.next()).value).toBe(0);
		await gen.return(undefined);
		await tick();
	});
});
