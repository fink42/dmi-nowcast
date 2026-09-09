import { describe, expect, it } from 'vitest';
import { initialPoint } from './initial';

const SUB = { lat: 55.352, lon: 10.347 };

describe('initialPoint', () => {
	it('opens on the subscribed point when nothing else names one', () => {
		expect(initialPoint('', SUB)).toEqual({ ...SUB, source: 'subscription' });
		expect(initialPoint('?foo=bar', SUB)).toEqual({ ...SUB, source: 'subscription' });
	});

	it('lets a deep link win over the subscription', () => {
		expect(initialPoint('?lat=56.04&lon=9.08', SUB)).toEqual({
			lat: 56.04,
			lon: 9.08,
			source: 'deep-link'
		});
	});

	it('opens on nothing when there is neither', () => {
		expect(initialPoint('', null)).toBeNull();
		expect(initialPoint('?lat=&lon=', null)).toBeNull();
	});

	it('ignores a stored copy whose coordinates are not numbers', () => {
		expect(initialPoint('', { lat: Number.NaN, lon: 10 })).toBeNull();
		expect(initialPoint('', { lat: 55, lon: Number.POSITIVE_INFINITY })).toBeNull();
	});
});
