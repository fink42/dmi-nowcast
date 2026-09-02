/**
 * Unfolding a minimised forecast panel.
 *
 * The trap this guards is the five-minute one: the store hands the panel a
 * brand-new `point` object every cycle for the *same* coordinates, so a rule
 * written against object identity would silently undo the user's minimise
 * every time a cycle landed.
 */
import { describe, expect, it } from 'vitest';
import { shouldExpandOnPointChange } from './panel';

const CPH = { lat: 55.676, lon: 12.568 };

describe('shouldExpandOnPointChange', () => {
	it('unfolds for the first point after none', () => {
		expect(shouldExpandOnPointChange(null, CPH)).toBe(true);
	});

	it('leaves a minimised panel alone when the same point is re-sampled', () => {
		// A different object, the same place: this is what every cycle does.
		expect(shouldExpandOnPointChange(CPH, { ...CPH })).toBe(false);
	});

	it('unfolds for a different point', () => {
		expect(shouldExpandOnPointChange(CPH, { lat: 56.163, lon: 10.204 })).toBe(true);
		// One coordinate moving is enough.
		expect(shouldExpandOnPointChange(CPH, { ...CPH, lon: 12.569 })).toBe(true);
		expect(shouldExpandOnPointChange(CPH, { ...CPH, lat: 55.677 })).toBe(true);
	});

	it('does nothing when the point goes away', () => {
		expect(shouldExpandOnPointChange(CPH, null)).toBe(false);
		expect(shouldExpandOnPointChange(null, null)).toBe(false);
	});
});
