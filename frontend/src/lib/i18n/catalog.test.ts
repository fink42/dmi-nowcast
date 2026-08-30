/**
 * Catalog completeness. TypeScript already forces English to have every key
 * Danish has; this covers what types cannot: keys that exist only in English,
 * placeholder-shaped mismatches, empty strings, and lists that lost an item
 * in translation (an array of three limits must stay three in both).
 */
import { describe, expect, it } from 'vitest';
import { da } from './da';
import { en } from './en';
import { LOCALES } from './types';

type Node = Record<string, unknown>;

/** Every leaf path, with the kind of value it holds. */
function describeLeaves(node: Node, prefix = ''): Map<string, string> {
	const out = new Map<string, string>();
	for (const [key, value] of Object.entries(node)) {
		const path = prefix ? `${prefix}.${key}` : key;
		if (typeof value === 'function') {
			out.set(path, `function/${value.length}`);
		} else if (Array.isArray(value)) {
			out.set(path, `array/${value.length}`);
		} else if (value && typeof value === 'object') {
			for (const [k, v] of describeLeaves(value as Node, path)) out.set(k, v);
		} else {
			out.set(path, typeof value);
		}
	}
	return out;
}

const daLeaves = describeLeaves(da as unknown as Node);
const enLeaves = describeLeaves(en as unknown as Node);

describe('i18n catalogs', () => {
	it('covers both locales the app offers', () => {
		expect([...LOCALES].sort()).toEqual(['da', 'en']);
	});

	it('has exactly the same keys in Danish and English', () => {
		const daKeys = [...daLeaves.keys()].sort();
		const enKeys = [...enLeaves.keys()].sort();
		expect(enKeys).toEqual(daKeys);
	});

	it('keeps the same value kinds, argument counts and list lengths', () => {
		for (const [path, kind] of daLeaves) {
			expect(`${path}: ${enLeaves.get(path)}`).toBe(`${path}: ${kind}`);
		}
	});

	it('has no empty strings', () => {
		for (const [name, catalog] of [
			['da', da],
			['en', en]
		] as const) {
			const walk = (node: Node, prefix = '') => {
				for (const [key, value] of Object.entries(node)) {
					const path = prefix ? `${prefix}.${key}` : key;
					if (typeof value === 'string') {
						expect(value.trim(), `${name}.${path}`).not.toBe('');
					} else if (Array.isArray(value)) {
						value.forEach((item, i) => {
							expect(String(item).trim(), `${name}.${path}[${i}]`).not.toBe('');
						});
					} else if (value && typeof value === 'object') {
						walk(value as Node, path);
					}
				}
			};
			walk(catalog as unknown as Node);
		}
	});

	it('renders every interpolated string with its arguments', () => {
		const sample = (fn: (...args: never[]) => unknown) =>
			(fn as (...args: number[]) => unknown)(...Array.from({ length: fn.length }, () => 12));
		const walk = (node: Node, name: string, prefix = '') => {
			for (const [key, value] of Object.entries(node)) {
				const path = prefix ? `${prefix}.${key}` : key;
				if (typeof value === 'function') {
					const rendered = sample(value as (...args: never[]) => unknown);
					expect(typeof rendered, `${name}.${path}`).toBe('string');
					expect(String(rendered).trim(), `${name}.${path}`).not.toBe('');
					// A placeholder that never got substituted is a bug.
					expect(String(rendered), `${name}.${path}`).not.toMatch(/\{|\}|undefined|NaN/);
				} else if (value && typeof value === 'object' && !Array.isArray(value)) {
					walk(value as Node, name, path);
				}
			}
		};
		walk(da as unknown as Node, 'da');
		walk(en as unknown as Node, 'en');
	});

	it('translates rather than copies the prose', () => {
		// A handful of anchors: if these ever match, something was left untranslated.
		expect(en.site.tagline).not.toBe(da.site.tagline);
		expect(en.footer.disclaimer).not.toBe(da.footer.disclaimer);
		expect(en.panel.headlineNoRain).not.toBe(da.panel.headlineNoRain);
	});
});
