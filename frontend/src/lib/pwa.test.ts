/**
 * Build smoke for the installable-app bits: the web manifest has to parse,
 * name the app, and point at icons that actually exist on disk. These are
 * the files a phone reads before it will offer "add to home screen", and a
 * typo in them fails silently in production.
 */
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const STATIC = path.resolve(import.meta.dirname, '..', '..', 'static');

interface WebManifest {
	name: string;
	short_name: string;
	start_url: string;
	display: string;
	theme_color: string;
	background_color: string;
	icons: { src: string; sizes: string; type: string; purpose?: string }[];
}

const manifest = JSON.parse(
	readFileSync(path.join(STATIC, 'manifest.webmanifest'), 'utf8')
) as WebManifest;

describe('web app manifest', () => {
	it('carries the fields a browser needs to offer installation', () => {
		expect(manifest.name).toBeTruthy();
		expect(manifest.short_name.length).toBeLessThanOrEqual(12);
		expect(manifest.start_url).toBe('/');
		expect(manifest.display).toBe('standalone');
		expect(manifest.theme_color).toMatch(/^#[0-9a-f]{6}$/i);
		expect(manifest.background_color).toMatch(/^#[0-9a-f]{6}$/i);
	});

	it('references icons that exist, including a maskable one', () => {
		expect(manifest.icons.length).toBeGreaterThanOrEqual(3);
		for (const icon of manifest.icons) {
			expect(existsSync(path.join(STATIC, icon.src)), icon.src).toBe(true);
		}
		const sizes = manifest.icons.map((i) => i.sizes);
		expect(sizes).toContain('192x192');
		expect(sizes).toContain('512x512');
		expect(manifest.icons.some((i) => i.purpose === 'maskable')).toBe(true);
	});

	it('ships PNG icons that really are PNGs of the stated size', () => {
		for (const icon of manifest.icons.filter((i) => i.type === 'image/png')) {
			const bytes = readFileSync(path.join(STATIC, icon.src));
			expect([...bytes.subarray(0, 8)]).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
			const width = bytes.readUInt32BE(16);
			const height = bytes.readUInt32BE(20);
			expect(`${width}x${height}`, icon.src).toBe(icon.sizes);
		}
	});
});
