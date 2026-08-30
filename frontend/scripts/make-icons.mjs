#!/usr/bin/env node
/**
 * make-icons.mjs — generate the PWA icons from one raindrop path.
 *
 * The shape lives here as cubic Béziers, and both the SVG and the PNGs are
 * derived from it, so the home-screen icon and the header logo can never
 * drift apart. Node has no SVG rasteriser, so the PNG path is done by hand:
 * flatten the curves, scanline-fill with 4×4 supersampling, deflate, write
 * the chunks. That is a page of code and zero dependencies.
 *
 * Run: node scripts/make-icons.mjs   (outputs into static/icons/)
 */
import { deflateSync } from 'node:zlib';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'static', 'icons');

// Palette (matches --accent / --surface in src/app.css).
const BLUE = [11, 107, 203];
const WHITE = [255, 255, 255];

/**
 * The drop, in a 24×24 box: tip at the top, round belly at the bottom.
 * [x0,y0, c1x,c1y, c2x,c2y, x1,y1] per cubic segment.
 */
const DROP = [
	[12, 2.4, 14.6, 6.2, 19.6, 11.2, 19.6, 14.6],
	[19.6, 14.6, 19.6, 18.8, 16.2, 21.6, 12, 21.6],
	[12, 21.6, 7.8, 21.6, 4.4, 18.8, 4.4, 14.6],
	[4.4, 14.6, 4.4, 11.2, 9.4, 6.2, 12, 2.4]
];

const SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
  <path fill="#0b6bcb" d="M12 2.4C14.6 6.2 19.6 11.2 19.6 14.6C19.6 18.8 16.2 21.6 12 21.6C7.8 21.6 4.4 18.8 4.4 14.6C4.4 11.2 9.4 6.2 12 2.4Z"/>
</svg>
`;

/** Flatten the cubics into a polygon in 24×24 space. */
function dropPolygon(steps = 48) {
	const pts = [];
	for (const [x0, y0, c1x, c1y, c2x, c2y, x1, y1] of DROP) {
		for (let i = 0; i < steps; i++) {
			const t = i / steps;
			const u = 1 - t;
			pts.push([
				u * u * u * x0 + 3 * u * u * t * c1x + 3 * u * t * t * c2x + t * t * t * x1,
				u * u * u * y0 + 3 * u * u * t * c1y + 3 * u * t * t * c2y + t * t * t * y1
			]);
		}
	}
	return pts;
}

const inside = (poly, x, y) => {
	let hit = false;
	for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
		const [xi, yi] = poly[i];
		const [xj, yj] = poly[j];
		if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
	}
	return hit;
};

/**
 * Render one icon: a (optionally rounded) coloured field with the white drop
 * on top, 4×4 supersampled.
 */
function renderIcon(size, { dropScale = 0.62, radius = 0.22 } = {}) {
	const poly = dropPolygon();
	const rgba = new Uint8Array(size * size * 4);
	const ss = 4;
	const cornerR = radius * size;
	// Drop placement: centred, scaled to `dropScale` of the icon.
	const scale = (size * dropScale) / 24;
	const offsetX = (size - 24 * scale) / 2;
	const offsetY = (size - 24 * scale) / 2;

	for (let py = 0; py < size; py++) {
		for (let px = 0; px < size; px++) {
			let bg = 0;
			let fg = 0;
			for (let sy = 0; sy < ss; sy++) {
				for (let sx = 0; sx < ss; sx++) {
					const x = px + (sx + 0.5) / ss;
					const y = py + (sy + 0.5) / ss;
					if (inRoundedRect(x, y, size, cornerR)) bg++;
					if (inside(poly, (x - offsetX) / scale, (y - offsetY) / scale)) fg++;
				}
			}
			const total = ss * ss;
			const bgA = bg / total;
			const fgA = fg / total;
			const i = (py * size + px) * 4;
			// White drop over blue field, both anti-aliased.
			const a = Math.max(bgA, fgA);
			if (a === 0) continue;
			const mix = fgA / Math.max(a, 1e-6);
			for (let c = 0; c < 3; c++) {
				rgba[i + c] = Math.round(BLUE[c] * (1 - mix) + WHITE[c] * mix);
			}
			rgba[i + 3] = Math.round(a * 255);
		}
	}
	return rgba;
}

function inRoundedRect(x, y, size, r) {
	if (r <= 0) return true;
	const cx = Math.min(Math.max(x, r), size - r);
	const cy = Math.min(Math.max(y, r), size - r);
	return (x - cx) ** 2 + (y - cy) ** 2 <= r * r;
}

// --- minimal PNG writer (RGBA8, no interlace) ------------------------------

const CRC_TABLE = (() => {
	const table = new Uint32Array(256);
	for (let n = 0; n < 256; n++) {
		let c = n;
		for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
		table[n] = c >>> 0;
	}
	return table;
})();

function crc32(buf) {
	let c = 0xffffffff;
	for (const byte of buf) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8);
	return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, body) {
	const out = Buffer.alloc(body.length + 12);
	out.writeUInt32BE(body.length, 0);
	out.write(type, 4, 'ascii');
	Buffer.from(body).copy(out, 8);
	out.writeUInt32BE(crc32(out.subarray(4, 8 + body.length)), 8 + body.length);
	return out;
}

function encodePng(rgba, size) {
	const raw = Buffer.alloc((size * 4 + 1) * size);
	for (let y = 0; y < size; y++) {
		raw[y * (size * 4 + 1)] = 0; // filter: none
		Buffer.from(rgba.buffer, y * size * 4, size * 4).copy(raw, y * (size * 4 + 1) + 1);
	}
	const ihdr = Buffer.alloc(13);
	ihdr.writeUInt32BE(size, 0);
	ihdr.writeUInt32BE(size, 4);
	ihdr[8] = 8; // bit depth
	ihdr[9] = 6; // RGBA
	return Buffer.concat([
		Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
		chunk('IHDR', ihdr),
		chunk('IDAT', deflateSync(raw, { level: 9 })),
		chunk('IEND', Buffer.alloc(0))
	]);
}

await mkdir(OUT, { recursive: true });
await writeFile(path.join(OUT, 'icon.svg'), SVG);
const targets = [
	['icon-192.png', 192, { dropScale: 0.62, radius: 0.22 }],
	['icon-512.png', 512, { dropScale: 0.62, radius: 0.22 }],
	// Maskable: full bleed, content inside the safe circle.
	['icon-maskable-512.png', 512, { dropScale: 0.48, radius: 0 }]
];
for (const [name, size, options] of targets) {
	const png = encodePng(renderIcon(size, options), size);
	await writeFile(path.join(OUT, name), png);
	console.log(`[icons] ${name} — ${size}×${size}, ${(png.length / 1024).toFixed(1)} kB`);
}
console.log('[icons] icon.svg');
