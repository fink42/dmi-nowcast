/**
 * A minimal decoder for the quantised **grayscale-8** product PNGs.
 *
 * Why not `createImageBitmap` + a canvas? Because the canvas path is the one
 * place a browser is allowed to lie to us: colour management, premultiplied
 * alpha and 8-bit rounding can all shift a pixel by a level, and a level here
 * is a physical value (0.4 % of probability, 0.5 min of ETA, 0.4 mm/h). This
 * decoder reads the exact bytes the sidecar wrote — and, being plain
 * TypeScript over `DecompressionStream`, it runs identically in the browser
 * and in the test runner, so the sampling math can be tested for real.
 *
 * Only what the sidecar actually writes is supported: 8-bit greyscale,
 * non-interlaced (PIL `Image.fromarray(levels, "L")`). Anything else throws,
 * and the caller falls back to the server's `/forecast` endpoint.
 */

export interface Gray8Image {
	width: number;
	height: number;
	/** Row-major levels, `width * height` bytes. 255 means "no value". */
	levels: Uint8Array;
}

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

export async function decodeGray8Png(bytes: Uint8Array): Promise<Gray8Image> {
	for (let i = 0; i < SIGNATURE.length; i++) {
		if (bytes[i] !== SIGNATURE[i]) throw new Error('not a PNG');
	}
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

	let width = 0;
	let height = 0;
	const idat: Uint8Array[] = [];
	let offset = 8;
	while (offset + 8 <= bytes.length) {
		const length = view.getUint32(offset);
		const type = String.fromCharCode(
			bytes[offset + 4],
			bytes[offset + 5],
			bytes[offset + 6],
			bytes[offset + 7]
		);
		const body = bytes.subarray(offset + 8, offset + 8 + length);
		if (type === 'IHDR') {
			width = view.getUint32(offset + 8);
			height = view.getUint32(offset + 12);
			const bitDepth = body[8];
			const colorType = body[9];
			const interlace = body[12];
			if (bitDepth !== 8 || colorType !== 0 || interlace !== 0) {
				throw new Error(
					`unsupported PNG: bitDepth=${bitDepth} colorType=${colorType} interlace=${interlace}`
				);
			}
		} else if (type === 'IDAT') {
			idat.push(body);
		} else if (type === 'IEND') {
			break;
		}
		offset += 12 + length; // length + type + body + CRC
	}
	if (!width || !height) throw new Error('PNG without IHDR');
	if (idat.length === 0) throw new Error('PNG without IDAT');

	const raw = await inflate(concat(idat));
	return { width, height, levels: unfilter(raw, width, height) };
}

function concat(chunks: Uint8Array[]): Uint8Array {
	const total = chunks.reduce((n, c) => n + c.length, 0);
	const out = new Uint8Array(total);
	let at = 0;
	for (const c of chunks) {
		out.set(c, at);
		at += c.length;
	}
	return out;
}

/** zlib stream → bytes. `DecompressionStream('deflate')` is the zlib wrapper. */
async function inflate(data: Uint8Array): Promise<Uint8Array> {
	const stream = new Blob([data as BlobPart]).stream().pipeThrough(new DecompressionStream('deflate'));
	const parts: Uint8Array[] = [];
	const reader = stream.getReader();
	for (;;) {
		const { done, value } = await reader.read();
		if (done) break;
		parts.push(value as Uint8Array);
	}
	return concat(parts);
}

/**
 * Reverse the PNG per-scanline filters. Bytes-per-pixel is 1 (grayscale-8),
 * which collapses the usual `a`/`c` lookups to the previous byte.
 */
function unfilter(raw: Uint8Array, width: number, height: number): Uint8Array {
	const out = new Uint8Array(width * height);
	const stride = width + 1; // filter byte per scanline
	if (raw.length < stride * height) throw new Error('truncated PNG data');
	for (let y = 0; y < height; y++) {
		const filter = raw[y * stride];
		const line = y * stride + 1;
		const dst = y * width;
		const up = dst - width;
		for (let x = 0; x < width; x++) {
			const value = raw[line + x];
			const a = x > 0 ? out[dst + x - 1] : 0;
			const b = y > 0 ? out[up + x] : 0;
			const c = x > 0 && y > 0 ? out[up + x - 1] : 0;
			let recon: number;
			switch (filter) {
				case 0:
					recon = value;
					break;
				case 1:
					recon = value + a;
					break;
				case 2:
					recon = value + b;
					break;
				case 3:
					recon = value + ((a + b) >> 1);
					break;
				case 4:
					recon = value + paeth(a, b, c);
					break;
				default:
					throw new Error(`unknown PNG filter ${filter} on row ${y}`);
			}
			out[dst + x] = recon & 0xff;
		}
	}
	return out;
}

function paeth(a: number, b: number, c: number): number {
	const p = a + b - c;
	const pa = Math.abs(p - a);
	const pb = Math.abs(p - b);
	const pc = Math.abs(p - c);
	if (pa <= pb && pa <= pc) return a;
	return pb <= pc ? b : c;
}
