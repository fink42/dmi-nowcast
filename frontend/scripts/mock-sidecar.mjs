#!/usr/bin/env node
/**
 * mock-sidecar.mjs — a fake `/nowcast/*` + `/forecast` origin for frontend work.
 *
 * The real sidecar needs three consecutive DMI radar frames before it serves
 * anything, which makes it a poor companion for UI work. This serves the same
 * contract with synthetic weather: two rain blobs (over Copenhagen and
 * Aarhus) drifting east-north-east with lead time, quantised exactly the way
 * `national_artifacts.py` quantises, and a manifest with the real grid
 * geometry. Because the blobs sit at known coordinates, it also makes
 * georeferencing mistakes obvious — a blob that is not over Copenhagen means
 * the overlay is misplaced.
 *
 * Usage:
 *   node scripts/mock-sidecar.mjs --port 8099
 *   VITE_SIDECAR_URL=http://localhost:8099 npm run dev
 */
import { createServer } from 'node:http';
import { deflateSync } from 'node:zlib';

const args = process.argv.slice(2);
const portIndex = args.indexOf('--port');
const PORT = portIndex === -1 ? 8099 : Number(args[portIndex + 1]);

const PROJ4 = '+proj=stere +lat_0=56 +lon_0=10.5666 +lat_ts=56 +ellps=WGS84 +units=m +no_defs';
const LEADS = [5, 10, 15, 20, 25, 30, 45, 60];
const NATIVE = { cols: 1984, rows: 1728, scale: 500, xUl: -496000, yUl: 432000 };
const FACTOR = 4;
const PRODUCT = {
	cols: NATIVE.cols / FACTOR,
	rows: NATIVE.rows / FACTOR,
	scale: NATIVE.scale * FACTOR
};

// Blob centres in native grid pixels (Copenhagen, Aarhus), from the same
// projection maths the app uses.
const BLOBS = [
	{ col: 1243.8, row: 932.5, radius: 55 },
	{ col: 946.9, row: 827.6, radius: 40 }
];
/** Drift per minute of lead, in native pixels (≈ 45 km/h to the ENE). */
const DRIFT = { col: 1.5, row: -0.6 };

const NODATA = 255;
const MAX_LEVEL = 254;
const SPECS = { p_rain: [0, 1], eta: [0, 120], intensity: [0, 100] };
const scaleOf = ([lo, hi]) => (hi - lo) / MAX_LEVEL;

const stamp = () => {
	const d = new Date(Math.floor(Date.now() / 300_000) * 300_000);
	const p = (n) => String(n).padStart(2, '0');
	return {
		text: `${d.getUTCFullYear()}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}${p(d.getUTCHours())}${p(d.getUTCMinutes())}`,
		iso: d.toISOString()
	};
};

/** Rain rate (mm/h) at a native grid pixel for a given lead. */
function rainAt(col, row, leadMin) {
	let value = 0;
	for (const blob of BLOBS) {
		const dc = col - (blob.col + DRIFT.col * leadMin);
		const dr = row - (blob.row + DRIFT.row * leadMin);
		const d = Math.hypot(dc, dr) / blob.radius;
		if (d < 1) value = Math.max(value, 12 * (1 - d) ** 1.5);
	}
	return value;
}

const quantise = (value, spec) =>
	value === null || !Number.isFinite(value)
		? NODATA
		: Math.round((Math.min(Math.max(value, spec[0]), spec[1]) - spec[0]) / scaleOf(spec));

// --- PNG ---------------------------------------------------------------------

const CRC_TABLE = (() => {
	const table = new Uint32Array(256);
	for (let n = 0; n < 256; n++) {
		let c = n;
		for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
		table[n] = c >>> 0;
	}
	return table;
})();
const crc32 = (buf) => {
	let c = 0xffffffff;
	for (const byte of buf) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8);
	return (c ^ 0xffffffff) >>> 0;
};
function chunk(type, body) {
	const out = Buffer.alloc(body.length + 12);
	out.writeUInt32BE(body.length, 0);
	out.write(type, 4, 'ascii');
	Buffer.from(body).copy(out, 8);
	out.writeUInt32BE(crc32(out.subarray(4, 8 + body.length)), 8 + body.length);
	return out;
}
function encodePng(pixels, width, height, channels) {
	const stride = width * channels;
	const raw = Buffer.alloc((stride + 1) * height);
	for (let y = 0; y < height; y++) {
		raw[y * (stride + 1)] = 0;
		Buffer.from(pixels.buffer, y * stride, stride).copy(raw, y * (stride + 1) + 1);
	}
	const ihdr = Buffer.alloc(13);
	ihdr.writeUInt32BE(width, 0);
	ihdr.writeUInt32BE(height, 4);
	ihdr[8] = 8;
	ihdr[9] = channels === 1 ? 0 : 6;
	return Buffer.concat([
		Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
		chunk('IHDR', ihdr),
		chunk('IDAT', deflateSync(raw, { level: 6 })),
		chunk('IEND', Buffer.alloc(0))
	]);
}

/** render.py's colours, roughly: blue → green → yellow → red with alpha. */
function colourFor(mmH) {
	if (mmH <= 0.1) return [0, 0, 0, 0];
	const stops = [
		[0.1, [90, 160, 255]],
		[1, [40, 200, 160]],
		[3, [230, 220, 60]],
		[8, [240, 140, 40]],
		[20, [220, 50, 60]]
	];
	let colour = stops[stops.length - 1][1];
	for (let i = 0; i < stops.length - 1; i++) {
		if (mmH < stops[i + 1][0]) {
			const f = (mmH - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
			colour = stops[i][1].map((c, k) => Math.round(c + f * (stops[i + 1][1][k] - c)));
			break;
		}
	}
	return [...colour, Math.round(Math.min(1, 0.35 + mmH / 8) * 255)];
}

const cache = new Map();
const cached = (key, make) => {
	if (!cache.has(key)) cache.set(key, make());
	return cache.get(key);
};

function overlayPng(leadMin) {
	return cached(`overlay:${leadMin}`, () => {
		const { cols, rows } = NATIVE;
		const px = new Uint8Array(cols * rows * 4);
		for (let row = 0; row < rows; row++) {
			for (let col = 0; col < cols; col++) {
				const mm = rainAt(col, row, leadMin);
				if (mm <= 0.1) continue;
				const [r, g, b, a] = colourFor(mm);
				const i = (row * cols + col) * 4;
				px[i] = r;
				px[i + 1] = g;
				px[i + 2] = b;
				px[i + 3] = a;
			}
		}
		return encodePng(px, cols, rows, 4);
	});
}

function productPng(product, leadMin) {
	return cached(`${product}:${leadMin}`, () => {
		const { cols, rows } = PRODUCT;
		const levels = new Uint8Array(cols * rows);
		for (let row = 0; row < rows; row++) {
			for (let col = 0; col < cols; col++) {
				const nCol = col * FACTOR;
				const nRow = row * FACTOR;
				let value = null;
				if (product === 'p_rain') {
					const mm = rainAt(nCol, nRow, leadMin);
					value = Math.min(1, mm / 4);
				} else if (product === 'eta') {
					// First lead at which rain reaches the pixel.
					value = null;
					for (const lead of [0, ...LEADS]) {
						if (rainAt(nCol, nRow, lead) > 0.5) {
							value = lead;
							break;
						}
					}
				} else {
					const arrival = LEADS.find((lead) => rainAt(nCol, nRow, lead) > 0.5);
					value = arrival === undefined ? null : rainAt(nCol, nRow, arrival);
				}
				levels[row * cols + col] = quantise(value, SPECS[product]);
			}
		}
		return encodePng(levels, cols, rows, 1);
	});
}

function manifest() {
	const s = stamp();
	const gridBlock = (scale, shape) => ({
		proj4: PROJ4,
		x_ul_m: NATIVE.xUl,
		y_ul_m: NATIVE.yUl,
		pixel_scale_x_m: scale,
		pixel_scale_y_m: scale,
		shape,
		downsample_factor: scale / NATIVE.scale
	});
	const gridEntry = (filename, product, lead, spec, shape, units) => ({
		filename,
		product,
		lead_min: lead,
		encoding: 'grayscale8',
		scale: scaleOf(spec),
		offset: spec[0],
		nodata: NODATA,
		units,
		shape
	});
	const productShape = [PRODUCT.rows, PRODUCT.cols];
	const overlayShape = [NATIVE.rows, NATIVE.cols];
	return {
		schema_version: 1,
		cycle: s.text,
		radar_ts_utc: s.iso,
		generated_at_utc: new Date().toISOString(),
		threshold_mm_h: 0.1,
		timestep_min: 5,
		frame_age_min: 2.1,
		n_members: 24,
		leads_min: LEADS,
		grid: gridBlock(PRODUCT.scale, productShape),
		overlay_grid: gridBlock(NATIVE.scale, overlayShape),
		calibration: {
			fitted_at: '2026-08-01T03:00:00+00:00',
			calibrated_leads: LEADS,
			n_samples: 3_800_000,
			brier_before: 0.191,
			brier_after: 0.163
		},
		artifacts: [
			...LEADS.map((lead) =>
				gridEntry(
					`p_rain_${lead}min_${s.text}.png`,
					'p_rain',
					lead,
					SPECS.p_rain,
					productShape,
					'probability'
				)
			),
			gridEntry(`eta_${s.text}.png`, 'eta', null, SPECS.eta, productShape, 'min'),
			gridEntry(
				`intensity_${s.text}.png`,
				'intensity',
				null,
				SPECS.intensity,
				productShape,
				'mm/h'
			),
			{
				filename: `overlay_now_${s.text}.png`,
				product: 'overlay',
				lead_min: 0,
				encoding: 'rgba8',
				shape: overlayShape
			},
			...LEADS.map((lead) => ({
				filename: `overlay_${lead}min_${s.text}.png`,
				product: 'overlay',
				lead_min: lead,
				encoding: 'rgba8',
				shape: overlayShape
			}))
		]
	};
}

createServer((req, res) => {
	const url = new URL(req.url, `http://localhost:${PORT}`);
	const send = (status, body, type, extra = {}) => {
		res.writeHead(status, { 'content-type': type, 'access-control-allow-origin': '*', ...extra });
		res.end(body);
	};
	if (url.pathname === '/nowcast/manifest.json') {
		return send(200, JSON.stringify(manifest(), null, 2), 'application/json', {
			'cache-control': 'public, max-age=30'
		});
	}
	const artifact = url.pathname.match(/^\/nowcast\/(\w+?)_?(?:(\d+)min|now)?_(\d{12})\.png$/);
	if (artifact) {
		const [, product, leadText] = artifact;
		const lead = leadText ? Number(leadText) : 0;
		const png =
			product === 'overlay' ? overlayPng(lead) : productPng(product, product === 'p_rain' ? lead : 0);
		return send(200, png, 'image/png', { 'cache-control': 'public, max-age=300, immutable' });
	}
	if (url.pathname === '/forecast') {
		const lat = Number(url.searchParams.get('lat'));
		const lon = Number(url.searchParams.get('lon'));
		const s = stamp();
		return send(
			200,
			JSON.stringify({
				lat,
				lon,
				radar_ts_utc: s.iso,
				n_members: 24,
				calibrated: true,
				calibration_fitted_at: '2026-08-01T03:00:00+00:00',
				per_lead: LEADS.map((lead) => ({ lead_min: lead, p_rain: 0.5 })),
				eta_min: 18,
				intensity_mm_h: 2.4,
				confidence: 0.72
			}),
			'application/json'
		);
	}
	if (url.pathname === '/healthz') return send(200, '{"status":"ok"}', 'application/json');
	send(404, JSON.stringify({ detail: 'not found' }), 'application/json');
}).listen(PORT, () => {
	console.log(`[mock-sidecar] http://localhost:${PORT} — synthetic nowcast data`);
	console.log('[mock-sidecar] run the app with: VITE_SIDECAR_URL=http://localhost:%d npm run dev', PORT);
});
