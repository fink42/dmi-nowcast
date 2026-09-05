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
 * Schema v2: overlays carry `kind` + `valid_ts_utc`, the manifest references
 * three prior cycles as observation history (negative leads, blobs rewound),
 * and the two cell-motion grids are served with nodata everywhere the echoes
 * are too far away for an estimate — so the timeline and the motion arrow can
 * both be developed against something that behaves like the real thing.
 *
 * It also serves `/api/push/*` with an in-memory subscription table and a
 * syntactically valid (but meaningless) VAPID key, so the notification UI can
 * be driven end to end without a push service. No message is ever sent — the
 * browser subscription is real, the delivery half is not.
 *
 * It also serves `/nowcast/quality.json` straight from the committed
 * fixture (src/lib/quality/fixture.json), so the quality page can be
 * developed without waiting for the nightly verification job.
 *
 * Usage:
 *   node scripts/mock-sidecar.mjs --port 8099
 *   VITE_SIDECAR_URL=http://localhost:8099 npm run dev
 */
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { deflateSync } from 'node:zlib';

const args = process.argv.slice(2);
const portIndex = args.indexOf('--port');
const PORT = portIndex === -1 ? 8099 : Number(args[portIndex + 1]);

const PROJ4 = '+proj=stere +lat_0=56 +lon_0=10.5666 +lat_ts=56 +ellps=WGS84 +units=m +no_defs';
/**
 * Two lead lists, because the sidecar has two: `forecast.leads_min` renders
 * the forecast overlay frames the timeline scrubs through (kept on the
 * radar's own 10 min cadence, so forecast ticks line up with history ticks),
 * while `forecast.national.leads_min` drives the calibrated probability
 * grids — the leads the isotonic curves were actually fitted for.
 */
const OVERLAY_LEADS = [10, 20, 30, 40, 50, 60];
/**
 * The leads the deterministic advected field is served on: the overlay leads
 * plus lead 0, which is the radar field advected forward by its own age — the
 * forecast for the instant the cycle was generated. The panel's headline is
 * read from this series, so it has to cover the same instants the loop draws.
 */
const FORECAST_LEADS = [0, ...OVERLAY_LEADS];
const PROB_LEADS = [10, 20, 30, 45, 60];
/**
 * Lead times scanned when asking "when does rain first reach this pixel?"
 * for the ETA and intensity grids. Neither published list: this is a search
 * grid, and a dense 5 min scan keeps the ETA field's resolution independent
 * of how coarse the published frames happen to be.
 */
const ARRIVAL_SCAN = Array.from({ length: 13 }, (_, i) => i * 5);
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
/** The same drift as a velocity, which is what the motion grids serve. */
const MOTION_KMH = {
	east: (DRIFT.col * NATIVE.scale * 60) / 1000,
	north: (-DRIFT.row * NATIVE.scale * 60) / 1000
};
/** Prior cycles referenced as observation history, at the 10 min cadence. */
const HISTORY_MIN = [-30, -20, -10];
/** How old the radar frame already is when the cycle runs (minutes). */
const FRAME_AGE_MIN = 2.1;

const NODATA = 255;
const MAX_LEVEL = 254;
const SPECS = {
	p_rain: [0, 1],
	eta: [0, 120],
	intensity: [0, 100],
	// The observation grid is quantised exactly like intensity — same units,
	// same range — because it is the same physical quantity, measured rather
	// than forecast.
	observed_mm_h: [0, 100],
	// The advected field is the same physical quantity as the observation,
	// forecast rather than measured, and is quantised identically.
	forecast_mm_h: [0, 100],
	motion_east_kmh: [-120, 120],
	motion_north_kmh: [-120, 120]
};
const scaleOf = ([lo, hi]) => (hi - lo) / MAX_LEVEL;

const p2 = (n) => String(n).padStart(2, '0');
const stampText = (d) =>
	`${d.getUTCFullYear()}${p2(d.getUTCMonth() + 1)}${p2(d.getUTCDate())}${p2(d.getUTCHours())}${p2(d.getUTCMinutes())}`;

const stamp = (offsetMin = 0) => {
	const d = new Date(Math.floor(Date.now() / 300_000) * 300_000 + offsetMin * 60_000);
	return { text: stampText(d), iso: d.toISOString() };
};

/** `YYYYMMDDHHMM` → epoch ms, so a history frame knows its own lead. */
const stampMs = (text) =>
	Date.UTC(
		Number(text.slice(0, 4)),
		Number(text.slice(4, 6)) - 1,
		Number(text.slice(6, 8)),
		Number(text.slice(8, 10)),
		Number(text.slice(10, 12))
	);

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

/**
 * The rain at one product pixel `leadMin` minutes after the radar frame: the
 * 90th percentile of the FACTOR x FACTOR native block behind it. Percentile
 * rather than max because that is what the sidecar reduces with — the
 * composite is column-max reflectivity, so a single hot virga pixel would
 * otherwise decide whether it is raining — and linear interpolation between
 * order statistics because that is numpy's default and the sidecar is numpy.
 *
 * At lead 0 this is the observation; at `FRAME_AGE_MIN + lead` it is the
 * advected forecast for the overlay frame of that lead. One function, because
 * the two must agree wherever they describe the same instant.
 */
function blockP90(col, row, leadMin) {
	const block = [];
	for (let dr = 0; dr < FACTOR; dr++) {
		for (let dc = 0; dc < FACTOR; dc++) {
			block.push(rainAt(col * FACTOR + dc, row * FACTOR + dr, leadMin));
		}
	}
	block.sort((a, b) => a - b);
	const pos = 0.9 * (block.length - 1);
	const lo = Math.floor(pos);
	const hi = Math.min(lo + 1, block.length - 1);
	return block[lo] + (pos - lo) * (block[hi] - block[lo]);
}

const quantise = (value, spec) =>
	value === null || !Number.isFinite(value)
		? NODATA
		: Math.round((Math.min(Math.max(value, spec[0]), spec[1]) - spec[0]) / scaleOf(spec));

// --- push --------------------------------------------------------------------

/**
 * A structurally valid application server key: 65 bytes, uncompressed-point
 * prefix 0x04, which base64url-encodes to something starting `BA…`. The bytes
 * behind it are nonsense — nothing here signs or sends anything — but the
 * browser rejects a key of the wrong shape before it ever gets to the server.
 */
const VAPID_PUBLIC_KEY = (() => {
	const bytes = Buffer.alloc(65);
	bytes[0] = 0x04;
	bytes[1] = 0x00;
	for (let i = 2; i < 65; i++) bytes[i] = (i * 37 + 11) % 256;
	return bytes.toString('base64url');
})();

const PUSH_CONFIG = {
	enabled: true,
	vapid_public_key: VAPID_PUBLIC_KEY,
	threshold_options_pct: [40, 60, 80],
	lead_options_min: [20, 30, 45, 60],
	defaults: {
		threshold_pct: 60,
		lead_min: 30,
		quiet_hours: { enabled: false, start: '22:00', end: '07:00' }
	},
	capacity_reached: false
};

/** endpoint → the stored row, exactly as the real service would upsert it. */
const subscriptions = new Map();

/** Roughly the composite's footprint — enough to exercise the 400 path. */
const inCoverage = (lat, lon) => lat >= 54.0 && lat <= 58.2 && lon >= 7.0 && lon <= 15.8;

const readJson = (req) =>
	new Promise((resolve) => {
		let text = '';
		req.on('data', (chunk) => (text += chunk));
		req.on('end', () => {
			try {
				resolve(JSON.parse(text));
			} catch {
				resolve(null);
			}
		});
		req.on('error', () => resolve(null));
	});

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
				if (product === 'motion_east_kmh' || product === 'motion_north_kmh') {
					// One uniform drift, served across the whole coverage: the
					// blobs exist, so every pixel gets the nearest cells' motion
					// (issue #6). Nodata only when there is no echo at all.
					value = product === 'motion_east_kmh' ? MOTION_KMH.east : MOTION_KMH.north;
				} else if (product === 'observed_mm_h') {
					// A measurement, so dry is 0.0 and not nodata: "no rain here"
					// is something the radar can say, unlike "no arrival".
					value = blockP90(col, row, 0);
				} else if (product === 'forecast_mm_h') {
					// The field the overlay frame of this lead draws, on the
					// product grid — frame-age corrected, so lead 0 is already
					// FRAME_AGE_MIN ahead of the radar image.
					value = blockP90(col, row, FRAME_AGE_MIN + leadMin);
				} else if (product === 'p_rain') {
					const mm = rainAt(nCol, nRow, leadMin);
					value = Math.min(1, mm / 4);
				} else if (product === 'eta') {
					// First lead at which rain reaches the pixel.
					value = null;
					for (const lead of ARRIVAL_SCAN) {
						if (rainAt(nCol, nRow, lead) > 0.5) {
							value = lead;
							break;
						}
					}
				} else {
					const arrival = ARRIVAL_SCAN.find((lead) => rainAt(nCol, nRow, lead) > 0.5);
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
	/** When a frame is valid: measurements at their own time, forecasts frame-age corrected. */
	const validAt = (lead, kind) =>
		new Date(
			Date.parse(s.iso) + (kind === 'forecast' ? FRAME_AGE_MIN + lead : lead) * 60_000
		).toISOString();
	/**
	 * Overlay entry, v2: every frame says what it is and when it is valid.
	 * The kind is explicit rather than inferred from the sign of the lead,
	 * because lead 0 carries two frames — the radar image, and the same field
	 * advected forward by the frame age.
	 */
	const overlayEntry = (filename, lead, kind = lead > 0 ? 'forecast' : 'observation') => ({
		filename,
		product: 'overlay',
		lead_min: lead,
		kind,
		// Forecast validity is frame-age corrected the way the sidecar does it:
		// radar_ts + frame_age + lead. An observation is valid when it was taken.
		valid_ts_utc: validAt(lead, kind),
		encoding: 'rgba8',
		shape: overlayShape
	});
	return {
		schema_version: 2,
		cycle: s.text,
		radar_ts_utc: s.iso,
		generated_at_utc: new Date().toISOString(),
		threshold_mm_h: 0.1,
		timestep_min: 5,
		frame_age_min: FRAME_AGE_MIN,
		// From radar-frame time; honest horizon from now is this minus
		// frame_age_min.
		ensemble_horizon_min: 90,
		n_members: 24,
		leads_min: PROB_LEADS,
		grid: gridBlock(PRODUCT.scale, productShape),
		overlay_grid: gridBlock(NATIVE.scale, overlayShape),
		motion: {
			grid: 'product',
			support_radius_km: null,
			fill: 'nearest-cells-v1',
			fill_scales_km: [25, 50, 100],
			max_abs_kmh: 120,
			convention:
				'motion_east_kmh / motion_north_kmh are the cell motion in km/h on ' +
				'the product grid, east- and north-positive. nodata (255) outside ' +
				'radar coverage, and everywhere when the composite has no echo at ' +
				'all. On the echo the vector is the measured optical flow; off it, ' +
				'it is the motion of the nearest cells — a rain-weighted average ' +
				'over a search area that starts small and widens (fill_scales_km) ' +
				'until it reaches echo.'
		},
		calibration: {
			fitted_at: '2026-08-01T03:00:00+00:00',
			calibrated_leads: PROB_LEADS,
			n_samples: 3_800_000,
			brier_before: 0.191,
			brier_after: 0.163
		},
		artifacts: [
			...PROB_LEADS.map((lead) =>
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
			// Observed rain: lead 0, because it is now — not a forecast lead.
			gridEntry(
				`observed_mm_h_${s.text}.png`,
				'observed_mm_h',
				0,
				SPECS.observed_mm_h,
				productShape,
				'mm/h'
			),
			gridEntry(
				`motion_east_kmh_${s.text}.png`,
				'motion_east_kmh',
				null,
				SPECS.motion_east_kmh,
				productShape,
				'km/h'
			),
			gridEntry(
				`motion_north_kmh_${s.text}.png`,
				'motion_north_kmh',
				null,
				SPECS.motion_north_kmh,
				productShape,
				'km/h'
			),
			// Observation history: prior cycles' own "now" overlays, oldest
			// first, at negative leads. Set HISTORY_MIN to [] to see the
			// cold-start case the timeline also has to handle.
			...HISTORY_MIN.map((lead) =>
				overlayEntry(`overlay_now_${stamp(lead).text}.png`, lead)
			),
			// The advected field, one grid per overlay lead — what the headline
			// is read from, sampled at the instant the loop's clock marker sits
			// on. Lead 0 is the radar field advected forward by its own age.
			...FORECAST_LEADS.map((lead) => ({
				...gridEntry(
					`forecast_mm_h_${lead}min_${s.text}.png`,
					'forecast_mm_h',
					lead,
					SPECS.forecast_mm_h,
					productShape,
					'mm/h'
				),
				kind: 'forecast',
				valid_ts_utc: validAt(lead, 'forecast')
			})),
			overlayEntry(`overlay_now_${s.text}.png`, 0),
			// The second lead-0 overlay: the same instant the lead-0 field
			// describes, and listed after the observation it ties with.
			overlayEntry(`overlay_0min_${s.text}.png`, 0, 'forecast'),
			...OVERLAY_LEADS.map((lead) => overlayEntry(`overlay_${lead}min_${s.text}.png`, lead))
		]
	};
}

createServer(async (req, res) => {
	const url = new URL(req.url, `http://localhost:${PORT}`);
	const send = (status, body, type, extra = {}) => {
		res.writeHead(status, { 'content-type': type, 'access-control-allow-origin': '*', ...extra });
		res.end(body);
	};
	const sendJson = (status, body) => send(status, JSON.stringify(body), 'application/json');

	if (url.pathname === '/api/push/config') {
		return sendJson(200, PUSH_CONFIG);
	}
	if (url.pathname === '/api/push/subscribe' && req.method === 'POST') {
		const body = await readJson(req);
		const endpoint = body?.subscription?.endpoint;
		if (!endpoint || typeof body.lat !== 'number' || typeof body.lon !== 'number') {
			return sendJson(422, { detail: 'invalid subscription payload' });
		}
		if (!inCoverage(body.lat, body.lon)) {
			return sendJson(400, { detail: 'coordinates outside the radar composite grid' });
		}
		const created = !subscriptions.has(endpoint);
		subscriptions.set(endpoint, { ...body, updated_at: new Date().toISOString() });
		console.log(
			`[mock-sidecar] ${created ? 'subscribed' : 'updated'} ${body.lat.toFixed(3)},${body.lon.toFixed(3)} ` +
				`· ${body.threshold_pct} % / ${body.lead_min} min · ${subscriptions.size} device(s)`
		);
		return sendJson(200, { ok: true, created });
	}
	if (url.pathname === '/api/push/unsubscribe' && req.method === 'POST') {
		const body = await readJson(req);
		const deleted = typeof body?.endpoint === 'string' && subscriptions.delete(body.endpoint);
		console.log(`[mock-sidecar] unsubscribed · ${subscriptions.size} device(s)`);
		return sendJson(200, { ok: true, deleted });
	}
	if (url.pathname === '/nowcast/quality.json') {
		// Read per request, so editing the fixture shows up on a reload; the
		// stamp is moved to now, or the page's "computed …" line would sit
		// permanently on the day the fixture was written.
		const quality = JSON.parse(
			readFileSync(new URL('../src/lib/quality/fixture.json', import.meta.url), 'utf8')
		);
		quality.generated_at_utc = new Date().toISOString();
		return send(200, JSON.stringify(quality), 'application/json', {
			'cache-control': 'no-cache'
		});
	}
	if (url.pathname === '/nowcast/manifest.json') {
		return send(200, JSON.stringify(manifest(), null, 2), 'application/json', {
			'cache-control': 'public, max-age=30'
		});
	}
	const artifact = url.pathname.match(/^\/nowcast\/(\w+?)_?(?:(\d+)min|now)?_(\d{12})\.png$/);
	if (artifact) {
		const [, product, leadText, cycleText] = artifact;
		// A "now" overlay stamped before the current cycle is a history frame:
		// its lead is negative, and the blobs are rewound to match.
		const lead = leadText
			? Number(leadText)
			: Math.round((stampMs(cycleText) - stampMs(stamp().text)) / 60_000);
		// A `…_Nmin_` overlay is a forecast frame and is drawn frame-age
		// corrected, so it depicts the instant its `valid_ts_utc` claims; a
		// `…_now_` one is the measurement itself.
		const png =
			product === 'overlay'
				? overlayPng(leadText ? FRAME_AGE_MIN + lead : lead)
				: productPng(product, product === 'p_rain' || product === 'forecast_mm_h' ? lead : 0);
		return send(200, png, 'image/png', { 'cache-control': 'public, max-age=300, immutable' });
	}
	if (url.pathname === '/forecast') {
		const lat = Number(url.searchParams.get('lat'));
		const lon = Number(url.searchParams.get('lon'));
		const s = stamp();
		const generated = new Date(Date.parse(s.iso) + FRAME_AGE_MIN * 60_000);
		return send(
			200,
			JSON.stringify({
				lat,
				lon,
				radar_ts_utc: s.iso,
				n_members: 24,
				calibrated: true,
				calibration_fitted_at: '2026-08-01T03:00:00+00:00',
				per_lead: PROB_LEADS.map((lead) => ({ lead_min: lead, p_rain: 0.5 })),
				eta_min: 18,
				intensity_mm_h: 2.4,
				// Fixed, like every other number on this endpoint. Deliberately
				// the incident's own shape: a measurement above the 0.5 mm/h
				// threshold (the composite behind it is minutes old) over a
				// field that is dry now and wet again at +30. The headline must
				// read the field, not the measurement, and so must say "rain in
				// about 18 min" here rather than "it is raining here now".
				observed_mm_h: 1.2,
				generated_at_utc: generated.toISOString(),
				forecast_mm_h: FORECAST_LEADS.map((lead) => ({
					lead_min: lead,
					valid_ts_utc: new Date(generated.getTime() + lead * 60_000).toISOString(),
					mm_h: lead >= 30 ? 2.4 : 0
				})),
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
