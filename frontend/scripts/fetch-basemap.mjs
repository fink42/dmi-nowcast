#!/usr/bin/env node
/**
 * fetch-basemap.mjs — one-time (per deploy) fetch of the self-hosted basemap.
 *
 * The site serves its own basemap so there is no third-party tile server, no
 * usage policy and no external runtime dependency: everything the map needs
 * comes from our own origin. Three things are fetched into `static/`, all of
 * them gitignored (they are build inputs, not source):
 *
 *   static/basemap.pmtiles           a Denmark bbox extract of the Protomaps
 *                                    daily planet build (OpenStreetMap data)
 *   static/basemap-assets/fonts/     Noto Sans glyph PBFs used by the style
 *   static/basemap-assets/sprites/   the Protomaps v4 sprite sheets
 *
 * The extract is done with `pmtiles extract`, which slices the bbox straight
 * out of the remote planet archive over HTTP range requests — no 137 GB
 * download. The binary comes from the go-pmtiles releases (the `pmtiles` npm
 * package is a browser/JS *library* only; the CLI is a Go program) and is
 * cached in `.cache/` unless a `pmtiles` is already on PATH.
 *
 * Usage:  node scripts/fetch-basemap.mjs [--maxzoom 12] [--force]
 */
import { createWriteStream } from 'node:fs';
import { mkdir, rm, stat, writeFile, chmod, readdir } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import os from 'node:os';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const STATIC = path.join(ROOT, 'static');
const CACHE = path.join(ROOT, '.cache');
const OUT_PMTILES = path.join(STATIC, 'basemap.pmtiles');
const ASSETS = path.join(STATIC, 'basemap-assets');

// Denmark + a margin of sea/neighbour coast, so panning off the coast does
// not fall off the basemap. Bornholm (15.2E) and Skagen (57.8N) included.
const BBOX = { west: 7.4, south: 54.3, east: 15.5, north: 58.1 };

// The three font stacks the Protomaps basemap layers ask for.
const FONT_STACKS = ['Noto Sans Regular', 'Noto Sans Medium', 'Noto Sans Italic'];
const ASSETS_BASE = 'https://protomaps.github.io/basemaps-assets';
const ASSETS_REPO_TREE =
	'https://api.github.com/repos/protomaps/basemaps-assets/git/trees/main?recursive=1';
const SPRITE_FLAVORS = ['light', 'dark'];

const args = process.argv.slice(2);
const flag = (name, fallback) => {
	const i = args.indexOf(`--${name}`);
	return i === -1 ? fallback : args[i + 1];
};
const MAXZOOM = Number(flag('maxzoom', '12'));
const FORCE = args.includes('--force');

const log = (...m) => console.log('[basemap]', ...m);
const mb = (bytes) => `${(bytes / 1e6).toFixed(1)} MB`;

async function fileSize(p) {
	try {
		return (await stat(p)).size;
	} catch {
		return null;
	}
}

async function fetchOk(url, init) {
	const res = await fetch(url, init);
	if (!res.ok) throw new Error(`GET ${url} → ${res.status} ${res.statusText}`);
	return res;
}

/** Newest available Protomaps daily build (walk back until one exists). */
async function latestDailyBuild(maxDaysBack = 10) {
	for (let back = 0; back <= maxDaysBack; back++) {
		const d = new Date(Date.now() - back * 86400_000);
		const stamp = d.toISOString().slice(0, 10).replaceAll('-', '');
		const url = `https://build.protomaps.com/${stamp}.pmtiles`;
		const res = await fetch(url, { method: 'HEAD' });
		if (res.ok) return { url, stamp, size: Number(res.headers.get('content-length') ?? 0) };
	}
	throw new Error('no Protomaps daily build found in the last 10 days');
}

/** `pmtiles` from PATH, else the cached go-pmtiles release, else download it. */
async function ensurePmtilesCli() {
	const onPath = await which('pmtiles');
	if (onPath) {
		log('using pmtiles from PATH:', onPath);
		return onPath;
	}
	const binDir = path.join(CACHE, 'pmtiles');
	const cached = path.join(binDir, 'pmtiles');
	if (await fileSize(cached)) {
		log('using cached pmtiles CLI:', cached);
		return cached;
	}

	const platform = { darwin: 'Darwin', linux: 'Linux', win32: 'Windows' }[process.platform];
	const arch = { arm64: 'arm64', x64: 'x86_64' }[process.arch];
	if (!platform || !arch) {
		throw new Error(`unsupported platform for the pmtiles CLI: ${process.platform}/${process.arch}`);
	}
	const release = await (
		await fetchOk('https://api.github.com/repos/protomaps/go-pmtiles/releases/latest', {
			headers: { accept: 'application/vnd.github+json' }
		})
	).json();
	const asset = release.assets.find(
		(a) => a.name.includes(`_${platform}_${arch}`) && /\.(zip|tar\.gz)$/.test(a.name)
	);
	if (!asset) throw new Error(`no go-pmtiles asset for ${platform}/${arch}`);

	await mkdir(binDir, { recursive: true });
	const archive = path.join(binDir, asset.name);
	log(`downloading ${asset.name} (${mb(asset.size)})`);
	const res = await fetchOk(asset.browser_download_url);
	await pipeline(Readable.fromWeb(res.body), createWriteStream(archive));
	if (asset.name.endsWith('.zip')) {
		await run('unzip', ['-o', '-q', archive, '-d', binDir]);
	} else {
		await run('tar', ['-xzf', archive, '-C', binDir]);
	}
	await rm(archive, { force: true });
	await chmod(cached, 0o755);
	log('pmtiles CLI ready:', cached, `(go-pmtiles ${release.tag_name})`);
	return cached;
}

function which(cmd) {
	return new Promise((resolve) => {
		const p = spawn('sh', ['-c', `command -v ${cmd}`]);
		let out = '';
		p.stdout.on('data', (c) => (out += c));
		p.on('close', (code) => resolve(code === 0 ? out.trim() : null));
		p.on('error', () => resolve(null));
	});
}

function run(cmd, argv) {
	return new Promise((resolve, reject) => {
		const p = spawn(cmd, argv, { stdio: 'inherit' });
		p.on('close', (code) =>
			code === 0 ? resolve() : reject(new Error(`${cmd} exited with ${code}`))
		);
		p.on('error', reject);
	});
}

async function extractBasemap() {
	const existing = await fileSize(OUT_PMTILES);
	if (existing && !FORCE) {
		log(`static/basemap.pmtiles already present (${mb(existing)}) — pass --force to refetch`);
		return;
	}
	const build = await latestDailyBuild();
	log(`daily build ${build.stamp} (${mb(build.size)} planet), extracting bbox`, BBOX);
	const cli = await ensurePmtilesCli();
	await mkdir(STATIC, { recursive: true });
	const tmp = path.join(os.tmpdir(), `basemap-${process.pid}.pmtiles`);
	await run(cli, [
		'extract',
		build.url,
		tmp,
		`--bbox=${BBOX.west},${BBOX.south},${BBOX.east},${BBOX.north}`,
		`--maxzoom=${MAXZOOM}`
	]);
	await run('sh', ['-c', `mv ${JSON.stringify(tmp)} ${JSON.stringify(OUT_PMTILES)}`]);
	log(`wrote static/basemap.pmtiles (${mb(await fileSize(OUT_PMTILES))}, maxzoom ${MAXZOOM})`);
}

/** Glyph + sprite assets, so the style needs nothing off-origin at runtime. */
async function fetchStyleAssets() {
	let paths = null;
	try {
		const tree = await (
			await fetchOk(ASSETS_REPO_TREE, { headers: { accept: 'application/vnd.github+json' } })
		).json();
		paths = tree.tree
			.filter((n) => n.type === 'blob')
			.map((n) => n.path)
			.filter(
				(p) =>
					(p.startsWith('fonts/') && FONT_STACKS.some((s) => p.startsWith(`fonts/${s}/`))) ||
					(p.startsWith('sprites/v4/') && SPRITE_FLAVORS.some((f) => path.basename(p).startsWith(f)))
			);
	} catch (err) {
		log('GitHub tree listing unavailable, falling back to the standard glyph ranges:', err.message);
		paths = [];
		for (const stack of FONT_STACKS) {
			for (let i = 0; i < 256; i++) paths.push(`fonts/${stack}/${i * 256}-${i * 256 + 255}.pbf`);
		}
		for (const f of SPRITE_FLAVORS) {
			for (const suffix of ['.json', '.png', '@2x.json', '@2x.png'])
				paths.push(`sprites/v4/${f}${suffix}`);
		}
	}

	let done = 0;
	let bytes = 0;
	let missing = 0;
	const queue = [...paths];
	const worker = async () => {
		for (let p = queue.pop(); p !== undefined; p = queue.pop()) {
			const target = path.join(ASSETS, p);
			if (!FORCE && (await fileSize(target))) {
				done++;
				continue;
			}
			const url = `${ASSETS_BASE}/${p.split('/').map(encodeURIComponent).join('/')}`;
			const res = await fetch(url);
			if (!res.ok) {
				missing++;
				continue;
			}
			const buf = Buffer.from(await res.arrayBuffer());
			await mkdir(path.dirname(target), { recursive: true });
			await writeFile(target, buf);
			bytes += buf.length;
			done++;
		}
	};
	await Promise.all(Array.from({ length: 12 }, worker));
	log(`style assets: ${done} files (${mb(bytes)} newly fetched, ${missing} absent upstream)`);
}

async function summarise() {
	const total = async (dir) => {
		let sum = 0;
		for (const e of await readdir(dir, { withFileTypes: true, recursive: true })) {
			if (e.isFile()) sum += (await fileSize(path.join(e.parentPath ?? e.path, e.name))) ?? 0;
		}
		return sum;
	};
	log(`static/basemap.pmtiles      ${mb((await fileSize(OUT_PMTILES)) ?? 0)}`);
	log(`static/basemap-assets/      ${mb(await total(ASSETS))}`);
}

await extractBasemap();
await fetchStyleAssets();
await summarise();
log('done.');
