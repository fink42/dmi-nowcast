import type { Catalog } from './types';

/**
 * English catalog. Typed as `Catalog`, so the compiler rejects it the moment
 * a string exists in Danish and not here.
 */
export const en: Catalog = {
	locale: 'en',
	htmlLang: 'en',
	site: {
		title: 'Rain radar',
		tagline: 'Short-range rain nowcast for Denmark',
		description:
			'When does the rain arrive? A short-range nowcast for all of Denmark, computed from DMI radar.',
		skipToMap: 'Skip to the map'
	},
	nav: {
		map: 'Map',
		about: 'About',
		data: 'Data',
		privacy: 'Privacy',
		support: 'Support',
		menu: 'Menu',
		close: 'Close'
	},
	lang: {
		toggle: 'Language',
		da: 'Dansk',
		en: 'English',
		switchToEn: 'Switch to English',
		switchToDa: 'Switch to Danish'
	},
	map: {
		label: 'Map of Denmark with rain radar',
		attributionRadar: 'Radar data: DMI',
		locate: 'Use my location',
		locating: 'Finding your location …',
		locateDenied: 'Location not permitted. Tap somewhere on the map instead.',
		locateFailed: 'Could not find your location. Tap somewhere on the map instead.',
		locateOutside: 'Your location is outside the radar coverage of Denmark.',
		hint: 'Tap anywhere on the map for a forecast',
		zoomIn: 'Zoom in',
		zoomOut: 'Zoom out',
		selectedPoint: 'Selected point'
	},
	loop: {
		play: 'Play',
		pause: 'Pause',
		now: 'Now',
		lead: (min: number) => `+${min} min`,
		frameOf: (i: number, n: number) => `Frame ${i} of ${n}`,
		scrubber: 'Timeline',
		opacity: 'Opacity',
		radarAge: (min: number) =>
			min < 1 ? 'Radar data: less than 1 min old' : `Radar data: ${min} min old`,
		radarTime: (time: string) => `Radar image at ${time}`,
		pipelineStale: 'The pipeline has stalled — the forecast is not being updated right now',
		radarOld: 'The radar image is older than usual — no newer one has arrived from DMI yet',
		noFrames: 'No radar frames available right now'
	},
	panel: {
		title: 'Forecast for this point',
		close: 'Close forecast',
		loading: 'Fetching forecast …',
		headlineRainingNow: 'It is raining here now',
		headlineEta: (min: number) => `Rain in about ${min} min`,
		headlineNoRain: 'No rain expected within the hour',
		headlineUnknown: 'No forecast for this point',
		etaLabel: 'Expected arrival',
		etaValue: (min: number) => `in about ${min} min`,
		etaNone: 'no rain within the horizon',
		intensityLabel: 'Intensity',
		intensityValue: (mmH: number) => `${mmH.toFixed(1)} mm/h`,
		intensityNone: 'none',
		intensityLight: 'light',
		intensityModerate: 'moderate',
		intensityHeavy: 'heavy',
		intensityViolent: 'violent',
		probabilityLabel: 'Probability of rain',
		probabilityValue: (pct: number) => `${pct}%`,
		probabilityWithin: (min: number, pct: number) =>
			`Probability of rain within ${min} min: ${pct}%`,
		confidenceLabel: 'Confidence',
		confidenceValue: (pct: number) => `${pct}%`,
		confidenceHigh: 'high',
		confidenceMedium: 'medium',
		confidenceLow: 'low',
		radarAgeLabel: 'Radar data',
		radarAgeValue: (min: number) => (min < 1 ? 'less than 1 min old' : `${min} min old`),
		calibratedBadge: 'Calibrated',
		calibratedTooltip:
			'Probabilities are isotonically calibrated against a backtest corpus from DMI’s archive.',
		uncalibratedBadge: 'Uncalibrated',
		uncalibratedTooltip:
			'This cycle serves raw probabilities — they have not been adjusted against backtest data.',
		offCoverage: 'Outside radar coverage',
		offCoverageBody:
			'This point lies outside the radar composite the forecast is built from. That is not a 0% chance — we simply do not know.',
		coordinates: (lat: number, lon: number) => `${lat.toFixed(3)}° N, ${lon.toFixed(3)}° E`,
		sourceLocal: 'Computed in your browser',
		sourceServer: 'Fetched from the server',
		error: 'The forecast could not be loaded. Try again in a moment.',
		leadAxis: 'Minutes ahead'
	},
	status: {
		loading: 'Loading data …',
		offline: 'No connection to the service',
		offlineCached: 'No connection to the service — showing the last data received',
		noData: 'The service has no data yet. The first cycle takes a few minutes.',
		retry: 'Try again',
		updated: (time: string) => `Updated at ${time}`
	},
	footer: {
		disclaimer: 'Not an official warning service',
		official: 'Official warnings come from DMI.',
		attribution: 'Radar data: DMI',
		source: 'Source code on GitHub'
	},
	about: {
		title: 'About',
		lead: 'A hobby project trying to answer one question: when does it start raining here?',
		honestyTitle: 'What this is',
		honesty:
			'A hobby project, built in spare evenings with heavy AI assistance — vibe-coded, because that is the time budget that exists. It runs on one small server, with no SLA and no promise that anything still works tomorrow. The algorithm and the validation behind it are real, though.',
		howTitle: 'How it works',
		how: [
			'Every five minutes the service fetches DMI’s national radar composite covering all of Denmark.',
			'Two consecutive images give a motion field — dense optical flow showing where the showers are heading.',
			'The rain field is then carried forward along that motion, minute by minute, up to an hour ahead.',
			'A STEPS ensemble repeats this 24 times with stochastic noise on the small scales that lose predictability first, and the fraction of members with rain becomes a probability.',
			'That probability is finally calibrated against a backtest corpus from DMI’s own archive, so “70%” is meant to be worth 70%.'
		],
		limitsTitle: 'The limits',
		limits: [
			'The radar measures the strongest echo in the column above the ground — not rain at the ground. Intensity is therefore an upper-bound proxy, not a rain gauge.',
			'Advected rain loses its edge fast: 0–30 minutes is useful, out towards 60 minutes it is weak, and beyond that a weather model wins decisively.',
			'Summer showers are roughly three times harder than steady winter rain.'
		],
		sourceTitle: 'Source code',
		sourceBody: 'Everything — algorithm, service and this site — is openly available on GitHub.',
		sourceLink: 'github.com/nsimonsen/dmi-nowcast'
	},
	data: {
		title: 'Data and attribution',
		radarTitle: 'Radar data',
		radarBody:
			'The radar composites come from DMI Open Data and are used under DMI’s free-data terms, which require attribution. DMI is not responsible for the derived products on this site.',
		radarLink: 'DMI Open Data',
		basemapTitle: 'Basemap',
		basemapBody:
			'The basemap is Protomaps cartography built on OpenStreetMap data, hosted on our own server as a single pmtiles file. No third-party tile server is involved.',
		basemapLinkOsm: '© OpenStreetMap contributors',
		basemapLinkProtomaps: 'Protomaps',
		boundaryTitle: 'Boundaries',
		boundaryBody:
			'The Denmark outline used in the calibration comes from Natural Earth and is public domain.',
		boundaryLink: 'Natural Earth',
		methodTitle: 'Method and literature',
		methodBody:
			'The nowcast uses optical flow with Lagrangian extrapolation plus the STEPS ensemble from pysteps (Pulkkinen et al. 2019). The yardstick for expected skill is Imhoff et al. 2020 over Dutch lowland catchments.'
	},
	privacy: {
		title: 'Privacy',
		lead: 'The short version: this site collects nothing about you.',
		points: [
			'No accounts, no logins, no newsletters.',
			'No analytics, no trackers, no ads.',
			'No cookies. The only thing stored in your browser is your language choice (localStorage).',
			'If you tap “use my location”, the coordinates are used inside your own browser to look up the forecast in grids you already downloaded. They are not sent anywhere.',
			'The server keeps ordinary web request logs so it can be operated, and they are used for nothing else.'
		],
		contactTitle: 'Questions',
		contactBody: 'Questions about the data are welcome as an issue on GitHub.'
	},
	support: {
		title: 'Support the project',
		lead: 'The site is free and ad-free. It runs on a small server at home, and that is the entire operating budget.',
		body: 'There are no donation links yet. When there are, they will appear here — until then the best support is to use the site, report bugs and tell someone about it.',
		soonTitle: 'Coming later',
		soon: 'GitHub Sponsors and Ko-fi will be activated later.',
		thanks: 'Thanks for stopping by.'
	},
	error: {
		title: 'Page not found',
		body: 'The page you are looking for does not exist.',
		backToMap: 'Back to the map'
	}
};
