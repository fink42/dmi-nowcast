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
		lag: (min: number) => `−${min} min`,
		frameOf: (i: number, n: number) => `Frame ${i} of ${n}`,
		scrubber: 'Timeline',
		stateObserved: 'observed',
		stateNow: 'latest observation',
		stateForecast: 'forecast',
		buffering: 'loading frame …',
		radarAge: (min: number) =>
			min < 1 ? 'Radar data: less than 1 min old' : `Radar data: ${min} min old`,
		pipelineStale: 'The pipeline has stalled — the forecast is not being updated right now',
		radarOld: 'The radar image is older than usual — no newer one has arrived from DMI yet',
		noFrames: 'No radar frames available right now'
	},
	panel: {
		title: 'Forecast for this point',
		close: 'Close forecast',
		collapse: 'Minimise forecast',
		expand: 'Show forecast',
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
		motionLabel: 'Cell motion',
		motionValue: (from: string, kmh: number) => `Coming from ${from} · ${kmh} km/h`,
		motionNone: 'No measured cell motion here',
		compass: {
			n: 'N',
			ne: 'NE',
			e: 'E',
			se: 'SE',
			s: 'S',
			sw: 'SW',
			w: 'W',
			nw: 'NW'
		},
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
	push: {
		title: 'Tell me when rain is coming',
		enable: 'Notify me here',
		working: 'Working …',
		showSettings: 'Settings',
		hideSettings: 'Hide settings',
		thresholdLabel: 'Probability at least',
		thresholdOption: (pct: number) => `${pct} %`,
		leadLabel: 'Warning time',
		leadOption: (min: number) => `${min} min`,
		quietLabel: 'No notifications at night',
		quietFrom: 'From',
		quietTo: 'To',
		save: 'Save',
		cancel: 'Cancel',
		edit: 'Change',
		moveHere: 'Move here',
		stop: 'Stop notifications',
		summary: (coordinates: string, leadMin: number, thresholdPct: number) =>
			`You will be notified at ${coordinates} when the chance of rain within ${leadMin} min is above ${thresholdPct} %.`,
		summaryQuiet: (start: string, end: string) =>
			`Nothing will be sent between ${start} and ${end}.`,
		movedAway: 'Notifications are set for a different point than the one you are looking at.',
		iosNotInstalled:
			'On iPhone and iPad the site has to be on your home screen before it may send notifications: tap Share, choose “Add to Home Screen”, and open it from there.',
		unsupported: 'This browser cannot send notifications about rain.',
		insecure: 'Notifications need a secure connection (https).',
		denied:
			'Notifications are blocked for this site. You can allow them again in your browser settings.',
		capacity: 'No room for more devices right now. Please try again later.',
		retry: 'Try again',
		fallbackTitle: 'Rain radar',
		fallbackBody: 'New notification',
		errors: {
			permission: 'You did not give the browser permission to show notifications.',
			offCoverage: 'That point is outside radar coverage — nothing can be sent from there.',
			unavailable: 'The server is not sending notifications at the moment.',
			failed: 'Notifications could not be turned on. Please try again shortly.'
		}
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
			'Notifications are opt-in. If you turn them on, the server stores your browser’s push address, the point you chose and your settings for as long as the subscription exists. They are deleted when you tap “stop notifications”, or when the push service reports the address as gone. Delivery goes through your browser vendor’s push service (Apple, Google or Mozilla).',
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
