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
		quality: 'Quality',
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
		/**
		 * Two marks on one track, and they are not the same instant: `latest` is
		 * the newest radar image (14–24 min old whenever anyone looks) and `now`
		 * is the viewer's own clock, out among the forecast frames.
		 */
		now: 'Now',
		latest: 'Latest',
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
		etaNow: 'now',
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
		calibratedMore: 'See how good we are.',
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
		qualityTitle: 'How good are we?',
		qualityBody:
			'The promise that “70%” is worth 70% is a testable one — and it has been tested, against both the radar and DMI’s rain gauges.',
		qualityLink: 'See the numbers',
		sourceTitle: 'Source code',
		sourceBody: 'Everything — algorithm, service and this site — is openly available on GitHub.',
		sourceLink: 'github.com/nsimonsen/dmi-nowcast'
	},
	quality: {
		title: 'How good are we?',
		description:
			'How good the nowcast and the rain notifications are, measured against DMI’s radar and DMI’s rain gauges.',
		lead: 'These numbers are measured against two truths that see different things: the radar, which covers the whole country, and DMI’s rain gauges, which measure what a person feels.',
		loading: 'Loading the numbers …',
		error: 'The numbers could not be loaded. Please try again shortly.',
		notMeasured: 'Not measured yet.',
		generatedAt: (when: string) => `Computed ${when}.`,
		/** Percent sign, English spacing; used everywhere on the page. */
		percent: (value: number) => `${value}%`,
		headline: {
			reliabilityTitle: 'Do the probabilities hold?',
			reliabilityBoth: (said: string, radar: string, gauge: string) =>
				`When we say ${said}, it rains ${radar} of the time measured on the radar, and ${gauge} measured at the gauges.`,
			reliabilityRadar: (said: string, radar: string) =>
				`When we say ${said}, it rains ${radar} of the time measured on the radar.`,
			reliabilityGauge: (said: string, gauge: string) =>
				`When we say ${said}, it rains ${gauge} of the time measured at the gauges.`,
			reliabilityLead: (lead: number, n: string) =>
				`Forecasts ${lead} minutes ahead · ${n} comparisons.`,
			reliabilityLeadPair: (radarLead: number, gaugeLead: number) =>
				`Radar ${radarLead} minutes ahead · gauges ${gaugeLead} minutes ahead.`,
			warningsTitle: 'Do the notifications hold?',
			warningsLate: (
				total: string,
				days: number,
				hits: string,
				falseAlarms: string,
				minutes: string
			) =>
				`Of ${total} notifications over ${days} measured days, ${hits} were followed by rain at the gauge within the window, ${falseAlarms} were false alarms, and the median notification came ${minutes} minutes late.`,
			warningsEarly: (
				total: string,
				days: number,
				hits: string,
				falseAlarms: string,
				minutes: string
			) =>
				`Of ${total} notifications over ${days} measured days, ${hits} were followed by rain at the gauge within the window, ${falseAlarms} were false alarms, and the median notification came ${minutes} minutes early.`,
			warningsOnTime: (total: string, days: number, hits: string, falseAlarms: string) =>
				`Of ${total} notifications over ${days} measured days, ${hits} were followed by rain at the gauge within the window, ${falseAlarms} were false alarms, and the median notification landed on time.`,
			warningsRates: (pod: string, far: string, stations: string) =>
				`We caught ${pod} of the rain at ${stations} stations; ${far} of the notifications were false.`,
			marginTitle: 'Are we better than nothing?',
			marginBeats: (horizon: number, points: string) =>
				`${horizon} minutes ahead we beat “assume nothing moves” by ${points} points of CSI.`,
			marginBehind: (horizon: number, points: string) =>
				`${horizon} minutes ahead we are ${points} points of CSI behind “assume nothing moves”.`,
			marginTied: (horizon: number) =>
				`${horizon} minutes ahead we are level with “assume nothing moves”.`,
			marginDetail: (advection: string, persistence: string, frames: string) =>
				`CSI ${advection} against ${persistence} · measured over ${frames} radar frames.`
		},
		reliability: {
			title: 'When we say a probability',
			intro: 'Each dot gathers the forecasts that promised about the same thing: across, what we said; up, how often it then rained. A dot on the diagonal means 70% was worth 70%. Below it, we promised too much.',
			markerNote: 'The size of a dot shows how many forecasts are behind it.',
			radar: 'Radar',
			gauge: 'Gauges',
			perfect: 'Perfect calibration',
			axisX: 'Probability we said',
			axisY: 'It rained',
			panel: (lead: number) => `${lead} minutes ahead`,
			brierRadar: (value: string) => `Brier radar ${value}`,
			brierGauge: (value: string) => `Brier gauges ${value}`,
			tableToggle: 'Show the numbers as a table',
			tableCaption: (lead: number) => `Calibration ${lead} minutes ahead`,
			colBin: 'We said',
			colRadar: 'Radar: it rained',
			colRadarN: 'Radar: n',
			colGauge: 'Gauges: it rained',
			colGaugeN: 'Gauges: n',
			empty: '—',
			none: 'Calibration has not been measured yet.'
		},
		stations: {
			title: 'The gauges, one by one',
			intro: 'Each dot is a DMI station. The colour shows how much of the rain at that station we managed to warn about. Where there are too few notifications for that number, the dot is coloured by its Brier score instead and drawn as a ring.',
			mapLabel: 'Map of Denmark with DMI’s measuring stations',
			legendTitle: 'Share of the rain we warned about',
			legendPoor: 'under 50%',
			legendFair: '50–65%',
			legendGood: '65–80%',
			legendBest: 'over 80%',
			legendUnknown: 'no score',
			legendBrier: 'Ring: the colour comes from the Brier score, not from the share.',
			hint: 'Point at a dot to see that station’s numbers — they are in the list below as well.',
			stationLabel: (name: string, kind: string) => `${name} (${kind})`,
			stationPod: (pod: string) => `Rain warned about: ${pod}`,
			stationFar: (far: string) => `false notifications: ${far}`,
			stationBrier: (brier: string) => `Brier ${brier}`,
			stationEvents: (events: string, warnings: string) =>
				`${events} rain events, ${warnings} notifications`,
			stationNoScore: 'Too few notifications for a score',
			none: 'There are no stations to show yet.'
		},
		rainingNow: {
			title: '“It is raining here now”',
			sentence: (agreement: string, pod: string, far: string) =>
				`When we report rain at a gauge, we are right ${agreement} of the time: we catch ${pod} of the wet ten-minute slots, and ${far} of our wet calls were dry on the ground.`,
			comparison: (observation: string) =>
				`The raw radar image alone would have agreed ${observation}.`,
			detail: (slots: string) => `Measured over ${slots} ten-minute slots.`,
			none: 'Not measured yet.'
		},
		events: {
			title: 'Latest verified notifications',
			intro: 'The newest notifications, held against the rain gauge at the station.',
			colStation: 'Station',
			colWarned: 'Notified',
			colSaid: 'We said',
			colHappened: 'What happened',
			colError: 'Off by',
			said: (min: number, probability: string) => `rain in about ${min} min (${probability})`,
			onset: (time: string) => `rain at ${time}`,
			noRain: 'no rain',
			early: (min: number) => `${min} min early`,
			late: (min: number) => `${min} min late`,
			onTime: 'on time',
			empty: '—',
			none: 'No notifications have been verified yet.'
		},
		methods: {
			title: 'How this is measured',
			radarTitle: 'The radar as truth',
			radarBody:
				'The radar covers the whole country and sees every shower, including the ones that never reach a gauge. What it cannot see is itself: it measures the strongest echo in the column above the ground, so virga, the melting layer and clutter from buildings and wind turbines all count as rain — and the threshold for “rain” is our own.',
			gaugeTitle: 'The gauges as truth',
			gaugeBody:
				'The gauges measure what a person feels: water in a funnel on the ground. But they are about a hundred points in the whole country, they count in ten-minute boxes, they cannot see less than 0.1 mm, and in wind they catch less than actually falls.',
			rulesTitle: 'The rules behind the numbers',
			wetRule: 'Wet gauge',
			onsetRule: 'Onset of rain',
			threshold: 'Rain threshold',
			thresholdValue: (mmH: string) => `${mmH} mm/h`,
			frameAge: 'Age of the radar image',
			frameAgeValue: (min: string, max: string) => `${min}–${max} minutes`,
			subscriberRule: 'When a notification is sent',
			subscriberRuleValue: (
				threshold: string,
				lead: number,
				persistence: number,
				rearm: number
			) =>
				`Probability above ${threshold} within ${lead} minutes, confirmed on ${persistence} consecutive cycles, and no new notification until ${rearm} minutes later.`,
			leadErrorNote:
				'A positive error means the rain had already started when the notification said it would arrive — the notification came late.',
			windowsTitle: 'Periods',
			radarWindow: (from: string, to: string, events: string, points: string) =>
				`Radar: ${from}–${to}, ${events} rain events, ${points} comparisons.`,
			gaugeWindow: (from: string, to: string, events: string, stations: string) =>
				`Gauges: ${from}–${to}, ${events} events at ${stations} stations.`,
			liveWindow: (days: number, from: string, to: string) =>
				`Notifications: the last ${days} days (${from}–${to}).`,
			cadence: 'Every number on this page is recomputed nightly.',
			sourcesTitle: 'Sources',
			sourceRadar: 'Radar',
			sourceGauges: 'Gauges',
			attribution: 'Radar data and observations: DMI.'
		}
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
