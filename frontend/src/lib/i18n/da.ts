/**
 * Danish catalog — the reference locale. Its shape is the `Catalog` type every
 * other locale must satisfy (see ./types.ts), so adding a string here makes
 * the compiler demand a translation everywhere else.
 *
 * Strings that need values are functions; everything else is a plain string.
 * Nothing user-visible may be written outside this file and its siblings.
 */
export const da = {
	locale: 'da',
	htmlLang: 'da',
	site: {
		title: 'Regnradar',
		tagline: 'Kortsigtet regnvarsel for Danmark',
		description:
			'Hvornår kommer regnen? Kortsigtet nowcast for hele Danmark, beregnet ud fra DMI’s radar.',
		skipToMap: 'Gå til kortet'
	},
	nav: {
		map: 'Kort',
		about: 'Om',
		quality: 'Kvalitet',
		data: 'Data',
		privacy: 'Privatliv',
		support: 'Støt',
		menu: 'Menu',
		close: 'Luk'
	},
	lang: {
		toggle: 'Sprog',
		da: 'Dansk',
		en: 'English',
		switchToEn: 'Skift til engelsk',
		switchToDa: 'Skift til dansk'
	},
	map: {
		label: 'Kort over Danmark med regnradar',
		attributionRadar: 'Radardata: DMI',
		locate: 'Brug min placering',
		locating: 'Finder din placering …',
		locateDenied: 'Placering ikke tilladt. Tryk et sted på kortet i stedet.',
		locateFailed: 'Kunne ikke finde din placering. Tryk et sted på kortet i stedet.',
		locateOutside: 'Din placering ligger uden for radarens dækning af Danmark.',
		hint: 'Tryk et sted på kortet for at få en prognose',
		zoomIn: 'Zoom ind',
		zoomOut: 'Zoom ud',
		selectedPoint: 'Valgt punkt'
	},
	loop: {
		play: 'Afspil',
		pause: 'Pause',
		/**
		 * To mærker på samme spor, og de er ikke samme tidspunkt: `latest` er det
		 * nyeste radarbillede (14–24 min gammelt, når nogen kigger), og `now` er
		 * beskuerens eget ur ude blandt prognosebillederne.
		 */
		now: 'Nu',
		latest: 'Seneste',
		lead: (min: number) => `+${min} min`,
		lag: (min: number) => `−${min} min`,
		frameOf: (i: number, n: number) => `Billede ${i} af ${n}`,
		scrubber: 'Tidslinje',
		stateObserved: 'observeret',
		stateNow: 'seneste observation',
		stateForecast: 'prognose',
		buffering: 'henter billede …',
		radarAge: (min: number) =>
			min < 1 ? 'Radardata: under 1 min gamle' : `Radardata: ${min} min gamle`,
		pipelineStale: 'Beregningen er sat i stå — prognosen bliver ikke opdateret lige nu',
		radarOld: 'Radarbilledet er ældre end normalt — der er endnu ikke kommet et nyere fra DMI',
		noFrames: 'Ingen radarbilleder tilgængelige lige nu'
	},
	panel: {
		title: 'Prognose for punktet',
		close: 'Luk prognose',
		collapse: 'Minimér prognose',
		expand: 'Vis prognose',
		loading: 'Henter prognose …',
		headlineRainingNow: 'Det regner her nu',
		headlineEta: (min: number) => `Regn om ca. ${min} min`,
		headlineNoRain: 'Ingen regn forventet den næste time',
		headlineUnknown: 'Ingen prognose for dette punkt',
		etaLabel: 'Forventet ankomst',
		etaValue: (min: number) => `om ca. ${min} min`,
		etaNow: 'nu',
		etaNone: 'ingen regn inden for horisonten',
		intensityLabel: 'Intensitet',
		intensityValue: (mmH: number) => `${mmH.toFixed(1)} mm/t`,
		intensityNone: 'ingen',
		intensityLight: 'let',
		intensityModerate: 'moderat',
		intensityHeavy: 'kraftig',
		intensityViolent: 'voldsom',
		motionLabel: 'Cellebevægelse',
		motionValue: (from: string, kmh: number) => `Kommer fra ${from} · ${kmh} km/t`,
		motionNone: 'Ingen målt cellebevægelse her',
		compass: {
			n: 'N',
			ne: 'NØ',
			e: 'Ø',
			se: 'SØ',
			s: 'S',
			sw: 'SV',
			w: 'V',
			nw: 'NV'
		},
		probabilityLabel: 'Sandsynlighed for regn',
		probabilityValue: (pct: number) => `${pct} %`,
		probabilityWithin: (min: number, pct: number) =>
			`Sandsynlighed for regn inden for ${min} min: ${pct} %`,
		confidenceLabel: 'Sikkerhed',
		confidenceValue: (pct: number) => `${pct} %`,
		confidenceHigh: 'høj',
		confidenceMedium: 'middel',
		confidenceLow: 'lav',
		radarAgeLabel: 'Radardata',
		radarAgeValue: (min: number) => (min < 1 ? 'under 1 min gamle' : `${min} min gamle`),
		calibratedBadge: 'Kalibreret',
		calibratedTooltip:
			'Sandsynlighederne er isotonisk kalibreret mod et backtest-korpus fra DMI’s arkiv.',
		uncalibratedBadge: 'Ukalibreret',
		uncalibratedTooltip:
			'Denne cyklus serverer rå sandsynligheder — de er ikke justeret mod backtest-data.',
		calibratedMore: 'Se hvor gode vi er.',
		offCoverage: 'Uden for radarens dækning',
		offCoverageBody:
			'Punktet ligger uden for det radarbillede, prognosen bygger på. Der står ikke 0 % — vi ved det simpelthen ikke.',
		coordinates: (lat: number, lon: number) => `${lat.toFixed(3)}° N, ${lon.toFixed(3)}° Ø`,
		sourceLocal: 'Beregnet i browseren',
		sourceServer: 'Hentet fra serveren',
		error: 'Prognosen kunne ikke hentes. Prøv igen om lidt.',
		leadAxis: 'Minutter frem'
	},
	/**
	 * Notifications. `fallbackTitle` / `fallbackBody` are read by the service
	 * worker for a push message it cannot parse — the only strings in the
	 * catalog that are rendered outside the app.
	 */
	push: {
		title: 'Besked når regnen nærmer sig',
		enable: 'Giv besked her',
		working: 'Arbejder …',
		showSettings: 'Indstillinger',
		hideSettings: 'Skjul indstillinger',
		thresholdLabel: 'Sandsynlighed mindst',
		thresholdOption: (pct: number) => `${pct} %`,
		leadLabel: 'Varsel',
		leadOption: (min: number) => `${min} min`,
		quietLabel: 'Ingen beskeder om natten',
		quietFrom: 'Fra',
		quietTo: 'Til',
		save: 'Gem',
		cancel: 'Fortryd',
		edit: 'Ændr',
		moveHere: 'Flyt hertil',
		stop: 'Stop besked',
		summary: (coordinates: string, leadMin: number, thresholdPct: number) =>
			`Du får besked ved ${coordinates}, når sandsynligheden for regn inden for ${leadMin} min er over ${thresholdPct} %.`,
		summaryQuiet: (start: string, end: string) =>
			`Du får ingen beskeder mellem ${start} og ${end}.`,
		movedAway: 'Beskeden er sat ved et andet punkt end det, du kigger på nu.',
		iosNotInstalled:
			'På iPhone og iPad skal siden lægges på hjemmeskærmen, før den må sende beskeder: tryk Del, vælg “Føj til hjemmeskærm”, og åbn siden derfra.',
		unsupported: 'Din browser kan ikke sende beskeder om regn.',
		insecure: 'Beskeder kræver en sikker forbindelse (https).',
		denied:
			'Beskeder er blokeret for denne side. Du kan tillade dem igen i browserens indstillinger.',
		capacity: 'Der er ikke plads til flere enheder lige nu. Prøv igen senere.',
		retry: 'Prøv igen',
		fallbackTitle: 'Regnradar',
		fallbackBody: 'Ny besked',
		errors: {
			permission: 'Du gav ikke browseren lov til at vise beskeder.',
			offCoverage: 'Punktet ligger uden for radarens dækning — derfra kan der ikke sendes besked.',
			unavailable: 'Serveren sender ingen beskeder lige nu.',
			failed: 'Beskeden kunne ikke slås til. Prøv igen om lidt.'
		}
	},
	status: {
		loading: 'Henter data …',
		offline: 'Ingen forbindelse til tjenesten',
		offlineCached: 'Ingen forbindelse til tjenesten — viser de sidst hentede data',
		noData: 'Tjenesten har endnu ingen data. Første cyklus tager et par minutter.',
		retry: 'Prøv igen',
		updated: (time: string) => `Opdateret kl. ${time}`
	},
	footer: {
		disclaimer: 'Ikke en officiel varslingstjeneste',
		official: 'Officielle varsler kommer fra DMI.',
		attribution: 'Radardata: DMI',
		source: 'Kildekode på GitHub'
	},
	about: {
		title: 'Om',
		lead: 'Et hobbyprojekt, der prøver at svare på ét spørgsmål: hvornår begynder det at regne her?',
		honestyTitle: 'Hvad det her er',
		honesty:
			'Det er et hobbyprojekt, bygget om aftenen med kraftig AI-hjælp — vibe-kodet, fordi det er den tid, der er. Det kører på én lille server, uden SLA og uden løfter om, at noget virker i morgen. Algoritmen og valideringen er til gengæld ægte.',
		howTitle: 'Sådan virker det',
		how: [
			'Hvert femte minut henter tjenesten DMI’s nationale radarkomposit for hele Danmark.',
			'To på hinanden følgende billeder giver et bevægelsesfelt — tæt optisk flow, der viser hvor byger er på vej hen.',
			'Regnfeltet flyttes derefter fremad langs bevægelsen, minut for minut, op til en time.',
			'Et STEPS-ensemble gentager det 24 gange med tilfældig støj på de små skalaer, som mister forudsigelighed først, og andelen af medlemmer med regn bliver til en sandsynlighed.',
			'Sandsynligheden kalibreres til sidst mod et backtest-korpus fra DMI’s eget arkiv, så “70 %” gerne skulle betyde 70 %.'
		],
		limitsTitle: 'Grænserne',
		limits: [
			'Radaren måler den kraftigste ekko i søjlen over jorden — ikke regn på jorden. Intensiteten er derfor et overkantsskøn, ikke en regnmåler.',
			'Fremskrivning af regn mister hurtigt sin skarphed: 0–30 minutter er nyttigt, op mod 60 minutter er svagt, og længere ude vinder en vejrmodel klart.',
			'Sommerens byger er omkring tre gange sværere end vinterens jævne regn.'
		],
		qualityTitle: 'Hvor gode er vi?',
		qualityBody:
			'Løftet om, at “70 %” skal være 70 % værd, er til at efterprøve — og det er prøvet efter mod både radaren og DMI’s regnmålere.',
		qualityLink: 'Se tallene',
		sourceTitle: 'Kildekode',
		sourceBody: 'Alt — algoritme, tjeneste og denne side — ligger frit tilgængeligt på GitHub.',
		sourceLink: 'github.com/nsimonsen/dmi-nowcast'
	},
	/**
	 * Kvalitetssiden. To sandheder ser hver sin ting — radaren, som dækker
	 * hele landet, og regnmålerne, som måler det man mærker — og hvert tal
	 * her siger hvilken af dem det er målt mod. Manglende måling har sin egen
	 * sætning (`notMeasured`): et nul og “ikke målt endnu” ser ens ud som tal
	 * og betyder det modsatte.
	 */
	quality: {
		title: 'Hvor gode er vi?',
		description:
			'Hvor gode nowcastet og regnvarslerne er, målt mod DMI’s radar og DMI’s regnmålere.',
		lead: 'Tallene her er målt mod to sandheder, der ser hver sin ting: radaren, som dækker hele landet, og DMI’s regnmålere, som måler det, man mærker.',
		loading: 'Henter tallene …',
		error: 'Tallene kunne ikke hentes. Prøv igen om lidt.',
		notMeasured: 'Ikke målt endnu.',
		generatedAt: (when: string) => `Beregnet ${when}.`,
		/** Procenttegn med dansk mellemrum; bruges overalt på siden. */
		percent: (value: number) => `${value} %`,
		headline: {
			reliabilityTitle: 'Passer sandsynlighederne?',
			reliabilityBoth: (said: string, radar: string, gauge: string) =>
				`Når vi siger ${said}, regner det ${radar} af gangene målt på radaren og ${gauge} målt på regnmålerne.`,
			reliabilityRadar: (said: string, radar: string) =>
				`Når vi siger ${said}, regner det ${radar} af gangene målt på radaren.`,
			reliabilityGauge: (said: string, gauge: string) =>
				`Når vi siger ${said}, regner det ${gauge} af gangene målt på regnmålerne.`,
			reliabilityLead: (lead: number, n: string) =>
				`Prognoser ${lead} minutter frem · ${n} sammenligninger.`,
			reliabilityLeadPair: (radarLead: number, gaugeLead: number) =>
				`Radar ${radarLead} minutter frem · regnmålere ${gaugeLead} minutter frem.`,
			warningsTitle: 'Holder varslerne?',
			warningsLate: (
				total: string,
				days: number,
				hits: string,
				falseAlarms: string,
				minutes: string
			) =>
				`Af ${total} varsler over ${days} målte dage blev ${hits} fulgt af regn ved måleren inden for vinduet, ${falseAlarms} var falske alarmer, og det midterste varsel kom ${minutes} minutter for sent.`,
			warningsEarly: (
				total: string,
				days: number,
				hits: string,
				falseAlarms: string,
				minutes: string
			) =>
				`Af ${total} varsler over ${days} målte dage blev ${hits} fulgt af regn ved måleren inden for vinduet, ${falseAlarms} var falske alarmer, og det midterste varsel kom ${minutes} minutter for tidligt.`,
			warningsOnTime: (total: string, days: number, hits: string, falseAlarms: string) =>
				`Af ${total} varsler over ${days} målte dage blev ${hits} fulgt af regn ved måleren inden for vinduet, ${falseAlarms} var falske alarmer, og det midterste varsel ramte tiden.`,
			warningsRates: (pod: string, far: string, stations: string) =>
				`Vi fangede ${pod} af regnen ved ${stations} stationer; ${far} af varslerne var falske.`,
			marginTitle: 'Er vi bedre end ingenting?',
			marginBeats: (horizon: number, points: string) =>
				`${horizon} minutter frem slår vi “antag at intet flytter sig” med ${points} CSI-point.`,
			marginBehind: (horizon: number, points: string) =>
				`${horizon} minutter frem ligger vi ${points} CSI-point under “antag at intet flytter sig”.`,
			marginTied: (horizon: number) =>
				`${horizon} minutter frem ligger vi lige med “antag at intet flytter sig”.`,
			marginDetail: (advection: string, persistence: string, frames: string) =>
				`CSI ${advection} mod ${persistence} · målt over ${frames} radarbilleder.`
		},
		reliability: {
			title: 'Når vi siger en sandsynlighed',
			intro: 'Hver prik samler de prognoser, der lovede omtrent det samme: vandret hvad vi sagde, lodret hvor ofte det så regnede. Ligger prikken på diagonalen, er 70 % 70 % værd. Ligger den under, lover vi for meget.',
			markerNote: 'Prikkens størrelse viser, hvor mange prognoser der ligger bag den.',
			radar: 'Radar',
			gauge: 'Regnmålere',
			perfect: 'Perfekt kalibrering',
			axisX: 'Sagt sandsynlighed',
			axisY: 'Det regnede',
			panel: (lead: number) => `${lead} minutter frem`,
			brierRadar: (value: string) => `Brier radar ${value}`,
			brierGauge: (value: string) => `Brier målere ${value}`,
			tableToggle: 'Vis tallene som tabel',
			tableCaption: (lead: number) => `Kalibrering ${lead} minutter frem`,
			colBin: 'Sagt',
			colRadar: 'Radar: det regnede',
			colRadarN: 'Radar: n',
			colGauge: 'Målere: det regnede',
			colGaugeN: 'Målere: n',
			empty: '—',
			none: 'Kalibreringen er ikke målt endnu.'
		},
		stations: {
			title: 'Målerne, én for én',
			intro: 'Hver prik er en DMI-station. Farven viser, hvor stor en del af regnen ved stationen vi nåede at varsle. Hvor der er for få varsler til det tal, er prikken farvet efter Brier-scoren i stedet og tegnet som en ring.',
			mapLabel: 'Kort over Danmark med DMI’s målestationer',
			legendTitle: 'Andel af regnen vi varslede',
			legendPoor: 'under 50 %',
			legendFair: '50–65 %',
			legendGood: '65–80 %',
			legendBest: 'over 80 %',
			legendUnknown: 'ingen score',
			legendBrier: 'Ring: farven kommer fra Brier-scoren, ikke fra andelen.',
			hint: 'Peg på en prik for at se stationens tal — de står også i listen nedenfor.',
			stationLabel: (name: string, kind: string) => `${name} (${kind})`,
			stationPod: (pod: string) => `Varslet regn: ${pod}`,
			stationFar: (far: string) => `falske varsler: ${far}`,
			stationBrier: (brier: string) => `Brier ${brier}`,
			stationEvents: (events: string, warnings: string) =>
				`${events} regnhændelser, ${warnings} varsler`,
			stationNoScore: 'For få varsler til en score',
			none: 'Der er endnu ingen stationer at vise.'
		},
		rainingNow: {
			title: '“Det regner her nu”',
			sentence: (agreement: string, pod: string, far: string) =>
				`Når vi melder om regn ved en måler, passer det ${agreement} af tiden: vi fanger ${pod} af de våde ti-minutter, og ${far} af vores våde meldinger var tørre på jorden.`,
			comparison: (observation: string) =>
				`Det rå radarbillede alene ville have ramt ${observation}.`,
			detail: (slots: string) => `Målt over ${slots} ti-minutters intervaller.`,
			none: 'Ikke målt endnu.'
		},
		events: {
			title: 'De seneste efterprøvede varsler',
			intro: 'De nyeste varsler, holdt op mod regnmåleren på stationen.',
			colStation: 'Station',
			colWarned: 'Varslet',
			colSaid: 'Vi sagde',
			colHappened: 'Hvad skete der',
			colError: 'Afvigelse',
			said: (min: number, probability: string) => `regn om ca. ${min} min (${probability})`,
			onset: (time: string) => `regn kl. ${time}`,
			noRain: 'ingen regn',
			early: (min: number) => `${min} min for tidligt`,
			late: (min: number) => `${min} min for sent`,
			onTime: 'ramte tiden',
			empty: '—',
			none: 'Der er endnu ingen efterprøvede varsler.'
		},
		methods: {
			title: 'Sådan er det målt',
			radarTitle: 'Radaren som sandhed',
			radarBody:
				'Radaren dækker hele landet og ser hver eneste byge, også dem der aldrig rammer en måler. Til gengæld ser den ikke sig selv: den måler den kraftigste ekko i søjlen over jorden, så virga, smeltelaget og støj fra bygninger og vindmøller tæller med som regn — og tærsklen for “regn” er vores egen.',
			gaugeTitle: 'Regnmålerne som sandhed',
			gaugeBody:
				'Regnmålerne måler det, man mærker: vand i en tragt på jorden. Til gengæld er de omkring hundrede punkter i hele landet, de tæller i ti-minutters kasser, de kan ikke se mindre end 0,1 mm, og i blæst fanger de mindre end der falder.',
			rulesTitle: 'Reglerne bag tallene',
			wetRule: 'Våd måler',
			onsetRule: 'Regnens start',
			threshold: 'Tærskel for regn',
			thresholdValue: (mmH: string) => `${mmH} mm/t`,
			frameAge: 'Radarbilledets alder',
			frameAgeValue: (min: string, max: string) => `${min}–${max} minutter`,
			subscriberRule: 'Hvornår et varsel sendes',
			subscriberRuleValue: (
				threshold: string,
				lead: number,
				persistence: number,
				rearm: number
			) =>
				`Sandsynlighed over ${threshold} inden for ${lead} minutter, bekræftet i ${persistence} cyklusser i træk, og tidligst et nyt varsel ${rearm} minutter senere.`,
			leadErrorNote:
				'En positiv afvigelse betyder, at regnen allerede var i gang, da varslet sagde den ville komme — varslet kom for sent.',
			windowsTitle: 'Perioder',
			radarWindow: (from: string, to: string, events: string, points: string) =>
				`Radar: ${from}–${to}, ${events} regnhændelser, ${points} sammenligninger.`,
			gaugeWindow: (from: string, to: string, events: string, stations: string) =>
				`Regnmålere: ${from}–${to}, ${events} hændelser på ${stations} stationer.`,
			liveWindow: (days: number, from: string, to: string) =>
				`Varsler: de seneste ${days} dage (${from}–${to}).`,
			cadence: 'Alle tal på siden beregnes forfra hver nat.',
			sourcesTitle: 'Kilder',
			sourceRadar: 'Radar',
			sourceGauges: 'Regnmålere',
			attribution: 'Radardata og observationer: DMI.'
		}
	},
	data: {
		title: 'Data og kilder',
		radarTitle: 'Radardata',
		radarBody:
			'Radarkompositterne kommer fra DMI Open Data og bruges under DMI’s vilkår for frie data, som kræver kildeangivelse. DMI har intet ansvar for de afledte produkter på denne side.',
		radarLink: 'DMI Open Data',
		basemapTitle: 'Baggrundskort',
		basemapBody:
			'Baggrundskortet er Protomaps-kortlag bygget på OpenStreetMap-data og hostet på vores egen server som én pmtiles-fil. Ingen tredjeparts kortserver er involveret.',
		basemapLinkOsm: '© OpenStreetMap-bidragydere',
		basemapLinkProtomaps: 'Protomaps',
		boundaryTitle: 'Landegrænser',
		boundaryBody:
			'Danmarks omrids, der bruges i kalibreringen, kommer fra Natural Earth og er public domain.',
		boundaryLink: 'Natural Earth',
		methodTitle: 'Metode og litteratur',
		methodBody:
			'Nowcastet bygger på optisk flow og Lagrangesk fremskrivning samt STEPS-ensemblet fra pysteps (Pulkkinen m.fl. 2019). Sammenligningsgrundlaget for forventet dygtighed er Imhoff m.fl. 2020 over hollandske lavlandsområder.'
	},
	privacy: {
		title: 'Privatliv',
		lead: 'Den korte version: siden indsamler ingenting om dig.',
		points: [
			'Ingen konti, ingen login, ingen nyhedsbreve.',
			'Ingen analyseværktøjer, ingen trackere, ingen annoncer.',
			'Ingen cookies. Det eneste, der gemmes i din browser, er dit valg af sprog (localStorage).',
			'Trykker du “brug min placering”, bliver koordinaterne brugt i din egen browser til at slå prognosen op i de kort, du allerede har hentet. De sendes ikke videre.',
			'Beskeder slår du selv til. Gør du det, gemmer serveren din browsers push-adresse, det punkt du valgte og dine indstillinger, så længe beskeden er slået til. De slettes, når du trykker “stop besked”, eller når push-tjenesten melder adressen død. Selve beskeden sendes gennem browserproducentens push-tjeneste (Apple, Google eller Mozilla).',
			'Serveren logger almindelige webforespørgsler for at kunne drives, og de bruges ikke til andet.'
		],
		contactTitle: 'Spørgsmål',
		contactBody: 'Spørgsmål om data kan stilles som et issue på GitHub.'
	},
	support: {
		title: 'Støt projektet',
		lead: 'Siden er gratis og uden reklamer. Den kører på en lille server derhjemme, og det er hele driftsbudgettet.',
		body: 'Der er ingen donationslinks endnu. Når de kommer, står de her — indtil da er den bedste støtte at bruge siden, melde fejl og fortælle andre om den.',
		soonTitle: 'På vej',
		soon: 'GitHub Sponsors og Ko-fi aktiveres senere.',
		thanks: 'Tak fordi du kigger med.'
	},
	error: {
		title: 'Siden findes ikke',
		body: 'Den side, du leder efter, findes ikke.',
		backToMap: 'Tilbage til kortet'
	}
} as const;
