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
		now: 'Nu',
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
		loading: 'Henter prognose …',
		headlineRainingNow: 'Det regner her nu',
		headlineEta: (min: number) => `Regn om ca. ${min} min`,
		headlineNoRain: 'Ingen regn forventet den næste time',
		headlineUnknown: 'Ingen prognose for dette punkt',
		etaLabel: 'Forventet ankomst',
		etaValue: (min: number) => `om ca. ${min} min`,
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
		sourceTitle: 'Kildekode',
		sourceBody: 'Alt — algoritme, tjeneste og denne side — ligger frit tilgængeligt på GitHub.',
		sourceLink: 'github.com/nsimonsen/dmi-nowcast'
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
