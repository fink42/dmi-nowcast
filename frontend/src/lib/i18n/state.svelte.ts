/**
 * The reactive half of i18n: which locale is active, how it is persisted, and
 * the `t()` accessor components read.
 *
 * Danish is the default; English is a toggle. The choice is kept in
 * localStorage, which can throw (private mode, storage disabled, iOS quirks),
 * so every access is guarded — a browser that refuses storage still gets a
 * working site, it just forgets the choice between visits.
 */
import { browser } from '$app/environment';
import { da } from './da';
import { en } from './en';
import { DEFAULT_LOCALE, isLocale, type Catalog, type Locale } from './types';

const STORAGE_KEY = 'dmi-nowcast.locale';

const catalogs: Record<Locale, Catalog> = { da, en };

/**
 * The stored choice, else Danish. A Denmark-only site opens in Danish and the
 * English toggle is one tap away, remembered from then on. Browser language
 * is deliberately not sniffed: plenty of phones in Denmark are set to
 * English, and the Danish text is the one written first.
 */
function initialLocale(): Locale {
	if (!browser) return DEFAULT_LOCALE;
	try {
		const stored = localStorage.getItem(STORAGE_KEY);
		if (isLocale(stored)) return stored;
	} catch {
		/* storage unavailable — Danish it is */
	}
	return DEFAULT_LOCALE;
}

const state = $state<{ locale: Locale }>({ locale: DEFAULT_LOCALE });

/** Called once from the root layout, after hydration. */
export function initLocale(): void {
	setLocale(initialLocale());
}

export function setLocale(next: Locale): void {
	state.locale = next;
	if (!browser) return;
	try {
		localStorage.setItem(STORAGE_KEY, next);
	} catch {
		/* not persisting is survivable */
	}
	document.documentElement.lang = catalogs[next].htmlLang;
}

/** The active locale (reactive). */
export function locale(): Locale {
	return state.locale;
}

/** The active catalog (reactive) — `t().panel.etaLabel`. */
export function t(): Catalog {
	return catalogs[state.locale];
}

export { catalogs };
export type { Catalog, Locale };
