/**
 * i18n entry point. Danish and English, two hand-written catalogs and about
 * thirty lines of state — no framework, because the whole feature is "pick
 * one of two objects and remember which".
 */
export { da } from './da';
export { en } from './en';
export { catalogs, initLocale, locale, setLocale, t } from './state.svelte';
export { DEFAULT_LOCALE, isLocale, LOCALES, type Catalog, type Locale } from './types';
