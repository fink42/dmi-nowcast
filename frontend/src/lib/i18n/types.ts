import type { da } from './da';

/**
 * Widen the literal types `as const` gives the Danish catalog, so another
 * locale can hold *different* strings while still being forced to hold
 * *every* string. Functions keep their exact signature — a translation
 * cannot quietly drop the argument it is supposed to interpolate.
 */
type Widen<T> = T extends string
	? string
	: T extends (...args: infer A) => infer R
		? (...args: A) => R
		: T extends readonly (infer E)[]
			? readonly Widen<E>[]
			: { -readonly [K in keyof T]: Widen<T[K]> };

/** The shape every locale file must satisfy. */
export type Catalog = Widen<typeof da>;

export const LOCALES = ['da', 'en'] as const;
export type Locale = (typeof LOCALES)[number];

/** Default locale: this is a Danish site first. */
export const DEFAULT_LOCALE: Locale = 'da';

export const isLocale = (value: unknown): value is Locale =>
	typeof value === 'string' && (LOCALES as readonly string[]).includes(value);
