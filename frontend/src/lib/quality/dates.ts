/**
 * UTC → the viewer's clock, for the quality page.
 *
 * Everything in `quality.json` is UTC, as everything internal is; these two
 * functions are the boundary where that stops being true. Both fall back to
 * `clockTime` (and it to the raw string) rather than throwing on a timestamp
 * that will not parse — a page about honesty should not blank a whole table
 * over one bad date.
 */
import { clockTime } from '$lib/format';
import type { Locale } from '$lib/i18n';

const tag = (locale: Locale): string => (locale === 'da' ? 'da-DK' : 'en-GB');

/** "1. mar. 2026" / "1 Mar 2026" — a window edge, where the time is noise. */
export function localDate(isoUtc: string, locale: Locale): string {
	try {
		return new Intl.DateTimeFormat(tag(locale), {
			day: 'numeric',
			month: 'short',
			year: 'numeric'
		}).format(new Date(isoUtc));
	} catch {
		return isoUtc.slice(0, 10);
	}
}

/**
 * "4. sep. 20.35" / "4 Sep 20:35" — an event. The clock time alone would put
 * two different afternoons on the same line.
 */
export function localDateTime(isoUtc: string, locale: Locale): string {
	try {
		return new Intl.DateTimeFormat(tag(locale), {
			day: 'numeric',
			month: 'short',
			hour: '2-digit',
			minute: '2-digit'
		}).format(new Date(isoUtc));
	} catch {
		return clockTime(isoUtc, locale);
	}
}
