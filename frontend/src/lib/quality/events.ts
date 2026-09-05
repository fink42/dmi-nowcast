/**
 * The latest verified warnings, as table rows.
 *
 * Everything the table shows is a string built here: the times converted from
 * UTC to the viewer's own clock (the one conversion the UTC-everywhere rule
 * allows, and it happens at this boundary and nowhere earlier), and the
 * outcome said in words rather than left as a code the reader has to decode.
 *
 * A false alarm has no onset and no error. Those cells are null, and the
 * table renders them as an em dash — not as a zero, which would read as "the
 * warning was perfectly timed".
 */
import { clockTime } from '$lib/format';
import type { Catalog, Locale } from '$lib/i18n';
import { localDateTime } from './dates';
import type { VerifiedEvent } from './schema';

export interface EventRow {
	/** Stable key for the `{#each}`; a station can appear more than once. */
	key: string;
	name: string;
	/** When the warning went out, local. */
	warnedAt: string;
	/** What it said: "rain in ~17 min", with the probability behind it. */
	said: string;
	/** What happened: the onset time, or "no rain". */
	happened: string;
	/** How far off the timing was, or null when there is nothing to compare. */
	error: string | null;
	hit: boolean;
}

export function eventRows(
	t: Catalog,
	locale: Locale,
	events: readonly VerifiedEvent[] | null
): EventRow[] {
	if (!events) return [];
	return events.map((event) => {
		const hit = event.outcome === 'hit';
		const minutes = event.lead_error_min;
		const error =
			minutes === null
				? null
				: Math.round(minutes) === 0
					? t.quality.events.onTime
					: minutes > 0
						? t.quality.events.late(Math.round(Math.abs(minutes)))
						: t.quality.events.early(Math.round(Math.abs(minutes)));
		return {
			key: `${event.station_id}-${event.warned_at_utc}`,
			name: event.name,
			warnedAt: localDateTime(event.warned_at_utc, locale),
			said: t.quality.events.said(
				Math.round(event.eta_min),
				t.quality.percent(Math.round(event.p_rain * 100))
			),
			happened:
				hit && event.gauge_onset_utc !== null
					? t.quality.events.onset(clockTime(event.gauge_onset_utc, locale))
					: t.quality.events.noRain,
			error,
			hit
		};
	});
}
