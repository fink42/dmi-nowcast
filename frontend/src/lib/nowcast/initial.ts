/**
 * Which point the map page opens on.
 *
 * Two things can name a point before the user touches the map: a deep link
 * (`/?lat=&lon=` from a notification, or the service worker's `open-point`
 * message) and the subscription this browser holds. A subscription is the
 * strongest statement of "the place I care about" the site has — it is the
 * point the pushes are for — so, with no deep link, the page opens there
 * instead of on an empty map that has to be clicked every visit.
 *
 * Nothing here reads a sensor: geolocation stays behind its button, because
 * a permission prompt on every open is exactly what the subscription saves
 * people from.
 */
import { pointFromUrl } from '$lib/push/notification';
import type { StoredSubscription } from '$lib/push/prefs';

export interface InitialPoint {
	lat: number;
	lon: number;
	/** Where the point came from, so the page can treat a link as a link. */
	source: 'deep-link' | 'subscription';
}

/**
 * The point to open on: the deep link if there is one, else the subscribed
 * point, else nothing (the map stays on Denmark).
 */
export function initialPoint(
	search: string,
	stored: Pick<StoredSubscription, 'lat' | 'lon'> | null
): InitialPoint | null {
	const link = pointFromUrl(search);
	if (link) return { lat: link.lat, lon: link.lon, source: 'deep-link' };
	if (stored && Number.isFinite(stored.lat) && Number.isFinite(stored.lon)) {
		return { lat: stored.lat, lon: stored.lon, source: 'subscription' };
	}
	return null;
}
