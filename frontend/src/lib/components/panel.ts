/**
 * The one piece of the forecast panel's fold-down behaviour worth testing on
 * its own: when a newly selected point should unfold a panel the user had
 * minimised.
 */

/** Just the identity of a selected point — where it is. */
export interface PanelPoint {
	lat: number;
	lon: number;
}

/**
 * Does the move from `prev` to `next` mean the panel should be unfolded?
 *
 * Only a *different* place counts. The nowcast store replaces the whole
 * `point` object every cycle when it re-samples the same coordinates, so
 * object identity would pop a minimised panel open every five minutes — the
 * coordinates are what the user actually changed. A first point after none
 * (a fresh tap, or one arriving from a notification) also unfolds: the user
 * asked a question and the answer should not start hidden.
 */
export function shouldExpandOnPointChange(
	prev: PanelPoint | null,
	next: PanelPoint | null
): boolean {
	if (!next) return false;
	if (!prev) return true;
	return prev.lat !== next.lat || prev.lon !== next.lon;
}
