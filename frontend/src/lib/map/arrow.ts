/**
 * The "rain comes from over there" arrow, as map geometry.
 *
 * This is the map-drawn twin of the Home Assistant card's server-rendered
 * arrow (`_draw_motion_arrow` in `src/dmi_nowcast_core/render.py`), with one
 * deliberate difference: **this arrow is measured from wall-clock now, not
 * from the radar image's timestamp.**
 *
 * Why that matters. The newest DMI composite is typically 20–30 minutes old by
 * the time it is on screen (scan cadence, DMI's publication delay, the
 * sidecar's own poll). An arrow that extrapolates 60 minutes from the *image*
 * puts its tip at `image_time + 60`, so somebody reading the map at 14:33 over
 * a 14:10 image sees a tip they will read as 15:33 when it is really 15:10.
 * The arrow sits on a map next to a clock-shaped question, so the clock wins:
 *
 *   - it is anchored at the selected point and points **upstream**, along the
 *     bearing the cell comes from, so it lies over the rain that is on its way;
 *   - a feature marked `minute = m` sits over the rain that reaches the point
 *     `m` minutes from **now**, which — the image being `age` minutes old — is
 *     `speedKmh × (age + m) / 60` km upstream on that image;
 *   - the tip is therefore the *now + 60 min* mark, and the shaft is
 *     `speedKmh × (age + 60) / 60` long. At the tip that is up to ~85 minutes
 *     of extrapolation from the last observation, which is the accepted price
 *     of an arrow whose marks mean what a person reading a clock takes them to
 *     mean;
 *   - the stretch between the tail and the **now tick** is not a forecast at
 *     all: it is rain the image caught upstream that has already arrived, or is
 *     arriving, by wall-clock now. The now tick (`role: 'now'`, `minute: 0`)
 *     sits `speedKmh × age / 60` from the tail, double the width of an ordinary
 *     tick so it reads as "you are here, in time". It is suppressed below a
 *     minute of age, where it would hide under the marker anyway;
 *   - the shaft is ticked at every radar timestep (the manifest's
 *     `timestep_min`: 10 min → ticks at 10, 20, 30, 40, 50), each tick marking
 *     where the rain that arrives in that many minutes from now is on this
 *     image. The tip is the 60 min mark and carries the head instead of a tick.
 *
 * Age is clamped to `ARROW_AGE_CAP_MIN`. A pipeline stalled for hours would
 * otherwise draw an arrow across the North Sea out of an image whose motion
 * says nothing about the present; the cap keeps a broken feed from turning
 * into a confident-looking line. That a feed *is* stale is reported honestly
 * elsewhere (`nowcast/freshness.ts`), which is where that claim belongs.
 *
 * Two things the card does that this deliberately does not:
 *
 *   - **No minimum speed.** The card refuses to draw below ~20 km/h because a
 *     short arrow disappears under the home dot in a fixed-size image. The map
 *     zooms, and — more importantly — the arrow must never disagree with the
 *     forecast panel: both are drawn from the same `cellMotion()` result, whose
 *     null is the *only* "no arrow" rule (see nowcast/motion.ts). A second
 *     threshold here would produce a panel that says "from NW at 3 km/h" over a
 *     map that shows nothing.
 *   - **No minute labels.** Pixel-space text does not belong in a GeoJSON
 *     source, and the panel already spells the motion out in words.
 *
 * Everything below is pure geometry with no MapLibre import, so it is testable
 * without a canvas. MapView.svelte owns the source, the layers and the paint.
 */
import type { Feature, FeatureCollection, LineString, Polygon, Position } from 'geojson';

/** Mean Earth radius (WGS84 mean), km. */
const EARTH_RADIUS_KM = 6371.0088;

/** Kilometres per degree of latitude on that sphere. */
const KM_PER_DEG_LAT = (Math.PI * EARTH_RADIUS_KM) / 180;

/**
 * How far the arrow reaches, in minutes of travel *from now*. Sixty, because
 * "where is the rain that hits me in an hour" is the question the arrow
 * answers.
 */
export const ARROW_HORIZON_MIN = 60;

/**
 * The most radar age the arrow will absorb, in minutes. A healthy feed's age
 * peaks around 27–28 min (see `nowcast/freshness.ts`), so sixty is well clear
 * of normal while still refusing to extrapolate an hours-old image out to the
 * width of a country.
 */
export const ARROW_AGE_CAP_MIN = 60;

/** Arrowhead length as a fraction of the shaft, then clamped, in km. */
const HEAD_FRACTION = 0.12;
const HEAD_MIN_KM = 1.5;
const HEAD_MAX_KM = 8;
/** Half-angle of the head, matching the card's 25°. */
const HEAD_HALF_ANGLE_DEG = 25;
/**
 * Tick half-length as a fraction of the head length. The card draws ticks
 * whose full width equals the head length (`tick_half = head_size / 2`), and
 * that ratio is what makes the shaft read as a ruler rather than a comb.
 */
const TICK_HALF_PER_HEAD = 0.5;
/** The now tick is twice an ordinary tick — the one mark that is not a forecast. */
const NOW_TICK_SCALE = 2;

/** What a feature of the arrow is, for styling and for tests. */
export type ArrowRole = 'shaft' | 'head' | 'tick' | 'now';

/**
 * A type alias rather than an interface on purpose: GeoJSON's `properties`
 * slot wants an implicit index signature, which interfaces do not get.
 */
export type ArrowProperties = {
	role: ArrowRole;
	/**
	 * Minutes from wall-clock now that this feature marks. Shaft and head carry
	 * the horizon; the now tick carries zero.
	 */
	minute: number;
};

export type ArrowFeature = Feature<LineString | Polygon, ArrowProperties>;
export type ArrowCollection = FeatureCollection<LineString | Polygon, ArrowProperties>;

/** The "draw nothing" data — what the source holds with no point selected. */
export const emptyArrow = (): ArrowCollection => ({ type: 'FeatureCollection', features: [] });

/**
 * The point `distanceKm` away from (lat, lon) along a constant bearing,
 * returned as GeoJSON `[lon, lat]`.
 *
 * Equirectangular: latitude moves by a fixed number of km per degree, and
 * longitude is scaled by the cosine of the *mid*-latitude of the step rather
 * than the start latitude, which is what keeps the second-order term small.
 *
 * A cell advected at a constant bearing follows a rhumb line, not a great
 * circle, so the rhumb destination is the thing to be accurate against.
 * Measured over Denmark (lat 54–58.4°, all bearings) this agrees with the
 * exact rhumb formula to **0.5 m at 60 km and 4.4 m at 120 km** — and the
 * endpoint's great-circle range is within 6 m of the requested distance at
 * 120 km. Both are three orders of magnitude under one 500 m radar pixel, and
 * 120 km is already a 120 km/h cell: past anything Denmark produces.
 */
export function destinationPoint(
	lat: number,
	lon: number,
	bearingDeg: number,
	distanceKm: number
): Position {
	const theta = (bearingDeg * Math.PI) / 180;
	const dLat = (distanceKm * Math.cos(theta)) / KM_PER_DEG_LAT;
	const lat2 = lat + dLat;
	const cosMid = Math.cos((((lat + lat2) / 2) * Math.PI) / 180);
	// Only reachable at the poles, where east/west has no meaning anyway.
	const dLon = Math.abs(cosMid) < 1e-12 ? 0 : (distanceKm * Math.sin(theta)) / (KM_PER_DEG_LAT * cosMid);
	return [lon + dLon, lat2];
}

export interface MotionArrowInput {
	/** The selected point — the arrow's tail. */
	lat: number;
	lon: number;
	/** Bearing the cell comes from, degrees clockwise from north. */
	bearingFromDeg: number;
	/** Speed over the ground, km/h. */
	speedKmh: number;
	/** Radar cadence from the manifest; the spacing of the shaft's ticks. */
	timestepMin: number;
	/**
	 * How old the radar image being drawn is *at the moment of drawing*, in
	 * minutes — `nowcast.radarAgeMin`, the same number the panel prints. It
	 * pushes every mark further upstream by that much travel, which is what
	 * keeps `minute` on the wall clock. Anything non-finite or negative counts
	 * as zero; the rest is clamped to `ARROW_AGE_CAP_MIN`.
	 */
	radarAgeMin: number;
}

const clamp = (value: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, value));

/** Radar age as the geometry may use it: finite, non-negative, capped. */
const usableAge = (radarAgeMin: number): number =>
	Number.isFinite(radarAgeMin) ? clamp(radarAgeMin, 0, ARROW_AGE_CAP_MIN) : 0;

/**
 * Build the arrow for one sampled motion: shaft, arrowhead, one tick per radar
 * timestep short of the tip, and the now tick once the image has aged.
 *
 * The caller decides *whether* there is an arrow — `cellMotion()` returning
 * null means there is not — so this assumes a real motion. It still refuses a
 * non-positive or non-finite speed (there is no such arrow to draw) and a
 * timestep under a minute (which is not a radar cadence, and would be a very
 * long loop), returning an empty collection and no ticks respectively.
 */
export function motionArrow(input: MotionArrowInput): ArrowCollection {
	const { lat, lon, bearingFromDeg, speedKmh, timestepMin, radarAgeMin } = input;
	if (!Number.isFinite(speedKmh) || speedKmh <= 0) return emptyArrow();
	if (!Number.isFinite(lat) || !Number.isFinite(lon) || !Number.isFinite(bearingFromDeg)) {
		return emptyArrow();
	}

	const ageMin = usableAge(radarAgeMin);
	/**
	 * Distance up the shaft, from the tail, of the rain that arrives `minute`
	 * minutes from now: counted from when the image was taken it still has
	 * `age + minute` minutes of travelling to do.
	 */
	const kmAt = (minute: number): number => (speedKmh * (ageMin + minute)) / 60;

	const lengthKm = kmAt(ARROW_HORIZON_MIN);
	const tail: Position = [lon, lat];
	const tip = destinationPoint(lat, lon, bearingFromDeg, lengthKm);

	const features: ArrowFeature[] = [
		{
			type: 'Feature',
			properties: { role: 'shaft', minute: ARROW_HORIZON_MIN },
			geometry: { type: 'LineString', coordinates: [tail, tip] }
		}
	];

	// Head and tick sizes stay proportional to the shaft actually drawn, so an
	// aged (longer) arrow keeps the card's ruler look instead of growing a comb
	// of hairlines. Both are clamped, so age cannot inflate them without bound.
	const headKm = clamp(lengthKm * HEAD_FRACTION, HEAD_MIN_KM, HEAD_MAX_KM);
	const tickHalfKm = headKm * TICK_HALF_PER_HEAD;

	/** One cross-piece, centred `distanceKm` up the shaft. */
	const crossPiece = (distanceKm: number, halfKm: number, properties: ArrowProperties) => {
		const [cLon, cLat] = destinationPoint(lat, lon, bearingFromDeg, distanceKm);
		features.push({
			type: 'Feature',
			properties,
			geometry: {
				type: 'LineString',
				coordinates: [
					destinationPoint(cLat, cLon, bearingFromDeg - 90, halfKm),
					destinationPoint(cLat, cLon, bearingFromDeg + 90, halfKm)
				]
			}
		});
	};

	// Wall-clock now. Under a minute of age it would sit on the tail, under the
	// marker, where a double-width tick is only a smudge — so it waits until
	// there is something to point out.
	if (ageMin >= 1) {
		crossPiece(kmAt(0), tickHalfKm * NOW_TICK_SCALE, { role: 'now', minute: 0 });
	}

	// Ticks, from the tail outwards, at every whole timestep before the tip.
	// Indexed rather than accumulated so the spacing cannot drift.
	if (Number.isFinite(timestepMin) && timestepMin >= 1) {
		for (let i = 1; i * timestepMin < ARROW_HORIZON_MIN; i++) {
			const minute = i * timestepMin;
			crossPiece(kmAt(minute), tickHalfKm, { role: 'tick', minute });
		}
	}

	// Head last so it paints over the shaft it caps. Apex at the tip, base
	// corners swept back ±25° from the reversed direction — the card's shape.
	const back = bearingFromDeg + 180;
	features.push({
		type: 'Feature',
		properties: { role: 'head', minute: ARROW_HORIZON_MIN },
		geometry: {
			type: 'Polygon',
			coordinates: [
				[
					tip,
					destinationPoint(tip[1], tip[0], back - HEAD_HALF_ANGLE_DEG, headKm),
					destinationPoint(tip[1], tip[0], back + HEAD_HALF_ANGLE_DEG, headKm),
					tip
				]
			]
		}
	});

	return { type: 'FeatureCollection', features };
}
