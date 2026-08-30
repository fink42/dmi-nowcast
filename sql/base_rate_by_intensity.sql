-- Base rates per event intensity band (Phase B, B3).
--
-- The corpus carries no mm/h field (raw_prob is an exceedance
-- probability, outcome a 0/1 threshold verdict), so "intensity" uses the
-- available spatial proxy: **event wet coverage** — the weighted share of
-- calibration points wet at the event's shortest lead. Widespread
-- stratiform events land in high-coverage bands; isolated convection in
-- low ones. Weights are constant within an event (they are event-level
-- inclusion weights), so the coverage measure is effectively the plain
-- share of wet points.
--
-- Columns:
--   intensity_band    dry / isolated / scattered / widespread / extensive
--   lead_min          forecast lead
--   n_events          distinct events in the band
--   n_rows            valid (point, lead) rows in the band
--   base_rate_weighted   weighted observed rain frequency
--   mean_raw_weighted    weighted mean forecast probability
--
-- Parameter: {corpus} — replaced with the quoted corpus Parquet path
-- (see sql/README.md).

WITH valid AS (
    SELECT event_time, lead_min, raw_prob, outcome, sample_weight
    FROM read_parquet({corpus})
    WHERE isfinite(raw_prob)
      AND outcome IS NOT NULL
      AND isfinite(sample_weight) AND sample_weight > 0
),
event_coverage AS (
    -- Wet coverage at the shortest lead = "how much of Denmark was about
    -- to be wet during this event".
    SELECT
        event_time,
        sum(sample_weight * outcome) / sum(sample_weight) AS wet_coverage
    FROM valid
    WHERE lead_min = (SELECT min(lead_min) FROM valid)
    GROUP BY event_time
),
banded AS (
    SELECT
        event_time,
        CASE
            WHEN wet_coverage = 0    THEN '0: dry (0%)'
            WHEN wet_coverage < 0.05 THEN '1: isolated (<5%)'
            WHEN wet_coverage < 0.20 THEN '2: scattered (5-20%)'
            WHEN wet_coverage < 0.50 THEN '3: widespread (20-50%)'
            ELSE                          '4: extensive (>=50%)'
        END AS intensity_band
    FROM event_coverage
)
SELECT
    b.intensity_band,
    v.lead_min,
    count(DISTINCT v.event_time)                              AS n_events,
    count(*)                                                  AS n_rows,
    sum(v.sample_weight * v.outcome) / sum(v.sample_weight)   AS base_rate_weighted,
    sum(v.sample_weight * v.raw_prob) / sum(v.sample_weight)  AS mean_raw_weighted
FROM valid v
JOIN banded b USING (event_time)
GROUP BY b.intensity_band, v.lead_min
ORDER BY b.intensity_band, v.lead_min;
