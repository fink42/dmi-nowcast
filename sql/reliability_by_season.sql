-- Per-season weighted reliability per lead (Phase B, B3 — diagnostic only).
--
-- Season is month-based meteorological: DJF / MAM / JJA / SON, taken from
-- the event_time ISO string. Phase B does NOT fit seasonal curves — this
-- is a sanity view against the Imhoff et al. 2020 expectation of roughly
-- 3× lower skill on summer convective vs winter stratiform (plan §B3).
-- Note the 180-day DMI archive depth: a single corpus covers at most two
-- seasons well, so expect empty/thin seasons in v1.
--
-- The month is parsed positionally from the ISO string
-- (substr(event_time, 6, 2): 'YYYY-MM-…') — deterministic, and avoids
-- TIMESTAMPTZ session-timezone semantics; event_time is always UTC with
-- a +00:00 offset (builder's _ts_str).
--
-- Parameter: {corpus} — replaced with the quoted corpus Parquet path
-- (see sql/README.md).

WITH valid AS (
    SELECT
        CAST(substr(event_time, 6, 2) AS INTEGER) AS month,
        lead_min, raw_prob, outcome, sample_weight
    FROM read_parquet({corpus})
    WHERE isfinite(raw_prob)
      AND outcome IS NOT NULL
      AND isfinite(sample_weight) AND sample_weight > 0
),
binned AS (
    SELECT
        CASE
            WHEN month IN (12, 1, 2) THEN 'DJF'
            WHEN month IN (3, 4, 5)  THEN 'MAM'
            WHEN month IN (6, 7, 8)  THEN 'JJA'
            ELSE 'SON'
        END AS season,
        lead_min,
        LEAST(CAST(floor(raw_prob * 10) AS INTEGER), 9) AS bin,
        raw_prob,
        CAST(outcome AS DOUBLE) AS y,
        sample_weight AS w
    FROM valid
)
SELECT
    season,
    lead_min,
    bin,
    count(*)                                   AS n_rows,
    sum(w)                                     AS sum_w,
    (sum(w) * sum(w)) / sum(w * w)             AS eff_n,
    sum(w * raw_prob) / sum(w)                 AS mean_raw_weighted,
    sum(w * y) / sum(w)                        AS obs_freq_weighted
FROM binned
GROUP BY season, lead_min, bin
ORDER BY season, lead_min, bin;
