-- Per-region weighted reliability per lead (Phase B, B3).
--
-- The regional view of reliability_pooled.sql — the input to the
-- regional-split criterion in scripts/national_calibration_report.py:
-- a region flags for splitting only when its curve leaves the pooled
-- curve's ±2·binomial-SE band in ≥ 2 probability bins (plan §B3;
-- SE formula documented in the report script).
--
-- Columns as in reliability_pooled.sql, plus `region` (the calibration
-- point's region tag from calibration_points_v2.json).
--
-- Parameter: {corpus} — replaced with the quoted corpus Parquet path
-- (see sql/README.md).

WITH valid AS (
    SELECT region, lead_min, raw_prob, outcome, sample_weight
    FROM read_parquet({corpus})
    WHERE isfinite(raw_prob)
      AND outcome IS NOT NULL
      AND isfinite(sample_weight) AND sample_weight > 0
),
binned AS (
    SELECT
        region,
        lead_min,
        LEAST(CAST(floor(raw_prob * 10) AS INTEGER), 9) AS bin,
        raw_prob,
        CAST(outcome AS DOUBLE) AS y,
        sample_weight AS w
    FROM valid
)
SELECT
    region,
    lead_min,
    bin,
    count(*)                                   AS n_rows,
    sum(w)                                     AS sum_w,
    (sum(w) * sum(w)) / sum(w * w)             AS eff_n,
    sum(w * raw_prob) / sum(w)                 AS mean_raw_weighted,
    sum(w * y) / sum(w)                        AS obs_freq_weighted
FROM binned
GROUP BY region, lead_min, bin
ORDER BY region, lead_min, bin;
