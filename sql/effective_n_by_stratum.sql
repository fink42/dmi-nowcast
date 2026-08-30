-- Kish effective sample size per stratum (Phase B, B3).
--
-- Design weights cost information: eff_n = (Σw)²/Σw² ≤ n, with equality
-- only for equal weights. The plan requires reports to state effective
-- sample sizes (§2 "Weighted fits only; reports state effective sample
-- sizes") — this query gives them per lead, per region × lead, and per
-- season × lead, so thin strata are visible before anyone trusts a
-- regional or seasonal curve.
--
-- Columns:
--   stratum_kind   'lead' | 'region' | 'season'
--   stratum        the region/season name ('(all)' for the pooled rows)
--   lead_min       forecast lead
--   n_rows         unweighted row count
--   sum_w          total weight
--   eff_n          Kish effective sample size
--   eff_ratio      eff_n / n_rows (1.0 = unweighted-equivalent)
--
-- Parameter: {corpus} — replaced with the quoted corpus Parquet path
-- (see sql/README.md).

WITH valid AS (
    SELECT
        region,
        CASE
            WHEN CAST(substr(event_time, 6, 2) AS INTEGER) IN (12, 1, 2) THEN 'DJF'
            WHEN CAST(substr(event_time, 6, 2) AS INTEGER) IN (3, 4, 5)  THEN 'MAM'
            WHEN CAST(substr(event_time, 6, 2) AS INTEGER) IN (6, 7, 8)  THEN 'JJA'
            ELSE 'SON'
        END AS season,
        lead_min,
        sample_weight AS w
    FROM read_parquet({corpus})
    WHERE isfinite(raw_prob)
      AND outcome IS NOT NULL
      AND isfinite(sample_weight) AND sample_weight > 0
)
SELECT 'lead' AS stratum_kind, '(all)' AS stratum, lead_min,
       count(*) AS n_rows, sum(w) AS sum_w,
       (sum(w) * sum(w)) / sum(w * w) AS eff_n,
       ((sum(w) * sum(w)) / sum(w * w)) / count(*) AS eff_ratio
FROM valid
GROUP BY lead_min

UNION ALL

SELECT 'region', region, lead_min,
       count(*), sum(w),
       (sum(w) * sum(w)) / sum(w * w),
       ((sum(w) * sum(w)) / sum(w * w)) / count(*)
FROM valid
GROUP BY region, lead_min

UNION ALL

SELECT 'season', season, lead_min,
       count(*), sum(w),
       (sum(w) * sum(w)) / sum(w * w),
       ((sum(w) * sum(w)) / sum(w * w)) / count(*)
FROM valid
GROUP BY season, lead_min

ORDER BY stratum_kind, stratum, lead_min;
