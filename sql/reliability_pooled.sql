-- Pooled weighted reliability per lead (Phase B, B3).
--
-- One row per (lead, probability bin): how often did it actually rain when
-- the national p_rain grid said "bin k"? Weighted by sample_weight — the
-- inverse-inclusion-probability weights that undo the corpus's wet-biased
-- event sampling (Finding 3) — so `obs_freq_weighted` estimates the true
-- climatological frequency, not the sampler's.
--
-- Columns:
--   lead_min            forecast lead
--   bin                 0..9; bin k covers raw_prob in [k/10, (k+1)/10),
--                       raw_prob = 1.0 folded into bin 9
--   n_rows              unweighted row count
--   sum_w               total sample weight in the bin
--   eff_n               Kish effective sample size (Σw)²/Σw²
--   mean_raw_weighted   weighted mean forecast probability in the bin
--   obs_freq_weighted   weighted observed rain frequency in the bin
--   obs_freq_unweighted unweighted frequency, for the bias-vs-weighting gap
--
-- Parameter: {corpus} — replaced with the quoted corpus Parquet path
-- (see sql/README.md).

WITH valid AS (
    SELECT lead_min, raw_prob, outcome, sample_weight
    FROM read_parquet({corpus})
    WHERE isfinite(raw_prob)
      AND outcome IS NOT NULL
      AND isfinite(sample_weight) AND sample_weight > 0
),
binned AS (
    SELECT
        lead_min,
        LEAST(CAST(floor(raw_prob * 10) AS INTEGER), 9) AS bin,
        raw_prob,
        CAST(outcome AS DOUBLE) AS y,
        sample_weight AS w
    FROM valid
)
SELECT
    lead_min,
    bin,
    count(*)                                   AS n_rows,
    sum(w)                                     AS sum_w,
    (sum(w) * sum(w)) / sum(w * w)             AS eff_n,
    sum(w * raw_prob) / sum(w)                 AS mean_raw_weighted,
    sum(w * y) / sum(w)                        AS obs_freq_weighted,
    avg(y)                                     AS obs_freq_unweighted
FROM binned
GROUP BY lead_min, bin
ORDER BY lead_min, bin;
