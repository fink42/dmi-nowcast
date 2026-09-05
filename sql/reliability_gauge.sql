-- Pooled weighted reliability per lead, against GAUGE truth (Phase F).
--
-- reliability_pooled.sql with one column swapped: the observed frequency
-- comes from `gauge_outcome` — a rain gauge at the point — instead of
-- `outcome`, which is the radar composite grading itself. Everything else
-- is deliberately identical (same valid-row filter, same weights, same
-- ten fixed bins), so the two files' outputs are directly comparable and
-- any difference is truth-source, not method.
--
-- Input: a corpus produced by scripts/join_gauge_truth.py — i.e. one
-- built over station points (scripts/build_station_points.py), because
-- only there does `point_id` name a station with a gauge in it.
--
-- Row validity here needs `gauge_outcome IS NOT NULL` rather than
-- `outcome IS NOT NULL`: the two are missing for different reasons (a
-- missing verification composite vs a missing gauge slot), and mixing the
-- filters would silently compare different row sets.
--
-- Columns:
--   lead_min            forecast lead
--   bin                 0..9; bin k covers raw_prob in [k/10, (k+1)/10),
--                       raw_prob = 1.0 folded into bin 9
--   n_rows              unweighted row count
--   sum_w               total sample weight in the bin
--   eff_n               Kish effective sample size (Σw)²/Σw²
--   mean_raw_weighted   weighted mean forecast probability in the bin
--   obs_freq_weighted   weighted observed GAUGE-wet frequency in the bin
--   obs_freq_unweighted unweighted frequency, for the bias-vs-weighting gap
--   mean_gauge_mm       weighted mean gauge accumulation in the 10-min slot
--
-- Parameter: {corpus} — replaced with the quoted corpus Parquet path
-- (see sql/README.md).

WITH valid AS (
    SELECT lead_min, raw_prob, gauge_outcome, gauge_mm, sample_weight
    FROM read_parquet({corpus})
    WHERE isfinite(raw_prob)
      AND gauge_outcome IS NOT NULL
      AND isfinite(sample_weight) AND sample_weight > 0
),
binned AS (
    SELECT
        lead_min,
        LEAST(CAST(floor(raw_prob * 10) AS INTEGER), 9) AS bin,
        raw_prob,
        CAST(gauge_outcome AS DOUBLE) AS y,
        gauge_mm,
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
    avg(y)                                     AS obs_freq_unweighted,
    sum(w * coalesce(gauge_mm, 0)) / sum(w)    AS mean_gauge_mm
FROM binned
GROUP BY lead_min, bin
ORDER BY lead_min, bin;
