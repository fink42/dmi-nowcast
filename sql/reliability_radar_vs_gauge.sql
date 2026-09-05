-- Radar truth vs gauge truth, side by side, per lead × probability bin
-- (Phase F).
--
-- The question this answers: when the service says "60% chance of rain in
-- 30 minutes", how often did the RADAR call it rain, and how often did the
-- GAUGE? A gap between the two columns is not a calibration error — it is
-- the truth definition disagreeing with itself, and it bounds how much of
-- the fitted calibration is really radar self-agreement.
--
-- Expect the radar column to run HIGH relative to the gauge: DMI's
-- composite is column-max reflectivity, so tall cores, virga and the
-- melting-layer bright band all read as rain aloft that never reaches the
-- gauge at ground level.
--
-- Row set: rows where BOTH truths exist, so every bin's two frequencies
-- are computed over the identical rows and the comparison is paired.
-- `n_radar_only` / `n_gauge_only` report what that pairing costs, per
-- lead — a large asymmetry there means the comparison is over a
-- non-representative subset and should be read with care.
--
-- Input: a corpus produced by scripts/join_gauge_truth.py.
--
-- Columns:
--   lead_min           forecast lead
--   bin                0..9; bin k covers raw_prob in [k/10, (k+1)/10)
--   n_rows             unweighted paired-row count in the bin
--   sum_w              total sample weight in the bin
--   eff_n              Kish effective sample size (Σw)²/Σw²
--   mean_raw_weighted  weighted mean forecast probability
--   obs_freq_radar     weighted observed frequency, radar truth
--   obs_freq_gauge     weighted observed frequency, gauge truth
--   gauge_minus_radar  the gap, gauge − radar (negative = radar over-reads)
--   n_radar_only       rows in the bin with radar truth but no gauge slot
--   n_gauge_only       rows in the bin with a gauge slot but no radar frame
--
-- Parameter: {corpus} — replaced with the quoted corpus Parquet path
-- (see sql/README.md).

WITH scored AS (
    SELECT
        lead_min,
        LEAST(CAST(floor(raw_prob * 10) AS INTEGER), 9) AS bin,
        raw_prob,
        outcome,
        gauge_outcome,
        sample_weight AS w
    FROM read_parquet({corpus})
    WHERE isfinite(raw_prob)
      AND isfinite(sample_weight) AND sample_weight > 0
      AND (outcome IS NOT NULL OR gauge_outcome IS NOT NULL)
),
paired AS (
    SELECT
        lead_min, bin, raw_prob, w,
        CAST(outcome AS DOUBLE)       AS y_radar,
        CAST(gauge_outcome AS DOUBLE) AS y_gauge
    FROM scored
    WHERE outcome IS NOT NULL AND gauge_outcome IS NOT NULL
),
unpaired AS (
    SELECT
        lead_min, bin,
        count(*) FILTER (WHERE gauge_outcome IS NULL) AS n_radar_only,
        count(*) FILTER (WHERE outcome IS NULL)       AS n_gauge_only
    FROM scored
    GROUP BY lead_min, bin
),
agg AS (
    SELECT
        lead_min,
        bin,
        count(*)                        AS n_rows,
        sum(w)                          AS sum_w,
        (sum(w) * sum(w)) / sum(w * w)  AS eff_n,
        sum(w * raw_prob) / sum(w)      AS mean_raw_weighted,
        sum(w * y_radar) / sum(w)       AS obs_freq_radar,
        sum(w * y_gauge) / sum(w)       AS obs_freq_gauge
    FROM paired
    GROUP BY lead_min, bin
)
SELECT
    agg.lead_min,
    agg.bin,
    agg.n_rows,
    agg.sum_w,
    agg.eff_n,
    agg.mean_raw_weighted,
    agg.obs_freq_radar,
    agg.obs_freq_gauge,
    agg.obs_freq_gauge - agg.obs_freq_radar AS gauge_minus_radar,
    coalesce(unpaired.n_radar_only, 0)      AS n_radar_only,
    coalesce(unpaired.n_gauge_only, 0)      AS n_gauge_only
FROM agg
LEFT JOIN unpaired USING (lead_min, bin)
ORDER BY agg.lead_min, agg.bin;
