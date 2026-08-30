-- Weighted Brier decomposition per lead (Murphy 1973) — Phase B, B3.
--
-- Over the 10 fixed probability bins (weighted bin statistics W_k = Σw,
-- p̄_k = weighted mean forecast, ō_k = weighted observed frequency,
-- ō = weighted overall base rate):
--
--     reliability = Σ_k W_k (p̄_k − ō_k)² / ΣW        (lower is better)
--     resolution  = Σ_k W_k (ō_k − ō)²  / ΣW          (higher is better)
--     uncertainty = ō (1 − ō)
--     Brier ≈ reliability − resolution + uncertainty  (binned approximation)
--
-- `brier_exact` is the direct weighted Brier Σw(p−y)²/Σw; it differs from
-- the decomposition sum by the within-bin variance term the binning
-- discards — a small positive `decomp_residual` is expected, a large one
-- means the bins are too coarse.
--
-- Isotonic calibration attacks the reliability term only: after a good
-- fit, `reliability` should collapse toward 0 while `resolution` and
-- `uncertainty` stay put.
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
),
bin_stats AS (
    SELECT
        lead_min,
        bin,
        sum(w)                     AS w_k,
        sum(w * raw_prob) / sum(w) AS p_k,
        sum(w * y) / sum(w)        AS o_k
    FROM binned
    GROUP BY lead_min, bin
),
lead_stats AS (
    SELECT
        lead_min,
        sum(w)                                 AS w_total,
        sum(w * y) / sum(w)                    AS o_bar,
        sum(w * (raw_prob - y) * (raw_prob - y)) / sum(w) AS brier_exact
    FROM binned
    GROUP BY lead_min
)
SELECT
    l.lead_min,
    l.brier_exact,
    sum(b.w_k * (b.p_k - b.o_k) * (b.p_k - b.o_k)) / l.w_total   AS reliability,
    sum(b.w_k * (b.o_k - l.o_bar) * (b.o_k - l.o_bar)) / l.w_total AS resolution,
    l.o_bar * (1 - l.o_bar)                                       AS uncertainty,
    l.o_bar                                                       AS base_rate_weighted,
    l.brier_exact
      - ( sum(b.w_k * (b.p_k - b.o_k) * (b.p_k - b.o_k)) / l.w_total
        - sum(b.w_k * (b.o_k - l.o_bar) * (b.o_k - l.o_bar)) / l.w_total
        + l.o_bar * (1 - l.o_bar) )                               AS decomp_residual
FROM bin_stats b
JOIN lead_stats l USING (lead_min)
GROUP BY l.lead_min, l.w_total, l.o_bar, l.brier_exact
ORDER BY l.lead_min;
