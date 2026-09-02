# DuckDB analysis queries over the calibration corpus (Phase B, B3)

Analytical queries over the v2 multi-point corpus Parquet
(`scripts/build_calibration_corpus.py` output; one row per
event × point × lead). Group-by-heavy reliability/verification work is
exactly what DuckDB is for (`public_site_idea.md` §8) — the queries live
here as SQL instead of bespoke pandas.

## Conventions

- **Parameterisation**: every query contains the literal placeholder
  `{corpus}`, replaced with the **quoted** Parquet path before execution.
  `scripts/national_calibration_report.py` does this substitution; by
  hand:

      sed "s|{corpus}|'reports/calibration_corpus.parquet'|" \
          sql/reliability_pooled.sql | duckdb

- **Valid rows**: each query filters to fit-usable rows — finite
  `raw_prob` (NaN = failed forecast / out-of-grid point), non-null
  `outcome` (null = missing verification frame), finite positive
  `sample_weight` — the same filter `scripts/fit_national_calibration.py`
  applies.
- **Weights everywhere** (Finding 3): frequencies and Brier terms use the
  `sample_weight` column (unnormalised inverse inclusion probability of
  the wet-biased event sampler). Unweighted counts are reported alongside
  as `n_rows`; `eff_n` is the Kish effective sample size `(Σw)²/Σw²`.
- **Probability bins**: 10 fixed-width bins,
  `bin = least(floor(raw_prob*10), 9)` — bin k covers
  `[k/10, (k+1)/10)`, with `raw_prob = 1.0` folded into bin 9.
- **Frame age**: every row carries `frame_age_min`, the event's simulated
  live compute latency (drawn per event from `frame_age_range_csv`,
  default 12–18 min). `lead_min` stays the NOMINAL lead — the one the
  service quotes and the fit groups by — while the row's outcome was
  verified at `T + ceil((lead_min + frame_age_min)/timestep_min) *
  timestep_min`, exactly the instant the served probability describes. A
  corpus without the column was built at zero age and its leads mean
  something else; don't pool the two (`settings_hash` differs, and both
  the builder and the fitter refuse the mix).

## Queries

| file | question |
|---|---|
| `reliability_pooled.sql` | pooled weighted reliability per lead (the fit's own view of the data) |
| `reliability_by_region.sql` | the same per region — input to the regional-split criterion |
| `reliability_by_season.sql` | the same per season (month-based DJF/MAM/JJA/SON; diagnostic only in Phase B) |
| `base_rate_by_intensity.sql` | base rates per event intensity band (spatial wet-coverage proxy) |
| `brier_decomposition.sql` | weighted Murphy decomposition: reliability − resolution + uncertainty per lead |
| `effective_n_by_stratum.sql` | Kish effective N per lead / region / season stratum |
