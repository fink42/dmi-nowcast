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
| `reliability_gauge.sql` | pooled weighted reliability per lead against **gauge** truth (`gauge_outcome`) |
| `reliability_radar_vs_gauge.sql` | radar vs gauge observed frequency, paired, per lead × bin |

## Gauge truth (Phase F)

The last two queries read a corpus that
`scripts/join_gauge_truth.py` has widened with three columns —
`gauge_mm`, `gauge_dur_min`, `gauge_outcome` — taken from DMI's metObs
rain gauges at the corpus's own points. They only mean anything on a
corpus built over **station** points
(`scripts/build_station_points.py`): a `point_id` has to be a
`stationId` for the join to find a gauge.

- `gauge_outcome` is 1 when the gauge slot at the row's verification
  instant recorded `precip_past10min >= 0.1 mm` OR
  `precip_dur_past10min >= 1 min`, 0 when it recorded neither, and NULL
  when the amount slot is missing. **Filter on it, not on `outcome`** —
  the two are missing for different reasons (no gauge slot vs no
  verification composite).
- The verification instant is the builder's, unchanged:
  `T + ceil((lead_min + frame_age_min)/timestep_min - 1e-9) *
  timestep_min`, matched to the 10-minute gauge stamp *ending* at it.
- Gauge data is DMI Open Data, licence **CC BY 4.0**. Attribute DMI in
  anything published from these queries.
