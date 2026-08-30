"""National calibration reliability report (website Phase B, B3).

Runs the DuckDB analysis queries in ``sql/`` over the v2 multi-point
corpus Parquet and renders ``reports/national_calibration_report.md``:
pooled weighted reliability per lead, the weighted Brier decomposition,
regional/seasonal reliability, intensity-band base rates, effective
sample sizes per stratum — and the **regional-split verdict**.

Regional-split criterion (plan §B3, applied exactly): a region flags for
splitting only when its reliability curve leaves the pooled curve's
±2·binomial-SE band in **≥ 2 probability bins**. Bin frequencies are
weighted (sample_weight = inverse inclusion probability, Finding 3).

The binomial SE for bin k of a region's curve is::

    SE_k = sqrt( p̂_k · (1 − p̂_k) / n_eff,k )

where ``p̂_k`` is the POOLED weighted observed frequency in bin k (the
null hypothesis — "this region behaves like the pool") and ``n_eff,k`` is
the REGION's Kish effective sample size ``(Σw)²/Σw²`` in that bin (the
region's estimate carries the sampling noise; the pooled curve is an
order of magnitude better-determined and treated as fixed). ``p̂_k`` is
clamped to ``[0.5/n_eff, 1 − 0.5/n_eff]`` so empty-frequency bins don't
produce zero-width bands, and bins with regional ``n_eff < 20`` are
ignored as too thin to judge either way.

Charts: matplotlib reliability diagrams (one PNG per lead, pooled +
regional curves) are written next to the markdown IF matplotlib imports;
otherwise the report degrades to tables-only with a note.

Dependencies: duckdb + numpy (+ optional matplotlib). This is an
analysis-side tool — the fit itself (``fit_national_calibration.py``)
deliberately needs neither.

Usage::

    python scripts/national_calibration_report.py \\
        --corpus reports/calibration_corpus.parquet \\
        --out-dir reports
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = REPO_ROOT / "sql"

N_BINS = 10
Z_CRIT = 2.0  # ± Z_CRIT · SE band
MIN_DIVERGENT_BINS = 2  # bins outside the band needed to flag a region
MIN_REGION_BIN_EFF_N = 20.0  # regional bins thinner than this are not judged

QUERY_FILES = (
    "reliability_pooled.sql",
    "reliability_by_region.sql",
    "reliability_by_season.sql",
    "base_rate_by_intensity.sql",
    "brier_decomposition.sql",
    "effective_n_by_stratum.sql",
)


# ---------------------------------------------------------------------------
# DuckDB plumbing
# ---------------------------------------------------------------------------


def _sql_quote_path(path: Path) -> str:
    """A single-quoted SQL string literal for the Parquet path."""
    return "'" + str(path).replace("'", "''") + "'"


def run_query(con, name: str, corpus: Path) -> tuple[list[str], list[tuple]]:
    """Execute ``sql/<name>`` with ``{corpus}`` substituted; return
    (column_names, rows)."""
    sql = (SQL_DIR / name).read_text().replace("{corpus}", _sql_quote_path(corpus))
    res = con.execute(sql)
    cols = [d[0] for d in res.description]
    return cols, res.fetchall()


# ---------------------------------------------------------------------------
# Regional-split criterion (pure functions — unit-tested directly)
# ---------------------------------------------------------------------------


def binomial_se(pooled_freq: float, region_eff_n: float) -> float:
    """Binomial standard error of a regional bin frequency under the
    pooled-curve null: ``sqrt(p̂(1−p̂)/n_eff)`` with ``p̂`` clamped away
    from 0/1 (see module docstring)."""
    n = float(region_eff_n)
    if n <= 0:
        return float("inf")
    p = float(np.clip(pooled_freq, 0.5 / n, 1.0 - 0.5 / n))
    return float(np.sqrt(p * (1.0 - p) / n))


def divergent_bin_count(
    pooled_freq: Sequence[float],
    region_freq: Sequence[float],
    region_eff_n: Sequence[float],
    *,
    z: float = Z_CRIT,
    min_eff_n: float = MIN_REGION_BIN_EFF_N,
) -> int:
    """Number of probability bins where the region's weighted reliability
    curve leaves the pooled curve's ``±z·SE`` band.

    Inputs are per-bin arrays (NaN = bin empty for that curve). Bins where
    either curve is empty, or where the region's effective N is below
    ``min_eff_n``, are not judged.
    """
    pooled = np.asarray(pooled_freq, dtype=np.float64)
    region = np.asarray(region_freq, dtype=np.float64)
    eff_n = np.asarray(region_eff_n, dtype=np.float64)
    if not (pooled.shape == region.shape == eff_n.shape):
        raise ValueError("shape mismatch between bin arrays")
    count = 0
    for k in range(pooled.size):
        if not (np.isfinite(pooled[k]) and np.isfinite(region[k])):
            continue
        if not np.isfinite(eff_n[k]) or eff_n[k] < min_eff_n:
            continue
        se = binomial_se(pooled[k], eff_n[k])
        if abs(region[k] - pooled[k]) > z * se:
            count += 1
    return count


def flag_regions(
    pooled_by_lead: dict[int, np.ndarray],
    regional: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    *,
    z: float = Z_CRIT,
    min_bins: int = MIN_DIVERGENT_BINS,
    min_eff_n: float = MIN_REGION_BIN_EFF_N,
) -> tuple[dict[tuple[str, int], int], list[str]]:
    """Apply the criterion across regions and leads.

    ``pooled_by_lead``: {lead: pooled weighted freq per bin (NaN=empty)}.
    ``regional``: {(region, lead): (freq per bin, eff_n per bin)}.

    Returns ``(divergence_table, flagged_regions)`` where the table maps
    (region, lead) → divergent-bin count and a region is flagged when ANY
    lead's curve diverges in ≥ ``min_bins`` bins (reliability curves are
    per lead throughout this project).
    """
    table: dict[tuple[str, int], int] = {}
    flagged: set[str] = set()
    for (region, lead), (freq, eff_n) in regional.items():
        pooled = pooled_by_lead.get(lead)
        if pooled is None:
            continue
        n_div = divergent_bin_count(
            pooled, freq, eff_n, z=z, min_eff_n=min_eff_n
        )
        table[(region, lead)] = n_div
        if n_div >= min_bins:
            flagged.add(region)
    return table, sorted(flagged)


# ---------------------------------------------------------------------------
# Result-shaping helpers
# ---------------------------------------------------------------------------


def _rows_to_bin_arrays(
    rows: list[tuple], cols: list[str], key_cols: tuple[str, ...]
) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    """Pivot (key…, bin, …) rows into {key: (freq[N_BINS], eff_n[N_BINS])}."""
    idx = {c: i for i, c in enumerate(cols)}
    out: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        key = tuple(row[idx[c]] for c in key_cols)
        if key not in out:
            out[key] = (np.full(N_BINS, np.nan), np.full(N_BINS, np.nan))
        b = int(row[idx["bin"]])
        out[key][0][b] = float(row[idx["obs_freq_weighted"]])
        out[key][1][b] = float(row[idx["eff_n"]])
    return out


def md_table(cols: list[str], rows: list[tuple], *, floatfmt: str = "{:.4f}") -> str:
    """Render rows as a GitHub-markdown table."""
    def fmt(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            if not np.isfinite(v):
                return "nan"
            return floatfmt.format(v)
        return str(v)

    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Charts (optional)
# ---------------------------------------------------------------------------


def render_reliability_pngs(
    out_dir: Path,
    pooled_cols: list[str],
    pooled_rows: list[tuple],
    regional: dict[tuple, tuple[np.ndarray, np.ndarray]],
) -> Optional[dict[int, str]]:
    """One reliability diagram PNG per lead. Returns {lead: filename} or
    None when matplotlib is unavailable (report degrades to tables)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 — any import/backend failure → tables-only
        return None

    idx = {c: i for i, c in enumerate(pooled_cols)}
    by_lead: dict[int, list[tuple]] = {}
    for row in pooled_rows:
        by_lead.setdefault(int(row[idx["lead_min"]]), []).append(row)

    files: dict[int, str] = {}
    for lead, rows in sorted(by_lead.items()):
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.plot([0, 1], [0, 1], color="#999999", lw=1.0, ls="--", label="perfect")
        # Regional curves first (light), pooled on top.
        for (region, r_lead), (freq, _eff) in sorted(regional.items()):
            if int(r_lead) != lead:
                continue
            mask = np.isfinite(freq)
            centers = (np.arange(N_BINS) + 0.5) / N_BINS
            ax.plot(
                centers[mask], freq[mask],
                lw=0.8, alpha=0.45, marker=".", ms=3, label=f"{region}",
            )
        x = [float(r[idx["mean_raw_weighted"]]) for r in rows]
        y = [float(r[idx["obs_freq_weighted"]]) for r in rows]
        ax.plot(x, y, color="#1f4e79", lw=2.0, marker="o", ms=5, label="pooled (weighted)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("forecast probability (raw)")
        ax.set_ylabel("observed frequency (weighted)")
        ax.set_title(f"Reliability — lead +{lead} min")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.grid(alpha=0.25)
        fname = f"national_reliability_lead{lead:02d}.png"
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130)
        plt.close(fig)
        files[lead] = fname
    return files


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(corpus: Path, out_dir: Path) -> Path:
    import duckdb

    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    # Corpus header facts (inline — the sql/ files are the analysis queries).
    hdr = con.execute(
        "SELECT count(*), count(DISTINCT event_time), count(DISTINCT point_id), "
        "       min(settings_hash), count(DISTINCT settings_hash), "
        "       min(lead_min), max(lead_min) "
        f"FROM read_parquet({_sql_quote_path(corpus)})"
    ).fetchone()
    n_rows, n_events, n_points, settings_hash, n_hashes, *_ = hdr

    pooled_cols, pooled_rows = run_query(con, "reliability_pooled.sql", corpus)
    region_cols, region_rows = run_query(con, "reliability_by_region.sql", corpus)
    season_cols, season_rows = run_query(con, "reliability_by_season.sql", corpus)
    inten_cols, inten_rows = run_query(con, "base_rate_by_intensity.sql", corpus)
    brier_cols, brier_rows = run_query(con, "brier_decomposition.sql", corpus)
    effn_cols, effn_rows = run_query(con, "effective_n_by_stratum.sql", corpus)

    # Criterion inputs.
    pooled_arrays = _rows_to_bin_arrays(pooled_rows, pooled_cols, ("lead_min",))
    pooled_by_lead = {int(k[0]): v[0] for k, v in pooled_arrays.items()}
    regional_arrays = _rows_to_bin_arrays(region_rows, region_cols, ("region", "lead_min"))
    regional = {(str(r), int(l)): v for (r, l), v in regional_arrays.items()}
    divergence, flagged = flag_regions(pooled_by_lead, regional)

    pngs = render_reliability_pngs(out_dir, pooled_cols, pooled_rows, regional)

    leads = sorted(pooled_by_lead)
    pidx = {c: i for i, c in enumerate(pooled_cols)}

    md: list[str] = []
    md.append("# National calibration reliability report")
    md.append("")
    md.append(f"Generated {datetime.now(timezone.utc).isoformat()} (Phase B, B3).")
    md.append("")
    md.append(f"- Corpus: `{corpus}`")
    md.append(f"- Rows: {n_rows} ({n_events} events x {n_points} points)")
    md.append(
        f"- Settings hash: `{settings_hash}`"
        + ("" if n_hashes == 1 else f" — **WARNING: {n_hashes} mixed hashes!**")
    )
    md.append(f"- Leads: {', '.join(f'+{ld} min' for ld in leads)}")
    md.append("")
    md.append(
        "All frequencies and Brier terms are weighted by `sample_weight` "
        "(inverse inclusion probability of the wet-biased event sampler — "
        "Finding 3); `eff_n` is the Kish effective sample size (sum w)^2 / sum w^2."
    )
    md.append("")

    md.append("## Pooled reliability per lead")
    md.append("")
    if pngs is None:
        md.append(
            "_matplotlib unavailable — reliability diagrams omitted, "
            "tables only._"
        )
        md.append("")
    for lead in leads:
        md.append(f"### Lead +{lead} min")
        md.append("")
        if pngs and lead in pngs:
            md.append(f"![Reliability lead +{lead} min]({pngs[lead]})")
            md.append("")
        rows = [r for r in pooled_rows if int(r[pidx["lead_min"]]) == lead]
        md.append(md_table(pooled_cols, rows))
        md.append("")

    md.append("## Weighted Brier decomposition (Murphy)")
    md.append("")
    md.append(
        "`Brier ≈ reliability − resolution + uncertainty` over the 10 "
        "probability bins; `decomp_residual` is the within-bin variance the "
        "binning discards. Calibration attacks the reliability term."
    )
    md.append("")
    md.append(md_table(brier_cols, brier_rows))
    md.append("")

    md.append("## Regional reliability and divergence")
    md.append("")
    md.append(
        f"Criterion (plan §B3): a region flags for splitting only when its "
        f"weighted reliability curve leaves the pooled curve's "
        f"±{Z_CRIT:g}·binomial-SE band in ≥{MIN_DIVERGENT_BINS} probability "
        f"bins for some lead. SE per bin: `sqrt(p_pool·(1−p_pool)/n_eff_region)` "
        f"with `p_pool` the pooled weighted frequency (clamped to "
        f"[0.5/n_eff, 1−0.5/n_eff]) and `n_eff_region` the region's Kish "
        f"effective bin count; regional bins with `n_eff < {MIN_REGION_BIN_EFF_N:g}` "
        f"are not judged."
    )
    md.append("")
    div_rows = [
        (region, lead, n_div, "FLAG" if n_div >= MIN_DIVERGENT_BINS else "")
        for (region, lead), n_div in sorted(divergence.items())
    ]
    md.append(md_table(["region", "lead_min", "divergent_bins", "verdict"], div_rows))
    md.append("")

    md.append("## Verdict — regional split")
    md.append("")
    if flagged:
        md.append(
            "The following regions leave the pooled reliability band in "
            f"≥{MIN_DIVERGENT_BINS} bins and are candidates for regional "
            "curves (a follow-up decision, not automatic — plan §3):"
        )
        md.append("")
        for region in flagged:
            hits = [
                f"+{lead} min ({n_div} bins)"
                for (r, lead), n_div in sorted(divergence.items())
                if r == region and n_div >= MIN_DIVERGENT_BINS
            ]
            md.append(f"- **{region}** — {', '.join(hits)}")
    else:
        md.append(
            "**No regions flagged.** Every regional reliability curve stays "
            f"within the pooled curve's ±{Z_CRIT:g}·SE band (fewer than "
            f"{MIN_DIVERGENT_BINS} divergent bins at every lead) — the pooled "
            "national curves are sufficient."
        )
    md.append("")

    md.append("## Seasonal reliability (diagnostic only)")
    md.append("")
    md.append(
        "Month-based seasons (DJF/MAM/JJA/SON). Phase B fits no seasonal "
        "curves — this is a sanity view against the Imhoff et al. 2020 "
        "expectation of ~3× lower skill on summer convective vs winter "
        "stratiform. The 180-day archive depth means a single corpus covers "
        "at most two seasons well."
    )
    md.append("")
    md.append(md_table(season_cols, season_rows))
    md.append("")

    md.append("## Base rates by event intensity band")
    md.append("")
    md.append(
        "Intensity proxy: event wet coverage — the weighted share of "
        "calibration points wet at the shortest lead (the corpus carries no "
        "mm/h field)."
    )
    md.append("")
    md.append(md_table(inten_cols, inten_rows))
    md.append("")

    md.append("## Effective sample size per stratum")
    md.append("")
    md.append(md_table(effn_cols, effn_rows))
    md.append("")

    report_path = out_dir / "national_calibration_report.md"
    report_path.write_text("\n".join(md))
    return report_path


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--corpus", type=Path, required=True,
        help="v2 multi-point corpus Parquet",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=Path("reports"),
        help="Directory for the markdown report + PNG charts (default: reports/)",
    )
    args = ap.parse_args(argv)

    try:
        import duckdb  # noqa: F401
    except ImportError:
        print(
            "duckdb is required for the analysis report (dev dependency): "
            "run `uv add --dev duckdb` from the repo root. The national FIT "
            "(fit_national_calibration.py) does not need it.",
            file=sys.stderr,
        )
        return 1

    if not args.corpus.exists():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 2

    report_path = build_report(args.corpus, args.out_dir)
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
