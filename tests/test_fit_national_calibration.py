"""Tests for the national weighted calibration fit + report (Phase B, B3).

Covers:
- weighted PAV / ``fit_isotonic_weighted`` in ``dmi_nowcast_core.calibrate``
  (hand-computable weighted cases; equal-weights == unweighted invariant);
- ``scripts/fit_national_calibration.py`` end-to-end on synthetic v2
  corpus Parquet files (weight correction, refusals, output format);
- ``scripts/national_calibration_report.py`` smoke test + the
  regional-split criterion on constructed reliability data.

NO network, NO STEPS, NO DMI — everything runs on synthetic Parquet built
with pyarrow, using the corpus builder's own Arrow schema so the fitter
and builder can never drift apart silently.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_calibration_corpus as bcc  # noqa: E402  (after sys.path edit)
import fit_national_calibration as fnc  # noqa: E402
import national_calibration_report as ncr  # noqa: E402

from dmi_nowcast_core.calibrate import (  # noqa: E402
    brier_score_weighted,
    fit_isotonic,
    fit_isotonic_weighted,
    load_calibration_curves,
    pava,
    pava_weighted,
)


# ---------------------------------------------------------------------------
# Weighted PAV — unit tests (calibrate.py additions)
# ---------------------------------------------------------------------------


def test_pava_weighted_hand_computed_pooling():
    # y = [0.4, 0.6, 0.2], w = [1, 1, 2]:
    #   0.6 > 0.2 → merge: (0.6·1 + 0.2·2)/3 = 1/3
    #   0.4 > 1/3 → merge: (0.4·1 + 1.0)/4 = 0.35
    y = np.array([0.4, 0.6, 0.2])
    w = np.array([1.0, 1.0, 2.0])
    out = pava_weighted(y, w)
    np.testing.assert_allclose(out, [0.35, 0.35, 0.35])


def test_pava_weighted_equal_weights_matches_pava():
    rng = np.random.default_rng(0)
    y = rng.random(50)
    w = np.full(50, 3.7)
    np.testing.assert_allclose(pava_weighted(y, w), pava(y), atol=1e-12)


def test_pava_weighted_monotone_identity():
    y = np.array([0.1, 0.3, 0.4, 0.7, 0.9])
    w = np.array([5.0, 1.0, 2.0, 0.5, 9.0])
    np.testing.assert_allclose(pava_weighted(y, w), y)


def test_fit_isotonic_weighted_groups_by_weighted_mean():
    # raw 0.2 group: outcomes [0 (w=1), 1 (w=3)] → weighted mean 0.75.
    raw = np.array([0.2, 0.2, 0.8])
    out = np.array([0.0, 1.0, 1.0])
    w = np.array([1.0, 3.0, 1.0])
    cal = fit_isotonic_weighted(raw, out, w)
    assert cal.raw_breakpoints == (0.2, 0.8)
    assert cal.predict(0.2) == pytest.approx(0.75)
    assert cal.predict(0.8) == pytest.approx(1.0)


def test_fit_isotonic_weighted_pools_with_weights():
    # Group means [1.0 (Σw=1), 0.0 (Σw=3)] violate → pooled weighted mean
    # = (1·1 + 3·0)/4 = 0.25 for both breakpoints.
    raw = np.array([0.2, 0.8])
    out = np.array([1.0, 0.0])
    w = np.array([1.0, 3.0])
    cal = fit_isotonic_weighted(raw, out, w)
    assert cal.predict(0.2) == pytest.approx(0.25)
    assert cal.predict(0.8) == pytest.approx(0.25)


def test_fit_isotonic_weighted_equal_weights_equals_unweighted():
    rng = np.random.default_rng(1)
    raw = rng.random(300)
    out = (rng.random(300) < raw).astype(np.float64)
    w = np.full(300, 2.7)
    weighted = fit_isotonic_weighted(raw, out, w)
    unweighted = fit_isotonic(raw, out)
    np.testing.assert_allclose(weighted.raw_breakpoints, unweighted.raw_breakpoints)
    np.testing.assert_allclose(
        weighted.calibrated_values, unweighted.calibrated_values, atol=1e-12
    )


def test_fit_isotonic_weighted_scale_invariant():
    rng = np.random.default_rng(2)
    raw = rng.random(100)
    out = (rng.random(100) < raw).astype(np.float64)
    w = rng.random(100) + 0.1
    a = fit_isotonic_weighted(raw, out, w)
    b = fit_isotonic_weighted(raw, out, w * 1000.0)
    np.testing.assert_allclose(a.calibrated_values, b.calibrated_values, atol=1e-12)


def test_fit_isotonic_weighted_nan_safe():
    raw = np.array([0.2, 0.2, 0.8, np.nan, 0.5, 0.5])
    out = np.array([0.0, 1.0, 1.0, 1.0, np.nan, 0.0])
    w = np.array([1.0, 3.0, 1.0, 1.0, 1.0, np.nan])
    cal = fit_isotonic_weighted(raw, out, w)
    # Only the first three (finite) samples survive → same as the grouped test.
    assert cal.raw_breakpoints == (0.2, 0.8)
    assert cal.predict(0.2) == pytest.approx(0.75)


def test_fit_isotonic_weighted_zero_weight_rows_have_no_effect():
    raw = np.array([0.2, 0.2, 0.8, 0.8])
    out = np.array([0.0, 1.0, 1.0, 0.0])
    w = np.array([1.0, 3.0, 1.0, 0.0])  # last row must not count
    cal = fit_isotonic_weighted(raw, out, w)
    ref = fit_isotonic_weighted(raw[:3], out[:3], w[:3])
    assert cal.raw_breakpoints == ref.raw_breakpoints
    np.testing.assert_allclose(cal.calibrated_values, ref.calibrated_values)


def test_fit_isotonic_weighted_degenerate_inputs():
    with pytest.raises(ValueError, match="empty"):
        fit_isotonic_weighted(np.array([]), np.array([]), np.array([]))
    with pytest.raises(ValueError, match="shape"):
        fit_isotonic_weighted(np.array([0.5]), np.array([1.0, 0.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="no finite samples"):
        fit_isotonic_weighted(
            np.array([np.nan]), np.array([1.0]), np.array([1.0])
        )
    with pytest.raises(ValueError, match="negative"):
        fit_isotonic_weighted(
            np.array([0.5, 0.6]), np.array([1.0, 0.0]), np.array([1.0, -1.0])
        )


def test_fit_isotonic_weighted_output_monotone():
    rng = np.random.default_rng(3)
    raw = rng.random(200)
    out = (rng.random(200) < raw).astype(np.float64)
    w = rng.random(200) * 10 + 0.1
    cal = fit_isotonic_weighted(raw, out, w)
    ys = cal.predict(np.linspace(0, 1, 101))
    assert np.all(np.diff(ys) >= -1e-6)


def test_brier_score_weighted_hand_computed():
    assert brier_score_weighted(
        np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 5.0])
    ) == 0.0
    # (3·0.04 + 1·0.64) / 4 = 0.19
    assert brier_score_weighted(
        np.array([0.8, 0.8]), np.array([1.0, 0.0]), np.array([3.0, 1.0])
    ) == pytest.approx(0.19)
    assert math.isnan(
        brier_score_weighted(np.array([np.nan]), np.array([1.0]), np.array([1.0]))
    )


def test_kish_effective_n():
    assert fnc.kish_effective_n(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(4.0)
    assert fnc.kish_effective_n(np.array([5.0, 5.0])) == pytest.approx(2.0)
    assert fnc.kish_effective_n(np.array([1.0, 3.0])) == pytest.approx(1.6)
    assert fnc.kish_effective_n(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# Synthetic v2 corpus helpers (builder's own Arrow schema — drift guard)
# ---------------------------------------------------------------------------

SETTINGS = bcc.CorpusSettings(
    ensemble_size=16,
    n_cascade_levels=6,
    downsample_factor=4,
    threshold_mm_h=0.5,
    disc_radius_m=1000.0,
    detection_stat="p90",
    leads_min=(10, 30),
)
SETTINGS_COLS = SETTINGS.settings_columns()

POINTS = {
    "home": ("fyn", 55.33, 10.32),
    "west": ("vestjylland", 56.0, 8.5),
    "east": ("sjaelland", 55.67, 12.56),
}


def _event_iso(i: int) -> str:
    return (
        datetime(2026, 6, 1, tzinfo=timezone.utc) + timedelta(hours=i)
    ).isoformat()


def _row(
    event_i: int,
    lead: int,
    raw,
    outcome,
    weight: float,
    *,
    point: str = "home",
    overrides: dict | None = None,
) -> dict:
    region, lat, lon = POINTS[point]
    row = {
        "event_time": _event_iso(event_i),
        "point_id": point,
        "lat": lat,
        "lon": lon,
        "region": region,
        "lead_min": lead,
        "raw_prob": float(raw),
        "outcome": outcome,
        "sample_weight": weight,
        # Per-event simulated frame age. The fit never reads it (it groups
        # by lead_min alone), but the builder writes it on every row, so the
        # synthetic corpora must carry it too — this helper doubles as the
        # drift guard against the builder's Arrow schema.
        "frame_age_min": 15.0,
        "error": "",
    }
    row.update(SETTINGS_COLS)
    if overrides:
        row.update(overrides)
    return row


def _write_corpus(path: Path, rows: list[dict], drop_columns: tuple[str, ...] = ()) -> Path:
    schema = bcc._parquet_schema()
    if drop_columns:
        schema = pa.schema([f for f in schema if f.name not in drop_columns])
        rows = [{k: v for k, v in r.items() if k not in drop_columns} for r in rows]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def _miscalibrated_rows(*, weight: float = 1.0) -> list[dict]:
    """Both leads: raw 0.125 observed 30%, raw 0.875 observed 50% — a known
    miscalibration in two bins, exact fractions, equal weights.

    (Raw levels are exact binary fractions so the Parquet float32 round-trip
    is lossless.)
    """
    rows: list[dict] = []
    i = 0
    for lead in (10, 30):
        for k in range(10):
            rows.append(_row(i, lead, 0.125, int(k < 3), weight))
            i += 1
        for k in range(10):
            rows.append(_row(i, lead, 0.875, int(k < 5), weight))
            i += 1
    return rows


# ---------------------------------------------------------------------------
# Fitter — end-to-end on synthetic corpora
# ---------------------------------------------------------------------------


def test_fitter_corrects_known_miscalibration(tmp_path: Path):
    corpus = _write_corpus(tmp_path / "c.parquet", _miscalibrated_rows())
    output = tmp_path / "national_curves.json"
    rc = fnc.main([
        "--corpus", str(corpus), "--output", str(output),
        "--min-samples-per-lead", "10",
    ])
    assert rc == 0
    curves = load_calibration_curves(output)
    assert sorted(curves) == [10, 30]
    for lead in (10, 30):
        assert curves[lead].predict(0.125) == pytest.approx(0.3, abs=1e-6)
        assert curves[lead].predict(0.875) == pytest.approx(0.5, abs=1e-6)

    meta = json.loads(output.read_text())["metadata"]
    # Exact keys B4's manifest echo reads.
    for key in ("fitted_at", "n_samples", "brier_before", "brier_after"):
        assert key in meta, key
    datetime.fromisoformat(meta["fitted_at"])  # parses, UTC ISO
    assert meta["fitted_at"].endswith("+00:00")
    assert meta["n_samples"] == 40  # 20 valid rows per lead × 2 leads
    assert meta["settings_hash"] == SETTINGS.settings_hash
    assert meta["n_events"] == 40
    assert meta["n_points"] == 1
    assert meta["settings"]["leads_min_csv"] == "10,30"
    # Correcting a real miscalibration must improve the (weighted) Brier.
    assert meta["brier_after"] < meta["brier_before"]
    for lead in ("10", "30"):
        lead_meta = meta["leads"][lead]
        for key in ("n_samples", "effective_n", "base_rate", "brier_before", "brier_after"):
            assert key in lead_meta, (lead, key)
        assert lead_meta["n_samples"] == 20
        assert lead_meta["effective_n"] == pytest.approx(20.0)  # equal weights
        assert lead_meta["base_rate"] == pytest.approx(0.4)


def test_fitter_uses_sample_weights(tmp_path: Path):
    """Wet-biased corpus: the fit must land on the WEIGHTED frequencies.

    raw 0.25: 8 dry rows (w=10, outcome 0) + 2 wet rows (w=2, outcome 1)
      → weighted 4/84 = 1/21;   unweighted would be 0.2.
    raw 0.75: 4 dry rows (w=10, outcome 0) + 6 wet rows (w=2, outcome 1)
      → weighted 12/52 = 3/13;  unweighted would be 0.6.
    """
    rows: list[dict] = []
    i = 0
    for lead in (10, 30):
        for _ in range(8):
            rows.append(_row(i, lead, 0.25, 0, 10.0)); i += 1
        for _ in range(2):
            rows.append(_row(i, lead, 0.25, 1, 2.0)); i += 1
        for _ in range(4):
            rows.append(_row(i, lead, 0.75, 0, 10.0)); i += 1
        for _ in range(6):
            rows.append(_row(i, lead, 0.75, 1, 2.0)); i += 1
    corpus = _write_corpus(tmp_path / "c.parquet", rows)
    output = tmp_path / "curves.json"
    assert fnc.main([
        "--corpus", str(corpus), "--output", str(output),
        "--min-samples-per-lead", "10",
    ]) == 0
    curves = load_calibration_curves(output)
    for lead in (10, 30):
        assert curves[lead].predict(0.25) == pytest.approx(1 / 21, abs=1e-6)
        assert curves[lead].predict(0.75) == pytest.approx(3 / 13, abs=1e-6)
        # Emphatically NOT the unweighted frequencies.
        assert abs(curves[lead].predict(0.25) - 0.2) > 0.1
        assert abs(curves[lead].predict(0.75) - 0.6) > 0.3
    # Kish effective N per lead: Σw = 80+4+40+12 = 136, Σw² = 12·100+8·4 = 1232.
    meta = json.loads(output.read_text())["metadata"]
    assert meta["leads"]["10"]["effective_n"] == pytest.approx(136.0**2 / 1232.0)


def test_fitter_equal_weights_matches_unweighted_fit(tmp_path: Path):
    rows = _miscalibrated_rows(weight=3.0)  # equal, non-unit weights
    corpus = _write_corpus(tmp_path / "c.parquet", rows)
    output = tmp_path / "curves.json"
    assert fnc.main([
        "--corpus", str(corpus), "--output", str(output),
        "--min-samples-per-lead", "10",
    ]) == 0
    curves = load_calibration_curves(output)
    # Reference: plain unweighted fit_isotonic on the same per-lead data.
    raw = np.array([r["raw_prob"] for r in rows if r["lead_min"] == 10])
    out = np.array([float(r["outcome"]) for r in rows if r["lead_min"] == 10])
    ref = fit_isotonic(raw, out)
    np.testing.assert_allclose(curves[10].raw_breakpoints, ref.raw_breakpoints)
    np.testing.assert_allclose(curves[10].calibrated_values, ref.calibrated_values)


def test_fitter_filters_nan_and_null_rows(tmp_path: Path):
    rows = _miscalibrated_rows()
    # Junk that must be ignored: NaN raw, null outcome, NaN weight.
    rows.append(_row(900, 10, float("nan"), 1, 1.0, overrides={"error": "boom"}))
    rows.append(_row(901, 10, 0.5, None, 1.0))
    rows.append(_row(902, 10, 0.5, 1, float("nan")))
    corpus = _write_corpus(tmp_path / "c.parquet", rows)
    output = tmp_path / "curves.json"
    assert fnc.main([
        "--corpus", str(corpus), "--output", str(output),
        "--min-samples-per-lead", "10",
    ]) == 0
    meta = json.loads(output.read_text())["metadata"]
    assert meta["leads"]["10"]["n_samples"] == 20  # junk rows filtered
    curves = load_calibration_curves(output)
    assert curves[10].predict(0.125) == pytest.approx(0.3, abs=1e-6)


def test_fitter_refuses_mixed_settings_hash(tmp_path: Path, capsys):
    rows = _miscalibrated_rows()
    for r in rows[: len(rows) // 2]:
        r["settings_hash"] = "deadbeefdeadbeef"
    corpus = _write_corpus(tmp_path / "c.parquet", rows)
    rc = fnc.main(["--corpus", str(corpus), "--output", str(tmp_path / "o.json")])
    assert rc == 2
    assert "mixes settings hashes" in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


def test_fitter_refuses_v1_schema_version(tmp_path: Path, capsys):
    rows = [
        _row(i, 10, 0.5, 1, 1.0, overrides={"schema_version": 1})
        for i in range(20)
    ]
    corpus = _write_corpus(tmp_path / "c.parquet", rows)
    rc = fnc.main(["--corpus", str(corpus), "--output", str(tmp_path / "o.json")])
    assert rc == 2
    assert "schema_version" in capsys.readouterr().err


def test_fitter_refuses_missing_schema_version_column(tmp_path: Path, capsys):
    corpus = _write_corpus(
        tmp_path / "c.parquet", _miscalibrated_rows(), drop_columns=("schema_version",)
    )
    rc = fnc.main(["--corpus", str(corpus), "--output", str(tmp_path / "o.json")])
    assert rc == 2
    assert "legacy" in capsys.readouterr().err


def test_fitter_refuses_missing_sample_weight(tmp_path: Path, capsys):
    corpus = _write_corpus(
        tmp_path / "c.parquet", _miscalibrated_rows(), drop_columns=("sample_weight",)
    )
    rc = fnc.main(["--corpus", str(corpus), "--output", str(tmp_path / "o.json")])
    assert rc == 2
    assert "sample_weight" in capsys.readouterr().err


def test_fitter_refuses_corpus_without_motion_method(tmp_path: Path, capsys):
    """R5: a corpus built before motion-field completion has no
    ``motion_method`` column at all, which makes it structurally unusable —
    the fit must say so rather than quietly calibrate against probabilities
    produced by a different motion field."""
    corpus = _write_corpus(
        tmp_path / "c.parquet", _miscalibrated_rows(), drop_columns=("motion_method",)
    )
    rc = fnc.main(["--corpus", str(corpus), "--output", str(tmp_path / "o.json")])
    assert rc == 2
    assert "motion_method" in capsys.readouterr().err


def test_fitter_refuses_thin_leads(tmp_path: Path, capsys):
    corpus = _write_corpus(tmp_path / "c.parquet", _miscalibrated_rows())
    # Default --min-samples-per-lead is 500; 20 valid rows per lead is thin.
    rc = fnc.main(["--corpus", str(corpus), "--output", str(tmp_path / "o.json")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "refusing to fit noise" in err
    assert not (tmp_path / "o.json").exists()


def test_fitter_skips_thin_lead_keeps_thick(tmp_path: Path, capsys):
    rows = _miscalibrated_rows()
    # Make lead 30 thin: keep only 5 of its rows.
    lead30 = [r for r in rows if r["lead_min"] == 30][:5]
    rows = [r for r in rows if r["lead_min"] == 10] + lead30
    corpus = _write_corpus(tmp_path / "c.parquet", rows)
    output = tmp_path / "curves.json"
    rc = fnc.main([
        "--corpus", str(corpus), "--output", str(output),
        "--min-samples-per-lead", "10",
    ])
    assert rc == 0
    assert "lead +30 min: only 5 valid samples" in capsys.readouterr().err
    curves = load_calibration_curves(output)
    assert sorted(curves) == [10]
    meta = json.loads(output.read_text())["metadata"]
    assert "30" not in meta["leads"]


# ---------------------------------------------------------------------------
# Regional-split criterion (constructed reliability data)
# ---------------------------------------------------------------------------


def test_binomial_se_formula():
    assert ncr.binomial_se(0.2, 500.0) == pytest.approx(math.sqrt(0.2 * 0.8 / 500))
    # Clamped away from 0 so empty-frequency bins keep a non-zero band.
    assert ncr.binomial_se(0.0, 100.0) == pytest.approx(
        math.sqrt(0.005 * 0.995 / 100)
    )
    assert ncr.binomial_se(0.5, 0.0) == float("inf")


def test_divergent_bin_count_hand_computed():
    pooled = np.full(10, 0.2)
    eff_n = np.full(10, 500.0)
    # 2·SE = 2·sqrt(0.2·0.8/500) ≈ 0.0358.
    aligned = np.full(10, 0.21)  # diff 0.01 < band everywhere
    assert ncr.divergent_bin_count(pooled, aligned, eff_n) == 0
    divergent = np.full(10, 0.21)
    divergent[[3, 4, 5]] = 0.30  # diff 0.10 > band in 3 bins
    assert ncr.divergent_bin_count(pooled, divergent, eff_n) == 3
    # Thin bins are not judged at all.
    thin_n = np.full(10, 5.0)
    assert ncr.divergent_bin_count(pooled, divergent, thin_n) == 0
    # Empty (NaN) bins skipped on either side.
    holey = divergent.copy()
    holey[3] = np.nan
    assert ncr.divergent_bin_count(pooled, holey, eff_n) == 2


def test_flag_regions_criterion():
    pooled = {10: np.full(10, 0.2)}
    eff_n = np.full(10, 500.0)
    aligned = np.full(10, 0.21)
    one_bin = np.full(10, 0.21)
    one_bin[4] = 0.5
    three_bins = np.full(10, 0.21)
    three_bins[[2, 5, 8]] = 0.45
    thin_wild = np.full(10, 0.9)
    regional = {
        ("aligned", 10): (aligned, eff_n),
        ("one_bin", 10): (one_bin, eff_n),
        ("divergent", 10): (three_bins, eff_n),
        ("thin", 10): (thin_wild, np.full(10, 5.0)),
    }
    table, flagged = ncr.flag_regions(pooled, regional)
    assert flagged == ["divergent"]  # ≥2 divergent bins flags; others don't
    assert table[("divergent", 10)] == 3
    assert table[("one_bin", 10)] == 1
    assert table[("aligned", 10)] == 0
    assert table[("thin", 10)] == 0


def test_flag_regions_any_lead_triggers():
    pooled = {10: np.full(10, 0.2), 30: np.full(10, 0.2)}
    eff_n = np.full(10, 500.0)
    ok = np.full(10, 0.2)
    bad = np.full(10, 0.2)
    bad[[1, 2]] = 0.5
    regional = {
        ("r1", 10): (ok, eff_n),
        ("r1", 30): (bad, eff_n),  # diverges only at lead 30 → still flags
    }
    _table, flagged = ncr.flag_regions(pooled, regional)
    assert flagged == ["r1"]


# ---------------------------------------------------------------------------
# Report — smoke test on a synthetic corpus
# ---------------------------------------------------------------------------


def _smoke_corpus_rows() -> list[dict]:
    rng = np.random.default_rng(7)
    rows: list[dict] = []
    i = 0
    for lead in (10, 30):
        for point in ("west", "east"):
            for _ in range(60):
                raw = float(rng.choice([0.125, 0.375, 0.625, 0.875]))
                outcome = int(rng.random() < raw)
                weight = float(rng.choice([1.0, 8.0]))
                rows.append(_row(i, lead, raw, outcome, weight, point=point))
                i += 1
    # Exercise the validity filters too.
    rows.append(_row(i, 10, float("nan"), 1, 1.0)); i += 1
    rows.append(_row(i, 10, 0.5, None, 1.0)); i += 1
    return rows


def test_report_smoke(tmp_path: Path):
    pytest.importorskip("duckdb")
    corpus = _write_corpus(tmp_path / "c.parquet", _smoke_corpus_rows())
    out_dir = tmp_path / "reports"
    rc = ncr.main(["--corpus", str(corpus), "--out-dir", str(out_dir)])
    assert rc == 0

    report = out_dir / "national_calibration_report.md"
    assert report.exists()
    text = report.read_text()

    # Per-lead pooled tables.
    assert "## Pooled reliability per lead" in text
    assert "### Lead +10 min" in text
    assert "### Lead +30 min" in text
    assert "obs_freq_weighted" in text
    # All analysis sections present.
    assert "## Weighted Brier decomposition" in text
    assert "## Regional reliability and divergence" in text
    assert "## Seasonal reliability" in text
    assert "## Base rates by event intensity band" in text
    assert "## Effective sample size per stratum" in text
    # The verdict section, with a definite outcome either way.
    assert "## Verdict — regional split" in text
    assert ("No regions flagged" in text) or ("candidates for regional" in text)
    # SE formula documented in the report itself.
    assert "binomial-SE" in text

    # Charts: PNGs exist iff matplotlib is importable.
    try:
        import matplotlib  # noqa: F401
        has_mpl = True
    except ImportError:
        has_mpl = False
    pngs = sorted(out_dir.glob("national_reliability_lead*.png"))
    if has_mpl:
        assert [p.name for p in pngs] == [
            "national_reliability_lead10.png",
            "national_reliability_lead30.png",
        ]
        for p in pngs:
            assert p.name in text  # referenced from the markdown
    else:
        assert pngs == []
        assert "matplotlib unavailable" in text


def test_report_refuses_missing_corpus(tmp_path: Path, capsys):
    pytest.importorskip("duckdb")
    rc = ncr.main([
        "--corpus", str(tmp_path / "nope.parquet"), "--out-dir", str(tmp_path),
    ])
    assert rc == 2
    assert "not found" in capsys.readouterr().err
