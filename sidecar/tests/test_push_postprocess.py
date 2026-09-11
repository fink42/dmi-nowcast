"""H-P — the serving path for the gauge-trained post-processing model.

The model was fitted and measured offline
(``archive/l3_and_postprocess_20260911/``: ΔBSS +0.14…+0.19 at the gauges,
ΔF1 +0.03…+0.06 on the shipped warning rule). This suite is about the
*plumbing* that puts it in front of a subscriber, and the seams are the
places where a wrong number would be invisible:

1. **Parity.** The live cycle and the offline replay must produce the same
   feature row from the same inputs. A model fitted on rows that mean one
   thing and applied to rows that mean another does not fail — it just
   quietly predicts badly.
2. **The reader is total.** No model, a truncated one, a schema version
   this build does not know: every one of them means "decide on the
   curve", never an exception in a fan-out.
3. **The fallback is per observation.** One point off the model's coverage
   must not silence it, and one model outage must not silence everyone.
4. **``probability_source: curve`` is the rollback** and restores the
   pre-Phase-H behaviour exactly.
5. **The rows and the thresholds agree.** The live decision row carries
   the probability the decision was taken on, the nightly sweep is fitted
   on that same column, and every existing reader of the decision schema
   is untouched by both.

Offline and synthetic throughout: no radar, no STEPS, no network.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import structlog

pytest.importorskip("pyarrow")

from dmi_nowcast_core import postprocess as pp
from dmi_nowcast_core.geo import GridIndex
from dmi_nowcast_core.national import NationalProducts
from dmi_nowcast_sidecar import compute as compute_mod
from dmi_nowcast_sidecar.config import Config, PushConfig
from dmi_nowcast_sidecar.push import service as service_mod
from dmi_nowcast_sidecar.push.engine import INITIAL_STATE, Observation, Rules, evaluate
from dmi_nowcast_sidecar.push.paths import (
    POSTPROCESS_FILENAME,
    resolved_db_path,
    resolved_postprocess_path,
    resolved_thresholds_path,
)
from dmi_nowcast_sidecar.push.postprocess import (
    CyclePostprocess,
    PostprocessTable,
    build_cycle_postprocess,
    point_key,
)
from dmi_nowcast_sidecar.push.service import PushService
from dmi_nowcast_sidecar.push.store import NewSubscription, PushStore
from dmi_nowcast_sidecar.push.thresholds import ThresholdTable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import replay_warnings as rw  # noqa: E402  (after the sys.path edit)

RADAR_TS = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
GENERATED_AT = RADAR_TS + timedelta(minutes=14)
LEADS = (10, 20, 30, 45, 60)
HOME_LAT, HOME_LON = 55.33, 10.32
ENDPOINT_A = "https://fcm.googleapis.com/fcm/send/AAAAAAAAAAA-token-a"
ENDPOINT_B = "https://updates.push.services.mozilla.com/wpush/v2/token-b"
P256DH = "B" + "x" * 86
AUTH = "y" * 22
SUBJECT = "mailto:ops@example.com"


# ---------------------------------------------------------------------------
# A model, fitted on noise. Only its SHAPE matters to the serving path.
# ---------------------------------------------------------------------------


def _training_rows(n: int = 400) -> tuple[dict, dict]:
    rng = np.random.default_rng(7)
    signal = rng.random(n)
    rows: dict[str, object] = {
        name: rng.random(n) * 5.0
        for name in pp.DESIGN_SOURCE_COLUMNS if name != "hour_utc"
    }
    for lead in LEADS:
        rows[pp.raw_fraction_column(lead)] = signal
    rows["hour_utc"] = rng.integers(0, 24, n).astype(float)
    rows["season"] = np.where(signal > 0.5, "summer", "winter").astype("<U8")
    truth = {
        lead: (
            (signal > 0.5).astype(float),
            np.ones(n, dtype=bool),
        )
        for lead in LEADS
    }
    return rows, truth


def _model() -> pp.PostprocessModel:
    rows, truth = _training_rows()
    return pp.fit_postprocess(
        rows, truth, LEADS, l2=1.0, design_leads=LEADS,
        fitted_at=datetime(2026, 9, 11, 3, 40, tzinfo=timezone.utc),
    )


def _write_model(path: Path, model: pp.PostprocessModel | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((model or _model()).dumps())
    return path


# ---------------------------------------------------------------------------
# A cycle without a cycle: grids, a flow, a geo, three points
# ---------------------------------------------------------------------------

GRID_PX = 64
PIXEL_KM = 0.5
DOWNSAMPLE = 4


class FakeGeo:
    """``(lon, lat)`` → a fractional native index. Nothing else."""

    def __init__(self, mapping: dict[tuple[float, float], tuple[float, float]]):
        self._mapping = mapping

    def lonlat_to_grid(self, lon: float, lat: float) -> GridIndex:
        row, col = self._mapping[(round(lon, 4), round(lat, 4))]
        return GridIndex(row=row, col=col)


POINTS: tuple[tuple[str, float, float, tuple[float, float]], ...] = (
    # id, lat, lon, native (row, col)
    ("home", HOME_LAT, HOME_LON, (20.0, 20.0)),
    ("06180", 55.614, 12.6454, (32.0, 36.0)),
    ("06120", 55.4735, 10.3297, (44.0, 12.0)),
)


def _geo() -> FakeGeo:
    return FakeGeo({
        (round(lon, 4), round(lat, 4)): native
        for _id, lat, lon, native in POINTS
    })


def _rain_field() -> np.ndarray:
    """A band of rain in the north-west quadrant, and dry elsewhere."""
    field = np.zeros((GRID_PX, GRID_PX), dtype=np.float32)
    rows, cols = np.mgrid[0:GRID_PX, 0:GRID_PX]
    band = (rows + cols > 20) & (rows + cols < 40)
    field[band] = 3.5
    field[rows > 56] = np.nan          # off the composite, to exercise NaN
    return field


def _flow() -> tuple[np.ndarray, np.ndarray]:
    """A uniform south-easterly drift of 2 px per frame."""
    vy = np.full((GRID_PX, GRID_PX), 2.0, dtype=np.float32)
    vx = np.full((GRID_PX, GRID_PX), 1.5, dtype=np.float32)
    return vy, vx


def _grid_features() -> dict[str, np.ndarray]:
    field = _rain_field()
    vy, vx = _flow()
    native = [_geo().lonlat_to_grid(lon, lat) for _i, lat, lon, _n in POINTS]
    return pp.station_features(
        field, vy, vx,
        np.array([idx.row for idx in native], dtype=np.float64),
        np.array([idx.col for idx in native], dtype=np.float64),
        pixel_km=PIXEL_KM, dt_min=10.0,
        bulk_vy=2.0, bulk_vx=1.5, stalled_share=0.011,
    )


def _products(p_rain: float = 0.7) -> NationalProducts:
    size = GRID_PX // DOWNSAMPLE
    def grid(value: float) -> np.ndarray:
        return np.full((size, size), value, dtype=np.float32)
    return NationalProducts(
        p_rain={
            lead: grid(p_rain * (0.5 + index / 10))
            for index, lead in enumerate(LEADS)
        },
        eta_min=grid(25.0),
        intensity_mm_h=grid(1.8),
        leads_min=LEADS,
        threshold_mm_h=0.5,
        timestep_min=10.0,
        frame_age_min=14.0,
        downsample_factor=DOWNSAMPLE,
        n_members=16,
    )


def _cycle(table: PostprocessTable, products=None) -> CyclePostprocess:
    """What ``CycleEngine._publish_postprocess`` builds, assembled by hand."""
    products = products if products is not None else _products()
    geo = _geo()
    native = [geo.lonlat_to_grid(lon, lat) for _i, lat, lon, _n in POINTS]
    point_products = compute_mod._read_points(products, native)
    assert point_products is not None
    observed = np.full(products.eta_min.shape, 0.2, dtype=np.float32)
    shared = []
    for pixel in point_products.pixels:
        assert pixel is not None
        row, col = pixel
        shared.append({
            "observed_mm_h": float(observed[row, col]),
            "eta_min": float(products.eta_min[row, col]),
            "intensity_mm_h": float(products.intensity_mm_h[row, col]),
        })
    return build_cycle_postprocess(
        table,
        radar_ts_utc=RADAR_TS,
        generated_at_utc=GENERATED_AT,
        keys=[point_key(lat, lon) for _i, lat, lon, _n in POINTS],
        grid_features=_grid_features(),
        raw_fractions=point_products.raw_fractions,
        shared=shared,
        station_radar_km=[
            float(rw.station_radar_km(lat, lon)) for _i, lat, lon, _n in POINTS
        ],
        leads=products.leads_min,
        season=pp.season_of_month(GENERATED_AT.month),
        hour_utc=GENERATED_AT.hour,
        frame_age_min=14.0,
    )


# ---------------------------------------------------------------------------
# 1. Parity with the replay
# ---------------------------------------------------------------------------


class TestParityWithTheReplay:
    def test_the_runtime_row_equals_the_replays_row_bit_for_bit(self) -> None:
        """Same grids, same flow, same point → the same stored row.

        This is the one that matters. The model is fitted on rows the
        replay wrote and applied to rows the cycle writes; if the two
        assemblies drift the service does not fail, it silently predicts
        on features that mean something else.
        """
        products = _products()
        geo = _geo()
        grid_features = _grid_features()
        native = [geo.lonlat_to_grid(lon, lat) for _i, lat, lon, _n in POINTS]
        raw_p_rain = dict(products.p_rain)

        runtime = _cycle(PostprocessTable(None))

        for index, (station, lat, lon, _n) in enumerate(POINTS):
            pixel = compute_mod.product_pixel_of(
                products, native[index].row, native[index].col,
            )
            replay = rw._feature_row(
                grid_features,
                index,
                rw.StationPoint(id=station, lat=lat, lon=lon, region=None),
                raw_p_rain=raw_p_rain,
                pixel=pixel,
                leads_min=products.leads_min,
                season=pp.season_of_month(GENERATED_AT.month),
                hour_utc=GENERATED_AT.hour,
                frame_age_min=14.0,
            )
            assert runtime.rows[index] == replay
            # Key ORDER too: the parquet writers append columns in this
            # order and a reordered row would write a different schema.
            assert list(runtime.rows[index]) == list(replay)

    def test_the_row_carries_every_feature_column_the_schema_names(self) -> None:
        runtime = _cycle(PostprocessTable(None))
        assert set(runtime.rows[0]) == set(pp.feature_schema(LEADS).names)

    def test_a_point_off_the_product_grid_reads_no_raw_fraction(self) -> None:
        """Off coverage is unknown, never 0 %."""
        products = _products()
        far = compute_mod._read_points(
            products, [GridIndex(row=10_000.0, col=10_000.0)],
        )
        assert far is not None
        assert far.pixels == (None,)
        assert all(values == (None,) for values in far.raw_fractions.values())


# ---------------------------------------------------------------------------
# 2. The reader
# ---------------------------------------------------------------------------


class TestPostprocessTable:
    def test_a_fitted_model_is_active_and_names_its_leads(
        self, tmp_path: Path,
    ) -> None:
        table = PostprocessTable(_write_model(tmp_path / "m.json"))
        table.load()
        assert table.active is True
        assert table.leads == list(LEADS)
        assert table.design_leads == list(LEADS)
        assert table.fitted_at_utc == "2026-09-11T03:40:00+00:00"

    def test_a_missing_file_is_inactive_and_one_log_line(
        self, tmp_path: Path,
    ) -> None:
        table = PostprocessTable(tmp_path / "nothing.json")
        with structlog.testing.capture_logs() as logs:
            assert table.load() is None
        assert [e["event"] for e in logs] == ["push_postprocess_missing"]
        assert table.active is False
        assert table.fitted_at_utc is None
        assert table.leads == []

    @pytest.mark.parametrize(
        "body",
        [
            "{not json",
            json.dumps({"schema_version": 99, "models": {}}),
            json.dumps({"hello": "world"}),
        ],
    )
    def test_a_broken_file_is_inactive_not_half_a_model(
        self, tmp_path: Path, body: str,
    ) -> None:
        path = tmp_path / "m.json"
        path.write_text(body)
        table = PostprocessTable(path)
        with structlog.testing.capture_logs() as logs:
            table.load()
        assert table.active is False
        assert [e["event"] for e in logs] == ["push_postprocess_unusable"]
        assert table.predict(30, {"observed_mm_h": 1.0}) is None

    def test_a_model_with_no_fitted_lead_is_inactive(self, tmp_path: Path) -> None:
        doc = _model().to_json()
        doc["models"] = {}
        path = tmp_path / "m.json"
        path.write_text(json.dumps(doc))
        table = PostprocessTable(path)
        table.load()
        assert table.active is False

    def test_a_replaced_file_is_re_read_at_the_next_reload(
        self, tmp_path: Path,
    ) -> None:
        path = _write_model(tmp_path / "m.json")
        table = PostprocessTable(path)
        assert table.maybe_reload() is True
        assert table.fitted_at_utc == "2026-09-11T03:40:00+00:00"
        assert table.maybe_reload() is False        # unchanged: no parse

        _write_model(path, pp.fit_postprocess(
            *_training_rows(), LEADS, l2=1.0, design_leads=LEADS,
            fitted_at=datetime(2026, 9, 12, 3, 40, tzinfo=timezone.utc),
        ))
        assert table.maybe_reload() is True
        assert table.fitted_at_utc == "2026-09-12T03:40:00+00:00"

    def test_note_changed_forces_a_re_read(self, tmp_path: Path) -> None:
        """A same-size rewrite inside one filesystem timestamp tick.

        The sync task and the nightly refit both call ``note_changed``
        after writing, precisely so a stamp that did not appear to move
        cannot strand the process on last night's model.
        """
        import os

        path = _write_model(tmp_path / "m.json")
        table = PostprocessTable(path)
        table.maybe_reload()
        stat = path.stat()
        _write_model(path, pp.fit_postprocess(
            *_training_rows(), LEADS, l2=1.0, design_leads=LEADS,
            fitted_at=datetime(2026, 9, 12, 3, 40, tzinfo=timezone.utc),
        ))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        table.note_changed()
        assert table.maybe_reload() is True
        assert table.fitted_at_utc == "2026-09-12T03:40:00+00:00"

    def test_the_default_path_is_beside_the_other_published_artefacts(
        self, minimal_config: Config,
    ) -> None:
        assert resolved_postprocess_path(minimal_config) == (
            minimal_config.storage.data_dir / POSTPROCESS_FILENAME
        )
        minimal_config.push.postprocess_path = Path("/somewhere/else.json")
        assert resolved_postprocess_path(minimal_config) == Path(
            "/somewhere/else.json",
        )

    def test_one_point_and_a_whole_table_give_the_same_number(
        self, tmp_path: Path,
    ) -> None:
        """``predict`` is ``predict_table`` of one row, not a second path."""
        table = PostprocessTable(_write_model(tmp_path / "m.json"))
        table.load()
        cycle = _cycle(table)
        row = dict(cycle.rows[1])
        assert cycle.p_post[30][1] == pytest.approx(
            table.predict(30, row), rel=1e-9, abs=1e-12,
        )


# ---------------------------------------------------------------------------
# 3. The cycle's answer
# ---------------------------------------------------------------------------


class TestCyclePostprocess:
    def test_it_scores_every_point_when_the_model_is_active(
        self, tmp_path: Path,
    ) -> None:
        table = PostprocessTable(_write_model(tmp_path / "m.json"))
        table.load()
        cycle = _cycle(table)
        assert cycle.active is True
        assert cycle.leads == LEADS
        for lead in LEADS:
            assert len(cycle.p_post[lead]) == len(POINTS)
            assert all(0.0 <= p <= 1.0 for p in cycle.p_post[lead])
        assert cycle.fitted_at_utc == "2026-09-11T03:40:00+00:00"

    def test_without_a_model_it_still_carries_the_features(self) -> None:
        """The rows are what the nightly refit trains on. They must not
        depend on there being a model yet — that is the bootstrap."""
        cycle = _cycle(PostprocessTable(None))
        assert cycle.active is False
        assert cycle.p_post == {}
        assert cycle.fitted_at_utc is None
        assert len(cycle.rows) == len(POINTS)
        assert cycle.probability(HOME_LAT, HOME_LON, 30) is None

    def test_a_point_the_cycle_did_not_serve_is_unknown(
        self, tmp_path: Path,
    ) -> None:
        table = PostprocessTable(_write_model(tmp_path / "m.json"))
        table.load()
        cycle = _cycle(table)
        assert cycle.probability(1.0, 1.0, 30) is None
        assert cycle.features(1.0, 1.0) is None
        assert cycle.columns(1.0, 1.0) == {}

    def test_an_unfitted_lead_is_unknown_rather_than_interpolated(
        self, tmp_path: Path,
    ) -> None:
        table = PostprocessTable(_write_model(tmp_path / "m.json"))
        table.load()
        cycle = _cycle(table)
        assert cycle.probability(HOME_LAT, HOME_LON, 25) is None

    def test_the_columns_are_the_features_plus_p_post(
        self, tmp_path: Path,
    ) -> None:
        table = PostprocessTable(_write_model(tmp_path / "m.json"))
        table.load()
        cycle = _cycle(table)
        columns = cycle.columns(HOME_LAT, HOME_LON)
        assert set(columns) == (
            set(pp.feature_schema(LEADS).names)
            | {pp.post_column(lead) for lead in LEADS}
        )


# ---------------------------------------------------------------------------
# 4. The decision engine
# ---------------------------------------------------------------------------


def _obs(**over) -> Observation:
    kwargs = {
        "radar_ts_utc": RADAR_TS,
        "p_rain": 0.30,
        "eta_min": 25.0,
        "intensity_mm_h": 1.2,
        "observed_mm_h": 0.0,
        "forecast_now_mm_h": 0.0,
    }
    kwargs.update(over)
    return Observation(**kwargs)  # type: ignore[arg-type]


class TestTheEngineReadsTheChosenProbability:
    def test_curve_is_the_default_and_reads_p_rain(self) -> None:
        obs = _obs(p_post=0.99)
        assert obs.p_source == "curve"
        assert obs.p_decision == 0.30
        assert obs.p_decision_source == "curve"

    def test_postprocess_reads_p_post(self) -> None:
        obs = _obs(p_post=0.71, p_source="postprocess")
        assert obs.p_decision == 0.71
        assert obs.p_decision_source == "postprocess"

    def test_a_missing_p_post_falls_back_to_the_curve_for_that_row(self) -> None:
        obs = _obs(p_post=None, p_source="postprocess")
        assert obs.p_decision == 0.30
        assert obs.p_decision_source == "curve"

    def test_the_rule_fires_on_the_post_processed_number(self) -> None:
        """Curve 0.30, model 0.71, threshold 45 %: the model fires.

        Persistence is 1 here so one observation decides; the point is
        which number the comparison is against, not the streak.
        """
        rules = Rules(persistence_obs=1)
        curve = evaluate(
            INITIAL_STATE, _obs(p_post=0.71), threshold_pct=45,
            quiet=None, tz="UTC", now_utc=GENERATED_AT, rules=rules,
        )
        assert curve.action == "none"

        post = evaluate(
            INITIAL_STATE, _obs(p_post=0.71, p_source="postprocess"),
            threshold_pct=45, quiet=None, tz="UTC", now_utc=GENERATED_AT,
            rules=rules,
        )
        assert post.action == "notify"

    def test_it_can_also_hold_a_warning_the_curve_would_have_sent(self) -> None:
        rules = Rules(persistence_obs=1)
        obs = _obs(p_rain=0.80, p_post=0.10, p_source="postprocess")
        assert evaluate(
            INITIAL_STATE, obs, threshold_pct=45, quiet=None, tz="UTC",
            now_utc=GENERATED_AT, rules=rules,
        ).action == "none"


# ---------------------------------------------------------------------------
# 5. The fan-out
# ---------------------------------------------------------------------------


class _Sample:
    def __init__(self, p_rain: float, lead: int) -> None:
        self.p_rain = {lead: p_rain}
        self.eta_min = 25.0
        self.intensity_mm_h = 1.2
        self.observed_mm_h = 0.0
        self.forecast_mm_h = {0: 0.0}


class _CycleStub:
    """Only what ``PushService`` reads off the engine."""

    def __init__(self, cycle: CyclePostprocess | None) -> None:
        self.postprocess_latest = cycle


def _pinned(
    base: CyclePostprocess,
    p_post: dict[int, tuple[float | None, ...]],
    *,
    radar_ts: datetime = RADAR_TS,
    fitted_at: str = "2026-09-11T03:40:00+00:00",
) -> CyclePostprocess:
    """``base``'s points and rows with the model's answer pinned.

    The model's own number on synthetic noise is not the thing under
    test — which probability the fan-out reads is.
    """
    return CyclePostprocess(
        radar_ts_utc=radar_ts,
        generated_at_utc=radar_ts + timedelta(minutes=14),
        keys=base.keys,
        rows=base.rows,
        p_post=p_post,
        fitted_at_utc=fitted_at,
    )


@pytest.fixture
def push_config(minimal_config: Config) -> Config:
    minimal_config.push = PushConfig(
        enabled=True, vapid_subject=SUBJECT, lead_options=[20, 30, 45, 60],
    )
    return minimal_config


def _service(config: Config, engine, monkeypatch, p_rain: float = 0.30):
    monkeypatch.setattr(
        service_mod, "sample_point",
        lambda products, geo, lat, lon, **kw: _Sample(p_rain, 30),
    )
    monkeypatch.setattr(
        service_mod.fanout, "send",
        lambda **kw: type(
            "R", (), {"ok": True, "gone": False, "status": 201, "error": None},
        )(),
    )
    store = PushStore(resolved_db_path(config))
    return PushService(
        config, engine=engine, store=store, vapid_private_pem=b"pem",
        public_key="key",
        thresholds=ThresholdTable(resolved_thresholds_path(config)),
    )


def _subscribe(service: PushService, endpoint: str = ENDPOINT_A, **over) -> None:
    kwargs = {
        "endpoint": endpoint, "p256dh": P256DH, "auth": AUTH,
        "lat": HOME_LAT, "lon": HOME_LON,
        "threshold_pct": 45, "lead_min": 30,
        "quiet_enabled": False, "quiet_start": "22:00",
        "quiet_end": "07:00", "tz": "Europe/Copenhagen", "lang": "da",
    }
    kwargs.update(over)
    service.store.upsert(NewSubscription(**kwargs))  # type: ignore[arg-type]


def _run(service: PushService, ts: datetime = RADAR_TS) -> tuple[list, dict]:
    with structlog.testing.capture_logs() as logs:
        summary = service._evaluate_and_send(None, None, ts, ts)
    return [e for e in logs if e["event"] == "push_eval"], summary


class TestTheFanOutUsesTheModel:
    def test_it_decides_on_p_post_and_logs_both_numbers(
        self, push_config: Config, tmp_path: Path, monkeypatch,
    ) -> None:
        table = PostprocessTable(_write_model(tmp_path / "m.json"))
        table.load()
        base = _cycle(table)
        answer = {30: (0.71, 0.10, 0.10)}
        engine = _CycleStub(_pinned(base, answer))
        service = _service(push_config, engine, monkeypatch)
        _subscribe(service)
        lines, summary = _run(service)

        assert len(lines) == 1
        line = lines[0]
        assert line["p_source"] == "postprocess"
        assert line["p_rain"] == pytest.approx(0.30)
        assert line["p_post"] == pytest.approx(0.71)
        assert line["action"] == "none"     # persistence is 2 by default
        assert summary["probability_source"] == "postprocess"
        assert summary["postprocess_active"] is True
        assert summary["postprocess_fitted_at"] == "2026-09-11T03:40:00+00:00"
        assert summary["postprocess_curve_fallbacks"] == 0

        # Second observation: the streak completes and the push goes out.
        # The curve never crosses 45 %, so the notification is the model's.
        later = RADAR_TS + timedelta(minutes=10)
        engine.postprocess_latest = _pinned(base, answer, radar_ts=later)
        lines, summary = _run(service, later)
        assert lines[0]["action"] == "notify"
        assert summary["notified"] == 1

    def test_a_point_the_model_cannot_speak_for_falls_back_and_is_counted(
        self, push_config: Config, tmp_path: Path, monkeypatch,
    ) -> None:
        table = PostprocessTable(_write_model(tmp_path / "m.json"))
        table.load()
        base = _cycle(table)
        # Home is unknown to the model; the other two points are not.
        cycle = _pinned(base, {30: (None, 0.5, 0.5)})
        service = _service(push_config, _CycleStub(cycle), monkeypatch)
        _subscribe(service)
        lines, summary = _run(service)
        assert lines[0]["p_source"] == "curve"
        assert lines[0]["p_post"] is None
        assert lines[0]["p_rain"] == pytest.approx(0.30)
        assert summary["postprocess_curve_fallbacks"] == 1

    def test_a_stale_cycle_object_is_refused(
        self, push_config: Config, tmp_path: Path, monkeypatch,
    ) -> None:
        """One frame's features must never be read against another's."""
        table = PostprocessTable(_write_model(tmp_path / "m.json"))
        table.load()
        base = _cycle(table)
        stale = _pinned(
            base, {30: (0.99, 0.99, 0.99)},
            radar_ts=RADAR_TS - timedelta(minutes=10),
        )
        service = _service(push_config, _CycleStub(stale), monkeypatch)
        _subscribe(service)
        lines, summary = _run(service)
        assert lines[0]["p_source"] == "curve"
        assert summary["postprocess_active"] is False

    def test_the_message_carries_the_probability_the_rule_used(
        self, push_config: Config, tmp_path: Path, monkeypatch,
    ) -> None:
        table = PostprocessTable(_write_model(tmp_path / "m.json"))
        table.load()
        base = _cycle(table)
        answer = {30: (0.71, 0.1, 0.1)}
        engine = _CycleStub(_pinned(base, answer))
        service = _service(push_config, engine, monkeypatch)
        _subscribe(service)
        sent: list[dict] = []
        monkeypatch.setattr(
            service_mod.PushService, "_fanout",
            lambda self, pending: (
                sent.extend(payload for _sub, payload in pending)
                or {"sent": len(pending), "failed": 0,
                    "removed": 0, "skipped": 0}
            ),
        )
        _run(service)
        later = RADAR_TS + timedelta(minutes=10)
        engine.postprocess_latest = _pinned(base, answer, radar_ts=later)
        _run(service, later)
        assert len(sent) == 1
        # 0.71 rounds to 71 %; the curve's 0.30 would have read 30 %.
        assert sent[0]["p_pct"] == 71

    def test_curve_restores_the_previous_behaviour_exactly(
        self, push_config: Config, tmp_path: Path, monkeypatch,
    ) -> None:
        """The rollback: one config key, and the model is not consulted."""
        push_config.push.probability_source = "curve"
        table = PostprocessTable(_write_model(tmp_path / "m.json"))
        table.load()
        base = _cycle(table)
        cycle = _pinned(base, {30: (0.99, 0.99, 0.99)})
        service = _service(push_config, _CycleStub(cycle), monkeypatch)
        _subscribe(service)
        lines, summary = _run(service)
        assert lines[0]["p_source"] == "curve"
        assert lines[0]["p_post"] is None
        assert lines[0]["action"] == "none"
        assert summary["probability_source"] == "curve"
        assert summary["postprocess_active"] is False

    def test_an_engine_without_the_attribute_is_simply_the_curve(
        self, push_config: Config, monkeypatch,
    ) -> None:
        service = _service(push_config, SimpleNamespace(), monkeypatch)
        _subscribe(service)
        lines, summary = _run(service)
        assert lines[0]["p_source"] == "curve"
        assert summary["postprocess_active"] is False


# ---------------------------------------------------------------------------
# 6. The cycle's point sources
# ---------------------------------------------------------------------------


class TestServingPoints:
    def test_home_is_always_scored_and_duplicates_collapse(
        self, minimal_config: Config,
    ) -> None:
        engine = compute_mod.CycleEngine(minimal_config)
        engine.add_point_source("a", lambda: [(HOME_LAT, HOME_LON), (56.0, 11.0)])
        engine.add_point_source("b", lambda: [(56.0, 11.0), (57.0, 12.0)])
        assert engine._serving_points() == [
            point_key(HOME_LAT, HOME_LON),
            point_key(56.0, 11.0),
            point_key(57.0, 12.0),
        ]

    def test_a_source_that_raises_costs_its_own_points_and_nothing_else(
        self, minimal_config: Config,
    ) -> None:
        def boom():
            raise RuntimeError("the points file is gone")

        engine = compute_mod.CycleEngine(minimal_config)
        engine.add_point_source("broken", boom)
        engine.add_point_source("fine", lambda: [(56.0, 11.0)])
        with structlog.testing.capture_logs() as logs:
            points = engine._serving_points()
        assert points == [point_key(HOME_LAT, HOME_LON), point_key(56.0, 11.0)]
        assert any(e["event"] == "postprocess_points_failed" for e in logs)

    def test_the_point_count_is_capped(self, minimal_config: Config) -> None:
        engine = compute_mod.CycleEngine(minimal_config)
        engine.add_point_source(
            "many",
            lambda: [(50.0 + i * 1e-4, 10.0) for i in range(5000)],
        )
        with structlog.testing.capture_logs() as logs:
            points = engine._serving_points()
        assert len(points) == compute_mod._MAX_POSTPROCESS_POINTS
        assert any(e["event"] == "postprocess_points_truncated" for e in logs)

    def test_no_source_means_home_alone(self, minimal_config: Config) -> None:
        engine = compute_mod.CycleEngine(minimal_config)
        assert engine._serving_points() == [point_key(HOME_LAT, HOME_LON)]


# ---------------------------------------------------------------------------
# 7. The live decision rows
# ---------------------------------------------------------------------------


STATION_POINTS = {
    "version": 2,
    "points": [
        {"id": s_id, "lat": lat, "lon": lon, "region": "dk"}
        for s_id, lat, lon, _n in POINTS[1:]
    ],
}


@pytest.fixture
def eval_config(tmp_path: Path) -> Config:
    points = tmp_path / "station_points.json"
    points.write_text(json.dumps(STATION_POINTS))
    return Config(
        home={"lat": HOME_LAT, "lon": HOME_LON},  # type: ignore[arg-type]
        calibration={  # type: ignore[arg-type]
            "curves_path": tmp_path / "curves.json",
            "national_curves_path": tmp_path / "national_curves.json",
        },
        storage={  # type: ignore[arg-type]
            "data_dir": tmp_path / "data", "corpus_dir": tmp_path / "corpus",
        },
        lightning={"archive_dir": tmp_path / "strikes"},  # type: ignore[arg-type]
        station_eval={  # type: ignore[arg-type]
            "enabled": True, "points_file": str(points),
        },
    )


class _EvalEngine:
    """``national_latest`` + ``geo`` + ``postprocess_latest``, nothing else."""

    def __init__(self, cycle: CyclePostprocess | None, products) -> None:
        snap = type("Snap", (tuple,), {})((products, RADAR_TS))
        snap.observed_mm_h = np.full(
            products.eta_min.shape, 0.0, dtype=np.float32,
        )
        snap.forecast_mm_h = {
            0: np.full(products.eta_min.shape, 0.0, dtype=np.float32),
        }
        snap.generated_at_utc = GENERATED_AT
        self.national_latest = snap
        self.geo = _geo()
        self.postprocess_latest = cycle


def _cycle_result(radar_ts: datetime = RADAR_TS):
    return SimpleNamespace(
        state=SimpleNamespace(radar=SimpleNamespace(latest_ts=radar_ts)),
        error=None,
    )


def _run_station_eval(config: Config, cycle: CyclePostprocess | None):
    import anyio

    from dmi_nowcast_sidecar.station_eval import StationEvalService, partition_path

    service = StationEvalService(config, _EvalEngine(cycle, _products()))
    anyio.run(service.after_cycle, _cycle_result())
    import pyarrow.parquet as pq

    return pq.read_table(partition_path(config, RADAR_TS))


class TestTheLiveDecisionRows:
    def test_they_carry_the_features_and_the_served_probability(
        self, eval_config: Config, tmp_path: Path,
    ) -> None:
        table = PostprocessTable(_write_model(tmp_path / "m.json"))
        table.load()
        cycle = _pinned(_cycle(table), {30: (0.11, 0.71, 0.22)})
        rows = _run_station_eval(eval_config, cycle).to_pylist()

        assert {r["station_id"] for r in rows} == {"06180", "06120"}
        by_station = {r["station_id"]: r for r in rows}
        # Row order in ``p_post`` follows the cycle's points: home first,
        # then 06180, then 06120.
        assert by_station["06180"]["p_post_30"] == pytest.approx(0.71)
        assert by_station["06120"]["p_post_30"] == pytest.approx(0.22)
        # And the features the nightly refit trains on.
        for name in pp.feature_schema(LEADS).names:
            assert name in by_station["06180"]
        assert by_station["06180"]["season"] == "summer"
        assert by_station["06180"]["hour_utc"] == GENERATED_AT.hour
        assert by_station["06180"]["stalled_share"] == pytest.approx(0.011)

    def test_the_decision_is_taken_on_the_served_probability(
        self, eval_config: Config, tmp_path: Path,
    ) -> None:
        """The scoreboard measures the rule the SERVICE runs.

        p_rain is 0.7 * 0.7 = 0.49 at lead 30 on this fixture, so the
        curve would be over the 40 % rule and the model's 0.11 is not.
        """
        eval_config.station_eval.rules.threshold_pct = 40
        table = PostprocessTable(_write_model(tmp_path / "m.json"))
        table.load()
        cycle = _pinned(_cycle(table), {30: (0.9, 0.11, 0.11)})
        rows = _run_station_eval(eval_config, cycle).to_pylist()
        assert {r["action"] for r in rows} == {"none"}

        eval_config.push.probability_source = "curve"
        import shutil

        shutil.rmtree(
            eval_config.storage.corpus_dir, ignore_errors=True,
        )
        rows = _run_station_eval(eval_config, cycle).to_pylist()
        # persistence_obs is 1 for the scoreboard, so the curve fires at
        # once on exactly the same inputs.
        assert {r["action"] for r in rows} == {"notify"}

    def test_without_a_cycle_object_the_rows_are_the_old_ones_plus_nulls(
        self, eval_config: Config,
    ) -> None:
        rows = _run_station_eval(eval_config, None).to_pylist()
        assert all(row["p_post_30"] is None for row in rows)
        assert all(row["obs_max_5km_mm_h"] is None for row in rows)

    def test_every_existing_reader_of_the_schema_is_untouched(
        self, eval_config: Config, tmp_path: Path,
    ) -> None:
        """The columns are additive: alignment drops them, the sweep asks."""
        from dmi_nowcast_core.warning_score import (
            align_decision_table,
            decision_columns,
        )
        from dmi_nowcast_sidecar.threshold_sweep import load_decisions

        table = PostprocessTable(_write_model(tmp_path / "m.json"))
        table.load()
        cycle = _pinned(_cycle(table), {30: (0.11, 0.71, 0.22)})
        written = _run_station_eval(eval_config, cycle)

        aligned = align_decision_table(written, LEADS)
        assert tuple(aligned.schema.names) == decision_columns(LEADS)

        rows, leads, counts = load_decisions(
            [eval_config.storage.corpus_dir / "stations" / "eval"],
            extra_columns=["p_post_30"],
        )
        assert counts["files"] == 1
        assert leads == LEADS
        assert sorted(
            round(row["p_post_30"], 4) for row in rows
        ) == [0.22, 0.71]


# ---------------------------------------------------------------------------
# 8. The published artefact and what /options says about it
# ---------------------------------------------------------------------------


API_KEY = "operator-key"


@pytest.fixture
def served_config(push_config: Config) -> Config:
    push_config.server.public_mode = True
    push_config.server.api_key = API_KEY
    return push_config


@pytest.fixture
def served_client(served_config: Config):
    from fastapi.testclient import TestClient

    from dmi_nowcast_sidecar.app import create_app

    app = create_app(served_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        yield c


class TestTheServedModel:
    def test_the_route_is_private_and_serves_the_file(
        self, served_config: Config, served_client,
    ) -> None:
        # Nothing fitted yet: 503, not a 200 with an empty body.
        assert served_client.get(
            "/calibration/postprocess.json",
            headers={"Authorization": f"Bearer {API_KEY}"},
        ).status_code == 503

        _write_model(resolved_postprocess_path(served_config))
        r = served_client.get(
            "/calibration/postprocess.json",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        assert r.status_code == 200
        assert r.json()["schema_version"] == pp.SCHEMA_VERSION
        assert r.headers["cache-control"] == "public, max-age=300"

    def test_it_is_hidden_from_the_public_surface(self, served_client) -> None:
        """Like the curves and the thresholds: pulled, never republished."""
        assert served_client.get(
            "/calibration/postprocess.json",
        ).status_code == 404

    def test_options_reports_the_model_the_fan_out_would_use(
        self, served_config: Config, served_client,
    ) -> None:
        body = served_client.get("/api/push/options").json()
        # Configured for the model, but there is none: the honest answer
        # is what a notification would actually be decided on.
        assert body["probability_source"] == "curve"
        assert body["postprocess_fitted_at_utc"] is None

        _write_model(resolved_postprocess_path(served_config))
        body = served_client.get("/api/push/options").json()
        assert body["probability_source"] == "postprocess"
        assert body["postprocess_fitted_at_utc"] == "2026-09-11T03:40:00+00:00"

    def test_the_rollback_shows_in_options_too(
        self, served_config: Config, served_client,
    ) -> None:
        _write_model(resolved_postprocess_path(served_config))
        served_config.push.probability_source = "curve"
        body = served_client.get("/api/push/options").json()
        assert body["probability_source"] == "curve"
        assert body["postprocess_fitted_at_utc"] is None

    def test_the_sync_task_nudges_the_engine_to_re_read(
        self, minimal_config: Config, tmp_path: Path,
    ) -> None:
        from dmi_nowcast_sidecar.sync import POSTPROCESS_FILE, build_artifact_sync

        minimal_config.sync.enabled = True
        minimal_config.sync.source_url = "http://private:8081"
        engine = compute_mod.CycleEngine(minimal_config)
        sync = build_artifact_sync(minimal_config, engine)
        assert sync is not None
        _write_model(resolved_postprocess_path(minimal_config))
        engine.postprocess.maybe_reload()
        assert engine.postprocess.active is True

        # A replaced file inside one timestamp tick: only the nudge saves it.
        sync._on_file_updated(
            POSTPROCESS_FILE, resolved_postprocess_path(minimal_config),
        )
        assert engine.postprocess._dirty is True
