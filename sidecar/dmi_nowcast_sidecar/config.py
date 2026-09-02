"""Sidecar configuration.

Loaded from a YAML file (path in ``DMI_NOWCAST_CONFIG``, default
``./config.yaml``) and overridable via environment variables with the
prefix ``DMI_NOWCAST_`` and ``__`` as the nested-field separator —
e.g. ``DMI_NOWCAST_SERVER__PORT=9000`` overrides ``server.port``.

This module is intentionally self-contained: it does no I/O until
:func:`load_config` is called, so tests can construct ``Config`` objects
directly without touching the filesystem.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class HomeConfig(BaseModel):
    """The geographic point we're nowcasting for.

    ``radius_km`` is the disc over which the integration samples max/mean/
    P90 rain rate — 1 km matches the radar's effective resolution.
    """
    lat: Annotated[float, Field(ge=-90, le=90)]
    lon: Annotated[float, Field(ge=-180, le=180)]
    radius_km: Annotated[float, Field(gt=0, le=20)] = 1.0


class PollConfig(BaseModel):
    interval_min: Annotated[int, Field(ge=1, le=60)] = 5
    jitter_sec: Annotated[int, Field(ge=0, le=120)] = 30


class DmiConfig(BaseModel):
    """DMI Open Data API settings.

    ``scan_type`` filters the composite collection server-side. DMI
    interleaves two products — **fullRange** at minutes :x0 and doppler at
    :x5 — and doppler covers only ~40% of fullRange's area while seeing
    ~25% more echo within the joint area, so mixing them alternates the
    input between materially different views (website Phase B plan,
    addendum 2026-08-29). The runtime is fullRange-only by decision; the
    ``Literal`` makes that non-configurable while keeping the setting
    visible where the calibration corpus builder derives its parity
    settings from (plan §B0/§C1: corpus settings derive from runtime
    config — this field IS the runtime scan type).
    """
    base_url: str = "https://opendataapi.dmi.dk/v1/radardata"
    user_agent: str = "dmi-nowcast-sidecar/0.1"
    scan_type: Literal["fullRange"] = "fullRange"


class StepsConfig(BaseModel):
    """STEPS ensemble settings (website Phase A plan §A0).

    ``enabled`` gates the per-cycle ensemble; when off — or when the run
    fails — the cycle emits exactly the deterministic-only state (the new
    ensemble fields stay at their None/0 defaults). ``downsample_factor``
    stride-slices the national grid before STEPS: ×4 turns 1728×1984 @
    500 m into 432×496 @ ~2 km effective, cutting a ~2 min run to ~6 s on
    dev hardware while staying finer than the radar's true resolution.
    """
    enabled: bool = True
    ensemble_size: Annotated[int, Field(ge=4, le=64)] = 24
    n_cascade_levels: Annotated[int, Field(ge=4, le=8)] = 6
    downsample_factor: Annotated[int, Field(ge=1, le=8)] = 4


class NationalConfig(BaseModel):
    """National grid products + artifacts (website Phase A plan §A1/§A2).

    ``enabled`` gates the ×4 product grids (probability/ETA/intensity,
    reduced from the STEPS ensemble — so it needs ``steps.enabled`` to have
    any effect) and their artifact writing under ``data_dir/nowcast/``.
    ``leads_min`` are the probability-grid leads; ``keep_cycles`` is the
    artifact retention (decided: 24 ≈ 2 h at the 5-min cadence).
    """
    enabled: bool = True
    leads_min: list[int] = Field(default_factory=lambda: [10, 20, 30, 45, 60])
    keep_cycles: Annotated[int, Field(ge=1, le=288)] = 24

    @field_validator("leads_min")
    @classmethod
    def _national_leads_valid(cls, v: list[int]) -> list[int]:
        if not v or v != sorted(v) or any(lead <= 0 or lead > 180 for lead in v):
            raise ValueError("national leads_min must be ascending, in (0, 180]")
        return v


class ForecastConfig(BaseModel):
    leads_min: list[int] = Field(default_factory=lambda: [5, 10, 15, 20, 25, 30, 45, 60])
    method: Literal["farneback", "tvl1", "mean-motion"] = "farneback"
    steps: StepsConfig = Field(default_factory=StepsConfig)
    national: NationalConfig = Field(default_factory=NationalConfig)
    # What counts as "rain" for raining_now and the rain_incoming warning.
    # ``rain_threshold_mm_h`` is the rain rate that must be reached over the home
    # disc; ``detection_stat`` is which disc statistic to test. ``p90`` is robust
    # to a single hot clutter/virga pixel (the DMI composite is column-max, so
    # faint echoes aloft over-read) — ``max`` over-warns, ``mean`` is strictest.
    # 0.5 mm/h ≈ 18 dBZ (genuine light rain), well above the ~7 dBZ that 0.1 hit.
    rain_threshold_mm_h: Annotated[float, Field(gt=0, le=50)] = 0.5
    detection_stat: Literal["max", "p90", "mean"] = "p90"

    @field_validator("leads_min")
    @classmethod
    def _leads_sorted(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("leads_min must not be empty")
        if any(x <= 0 or x > 180 for x in v):
            raise ValueError("leads_min entries must be in (0, 180]")
        if v != sorted(v):
            raise ValueError("leads_min must be sorted ascending")
        return v


class CalibrationConfig(BaseModel):
    # Legacy home-point curves for the binary deterministic forecast
    # (``p_calibrated`` — the HA contract). Untouched by Phase B.
    curves_path: Path = Path("./calibration_curves.json")
    # Pooled national curves fitted on ensemble exceedance fractions
    # (website Phase B plan §B4). Same JSON format as the legacy file;
    # missing/corrupt → raw fractions with ``calibrated: false``, exactly
    # the pre-B4 behaviour. Loaded once at startup, alongside the legacy
    # curves, so calibrate.sh's restart-to-pick-up flow covers both.
    national_curves_path: Path = Path("/var/lib/dmi-nowcast/national_curves.json")


class LightningConfig(BaseModel):
    """Blitzortung lightning-ETA settings (see dmi_nowcast_core.lightning).

    Strikes are POSTed to ``/lightning/strikes`` by HA and held in a rolling
    in-memory buffer; ``/lightning/eta`` estimates when the threatening cell's
    leading edge reaches each ring of a queried target point.
    """
    enabled: bool = True
    buffer_window_min: Annotated[int, Field(ge=5, le=120)] = 30
    # Rings (km) to report ETAs for. Default: imminent (3 km) + early (10 km).
    rings_km: list[float] = Field(default_factory=lambda: [3.0, 10.0])
    min_strikes: Annotated[int, Field(ge=1, le=100)] = 6
    cluster_eps_km: Annotated[float, Field(gt=0, le=100)] = 15.0
    relevance_radius_km: Annotated[float, Field(gt=0, le=300)] = 60.0
    leading_edge_recent_min: Annotated[float, Field(gt=0, le=60)] = 5.0
    min_closing_kmh: Annotated[float, Field(ge=0, le=200)] = 5.0
    min_fit_span_min: Annotated[float, Field(ge=0, le=60)] = 3.0
    # Hard cap on buffered strikes (memory guard).
    max_buffer: Annotated[int, Field(ge=100, le=50000)] = 2000
    # Cross-cycle EMA smoothing of the ETA (closing speed + leading edge), per
    # target. Damps the whipsaw from refitting sparse strikes each cycle.
    # Applied only to /lightning/eta (sensors/alerts); the debug map stays raw.
    smoothing_enabled: bool = True
    smoothing_tau_min: Annotated[float, Field(ge=0, le=60)] = 3.0    # EMA time constant
    smoothing_max_gap_min: Annotated[float, Field(ge=1, le=120)] = 10.0  # reset if older
    # Persist every received strike to an append-only NDJSON archive (one file
    # per UTC day) for backtesting/calibration — self-archived ground truth,
    # since Blitzortung's own archive is participant-gated/non-redistributable.
    archive_enabled: bool = True
    archive_dir: Path = Path("/var/lib/dmi-nowcast-corpus/strikes")


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: Annotated[int, Field(ge=1, le=65535)] = 8081
    # Optional shared secret. When None, write endpoints are open
    # (LAN-trust). When set, write endpoints require Bearer auth.
    api_key: str | None = None
    # Pretty (dev) vs JSON (prod) logs.
    log_format: Literal["pretty", "json"] = "pretty"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Public service mode (website Phase C plan §P1). False keeps the
    # LAN instance exactly as it is. True turns the process into the
    # internet-facing instance: only the static frontend, /healthz,
    # /nowcast/* and /forecast are reachable — every other route
    # (``/state.json``, ``/frames/*``, ``/lightning/*``, the archive
    # dashboards, /docs) answers 404 unless a valid ``api_key`` bearer
    # accompanies the request. See ``app.py``'s docstring for the gate.
    # It also skips the home-crop rendering + OSM basemap fetch in the
    # cycle, since both only feed hidden endpoints.
    public_mode: bool = False
    # Directory of a built static frontend (SvelteKit ``build/``) served
    # at ``/`` with SPA fallback semantics. None (default) serves no
    # frontend at all — the LAN instance is API-only.
    frontend_dir: Path | None = None


class StorageConfig(BaseModel):
    data_dir: Path = Path("/var/lib/dmi-nowcast")
    # LRU eviction cap on the working cache (data_dir/composites). The cache
    # only needs the last ~hour of frames; the persistent record lives in the
    # corpus archive below. Default 500 MB ≈ ~21 days of frames at ~80 KB.
    working_cache_max_bytes: Annotated[int, Field(ge=10 * 1024 * 1024)] = (
        500 * 1024 * 1024
    )
    # Persistent corpus archive. When
    # set, every fetched composite is copied here in addition to the working
    # cache. Use a host bind-mount in compose so the archive survives
    # `docker compose down -v`. ``None`` disables archiving.
    corpus_dir: Path | None = Path("/var/lib/dmi-nowcast-corpus")


# 24-hour ``HH:MM`` wall-clock strings (quiet hours). Local time in the
# subscriber's own zone — never UTC, never a datetime.
_HHMM_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


class PushConfig(BaseModel):
    """Web Push notifications (website Phase D).

    Off by default: the LAN instance has Home Assistant for alerting and
    needs none of this. Turning it on makes the process hold a VAPID
    private key and a SQLite table of browser push subscriptions, both
    under ``storage.data_dir/push/`` unless overridden.

    The stored row is deliberately minimal — endpoint, keys, a point, the
    alert preferences and the per-subscription state machine. No email, no
    name, no IP address, nothing that identifies a person beyond the
    coordinate they asked to be warned about.
    """

    enabled: bool = False
    # VAPID identity. The private key PEM lives in the data volume and is
    # auto-generated (mode 0600) on first start when enabled and missing.
    # None → ``<storage.data_dir>/push/vapid_private.pem``; resolved by
    # ``push.paths.resolved_key_path`` (this model does not know storage).
    vapid_private_key_file: Path | None = None
    # The VAPID ``sub`` claim push services see: an operator contact, either
    # ``mailto:...`` or ``https://...``. Required when enabled — push
    # services may reject or rate-limit a JWT without a usable contact.
    vapid_subject: str | None = None
    # SQLite subscription store. None → ``<storage.data_dir>/push/
    # subscriptions.sqlite`` (``push.paths.resolved_db_path``).
    db_path: Path | None = None
    # Offered lead times are the national probability leads at or beyond
    # this. Below ~20 min a browser notification arrives too late to act on.
    min_lead_min: Annotated[int, Field(ge=0, le=180)] = 20
    threshold_options_pct: list[int] = Field(default_factory=lambda: [40, 60, 80])
    default_threshold_pct: Annotated[int, Field(ge=1, le=99)] = 60
    default_lead_min: Annotated[int, Field(ge=1, le=180)] = 30
    default_quiet_start: str = "22:00"
    default_quiet_end: str = "07:00"
    # Hard cap on stored subscriptions. The fan-out is sequential and runs
    # inside the 5-minute cycle, so the cap is a latency budget as much as
    # a storage one.
    max_subscriptions: Annotated[int, Field(ge=1, le=100_000)] = 200
    # SSRF guard: the service POSTs to whatever URL a browser hands it, from
    # a VM that can see the LAN. Only these push services are reachable.
    allowed_endpoint_host_suffixes: list[str] = Field(
        default_factory=lambda: [
            "fcm.googleapis.com",
            "updates.push.services.mozilla.com",
            "push.services.mozilla.com",
            "web.push.apple.com",
            "notify.windows.com",
        ],
    )
    # Web Push TTL: how long the push service holds an undelivered message.
    # 15 min — a rain warning that arrives later than that is noise.
    ttl_s: Annotated[int, Field(ge=0, le=86_400)] = 900
    # Wall-clock budget for one cycle's sequential fan-out. Anything still
    # queued when it expires is dropped (and counted), never allowed to
    # delay the next cycle.
    fanout_budget_s: Annotated[float, Field(gt=0, le=300)] = 20.0
    # Decision-engine rules (see ``push.engine``): how long a notified
    # subscription stays disarmed, and how many consecutive observations
    # above threshold are required before firing.
    rearm_after_min: Annotated[int, Field(ge=0, le=1440)] = 60
    persistence_obs: Annotated[int, Field(ge=1, le=10)] = 2

    @field_validator("threshold_options_pct")
    @classmethod
    def _thresholds_valid(cls, v: list[int]) -> list[int]:
        if not v or v != sorted(v) or len(set(v)) != len(v):
            raise ValueError(
                "push threshold_options_pct must be ascending and unique",
            )
        if any(p <= 0 or p >= 100 for p in v):
            raise ValueError("push threshold_options_pct entries must be in (0, 100)")
        return v

    @field_validator("default_quiet_start", "default_quiet_end")
    @classmethod
    def _quiet_hhmm(cls, v: str) -> str:
        if not _HHMM_RE.match(v):
            raise ValueError("push quiet hours must be 24-hour 'HH:MM' strings")
        return v

    @field_validator("allowed_endpoint_host_suffixes")
    @classmethod
    def _suffixes_valid(cls, v: list[str]) -> list[str]:
        out = [s.strip().lower().strip(".") for s in v]
        if any(not s or "/" in s for s in out):
            raise ValueError(
                "push allowed_endpoint_host_suffixes must be bare host suffixes",
            )
        return out

    @model_validator(mode="after")
    def _coherent(self) -> "PushConfig":
        if self.vapid_subject is not None:
            subject = self.vapid_subject.strip()
            if not (subject.startswith("mailto:") or subject.startswith("https:")):
                raise ValueError(
                    "push vapid_subject must be a 'mailto:' or 'https:' URI",
                )
        if self.enabled and not self.vapid_subject:
            raise ValueError(
                "push.enabled requires push.vapid_subject "
                "(a 'mailto:' or 'https:' operator contact)",
            )
        if self.default_threshold_pct not in self.threshold_options_pct:
            raise ValueError(
                "push default_threshold_pct must be one of threshold_options_pct",
            )
        if self.default_lead_min < self.min_lead_min:
            raise ValueError("push default_lead_min must be >= push min_lead_min")
        return self


class Config(BaseSettings):
    """Sidecar config root. Env-var overrides via ``DMI_NOWCAST_*``."""

    model_config = SettingsConfigDict(
        env_prefix="DMI_NOWCAST_",
        env_nested_delimiter="__",
        env_file=None,  # Sidecar uses YAML, not .env (devs use config.yaml).
        extra="forbid",
    )

    home: HomeConfig
    poll: PollConfig = Field(default_factory=PollConfig)
    dmi: DmiConfig = Field(default_factory=DmiConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    lightning: LightningConfig = Field(default_factory=LightningConfig)
    push: PushConfig = Field(default_factory=PushConfig)

    @classmethod
    def settings_customise_sources(  # noqa: PLR0913
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Make env vars beat YAML (init_settings).

        Default pydantic-settings precedence is init > env, which means
        a value present in config.yaml can't be overridden via the
        ``DMI_NOWCAST_*`` env vars. For docker-compose-style deployments
        we want the opposite: YAML is the baseline, env vars override.
        """
        return env_settings, init_settings, dotenv_settings, file_secret_settings


DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: Path | None = None) -> Config:
    """Read YAML + env overrides into a validated ``Config``.

    ``path`` overrides the env-var lookup. Missing file raises
    ``FileNotFoundError`` — sidecar refuses to start without an explicit
    config because every field below has a domain meaning and silent
    defaults would mask deployment mistakes (e.g. nowcasting for the
    wrong location).
    """
    if path is None:
        path = Path(os.environ.get("DMI_NOWCAST_CONFIG", str(DEFAULT_CONFIG_PATH)))
    if not path.is_file():
        raise FileNotFoundError(
            f"sidecar config not found at {path}; copy "
            f"sidecar/config.example.yaml to {path} or set DMI_NOWCAST_CONFIG"
        )
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a mapping, got {type(raw).__name__}")
    # pydantic-settings env-var overrides happen automatically inside
    # ``Config(**raw)`` since BaseSettings merges env vars over kwargs.
    return Config(**raw)
