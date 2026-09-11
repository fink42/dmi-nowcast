"""Decision rows as columns, and the Layer B gauge outcome on them.

Three things that started life inside ``scripts/benchmark_report.py`` and
now have two callers each — the offline report, and the sidecar's nightly
post-processing refit (:mod:`dmi_nowcast_sidecar.postprocess_fit`). The
scripts directory is deliberately NOT baked into the runtime image, so a
step that runs inside the container cannot import it; leaving these here
and importing them back into the report is what keeps ONE definition of

* which rows a run contributes (:func:`load_probabilities`) — the
  deduplication, the last-directory-wins rule and the all-NaN column for a
  file that predates a feature;
* what "the gauge was wet within L" means (:class:`GaugeGrid`) — the
  half-open window, and the refusal to grade a window a gauge said nothing
  about;
* which stations are scored at all (:func:`build_gauge_grid`) — the window
  pad and the dead-gauge exclusion.

A second opinion about any of them would mean "the nightly fit beats the
baseline" was a claim about two different samples.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from dmi_nowcast_core.warning_score import (
    dead_gauge_scan,
    gauge_truth_vectorised,
    p_rain_column,
)

from .threshold_sweep import GAUGE_PAD_MIN, SweepError, decision_parquets

#: Seconds in a day — the day-block bootstrap's block key.
DAY_SEC = 86_400


# ---------------------------------------------------------------------------
# Loading probabilities without materialising a row per dict
# ---------------------------------------------------------------------------

DEFAULT_PROBABILITY_TEMPLATE = "p_rain_{lead}"


def column_template(template: str) -> Callable[[int], str]:
    """``"p_post_{lead}"`` → a function naming that run's probability column.

    The template is how a post-processed run is scored by the same report:
    ``scripts/fit_postprocess.py`` writes its out-of-fold probabilities into
    a COPY of the run as ``p_post_<lead>``, leaving ``p_rain_<lead>``
    untouched beside it, and ``--probability-column p_post_{lead}`` points
    both layers at the new column. Everything else — the outcome window,
    the dead-gauge rule, the strata, the bootstrap — is unchanged, which is
    the whole point of doing it this way instead of writing a second
    report.
    """
    if "{lead}" not in template:
        raise ValueError(
            f"--probability-column must contain '{{lead}}', got {template!r}"
        )
    probe = template.format(lead=30)
    if not probe or probe != probe.strip():
        raise ValueError(f"--probability-column is not a column name: {template!r}")

    def name(lead: int) -> str:
        return template.format(lead=int(lead))

    return name




def load_probabilities(
    directories: Sequence[Path],
    leads: Sequence[int],
    *,
    stations: Sequence[str] | None = None,
    column_for: Callable[[int], str] = p_rain_column,
    extra_columns: Sequence[str] = (),
    log=None,
) -> dict:
    """Decision rows as columns: ``t``, station code, and one p per lead.

    The Arrow path, deliberately separate from
    ``threshold_sweep.load_decisions``: that one hands back row dicts,
    which is what the state-machine replay needs and what Layer B must not
    pay for. Half a million rows of twelve fields is around a gigabyte as
    Python objects and about twelve megabytes as numpy columns, and every
    number Layer B computes is a whole-column operation.

    Deduplication matches ``load_decisions`` exactly — one row per
    ``(radar_ts, station_id)``, the LAST one read winning, directories in
    the order given — so a pooled replay-plus-live read scores the same
    rows both layers replay.

    ``column_for`` names the probability column per lead (default
    ``p_rain_<lead>``; see :func:`column_template`). ``extra_columns`` are
    additional NUMERIC columns carried along as float64 — the H-P feature
    columns, for ``scripts/fit_postprocess.py``, which needs them
    deduplicated exactly the way the scored probabilities are. A column no
    file carries comes back all-NaN rather than missing, so a model can be
    fitted against a run that predates a feature.

    Returns ``{"t": int64 epoch seconds of the decision instant,
    "radar_ts": int64 epoch seconds of the anchor frame, "station": int32
    codes, "stations": [id], "p": {lead: float64 with NaN for null},
    "extra": {name: float64}, "files": n, "rows": n, "duplicates": n}``.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    wanted = [int(lead) for lead in leads]
    p_columns = {lead: column_for(lead) for lead in wanted}
    extras = [str(name) for name in dict.fromkeys(extra_columns)]
    if len(set(p_columns.values())) != len(p_columns):
        raise SweepError("the probability column template collides across leads")
    #: The columns asked of every file, in one order: the scored
    #: probabilities first, then whatever extras the caller wants.
    asked = list(dict.fromkeys(list(p_columns.values()) + extras))
    tables: list[Any] = []
    counts = {"files": 0, "skipped": 0, "rows": 0}
    for directory in directories:
        for path in decision_parquets(Path(directory)):
            try:
                schema = pq.read_schema(path)
            except Exception as exc:  # noqa: BLE001 — one unreadable file
                counts["skipped"] += 1
                if log:
                    log(f"skipping {path.name}: {type(exc).__name__}: {exc}")
                continue
            names = set(schema.names)
            if not {"radar_ts", "station_id", "action"} <= names:
                counts["skipped"] += 1
                if log:
                    log(f"skipping {path.name}: not a decision table")
                continue
            present = [name for name in asked if name in names]
            columns = ["radar_ts", "generated_at", "station_id"] + present
            try:
                table = pq.read_table(path, columns=columns)
            except Exception as exc:  # noqa: BLE001
                counts["skipped"] += 1
                if log:
                    log(f"skipping {path.name}: {type(exc).__name__}: {exc}")
                continue
            for name in asked:
                if name not in names:
                    table = table.append_column(
                        name, pa.nulls(table.num_rows, pa.float64()),
                    )
            table = table.select(
                ["radar_ts", "generated_at", "station_id"] + asked
            )
            counts["files"] += 1
            counts["rows"] += table.num_rows
            tables.append(table.cast(pa.schema([
                ("radar_ts", pa.timestamp("us", tz="UTC")),
                ("generated_at", pa.timestamp("us", tz="UTC")),
                ("station_id", pa.string()),
                *[(name, pa.float64()) for name in asked],
            ])))
    if not tables:
        raise SweepError("no decision rows found")

    merged = pa.concat_tables(tables)
    del tables
    if stations is not None:
        keep = pc.is_in(
            merged.column("station_id"),
            value_set=pa.array([str(s) for s in stations], pa.string()),
        )
        merged = merged.filter(pc.fill_null(keep, False))
    # Last-read wins, as in ``load_decisions``: tag every row with its
    # arrival order, keep the maximum per key, take those rows. An Arrow
    # group-by rather than a Python dict — the dict was the memory.
    order = pa.array(np.arange(merged.num_rows, dtype=np.int64))
    merged = merged.append_column("__idx", order)
    winners = merged.group_by(["radar_ts", "station_id"]).aggregate(
        [("__idx", "max")],
    )
    duplicates = merged.num_rows - winners.num_rows
    merged = merged.take(
        winners.column("__idx_max").combine_chunks(),
    ).drop_columns(["__idx"]).sort_by([
        ("radar_ts", "ascending"), ("station_id", "ascending"),
    ])
    del winners

    station_ids = sorted(set(merged.column("station_id").to_pylist()))
    codes = np.asarray(
        pc.index_in(
            merged.column("station_id"),
            value_set=pa.array(station_ids, pa.string()),
        ).combine_chunks().cast(pa.int32()).to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    # ``generated_at`` is the decision instant; a row written before that
    # column existed falls back to its frame stamp rather than dropping out.
    generated = merged.column("generated_at").combine_chunks()
    stamp = pc.if_else(
        pc.is_valid(generated), generated, merged.column("radar_ts"),
    )
    t = np.asarray(
        stamp.cast(pa.int64()).to_numpy(zero_copy_only=False), dtype=np.int64,
    ) // 1_000_000

    def _numeric(name: str) -> np.ndarray:
        return np.asarray(
            merged.column(name).combine_chunks()
            .cast(pa.float64()).to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )

    radar_ts = np.asarray(
        merged.column("radar_ts").combine_chunks()
        .cast(pa.int64()).to_numpy(zero_copy_only=False), dtype=np.int64,
    ) // 1_000_000
    probabilities = {lead: _numeric(p_columns[lead]) for lead in wanted}
    out = {
        "t": t,
        "radar_ts": radar_ts,
        "station": codes,
        "stations": station_ids,
        "p": probabilities,
        "extra": {name: _numeric(name) for name in extras},
        "files": counts["files"],
        "skipped": counts["skipped"],
        "rows": int(merged.num_rows),
        "duplicates": int(duplicates),
    }
    if log:
        log(
            f"layer B: {out['rows']} unique row(s) from {counts['files']} "
            f"file(s) over {len(station_ids)} station(s) "
            f"({duplicates} duplicate key(s))"
        )
    return out


# ---------------------------------------------------------------------------
# The gauge grid, and the outcome window on it
# ---------------------------------------------------------------------------


class GaugeGrid:
    """A station × slot wet/known grid with prefix sums over both.

    ``gauge_truth_vectorised`` already builds the grid; this adds the two
    cumulative sums that turn "was it wet in (t, t+L]" from a slice-and-
    reduce per row into two lookups per row. At ~460 000 rows × four leads
    that is the difference between minutes of Python and milliseconds of
    numpy, and the sums cost about 40 MB for a ten-month archive.
    """

    def __init__(self, truth: Any, stations: Sequence[str]) -> None:
        series = truth.series
        known = [s for s in stations if s in series]
        if not known:
            raise SweepError("no station has gauge observations")
        first = series[known[0]]
        self.stations = list(stations)
        self.first_sec = int(first.slot_end[0])
        self.n_slots = int(first.slot_end.size)
        self.step_sec = int(first.slot_min) * 60
        wet = np.zeros((len(stations), self.n_slots), dtype=bool)
        seen = np.zeros_like(wet)
        for row, station in enumerate(stations):
            slots = series.get(station)
            if slots is None:
                continue
            if int(slots.slot_end[0]) != self.first_sec or int(
                slots.slot_end.size
            ) != self.n_slots:
                # Every station shares one grid by construction; if that
                # ever stops being true the prefix sums would silently
                # index the wrong slots.
                raise SweepError(f"station {station} has a different slot grid")
            wet[row] = slots.wet & slots.known
            seen[row] = slots.known
        # One extra leading zero column so a half-open difference needs no
        # special case at index 0.
        self._wet = np.zeros((len(stations), self.n_slots + 1), dtype=np.int32)
        self._known = np.zeros_like(self._wet)
        np.cumsum(wet, axis=1, dtype=np.int32, out=self._wet[:, 1:])
        np.cumsum(seen, axis=1, dtype=np.int32, out=self._known[:, 1:])
        self._stride = self.n_slots + 1
        self._wet_flat = self._wet.reshape(-1)
        self._known_flat = self._known.reshape(-1)

    def outcome(
        self, t: np.ndarray, station: np.ndarray, lead_min: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(outcome, usable)`` for "wet within ``lead_min`` of ``t``".

        The window is the slots whose END falls in ``(t, t + lead]`` — see
        the module docstring for why the boundary is there and what it
        costs. ``usable`` is False where the window ran off the grid, or
        where nothing was wet and a slot was unknown: a gauge that said
        nothing cannot certify a dry half hour.
        """
        step = self.step_sec
        # Smallest index whose slot END is strictly after t, and the
        # largest whose end is at or before t + lead. Floor division on
        # int64 handles instants off the slot grid without a branch.
        lo = (t - self.first_sec) // step + 1
        hi = (t + int(lead_min) * 60 - self.first_sec) // step
        inside = (lo >= 0) & (hi < self.n_slots) & (hi >= lo)
        safe_lo = np.where(inside, lo, 0)
        safe_hi = np.where(inside, hi, 0)
        base = station * self._stride
        n_wet = (
            self._wet_flat[base + safe_hi + 1] - self._wet_flat[base + safe_lo]
        )
        n_known = (
            self._known_flat[base + safe_hi + 1]
            - self._known_flat[base + safe_lo]
        )
        span = safe_hi - safe_lo + 1
        wet = n_wet > 0
        dry = (~wet) & (n_known == span)
        usable = inside & (wet | dry)
        return wet.astype(np.float64), usable


def build_gauge_grid(
    corpus_dir: Path,
    station_ids: Sequence[str],
    window: tuple[datetime, datetime],
    *,
    dry_min: int,
    onset_min_mm: float,
    min_known_slots: int,
    log=None,
) -> tuple["GaugeGrid", list[dict], list[str]]:
    """``(grid, dead-gauge rows, scored stations)`` for a decision window.

    Layer B's truth, in one function so that anything else scoring against
    the gauges — ``scripts/fit_postprocess.py`` fits on exactly this
    outcome — uses the same window pad, the same wet rule and the same
    dead-gauge exclusions rather than a second opinion about any of them.
    """
    pad = timedelta(minutes=GAUGE_PAD_MIN)
    truth = gauge_truth_vectorised(
        Path(corpus_dir), window[0] - pad, window[1] + pad, list(station_ids),
        dry_min=int(dry_min), onset_min_mm=float(onset_min_mm),
        pad_min=GAUGE_PAD_MIN, log=log,
    )
    dead = dead_gauge_scan(truth, min_known_slots=int(min_known_slots))
    if log:
        for row in dead:
            log(
                f"dead gauge {row.station_id}: {row.known_slots} known "
                "slot(s), never wet — excluded"
            )
    rows = [
        {
            "station_id": row.station_id,
            "known_slots": row.known_slots,
            "wet_slots": row.wet_slots,
        }
        for row in dead
    ]
    excluded = {row.station_id for row in dead}
    scored = [s for s in station_ids if s not in excluded]
    grid = GaugeGrid(truth, scored)
    del truth
    return grid, rows, scored


def decision_window(t: np.ndarray) -> tuple[datetime, datetime]:
    """First and last decision instant, as aware UTC datetimes."""
    return (
        datetime.fromtimestamp(int(t.min()), timezone.utc),
        datetime.fromtimestamp(int(t.max()), timezone.utc),
    )


def recode_stations(rows: dict, scored: Sequence[str]) -> None:
    """Map the row station codes onto ``scored``; unknown becomes dropped.

    Rows at a station the grid does not carry are given a probability of
    NaN rather than a code that would index some other station's slots —
    a silent mis-join is the one failure mode a benchmark must not have.
    """
    lookup = {station: index for index, station in enumerate(scored)}
    codes = np.array(
        [lookup.get(station, -1) for station in rows["stations"]], dtype=np.int64,
    )
    mapped = codes[rows["station"]]
    missing = mapped < 0
    rows["station"] = np.where(missing, 0, mapped)
    # Kept so a caller that scores something other than ``p`` — the H-P fit
    # scores its own predictions on the same rows — can drop exactly the
    # rows this function silenced.
    rows["dropped"] = missing
    if np.any(missing):
        # A new array rather than an in-place write: Arrow's own buffers
        # come back read-only, and a zero-copy column is one of them.
        rows["p"] = {
            lead: np.where(missing, np.nan, values)
            for lead, values in rows["p"].items()
        }


__all__ = [
    "DAY_SEC",
    "DEFAULT_PROBABILITY_TEMPLATE",
    "GaugeGrid",
    "build_gauge_grid",
    "column_template",
    "decision_window",
    "load_probabilities",
    "recode_stations",
]
