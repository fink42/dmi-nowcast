"""The protocol travels with the model, and the mask travels with it.

``scripts/fit_postprocess.py --protocol random-point`` masks the point's
own gauge out of the WHOLE feature table before a single fold is cut, so
the model it produces has never seen ``g_mm_10`` / ``g_mm_30`` /
``g_mm_60`` / ``g_min_since_wet`` / ``g_dry_60`` as anything but NaN and
``g_known`` as anything but 0. That is a statement about how it must be
SERVED, not a note about how it was made: hand the same model a live
gauge reading at a gauge station and it is being asked a question in a
language nobody taught it, and — worse for this project — the number it
returns at a gauge stops being the number a subscriber's address would
get, which is exactly the equivalence the random-point protocol exists
to establish.

So the protocol is a field of :class:`~dmi_nowcast_core.postprocess.PostprocessModel`,
written at the top level of ``postprocess.json``, and the model applies
:func:`~dmi_nowcast_core.postprocess.mask_own_gauge` itself before the
design is built. Every caller — the cycle's table, the single-row twin
the on-demand lookup goes through, the sidecar — is covered by
construction rather than by remembering.

The tests below are behavioural in both directions: a random-point model
must be UNABLE to tell a live gauge from an absent one, and an at-gauge
model must still be able to, or the mask is not being applied where it
was claimed.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from dmi_nowcast_core import postprocess as pp

LEADS = (20, 30)
DESIGN_LEADS = (10, 20, 30)

#: Every column the mask touches, with a value that is unmistakably a
#: live reading: rain in the last ten minutes, a gauge that went wet two
#: minutes ago, not dry, and ``g_known=1`` to say so.
LIVE_GAUGE = {
    "g_mm_10": 1.6, "g_mm_30": 3.1, "g_mm_60": 4.4,
    "g_min_since_wet": 2.0, "g_dry_60": 0.0, "g_known": 1.0,
}


def _rows(n: int = 2400, seed: int = 11) -> tuple[dict, dict]:
    """Rows where the own gauge carries real signal, so masking BITES.

    ``g_min_since_wet`` is built to predict the outcome on its own — a
    point that was wet a minute ago is usually about to be wet again — so
    a fit that can see it leans on it, and a model that masks it has
    visibly different coefficients and visibly different answers. A
    fixture where the gauge block were noise would pass every test here
    for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    t = (
        np.datetime64("2026-02-01T00:00:00").astype("datetime64[s]").astype(np.int64)
        + np.arange(n) * 600
    )
    signal = rng.uniform(size=n)
    wet_now = rng.uniform(size=n) < 0.35
    since_wet = np.where(wet_now, rng.uniform(0.0, 20.0, size=n), 240.0)
    rows: dict = {
        "season": pp.seasons_from_epoch(t),
        "hour_utc": pp.hours_from_epoch(t).astype(np.float64),
        "up_dist_km": np.where(signal > 0.2, 40.0 * (1.0 - signal), np.nan),
        "up_max_20km_mm_h": signal * 6.0,
        "bulk_kmh": rng.uniform(5.0, 60.0, size=n),
        "observed_mm_h": np.where(wet_now, rng.uniform(0.2, 3.0, size=n), 0.0),
        "station_radar_km": rng.uniform(10.0, 90.0, size=n),
        "station_id": np.array([f"S{i % 6:02d}" for i in range(n)]),
        # The own-gauge block, the thing the protocol takes away.
        "g_known": np.ones(n, dtype=np.float64),
        "g_mm_10": np.where(wet_now, rng.uniform(0.1, 2.5, size=n), 0.0),
        "g_mm_30": np.where(wet_now, rng.uniform(0.2, 5.0, size=n), 0.0),
        "g_mm_60": np.where(wet_now, rng.uniform(0.2, 8.0, size=n), 0.0),
        "g_min_since_wet": since_wet,
        "g_dry_60": (~wet_now).astype(np.float64),
        # The neighbour block, which the protocol deliberately keeps.
        "ng_near_km": rng.uniform(2.0, 45.0, size=n),
        "ng_mm_10_idw": rng.uniform(0.0, 2.0, size=n),
    }
    for lead in DESIGN_LEADS:
        rows[pp.raw_fraction_column(lead)] = np.clip(
            signal * (lead / 60.0) + rng.normal(0, 0.03, size=n), 0.0, 1.0,
        )
    # The outcome leans on the own gauge as well as on the upstream rain,
    # so both an at-gauge and a random-point fit have something to find.
    p = np.clip(0.15 + 0.55 * signal + 0.30 * wet_now, 0.0, 1.0)
    truth = {
        lead: (
            (rng.uniform(size=n) < np.clip(p * lead / 30.0, 0, 1)).astype(float),
            np.ones(n, dtype=bool),
        )
        for lead in LEADS
    }
    return rows, truth


def _fit(protocol: str, rows: dict, truth: dict) -> pp.PostprocessModel:
    """One v2 logistic under ``protocol``, on the rows the protocol implies.

    The random-point arm masks the table first, exactly as the fit script
    does, because a model that claims the protocol while having been
    fitted on unmasked rows is the bug this whole file is about.
    """
    training = (
        pp.mask_own_gauge(rows) if protocol == pp.PROTOCOL_RANDOM_POINT else rows
    )
    return pp.fit_postprocess(
        training, truth, LEADS, design_leads=DESIGN_LEADS,
        settings=pp.FitSettings(design=pp.DESIGN_V2),
        protocol=protocol,
    )


def _with_gauge(rows: dict) -> dict:
    """``rows`` with the own-gauge block set to an unmistakable reading."""
    out = dict(rows)
    n = rows["bulk_kmh"].size
    for name, value in LIVE_GAUGE.items():
        out[name] = np.full(n, float(value), dtype=np.float64)
    return out


@pytest.fixture(scope="module")
def dataset() -> tuple[dict, dict]:
    return _rows()


# ---------------------------------------------------------------------------
# The document carries the protocol
# ---------------------------------------------------------------------------


class TestTheDocumentSaysWhichProtocol:
    def test_the_default_is_the_protocol_that_shipped(
        self, dataset: tuple[dict, dict],
    ) -> None:
        rows, truth = dataset
        model = pp.fit_postprocess(
            rows, truth, LEADS, design_leads=DESIGN_LEADS,
        )
        assert model.protocol == pp.PROTOCOL_AT_GAUGE
        assert model.masks_own_gauge is False

    def test_the_key_is_at_the_top_level_and_survives_a_round_trip(
        self, dataset: tuple[dict, dict],
    ) -> None:
        rows, truth = dataset
        model = _fit(pp.PROTOCOL_RANDOM_POINT, rows, truth)
        payload = json.loads(model.dumps())
        assert payload["protocol"] == pp.PROTOCOL_RANDOM_POINT
        # Additive: the schema version does not move for a new key with a
        # default, or every running process would refuse the file.
        assert payload["schema_version"] == pp.SCHEMA_VERSION
        # And the provenance block keeps saying it too, because that is
        # where every artefact fitted before the key existed says it.
        assert payload["training"]["protocol"] == pp.PROTOCOL_RANDOM_POINT
        restored = pp.PostprocessModel.loads(model.dumps())
        assert restored.protocol == pp.PROTOCOL_RANDOM_POINT
        assert restored.masks_own_gauge is True

    def test_it_falls_back_to_the_provenance_block(
        self, dataset: tuple[dict, dict],
    ) -> None:
        """The already-fitted artefacts: masked, but with no top-level key.

        ``postprocess_rp_trees_all/postprocess.json`` was written before
        this field existed and says ``random-point`` only inside
        ``training``. Loading it as at-gauge would serve it unmasked.
        """
        rows, truth = dataset
        model = _fit(pp.PROTOCOL_RANDOM_POINT, rows, truth)
        payload = json.loads(model.dumps())
        del payload["protocol"]
        assert payload["training"]["protocol"] == pp.PROTOCOL_RANDOM_POINT
        restored = pp.PostprocessModel.from_json(payload)
        assert restored.protocol == pp.PROTOCOL_RANDOM_POINT
        assert restored.masks_own_gauge is True

    def test_a_document_from_before_the_field_loads_as_at_gauge(
        self, dataset: tuple[dict, dict],
    ) -> None:
        """Old file, new build — the deploy-skew direction that happens."""
        rows, truth = dataset
        model = pp.fit_postprocess(
            rows, truth, LEADS, design_leads=DESIGN_LEADS,
        )
        payload = json.loads(model.dumps())
        del payload["protocol"]
        payload["training"].pop("protocol", None)
        restored = pp.PostprocessModel.from_json(payload)
        assert restored.protocol == pp.PROTOCOL_AT_GAUGE
        assert restored.masks_own_gauge is False

    def test_an_unknown_protocol_is_refused_at_the_fit(
        self, dataset: tuple[dict, dict],
    ) -> None:
        rows, truth = dataset
        with pytest.raises(ValueError, match="unknown protocol"):
            pp.fit_postprocess(
                rows, truth, LEADS, design_leads=DESIGN_LEADS,
                protocol="whatever-we-try-next",
            )


# ---------------------------------------------------------------------------
# The model masks what its protocol says it cannot see
# ---------------------------------------------------------------------------


class TestServingHonoursTheProtocol:
    def test_a_random_point_model_cannot_tell_a_live_gauge_from_none(
        self, dataset: tuple[dict, dict],
    ) -> None:
        """The contract, stated as the equality that has to hold.

        Same rows twice: once with a gauge that says it rained two minutes
        ago, once with the block masked the way the fit saw it. A model
        fitted under random-point must return the SAME probability, to the
        bit, or the gauge is leaking into service.

        End to end, and — today — true for TWO reasons: see
        :meth:`test_the_mask_is_what_makes_it_true_and_not_the_fit`.
        """
        rows, truth = dataset
        model = _fit(pp.PROTOCOL_RANDOM_POINT, rows, truth)
        live = _with_gauge(rows)
        masked = pp.mask_own_gauge(live)
        for lead in LEADS:
            np.testing.assert_array_equal(
                model.predict(live, lead), model.predict(masked, lead),
            )

    def test_the_fit_leaves_the_masked_block_dead_on_its_own(
        self, dataset: tuple[dict, dict],
    ) -> None:
        """Why the test above is not enough, written down as a fact.

        Masking the whole table before fitting leaves the own-gauge
        columns CONSTANT, so a logistic gives them a coefficient of
        exactly zero and LightGBM never splits on them — the shipped
        ``postprocess_rp_trees_all`` ensembles use 94-96 of their 104
        columns and none of the six. A serve-time mask therefore changes
        no number in today's artefact, and the equality above would hold
        even with no mask at all.

        That is a property of this fit, not of the serving path. The next
        one need not have it — a fold-local mask, a protocol that masks
        after some feature is derived, a document assembled by hand — and
        the serving path has to be right then too.
        """
        rows, truth = dataset
        model = _fit(pp.PROTOCOL_RANDOM_POINT, rows, truth)
        index = {name: i for i, name in enumerate(model.feature_names)}
        for lead in LEADS:
            for name in (*pp.OWN_GAUGE_COLUMNS, pp.GAUGE_KNOWN_COLUMN):
                assert model.models[lead].coefficients[index[name]] == 0.0

    def test_the_mask_is_what_makes_it_true_and_not_the_fit(
        self, dataset: tuple[dict, dict],
    ) -> None:
        """The guard, on weights that demonstrably CAN read the gauge.

        Coefficients from an at-gauge fit, carried by a document that says
        ``random-point``: the only thing standing between them and a live
        ``g_min_since_wet`` is the mask. So the equality here fails on any
        build where the masking is not applied, which is exactly what the
        test above cannot promise.
        """
        rows, truth = dataset
        weights = _fit(pp.PROTOCOL_AT_GAUGE, rows, truth)
        labelled = dataclasses.replace(
            weights, protocol=pp.PROTOCOL_RANDOM_POINT,
        )
        assert labelled.masks_own_gauge is True
        live = _with_gauge(rows)
        masked = pp.mask_own_gauge(live)
        for lead in LEADS:
            np.testing.assert_array_equal(
                labelled.predict(live, lead), labelled.predict(masked, lead),
            )
            # And it is not a no-op: the same weights read unmasked give a
            # different answer, which is the answer the mask is refusing.
            assert not np.allclose(
                labelled.predict(live, lead), weights.predict(live, lead),
            )

    def test_the_masking_is_what_makes_them_equal(
        self, dataset: tuple[dict, dict],
    ) -> None:
        """The control: the same rows DO separate an at-gauge model.

        Without this, the equality above could be an artefact of a fit
        that simply ignored the gauge block, and the test would pass on a
        build where the mask was never applied.
        """
        rows, truth = dataset
        model = _fit(pp.PROTOCOL_AT_GAUGE, rows, truth)
        assert model.masks_own_gauge is False
        live = _with_gauge(rows)
        masked = pp.mask_own_gauge(live)
        separated = [
            not np.allclose(model.predict(live, lead), model.predict(masked, lead))
            for lead in LEADS
        ]
        assert all(separated)

    def test_an_at_gauge_model_is_handed_the_very_mapping_it_was_given(
        self, dataset: tuple[dict, dict],
    ) -> None:
        """No copy, no mask, nothing: the at-gauge path is untouched."""
        rows, _truth = dataset
        model = _fit(pp.PROTOCOL_AT_GAUGE, *dataset)
        assert model.masked_features(rows) is rows

    def test_the_masked_view_is_exactly_mask_own_gauge(
        self, dataset: tuple[dict, dict],
    ) -> None:
        """Not 'a mask like it' — the same function, so the two cannot drift."""
        rows, truth = dataset
        model = _fit(pp.PROTOCOL_RANDOM_POINT, rows, truth)
        seen = model.masked_features(_with_gauge(rows))
        expected = pp.mask_own_gauge(_with_gauge(rows))
        assert set(seen) == set(expected)
        for name in pp.OWN_GAUGE_COLUMNS:
            assert np.all(np.isnan(np.asarray(seen[name], dtype=np.float64)))
        assert np.all(np.asarray(seen[pp.GAUGE_KNOWN_COLUMN]) == 0.0)
        # The neighbour block is NOT the point's own gauge and survives.
        np.testing.assert_array_equal(seen["ng_near_km"], rows["ng_near_km"])
        np.testing.assert_array_equal(seen["ng_mm_10_idw"], rows["ng_mm_10_idw"])

    def test_the_single_row_path_masks_too(
        self, dataset: tuple[dict, dict],
    ) -> None:
        """One row, lifted to one-element columns — the sidecar's twin.

        ``PostprocessTable.predict`` / ``predict_row`` build exactly this
        shape, so if a single row is masked here it is masked there.
        """
        rows, truth = dataset
        model = _fit(pp.PROTOCOL_RANDOM_POINT, rows, truth)
        live = _with_gauge(rows)
        row = {
            name: np.asarray(values[3:4])
            for name, values in live.items()
        }
        blank = dict(row)
        for name in pp.OWN_GAUGE_COLUMNS:
            blank[name] = np.array([np.nan])
        blank[pp.GAUGE_KNOWN_COLUMN] = np.array([0.0])
        for lead in LEADS:
            assert float(model.predict(row, lead)[0]) == float(
                model.predict(blank, lead)[0]
            )

    def test_the_design_itself_comes_back_masked(
        self, dataset: tuple[dict, dict],
    ) -> None:
        """Masking sits before the matrix, not after it.

        A tree model reads the RAW design, so a mask applied to the scored
        probability rather than to the columns would be no mask at all.
        """
        rows, truth = dataset
        model = _fit(pp.PROTOCOL_RANDOM_POINT, rows, truth)
        names = list(model.feature_names)
        design = model.design(_with_gauge(rows))
        assert pp.GAUGE_KNOWN_COLUMN in names
        # A logistic design is standardised, so 'masked' means 'the value
        # every row has once the block is gone' — one constant column, not
        # the varying one the live gauge would have produced.
        for name in (*pp.OWN_GAUGE_COLUMNS, pp.GAUGE_KNOWN_COLUMN):
            column = design[:, names.index(name)]
            assert np.unique(column).size == 1
