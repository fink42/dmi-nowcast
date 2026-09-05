"""Web Push notifications (website Phase D).

Modules:

- ``paths``            — where the key and the DB live, resolved against
                         ``storage.data_dir``.
- ``store``            — SQLite subscription store (the only persistence).
- ``vapid``            — VAPID key generation / loading / public-key encoding.
- ``keygen``           — ``python -m dmi_nowcast_sidecar.push.keygen`` CLI.
- ``endpoint_policy``  — SSRF guard over browser-supplied endpoints.
- ``routes``           — the ``/api/push/*`` router.
- ``service``          — per-cycle evaluation + fan-out.
- ``engine``           — the per-subscription decision state machine.
- ``thresholds``       — the fitted horizon -> threshold table (Phase G),
                         hot-reloaded from ``push.thresholds_path``.
- ``messages``         — notification payload construction (da/en).
- ``fanout``           — one encrypted Web Push delivery.

Nothing is imported here: ``service``/``fanout`` pull in ``pywebpush`` and
``requests``, which an instance with ``push.enabled: false`` should never
have to load. ``app.py`` imports them lazily, inside the enabled branch.
"""
