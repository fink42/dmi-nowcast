"""Pytest configuration for the core suite.

Nothing global is needed here: every test in this package is pure Python
/ numpy against committed fixtures — no network, no service to spin up.
The file exists so ``tests/`` stays an unambiguous test package and so
shared fixtures have an obvious home if any are ever added.
"""
from __future__ import annotations
