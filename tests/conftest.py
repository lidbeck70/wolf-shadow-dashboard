"""
Gemensamt för alla tester.

Universum-registrets förvalda läsare är "inget register" i testerna: Gisten går att läsa utan token, så i CI (som
har nätverk) skulle skannrarnas tester annars få det riktiga registret i stället för sina fejkade listor
(insider_scan, screens_scan …). Tester som vill ha ett register skickar en egen loader (se tests/test_*_registry.py).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _no_live_universe_register(monkeypatch):
    from universe import registry as ur
    monkeypatch.setattr(ur, "default_loader", lambda: (lambda name, fallback: fallback))
