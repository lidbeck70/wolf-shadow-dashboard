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


@pytest.fixture(autouse=True)
def _journal_stays_out_of_the_repo(monkeypatch, tmp_path):
    """Tradejournalen skriver aldrig i repot eller till den riktiga Gisten under testerna.

    positions.close → journal_bridge.record_close → save_journal körs av flera registertester. Förr skrev det
    .journal_data.json i arbetskatalogen (16 testaffärer hamnade i repot), och med en GITHUB_TOKEN i miljön skulle
    det PATCHa den riktiga journalen. Lokala filen pekas till tmp_path och token är borta; tester som vill pröva
    Gist-vägen sätter en egen.
    """
    import trade_journal as tj
    monkeypatch.setattr(tj, "_real_get_github_token", tj._get_github_token, raising=False)   # för tokentestet
    monkeypatch.setattr(tj, "LOCAL_FALLBACK", str(tmp_path / ".journal_data.json"))
    monkeypatch.setattr(tj, "_get_github_token", lambda: None)
