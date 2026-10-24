"""Beslut 5: ett gemensamt kapital blir standardvärde i Wolf Regime, Thor, Positionskalkylatorn och journalen."""
import pathlib

from ui import settings as us

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_capital_or_falls_back_to_the_page_default(monkeypatch):
    monkeypatch.setattr(us, "capital", lambda state=None: None)
    assert us.capital_or(100000) == 100000.0
    monkeypatch.setattr(us, "capital", lambda state=None: 60000.0)
    assert us.capital_or(100000) == 60000.0


def test_pages_use_the_shared_capital(monkeypatch):
    monkeypatch.setattr(us, "capital", lambda state=None: 60000.0)
    from thor import regim
    import position_sizing as ps
    import trade_journal as tj
    assert regim._capital_default() == 60000.0
    assert ps._trading_default() == 60000.0 and ps.DEFAULT_ASK_ACCOUNT == 37_000.0
    assert tj._start_capital() == 60000.0
    assert "capital_or(100000)" in (ROOT / "tabs" / "regime.py").read_text(encoding="utf-8")


def test_old_viking_does_not_read_the_capital():
    for f in ("ovtlyr/ui/layout.py", "screener_ovtlyr.py", "ovtlyr/ui/viking_nine_regime.py", "ovtlyr/ui/viking_screens.py"):
        p = ROOT / f
        if p.exists():
            src = p.read_text(encoding="utf-8")
            assert "capital_or" not in src and "ui.settings" not in src, f
