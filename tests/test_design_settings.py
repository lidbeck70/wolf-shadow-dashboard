"""Designförslaget PR 4: ⚙️ Inställningar — kapitalet sparas, reglerna visas låsta, inga standardvärden ändras."""
import pathlib

from ui import nav
from ui import settings as us

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_capital_parsing():
    assert us.capital_of({"capital": 75000}) == 75000.0
    assert us.capital_of({"capital": "100000"}) == 100000.0
    for bad in (None, {}, {"capital": None}, {"capital": 0}, {"capital": -5}, {"capital": "x"}, []):
        assert us.capital_of(bad) is None


def test_capital_from_state():
    assert us.capital({us.STORE: {"capital": 50000}}) == 50000.0
    assert us.capital({}) is None


def test_rules_table_covers_every_playbook_and_reads_the_rules():
    import strategy_rules as sr
    rows = us.rules_rows()
    assert len(rows) == len(sr.PLAYBOOKS)
    by = {r["Strategi"]: r for r in rows}
    for p in sr.PLAYBOOKS.values():
        assert by[p.name]["Risk per affär"] == p.risk.risk_per_trade


def test_settings_page_is_in_the_tree():
    assert nav.resolve(nav.target_slug("settings")) == ["more", "⚙️ Inställningar"]


def test_old_viking_never_reads_settings():
    for f in ("ovtlyr/ui/layout.py", "screener_ovtlyr.py", "ovtlyr/ui/viking_nine_regime.py",
              "ovtlyr/ui/viking_screens.py", "ovtlyr_nine.py", "viking_execution.py", "viking_exit.py"):
        path = ROOT / f
        if path.exists():
            assert "ui.settings" not in path.read_text(encoding="utf-8"), f
