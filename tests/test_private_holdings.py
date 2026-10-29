"""Privata innehav: kryptering i det publika repot, lösenord på Positioner och fysiska metaller."""
import base64
import json
import pathlib

import pandas as pd
import pytest

import private_data as pdata
from portfolio import metals as mt
from portfolio import private_lock as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ── Kryptering ──────────────────────────────────────────────────────────────
def test_seal_and_unseal_round_trip_hides_the_content():
    data = {"swing": [{"ticker": "VOLV-B.ST", "shares": 120}]}
    box = pdata.seal(data, "en lång lösenfras")
    assert pdata.is_encrypted(box) and "VOLV" not in json.dumps(box)
    assert pdata.unseal(box, "en lång lösenfras") == data
    with pytest.raises(pdata.PrivateDataError):
        pdata.unseal(box, "fel fras")


def test_missing_key_refuses_to_seal_and_plaintext_passes_through(monkeypatch):
    monkeypatch.delenv(pdata.KEY_NAME, raising=False)
    monkeypatch.setattr(pdata, "passphrase", lambda: None)
    with pytest.raises(pdata.PrivateDataError):
        pdata.seal({"a": 1})
    assert pdata.unseal({"a": 1}) == {"a": 1}
    assert pdata.is_private("holdings") and pdata.is_private("metals") and not pdata.is_private("tiggre")


class _Resp:
    def __init__(self, code, body=None):
        self.status_code, self._body = code, body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        pass


def test_storage_encrypts_private_stores_and_decrypts_on_read(monkeypatch):
    import storage
    monkeypatch.setenv(pdata.KEY_NAME, "fras-för-test")
    monkeypatch.setattr(storage, "get_config", lambda: storage.Config("t", "o/r", "panel-data"))
    sent = {}

    def put(url, headers=None, json=None, timeout=None):
        sent["content"] = json["content"]
        return _Resp(201, {"commit": {"sha": "abc", "html_url": "u"}})
    monkeypatch.setattr(storage.requests, "put", put)
    monkeypatch.setattr(storage, "_get_file", lambda cfg, path: None)
    storage.save_json("holdings", {"swing": [{"ticker": "ABB.ST"}]})
    written = base64.b64decode(sent["content"]).decode("utf-8")
    assert "ABB" not in written and pdata.SCHEME in written
    storage.save_json("tiggre", {"rows": ["X"]})                       # inte privat → klartext
    assert "X" in base64.b64decode(sent["content"]).decode("utf-8")
    meta = {"content": base64.b64encode(written.encode()).decode(), "sha": "s"}
    monkeypatch.setattr(storage, "_get_file", lambda cfg, path: meta)
    assert storage.load_json("holdings") == {"swing": [{"ticker": "ABB.ST"}]}
    monkeypatch.setenv(pdata.KEY_NAME, "annan fras")
    with pytest.raises(storage.StorageError):
        storage.load_json("holdings")


def test_alert_scan_reads_the_encrypted_holdings(monkeypatch):
    import alert_scan
    monkeypatch.setenv(pdata.KEY_NAME, "fras")
    box = pdata.seal({"swing": [{"ticker": "SAND.ST"}]}, "fras")
    raw = base64.b64encode(json.dumps(box).encode()).decode()
    monkeypatch.setattr(alert_scan.requests if hasattr(alert_scan, "requests") else __import__("requests"), "get",
                        lambda *a, **k: _Resp(200, {"content": raw}))
    assert alert_scan._repo_file("data/holdings.json", {}) == {"swing": [{"ticker": "SAND.ST"}]}


def test_workflows_pass_the_key_to_the_readers():
    for f in ("scheduled-scan.yml", "wolf-data.yml"):
        assert "PANEL_DATA_KEY: ${{ secrets.PANEL_DATA_KEY }}" in (ROOT / ".github" / "workflows" / f).read_text()
    assert "cryptography" in (ROOT / "requirements.txt").read_text()


# ── Lösenordet ──────────────────────────────────────────────────────────────
def test_password_check_and_lockout():
    st = {}
    assert pl.check("x", st, None) == "saknas" and not pl.unlocked(st)
    for _ in range(pl.MAX_FAILS - 1):
        assert pl.check("fel", st, "rätt", now=0) == "fel"
    assert pl.check("fel", st, "rätt", now=0) == "spärrad"
    assert pl.check("rätt", st, "rätt", now=10) == "spärrad"                     # väntetiden gäller
    assert pl.check("rätt", st, "rätt", now=pl.BLOCK_S + 1) == "ok" and pl.unlocked(st)
    pl.lock(st)
    assert not pl.unlocked(st)


def test_positions_and_start_page_are_behind_the_lock():
    panel = (ROOT / "wolf_panel.py").read_text(encoding="utf-8")
    assert "if gate():" in panel and "render_metals_section" in panel
    assert 'if gate(key="holdings_pw_alloc"):' in panel                         # Allokering bakom samma lås
    home = (ROOT / "tabs" / "home.py").read_text(encoding="utf-8")
    assert "private_lock.unlocked()" in home
    assert "wolf2026" in (ROOT / "auth.py").read_text(encoding="utf-8")         # huvudlösenordets reserv orörd


def test_allokering_shows_nothing_from_the_register_while_locked(monkeypatch):
    """Allokering listar registrets positioner, antal och kassa — samma lås som Positioner."""
    import copy

    from streamlit.testing.v1 import AppTest

    import refresh_ui
    import storage
    seed = {"holdings": {"long": [{"id": "q1", "ticker": "HEMLIG.ST", "strategy": "Quality", "entry_price": 70.0,
                                   "shares": 33}], "cash": 21000}}
    monkeypatch.setattr(storage, "load_json", lambda name, default=None: copy.deepcopy(seed.get(name, default)))
    monkeypatch.setattr(storage, "legacy_gist", lambda *a, **k: None)
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setenv("LOCK_TEST_ROOT", str(ROOT))

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["LOCK_TEST_ROOT"])
        import wolf_panel
        wolf_panel._tab_portfolio()

    def texts(at):
        colls = (at.markdown, at.caption, at.error, at.info, at.warning)
        return " ".join(getattr(e, "value", "") or "" for coll in colls for e in coll)

    at = AppTest.from_function(app, default_timeout=120)
    at.session_state["sub_portfolio"] = "Allokering"
    at.run()
    assert not at.exception, at.exception
    t = texts(at)
    assert "låsta" in t and "HEMLIG" not in t and "21,000" not in t and "Ur registret" not in t
    assert not [b for b in at.get("download_button")]                          # ingen Positioner-CSV

    at = AppTest.from_function(app, default_timeout=120)                        # upplåst: sidan ritas som förut
    at.session_state["sub_portfolio"] = "Allokering"
    at.session_state[pl.STATE] = True
    at.run()
    assert not at.exception, at.exception
    assert "HEMLIG.ST" in texts(at) and any(b.label == "🔒 Lås innehaven" for b in at.button)


def test_gate_renders_locked_without_a_secret():
    from streamlit.testing.v1 import AppTest

    def app():
        from portfolio.private_lock import gate
        gate()

    at = AppTest.from_function(app).run()
    assert not at.exception and any("låsta" in m.value for m in at.markdown)


# ── Metallerna ──────────────────────────────────────────────────────────────
def _row(**kw):
    base = dict(metal="guld", desc="tacka", count=1, weight=100.0, unit="g", purity=0.9999, cost_sek=100_000.0,
                date="2024-01-02")
    base.update(kw)
    return mt.new_row(**base)


def test_fine_grams_units_and_purity():
    assert mt.fine_grams(_row()) == pytest.approx(99.99)
    assert mt.fine_grams(_row(unit="oz", weight=1, count=2, purity=1.0)) == pytest.approx(2 * mt.TROY_OZ_G)
    assert mt.fine_grams(_row(metal="silver", unit="kg", weight=1, purity=0.999)) == pytest.approx(999.0)


def test_spot_in_sek_per_gram():
    idx = pd.date_range("2026-10-20", periods=3)
    series = {"USDSEK=X": pd.Series([10.0, 10.0, 10.5], index=idx), "GC=F": pd.Series([4000.0] * 3, index=idx),
              "SI=F": pd.Series([50.0] * 3, index=idx)}
    spot = mt.spot_sek_per_gram(lambda s, p: series[s])
    assert spot["guld"][0] == pytest.approx(4000 * 10.5 / mt.TROY_OZ_G) and spot["guld"][1] == "2026-10-22"
    assert mt.spot_sek_per_gram(lambda s, p: pd.Series(dtype=float)) == {}


def test_valuation_and_totals():
    rows = [_row(), _row(metal="silver", unit="kg", weight=1, purity=0.999, cost_sek=8000.0, date="2026-06-01")]
    spot = {"guld": (1300.0, "2026-10-23"), "silver": (12.0, "2026-10-23")}
    v = mt.valuation(rows, spot, today="2026-10-25")
    g, s = v
    assert g["Värde spot (kr)"] == round(99.99 * 1300) and g["Avkastning %"] == pytest.approx(30.0, abs=0.1)
    assert g["Årstakt %"] is not None and s["Årstakt %"] is None                # under ett år: ingen årstakt
    assert g["Break-even (kr/g)"] == pytest.approx(100_000 / 99.99, abs=0.1)
    tot = mt.totals(v)
    assert tot["alla"]["cost"] == 108_000 and tot["alla"]["value"] == g["Värde spot (kr)"] + s["Värde spot (kr)"]
    no_spot = mt.totals(mt.valuation(rows, {"guld": (1300.0, "x")}, today="2026-10-25"))
    assert no_spot["alla"]["value"] is None and no_spot["guld"]["value"] is not None


def test_new_row_validation():
    with pytest.raises(ValueError):
        _row(metal="platina")
    with pytest.raises(ValueError):
        _row(purity=999.9)                                                  # andel, inte promille
    with pytest.raises(ValueError):
        _row(weight=0)


def test_metals_section_renders(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import market_prices
    import storage
    idx = pd.date_range("2026-10-20", periods=3)
    prices = {"USDSEK=X": pd.Series([10.0] * 3, index=idx), "GC=F": pd.Series([4000.0] * 3, index=idx),
              "SI=F": pd.Series([50.0] * 3, index=idx)}
    monkeypatch.setattr(market_prices, "close", lambda s, p="1y": prices[s])          # återställs efter testet
    monkeypatch.setattr(storage, "load_json", lambda name, default=None: {"rows": [
        {"id": "a1", "metal": "guld", "desc": "tacka", "count": 1, "weight": 100.0, "unit": "g",
         "purity": 0.9999, "cost_sek": 100000.0, "date": "2024-01-02"}]})

    def app():
        from portfolio import metals
        metals.render_metals_section()

    at = AppTest.from_function(app, default_timeout=30).run()
    assert not at.exception
    assert len(at.dataframe) == 1 and any("Fysiska metaller" in m.value for m in at.markdown)
