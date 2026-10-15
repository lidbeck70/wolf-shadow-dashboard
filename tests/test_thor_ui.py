"""
Thor fas 3 — 🔨 THOR-fliken: navigationen, läget, rullgardinen och filtren, korten (rena HTML-funktioner)
och sidan i AppTest — listan, och analysblocket som öppnas för en aktie. Inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_candidates as fx  # noqa: E402
from thor import candidates as tc  # noqa: E402
from thor import cards  # noqa: E402
from thor import registry as reg  # noqa: E402
from thor import ui as tui  # noqa: E402
from ui import nav  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = {**fx.BLOBS, "regime": {"regime": "GRÖN", "generated": "2026-10-14T04:17"}, "loaded": "nu"}


def _rows():
    return tc.build(DATA, today="2026-10-15")


# ── Navigationen ─────────────────────────────────────────────────────────────
def test_nav_has_thor_after_valhall_without_sub_tabs():
    assert [k for k, _l in nav.TOP][:3] == ["home", "valhall", "thor"] and nav.TOP_LABEL["thor"] == "🔨 THOR"
    assert nav.options("thor") == [] and nav.resolve("thor") == ["thor"] and nav.slug(["thor"]) == "thor"
    src = open(os.path.join(ROOT, "wolf_panel.py"), encoding="utf-8").read()
    assert 'tabs["thor"]' in src and "render_thor_page()" in src
    from ovtlyr.ui.rules_page import _PANEL_GUIDE
    assert any(t == "THOR" for t, _r, _u in _PANEL_GUIDE)


# ── Ren logik ────────────────────────────────────────────────────────────────
def test_lage_items_and_pick_filter():
    rows = _rows()
    items = tui.lage_items(DATA, rows)
    assert [i[0] for i in items] == ["Swing-regim", "Norden", "Globalt", "Kandidater"]
    assert items[0][1] == "GRÖN" and items[1][1] == "1/2 GRÖN" and items[3][1] == "31 · 13 KÖP" and "2026-10-14" in items[3][3]
    assert tui.lage_items({}, [])[0][1] == "OKÄND" and tui.lage_items({}, [])[1][1] == "—"
    assert tui.pick_to_filter(tui.ALL) == (None, None) and tui.pick_to_filter("Swing (alla)") == (reg.SWING, None)
    assert tui.pick_to_filter("Viking Nine") == (None, "viking_nine")
    assert {c.family for c in tui.shown_rows(rows, "Råvara (alla)", [])} == {reg.RAVARA}
    assert all(c.status_class == tc.KOP and c.nordic for c in tui.shown_rows(rows, tui.ALL, ["bara KÖP", "bara Norden"]))
    assert [c.ticker for c in tui.shown_rows(rows, tui.ALL, ["bara nya"])] == ["AEM"]
    assert [c.strategy for c in tui.shown_rows(rows, "Long Screener", None)] == ["long"]


def test_cards_are_plain_html():
    rows = {(c.strategy, c.ticker): c for c in _rows()}
    h = cards.candidate_card(rows[("viking_nine", "HEXA-B.ST")])
    for text in ("HEXA-B.ST", "Hexagon", ">86<", "Viking Nine", "GOLDEN TICKET", "Teknik · GRÖN · MITTEN", "8/9 · 86",
                 "entry 110.00", "stopp 104.00", "R 2.4", "Pullback till EMA10"):
        assert text in h, text
    assert "line-through" in cards.candidate_card(rows[("quality", "SSAB-A.ST")])                 # SPÄRRAD
    assert "NY" in cards.candidate_card(rows[("hav", "AEM")]) and "gammal" not in cards.candidate_card(rows[("hav", "AEM")])
    stale = tc.build(DATA, today="2026-10-30")[0]
    assert "gammal · 2026-10-14" in cards.candidate_card(stale)
    v = cards.verdict_for(rows[("viking_nine", "HEXA-B.ST")], "viking_nine")
    assert "VAD SKA JAG GÖRA?" in v and ">KÖP<" in v and "Viking Nine: GOLDEN TICKET" in v and "körd 2026-10-14" in v
    none = cards.verdict_for(None, "wolf")
    assert "INGEN SIGNAL" in none and "Wolf" in none
    assert "&lt;b&gt;" in cards.chip("<b>", "#fff")                                               # HTML escapas
    strip = cards.phase_strip("HOPE", ["HOPE"], ["BELIEF"], ["EUPHORIA"])
    assert strip.count("<span") == 14 and "Hope" in strip
    assert "Cykelgrinden (quality): PASS" in cards.gate_card({"status": "PASS", "note": "köpfas"}, "quality")
    assert "Cykelgrinden: MANUAL" in cards.gate_card({}, "")
    head = cards.header_card({"yf": "BOL.ST", "name": "Boliden", "home_index": "^OMX", "market": "OMXS30", "segment": "materials",
                              "theme": "koppar", "theme_label": "Koppar", "theme_source": "BERSERK-listan", "sector_etf": "XLB"},
                             [rows[("berserk", "BOL.ST")]], held=True)
    for text in ("BOL.ST", "Boliden", "^OMX · OMXS30", "Material · RÖD · TOPP", "Koppar · BERSERK-listan", "INNEHAV", "finns i dag i"):
        assert text in head, text
    snap = cards.snapshot_row({"price": 312.5, "as_of": "2026-10-14", "ret_1m_pct": 4.2, "ret_3m_pct": -1.0, "atr_pct": 2.1,
                               "dist_ema50_pct": 3.0, "dist_ema200_pct": -2.0, "from_high_pct": -8.0})
    assert "312.50" in snap and "+4.2 %" in snap and "-8.0 %" in snap and cards.snapshot_row(None) == ""


# ── Sidan ────────────────────────────────────────────────────────────────────
def _df(n=260):
    idx = pd.bdate_range("2025-10-01", periods=n)
    c = 100 + np.arange(n) * 0.3
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1e6}, index=idx)


def _script():
    from thor import ui as t
    t.render_thor_page()


def _patch(monkeypatch):
    monkeypatch.setattr(tui, "load", lambda loader=None: dict(DATA))
    monkeypatch.setattr(tui, "_stores", lambda: {})
    monkeypatch.setattr(tui, "_ohlcv", lambda t: _df())
    monkeypatch.setattr(tui, "_phase", lambda t: ({"phase": "HOPE", "confidence": 69.0}, None))
    monkeypatch.setattr(tui, "_api", lambda: None)
    monkeypatch.setattr(tui, "_table", lambda: None)


def test_page_renders_the_list(monkeypatch):
    from streamlit.testing.v1 import AppTest
    _patch(monkeypatch)
    at = AppTest.from_function(_script)
    at.run(timeout=60)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "THOR" in text and "Swing-regim" in text and "31 KANDIDATER · 13 KÖP" in text and "VAD SKA JAG GÖRA?" not in text
    opens = [b for b in at.button if b.label == "Öppna →"]
    assert len(opens) == tui.PAGE and any(b.label.startswith("Visa fler") for b in at.button)
    assert "HEXA-B.ST" in text and "GOLDEN TICKET" in text


def test_opening_a_ticker_shows_the_analysis_block(monkeypatch):
    from streamlit.testing.v1 import AppTest
    _patch(monkeypatch)
    at = AppTest.from_function(_script)
    at.session_state["thor_ticker"] = "bol.st"
    at.run(timeout=90)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "VAD SKA JAG GÖRA?" in text and "BOL.ST" in text and "Kandidater (31)" in " ".join(e.label for e in at.expander)
    assert at.selectbox(key="thor_strategy").value == "BERSERK"                   # bästa raden: BERSERK KÖP
    labels = [b.label for b in at.button]
    assert "Öppna i BERSERK Regime →" in labels and "◀ Tillbaka till listan" in labels   # bara Regim-fliken ritas
    assert "Cykelgrinden" not in text                                               # Cykel ritas först när den öppnas
    assert "Koppar · BERSERK-listan" in text and "Råvara" in [t.label for t in at.tabs]


def _script_tabs():
    import streamlit as st
    from thor import dossier as d
    from thor import ui as t
    ident = d.identify("BOL.ST")
    stores = {"producers": {"producers": [{"ticker": "BOL", "name": "Boliden", "commodity": "Koppar", "price": 9000.0,
                                           "unit_cost": 6000.0, "ev_ebitda": 5.0, "nd_ebitda": 0.5, "mine_life": 12.0,
                                           "rp_ratio": 15.0}], "royalty": []},
              "holdings": {"long": [{"ticker": "BOL.ST", "name": "Boliden", "strategy": "Quality", "entry_price": 300.0,
                                     "shares": 10}]}}
    sheets = d.sheets_for("BOL.ST", stores)
    data = st.session_state.get("_data") or {}
    t._tab_cykel(st, data, ident, "quality", sheets, [])
    t._tab_arken(st, data, ident, "quality", sheets, [])
    t._tab_regler(st, data, ident, "viking_nine", sheets, [])
    t._tab_backtest(st, data, ident, "quality", sheets, [])


def test_cycle_sheets_and_rules_tabs_render(monkeypatch):
    from streamlit.testing.v1 import AppTest
    _patch(monkeypatch)
    at = AppTest.from_function(_script_tabs)
    at.session_state["_data"] = {"board": fx.BOARD}
    at.run(timeout=90)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Cykelgrinden (quality): PASS" in text and "Hope" in text and "^OMX: HOPE 69 %" in text
    assert "Rick Rule:" in text and len(at.dataframe) == 1                               # granskningen och positionen
    labels = [b.label for b in at.button]
    assert "Rick Rule →" in labels and "Öppna Backtest →" in labels
    assert any(e.label.startswith("Regler") for e in at.expander)                       # playbooken (Viking)


def test_deep_link_t_opens_the_ticker(monkeypatch):
    from streamlit.testing.v1 import AppTest
    _patch(monkeypatch)
    at = AppTest.from_function(_script)
    at.query_params["t"] = "ERIC-B.ST"
    at.run(timeout=90)
    assert not at.exception
    assert at.text_input(key="thor_ticker").value == "ERIC-B.ST"
    assert at.selectbox(key="thor_strategy").value == "Momentum Swing"            # rad 1 · setup A vinner
    text = " ".join(m.value for m in at.markdown)
    assert "Momentum Swing: rad 1 · setup A" in text and "Råvara" not in [t.label for t in at.tabs]


@pytest.mark.parametrize("fn", [tui._goto, tui._close, tui._more, tui._jump])
def test_callbacks_are_importable(fn):
    assert callable(fn)
