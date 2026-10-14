"""
Valhall fas 5 — ⚡ VALHALL-fliken: korten (rena HTML-funktioner), tabellerna, inläsningen ur Gisten och att
sidan ritas i alla tre underflikarna (AppTest, inget nätverk).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui import nav  # noqa: E402
from valhall import ui as vui  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROW = {"key": "technology", "label": "Teknik", "spdr": "XLK", "ucits": ["IUIT.L"],
       "global": {"status": "RÖD", "phase": "SVAG", "cykel": "TOPP"},
       "nordic": {"status": "GRÖN", "phase": "STARK", "cykel": "MITTEN", "cycle_start": False},
       "themes": []}
BOARD = {"generated": "2026-10-14T16:50:00+00:00",
         "board": {"segments": [ROW, {"key": "materials", "label": "Material", "spdr": "XLB", "ucits": [],
                                      "global": {"status": "GUL"}, "nordic": {"status": "RÖD"},
                                      "themes": [{"theme": "guld", "label": "Guld", "driver": "GC=F",
                                                  "state": {"status": "GRÖN", "phase": "STARK", "cykel": "TIDIG",
                                                            "cycle_start": True}}]}]},
         "cycle_starts": [{"segment": "Material · Guld", "where": "tema", "since": "2026-09-01"}],
         "swing": {"segments": {"technology": {"strategy": "Viking Nine", "picks": [{"ticker": "HEXA-B.ST",
                                                                                      "category": "GOLDEN TICKET"}]}},
                   "themes": {"guld": {"strategy": "BERSERK", "picks": [{"ticker": "BOL.ST", "setup": "S2 Vändning"}]}}},
         "buys": {"segments": {"technology": {"picks": [{"ticker": "MYCR.ST", "verdict": "BUY"}],
                                              "gate": {"rule": "ej RÖD", "status": "GRÖN", "open": True}},
                               "materials": {"picks": [{"ticker": "HOLM-B.ST", "verdict": "BUY", "gate_blocked": True}],
                                             "gate": {"rule": "ej RÖD", "status": "RÖD", "open": False}}},
                  "themes": {"guld": {"quality": {"picks": [{"ticker": "NEM"}]},
                                      "leverage": {"strategy": "Durrett", "picks": [{"ticker": "MMY.V", "bagger": True}]}}}},
         "long": {"segments": {"technology": {"picks": [{"ticker": "LAGR-B.ST", "signal": "STRONG BUY"}]}}},
         "niche": {"technology": {"berserk": [], "insider": [{"ticker": "X"}]}},
         "paper": {"since": "2026-10-13", "positions": 3, "orders": 1,
                   "by_strategy": [{"strategy": "long", "n": 2, "open": 2, "closed": 0, "avg_ret": 1.5, "win": 50.0,
                                    "avg_r": None, "avg_excess": 0.4, "avg_red_ret": 1.5}],
                   "by_segment": [{"strategy": "long", "segment": "technology", "n": 2}],
                   "by_status": [{"strategy": "long", "status_in": "GRÖN", "n": 2}],
                   "by_cykel": [{"strategy": "long", "cykel_in": "MITTEN", "n": 2}]}}
FAS4C = {"portfolios": [{"variant": "gron", "period": "alla", "cagr": 12.3, "dd": -18.4, "avg_n": 8.8, "bench_cagr": 8.4},
                        {"variant": "valhall", "period": "alla", "months": 0}]}


def test_nav_has_valhall_top_tab_after_home():
    assert [k for k, _l in nav.TOP][:2] == ["home", "valhall"]
    assert nav.options("valhall") == ["Kartan", "Papperskonton", "Backtester"]
    assert nav.STATE_KEY["valhall"] == "sub_valhall" and nav.resolve("valhall/papperskonton") == ["valhall", "Papperskonton"]
    src = open(os.path.join(ROOT, "wolf_panel.py"), encoding="utf-8").read()
    assert 'tabs["valhall"]' in src and "render_valhall_page(sub)" in src and '_sub("valhall")' in src


def test_segment_card_shows_status_picks_and_gate():
    sw, buys = BOARD["swing"]["segments"]["technology"], BOARD["buys"]["segments"]["technology"]
    html = vui.segment_card(ROW, sw, buys, BOARD["long"]["segments"]["technology"], BOARD["niche"]["technology"])
    for text in ("Teknik", "Norden GRÖN", "Globalt RÖD", "XLK → IUIT.L", "Swing (Viking Nine)", "HEXA-B.ST",
                 "GOLDEN TICKET", "MYCR.ST", "LAGR-B.ST", "Insider 1"):
        assert text in html, text
    assert "SPÄRRAD" not in html
    mat = BOARD["board"]["segments"][1]
    blocked = vui.segment_card(mat, {}, BOARD["buys"]["segments"]["materials"], {}, {})
    assert "SPÄRRAD — sektorn är RÖD" in blocked and "line-through" in blocked and "HOLM-B.ST" in blocked
    assert "—" in vui.segment_card({"key": "x"}, None, None, None, None)                   # tom ruta kraschar inte


def test_theme_card_and_chips():
    t = BOARD["board"]["segments"][1]["themes"][0]
    html = vui.theme_card(t, BOARD["swing"]["themes"]["guld"], BOARD["buys"]["themes"]["guld"])
    for text in ("Guld", "GC=F", "GRÖN", "CYKELSTART", "BOL.ST", "NEM", "Hävstång (Durrett)", "15-BAGGER"):
        assert text in html, text
    assert "&lt;b&gt;" in vui.pick_chips([{"ticker": "<b>"}])                                 # HTML escapas


def test_tables():
    t = vui.paper_table(BOARD["paper"]["by_strategy"], ("strategy",))
    assert list(t["Strategi"]) == ["Long Screener"] and "Mot riktmärke %" in t.columns
    p = vui.portfolio_table(FAS4C["portfolios"], "variant")
    assert len(p) == 1 and p.iloc[0]["CAGR %"] == 12.3                                       # tom variant hoppas över


def test_load_reads_every_blob_and_survives_errors():
    seen = []

    def loader(name, fb):
        seen.append(name)
        if name == "valhall_fas3.json":
            raise RuntimeError("nät")
        return {"name": name}

    out = vui.load(loader)
    assert set(seen) == set(vui.BLOBS.values()) and out["fas3"] is None and out["board"] == {"name": "valhall_board.json"}


def _script():
    import streamlit as st
    from valhall import ui as v
    v.render_valhall_page(st.session_state.get("_sub", "Kartan"))


@pytest.mark.parametrize("sub", ["Kartan", "Papperskonton", "Backtester"])
def test_page_renders_each_sub_tab(monkeypatch, sub):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(vui, "load", lambda loader=None: {"board": BOARD, "fas3": None, "fas4b": None,
                                                          "fas4c": FAS4C, "loaded": "nu"})
    at = AppTest.from_function(_script)
    at.session_state["_sub"] = sub
    at.run(timeout=30)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "VALHALL" in text
    if sub == "Kartan":
        assert "SEKTORERNA" in text and "Cykelstart nu" in text and "RÅVAROR · Material" in text
    elif sub == "Papperskonton":
        assert len(at.dataframe) == 2 and "Sedan 2026-10-13" in text
    else:
        assert "BESLUTEN" in text and len(at.dataframe) == 1


def test_page_without_runs_explains_what_to_do(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(vui, "load", lambda loader=None: {"board": None, "fas3": None, "fas4b": None, "fas4c": None,
                                                          "loaded": "nu"})
    at = AppTest.from_function(_script)
    at.run(timeout=30)
    assert not at.exception and "Valhall Scan" in " ".join(m.value for m in at.markdown)
