"""
LIDSKJALV PR 22 — Positioner: åtgärd i dag och ett kort per innehav (§4.9, §3.5.12–§3.5.14, beslut 10).

  · ordlistan i sammanhanget HOLDING: EXIT → SÄLJ, REDUCE → MINSKA, WATCH → VAKSAM, HOLD → HÅLL — och det lagrade
    engelska ordet står kvar ordagrant som grått källchip
  · kortet är EN st.markdown med id='pos-<slug>' och kursens siffror ('12000.00') i samma text
  · remsan "BEHÖVER ÅTGÄRD I DAG" listar exakt de innehav vars redan räknade signal är EXIT/REDUCE/WATCH och
    regelvaktens egna varningar ('>' escapad) — grön rad när inget behövs; signalen räknas en gång per innehav
  · ✕ är borta: confirm_delete('Ta bort', key='rm_<hink>_<ticker>') tar bort först efter "Ja, ta bort"
  · nyckeltalen behåller etiketter och format i kpis_holdings; att öppna sidan skriver ingenting
Inget nätverk: kurser och signaler stubbade (samma dict-former som holdings._compute_*), lagringen fejkad.
"""
import copy
import os
import re
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import holdings  # noqa: E402
import positions  # noqa: E402
from ui import vocab  # noqa: E402
from ui.components import source_chip  # noqa: E402

GREEN, RED, YELLOW, MAGENTA, DIM = holdings.GREEN, holdings.RED, holdings.YELLOW, holdings.MAGENTA, holdings.DIM
SEED = {"long": [
    {"id": "q1", "ticker": "EQX", "strategy": "Quality", "entry_price": 8.0, "shares": 1000, "sector": "Materials"},
    {"id": "q2", "ticker": "VOLV-B.ST", "strategy": "Quality", "entry_price": 280.0, "shares": 100},
    {"id": "q3", "ticker": "AZN.L", "strategy": "Quality", "entry_price": 11000.0, "shares": 10},
    {"id": "c1", "ticker": "NEM", "strategy": "Deep Contrarian", "entry_price": 40.0, "shares": 50,
     "sector": "Materials", "extras": {"tranches_deployed": 2}}],
    "swing": [{"id": "w1", "ticker": "SAND.ST", "strategy": "Wolf", "entry_price": 230.0, "shares": 40}],
    "momentum": [{"id": "m1", "ticker": "ERIC-B.ST", "strategy": "Momentum", "entry_price": 62.0, "shares": 300}],
    "cash": 25000}
PRICES = {"EQX": (10.0, 1.2), "VOLV-B.ST": (300.0, -2.6), "AZN.L": (12000.0, 0.4), "NEM": (44.0, -0.8),
          "SAND.ST": (214.5, -1.9), "ERIC-B.ST": (71.3, 0.9)}
SIGNALS = {
    "EQX": {"signal": "HOLD ✓", "color": GREEN, "detail": "Price +12.4% vs EMA200"},
    "VOLV-B.ST": {"signal": "REDUCE", "color": MAGENTA, "detail": "5d below EMA200"},
    "AZN.L": {"signal": "HOLD ✓", "color": GREEN, "detail": "Price +6.1% vs EMA200"},
    "NEM": {"signal": "WATCH", "color": YELLOW, "detail": "EMA50 below EMA200"},
    "SAND.ST": {"signal": "EXIT", "color": RED, "detail": "Red regime"},
    "ERIC-B.ST": {"signal": "HOLD ✓", "color": GREEN, "detail": "Price +3.0% vs EMA200"},
}


# ── Ordlistan och chippen ────────────────────────────────────────────────────
@pytest.mark.parametrize("stored, main", [("EXIT", vocab.SALJ), ("SELL", vocab.SALJ), ("REDUCE", vocab.MINSKA),
                                          ("WATCH", vocab.VAKSAM), ("HOLD", vocab.HALL), ("HOLD ✓", vocab.HALL)])
def test_holding_vocab_maps_the_stored_word_and_keeps_it_as_a_grey_source_chip(stored, main):
    assert holdings.signal_main(stored) == main
    if stored != "HOLD ✓":
        assert vocab.main(stored, vocab.HOLDING) == main
    html = holdings.signal_chips(stored)
    shown = {vocab.SALJ: ">SÄLJ<", vocab.MINSKA: ">MINSKA<", vocab.VAKSAM: ">● VAKSAM<", vocab.HALL: ">HÅLL ✓<"}[main]
    assert shown in html
    assert html.endswith(source_chip(stored))              # det lagrade ordet ordagrant, grått, aldrig omfärgat
    assert html.index(shown) < html.index(f">{stored}<")   # huvudordet först, källans ord bredvid


def test_watch_on_a_holding_is_vaksam_not_bevaka_and_missing_is_saknas():
    assert vocab.main("WATCH", vocab.VERDICT) == vocab.BEVAKA and holdings.signal_main("WATCH") == vocab.VAKSAM
    na = holdings.signal_chips("N/A", "ingen kurs")
    assert ">Saknas<" in na and 'title="ingen kurs"' in na and ">N/A<" not in na      # inget källchip för "N/A"
    assert holdings.signal_main("N/A") is None and holdings.signal_main(None) is None


def test_the_card_is_one_html_string_with_its_id_the_price_digits_and_only_existing_numbers():
    h = {"ticker": "AZN.L", "strategy": "Quality", "entry_price": 11000.0, "shares": 10, "sector": "Health Care",
         "_portfolio_key": "long", "tranches_deployed": 0}
    html = holdings.holding_card_html(h, {"price": 12000.0, "change_1d": 0.4, "df": None}, SIGNALS["AZN.L"], "long")
    assert html.startswith('<div class="pos-card" id="pos-azn-l-long"') and html.count("<div class=\"pos-card\"") == 1
    for text in ("12000.00", "(+0.4%)", "120,000", "+9.1%", "(+10,000)", "Health Care", "10 st",
                 "Price +6.1% vs EMA200", ">HÅLL ✓<", ">HOLD ✓<"):
        assert text in html, text
    assert holdings.card_id("long", "VOLV-B.ST") == "pos-volv-b-st-long" != holdings.card_id("momentum", "VOLV-B.ST")
    # ingen kurs: Saknas — aldrig en tyst 0.00
    none = holdings.holding_card_html(h, {"price": 0, "change_1d": 0, "df": None}, {"signal": "N/A"}, "long")
    assert "Saknas" in none and "0.00" not in none
    # tickern escapas
    bad = holdings.holding_card_html({**h, "ticker": "<b>X"}, {"price": 1.0}, SIGNALS["AZN.L"], "long")
    assert "<b>X" not in bad and "&lt;b&gt;X" in bad


def test_third_line_shows_only_what_the_old_row_showed_for_the_alpha_regime():
    h = {"ticker": "NEM", "strategy": "Deep Contrarian", "entry_price": 40.0, "shares": 50, "sector": "Materials",
         "_portfolio_key": "long", "tranches_deployed": 2}
    live = {"price": 44.0, "change_1d": -0.8, "df": None}
    q = holdings.holding_card_html(h, live, SIGNALS["NEM"], "long",
                                   {"type": "quality", "verdict": "BUY", "passed": 6, "total": 7})
    assert ">KÖP<" in q and ">BUY<" in q and "6/7 signals" in q and "EMA50 below EMA200" in q
    c = holdings.holding_card_html(h, live, SIGNALS["NEM"], "long",
                                   {"type": "contrarian", "stage": "ACCUMULATE_2", "color": "#FF6B3D"})
    assert ">ACC·2<" in c and c.count(">●<") == 2 and c.count(">○<") == 1           # två av tre trancher
    err = holdings.holding_card_html(h, live, SIGNALS["NEM"], "long", {"type": "error", "error": "timeout"})
    assert ">Saknas<" in err and "Alpha Regime: timeout" in err
    plain = holdings.holding_card_html(h, live, SIGNALS["NEM"], "long")
    assert "signals" not in plain and ">●<" not in plain                              # utan regim: ingen regimdel


# ── Remsan (rena funktioner) ─────────────────────────────────────────────────
def _items(sigs: dict, rules=()) -> list:
    out = [{"kind": "holding", "ticker": t, "strategy": "Quality", "pkey": "long", "signal": s} for t, s in sigs.items()]
    return out + [{"kind": "rule", "level": lv, "text": tx, "strategy": "Quality"} for lv, tx in rules]


def test_strip_lists_exit_reduce_watch_in_that_order_plus_the_rule_warnings_escaped():
    items = _items({"A": SIGNALS["NEM"], "B": SIGNALS["EQX"], "C": SIGNALS["VOLV-B.ST"], "D": SIGNALS["SAND.ST"]},
                   [("warning", "⚠ >10% position: A (39.1%)")])
    html = holdings.strip_html(items)
    assert holdings.STRIP_KICKER in html
    assert re.findall(r'data-jump="([^"]+)"', html) == ["pos-d-long", "pos-c-long", "pos-a-long", "pos-rules-quality"]
    assert "B" not in re.findall(r">([A-Z]) · ", html)                  # HÅLL hör inte hemma i remsan
    assert "⚠ &gt;10% position: A (39.1%)" in html and ">10%" not in html.replace("&gt;10%", "")
    assert html.index(">SÄLJ<") < html.index(">MINSKA<") < html.index(">● VAKSAM<") < html.index(">REGEL<")
    assert html.count('class="pos-more"') == 1 and "+1 till ↓" in html   # tre rader syns, resten bakom "+n till"
    assert holdings.STRIP_EMPTY not in html


def test_strip_is_green_when_nothing_needs_action_and_honest_when_a_signal_is_missing():
    ok = holdings.strip_html(_items({"A": SIGNALS["EQX"], "B": {"signal": "HOLD", "detail": "Trend intact"}}))
    assert holdings.STRIP_EMPTY in ok and "data-jump" not in ok and "pos-ok" in ok
    gap = holdings.strip_html(_items({"A": SIGNALS["EQX"], "B": {"signal": "N/A", "color": DIM}}))
    assert "✓ Inga säljlägen i dag" in gap and holdings.STRIP_EMPTY not in gap       # inte "alla HÅLL" när B saknas
    assert "Saknas" in gap and "signal för 1 av 2 innehav: B" in gap
    assert holdings.strip_html([]) == ""                                              # inga innehav: ingen remsa


def test_strip_derives_nothing_it_only_reads_the_collected_signals(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("remsan räknade en signal")
    for fn in ("_signal_for_holding", "_compute_long_signal", "_compute_swing_signal", "_compute_ovtlyr_signal",
               "_fetch_live_price", "_get_regime_signal"):
        monkeypatch.setattr(holdings, fn, boom)
    assert "pos-strip" in holdings.strip_html(_items({"A": SIGNALS["SAND.ST"]}))


# ── Sidan (AppTest) ──────────────────────────────────────────────────────────
@pytest.fixture
def page(monkeypatch):
    """Positioner ur en fejkad lagring: sparningar spelas in, kurser och signaler stubbade, inget nätverk."""
    from streamlit.testing.v1 import AppTest

    import earnings_calendar
    import gist_storage
    import refresh_ui
    import risk_dashboard
    import storage
    data = {"holdings": copy.deepcopy(SEED)}
    saves, calls = [], []
    monkeypatch.setattr(storage, "load_json", lambda n, default=None: copy.deepcopy(data.get(n, default)))
    monkeypatch.setattr(storage, "legacy_gist", lambda *a, **k: None)
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(gist_storage, "load_blob", lambda n, default=None: default)

    def _save(n):
        import streamlit as st
        saves.append((n, copy.deepcopy(st.session_state[n])))
        return storage.SaveResult(n, "abc1234def", "https://x", "2026-11-23T10:00:00Z")
    monkeypatch.setattr(storage, "save_session", _save)

    def _signal(h, df):
        calls.append(h["ticker"])
        return dict(data.get("signals", SIGNALS)[h["ticker"]])
    monkeypatch.setattr(holdings, "_fetch_live_price", lambda t, period="6mo": {"price": PRICES[t][0],
                                                                                "change_1d": PRICES[t][1], "df": None})
    monkeypatch.setattr(holdings, "_signal_for_holding", _signal)
    monkeypatch.setattr(holdings, "_get_regime_signal", lambda t, s: None)
    monkeypatch.setattr(holdings, "_fetch_returns_for_correlation", lambda t, period="6mo": pd.DataFrame())
    monkeypatch.setattr(risk_dashboard, "render_risk_dashboard", lambda d: None)
    monkeypatch.setattr(earnings_calendar, "render_earnings_calendar", lambda d: None)

    def make():
        def app(root):
            import sys as _s
            _s.path.insert(0, root)
            import holdings as h
            h.render_holdings_page()
        at = AppTest.from_function(app, args=(ROOT,), default_timeout=90)
        at.run()
        assert not at.exception, at.exception
        return at
    make.data, make.saves, make.calls = data, saves, calls
    return make


def _strip(at) -> str:
    found = [m.value for m in at.markdown if 'class="pos-strip' in m.value]
    assert len(found) == 1
    return found[0]


def test_page_strip_lists_exactly_the_exit_reduce_watch_holdings_and_the_rule_warnings(page):
    at = page()
    strip = _strip(at)
    assert re.findall(r'data-jump="([^"]+)"', strip) == [
        "pos-sand-st-swing", "pos-volv-b-st-long", "pos-nem-long",                 # SÄLJ, MINSKA, VAKSAM
        "pos-rules-quality", "pos-rules-quality", "pos-rules-deep-contrarian"]     # regelvaktens tre varningar
    assert "SAND.ST · Wolf · Red regime" in strip and "+3 till ↓" in strip
    assert "⚠ &gt;10% position: EQX (" in strip and "⚠ Sector &gt;25%: Materials (100.0%)" in strip
    # regelvaktens st.warning står kvar (samma text; '>' escapad för markdown så att den syns)
    warn = [w.value for w in at.warning if ">10% position" in w.value]
    assert warn and "EQX (" in warn[0]
    assert sorted(page.calls) == sorted(t for b in ("long", "swing", "momentum") for t in
                                        (r["ticker"] for r in SEED[b]))            # en signal per innehav, ingen till
    ids = re.findall(r'<div class="pos-card" id="([^"]+)"', " ".join(m.value for m in at.markdown))
    for jump in re.findall(r'data-jump="(pos-[a-z0-9-]+-(?:long|swing))"', strip):
        assert ids.count(jump) == 1                                                # varje rad hoppar till ett kort
    assert any(f'id="{holdings.rules_id("Quality")}"' in m.value for m in at.markdown)


def test_page_strip_is_green_when_every_holding_holds(page):
    page.data["holdings"] = {"swing": [SEED["swing"][0]], "cash": 0}
    page.data["signals"] = {"SAND.ST": {"signal": "HOLD", "color": GREEN, "detail": "Trend intact"}}
    at = page()
    strip = _strip(at)
    assert holdings.STRIP_EMPTY in strip and "data-jump" not in strip


def test_each_holding_is_one_card_markdown_with_its_id_and_the_price_digits(page):
    at = page()
    cards = [m.value for m in at.markdown if 'class="pos-card"' in m.value]
    assert len(cards) == 6 and all(c.count('class="pos-card"') == 1 for c in cards)
    azn = [c for c in cards if 'id="pos-azn-l-long"' in c]
    assert len(azn) == 1 and "12000.00" in azn[0]                       # kursen i pence, som förut
    assert any('id="pos-eqx-long"' in c and "10.00" in c and ">HÅLL ✓<" in c for c in cards)
    assert any('id="pos-sand-st-swing"' in c and ">SÄLJ<" in c and ">EXIT<" in c for c in cards)
    labels = [e.label for e in at.expander]
    assert {"⚙ Edit EQX", "⚙ Edit SAND.ST", "⚙ Edit AZN.L"} <= set(labels)
    assert at.number_input(key="edit_stop_long_EQX").label == "Stopp"          # nycklarna och etiketterna kvar
    assert at.number_input(key="edit_target_long_EQX").label == "Mål"
    assert at.text_input(key="edit_sec_long_EQX").label == "Sektor"
    assert at.button(key="save_edit_long_EQX").label == "Save"


def test_the_x_is_gone_and_ta_bort_removes_only_after_confirming(page):
    at = page()
    assert not [b for b in at.button if b.label == "✕" or b.key == "rm_long_EQX"]
    confirm = at.button(key="rm_long_EQX__confirm")
    assert confirm.label == "Ja, ta bort"
    assert not page.saves
    assert "EQX" in [r["ticker"] for r in at.session_state["holdings"]["long"]]
    confirm.click().run()
    assert not at.exception, at.exception
    assert len(page.saves) == 1 and page.saves[0][0] == "holdings"
    assert "EQX" not in [r["ticker"] for r in page.saves[0][1]["long"]]           # bara EQX, bara efter "Ja"
    assert "VOLV-B.ST" in [r["ticker"] for r in page.saves[0][1]["long"]]
    assert not any('id="pos-eqx-long"' in m.value for m in at.markdown)


def _blocks(node):
    yield node
    for c in (node.children.values() if isinstance(getattr(node, "children", None), dict) else
              getattr(node, "children", None) or []):
        yield from _blocks(c)


def _path_to(node, want, path=()):
    """Förfäderna (rot → förälder) till den första noden där want(nod) är sann, annars None."""
    if want(node):
        return list(path)
    ch = getattr(node, "children", None)
    for c in (ch.values() if isinstance(ch, dict) else ()):
        found = _path_to(c, want, path + (node,))
        if found is not None:
            return found
    return None


def test_strip_slot_is_a_container_so_a_rerun_keeps_the_old_strip_in_place(page):
    """Remsans plats är en st.container() i kolumnen, inte en st.empty(): en tom plats blankar förra körningens
    remsa direkt vid omkörningen tills sektionerna är klara, och allt under hoppar ~280 px upp och tillbaka."""
    at = page()
    path = _path_to(at.main, lambda n: type(n).__name__ == "Markdown" and 'class="pos-strip' in n.value)
    assert path is not None
    parent, grand = path[-1], path[-2]
    assert type(parent).__name__ == "Block" and parent.type == "flex_container"   # containern …
    assert type(grand).__name__ == "Column"                                         # … i remsans kolumn
    assert any(str(getattr(getattr(b, "proto", None), "id", "")).endswith("-pos_top") for b in path)
    assert len(parent.children) == 1                                                # bara remsan i platsen
    at.run()                                                                         # omkörning: samma plats
    assert not at.exception, at.exception
    assert len([m for m in at.markdown if 'class="pos-strip' in m.value]) == 1
    src = open(os.path.join(ROOT, "holdings.py"), encoding="utf-8").read()
    head = src[src.index("def _render_summary_header"):src.index("def _fetch_returns_for_correlation")]
    assert "strip_col.container()" in head and "strip_col.empty()" not in head


def test_kpis_keep_labels_and_format_inside_kpis_holdings(page):
    at = page()
    box = [b for b in _blocks(at.main) if str(getattr(getattr(b, "proto", None), "id", "")).endswith("-kpis_holdings")]
    assert len(box) == 1
    inside = {m.label: m.value for m in _blocks(box[0]) if type(m).__name__ == "Metric"}
    eqx, volv, azn = 10.0 * 1000 * positions.FX_TO_SEK["USD"], 300.0 * 100, 12000.0 / 100 * 10 * positions.FX_TO_SEK["GBP"]
    invested = eqx + volv + azn + 44.0 * 50 * positions.FX_TO_SEK["USD"] + 214.5 * 40 + 71.3 * 300
    assert list(inside) == ["Total Holdings", "Invested", "Cash", "Portfolio Total"]
    assert inside["Total Holdings"] == "6" and inside["Cash"] == "25,000 SEK"
    assert inside["Invested"] == f"{invested:,.0f} SEK" and inside["Portfolio Total"] == f"{invested + 25000:,.0f} SEK"
    assert at.number_input(key="holdings_cash_input").label == "Cash (SEK)"


def test_opening_the_page_writes_nothing_and_draws_the_head_row(page):
    at = page()
    assert page.saves == []                                              # inget sparat när sidan öppnas
    assert "morning" not in at.session_state                              # rundan rörs inte
    heads = [m.value for m in at.markdown if "wolf-head" in m.value]
    assert heads and ">POSITIONER</h2>" in heads[0] and "<h1" not in heads[0]
    assert any("Alla strategiers positioner · säljreglerna · signalerna" in m.value for m in at.markdown)
    assert not [b for b in at.button if (b.key or "").startswith("morning_")]   # pillret bara under rundan


def test_sections_are_labels_not_rules_and_metals_get_their_label():
    src = open(os.path.join(ROOT, "holdings.py"), encoding="utf-8").read()
    body = src[src.index("def render_holdings_page"):]
    assert "<hr" not in body and 'section("Risk")' in body and 'section("Rapporter")' in body
    assert 'section("Korrelation")' in body and "st.button(\"✕\"" not in src
    panel = open(os.path.join(ROOT, "wolf_panel.py"), encoding="utf-8").read()
    branch = panel.split("if gate():", 1)[1].split("elif sub ==", 1)[0]
    assert 'section("Fysiska metaller")' in branch and 'st.markdown("---")' not in branch
    assert branch.index("render_holdings_page") < branch.index('section("Fysiska metaller")') \
        < branch.index("render_metals_section") < branch.index('routine.footer("holdings")')
