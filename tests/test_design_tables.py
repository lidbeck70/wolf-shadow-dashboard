"""
LIDSKJALV PR 24 — tabellerna på telefonen (docs/design_lidskjalv.md §3.5.15, §7 rad 24).

  · ui/tables.table ritar EN st.dataframe i tbl_<nyckel>: första spalten fastnålad, saknat som "—", utan index.
    Från sju spalter visar den huvudspalterna (column_order) med reglaget "Visa alla kolumner" — tabellens data är
    ändå hela df i samma ordning, med reglaget av och på
  · 👀 Övervakning: urvalet har Avstånd och varje spalt ett test läser (läses ur testfilerna); Läget färgas som förut
  · 🎯 Mot målet: en strategi med för få affärer har None i raderna — tabellen visar "—", aldrig "None"/"nan"
  · ⚙️ Inställningar: Reglerna (låsta) är ett kort per strategi ur rules_rows(), varje värde ordagrant — ingen tabell
  · inget skrivs när sidorna öppnas; CSS:en ligger i skalet och är avgränsad till tbl_-behållarna
"""
import json
import os
import re
import sys
from html import escape, unescape

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_watch_ui as twu  # noqa: E402
from ui import settings as us  # noqa: E402
from ui import shell, tables  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIDE = ["Aktie", "Strategi", "Läge", "Ord", "Avstånd", "Nivåer", "Körd", "Sedan", "Vid tillägg"]


def _src(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _columns(df_el) -> dict:
    """Dataframe-protons spaltinställningar (JSON) som dict."""
    return json.loads(df_el.proto.columns or "{}")


def _box_key(at, df_el):
    """Nyckeln på behållaren som håller tabellen (tbl_<nyckel>), ur elementträdet."""
    def walk(node, box):
        for ch in (getattr(node, "children", None) or {}).values():
            pid = getattr(getattr(ch, "proto", None), "id", "") or ""
            mark = f"-{tables.KEY_PREFIX}"
            here = pid[pid.rindex(mark) + 1:] if mark in pid else box
            if ch is df_el:
                return here
            got = walk(ch, here)
            if got:
                return got
        return None
    return walk(at.main, None)


# ── Hjälparen ────────────────────────────────────────────────────────────────
def _demo():
    import pandas as pd
    import streamlit as st

    from ui import tables
    df = pd.DataFrame([{c: f"{c} {i}" for c in ["Aktie", "Strategi", "Läge", "Ord", "Avstånd", "Nivåer", "Körd",
                                                 "Sedan", "Vid tillägg"]} for i in range(3)])
    df.loc[1, "Ord"] = None
    tables.table(df, "demo", st.session_state.get("primary", ["Läge", "Avstånd", "Nivåer"]))


def test_visible_is_the_first_column_then_the_primary_columns_in_their_order():
    assert tables.visible(WIDE, ["Nivåer", "Läge"]) == ["Aktie", "Nivåer", "Läge"]
    assert tables.visible(WIDE, ["Saknas", "Avstånd"]) == ["Aktie", "Avstånd"]
    assert tables.visible(WIDE, ["Aktie", "Läge", "Läge"]) == ["Aktie", "Läge"]     # första spalten och dubbletter en gång
    assert tables.visible(WIDE[:6], ["Läge"]) is None                    # under sju spalter: alla
    assert tables.visible(WIDE, None) is None and tables.visible(WIDE, []) is None
    assert tables.visible(WIDE, WIDE) is None                           # urvalet är redan alla
    assert tables.SUBSET_FROM == 7 and tables.TOGGLE == "Visa alla kolumner" and tables.MISSING == "—"


def test_helper_draws_one_dataframe_with_the_full_data_whatever_the_toggle():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_demo).run()
    assert not at.exception
    assert len(at.dataframe) == 1 and len(at.toggle) == 1
    tog = at.toggle(key="tbl_demo_all")
    assert tog.label == "Visa alla kolumner" and tog.value is False
    df = at.dataframe[0]
    assert list(df.value.columns) == WIDE and len(df.value) == 3            # datan är hel, i samma ordning
    assert list(df.proto.column_order) == ["Aktie", "Läge", "Avstånd", "Nivåer"]
    assert _box_key(at, df) == "tbl_demo"
    tog.set_value(True).run()
    assert not at.exception and len(at.dataframe) == 1
    assert list(at.dataframe[0].proto.column_order) == []                  # alla spalter
    assert list(at.dataframe[0].value.columns) == WIDE
    at.toggle(key="tbl_demo_all").set_value(False).run()
    assert list(at.dataframe[0].proto.column_order) == ["Aktie", "Läge", "Avstånd", "Nivåer"]


def test_first_column_is_pinned_missing_is_a_dash_and_the_index_hidden():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_demo).run()
    df = at.dataframe[0]
    cfg = _columns(df)
    assert cfg["Aktie"]["pinned"] is True
    assert sum(1 for k, v in cfg.items() if isinstance(v, dict) and v.get("pinned")) == 1
    assert df.proto.placeholder == "—"
    assert any(v.get("hidden") for k, v in cfg.items() if k.startswith("_index") or k == "_index")
    assert df.value["Ord"].isna().tolist() == [False, True, False]         # None står kvar i datan


def test_under_seven_columns_there_is_no_toggle_and_every_column_shows():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_demo)
    at.session_state["primary"] = None
    at.run()
    assert not at.toggle and len(at.dataframe) == 1 and list(at.dataframe[0].proto.column_order) == []


# ── 👀 Övervakning ───────────────────────────────────────────────────────────
def _read_by_tests() -> set:
    """Spalterna testerna läser ur Övervakningens tabell (df["…"] / .value["…"]) — ur testfilerna själva."""
    out = set()
    for f in ("test_watch_ui.py", "test_thor_watch.py"):
        out |= set(re.findall(r'(?:\bdf|\.value)\["([^"]+)"\]', _src("tests", f)))
    return out


def test_watch_subset_has_distance_and_every_column_a_test_reads(monkeypatch):
    from portfolio import watch_ui as wu
    read = _read_by_tests()
    assert {"Aktie", "Läge", "Strategi", "Entry/Stopp", "Strategins ord"} <= read      # läsningen hittar dem
    assert read <= set(wu.PRIMARY) and "Avstånd" in wu.PRIMARY
    at, repo = twu._app(monkeypatch, twu.ITEMS)
    before = json.dumps(repo, sort_keys=True)
    at.run(timeout=60)
    assert not at.exception and len(at.dataframe) == 1
    assert json.dumps(repo, sort_keys=True) == before                              # öppna sidan skriver inget
    df = at.dataframe[0]
    assert _box_key(at, df) == "tbl_watch"
    cols = list(df.value.columns)
    assert cols[:9] == ["Aktie", "Strategi", "Läge", "Strategins ord", "Avstånd", "Entry/Stopp", "Körd", "Sedan",
                        "Vid tillägg"]
    shown = list(df.proto.column_order)
    # beslutet först: läget och avståndet på telefonens första skärm, strategin och ordet efter nivåerna
    assert shown == ["Aktie", "Läge", "Avstånd", "Entry/Stopp", "Strategi", "Strategins ord", "Anteckning"]
    assert (read & set(cols)) <= set(shown) and _columns(df)["Aktie"]["pinned"] is True
    styles = df.proto.arrow_data.styler.styles
    assert "#2d8a4e" in styles and "font-weight: 700" in styles                   # lägets färg som förut
    at.toggle(key="tbl_watch_all").set_value(True).run(timeout=60)
    assert not at.exception and list(at.dataframe[0].proto.column_order) == []
    assert list(at.dataframe[0].value.columns) == cols
    assert json.dumps(repo, sort_keys=True) == before                              # reglaget sparar inget


def test_watch_prices_join_the_subset(monkeypatch):
    import market_prices
    at, repo = twu._app(monkeypatch, twu.ITEMS)
    s = pd.Series([100.0, 115.5], index=pd.to_datetime(["2026-10-27", "2026-10-28"]))
    monkeypatch.setattr(market_prices, "closes", lambda tickers, period="1y": {"HEXA-B.ST": s})
    at.run(timeout=60)
    at.button(key="watch_prices").click().run(timeout=60)
    assert not at.exception and len(at.dataframe) == 1
    shown = list(at.dataframe[0].proto.column_order)
    assert {"Kurs", "Till entry", "Över stopp"} <= set(shown) and "Körd" not in shown and "Vid tillägg" not in shown
    assert at.dataframe[0].value["Kurs"].tolist()[0] == "115.50"


# ── 🎯 Mot målet ─────────────────────────────────────────────────────────────
PAPER = {"by_strategy": [
    {"strategy": "viking", "closed": 25, "closed_win": 40.0, "avg_r": 0.3, "payoff": 2.1, "unit_cagr": 12.0},
    {"strategy": "wolf_arc", "closed": 5, "closed_win": 80.0, "avg_r": 2.0, "unit_cagr": 90.0}]}
TRADES = [{"entry_date": "2026-01-01", "exit_date": "2026-02-01", "pnl_amount": 100, "pnl_pct": 1.0,
           "exit_price": 1.0, "strategy": "viking_nine"}]


def _goal():
    from portfolio import mot_malet
    mot_malet.render_mot_malet_page()


def test_goal_table_shows_a_dash_for_none_and_keeps_the_rows(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from portfolio import mot_malet as mm
    repo = twu.fake_repo(monkeypatch, {"settings": {"capital": None}, "copilot_journal": {"entries": []}})
    before = json.dumps(repo, sort_keys=True)
    rows = mm.strategy_rows(PAPER, TRADES)
    wolf = next(r for r in rows if r["Strategi"] == "Wolf")
    assert wolf["Papper vinst %"] is None and wolf["Papper årstakt %"] is None          # raderna har None
    at = AppTest.from_function(_goal, default_timeout=60)
    at.session_state["goal_data"] = {"paper": PAPER, "trades": TRADES}
    at.run()
    assert not at.exception, at.exception
    assert len(at.dataframe) == 1 and json.dumps(repo, sort_keys=True) == before
    df = at.dataframe[0]
    assert _box_key(at, df) == "tbl_goal" and not at.toggle(key="goal_backtests").value
    assert [t.key for t in at.toggle] == ["goal_backtests"]                         # inget urval här
    assert df.proto.placeholder == "—" and _columns(df)["Strategi"]["pinned"] is True
    assert not df.proto.arrow_data.styler.display_values                            # ingen Styler som skriver "None"
    assert list(df.value.columns) == list(rows[0]) and df.value["Strategi"].tolist() == [r["Strategi"] for r in rows]
    assert df.value.isna().values.tolist() == pd.DataFrame(rows).isna().values.tolist()
    text = " ".join(m.value for m in at.markdown)
    assert "None" not in text and "nan" not in text.split()
    assert mm.strategy_rows(PAPER, TRADES) == rows                                  # raderna oförändrade


# ── ⚙️ Inställningar ──────────────────────────────────────────────────────────
def _settings():
    from ui.settings import render_settings_page
    render_settings_page()


def _settings_app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import data_health
    monkeypatch.setattr(data_health, "run_all_probes", lambda: [])                  # inga nätverksprober
    repo = twu.fake_repo(monkeypatch, {"settings": {"capital": None}})
    return AppTest.from_function(_settings, default_timeout=60), repo


def _cards(at) -> list:
    """Regelkorten: (kortets HTML) i ordning, ur markdownen som håller kortlistan."""
    (md,) = [m.value for m in at.markdown if "wolf-rule-cards" in m.value]
    return re.findall(r'<div class="wolf-rule-card".*?(?=<div class="wolf-rule-card"|$)', md, flags=re.S)


def test_settings_rules_are_one_card_per_playbook_with_every_value_verbatim(monkeypatch):
    import strategy_rules as sr
    at, repo = _settings_app(monkeypatch)
    before = json.dumps(repo, sort_keys=True)
    at.run()
    assert not at.exception, at.exception
    assert json.dumps(repo, sort_keys=True) == before                              # öppna sidan skriver inget
    rows = us.rules_rows()
    cards = _cards(at)
    assert len(cards) == len(rows) == len(sr.PLAYBOOKS)
    for card, row in zip(cards, rows):
        text = unescape(re.sub(r"<[^>]+>", "\x00", card))
        parts = [p for p in text.split("\x00") if p]
        assert parts[0] == row["Strategi"]                                         # rubriken
        for k, v in row.items():
            if k != "Strategi":
                assert parts[parts.index(k) + 1] == str(v), (row["Strategi"], k)  # etiketten, sedan värdet helt
                assert escape(str(v), quote=False) in card
    assert [c for c in cards if "…" in c] == []
    # ingen tabell för reglerna; Benchmark är sidans enda tabell (i tbl_benchmarks)
    assert not [d for d in at.dataframe if "Risk per affär" in d.value.columns]
    bench = [d for d in at.dataframe if "Standardbolag" in d.value.columns]
    assert len(bench) == 1 and _box_key(at, bench[0]) == "tbl_benchmarks"
    assert list(bench[0].value.columns) == ["Sida", "Benchmark", "Standardbolag"]
    assert _columns(bench[0])["Sida"]["pinned"] is True and bench[0].proto.placeholder == "—"
    assert not [t for t in at.toggle if (t.key or "").startswith(tables.KEY_PREFIX)]   # tre spalter: inget reglage
    assert at.number_input(key="settings_capital").value == 0.0


def test_rule_cards_escape_every_value_and_have_a_12px_floor():
    html = tables.cards_html([{"Strategi": "<b>X</b>", "Stopp": "Runway < 18 månader & mer", "Tom": None}],
                             "Strategi")
    assert "&lt;b&gt;X&lt;/b&gt;" in html and "Runway &lt; 18 månader &amp; mer" in html and "<b>" not in html
    assert "—" in html and "None" not in html and "\n\n" not in html and "\n" not in html
    assert all(int(s) >= 12 for s in re.findall(r"font-size:(\d+)px", html))
    assert "repeat(auto-fill,minmax(min(320px,100%),1fr))" in html                 # 1 spalt på telefonen, 2–3 på datorn
    full = tables.cards_html(us.rules_rows(), "Strategi")
    assert full.count('class="wolf-rule-card"') == len(us.rules_rows())


def test_settings_source_draws_the_cards_from_rules_rows_and_rules_rows_is_unchanged():
    src = _src("ui", "settings.py")
    assert 'tables.cards_html(rules_rows(), "Strategi")' in src
    assert "st.dataframe(pd.DataFrame(rules_rows())" not in src
    rows = us.rules_rows()
    assert list(rows[0]) == ["Strategi", "Risk per affär", "Position", "Max positioner", "Stopp",
                             "Hårt tak (allokeraren)"]


# ── CSS:en ───────────────────────────────────────────────────────────────────
def _rules(css: str) -> dict:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out = {}
    for sels, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        decls = {k.strip(): v.strip() for k, v in (d.split(":", 1) for d in body.split(";") if ":" in d)}
        for s in sels.split(","):
            out.setdefault(" ".join(s.split()), {}).update(decls)
    return out


KEYFRAMES = r"@keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}"


def test_table_css_sits_in_the_shell_and_only_reaches_the_tbl_boxes():
    assert tables.CSS in shell.parts_css() and tables.CSS in shell.shell_css()
    bare = re.sub(r"/\*.*?\*/", "", tables.CSS, flags=re.S)
    sels = [s.strip() for chunk in re.findall(r"([^{}]+)\{[^{}]*\}", re.sub(KEYFRAMES, "", bare))
            for s in chunk.split(",") if s.strip()]
    assert sels and all(s.startswith('[class*="st-key-tbl_"]') for s in sels)       # inget når andra element
    grid = '[class*="st-key-tbl_"] [data-testid="stDataFrameResizable"]'
    # reserven: en fast toning på telefonen
    phone = re.search(r"@media \(max-width: 640px\)\s*\{(.*?\}\s*)\}", bare, flags=re.S).group(1)
    assert _rules(phone)[grid]["mask-image"] == _rules(phone)[grid]["-webkit-mask-image"] == shell.FADE
    # med rullningsstyrd animering: toningen följer glide-tabellens rullningsruta och är borta vid änden
    sup = re.search(r"@supports \(animation-timeline: scroll\(\)\) and \(timeline-scope: --x\)\s*\{"
                    r"\s*@media \(max-width: 640px\) and \(prefers-reduced-motion: no-preference\)\s*\{(.*?)\}\s*\}\s*\}",
                    bare, flags=re.S)
    live = _rules(sup.group(1) + "}")
    assert live[grid]["mask-image"] == "none" and live[grid]["timeline-scope"] == "--wolf-tbl"
    assert live[grid]["animation-timeline"] == "--wolf-tbl" and live[grid]["animation"].startswith("wolf-tbl-edge")
    assert live['[class*="st-key-tbl_"] .dvn-scroller']["scroll-timeline"] == "--wolf-tbl x"
    frames = _rules(re.search(KEYFRAMES, bare).group(0).split("{", 1)[1].rsplit("}", 1)[0])
    assert frames["from"]["mask-image"] == frames["to"]["mask-image"] == shell.FADE
    assert frames["92%"]["mask-size"] == "100% 100%" and frames["to"]["mask-size"] == "calc(100% + 28px) 100%"
    tog = _rules(re.sub(r"@(media|supports)[^{]*\{(?:[^{}]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\})*[^{}]*\}", "",
                        re.sub(KEYFRAMES, "", bare)))
    assert tog['[class*="st-key-tbl_"] [data-testid="stCheckbox"] label']["min-height"] == "44px"
    touch = re.search(r"@media \(hover: none\)\s*\{(.*?\}\s*)\}", bare, flags=re.S).group(1)
    assert _rules(touch)['[class*="st-key-tbl_"]:not(:has([data-testid="stCheckbox"]))']["padding-top"] == "32px"
    assert "!important" not in tables.CSS


@pytest.mark.parametrize("path", ["ui/tables.py"])
def test_no_model_names_or_secrets(path):
    src = _src(*path.split("/"))
    assert "wolf2026" not in src and "claude" not in src.lower() and "gpt" not in src.lower()
