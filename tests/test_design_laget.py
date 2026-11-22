"""Designförslaget PR 5: ODIN → Läget — fyra ljus ur sparade källor, ingen egen slutsats, inget live."""
import pathlib

from odin import laget as lg
from ui import nav

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOARD = {"segments": [{"key": "a", "nordic": {"status": "GRÖN"}}, {"key": "b", "nordic": {"status": "GRÖN"}},
                      {"key": "c", "nordic": {"status": "RÖD"}}, {"key": "d", "nordic": None}]}
RISK = {"generated": "2026-10-24 06:00 UTC",
        "levels": {"SPY": {"label": "SPY", "level": "LÅG"}, "OMXS30": {"label": "OMXS30", "level": "FÖRHÖJD"}}}


def test_allocation_uses_the_breaker():
    assert lg.allocation_light({"peak": 100, "current": 95})[0] == "NORMAL"
    assert lg.allocation_light({"peak": 100, "current": 85})[0] == "SKÄRPT"
    assert lg.allocation_light({"peak": 100, "current": 80})[0] == "HALVERAD RISK"
    assert lg.allocation_light(None)[0] == lg.SAKNAS


def test_swing_light():
    assert lg.swing_light({"regime": "gul", "generated": "2026-10-24T06:00"}) == ("GUL", "2026-10-24")
    assert lg.swing_light({})[0] == lg.SAKNAS


def test_risk_light_per_market_and_worst():
    assert lg.risk_light(RISK)[0] == "SPY LÅG · OMXS30 FÖRHÖJD"
    assert lg.risk_worst(RISK) == "FÖRHÖJD"
    assert lg.risk_light(None)[0] == lg.SAKNAS


def test_nordic_counts_green_segments():
    assert lg.nordic_light(BOARD) == ("2/3 GRÖN", "GRÖN")
    assert lg.nordic_light({})[0] == lg.SAKNAS


def test_row_has_four_lights_in_order_and_no_verdict():
    items = lg.capital_row({"peak": 100, "current": 99}, {"regime": "GRÖN"}, RISK, BOARD)
    assert [t for t, *_ in items] == ["Allokering", "Swing", "Marknadsrisk", "Norden"]
    text = lg.row_text(items)
    assert text == "Allokering NORMAL · Swing GRÖN · Marknadsrisk SPY LÅG · OMXS30 FÖRHÖJD · Norden 2/3 GRÖN"
    assert "köp" not in text.lower()


def test_laget_is_first_under_marknad_and_swing_alert_lands_there():
    assert nav.options("odin")[0] == "Läget"
    assert nav.target_slug("swing_regime") == "odin/laget" and nav.resolve("odin/marknad/swing-regime") == ["odin", "Läget"]


def test_alert_scan_saves_the_risk_level():
    src = (ROOT / "alert_scan.py").read_text(encoding="utf-8")
    assert 'save_blob("market_risk.json"' in src


def test_laget_does_not_compute_market_risk_on_load():
    src = (ROOT / "odin" / "laget.py").read_text(encoding="utf-8")
    load_src = src.split("def load()")[1].split("def render_laget_page")[0]
    assert "evaluate" not in load_src and "market_risk_gate" not in load_src


# ── LIDSKJALV PR 19: Läget — ljusen först, EN 📡, regimkortet och bevisen (§4.4, beslut 11) ──────────────────
import hashlib  # noqa: E402
import html as _html  # noqa: E402
import re  # noqa: E402

import wolf_regime_ui as wr  # noqa: E402

CAPTION = "> 55 % friskt · 45–55 % smalnar · < 45 % bredden dör — den dör före index."
REGIME = {"regime": "GUL", "generated": "2026-11-20T18:05",
          "rules": ["• Nya köp tillåtna men HALV positionsstorlek.", "Max 1 nytt köp/vecka.",
                    "Bredden dör före index: var beredd på RÖD."],
          "index": {"above": True, "dist": 0.031, "close": 1012, "ma200": 981, "spark": [990 + i for i in range(30)]},
          "breadth": 0.48, "qualifying": 23, "universe": 310,
          "history": [{"date": f"2026-11-{d:02d}", "regime": "GUL", "breadth": 0.48, "qualifying": 23}
                      for d in range(2, 21)]}


def _plain(value: str) -> str:
    """Texten som webbläsaren visar ur ett HTML-block: taggarna bort, entiteterna avkodade."""
    return _html.unescape(re.sub(r"<[^>]+>", "", value)).replace(" ", " ")


def _regime_page(embedded):
    import wolf_regime_ui
    if embedded:
        wolf_regime_ui.render_wolf_regime_page(embedded=True)
    else:
        wolf_regime_ui.render_wolf_regime_page()                    # motorsidans anrop, som förut


def _run(script, monkeypatch, data=REGIME, args=(), state=None):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(wr, "_get_data", lambda force=False: data)
    at = AppTest.from_function(script, args=args, default_timeout=60)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    return at.run()


def _tree(at) -> list:
    """Sidans element i ordning: typ och text (markdown/bildtext), nyckel och etikett (knapp/fält), djup."""
    out = []

    def walk(node, depth=0):
        children = getattr(node, "children", None)
        for ch in (children.values() if isinstance(children, dict) else []):
            t = type(ch).__name__
            if t in ("Markdown", "Caption"):
                out.append(f"{depth}|{t}|{ch.value}")
            elif t in ("Button", "TextInput"):
                out.append(f"{depth}|{t}|{ch.key}|{ch.label}")
            else:
                out.append(f"{depth}|{t}|{getattr(ch, 'type', '')}")
            walk(ch, depth + 1)
    walk(at._tree)
    return out


def test_motor_page_is_unchanged_when_not_embedded(monkeypatch):
    """embedded=False (förval): samma element i samma ordning och med samma text som före PR 19 — rubriken 'Regim',
    sidans egen 📡 (wolf_reg_refresh), kortet med listans punkter, de tre kolumnerna, historiken och tickerkollen.
    Fingeravtrycket räknades på sidan före ändringen, med regeltexterna som wolf_data skriver dem (utan '• ')."""
    data = dict(REGIME, rules=wr.rule_lines(REGIME["rules"]))
    at = _run(_regime_page, monkeypatch, data=data, args=(False,))
    assert not at.exception, at.exception
    tree = _tree(at)
    assert hashlib.sha256("\n".join(tree).encode()).hexdigest() == \
        "3887dd5a1d4d98014cb7e752c1bf268ddb589969c1eb88d27aa4fcda7fbabc08", "\n".join(tree)
    assert "3|Markdown|<h1 style='color:#e8e4dc;margin:0;'>Regim</h1>" in tree
    assert [b.key for b in at.button] == ["wolf_reg_refresh"] and at.caption[0].value == "Uppdaterad 2026-11-20T18:05"
    assert CAPTION in [c.value for c in at.caption]                     # som förut: st.caption
    import inspect
    assert inspect.signature(wr.render_wolf_regime_page).parameters["embedded"].default is False


def test_breadth_caption_renders_a_literal_gt_not_a_blockquote(monkeypatch):
    """Som markdown blev '> 55 % friskt' ett citatblock (tecknet försvann). Inbäddat står bildtexten escapad i ett
    HTML-block: webbläsaren visar exakt '> 55 % friskt · 45–55 % smalnar · < 45 % bredden dör — den dör före index.'"""
    at = _run(_regime_page, monkeypatch, args=(True,))
    assert not at.exception, at.exception
    assert CAPTION not in [c.value for c in at.caption]                # aldrig som markdown-bildtext
    block = [m.value for m in at.markdown if "smalnar" in m.value]
    assert len(block) == 1 and block[0].startswith("<div")             # ett HTML-block: ingen markdown-tolkning
    assert "&gt; 55 % friskt" in block[0] and "&lt; 45 % bredden dör" in block[0]
    assert CAPTION in _plain(block[0])
    assert not [m for m in at.markdown if m.value.lstrip().startswith(">")]
    assert wr.evidence_html(REGIME).count("&gt; 55 %") == 1


def test_embedded_has_no_own_heading_or_refresh_and_shows_card_evidence_and_history(monkeypatch):
    at = _run(_regime_page, monkeypatch, args=(True,))
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "<h1" not in text and not [b for b in at.button]           # ingen 'Regim'-rubrik, ingen andra 📡
    assert "Swing-läget" in text and "uppdaterad 20 nov 18:05" in text
    assert "● GUL" in text and "selektiv, halv storlek" in text and 'class="wolf-regime-card"' in text
    assert '<svg class="wolf-spark"' in text and "3,1 %" in text and "över" in text
    assert "48,0 %" in text and "23" in text and "av 310" in text
    assert [e.label for e in at.expander] == ["Historik"]
    assert not [t for t in at.text_input if t.key == "swreg_ticker"]   # tickerkollen ritar Läget, sist


def test_rule_bullets_are_not_duplicated_and_texts_are_unchanged():
    rules = list(REGIME["rules"])
    assert wr.rule_lines(rules) == ["Nya köp tillåtna men HALV positionsstorlek.", "Max 1 nytt köp/vecka.",
                                    "Bredden dör före index: var beredd på RÖD."]
    assert REGIME["rules"] == rules                                     # datan rörs inte
    card = wr.regime_card_html(REGIME)
    assert card.count("•") == 3 and "• •" not in _plain(card) and "<li>•" not in card
    assert "list-style:none" in card and "font-size:20px;font-weight:800" in card
    assert "● GUL</span> — selektiv, halv storlek" in card
    assert wr.rule_lines(None) == [] and wr.rule_lines(["", " • "]) == []
    unknown = wr.regime_card_html({"regime": "BLÅ", "rules": []})
    assert "OKÄND" in unknown and "<ul" not in unknown


def test_evidence_without_index_or_breadth_shows_saknas_not_zero():
    ev = wr.evidence_html({"regime": "GUL", "index": None, "qualifying": 5, "universe": 9})
    assert ev.count(">Saknas<") == 2 and "0,0 %" not in ev and "wolf-spark" not in ev
    assert wr.sv_stamp("2026-11-20T18:05") == "20 nov 18:05" and wr.sv_stamp("2026-11-20") == "20 nov"
    assert wr.sv_stamp(None) == "?" and wr.sv_stamp("okänt") == "okänt"


def test_embedded_without_data_is_grey_and_honest(monkeypatch):
    at = _run(_regime_page, monkeypatch, data=None, args=(True,))
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert ">Saknas<" in text and "Ingen regimdata hittad ännu." in text and "python wolf_data.py" in text
    assert not at.caption and not at.expander                           # ingen kodruta på 10,5 px


def _laget_page():
    from odin import laget
    laget.render_laget_page()


def test_one_refresh_on_laget_reloads_the_lights_and_the_swing_regime(monkeypatch):
    """Beslut 11: EN '📡 Uppdatera' (laget_refresh) läser om både ljusens källor och wolf_regime.json."""
    from streamlit.testing.v1 import AppTest
    calls = {"laget": 0, "wolf": 0}

    def load():
        calls["laget"] += 1
        return {"alloc": {"peak": 100, "current": 94}, "regime": REGIME, "risk": RISK, "board": BOARD}

    def wolf(name):
        calls["wolf"] += 1
        return REGIME
    monkeypatch.setattr(lg, "load", load)
    monkeypatch.setattr(wr, "_load_wolf", wolf)
    at = AppTest.from_function(_laget_page, default_timeout=60).run()
    assert not at.exception, at.exception
    assert calls == {"laget": 1, "wolf": 1}
    refresh = [b for b in at.button if b.label == "📡 Uppdatera"]
    assert [b.key for b in refresh] == ["laget_refresh"]               # ETT 📡 på sidan
    assert "wolf_reg_refresh" not in [b.key for b in at.button]
    at.run()                                                            # vanlig omritning: inget läses om
    assert calls == {"laget": 1, "wolf": 1}
    at.button(key="laget_refresh").click().run()
    assert not at.exception, at.exception
    assert calls == {"laget": 2, "wolf": 2}                            # båda läses om av samma tryck


def test_laget_page_order_lights_note_section_risk_ticker(monkeypatch):
    import market_risk_ui
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(lg, "load", lambda: {"alloc": {"peak": 100, "current": 94}, "regime": REGIME, "risk": RISK,
                                             "board": BOARD})
    monkeypatch.setattr(wr, "_load_wolf", lambda name: REGIME)
    at = AppTest.from_function(_laget_page, default_timeout=60).run()
    assert not at.exception, at.exception
    md = [m.value for m in at.markdown]
    first = lambda s: next(i for i, v in enumerate(md) if s in v)      # noqa: E731
    assert first("👁 LÄGET") < first('class="wolf-grid"') < first(lg.NOTE_TEXT) < first("Swing-läget") \
        < first("wolf-regime-card") < first("Kolla en ticker")
    grid = md[first('class="wolf-grid"')]
    assert [t for t in ("ALLOKERING", "Allokering") if t in grid] and "Marknadsrisk" in grid and "Norden" in grid
    assert lg.SUBTITLE in " ".join(md)
    assert at.toggle(key="laget_risk").label == lg.RISK_LABEL and not at.toggle(key="laget_risk").value
    assert not at.get("status")                                         # inget räknas live förrän du slår på den
    assert at.text_input(key="swreg_ticker").label == "Ticker"
    drawn = []
    monkeypatch.setattr(market_risk_ui, "render_market_risk_page", lambda: drawn.append(1))
    at.toggle(key="laget_risk").set_value(True).run()
    assert not at.exception, at.exception
    assert drawn == [1] and len(at.get("status")) == 1                 # i st.status när den är på


def test_risk_deep_link_still_turns_the_toggle_on():
    panel = (ROOT / "wolf_panel.py").read_text(encoding="utf-8")
    assert 'st.session_state["laget_risk"] = True' in panel
    src = (ROOT / "odin" / "laget.py").read_text(encoding="utf-8")
    assert 'key="laget_risk"' in src and 'key="laget_refresh"' in src and src.count("📡 Uppdatera") == 1
