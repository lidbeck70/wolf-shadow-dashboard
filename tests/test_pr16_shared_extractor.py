"""
PR 16 — ett dokument, ett utdrag, förslag i alla ark.

Extraktorn frågade förut efter ett arks fält i taget, så samma PDF fick
laddas upp i Rick Rule, Poängmodellen, Tiggre och Durrett-arket var för
sig. Nu ställs alla arkens fält i ett anrop, svaret sparas per ticker i
sessionen (extract_store) och varje ark läser sina förslag därifrån. Alias
mellan arken (AISC = unit_cost, NAV = NPV, burn/år = burn/kvartal × 4 …)
gör att ett äldre utdrag från ett ark ändå ger förslag i de andra.
"""
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ai import extract_prompt as xp   # noqa: E402
import extract_store                  # noqa: E402


def _src(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


# ── Unionsprompten ───────────────────────────────────────────────────────────
def test_union_prompt_lists_every_key_once_and_keeps_the_catalysts():
    fields = xp.union_fields(xp.ALL_SHEETS)
    keys = [f.key for f in fields]
    assert len(keys) == len(set(keys))                          # varje nyckel en gång
    assert keys.count("aisc") == 1 and "unit_cost" in keys and "npv_musd" in keys
    every = {f.key for sh in xp.ALL_SHEETS for f in xp.FIELDS[sh]}
    assert set(keys) == every                                   # inget ark tappas
    p = xp.build_extract_prompt(xp.ALL_SHEETS, "KDK", "Kodiak", "[Sida 1]\nAISC 1,250 USD/oz")
    assert "Rick Rule + Royalty C + Sprott + Durrett + Tiggre + Confidence score" in p
    assert p.count("- aisc — ") == 1 and "catalysts" in p
    # ett ark i taget fungerar som förut
    single = xp.build_extract_prompt("rule", "KDK", "Kodiak", "x")
    assert "Ark: Rick Rule" in single and "npv_musd" not in single
    with pytest.raises(ValueError):
        xp.build_extract_prompt([], "X", "", "")


# ── Aliasen mellan arken ─────────────────────────────────────────────────────
def _parsed(**fields):
    return {"fields": {k: {"value": v, "unit": "", "page": 3, "quote": "q", "confidence": "high"}
                       for k, v in fields.items()}}


def test_an_extraction_from_one_sheet_proposes_in_the_others():
    # Durrett-arkets nycklar → Rick Rule, Sprott och Tiggre
    parsed = _parsed(aisc=1250, mine_life_years=12, cash_musd=80, quarterly_burn_musd=5,
                     npv_musd=600, permits_granted=True, insider_ownership_pct=14)
    rule = {p["key"]: p["value"] for p in xp.proposals("rule", parsed)}
    assert rule["unit_cost"] == 1250.0 and rule["mine_life"] == 12.0 and rule["insider_ownership"] == 14.0
    sprott = {p["key"]: p["value"] for p in xp.proposals("sprott", parsed)}
    assert sprott["cash"] == 80.0 and sprott["burn"] == 20.0     # per kvartal × 4
    tiggre = {p["key"]: p["value"] for p in xp.proposals("tiggre", parsed)}
    assert tiggre["nav"] == 600.0 and tiggre["permits"] is True
    # …och åt andra hållet: Sprotts burn/år → Durrett-arkets burn per kvartal
    back = {p["key"]: p["value"] for p in xp.proposals("confidence", _parsed(burn=40, unit_cost=900, nav=300))}
    assert back["quarterly_burn_musd"] == 10.0 and back["aisc"] == 900.0 and back["npv_musd"] == 300.0
    # arkets egen nyckel vinner över aliaset
    both = {p["key"]: p["value"] for p in xp.proposals("rule", _parsed(unit_cost=1000, aisc=1250))}
    assert both["unit_cost"] == 1000.0
    for key, alts in xp.ALIASES.items():                        # aliasen pekar på riktiga nycklar
        every = {f.key for sh in xp.ALL_SHEETS for f in xp.FIELDS[sh]}
        assert key in every and all(a in every for a, _f in alts), key


# ── Lagret ───────────────────────────────────────────────────────────────────
def test_store_keeps_one_extraction_per_ticker(monkeypatch):
    monkeypatch.setattr(streamlit, "session_state", {})
    assert extract_store.get("kdk") is None
    e = extract_store.save("kdk", _parsed(nav=1), doc="DFS 2025", sheet="Tiggre", model="m", pages=40, chars=9000)
    assert extract_store.get("KDK") is e and e["when"]
    assert extract_store.describe(e, "Rick Rule") == f"Utdrag ur DFS 2025 ({e['when']}, via Tiggre)"
    assert extract_store.describe(e, "Tiggre") == f"Utdrag ur DFS 2025 ({e['when']})"
    extract_store.save("KDK", _parsed(nav=2), doc="PFS")
    assert extract_store.get("kdk")["doc"] == "PFS"              # ersätter
    extract_store.forget("KDK")
    assert extract_store.get("KDK") is None


def test_both_extractor_uis_ask_for_every_sheet_and_read_the_shared_store():
    ui = _src("extract_ui.py")
    assert "xp.build_extract_prompt(xp.ALL_SHEETS" in ui and "extract_store.save(" in ui
    assert "extract_store.get(ticker)" in ui and 'f"{skey}_name"' in ui
    cui = _src("confidence/ui.py")
    assert "xp.build_extract_prompt(xp.ALL_SHEETS" in cui and "extract_store.get(company.ticker)" in cui


# ── Ett ark ritar förslag ur ett utdrag gjort i ett annat ────────────────────
def test_rick_rule_shows_and_applies_a_proposal_extracted_in_tiggre(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from ai import openai_client as oc
    monkeypatch.setattr(oc, "configured", lambda: True)
    monkeypatch.setenv("PR16_TEST_ROOT", ROOT)

    def app():
        import os as _o, sys as _s
        _s.path.insert(0, _o.environ["PR16_TEST_ROOT"])
        import streamlit as st
        import extract_ui
        row = st.session_state.setdefault("row", {"id": "r1", "ticker": "KDK", "name": "Kodiak"})
        extract_ui.render_extractor("rule", row, widget_keys={"unit_cost": "pr_c_r1"})

    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["xt_doc:KDK"] = {
        "parsed": _parsed(aisc=1250, mine_life_years=12), "doc": "PFS 2026", "sheet": "Tiggre",
        "model": "m", "pages": 12, "chars": 5000, "when": "2026-09-23 10:00"}
    at.run()
    assert not at.exception, at.exception
    caps = " ".join(c.value for c in at.caption)
    assert "Utdrag ur PFS 2026" in caps and "via Tiggre" in caps
    use = [b for b in at.button if b.key == "xt_rule_r1_use_unit_cost"]
    assert use, "förslaget på AISC → unit_cost saknas"
    use[0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["row"]["unit_cost"] == 1250.0
