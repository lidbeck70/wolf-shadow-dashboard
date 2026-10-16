"""
Thor fas 1 — strategiregistret och kandidatlistan: varje källa blir Candidate-rader med strategins eget
verdikt, Thor-rank och kartans läge; sortering, dubbletter, filter, sammanfattning och inläsning som
överlever fel. Syntetiska filer — inget nätverk.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from berserk import signals as bsg  # noqa: E402
from thor import candidates as tc  # noqa: E402
from thor import registry as reg  # noqa: E402

GEN = "2026-10-14T16:50:00+00:00"
BOARD = {
    "generated": GEN,
    "board": {"segments": [
        {"key": "technology", "label": "Teknik", "nordic": {"status": "GRÖN", "cykel": "MITTEN", "phase": "STARK"},
         "themes": []},
        {"key": "materials", "label": "Material", "nordic": {"status": "RÖD", "cykel": "TOPP", "phase": "SVAG"},
         "themes": [{"theme": "guld", "label": "Guld",
                     "state": {"status": "GRÖN", "cykel": "TIDIG", "phase": "BAISSE · VÄNDER"}}]}]},
    "swing": {"segments": {
        "technology": {"strategy": "Viking Nine", "picks": [
            {"ticker": "HEXA-B.ST", "name": "Hexagon", "category": "GOLDEN TICKET", "ovtlyr_nine": "8/9",
             "nine_weighted": 86.0, "entry": 110.0, "stop": 104.0, "rr": 2.4, "reason": "Pullback till EMA10"},
            {"ticker": "ERIC-B.ST", "name": "Ericsson", "category": "READY", "ovtlyr_nine": "7/9", "nine_weighted": 71.0}]},
        "materials": {"strategy": "BERSERK", "picks": [{"ticker": "BOL.ST", "setup": bsg.S2, "strategy": "BERSERK"}]}}},
    "long": {"segments": {"technology": {"picks": [
        {"ticker": "LAGR-B.ST", "name": "Lagercrantz", "signal": "STRONG BUY", "score_pct": 0.74, "cycle": 3,
         "fund": 14, "fund_max": 17}]}}},
    "buys": {"segments": {"technology": {"picks": [{"ticker": "MYCR.ST", "verdict": "BUY", "source": "Quality Contrarian"}],
                                         "gate": {"open": True}}},
             "themes": {
                 "guld": {"quality": {"picks": [
                     {"ticker": "NEM", "name": "Newmont", "source": "Rick Rule-arket", "score": 4, "verdict": "Köpkandidat"},
                     {"ticker": "AEM", "name": "Agnico", "source": "Rule-håven", "ev_ebitda": 5.1}]},
                          "leverage": {"strategy": "Durrett", "picks": [
                              {"ticker": "MMY.V", "name": "Monument", "source": "Durrett", "bagger": True,
                               "bull_multiple": 18.0, "poang": 9, "quality": 72},
                              {"ticker": "XX.V", "name": "Xx", "source": "Durrett", "bagger": False,
                               "bull_multiple": 4.0, "poang": 5, "quality": 40}]}},
                 "koppar": {"quality": {"picks": []},
                            "leverage": {"strategy": "Tiggre", "picks": [
                                {"ticker": "CU.V", "name": "Cu", "source": "Tiggre", "buy": True, "score": 8, "un": 4.2,
                                 "gates_passed": 6, "gates_total": 6}]}}},
             "unknown_commodity": [
                 {"ticker": "UNK.V", "name": "Unknown Minerals", "source": "Tiggre", "buy": True, "score": 7, "un": 3.0,
                  "gates_passed": 6, "gates_total": 6}]},
    "wolf": {"segments": {"technology": {"arc": [{"ticker": "ERIC-B.ST", "entry": True, "score": 88.0, "stop": 70.0,
                                                  "tp1": 90.0, "segment": "technology"}], "momentum": []}}},
}
BERSERK = {"generated": GEN, "rows": [
    {"ticker": "BOL.ST", "status": "KÖP", "setup": bsg.S2, "theme": "koppar", "complex": "basmetaller", "region": "Norden",
     "close": 300.0, "stop": 280.0, "why": ["RSI2 under 10", "volym över snitt", "tredje"]},
    {"ticker": "FCX", "status": "BEVAKA", "setup": bsg.S1, "theme": "koppar", "region": "USA", "close": 40.0, "stop": 37.0},
    {"ticker": "ERR.ST", "status": "KÖP", "error": "ingen data"}]}
ARC = {"wolf": {"generated": GEN, "rows": [{"ticker": "ERIC-B.ST", "name": "Ericsson", "score": 88.0},
                                           {"ticker": "VOLV-B.ST", "name": "Volvo", "score": 66.0},
                                           {"ticker": "SAND.ST", "name": "Sandvik", "score": 50.0}]},
       "viking": {"generated": GEN, "rows": [{"ticker": "ATCO-A.ST", "name": "Atlas", "v9": 8, "signal": "BUY", "composite": 77.5},
                                             {"ticker": "SKF-B.ST", "v9": 6, "signal": "WATCH", "composite": 60.0}]}}
MOMENTUM = {"generated": "2026-10-14T04:20", "top": [
    {"ticker": "ERIC B", "name": "Ericsson", "rank": 1, "setupA": True, "price": 80.0},
    {"ticker": "ANOT", "name": "Anoto", "rank": 21, "setupA": True, "price": 1.0},
    {"ticker": "ABB", "name": "ABB", "rank": 5, "setupA": False, "price": 500.0}]}
EMBER = {"timestamp": "2026-10-14T10:00:00",
         "eligible": [{"ticker": "CCJ", "verdict": "KÖPLÄGE", "setup_score": 78.0, "asymmetry_score": 60.0, "entry": 50.0,
                       "stop": 45.0, "t1": 60.0, "rr": 2.0, "theme_key": "uran", "cykel_label": "TIDIG"}],
         "near_misses": [{"ticker": "NHY.OL", "verdict": "AVVAKTA", "setup_score": 42.0, "theme_key": "aluminium"}]}
CA_Q = {"timestamp": "2026-10-14T06:00", "mode": "quality",
        "results": [{"ticker": "MYCR.ST", "name": "Mycronic", "composite_score": 103.0, "rank": 1,
                     "sector": "Information Technology", "branch": "Elektronik", "all_flags": []},
                    {"ticker": "HOLM B.ST", "name": "Holmen B", "composite_score": 70.0, "rank": 2, "sector": "Materials",
                     "branch": "Skog"},
                    {"ticker": "SSAB A.ST", "name": "SSAB A", "composite_score": 65.0, "rank": 3, "sector": "Materials",
                     "branch": "Stål"},
                    {"ticker": "BOL.ST", "name": "Boliden", "composite_score": 60.0, "rank": 4, "sector": "Materials",
                     "branch": "Gruvor"},
                    {"ticker": "SEB A.ST", "name": "SEB A", "composite_score": 55.0, "rank": 30, "sector": "Financials"}],
        "quality_signals": {"MYCR.ST": {"verdict": "BUY", "passed": 4, "total": 4},
                            "HOLM B.ST": {"verdict": "WATCH", "passed": 3, "total": 4},
                            "SSAB A.ST": {"verdict": "BUY", "passed": 4, "total": 4},
                            "BOL.ST": {"verdict": "WAIT", "passed": 2, "total": 4}}}
CA_D = {"timestamp": "2026-10-14T06:00", "mode": "deep_contrarian",
        "results": [{"ticker": "NEM", "name": "Newmont", "composite_score": 66.0, "rank": 1, "sector": "Materials",
                     "all_flags": ["ROIC ok", "nettokassa"]}]}
INSIDER = {"generated": GEN, "clusters": [
    {"ticker": "EVO.ST", "name": "Evolution", "score": 8, "status": "KÖP — logga i journalen", "stop": 85.0,
     "price_now": 105.0, "chase": False, "insiders": 3, "role": "VD/CFO", "cluster_avg": 100.0},
    {"ticker": "SKA-B.ST", "name": "Skanska", "score": 5, "status": "Bevaka — vänta på fler köp", "chase": False},
    {"ticker": "X.ST", "name": "X", "score": 7, "status": "KÖP — logga i journalen", "chase": True}]}
SCREENS = {"generated": GEN, "screens": {
    "rule": {"label": "Överlevarna (Rule)", "sheet": "Rick Rule",
             "rows": [{"ticker": "AEM", "name": "Agnico", "universe": "CA", "mcap_musd": 30000}], "new": ["AEM"]},
    "tiggre": {"label": "Tiggre (sweet spot)", "sheet": "Tiggre",
               "rows": [{"ticker": "CU.V", "name": "Cu", "universe": "CA"}], "new": []}}}
BLOBS = {"board": BOARD, "berserk": BERSERK, "arc": ARC, "momentum": MOMENTUM, "ember": EMBER, "ca_quality": CA_Q,
         "ca_deep": CA_D, "insider": INSIDER, "screens": SCREENS}
CTX = tc.segment_context(BOARD)


def _by(rows):
    return {(c.strategy, c.ticker): c for c in rows}


# ── Registret ────────────────────────────────────────────────────────────────
def test_registry_is_complete_and_consistent():
    assert set(reg.FAMILY_LABEL) == {reg.SWING, reg.LANG, reg.RAVARA, reg.INSIDER, reg.HAV} == set(reg.FAMILY_COLOR)
    assert all(s.family in reg.FAMILY_LABEL for s in reg.STRATEGIES.values())
    assert [s.key for s in reg.by_family(reg.SWING)] == ["viking_nine", "wolf", "viking", "momentum"]
    assert reg.label("quality") == "Quality Contrarian" and reg.label("okänd") == "okänd" and reg.get("x") is None
    import strategy_rules
    for s in reg.STRATEGIES.values():
        assert not s.playbook or s.playbook in strategy_rules.PLAYBOOKS, s.key
    from valhall import paper
    for s in reg.STRATEGIES.values():
        assert not s.paper or s.paper in paper.STRATEGIES or s.paper == "wolf_arc", s.key


# ── Källorna ─────────────────────────────────────────────────────────────────
def test_segment_context():
    assert CTX["segments"]["technology"] == {"status": "GRÖN", "cykel": "MITTEN", "phase": "STARK"}
    assert CTX["themes"]["guld"]["status"] == "GRÖN" and CTX["themes"]["guld"]["segment"] == "materials"
    assert tc.segment_context(None) == {"segments": {}, "themes": {}}


def test_viking_nine_rows():
    rows = _by(tc.from_viking_nine(BOARD, CTX))
    assert set(rows) == {("viking_nine", "HEXA-B.ST"), ("viking_nine", "ERIC-B.ST")}       # BERSERK-segmentet hoppas över
    h = rows[("viking_nine", "HEXA-B.ST")]
    assert h.status == "GOLDEN TICKET" and h.status_class == tc.KOP and h.score == 86.0 and h.score_text == "8/9 · 86"
    assert h.segment_status == "GRÖN" and h.cykel == "MITTEN" and h.entry == 110.0 and h.stop == 104.0 and h.rr == 2.4
    assert h.nordic and h.asof == "2026-10-14" and h.why == "Pullback till EMA10" and h.family == reg.SWING
    assert rows[("viking_nine", "ERIC-B.ST")].status_class == tc.BEVAKA


def test_long_rows():
    (c,) = tc.from_long(BOARD, CTX)
    assert c.strategy == "long" and c.status == "STRONG BUY" and c.status_class == tc.KOP and c.score == 74.0
    assert c.score_text == "74 % · cykel 3/3 · fundamenta 14/17" and c.segment == "technology" and c.segment_status == "GRÖN"


def test_theme_rows_follow_the_paper_accounts_buy_rule():
    rows = _by(tc.from_themes(BOARD, CTX))
    nem = rows[("rule", "NEM")]
    assert nem.status == "Köpkandidat" and nem.status_class == tc.KOP and nem.score == 80.0 and nem.score_text == "4/5 · Köpkandidat"
    assert nem.theme == "guld" and nem.segment == "materials" and nem.segment_status == "GRÖN" and nem.cykel == "TIDIG"   # temats läge
    aem = rows[("rule", "AEM")]
    assert aem.status_class == tc.KOP and aem.score is None and "EV/EBITDA 5.1" in aem.score_text and not aem.nordic
    mmy = rows[("durrett", "MMY.V")]
    assert mmy.status == "15-BAGGER" and mmy.status_class == tc.KOP and mmy.score == 72.0 and "poäng 9/10" in mmy.score_text
    xx = rows[("durrett", "XX.V")]
    assert xx.status == "bull 4×" and xx.status_class == tc.BEVAKA                       # varken 15× eller nordiskt
    cu = rows[("tiggre", "CU.V")]
    assert cu.status == "grindar 6/6" and cu.status_class == tc.KOP and cu.score == 80.0 and cu.score_text == "8/10 · U/N 4.2"
    assert cu.theme == "koppar" and cu.segment == "materials" and cu.segment_status == "RÖD"  # temat saknas på kartan → segmentet
    unk = rows[("tiggre", "UNK.V")]                                                          # okänd råvara: aldrig KÖP
    assert unk.status == "grindar 6/6 · okänd råvara" and unk.status_class == tc.BEVAKA and unk.score == 70.0
    assert unk.score_text == "7/10 · U/N 3" and unk.theme is None and unk.segment is None and "råvaran saknas" in unk.why


def test_berserk_rows():
    rows = _by(tc.from_berserk(BERSERK, CTX))
    assert set(rows) == {("berserk", "BOL.ST"), ("berserk", "FCX")}                         # felrader hoppas över
    b = rows[("berserk", "BOL.ST")]
    assert b.status == f"KÖP · {bsg.S2}" and b.status_class == tc.KOP and b.score == 100.0 and b.nordic
    assert b.theme == "koppar" and b.segment == "materials" and b.segment_status == "RÖD" and b.entry == 300.0 and b.stop == 280.0
    assert b.why == "RSI2 under 10; volym över snitt" and b.family == reg.RAVARA
    f = rows[("berserk", "FCX")]
    assert f.status_class == tc.BEVAKA and f.score == 60.0 and not f.nordic


def test_arc_rows_wolf_and_viking():
    rows = _by(tc.from_arc(ARC, BOARD, CTX))
    e = rows[("wolf", "ERIC-B.ST")]
    assert e.status == "ENTRY · BULL" and e.status_class == tc.KOP and e.score == 70.4 and e.score_text == "88/125"
    assert e.stop == 70.0 and e.target == 90.0 and e.segment == "technology" and e.segment_status == "GRÖN"
    assert rows[("wolf", "VOLV-B.ST")].status == "MODERATE" and rows[("wolf", "VOLV-B.ST")].status_class == tc.BEVAKA
    s = rows[("wolf", "SAND.ST")]
    assert s.status == "NEUTRAL" and s.status_class == tc.HALL and s.score == 40.0 and s.stop is None
    a = rows[("viking", "ATCO-A.ST")]
    assert a.status == "BUY · V9 8/9" and a.status_class == tc.KOP and a.score == 88.9 and a.score_text == "V9 8/9 · composite 78"
    assert rows[("viking", "SKF-B.ST")].status_class == tc.BEVAKA
    assert tc.from_arc(None, None, CTX) == []


def test_momentum_rows_map_borsdata_form_and_rank():
    rows = _by(tc.from_momentum(MOMENTUM, CTX))
    e = rows[("momentum", "ERIC-B.ST")]
    assert e.status == "rad 1 · setup A" and e.status_class == tc.KOP and e.score == 100.0 and e.entry == 80.0 and e.stop == 72.0
    assert rows[("momentum", "ANOT.ST")].status_class == tc.HALL and rows[("momentum", "ANOT.ST")].score == 50.0
    assert rows[("momentum", "ABB.ST")].status_class == tc.BEVAKA and rows[("momentum", "ABB.ST")].score == 90.0
    assert tc.momentum_yf("ERIC B", {"ERIC B": "ERIC-B.ST"}) == "ERIC-B.ST" and tc.momentum_yf("NHY", {"NHY": "NHY.OL"}) == "NHY.OL"


def test_ember_rows():
    rows = _by(tc.from_ember(EMBER, CTX))
    c = rows[("ember", "CCJ")]
    assert c.status == "KÖPLÄGE" and c.status_class == tc.KOP and c.score == 78.0 and c.score_text == "setup 78 · asymmetri 60"
    assert c.theme == "uran" and c.segment == "energy" and c.entry == 50.0 and c.stop == 45.0 and c.target == 60.0 and c.rr == 2.0
    assert c.why == "cykel TIDIG" and not c.nordic and c.asof == "2026-10-14"
    n = rows[("ember", "NHY.OL")]
    assert n.status_class == tc.HALL and n.segment == "materials" and n.nordic


def test_contrarian_rows_with_verdict_and_gate():
    rows = _by(tc.from_contrarian(CA_Q, CTX, "quality"))
    m = rows[("quality", "MYCR.ST")]
    assert m.status == "BUY 4/4" and m.status_class == tc.KOP and m.score == 100.0 and m.segment == "technology"   # tak 100
    h = rows[("quality", "HOLM-B.ST")]
    assert h.status == "WATCH 3/4" and h.status_class == tc.BEVAKA and h.segment == "materials" and h.segment_status == "RÖD"
    s = rows[("quality", "SSAB-A.ST")]
    assert s.status_class == tc.SPARRAD and s.status == "BUY 4/4 · sektorn RÖD"                      # grinden ej RÖD
    assert rows[("quality", "BOL.ST")].status_class == tc.HALL and rows[("quality", "BOL.ST")].theme == "koppar"
    assert rows[("quality", "SEB-A.ST")].status == "rank 30" and rows[("quality", "SEB-A.ST")].status_class == tc.BEVAKA
    (d,) = tc.from_contrarian(CA_D, CTX, "deep")
    assert d.strategy == "deep" and d.status == "rank 1" and d.status_class == tc.BEVAKA and d.why == "ROIC ok; nettokassa"
    assert d.family == reg.LANG and not d.nordic


def test_insider_rows():
    rows = _by(tc.from_insider(INSIDER, CTX))
    e = rows[("insider", "EVO.ST")]
    assert e.status == "KÖP" and e.status_class == tc.KOP and e.score == 80.0 and e.stop == 85.0 and e.entry == 105.0
    assert e.score_text == "8/10 · 3 insiders · VD/CFO" and e.why == "klustersnitt 100.0"
    assert rows[("insider", "SKA-B.ST")].status_class == tc.BEVAKA
    x = rows[("insider", "X.ST")]
    assert x.status_class == tc.BEVAKA and "passa" in x.status                                    # jagar kursen


def test_screen_rows():
    rows = _by(tc.from_screens(SCREENS, CTX))
    a = rows[("hav", "AEM")]
    assert a.status == "Överlevarna (Rule)" and a.status_class == tc.TRAFF and a.new and a.score is None
    assert a.score_text == "CA · 30,000 MUSD" and a.why == "Rick Rule" and a.family == reg.HAV
    assert not rows[("hav", "CU.V")].new
    # håvar med egen strategi i registret ger träffen även där — samma rad, annan strategi (Sprott har ingen)
    r = rows[("rule", "AEM")]
    assert r.status == a.status and r.status_class == tc.TRAFF and r.new and r.score is None and r.family == reg.RAVARA
    assert rows[("tiggre", "CU.V")].status == "Tiggre (sweet spot)" and not rows[("tiggre", "CU.V")].new
    # screens_scan skriver "new" som tickersträngar (riktiga filen); rader som dict tolereras också, skräp hoppas över
    alt = {"generated": GEN, "screens": {"rule": {"rows": [{"ticker": "AEM"}, "skräp", None], "new": [{"ticker": "aem"}]},
                                         "sprott": {"rows": [{"ticker": "SPR.V"}]},
                                         "trasig": "inte en dict", "tom": None}}
    rows = tc.from_screens(alt, CTX)
    assert [(c.strategy, c.ticker, c.new) for c in rows] == [("hav", "AEM", True), ("rule", "AEM", True),
                                                             ("hav", "SPR.V", False)]


def test_screen_hits_under_their_own_strategy_yield_to_the_sheets():
    alt = {"generated": GEN, "screens": {"durrett": {
        "label": "Durrett", "sheet": "Durrett", "new": ["NEW.V"],
        "rows": [{"ticker": "MMY.V", "name": "Monument"}, {"ticker": "NEW.V", "name": "New Gold"}]}}}
    rows = _by(tc.build({**BLOBS, "screens": alt}, today="2026-10-15"))
    mmy = rows[("durrett", "MMY.V")]
    assert mmy.status == "15-BAGGER" and mmy.status_class == tc.KOP                   # arkets KÖP vinner över träffen
    assert rows[("durrett", "NEW.V")].status_class == tc.TRAFF and rows[("durrett", "NEW.V")].new
    assert rows[("hav", "NEW.V")].new and rows[("hav", "MMY.V")].status_class == tc.TRAFF   # Håvarna som förut


def test_build_survives_a_broken_source(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("fel form")
    monkeypatch.setattr(tc, "from_ember", boom)
    errors = {}
    rows = tc.build(BLOBS, today="2026-10-15", errors=errors)
    assert errors == {"ember": "RuntimeError: fel form"}
    assert rows and not any(c.strategy == "ember" for c in rows) and any(c.strategy == "berserk" for c in rows)
    assert len(tc.build(BLOBS, today="2026-10-15")) == len(rows)                                   # utan errors: tyst
    bad = {**BLOBS, "insider": {"generated": GEN, "clusters": "x"}, "board": {"board": {"segments": "x"}}}
    errors = {}
    tc.build(bad, errors=errors)
    assert "insider" in errors and "AttributeError" in errors["insider"]


# ── Listan ───────────────────────────────────────────────────────────────────
def test_build_sorts_dedupes_marks_stale_and_filters():
    rows = tc.build(BLOBS, today="2026-10-15")
    keys = [(c.strategy, c.ticker) for c in rows]
    assert len(keys) == len(set(keys)) and len(rows) == 32
    assert rows[0].status_class == tc.KOP and rows[0].score == 100.0
    classes = [tc.CLASS_ORDER[c.status_class] for c in rows]
    assert classes == sorted(classes)                                                             # KÖP → BEVAKA → TRÄFF → HÅLL → SPÄRRAD
    kop = [c for c in rows if c.status_class == tc.KOP]
    assert [c.score if c.score is not None else -1 for c in kop] == sorted(
        (c.score if c.score is not None else -1 for c in kop), reverse=True)                       # rang inom klassen
    assert not any(c.stale for c in rows) and all(c.stale for c in tc.build(BLOBS, today="2026-10-30"))
    sw = tc.filter_rows(rows, family=reg.SWING)
    assert {c.strategy for c in sw} == {"viking_nine", "wolf", "viking", "momentum"}
    assert all(c.status_class == tc.KOP for c in tc.filter_rows(rows, only_buy=True))
    assert {c.ticker for c in tc.filter_rows(rows, nordic_only=True)}.isdisjoint({"NEM", "AEM", "FCX", "CCJ", "MMY.V", "CU.V"})
    assert {c.strategy for c in tc.filter_rows(rows, segment="technology")} == {"viking_nine", "long", "wolf", "quality"}
    assert {(c.strategy, c.ticker) for c in tc.filter_rows(rows, search="eric")} == \
        {("viking_nine", "ERIC-B.ST"), ("wolf", "ERIC-B.ST"), ("momentum", "ERIC-B.ST")}
    assert [c.ticker for c in tc.filter_rows(rows, strategy="hav")] == ["AEM", "CU.V"]           # NY först
    d = rows[0].as_dict()
    assert d["family"] == reg.SWING and d["label"] == "Momentum Swing" and d["ticker"] == "ERIC-B.ST"


def test_summary():
    s = tc.summary(tc.build(BLOBS, today="2026-10-15"))
    assert s["n"] == 32 and s["buy"] == 13
    assert s["by_family"][reg.HAV] == 2 and s["by_strategy"]["quality"] == {"n": 5, "buy": 1, "asof": "2026-10-14", "stale": False}


def test_empty_sources_give_empty_list():
    assert tc.build({}) == [] and tc.build({"board": {}, "berserk": None}) == []
    assert tc.summary([]) == {"n": 0, "buy": 0, "by_family": {}, "by_strategy": {}}


def test_load_blobs_survives_errors():
    seen = []

    def loader(name, fb):
        seen.append(name)
        if name == "berserk_scan.json":
            raise RuntimeError("nät")
        return {"name": name}

    def ember_loader():
        raise RuntimeError("ingen gist")

    out = tc.load_blobs(loader, ca_loader=lambda mode: {"mode": mode}, ember_loader=ember_loader)
    assert set(seen) == set(tc.BLOBS.values()) and out["berserk"] is None and out["board"] == {"name": "valhall_board.json"}
    assert out["ca_quality"] == {"mode": "quality"} and out["ca_deep"] == {"mode": "deep_contrarian"} and out["ember"] is None


@pytest.mark.parametrize("raw,expected", [("ekta b.st", "EKTA-B.ST"), (" NEM ", "NEM"), (None, ""), ("HOLM B.ST", "HOLM-B.ST")])
def test_yf_form(raw, expected):
    assert tc.yf(raw) == expected
