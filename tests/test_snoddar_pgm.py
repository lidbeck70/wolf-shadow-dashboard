"""
Snoddarna för platina och palladium — Gold/Platinum (GC=F/PL=F, reserv GLD/PPLT) och Gold/Palladium
(GC=F/PA=F, reserv GLD/PALL), riktning 'high' (hög kvot = PGM-metallen billig mot guld).

Sedan ägarens beslut A (2026-11-03, efter backtestet docs/forskning/pgm_snoddar_spec.md) är de två kvoterna BARA
KONTEXT (alpha_regime/buy_support.CONTEXT_ONLY): de visas (kort, Thor, temabordet, väljaren) men ger aldrig köpstöd —
inget +1, ingen KÖP-STÖD-räkning, ingen Why Now-poäng. Testerna här som förr låste drivarregeln för PGM säger nu
det (varje ändring förklaras i testet); de breda tabellerna finns i tests/test_pgm_kontext.py. Gruvornas
marginalregel och alla befintliga kvoter är orörda. Varje register som listar silvrets kvoter/exponering listar
också platina och palladium (prövas med en genomsökning av källkoden). Inget nätverk — fejkade kursserier.
"""
import ast
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_regime import buy_support as bs  # noqa: E402
from alpha_regime import commodity_ratios as cr  # noqa: E402
from alpha_regime import miner_margin as mm  # noqa: E402
from alpha_regime import ui as aui  # noqa: E402
from test_kopstod_gruvor import (  # noqa: E402
    DRIVERS, STATUSES, _St, _margin, _old_engine_confirms, _old_summary, _ratio, _regime, _run_engine)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = " "
NEUTRAL_LINE = "Snodden är inte spänd — benens prisläge förklarar inget köpläge."
PGM = (("gold_platinum", "Gold / Platinum", "Platinum", "PL=F", "PPLT"),
       ("gold_palladium", "Gold / Palladium", "Palladium", "PA=F", "PALL"))
PGM_KEYS = ("gold_platinum", "gold_palladium")
N = 2600
IDX = pd.bdate_range(end="2026-10-30", periods=N)

# Ordagrann kopia av _RATIO_SPECS före platina/palladium — ingen befintlig kvot får ändras.
OLD_SPECS = [
    ("gold_silver", "Gold / Silver", "Silver", "GC=F", "SI=F", "GLD", "SLV", "high", 0),
    ("gold_oil", "Gold / Oil", "Oil", "GC=F", "CL=F", "GLD", "USO", "high", 0),
    ("metal_miners", "Gold ETF / Gold Miners", "Gold Miners", "GLD", "GDX", "GLD", "GDX", "high", 0),
    ("silver_miners", "Silver ETF / Silver Miners", "Silver Miners", "SLV", "SIL", "SLV", "SIL", "high", 0),
    ("copper_gold", "Copper / Gold", "Copper", "HG=F", "GC=F", "COPX", "GLD", "low", 0),
    ("gdxj_gdx", "GDXJ / GDX (Juniors/Majors)", "Junior Miners", "GDXJ", "GDX", "GDXJ", "GDX", "high", 0),
    ("gold_gdxj", "GLD / GDXJ (Metal/Juniors)", "Junior Miners", "GLD", "GDXJ", "GLD", "GDXJ", "high", 0),
    ("silver_juniors", "SLV / SILJ (Silver Juniors)", "Silver Juniors", "SLV", "SILJ", "", "", "high", 5),
    ("gdx_spy", "GDX / SPY (Miners/Market)", "Gold Miners", "GDX", "SPY", "GDX", "SPY", "low", 0),
]

# Ordagrann kopia av _KEYWORD_EXPOSURE före platina/palladium.
OLD_KEYWORDS = [
    (["junior miner", "junior mine", "juniormine", "gdxj"], "junior_miner"),
    (["silver miner", "silver mine", "silvr", "silber"], "silver"),
    (["gold miner", "gold mine", "goldminer"], "gold_miner"),
    (["uranium", "uran"], "uranium"),
    (["lithium", "litium"], "lithium"),
    (["natural gas", "naturgas", "lng"], "gas"),
    (["copper", "koppar"], "copper"),
    (["silver"], "silver"),
    (["gold", "guld", "gld"], "gold_miner"),
    (["mining", "gruv", "metals"], "gold_miner"),
    (["oil", "olja", "petroleum", "petro", "crude"], "oil"),
]

# Panelens exponeringsväljare före platina/palladium (hårdkodade listor då).
OLD_PICKER = {"Auto": None, "Gold Miners": ["metal_miners", "gdxj_gdx", "gold_gdxj", "gdx_spy"],
              "Junior Miners": ["gdxj_gdx", "gold_gdxj"], "Silver": ["gold_silver", "silver_juniors"],
              "Oil": ["gold_oil"], "Copper": ["copper_gold"], "None": []}


def _old_detect(name):
    if not name:
        return None
    text = name.lower()
    return next((exp for kws, exp in OLD_KEYWORDS if any(k in text for k in kws)), None)


def _spec(key):
    return next(s for s in cr._RATIO_SPECS if s.key == key)


# ── Fejkade kursserier ───────────────────────────────────────────────────────
def _cheap_pgm():
    """Guldet sidledes runt 2 000 (sista dagen mitt i sitt spann), PGM-metallen från 1 500 ned till 800
    (sista dagen lägst) → kvoten nära toppen: sträckt, och det billiga benet driver (DENOMINATOR_CHEAP)."""
    gold = 2000.0 + 5.0 * np.sin(np.linspace(0.0, 20 * np.pi, N))
    return pd.Series(gold, index=IDX), pd.Series(np.linspace(1500.0, 800.0, N), index=IDX)


def _dear_gold():
    """Guldet till nya toppar, PGM-metallen sidledes → kvoten sträckt men bara för att guldet är dyrt."""
    pgm = 1000.0 + 5.0 * np.sin(np.linspace(0.0, 20 * np.pi, N))
    return pd.Series(np.linspace(1000.0, 3000.0, N), index=IDX), pd.Series(pgm, index=IDX)


def _compute(monkeypatch, key, legs, pair="primary"):
    """_compute_ratio med benen under primär- eller reservsymbolerna (ETF:erna skalade 1/10)."""
    spec = _spec(key)
    gold, pgm = legs
    if pair == "primary":
        prices = {spec.primary_num: gold, spec.primary_den: pgm}
    else:
        prices = {spec.fallback_num: gold / 10.0, spec.fallback_den: pgm / 10.0}
    monkeypatch.setattr(cr, "_download_close", lambda t, period="10y": prices.get(t, pd.Series(dtype=float)))
    return cr._compute_ratio(spec)


# ── Specarna ─────────────────────────────────────────────────────────────────
def test_specs_exist_and_the_old_ones_are_unchanged():
    assert [tuple(vars(s).values()) for s in cr._RATIO_SPECS[:len(OLD_SPECS)]] == OLD_SPECS
    assert [s.key for s in cr._RATIO_SPECS[len(OLD_SPECS):]] == list(PGM_KEYS)
    for key, label, den_label, den, fb_den in PGM:
        s = _spec(key)
        assert (s.label, s.denominator_label, s.primary_num, s.primary_den, s.fallback_num, s.fallback_den) == (
            label, den_label, "GC=F", den, "GLD", fb_den)
        assert s.cheap_direction == "high" and s.min_years == 0
    assert len({s.key for s in cr._RATIO_SPECS}) == len(cr._RATIO_SPECS)
    assert "Five cross-asset ratios" not in cr.__doc__ and "all 5" not in cr.fetch_all_ratios.__doc__


@pytest.mark.parametrize("key,_label,_den_label,den,fb_den", PGM)
@pytest.mark.parametrize("pair", ["primary", "fallback"])
def test_compute_with_primary_and_fallback_pairs(monkeypatch, key, _label, _den_label, den, fb_den, pair):
    r = _compute(monkeypatch, key, _cheap_pgm(), pair)
    assert r.error is None and r.cheap_direction == "high"
    assert (r.numerator_label, r.denominator_leg_label) == (("GC=F", den) if pair == "primary" else ("GLD", fb_den))
    assert r.status == "RUBBER_BAND_STRETCHED" and r.percentile >= 99.0 and r.driver == "DENOMINATOR_CHEAP"
    assert r.denominator_pctile < 1.0 and 20.0 < r.numerator_pctile < 80.0
    assert r.current == pytest.approx(2.5, abs=1e-3) and r.history_start == str(IDX[0].date())
    # exakt regelfunktionerna på samma serier
    gold, pgm = _cheap_pgm()
    arr = (gold / pgm).values
    assert r.status == cr._classify(cr._percentile_of(arr, float(arr[-1])), "high")
    assert r.driver == cr._classify_driver(cr._percentile_of(gold.values, float(gold.values[-1])),
                                           cr._percentile_of(pgm.values, float(pgm.values[-1])), "high")


@pytest.mark.parametrize("key", PGM_KEYS)
def test_numerator_expensive_and_data_gap(monkeypatch, key):
    r = _compute(monkeypatch, key, _dear_gold())
    assert r.status == "RUBBER_BAND_STRETCHED" and r.driver == "NUMERATOR_EXPENSIVE"
    monkeypatch.setattr(cr, "_download_close", lambda t, period="10y": pd.Series(dtype=float))
    gap = cr._compute_ratio(_spec(key))
    assert gap.status == "DATA_GAP" and gap.error == f"Data unavailable for {key} (primary + fallback failed)"


def test_high_direction_thresholds_are_the_shared_ones():
    for pct, status in ((90.0, "RUBBER_BAND_STRETCHED"), (89.9, "TENSION_BUILDING"), (80.0, "TENSION_BUILDING"),
                        (79.9, "NEUTRAL"), (5.0, "NEUTRAL")):
        assert cr._classify(pct, _spec("gold_platinum").cheap_direction) == status


# ── Exponeringen ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name,exp", [
    ("Gold Miners", "gold_miner"), ("Silver Mining", "silver"), ("Copper Mining", "copper"),
    ("Platinum Group Metals", "platinum"), ("Sibanye Stillwater", None), ("Impala Platinum", "platinum"),
    ("Palladium One", "palladium"), ("Platina", "platinum"), ("PGM producers", "platinum"),
    ("Basic Materials Other Precious Metals & Mining Sibanye Platinum", "platinum"),
])
def test_detect_exposure_representative_names(name, exp):
    assert cr.detect_exposure(name) == exp


@pytest.mark.parametrize("name,exp", [
    ("Restaurang & Café", None), ("Toiletries", None), ("Turmoil Holdings", None), ("Uranium Royalty", "uranium"),
    ("Urangruvor", "uranium"), ("Guldgruvor", "gold_miner"), ("Kopparproducent", "copper"), ("Oil & Gas", "oil"),
    ("Petrochemicals", "oil"), ("LNG shipping", "gas"), ("Gruv - Guld & Silver", "silver"),
])
def test_keywords_must_start_a_word(name, exp):
    """Förr räckte det att nyckelordet stod någonstans i namnet: 'Restaurang & Café' blev uran (och fick URA som
    sektor-ETF i relativstyrkan). Nu ska ordet börja ett ord; svenska sammansättningar börjar med råvaran."""
    assert cr.detect_exposure(name) == exp


def test_names_without_a_pgm_word_route_exactly_as_before():
    names = ["Gold Miners", "Silver Mining", "Copper Mining", "Sibanye Stillwater", "Gruvor & metaller", "Uranium",
             "Natural gas", "Lithium mining", "Förnybar energi", "Oil & Gas E&P", "Junior Miners", "Silver", "Oil",
             "Copper", "Guldgruvor", "Precious Metals & Mining", "Metals & Mining", "Gold", "GLD", "Petroleum",
             "Steel", "Software", "Silver miner", "Juniormine", "LNG shipping", "Koppar", "Litium", "", None]
    for name in names:
        assert cr.detect_exposure(name) == _old_detect(name), name
    pgm_words = ("platinum", "platina", "pgm", "palladium")
    for kws, exp in OLD_KEYWORDS:                         # de gamla orden: samma exponering, samma inbördes ordning
        assert not any(w in k for w in pgm_words for k in kws)
    assert [e for e in cr._KEYWORD_EXPOSURE if e[1] not in ("platinum", "palladium")] == OLD_KEYWORDS


def test_pgm_keywords_stand_before_the_generic_gold_and_mining_entries():
    exps = [exp for _kws, exp in cr._KEYWORD_EXPOSURE]
    kws = {exp: k for k, exp in cr._KEYWORD_EXPOSURE if exp in ("platinum", "palladium")}
    assert kws == {"platinum": ["platinum group", "platinum", "platina", "pgm"], "palladium": ["palladium"]}
    generic = [i for i, (k, _e) in enumerate(cr._KEYWORD_EXPOSURE) if "gold" in k or "mining" in k]
    assert max(exps.index("platinum"), exps.index("palladium")) < min(generic)


# ── Registren ────────────────────────────────────────────────────────────────
def test_exposure_and_theme_registries():
    from blindspot.theme_board import THEME_RATIO_KEYS, _THEMES
    from thor import ravara as rv
    assert cr.EXPOSURE_TO_RATIO["platinum"] == ["gold_platinum", "gold_palladium"]
    assert cr.EXPOSURE_TO_RATIO["palladium"] == ["gold_palladium", "gold_platinum"]
    assert THEME_RATIO_KEYS["platina"] == cr.EXPOSURE_TO_RATIO["platinum"]
    assert THEME_RATIO_KEYS["palladium"] == cr.EXPOSURE_TO_RATIO["palladium"]
    assert {"platina", "palladium"} <= {t.key for t in _THEMES}          # blindspot-temana finns redan
    assert rv.EXPOSURE["platina"] == "platinum" and rv.EXPOSURE["palladium"] == "palladium"
    assert rv.ratio_keys("platina") == ["gold_platinum", "gold_palladium"]
    assert rv.ratio_keys("palladium") == ["gold_palladium", "gold_platinum"]
    for theme in ("platina", "palladium"):
        cov = rv.coverage(theme)
        assert cov["ratios"] == rv.ratio_keys(theme) and cov["gold_silver"] is False
        assert cov["blindspot"] == theme and cov["gold_pair"] == theme      # Guldkvoter-paret som förut
    # befintliga teman orörda
    assert rv.ratio_keys("silver") == ["gold_silver", "silver_juniors"] and rv.ratio_keys("uran") == []


def test_tactical_sector_etf():
    from alpha_regime import tactical_entry as te
    assert te._EXPOSURE_TO_ETF["platinum"] == "PPLT" and te._EXPOSURE_TO_ETF["palladium"] == "PALL"
    assert te.resolve_sector_etf(cr.EXPOSURE_TO_RATIO["platinum"]) == "PPLT"
    assert te.resolve_sector_etf(cr.EXPOSURE_TO_RATIO["palladium"]) == "PALL"
    assert te.resolve_sector_etf(["gold_silver"]) == "SIL" and te.resolve_sector_etf(["metal_miners"]) == "GDX"
    assert te.resolve_sector_etf(None) == "SPY"
    for exp in cr.EXPOSURE_TO_RATIO:
        assert exp in te._EXPOSURE_TO_ETF, exp


def test_exposure_picker_reads_the_registry():
    assert aui._EXP_OPTIONS == ["Auto", "Gold Miners", "Junior Miners", "Silver", "Platinum", "Palladium", "Oil",
                                "Copper", "None"]
    for option, keys in OLD_PICKER.items():                        # gamla valen: exakt samma listor som förut
        assert aui._exposure_override(option) == keys, option
    assert aui._exposure_override("Platinum") == cr.EXPOSURE_TO_RATIO["platinum"]
    assert aui._exposure_override("Palladium") == cr.EXPOSURE_TO_RATIO["palladium"]
    assert aui._exposure_override("okänt") is None
    got = aui._exposure_override("Platinum")
    got.append("x")                                               # en kopia — registret ändras aldrig
    assert cr.EXPOSURE_TO_RATIO["platinum"] == ["gold_platinum", "gold_palladium"]
    for option, exp in aui._EXPOSURE_PICKER.items():
        assert exp in cr.EXPOSURE_TO_RATIO, option


def test_confidence_registry_does_not_connect_the_pgm_ratios():
    """Förr (#300) kopplade registret platina/palladium till sina kvoter (ratio_exposure). Sedan ägarens beslut
    2026-11-03 är PGM-snoddarna bara kontext: ratio_exposure är tomt som före #300, så Why Now läser dem inte."""
    from confidence import commodities as com
    from confidence.why_now import signals_from_sources
    assert com.get("platinum").ratio_exposure == "" and com.get("palladium").ratio_exposure == ""
    ratios = {"gold_platinum": {"status": "RUBBER_BAND_STRETCHED"}, "gold_palladium": {"status": "NEUTRAL"}}
    for metal in ("platinum", "palladium"):
        s = signals_from_sources(com.get(metal), ratio_results=ratios)
        assert (s.ratio_status, s.ratio_key) == (None, ""), metal


def _literals(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _sources(*needles):
    """{relativ sökväg: strängliteraler} för panelens Python-filer (inte testerna) som nämner någon av needles."""
    skip = {"tests", ".claude", ".git", "__pycache__", "venv", ".venv", "node_modules"}
    out = {}
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in skip]
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(d, f)
            with open(p, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            if any(n in text for n in needles):
                out[os.path.relpath(p, ROOT)] = _literals(p)
    return out


def test_every_registry_that_lists_silver_also_lists_platinum_and_palladium():
    """Genomsökning: varje fil som listar silvrets kvoter (strängen 'silver_juniors' finns bara i kvotlistor)
    listar Gold/Platinum och Gold/Palladium; varje fil som mappar exponeringar ('gold_miner' och 'silver')
    känner 'platinum' och 'palladium'."""
    ratio_regs = {p: lits for p, lits in _sources("silver_juniors").items() if "silver_juniors" in lits}
    assert {"alpha_regime/commodity_ratios.py", "alpha_regime/tactical_entry.py", "blindspot/theme_board.py"} <= set(
        ratio_regs)
    for path, lits in ratio_regs.items():
        assert set(PGM_KEYS) <= lits, path
    exp_regs = {p: lits for p, lits in _sources("gold_miner").items() if {"gold_miner", "silver"} <= lits}
    assert {"alpha_regime/commodity_ratios.py", "alpha_regime/tactical_entry.py", "alpha_regime/ui.py",
            "confidence/commodities.py", "thor/ravara.py"} <= set(exp_regs)
    for path, lits in exp_regs.items():
        assert {"platinum", "palladium"} <= lits, path
    # buy_support: kvoterna står uppräknade i modulens docstring (PGM-kvoterna som bara kontext sedan 2026-11-03)
    assert "gold_silver" in bs.__doc__ and "gold_platinum" in bs.__doc__ and "gold_palladium" in bs.__doc__
    # regeldokumenten nämner dem där Gold/Silver står — sedan 2026-11-03 som bara kontext (tests/test_pgm_kontext.py)
    for rel in ("strategy_rules.py", os.path.join("ovtlyr", "ui", "rules_page.py")):
        text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        assert "Gold/Silver" in text and "Gold/Platinum" in text and "Gold/Palladium" in text, rel


# ── Regeln: bara kontext sedan 2026-11-03 (förr den generiska, som Gold/Silver) ──
def test_the_gold_miner_margin_rule_is_untouched():
    assert bs.MARGIN_RULE == {"metal_miners": "GDX", "gold_gdxj": "GDXJ"} and bs.MARGIN_MAX_PCT == 30.0
    assert bs.ENGINE_DRIVERS == ("DENOMINATOR_CHEAP", "BOTH")
    assert bs.SUMMARY_DRIVERS == ("DENOMINATOR_CHEAP", "BOTH", "UNKNOWN")
    for key in PGM_KEYS:
        assert not bs.uses_margin(key)
    assert bs.margin_miners(cr.EXPOSURE_TO_RATIO["platinum"]) == []
    assert bs.compute_margins(cr.EXPOSURE_TO_RATIO["palladium"]) == {}
    assert cr.EXPOSURE_TO_RATIO["gold_miner"] == ["metal_miners", "gdxj_gdx", "gold_gdxj", "gdx_spy"]
    assert cr.EXPOSURE_TO_RATIO["junior_miner"] == ["gdxj_gdx", "gold_gdxj"]


@pytest.mark.parametrize("key", PGM_KEYS)
def test_rule_and_summary_never_support_pgm_for_any_status_driver_and_percentile(key):
    """Förr: samma utfall som Gold/Silver (drivarregeln). Sedan 2026-11-03 bara kontext: aldrig köpstöd och aldrig
    KÖP-STÖD i sammanräkningen, där Gold/Silver med samma läge fortfarande följer drivarregeln."""
    margins = {"GDX": _margin("GDX", 5.0), "GDXJ": _margin("GDXJ", 5.0)}
    for status in STATUSES:
        for driver in DRIVERS:
            for pct in (3.0, 10.0, 50.0, 85.0, 89.9, 90.0, 97.0):
                r, silver = _ratio(key, status=status, percentile=pct, driver=driver), _ratio(
                    "gold_silver", status=status, percentile=pct, driver=driver)
                sup = bs.evaluate(key, r, margins)
                assert (sup.supported, sup.reason, sup.details["rule"]) == (False, bs.CONTEXT, "context")
                assert "margin_pct" not in sup.details
                assert bs.evaluate("gold_silver", silver, margins).supported == _old_engine_confirms(silver)
                want = "Gummisnoddar: 1 DATA_GAP" if status == "DATA_GAP" else "Gummisnoddar: 1 KONTEXT"
                assert aui._compute_ratio_summary({key: r}, [key], margins) == want


@pytest.mark.parametrize("branch,first,second", [("Platinum Group Metals", "gold_platinum", "gold_palladium"),
                                                 ("Palladium", "gold_palladium", "gold_platinum")])
def test_engine_never_gives_plus_one_for_pgm(monkeypatch, branch, first, second):
    """Förr gav den egna metallens kvot (eller den andra) +1 med en billig drivare. Sedan 2026-11-03: aldrig —
    och marginalmätaren läses fortfarande aldrig för PGM."""
    margins = {"GDX": _margin("GDX", 5.0), "GDXJ": _margin("GDXJ", 5.0)}
    for driver in DRIVERS:
        ratios = {first: _ratio(first, driver=driver, denominator_pctile=9.0),
                  second: _ratio(second, driver=driver, denominator_pctile=9.0)}
        r, rb, calls = _run_engine(monkeypatch, branch, ratios, margins)
        assert r.detected_exposure == [first, second]
        assert rb == [], driver
        assert calls == [] and r.miner_margins == {}               # marginalmätaren läses aldrig för PGM


@pytest.mark.parametrize("branch", ["Platinum Group Metals", "Palladium"])
def test_engine_table_never_confirms_pgm(monkeypatch, branch):
    """Förr: samma rad som den gamla drivarregeln gav. Nu: ingen rad i någon status × drivare."""
    keys = cr.EXPOSURE_TO_RATIO[cr.detect_exposure(branch)]
    for status in STATUSES:
        for driver in DRIVERS:
            ratios = {k: _ratio(k, status=status, driver=driver) for k in keys}
            _r, rb, _calls = _run_engine(monkeypatch, branch, ratios, {})
            assert rb == [], (branch, status, driver)


def test_engine_from_fake_price_series(monkeypatch):
    """Hela vägen: fejkade kurser → _compute_ratio → motorn. Förr gav billig platina +1; sedan 2026-11-03 ger
    varken billig platina eller dyrt guld någon rad — kvoten är fortfarande sträckt och visas som kontext."""
    from alpha_regime import engine as eng
    cheap = _compute(monkeypatch, "gold_platinum", _cheap_pgm())
    pall = _compute(monkeypatch, "gold_palladium", _dear_gold())
    assert cheap.status == "RUBBER_BAND_STRETCHED" and cheap.driver == "DENOMINATOR_CHEAP"
    for branch in ("Impala Platinum", "Palladium One"):
        _r, rb, _calls = _run_engine(monkeypatch, branch, {"gold_platinum": cheap, "gold_palladium": pall}, {})
        assert rb == [], branch
    _r, rb, _calls = _run_engine(monkeypatch, "Palladium One", {"gold_palladium": pall}, {})
    assert rb == [] and eng._bs is bs


# ── Kortet (steg 1: ärliga ord och terms-raden; sedan 2026-11-03 bara kontext) ──
PLATINUM_LINE = ("Bara kontext sedan 2026-11-03 — backtestet visade att en sträckt kvot följdes av att metallen gick "
                 f"sämre än guld (platina: den tidigare regeln gav 3 köplägen, snitt −16,9{NB}% mot GLD på ett år, "
                 f"0{NB}% träffar); räknas inte som köpstöd.")
PALLADIUM_LINE = ("Bara kontext sedan 2026-11-03 — backtestet visade att en sträckt kvot följdes av att metallen gick "
                  f"sämre än guld (palladium: för få köplägen, de två liknande gav −36{NB}% och −31{NB}% mot GLD); "
                  "räknas inte som köpstöd.")


def test_card_texts_for_gold_platinum():
    """Förr: märket 🟢 KÖP-STÖD och drivarens rad ('panelens regel räknar det som köpstöd i PL=F'), 🔴 SÄLJ-STÖD i
    säljzonen. Sedan 2026-11-03: märket KONTEXT med snoddens läge och beslutets rad i varje zon; terms-raden och
    benens ord (steg 1) är kvar som information."""
    r = _ratio("gold_platinum", driver="DENOMINATOR_CHEAP", numerator_pctile=55.0, denominator_pctile=9.0,
               current=2.5, median=1.6)
    assert aui._ratio_verdict(r)[:2] == ("⚪ KONTEXT · sträckt", "#607080")
    assert aui._driver_verdict_text(r) == ("⚪", PLATINUM_LINE)
    assert aui._cheap_terms_text(r) == (f"I GC=F: 1 PL=F = 0,400 GC=F — 36{NB}% mindre än medianen sedan 2016 "
                                        f"(0,625), alltså billigare i GC=F än normalt.")
    assert aui._leg_words(r) == ("mitten", "nära botten")
    dear = _ratio("gold_platinum", driver="NUMERATOR_EXPENSIVE", numerator_pctile=95.0, denominator_pctile=50.0)
    assert aui._driver_verdict_text(dear) == ("⚪", PLATINUM_LINE)
    for driver in DRIVERS:
        assert aui._driver_verdict_text(_ratio("gold_platinum", status="NEUTRAL", driver=driver)) == (
            "⚪", PLATINUM_LINE)
    sell = _ratio("gold_palladium", status="NEUTRAL", percentile=5.0)
    assert aui._ratio_verdict(sell)[0] == "⚪ KONTEXT · dyr"
    assert aui._driver_verdict_text(sell) == ("⚪", PALLADIUM_LINE)


def _card_app():
    import streamlit as st
    from alpha_regime import ui as u
    u._render_single_ratio(st.session_state["_ratio"], idx=0, margins=st.session_state.get("_margins"))


def _render_card(ratio, margins=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_card_app, default_timeout=60)
    at.session_state["_ratio"] = ratio
    at.session_state["_margins"] = margins
    at.run()
    assert not at.exception, at.exception
    return " ".join(m.value for m in at.markdown)


def test_apptest_gold_platinum_card_renders():
    """Förr: 🟢 KÖP-STÖD och drivarens köpstödsrad. Sedan 2026-11-03: KONTEXT-märket och beslutets rad — benens
    rader och terms-raden (steg 1) visas som förut."""
    r = _ratio("gold_platinum", driver="DENOMINATOR_CHEAP", numerator_pctile=55.0, denominator_pctile=9.0,
               current=2.5, median=1.6, sparkline_dates=["2026-10-23", "2026-10-30"], sparkline_values=[2.45, 2.5])
    html = _render_card(r, {"GDX": _margin("GDX", 45.0)})
    for part in ("Gold / Platinum", "⚪ KONTEXT · sträckt",
                 "Gold / Platinum vid 94:e percentilen — Platinum historiskt max-billig",
                 "Täljare (GC=F):", "55:e pct — mitten av sitt eget prisspann sedan 2016",
                 "Nämnare (PL=F):", "9:e pct — nära botten av sitt eget prisspann sedan 2016",
                 f"⚪ {PLATINUM_LINE}",
                 f"I GC=F: 1 PL=F = 0,400 GC=F — 36{NB}% mindre än medianen sedan 2016 (0,625)"):
        assert part in html, part
    assert "KÖP-STÖD" not in html and "🟢" not in html and "panelens regel räknar" not in html
    assert "Benens prisläge är bara information" not in html and "marginalmätaren" not in html
    neutral = _render_card(_ratio("gold_platinum", status="NEUTRAL", driver="DENOMINATOR_CHEAP"))
    assert "⚪ KONTEXT · neutral" in neutral and PLATINUM_LINE in neutral and NEUTRAL_LINE not in neutral


def test_apptest_fallback_card_names_the_etfs(monkeypatch):
    """Reservparet: benen och terms-raden namnger ETF:erna som förut; köpstödsraden ('köpstöd i PALL.') är borta."""
    r = _compute(monkeypatch, "gold_palladium", _cheap_pgm(), "fallback")
    html = _render_card(r)
    assert "Täljare (GLD):" in html and "Nämnare (PALL):" in html and "I GLD: 1 PALL = " in html
    assert PALLADIUM_LINE in html and "köpstöd i PALL." not in html and "⚪ KONTEXT · sträckt" in html


# ── Panelen och Thor ─────────────────────────────────────────────────────────
def _panel_app():
    from alpha_regime import ui as u
    u.render_alpha_regime()


def test_apptest_picker_shows_the_pgm_gauges(monkeypatch):
    from streamlit.testing.v1 import AppTest
    ratios = {"gold_platinum": _ratio("gold_platinum", driver="DENOMINATOR_CHEAP", denominator_pctile=9.0),
              "gold_palladium": _ratio("gold_palladium", status="NEUTRAL", driver="MIXED"),
              "gold_silver": _ratio("gold_silver", status="NEUTRAL", driver="MIXED")}
    result = _regime(ratios, [], {})
    monkeypatch.setattr(aui, "run_regime_analysis", lambda **_kw: result)
    monkeypatch.setattr(aui, "_cached_deep_rows", lambda: ([], ""))
    monkeypatch.setattr(mm, "compute", lambda miner="GDX": pytest.fail("mätaren lästes"))
    at = AppTest.from_function(_panel_app, default_timeout=60)
    at.session_state["ar_mode"] = "contrarian"
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="ar_exposure").options == aui._EXP_OPTIONS
    at.selectbox(key="ar_exposure").select("Platinum").run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert html.index(">Gold / Platinum<") < html.index(">Gold / Palladium<") and ">Gold / Silver<" not in html
    assert "Gummisnoddar: 2 KONTEXT" in html          # förr '1 KÖP-STÖD, 1 NEUTRAL' — bara kontext sedan 2026-11-03
    assert "KÖP-STÖD" not in html
    at.selectbox(key="ar_exposure").select("Palladium").run()
    html = " ".join(m.value for m in at.markdown)
    assert html.index(">Gold / Palladium<") < html.index(">Gold / Platinum<")


def test_thor_ravara_shows_the_pgm_bands(monkeypatch):
    from gold_ratios import ui as gu
    from thor import ravara as rv
    ratios = {"gold_platinum": _ratio("gold_platinum", driver="DENOMINATOR_CHEAP", denominator_pctile=9.0),
              "gold_palladium": _ratio("gold_palladium", status="NEUTRAL", driver="MIXED")}
    monkeypatch.setattr(cr, "fetch_all_ratios", lambda: ratios)
    monkeypatch.setattr(gu, "_load", lambda: {})
    monkeypatch.setattr(mm, "compute", lambda miner="GDX": pytest.fail("mätaren lästes"))
    for theme in ("platina", "palladium"):
        st = _St()
        rv._ratios(st, theme, lambda text: st.md.append(text))
        html = " ".join(st.md)
        # förr 'P94 · rubber band stretched · denominator cheap' — sedan 2026-11-03 läget som kontext, ingen drivare
        assert "Gold / Platinum" in html and "kontext · P94 · sträckt" in html
        assert "Gold / Palladium" in html and "kontext · P50 · neutral" in html
        assert "denominator cheap" not in html and "mixed" not in html and "Inga kvoter" not in html
        assert "köpstöd ·" not in html and "· köpstöd" not in html
        assert "Gold/Platinum och Gold/Palladium är bara kontext sedan 2026-11-03" in html


# ── Rättelser efter verifieringen ────────────────────────────────────────────
def _borsdata_names():
    from contrarian_alpha.necessity import BORSDATA_BRANCH_MAP, BORSDATA_SECTOR_MAP
    return [e.label for m in (BORSDATA_BRANCH_MAP, BORSDATA_SECTOR_MAP) for e in m.values()]


def test_why_now_has_no_band_for_pgm_and_silver_is_unchanged():
    """Förr (#300) räknade Why Now gummibandet för platina och palladium som för silver (15/10/5 poäng, högre
    täckning). Sedan ägarens beslut 2026-11-03 är PGM-snoddarna bara kontext: komponenten saknas för platina och
    palladium som före #300, och silvret räknas exakt som förut."""
    import dataclasses
    from confidence import commodities as com
    from confidence import config as ccfg
    from confidence.why_now import signals_from_sources, why_now
    assert ccfg.WHY_NOW_RATIO == {"RUBBER_BAND_STRETCHED": 15, "TENSION_BUILDING": 10, "NEUTRAL": 5}
    for status, pts in ccfg.WHY_NOW_RATIO.items():
        ratios = {k: {"status": status} for k in ("gold_silver", "silver_juniors", "gold_platinum", "gold_palladium")}
        c = com.get("silver")
        w = why_now(c, signals_from_sources(c, ratio_results=ratios))
        assert w.components["Gummiband"] == pts and "ratio_status" not in w.missing, status
        for metal in ("platinum", "palladium"):
            c = com.get(metal)
            w = why_now(c, signals_from_sources(c, ratio_results=ratios))
            assert "Gummiband" not in w.components and "ratio_status" in w.missing, (metal, status)
            before = dataclasses.replace(c, ratio_exposure="")             # registret före #300
            assert w == why_now(before, signals_from_sources(before, ratio_results=ratios))
    # silvret: DATA_GAP i den egna kvoten → silver_juniors tar över som förut
    gap = {"gold_palladium": {"status": "DATA_GAP"}, "gold_platinum": {"status": "RUBBER_BAND_STRETCHED"},
           "gold_silver": {"status": "DATA_GAP"}, "silver_juniors": {"status": "TENSION_BUILDING"}}
    s = signals_from_sources(com.get("palladium"), ratio_results=gap)
    assert (s.ratio_status, s.ratio_key) == (None, "")
    s = signals_from_sources(com.get("silver"), ratio_results=gap)
    assert (s.ratio_status, s.ratio_key) == ("TENSION_BUILDING", "silver_juniors")


def test_auto_never_reaches_pgm_from_a_borsdata_branch_and_the_help_says_so():
    """Auto läser bara Börsdatas branschnamn — inget av dem innehåller ett PGM-ord, så de routas exakt som förut
    (Sibanyes 'Gruv - Guld & Silver' är silver). PGM-snoddarna nås via väljaren; hjälptexten säger det."""
    names = _borsdata_names()
    assert "Gruv - Guld & Silver" in names and len(names) > 90
    fixed = {"Restaurang & Café": ("uranium", None)}      # 'uran' mitt i ordet — nyckelorden börjar nu ett ord
    for name in names:
        if name in fixed:
            assert (_old_detect(name), cr.detect_exposure(name)) == fixed[name], name
            continue
        assert cr.detect_exposure(name) == _old_detect(name), name
        assert cr.detect_exposure(name) not in ("platinum", "palladium"), name
    assert cr.detect_exposure("Gruv - Guld & Silver") == "silver"
    for part in ("Auto reads the Börsdata branch", "no platinum or palladium branch", "Platinum or Palladium"):
        assert part in aui._EXPOSURE_HELP, part


@pytest.mark.parametrize("name,exp,before", [("Gold, Platinum & Palladium", "platinum", "gold_miner"),
                                             ("Gold & Palladium", "palladium", "gold_miner"),
                                             ("Oil & PGM", "platinum", "oil")])
def test_mixed_names_route_to_pgm_as_specified(name, exp, before):
    """Enligt specen (PGM-orden före de generiska guld-/gruvorden): ett namn med både guld/olja och ett PGM-ord blir
    PGM. Ägarbeslut om det ska ändras — inget Börsdata-branschnamn träffas (testet ovan)."""
    assert cr.detect_exposure(name) == exp and _old_detect(name) == before


@pytest.mark.parametrize("own,other", [("gold_palladium", "gold_platinum"), ("gold_platinum", "gold_palladium")])
def test_no_cross_metal_confirmation(monkeypatch, own, other):
    """Förr (enligt #300:s spec) kunde den andra metallens kvot ge +1 när bara den metallen var billig. Sedan
    2026-11-03 är båda bara kontext: ingen korsbekräftelse — exponeringslistan visar fortfarande båda."""
    branch = {"gold_palladium": "Palladium One", "gold_platinum": "Impala Platinum"}[own]
    ratios = {own: _ratio(own, status="NEUTRAL", driver="MIXED"), other: _ratio(other, driver="DENOMINATOR_CHEAP")}
    r, rb, _calls = _run_engine(monkeypatch, branch, ratios, {})
    assert r.detected_exposure == [own, other]
    assert rb == []


def test_picker_without_the_module_returns_a_non_empty_list(monkeypatch):
    monkeypatch.setattr(aui, "_CR_EXPOSURE_TO_RATIO", None)
    assert aui._exposure_override("None") == [] and aui._exposure_override("Auto") is None
    assert aui._exposure_override("okänt") is None
    ratio_keys = {s.key for s in cr._RATIO_SPECS}
    for option, exp in aui._EXPOSURE_PICKER.items():                # exponeringsnyckeln — aldrig en kvotnyckel
        assert aui._exposure_override(option) == [exp] and exp not in ratio_keys, option


def _run_panel(monkeypatch, result, option):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(aui, "run_regime_analysis", lambda **_kw: result)
    monkeypatch.setattr(aui, "_cached_deep_rows", lambda: ([], ""))
    monkeypatch.setattr(mm, "compute", lambda miner="GDX": pytest.fail("mätaren lästes"))
    at = AppTest.from_function(_panel_app, default_timeout=60)
    at.session_state["ar_mode"] = "contrarian"
    at.run()
    assert not at.exception, at.exception
    at.selectbox(key="ar_exposure").select(option).run()
    assert not at.exception, at.exception
    return at, " ".join(c.value for c in at.caption)


@pytest.mark.parametrize("option", ["Platinum", "Silver", "Gold Miners"])
def test_apptest_module_not_loaded_caption_survives_the_picker(monkeypatch, option):
    """Utan commodity_ratios (motorn har då inga kvoter): ett manuellt val visar raden 'module not loaded' som
    förut i stället för att tyst läsas som "None"."""
    monkeypatch.setattr(aui, "_CR_EXPOSURE_TO_RATIO", None)
    at, caption = _run_panel(monkeypatch, _regime({}, [], {}), option)
    assert "Commodity ratio data unavailable (module not loaded)." in caption
    assert at.selectbox(key="ar_exposure").help == aui._EXPOSURE_HELP
    _at, caption = _run_panel(monkeypatch, _regime({}, [], {}), "None")
    assert "module not loaded" not in caption and "not detected" not in caption
