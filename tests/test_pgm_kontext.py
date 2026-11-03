"""
PGM-snoddarna är bara kontext — ägarens beslut A (2026-11-03) efter det förregistrerade backtestet
docs/forskning/pgm_snoddar_spec.md.

Gold/Platinum (gold_platinum) och Gold/Palladium (gold_palladium) visas kvar (kort, Thors Råvara-flik, temabordet,
exponeringsväljaren) men ger aldrig köpstöd: buy_support.evaluate svarar CONTEXT_ONLY i varje läge, motorns +1
ACCUMULATE och KÖP-STÖD i 'Gummisnoddar' räknar dem aldrig, kortets märke är aldrig grönt och Why Now har ingen
gummibandskomponent för platina och palladium. Varje annan kvot (gruvornas marginalregel och drivarregeln) avgör
exakt som origin/main före beslutet — prövat tabelldrivet mot en ordagrann, fryst kopia av den koden. Inget nätverk:
fejkade kvoter och marginaler, motorns nedladdningar ersatta.
"""
import dataclasses
import itertools
import math
import os
import re
import sys
from typing import NamedTuple, Optional

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_regime import buy_support as bs  # noqa: E402
from alpha_regime import commodity_ratios as cr  # noqa: E402
from alpha_regime import engine as eng  # noqa: E402
from alpha_regime import miner_margin as mm  # noqa: E402
from alpha_regime import ui as aui  # noqa: E402
from test_kopstod_gruvor import (  # noqa: E402
    DRIVERS, PCTS, STATUSES, _St, _gaps, _margin, _ratio, _regime, _run_engine)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = " "
PGM_KEYS = ("gold_platinum", "gold_palladium")
OTHER_KEYS = tuple(s.key for s in cr._RATIO_SPECS if s.key not in PGM_KEYS)
CUSTOM_KEYS = ("custom_GLD_GDX", "custom_SLV_SIL", "custom_GC=F_PL=F")
PGM_BRANCHES = ("Platinum Group Metals", "Palladium")
PLATINUM_LINE = ("Bara kontext sedan 2026-11-03 — backtestet visade att en sträckt kvot följdes av att metallen gick "
                 f"sämre än guld (platina: den tidigare regeln gav 3 köplägen, snitt −16,9{NB}% mot GLD på ett år, "
                 f"0{NB}% träffar); räknas inte som köpstöd.")
PALLADIUM_LINE = ("Bara kontext sedan 2026-11-03 — backtestet visade att en sträckt kvot följdes av att metallen gick "
                  f"sämre än guld (palladium: för få köplägen, de två liknande gav −36{NB}% och −31{NB}% mot GLD); "
                  "räknas inte som köpstöd.")
LINES = {"gold_platinum": PLATINUM_LINE, "gold_palladium": PALLADIUM_LINE}


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear = getattr(mm._compute_cached, "clear", lambda: None)
    clear()
    yield
    clear()


# ── Fryst kopia av origin/main (f9bf20a) före beslutet — ordagrant, bara namnen har prefixet _main ──────────────
_MAIN_MARGIN_RULE: dict = {"metal_miners": "GDX", "gold_gdxj": "GDXJ"}
_MAIN_MARGIN_MAX_PCT = 30.0
_MAIN_STRETCHED = "RUBBER_BAND_STRETCHED"
_MAIN_ENGINE_DRIVERS = ("DENOMINATOR_CHEAP", "BOTH")
_MAIN_SUMMARY_DRIVERS = ("DENOMINATOR_CHEAP", "BOTH", "UNKNOWN")


class _MainBuySupport(NamedTuple):
    supported: bool
    reason: str
    details: dict


def _main_margin_percentile(res) -> Optional[float]:
    if res is None or getattr(res, "status", "") != "OK":
        return None
    try:
        p = float(getattr(res, "percentile", None))
    except (TypeError, ValueError):
        return None
    return p if math.isfinite(p) else None


def _main_evaluate(key, ratio, margins: Optional[dict] = None, *, stretched: Optional[bool] = None,
                   drivers: tuple = _MAIN_ENGINE_DRIVERS) -> _MainBuySupport:
    key = getattr(ratio, "key", None) or key
    rule = "margin" if key in _MAIN_MARGIN_RULE else "driver"
    det: dict = {"key": key, "rule": rule, "stretched": False}
    if rule == "margin":
        miner = _MAIN_MARGIN_RULE[key]
        res = (margins or {}).get(miner)
        pct = _main_margin_percentile(res)
        det.update(miner=miner, margin_pct=pct, margin_max=_MAIN_MARGIN_MAX_PCT,
                   margin_error=None if pct is not None else (getattr(res, "error", None) or "marginalen saknas"),
                   stale=bool(getattr(res, "stale", False)))
    if ratio is None or getattr(ratio, "status", "DATA_GAP") == "DATA_GAP":
        return _MainBuySupport(False, "NO_DATA", det)

    is_stretched = (ratio.status == _MAIN_STRETCHED) if stretched is None else bool(stretched)
    det.update(stretched=is_stretched, ratio_pct=getattr(ratio, "percentile", None), status=ratio.status,
               driver=getattr(ratio, "driver", "UNKNOWN"))
    if not is_stretched:
        return _MainBuySupport(False, "NOT_STRETCHED", det)
    if rule == "margin":
        pct = det["margin_pct"]
        if pct is None:
            return _MainBuySupport(False, "MARGIN_MISSING", det)
        if pct <= _MAIN_MARGIN_MAX_PCT:
            return _MainBuySupport(True, "MARGIN_CHEAP", det)
        return _MainBuySupport(False, "MARGIN_NOT_CHEAP", det)
    if det["driver"] in drivers:
        return _MainBuySupport(True, "DRIVER_CHEAP", det)
    return _MainBuySupport(False, "DRIVER_NOT_CHEAP", det)


def _main_ratio_verdict(ratio) -> tuple:
    pct = ratio.percentile
    d = ratio.cheap_direction
    lbl = ratio.label
    den = ratio.denominator_label
    _sv_ord = aui._sv_ord
    if d == "high":
        if pct >= 90.0:
            return ("🟢 KÖP-STÖD", "#1aaa5a",
                    f"{lbl} vid {_sv_ord(pct)} percentilen — {den} historiskt max-billig")
        if pct >= 80.0:
            return ("⚡ SPÄNNING BYGGS", "#c9a84c",
                    f"{lbl} vid {_sv_ord(pct)} percentilen — {den} närmar sig billighetszonen")
        if pct <= 10.0:
            return ("🔴 SÄLJ-STÖD", "#cc3333",
                    f"{lbl} vid {_sv_ord(pct)} percentilen — {den} historiskt max-dyr")
        return ("⚪ NEUTRAL", "#607080",
                f"{lbl} vid {_sv_ord(pct)} percentilen — ingen extrem")
    else:
        if pct <= 10.0:
            return ("🟢 KÖP-STÖD", "#1aaa5a",
                    f"{lbl} vid {_sv_ord(pct)} percentilen — {den} historiskt max-billig vs motvikten")
        if pct <= 20.0:
            return ("⚡ SPÄNNING BYGGS", "#c9a84c",
                    f"{lbl} vid {_sv_ord(pct)} percentilen — {den} närmar sig billighetszonen")
        if pct >= 90.0:
            return ("🔴 SÄLJ-STÖD", "#cc3333",
                    f"{lbl} vid {_sv_ord(pct)} percentilen — {den} historiskt max-dyr vs motvikten")
        return ("⚪ NEUTRAL", "#607080",
                f"{lbl} vid {_sv_ord(pct)} percentilen — ingen extrem")


def _main_summary(commodity_ratios: dict, keys: list, margins: Optional[dict] = None) -> str:
    counts: dict = {"KÖP-STÖD": 0, "SÄLJ-STÖD": 0, "SPÄNNING": 0, "NEUTRAL": 0, "DATA_GAP": 0}
    for key in keys:
        r = commodity_ratios.get(key)
        if r is None or r.status == "DATA_GAP":
            counts["DATA_GAP"] += 1
            continue
        badge, _, _ = _main_ratio_verdict(r)
        if "KÖP-STÖD" in badge:
            if _main_evaluate(key, r, margins, stretched=True, drivers=_MAIN_SUMMARY_DRIVERS).supported:
                counts["KÖP-STÖD"] += 1
            else:
                counts["NEUTRAL"] += 1
        elif "SÄLJ-STÖD" in badge:
            counts["SÄLJ-STÖD"] += 1
        elif "SPÄNNING" in badge:
            counts["SPÄNNING"] += 1
        else:
            counts["NEUTRAL"] += 1
    parts = []
    for label, key in [("KÖP-STÖD", "KÖP-STÖD"), ("SÄLJ-STÖD", "SÄLJ-STÖD"),
                       ("SPÄNNING", "SPÄNNING"), ("NEUTRAL", "NEUTRAL")]:
        if counts[key] > 0:
            parts.append(f"{counts[key]} {label}")
    if counts["DATA_GAP"] > 0:
        parts.append(f"{counts['DATA_GAP']} DATA_GAP")
    return ("Gummisnoddar: " + ", ".join(parts)) if parts else ""


# ── Indata för tabellerna ────────────────────────────────────────────────────
_GAP = mm.MarginResult(miner="GDX", error="saknas")
MARGIN_SETS = (None, {}, {"GDX": _margin("GDX", 5.0), "GDXJ": _margin("GDXJ", 30.0)},
               {"GDX": _margin("GDX", 30.1), "GDXJ": _margin("GDXJ", 95.0)},
               {"GDX": _margin("GDX", 22.0, stale=True), "GDXJ": _GAP},
               {"GDX": _margin("GDX", float("nan")), "GDXJ": _margin("GDXJ", None)})
STRETCHED_ARGS = (None, True, False)
DRIVER_ARGS = (bs.ENGINE_DRIVERS, bs.SUMMARY_DRIVERS, ("UNKNOWN",), tuple(DRIVERS))


def _states(key):
    for status, driver, pct in itertools.product(STATUSES, DRIVERS, PCTS):
        yield _ratio(key, status=status, percentile=pct, driver=driver, denominator_pctile=9.0)


# ── Regeln ───────────────────────────────────────────────────────────────────
def test_context_only_registry_and_the_rest_of_the_rule_constants():
    assert set(bs.CONTEXT_ONLY) == set(PGM_KEYS)
    for key in PGM_KEYS:
        assert bs.CONTEXT_ONLY[key] == {"decided": "2026-11-03", "spec": "docs/forskning/pgm_snoddar_spec.md"}
        assert bs.context_only(key) and not bs.uses_margin(key)
    assert os.path.exists(os.path.join(ROOT, bs.CONTEXT_SPEC)) and bs.CONTEXT_DECIDED == "2026-11-03"
    assert bs.CONTEXT == "CONTEXT_ONLY"
    for key in OTHER_KEYS + CUSTOM_KEYS + ("", None):
        assert not bs.context_only(key), key
    # gruvornas marginalregel och drivarna oförändrade
    assert bs.MARGIN_RULE == _MAIN_MARGIN_RULE and bs.MARGIN_MAX_PCT == _MAIN_MARGIN_MAX_PCT
    assert bs.ENGINE_DRIVERS == _MAIN_ENGINE_DRIVERS and bs.SUMMARY_DRIVERS == _MAIN_SUMMARY_DRIVERS
    assert bs.STRETCHED == _MAIN_STRETCHED and bs.DECIDED == "2026-11-03"
    assert bs.SPEC == "docs/forskning/gruvor_marginal_spec.md"
    assert bs.margin_miners(list(PGM_KEYS) + ["metal_miners"]) == ["GDX"]


@pytest.mark.parametrize("key", PGM_KEYS)
def test_pgm_never_has_buy_support_whatever_the_status_driver_margins_or_caller(key):
    for r in _states(key):
        for margins, stretched, drivers in itertools.product(MARGIN_SETS, STRETCHED_ARGS, DRIVER_ARGS):
            sup = bs.evaluate(key, r, margins, stretched=stretched, drivers=drivers)
            assert (sup.supported, sup.reason) == (False, bs.CONTEXT), (r.status, r.driver, r.percentile)
            assert sup.details["rule"] == "context" and sup.details["key"] == key
            assert sup.details["decided"] == "2026-11-03" and sup.details["spec"] == bs.CONTEXT_SPEC
            assert "margin_pct" not in sup.details and "miner" not in sup.details
    for ratio in (None, _ratio(key, status="DATA_GAP")):
        assert bs.evaluate(key, ratio, MARGIN_SETS[2]) == (False, bs.CONTEXT, {
            "key": key, "rule": "context", "stretched": False, "decided": "2026-11-03", "spec": bs.CONTEXT_SPEC})
    # kvotens egen .key gäller (som förut): en PGM-kvot under en annan nyckel är fortfarande bara kontext,
    # och en annan kvot under PGM-nyckeln följer sin egen regel
    assert bs.evaluate("gold_silver", _ratio(key, driver="DENOMINATOR_CHEAP")).reason == bs.CONTEXT
    silver = _ratio("gold_silver", driver="DENOMINATOR_CHEAP")
    assert bs.evaluate(key, silver) == _main_evaluate(key, silver)


@pytest.mark.parametrize("key", OTHER_KEYS + CUSTOM_KEYS)
def test_every_other_ratio_decides_exactly_as_origin_main(key):
    """Alla andra kvoter — guldgruvornas marginalregel, drivarregeln och egna kvoter — ger exakt samma BuySupport
    (supported, reason och details) som origin/main, för varje status × drivare × percentil × marginaler ×
    anroparens stretched × drivare."""
    spec_key = key if not key.startswith("custom_") else "metal_miners"
    for r in _states(spec_key):
        r = dataclasses.replace(r, key=key)
        for margins, stretched, drivers in itertools.product(MARGIN_SETS, STRETCHED_ARGS, DRIVER_ARGS):
            new = bs.evaluate(key, r, margins, stretched=stretched, drivers=drivers)
            old = _main_evaluate(key, r, margins, stretched=stretched, drivers=drivers)
            assert tuple(new) == tuple(old), (key, r.status, r.driver, r.percentile, stretched, drivers)
    for margins in MARGIN_SETS:
        assert tuple(bs.evaluate(key, None, margins)) == tuple(_main_evaluate(key, None, margins))


@pytest.mark.parametrize("key,miner", [("metal_miners", "GDX"), ("gold_gdxj", "GDXJ")])
def test_the_gold_miner_margin_rule_is_unchanged(key, miner):
    for pct, ok in ((0.0, True), (22.0, True), (30.0, True), (30.1, False), (45.0, False)):
        for driver in DRIVERS:
            sup = bs.evaluate(key, _ratio(key, driver=driver), {miner: _margin(miner, pct)})
            assert (sup.supported, sup.reason) == (ok, bs.MARGIN_CHEAP if ok else bs.MARGIN_NOT_CHEAP)
    for gap in _gaps(miner).values():
        assert bs.evaluate(key, _ratio(key, driver="BOTH"), {miner: gap}).reason == bs.MARGIN_MISSING


# ── Motorn ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("branch", PGM_BRANCHES)
def test_engine_never_gives_plus_one_for_platinum_or_palladium_exposures(monkeypatch, branch):
    """Varje status × drivare för den egna kvoten × varje status × drivare för den andra: ingen 'Rubber Band'-rad,
    och marginalmätaren läses aldrig."""
    keys = cr.EXPOSURE_TO_RATIO[cr.detect_exposure(branch)]
    assert set(keys) == set(PGM_KEYS)
    margins = {"GDX": _margin("GDX", 5.0), "GDXJ": _margin("GDXJ", 5.0)}
    combos = list(itertools.product(STATUSES, DRIVERS))
    for (s1, d1), (s2, d2) in itertools.product(combos, combos):
        ratios = {keys[0]: _ratio(keys[0], status=s1, driver=d1, denominator_pctile=9.0),
                  keys[1]: _ratio(keys[1], status=s2, driver=d2, denominator_pctile=9.0)}
        r, rb, calls = _run_engine(monkeypatch, branch, ratios, margins)
        assert rb == [] and calls == [] and r.miner_margins == {}, (s1, d1, s2, d2)


def _with_list(monkeypatch, keys):
    """Platinaexponeringen med en annan kvotlista (motorn läser sin egen referens till EXPOSURE_TO_RATIO)."""
    monkeypatch.setattr(eng, "_EXPOSURE_TO_RATIO", {**eng._EXPOSURE_TO_RATIO, "platinum": list(keys)})


def test_engine_loop_still_lets_a_later_non_pgm_ratio_confirm(monkeypatch):
    """Ingen PGM-kvot bekräftar, men loopen går vidare: en senare kvot med köpstöd får raden (max +1 som förut)."""
    pgm = {k: _ratio(k, driver="DENOMINATOR_CHEAP", denominator_pctile=9.0) for k in PGM_KEYS}
    _with_list(monkeypatch, [*PGM_KEYS, "gold_silver"])
    ratios = {**pgm, "gold_silver": _ratio("gold_silver", driver="DENOMINATOR_CHEAP")}
    r, rb, calls = _run_engine(monkeypatch, "Platinum Group Metals", ratios, {})
    assert r.detected_exposure == [*PGM_KEYS, "gold_silver"]
    assert rb == ["Rubber Band: Gold / Silver at 94th percentile — Silver historically stretched cheap"]
    assert calls == []
    ratios["gold_silver"] = _ratio("gold_silver", driver="NUMERATOR_EXPENSIVE")
    _r, rb, _calls = _run_engine(monkeypatch, "Platinum Group Metals", ratios, {})
    assert rb == []
    # en senare guldkvot: marginalregeln som förut
    _with_list(monkeypatch, ["gold_platinum", "metal_miners", "gold_palladium"])
    ratios = {**pgm, "metal_miners": _ratio("metal_miners", driver="NUMERATOR_EXPENSIVE")}
    r, rb, calls = _run_engine(monkeypatch, "Platinum Group Metals", ratios, {"GDX": _margin("GDX", 22.0)})
    assert rb == ["Rubber Band: GLD/GDX at 94th percentile — GDX margin gauge at 22nd pct (≤ 30)"]
    assert calls == ["GDX"]
    _r, rb, _calls = _run_engine(monkeypatch, "Platinum Group Metals", ratios, {"GDX": _margin("GDX", 45.0)})
    assert rb == []


@pytest.mark.parametrize("branch", ["Silver", "Oil", "Copper", "Junior Miners", "Gold Miners"])
def test_engine_for_every_other_exposure_matches_origin_main(monkeypatch, branch):
    """Motorn i övriga exponeringar: samma rad (eller ingen) som origin/main:s regel ger — med marginaler."""
    keys = cr.EXPOSURE_TO_RATIO[cr.detect_exposure(branch)]
    for margins in ({"GDX": _margin("GDX", 22.0), "GDXJ": _margin("GDXJ", 45.0)}, {}):
        for status, driver in itertools.product(STATUSES, DRIVERS):
            ratios = {k: _ratio(k, status=status, driver=driver) for k in keys}
            _r, rb, _calls = _run_engine(monkeypatch, branch, ratios, margins)
            want = []
            for k in keys:                                  # origin/main: första kvoten med köpstöd, max +1
                sup = _main_evaluate(k, ratios[k], margins)
                if sup.supported:
                    want = [eng._rubber_band_rationale(ratios[k], sup)]
                    break
            assert rb == want, (branch, status, driver)


# ── Sammanräkningen ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("key", PGM_KEYS)
def test_summary_never_counts_pgm_as_kop_stod(key):
    for r in _states(key):
        for margins in MARGIN_SETS:
            text = aui._compute_ratio_summary({key: r}, [key], margins)
            want = "Gummisnoddar: 1 DATA_GAP" if r.status == "DATA_GAP" else "Gummisnoddar: 1 KONTEXT"
            assert text == want, (r.status, r.driver, r.percentile)
    both = {k: _ratio(k, driver="DENOMINATOR_CHEAP", denominator_pctile=9.0) for k in PGM_KEYS}
    assert aui._compute_ratio_summary(both, list(PGM_KEYS)) == "Gummisnoddar: 2 KONTEXT"
    assert aui._compute_ratio_summary({key: both[key]}, list(PGM_KEYS)) == "Gummisnoddar: 1 KONTEXT, 1 DATA_GAP"


def test_summary_with_pgm_beside_other_ratios_keeps_their_counts():
    pgm = {k: _ratio(k, driver="BOTH", denominator_pctile=9.0) for k in PGM_KEYS}
    keys = ["gold_platinum", "gold_silver", "gold_palladium"]
    for status, driver, pct in itertools.product(STATUSES, DRIVERS, PCTS):
        silver = _ratio("gold_silver", status=status, percentile=pct, driver=driver)
        text = aui._compute_ratio_summary({**pgm, "gold_silver": silver}, keys)
        old = _main_summary({"gold_silver": silver}, ["gold_silver"])          # silvret ensamt, som origin/main
        if status == "DATA_GAP":
            assert old == "Gummisnoddar: 1 DATA_GAP" and text == "Gummisnoddar: 2 KONTEXT, 1 DATA_GAP"
        else:
            assert old.startswith("Gummisnoddar: 1 ") and "DATA_GAP" not in old
            assert text == f"{old}, 2 KONTEXT", (status, driver, pct)
    assert aui._compute_ratio_summary({**pgm, "gold_silver": _ratio("gold_silver", driver="DENOMINATOR_CHEAP")},
                                      keys) == "Gummisnoddar: 1 KÖP-STÖD, 2 KONTEXT"


@pytest.mark.parametrize("exposure", ["gold_miner", "junior_miner", "silver", "oil", "copper"])
def test_summary_for_every_other_exposure_matches_origin_main(exposure):
    keys = cr.EXPOSURE_TO_RATIO[exposure]
    for margins in MARGIN_SETS:
        for status, driver, pct in itertools.product(STATUSES, DRIVERS, PCTS):
            ratios = {k: _ratio(k, status=status, percentile=pct, driver=driver) for k in keys}
            assert aui._compute_ratio_summary(ratios, keys, margins) == _main_summary(ratios, keys, margins)
        mixed = {k: _ratio(k, status=s, driver=d) for k, s, d in zip(
            keys, ("RUBBER_BAND_STRETCHED", "TENSION_BUILDING", "NEUTRAL", "DATA_GAP"),
            ("NUMERATOR_EXPENSIVE", "DENOMINATOR_CHEAP", "BOTH", "UNKNOWN"))}
        assert aui._compute_ratio_summary(mixed, keys, margins) == _main_summary(mixed, keys, margins)


# ── Kortet ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("key", PGM_KEYS)
def test_card_badge_is_never_green_and_states_the_context(key):
    words = {"buy": "sträckt", "tension": "spänning", "sell": "dyr", "neutral": "neutral"}
    for pct in np.round(np.arange(0.0, 100.001, 0.1), 1):
        for driver in DRIVERS:
            r = _ratio(key, status="NEUTRAL", percentile=float(pct), driver=driver)
            badge, color, explanation = aui._ratio_verdict(r)
            assert badge == f"⚪ KONTEXT · {words[aui._band_zone(r)]}" and color == "#607080", (pct, driver)
            assert "KÖP-STÖD" not in badge and "SÄLJ-STÖD" not in badge and "🟢" not in badge
            assert explanation == _main_ratio_verdict(r)[2]                   # förklaringen som förut
            assert aui._driver_verdict_text(r) == ("⚪", LINES[key])
    assert aui._ratio_verdict(_ratio(key, percentile=94.0))[0] == "⚪ KONTEXT · sträckt"
    assert aui._ratio_verdict(_ratio(key, percentile=85.0))[0] == "⚪ KONTEXT · spänning"
    assert aui._ratio_verdict(_ratio(key, percentile=50.0))[0] == "⚪ KONTEXT · neutral"
    assert aui._ratio_verdict(_ratio(key, percentile=5.0))[0] == "⚪ KONTEXT · dyr"


@pytest.mark.parametrize("key", OTHER_KEYS)
def test_other_cards_keep_their_badges(key):
    for pct in np.round(np.arange(0.0, 100.001, 0.5), 1):
        r = _ratio(key, status="NEUTRAL", percentile=float(pct))
        assert aui._ratio_verdict(r) == _main_ratio_verdict(r)


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


@pytest.mark.parametrize("key,den", [("gold_platinum", "PL=F"), ("gold_palladium", "PA=F")])
@pytest.mark.parametrize("pct,word", [(94.0, "sträckt"), (85.0, "spänning"), (50.0, "neutral"), (5.0, "dyr")])
def test_apptest_pgm_card_shows_the_state_the_legs_and_the_context_line(key, den, pct, word):
    r = _ratio(key, status="NEUTRAL", percentile=pct, driver="DENOMINATOR_CHEAP", numerator_pctile=55.0,
               denominator_pctile=9.0, current=2.5, median=1.6)
    html = _render_card(r, {"GDX": _margin("GDX", 5.0)})
    assert aui._css_badge(f"⚪ KONTEXT · {word}", "#607080") in html
    for part in ("Täljare (GC=F):", f"Nämnare ({den}):", "55:e pct — mitten av sitt eget prisspann sedan 2016",
                 "9:e pct — nära botten av sitt eget prisspann sedan 2016", f"I GC=F: 1 {den} = 0,400 GC=F",
                 f'<div style="color:#607080;font-size:0.78rem;">⚪ {LINES[key]}</div>'):
        assert part in html, part
    assert "KÖP-STÖD" not in html and "SÄLJ-STÖD" not in html and "🟢" not in html and "#1aaa5a" not in html
    assert "panelens regel räknar" not in html and "marginalmätaren" not in html


def test_apptest_panel_banner_counts_the_pgm_bands_as_context(monkeypatch):
    from streamlit.testing.v1 import AppTest
    ratios = {k: _ratio(k, driver="DENOMINATOR_CHEAP", denominator_pctile=9.0) for k in PGM_KEYS}

    def app():
        import streamlit as st
        from alpha_regime import ui as u
        u._render_contrarian_mode(st.session_state["_r"], st.session_state.get("_override"))
    monkeypatch.setattr(aui, "_cached_deep_rows", lambda: ([], ""))
    monkeypatch.setattr(mm, "compute", lambda miner="GDX": pytest.fail("mätaren lästes"))
    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["_r"] = _regime(ratios, list(PGM_KEYS), {})
    at.session_state["_override"] = None
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Gummisnoddar: 2 KONTEXT</div>" in html and "KÖP-STÖD" not in html


# ── Thor ─────────────────────────────────────────────────────────────────────
def test_thor_subtitle_for_the_pgm_keys():
    from thor import ravara as rv
    for key in PGM_KEYS:
        for status, pct, want in (("RUBBER_BAND_STRETCHED", 93.4, "kontext · P93 · sträckt"),
                                  ("TENSION_BUILDING", 85.0, "kontext · P85 · spänning"),
                                  ("NEUTRAL", 50.0, "kontext · P50 · neutral")):
            for driver in DRIVERS:
                assert rv.context_text(_ratio(key, status=status, percentile=pct, driver=driver)) == want


def test_thor_ravara_rows(monkeypatch):
    from fiat_debasement import ui as fu
    from gold_ratios import ui as gu
    from thor import ravara as rv
    ratios = {"gold_platinum": _ratio("gold_platinum", percentile=93.0, driver="DENOMINATOR_CHEAP"),
              "gold_palladium": _ratio("gold_palladium", status="TENSION_BUILDING", driver="BOTH"),
              "gold_silver": _ratio("gold_silver", driver="DENOMINATOR_CHEAP"),
              "silver_juniors": _ratio("silver_juniors", status="NEUTRAL", driver="MIXED")}
    monkeypatch.setattr(cr, "fetch_all_ratios", lambda: ratios)
    monkeypatch.setattr(gu, "_load", lambda: {})
    monkeypatch.setattr(fu, "gs_ratio", lambda asset_loader=None: (None, []))
    monkeypatch.setattr(mm, "compute", lambda miner="GDX": pytest.fail("mätaren lästes"))
    for theme in ("platina", "palladium"):
        st = _St()
        rv._ratios(st, theme, lambda text: st.md.append(text))
        html = " ".join(st.md)
        assert "kontext · P93 · sträckt" in html and "kontext · P85 · spänning" in html
        assert "denominator cheap" not in html and "both" not in html and "köpstöd ·" not in html
        assert "Gold/Platinum och Gold/Palladium är bara kontext sedan 2026-11-03" in html
    st = _St()
    rv._ratios(st, "silver", lambda text: st.md.append(text))                # silvret som förut
    html = " ".join(st.md)
    assert "P94 · rubber band stretched · denominator cheap" in html and "P50 · neutral · mixed" in html
    assert "kontext" not in html


# ── Why Now ──────────────────────────────────────────────────────────────────
def test_why_now_has_no_band_component_for_platinum_or_palladium():
    from confidence import commodities as com
    from confidence import config as ccfg
    from confidence.why_now import signals_from_sources, why_now
    for status, pts in ccfg.WHY_NOW_RATIO.items():
        ratios = {k: {"status": status} for k in (*PGM_KEYS, "gold_silver", "silver_juniors")}
        for metal in ("platinum", "palladium"):
            w = why_now(com.get(metal), signals_from_sources(com.get(metal), ratio_results=ratios))
            assert "Gummiband" not in w.components and "ratio_status" in w.missing
        w = why_now(com.get("silver"), signals_from_sources(com.get("silver"), ratio_results=ratios))
        assert w.components["Gummiband"] == pts


# ── Regeltexterna och specen ─────────────────────────────────────────────────
def test_rule_docs_list_the_pgm_ratios_as_context_only():
    import strategy_rules as sr
    pb = sr.PLAYBOOKS["contrarian"]
    entry = " ".join(f"{r.text} {r.explanation} {r.panel_guide}" for r in pb.entry)
    cheat = dict(pb.cheatsheet)["Bekräftelse"]
    page = open(os.path.join(ROOT, "ovtlyr", "ui", "rules_page.py"), encoding="utf-8").read()
    page = re.sub(r'"\s*\n\s*"', "", page)                        # sidans strängliteraler sammanfogade
    assert "Gold/Platinum och Gold/Palladium är bara kontext, aldrig bekräftelse (sedan 2026-11-03" in entry
    assert "Gold/Platinum, Gold/Palladium: bara kontext" in cheat
    assert "Gold/Platinum och Gold/Palladium är bara kontext, aldrig bekräftelse (sedan 2026-11-03" in page
    assert "en sträckt kvot följdes av att metallen gick sämre än guld" in entry
    for text in (entry, page):
        assert "Gold/Silver-, Gold/Platinum-" not in text and "t.ex. Gold/Silver, Gold/Platinum" not in text
        assert "T.ex. Gold/Silver" in text or "t.ex. Gold/Silver" in text


def test_spec_records_the_result_and_the_decision_after_the_registered_sections():
    text = open(os.path.join(ROOT, bs.CONTEXT_SPEC), encoding="utf-8").read()
    log, res, dec = (text.index(h) for h in ("## Ändringslogg", "## Resultat (körning 2026-11-03)",
                                             "## Beslut (2026-11-03)"))
    assert log < res < dec
    assert "| 1 | 2026-11-03 | Efter första körningen: avsnitten *Resultat (körning 2026-11-03)*" in text[log:res]
    for w in ("| S0 nuvarande regel | 3 | −16,9 % | 0 % |", "| S1 bara kvoten | 5 (3 hela) | −14,5 % | 0 % |",
              "| S2 kvoten och trenden | 8 | −7,1 % | 25 % |", "| Baslinje | – | −4,9 % | 31 % |",
              "**H1 ej godkänd.**", "**H2 ej godkänd.**", "**H3 ej godkänd.**", "2 371 av 3 729 dagar (64 %)",
              "sträckt utan avbrott sedan 2023", "| S0 nuvarande regel | 0 (1 episod, 2026-11-02) | – |",
              "| S1 bara kvoten | 1 | −36,5 % |", "| S2 kvoten och trenden | 1 | −31,1 % |",
              "| Baslinje | – | +1,3 % |", "**H1, H2 och H3: för få episoder.**", "−44,9 % mot GLD",
              "IMPUY −34,6 % och SBSW −14,0 % mot GLD"):
        assert w in text[res:dec], w
    for w in ("Ägaren valde **A: båda PGM-snoddarna blir bara kontext.**", "inget +1 ACCUMULATE",
              "ingen KÖP-STÖD-räkning", "inga gummibandspoäng i Why Now", "(fall 4)", "(fall 5)",
              "förlorade över 30 % mot GLD", "ett oprövat köpstöd var aktivt just nu", "`CONTEXT_ONLY`"):
        assert w in text[dec:], w
    # förregistreringen är orörd
    assert "Gold/Palladium går genom samma regel som i dag tills ägaren beslutar något annat." in text[:log]
    assert "| **S0 nuvarande regel** |" in text[:log] and "version 1, 2026-11-04" in text[:log]
