"""
Gruvornas köpstöd — ägarens beslut A (2026-11-03) efter backtestet docs/forskning/gruvor_marginal_spec.md.

För GLD/GDX (metal_miners, GDX) och GLD/GDXJ (gold_gdxj, GDXJ) kräver 🟢 KÖP-STÖD i 'Gummisnoddar' och motorns
+1 ACCUMULATE att kvoten är RUBBER_BAND_STRETCHED OCH att gruv-ETF:ns marginalmätare står ≤ 30:e percentilen
(specens S3) — i stället för drivarens prisvillkor. Saknas marginalen → inget köpstöd; en inaktuell AISC-tabell
blockerar inte. Alla andra kvoter beter sig exakt som före beslutet: det prövas tabelldrivet mot en ordagrann
kopia av den gamla koden. En regel (alpha_regime/buy_support.py) för motorn, sammanräkningen, kortet och Thor.
Inget nätverk.
"""
import dataclasses
import itertools
import os

import numpy as np
import pandas as pd
import pytest

from alpha_regime import buy_support as bs
from alpha_regime import commodity_ratios as cr
from alpha_regime import engine as eng
from alpha_regime import miner_margin as mm
from alpha_regime import ui as aui

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVERS = ("DENOMINATOR_CHEAP", "NUMERATOR_EXPENSIVE", "BOTH", "MIXED", "UNKNOWN")
STATUSES = ("RUBBER_BAND_STRETCHED", "TENSION_BUILDING", "NEUTRAL", "DATA_GAP")
PCTS = (3.0, 10.0, 15.0, 50.0, 85.0, 89.9, 90.0, 97.0)
GOLD = (("metal_miners", "GDX", "Gold Miners"), ("gold_gdxj", "GDXJ", "Junior Miners"))
GOLD_KEYS = tuple(k for k, _m, _b in GOLD)
OTHER_KEYS = tuple(s.key for s in cr._RATIO_SPECS if s.key not in GOLD_KEYS)
NEUTRAL_LINE = "Snodden är inte spänd — benens prisläge förklarar inget köpläge."


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear = getattr(mm._compute_cached, "clear", lambda: None)
    clear()
    yield
    clear()


def _ratio(key, status="RUBBER_BAND_STRETCHED", percentile=None, driver="NUMERATOR_EXPENSIVE", **kw):
    spec = next(s for s in cr._RATIO_SPECS if s.key == key)
    if percentile is None:
        percentile = {"RUBBER_BAND_STRETCHED": 94.0, "TENSION_BUILDING": 85.0, "NEUTRAL": 50.0, "DATA_GAP": 0.0}[status]
        if spec.cheap_direction == "low" and status != "DATA_GAP":
            percentile = 100.0 - percentile
    base = dict(key=key, label=spec.label, denominator_label=spec.denominator_label,
                cheap_direction=spec.cheap_direction, current=8.0, percentile=percentile, zscore=1.0, status=status,
                numerator_label=spec.primary_num, denominator_leg_label=spec.primary_den, numerator_pctile=95.0,
                denominator_pctile=50.0, driver=driver, history_start="2016-11-03", history_years=10.0, median=6.25)
    base.update(kw)
    return cr.RatioResult(**base)


def _margin(miner="GDX", pct=22.0, **kw):
    base = dict(miner=miner, status="OK", current=0.031, percentile=pct, median=0.045, miner_price=75.0,
                gold_price=4010.0, aisc=1785.0, aisc_period="2026Q1", aisc_source="World Gold Council/Metals Focus",
                margin=2225.0, as_of="2026-10-30", history_start="2016-11-03", history_years=10.0,
                newest_period="2026Q1", sparkline_dates=["2026-10-25", "2026-11-01"], sparkline_values=[0.032, 0.031])
    base.update(kw)
    return mm.MarginResult(**base)


def _closes(n=400, gold=2000.0, miner=50.0, last_gold=None):
    idx = pd.bdate_range(end="2026-10-30", periods=n)
    g = np.full(n, gold)
    if last_gold is not None:
        g[-1] = last_gold
    return pd.Series(np.linspace(miner * 0.8, miner, n), index=idx), pd.Series(g, index=idx)


_ROWS = [mm.AiscRow("2010", 1000.0, "2010-01-01", "MF_A", "Testkälla", "test", "https://example.org")]


def _gaps(miner):
    """De riktiga DATA_GAP-fallen ur build_result: inget guldpris, ingen gruvkurs, marginal ≤ 50 USD/uns."""
    close, gold = _closes()
    _c, thin = _closes(last_gold=1040.0)                     # 1 040 − 1 000 = 40 USD/uns → M maskas
    out = {"inget guldpris": mm.build_result(miner, close, None, rows=_ROWS),
           "ingen gruvkurs": mm.build_result(miner, None, gold, rows=_ROWS),
           "marginal ≤ 50": mm.build_result(miner, close, thin, rows=_ROWS),
           "beräkningen misslyckades": mm.MarginResult(miner=miner, error="beräkningen misslyckades")}
    assert all(r.status == "DATA_GAP" and r.percentile is None for r in out.values())
    assert "Marginalen är bara 40 USD/uns" in out["marginal ≤ 50"].error
    return out


# ── Motorn ───────────────────────────────────────────────────────────────────
def _run_engine(monkeypatch, branch, ratios, margins):
    idx = pd.bdate_range(end="2026-10-30", periods=300)
    c = np.linspace(100.0, 80.0, 300)
    df = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e5}, index=idx)
    calls = []

    def compute(miner="GDX"):
        calls.append(miner)
        return margins.get(miner) or mm.MarginResult(miner=miner, error="saknas i testet")
    monkeypatch.setattr(eng, "_fetch_price", lambda t, period="2y": df)
    monkeypatch.setattr(eng, "_fetch_fundamentals", lambda t: {"branch_name": branch})
    monkeypatch.setattr(eng, "_fetch_ratios", lambda: ratios)
    monkeypatch.setattr(eng, "_fetch_context_gauges", lambda: {})
    monkeypatch.setattr(mm, "compute", compute)
    r = eng.run_regime_analysis("NEM", mode="contrarian", market_ticker="SPY", with_sentiment=False)
    return r, [x for x in r.contrarian.rationale if x.startswith("Rubber Band:")], calls


def _exposure_ratios(branch, **override):
    keys = cr.EXPOSURE_TO_RATIO[cr.detect_exposure(branch)]
    ratios = {k: _ratio(k, status="NEUTRAL", driver="MIXED") for k in keys}
    ratios.update(override)
    return ratios


@pytest.mark.parametrize("key,miner,branch", GOLD)
def test_engine_plus_one_when_stretched_and_margin_22_even_with_numerator_expensive(monkeypatch, key, miner, branch):
    ratios = _exposure_ratios(branch, **{key: _ratio(key, driver="NUMERATOR_EXPENSIVE")})
    r, rb, calls = _run_engine(monkeypatch, branch, ratios, {miner: _margin(miner, 22.0)})
    assert rb == [f"Rubber Band: GLD/{miner} at 94th percentile — {miner} cheap vs margin (22nd pct ≤ 30)"]
    assert r.miner_margins[miner].percentile == 22.0 and calls.count(miner) == 1


@pytest.mark.parametrize("key,miner,branch", GOLD)
@pytest.mark.parametrize("driver", ["DENOMINATOR_CHEAP", "BOTH"])
def test_engine_no_plus_one_with_a_cheap_driver_but_margin_45(monkeypatch, key, miner, branch, driver):
    ratios = _exposure_ratios(branch, **{key: _ratio(key, driver=driver, denominator_pctile=9.0)})
    _r, rb, _calls = _run_engine(monkeypatch, branch, ratios, {miner: _margin(miner, 45.0)})
    assert rb == []


@pytest.mark.parametrize("key,miner,branch", GOLD)
@pytest.mark.parametrize("gap", ["inget guldpris", "ingen gruvkurs", "marginal ≤ 50", "beräkningen misslyckades"])
def test_engine_no_plus_one_when_the_margin_is_a_data_gap(monkeypatch, key, miner, branch, gap):
    ratios = _exposure_ratios(branch, **{key: _ratio(key, driver="DENOMINATOR_CHEAP", denominator_pctile=9.0)})
    r, rb, _calls = _run_engine(monkeypatch, branch, ratios, {miner: _gaps(miner)[gap]})
    assert rb == [] and r.miner_margins[miner].status == "DATA_GAP"


def test_engine_gold_gdxj_reads_the_gdxj_margin_not_gdx(monkeypatch):
    ratios = _exposure_ratios("Gold Miners", gold_gdxj=_ratio("gold_gdxj", driver="NUMERATOR_EXPENSIVE"))
    _r, rb, calls = _run_engine(monkeypatch, "Gold Miners", ratios,
                                {"GDX": _margin("GDX", 5.0), "GDXJ": _margin("GDXJ", 45.0)})
    assert rb == [] and calls == ["GDX", "GDXJ"]
    _r, rb, _calls = _run_engine(monkeypatch, "Gold Miners", ratios,
                                 {"GDX": _margin("GDX", 95.0), "GDXJ": _margin("GDXJ", 30.0)})
    assert rb == ["Rubber Band: GLD/GDXJ at 94th percentile — GDXJ cheap vs margin (30th pct ≤ 30)"]


def test_engine_only_computes_the_gauges_it_needs(monkeypatch):
    _r, _rb, calls = _run_engine(monkeypatch, "Junior Miners", _exposure_ratios("Junior Miners"), {})
    assert calls == ["GDXJ"]
    r, _rb, calls = _run_engine(monkeypatch, "Silver", _exposure_ratios("Silver"), {})
    assert calls == [] and r.miner_margins == {}


def test_engine_keeps_max_one_confirmation(monkeypatch):
    ratios = _exposure_ratios("Gold Miners", metal_miners=_ratio("metal_miners"),
                              gdxj_gdx=_ratio("gdxj_gdx", driver="DENOMINATOR_CHEAP"),
                              gold_gdxj=_ratio("gold_gdxj"))
    _r, rb, _calls = _run_engine(monkeypatch, "Gold Miners", ratios,
                                 {"GDX": _margin("GDX", 10.0), "GDXJ": _margin("GDXJ", 10.0)})
    assert rb == ["Rubber Band: GLD/GDX at 94th percentile — GDX cheap vs margin (10th pct ≤ 30)"]
    # GLD/GDX utan köpstöd → nästa kvot i listan (GDXJ/GDX, drivaren som förut) får sin rad
    _r, rb, _calls = _run_engine(monkeypatch, "Gold Miners", ratios,
                                 {"GDX": _margin("GDX", 60.0), "GDXJ": _margin("GDXJ", 10.0)})
    assert rb == ["Rubber Band: GDXJ / GDX (Juniors/Majors) at 94th percentile — Junior Miners historically "
                  "stretched cheap"]


def test_stale_aisc_table_does_not_block(monkeypatch):
    stale = _margin("GDX", 22.0, stale=True, newest_period="2025Q4")
    assert bs.evaluate("metal_miners", _ratio("metal_miners"), {"GDX": stale}).supported
    ratios = _exposure_ratios("Gold Miners", metal_miners=_ratio("metal_miners"))
    _r, rb, _calls = _run_engine(monkeypatch, "Gold Miners", ratios, {"GDX": stale})
    assert rb == ["Rubber Band: GLD/GDX at 94th percentile — GDX cheap vs margin (22nd pct ≤ 30)"]
    assert aui._compute_ratio_summary(ratios, list(ratios), {"GDX": stale}) == "Gummisnoddar: 1 KÖP-STÖD, 3 NEUTRAL"
    # det verkliga flaggfallet: gammal tabell i build_result → stale, men OK och räknas
    close, gold = _closes()
    res = mm.build_result("GDX", close, gold, today="2026-11-03", rows=_ROWS)
    assert res.stale and res.status == "OK"
    assert bs.margin_percentile(res) == res.percentile


# ── Övriga kvoter: exakt som före 2026-11-03 ─────────────────────────────────
def _old_engine_confirms(ratio):
    """Ordagrant villkoret i alpha_regime/engine.py före 2026-11-03."""
    return bool(ratio and ratio.status == "RUBBER_BAND_STRETCHED"
                and getattr(ratio, "driver", "UNKNOWN") in ("DENOMINATOR_CHEAP", "BOTH"))


def _old_summary(commodity_ratios, keys):
    """Ordagrant alpha_regime/ui._compute_ratio_summary före 2026-11-03."""
    counts = {"KÖP-STÖD": 0, "SÄLJ-STÖD": 0, "SPÄNNING": 0, "NEUTRAL": 0, "DATA_GAP": 0}
    for key in keys:
        r = commodity_ratios.get(key)
        if r is None or r.status == "DATA_GAP":
            counts["DATA_GAP"] += 1
            continue
        badge, _, _ = aui._ratio_verdict(r)
        if "KÖP-STÖD" in badge:
            if getattr(r, "driver", "UNKNOWN") in ("DENOMINATOR_CHEAP", "BOTH", "UNKNOWN"):
                counts["KÖP-STÖD"] += 1
            else:
                counts["NEUTRAL"] += 1
        elif "SÄLJ-STÖD" in badge:
            counts["SÄLJ-STÖD"] += 1
        elif "SPÄNNING" in badge:
            counts["SPÄNNING"] += 1
        else:
            counts["NEUTRAL"] += 1
    parts = [f"{counts[k]} {k}" for k in ("KÖP-STÖD", "SÄLJ-STÖD", "SPÄNNING", "NEUTRAL") if counts[k] > 0]
    if counts["DATA_GAP"] > 0:
        parts.append(f"{counts['DATA_GAP']} DATA_GAP")
    return ("Gummisnoddar: " + ", ".join(parts)) if parts else ""


_MARGIN_SETS = (None, {}, {"GDX": _margin("GDX", 5.0), "GDXJ": _margin("GDXJ", 5.0)},
                {"GDX": _margin("GDX", 95.0), "GDXJ": _margin("GDXJ", 95.0)},
                {"GDX": mm.MarginResult(miner="GDX", error="x"), "GDXJ": mm.MarginResult(miner="GDXJ", error="x")})


@pytest.mark.parametrize("key", OTHER_KEYS)
def test_other_ratios_keep_the_old_rule_for_every_driver_status_and_percentile(key):
    """Alla kvoter utom GLD/GDX och GLD/GDXJ: motorns villkor och sammanräkningen ger exakt samma utfall som
    koden före 2026-11-03, för varje drivare × status × percentil — vad marginalmätaren än säger."""
    assert key in ("gold_silver", "gold_oil", "silver_miners", "copper_gold", "gdxj_gdx", "silver_juniors", "gdx_spy")
    for status, driver, pct in itertools.product(STATUSES, DRIVERS, PCTS):
        r = _ratio(key, status=status, percentile=pct, driver=driver)
        old_engine, old_summary = _old_engine_confirms(r), _old_summary({key: r}, [key])
        for margins in _MARGIN_SETS:
            sup = bs.evaluate(key, r, margins)
            assert sup.supported == old_engine, (status, driver, pct)
            assert sup.details["rule"] == "driver" and "margin_pct" not in sup.details
            assert aui._compute_ratio_summary({key: r}, [key], margins) == old_summary, (status, driver, pct)
    assert bs.evaluate(key, None).supported is False and _old_engine_confirms(None) is False


@pytest.mark.parametrize("branch", ["Silver", "Oil", "Copper", "Junior Miners", "Gold Miners"])
def test_engine_for_other_ratios_matches_the_old_code(monkeypatch, branch):
    """Hela motorn: guldkvoterna NEUTRAL, övriga kvoter i exponeringen i varje status × drivare — samma
    'Rubber Band'-rad (eller ingen) som koden före 2026-11-03 gav."""
    keys = cr.EXPOSURE_TO_RATIO[cr.detect_exposure(branch)]
    margins = {"GDX": _margin("GDX", 5.0), "GDXJ": _margin("GDXJ", 5.0)}
    for status, driver in itertools.product(STATUSES, DRIVERS):
        ratios = {k: (_ratio(k, status="NEUTRAL", driver="BOTH") if k in GOLD_KEYS
                      else _ratio(k, status=status, driver=driver)) for k in keys}
        _r, rb, _calls = _run_engine(monkeypatch, branch, ratios, margins)
        first = next((ratios[k] for k in keys if _old_engine_confirms(ratios[k])), None)
        old = [] if first is None else [f"Rubber Band: {first.label} at {first.percentile:.0f}th percentile"
                                        f" — {first.denominator_label} historically stretched cheap"]
        assert rb == old, (branch, status, driver)


def test_custom_ratio_over_gold_legs_is_not_a_gold_key():
    r = dataclasses.replace(_ratio("metal_miners", driver="DENOMINATOR_CHEAP", denominator_pctile=9.0),
                            key="custom_GLD_GDX")
    assert not bs.uses_margin(r.key)
    assert bs.evaluate(r.key, r, {}).supported
    assert aui._driver_verdict_text(r, counted=False)[1].endswith("egen kvot — räknas inte i panelens bekräftelse.")


# ── Den delade regeln ────────────────────────────────────────────────────────
def test_rule_constants_and_the_spec():
    assert bs.MARGIN_RULE == {"metal_miners": "GDX", "gold_gdxj": "GDXJ"} and bs.MARGIN_MAX_PCT == 30.0
    assert bs.DECIDED == "2026-11-03" and os.path.exists(os.path.join(ROOT, bs.SPEC))
    assert bs.margin_miners(["gdx_spy", "gold_gdxj", "metal_miners"]) == ["GDX", "GDXJ"]
    assert bs.margin_miners(None) == [] and bs.margin_miners(["gold_silver"]) == []


@pytest.mark.parametrize("key,miner,_b", GOLD)
@pytest.mark.parametrize("pct,ok,reason", [(0.0, True, bs.MARGIN_CHEAP), (22.0, True, bs.MARGIN_CHEAP),
                                           (30.0, True, bs.MARGIN_CHEAP), (30.1, False, bs.MARGIN_NOT_CHEAP),
                                           (45.0, False, bs.MARGIN_NOT_CHEAP), (100.0, False, bs.MARGIN_NOT_CHEAP)])
def test_threshold_is_at_most_30_for_every_driver(key, miner, _b, pct, ok, reason):
    for driver in DRIVERS:
        sup = bs.evaluate(key, _ratio(key, driver=driver), {miner: _margin(miner, pct)})
        assert (sup.supported, sup.reason) == (ok, reason)
        assert sup.details["miner"] == miner and sup.details["margin_pct"] == pct


@pytest.mark.parametrize("key,miner,_b", GOLD)
def test_gold_keys_need_the_stretched_ratio_and_their_own_miner(key, miner, _b):
    other = "GDXJ" if miner == "GDX" else "GDX"
    for status in ("TENSION_BUILDING", "NEUTRAL"):
        sup = bs.evaluate(key, _ratio(key, status=status, driver="BOTH"), {miner: _margin(miner, 5.0)})
        assert (sup.supported, sup.reason) == (False, bs.NOT_STRETCHED)
    assert bs.evaluate(key, _ratio(key, status="DATA_GAP"), {miner: _margin(miner, 5.0)}).reason == bs.NO_DATA
    assert bs.evaluate(key, None, {miner: _margin(miner, 5.0)}).reason == bs.NO_DATA
    sup = bs.evaluate(key, _ratio(key), {other: _margin(other, 5.0)})
    assert (sup.supported, sup.reason) == (False, bs.MARGIN_MISSING)
    for margins in (None, {}, {miner: None}, {miner: _margin(miner, float("nan"))}, {miner: _margin(miner, None)}):
        assert bs.evaluate(key, _ratio(key), margins).reason == bs.MARGIN_MISSING


@pytest.mark.parametrize("p,limit,at_most,text", [(22.0, 30.0, True, "22"), (30.0, 30.0, True, "30"),
                                                  (29.6, 30.0, True, "30"), (30.1, 30.0, True, "30,1"),
                                                  (30.4, 30.0, True, "30,4"), (89.9, 90.0, False, "89,9"),
                                                  (89.96, 90.0, False, "89,9"), (90.0, 90.0, False, "90"),
                                                  (94.4, 90.0, False, "94")])
def test_pct_text_never_contradicts_the_limit(p, limit, at_most, text):
    assert bs.pct_text(p, limit, at_most) == text


# ── Sammanräkningen ──────────────────────────────────────────────────────────
def test_summary_counts_gold_keys_by_the_margin():
    ratios = {"metal_miners": _ratio("metal_miners", driver="NUMERATOR_EXPENSIVE"),
              "gdxj_gdx": _ratio("gdxj_gdx", driver="DENOMINATOR_CHEAP"),
              "gold_gdxj": _ratio("gold_gdxj", driver="DENOMINATOR_CHEAP"),
              "gdx_spy": _ratio("gdx_spy", status="NEUTRAL", driver="MIXED")}
    keys = list(ratios)
    assert aui._compute_ratio_summary(ratios, keys, {"GDX": _margin("GDX", 22.0), "GDXJ": _margin("GDXJ", 45.0)}) == (
        "Gummisnoddar: 2 KÖP-STÖD, 2 NEUTRAL")
    assert aui._compute_ratio_summary(ratios, keys, {"GDX": _gaps("GDX")["inget guldpris"],
                                                     "GDXJ": _margin("GDXJ", 30.0)}) == (
        "Gummisnoddar: 2 KÖP-STÖD, 2 NEUTRAL")
    assert aui._compute_ratio_summary(ratios, keys, {"GDX": _margin("GDX", 22.0), "GDXJ": _margin("GDXJ", 5.0)}) == (
        "Gummisnoddar: 3 KÖP-STÖD, 1 NEUTRAL")
    assert aui._compute_ratio_summary(ratios, keys, None) == "Gummisnoddar: 1 KÖP-STÖD, 3 NEUTRAL"
    # spänning, säljzon och DATA_GAP räknas som förut
    other = {"metal_miners": _ratio("metal_miners", status="TENSION_BUILDING"),
             "gold_gdxj": _ratio("gold_gdxj", status="NEUTRAL", percentile=5.0),
             "gdxj_gdx": _ratio("gdxj_gdx", status="DATA_GAP")}
    assert aui._compute_ratio_summary(other, list(other), {"GDX": _margin("GDX", 5.0)}) == (
        "Gummisnoddar: 1 SÄLJ-STÖD, 1 SPÄNNING, 1 DATA_GAP")


def test_summary_uses_the_badge_gate_as_before_at_the_rounding_edge(monkeypatch):
    """Lagrad 90,0 men råa 89,98 → status TENSION_BUILDING: märket och sammanräkningen säger KÖP-STÖD när
    marginalen är ≤ 30 (märkesgrinden som förut), motorns +1 läser statusen och ger ingen rad."""
    r = _ratio("metal_miners", status="TENSION_BUILDING", percentile=90.0, driver="NUMERATOR_EXPENSIVE")
    margins = {"GDX": _margin("GDX", 22.0)}
    assert aui._ratio_verdict(r)[0] == "🟢 KÖP-STÖD"
    assert aui._compute_ratio_summary({"metal_miners": r}, ["metal_miners"], margins) == "Gummisnoddar: 1 KÖP-STÖD"
    assert aui._driver_verdict_text(r, margins=margins)[0] == "🟢"
    assert not bs.evaluate("metal_miners", r, margins).supported


# ── Kortets texter ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("key,miner,_b", GOLD)
def test_card_verdict_line_describes_the_margin_rule(key, miner, _b):
    for driver in DRIVERS:
        stretched = _ratio(key, driver=driver, numerator_pctile=95.0, denominator_pctile=9.0)
        ok = (f"Kvoten vid 94:e percentilen och marginalmätaren för {miner} vid 22:a (≤ 30:e) — panelens regel "
              f"räknar det som köpstöd i {miner}.")
        assert aui._driver_verdict_text(stretched, margins={miner: _margin(miner, 22.0)}) == ("🟢", ok)
        dear = (f"Kvoten vid 94:e percentilen, men marginalmätaren för {miner} står vid 45:e (köpstöd kräver "
                f"≤ 30:e) — inget köpstöd i {miner}.")
        assert aui._driver_verdict_text(stretched, margins={miner: _margin(miner, 45.0)}) == ("⚠️", dear)
        for margins in ({miner: _gaps(miner)["marginal ≤ 50"]}, {}, None):
            assert aui._driver_verdict_text(stretched, margins=margins) == (
                "⚪", f"Kvoten vid 94:e percentilen, men marginalen för {miner} saknas — inget köpstöd i {miner}.")
        tension = _ratio(key, status="TENSION_BUILDING", driver=driver, denominator_pctile=9.0)
        needs = (f"Köpstöd i {miner} kräver kvoten ≥ 90:e percentilen och marginalmätaren ≤ 30:e — nu kvoten vid "
                 f"85:e och marginalmätaren vid 22:a.")
        assert aui._driver_verdict_text(tension, margins={miner: _margin(miner, 22.0)}) == ("⚡", needs)
        assert aui._driver_verdict_text(tension, margins={})[1].endswith("— nu kvoten vid 85:e, och marginalen saknas.")
        neutral = _ratio(key, status="NEUTRAL", driver=driver)
        assert aui._driver_verdict_text(neutral, margins={miner: _margin(miner, 5.0)}) == ("⚪", NEUTRAL_LINE)
        for text in (aui._driver_verdict_text(r, margins={miner: _margin(miner, 22.0)})[1]
                     for r in (stretched, tension)):
            assert "nära botten" not in text and "nära toppen" not in text and "Blandad" not in text


def test_card_ordinals_never_contradict_the_limits():
    assert "vid 30,4:e (köpstöd kräver ≤ 30:e)" in aui._driver_verdict_text(
        _ratio("metal_miners"), margins={"GDX": _margin("GDX", 30.4)})[1]
    assert "nu kvoten vid 89,9:e och marginalmätaren vid 22:a." in aui._driver_verdict_text(
        _ratio("metal_miners", status="TENSION_BUILDING", percentile=89.9), margins={"GDX": _margin("GDX", 22.0)})[1]


def _card_app():
    import streamlit as st
    from alpha_regime import ui as u
    u._render_single_ratio(st.session_state["_ratio"], idx=0, margins=st.session_state.get("_margins"))


def _render_card(ratio, margins):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_card_app, default_timeout=60)
    at.session_state["_ratio"] = ratio
    at.session_state["_margins"] = margins
    at.run()
    assert not at.exception, at.exception
    return " ".join(m.value for m in at.markdown)


def test_apptest_gold_card_keeps_the_legs_as_information_and_states_the_rule():
    r = _ratio("metal_miners", driver="DENOMINATOR_CHEAP", numerator_pctile=55.0, denominator_pctile=9.0)
    html = _render_card(r, {"GDX": _margin("GDX", 45.0)})
    assert "🟢 KÖP-STÖD" in html                                      # märket: kvoten är sträckt
    assert "Täljare (GLD):" in html and "9:e pct — nära botten av sitt eget prisspann sedan 2016" in html
    assert ("Benens prisläge är bara information — sedan 2026-11-03 avgör marginalmätaren köpstödet för den här "
            "kvoten, inte GDX:s eget prisläge.") in html
    assert "⚠️ Kvoten vid 94:e percentilen, men marginalmätaren för GDX står vid 45:e" in html and "#e8a020" in html
    assert "panelens regel räknar det som köpstöd" not in html
    html = _render_card(r, {"GDX": _margin("GDX", 22.0)})
    assert "panelens regel räknar det som köpstöd i GDX." in html


def test_apptest_non_gold_card_has_no_margin_note():
    html = _render_card(_ratio("gdxj_gdx", driver="DENOMINATOR_CHEAP", denominator_pctile=9.0),
                        {"GDX": _margin("GDX", 45.0)})
    assert "Benens prisläge är bara information" not in html and "marginalmätaren" not in html
    assert "GDX står nära botten av sitt eget prisspann (9:e pct) — panelens regel räknar det som köpstöd i GDX." in html


# ── Panelen återanvänder motorns mätare ──────────────────────────────────────
def _contrarian_app():
    import streamlit as st
    from alpha_regime import ui as u
    u._render_contrarian_mode(st.session_state["_r"], st.session_state.get("_override"))


def _regime(ratios, keys, margins):
    from alpha_regime.contrarian_signals import ContrairianStageResult
    from alpha_regime.engine import RegimeResult
    return RegimeResult(mode="contrarian", ticker="NEM", market_ticker="SPY", market_phase="DISBELIEF",
                        market_confidence=60.0, price_vs_ma200=-12.0, trend_phase="Bearish",
                        contrarian=ContrairianStageResult("HOLD", "HOLD", "#607080", rationale=["Kontrollrad"]),
                        commodity_ratios=ratios, detected_exposure=list(keys), miner_margins=margins)


def _run_contrarian(monkeypatch, r, override=None):
    from streamlit.testing.v1 import AppTest
    calls = []

    def compute(miner="GDX"):
        calls.append(miner)
        return _margin(miner, 5.0)
    monkeypatch.setattr(aui, "_cached_deep_rows", lambda: ([], ""))
    monkeypatch.setattr(mm, "compute", compute)
    at = AppTest.from_function(_contrarian_app, default_timeout=60)
    at.session_state["_r"] = r
    at.session_state["_override"] = override
    at.run()
    assert not at.exception, at.exception
    return at, [m.value for m in at.markdown], calls


def test_apptest_panel_reuses_the_engines_margins(monkeypatch):
    keys = cr.EXPOSURE_TO_RATIO["gold_miner"]
    ratios = _exposure_ratios("Gold Miners", metal_miners=_ratio("metal_miners"), gold_gdxj=_ratio("gold_gdxj"))
    stale = _margin("GDXJ", 22.0, stale=True)
    r = _regime(ratios, keys, {"GDX": _margin("GDX", 45.0), "GDXJ": stale})
    at, md, calls = _run_contrarian(monkeypatch, r)
    assert calls == []                                               # inga nya beräkningar
    assert any("Gummisnoddar: 1 KÖP-STÖD, 3 NEUTRAL</div>" in v for v in md)   # GLD/GDXJ ja (22), GLD/GDX nej (45)
    assert any("GDX kostar" in v and "45:e percentilen" in v for v in md)       # mätarkortet visar samma objekt
    assert [w.value for w in at.warning] == [aui._MM_STALE_TEXT]                 # inaktuell: varnar, blockerar inte
    caption = " ".join(c.value for c in at.caption)
    assert "är marginalen en del av köpstödet för GLD/GDX och GLD/GDXJ" in caption


def test_apptest_panel_computes_a_hand_picked_gold_ratio_once(monkeypatch):
    ratios = {**_exposure_ratios("Silver"), **_exposure_ratios("Junior Miners", gold_gdxj=_ratio("gold_gdxj"))}
    r = _regime(ratios, cr.EXPOSURE_TO_RATIO["silver"], {})
    _at, md, calls = _run_contrarian(monkeypatch, r, override=cr.EXPOSURE_TO_RATIO["junior_miner"])
    assert calls == ["GDXJ"]                                          # en gång — sammanräkning, kort och mätare delar
    assert any("Gummisnoddar: 1 KÖP-STÖD, 1 NEUTRAL</div>" in v for v in md)


# ── Thor ─────────────────────────────────────────────────────────────────────
class _St:
    def __init__(self):
        self.md = []

    def markdown(self, text, **_kw):
        self.md.append(text)


def test_thor_subtitle_shows_the_rule_for_gold_keys_and_the_driver_for_the_rest(monkeypatch):
    from fiat_debasement import ui as fu
    from thor import ravara as rv
    ratios = {"metal_miners": _ratio("metal_miners", driver="NUMERATOR_EXPENSIVE"),
              "gdxj_gdx": _ratio("gdxj_gdx", driver="NUMERATOR_EXPENSIVE"),
              "gold_gdxj": _ratio("gold_gdxj", driver="DENOMINATOR_CHEAP"),
              "gdx_spy": _ratio("gdx_spy", status="NEUTRAL", driver="MIXED")}
    calls = []
    gauges = {"GDX": _margin("GDX", 22.0), "GDXJ": _margin("GDXJ", 45.0)}

    def compute(miner="GDX"):
        calls.append(miner)
        return gauges[miner]
    monkeypatch.setattr(cr, "fetch_all_ratios", lambda: ratios)
    monkeypatch.setattr(fu, "gs_ratio", lambda asset_loader=None: (None, []))
    monkeypatch.setattr(mm, "compute", compute)
    st = _St()
    rv._ratios(st, "guld", lambda text: st.md.append(text))
    html = " ".join(st.md)
    assert calls == ["GDX", "GDXJ"]
    assert "P94 · rubber band stretched · köpstöd · marginal P22 ≤ 30" in html
    assert "P94 · rubber band stretched · inget köpstöd · marginal P45 &gt; 30" in html
    assert "P94 · rubber band stretched · numerator expensive" in html               # GDXJ/GDX: som förut
    assert "P50 · neutral · mixed" in html                                           # GDX/SPY: som förut
    assert "denominator cheap" not in html                                           # guldkvotens drivare visas inte
    assert "För GLD/GDX och GLD/GDXJ visar raden panelens köpstöd sedan 2026-11-03" in html


def test_thor_subtitle_texts_for_every_outcome():
    from thor import ravara as rv
    r = _ratio("metal_miners")
    assert rv.margin_rule_text(bs.evaluate("metal_miners", r, {"GDX": _margin("GDX", 30.4)})) == (
        "inget köpstöd · marginal P30,4 > 30")
    assert rv.margin_rule_text(bs.evaluate("metal_miners", r, {})) == "inget köpstöd · marginalen saknas"
    tension = _ratio("metal_miners", status="TENSION_BUILDING")
    assert rv.margin_rule_text(bs.evaluate("metal_miners", tension, {"GDX": _margin("GDX", 22.0)})) == (
        "inget köpstöd · kvoten inte fullt sträckt · marginal P22")


def test_thor_without_a_gold_ratio_never_reads_the_gauge(monkeypatch):
    from thor import ravara as rv
    monkeypatch.setattr(cr, "fetch_all_ratios", lambda: {"copper_gold": _ratio("copper_gold", driver="BOTH")})
    monkeypatch.setattr(mm, "compute", lambda miner="GDX": pytest.fail("mätaren lästes"))
    from gold_ratios import ui as gu
    monkeypatch.setattr(gu, "_load", lambda: {})
    st = _St()
    rv._ratios(st, "koppar", lambda text: st.md.append(text))
    html = " ".join(st.md)
    assert "P6 · rubber band stretched · both" in html and "köpstöd" not in html


# ── Regeldokumenten ──────────────────────────────────────────────────────────
def test_rule_docs_state_the_confirmation_precisely():
    import strategy_rules as sr
    pb = sr.PLAYBOOKS["contrarian"]
    entry = " ".join(f"{r.text} {r.explanation} {r.panel_guide}" for r in pb.entry)
    cheat = dict(pb.cheatsheet)["Bekräftelse"]
    for text in (entry, cheat):
        assert "≥ 90:e percentilen" in text and "≤ 30:e" in text
    assert "GLD/GDX, GLD/GDXJ" in entry and "nära botten av sitt eget prisspann" in entry
    assert "marginalmätaren ≤ 30:e percentilen" in entry
    page = open(os.path.join(ROOT, "ovtlyr", "ui", "rules_page.py"), encoding="utf-8").read()
    assert "guldgruvorna (GLD/GDX, GLD/GDXJ) ska gruvorna dessutom vara billiga mot marginalen" in page
    assert "(marginalmätaren ≤ 30:e percentilen, sedan 2026-11-03)" in page
    assert "benet stå nära botten av sitt eget prisspann" in page


def test_spec_records_the_result_and_the_decision_after_the_registered_sections():
    text = open(os.path.join(ROOT, bs.SPEC), encoding="utf-8").read()
    log, res, dec = (text.index(h) for h in ("## Ändringslogg", "## Resultat (körning 2026-11-03)",
                                             "## Beslut (2026-11-03)"))
    assert log < res < dec
    for w in ("+43,1 %", "+28,6 %", "[−7,0 %; +14,2 %]", "+47,8 %", "+57,6 %", "2024-12-18", "+94,9 %",
              "+113,2 %", "−36,6 %", "Ägaren valde **A: S3 för båda kvoterna.**",
              "kvoten\" gav förlusterna 2012 och 2014", "`alpha_regime/buy_support.py`"):
        assert w in text[res:], w
    assert "Marginalmätaren i panelen\när fortfarande **bara kontext**" in text[:log]        # förregistreringen orörd
