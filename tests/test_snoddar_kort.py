"""
Snoddarnas kort ska vara ärliga.

(1) Förklaringen av vad som drar snodden visas bara när snodden faktiskt är spänd
    (TENSION_BUILDING / RUBBER_BAND_STRETCHED) — vid NEUTRAL en neutral rad, vid motsatt
    extrem att det "billiga" benet är historiskt dyrt mot det andra.
(2) Benens percentil är bara var PRISET står i sitt eget prisspann — orden säger det,
    och fönstret ("sedan <år>") står utskrivet.
(3) En rad med det billiga benet uttryckt i det andra benet, mot medianen i samma fönster.

Allt är ordval: _classify, _classify_driver och KÖP-STÖD-degraderingen ska ge exakt samma
utfall som förut (låses nedan). Inget nätverk — fejkade kursserier.
"""
import os

import numpy as np
import pandas as pd
import pytest

from alpha_regime import commodity_ratios as cr
from alpha_regime import ui as aui

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEUTRAL_LINE = "Snodden är inte spänd — benens prisläge förklarar inget köpläge."


def _gld_gdx(**kw):
    """Ägarens skärmdump: GLD/GDX på 35:e pct (NEUTRAL), GLD högt i sitt spann, GDX inte lågt."""
    base = dict(key="metal_miners", label="Gold ETF / Gold Miners", denominator_label="Gold Miners",
                cheap_direction="high", current=4.672, percentile=35.0, zscore=-0.4, status="NEUTRAL",
                numerator_label="GLD", denominator_leg_label="GDX", numerator_pctile=95.0,
                denominator_pctile=30.0, driver="NUMERATOR_EXPENSIVE", history_start="2016-11-03",
                history_years=10.0, median=6.25)
    base.update(kw)
    return cr.RatioResult(**base)


def _copper_gold(**kw):
    base = dict(key="copper_gold", label="Copper / Gold", denominator_label="Copper", cheap_direction="low",
                current=0.0011, percentile=50.0, zscore=0.0, status="NEUTRAL", numerator_label="HG=F",
                denominator_leg_label="GC=F", numerator_pctile=50.0, denominator_pctile=50.0, driver="MIXED",
                history_start="2016-11-03", history_years=10.0, median=0.0016)
    base.update(kw)
    return cr.RatioResult(**base)


# ── (1) Förklaringen bara när snodden är spänd ───────────────────────────────
@pytest.mark.parametrize("driver", ["DENOMINATOR_CHEAP", "NUMERATOR_EXPENSIVE", "BOTH", "MIXED", "UNKNOWN"])
def test_neutral_ratio_makes_no_driver_claim(driver):
    icon, text = aui._driver_verdict_text(_gld_gdx(driver=driver))
    assert text == NEUTRAL_LINE and icon == "⚪"
    for claim in ("Snodden är spänd för att", "köpstöd", "genuint", "nära toppen", "nära botten", "Blandad"):
        assert claim not in text


def test_owner_screenshot_case_no_longer_claims_a_driver():
    """GLD/GDX vid 35:e pct sa förut 'Snodden är spänd för att GLD är dyr (95:e pct)'."""
    r = _gld_gdx()
    assert aui._ratio_verdict(r)[0] == "⚪ NEUTRAL"
    assert aui._driver_verdict_text(r)[1] == NEUTRAL_LINE


def test_tension_and_stretched_show_the_driver():
    for status, pct in (("TENSION_BUILDING", 85.0), ("RUBBER_BAND_STRETCHED", 95.0)):
        icon, text = aui._driver_verdict_text(_gld_gdx(status=status, percentile=pct))
        assert icon == "⚠️"
        assert text == ("GLD står nära toppen av sitt eget prisspann (95:e pct), GDX står inte nära botten "
                        "av sitt (30:e pct) — panelens regel räknar det inte som köpstöd i GDX.")


def test_genuine_cheap_leg_wording_follows_what_the_rule_counts():
    stretched = _gld_gdx(status="RUBBER_BAND_STRETCHED", percentile=95.0, driver="DENOMINATOR_CHEAP",
                         numerator_pctile=50.0, denominator_pctile=12.0)
    icon, text = aui._driver_verdict_text(stretched)
    assert icon == "🟢"
    assert text == "GDX står nära botten av sitt eget prisspann (12:e pct) — panelens regel räknar det som köpstöd i GDX."
    # Spänning räknas inte som KÖP-STÖD av panelen förrän snodden är fullt sträckt
    tension = _gld_gdx(status="TENSION_BUILDING", percentile=85.0, driver="DENOMINATOR_CHEAP",
                       numerator_pctile=50.0, denominator_pctile=12.0)
    assert aui._driver_verdict_text(tension)[1].endswith("köpstöd i GDX först när snodden är fullt sträckt.")
    both = _gld_gdx(status="RUBBER_BAND_STRETCHED", percentile=95.0, driver="BOTH", denominator_pctile=12.0)
    icon, text = aui._driver_verdict_text(both)
    assert icon == "🟢⚡" and "GDX står nära botten" in text and "GLD nära toppen av sitt (95:e pct)" in text
    assert text.endswith("panelens regel räknar det som köpstöd i GDX.") and "Starkt" not in text
    mixed = _gld_gdx(status="RUBBER_BAND_STRETCHED", percentile=95.0, driver="MIXED", numerator_pctile=60.0)
    assert aui._driver_verdict_text(mixed)[1].startswith("Blandad signal — varken GDX nära botten (30:e pct)")
    assert aui._driver_verdict_text(mixed)[1].endswith("räknar det inte som köpstöd i GDX.")


def test_low_direction_driver_names_the_numerator_as_cheap_leg():
    r = _copper_gold(status="RUBBER_BAND_STRETCHED", percentile=8.0, driver="NUMERATOR_EXPENSIVE",
                     numerator_pctile=45.0, denominator_pctile=97.0)
    assert aui._driver_verdict_text(r)[1].startswith(
        "GC=F står nära toppen av sitt eget prisspann (97:e pct), HG=F står inte nära botten av sitt (45:e pct)")


def test_sell_extreme_says_the_cheap_leg_is_dear_and_never_talks_buy():
    high = _gld_gdx(percentile=5.0)                      # GLD/GDX lågt → GDX dyr mot GLD
    assert aui._ratio_verdict(high)[0] == "🔴 SÄLJ-STÖD"
    icon, text = aui._driver_verdict_text(high)
    assert icon == "🔴" and text == "Snodden är spänd åt andra hållet — GDX är historiskt dyr mot GLD."
    low = _copper_gold(percentile=95.0)                  # Copper/Gold högt → koppar dyr mot guld
    assert aui._ratio_verdict(low)[0] == "🔴 SÄLJ-STÖD"
    text = aui._driver_verdict_text(low)[1]
    assert text == "Snodden är spänd åt andra hållet — HG=F är historiskt dyr mot GC=F."
    for t in (aui._driver_verdict_text(high)[1], text):
        assert "köp" not in t.lower()


# ── (2) Ärliga ord för benen ─────────────────────────────────────────────────
@pytest.mark.parametrize("pct,word", [(100.0, "nära toppen"), (95.0, "nära toppen"), (80.0, "nära toppen"),
                                      (79.9, "övre delen"), (60.0, "övre delen"), (59.9, "mitten"),
                                      (40.0, "mitten"), (39.9, "nedre delen"), (20.1, "nedre delen"),
                                      (20.0, "nära botten"), (3.0, "nära botten")])
def test_leg_words_are_price_positions(pct, word):
    assert aui._leg_pctile_label(pct) == word
    assert aui._leg_pctile_color(pct) not in ("#cc3333", "#1aaa5a")   # inget rött "dyrt" / grönt "billigt"


def test_leg_word_ends_match_the_driver_rule_cut_offs():
    """'nära botten' exakt när regeln kallar det billiga benet billigt, 'nära toppen' exakt när
    regeln kallar det andra benet dyrt — orden kan aldrig motsäga förklaringsraden."""
    for p in np.arange(0.0, 100.01, 0.1):
        p = round(float(p), 1)
        cheap_rule = cr._classify_driver(50.0, p, "high") == "DENOMINATOR_CHEAP"
        dear_rule = cr._classify_driver(p, 50.0, "high") == "NUMERATOR_EXPENSIVE"
        assert (aui._leg_pctile_label(p) == "nära botten") == cheap_rule, p
        assert (aui._leg_pctile_label(p) == "nära toppen") == dear_rule, p


def test_window_is_named_from_the_aligned_series_start():
    assert aui._leg_window_text(_gld_gdx()) == "sitt eget prisspann sedan 2016"
    assert aui._leg_window_text(_gld_gdx(history_start="")) == "sitt eget prisspann"


# ── (3) Det billiga benet i det andra benets termer ─────────────────────────
def test_cheap_terms_high_direction_is_the_inverse_ratio():
    # 1/4,672 = 0,2140 GLD per GDX; medianen 1/6,25 = 0,160 → +34 %
    assert aui._cheap_terms_text(_gld_gdx()) == (
        "I GLD: 1 GDX = 0,214 GLD — 34 % mer än medianen sedan 2016 (0,160), alltså dyrare i GLD än normalt.")
    # 1/8 = 0,125 → −22 %
    assert aui._cheap_terms_text(_gld_gdx(current=8.0)) == (
        "I GLD: 1 GDX = 0,125 GLD — 22 % mindre än medianen sedan 2016 (0,160), alltså billigare i GLD än normalt.")


def test_cheap_terms_low_direction_is_the_ratio_itself():
    # 0,0011 / 0,0016 = 0,6875 → −31 %
    assert aui._cheap_terms_text(_copper_gold()) == (
        "I GC=F: 1 HG=F = 0,00110 GC=F — 31 % mindre än medianen sedan 2016 (0,00160), "
        "alltså billigare i GC=F än normalt.")
    assert "mer än medianen" in aui._cheap_terms_text(_copper_gold(current=0.002))


def test_cheap_terms_edge_cases():
    assert aui._cheap_terms_text(_gld_gdx(current=6.25)) == (
        "I GLD: 1 GDX = 0,160 GLD — i nivå med medianen sedan 2016 (0,160), alltså normalt prissatt i GLD.")
    assert aui._cheap_terms_text(_gld_gdx(median=None)) is None          # gammal cache/egen ratio utan median
    assert aui._cheap_terms_text(_gld_gdx(current=0.0)) is None
    assert aui._cheap_terms_text(_gld_gdx(numerator_label="")) is None
    assert "medianen (0,160)" in aui._cheap_terms_text(_gld_gdx(history_start=""))


def test_swedish_number_formatting():
    assert aui._sv_num(0.21404, aui._sig_decimals(0.21404)) == "0,214"
    assert aui._sv_num(0.0998, aui._sig_decimals(0.0998)) == "0,0998"
    assert aui._sv_num(4.6721, aui._sig_decimals(4.6721)) == "4,67"
    assert aui._sv_num(1234.6, aui._sig_decimals(1234.6)) == "1 235"


# ── Motorn fyller fälten — och reglerna är orörda ───────────────────────────
def _series(values, end="2026-10-30"):
    return pd.Series(np.asarray(values, dtype=float), index=pd.bdate_range(end=end, periods=len(values)))


def test_compute_ratio_fills_window_and_median_from_the_aligned_series(monkeypatch):
    n = 2600
    rng = np.random.default_rng(7)
    gld = _series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))))
    gdx = _series(30 * np.exp(np.cumsum(rng.normal(0.0001, 0.02, n))))
    gdx = gdx.iloc[200:]                                   # kortare ben → fönstret börjar där båda finns
    monkeypatch.setattr(cr, "_download_close", lambda t, period="10y": {"GLD": gld, "GDX": gdx}[t])
    spec = next(s for s in cr._RATIO_SPECS if s.key == "metal_miners")
    r = cr._compute_ratio(spec)
    aligned = pd.concat({"n": gld, "d": gdx}, axis=1).dropna()
    ratio = aligned["n"] / aligned["d"]
    assert r.history_start == str(aligned.index[0].date()) and r.history_start[:4].isdigit()
    assert r.history_years == round((aligned.index[-1] - aligned.index[0]).days / 365.25, 1)
    assert r.median == pytest.approx(float(np.median(ratio.values)), rel=1e-5)
    # Status och driver exakt som regelfunktionerna ger dem
    arr = ratio.values
    assert r.status == cr._classify(cr._percentile_of(arr, float(arr[-1])), "high")
    num_p = cr._percentile_of(aligned["n"].values, float(aligned["n"].values[-1]))
    den_p = cr._percentile_of(aligned["d"].values, float(aligned["d"].values[-1]))
    assert r.driver == cr._classify_driver(num_p, den_p, "high")


def test_new_fields_have_safe_defaults():
    r = cr.RatioResult(key="x", label="X", denominator_label="Y", cheap_direction="high")
    assert (r.history_start, r.history_years, r.median) == ("", 0.0, None)
    assert cr._window_info(pd.DatetimeIndex([]), np.array([])) == ("", 0.0, None)


def test_rule_functions_keep_their_exact_behaviour():
    assert cr._classify_driver(95.0, 30.0, "high") == "NUMERATOR_EXPENSIVE"
    assert cr._classify_driver(50.0, 20.0, "high") == "DENOMINATOR_CHEAP"
    assert cr._classify_driver(80.0, 20.0, "high") == "BOTH"
    assert cr._classify_driver(79.9, 20.1, "high") == "MIXED"
    assert cr._classify_driver(15.0, 85.0, "low") == "BOTH"
    assert [cr._classify(p, "high") for p in (90.0, 89.9, 80.0, 79.9)] == [
        "RUBBER_BAND_STRETCHED", "TENSION_BUILDING", "TENSION_BUILDING", "NEUTRAL"]
    assert [cr._classify(p, "low") for p in (10.0, 10.1, 20.0, 20.1)] == [
        "RUBBER_BAND_STRETCHED", "TENSION_BUILDING", "TENSION_BUILDING", "NEUTRAL"]
    # KÖP-STÖD räknas bara när det billiga benet driver (UNKNOWN släpps igenom som förut)
    ratios = {
        "a": _gld_gdx(status="RUBBER_BAND_STRETCHED", percentile=95.0, driver="NUMERATOR_EXPENSIVE"),
        "b": _gld_gdx(status="RUBBER_BAND_STRETCHED", percentile=95.0, driver="DENOMINATOR_CHEAP"),
        "c": _gld_gdx(status="RUBBER_BAND_STRETCHED", percentile=95.0, driver="MIXED"),
        "d": _gld_gdx(status="RUBBER_BAND_STRETCHED", percentile=95.0, driver="UNKNOWN"),
        "e": _gld_gdx(status="TENSION_BUILDING", percentile=85.0, driver="DENOMINATOR_CHEAP"),
    }
    assert aui._compute_ratio_summary(ratios, list(ratios)) == "Gummisnoddar: 2 KÖP-STÖD, 1 SPÄNNING, 2 NEUTRAL"


def test_engine_confirmation_is_untouched():
    """+1 ACCUMULATE kräver fortfarande RUBBER_BAND_STRETCHED och DENOMINATOR_CHEAP/BOTH — och läser
    aldrig de nya informationsfälten."""
    src = open(os.path.join(ROOT, "alpha_regime", "engine.py"), encoding="utf-8").read()
    assert '_ratio.status == "RUBBER_BAND_STRETCHED"' in src
    assert 'in ("DENOMINATOR_CHEAP", "BOTH")' in src
    for field in ("history_start", "history_years", ".median"):
        assert field not in src


# ── AppTest: kortet renderas (inget nätverk) ─────────────────────────────────
def _card_app():
    import streamlit as st
    from alpha_regime import commodity_ratios as c
    from alpha_regime import ui as u
    case = st.session_state.get("_case", "neutral")
    kw = dict(key="metal_miners", label="Gold ETF / Gold Miners", denominator_label="Gold Miners",
              cheap_direction="high", current=4.672, percentile=35.0, zscore=-0.4, status="NEUTRAL",
              numerator_label="GLD", denominator_leg_label="GDX", numerator_pctile=95.0,
              denominator_pctile=30.0, driver="NUMERATOR_EXPENSIVE", history_start="2016-11-03",
              history_years=10.0, median=6.25, sparkline_dates=["2026-10-23", "2026-10-30"],
              sparkline_values=[4.70, 4.672])
    if case == "stretched":
        kw.update(current=8.4, percentile=96.0, status="RUBBER_BAND_STRETCHED", driver="DENOMINATOR_CHEAP",
                  numerator_pctile=55.0, denominator_pctile=9.0)
    u._render_single_ratio(c.RatioResult(**kw), idx=0)


def _html(at):
    return " ".join(m.value for m in at.markdown)


def test_apptest_neutral_card_is_honest():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_card_app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    html = _html(at)
    assert "⚪ NEUTRAL" in html and NEUTRAL_LINE in html
    assert "95:e pct — nära toppen av sitt eget prisspann sedan 2016" in html
    assert "30:e pct — nedre delen av sitt eget prisspann sedan 2016" in html
    assert "I GLD: 1 GDX = 0,214 GLD — 34 % mer än medianen sedan 2016 (0,160), alltså dyrare i GLD än normalt." in html
    for old in ("DYRT", "BILLIGT", "NEUTRALT", "Snodden är spänd för att", "Inte ett äkta köpläge", "genuint billig"):
        assert old not in html, old


def test_apptest_stretched_card_shows_the_driver():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_card_app, default_timeout=60)
    at.session_state["_case"] = "stretched"
    at.run()
    assert not at.exception, at.exception
    html = _html(at)
    assert "🟢 KÖP-STÖD" in html
    assert "GDX står nära botten av sitt eget prisspann (9:e pct) — panelens regel räknar det som köpstöd i GDX." in html
    assert "9:e pct — nära botten av sitt eget prisspann sedan 2016" in html
    assert "I GLD: 1 GDX = 0,119 GLD — 26 % mindre än medianen sedan 2016 (0,160), alltså billigare i GLD än normalt." in html
    assert NEUTRAL_LINE not in html


def _builder_app():
    from alpha_regime import ui as u
    u._render_custom_ratio_builder()


def test_apptest_custom_ratio_gets_window_and_median(monkeypatch):
    from streamlit.testing.v1 import AppTest
    idx = pd.bdate_range(end="2026-10-30", periods=1300)
    gld = pd.Series(np.linspace(150.0, 250.0, 1300), index=idx)
    gdxj = pd.Series(np.linspace(30.0, 40.0, 1300), index=idx)
    monkeypatch.setattr(aui, "_cr_download_close", lambda t, period="10y": {"GLD": gld, "GDXJ": gdxj}[t])
    at = AppTest.from_function(_builder_app, default_timeout=60)
    at.run()
    at.button(key="cr_run").click().run()
    assert not at.exception, at.exception
    saved = at.session_state["ar_custom_ratios"]
    assert saved and saved[0].history_start == str(idx[0].date()) and saved[0].median
    ratio = gld / gdxj
    assert saved[0].median == pytest.approx(float(np.median(ratio.values)), rel=1e-5)
    html = _html(at)
    assert f"sitt eget prisspann sedan {idx[0].year}" in html
    assert "I GLD: 1 GDXJ = " in html and "än medianen sedan" in html
