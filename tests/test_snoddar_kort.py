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
NB = "\u00a0"   # no-break space före % och som tusentalsavgränsare


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
        f"I GLD: 1 GDX = 0,214 GLD — 34{NB}% mer än medianen sedan 2016 (0,160), alltså dyrare i GLD än normalt.")
    # 1/8 = 0,125 → −22 %
    assert aui._cheap_terms_text(_gld_gdx(current=8.0)) == (
        f"I GLD: 1 GDX = 0,125 GLD — 22{NB}% mindre än medianen sedan 2016 (0,160), alltså billigare i GLD än normalt.")


def test_cheap_terms_low_direction_is_the_ratio_itself():
    # 0,0011 / 0,0016 = 0,6875 → −31 %
    assert aui._cheap_terms_text(_copper_gold()) == (
        f"I GC=F: 1 HG=F = 0,00110 GC=F — 31{NB}% mindre än medianen sedan 2016 (0,00160), "
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
    assert aui._sv_num(1234.6, aui._sig_decimals(1234.6)) == f"1{NB}235"


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
    assert f"I GLD: 1 GDX = 0,214 GLD — 34{NB}% mer än medianen sedan 2016 (0,160), alltså dyrare i GLD än normalt." in html
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
    assert f"I GLD: 1 GDX = 0,119 GLD — 26{NB}% mindre än medianen sedan 2016 (0,160), alltså billigare i GLD än normalt." in html
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


# ══ Rättelser efter verifieringen ═══════════════════════════════════════════
def _ranked(n, k, seed, lo=1.0, hi=10.0):
    """n olika värden där det SISTA har exakt k värden ≤ sig (sig självt inräknat) → råpercentil k/n·100."""
    rng = np.random.default_rng(seed)
    others = np.sort(rng.uniform(lo, hi, n - 1))
    cur = (others[k - 2] + others[k - 1]) / 2
    return np.concatenate([rng.permutation(others), [cur]])


def _run_spec(monkeypatch, key, num_vals, den_vals):
    """_compute_ratio på fejkade kursserier (inget nätverk)."""
    spec = next(s for s in cr._RATIO_SPECS if s.key == key)
    idx = pd.bdate_range(end="2026-10-30", periods=len(num_vals))
    series = {spec.primary_num: pd.Series(np.asarray(num_vals, dtype=float), index=idx),
              spec.primary_den: pd.Series(np.asarray(den_vals, dtype=float), index=idx)}
    monkeypatch.setattr(cr, "_download_close", lambda t, period="10y": series.get(t, pd.Series(dtype=float)))
    return cr._compute_ratio(spec)


def _ratio_app():
    import streamlit as st
    from alpha_regime import ui as u
    u._render_single_ratio(st.session_state["_ratio"], idx=0, counted=st.session_state.get("_counted", True))


def _render(ratio, counted=True):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_ratio_app, default_timeout=60)
    at.session_state["_ratio"] = ratio
    at.session_state["_counted"] = counted
    at.run()
    return at


# ── (major) En trasig stängning får aldrig krascha kortet ───────────────────
@pytest.mark.parametrize("direction", ["high", "low"])
@pytest.mark.parametrize("current,median", [
    (float("inf"), 6.25), (float("nan"), 6.25), (float("-inf"), 6.25), (-1.0, 6.25),
    (4.672, float("inf")), (4.672, float("nan")), (4.672, 0.0), (4.672, None), (float("nan"), float("nan")),
])
def test_cheap_terms_rejects_non_finite_values(direction, current, median):
    r = _gld_gdx(cheap_direction=direction, current=current, median=median)
    assert aui._cheap_terms_text(r) is None


def test_cheap_terms_rejects_an_inverse_that_overflows():
    """'high' räknar 1/kvoten — en subnormal kvot ger 1/x = inf, som aldrig får nå round()."""
    assert aui._cheap_terms_text(_gld_gdx(current=1e-320)) is None
    assert aui._cheap_terms_text(_gld_gdx(median=1e-320)) is None


@pytest.mark.parametrize("key", ["gdx_spy", "metal_miners"])
def test_zero_last_close_renders_the_card_without_exception(monkeypatch, key):
    """Sista nämnarstängningen = 0 → current = inf. Förut OverflowError i _render_single_ratio för
    'low' och nonsensraden '1 GDX = 0 GLD — 100 % mindre' för 'high'."""
    rng = np.random.default_rng(8)
    num_v = 30 * np.exp(np.cumsum(rng.normal(0, 0.01, 1500)))
    den_v = 400 * np.exp(np.cumsum(rng.normal(0, 0.01, 1500)))
    den_v[-1] = 0.0
    r = _run_spec(monkeypatch, key, num_v, den_v)
    assert r.status != "DATA_GAP" and not np.isfinite(r.current)
    assert aui._cheap_terms_text(r) is None
    at = _render(r)
    assert not at.exception, at.exception
    assert "⚖️" not in _html(at) and "= 0 " not in _html(at)


@pytest.mark.parametrize("direction", ["high", "low"])
@pytest.mark.parametrize("current", [float("inf"), float("nan")])
def test_apptest_inf_and_nan_current_render_without_exception(direction, current):
    at = _render(_gld_gdx(cheap_direction=direction, current=current))
    assert not at.exception, at.exception
    assert "⚖️" not in _html(at)


# ── (minor) Förklaringsraden följer samma källa som märket och sammanräkningen ─
def test_ratio_just_under_80_follows_the_tension_badge(monkeypatch):
    """Råpercentil 2012/2516 = 79,968 → status NEUTRAL (regeln orörd), men lagrad 80,0 → märket
    ⚡ SPÄNNING BYGGS och 'Gummisnoddar: 1 SPÄNNING'. Raden får inte säga 'inte spänd' bredvid."""
    n = 2516
    ratio = _ranked(n, 2012, 3)
    den = 10.0 * np.linspace(1.0, 1.01, n)
    r = _run_spec(monkeypatch, "metal_miners", ratio * den, den)
    assert r.status == "NEUTRAL" and r.percentile == 80.0                 # regeln: oförändrad
    assert aui._ratio_verdict(r)[0] == "⚡ SPÄNNING BYGGS"
    assert aui._compute_ratio_summary({"m": r}, ["m"]) == "Gummisnoddar: 1 SPÄNNING"
    text = aui._driver_verdict_text(r)[1]
    assert text != NEUTRAL_LINE and "inte spänd" not in text


def test_ratio_just_under_90_with_cheap_leg_says_counted_like_the_badge(monkeypatch):
    """Råpercentil 2264/2516 = 89,98 → status TENSION_BUILDING (motorns +1 orörd), lagrad 90,0 →
    🟢 KÖP-STÖD och '1 KÖP-STÖD'. Raden får inte säga 'först när snodden är fullt sträckt'."""
    n = 2516
    ratio = _ranked(n, 2264, 4)
    den = _ranked(n, 200, 5, 20, 40)                                      # GDX på 7,9:e pct i eget spann
    r = _run_spec(monkeypatch, "metal_miners", ratio * den, den)
    assert r.status == "TENSION_BUILDING" and r.percentile == 90.0 and r.driver == "DENOMINATOR_CHEAP"
    assert aui._ratio_verdict(r)[0] == "🟢 KÖP-STÖD"
    assert aui._compute_ratio_summary({"m": r}, ["m"]) == "Gummisnoddar: 1 KÖP-STÖD"
    icon, text = aui._driver_verdict_text(r)
    assert icon == "🟢" and text.endswith("panelens regel räknar det som köpstöd i GDX.")
    assert "först när" not in text


@pytest.mark.parametrize("k,badge,status", [(251, "🟢 KÖP-STÖD", "TENSION_BUILDING"),
                                            (501, "⚡ SPÄNNING BYGGS", "NEUTRAL")])
def test_low_direction_rounding_edges_follow_the_badge(monkeypatch, k, badge, status):
    """GDX/SPY: råpercentil 10,04 / 20,04 lagras 10,0 / 20,0 — märket och raden ska vara överens."""
    n = 2500
    ratio = _ranked(n, k, 11, 0.05, 0.12)
    den = 400.0 * np.linspace(1.0, 1.02, n)
    r = _run_spec(monkeypatch, "gdx_spy", ratio * den, den)
    assert r.status == status                                             # regeln: oförändrad
    assert aui._ratio_verdict(r)[0] == badge
    text = aui._driver_verdict_text(r)[1]
    assert text != NEUTRAL_LINE
    if "KÖP-STÖD" in badge:
        assert "först när" not in text


@pytest.mark.parametrize("pct,zone", [(95.0, "buy"), (90.0, "buy"), (85.0, "tension"), (80.0, "tension"),
                                      (79.9, "neutral"), (35.0, "neutral"), (10.1, "neutral"), (10.0, "sell")])
def test_band_zone_is_the_badge(pct, zone):
    r = _gld_gdx(percentile=pct)
    assert aui._band_zone(r) == zone
    assert aui._driver_applies(r) == (zone in ("buy", "tension"))


# ── (minor) Benens ord kan aldrig motsäga förklaringsraden ──────────────────
def test_cheap_leg_at_20_04_reads_lower_part_not_near_bottom(monkeypatch):
    """GDX-benet på råa 501/2500 = 20,04 lagras 20,0 — regeln kallar det inte billigt, så raden får
    inte säga 'nära botten' bredvid 'varken GDX nära botten'."""
    n = 2500
    den = _ranked(n, 501, 6, 20, 40)
    ratio = _ranked(n, 2450, 7)                                          # 98:e pct → sträckt
    r = _run_spec(monkeypatch, "metal_miners", ratio * den, den)
    assert r.denominator_pctile == 20.0 and r.status == "RUBBER_BAND_STRETCHED"
    assert r.driver not in ("DENOMINATOR_CHEAP", "BOTH")
    assert aui._leg_words(r)[1] == "nedre delen"
    at = _render(r)
    assert not at.exception, at.exception
    html = _html(at)
    assert "20:e pct — nedre delen av sitt eget prisspann" in html
    assert "20:e pct — nära botten" not in html


def test_other_leg_at_79_96_reads_upper_part_not_near_top(monkeypatch):
    """GLD-benet på råa 1999/2500 = 79,96 lagras 80,0 — regeln kallar det inte dyrt."""
    n = 2500
    num = _ranked(n, 1999, 9, 100, 200)
    ratio = _ranked(n, 2450, 10)
    r = _run_spec(monkeypatch, "metal_miners", num, num / ratio)
    assert r.numerator_pctile == 80.0 and r.status == "RUBBER_BAND_STRETCHED"
    assert r.driver not in ("NUMERATOR_EXPENSIVE", "BOTH")
    assert aui._leg_words(r)[0] == "övre delen"
    assert "GLD står nära toppen" not in aui._driver_verdict_text(r)[1]


@pytest.mark.parametrize("direction", ["high", "low"])
@pytest.mark.parametrize("driver", ["DENOMINATOR_CHEAP", "NUMERATOR_EXPENSIVE", "BOTH", "MIXED"])
def test_leg_word_ends_come_from_the_driver(direction, driver):
    """'nära botten' på det billiga benet exakt när drivaren säger billigt, 'nära toppen' på det andra
    benet exakt när drivaren säger dyrt — oavsett avrundningen av den lagrade percentilen."""
    cheap_low = driver in ("DENOMINATOR_CHEAP", "BOTH")
    other_high = driver in ("NUMERATOR_EXPENSIVE", "BOTH")
    for cheap_p in (0.0, 5.0, 19.9, 20.0, 20.1, 30.0, 50.0):
        for other_p in (50.0, 70.0, 79.9, 80.0, 80.1, 95.0):
            num_p, den_p = (other_p, cheap_p) if direction == "high" else (cheap_p, other_p)
            r = _gld_gdx(cheap_direction=direction, driver=driver, numerator_pctile=num_p,
                         denominator_pctile=den_p)
            num_w, den_w = aui._leg_words(r)
            cheap_w, other_w = (den_w, num_w) if direction == "high" else (num_w, den_w)
            assert (cheap_w == "nära botten") == cheap_low, (cheap_p, other_p)
            assert (other_w == "nära toppen") == other_high, (cheap_p, other_p)
            # Mittenorden är hinkarna; en hink-ände som drivaren inte bär blir närmaste mittenord
            cheap_bucket, other_bucket = aui._leg_pctile_label(cheap_p), aui._leg_pctile_label(other_p)
            if not cheap_low:
                assert cheap_w == ("nedre delen" if cheap_bucket == "nära botten" else cheap_bucket)
            if not other_high:
                assert other_w == ("övre delen" if other_bucket == "nära toppen" else other_bucket)


def test_unknown_driver_keeps_the_plain_buckets():
    r = _gld_gdx(driver="UNKNOWN", numerator_pctile=95.0, denominator_pctile=5.0)
    assert aui._leg_words(r) == ("nära toppen", "nära botten")


# ── (minor) Egna kvoter räknas inte — kortet säger det ──────────────────────
@pytest.mark.parametrize("driver", ["DENOMINATOR_CHEAP", "NUMERATOR_EXPENSIVE", "BOTH", "MIXED"])
@pytest.mark.parametrize("pct", [95.0, 85.0])
def test_custom_ratio_never_claims_the_panel_counts_it(driver, pct):
    r = _gld_gdx(key="custom_GLD_GDXJ", percentile=pct, driver=driver, denominator_pctile=12.0)
    text = aui._driver_verdict_text(r, counted=False)[1]
    assert text.endswith("egen kvot — räknas inte i panelens bekräftelse.")
    assert "panelens regel" not in text
    # Panelens egna kvoter säger fortfarande vad regeln gör
    assert "panelens regel" in aui._driver_verdict_text(r)[1]


def test_apptest_custom_ratio_card_says_it_is_not_counted(monkeypatch):
    from streamlit.testing.v1 import AppTest
    idx = pd.bdate_range(end="2026-10-30", periods=1300)
    gld = pd.Series(np.linspace(150.0, 250.0, 1300), index=idx)
    gdxj = pd.Series(np.linspace(40.0, 30.0, 1300), index=idx)          # GDXJ på botten → BOTH, sträckt
    monkeypatch.setattr(aui, "_cr_download_close", lambda t, period="10y": {"GLD": gld, "GDXJ": gdxj}[t])
    at = AppTest.from_function(_builder_app, default_timeout=60)
    at.run()
    at.button(key="cr_run").click().run()
    assert not at.exception, at.exception
    saved = at.session_state["ar_custom_ratios"]
    assert saved[0].status == "RUBBER_BAND_STRETCHED" and saved[0].driver == "BOTH"
    html = _html(at)
    assert "egen kvot — räknas inte i panelens bekräftelse." in html
    assert "panelens regel räknar det som köpstöd" not in html


# ── (minor) Svensk formatering ──────────────────────────────────────────────
@pytest.mark.parametrize("v,word", [(0, "0:e"), (1, "1:a"), (2, "2:a"), (3, "3:e"), (11, "11:e"), (12, "12:e"),
                                    (21, "21:a"), (22, "22:a"), (31, "31:a"), (95, "95:e"), (100, "100:e"),
                                    (101, "101:a"), (111, "111:e"), (1.4, "1:a"), (21.6, "22:a")])
def test_swedish_ordinals(v, word):
    assert aui._sv_ord(v) == word


def test_sv_ord_never_raises_on_bad_values():
    for v in (float("nan"), float("inf"), None, "x"):
        assert aui._sv_ord(v).endswith(":e")


def test_card_texts_use_swedish_ordinals():
    cheap = _gld_gdx(status="RUBBER_BAND_STRETCHED", percentile=95.0, driver="DENOMINATOR_CHEAP",
                     numerator_pctile=50.0, denominator_pctile=1.0)
    assert "(1:a pct)" in aui._driver_verdict_text(cheap)[1]
    dear = _gld_gdx(status="RUBBER_BAND_STRETCHED", percentile=95.0, numerator_pctile=81.0, denominator_pctile=22.0)
    text = aui._driver_verdict_text(dear)[1]
    assert "(81:a pct)" in text and "(22:a pct)" in text and ":e pct" not in text
    assert aui._ratio_verdict(_gld_gdx(percentile=31.0))[2].endswith("vid 31:a percentilen — ingen extrem")
    at = _render(_gld_gdx(numerator_pctile=92.0, denominator_pctile=21.0))
    assert not at.exception, at.exception
    html = _html(at)
    assert "92:a pct — nära toppen" in html and "21:a pct — nedre delen" in html


def test_cheap_terms_uses_no_break_spaces_and_the_medians_own_decimals():
    # 'high': 1/0,01 = 100 GLD per GDX mot medianen 1/0,29 = 3,45 — medianen med egna decimaler, inte '(3)'
    text = aui._cheap_terms_text(_gld_gdx(current=0.01, median=0.29))
    assert text == (f"I GLD: 1 GDX = 100 GLD — 2{NB}800{NB}% mer än medianen sedan 2016 (3,45), "
                    "alltså dyrare i GLD än normalt.")
    text = aui._cheap_terms_text(_gld_gdx(current=0.29, median=0.01))
    assert "1 GDX = 3,45 GLD" in text and "(100)" in text and f"97{NB}% mindre" in text
    for t in (aui._cheap_terms_text(_gld_gdx()), aui._cheap_terms_text(_copper_gold())):
        assert " %" not in t and f"{NB}%" in t
