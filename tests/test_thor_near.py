"""
👀 Övervakningen — thor/near.py: "hur långt från köp" per strategi ur fälten som redan finns i Thors rad och
körningens blob. Inga nya regler: KÖP-reglerna läses ur strategins egna konstanter. Syntetiska filer, inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_candidates as fx  # noqa: E402
from thor import candidates as tc  # noqa: E402
from thor.near import NEAR, near_text  # noqa: E402

ROWS = {(c.strategy, c.ticker): c for c in tc.build(fx.BLOBS, today="2026-10-15")}


def _t(strategy, ticker, data=fx.BLOBS):
    return near_text(ROWS[(strategy, ticker)], data)


def test_every_strategy_in_the_register_has_a_branch():
    from thor import registry as reg
    assert set(NEAR) == set(reg.STRATEGIES)


def test_buy_and_blocked_rows_say_so_in_the_strategys_own_words():
    assert _t("viking_nine", "HEXA-B.ST") == "köpläge nu — GOLDEN TICKET"
    assert _t("momentum", "ERIC-B.ST") == "köpläge nu — rad 1 · setup A"
    assert _t("quality", "SSAB-A.ST") == "köpsignal men spärrad — BUY 4/4 · sektorn RÖD"
    assert near_text(None, fx.BLOBS) == ""


def test_watch_rows_say_how_far_from_buy():
    assert _t("momentum", "ABB.ST") == "rad 5 — setup A saknas"
    assert _t("momentum", "ANOT.ST") == "rad 21 — topp 20 krävs (setup A finns)"
    assert _t("viking_nine", "ERIC-B.ST") == "7/9 — 2 faktor(er) kvar till 9/9"
    assert _t("wolf", "VOLV-B.ST") == "66/125 · MODERATE — ingen entry ännu (köpregeln ej uppfylld)"
    assert _t("quality", "HOLM-B.ST") == "Alpha Regime 3/4 — 1 signal(er) kvar till BUY"
    assert _t("quality", "SEB-A.ST") == "rank 30 — inte analyserad av Alpha Regime än"
    assert _t("viking", "SKF-B.ST") == "WATCH · screener 6/9 — BUY = över EMA200, ADX ≥ 20 och composite ≥ 60"
    assert _t("insider", "SKA-B.ST") == "Bevaka — vänta på fler köp"                   # statusflödets hela text
    assert _t("insider", "X.ST") == "KÖP — logga i journalen · passa (jagar kursen)"
    assert _t("durrett", "XX.V") == "bull 4× · poäng 5/10 · kvalitet 40 — papperet köper 15-BAGGER eller nordiska"
    assert _t("deep", "NEM").startswith("rank 1 — listan rangordnar bara")
    assert _t("hav", "AEM").startswith("håvträff (Överlevarna (Rule))")


def test_raw_blob_fields_fill_in_the_details():
    vn = {"generated": "2026-10-14", "rows": [
        {"ticker": "VOLV-B.ST", "category": "READY", "ovtlyr_nine": "9/9", "nine_weighted": 90.0, "viking": "5/7",
         "reason": "WAIT FOR CLOSE"},
        {"ticker": "SAND.ST", "category": "DEVELOPING", "ovtlyr_nine": "7/9", "nine_weighted": 70.0,
         "market": "2/3", "sector": "1/2", "stock": "4/4"}]}
    ember = {"timestamp": "2026-10-14T10:00:00", "eligible": [], "near_misses": [
        {"ticker": "NHY.OL", "verdict": "BEVAKA", "setup_score": 55.0, "trend_pass": True, "entry_pass": False,
         "notrade_clear": True}]}
    royalty = {"generated": "2026-10-14",
               "rows": [{"ticker": "WPM", "verdict": "WATCH", "composite": 58.0, "pillars": {"hate": 50}}]}
    data = {**fx.BLOBS, "viking_nine": vn, "ember": ember, "royalty": royalty}
    rows = {(c.strategy, c.ticker): c for c in tc.build(data, today="2026-10-15")}
    assert near_text(rows[("viking_nine", "VOLV-B.ST")], data) == "9/9 — Execution väntar (Viking 5/7): WAIT FOR CLOSE"
    assert near_text(rows[("viking_nine", "SAND.ST")], data) == \
        "7/9 — 2 faktor(er) kvar till 9/9 (marknad 2/3 · sektor 1/2 · aktie 4/4)"
    assert near_text(rows[("ember", "NHY.OL")], data) == "BEVAKA — 2/3 grindar (trend ✓ · entry ✗ · ingen spärr ✓)"
    from royalty.engine import BUY_MIN
    assert near_text(rows[("royalty", "WPM")], data) == f"composite 58 — BUY CANDIDATE från {BUY_MIN:.0f}"


def test_broken_blobs_fall_back_to_the_strategys_own_word():
    c = ROWS[("ember", "NHY.OL")]
    assert near_text(c, {"ember": "trasig"}) == "AVVAKTA" and near_text(c, None) == "AVVAKTA"
    vn = ROWS[("viking_nine", "ERIC-B.ST")]
    assert near_text(vn, {"viking_nine": {"rows": "trasig"}}) == "7/9 — 2 faktor(er) kvar till 9/9"
    odd = tc.Candidate("Z.ST", "", "okänd", "något", tc.BEVAKA)
    assert near_text(odd, {}) == "något"
