"""
Temaproben: de största bolagen med positivt kassaflöde (KPI 62) per tema, rangordnade på börsvärde (KPI 50)
i USD. Bolag utan kassaflöde och bolag som saknas i Börsdata redovisas separat. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universe import theme_probe as tp  # noqa: E402


def test_rank_keeps_cash_flow_and_sorts_by_usd_market_cap():
    index = {"MOWI.OL": {"insId": 1, "name": "Mowi", "stockPriceCurrency": "NOK"},
             "SALM.OL": {"insId": 2, "name": "SalMar", "stockPriceCurrency": "NOK"},
             "BAKKA.OL": {"insId": 3, "name": "Bakkafrost", "stockPriceCurrency": "NOK"},
             "BAKKA.CO": {"insId": 3, "name": "Bakkafrost", "stockPriceCurrency": "DKK"},   # samma bolag två gånger
             "ISLAX.OL": {"insId": 4, "name": "Icelandic Salmon", "stockPriceCurrency": "NOK"}}
    mcap = {1: 100_000, 2: 80_000, 3: 30_000, 4: 5_000}
    ocf = {1: 8_000, 2: 6_000, 3: 2_000, 4: -50}
    rows, missing, negative = tp.rank({"MOWI.OL", "SALM.OL", "BAKKA.OL", "BAKKA.CO", "ISLAX.OL", "XYZ.OL"},
                                      index, mcap, ocf, {}, {"NOK": 0.1, "DKK": 0.15})
    assert [r["ticker"] for r in rows] == ["MOWI.OL", "SALM.OL", "BAKKA.CO"]      # en rad per bolag
    assert rows[0]["mcap_usd"] == 10_000 and [r["ticker"] for r in negative] == ["ISLAX.OL"]
    assert missing == ["XYZ.OL"]


def test_every_map_theme_has_candidates_and_fx_fallback():
    assert set(tp.map_themes()) <= set(tp.CANDIDATES)
    fx = tp.fx_rates(lambda sym: (_ for _ in ()).throw(RuntimeError("inget nät")))
    assert fx["USD"] == 1.0 and abs(fx["GBX"] - fx["GBP"] / 100) < 1e-12


def test_theme_members_give_new_companies_a_theme_without_touching_the_panels_own():
    from berserk import themes as bth
    from thor.ticker import theme_for
    from universe import theme_members as tm
    assert list(tm.THEME_MEMBERS) == tp.map_themes() and set(tm.THEME_MEMBERS) <= set(bth.THEMES)
    assert all(len(ts) <= tp.TOP_N and len(set(ts)) == len(ts) for ts in tm.THEME_MEMBERS.values())
    assert theme_for("GRNG.ST")[0] == "aluminium" and theme_for("SBUX")[0] == "kaffe"
    assert theme_for("ADM")[0] == "majs" and theme_for("BOL.ST")[0] == "koppar"        # panelens tema vinner
    assert tm.themes_of("ADM") == ("vete", "majs", "soja", "socker") and tm.theme_of("AGRO") == "socker"
    assert all(t not in tm.tickers() for t in ("ARR", "PROT.OL", "SMHN.DE"))           # felträffar i proben


def test_register_lists_carry_the_theme_lists():
    from universe import registry as ur
    mem = {m.ticker: m for m in ur.from_lists() if "Temalistor" in m.sources}
    assert mem["SBUX"].theme == "kaffe" and mem["VALT.L"].market == ur.LONDON and mem["PDN.TO"].theme == "uran"
