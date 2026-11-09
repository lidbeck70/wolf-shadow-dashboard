"""
engines/durrett/refresh.py — sifferuppdateringens förslag för Durrett-arket.

sheets_refresh.py (GitHub Actions) läser data/confidence.json, hämtar
färska tal ur Börsdata och skriver dem till Gisten (sheets_refresh.json)
under nyckeln "confidence:<TICKER>". Här översätts en sådan rad till
förslag {fältnyckel: Datapoint} — bara sådant som skiljer sig från det
bolaget redan har. Inget skrivs utan Använd. Rena funktioner.
"""

from __future__ import annotations

from typing import Optional

from confidence.config import FIELD_BY_KEY
from confidence.data.models import CompanyInput
from confidence.data.provenance import Datapoint, dp

SHEET = "confidence"
_FX_FIELDS = frozenset({"price", "mcap_musd", "ev_musd", "fx_to_usd"})   # tal som jobbet räknar med kursvalutan
# Valutorna fältet market_currency kan ta (samma lista som valen i arket) — DKK/CHF/PLN ingår, annars lästes en
# dansk kurs som USD bredvid ett börsvärde i USD.
_CCY_CHOICES = tuple(FIELD_BY_KEY["market_currency"].choices)
# Källor vars MUSD-tal redan är omräknade till USD: sifferuppdateringen och håven ("Börsdata …") och
# Asymmetrys Yahoo-reserv (asymmetry/fetch.yahoo_row: "Yahoo Finance (…)"). Allt annat kan stå i lokal valuta.
AUTO_USD_SOURCES = ("Börsdata", "Yahoo Finance")
# (fält i bloben, fält i bolaget, enhet, kind)
_MAP = (("mcap_musd", "market_cap_musd", "MUSD", "ACTUAL"), ("price", "share_price", "valuta/aktie", "ACTUAL"),
        ("ev_musd", "enterprise_value_musd", "MUSD", "ACTUAL"), ("ev_ebitda", "ev_ebitda", "×", "ACTUAL"),
        ("nd_ebitda", "net_debt_ebitda", "×", "ACTUAL"), ("pe", "pe", "×", "ACTUAL"),
        ("revenue_musd", "revenue_musd", "MUSD", "ACTUAL"), ("fcf_musd", "free_cash_flow_musd", "MUSD", "ACTUAL"),
        ("ocf_musd", "operating_cash_flow_musd", "MUSD", "ACTUAL"), ("rs_rank", "rs_rank", "", "ACTUAL"),
        ("ebitda_margin", "ebitda_margin_pct", "%", "ACTUAL"), ("fx_to_usd", "fx_to_usd", "USD per enhet", "ASSUMPTION"),
        # rapportfälten (PR 15): kassa, skuld, aktiehistorik, ROIC, FCF-yield, EV/EBIT
        ("cash_musd", "cash_musd", "MUSD", "ACTUAL"), ("debt_musd", "debt_musd", "MUSD", "ESTIMATE"),
        ("shares_now_m", "basic_shares_m", "M", "ACTUAL"), ("shares_1y_ago_m", "shares_1y_ago_m", "M", "ACTUAL"),
        ("shares_3y_ago_m", "shares_3y_ago_m", "M", "ACTUAL"), ("shares_5y_ago_m", "shares_5y_ago_m", "M", "ACTUAL"),
        ("roic_pct", "roic_pct", "%", "ACTUAL"), ("fcf_yield_pct", "fcf_yield_pct", "%", "ACTUAL"),
        ("ev_ebit", "ev_ebit", "×", "ACTUAL"))
_NOTES = {"debt_musd": "bruttoskuld ≈ nettoskuld + kassa ur senaste rapporten",
          "shares_now_m": "numberOfShares ur senaste årsrapporten (miljoner)"}
_PCT_FIELDS = {"ebitda_margin": 100.0}          # snapshoten ger decimaltal → %


def refresh_row(blob: Optional[dict], ticker: str, sheet: str = SHEET) -> Optional[dict]:
    """Raden ur bloben: 'confidence:<TICKER>' (Durrett-arket) eller 'asymmetry:<TICKER>'."""
    return ((blob or {}).get("rows") or {}).get(f"{sheet}:{(ticker or '').strip().upper()}")


def ins_id_offer(row: Optional[dict], company: CompanyInput) -> Optional[int]:
    """Nattradens Börsdata-id när bolaget saknar ett: jobbet matchade tickern (sheets_refresh.resolve), och sparat
    på bolaget hämtas samma bolag varje natt. None när bolaget redan har ett id eller raden saknar ett."""
    if company.ins_id is not None or not isinstance(row, dict) or isinstance(row.get("ins_id"), bool):
        return None
    try:
        iid = int(row.get("ins_id"))
    except (TypeError, ValueError):
        return None
    return iid if iid > 0 else None


def proposals(blob: Optional[dict], company: CompanyInput, sheet: str = SHEET) -> list:
    """[(fältnyckel, Datapoint, nuvarande värde | None)] — bara avvikande tal."""
    s = refresh_row(blob, company.ticker, sheet)
    if not s:
        return []
    asof = str(s.get("asof") or (blob or {}).get("generated") or "")[:10] or None
    src = s.get("source_label") or f"Börsdata (sifferuppdatering {asof or '?'})"
    out = []
    ccy = str(s.get("currency") or "").upper()
    # En kursvaluta utanför jobbets växelkurstabell (London i pence, 'GBX'): jobbet lämnar börsvärde, EV och
    # växelkurs tomma, men en äldre blob räknade om med 1,0 (miljoner GBX som USD, ~79× för högt) — och kursen står i
    # en valuta arket inte kan välja. De talen föreslås inte alls.
    try:
        import sheets_refresh as _sr
        unknown_ccy = bool(ccy) and ccy not in _sr.FX_TO_USD
    except Exception:
        unknown_ccy = False
    for bkey, fkey, unit, kind in _MAP:
        if unknown_ccy and bkey in _FX_FIELDS:
            continue
        v = s.get(bkey)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if bkey in _PCT_FIELDS:
            v = v * _PCT_FIELDS[bkey]
        # det som sparas: kursen med 6 värdesiffror (en pencekurs i pund, 0,00675, tål inte 4 decimaler), annars 4
        # decimaler — och jämförelsen görs mot samma tal, så ett använt förslag inte dyker upp igen
        v = float(f"{v:.6g}") if bkey == "price" else round(v, 4)
        cur = company.num(fkey) if company.has(fkey) else None
        if cur is not None and abs(cur - v) < 1e-6:
            continue
        note = _NOTES.get(bkey, "")
        source = src
        if bkey == "fx_to_usd":
            source = str(s.get("fx_table") or "sheets_refresh FX-tabell")
            note = f"{ccy} → USD ur jobbets fasta tabell — kontrollera mot dagskurs"
        elif bkey == "price":
            unit = f"{ccy}/aktie" if ccy else unit
        out.append((fkey, dp(v, kind=kind, source=source, source_type="secondary", pub_date=asof,
                             unit=unit, note=note), cur))
    if ccy and (company.get("market_currency") is None or str(company.get("market_currency").value) != ccy) \
            and ccy in _CCY_CHOICES:
        out.append(("market_currency", dp(ccy, kind="ACTUAL", source=src, source_type="secondary", pub_date=asof),
                    str(company.get("market_currency").value) if company.get("market_currency") else None))
    offer = commodity_price_offer(s, company)
    if offer:
        out.append(offer)
    return out


def commodity_price_offer(s: Optional[dict], company: CompanyInput) -> Optional[tuple]:
    """('commodity_price', Datapoint, nuvarande | None) ur nattens terminspris (sheets_refresh: commodity_price på
    raden), eller None. Bara när priset hämtades för bolagets råvara (commodity_prices.night_fits — råvaran kan ha
    bytts sedan natten) och står i registrets enhet (commodity_prices.unit_fits: guld/silver/platina/palladium USD/oz,
    koppar USD/lb, gas USD/MMBtu, järnmalm USD/t; olja i USD/fat mot USD/boe, ett fat = en boe enligt ägarens beslut).
    Råvarupriset står alltid i USD, så kursvalutans regel (okänd valuta → inga omräknade tal) gäller inte det. Uran,
    litium, zink, nickel och kol har ingen termin och får inget. Skiljer sig inte från det sparade → None."""
    if not isinstance(s, dict):
        return None
    try:
        import commodity_prices as cp
        from confidence import commodities as com
        reg = com.get(company.commodity)
        name = cp.registry_name(company.commodity)
        if reg is None or not name or not cp.night_fits(s, name) or not cp.unit_fits(s.get("commodity_unit"), reg.unit):
            return None
        v = round(float(s.get("commodity_price")), 4)
    except Exception:                                   # inget pris, eller en rad som inte går att läsa: inget förslag
        return None
    if not v > 0:
        return None
    cur = company.num("commodity_price") if company.has("commodity_price") else None
    if cur is not None and abs(cur - v) < 1e-6:
        return None
    tick = str(s.get("commodity_ticker") or (cp.ticker_for(name) or ("",))[0])
    night_unit = str(s.get("commodity_unit") or "").strip()
    note = "; ".join(x for x in (
        cp.BASIS.get(tick, ""),
        f"{tick} står i {night_unit} — ett fat räknas som en boe" if night_unit != reg.unit else "",
        "terminspris, inte spot eller realiserat pris — kan vara en dag gammalt") if x)
    asof = str(s.get("commodity_asof") or "")[:10] or None
    return ("commodity_price", dp(v, kind="ASSUMPTION", source=f"Yahoo {tick} (termin)", source_type="secondary",
                                  pub_date=asof, unit=reg.unit, note=note), cur)


def spot_earnings(s: Optional[dict], company: CompanyInput) -> Optional[dict]:
    """Motorns framtida vinst (future_fcf, MUSD) räknad på nattens terminspris i stället för det sparade — bara för
    raden vid Råvarupris, inget skrivs. None när priset inte erbjuds (commodity_price_offer: fel råvara, fel enhet,
    samma tal) eller motorn inte kan räkna. {price, unit, asof, fcf_spot, fcf_saved}; fcf_* None när motorn saknar
    produktion eller kostnad (fcf_saved också utan sparat pris)."""
    offer = commodity_price_offer(s, company)
    if not offer:
        return None
    point = offer[1]
    try:
        from engines.durrett.engine import analyze

        def fcf(c):
            return (analyze(c).metrics.get("future_fcf") or {}).get("value")
        fresh = CompanyInput.from_dict(company.as_dict())
        fresh.set("commodity_price", point)
        out = {"price": point.value, "unit": point.unit, "asof": point.pub_date, "fcf_spot": fcf(fresh),
               "fcf_saved": fcf(company) if company.has("commodity_price") else None}
    except Exception:                                   # raden är en hjälp — motorn får aldrig fälla sidan
        return None
    return out


__all__ = ["proposals", "refresh_row", "ins_id_offer", "commodity_price_offer", "spot_earnings", "Datapoint",
           "AUTO_USD_SOURCES"]
