"""
asymmetry/fetch.py — fyll arket utan handskrift. Rena funktioner.

  från registret   öppna positioner (positions.py) → bolagsskal med ticker,
                   namn och strategi-tagg
  ur Börsdata      sifferuppdateringens rad 'confidence:<TICKER>' → förslag
                   (engines/durrett/refresh, samma ark)
"""

from __future__ import annotations

from typing import Optional

from confidence import store as cs
from confidence.data.models import CompanyInput
from engines.durrett import refresh as dr

from asymmetry import store as ast


# ── registret ────────────────────────────────────────────────────────────────
def register_candidates(rows: list, data: dict) -> list:
    """[(ticker, namn, strategi)] ur registrets rader som inte redan finns i arket.
    En ticker en gång, första raden vinner."""
    have = set(cs.companies(data))
    seen, out = set(), []
    for r in rows or []:
        t = str(r.get("ticker") or "").strip().upper()
        if not t or t in have or t in seen:
            continue
        seen.add(t)
        strat = str(r.get("strategy") or "")
        out.append((t, str(r.get("name") or ""), strat if strat in ast.strategy_tags() else ast.NO_STRATEGY))
    return out


def add_from_register(data: dict, ticker: str, name: str, strategy: str) -> CompanyInput:
    """Skal: identitet fylls i Ark (råvara, stage, land). Inga tal gissas."""
    c = CompanyInput(ticker=ticker.strip().upper(), name=name)
    cs.put(data, c)
    ast.set_strategy(data, c.ticker, strategy)
    return c


# ── Börsdata (sifferuppdateringen) ───────────────────────────────────────────
def refresh_row(blob: Optional[dict], ticker: str) -> Optional[dict]:
    return dr.refresh_row(blob, ticker)


def refresh_proposals(blob: Optional[dict], company: CompanyInput) -> list:
    """[(fältnyckel, Datapoint, nuvarande värde | None)] — bara avvikande tal."""
    return dr.proposals(blob, company)


def _borsdata_answer(company: CompanyInput, api=None) -> tuple:
    """borsdata_now och om Börsdata svarade: (blob, förslag, meddelande, svarade). svarade=False när klienten,
    instrumentlistan, raden eller den globala listan kastade (refresh_ui.borsdata_row) — då ska ingen Yahoo-reserv
    provas. blob är None både då och när Börsdata svarade men inte känner tickern."""
    from datetime import datetime, timezone

    import refresh_ui
    import sheets_refresh as sr

    row, err = refresh_ui.borsdata_row(company.ticker, company.ins_id, api)
    if err:
        return None, [], err, False                    # 'Börsdata svarade inte: …', utan nyckeln i adressen
    if row is None:
        return None, [], f"Börsdata känner inte {company.ticker}.", True
    blob = {"generated": datetime.now(tz=timezone.utc).isoformat(), "rows": {f"{sr.ref_key('confidence', {'id': company.ticker})}": row}}
    props = dr.proposals(blob, company)
    return blob, props, (f"{len(props)} tal ur Börsdata" + (f" (kurs {row['price']:g} {row.get('currency') or ''})"
                                                            if row.get("price") is not None else "")), True


def borsdata_now(company: CompanyInput, api=None) -> tuple:
    """Hämta bolagets rad från Börsdata direkt (samma rad som det nattliga
    jobbet) och översätt till förslag. Returnerar (blob, förslag, meddelande);
    blob är None när Börsdata inte svarade (meddelandet 'Börsdata svarade
    inte: …') eller inte känner tickern."""
    blob, props, msg, _answered = _borsdata_answer(company, api)
    return blob, props, msg


# ── Yahoo som reserv (bolag utanför Börsdata: NYSE, TSX, ASX …) ──────────────
YAHOO_SUFFIXES = ("", ".TO", ".V", ".AX", ".L", ".CN")


def yahoo_row(ticker: str, info_getter=None) -> Optional[dict]:
    """Samma radform som Börsdata ur Yahoo: kurs, valuta, börsvärde, aktier,
    kassa, skuld i MUSD. Provar tickern som den är och med vanliga
    börssuffix. None när Yahoo inte har börsvärde för någon form."""
    import lukacs as lk
    import sheets_refresh as sr
    from datetime import date as _date

    t = str(ticker or "").strip().upper()
    if not t:
        return None
    if info_getter is None:
        def info_getter(sym):
            import yfinance as yf
            return yf.Ticker(sym).info or {}
    forms = [t] + [t + sfx for sfx in YAHOO_SUFFIXES[1:] if "." not in t]
    for sym in forms:
        try:
            info = info_getter(sym) or {}
        except Exception:
            continue
        mc = info.get("marketCap")
        if not mc:
            continue
        raw_ccy = str(info.get("currency") or "USD")
        pence = lk._is_pence(raw_ccy)                 # London: kursen i pence (GBp/GBX), börsvärdet i GBP
        ccy = "GBP" if pence else raw_ccy.upper()
        fx = sr.fx_rate(ccy)                          # None för en valuta utanför tabellen — inte 1,0
        fccy = str(info.get("financialCurrency") or ccy).upper()
        ffx = sr.fx_rate(fccy)

        def musd(v, f):
            try:
                return round(float(v) * f / 1e6, 1) if v is not None and f is not None else None
            except (TypeError, ValueError):
                return None

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if pence and price is not None:
            price = float(price) / 100.0               # pence → GBP, så att kurs, valuta och växelkurs stämmer
        shares = info.get("sharesOutstanding")
        name = str(info.get("longName") or info.get("shortName") or "").strip() or None
        return {"ticker": t, "yahoo": sym, "yahoo_name": name, "ins_id": None, "source": "yfinance",
                "source_label": f"Yahoo Finance ({sym}, {_date.today().isoformat()})",
                "price": float(price) if price is not None else None, "asof": _date.today().isoformat(),
                "currency": ccy, "mcap_musd": musd(mc, fx), "cash_musd": musd(info.get("totalCash"), ffx),
                "debt_musd": musd(info.get("totalDebt"), ffx),
                "shares_now_m": round(float(shares) / 1e6, 2) if shares else None,
                "fx_to_usd": fx, "fx_table": "sheets_refresh.FX_TO_USD (fast tabell)",
                "ev_ebitda": info.get("enterpriseToEbitda"), "pe": info.get("trailingPE")}
    return None


def fetch_now(company: CompanyInput, api=None, info_getter=None) -> tuple:
    """Börsdata först, Yahoo som reserv när Börsdata svarade men inte känner tickern. (blob, förslag, meddelande).
    Svarar Börsdata inte (kvoten, nätet, nyckeln) provas inte Yahoo: den söker på bara tickern och kan träffa ett
    annat bolag."""
    blob, props, msg, answered = _borsdata_answer(company, api)
    if blob is not None or not answered:
        return blob, props, msg
    row = yahoo_row(company.ticker, info_getter)
    if row is None:
        return None, [], (msg + f" Yahoo har inte heller {company.ticker} — skriv börsvärde, kassa och skuld själv "
                          "nedan, eller ange Börsdata-id under Identitet.")
    import sheets_refresh as sr
    from datetime import datetime, timezone
    yblob = {"generated": datetime.now(tz=timezone.utc).isoformat(),
             "rows": {sr.ref_key("confidence", {"id": company.ticker}): row}}
    yprops = dr.proposals(yblob, company)
    return yblob, yprops, (f"{msg} Yahoo ({row['yahoo']}): {len(yprops)} tal"
                          + (f" (kurs {row['price']:g} {row['currency']})" if row.get("price") is not None else ""))


def fetch_to_session(company: CompanyInput, api=None, info_getter=None) -> tuple:
    """'Hämta nu' på Wolf Asymmetry- och Durrett-arket: samma väg som de andra arkens knapp (refresh_ui.fetch_live) —
    Börsdata först, Yahoo bara när Börsdata svarade men inte känner tickern — och raden in i sessionens överlagring
    under confidence:<TICKER>, som båda arkens förslag läser före nattens blob. Svarar Börsdata inte (kvoten, nätet,
    nyckeln) provas inte Yahoo, inte heller med ins_id: nattens rad och en tidigare hämtning står kvar. Ingenting
    skrivs i bolaget; det gör Använd. (True, meddelande) när en rad hämtades, annars (False, varför)."""
    import refresh_ui
    t = company.ticker
    row, why = refresh_ui.fetch_live("confidence", {"id": t, "ticker": t, "ins_id": company.ins_id,
                                                    "name": company.name},
                                     row_id=t, api=api, info_getter=info_getter)
    if row is None:
        return False, why
    n = len(dr.proposals({"rows": {f"confidence:{t}": row}}, company))
    px, ccy = row.get("price"), str(row.get("currency") or "")
    kurs = (f" (kurs {px:g}{' ' + ccy if ccy else ''})"
            if isinstance(px, (int, float)) and not isinstance(px, bool) else "")
    return True, (f"{why} " if why else "") + f"{n} tal ur {refresh_ui.source_name(row)}{kurs}"
