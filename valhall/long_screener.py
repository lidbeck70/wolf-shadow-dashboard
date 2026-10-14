"""
valhall/long_screener.py — Long Screener (SCREENING → Contrarian Alpha → Long Screener, cagr/) i Valhall.

Screenern (Kvalitetsaktiepodden-inspirerad: vinstacceleration, ROE, F-score, relativ styrka, finansiell
styrka + trend) körs OFÖRÄNDRAD: cagr_fundamentals.score_fundamentals (Börsdata, 0–20),
cagr_technical.score_technical (0–7) och cagr_scoring.calculate_total_score (signal och grindar).

Bara cykelpoängen (0–3) byts: i panelen är den ett manuellt reglage per sektor (förval i
cagr_cycle.SECTOR_CONFIG). Här räknas den ur Valhall, punkt för punkt:

  marknaden  hemmamarknadens största index (OMXS30, OBX, OMXC25, OMXH25 — samma val som Alpha Regime)
             GRÖN = kurs > EMA200 och EMA50 > EMA200 · RÖD = under båda · annars GUL
  sektorn    Valhalls nordiska sektorindex (status GRÖN/GUL/RÖD)

Varianten "gron" (vald efter backtestet, fas 4c 2008–idag: CAGR 12,3 % och max drawdown −18 % mot
8,7 % / −31 % för "valhall" — köp i GUL sektor var svaga i alla varianter):

  marknaden RÖD                → 0  (screenerns regel 7: röd regim = sälj)
  sektorn GRÖN                 → 2, +1 om marknaden är GRÖN
  sektorn GUL / RÖD / saknas   → 1  (inga köp, men tvingar inte fram sälj)

Screenerns köpgrind kräver cykel ≥ 2, alltså GRÖN sektor. "valhall" (första versionen) gav dessutom
GUL sektor med grön marknad 2 — den finns kvar för backtestet. Parallellt räknas screenern UTAN cykel
(alltid 3) för papperskontot "Long Screener (utan cykel)" — verkligheten får bekräfta valet.

Universumet är Valhalls likvida nordiska aktier (screenerns egen lista är 50 storbolag — de märks
in_list). Insiderkriteriet (1 av 20 poäng) räknas inte här (en insynshämtning per aktie är för dyr);
det sänker alla lika mycket.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from valhall import cycle
from valhall import segments as sgm

N_PICKS = 2
BUY_SIGNALS = ("STRONG BUY", "BUY")
SELL_SIGNALS = ("SELL", "STRONG SELL")
TECH_BARS = 600                    # EMA200 + 52 veckor räcker


BD_INDEX = {"OBX.OL": ("OBX", "OBX Index", "OBX Total Return Index")}   # Yahoo ger inte längre OBX.OL-historik


def index_series(sym: str, yahoo, api=None, bars: int = 5040) -> Optional[pd.Series]:
    """Hemmamarknadens index: Börsdata för OBX (market_risk.borsdata_index, samma uppslag som
    marknadsrisken), annars Yahoo. yahoo(sym) → Close-serie eller None."""
    if sym in BD_INDEX and api is not None:
        try:
            import market_risk as mr
            s, _src = mr.borsdata_index(api, BD_INDEX[sym], bars)
            if s is not None and len(s):
                if getattr(s.index, "tz", None) is not None:
                    s.index = s.index.tz_localize(None)
                return s
        except Exception:
            pass
    try:
        return yahoo(sym)
    except Exception:
        return None


def market_index(ticker: str) -> str:
    from alpha_regime.quality_scan import benchmark_for
    return benchmark_for(ticker)


def market_regime(close: Optional[pd.Series]) -> str:
    """GRÖN/GUL/RÖD för ett index ur EMA50/EMA200 (sista dagen)."""
    if close is None or len(close.dropna()) < 200:
        return cycle.NODATA
    c = close.dropna().astype(float)
    e50, e200 = c.ewm(span=50, adjust=False).mean().iloc[-1], c.ewm(span=200, adjust=False).mean().iloc[-1]
    last = c.iloc[-1]
    if last > e200 and e50 > e200:
        return cycle.GRON
    if last < e200 and e50 < e200:
        return cycle.ROD
    return cycle.GUL


NO_CYCLE = 3                       # "utan cykel": bara bolagets egna kriterier och trendgrindar


def cycle_score(sector_status: str, market: str, variant: str = "gron") -> int:
    """Valhalls läge → screenerns cykelpoäng 0–3 (se modulens docstring). variant "gron" (daglig) eller
    "valhall" (första versionen: GUL sektor med grön marknad köper också)."""
    if market == cycle.ROD:
        return 0
    bonus = 1 if market == cycle.GRON else 0
    if sector_status == cycle.GRON:
        return 2 + bonus
    if sector_status == cycle.GUL and variant == "valhall":
        return 1 + bonus
    return 1


def parts(close: pd.Series, fund_info: dict) -> tuple:
    """(fundamenta, teknik) — screenerns delpoäng, räknade en gång per aktie."""
    from cagr.cagr_fundamentals import score_fundamentals
    from cagr.cagr_technical import score_technical
    return (score_fundamentals(fund_info or {}, None),
            score_technical(pd.DataFrame({"Close": close.astype(float).tail(TECH_BARS)})))


def score(close: pd.Series, fund_info: dict, cyc: int, pre: Optional[tuple] = None) -> dict:
    """Screenerns egen poäng och signal för en aktie (cagr-modulerna oförändrade). pre = parts(...)."""
    from cagr.cagr_cycle import score_cycle_from_int
    from cagr.cagr_scoring import calculate_total_score
    fund, tech = pre or parts(close, fund_info)
    total = calculate_total_score(fund, score_cycle_from_int(cyc), tech)
    return {"signal": total["signal"], "score_pct": total["score_pct"], "total": total["total_score"],
            "max": total["max_score"], "fund": total["fund_score"], "fund_max": total["fund_max"],
            "tech": total["tech_score"], "cycle": total["cycle_score"],
            "source": fund.get("_data_source", (fund_info or {}).get("_data_source"))}


def scan(board: dict, universe: dict, data: dict, tickers: list, fundamentals: Callable,
         index_close: Callable, n: int = N_PICKS) -> dict:
    """{segments: {key: {sector_status, picks, picks_ingen}}, signals / signals_ingen: {ticker: signal},
    markets: {index: status}, scanned}. Kortet = cykelvarianten "gron"; *_ingen = screenern utan cykel
    (papperskontot "Long Screener (utan cykel)"). tickers = de likvida nordiska; fundamentals(tuple) →
    {ticker: info}; index_close(symbol) → Close-serie."""
    from cagr.cagr_loader import NORDIC_TICKERS
    status = {r["key"]: (r.get("nordic") or {}).get("status", cycle.NODATA) for r in (board or {}).get("segments") or []}
    markets: dict = {}
    for t in tickers:
        sym = market_index(t)
        if sym not in markets:
            try:
                markets[sym] = market_regime(index_close(sym))
            except Exception:
                markets[sym] = cycle.NODATA
    fund = fundamentals(tuple(tickers)) if tickers else {}
    rows = []
    for t in tickers:
        df = data.get(t)
        seg = (universe.get(t) or {}).get("segment")
        if df is None or not len(df) or not seg:
            continue
        cyc = cycle_score(status.get(seg, cycle.NODATA), markets.get(market_index(t), cycle.NODATA))
        try:
            pre = parts(df["Close"], fund.get(t) or {})
            s, s0 = score(df["Close"], {}, cyc, pre), score(df["Close"], {}, NO_CYCLE, pre)
        except Exception:
            continue
        base = {"ticker": t, "name": (universe.get(t) or {}).get("name", ""), "segment": seg,
                "in_list": t in NORDIC_TICKERS, "market": market_index(t)}
        rows.append(({**base, **s}, {**base, **s0}))
    out = {"segments": {}, "signals": {r["ticker"]: r["signal"] for r, _ in rows},
           "signals_ingen": {r["ticker"]: r["signal"] for _, r in rows}, "markets": markets, "scanned": len(rows),
           "buys": sum(1 for r, _ in rows if r["signal"] in BUY_SIGNALS),
           "buys_ingen": sum(1 for _, r in rows if r["signal"] in BUY_SIGNALS)}

    def best(sel, key):
        mine = sorted((r for r in sel if r["segment"] == key and r["signal"] in BUY_SIGNALS),
                      key=lambda r: (r["signal"] != "STRONG BUY", -r["score_pct"], r["ticker"]))
        return mine[:n]

    for sg in sgm.SEGMENTS:
        out["segments"][sg.key] = {
            "sector_status": status.get(sg.key, cycle.NODATA),
            "picks": [dict(r, source="Long Screener") for r in best([a for a, _ in rows], sg.key)],
            "picks_ingen": [dict(r, source="Long Screener (utan cykel)") for r in best([b for _, b in rows], sg.key)]}
    return out
