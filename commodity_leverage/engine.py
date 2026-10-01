"""
commodity_leverage/engine.py — ett bolags hävstång mot sin råvara.

Två mått, sida vid sida:
  Resultathävstång  Snabbkollens motor (asymmetry.quick_leverage): bolagets
                    egen EBITDA-/FCF-linje mot råvarupriset → 0–10, break-even,
                    nedsida. Kräver årsrapporter med resultat.
  Kurshävstång      commodity_leverage.beta: aktiens veckoavkastning mot
                    råvarans → beta, upp/ned. Kräver bara kurser.

Ingen egen räknelogik för resultat eller värdering — allt går genom
asymmetry.quick_data / quick_leverage / quick_scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from commodity_leverage import beta as cb
from commodity_leverage import config as cc


@dataclass
class CompanyLeverage:
    ticker: str
    name: str = ""
    commodity: str = ""
    locked: bool = False               # råvaran vald manuellt
    data: dict = field(default_factory=dict)
    lev: object = None                 # asymmetry.quick_leverage.LeverageEstimate
    eng: object = None                 # asymmetry.quick_scenarios.EngineResult
    beta: Optional[cb.StockBeta] = None
    beta_error: Optional[str] = None


def _series_default(sym: str, period: str):
    from market_prices import close
    return close(sym, period)


def analyze(ticker: str, commodity: Optional[str] = None, fetcher: Optional[Callable] = None,
            series_getter: Optional[Callable] = None) -> CompanyLeverage:
    from asymmetry import quick_config as qc
    from asymmetry import quick_leverage as ql
    from asymmetry import quick_scenarios as qs
    if fetcher is None:
        from asymmetry import quick_data
        fetcher = quick_data.fetch
    series_getter = series_getter or _series_default
    t = str(ticker or "").strip().upper()
    kw = {"series_getter": series_getter}
    if commodity:
        kw["theme_getter"] = lambda s: commodity
    d = fetcher(t, **kw)
    lev = ql.from_data(d)
    out = CompanyLeverage(t, d.get("name") or t, lev.commodity or (commodity or ""), bool(commodity), d, lev,
                          qs.run(d, lev))
    ctick = (d.get("commodity_px") or {}).get("ticker") or qc.LEV_PRICE_TICKERS.get(out.commodity, "")
    if not ctick:
        out.beta_error = (f"ingen prisserie för {out.commodity} på Yahoo" if out.commodity
                          else "råvaran okänd — välj den manuellt")
        return out
    try:
        stock = series_getter(d.get("yf_ticker") or t, cc.BETA_PERIOD)
        com = series_getter(ctick, cc.BETA_PERIOD)
    except Exception as exc:                                       # pragma: no cover
        out.beta_error = f"kurserna kunde inte hämtas ({exc})"
        return out
    out.beta, out.beta_error = cb.stock_beta(stock, com)
    return out


def rank_key(c: CompanyLeverage) -> tuple:
    """Högst resultathävstång först, sedan kursbeta. Saknat sist."""
    score = c.lev.score if c.lev is not None and c.lev.score is not None else -1
    beta = c.beta.beta if c.beta is not None else -99.0
    return (-score, -beta, c.ticker)


def parse_tickers(raw: str) -> list:
    seen, out = set(), []
    for part in str(raw or "").replace(";", ",").replace("\n", ",").split(","):
        t = part.strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out[:cc.MAX_TICKERS]
