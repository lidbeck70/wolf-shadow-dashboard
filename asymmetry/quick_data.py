"""
asymmetry/quick_data.py — hämtar allt Snabbkollen behöver för en ticker,
helt automatiskt. Börsdata först (nordiskt, sedan globalt: nyckeltal,
årsrapporter, KPI-historik), Yahoo som reserv för saknade fält och för
kursen. Resultatet är en platt dict som asymmetry/quick.score läser, plus
serierna till graferna. Inget skrivs någonstans.

Belopp är i miljoner i bolagets rapportvaluta; bara kvoter och procent
jämförs, så valutan spelar ingen roll för poängen.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from asymmetry import quick_config as qc

logger = logging.getLogger(__name__)

# Fälten poängen läser — täckningen räknas på dem
SCORED_FIELDS: tuple = ("fcf", "nd_ebitda", "current_ratio", "shares_growth_3y_pct", "equity_ratio_pct",
                        "ebitda_margin_pct", "ev_ebitda", "p_fcf", "earnings_stability", "fcf_stability",
                        "fcf_positive_share", "cash_conversion", "vs_sma200_pct")
# Operativt kassaflöde i Börsdatas årsrapport (camelCase efter aliasningen i #102)
OCF_KEYS: tuple = ("cashFlowFromOperatingActivities", "operatingCashFlow")


def _n(v) -> Optional[float]:
    try:
        return None if v is None or v != v else float(v)
    except (TypeError, ValueError):
        return None


def _api_default():
    try:
        from borsdata_api import BorsdataAPI
        api = BorsdataAPI()
        return api if getattr(api, "is_configured", False) else None
    except Exception as exc:                            # pragma: no cover
        logger.debug("BorsdataAPI: %s", exc)
        return None


def _prices_default(sym: str):
    import yfinance as yf
    h = yf.Ticker(sym).history(period="2y", auto_adjust=True)
    return None if h is None or h.empty else h["Close"].dropna()


def _info_default(sym: str) -> dict:
    from contrarian_alpha.engine import yahoo_info
    return yahoo_info(sym)


def _meta(api, iid: int) -> tuple:
    """(instrument, scope) ur den nordiska listan, annars den globala."""
    for scope, getter in (("nordic", "get_instruments"), ("global", "get_global_instruments_list")):
        try:
            for inst in getattr(api, getter)() or []:
                if inst.get("insId") == iid:
                    return inst, scope
        except Exception as exc:
            logger.debug("%s: %s", getter, exc)
    return {}, "nordic"


def _kpi_hist(api, iid: int, kpi_id: int) -> list:
    """[(år, värde)] äldst först."""
    try:
        raw = api.get_kpi_history(iid, kpi_id, "year", "mean") or []
    except Exception as exc:
        logger.debug("kpi_history %s/%s: %s", iid, kpi_id, exc)
        return []
    return [(int(r.get("y")), float(r["v"])) for r in raw if r.get("v") is not None and r.get("y") is not None]


# Börsdatas screener (last/latest) lämnar ibland de här tomma — då läses det
# senaste värdet ur bolagets egen KPI-historik. (fält, KPI-id)
HISTORY_FALLBACK: tuple = (("earnings_stability", 174), ("fcf_stability", 179), ("f_score", 167),
                           ("current_ratio", 44))


def _kpi_latest(api, iid: int, kpi_id: int) -> tuple:
    """(senaste värdet, rapporttyp) ur KPI-historiken, år först och sedan r12."""
    for rt in ("year", "r12"):
        try:
            raw = api.get_kpi_history(iid, kpi_id, rt, "mean") or []
        except Exception as exc:
            logger.debug("kpi_history %s/%s/%s: %s", iid, kpi_id, rt, exc)
            continue
        rows = [r for r in raw if isinstance(r, dict) and _n(r.get("v")) is not None]
        if rows:
            last = max(rows, key=lambda r: (int(r.get("y") or 0), int(r.get("p") or 0)))
            return float(last["v"]), rt
    return None, None


def trend_stability(values: list) -> Optional[float]:
    """Stabilitet 0–1 som Börsdatas: R² för en rak trendlinje genom årsvärdena.
    Fallande trend ger 0 — stadigt krympande är inte stabilt. None vid för få år."""
    ys = [float(v) for v in values if _n(v) is not None]
    n = len(ys)
    if n < qc.STABILITY_MIN_YEARS:
        return None
    xm, ym = (n - 1) / 2.0, sum(ys) / n
    sxx = sum((i - xm) ** 2 for i in range(n))
    sxy = sum((i - xm) * (y - ym) for i, y in enumerate(ys))
    syy = sum((y - ym) ** 2 for y in ys)
    if syy == 0:
        return 1.0 if ym > 0 else 0.0
    if sxy <= 0:
        return 0.0
    return round(sxy * sxy / (sxx * syy), 2)


def fcf_positive_share(values: list) -> Optional[float]:
    """Andel år med positivt fritt kassaflöde. None vid för få år."""
    ys = [x for x in (_n(v) for v in values) if x is not None]
    if len(ys) < qc.QUALITY_MIN_YEARS:
        return None
    return round(sum(1 for y in ys if y > 0) / len(ys), 2)


def cash_conversion(ocf: list, profit: list) -> Optional[float]:
    """Operativt kassaflöde / nettoresultat, summerat över år där båda finns.
    Summor i stället för årskvoter — ett år med resultat nära noll ger annars
    orimliga tal. None vid för få år eller summerad förlust (inte meningsfullt)."""
    pairs = [(o, p) for o, p in ((_n(a), _n(b)) for a, b in zip(ocf, profit)) if o is not None and p is not None]
    if len(pairs) < qc.QUALITY_MIN_YEARS:
        return None
    tot_p = sum(p for _, p in pairs)
    if tot_p <= 0:
        return None
    return round(sum(o for o, _ in pairs) / tot_p, 2)


def _price_stats(closes) -> dict:
    out: dict = {"prices": [], "sma200": []}
    if closes is None or len(closes) == 0:
        return out
    vals = [float(x) for x in closes]
    idx = [str(i)[:10] for i in getattr(closes, "index", range(len(vals)))]
    last = vals[-1]
    hi = max(vals[-252:])
    out["price"] = round(last, 4)
    out["from_52w_high_pct"] = round((1 - last / hi) * 100, 1) if hi > 0 else None
    sma = []
    for i in range(len(vals)):
        sma.append(round(sum(vals[i - 199:i + 1]) / 200, 4) if i >= 199 else None)
    if sma[-1]:
        out["vs_sma200_pct"] = round((last / sma[-1] - 1) * 100, 1)
    out["prices"] = list(zip(idx, vals))
    out["sma200"] = list(zip(idx, sma))
    return out


def _annual_mean(series) -> dict:
    """{år: årssnitt} ur en daglig kursserie (pandas Series med datumindex)."""
    if series is None or len(series) == 0:
        return {}
    try:
        g = series.groupby(series.index.year).mean()
        return {int(y): float(v) for y, v in g.items() if v == v}
    except Exception:
        return {}


def _series_default(sym: str, period: str = "10y"):
    from market_prices import close
    return close(sym, period)


def _theme_default(sym: str) -> Optional[str]:
    try:
        from ember.regime import detect_theme
        return detect_theme(sym)
    except Exception as exc:
        logger.debug("detect_theme %s: %s", sym, exc)
        return None


def commodity_prices(theme: Optional[str], currency: Optional[str], series_getter: Callable) -> dict:
    """Råvarans årssnitt och dagspris i rapportvalutan (Commodity Leverage).
    {commodity, ticker, prices: {år: pris}, p0, fx_now} — tomt pris när serien saknas."""
    ticker = qc.LEV_PRICE_TICKERS.get(theme or "", "")
    out = {"commodity": theme or "", "ticker": ticker, "prices": {}, "p0": None, "fx_now": 1.0}
    if not ticker:
        return out
    try:
        px = series_getter(ticker, "10y")
    except Exception as exc:
        logger.debug("pris %s: %s", ticker, exc)
        return out
    if px is None or len(px) == 0:
        return out
    ccy = str(currency or "USD").upper()
    fx_year, fx_now = {}, 1.0
    if ccy != "USD":
        try:
            fxs = series_getter(f"USD{ccy}=X", "10y")
        except Exception:
            fxs = None
        if fxs is None or len(fxs) == 0:
            return out                              # utan valutakurs blir sambandet fel — hellre inget
        fx_year, fx_now = _annual_mean(fxs), float(fxs.iloc[-1])
    yearly = _annual_mean(px)
    out["prices"] = {y: v * fx_year.get(y, fx_now if not fx_year else None) for y, v in yearly.items()
                     if not fx_year or y in fx_year}
    out["p0"] = float(px.iloc[-1]) * fx_now
    out["fx_now"] = fx_now
    return out


def fetch(ticker: str, api=None, price_getter: Optional[Callable] = None,
          info_getter: Optional[Callable] = None, use_api_default: bool = True,
          series_getter: Optional[Callable] = None, theme_getter: Optional[Callable] = None) -> dict:
    """Allt för Snabbkollen. Nycklar som inte gick att få fram saknas eller är None."""
    import markets
    import sheets_refresh as sr

    t = str(ticker or "").strip().upper()
    d: dict = {"ticker": t, "name": t, "source": "", "filled_yahoo": [], "kpi_source": {},
               "fetched": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    if not t:
        return d
    if api is None and use_api_default:
        api = _api_default()
    price_getter = price_getter or _prices_default
    info_getter = info_getter or _info_default
    yf_sym = t

    # ── Börsdata ─────────────────────────────────────────────────────────────
    snap: dict = {}
    if api is not None:
        try:
            iid = sr.resolve(api, t)
        except Exception as exc:
            logger.debug("resolve %s: %s", t, exc)
            iid = None
        if iid is not None:
            inst, scope = _meta(api, iid)
            d.update(ins_id=iid, name=inst.get("name") or t,
                     source="Börsdata" + (" global" if scope == "global" else ""),
                     currency=inst.get("reportCurrency") or inst.get("stockPriceCurrency"))
            if inst.get("ticker") and inst.get("marketId") is not None:
                try:
                    yf_sym = markets.to_yf(inst["ticker"], inst["marketId"], markets.current()) or t
                except Exception:
                    yf_sym = t
            try:
                snap = (api.get_fundamentals_snapshot_fast([iid], scope=scope) or {}).get(iid) or {}
            except Exception as exc:
                logger.debug("snapshot %s: %s", iid, exc)
            try:
                years = sorted((r for r in api.get_reports(iid, "year", max_count=10) or []
                                if isinstance(r, dict) and r.get("year")), key=lambda r: r["year"])
            except Exception as exc:
                logger.debug("reports year %s: %s", iid, exc)
                years = []
            try:
                r12 = (api.get_reports(iid, "r12", max_count=1) or [None])[0]
            except Exception:
                r12 = None
            latest = r12 if isinstance(r12, dict) else (years[-1] if years else {})
            d["report_years"] = len(years)
            d["fcf"] = _n(latest.get("freeCashFlow")) if latest else None
            d["cash"] = _n(latest.get("cashAndEquivalents")) if latest else None
            d["net_debt"] = _n(latest.get("netDebt")) if latest else None
            d["fcf_series"] = [(int(r["year"]), _n(r.get("freeCashFlow"))) for r in years]
            d["shares_series"] = [(int(r["year"]), _n(r.get("numberOfShares"))) for r in years]
            by_year = {y: s for y, s in d["shares_series"] if s}
            if by_year:
                y_now = max(by_year)
                if by_year.get(y_now - 3):
                    d["shares_growth_3y_pct"] = round((by_year[y_now] / by_year[y_now - 3] - 1) * 100, 1)
            last_year = years[-1] if years else {}
            ta, eq = _n(last_year.get("totalAssets")), _n(last_year.get("totalEquity"))
            er = _n(snap.get("equity_ratio"))
            d["equity_ratio_pct"] = round(er * 100, 1) if er is not None else (
                round(eq / ta * 100, 1) if ta and eq is not None else None)
            em = _n(snap.get("ebitda_margin"))
            d["ebitda_margin_pct"] = round(em * 100, 1) if em is not None else None
            for key in ("current_ratio", "earnings_stability", "fcf_stability", "f_score", "ev_ebitda", "p_fcf"):
                d[key] = _n(snap.get(key))
            d["nd_ebitda"] = _n(snap.get("net_debt_ebitda"))
            if d.get("fcf") is None:
                d["fcf"] = _n(snap.get("fcf_m"))
            if d.get("net_debt") is None:
                d["net_debt"] = _n(snap.get("net_debt_m"))
            for key, kpi_id in HISTORY_FALLBACK:
                if d.get(key) is None:
                    v, rt = _kpi_latest(api, iid, kpi_id)
                    if v is not None:
                        d[key] = v
                        d["kpi_source"][key] = f"Börsdata historik ({'år' if rt == 'year' else 'r12'})"
            earn = [_n(r.get("profitToEquityHolders")) for r in years]
            ocf = [next((_n(r.get(k)) for k in OCF_KEYS if _n(r.get(k)) is not None), None) for r in years]
            fcf_vals = [v for _, v in d["fcf_series"]]
            d["quality_years"] = sum(1 for v in fcf_vals if v is not None)
            d["fcf_positive_share"] = fcf_positive_share(fcf_vals)
            d["cash_conversion"] = cash_conversion(ocf, earn)
            if d["cash_conversion"] is None and sum(1 for p in earn if p is not None) >= qc.QUALITY_MIN_YEARS \
                    and sum(p for p in earn if p is not None) <= 0:
                d["cash_conversion_note"] = "summerad förlust — kvoten är inte meningsfull"
            for key, series in (("earnings_stability", earn), ("fcf_stability", fcf_vals)):
                if d.get(key) is None:
                    v = trend_stability(series)
                    if v is not None:
                        d[key] = v
                        d["kpi_source"][key] = "beräknad ur årsrapporterna"
            d["ev_ebitda_series"] = _kpi_hist(api, iid, 11)
            d["p_fcf_series"] = _kpi_hist(api, iid, 76)
            d["ev_ebitda_hist"] = [v for _, v in d["ev_ebitda_series"]]
            d["p_fcf_hist"] = [v for _, v in d["p_fcf_series"]]
            d["mcap_bd"] = _n(snap.get("market_cap"))
            # Commodity Leverage: omsättning och EBITDA per år (KPI 53/54, rapportvalutan)
            d["revenue_series"] = _kpi_hist(api, iid, 53) or [
                (int(r["year"]), _n(r.get("revenues"))) for r in years if _n(r.get("revenues")) is not None]
            d["ebitda_series"] = _kpi_hist(api, iid, 54)
            if not d["ebitda_series"] and d["revenue_series"]:
                # reserv: EBITDA-marginal (KPI 32, %) × omsättning samma år
                marg = dict(_kpi_hist(api, iid, 32))
                d["ebitda_series"] = [(y, round(r * marg[y] / 100, 1)) for y, r in d["revenue_series"]
                                      if r is not None and y in marg]

    # ── Yahoo: reserv för luckor + jämförelse ────────────────────────────────
    try:
        info = info_getter(yf_sym) or {}
    except Exception as exc:
        logger.debug("yahoo info %s: %s", yf_sym, exc)
        info = {}
    if info:
        if not d.get("source"):
            d["source"] = "Yahoo"
            d["name"] = info.get("longName") or info.get("shortName") or t
            d["currency"] = info.get("financialCurrency") or info.get("currency")
        ebitda = _n(info.get("ebitda"))
        # FCF-data: Börsdata mot Yahoo för senaste 12 mån — bara i samma valuta,
        # och bara när Börsdata faktiskt hade ett tal (innan Yahoo fyller luckan).
        y_fcf = _n(info.get("freeCashflow"))
        bd_fcf = _n(d.get("fcf"))
        same_ccy = str(info.get("financialCurrency") or "").upper() == str(d.get("currency") or "").upper() != ""
        if bd_fcf is not None and y_fcf is not None and same_ccy:
            denom = max(abs(bd_fcf), abs(y_fcf / 1e6))
            if denom > 0:
                d["fcf_source_gap_pct"] = round((y_fcf / 1e6 - bd_fcf) / denom * 100, 1)
        fills = {
            "fcf": (_n(info.get("freeCashflow")) / 1e6) if _n(info.get("freeCashflow")) is not None else None,
            "cash": (_n(info.get("totalCash")) / 1e6) if _n(info.get("totalCash")) is not None else None,
            "current_ratio": _n(info.get("currentRatio")),
            "ebitda_margin_pct": (_n(info.get("ebitdaMargins")) * 100) if _n(info.get("ebitdaMargins")) is not None else None,
            "ev_ebitda": _n(info.get("enterpriseToEbitda")) if ebitda and ebitda > 0 else None,
            "nd_ebitda": (((_n(info.get("totalDebt")) or 0.0) - (_n(info.get("totalCash")) or 0.0)) / ebitda)
            if ebitda and ebitda > 0 and info.get("totalDebt") is not None else None,
            "net_debt": (((_n(info.get("totalDebt")) or 0.0) - (_n(info.get("totalCash")) or 0.0)) / 1e6)
            if info.get("totalDebt") is not None or info.get("totalCash") is not None else None,
        }
        for key, v in fills.items():
            if d.get(key) is None and v is not None:
                d[key] = round(v, 4)
                d["filled_yahoo"].append(key)
        mc_y = _n(info.get("marketCap"))
        if d.get("mcap_bd") and mc_y:
            d["source_gap_pct"] = round((d["mcap_bd"] / (mc_y / 1e6) - 1) * 100, 1)
        pf = _n(d.get("p_fcf"))
        if pf and pf > 0:
            d["fcf_yield_pct"] = round(100.0 / pf, 1)
        elif _n(info.get("freeCashflow")) is not None and mc_y:
            fx = sr.FX_TO_USD.get(str(info.get("financialCurrency") or "USD").upper(), 1.0)
            fx_mc = sr.FX_TO_USD.get(str(info.get("currency") or "USD").upper(), 1.0)
            d["fcf_yield_pct"] = round(_n(info["freeCashflow"]) * fx / (mc_y * fx_mc) * 100, 1)
    elif _n(d.get("p_fcf")) and d["p_fcf"] > 0:
        d["fcf_yield_pct"] = round(100.0 / d["p_fcf"], 1)

    # ── Kurs ─────────────────────────────────────────────────────────────────
    try:
        closes = price_getter(yf_sym)
    except Exception as exc:
        logger.debug("prices %s: %s", yf_sym, exc)
        closes = None
    d.update(_price_stats(closes))
    d["yf_ticker"] = yf_sym

    # ── Råvaran (Commodity Leverage, utanför 300) ────────────────────────────
    theme = (theme_getter or _theme_default)(yf_sym)
    d["commodity_px"] = commodity_prices(theme, d.get("currency"), series_getter or _series_default)

    have = sum(1 for k in SCORED_FIELDS if _n(d.get(k)) is not None)
    d["coverage"] = (have, len(SCORED_FIELDS))
    if not d.get("source"):
        d["source"] = "—"
    return d
