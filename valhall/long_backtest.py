"""
valhall/long_backtest.py — Long Screener punkt för punkt 2008–idag (utan facit).

Screenerns egen poäng (cagr_fundamentals / cagr_technical / cagr_scoring, oförändrade) räknas varje
månadsskifte på de likvida nordiska aktierna, med nyckeltalen som de SÅG UT då:

  Nyckeltal   Börsdatas årshistorik (samma KPI-id och skalning som cagr_loader.fetch_fundamentals_batch_fast);
              ett år räknas som känt fyra månader efter räkenskapsårets slut (longterm.LAG_MONTHS).
              Pris-KPI:er (P/E, EV/EBITDA, P/B, direktavkastning) är årets snittkurs — en approximation.
  Omsättning  CAGR 5 år ur omsättningshistoriken (KPI 53).
  RS-rank     räknas här: percentilen 1–100 av 6-månadersavkastningen bland de likvida aktierna den dagen
              (Börsdatas RS-rank har ingen historik).
  Magic F.    räknas här: Greenblatts rangsumma av EV/EBIT (KPI 10, lågt bäst) och avkastning på kapital
              (KPI 36, högt bäst), som percentil 1–100 (1 = bäst).
  Insider     uteslutet (1 av 20 poäng) — ingen historik.
  Saknas      F-score, vinst- och FCF-stabilitet har ingen historik i Börsdata-licensen. Poängen räknas då
              mot det som går att nå (t.ex. 16 av 20), så att procentgränserna för BUY/STRONG BUY är
              desamma som i panelen.

Cykelpoängen i fem varianter:
  valhall    första versionen (long_screener.cycle_score "valhall": sektorindex + hemmamarknadens index)
  gron       som valhall, men köp bara när sektorn är GRÖN — den dagliga körningen sedan fas 4c
  gron_utan_marknad  köp bara i GRÖN sektor; hemmamarknaden tvingar aldrig fram sälj
  forval     screenerns statiska förval per sektor (cagr_cycle.SECTOR_CONFIG) — som panelen utan reglage
  ingen      alltid 3 — bara bolagets egna kriterier och trendgrindar

Portföljen följer screenerns regler (cagr_backtest): högst 10 innehav, STRONG BUY 10 % och BUY 7 % av
kapitalet, högst 25 % per sektor, sälj på SELL/STRONG SELL. Månadsvis, kassa = 0 %, utan courtage.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from valhall import cycle
from valhall import long_screener as ls
from valhall import longterm as lt

# fält → (Börsdata KPI-id, skalning) — som cagr_loader.fetch_fundamentals_batch_fast
KPIS = {"enterpriseToEbitda": (11, 1), "priceToBook": (4, 1), "trailingPE": (2, 1), "returnOnEquity": (33, 0.01),
        "equity_ratio": (39, 0.01), "operating_margin": (29, 0.01), "net_debt_ebitda": (42, 1),
        "current_ratio": (44, 1), "dividend_yield": (1, 0.01), "revenue_growth": (94, 0.01),
        "earnings_growth": (97, 0.01), "earnings_stability": (174, 1), "fcf_stability": (179, 1),
        "f_score": (167, 1)}
EXTRA = {"revenue_m": 53, "ev_ebit": 10, "roc": 36, "roic": 37}      # CAGR, Magic Formula, ROIC-grinden (4b)
VARIANTS = ("valhall", "gron", "gron_utan_marknad", "forval", "ingen")
MAX_POS, STRONG_W, BUY_W, SECTOR_CAP = 10, 0.10, 0.07, 0.25          # cagr_backtest: regel 8–9
RS_MONTHS = 6
FUND_MAX = 20                      # screenerns Börsdata-skala
OPTIONAL_HISTORY = ("f_score", "earnings_stability", "fcf_stability")   # ett kriterium (1 poäng) var

# Valhalls segment → screenerns sektornamn (cagr_cycle.SECTOR_CONFIG)
SECTOR_OF = {"energy": "Energy", "materials": "Materials", "industrials": "Industrials",
             "financials": "Financials", "real_estate": "Real Estate", "technology": "Technology",
             "healthcare": "Healthcare", "communication": "Communication Services",
             "consumer_disc": "Consumer Discretionary", "consumer_staples": "Consumer Staples",
             "utilities": "Utilities"}


def kpi_ids() -> dict:
    return {**{k: v[0] for k, v in KPIS.items()}, **EXTRA}


def _known(vals: list, date) -> list:
    """[(år, värde)] kända på datumet (årsrapporten känd LAG_MONTHS efter årets slut)."""
    out = []
    for v in vals or []:
        y, x = v.get("y"), v.get("v")
        if y is None or x is None:
            continue
        if pd.Timestamp(int(y) + 1, 1, 1) + pd.DateOffset(months=lt.LAG_MONTHS) <= pd.Timestamp(date):
            out.append((int(y), float(x)))
    return sorted(out)


def fundamentals_at(hist: dict, date) -> dict:
    """Screenerns fundamentaindata som de såg ut på datumet. hist = {fält/extra-nyckel: [{y, v}]}."""
    info = {"_data_source": "borsdata"}
    from cagr.cagr_loader import GROWTH_FIELDS, _cap_growth
    for field, (_kid, scale) in KPIS.items():
        k = _known(hist.get(field), date)
        if k:
            v = k[-1][1] * scale
            info[field] = _cap_growth(v) if field in GROWTH_FIELDS else v       # som den dagliga körningen
    rev = dict(_known(hist.get("revenue_m"), date))
    if rev:
        y = max(rev)
        a, b = rev.get(y), rev.get(y - 5)
        if a and b and a > 0 and b > 0:
            info["revenue_cagr_5y"] = _cap_growth((a / b) ** (1 / 5) - 1)
    for k in ("ev_ebit", "roc", "roic"):
        kk = _known(hist.get(k), date)
        info[f"_{k}"] = kk[-1][1] if kk else None
    info["_roic_hist"] = [x for _y, x in _known(hist.get("roic"), date)]
    return info


def _pct_rank(values: dict, higher_better: bool = True) -> dict:
    """{ticker: 1–100} percentil (100 = bäst när higher_better)."""
    s = pd.Series({k: v for k, v in values.items() if v is not None and np.isfinite(v)}, dtype=float)
    if s.empty:
        return {}
    r = s.rank(pct=True, ascending=higher_better) * 100
    return {k: max(1.0, round(float(v), 0)) for k, v in r.items()}


def cross_section(infos: dict, closes: dict, date) -> None:
    """RS-rank (6 mån) och Magic Formula (EV/EBIT + avkastning på kapital) bland dagens aktier — in place."""
    mom = {t: lt.fwd(closes[t], pd.Timestamp(date) - pd.DateOffset(months=RS_MONTHS), RS_MONTHS) for t in infos}
    rs = _pct_rank(mom, higher_better=True)
    ey = {t: (1 / i["_ev_ebit"]) if i.get("_ev_ebit") and i["_ev_ebit"] > 0 else None for t, i in infos.items()}
    rank_ey = pd.Series({t: v for t, v in ey.items() if v is not None}).rank(ascending=False)
    rank_roc = pd.Series({t: i["_roc"] for t, i in infos.items() if i.get("_roc") is not None}).rank(ascending=False)
    both = (rank_ey + rank_roc).dropna()
    magic = (both.rank(pct=True) * 100).clip(lower=1).round(0).to_dict() if len(both) else {}
    for t, i in infos.items():
        if t in rs:
            i["rs_rank"] = rs[t]
        if t in magic:
            i["magic_formula"] = float(magic[t])


def cycle_for(variant: str, segment: str, sector_status: str, market: str) -> int:
    """Cykelpoängen 0–3 per variant. gron = som valhall men köp bara när sektorn är GRÖN (GUL ger 1 — inga
    köp); gron_utan_marknad = köp bara i GRÖN sektor, och marknaden tvingar aldrig fram sälj (aldrig 0)."""
    if variant == "ingen":
        return 3
    if variant == "gron":
        return ls.cycle_score(sector_status, market, "gron")
    if variant == "gron_utan_marknad":
        return 3 if sector_status == cycle.GRON else 1
    if variant == "forval":
        from cagr.cagr_cycle import score_cycle_for_sector
        return int(score_cycle_for_sector(SECTOR_OF.get(segment, "Unknown"))["cycle_score"])
    return ls.cycle_score(sector_status, market, "valhall")


def market_regimes(index_closes: dict, dates: list) -> dict:
    """{index: {datum: GRÖN/GUL/RÖD}} — EMA50/EMA200 på data t.o.m. dagen."""
    out = {}
    for sym, s in (index_closes or {}).items():
        reg = {}
        if s is not None and len(s):
            s = s.astype(float).dropna()
            e50, e200 = s.ewm(span=50, adjust=False).mean(), s.ewm(span=200, adjust=False).mean()
            for d in dates:
                k = int(s.index.searchsorted(pd.Timestamp(d), side="right")) - 1
                if k < 199:
                    reg[d] = cycle.NODATA
                    continue
                c, a, b = s.iloc[k], e50.iloc[k], e200.iloc[k]
                reg[d] = cycle.GRON if c > b and a > b else cycle.ROD if c < b and a < b else cycle.GUL
        out[sym] = reg
    return out


def signals(closes: dict, turnover: dict, universe: dict, hist: dict, nordic_frames: dict,
            index_closes: dict, dates: list, variants=VARIANTS, min_turnover: float = 1e6,
            days: int = 60, progress: Optional[Callable] = None, fund_max: int = FUND_MAX) -> list:
    """En rad per aktie och månadsskifte: screenerns signal per variant, poäng, Valhalls läge och ROIC-grind.
    closes ska vara rensade (longterm.clean). hist = {ticker: {fält: [{y, v}]}}. fund_max = poängen som går
    att nå (20 minus kriterier utan historik) — procentgränserna för BUY/STRONG BUY räknas mot den."""
    from cagr.cagr_scoring import calculate_total_score
    from cagr.cagr_cycle import score_cycle_from_int
    from cagr.cagr_fundamentals import score_fundamentals
    from cagr.cagr_technical import score_technical
    regimes = market_regimes(index_closes, dates)
    liq = {t: tv.rolling(days, min_periods=days // 2).mean().shift(1) for t, tv in turnover.items()}
    out = []
    for n, d in enumerate(dates):
        infos = {}
        for t, s in closes.items():
            seg = (universe.get(t) or {}).get("segment")
            if not seg or t not in hist or d < s.index[0] + pd.Timedelta(days=300):
                continue
            if t in liq and not (lt._asof(liq[t], d) or 0) >= min_turnover:
                continue
            infos[t] = fundamentals_at(hist[t], d)
        cross_section(infos, closes, d)
        for t, info in infos.items():
            s = closes[t]
            k = int(s.index.searchsorted(pd.Timestamp(d), side="right"))
            tech = score_technical(pd.DataFrame({"Close": s.iloc[max(0, k - ls.TECH_BARS):k]}))
            fund = dict(score_fundamentals(info, None), fund_max=fund_max)
            seg = universe[t]["segment"]
            st = lt._state(nordic_frames.get(seg), d)
            mkt = regimes.get(ls.market_index(t), {}).get(d, cycle.NODATA)
            row = {"ticker": t, "segment": seg, "date": str(pd.Timestamp(d).date()), "period": lt.period_of(d),
                   "fund": fund["fund_score"], "tech": tech["tech_score"], "market": mkt, **st,
                   **lt.gates(info["_roic_hist"])}
            for v in variants:
                total = calculate_total_score(fund, score_cycle_from_int(cycle_for(v, seg, st["status"], mkt)), tech)
                row[f"sig_{v}"], row[f"pct_{v}"] = total["signal"], total["score_pct"]
            out.append(row)
        if progress:
            progress(n + 1, len(dates), len(infos))
    return out


def add_returns(rows: list, closes: dict, bench: pd.Series) -> None:
    """Framåtavkastning (1 mån för portföljen, 12/24/36 mån mot Norden) — in place."""
    for r in rows:
        s = closes[r["ticker"]]
        r["r1"] = lt.fwd(s, r["date"], 1)
        for h in lt.HORIZONS:
            a, b = lt.fwd(s, r["date"], h), lt.fwd(bench, r["date"], h)
            r[f"r{h}"], r[f"x{h}"] = a, (a - b if a is not None and b is not None else None)


def portfolio(rows: list, variant: str, bench: Optional[pd.Series] = None, start=None, end=None) -> dict:
    """Screenerns portfölj: högst 10, STRONG BUY 10 % / BUY 7 %, högst 25 % per sektor, sälj på SELL.
    Vikterna är andelar av kapitalet och driver med kursen; resten är kassa (0 %)."""
    by_date: dict = {}
    for r in rows:
        by_date.setdefault(r["date"], {})[r["ticker"]] = r
    dates = sorted(d for d in by_date if (start is None or d >= start) and (end is None or d <= end))
    if not dates:
        return {"months": 0}
    eq, peak, dd, beq, buys, sizes = 1.0, 1.0, 0.0, 1.0, 0, []
    held: dict = {}                                                  # ticker → [vikt, segment]
    sig, pct = f"sig_{variant}", f"pct_{variant}"
    for d in dates:
        today = by_date[d]
        for t in [t for t in held if (today.get(t) or {}).get(sig) in ls.SELL_SIGNALS]:
            held.pop(t)
        cands = sorted((r for t, r in today.items() if t not in held and r[sig] in ls.BUY_SIGNALS),
                       key=lambda r: (r[sig] != "STRONG BUY", -r[pct], r["ticker"]))
        for r in cands:
            if len(held) >= MAX_POS:
                break
            w = STRONG_W if r[sig] == "STRONG BUY" else BUY_W
            if sum(x[0] for x in held.values()) + w > 1 + 1e-9:
                continue
            if sum(x[0] for x in held.values() if x[1] == r["segment"]) + w > SECTOR_CAP + 1e-9:
                continue
            held[r["ticker"]] = [w, r["segment"]]
            buys += 1
        sizes.append(len(held))
        rets = {t: ((today.get(t) or {}).get("r1") or 0.0) / 100 for t in held}
        month = sum(held[t][0] * rets[t] for t in held)
        for t in held:                                               # nya andelar efter månaden
            held[t][0] = held[t][0] * (1 + rets[t]) / (1 + month) if month > -1 else 0.0
        eq *= 1 + month
        peak = max(peak, eq)
        dd = min(dd, eq / peak - 1)
        b = lt.fwd(bench, d, 1) if bench is not None else None
        beq *= 1 + (b or 0.0) / 100
    years = len(dates) / 12
    return {"months": len(dates), "cagr": round((eq ** (1 / years) - 1) * 100, 1), "dd": round(dd * 100, 1),
            "avg_n": round(float(np.mean(sizes)), 1), "buys": buys,
            "bench_cagr": round((beq ** (1 / years) - 1) * 100, 1) if bench is not None else None}


def portfolios(rows: list, bench: Optional[pd.Series], variants=VARIANTS) -> list:
    out = []
    for v in variants:
        for per, end in (("alla", None),) + lt.PERIODS:
            start = "2021-01-01" if per == "2021–" else None
            out.append({"variant": v, "period": per, **portfolio(rows, v, bench, start, end)})
    return out


def buy_table(rows: list, variants=VARIANTS) -> list:
    """Framåtavkastningen för screenerns köp (STRONG BUY/BUY) per variant och per Valhall-status."""
    out = []
    for v in variants:
        sel = [dict(r, variant=v) for r in rows if r[f"sig_{v}"] in ls.BUY_SIGNALS]
        out += lt.summarize(sel, ("variant",)) if sel else [{"variant": v, "period": "alla", "n": 0}]
        out += lt.summarize(sel, ("variant", "status")) if sel else []
    out += lt.summarize([dict(r, variant="alla likvida") for r in rows], ("variant",))
    return out
