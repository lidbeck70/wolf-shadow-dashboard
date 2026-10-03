"""
swing_backtest.py — backtest av Momentum Swing (veckorutinen) på svenska
Large + Mid Cap, som en portfölj, utan look-ahead.

Reglerna är exakt screenerns och regimens (wolf_data.py, strategy_rules.py):

  RANKING    varje veckas sista handelsdag efter stängning: score = 50 % 3-mån
             + 50 % 6-mån avkastning; kvalar = pris > MA200, 3-mån > 0,
             6-mån ≥ +10 %. Topp 20 = köpbara, topp 40 = rank-exit-gränsen.
  SETUP      A = pris inom ±2 % från MA20 eller MA50 och RSI 35–55
             B = inom 3 % från 52-veckorshögsta
  REGIM      RÖD (OMXSPI < MA200): inga köp · GUL (bredd < 45 % eller index
             < 2 % över MA200): halv storlek, bara setup A, max 1 köp ·
             GRÖN: full storlek, A eller B, max 2 köp
  KÖP        nästa handelsdags öppning, högst score först, max 8 positioner,
             storlek 16 % av kontot (GRÖN) / 8 % (GUL), ingen belåning
  SÄLJ       stop −10 % (intradag; gap under → öppningen) · stängning under
             MA50 → nästa öppning · ur topp 40 (veckovis) → nästa öppning ·
             +20 % → sälj halva och flytta stopen till entry

Allt bygger på data t.o.m. signaldagen. Courtage dras per transaktion.
Ingår inte: Börsdata-screenerns kvalitetsfilter (F-score, börsvärde) — kan
inte återskapas historiskt; universumet är dagens (överlevnadsbias).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

import wolf_data as wd

GREEN, YELLOW, RED, UNKNOWN = "GRÖN", "GUL", "RÖD", "OKÄND"
INDEX_HINTS = ("OMX Stockholm PI", "OMXSPI", "OMX Stockholm_PI")
BD_MARKETS = (1, 2)                 # Börsdata: Large Cap och Mid Cap Stockholm
WARMUP_BARS = 260                   # MA200 + 52 veckors högsta


@dataclass
class Config:
    years: int = 5
    top_n: int = wd.CONFIG["TOP_N"]               # 20
    rank_exit: int = wd.CONFIG["RANK_EXIT"]       # 40
    max_positions: int = 8
    size_green: float = 16.0                      # % av kontot (regel: 12–20 %)
    size_yellow: float = 8.0                      # halv storlek
    max_buys_green: int = 2
    max_buys_yellow: int = 1
    stop_pct: float = 10.0
    half_at_pct: float = 20.0
    fee_pct: float = 0.1                          # courtage per transaktion
    require_setup: bool = True
    regime_filter: bool = True
    mom_long_min: float = wd.CONFIG["MOM_LONG_MIN"]
    near_ma: float = wd.CONFIG["NEAR_MA_PCT"]
    near_high: float = wd.CONFIG["NEAR_HIGH_PCT"]


VARIANTS = {
    "Reglerna": {},
    "Utan setup-krav": {"require_setup": False},
    "Utan regimfilter": {"regime_filter": False},
}


@dataclass
class Trade:
    ticker: str
    entry_date: str
    entry: float
    setup: str
    regime: str
    exit_date: Optional[str] = None
    exit: Optional[float] = None
    reason: str = ""
    ret_pct: Optional[float] = None               # hela positionen inkl. halvsälj och courtage
    days: int = 0
    half_sold: bool = False
    open: bool = False


# ── Data → paneler ──────────────────────────────────────────────────────────
def panels(prices: dict, index_close: Optional[pd.Series]) -> dict:
    """{O, H, L, C}: DataFrames (dagar × tickers) på indexets kalender (eller unionen)."""
    frames = {}
    for t, df in prices.items():
        if df is None or len(df) == 0:
            continue
        d = df.copy()
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        d = d[~d.index.duplicated(keep="last")].sort_index()
        frames[t] = d
    if not frames:
        return {}
    if index_close is not None and len(index_close):
        cal = pd.DatetimeIndex(index_close.index).sort_values()
    else:
        cal = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values()])))
    out = {}
    for k in ("Open", "High", "Low", "Close"):
        out[k[0]] = pd.concat({t: f[k].astype(float) for t, f in frames.items() if k in f}, axis=1).reindex(cal)
    return out


def indicators(C: pd.DataFrame, cfg: Config = Config()) -> dict:
    """Screenerns nyckeltal per dag och aktie — allt kausalt (rullande bakåt)."""
    ma20, ma50, ma200 = C.rolling(20).mean(), C.rolling(50).mean(), C.rolling(200).mean()
    # Exakt screenerns definition: px / c.iloc[−63] → 62 handelsdagar bakåt (och 125 för 6 mån)
    r3 = C / C.shift(wd.CONFIG["MOM_SHORT"] - 1) - 1
    r6 = C / C.shift(wd.CONFIG["MOM_LONG"] - 1) - 1
    hi52 = C.rolling(252, min_periods=200).max()
    delta = C.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = (100 - 100 / (1 + up / dn)).where(dn > 0, 100.0)
    qualifies = (C > ma200) & (r3 > 0) & (r6 > cfg.mom_long_min)
    near_ma = ((C / ma20 - 1).abs() <= cfg.near_ma) | ((C / ma50 - 1).abs() <= cfg.near_ma)
    setup_a = qualifies & near_ma & (rsi >= 35) & (rsi <= 55)
    setup_b = qualifies & (C >= hi52 * (1 - cfg.near_high))
    valid = ma200.notna()
    breadth = (C > ma200).where(valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    return {"ma50": ma50, "score": 0.5 * r3 + 0.5 * r6, "qualifies": qualifies, "setup_a": setup_a,
            "setup_b": setup_b, "breadth": breadth}


def regimes(index_close: Optional[pd.Series], breadth: pd.Series) -> pd.Series:
    """Regim per dag med wolf_data.classify_regime (samma funktion som Swing Regime)."""
    if index_close is None or not len(index_close):
        return pd.Series(UNKNOWN, index=breadth.index)
    c = index_close.astype(float).reindex(breadth.index).ffill()
    ma = c.rolling(200).mean()
    out = []
    for d in breadth.index:
        if pd.isna(c[d]) or pd.isna(ma[d]) or pd.isna(breadth[d]):
            out.append(UNKNOWN)
            continue
        block = {"above": bool(c[d] > ma[d]), "dist": float(c[d] / ma[d] - 1)}
        out.append(wd.classify_regime(block, float(breadth[d]))[0])
    return pd.Series(out, index=breadth.index)


def week_ends(idx: pd.DatetimeIndex) -> set:
    """Sista handelsdagen i varje ISO-vecka."""
    s = pd.Series(idx, index=idx)
    iso = idx.isocalendar()
    return set(s.groupby([iso.year.values, iso.week.values]).max().values)


# ── Simuleringen ────────────────────────────────────────────────────────────
def run(pn: dict, index_close: Optional[pd.Series], cfg: Config = Config(), today=None) -> dict:
    """Portföljsimulering. pn = panels(...). Returnerar trades, kontokurva och nyckeltal."""
    if not pn:
        return {"trades": [], "equity": None, "metrics": {}, "regime_share": {}, "config": cfg}
    O, H, L, C = pn["O"], pn["H"], pn["L"], pn["C"]
    ind = indicators(C, cfg)
    reg = regimes(index_close, ind["breadth"])
    end = pd.Timestamp(today) if today is not None else C.index[-1]
    start = end - pd.DateOffset(years=int(cfg.years))
    days = [d for d in C.index if start <= d <= end]
    if len(C.index) and days and C.index.get_loc(days[0]) < WARMUP_BARS:
        days = [d for d in days if C.index.get_loc(d) >= WARMUP_BARS]
    weekly = week_ends(C.index)
    fee = cfg.fee_pct / 100
    cash, equity0 = 100_000.0, 100_000.0
    pos: dict = {}                       # ticker → dict(shares, entry, stop, half, trade, cost)
    pend_sell: dict = {}                 # ticker → orsak
    pend_buy: list = []                  # [(ticker, storlek %, setup, regim)]
    trades, curve, invested = [], [], []
    last_px = {}

    def px(t, frame, d):
        v = frame.at[d, t] if t in frame.columns else np.nan
        return None if pd.isna(v) else float(v)

    def close_trade(t, price, d, reason):
        nonlocal cash
        p = pos.pop(t)
        cash += p["shares"] * price * (1 - fee)
        tr = p["trade"]
        tr.exit_date, tr.exit, tr.reason = str(d.date()), round(price, 4), reason
        proceeds = p["proceeds"] + p["shares"] * price * (1 - fee)
        tr.ret_pct = round((proceeds / p["cost"] - 1) * 100, 2)
        tr.days = int(np.busday_count(pd.Timestamp(tr.entry_date).date(), d.date()))
        trades.append(tr)

    for d in days:
        # 1. Öppning: säljorder först, sedan köp
        for t, reason in list(pend_sell.items()):
            if t not in pos:
                pend_sell.pop(t, None)
                continue
            o = px(t, O, d)
            if o is not None:                     # ingen kurs i dag → ordern ligger kvar
                close_trade(t, o, d, reason)
                pend_sell.pop(t, None)
        equity_open = cash + sum(p["shares"] * (px(t, O, d) or last_px.get(t, p["entry"])) for t, p in pos.items())
        for t, size, setup, regime in pend_buy:
            if t in pos or len(pos) >= cfg.max_positions:
                continue
            o = px(t, O, d)
            if o is None or o <= 0:
                continue
            amount = min(equity_open * size / 100, cash / (1 + fee))
            if amount <= equity_open * 0.01:
                continue
            shares = amount / o
            cash -= amount * (1 + fee)
            pos[t] = {"shares": shares, "entry": o, "stop": o * (1 - cfg.stop_pct / 100), "half": False,
                      "cost": amount * (1 + fee), "proceeds": 0.0,
                      "trade": Trade(t, str(d.date()), round(o, 4), setup, regime)}
        pend_buy = []
        # 2. Intradag: stop före +20 %
        for t in list(pos):
            p = pos[t]
            o, lo, hi = px(t, O, d), px(t, L, d), px(t, H, d)
            if lo is None:
                continue
            if lo <= p["stop"]:
                close_trade(t, o if o is not None and o < p["stop"] else p["stop"], d,
                            "breakeven-stopp" if p["half"] else "stopp −10 %")
                continue
            target = p["entry"] * (1 + cfg.half_at_pct / 100)
            if not p["half"] and hi is not None and hi >= target:
                fill = o if o is not None and o >= target else target
                half = p["shares"] / 2
                cash += half * fill * (1 - fee)
                p["proceeds"] += half * fill * (1 - fee)
                p["shares"] -= half
                p["half"], p["stop"] = True, p["entry"]
                p["trade"].half_sold = True
        # 3. Stängning: MA50-regeln dagligen, ranking och köp veckovis
        for t in pos:
            c = px(t, C, d)
            if c is not None:
                last_px[t] = c
                m50 = ind["ma50"].at[d, t]
                if not pd.isna(m50) and c < m50:
                    pend_sell.setdefault(t, "under MA50")
        if d in weekly:
            score = ind["score"].loc[d].where(ind["qualifies"].loc[d]).dropna().sort_values(ascending=False)
            top40, top20 = set(score.index[:cfg.rank_exit]), list(score.index[:cfg.top_n])
            for t in pos:
                if t not in top40:
                    pend_sell.setdefault(t, "ur topp 40")
            regime = reg.get(d, UNKNOWN) if cfg.regime_filter else GREEN
            if regime != RED:
                yellow = regime == YELLOW
                size = cfg.size_yellow if yellow else cfg.size_green
                max_buys = cfg.max_buys_yellow if yellow else cfg.max_buys_green
                free = cfg.max_positions - len([t for t in pos if t not in pend_sell])
                for t in top20:
                    if len(pend_buy) >= min(max_buys, free):
                        break
                    if t in pos:
                        continue
                    a, b = bool(ind["setup_a"].at[d, t]), bool(ind["setup_b"].at[d, t])
                    if cfg.require_setup and not (a or (b and not yellow)):
                        continue
                    pend_buy.append((t, size, "A" if a else "B" if b else "—", regime))
        value = cash + sum(p["shares"] * last_px.get(t, p["entry"]) for t, p in pos.items())
        curve.append((d, value))
        invested.append((value - cash) / value if value > 0 else 0.0)

    for t in list(pos):                           # öppna positioner värderas, räknas inte som affärer
        p = pos[t]
        tr = p["trade"]
        tr.open, tr.exit_date, tr.exit = True, str(days[-1].date()), last_px.get(t, p["entry"])
        tr.ret_pct = round(((p["proceeds"] + p["shares"] * tr.exit) / p["cost"] - 1) * 100, 2)
        trades.append(tr)
    equity = pd.Series(dict(curve)) if curve else None
    share = reg.loc[days].value_counts(normalize=True).round(3).to_dict() if days else {}
    return {"trades": trades, "equity": equity, "metrics": metrics(trades, equity, invested, cfg.years),
            "regime_share": share, "config": cfg,
            "benchmark": benchmark(index_close, days[0], days[-1]) if days else {}}


# ── Nyckeltal ───────────────────────────────────────────────────────────────
def _max_dd(eq: pd.Series) -> float:
    peak = eq.cummax()
    return float(((eq / peak) - 1).min() * -100) if len(eq) else 0.0


def metrics(trades: list, equity: Optional[pd.Series], invested: list, years: float) -> dict:
    closed = [t for t in trades if not t.open and t.ret_pct is not None]
    out = {"trades": len(closed), "open": len([t for t in trades if t.open])}
    if equity is not None and len(equity) > 1:
        total = float(equity.iloc[-1] / equity.iloc[0] - 1)
        yrs = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
        out.update({"total_return_pct": round(total * 100, 1),
                    "cagr_pct": round(((1 + total) ** (1 / yrs) - 1) * 100, 1) if total > -1 else None,
                    "max_dd_pct": round(_max_dd(equity), 1),
                    "exposure_pct": round(float(np.mean(invested)) * 100, 1) if invested else None})
    if closed:
        rs = [t.ret_pct for t in closed]
        wins, losses = [r for r in rs if r > 0], [r for r in rs if r <= 0]
        avg_w = float(np.mean(wins)) if wins else 0.0
        avg_l = float(np.mean(losses)) if losses else 0.0
        out.update({"win_rate": round(len(wins) / len(rs) * 100, 1), "avg_win_pct": round(avg_w, 2),
                    "avg_loss_pct": round(avg_l, 2),
                    "payoff": round(avg_w / abs(avg_l), 2) if avg_l < 0 else (math.inf if avg_w > 0 else None),
                    "profit_factor": round(sum(wins) / -sum(losses), 2) if losses and sum(losses) < 0 else None,
                    "avg_days": round(float(np.mean([t.days for t in closed])), 1),
                    "half_sold_share": round(sum(t.half_sold for t in closed) / len(closed) * 100, 1)})
    return out


def benchmark(index_close: Optional[pd.Series], start, end) -> dict:
    if index_close is None or not len(index_close):
        return {}
    s = index_close.astype(float)
    s = s[(s.index >= start) & (s.index <= end)].dropna()
    if len(s) < 2:
        return {}
    total = float(s.iloc[-1] / s.iloc[0] - 1)
    yrs = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    return {"total_return_pct": round(total * 100, 1), "cagr_pct": round(((1 + total) ** (1 / yrs) - 1) * 100, 1),
            "max_dd_pct": round(_max_dd(s), 1), "curve": s}


def exit_table(trades: list) -> list:
    rows = {}
    for t in trades:
        if t.open or t.ret_pct is None:
            continue
        rows.setdefault(t.reason, []).append(t.ret_pct)
    return [{"Exit": k, "Antal": len(v), "Snitt %": round(float(np.mean(v)), 2), "Summa %": round(float(np.sum(v)), 1)}
            for k, v in sorted(rows.items(), key=lambda kv: -len(kv[1]))]


# ── Datahämtning ────────────────────────────────────────────────────────────
NOTES = (
    "Veckans ranking och regim räknas på sista handelsdagens stängning; köp sker nästa handelsdags öppning.",
    "Stop −10 % gäller intradag (gap under → öppningskursen); MA50-brott och 'ur topp 40' säljs nästa öppning.",
    "Courtage dras på varje köp och sälj. Ingen belåning — köp som inte ryms i kassan hoppas över.",
    "Börsdata-screenerns kvalitetsfilter (F-score, börsvärde) ingår inte — det går inte att återskapa historiskt.",
    "Universumet är dagens Large + Mid Cap: bolag som försvunnit saknas (överlevnadsbias).",
)


def load_data(years: int, api=None, yahoo_getter: Optional[Callable] = None,
              progress: Optional[Callable] = None) -> dict:
    """{prices: {ticker: df}, index: serie, source, universe, missing}. Börsdata Large + Mid Cap;
    utan Börsdata-nyckel: de svenska bolagen i Norden 50 via Yahoo (märkt)."""
    bars = (int(years) + 2) * 262
    if api is None:
        try:
            from borsdata_api import BorsdataAPI
            api = BorsdataAPI()
            api = api if api.is_configured else None
        except Exception:
            api = None
    prices, missing = {}, 0
    if api is not None:
        ins = [i for i in (api.get_instruments() or []) if i.get("marketId") in BD_MARKETS
               and i.get("instrumentType", 0) in (0, None)]
        for k, i in enumerate(ins):
            try:
                df = api.get_stockprices_df(int(i["insId"]), max_count=bars)
            except Exception:
                df = None
            if df is not None and len(df):
                prices[str(i.get("ticker") or i["insId"])] = df
            else:
                missing += 1
            if progress is not None:
                progress(k + 1, len(ins), str(i.get("ticker", "")))
        import market_risk as mr
        idx, why = mr.borsdata_index(api, INDEX_HINTS, bars)
        return {"prices": prices, "index": idx, "source": f"Börsdata Large + Mid Cap ({len(prices)} bolag) · index: {why}",
                "universe": len(ins), "missing": missing}
    if yahoo_getter is None:
        from market_prices import ohlcv as yahoo_getter
    import viking_backtest as vb
    tickers = [t for t in vb.NORDIC_50 if t.endswith(".ST")]
    for k, t in enumerate(tickers):
        df = yahoo_getter(t, f"{int(years) + 2}y")
        if df is not None and len(df):
            prices[t] = df
        else:
            missing += 1
        if progress is not None:
            progress(k + 1, len(tickers), t)
    idx_df = yahoo_getter("^OMXSPI", f"{int(years) + 2}y")
    idx = idx_df["Close"] if idx_df is not None and len(idx_df) else None
    return {"prices": prices, "index": idx, "universe": len(tickers), "missing": missing,
            "source": f"RESERV — Börsdata-nyckel saknas: svenska bolag i Norden 50 via Yahoo ({len(prices)} st), "
                      f"inte Large + Mid Cap"}
