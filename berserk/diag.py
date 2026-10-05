#!/usr/bin/env python3
"""
berserk/diag.py — diagnos av 🪓 BERSERK-backtestet "Allt, 5 år" (tillfällig, körs i Actions).

Letar efter det som kan ge stor drawdown trots positiv kant: extrema R-värden
(gap genom stopp eller datafel som pence/pund-byten och splittar), kurshopp i
datan, och skillnaden mellan alla affärer och portföljens urval per region/setup.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import viking_portfolio as vp  # noqa: E402
from berserk import backtest as bt  # noqa: E402
from berserk import universe as uv  # noqa: E402


def stats(trs):
    rs = [t.r for t in trs if not t.open and t.r is not None]
    if not rs:
        return "0"
    return f"n={len(rs)} exp={np.mean(rs):+.3f} sumR={np.sum(rs):+.1f} min={min(rs):+.1f} max={max(rs):+.1f}"


def main():
    tickers = list(dict.fromkeys(tuple(uv.NORDIC) + tuple(uv.GLOBAL) + tuple(uv.ETFS)))
    res = bt.run(tickers, cfg=bt.Config(years=5))
    trades = res["trades"]
    years = (res.get("period") or {}).get("years")
    p = vp.simulate(trades, years=years, pc=bt.portfolio_config())
    print(f"PERIOD {res.get('period')} tickers={len(tickers)} trades={len(trades)}")
    print(f"ALLA {stats(trades)}")
    print(f"PORTFÖLJ taken={p['taken']} ret={p['return_pct']} cagr={p['cagr_pct']} dd_mtm={p['max_dd_pct']} "
          f"dd_closed={p['closed_dd_pct']} skip full={p['skipped_full']} sektor={p['skipped_sector']} "
          f"grupp={p['skipped_group']} värme={p['skipped_heat']} förlust={p['skipped_losses']} maxopen={p['max_open']}")
    taken = [r["trade"] for r in p["rows"]]
    print(f"PORTFÖLJENS AFFÄRER {stats(taken)}")
    for name, key in (("REGION", lambda t: "ETF" if t.features.get("kind") == "etf" else uv.region_of(t.ticker)),
                      ("SETUP", lambda t: t.features.get("setup"))):
        g_all, g_tk = defaultdict(list), defaultdict(list)
        for t in trades:
            g_all[key(t)].append(t)
        for t in taken:
            g_tk[key(t)].append(t)
        for k in sorted(g_all, key=str):
            print(f"{name} {k}: alla {stats(g_all[k])} | portfölj {stats(g_tk[k])}")
    print("\nEXTREMA R (|R| > 4):")
    for t in sorted((t for t in trades if t.r is not None and abs(t.r) > 4), key=lambda t: t.r):
        print(f"  {t.ticker:10s} {t.features.get('setup'):18s} {t.entry_date}→{t.exit_date} entry={t.entry} "
              f"stop={t.stop} exit={t.exit} R={t.r:+.2f} {t.exit_reason} atr%={t.features.get('atr_pct')}")
    print("\nSÄMSTA PORTFÖLJAFFÄRER (% av kontot):")
    for r in sorted((r for r in p["rows"] if r["return_pct"] is not None), key=lambda r: r["return_pct"])[:20]:
        t = r["trade"]
        print(f"  {t.ticker:10s} {t.features.get('setup'):18s} {t.entry_date}→{t.exit_date} R={t.r:+.2f} "
              f"pos={r['position_pct']}% konto={r['return_pct']:+.2f}%")
    print("\nBÄSTA PORTFÖLJAFFÄRER (% av kontot):")
    for r in sorted((r for r in p["rows"] if r["return_pct"] is not None), key=lambda r: -r["return_pct"])[:10]:
        t = r["trade"]
        print(f"  {t.ticker:10s} {t.features.get('setup'):18s} {t.entry_date}→{t.exit_date} R={t.r:+.2f} "
              f"pos={r['position_pct']}% konto={r['return_pct']:+.2f}%")
    print("\nPORTFÖLJ PER ÅR (summa % av kontot, stängda):")
    yr = defaultdict(float)
    for r in p["rows"]:
        if r["return_pct"] is not None:
            yr[r["trade"].exit_date[:4]] += r["return_pct"]
    for y in sorted(yr):
        print(f"  {y}: {yr[y]:+.1f} %")
    print("\nKURSHOPP I DATAN (|dag| > 35 %):")
    from market_prices import ohlcv
    for tk in tickers:
        try:
            df = ohlcv(tk, "7y")
            c = df["Close"].astype(float).dropna()
            ch = c.pct_change().abs()
            big = ch[ch > 0.35]
            if len(big):
                print(f"  {tk}: " + ", ".join(f"{d.date()} {c.shift(1).loc[d]:.2f}→{c.loc[d]:.2f}"
                                               for d in big.index[:6]))
        except Exception as exc:
            print(f"  {tk}: fel {exc}")


if __name__ == "__main__":
    main()
