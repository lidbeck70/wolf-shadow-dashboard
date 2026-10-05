#!/usr/bin/env python3
"""berserk/diag.py — tillfällig diagnos (mergas inte): var finns kanten 2008–2026?"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from berserk import backtest as bt  # noqa: E402
from berserk import signals as sg  # noqa: E402
from berserk import themes as th  # noqa: E402
from berserk import universe as uv  # noqa: E402


def st(trs):
    rs = [t.r for t in trs if not t.open and t.r is not None]
    return f"n={len(rs):4d} exp={np.mean(rs):+.3f} sum={np.sum(rs):+7.1f}" if rs else "n=   0"


def table(title, trades, key):
    print(f"\n== {title}")
    g = defaultdict(lambda: defaultdict(list))
    for t in trades:
        per = "≤2020" if t.entry_date < "2021" else "2021+"
        g[key(t)][per].append(t)
    for k in sorted(g, key=str):
        print(f"  {str(k):28s} ≤2020 {st(g[k]['≤2020'])} | 2021+ {st(g[k]['2021+'])}")


def bucket(v, edges):
    if v is None:
        return "—"
    for e in edges:
        if v < e:
            return f"<{e}"
    return f"≥{edges[-1]}"


def main():
    tickers = list(dict.fromkeys(tuple(uv.NORDIC) + tuple(uv.GLOBAL) + tuple(uv.ETFS)))
    res = bt.run(tickers, cfg=bt.Config(start_year=2008, end_year=2026))
    tr = res["trades"]
    print("PERIOD", res["period"], "trades", len(tr), "utan data",
          [p["ticker"] for p in res["per_ticker"] if p.get("error")])
    print("DRIVARE", res["drivers"])
    print("\n== ÅR × SETUP")
    for y in range(2008, 2027):
        ys = [t for t in tr if t.entry_date[:4] == str(y)]
        print(f"  {y} alla {st(ys)} | " + " | ".join(
            f"{s.split()[0]} {st([t for t in ys if t.features.get('setup') == s])}" for s in sg.SETUPS))
    table("SETUP", tr, lambda t: t.features.get("setup"))
    table("REGION", tr, lambda t: "ETF" if t.features.get("kind") == "etf" else uv.region_of(t.ticker))
    table("KOMPLEX", tr, lambda t: t.features.get("complex"))
    table("TEMA", tr, lambda t: t.features.get("theme"))
    table("EXITORSAK", tr, lambda t: t.exit_reason)
    s1 = [t for t in tr if t.features.get("setup") == sg.S1]
    s2 = [t for t in tr if t.features.get("setup") == sg.S2]
    table("S1 DIVERGENS (pe)", s1, lambda t: bucket(t.features.get("divergence"), [-30, -20, -15]))
    table("S1 RVOL", s1, lambda t: bucket(t.features.get("rvol"), [1.5, 2.0, 3.0]))
    table("S1 ATR %", s1, lambda t: bucket(t.features.get("atr_pct"), [2, 3, 4, 6]))
    table("S2 HATAD dd252 %", s2, lambda t: bucket(t.features.get("dd252"), [-70, -60, -50]))
    table("S2 RVOL", s2, lambda t: bucket(t.features.get("rvol"), [2.0, 3.0]))
    table("S1+S2 GAP ATR", s1 + s2, lambda t: bucket(t.features.get("gap_atr"), [-0.5, 0, 0.5, 1]))


if __name__ == "__main__":
    main()
