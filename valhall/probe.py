#!/usr/bin/env python3
"""
valhall/probe.py — datasonden för Valhall (fas 1). Ändrar ingenting.

  1. SPDR-drivarna och UCITS-kandidaterna hos Yahoo: första/sista datum, rader, senaste kurs.
     (Om UCITS-fonden går att köpa hos Nordnet måste kontrolleras där — sonden visar bara att det
     finns kursdata.)
  2. De nordiska sektorindexen: medlemmar per segment ur Börsdata och hur många som har kurser.
  3. Kartan idag: status, fas och cykelläge per segment (global och Norden) och råvarutema.

Körs i GitHub Actions (.github/workflows/valhall-probe.yml):  python -m valhall.probe
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valhall import board as vb  # noqa: E402
from valhall import nordic_index as ni  # noqa: E402
from valhall import segments as sgm  # noqa: E402


def _series_line(symbol: str, getter) -> str:
    try:
        df = getter(symbol, "max")
    except Exception as exc:
        return f"  {symbol:10s} FEL {exc}"
    if df is None or len(df) == 0:
        return f"  {symbol:10s} SAKNAS"
    c = df["Close"].dropna()
    return f"  {symbol:10s} OK  {c.index[0].date()} → {c.index[-1].date()}  {len(c):5d} rader  senast {c.iloc[-1]:.2f}"


def main() -> int:
    from market_prices import ohlcv
    print("== 1. DRIVARE OCH UCITS-KANDIDATER")
    for s in sgm.SEGMENTS:
        print(f"{s.label} ({s.key})")
        for sym in (s.spdr,) + tuple(s.ucits):
            print(_series_line(sym, ohlcv))
    print("\n== 2. NORDISKA SEKTORINDEX")
    nordic = {}
    try:
        nordic = ni.load()                                   # full historik (Yahoo "max")
        for k, v in nordic.items():
            idx = v["index"]
            span = f"{idx.index[0].date()} → {idx.index[-1].date()}" if idx is not None else "inget index"
            print(f"  {k:18s} medlemmar {v['members']:4d}  med kurs {v['with_data']:4d}  likvida idag {v.get('liquid')}  {span}")
    except Exception as exc:
        print(f"  FEL: {exc}")
    print("\n== 3. KARTAN IDAG")
    nm = None
    try:
        import ovtlyr_nine as on
        nm = (on.nordic_market() or {}).get("close")
    except Exception as exc:
        print(f"  (OMXS30 saknas: {exc})")
    b = vb.build(nordic=nordic, nordic_market=nm)
    for r in b["segments"]:
        g, n = r["global"], r.get("nordic") or {}
        print(f"  {r['label']:16s} global {g['status']:11s} {g['phase']:16s} {g['cykel']:7s} "
              f"pct10 {g.get('pct10')} hat {g.get('hat')} | Norden {n.get('status', '—'):11s} "
              f"{n.get('phase', '—'):16s} {n.get('cykel', '—')}")
        for t in r["themes"]:
            st = t["state"]
            print(f"      {t['label']:22s} {t['driver'] or '—':8s} {st['status']:11s} {st['phase']:16s} "
                  f"{st['cykel']:7s} pct10 {st.get('pct10')} hat {st.get('hat')}")
    print("\nCYKELSTART NU:", vb.cycle_starts(b) or "inga")
    return 0


if __name__ == "__main__":
    sys.exit(main())
