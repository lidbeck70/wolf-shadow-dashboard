#!/usr/bin/env python3
"""A/B-kalibrering av ränteshock-signalen (duration_stress, TLT) i marknadsriskfliken.

Kör i GitHub Actions (eller lokalt) där Yahoo Finance och FRED är nåbara.
Jämför träffandel per nivå, varnade episoder och HÖG-larm — med och utan
TLT-signalen — på identisk data. Ingen optimering: signalen är satt i förväg.
"""
from __future__ import annotations
import sys
import pathlib
import warnings
warnings.filterwarnings("ignore")

# Körs som `python scripts/calibrate_tlt_ab.py` — lägg repo-roten på sys.path
# så `import market_risk` fungerar (sys.path[0] är annars scripts/).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

import market_risk as mr

TICKERS = sorted({"SPY", "^OMX", "^VIX", "^VIX3M", "HYG", "IEF", "TLT",
                  "XLU", "XLP", "XLK", "XLY", *mr.BREADTH_ETFS})


def fetch() -> dict:
    raw = yf.download(TICKERS, period="max", progress=False, group_by="ticker",
                      auto_adjust=True, threads=True)
    data = {}
    for t in TICKERS:
        try:
            df = raw[t].dropna(how="all")
            df.index = pd.DatetimeIndex(df.index).tz_localize(None)
            data[t] = df if len(df) else None
        except Exception:
            data[t] = None
    # Enstaka tickers faller ibland bort i batch-nedladdningen — hämta om individuellt
    import time
    for t in TICKERS:
        if data.get(t) is None or not len(data[t]):
            for attempt in range(3):
                try:
                    df = yf.Ticker(t).history(period="max", auto_adjust=True)
                    if df is not None and len(df):
                        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
                        data[t] = df
                        break
                except Exception:
                    pass
                time.sleep(3 * (attempt + 1))
    data["T10Y2Y"] = mr._fred_default("T10Y2Y")
    return data


def report(name: str, close: pd.Series, data: dict, breadth: bool) -> None:
    act, avail = mr.compute_signals(close, data, breadth=breadth)
    assert "duration_stress" in act.columns, "signalen saknas — kör mot feature-branchen"
    before = mr.calibrate(close, act.drop(columns=["duration_stress"]),
                          avail.drop(columns=["duration_stress"]))
    after = mr.calibrate(close, act, avail)
    if before is None or after is None:
        print(f"\n==== {name}: kalibrering ej möjlig (saknad data) ====")
        return
    print(f"\n==== {name} ====")
    print(f"Fönster: {after.start} → {after.end} · {after.days} dagar · "
          f"basnivå (10 %-fall inom 63 d): {after.base_rate} %")
    print(f"{'Nivå':<9}{'UTAN TLT':>24}{'MED TLT':>24}")
    for lvl, _lo in mr.LEVELS:
        b, a = before.by_level[lvl], after.by_level[lvl]
        fb = f"{b['hit_rate']} % ({b['days']} d)" if b["hit_rate"] is not None else "—"
        fa = f"{a['hit_rate']} % ({a['days']} d)" if a["hit_rate"] is not None else "—"
        print(f"{lvl:<9}{fb:>24}{fa:>24}")
    wb = sum(e["warned"] for e in before.episodes)
    wa = sum(e["warned"] for e in after.episodes)
    print(f"Episoder varnade i förväg: {wb}/{len(before.episodes)} → {wa}/{len(after.episodes)}")
    print(f"HÖG-larm totalt/träff/falska: "
          f"{before.alarms['total']}/{before.alarms['hits']}/{before.alarms['false']} → "
          f"{after.alarms['total']}/{after.alarms['hits']}/{after.alarms['false']}")
    fired = act.index[act["duration_stress"] & avail["duration_stress"]]
    print(f"TLT-signalen aktiv {len(fired)} dagar · år: {sorted({d.year for d in fired})}")


def main() -> int:
    data = fetch()
    missing = [t for t in ("SPY", "TLT", "HYG", "^VIX", "^VIX3M") if data.get(t) is None]
    if missing:
        print("AVBRYTER — saknar kritisk data:", ", ".join(missing))
        return 1
    spy = mr._close(data["SPY"]).dropna()
    report("SPY (med US-sektorbredd)", spy, data, breadth=True)
    omx = mr._close(data.get("^OMX")) if data.get("^OMX") is not None else None
    if omx is not None and len(omx.dropna()) >= 260:
        report("OMXS30 (^OMX, utan Börsdata-bredd)", omx.dropna(), data, breadth=False)
    else:
        print("\nOMXS30: ^OMX saknas hos Yahoo — hoppar över.")
    print("\nBeslutsregel: behåll signalen bara om FÖRHÖJD/HÖG-träffandelen förbättras "
          "eller fler episoder varnas i förväg utan fler falska HÖG-larm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
