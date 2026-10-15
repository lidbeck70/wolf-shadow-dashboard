#!/usr/bin/env python3
"""
valhall_nordic_probe.py — Norden-proben: är hela Norden en bättre mätpunkt än OMXS30? (2008–idag)

  1. Serierna: OMXS30 (Börsdata, dagens grind i Viking Nine, BERSERK och marknadsrisken) med svensk
     Large Cap-bredd, OMXSPI (Börsdata, Momentum Swings index), hela Norden likaviktad (valhall.nordic_index,
     likvida bolag) med Nordens egen bredd, och OMXN40 (Yahoo) som kontroll.
  2. Samvariation, vem som leder (veckor), SMA200-korsningar (dagar före/efter) — valhall.nordic_probe.
  3. Sektorernas tidighet: Valhall-segmentens gröna vändningar mot marknadens grind (OMXS30 och Norden)
     och avkastningen 63/126 dagar efter.
  4. Grindvarianterna: Viking Nine (Norden 150) och BERSERK (gällande regler) med OMXS30, OMXSPI och Norden
     som injicerad marknad — motorerna är oförändrade, bara nordic_provider skiljer.

Körs manuellt i Actions (nordic-probe.yml). Resultatet skrivs i loggen och sparas i Gisten
(valhall_nordic_probe.json) om GITHUB_TOKEN finns. Env: BORSDATA_API_KEY. Inga regler ändras.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd

from valhall import nordic_probe as np_

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("valhall_nordic_probe")

BLOB = "valhall_nordic_probe.json"
START_YEAR = 2008
OMXS30, OMXSPI, NORDEN, OMXN40 = "OMXS30", "OMXSPI", "Norden (likaviktad)", "OMXN40 (Yahoo)"
NORDEN_LC = "Norden-index + Large Cap-bredd"       # isolerar indexbytet från breddbytet (bara Viking Nine)
PAIRS = ((NORDEN, OMXS30), (NORDEN, OMXSPI), (OMXSPI, OMXS30), (OMXN40, OMXS30), (NORDEN, OMXN40))
YAHOO_NORDIC = "^OMXN40"

NOTES = (
    "Proben mäter — kartan, grindarna och reglerna läser inte resultatet.",
    "Norden likaviktad = dagens noterade bolag med likviditetsfilter (överlevnadsbias): jämför varianterna, "
    "inte nivåerna.",
    "Large Cap-bredden är dagens svenska Large Cap-lista (Börsdata); Nordens bredd räknar varje aktie från sin "
    "50:e stängning.",
    "Grindvarianterna kör samma backtest (viking_backtest, berserk.backtest) — bara den injicerade marknaden skiljer.",
    "Rapportspärr, max två förluster per dag och riskspärren ingår inte (som i Valhall fas 3a).",
)


def _close(df) -> Optional[pd.Series]:
    if df is None or len(df) == 0 or "Close" not in df:
        return None
    s = df["Close"].astype(float).dropna()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s if len(s) else None


def series(api, getter: Callable, fr: dict, keep: dict, nordic_market: Optional[Callable] = None,
           index_fn: Optional[Callable] = None) -> dict:
    """{namn: {close, breadth, source, breadth_source}} för OMXS30, OMXSPI, Norden (likaviktad) och OMXN40.
    nordic_market(bars) = ovtlyr_nine.nordic_market, index_fn(api, hints, bars) = market_risk.borsdata_index."""
    import market_risk as mr
    import ovtlyr_nine as on
    import swing_backtest as sb
    nm = (nordic_market or on.nordic_market)(mr.BD_MAX_BARS) or {}
    lc_breadth = nm.get("breadth")
    lc_src = nm.get("breadth_source") or "Large Cap-bredd saknas"
    out = {OMXS30: {"close": np_._clean(nm.get("close")), "breadth": lc_breadth,
                    "source": nm.get("source") or "OMXS30 saknas", "breadth_source": lc_src}}
    spi, spi_src = (index_fn or mr.borsdata_index)(api, sb.INDEX_HINTS, mr.BD_MAX_BARS)
    out[OMXSPI] = {"close": np_._clean(spi), "breadth": lc_breadth, "source": spi_src, "breadth_source": lc_src}
    closes = {t: d["Close"] for t, d in (keep or {}).items() if d is not None and len(d) and "Close" in d}
    nb = np_.nordic_breadth(closes) if closes else None
    bsrc = (f"Nordens bredd: {len(closes)} aktier över EMA50 (varje aktie från sin 50:e stängning)" if nb is not None
            else "Nordens bredd saknas (för få aktier)")
    out[NORDEN] = {"close": np_._clean((fr.get("bench") or {}).get("nordic")), "breadth": nb, "breadth_source": bsrc,
                   "source": f"Norden likaviktad, likvida bolag (valhall.nordic_index, {len(closes)} aktier med kurs)"}
    y = _close(getter(YAHOO_NORDIC, "max"))
    out[OMXN40] = {"close": y, "breadth": None, "source": f"Yahoo Finance {YAHOO_NORDIC}" if y is not None
                   else f"{YAHOO_NORDIC} saknas (Yahoo)", "breadth_source": "ingen bredd"}
    return out


def sources(ser: dict) -> dict:
    """Källa och täckning per serie — så att resultatet går att läsa utan loggen."""
    out = {}
    for name, s in ser.items():
        c = s.get("close")
        out[name] = {"source": s.get("source"), "breadth_source": s.get("breadth_source"),
                     "bars": int(len(c)) if c is not None else 0,
                     "from": str(c.index[0].date()) if c is not None and len(c) else None,
                     "to": str(c.index[-1].date()) if c is not None and len(c) else None}
    return out


def probe(ser: dict, fr: dict) -> dict:
    """Samvariation, ledning och korsningar per par, och segmentens tidighet mot OMXS30 och Norden."""
    from valhall import segments as sgm
    close = {k: v.get("close") for k, v in ser.items()}
    comov, lead, cross = {}, {}, {}
    for a, b in PAIRS:
        if close.get(a) is None or close.get(b) is None:
            continue
        key = f"{a} mot {b}"
        comov[key] = np_.comovement(close[a], close[b])
        lead[key] = np_.lead_lag(close[a], close[b], names=(a, b))
        cross[key] = np_.crossing_lead(close[a], close[b])
    early = {}
    for market in (OMXS30, NORDEN):
        if close.get(market) is None:
            continue
        early[market] = {}
        for seg, frame in (fr.get("nordic") or {}).items():
            if frame is None or len(frame) == 0 or "status" not in frame:
                continue
            label = sgm.get(seg).label if sgm.get(seg) else seg
            early[market][label] = np_.earliness(frame["close"], frame["status"], close[market])
    return {"comovement": comov, "lead_lag": lead, "crossing_lead": cross, "earliness": early}


def gates(ser: dict, getter: Callable, today, viking_run: Optional[Callable] = None,
          berserk_run: Optional[Callable] = None) -> dict:
    """Viking Nine (Norden 150) och BERSERK (gällande regler) med varje marknad som grind — i R."""
    import viking_backtest as vbt
    from berserk import backtest as bbt
    from berserk import universe as buv
    lc = ser.get(OMXS30) or {}
    nord = ser.get(NORDEN) or {}
    vik, ber = {}, {}
    for name in (OMXS30, OMXSPI):
        s = ser.get(name) or {}
        if s.get("close") is not None:
            vik[name] = np_.provider(s["close"], lc.get("breadth"), s["source"], lc.get("breadth_source"))
            ber[name] = vik[name]
    if nord.get("close") is not None:
        vik[NORDEN_LC] = np_.provider(nord["close"], lc.get("breadth"), nord["source"], lc.get("breadth_source"))
        vik[NORDEN] = np_.provider(nord["close"], nord.get("breadth"), nord["source"], nord.get("breadth_source"))
        ber[NORDEN] = vik[NORDEN]
    tickers = list(dict.fromkeys(vbt.NORDIC_50 + vbt.NORDIC_OOS_100))
    vrun, brun = viking_run or vbt.run, berserk_run or bbt.run

    def _viking(p):
        log.info("Viking Nine: %d tickers, grind %s", len(tickers), p()["source"])
        return vrun(tickers, getter=getter, cfg=vbt.Config(start_year=START_YEAR), today=today, nordic_provider=p)

    def _berserk(p):
        log.info("BERSERK: %d tickers, grind %s", len(buv.NORDIC), p()["source"])
        return brun(list(buv.NORDIC), getter=getter, cfg=bbt.live_config(start_year=START_YEAR), today=today,
                    nordic_provider=p)

    return {"Viking Nine": np_.gate_variants(vik, _viking, vbt.metrics),
            "BERSERK": np_.gate_variants(ber, _berserk, vbt.metrics)}


def run(api=None, getter=None, data_getter=None, today=None, save=None, no_save: bool = False,
        no_gates: bool = False, frames_fn: Optional[Callable] = None, series_fn: Optional[Callable] = None,
        viking_run: Optional[Callable] = None, berserk_run: Optional[Callable] = None) -> dict:
    import valhall_fas3 as vf
    if api is None:
        from borsdata_api import get_api
        api = get_api()
    if getter is None:
        from market_prices import ohlcv as getter
    if data_getter is None:
        from market_prices import closes_volumes as data_getter
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    keep: dict = {}
    fr = (frames_fn or vf.frames)(api, getter, data_getter, keep=keep)
    ser = (series_fn or series)(api, getter, fr, keep)
    result = {"generated": datetime.now(tz=timezone.utc).isoformat(), "start_year": START_YEAR,
              "sources": sources(ser), **probe(ser, fr)}
    for name, s in result["sources"].items():
        log.info("%s: %s · %s · %d dagar %s–%s", name, s["source"], s["breadth_source"], s["bars"], s["from"], s["to"])
    if not no_gates:
        result["gates"] = gates(ser, getter, today, viking_run, berserk_run)
    result["summary"] = np_.summary(result)
    result["notes"] = list(NOTES)
    for line in result["summary"]:
        log.info("%s", line)
    if not no_save:
        if save is None:
            from gist_storage import save_blob as save
        result["saved"] = bool(save(BLOB, result))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-gates", action="store_true", help="bara serierna och tidigheten (inga backtester)")
    args = parser.parse_args(argv)
    run(no_save=args.no_save, no_gates=args.no_gates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
