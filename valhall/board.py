"""
valhall/board.py — Valhalls karta idag: en rad per segment och råvarutema.

  global   segmentets SPDR-fond (signal; handel i UCITS-ETF hos Nordnet) — cykelläget mot S&P 500
  nordic   det nordiska sektorindexet (där swingaffärerna görs) — cykelläget mot hela Norden
  themes   BERSERK:s råvaruteman under segmentet (energi, material, dagligvaror), med temats drivare;
           teman utan drivare (kol, lax) får en likaviktad korg av temats producenter. Råvaror mäts
           absolut, som i Blindspot.

Allt med injicerbar hämtare (getter(ticker, period) → OHLCV) så att den schemalagda körningen, sidan
och testerna delar logiken. Nödvändigheten per tema är Blindspots (blindspot/theme_board.py) där den
finns, annars segmentets.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from berserk import themes as bth
from berserk import universe as buv
from valhall import cycle
from valhall import nordic_index as ni
from valhall import segments as sgm

PERIOD = "15y"
# Blindspots nödvändighet per tema (NECESSITY_* i blindspot/theme_board.py); övriga ärver segmentets
THEME_NECESSITY = {"uran": 90, "koppar": 95, "olja": 90, "silver": 75, "guld": 70, "naturgas": 80,
                   "naturgas_eu": 80, "kol": 60, "sallsynta": 85, "platina": 70, "palladium": 65, "vete": 90,
                   "majs": 70, "soja": 70, "socker": 70, "skog": 60, "kaffe": 50, "kakao": 45,
                   # BERSERKs övriga teman (fas 8) — samma värden som brädan
                   "aluminium": 80, "zink_nickel": 80, "stal": 85, "jarnmalm": 80, "litium": 70, "godsel": 85,
                   "lax": 50, "tank": 65, "torrbulk": 65}


def _ohlcv(getter: Callable, symbol: str, period: str = PERIOD):
    try:
        df = getter(symbol, period)
    except Exception:
        return None, None
    if df is None or len(df) == 0 or "Close" not in df:
        return None, None
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    vol = df["Volume"].astype(float) if "Volume" in df else None
    return df["Close"].astype(float), vol


def build(getter: Optional[Callable] = None, nordic: Optional[dict] = None,
          nordic_market: Optional[pd.Series] = None, drivers: Optional[dict] = None,
          today=None, baskets: Optional[dict] = None) -> dict:
    """{segments: [rad], when}. nordic = nordic_index.load() (valfri; nyckeln nordic_index.ALL = hela
    Norden), nordic_market = OMXS30 (hat: relativ styrka), drivers = {tema: (symbol, serie)} (förval
    berserk.live.load_drivers), baskets = {tema: serie} för teman utan drivare (förval: producentkorg)."""
    if getter is None:
        from market_prices import ohlcv as getter
    spy, _v = _ohlcv(getter, "SPY")
    if drivers is None:
        from berserk import live
        drivers = live.load_drivers([t for s in sgm.SEGMENTS for t in s.themes], getter, today)
    nordic_all = ((nordic or {}).get(ni.ALL) or {}).get("index")
    baskets = baskets if baskets is not None else {}
    rows = []
    for s in sgm.SEGMENTS:
        close, vol = _ohlcv(getter, s.spdr)
        g = cycle.state(cycle.cycle_frame(close, vol, spy, s.necessity, cykel_base=spy)) \
            if close is not None else cycle.state(None)
        n_info = (nordic or {}).get(s.key) or {}
        n_idx = n_info.get("index")
        n_state = cycle.state(cycle.cycle_frame(n_idx, None, nordic_market if nordic_market is not None
                                                else nordic_all, s.necessity, cykel_base=nordic_all)) \
            if n_idx is not None else None
        themes = []
        for t in s.themes:
            sym, d = drivers.get(t, (None, None))
            if d is None:
                d = baskets.get(t) if t in baskets else producer_basket(t, getter)
                sym = "producentkorg" if d is not None else None
            nec = THEME_NECESSITY.get(t, s.necessity)
            themes.append({"theme": t, "label": bth.label(t), "driver": sym, "necessity": nec,
                           "state": cycle.state(cycle.cycle_frame(d, None, spy, nec)) if d is not None
                           else cycle.state(None)})
        rows.append({"key": s.key, "label": s.label, "spdr": s.spdr, "ucits": list(s.ucits),
                     "necessity": s.necessity, "niche": s.niche, "global": g, "nordic": n_state,
                     "nordic_members": n_info.get("members"), "nordic_with_data": n_info.get("with_data"),
                     "themes": themes})
    return {"segments": rows, "when": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}


def producer_basket(theme: str, getter: Callable, period: str = PERIOD) -> Optional[pd.Series]:
    """Likaviktad korg av temats producenter (berserk.universe) — drivare för teman utan råvaruserie."""
    closes = {}
    for t in buv.by_theme().get(theme, []):
        if buv.kind_of(t) != "producent":
            continue
        c, _v = _ohlcv(getter, t, period)
        if c is not None:
            closes[t] = c
    return ni.equal_weight_index(closes, min_members=2)


def cycle_starts(board: dict) -> list:
    """Segment och teman där en cykel startar nu (BAISSE · VÄNDER) — det contrarian-strategin letar efter."""
    out = []
    for r in board.get("segments") or []:
        for where, st in (("global", r["global"]), ("Norden", r.get("nordic"))):
            if st and st.get("cycle_start"):
                out.append({"segment": r["label"], "key": r["key"], "where": where, "since": st.get("status_since")})
        for t in r["themes"]:
            if t["state"].get("cycle_start"):
                out.append({"segment": f"{r['label']} · {t['label']}", "key": r["key"], "theme": t["theme"],
                            "where": "tema", "since": t["state"].get("status_since")})
    return out
