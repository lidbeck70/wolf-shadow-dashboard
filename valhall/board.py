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
    series: dict = {}                                     # temats serie (drivare eller producentkorg) → komplexkorgen
    for s in sgm.SEGMENTS:
        close, vol = _ohlcv(getter, s.spdr)
        g = cycle.state(cycle.cycle_frame(close, vol, spy, s.necessity, cykel_base=spy), cycle.SEGMENT) \
            if close is not None else cycle.state(None)
        n_info = (nordic or {}).get(s.key) or {}
        n_idx = n_info.get("index")
        n_state = cycle.state(cycle.cycle_frame(n_idx, None, nordic_market if nordic_market is not None
                                                else nordic_all, s.necessity, cykel_base=nordic_all), cycle.SEGMENT) \
            if n_idx is not None else None
        themes = []
        for t in s.themes:
            sym, d = drivers.get(t, (None, None))
            if d is None:
                d = baskets.get(t) if t in baskets else producer_basket(t, getter)
                sym = "producentkorg" if d is not None else None
            nec = THEME_NECESSITY.get(t, s.necessity)
            if d is not None:
                series[t] = (sym, d)
            themes.append({"theme": t, "label": bth.label(t), "driver": sym, "necessity": nec,
                           "state": cycle.state(cycle.cycle_frame(d, None, spy, nec)) if d is not None
                           else cycle.state(None)})
        rows.append({"key": s.key, "label": s.label, "spdr": s.spdr, "ucits": list(s.ucits),
                     "necessity": s.necessity, "niche": s.niche, "global": g, "nordic": n_state,
                     "nordic_members": n_info.get("members"), "nordic_with_data": n_info.get("with_data"),
                     "themes": themes})
    return {"segments": rows, "complexes": complex_rows(rows, series, spy),
            "when": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}


def complex_rows(rows: list, series: dict, spy: Optional[pd.Series]) -> list:
    """En rad per råvarukomplex: komplexdrivaren (likaviktad korg av temadrivarna) genom cykelmotorn, temana
    under komplexet och hur många av dem som vänder/är starka. Analysen 2026-10-24 §5.1 (beslut 1–2)."""
    from valhall import complexes as vcx
    by_cx = vcx.complex_themes(rows)
    out = []
    for key, label in vcx.COMPLEXES:
        syms, basket = vcx.basket(key, series)
        st = cycle.state(cycle.cycle_frame(basket, None, spy, vcx.NECESSITY[key])) if basket is not None \
            else cycle.state(None)
        ths = by_cx.get(key) or []
        turning = sum(1 for t in ths if (t.get("state") or {}).get("phase") in (cycle.VANDER, cycle.STARK))
        out.append({"key": key, "label": label, "driver": ("korg: " + ", ".join(syms)) if syms else None,
                    "necessity": vcx.NECESSITY[key], "state": st,
                    "themes": [{"theme": t["theme"], "label": t.get("label"), "state": t.get("state")} for t in ths],
                    "themes_turning": turning, "n_themes": len(ths)})
    return out


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


def turns(board: dict) -> list:
    """Vändningar som pågår i dag, en post per ruta: segment (Norden/globalt: RÖD → GRÖN), tema och komplex
    (VÄNDER). Fälten kind/since/days/confirmed/confirmed_since ur cycle.turn_state — "VÄNDNING — köp i dag" i
    kartan, Thors chip och händelsemotorn läser den här listan."""
    out = []

    def add(st, **meta):
        t = (st or {}).get("turn") or {}
        if t.get("kind"):
            out.append({**meta, **{k: t.get(k) for k in ("kind", "since", "days", "confirmed", "confirmed_since")},
                        "status": (st or {}).get("status"), "phase": (st or {}).get("phase"),
                        "cykel": (st or {}).get("cykel")})
    for r in board.get("segments") or []:
        add(r.get("global"), level="segment", key=r["key"], label=r["label"], where="global")
        add(r.get("nordic"), level="segment", key=r["key"], label=r["label"], where="Norden")
        for t in r.get("themes") or []:
            add(t.get("state"), level="tema", key=t["theme"], label=t.get("label"), segment=r["key"], where="tema")
    for c in board.get("complexes") or []:
        add(c.get("state"), level="komplex", key=c["key"], label=c["label"], where="komplex",
            themes_turning=c.get("themes_turning"), n_themes=c.get("n_themes"))
    out.sort(key=lambda x: (not x["confirmed"], x.get("days") or 0))
    return out


def cycle_starts(board: dict) -> list:
    """Segment, teman och komplex där en cykel startar nu (BAISSE · VÄNDER) — det contrarian-strategin letar efter."""
    out = []
    for r in board.get("segments") or []:
        for where, st in (("global", r["global"]), ("Norden", r.get("nordic"))):
            if st and st.get("cycle_start"):
                out.append({"segment": r["label"], "key": r["key"], "where": where, "since": st.get("status_since")})
        for t in r["themes"]:
            if t["state"].get("cycle_start"):
                out.append({"segment": f"{r['label']} · {t['label']}", "key": r["key"], "theme": t["theme"],
                            "where": "tema", "since": t["state"].get("status_since")})
    for c in board.get("complexes") or []:
        if (c.get("state") or {}).get("cycle_start"):
            out.append({"segment": c["label"], "key": c["key"], "complex": c["key"], "where": "komplex",
                        "since": c["state"].get("status_since")})
    return out
