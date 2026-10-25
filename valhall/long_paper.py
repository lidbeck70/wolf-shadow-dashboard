"""
valhall/long_paper.py — papperskontona för långsidans cykeltest: E1 billig och hatad mot E3 bekräftad vändning,
från i dag och framåt (data som inget test har sett). Inträdena är exakt testets (valhall.long_study,
docs/forskning/lang_cykel_spec.md) och räknas på samma serier:

  Producent-ETF:er    XOP, XES, FCG, XME, GDX, GDXJ, SIL, COPX, URA, PICK, SLX, WOOD, MOO, REMX, LIT
  Sektorfonder        de 11 SPDR-fonderna, cykelläget mot SPY
  Nordiska storbolag  när en sektorfond ger en händelse: sektorns 5 största nordiska bolag efter dagens
                      börsvärde (Börsdata KPI 50) — framåt finns ingen överlevnadsbias

Kontona (valhall/paper.py): "lang_e1" och "lang_e3". En enhet per händelse, köp nästa öppning, ingen stopp,
sälj efter två år (testets huvudhorisont). Ingen regel ändras — kontona mäter bara.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from valhall import event_study as es
from valhall import long_study as lst

E1_KEY, E3_KEY = "lang_e1", "lang_e3"
ENTRY_KEY = {lst.E1: E1_KEY, lst.E3: E3_KEY}
SIGNAL_DAYS = 5                    # händelsen räknas som dagens signal i fem handelsdagar (missad körning)
TOP_N = 5
PRODUCER_SEGMENT = {"XOP": "energy", "XES": "energy", "FCG": "energy", "URA": "energy",
                    "XME": "materials", "GDX": "materials", "GDXJ": "materials", "SIL": "materials",
                    "COPX": "materials", "PICK": "materials", "SLX": "materials", "WOOD": "materials",
                    "REMX": "materials", "LIT": "materials", "MOO": "consumer_staples"}


def recent_events(events: dict, index: pd.DatetimeIndex, days: int = SIGNAL_DAYS) -> dict:
    """{inträde: händelsedag} för händelser (efter testets rensning) bland de `days` sista handelsdagarna."""
    out = {}
    if index is None or not len(index):
        return out
    first = index[max(0, len(index) - days)]
    for entry, dates in events.items():
        kept = es.decluster([d for d in dates if d in index], index, lst.DECLUSTER)
        if kept and kept[-1] >= first:
            out[entry] = kept[-1]
    return out


def series_events(getter: Callable) -> list:
    """[(symbol, segment, grupp, {inträde: händelsedag})] för producenter och sektorfonder med en färsk händelse."""
    from valhall import cycle
    from valhall import segments as sgm
    from valhall_event_study import _close
    spy = _close(getter("SPY", "15y"))
    out = []
    pairs = [(sym, seg, lst.PRODUCENT, cycle.RAVARA) for sym, seg in PRODUCER_SEGMENT.items()]
    pairs += [(s.spdr, s.key, lst.SEKTOR, cycle.SEGMENT) for s in sgm.SEGMENTS]
    for sym, seg, group, kind in pairs:
        df = getter(sym, "15y")
        c = _close(df)
        if c is None:
            continue
        fr = cycle.cycle_frame(c, df["Volume"] if "Volume" in df else None, spy, 50, cykel_base=spy)
        ev = {e: d for e, d in lst.entry_events(fr, kind).items() if e in ENTRY_KEY}
        rec = recent_events(ev, fr.index)
        if rec:
            out.append((sym, seg, group, rec))
    return out


def nordic_top(mcap: dict, instruments: list, nordic_ids: set, to_yf: Callable, n: int = TOP_N) -> dict:
    """{segment: [(yf-ticker, namn)]} — de n största nordiska bolagen per segment efter börsvärde i kronor,
    ett aktieslag per bolag. mcap = {insId: börsvärde} (Börsdata KPI 50)."""
    from valhall import nordic_index as ni
    from valhall import picks as vp
    from valhall import segments as sgm
    rows: dict = {}
    for i in instruments or []:
        iid, v = i.get("insId"), mcap.get(i.get("insId"))
        if iid is None or v is None or i.get("marketId") not in nordic_ids or not i.get("ticker"):
            continue
        seg = sgm.segment_for(i.get("sectorId"), i.get("branchId"))
        if not seg:
            continue
        fx = ni.FX_TO_SEK.get(str(i.get("reportCurrency") or i.get("stockPriceCurrency") or "SEK").upper(), 1.0)
        rows.setdefault(seg, []).append((float(v) * fx, to_yf(i["ticker"], i["marketId"]), str(i.get("name") or "")))
    out = {}
    for seg, rs in rows.items():
        seen, picked = set(), []
        for _v, t, name in sorted(rs, key=lambda r: -r[0]):
            k = vp.company_key({"name": name, "ticker": t})
            if k in seen:
                continue
            seen.add(k)
            picked.append((t, name))
            if len(picked) == n:
                break
        out[seg] = picked
    return out


def signals(found: list, top: Optional[dict] = None) -> list:
    """Papperets köpsignaler: ETF:en själv, och för sektorfonderna även sektorns nordiska storbolag."""
    from valhall import paper as vpa
    out = []
    for sym, seg, group, rec in found:
        for entry, day in rec.items():
            key = ENTRY_KEY[entry]
            why = f"{entry} {pd.Timestamp(day).date()} på {sym}"
            out.append(vpa._sig(key, sym, sym, seg, None, False, f"{why} ({group})"))
            if group == lst.SEKTOR:
                for t, name in (top or {}).get(seg) or []:
                    out.append(vpa._sig(key, t, name, seg, None, True, f"{why} · sektorns storbolag"))
    return out


def run(getter: Callable, api=None, instruments: Optional[list] = None, nordic_ids: Optional[set] = None,
        to_yf: Optional[Callable] = None) -> list:
    """Dagens signaler till lang_e1/lang_e3. Storbolagen bara när en sektorfond har en händelse och Börsdata svarar."""
    found = series_events(getter)
    top = None
    if api is not None and instruments and nordic_ids and to_yf and any(g == lst.SEKTOR for _s, _k, g, _r in found):
        try:
            top = nordic_top(api.get_kpi_screener_dict(50), instruments, nordic_ids, to_yf)
        except Exception:
            top = None
    return signals(found, top)
