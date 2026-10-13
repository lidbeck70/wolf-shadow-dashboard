"""
valhall/nordic_index.py — nordiska sektorindex, likaviktade, ur Börsdatas sektorindelning.

Det finns inga nordiska sektorindex i panelen (Viking Nine mäter sektorn mot amerikanska SPDR-
fonder). Swingaffärerna görs i Norden, så Valhall räknar segmentets cykel även på Norden:

  members()             Börsdatas instrument (sectorId, branchId, marknad) → {segment: [Yahoo-tickers]}
  equal_weight_index()  dagens snittavkastning för medlemmarna med kurs, kedjad från 100. Dagsrörelser
                        över 60 % räknas inte (ojusterade splittar och datafel — samma gräns som BERSERK:s
                        datavakt); minst tre medlemmar med kurs per dag.

Medlemmarna är dagens bolag (överlevnadsbias: avnoterade saknas) — indexet visar sektorns läge, inte
en investerbar avkastning.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import pandas as pd

from berserk import signals as bsg
from valhall import segments as sgm

MIN_MEMBERS = 3


def members(instruments: Iterable[dict], nordic_ids: set, to_yf: Callable) -> dict:
    """{segment: [ticker]} för nordiska aktier. to_yf(ticker, marketId) → Yahoo-ticker."""
    out: dict = {s.key: [] for s in sgm.SEGMENTS}
    for inst in instruments:
        if inst.get("marketId") not in nordic_ids:
            continue
        seg = sgm.segment_for(inst.get("sectorId"), inst.get("branchId"))
        tk = to_yf(inst.get("ticker"), inst.get("marketId")) if inst.get("ticker") else None
        if seg and tk:
            out[seg].append(tk)
    return {k: sorted(set(v)) for k, v in out.items()}


def equal_weight_index(closes: dict, min_members: int = MIN_MEMBERS,
                       max_abs_ret: float = bsg.JUMP_MAX) -> Optional[pd.Series]:
    """Likaviktat index (start 100) ur medlemmarnas stängningskurser. None när underlaget är för tunt."""
    series = {}
    for t, s in (closes or {}).items():
        if s is None or len(s) == 0:
            continue
        s = s.astype(float).dropna()
        if getattr(s.index, "tz", None) is not None:
            s = s.copy()
            s.index = s.index.tz_localize(None)
        series[t] = s[~s.index.duplicated(keep="last")]
    if len(series) < min_members:
        return None
    df = pd.DataFrame(series).sort_index()
    rets = df.pct_change(fill_method=None)
    rets = rets.where(rets.abs() <= max_abs_ret)
    n = rets.notna().sum(axis=1)
    mean = rets.mean(axis=1).where(n >= min_members)
    valid = mean.dropna()
    if valid.empty:
        return None
    pos = df.index.get_loc(valid.index[0])
    mean = mean.loc[valid.index[0]:]
    idx = (1 + mean.fillna(0.0)).cumprod() * 100
    base = pd.Series([100.0], index=[df.index[pos - 1]]) if pos > 0 else pd.Series(dtype=float)
    idx = pd.concat([base, idx])                     # dagen före första avkastningen = 100
    idx.name = "index"
    return idx


def load(api=None, closes_getter: Optional[Callable] = None, period: str = "15y",
         max_members: Optional[int] = None) -> dict:
    """{segment: {"index": Series|None, "members": int, "with_data": int}} ur Börsdata + Yahoo.

    closes_getter(tickers, period) → {ticker: Close} (förval market_prices.closes, en batch per segment)."""
    import markets as _markets
    if api is None:
        from borsdata_api import get_api
        api = get_api()
    if closes_getter is None:
        from market_prices import closes as closes_getter
    table = _markets.load(api=api)
    nordic = _markets.nordic_stock_ids(table)
    mem = members(api.get_instruments(), nordic, lambda t, m: _markets.to_yf(t, m, table))
    out = {}
    for seg, tickers in mem.items():
        tickers = tickers[:max_members] if max_members else tickers
        closes = closes_getter(tickers, period) if tickers else {}
        out[seg] = {"index": equal_weight_index(closes), "members": len(tickers),
                    "with_data": sum(1 for s in closes.values() if s is not None and len(s))}
    return out
