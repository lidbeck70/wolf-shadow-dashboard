"""
valhall/nordic_index.py — nordiska sektorindex, likaviktade, ur Börsdatas sektorindelning.

Det finns inga nordiska sektorindex i panelen (Viking Nine mäter sektorn mot amerikanska SPDR-
fonder). Swingaffärerna görs i Norden, så Valhall räknar segmentets cykel även på Norden:

  members()             Börsdatas instrument (sectorId, branchId, marknad) → {segment: [Yahoo-tickers]}
  equal_weight_index()  dagens snittavkastning för medlemmarna med kurs, kedjad från 100. Dagsrörelser
                        över 60 % räknas inte (ojusterade splittar och datafel — samma gräns som BERSERK:s
                        datavakt); minst tre medlemmar med kurs per dag.
  likviditetsfilter     en aktie räknas en dag bara om snittomsättningen de 60 dagarna FÖRE dagen var minst
                        MIN_TURNOVER_SEK (1 milj kr) — dag för dag, så att småbolag på First North inte
                        får styra indexet och inget facit används (dagens likviditet läggs inte på historiken).

Medlemmarna är dagens bolag (överlevnadsbias: avnoterade saknas) — indexet visar sektorns läge, inte
en investerbar avkastning.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import pandas as pd

from berserk import signals as bsg
from valhall import segments as sgm

MIN_MEMBERS = 3
MIN_TURNOVER_SEK = 1_000_000
TURNOVER_DAYS = 60
FX_TO_SEK = {"SEK": 1.0, "NOK": 0.95, "DKK": 1.55, "EUR": 11.3, "USD": 10.5, "GBP": 13.5}   # som insider_scan


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


def liquid_mask(turnover: pd.DataFrame, min_turnover: float = MIN_TURNOVER_SEK,
                days: int = TURNOVER_DAYS) -> pd.DataFrame:
    """True där snittomsättningen de `days` dagarna före dagen (i kr) är minst min_turnover."""
    avg = turnover.rolling(days, min_periods=max(10, days // 3)).mean().shift(1)
    return avg >= min_turnover


def equal_weight_index(closes: dict, min_members: int = MIN_MEMBERS,
                       max_abs_ret: float = bsg.JUMP_MAX, turnover: Optional[dict] = None,
                       min_turnover: float = MIN_TURNOVER_SEK) -> Optional[pd.Series]:
    """Likaviktat index (start 100) ur medlemmarnas stängningskurser. turnover = {ticker: omsättning i kr
    per dag} slår på likviditetsfiltret. None när underlaget är för tunt."""
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
    if turnover:
        tv = pd.DataFrame({t: s for t, s in turnover.items() if t in df.columns}).reindex(df.index)
        mask = liquid_mask(tv, min_turnover).reindex(columns=df.columns).fillna(False)
        rets = rets.where(mask)
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


ALL = "_all"                       # hela Norden (alla segment) — marknaden som sektorerna mäts mot


def members_with_currency(instruments: Iterable[dict], nordic_ids: set, to_yf: Callable) -> dict:
    """{ticker: valuta} för nordiska aktier (omsättningen räknas om till kronor)."""
    return {to_yf(i.get("ticker"), i.get("marketId")): str(i.get("stockPriceCurrency") or "SEK").upper()
            for i in instruments if i.get("marketId") in nordic_ids and i.get("ticker")}


def load(api=None, data_getter: Optional[Callable] = None, period: str = "max",
         max_members: Optional[int] = None, min_turnover: float = MIN_TURNOVER_SEK) -> dict:
    """{segment: {"index", "members", "with_data", "liquid"}} ur Börsdata + Yahoo, plus ALL = hela Norden.
    data_getter(tickers, period) → {ticker: DataFrame[Close, Volume]} (förval market_prices.closes_volumes,
    en batch per segment). min_turnover = 0 stänger av likviditetsfiltret."""
    import markets as _markets
    if api is None:
        from borsdata_api import get_api
        api = get_api()
    if data_getter is None:
        from market_prices import closes_volumes as data_getter
    table = _markets.load(api=api)
    nordic = _markets.nordic_stock_ids(table)
    instruments = api.get_instruments()
    to_yf = lambda t, m: _markets.to_yf(t, m, table)  # noqa: E731
    mem = members(instruments, nordic, to_yf)
    ccy = members_with_currency(instruments, nordic, to_yf)
    return build_indices(mem, ccy, data_getter, period, max_members, min_turnover)


def build_indices(mem: dict, ccy: dict, data_getter: Callable, period: str = "max",
                  max_members: Optional[int] = None, min_turnover: float = MIN_TURNOVER_SEK,
                  keep: Optional[dict] = None) -> dict:
    """Indexen ur medlemslistan (testbar utan Börsdata). keep (valfri dict) fylls med {ticker: DataFrame}
    — swingskanningen återanvänder samma kurser."""
    out, every_c, every_t = {}, {}, {}
    for seg, tickers in mem.items():
        tickers = tickers[:max_members] if max_members else tickers
        data = data_getter(tickers, period) if tickers else {}
        if keep is not None:
            keep.update({t: d for t, d in data.items() if d is not None and len(d)})
        closes = {t: d["Close"] for t, d in data.items() if d is not None and len(d) and "Close" in d}
        turnover = {t: d["Close"] * d["Volume"] * FX_TO_SEK.get(ccy.get(t, "SEK"), 1.0)
                    for t, d in data.items() if d is not None and "Volume" in d} if min_turnover else None
        every_c.update(closes)
        if turnover:
            every_t.update(turnover)
        out[seg] = {"index": equal_weight_index(closes, turnover=turnover, min_turnover=min_turnover),
                    "members": len(tickers), "with_data": len(closes), "liquid": _liquid_now(turnover, min_turnover)}
    out[ALL] = {"index": equal_weight_index(every_c, turnover=every_t or None, min_turnover=min_turnover),
                "members": len(every_c), "with_data": len(every_c), "liquid": _liquid_now(every_t, min_turnover)}
    return out


def _liquid_now(turnover: Optional[dict], min_turnover: float) -> Optional[int]:
    """Hur många medlemmar som klarar likviditetskravet idag."""
    if not turnover:
        return None
    return int(sum(1 for s in turnover.values()
                   if len(s) and s.tail(TURNOVER_DAYS).mean() >= min_turnover))
