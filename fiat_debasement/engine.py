"""
fiat_debasement/engine.py — beräkningarna. Rena funktioner på pandas-serier
(DatetimeIndex → värde); ingen hämtning, ingen Streamlit.

Saknade värden ger None/NaN — aldrig 0, aldrig en interpolation. Jämförelser
"ett år bakåt" görs på kalenderdatum (samma månad förra året), inte på antal
rader, så luckor i en serie inte flyttar jämförelsepunkten.

  yoy                  årsförändring %: (x_t / x_t−12 mån − 1) × 100
  cagr                 årlig tillväxt %: ((x_t / x_t−n år)^(1/n) − 1) × 100
  cagr_since           samma från ett valt startdatum (bråkdel av år)
  purchasing_power     köpkraftsindex: 100 × KPI_start / KPI_t
  cumulative_change    KPI:s totala förändring % sedan start
  gap                  Monetary Gap = penningmängdstillväxt − real BNP-tillväxt (procentenheter)
  price_in             tillgångens pris i SEK/EUR/USD
  fiat_vs_asset        valutans värde mätt i tillgången, start = 100
  normalize_100        valfri serie, start = 100
  real_price           pris i fast penningvärde (KPI-justerat till ett basdatum)
  percentile           senaste värdets percentil i egen historik (0–100)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def _clean(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None or len(s) == 0:
        return None
    s = s.astype(float).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    return s if len(s) else None


def _positive(*vals) -> bool:
    return all(v is not None and not math.isnan(v) and v > 0 for v in vals)


def monthly(s: Optional[pd.Series], how: str = "last") -> Optional[pd.Series]:
    """Månadsserie med periodens första dag som datum (som FRED/ECB/SCB)."""
    s = _clean(s)
    if s is None:
        return None
    r = s.resample("MS")
    out = r.last() if how == "last" else r.mean()
    return out.dropna()


def quarterly(s: Optional[pd.Series], how: str = "mean") -> Optional[pd.Series]:
    s = _clean(s)
    if s is None:
        return None
    r = s.resample("QS")
    out = r.mean() if how == "mean" else r.last()
    return out.dropna()


def value_at(s: Optional[pd.Series], date) -> Optional[float]:
    """Värdet på exakt det datumet (eller None)."""
    s = _clean(s)
    if s is None:
        return None
    d = pd.Timestamp(date)
    return float(s[d]) if d in s.index else None


def value_on_or_after(s: Optional[pd.Series], date) -> tuple:
    """(datum, värde) för första observationen på eller efter datumet — (None, None) om ingen."""
    s = _clean(s)
    if s is None:
        return None, None
    later = s[s.index >= pd.Timestamp(date)]
    return (later.index[0], float(later.iloc[0])) if len(later) else (None, None)


def latest(s: Optional[pd.Series]) -> tuple:
    s = _clean(s)
    return (s.index[-1], float(s.iloc[-1])) if s is not None else (None, None)


def _shift_years(idx, years: int):
    return pd.DatetimeIndex(idx) + pd.DateOffset(years=years)


def yoy(s: Optional[pd.Series]) -> Optional[pd.Series]:
    """Årsförändring i % mot samma datum ett år tidigare. Saknas jämförelsepunkten → NaN (tas bort)."""
    s = _clean(s)
    if s is None:
        return None
    prev = pd.Series(s.values, index=_shift_years(s.index, 1))
    base = prev.reindex(s.index)
    ok = (base > 0) & (s > 0)
    out = ((s / base - 1) * 100)[ok]
    return out.dropna() if len(out.dropna()) else None


def cagr(s: Optional[pd.Series], years: int, at=None) -> Optional[float]:
    """Årlig tillväxttakt i % över `years` år, slutande på `at` (default senaste). None om
    startpunkten saknas eller något värde ≤ 0."""
    s = _clean(s)
    if s is None or years <= 0:
        return None
    end_d = pd.Timestamp(at) if at is not None else s.index[-1]
    end = value_at(s, end_d)
    start = value_at(s, end_d - pd.DateOffset(years=years))
    if not _positive(start, end):
        return None
    return ((end / start) ** (1 / years) - 1) * 100


def cagr_since(s: Optional[pd.Series], start_date) -> Optional[float]:
    """Årlig tillväxt i % från första observationen på/efter start_date till senaste."""
    s = _clean(s)
    d0, v0 = value_on_or_after(s, start_date)
    d1, v1 = latest(s)
    if d0 is None or d1 is None or not _positive(v0, v1):
        return None
    yrs = (d1 - d0).days / 365.25
    if yrs <= 0:
        return None
    return ((v1 / v0) ** (1 / yrs) - 1) * 100


def change_pct(s: Optional[pd.Series], years: float, at=None) -> Optional[float]:
    """Total förändring i % över `years` år (för 1Y/5Y/10Y-fönster). Startvärdet = senaste
    observationen på eller före startdatumet (dagliga priser saknar helger)."""
    s = _clean(s)
    if s is None:
        return None
    end_d = pd.Timestamp(at) if at is not None else s.index[-1]
    end = s[s.index <= end_d]
    if not len(end):
        return None
    start_d = end_d - pd.DateOffset(days=int(round(years * 365.25)))
    before = s[s.index <= start_d]
    if not len(before) or (start_d - before.index[-1]).days > 10:
        return None                                              # ingen observation nära startdatumet
    v0, v1 = float(before.iloc[-1]), float(end.iloc[-1])
    return (v1 / v0 - 1) * 100 if _positive(v0, v1) else None


def purchasing_power(cpi: Optional[pd.Series], start_date=None, base: float = 100.0) -> Optional[pd.Series]:
    """Köpkraftsindex: base × KPI_start / KPI_t från första observationen på/efter start.
    KPI +25 % → 100 / 1,25 = 80."""
    s = _clean(cpi)
    if s is None:
        return None
    d0, v0 = value_on_or_after(s, start_date if start_date is not None else s.index[0])
    if d0 is None or not _positive(v0):
        return None
    part = s[s.index >= d0]
    part = part[part > 0]
    return base * v0 / part


def cumulative_change(cpi: Optional[pd.Series], start_date) -> Optional[float]:
    """KPI:s totala förändring i % från start till senaste."""
    s = _clean(cpi)
    d0, v0 = value_on_or_after(s, start_date)
    d1, v1 = latest(s)
    if d0 is None or not _positive(v0, v1):
        return None
    return (v1 / v0 - 1) * 100


def gap(money_growth: Optional[float], gdp_growth: Optional[float]) -> Optional[float]:
    """Monetary Gap i procentenheter: penningmängdens tillväxt minus real BNP-tillväxt.
    Mäter penningmängd relativt real produktion — inte faktisk inflation."""
    if money_growth is None or gdp_growth is None or any(map(math.isnan, (money_growth, gdp_growth))):
        return None
    return money_growth - gdp_growth


def gap_series(money: Optional[pd.Series], real_gdp: Optional[pd.Series]) -> Optional[pd.Series]:
    """Monetary Gap per kvartal: YoY för penningmängdens kvartalsmedel minus real BNP:s YoY."""
    m, g = yoy(quarterly(money)), yoy(quarterly(real_gdp, how="mean"))
    if m is None or g is None:
        return None
    out = (m - g.reindex(m.index)).dropna()
    return out if len(out) else None


def price_in(asset_usd: Optional[pd.Series], currency: str, fx: Optional[pd.Series] = None) -> Optional[pd.Series]:
    """Tillgångens pris i valutan. fx: SEK per USD för SEK, USD per EUR för EUR (som config.FX).
    Bara datum där båda finns används — växelkursen fylls aldrig i."""
    a = _clean(asset_usd)
    if a is None:
        return None
    if currency == "USD":
        return a
    f = _clean(fx)
    if f is None:
        return None
    both = pd.concat({"a": a, "fx": f}, axis=1, join="inner").dropna()
    both = both[both["fx"] > 0]
    if not len(both):
        return None
    if currency == "SEK":
        return both["a"] * both["fx"]
    if currency == "EUR":
        return both["a"] / both["fx"]
    raise ValueError(f"okänd valuta: {currency}")


def normalize_100(s: Optional[pd.Series], start_date=None) -> Optional[pd.Series]:
    s = _clean(s)
    if s is None:
        return None
    d0, v0 = value_on_or_after(s, start_date if start_date is not None else s.index[0])
    if d0 is None or not _positive(v0):
        return None
    return 100 * s[s.index >= d0] / v0


def fiat_vs_asset(asset_price_in_ccy: Optional[pd.Series], start_date=None) -> Optional[pd.Series]:
    """Hur mycket av tillgången en valutaenhet köper, start = 100: 100 × pris_start / pris_t.
    Faller serien från 100 till 40 köper valutan 60 % mindre av tillgången än vid start."""
    s = _clean(asset_price_in_ccy)
    if s is None:
        return None
    d0, v0 = value_on_or_after(s, start_date if start_date is not None else s.index[0])
    if d0 is None or not _positive(v0):
        return None
    part = s[(s.index >= d0) & (s > 0)]
    return 100 * v0 / part


def units_of_100(asset_price_in_ccy: Optional[pd.Series], start_date=None) -> Optional[pd.Series]:
    """'What happened to 100 units?': värdet i valutan av den tillgångsmängd som 100 enheter
    köpte vid start. (100 / pris_start) × pris_t."""
    return normalize_100(asset_price_in_ccy, start_date)


def real_price(price: Optional[pd.Series], cpi: Optional[pd.Series], base_date) -> Optional[pd.Series]:
    """Pris i fast penningvärde vid base_date: pris_t × KPI_bas / KPI_t (månadsvis)."""
    p, c = monthly(price), monthly(cpi)
    if p is None or c is None:
        return None
    _d, base = value_on_or_after(c, base_date)
    if not _positive(base):
        return None
    both = pd.concat({"p": p, "c": c}, axis=1, join="inner").dropna()
    both = both[both["c"] > 0]
    return (both["p"] * base / both["c"]) if len(both) else None


def percentile(s: Optional[pd.Series], value: Optional[float] = None, since=None) -> Optional[float]:
    """Percentil (0–100) för value (default senaste) i seriens egen historik sedan `since`."""
    s = _clean(s)
    if s is None:
        return None
    if since is not None:
        s = s[s.index >= pd.Timestamp(since)]
    if len(s) < 2:
        return None
    v = float(s.iloc[-1]) if value is None else float(value)
    return float((s < v).sum() + 0.5 * (s == v).sum()) / len(s) * 100
