"""
fiat_debasement/snapshot.py — nyckeltalen per valuta (FIAT OVERVIEW), var och
ett med sin källa och sitt datum så att varje siffra går att spåra till
rådata.

Begreppen hålls isär: penningmängdens tillväxt, KPI, real BNP, Monetary Gap,
statsskuld och valutans köpkraft mot guld är separata mått. Inget av dem
kallas "inflation" om det inte är KPI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from fiat_debasement import config as cfg
from fiat_debasement import data as fd
from fiat_debasement import engine as fe


@dataclass
class Metric:
    value: Optional[float] = None                  # None = DATA UNAVAILABLE
    as_of: Optional[str] = None                    # datum för senaste observationen som ingår
    source: str = ""                               # serie(r) som ligger bakom
    unit: str = "%"
    stale: bool = False
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None


@dataclass
class Snapshot:
    currency: str
    metrics: dict = field(default_factory=dict)    # nyckel → Metric
    loaded: dict = field(default_factory=dict)     # begrepp → data.Loaded (rådata och försök)

    def get(self, key: str) -> Metric:
        return self.metrics.get(key, Metric(note="inte beräknad"))


def _src(ld: Optional[fd.Loaded]) -> str:
    if ld is None or not ld.ok:
        return ""
    d = ld.data
    return f"{d.label or d.source} [{d.source} {d.series_id}]"


def _date(ts) -> Optional[str]:
    return str(pd.Timestamp(ts).date()) if ts is not None else None


def _metric(value, ld: Optional[fd.Loaded], as_of=None, unit="%", note="") -> Metric:
    if ld is None or not ld.ok:
        return Metric(None, None, "", unit, False, note or "DATA UNAVAILABLE")
    return Metric(None if value is None else round(float(value), 2), _date(as_of) if as_of is not None
                  else ld.data.last, _src(ld), unit, ld.stale, note)


def _latest_yoy(s) -> tuple:
    y = fe.yoy(s)
    return fe.latest(y) if y is not None else (None, None)


def snapshot(currency: str, loader: Optional[Callable] = None, asset_loader: Optional[Callable] = None,
             today=None) -> Snapshot:
    load = loader or (lambda c, cur: fd.load(c, cur, today=today))
    load_asset = asset_loader or (lambda n: fd.load_asset(n, today=today))
    snap = Snapshot(currency)
    L = {c: load(c, currency) for c in (cfg.M2, cfg.CPI, cfg.CORE, cfg.GDP, cfg.DEBT)}
    snap.loaded.update(L)
    m = snap.metrics

    m2 = L[cfg.M2].values
    d, v = _latest_yoy(m2)
    m["m2_yoy"] = _metric(v, L[cfg.M2], d, note="Money Supply Growth — inte konsumentinflation")
    m["m2_cagr5"] = _metric(fe.cagr(m2, 5), L[cfg.M2])
    m["m2_cagr10"] = _metric(fe.cagr(m2, 10), L[cfg.M2])

    cpi = L[cfg.CPI].values
    d, v = _latest_yoy(cpi)
    m["cpi_yoy"] = _metric(v, L[cfg.CPI], d)
    m["cpi_cagr5"] = _metric(fe.cagr(cpi, 5), L[cfg.CPI])
    m["cpi_cagr10"] = _metric(fe.cagr(cpi, 10), L[cfg.CPI])
    d, v = _latest_yoy(L[cfg.CORE].values)
    m["core_yoy"] = _metric(v, L[cfg.CORE], d)

    gdp = L[cfg.GDP].values
    gq = fe.quarterly(gdp)
    d, v = _latest_yoy(gq)
    m["gdp_yoy"] = _metric(v, L[cfg.GDP], d)
    m["gdp_cagr5"] = _metric(fe.cagr(gq, 5), L[cfg.GDP])
    m["gdp_cagr10"] = _metric(fe.cagr(gq, 10), L[cfg.GDP])

    gs = fe.gap_series(m2, gdp)
    gd, gv = fe.latest(gs) if gs is not None else (None, None)
    both = L[cfg.M2] if (L[cfg.M2].ok and L[cfg.GDP].ok) else None
    gap_note = "Money supply growth relative to real economic output — inte faktisk inflation"
    m["monetary_gap"] = _metric(gv, both, gd, unit="pe", note=gap_note)
    if both is not None:
        m["monetary_gap"].source = f"{_src(L[cfg.M2])} − {_src(L[cfg.GDP])}"
    mq = fe.quarterly(m2)
    common = min(mq.index[-1], gq.index[-1]) if mq is not None and gq is not None else None
    g5 = fe.gap(fe.cagr(mq, 5, at=common), fe.cagr(gq, 5, at=common)) if common is not None else None
    m["monetary_gap5"] = _metric(g5, both, common, unit="pe", note=gap_note + " (5 år, årstakt)")
    if both is not None:
        m["monetary_gap5"].source = m["monetary_gap"].source

    d, v = fe.latest(L[cfg.DEBT].values)
    m["debt_gdp"] = _metric(v, L[cfg.DEBT], d, unit="% av BNP")

    gold = load_asset(cfg.GOLD)
    snap.loaded[cfg.GOLD] = gold
    fx = load(cfg.FX, currency) if currency != "USD" else None
    if fx is not None:
        snap.loaded[cfg.FX] = fx
    price = fe.price_in(gold.values, currency, fx.values if fx is not None else None)
    ld = gold if (gold.ok and (fx is None or fx.ok)) else None
    vs = fe.fiat_vs_asset(price)
    for key, yrs in (("gold_1y", 1), ("gold_5y", 5), ("gold_10y", 10)):
        m[key] = _metric(fe.change_pct(vs, yrs), ld, note="valutans köpkraft i guld, förändring")
    if ld is not None:
        src_txt = _src(gold) + (f" × {_src(fx)}" if fx is not None else "")
        for key in ("gold_1y", "gold_5y", "gold_10y"):
            m[key].source = src_txt
            m[key].as_of = _date(price.index[-1]) if price is not None else None
    return snap
