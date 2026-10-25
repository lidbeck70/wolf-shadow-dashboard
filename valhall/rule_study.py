"""
valhall/rule_study.py — Rule-köp i Norden: hat + kvalitet + balansräkning + katalysator
(docs/forskning/rule_kop_spec.md, version 1 — parametrarna här är låsta av den sidan).

Rent och utan nätverk (rule_study.py hämtar data). Hat och hatperioder kommer ur valhall.hate_study; grindarna
är Contrarian Alphas (Deep Contrarians ROIC-grind, balansräkningens fyra krav) räknade på årssiffror med
fyra månaders fördröjning, och katalysatorn är dess tekniska vändning (över SMA50 och SMA50 stigande).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from valhall import hate_study as hs

SPEC_VERSION = 1
SPEC_PATH = "docs/forskning/rule_kop_spec.md"
TOP_N = 30
CAT_SMA, CAT_SLOPE, CAT_AFTER = 50, 10, 63
ROIC_MIN, ROIC_YEARS, ROIC_MIN_YEARS = 8.0, 4, 2
DE_MAX = 0.6
KPIS = {"roic": 37, "fcf": 63, "de": 40, "ebitda": 54, "equity": 58}
MIN_N, MIN_YEARS = 30, 5


def catalyst(close: pd.Series) -> pd.Series:
    """Sant när stängningen ligger över SMA50 och SMA50 är högre än för 10 dagar sedan."""
    c = close.astype(float)
    sma = c.rolling(CAT_SMA, min_periods=CAT_SMA).mean()
    return ((c > sma) & (sma > sma.shift(CAT_SLOPE))).fillna(False)


def catalyst_day(cat: pd.Series, ep: dict) -> Optional[pd.Timestamp]:
    """Första dagen katalysatorn slår om till sant, från hatets start till 63 dagar efter slutet."""
    idx = cat.index
    if ep["start"] not in idx or ep["end"] not in idx:
        return None
    a = idx.get_loc(ep["start"])
    b = min(len(idx) - 1, idx.get_loc(ep["end"]) + CAT_AFTER)
    for k in range(max(a, 1), b + 1):
        if cat.iloc[k] and not cat.iloc[k - 1]:
            return idx[k]
    return None


def _known(vals: list, date) -> list:
    """[(år, värde)] kända på datumet (årssiffran känd 4 månader efter årets slut), äldst först."""
    d = pd.Timestamp(date)
    out = []
    for v in vals or []:
        y, x = v.get("y"), v.get("v")
        if y is None or x is None:
            continue
        if pd.Timestamp(int(y) + 1, 1, 1) + pd.DateOffset(months=hs.LAG_MONTHS) <= d:
            out.append((int(y), float(x)))
    return sorted(out)


def gates(hist: dict, date) -> dict:
    """{"quality": bool|None, "balance": bool|None} på datumet. hist = {nyckel i KPIS: [{y, v}]}."""
    roic = [x for _y, x in _known(hist.get("roic"), date)][-ROIC_YEARS:]
    quality = (float(np.median(roic)) >= ROIC_MIN) if len(roic) >= ROIC_MIN_YEARS else None
    last = {}
    for k in ("fcf", "de", "ebitda", "equity"):
        kv = _known(hist.get(k), date)
        last[k] = kv[-1][1] if kv else None
    if any(v is None for v in last.values()):
        balance = None
    else:
        balance = last["fcf"] > 0 and last["de"] < DE_MAX and last["ebitda"] > 0 and last["equity"] > 0
    return {"quality": quality, "balance": balance}


def _rel(c: pd.Series, mk: pd.Series, day) -> Optional[float]:
    a = c.index.get_loc(day) + 1
    b = a + hs.HOLD
    if b >= len(c) or pd.isna(mk.iloc[a]) or pd.isna(mk.iloc[b]):
        return None
    return float(c.iloc[b] / c.iloc[a] - 1) - float(mk.iloc[b] / mk.iloc[a] - 1)


def event_rows(close: pd.Series, market: pd.Series, eps: list, hist: dict, start=hs.START) -> list:
    """{year, day, group, rel, rel_start} per hatperiod med katalysator, kända grindar och hel horisont.
    group = "rule" (kvalitet och balansräkning godkända) eller "utan" (någon underkänd)."""
    c = close.astype(float).dropna()
    mk = market.reindex(c.index).ffill()
    cat = catalyst(c)
    out = []
    for e in eps:
        if e["start"] < pd.Timestamp(start):
            continue
        d = catalyst_day(cat, e)
        if d is None:
            continue
        g = gates(hist, d)
        if g["quality"] is None or g["balance"] is None:
            continue
        rel = _rel(c, mk, d)
        if rel is None:
            continue
        out.append({"year": int(d.year), "day": str(d.date()), "start": str(e["start"].date()),
                    "group": "rule" if g["quality"] and g["balance"] else "utan",
                    "quality": g["quality"], "balance": g["balance"], "rel": rel, "rel_start": _rel(c, mk, e["start"])})
    return out


def summary(rows: list, rng: np.random.Generator, n_boot: int = hs.N_BOOT) -> dict:
    """K1 (Rule-köp mot marknaden), K2 (Rule − utan kvalitet) och K3 (katalysator mot hatets start)."""
    rule = [(r["year"], r["rel"]) for r in rows if r["group"] == "rule"]
    utan = [(r["year"], r["rel"]) for r in rows if r["group"] == "utan"]

    def grp(x):
        v = [b for _a, b in x]
        return {"n": len(v), "years": len({a for a, _b in x}), "mean": hs._pct(np.mean(v)) if v else None,
                "median": hs._pct(np.median(v)) if v else None, "hit": round(float(np.mean([b > 0 for b in v])) * 100, 1) if v else None}
    gr, gu = grp(rule), grp(utan)
    ci1 = hs.cluster_ci([b for _a, b in rule], [a for a, _b in rule], rng, n_boot) if rule else [None, None]
    k1 = gr["n"] >= MIN_N and gr["years"] >= MIN_YEARS and ci1[0] is not None and ci1[0] > 0 and (gr["median"] or 0) > 0
    ci2 = hs._diff_ci(rule, utan, rng, n_boot)
    k2 = (gr["n"] >= MIN_N and gu["n"] >= MIN_N and min(gr["years"], gu["years"]) >= MIN_YEARS and ci2[0] is not None
          and ci2[0] > 0 and gr["median"] is not None and gu["median"] is not None and gr["median"] > gu["median"])
    k3 = [r["rel"] - r["rel_start"] for r in rows if r["group"] == "rule" and r["rel_start"] is not None]
    return {"rule": gr, "utan": gu,
            "K1": {"ci": [hs._pct(v) for v in ci1], "pass": bool(k1)},
            "K2": {"diff": hs._pct((np.mean([b for _a, b in rule]) - np.mean([b for _a, b in utan]))) if rule and utan else None,
                   "ci": [hs._pct(v) for v in ci2], "pass": bool(k2)},
            "K3": {"n": len(k3), "mean": hs._pct(np.mean(k3)) if k3 else None, "median": hs._pct(np.median(k3)) if k3 else None},
            "fail_quality": sum(1 for r in rows if r["group"] == "utan" and not r["quality"]),
            "fail_balance": sum(1 for r in rows if r["group"] == "utan" and not r["balance"])}
