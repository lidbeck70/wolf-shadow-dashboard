"""
fiat_debasement/index.py — Wolf Debasement Index och scenarierna.

WOLF DEBASEMENT INDEX (Wolfpanel Composite Indicator, 0–100)
  Fyra komponenter, alla orienterade så att högre = mer utspädning:
    gap      Money Supply Gap: M2-tillväxt − real BNP-tillväxt (kvartal, procentenheter)
    pp_loss  köpkraftsförlust över rullande 5 år: (1 − KPI_t−5 / KPI_t) × 100
    debt     statsskuld/BNP (nivå)
    gold     valutans värdetapp mot guld över rullande 5 år: (1 − pris_t−5 / pris_t) × 100
  Varje komponent görs om till månadsserie (senast kända värde) och normaliseras
  till sin percentil i valutans EGEN historik sedan INDEX_SINCE (2000, samma
  period för alla valutor) — bara data fram till dagen
  (expanderande fönster, minst INDEX_MIN_OBS månader). Index = viktat snitt av
  percentilerna. Saknas en komponent viktas de övriga om; saknas mer än hälften
  av vikten blir indexet DATA UNAVAILABLE.

  En modellbaserad indikator — inte ett officiellt mått. 70 för SEK betyder
  "högt för SEK historiskt", inte "högre än USD".

SCENARIER — räkneexempel med användarens antaganden, inga prognoser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from fiat_debasement import config as cfg
from fiat_debasement import engine as fe


# ── Komponenterna ───────────────────────────────────────────────────────────
def _to_month(s: Optional[pd.Series]) -> Optional[pd.Series]:
    """Månadsserie (månadens första dag) med senast kända värde — aldrig ett senare."""
    s = fe._clean(s)
    if s is None:
        return None
    m = s.resample("MS").last().dropna()
    idx = pd.date_range(m.index[0], m.index[-1], freq="MS")
    return m.reindex(idx).ffill()


def rolling_loss(level: Optional[pd.Series], years: int = cfg.INDEX_WINDOW_YEARS) -> Optional[pd.Series]:
    """(1 − x_t−n år / x_t) × 100 på månadsserie — hur mycket mindre en enhet köper än för n år sedan."""
    m = _to_month(level)
    if m is None:
        return None
    prev = pd.Series(m.values, index=m.index + pd.DateOffset(years=years)).reindex(m.index)
    ok = (prev > 0) & (m > 0)
    out = ((1 - prev / m) * 100)[ok].dropna()
    return out if len(out) else None


def components(m2=None, gdp=None, cpi=None, debt=None, gold_in_ccy=None) -> dict:
    """namn → månadsserie (högre = mer utspädning) eller None."""
    gap = fe.gap_series(m2, gdp)
    return {"gap": _to_month(gap), "pp_loss": rolling_loss(cpi), "debt": _to_month(debt),
            "gold": rolling_loss(gold_in_ccy)}


def expanding_percentile(s: Optional[pd.Series], min_obs: int = cfg.INDEX_MIN_OBS) -> Optional[pd.Series]:
    """Percentil (0–100) för varje värde mot seriens historik FRAM TILL den dagen."""
    s = fe._clean(s)
    if s is None or len(s) < min_obs:
        return None
    v = s.values
    out = np.full(len(v), np.nan)
    for i in range(min_obs - 1, len(v)):
        hist = v[:i + 1]
        out[i] = ((hist < v[i]).sum() + 0.5 * (hist == v[i]).sum()) / len(hist) * 100
    r = pd.Series(out, index=s.index).dropna()
    return r if len(r) else None


@dataclass
class IndexResult:
    currency: str
    value: Optional[float] = None                  # None = DATA UNAVAILABLE
    as_of: Optional[str] = None
    rows: list = field(default_factory=list)       # per komponent: rådata, percentil, vikt, bidrag
    history: Optional[pd.Series] = None
    weight_share: float = 0.0                      # andel av vikten som hade data
    note: str = ""


def normalize_weights(weights: dict) -> dict:
    w = {k: max(0.0, float(weights.get(k, 0) or 0)) for k in cfg.INDEX_COMPONENTS}
    total = sum(w.values())
    return {k: (v / total if total > 0 else 0.0) for k, v in w.items()}


def compute(currency: str, comps: dict, weights: Optional[dict] = None) -> IndexResult:
    w = normalize_weights(weights or cfg.DEFAULT_WEIGHTS)
    res = IndexResult(currency)
    since = pd.Timestamp(cfg.INDEX_SINCE)
    comps = {k: (None if v is None else v[v.index >= since]) for k, v in comps.items()}
    pct = {k: expanding_percentile(comps.get(k)) for k in cfg.INDEX_COMPONENTS}
    for k, label in cfg.INDEX_COMPONENTS.items():
        raw = fe._clean(comps.get(k))
        p = pct[k]
        res.rows.append({"key": k, "label": label, "weight": round(w[k] * 100, 1),
                         "raw": None if raw is None else round(float(raw.iloc[-1]), 2),
                         "raw_date": None if raw is None else str(raw.index[-1].date()),
                         "percentile": None if p is None else round(float(p.iloc[-1]), 1),
                         "since": None if raw is None else str(raw.index[0].date())})
    avail = {k: p for k, p in pct.items() if p is not None and w[k] > 0}
    res.weight_share = round(sum(w[k] for k in avail), 3)
    if res.weight_share < cfg.INDEX_MIN_WEIGHT_SHARE:
        res.note = "DATA UNAVAILABLE — för lite av vikten har data"
        return res
    df = pd.concat(avail, axis=1, sort=True).ffill()
    wv = pd.Series({k: w[k] for k in avail})
    have = df.notna()
    share = have.mul(wv, axis=1).sum(axis=1)
    hist = (df.fillna(0).mul(wv, axis=1).sum(axis=1) / share)[share >= cfg.INDEX_MIN_WEIGHT_SHARE]
    if not len(hist):
        res.note = "DATA UNAVAILABLE — komponenterna överlappar inte i tid"
        return res
    res.history = hist.round(1)
    res.value = round(float(hist.iloc[-1]), 1)
    res.as_of = str(hist.index[-1].date())
    for r in res.rows:
        if r["key"] in avail and r["percentile"] is not None:
            r["contribution"] = round(r["percentile"] * w[r["key"]] / res.weight_share, 1)
    missing = [cfg.INDEX_COMPONENTS[k] for k in cfg.INDEX_COMPONENTS if k not in avail and w[k] > 0]
    if missing:
        res.note = "Saknas (vikten fördelad på övriga): " + ", ".join(missing)
    return res


# ── Scenarier ───────────────────────────────────────────────────────────────
def scenario(m2: float, gdp: float, cpi: float, years: int = cfg.SCENARIO_YEARS) -> dict:
    """Räkneexempel: vad antagandena innebär efter `years` år. Inte en prognos."""
    years = max(1, int(years))
    path = pd.Series([100 / (1 + cpi / 100) ** t for t in range(years + 1)], index=range(years + 1))
    return {"monetary_gap": round(m2 - gdp, 2),
            "purchasing_power_end": round(float(path.iloc[-1]), 1),
            "money_index_end": round(100 * (1 + m2 / 100) ** years, 1),
            "real_output_index_end": round(100 * (1 + gdp / 100) ** years, 1),
            "money_per_output_end": round(100 * ((1 + m2 / 100) / (1 + gdp / 100)) ** years, 1),
            "path": path}
