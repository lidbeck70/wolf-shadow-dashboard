"""
fiat_debasement/data.py — laddar serierna ur config.SERIES med reserver,
cache och färskhetskontroll.

  fetch(spec)            en källa → SeriesData
  load(begrepp, valuta)  första fungerande källan i preferensordning → Loaded
  load_asset(namn)       guld/silver: Börsdata spot + Yahoo-terminer före 2006
                         (märkt "terminspris"); övriga tillgångar som load()

Cache i processen: lyckad hämtning sparas 6 timmar, misslyckad 10 minuter
(ett fel ska inte fastna). Inaktuell data visas med varning — aldrig som 0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from fiat_debasement import config as cfg
from fiat_debasement import sources as src

TTL_OK_S = 6 * 3600
TTL_FAIL_S = 600
_CACHE: dict = {}


@dataclass
class Loaded:
    """Resultatet för ett (begrepp, valuta): vald serie + alla försök + metadata."""
    concept: str
    currency: str
    data: Optional[src.SeriesData] = None          # vald serie (None = DATA UNAVAILABLE)
    attempts: list = field(default_factory=list)   # alla försök i ordning
    definition: str = ""
    stale: bool = False
    segments: list = field(default_factory=list)   # skarvade serier: [{from, to, source, kind}]
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.data is not None and self.data.ok

    @property
    def values(self) -> Optional[pd.Series]:
        return self.data.values if self.ok else None


def is_stale(sd: Optional[src.SeriesData], today=None) -> bool:
    if sd is None or not sd.ok or not sd.frequency:
        return False
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    return (today - sd.values.index[-1]).days > cfg.STALE_DAYS.get(sd.frequency, 640)


def fetch(spec: dict, http=None, yahoo_getter=None, bd_api=None) -> src.SeriesData:
    """En källa ur config → SeriesData. Okänd kind ger ett fel, aldrig ett undantag."""
    meta = {k: spec[k] for k in ("unit", "label") if k in spec}
    kind = spec.get("kind")
    if kind == "fred":
        return src.fred(spec["id"], http=http, **meta)
    if kind == "ecb":
        return src.ecb(spec["flow"], spec["key"], http=http, **meta)
    if kind == "eurostat":
        return src.eurostat(spec["dataset"], spec["params"], http=http, **meta)
    if kind == "scb":
        return src.scb_table(spec["id"], prefer=spec.get("prefer"), prefer_text=spec.get("prefer_text", ()),
                             http=http, **meta)
    if kind == "riksbank":
        return src.riksbank(spec["id"], start=spec.get("start", "1990-01-01"), http=http, **meta)
    if kind == "yahoo":
        return src.yahoo(spec["id"], getter=yahoo_getter, **meta)
    if kind == "borsdata":
        return src.borsdata(spec["id"], api=bd_api, **meta)
    return src.SeriesData(str(kind), str(spec.get("id")), error=f"okänd källa: {kind}", **meta)


def _cache_key(spec: dict) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in spec.items() if k in ("kind", "id", "flow", "key", "dataset",
                                                                        "params", "prefer", "prefer_text")))


def _cached_fetch(spec: dict, fetcher: Callable) -> src.SeriesData:
    key = _cache_key(spec)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < (TTL_OK_S if hit[1].ok else TTL_FAIL_S):
        return hit[1]
    sd = fetcher(spec)
    _CACHE[key] = (time.time(), sd)
    return sd


def load(concept: str, currency: str, fetcher: Optional[Callable] = None, today=None) -> Loaded:
    """Första fungerande källan i preferensordning. Inga källor → DATA UNAVAILABLE."""
    specs = cfg.SERIES.get((concept, currency), [])
    out = Loaded(concept, currency, definition=next((s.get("definition", "") for s in specs if s.get("definition")), ""))
    if not specs:
        out.note = "ingen källa konfigurerad"
        return out
    for spec in specs:
        sd = _cached_fetch(spec, fetcher or fetch)
        sd.currency = sd.currency or currency
        out.attempts.append(sd)
        if sd.ok:
            out.data = sd
            break
    out.stale = is_stale(out.data, today)
    if out.data is not None and out.data.ok:
        out.segments = [{"from": out.data.first, "to": out.data.last, "source": out.data.label or out.data.source,
                         "kind": ""}]
    return out


def splice(primary: Optional[src.SeriesData], backfill: Optional[src.SeriesData]) -> tuple:
    """(serie, segment) — backfill bara FÖRE primärens första datum, ingen nivåjustering.
    Saknas primären används backfill för hela perioden, märkt."""
    p_ok = primary is not None and primary.ok
    b_ok = backfill is not None and backfill.ok
    if not p_ok and not b_ok:
        return None, []
    if not p_ok:
        b = backfill.values
        return b, [{"from": str(b.index[0].date()), "to": str(b.index[-1].date()),
                    "source": backfill.label or backfill.source, "kind": cfg.FUTURES}]
    p = primary.values
    segs = []
    parts = [p]
    if b_ok:
        early = backfill.values[backfill.values.index < p.index[0]]
        if len(early):
            parts.insert(0, early)
            segs.append({"from": str(early.index[0].date()), "to": str(early.index[-1].date()),
                         "source": backfill.label or backfill.source, "kind": cfg.FUTURES})
    segs.append({"from": str(p.index[0].date()), "to": str(p.index[-1].date()),
                 "source": primary.label or primary.source, "kind": cfg.SPOT})
    return pd.concat(parts).sort_index(), segs


def splice_gap_pct(primary: Optional[src.SeriesData], backfill: Optional[src.SeriesData]) -> Optional[float]:
    """Skillnaden (%) mellan terminspris och spot på primärens första gemensamma datum."""
    if primary is None or backfill is None or not primary.ok or not backfill.ok:
        return None
    common = primary.values.index.intersection(backfill.values.index)
    if not len(common):
        return None
    d = common[0]
    return round((float(backfill.values[d]) / float(primary.values[d]) - 1) * 100, 2)


def load_asset(name: str, fetcher: Optional[Callable] = None, today=None) -> Loaded:
    """Guld/silver skarvas (spot + terminspris före); övriga tillgångar via load()."""
    if name not in cfg.ASSET_SPLICE:
        return load(name, "USD", fetcher, today)
    conf = cfg.ASSET_SPLICE[name]
    f = fetcher or fetch
    prim, back = _cached_fetch(conf["primary"], f), _cached_fetch(conf["backfill"], f)
    out = Loaded(name, "USD", attempts=[prim, back])
    values, segs = splice(prim, back)
    if values is None:
        return out
    chosen = prim if prim.ok else back
    out.data = src.SeriesData(chosen.source, chosen.series_id, values=values, frequency=src.frequency_of(values.index),
                              unit=conf["primary"]["unit"], currency="USD", label=cfg.CONCEPT_LABEL[name],
                              last_updated=chosen.last_updated)
    out.segments = segs
    gap = splice_gap_pct(prim, back)
    if any(s["kind"] == cfg.FUTURES for s in segs):
        out.note = ("Före spotseriens start används terminspris (Yahoo) — märkt i grafen."
                    + (f" Skillnad termin/spot vid skarven: {gap:+.2f} %." if gap is not None else ""))
    out.stale = is_stale(out.data, today)
    return out


def clear_cache() -> None:
    _CACHE.clear()
