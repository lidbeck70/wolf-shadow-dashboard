"""
fiat_debasement/sources.py — datalagret: en funktion per källa som returnerar
en SeriesData (värden + metadata). Källorna går att byta utan att resten
av modulen ändras.

Varje serie bär source, series_id, frequency, unit, currency och
last_updated. Ett fel ger values=None och error satt — aldrig 0, aldrig en
gissning.

  fred(id)                 FRED, CSV utan nyckel
  ecb(flow, key)           ECB Data Portal (SDMX, CSV)
  eurostat(dataset, ...)   Eurostat (JSON-stat)
  scb_list(path)           SCB PxWeb: tabeller/mappar under en sökväg
  scb_table(path, ...)     SCB PxWeb: en tabell, ett värde per variabel
  riksbank(id)             Riksbanken SWEA (växelkurser)
  yahoo(ticker)            Yahoo Finance via market_prices (delad cache)
  borsdata(ins_id)         Börsdata (kräver BORSDATA_API_KEY)
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

TIMEOUT_S = 30
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
ECB_URL = "https://data-api.ecb.europa.eu/service/data/{flow}/{key}?format=csvdata"
EUROSTAT_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}"
SCB_URL = "https://api.scb.se/OV0104/v1/doris/sv/ssd/{path}"
RIKSBANK_URL = "https://api.riksbank.se/swea/v1/Observations/{id}/{start}"


@dataclass
class SeriesData:
    source: str
    series_id: str
    values: Optional[pd.Series] = None            # DatetimeIndex → float, sorterad, utan NaN
    frequency: str = ""                           # D / W / M / Q / A (härledd ur datumen)
    unit: str = ""
    currency: str = ""
    label: str = ""
    last_updated: str = ""                        # när vi hämtade (UTC)
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.values is not None and len(self.values) > 0

    @property
    def first(self) -> Optional[str]:
        return str(self.values.index[0].date()) if self.ok else None

    @property
    def last(self) -> Optional[str]:
        return str(self.values.index[-1].date()) if self.ok else None

    @property
    def last_value(self) -> Optional[float]:
        return float(self.values.iloc[-1]) if self.ok else None


# ── Hjälpare ────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _get(url: str, http=None, **kw):
    if http is None:
        import requests as http
    r = http.get(url, timeout=TIMEOUT_S, **kw)
    r.raise_for_status()
    return r


def _post(url: str, payload: dict, http=None):
    if http is None:
        import requests as http
    r = http.post(url, json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r


def parse_period(p) -> Optional[pd.Timestamp]:
    """'2024-03-15', '2024-03', '2024M03', '2024-Q1', '2024Q1', '2024K1', '2024' → periodens första dag."""
    s = str(p).strip()
    m = re.fullmatch(r"(\d{4})[-]?[QK](\d)", s)
    if m:
        return pd.Timestamp(int(m.group(1)), 3 * int(m.group(2)) - 2, 1)
    m = re.fullmatch(r"(\d{4})M(\d{2})", s)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    if re.fullmatch(r"\d{4}", s):
        return pd.Timestamp(int(s), 1, 1)
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return pd.Timestamp(s + "-01")
    try:
        return pd.Timestamp(s[:10])
    except (ValueError, TypeError):
        return None


def frequency_of(idx) -> str:
    """D / W / M / Q / A ur medianavståndet mellan datumen."""
    if idx is None or len(idx) < 3:
        return ""
    gap = float(pd.Series(pd.DatetimeIndex(idx)).diff().dt.days.median())
    for limit, f in ((4, "D"), (10, "W"), (45, "M"), (120, "Q")):
        if gap <= limit:
            return f
    return "A"


def _series(pairs) -> Optional[pd.Series]:
    """[(period, värde)] → sorterad float-serie utan saknade värden."""
    rows = []
    for p, v in pairs:
        ts = parse_period(p)
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        if ts is not None and pd.notna(val):
            rows.append((ts, val))
    if not rows:
        return None
    s = pd.Series(dict(rows)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.astype(float)


def _done(sd: SeriesData, values: Optional[pd.Series], why: str = "inga observationer") -> SeriesData:
    sd.last_updated = _now()
    if values is None or len(values) == 0:
        sd.error = sd.error or why
        sd.values = None
        return sd
    sd.values = values
    sd.frequency = frequency_of(values.index)
    return sd


def _fail(sd: SeriesData, exc: Exception) -> SeriesData:
    sd.last_updated = _now()
    sd.error = f"{type(exc).__name__}: {str(exc)[:160]}"
    logger.warning("fiat %s %s: %s", sd.source, sd.series_id, sd.error)
    return sd


# ── Källorna ────────────────────────────────────────────────────────────────
def fred(series_id: str, http=None, **meta) -> SeriesData:
    sd = SeriesData("FRED", series_id, **meta)
    try:
        df = pd.read_csv(io.StringIO(_get(FRED_URL.format(id=series_id), http).text))
        if df.shape[1] < 2:
            return _done(sd, None, "oväntat CSV-format")
        return _done(sd, _series(zip(df.iloc[:, 0], pd.to_numeric(df.iloc[:, 1], errors="coerce"))))
    except Exception as exc:
        return _fail(sd, exc)


def ecb(flow: str, key: str, http=None, **meta) -> SeriesData:
    sd = SeriesData("ECB", f"{flow}/{key}", **meta)
    try:
        df = pd.read_csv(io.StringIO(_get(ECB_URL.format(flow=flow, key=key), http).text))
        if not {"TIME_PERIOD", "OBS_VALUE"} <= set(df.columns):
            return _done(sd, None, "oväntat CSV-format (TIME_PERIOD/OBS_VALUE saknas)")
        if "UNIT" in df.columns and len(df):
            sd.meta["unit_code"] = str(df["UNIT"].iloc[-1])
        return _done(sd, _series(zip(df["TIME_PERIOD"], df["OBS_VALUE"])))
    except Exception as exc:
        return _fail(sd, exc)


def eurostat(dataset: str, params: dict, http=None, **meta) -> SeriesData:
    """JSON-stat med alla dimensioner utom tid låsta till ett värde via params."""
    sd = SeriesData("Eurostat", f"{dataset}?" + "&".join(f"{k}={v}" for k, v in params.items()), **meta)
    try:
        data = _get(EUROSTAT_URL.format(dataset=dataset), http, params=params).json()
        dims = dict(zip(data.get("id") or [], data.get("size") or []))
        empty = [d for d, n in dims.items() if n == 0]
        if empty:
            return _done(sd, None, f"inga värden för {', '.join(empty)} (koden finns inte i datasetet)")
        if any(n != 1 for d, n in dims.items() if d != "time"):
            return _done(sd, None, f"fler än ett värde i någon dimension: {dims}")
        time_idx = data["dimension"]["time"]["category"]["index"]
        vals = data.get("value") or {}
        return _done(sd, _series((p, vals.get(str(i))) for p, i in time_idx.items()))
    except Exception as exc:
        return _fail(sd, exc)


def scb_list(path: str, http=None) -> list:
    """[(id, typ 'l'/'t', text)] under en SCB-sökväg — [] vid fel."""
    try:
        return [(x.get("id"), x.get("type"), x.get("text")) for x in _get(SCB_URL.format(path=path), http).json()]
    except Exception as exc:
        logger.warning("SCB %s: %s", path, exc)
        return []


def _pick(values: list, texts: dict, code: str, prefer: dict, prefer_text: tuple):
    """Värdet för en SCB-variabel: prefer[kod] → första värdet vars text innehåller
    ett nyckelord ur prefer_text (i ordning) → första värdet."""
    if prefer.get(code) in values:
        return prefer[code]
    for word in prefer_text:
        for v in values:
            if word.lower() in str(texts.get(v, v)).lower():
                return v
    return values[0] if values else None


def scb_table(path: str, prefer: Optional[dict] = None, prefer_text: tuple = (), http=None,
              **meta) -> SeriesData:
    """En SCB-tabell som tidsserie. Varje variabel utom tiden låses till ett värde
    (se _pick). Valen sparas i meta["chosen"] så det syns exakt vad som hämtades."""
    sd = SeriesData("SCB", path, **meta)
    prefer = prefer or {}
    try:
        info = _get(SCB_URL.format(path=path), http).json()
        query, chosen, time_code = [], {}, None
        for var in info.get("variables", []):
            code = var.get("code")
            if var.get("time") or code in ("Tid", "TID"):
                time_code = code
                continue
            values = var.get("values") or []
            texts = dict(zip(values, var.get("valueTexts") or values))
            pick = _pick(values, texts, code, prefer, prefer_text)
            if pick is None:
                continue
            chosen[code] = f"{pick} ({texts.get(pick, pick)})"
            query.append({"code": code, "selection": {"filter": "item", "values": [pick]}})
        sd.meta.update({"title": info.get("title", ""), "chosen": chosen})
        if time_code is None:
            return _done(sd, None, "ingen tidsvariabel i tabellen")
        data = _post(SCB_URL.format(path=path), {"query": query, "response": {"format": "json"}}, http).json()
        cols = [c.get("code") for c in data.get("columns", [])]
        ti = cols.index(time_code) if time_code in cols else len(cols) - 2
        return _done(sd, _series((row["key"][ti], (row.get("values") or [None])[0]) for row in data.get("data", [])))
    except Exception as exc:
        return _fail(sd, exc)


def riksbank(series_id: str, start: str = "1990-01-01", http=None, **meta) -> SeriesData:
    sd = SeriesData("Riksbanken", series_id, **meta)
    try:
        rows = _get(RIKSBANK_URL.format(id=series_id, start=start), http).json()
        return _done(sd, _series((r.get("date"), r.get("value")) for r in rows or []))
    except Exception as exc:
        return _fail(sd, exc)


def yahoo(ticker: str, getter: Optional[Callable] = None, **meta) -> SeriesData:
    sd = SeriesData("Yahoo", ticker, **meta)
    try:
        if getter is None:
            from market_prices import ohlcv as getter
        df = getter(ticker, "max")
        if df is None or len(df) == 0:
            return _done(sd, None, "ingen kurshistorik")
        s = df["Close"].astype(float).dropna()
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        return _done(sd, s)
    except Exception as exc:
        return _fail(sd, exc)


def borsdata(ins_id: int, api=None, **meta) -> SeriesData:
    sd = SeriesData("Börsdata", str(ins_id), **meta)
    try:
        if api is None:
            from borsdata_api import BorsdataAPI
            api = BorsdataAPI()
            if not api.is_configured:
                return _done(sd, None, "BORSDATA_API_KEY saknas")
        df = api.get_stockprices_df(int(ins_id), max_count=10000)
        if df is None or df.empty:
            return _done(sd, None, "ingen kurshistorik")
        return _done(sd, df["Close"].astype(float).dropna().sort_index())
    except Exception as exc:
        return _fail(sd, exc)
