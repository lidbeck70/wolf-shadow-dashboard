"""
market_prices.py — EN cachad väg till kurshistorik: Börsdata först, Yahoo som reserv.

Panelen hade minst tio nästan identiska nedladdningswrappers utan gemensam
cache: temabordet laddade samma ticker tre gånger och SPY nio gånger,
kvotmodulen 23 nedladdningar för 13 symboler, Ember DXY tre gånger, och
Wolf Regime två okachade nedladdningar per klick. Här finns en funktion
per behov, cachad i Streamlit (6 h, begränsat antal poster) och med en
process-cache när Streamlit inte finns (Actions-jobben).

Sedan 22 oktober hämtas varje bolag med ett Börsdata-id (universum-registret: Norden och, med Pro+, USA,
Kanada, London, Australien) från Börsdata: licensierat, 90 anrop per 10 s, bara stängda dagsstaplar, 20 års
historik. Yahoo är kvar för det Börsdata saknar — ETF:er, terminer, dollarindex, tickers utan id — och som
reserv när Börsdata inte svarar. En schemakörning fick förut ~1 300 bolag strypta av Yahoo (Too Many
Requests); de Yahoo-anrop som finns kvar får en paus och ett nytt försök. STATS räknar källorna per process.

  ohlcv(ticker, period)  → DataFrame Open/High/Low/Close/Volume (tom vid fel)
  close(ticker, period)  → Series Close
  closes(tickers, period)→ dict ticker → Series (Börsdata per bolag, Yahoo i en batch för resten)

Ingen modul behöver ändra sin logik: samma period-strängar som yfinance. Börsdata justerar för splittar,
Yahoo (auto_adjust) även för utdelningar — serierna kan skilja marginellt runt utdelningsdagar.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, Optional

import pandas as pd

log = logging.getLogger(__name__)

TTL_SECONDS = 6 * 3600
MAX_ENTRIES = 300

try:
    import streamlit as st
    _cache = st.cache_data(ttl=TTL_SECONDS, max_entries=MAX_ENTRIES, show_spinner=False)
except Exception:                                   # pragma: no cover — headless
    st = None

    def _cache(fn):
        store: dict = {}

        def wrapped(*args):
            now = time.time()
            hit = store.get(args)
            if hit and now - hit[0] < TTL_SECONDS:
                return hit[1]
            val = fn(*args)
            if len(store) >= MAX_ENTRIES:
                store.pop(next(iter(store)))
            store[args] = (now, val)
            return val
        wrapped.clear = store.clear                   # samma API som st.cache_data
        return wrapped


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


class _NoData(Exception):
    """Tom eller misslyckad nedladdning. Kastas INNE i den cachade funktionen så
    att ett tillfälligt fel (t.ex. Yahoos 'Too Many Requests') aldrig cachas i
    sex timmar — nästa anrop försöker igen."""


# ── Börsdata först ───────────────────────────────────────────────────────────
PERIOD_BARS = {"1mo": 23, "3mo": 66, "6mo": 132, "1y": 262, "2y": 524, "5y": 1310, "10y": 2620,
               "ytd": 262, "max": 5040}                    # yfinance-period → antal Börsdata-staplar (tak 5040)
NORDIC_SUFFIXES = (".ST", ".OL", ".CO", ".HE")
STATS = {"borsdata": 0, "yahoo": 0, "missing": 0}       # källa per anrop i den här processen
_IDS: dict = {}                                          # {YF-ticker: Börsdata-id} ur registret, en gång per process
_IDS_READY = False


def _bd_api():
    """Börsdata-klienten när nyckeln finns, annars None (då är Yahoo enda vägen — som förut)."""
    try:
        from borsdata_api import get_api
        api = get_api()
        return api if getattr(api, "is_configured", False) else None
    except Exception:
        return None


_IDS_RETRY_S = 600
_IDS_FAILED_AT = 0.0


def _ids() -> dict:
    """Registrets Börsdata-id. En misslyckad läsning provas igen efter tio minuter — förut frös den till omstart
    (granskning 2, F11)."""
    global _IDS_READY, _IDS_FAILED_AT
    import time
    if not _IDS_READY and time.time() - _IDS_FAILED_AT >= _IDS_RETRY_S:
        try:
            from universe import registry as ur
            u = ur.load(None)
            if u is None:
                raise RuntimeError("registret tomt")
            _IDS.update({str(t).upper(): int(m.ins_id) for t, m in u.members.items() if m.ins_id})
            _IDS_READY = True
            log.info("market_prices: registret ger Börsdata-id för %d bolag.", len(_IDS))
        except Exception as exc:
            _IDS_FAILED_AT = time.time()
            log.warning("market_prices: registret kunde inte läsas (%s) — Börsdata-id bara via tickern, nytt "
                        "försök om %d min.", exc, _IDS_RETRY_S // 60)
    return _IDS


def ins_id(ticker) -> Optional[int]:
    """Börsdata-id för tickern (Yahoo-form): registret först, sedan Börsdatas egen tickerlista för nordiska
    tickers. None = ingen Börsdata-väg (ETF, termin, index, okänd) → Yahoo."""
    api = _bd_api()
    if api is None:
        return None
    t = str(ticker or "").strip().upper()
    if not t:
        return None
    hit = _ids().get(t)
    if hit:
        return hit
    if t.endswith(NORDIC_SUFFIXES):
        try:
            return api.resolve_instrument_id(t)
        except Exception:
            return None
    return None


@_cache
def _bd_ohlcv_raw(ins: int, period: str) -> pd.DataFrame:
    api = _bd_api()
    if api is None:
        raise _NoData("Börsdata saknas")
    bars = PERIOD_BARS.get(period, 5040)
    df = api.get_stockprices_df(int(ins), max_count=bars)
    if df is None or len(df) == 0:
        raise _NoData(f"Börsdata {ins} {period}: tom")
    return df[["Open", "High", "Low", "Close", "Volume"]].tail(bars)      # Börsdata kan svara med hela historiken


def _from_borsdata(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """Börsdata-ramen för tickern, eller None när tickern saknar id eller Börsdata inte levererar."""
    iid = ins_id(ticker)
    if not iid:
        return None
    try:
        df = _bd_ohlcv_raw(iid, period)
        STATS["borsdata"] += 1
        return df
    except Exception as exc:
        log.warning("market_prices: Börsdata %s (id %s) %s: %s — Yahoo som reserv", ticker, iid, period, exc)
        return None


# ── Yahoo: paus och nytt försök vid strypning ───────────────────────────────
YAHOO_RETRIES = 2
YAHOO_WAIT_S = 20


def _rate_limited(tickers) -> bool:
    """yfinance.download kastar inte vid Too Many Requests — felet står i yfinance.shared._ERRORS."""
    try:
        import yfinance.shared as sh
        errs = getattr(sh, "_ERRORS", None) or {}
        return any("Rate limit" in str(errs.get(t, "")) or "Too Many Requests" in str(errs.get(t, ""))
                   for t in tickers)
    except Exception:
        return False


def _yf_download(tickers, period: str, **kw):
    """yf.download som väntar och försöker igen när Yahoo stryper — i stället för att en hel batch blir tom."""
    import yfinance as yf
    names = [tickers] if isinstance(tickers, str) else list(tickers)
    raw = None
    for attempt in range(YAHOO_RETRIES + 1):
        try:
            raw = yf.download(tickers, period=period, auto_adjust=True, progress=False, **kw)
        except Exception as exc:
            if attempt < YAHOO_RETRIES and ("Rate limit" in str(exc) or "Too Many Requests" in str(exc)):
                log.warning("market_prices: Yahoo stryper (%s) — väntar %d s", names[0], YAHOO_WAIT_S * (attempt + 1))
                time.sleep(YAHOO_WAIT_S * (attempt + 1))
                continue
            raise
        if attempt < YAHOO_RETRIES and _rate_limited(names):
            log.warning("market_prices: Yahoo stryper (%s …) — väntar %d s och försöker igen",
                        names[0], YAHOO_WAIT_S * (attempt + 1))
            time.sleep(YAHOO_WAIT_S * (attempt + 1))
            continue
        return raw
    return raw


@_cache
def _ohlcv_raw(ticker: str, period: str) -> pd.DataFrame:
    df = _flatten(_yf_download(ticker, period, threads=False))
    df = df.dropna(how="all")
    if df.empty:
        raise _NoData(f"{ticker} {period}: tom")
    return df


def ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Kurshistorik för EN ticker: Börsdata när tickern har ett id, annars (och som reserv) Yahoo.
    Tom DataFrame (aldrig undantag) vid fel — och felet cachas inte."""
    bd = _from_borsdata(ticker, period)
    if bd is not None:
        return bd
    try:
        df = _ohlcv_raw(ticker, period)
        STATS["yahoo"] += 1
        return df
    except Exception as exc:
        STATS["missing"] += 1
        log.warning("market_prices: %s %s misslyckades: %s", ticker, period, exc)
        return pd.DataFrame()


def stats_text() -> str:
    """'kurskällor: Börsdata 1 970 · Yahoo 60 · saknas 3' — för loggen och Data Health."""
    return (f"kurskällor: Börsdata {STATS['borsdata']} · Yahoo {STATS['yahoo']} · saknas {STATS['missing']}")


def close(ticker: str, period: str = "1y") -> pd.Series:
    df = ohlcv(ticker, period)
    if df.empty or "Close" not in df:
        return pd.Series(dtype=float)
    return df["Close"].dropna()


@_cache
def _batch_raw(tickers: tuple, period: str) -> dict:
    raw = _yf_download(list(tickers), period, group_by="ticker", threads=True)
    out: dict = {}
    if raw is None or len(raw) == 0:
        raise _NoData(f"batch {period}: tom")
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                s = raw[t]["Close"].dropna() if "Close" in raw[t] else pd.Series(dtype=float)
                out[t] = s
    elif len(tickers) == 1 and "Close" in raw:
        out[tickers[0]] = raw["Close"].dropna()
    if not any(len(s) for s in out.values()):
        raise _NoData(f"batch {period}: inga kurser")
    return out


def _batch(tickers: tuple, period: str) -> dict:
    try:
        return _batch_raw(tickers, period)
    except Exception as exc:
        log.warning("market_prices: batch %s misslyckades: %s", period, exc)
        return {}


@_cache
def _batch_cv_raw(tickers: tuple, period: str) -> dict:
    raw = _yf_download(list(tickers), period, group_by="ticker", threads=True)
    if raw is None or len(raw) == 0:
        raise _NoData(f"batch {period}: tom")
    out: dict = {}
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t]
        elif len(tickers) == 1:
            df = raw
        else:
            continue
        cols = [c for c in ("Close", "Volume") if c in df]
        if "Close" in cols:
            out[t] = df[cols].dropna(subset=["Close"])
    if not any(len(d) for d in out.values()):
        raise _NoData(f"batch {period}: inga kurser")
    return out


def _split(tickers: Iterable[str], period: str, take) -> tuple:
    """(Börsdata-resultat, tickers kvar för Yahoo): varje bolag med id hämtas från Börsdata (cachat), resten
    går i Yahoo-batchen. take(df) plockar det anroparen vill ha ur ramen."""
    tup = tuple(dict.fromkeys(str(t).strip() for t in tickers if t))
    out, rest = {}, []
    for t in tup:
        df = _from_borsdata(t, period)
        if df is not None:
            out[t] = take(df)
        else:
            rest.append(t)
    return out, tuple(rest)


def closes_volumes(tickers: Iterable[str], period: str = "1y") -> dict:
    """{ticker: DataFrame[Close, Volume]} — Börsdata per bolag, Yahoo i en batch för resten (omsättningsfilter)."""
    out, rest = _split(tickers, period, lambda df: df[["Close", "Volume"]].dropna(subset=["Close"]))
    if rest:
        try:
            got = dict(_batch_cv_raw(rest, period))
        except Exception as exc:
            log.warning("market_prices: batch close/volume %s misslyckades: %s", period, exc)
            got = {}
        STATS["yahoo"] += len(got)
        STATS["missing"] += len(rest) - len(got)
        out.update(got)
    return out


def closes(tickers: Iterable[str], period: str = "1y") -> dict:
    """{ticker: Close-serie} — Börsdata per bolag, Yahoo i en batch för resten (saknade tickers utelämnas)."""
    out, rest = _split(tickers, period, lambda df: df["Close"].dropna())
    if rest:
        got = dict(_batch(rest, period))
        STATS["yahoo"] += len(got)
        STATS["missing"] += len(rest) - len(got)
        out.update(got)
    return out


def clear() -> None:
    """Töm BARA prisdatacachen (inte hela st.cache_data)."""
    for fn in (_ohlcv_raw, _batch_raw, _batch_cv_raw, _bd_ohlcv_raw):
        try:
            fn.clear()
        except Exception:
            pass
