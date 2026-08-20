"""
market_data.py — en ögonblicksbild av en ticker, i tal.

Copiloten var blind: den fick ticker, strategi och tre priser, och ombads
kommentera regler som ingen hade kontrollerat. Det här är siffrorna som gör
regelkontrollen mekanisk och ger modellen något verkligt att invända mot.

Indikatordefinitionerna är identiska med optim/indicators.py — EMA, Wilder-RSI
och Wilder-ATR — så att Copilotens tal och regimmotorns tal är samma tal. Se
kommentaren vid _ema() för varför de är kopierade och inte importerade.

Hämtningen cachas i 15 minuter. En swingsetup ändrar inte karaktär mellan två
klick, och varje rerun i Streamlit hade annars blivit en nedladdning.

Fel sväljs inte. snapshot() kastar MarketDataError med vad som gick fel, för
en tom ögonblicksbild som tyst blir None läses som "inget anmärkningsvärt".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

CACHE_TTL = 900          # sekunder
HISTORY = "1y"           # räcker för EMA200 med marginal
MIN_BARS = 200           # under detta går EMA200 inte att räkna


class MarketDataError(RuntimeError):
    """Kunde inte bygga en ögonblicksbild. Texten visas i panelen."""


@dataclass(frozen=True)
class Snapshot:
    """Allt regelkontrollen och prompten behöver, i ett objekt."""
    ticker: str
    as_of: str
    price: float
    atr14: Optional[float]
    atr_pct: Optional[float]           # ATR i procent av kursen
    ema50: Optional[float]
    ema200: Optional[float]
    dist_ema50_pct: Optional[float]     # negativt = under
    dist_ema200_pct: Optional[float]
    rsi14: Optional[float]
    vol_ratio: Optional[float]         # dagens volym mot 20-dagarssnittet
    high_52w: Optional[float]
    low_52w: Optional[float]
    from_high_pct: Optional[float]     # negativt = under toppen
    swing_low_20: Optional[float]
    swing_high_20: Optional[float]
    ret_1m_pct: Optional[float]
    ret_3m_pct: Optional[float]
    bars: int

    @property
    def above_ema50(self) -> Optional[bool]:
        return None if self.dist_ema50_pct is None else self.dist_ema50_pct >= 0

    @property
    def above_ema200(self) -> Optional[bool]:
        return None if self.dist_ema200_pct is None else self.dist_ema200_pct >= 0

    def as_dict(self) -> dict:
        return asdict(self)


def _f(value) -> Optional[float]:
    """float eller None. NaN är inte ett värde."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """a mot b i procent. None om något saknas eller b är noll."""
    if a is None or b is None or b == 0:
        return None
    return round((a / b - 1) * 100, 4)


# ── Indikatorerna ────────────────────────────────────────────────────────────
# Definitionerna är HÄMTADE UR optim/indicators.py, rad 73–116, och ska hållas
# identiska med dem: EMA via ewm(adjust=False), RSI och ATR med Wilders RMA.
# De ligger kopierade här av en teknisk anledning, inte en designmässig —
# optim/indicators.py gör "from data_loader import _CACHE_DIR" mot en modul i
# sin egen mapp, och går därför inte att importera från panelens rot utan att
# manipulera sys.path och dra in en joblib-cache som bieffekt.
#
# Ändras en definition i optim måste den ändras här också. Testerna räknar
# talen för hand så att en tyst avvikelse syns.

def _ema(series, length: int):
    return series.ewm(span=length, adjust=False).mean()


def _rsi(series, length: int = 14):
    import numpy as np
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high, low, close, length: int = 14):
    import pandas as pd
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def build(df, ticker: str = "") -> Snapshot:
    """Ögonblicksbilden ur en OHLCV-ram. Ren funktion — inget nätverk.

    Separerad från hämtningen så att testerna kan mata in en känd ram och
    kontrollera varje tal utan att röra yfinance.
    """
    if df is None or len(df) == 0:
        raise MarketDataError(f"Ingen kursdata för {ticker or 'tickern'}.")
    missing = [c for c in ("High", "Low", "Close") if c not in df.columns]
    if missing:
        raise MarketDataError(
            f"Kursdatan för {ticker or 'tickern'} saknar kolumnerna "
            f"{', '.join(missing)}.")

    # Före börsöppning skickar yfinance med DAGENS rad utan kurs (NaN). Läses
    # den bokstavligt blir "senaste kursen" inget tal och hela fliken felar
    # klockan 05:41 trots att gårdagens data är komplett. Rader utan Close är
    # inte handelsdagar — bort med dem.
    df = df[df["Close"].notna()]
    if len(df) == 0:
        raise MarketDataError(f"Kursdatan för {ticker or 'tickern'} saknar "
                              f"stängningskurser helt.")

    close, high, low = df["Close"], df["High"], df["Low"]
    price = _f(close.iloc[-1])
    if price is None or price <= 0:
        raise MarketDataError(f"Senaste kursen för {ticker or 'tickern'} är "
                              f"inte ett användbart tal.")

    n = len(df)
    ema50 = _f(_ema(close, 50).iloc[-1]) if n >= 50 else None
    ema200 = _f(_ema(close, 200).iloc[-1]) if n >= MIN_BARS else None
    atr14 = _f(_atr(high, low, close, 14).iloc[-1]) if n >= 15 else None
    rsi14 = _f(_rsi(close, 14).iloc[-1]) if n >= 15 else None

    vol_ratio = None
    if "Volume" in df.columns and n >= 20:
        vol = _f(df["Volume"].iloc[-1])
        vol_ma = _f(df["Volume"].rolling(20).mean().iloc[-1])
        if vol is not None and vol_ma:
            vol_ratio = round(vol / vol_ma, 2)

    window = min(n, 252)
    high_52w = _f(high.iloc[-window:].max())
    low_52w = _f(low.iloc[-window:].min())

    swing_low = _f(low.rolling(20).min().iloc[-1]) if n >= 20 else None
    swing_high = _f(high.rolling(20).max().iloc[-1]) if n >= 20 else None

    return Snapshot(
        ticker=(ticker or "").upper(),
        as_of=str(df.index[-1])[:10],
        price=round(price, 4),
        atr14=round(atr14, 4) if atr14 is not None else None,
        atr_pct=round(atr14 / price * 100, 2) if atr14 is not None else None,
        ema50=round(ema50, 4) if ema50 is not None else None,
        ema200=round(ema200, 4) if ema200 is not None else None,
        dist_ema50_pct=_pct(price, ema50),
        dist_ema200_pct=_pct(price, ema200),
        rsi14=round(rsi14, 1) if rsi14 is not None else None,
        vol_ratio=vol_ratio,
        high_52w=round(high_52w, 4) if high_52w is not None else None,
        low_52w=round(low_52w, 4) if low_52w is not None else None,
        from_high_pct=_pct(price, high_52w),
        swing_low_20=round(swing_low, 4) if swing_low is not None else None,
        swing_high_20=round(swing_high, 4) if swing_high is not None else None,
        ret_1m_pct=_pct(price, _f(close.iloc[-22])) if n >= 22 else None,
        ret_3m_pct=_pct(price, _f(close.iloc[-66])) if n >= 66 else None,
        bars=n)


def _download(ticker: str):
    """yfinance-hämtningen, isolerad så att den går att monkeypatcha."""
    import yfinance as yf
    return yf.download(ticker, period=HISTORY, progress=False,
                       multi_level_index=False)


# Suffixen som prövas när en ticker utan punkt inte hittas som den är.
# Ordningen är panelens hemmamarknader: Stockholm först, sedan Oslo,
# Köpenhamn och Helsingfors.
NORDIC_SUFFIXES = (".ST", ".OL", ".CO", ".HE")


def _usable(df) -> bool:
    return (df is not None and len(df) > 0
            and "Close" in df.columns and df["Close"].notna().any())


def _candidates(name: str) -> list:
    """Symbolformerna som prövas, i ordning.

    Skriver man "ABB" ska panelen själv hitta ABB.ST — och "ERIC B"
    (Börsdatas form) ska bli ERIC-B.ST. En ticker som redan har en punkt
    prövas bara som den är: den som skrivit ett suffix menade det.
    """
    out = [name]
    if "." not in name:
        base = name.replace(" ", "-")
        out += [base + sfx for sfx in NORDIC_SUFFIXES]
    return out


def fetch(ticker: str) -> tuple:
    """(OHLCV, symbolen som fungerade), cachad. Kastar MarketDataError.

    Ett nätverksfel kastas direkt — det gäller alla former lika. Bara ett
    TOMT svar går vidare till nästa suffixkandidat.
    """
    name = (ticker or "").strip().upper()
    if not name:
        raise MarketDataError("Ingen ticker angiven.")
    tried = _candidates(name)
    for symbol in tried:
        try:
            df = _cached_download(symbol)
        except MarketDataError:
            raise
        except Exception as exc:
            raise MarketDataError(f"Kunde inte hämta kursdata för {symbol}: "
                                  f"{exc}") from exc
        if _usable(df):
            return df, symbol
    raise MarketDataError(
        f"Ingen kursdata för {name} — prövade {', '.join(tried)}. "
        f"Kontrollera symbolen; andra marknader behöver sitt börssuffix "
        f"utskrivet (t.ex. .TO för Toronto).")


def _cached_download(name: str):
    """st.cache_data om Streamlit finns, annars rakt igenom (tester)."""
    try:
        import streamlit as st
    except ImportError:                                  # pragma: no cover
        return _download(name)
    cached = st.cache_data(ttl=CACHE_TTL, show_spinner=False)(_download)
    return cached(name)


def snapshot(ticker: str) -> Snapshot:
    """Hämta och räkna. Kastar MarketDataError med orsaken.

    Ögonblicksbilden bär symbolen som FUNGERADE — skrev du ABB och panelen
    hittade ABB.ST är det ABB.ST som visas, så du ser vad som faktiskt
    hämtades.
    """
    df, resolved = fetch(ticker)
    return build(df, resolved)


def try_snapshot(ticker: str) -> tuple[Optional[Snapshot], Optional[str]]:
    """(ögonblicksbild, felmeddelande) — för anropare som vill visa felet
    i stället för att avbryta hela sidan."""
    try:
        return snapshot(ticker), None
    except MarketDataError as exc:
        return None, str(exc)
