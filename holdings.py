"""
holdings.py — Portfolio Holdings Manager

Multi-strategy portfolio with regime integration.
Strategies: Quality, Deep Contrarian, Viking, Wolf, Untagged.

Raderna bor i positions.py — panelens enda positionsregister
(data/holdings.json via storage.py; den gamla Gisten läses in en gång).
Den här modulen ritar dem.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import functools
import logging
import re
from html import escape
from typing import Dict, List, Optional

import positions
import storage
from ui import tokens as tk
from ui import vocab
from ui.components import MAIN_COLOR, action_row, chip, confirm_delete, missing, section, source_chip

logger = logging.getLogger(__name__)

# ── Theme (Nordic Arc) ────────────────────────────────────────────────────────
BG      = "#0c0c12"
BG2     = "#14141e"
CYAN    = "#c9a84c"   # legacy gold (kept for backward compat)
MAGENTA = "#8b7340"
GREEN   = "#2d8a4e"
RED     = "#c44545"
YELLOW  = "#d4943a"
TEXT    = "#e8e4dc"
DIM     = "#8a8578"

# ── Strategy constants ────────────────────────────────────────────────────────
STRATEGY_TAGS = list(positions.STRATEGIES)

STRATEGY_COLORS: dict = {
    "Quality":         "#00E5FF",   # Arc Cyan  (PALETTE["gold"])
    "Deep Contrarian": "#FF6B3D",   # Ember     (PALETTE["amber"])
    "Viking":          "#B400FF",   # Aurora Purple (PALETTE["silver"])
    "Wolf":            "#2d8a4e",   # Green
    "Untagged":        "#6B7280",   # Text dim
    "Momentum":        "#d4943a",   # Amber — Swing-flikens positioner
    "Tiggre":          "#c9a84c",   # Guld — Lobo-arkets positioner
}

# Hink per strategi och taken bor i positions.py (registret) — här bara färg.
_STRATEGY_TO_KEY: dict = dict(positions.STRATEGY_BUCKET)

PORTFOLIOS: dict = {
    key: {**cfg,
          "color": {"swing": CYAN, "ovtlyr": MAGENTA, "long": GREEN,
                    "momentum": YELLOW, "tiggre": CYAN}[key],
          "strategy": {"swing": "Wolf", "ovtlyr": "Viking", "long": "Alpha",
                       "momentum": "Momentum", "tiggre": "Tiggre"}[key]}
    for key, cfg in positions.BUCKETS.items()
}

# ── Signal imports (with fallbacks) ──────────────────────────────────────────
try:
    from ovtlyr.indicators.trend import compute_trend
    from ovtlyr.indicators.momentum import compute_momentum
    from ovtlyr.indicators.volatility import compute_volatility
    from ovtlyr.indicators.orderblocks import detect_orderblocks, classify_price_vs_ob
    from ovtlyr.signals.longterm_signals import compute_longterm_signal
    from ovtlyr.signals.swing_signals import compute_swing_signal
    _OVTLYR_OK = True
except ImportError:
    _OVTLYR_OK = False

try:
    from cagr.cagr_technical import score_technical
    from cagr.cagr_cycle import SECTOR_CONFIG, score_cycle_from_int
    from cagr.cagr_scoring import calculate_total_score
    _CAGR_OK = True
except ImportError:
    _CAGR_OK = False


# ── Lagring: positions.py är registret ───────────────────────────────────────
# Holdings ägde förut sina rader i en Gist som inget annat ark läste. Nu bor
# de i data/holdings.json via positions.py (samma storage-väg som de andra
# arken, Gisten som engångskälla), och de här hjälparna är tunna skal så att
# resten av fliken kan vara oförändrad.
STORE = positions.STORE


def _load_all() -> dict:
    """Registret: {swing, ovtlyr, long, closed, cash}."""
    return positions.load()


def _get_holdings(portfolio_key: str) -> List[dict]:
    return positions.by_bucket().get(portfolio_key, [])


def _strategy_of(h: dict) -> str:
    """Return the strategy tag for a holding; defaults to 'Untagged'."""
    return h.get("strategy", "Untagged")


_num = positions._num


def _sek(ticker: str, amount: float) -> float:
    """Belopp i kursens valuta → SEK (positions.sek_factor, London i pence). Summor och
    procent märkta SEK räknas så; kursen på raden visas fortfarande i sin egen valuta."""
    return amount * positions.sek_factor(ticker)


def _get_all_holdings_flat() -> List[dict]:
    """Alla öppna positioner, platta, med '_portfolio_key' och Deep
    Contrarian-trancherna lyfta ur extras — formen korten ritar."""
    result: List[dict] = []
    for r in positions.all_positions():
        result.append({
            **r,
            "_portfolio_key":    r["_bucket"],
            "tranches_deployed": int(_num((r.get("extras") or {}).get("tranches_deployed", 0))),
        })
    return result


def _registry_name(ticker: str, name: str = "") -> str:
    """Bolagsnamnet ur universum-registret när du inte skrivit något (formuläret har inget namnfält). Bara på
    + ADD; tomt utan register (ingen Gist) eller okänd ticker — då sparas raden som förut."""
    if str(name or "").strip():
        return str(name).strip()
    try:
        import refresh_ui
        return refresh_ui.registry_identity(ticker, "").get("name", "")
    except Exception:
        return ""


def _add_holding(
    portfolio_key: str,
    ticker: str,
    entry_price: float = 0,
    sector: str = "Unknown",
    shares: int = 0,
    strategy: str = "Untagged",
    tranches_deployed: int = 0,
) -> bool:
    extras = {"tranches_deployed": max(0, min(3, tranches_deployed))} \
        if strategy == "Deep Contrarian" else {}
    ok, msg = positions.add(ticker, strategy, entry_price, shares, sector=sector,
                            name=_registry_name(ticker), extras=extras, bucket=portfolio_key)
    if not ok or positions.UNSAVED in msg:
        st.warning(msg)
    return ok


def _remove_holding(portfolio_key: str, ticker: str) -> None:
    positions.remove(ticker, portfolio_key)


def _edit_holding_field(portfolio_key: str, ticker: str, field: str, value) -> None:
    """Update a single field on a holding in place and persist."""
    positions.update(ticker, portfolio_key, **{field: value})


# ── Live data & signals ───────────────────────────────────────────────────────

class _NoPrice(Exception):
    """Yahoo gav ingen kurs — kastas så att st.cache_data inte sparar missen."""


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_live_price_cached(ticker: str, period: str = "6mo") -> dict:
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, auto_adjust=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if df.empty or "Close" not in df or df["Close"].dropna().empty:
        raise _NoPrice(ticker)
    close = df["Close"].dropna()
    price = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else price
    change = (price / prev - 1) * 100
    return {"price": price, "change_1d": change, "df": df}


_PRICE_MISSES: dict = {}          # (ticker, period) → tid för senaste missen
_MISS_SECONDS = 60


def _fetch_live_price(ticker: str, period: str = "6mo") -> dict:
    """Senaste kursen ur Yahoo, sparad i 15 min. En miss (Yahoo strular, rate limit) sparas bara en minut — förut
    stod kursen 0.00 en kvart efter en enda tillfällig miss. Ett nytt försök direkt. En ticker med mellanslag är
    ingen Yahoo-ticker och frågas aldrig."""
    import time
    none = {"price": 0, "change_1d": 0, "df": None}
    t = str(ticker or "").strip()
    if not t or any(ch.isspace() for ch in t):
        return none
    key = (t, period)
    if time.time() - _PRICE_MISSES.get(key, 0.0) < _MISS_SECONDS:
        return none
    for _try in range(2):
        try:
            got = _fetch_live_price_cached(t, period)
            _PRICE_MISSES.pop(key, None)
            return got
        except Exception:
            continue
    _PRICE_MISSES[key] = time.time()
    return none


def _compute_swing_signal(df: pd.DataFrame) -> dict:
    if not _OVTLYR_OK or df is None or df.empty or len(df) < 50:
        return {"signal": "N/A", "color": DIM}
    try:
        trend = compute_trend(df)

        def _last(val, default=0.0):
            if val is None:
                return default
            if isinstance(val, pd.Series):
                return float(val.iloc[-1]) if len(val) > 0 else default
            return float(val)

        regime = trend.get("regime_color", "orange")
        price  = _last(trend.get("price", df["Close"].iloc[-1]))
        ema50  = _last(trend.get("ema50"))

        if regime == "green" and price > ema50:
            return {"signal": "HOLD",  "color": GREEN,  "detail": "Trend intact"}
        elif regime == "green":
            return {"signal": "HOLD",  "color": YELLOW, "detail": "Below EMA50"}
        elif regime == "orange":
            return {"signal": "WATCH", "color": YELLOW, "detail": "Orange regime"}
        else:
            return {"signal": "EXIT",  "color": RED,    "detail": "Red regime"}
    except Exception:
        return {"signal": "N/A", "color": DIM}


def _compute_ovtlyr_signal(df: pd.DataFrame) -> dict:
    if not _OVTLYR_OK or df is None or df.empty or len(df) < 50:
        return {"signal": "N/A", "color": DIM}
    try:
        trend      = compute_trend(df)
        momentum   = compute_momentum(df)
        volatility = compute_volatility(df)
        obs        = detect_orderblocks(df)
        cur_price  = float(df["Close"].iloc[-1])
        ob_analysis = classify_price_vs_ob(cur_price, obs) if obs else {"signal_bias": "HOLD"}

        def _last(val, default=0.0):
            if val is None:
                return default
            if isinstance(val, pd.Series):
                return float(val.iloc[-1]) if len(val) > 0 else default
            return float(val)

        trend_scalar = {
            "price":       _last(trend.get("price", cur_price)),
            "ema10":       _last(trend.get("ema10", df["Close"].ewm(span=10).mean().iloc[-1])),
            "ema20":       _last(trend.get("ema20", df["Close"].ewm(span=20).mean().iloc[-1])),
            "ema50":       _last(trend.get("ema50")),
            "ema200":      _last(trend.get("ema200")),
            "regime_color": trend.get("regime_color", "orange"),
            "trend_state":  trend.get("trend_state", "neutral"),
        }
        sentiment = {"score": 50, "label": "Neutral"}
        lt = compute_longterm_signal(trend_scalar, sentiment, volatility, ob_analysis, True)
        signal = lt.get("signal", "HOLD")

        if signal == "BUY":
            return {"signal": "HOLD ✓", "color": GREEN,  "detail": f"NINE: {lt.get('ovtlyr_nine', 0)}"}
        elif signal == "SELL":
            return {"signal": "EXIT",   "color": RED,    "detail": f"NINE: {lt.get('ovtlyr_nine', 0)}"}
        elif signal == "REDUCE":
            return {"signal": "REDUCE", "color": MAGENTA,"detail": f"NINE: {lt.get('ovtlyr_nine', 0)}"}
        else:
            return {"signal": "HOLD",   "color": YELLOW, "detail": f"NINE: {lt.get('ovtlyr_nine', 0)}"}
    except Exception as e:
        return {"signal": "N/A", "color": DIM, "detail": str(e)[:40]}


def _compute_long_signal(df: pd.DataFrame, sector: str = "Unknown") -> dict:
    if df is None or df.empty or len(df) < 200:
        return {"signal": "N/A", "color": DIM}
    try:
        close      = df["Close"].astype(float)
        price      = float(close.iloc[-1])
        ema50      = float(close.ewm(span=50).mean().iloc[-1])
        ema200     = float(close.ewm(span=200).mean().iloc[-1])
        ema200_ser = close.ewm(span=200).mean()
        days_below = 0
        for i in range(len(close) - 1, max(len(close) - 10, 0), -1):
            if float(close.iloc[i]) < float(ema200_ser.iloc[i]):
                days_below += 1
            else:
                break
        if price > ema200 and ema50 > ema200:
            return {"signal": "HOLD ✓", "color": GREEN,  "detail": f"Price {(price/ema200-1)*100:+.1f}% vs EMA200"}
        elif price > ema200:
            return {"signal": "WATCH",  "color": YELLOW, "detail": "EMA50 below EMA200"}
        elif days_below >= 3:
            return {"signal": "REDUCE", "color": MAGENTA,"detail": f"{days_below}d below EMA200"}
        elif price < ema200:
            return {"signal": "EXIT",   "color": RED,    "detail": "Below EMA200"}
        else:
            return {"signal": "HOLD",   "color": YELLOW, "detail": "Neutral"}
    except Exception:
        return {"signal": "N/A", "color": DIM}


def _signal_for_holding(holding: dict, df) -> dict:
    """Route to the correct signal function based on portfolio_key."""
    pkey = holding.get("_portfolio_key", "long")
    if pkey == "swing":
        return _compute_swing_signal(df)
    elif pkey == "ovtlyr":
        return _compute_ovtlyr_signal(df)
    else:
        return _compute_long_signal(df, holding.get("sector", "Unknown"))


# ── Alpha Regime integration (cached 1h) ─────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _get_regime_signal(ticker: str, strategy_tag: str) -> dict:
    """
    Fetch Alpha Regime signal for a holding. Cached 1h.
    Returns type-tagged dict: quality | contrarian | n/a | unavailable | error.
    """
    if strategy_tag not in ("Quality", "Deep Contrarian"):
        return {"type": "n/a"}
    try:
        from alpha_regime.engine import run_regime_analysis
        mode = "quality" if strategy_tag == "Quality" else "contrarian"
        r = run_regime_analysis(ticker=ticker, mode=mode)
        if r.error:
            return {"type": "error", "error": r.error[:40]}
        if mode == "quality":
            available = [s for s in r.signals if s.label != "NO DATA"]
            return {
                "type":       "quality",
                "verdict":    r.quality_verdict,
                "passed":     r.signals_passed,
                "total":      len(available),
                "phase":      r.market_phase,
                "confidence": r.market_confidence,
            }
        else:
            c = r.contrarian
            return {
                "type":  "contrarian",
                "stage": c.stage if c else "HOLD",
                "label": c.label if c else "HOLD",
                "color": c.color if c else "#607080",
                "phase": r.market_phase,
            }
    except ImportError:
        return {"type": "unavailable"}
    except Exception as exc:
        logger.debug("_get_regime_signal(%s, %s): %s", ticker, strategy_tag, exc)
        return {"type": "error", "error": str(exc)[:40]}


# ── Förslag i redigeringsrutan: stopp och sektor (batch 16) ──────────────────
# Förslagen räknas i portfolio/position_hints.py ur strategins egna regler och ritas med refresh_ui.offer. Använd
# skriver in värdet och sparar positionen; att öppna sidan skriver ingenting och hämtar inget nytt — stoppen räknas
# på kurserna sidan redan hämtat för raden (samma som signalkolumnen), sektorn hämtas först på ditt tryck.

def _price_or_none(value) -> Optional[float]:
    """Stopp/mål i rutan: tomt (eller ≤ 0) är tomt, inte 0."""
    v = positions._opt(value)
    return v if v is not None and v > 0 else None


def _save_fields(pkey: str, row: dict, field: str, extra: dict) -> None:
    """on_apply för Använd: fältet (redan skrivet i raden av refresh_ui._apply) och ev. följefält till registret."""
    positions.update(row["ticker"], pkey, row_id=row.get("id"), **{field: row.get(field)}, **(extra or {}))


def _esc(text: str) -> str:
    return str(text or "").replace("$", "\\$")          # två '$' i en caption blir annars matematik


def _stop_line(holding: dict, df, pkey: str, widget_key: str) -> None:
    """Stoppförslaget under Stopp-rutan, per strategi ur regler som redan finns (position_hints.stop_hint). Nivåer
    som rör sig (ATR, EMA200) dateras, och Använd sparar det fasta talet med regeln och dagen. Arkstrategierna får
    texten 'ingen kursstopp'. Inget visas när kurserna saknas."""
    try:
        import refresh_ui
        from portfolio import position_hints as ph
        signals = None
        if holding.get("strategy") == "Insider":            # insiderarket läses bara för Insider-rader
            import insider
            signals = (insider._load() or {}).get("signals")
        hint = ph.stop_hint(holding, df, signals)
        basis = str((holding.get("extras") or {}).get("stop_basis") or "")
        if basis and _price_or_none(holding.get("stop")) is not None:
            st.caption(_esc(f"Stoppen sattes ur {basis}."))
        if not hint:
            return
        if hint.get("value") is None:
            st.caption(_esc(hint["text"]))
            return
        save = functools.partial(_save_fields, pkey, holding, "stop", {"stop_basis": hint["basis"]})
        refresh_ui.offer(holding, "stop", hint["value"], hint["label"], key=f"hold_stop_{pkey}_{holding['ticker']}",
                         source=hint["source"], widget_key=widget_key, unit=hint["unit"], note=hint["note"],
                         fmt=ph.fmt_num, on_apply=save)
    except Exception as exc:                                  # ett förslag får aldrig fälla kortet
        logger.debug("stoppförslaget för %s: %s", holding.get("ticker"), exc)


def bd_sector(ticker: str, own_name: str = "") -> dict:
    """Börsdatas sektor för tickern: sektor-id ur universum-registret, annars ur Börsdatas instrument
    (thor.ticker.lookup), och sektornamnen ur Börsdata. {'value', 'note'} vid träff, annars {'msg'}. Körs bara på
    ditt tryck."""
    try:
        from borsdata_api import get_api
        api = get_api()
        if not getattr(api, "is_configured", False):
            return {"msg": "Börsdata-nyckeln saknas — skriv sektorn själv."}
        import refresh_ui
        member = refresh_ui._registry_member(ticker)
        sid, bd_name = getattr(member, "sector_id", None), str(getattr(member, "name", "") or "")
        if sid is None:
            from thor import ticker as thor_ticker
            info = thor_ticker.lookup(api, ticker) or {}
            sid, bd_name = info.get("sectorId"), str(info.get("name") or bd_name)
        sectors = {int(s["id"]): str(s.get("name") or "") for s in (api.get_sectors() or [])
                   if isinstance(s, dict) and s.get("id") is not None}
    except Exception as exc:                # bara typen: felets text bär anropets URL med nyckeln (authKey=…)
        logger.debug("Börsdatas sektor för %s: %s", ticker, type(exc).__name__)
        return {"msg": "Börsdata svarade inte — skriv sektorn själv."}
    from portfolio import position_hints as ph
    hint = ph.sector_hint(sid, sectors, bd_name, own_name)
    return hint or {"msg": f"Börsdata har ingen sektor för {ticker}."}


def _fetch_sector(holding: dict, state_key: str) -> None:
    """on_click: hämtar sektorn och lägger förslaget i sessionen — ingenting skrivs i raden."""
    st.session_state[state_key] = bd_sector(holding.get("ticker", ""), holding.get("name", ""))


def _sector_line(holding: dict, pkey: str, widget_key: str) -> None:
    """Under Sektor: hämta Börsdatas sektor på tryck, sedan förslaget med Använd (sparar positionen)."""
    try:
        import refresh_ui
        state_key = f"hold_sec_{holding.get('id') or holding['ticker']}"
        c1, c2 = st.columns([1, 3], vertical_alignment="center")
        c1.button("Börsdatas sektor", key=f"{state_key}_get", on_click=_fetch_sector, args=(holding, state_key),
                  help="Hämtar sektorn ur Börsdata (universum-registret, annars Börsdatas instrumentlista). "
                       "Inget sparas förrän du trycker Använd.")
        res = st.session_state.get(state_key) or {}
        if res.get("msg"):
            c2.caption(_esc(res["msg"]))
        if res.get("value"):
            drawn = refresh_ui.offer(holding, "sector", res["value"], "sektor", key=state_key, widget_key=widget_key,
                                     note=res.get("note", ""),
                                     on_apply=functools.partial(_save_fields, pkey, holding, "sector", {}))
            if not drawn:
                c2.caption(_esc(f"Sektorn är Börsdatas ({res['value']})."))
    except Exception as exc:
        logger.debug("sektorförslaget för %s: %s", holding.get("ticker"), exc)


# ── UI Components ─────────────────────────────────────────────────────────────
# LIDSKJALV §4.9 (beslut 10): ett kort per innehav. Kortet är EN st.markdown (at.markdown hittar kursen '12000.00'),
# signalen som chippar — ordlistans svenska huvudord (ui/vocab, sammanhanget HOLDING) och det lagrade ordet grått
# bredvid. Inga nya tal: kursen, värdet, P/L, signalen och regimen är exakt det raden räknade förut.

def _h(text) -> str:
    """Text i kortets HTML: < > & escapas — en ticker eller sektor kan aldrig bli HTML."""
    return escape("" if text is None else str(text), quote=False)


def card_id(portfolio_key: str, ticker: str) -> str:
    """Kortets id, målet för remsans data-jump: 'pos-<ticker>-<hink>' (samma ticker kan stå i två hinkar)."""
    return "pos-" + (re.sub(r"[^a-z0-9]+", "-", f"{ticker}-{portfolio_key}".lower()).strip("-") or "x")


def _sig_word(sig) -> str:
    """Det lagrade ordet utan bocken: 'HOLD ✓' läses som HOLD (ordlistan känner inte '✓')."""
    return str(sig if sig is not None else "").replace("✓", "").strip()


def signal_main(sig) -> Optional[str]:
    """Ordlistans huvudord för signalen (EXIT → SÄLJ, REDUCE → MINSKA, WATCH → VAKSAM, HOLD → HÅLL), None när
    signalen saknas eller inte är känd (N/A)."""
    m = vocab.main(_sig_word(sig), vocab.HOLDING)
    return None if m == vocab.SAKNAS else m


def signal_chips(sig, why: str = "") -> str:
    """[MINSKA] [REDUCE], [SÄLJ] [EXIT], [● VAKSAM] [WATCH], [HÅLL ✓] [HOLD ✓]: huvudordet i sin färg och det
    lagrade ordet ordagrant som grått chip. Saknas (N/A) blir det streckade Saknas-chipet med orsaken i tooltipen."""
    raw = str(sig if sig is not None else "").strip()
    main = signal_main(raw)
    if main is None:
        return chip(_sig_word(raw) or None, vocab.HOLDING, why=why or raw)
    head = chip(_sig_word(raw), vocab.HOLDING, source=False)
    return head + ("" if vocab._norm(raw) == vocab._norm(main) else " " + source_chip(raw))


_MONO = f"font-family:{tk.FONT_MONO};font-variant-numeric:tabular-nums;"


def _signed(text: str, value) -> str:
    """En förändring i teckenfärg (bläcktonen, texten är under 19 px) — 0 och saknat står i MUTED."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0.0
    col = tk.ink(tk.GREEN) if v > 0 else tk.ink(tk.RED) if v < 0 else tk.MUTED
    return f'<span style="color:{col};">{_h(text)}</span>'


def _regime_html(regime_signal: Optional[dict], tranches: int) -> str:
    """Regimdelen av kortets tredje rad — bara det raden visade förut: Qualitys verdikt med 'n/m signals', eller
    Contrarians steg med trancheprickarna; fel och saknat som Saknas-chipet."""
    if not regime_signal:
        return ""
    rtype = regime_signal.get("type", "n/a")
    if rtype == "quality":
        verdict = regime_signal.get("verdict", "N/A")
        return (chip(verdict, why="Alpha Regime") + f' <span style="{_MONO}white-space:nowrap;">'
                f'{_h(regime_signal.get("passed", "?"))}/{_h(regime_signal.get("total", "?"))} signals</span>')
    if rtype == "contrarian":
        rc = regime_signal.get("color", DIM)
        stage = str(regime_signal.get("stage", "HOLD"))
        short = stage.replace("ACCUMULATE_", "ACC·").replace("DISTRIBUTE_", "DIST·")
        dots = "".join(f'<span style="color:{STRATEGY_COLORS["Deep Contrarian"]};">●</span>' if i < tranches
                       else f'<span style="color:{tk.MUTED};">○</span>' for i in range(3))
        return (f'<span style="color:{escape(str(rc))};font-weight:700;">{_h(short)}</span> '
                f'<span style="letter-spacing:2px;">{dots}</span>')
    if rtype == "error":
        return missing(f"Alpha Regime: {regime_signal.get('error', '')}")
    if rtype == "unavailable":
        return missing("Alpha Regime saknas")
    return ""


def holding_card_html(holding: dict, live: dict, signal: dict, portfolio_key: str,
                      regime_signal: Optional[dict] = None) -> str:
    """Kortet som EN HTML-sträng (ritas med st.markdown), id='pos-…':
      L1  TICKER (17 px) · sektor · n st — till höger signalens chippar
      L2  kurs (dagens %) · värde · P/L % (belopp), mono 14 px — bara dagens % och P/L i teckenfärg, kursen i TEXT
      L3  signalens detalj, Qualitys verdikt eller Contrarians steg och trancheprickarna, 12 px MUTED
    Samma tal och format som raden hade (f'{price:.2f}', f'{pnl:+.1f}%', f'{v:,.0f}'). På datorn (≥ 1024 px)
    lägger sidans CSS delarna i de sex gamla kolumnerna på en rad."""
    ticker = holding["ticker"]
    entry = holding.get("entry_price", 0)
    shares = holding.get("shares", 0)
    price = live.get("price", 0)
    change = live.get("change_1d", 0)
    sig = signal.get("signal", "N/A")
    detail = signal.get("detail", "")
    strat_c = STRATEGY_COLORS.get(_strategy_of(holding), DIM)
    tranches = holding.get("tranches_deployed", 0)

    if entry > 0 and price > 0:
        pnl = (price / entry - 1) * 100
        pnl_html = _signed(f"{pnl:+.1f}%", pnl)
    else:
        pnl = 0
        pnl_html = f'<span style="color:{tk.MUTED};">—</span>'
    if shares > 0 and entry > 0 and price > 0:
        abs_pnl = (price - entry) * shares
        pnl_html += " " + _signed(f"({abs_pnl:+,.0f})", abs_pnl)
        pos_text = f"{shares * price:,.0f}"
    else:
        pos_text = "—"
    if price > 0:
        px_html = f'{price:.2f} ' + _signed(f"({change:+.1f}%)", change)
    else:                                        # ingen kurs: Saknas, aldrig en tyst 0.00
        px_html = missing("Ingen kurs från Yahoo just nu")

    # skiljepunkten hänger på ordet FÖRE (aldrig först på en ombruten rad): "TICKER ·" "Sektor ·" "n st"
    parts = [p for p in (str(holding.get("sector", "") or ""), f"{shares} st" if shares > 0 else "") if p]
    meta = "".join(f' <span style="white-space:nowrap;">{_h(p)}{"&nbsp;·" if i < len(parts) - 1 else ""}</span>'
                   for i, p in enumerate(parts))
    reg = _regime_html(regime_signal, tranches)
    sep = f'<span class="pos-sep" style="color:{tk.MUTED};">· </span>'
    lbl = f'class="pos-lbl" style="color:{tk.MUTED};font-family:{tk.FONT_UI};font-size:12px;"'
    l3 = " ".join(p for p in (f'<span class="pos-det">{_h(detail)}</span>' if detail else "",
                              f'<span class="pos-reg">{sep if detail else ""}{reg}</span>' if reg else "") if p)
    dash = "" if reg else '<span class="pos-reg pos-dash" style="display:none;">—</span>'   # bara på datorn
    return (
        f'<div class="pos-card" id="{card_id(portfolio_key, ticker)}" style="background:{tk.BG_CARD};'
        f'border:1px solid {tk.BORDER_CARD};border-left:3px solid {strat_c};border-radius:12px;padding:10px 12px;'
        f'display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px 8px;align-items:center;min-width:0;'
        f'box-sizing:border-box;">'
        f'<div class="pos-id" style="min-width:0;line-height:1.3;">'
        f'<span style="white-space:nowrap;"><span style="color:{tk.TEXT};font-family:{tk.FONT_UI};font-size:17px;'
        f'font-weight:700;">{_h(ticker)}</span>'
        + (f'<span style="color:{tk.MUTED};font-size:13px;">&nbsp;·</span>' if meta else "") + '</span>'
        + (f'<span class="pos-meta" style="color:{tk.MUTED};font-size:13px;">{meta}</span>' if meta else "")
        + f'</div><div class="pos-sig" style="display:flex;flex-wrap:wrap;gap:4px;justify-content:flex-end;">'
        f'{signal_chips(sig, detail)}</div>'
        f'<div class="pos-l2" style="grid-column:1/-1;display:flex;flex-wrap:wrap;align-items:baseline;gap:2px 14px;'
        f'color:{tk.TEXT};{_MONO}font-size:14px;line-height:1.4;min-width:0;">'
        f'<span class="pos-px" style="white-space:nowrap;">{px_html}</span>'
        f'<span class="pos-val" style="white-space:nowrap;"><span {lbl}>värde </span>{_h(pos_text)}</span>'
        f'<span class="pos-pl" style="white-space:nowrap;"><span {lbl}>P/L </span>{pnl_html}</span></div>'
        f'<div class="pos-l3" style="grid-column:1/-1;color:{tk.MUTED};font-size:12px;line-height:1.5;min-width:0;'
        f'overflow-wrap:anywhere;{"" if l3 else "display:none;"}">{l3}{dash}</div>'
        "</div>")


def columns_html() -> str:
    """De sex gamla kolumnrubrikerna (TICKER … REGIME) som EN rad — bara på datorn, där korten är tabellrader;
    på telefonen är de dolda (stilen display:none, sidans CSS visar dem från 1024 px)."""
    cells = "".join(f'<span class="pos-col-{k}">{t}</span>' for k, t in
                    (("id", "TICKER"), ("px", "PRICE (1D%)"), ("val", "VALUE"), ("pl", "P&amp;L"),
                     ("sig", "SIGNAL"), ("reg", "REGIME")))
    return (f'<div class="pos-cols" style="display:none;color:{tk.MUTED};font-family:{tk.FONT_UI};font-size:12px;'
            f'font-weight:600;letter-spacing:.08em;">{cells}</div>')


def _holding_card(
    holding: dict,
    live: dict,
    signal: dict,
    portfolio_key: str,
    regime_signal: Optional[dict] = None,
) -> None:
    """Ett innehav: kortet (EN st.markdown) och under det '⚙ Edit <ticker>' — strategi, trancher, stopp, mål,
    sektor och Save (samma nycklar och samma sparlogik: inget sparas utan Save) och nu också 'Ta bort' bakom en
    bekräftelse (components.confirm_delete, nyckeln rm_<hink>_<ticker>) i stället för förra ✕-knappen."""
    try:
        ticker   = holding["ticker"]
        strategy = _strategy_of(holding)
        tranches = holding.get("tranches_deployed", 0)
        with st.container(key=f"posc_{portfolio_key}_{ticker}"):
            st.markdown(holding_card_html(holding, live, signal, portfolio_key, regime_signal),
                        unsafe_allow_html=True)

            # Inline edit expander (strategy tag + tranches, stopp, mål och sektor — och Ta bort)
            with st.expander(f"⚙ Edit {ticker}", expanded=False, key=f"pos_edit_{portfolio_key}_{ticker}"):
                e1, e2 = st.columns([2, 1.5])
                with e1:
                    idx = STRATEGY_TAGS.index(strategy) if strategy in STRATEGY_TAGS else 4
                    new_strat = st.selectbox(
                        "Strategy",
                        STRATEGY_TAGS,
                        index=idx,
                        key=f"edit_strat_{portfolio_key}_{ticker}",
                    )
                with e2:
                    show_tr = (strategy == "Deep Contrarian") or (new_strat == "Deep Contrarian")
                    new_tranches = tranches
                    if show_tr:
                        new_tranches = st.selectbox(
                            "Tranches deployed",
                            [0, 1, 2, 3],
                            index=min(tranches, 3),
                            format_func=lambda x: f"{x}/3",
                            key=f"edit_tr_{portfolio_key}_{ticker}",
                        )
                # Stopp och mål på positionen (journalen räknar R ur stoppen) och sektorn — sparas med Save
                stop, target, sector = _price_or_none(holding.get("stop")), _price_or_none(holding.get("target")), \
                    str(holding.get("sector") or "Unknown")
                k_stop, k_sec = f"edit_stop_{portfolio_key}_{ticker}", f"edit_sec_{portfolio_key}_{ticker}"
                s1, s2, s3 = st.columns([1, 1, 1.6])
                new_stop = s1.number_input("Stopp", min_value=0.0, value=stop, step=0.01, key=k_stop,
                                           help="Kursen där positionen säljs. Journalen räknar R ur den när "
                                                "positionen stängs.")
                new_target = s2.number_input("Mål", min_value=0.0, value=target, step=0.01,
                                             key=f"edit_target_{portfolio_key}_{ticker}")
                new_sector = s3.text_input("Sektor", value=sector, key=k_sec)
                _stop_line(holding, live.get("df"), portfolio_key, k_stop)
                _sector_line(holding, portfolio_key, k_sec)
                # Save och Ta bort på en rad; raderingen först efter "Ja, ta bort" i popovern
                with st.container(horizontal=True, vertical_alignment="center",
                                  key=f"pos_act_{portfolio_key}_{ticker}"):
                    save = st.button("Save", key=f"save_edit_{portfolio_key}_{ticker}")
                    remove = confirm_delete("Ta bort", key=f"rm_{portfolio_key}_{ticker}",
                                            help=f"Tar bort {ticker} ur Positioner — öppnar en bekräftelse.")
                if remove:
                    _remove_holding(portfolio_key, ticker)
                    st.rerun()
                if save:
                    fields: dict = {}
                    if storage.differs(new_stop, stop):
                        fields.update(stop=new_stop, stop_basis="")      # egen stopp: ingen regel bakom den
                    if storage.differs(new_target, target):
                        fields["target"] = new_target
                    if (new_sector or "").strip() and new_sector.strip() != sector:
                        fields["sector"] = new_sector.strip()
                    if fields:                                           # före strategin: den kan flytta raden
                        positions.update(ticker, portfolio_key, row_id=holding.get("id"), **fields)
                    if new_strat != strategy:
                        _edit_holding_field(portfolio_key, ticker, "strategy", new_strat)
                    if new_tranches != tranches:
                        _edit_holding_field(portfolio_key, ticker, "tranches_deployed", new_tranches)
                    st.rerun()

    except Exception as e:
        st.warning(f"Error displaying {holding.get('ticker', '?')}: {e}")


# ── Beslutsremsan "BEHÖVER ÅTGÄRD I DAG" (§4.9) ──────────────────────────────
# Fylls EFTER sektionerna: de lägger varje innehavs redan räknade signal och regelvaktens varningar i en lista
# (run-lokal, ingenting sparas) — remsan härleder inget eget ord, den visar bara de raderna som redan säger
# EXIT/SELL, REDUCE eller WATCH, och regelvaktens egna varningar ordagrant.
STRIP_KICKER = "BEHÖVER ÅTGÄRD I DAG"
STRIP_EMPTY = "✓ Inga säljlägen i dag — alla innehav HÅLL"
STRIP_SHOWN = 3                                   # rader som syns; resten bakom "+n till ↓"
_ORDER = (vocab.SALJ, vocab.MINSKA, vocab.VAKSAM)
RULE_WORD = "REGEL"                               # chipet på regelvaktens rader (källan: QUALITY/CONTRARIAN RULES)


def rules_id(strategy_tag: str) -> str:
    """Id på regelvaktens rubrik — målet för remsans regelrader."""
    return "pos-rules-" + (re.sub(r"[^a-z0-9]+", "-", str(strategy_tag).lower()).strip("-") or "x")


def _md_text(text: str) -> str:
    """Text i st.warning/st.error (markdown): '>' escapas så att '⚠ >10% position' visas med '>' (annars blev
    resten ett citatblock och '>' försvann)."""
    return str(text).replace(">", "\\>")


def strip_items(items: list) -> tuple:
    """(remsans rader, innehav vars signal saknas, antal innehav) ur sektionernas lista. Raderna: innehav med
    SÄLJ (EXIT/SELL), sedan MINSKA (REDUCE), sedan VAKSAM (WATCH) i sidans ordning — och regelvaktens
    meddelanden i den ordning de ritades."""
    holds = [i for i in items if i.get("kind") == "holding"]
    rows = [i for m in _ORDER for i in holds if signal_main(i["signal"].get("signal")) == m]
    rows += [i for i in items if i.get("kind") == "rule"]
    absent = [i for i in holds if signal_main(i["signal"].get("signal")) is None]
    return rows, absent, len(holds)


def _strip_row(item: dict) -> str:
    if item.get("kind") == "rule":
        col = tk.RED if item.get("level") == "error" else tk.AMBER
        return action_row(RULE_WORD, col, item["text"], detail=f"{str(item['strategy']).upper()} RULES",
                          jump=rules_id(item["strategy"]))
    sig = item["signal"]
    main = signal_main(sig.get("signal"))
    raw = str(sig.get("signal", "")).strip()
    title = " · ".join(p for p in (item["ticker"], item["strategy"], str(sig.get("detail", "") or "")) if p)
    chips = () if vocab._norm(raw) == vocab._norm(main) else (source_chip(raw),)
    return action_row(main, MAIN_COLOR[main], title, chips=chips, jump=card_id(item["pkey"], item["ticker"]))


def strip_html(items: list) -> str:
    """Remsan som EN HTML-sträng: rubriken, de tre första raderna och '+n till ↓' (resten i en <details>).
    Inga rader: den gröna raden '✓ Inga säljlägen i dag — alla innehav HÅLL' — när någon signal saknas står i
    stället en grå Saknas-rad med antalet (aldrig 'alla HÅLL' när det inte är sant). Inga innehav: ''."""
    rows, absent, n = strip_items(items)
    if not rows and not n:
        return ""
    kicker = (f'<div style="color:{tk.MUTED};font-family:{tk.FONT_UI};font-size:12px;font-weight:600;'
              f'line-height:1.2;letter-spacing:.12em;text-transform:uppercase;">{STRIP_KICKER}</div>')
    gap = ""
    if absent:
        names = ", ".join(i["ticker"] for i in absent)
        gap = (f'<div class="pos-gap" style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-top:8px;'
               f'color:{tk.MUTED};font-size:13px;line-height:1.4;">{missing("Signalen kunde inte räknas (N/A)")}'
               f'<span>signal för {len(absent)} av {n} innehav: {_h(names)}</span></div>')
    if not rows:
        ok = len(absent) == 0
        text = STRIP_EMPTY if ok else "✓ Inga säljlägen i dag"
        return (f'<div class="pos-strip pos-ok" style="background:{tk.tint(tk.GREEN, .06)};'
                f'border:1px solid {tk.tint(tk.GREEN, .4)};border-left:4px solid {tk.GREEN};border-radius:14px;'
                f'padding:12px 14px;min-width:0;">{kicker}'
                f'<div style="color:{tk.ink(tk.GREEN)};font-family:{tk.FONT_UI};font-size:16px;font-weight:700;'
                f'line-height:1.35;margin-top:6px;">{_h(text)}</div>{gap}</div>')
    # vänsterlinjen (§3.5.9): ordets färg när remsan bara har ETT ord, annars ACCENT
    words = {(RULE_WORD, tk.RED if i.get("level") == "error" else tk.AMBER) if i.get("kind") == "rule"
             else (m, MAIN_COLOR[m]) for i in rows for m in (signal_main(i.get("signal", {}).get("signal")),)}
    rule = next(iter(words))[1] if len(words) == 1 else tk.ACCENT
    shown = "".join(_strip_row(i) for i in rows[:STRIP_SHOWN])
    rest = rows[STRIP_SHOWN:]
    more = ""
    if rest:
        more = (f'<details class="pos-more"><summary style="display:flex;align-items:center;min-height:44px;'
                f'cursor:pointer;color:{tk.ACCENT};font-family:{tk.FONT_UI};font-size:14px;font-weight:600;'
                f'list-style:none;">+{len(rest)} till ↓</summary>{"".join(_strip_row(i) for i in rest)}</details>')
    return (f'<div class="pos-strip" style="background:{tk.BG_ALT};border:1px solid {tk.BORDER_CARD};'
            f'border-left:4px solid {rule};border-radius:12px;padding:12px 14px 6px;min-width:0;">'
            f'{kicker}{shown}{more}{gap}</div>')


def _rule_label(strategy_tag: str, text: str) -> None:
    """Regelvaktens rubrik (12 px MUTED) med id — remsans regelrader hoppar hit."""
    st.markdown(f'<div id="{rules_id(strategy_tag)}">{section(text)}</div>', unsafe_allow_html=True)


def _render_rule_monitor(strategy_tag: str, strategy_holdings: list, actions: Optional[list] = None) -> None:
    """Rule compliance checks for Quality (position size) and Deep Contrarian (sector).
    Varningarna och felen står kvar som st.warning/st.error (samma text, '>' escapad så att den syns); med
    actions läggs samma text också i remsans lista — ordagrant, inget nytt räknat."""
    if not strategy_holdings:
        return

    def _flag(level: str, text: str) -> None:
        (st.error if level == "error" else st.warning)(_md_text(text))
        if actions is not None:
            actions.append({"kind": "rule", "level": level, "text": text, "strategy": strategy_tag})

    if strategy_tag == "Quality":
        count = len(strategy_holdings)
        total_q = 0.0
        pos_vals: dict = {}
        for h in strategy_holdings:
            pkey = h.get("_portfolio_key", "long")
            live = _fetch_live_price(h["ticker"], period="2y" if pkey == "long" else "6mo")
            val = _sek(h["ticker"], live.get("price", 0) * h.get("shares", 0))
            pos_vals[h["ticker"]] = val
            total_q += val

        _rule_label(strategy_tag, "QUALITY RULES")
        rc1, rc2 = st.columns(2)
        with rc1:
            if count < 8:
                _flag("warning", f"⚠ {count}/10 holdings — target 8–10")
            elif count > 10:
                _flag("error", f"🔴 {count} holdings — max is 10")
            else:
                st.success(f"✓ {count}/10 holdings in target range")
        with rc2:
            if total_q > 0:
                oversized = [
                    f"{t} ({v / total_q * 100:.1f}%)"
                    for t, v in pos_vals.items()
                    if v / total_q * 100 > 10
                ]
                if oversized:
                    _flag("warning", f"⚠ >10% position: {', '.join(oversized)}")
                else:
                    st.success("✓ All positions ≤ 10%")

    elif strategy_tag == "Deep Contrarian":
        sector_vals: dict = {}
        total_c = 0.0
        tranche_counts: dict = {0: 0, 1: 0, 2: 0, 3: 0}

        for h in strategy_holdings:
            pkey = h.get("_portfolio_key", "long")
            live = _fetch_live_price(h["ticker"], period="2y" if pkey == "long" else "6mo")
            val = _sek(h["ticker"], live.get("price", 0) * h.get("shares", 0))
            sector = h.get("sector", "Unknown")
            sector_vals[sector] = sector_vals.get(sector, 0.0) + val
            total_c += val
            t = min(h.get("tranches_deployed", 0), 3)
            tranche_counts[t] = tranche_counts[t] + 1

        _rule_label(strategy_tag, "CONTRARIAN RULES")
        rc1, rc2 = st.columns(2)
        with rc1:
            if total_c > 0:
                oversized = [
                    f"{s} ({v / total_c * 100:.1f}%)"
                    for s, v in sector_vals.items()
                    if v / total_c * 100 > 25
                ]
                if oversized:
                    _flag("warning", f"⚠ Sector >25%: {', '.join(oversized)}")
                else:
                    st.success("✓ All sectors ≤ 25%")
        with rc2:
            st.caption(
                f"Tranches — 0/3: {tranche_counts[0]} · "
                f"1/3: {tranche_counts[1]} · "
                f"2/3: {tranche_counts[2]} · "
                f"3/3: {tranche_counts[3]}"
            )


def section_head_html(strategy_tag: str, count: int, alloc: Optional[tuple] = None) -> str:
    """Strategins rubrikrad (§4.9): en 4 px stapel i STRATEGY_COLORS, namnet (17 px) och till höger det som redan
    fanns — antalet, och med allokeringsstaplarnas tal (värde i SEK, andel) när de är kända."""
    color = STRATEGY_COLORS.get(strategy_tag, DIM)
    right = f"{count} holding{'s' if count != 1 else ''}"
    if alloc and alloc[0] > 0:
        right += f" · {alloc[0]:,.0f} SEK · {alloc[1]:.1f}%"
    return (f'<div class="pos-sec" style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;'
            f'gap:4px 12px;border-left:4px solid {color};padding:2px 0 2px 10px;margin:20px 0 0;min-width:0;">'
            f'<span style="color:{tk.TEXT};font-family:{tk.FONT_UI};font-size:17px;font-weight:700;line-height:1.3;'
            f'letter-spacing:.06em;">{_h(strategy_tag.upper())}</span>'
            f'<span style="color:{tk.MUTED};{_MONO}font-size:12px;line-height:1.3;white-space:nowrap;">'
            f'{_h(right)}</span></div>')


def _render_strategy_section(strategy_tag: str, actions: Optional[list] = None,
                             alloc: Optional[tuple] = None) -> None:
    """Render one strategy's holdings section: add form, cards, rule monitor.
    actions = sidans run-lokala lista för remsan: varje innehavs redan räknade signal och regelvaktens varningar
    läggs där (inget sparas). alloc = (värde SEK, andel %) ur allokeringsstaplarna, till rubrikraden."""
    all_flat = _get_all_holdings_flat()
    strategy_holdings = [h for h in all_flat if _strategy_of(h) == strategy_tag]
    pkey_for_add = _STRATEGY_TO_KEY.get(strategy_tag, "long")

    # Section header
    st.markdown(section_head_html(strategy_tag, len(strategy_holdings), alloc), unsafe_allow_html=True)

    # Add holding form
    with st.expander(f"Add {strategy_tag} holding", expanded=len(strategy_holdings) == 0):
        if strategy_tag == "Deep Contrarian":
            ac1, ac2, ac3, ac4, ac5, ac6 = st.columns([2, 1.2, 1, 1.2, 1.2, 0.8])
        else:
            ac1, ac2, ac3, ac4, ac5 = st.columns([2, 1.2, 1, 1.2, 0.8])

        with ac1:
            new_ticker = st.text_input(
                "Ticker", key=f"add_t_{strategy_tag}",
                placeholder="VOLV-B.ST",
            ).strip().upper()
        with ac2:
            new_entry = st.number_input(
                "Entry price", value=0.0, min_value=0.0, step=0.01,
                key=f"add_e_{strategy_tag}",
            )
        with ac3:
            new_shares = st.number_input(
                "Shares", value=0, min_value=0, step=1,
                key=f"add_s_{strategy_tag}",
            )
        with ac4:
            new_sector = st.text_input(
                "Sector", value="Unknown", key=f"add_sec_{strategy_tag}",
            )
        new_tranches = 0
        if strategy_tag == "Deep Contrarian":
            with ac5:
                new_tranches = st.selectbox(
                    "Tranches",
                    [0, 1, 2, 3],
                    format_func=lambda x: f"{x}/3",
                    key=f"add_tr_{strategy_tag}",
                )
            with ac6:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                add_btn = st.button("+ ADD", key=f"add_btn_{strategy_tag}", use_container_width=True)
        else:
            with ac5:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                add_btn = st.button("+ ADD", key=f"add_btn_{strategy_tag}", use_container_width=True)

        if add_btn and new_ticker:
            if _add_holding(
                pkey_for_add, new_ticker, new_entry, new_sector, new_shares,
                strategy=strategy_tag, tranches_deployed=new_tranches,
            ):
                st.rerun()

    if not strategy_holdings:
        st.markdown(
            f"<div style='color:{tk.MUTED};text-align:center;padding:10px;"
            f"font-size:13px;'>No {strategy_tag} holdings yet.</div>",
            unsafe_allow_html=True,
        )
        return

    # Regime signal toggle (Quality + Deep Contrarian only)
    show_regime = False
    if strategy_tag in ("Quality", "Deep Contrarian"):
        show_regime = st.checkbox(
            "Show Alpha Regime signals (1h cache — may be slow on first load)",
            key=f"regime_toggle_{strategy_tag}",
            value=False,
        )

    # Kolumnrubrikerna: EN rad, bara på datorn där korten läses som en tabell (dold på telefonen)
    st.markdown(columns_html(), unsafe_allow_html=True)

    # Holdings rows
    total_value  = 0.0
    total_abs_pnl = 0.0
    total_pnl    = 0.0
    pnl_count    = 0

    for holding in strategy_holdings:
        ticker     = holding["ticker"]
        pkey       = holding["_portfolio_key"]
        fetch_p    = "2y" if pkey == "long" else "6mo"
        live       = _fetch_live_price(ticker, period=fetch_p)
        df         = live.get("df")
        signal     = _signal_for_holding(holding, df)
        regime_sig = _get_regime_signal(ticker, strategy_tag) if show_regime else None
        if actions is not None:                  # remsan visar samma signal — ingenting räknas om
            actions.append({"kind": "holding", "ticker": ticker, "strategy": strategy_tag, "pkey": pkey,
                            "signal": dict(signal or {})})

        _holding_card(holding, live, signal, pkey, regime_signal=regime_sig)

        price  = live.get("price", 0)
        shares = holding.get("shares", 0)
        entry  = holding.get("entry_price", 0)
        if shares > 0 and price > 0:
            total_value += _sek(ticker, price * shares)
        if entry > 0 and price > 0:
            total_pnl += (price / entry - 1) * 100
            pnl_count += 1
            if shares > 0:
                total_abs_pnl += _sek(ticker, (price - entry) * shares)

    # Section summary footer (samma tal och format; värdet i TEXT, P/L i teckenfärg, etiketterna 12 px MUTED)
    if pnl_count > 0:
        avg_pnl = total_pnl / pnl_count
        lab = f"color:{tk.MUTED};font-size:12px;font-family:{tk.FONT_UI};"
        num = f"{_MONO}font-size:14px;font-weight:700;"
        val_part = (
            f'<span style="{lab}">Value: </span>'
            f'<span style="{num}color:{tk.TEXT};">{total_value:,.0f} SEK</span>  '
        ) if total_value > 0 else ""
        abs_part = (
            f'<span style="{lab}">P&L: </span>'
            f'<span style="{num}color:{tk.ink(GREEN if total_abs_pnl >= 0 else RED)};">'
            f"{total_abs_pnl:+,.0f} SEK</span>  "
        ) if total_value > 0 else ""
        st.markdown(
            f'<div class="pos-foot" style="text-align:right;padding:6px 0 0;line-height:1.6;'
            f'border-top:1px solid {tk.BORDER_FAINT};margin-top:2px;">'
            f"{val_part}{abs_part}"
            f'<span style="{lab}">Avg P&L: </span>'
            f'<span style="{num}color:{tk.ink(GREEN if avg_pnl >= 0 else RED)};">'
            f"{avg_pnl:+.1f}%</span></div>",
            unsafe_allow_html=True,
        )

    _render_rule_monitor(strategy_tag, strategy_holdings, actions)


PAGE_SUBTITLE = "Alla strategiers positioner · säljreglerna · signalerna"


def allocation_html(strat_vals: dict, counts: dict, cash: float, grand_total: float) -> str:
    """STRATEGY ALLOCATION-staplarna som EN markdown — samma tal och format (f'{v:,.0f} SEK', f'{p:.1f}%', '(n)'),
    all text minst 12 px; etiketten bryts hellre än att trycka ihop stapeln på telefonen."""
    def _row(tag: str, color: str, val: float, pct: float, n_txt: str, opacity: float) -> str:
        # fasta kolumner för talen: raderna är egna rutnät, så talen står ändå i linje
        return (f'<div class="pos-alloc" style="display:grid;grid-template-columns:minmax(0,96px) minmax(32px,1fr) '
                f'84px 44px 24px;align-items:center;gap:6px;margin:3px 0;">'
                f'<span style="color:{color};font-family:{tk.FONT_UI};font-size:12px;font-weight:700;line-height:1.2;'
                f'text-align:right;letter-spacing:.03em;overflow-wrap:anywhere;">{_h(tag)}</span>'
                f'<div style="background:rgba(255,255,255,0.05);border-radius:2px;height:9px;">'
                f'<div style="width:{max(pct, 0.3):.1f}%;background:{color};opacity:{opacity};height:9px;'
                f'border-radius:2px;"></div></div>'
                f'<span style="color:{tk.MUTED};{_MONO}font-size:12px;text-align:right;white-space:nowrap;">'
                f'{val:,.0f} SEK</span>'
                f'<span style="color:{tk.TEXT};{_MONO}font-size:12px;text-align:right;">{pct:.1f}%</span>'
                f'<span style="color:{tk.MUTED};{_MONO}font-size:12px;">{n_txt}</span></div>')
    rows = []
    for tag in STRATEGY_TAGS:
        val, count = strat_vals.get(tag, 0.0), counts.get(tag, 0)
        if val == 0 and count == 0:
            continue
        pct = val / grand_total * 100 if grand_total > 0 else 0
        color = STRATEGY_COLORS.get(tag, DIM)        # Insider, Alpha m.fl. saknar egen färg — fällde sidan
        rows.append(_row(tag.upper(), tk.ink(color), val, pct, f"({count})", 0.85))
    if cash > 0:
        rows.append(_row("CASH", tk.MUTED, cash, cash / grand_total * 100, "", 0.7))
    return section("STRATEGY ALLOCATION") + "".join(rows)


def _render_summary_header():
    """Sidans topp (§4.9): huvudraden POSITIONER (guld, vänster) med rundans pill, underraden, platsen för
    remsan 'BEHÖVER ÅTGÄRD I DAG' (fylls efter sektionerna) och de fyra st.metric i kpis_holdings — på datorn
    sida vid sida — sedan Cash (SEK) över hela bredden och allokeringsstaplarna.
    Returnerar (remsans plats, {strategi: (värde SEK, andel %)}) — talen är allokeringsstaplarnas egna."""
    all_flat = _get_all_holdings_flat()
    cash = positions.cash()

    # Compute per-strategy invested values (uses cached price data)
    strat_vals: dict = {tag: 0.0 for tag in STRATEGY_TAGS}
    total_invested = 0.0
    for h in all_flat:
        shares = h.get("shares", 0)
        if shares <= 0:
            continue
        pkey = h.get("_portfolio_key", "long")
        try:
            live = _fetch_live_price(h["ticker"], period="2y" if pkey == "long" else "6mo")
            val  = _sek(h["ticker"], live.get("price", 0) * shares)
            strat_vals[_strategy_of(h)] = strat_vals.get(_strategy_of(h), 0.0) + val
            total_invested += val
        except Exception:
            pass

    grand_total = total_invested + cash
    counts = {tag: sum(1 for h in all_flat if _strategy_of(h) == tag) for tag in STRATEGY_TAGS}

    # Huvudraden: rubriken (guld, vänster) och Morgonrundans pill — pillret bara medan rundan pågår
    from ui import routine
    from ui.components import head_row, page_header
    with head_row("holdings", subtitle=PAGE_SUBTITLE):
        page_header("Positioner")
        routine.pill("holdings")

    # Remsan och nyckeltalen: under varandra på telefonen, sida vid sida på datorn (sidans CSS)
    with st.container(key="pos_top"):
        strip_col, kpi_col = st.columns([3, 2])
        # En container, inte st.empty(): en tom plats blankar förra körningens remsa direkt vid omkörningen tills
        # sektionerna är klara, och allt under hoppar ~280 px upp och tillbaka. Containern behåller den gamla
        # remsan tills den nya skrivs (samma index); utan innehav rensar Streamlit den vid körningens slut.
        slot = strip_col.container()
        with kpi_col:
            with st.container(horizontal=True, key="kpis_holdings"):
                st.metric("Total Holdings", len(all_flat))
                st.metric("Invested", f"{total_invested:,.0f} SEK" if total_invested > 0 else "—")
                st.metric("Cash", f"{cash:,.0f} SEK" if cash > 0 else "—")
                st.metric("Portfolio Total", f"{grand_total:,.0f} SEK" if grand_total > 0 else "—")

    new_cash = st.number_input(
        "Cash (SEK)", value=cash, min_value=0.0, step=1000.0,
        key="holdings_cash_input",
    )
    if new_cash != cash:
        positions.set_cash(new_cash)

    # Allocation bars
    alloc: dict = {}
    if grand_total > 0:
        st.markdown(allocation_html(strat_vals, counts, cash, grand_total), unsafe_allow_html=True)
        alloc = {tag: (strat_vals.get(tag, 0.0), strat_vals.get(tag, 0.0) / grand_total * 100)
                 for tag in STRATEGY_TAGS}
    return slot, alloc


# ── Correlation Matrix ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_returns_for_correlation(tickers: tuple, period: str = "6mo") -> pd.DataFrame:
    """Fetch daily returns for a list of tickers."""
    try:
        data = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data[["Close"]]
            close.columns = [tickers[0]]
        returns = close.pct_change().dropna()
        if len(returns) > 90:
            returns = returns.tail(90)
        return returns
    except Exception:
        return pd.DataFrame()


def _render_correlation_matrix(all_holdings: list) -> None:
    """Render a correlation heatmap for all active holdings."""
    try:
        if len(all_holdings) < 2:
            st.info("Add at least 2 holdings to see correlation.")
            return

        tickers      = [h["ticker"] for h in all_holdings]
        strategy_map = {h["ticker"]: h.get("strategy", h.get("_portfolio_key", "")) for h in all_holdings}

        returns_df = _fetch_returns_for_correlation(tuple(tickers))
        if returns_df.empty or len(returns_df.columns) < 2:
            st.warning("Correlation: insufficient price data.")
            return

        valid_tickers = [t for t in tickers if t in returns_df.columns]
        if len(valid_tickers) < 2:
            st.warning("Correlation: fewer than 2 tickers with data.")
            return

        corr = returns_df[valid_tickers].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=valid_tickers,
            y=valid_tickers,
            colorscale=[
                [0.0, "rgba(45,138,78,0.9)"],
                [0.5, "rgba(255,255,255,0.9)"],
                [1.0, "rgba(196,69,69,0.9)"],
            ],
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont={"size": 11, "color": "rgba(232,228,220,0.9)"},
            hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>Corr: %{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title=dict(text="Corr", font=dict(color=DIM)),
                tickfont=dict(color=DIM),
            ),
        ))
        fig.update_layout(
            title=dict(
                text="PORTFOLIO CORRELATION — 90d daily returns",
                font=dict(color=STRATEGY_COLORS["Quality"], size=13),
            ),
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXT),
            xaxis=dict(tickfont=dict(color=CYAN, size=10), gridcolor="rgba(201,168,76,0.1)"),
            yaxis=dict(tickfont=dict(color=CYAN, size=10), gridcolor="rgba(201,168,76,0.1)", autorange="reversed"),
            height=max(350, 60 * len(valid_tickers)),
            margin=dict(l=80, r=40, t=50, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)

        warned: set = set()
        for i in range(len(valid_tickers)):
            for j in range(i + 1, len(valid_tickers)):
                c = corr.iloc[i, j]
                pair = (valid_tickers[i], valid_tickers[j])
                if pair in warned:
                    continue
                if c > 0.85:
                    st.error(f"🔴 Extreme correlation: {pair[0]}↔{pair[1]} ({c:.2f})")
                    warned.add(pair)
                elif c > 0.70:
                    st.warning(f"⚠ High correlation: {pair[0]}↔{pair[1]} ({c:.2f})")
                    warned.add(pair)
    except Exception as e:
        st.warning(f"Correlation matrix: {e}")


# ── Main render ───────────────────────────────────────────────────────────────

# Sidans egna regler (LIDSKJALV §4.9) i en dold stilbehållare (ui.shell.page_style), valda på sidans egna nycklar:
# sparraden och låset på EN rad, remsan och nyckeltalen (sida vid sida från 1024 px), korten (de sex gamla kolumnerna
# på en rad från 1024 px), 44 px tryckytor och 12 px-golvet för det sidan och dess delar (risk, rapporter,
# metaller) ritar. Inga etiketter i selektorerna.
_PAGE = ".st-key-pos_page"
_LOCK = '.st-key-holdings_pw_lock:has(+ [data-testid="stLayoutWrapper"] > .st-key-save_holdings_row)'
_SAVE_ROW = '.st-key-holdings_pw_lock + [data-testid="stLayoutWrapper"] > .st-key-save_holdings_row'
_COLS = '.st-key-pos_top [data-testid="stHorizontalBlock"]:not([class*="st-key-kpis_"])'   # remsan | nyckeltalen
_SMALL = ("0.6rem", "0.64rem", "0.65rem", "0.68rem", "0.7rem", "0.72rem", "0.73rem", ".85em")
LOCK_W, SAVE_W = 148, 92            # px; etiketterna mäter 108 och 59 px i 13 px — luft för ett bredare reservtypsnitt


def page_css() -> str:
    from ui.shell import page_scope
    scope = page_scope("holdings")
    floor = ",\n".join(f'{scope} [style*="font-size:{v}"], {scope} [style*="font-size: {v}"]' for v in _SMALL)
    # knappar med help= (Börsdatas sektor) ligger ett omslag djupare än .stButton > button: på datorn i
    # stTooltipHoverTarget, på pekskärm i ett eget omslag — samma sekundära utseende och 44 px (som Thor)
    tip = (f'{_PAGE} .stButton > div > button[kind="secondary"], '
           f'{_PAGE} .stButton [data-testid="stTooltipHoverTarget"] > button[kind="secondary"]')
    tip_in = ", ".join(f"{t} [data-testid=\"stMarkdownContainer\"], {t} p" for t in tip.split(", "))
    return f"""
/* Sparraden och låset (§4.9): [💾 Spara] · [🔒 Lås innehaven] · "Inga osparade ändringar" på EN rad. Låset ritas
   av private_lock.gate före sidan och sparraden direkt efter — låset dras ned i radens plats (negativ marginal) och
   texten börjar efter det. Ett läsfel mellan dem bryter grannskapet: då står de under varandra som förut. */
.st-key-holdings_pw_lock .stButton > button {{ min-height: 44px; padding: 0 12px; font-size: 13px;
    letter-spacing: .04em; white-space: nowrap; }}
{_LOCK} {{ position: relative; z-index: 2; width: {LOCK_W}px !important; margin-left: {SAVE_W + 8}px;
    margin-bottom: calc(-44px - 1rem); }}
{_LOCK} .stButton > button {{ width: {LOCK_W}px; padding: 0 8px; overflow: hidden; text-overflow: ellipsis; }}
{_SAVE_ROW} {{ min-height: 44px; }}
/* Spara överst i raden som låset, så knapparna står i linje även när texten bryts ("· senast 16:40 UTC") */
{_SAVE_ROW} > .st-key-save_holdings {{ align-self: flex-start; }}
{_SAVE_ROW} .st-key-save_holdings .stButton > button {{ width: {SAVE_W}px; padding: 0 8px; }}
{_SAVE_ROW} > .st-key-save_holdings + [data-testid="stElementContainer"] {{ margin-left: {LOCK_W + 4}px;
    flex: 1 1 0; min-width: 64px; }}
/* Under 390 px ryms inte texten bredvid knapparna: den tar nästa rad (knapparna står kvar på en rad) */
@media (max-width: 389px) {{
    {_SAVE_ROW} > .st-key-save_holdings + [data-testid="stElementContainer"] {{ margin-left: 0;
        flex: 1 1 100%; }}
}}

/* Sidan: 12 px mellan blocken, 4 px mellan kortet och dess ⚙ Edit; markdown-rutans −16 px bort under korten */
{_PAGE} {{ gap: 12px; }}
[class*="st-key-posc_"] {{ gap: 4px; }}
/* ⚙ Edit under kortet: en lätt rad på 44 px (ramen syns först när den är öppen) */
[class*="st-key-pos_edit_"] details {{ border-color: transparent !important; }}
[class*="st-key-pos_edit_"] details[open] {{ border-color: rgba(0,229,255,0.2) !important; }}
[class*="st-key-pos_edit_"] summary {{ min-height: 44px; padding-top: 0; padding-bottom: 0; }}
[class*="st-key-pos_edit_"] summary p {{ font-size: 13px; font-weight: 600 !important; color: {tk.MUTED} !important; }}
[class*="st-key-pos_edit_"] summary:hover p {{ color: {tk.ACCENT} !important; }}
{_PAGE} [data-testid="stMarkdownContainer"]:has(> .pos-card, > .pos-strip, > .pos-sec, > .pos-cols, > .wolf-sec,
    > .pos-foot, > div > .wolf-sec) {{ margin-bottom: 0; }}
/* Underrubriken: ~12 px luft till remsan (markdown-rutans −16 px gav 2–3 px) */
{_PAGE} [data-testid="stMarkdownContainer"]:has(> .wolf-note) {{ margin-bottom: -8px; }}

/* Remsan och nyckeltalen: under varandra upp till 1023 px, sida vid sida (3:2) från 1024 px; nyckeltalen 2 × 2 */
{_COLS} {{ gap: 12px; row-gap: 12px; }}
@media (max-width: 1023px) {{
    {_COLS} {{ flex-wrap: wrap; }}
    .st-key-pos_top [data-testid="stColumn"] {{ flex: 1 1 100% !important; width: 100% !important;
        min-width: 100% !important; }}
}}
@media (min-width: 1024px) {{
    .st-key-kpis_holdings > [data-testid="stElementContainer"] {{ flex-basis: calc(50% - 4px); }}
}}
.pos-strip .wolf-row:last-of-type {{ border-bottom: 0; }}
.pos-more > summary::-webkit-details-marker {{ display: none; }}
.pos-strip [data-jump] {{ cursor: pointer; }}
.pos-card, [id^="pos-rules-"] {{ scroll-margin-top: 72px; }}
@media (min-width: 641px) {{
    .pos-alloc {{ grid-template-columns: minmax(0, 150px) minmax(32px, 1fr) 96px 52px 28px !important; }}
}}

/* Datorn (≥ 1024 px): kortet är en tabellrad — de sex gamla kolumnerna, rubrikraden synlig */
@media (min-width: 1024px) {{
    .pos-cols {{ display: grid !important; grid-template-columns: 1.8fr 1.1fr .9fr 1fr 1.2fr 1.2fr; gap: 8px;
        padding: 0 12px 0 15px; }}
    .pos-cols .pos-col-px {{ text-align: right; }}
    .pos-cols .pos-col-val, .pos-cols .pos-col-pl, .pos-cols .pos-col-sig, .pos-cols .pos-col-reg {{
        text-align: center; }}
    .pos-card {{ grid-template-columns: 1.8fr 1.1fr .9fr 1fr 1.2fr 1.2fr !important;
        grid-template-areas: "id px val pl sig reg" "id px val pl det reg"; }}
    .pos-card .pos-l2, .pos-card .pos-l3 {{ display: contents !important; }}
    .pos-card .pos-sep, .pos-card .pos-lbl {{ display: none !important; }}
    .pos-card .pos-id {{ grid-area: id; }}
    .pos-card .pos-px {{ grid-area: px; text-align: right; }}
    .pos-card .pos-val {{ grid-area: val; text-align: center; }}
    .pos-card .pos-pl {{ grid-area: pl; text-align: center; }}
    .pos-card .pos-sig {{ grid-area: sig; justify-content: center; }}
    .pos-card .pos-det {{ grid-area: det; text-align: center; color: {tk.MUTED}; font-size: 12px; }}
    .pos-card .pos-reg {{ grid-area: reg; text-align: center; color: {tk.MUTED}; font-size: 12px; }}
    .pos-card .pos-dash {{ display: inline !important; }}
}}

/* Regelvaktens varningar som bärnstensfärgade kort (st.warning står kvar — testerna läser texten) */
{_PAGE} [data-testid="stAlert"] {{ border-radius: 12px; }}
{_PAGE} [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{
    border-left: 4px solid {tk.AMBER}; border-radius: 12px; }}
{_PAGE} [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{
    border-left: 4px solid {tk.RED}; border-radius: 12px; }}

/* Tryckytor (§2.4): expandrar, kryssrutor, fält, stegknappar, väljare, hjälpikoner — minst 44 px */
{scope} summary {{ min-height: 44px; box-sizing: border-box; }}
{_PAGE} [data-testid="stCheckbox"] label {{ min-height: 44px; align-items: center; box-sizing: border-box; }}
{scope} [data-testid="stNumberInputContainer"] {{ min-height: 44px; }}
{scope} [data-testid="stNumberInputContainer"] button {{ min-width: 44px; min-height: 44px; }}
{scope} .react-aria-ComboBox > div {{ min-height: 44px; }}
{scope} .react-aria-ComboBox button {{ min-width: 44px; min-height: 44px; }}
{scope} [data-testid="stPopoverButton"] {{ min-height: 44px; }}
{scope} :is([data-testid="stWidgetLabel"], [data-testid="stMetricLabel"]) [data-testid="stTooltipHoverTarget"] button {{
    min-width: 44px; min-height: 44px; margin: -14px; display: inline-flex; align-items: center;
    justify-content: center; }}
{scope} [data-testid="stCheckbox"] [data-testid="stTooltipHoverTarget"] button {{ min-width: 44px;
    min-height: 44px; display: inline-flex; align-items: center; justify-content: center; }}
{scope} [data-testid="stFileUploader"] button {{ min-height: 44px; }}
{tip} {{ min-height: 44px; padding: 0 16px; border-radius: 10px; border: 1px solid {tk.ACCENT};
    background: rgba(0,229,255,.08); color: {tk.ACCENT}; font-family: 'Space Grotesk', sans-serif; font-size: 14px;
    font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }}
{tip_in} {{ font: inherit; letter-spacing: inherit; }}

/* 12 px-golvet: inline-storlekar under 12 px i det som ritas på sidan (risk, rapporter, mäklarrutan, metaller) */
{floor} {{ font-size: 12px !important; }}
{scope} [data-testid="stCaptionContainer"], {scope} [data-testid="stCaptionContainer"] p {{ font-size: 13px; }}
"""


def render_holdings_page() -> None:
    """Positioner (LIDSKJALV §4.9, beslut 10). Ritordningen: sparraden (låset på samma rad), huvudraden, remsan
    'BEHÖVER ÅTGÄRD I DAG' och nyckeltalen, Cash, allokeringen, mäklarrutan, en sektion per strategi med ett kort
    per innehav och regelvakten — sedan Risk, Rapporter och Korrelation. Remsan fylls sist ur det sektionerna redan
    räknat (en run-lokal lista); inget sparas när sidan öppnas, inga regler eller tal är ändrade."""
    positions.load()
    # Sparraden som de andra arken: läsfel, osparat (bara efter en
    # Gist-migrering — ändringar sparas direkt) och senaste commit.
    import storage_ui
    from ui.shell import page_style
    storage_ui.save_bar(STORE, "Holdings")
    page_style(page_css(), "holdings")
    with st.container(key="pos_page"):
        slot, alloc = _render_summary_header()
        try:                                       # mäklarens filer → antal och snittkurs (sparas bara på klick)
            from portfolio.broker_sync_ui import render_broker_sync
            render_broker_sync()
        except Exception as exc:                   # rutan får aldrig fälla Positioner
            st.caption(f"Mäklarimporten gick inte att visa: {type(exc).__name__}")

        # Per-strategy sections — och remsans lista (varje innehavs signal, regelvaktens varningar)
        actions: list = []
        all_flat = _get_all_holdings_flat()
        for strategy_tag in STRATEGY_TAGS:
            has = any(_strategy_of(h) == strategy_tag for h in all_flat)
            # Always show Quality and Deep Contrarian; others only when they have holdings
            if strategy_tag in ("Quality", "Deep Contrarian") or has:
                _render_strategy_section(strategy_tag, actions, alloc.get(strategy_tag))
        html = strip_html(actions)
        if html:
            slot.markdown(html, unsafe_allow_html=True)

        # Risk dashboard
        st.markdown(section("Risk"), unsafe_allow_html=True)
        holdings_data = positions.by_bucket()
        try:
            from risk_dashboard import render_risk_dashboard
            render_risk_dashboard(holdings_data)
        except Exception:
            pass

        # Earnings calendar
        st.markdown(section("Rapporter"), unsafe_allow_html=True)
        try:
            from earnings_calendar import render_earnings_calendar
            render_earnings_calendar(holdings_data)
        except Exception:
            pass

        # Correlation matrix
        st.markdown(section("Korrelation"), unsafe_allow_html=True)
        _render_correlation_matrix(all_flat)
