"""
viking_backtest.py — backtest av Viking Nine (OVTLYR Nine + Viking Execution
+ exitmotorn), rapporterat i R. Utan look-ahead:

  * Varje indikator är en serie där värdet en dag bara bygger på data fram
    till och med den dagen (EMA med adjust=False, rullande fönster). Order
    blocks räknas om på kursdata t.o.m. signaldagen, bara de dagar då de kan
    avgöra signalen.
  * Signalen bedöms på signaldagens STÄNGNING (stängd candle). Entry sker på
    NÄSTA dags öppning — aldrig samma dags pris som signalen räknades på.
  * Marknads- och sektordata följer aktiens kalender med senast KÄNDA värde
    (ffill), aldrig ett senare.
  * Stoppen ligger intradag: low ≤ stopp → exit på stoppen (eller öppningen om
    dagen gappar under). Stängningsregler (SPY < EMA20, EMA10, breakeven,
    gap & crap, signal, bredd, F&G) ger exit på NÄSTA dags öppning.

Marknadsriskspärren (🌩️ Marknadsrisk) kan slås på: en signaldag där
marknadens risknivå (SPY, OMXS30 för nordiska) är spärrad ger ingen entry —
samma regel som live (market_risk_gate: OMXS30 från FÖRHÖJD, SPY vid HÖG). Risknivån är
poängen ur market_risk, som bara bygger på data t.o.m. dagen.

OVTLYR Golden Ticket (aktieversionen, inga optioner) kan testas regel för
regel — alla är av som förval, live-reglerna ändras inte (OVTLYR_RULES):
½ ATR-stopp på stängning med risken räknad på 2 × ATR, ATR-stegtrailing,
OVTLYR:s breddregler (< 25 / > 75 och EMA10), F&G vänder → exit,
likviditetsfilter och bara aktier med positiv egen historik (walk-forward).

Ingår inte (historiska data saknas): rapportspärren, max två förluster per
dag (portfölj), bearish block som exit. Universum och sektor är dagens —
överlevnads- och sektorbias. Allt står i resultatets notes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

import market_risk as mr
import market_risk_gate as mrg
import ovtlyr_nine as on
import viking_execution as vx
import viking_exit as vex

WARMUP_BARS = 60
# Hypotes att testa (CLAUDE.md: "enter after pullback"): entry bara om dagens låg någon av de
# senaste PULLBACK_DAYS dagarna nådde ner till EMA20 + PULLBACK_ATR × ATR14 — inte efter en lång rusning.
PULLBACK_DAYS = 5
PULLBACK_ATR = 0.5
ENTRY_MODES = {"Som live (stark dag)": False, "Efter pullback till EMA20": True}

# Exitregler som kan väljas (stopp och breakeven-stopp gäller alltid)
EXIT_RULES = {"market": "Marknaden < EMA20", "trail": "Trailing EMA10", "be_exit": "BE exit (under gårdagens low)",
              "gap": "Gap & crap", "signal": "Aktiens signal", "breadth": "Sektor + bredd", "fg": "F&G-target"}
ALL_EXITS = tuple(EXIT_RULES)
# Förval: (regler, EMA10 först efter breakeven). Första = systemets regel (samma som viking_exit).
EXIT_PRESETS = {
    "EMA10 först efter breakeven": (ALL_EXITS, True),
    "Alla regler": (ALL_EXITS, False),
    "Kärnan (stopp, breakeven, EMA10, marknad)": (("market", "trail", "be_exit"), False),
    "Bara stopp + EMA10": (("trail",), False),
}
# Marknadsriskspärr: nivåer som stoppar en ny entry — en tuple för alla marknader eller
# {marknad: nivåer}. Första = samma som live (market_risk_gate.VIKING_BLOCK_BY_MARKET).
RISK_GATES = {"Per marknad (som live)": dict(mrg.VIKING_BLOCK_BY_MARKET),
              "FÖRHÖJD eller HÖG": (mrg.ELEVATED, mrg.HIGH), "Bara HÖG": (mrg.HIGH,), "Av": ()}


# OVTLYR Golden Ticket — aktieregler som kan testas (av = live-regeln). Nyckel → (etikett, förklaring).
OVTLYR_RULES = {
    "ovt_stop": ("½ ATR-stopp på stängning",
                 "stängning under entry − ½ × ATR → exit nästa öppning; risk och storlek räknas på 2 × ATR "
                 "(nödstopp intradag där) i stället för 1,5 × ATR intradag"),
    "atr_step": ("ATR-stegtrailing",
                 "för varje helt ATR över entry flyttas stoppen till ½ ATR under steget (+1 ATR → +½ ATR, "
                 "+2 → +1½ …), intradag"),
    "ovt_breadth": ("Bredd enligt OVTLYR",
                    "bredden över sin EMA10 (ökar); under 25 bara efter uppvändning, över 75 och nedvänd = "
                    "inga nya affärer — ersätter '≥ 50 % och stigande'"),
    "fg_turn": ("F&G vänder → exit", "aktiens F&G lägre än för fem dagar sedan → exit nästa öppning"),
    "liquidity": ("Likviditetsfilter",
                  "USA: kurs > 20 $ och snittvolym 30 d > 1 milj aktier · Norden: snittomsättning 30 d "
                  "> 10 milj SEK/NOK, 7 milj DKK, 1 milj EUR"),
    "history": ("Positiv egen historik",
                "bara aktier vars tidigare stängda Viking-affärer summerar ≥ 0R (walk-forward, ingen "
                "historik = tillåten)"),
}
OVT_CLOSE_STOP_ATR = 0.5           # ½ ATR-stopp — på stängning
OVT_RISK_ATR = 2.0                 # risk/storlek på 2 × ATR (OVTLYR: konto × risk % / (2 × ATR)) — nödstopp intradag
ATR_STEP_GIVEBACK = 0.5            # stegtrailing: stoppen ½ ATR under senast nådda hela ATR-steg
BREADTH_LOW, BREADTH_HIGH, BREADTH_SIGNAL_EMA = 25.0, 75.0, 10
LIQ_DAYS = 30
LIQ_US_PRICE, LIQ_US_VOLUME = 20.0, 1_000_000
LIQ_TURNOVER = {".ST": 10e6, ".OL": 10e6, ".CO": 7e6, ".HE": 1e6}   # lokal valuta per dag


def gate_levels(gate, market: str) -> tuple:
    """Spärrens nivåer för en marknad (gate = tuple för alla, eller {marknad: nivåer})."""
    if isinstance(gate, dict):
        return tuple(gate.get(market, ()))
    return tuple(gate or ())
# Fasta tickerlistor för backtestet — samma lista varje gång ger rättvisa jämförelser.
NORDIC_50 = (
    "CBRAIN.CO", "DLAB.ST", "TRUE-B.ST", "YUBICO.ST", "ELON.ST", "EMBRAC-B.ST", "MIPS.ST", "OBAB.ST", "SF.ST",
    "BICO.ST", "BULTEN.ST", "CONTX.OL", "EGTX.ST", "ELTEL.ST", "FASTAT.ST", "FMM-B.ST", "HFRTO-B.ST", "HMPLY.ST",
    "HUNT.OL", "KHG.HE", "VAR.OL", "ABB.ST", "BOL.ST",
    "VOLV-B.ST", "ATCO-A.ST", "SAND.ST", "ERIC-B.ST", "HM-B.ST", "INVE-B.ST", "ASSA-B.ST", "EVO.ST", "SAAB-B.ST",
    "AZN.ST", "SEB-A.ST", "ESSITY-B.ST", "NIBE-B.ST", "SSAB-A.ST", "LIFCO-B.ST",
    "EQNR.OL", "DNB.OL", "MOWI.OL", "NHY.OL", "KOG.OL",
    "NOVO-B.CO", "DSV.CO", "VWS.CO", "PNDORA.CO",
    "NOKIA.HE", "NESTE.HE", "SAMPO.HE",
)
US_25 = (
    "AMC", "KO", "FNV", "AAPL", "MSFT", "NVDA", "AMZN", "META", "JPM", "XOM", "LLY", "UNH", "CAT", "HD", "WMT",
    "COST", "NEM", "FCX", "LMT", "AMD", "TSLA", "NFLX", "PFE", "INTC", "BA",
)
TICKER_LISTS = {"Norden 50": NORDIC_50, "USA 25": US_25, "Norden 50 + USA 25": NORDIC_50 + US_25}

NOTES = (
    "Entry på nästa dags öppning efter en stängd signaldag; stängningsregler ger exit på nästa öppning.",
    "Rapportspärren ingår inte — historiska rapportdatum saknas.",
    "Max två förluster per dag (portföljregel) ingår inte — varje ticker testas för sig.",
    "Bearish block används som entryfilter men inte som exit.",
    "Universum och sektor är dagens: aktier som försvunnit saknas (överlevnadsbias).",
    "Nordiska aktier testas mot OMXS30 och svensk Large Cap-bredd (Börsdata), övriga mot SPY.",
    "OVTLYR Nine är WOLF APPROXIMATION — panelens definitioner, inte OVTLYR:s data.",
)


@dataclass
class Config:
    min_nine: int = 9
    require_volume: bool = vx.VOLUME_REQUIRED
    max_chase_pct: float = vx.MAX_CHASE_PCT
    min_rr: float = vx.MINIMUM_RR
    atr_mult: float = vx.ATR_STOP_MULT
    years: int = 3
    exit_rules: tuple = ALL_EXITS
    trail_after_be: bool = True        # EMA10 gäller först när stoppen flyttats till breakeven (som viking_exit)
    pullback: bool = False             # True = kräv pullback till EMA20 (ENTRY_MODES) — inte live-regeln
    risk_gate: object = field(default_factory=lambda: dict(mrg.VIKING_BLOCK_BY_MARKET))   # se RISK_GATES
    # OVTLYR Golden Ticket (OVTLYR_RULES) — av = live-regeln
    ovt_stop: bool = False
    atr_step: bool = False
    ovt_breadth: bool = False
    fg_turn: bool = False
    liquidity: bool = False
    history: bool = False

    def ovtlyr_rules(self) -> list:
        """Påslagna OVTLYR-regler (nycklar i OVTLYR_RULES)."""
        return [k for k in OVTLYR_RULES if getattr(self, k, False)]


@dataclass
class Trade:
    ticker: str
    signal_date: str
    entry_date: str
    entry: float
    stop: float
    risk: float
    nine: int
    exit_date: Optional[str] = None
    exit: Optional[float] = None
    exit_reason: str = ""
    r: Optional[float] = None
    days: int = 0
    open: bool = False
    mom63: Optional[float] = None       # 63-dagarsavkastning på signaldagen (prioritet i portföljläget)
    sector: Optional[str] = None        # sektor-ETF (en aktie per sektor i portföljläget)
    hist_r: Optional[float] = None      # summa R för aktiens tidigare stängda affärer vid signalen (walk-forward)


# ── Serierna (allt kausalt) ─────────────────────────────────────────────────
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _trend(c: pd.Series) -> pd.Series:
    return (_ema(c, 10) > _ema(c, 20)) & (c > _ema(c, 50))


def _signal(c: pd.Series) -> pd.Series:
    return c >= _ema(c, 20)


def _align(s: Optional[pd.Series], idx) -> pd.Series:
    """Senast KÄNDA värde på aktiens dagar — aldrig ett senare."""
    if s is None or len(s) == 0:
        return pd.Series(False, index=idx)
    return s.astype(float).reindex(s.index.union(idx)).ffill().reindex(idx).fillna(0).astype(bool)


def ovtlyr_breadth_ok(b: pd.Series) -> pd.Series:
    """OVTLYR:s breddregler per dag (kausalt): bredden över sin EMA10 (ökar) — under 25 bara
    efter en uppvändning, över 75 och nedvänd = inga nya affärer."""
    b = b.astype(float)
    above = b > _ema(b, BREADTH_SIGNAL_EMA)
    up, down = b > b.shift(1), b < b.shift(1)
    return above & ((b >= BREADTH_LOW) | up) & ~((b > BREADTH_HIGH) & down) & b.notna()


def liquid_series(stock: pd.DataFrame, ticker: str) -> pd.Series:
    """Likviditetsfiltret per dag (snitt över LIQ_DAYS dagar t.o.m. dagen)."""
    c, v = stock["Close"].astype(float), stock["Volume"].astype(float)
    t = str(ticker or "").upper()
    suffix = next((s for s in LIQ_TURNOVER if t.endswith(s)), None)
    if suffix is None:
        return (c > LIQ_US_PRICE) & (v.rolling(LIQ_DAYS).mean() > LIQ_US_VOLUME)
    return (c * v).rolling(LIQ_DAYS).mean() > LIQ_TURNOVER[suffix]


def factor_frame(stock: pd.DataFrame, spy: Optional[pd.DataFrame], sector: Optional[pd.DataFrame],
                 breadth: Optional[pd.Series], ovt_breadth: bool = False) -> pd.DataFrame:
    """OVTLYR Nine:s åtta prisfaktorer per dag (order blocks räknas separat).
    ovt_breadth = marknadsbredden enligt OVTLYR (ovtlyr_breadth_ok) i stället för panelens regel."""
    idx = stock.index
    c = stock["Close"].astype(float)
    out = pd.DataFrame(index=idx)
    if spy is not None and len(spy):
        sc = spy["Close"].astype(float)
        out["market.trend"] = _align(_trend(sc), idx)
        out["market.signal"] = _align(_signal(sc), idx)
    else:
        out["market.trend"] = out["market.signal"] = False
    if breadth is not None and len(breadth):
        ok = (ovtlyr_breadth_ok(breadth) if ovt_breadth
              else (breadth >= on.BREADTH_MIN_PCT) & (breadth >= _ema(breadth, on.BREADTH_EMA)))
        out["market.breadth"] = _align(ok, idx)
    else:
        out["market.breadth"] = False
    if sector is not None and len(sector) >= on.MIN_BARS:
        fg = on.fear_greed_series(sector)
        out["sector.fear_greed"] = _align((fg >= fg.shift(on.FG_LOOKBACK)) & (fg < on.FG_MAX), idx)
        e20 = _ema(sector["Close"].astype(float), 20)
        out["sector.breadth"] = _align((sector["Close"].astype(float) > e20)
                                       & (e20 > e20.shift(on.SECTOR_EMA_RISE_DAYS)), idx)
    else:
        out["sector.fear_greed"] = out["sector.breadth"] = False
    fg = on.fear_greed_series(stock)
    out["stock.trend"] = _trend(c)
    out["stock.signal"] = _signal(c)
    out["stock.fear_greed"] = (fg >= fg.shift(on.FG_LOOKBACK)) & (fg < on.FG_MAX)
    out["fg"] = fg
    return out


def execution_frame(stock: pd.DataFrame) -> pd.DataFrame:
    c, o, lo = (stock[k].astype(float) for k in ("Close", "Open", "Low"))
    r = vx.rsi(c)
    out = pd.DataFrame(index=stock.index)
    out["momentum"] = (r > vx.RSI_MIN) & (r > r.shift(1)) & (c > lo.shift(1))
    out["trend"] = (c > _ema(c, 10)) & (_ema(c, 10) > _ema(c, 20)) & (_ema(c, 20) > _ema(c, 50))
    out["candle"] = (c > o) & (c >= c.shift(1))
    v = stock["Volume"].astype(float)
    out["volume"] = v / v.shift(1).rolling(20).mean() >= vx.RELATIVE_VOLUME_MIN
    out["atr"] = vx.atr(stock)
    out["ema10"], out["ema20"] = _ema(c, 10), _ema(c, 20)
    touch = lo <= out["ema20"] + PULLBACK_ATR * out["atr"]
    out["pullback"] = touch.astype(float).rolling(PULLBACK_DAYS, min_periods=1).max() > 0
    return out


def risk_levels(points: Optional[pd.Series], idx) -> pd.Series:
    """Marknadsriskens nivå (LÅG/FÖRHÖJD/HÖG) på aktiens dagar — senast KÄNDA poäng,
    tom sträng före första kända dagen."""
    if points is None or len(points) == 0:
        return pd.Series("", index=idx)
    p = points.astype(float)
    if getattr(p.index, "tz", None) is not None:
        p.index = p.index.tz_localize(None)
    p = p.reindex(p.index.union(idx)).ffill().reindex(idx)
    return p.map(lambda v: "" if pd.isna(v) else mr.level_of(int(v)))


def _blocks_at(stock: pd.DataFrame, i: int) -> tuple:
    """(fritt?, ob_analysis) på kursdata t.o.m. dag i."""
    try:
        from ovtlyr.indicators.orderblocks import classify_price_vs_ob, detect_orderblocks
        part = stock.iloc[:i + 1]
        obs = detect_orderblocks(part)
        oa = classify_price_vs_ob(float(part["Close"].iloc[-1]), obs) if obs else {"signal_bias": "HOLD"}
    except Exception:
        oa = {"signal_bias": "HOLD"}
    bias = str(oa.get("signal_bias", "HOLD")).upper()
    return bias not in ("SELL", "REDUCE") and not oa.get("approaching_bearish", False), oa


# ── En ticker ───────────────────────────────────────────────────────────────
def backtest_ticker(ticker: str, stock: pd.DataFrame, spy: Optional[pd.DataFrame], sector: Optional[pd.DataFrame],
                    breadth: Optional[pd.Series], cfg: Config = Config(), start=None,
                    market_label: str = "SPY", risk: Optional[pd.Series] = None) -> dict:
    """spy = marknadens index (SPY, eller OMXS30 för nordiska aktier); breadth = marknadens bredd i %;
    risk = marknadsriskens poäng per dag (market_risk) — None = ingen spärr."""
    stock = stock.dropna(subset=["Open", "High", "Low", "Close"])
    n = len(stock)
    res = {"ticker": ticker, "trades": [], "signals": 0, "no_chase": 0, "low_rr": 0, "risk_blocked": 0,
           "no_pullback": 0, "illiquid": 0}
    if n < WARMUP_BARS + 2:
        return res
    f, x = factor_frame(stock, spy, sector, breadth, ovt_breadth=cfg.ovt_breadth), execution_frame(stock)
    liquid = liquid_series(stock, ticker) if cfg.liquidity else None
    eight = [k for k in f.columns if "." in k]
    count8 = f[eight].sum(axis=1)
    o, h, lo, c = (stock[k].astype(float).values for k in ("Open", "High", "Low", "Close"))
    idx = stock.index
    blocked_levels = gate_levels(cfg.risk_gate, mrg.market_for(ticker))
    levels = risk_levels(risk, idx) if blocked_levels else pd.Series("", index=idx)
    # Robust start-position lookup: np.searchsorted on a DatetimeIndex crashes in
    # pandas>=2 when the index resolution (s/ms/us) differs from the Timestamp's
    # (ns) — _unbox_scalar uses round_ok=False. A boolean comparison converts
    # units/tz gracefully and is equivalent to searchsorted(side="left").
    start_pos = 0
    if start is not None:
        _ts = pd.Timestamp(start)
        if getattr(idx, "tz", None) is not None and _ts.tz is None:
            _ts = _ts.tz_localize(idx.tz)
        elif getattr(idx, "tz", None) is None and _ts.tz is not None:
            _ts = _ts.tz_localize(None)
        start_pos = int((idx < _ts).sum())
    first = max(WARMUP_BARS, start_pos)
    i = first
    while i < n - 1:
        if count8.iloc[i] + 1 < cfg.min_nine or not f["market.signal"].iloc[i]:
            i += 1
            continue
        ex = x.iloc[i]
        if not (ex["momentum"] and ex["trend"] and ex["candle"] and (ex["volume"] or not cfg.require_volume)):
            i += 1
            continue
        free, oa = _blocks_at(stock, i)
        nine = int(count8.iloc[i]) + int(free)
        if nine < cfg.min_nine:
            i += 1
            continue
        res["signals"] += 1
        if liquid is not None and not bool(liquid.iloc[i]):              # OVTLYR: för låg likviditet
            res["illiquid"] += 1
            i += 1
            continue
        if cfg.pullback and not ex["pullback"]:                    # ingen pullback till EMA20 nyligen
            res["no_pullback"] += 1
            i += 1
            continue
        if levels.iloc[i] in blocked_levels:                        # marknadsrisken spärrar nya entries
            res["risk_blocked"] += 1
            i += 1
            continue
        stop_dist = cfg.atr_mult * float(ex["atr"])
        resist, _src = vx.nearest_resistance(stock.iloc[:i + 1], float(c[i]), oa)
        if resist is not None and stop_dist > 0 and (resist - c[i]) / stop_dist < cfg.min_rr:
            res["low_rr"] += 1
            i += 1
            continue
        entry = float(o[i + 1])                                     # nästa dags öppning
        if entry > c[i] * (1 + cfg.max_chase_pct / 100):
            res["no_chase"] += 1
            i += 1
            continue
        atr0 = float(ex["atr"])
        # OVTLYR: risken (R och storlek) på 2 × ATR med nödstopp där — själva stoppen är ½ ATR på stängning.
        # R/R-filtret ovan använder live-avståndet, så entryerna blir desamma och bara exiten skiljer.
        stop = entry - (OVT_RISK_ATR * atr0 if cfg.ovt_stop else stop_dist)
        risk = entry - stop
        if not (risk > 0):
            i += 1
            continue
        t = Trade(ticker, str(idx[i].date()), str(idx[i + 1].date()), round(entry, 4), round(stop, 4),
                  round(risk, 4), nine, mom63=round(float(c[i] / c[i - 63] - 1), 4) if i >= 63 else None)
        fg_target = vex.fg_target(float(f["fg"].iloc[i])) if pd.notna(f["fg"].iloc[i]) else None
        pre_high = float(np.max(h[max(0, i + 1 - vex.BE_LOOKBACK):i + 2]))
        armed = False
        step_stop = -math.inf                                       # ATR-stegtrailing (från tidigare dagars high)
        run_high = -math.inf
        j = i + 1
        pending = None                                              # stängningsregel → exit nästa öppning
        while j < n:
            cur_stop = max(stop, entry) if armed else stop
            cur_stop = max(cur_stop, step_stop)
            if pending is not None:
                t.exit, t.exit_reason, t.exit_date = float(o[j]), pending, str(idx[j].date())
                break
            if lo[j] <= cur_stop:
                px = float(o[j]) if o[j] < cur_stop else cur_stop
                t.exit, t.exit_date = px, str(idx[j].date())
                t.exit_reason = ("ATR-steg" if step_stop >= cur_stop and step_stop > entry
                                 else "breakeven-stopp" if armed and cur_stop >= entry
                                 else "nödstopp 2 ATR" if cfg.ovt_stop else "stopp")
                break
            reasons, rules = [], cfg.exit_rules
            if cfg.ovt_stop and c[j] < entry - OVT_CLOSE_STOP_ATR * atr0:
                reasons.append("½ ATR-stopp")
            if "market" in rules and not f["market.signal"].iloc[j]:
                reasons.append(f"{market_label} < EMA20")
            if "trail" in rules and (armed or not cfg.trail_after_be) and c[j] < x["ema10"].iloc[j]:
                reasons.append("trailing EMA10")
            if "be_exit" in rules and armed and j >= 1 and c[j] < lo[j - 1]:
                reasons.append("BE exit")
            if "gap" in rules and j >= 1 and o[j] > h[j - 1] and c[j] < c[j - 1]:
                reasons.append("gap & crap")
            if "signal" in rules and not f["stock.signal"].iloc[j]:
                reasons.append("stock signal")
            if "breadth" in rules and not f["sector.breadth"].iloc[j] and not f["market.breadth"].iloc[j]:
                reasons.append("sektor + bredd")
            if ("fg" in rules and fg_target is not None and pd.notna(f["fg"].iloc[j])
                    and f["fg"].iloc[j] >= fg_target):
                reasons.append("F&G-target")
            if (cfg.fg_turn and j >= on.FG_LOOKBACK and pd.notna(f["fg"].iloc[j])
                    and pd.notna(f["fg"].iloc[j - on.FG_LOOKBACK]) and f["fg"].iloc[j] < f["fg"].iloc[j - on.FG_LOOKBACK]):
                reasons.append("F&G vänder")
            if h[j] > pre_high:
                armed = True
            if cfg.atr_step and atr0 > 0:
                run_high = max(run_high, float(h[j]))
                k = math.floor((run_high - entry) / atr0)
                if k >= 1:
                    step_stop = max(step_stop, entry + (k - ATR_STEP_GIVEBACK) * atr0)
            if reasons:
                pending = reasons[0]
                if j == n - 1:                                      # ingen nästa dag — stäng på stängningen
                    t.exit, t.exit_reason, t.exit_date = float(c[j]), pending, str(idx[j].date())
                    break
            j += 1
        if t.exit is None:
            t.exit, t.exit_reason, t.exit_date, t.open = float(c[-1]), "öppen", str(idx[-1].date()), True
        t.r = round((t.exit - entry) / risk, 3)
        t.days = int(np.busday_count(pd.Timestamp(t.entry_date).date(), pd.Timestamp(t.exit_date).date()))
        res["trades"].append(t)
        i = max(j, i + 1)                                           # en position åt gången per ticker
    return res


# ── Nyckeltal i R ───────────────────────────────────────────────────────────
def metrics(trades: list) -> dict:
    closed = [t for t in trades if not t.open and t.r is not None]
    rs = [t.r for t in closed]
    if not rs:
        return {"trades": 0}
    wins, losses = [r for r in rs if r > 0], [r for r in rs if r <= 0]
    wr = len(wins) / len(rs)
    avg_w = float(np.mean(wins)) if wins else 0.0
    avg_l = float(np.mean(losses)) if losses else 0.0
    order = sorted(closed, key=lambda t: (t.exit_date, t.ticker))
    curve = np.cumsum([t.r for t in order])
    peak = np.maximum.accumulate(np.concatenate([[0.0], curve]))[1:]
    max_dd = float(np.max(peak - curve)) if len(curve) else 0.0
    streak = best = 0
    for t in order:
        streak = streak + 1 if t.r <= 0 else 0
        best = max(best, streak)
    gross_w, gross_l = sum(wins), -sum(losses)
    return {
        "trades": len(rs), "win_rate": round(wr * 100, 1), "avg_r": round(float(np.mean(rs)), 3),
        "median_r": round(float(np.median(rs)), 3),
        "profit_factor": round(gross_w / gross_l, 2) if gross_l > 0 else (math.inf if gross_w > 0 else None),
        "max_drawdown_r": round(max_dd, 2), "avg_winner": round(avg_w, 3), "avg_loser": round(avg_l, 3),
        "expectancy": round(wr * avg_w - (1 - wr) * abs(avg_l), 3), "max_consecutive_losses": best,
        "avg_holding_days": round(float(np.mean([t.days for t in closed])), 1), "total_r": round(float(sum(rs)), 2),
        "curve": [(t.exit_date, round(float(v), 3)) for t, v in zip(order, curve)],
    }


# ── Flera tickers ───────────────────────────────────────────────────────────
def run(tickers: list, getter: Optional[Callable] = None, sector_getter: Optional[Callable] = None,
        cfg: Config = Config(), progress: Optional[Callable] = None, today=None,
        nordic_provider: Optional[Callable] = None, risk_provider: Optional[Callable] = None) -> dict:
    """Nordiska tickers (.ST .OL .CO .HE) testas mot OMXS30 och svensk Large Cap-bredd,
    övriga mot SPY och sektor-ETF-bredden — samma regel som i Viking Nine.
    risk_provider(marknad) → marknadsriskens poäng per dag ("SPY"/"OMXS30"); None = ingen spärr."""
    if getter is None:
        from market_prices import ohlcv as getter
    period = f"{int(cfg.years) + 1}y"                               # ett extra år för uppvärmning

    def _get(t):
        try:
            df = getter(t, period)
        except Exception:
            return None
        if df is None or len(df) == 0:
            return None
        if getattr(df.index, "tz", None) is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        return df

    spy = _get(on.MARKET_TICKER)
    etfs = {t: _get(t) for t in on.SECTOR_ETFS.values()}
    breadth = on.breadth_series({t: on._close(d) for t, d in etfs.items()})
    end = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    start = end - pd.DateOffset(years=int(cfg.years))
    nordic = None
    if any(on.market_for(t) == on.NORDIC_LABEL for t in tickers):
        bars = (int(cfg.years) + 1) * 262
        nordic = (nordic_provider or (lambda: on.nordic_market(bars)))()
    risk, risk_info = {}, {}
    if cfg.risk_gate:
        for m in sorted({mrg.market_for(t) for t in tickers}):
            if not gate_levels(cfg.risk_gate, m):
                continue
            try:
                pts = risk_provider(m) if risk_provider is not None else None
            except Exception:
                pts = None
            risk[m] = pts if pts is not None and len(pts) else None
            if risk[m] is None:
                risk_info[m] = {"status": "DATA UNAVAILABLE — ingen spärr", "blocked_pct": None}
            else:
                lv = risk_levels(risk[m], risk[m].index[risk[m].index >= start])
                risk_info[m] = {"status": "ok", "blocked_pct": round(float(lv.isin(gate_levels(cfg.risk_gate, m))
                                                                         .mean()) * 100, 1)
                                if len(lv) else None}
    per, trades = [], []
    for k, t in enumerate(tickers):
        df = _get(t)
        if df is None:
            per.append({"ticker": t, "trades": [], "signals": 0, "no_chase": 0, "low_rr": 0, "risk_blocked": 0,
                        "error": "DATA UNAVAILABLE"})
        else:
            etf, _src = on.resolve_sector(t, sector_getter)
            m_df, m_breadth, label = spy, breadth, "SPY"
            if on.market_for(t) == on.NORDIC_LABEL and nordic and nordic.get("close") is not None:
                m_df, label = pd.DataFrame({"Close": nordic["close"]}), on.NORDIC_LABEL
                m_breadth = nordic.get("breadth")
            # Med historikfiltret körs även uppvärmningsåret, så att aktien har en egen historik vid periodstart.
            r = backtest_ticker(t, df, m_df, etfs.get(etf) if etf else None, m_breadth, cfg,
                                start=None if cfg.history else start, market_label=label,
                                risk=risk.get(mrg.market_for(t)))
            apply_history(r, start, cfg.history)
            for tr in r["trades"]:
                tr.sector = etf
            r["sector_etf"], r["market"] = etf, label
            per.append(r)
            trades += r["trades"]
        if progress is not None:
            progress(k + 1, len(tickers), t)
    return {"trades": trades, "per_ticker": per, "metrics": metrics(trades), "notes": NOTES, "config": cfg,
            "risk": risk_info, "risk_blocked": sum(p.get("risk_blocked", 0) for p in per),
            "no_pullback": sum(p.get("no_pullback", 0) for p in per),
            "illiquid": sum(p.get("illiquid", 0) for p in per),
            "neg_history": sum(p.get("neg_history", 0) for p in per)}


def apply_history(res: dict, start=None, require_positive: bool = False) -> dict:
    """Sätter Trade.hist_r = summa R för aktiens affärer som stängts FÖRE signaldagen (walk-forward).
    require_positive: affärer med hist_r < 0 tas bort (räknas i neg_history), liksom affärer med
    signal före start (de var bara historik). Borttagna affärer räknas ändå in i senare historik
    — som att följa signalen på papper."""
    trades = sorted(res.get("trades") or [], key=lambda t: t.signal_date)
    s0 = str(pd.Timestamp(start).date()) if start is not None else None
    kept, neg = [], 0
    for t in trades:
        prior = [p.r for p in trades if not p.open and p.r is not None and p.exit_date < t.signal_date]
        t.hist_r = round(float(sum(prior)), 3) if prior else None
        if require_positive and s0 is not None and t.signal_date < s0:
            continue
        if require_positive and t.hist_r is not None and t.hist_r < 0:
            neg += 1
            continue
        kept.append(t)
    res["trades"], res["neg_history"] = kept, neg
    return res
