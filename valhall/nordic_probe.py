"""
valhall/nordic_probe.py — Norden-proben: är hela Norden en bättre mätpunkt än OMXS30?

Fyra frågor, var och en som en ren funktion över serier (inget nätverk här — körskriptet
valhall_nordic_probe.py hämtar data och sparar resultatet i Gisten, valhall_nordic_probe.json):
  comovement      trendar Norden (likaviktat, likvida bolag) med OMXS30 och OMXSPI? Korrelation i dags- och
                  veckoavkastning, andel dagar på samma sida om SMA200, andel 20-dagarsperioder med samma riktning.
  lead_lag        vem leder? Korskorrelation av veckoavkastningar ±max_lag veckor: bästa lag och tolkning.
  crossing_lead   när OMXS30 korsar upp genom SMA200 — korsade Norden före eller efter, och hur många dagar?
  earliness       sektorernas tidighet: hur många dagar före marknadens grind (index över SMA200) blev
                  Valhall-segmentet GRÖN, och avkastningen 63/126 dagar efter den gröna vändningen.
  gate_variants   samma Viking Nine- och BERSERK-backtest med olika marknadsgrind (OMXS30 · OMXSPI · Norden),
                  jämfört i R — motorerna oförändrade, bara den injicerade marknaden skiljer.
Inga regler ändras: proben mäter, kartan och grindarna läser inte resultatet. Allt punkt för punkt utan facit,
med dagens noterade bolag i de likaviktade indexen (överlevnadsbias — jämför varianterna, inte nivåerna).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from valhall import cycle

WINDOWS = ((252, "1 år"), (756, "3 år"), (1260, "5 år"), (None, "allt"))
SMA = 200
DIRECTION_DAYS = 20
MAX_LAG_WEEKS = 8
CROSS_WINDOW_DAYS = 60
HORIZONS = (63, 126)
MIN_OVERLAP = 120


def _clean(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None or len(s) == 0:
        return None
    s = pd.Series(s).astype(float).dropna()
    if getattr(s.index, "tz", None) is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s if len(s) else None


def _pair(a, b) -> Optional[pd.DataFrame]:
    a, b = _clean(a), _clean(b)
    if a is None or b is None:
        return None
    df = pd.concat({"a": a, "b": b}, axis=1, join="inner").dropna()
    return df if len(df) >= MIN_OVERLAP else None


def _r(v, nd=2):
    return None if v is None or (isinstance(v, float) and v != v) else round(float(v), nd)


# ── Samvariation ────────────────────────────────────────────────────────────
def comovement(a, b, windows=WINDOWS) -> list:
    """[{window, days, corr_daily, corr_weekly, same_sma200_pct, same_direction_pct}] per fönster (sista n dagar)."""
    df = _pair(a, b)
    if df is None:
        return []
    out = []
    for n, label in windows:
        w = df.tail(n) if n else df
        if len(w) < MIN_OVERLAP:
            continue
        ra, rb = w["a"].pct_change(), w["b"].pct_change()
        wa, wb = w["a"].resample("W-FRI").last().pct_change(), w["b"].resample("W-FRI").last().pct_change()
        sma_a, sma_b = df["a"].rolling(SMA).mean().reindex(w.index), df["b"].rolling(SMA).mean().reindex(w.index)
        side_a, side_b = w["a"] > sma_a, w["b"] > sma_b
        valid = sma_a.notna() & sma_b.notna()
        da = np.sign(w["a"].pct_change(DIRECTION_DAYS))
        db = np.sign(w["b"].pct_change(DIRECTION_DAYS))
        dv = da.notna() & db.notna() & (da != 0) & (db != 0)
        out.append({"window": label, "days": int(len(w)),
                    "corr_daily": _r(ra.corr(rb)), "corr_weekly": _r(wa.corr(wb)),
                    "same_sma200_pct": _r((side_a == side_b)[valid].mean() * 100, 1) if valid.any() else None,
                    "same_direction_pct": _r((da == db)[dv].mean() * 100, 1) if dv.any() else None})
    return out


LEAD_MIN_GAIN = 0.03               # bästa lag måste slå lag 0 med så mycket för att kallas ledning


def lead_lag(a, b, max_lag: int = MAX_LAG_WEEKS, names: tuple = ("a", "b")) -> dict:
    """Korskorrelation av veckoavkastningar: corr(a_t, b_{t+k}) för k = −max_lag … +max_lag.
    Positivt bästa k = a leder b med k veckor (a:s vecka förklarar b:s senare vecka)."""
    na, nb = names
    empty = {"best_lag": None, "best_corr": None, "lag0_corr": None, "by_lag": [], "text": "för lite data"}
    df = _pair(a, b)
    if df is None:
        return empty
    wa = df["a"].resample("W-FRI").last().pct_change().dropna()
    wb = df["b"].resample("W-FRI").last().pct_change().dropna()
    rows = []
    for k in range(-max_lag, max_lag + 1):
        c = wa.corr(wb.shift(-k))
        if c == c:
            rows.append({"lag": k, "corr": _r(c, 3)})
    if not rows:
        return empty
    lag0 = next((r["corr"] for r in rows if r["lag"] == 0), None)
    best = max(rows, key=lambda r: r["corr"] if r["corr"] is not None else -9)
    if best["lag"] == 0 or lag0 is None or (best["corr"] or 0) - lag0 < LEAD_MIN_GAIN:
        text = (f"samtidigt — ingen vecka leder (bästa lag {best['lag']}: {best['corr']:.2f} mot {lag0:.2f} vid lag 0)"
                if lag0 is not None else "samtidigt")
    elif best["lag"] > 0:
        text = f"{na} leder {nb} med {best['lag']} vecka/veckor ({best['corr']:.2f} mot {lag0:.2f} vid lag 0)"
    else:
        text = f"{nb} leder {na} med {-best['lag']} vecka/veckor ({best['corr']:.2f} mot {lag0:.2f} vid lag 0)"
    return {"best_lag": best["lag"], "best_corr": best["corr"], "lag0_corr": lag0, "by_lag": rows, "text": text}


# ── SMA200-korsningar ───────────────────────────────────────────────────────
def sma_crossings(s, n: int = SMA) -> list:
    """[(datum, 'upp'|'ned')] — dagar då stängningen korsar sitt SMA(n)."""
    s = _clean(s)
    if s is None or len(s) <= n:
        return []
    above = (s > s.rolling(n).mean()).astype(float)
    above[s.rolling(n).mean().isna()] = np.nan
    prev = above.shift()
    out = []
    for d, cur, pre in zip(above.index, above.values, prev.values):
        if cur != cur or pre != pre or cur == pre:
            continue
        out.append((d, "upp" if cur > pre else "ned"))
    return out


def crossing_lead(a, b, direction: str = "upp", window: int = CROSS_WINDOW_DAYS) -> dict:
    """För varje korsning i b (marknadsgrinden): närmaste korsning åt samma håll i a inom ±window dagar.
    diffs < 0 = a korsade före b. {n, matched, earlier, later, same_day, median_days, mean_days, pairs}."""
    xa = [d for d, k in sma_crossings(a) if k == direction]
    xb = [d for d, k in sma_crossings(b) if k == direction]
    pairs, used = [], set()
    for d in xb:
        cands = [(abs((x - d).days), x) for x in xa if abs((x - d).days) <= window and x not in used]
        if not cands:
            pairs.append({"b": str(d.date()), "a": None, "days": None})
            continue
        _dist, x = min(cands)
        used.add(x)
        pairs.append({"b": str(d.date()), "a": str(x.date()), "days": int((x - d).days)})
    diffs = [p["days"] for p in pairs if p["days"] is not None]
    return {"n": len(xb), "matched": len(diffs), "earlier": sum(1 for v in diffs if v < 0),
            "later": sum(1 for v in diffs if v > 0), "same_day": sum(1 for v in diffs if v == 0),
            "median_days": _r(np.median(diffs), 1) if diffs else None,
            "mean_days": _r(np.mean(diffs), 1) if diffs else None, "pairs": pairs}


# ── Sektorernas tidighet ────────────────────────────────────────────────────
def green_turns(status: pd.Series) -> list:
    """Dagar då Valhall-statusen blir GRÖN från något annat (DATA SAKNAS räknas inte som 'annat')."""
    st = status.dropna()
    out, prev = [], None
    for d, v in st.items():
        if v == cycle.GRON and prev is not None and prev not in (cycle.GRON, cycle.NODATA):
            out.append(d)
        if v != cycle.NODATA:
            prev = v
    return out


def earliness(segment_close, segment_status: pd.Series, market_close, horizons=HORIZONS,
              window: int = CROSS_WINDOW_DAYS) -> dict:
    """Segmentets gröna vändningar mot marknadens gränd (SMA200-upkorsning): dagar till närmaste öppning
    (negativt = segmentet först) och segmentindexets avkastning 63/126 dagar efter den gröna vändningen."""
    c, m = _clean(segment_close), _clean(market_close)
    turns = green_turns(segment_status) if segment_status is not None else []
    gates = [d for d, k in sma_crossings(m) if k == "upp"] if m is not None else []
    rows = []
    for d in turns:
        near = min(((abs((g - d).days), g) for g in gates if abs((g - d).days) <= window), default=None)
        row = {"turn": str(d.date()), "gate": str(near[1].date()) if near else None,
               "days_vs_gate": int((d - near[1]).days) if near else None}
        if c is not None and d in c.index:
            i = int(c.index.get_loc(d))
            for h in horizons:
                row[f"ret_{h}"] = _r((float(c.iloc[i + h]) / float(c.iloc[i]) - 1) * 100, 1) if i + h < len(c) else None
        rows.append(row)
    diffs = [r["days_vs_gate"] for r in rows if r["days_vs_gate"] is not None]
    summary = {"turns": len(turns), "near_gate": len(diffs), "earlier": sum(1 for v in diffs if v < 0),
               "median_days": _r(np.median(diffs), 1) if diffs else None}
    for h in horizons:
        vals = [r[f"ret_{h}"] for r in rows if r.get(f"ret_{h}") is not None]
        summary[f"avg_ret_{h}"] = _r(np.mean(vals), 1) if vals else None
        summary[f"hit_{h}"] = _r(np.mean([v > 0 for v in vals]) * 100, 0) if vals else None
    return {"summary": summary, "rows": rows}


# ── Grindvarianterna ────────────────────────────────────────────────────────
def gate_variants(providers: dict, run_fn: Callable, metrics_fn: Callable) -> list:
    """[{variant, source, trades, win_rate, expectancy, profit_factor, max_drawdown_r, total_r, error}] —
    run_fn(nordic_provider) → backtestresultat ({trades, …}); metrics_fn(trades) → nyckeltal i R
    (viking_backtest.metrics). Ett fel i en variant stoppar inte de andra."""
    out = []
    for label, prov in providers.items():
        try:
            src = (prov() or {}).get("source") if callable(prov) else None
        except Exception:
            src = None
        try:
            res = run_fn(prov)
            m = metrics_fn(res.get("trades") or []) or {}
            pf = m.get("profit_factor")
            out.append({"variant": label, "source": src, "trades": int(m.get("trades") or 0),
                        "win_rate": m.get("win_rate"), "expectancy": m.get("expectancy"),
                        "profit_factor": None if pf is None or pf == float("inf") else pf,
                        "max_drawdown_r": m.get("max_drawdown_r"), "total_r": m.get("total_r"), "error": None})
        except Exception as exc:
            out.append({"variant": label, "source": src, "trades": 0, "error": f"{type(exc).__name__}: {exc}"})
    return out


def nordic_breadth(closes: dict, min_stocks: int = 30) -> Optional[pd.Series]:
    """Bredden (% över EMA50) över de nordiska aktierna — market_risk.stocks_breadth_pct, som räknar varje
    aktie från sin 50:e stängning (olika noteringsår), till skillnad från ETF-bredden som kräver alla."""
    import market_risk as mr
    return mr.stocks_breadth_pct({t: _clean(s) for t, s in (closes or {}).items()}, min_stocks=min_stocks)


def provider(close, breadth, source: str, breadth_source: Optional[str] = None) -> Callable:
    """En nordic_provider åt viking_backtest.run / berserk.backtest.run: () → {close, breadth, source}."""
    def _p():
        return {"close": _clean(close), "breadth": breadth, "source": source,
                "breadth_source": breadth_source or source}
    return _p


# ── Sammanfattningen ────────────────────────────────────────────────────────
def summary(payload: dict) -> list:
    """Rader i klartext ur resultatet — fakta, inga rekommendationer."""
    lines = []
    for key, row in (payload.get("comovement") or {}).items():
        allt = next((w for w in row if w["window"] == "allt"), row[-1] if row else None)
        if allt:
            lines.append(f"{key}: korrelation {allt['corr_daily']} (dag) · {allt['corr_weekly']} (vecka) · "
                         f"samma sida om SMA200 {allt['same_sma200_pct']} % av dagarna · "
                         f"samma 20-dagarsriktning {allt['same_direction_pct']} %")
    for key, ll in (payload.get("lead_lag") or {}).items():
        lines.append(f"{key}: {ll.get('text')}")
    for key, cl in (payload.get("crossing_lead") or {}).items():
        if cl.get("n"):
            lines.append(f"{key}: {cl['n']} uppkorsningar genom SMA200 i marknaden — den andra korsade "
                         f"före {cl['earlier']} gånger, efter {cl['later']}, samma dag {cl['same_day']}; "
                         f"median {cl['median_days']} dagar")
    for market, segs in (payload.get("earliness") or {}).items():
        for seg, e in (segs or {}).items():
            s = e.get("summary") or {}
            if s.get("turns"):
                lines.append(f"{seg} mot {market}: {s['turns']} gröna vändningar, {s['earlier']} av {s['near_gate']} "
                             f"före marknadens grind (median {s['median_days']} dagar) · "
                             f"63 d {s.get('avg_ret_63')} % (träff {s.get('hit_63')} %) · "
                             f"126 d {s.get('avg_ret_126')} %")
    for engine, rows in (payload.get("gates") or {}).items():
        for r in rows:
            if r.get("error"):
                lines.append(f"{engine} · {r['variant']}: FEL {r['error']}")
            else:
                lines.append(f"{engine} · {r['variant']}: {r['trades']} affärer · träff {r['win_rate']} % · "
                             f"förväntan {r['expectancy']} R · PF {r['profit_factor']} · "
                             f"max DD {r['max_drawdown_r']} R · summa {r['total_r']} R")
    return lines
