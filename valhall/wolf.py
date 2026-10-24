"""
valhall/wolf.py — Wolf i Valhall: Arc-Wolf och momentumlistan per segment på Kartan, och Arc-Wolfs
köpsignaler till papperskontot "Wolf" (wolf_arc).

Två listor heter Wolf i panelen, och Valhall sa inte vilken den menade:

  Arc-Wolf       wolf_shadow_screener (EMA/Ichimoku, fyra lagers regimpoäng 0–125). Körs nattligt för Norden
                 av arc_scan.py → arc_screeners.json (rader med Total Score ≥ 50). Fanns inte i Valhall.
  Momentum       Swing Screener (wolf_screener.json, topp 40 på 3/6-månadersmomentum). Papperskontot "Wolf"
                 sedan fas 4a — heter nu "Momentum Swing", och syns också på Kartan (topp 20 med setup A).

Arc-Wolfs köpregel är strategins egen (strategies/wolf.py entry_fn): EMA10 > 21 > 50 > 200, kurs > EMA50,
RSI 45–70, ADX ≥ 19, räknad på aktiens dagsdata t.o.m. signaldagen. Stoppen (2,5 × ATR14) och exiten (hård
stopp, tre stängningar under EMA50, under Kijun och EMA10) sköts i valhall/paper.py med samma modul.
Delvinsterna TP1/TP2 tas inte i papperet — en enhet per signal, som för alla konton. Inga regler ändras.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from valhall import segments as sgm

ARC_BLOB = "arc_screeners.json"
ARC_STALE_DAYS = 4                 # tre körningar per vardag — en äldre lista är död, inte en signal
ARC_MIN_BARS = 210                 # entry_fn kräver EMA200
MOMENTUM_TOP = 20                  # papperets köpregel: rad 1–20 med setup A (paper.WOLF_TOP)
N_PICKS = 2
WOLF, MOMENTUM = "Wolf", "Momentum Swing"
ARC_ACCOUNT = "wolf_arc"


def arc_live(arc_blob: Optional[dict], universe: dict, today=None) -> Optional[dict]:
    """{yf-ticker: rad} för Arc-Wolfs rader i Valhalls universum — None när listan saknas, felade eller är
    äldre än ARC_STALE_DAYS (då läggs inga Wolf-köp i dag)."""
    w = (arc_blob or {}).get("wolf") or {}
    if not w.get("generated") or w.get("error"):
        return None
    try:
        age = (pd.Timestamp(today or pd.Timestamp.today()) - pd.Timestamp(str(w["generated"])[:10])).days
    except (TypeError, ValueError):
        return None
    if age > ARC_STALE_DAYS:
        return None
    out = {}
    for r in w.get("rows") or []:
        t = str(r.get("ticker") or "").upper()
        if t in universe and r.get("score") is not None:
            out[t] = r
    return out


def arc_entries(arc: Optional[dict], ohlcv: Callable, entry_fn: Optional[Callable] = None) -> dict:
    """{ticker: {signal, entry, stop, tp1, tp2, gates, gates_n}} — strategins köpregel på varje Arc-rad.
    ohlcv(ticker) → dagsdata t.o.m. signaldagen; entry_fn = strategies.wolf.entry_fn (utbytbar i tester)."""
    if not arc:
        return {}
    if entry_fn is None:
        from strategies.wolf import entry_fn
    out = {}
    for t in sorted(arc):
        try:
            df = ohlcv(t)
        except Exception:
            df = None
        if df is None or len(df) < ARC_MIN_BARS:
            continue
        try:
            e = entry_fn(df)
        except Exception:
            continue
        gates = e.get("gates") or []
        out[t] = {"signal": e.get("signal"), "entry": e.get("entry_price"), "stop": e.get("stop_loss"),
                  "tp1": e.get("tp1_price"), "tp2": e.get("tp2_price"),
                  "gates": sum(1 for g in gates if g.get("passed")), "gates_n": len(gates)}
    return out


def _name(r: dict, t: str, universe: dict) -> str:
    return str(r.get("name") or (universe.get(t) or {}).get("name") or "")


def arc_picks(arc: Optional[dict], entries: dict, universe: dict, segment: str, n: int = N_PICKS) -> list:
    """Segmentets Arc-Wolf-rader på Kartan: köpregeln uppfylld först (ENTRY), sedan poäng."""
    rows = [(t, r) for t, r in (arc or {}).items() if (universe.get(t) or {}).get("segment") == segment]
    rows.sort(key=lambda tr: ((entries.get(tr[0]) or {}).get("signal") != "BUY", -float(tr[1]["score"]), tr[0]))
    out = []
    for t, r in rows[:n]:
        e = entries.get(t) or {}
        buy = e.get("signal") == "BUY"
        score = float(r["score"])
        out.append({"ticker": t, "name": _name(r, t, universe), "segment": segment, "strategy": WOLF,
                    "score": score, "entry": buy, "price": e.get("entry") if buy else None,
                    "stop": e.get("stop") if buy else None,
                    "tp1": e.get("tp1") if buy else None, "tp2": e.get("tp2") if buy else None,
                    "gates": f"{e['gates']}/{e['gates_n']}" if e else None,
                    "tag": f"ENTRY · {score:.0f}" if buy else f"{score:.0f}"})
    return out


def momentum_picks(wolf: Optional[dict], universe: dict, segment: str, n: int = N_PICKS) -> list:
    """Segmentets momentumrader på Kartan: topp 20 med setup A, i rangordning (papperets köpregel)."""
    rows = [(t, r) for t, r in (wolf or {}).items() if (universe.get(t) or {}).get("segment") == segment
            and (r.get("rank") or 99) <= MOMENTUM_TOP and r.get("setupA")]
    rows.sort(key=lambda tr: (tr[1].get("rank") or 99, tr[0]))
    return [{"ticker": t, "name": _name(r, t, universe), "segment": segment, "strategy": MOMENTUM,
             "rank": r.get("rank"), "price": r.get("price"), "tag": f"#{r.get('rank')}"} for t, r in rows[:n]]


def by_segment(arc: Optional[dict], entries: dict, wolf: Optional[dict], universe: dict, n: int = N_PICKS) -> dict:
    """Kartans Wolf-rader: {arc_live, momentum_live, segments: {key: {arc: [...], momentum: [...]}}}."""
    return {"arc_live": arc is not None, "momentum_live": wolf is not None,
            "segments": {s.key: {"arc": arc_picks(arc, entries, universe, s.key, n),
                                 "momentum": momentum_picks(wolf, universe, s.key, n)} for s in sgm.SEGMENTS}}


def arc_signals(arc: Optional[dict], entries: dict, universe: dict) -> list:
    """Papperskontot Wolf: dagens köpsignaler — en Arc-rad vars köpregel är uppfylld (valhall.paper.signals)."""
    from valhall.paper import _sig
    out = []
    for t, r in sorted((arc or {}).items()):
        e = entries.get(t) or {}
        seg = (universe.get(t) or {}).get("segment")
        if seg and e.get("signal") == "BUY":
            out.append(_sig(ARC_ACCOUNT, t, _name(r, t, universe), seg, None, True,
                            f"poäng {float(r['score']):.0f} · {e['gates']}/{e['gates_n']} grindar"))
    return out
