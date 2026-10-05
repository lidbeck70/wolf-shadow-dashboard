"""
berserk/paper.py — 🪓 BERSERK:s papperskonto, förvaltat automatiskt.

Rent tillstånd (JSON, i procent av startkapitalet = 100) som den schemalagda
skanningen (berserk_scan.py) stegar fram en gång per handelsdag:

  1. Förvaltning  öppna positioner räknas om FRÅN ENTRY med backtestets
                  exitregler (backtest.simulate_exit, close_at_end=False) —
                  samma källa som backtestet. Stopp intradag → STOPPAD;
                  stängningsregel igår → SÅLD på dagens öppning; stängnings-
                  regel idag → SÄLJ PÅ ÖPPNING; höjt stopp → FLYTTA STOPP.
  2. Fyllning     gårdagens KÖP-order fylls på signaldagens nästa öppning,
                  med BERSERK:s spärrar: 8 positioner, 2 per tema, 4 per
                  komplex, 6 % värme, ingen belåning.
  3. Nya order    dagens KÖP-rader blir order (KÖP PÅ ÖPPNING) — spärrade
                  redan här när platsen saknas.
  4. Kurvan       dagsvärdering (kassa + andelar × senaste stängning).

Allt räknas om ur kursdata, så två körningar samma dag ger inga nya händelser
(händelser = övergångar mot det sparade tillståndet). Valutor ignoreras som i
backtestet (avkastning i lokal valuta). Riktiga order läggs alltid manuellt.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from berserk import backtest as bt
from berserk import signals as sg
from berserk import themes as th
from berserk import universe as uv

START = 100.0
ORDER_TTL_DAYS = 7                 # en order som inte fyllts på en vecka (ingen ny kursdag) stryks
EVENTS_MAX, CLOSED_MAX, CURVE_MAX = 500, 1000, 2000

KOP_OPEN, KOPT, SALJ_OPEN, SALD, STOPPAD = "KÖP PÅ ÖPPNING", "KÖPT", "SÄLJ PÅ ÖPPNING", "SÅLD", "STOPPAD"
FLYTTA, SPARRAD, UTGANGEN = "FLYTTA STOPP", "SPÄRRAD", "UTGÅNGEN"
ICON = {KOP_OPEN: "🟢", KOPT: "✅", SALJ_OPEN: "🟠", SALD: "🔴", STOPPAD: "⛔", FLYTTA: "🔼", SPARRAD: "🚫",
        UTGANGEN: "⌛"}


def new_state(today=None) -> dict:
    return {"version": 1, "start": _d(today or pd.Timestamp.today()), "cash": START, "positions": [],
            "orders": [], "closed": [], "events": [], "curve": [], "last_run": None}


def _d(ts) -> str:
    return str(pd.Timestamp(ts).date())


def equity(state: dict) -> float:
    return state["cash"] + sum(p["units"] * p["last_close"] for p in state["positions"])


def heat(state: dict) -> float:
    """Öppen risk i % av kapitalet på initialstoppen (som viking_portfolio)."""
    eq = equity(state) or START
    return sum(p["units"] * (p["entry"] - p["init_stop"]) for p in state["positions"]) / eq * 100


# ── Kursdata ────────────────────────────────────────────────────────────────
def _arrays(ticker: str, ctx: Optional[dict]):
    """(index, o, h, lo, c, frame) eller None."""
    if not ctx or ctx.get("stock") is None:
        return None
    stock = ctx["stock"].dropna(subset=["Open", "High", "Low", "Close"])
    if len(stock) < 30:
        return None
    f = sg.frame(stock, ctx.get("driver"), is_etf=uv.kind_of(ticker) == "etf")
    o, h, lo, c = (stock[k].astype(float).values for k in ("Open", "High", "Low", "Close"))
    return stock.index, o, h, lo, c, f


def _signal_idx(idx, date: str) -> Optional[int]:
    """Signaldagens rad (sista dagen ≤ datumet)."""
    k = int(idx.searchsorted(pd.Timestamp(date), side="right")) - 1
    return k if k >= 0 else None


# ── Spärrar ─────────────────────────────────────────────────────────────────
def block_reason(state: dict, theme: str, weight_pct: float, risk_pct: float, extra: list = ()) -> Optional[str]:
    """Varför en ny position inte får plats (None = OK). extra = redan accepterade order samma dag."""
    pc = bt.portfolio_config()
    held = [(p["theme"], p["complex"]) for p in state["positions"]] + [(o["theme"], o["complex"]) for o in extra]
    if pc.max_positions and len(held) >= pc.max_positions:
        return f"MAX {pc.max_positions} POSITIONER"
    if sum(1 for t, _c in held if t == theme) >= pc.sector_cap:
        return "TEMA FULLT"
    cap = dict(pc.group_caps).get("complex")
    if cap and sum(1 for _t, c in held if c == th.complex_of(theme)) >= cap:
        return "KOMPLEX FULLT"
    if pc.max_heat_pct is not None:
        planned = sum(o.get("risk_pct") or 0 for o in extra)
        if heat(state) + planned + risk_pct > pc.max_heat_pct + 1e-9:
            return f"VÄRME ÖVER {pc.max_heat_pct:g} %"
    return None


# ── Steget ──────────────────────────────────────────────────────────────────
def step(state: Optional[dict], rows: list, frames: dict, today=None) -> tuple:
    """(nytt tillstånd, nya händelser). rows = skannerns rader, frames = scan(keep=…)."""
    today = _d(today or pd.Timestamp.today())
    state = dict(state) if state else new_state(today)
    for k in ("positions", "orders", "closed", "events", "curve"):
        state[k] = [dict(x) for x in state.get(k) or []]
    events: list = []
    seen = {(e["date"], e["kind"], e["ticker"]) for e in state["events"]}

    def ev(kind, ticker, text, **extra):
        if (today, kind, ticker) in seen:                  # omkörning samma dag — redan larmat
            return
        seen.add((today, kind, ticker))
        events.append({"date": today, "kind": kind, "ticker": ticker, "text": text, **extra})

    cache = {}

    def arr(t):
        if t not in cache:
            cache[t] = _arrays(t, frames.get(t))
        return cache[t]

    # 0. Borttagna ur universumet (t.ex. Australien 2026-10): order stryks, positioner stängs på senaste kurs
    for o in [o for o in state["orders"] if not uv.theme_of(o["ticker"])]:
        state["orders"].remove(o)
        ev(UTGANGEN, o["ticker"], "ordern struken — tickern är borttagen ur BERSERK-universumet")
    for p in [p for p in state["positions"] if not uv.theme_of(p["ticker"])]:
        _close(state, p, p["last_close"], "borttagen ur universumet", p["last_date"], None, ev, SALD)

    # 1. Förvaltning av öppna positioner
    _manage(state, list(state["positions"]), arr, ev)

    # 2. Fyll gårdagens order på öppningen
    filled = []
    for o in sorted(state["orders"], key=lambda o: (-sg.PRIORITY.get(o["setup"], 0), o["ticker"])):
        a = arr(o["ticker"])
        i = _signal_idx(a[0], o["signal_date"]) if a else None
        if a is None or i is None or i + 1 >= len(a[0]):
            if (pd.Timestamp(today) - pd.Timestamp(o["signal_date"])).days > ORDER_TTL_DAYS:
                ev(UTGANGEN, o["ticker"], "ordern fylldes inte inom en vecka (ingen ny kursdag) — struken")
                o["_drop"] = True
            continue
        o["_drop"] = True
        idx, op = a[0], a[1]
        entry = float(op[i + 1])
        atr0 = float(o["atr"])
        stop = entry - bt.STOP_ATR[o["setup"]] * atr0
        if not (entry > stop > 0):
            ev(SPARRAD, o["ticker"], "ogiltigt stopp vid fyllning")
            continue
        stop_pct = (entry - stop) / entry * 100
        pos_pct = min(o["risk_pct"] / stop_pct * 100, bt.portfolio_config().max_position_pct)
        eq = equity(state)
        reason = block_reason(state, o["theme"], pos_pct, pos_pct * stop_pct / 100)
        weight = min(eq * pos_pct / 100, state["cash"])
        if reason is None and weight < 0.5:
            reason = "INGEN KASSA"
        if reason:
            ev(SPARRAD, o["ticker"], f"fylldes inte: {reason}")
            continue
        state["cash"] -= weight
        p = {"ticker": o["ticker"], "setup": o["setup"], "theme": o["theme"], "complex": o["complex"],
             "region": o.get("region"), "signal_date": o["signal_date"], "entry_date": _d(idx[i + 1]),
             "entry": round(entry, 4), "init_stop": round(stop, 4), "cur_stop": round(stop, 4), "atr": atr0,
             "weight": round(weight, 4), "units": weight / entry, "last_close": entry, "last_date": _d(idx[i + 1]),
             "pending": None, "armed_be": False, "trailing": False}
        state["positions"].append(p)
        filled.append(p)
        ev(KOPT, p["ticker"], f"{_short(p['setup'])} fylld {p['entry_date']} @ {_px(entry)} · stopp {_px(stop)} · "
                              f"{weight / eq * 100:.1f} % av kapitalet")
    state["orders"] = [o for o in state["orders"] if not o.pop("_drop", False)]

    # Nyfyllda: kan ha stoppats eller fått en stängningsregel redan
    _manage(state, filled, arr, ev)

    # 3. Dagens KÖP blir order
    taken = {p["ticker"] for p in state["positions"]} | {o["ticker"] for o in state["orders"]}
    done = {(c["ticker"], c["signal_date"]) for c in state["closed"]}
    accepted = []
    for r in sorted((r for r in rows if r.get("status") == "KÖP" and not r.get("error") and r.get("setup")),
                    key=lambda r: (-sg.PRIORITY.get(r["setup"], 0), r["ticker"])):
        t, sd = r["ticker"], r.get("date") or today
        if t in taken or (t, sd) in done:
            continue
        reason = block_reason(state, r["theme"], r.get("position_pct") or 0, r.get("risk_pct") or 0, accepted)
        if reason:
            ev(SPARRAD, t, f"{_short(r['setup'])} idag men {reason} — ingen order")
            continue
        o = {"ticker": t, "setup": r["setup"], "theme": r["theme"], "complex": r["complex"],
             "region": r.get("region"), "signal_date": sd, "close": r.get("close"), "stop": r.get("stop"),
             "atr": r.get("atr"), "risk_pct": r.get("risk_pct") or bt.RISK_BY_SETUP.get(r["setup"], 1.0),
             "position_pct": r.get("position_pct")}
        if not o["atr"]:
            continue
        accepted.append(o)
        taken.add(t)
        ev(KOP_OPEN, t, f"{_short(r['setup'])} {r.get('label', '')} · stängde {_px(r.get('close'))} · stopp ≈ "
                        f"{_px(r.get('stop'))} · {r.get('position_pct')} % · {(r.get('why') or [''])[0]}")
    state["orders"] += accepted

    # 4. Kurvan
    eq = equity(state)
    curve = [c for c in state["curve"] if c["date"] != today]
    curve.append({"date": today, "equity": round(eq, 4), "cash": round(state["cash"], 4),
                  "open": len(state["positions"]), "heat": round(heat(state), 2)})
    state["curve"] = curve[-CURVE_MAX:]
    state["events"] = (state["events"] + events)[-EVENTS_MAX:]
    state["closed"] = state["closed"][-CLOSED_MAX:]
    state["last_run"] = today
    return state, events


def _manage(state: dict, positions: list, arr, ev) -> None:
    """Räkna om positionerna från entry; stäng, flytta stopp eller flagga sälj på öppning."""
    for p in positions:
        if p not in state["positions"]:
            continue
        a = arr(p["ticker"])
        if a is None:
            continue
        idx, o, h, lo, c, f = a
        i = _signal_idx(idx, p["signal_date"])
        if i is None or i + 1 >= len(idx):
            continue
        out = bt.simulate_exit(o, h, lo, c, f, i, p["setup"], p["entry"], p["init_stop"], p["atr"],
                               close_at_end=False)
        if out["exit_idx"] is not None:
            reason = out["reason"]
            kind = STOPPAD if "stopp" in reason and "tids" not in reason else SALD
            _close(state, p, float(out["exit"]), reason, _d(idx[out["exit_idx"]]), int(out["exit_idx"] - i), ev, kind)
            continue
        p["last_close"], p["last_date"] = float(c[-1]), _d(idx[-1])
        if out["cur_stop"] > p["cur_stop"] + 1e-9:
            ev(FLYTTA, p["ticker"], f"höj stoppet till {_px(out['cur_stop'])} (breakeven-läge)")
            p["cur_stop"] = round(float(out["cur_stop"]), 4)
        if out["pending"] and out["pending"] != p.get("pending"):
            ev(SALJ_OPEN, p["ticker"], f"{_short(p['setup'])} {out['pending']} — sälj på nästa öppning · "
                                       f"stängde {_px(c[-1])} · {(c[-1] - p['entry']) / (p['entry'] - p['init_stop']):+.2f} R")
        p["pending"], p["armed_be"], p["trailing"] = out["pending"], out["armed_be"], out["trailing"]


def _close(state: dict, p: dict, px: float, reason: str, date: str, days, ev, kind: str) -> None:
    """Stäng positionen på px: kassan, avslutade affärer och händelsen."""
    pnl = p["units"] * (px - p["entry"])
    state["cash"] += p["units"] * px
    r = (px - p["entry"]) / (p["entry"] - p["init_stop"])
    closed = {**{k: p.get(k) for k in ("ticker", "setup", "theme", "complex", "region", "signal_date",
                                       "entry_date", "entry", "init_stop", "weight")},
              "exit_date": date, "exit": round(px, 4), "reason": reason, "r": round(r, 3),
              "result_pct": round((px / p["entry"] - 1) * 100, 2), "pnl": round(pnl, 4), "days": days}
    state["closed"].append(closed)
    state["positions"].remove(p)
    ev(kind, p["ticker"], f"{_short(p['setup'])} {reason} {date} @ {_px(px)} · {r:+.2f} R · "
                          f"{closed['result_pct']:+.1f} %", r=round(r, 3))


def _short(setup: str) -> str:
    return (setup or "").split(" ", 1)[0]


def _px(v) -> str:
    return "—" if v is None else f"{float(v):,.2f}"


# ── Sammanfattning och larm ─────────────────────────────────────────────────
def summary(state: dict) -> dict:
    curve = state.get("curve") or []
    eqs = np.array([c["equity"] for c in curve], dtype=float) if curve else np.array([START])
    dd = float((eqs / np.maximum.accumulate(eqs) - 1).min() * 100) if len(eqs) else 0.0
    closed = state.get("closed") or []
    rs = [c["r"] for c in closed]
    eq = equity(state)
    return {"equity": round(eq, 2), "return_pct": round(eq / START * 100 - 100, 2), "max_dd_pct": round(dd, 2),
            "trades": len(rs), "win_rate": round(sum(1 for r in rs if r > 0) / len(rs) * 100, 1) if rs else None,
            "avg_r": round(float(np.mean(rs)), 2) if rs else None, "total_r": round(float(np.sum(rs)), 2),
            "open": len(state.get("positions") or []), "orders": len(state.get("orders") or []),
            "heat": round(heat(state), 2), "since": state.get("start"), "last_run": state.get("last_run")}


def messages(events: list, state: dict, limit: int = 1900) -> list:
    """Discordtext: en rubrik och händelserna, uppdelat under Discords gräns. Tomt om inget hänt."""
    if not events:
        return []
    s = summary(state)
    head = (f"🪓 BERSERK · {state.get('last_run')} · papper {s['equity']:.1f} ({s['return_pct']:+.1f} %) · "
            f"{s['open']} öppna · värme {s['heat']:.1f} %")
    order = [STOPPAD, SALD, SALJ_OPEN, FLYTTA, KOPT, KOP_OPEN, SPARRAD, UTGANGEN]
    lines = [f"{ICON.get(e['kind'], '•')} **{e['kind']}** {e['ticker']} — {e['text']}"
             for e in sorted(events, key=lambda e: (order.index(e["kind"]) if e["kind"] in order else 99,
                                                    e["ticker"]))]
    lines.append("_Papperskonto — riktiga order läggs manuellt._")
    out, cur = [], head
    for line in lines:
        line = line[:limit - 50]
        if len(cur) + 1 + len(line) > limit:
            out.append(cur)
            cur = "🪓 BERSERK (forts.)"
        cur += "\n" + line
    out.append(cur)
    return out
