"""
valhall/paper.py — fas 4: Valhalls papperskonton per strategi, varje affär märkt med segment och cykelläge.

Målet är att hitta den bästa strategin per segment och cykel (fas 3a visade att cykelstatusen inte duger
som spärr för swing — nu får papperet visa vad som fungerar). Ett konto per strategi, en "enhet" per
signal (ingen kassa eller platsbrist som skymmer jämförelsen). Varje köp märks PÅ SIGNALDAGEN med
Valhalls läge: segmentets status, fas och cykelläge i Norden och globalt och råvarutemats.

Strategi         Köp (signalen ur dagens Valhall-körning)            Sälj (befintliga regler, oförändrade)
Viking Nine      GOLDEN TICKET i segment med Viking Nine (swing.py)    viking_exit.evaluate_exit → nästa öppning
Insider Norden   status KÖP utan passa-flagga (insider.buy_candidates) stopp klustersnitt × 0,85 · 18 mån
Wolf             topp 20 + setup A i wolf_screener.json                 ur topp 40 → nästa öppning
Quality          segmentets kvalitetsköp med Alpha Regime BUY           tidsstopp 3 år
Rick Rule        råvarans kvalitetsköp (Köpkandidat / Rule-håven / BUY) tidsstopp 3 år
Durrett          guld/silver: 15-BAGGER, eller nordiskt bolag           tidsstopp 3 år
Tiggre           övriga råvaror: alla köpgrindar gröna                  tidsstopp 3 år
Long Screener    segmentets två bästa STRONG BUY/BUY (long_screener.py) screenerns SELL → nästa öppning · 3 år
BERSERK          eget konto (berserk_paper.json) — bara märkt här

Långsiktiga positioner får dessutom en SKUGGEXIT: första dagen segmentets (temats) status är RÖD noteras
priset — resultaten visar både "behåll" och "sälj på RÖD" (CLAUDE.md: minska vid röd regim — prövas här).
Avkastningen jämförs med segmentets riktmärke (nordiska sektorindexet, SPDR-fonden eller temats drivare).

Köp fylls på nästa öppning efter signaldagen. Valutor ignoreras (avkastning i lokal valuta). Riktiga
order läggs alltid manuellt.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from berserk import universe as buv
from valhall import cycle
from valhall import segments as sgm
from valhall import swing as vsw

SWING, LONG = "swing", "lång"
STRATEGIES = {"viking": ("Viking Nine", SWING), "insider": ("Insider Norden", SWING), "wolf": ("Wolf", SWING),
              "berserk": ("BERSERK", SWING), "quality": ("Quality Contrarian", LONG), "rule": ("Rick Rule", LONG),
              "durrett": ("Durrett", LONG), "tiggre": ("Tiggre", LONG), "long": ("Long Screener", LONG)}
LONG_SELL = ("SELL", "STRONG SELL")      # cagr_scoring: EMA200-brott, röd regim eller poäng under 35 %
LONG_DAYS = 3 * 365                # horisont 0,5–3 år
INSIDER_DAYS = 548                 # insiderns tidsstopp: 18 månader
COOLDOWN_DAYS = 14                 # ingen ny affär i samma aktie och strategi två veckor efter en stängning
ORDER_TTL_DAYS = 7
WOLF_TOP, WOLF_STALE_DAYS = 20, 4
EVENTS_MAX, CLOSED_MAX = 1000, 3000
VIKING_WINDOW = 300                # dagar som viking_exit får se (EMA10, ATR14, F&G)

KOP_OPEN, KOPT, SALJ_OPEN, SALD, STOPPAD = "KÖP PÅ ÖPPNING", "KÖPT", "SÄLJ PÅ ÖPPNING", "SÅLD", "STOPPAD"
SPARRAD, UTGANGEN, ROD = "SPÄRRAD", "UTGÅNGEN", "RÖD (skuggexit)"


def new_state(today=None) -> dict:
    return {"version": 1, "start": _d(today or pd.Timestamp.today()), "orders": [], "positions": [], "closed": [],
            "events": [], "berserk_tags": {}, "last_run": None}


def _d(ts) -> str:
    return str(pd.Timestamp(ts).date())


# ── Cykelläget ──────────────────────────────────────────────────────────────
def cycle_tag(board: dict, segment: Optional[str], theme: Optional[str] = None) -> dict:
    """Valhalls läge för segmentet (och temat) i dagens karta: status, fas, cykel och riktmärkets kurs."""
    row = next((r for r in (board or {}).get("segments") or [] if r.get("key") == segment), {})
    th = next((t.get("state") or {} for t in row.get("themes") or [] if t.get("theme") == theme), {}) if theme else {}
    out = {}
    for p, st in (("n", row.get("nordic") or {}), ("g", row.get("global") or {}), ("t", th)):
        out.update({f"{p}_status": st.get("status", cycle.NODATA), f"{p}_phase": st.get("phase"),
                    f"{p}_cykel": st.get("cykel"), f"{p}_close": st.get("close")})
    return out


def _gov(x: dict) -> str:
    """Vilket läge som styr affären: temats för råvaror, annars Nordens för nordiska bolag, annars globalt."""
    return "t" if x.get("theme") else "n" if x.get("nordic") else "g"


# ── Signalerna ur dagens körning ────────────────────────────────────────────
def _sig(strategy, ticker, name, segment, theme=None, nordic=None, why="", stop=None) -> dict:
    nordic = buv.region_of(ticker or "") == "Norden" if nordic is None else bool(nordic)
    return {"strategy": strategy, "ticker": ticker, "name": name or "", "segment": segment, "theme": theme,
            "nordic": nordic, "why": why, "stop": stop}


def buy_ok(p: dict) -> bool:
    """Kortets köp som papperet tar — varje källas egen köpregel."""
    src = p.get("source")
    if src == "Rick Rule-arket":
        return p.get("verdict") == "Köpkandidat"
    if src == "Quality Contrarian":
        return p.get("verdict") == "BUY"
    if src == "Durrett":
        return bool(p.get("bagger")) or buv.region_of(p.get("ticker") or "") == "Norden"
    if src == "Tiggre":
        return bool(p.get("buy"))
    return True                                                    # Rule-håven: håven är själva regeln


def wolf_live(wolf_blob: Optional[dict], wolf_map: dict, today) -> Optional[dict]:
    """{yf-ticker: rad} för Wolfs topp 40 — None när listan saknas eller är äldre än WOLF_STALE_DAYS."""
    if not wolf_blob or not wolf_blob.get("generated"):
        return None
    try:
        age = (pd.Timestamp(today) - pd.Timestamp(str(wolf_blob["generated"])[:10])).days
    except (TypeError, ValueError):
        return None
    if age > WOLF_STALE_DAYS:
        return None
    out = {}
    for r in wolf_blob.get("top") or []:
        t = wolf_map.get(str(r.get("ticker") or "").upper())
        if t:
            out[t] = r
    return out


def signals(swing: dict, buys: dict, universe: dict, insider_blob: Optional[dict] = None,
            wolf: Optional[dict] = None, long_scan: Optional[dict] = None) -> list:
    """Dagens köpsignaler per strategi. wolf = wolf_live(...), long_scan = long_screener.scan(...)."""
    out = []
    for key, e in ((long_scan or {}).get("segments") or {}).items():
        for p in e.get("picks") or []:
            out.append(_sig("long", p["ticker"], p.get("name"), key, None, True,
                            f"{p['signal']} · {p['score_pct'] * 100:.0f} % · cykel {p['cycle']}/3"))
    for key, e in ((swing or {}).get("segments") or {}).items():
        if e.get("strategy") in (vsw.VIKING, vsw.TEST):
            for p in e.get("picks") or []:
                if p.get("category") == "GOLDEN TICKET":
                    out.append(_sig("viking", p["ticker"], p.get("name"), key, None, True,
                                    f"{p.get('ovtlyr_nine')} · {e['strategy']}"))
    if insider_blob:
        import insider as ins
        for r in ins.buy_candidates([ins.auto_to_signal(c) for c in insider_blob.get("clusters") or []]):
            s = r["signal"]
            seg = (universe.get(s.get("ticker")) or {}).get("segment")
            if seg:
                out.append(_sig("insider", s["ticker"], s.get("name"), seg, None, True, f"poäng {r['score']}",
                                stop=r["stop"]))
    for t, r in (wolf or {}).items():
        seg = (universe.get(t) or {}).get("segment")
        if seg and (r.get("rank") or 99) <= WOLF_TOP and r.get("setupA"):
            out.append(_sig("wolf", t, r.get("name"), seg, None, True, f"rank {r.get('rank')} · setup A"))
    for key, e in ((buys or {}).get("segments") or {}).items():
        for p in e.get("picks") or []:
            if buy_ok(p):
                out.append(_sig("quality", p["ticker"], p.get("name"), key, None, None, f"{p.get('verdict')}"))
    for theme, e in ((buys or {}).get("themes") or {}).items():
        seg = sgm.segment_of_theme(theme)
        for p in (e.get("quality") or {}).get("picks") or []:
            if buy_ok(p):
                out.append(_sig("rule", p["ticker"], p.get("name"), seg, theme, None, p.get("source", "")))
        lv = e.get("leverage") or {}
        strat = "durrett" if lv.get("strategy") == "Durrett" else "tiggre"
        for p in lv.get("picks") or []:
            if buy_ok(p):
                out.append(_sig(strat, p["ticker"], p.get("name"), seg, theme, None,
                                "15-BAGGER" if p.get("bagger") else p.get("source", "")))
    seen, uniq = set(), []
    for s in out:
        if s["ticker"] and (s["strategy"], s["ticker"]) not in seen:
            seen.add((s["strategy"], s["ticker"]))
            uniq.append(s)
    return uniq


# ── Steget ──────────────────────────────────────────────────────────────────
def _clean(df) -> Optional[pd.DataFrame]:
    if df is None or len(df) == 0 or not {"Open", "High", "Low", "Close"} <= set(df.columns):
        return None
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df if len(df) else None


def _after(df: pd.DataFrame, date: str) -> int:
    """Första raden efter datumet (len(df) när den saknas)."""
    return int(df.index.searchsorted(pd.Timestamp(date), side="right"))


def step(state: Optional[dict], sigs: list, board: dict, ohlcv: Callable, market: Optional[pd.DataFrame] = None,
         today=None, wolf: Optional[dict] = None, berserk_state: Optional[dict] = None,
         universe: Optional[dict] = None, long_signals: Optional[dict] = None) -> tuple:
    """(nytt tillstånd, nya händelser). ohlcv(ticker) → dagsdata, market = OMXS30 (Viking Nines marknadsexit),
    wolf = wolf_live(...) (None = listan saknas → inga Wolf-sälj i dag), berserk_state = BERSERK:s konto,
    long_signals = {ticker: Long Screener-signal} (None = skanningen saknas → inga sälj i dag)."""
    today = _d(today or pd.Timestamp.today())
    state = dict(state) if state else new_state(today)
    for k in ("orders", "positions", "closed", "events"):
        state[k] = [dict(x) for x in state.get(k) or []]
    state["berserk_tags"] = dict(state.get("berserk_tags") or {})
    events: list = []
    cache: dict = {}

    def ev(kind, x, text):
        events.append({"date": today, "kind": kind, "strategy": x["strategy"], "ticker": x["ticker"], "text": text})

    def data(t):
        if t not in cache:
            try:
                cache[t] = _clean(ohlcv(t))
            except Exception:
                cache[t] = None
        return cache[t]

    # 1. Gårdagens order fylls på öppningen
    keep = []
    for o in state["orders"]:
        df = data(o["ticker"])
        i = _after(df, o["signal_date"]) if df is not None else None
        if df is None or i >= len(df):
            if (pd.Timestamp(today) - pd.Timestamp(o["signal_date"])).days > ORDER_TTL_DAYS:
                ev(UTGANGEN, o, "ordern fylldes inte inom en vecka — struken")
            else:
                keep.append(o)
            continue
        entry = float(df["Open"].iloc[i])
        stop = o.get("stop")
        if o["strategy"] == "viking":
            import viking_execution as vx
            a = vx.atr(df.iloc[:i + 1])
            stop = entry - vx.ATR_STOP_MULT * float(a.iloc[-1]) if len(a) and pd.notna(a.iloc[-1]) else None
        if stop is not None and not (entry > stop > 0):
            ev(SPARRAD, o, f"fylldes inte: öppningen {entry:,.2f} ligger under stoppen {stop:,.2f}")
            continue
        p = {**o, "entry_date": _d(df.index[i]), "entry": round(entry, 4),
             "init_stop": None if stop is None else round(float(stop), 4), "checked": o["signal_date"],
             "last_close": entry, "last_date": _d(df.index[i]), "pending": None, "red": None,
             "bench_last": o["cycle"].get(f"{_gov(o)}_close")}
        state["positions"].append(p)
        ev(KOPT, p, f"fylld {p['entry_date']} @ {entry:,.2f}" + (f" · stopp {stop:,.2f}" if stop else ""))
    state["orders"] = keep

    # 2. Förvaltning
    for p in list(state["positions"]):
        _manage(state, p, data(p["ticker"]), market, wolf, board, today, ev, long_signals)

    # 3. Dagens signaler blir order
    busy = {(x["strategy"], x["ticker"]) for x in state["positions"] + state["orders"]}
    cool = pd.Timestamp(today) - pd.Timedelta(days=COOLDOWN_DAYS)
    recent = {(c["strategy"], c["ticker"]) for c in state["closed"] if pd.Timestamp(c["exit_date"]) >= cool}
    for s in sigs:
        k = (s["strategy"], s["ticker"])
        if k in busy or k in recent:
            continue
        o = {**s, "signal_date": today, "cycle": cycle_tag(board, s["segment"], s.get("theme"))}
        state["orders"].append(o)
        busy.add(k)
        st = o["cycle"][f"{_gov(o)}_status"]
        ev(KOP_OPEN, o, f"{STRATEGIES[s['strategy']][0]} · {s.get('why') or ''} · {s['segment']} {st}")

    # 4. BERSERK:s affärer märks första gången de syns
    _tag_berserk(state, berserk_state, board, universe or {}, today)

    state["events"] = (state["events"] + events)[-EVENTS_MAX:]
    state["closed"] = state["closed"][-CLOSED_MAX:]
    state["last_run"] = today
    return state, events


def _manage(state: dict, p: dict, df: Optional[pd.DataFrame], market, wolf, board, today, ev,
            long_signals: Optional[dict] = None) -> None:
    if df is None:
        return
    o, h, lo, c = (df[k].astype(float).values for k in ("Open", "High", "Low", "Close"))
    idx = df.index
    if p.get("pending"):                                            # sälj på nästa öppning
        j = _after(df, p["pending"]["date"])
        if j < len(df):
            _close(state, p, float(o[j]), p["pending"]["reason"], _d(idx[j]), ev)
        return
    kind = STRATEGIES[p["strategy"]][1]
    start = _after(df, p["checked"])
    for j in range(start, len(df)):
        date = _d(idx[j])
        p["checked"], p["last_close"], p["last_date"] = date, float(c[j]), date
        held = (pd.Timestamp(date) - pd.Timestamp(p["entry_date"])).days
        if p["strategy"] == "viking":
            import viking_exit as vex
            mkt = market.loc[:idx[j]] if market is not None else None
            dec = vex.evaluate_exit(p["ticker"], df.iloc[max(0, j + 1 - VIKING_WINDOW):j + 1], p["entry"],
                                    p["entry_date"], spy_df=mkt, today=idx[j])
            if dec.status != vex.HOLD:
                reason = (dec.reasons or [dec.status])[0][:160]
                if j + 1 < len(df):
                    _close(state, p, float(o[j + 1]), reason, _d(idx[j + 1]), ev)
                else:
                    p["pending"] = {"date": date, "reason": reason}
                    ev(SALJ_OPEN, p, f"{reason} — sälj på nästa öppning")
                return
        elif p["strategy"] == "insider":
            stop = p.get("init_stop")
            if stop and lo[j] <= stop:
                _close(state, p, float(min(o[j], stop)), "stopp (klustersnitt × 0,85)", date, ev, STOPPAD)
                return
            if held >= INSIDER_DAYS:
                _close(state, p, float(c[j]), "tidsstopp 18 månader", date, ev)
                return
        elif kind == LONG and held >= LONG_DAYS:
            _close(state, p, float(c[j]), "tidsstopp 3 år", date, ev)
            return
    if p["strategy"] == "wolf" and wolf is not None and p["ticker"] not in wolf:
        p["pending"] = {"date": p["last_date"], "reason": "föll ur Wolfs topp 40"}
        ev(SALJ_OPEN, p, "föll ur Wolfs topp 40 — sälj på nästa öppning")
    elif p["strategy"] == "long" and (long_signals or {}).get(p["ticker"]) in LONG_SELL:
        reason = f"Long Screener {long_signals[p['ticker']]}"
        p["pending"] = {"date": p["last_date"], "reason": reason}
        ev(SALJ_OPEN, p, f"{reason} — sälj på nästa öppning")
    tag = cycle_tag(board, p["segment"], p.get("theme"))
    g = _gov(p)
    if tag.get(f"{g}_close") is not None:
        p["bench_last"] = tag[f"{g}_close"]
    if kind == LONG and not p.get("red") and tag.get(f"{g}_status") == cycle.ROD:
        p["red"] = {"date": p["last_date"], "price": p["last_close"], "bench": p.get("bench_last")}
        ev(ROD, p, f"{p['segment']}{' · ' + p['theme'] if p.get('theme') else ''} blev RÖD — skuggexit "
                   f"@ {p['last_close']:,.2f} ({(p['last_close'] / p['entry'] - 1) * 100:+.1f} %); positionen behålls")


def _close(state: dict, p: dict, px: float, reason: str, date: str, ev, kind: str = SALD) -> None:
    c = {**p, "exit_date": date, "exit": round(px, 4), "reason": reason, "bench_out": p.get("bench_last")}
    c.pop("pending", None)
    state["closed"].append(c)
    state["positions"].remove(p)
    ev(kind, p, f"{reason} {date} @ {px:,.2f} · {(px / p['entry'] - 1) * 100:+.1f} %")


def _tag_berserk(state: dict, bstate: Optional[dict], board: dict, universe: dict, today: str) -> None:
    """BERSERK:s konto är eget — här märks bara varje affär med cykelläget (första gången den syns; affärer
    som redan var äldre än en vecka när Valhall startade får inget läge)."""
    if not bstate:
        return
    tags = state["berserk_tags"]
    live = set()
    old = pd.Timestamp(today) - pd.Timedelta(days=ORDER_TTL_DAYS)
    for x in (bstate.get("orders") or []) + (bstate.get("positions") or []) + (bstate.get("closed") or []):
        key = f"{x.get('ticker')}|{x.get('signal_date')}"
        live.add(key)
        seg = (universe.get(x.get("ticker")) or {}).get("segment") or sgm.segment_of_theme(x.get("theme"))
        is_open = x in (bstate.get("positions") or []) or x in (bstate.get("orders") or [])
        if key not in tags:
            fresh = pd.Timestamp(x.get("signal_date") or today) >= old
            tags[key] = {"segment": seg, "cycle": cycle_tag(board, seg, x.get("theme")) if fresh else {}}
        if is_open and tags[key]["cycle"]:
            close = cycle_tag(board, seg, x.get("theme")).get("t_close")
            if close is not None:
                tags[key]["bench_last"] = close
    state["berserk_tags"] = {k: v for k, v in tags.items() if k in live}


# ── Resultaten ──────────────────────────────────────────────────────────────
def _row(x: dict, is_open: bool, strategy: Optional[str] = None, cyc: Optional[dict] = None,
         bench_out=None) -> dict:
    strategy = strategy or x["strategy"]
    cyc = x.get("cycle") if cyc is None else cyc
    px = x.get("last_close") if is_open else x.get("exit")
    entry, stop = x.get("entry"), x.get("init_stop")
    ret = (px / entry - 1) * 100 if px and entry else None
    r = (px - entry) / (entry - stop) if px and entry and stop and entry > stop else None
    g = "t" if (strategy == "berserk" or x.get("theme")) else _gov(x)
    bench_in = (cyc or {}).get(f"{g}_close")
    bench_now = x.get("bench_last") if is_open else (bench_out if bench_out is not None else x.get("bench_out"))
    excess = ret - (bench_now / bench_in - 1) * 100 if ret is not None and bench_in and bench_now else None
    red = x.get("red")
    red_ret = (red["price"] / entry - 1) * 100 if red and entry else ret
    return {"strategy": strategy, "segment": x.get("segment"), "theme": x.get("theme"), "open": is_open,
            "status_in": (cyc or {}).get(f"{g}_status") or "OKÄND", "cykel_in": (cyc or {}).get(f"{g}_cykel") or "OKÄND",
            "phase_in": (cyc or {}).get(f"{g}_phase") or "OKÄND", "ret_pct": ret, "r": r, "red_ret_pct": red_ret,
            "excess_pct": excess, "red": bool(red)}


def rows(state: dict, berserk_state: Optional[dict] = None) -> list:
    """En rad per affär (öppna värderade på senaste stängning), BERSERK:s konto inräknat."""
    out = [_row(p, True) for p in state.get("positions") or []] + [_row(c, False) for c in state.get("closed") or []]
    tags = state.get("berserk_tags") or {}
    for is_open, xs in ((True, (berserk_state or {}).get("positions") or []),
                        (False, (berserk_state or {}).get("closed") or [])):
        for x in xs:
            t = tags.get(f"{x.get('ticker')}|{x.get('signal_date')}") or {}
            out.append(_row({**x, "segment": t.get("segment"), "bench_last": t.get("bench_last")}, is_open,
                            "berserk", t.get("cycle") or {}, bench_out=t.get("bench_last")))
    return out


def _mean(v: list, nd: int = 2):
    v = [x for x in v if x is not None]
    return round(float(np.mean(v)), nd) if v else None


def group(rs: list, keys: tuple) -> list:
    """[{nycklar…, n, open, closed, avg_ret, win, avg_r, sum_r, avg_red_ret, avg_excess}] per grupp."""
    groups: dict = {}
    for r in rs:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for k, g in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rets = [r["ret_pct"] for r in g if r["ret_pct"] is not None]
        closed_r = [r["r"] for r in g if not r["open"] and r["r"] is not None]
        out.append({**dict(zip(keys, k)), "n": len(g), "open": sum(1 for r in g if r["open"]),
                    "closed": sum(1 for r in g if not r["open"]), "avg_ret": _mean(rets),
                    "win": round(sum(1 for x in rets if x > 0) / len(rets) * 100, 1) if rets else None,
                    "avg_r": _mean(closed_r, 3), "sum_r": round(float(np.sum(closed_r)), 2) if closed_r else 0.0,
                    "avg_red_ret": _mean([r["red_ret_pct"] for r in g]), "red": sum(1 for r in g if r["red"]),
                    "avg_excess": _mean([r["excess_pct"] for r in g])})
    return out


def results(state: dict, berserk_state: Optional[dict] = None) -> dict:
    """Sammanställningen som sparas i valhall_board.json: per strategi, segment, status och cykelläge."""
    rs = rows(state or new_state(), berserk_state)
    return {"since": (state or {}).get("start"), "last_run": (state or {}).get("last_run"),
            "orders": len((state or {}).get("orders") or []), "positions": len((state or {}).get("positions") or []),
            "by_strategy": group(rs, ("strategy",)), "by_segment": group(rs, ("strategy", "segment")),
            "by_status": group(rs, ("strategy", "status_in")), "by_cykel": group(rs, ("strategy", "cykel_in"))}


def summary_lines(res: dict) -> list:
    """Läsbar sammanfattning för loggen."""
    lines = [f"Papperskonton sedan {res.get('since')}: {res.get('positions')} öppna · {res.get('orders')} order"]
    for g in res.get("by_strategy") or []:
        lines.append(f"  {STRATEGIES.get(g['strategy'], (g['strategy'],))[0]:18s} n={g['n']:3d} öppna={g['open']:3d} "
                     f"snitt={g['avg_ret']} % vinst={g['win']} % R={g['avg_r']} mot riktmärke={g['avg_excess']} % "
                     f"sälj på RÖD={g['avg_red_ret']} %")
    return lines
