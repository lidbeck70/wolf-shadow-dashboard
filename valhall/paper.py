"""
valhall/paper.py — fas 4: Valhalls papperskonton per strategi, varje affär märkt med segment och cykelläge.

Målet är att hitta den bästa strategin per segment och cykel (fas 3a visade att cykelstatusen inte duger
som spärr för swing — nu får papperet visa vad som fungerar). Ett konto per strategi, en "enhet" per
signal (ingen kassa eller platsbrist som skymmer jämförelsen). Varje köp märks PÅ SIGNALDAGEN med
Valhalls läge: segmentets status, fas och cykelläge i Norden och globalt och råvarutemats.

Strategi         Köp (signalen ur dagens Valhall-körning)            Sälj (befintliga regler, oförändrade)
Viking Nine      GOLDEN TICKET i segment med Viking Nine (swing.py)    viking_exit.evaluate_exit → nästa öppning
Insider Norden   status KÖP utan passa-flagga (insider.buy_candidates) stopp klustersnitt × 0,85 · 18 mån
Momentum Swing   topp 20 + setup A i wolf_screener.json (kontot "wolf") ur topp 40 → nästa öppning
Wolf             Arc-Wolf-rad vars köpregel är uppfylld (valhall.wolf,   hård stopp 2,5 × ATR14 (intradag) · tre
                 strategies/wolf.py entry_fn; kontot "wolf_arc")         stängningar under EMA50 eller under Kijun
                                                                         och EMA10 → nästa öppning (inga delvinster)
Quality          segmentets kvalitetsköp med Alpha Regime BUY, ej RÖD   tidsstopp 3 år
                 sektor (picks.gate_segments)
Rick Rule        råvarans kvalitetsköp (Köpkandidat / Rule-håven / BUY) tidsstopp 3 år
Durrett          guld/silver: 15-BAGGER, eller nordiskt bolag           tidsstopp 3 år
Tiggre           övriga råvaror: alla köpgrindar gröna                  tidsstopp 3 år
Long Screener    segmentets två bästa STRONG BUY/BUY (cykel "gron")     screenerns SELL → nästa öppning · 3 år
Long Screener (utan cykel)  samma screener, cykelpoäng alltid 3            screenerns SELL → nästa öppning · 3 år
BERSERK          eget konto (berserk_paper.json) — bara märkt här

Långsiktiga positioner får dessutom en SKUGGEXIT: första dagen segmentets (temats) status är RÖD noteras
priset — resultaten visar både "behåll" och "sälj på RÖD" (CLAUDE.md: minska vid röd regim — prövas här).
Avkastningen jämförs med segmentets riktmärke (nordiska sektorindexet, SPDR-fonden eller temats drivare).

Köp fylls på nästa öppning efter signaldagen. Ett bolag ägs en gång per konto (aktieslag slås ihop, även i
karantänen efter en stängning), och Quality-order prövas mot grinden ej RÖD med signaldagens läge även vid
fyllningen (order från före grinden). Valutor ignoreras (avkastning i lokal valuta). Riktiga
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
STRATEGIES = {"viking": ("Viking Nine", SWING), "insider": ("Insider Norden", SWING),
              "wolf": ("Momentum Swing", SWING), "wolf_arc": ("Wolf", SWING), "berserk": ("BERSERK", SWING), "quality": ("Quality Contrarian", LONG), "rule": ("Rick Rule", LONG),
              "durrett": ("Durrett", LONG), "tiggre": ("Tiggre", LONG), "long": ("Long Screener", LONG),
              "long_ingen": ("Long Screener (utan cykel)", LONG),
              # K7 (analysen 2026-10-24 §7.1): komplexstrategierna och kontona som saknades
              "komplex_viking": ("Viking Nine · komplex", SWING), "komplex_deep": ("Deep Contrarian · vändning", LONG),
              "komplex_long": ("Lång · vändning", LONG), "komplex_etf": ("UCITS-ETF · vändning", LONG),
              "ember": ("EMBER", SWING), "deep": ("Deep Contrarian", LONG), "royalty": ("Royalty", LONG)}
EMBER_DAYS = 60                    # EMBER: tidsstopp (beslut 11 — EMBER har ingen egen exitmotor; stoppen ur setup-kortet)
N_DEEP_DAILY = 2                   # kontot deep (kontrollgrupp): två bästa per segment varje dag, som quality
LONG_SELL = ("SELL", "STRONG SELL")      # cagr_scoring: EMA200-brott, röd regim eller poäng under 35 %
LONG_DAYS = 3 * 365                # horisont 0,5–3 år
INSIDER_DAYS = 548                 # insiderns tidsstopp: 18 månader
COOLDOWN_DAYS = 14                 # ingen ny affär i samma aktie och strategi två veckor efter en stängning
NODATA_RUNS_MAX = 10               # position utan kursdata så många körningar i rad stängs (avnoterad)
ORDER_TTL_DAYS = 7
WOLF_TOP, WOLF_STALE_DAYS = 20, 4
EVENTS_MAX, CLOSED_MAX = 1000, 3000
VIKING_WINDOW = 300                # dagar som viking_exit får se (EMA10, ATR14, F&G)
WOLF_WINDOW = 300                  # dagar som strategies/wolf.exit_fn får se (EMA10/50, Kijun 26, ATR14)
WOLF_EXIT = {"STOP_LOSS": "hård stopp (2,5 × ATR14)", "EMA50_EXIT": "tre stängningar under EMA50",
             "TRAIL_EXIT": "under Kijun och EMA10"}

KOP_OPEN, KOPT, SALJ_OPEN, SALD, STOPPAD = "KÖP PÅ ÖPPNING", "KÖPT", "SÄLJ PÅ ÖPPNING", "SÅLD", "STOPPAD"
SPARRAD, UTGANGEN, ROD = "SPÄRRAD", "UTGÅNGEN", "RÖD (skuggexit)"


def new_state(today=None) -> dict:
    return {"version": 1, "start": _d(today or pd.Timestamp.today()), "orders": [], "positions": [], "closed": [],
            "events": [], "berserk_tags": {}, "last_run": None}


def _d(ts) -> str:
    return str(pd.Timestamp(ts).date())


def _company(x: dict) -> tuple:
    """(strategi, bolag) — samma bolag i flera aktieslag (HLUN-A/HLUN-B) räknas som ett per konto."""
    from valhall.picks import company_key
    return x["strategy"], company_key({"name": x.get("name") or "", "ticker": x.get("ticker")})


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
def _sig(strategy, ticker, name, segment, theme=None, nordic=None, why="", stop=None, complex=None,
         profile=None) -> dict:
    nordic = buv.region_of(ticker or "") == "Norden" if nordic is None else bool(nordic)
    from valhall import complexes as vcx
    return {"strategy": strategy, "ticker": ticker, "name": name or "", "segment": segment, "theme": theme,
            "nordic": nordic, "why": why, "stop": stop, "complex": complex or vcx.complex_of(theme), "profile": profile}


def buy_ok(p: dict) -> bool:
    """Kortets köp som papperet tar — varje källas egen köpregel, och Quality-kortets grind (ej RÖD sektor)."""
    if p.get("gate_blocked"):
        return False
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
            wolf: Optional[dict] = None, long_scan: Optional[dict] = None, arc_sigs: Optional[list] = None,
            complex_swing: Optional[dict] = None, turn_plan: Optional[dict] = None, ember_blob: Optional[dict] = None,
            deep_blob: Optional[dict] = None, royalty_blob: Optional[dict] = None) -> list:
    """Dagens köpsignaler per strategi. wolf = wolf_live(...) (momentumlistan), long_scan = long_screener.scan(...),
    arc_sigs = valhall.wolf.arc_signals(...) (Arc-Wolf, kontot wolf_arc), complex_swing = swing.complex_candidates
    (kontot komplex_viking), turn_plan = turns.plan (komplex_deep/komplex_long/komplex_etf), ember_blob =
    ember_results.json (kontot ember), deep_blob = CA deep_contrarian (kontot deep), royalty_blob = royalty_scan.json."""
    out = list(arc_sigs or []) + complex_signals(complex_swing, turn_plan, ember_blob, deep_blob, royalty_blob, universe)
    for key, e in ((long_scan or {}).get("segments") or {}).items():
        for strat, field in (("long", "picks"), ("long_ingen", "picks_ingen")):
            for p in e.get(field) or []:
                out.append(_sig(strat, p["ticker"], p.get("name"), key, None, True,
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


def complex_signals(complex_swing: Optional[dict], turn_plan: Optional[dict], ember_blob: Optional[dict],
                    deep_blob: Optional[dict], royalty_blob: Optional[dict], universe: dict) -> list:
    """K7: signalerna till de nya kontona — allt dokumenteras i papper (analysen §7.1)."""
    from valhall import complexes as vcx
    out = []
    for key, e in ((complex_swing or {}).get("complexes") or {}).items():          # Viking Nine · komplex
        for p in e.get("picks") or []:
            if p.get("category") == "GOLDEN TICKET":
                out.append(_sig("komplex_viking", p["ticker"], p.get("name"), p.get("segment"), None, True,
                                f"{p.get('ovtlyr_nine')} · {e.get('strategy')}", complex=key, profile=key))
    for w in (turn_plan or {}).get("window") or []:                               # 2 deep + 2 lång + ETF vid vändning
        cx = w.get("key") if w.get("level") == "komplex" else vcx.complex_of(w.get("key")) if w.get("level") == "tema" else None
        seg = w.get("key") if w.get("level") == "segment" else None
        head = f"vändning {w.get('label') or w.get('key')} dag {w.get('days') or '?'}"
        for p in w.get("deep") or []:
            out.append(_sig("komplex_deep", p["ticker"], p.get("name"), p.get("segment") or seg, p.get("theme"), None,
                            f"{head} · composite {p.get('score')}", complex=cx))
        for p in w.get("long") or []:
            why = f"{head} · {p.get('signal') or p.get('verdict') or ''} ({p.get('source') or 'lång'})"
            out.append(_sig("komplex_long", p["ticker"], p.get("name"), p.get("segment") or seg, p.get("theme"), None, why,
                            complex=cx))
        e = w.get("etf") or {}
        if e.get("ticker"):
            out.append(_sig("komplex_etf", e["ticker"], e.get("name"), seg, None, False,
                            f"{head} · {e.get('exposure')} · poäng {e.get('score')}"
                            + ("" if e.get("verified") else " · linjen inte verifierad hos Nordnet"), complex=cx))
    for r in (ember_blob or {}).get("eligible") or []:                            # EMBER: köplägen med kortets stopp
        t = r.get("ticker")
        if not t:
            continue
        theme = r.get("theme_key") or None
        seg = (universe.get(t) or {}).get("segment") or sgm.segment_of_theme(theme)
        out.append(_sig("ember", t, (universe.get(t) or {}).get("name"), seg, theme, None,
                        f"KÖPLÄGE {r.get('setup_score')} · {r.get('typ') or ''}".strip(), stop=r.get("stop")))
    from valhall import picks as vp
    deep_rows = sorted(vp._ca_rows([deep_blob] if deep_blob else []), key=lambda r: -(r.get("score") or 0))
    for sgm_ in sgm.SEGMENTS:                                                     # deep (kontroll): 2 bästa/segment, varje dag
        mine = vp._one_per_company([r for r in deep_rows if r.get("segment") == sgm_.key])[:N_DEEP_DAILY]
        for r in mine:
            out.append(_sig("deep", r["ticker"], r.get("name"), sgm_.key, r.get("theme"), None,
                            f"composite {r.get('score')} · rank {r.get('rank')}"))
    for r in (royalty_blob or {}).get("rows") or []:                              # Royalty: BUY CANDIDATE
        if str(r.get("verdict") or "").upper() == "BUY CANDIDATE" and r.get("ticker"):
            out.append(_sig("royalty", r["ticker"], r.get("name"), None, None, False,
                            f"composite {r.get('composite') or r.get('composite_score') or ''}"))
    return out


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
         universe: Optional[dict] = None, long_signals: Optional[dict] = None,
         long_ingen_signals: Optional[dict] = None, exit_ctx: Optional[dict] = None,
         turns: Optional[list] = None) -> tuple:
    """(nytt tillstånd, nya händelser). ohlcv(ticker) → dagsdata, market = OMXS30 (Viking Nines marknadsexit),
    wolf = wolf_live(...) (None = listan saknas → inga Wolf-sälj i dag), berserk_state = BERSERK:s konto,
    long_signals = {ticker: Long Screener-signal} (None = skanningen saknas → inga sälj i dag),
    long_ingen_signals = samma utan cykel (kontot "Long Screener (utan cykel)"),
    exit_ctx = {nine(ticker, df), ob(df), earnings(ticker), driver(profil)} — ger Viking-exiten hela playbooken
    (sektor/bredd, aktiens signal, bearish block, rapport) på dagens stapel och komplexvarianten drivarexiten;
    turns = kartans vändningar (board.turns) — ordern märks med rutans vändning (kind, dag, bekräftad)."""
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
    held = {_company(p) for p in state["positions"]}
    for o in state["orders"]:
        # Grinden ej RÖD med läget PÅ SIGNALDAGEN (ordern bär det) — dagens karta bygger på dagens
        # stängning, efter öppningen där ordern fylls, och skulle titta framåt. Fångar order som lades
        # innan grinden fanns (14/10: HEXA-B, NFGAB, BIOMAR var RÖDA redan på signaldagen).
        if o["strategy"] == "quality" and (o.get("cycle") or {}).get("n_status") == cycle.ROD:
            ev(SPARRAD, o, "ordern struken: grinden ej RÖD — sektorn var RÖD på signaldagen")
            continue
        if _company(o) in held:
            ev(UTGANGEN, o, "ordern struken: samma bolag finns redan i kontot (annat aktieslag)")
            continue
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
        if o["strategy"] in ("viking", "komplex_viking"):
            import viking_execution as vx
            a = vx.atr(df.iloc[:i + 1])
            stop = entry - vx.ATR_STOP_MULT * float(a.iloc[-1]) if len(a) and pd.notna(a.iloc[-1]) else None
        elif o["strategy"] == "wolf_arc":
            from strategies.wolf import risk_fn
            d = risk_fn(df.iloc[:i + 1], 1.0)["stop_distance"]              # 2,5 × ATR14 på fyllningsdagen
            stop = entry - float(d) if d and d > 0 else None
        if stop is not None and not (entry > stop > 0):
            ev(SPARRAD, o, f"fylldes inte: öppningen {entry:,.2f} ligger under stoppen {stop:,.2f}")
            continue
        p = {**o, "entry_date": _d(df.index[i]), "entry": round(entry, 4),
             "init_stop": None if stop is None else round(float(stop), 4), "checked": o["signal_date"],
             "last_close": entry, "last_date": _d(df.index[i]), "pending": None, "red": None,
             "bench_last": o["cycle"].get(f"{_gov(o)}_close")}
        if o["strategy"] == "wolf_arc":
            p["below50"] = 0                                               # stängningar i rad under EMA50
        state["positions"].append(p)
        held.add(_company(p))
        ev(KOPT, p, f"fylld {p['entry_date']} @ {entry:,.2f}" + (f" · stopp {stop:,.2f}" if stop else ""))
    state["orders"] = keep

    # 2. Förvaltning — en position utan kursdata NODATA_RUNS_MAX körningar i rad (avnoterad, bytt ticker)
    # stängs på senast kända kurs i stället för att ligga öppen för evigt
    for p in list(state["positions"]):
        df = data(p["ticker"])
        if df is None:
            p["nodata_runs"] = int(p.get("nodata_runs") or 0) + 1
            if p["nodata_runs"] >= NODATA_RUNS_MAX:
                _close(state, p, float(p.get("last_close") or p["entry"]),
                       f"ingen kursdata på {p['nodata_runs']} körningar — avnoterad eller bytt ticker; stängd på senast kända kurs",
                       today, ev)
            continue
        p["nodata_runs"] = 0
        _manage(state, p, df, market, wolf, board, today, ev,
                long_ingen_signals if p["strategy"] == "long_ingen" else long_signals, exit_ctx=exit_ctx)

    # 3. Dagens signaler blir order
    busy = {(x["strategy"], x["ticker"]) for x in state["positions"] + state["orders"]}
    busy_co = {_company(x) for x in state["positions"] + state["orders"]}
    cool = pd.Timestamp(today) - pd.Timedelta(days=COOLDOWN_DAYS)
    closed_recent = [c for c in state["closed"] if pd.Timestamp(c["exit_date"]) >= cool]
    recent = {(c["strategy"], c["ticker"]) for c in closed_recent}
    recent_co = {_company(c) for c in closed_recent}               # karantänen gäller tickern OCH bolaget
    for s in sigs:
        k = (s["strategy"], s["ticker"])
        if k in busy or k in recent or _company(s) in busy_co or _company(s) in recent_co:
            continue
        o = {**s, "signal_date": today, "cycle": cycle_tag(board, s["segment"], s.get("theme")),
             "turn": turn_tag(turns, s.get("segment"), s.get("theme"), s.get("complex"))}
        state["orders"].append(o)
        busy.add(k)
        busy_co.add(_company(o))
        st = o["cycle"][f"{_gov(o)}_status"]
        ev(KOP_OPEN, o, f"{STRATEGIES[s['strategy']][0]} · {s.get('why') or ''} · {s['segment']} {st}")

    # 4. BERSERK:s affärer märks första gången de syns
    _tag_berserk(state, berserk_state, board, universe or {}, today)

    state["events"] = (state["events"] + events)[-EVENTS_MAX:]
    state["closed_dropped"] = int(state.get("closed_dropped") or 0) + max(0, len(state["closed"]) - CLOSED_MAX)
    state["closed"] = state["closed"][-CLOSED_MAX:]
    state["last_run"] = today
    return state, events


def _exit_inputs(exit_ctx: Optional[dict], ticker: str, window: pd.DataFrame) -> tuple:
    """(nine, ob_analysis, earnings_date) för Viking-exiten ur exit_ctx — None för det som inte går att räkna."""
    out = []
    for key, args in (("nine", (ticker, window)), ("ob", (window,)), ("earnings", (ticker,))):
        fn = (exit_ctx or {}).get(key)
        try:
            out.append(fn(*args) if fn else None)
        except Exception:
            out.append(None)
    return tuple(out)


def _manage(state: dict, p: dict, df: Optional[pd.DataFrame], market, wolf, board, today, ev,
            long_signals: Optional[dict] = None, exit_ctx: Optional[dict] = None) -> None:
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
    e50 = None
    for j in range(start, len(df)):
        date = _d(idx[j])
        p["checked"], p["last_close"], p["last_date"] = date, float(c[j]), date
        held = (pd.Timestamp(date) - pd.Timestamp(p["entry_date"])).days
        if p["strategy"] in ("viking", "komplex_viking"):
            import viking_exit as vex
            mkt = market.loc[:idx[j]] if market is not None else None
            window = df.iloc[max(0, j + 1 - VIKING_WINDOW):j + 1]
            if p["strategy"] == "komplex_viking" and j == len(df) - 1 and (exit_ctx or {}).get("driver"):
                try:                                                   # drivaren under EMA50 (BERSERK S2) → sälj
                    drv = exit_ctx["driver"](p.get("profile"))
                except Exception:
                    drv = None
                if drv and drv.get("triggered"):
                    reason = f"drivarexit: {drv.get('detail')}"
                    p["pending"] = {"date": date, "reason": reason}
                    ev(SALJ_OPEN, p, f"{reason} — sälj på nästa öppning")
                    return
            # hela playbooken bara på dagens stapel (Nine kostar nätanrop); ikappdagar får den smala exiten
            nine, ob, ed = _exit_inputs(exit_ctx, p["ticker"], window) if j == len(df) - 1 else (None, None, None)
            dec = vex.evaluate_exit(p["ticker"], window, p["entry"], p["entry_date"], nine=nine, ob_analysis=ob,
                                    spy_df=mkt, earnings_date=ed, today=idx[j])
            if dec.status != vex.HOLD:
                reason = (dec.reasons or [dec.status])[0][:160]
                if j + 1 < len(df):
                    _close(state, p, float(o[j + 1]), reason, _d(idx[j + 1]), ev)
                else:
                    p["pending"] = {"date": date, "reason": reason}
                    ev(SALJ_OPEN, p, f"{reason} — sälj på nästa öppning")
                return
        elif p["strategy"] == "wolf_arc":
            stop = p.get("init_stop")
            if stop and lo[j] <= stop:
                _close(state, p, float(min(o[j], stop)), WOLF_EXIT["STOP_LOSS"], date, ev, STOPPAD)
                return
            from strategies.wolf import exit_fn
            if e50 is None:
                e50 = df["Close"].astype(float).ewm(span=50, adjust=False).mean().values
            p["below50"] = p.get("below50", 0) + 1 if c[j] < e50[j] else 0
            pos = {"entry_price": p["entry"], "bars_below_ema50": p["below50"]}
            if stop:
                pos["stop_loss"] = stop
            dec = exit_fn(pos, df.iloc[max(0, j + 1 - WOLF_WINDOW):j + 1])
            if dec.get("exit"):
                reason = WOLF_EXIT.get(dec.get("reason"), dec.get("reason") or "exit")
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
        elif p["strategy"] == "ember":
            stop = p.get("init_stop")
            if stop and lo[j] <= stop:
                _close(state, p, float(min(o[j], stop)), "stopp (EMBER:s setup-kort)", date, ev, STOPPAD)
                return
            if held >= EMBER_DAYS:
                _close(state, p, float(c[j]), f"tidsstopp {EMBER_DAYS} dagar", date, ev)
                return
        elif kind == LONG and held >= LONG_DAYS:
            _close(state, p, float(c[j]), "tidsstopp 3 år", date, ev)
            return
    if p["strategy"] == "wolf" and wolf is not None and p["ticker"] not in wolf:
        p["pending"] = {"date": p["last_date"], "reason": "föll ur Wolfs topp 40"}
        ev(SALJ_OPEN, p, "föll ur Wolfs topp 40 — sälj på nästa öppning")
    elif p["strategy"] in ("long", "long_ingen") and (long_signals or {}).get(p["ticker"]) in LONG_SELL:
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
def turn_tag(turns: Optional[list], segment: Optional[str], theme: Optional[str] = None,
             complex_key: Optional[str] = None) -> dict:
    """Rutans vändning på signaldagen ur board.turns: temat först, sedan komplexet, sedan segmentet (Norden).
    {kind, since, days, confirmed, confirmed_since, level} — tomt när ingen pågår."""
    ts = [t for t in turns or [] if isinstance(t, dict) and t.get("kind")]
    for level, key in (("tema", theme), ("komplex", complex_key), ("segment", segment)):
        if not key:
            continue
        for t in ts:
            if t.get("level") == level and t.get("key") == key and (level != "segment" or t.get("where") == "Norden"):
                return {k: t.get(k) for k in ("kind", "since", "days", "confirmed", "confirmed_since")} | {"level": level}
    return {}


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
    turn = x.get("turn") or {}
    return {"strategy": strategy, "segment": x.get("segment"), "theme": x.get("theme"), "open": is_open,
            "complex": x.get("complex") or "—",
            "status_in": (cyc or {}).get(f"{g}_status") or "OKÄND", "cykel_in": (cyc or {}).get(f"{g}_cykel") or "OKÄND",
            "phase_in": (cyc or {}).get(f"{g}_phase") or "OKÄND",
            "turn_in": turn.get("kind") or "ingen", "confirmed_in": bool(turn.get("confirmed")),
            "days_since_turn": turn.get("days"), "ret_pct": ret, "r": r, "red_ret_pct": red_ret,
            "excess_pct": excess, "red": bool(red), "entry_date": x.get("entry_date"),
            "exit_date": None if is_open else x.get("exit_date"), "held_days": _held(x, is_open)}


def _held(x: dict, is_open: bool) -> Optional[int]:
    end = x.get("last_date") if is_open else x.get("exit_date")
    try:
        return int((pd.Timestamp(end) - pd.Timestamp(x.get("entry_date"))).days)
    except Exception:
        return None


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


MIN_CLOSED = 20                    # journalens gräns (journal_stats) — färre stängda affärer = ingen slutsats
UNIT_PCT = 1.0                     # enhetskurvan: varje stängd affär väger 1 % av ett nominellt kapital


def closed_stats(g: list) -> dict:
    """Mått på STÄNGDA affärer (öppna är mark-to-market och visas separat): vinstandel, payoff, profit factor,
    hålltid, överavkastning per 30 dagar, enhetskurvans årstakt och max drawdown, och om n räcker för slutsats."""
    c = [r for r in g if not r["open"] and r["ret_pct"] is not None]
    rets = [r["ret_pct"] for r in c]
    wins, losses = [x for x in rets if x > 0], [x for x in rets if x <= 0]
    held = [r["held_days"] for r in c if r.get("held_days") is not None]
    ex30 = [r["excess_pct"] / max(r["held_days"], 1) * 30 for r in c
            if r.get("excess_pct") is not None and r.get("held_days") is not None]
    out = {"closed_win": round(len(wins) / len(rets) * 100, 1) if rets else None,
           "closed_avg_ret": _mean(rets), "payoff": round(np.mean(wins) / abs(np.mean(losses)), 2) if wins and losses and np.mean(losses) else None,
           "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if wins and losses and sum(losses) else None,
           "avg_hold": _mean(held, 1), "excess_30d": _mean(ex30), "enough": len(c) >= MIN_CLOSED,
           "closed_n": f"{len(c)}/{MIN_CLOSED}"}
    out.update(unit_curve(c))
    return out


def unit_curve(closed: list) -> dict:
    """Enhetskurvan: 100 + summan av UNIT_PCT × avkastningen per stängd affär i exitordning → årstakt (CAGR) och
    max drawdown. Ingen kassa och ingen platsbrist — en jämförelse mellan strategier, inte en portfölj."""
    xs = sorted((r for r in closed if r.get("exit_date") and r.get("entry_date")), key=lambda r: r["exit_date"])
    if not xs:
        return {"unit_cagr": None, "unit_dd": None}
    eq, peak, dd = 100.0, 100.0, 0.0
    for r in xs:
        eq += UNIT_PCT * r["ret_pct"]
        peak = max(peak, eq)
        dd = min(dd, (eq / peak - 1) * 100)
    years = (pd.Timestamp(xs[-1]["exit_date"]) - pd.Timestamp(min(r["entry_date"] for r in xs))).days / 365.25
    cagr = ((eq / 100.0) ** (1 / years) - 1) * 100 if years >= 0.25 and eq > 0 else None
    return {"unit_cagr": round(cagr, 2) if cagr is not None else None, "unit_dd": round(dd, 2)}


def group(rs: list, keys: tuple) -> list:
    """[{nycklar…, n, open, closed, avg_ret, win, avg_r, sum_r, avg_red_ret, avg_excess, + closed_stats}] per grupp."""
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
                    "avg_excess": _mean([r["excess_pct"] for r in g]), **closed_stats(g)})
    return out


def snapshot(res: dict, today) -> dict:
    """En rad per strategi för historiken (valhall_paper_history.json): så syns hur måtten utvecklas över tid."""
    keep = ("n", "closed", "closed_win", "avg_r", "payoff", "profit_factor", "avg_hold", "excess_30d", "unit_cagr", "unit_dd")
    return {"date": str((pd.Timestamp(today) if today is not None else pd.Timestamp.today()).date()),
            "by_strategy": {r["strategy"]: {k: r.get(k) for k in keep} for r in (res or {}).get("by_strategy") or []}}


def append_history(history: Optional[list], snap: dict, max_rows: int = 800) -> list:
    """Dagens ögonblicksbild läggs till (samma datum ersätts) — äldst först, högst max_rows."""
    rows = [h for h in (history or []) if isinstance(h, dict) and h.get("date") != snap["date"]]
    return (rows + [snap])[-max_rows:]


def results(state: dict, berserk_state: Optional[dict] = None) -> dict:
    """Sammanställningen som sparas i valhall_board.json: per strategi, segment, status och cykelläge."""
    rs = rows(state or new_state(), berserk_state)
    return {"since": (state or {}).get("start"), "last_run": (state or {}).get("last_run"),
            "orders": len((state or {}).get("orders") or []), "positions": len((state or {}).get("positions") or []),
            "closed_dropped": int((state or {}).get("closed_dropped") or 0),      # stängda affärer utanför CLOSED_MAX
            "by_strategy": group(rs, ("strategy",)), "by_segment": group(rs, ("strategy", "segment")),
            "by_status": group(rs, ("strategy", "status_in")), "by_cykel": group(rs, ("strategy", "cykel_in")),
            "by_complex": group(rs, ("strategy", "complex")), "by_phase": group(rs, ("strategy", "phase_in")),
            "by_turn": group(rs, ("strategy", "turn_in", "confirmed_in"))}


def summary_lines(res: dict) -> list:
    """Läsbar sammanfattning för loggen."""
    lines = [f"Papperskonton sedan {res.get('since')}: {res.get('positions')} öppna · {res.get('orders')} order"]
    for g in res.get("by_strategy") or []:
        lines.append(f"  {STRATEGIES.get(g['strategy'], (g['strategy'],))[0]:18s} n={g['n']:3d} öppna={g['open']:3d} "
                     f"snitt={g['avg_ret']} % vinst={g['win']} % R={g['avg_r']} mot riktmärke={g['avg_excess']} % "
                     f"sälj på RÖD={g['avg_red_ret']} %")
    return lines
