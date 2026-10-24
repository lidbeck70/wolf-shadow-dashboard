"""
tabs/home_today.py — 🏠 I DAG (designförslaget PR 8, avsnitt 2): morgonbeskedet på en skärm.

Allt läses ur det som redan är sparat (Gisten och arken) — inget räknas live. Sidan visar och länkar; den väljer,
rangordnar och slår inte ihop något som motorerna inte redan gör. Ordningen:

  0. VÄNDNING — KÖP I DAG   Valhalls köpfönster (turn_plan): 2 Deep Contrarian, 2 långa, 1 UCITS-ETF per ruta
  1. Kapitalet              ODIN → Lägets fyra ljus (odin/laget.py)
  2. Kvalar                 antal KÖP/BEVAKA och de fem första KÖP-raderna i Thors ordning
  3. Storlek                Storlek-raden (thor/storlek.py) för samma fem
  4. Sälj och bevaka        innehavens verdikt ur larmtillståndet (swing_verdict, sparat av alert_scan)
  5. Mot målet              portfolio/mot_malet.today_line
  Senaste larmen            larmloggen (alerts/history)
  I dag förfaller           Årshjulets rutiner (routines.due_today)

Rena funktioner här; tabs/home.py ritar.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

TOP_N = 5


def _bdays(since, today) -> Optional[int]:
    try:
        return len(pd.bdate_range(pd.Timestamp(since), pd.Timestamp(today))) - 1
    except Exception:
        return None


def turn_cards(board: Optional[dict], today=None) -> list:
    """Rutorna i köpfönstret: [{label, where, kind, day, of, deep, long, etf}] — dag räknas från bekräftelsen."""
    plan = (board or {}).get("turn_plan") or {}
    of = int(plan.get("window_days") or 5)
    today = today if today is not None else pd.Timestamp.today().normalize()
    out = []
    from valhall import turns as vt
    for w in plan.get("window") or []:
        if not isinstance(w, dict) or not vt.in_window(w, today, of):       # stängt fönster (gammal karta) visas inte
            continue
        d = _bdays(w.get("confirmed_since") or w.get("since"), today)
        where = {"komplex": "komplex", "tema": "tema"}.get(w.get("level"), w.get("where") or "segment")
        out.append({"label": w.get("label") or w.get("key"), "where": where, "kind": w.get("kind"),
                    "day": d, "of": of,                       # handelsdagar efter bekräftelsen: 0 … of
                    "deep": [x.get("ticker") for x in w.get("deep") or [] if x.get("ticker")],
                    "long": [x.get("ticker") for x in w.get("long") or [] if x.get("ticker")],
                    "etf": w.get("etf") or None})
    return out


def on_the_way(board: Optional[dict]) -> int:
    """Vändningar som pågår men inte är bekräftade än."""
    return sum(1 for t in (board or {}).get("turns") or [] if isinstance(t, dict) and not t.get("confirmed"))


def turn_headline(card: dict) -> str:
    d = card.get("day")
    day = (" · bekräftad i dag" if d == 0 else f" · dag {d} av {card['of']}" if d is not None else "")
    return f"⚡ VÄNDNING — {card['label']}{day}"


def no_turn_line(board: Optional[dict]) -> str:
    n = on_the_way(board)
    return "Ingen vändning i köpfönstret" + (f" · {n} på väg" if n else "")


def qualifiers(rows: list) -> tuple:
    """(antal KÖP, antal BEVAKA, de TOP_N första KÖP-raderna i Thors ordning). Gamla körningar räknas med men
    märks GAMMAL på raden (stale_tag)."""
    from thor import candidates as tc
    rows = sorted(rows or [], key=tc.sort_key)
    buy = [c for c in rows if c.status_class == tc.KOP]
    watch = [c for c in rows if c.status_class == tc.BEVAKA]
    return len(buy), len(watch), buy[:TOP_N]


def stale_tag(c) -> str:
    """' · GAMMAL (körd 2026-10-01)' när raden kommer ur en körning äldre än Thors gräns, annars ''."""
    return f" · GAMMAL (körd {c.asof})" if getattr(c, "stale", False) else ""


def size_line(c, capital: float) -> str:
    from thor import registry as reg
    from thor import storlek as sk
    ccy, fx = sk.fx_for(c.ticker)
    sz = sk.size(c.entry, c.stop, capital, sk.risk_pct(c.strategy), sk.cap_pct(c.strategy), fx=fx, currency=ccy)
    if sz and "GUL regim" in (c.status or ""):
        sz = sk.halved(sz)
    return sk.row_text(c.ticker, reg.label(c.strategy), sz) if sz else f"{c.ticker} · {reg.label(c.strategy)} · nivåer saknas"


SELL_WORDS = {"SELL": "SÄLJ", "SÄLJ": "SÄLJ", "PARTIAL": "MINSKA", "REDUCE": "MINSKA", "WATCH": "VAKSAM"}


def holdings_lines(alert_state: Optional[dict], n_open: Optional[int] = None) -> list:
    """[(ticker, ord, skäl)] för innehav med SÄLJ/MINSKA/VAKSAM ur larmtillståndets swing-verdikt."""
    verdicts = (((alert_state or {}).get("swing") or {}).get("verdicts")) or {}
    out = []
    for t, v in sorted(verdicts.items()):
        w = SELL_WORDS.get(str((v or {}).get("verdict") or "").upper())
        if w:
            out.append((t, w, str((v or {}).get("reason") or "")))
    order = {"SÄLJ": 0, "MINSKA": 1, "VAKSAM": 2}
    return sorted(out, key=lambda x: (order[x[1]], x[0]))


def all_hold_line(alert_state: Optional[dict], n_open: Optional[int] = None) -> str:
    """Raden när inget ska säljas. Larmtillståndet bedömer bara Momentum-positionerna (alert_scan) — säg det, och
    säg 'saknas' när tillståndet inte gick att läsa i stället för att påstå HÅLL."""
    if not isinstance(alert_state, dict):
        return "Säljläget saknas — larmtillståndet gick inte att läsa"
    n = len((((alert_state or {}).get("swing") or {}).get("verdicts")) or {})
    other = f" · övriga {n_open - n} positioner bedöms på sina egna sidor" if n_open and n_open > n else ""
    return (f"Alla {n} Momentum-positioner: HÅLL" if n else "Inga Momentum-positioner bedöms") + other


RELOAD_MIN = 30


def needs_reload(loaded: Optional[str], now=None) -> bool:
    """Läs om när datan hämtades en annan dag eller för mer än RELOAD_MIN minuter sedan."""
    if not loaded:
        return False
    try:
        t = pd.Timestamp(loaded)
    except Exception:
        return True
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now()
    return t.date() != now.date() or (now - t) > pd.Timedelta(minutes=RELOAD_MIN)


def load(loader=None) -> dict:
    """Det sparade, aldrig ett fel. Thors lista byggs ur samma blobbar som Thor läser."""
    out: dict = {}
    if loader is None:
        try:
            from gist_storage import load_blob as loader
        except Exception:
            loader = None
    for k, name in (("board", "valhall_board.json"), ("alert_state", "alert_state.json"),
                    ("alert_log", "alert_log.json"), ("risk", "market_risk.json")):
        try:
            out[k] = loader(name, None) if loader else None
        except Exception:
            out[k] = None
    try:
        from thor import ui as tui
        out["thor"] = tui.rebuild(tui.load(loader))
    except Exception:
        out["thor"] = {"rows": []}
    try:
        from odin import laget
        lg = laget.load()
        out["alloc"], out["regime"] = lg.get("alloc"), lg.get("regime")
    except Exception:
        out["alloc"], out["regime"] = None, None
    try:
        import positions
        out["n_open"] = len(positions.all_positions())
    except Exception:
        out["n_open"] = None
    try:
        from portfolio import mot_malet as mm
        g = mm.load()
        out["paper"], out["trades"] = g.get("paper"), g.get("trades")
    except Exception:
        out["paper"], out["trades"] = None, []
    out["loaded"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return out
