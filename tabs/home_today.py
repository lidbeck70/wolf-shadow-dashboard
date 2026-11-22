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

Överst Dagens besked (LIDSKJALV §4.3, beslut 6): counts() räknar det som raderna ovan redan visar — rutorna i
köpfönstret, kvalarna, säljraderna och rutinerna — och besked() väljer meningen. Bara räknat: inget nytt ord,
ingen ny regel, ingen aritmetik på kurser. size_parts() är Storlek-radens tal för kvalarnas mono-rad.

Rena funktioner här; tabs/home.py ritar.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

TOP_N = 5
SE_TZ = "Europe/Stockholm"


def now_se() -> datetime:
    """Klockan i Sverige, utan tidszon. Servern (Streamlit Cloud) går på UTC — efter midnatt svensk tid var
    det annars fortfarande gårdagen på startsidan: datumet, rutinerna och vändningens dag."""
    try:
        return datetime.now(ZoneInfo(SE_TZ)).replace(tzinfo=None)
    except Exception:                      # ingen tidszonsdatabas — hellre serverns klocka än en krasch
        return datetime.now()


def _bdays(since, today) -> Optional[int]:
    try:
        return len(pd.bdate_range(pd.Timestamp(since), pd.Timestamp(today))) - 1
    except Exception:
        return None


def turn_cards(board: Optional[dict], today=None) -> list:
    """Rutorna i köpfönstret: [{label, where, kind, day, of, deep, long, etf}] — dag räknas från bekräftelsen."""
    plan = (board.get("turn_plan") if isinstance(board, dict) else None) or {}
    plan = plan if isinstance(plan, dict) else {}
    of = int(plan.get("window_days") or 5)
    today = today if today is not None else pd.Timestamp(now_se()).normalize()
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
    turns = board.get("turns") if isinstance(board, dict) else None
    return sum(1 for t in (turns if isinstance(turns, list) else []) if isinstance(t, dict) and not t.get("confirmed"))


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


def _sized(c, capital: float) -> Optional[dict]:
    """Storleken som Storlek-raden visar: sk.size med strategins risk och tak, halverad i GUL regim (strategins
    egen regel). None utan giltiga nivåer."""
    from thor import storlek as sk
    ccy, fx = sk.fx_for(c.ticker)
    sz = sk.size(c.entry, c.stop, capital, sk.risk_pct(c.strategy), sk.cap_pct(c.strategy), fx=fx, currency=ccy)
    if sz and "GUL regim" in (c.status or ""):
        sz = sk.halved(sz)
    return sz


def size_line(c, capital: float) -> str:
    from thor import registry as reg
    from thor import storlek as sk
    sz = _sized(c, capital)
    return sk.row_text(c.ticker, reg.label(c.strategy), sz) if sz else f"{c.ticker} · {reg.label(c.strategy)} · nivåer saknas"


def size_parts(c, capital: float) -> list:
    """Storlek-radens tal för kvalarnas mono-rad: ['120 st', 'risk 1 500 kr', 'tak 22 st'] — ur samma storlek som
    size_line (samma sz, ingen ny räkning). Utan risk-% och tak: köp och stopp. [] när nivåerna saknas."""
    sz = _sized(c, capital)
    if not sz:
        return []
    out = []
    if sz.get("risk_shares") is not None:
        out.append(f"{sz['risk_shares']} st")
    if sz.get("risk_amt") is not None:
        out.append(f"risk {sz['risk_amt']:,.0f} kr".replace(",", " "))
    if sz.get("cap_shares") is not None:
        out.append(f"tak {sz['cap_shares']} st")
    if not out:
        out = [f"köp {sz['entry']:.2f}", f"stopp {sz['stop']:.2f}"]
    if "halverat" in (sz.get("formula") or ""):
        out.append("halverat")
    return out


# swing_verdict sparar sina egna strängar (SÄLJ, 'DELVINST — sälj halva', BEVAKA) — nyckla på dem (granskning 2, D3)
SELL_WORDS = {"SÄLJ": "SÄLJ", "SELL": "SÄLJ", "DELVINST — SÄLJ HALVA": "MINSKA", "DELVINST": "MINSKA",
              "PARTIAL": "MINSKA", "REDUCE": "MINSKA", "BEVAKA": "VAKSAM", "WATCH": "VAKSAM"}


def holdings_lines(alert_state: Optional[dict], n_open: Optional[int] = None) -> list:
    """[(ticker, ord, skäl)] för innehav med SÄLJ/MINSKA/VAKSAM ur larmtillståndets swing-verdikt."""
    swing = (alert_state or {}).get("swing") if isinstance(alert_state, dict) else None
    verdicts = (swing or {}).get("verdicts") if isinstance(swing, dict) else None
    verdicts = verdicts if isinstance(verdicts, dict) else {}          # trasig form ger ingen rad, aldrig ett fel
    out = []
    for t, v in sorted(verdicts.items()):
        v = v if isinstance(v, dict) else {}
        w = SELL_WORDS.get(str(v.get("verdict") or "").upper())
        if w:
            out.append((t, w, str((v or {}).get("reason") or "")))
    order = {"SÄLJ": 0, "MINSKA": 1, "VAKSAM": 2}
    return sorted(out, key=lambda x: (order[x[1]], x[0]))


def all_hold_line(alert_state: Optional[dict], n_open: Optional[int] = None) -> str:
    """Raden när inget ska säljas. Larmtillståndet bedömer bara Momentum-positionerna (alert_scan) — säg det, och
    säg 'saknas' när tillståndet inte gick att läsa i stället för att påstå HÅLL."""
    if not isinstance(alert_state, dict):
        return "Säljläget saknas — larmtillståndet gick inte att läsa"
    swing = alert_state.get("swing")
    v = (swing.get("verdicts") if isinstance(swing, dict) else None) or {}
    v = v if isinstance(v, dict) else {}
    n = len(v)
    hold = sum(1 for x in v.values() if str((x or {}).get("verdict") or "").upper() in ("HÅLL", "HOLD"))
    unknown = n - hold
    other = f" · övriga {n_open - n} positioner bedöms på sina egna sidor" if n_open and n_open > n else ""
    head = f"Alla {n} Momentum-positioner: HÅLL" if n and not unknown else \
        (f"{hold} av {n} Momentum-positioner: HÅLL · {unknown} saknar läge" if n else "Inga Momentum-positioner bedöms")
    return head + other


# ── Dagens besked (LIDSKJALV §4.3) ───────────────────────────────────────────
GIST_HEAD = "Gisten gick inte att läsa"
GIST_MISSING = f"Saknas — {GIST_HEAD}"
NOTHING_TODAY = "Inga KÖP, SÄLJ eller vändningar i dag"
MAP_MISSING = "Saknas — Valhalls karta gick inte att läsa"     # vändningen okänd: grått, aldrig "Ingen vändning"


def gist_error(d: Optional[dict]) -> Optional[str]:
    """Thors fel när ingen enda körning gick att läsa (thor.candidates sätter errors['gist']), annars None."""
    thor = (d or {}).get("thor") if isinstance(d, dict) else None
    errors = thor.get("errors") if isinstance(thor, dict) else None
    err = errors.get("gist") if isinstance(errors, dict) else None
    return str(err) if err else None


def gist_reason(err) -> str:
    """Felets orsak utan inledningen 'Gisten gick inte att läsa' — meningen ovanför säger den redan, en gång."""
    s = str(err or "").strip()
    if s.startswith(GIST_HEAD):
        s = s[len(GIST_HEAD):].strip(" —–-:·.")
    return s


_ASK = object()


def counts(d: Optional[dict], unlocked: bool = True, today=None, due=_ASK) -> dict:
    """De fyra rutornas antal — bara ur det sidan redan visar: rutorna i köpfönstret (turn_cards), kvalarna
    (qualifiers), säljraderna (holdings_lines) och Årshjulets rutiner (routines.due_today). Ingen ny regel, ingen
    aritmetik på kurser. None = okänt och visas Saknas, aldrig en tyst nolla: kartan saknas → vändning okänd,
    Gisten oläslig → KÖP okänt, innehaven låsta eller larmtillståndet oläsligt → SÄLJ okänt.
    due = dagens rutiner om sidan redan läst dem (en lista, eller None när de inte gick att läsa).

    {turn, on_way, buy, watch, sell, vaksam, due, locked, gist}"""
    d = d if isinstance(d, dict) else {}
    day = pd.Timestamp(today if today is not None else now_se()).normalize()
    board, gist = d.get("board"), gist_error(d)
    thor = d.get("thor") if isinstance(d.get("thor"), dict) else {}
    out = {"turn": len(turn_cards(board, day)) if isinstance(board, dict) else None,
           "on_way": on_the_way(board), "buy": None, "watch": None, "sell": None, "vaksam": None, "due": None,
           "locked": not unlocked, "gist": gist}
    if not gist:
        out["buy"], out["watch"], _top = qualifiers(thor.get("rows"))
    if unlocked and isinstance(d.get("alert_state"), dict):
        words = [w for _t, w, _why in holdings_lines(d.get("alert_state"))]
        out["sell"] = sum(1 for w in words if w in ("SÄLJ", "MINSKA"))
        out["vaksam"] = words.count("VAKSAM")
    if due is _ASK:
        try:
            import routines
            due = routines.due_today(day.date())
        except Exception:
            due = None
    out["due"] = len(due) if isinstance(due, (list, tuple)) else None
    return out


def besked(n: dict, cards: list, board: Optional[dict]) -> tuple:
    """(meningen, saknas?) under DAGENS BESKED: Gisten oläslig → GIST_MISSING (grå); ett öppet köpfönster → första
    turn_headline; inga KÖP, SÄLJ, vändningar eller vändningar på väg → NOTHING_TODAY; annars no_turn_line.
    Säger aldrig 'inga' om ett antal som är okänt (None)."""
    if n.get("gist"):
        return GIST_MISSING, True
    if cards:
        return turn_headline(cards[0]), False
    if n.get("turn") is None:                       # kartan saknas: säg inte "Ingen vändning" bredvid en Saknas-ruta
        return MAP_MISSING, True
    if all(n.get(k) == 0 for k in ("turn", "on_way", "buy", "sell")):
        return NOTHING_TODAY, False
    return no_turn_line(board), False


RELOAD_MIN = 30


def needs_reload(loaded: Optional[str], now=None) -> bool:
    """Läs om när datan hämtades en annan dag eller för mer än RELOAD_MIN minuter sedan."""
    if not loaded:
        return False
    try:
        t = pd.Timestamp(loaded)
    except Exception:
        return True
    now = pd.Timestamp(now) if now is not None else pd.Timestamp(now_se())
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
    out["loaded"] = now_se().strftime("%Y-%m-%d %H:%M")          # samma klocka som needs_reload
    return out
