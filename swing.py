"""
swing.py — Swing (momentum) veckorutin, nativ Streamlit-flik.

Portad från en fristående React/Tailwind-spec till panelens Python/Streamlit-
stack. Hela veckorutinen på ett ställe:

  1. Marknadsfilter  — OMXSPI vs MA200 (manuell toggle). Rött låser nya köp.
  2. Veckochecklista — 5 moment, nollställs automatiskt varje ny ISO-vecka.
  3. Bevakningslista — topp 20 med setup A/B; köp endast via vald setup.
  4. Positioner      — auto-stop −10 %, live P/L, säljregel-flaggor → "SÄLJ",
                       +20 %-halvsälj-påminnelse, max 8 positioner.
  5. Statistik       — vinstandel, payoff-kvot, snitt från stängda affärer.

Lagring: samma modell som holdings — GitHub Gist (egen fil "swing_data.json")
med lokal fallback, cachad per session (en nätverkshämtning, inte per rerun).
"""

from __future__ import annotations

import streamlit as st
from datetime import date
from typing import Optional

# ── Persistens (Gist + lokal fallback, som holdings) ──────────────────────────
try:
    from gist_storage import load_blob as _blob_load, save_blob as _blob_save
    _HAS_GIST = True
except Exception:
    _HAS_GIST = False

_STORE_FILE = "swing_data.json"
_CACHE_KEY = "swing_data"

# ── Tema (mörkt, grön swing-accent — matchar panelens palett) ─────────────────
BG_CARD  = "#14141e"
BG_ALT   = "#1a1f25"
BORDER   = "#2a2a38"
TEXT     = "#e8e4dc"
DIM      = "#8a8578"
GREEN    = "#2d8a4e"
GREEN_BG = "#12351f"
RED      = "#c44545"
RED_BG   = "#3a1414"
AMBER    = "#d4943a"
AMBER_BG = "#3a2a10"

MAX_POSITIONS = 8

CHECKLIST = [
    "OMXSPI mot MA200 kontrollerad (filtret nedan uppdaterat)",
    "Screenern körd — topp 20 in i bevakningslistan",
    "Alla positioner kollade mot säljregel 1–3",
    "Setup A/B letad i topp 20-graferna (max 1–2 nya köp)",
    "Affärer loggade i journalen + stop lagd hos mäklaren",
]


# ── Hjälpare ──────────────────────────────────────────────────────────────────
def _iso_week() -> str:
    y, w, _ = date.today().isocalendar()
    return f"{y}-V{w:02d}"


def _today() -> str:
    return date.today().isoformat()


def _uid() -> str:
    # Deterministisk-nog unik nyckel utan att importera random (håller AppTest glad).
    import time
    return f"{int(time.time() * 1000):x}"


def _num(value, default: Optional[float] = None) -> Optional[float]:
    """Coerce till ändlig float; None/tom/icke-numerisk → default."""
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):
        return default
    return f


def _fmt(n, dec: int = 2) -> str:
    v = _num(n)
    return "–" if v is None else f"{v:.{dec}f}"


def _pct(n) -> str:
    v = _num(n)
    return "–" if v is None else f"{v * 100:.1f}%"


# ── Lagring ───────────────────────────────────────────────────────────────────
def _default() -> dict:
    return {
        "market":    {"aboveMA200": True, "checked": _today()},
        "checklist": {"week": _iso_week(), "done": []},
        "watchlist": [],
        "positions": [],
        "closed":    [],
    }


def _normalize(data: dict) -> dict:
    """Fyll saknade nycklar så gammal/ofullständig lagrad data aldrig kraschar."""
    base = _default()
    for k, v in base.items():
        data.setdefault(k, v)
    if not isinstance(data.get("market"), dict):
        data["market"] = base["market"]
    data["market"].setdefault("aboveMA200", True)
    data["market"].setdefault("checked", _today())
    if not isinstance(data.get("checklist"), dict):
        data["checklist"] = base["checklist"]
    data["checklist"].setdefault("week", _iso_week())
    data["checklist"].setdefault("done", [])
    for key in ("watchlist", "positions", "closed"):
        if not isinstance(data.get(key), list):
            data[key] = []
    return data


def _load() -> dict:
    """Ladda en gång per session (cachad i session_state)."""
    if _CACHE_KEY in st.session_state:
        return st.session_state[_CACHE_KEY]
    data = _blob_load(_STORE_FILE, None) if _HAS_GIST else None
    if not isinstance(data, dict):
        data = _default()
    _normalize(data)
    st.session_state[_CACHE_KEY] = data
    return data


def _save(data: dict) -> None:
    st.session_state[_CACHE_KEY] = data
    if _HAS_GIST:
        _blob_save(_STORE_FILE, data)


# ── Entry point ───────────────────────────────────────────────────────────────
def render_swing_page() -> None:
    """Huvud-entry point för Swing-fliken (anropas från wolf_panel.py)."""
    try:
        data = _load()

        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:baseline;flex-wrap:wrap;gap:8px;'>"
            f"<h1 style='color:{TEXT};margin:0;letter-spacing:0.06em;'>"
            f"Swing <span style='color:{GREEN};'>· momentum</span></h1>"
            f"<span style='color:{DIM};font-size:0.85rem;'>{_iso_week()} · {_today()}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        buy_locked = not data["market"].get("aboveMA200", True)

        _market_filter(data)
        _weekly_checklist(data)
        _positions(data)
        _watchlist(data, buy_locked)
        _stats(data)
    except Exception as e:  # panelen ska aldrig ta ner hela appen
        st.error(f"Swing-fliken kunde inte renderas: {e}")


# ── 1. Marknadsfilter ─────────────────────────────────────────────────────────
def _market_filter(data: dict) -> None:
    m = data["market"]
    ok = m.get("aboveMA200", True)
    bg, bd = (GREEN_BG, GREEN) if ok else (RED_BG, RED)
    txt = ("Index över MA200 — nya köp tillåtna enligt reglerna."
           if ok else
           "Index UNDER MA200 — INGA nya köp denna vecka. Hantera endast exits.")
    st.markdown(
        f"<div style='background:{bg};border:1px solid {bd};border-radius:12px;"
        f"padding:14px;margin-bottom:12px;'>"
        f"<div style='font-weight:700;color:{TEXT};'>Marknadsfilter — OMXSPI vs MA200</div>"
        f"<div style='color:{DIM};font-size:0.85rem;margin-top:2px;'>{txt}</div>"
        f"<div style='color:{DIM};font-size:0.72rem;margin-top:4px;'>"
        f"Senast kollad: {m.get('checked', '–')}</div></div>",
        unsafe_allow_html=True,
    )
    label = "ÖVER MA200 ✓ — klicka om index fallit under" if ok else "UNDER MA200 ✕ — klicka om index stigit över"
    if st.button(label, key="swing_market_toggle"):
        m["aboveMA200"] = not ok
        m["checked"] = _today()
        _save(data)
        st.rerun()


# ── 2. Veckochecklista ────────────────────────────────────────────────────────
def _weekly_checklist(data: dict) -> None:
    cl = data["checklist"]
    # Nollställ automatiskt vid ny ISO-vecka
    if cl.get("week") != _iso_week():
        cl["week"] = _iso_week()
        cl["done"] = []
        _save(data)

    done = set(cl.get("done", []))
    complete = len(done) == len(CHECKLIST)
    head_c = GREEN if complete else DIM
    st.markdown(
        f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:12px;"
        f"padding:14px 14px 4px 14px;margin-bottom:12px;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<span style='font-weight:700;color:{TEXT};'>Veckorutinen ({cl.get('week')})</span>"
        f"<span style='color:{head_c};font-size:0.85rem;'>{len(done)}/{len(CHECKLIST)}"
        f"{' — klart för veckan ✓' if complete else ''}</span></div></div>",
        unsafe_allow_html=True,
    )
    changed = False
    for i, item in enumerate(CHECKLIST):
        checked = st.checkbox(item, value=(i in done), key=f"swing_cl_{i}")
        if checked and i not in done:
            done.add(i); changed = True
        elif not checked and i in done:
            done.discard(i); changed = True
    if changed:
        cl["done"] = sorted(done)
        _save(data)
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)


# ── 3. Positioner ─────────────────────────────────────────────────────────────
def _close_position(data: dict, p: dict, reason: str) -> None:
    entry = _num(p.get("entry"))
    cur = _num(p.get("current"))
    if entry is None or cur is None or entry <= 0:
        st.warning("Fyll i entry och aktuell kurs först.")
        return
    data["closed"].append({
        "id": p.get("id", _uid()), "ticker": p.get("ticker", "?"),
        "entry": entry, "exit": cur, "entryDate": p.get("date", ""),
        "exitDate": _today(), "ret": cur / entry - 1, "reason": reason,
    })
    data["positions"] = [x for x in data["positions"] if x.get("id") != p.get("id")]
    _save(data)


def _positions(data: dict) -> None:
    positions = data["positions"]
    st.markdown(
        f"<div style='font-weight:700;color:{TEXT};margin-bottom:6px;'>"
        f"Positioner ({len(positions)}/6–8)</div>",
        unsafe_allow_html=True,
    )
    if not positions:
        st.caption("Inga öppna positioner. Kontanter är också en position.")

    for p in positions:
        entry = _num(p.get("entry"), 0.0)
        cur = _num(p.get("current"), 0.0)
        stop = entry * 0.9 if entry > 0 else 0.0
        ret = (cur / entry - 1) if (entry > 0 and cur > 0) else None
        stop_hit = cur > 0 and cur <= stop
        sell = bool(stop_hit or p.get("belowMA50") or p.get("outOfRank"))
        take_half = ret is not None and ret >= 0.2 and not p.get("halfTaken")

        bd = RED if sell else BORDER
        badge = ""
        if sell:
            badge = (f"<span style='background:{RED};color:#fff;font-size:0.7rem;"
                     f"font-weight:700;padding:2px 8px;border-radius:4px;'>SÄLJ — regel utlöst</span>")
        elif take_half:
            badge = (f"<span style='background:{AMBER};color:#000;font-size:0.7rem;"
                     f"font-weight:700;padding:2px 8px;border-radius:4px;'>"
                     f"+20 % — sälj halva, stop till entry</span>")
        st.markdown(
            f"<div style='border:1px solid {bd};background:{RED_BG if sell else BG_ALT};"
            f"border-radius:10px;padding:10px 12px 4px 12px;margin-bottom:4px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"flex-wrap:wrap;gap:6px;'>"
            f"<span style='font-weight:700;color:{TEXT};'>{p.get('ticker','?')}"
            f"<span style='color:{DIM};font-size:0.72rem;margin-left:8px;'>"
            f"setup {p.get('setup','?')} · {p.get('date','')}</span></span>{badge}</div></div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        new_entry = c1.number_input("Entry", min_value=0.0, value=float(entry),
                                    step=0.5, key=f"swing_entry_{p['id']}")
        new_cur = c2.number_input("Kurs nu", min_value=0.0, value=float(cur),
                                  step=0.5, key=f"swing_cur_{p['id']}")
        c3.metric("Stop (−10 %)", _fmt(stop), delta="hit" if stop_hit else None,
                  delta_color="inverse")
        c4.metric("P/L", _pct(ret))
        c5.metric("Mot stop", _pct(cur / stop - 1) if (stop > 0 and cur > 0) else "–")
        if new_entry != entry or new_cur != cur:
            p["entry"] = new_entry
            p["current"] = new_cur
            _save(data)

        f1, f2, f3, f4 = st.columns([1.4, 1.2, 1.2, 1])
        b_ma50 = f1.checkbox("Stängt under MA50 (regel 1)", value=bool(p.get("belowMA50")),
                             key=f"swing_ma50_{p['id']}")
        b_rank = f2.checkbox("Ur topp 40 (regel 3)", value=bool(p.get("outOfRank")),
                             key=f"swing_rank_{p['id']}")
        b_half = f3.checkbox("Halva såld vid +20 %", value=bool(p.get("halfTaken")),
                             key=f"swing_half_{p['id']}")
        if (b_ma50 != bool(p.get("belowMA50")) or b_rank != bool(p.get("outOfRank"))
                or b_half != bool(p.get("halfTaken"))):
            p["belowMA50"], p["outOfRank"], p["halfTaken"] = b_ma50, b_rank, b_half
            _save(data)
        if f4.button("Stäng", key=f"swing_close_{p['id']}"):
            reason = ("stop -10%" if stop_hit else "under MA50" if p.get("belowMA50")
                      else "ur topp 40" if p.get("outOfRank") else "manuell")
            _close_position(data, p, reason)
            st.rerun()
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)


# ── 4. Bevakningslista ────────────────────────────────────────────────────────
def _promote(data: dict, w: dict, entry_price: float, buy_locked: bool) -> None:
    if buy_locked:
        st.warning("Marknadsfiltret är rött — inga nya köp."); return
    if len(data["positions"]) >= MAX_POSITIONS:
        st.warning(f"Max {MAX_POSITIONS} positioner."); return
    if entry_price is None or entry_price <= 0:
        st.warning("Ange en entry-kurs > 0 innan köp."); return
    data["positions"].append({
        "id": _uid(), "ticker": w.get("ticker", "?"),
        "entry": entry_price, "current": entry_price,
        "setup": w.get("setup") if w.get("setup") != "väntar" else "?",
        "date": _today(), "belowMA50": False, "outOfRank": False, "halfTaken": False,
    })
    data["watchlist"] = [x for x in data["watchlist"] if x.get("id") != w.get("id")]
    _save(data)


def _watchlist(data: dict, buy_locked: bool) -> None:
    lock_note = (f" <span style='color:{RED};font-size:0.75rem;'>(köp låsta — filtret rött)</span>"
                 if buy_locked else "")
    st.markdown(
        f"<div style='font-weight:700;color:{TEXT};margin-bottom:6px;'>"
        f"Bevakningslista — topp 20 från screenern{lock_note}</div>",
        unsafe_allow_html=True,
    )

    a1, a2, a3, a4 = st.columns([1.2, 1, 2, 0.8])
    tkr = a1.text_input("Ticker", key="swing_add_ticker")
    mom = a2.text_input("6-mån %", key="swing_add_mom")
    note = a3.text_input("Notering (t.ex. 'nära MA20')", key="swing_add_note")
    if a4.button("Lägg till", key="swing_add_btn"):
        if tkr.strip():
            data["watchlist"].append({
                "id": _uid(), "ticker": tkr.upper().strip(),
                "mom6": mom.strip(), "note": note.strip(),
                "setup": "väntar", "added": _today(),
            })
            _save(data)
            st.rerun()

    if not data["watchlist"]:
        st.caption("Tom — kör screenern och lägg in veckans topp 20.")
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        return

    _setups = ["väntar", "A", "B"]
    _setup_label = {"väntar": "väntar på setup", "A": "Setup A — pullback", "B": "Setup B — utbrott"}
    for w in data["watchlist"]:
        c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 2, 1.1, 1.1, 0.5])
        c1.markdown(f"**{w.get('ticker','?')}**")
        _m = w.get("mom6", "")
        c2.markdown(f"<span style='color:{DIM};'>{('+' + str(_m) + '%') if _m else ''}</span>",
                    unsafe_allow_html=True)
        cur_setup = w.get("setup", "väntar")
        new_setup = c3.selectbox("setup", _setups, index=_setups.index(cur_setup)
                                 if cur_setup in _setups else 0,
                                 format_func=lambda s: _setup_label[s],
                                 key=f"swing_setup_{w['id']}", label_visibility="collapsed")
        if new_setup != cur_setup:
            w["setup"] = new_setup
            _save(data)
            st.rerun()
        entry_price = c4.number_input("entry", min_value=0.0, value=0.0, step=0.5,
                                      key=f"swing_promo_entry_{w['id']}",
                                      label_visibility="collapsed")
        can_buy = (w.get("setup") != "väntar") and not buy_locked
        if c5.button("KÖP →", key=f"swing_buy_{w['id']}", disabled=not can_buy):
            _promote(data, w, _num(entry_price), buy_locked)
            st.rerun()
        if c6.button("✕", key=f"swing_del_{w['id']}"):
            data["watchlist"] = [x for x in data["watchlist"] if x.get("id") != w.get("id")]
            _save(data)
            st.rerun()
        if w.get("note"):
            st.markdown(f"<div style='color:{DIM};font-size:0.75rem;margin:-4px 0 4px 4px;'>"
                        f"{w['note']}</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)


# ── 5. Statistik ──────────────────────────────────────────────────────────────
def _stats(data: dict) -> None:
    closed = data["closed"]
    st.markdown(
        f"<div style='font-weight:700;color:{TEXT};margin-bottom:6px;'>"
        f"Statistik — stängda affärer</div>",
        unsafe_allow_html=True,
    )
    if not closed:
        st.caption("Inga stängda affärer ännu.")
    else:
        rets = [_num(t.get("ret"), 0.0) for t in closed]
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        avg_w = sum(wins) / len(wins) if wins else 0.0
        avg_l = sum(losses) / len(losses) if losses else 0.0
        win_rate = len(wins) / len(rets) if rets else 0.0
        payoff = (avg_w / abs(avg_l)) if avg_l != 0 else None
        avg = sum(rets) / len(rets) if rets else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Affärer", len(closed))
        k2.metric("Vinstandel", _pct(win_rate), help="40–55 % normalt")
        k3.metric("Payoff-kvot", _fmt(payoff), help="mål > 2,0")
        k4.metric("Snitt/affär", _pct(avg))

        with st.expander(f"Visa alla ({len(closed)}) / rensa", expanded=False):
            for t in reversed(closed):
                r = _num(t.get("ret"), 0.0)
                col = GREEN if r > 0 else RED
                st.markdown(
                    f"<div style='font-size:0.8rem;'>"
                    f"<b>{t.get('ticker','?')}</b> "
                    f"<span style='color:{DIM};'>{t.get('entryDate','')} → {t.get('exitDate','')}</span> "
                    f"<span style='color:{col};'>{_pct(r)}</span> "
                    f"<span style='color:{DIM};'>({t.get('reason','')})</span></div>",
                    unsafe_allow_html=True,
                )
            if st.button("Rensa historiken", key="swing_clear_stats"):
                data["closed"] = []
                _save(data)
                st.rerun()

    st.caption("Kom ihåg: journalen i Excel är fortfarande huvudboken — detta är "
               "panelens snabbvy. Under 15–20 affärer: dra inga slutsatser.")
