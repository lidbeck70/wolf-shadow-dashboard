"""
insider.py — Insiderbevakaren (ersätter insiderbevakaren.xlsx).

Poängsätter varje insynskluster 0–10, kör det genom kvalitetsgrinden och den
tekniska triggern, och säger vad det faktiskt betyder: brus, bevaka, kör
grinden, stoppad, vänta på trigger, eller köp.

Formlerna är arkets, cell för cell:

  poäng   = J-kolumnen  (kluster + roll + belopp + tre 0/1-fält)
  status  = N-kolumnen  (nästlade IF, ordningen är regeln)
  vs snitt= P-kolumnen  (kurs_nu / klustersnitt − 1)
  stopp   = Q-kolumnen  (klustersnitt × 0,85)

Endast riktiga marknadsköp räknas — inte optionslösen, tilldelningsprogram,
arv eller interna omflyttningar.
"""

from __future__ import annotations

import streamlit as st
from dataclasses import dataclass
from datetime import date
from typing import Optional

import csv_export
import storage
import storage_ui

_CACHE_KEY = "insider_data"
STORE = "insider"   # data/insider.json

TEXT, DIM = "#e8e4dc", "#8a8578"
GREEN, AMBER, RED, CYAN = "#2d8a4e", "#d4943a", "#c44545", "#00E5FF"
BG_ALT, BORDER = "#1a1f25", "#2a2a38"

# ── Rollerna (arkets dropdown i kolumn E) ────────────────────────────────────
ROLE_TOP = "VD/CFO"
ROLE_BOARD = "Styrelse"
ROLE_OTHER = "Övrig"
ROLES = (ROLE_TOP, ROLE_BOARD, ROLE_OTHER)

# ── Poängsättningen (kolumn J) ───────────────────────────────────────────────
CLUSTER_MIN = 3           # 3+ insiders inom 30 dgr = 3p, exakt 2 = 2p
ROLE_POINTS = {ROLE_TOP: 2, ROLE_BOARD: 1, ROLE_OTHER: 0}
AMOUNT_HIGH, AMOUNT_MID = 1000.0, 500.0     # tkr
MAX_SCORE = 10

BOOL_FIELDS = (
    ("okar_25", "Ökar innehav > 25 %",
     "Köpet ökar köparens eget innehav med mer än en fjärdedel. "
     "Kolla innehavet i insynsfliken."),
    ("efter_fall", "Efter fall > 20 % i friskt bolag",
     "Kursen har fallit mer än 20 % och F-score ≥ 5 — de köper svaghet, "
     "inte styrka."),
    ("aterkommande", "Återkommande köpare",
     "Samma person har köpt flera gånger. Syns i bolagets insynshistorik."),
)

# ── Statusflödet (kolumn N — ordningen ÄR regeln) ────────────────────────────
S_NOISE = "Ignorera — brus"
S_WATCH = "Bevaka — vänta på fler köp"
S_RUN_GATE = "Kör kvalitetsgrinden!"
S_GATE_FAIL = "Stoppad i kvalitetsgrinden"
S_WAIT_TRIGGER = "Väntar på teknisk trigger"
S_BUY = "KÖP — logga i journalen"

STATUS_COLOR = {S_NOISE: DIM, S_WATCH: AMBER, S_RUN_GATE: CYAN,
                S_GATE_FAIL: RED, S_WAIT_TRIGGER: AMBER, S_BUY: GREEN}

SCORE_NOISE_MAX = 5       # under 5 = brus
SCORE_WATCH_MAX = 7       # 5–6 = bevaka, 7+ går vidare till grinden

# ── Grind och trigger ────────────────────────────────────────────────────────
GATE_YES, GATE_NO, GATE_BLANK = "Ja", "Nej", ""
TRIGGERS = ("A", "B", "C", "Nej")

STOP_FRAC = 0.85          # stopp = klustersnitt × 0,85 (−15 %)
CHASE_PCT = 30.0          # över +30 % mot klustersnittet är edgen förbrukad

GATE_CRITERIA = (
    "Börsvärde > 300 MSEK",
    "F-score ≥ 5",
    "Nettoskuld/EBITDA < 2 (eller nettokassa)",
    "Positivt FCF eller tydlig väg dit",
    "Ej strukturellt fallande omsättning",
)

TRIGGER_CRITERIA = (
    ("A. Stabilisering", "Stängning över MA20, MA20 planat eller vänt",
     "Grafen i Börsdata"),
    ("B. Bekräftelse", "Positiv 1-månadsutveckling + kurs över klustersnittet",
     "Kolumnen 'Vs kluster' visar det"),
    ("C. Fundamental", "Första rapporten efter köpen bekräftar",
     "Köp på rapportdagen"),
)

SELL_RULES = (
    ("1. Säljkluster", "2+ insiders säljer (ej småposter)",
     "Bolagets insynsflik, veckokoll"),
    ("2. Stopp", "−15 % under klustrets snittkurs", "Lägg som order direkt"),
    ("3. Grinden bryts", "F-score < 4 eller skulden drar iväg", "Kvartalsvis koll"),
    ("4. Tes utspelad", "+50–100 % och värderingen ikapp",
     "Jämför sektorns multiplar"),
    ("5. Tidsstopp", "18 månader utan materialisering", "Upptäckt + 18 mån"),
)


@dataclass(frozen=True)
class Signal:
    """En poängsatt insynshändelse — enbart det som matas in."""
    id: str
    ticker: str = ""
    name: str = ""
    found: str = ""
    insiders: Optional[float] = None
    role: str = ROLE_OTHER
    amount: Optional[float] = None       # tkr
    okar_25: bool = False
    efter_fall: bool = False
    aterkommande: bool = False
    cluster_avg: Optional[float] = None
    gate: str = GATE_BLANK
    trigger: str = ""
    price_now: Optional[float] = None
    comment: str = ""


# ── Rena beräkningar ─────────────────────────────────────────────────────────
def _num(value, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def cluster_points(insiders) -> int:
    """3+ insiders inom 30 dagar = 3p, exakt 2 = 2p, ensam köpare = 0p."""
    n = _num(insiders, 0) or 0
    if n >= CLUSTER_MIN:
        return 3
    return 2 if n == 2 else 0


def role_points(role: str) -> int:
    return ROLE_POINTS.get(role, 0)


def amount_points(amount) -> int:
    """> 1 MSEK = 2p, 500 tkr–1 MSEK = 1p, under = 0p."""
    a = _num(amount)
    if a is None:
        return 0
    if a >= AMOUNT_HIGH:
        return 2
    return 1 if a >= AMOUNT_MID else 0


def score(sig: dict) -> Optional[int]:
    """Insiderpoäng 0–10, eller None när arket skulle visa tom cell.

    Arket: =IF(OR(D="",E="",F=""),"", ...) — antal insiders, roll och belopp
    måste alla vara ifyllda innan poängen betyder något.
    """
    if (_num(sig.get("insiders")) is None or not sig.get("role")
            or _num(sig.get("amount")) is None):
        return None
    return (cluster_points(sig.get("insiders"))
            + role_points(sig.get("role"))
            + amount_points(sig.get("amount"))
            + (1 if sig.get("okar_25") else 0)
            + (1 if sig.get("efter_fall") else 0)
            + (1 if sig.get("aterkommande") else 0))


def status(sig: dict) -> Optional[str]:
    """Statusflödet ur kolumn N. Ordningen är regeln, inte en presentation."""
    pts = score(sig)
    if pts is None:
        return None
    if pts < SCORE_NOISE_MAX:
        return S_NOISE
    if pts < SCORE_WATCH_MAX:
        return S_WATCH
    gate = sig.get("gate", GATE_BLANK)
    if gate == GATE_NO:
        return S_GATE_FAIL
    if gate != GATE_YES:
        return S_RUN_GATE
    if sig.get("trigger", "") in ("", "Nej"):
        return S_WAIT_TRIGGER
    return S_BUY


def vs_cluster(sig: dict) -> Optional[float]:
    """Kurs nu mot klustrets snittkurs, i procent."""
    now, avg = _num(sig.get("price_now")), _num(sig.get("cluster_avg"))
    if now is None or avg is None or avg == 0:
        return None
    # Avrundad: en kurs exakt 30 % över snittet blir 30.000000000000004 i
    # binär flyttalsform, och passa-regeln säger "mer än 30 %".
    return round((now / avg - 1) * 100, 6)


def stop_price(sig: dict) -> Optional[float]:
    """Stoppen: klustersnittet × 0,85."""
    avg = _num(sig.get("cluster_avg"))
    return None if avg is None else avg * STOP_FRAC


def is_chase(sig: dict) -> bool:
    """Passa-regeln: mer än 30 % över klustersnittet = edgen förbrukad."""
    vs = vs_cluster(sig)
    return vs is not None and vs > CHASE_PCT


def ranked(signals: list) -> list:
    """Signalerna med poäng och status, högst poäng först."""
    out = []
    for s in signals or []:
        out.append({"signal": s, "score": score(s), "status": status(s),
                    "vs_cluster": vs_cluster(s), "stop": stop_price(s),
                    "chase": is_chase(s)})
    out.sort(key=lambda r: (-(r["score"] if r["score"] is not None else -1),
                            (r["signal"].get("ticker") or "")))
    return out


def buy_candidates(signals: list) -> list:
    """De som passerat hela flödet — status KÖP och ingen passa-flagga."""
    return [r for r in ranked(signals)
            if r["status"] == S_BUY and not r["chase"]]


# ── Lagring ──────────────────────────────────────────────────────────────────
def _today() -> str:
    return date.today().isoformat()


def _uid() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


def _default() -> dict:
    return {"signals": []}


def _load() -> dict:
    """Laddas EN gång per session; därefter äger sessionen sanningen."""
    data = storage.session_load(STORE, _default(),
                                legacy_file="insider_data.json")
    if not isinstance(data, dict):
        data = _default()
        st.session_state[STORE] = data
    for k, v in _default().items():
        data.setdefault(k, v)
    return data


def _save(data: dict) -> None:
    """Skriver till sessionen. Persistensen sker via 💾 Spara.

    Medvetet ingen nätverkstrafik här: en commit per
    tangenttryckning skulle slå i GitHubs rate limits, och en
    tyst misslyckad sådan var precis den bugg det här ersätter.
    """
    st.session_state[STORE] = data


# ── UI ───────────────────────────────────────────────────────────────────────
def render_insider_page() -> None:
    data = _load()
    storage_ui.save_bar(STORE, "Insiderbevakaren")
    st.markdown(
        f"<div style='text-align:center;padding:10px 0 4px;'>"
        f"<h2 style='color:{CYAN};letter-spacing:0.12em;margin:0;'>"
        f"INSIDERBEVAKAREN</h2>"
        f"<p style='color:{DIM};font-size:0.78rem;margin:6px 0 0;'>"
        f"Poängsätt varje kluster. Endast riktiga marknadsköp räknas — inte "
        f"optionslösen, program, arv eller interna omflyttningar.</p></div>",
        unsafe_allow_html=True)

    _summary(data)
    _export(data)
    _new_signal(data)
    _signals(data)
    _criteria()


def _summary(data: dict) -> None:
    rows = ranked(data.get("signals", []))
    buys = buy_candidates(data.get("signals", []))
    watch = [r for r in rows if r["status"] in (S_WATCH, S_RUN_GATE,
                                                S_WAIT_TRIGGER)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Signaler", len(rows))
    c2.metric("I flödet", len(watch),
              help="Bevakas, väntar på grind eller på teknisk trigger.")
    c3.metric("Köpklara", len(buys),
              help="Poäng ≥ 7, grinden OK, teknisk trigger satt — och inte "
                   "mer än 30 % över klustersnittet.")


CSV_COLUMNS = [
    ("found", "Upptäckt"), ("ticker", "Ticker"), ("name", "Bolag"),
    ("insiders", "Antal insiders 30 dgr"), ("role", "Högsta roll"),
    ("amount", "Belopp tot (tkr)"), ("okar_25", "Ökar innehav > 25 %"),
    ("efter_fall", "Efter fall > 20 %"), ("aterkommande", "Återkommande"),
    ("_score", "Insiderpoäng"), ("cluster_avg", "Klustrets snittkurs"),
    ("gate", "Kvalitetsgrind"), ("trigger", "Teknisk trigger"),
    ("_status", "Status"), ("price_now", "Kurs nu"),
    ("_vs", "Vs kluster %"), ("_stop", "Stopp (−15 %)"),
    ("comment", "Kommentar"),
]


def _export(data: dict) -> None:
    """Beräknade kolumner räknas här — lagringen håller bara inmatningar."""
    rows = []
    for r in ranked(data.get("signals", [])):
        s = r["signal"]
        rows.append({**s, "_score": r["score"], "_status": r["status"],
                     "_vs": None if r["vs_cluster"] is None
                            else round(r["vs_cluster"], 1),
                     "_stop": None if r["stop"] is None else round(r["stop"], 2)})
    csv_export.download_button(rows, CSV_COLUMNS, "insiderbevakaren",
                               key="csv_insider")


def _new_signal(data: dict) -> None:
    with st.expander("➕ Ny signal", expanded=not data.get("signals")):
        c1, c2, c3 = st.columns([1, 2, 1])
        ticker = c1.text_input("Ticker", key="ins_new_ticker")
        name = c2.text_input("Bolag", key="ins_new_name")
        found = c3.text_input("Upptäckt", value=_today(), key="ins_new_found")
        if st.button("Lägg till", key="ins_new_add"):
            if ticker.strip():
                data["signals"].append({
                    "id": _uid(), "ticker": ticker.strip().upper(),
                    "name": name.strip(), "found": found.strip() or _today(),
                    "role": ROLE_OTHER, "gate": GATE_BLANK, "trigger": "",
                })
                _save(data)
                st.rerun()
            else:
                st.warning("Ticker krävs.")


def _signals(data: dict) -> None:
    rows = ranked(data.get("signals", []))
    if not rows:
        st.caption("Inga signaler ännu. Insynsflödet ligger i Börsdata → "
                   "Holdings.")
        return

    for r in rows:
        sig = r["signal"]
        pts, stat = r["score"], r["status"]
        c = STATUS_COLOR.get(stat, DIM)
        head = (f"{sig.get('ticker','?')}"
                f"  ·  {pts if pts is not None else '–'}/{MAX_SCORE} p"
                f"  ·  {stat or 'ofullständig'}")
        with st.expander(head, expanded=False):
            _signal_body(data, sig, r, c)


def _signal_body(data: dict, sig: dict, r: dict, color: str) -> None:
    changed = False

    st.markdown(f"<b style='color:{TEXT};'>Poängsättning</b> "
                f"<span style='color:{DIM};font-size:0.78rem;'>"
                f"— kluster, roll och belopp måste alla vara ifyllda</span>",
                unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    ins = p1.number_input("Antal insiders 30 dgr", min_value=0, step=1,
                          value=int(_num(sig.get("insiders"), 0) or 0),
                          key=f"ins_n_{sig['id']}",
                          help=f"{CLUSTER_MIN}+ = 3p · 2 = 2p · ensam = 0p")
    role = p2.selectbox("Högsta roll", list(ROLES),
                        index=ROLES.index(sig.get("role")
                                          if sig.get("role") in ROLES
                                          else ROLE_OTHER),
                        key=f"ins_r_{sig['id']}",
                        help="VD/CFO = 2p · Styrelse = 1p · Övrig = 0p")
    amt = p3.number_input("Belopp totalt (tkr)", min_value=0.0, step=100.0,
                          value=float(_num(sig.get("amount"), 0.0) or 0.0),
                          key=f"ins_a_{sig['id']}",
                          help=f"≥ {AMOUNT_HIGH:g} tkr = 2p · "
                               f"≥ {AMOUNT_MID:g} tkr = 1p")
    if (ins != sig.get("insiders") or role != sig.get("role")
            or amt != sig.get("amount")):
        sig["insiders"], sig["role"], sig["amount"] = ins, role, amt
        changed = True

    bcols = st.columns(3)
    for (bkey, blabel, bhelp), col in zip(BOOL_FIELDS, bcols):
        v = col.checkbox(blabel, value=bool(sig.get(bkey)),
                         key=f"ins_b_{sig['id']}_{bkey}", help=bhelp)
        if v != bool(sig.get(bkey)):
            sig[bkey] = v
            changed = True

    # Poäng + status
    pts, stat = score(sig), status(sig)
    st.markdown(
        f"<div style='border:1px solid {color}55;background:{color}0d;"
        f"border-radius:8px;padding:10px 14px;margin:10px 0;'>"
        f"<span style='color:{color};font-weight:700;font-size:1.05rem;'>"
        f"{pts if pts is not None else '–'} / {MAX_SCORE}</span>"
        f"<span style='color:{color};font-weight:700;margin-left:12px;'>"
        f"{stat or 'Fyll i kluster, roll och belopp'}</span></div>",
        unsafe_allow_html=True)

    # Grind + trigger
    g1, g2 = st.columns(2)
    gate_opts = [GATE_BLANK, GATE_YES, GATE_NO]
    gate = g1.selectbox("Kvalitetsgrind OK?", gate_opts,
                        index=gate_opts.index(sig.get("gate")
                                              if sig.get("gate") in gate_opts
                                              else GATE_BLANK),
                        format_func=lambda x: x or "— ej körd —",
                        key=f"ins_g_{sig['id']}")
    trig_opts = [""] + list(TRIGGERS)
    trig = g2.selectbox("Teknisk trigger", trig_opts,
                        index=trig_opts.index(sig.get("trigger")
                                              if sig.get("trigger") in trig_opts
                                              else ""),
                        format_func=lambda x: x or "— ingen —",
                        key=f"ins_t_{sig['id']}")
    if gate != sig.get("gate") or trig != sig.get("trigger"):
        sig["gate"], sig["trigger"] = gate, trig
        changed = True

    # Kurser
    k1, k2, k3, k4 = st.columns(4)
    avg = k1.number_input("Klustrets snittkurs", min_value=0.0, step=0.5,
                          value=float(_num(sig.get("cluster_avg"), 0.0) or 0.0),
                          key=f"ins_ca_{sig['id']}")
    now = k2.number_input("Kurs nu", min_value=0.0, step=0.5,
                          value=float(_num(sig.get("price_now"), 0.0) or 0.0),
                          key=f"ins_pn_{sig['id']}")
    if avg != sig.get("cluster_avg") or now != sig.get("price_now"):
        sig["cluster_avg"], sig["price_now"] = avg, now
        changed = True

    vs, stop = vs_cluster(sig), stop_price(sig)
    k3.metric("Vs kluster", f"{vs:+.1f} %" if vs is not None else "–",
              help=f"Över +{CHASE_PCT:g} % är edgen förbrukad — passa.")
    k4.metric("Stopp (−15 %)", f"{stop:.2f}" if stop is not None else "–",
              help="Klustersnittet × 0,85. Lägg som order direkt vid köp.")

    if is_chase(sig):
        st.warning(f"Kursen är {vs:+.1f} % mot klustersnittet — över "
                   f"+{CHASE_PCT:g} % är edgen förbrukad. Passa.")

    com = st.text_input("Kommentar", value=sig.get("comment", ""),
                        key=f"ins_c_{sig['id']}")
    if com != sig.get("comment", ""):
        sig["comment"] = com
        changed = True

    if st.button("Ta bort signalen", key=f"ins_del_{sig['id']}"):
        data["signals"] = [s for s in data["signals"] if s["id"] != sig["id"]]
        _save(data)
        st.rerun()

    if changed:
        _save(data)


def _criteria() -> None:
    with st.expander("📋 Kriterier — grind, triggers och säljregler",
                     expanded=False):
        st.markdown(f"<b style='color:{TEXT};'>Kvalitetsgrinden</b> "
                    f"<span style='color:{DIM};font-size:0.78rem;'>"
                    f"— alla ska vara uppfyllda för 'Ja'</span>",
                    unsafe_allow_html=True)
        st.markdown("".join(
            f"<div style='color:{DIM};font-size:0.82rem;padding:2px 0 2px 16px;'>"
            f"· {c}</div>" for c in GATE_CRITERIA), unsafe_allow_html=True)

        st.markdown(f"<div style='height:10px;'></div>"
                    f"<b style='color:{TEXT};'>Teknisk trigger</b>",
                    unsafe_allow_html=True)
        for label, what, where in TRIGGER_CRITERIA:
            st.markdown(
                f"<div style='color:{TEXT};font-size:0.82rem;padding:3px 0;'>"
                f"<b>{label}</b> — {what} "
                f"<span style='color:{DIM};'>({where})</span></div>",
                unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:{AMBER};font-size:0.8rem;padding:6px 0;'>"
            f"Passa-regel: kurs mer än {CHASE_PCT:g} % över klustrets "
            f"snittkurs = edgen förbrukad.</div>", unsafe_allow_html=True)

        st.markdown(f"<div style='height:10px;'></div>"
                    f"<b style='color:{TEXT};'>Säljregler</b> "
                    f"<span style='color:{DIM};font-size:0.78rem;'>"
                    f"— först inträffad gäller</span>", unsafe_allow_html=True)
        for label, what, where in SELL_RULES:
            st.markdown(
                f"<div style='color:{TEXT};font-size:0.82rem;padding:3px 0;'>"
                f"<b>{label}</b> — {what} "
                f"<span style='color:{DIM};'>({where})</span></div>",
                unsafe_allow_html=True)
