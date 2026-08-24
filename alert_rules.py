"""
alert_rules.py — vad som är värt ett larm, och när.

Grundregeln: ÖVERGÅNGAR larmar, lägen gör det inte. "Uran är i tidig cykel"
tre gånger om dagen i veckor är inte ett larm, det är tapetsering — och
tapetserade larm slutar man läsa. Därför jämförs varje körning mot förra
körningens tillstånd, och bara det som ÄNDRATS skickas.

Första körningen (inget tidigare tillstånd) larmar ingenting: den lägger
baslinjen. Utan den regeln hade dag ett spammat ett larm per tema och
kandidat som redan låg rätt.

Två källor:

  Swing      regimskiften (GRÖN/GUL/RÖD), nya setup-kandidater i topp 20,
             och säljsignaler på innehav — allt via swing_verdict, samma
             dom som Copiloten och Swing Regime-fliken visar.
  Blindspot  ett råvarutema som GÅR IN i TIDIG cykel — Odins egen
             definition (10-årspercentil ≤ 30, temakartans cykel_label).

Rena funktioner: all data in som argument, (larmlista, nytt tillstånd) ut.
Ingen Streamlit, inget nätverk.
"""

from __future__ import annotations

from typing import Optional

import swing_verdict as sv

EARLY = "TIDIG"


def _alert(kind: str, title: str, body: str) -> dict:
    return {"kind": kind, "title": title, "body": body}


def format_alert(alert: dict) -> str:
    """Larmet som text — Discord och mejl får samma innehåll."""
    return f"🐺 {alert['title']}\n{alert['body']}"


# ── Swing ────────────────────────────────────────────────────────────────────
def _buyable(screener_data: dict) -> dict:
    """{ticker: setupbeskrivning} för köpbara kandidater (topp 20 med setup)."""
    out = {}
    for row in (screener_data or {}).get("top", []) or []:
        if not isinstance(row, dict):
            continue
        rank = row.get("rank")
        if not isinstance(rank, (int, float)) or rank > sv.TOP_BUYABLE:
            continue
        if row.get("setupA"):
            out[str(row.get("ticker", ""))] = f"rank {rank:g} · setup A (pullback)"
        elif row.get("nearHigh"):
            out[str(row.get("ticker", ""))] = (f"rank {rank:g} · setup B? "
                                               f"(nära 52v-högsta)")
    return out


def swing_alerts(regime_data: dict, screener_data: dict, swing_data: dict,
                 prev: Optional[dict]) -> tuple:
    """(larm, nytt tillstånd) för swing-benet.

    prev är förra körningens tillstånd — None betyder första körningen:
    baslinje läggs, inga larm skickas.
    """
    alerts = []
    light, _source = sv.regime_light(regime_data, swing_data)
    buyable = _buyable(screener_data)

    verdicts = {}
    for p in (swing_data or {}).get("positions", []) or []:
        ticker = str((p or {}).get("ticker", "")).strip()
        if not ticker:
            continue
        v = sv.verdict(ticker, screener_data, regime_data, swing_data)
        verdicts[ticker.upper()] = {"verdict": v["verdict"],
                                    "reason": v["reasons"][0] if v["reasons"]
                                    else ""}

    state = {"light": light, "buyable": buyable, "verdicts": verdicts}
    if prev is None:
        return [], state

    prev_light = (prev or {}).get("light", "")
    if light and prev_light and light != prev_light:
        rules = (regime_data or {}).get("rules", []) or []
        body = ("\n".join(f"• {r}" for r in rules)
                or "Se REGIME → Swing Regime för regelverket.")
        alerts.append(_alert(
            "swing_regime",
            f"Swing-regimen skiftade: {prev_light} → {light}",
            body))

    prev_buyable = set((prev or {}).get("buyable", {}) or {})
    for ticker, desc in buyable.items():
        if ticker not in prev_buyable:
            alerts.append(_alert(
                "swing_setup",
                f"Ny swing-kandidat: {ticker}",
                f"{desc} · regim {light or 'okänd'}. Kör Copiloten på den "
                f"innan du agerar — R:R och veckotak prövas där."))

    prev_verdicts = (prev or {}).get("verdicts", {}) or {}
    for ticker, now in verdicts.items():
        before = (prev_verdicts.get(ticker) or {}).get("verdict", "")
        if now["verdict"] in (sv.SELL, sv.PARTIAL) and now["verdict"] != before:
            alerts.append(_alert(
                "swing_exit",
                f"Säljsignal på innehav: {ticker} — {now['verdict']}",
                now["reason"]))
    return alerts, state


# ── Blindspot ────────────────────────────────────────────────────────────────
def blindspot_alerts(themes: list, prev: Optional[dict]) -> tuple:
    """(larm, nytt tillstånd) för temakartan.

    themes: [{"name", "cykel_label", "blindspot_score", "hat_score"}] — rena
    dicts, så testerna slipper temakartans tio års kursnedladdningar.

    Larmet gäller ÖVERGÅNGEN in i TIDIG. Ett tema som redan låg där vid
    baslinjen larmar först om det lämnar och kommer tillbaka — därför skrivs
    baslinjen ut i loggen av körskriptet, så man ser vad som INTE kommer att
    larma.
    """
    alerts = []
    labels = {}
    for t in themes or []:
        name = str((t or {}).get("name", "")).strip()
        if not name:
            continue
        labels[name] = {
            "label": str(t.get("cykel_label", "")),
            "score": t.get("blindspot_score"),
            "hat": t.get("hat_score"),
        }

    if prev is None:
        return [], {"themes": labels}

    prev_themes = (prev or {}).get("themes", {}) or {}
    for name, cur in labels.items():
        before = (prev_themes.get(name) or {}).get("label", "")
        if cur["label"] == EARLY and before and before != EARLY:
            score = cur.get("score")
            hat = cur.get("hat")
            alerts.append(_alert(
                "blindspot_early",
                f"Blindspot: {name} gick in i TIDIG cykel",
                f"10-årspercentilen är nere vid botten (Odins tröskel ≤ 30). "
                f"Blindspot-poäng {score if score is not None else '–'} · "
                f"hat {hat if hat is not None else '–'}. "
                f"Från {before}. Nästa steg: rotationsflikens Triple "
                f"Signal-betyg — hat räcker inte, case och katalysator ska "
                f"med."))
    return alerts, {"themes": labels}


# ── EMBER ────────────────────────────────────────────────────────────────────
def ember_alerts(ember_data: Optional[dict], prev: Optional[dict]) -> tuple:
    """(larm, nytt tillstånd) för EMBER-screenern.

    ember_data är ember.cache.load_ember_results(): {"timestamp", "eligible",
    "near_misses"}. timestamp None betyder att källan inte gick att läsa —
    då behålls förra baslinjen orörd, annars skulle nästa lyckade läsning
    "återupptäcka" hela listan och larma om allt igen.

    Larmet gäller ÖVERGÅNGEN in i eligible (alla tre grindarna passerade) —
    och kroppen bär hela setupen: entry, stopp, mål och R:R, för det är
    exakt det EMBER redan räknat ut.
    """
    if not isinstance(ember_data, dict) or ember_data.get("timestamp") is None:
        return [], (prev if isinstance(prev, dict) else {"eligible": {}})

    eligible = {}
    for row in ember_data.get("eligible", []) or []:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue
        eligible[ticker.upper()] = {
            "entry": row.get("entry"), "stop": row.get("stop"),
            "t1": row.get("t1"), "rr": row.get("rr"),
            "cykel": row.get("cykel_label", ""),
            "quality": row.get("setup_quality"),
        }

    state = {"eligible": eligible}
    if prev is None:
        return [], state

    alerts = []
    prev_eligible = set((prev or {}).get("eligible", {}) or {})
    for ticker, d in eligible.items():
        if ticker in prev_eligible:
            continue

        def _n(v):
            return f"{v:.2f}" if isinstance(v, (int, float)) else "–"
        alerts.append(_alert(
            "ember_eligible",
            f"🔥 EMBER: {ticker} klarade alla grindar",
            f"Entry {_n(d['entry'])} · stopp {_n(d['stop'])} · mål "
            f"{_n(d['t1'])} · R:R {_n(d['rr'])}"
            f"{' · cykel ' + d['cykel'] if d['cykel'] else ''}. "
            f"Setupen i sin helhet finns i EMBER-fliken."))
    return alerts, state


# ── Wolf (Arc, 4-lagers regimscore) ──────────────────────────────────────────
WOLF_MIN_SCORE = 80.0    # av 125 — flikens default är 50, larmribban är högre
VIKING_MIN_NINE = 8      # av 9 — fullträff 9/9 är sällsynt, 8+ fångar det viktiga


def wolf_alerts(wolf_data: Optional[dict], prev: Optional[dict],
                min_score: float = WOLF_MIN_SCORE) -> tuple:
    """(larm, nytt tillstånd). Larm när en ticker NYTT når regimscore-ribban.

    wolf_data: {"generated", "rows": [{"ticker","name","score"}]} från
    arc_scan.py. None (källan oläsbar/aldrig körd) → behåll förra baslinjen.
    Tillståndet sparar ALLA tickers över ribban, så en ribbändring i
    inställningarna inte retro-larmar om gamla kvalificerade.
    """
    if not isinstance(wolf_data, dict):
        return [], (prev if isinstance(prev, dict) else {"qualified": {}})

    qualified = {}
    for row in wolf_data.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        score = row.get("score")
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker or not isinstance(score, (int, float)):
            continue
        if float(score) >= float(min_score):
            qualified[ticker] = {"score": round(float(score), 1),
                                 "name": str(row.get("name", ""))}

    state = {"qualified": qualified}
    if prev is None:
        return [], state

    alerts = []
    prev_q = set((prev or {}).get("qualified", {}) or {})
    for ticker, d in qualified.items():
        if ticker in prev_q:
            continue
        alerts.append(_alert(
            "wolf_score",
            f"🐺 Wolf-screenern: {ticker} nådde {d['score']:g}/125",
            f"{d['name'] or ticker} klev över larmribban {min_score:g} i "
            f"4-lagersscoren (Market · Sector · Stock · Ichimoku). "
            f"Detaljer i SCREENING → Arc Screener → Wolf."))
    return alerts, state


def viking_alerts(viking_data: Optional[dict], prev: Optional[dict],
                  min_nine: int = VIKING_MIN_NINE) -> tuple:
    """(larm, nytt tillstånd). Larm när en ticker NYTT når Vikings Nine-ribban
    OCH klarar den absoluta grinden (pris > EMA200, ADX >= 20) — utan den är
    en hög Nine bara "bäst i en svag skara".
    """
    if not isinstance(viking_data, dict):
        return [], (prev if isinstance(prev, dict) else {"qualified": {}})

    qualified = {}
    for row in viking_data.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker", "")).strip().upper()
        nine = row.get("v9")
        if not ticker or not isinstance(nine, (int, float)):
            continue
        if int(nine) >= int(min_nine) and row.get("eligible"):
            qualified[ticker] = {"v9": int(nine),
                                 "name": str(row.get("name", "")),
                                 "signal": str(row.get("signal", ""))}

    state = {"qualified": qualified}
    if prev is None:
        return [], state

    alerts = []
    prev_q = set((prev or {}).get("qualified", {}) or {})
    for ticker, d in qualified.items():
        if ticker in prev_q:
            continue
        extra = f" · signal {d['signal']}" if d["signal"] else ""
        alerts.append(_alert(
            "viking_nine",
            f"⚔️ Viking: {ticker} på {d['v9']}/9",
            f"{d['name'] or ticker} nådde Vikings Nine {d['v9']}/9 och "
            f"klarar den absoluta grinden (pris > EMA200, ADX ≥ 20){extra}. "
            f"Detaljer i SCREENING → Arc Screener → Viking."))
    return alerts, state


# ── Sammanvägningen ──────────────────────────────────────────────────────────
def evaluate(regime_data: dict, screener_data: dict, swing_data: dict,
             themes: list, prev_state: Optional[dict],
             settings: Optional[dict] = None,
             ember_data: Optional[dict] = None,
             wolf_data: Optional[dict] = None,
             viking_data: Optional[dict] = None) -> tuple:
    """(larm-med-kanaler, nytt tillstånd) för hela körningen.

    settings: data/alerts.json — {"swing": {"enabled", "channels"},
    "blindspot"/"ember"/"wolf"/"viking": {...}; wolf har även "min_score",
    viking "min_nine"}. Saknas den gäller påslaget med Discord: hellre ett
    larm i en kanal som inte finns (loggas och hoppas över) än ett system som
    är tyst för att ingen sparat en inställningsfil.

    ember/wolf/viking-källorna är None när de inte gick att läsa (eller
    aldrig körts) — respektive ben behåller då sin gamla baslinje orörd.
    Avstängda ben räknar ändå sitt tillstånd, så en återaktivering inte
    exploderar i retro-larm.
    """
    cfg = settings or {}
    prev = prev_state if isinstance(prev_state, dict) and prev_state else None

    def _prev(leg):
        return (prev or {}).get(leg) if prev else None

    def _route(leg, alerts):
        leg_cfg = cfg.get(leg) or {}
        if not leg_cfg.get("enabled", True):
            return []
        channels = leg_cfg.get("channels") or ["discord"]
        return [{**a, "channels": channels} for a in alerts]

    out = []
    s_alerts, s_state = swing_alerts(regime_data, screener_data, swing_data,
                                     _prev("swing"))
    out += _route("swing", s_alerts)

    b_alerts, b_state = blindspot_alerts(themes, _prev("blindspot"))
    out += _route("blindspot", b_alerts)

    e_alerts, e_state = ember_alerts(ember_data, _prev("ember"))
    out += _route("ember", e_alerts)

    wolf_cfg = cfg.get("wolf") or {}
    w_alerts, w_state = wolf_alerts(
        wolf_data, _prev("wolf"),
        min_score=float(wolf_cfg.get("min_score") or WOLF_MIN_SCORE))
    out += _route("wolf", w_alerts)

    viking_cfg = cfg.get("viking") or {}
    v_alerts, v_state = viking_alerts(
        viking_data, _prev("viking"),
        min_nine=int(viking_cfg.get("min_nine") or VIKING_MIN_NINE))
    out += _route("viking", v_alerts)

    return out, {"swing": s_state, "blindspot": b_state, "ember": e_state,
                 "wolf": w_state, "viking": v_state}
