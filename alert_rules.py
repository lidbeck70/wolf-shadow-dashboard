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


# ── Sammanvägningen ──────────────────────────────────────────────────────────
def evaluate(regime_data: dict, screener_data: dict, swing_data: dict,
             themes: list, prev_state: Optional[dict],
             settings: Optional[dict] = None) -> tuple:
    """(larm-med-kanaler, nytt tillstånd) för hela körningen.

    settings: data/alerts.json — {"swing": {"enabled", "channels"},
    "blindspot": {...}}. Saknas den gäller påslaget med Discord: hellre ett
    larm i en kanal som inte finns (loggas och hoppas över) än ett system som
    är tyst för att ingen sparat en inställningsfil.
    """
    cfg = settings or {}
    swing_cfg = cfg.get("swing") or {}
    blind_cfg = cfg.get("blindspot") or {}
    prev = prev_state if isinstance(prev_state, dict) and prev_state else None

    out = []
    s_alerts, s_state = swing_alerts(
        regime_data, screener_data, swing_data,
        (prev or {}).get("swing") if prev else None)
    if swing_cfg.get("enabled", True):
        channels = swing_cfg.get("channels") or ["discord"]
        out += [{**a, "channels": channels} for a in s_alerts]

    b_alerts, b_state = blindspot_alerts(
        themes, (prev or {}).get("blindspot") if prev else None)
    if blind_cfg.get("enabled", True):
        channels = blind_cfg.get("channels") or ["discord"]
        out += [{**a, "channels": channels} for a in b_alerts]

    return out, {"swing": s_state, "blindspot": b_state}
