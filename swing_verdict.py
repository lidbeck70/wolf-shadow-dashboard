"""
swing_verdict.py — Momentum Swing-domen: köp, bevaka, håll, sälj eller avstå.

Momentum-reglerna bor redan i tre datakällor som panelen har:

  wolf_regime.json     trafikljuset GRÖN/GUL/RÖD plus regelverket per läge
  wolf_screener.json   topp 40-rankingen med setup A-flagga och B? (nära
                       52v-högsta) — rad 1–20 är köpbara
  data/swing.json      dina positioner (med datum) och marknadsfiltret

Copiloten frågade ändå "kontrollera manuellt" på alla sex entry-reglerna,
och Swing Regime-fliken kunde inte svara på en ticker. Den här modulen läser
källorna och dömer — samma dom i båda flikarna, för det är samma funktion.

Rena funktioner: all data skickas in. Ingen Streamlit, inget nätverk.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

# Trafikljusets värden i wolf_regime.json
GREEN_LIGHT, YELLOW_LIGHT, RED_LIGHT = "GRÖN", "GUL", "RÖD"

TOP_BUYABLE = 20          # rad 1–20 är köpbara
RANK_EXIT = 40            # ur topp 40 = säljregel för innehav
STOP_PCT = -10.0          # −10 % stop
HALF_AT_PCT = 20.0        # +20 % = sälj halva, stop till entry
MAX_POSITIONS = 8
MAX_WEEKLY_BUYS = 2       # "max 1–2 nya köp per vecka"
STALE_DAYS = 7            # äldre screenerdata än så är en varning

# Domarna
BUY = "KÖP-KANDIDAT"
WATCH = "BEVAKA"
HOLD = "HÅLL"
PARTIAL = "DELVINST — sälj halva"
SELL = "SÄLJ"
ABSTAIN = "AVSTÅ"
UNKNOWN = "OKÄNT LÄGE"


def _num(value, default=None):
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def normalize_ticker(ticker: str) -> str:
    """Jämförbar form av en ticker.

    Screenern använder Börsdatas form ("ERIC B", "ANOT"); användaren skriver
    ofta yfinance-formen ("ERIC-B.ST", "ANOT.ST"). Suffix och skiljetecken
    bort, så matchar båda formerna varandra.
    """
    t = (ticker or "").strip().upper()
    for suffix in (".ST", ".OL", ".CO", ".HE", ".IC"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
            break
    return t.replace("-", "").replace(".", "").replace(" ", "")


def screener_row(ticker: str, screener_data: dict) -> Optional[dict]:
    wanted = normalize_ticker(ticker)
    if not wanted:
        return None
    for row in (screener_data or {}).get("top", []) or []:
        if isinstance(row, dict) and normalize_ticker(row.get("ticker", "")) == wanted:
            return row
    return None


def held_position(ticker: str, swing_data: dict) -> Optional[dict]:
    wanted = normalize_ticker(ticker)
    if not wanted:
        return None
    for p in (swing_data or {}).get("positions", []) or []:
        if isinstance(p, dict) and normalize_ticker(p.get("ticker", "")) == wanted:
            return p
    return None


def regime_light(regime_data: dict, swing_data: dict) -> tuple:
    """(ljus, källa). Regimmotorn först; utan den faller vi tillbaka på
    swing-flikens manuella MA200-knapp — sämre upplösning, men aldrig tyst."""
    light = ((regime_data or {}).get("regime") or "").upper()
    if light in (GREEN_LIGHT, YELLOW_LIGHT, RED_LIGHT):
        return light, f"Swing Regime ({(regime_data or {}).get('generated', '?')})"
    market = (swing_data or {}).get("market") or {}
    if "aboveMA200" in market:
        manual = GREEN_LIGHT if market.get("aboveMA200") else RED_LIGHT
        return manual, (f"manuellt marknadsfilter i Swing-fliken "
                        f"({market.get('checked', '?')})")
    return "", ""


def data_age_days(data: dict, today: Optional[date] = None) -> Optional[int]:
    """Hur gammal en genererad JSON är. None när datumet inte går att läsa."""
    raw = str((data or {}).get("generated", ""))[:10]
    try:
        generated = datetime.fromisoformat(raw).date()
    except (ValueError, TypeError):
        return None
    return ((today or date.today()) - generated).days


def _iso_week(day: Optional[date] = None) -> str:
    d = day or date.today()
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def weekly_buys(swing_data: dict, today: Optional[date] = None) -> int:
    """Positioner öppnade den här ISO-veckan — räknat, inte uppskattat."""
    week = _iso_week(today)
    count = 0
    for p in (swing_data or {}).get("positions", []) or []:
        raw = str((p or {}).get("date", ""))[:10]
        try:
            d = datetime.fromisoformat(raw).date()
        except (ValueError, TypeError):
            continue
        if _iso_week(d) == week:
            count += 1
    return count


def position_count(swing_data: dict) -> int:
    return len((swing_data or {}).get("positions", []) or [])


def verdict(ticker: str, screener_data: dict, regime_data: dict,
            swing_data: dict, today: Optional[date] = None) -> dict:
    """Domen för en ticker, med varje skäl utskrivet.

    Säljreglerna prövas i playbookens ordning för innehav; köpgrindarna för
    kandidater. Saknad data blir OKÄNT LÄGE med instruktion — en dom utan
    underlag är ingen dom.
    """
    reasons = []
    light, light_source = regime_light(regime_data, swing_data)
    row = screener_row(ticker, screener_data)
    pos = held_position(ticker, swing_data)
    rank = _num((row or {}).get("rank"))

    age = data_age_days(screener_data, today)
    if age is not None and age > STALE_DAYS:
        reasons.append(f"VARNING: screenerdatan är {age} dagar gammal — kör "
                       f"wolf_data.py för en färsk ranking.")

    # ── Innehav: säljreglerna, först inträffad gäller ────────────────────────
    if pos is not None:
        entry = _num(pos.get("entry"))
        price = _num((row or {}).get("price"))
        pnl = (round((price / entry - 1) * 100, 1)
               if price and entry else None)

        if row is None:
            reasons.insert(0, f"Ur topp {RANK_EXIT} — rank-exit. Innehavet "
                              f"finns inte längre i screenerlistan.")
            return {"verdict": SELL, "reasons": reasons, "rank": None,
                    "held": True, "light": light, "pnl_pct": None}
        if pnl is not None and pnl <= STOP_PCT:
            reasons.insert(0, f"{pnl:+.1f} % mot entry — stoppen på "
                              f"{STOP_PCT:g} % är passerad.")
            return {"verdict": SELL, "reasons": reasons, "rank": rank,
                    "held": True, "light": light, "pnl_pct": pnl}
        dist50 = _num(row.get("dist_ma50"))
        if dist50 is not None and dist50 < 0:
            reasons.insert(0, f"Kurs {dist50 * 100:+.1f} % mot MA50 — "
                              f"stängning under MA50 är säljregeln.")
            return {"verdict": SELL, "reasons": reasons, "rank": rank,
                    "held": True, "light": light, "pnl_pct": pnl}
        if pnl is not None and pnl >= HALF_AT_PCT and not pos.get("halfTaken"):
            reasons.insert(0, f"{pnl:+.1f} % — sälj halva och flytta stoppen "
                              f"till entry (free ride på resten).")
            return {"verdict": PARTIAL, "reasons": reasons, "rank": rank,
                    "held": True, "light": light, "pnl_pct": pnl}
        reasons.insert(0, f"Rank {rank:g}, över MA50"
                       + (f", {pnl:+.1f} % mot entry" if pnl is not None else "")
                       + " — ingen säljregel utlöst.")
        return {"verdict": HOLD, "reasons": reasons, "rank": rank,
                "held": True, "light": light, "pnl_pct": pnl}

    # ── Kandidat: köpgrindarna ───────────────────────────────────────────────
    if not light:
        reasons.insert(0, "Regimljuset saknas — kör wolf_data.py eller sätt "
                          "marknadsfiltret i Swing-fliken.")
        return {"verdict": UNKNOWN, "reasons": reasons, "rank": rank,
                "held": False, "light": "", "pnl_pct": None}
    if light == RED_LIGHT:
        reasons.insert(0, f"Regimen är RÖD ({light_source}) — inga nya "
                          f"swingköp, hantera endast exits.")
        return {"verdict": ABSTAIN, "reasons": reasons, "rank": rank,
                "held": False, "light": light, "pnl_pct": None}

    if row is None:
        gen = (screener_data or {}).get("generated", "okänt datum")
        reasons.insert(0, f"Finns inte i screenerlistan (genererad {gen}) — "
                          f"utanför topp {RANK_EXIT} eller utanför universum. "
                          f"Kontrollera tickerformen om det ser fel ut.")
        return {"verdict": ABSTAIN, "reasons": reasons, "rank": None,
                "held": False, "light": light, "pnl_pct": None}
    if rank is not None and rank > TOP_BUYABLE:
        reasons.insert(0, f"Rank {rank:g} — köpbar kräver topp {TOP_BUYABLE}. "
                          f"Rad 21–{RANK_EXIT} är endast bevakning.")
        return {"verdict": WATCH, "reasons": reasons, "rank": rank,
                "held": False, "light": light, "pnl_pct": None}

    setup_a = bool(row.get("setupA"))
    setup_b = bool(row.get("nearHigh"))
    if light == YELLOW_LIGHT and not setup_a:
        reasons.insert(0, "GUL regim: endast setup A (pullback) och halv "
                          "positionsstorlek — B-utbrott jagas inte i GUL.")
        return {"verdict": WATCH, "reasons": reasons, "rank": rank,
                "held": False, "light": light, "pnl_pct": None}
    if not setup_a and not setup_b:
        reasons.insert(0, f"Rank {rank:g} men ingen setup — vänta på pullback "
                          f"mot MA20/50 (A) eller läge nära 52v-högsta (B).")
        return {"verdict": WATCH, "reasons": reasons, "rank": rank,
                "held": False, "light": light, "pnl_pct": None}

    setup = "A (pullback)" if setup_a else "B? (nära 52v-högsta)"
    size = ("HALV positionsstorlek — GUL regim"
            if light == YELLOW_LIGHT else "full storlek 12–20 %")
    reasons.insert(0, f"Rank {rank:g} av {TOP_BUYABLE}, setup {setup}, "
                      f"regim {light} → {size}.")
    return {"verdict": BUY, "reasons": reasons, "rank": rank,
            "held": False, "light": light, "pnl_pct": None,
            "setup": setup}


# ── Copilotens sex entry-regler, mekaniskt ───────────────────────────────────
def rule_checks(ticker: str, screener_data: dict, regime_data: dict,
                swing_data: dict, today: Optional[date] = None) -> dict:
    """{nyckelord: (status, notering)} för momentum-playbookens entry-regler.

    Nycklarna matchas mot regeltexterna i Copiloten. Saknad data ger MANUAL
    med instruktion — en regel utan underlag får inte se ut som ett ja.
    """
    light, source = regime_light(regime_data, swing_data)
    row = screener_row(ticker, screener_data)
    rank = _num((row or {}).get("rank"))
    out = {}

    # "Marknadsfiltret måste vara grönt"
    if not light:
        out["marknadsfilter"] = ("MANUAL", "Regimljuset saknas — kör "
                                           "wolf_data.py eller sätt filtret i "
                                           "Swing-fliken.")
    elif light == GREEN_LIGHT:
        out["marknadsfilter"] = ("PASS", f"GRÖN — {source}")
    elif light == YELLOW_LIGHT:
        out["marknadsfilter"] = ("MANUAL", f"GUL ({source}) — köp tillåtna "
                                           f"men halv storlek, endast setup A.")
    else:
        out["marknadsfilter"] = ("FAIL", f"RÖD ({source}) — inga nya "
                                         f"swingköp.")

    # "Bolaget ska ligga i topp 20 på rankingen"
    if row is None:
        gen = (screener_data or {}).get("generated", "okänt datum")
        out["ranking"] = ("MANUAL", f"Hittas inte i screenerlistan (genererad "
                                    f"{gen}). Utanför topp {RANK_EXIT}, eller "
                                    f"fel tickerform.")
    elif rank is not None and rank <= TOP_BUYABLE:
        out["ranking"] = ("PASS", f"Rank {rank:g} av topp {TOP_BUYABLE}")
    else:
        out["ranking"] = ("FAIL", f"Rank {rank:g} — rad 21–{RANK_EXIT} är "
                                  f"endast bevakning")

    # "Det krävs en setup — A eller B"
    if row is None:
        out["setup"] = ("MANUAL", "Ingen screenerrad — setupflaggan kan inte "
                                  "läsas.")
    elif row.get("setupA"):
        out["setup"] = ("PASS", "Setup A — pullback mot MA20/50 med RSI 35–55")
    elif row.get("nearHigh"):
        out["setup"] = ("MANUAL" if light == YELLOW_LIGHT else "PASS",
                        "Setup B? — inom 3 % av 52v-högsta"
                        + (" (jagas inte i GUL regim)"
                           if light == YELLOW_LIGHT else ""))
    else:
        out["setup"] = ("FAIL", "Ingen setupflagga i screenern — vänta på "
                                "pullback eller utbrottsläge.")

    # "Max 1–2 nya köp per vecka"
    bought = weekly_buys(swing_data, today)
    if bought >= MAX_WEEKLY_BUYS:
        out["köp per vecka"] = ("FAIL", f"{bought} köp redan denna vecka "
                                        f"({_iso_week(today)}) — taket är "
                                        f"{MAX_WEEKLY_BUYS}.")
    elif bought == MAX_WEEKLY_BUYS - 1:
        out["köp per vecka"] = ("MANUAL", f"{bought} köp denna vecka — nästa "
                                          f"är veckans sista.")
    else:
        out["köp per vecka"] = ("PASS", f"{bought} köp denna vecka "
                                        f"({_iso_week(today)})")

    # "Max 8 positioner"
    held = position_count(swing_data)
    if held >= MAX_POSITIONS:
        out["positioner"] = ("FAIL", f"{held}/{MAX_POSITIONS} positioner — "
                                     f"fullt. Sälj något först.")
    else:
        out["positioner"] = ("PASS", f"{held}/{MAX_POSITIONS} positioner")

    # "Positionsstorlek 12–20 % — halv vid GUL regim"
    if light == GREEN_LIGHT:
        out["positionsstorlek"] = ("PASS", "GRÖN regim — full storlek "
                                           "12–20 %.")
    elif light == YELLOW_LIGHT:
        out["positionsstorlek"] = ("MANUAL", "GUL regim — HALV storlek "
                                             "(6–10 %).")
    elif light == RED_LIGHT:
        out["positionsstorlek"] = ("FAIL", "RÖD regim — ingen storlek alls.")
    else:
        out["positionsstorlek"] = ("MANUAL", "Regimljuset saknas.")

    return out
