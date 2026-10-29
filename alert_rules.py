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


# Vart varje larm leder (Flikguiden ritar tabellen; testet låser att anropen följer den). Nyckel = larmets kind
# (screen_*/sheet_* täcks av SCREEN_TARGET/SHEET_TARGET nedan), värde = (rubrik, målnyckel i ui/nav.TARGETS).
KIND_TARGET = {
    "cycle_turn": ("⚡ VÄNDNING", "turn"),
    "swing_regime": ("Swing-läget skiftade", "swing_regime"),
    "swing_setup": ("Ny swing-kandidat", "thor"),
    "swing_exit": ("Säljsignal på innehav", "holdings"),
    "market_risk_high": ("🌩️ Marknadsrisk HÖG", "market_risk"),
    "market_risk_clear": ("✅ Marknadsrisk ner", "market_risk"),
    "wolf_score": ("🐺 Wolf", "thor"),
    "viking_nine": ("🛡 Viking (Execution-filtret)", "thor"),
    "ember_eligible": ("🔥 EMBER", "thor"),
    "contrarian_deep": ("🎯 Deep Contrarian", "thor"),
    "quality_buy": ("💎 Quality", "thor"),
    "insider_cluster": ("👥 Insiderkluster", "insider"),
    "insider_buy": ("👥 Insider KÖP-läge", "thor"),
    "blindspot_early": ("Blindspot TIDIG", "rotation"),
    "fiat": ("🐺 Fiat Debasement", "fiat"),
    "watch_buy": ("👀 Bevakad aktie blev KÖP", "thor"),
    "watch_out": ("👀 Bevakad aktie föll ur listan", "thor"),
}


def _alert(kind: str, title: str, body: str, target: str = "", ticker: str = "", strategy: str = "") -> dict:
    """Ett larm. target = målnyckeln i ui/nav.TARGETS dit länken leder, ticker/strategy fyller i Thor."""
    a = {"kind": kind, "title": title, "body": body}
    if target:
        a["target"] = target
        if ticker:
            a["ticker"] = ticker
        if strategy:
            a["strategy"] = strategy
    return a


def link_of(alert: dict, base: str = "") -> str:
    """Länken till larmets vy, eller '' utan panelens adress (PANEL_URL) eller utan mål."""
    if not base or not alert.get("target"):
        return ""
    from ui.nav import target_link
    return target_link(alert["target"], base, alert.get("ticker") or "", alert.get("strategy") or "")


def format_alert(alert: dict, base: str = "") -> str:
    """Larmet som text — Discord och mejl får samma innehåll. Med panelens adress (base) en sista rad med länken
    till rätt vy; utan adress är texten som förut."""
    text = f"🐺 {alert['title']}\n{alert['body']}"
    link = link_of(alert, base)
    return f"{text}\n→ {link}" if link else text


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
                or "Se ODIN → Läget för regelverket.")
        alerts.append(_alert(
            "swing_regime",
            f"Swing-regimen skiftade: {prev_light} → {light}",
            body, target="swing_regime"))

    prev_buyable = set((prev or {}).get("buyable", {}) or {})
    for ticker, desc in buyable.items():
        if ticker not in prev_buyable:
            alerts.append(_alert(
                "swing_setup",
                f"Ny swing-kandidat: {ticker}",
                f"{desc} · regim {light or 'okänd'}. Öppna den i THOR (Momentum Swing) "
                f"innan du agerar — Copiloten prövar R:R och veckotak.",
                target="thor", ticker=ticker, strategy="momentum"))

    prev_verdicts = (prev or {}).get("verdicts", {}) or {}
    for ticker, now in verdicts.items():
        before = (prev_verdicts.get(ticker) or {}).get("verdict", "")
        if now["verdict"] in (sv.SELL, sv.PARTIAL) and now["verdict"] != before:
            alerts.append(_alert(
                "swing_exit",
                f"Säljsignal på innehav: {ticker} — {now['verdict']}",
                now["reason"], target="holdings", ticker=ticker))
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
                f"med.", target="rotation"))
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
            f"Hela setupen finns i THOR med strategin EMBER.",
            target="thor", ticker=ticker, strategy="ember"))
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
            f"Detaljer i THOR med strategin Wolf.", target="thor", ticker=ticker, strategy="wolf"))
    return alerts, state


def viking_alerts(viking_data: Optional[dict], prev: Optional[dict],
                  min_nine: int = VIKING_MIN_NINE) -> tuple:
    """(larm, nytt tillstånd). Larm när en ticker NYTT når Viking Execution-filtrets ribba (V9, egna indikatorer — inte OVTLYR Nine)
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
            f"🛡 Viking: {ticker} på {d['v9']}/9",
            f"{d['name'] or ticker} nådde Viking Execution-filtret {d['v9']}/9 och "
            f"klarar den absoluta grinden (pris > EMA200, ADX ≥ 20){extra}. "
            f"Detaljer i THOR med strategin Viking.", target="thor", ticker=ticker, strategy="viking"))
    return alerts, state


# ── Contrarian Alpha (Deep Contrarian) ───────────────────────────────────────
def contrarian_alerts(ca_data: Optional[dict], prev: Optional[dict]) -> tuple:
    """(larm, nytt tillstånd) för Deep Contrarian-listan.

    ca_data är contrarian_alpha.cache.load_screener_results("deep_contrarian"):
    {"timestamp", "results": [{ticker, name, composite_score, rank,
    necessity_score, hat_score, ...}]}. Tom/oläsbar källa (ingen timestamp)
    fryser baslinjen. Larmet gäller övergången IN i listan — hatat men
    behövt, alla grindar passerade — och kroppen bär pelarpoängen så man
    ser VARFÖR den kom in.
    """
    if not isinstance(ca_data, dict) or not ca_data.get("timestamp"):
        return [], (prev if isinstance(prev, dict) else {"ranked": {}})
    # Deep Contrarian är bara råvaror (#98). En lista utan commodity_only=True
    # är från före råvarugrinden (eller fel läge) — larma inte på den, frys
    # baslinjen tills scheduled_scan skrivit en ny.
    if ca_data.get("commodity_only") is not True:
        return [], (prev if isinstance(prev, dict) else {"ranked": {}})

    ranked = {}
    for row in ca_data.get("results", []) or []:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        ranked[ticker] = {
            "name": str(row.get("name", "") or ""),
            "score": row.get("composite_score"),
            "rank": row.get("rank"),
            "necessity": row.get("necessity_score"),
            "hat": row.get("hat_score"),
            "sector": str(row.get("sector", "") or ""),
        }

    state = {"ranked": ranked}
    if prev is None:
        return [], state

    alerts = []
    prev_ranked = set((prev or {}).get("ranked", {}) or {})

    def _n(v, d=0):
        return f"{v:.{d}f}" if isinstance(v, (int, float)) else "–"
    for ticker, d in ranked.items():
        if ticker in prev_ranked:
            continue
        sector = f" · {d['sector']}" if d["sector"] else ""
        alerts.append(_alert(
            "contrarian_deep",
            f"🎯 Deep Contrarian: {ticker} in i listan (#{_n(d['rank'])})",
            f"{d['name'] or ticker}{sector} · composite {_n(d['score'], 1)} · "
            f"necessity {_n(d['necessity'])} · hat {_n(d['hat'])}. Hatat men "
            f"behövt, alla grindar passerade — kör Rick Rule-granskningen "
            f"innan du agerar.", target="thor", ticker=ticker, strategy="deep"))
    return alerts, state


# ── Quality-köpsignaler (alpha_regime.quality_scan) ─────────────────────────
def quality_alerts(q_data: Optional[dict], prev: Optional[dict]) -> tuple:
    """(larm, nytt tillstånd) för Quality-listans köpsignaler.

    q_data är contrarian_alpha.cache.load_screener_results("quality") med
    "quality_signals": {ticker: {verdict, passed, total, benchmark, phase,
    gates}} ur den schemalagda skanningen. Larmet gäller övergången TILL BUY
    — alla fyra gates gröna (trend, värdering, cykel, kvalitet), strategins
    egen köpregel. Saknas signalerna (gammal lista, fel) fryser baslinjen.
    """
    frozen = prev if isinstance(prev, dict) else {"buy": {}}
    if not isinstance(q_data, dict) or not q_data.get("timestamp") \
            or not isinstance(q_data.get("quality_signals"), dict):
        return [], frozen

    rows = {str(r.get("ticker", "")).strip().upper(): r
            for r in (q_data.get("results") or []) if isinstance(r, dict)}
    buy = {}
    for ticker, sig in q_data["quality_signals"].items():
        if not isinstance(sig, dict) or sig.get("verdict") != "BUY":
            continue
        row = rows.get(str(ticker).upper()) or {}
        buy[str(ticker).upper()] = {
            "name": str(row.get("name", "") or ""), "sector": str(row.get("sector", "") or ""),
            "rank": row.get("rank"), "score": row.get("composite_score"),
            "passed": sig.get("passed"), "total": sig.get("total") or 4,
            "benchmark": sig.get("benchmark") or "", "phase": sig.get("phase") or "",
            "gates": sig.get("gates") or {},
        }

    state = {"buy": buy}
    if prev is None:
        return [], state

    seen = set((prev or {}).get("buy", {}) or {})
    alerts = []
    for ticker, d in buy.items():
        if ticker in seen:
            continue
        gates = " · ".join(f"{k} {'✓' if g.get('passed') else '✗'}" for k, g in d["gates"].items())
        sector = f" · {d['sector']}" if d["sector"] else ""
        rank = f" · #{d['rank']:.0f} i listan" if isinstance(d["rank"], (int, float)) else ""
        alerts.append(_alert(
            "quality_buy",
            f"💎 Quality: {ticker} köpsignal ({d['passed']}/{d['total']})",
            f"{d['name'] or ticker}{sector}{rank}. {gates}. Cykel {d['phase'] or '?'} "
            f"mot {d['benchmark'] or '?'}. Alla fyra gates gröna = köp enligt Quality — "
            f"kontrollera i THOR med strategin Quality Contrarian, max 10 % per "
            f"aktie och 20–25 % per sektor.", target="thor", ticker=ticker, strategy="quality"))
    return alerts, state


# ── Insiderbevakaren (insider_scan.py) ───────────────────────────────────────
INSIDER_MIN_SCORE = 7    # av 10 — arkets gräns för "går vidare till grinden"
_S_INSIDER_BUY = "KÖP — logga i journalen"   # = insider.S_BUY (testad)


def insider_alerts(insider_data: Optional[dict], prev: Optional[dict],
                   min_score: int = INSIDER_MIN_SCORE) -> tuple:
    """(larm, nytt tillstånd) för insynskluster.

    insider_data är insider_scan.py:s blob: {"generated", "clusters": [{ticker,
    name, insiders, role, amount, gate, trigger, cluster_avg, price_now,
    score, status, vs_cluster, stop, chase, ...}], "error"}. None eller error
    → baslinjen fryser. Två övergångar larmar: ett kluster som NYTT når
    poängribban, och ett bolag som NYTT står i KÖP-läge (grind Ja, trigger
    satt, inte över +30 % mot klustersnittet).
    """
    empty = {"qualified": {}, "buy": {}}
    if not isinstance(insider_data, dict) or insider_data.get("error"):
        return [], (prev if isinstance(prev, dict) else empty)

    qualified, buy = {}, {}
    for row in insider_data.get("clusters", []) or []:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker", "")).strip().upper()
        score = row.get("score")
        if not ticker or not isinstance(score, (int, float)):
            continue
        d = {"score": int(score), "status": str(row.get("status") or ""),
             "name": str(row.get("name", "") or ""),
             "insiders": row.get("insiders"), "role": row.get("role"),
             "amount": row.get("amount"), "gate": row.get("gate"),
             "trigger": row.get("trigger"), "cluster_avg": row.get("cluster_avg"),
             "price_now": row.get("price_now"), "stop": row.get("stop"),
             "vs_cluster": row.get("vs_cluster")}
        if int(score) >= int(min_score):
            qualified[ticker] = d
        if d["status"] == _S_INSIDER_BUY and not row.get("chase"):
            buy[ticker] = d

    state = {"qualified": qualified, "buy": buy}
    if prev is None:
        return [], state

    def _n(v, d=0, suffix=""):
        return f"{v:,.{d}f}{suffix}" if isinstance(v, (int, float)) else "–"

    def _body(d):
        return (f"{d['name'] or '?'} · {_n(d['insiders'])} insiders · {d['role'] or '–'} · "
                f"{_n(d['amount'])} tkr · grind {d['gate'] or 'kör själv'} · "
                f"trigger {d['trigger'] or '–'} · klustersnitt {_n(d['cluster_avg'], 2)} · "
                f"kurs {_n(d['price_now'], 2)} ({_n(d['vs_cluster'], 1, ' %')}) · "
                f"stopp {_n(d['stop'], 2)}.")

    alerts = []
    prev_q = set((prev or {}).get("qualified", {}) or {})
    prev_b = set((prev or {}).get("buy", {}) or {})
    for ticker, d in qualified.items():
        if ticker in prev_q:
            continue
        alerts.append(_alert(
            "insider_cluster",
            f"👥 Insiderkluster: {ticker} {d['score']}/10 p — {d['status']}",
            _body(d) + " Öppna ARKEN → Insider → Automatisk skanning och "
                       "lägg in klustret i arket.", target="insider", ticker=ticker))
    for ticker, d in buy.items():
        if ticker in prev_b:
            continue
        alerts.append(_alert(
            "insider_buy",
            f"👥 Insider KÖP-läge: {ticker} ({d['score']}/10 p)",
            _body(d) + " Grinden OK och teknisk trigger på plats — öppna i THOR (Insider Norden), "
                       "logga i journalen och lägg stoppen som order.",
            target="thor", ticker=ticker, strategy="insider"))
    return alerts, state


# ── Håvarna (screens_scan.py) ────────────────────────────────────────────────
_SCREEN_ICON = {"rule": "🪨", "sprott": "🔭", "durrett": "🥇", "tiggre": "🎯",
                "royalty": "👑"}
# Håven och arkets händelser → arket i panelen (ui/nav.TARGETS)
SCREEN_TARGET = {"rule": "rule", "sprott": "scoring", "durrett": "durrett", "tiggre": "tiggre", "royalty": "royalty"}
SHEET_TARGET = {"insider": "insider", "tiggre": "tiggre", "royalty": "royalty", "rule": "rule", "durrett": "durrett"}


def sheet_target(kind: str) -> str:
    """'tiggre_free_ride' → 'tiggre', 'durrett_engine_buy_rule' → 'durrett'."""
    return SHEET_TARGET.get(str(kind or "").split("_")[0], "")


def screen_alerts(screens_data: Optional[dict], prev: Optional[dict]) -> tuple:
    """(larm, nytt tillstånd) för guidens fem Börsdata-håvar.

    screens_data: {"generated", "screens": {key: {"label", "rows":
    [{ticker, name, mcap_musd, universe}], "error"}}}. Larm när ett bolag
    NYTT kvalar in i en håv — ett larm per håv med de nya tickrarna.
    En håv med error behåller sin gamla baslinje; hela källan None →
    allt fryser.
    """
    if not isinstance(screens_data, dict) or not isinstance(screens_data.get("screens"), dict):
        return [], (prev if isinstance(prev, dict) else {})

    prev = prev if isinstance(prev, dict) else None
    state, alerts = {}, []
    for key, s in screens_data["screens"].items():
        if not isinstance(s, dict):
            continue
        if s.get("error"):
            if prev is not None and key in prev:
                state[key] = prev[key]
            continue
        cur = {}
        for row in s.get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            t = str(row.get("ticker", "")).strip().upper()
            if t:
                cur[t] = {"name": str(row.get("name", "") or ""),
                          "mcap_musd": row.get("mcap_musd"),
                          "universe": str(row.get("universe", "") or "")}
        state[key] = cur
        if prev is None or key not in prev:
            continue                                   # baslinje för håven
        new = [t for t in cur if t not in (prev.get(key) or {})]
        if not new:
            continue
        label = str(s.get("label") or key)
        parts = []
        for t in new[:12]:
            d = cur[t]
            mc = f" {d['mcap_musd']:,.0f} MUSD" if isinstance(d["mcap_musd"], (int, float)) else ""
            parts.append(f"{t} ({d['name'] or '?'}{mc})")
        more = f" … +{len(new) - 12}" if len(new) > 12 else ""
        alerts.append(_alert(
            f"screen_{key}",
            f"{_SCREEN_ICON.get(key, '🕸')} Håven {label}: {len(new)} ny"
            f"{'a' if len(new) > 1 else 'tt'} bolag",
            f"{' · '.join(parts)}{more}. Kvalar in på håvens kriterier — "
            f"öppna ARKEN → {s.get('sheet') or label} → Håven och lägg in "
            f"den du vill granska.", target=SCREEN_TARGET.get(key, "")))
    return alerts, state


# ── Arkens övergångar (sheets_refresh.py) ────────────────────────────────────
def sheet_alerts(sheets_data: Optional[dict], prev: Optional[dict]) -> tuple:
    """(larm, nytt tillstånd) för händelser räknade på arkens rader med
    färska tal: Insider-stopp/passa, Tiggre free ride/NAV-mål, Royalty-
    signal, Rick Rule-skuld, Durretts köpregel.

    sheets_data: {"generated", "events": [{key, kind, ticker, title, body}],
    "error"}. Tillståndet är de händelser som GÄLLER just nu; ett larm går
    när en händelse är ny. Försvinner den och kommer igen larmar den igen.
    None/error → baslinjen fryser.
    """
    if not isinstance(sheets_data, dict) or sheets_data.get("error"):
        return [], (prev if isinstance(prev, dict) else {"events": {}})
    cur = {}
    for e in sheets_data.get("events", []) or []:
        if isinstance(e, dict) and e.get("key"):
            cur[str(e["key"])] = {"title": str(e.get("title") or ""),
                                  "body": str(e.get("body") or ""),
                                  "kind": str(e.get("kind") or ""),
                                  "ticker": str(e.get("ticker") or "")}
    state = {"events": cur}
    if prev is None:
        return [], state
    seen = set((prev or {}).get("events", {}) or {})
    return [_alert(f"sheet_{d['kind']}", d["title"], d["body"], target=sheet_target(d["kind"]),
                   ticker=d.get("ticker") or "")
            for k, d in cur.items() if k not in seen], state


# ── Sammanvägningen ──────────────────────────────────────────────────────────
# ── 🌩️ Marknadsrisk ─────────────────────────────────────────────────────────
def market_risk_alerts(levels: Optional[dict], prev: Optional[dict]) -> tuple:
    """(larm, nytt tillstånd) — när SPY eller OMXS30 går IN i nivå HÖG, och när
    den lämnar HÖG. FÖRHÖJD larmar inte (bara information i fliken).

    levels: {marknad: market_risk_gate.summarize(...) | None}. En marknad som
    inte gick att räkna behåller sin gamla nivå; levels None fryser benet."""
    prev_levels = dict((prev or {}).get("levels") or {}) if isinstance(prev, dict) else {}
    if not isinstance(levels, dict):
        return [], (prev if isinstance(prev, dict) else {"levels": {}})
    now = dict(prev_levels)
    for m, r in levels.items():
        if isinstance(r, dict) and r.get("level"):
            now[m] = r["level"]
    state = {"levels": now}
    if prev is None:
        return [], state
    alerts = []
    for m, r in levels.items():
        if not isinstance(r, dict) or not r.get("level"):
            continue
        old, new = prev_levels.get(m), r["level"]
        act = ", ".join(r.get("active") or []) or "inga"
        if new == "HÖG" and old != "HÖG":
            alerts.append(_alert(
                "market_risk_high", f"🌩️ Marknadsrisk HÖG: {r.get('label', m)}",
                f"{r.get('points')} av {r.get('possible')} varningar lyser: {act} ({r.get('date')}). "
                f"Viking Nine tar inga nya entries på marknaden och Wolf halverar positionerna. Historiskt har "
                f"−10 % inom tre månader följt betydligt oftare än normalt i HÖG — träffbilden finns i "
                f"ODIN → Läget (🌩️ Marknadsrisk).", target="market_risk"))
        elif old == "HÖG" and new != "HÖG":
            alerts.append(_alert(
                "market_risk_clear", f"✅ Marknadsrisk {r.get('label', m)} ner till {new}",
                f"{r.get('points')} av {r.get('possible')} varningar lyser nu ({act}). Spärren släpper: "
                f"Viking Nine kan ta entries igen och Wolf får full storlek.", target="market_risk"))
    return alerts, state


# ── 🐺 Fiat Debasement ───────────────────────────────────────────────────────
FIAT_DEFAULT_RULES = ("m2_high", "cpi_high", "gap_high", "gs_high", "gs_low", "gold_fall")


def fiat_alerts(fiat_data: Optional[dict], prev: Optional[dict], rules=None) -> tuple:
    """(larm, nytt tillstånd) — när ett mått NYTT passerar en gräns ur
    fiat_debasement.config.FIAT_ALERT_RULES. Bara övergångar larmar; okända
    värden behåller sitt gamla läge. rules = aktiva regelnycklar (None = alla).
    Larmen beskriver vad indikatorerna visar — aldrig köp eller sälj."""
    from fiat_debasement.config import FIAT_ALERT_RULES, CURRENCIES
    prev_on = dict((prev or {}).get("on") or {}) if isinstance(prev, dict) else {}
    if not isinstance(fiat_data, dict):
        return [], (prev if isinstance(prev, dict) else {"on": {}})
    active = set(FIAT_DEFAULT_RULES if rules is None else rules)
    now, alerts = dict(prev_on), []
    metrics = fiat_data.get("metrics") or {}
    dates = fiat_data.get("dates") or {}
    for key, (label, metric, op, limit, per_ccy) in FIAT_ALERT_RULES.items():
        targets = CURRENCIES if per_ccy else ("",)
        for ccy in targets:
            value = (metrics.get(ccy) or {}).get(metric) if per_ccy else fiat_data.get(metric)
            if value is None:
                continue
            hit = value > limit if op == ">" else value < limit
            k = f"{key}|{ccy}"
            was = prev_on.get(k)
            now[k] = hit
            if prev is None or key not in active or not hit or was:
                continue
            who = f"{ccy}: " if ccy else ""
            date = ((dates.get(ccy) or {}).get(metric) if per_ccy else dates.get(metric)) or ""
            alerts.append(_alert(
                "fiat_" + key, f"🐺 Fiat Debasement — {who}{label}",
                f"Enligt valda indikatorer är {metric.replace('_', ' ')} {value:+.1f} "
                f"(gräns {op} {limit:g}){f', {date}' if date else ''}. Bakgrundsinformation ur "
                f"ODIN → Makro → 🐺 Fiat Debasement — ingen köp- eller säljsignal.", target="fiat"))
    return alerts, {"on": now}


def turn_alerts(board_data: Optional[dict], prev: Optional[dict]) -> tuple:
    """(larm, nytt tillstånd) för Valhalls händelsemotor (K10): en ruta som går in i köpfönstret efter BEKRÄFTAD
    cykelvändning larmar en gång — med planen 2 Deep Contrarian + 2 lång + 1 UCITS-ETF ur turn_plan.
    board_data = valhall_board.json; None/utan turn_plan = oläsbar → baslinjen står still."""
    if not isinstance(board_data, dict) or not isinstance(board_data.get("turn_plan"), dict):
        return [], (prev if isinstance(prev, dict) else {"rooms": {}})
    rooms = {}
    for w in board_data["turn_plan"].get("window") or []:
        if not isinstance(w, dict) or not w.get("key"):
            continue
        rooms[f"{w.get('level')}:{w['key']}:{w.get('where')}"] = w
    state = {"rooms": {k: str(v.get("confirmed_since") or v.get("since") or "") for k, v in rooms.items()}}
    if prev is None:
        return [], state
    seen = (prev or {}).get("rooms") or {}
    alerts = []
    for k, w in rooms.items():
        if k in seen and seen[k] == state["rooms"][k]:
            continue
        parts = []
        if w.get("deep"):
            parts.append("Deep Contrarian: " + ", ".join(x.get("ticker", "") for x in w["deep"]))
        if w.get("long"):
            parts.append("Lång: " + ", ".join(x.get("ticker", "") for x in w["long"]))
        e = w.get("etf") or {}
        if e.get("ticker"):
            parts.append(f"ETF: {e['ticker']} ({e.get('exposure')}{'' if e.get('verified') else ', verifiera hos Nordnet'})")
        alerts.append(_alert(
            "cycle_turn",
            f"⚡ VÄNDNING: {w.get('label') or w.get('key')} — {w.get('kind')} bekräftad",
            ("; ".join(parts) or "Inga kandidater i rutan i dag") + ". Papperet köper på nästa öppning; riktiga order "
            "läggs manuellt. Se ODIN → Kartan och PORTFÖLJ → 📊 Papperskonton.", target="turn"))
    return alerts, state


# ── 👀 Övervakningen (data/watchlist.json mot Thors rader) ──────────────────
# Källorna som varje strategis rader i Thor byggs ur (thor.candidates.build; nycklarna i load_blobs, regime =
# wolf_regime.json, sheet_map = registrets karta för arkens bara nordiska former, 'BOL' → 'BOL.ST', som Thor bygger
# med). Saknas eller felar en av dem står strategins par still — annars vore "föll ur listan" falskt, liksom KÖP när
# källan kommer tillbaka (Wolfs ENTRY ur kartan, Momentums regimljus, Qualitys RÖDA sektor, arkens 'BOL' mot 'BOL.ST').
WATCH_SOURCES = {"viking_nine": ("board", "viking_nine"), "wolf": ("arc", "board"), "viking": ("arc",),
                 "momentum": ("momentum", "regime"), "berserk": ("berserk",), "ember": ("ember",),
                 "quality": ("ca_quality", "board"), "deep": ("ca_deep",), "long": ("board",),
                 "insider": ("insider",), "rule": ("board", "screens", "sheet_map"),
                 "durrett": ("board", "screens", "sheet_map"), "tiggre": ("board", "screens", "sheet_map"),
                 "royalty": ("royalty",), "hav": ("screens",)}
_ARC_PARTS = {"wolf": ("wolf", "wolf_usa"), "viking": ("viking", "viking_usa")}
_STAMPED = ("ember", "ca_quality", "ca_deep")      # läsarna ger timestamp None när källan inte gick att läsa
_NOT_LISTS = ("regime", "sheet_map")               # ger inga rader: prövas bara för att de finns, ingen tidsstämpel
# thor.candidates.build:s delar (nycklarna i errors) som ger rader åt andra strategier än sitt namn. En del som föll i
# bygget gav inga rader — det är ingen tom lista.
_BUILD_PARTS = {"themes": ("rule", "durrett", "tiggre"), "arc": ("wolf", "viking"),
                "screens": ("hav", "rule", "durrett", "tiggre"),
                "board": tuple(s for s, keys in WATCH_SOURCES.items() if "board" in keys)}
# valhall_board.json bär flera strategiers listor, och en del kan falla utan felfält (valhall_scan): Long Screenern
# som kastar sparas som long = None (då står Long Screener still), och ett ark, håvarna eller Quality Contrarian som
# inte gick att läsa syns bara som en lucka i sources. Då står Rick Rules, Durretts och Tiggres par på brädans rader
# (KÖP/BEVAKA) still, och brädans stämpel gör inte deras lista färsk. Ett par som stod på håvens TRÄFF-rad för samma
# strategier prövas i stället mot håvens egen körning (screens.json: stämpeln och håvens felfält, _screen_part) —
# oberoende av brädan, så att en färsk men tom håv är 'föll ur listan' även i en panel utan Durrett-arket.
# Arken som strategins rader på brädan byggs ur (valhall.picks): Rick Rules producenter ur producentarket; Durretts
# bolag ur Durrett-arket, och 15-BAGGER och topp 5 ur Poängmodellen (scoring); Tiggres kandidater ur tiggararket, med
# temat ur Durrett-arket, producentarket och Poängmodellen (_commodity_lookup). Ett ark som brädan lästes ur någon
# gång de senaste SHEET_MEMORY_DAYS dagarna men inte i dag är tappat (watch_alerts minns arken) — en lucka för
# strategierna som beror av det. Ett ark som aldrig funnits (Durrett-arket finns inte i alla paneler), eller saknats
# längre än så, är ingen lucka: annars fick strategin aldrig larma.
_BOARD_DEPS = {"rule": ("producers",), "durrett": ("confidence", "scoring"),
               "tiggre": ("tiggre", "confidence", "producers", "scoring")}
SHEET_MEMORY_DAYS = 14
# Utan minne (första körningen, eller ett tillstånd från före minnet) krävs arket som strategin byggs ur, som förut.
_BOARD_SHEETS = {"rule": ("producers",), "durrett": ("confidence",), "tiggre": ("tiggre",)}
_CA_MODES = 2                                       # quality och deep_contrarian (valhall_scan._load_sources)


def watch_key(ticker: str, strategy: str) -> str:
    """Parets nyckel i tillståndet och i frozen: 'HEXA-B.ST|viking_nine'."""
    return f"{ticker}|{strategy}"


def _dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _list(x) -> list:
    return x if isinstance(x, list) else []


def _source_ok(key: str, blob, strategy: str) -> bool:
    """Källan finns och strategins del av den bär inget felfält (en del som fallerade har inga rader)."""
    if not isinstance(blob, dict):
        return False
    if key in _STAMPED:
        return bool(blob.get("timestamp")) and not blob.get("error")
    if key == "regime":
        return bool(blob.get("regime"))
    if key == "sheet_map":
        return bool(blob)
    if blob.get("error"):
        return False
    if key == "board" and strategy == "long":
        return isinstance(blob.get("long"), dict)            # Long Screenern som kastade: long = None, inga rader
    if key == "arc":
        return not any(_dict(blob.get(p)).get("error") for p in _ARC_PARTS.get(strategy, ()))
    if key == "viking_nine":
        return not any(_dict(m).get("error") for m in _dict(blob.get("markets")).values())
    if key == "screens":
        return not any(_dict(v).get("error") for k, v in _dict(blob.get("screens")).items()
                       if strategy == "hav" or k == strategy)
    return True


def _board_read(board, strategy: str) -> Optional[bool]:
    """Lästes strategins delar av valhall_board.json i körningen (brädans sources)? För Rick Rule håvarna och båda
    Quality Contrarian-körningarna; arken prövas mot minnet i watch_alerts (_sheet_gaps). None = brädan saknar sources
    och går inte att pröva (sources kom med arkens köp, valhall fas 2b) — då gör brädan inte listan färsk, men fryser
    inget heller."""
    if strategy not in _BOARD_DEPS:
        return True
    src = _dict(board).get("sources")
    if not isinstance(src, dict):
        return None
    ca = src.get("ca_runs")
    return strategy != "rule" or (bool(src.get("screens")) and isinstance(ca, int) and ca >= _CA_MODES)


def _board_gap(strategy: str) -> str:
    """frozen-nyckeln 'rule@brädan': strategins del av Valhalls bräda lästes inte (_board_read) — paren som står eller
    stod på en rad ur brädan (KÖP/BEVAKA) står still; håvarnas TRÄFF-rader går som vanligt."""
    return f"{strategy}@brädan"


def _screen_part(strategy: str) -> str:
    """fresh-nyckeln 'durrett@håven': strategins håv i screens.json är färsk och felfri (watch_fresh). Ett par som stod
    på håvens TRÄFF-rad prövas mot den i stället för mot brädan (watch_alerts)."""
    return f"{strategy}@håven"


def _on_board(state: Optional[dict]) -> bool:
    """Parets läge kommer ur brädans rader: Rick Rule, Durrett och Tiggre har KÖP/BEVAKA bara där (håvarna: TRÄFF)."""
    from thor import candidates as tc
    return isinstance(state, dict) and bool(state.get("in_list")) and state.get("class") in (tc.KOP, tc.BEVAKA)


def watch_frozen(blobs: Optional[dict], errors: Optional[dict] = None) -> set:
    """Det som står still i dag: strategier vars källa saknas eller felar (WATCH_SOURCES, thor.live.blob_errors)
    eller vars del föll i bygget (errors ur thor.candidates.build), 'strategi@brädan' när strategins del av Valhalls
    bräda inte lästes (_board_gap; ett tappat ark avgör watch_alerts, som minns arken), och par 'TICKER|strategi'
    där just aktiens hämtning fallerade (BERSERK:s missing, EMBER-rader med error)."""
    from thor import candidates as tc
    from thor import live
    b = _dict(blobs)
    out = set(live.blob_errors(b))
    out |= {s for s, keys in WATCH_SOURCES.items() if not all(_source_ok(k, b.get(k), s) for k in keys)}
    for part in _dict(errors):
        out |= set(_BUILD_PARTS.get(part) or ((part,) if part in WATCH_SOURCES else WATCH_SOURCES))
    out |= {_board_gap(s) for s in _BOARD_DEPS if s not in out and _board_read(b.get("board"), s) is False}
    out |= {watch_key(tc.yf(t), "berserk") for t in _list(_dict(b.get("berserk")).get("missing")) if t}
    ember = _dict(b.get("ember"))
    for r in _list(ember.get("eligible")) + _list(ember.get("near_misses")):
        if isinstance(r, dict) and r.get("error") and r.get("ticker"):
            out.add(watch_key(tc.yf(r["ticker"]), "ember"))
    return out


def _stamps(key: str, blob, strategy: str) -> list:
    """Listans egna tidsstämplar för strategin — samma fält som Thor visar som körd (arc per del, Viking Nine per
    marknad)."""
    b = _dict(blob)
    if key == "arc":
        return [_dict(b.get(p)).get("generated") for p in _ARC_PARTS.get(strategy, ()) if isinstance(b.get(p), dict)]
    if key == "viking_nine":
        return [_dict(m).get("generated") or b.get("generated") for m in _dict(b.get("markets")).values()] or \
            [b.get("generated")]
    return [b.get("timestamp" if key in _STAMPED else "generated")]


def _recent(stamp, today, days: Optional[int] = None) -> bool:
    """Stämpeln är högst days (förval tc.STALE_DAYS, samma gräns som Thors GAMMAL) dagar gammal. Ingen eller
    oläslig = nej."""
    from datetime import date

    from thor import candidates as tc
    try:
        return (today - date.fromisoformat(str(stamp or "")[:10])).days <= (tc.STALE_DAYS if days is None else days)
    except ValueError:
        return False


def watch_fresh(blobs: Optional[dict], today=None) -> dict:
    """{strategi: körd} för strategier vars listor är färska enligt sina EGNA tidsstämplar (högst tc.STALE_DAYS
    gamla) och felfria. Där betyder en aktie som saknas 'inte i listan' — även när listan är tom (BERSERK bär bara
    KÖP/BEVAKA, Viking Nine bara GOLDEN/READY/DEVELOPING). En lista utan användbar stämpel räknas inte som färsk,
    inte heller en del av Valhalls bräda som inte lästes eller inte går att pröva (_board_read, Long Screener i
    _source_ok; ett tappat ark tar watch_alerts bort): utan färska rader står strategins par då still, som när
    körningen saknas. Rick Rules, Durretts och Tiggres håvar får dessutom en egen nyckel ('durrett@håven',
    _screen_part) när håven finns i screens.json, är felfri och screens.json är färsk — oberoende av brädan."""
    from datetime import date
    t0 = date.fromisoformat(str(today)[:10]) if today else date.today()
    b = _dict(blobs)
    out = {}
    for s, keys in WATCH_SOURCES.items():
        if not all(_source_ok(k, b.get(k), s) for k in keys) or _board_read(b.get("board"), s) is not True:
            continue
        stamps = [x for k in keys if k not in _NOT_LISTS for x in _stamps(k, b.get(k), s)]
        if stamps and all(_recent(x, t0) for x in stamps):
            out[s] = max(str(x)[:10] for x in stamps)
    scr = b.get("screens")
    for s in _BOARD_DEPS:                                # en håv som saknas i körningen är ingen tom lista
        if isinstance(_dict(_dict(scr).get("screens")).get(s), dict) and _source_ok("screens", scr, s):
            stamp = _stamps("screens", scr, s)[0]
            if _recent(stamp, t0):
                out[_screen_part(s)] = str(stamp)[:10]
    return out


def watch_sheets(blobs: Optional[dict], today=None) -> Optional[dict]:
    """Arken som Valhalls bräda lästes ur i körningen (brädans sources.sheets) och dagens datum:
    {"date": "2026-10-17", "read": ["confidence", ...]} — watch_alerts minns dem och ser vilka som tappats. None när
    brädan saknas, felar eller saknar sources (sources kom med arkens köp, valhall fas 2b): arken går inte att pröva."""
    from datetime import date
    board = _dict(_dict(blobs).get("board"))
    src = board.get("sources")
    if board.get("error") or not isinstance(src, dict):
        return None
    t0 = date.fromisoformat(str(today)[:10]) if today else date.today()
    return {"date": t0.isoformat(), "read": sorted({str(x) for x in _list(src.get("sheets"))})}


def _sheet_gaps(seen: Optional[dict], sheets: Optional[dict]) -> set:
    """Strategierna vars rader på brädan i dag byggdes utan ett ark de beror av (_BOARD_DEPS): ett ark som brädan
    lästes ur högst SHEET_MEMORY_DAYS dagar före i dag (seen = {ark: senast läst}) men inte i dag. Utan minne
    (seen None) krävs arket strategin byggs ur (_BOARD_SHEETS), som förut. sheets = watch_sheets; None = inget att
    pröva."""
    from datetime import date
    if not isinstance(sheets, dict):
        return set()
    read = set(_list(sheets.get("read")))
    if seen is None:
        return {s for s, need in _BOARD_SHEETS.items() if not set(need) <= read}
    t0 = date.fromisoformat(str(sheets.get("date"))[:10])
    lost = {str(n) for n, d in seen.items() if n not in read and _recent(d, t0, SHEET_MEMORY_DAYS)}
    return {s for s, deps in _BOARD_DEPS.items() if lost & set(deps)}


def _sheets_seen(seen: Optional[dict], sheets: Optional[dict]) -> Optional[dict]:
    """Minnet efter i dag: {ark: senast läst} — dagens ark får dagens datum, de andra står kvar."""
    if not isinstance(sheets, dict):
        return seen
    return {**(seen or {}), **{str(n): str(sheets.get("date")) for n in _list(sheets.get("read"))}}


def _px(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "–"


def _watch_alert(item: dict, before: dict, now: dict, st: dict, asof: str) -> Optional[dict]:
    """Larmet för ett pars övergång — eller None när övergången inte är en av de två som larmar."""
    from thor import candidates as tc
    from thor import registry as reg
    t, s = item["ticker"], item["strategy"]
    label, who = reg.label(s), item.get("name") or t
    since = f" Bevakad sedan {item['added']}." if item.get("added") else ""
    if now["class"] == tc.KOP and before.get("class") != tc.KOP:
        was = (before.get("status") or before.get("class")) if before.get("in_list") else "inte i listan"
        lv = " · ".join(f"{n} {_px(st.get(k))}" for k, n in (("entry", "entry"), ("stop", "stopp"), ("target", "mål"))
                        if st.get(k) is not None)
        lv = f"{lv}{' (dina egna nivåer)' if st.get('own') else ''}. " if lv else ""
        extra = f" · {st['score_text']}" if st.get("score_text") else ""
        return _alert(
            "watch_buy", f"👀 {t} blev KÖP — {label}",
            f"{who}: {was or '?'} → {now['status']}{extra}. {lv}Strategins egen körning ({st.get('asof') or '?'}) "
            f"säger KÖP — öppna i THOR och pröva storlek och R:R innan du agerar.{since}",
            target="thor", ticker=t, strategy=s)
    if before.get("in_list") and not now["in_list"]:
        return _alert(
            "watch_out", f"👀 {t} föll ur listan — {label}",
            f"{who} stod som {before.get('status') or before.get('class') or '?'} men finns inte i strategins körning "
            f"({asof or 'i dag'}). Öppna i THOR för strategins dom i dag, eller ta bort den ur PORTFÖLJ → "
            f"👀 Övervakning.{since}", target="thor", ticker=t, strategy=s)
    return None


def watch_alerts(watch_data: Optional[dict], rows: Optional[list], prev: Optional[dict], frozen=(),
                 fresh: Optional[dict] = None, sheets: Optional[dict] = None) -> tuple:
    """(larm, nytt tillstånd) för 👀 Övervakningen: de bevakade paren (aktie, strategi) mot Thors rader.

    Ägarens beslut: bara två övergångar larmar — paret BLIR KÖP (från vad som helst annat, även från att inte stå i
    listan) eller FALLER UR strategins lista (stod där, saknas nu). Andra byten är tysta. Första körningen och
    nytillagda par lägger baslinjen tyst; ett par med alert = False räknas men larmar inte.

    rows = thor.candidates.build(...); None = körningarna gick inte att läsa → benet fryser, liksom när listan
    (watch_data) inte gick att läsa. frozen = watch_frozen(blobs), fresh = watch_fresh(blobs). En strategi som varken
    har en färsk rad eller en färsk, felfri lista enligt källans egen stämpel (saknas, eller äldre än tc.STALE_DAYS)
    och ett par vars egen rad är gammal behåller förra läget — en körning som inte gick är ingen övergång; likaså ett
    par på brädans rader när strategins del av Valhalls bräda inte lästes ('strategi@brädan' i frozen) eller byggdes
    utan ett ark den beror av (sheets = watch_sheets(blobs) mot minnet, _sheet_gaps) — då är brädan inte heller
    färsk för strategin. Ett par som stod på håvens TRÄFF-rad (Rick Rule, Durrett, Tiggre) och inte står på brädan
    i dag prövas i stället mot håvens egen körning ('durrett@håven' i fresh, eller en färsk TRÄFF-rad), oberoende av
    brädan — brädans rader visar inte att håven körde, så utan fresh står ett sådant par still tills håven har en färsk
    TRÄFF-rad. En färsk lista utan aktien är däremot 'föll ur listan', även när listan är tom.
    Tillståndet: {"items": {"TICKER|strategi": {"id", "in_list", "class", "status"}}, "sheets": {ark: senast läst}};
    "sheets" saknas tills brädans ark gått att pröva en gång."""
    if rows is None or not isinstance(watch_data, dict):
        return [], (prev if isinstance(prev, dict) else {"items": {}})
    import watchlist as wl
    from thor import candidates as tc
    first = not isinstance(prev, dict)
    seen = _dict(_dict(prev).get("items"))
    known = _dict(prev).get("sheets")
    known = known if isinstance(known, dict) else None
    gaps = {s for s in _BOARD_DEPS if _board_gap(s) in frozen} | _sheet_gaps(known, sheets)
    raw = watch_data.get("items") if isinstance(watch_data.get("items"), list) else []
    raw_ids = {str(r.get("id")) for r in raw if isinstance(r, dict) and r.get("id")}   # lagrade id (inte påhittade)
    asof: dict = {s: d for s, d in (fresh or {}).items() if s not in gaps}
    ran = set(asof) | {c.strategy for c in rows if not c.stale}
    for c in rows:
        asof[c.strategy] = max(asof.get(c.strategy, ""), c.asof or "")
        if c.strategy in _BOARD_DEPS and c.status_class == tc.TRAFF and not c.stale:   # en färsk TRÄFF-rad: håven körde
            ran.add(_screen_part(c.strategy))
            asof[_screen_part(c.strategy)] = max(asof.get(_screen_part(c.strategy), ""), c.asof or "")
    items, alerts = {}, []
    for it in wl.normalize({"items": list(raw)})["items"]:
        t, s = it["ticker"], it["strategy"]
        k = watch_key(t, s)
        before = seen.get(k) if isinstance(seen.get(k), dict) else None
        if before and before.get("id") and it["id"] in raw_ids and before["id"] != it["id"]:
            before = None                                # borttagen och tillagd igen: ett nytt par, ny baslinje
        st = wl.status_of(it, rows)
        now = {"id": it["id"] if it["id"] in raw_ids else "", "in_list": st["in_list"],
               "class": st["status_class"], "status": st["status"] if st["in_list"] else ""}
        # ett nytt par en luckdag får sin baslinje först när brädan är hel (första körningen larmar ändå inte)
        gap = s in gaps and ((before is None and not first) or _on_board(before) or _on_board(now))
        # håvparet (stod på TRÄFF-raden, står inte på brädan i dag) går på håvens körning; brädans par som förut
        run = _screen_part(s) if (s in _BOARD_DEPS and before is not None and before.get("in_list")
                                  and not _on_board(before) and not _on_board(now)) else s
        if s in frozen or k in frozen or run not in ran or st["stale"] or gap:
            if before is not None:
                items[k] = before
            continue
        items[k] = now
        if first or before is None or not it["alert"]:
            continue
        a = _watch_alert(it, before, now, st, asof.get(run, ""))
        if a:
            alerts.append(a)
    known = _sheets_seen(known, sheets)
    return alerts, ({"items": items} if known is None else {"items": items, "sheets": known})


def evaluate(regime_data: dict, screener_data: dict, swing_data: dict,
             themes: list, prev_state: Optional[dict],
             settings: Optional[dict] = None,
             ember_data: Optional[dict] = None,
             wolf_data: Optional[dict] = None,
             viking_data: Optional[dict] = None,
             contrarian_data: Optional[dict] = None,
             quality_data: Optional[dict] = None,
             insider_data: Optional[dict] = None,
             screens_data: Optional[dict] = None,
             sheets_data: Optional[dict] = None,
             market_risk_data: Optional[dict] = None,
             fiat_data: Optional[dict] = None,
             board_data: Optional[dict] = None,
             watch_data: Optional[dict] = None,
             watch_rows: Optional[list] = None,
             frozen_watch=None,
             fresh_watch: Optional[dict] = None,
             sheets_watch: Optional[dict] = None) -> tuple:
    """(larm-med-kanaler, nytt tillstånd) för hela körningen.

    settings: data/alerts.json — {"swing": {"enabled", "channels"},
    "blindspot"/"ember"/"wolf"/"viking": {...}; wolf har även "min_score",
    viking "min_nine"}. Saknas den gäller påslaget med Discord: hellre ett
    larm i en kanal som inte finns (loggas och hoppas över) än ett system som
    är tyst för att ingen sparat en inställningsfil.

    ember/wolf/viking-källorna är None när de inte gick att läsa (eller
    aldrig körts) — respektive ben behåller då sin gamla baslinje orörd.
    watch_data/watch_rows/frozen_watch/fresh_watch/sheets_watch = Övervakningens
    lista, Thors rader, watch_frozen, watch_fresh och watch_sheets (blobbarna)
    — se watch_alerts.
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

    c_alerts, c_state = contrarian_alerts(contrarian_data, _prev("contrarian"))
    out += _route("contrarian", c_alerts)

    q_alerts, q_state = quality_alerts(quality_data, _prev("quality"))
    out += _route("quality", q_alerts)

    insider_cfg = cfg.get("insider") or {}
    i_alerts, i_state = insider_alerts(
        insider_data, _prev("insider"),
        min_score=int(insider_cfg.get("min_score") or INSIDER_MIN_SCORE))
    out += _route("insider", i_alerts)

    sc_alerts, sc_state = screen_alerts(screens_data, _prev("screens"))
    out += _route("screens", sc_alerts)

    sh_alerts, sh_state = sheet_alerts(sheets_data, _prev("sheets"))
    out += _route("sheets", sh_alerts)

    mr_alerts, mr_state = market_risk_alerts(market_risk_data, _prev("market_risk"))
    out += _route("market_risk", mr_alerts)

    fiat_cfg = cfg.get("fiat") or {}
    f_alerts, f_state = fiat_alerts(fiat_data, _prev("fiat"), fiat_cfg.get("rules"))
    out += _route("fiat", f_alerts)

    t_alerts, t_state = turn_alerts(board_data, _prev("turns"))
    out += _route("turns", t_alerts)

    wa_alerts, wa_state = watch_alerts(watch_data, watch_rows, _prev("watch"), frozen_watch or (),
                                       fresh_watch, sheets_watch)
    out += _route("watch", wa_alerts)

    return out, {"swing": s_state, "blindspot": b_state, "ember": e_state,
                 "wolf": w_state, "viking": v_state, "contrarian": c_state,
                 "quality": q_state, "insider": i_state, "screens": sc_state, "sheets": sh_state,
                 "market_risk": mr_state, "fiat": f_state, "turns": t_state, "watch": wa_state}
