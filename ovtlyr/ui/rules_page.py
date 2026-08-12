"""
Rules page — clean cyberpunk-styled display of all trading rules.
Every rule has a PANEL guide showing exactly which Nordic Alpha tab to use.
Updated for the consolidated 9-tab layout.
"""

import streamlit as st

_BG2     = "#14141e"
_CYAN    = "#00E5FF"
_GREEN   = "#2d8a4e"
_MAGENTA = "#00A8BF"
_RED     = "#c44545"
_YELLOW  = "#d4943a"
_BLUE    = "#00E5FF"
_TEXT    = "#e8e4dc"
_DIM     = "#8a8578"
_AMBER   = "#c9a84c"
_EMBER   = "#FF6B3D"

# ------------------------------------------------------------------ #
#  Rule data — single source of truth lives in strategy_rules.py
#  (both this tab and the STRATEGIES tab read from it, so the numbers
#   can never drift apart again).
# ------------------------------------------------------------------ #

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from strategy_rules import (            # noqa: E402
    PLAYBOOKS, LEARNING_ORDER, LEVEL_COLOR,
)
# Re-exported for backwards compatibility — this module used to own these lists.
from strategy_rules import (            # noqa: E402,F401
    as_rule_dicts, SWING_RULES, LONGTERM_RULES,
    OVTLYR_ENTRY_RULES, OVTLYR_EXIT_RULES, OVTLYR_MINDSET,
)


# ------------------------------------------------------------------ #
#  HTML helpers
# ------------------------------------------------------------------ #

def _rule_card_html(rule: dict, color: str) -> str:
    num = rule["number"]
    text = rule["text"]
    expl = rule["explanation"]
    guide = rule.get("panel_guide", "")

    guide_html = ""
    if guide:
        guide_html = (
            f"<div style='color:{_CYAN};font-size:0.62rem;margin-top:5px;"
            f"padding:5px 8px;background:rgba(0,229,255,0.05);border-radius:3px;"
            f"border-left:2px solid rgba(0,229,255,0.2);'>"
            f"<b>NORDIC ALPHA:</b> {guide}</div>"
        )

    return (
        f"<div style='background:{_BG2};border-left:3px solid {color};"
        f"padding:10px 14px;margin:6px 0;border-radius:4px;'>"
        f"<span style='color:{color};font-size:1.1rem;font-weight:700;'>#{num}</span>"
        f"<span style='color:{_TEXT};font-size:0.88rem;margin-left:10px;'>{text}</span>"
        f"<div style='color:{_DIM};font-size:0.7rem;margin-top:3px;'>{expl}</div>"
        f"{guide_html}"
        f"</div>"
    )


def _section_header_html(title: str, subtitle: str, color: str) -> str:
    return (
        f"<div style='border-bottom:2px solid {color};padding-bottom:8px;margin-bottom:12px;'>"
        f"<h3 style='color:{color};margin:0;letter-spacing:0.1em;'>{title}</h3>"
        f"<span style='color:{_DIM};font-size:0.7rem;letter-spacing:0.08em;'>{subtitle}</span>"
        f"</div>"
    )


# ------------------------------------------------------------------ #
#  Strategy guides
# ------------------------------------------------------------------ #

def _gs(title: str, body: str, color: str = _CYAN) -> str:
    """Return styled guide-section HTML block."""
    return (
        f"<div style='margin-bottom:22px;'>"
        f"<div style='color:{color};font-size:0.7rem;font-weight:700;"
        f"letter-spacing:0.1em;text-transform:uppercase;"
        f"border-bottom:1px solid {color}44;padding-bottom:4px;margin-bottom:8px;'>"
        f"{title}</div>"
        f"<div style='color:{_TEXT};font-size:0.87rem;line-height:1.6;'>{body}</div>"
        f"</div>"
    )


def _ul(items: list, color: str = _DIM) -> str:
    lis = "".join(f"<li style='margin-bottom:5px;'>{it}</li>" for it in items)
    return (
        f"<ul style='color:{color};font-size:0.85rem;line-height:1.55;"
        f"margin:4px 0 0 16px;padding:0;'>{lis}</ul>"
    )


def _guide_quality() -> None:
    st.markdown(
        f"<h3 style='color:{_CYAN};letter-spacing:0.1em;margin-bottom:4px;'>Quality</h3>"
        f"<div style='color:{_DIM};font-size:0.75rem;margin-bottom:20px;'>"
        f"Inspirerad av Warren Buffett och Kvalitetsaktiepodden (KAP)</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            _gs("1. Målet",
                "Hitta välskötta bolag med stark ekonomi, köp dem till rimliga priser när "
                "marknadsläget är gynnsamt och håll dem länge (0,5–3 år). "
                "Strategin letar efter <em>wonderful companies at fair prices</em> — "
                "Buffetts princip kombinerad med KAPs krav på hög lönsamhet, "
                "stabila marginaler och låg skuldsättning.",
                _CYAN),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("2. När köper du",
                "Alla fyra villkor måste vara gröna (systemet visar BUY):"
                + _ul([
                    "<b>Trend:</b> Aktiekursen ÖVER 200-dagars EMA <em>och</em> 50-dagars EMA ÖVER 200-dagars EMA (gyllene kors).",
                    "<b>Värdering:</b> P/E-tal 7–25 <em>och</em> EV/EBIT 4–20. Varken för dyrt eller suspekt billigt.",
                    "<b>Marknadscykel:</b> Marknaden i DISBELIEF, HOPE, OPTIMISM, BELIEF eller DISBELIEF_NEW (tidigt till mitt i en uppgång — inte eufori).",
                    "<b>Bolagskvalitet:</b> Kvalitetspoäng ≥ 55/100 (ROIC, marginaler, tillväxt) ELLER KAP-badge.",
                ], _TEXT) +
                "<div style='color:#607080;font-size:0.8rem;margin-top:6px;'>"
                "WATCH = 3 av 4 gröna. WAIT = 2 eller färre.</div>",
                _CYAN),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("3. Positionsstorlek",
                _ul([
                    "Max <b>10%</b> av portföljvärdet per enskild aktie.",
                    "Sträva efter <b>8–10 innehav</b> (koncentrerat men diversifierat).",
                    "Bygg hela positionen i en omgång när BUY-signalen tänds.",
                    "Max <b>20–25% exponering per sektor</b>.",
                ], _TEXT),
                _AMBER),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _gs("4. När säljer du",
                _ul([
                    "<b>Trenden bryts:</b> Kursen faller UNDER 200-dagars EMA — halvera positionen vid brottet, sälj resten vid regimskifte.",
                    "<b>Värderingen spricker:</b> P/E > 25 eller EV/EBIT > 20 — cykeln kan vara på topp.",
                    "<b>Cykeln skiftar:</b> Marknadsfas THRILL, EUPHORIA, COMPLACENCY, ANXIETY eller DENIAL.",
                    "<b>Bolagskvaliteten faller:</b> Kvalitetspoängen rasar kraftigt eller marginaler försämras konsekvent.",
                ], _TEXT),
                _RED),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("5. Var i panelen",
                _ul([
                    "<b>Hitta kandidater:</b> SIGNALS → Contrarian Alpha (sortera på kvalitetspoäng).",
                    "<b>Bekräfta köpläge:</b> REGIME → Alpha Regime → Quality-läge → tryck ANALYSERA.",
                    "<b>Kontrollera portföljrisk:</b> PORTFOLIO → Holdings (korrelation + sektorexponering).",
                    "<b>Bevaka regimskifte:</b> REGIME → Wolf Regime (trendfilter) eller Viking Regime (EMA-alignment).",
                ], _CYAN),
                _CYAN),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("6. Vanliga misstag",
                _ul([
                    "Köper för sent — när nyheterna är positiva är kursen ofta redan i THRILL-fasen.",
                    "Struntar i kvalitetspoängen — ett billigt bolag med dålig lönsamhet är en värdefälla.",
                    "Sprider för brett — 20+ aktier späder ur hela fördelen med strategin.",
                    "Säljer inte vid EMA200-brott — <em>det vänder snart</em> kostar dig 30–50%.",
                ], _DIM),
                _RED),
            unsafe_allow_html=True,
        )


def _guide_contrarian() -> None:
    st.markdown(
        f"<h3 style='color:{_EMBER};letter-spacing:0.1em;margin-bottom:4px;'>Deep Contrarian</h3>"
        f"<div style='color:{_DIM};font-size:0.75rem;margin-bottom:20px;'>"
        f"Inspirerad av Rick Rule och Eric Sprott</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            _gs("1. Målet",
                "Köp råvarurelaterade tillgångar (gruvbolag, guld, silver, olja) i faser av "
                "maximalt hat och förtvivlan — och sälj när alla är euforiska. "
                "Rick Rule: <em>du är antingen kontrarian eller du är ett offer.</em> "
                "Eric Sprott: bygg positioner i tredjedelar, aldrig allt på en gång. "
                "Tidshorisonten är 1–3 år per cykel.",
                _EMBER),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("2. När köper du",
                "Tre köpfaser baserade på marknadscykeln:"
                + _ul([
                    "<b>ACCUMULATE 1/3</b> — Marknadsfas: CAPITULATION eller DEPRESSION. "
                    "Maximalt pessimism, alla ger upp. Köp första tredjedelen.",
                    "<b>ACCUMULATE 2/3</b> — Marknadsfas: DISBELIEF, ANGER eller PANIC. "
                    "Tvivel och ilska. Kursen fortfarande under 200-dagars MA. Köp andra tredjedelen.",
                    "<b>ACCUMULATE 3/3</b> — Marknadsfas: HOPE. Kurs nära 200-dagars MA (–5% till +15%). "
                    "Sista chansen att ackumulera billigt.",
                    "<b>Bekräftelse:</b> Gummisnodde ≥ 90:e percentilen (t.ex. Gold/Silver) = "
                    "råvaran historiskt max-billig vs motvikten → stärker köpfallet.",
                    "<b>Sentiment-overlay:</b> Retail sentiment < 30/100 = extrem rädsla → "
                    "HIGH confidence-signal.",
                ], _TEXT),
                _EMBER),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("3. Positionsstorlek",
                _ul([
                    "Bygg i <b>tre lika delar</b> — aldrig allt på en gång.",
                    "Max <b>~25% per råvarukategori</b> (t.ex. max 25% i guldgruvbolag totalt).",
                    "Räkna med att kunna hålla genom <b>50% nedgång</b> utan att sälja i panik.",
                    "Ingen hård procentsats per trade — storleken bestäms av din totala portfölj och conviction.",
                ], _TEXT),
                _AMBER),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _gs("4. När säljer du",
                "Tre distributionsfaser:"
                + _ul([
                    "<b>DISTRIBUTE 1/3</b> — Fas: THRILL. Entusiasm och FOMO driver priserna. Sälj 25–33%.",
                    "<b>DISTRIBUTE 2/3</b> — Fas: EUPHORIA eller COMPLACENCY. Alla är euforiska. Sälj 50–75%.",
                    "<b>DISTRIBUTE 3/3</b> — Fas: ANXIETY eller DENIAL. Trenden bryter ner. Sälj resterande.",
                    "<b>Sentiment-overlay:</b> Retail sentiment > 70/100 = extrem girighet → "
                    "stärker sälj-signalen.",
                    "<b>HOLD</b>-fasen (OPTIMISM/BELIEF): håll, köp inte mer, sälj inte ännu.",
                ], _TEXT),
                _RED),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("5. Var i panelen",
                _ul([
                    "<b>Hitta fas:</b> REGIME → Alpha Regime → välj Deep Contrarian → tryck ANALYSERA.",
                    "<b>Gummisnodde-bekräftelse:</b> Välj 'Commodity exposure' i Alpha Regime för att se "
                    "om råvaruratios är historiskt sträckta (≥ 90:e percentilen).",
                    "<b>Hitta kandidater:</b> SIGNALS → Contrarian Alpha (Hat Score + Necessity).",
                    "<b>Sentiment:</b> INTELLIGENCE → Retail Pulse.",
                ], _CYAN),
                _CYAN),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("6. Vanliga misstag",
                _ul([
                    "Köper allt på en gång istället för i tredjedelar — du missar möjligheten att pressa ner snittköp.",
                    "Säljer i HOLD-fasen för tidigt — <em>trenden har precis börjat, du är på väg att missa det mesta.</em>",
                    "Ignorerar gummisnodde-signalerna — de ger objektiv bekräftelse när magkänslan sviker.",
                    "Köper i DENIAL/ANXIETY (som är ett SÄLJ-läge, inte ett KÖPLÄGE).",
                ], _DIM),
                _RED),
            unsafe_allow_html=True,
        )


def _guide_wolf() -> None:
    st.markdown(
        f"<h3 style='color:{_CYAN};letter-spacing:0.1em;margin-bottom:4px;'>Wolf</h3>"
        f"<div style='color:{_DIM};font-size:0.75rem;margin-bottom:20px;'>"
        f"Kortsiktig swing-handel med trend + Ichimoku-exit</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            _gs("1. Målet",
                "Fånga kortsiktiga trendrörelser (dagar till veckor) med hög precision och "
                "tight riskkontroll. Varje trade har definierat stop-loss och minst 1:2 i "
                "risk/reward (förhållande mellan risk och potentiell vinst). "
                "Max 1% av portföljkapitalet i risk per trade.",
                _CYAN),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("2. När köper du (11 regler, alla krävs)",
                _ul([
                    "<b>Regel 1:</b> Handla <em>med</em> trenden — upptrend = long, nedtrend = short.",
                    "<b>Regel 2:</b> Ingen trade i konsolidering (sidorörelser) — vänta på breakout.",
                    "<b>Regel 3:</b> En key level krävs — tydligt stöd/motstånd eller supply/demand-zon inom 3%.",
                    "<b>Regel 4:</b> Entry efter pullback — pris inom 2% av EMA10 eller EMA20.",
                    "<b>Regel 5:</b> Candlestick-trigger — hammer, engulfing eller break-and-retest.",
                    "<b>Regel 6:</b> Volymbekräftelse — volymratio ≥ 1,0× (ej undre volym).",
                    "<b>Regel 7:</b> Min R:R 1:2, helst 1:3.",
                    "<b>Regel 8:</b> Max 1% risk av portföljkapitalet per trade.",
                ], _TEXT),
                _CYAN),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("3. Positionsstorlek",
                _ul([
                    "Beräkna: <b>Positionsstorlek = (Kapital × 1%) / SL-avstånd i kr</b>.",
                    "Sätt alltid SL baserat på struktur (key level), aldrig ett fast procenttal.",
                    "Max 2 förluster per dag — stäng plattformen direkt efter den andra.",
                ], _TEXT),
                _AMBER),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _gs("4. När säljer du",
                _ul([
                    "<b>Regel 9:</b> Flytta SL till breakeven (entry-priset) <em>efter</em> ny HH (higher high) eller HL (higher low) — inte tidigare.",
                    "<b>Regel 10:</b> Max 2 förluster per dag — stopp dagen efter den andra.",
                    "<b>Regel 11 — exit:</b> Trailing stop via Kijun-sen (Ichimoku, 26-period). "
                    "Stäng om pris stänger UNDER Kijun <em>och</em> under EMA10. "
                    "½ × ATR som hård nödstopp.",
                ], _TEXT),
                _RED),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("5. Var i panelen",
                _ul([
                    "<b>Screena kandidater:</b> SIGNALS → Arc Screener.",
                    "<b>Kontrollera alla 11 regler:</b> REGIME → Wolf Regime → Entry Checklist (auto pass/fail).",
                    "<b>SL/TP-kalkylator:</b> REGIME → Wolf Regime → SL/TP-sektion.",
                    "<b>Trendfilter:</b> REGIME → Wolf Regime → Regime Score (grön badge = handla long).",
                ], _CYAN),
                _CYAN),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("6. Vanliga misstag",
                _ul([
                    "Handlar mot trenden — den enda regeln som aldrig har undantag.",
                    "Tar trade utan key level — utan OB/stöd finns inget logiskt stopp.",
                    "Dåligt R:R — tar 1:1 trades och undrar varför kontot krymper.",
                    "Skippar volymkravet — låg volym = rörelsen är inte äkta.",
                ], _DIM),
                _RED),
            unsafe_allow_html=True,
        )


def _guide_viking() -> None:
    st.markdown(
        f"<h3 style='color:{_BLUE};letter-spacing:0.1em;margin-bottom:4px;'>Viking</h3>"
        f"<div style='color:{_DIM};font-size:0.75rem;margin-bottom:20px;'>"
        f"OVTLYR Golden Ticket — momentum-swing med 10 entry + 10 exit-regler</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            _gs("1. Målet",
                "Fånga starka momentumrörelser i aktier som redan är i tydlig upptrend. "
                "Strategin kräver att ALLT är alignat — marknad, sektor och aktie. "
                "Baserad på OVTLYR Golden Ticket-systemet: "
                "<em>Where Outliers Win.</em> "
                "Max 1–2% risk per trade, alltid med exakt stop-loss.",
                _BLUE),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("2. När köper du (10 entry-regler, alla måste passera)",
                _ul([
                    "<b>1. Marknadstrend:</b> SPY — 10 EMA ÖVER 20 EMA, aktiekurs ÖVER 50 EMA.",
                    "<b>2. Marknadssignal:</b> Viking-signal på $SPY visar BUY (grön).",
                    "<b>3. Marknadsbreadd:</b> Bull List % stiger (bullish crossover).",
                    "<b>4. Sektorbreadd:</b> Sektorn stigande — bullish 10 EMA-kors.",
                    "<b>5. Sektor Fear &amp; Greed:</b> Sektorsentimentet förbättras (stiger).",
                    "<b>6. Aktiesignal:</b> Viking-signal på aktien visar BUY.",
                    "<b>7. Aktietrend:</b> 10 EMA > 20 EMA > 50 EMA, aktie ÖVER alla tre.",
                    "<b>8. Aktie F&amp;G:</b> Aktiens Fear &amp; Greed stiger (ej 'Exhausted').",
                    "<b>9. Order Blocks:</b> Inga bearish OBs (motståndszoner) blockerar vägen uppåt.",
                    "<b>10. Momentum:</b> Pris ovanför gårdagens lägsta — RSI > 50.",
                ], _TEXT),
                _BLUE),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("3. Positionsstorlek",
                _ul([
                    "Beräkna: <b>Positionsstorlek = (Kapital × 1–2%) / SL-avstånd</b>.",
                    "SL = entry minus ½ × ATR (14-period).",
                    "Max 2 förluster per dag — stäng plattformen.",
                ], _TEXT),
                _AMBER),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _gs("4. När säljer du (10 exit-regler)",
                _ul([
                    "<b>1. Hård exit:</b> SPY stänger UNDER 20 EMA → stäng ALLT omedelbart.",
                    "<b>2. SL:</b> ½ × ATR (14p) under entry — initial stop-loss.",
                    "<b>3. Trailing stop:</b> Stäng om pris stänger under 10 EMA.",
                    "<b>4. Order Block hit:</b> Pris rör sig in i bearish OB-zon → exit.",
                    "<b>5. Gap &amp; Crap:</b> Gap upp följt av fall under gårdagens stängning → exit direkt.",
                    "<b>6. Stängning under gårdagens lägsta</b> (efter att SL rullats till BE).",
                    "<b>7. Sektor + breadth crossover:</b> Sektorn byter grön → röd AND Bull List vänder ner.",
                    "<b>8. Aktiesignal flippar:</b> Viking-signal byter till SELL eller REDUCE.",
                    "<b>9. F&amp;G target:</b> Entry F&amp;G 0–50 → exit vid +63. 50–75 → +10p. 75+ → +5p.",
                    "<b>10. Earnings risk:</b> Stäng minst 5 handelsdagar före rapportdag.",
                ], _TEXT),
                _RED),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("5. Var i panelen",
                _ul([
                    "<b>Alla 10 entry-regler:</b> REGIME → Viking Regime (Vikings Nine-checklist).",
                    "<b>Screena momentum:</b> SIGNALS → Arc Screener (Viking-filter).",
                    "<b>Marknadsbreadd:</b> REGIME → Viking Regime → Bull List % gauge.",
                    "<b>Sektor:</b> INTELLIGENCE → Heatmap / Flow Divergence.",
                    "<b>F&amp;G per aktie:</b> REGIME → Viking Regime → Oscillator Direction.",
                ], _CYAN),
                _CYAN),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("6. Vanliga misstag",
                _ul([
                    "Köper när SPY är svag — den hårda exit-regeln (SPY under 20 EMA) visar varför det är farligt.",
                    "Ignorerar marknadsbreadden — utan Bull List-bekräftelse köper du in i en försvagning.",
                    "Håller igenom earnings — en rapport kan radera hela vinsten på sekunder.",
                    "Använder för stort SL-avstånd — tar mer risk per trade än 1–2%, kontot töms snabbt vid en serie förluster.",
                ], _DIM),
                _RED),
            unsafe_allow_html=True,
        )


def _guide_ember() -> None:
    st.markdown(
        f"<h3 style='color:{_EMBER};letter-spacing:0.1em;margin-bottom:4px;'>🔥 EMBER</h3>"
        f"<div style='color:{_DIM};font-size:0.75rem;margin-bottom:20px;'>"
        f"Commodity swing trading med cykelfiltrar, trendgates och ATR-riskmodell</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            _gs("1. Målet",
                "Fånga kortsiktiga rörelser i råvaror och råvarurelaterade ETF:er "
                "och aktier som befinner sig tidigt i sin cykel (TIDIG/MITTEN). "
                "Strategin kombinerar Rick Rules kontrarianska cykellogik med "
                "taktisk swing-timing: köp pullback i upptrend, bekräftat av "
                "makromedvind (Copper/Gold-ratio, DXY, räntekurva). "
                "Tidshorisont: dagar till veckor.",
                _EMBER),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("2. När köper du",
                "Fem villkor måste vara uppfyllda:"
                + _ul([
                    "<b>Trendgates (alla 4):</b> Pris > 50V EMA · 20D EMA > 50D EMA · "
                    "RS positiv vs sektoretf (3 mån) · ≥3 stigande bottnar (6V).",
                    "<b>Pullback:</b> Pris inom 3% av 20D EMA — köp närheten av EMA, inte toppen.",
                    "<b>RSI(14) < 45:</b> Bekräftar att pullbacken är äkta, inte ett trendbrott.",
                    "<b>Makroscore ≥ 50/100:</b> Minst hälften av makrokontexten (Copper/Gold, "
                    "DXY, T10Y2Y, cykelfas) pekar i rätt riktning.",
                    "<b>Inga ingen-handel-flaggor:</b> Pris ej i chop-zon · ingen ATR-surge · "
                    "ej sen cykel (10å-percentil > 85%) · DXY ej rallying (+2% på 2V).",
                ], _TEXT),
                _EMBER),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("3. Positionsstorlek",
                _ul([
                    "Beräkna: <b>Positionsstorlek = (Kapital × 2%) / (2,5 × ATR14)</b>.",
                    "Stop = entry − 2,5 × ATR(14). Alltid strukturbaserat, aldrig fast %.",
                    "T1 = entry + 2 × risk (1:2 R:R). T2 = entry + 3 × risk (1:3 R:R).",
                    "Max 4 positioner per råvarusektor · max 8 totalt.",
                ], _TEXT),
                _AMBER),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _gs("4. När säljer du",
                _ul([
                    "<b>Stoploss:</b> Stop = entry − 2,5 × ATR(14). Flytta till BE efter ny HH.",
                    "<b>Trendbrott:</b> 20D EMA bryter under 50D EMA — stäng omedelbart.",
                    "<b>Ogiltighetsscenario:</b> Fyll i 'Ogiltigförklaras om' i setup-kortet "
                    "— exakt vad som måste hända för att teser är fel.",
                    "<b>Cykelbyte:</b> Om temat går från MITTEN → SEN/TOPP — minska.",
                    "<b>DXY-surge:</b> DXY upp > 2% på 2 veckor — råvaror möter mothåll.",
                ], _TEXT),
                _RED),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("5. Var i panelen — full kill chain",
                "<b>Använd i denna ordning:</b>"
                + _ul([
                    "<b>1. INTELLIGENCE → Odin's Blindspot → Tema-tavlan:</b> "
                    "vilka råvarukategorier är TIDIG/MITTEN i cykeln just nu?",
                    "<b>2. REGIME → 🌍 EMBER Regime:</b> "
                    "är makromiljön rätt? (PÅ / SELEKTIV / AV) — "
                    "handla aldrig mot regimen.",
                    "<b>3. SIGNALS → 🔥 EMBER → SKANNA:</b> "
                    "välj universum, tryck SKANNA, granska TOPP 3 — "
                    "bekräfta att alla 4 trendgates och båda entry-hårda gates är gröna.",
                    "<b>4. Discipline (i setup-kortet):</b> "
                    "fyll i 'Ogiltigförklaras om' och 'Trolig trigger' "
                    "<em>innan</em> du öppnar positionen.",
                    "<b>5. PORTFOLIO → Trade Journal:</b> "
                    "logga trade med entry, stop, target och exit-orsak.",
                ], _CYAN),
                _CYAN),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("6. Vanliga misstag",
                _ul([
                    "<b>Handlar när regimen är AV</b> — regimen är den yttersta garanten "
                    "mot fel makrotiming. AV = stå utanför, oavsett hur bra setup-kortet ser ut.",
                    "<b>Tror att DATA_GAP betyder godkänt</b> — DATA_GAP innebär att data saknas, "
                    "inte att indikatorn är bullish. Behandla DATA_GAP som AMBER eller sämre.",
                    "Köper utan att trendgaterna passerat — råvaran är i nedtrend och "
                    "du fångar en fallande kniv.",
                    "Ignorerar DXY-flaggan — ett DXY-rally &gt;2% på 2 veckor sätter "
                    "konstant press på råvarorna oavsett hur bra de ser ut tekniskt.",
                    "Tar entry när RSI &gt; 45 — pullen är inte klar, "
                    "du tar topp istället för botten.",
                    "Köper in i SEN/TOPP-cykelfas — risk/reward försämrat, priset är inprisat.",
                    "Glömmer ogiltighetsscenario — utan det vet du inte när du har fel.",
                ], _DIM),
                _RED),
            unsafe_allow_html=True,
        )


def _render_ember_full_ruleset() -> None:
    """Render EMBER complete 13-section ruleset with actual thresholds pulled from code."""
    # Pull named constants — fallback to hardcoded values if ember not installed
    try:
        from ember.config import (
            PULLBACK_EMA_PCT, RSI_ENTRY_MAX, RSI_PERIOD,
            ATR_STOP_MULT, ATR_PERIOD, RISK_PCT, MIN_RR,
            ATR_SURGE_PCT, ATR_SURGE_LOOKBACK_W,
            LATE_CYCLE_PCT, DXY_SURGE_PCT, DXY_SURGE_LOOKBACK_W,
            HIGHER_LOWS_MIN, HIGHER_LOWS_LOOKBACK_W, RS_LOOKBACK_DAYS,
            MAX_PER_SECTOR, MAX_TOTAL,
            MACRO_W_COPPER_GOLD, MACRO_W_DXY, MACRO_W_YIELD_CURVE, MACRO_W_CYCLE,
            CYCLE_BONUS_TIDIG, CYCLE_BONUS_MITTEN, CYCLE_BONUS_SEN, CYCLE_BONUS_TOPP,
        )
        from ember.regime import (
            DXY_SURGE_REGIME, CG_FLAT_PCT, YC_STEEPEN_PP, YC_INVERT_PP,
            TEMA_GREEN_PCT, TEMA_AMBER_PCT,
            VERDICT_PA, VERDICT_SELEKTIV, VERDICT_AV,
        )
    except ImportError:
        PULLBACK_EMA_PCT = 3.0; RSI_ENTRY_MAX = 45; RSI_PERIOD = 14
        ATR_STOP_MULT = 2.5; ATR_PERIOD = 14; RISK_PCT = 0.02; MIN_RR = 2.0
        ATR_SURGE_PCT = 40.0; ATR_SURGE_LOOKBACK_W = 2
        LATE_CYCLE_PCT = 85.0; DXY_SURGE_PCT = 2.0; DXY_SURGE_LOOKBACK_W = 2
        HIGHER_LOWS_MIN = 3; HIGHER_LOWS_LOOKBACK_W = 6; RS_LOOKBACK_DAYS = 63
        MAX_PER_SECTOR = 4; MAX_TOTAL = 8
        MACRO_W_COPPER_GOLD = 30; MACRO_W_DXY = 25
        MACRO_W_YIELD_CURVE = 25; MACRO_W_CYCLE = 20
        CYCLE_BONUS_TIDIG = 20.0; CYCLE_BONUS_MITTEN = 10.0
        CYCLE_BONUS_SEN = 0.0; CYCLE_BONUS_TOPP = -10.0
        DXY_SURGE_REGIME = 2.0; CG_FLAT_PCT = 2.0
        YC_STEEPEN_PP = 0.05; YC_INVERT_PP = -0.20
        TEMA_GREEN_PCT = 50.0; TEMA_AMBER_PCT = 25.0
        VERDICT_PA = "PÅ"; VERDICT_SELEKTIV = "SELEKTIV"; VERDICT_AV = "AV"

    rp = f"{RISK_PCT*100:.0f}%"

    st.markdown(
        f"<div style='text-align:center;margin:30px 0 20px;'>"
        f"<h2 style='color:{_EMBER};letter-spacing:0.15em;'>🔥 EMBER REGELVERK</h2>"
        f"<p style='color:{_DIM};font-size:0.72rem;'>Fullständigt regelset — 13 sektioner · "
        f"Trösklar hämtas direkt ur koden</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            _gs("1. Mål & Filosofi",
                "Rick Rule-inspirerad cykelstrategi. Köp råvaror och råvaruaktier "
                "<b>tidigt i cykeln</b> (TIDIG/MITTEN), bekräftat av makromedvind. "
                "Kombinerar kontrariansk timing med teknisk precision: "
                "pullback i upptrend, makroscore ≥ 50/100, och en makromiljö (regim) "
                "som är PÅ eller SELEKTIV. "
                "Tidshorisont: dagar till veckor.",
                _EMBER),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("2. Univers",
                _ul([
                    f"<b>Kurerad lista (snabb):</b> {13} ETF:er + {12} aktier — statisk seed-lista.",
                    "<b>Auto: Norden + US-råvaror (~150+ tickers):</b> "
                    "Börsdata råvarufilter (energi, gruv, metall, olja, gas) + "
                    "GDX/GDXJ/SIL/COPX/URA/XLE-konstituenter + Canada/UK-namn.",
                    "<b>Förfilter (Auto/Båda):</b> Omsättning &gt; 5 MSEK/dag (20D snitt) "
                    "<em>och</em> pris &gt; SMA(200) — utför batchar om 50 tickers.",
                ], _TEXT),
                _EMBER),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("3. Makro/Cykelfilter — EMBER Regime",
                f"<b>5 pelare, varje grön/amber/röd</b> — ≥4 gröna = {VERDICT_PA} (full size), "
                f"3 = {VERDICT_SELEKTIV} (halverad size), ≤2 = {VERDICT_AV} (inga trades):"
                + _ul([
                    f"<b>DOLLAR (DXY 4V):</b> fallande = GRÖN · &gt;{DXY_SURGE_REGIME}% = RÖD.",
                    f"<b>TILLVÄXTPULS (Copper/Gold 3M):</b> stigande (|Δ| &gt; {CG_FLAT_PCT}%) = GRÖN.",
                    f"<b>RÄNTEKURVA (T10Y2Y FRED 4V):</b> brantnar &gt;{YC_STEEPEN_PP}pp = GRÖN · "
                    f"inverterar djupare &lt;{YC_INVERT_PP}pp = RÖD.",
                    f"<b>TEMA-BREDD:</b> &gt;{TEMA_GREEN_PCT:.0f}% av 9 teman TIDIG/MITTEN "
                    f"+ positiv 3M sparkline = GRÖN.",
                    "<b>RISKAPTIT:</b> GDX/SPY 3M + HYG/TLT 1M båda positiva = GRÖN.",
                ], _TEXT)
                + f"<br/><b>Makroscore per setup (0–100):</b> Copper/Gold {MACRO_W_COPPER_GOLD}p · "
                f"DXY {MACRO_W_DXY}p · T10Y2Y {MACRO_W_YIELD_CURVE}p · Cykelfas {MACRO_W_CYCLE}p. "
                f"DATA_GAP = aldrig GRÖN.",
                _EMBER),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("4. Sentimentfilter",
                _ul([
                    "<b>US-tickers:</b> Short float, analytikerkonsensus (KÖP-andel), "
                    "put/call-ratio — viktad 0–100.",
                    "<b>Nordiska tickers (.ST/.OL/.CO/.HE):</b> DATA_GAP — "
                    "yfinance saknar sentimentdata. Behandla som neutral.",
                    "DATA_GAP-komponenter exkluderas från viktsumman — "
                    "totalpoängen baseras på tillgänglig data.",
                ], _TEXT),
                _EMBER),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("5. Trendfilter — 4 grindar (alla hårda, alla måste passera)",
                _ul([
                    "<b>T1 — Pris &gt; 50-veckors EMA:</b> bekräftar långsiktig upptrend.",
                    "<b>T2 — 20D EMA &gt; 50D EMA:</b> kortsiktig trend alignad med medium.",
                    f"<b>T3 — Relativ styrka vs sektor-ETF ({RS_LOOKBACK_DAYS} dagar / 3 mån):</b> "
                    "ticker måste leda sektorn.",
                    f"<b>T4 — ≥{HIGHER_LOWS_MIN} stigande bottnar på "
                    f"{HIGHER_LOWS_LOOKBACK_W} veckor:</b> strukturell prisuppgång bekräftad.",
                ], _TEXT),
                _EMBER),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("6. Entryfilter — 2 hårda + 4 bekräftande",
                "<b>Hårda (MÅSTE passa för elitcase):</b>"
                + _ul([
                    f"<b>E1 — Pullback till 20D EMA ≤ {PULLBACK_EMA_PCT}%:</b> "
                    "köp närheten av EMA, inte toppen av rörelsen.",
                    f"<b>E2 — RSI({RSI_PERIOD}) &lt; {RSI_ENTRY_MAX}:</b> "
                    "bekräftar att pullbacken är äkta, inte ett trendbrott.",
                ], _TEXT)
                + "<b>Bekräftande (stärker rankning):</b>"
                + _ul([
                    "<b>E3 — MACD-histogram: stigande botten</b> (12/26/9).",
                    "<b>E4 — Volym ≥ 1,0× 20D-snitt.</b>",
                    "<b>E5 — Bullish candlestick:</b> HAMMER / ENGULFING / BREAKOUT.",
                    f"<b>E6 — ATR({ATR_PERIOD}) sjunker under pullback</b> (lugn, ej panik).",
                ], _TEXT),
                _EMBER),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("7. Riskmodell",
                _ul([
                    f"<b>Stop:</b> entry − {ATR_STOP_MULT} × ATR({ATR_PERIOD}). "
                    "Alltid strukturbaserat.",
                    f"<b>Positionsstorlek:</b> (kapital × {rp}) / stop-avstånd.",
                    f"<b>T1 = entry + 2 × risk</b> (1:{MIN_RR:.0f} R:R).",
                    "<b>T2 = entry + 3 × risk</b> (1:3 R:R).",
                    f"<b>Portföljgränser:</b> max {MAX_PER_SECTOR} positioner per "
                    f"råvarusektor · max {MAX_TOTAL} totalt.",
                    f"<b>SELEKTIV regime:</b> halvera positionsstorleken automatiskt.",
                ], _TEXT),
                _AMBER),
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            _gs("8. Exitregler",
                _ul([
                    f"<b>Initial SL:</b> entry − {ATR_STOP_MULT} × ATR({ATR_PERIOD}).",
                    "<b>Flytta SL till breakeven</b> (entry) efter ny HH (higher high) — "
                    "inte tidigare, inte senare.",
                    "<b>Trendbrott:</b> 20D EMA bryter under 50D EMA → stäng omedelbart.",
                    "<b>Ogiltighetsscenario:</b> fyll i 'Ogiltigförklaras om'-fältet i "
                    "setup-kortet <em>innan</em> du öppnar positionen. "
                    "MANUELL BEDÖMNING.",
                    "<b>Cykelbyte:</b> tema går MITTEN → SEN/TOPP → minska eller stäng.",
                    f"<b>DXY-surge:</b> DXY upp &gt;{DXY_SURGE_PCT}% på "
                    f"{DXY_SURGE_LOOKBACK_W} veckor → råvaror möter mothåll.",
                ], _TEXT),
                _RED),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("9. No-Trade-Zoner — 4 flaggor",
                "Om <em>någon</em> flagga är aktiv blockeras entry:"
                + _ul([
                    "<b>F1 — Chop-zon:</b> pris MELLAN 20D och 50D EMA → "
                    "ingen riktning, ingen edge.",
                    f"<b>F2 — ATR-surge:</b> ATR upp &gt;{ATR_SURGE_PCT:.0f}% på "
                    f"{ATR_SURGE_LOOKBACK_W} veckor → volatilitetsexplosion, vänta.",
                    f"<b>F3 — Sen cykel:</b> 10å-percentil &gt;{LATE_CYCLE_PCT:.0f}% "
                    "→ risk/reward försämrat, topp-risk.",
                    f"<b>F4 — DXY-rally:</b> DXY upp &gt;{DXY_SURGE_PCT}% på "
                    f"{DXY_SURGE_LOOKBACK_W} veckor → råvarumotvind.",
                ], _TEXT),
                _RED),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("10. Setup-format — fältlista per kort",
                _ul([
                    "<b>Auto (beräknat):</b> Ticker, Typ (ETF/Aktie), Sektor/Tema, "
                    "Var i cykeln (TIDIG/MITTEN/SEN/TOPP + 10å-percentil), "
                    "Trendstatus (4 gates), Varför intressant, Marknaden ogillar (HAT-score), "
                    "Varför behövs (Nödvändighetspoäng), Tecken på undervärdering, "
                    "Pris, Entry (20D EMA), Stop, T1, T2, R:R, "
                    f"Position ({rp} risk, justerat för regim).",
                    "<b>Manuell bedömning (MANUELL — fylls i av handlaren):</b> "
                    "Ogiltigförklaras om · Trolig trigger · "
                    "LME-lager sjunker [checkbox] · "
                    "Capex-narrativ försämrad [checkbox].",
                ], _TEXT),
                _CYAN),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("11. Rangordning — asymmetry_score",
                f"<b>asymmetry_score = (R:R × makroscore/100) + cykelbonus</b><br/>"
                f"Cykelbonus: "
                f"TIDIG +{CYCLE_BONUS_TIDIG:.0f} · "
                f"MITTEN +{CYCLE_BONUS_MITTEN:.0f} · "
                f"SEN {CYCLE_BONUS_SEN:+.0f} · "
                f"TOPP {CYCLE_BONUS_TOPP:+.0f}<br/>"
                "TOPP 3 etiketter: <b>Säkrast</b> (högst poäng) · "
                "<b>Högst Potential</b> · <b>Mest Konträr</b>.",
                _EMBER),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("12. Output — tre nivåer",
                _ul([
                    "<b>TOPP 3 JUST NU:</b> de tre högst rankade elitcasen "
                    "med etiketter och nyckeltal.",
                    "<b>Alla elitcase — rankade:</b> fullständigt setup-kort med "
                    "alla 18 fält, gates och makroscore per ticker.",
                    "<b>Nästan-kandidater:</b> tickers som passerade minst en gate "
                    "men ej alla — visar exakt vad som saknas.",
                    "<b>INGA ELITCASE:</b> om ingenting passerar visas "
                    "'Disciplin är position. Vänta på rätt setup.'",
                ], _TEXT),
                _CYAN),
            unsafe_allow_html=True,
        )
        st.markdown(
            _gs("13. Kvalitetskrav — definition av elitcase",
                "<b>Elitcase kräver samtliga:</b>"
                + _ul([
                    "ALLA 4 trendgater = PASS (T1–T4).",
                    "BÅDA hårda entry-gatar = PASS (E1 pullback + E2 RSI).",
                    "INGA aktiva no-trade-flaggor (F1–F4 alla inaktiva).",
                    "DATA_GAP på trenddata = automatiskt FAIL (aldrig tyst pass).",
                ], _TEXT)
                + "<b>Near-miss</b> = minst en trendgate ELLER en entry-gate passerad men ej komplett.<br/>"
                "Makroscore ≥ 50/100 krävs inte formellt men lyfter rankningen.",
                _EMBER),
            unsafe_allow_html=True,
        )


def render_strategy_guides() -> None:
    """Sub-page with beginner strategy guides in plain Swedish."""
    st.markdown(
        f"<div style='text-align:center;padding:16px 0 12px 0;'>"
        f"<h2 style='color:{_CYAN};letter-spacing:0.12em;margin:0;'>STRATEGIGUIDER</h2>"
        f"<p style='color:{_DIM};font-size:0.75rem;letter-spacing:0.08em;'>"
        f"Enkla förklaringar i klartext — vad varje strategi gör, när du handlar och var i panelen du hittar det.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    sel = st.radio(
        "Välj strategi",
        ["Quality", "Deep Contrarian", "Wolf", "Viking", "🔥 Ember"],
        horizontal=True,
        key="sg_pick",
    )
    st.markdown("<hr style='border-color:rgba(0,229,255,0.15);margin:12px 0 20px 0;'>",
                unsafe_allow_html=True)
    if sel == "Quality":
        _guide_quality()
    elif sel == "Deep Contrarian":
        _guide_contrarian()
    elif sel == "Wolf":
        _guide_wolf()
    elif sel == "Viking":
        _guide_viking()
    elif sel == "🔥 Ember":
        _guide_ember()


# ------------------------------------------------------------------ #
#  Main render
# ------------------------------------------------------------------ #

def _level_badge(level: str) -> str:
    c = LEVEL_COLOR.get(level, _DIM)
    return (f"<span style='background:{c}22;border:1px solid {c};color:{c};"
            f"font-size:0.6rem;font-weight:700;letter-spacing:0.09em;"
            f"padding:2px 9px;border-radius:10px;'>{level.upper()}</span>")


def _hard_badge() -> str:
    return (f"<span style='background:{_RED}22;border:1px solid {_RED};color:{_RED};"
            f"font-size:0.55rem;font-weight:700;letter-spacing:0.06em;"
            f"padding:1px 6px;border-radius:3px;margin-left:8px;'>HÅRD REGEL</span>")


def _rule_row(rule, color: str) -> str:
    """One rule: what it is, why, and exactly where in the panel to look."""
    hard = _hard_badge() if getattr(rule, "hard", False) else ""
    return (
        f"<div style='border-left:3px solid {color};background:rgba(255,255,255,0.02);"
        f"border-radius:0 6px 6px 0;padding:10px 14px;margin-bottom:8px;'>"
        f"<div style='color:{_TEXT};font-size:0.9rem;font-weight:700;line-height:1.4;'>"
        f"<span style='color:{color};'>{rule.number}.</span> {rule.text}{hard}</div>"
        f"<div style='color:{_DIM};font-size:0.78rem;margin-top:3px;'>{rule.explanation}</div>"
        f"<div style='color:{color};font-size:0.72rem;margin-top:6px;line-height:1.5;'>"
        f"📍 {rule.panel_guide}</div></div>"
    )


def _pick_playbook(suffix: str):
    """Strategy picker, ordered easiest-first."""
    names = [PLAYBOOKS[k].name for k in LEARNING_ORDER]
    pick = st.radio("Strategi", names, horizontal=True,
                    key=f"rules_pb_{suffix}", label_visibility="collapsed")
    for k in LEARNING_ORDER:
        if PLAYBOOKS[k].name == pick:
            return PLAYBOOKS[k]
    return PLAYBOOKS["momentum"]


def _risk_box(pb) -> None:
    """The numbers, written out — no hunting through prose."""
    r = pb.risk
    rows = [("Risk per affär", r.risk_per_trade), ("Positionsstorlek", r.position_size),
            ("Antal positioner", r.max_positions), ("Stop loss", r.stop),
            ("Vinstmål", r.targets)]
    html = (f"<div style='border:1px solid {pb.color}55;background:{pb.color}0d;"
            f"border-radius:8px;padding:12px 16px;margin:6px 0 18px;'>"
            f"<div style='color:{pb.color};font-size:0.65rem;font-weight:700;"
            f"letter-spacing:0.14em;margin-bottom:8px;'>RISKMODELL</div>")
    for label, val in rows:
        html += (f"<div style='display:flex;gap:12px;padding:3px 0;font-size:0.8rem;'>"
                 f"<span style='color:{_DIM};min-width:140px;'>{label}</span>"
                 f"<span style='color:{_TEXT};font-weight:600;'>{val}</span></div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


def _page_start() -> None:
    """Kom igång — which strategy, in what order, and the routine."""
    st.markdown(
        f"<div style='text-align:center;padding:14px 0 6px;'>"
        f"<h2 style='color:{_CYAN};letter-spacing:0.12em;margin:0;'>KOM IGÅNG</h2>"
        f"<p style='color:{_DIM};font-size:0.78rem;margin:6px 0 0;'>"
        f"Fyra strategier, olika svårighetsgrad. Lär dig en i taget — "
        f"börja överst.</p></div>",
        unsafe_allow_html=True)

    st.markdown(
        f"<div style='border:1px solid {_GREEN};background:{_GREEN}18;border-radius:8px;"
        f"padding:12px 16px;margin:14px 0 20px;'>"
        f"<b style='color:{_GREEN};'>👉 Börja här:</b> "
        f"<span style='color:{_TEXT};'>Momentum Swing.</span> "
        f"<span style='color:{_DIM};font-size:0.82rem;'>Panelen räknar allt åt dig, "
        f"reglerna är mekaniska och rutinen tar ~15 min i veckan. Lär dig den helt "
        f"innan du går vidare — de andra kräver att du tolkar grafer själv.</span></div>",
        unsafe_allow_html=True)

    for i, key in enumerate(LEARNING_ORDER, start=1):
        pb = PLAYBOOKS[key]
        st.markdown(
            f"<div style='border:1px solid {pb.color}44;border-left:4px solid {pb.color};"
            f"border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:12px;"
            f"background:rgba(255,255,255,0.015);'>"
            f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>"
            f"<span style='color:{pb.color};font-size:1.05rem;font-weight:800;'>"
            f"{i}. {pb.name}</span>{_level_badge(pb.level)}</div>"
            f"<div style='color:{_TEXT};font-size:0.85rem;margin-top:4px;'>{pb.tagline}</div>"
            f"<div style='color:{_DIM};font-size:0.8rem;margin-top:8px;line-height:1.6;'>"
            f"{pb.idea}</div>"
            f"<div style='color:{_DIM};font-size:0.72rem;margin-top:10px;line-height:1.7;'>"
            f"⏱ <b style='color:{_TEXT};'>Horisont:</b> {pb.horizon}<br>"
            f"🎯 <b style='color:{_TEXT};'>Risk per affär:</b> {pb.risk.risk_per_trade}<br>"
            f"📍 <b style='color:{_TEXT};'>Finns i panelen:</b> {pb.where}</div>"
            f"</div>",
            unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{_CYAN}22;margin:24px 0 14px;'/>",
                unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{_CYAN};font-size:0.7rem;font-weight:700;"
        f"letter-spacing:0.14em;margin-bottom:10px;'>VECKORUTINEN — MOMENTUM SWING</div>",
        unsafe_allow_html=True)
    steps = PLAYBOOKS["momentum"].workflow
    flow = ""
    for n, step in enumerate(steps, start=1):
        flow += (f"<div style='display:flex;gap:12px;margin-bottom:9px;align-items:flex-start;'>"
                 f"<span style='background:{_GREEN};color:#000;font-weight:800;"
                 f"font-size:0.7rem;min-width:22px;height:22px;border-radius:50%;"
                 f"display:flex;align-items:center;justify-content:center;'>{n}</span>"
                 f"<span style='color:{_TEXT};font-size:0.82rem;line-height:1.5;'>{step}</span></div>")
    st.markdown(flow, unsafe_allow_html=True)


def _page_rules() -> None:
    """Handelsregler — full rules per strategy, with panel guides."""
    st.markdown(
        f"<div style='text-align:center;padding:14px 0 4px;'>"
        f"<h2 style='color:{_CYAN};letter-spacing:0.12em;margin:0;'>HANDELSREGLER</h2>"
        f"<p style='color:{_DIM};font-size:0.76rem;margin:6px 0 12px;'>"
        f"Varje regel visar 📍 exakt var i panelen du hittar svaret. "
        f"<span style='color:{_RED};'>HÅRD REGEL</span> = bryts aldrig, ingen "
        f"bedömningsfråga.</p></div>",
        unsafe_allow_html=True)

    pb = _pick_playbook("rules")
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;"
        f"margin:14px 0 4px;'>"
        f"<span style='color:{pb.color};font-size:1.15rem;font-weight:800;'>{pb.name}</span>"
        f"{_level_badge(pb.level)}"
        f"<span style='color:{_DIM};font-size:0.78rem;'>{pb.tagline}</span></div>"
        f"<div style='color:{_DIM};font-size:0.72rem;margin-bottom:8px;'>"
        f"📍 {pb.where}</div>",
        unsafe_allow_html=True)

    _risk_box(pb)

    if pb.workflow:
        with st.expander("🔄 Så gör du — steg för steg", expanded=False):
            for n, step in enumerate(pb.workflow, start=1):
                st.markdown(
                    f"<div style='color:{_TEXT};font-size:0.82rem;margin-bottom:6px;'>"
                    f"<b style='color:{pb.color};'>{n}.</b> {step}</div>",
                    unsafe_allow_html=True)

    col_in, col_out = st.columns(2)
    with col_in:
        st.markdown(_section_header_html("Entry — när du får köpa",
                                         f"{len(pb.entry)} regler", pb.color),
                    unsafe_allow_html=True)
        st.markdown("".join(_rule_row(r, pb.color) for r in pb.entry),
                    unsafe_allow_html=True)
    with col_out:
        st.markdown(_section_header_html("Exit — när du måste sälja",
                                         f"{len(pb.exit)} regler", _RED),
                    unsafe_allow_html=True)
        st.markdown("".join(_rule_row(r, _RED) for r in pb.exit),
                    unsafe_allow_html=True)

    if pb.mindset:
        st.markdown(_section_header_html("Mindset", "Handla planen — inte prediktionen",
                                         _YELLOW), unsafe_allow_html=True)
        st.markdown("".join(_rule_row(r, _YELLOW) for r in pb.mindset),
                    unsafe_allow_html=True)

    if pb.pitfalls:
        st.markdown(_section_header_html("Vanliga misstag",
                                         "Det som faktiskt kostar pengar", _RED),
                    unsafe_allow_html=True)
        st.markdown("".join(
            f"<div style='color:{_DIM};font-size:0.82rem;padding:5px 0 5px 18px;"
            f"border-left:2px solid {_RED}55;margin-bottom:4px;line-height:1.5;'>"
            f"⚠️ {p}</div>" for p in pb.pitfalls), unsafe_allow_html=True)

    if pb.note:
        st.info(pb.note)


def _page_cheatsheet() -> None:
    """Fusklapp — one screen per strategy, printable."""
    st.markdown(
        f"<div style='text-align:center;padding:14px 0 4px;'>"
        f"<h2 style='color:{_CYAN};letter-spacing:0.12em;margin:0;'>FUSKLAPP</h2>"
        f"<p style='color:{_DIM};font-size:0.76rem;margin:6px 0 12px;'>"
        f"Allt du behöver komma ihåg — på en skärm. Ha den uppe medan du handlar.</p></div>",
        unsafe_allow_html=True)

    pb = _pick_playbook("cheat")
    st.markdown(
        f"<div style='color:{pb.color};font-size:1.1rem;font-weight:800;"
        f"margin:12px 0 8px;'>{pb.name}</div>", unsafe_allow_html=True)

    cells = ""
    for label, val in pb.cheatsheet:
        cells += (f"<div style='border:1px solid {pb.color}33;border-radius:6px;"
                  f"padding:9px 12px;background:rgba(255,255,255,0.02);'>"
                  f"<div style='color:{_DIM};font-size:0.62rem;letter-spacing:0.1em;"
                  f"text-transform:uppercase;'>{label}</div>"
                  f"<div style='color:{_TEXT};font-size:0.86rem;font-weight:600;"
                  f"margin-top:2px;'>{val}</div></div>")
    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));"
        f"gap:8px;margin-bottom:20px;'>{cells}</div>", unsafe_allow_html=True)

    hard = [r for r in (pb.entry + pb.exit) if getattr(r, "hard", False)]
    if hard:
        st.markdown(
            f"<div style='border:1px solid {_RED};background:{_RED}12;border-radius:8px;"
            f"padding:12px 16px;'>"
            f"<div style='color:{_RED};font-size:0.68rem;font-weight:700;"
            f"letter-spacing:0.14em;margin-bottom:8px;'>HÅRDA REGLER — BRYTS ALDRIG</div>"
            + "".join(
                f"<div style='color:{_TEXT};font-size:0.84rem;padding:3px 0;'>"
                f"<span style='color:{_RED};'>✕</span> {r.text}</div>" for r in hard)
            + "</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Main render
# ------------------------------------------------------------------ #

def render_rules_page() -> None:
    sub = st.radio(
        "",
        ["🚀 KOM IGÅNG", "📋 HANDELSREGLER", "⚡ FUSKLAPP", "📖 STRATEGIGUIDER"],
        horizontal=True,
        key="rules_sub",
    )
    st.markdown("<hr style='border-color:rgba(0,229,255,0.15);margin:8px 0 16px 0;'>",
                unsafe_allow_html=True)

    if sub == "🚀 KOM IGÅNG":
        _page_start()
    elif sub == "📋 HANDELSREGLER":
        _page_rules()
    elif sub == "⚡ FUSKLAPP":
        _page_cheatsheet()
    else:
        render_strategy_guides()


# ------------------------------------------------------------------ #
#  Inline rules helper — renders a collapsible rules section
# ------------------------------------------------------------------ #

def render_inline_rules(strategy: str) -> None:
    """Render a collapsible rules section for a given strategy.

    strategy: 'wolf', 'viking', or 'alpha'
    """
    try:
        pb = PLAYBOOKS.get(strategy.lower())
        if pb is None:
            return
        with st.expander(f"Regler — {pb.name}", expanded=False):
            st.caption(f"Risk per affär: {pb.risk.risk_per_trade} · "
                       f"Stop: {pb.risk.stop}")
            st.markdown("".join(_rule_row(r, pb.color) for r in pb.entry),
                        unsafe_allow_html=True)
            st.markdown("".join(_rule_row(r, _RED) for r in pb.exit),
                        unsafe_allow_html=True)
    except Exception:
        pass
