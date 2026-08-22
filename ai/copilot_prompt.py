"""
ai/copilot_prompt.py — vad Copiloten får se, och vad den får säga.

Prompten byggs av det regelmotorn REDAN kommit fram till. Modellen får inte
räkna om R:R, risk eller regelutfall — de siffrorna skickas färdiga och
instruktionen säger uttryckligen att de ska citeras, inte räknas om. Det är
CLAUDE.md:s regel i praktiken: motorerna äger besluten, AI:n förklarar dem.

Rena funktioner, ingen Streamlit och inget nätverk — så prompten går att
testa utan API-nyckel.
"""

from __future__ import annotations

SYSTEM = """Du är en analytisk andrahandsläsare i en svensk tradingpanel.

Regelmotorn har redan gjort bedömningen. Ditt jobb är att förklara den och
peka på det som är svagt — inte att göra om den.

Absoluta krav:
- Räkna ALDRIG om R:R, risk i procent eller regelutfall. Siffrorna du får är
  panelens; citera dem exakt som de står.
- Föreslå ALDRIG en positionsstorlek, och ändra aldrig stop eller target.
- Motsäg aldrig statusen. Står det REJECT förklarar du varför den föll, du
  argumenterar inte för affären.
- Hittar du inget att invända mot, skriv det. Fyll inte ut med brasklappar.
- Underlaget listar granskningsarkets IFYLLDA fält. Be ALDRIG användaren
  kontrollera något som redan står besvarat där — bygg vidare på svaren.
  Står det "jurisdiktion Ja" är landrisken granskad; ifrågasätt värdet om du
  vill, men be inte om en kontroll som redan är gjord.

Skriv svenska, 120–180 ord, i löpande text utan rubriker. Var konkret:
vilken regel är svagast, vad skulle få dig att ändra dig, vad ska kollas
manuellt innan avslut. Avsluta med den enskilt viktigaste frågan att besvara
före ett köp."""


def _bullets(items) -> str:
    return "\n".join(f"  - {t}" for t in items) if items else "  (inga)"


def _market_block(snap) -> str:
    """Marknadsdatan, om den finns.

    Utan den här saknade modellen allt utom tre priser och kunde bara
    upprepa regeltexterna tillbaka.
    """
    if snap is None:
        return ("Marknadsdata: SAKNAS — kursdatan gick inte att hämta. Kommentera "
                "inte trend, volym eller nivåernas rimlighet mot kursen; du har "
                "inget underlag för det.")

    def _n(value, suffix="", fmt="{:.2f}"):
        return "–" if value is None else (fmt.format(value) + suffix)

    return f"""Marknadsdata ({snap.ticker}, per {snap.as_of}, {snap.bars} dagar):
  Kurs {_n(snap.price)}
  ATR(14) {_n(snap.atr14)} = {_n(snap.atr_pct, ' %')} av kursen
  EMA50 {_n(snap.ema50)} ({_n(snap.dist_ema50_pct, ' %', '{:+.1f}')} mot kurs)
  EMA200 {_n(snap.ema200)} ({_n(snap.dist_ema200_pct, ' %', '{:+.1f}')} mot kurs)
  RSI(14) {_n(snap.rsi14, '', '{:.0f}')}
  Volym mot 20-dagarssnitt {_n(snap.vol_ratio, '×')}
  52v-intervall {_n(snap.low_52w)}–{_n(snap.high_52w)}, \
{_n(snap.from_high_pct, ' %', '{:+.1f}')} från toppen
  20-dagars swing-low {_n(snap.swing_low_20)} · swing-high {_n(snap.swing_high_20)}
  Avkastning 1 mån {_n(snap.ret_1m_pct, ' %', '{:+.1f}')} · \
3 mån {_n(snap.ret_3m_pct, ' %', '{:+.1f}')}"""


def _levels_block(assessment, alternatives) -> str:
    """Vad de valda nivåerna innebär, och vilka alternativ motorn räknat fram."""
    if assessment is None:
        return ""
    lines = ["Nivåbedömning (räknad av panelen):"]
    if assessment.notes:
        lines += [f"  - {n}" for n in assessment.notes]
    else:
        lines.append("  - Inga invändningar mot nivåerna.")
    if alternatives:
        lines.append("Alternativa stoppnivåer motorn räknat fram:")
        for s in alternatives:
            lines.append(f"  - {s.name}: {s.price:g} ({s.risk_pct:.1f} % risk) "
                         f"→ target {s.target_for_min_rr:g} för 2:1. {s.why}")
    return "\n".join(lines)


def _cycle_block(cycle_state, blindspot, market_phase=None) -> str:
    """Cykelläget ur rotationsfliken och senaste Blindspot-raden.

    Modellen får läget som fakta med källa och datum — inte som en åsikt den
    ska bilda sig. Är läget Vila ska den förklara varför kandidaten faller,
    inte leta skäl runt det.
    """
    lines = []
    if cycle_state is not None:
        warn = "".join(f" · VARNING: {w}" for w in cycle_state.get("warnings", []))
        lines.append(
            f"Cykelläge (rotationsflikens Triple Signal, "
            f"{cycle_state.get('month') or 'okänd månad'}): "
            f"{cycle_state.get('commodity')} = {cycle_state.get('status')} "
            f"{cycle_state.get('sum')}/{cycle_state.get('max')} — "
            f"{cycle_state.get('why')}{warn}")
    if market_phase is not None:
        lines.append(
            f"Marknadscykelfas (Market Cycle Engine): "
            f"{market_phase.get('phase', '?').replace('_', ' ')} med "
            f"{market_phase.get('confidence')} % säkerhet. Playbookens "
            f"fasregler avgör köp/håll/sälj — motsäg dem inte.")
    if blindspot is not None:
        lines.append(
            f"Blindspot (senast sparade rapport, {str(blindspot.get('timestamp', ''))[:10]}): "
            f"opportunity {blindspot.get('opportunity')} · hat {blindspot.get('hat')} · "
            f"styrka {blindspot.get('strength')} · katalysator {blindspot.get('catalyst')} "
            f"· sektor {blindspot.get('sector', '–')}. Rapporten är en "
            f"ögonblicksbild — kommentera dess ålder om den är gammal.")
    return "\n".join(lines)


def build_prompt(ticker: str, strategy: str, status: str,
                 entry: float, stop: float, target: float,
                 rr, risk_pct,
                 passed: list, manual: list, failed: list,
                 risk_per_trade: str = "",
                 snapshot=None, assessment=None, alternatives=None,
                 cycle_state=None, blindspot=None,
                 review_lines=None, market_phase=None) -> str:
    """Underlaget, i klartext. Allt är redan räknat av motorn."""
    rr_txt = "–" if rr is None else f"{rr:.1f}x"
    risk_txt = "–" if risk_pct is None else f"{risk_pct:.1f} %"
    frame = (f"\n  Strategins riskram: {risk_per_trade}"
             if risk_per_trade else "")

    return f"""Kandidat: {ticker}
Strategi: {strategy}
Panelens status: {status}

Nivåer (satta av användaren, redan validerade av motorn):
  Entry {entry:g} · Stop {stop:g} · Target {target:g}
  R:R {rr_txt} · risk mot entry {risk_txt}{frame}

{_market_block(snapshot)}

{_cycle_block(cycle_state, blindspot, market_phase)}

{chr(10).join(review_lines) if review_lines else ""}

{_levels_block(assessment, alternatives)}

Regler som PASSERADE ({len(passed)}):
{_bullets(passed)}

Regler som kräver MANUELL kontroll ({len(manual)}):
{_bullets(manual)}

Regler som FÖLL ({len(failed)}):
{_bullets(failed)}

Kommentera underlaget enligt instruktionen."""


# ── Journalgranskning ────────────────────────────────────────────────────────
# Den här är den enda av modulens prompter som kan påverka avkastningen
# mätbart: den läser vad som FAKTISKT hänt, inte vad som kan komma att hända.

REVIEW_SYSTEM = """Du granskar en svensk swingtraders egen affärsjournal.

Du får summerad statistik och de senaste avslutade affärerna. Leta mönster i
det som faktiskt hänt — inte i marknaden.

Absoluta krav:
- Räkna ALDRIG om statistiken. Siffrorna är journalens; citera dem.
- Uttala dig inte om enskilda affärers utfall som om de var förutsägbara.
- Säg tydligt ifrån när underlaget är för litet. Under 20 avslutade affärer
  är vinstandel och payoff-kvot brus, och då är det det du ska svara.
- Föreslå inga nya strategier. Håll dig till de regler som redan finns.

Sikta på tre konkreta observationer: vilken utgångsregel som kostar mest,
om någon setup bär resultatet, och om innehavstiden avviker från det
strategin är byggd för. Avsluta med EN sak att ändra till nästa månad.
Svenska, 150–220 ord."""


def build_review_prompt(stats: dict, recent: list, min_trades: int = 20) -> str:
    """Statistikbladet plus de senaste avsluten, i klartext."""
    def _n(value, suffix="", fmt="{:.1f}"):
        return "–" if value is None else (fmt.format(value) + suffix)

    setups = "\n".join(
        f"  {key}: {v['count']} affärer, snitt {_n(v['avg_pct'], ' %')}"
        for key, v in (stats.get("setups") or {}).items())
    exits = "\n".join(f"  {key}: {n}"
                      for key, n in (stats.get("exits") or {}).items())
    trades = "\n".join(
        f"  {t.get('date', '?')} {t.get('ticker', '?')} "
        f"{t.get('strategy', '')} · resultat {_n(t.get('pnl_pct'), ' %')} "
        f"· {_n(t.get('r_multiple'), 'R', '{:.2f}')} "
        f"· ut via {t.get('sell_rule') or 'okänt'} "
        f"· plan följd: {t.get('followed_plan', 'ej angivet')}"
        for t in (recent or [])) or "  (inga avslutade affärer)"

    closed = stats.get("closed", 0)
    warning = ("\nOBS: underlaget är under {} avslutade affärer. Vinstandel och "
               "payoff-kvot är brus här — säg det.".format(min_trades)
               if not stats.get("enough") else "")

    return f"""Journalstatistik ({closed} avslutade affärer):
  Vinstandel {_n(stats.get('win_rate'), ' %')}
  Payoff-kvot {_n(stats.get('payoff'), '', '{:.2f}')}
  Snitt-R {_n(stats.get('avg_r'), 'R', '{:.2f}')}
  Snittresultat {_n(stats.get('avg_pct'), ' %')}
  Snitt innehavstid {_n(stats.get('avg_days'), ' dagar', '{:.0f}')}

Setup:
{setups or '  (ingen setup angiven)'}

Utgångar per säljregel:
{exits or '  (ingen säljregel angiven)'}

Senaste avslutade affärer:
{trades}
{warning}

Granska enligt instruktionen."""
