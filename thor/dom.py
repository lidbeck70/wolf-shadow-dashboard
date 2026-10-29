"""
thor/dom.py — Arkens dom: Regim-fliken i Thor för arkstrategierna (Rick Rule, Durrett, Tiggre, Insider — och
Håvarna). Det du själv fyllt i arket ställs samman med nyckeldatan ur regimmotorerna till ETT domslut:

  VAD SKA JAG GÖRA?   KÖP · BEVAKA · AVVAKTA · NEJ — deterministiskt ur underlaget, aldrig ur en modell
  Underlaget          granskningen i arket (review_link: PASS/MANUAL/FAIL), kontrollerna DS/AQS/CSM, scorecardets
                      sju kryss, Confidence/Durrett-motorn, och cykeln: Triple Signal (rotationen), Valhall-temat
                      eller -segmentet, Odins blindspot, BERSERK-fasen — och fiat-miljön som fakta
  Copiloten           valfri kommentar på domen (ai.openai_client, samma instruktion som Copilot-fliken) —
                      förklarar, motsäger aldrig domslutet

Domslutet: ett FAIL i underlaget → NEJ (arkets egna spärrar, motorns nej, en betygsatt Vila eller cykeltopp).
Utan FAIL: granskningen PASS och allt annat PASS → KÖP; granskningen PASS med MANUAL kvar → BEVAKA; granskningen
MANUAL eller bolaget inte i arket → AVVAKTA. Inga regler ändras — varje rad är ett ark eller en motor som redan
finns, och domen läser dem ordagrant. Riktiga order läggs alltid manuellt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from typing import Callable, Optional

from thor import cards
from thor import registry as reg
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, GREY, RED, TEXT

PASS, MANUAL, FAIL = "PASS", "MANUAL", "FAIL"
KOP, BEVAKA, AVVAKTA, NEJ = "KÖP", "BEVAKA", "AVVAKTA", "NEJ"
WORD_COLOR = {KOP: GREEN, BEVAKA: AMBER, AVVAKTA: GREY, NEJ: RED}
# Vilka granskningar (review_link.SHEET) som gäller för Thor-strategin, i ordning
REVIEW_KEYS = {"rule": ("rule", "royalty"), "durrett": ("durrett", "sprott"), "tiggre": ("tiggre",),
               "insider": ("insider",), "hav": ()}
COMMODITY_STRATEGIES = ("rule", "durrett", "tiggre")
AI_STATE = "thor_dom_ai"


@dataclass
class Check:
    key: str
    label: str
    status: str                  # PASS | MANUAL | FAIL
    note: str = ""
    source: str = ""


@dataclass
class Dom:
    strategy: str
    word: str
    sentence: str
    detail: str = ""
    color: str = GREY
    checks: list = field(default_factory=list)
    facts: list = field(default_factory=list)      # kpi_row-poster (etikett, värde, färg, underrad)
    held: bool = False
    review_key: Optional[str] = None

    def by_status(self, status: str) -> list:
        return [c for c in self.checks if c.status == status]


# ── Cykeldatan (motorerna) ──────────────────────────────────────────────────
def cycle_context(ident, data: dict, sheets: dict, prices: Optional[Callable] = None,
                  theme_board: Optional[Callable] = None, fiat: Optional[Callable] = None) -> dict:
    """{rotation, rotation_key, valhall, blindspot, berserk, fiat, errors} för aktien — varje motor för sig,
    ett fel blir en rad i errors. theme_board() = blindspot.theme_board.build_theme_board, fiat() =
    fiat_debasement.environment.current('USD') (bytbara i test)."""
    from thor import ravara as rv
    out: dict = {"rotation": None, "rotation_key": None, "valhall": None, "blindspot": None, "berserk": None,
                 "fiat": None, "errors": {}}
    theme = ident.theme or ""
    board = (data or {}).get("board") or {}
    inner = board.get("board") if isinstance(board.get("board"), dict) else board
    try:
        if inner and (ident.segment or theme):
            from thor.dossier import valhall_tag
            from valhall import cycle as vcy
            tag = valhall_tag(inner, ident)                       # temat ur sin egen segmentrad (T8)
            g = "t" if theme and tag.get("t_status") not in (None, vcy.NODATA) else "n" if ident.nordic else "g"
            if tag.get(f"{g}_status") not in (None, vcy.NODATA):
                out["valhall"] = {"view": {"t": "temat", "n": "Norden", "g": "globalt"}[g],
                                  "status": tag.get(f"{g}_status"), "phase": tag.get(f"{g}_phase"),
                                  "cykel": tag.get(f"{g}_cykel")}
    except Exception as exc:
        out["errors"]["Valhall"] = str(exc)
    if theme:
        try:
            out["rotation_key"] = rv.rotation_key(theme)
            if out["rotation_key"]:
                out["rotation"] = rv.rotation_state(theme, (sheets or {}).get("rotation"))
        except Exception as exc:
            out["errors"]["Rotationen"] = str(exc)
        try:
            key = rv.blindspot_key(theme)
            if key:
                if theme_board is None:
                    from blindspot import theme_board as tb
                    theme_board = tb.build_theme_board
                themes = theme_board() or []
                r = next((t for t in themes if getattr(t, "key", None) == key), None)
                if r is not None and not getattr(r, "error", None):
                    out["blindspot"] = {"cykel": r.cykel_label, "hat": r.hat_score, "pct": r.percentile_10y,
                                        "score": r.blindspot_score, "rank": themes.index(r) + 1, "n": len(themes)}
        except Exception as exc:
            out["errors"]["Odins blindspot"] = str(exc)
        try:
            from berserk import live
            drivers = live.load_drivers([theme], getter=prices)
            row = next((r for r in live.theme_states(drivers) if r.get("theme") == theme), None)
            if row and row.get("driver"):
                out["berserk"] = {"phase": row.get("phase"), "driver": row.get("driver"), "close": row.get("close"),
                                  "date": row.get("date")}
        except Exception as exc:
            out["errors"]["BERSERK"] = str(exc)
    try:
        if fiat is None:
            from fiat_debasement import environment as env
            fiat = env.current
        out["fiat"] = fiat("USD")
    except Exception as exc:
        out["errors"]["Debasement"] = str(exc)
    return out


# ── Domen (ren funktion) ────────────────────────────────────────────────────
def _review_for(strategy: str, sheets: dict) -> tuple:
    """(nyckel, granskningen) — första arket i REVIEW_KEYS som har bolaget, annars (första nyckeln, None)."""
    keys = REVIEW_KEYS.get(strategy, ())
    reviews = (sheets or {}).get("reviews") or {}
    for k in keys:
        if reviews.get(k):
            return k, reviews[k]
    return (keys[0] if keys else None), None


def _scorecard_check(strategy: str, sheets: dict, ticker: str, scorecard_fn: Optional[Callable]) -> Check:
    cards_ = (sheets or {}).get("scorecard") or {}
    src = "🎯 Scorecard"
    if not cards_:
        return Check("scorecard", "Scorecardet", MANUAL,
                     "inget scorecard — fyll i de sju kryssen (luckor = INGEN AFFÄR)", src)
    label = reg.label(strategy).lower()
    key = next((k for k, c in cards_.items() if label in str((c or {}).get("strategy") or "").lower()),
               next(iter(cards_)))
    card = cards_[key] or {}
    entry = ((sheets or {}).get("scorecard_entries") or {}).get(key)     # samma kandidatrad som 🎯 Scorecard läser
    try:
        if scorecard_fn is None:
            from scorecard import gate_state as scorecard_fn
        g = scorecard_fn(card, entry)
    except Exception as exc:
        return Check("scorecard", "Scorecardet", MANUAL, f"kunde inte läsas: {exc}", src)
    strat = str(card.get("strategy") or "")
    if g.get("ready"):
        return Check("scorecard", "Scorecardet", PASS, f"KLAR FÖR KÖP · {strat}".strip(" ·"), src)
    missing = list(g.get("missing") or [])
    tail = " …" if len(missing) > 4 else ""
    return Check("scorecard", "Scorecardet", MANUAL, "saknar: " + " · ".join(missing[:4]) + tail, src)


def _engine_checks(strategy: str, sheets: dict, ticker: str, company_fn: Optional[Callable],
                   analyze_fn: Optional[Callable]) -> tuple:
    """([Check], [fakta]) ur Confidence/Durrett-motorn och Wolf Asymmetry för arkets rad."""
    checks, facts = [], []
    label, src = "Confidence/Durrett-motorn", "🧭 Durrett"
    try:
        if company_fn is None:
            from thor.arken import company_for as company_fn
        company = company_fn((sheets or {}).get("confidence_store"), ticker)
    except Exception:
        company = None
    if company is None:
        if strategy in COMMODITY_STRATEGIES:
            checks.append(Check("engine", label, MANUAL, "bolaget finns inte i Confidence/Durrett-arket", src))
        return checks, facts
    try:
        if analyze_fn is None:
            from confidence.reports import analyze as analyze_fn
        a = analyze_fn(company)
        rec = str(a.recommendation)
        status = PASS if rec == "BUY CANDIDATE" else MANUAL if rec == "WATCH" else FAIL
        checks.append(Check("engine", label, status, f"{rec} — {a.recommendation_why}", src))
        col = GREEN if status == PASS else AMBER if status == MANUAL else RED
        facts.append(("Case · Confidence", f"{a.case.total:g} · {a.confidence.total:g}", col,
                      f"{a.case.rating} · {a.confidence.band}"))
    except Exception as exc:
        checks.append(Check("engine", label, MANUAL, f"motorn: {exc}", src))
    try:
        import asymmetry
        w = asymmetry.analyze(company)
        be, up = w.break_even, w.base_upside_pct
        sub = f"break-even {be.margin_pct:+.0f} % · {be.band}" if be.margin_pct is not None else str(be.band or "")
        facts.append(("Wolf Asymmetry", f"{up:+.0f} %" if up is not None else "—",
                      GREEN if up and up > 0 else GREY, sub))
    except Exception:
        pass
    return checks, facts


def _cycle_checks(strategy: str, ctx: dict) -> tuple:
    """([Check], [fakta]) ur cykeldatan: Triple Signal, Valhall, Blindspot, BERSERK-fasen; fiat som fakta."""
    checks, facts = [], []
    ctx = ctx or {}
    rot = ctx.get("rotation")
    if ctx.get("rotation_key"):
        if rot is None:
            checks.append(Check("rotation", "Triple Signal (rotationen)", MANUAL,
                                "inte betygsatt — sätt månadens betyg i Råvarurotation; obetygsatt är inte godkänt",
                                "Råvarurotation"))
        else:
            st = str(rot.get("status") or "")
            status = PASS if st == "AGERA" else MANUAL if st == "Bevaka" else FAIL
            warn = "".join(f" · VARNING: {w}" for w in rot.get("warnings") or [])
            checks.append(Check("rotation", "Triple Signal (rotationen)", status,
                                f"{st} {rot.get('sum')}/{rot.get('max')} ({rot.get('month') or 'okänd månad'}) — "
                                f"{rot.get('why')}{warn}", "Råvarurotation"))
            facts.append(("Triple Signal", f"{st} {rot.get('sum')}/{rot.get('max')}",
                          GREEN if status == PASS else AMBER if status == MANUAL else RED, rot.get("month") or ""))
    v = ctx.get("valhall")
    if v:
        cyk, stt = str(v.get("cykel") or ""), str(v.get("status") or "")
        head = f"{v['view']} {stt} · fas {v.get('phase') or '—'} · cykel {cyk or '—'}"
        if v.get("view") == "temat" or strategy in COMMODITY_STRATEGIES:
            status = {"TIDIG": PASS, "MITTEN": PASS, "SEN": MANUAL, "TOPP": FAIL}.get(cyk, MANUAL)
            why = {"TIDIG": "tidig cykel, contrarian-läge", "MITTEN": "mitt i cykeln",
                   "SEN": "sen cykel — ingen ny position utan starkt case",
                   "TOPP": "cykeltopp — " + (f"{reg.label(strategy)} köper inte här" if strategy in COMMODITY_STRATEGIES
                                             else "ingen ny position")}.get(cyk, "cykelläget saknas")
        else:
            status = PASS if stt == "GRÖN" else MANUAL
            why = "" if stt == "GRÖN" else "segmentet är inte GRÖN: arkets egen regel avgör"
        checks.append(Check("valhall", "Valhall-kartan", status, head + (f" — {why}" if why else ""), "⚡ Valhall"))
        from ui.tokens import regime_color
        col = regime_color(stt) if stt in ("GRÖN", "GUL", "RÖD") else GREY
        facts.append((f"Valhall · {v['view']}", stt or "—", col, " · ".join(x for x in (v.get("phase"), cyk) if x)))
    b = ctx.get("blindspot")
    if b:
        cyk = str(b.get("cykel") or "")
        status = PASS if cyk in ("TIDIG", "MITTEN") else MANUAL
        place = f"plats {b.get('rank')} av {b.get('n')}"
        checks.append(Check("blindspot", "Odins blindspot", status,
                            f"cykel {cyk or '—'} · hat {b.get('hat'):.0f}/100 · 10 år P{b.get('pct'):.0f} · {place}"
                            + ("" if status == PASS else " — proxy säger sen cykel/topp"), "Odins blindspot"))
        facts.append(("Blindspot", f"{cyk or '—'} · hat {b.get('hat'):.0f}", GOLD, place))
    k = ctx.get("berserk")
    if k:
        ph = str(k.get("phase") or "")
        status = PASS if ph in ("BAISSE · VÄNDER", "STARK", "UPPTREND") else MANUAL
        checks.append(Check("berserk", "BERSERK-fasen (drivaren)", status,
                            f"{ph} · {k.get('driver')} {k.get('close') or ''} {k.get('date') or ''}".strip()
                            + {"BAISSE · VÄNDER": " — ⚡ vändningen", "BAISSE": " — baisse utan vändning, tidigt",
                               "SVAG": " — svag, ingen trend"}.get(ph, ""), "🪓 BERSERK"))
        facts.append(("BERSERK", ph, GREEN if status == PASS else AMBER, str(k.get("driver") or "")))
    f = ctx.get("fiat")
    if f:
        sub = f"{f.get('value')} · {f.get('as_of')}" if f.get("value") is not None else str(f.get("as_of") or "")
        facts.append(("Fiat-miljö", str(f.get("level") or "—"), CYAN, sub))
    return checks, facts


def judge(strategy: str, sheets: dict, ctx: Optional[dict] = None, ticker: str = "", held: bool = False,
          scorecard_fn: Optional[Callable] = None, company_fn: Optional[Callable] = None,
          analyze_fn: Optional[Callable] = None) -> Dom:
    """Domslutet ur arket, kontrollerna, scorecardet, motorerna och cykeldatan — ren funktion."""
    sheets = sheets or {}
    key, rev = _review_for(strategy, sheets)
    checks: list = []
    if key is None:
        checks.append(Check("review", "Granskningen", MANUAL,
                            "Håvarna är en lista, inte ett ark — granska i Rick Rule, Poängmodellen eller Tiggre",
                            "Håvarna"))
    elif rev is None:
        import review_link
        checks.append(Check("review", f"Granskningen i {review_link.sheet_name(key)}", MANUAL,
                            "bolaget finns inte i arket — lägg in och granska det först (screener → granskning → köp)",
                            review_link.sheet_name(key)))
    else:
        checks.append(Check("review", f"Granskningen i {rev.get('sheet') or key}", str(rev.get("status") or MANUAL),
                            str(rev.get("note") or ""), str(rev.get("sheet") or key)))
        controls = list(rev.get("controls") or [])
        for st_, lbl, note in controls:
            checks.append(Check(f"ctl_{lbl}", f"Kontrollen {lbl}", str(st_), str(note or ""), "arket"))
        if not controls and rev.get("found"):
            checks.append(Check("ctl", "Kontrollerna", PASS,
                                "inga krävs för positionens storlek (proportionalitetsregeln)", "arket"))
    checks.append(_scorecard_check(strategy, sheets, ticker, scorecard_fn))
    ec, ef = _engine_checks(strategy, sheets, ticker, company_fn, analyze_fn)
    cc, cf = _cycle_checks(strategy, ctx or {})
    checks += ec + cc
    review = checks[0]
    fails = [c for c in checks if c.status == FAIL]
    manuals = [c for c in checks if c.status == MANUAL]
    if fails:
        word, sentence = NEJ, f"{fails[0].label}: {fails[0].note}"
    elif review.status == PASS and not manuals:
        word = KOP
        sentence = "Arket säger köp och underlaget är grönt rakt igenom — lägg ordern manuellt enligt playbooken."
    elif review.status == PASS:
        word = BEVAKA
        sentence = "Granskningen är klar — kvar att göra: " + "; ".join(f"{c.label} ({c.note})" for c in manuals[:3])
    else:
        word, sentence = AVVAKTA, f"{review.label}: {review.note}"
    n_pass = sum(1 for c in checks if c.status == PASS)
    detail = f"{n_pass} PASS · {len(manuals)} MANUAL · {len(fails)} FAIL" + (" · INNEHAV" if held else "")
    return Dom(strategy, word, sentence, detail, WORD_COLOR[word], checks, ef + cf, held, key)


# ── Copiloten ───────────────────────────────────────────────────────────────
def dom_prompt(dom: Dom, ticker: str, name: str = "") -> str:
    """Underlaget till Copiloten i klartext — domen är redan fälld, modellen förklarar den."""
    lines = [f"Kandidat: {ticker}" + (f" ({name})" if name else ""), f"Strategi: {reg.label(dom.strategy)}",
             f"Arkens dom: {dom.word} — {dom.sentence}", f"Sammanfattning: {dom.detail}", ""]
    for status in (PASS, MANUAL, FAIL):
        rows = dom.by_status(status)
        lines.append({PASS: "Underlag som PASSERADE", MANUAL: "Underlag som kräver MANUELL kontroll",
                      FAIL: "Underlag som FÖLL"}[status] + f" ({len(rows)}):")
        lines += [f"  - {c.label}: {c.note} [{c.source}]" for c in rows] or ["  (inga)"]
    if dom.facts:
        lines += ["", "Nyckeldata ur regimmotorerna:"]
        lines += [f"  - {lbl}: {val} {sub}".rstrip() for lbl, val, _c, sub in dom.facts]
    lines += ["", "Nivåer: inga satta här — inga ordrar läggs från den här vyn.",
              "Kommentera domen enligt instruktionen: förklara den, peka på det svagaste underlaget, "
              "motsäg aldrig domslutet."]
    return "\n".join(lines)


def _copilot(st, dom: Dom, ident) -> None:
    from ui.components import note
    try:
        from ai import copilot_prompt, openai_client
    except Exception as exc:                                     # pragma: no cover - paketet saknas
        note(f"Copiloten är inte tillgänglig: {exc}")
        return
    configured = openai_client.configured()
    store = st.session_state.setdefault(AI_STATE, {})
    key = f"{ident.yf}|{dom.strategy}|{dom.word}|{len(dom.checks)}"
    c1, c2 = st.columns([1, 3])
    asked = c1.button("🤖 Be Copiloten kommentera", key="thor_dom_ask", disabled=not configured)
    if not configured:
        c2.caption(f"Modellen är inte påslagen — {openai_client.KEY_NAME} saknas i secrets. Domen gäller utan den.")
    if asked:
        with st.spinner("Copiloten läser domen …"):
            try:
                reply = openai_client.complete(copilot_prompt.SYSTEM, dom_prompt(dom, ident.yf, ident.name))
                store[key] = {"text": reply.text, "model": getattr(reply, "model", "")}
            except openai_client.AIError as exc:
                st.error(f"Ingen kommentar: {exc}")
    hit = store.get(key)
    if hit:
        model = escape(str(hit.get("model") or ""))
        st.markdown(f"<div style='border-left:3px solid {CYAN};padding:4px 10px;color:{TEXT};font-size:0.82rem;"
                    f"line-height:1.5;'>{escape(str(hit.get('text') or ''))}"
                    f"<div style='color:{DIM};font-size:0.66rem;margin-top:4px;'>{model} · förklarar domen, "
                    f"fäller den inte</div></div>", unsafe_allow_html=True)


# ── Fliken ──────────────────────────────────────────────────────────────────
def _section(st, title: str) -> None:
    st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;"
                f"margin:12px 0 6px;'>{escape(title)}</div>", unsafe_allow_html=True)


def render(st, data: dict, ident, strategy: str, sheets: dict, cands: list, prices: Optional[Callable] = None,
           ctx: Optional[dict] = None) -> None:
    """Arkens dom för aktien: domslutet, nyckeldatan, underlaget rad för rad, arkets ifyllda fält och Copiloten."""
    from ui.components import note
    if ctx is None:
        with st.spinner("Läser arken och regimmotorerna …"):
            ctx = cycle_context(ident, data, sheets, prices)
    dom = judge(strategy, sheets, ctx, ident.yf, held=bool((sheets or {}).get("positions")))
    st.markdown(cards.verdict_card(dom.word, dom.sentence, dom.detail, dom.color,
                                   label="ARKENS DOM — VAD SKA JAG GÖRA?"), unsafe_allow_html=True)
    if dom.facts:
        st.markdown(cards.kpi_row(dom.facts), unsafe_allow_html=True)
    _section(st, "UNDERLAGET")
    st.markdown(cards.check_rows({c.label: (c.status, c.note) for c in dom.checks}), unsafe_allow_html=True)
    rev = ((sheets or {}).get("reviews") or {}).get(dom.review_key) if dom.review_key else None
    if rev and rev.get("row"):
        try:
            import review_link
            details = review_link.detail_lines(dom.review_key, rev.get("row") or {})
        except Exception:
            details = []
        if details:
            with st.expander(f"Det du fyllt i — {rev.get('sheet') or dom.review_key}", expanded=False):
                for line in details:
                    st.caption(line.strip())
    for name, msg in ((ctx or {}).get("errors") or {}).items():
        note(f"{name} kunde inte läsas: {msg}")
    _section(st, "COPILOTEN")
    _copilot(st, dom, ident)
    note("Domen är deterministisk: FAIL i underlaget → NEJ; granskningen PASS och allt grönt → KÖP; PASS med MANUAL "
         "kvar → BEVAKA; inte granskad → AVVAKTA. Varje rad är ett ark eller en motor som redan finns (Rick Rule, "
         "Poängmodellen, Tiggre, Insiderbevakaren, Scorecard, Durrett, Råvarurotation, Valhall, Odins blindspot, "
         "BERSERK). Riktiga order läggs alltid manuellt.")
