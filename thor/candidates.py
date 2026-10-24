"""
thor/candidates.py — kandidatlistan: alla strategiers kandidater ur de sparade körningarna, i EN form.

Inga nya beräkningar. Varje rad bär strategins eget verdikt (status), dess egen siffra (score_text) och
Thor-rank (score, thor/registry.py) som bara är en gemensam skala för sortering. status_class är listans
grova läge för filter och ordning — KÖP · BEVAKA · TRÄFF · HÅLL · SPÄRRAD · SÄLJ — översatt rakt av från
varje strategis egna ord (GOLDEN TICKET → KÖP, READY → BEVAKA, …), aldrig tolkat.

Källorna (alla ur Gisten, inga live-skanningar):
  valhall_board.json   Viking Nine per segment (swing), Long Screener (long), Rick Rule/Durrett/Tiggre
                       (buys.themes), Arc-Wolfs ENTRY-märken (wolf), och segmentens status/cykel (board)
  berserk_scan.json    BERSERK:s KÖP/BEVAKA (alla regioner)
  arc_screeners.json   Arc-Wolf (regimpoäng) och Viking (V9)
  wolf_screener.json   Momentum Swing (topp 40; Börsdatas Stockholmsform "ERIC B" → "ERIC-B.ST")
  ember_results.json   EMBER:s setups (egen Gist, EMBER_GIST_ID)
  contrarian_alpha_results_{quality,deep_contrarian}.json   Quality Contrarian (med Alpha Regimes verdikt)
                       och Deep Contrarian
  insider_scan.json    Insider Norden (kluster med poäng ≥ 5)
  screens.json         håvarna (Rule, Sprott, Durrett, Tiggre, Royalty) — träffar, "NY" sorteras först

Allt är rena funktioner över dicts. load_blobs är det enda som rör nätet, och den är injicerbar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional

import pandas as pd

from thor import registry as reg

KOP, BEVAKA, TRAFF, HALL, SPARRAD, SALJ = "KÖP", "BEVAKA", "TRÄFF", "HÅLL", "SPÄRRAD", "SÄLJ"
CLASS_ORDER = {KOP: 0, BEVAKA: 1, TRAFF: 2, HALL: 3, SPARRAD: 4, SALJ: 5}
NORDIC_SUFFIX = (".ST", ".OL", ".CO", ".HE", ".IC")
STALE_DAYS = 4                                 # en körning äldre än så visas, men märks gammal
UNREADABLE = "_unreadable"          # sätts av load_blobs när ingen enda blob gick att läsa (nät/token), inte "ingen körning"
BLOBS = {"board": "valhall_board.json", "berserk": "berserk_scan.json", "arc": "arc_screeners.json",
         "momentum": "wolf_screener.json", "insider": "insider_scan.json", "screens": "screens.json",
         "royalty": "royalty_scan.json", "viking_nine": "viking_nine_scan.json"}
# Håvar vars träffar INTE dubbleras under en egen strategi: Royalty-motorn väljer själv ur håven (producenterna
# som passerar marginalfiltret hör till Rick Rule) och ger varje bolag verdikt och rang i from_royalty.
SCREEN_OWN_SKIP = frozenset({"royalty"})
WOLF_BANDS = ((85, "BULL"), (65, "MODERATE"), (45, "NEUTRAL"))      # Wolf Regime-sidans etiketter


@dataclass
class Candidate:
    ticker: str                    # Yahoo-form
    name: str
    strategy: str                  # thor.registry-nyckel
    status: str                    # strategins eget verdikt, ordagrant
    status_class: str              # KÖP · BEVAKA · TRÄFF · HÅLL · SPÄRRAD · SÄLJ
    score: Optional[float] = None  # Thor-rank 0–100 (None = ingen rang)
    score_text: str = ""           # strategins egen siffra
    segment: Optional[str] = None
    segment_status: Optional[str] = None   # GRÖN/GUL/RÖD på Valhall-kartan (Norden, eller temat)
    cykel: Optional[str] = None            # TIDIG/MITTEN/SEN/TOPP
    theme: Optional[str] = None            # råvarutema (BERSERK-vokabulär)
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    rr: Optional[float] = None
    nordic: bool = False
    new: bool = False
    asof: str = ""                 # körningens datum
    stale: bool = False            # körningen äldre än STALE_DAYS
    why: str = ""
    data_asof: str = ""            # stängningen motorn räknade på (Viking Nine) — tomt när källan inte säger

    @property
    def family(self) -> str:
        return reg.STRATEGIES[self.strategy].family

    @property
    def label(self) -> str:
        return reg.STRATEGIES[self.strategy].label

    def as_dict(self) -> dict:
        return {**asdict(self), "family": self.family, "label": self.label}


# ── Hjälpare ────────────────────────────────────────────────────────────────
def yf(ticker) -> str:
    """Yahoo-form av en screener-ticker: 'ekta b.st' → 'EKTA-B.ST'. Utan suffix lämnas tickern (NEM)."""
    return str(ticker or "").strip().upper().replace(" ", "-")


def momentum_yf(ticker, mapping: Optional[dict] = None, yf_hint: Optional[str] = None) -> tuple:
    """Momentumlistan skriver Börsdatas form ('ERIC B', 'NHY'). (Yahoo-ticker, gissad?): radens egen yf-form
    (wolf_data skriver den ur marknads-id) först, sedan registrets karta, sist '.ST' — då är den en gissning."""
    t = str(ticker or "").strip().upper()
    if yf_hint and "." in str(yf_hint):
        return str(yf_hint).strip().upper(), False
    if mapping and t in mapping:
        return mapping[t], False
    if "." in t:
        return t, False
    return t.replace(" ", "-") + ".ST", True


def is_nordic(ticker) -> bool:
    return str(ticker or "").upper().endswith(NORDIC_SUFFIX)


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def _day(ts) -> str:
    return str(ts or "")[:10]


def _norm_name(r: dict, fallback: str = "") -> str:
    return str(r.get("name") or fallback or "")


def _ticker_of(x) -> str:
    """Tickern ur en rad (dict) eller en ren sträng — screens.json skriver dagens nya träffar som strängar."""
    return str(x.get("ticker") or "") if isinstance(x, dict) else str(x or "")


def segment_context(board: Optional[dict]) -> dict:
    """{segments: {key: {status, cykel, phase}}, themes: {tema: {status, cykel, phase, segment}}} ur kartan."""
    out: dict = {"segments": {}, "themes": {}}
    for r in board_segments(board):
        n = r.get("nordic") if isinstance(r.get("nordic"), dict) else {}
        g = r.get("global") if isinstance(r.get("global"), dict) else {}
        out["segments"][r["key"]] = {"status": n.get("status"), "cykel": n.get("cykel"), "phase": n.get("phase"),
                                     "global_status": g.get("status")}
        for t in r.get("themes") if isinstance(r.get("themes"), list) else []:
            if not isinstance(t, dict) or not t.get("theme"):
                continue
            s = t.get("state") if isinstance(t.get("state"), dict) else {}
            out["themes"][t["theme"]] = {"status": s.get("status"), "cykel": s.get("cykel"), "phase": s.get("phase"),
                                         "segment": r["key"]}
    return out


def board_segments(board: Optional[dict]) -> list:
    """Kartans segmentrader som en lista av dict med nyckel — allt annat (trasig blob) ger tom lista, aldrig ett fel."""
    inner = (board or {}).get("board") if isinstance(board, dict) else None
    segs = inner.get("segments") if isinstance(inner, dict) else None
    return [r for r in segs if isinstance(r, dict) and r.get("key")] if isinstance(segs, list) else []


def segment_map_from_register(universe) -> dict:
    """{ticker: Valhall-segment} ur universum-registret (Börsdatas sektor/bransch via valhall.segments, annars
    råvarutemats segment) — så att rader utan eget segment ändå går att filtrera per segment."""
    from thor.ticker import segment_for
    out = {}
    for t, m in (getattr(universe, "members", None) or {}).items():
        try:
            seg = segment_for(getattr(m, "sector_id", None), getattr(m, "branch_id", None), getattr(m, "theme", None))
        except Exception:
            seg = None
        if seg:
            out[str(t).upper()] = seg
    return out


def _tag(c: Candidate, ctx: dict) -> Candidate:
    """Kartans läge på raden: temats om raden har ett tema på kartan, annars segmentets. Saknar raden segment
    (Viking, Wolf utan entry, Insider …) tas det ur registrets sektor (ctx["segment_of"]: ticker → segment)."""
    if not c.segment and not c.theme:
        c.segment = (ctx.get("segment_of") or {}).get(c.ticker) or None
    st = (ctx.get("themes") or {}).get(c.theme) if c.theme else None
    if not st and c.segment:
        st = (ctx.get("segments") or {}).get(c.segment)
    if st:
        c.segment_status, c.cykel = st.get("status"), st.get("cykel")
        if not c.segment:
            c.segment = st.get("segment")
    return c


def _segment_of_theme(theme) -> Optional[str]:
    from valhall import segments as sgm
    return sgm.segment_of_theme(theme) if theme else None


# ── Källorna ────────────────────────────────────────────────────────────────
def from_viking_nine(board: Optional[dict], ctx: dict, scan: Optional[dict] = None) -> list:
    """Viking Nine: kartans swing-val i segment som styrs av Viking Nine (även "prövas") och den schemalagda
    skanningen över hela registret (viking_nine_scan.json, Norden och USA). GOLDEN TICKET → KÖP, annars BEVAKA;
    samma bolag i båda → den bästa raden vinner i dedupe."""
    from valhall import swing as vsw

    def row(p, segment, asof, why):
        cat = str(p.get("category") or p.get("action") or "")
        w = _f(p.get("nine_weighted"))
        t = yf(p.get("ticker"))
        return _tag(Candidate(
            t, _norm_name(p), "viking_nine", cat, KOP if cat == "GOLDEN TICKET" else BEVAKA,
            score=w, score_text=f"{p.get('ovtlyr_nine') or '?'} · {w:.0f}" if w is not None else str(p.get("ovtlyr_nine") or ""),
            segment=segment, entry=_f(p.get("entry")), stop=_f(p.get("stop")), rr=_f(p.get("rr")), nordic=is_nordic(t),
            asof=asof, why=why, data_asof=str(p.get("bar_date") or "")), ctx)
    out, asof = [], _day((board or {}).get("generated"))
    for key, e in (((board or {}).get("swing") or {}).get("segments") or {}).items():
        if e.get("strategy") not in (vsw.VIKING, vsw.TEST):
            continue
        for p in e.get("picks") or []:
            out.append(row(p, key, asof, str(p.get("reason") or e.get("strategy") or "")))
    asof = _day((scan or {}).get("generated"))
    for p in (scan or {}).get("rows") or []:
        if isinstance(p, dict) and p.get("ticker"):
            out.append(row(p, p.get("segment") or None, asof, str(p.get("reason") or "Viking Nine-skanningen")))
    return out


def from_long(board: Optional[dict], ctx: dict) -> list:
    """Long Screener med Valhalls cykel (gron): segmentets STRONG BUY/BUY → KÖP."""
    out, asof = [], _day((board or {}).get("generated"))
    for key, e in (((board or {}).get("long") or {}).get("segments") or {}).items():
        for p in e.get("picks") or []:
            pct = _f(p.get("score_pct"))
            sig = str(p.get("signal") or "")
            out.append(_tag(Candidate(
                yf(p.get("ticker")), _norm_name(p), "long", sig, KOP if sig in ("STRONG BUY", "BUY") else HALL,
                score=None if pct is None else round(pct * 100, 1),
                score_text=(f"{pct * 100:.0f} % · cykel {p.get('cycle')}/3 · fundamenta {p.get('fund')}/{p.get('fund_max')}"
                            if pct is not None else sig),
                segment=key, nordic=True, asof=asof, why="Long Screener (cykel gron)"), ctx))
    return out


def from_themes(board: Optional[dict], ctx: dict) -> list:
    """Råvarurutornas köp: Rick Rule (arket, Rule-håven, Quality Contrarian), Durrett och Tiggre —
    KÖP när Valhalls papperskonto skulle köpa (valhall.paper.buy_ok), annars BEVAKA."""
    from valhall.paper import buy_ok
    out, asof = [], _day((board or {}).get("generated"))
    for theme, e in (((board or {}).get("buys") or {}).get("themes") or {}).items():
        seg = _segment_of_theme(theme)
        for p in ((e.get("quality") or {}).get("picks") or []):
            src = str(p.get("source") or "")
            verdict = str(p.get("verdict") or src)
            sc = _f(p.get("score"))
            if src == "Rick Rule-arket":
                score, text = (None if sc is None else sc * 20), f"{sc:g}/5 · {verdict}" if sc is not None else verdict
            elif src == "Quality Contrarian":
                score, text = (None if sc is None else min(sc, 100.0)), f"composite {sc:.0f} · {verdict}" if sc is not None else verdict
            else:                                                          # Rule-håven: håven är regeln
                ev = _f(p.get("ev_ebitda"))
                score, text = None, f"Rule-håven · EV/EBITDA {ev:g}" if ev is not None else "Rule-håven"
            out.append(_tag(Candidate(
                yf(p.get("ticker")), _norm_name(p), "rule", verdict, KOP if buy_ok(p) else BEVAKA, score=score,
                score_text=text, segment=seg, theme=theme, nordic=bool(p.get("nordic")) or is_nordic(p.get("ticker")),
                asof=asof, why=src), ctx))
        lv = e.get("leverage") or {}
        strat = "durrett" if lv.get("strategy") == "Durrett" else "tiggre"
        for p in lv.get("picks") or []:
            if strat == "durrett":
                q, bull, poang = _f(p.get("quality")), _f(p.get("bull_multiple")), p.get("poang")
                status = "15-BAGGER" if p.get("bagger") else f"bull {bull:g}×" if bull is not None else "Durrett"
                text = " · ".join(x for x in ((f"bull {bull:g}×" if bull is not None else ""),
                                              (f"poäng {poang}/10" if poang is not None else ""),
                                              (f"kvalitet {q:.0f}" if q is not None else "")) if x)
                score = q
            else:
                status, text, score = _tiggre_fields(p)
            out.append(_tag(Candidate(
                yf(p.get("ticker")), _norm_name(p), strat, status, KOP if buy_ok(p) else BEVAKA, score=score,
                score_text=text, segment=seg, theme=theme, nordic=is_nordic(p.get("ticker")), asof=asof,
                why=str(p.get("source") or lv.get("strategy") or "")), ctx))
    for p in ((board or {}).get("buys") or {}).get("unknown_commodity") or []:   # Tiggre-arket utan känd råvara
        status, text, score = _tiggre_fields(p)
        out.append(_tag(Candidate(
            yf(p.get("ticker")), _norm_name(p), "tiggre", f"{status} · okänd råvara", BEVAKA, score=score,
            score_text=text, nordic=is_nordic(p.get("ticker")), asof=asof,
            why="Tiggre — råvaran saknas i arket, så kartan kan inte ge tema och cykel (papperet köper inte)"), ctx))
    return out


def _tiggre_fields(p: dict) -> tuple:
    """(status, siffran, rang) för en Tiggre-rad: grindarna, faktorpoängen 0–10 × 10 och U/N."""
    sc, un = _f(p.get("score")), _f(p.get("un"))
    status = f"grindar {p.get('gates_passed')}/{p.get('gates_total')}"
    text = " · ".join(x for x in ((f"{sc:g}/10" if sc is not None else ""),
                                  (f"U/N {un:g}" if un is not None else "")) if x)
    return status, text, (None if sc is None else sc * 10)


def from_berserk(blob: Optional[dict], ctx: dict) -> list:
    """BERSERK:s KÖP och BEVAKA: rang efter setupens sällsynthet (S2 100 · S1 80 · S3 60), BEVAKA −20."""
    from berserk import signals as bsg
    out, asof = [], _day((blob or {}).get("generated"))
    for r in (blob or {}).get("rows") or []:
        st = str(r.get("status") or "")
        if st not in ("KÖP", "BEVAKA") or r.get("error"):
            continue
        prio = bsg.PRIORITY.get(r.get("setup"), 0)
        score = {3: 100, 2: 80, 1: 60}.get(prio, 50) - (20 if st == "BEVAKA" else 0)
        theme = r.get("theme") or None
        why = r.get("why") or []
        out.append(_tag(Candidate(
            yf(r.get("ticker")), _norm_name(r, (ctx.get("names") or {}).get(yf(r.get("ticker")), "")), "berserk", f"{st} · {r.get('setup') or '—'}",
            KOP if st == "KÖP" else BEVAKA, score=float(score), score_text=f"{r.get('setup') or '—'} · {r.get('complex') or ''}".strip(" ·"),
            segment=_segment_of_theme(theme), theme=theme, entry=_f(r.get("close")), stop=_f(r.get("stop")),
            nordic=(r.get("region") == "Norden") or is_nordic(r.get("ticker")), asof=asof,
            why="; ".join(str(w) for w in (why if isinstance(why, list) else [why])[:2])), ctx))
    return out


def _wolf_band(score: float) -> str:
    return next((lbl for lo, lbl in WOLF_BANDS if score >= lo), "BEAR")


def from_arc(blob: Optional[dict], board: Optional[dict], ctx: dict) -> list:
    """Arc-Wolf (regimpoäng 0–125; ENTRY ur kartans Wolf-rader när köpregeln är uppfylld) och Viking (V9)."""
    out = []
    entries = {}
    for e in (((board or {}).get("wolf") or {}).get("segments") or {}).values():
        for p in e.get("arc") or []:
            entries[yf(p.get("ticker"))] = p
    for part in ("wolf", "wolf_usa"):                        # Norden och registrets USA (arc_scan) — samma form
        w = (blob or {}).get(part) or {}
        asof = _day(w.get("generated"))
        for r in w.get("rows") or []:
            sc = _f(r.get("score"))
            if sc is None:
                continue
            t = yf(r.get("ticker"))
            p = entries.get(t) or {}
            band = _wolf_band(sc)
            entry = bool(p.get("entry"))
            cls = KOP if entry else BEVAKA if band in ("BULL", "MODERATE") else HALL
            out.append(_tag(Candidate(
                t, _norm_name(r), "wolf", f"ENTRY · {band}" if entry else band, cls, score=round(sc / 1.25, 1),
                score_text=f"{sc:.0f}/125" + (f" · grindar {p['gates']}" if p.get("gates") else ""),
                segment=p.get("segment"), entry=_f(p.get("price")) if entry else None,
                stop=_f(p.get("stop")) if entry else None,
                target=_f(p.get("tp1")) if entry else None, nordic=is_nordic(t), asof=asof,
                why="strategins köpregel uppfylld" if entry else "regimpoängen — ingen entry ännu"), ctx))
    for part in ("viking", "viking_usa"):
        v = (blob or {}).get(part) or {}
        asof = _day(v.get("generated"))
        for r in v.get("rows") or []:
            v9 = _f(r.get("v9"))
            if v9 is None:
                continue
            sig = str(r.get("signal") or "")
            comp = _f(r.get("composite"))
            t = yf(r.get("ticker"))
            out.append(_tag(Candidate(
                t, _norm_name(r), "viking", f"{sig} · screener {v9:.0f}/9" if sig else f"screener {v9:.0f}/9",
                KOP if sig == "BUY" else BEVAKA if sig == "WATCH" else HALL, score=round(v9 / 9 * 100, 1),
                score_text=f"screener {v9:.0f}/9" + (f" · composite {comp:.0f}" if comp is not None else ""),
                nordic=is_nordic(t), asof=asof,
                why="Viking-screenern: BUY = över EMA200, ADX ≥ 20 och composite ≥ 60 — screenerns egna nio faktorer, "
                    "inte Viking Nines OVTLYR Nine"), ctx))
    return out


def from_momentum(blob: Optional[dict], ctx: dict, mapping: Optional[dict] = None, regime: Optional[dict] = None) -> list:
    """Momentum Swing: rad 1–20 med setup A → KÖP, rad 1–20 utan → BEVAKA, 21–40 → HÅLL. Stoppen −10 %.
    Regimljuset (wolf_regime.json) är strategins egen regel: RÖD låser köpen (SPÄRRAD), GUL = halv storlek,
    saknat ljus = BEVAKA — samma som swing_verdict och strategy_rules (Momentum Swing)."""
    out, asof = [], _day((blob or {}).get("generated"))
    light = str((regime or {}).get("regime") or "").upper() if isinstance(regime, dict) else ""
    for r in (blob or {}).get("top") or []:
        rank = r.get("rank")
        if rank is None:
            continue
        rank = int(rank)
        setup = bool(r.get("setupA"))
        cls = KOP if rank <= 20 and setup else BEVAKA if rank <= 20 else HALL
        status = f"rad {rank}" + (" · setup A" if setup else "")
        why = "Swing Screener (stopp −10 %, halva vid +20 %)"
        if cls == KOP and light == "RÖD":
            cls, status, why = SPARRAD, status + " · RÖD regim", "Swing Regime RÖD — inga nya swingköp (strategins regel)"
        elif cls == KOP and light == "GUL":
            status, why = status + " · GUL regim", why + " · GUL regim: halv storlek"
        elif cls == KOP and not light:
            cls, why = BEVAKA, why + " · regimljuset saknas (kör wolf_data) — inget köp utan ljus"
        price = _f(r.get("price"))
        t, guessed = momentum_yf(r.get("ticker"), mapping, r.get("yf"))
        if guessed:
            why += " · tickern antagen .ST (registret saknas)"
        out.append(_tag(Candidate(
            t, _norm_name(r), "momentum", status, cls,
            score=max(0.0, 100.0 - (rank - 1) * 2.5), score_text=f"rad {rank}/40", entry=price,
            stop=None if price is None else round(price * 0.9, 2), nordic=True, asof=asof, why=why), ctx))
    return out


def from_ember(blob: Optional[dict], ctx: dict) -> list:
    """EMBER:s setups: KÖPLÄGE → KÖP, BEVAKA → BEVAKA, AVVAKTA → HÅLL (även nästan-träffar)."""
    out, asof = [], _day((blob or {}).get("timestamp"))
    for r in ((blob or {}).get("eligible") or []) + ((blob or {}).get("near_misses") or []):
        if r.get("error"):
            continue
        verdict = str(r.get("verdict") or "AVVAKTA")
        sc, asym = _f(r.get("setup_score")), _f(r.get("asymmetry_score"))
        theme = r.get("theme_key") or None
        t = yf(r.get("ticker"))
        out.append(_tag(Candidate(
            t, _norm_name(r), "ember", verdict, KOP if verdict == "KÖPLÄGE" else BEVAKA if verdict == "BEVAKA" else HALL,
            score=sc, score_text=(f"setup {sc:.0f}" if sc is not None else "") + (f" · asymmetri {asym:.0f}" if asym is not None else ""),
            segment=_segment_of_theme(theme), theme=theme, entry=_f(r.get("entry")), stop=_f(r.get("stop")),
            target=_f(r.get("t1")), rr=_f(r.get("rr")), nordic=is_nordic(t), asof=asof,
            why=f"cykel {r.get('cykel_label')}" if r.get("cykel_label") else ""), ctx))
    return out


def from_contrarian(blob: Optional[dict], ctx: dict, mode: str = "quality") -> list:
    """Quality Contrarian med Alpha Regimes verdikt (BUY → KÖP, WATCH → BEVAKA, WAIT → HÅLL; ej analyserad →
    BEVAKA) och Valhalls grind ej RÖD sektor (BUY i RÖD segment → SPÄRRAD). Deep Contrarian: rang → BEVAKA."""
    from valhall import picks as vp
    out, asof = [], _day((blob or {}).get("timestamp"))
    signals = (blob or {}).get("quality_signals") or {}
    strat = "quality" if mode == "quality" else "deep"
    for r in (blob or {}).get("results") or []:
        raw = str(r.get("ticker") or "")
        if not raw:
            continue
        t = yf(raw)
        sc = _f(r.get("composite_score"))
        seg = vp.segment_of_ca(r)
        theme = None
        try:
            from berserk import universe as buv
            theme = buv.theme_of(t) or None
        except Exception:
            pass
        rank = r.get("rank")
        sig = signals.get(raw) or signals.get(t) or {}
        verdict = str(sig.get("verdict") or "")
        if strat == "quality" and verdict in ("BUY", "WATCH", "WAIT"):
            status = f"{verdict} {sig.get('passed')}/{sig.get('total')}"
            cls = {"BUY": KOP, "WATCH": BEVAKA, "WAIT": HALL}[verdict]
            if cls == KOP and ((ctx.get("segments") or {}).get(seg) or {}).get("status") == "RÖD":
                cls, status = SPARRAD, f"{status} · sektorn RÖD"
        else:
            status, cls = f"rank {rank}" if rank is not None else "listan", BEVAKA
        flags = r.get("all_flags") or []
        out.append(_tag(Candidate(
            t, _norm_name(r), strat, status, cls, score=None if sc is None else min(round(sc, 1), 100.0),
            score_text=(f"composite {sc:.0f}" if sc is not None else "") + (f" · rank {rank}" if rank is not None else ""),
            segment=seg, theme=theme, nordic=is_nordic(t), asof=asof,
            why="; ".join(str(x) for x in flags[:2]) if flags else ("Alpha Regime: " + verdict if verdict else "")), ctx))
    return out


def from_insider(blob: Optional[dict], ctx: dict) -> list:
    """Insider Norden: statusflödet som det är — KÖP → KÖP (utom passa-flaggan), brus/grind → HÅLL, annars BEVAKA."""
    import insider as ins
    out, asof = [], _day((blob or {}).get("generated"))
    for c in (blob or {}).get("clusters") or []:
        sc = _f(c.get("score"))
        st = str(c.get("status") or "")
        chase = bool(c.get("chase"))
        cls = KOP if (st == ins.S_BUY and not chase) else HALL if st in (ins.S_NOISE, ins.S_GATE_FAIL) else BEVAKA
        short = st.split(" — ")[0] + (" · passa (jagar kursen)" if chase else "")
        t = yf(c.get("ticker"))
        out.append(_tag(Candidate(
            t, _norm_name(c), "insider", short, cls, score=None if sc is None else sc * 10,
            score_text=f"{sc:g}/10 · {c.get('insiders')} insiders · {c.get('role') or ''}".strip(" ·") if sc is not None else "",
            entry=_f(c.get("price_now")), stop=_f(c.get("stop")), nordic=True, asof=asof,
            why=f"klustersnitt {c.get('cluster_avg')}" if c.get("cluster_avg") is not None else ""), ctx))
    return out


def from_screens(blob: Optional[dict], ctx: dict) -> list:
    """Håvarna: varje träff är en rad (TRÄFF), nya träffar märks — utan rang (en håv säger ja eller nej).
    Har håven en egen strategi i registret (Rick Rule, Durrett, Tiggre …) får träffen även en rad där, så att
    strategin visar sina träffar; arkets KÖP/BEVAKA för samma bolag vinner i dedupe. Sprott och andra håvar utan
    egen strategi syns bara under Håvarna."""
    out, asof = [], _day((blob or {}).get("generated"))
    for key, sc in ((blob or {}).get("screens") or {}).items():
        sc = sc if isinstance(sc, dict) else {}
        new = {yf(_ticker_of(x)) for x in sc.get("new") or []}
        own = key != "hav" and key not in SCREEN_OWN_SKIP and reg.get(key) is not None
        strategies = ("hav",) + ((key,) if own else ())
        for r in sc.get("rows") or []:
            if not isinstance(r, dict):
                continue
            t = yf(r.get("ticker"))
            mc = _f(r.get("mcap_musd"))
            for strategy in strategies:
                out.append(_tag(Candidate(
                    t, _norm_name(r), strategy, str(sc.get("label") or key), TRAFF, score=None,
                    score_text=" · ".join(x for x in (str(r.get("universe") or ""),
                                                      f"{mc:,.0f} MUSD" if mc is not None else "") if x),
                    nordic=is_nordic(t), new=t in new, asof=asof, why=str(sc.get("sheet") or "")), ctx))
    return out


def from_royalty(blob: Optional[dict], ctx: dict) -> list:
    """Royalty-motorn (royalty_scan.json): BUY CANDIDATE → KÖP, WATCH → BEVAKA, PASS → HÅLL med orsaken i statusen;
    rang = composite 0–100, siffran visar pelarna. Temat ger kartans läge."""
    out, asof = [], _day((blob or {}).get("generated"))
    for r in (blob or {}).get("rows") or []:
        if not isinstance(r, dict):
            continue
        t = yf(r.get("ticker"))
        if not t:
            continue
        v = str(r.get("verdict") or "")
        cls = {"BUY CANDIDATE": KOP, "WATCH": BEVAKA}.get(v, HALL)
        why = [str(x) for x in (r.get("why") or [])]
        comp = _f(r.get("composite"))
        p = r.get("pillars") or {}
        labels = (("hate", "hat"), ("value", "värde"), ("quality", "kvalitet"), ("catalyst", "kat"))
        parts = [f"{lbl} {_f(p.get(k)) or 0:.0f}" for k, lbl in labels]
        theme = r.get("theme") or None
        out.append(_tag(Candidate(
            t, _norm_name(r), "royalty", (f"{v} · {why[0]}" if cls == HALL and why else v) or "—", cls,
            score=None if comp is None else min(round(comp, 1), 100.0),
            score_text=" · ".join(([f"composite {comp:.0f}"] if comp is not None else []) + parts),
            segment=_segment_of_theme(theme), theme=theme, nordic=is_nordic(t), asof=asof,
            why="; ".join(why[:2]) if why else str(r.get("kind") or "royalty")), ctx))
    return out


# ── Listan ──────────────────────────────────────────────────────────────────
def sort_key(c: Candidate) -> tuple:
    """KÖP först, sedan rang (ingen rang sist), nya håvträffar före gamla, strategiordningen, tickern."""
    return (CLASS_ORDER.get(c.status_class, 9), 0 if c.new else 1, -(c.score if c.score is not None else -1.0),
            reg.ORDER.index(c.strategy) if c.strategy in reg.ORDER else 99, c.ticker)


def _freshest_viking_nine(rows: list) -> list:
    """Viking Nine-rader kommer ur två källor (kartans swing-val och skanningen); per ticker vinner den som
    räknats på den senaste stängningen — en äldre GOLDEN TICKET får inte dölja dagens READY eller avsaknad."""
    best: dict = {}
    for c in rows:
        if c.strategy != "viking_nine":
            continue
        k = c.ticker
        stamp = (c.data_asof or c.asof or "", c.asof or "")
        if k not in best or stamp > (best[k].data_asof or best[k].asof or "", best[k].asof or ""):
            best[k] = c
    return [c for c in rows if c.strategy != "viking_nine" or best.get(c.ticker) is c]


def dedupe(rows: list) -> list:
    """En rad per strategi och ticker — den bästa (efter sort_key) vinner; Viking Nine: den färskaste."""
    seen, out = set(), []
    for c in sorted(_freshest_viking_nine(rows), key=sort_key):
        k = (c.strategy, c.ticker)
        if c.ticker and k not in seen:
            seen.add(k)
            out.append(c)
    return out


def mark_stale(rows: list, today=None, days: int = STALE_DAYS) -> list:
    t0 = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    for c in rows:
        try:
            c.stale = bool(c.asof) and (t0 - pd.Timestamp(c.asof)).days > days
        except (TypeError, ValueError):
            c.stale = False
    return rows


def _collect(errors: Optional[dict], name: str, fn, *args) -> list:
    """En källa i taget: en fil med oväntad form ger en tom lista och ett fel i errors — aldrig en krasch."""
    try:
        return list(fn(*args) or [])
    except Exception as exc:
        if errors is not None:
            errors[name] = f"{type(exc).__name__}: {exc}"
        return []


def build(blobs: dict, today=None, momentum_map: Optional[dict] = None, errors: Optional[dict] = None,
          segment_map: Optional[dict] = None, names: Optional[dict] = None) -> list:
    """Hela listan ur de lästa filerna (load_blobs-nycklar), sorterad och utan dubbletter. Källor som inte går
    att läsa hoppas över och skrivs i errors ({källa: fel}) så att sidan kan säga vad som saknas.
    segment_map = {ticker: segment} ur registret (segment_map_from_register) för rader utan eget segment."""
    b = blobs or {}
    board = b.get("board") or {}
    try:
        ctx = segment_context(board)
    except Exception as exc:
        if errors is not None:
            errors["board"] = f"{type(exc).__name__}: {exc}"
        ctx = {"segments": {}, "themes": {}}
    ctx["segment_of"] = {str(k).upper(): v for k, v in (segment_map or {}).items() if v}
    ctx["names"] = {str(k).upper(): v for k, v in (names or {}).items() if v}
    if b.get(UNREADABLE) and errors is not None:
        errors["gist"] = "Gisten gick inte att läsa (nätverk eller token) — inga körningar kunde hämtas; tryck 📡 Uppdatera"
    sources = (("viking_nine", from_viking_nine, (board, ctx, b.get("viking_nine"))), ("long", from_long, (board, ctx)),
               ("themes", from_themes, (board, ctx)), ("berserk", from_berserk, (b.get("berserk"), ctx)),
               ("arc", from_arc, (b.get("arc"), board, ctx)),
               ("momentum", from_momentum, (b.get("momentum"), ctx, momentum_map, b.get("regime"))),
               ("ember", from_ember, (b.get("ember"), ctx)), ("quality", from_contrarian, (b.get("ca_quality"), ctx, "quality")),
               ("deep", from_contrarian, (b.get("ca_deep"), ctx, "deep")), ("insider", from_insider, (b.get("insider"), ctx)),
               ("screens", from_screens, (b.get("screens"), ctx)), ("royalty", from_royalty, (b.get("royalty"), ctx)))
    rows: list = []
    for name, fn, args in sources:
        rows += _collect(errors, name, fn, *args)
    return mark_stale(dedupe(rows), today)


def filter_rows(rows: list, family: Optional[str] = None, strategy: Optional[str] = None, only_buy: bool = False,
                nordic_only: bool = False, segment: Optional[str] = None, search: str = "") -> list:
    q = str(search or "").strip().upper()
    out = []
    for c in rows:
        if family and c.family != family:
            continue
        if strategy and c.strategy != strategy:
            continue
        if only_buy and c.status_class != KOP:
            continue
        if nordic_only and not c.nordic:
            continue
        if segment and c.segment != segment:
            continue
        if q and q not in c.ticker.upper() and q not in c.name.upper():
            continue
        out.append(c)
    return out


def summary(rows: list) -> dict:
    """{n, buy, by_family: {familj: n}, by_strategy: {nyckel: {n, buy, asof, stale}}}."""
    fam: dict = {}
    strat: dict = {}
    for c in rows:
        fam[c.family] = fam.get(c.family, 0) + 1
        s = strat.setdefault(c.strategy, {"n": 0, "buy": 0, "asof": "", "stale": False})
        s["n"] += 1
        s["buy"] += int(c.status_class == KOP)
        s["asof"] = max(s["asof"], c.asof or "")
        s["stale"] = s["stale"] or c.stale
    return {"n": len(rows), "buy": sum(1 for c in rows if c.status_class == KOP), "by_family": fam, "by_strategy": strat}


def load_blobs(loader: Optional[Callable] = None, ca_loader: Optional[Callable] = None,
               ember_loader: Optional[Callable] = None) -> dict:
    """Alla källor ur Gisten (None för det som inte går att läsa). loader(namn, fallback) = gist_storage.load_blob;
    ca_loader(mode) = contrarian_alpha.cache.load_screener_results; ember_loader() = ember.cache.load_ember_results."""
    if loader is None:
        from gist_storage import load_blob as loader
    out: dict = {}
    for key, name in BLOBS.items():
        try:
            out[key] = loader(name, None)
        except Exception:
            out[key] = None
    out[UNREADABLE] = all(out.get(k) is None for k in BLOBS)
    for key, mode in (("ca_quality", "quality"), ("ca_deep", "deep_contrarian")):
        try:
            if ca_loader is None:
                from contrarian_alpha.cache import load_screener_results as ca_loader
            out[key] = ca_loader(mode)
        except Exception:
            out[key] = None
    try:
        if ember_loader is None:
            from ember.cache import load_ember_results as ember_loader
        out["ember"] = ember_loader()
    except Exception:
        out["ember"] = None
    return out


def vn_market_notes(blob) -> list:
    """Viking Nine: en rad per marknad där marknadslagret (Trend, Signal, Breadth för OMXS30/SPY) stoppar allt —
    högst 1 av 3 godkända betyder att ingen aktie där kan nå 7/9 (DEVELOPING) utan att resten är perfekt.
    Läses ur viking_nine_scan.json (market_layer); tom lista när fältet saknas. Diagnostik, ingen regel."""
    out = []
    for key, m in ((blob or {}).get("markets") or {}).items():
        layer = (m or {}).get("market_layer") or []
        if not layer:
            continue
        ok = sum(1 for f in layer if f.get("status") == "PASS")
        if ok <= 1:
            bad = ", ".join(f"{f.get('label')} {f.get('status')}" for f in layer if f.get("status") != "PASS")
            name = {"norden": "OMXS30", "usa": "SPY"}.get(key, key)
            out.append(f"Viking Nine {key}: marknadslagret {name} {ok}/3 ({bad}) — inga bolag där når DEVELOPING "
                       f"förrän marknaden vänder. Det är marknadsläget, inte ett fel.")
    return out

