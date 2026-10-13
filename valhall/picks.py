"""
valhall/picks.py — kvalitets- och hävstångsköpen i Valhalls rutor (långsiktigt — bulten i systemet).

Per SEKTORRUTA
  2 kvalitetsköp   Quality Contrarian i Norden (contrarian_alpha_results_quality / _deep_contrarian),
                   rangordnade på Alpha Regimes köpsignal (BUY → WATCH → WAIT) och sedan totalpoäng.
                   Utanför Norden: segmentets UCITS-ETF (signal från SPDR-fonden).
Per RÅVARURUTA
  2 kvalitetsköp   Rick Rule-producenter: ditt producentark (producer_score/verdict) först, sedan
                   Rule-håven (screens.json) och Quality Contrarians nordiska råvarubolag. Utanför
                   Norden bara för guld, koppar och uran (undantag 1: extremt bra producenter);
                   övriga råvaror: nordiska bolag + temats ETF.
  5 hävstångsköp   guld och silver → Durrett: Durrett-arket (motorn) och Poängmodellen. 15-BAGGER =
                   uppsida minst 15× i motorns bull-fall OCH Poängmodellen minst 8 (undantag 1).
                   Övriga råvaror → Tiggre: tiggararket (köpgrindarna, faktorpoäng, U/N). Platina- och
                   palladiumproducenter samt uranutvecklare/-prospektörer körs med Tiggre (undantag 2);
                   temats ETF visas bredvid.

Allt är rena funktioner över de sparade källorna (Gisten och arken på panel-data) — ingenting hämtas
här. Arken saknar ibland råvara (tiggararket har inget råvarufält): då slås råvaran upp i Durrett-arket,
producentarket och Poängmodellen; hittas den inte hamnar bolaget under "råvara okänd".
"""

from __future__ import annotations

from typing import Optional

from berserk import themes as bth
from berserk import universe as buv
from valhall import segments as sgm

N_QUALITY, N_LEVERAGE = 2, 5
DURRETT_THEMES = ("guld", "silver")
GLOBAL_PRODUCER_THEMES = ("guld", "koppar", "uran")       # undantag 1: extremt bra producenter utanför Norden
TIGGRE_EXCEPTION_THEMES = ("platina", "palladium", "uran")  # undantag 2: PGM-producenter, uranutvecklare
BAGGER_MULTIPLE, BAGGER_SCORE = 15.0, 8

# confidence.commodities-nyckel → BERSERK-tema
_COMMODITY_THEME = {"gold": "guld", "silver": "silver", "copper": "koppar", "uranium": "uran", "platinum": "platina",
                    "palladium": "palladium", "lithium": "litium", "nickel": "zink_nickel", "zinc": "zink_nickel",
                    "iron_ore": "jarnmalm", "oil_gas": "olja", "oil": "olja", "natural_gas": "naturgas",
                    "coal": "kol", "rare_earth": "sallsynta", "aluminum": "aluminium", "steel": "stal"}

# Quality Contrarians sektornamn (engelska/svenska) → segment
_SECTOR_NAME = {"energy": "energy", "energi": "energy", "materials": "materials", "material": "materials",
                "industrials": "industrials", "industri": "industrials", "financials": "financials",
                "finans": "financials", "finans & fastighet": "financials", "real estate": "real_estate",
                "fastighet": "real_estate", "information technology": "technology", "technology": "technology",
                "informationsteknik": "technology", "health care": "healthcare", "healthcare": "healthcare",
                "hälsovård": "healthcare", "communication services": "communication",
                "telecommunication services": "communication", "telekommunikation": "communication",
                "consumer discretionary": "consumer_disc", "sällanköpsvaror": "consumer_disc",
                "consumer staples": "consumer_staples", "dagligvaror": "consumer_staples",
                "utilities": "utilities", "kraftförsörjning": "utilities"}
_REAL_ESTATE_BRANCH = ("fastighet",)                         # bransch "Fastighetsbolag", "Fastighet - REIT"
_VERDICT_ORDER = {"BUY": 0, "WATCH": 1, "WAIT": 2}


def theme_of_commodity(name) -> Optional[str]:
    """Råvarunamn (svenska/engelska) → BERSERK-tema."""
    if not name:
        return None
    try:
        from confidence.commodities import resolve_key
        key = resolve_key(name)
    except Exception:
        key = None
    if key in _COMMODITY_THEME:
        return _COMMODITY_THEME[key]
    s = str(name).strip().lower()
    return next((t for t in bth.THEMES if t == s or bth.label(t).lower() == s), None)


def segment_of_ca(row: dict) -> Optional[str]:
    """Quality Contrarian-raden → segment (sektornamn; fastighetsbranscher → Fastighet)."""
    branch = str(row.get("branch") or "").lower()
    if any(b in branch for b in _REAL_ESTATE_BRANCH):
        return "real_estate"
    return _SECTOR_NAME.get(str(row.get("sector") or "").strip().lower())


def _ca_rows(ca_blobs: list) -> list:
    """Quality Contrarian-rader ur en eller flera sparade körningar (bästa raden per ticker)."""
    best = {}
    for blob in ca_blobs or []:
        if not blob:
            continue
        signals = blob.get("quality_signals") or {}
        for r in blob.get("results") or []:
            t = r.get("ticker")
            if not t:
                continue
            sig = signals.get(t) or {}
            row = {"ticker": t, "name": r.get("name"), "score": r.get("composite_score"), "rank": r.get("rank"),
                   "verdict": sig.get("verdict"), "mode": blob.get("mode"), "sector": r.get("sector"),
                   "branch": r.get("branch"), "roic": r.get("roic"), "hat": r.get("hat_score"),
                   "segment": segment_of_ca(r), "theme": buv.theme_of(t) or None}
            prev = best.get(t)
            if prev is None or _ca_key(row) < _ca_key(prev):
                best[t] = row
    return list(best.values())


_CLASS_TOKENS = {"a", "b", "c", "d", "r", "ser", "ser.", "pref", "a-aktie", "b-aktie"}


def company_key(r: dict) -> str:
    """Samma bolag i flera aktieslag (Orion A/B, Volvo A/B) → en nyckel: namnet utan aktieslag."""
    words = str(r.get("name") or r.get("ticker") or "").lower().replace(",", " ").split()
    while len(words) > 1 and words[-1] in _CLASS_TOKENS:
        words.pop()
    return " ".join(words) or str(r.get("ticker") or "")


def _one_per_company(rows: list) -> list:
    seen, out = set(), []
    for r in rows:
        k = company_key(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _ca_key(r: dict) -> tuple:
    return (_VERDICT_ORDER.get(r.get("verdict"), 3), -(r.get("score") or 0))


def segment_quality(ca_blobs: list, n: int = N_QUALITY) -> dict:
    """{segment: {"picks": [...], "etf": {spdr, ucits}}} — två bästa nordiska kvalitetsköp per segment."""
    rows = _ca_rows(ca_blobs)
    out = {}
    for s in sgm.SEGMENTS:
        mine = _one_per_company(sorted((r for r in rows if r["segment"] == s.key), key=_ca_key))
        out[s.key] = {"picks": [dict(r, source="Quality Contrarian") for r in mine[:n]],
                      "etf": {"signal": s.spdr, "ucits": list(s.ucits)}}
    return out


def _producer_rows(sheets: dict) -> list:
    """Producentarket (Rick Rule) med poäng och utlåtande."""
    import producers as pr
    out = []
    for r in ((sheets or {}).get("producers") or {}).get("producers") or []:
        theme = theme_of_commodity(r.get("commodity"))
        score = pr.producer_score(r)
        verdict = pr.producer_verdict(score, pr.asset_dying(r))
        out.append({"ticker": r.get("ticker"), "name": r.get("name"), "theme": theme, "score": score,
                    "verdict": getattr(verdict, "label", verdict), "source": "Rick Rule-arket",
                    "nordic": buv.region_of(r.get("ticker") or "") == "Norden"})
    return out


def _rule_screen_rows(screens: dict) -> list:
    """Rule-håven (screens.json) — bara råvarubranscher som går att knyta till ett tema."""
    branch_theme = {7: "uran", 6: "kol", 18: "guld"}           # 1–5 olja & gas, 16–17 metaller (för grova)
    for b in (1, 2, 3, 4, 5):
        branch_theme[b] = "olja"
    out = []
    for r in (((screens or {}).get("screens") or {}).get("rule") or {}).get("rows") or []:
        theme = buv.theme_of(r.get("ticker")) or branch_theme.get(r.get("branch_id"))
        m = r.get("m") or {}
        out.append({"ticker": r.get("ticker"), "name": r.get("name"), "theme": theme, "source": "Rule-håven",
                    "ev_ebitda": m.get("ev_ebitda"), "first_seen": r.get("first_seen"),
                    "nordic": buv.region_of(r.get("ticker") or "") == "Norden"})
    return out


def theme_quality(sheets: dict, screens: dict, ca_blobs: list, n: int = N_QUALITY) -> dict:
    """{tema: {"picks": [...], "etf": [...]}} — två bästa Rick Rule-producenter per råvara."""
    sheet = _producer_rows(sheets)
    screen = _rule_screen_rows(screens)
    ca = [dict(r, source="Quality Contrarian", nordic=True) for r in _ca_rows(ca_blobs) if r.get("theme")]
    out = {}
    for theme in bth.THEMES:
        allow_global = theme in GLOBAL_PRODUCER_THEMES

        def ok(r):
            return r.get("theme") == theme and (r.get("nordic") or allow_global)

        ranked = sorted((r for r in sheet if ok(r)), key=lambda r: (r.get("verdict") != "Köpkandidat",
                                                                     -(r.get("score") or 0)))
        ranked += sorted((r for r in screen if ok(r)), key=lambda r: (r.get("ev_ebitda") is None,
                                                                       r.get("ev_ebitda") or 0))
        ranked += sorted((r for r in ca if ok(r)), key=_ca_key)
        picks = _one_per_company([r for r in ranked if r.get("ticker")])
        out[theme] = {"picks": picks[:n], "etf": [e for e, t in bth.ETFS.items() if t == theme],
                      "global_allowed": allow_global}
    return out


def _commodity_lookup(sheets: dict) -> dict:
    """{TICKER: tema} ur Durrett-arket, producentarket och Poängmodellen (tiggararket saknar råvara)."""
    out = {}
    for t, c in (((sheets or {}).get("confidence") or {}).get("companies") or {}).items():
        th = theme_of_commodity((c or {}).get("commodity"))
        if th:
            out[str(t).upper()] = th
    for b in ("producers",):
        for r in ((sheets or {}).get("producers") or {}).get(b) or []:
            th = theme_of_commodity(r.get("commodity"))
            if th and r.get("ticker"):
                out.setdefault(str(r["ticker"]).upper(), th)
    for b in ("durrett", "sprott"):
        for r in ((sheets or {}).get("scoring") or {}).get(b) or []:
            th = theme_of_commodity(r.get("commodity"))
            if th and r.get("ticker"):
                out.setdefault(str(r["ticker"]).upper(), th)
    return out


def durrett_rows(sheets: dict, analyze=None) -> list:
    """Durrett-arkets guld- och silverbolag genom motorn + Poängmodellen. 15-BAGGER = bull-fallets
    uppsida minst 15× och Poängmodellen minst 8."""
    import scoring as sc
    if analyze is None:
        from engines.durrett.engine import analyze
    from confidence.data.models import CompanyInput
    poang = {str(r.get("ticker") or "").upper(): sc.total_score(r.get("factors") or {})
             for r in ((sheets or {}).get("scoring") or {}).get("durrett") or []}
    out = []
    for t, c in (((sheets or {}).get("confidence") or {}).get("companies") or {}).items():
        theme = theme_of_commodity((c or {}).get("commodity"))
        if theme not in DURRETT_THEMES:
            continue
        try:
            a = analyze(CompanyInput.from_dict({k: v for k, v in c.items() if k != "id"}))
        except Exception:
            continue
        bull = getattr(getattr(a, "bull_case", None), "upside_multiple", None)
        score = poang.get(str(t).upper())
        q = getattr(a, "quality_score", None)
        quality = getattr(q, "value", q)
        out.append({"ticker": str(t).upper(), "name": c.get("name"), "theme": theme, "source": "Durrett",
                    "bull_multiple": None if bull is None else round(float(bull), 1), "poang": score,
                    "quality": None if quality is None else round(float(quality), 0),
                    "stage": c.get("stage"),
                    "bagger": bool(bull is not None and bull >= BAGGER_MULTIPLE and score is not None
                                   and score >= BAGGER_SCORE)})
    return out


def tiggre_rows(sheets: dict) -> list:
    """Tiggararkets kandidater: köpgrindarna, faktorpoäng och U/N; råvara ur de andra arken."""
    import tiggre as tg
    lookup = _commodity_lookup(sheets)
    out = []
    for c in ((sheets or {}).get("tiggre") or {}).get("candidates") or []:
        t = str(c.get("ticker") or "").upper()
        gates = tg.buy_gates(c)
        up = tg.upside_pct(tg._num(c.get("mcap")), tg._num(c.get("nav")))
        un = tg.un_ratio(up, tg._num(c.get("downside")))
        out.append({"ticker": t, "name": c.get("name"), "theme": lookup.get(t) or buv.theme_of(t) or None,
                    "source": "Tiggre", "score": tg.factor_score(c.get("factors") or {}, un),
                    "un": None if un is None else round(un, 1), "buy": all(p for _l, p, _d in gates),
                    "gates_passed": sum(1 for _l, p, _d in gates if p), "gates_total": len(gates)})
    return out


def theme_leverage(sheets: dict, screens: dict, analyze=None, n: int = N_LEVERAGE) -> dict:
    """{tema: {"strategy": Durrett|Tiggre, "picks": [...], "to_analyse": [...], "etf": [...]}};
    plus "_okand" för tiggarkandidater utan känd råvara."""
    dur = durrett_rows(sheets, analyze)
    tig = tiggre_rows(sheets)
    scr = ((screens or {}).get("screens") or {})
    out = {}
    for theme in bth.THEMES:
        if theme in DURRETT_THEMES:
            mine = sorted((r for r in dur if r["theme"] == theme),
                          key=lambda r: (not r["bagger"], -(r.get("bull_multiple") or 0)))
            known = {r["ticker"] for r in mine}
            todo = [{"ticker": r.get("ticker"), "name": r.get("name"), "mcap_musd": r.get("mcap_musd")}
                    for r in (scr.get("durrett") or {}).get("rows") or [] if r.get("ticker") not in known]
            strategy = "Durrett"
        else:
            mine = sorted((r for r in tig if r["theme"] == theme),
                          key=lambda r: (not r["buy"], -(r.get("score") or 0), -(r.get("un") or 0)))
            todo = []
            if theme == "uran":
                known = {r["ticker"] for r in mine}
                todo = [{"ticker": r.get("ticker"), "name": r.get("name"), "mcap_musd": r.get("mcap_musd")}
                        for r in (scr.get("tiggre") or {}).get("rows") or []
                        if r.get("branch_id") == 7 and r.get("ticker") not in known]
            strategy = "Tiggre"
        out[theme] = {"strategy": strategy, "picks": mine[:n], "to_analyse": todo[:n],
                      "etf": [e for e, t in bth.ETFS.items() if t == theme],
                      "exception": theme in TIGGRE_EXCEPTION_THEMES}
    out["_okand"] = {"strategy": "Tiggre", "picks": [r for r in tig if not r["theme"]][:n], "to_analyse": [],
                     "etf": [], "exception": False}
    return out


def build(ca_blobs: list, screens: Optional[dict], sheets: Optional[dict], analyze=None) -> dict:
    """Alla köp per ruta: {"segments": segment_quality, "themes": {tema: {quality, leverage}}}."""
    tq = theme_quality(sheets, screens, ca_blobs)
    tl = theme_leverage(sheets, screens, analyze)
    return {"segments": segment_quality(ca_blobs),
            "themes": {t: {"quality": tq[t], "leverage": tl[t]} for t in bth.THEMES},
            "unknown_commodity": tl["_okand"]["picks"]}
