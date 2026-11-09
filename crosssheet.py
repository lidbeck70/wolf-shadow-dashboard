"""
crosssheet.py — tal ur Durrett-arket (ARKEN → 🧭 Durrett) som förslag i granskningsarken.

Samma tal skrivs annars för hand i flera ark: AISC, gruvlivslängd, NAV och studiestatus. Här översätts Durrett-
bolagets datapunkter till färdiga förslag (värde + text med källa och datum) som Rick Rule, Tiggre och Poängmodellen
visar med Använd (refresh_ui.offer). Att rita förslaget skriver ingenting.

  Bolaget        hittas på tickerns former (thor.ticker.forms: 'EQX.TO' når Durrett-raden 'EQX').
  Cirkeln        confidence/prefill.py kopierar åt andra hållet (Rick Rule/Tiggre → Durrett) med källan
                 "Granskningsarket …". En sådan datapunkt föreslås aldrig tillbaka.
  Enheten        AISC och metallpris föreslås bara när Durrett-bolagets råvara är kortets och enheten stämmer
                 (review/units.py) — AISC i USD/oz på ett kopparkort i USD/lb vänder domen.
  NAV            nav_musd vinner; finns bara npv_musd föreslås det märkt "projekt-NPV (100 %), inte bolagets NAV"
                 (ägarandelen är inte avräknad). Studiens datum står alltid med.

Rena funktioner; durrett_company och producers_data läser sessionen som judgment_hints.gather_for.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

SOURCE = "Durrett-arket"
LOOP_SOURCE = "Granskningsarket"        # prefill.py:s källor — talet kom ur granskningsarken
NPV_LABEL = "projekt-NPV (100 %), inte bolagets NAV"
NPV_SOURCE = "projekt-NPV (100 %)"
NO_SOURCE, NO_DATE = "källa saknas", "datum saknas"
FS_DONE = ("dfs", "fid", "construction", "production")      # mognad där FS är klar (confidence.config.MATURITY)
_GENERIC_PER = ("enhet", "enheter", "unit", "units")         # confidence-fältens 'USD/enhet' = råvarans enhet
COST_NOTE = "all-in, inte C1; biproduktkrediter enligt källan"


class Copy(NamedTuple):
    """En datapunkt ur Durrett-arket: värde, enhet, källa och publiceringsdatum (tomt när det saknas)."""
    value: object
    unit: str
    source: str
    pub_date: str


def _num(v) -> Optional[float]:
    if v is None or v == "" or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _g(v: float) -> str:
    """Talet som arken skriver det: 1 450 · 12 · 3,25."""
    if abs(v) >= 100:
        return f"{v:,.0f}".replace(",", " ")
    return f"{v:.2f}".rstrip("0").rstrip(".").replace(".", ",")


def _key(ticker) -> str:
    return str(ticker or "").strip().upper()


# ── Bolaget ──────────────────────────────────────────────────────────────────
def company_for(ticker, confidence_data: Optional[dict]):
    """Durrett-bolaget (CompanyInput) för tickern i någon av dess former, exakt form först. None utan träff."""
    if not _key(ticker) or not confidence_data:
        return None
    try:
        from confidence import store as cs
        from thor.ticker import forms
        for f in forms(ticker):
            c = cs.get(confidence_data, f)
            if c is not None:
                return c
    except Exception:
        return None
    return None


def durrett_company(ticker):
    """Som company_for, men läser Durrett-lagret ur sessionen. None vid fel — ett förslag får aldrig fälla ett ark."""
    try:
        import storage
        from confidence import store as cs
        return company_for(ticker, storage.session_load(cs.STORE, cs.default()))
    except Exception:
        return None


def producers_data() -> dict:
    """Rick Rule-lagret ur sessionen, laddat som Rick Rule-sidan själv laddar det (producers._load, med den gamla
    Gist-filen). Tomt vid fel."""
    try:
        import producers
        d = producers._load()
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def rule_row_for(ticker, data: Optional[dict]) -> Optional[dict]:
    """Rick Rule-raden för tickern i någon av dess former, exakt form först."""
    rows = [r for r in ((data or {}).get("producers") or []) if isinstance(r, dict)]
    if not _key(ticker) or not rows:
        return None
    try:
        from thor.ticker import forms
        fs = [_key(f) for f in forms(ticker)]
    except Exception:
        fs = [_key(ticker)]
    for f in fs:
        hit = next((r for r in rows if _key(r.get("ticker")) == f), None)
        if hit is not None:
            return hit
    return None


def copy(company, key: str) -> Optional[Copy]:
    """Datapunkten som (värde, enhet, källa, datum), eller None när den saknas eller kom ur granskningsarken."""
    if company is None:
        return None
    try:
        p = company.get(key)
    except Exception:
        return None
    if p is None or p.missing:
        return None
    src = str(p.source or "").strip()
    if src.startswith(LOOP_SOURCE):
        return None
    return Copy(p.value, str(p.unit or "").strip(), src, str(p.pub_date or p.data_date or "")[:10])


def _positive(c: Optional[Copy]) -> Optional[Copy]:
    v = _num(c.value) if c is not None else None
    return c._replace(value=v) if v is not None and v > 0 else None


# ── Enheten ──────────────────────────────────────────────────────────────────
def price_unit(commodity) -> str:
    """Råvarans prisenhet ('Koppar' → 'USD/lb'), tom när råvaran är okänd."""
    try:
        from review import units
        return units.commodity_unit({"commodity": commodity})
    except Exception:
        return ""


def unit_label(commodity) -> str:
    """Den fasta etiketten vid kostnads- och prisrutorna: 'USD/lb · Koppar'. Tom utan känd råvara."""
    u = price_unit(commodity)
    return f"{u} · {str(commodity).strip()}" if u else ""


def same_commodity(card_commodity, company) -> bool:
    """Är Durrett-bolagets råvara kortets? Okänd råvara på någon sida är aldrig samma."""
    try:
        from confidence import commodities as com
        a, b = com.resolve_key(card_commodity), com.resolve_key(getattr(company, "commodity", ""))
    except Exception:
        return False
    return a is not None and a == b


def unit_fits(given, expected) -> bool:
    """Står datapunktens enhet i kortets prisenhet? Valuta och per-enhet jämförs (review.units.parts); en tom eller
    allmän enhet ('USD/enhet') är råvarans egen. Okänd förväntad enhet passar aldrig."""
    from review import units
    e_ccy, e_per, _s = units.parts(expected)
    if not e_per:
        return False
    g_ccy, g_per, g_scale = units.parts(given)
    if g_scale:
        return False
    if g_ccy and e_ccy and g_ccy != e_ccy:
        return False
    return not (g_per and g_per not in _GENERIC_PER and g_per != e_per)


def musd_fits(given) -> bool:
    """Står ett belopp i miljoner USD (tom enhet = fältets MUSD)?"""
    from review import units
    g_ccy, g_per, g_scale = units.parts(given)
    return (not g_ccy or g_ccy == "usd") and (not g_scale or g_scale == "M") and not g_per


def _price_copy(company, key: str, card_commodity) -> Optional[Copy]:
    """AISC eller metallpris: bara när råvaran är kortets och enheten är kortets prisenhet."""
    if not same_commodity(card_commodity, company):
        return None
    c = _positive(copy(company, key))
    if c is None or not unit_fits(c.unit, price_unit(card_commodity)):
        return None
    return c


# ── Texterna ─────────────────────────────────────────────────────────────────
def _head(company, card_ticker) -> str:
    """'🤖 Durrett-arket' — med Durrett-radens ticker när den är en annan form än kortets ('(EQX)')."""
    t = _key(getattr(company, "ticker", ""))
    return f"🤖 {SOURCE}" + (f" ({t})" if t and t != _key(card_ticker) else "")


def when(c: Copy) -> str:
    """'FS 2025, 2025-03-01' — källa och datum, alltid båda (en gammal studie ska synas)."""
    return f"{c.source or NO_SOURCE}, {c.pub_date or NO_DATE}"


def nav_source_text(c: Copy, npv: bool = False) -> str:
    """NAV-källan Använd skriver på Rick Rule-kortet: 'Durrett-arket: <källa>, <datum>'."""
    return f"{SOURCE}: " + (f"{NPV_SOURCE}, " if npv else "") + when(c)


def _ownership_note(company) -> str:
    own = _positive(copy(company, "ownership_pct"))
    pct = f" (ägarandel {_g(own.value)} %)" if own is not None else ""
    return f"ägarandelen är inte avräknad{pct}"


# ── Rick Rule ────────────────────────────────────────────────────────────────
def cost_offer(company, card_commodity, card_ticker="") -> Optional[tuple]:
    """(AISC, text) för Rick Rules 'Kostnad per enhet' — bara samma råvara och samma enhet."""
    c = _price_copy(company, "aisc", card_commodity)
    if c is None:
        return None
    return c.value, (f"{_head(company, card_ticker)}: AISC {_g(c.value)} {price_unit(card_commodity)} "
                     f"({COST_NOTE}) — {when(c)}")


def life_offer(company, card_ticker="") -> Optional[tuple]:
    """(år, text) för 'Gruvlivslängd (år)' ur Durrett-arkets mine_life_years."""
    c = _positive(copy(company, "mine_life_years"))
    if c is None:
        return None
    return c.value, f"{_head(company, card_ticker)}: gruvlivslängd {_g(c.value)} år — {when(c)}"


def nav_copy(company) -> Optional[tuple]:
    """(Copy, npv) — nav_musd först, annars npv_musd (npv=True). Bara belopp i MUSD."""
    for key, npv in (("nav_musd", False), ("npv_musd", True)):
        c = _positive(copy(company, key))
        if c is not None and musd_fits(c.unit):
            return c, npv
    return None


class NavPair(NamedTuple):
    nav: float
    price: Optional[float]          # metallpriset i NAV (npv_price_assumption), None när det inte följer med
    source: str                     # det Använd skriver i NAV-källa
    text: str


def nav_offer(company, card_commodity, card_ticker="") -> Optional[NavPair]:
    """NAV-paret för Rick Rule: NAV (nav_musd, annars projekt-NPV) + metallpriset i NPV — priset bara när råvaran
    och enheten är kortets. Använd skriver också NAV-källan."""
    hit = nav_copy(company)
    if hit is None:
        return None
    c, npv = hit
    price = _price_copy(company, "npv_price_assumption", card_commodity)
    what = f"{NPV_LABEL}: {_g(c.value)} MUSD — {_ownership_note(company)}" if npv else f"NAV {_g(c.value)} MUSD"
    at = f" vid metallpris {_g(price.value)} {price_unit(card_commodity)}" if price is not None else ""
    fills = "metallpris och NAV-källa" if price is not None else "NAV-källa"
    text = f"{_head(company, card_ticker)}: {what}{at} — {when(c)}. Använd fyller också {fills}."
    return NavPair(c.value, price.value if price is not None else None, nav_source_text(c, npv), text)


# ── Tiggre ───────────────────────────────────────────────────────────────────
def tiggre_nav_offer(company, rule_row: Optional[dict], card_ticker="") -> Optional[tuple]:
    """(NAV, text) för Tiggres 'NAV = NPV after tax': Durrett nav_musd, annars npv_musd (märkt projekt-NPV),
    annars Rick Rule-kortets NAV efter skatt med dess NAV-källa."""
    hit = nav_copy(company)
    if hit is not None:
        c, npv = hit
        what = f"{NPV_LABEL}: {_g(c.value)} MUSD — {_ownership_note(company)}" if npv else f"NAV {_g(c.value)} MUSD"
        return c.value, f"{_head(company, card_ticker)}: {what} — {when(c)}"
    nav = _num((rule_row or {}).get("nav_musd"))
    if nav is None or nav <= 0:
        return None
    src = str((rule_row or {}).get("nav_source") or "").strip()
    t = _key((rule_row or {}).get("ticker"))
    who = "🤖 Rick Rule-kortet" + (f" ({t})" if t and t != _key(card_ticker) else "")
    return nav, f"{who}: NAV efter skatt {_g(nav)} MUSD — NAV-källa: {src or 'saknas på kortet'}"


def tiggre_nav_for(ticker, company) -> Optional[tuple]:
    """Som tiggre_nav_offer, men Rick Rule-raden läses ur sessionen — och bara när Durrett-arket inte har något NAV."""
    hit = tiggre_nav_offer(company, None, ticker)
    if hit is not None:
        return hit
    return tiggre_nav_offer(None, rule_row_for(ticker, producers_data()), ticker)


SCREEN_FIELDS = (("permits", "permits_granted", "nyckeltillstånd beviljade"),
                 ("funded", "financing_committed", "finansiering åtagen"))


def screen_offers(company, card_ticker="") -> list:
    """[(nyckel, text)] för grovsållningen — bara som True: FS klar när mognaden är DFS/BFS eller senare, tillstånd och
    finansiering när Durrett-arket säger ja. Mognaden har standardvärdet 'pea', så ett nej föreslås aldrig."""
    if company is None:
        return []
    out = []
    head = _head(company, card_ticker)
    mat = str(getattr(company, "maturity", "") or "")
    if mat in FS_DONE:
        try:
            from confidence.config import MATURITY_LABEL
            label = MATURITY_LABEL.get(mat, mat)
        except Exception:
            label = mat
        out.append(("fs", f"{head}: mognad {label} → FS klar"))
    from confidence.data.provenance import Datapoint, truth
    for key, field, label in SCREEN_FIELDS:
        c = copy(company, field)
        if c is not None and truth(Datapoint(c.value)) is True:
            out.append((key, f"{head}: {label} — {when(c)}"))
    return out


def csm_base_nav_text(nav: float) -> str:
    """Texten vid CSM:s Base-NAV ur kortets eget NAV."""
    return f"🤖 Kortet: NAV (Base) {_g(nav)} MUSD — kortets NAV efter skatt, samma definition"


# ── Poängmodellen (Durrett-kortet) ───────────────────────────────────────────
GOLD = "gold"
CALC_UNIT = "USD/oz"


def scoring_aisc_offer(company, card_commodity, card_ticker="") -> Optional[tuple]:
    """(AISC, text) för hjälpräknarens 'AISC ($/oz)' — bara guld i USD/oz."""
    try:
        from confidence import commodities as com
        if com.resolve_key(card_commodity) != GOLD:
            return None
    except Exception:
        return None
    c = _price_copy(company, "aisc", card_commodity)
    if c is None or not unit_fits(c.unit, CALC_UNIT):
        return None
    return c.value, f"{_head(company, card_ticker)}: AISC {_g(c.value)} {CALC_UNIT} ({COST_NOTE}) — {when(c)}"
