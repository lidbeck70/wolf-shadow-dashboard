"""
crosssheet.py — tal ur Durrett-arket (ARKEN → 🧭 Durrett) som förslag i granskningsarken.

Samma tal skrivs annars för hand i flera ark: AISC, gruvlivslängd, NAV och studiestatus. Här översätts Durrett-
bolagets datapunkter till färdiga förslag (värde + text med källa och datum) som Rick Rule, Tiggre och Poängmodellen
visar med Använd (refresh_ui.offer). Att rita förslaget skriver ingenting.

  Bolaget        hittas på tickerns former (thor.ticker.forms: 'EQX.TO' når Durrett-raden 'EQX').
  Cirkeln        confidence/prefill.py kopierar åt andra hållet (Rick Rule/Tiggre → Durrett) med källan
                 "Granskningsarket …". En sådan datapunkt föreslås aldrig tillbaka.
  Enheten        AISC och metallpris föreslås bara när Durrett-bolagets råvara är kortets och enheten stämmer
                 (review/units.py) — AISC i USD/oz på ett kopparkort i USD/lb vänder domen. Kontrollen stänger:
                 cent ('US c/lb'), okänd valuta ('R/oz'), ingen per-enhet ('USD per tonne') eller ett annat
                 tillägg ('USD/t SC6' mot 'USD/t LCE') föreslås aldrig.
  Kostnaden      Måttet följer kortets prisenhet (engines.durrett.config.COST_METRIC_BY_UNIT): bara AISC kallas
                 all-in; olja, koppar m.fl. får måttet och en uppmaning att se efter i källan.
  Livslängden    Gruvlivslängd föreslås inte för olja och gas — Rick Rule dömer dem på R/P (8 år, inte 5).
  NAV            nav_musd vinner; finns bara npv_musd föreslås det märkt "projekt-NPV (100 %), inte bolagets NAV"
                 (ägarandelen är inte avräknad). Studiens datum står alltid med. Metallpriset följer bara med när
                 det hör till samma tal (projekt-NPV:n, eller samma källa och datum som NAV:n). Bara belopp i
                 miljoner USD (musd_fits): rand ('R million'), tusental ('US$ 000') och USD utan skala föreslås
                 aldrig.

Rena funktioner; durrett_company och producers_data läser sessionen som judgment_hints.gather_for.
"""

from __future__ import annotations

import re
from typing import NamedTuple, Optional

SOURCE = "Durrett-arket"
LOOP_SOURCE = "Granskningsarket"        # prefill.py:s källor — talet kom ur granskningsarken
NPV_LABEL = "projekt-NPV (100 %), inte bolagets NAV"
NPV_SOURCE = "projekt-NPV (100 %)"
NO_SOURCE, NO_DATE = "källa saknas", "datum saknas"
FS_DONE = ("dfs", "fid", "construction", "production")      # mognad där FS är klar (confidence.config.MATURITY)
_GENERIC_PER = ("enhet", "enheter", "unit", "units")         # confidence-fältens 'USD/enhet' = råvarans enhet
_CENTS = ("c", "cent", "cents")                              # 'US c/lb', 'cents/lb' — cent, inte dollar
_NEUTRAL_TAIL = ("sold", "produced", "payable", "paid")      # 'US$/oz sold' — samma enhet
COST_NOTE = "all-in, inte C1; biproduktkrediter enligt källan"
COST_CHECK = "Durrett-fältet AISC/opex: se i källan om det är all-in, C1 eller opex"
RP_COMMODITIES = ("oil_gas", "natural_gas")                  # Rick Rule dömer dem på R/P, inte gruvlivslängd


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


def _words(unit) -> tuple:
    """(ord före '/', tillägg efter per-enheten) i en enhetstext: 'US c/lb' → (('us', 'c'), ()), 'USD/t SC6' →
    (('usd',), ('sc6',)). 'dmt' (torrton koncentrat) räknas som tillägg; 'sold'/'payable' räknas inte."""
    from review import units
    head, _sep, tail = units._clean(unit).partition("/")
    words = re.findall(r"[a-z0-9åäöμ]+(?:\.[0-9]+)?", tail)
    extra = tuple(w for w in words[1:] if w not in _NEUTRAL_TAIL)
    if words[:1] == ["dmt"]:
        extra = ("dmt",) + extra
    return tuple(re.findall(r"[a-zåäöμ]+", head)), extra


def unit_fits(given, expected) -> bool:
    """Står datapunktens enhet i kortets prisenhet? Valuta och per-enhet jämförs (review.units.parts); en tom eller
    allmän enhet ('USD/enhet') är råvarans egen. Kontrollen stänger — det den inte kan läsa passar inte: cent
    ('US c/lb', 'US¢/lb'), okänd valuta ('R/oz'), ingen per-enhet ('USD per tonne') och ett annat tillägg efter
    per-enheten ('USD/t SC6' mot 'USD/t LCE'). Okänd förväntad enhet passar aldrig."""
    from review import units
    e_ccy, e_per, _s = units.parts(expected)
    if not e_per:
        return False
    if not str(given or "").strip():
        return True
    g_ccy, g_per, g_scale = units.parts(given)
    g_head, g_tail = _words(given)
    if g_scale or "¢" in str(given) or any(w in _CENTS for w in g_head):
        return False
    if g_ccy and e_ccy and g_ccy != e_ccy:
        return False
    if g_per in _GENERIC_PER:
        return True
    if (e_ccy and not g_ccy) or not g_per or g_per != e_per:
        return False                    # valutan okänd ('c/lb', 'R/oz'), ingen per-enhet, eller en annan
    return not g_tail or g_tail == _words(expected)[1]


def musd_fits(given) -> bool:
    """Står ett belopp i miljoner USD? Tom enhet = fältets MUSD. Kontrollen stänger som unit_fits: USD och miljoner
    ska stå utskrivna och varje ord vara just det. Okänd valuta ('R million', 'R$ M', 'miljoner kronor'), tusental
    ('US$ 000', 'USD thousands', 'TUSD', 'USD k') och USD utan skala föreslås aldrig."""
    from review import units
    s = str(given or "").strip()
    if not s:
        return True
    g_ccy, g_per, g_scale = units.parts(s)
    if g_ccy != "usd" or g_scale != "M" or g_per or "000" in s:
        return False
    head, _tail = _words(s)
    return all((c or sc) and c in ("", "usd") and sc in ("", "M")
               for c, _p, sc in map(units.parts, head))       # varje ord är USD eller miljoner


def _price_copy(company, key: str, card_commodity) -> Optional[Copy]:
    """AISC eller metallpris: bara när råvaran är kortets och enheten är kortets prisenhet."""
    if not same_commodity(card_commodity, company):
        return None
    c = _positive(copy(company, key))
    if c is None or not unit_fits(c.unit, price_unit(card_commodity)):
        return None
    return c


def _scale_max() -> float:
    try:
        from controls_ui import NAV_PRICE_SCALE_MAX
        return float(NAV_PRICE_SCALE_MAX)
    except Exception:
        return 10.0


def _far(value, card_price, below: bool = True) -> bool:
    """Står talet mer än NAV_PRICE_SCALE_MAX× från kortets sparade råvarupris (som spot_margin_line)? Då står det nog
    i en annan enhet. below=False: bara ett tal över priset räknas (opex/boe kan ligga långt under oljepriset)."""
    v, p = _num(value), _num(card_price)
    if v is None or p is None or v <= 0 or p <= 0:
        return False
    lim = _scale_max()
    return v / p > lim or (below and p / v > lim)


def _unit_text(c: Copy, expected: str) -> str:
    """Enheten i förslagets text: datapunktens egen när den står utskriven och skiljer sig från kortets ('US$/oz'),
    annars kortets (en tom eller 'USD/enhet' är kortets)."""
    from review import units
    if c.unit and units.parts(c.unit)[1] not in _GENERIC_PER and c.unit != expected:
        return c.unit
    return expected


def cost_metric(card_commodity) -> str:
    """Kostnadsmåttet i kortets prisenhet (engines.durrett.config.COST_METRIC_BY_UNIT): 'AISC' för USD/oz, 'opex'
    för USD/boe, 'C1/AISC' för USD/lb. 'AISC/opex' när enheten är okänd."""
    try:
        from engines.durrett.config import COST_METRIC_BY_UNIT
        return COST_METRIC_BY_UNIT.get(price_unit(card_commodity), "AISC/opex")
    except Exception:
        return "AISC/opex"


def _cost_text(metric: str, c: Copy, unit: str) -> str:
    """'AISC 1 450 USD/oz (all-in, inte C1; …)' — bara AISC kallas all-in; andra mått får en uppmaning att se efter."""
    return f"{metric} {_g(c.value)} {unit} ({COST_NOTE if metric == 'AISC' else COST_CHECK})"


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
def cost_offer(company, card_commodity, card_ticker="", card_price=None) -> Optional[tuple]:
    """(kostnad, text) för Rick Rules 'Kostnad per enhet' — bara samma råvara och samma enhet, och aldrig en kostnad
    mer än NAV_PRICE_SCALE_MAX× över kortets sparade pris (cent som dollar). Måttet följer prisenheten (cost_metric):
    bara AISC kallas all-in."""
    c = _price_copy(company, "aisc", card_commodity)
    if c is None or _far(c.value, card_price, below=False):
        return None
    unit = _unit_text(c, price_unit(card_commodity))
    return c.value, f"{_head(company, card_ticker)}: {_cost_text(cost_metric(card_commodity), c, unit)} — {when(c)}"


def _rp_commodity(value) -> bool:
    try:
        from confidence import commodities as com
        return com.resolve_key(value) in RP_COMMODITIES
    except Exception:
        return False


def life_offer(company, card_ticker="", card_commodity=None) -> Optional[tuple]:
    """(år, text) för 'Gruvlivslängd (år)' ur Durrett-arkets mine_life_years. Inte för olja och gas (kortets eller
    Durrett-radens råvara): där är talet reservlivslängd, och Rick Rule dömer dem på R/P med en annan gräns."""
    if _rp_commodity(card_commodity) or _rp_commodity(getattr(company, "commodity", "")):
        return None
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


def nav_offer(company, card_commodity, card_ticker="", card_price=None, card_nav_price=None) -> Optional[NavPair]:
    """NAV-paret för Rick Rule: NAV (nav_musd, annars projekt-NPV) + metallpriset i NPV. Priset följer bara med när
    det hör till talet — projekt-NPV:n, eller samma källa och datum som NAV:n — och när råvaran och enheten är
    kortets och priset står inom NAV_PRICE_SCALE_MAX× från kortets råvarupris. Använd skriver också NAV-källan, och
    utan pris i paret töms kortets gamla Metallpris i NAV (card_nav_price) — texten säger det."""
    hit = nav_copy(company)
    if hit is None:
        return None
    c, npv = hit
    price = _price_copy(company, "npv_price_assumption", card_commodity)
    if price is not None and not npv and (price.source, price.pub_date) != (c.source, c.pub_date):
        price = None                    # studiens pris hör inte till ett NAV ur en annan källa
    if price is not None and _far(price.value, card_price):
        price = None
    what = f"{NPV_LABEL}: {_g(c.value)} MUSD — {_ownership_note(company)}" if npv else f"NAV {_g(c.value)} MUSD"
    if price is not None:
        at = f" vid metallpris {_g(price.value)} {_unit_text(price, price_unit(card_commodity))}"
        fills = "metallpris och NAV-källa."
    else:
        at = ""
        old = _num(card_nav_price)
        fills = ("NAV-källa och tömmer Metallpris i NAV — Durrett-arket har inget pris till den här NAV:n."
                 if old not in (None, 0.0) else "NAV-källa.")
    text = f"{_head(company, card_ticker)}: {what}{at} — {when(c)}. Använd fyller också {fills}"
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
    return c.value, f"{_head(company, card_ticker)}: {_cost_text('AISC', c, CALC_UNIT)} — {when(c)}"
