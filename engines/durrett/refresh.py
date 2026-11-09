"""
engines/durrett/refresh.py — sifferuppdateringens förslag för Durrett-arket.

sheets_refresh.py (GitHub Actions) läser data/confidence.json, hämtar
färska tal ur Börsdata och skriver dem till Gisten (sheets_refresh.json)
under nyckeln "confidence:<TICKER>". Här översätts en sådan rad till
förslag {fältnyckel: Datapoint} — bara sådant som skiljer sig från det
bolaget redan har, bara fält som gäller bolagets stage, och inga negativa
tal där fältet inte får vara negativt. Dessutom förslag som härleds ur
arket självt: stage (motorns klassificering), produktionsenhet (råvarans
prisenhet) och högsta resurskategori (arkets reserver och resurser).
Inget skrivs utan Använd. Rena funktioner.
"""

from __future__ import annotations

from typing import Optional

from confidence.config import FIELD_BY_KEY, STAGES
from confidence.data import validation as cval
from confidence.data.models import CompanyInput
from confidence.data.provenance import Datapoint, dp

SHEET = "confidence"
_FX_FIELDS = frozenset({"price", "mcap_musd", "ev_musd", "fx_to_usd"})   # tal som jobbet räknar med kursvalutan
# Valutorna fältet market_currency kan ta (samma lista som valen i arket) — DKK/CHF/PLN ingår, annars lästes en
# dansk kurs som USD bredvid ett börsvärde i USD.
_CCY_CHOICES = tuple(FIELD_BY_KEY["market_currency"].choices)
# Källor vars MUSD-tal redan är omräknade till USD: sifferuppdateringen och håven ("Börsdata …") och
# Asymmetrys Yahoo-reserv (asymmetry/fetch.yahoo_row: "Yahoo Finance (…)"). Allt annat kan stå i lokal valuta.
AUTO_USD_SOURCES = ("Börsdata", "Yahoo Finance")
# (fält i bloben, fält i bolaget, enhet, kind)
_MAP = (("mcap_musd", "market_cap_musd", "MUSD", "ACTUAL"), ("price", "share_price", "valuta/aktie", "ACTUAL"),
        ("ev_musd", "enterprise_value_musd", "MUSD", "ACTUAL"), ("ev_ebitda", "ev_ebitda", "×", "ACTUAL"),
        ("nd_ebitda", "net_debt_ebitda", "×", "ACTUAL"), ("pe", "pe", "×", "ACTUAL"),
        ("revenue_musd", "revenue_musd", "MUSD", "ACTUAL"), ("fcf_musd", "free_cash_flow_musd", "MUSD", "ACTUAL"),
        ("ocf_musd", "operating_cash_flow_musd", "MUSD", "ACTUAL"), ("rs_rank", "rs_rank", "", "ACTUAL"),
        ("ebitda_margin", "ebitda_margin_pct", "%", "ACTUAL"), ("fx_to_usd", "fx_to_usd", "USD per enhet", "ASSUMPTION"),
        # rapportfälten (PR 15): kassa, skuld, aktiehistorik, ROIC, FCF-yield, EV/EBIT
        ("cash_musd", "cash_musd", "MUSD", "ACTUAL"), ("debt_musd", "debt_musd", "MUSD", "ESTIMATE"),
        ("shares_now_m", "basic_shares_m", "M", "ACTUAL"), ("shares_1y_ago_m", "shares_1y_ago_m", "M", "ACTUAL"),
        ("shares_3y_ago_m", "shares_3y_ago_m", "M", "ACTUAL"), ("shares_5y_ago_m", "shares_5y_ago_m", "M", "ACTUAL"),
        ("roic_pct", "roic_pct", "%", "ACTUAL"), ("fcf_yield_pct", "fcf_yield_pct", "%", "ACTUAL"),
        ("ev_ebit", "ev_ebit", "×", "ACTUAL"))
_NOTES = {"debt_musd": "bruttoskuld ≈ nettoskuld + kassa ur senaste rapporten",
          "shares_now_m": "numberOfShares ur senaste årsrapporten (miljoner)"}
_PCT_FIELDS = {"ebitda_margin": 100.0}          # snapshoten ger decimaltal → %


def refresh_row(blob: Optional[dict], ticker: str, sheet: str = SHEET) -> Optional[dict]:
    """Raden ur bloben: 'confidence:<TICKER>' (Durrett-arket) eller 'asymmetry:<TICKER>'."""
    return ((blob or {}).get("rows") or {}).get(f"{sheet}:{(ticker or '').strip().upper()}")


def ins_id_offer(row: Optional[dict], company: CompanyInput) -> Optional[int]:
    """Nattradens Börsdata-id när bolaget saknar ett: jobbet matchade tickern (sheets_refresh.resolve), och sparat
    på bolaget hämtas samma bolag varje natt. None när bolaget redan har ett id eller raden saknar ett."""
    if company.ins_id is not None or not isinstance(row, dict) or isinstance(row.get("ins_id"), bool):
        return None
    try:
        iid = int(row.get("ins_id"))
    except (TypeError, ValueError):
        return None
    return iid if iid > 0 else None


def proposals(blob: Optional[dict], company: CompanyInput, sheet: str = SHEET) -> list:
    """[(fältnyckel, Datapoint, nuvarande värde | None)] — bara avvikande tal. Fält vars stages inte omfattar bolagets
    stage föreslås inte (valideringen ignorerar dem ändå), och inte heller negativa tal för fält utanför
    validation._MAY_BE_NEGATIVE — samma golv 0 som refresh_ui.suggest(lo=0): en explorers EV/EBITDA −3,2 blev förr ett
    valideringsfel efter Använd. Negativt FCF, OCF och rörelsekapital föreslås (de får vara negativa)."""
    s = refresh_row(blob, company.ticker, sheet)
    if not s:
        return []
    asof = str(s.get("asof") or (blob or {}).get("generated") or "")[:10] or None
    src = s.get("source_label") or f"Börsdata (sifferuppdatering {asof or '?'})"
    out = []
    ccy = str(s.get("currency") or "").upper()
    # En kursvaluta utanför jobbets växelkurstabell (London i pence, 'GBX'): jobbet lämnar börsvärde, EV och
    # växelkurs tomma, men en äldre blob räknade om med 1,0 (miljoner GBX som USD, ~79× för högt) — och kursen står i
    # en valuta arket inte kan välja. De talen föreslås inte alls.
    try:
        import sheets_refresh as _sr
        unknown_ccy = bool(ccy) and ccy not in _sr.FX_TO_USD
    except Exception:
        unknown_ccy = False
    for bkey, fkey, unit, kind in _MAP:
        if unknown_ccy and bkey in _FX_FIELDS:
            continue
        stages = FIELD_BY_KEY[fkey].stages
        if stages and company.stage not in stages:
            continue                                    # t.ex. EV/EBITDA och P/E för en explorer
        v = s.get(bkey)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if bkey in _PCT_FIELDS:
            v = v * _PCT_FIELDS[bkey]
        if v < 0 and fkey not in cval._MAY_BE_NEGATIVE:
            continue                                    # rutan och valideringen tar ≥ 0 (negativt EBITDA m.m.)
        # det som sparas: kursen med 6 värdesiffror (en pencekurs i pund, 0,00675, tål inte 4 decimaler), annars 4
        # decimaler — och jämförelsen görs mot samma tal, så ett använt förslag inte dyker upp igen
        v = float(f"{v:.6g}") if bkey == "price" else round(v, 4)
        cur = company.num(fkey) if company.has(fkey) else None
        if cur is not None and abs(cur - v) < 1e-6:
            continue
        note = _NOTES.get(bkey, "")
        source = src
        if bkey == "fx_to_usd":
            source = str(s.get("fx_table") or "sheets_refresh FX-tabell")
            note = f"{ccy} → USD ur jobbets fasta tabell — kontrollera mot dagskurs"
        elif bkey == "price":
            unit = f"{ccy}/aktie" if ccy else unit
        out.append((fkey, dp(v, kind=kind, source=source, source_type="secondary", pub_date=asof,
                             unit=unit, note=note), cur))
    burn = quarterly_burn_offer(s, company, src, asof)
    if burn:
        out.append(burn)
    if ccy and (company.get("market_currency") is None or str(company.get("market_currency").value) != ccy) \
            and ccy in _CCY_CHOICES:
        out.append(("market_currency", dp(ccy, kind="ACTUAL", source=src, source_type="secondary", pub_date=asof),
                    str(company.get("market_currency").value) if company.get("market_currency") else None))
    offer = commodity_price_offer(s, company)
    if offer:
        out.append(offer)
    return out


def commodity_price_offer(s: Optional[dict], company: CompanyInput) -> Optional[tuple]:
    """('commodity_price', Datapoint, nuvarande | None) ur nattens terminspris (sheets_refresh: commodity_price på
    raden), eller None. Bara när priset hämtades för bolagets råvara (commodity_prices.night_fits — råvaran kan ha
    bytts sedan natten) och står i registrets enhet (commodity_prices.unit_fits: guld/silver/platina/palladium USD/oz,
    koppar USD/lb, gas USD/MMBtu, järnmalm USD/t; olja i USD/fat mot USD/boe, ett fat = en boe enligt ägarens beslut).
    Råvarupriset står alltid i USD, så kursvalutans regel (okänd valuta → inga omräknade tal) gäller inte det. Uran,
    litium, zink, nickel och kol har ingen termin och får inget. Skiljer sig inte från det sparade → None."""
    if not isinstance(s, dict):
        return None
    try:
        import commodity_prices as cp
        from confidence import commodities as com
        reg = com.get(company.commodity)
        name = cp.registry_name(company.commodity)
        if reg is None or not name or not cp.night_fits(s, name) or not cp.unit_fits(s.get("commodity_unit"), reg.unit):
            return None
        v = round(float(s.get("commodity_price")), 4)
    except Exception:                                   # inget pris, eller en rad som inte går att läsa: inget förslag
        return None
    if not v > 0:
        return None
    cur = company.num("commodity_price") if company.has("commodity_price") else None
    if cur is not None and abs(cur - v) < 1e-6:
        return None
    tick = str(s.get("commodity_ticker") or (cp.ticker_for(name) or ("",))[0])
    night_unit = str(s.get("commodity_unit") or "").strip()
    note = "; ".join(x for x in (
        cp.BASIS.get(tick, ""),
        f"{tick} står i {night_unit} — ett fat räknas som en boe" if night_unit != reg.unit else "",
        "terminspris, inte spot eller realiserat pris — kan vara en dag gammalt") if x)
    asof = str(s.get("commodity_asof") or "")[:10] or None
    return ("commodity_price", dp(v, kind="ASSUMPTION", source=f"Yahoo {tick} (termin)", source_type="secondary",
                                  pub_date=asof, unit=reg.unit, note=note), cur)


def quarterly_burn_offer(s: Optional[dict], company: CompanyInput, source: str, asof: Optional[str]) -> Optional[tuple]:
    """('quarterly_burn_musd', Datapoint, nuvarande | None): nattens burn/år ÷ 4. Burn är definierad som i dag —
    negativt fritt kassaflöde 12 mån (sheets_refresh.report_fields), alltså med capex. Bara för explorer och developer
    (fältets stages) och bara när burn > 0 (positivt FCF ger burn 0, inget förslag). Samma tal som det sparade → None."""
    spec = FIELD_BY_KEY["quarterly_burn_musd"]
    if not isinstance(s, dict) or company.stage not in spec.stages:
        return None
    try:
        burn = float(s.get("burn_musd"))
    except (TypeError, ValueError):
        return None
    if not burn > 0:
        return None
    v = round(burn / 4.0, 4)
    cur = company.num("quarterly_burn_musd") if company.has("quarterly_burn_musd") else None
    if cur is not None and abs(cur - v) < 1e-6:
        return None
    return ("quarterly_burn_musd", dp(v, kind="ESTIMATE", source=source, source_type="secondary", pub_date=asof,
                                      unit="MUSD", note=f"burn/år {burn:g} MUSD ÷ 4 — negativt FCF 12 mån, capex ingår"),
            cur)


# ── förslag som härleds ur arket självt (inte ur natten) ─────────────────────
_CONF_SV = {"high": "hög", "medium": "medel"}


def stage_offer(company: CompanyInput) -> Optional[tuple]:
    """(stage, säkerhet, skäl) ur Durrett-motorns klassificering (classifier.classify) när den är säker (hög eller
    medel) och kommer fram till ett annat stage än fältet — t.ex. ett skal från Holdings eller Snabbkoll (developer)
    med en omsättning på minst 5 MUSD, som motorn redan läser som producer. Skälen är motorns egna. None annars."""
    try:
        from engines.durrett import classifier
        from engines.durrett._base import Ctx
        cl = classifier.classify(Ctx(company))
    except Exception:                                   # förslaget är en hjälp — motorn får aldrig fälla sidan
        return None
    stage = cl.profile
    if cl.confidence not in _CONF_SV or stage not in STAGES or stage == company.stage:
        return None
    return stage, cl.confidence, list(cl.reasons)


def stage_offer_text(offer: tuple) -> str:
    """'Förslag: stage producer (Durrett-motorn, säkerhet hög): omsättning 600 MUSD (≥ 5); …'."""
    stage, conf, reasons = offer
    return f"Förslag: stage {stage} (Durrett-motorn, säkerhet {_CONF_SV.get(conf, conf)}): " + "; ".join(reasons)


def _text(company: CompanyInput, key: str) -> str:
    return str(company.get(key).value).strip() if company.has(key) else ""


def _price_base(company: CompanyInput) -> tuple:
    """(registerpost, basenhet) för bolagets råvara: prisenheten USD/oz → oz, USD/lb U3O8 → lb, USD/boe → boe —
    samma tabell som Durrett-motorn (_base._PRICE_UNIT_BASE). (None, None) utan råvara eller utan bas (USc/bu)."""
    from confidence import commodities as com
    from engines.durrett import _base as db
    reg = com.get(company.commodity) if company.commodity else None
    base = db._PRICE_UNIT_BASE.get(reg.unit) if reg else None
    return (reg, base) if base else (None, None)


def _unit_base(unit: str) -> Optional[str]:
    """Basenheten för en mängdenhet ur motorns tabeller (koz → oz, Mlb → lb, kt → t; boe, MMBtu, kg och ct är sina
    egna). None för en egen text motorn inte känner ('oz (attributable)')."""
    from engines.durrett import _base as db
    if unit in db._UNIT_FACTOR:
        return db._UNIT_FACTOR[unit][0]
    return unit if unit in set(db._PRICE_UNIT_BASE.values()) else None


def _base_mismatch(company: CompanyInput, reg, base: str) -> Optional[str]:
    """Produktionsenheten har en annan bas än råvarans prisenhet (t i stället för lb för koppar) — samma kontroll som
    Durrett-motorns Ctx.quantity (enhetskonflikt: ingen mängd). Talet står i den enheten, så det måste räknas om — en
    ny etikett räcker inte (50 000 t är inte 50 000 lb). None när enheten passar, saknas eller är egen text."""
    unit = _text(company, "production_unit")
    ub = _unit_base(unit) if unit else None
    if ub is None or ub == base:
        return None
    return (f"Produktionsenheten {unit} passar inte råvarans pris i {reg.unit} — produktion × pris blir fel. "
            f"Skriv produktionen i {base} (räkna om talet) och enheten {base}.")


def quantity_unit_mismatch(company: CompanyInput) -> Optional[tuple]:
    """(fält, enhet, prisenhet, passande val) när enheten Durrett-motorn läser mängderna i har en annan bas än råvarans
    prisenhet (koppar i kt mot USD/lb) — samma ordning som Ctx.quantity: Enhet för resurser/produktion först, annars
    produktionsenheten när motorns tabell känner den. Passande val = motorns enheter med prisets bas (lb/Mlb), tom
    lista när ingen finns (boe, kg …). Motorn räknar då inga mängder (enhetskonflikt). None när enheten passar, saknas
    eller är okänd, eller råvaran saknar prisenhet."""
    from engines.durrett import _base as db
    key = "resource_unit"
    unit = _text(company, key)
    if not unit:
        key, unit = "production_unit", _text(company, "production_unit")
    reg, base = _price_base(company)
    if not base or unit not in db._UNIT_FACTOR or db._UNIT_FACTOR[unit][0] == base:
        return None
    return key, unit, reg.unit, [k for k, (b, _f) in db._UNIT_FACTOR.items() if b == base]


def _resource_base_mismatch(company: CompanyInput) -> Optional[str]:
    """Enhet för resurser/produktion har en annan bas än råvarans prisenhet (kt mot USD/lb): motorn räknar inga mängder,
    och produktionen som står i den enheten blir fel tal i produktion × pris. None annars (produktionsenheten är
    _base_mismatch sak)."""
    m = quantity_unit_mismatch(company)
    if not m or m[0] != "resource_unit":
        return None
    _key, unit, price_unit, fits = m
    fix = f"Välj {'/'.join(fits)} och räkna om talen." if fits else f"Inget av arkets val passar {price_unit}."
    return (f"{FIELD_BY_KEY['resource_unit'].label} {unit} passar inte råvarans pris i {price_unit} — produktion × "
            f"pris blir fel. {fix}")


def _known_unit(company: CompanyInput, key: str, base: str) -> Optional[str]:
    """Fältets enhet när motorns tabell (_base._UNIT_FACTOR) känner den och basen passar priset (koz mot USD/oz).
    None för tomt, egen text eller annan bas (den sista är _base_mismatch sak)."""
    from engines.durrett import _base as db
    unit = _text(company, key)
    return unit if unit in db._UNIT_FACTOR and db._UNIT_FACTOR[unit][0] == base else None


def _scaled(unit: Optional[str]) -> float:
    """Faktorn till basenheten (koz → 1 000, Mlb → 1 000 000); 1 för basenheten själv och för None."""
    from engines.durrett import _base as db
    return db._UNIT_FACTOR[unit][1] if unit in db._UNIT_FACTOR else 1.0


def _scale_conflict(company: CompanyInput, base: str) -> Optional[str]:
    """Asymmetry (asymmetry/model.py) och scenarierna (confidence/scenarios/engine.py) räknar produktion × pris utan
    omräkning, så produktionen måste stå i prisets basenhet. Durrett-motorn läser produktionen i Enhet för
    resurser/produktion (Ctx.quantity) och räknar om — här blir koz, Moz, Mlb, kt och Mt i stället fel tal.
    Bara det tal Asymmetry faktiskt läser räknas: årsproduktionen, annars Produktion nu. Arkets resursenhet gäller det
    talet bara när det är Produktion nu (Durretts fält, i resursenheten) — resurser i Moz utan produktion, eller bara
    Produktion plan i koz, varnas inte. Två enheter som säger olika (oz mot koz) varnas bara när båda motorerna läser
    samma tal: Produktion nu, eller årsproduktionen när Durrett saknar Produktion plan och Produktion nu och faller
    tillbaka på den. None när talet står i basenheten eller ingen produktion räknas."""
    pu = _known_unit(company, "production_unit", base)
    ru = _known_unit(company, "resource_unit", base)
    ru_label = FIELD_BY_KEY["resource_unit"].label
    key = next((k for k in ("annual_production", "production_current") if cval.usable(company, k)), None)
    shared = key == "production_current" or (key == "annual_production" and not any(
        cval.usable(company, k) for k in ("production_future", "production_current")))
    if shared and pu and ru and pu != ru:
        return (f"Produktionsenheten {pu} och {ru_label} {ru} säger olika om samma tal — produktion × pris räknar "
                f"det som {base}. Skriv produktionen i {base} och båda enheterna {base} (resurserna också i {base}).")
    unit = pu or (ru if key == "production_current" else None)
    factor = _scaled(unit)
    if factor == 1.0:
        return None
    v = company.num(key) if key else None              # samma tal som Asymmetry räknar med
    shown = f" ({_amount(v * factor)})" if v else ""
    also = f", också {ru_label} (Durrett-arket) med resurserna i {base}" if ru and shared else ""
    return (f"Produktionen i {unit} räknas som {base} — produktion × pris blir {_amount(factor)} gånger för litet. "
            f"Skriv produktionen i {base}{shown} och enheten {base}{also}.")


def _read_in_resource_unit(company: CompanyInput, base: str) -> bool:
    """Står produktionstalet Asymmetry läser (årsproduktionen, annars Produktion nu) i arkets resursenhet? Ja för
    Produktion nu (Durretts fält, i resursenheten), och för årsproduktionen när produktionsenheten inte är en känd
    enhet på prisets bas (tom, t för koppar, egen text). Nej utan produktion, och för årsproduktion i lb med
    produktionsenheten lb bredvid resurser i kt — den läses rätt."""
    key = next((k for k in ("annual_production", "production_current") if cval.usable(company, k)), None)
    return key == "production_current" or (
        key == "annual_production" and _unit_base(_text(company, "production_unit")) != base)


def resource_unit_conflict(company: CompanyInput) -> Optional[str]:
    """Varningen för arkets resursenhet på en annan bas än råvarans prisenhet (kt mot USD/lb), bara när produktion ×
    pris läser ett tal i den enheten (_read_in_resource_unit). Fältet finns bara i Durrett-arket. Durrett-motorns
    egen kontroll (quantity_unit_mismatch, värderingsraden) flaggar enheten ändå. None annars."""
    _reg, base = _price_base(company)
    if not base or not _read_in_resource_unit(company, base):
        return None
    return _resource_base_mismatch(company)


def production_unit_conflict(company: CompanyInput) -> Optional[str]:
    """Varningen där produktion × pris räknas utan omräkning (Asymmetrys Ekonomi och Analys, Durrett-arkets
    Economics), i Ctx.quantitys ordning: arkets resursenhet har en annan bas än råvarans prisenhet (kt mot USD/lb)
    och produktionen som läses står i den (resource_unit_conflict), produktionsenheten har en annan bas (t i stället
    för lb för koppar), produktionen står i en skalad enhet (koz, Mlb, kt …) som räknas som basenheten, eller
    produktionsenheten och arkets resursenhet säger olika om samma tal. None när enheten passar, saknas eller är en
    egen text, eller när råvaran saknar prisenhet."""
    reg, base = _price_base(company)
    if not base:
        return None
    return (resource_unit_conflict(company) or _base_mismatch(company, reg, base)
            or _scale_conflict(company, base))


def production_unit_offer(company: CompanyInput) -> Optional[tuple]:
    """('production_unit', Datapoint, None): enheten råvarans pris står i (registrets prisenhet: USD/oz → oz). Bara när
    fältet är tomt — aldrig över en skriven enhet, inte heller en på annan bas (t för koppar): talet står i den
    enheten, och en bar lb skulle märka 50 000 t som 50 000 lb utan att räkna om dem (_base_mismatch varnar och säger
    vad som behöver skrivas om). Aldrig heller när arkets resursenhet är skalad (koz, Mlb …) eller har en annan bas
    (kt mot USD/lb), av samma skäl. Datapunkten får varken datum eller källtyp — som en enhet skriven för hand, så Data
    Quality inte blåses upp. None annars."""
    reg, base = _price_base(company)
    if not base or _scaled(_known_unit(company, "resource_unit", base)) != 1.0 or _resource_base_mismatch(company):
        return None
    cur = _text(company, "production_unit") or None
    if cur:
        return None
    return ("production_unit", dp(base, kind="ASSUMPTION", source=f"Råvarans prisenhet ({reg.unit})",
                                  note=f"{reg.label} prissätts i {reg.unit}"), cur)


# (fält, kategori) från den högsta kategorin — samma ordning som vikterna i confidence.config.RESOURCE_CATEGORY_WEIGHT
CATEGORY_FIELDS = (("reserve_proven", "proven"), ("reserve_probable", "probable"), ("resource_measured", "measured"),
                   ("resource_indicated", "indicated"), ("resource_inferred", "inferred"))


def _amount(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ") if abs(v) >= 1000 else f"{v:g}"


def resource_category_offer(company: CompanyInput) -> Optional[tuple]:
    """('resource_category', Datapoint, nuvarande | None): den högsta kategorin med ett tal > 0 bland arkets egna
    reserver och resurser (reserve_*/resource_*). Katalogen räknar fältet som en bedömning, så förslaget säger att det
    är härlett ur arkets tal; datapunkten bär det talets källa, källtyp och datum. None utan ett sådant tal, eller när
    fältet redan har kategorin."""
    for key, cat in CATEGORY_FIELDS:
        if not cval.usable(company, key):
            continue
        v = company.num(key)
        if v is None or not v > 0:
            continue
        cur = str(company.get("resource_category").value) if company.has("resource_category") else None
        if cur == cat:
            return None
        basis = company.get(key)
        label = FIELD_BY_KEY[key].label
        unit = _text(company, "resource_unit")
        return ("resource_category", Datapoint(
            value=cat, kind=basis.kind, source=f"Härlett ur arket: {label}" + (f" ({basis.source})" if basis.source else ""),
            source_type=basis.source_type, pub_date=basis.pub_date, data_date=basis.data_date,
            note=f"härlett ur arkets egna tal: {label} {_amount(v)}{' ' + unit if unit else ''}"), cur)
    return None


def derived_offers(company: CompanyInput, keys=None) -> list:
    """[(fältnyckel, Datapoint, nuvarande | None)] som härleds ur arket och registret, inte ur natten: produktionsenhet
    och högsta resurskategori. keys = bara de fälten (formen förslaget står över)."""
    out = [o for o in (production_unit_offer(company), resource_category_offer(company)) if o]
    return [o for o in out if keys is None or o[0] in keys]


def spot_earnings(s: Optional[dict], company: CompanyInput) -> Optional[dict]:
    """Motorns framtida vinst (future_fcf, MUSD) räknad på nattens terminspris i stället för det sparade — bara för
    raden vid Råvarupris, inget skrivs. None när priset inte erbjuds (commodity_price_offer: fel råvara, fel enhet,
    samma tal) eller motorn inte kan räkna. {price, unit, asof, fcf_spot, fcf_saved}; fcf_* None när motorn saknar
    produktion eller kostnad (fcf_saved också utan sparat pris)."""
    offer = commodity_price_offer(s, company)
    if not offer:
        return None
    point = offer[1]
    try:
        from engines.durrett.engine import analyze

        def fcf(c):
            return (analyze(c).metrics.get("future_fcf") or {}).get("value")
        fresh = CompanyInput.from_dict(company.as_dict())
        fresh.set("commodity_price", point)
        out = {"price": point.value, "unit": point.unit, "asof": point.pub_date, "fcf_spot": fcf(fresh),
               "fcf_saved": fcf(company) if company.has("commodity_price") else None}
    except Exception:                                   # raden är en hjälp — motorn får aldrig fälla sidan
        return None
    return out


__all__ = ["proposals", "refresh_row", "ins_id_offer", "commodity_price_offer", "spot_earnings", "Datapoint",
           "AUTO_USD_SOURCES", "quarterly_burn_offer", "stage_offer", "stage_offer_text", "production_unit_offer",
           "production_unit_conflict", "resource_unit_conflict", "quantity_unit_mismatch", "resource_category_offer",
           "derived_offers",
           "CATEGORY_FIELDS"]
