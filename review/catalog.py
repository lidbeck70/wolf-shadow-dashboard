"""
review/catalog.py — fältkatalogen: varje inmatningsfält i arken som ett SheetField.

Byggd ur arkens egna konstanter: producers.CHECKS, controls.DS_FIELDS/DS_INFO_FIELDS/AQS_FIELDS och
CSM-scenarierna, lukacs.FV_SCENARIOS/CLASS_CODES, scoring.FACTORS, tiggre.FACTORS/SCREEN_PHRASES,
insider.ROLES/TRIGGERS/BOOL_FIELDS, scorecard.GATES och confidence.config.FIELDS (Wolf Asymmetry:
asymmetry.fields.all_keys). Ledtrådarna kommer ur ai/extract_prompt.FIELDS och FieldSpec.hint. Ingen tröskel
kopieras hit — vilka kontroller som hör till ett kort avgör controls.required_sections, precis som i arket.

Vem fyller:
  Börsdata   sifferuppdateringen har talet (förslag i arket, "Använd" skriver in det)
  ChatGPT    fakta med källa: rapport, presentation, teknisk rapport
  Bedömning  poäng 0–2 och omdömen som ChatGPT får FÖRESLÅ — du bekräftar
  Bara du    beslut: position, köpgrind, säljregel, kvalitetsgrind, trigger (AI beslutar aldrig storlek)

Sökvägarna är punktade: 'factors.balans', 'screen.fs', 'csm.Bear.fcf_musd', 'fv.Base.target_yield',
'catalysts.0.date' och 'fields.aisc' (Durrett/Wolf Asymmetry-arkets Datapoint). Rena funktioner, ingen Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import controls as ctl
import insider as ins
import lukacs
import producers as prod
import review_link
import scorecard as sc
import scoring as sco
import tiggre as tig
from ai import extract_prompt as xp
from asymmetry import fields as afields
from confidence import config as cfg
from confidence import store as cs
from confidence.data import validation as cval
from confidence.data.provenance import Datapoint, from_dict
from engines.durrett import refresh as dref

# ── Typer och vem som fyller ─────────────────────────────────────────────────
NUMBER, INT, SCORE02, BOOL, CHOICE, TEXT, DATE, MONTH = (
    "number", "int", "score02", "bool", "choice", "text", "date", "month")
KINDS = (NUMBER, INT, SCORE02, BOOL, CHOICE, TEXT, DATE, MONTH)
BORSDATA, CHATGPT, BEDOMNING, BARA_DU = "Börsdata", "ChatGPT", "Bedömning", "Bara du"
WHO = (BORSDATA, CHATGPT, BEDOMNING, BARA_DU)


@dataclass(frozen=True)
class SheetField:
    """Ett inmatningsfält: var det bor (ark, lager, sökväg), vad det är och vem som fyller det."""
    sheet: str
    store: str
    path: str
    label: str
    kind: str
    who: str
    unit: str = ""
    choices: tuple = ()
    lo: Optional[float] = None
    hi: Optional[float] = None
    hint: str = ""
    gate: str = ""          # avsnittet eller grinden som läser fältet
    refresh: str = ""       # fältet i sifferuppdateringens rad (sheets_refresh.json) när Börsdata har talet

    @property
    def key(self) -> str:
        """Fältnyckeln i exporten: 'rule:unit_cost' — unik även när flera ark ligger i samma underlag."""
        return f"{self.sheet}:{self.path}"


@dataclass(frozen=True)
class Sheet:
    key: str
    label: str
    store: str              # storage-namnet (data/<store>.json)
    bucket: str             # listan/dicten i lagret
    strategy: str           # scorecardets och kontrollernas strateginamn
    refresh: str            # sifferuppdateringens arknyckel ('producers:<id>' …)


def _label(key: str) -> str:
    return review_link.sheet_name(key)


SHEETS: dict = {s.key: s for s in (
    Sheet("rule", _label("rule"), prod.STORE, prod.PRODUCERS, "producenter", "producers"),
    Sheet("royalty", _label("royalty"), prod.STORE, prod.ROYALTY, "royalty", "producers"),
    Sheet("sprott", _label("sprott"), sco.STORE, sco.SPROTT, "sprott", "scoring"),
    Sheet("durrett", _label("durrett"), sco.STORE, sco.DURRETT, "durrett", "scoring"),
    Sheet("tiggre", _label("tiggre"), tig.STORE, "candidates", "tiggre", "tiggre"),
    Sheet("insider", _label("insider"), ins.STORE, "signals", "insider", "insider"),
    Sheet("scorecard", "Scorecard", sc.STORE, "cards", "", ""),
    Sheet("confidence", "Durrett 10-steg", cs.STORE, "companies", "confidence", "confidence"),
    Sheet("asymmetry", "Wolf Asymmetry", cs.STORE, "companies", "", "confidence"),
)}
# scorecardets strateginamn → arket kandidaten granskas i
STRATEGY_SHEET: dict = {s.strategy: s.key for s in SHEETS.values() if s.strategy}
FLAT_SHEETS = ("rule", "royalty", "sprott", "durrett", "tiggre", "insider")
CONFIDENCE_SHEETS = ("confidence", "asymmetry")


def _f(sheet: str, path: str, label: str, kind: str, who: str, **kw) -> SheetField:
    return SheetField(sheet, SHEETS[sheet].store, path, label, kind, who, **kw)


def _xp(xp_sheet: str, key: str) -> str:
    """Ledtråden ur Copilot-extraktionens fältlista (ai/extract_prompt.FIELDS), tom om fältet saknas där."""
    for f in xp.FIELDS.get(xp_sheet, ()):
        if f.key == key:
            return f.hint
    return ""


# ── Kontrollerna: samma urval som arket visar ────────────────────────────────
# Arken med egen positionsstorlek (Rick Rule: fältet på kortet; Tiggre: arkets standard 2–4 %).
_OWN_POSITION = {"rule": lambda r: r.get("position_pct"),
                 "tiggre": lambda r: tig._num(r.get("position_pct"), tig.POS_MAX_PCT)}
_DS_ALWAYS = ("sprott", "durrett")           # Poängmodellen frågar DS oavsett storlek


def position(sheet: str, row: Optional[dict], position_pct=None):
    """Positionen proportionalitetsregeln läser: scorecardets när den skickas in, annars kortets egen."""
    if position_pct is not None:
        return position_pct
    fn = _OWN_POSITION.get(sheet)
    return fn(row or {}) if fn else None


def sections(sheet: str, row: Optional[dict], position_pct=None) -> set:
    """Kontrollerna (DS/AQS/CSM) för kortet — controls.required_sections, som i arket.

    Utan inskickad position visas det arket självt visar; med scorecardets position det köpgrinden kräver.
    """
    r = row or {}
    req: set = set()
    if position_pct is not None or sheet in _OWN_POSITION:
        req = set(ctl.required_sections(position(sheet, r, position_pct), SHEETS[sheet].strategy,
                                        bool(r.get("dilution_risk"))))
    if sheet in _DS_ALWAYS:
        req.add(ctl.SEC_DS)
    req.discard(ctl.SEC_STRATEGY)
    return req


def csm_kind(sheet: str, row: Optional[dict]) -> str:
    """Producent eller utvecklare i CSM: radens val, annars arkets (Tiggre är utvecklare per definition)."""
    k = (row or {}).get("csm_kind")
    if k in (ctl.PRODUCER, ctl.DEVELOPER):
        return k
    return ctl.DEVELOPER if sheet == "tiggre" else ctl.PRODUCER


def fv_shown(sheet: str, row: Optional[dict], position_pct=None) -> bool:
    """Lukacs FV: obligatorisk i CSM-blocket för producentköp över gränsen, annars bara när den slagits på."""
    r = row or {}
    strategy = SHEETS[sheet].strategy
    if ctl.SEC_CSM in sections(sheet, r, position_pct):
        return lukacs.fv_required(position(sheet, r, position_pct), strategy) and csm_kind(sheet, r) == ctl.PRODUCER
    return lukacs.fv_applicable(strategy) and bool(r.get("fv_manual"))


def _ds(sheet: str) -> list:
    out = [_f(sheet, f.key, f"DS · {f.label}", SCORE02, BEDOMNING, gate="DS",
              hint=f"Riskpoäng — lägre är bättre. 0 = {f.zero} · 1 = {f.one} · 2 = {f.two}. {f.help}")
           for f in ctl.DS_FIELDS]
    out += [_f(sheet, k, label, NUMBER, CHATGPT, lo=0.0, gate="DS",
               unit="%" if "(%)" in label else "",
               hint="" if "(%)" in label else "Samma enhet som bolaget redovisar — skriv den i Fyll i: Enhet.")
            for k, label in ctl.DS_INFO_FIELDS]
    out += [_f(sheet, "fin_catalyst_text", "Finansieringskatalysator — vad", TEXT, CHATGPT, gate="DS",
               hint=f"Krävs när DS är {ctl.DS_BLOCK_MIN} eller mer, annars är köpet låst. T.ex. riktad emission "
                    f"fulltecknad."),
            _f(sheet, "fin_catalyst_date", "Finansieringskatalysator — när", MONTH, CHATGPT, gate="DS",
               hint="När beskedet väntas.")]
    return out


def _aqs(sheet: str) -> list:
    return [_f(sheet, f.key, f"AQS · {f.label}", SCORE02, BEDOMNING, gate="AQS",
               hint=f"0 = {f.zero} · 1 = {f.one} · 2 = {f.two}. {f.help}") for f in ctl.AQS_FIELDS]


def _csm(sheet: str, row: dict) -> list:
    if csm_kind(sheet, row) == ctl.PRODUCER:
        cols = (("price", "råvarupris", "råvarans prisenhet", 0.0), ("fcf_musd", "fritt kassaflöde", "MUSD", None))
    else:
        cols = (("price", "råvarupris", "råvarans prisenhet", 0.0), ("nav_musd", "NAV", "MUSD", None),
                ("financing_need", "finansieringsbehov", "MUSD", 0.0))
    out = [_f(sheet, "is_core", "Kärninnehav (fem scenarier)", BOOL, BARA_DU, gate="CSM")]
    for s in ctl.scenarios(bool(row.get("is_core"))):
        for k, lbl, unit, lo in cols:
            out.append(_f(sheet, f"csm.{s}.{k}", f"CSM {s} · {lbl}", NUMBER, BEDOMNING, unit=unit, lo=lo, gate="CSM",
                          hint="Scenariots antagande — modellerat, inte rapporterat. Skriv antagandet i kommentaren."))
    out += [_f(sheet, "secured_cash", "Kassan räcker genom Bear (säkrad finansiering)", BOOL, BARA_DU, gate="CSM"),
            _f(sheet, "csm.bear_survival", "Överlever Bear?", BOOL, BARA_DU, gate="CSM",
               hint="Tomt = följ förslaget ur matrisen.")]
    return out


def _fv(sheet: str, row: dict) -> list:
    import lukacs_ui
    band = lukacs.yield_band(row.get("fcf_kvalitet"))
    g = "Lukacs FV"
    out = [_f(sheet, "fcf_kvalitet", "FCF-kvalitet", CHOICE, BEDOMNING, choices=lukacs.CLASS_CODES, gate=g,
              hint=" · ".join(lukacs_ui._class_label(q) for q in lukacs.QUALITY_CLASSES)),
           _f(sheet, "framtida_antal_aktier", "Framtida antal aktier", NUMBER, BEDOMNING, unit="miljoner", lo=0.0,
              gate=g, hint="Prognos inklusive återköp och utspädning."),
           _f(sheet, "aktuell_kurs", "Aktuell kurs", NUMBER, BORSDATA, lo=0.0, gate=g, refresh="price",
              hint="Samma valuta som forward FCF anges i.")]
    for s in lukacs.FV_SCENARIOS:
        out.append(_f(sheet, f"fv.{s}.forward_fcf_musd", f"FV {s} · forward FCF", NUMBER, BEDOMNING, unit="MUSD",
                      gate=g, hint="Normaliserat råvarupris, aldrig toppår."))
        out.append(_f(sheet, f"fv.{s}.target_yield", f"FV {s} · target-yield", NUMBER, BEDOMNING, unit="%",
                      lo=band[0] if band else 0.0, hi=band[1] if band else None, gate=g,
                      hint="Låst till FCF-klassens band — ändra klassen, inte yielden."))
    out.append(_f(sheet, "what_must_go_right", "What must go right — Base", TEXT, BEDOMNING, gate=g,
                  hint="Råvarupris · produktion · capex · amorteringstakt · återköp."))
    if lukacs.deleveraging_state(row.get("nd_ebitda"))["applies"]:
        out.append(_f(sheet, "ar_till_lag_skuld", "År till låg skuld", NUMBER, CHATGPT, unit="år", lo=0.0, gate=g,
                      hint="Nettoskuld ÷ årlig amortering."))
    out.append(_f(sheet, "is_holding", "Detta är ett innehav", BOOL, BARA_DU, gate=g))
    return out


def control_fields(sheet: str, row: Optional[dict], position_pct=None) -> list:
    """DS, AQS, CSM och Lukacs FV för kortet — samma urval som controls_ui.render_all ritar."""
    r = row or {}
    req = sections(sheet, r, position_pct)
    out = []
    if ctl.SEC_DS in req:
        out += _ds(sheet)
    if ctl.SEC_AQS in req:
        out += _aqs(sheet)
    if ctl.SEC_CSM in req:
        out += _csm(sheet, r)
    if fv_shown(sheet, r, position_pct):
        out += _fv(sheet, r)
    return out


# ── Arken ────────────────────────────────────────────────────────────────────
def _rule(row: dict, pos) -> list:
    s = "rule"
    out = [
        _f(s, "ev_ebitda", "EV/EBITDA", NUMBER, BORSDATA, unit="×", lo=0.0, gate="Arket", refresh="ev_ebitda"),
        _f(s, "nd_ebitda", "Nettoskuld/EBITDA", NUMBER, BORSDATA, unit="×", gate="Arket", refresh="nd_ebitda",
           hint="Negativt = nettokassa."),
        _f(s, "unit_cost", "Kostnad per enhet (AISC/C1)", NUMBER, CHATGPT, unit="samma enhet som råvarupriset",
           lo=0.0, gate="Rick Rule-poängen", hint=_xp("rule", "unit_cost")),
        _f(s, "price", "Råvarupris nu", NUMBER, BORSDATA, unit="råvarans prisenhet", lo=0.0, gate="Rick Rule-poängen",
           refresh="commodity_price", hint="Terminspriset — kostnaden per enhet ska stå i samma enhet."),
        _f(s, "mine_life", "Gruvlivslängd", NUMBER, CHATGPT, unit="år", lo=0.0, gate="Strykregeln",
           hint=_xp("rule", "mine_life")),
        _f(s, "rp_ratio", "R/P-kvot (olja/gas)", NUMBER, CHATGPT, unit="år", lo=0.0, gate="Strykregeln",
           hint=_xp("rule", "rp_ratio")),
    ]
    out += [_f(s, k, label, BOOL, BEDOMNING, gate="Rick Rule-poängen", hint=h) for k, label, h in prod.CHECKS]
    out += [
        _f(s, "ev_musd", "EV", NUMBER, BORSDATA, unit="MUSD", lo=0.0, gate="EV/NAV", refresh="ev_musd",
           hint="Börsvärde + nettoskuld i miljoner USD."),
        _f(s, "nav_musd", "NAV efter skatt", NUMBER, CHATGPT, unit="MUSD", lo=0.0, gate="EV/NAV",
           hint=_xp("rule", "nav_musd")),
        _f(s, "nav_price", "Metallpris i NAV", NUMBER, CHATGPT, unit="råvarans prisenhet", lo=0.0, gate="EV/NAV",
           hint=_xp("rule", "nav_price")),
        _f(s, "nav_source", "NAV-källa", TEXT, CHATGPT, gate="EV/NAV",
           hint="Studie eller analytikerkonsensus, år och diskonteringsränta (t.ex. NI 43-101 2025, 5 %)."),
        _f(s, "position_pct", "Tänkt position", NUMBER, BARA_DU, unit="% av total", lo=0.0,
           gate="Proportionalitetsregeln"),
    ]
    return out + control_fields(s, row, pos)


def _royalty(row: dict, pos) -> list:
    s, g = "royalty", "Royalty-signalen"
    out = [
        _f(s, "pnav_now", "P/NAV nu", NUMBER, CHATGPT, unit="×", lo=0.0, gate=g, hint=_xp("royalty", "pnav_now")),
        _f(s, "pnav_bottom", "P/NAV historisk botten", NUMBER, CHATGPT, unit="×", lo=0.0, gate=g,
           hint="Bolagets egen lägsta P/NAV i historiken."),
        _f(s, "ev_now", "EV/EBITDA nu", NUMBER, BORSDATA, unit="×", lo=0.0, gate=g, refresh="ev_ebitda"),
        _f(s, "ev_median", "EV/EBITDA median", NUMBER, BORSDATA, unit="×", lo=0.0, gate=g,
           refresh="ev_ebitda_median", hint="Bolagets egen median, tio år."),
        _f(s, "geo_now", "GEO/aktie nu", NUMBER, CHATGPT, unit="GEO per aktie", lo=0.0, gate=g,
           hint=_xp("royalty", "geo_now")),
        _f(s, "geo_3y", "GEO/aktie för 3 år sedan", NUMBER, CHATGPT, unit="GEO per aktie", lo=0.0, gate=g,
           hint=_xp("royalty", "geo_3y")),
        _f(s, "level", "Nivå", CHOICE, BEDOMNING, choices=tuple(prod.ROYALTY_LEVELS), gate="Nivå",
           hint=" ".join(prod.LEVEL_NOTE[lv] for lv in prod.ROYALTY_LEVELS)),
    ]
    return out + control_fields(s, row, pos)


def _scoring(sheet: str, row: dict, pos) -> list:
    if sheet == "sprott":
        out = [_f(sheet, "cash", "Kassa", NUMBER, BORSDATA, unit="MUSD", lo=0.0, gate="Runway-grinden",
                  refresh="cash_musd", hint=_xp("sprott", "cash")),
               _f(sheet, "burn", "Burn/år", NUMBER, BORSDATA, unit="MUSD/år", lo=0.0, gate="Runway-grinden",
                  refresh="burn_musd", hint=_xp("sprott", "burn"))]
    else:
        g = "Köpregeln"
        out = [_f(sheet, "mcap", "Börsvärde", NUMBER, BORSDATA, unit="MUSD", lo=0.0, gate=g, refresh="mcap_musd"),
               _f(sheet, "moz", "Uns", NUMBER, CHATGPT, unit="Moz AuEq", lo=0.0, gate="Värderingen",
                  hint=_xp("durrett", "moz")),
               _f(sheet, "profit", "Framtida vinst/år", NUMBER, BEDOMNING, unit="MUSD/år", lo=0.0, gate=g,
                  hint="Hjälpräknaren: produktion × (målpris − AISC)."),
               _f(sheet, "prod", "Produktion", NUMBER, CHATGPT, unit="koz/år", lo=0.0, gate="Hjälpräknaren",
                  hint=_xp("durrett", "prod")),
               _f(sheet, "target", "Målpris", NUMBER, BEDOMNING, unit="USD/oz", lo=0.0, gate="Hjälpräknaren"),
               _f(sheet, "aisc", "AISC", NUMBER, CHATGPT, unit="USD/oz", lo=0.0, gate="Hjälpräknaren",
                  hint=_xp("durrett", "aisc"))]
    out += [_f(sheet, f"factors.{f.key}", f.label, SCORE02, BEDOMNING, gate="Fem faktorer",
               hint=f"2 = {f.two} · 1 = {f.one} · 0 = {f.zero}") for f in sco.FACTORS]
    return out + control_fields(sheet, row, pos)


def _tiggre(row: dict, pos) -> list:
    s = "tiggre"
    out = [_f(s, f"screen.{k}", label, BOOL, CHATGPT, gate="Grovsållning", hint=_xp("tiggre", k))
           for k, label in tig.SCREEN_PHRASES]
    out += [_f(s, "mcap", "Börsvärde", NUMBER, BORSDATA, unit="MUSD", lo=0.0, gate="U/N-kvoten", refresh="mcap_musd"),
            _f(s, "nav", "NAV = NPV after tax", NUMBER, CHATGPT, unit="MUSD", lo=0.0, gate="U/N-kvoten",
               hint=_xp("tiggre", "nav")),
            _f(s, "downside", "Nedsida", NUMBER, BEDOMNING, unit="%", gate="U/N-kvoten",
               hint="Värsta rimliga scenariot: utspädning, byggförsening.")]
    # U/N-faktorn räknas ur kvoten i arket — den matas aldrig in
    out += [_f(s, f"factors.{k}", label, SCORE02, BEDOMNING, gate="Fem faktorer", hint=h)
            for k, label, h in tig.FACTORS if k != "un"]
    out += control_fields(s, row, pos)
    for i in range(max(tig.CATALYSTS_MIN, len(row.get("catalysts") or []))):
        out.append(_f(s, f"catalysts.{i}.name", f"Katalysator {i + 1} — händelse", TEXT, CHATGPT,
                      gate="Katalysatorer", hint=f"Namngiven händelse inom 12 månader. Kedjan: "
                                                 f"{' → '.join(tig.CATALYST_CHAIN)}."))
        out.append(_f(s, f"catalysts.{i}.date", f"Katalysator {i + 1} — när", MONTH, CHATGPT,
                      gate="Katalysatorer", hint="Förväntad månad."))
    return out


def _insider(row: dict, pos) -> list:
    s, g = "insider", "Insiderpoängen"
    out = [_f(s, "insiders", "Antal insiders 30 dgr", INT, CHATGPT, unit="st", lo=0.0, gate=g,
              hint="Olika insiders med riktiga marknadsköp inom 30 dagar — inte optionslösen, program, arv eller "
                   "interna omflyttningar."),
           _f(s, "role", "Högsta roll", CHOICE, CHATGPT, choices=tuple(ins.ROLES), gate=g,
              hint="Högsta roll bland köparna."),
           _f(s, "amount", "Belopp totalt", NUMBER, CHATGPT, unit="tkr", lo=0.0, gate=g,
              hint="Summan av klustrets marknadsköp i tusen kronor.")]
    out += [_f(s, k, label, BOOL, CHATGPT, gate=g, hint=h) for k, label, h in ins.BOOL_FIELDS]
    out += [_f(s, "cluster_avg", "Klustrets snittkurs", NUMBER, CHATGPT, unit="valuta/aktie", lo=0.0,
               gate="Passa-regeln", hint="Snittkursen i klustrets köp."),
            _f(s, "price_now", "Kurs nu", NUMBER, BORSDATA, unit="valuta/aktie", lo=0.0, gate="Passa-regeln",
               refresh="price"),
            _f(s, "gate", "Kvalitetsgrind OK?", CHOICE, BARA_DU, choices=(ins.GATE_YES, ins.GATE_NO),
               gate="Kvalitetsgrinden", hint="Ja kräver alla: " + " · ".join(ins.GATE_CRITERIA)),
            _f(s, "trigger", "Teknisk trigger", CHOICE, BARA_DU, choices=tuple(ins.TRIGGERS), gate="Teknisk trigger",
               hint=" · ".join(f"{lbl}: {what}" for lbl, what, _where in ins.TRIGGER_CRITERIA))]
    return out + control_fields(s, row, pos)


def _scorecard(card: dict, strategies: tuple) -> list:
    s, g = "scorecard", "Beslutet"
    out = [_f(s, "strategy", "Strategi", CHOICE, BARA_DU, choices=tuple(strategies), gate=g),
           _f(s, "position_pct_total", "Position", NUMBER, BARA_DU, unit="% av total", lo=0.0, gate=g),
           _f(s, "position_pct_strategy", "Position av strategidelen", NUMBER, BARA_DU, unit="% av strategidelen",
              lo=0.0, gate=g),
           _f(s, "valuation", "Värdering", TEXT, BEDOMNING, gate=g, hint="T.ex. 4,2× EV/EBITDA, P/NAV 0,35."),
           _f(s, "trigger", "Trigger / katalysator", TEXT, BEDOMNING, gate=g, hint="Vad ska hända, och när?"),
           _f(s, "sell_rule", "Vald säljregel", TEXT, BARA_DU, gate=g,
              hint="Vilken regel tar dig ur — bestäms före köpet.")]
    mechanical_fv = lukacs.fv_required(card.get("position_pct_total"), card.get("strategy", ""))
    for gt in sc.GATES:
        # två steg räknas av grinden själv och kan inte kryssas (scorecard.gate_state)
        if gt.key == "inga_roda_flaggor" or (gt.key == "sakerhetsmarginal" and mechanical_fv):
            continue
        out.append(_f(s, gt.key, gt.label, BOOL, BARA_DU, gate="Köpgrinden", hint=gt.help))
    return out


# Confidence-arket: Börsdata fyller det sifferuppdateringen hämtar (engines/durrett/refresh) och det Wolf
# Asymmetry kallar AUTO; poäng, kill switches och scenarioantaganden är bedömningar.
_CONF_BORSDATA = frozenset({fkey for _b, fkey, _u, _k in dref._MAP} | {"market_currency"} | set(afields.AUTO))
_CONF_JUDGED = ("resource_category", "lassonde_stage", "target_ev_ebitda", "target_p_nav", "dilution_score")
_PILLAR = {**{k: lbl for k, lbl, _m in cfg.PILLARS}, **{k: lbl for k, lbl, _m in cfg.CONFIDENCE_PARTS},
           "kill": "Kill switches", "scenarios": "Scenarier"}


def _pillar(p: str) -> str:
    if p in _PILLAR:
        return _PILLAR[p]
    return "Durrett · " + p.split("_", 1)[1].replace("_", " ") if p.startswith("durrett_") else p


def _spec_who(spec) -> str:
    if spec.key in _CONF_BORSDATA:
        return BORSDATA
    if spec.pillar == "kill" or spec.key in _CONF_JUDGED or (spec.kind == "int" and spec.unit == "p"):
        return BEDOMNING
    return CHATGPT


def _spec_field(sheet: str, spec) -> SheetField:
    if spec.kind == "int":
        kind = SCORE02 if (spec.unit == "p" and spec.max == 2) else INT
    else:
        kind = {"number": NUMBER, "bool": BOOL, "choice": CHOICE, "text": TEXT, "date": DATE}.get(spec.kind, TEXT)
    choices = tuple(spec.choices)
    if kind == INT and spec.max is not None and spec.max <= 10:
        choices = tuple(range(0, int(spec.max) + 1))
    numeric = spec.kind in ("number", "int")
    lo = 0.0 if numeric and spec.key not in cval._MAY_BE_NEGATIVE else None
    return _f(sheet, f"fields.{spec.key}", spec.label, kind, _spec_who(spec), unit=spec.unit if spec.unit != "p" else "",
              choices=choices, lo=lo, hi=spec.max if numeric else None, hint=_xp("confidence", spec.key) or spec.hint,
              gate=_pillar(spec.pillar))


def _stage(row: dict, stage: Optional[str]) -> str:
    s = stage or (row or {}).get("stage") or "developer"
    return s if s in cfg.STAGES else "developer"


def _confidence(sheet: str, row: dict, stage: Optional[str]) -> list:
    st_ = _stage(row, stage)
    if sheet == "asymmetry":                     # bara fälten de tre poängen läser
        specs = [cfg.FIELD_BY_KEY[k] for k in dict.fromkeys(afields.all_keys(st_))]
    else:
        specs = cfg.fields_for(st_)
    return [_spec_field(sheet, spec) for spec in specs]


def _as_row(row) -> dict:
    if isinstance(row, dict):
        return row
    if hasattr(row, "as_dict"):                  # CompanyInput
        return row.as_dict()
    return {}


def fields_for(sheet: str, row=None, stage: Optional[str] = None, position_pct=None,
               strategies: tuple = ()) -> list:
    """Fälten för ett kort i arket, i arkets ordning.

    row avgör det som beror på raden (CSM-typ, kärninnehav, FV-klassens band, antal katalysatorer, Confidence-
    stage). position_pct är scorecardets position — då följer kontrollerna köpgrinden i stället för kortet.
    strategies är scorecardkortets strategival.
    """
    if sheet not in SHEETS:
        raise KeyError(f"okänt ark: {sheet}")
    r = _as_row(row)
    if sheet == "rule":
        return _rule(r, position_pct)
    if sheet == "royalty":
        return _royalty(r, position_pct)
    if sheet in ("sprott", "durrett"):
        return _scoring(sheet, r, position_pct)
    if sheet == "tiggre":
        return _tiggre(r, position_pct)
    if sheet == "insider":
        return _insider(r, position_pct)
    if sheet == "scorecard":
        return _scorecard(r, tuple(strategies) or ((r.get("strategy"),) if r.get("strategy") else ()))
    return _confidence(sheet, r, stage)


# ── Läsa och skriva ett fält ─────────────────────────────────────────────────
def _is_dp(path: str) -> bool:
    return path.startswith("fields.")


def get_value(row, path: str):
    """Värdet på en punktad sökväg, eller None. Listindex som siffra ('catalysts.0.date'); Confidence-arkets
    Datapoint ('fields.aisc') ger sitt värde."""
    cur = _as_row(row)
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            i = int(part)
            cur = cur[i] if i < len(cur) else None
        else:
            return None
        if cur is None:
            return None
    if isinstance(cur, Datapoint):
        return cur.value
    if _is_dp(path) and isinstance(cur, dict):
        return cur.get("value")
    return cur


def get_point(row, path: str) -> Optional[Datapoint]:
    """Confidence-arkets Datapoint med källa och datum, eller None (även för de platta arken)."""
    if not _is_dp(path):
        return None
    raw = (_as_row(row).get("fields") or {}).get(path.split(".", 1)[1])
    return from_dict(raw) if raw is not None else None


def set_value(row: dict, path: str, value) -> None:
    """Skriv ett värde på en punktad sökväg och skapa det som saknas på vägen. None tar bort nyckeln (som när
    ett fält töms i arket). En Confidence-sökväg ('fields.x') lagras som Datapoint: ett Datapoint/dict med value
    tas som det är, ett rått värde blir ett Datapoint utan källa (ASSUMPTION) — aldrig ett gammalt fälts källa."""
    parts = path.split(".")
    cur = row
    for i, part in enumerate(parts[:-1]):
        nxt_list = parts[i + 1].isdigit()
        if isinstance(cur, list):
            idx = int(part)
            while len(cur) <= idx:
                cur.append({})
            if not isinstance(cur[idx], (dict, list)):
                cur[idx] = [] if nxt_list else {}
            cur = cur[idx]
        else:
            child = cur.get(part)
            if not isinstance(child, (dict, list)):
                child = [] if nxt_list else {}
                cur[part] = child
            cur = child
    last = parts[-1]
    if _is_dp(path) and value is not None:
        if isinstance(value, Datapoint):
            value = value.as_dict()
        elif not (isinstance(value, dict) and "value" in value):
            value = from_dict({"value": value}).as_dict()
    if isinstance(cur, list):
        idx = int(last)
        while len(cur) <= idx:
            cur.append(None)
        cur[idx] = value
    elif value is None:
        cur.pop(last, None)
    else:
        cur[last] = value


def is_empty(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def same_value(old, new) -> bool:
    """Samma värde? Tal jämförs som tal, Ja/Nej som Ja/Nej, text som text."""
    if is_empty(old):
        return is_empty(new)
    if isinstance(old, bool) or isinstance(new, bool):
        return isinstance(old, bool) and isinstance(new, bool) and old == new
    try:
        return abs(float(old) - float(new)) < 1e-9
    except (TypeError, ValueError):
        return str(old).strip() == str(new).strip()


# ── Källan till ett inläst värde (platta arken) ──────────────────────────────
# row['sources'] = {sökväg: {"value": det inlästa värdet, "source": "ChatGPT: <källa> · <datum>"}}. Värdet sparas med
# källan, så att källan bara visas medan arket har kvar det inlästa värdet — ett värde du skrivit själv har ingen.
SOURCES = "sources"


def set_source(row: dict, path: str, value, source: str) -> None:
    """Källan för ett inläst värde i ett platt ark (review.importer.apply)."""
    src = row.get(SOURCES) if isinstance(row.get(SOURCES), dict) else {}
    src[path] = {"value": value, "source": source}
    row[SOURCES] = src


def source_for(row, path: str) -> str:
    """Källan för fältets värde i ett platt ark, eller tomt: bara medan värdet är det som lästes in med källan."""
    src = _as_row(row).get(SOURCES)
    rec = src.get(path) if isinstance(src, dict) else None
    if not isinstance(rec, dict) or "value" not in rec:
        return ""
    v = get_value(row, path)
    if is_empty(v) or not same_value(rec["value"], v):
        return ""
    return str(rec.get("source") or "")


def progress(sheet: str, row, stage: Optional[str] = None, position_pct=None) -> tuple:
    """(ifyllda, totalt, saknade sökvägar) för kortet — underlaget till 'x/y fält ifyllda'."""
    fs = fields_for(sheet, row, stage, position_pct)
    missing = [f.path for f in fs if is_empty(get_value(row, f.path))]
    return len(fs) - len(missing), len(fs), missing
