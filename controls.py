"""
controls.py — Kontrollsystemen (Masterguiden 4.0).

De tre kontrollerna som ligger ovanpå strategiernas egna poängmodeller, plus
proportionalitetsregeln som avgör vilka av dem som faktiskt krävs:

  DS  — Dilution Score, 0–10. RISKPOÄNG: lägre är bättre. Frågan den svarar på
        är inte "är bolaget bra" utan "hur mycket av min uppsida kommer att
        spädas bort innan tesen hinner spela ut".
  AQS — Asset Quality Score, 0–16. Tillgångens kvalitet oberoende av priset.
  CSM — Commodity Sensitivity Matrix. Överlever caset ett Bear-scenario, och
        hur mycket hävstång får jag i Bull?

Proportionalitetsregeln: en 1 %-position ska inte kosta lika mycket arbete som
en 5 %-position. Positionsstorleken avgör vad som krävs — allt annat döljs.

Rena funktioner, ingen Streamlit. Flikarna (scoring, tiggre, producers,
scorecard) renderar dem; testerna kör dem mot specens siffror.

NOTE: Masterguiden 4.0 hänvisar till exempel i Del 4 (Kontrollsystemen) som
inte funnits tillgängliga här. Trösklarna nedan är tilläggsspecens, och de
räknar bara det specen skriver ut — inget är extrapolerat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

GREEN, AMBER, ORANGE, RED, DIM = ("#2d8a4e", "#d4943a", "#d4701f", "#c44545",
                                  "#8a8578")


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    zero: str
    one: str
    two: str
    help: str = ""


def _num(value, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def _score(value) -> int:
    v = _num(value)
    return 0 if v is None else max(0, min(2, int(v)))


def _any_set(row: dict, fields) -> bool:
    return any(_num((row or {}).get(f.key)) is not None for f in fields)


# ═════════════════════════════════════════════════════════════════════════════
#  DS — Dilution Score
# ═════════════════════════════════════════════════════════════════════════════
DS_FIELDS: tuple[Field, ...] = (
    Field("ds_runway", "Runway",
          "> 24 mån", "12–24 mån", "< 12 mån",
          "Auto-förslag ur runway-beräkningen där den finns. Överskrivbar."),
    Field("ds_capex", "Capex-gap",
          "Finansierad eller ej relevant", "Hanterbart gap", "Stort gap",
          "Vad som saknas mellan kassan och det som ska byggas."),
    Field("ds_warrants", "Teckningsoptioner",
          "Försumbara", "Måttliga", "Stora utestående",
          "Utestående optioner är utspädning som redan är beslutad."),
    Field("ds_aktier_3ar", "Aktier 3 år",
          "Stabilt", "+10–25 %", "> +25 %",
          "Hur många fler aktier finns det nu än för tre år sedan?"),
    Field("ds_historik", "Historik",
          "Disciplinerad", "Blandad", "Serieutspädare",
          "Det starkaste enskilda tecknet: den som spätt ut förr gör det igen."),
)

DS_MAX = 10
DS_LOW, DS_OK, DS_HIGH = "Låg", "Acceptabel", "Hög"
DS_EXTREME = "EXTREM"
DS_BAND_COLOR = {DS_LOW: GREEN, DS_OK: AMBER, DS_HIGH: ORANGE, DS_EXTREME: RED}
DS_BLOCK_MIN = 6          # från och med 6 låses köp utan finansieringskatalysator

DS_INFO_FIELDS = (
    ("fully_diluted_shares", "Fullt utspätt antal aktier"),
    ("forvantad_utspadning", "Förväntad utspädning till katalysator (%)"),
)

DS_BLOCK_TEXT = "Hög DS — köp ENDAST på finansieringsbeskedet"


def ds_total(row: dict) -> Optional[int]:
    """0–10. None när ingen delfråga är satt — obedömd är inte samma som låg."""
    if not _any_set(row, DS_FIELDS):
        return None
    return sum(_score((row or {}).get(f.key)) for f in DS_FIELDS)


def ds_band(total: Optional[int]) -> Optional[str]:
    if total is None:
        return None
    if total <= 2:
        return DS_LOW
    if total <= 5:
        return DS_OK
    return DS_HIGH if total <= 7 else DS_EXTREME


def ds_runway_suggestion(runway_years) -> Optional[int]:
    """Förslag ur runwayen. Notera att skalan är omvänd mot poängmodellens:
    här är hög runway 0 riskpoäng, där är den 2 kvalitetspoäng."""
    y = _num(runway_years)
    if y is None:
        return None
    if y > 2:
        return 0
    return 1 if y >= 1 else 2


def has_financing_catalyst(row: dict) -> bool:
    """Både vad och när krävs — 'kommer nog en emission snart' är inget besked."""
    r = row or {}
    return bool(str(r.get("fin_catalyst_text", "")).strip()
                and str(r.get("fin_catalyst_date", "")).strip())


def ds_blocks_buy(row: dict) -> bool:
    """Låser köp: DS >= 6 utan en tidsatt finansieringskatalysator.

    Obedömd DS låser inte här — den fångas av köpgrindens luck-regel i
    scorecardet, där tomt fält är standardbeslutet INGEN AFFÄR.
    """
    total = ds_total(row)
    if total is None or total < DS_BLOCK_MIN:
        return False
    return not has_financing_catalyst(row)


def ds_note(row: dict) -> Optional[str]:
    """Texten som ska synas när DS är hög."""
    total = ds_total(row)
    if total is None or total < DS_BLOCK_MIN:
        return None
    if has_financing_catalyst(row):
        return DS_BLOCK_TEXT
    return (f"DS {total}/{DS_MAX} — köp låst. Fyll i finansieringskatalysator "
            f"(vad och när) för att låsa upp, annars är utspädningen din "
            f"uppsida.")


# ═════════════════════════════════════════════════════════════════════════════
#  AQS — Asset Quality Score
# ═════════════════════════════════════════════════════════════════════════════
AQS_FIELDS: tuple[Field, ...] = (
    Field("aqs_kostnad", "Kostnadsposition",
          "Övre halvan av kurvan", "Mitten", "Lägsta kvartilen",
          "De dyraste producenterna dör först i prisfall."),
    Field("aqs_livslangd", "Livslängd / resurs",
          "< 5 år", "5–10 år", "> 10 år",
          "En kort gruva hinner inte igenom en cykel."),
    Field("aqs_metallurgi", "Metallurgi / recovery",
          "Problematisk eller obevisad", "Normal", "Enkel, hög utvinning",
          "Svår metallurgi är den vanligaste tekniska dödsorsaken."),
    Field("aqs_capex", "Capex-ekonomi",
          "Tung capex mot NPV", "Rimlig", "Låg capex-intensitet",
          "Capex per producerad enhet avgör om projektet någonsin byggs."),
    Field("aqs_jurisdiktion", "Jurisdiktion",
          "Nedre halvan", "Topphalvan", "Toppkvartilen",
          "Fraser-rankingen."),
    Field("aqs_infrastruktur", "Infrastruktur / tillstånd",
          "Saknas, långt kvar", "Delvis på plats", "Väg, kraft, tillstånd klara",
          "Infrastruktur är dold capex."),
    Field("aqs_management", "Management / ägande",
          "Serieutspädare, inga meriter", "Delvis meriter",
          "Byggt förr, äger själva",
          "Samma fråga som strategipoängens ägarfaktor."),
    Field("aqs_expansion", "Expansionsoptionalitet",
          "Ingen", "Viss", "Tydlig, billig expansion",
          "Gratis uppsida ovanpå basfallet."),
)

AQS_MAX = 16
AQS_HIGH, AQS_OK, AQS_DISCOUNT, AQS_PASS = ("Hög kvalitet", "Godkänd",
                                            "Kräver stor rabatt", "PASS")
AQS_BAND_COLOR = {AQS_HIGH: GREEN, AQS_OK: AMBER, AQS_DISCOUNT: ORANGE,
                  AQS_PASS: RED}


def aqs_total(row: dict) -> Optional[int]:
    if not _any_set(row, AQS_FIELDS):
        return None
    return sum(_score((row or {}).get(f.key)) for f in AQS_FIELDS)


def aqs_band(total: Optional[int]) -> Optional[str]:
    if total is None:
        return None
    if total >= 13:
        return AQS_HIGH
    if total >= 10:
        return AQS_OK
    return AQS_DISCOUNT if total >= 7 else AQS_PASS


# Återanvändning i stället för dubbelinmatning: jurisdiktion och management
# frågas redan i Producenter A (0/1) och i Lobo (0–2).
def aqs_prefill_from_producer(row: dict) -> dict:
    """Producenter A: 0/1-kryssen skalas till AQS 0–2."""
    r = row or {}
    out = {}
    if r.get("jurisdiktion") is not None:
        out["aqs_jurisdiktion"] = 2 if r.get("jurisdiktion") else 0
    if r.get("insyn") is not None:
        out["aqs_management"] = 2 if r.get("insyn") else 0
    return out


def aqs_prefill_from_tiggre(factors: dict) -> dict:
    """Lobo: faktorerna är redan 0–2 och går in oförändrade."""
    f = factors or {}
    out = {}
    if _num(f.get("jurisdiktion")) is not None:
        out["aqs_jurisdiktion"] = _score(f.get("jurisdiktion"))
    if _num(f.get("manniskor")) is not None:
        out["aqs_management"] = _score(f.get("manniskor"))
    return out


AQS_PREFILLED = ("aqs_jurisdiktion", "aqs_management")


# ═════════════════════════════════════════════════════════════════════════════
#  CSM — Commodity Sensitivity Matrix
# ═════════════════════════════════════════════════════════════════════════════
BEAR, BASE, BULL = "Bear", "Base", "Bull"
DEEP_BEAR, SUPER_BULL = "Djup bear", "Super bull"
SCENARIOS_3 = (BEAR, BASE, BULL)
SCENARIOS_5 = (DEEP_BEAR, BEAR, BASE, BULL, SUPER_BULL)

PRODUCER, DEVELOPER = "producent", "utvecklare"

CSM_BEAR_FAIL = "CSM: Bear kräver kapital — PASS eller vänta på finansiering"


def scenarios(is_core: bool = False) -> tuple:
    """Tre scenarier räcker; fem bara för kärninnehav."""
    return SCENARIOS_5 if is_core else SCENARIOS_3


def bear_survival_suggestion(kind: str, bear: dict,
                             secured_cash: bool = False) -> Optional[bool]:
    """Auto-förslag för bear_överlevnad.

    Nej när en utvecklare har ett finansieringsbehov i Bear utan säkrad kassa
    — då är Bear-scenariot en emission till kursen där ingen vill teckna.
    Förslag, inte facit: fältet får skrivas över.
    """
    b = bear or {}
    if kind == DEVELOPER:
        need = _num(b.get("financing_need"))
        if need is None:
            return None
        if need > 0 and not secured_cash:
            return False
        return True
    fcf = _num(b.get("fcf_musd"))
    if fcf is None:
        return None
    return fcf >= 0


def csm_red_flag(kind: str, matrix: dict, secured_cash: bool = False) -> bool:
    """Röd flagga när Bear inte överlevs.

    Ett uttryckligt Ja i matrisen väger tyngre än förslaget — du kan veta
    något om kassan som fälten inte fångar.
    """
    m = matrix or {}
    stated = m.get("bear_survival")
    if stated is not None:
        return not bool(stated)
    return bear_survival_suggestion(kind, m.get(BEAR, {}), secured_cash) is False


def leverage_ratio(matrix: dict) -> Optional[float]:
    """bull_fcf / bear_fcf — skiljer billigt från billigt MED hävstång.

    Odefinierad när Bear-kassaflödet är noll eller negativt: en kvot mot noll
    är inte oändlig hävstång, den är ett bolag som inte överlever Bear.
    """
    m = matrix or {}
    bull = _num((m.get(BULL) or {}).get("fcf_musd"))
    bear = _num((m.get(BEAR) or {}).get("fcf_musd"))
    if bull is None or bear is None or bear <= 0:
        return None
    return round(bull / bear, 2)


def csm_complete(matrix: dict, is_core: bool = False) -> bool:
    """Alla scenarier har minst ett ifyllt tal."""
    m = matrix or {}
    for s in scenarios(is_core):
        row = m.get(s) or {}
        if not any(_num(row.get(k)) is not None
                   for k in ("price", "fcf_musd", "nav_musd", "financing_need")):
            return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
#  Proportionalitetsregeln
# ═════════════════════════════════════════════════════════════════════════════
SEC_STRATEGY = "strategipoäng"
SEC_DS = "DS"
SEC_AQS = "AQS"
SEC_CSM = "CSM"

FULL_WORK_MIN_PCT = 2.0        # över 2 % av total krävs allt
LIGHT_WORK_MIN_PCT = 1.0       # 1–2 %: DS + strategipoäng

# Swing och insider handlar inte om tillgångskvalitet eller råvarupris.
NO_ASSET_STRATEGIES = ("swing", "momentum", "insider")


def required_sections(position_pct, strategy: str = "",
                      dilution_risk: bool = False) -> set:
    """Vilka kontroller som krävs för en position av den här storleken.

    Poängen är att en 1 %-lott inte ska kosta samma arbete som ett
    kärninnehav — och att swing och insider aldrig ska kosta det alls.
    """
    key = (strategy or "").strip().lower()
    if key in NO_ASSET_STRATEGIES:
        return {SEC_STRATEGY, SEC_DS} if dilution_risk else {SEC_STRATEGY}

    pct = _num(position_pct, 0.0) or 0.0
    if pct > FULL_WORK_MIN_PCT:
        return {SEC_STRATEGY, SEC_DS, SEC_AQS, SEC_CSM}
    return {SEC_STRATEGY, SEC_DS}


def section_required(section: str, position_pct, strategy: str = "",
                     dilution_risk: bool = False) -> bool:
    return section in required_sections(position_pct, strategy, dilution_risk)
