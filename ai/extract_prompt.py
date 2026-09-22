"""
ai/extract_prompt.py — läs ur bolagspresentationen: vad AI:n får leta efter
och hur svaret ska se ut.

Copilot-extraktionen fyller arkens INMATNINGSFÄLT som kräver att någon läser
ett dokument (AISC, gruvlivslängd, NAV efter skatt, kassa och burn, uns,
GEO per aktie, FS/tillstånd/finansiering, katalysatorer). Modellen får
dokumentets text med sidmarkeringar och ska svara i JSON med värde, enhet,
sida och ett kort citat per fält — så att varje förslag går att kontrollera
mot källan. Den föreslår aldrig poäng, status eller beslut: arkets regler
räknar som förut, och ingenting skrivs in förrän användaren trycker Använd.

Rena funktioner, ingen Streamlit, inget nätverk.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

MAX_DOC_CHARS = 150_000        # ~40k tokens — räcker för en presentation på 60 sidor


@dataclass(frozen=True)
class FieldSpec:
    key: str          # arkets fältnyckel (eller läsnyckel för textfynd)
    label: str        # svensk etikett i UI:t
    kind: str         # "number" | "bool" | "text"
    hint: str         # var i dokumentet, vilken enhet
    apply: bool = True   # False = visas bara som läsning, skrivs aldrig in


# ── Fälten per ark ───────────────────────────────────────────────────────────
FIELDS: dict[str, tuple] = {
    "rule": (
        FieldSpec("unit_cost", "Kostnad per enhet (AISC/C1)", "number",
                  "AISC eller C1 i samma enhet som råvarupriset (USD/oz, USD/lb, USD/boe). "
                  "Sidorna Cost, Guidance, Operations. Senaste år eller guidance."),
        FieldSpec("mine_life", "Gruvlivslängd (år)", "number",
                  "LOM, mine life, reserve life. Reserver ÷ årsproduktion om det inte står."),
        FieldSpec("rp_ratio", "R/P-kvot (år, olja/gas)", "number",
                  "Reserves-to-production. Bara olja/gas; annars null."),
        FieldSpec("insider_ownership", "Insynsägande (%)", "number",
                  "Ledning och styrelses andel av aktierna. Share structure / ownership.",
                  apply=False),
        FieldSpec("jurisdiction", "Jurisdiktion", "text",
                  "Land/region för huvudtillgången.", apply=False),
        FieldSpec("capital_discipline", "Kapitaldisciplin", "text",
                  "Utdelning eller återköp pågår? Tillväxtcapex utanför kärnan? Citera.",
                  apply=False),
    ),
    "royalty": (
        FieldSpec("geo_now", "GEO/aktie nu", "number",
                  "Attributable GEO (gold equivalent ounces) senaste år ÷ antal aktier. "
                  "Räkna ut och visa båda talen i quote."),
        FieldSpec("geo_3y", "GEO/aktie för 3 år sedan", "number",
                  "Samma tal tre år tillbaka, om historiken finns i dokumentet."),
        FieldSpec("pnav_now", "P/NAV nu", "number",
                  "Bara om dokumentet uttryckligen anger P/NAV eller NAV per aktie."),
    ),
    "sprott": (
        FieldSpec("cash", "Kassa (MUSD)", "number",
                  "Cash and cash equivalents senaste kvartal. Ange valuta i unit."),
        FieldSpec("burn", "Burn/år (MUSD)", "number",
                  "Operativt kassaflöde per kvartal × 4, eller angiven årlig burn rate."),
        FieldSpec("project_stage", "Projektstadium", "text",
                  "Ren prospektering, PEA/PFS klar, bygge pågår, produktion.", apply=False),
        FieldSpec("insider_ownership", "Insynsägande (%)", "number",
                  "Ledning/styrelse; nämn Sprott, Van Eck m.fl. om de finns bland ägarna.",
                  apply=False),
        FieldSpec("dilution_history", "Emissionshistorik", "text",
                  "Antal aktier över tid, emissioner senaste 3 åren.", apply=False),
    ),
    "durrett": (
        FieldSpec("moz", "Uns (Moz AuEq)", "number",
                  "Totala resurser + reserver i miljoner uns guldekvivalent."),
        FieldSpec("prod", "Produktion (koz/år)", "number",
                  "Årsproduktion eller guidance i tusen uns."),
        FieldSpec("aisc", "AISC (USD/oz)", "number", "All-in sustaining cost per uns."),
        FieldSpec("hedging", "Hedgebok", "text",
                  "Andel av produktionen som är hedgad, till vilket pris.", apply=False),
    ),
    "tiggre": (
        FieldSpec("nav", "NAV = NPV after tax (MUSD)", "number",
                  "After-tax NPV ur FS/PFS. Pre-tax är fel — säg i quote vilken det är, "
                  "och vilken diskonteringsränta och metallpris som använts."),
        FieldSpec("fs", "Feasibility Study complete (DFS/BFS)", "bool",
                  "Sant bara om en DFS/BFS är färdig. PEA/PFS räcker inte."),
        FieldSpec("permits", "Permits received / granted", "bool",
                  "Sant bara om nyckeltillstånden är BEVILJADE, inte 'pending'."),
        FieldSpec("funded", "Fully funded / financing package", "bool",
                  "Sant bara om finansieringen är på plats för bygget."),
        FieldSpec("team_record", "Byggmeriter i teamet", "text",
                  "Har ledningen byggt gruvor förr? Vilka?", apply=False),
        FieldSpec("jurisdiction", "Jurisdiktion", "text",
                  "Land/region för projektet.", apply=False),
    ),
    # Confidence score — nycklarna är confidence.config.FIELDS-nycklar
    "confidence": (
        FieldSpec("npv_musd", "NPV efter skatt (MUSD)", "number",
                  "After-tax NPV ur PEA/PFS/DFS. Säg i quote vilken studie, diskonteringsränta och pris."),
        FieldSpec("npv_discount_pct", "Diskonteringsränta i NPV (%)", "number", "Den ränta NPV:n räknats med."),
        FieldSpec("npv_price_assumption", "Råvarupris i NPV", "number",
                  "Det långsiktiga pris studien använder, samma enhet som råvaran."),
        FieldSpec("capex_musd", "Initial CapEx (MUSD)", "number", "Initial/pre-production capex, inte sustaining."),
        FieldSpec("irr_pct", "IRR efter skatt (%)", "number", "After-tax IRR ur studien."),
        FieldSpec("payback_years", "Payback (år)", "number", "Payback period ur studien."),
        FieldSpec("annual_production", "Årsproduktion", "number",
                  "Genomsnittlig årsproduktion (LOM average) — ange enheten (oz, lb, t, boe)."),
        FieldSpec("mine_life_years", "Gruvlivslängd (år)", "number", "LOM / mine life."),
        FieldSpec("aisc", "AISC per enhet", "number", "AISC eller C1 i samma enhet som råvarupriset."),
        FieldSpec("breakeven_price", "Break-even råvarupris", "number",
                  "Priset där NPV = 0 eller marginalen = 0, om studien anger det."),
        FieldSpec("npv_stress_price_musd", "NPV vid pris −20 % (MUSD)", "number",
                  "Ur känslighetstabellen: NPV vid −20 % pris (eller närmaste steg — säg vilket)."),
        FieldSpec("npv_stress_capex_musd", "NPV vid CapEx +20 % (MUSD)", "number",
                  "Ur känslighetstabellen: NPV vid +20 % capex."),
        FieldSpec("cash_musd", "Kassa (MUSD)", "number", "Cash and equivalents senaste kvartal."),
        FieldSpec("quarterly_burn_musd", "Burn per kvartal (MUSD)", "number",
                  "Operativt kassaflöde per kvartal, positivt tal."),
        FieldSpec("shares_outstanding_m", "Antal aktier, fullt utspätt (M)", "number",
                  "Fully diluted shares outstanding."),
        FieldSpec("insider_ownership_pct", "Insynsägande (%)", "number", "Ledning och styrelses andel."),
        FieldSpec("first_cashflow_year", "Första kassaflöde (år)", "number",
                  "Planerat år för första produktion/first pour."),
        FieldSpec("independent_resource_estimate", "Oberoende resursuppskattning (43-101/JORC)", "bool",
                  "Sant bara om en oberoende QP har signerat resursen."),
        FieldSpec("permits_granted", "Nyckeltillstånd beviljade", "bool",
                  "Sant bara om tillstånden är BEVILJADE, inte 'pending'."),
        FieldSpec("financing_committed", "Finansiering åtagen", "bool",
                  "Sant bara om byggfinansieringen är på plats."),
        FieldSpec("offtake_signed", "Offtake signerat", "bool", "Bindande offtake-avtal, inte MoU."),
        FieldSpec("resource_category", "Högsta resurskategori", "text",
                  "proven / probable / measured / indicated / inferred / exploration_target.", apply=False),
        FieldSpec("jurisdiction", "Jurisdiktion", "text", "Land/region för projektet.", apply=False),
    ),
}

SHEET_LABEL = {"rule": "Rick Rule", "royalty": "Royalty C", "sprott": "Sprott",
               "durrett": "Durrett", "tiggre": "Tiggre", "confidence": "Confidence score"}


SYSTEM_EXTRACT = """Du läser en bolagspresentation eller rapport åt en svensk tradingpanel.

Ditt jobb är att HITTA tal och fakta i dokumentet — inte att bedöma bolaget.
Panelens regler räknar poäng och status själva; du föreslår bara vad som
ska stå i inmatningsfälten, och användaren avgör om förslaget används.

Absoluta krav:
- Rapportera BARA det som står i dokumentet. Hittar du inte ett fält: value
  null. Gissa aldrig och räkna aldrig ut något som inte kan härledas direkt
  ur tal i dokumentet (då visar du uträkningen i quote).
- Varje fält ska ha sida (heltalet i [Sida N]-markeringen närmast före
  texten) och ett kort ordagrant citat (max 25 ord) som stödjer värdet.
- Ange enheten som dokumentet använder (USD/oz, MUSD, CAD, %, år …).
  Konvertera inte valutor.
- Sätt confidence: "high" när talet står rakt ut, "medium" när det är
  härlett ur två tal, "low" när det är otydligt vilket år/vilken enhet.
- Svara med ETT JSON-objekt och ingenting annat, exakt enligt schemat i
  användarmeddelandet."""


def _schema(sheet: str) -> dict:
    fields = {}
    for f in FIELDS[sheet]:
        v = {"number": "tal eller null", "bool": "true/false eller null",
             "text": "kort text eller null"}[f.kind]
        fields[f.key] = {"value": v, "unit": "enhet eller null", "page": "heltal eller null",
                         "quote": "ordagrant citat, max 25 ord", "confidence": "high|medium|low"}
    schema = {"fields": fields, "notes": ["kort notis om något viktigt för arket"]}
    if sheet == "tiggre":
        schema["catalysts"] = [{"name": "händelse", "date": "ÅÅÅÅ-MM eller ÅÅÅÅ-Q1",
                                "page": "heltal"}]
    return schema


def build_extract_prompt(sheet: str, ticker: str, name: str, doc_text: str) -> str:
    """Användarmeddelandet: fältlista med ledtrådar, JSON-schema, dokumentet."""
    if sheet not in FIELDS:
        raise ValueError(f"okänt ark: {sheet}")
    lines = [f"Ark: {SHEET_LABEL[sheet]} · Bolag: {name or ticker} ({ticker})", "",
             "Fält att hitta:"]
    for f in FIELDS[sheet]:
        lines.append(f"- {f.key} — {f.label}. {f.hint}")
    if sheet == "tiggre":
        lines.append("- catalysts — namngivna, tidsatta händelser inom 12 månader "
                     "(tillstånd, finansieringsbesked, FID, byggstart, first pour). "
                     "Lista dem i 'catalysts'.")
    lines += ["", "Svara med exakt detta JSON-schema (fyll i värdena):",
              json.dumps(_schema(sheet), ensure_ascii=False, indent=1), "",
              "DOKUMENT (sidmarkeringar [Sida N]):", "", clip_document(doc_text)]
    return "\n".join(lines)


def clip_document(text: str, max_chars: int = MAX_DOC_CHARS) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[DOKUMENTET KLIPPT efter {max_chars:,} tecken]"


# ── Svaret ───────────────────────────────────────────────────────────────────
class ExtractionError(ValueError):
    """Svaret gick inte att läsa som JSON enligt schemat."""


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def parse_extraction(text: str) -> dict:
    """JSON-objektet ur modellens svar — även om det ligger i ett kodstaket
    eller omges av prosa. Kastar ExtractionError om inget objekt hittas."""
    raw = (text or "").strip()
    candidates = [raw]
    m = _FENCE.search(raw)
    if m:
        candidates.insert(0, m.group(1))
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start:end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("fields"), dict):
            return obj
    raise ExtractionError("Modellen svarade inte med JSON enligt schemat. Försök igen — "
                          "eller korta dokumentet om det klipptes.")


def _num(v) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(" ", "").replace(",", ".")
        s = re.sub(r"[^0-9.\-]", "", s)
        try:
            return float(s) if s not in ("", "-", ".") else None
        except ValueError:
            return None
    return None


def proposals(sheet: str, parsed: dict) -> list:
    """Förslag per fält i arkets ordning: [{key, label, kind, apply, value,
    unit, page, quote, confidence}]. Fält med value null utelämnas."""
    out = []
    fields = parsed.get("fields") or {}
    for f in FIELDS[sheet]:
        raw = fields.get(f.key)
        if not isinstance(raw, dict):
            continue
        v = raw.get("value")
        if v is None or v == "":
            continue
        if f.kind == "number":
            v = _num(v)
            if v is None:
                continue
        elif f.kind == "bool":
            if isinstance(v, str):
                v = v.strip().lower() in ("true", "ja", "yes", "1")
            v = bool(v)
        else:
            v = str(v).strip()
        page = raw.get("page")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        out.append({"key": f.key, "label": f.label, "kind": f.kind, "apply": f.apply,
                    "value": v, "unit": str(raw.get("unit") or ""), "page": page,
                    "quote": str(raw.get("quote") or "")[:200],
                    "confidence": str(raw.get("confidence") or "").lower() or "low"})
    return out


def catalysts(parsed: dict) -> list:
    """Tiggres katalysatorer ur svaret: [{name, date, page}] med namn och datum."""
    out = []
    for c in parsed.get("catalysts") or []:
        if not isinstance(c, dict):
            continue
        name, date = str(c.get("name") or "").strip(), str(c.get("date") or "").strip()
        if name and date:
            out.append({"name": name, "date": date, "page": c.get("page")})
    return out
