"""
lukacs.py — Lukacs FV-modulen (Masterguiden 4.1, Kontrollsystemen).

CSM svarar på om bolaget överlever varje scenario. Den här modulen svarar på
vad det är VÄRT där. Kedjan, fritt efter Peter Lukacs FCF-ramverk:

  normaliserat Forward FCF
    → skuldnedbetalning
    → prognostiserat aktieantal (återköp = DS:ens spegelbild)
    → Forward FCF per aktie ÷ target FCF-yield = fair value
    → uppsida %

Modulens hela poäng är att target-yielden är LÅST till FCF-kvalitetsklassen.
Utan den låsningen kan man räkna fram vilken uppsida man vill genom att välja
yield efter smak — och då mäter modellen sin egen önsketänkande i stället för
bolaget. Därför är en yield utanför klassens band ett FEL, inte en varning.

Forward FCF räknas ALLTID på normaliserat råvarupris (kol-regeln), aldrig
toppår. Det kan modulen inte kontrollera åt dig; hjälptexten säger det, och
'what must go right' är där antagandet skrivs ned så att ett brutet antagande
blir en modellförlust i riskdoktrinen och inte ett väntläge.

Rena funktioner, ingen Streamlit — lukacs_ui.py renderar, testerna räknar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import controls as ctl

GREEN, AMBER, ORANGE, RED, DIM = ("#2d8a4e", "#d4943a", "#d4701f", "#c44545",
                                  "#8a8578")

# Scenariospinen. Bear/Base/Bull finns i både tre- och femscenariomatrisen, så
# FV räknas likadant för kärninnehav som för resten.
FV_SCENARIOS = (ctl.BEAR, ctl.BASE, ctl.BULL)


# ═════════════════════════════════════════════════════════════════════════════
#  FCF-kvalitetsklassen — och yield-bandet den låser
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class QualityClass:
    code: str
    name: str
    desc: str
    lo: Optional[float]      # target FCF-yield, procent
    hi: Optional[float]


QUALITY_CLASSES: tuple[QualityClass, ...] = (
    QualityClass("A", "Återkommande",
                 "Återkommande, förutsägbart FCF.", 6.0, 8.0),
    QualityClass("B", "Stabilt",
                 "Stabilt, begränsat cykliskt.", 8.0, 10.0),
    QualityClass("C", "Cykliskt",
                 "Tydligt cykliskt — de flesta råvarubolag.", 10.0, 14.0),
    QualityClass("D", "Temporärt",
                 "Extraordinärt eller temporärt kassaflöde.", None, None),
)

CLASS_BY_CODE = {q.code: q for q in QUALITY_CLASSES}
CLASS_CODES = tuple(q.code for q in QUALITY_CLASSES)

NOT_FCF_VALUED = "Klass D — värderas ej på FCF"


def quality_class(code) -> Optional[QualityClass]:
    return CLASS_BY_CODE.get(str(code or "").strip().upper()[:1] or "_")


def yield_band(code) -> Optional[tuple]:
    """(låg, hög) i procent, eller None för klass D och osatt klass."""
    q = quality_class(code)
    if q is None or q.lo is None:
        return None
    return (q.lo, q.hi)


def yield_error(code, value) -> Optional[str]:
    """Felmeddelandet när yielden inte hör hemma i klassens band.

    Fel, inte varning: låsningen ÄR modulen. Kan man välja yield fritt räknar
    man fram den uppsida man redan bestämt sig för.
    """
    q = quality_class(code)
    if q is None:
        return None
    if q.lo is None:
        return NOT_FCF_VALUED
    v = _num(value)
    if v is None or v == 0:
        return None
    v = round(v, 6)
    if v < q.lo or v > q.hi:
        return (f"Target-yield {v:g} % ligger utanför klass {q.code}:s band "
                f"({q.lo:g}–{q.hi:g} %). Yielden är låst till kvalitetsklassen "
                f"— ändra klassen om du menar allvar, inte yielden.")
    return None


def _num(value, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def _pos(value) -> Optional[float]:
    """Positivt tal, annars None. Noll i ett inmatningsfält = ej ifyllt."""
    v = _num(value)
    return v if v is not None and v > 0 else None


def num(value, default: Optional[float] = None) -> Optional[float]:
    """Publik variant av _num — ytan behöver den för number_input-värden."""
    return _num(value, default)


# ═════════════════════════════════════════════════════════════════════════════
#  Fair value
# ═════════════════════════════════════════════════════════════════════════════
def fair_value_per_share(forward_fcf_musd, target_yield_pct,
                         future_shares_m) -> Optional[float]:
    """(Forward FCF ÷ target-yield) ÷ prognostiserat aktieantal.

    MUSD ÷ miljoner aktier ger USD per aktie, så enheterna går ihop utan
    omräkning. Guiden skriver kedjan som FCF per aktie ÷ yield; det är samma
    tal, och den här ordningen gör mellansteget (bolagets fair value totalt)
    läsbart om man vill kontrollräkna.
    """
    fcf = _num(forward_fcf_musd)
    y = _pos(target_yield_pct)
    shares = _pos(future_shares_m)
    if fcf is None or y is None or shares is None:
        return None
    return round((fcf / (y / 100.0)) / shares, 6)


def upside_pct(fair_value, price) -> Optional[float]:
    """fv/kurs − 1, i procent."""
    fv = _num(fair_value)
    p = _pos(price)
    if fv is None or p is None:
        return None
    return round((fv / p - 1) * 100, 6)


def margin_of_safety(fv_base, price) -> Optional[float]:
    """(FV_base − kurs) / FV_base, i procent.

    Mot Base, aldrig mot Bull: säkerhetsmarginalen ska mätas mot det man tror
    händer, inte mot det man hoppas.
    """
    fv = _pos(fv_base)
    p = _num(price)
    if fv is None or p is None:
        return None
    return round((fv - p) / fv * 100, 6)


MOS_VERY = "Mycket attraktiv"
MOS_ATTRACTIVE = "Attraktiv"
MOS_WATCH = "Bevaka"
MOS_NONE = "Ingen marginal"
MOS_COLOR = {MOS_VERY: GREEN, MOS_ATTRACTIVE: GREEN, MOS_WATCH: AMBER,
             MOS_NONE: RED}

MOS_VERY_MIN = 40.0
MOS_BUY_MIN = 25.0        # köpgrindens steg 5
MOS_WATCH_MIN = 10.0


def mos_band(mos) -> Optional[str]:
    """> 40 mycket attraktiv · 25–40 attraktiv · 10–25 bevaka · < 10 ingen.

    Avrundat innan jämförelsen: exakt 25,0 % ska passera köpgrinden, och
    flyttalsrepresentationen av en kvot ska inte vara det som avgör.
    """
    m = _num(mos)
    if m is None:
        return None
    m = round(m, 6)
    if m > MOS_VERY_MIN:
        return MOS_VERY
    if m >= MOS_BUY_MIN:
        return MOS_ATTRACTIVE
    return MOS_WATCH if m >= MOS_WATCH_MIN else MOS_NONE


def mos_passes_gate(mos) -> bool:
    """Steg 5 i köpgrinden.

    Snabbreferensen: "MoS < 25 % = inget nytt köp" — alltså passerar exakt
    25,0 %. Kapiteltexten skriver "kräver > 25 %"; de två meningarna skiljer
    sig bara i den punkten och den lösare läsningen är den som står i
    regeltabellen.
    """
    m = _num(mos)
    return m is not None and round(m, 6) >= MOS_BUY_MIN


# ═════════════════════════════════════════════════════════════════════════════
#  Sannolikheter och expected value
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_PROBS = {ctl.BEAR: 20.0, ctl.BASE: 60.0, ctl.BULL: 20.0}
PROB_SUM = 100.0


def probabilities(row: dict) -> dict:
    """Sannolikheterna. Default 20/60/20 om användaren inte avvikit."""
    r = row or {}
    if not r.get("prob_deviation"):
        return dict(DEFAULT_PROBS)
    stored = r.get("probs") or {}
    return {s: _num(stored.get(s), DEFAULT_PROBS[s]) for s in FV_SCENARIOS}


def probability_errors(row: dict) -> list:
    """Vad som hindrar en avvikelse från default.

    Guiden låser 20/60/20 och kräver skriftlig motivering för avvikelser. Utan
    motivering är en justerad sannolikhet bara en tumme på vågen.
    """
    r = row or {}
    if not r.get("prob_deviation"):
        return []
    out = []
    probs = probabilities(r)
    total = round(sum(probs.values()), 6)
    if total != PROB_SUM:
        out.append(f"Sannolikheterna summerar till {total:g} %, inte "
                   f"{PROB_SUM:g} %.")
    if not str(r.get("prob_motivation", "")).strip():
        out.append("Avvikelse från 20/60/20 kräver skriftlig motivering.")
    return out


def expected_value(fair_values: dict, probs: dict) -> Optional[float]:
    """Σ(scenariovärde × sannolikhet). None om något scenario saknar FV."""
    fvs, p = fair_values or {}, probs or {}
    total = 0.0
    for s in FV_SCENARIOS:
        fv = _num(fvs.get(s))
        w = _num(p.get(s))
        if fv is None or w is None:
            return None
        total += fv * (w / 100.0)
    return round(total, 6)


# ═════════════════════════════════════════════════════════════════════════════
#  Säljregeln och deleveraging-flaggan
# ═════════════════════════════════════════════════════════════════════════════
TRIM_UPSIDE_MIN = 20.0
TRIM_TEXT = "Omvärdering gjord — trimma enligt regel"


def trim_warning(upside_base, is_holding: bool) -> bool:
    """Säljregeln: trimma när uppsidan mot Base-FV understiger ~20 %.

    Bara för innehav. På en kandidat är låg uppsida ett skäl att avstå, och
    det säger säkerhetsmarginalen redan.
    """
    if not is_holding:
        return False
    u = _num(upside_base)
    return u is not None and round(u, 6) < TRIM_UPSIDE_MIN


DELEV_ND_MIN = 1.0          # nettoskuld/EBITDA över detta = halv position
DELEV_YEARS_MAX = 3.0       # och år-till-låg-skuld måste vara under detta
DELEV_TEXT = ("Skuld över 1,0× vid köp — max halv position, och år till låg "
              "skuld måste vara under 3.")


def deleveraging_state(nd_ebitda, years_to_low_debt=None) -> dict:
    """Discovery-screenerns hårda regel.

    Guiden: "skuld > 1,0 vid köp → max halv position OCH krav år-till-låg-skuld
    < 3". Halveringen hänger på skulden ensam; åren är ett SEPARATE krav
    ovanpå, inte villkoret för halveringen.
    """
    nd = _num(nd_ebitda)
    if nd is None or round(nd, 6) <= DELEV_ND_MIN:
        return {"applies": False, "half_position": False, "years_ok": None,
                "gaps": []}
    years = _pos(years_to_low_debt)
    gaps = []
    if years is None:
        years_ok = None
        gaps.append("År till låg skuld är inte ifyllt — kravet är under "
                    f"{DELEV_YEARS_MAX:g} år.")
    else:
        years_ok = round(years, 6) < DELEV_YEARS_MAX
        if not years_ok:
            gaps.append(f"År till låg skuld {years:g} — kravet är under "
                        f"{DELEV_YEARS_MAX:g} år.")
    return {"applies": True, "half_position": True, "years_ok": years_ok,
            "gaps": gaps}


def max_position_pct(nd_ebitda, normal_cap, years_to_low_debt=None) -> Optional[float]:
    """Positionstaket efter deleveraging-regeln. None när taket är okänt."""
    cap = _pos(normal_cap)
    if cap is None:
        return None
    state = deleveraging_state(nd_ebitda, years_to_low_debt)
    return round(cap / 2, 6) if state["half_position"] else cap


# ═════════════════════════════════════════════════════════════════════════════
#  Sammanställningen — allt kortet och grinden behöver
# ═════════════════════════════════════════════════════════════════════════════
FV_STRATEGIES = ("producenter",)      # modulen är byggd för producentkorten
WMGR_MISSING = "'What must go right' för Base är inte ifyllt"


def fv_applicable(strategy: str = "") -> bool:
    """Om modulen alls är meningsfull för strategin.

    Insider och swing värderas inte på forward FCF, och en utvecklare utan
    kassaflöde har inget att dividera med.
    """
    return (strategy or "").strip().lower() in FV_STRATEGIES


def fv_required(position_pct, strategy: str = "") -> bool:
    """Obligatorisk för producentköp över 2 % av totalen."""
    if not fv_applicable(strategy):
        return False
    return ctl.SEC_CSM in ctl.required_sections(position_pct, strategy)


def evaluate(row: dict) -> dict:
    """Hela modulen för en rad — räknas om vid varje anrop, lagras aldrig.

    Allt härleds ur inmatningen. Sparas ett resultat blir det inaktuellt i
    samma sekund som ett antagande ändras, och en inaktuell fair value är
    farligare än ingen alls.
    """
    r = row or {}
    fv_in = r.get("fv") or {}
    code = r.get("fcf_kvalitet")
    q = quality_class(code)
    shares = r.get("framtida_antal_aktier")
    price = r.get("aktuell_kurs")

    values, upsides, errors = {}, {}, []
    for s in FV_SCENARIOS:
        sc = fv_in.get(s) or {}
        err = yield_error(code, sc.get("target_yield"))
        if err and err != NOT_FCF_VALUED:
            errors.append(f"{s}: {err}")
        if q is not None and q.lo is None:
            continue                       # klass D värderas inte på FCF
        fv = fair_value_per_share(sc.get("forward_fcf_musd"),
                                  sc.get("target_yield"), shares)
        if fv is not None:
            values[s] = fv
            upsides[s] = upside_pct(fv, price)

    fv_base = values.get(ctl.BASE)
    mos = margin_of_safety(fv_base, price)
    probs = probabilities(r)
    ev = expected_value(values, probs)

    return {
        "quality": q,
        "not_fcf_valued": q is not None and q.lo is None,
        "values": values,
        "upsides": upsides,
        "fv_base": fv_base,
        "upside_base": upsides.get(ctl.BASE),
        "mos": mos,
        "mos_band": mos_band(mos),
        "expected_value": ev,
        "ev_upside": upside_pct(ev, price),
        "probs": probs,
        "prob_errors": probability_errors(r),
        "yield_errors": errors,
        "what_must_go_right": str(r.get("what_must_go_right", "")).strip(),
        "trim": trim_warning(upsides.get(ctl.BASE), bool(r.get("is_holding"))),
        "delev": deleveraging_state(r.get("nd_ebitda"),
                                    r.get("ar_till_lag_skuld")),
    }


def gate_gaps(row: dict, position_pct=None, strategy: str = "") -> list:
    """Vad som hindrar köpgrindens steg 5 från att bli grönt.

    Tom lista = mekaniskt grönt. Modulen krävs bara där guiden kräver den;
    är den inte obligatorisk returneras inga luckor och grinden faller
    tillbaka på det manuella krysset.
    """
    if not fv_required(position_pct, strategy):
        return []
    ev = evaluate(row)
    gaps = list(ev["yield_errors"]) + list(ev["prob_errors"])
    gaps += list(ev["delev"]["gaps"])
    if ev["not_fcf_valued"]:
        gaps.append(f"{NOT_FCF_VALUED} — säkerhetsmarginalen kan inte "
                    f"härledas ur FCF för den här klassen.")
        return gaps
    if ev["fv_base"] is None:
        gaps.append("Fair value för Base går inte att räkna — fyll i forward "
                    "FCF, target-yield, aktieantal och kurs.")
        return gaps
    if not mos_passes_gate(ev["mos"]):
        gaps.append(f"Säkerhetsmarginal {ev['mos']:.1f} % — kravet är "
                    f"{MOS_BUY_MIN:g} %.")
    if not ev["what_must_go_right"]:
        gaps.append(WMGR_MISSING)
    return gaps


def gate_ok(row: dict, position_pct=None, strategy: str = "") -> bool:
    """Steg 5 mekaniskt: MoS >= 25 % OCH 'what must go right' ifyllt."""
    return not gate_gaps(row, position_pct, strategy)
