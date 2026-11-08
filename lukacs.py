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


MOS_NO_PRICE = "Aktuell kurs saknas — säkerhetsmarginalen mot fair value för Base kan inte räknas."
MOS_NO_VALUE = "Fair value för Base är noll eller negativ — ingen säkerhetsmarginal att räkna."


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
    if ev["mos"] is None:
        # Fair value finns men marginalen går inte att räkna: kursen saknas, eller Base-värdet är noll/negativt.
        if _num(row.get("aktuell_kurs") if isinstance(row, dict) else None) is None:
            gaps.append(MOS_NO_PRICE)
        else:
            gaps.append(MOS_NO_VALUE)
    elif not mos_passes_gate(ev["mos"]):
        gaps.append(f"Säkerhetsmarginal {ev['mos']:.1f} % — kravet är "
                    f"{MOS_BUY_MIN:g} %.")
    if not ev["what_must_go_right"]:
        gaps.append(WMGR_MISSING)
    return gaps


# ═════════════════════════════════════════════════════════════════════════════
#  Rimlighetskontroller — bara varningar, de ändrar ingen beräkning eller grind
# ═════════════════════════════════════════════════════════════════════════════
PRICE_MATCH = 0.03          # sparad kurs inom 3 % av Börsdatas = samma tal, i Börsdatas valuta
FX_DISTINCT = 2.0           # valutor minst 2× från USD (NOK, SEK, DKK, pence): kursen hör till den valuta den ligger närmast
FX_LIVE_SANE = 0.30         # en dagskurs mer än 30 % från tabellens är ett datafel — då gäller tabellen
FCF_TOO_SMALL = 0.02        # forward FCF under 2 % av referens-FCF:en — troligen fel enhet
FCF_TOO_LARGE = 50.0        # … eller över 50× den
FCF_EV_MIN = 0.002          # utan referens-FCF: under 0,2 % av EV (och en referens under 0,2 % av storleken = break-even)
FCF_SIZE_MAX = 2.0          # över 2× bolagets storlek (EV eller börsvärde, det större) — en FCF-yield över 200 %
FCF_LOCAL_BAND = 0.30       # forward FCF × växelkursen inom ±30 % av referensen = talet står i lokal valuta
FCF_LOCAL_FX_MIN = 5.0      # … bara för valutor minst 5× från USD (NOK, SEK, DKK) — PLN:s 4× ligger inom vanliga modellgap
FCF_LOCAL_REF_MIN = 0.02    # … och bara mot en referens på minst 2 % av storleken: nära noll är en kvot på 10× ingen enhet
FCF_LOCAL_YIELD = 0.40      # Base-FCF över 40 % av storleken men under 40 % omräknad från NOK/SEK/DKK = lokal valuta
FCF_LOCAL_EBITDA = 1.5      # … och (när Börsdatas EV/EBITDA finns) över 1,5× EBITDA: en äkta hög yield ligger under EBITDA
PENCE = ("GBX", "GBp")      # London i pence: GBP/100


def _job_table() -> dict:
    """Sifferuppdateringens fasta tabell (sheets_refresh.FX_TO_USD) — den nattjobbet räknar om MUSD-talen med."""
    try:
        from sheets_refresh import FX_TO_USD
        return FX_TO_USD
    except Exception:                   # pragma: no cover — tabellen är en ren modulkonstant
        return {}


def _is_pence(raw: str) -> bool:
    return raw in PENCE or raw.upper() == "GBX"


def fx_to_usd(currency) -> Optional[float]:
    """USD per kursenhet ur sifferuppdateringens fasta tabell. London i pence (GBX/GBp) = GBP/100. None för en
    valuta tabellen inte har: den räknas inte om som om den vore USD. Tabellen är grov (den följer inte dagens
    kurs) — kursförslaget föredrar dagens växelkurs när den finns."""
    raw = str(currency or "").strip()
    if not raw:
        return None
    table = _job_table()
    if _is_pence(raw):
        return table["GBP"] / 100.0 if "GBP" in table else None
    return table.get(raw.upper())


def fx_pair(currency) -> Optional[tuple]:
    """(Yahoo-ticker, delare) för dagens växelkurs till USD: ('NOKUSD=X', 1); pence ('GBPUSD=X', 100).
    None för USD och tomt."""
    raw = str(currency or "").strip()
    if not raw or raw.upper() == "USD":
        return None
    return ("GBPUSD=X", 100.0) if _is_pence(raw) else (f"{raw.upper()}USD=X", 1.0)


def _looks_local(p: Optional[float], local: float, fx: Optional[float]) -> bool:
    """Den sparade kursen är i Börsdatas valuta: inom 3 % av Börsdatas kurs — eller, när valutan ligger minst 2× från
    USD, närmare den lokala kursen än den omräknade (kursen har rört sig sedan den skrevs in, men en NOK-kurs blir
    inte en USD-kurs av det). Jämförs mot den oavrundade USD-kursen: en öreskurs blir aldrig 0 USD."""
    if p is None or p <= 0:
        return False
    if abs(p - local) / local <= PRICE_MATCH:
        return True
    if not fx or max(fx, 1.0 / fx) < FX_DISTINCT:
        return False
    import math
    return abs(math.log(p / local)) < abs(math.log(p / (local * fx)))


def price_currency(price, ref: Optional[dict], live: Optional[tuple] = None) -> Optional[dict]:
    """Kursens valuta mot forward FCF:ens (USD — MUSD ÷ miljoner aktier ger USD per aktie).

    ref = sifferuppdateringens rad för bolaget (price, currency). live = (USD per enhet, datum) — dagens växelkurs
    (pence: redan delad med 100), eller None. None när kursen saknas eller redan är i USD. Annars {currency,
    local, usd, fx, fx_src, fx_asof, stored_is_local}: kursen i sin valuta omräknad till USD med dagens växelkurs
    (fx_src 'live') eller den grova fasta tabellen ('tabell'); usd/fx None när ingen av dem har valutan. En dagskurs
    mer än 30 % från tabellens förkastas som datafel. stored_is_local = kursen i rutan ser ut att stå i den lokala
    valutan — då jämför säkerhetsmarginalen USD mot en annan valuta."""
    if not isinstance(ref, dict):
        return None
    raw = str(ref.get("currency") or "").strip()
    local = _pos(ref.get("price"))
    if not raw or raw.upper() == "USD" or local is None:
        return None
    fx, src, asof = fx_to_usd(raw), "tabell", None
    rate = _pos(live[0]) if isinstance(live, (tuple, list)) and live else None
    if rate is not None and (fx is None or abs(rate / fx - 1.0) <= FX_LIVE_SANE):
        fx, src, asof = rate, "live", (str(live[1])[:10] if len(live) > 1 and live[1] else None)
    usd = round(local * fx, 4) if fx else None
    return {"currency": "GBX" if _is_pence(raw) else raw.upper(), "local": local, "usd": usd, "fx": fx,
            "fx_src": src if fx else None, "fx_asof": asof,
            "stored_is_local": _looks_local(_num(price), local, fx)}


def _ref_sizes_ok(ref: Optional[dict]) -> bool:
    """Sifferuppdateringens MUSD-tal går att lita på: kursvalutan saknas eller finns i jobbets fasta tabell
    (USD ingår). För en valuta utanför den — även London i pence — har jobbet räknat om med 1,0."""
    if not isinstance(ref, dict):
        return False
    raw = str(ref.get("currency") or "").strip()
    return not raw or raw.upper() in _job_table()


def ref_fcf_musd(ref: Optional[dict]) -> Optional[float]:
    """Börsdatas FCF (12 mån) i MUSD ur sifferuppdateringens rad — samma enhet som kortets FCF-rutor (miljoner USD).
    None när talet inte säkert står i USD: kursvalutan utanför jobbets fasta tabell (som _ref_sizes_ok), eller
    rapportvalutan utanför den — jobbet räknar då om rapportens miljoner med 1,0, och talet vore lokal valuta."""
    if not _ref_sizes_ok(ref):
        return None
    rc = str(ref.get("report_currency") or "").strip().upper()
    if rc and rc not in _job_table():
        return None
    return _num(ref.get("fcf_musd"))


def _fcf_scan(row: dict, ref: Optional[dict] = None) -> list:
    """[(scenario, sort, text)] — se fcf_unit_warnings. sort: 'unit' (tusental/miljarder/yield), 'local' (miljoner
    NOK/SEK/DKK) eller 'size' (över 2× bolagets storlek)."""
    r = row or {}
    fv_in, csm = r.get("fv") or {}, r.get("csm") or {}
    rf = ref if _ref_sizes_ok(ref) else {}
    bd_fcf = _pos(ref_fcf_musd(ref))
    ev = _pos(rf.get("ev_musd")) or _pos(r.get("ev_musd"))     # Börsdatas EV först — rutan skrivs för hand (MSEK?)
    ev_eb = _pos(rf.get("ev_ebitda"))
    ebitda = _pos(rf.get("ev_musd")) / ev_eb if ev_eb and _pos(rf.get("ev_musd")) else None
    mcap, mcap_lbl = _pos(rf.get("mcap_musd")), "börsvärdet"
    shares = _pos(r.get("framtida_antal_aktier"))
    if mcap is None and shares is not None:             # inget betrott börsvärde (pence, Yahoo, ny rad): aktier × kurs i USD
        cur = price_currency(r.get("aktuell_kurs"), ref)
        p = (cur["usd"] or None) if cur is not None else _pos(r.get("aktuell_kurs"))
        if p:
            mcap, mcap_lbl = shares * p, "börsvärdet (aktier × kurs)"
    size, size_lbl = max(((v, lbl) for v, lbl in ((ev, "EV"), (mcap, mcap_lbl)) if v is not None),
                         default=(None, ""))
    ccy = str((ref or {}).get("currency") or "").strip() if isinstance(ref, dict) else ""
    fx_l = fx_to_usd(ccy) if ccy and not _is_pence(ccy) else None
    if fx_l is not None and max(fx_l, 1.0 / fx_l) < FCF_LOCAL_FX_MIN:
        fx_l = None                                     # EUR/CAD/AUD/PLN … går inte att skilja från ett annat scenario
    C = ccy.upper()
    out = []

    def _m(v):
        return f"{v:,.0f}".replace(",", " ")

    def _f(v):                                          # det inskrivna talet som det står: 12 · 12,5 · 3 550 000
        return f"{v:,.12g}".replace(",", " ").replace(".", ",")

    def _ref(v):                                        # en positiv referens visas aldrig som 0
        return _f(round(v, 2)) if v < 10 else _m(v)

    def big(v):
        return size is not None and v > FCF_SIZE_MAX * size

    def over_ebitda(v):                                 # enhetsfritt stöd: FCF i lokal valuta ligger ~10× över EBITDA
        return ebitda is None or v > FCF_LOCAL_EBITDA * ebitda
    too_big = f"över {FCF_SIZE_MAX:g}× {size_lbl} {_m(size) if size else ''} MUSD — står talet i miljoner USD (inte tusental)?"
    for s in FV_SCENARIOS:
        f = _num((fv_in.get(s) or {}).get("forward_fcf_musd") if isinstance(fv_in.get(s), dict) else None)
        if f is None or f <= 0:
            continue
        sc = csm.get(s) if isinstance(csm, dict) else None
        base = _pos(sc.get("fcf_musd")) if isinstance(sc, dict) else None
        src = f"CSM:s {s}-FCF"
        if base is not None and big(base):             # CSM i fel enhet duger inte som referens
            out.append((s, "size", f"{s}: {src} {_f(base)} MUSD är {too_big}"))
            base = None
        if base is not None and size is not None and base < FCF_EV_MIN * size:
            base = None                                 # break-even: en kvot mot nästan noll säger inget om enheten
        if base is None and bd_fcf is not None:
            base, src = bd_fcf, "Börsdatas FCF (12 mån)"
        if base is not None and (f < FCF_TOO_SMALL * base or f > FCF_TOO_LARGE * base):
            out.append((s, "unit", f"{s}: forward FCF {_f(f)} MUSD mot {src} {_ref(base)} MUSD — står talet i "
                                   f"miljoner USD?"))
        elif (base is not None and fx_l is not None and (size is None or base >= FCF_LOCAL_REF_MIN * size)
              and abs(f * fx_l / base - 1.0) <= FCF_LOCAL_BAND):
            out.append((s, "local", f"{s}: forward FCF {_f(f)} MUSD är {f / base:.1f}× {src} {_ref(base)} MUSD"
                        .replace(".", ",") + f" — omräknat från {C} stämmer det. Står talet i miljoner {C}?"))
        elif (s == ctl.BASE and fx_l is not None and size is not None
              and f >= FCF_LOCAL_YIELD * size and f * fx_l <= FCF_LOCAL_YIELD * size and over_ebitda(f)):
            out.append((s, "local", f"{s}: forward FCF {_f(f)} MUSD är {100 * f / size:.0f} % av {size_lbl} "
                                    f"{_m(size)} MUSD — omräknat från {C} {100 * f * fx_l / size:.0f} %. Står talet i "
                                    f"miljoner {C}?"))
        elif big(f) and fx_l is not None and not big(f * fx_l) and over_ebitda(f):
            out.append((s, "local", f"{s}: forward FCF {_f(f)} MUSD är över {FCF_SIZE_MAX:g}× {size_lbl} {_m(size)} "
                                    f"MUSD — omräknat från {C} {100 * f * fx_l / size:.0f} %. Står talet i miljoner "
                                    f"{C}?"))
        elif big(f):
            out.append((s, "size", f"{s}: forward FCF {_f(f)} MUSD är {too_big}"))
        elif base is None and s != ctl.BEAR and ev is not None and f < FCF_EV_MIN * ev:
            out.append((s, "unit", f"{s}: forward FCF {_f(f)} MUSD är under 0,2 % av EV {_m(ev)} MUSD — står talet "
                                   f"i miljoner USD?"))
        if (s == ctl.BASE and fx_l is not None and size is not None and not over_ebitda(f)
                and f >= FCF_LOCAL_YIELD * size and f * fx_l <= FCF_LOCAL_YIELD * size):
            # Under 1,5× EBITDA: ett riktigt billigt bolag — eller miljoner NOK hos en producent med låg konvertering
            # (norsk E&P: FCF 5–15 % av EBITDA). EBITDA kan inte skilja dem åt: ingen varningstext, men inget klick.
            out.append((s, "maybe_local", ""))
    return out


def fcf_unit_warnings(row: dict, ref: Optional[dict] = None) -> list:
    """Varningar när forward FCF ser ut att stå i fel enhet (MUSD = miljoner USD). Bara text — räkningen är orörd.

    Varje scenario prövas mot CSM:s FCF för samma scenario, annars Börsdatas FCF (12 mån): under 2 % eller över 50×
    referensen (tusental, miljarder, en yield i fel ruta). För ett bolag noterat i NOK/SEK/DKK (valutor minst 5× från
    USD) också miljoner i den valutan: inom ±30 % av en referens på minst 2 % av storleken omräknat, eller — när CSM
    står i samma enhet eller saknas — en Base-FCF över 40 % av storleken som omräknad blir en vanlig yield; när
    Börsdatas EV/EBITDA finns krävs också en FCF över 1,5× EBITDA (ett riktigt billigt bolag har hög yield men FCF
    under EBITDA). En FCF över 2× storleken som omräknad blir rimlig (och ligger över EBITDA) räknas också som
    lokal valuta, inte tusental. Bolagets storlek (Börsdatas EV före kortets handskrivna, eller börsvärde, det större; utan betrott
    börsvärde aktier × kurs i USD) är ett tak för båda
    talen: CSM:s FCF skrivs också för hand, så en referens över 2× storleken varnas själv och används inte, och en
    referens under 0,2 % av storleken (break-even) säger inget om enheten. Utan referens: över 2× storleken, eller
    under 0,2 % av EV — utom i Bear, som får ligga nära noll."""
    return [text for _, kind, text in _fcf_scan(row, ref) if kind != "maybe_local"]


def fcf_maybe_local(row: dict, ref: Optional[dict] = None) -> Optional[str]:
    """Valutan när Base-FCF:en kan stå i miljoner av noteringsvalutan men inte går att skilja från ett riktigt billigt
    bolag (över 40 % av storleken, under 1,5× EBITDA). Ingen varning om enheten — men ingen kurs i USD med ett
    klick heller: det ena fallet öppnar grinden med en ~10× för god marginal."""
    if any(kind == "maybe_local" for _, kind, _ in _fcf_scan(row, ref)):
        return str((ref or {}).get("currency") or "").strip().upper() or None
    return None


def fcf_in_local_currency(row: dict, ref: Optional[dict] = None) -> Optional[str]:
    """Valutan ('SEK') när forward FCF ser ut att stå i miljoner av bolagets noteringsvaluta — då hör en kurs i samma
    valuta ihop med den, och att bara byta kursen till USD gör raden fel åt det optimistiska hållet."""
    if any(kind == "local" for _, kind, _ in _fcf_scan(row, ref)):
        return str((ref or {}).get("currency") or "").strip().upper() or None
    return None


def gate_ok(row: dict, position_pct=None, strategy: str = "") -> bool:
    """Steg 5 mekaniskt: MoS >= 25 % OCH 'what must go right' ifyllt."""
    return not gate_gaps(row, position_pct, strategy)
