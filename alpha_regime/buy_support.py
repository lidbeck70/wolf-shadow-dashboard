"""
alpha_regime/buy_support.py
Köpstödet i en gummisnodd — EN regel som motorns +1 ACCUMULATE, panelens 'Gummisnoddar: …', kvotkortet och
Thors Råvara-flik läser. Ingen annan modul får ha en egen kopia av villkoren.

Guldgruvornas två kvoter (ägarens beslut A, 2026-11-03, efter det förregistrerade backtestet
docs/forskning/gruvor_marginal_spec.md, signal S3):

    metal_miners (GLD/GDX) och gold_gdxj (GLD/GDXJ) ger köpstöd när
      kvoten är RUBBER_BAND_STRETCHED (≥ 90:e percentilen, commodity_ratios._classify, oförändrad)
      OCH gruv-ETF:ns marginalmätare (miner_margin.compute: M = kurs / (guld − branschens AISC)) står
      på ≤ 30:e percentilen (MARGIN_MAX_PCT, specens förregistrerade S3-tröskel).

Det ersätter drivarens prisvillkor (det billiga benets eget prisläge) för just de två kvoterna. Saknas marginalen
(DATA_GAP: inget guldpris, ingen gruvkurs, marginal ≤ 50 USD/uns) blir det inget köpstöd. En inaktuell AISC-tabell
(stale) blockerar inte — kortet fortsätter att varna. Marginalmätaren ENSAM är ingen signal (H2 föll).

PGM-snoddarna gold_platinum (Gold/Platinum) och gold_palladium (Gold/Palladium) är BARA KONTEXT (ägarens beslut A,
2026-11-03, efter det förregistrerade backtestet docs/forskning/pgm_snoddar_spec.md): de visas (kort, Thors
Råvara-flik, temabordet, exponeringsväljaren) men ger aldrig köpstöd — evaluate svarar CONTEXT_ONLY vad status,
drivare och marginaler än säger, så motorns +1 ACCUMULATE och KÖP-STÖD i 'Gummisnoddar' räknar dem aldrig.
Backtestet: en sträckt kvot följdes av att metallen gick sämre än guld.

Alla andra kvoter (gold_silver, gold_oil, silver_miners, copper_gold, gdxj_gdx, silver_juniors, gdx_spy, egna)
behåller regeln från före 2026-11-03: sträckt kvot och drivaren DENOMINATOR_CHEAP/BOTH (motorn, ENGINE_DRIVERS).
Panelens sammanräkning släpper dessutom igenom UNKNOWN som förut (SUMMARY_DRIVERS) — två befintliga betydelser som
behålls exakt; vad som räknas som "sträckt" anger anroparen (motorn: status, panelen: märket 🟢 KÖP-STÖD).
"""
from __future__ import annotations

import logging
import math
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)

MARGIN_RULE: dict = {"metal_miners": "GDX", "gold_gdxj": "GDXJ"}   # kvotnyckel → gruv-ETF:n vars marginal avgör
MARGIN_MAX_PCT = 30.0     # S3: P_M ≤ 30 — förregistrerad i docs/forskning/gruvor_marginal_spec.md, inte optimerad
DECIDED = "2026-11-03"    # ägarens beslut (A) efter backtestet
SPEC = "docs/forskning/gruvor_marginal_spec.md"

STRETCHED = "RUBBER_BAND_STRETCHED"
ENGINE_DRIVERS = ("DENOMINATOR_CHEAP", "BOTH")              # motorns +1 för övriga kvoter (oförändrat)
SUMMARY_DRIVERS = ("DENOMINATOR_CHEAP", "BOTH", "UNKNOWN")  # 'Gummisnoddar'-räkningen för övriga kvoter (oförändrat)

# Bara kontext: kvotnyckel → beslutet (datum och specen med backtestet). Visas men ger aldrig köpstöd.
CONTEXT_DECIDED = "2026-11-03"    # ägarens beslut (A) efter PGM-backtestet
CONTEXT_SPEC = "docs/forskning/pgm_snoddar_spec.md"
CONTEXT_ONLY: dict = {
    "gold_platinum": {"decided": CONTEXT_DECIDED, "spec": CONTEXT_SPEC},
    "gold_palladium": {"decided": CONTEXT_DECIDED, "spec": CONTEXT_SPEC},
}

# Reason codes
CONTEXT = "CONTEXT_ONLY"                 # bara kontext (CONTEXT_ONLY): aldrig köpstöd, vad kvoten än säger
MARGIN_CHEAP = "MARGIN_CHEAP"            # guldkvot: sträckt och marginalmätaren ≤ 30 → köpstöd
MARGIN_NOT_CHEAP = "MARGIN_NOT_CHEAP"    # guldkvot: sträckt men marginalmätaren > 30
MARGIN_MISSING = "MARGIN_MISSING"        # guldkvot: sträckt men marginalen saknas (DATA_GAP)
DRIVER_CHEAP = "DRIVER_CHEAP"            # övrig kvot: sträckt och drivaren bekräftar → köpstöd
DRIVER_NOT_CHEAP = "DRIVER_NOT_CHEAP"    # övrig kvot: sträckt men drivaren bekräftar inte
NOT_STRETCHED = "NOT_STRETCHED"          # kvoten är inte fullt sträckt
NO_DATA = "NO_DATA"                      # kvoten saknas eller är DATA_GAP


class BuySupport(NamedTuple):
    """(supported, reason, details) — reason är en av koderna ovan; details bär talen texterna behöver."""
    supported: bool
    reason: str
    details: dict


def uses_margin(key) -> bool:
    """True för GLD/GDX och GLD/GDXJ, där marginalmätaren avgör i stället för drivaren."""
    return key in MARGIN_RULE


def context_only(key) -> bool:
    """True för Gold/Platinum och Gold/Palladium: bara kontext sedan 2026-11-03 — aldrig köpstöd."""
    return key in CONTEXT_ONLY


def margin_miners(keys) -> list:
    """GDX när metal_miners finns bland keys, GDXJ när gold_gdxj gör det (i MARGIN_RULE:s ordning)."""
    shown = keys or []
    return [miner for key, miner in MARGIN_RULE.items() if key in shown]


def margin_percentile(res) -> Optional[float]:
    """M:s percentil när mätarens resultat går att använda (status OK, ändlig percentil), annars None (saknas).
    Stale spelar ingen roll här."""
    if res is None or getattr(res, "status", "") != "OK":
        return None
    try:
        p = float(getattr(res, "percentile", None))
    except (TypeError, ValueError):
        return None
    return p if math.isfinite(p) else None


def evaluate(key, ratio, margins: Optional[dict] = None, *, stretched: Optional[bool] = None,
             drivers: tuple = ENGINE_DRIVERS) -> BuySupport:
    """Köpstöd för en kvot.

    key       kvotnyckeln; kvotens egen .key gäller när den finns (samma i panelen).
    margins   {gruv-ETF: MarginResult} — behövs bara för GLD/GDX och GLD/GDXJ; saknas den → inget köpstöd.
    stretched anroparens "fullt sträckt": None = ratio.status == RUBBER_BAND_STRETCHED (motorn och Thor);
              panelens sammanräkning och kort skickar märket 🟢 KÖP-STÖD (den lagrade percentilen), som förut.
    drivers   för övriga kvoter: drivarna som bekräftar (motorn ENGINE_DRIVERS, sammanräkningen SUMMARY_DRIVERS).

    Kvoterna i CONTEXT_ONLY (Gold/Platinum, Gold/Palladium) ger alltid (False, CONTEXT, …), vad status, drivare,
    stretched och margins än säger; details bär kvotens läge (status, percentil, drivare) för texterna.
    """
    key = getattr(ratio, "key", None) or key
    if context_only(key):
        det = {"key": key, "rule": "context", "stretched": False, **CONTEXT_ONLY[key]}
        if ratio is not None and getattr(ratio, "status", "DATA_GAP") != "DATA_GAP":
            det.update(stretched=(ratio.status == STRETCHED) if stretched is None else bool(stretched),
                       ratio_pct=getattr(ratio, "percentile", None), status=ratio.status,
                       driver=getattr(ratio, "driver", "UNKNOWN"))
        return BuySupport(False, CONTEXT, det)
    rule = "margin" if uses_margin(key) else "driver"
    det: dict = {"key": key, "rule": rule, "stretched": False}
    if rule == "margin":
        miner = MARGIN_RULE[key]
        res = (margins or {}).get(miner)
        pct = margin_percentile(res)
        det.update(miner=miner, margin_pct=pct, margin_max=MARGIN_MAX_PCT,
                   margin_error=None if pct is not None else (getattr(res, "error", None) or "marginalen saknas"),
                   stale=bool(getattr(res, "stale", False)))
    if ratio is None or getattr(ratio, "status", "DATA_GAP") == "DATA_GAP":
        return BuySupport(False, NO_DATA, det)

    is_stretched = (ratio.status == STRETCHED) if stretched is None else bool(stretched)
    det.update(stretched=is_stretched, ratio_pct=getattr(ratio, "percentile", None), status=ratio.status,
               driver=getattr(ratio, "driver", "UNKNOWN"))
    if not is_stretched:
        return BuySupport(False, NOT_STRETCHED, det)
    if rule == "margin":
        pct = det["margin_pct"]
        if pct is None:
            return BuySupport(False, MARGIN_MISSING, det)
        if pct <= MARGIN_MAX_PCT:
            return BuySupport(True, MARGIN_CHEAP, det)
        return BuySupport(False, MARGIN_NOT_CHEAP, det)
    if det["driver"] in drivers:
        return BuySupport(True, DRIVER_CHEAP, det)
    return BuySupport(False, DRIVER_NOT_CHEAP, det)


def compute_margins(keys, have: Optional[dict] = None) -> dict:
    """{gruv-ETF: MarginResult} för guldkvoterna bland keys. Återanvänder `have` (t.ex. RegimeResult.miner_margins)
    och räknar resten med miner_margin.compute — cachad 6 h; en misslyckad nedladdning är en DATA_GAP som inte
    cachas. {} när ingen guldkvot finns bland keys (då läses mätaren inte alls)."""
    miners = margin_miners(keys)
    have = have or {}
    out = {m: have[m] for m in miners if have.get(m) is not None}
    need = [m for m in miners if m not in out]
    if not need:
        return out
    try:
        from alpha_regime import miner_margin as mm
    except ImportError as exc:          # utan mätaren: marginalen saknas → inget köpstöd för guldkvoterna
        logger.debug("miner_margin: %s", exc)
        return out
    for miner in need:
        try:
            out[miner] = mm.compute(miner)
        except Exception as exc:
            logger.debug("miner_margin(%s): %s", miner, exc)
            out[miner] = mm.MarginResult(miner=miner, error="beräkningen misslyckades")
    return out


def pct_text(p: float, limit: float, at_most: bool) -> str:
    """En percentil som heltal — men med en decimal (decimalkomma) när avrundningen skulle korsa regelns gräns,
    så att en text aldrig motsäger utfallet: 30,4 mot '≤ 30' blir '30,4', inte '30'; 89,96 mot '≥ 90' blir
    '89,9', inte '90'. at_most=True för '≤ limit', False för '≥ limit'."""
    p = float(p)
    n = int(round(p))
    tenths = round(p * 10, 6)             # 30,1 · 10 = 301,00000000000006 får inte bli 30,2
    if at_most and n <= limit < p:
        return f"{math.ceil(tenths) / 10:.1f}".replace(".", ",")
    if not at_most and p < limit <= n:
        return f"{math.floor(tenths) / 10:.1f}".replace(".", ",")
    return str(n)
