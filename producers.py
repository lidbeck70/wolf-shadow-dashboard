"""
producers.py — Rick Rule + Royalty C (ur ravarurotation.xlsx).

De två granskningsarken som ligger mellan rotationen och köpet: när
rotationen sagt VAR kapitalet ska, säger de här VILKET bolag.

  Rick Rule     — Överlevarna. Guiden kallar arket "Producenter A" ("2 p i
                  Producenter A-arket"); här heter det efter sin upphovsman
                  som de andra granskningsarken gör. Dess fyra
                  granskningssteg — landrisk, kostnadsposition, ledning,
                  kapitaldisciplin — är exakt arkets fyra fält. Marginalen mot
                  kostnadskurvan plus tre disciplinfrågor: 0–5 poäng, ≥ 4
                  köpkandidat, 3 bevaka, under 3 passa.
  Royalty C     — rabatten mot egen historik, och om GEO per aktie växer.
                  En royalty som krymper per aktie är inte billig, den är
                  utspädd.

Formlerna är migrationsspecens (§6), som i sin tur är arkets:

  marginal   = (råvarupris − kostnad per enhet) / råvarupris
  poäng      = (marginal >= 40 % -> 2p, >= 25 % -> 1p) + jurisdiktion
               + kapitaldisciplin + insyn  (0/1 var)
  rabatt     = pnav_nu / pnav_botten − 1
  vs median  = pnav_nu / ev_ebitda_median-jämförelsen
  GEO-tillväxt = geo_aktie_nu / geo_aktie_3år − 1
  EV/NAV     = EV (MUSD) / NAV efter skatt (MUSD) — Rules värdering av
               producenter. Egen rad utanför poängen: poängen säger om
               bolaget är bra, EV/NAV om det är billigt.

Royalty C:s räknare föreslår (Använd skriver in, inget sparas av sig självt):
  GEO/aktie  = total GEO (koz) × 1 000 / (Börsdatas aktieantal, milj. × 1 000 000) — nu och för tre år sedan, som par
  P/NAV nu   = nattens kurs / NAV per aktie — bara i samma valuta
"""

from __future__ import annotations

import streamlit as st
from dataclasses import dataclass
from datetime import date
from typing import Optional

import csv_export
import lukacs
import storage
import storage_ui
from ui.components import confirm_delete, page_header
import controls as ctl
import controls_ui

try:
    from rotation import COMMODITIES as _ROT_COMMODITIES
    COMMODITIES = tuple(c.name for c in _ROT_COMMODITIES)
except Exception:                                   # pragma: no cover
    COMMODITIES = ("Guld", "Silver", "Uran", "Koppar")

_CACHE_KEY = "producers_data"
STORE = "producers"   # data/producers.json

TEXT, DIM = "#e8e4dc", "#8a8578"
GREEN, AMBER, RED, CYAN, GOLD = "#2d8a4e", "#d4943a", "#c44545", "#00E5FF", "#c9a84c"

PRODUCERS, ROYALTY = "producers", "royalty"

# Fliknamn. Lagringsnycklarna ovan står kvar — de ligger i Gist och ett byte
# där hade tappat sparade rader för att ändra en rubrik.
SHEET_RULE = "Rick Rule"
SHEET_ROYALTY = "Royalty C"
GUIDE_NAME_RULE = "Producenter A"     # guidens eget namn, behålls som spårbarhet

# ── Rick Rule — Överlevarna (arkets namn i guiden: Producenter A) ────────────────────────────────────────────────────────────
MARGIN_STRONG, MARGIN_OK = 40.0, 25.0      # procent
PROD_BUY_MIN, PROD_WATCH_MIN = 4, 3
PROD_MAX_SCORE = 5

# Den döende tillgången — Rules egen strykregel. Guiden, om screeningen:
# "stryk direkt ... bolag där låg multipel förklaras av döende tillgång
# (gruvlivslängd < 5 år, R/P < 8 år)". Och i mini-exemplet: samma 5/5-bolag
# med fyra års gruvlivslängd är en passa, för priset är lågt av ett skäl.
LIFE_MIN_YEARS = 5.0        # gruvlivslängd
RP_MIN_YEARS = 8.0          # reserver/produktion, olja och gas
DYING_ASSET = "Döende tillgång — låg multipel av ett skäl"

# EV/NAV — värderingsraden bredvid poängen (inte i den)
EV_NAV_CHEAP, EV_NAV_FAIR = 0.7, 1.0      # under 0,7× billigt, 0,7–1,0× rimligt, över dyrt
NAV_PRICE_GAP_PCT = 20.0                  # NAV-pris mer än så från spot → varning
NAV_CHEAP, NAV_FAIR, NAV_RICH = "Billigt mot NAV", "Rimligt mot NAV", "Dyrt mot NAV"

P_BUY = "Köpkandidat"
P_WATCH = "Bevaka"
P_PASS = "Passa"
PROD_COLOR = {P_BUY: GREEN, P_WATCH: AMBER, P_PASS: DIM}
NAV_COLOR = {NAV_CHEAP: GREEN, NAV_FAIR: AMBER, NAV_RICH: RED}

CHECKS = (
    ("jurisdiktion", "Jurisdiktion OK",
     "Fraser-rankingens topphalva. Nedre halvan är passa oavsett siffror."),
    ("kapitaldisciplin", "Kapitaldisciplin",
     "Utdelning eller återköp pågår, ingen tillväxtcapex utanför kärnan."),
    ("insyn", "Insynsägande",
     "Ledning och styrelse äger själva, och köper snarare än säljer."),
)

# ── Royalty C ────────────────────────────────────────────────────────────────
DISCOUNT_MAX = 15.0        # högst 15 % över egen botten för köpläge
ROYALTY_LEVELS = (1, 2, 3)

R_BUY = "KÖPLÄGE"
R_NEAR = "Nära historisk botten — kolla GEO"
R_GEO_WARN = "Varning: GEO/aktie krymper"
R_NEUTRAL = "Neutral"
ROYALTY_COLOR = {R_BUY: GREEN, R_NEAR: AMBER, R_GEO_WARN: RED, R_NEUTRAL: DIM}

LEVEL_NOTE = {
    1: "Nivå 1 — de stora. Säljs i princip aldrig.",
    2: "Nivå 2 — mellanskiktet. Säljs om GEO per aktie stagnerar två år.",
    3: "Nivå 3 — de små. Säljs vid uppköp eller när tesen är klar.",
}


@dataclass(frozen=True)
class Verdict:
    label: str
    why: str


# ── Rena beräkningar ─────────────────────────────────────────────────────────
def _num(value, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def margin_pct(price, unit_cost) -> Optional[float]:
    """(råvarupris − kostnad per enhet) / råvarupris, i procent. Ett negativt pris eller en negativ kostnad (rutorna
    har golvet 0) är en lucka, som i rutan — aldrig en marginal över 100 % som ger full poäng."""
    p, c = _nonneg(price), _nonneg(unit_cost)
    if p is None or c is None or p == 0:
        return None
    return round((p - c) / p * 100, 6)


def margin_points(margin) -> int:
    """≥ 40 % marginal = 2p, ≥ 25 % = 1p, annars 0."""
    m = _num(margin)
    if m is None:
        return 0
    if m >= MARGIN_STRONG:
        return 2
    return 1 if m >= MARGIN_OK else 0


# Terminer vars enhet inte är den kortens kostnad brukar stå i (extraktorn: USD/oz, USD/lb, USD/boe). NG=F står i
# USD/MMBtu och en boe är ungefär 5,8 MMBtu — inom NAV_PRICE_SCALE_MAX. Där måste kortets sparade pris stå inom så
# många gånger från terminen; utan sparat pris räknas ingen marginal.
SPOT_SCALE_TIGHT: dict = {"NG=F": 3.0}


def spot_margin_line(night: Optional[dict], commodity, unit_cost, saved_price) -> str:
    """Raden vid Marginal: '🤖 Med dagens pris 4 012 USD/oz (2026-11-06): marginal 64.2 % (2 p) — poängen räknas på
    det sparade priset.' Dagens pris är nattens terminspris (sheets_refresh: commodity_price på producentraden), bara
    när det gäller kortets råvara (commodity_prices.night_fits); marginalen är margin_pct som kortets egen. Bara
    visning, ingen Använd. Tom sträng utan pris eller kostnad, och när det sparade priset redan är dagens.

    Enhetsvakten är CSM-radens (controls_ui.spot_context): står kortets sparade pris (eller kostnaden, utan sparat
    pris) mer än NAV_PRICE_SCALE_MAX× från terminen — koppar i USD/t mot HG=F i USD/lb — räknas ingen marginal, bara
    en kort notis. Gas (SPOT_SCALE_TIGHT) kräver ett sparat pris inom 3× från terminen."""
    r = night if isinstance(night, dict) else {}
    spot = _nonneg(r.get("commodity_price"))
    m = margin_pct(spot, unit_cost)
    if not spot or m is None:
        return ""
    saved = _nonneg(saved_price)
    if saved is not None and abs(saved - spot) < 1e-6:
        return ""
    try:
        import commodity_prices
        from controls_ui import NAV_PRICE_SCALE_MAX, _price_txt
        if not commodity_prices.night_fits(r, commodity):
            return ""
        tight = SPOT_SCALE_TIGHT.get(commodity_prices.ticker_for(commodity)[0])
    except Exception:
        return ""
    unit = str(r.get("commodity_unit") or "").strip()
    spot_txt = f"{_price_txt(spot)}{' ' + unit if unit else ''}"
    ref = saved or (None if tight else _nonneg(unit_cost))
    if tight and not ref:
        return ""
    limit = tight or NAV_PRICE_SCALE_MAX
    if ref and (ref / spot > limit or spot / ref > limit):
        return f"🤖 Dagens pris {spot_txt} står nog i en annan enhet än kortets — ingen marginal räknad."
    asof = str(r.get("commodity_asof") or "")[:10]
    return (f"🤖 Med dagens pris {spot_txt}{f' ({asof})' if asof else ''}: "
            f"marginal {m:.1f} % ({margin_points(m)} p) — poängen räknas på det sparade priset.")


def _years(value) -> Optional[float]:
    """Ett årsfält, eller None om det inte är ifyllt.

    Noll är inte ett svar. Fälten ritas med number_input(min_value=0.0), som
    startar på 0,0 och inte kan lämnas tomt — så en nolla betyder att frågan
    aldrig ställdes. En gruva med noll års livslängd finns inte; en sådan är
    stängd. Läser man nollan bokstavligt blir varje guldproducent en döende
    tillgång så fort någon öppnat kortet, eftersom R/P bara gäller olja och
    gas och därför alltid står kvar på noll.
    """
    v = _num(value)
    return v if v is not None and v > 0 else None


def asset_dying(row: dict) -> Optional[bool]:
    """Är den låga multipeln förklarad av att tillgången tar slut?

    None när varken gruvlivslängd eller R/P är ifylld — obesvarad fråga är
    inte ett friskintyg, men den ska heller inte döma ut bolaget här. Den
    fångas i stället av köpgrindens luck-regel om positionen är stor nog.

    Ett bolag behöver bara ett av måtten: gruvor har gruvlivslängd, olja och
    gas har R/P. Att det andra står tomt är normalfallet, inte en brist.
    """
    life = _years(row.get("mine_life"))
    rp = _years(row.get("rp_ratio"))
    if life is None and rp is None:
        return None
    if life is not None and life < LIFE_MIN_YEARS:
        return True
    if rp is not None and rp < RP_MIN_YEARS:
        return True
    return False


def producer_score(row: dict) -> Optional[int]:
    """0–5: marginalpoängen plus de tre disciplinfrågorna.

    None när marginalen inte går att räkna — utan den är poängen bara de tre
    kryssrutorna, och det är inte modellen.
    """
    m = margin_pct(row.get("price"), row.get("unit_cost"))
    if m is None:
        return None
    return margin_points(m) + sum(1 for k, _l, _h in CHECKS if row.get(k))


def producer_verdict(score: Optional[int],
                     dying: Optional[bool] = None) -> Optional[Verdict]:
    """Bedömningen. En döende tillgång överprövar poängen helt.

    Guidens mini-exempel är just den fällan: EV/EBITDA 3,8, nettokassa,
    soliditet 64 %, AISC i 25:e percentilen, Nevada, insynsägande, 6 %
    direktavkastning ger 5/5 — men med fyra års gruvlivslängd är det ändå en
    passa. Poängen mäter hur bra bolaget är på att vänta; den mäter inte om
    det finns något kvar att vänta på.
    """
    if score is None:
        return None
    if dying:
        return Verdict(P_PASS, f"{DYING_ASSET}. Poängen ({score}/"
                               f"{PROD_MAX_SCORE}) spelar ingen roll — "
                               f"gruvan tar slut före cykeln vänder.")
    if score >= PROD_BUY_MIN:
        return Verdict(P_BUY, "Marginal och disciplin på plats.")
    if score >= PROD_WATCH_MIN:
        return Verdict(P_WATCH, "Nästan — en faktor saknas.")
    return Verdict(P_PASS, "För tunn marginal eller för lös disciplin.")


def ev_nav(ev, nav) -> Optional[float]:
    """EV / NAV efter skatt. None när något saknas, EV är negativt (rutan har golvet 0) eller NAV ≤ 0."""
    e, n = _nonneg(ev), _num(nav)
    if e is None or n is None or n <= 0:
        return None
    return round(e / n, 4)


def nav_price_gap(nav_price, spot) -> Optional[float]:
    """Hur många procent spot ligger över (+) eller under (−) priset NAV:n är räknad på."""
    p, s_ = _num(nav_price), _num(spot)
    if not p or not s_ or p <= 0 or s_ <= 0:
        return None
    return round((s_ / p - 1) * 100, 1)


def ev_nav_verdict(row: dict) -> Optional[Verdict]:
    """Billigt/rimligt/dyrt mot NAV, med varning när NAV:n är räknad på ett
    metallpris långt från dagens. None utan EV och NAV."""
    r = ev_nav(row.get("ev_musd"), row.get("nav_musd"))
    if r is None:
        return None
    label = NAV_CHEAP if r < EV_NAV_CHEAP else NAV_FAIR if r <= EV_NAV_FAIR else NAV_RICH
    why = (f"EV {_num(row.get('ev_musd')):,.0f} MUSD / NAV {_num(row.get('nav_musd')):,.0f} MUSD = {r:.2f}×. "
           f"Under {EV_NAV_CHEAP:g}× billigt, {EV_NAV_CHEAP:g}–{EV_NAV_FAIR:g}× rimligt, över dyrt.")
    gap = nav_price_gap(row.get("nav_price"), row.get("price"))
    if gap is not None and abs(gap) > NAV_PRICE_GAP_PCT:
        why += (f" OBS: NAV räknad på {_num(row.get('nav_price')):,.0f}, spot {_num(row.get('price')):,.0f} "
                f"({gap:+.0f} %) — NAV:n {'underskattar' if gap > 0 else 'överskattar'} värdet i dag.")
    elif _num(row.get("nav_price")) is None:
        why += " Ange metallpriset NAV:n är räknad på — annars går den inte att jämföra med spot."
    return Verdict(label, why)


def discount_vs_bottom(now, bottom) -> Optional[float]:
    """Hur många procent över sin egen P/NAV-botten bolaget handlas. Negativt P/NAV = lucka (rutan har golvet 0)."""
    n, b = _nonneg(now), _nonneg(bottom)
    if n is None or b is None or b == 0:
        return None
    return round((n / b - 1) * 100, 6)


def vs_median(now, median) -> Optional[float]:
    """EV/EBITDA mot egen median, i procent. Negativt = billigare än vanligt. En negativ multipel (negativt EBITDA)
    är en lucka, som i rutan — inte ett 'billigare än vanligt'."""
    n, m = _nonneg(now), _nonneg(median)
    if n is None or m is None or m == 0:
        return None
    return round((n / m - 1) * 100, 6)


def geo_growth(now, three_years_ago) -> Optional[float]:
    """GEO per aktie idag mot för tre år sedan, i procent. Negativ GEO = lucka (rutan har golvet 0)."""
    n, o = _nonneg(now), _nonneg(three_years_ago)
    if n is None or o is None or o == 0:
        return None
    return round((n / o - 1) * 100, 6)


def _sv(v: float, dec: int = 1) -> str:
    return f"{v:,.{dec}f}".replace(",", " ").replace(".", ",")


def dilution_line(fresh: Optional[dict]) -> str:
    """Utspädningen under GEO ur nattens royalty-rad (producers:<id>): antalet aktier nu och för tre år sedan
    (shares_now_m, shares_3y_ago_m, året ur shares_year). Bara visning. Tom sträng när något av talen saknas."""
    f = fresh or {}
    now, old = _num(f.get("shares_now_m")), _num(f.get("shares_3y_ago_m"))
    if now is None or old is None or now <= 0 or old <= 0:
        return ""
    pct = (now / old - 1) * 100
    yr = _num(f.get("shares_year"))
    years = f" ({int(yr)} mot {int(yr) - 3})" if yr else ""
    asof = str(f.get("asof") or "")[:10]
    return (f"Utspädning: {_sv(now, 2)} milj. aktier mot {_sv(old, 2)} milj. för tre år sedan{years} = "
            f"{'+' if pct >= 0 else '−'}{_sv(abs(pct))} %. Börsdata{' ' + asof if asof else ''}.")


def _g(v: float) -> str:
    """Tal utan onödiga decimaler, med decimalkomma: 4 · 1,4 · 12,5."""
    return f"{v:g}".replace(".", ",").replace("-", "−")


def position_cap_line(row: dict) -> str:
    """Raden under 'Tänkt position': Rule-producentens hårda tak ur allokeraren (allocator.rule_for_strategy('rule'))
    och halveringen vid skuld (lukacs.max_position_pct) — samma funktioner som scorecardet. Bara visning: taket är en
    gräns, inget förslag på storlek. Tom sträng utan regel."""
    try:
        import allocator
        rule = allocator.rule_for_strategy("rule")
    except Exception:                                   # pragma: no cover
        return ""
    if rule is None:
        return ""
    cap = rule.hard_cap
    txt = f"📏 Tak {_g(cap)} % av totalen"
    eff = lukacs.max_position_pct(row.get("nd_ebitda"), cap, row.get("ar_till_lag_skuld"))
    if eff is not None and eff < cap:
        txt += f" ({_g(eff)} % med skuld/EBITDA {_g(_num(row.get('nd_ebitda')))})"
    return txt + " — en gräns, inte ett förslag på storlek."


# ── Royalty C: räknarna för P/NAV och GEO per aktie (bara förslag) ───────────
# Dina långsamma tal (NAV per aktie, total GEO) skrivs i tillfälliga rutor som inte sparas; Börsdatas snabba tal
# (kurs, aktieantal) kommer ur nattens rad. Förslaget skrivs in först med Använd. Signalen och nattlarmet läser
# som förut de sparade fälten. Gränserna nedan är enhetsvakter, inga regler: de största royaltybolagen har under
# 1 000 koz GEO per år och Börsdatas aktieantal står i miljoner, så ett tal utanför står nästan säkert i fel enhet
# (uns i stället för koz, styck i stället för miljoner). Då ges inget förslag — ett 1000×-fel får aldrig vända
# GEO-signalen eller ±15 %-grinden. De små royaltybolagen har bara några tusen uns, så uns i koz-rutorna syns först
# per aktie: inget royaltybolag har över GEO_PS_MAX uns per aktie, och inget handlas under sin intäkt per aktie
# räknad på ett lågt guldpris (GEO_PS_PRICE) — ett uns/koz-fel ger 20–50× kursen.
KOZ_MAX = 1_500.0           # koz GEO per år
GEO_PS_MAX = 0.02           # uns GEO per aktie (Royal Gold, högst per aktie, har cirka 0,005)
GEO_PS_PRICE = 1_000.0      # USD per uns, medvetet lågt: GEO per aktie × det över kursen = talen står i uns
SHARES_M_MAX = 100_000.0    # miljoner aktier
UNIT_JUMP = 100.0           # två tal i samma enhet för samma bolag skiljer sig inte 100×
PNAV_SANE = (0.05, 20.0)    # P/NAV utanför = NAV per aktie står i fel enhet (t.ex. total NAV i miljoner)
NAV_BASIS_NOTE = ("Samma NAV-bas som botten: ta NAV per aktie från samma källa och på samma metallpris som när "
                  "botten sattes — annars jämförs olika saker och ±15 %-gränsen flyttas.")


@dataclass(frozen=True)
class Calc:
    """En räknares förslag: fälten med sina värden (tomt = inget förslag), raden med talen bakom förslaget (eller
    varför inget föreslås) och en varning som visas bredvid."""
    values: dict
    text: str = ""
    note: str = ""


def _pct(v: float) -> str:
    """+9,3 % · −4,0 %"""
    return f"{'+' if v >= 0 else '−'}{_sv(abs(v))} %"


def _px(v: float) -> str:
    """Kurs med decimalkomma: två decimaler, fyra under 1."""
    return _sv(v, 2 if v >= 1 else 4)


def geo_per_share(koz, shares_m) -> Optional[float]:
    """GEO per aktie i kortets enhet (uns per aktie) ur total GEO i koz och aktieantalet i miljoner:
    koz × 1 000 / (miljoner × 1 000 000). Sex värdesiffror. None utan båda talen."""
    k, s = _nonneg(koz), _nonneg(shares_m)
    if not k or not s:
        return None
    return float(f"{k * 1000.0 / (s * 1e6):.6g}")


def shares_note(fresh: Optional[dict]) -> str:
    """Raden i GEO-räknaren: Börsdatas aktieantal nu och för tre år sedan med årsrapportens år, och att GEO ska tas
    för samma år. Tom sträng när något av talen saknas — då kan räknaren inte räkna."""
    f = fresh or {}
    now, old = _num(f.get("shares_now_m")), _num(f.get("shares_3y_ago_m"))
    if now is None or old is None or now <= 0 or old <= 0:
        return ""
    yr = _num(f.get("shares_year"))
    if yr:
        what = f"{_sv(now, 2)} milj. ({int(yr)}) och {_sv(old, 2)} milj. ({int(yr) - 3})"
    else:
        what = f"{_sv(now, 2)} milj. nu och {_sv(old, 2)} milj. för tre år sedan"
    return (f"Börsdatas aktieantal: {what}, från årsrapporterna (basic vid årsskiftet, inte vägt). Ta GEO för "
            f"samma år — GEO-året kan skilja sig från aktieåret.")


def geo_pair_suggestion(koz_now, koz_3y, fresh: Optional[dict], saved_now=None) -> Calc:
    """GEO per aktie nu och för tre år sedan ur total GEO (koz) och nattens aktieantal (shares_now_m,
    shares_3y_ago_m): bara som par, så att tillväxten alltid räknas på samma enhet och samma sorts aktieantal.
    Inget förslag när ett tal ser ut att stå i fel enhet (KOZ_MAX, SHARES_M_MAX, UNIT_JUMP, och per aktie
    GEO_PS_MAX och GEO_PS_PRICE mot nattens kurs). saved_now = det sparade GEO/aktie nu: skiljer förslaget sig mer
    än UNIT_JUMP× från det blir det en varning."""
    kn, k3 = _nonneg(koz_now), _nonneg(koz_3y)
    if not kn and not k3:
        return Calc({})
    if not kn or not k3:
        return Calc({}, "Fyll i GEO både nu och för tre år sedan — förslaget ges bara som par.")
    f = fresh or {}
    sn, s3 = _num(f.get("shares_now_m")), _num(f.get("shares_3y_ago_m"))
    if sn is None or s3 is None or sn <= 0 or s3 <= 0:
        return Calc({}, "Börsdata saknar aktieantalet nu eller för tre år sedan — ingen GEO per aktie.")
    if max(kn, k3) > KOZ_MAX:
        return Calc({}, f"{_g(max(kn, k3))} koz är mer än något royaltybolag har — står talet i uns? "
                        f"Skriv koz (tusen uns).")
    if max(kn, k3) / min(kn, k3) > UNIT_JUMP:
        return Calc({}, f"GEO nu och för tre år sedan skiljer sig mer än {UNIT_JUMP:g}× — står båda i koz?")
    if max(sn, s3) > SHARES_M_MAX or max(sn, s3) / min(sn, s3) > UNIT_JUMP:
        return Calc({}, "Börsdatas aktieantal ser ut att stå i styck, inte miljoner — ingen GEO per aktie.")
    gn, g3 = geo_per_share(kn, sn), geo_per_share(k3, s3)
    hi, price = max(gn, g3), _nonneg(f.get("price"))
    if price and hi * GEO_PS_PRICE > price:
        ccy = str(f.get("currency") or "").strip().upper()
        return Calc({}, f"GEO/aktie {_sv(hi, 6)} ger {_px(hi * GEO_PS_PRICE)} i intäkt per aktie redan vid "
                        f"{_sv(GEO_PS_PRICE, 0)} USD/uns — mer än kursen {_px(price)}{' ' + ccy if ccy else ''}. "
                        f"Står talen i uns? Skriv koz (tusen uns).")
    if hi > GEO_PS_MAX:
        return Calc({}, f"GEO/aktie {_sv(hi, 6)} är mer än något royaltybolag har — står talen i uns? "
                        f"Skriv koz (tusen uns).")
    growth = geo_growth(gn, g3)
    yr = _num(f.get("shares_year"))
    y_now, y_old = (f" {int(yr)}", f" {int(yr) - 3}") if yr else ("", "")
    text = (f"🤖 Räknat med Börsdatas aktieantal: GEO/aktie nu {_sv(gn, 6)} ({_g(kn)} koz / {_sv(sn, 2)} milj. "
            f"aktier{y_now}) och för 3 år sedan {_sv(g3, 6)} ({_g(k3)} koz / {_sv(s3, 2)} milj.{y_old})"
            + (f" = {_pct(growth)}" if growth is not None else ""))
    notes = []
    if max(kn, k3) < 1:
        notes.append("Under 1 koz — står talen i Moz? Skriv koz (tusen uns).")
    saved = _nonneg(saved_now)
    if saved and (gn / saved > UNIT_JUMP or saved / gn > UNIT_JUMP):
        notes.append(f"GEO/aktie nu skiljer sig mer än {UNIT_JUMP:g}× från det sparade ({_sv(saved, 6)}) — "
                     f"kontrollera koz mot uns innan du använder.")
    return Calc({"geo_now": gn, "geo_3y": g3}, text, " ".join(notes))


def _gate_pct(disc: float) -> str:
    """'Mot botten' i P/NAV-förslaget: en decimal, men fler när avrundningen skulle hamna på andra sidan köplägets
    gräns eller 0 (scorecard._edge, kantregeln) — +15,04 % står inte som '+15,0 %' bredvid 'högst 15 %'."""
    import scorecard                                    # här, inte överst: scorecard läser producers
    return scorecard._edge(disc, 1, [(DISCOUNT_MAX, "<="), (0.0, "<")], prefix="+" if disc >= 0 else "") + " %"


def pnav_suggestion(fresh: Optional[dict], navps, nav_ccy, bottom=None) -> Calc:
    """P/NAV nu = nattens kurs / NAV per aktie, sex värdesiffror (som GEO per aktie) — inte rutans två decimaler,
    som kan flytta talet över ±15 %-grinden. Står de sex siffrorna på andra sidan grinden än kvoten skrivs kvoten.
    Bara när kursens valuta är NAV:ns: samma bolag har kurs i USD på NYSE och CAD på TSX (cirka 27 % isär), så en
    okänd eller annan valuta ger inget förslag. Ett P/NAV utanför PNAV_SANE tyder på fel enhet (total NAV i stället
    för per aktie) och ger inget förslag. bottom = sparad botten: förslaget visar då var det hamnar mot grinden."""
    n = _nonneg(navps)
    if not n:
        return Calc({})
    f = fresh or {}
    price = _nonneg(f.get("price"))
    if not price:
        return Calc({}, "Ingen kurs i nattens rad — P/NAV går inte att räkna här.")
    pccy = str(f.get("currency") or "").strip().upper()
    nccy = str(nav_ccy or "").strip().upper()
    if not pccy:
        return Calc({}, "Kursens valuta är okänd (kursen kommer inte från Börsdata) — inget förslag.")
    if not nccy:
        return Calc({}, f"Välj NAV:ns valuta — kursen är i {pccy}.")
    if nccy != pccy:
        return Calc({}, f"Kursen är i {pccy}, NAV per aktie i {nccy} — inget förslag. Samma bolag har kurs i USD "
                        f"på NYSE och CAD på TSX; ta NAV per aktie i kursens valuta.")
    raw = price / n
    if not PNAV_SANE[0] <= raw <= PNAV_SANE[1]:
        return Calc({}, f"Kurs {_px(price)} / NAV per aktie {_g(n)} = {_sv(raw, 3)} — orimligt. Står NAV per aktie "
                        f"per aktie (inte total NAV i miljoner)?")
    v = float(f"{raw:.6g}")
    disc, exact = discount_vs_bottom(v, bottom), discount_vs_bottom(raw, bottom)
    if disc is not None and (disc <= DISCOUNT_MAX) != (exact <= DISCOUNT_MAX):
        v, disc = raw, exact                            # avrundningen får aldrig flytta grinden
    asof = str(f.get("asof") or "")[:10]
    src = "Börsdata" if f.get("source") in (None, "borsdata") else "nattens kurs"
    text = (f"🤖 P/NAV nu = kurs {_px(price)} {pccy} ({src}{' ' + asof if asof else ''}) / NAV per aktie {_g(n)} "
            f"{nccy} = {_sv(v, 2)}")
    if disc is not None:
        text += f" — mot botten {_sv(_num(bottom), 2)}: {_gate_pct(disc)}"
    return Calc({"pnav_now": v}, text)


def new_bottom(pnav_now, bottom) -> Optional[float]:
    """'Ny egen botten?': P/NAV nu när det ligger under den sparade botten, annars None — också utan sparad botten,
    för dagens P/NAV är ingen historisk botten bara för att ingen är ifylld."""
    n, b = _nonneg(pnav_now), _nonneg(bottom)
    if not n or not b or n >= b:
        return None
    return n


def royalty_signal(row: dict) -> Verdict:
    """Signalen, i specens ordning.

    Ordningen betyder något: en krympande GEO per aktie slår ut köpläget även
    när priset ser billigt ut, för det är just utspädningen som gjorde det
    billigt.
    """
    disc = discount_vs_bottom(row.get("pnav_now"), row.get("pnav_bottom"))
    med = vs_median(row.get("ev_now"), row.get("ev_median"))
    geo = geo_growth(row.get("geo_now"), row.get("geo_3y"))

    cheap = disc is not None and disc <= DISCOUNT_MAX
    if cheap and med is not None and med <= 0 and geo is not None and geo > 0:
        return Verdict(R_BUY, "Nära egen botten, under egen median, och GEO "
                              "per aktie växer.")
    if geo is not None and geo < 0:
        return Verdict(R_GEO_WARN, "GEO per aktie krymper — utspädningen äter "
                                   "tillväxten. Billigt av fel skäl.")
    if cheap:
        return Verdict(R_NEAR, "Nära historisk botten. Kolla GEO per aktie "
                               "innan du köper.")
    return Verdict(R_NEUTRAL, "Varken billig mot egen historik eller varnande.")


def ranked_producers(rows: list) -> list:
    out = []
    for r in rows or []:
        sc = producer_score(r)
        dying = asset_dying(r)
        out.append({"row": r, "score": sc,
                    "verdict": producer_verdict(sc, dying), "dying": dying,
                    "margin": margin_pct(r.get("price"), r.get("unit_cost")),
                    "ev_nav": ev_nav(r.get("ev_musd"), r.get("nav_musd"))})
    out.sort(key=lambda x: (-(x["score"] if x["score"] is not None else -1),
                            (x["row"].get("ticker") or "")))
    return out


def ranked_royalty(rows: list) -> list:
    order = {R_BUY: 0, R_NEAR: 1, R_NEUTRAL: 2, R_GEO_WARN: 3}
    out = []
    for r in rows or []:
        v = royalty_signal(r)
        out.append({"row": r, "verdict": v,
                    "discount": discount_vs_bottom(r.get("pnav_now"),
                                                   r.get("pnav_bottom")),
                    "median": vs_median(r.get("ev_now"), r.get("ev_median")),
                    "geo": geo_growth(r.get("geo_now"), r.get("geo_3y"))})
    out.sort(key=lambda x: (order.get(x["verdict"].label, 9),
                            (x["row"].get("ticker") or "")))
    return out


# ── Lagring ──────────────────────────────────────────────────────────────────
def _today() -> str:
    return date.today().isoformat()


def _uid() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


def _default() -> dict:
    return {PRODUCERS: [], ROYALTY: []}


def _load() -> dict:
    """Laddas EN gång per session; därefter äger sessionen sanningen."""
    data = storage.session_load(STORE, _default(),
                                legacy_file="producers_data.json")
    # saknade nycklar fylls i utan att det räknas som en osparad ändring
    return storage.session_normalize(STORE, data, lambda d: storage.fill_defaults(d, _default()))


def _save(data: dict) -> None:
    """Skriver till sessionen. Persistensen sker via 💾 Spara.

    Medvetet ingen nätverkstrafik här: en commit per
    tangenttryckning skulle slå i GitHubs rate limits, och en
    tyst misslyckad sådan var precis den bugg det här ersätter.
    """
    st.session_state[STORE] = data


def _extract_section(data: dict, row: dict, sheet: str, widget_keys: dict) -> None:
    """Copilot-extraktion ur presentationen (AISC, gruvlivslängd, R/P; GEO per
    aktie) — som förslag med sida och citat. Disciplinfrågorna kryssar du."""
    try:
        import extract_ui
    except Exception:
        return
    extract_ui.render_extractor(sheet, row, widget_keys=widget_keys,
                                on_apply=lambda: _save(data))


def _nonneg(value) -> Optional[float]:
    """Värdet för en ruta med golvet 0: ett negativt tal (Börsdatas EV/EBITDA vid negativt EBITDA) visas som tomt —
    en lucka — i stället för att fälla arket."""
    v = _num(value)
    return None if v is not None and v < 0 else v


def _write_changed(row: dict, pairs) -> bool:
    """Skriv bara de fält vars ruta du ändrat: (fält, rutans värde, värdet rutan visades med). Ett sparat negativt
    tal som rutan visar som tomt skrivs inte över när arket bara öppnas — inget blir osparat av att titta."""
    hit = False
    for field, new, shown in pairs:
        if storage.differs(new, shown):
            row[field] = new
            hit = True
    return hit


def _neg_note(row: dict, labels: dict) -> None:
    """En rad om sparade negativa tal som rutorna (golvet 0) inte kan visa — de står kvar tills du skriver om dem."""
    neg = [f"{lbl} {_num(row.get(k)):g}".replace(".", ",") for k, lbl in labels.items()
           if _num(row.get(k)) is not None and _num(row.get(k)) < 0]
    if neg:
        st.caption("Sparat men negativt — visas inte i rutan (golvet 0): " + " · ".join(neg)
                   + ". Skriv in rätt tal.")


def _suggest(*args, **kw) -> None:
    """Börsdata-förslag bredvid ett fält (sheets_refresh.py). Tyst utan blob."""
    try:
        import refresh_ui
        refresh_ui.suggest(*args, **kw)
    except Exception:
        pass


def _durrett(row: dict):
    """Durrett-bolaget för kortets ticker (crosssheet.durrett_company). None utan Durrett-rad eller vid fel."""
    try:
        import crosssheet
        return crosssheet.durrett_company(row.get("ticker"))
    except Exception:
        return None


def _durrett_offer(data: dict, row: dict, field: str, make, key: str, widget_key: str, on_apply=None) -> None:
    """Ett förslag ur Durrett-arket med Använd — samma knapp som arkens andra förslag (refresh_ui.offer). make(
    crosssheet) ger (värde, text) eller None. Att rita skriver ingenting; klicket skriver fältet och sparar i
    sessionen. Tyst vid fel — ett förslag får aldrig fälla kortet."""
    try:
        import crosssheet
        import refresh_ui
        hit = make(crosssheet)
        if hit:
            refresh_ui.offer(row, field, hit[0], field, key=key, widget_key=widget_key,
                             on_apply=on_apply or (lambda: _save(data)), text=hit[1])
    except Exception:
        pass


def _apply_nav_pair(data: dict, row: dict, price, source: str) -> None:
    """on_apply för NAV-paret: metallpriset i NAV och NAV-källan följer med NAV:n (rutorna via sina nycklar). Paret
    skrivs ihop: utan pris töms kortets gamla Metallpris i NAV — det hörde till NAV:n som byttes ut."""
    rid = row["id"]
    if price is not None:
        row["nav_price"] = price
    else:
        row.pop("nav_price", None)
    st.session_state[f"pr_navp_{rid}"] = price
    row["nav_source"] = source
    st.session_state[f"pr_navs_{rid}"] = source
    _save(data)


def _unit_label(row: dict) -> str:
    """'USD/lb · Koppar' vid kostnads- och prisrutorna (crosssheet.unit_label). Tom utan känd råvara."""
    try:
        import crosssheet
        return crosssheet.unit_label(row.get("commodity"))
    except Exception:
        return ""


def _commodity(row: dict, key: str, on_apply) -> bool:
    """Råvaran på kortet: väljaren och förslaget med Använd (refresh_ui.commodity_choice). True när du bytte."""
    try:
        import refresh_ui
    except Exception:
        return False
    sel, _rest = st.columns([1.2, 3])
    return refresh_ui.commodity_choice(row, COMMODITIES, key, on_apply, container=sel)


def _fresh_row(row: dict) -> dict:
    """Sifferuppdateringens rad för arkraden (producers:<id>) — den du hämtat nu (Hämta nu) före nattens
    (refresh_ui.current_blob). Tom utan blob."""
    try:
        import refresh_ui
        return dict(((refresh_ui.current_blob() or {}).get("rows") or {}).get(f"producers:{row.get('id')}") or {})
    except Exception:
        return {}


def _fetch_button(row: dict) -> None:
    """'Hämta nu' överst på Rick Rule-kortet (refresh_ui.fetch_button): hela raden ur Börsdata och dagens terminspris för
    kortets råvara blir kortets förslag (EV/EBITDA, nettoskuld/EBITDA, råvarupris, EV, aktier, kurs, FCF 12 mån, DS-
    runway …) — bara i sessionen; Använd skriver, 💾 sparar."""
    try:
        import refresh_ui
    except Exception:                                   # pragma: no cover
        return
    refresh_ui.fetch_button("producers", row, key=f"pr_live_{row['id']}", commodity=row.get("commodity") or None)


def _night_price_fits(row: dict) -> bool:
    """Gäller nattens råvarupris råvaran som står på kortet nu (commodity_prices.night_fits)? Natten hämtade priset
    för råvaran som stod där då — Guld bytt till Olja ger inget guldpris på oljekortet."""
    try:
        import commodity_prices
        return commodity_prices.night_fits(_fresh_row(row), row.get("commodity"))
    except Exception:
        return False


def _registry_identity(ticker: str, name: str) -> dict:
    """ins_id och namn ur universum-registret för ett bolag du lägger in (refresh_ui.registry_identity). Bara på
    klicket; tom utan register — då blir raden som förut."""
    try:
        import refresh_ui
        return refresh_ui.registry_identity(ticker, name)
    except Exception:
        return {}


def _identity_note(row: dict) -> None:
    """'Börsdata: <namn>' på en rad som fått sitt ins_id ur registret."""
    try:
        import refresh_ui
        note = refresh_ui.identity_note(row)
    except Exception:
        note = ""
    if note:
        st.caption(note)


def _chatgpt(data: dict, row: dict, sheet: str) -> None:
    """📋 Underlag till ChatGPT för kortets aktie — Excel/CSV och prompten, och svaret tillbaka efter granskning."""
    try:
        from review import chatgpt_ui
    except Exception:
        return
    chatgpt_ui.render_export(row.get("ticker", ""), {STORE: data}, [sheet], key=f"{sheet}_{row['id']}",
                             name=row.get("name", ""), rows={sheet: row}, on_apply=lambda: _save(data))


def _rc():
    """Kortstödet (fokus, x/y ifyllda, saknas-raden) — None om review-paketet inte går att läsa."""
    try:
        from review import card
        return card
    except Exception:
        return None


def _screen_section(screen_key: str, data: dict, bucket: str, extra) -> None:
    """Håvens träffar (screens_scan.py) med "lägg in i arket"."""
    try:
        import screens_ui
    except Exception:
        return
    existing = {str(r.get("ticker", "")).upper() for r in data.get(bucket, [])}

    def _add(fields: dict) -> None:
        data[bucket].append({"id": _uid(), "date": _today(), **extra(fields)})
        _save(data)

    screens_ui.render_screen_section(screen_key, existing, _add,
                                     key_prefix=f"pr_{bucket}")


# ── UI ───────────────────────────────────────────────────────────────────────
def render_producers_page(sheet: Optional[str] = None) -> None:
    """Ett av de två granskningsarken.

    sheet väljer arket direkt. Utan det ritas en egen väljare, så funktionen
    fungerar både som underflik och fristående — panelen skickar in valet
    eftersom arken numera är egna underflikar under ARKEN.
    """
    data = _load()
    storage_ui.save_bar(STORE, "Granskningsarken")
    title = sheet if sheet in (SHEET_RULE, SHEET_ROYALTY) else "Granskningsarken"
    page_header(title, "Rotationen säger var kapitalet ska, screenern vilka bolag som "
                "kvalar. Det här säger vilket av dem du köper.")

    which = sheet
    if which not in (SHEET_RULE, SHEET_ROYALTY):
        which = st.radio("Ark", [SHEET_RULE, SHEET_ROYALTY], horizontal=True,
                         key="pr_which", label_visibility="collapsed")
    if which == SHEET_RULE:
        _producers(data)
    else:
        _royalty(data)


PROD_CSV = [("date", "Datum"), ("ticker", "Ticker"), ("name", "Bolag"),
            ("commodity", "Råvara"), ("ev_ebitda", "EV/EBITDA"),
            ("nd_ebitda", "Nettoskuld/EBITDA"), ("unit_cost", "Kostnad/enhet"),
            ("price", "Råvarupris"), ("_margin", "Marginal %"),
            ("jurisdiktion", "Jurisdiktion"),
            ("kapitaldisciplin", "Kapitaldisciplin"), ("insyn", "Insyn"),
            ("_score", "Poäng (0–5)"), ("_verdict", "Bedömning"),
            ("position_pct", "Position % av total"),
            ("mine_life", "Gruvlivslängd (år)"), ("rp_ratio", "R/P (år)"),
            ("_dying", "Döende tillgång"),
            ("_ds", "DS"), ("_aqs", "AQS"), ("_csm_flag", "CSM röd flagga"),
            ("fcf_kvalitet", "FCF-klass"), ("_fv_base", "Fair value Base"),
            ("_mos", "Säkerhetsmarginal %"), ("_mos_band", "MoS-bedömning"),
            ("ev_musd", "EV (MUSD)"), ("nav_musd", "NAV efter skatt (MUSD)"),
            ("nav_price", "Metallpris i NAV"), ("nav_source", "NAV-källa"),
            ("_ev_nav", "EV/NAV"), ("_ev_nav_band", "EV/NAV-bedömning")]

ROYALTY_CSV = [("date", "Datum"), ("ticker", "Ticker"), ("name", "Bolag"),
               ("level", "Nivå"), ("pnav_now", "P/NAV nu"),
               ("pnav_bottom", "P/NAV botten"), ("_discount", "Mot botten %"),
               ("ev_now", "EV/EBITDA nu"), ("ev_median", "EV/EBITDA median"),
               ("_median", "Mot median %"), ("geo_now", "GEO/aktie nu"),
               ("geo_3y", "GEO/aktie 3 år"), ("_geo", "GEO-tillväxt %"),
               ("_signal", "Signal")]


def _r1(v):
    return None if v is None else round(v, 1)


def _csv_row(r: dict) -> dict:
    """En rad i Rick Rule-exporten: arket, kontrollerna och Lukacs FV."""
    row = r["row"]
    fv = lukacs.evaluate(row)
    return {**row, "_margin": _r1(r["margin"]), "_score": r["score"],
            "_verdict": r["verdict"].label if r["verdict"] else None,
            "_dying": r["dying"],
            "_ds": ctl.ds_total(row), "_aqs": ctl.aqs_total(row),
            "_csm_flag": ctl.csm_red_flag(row.get("csm_kind", ctl.PRODUCER),
                                          row.get("csm", {}),
                                          bool(row.get("secured_cash"))),
            "_fv_base": _r1(fv["fv_base"]), "_mos": _r1(fv["mos"]),
            "_mos_band": fv["mos_band"],
            "_ev_nav": None if r.get("ev_nav") is None else round(r["ev_nav"], 2),
            "_ev_nav_band": (ev_nav_verdict(row).label if ev_nav_verdict(row) else None)}


def _producers(data: dict) -> None:
    st.markdown(f"<div style='color:{DIM};font-size:0.78rem;margin-bottom:4px;'>"
                f"<b style='color:{TEXT};'>{SHEET_RULE} — Överlevarna.</b> "
                f"Överlevar-screenern körs i Börsdata; de fyra "
                f"granskningsstegen — landrisk, kostnadsposition, ledning, "
                f"kapitaldisciplin — är fälten nedan. "
                f"<span style='color:{DIM};'>Guiden kallar arket "
                f"{GUIDE_NAME_RULE}.</span></div>",
                unsafe_allow_html=True)
    csv_export.download_button([_csv_row(r)
                                for r in ranked_producers(data[PRODUCERS])],
                               PROD_CSV, "rick_rule", key="csv_producers")
    st.caption(f"Marginal ≥ {MARGIN_STRONG:g} % ger 2 p, ≥ {MARGIN_OK:g} % ger "
               f"1 p. Tre disciplinfrågor ger 1 p var. "
               f"{PROD_BUY_MIN}–{PROD_MAX_SCORE} = köpkandidat · "
               f"{PROD_WATCH_MIN} = bevaka · under {PROD_WATCH_MIN} = passa.")

    with st.expander("➕ Nytt bolag", expanded=not data[PRODUCERS]):
        c1, c2, c3 = st.columns([1, 2, 1.2])
        ticker = c1.text_input("Ticker", key="pr_new_t")
        name = c2.text_input("Bolag", key="pr_new_n")
        commodity = c3.selectbox("Råvara", list(COMMODITIES), key="pr_new_c")
        if st.button("Lägg till", key="pr_new_add"):
            if ticker.strip():
                data[PRODUCERS].append({
                    "id": _uid(), "ticker": ticker.strip().upper(),
                    "name": name.strip(), "commodity": commodity,
                    "date": _today()})
                _save(data)
                st.rerun()
            else:
                st.warning("Ticker krävs.")

    # Ingen gissad råvara (förr alltid 'Guld'): tom tills du väljer — kortet föreslår den med Använd.
    _screen_section("rule", data, PRODUCERS, lambda f: {"commodity": "", **f})

    rows = ranked_producers(data[PRODUCERS])
    if not rows:
        st.caption("Inga bolag ännu. Överlevar-screenern körs i Börsdata — "
                   "filtret står i MER → Regler & Guider → 📚 SNABBREFERENS.")
        return

    rc = _rc()
    foc = rc.focus("rule", [r["row"].get("ticker") for r in rows]) if rc else None
    for r in rows:
        row, sc, vd = r["row"], r["score"], r["verdict"]
        if rc and not rc.same_ticker(row.get("ticker"), foc):
            continue
        head = (f"{row.get('ticker','?')} · {row.get('commodity') or '—'} · "
                f"{sc if sc is not None else '–'}/{PROD_MAX_SCORE} p · "
                f"{vd.label if vd else 'ofullständig'}"
                + (f" · {r['ev_nav']:.2f}× NAV" if r.get("ev_nav") is not None else ""))
        with st.expander(head, expanded=bool(foc)):
            top = rc.slot() if rc else None
            changed = _commodity(row, f"pr_com_{row['id']}", lambda: _save(data))
            _fetch_button(row)
            v1, v2, v3, v4 = st.columns(4)
            shown = {k: _nonneg(row.get(k)) for k in ("ev_ebitda", "unit_cost", "price")}
            ev = v1.number_input("EV/EBITDA", min_value=0.0, step=0.5,
                                 value=shown["ev_ebitda"],
                                 key=f"pr_ev_{row['id']}")
            nd = v2.number_input("Nettoskuld/EBITDA", step=0.1,
                                 value=_num(row.get("nd_ebitda")),
                                 key=f"pr_nd_{row['id']}",
                                 help="Negativt = nettokassa.")
            cost = v3.number_input("Kostnad per enhet", min_value=0.0, step=10.0,
                                   value=shown["unit_cost"],
                                   key=f"pr_c_{row['id']}",
                                   help="AISC eller C1 — samma enhet som priset.")
            price = v4.number_input("Råvarupris nu", min_value=0.0, step=10.0,
                                    value=shown["price"],
                                    key=f"pr_p_{row['id']}")
            ulabel = _unit_label(row)                   # fast enhet: kostnad och pris i råvarans prisenhet
            if ulabel:
                v3.caption(ulabel)
                v4.caption(ulabel)
            _suggest("producers", row, "ev_ebitda", "ev_ebitda", "EV/EBITDA",
                     f"pr_ev_{row['id']}", lambda: _save(data), fmt="{:.1f}", lo=0.0)
            _suggest("producers", row, "nd_ebitda", "nd_ebitda", "nettoskuld/EBITDA",
                     f"pr_nd_{row['id']}", lambda: _save(data), fmt="{:.2f}")
            # Råvarupriset ur sifferuppdateringen (Yahoo-terminer, USD per
            # oz/lb/fat) — kontrollera att AISC står i samma enhet. Bara medan
            # nattens pris gäller kortets råvara: bytt råvara (eller '—') sedan
            # natten ger inget förslag.
            if _night_price_fits(row):
                _suggest("producers", row, "price", "commodity_price", "råvarupris",
                         f"pr_p_{row['id']}", lambda: _save(data), fmt="{:,.2f}",
                         unit_field="commodity_unit")
            # AISC ur Durrett-arket — bara samma råvara i samma enhet (crosssheet.cost_offer).
            durrett = _durrett(row)
            if durrett is not None:
                _durrett_offer(data, row, "unit_cost",
                               lambda xs: xs.cost_offer(durrett, row.get("commodity"), row.get("ticker"),
                                                        row.get("price")),
                               f"xs_pr_cost_{row['id']}", f"pr_c_{row['id']}")
            _neg_note(row, {"ev_ebitda": "EV/EBITDA", "unit_cost": "kostnad per enhet", "price": "råvarupris"})
            if _write_changed(row, (("ev_ebitda", ev, shown["ev_ebitda"]), ("nd_ebitda", nd, row.get("nd_ebitda")),
                                    ("unit_cost", cost, shown["unit_cost"]), ("price", price, shown["price"]))):
                changed = True

            m = r["margin"]
            l1, l2, l3 = st.columns(3)
            l1.metric("Marginal", f"{m:.1f} %" if m is not None else "–",
                      help=f"Ger {margin_points(m)} p. Kostnadskurvan är hela "
                           f"caset: de dyraste producenterna dör först i "
                           f"prisfall.")
            spot_m = spot_margin_line(_fresh_row(row), row.get("commodity"), row.get("unit_cost"), row.get("price"))
            if spot_m:
                l1.caption(spot_m.replace("$", "\\$"))
            life = l2.number_input(
                "Gruvlivslängd (år)", min_value=0.0, step=1.0,
                value=_num(row.get("mine_life")),
                key=f"pr_life_{row['id']}",
                help=f"Under {LIFE_MIN_YEARS:g} år = passa oavsett poäng. "
                     f"Står i presentationen.")
            rp = l3.number_input(
                "R/P-kvot (olja/gas, år)", min_value=0.0, step=1.0,
                value=_num(row.get("rp_ratio")),
                key=f"pr_rp_{row['id']}",
                help=f"Reserver ÷ produktion. Under {RP_MIN_YEARS:g} år = "
                     f"samma sak.")
            if (storage.differs(life, row.get("mine_life"))
                    or storage.differs(rp, row.get("rp_ratio"))):
                row["mine_life"], row["rp_ratio"] = life, rp
                changed = True
            if durrett is not None:
                _durrett_offer(data, row, "mine_life",
                               lambda xs: xs.life_offer(durrett, row.get("ticker"), row.get("commodity")),
                               f"xs_pr_life_{row['id']}", f"pr_life_{row['id']}")

            _extract_section(data, row, "rule",
                             {"unit_cost": f"pr_c_{row['id']}",
                              "mine_life": f"pr_life_{row['id']}",
                              "rp_ratio": f"pr_rp_{row['id']}",
                              "nav_musd": f"pr_nav_{row['id']}",
                              "nav_price": f"pr_navp_{row['id']}"})
            dying = asset_dying(row)
            if dying:
                st.error(f"{DYING_ASSET} — den låga multipeln förklaras av "
                         f"att tillgången tar slut. Passa.")
            elif dying is None:
                st.caption(f"Fyll i gruvlivslängd — eller R/P för olja och gas. "
                           f"Strykregeln (under {LIFE_MIN_YEARS:g} respektive "
                           f"{RP_MIN_YEARS:g} år) kan inte prövas utan den, och "
                           f"poängen ensam ser inte att tillgången tar slut.")

            ccols = st.columns(3)
            for (ckey, clabel, chelp), col in zip(CHECKS, ccols):
                v = col.checkbox(clabel, value=bool(row.get(ckey)),
                                 key=f"pr_ch_{row['id']}_{ckey}", help=chelp)
                if v != bool(row.get(ckey)):
                    row[ckey] = v
                    changed = True
            # Förslag ur underlaget (utdraget, Durrett-arket) — text, inte kryss.
            try:
                import judgment_hints as jh
                for hint in (jh.jurisdiction_hint(row.get("ticker", ""), bool(row.get("jurisdiktion"))),
                             jh.insider_hint(row.get("ticker", ""), bool(row.get("insyn")))):
                    if hint:
                        st.caption(hint)
            except Exception:
                pass

            sc2 = producer_score(row)
            vd2 = producer_verdict(sc2, asset_dying(row))
            c2_ = PROD_COLOR.get(vd2.label if vd2 else "", DIM)
            st.markdown(
                f"<div style='border:1px solid {c2_}55;background:{c2_}0d;"
                f"border-radius:8px;padding:10px 14px;margin:8px 0;'>"
                f"<span style='color:{c2_};font-weight:700;font-size:1.05rem;'>"
                f"{sc2 if sc2 is not None else '–'} / {PROD_MAX_SCORE}</span>"
                f"<span style='color:{c2_};font-weight:700;margin-left:12px;'>"
                f"{vd2.label if vd2 else 'Fyll i pris och kostnad'}</span>"
                f"<div style='color:{TEXT};font-size:0.8rem;margin-top:3px;'>"
                f"{vd2.why if vd2 else ''}</div></div>", unsafe_allow_html=True)

            if _ev_nav_section(data, row, durrett):
                changed = True

            # Positionsstorleken styr vilka kontroller som krävs.
            pos = st.number_input(
                "Tänkt position (% av total)", min_value=0.0, step=0.5,
                value=_num(row.get("position_pct")),
                key=f"pr_pos_{row['id']}",
                help=f"Över {ctl.FULL_WORK_MIN_PCT:g} % krävs AQS och CSM. "
                     f"Under räcker DS.")
            if storage.differs(pos, row.get("position_pct")):
                row["position_pct"] = pos
                changed = True
            cap_line = position_cap_line(row)
            if cap_line:
                st.caption(cap_line)

            # Runwayen till DS:s förslag ur sifferuppdateringens kassa och burn (producers:<id>) — vid positivt FCF
            # (burn 0) blir förslaget '> 24 mån'. Gruvlivslängden ger AQS-förslaget på 'Livslängd / resurs'.
            rw, rw_note = controls_ui.sheet_runway("producers", row["id"], row)
            if controls_ui.render_all(
                    row, row["id"], position_pct=pos, strategy="producenter",
                    runway_years=rw, runway_note=rw_note,
                    aqs_prefill=ctl.aqs_prefill_from_producer(row),
                    kind=ctl.PRODUCER, life_years=row.get("mine_life")):
                changed = True

            _chatgpt(data, row, "rule")
            if confirm_delete("Ta bort", key=f"pr_del_{row['id']}"):
                data[PRODUCERS] = [x for x in data[PRODUCERS]
                                   if x["id"] != row["id"]]
                _save(data)
                st.rerun()
            if changed:
                _save(data)
            if rc:
                rc.fill(top, "rule", row)


def _fetch_ev(row: dict, widget_key: str) -> None:
    """Hämta EV (MUSD) ur Börsdata direkt — samma rad som det nattliga jobbet. Hela raden går in i sessionen
    (refresh_ui.fetch_live, som 'Hämta nu'), så kortets andra förslag får den också; bara EV skrivs i rutan, som förut."""
    try:
        import refresh_ui
        import sheets_refresh as sr
        s_, why = refresh_ui.fetch_live("producers", row, commodity=row.get("commodity") or None, yahoo=False)
    except Exception as exc:
        import refresh_ui
        st.session_state[f"{widget_key}_msg"] = f"Börsdata svarade inte: {refresh_ui.bd_error(exc)}"   # utan nyckeln
        return
    if s_ is None and why.startswith("Börsdata svarade inte"):
        st.session_state[f"{widget_key}_msg"] = why
        return
    ev = (s_ or {}).get("ev_musd")
    if ev is not None and ev < 0:                      # rutan har golvet 0 — ett negativt EV skrivs inte in
        st.session_state[f"{widget_key}_msg"] = (f"Börsdata ger negativt EV ({ev:,.0f} MUSD — mer nettokassa än "
                                                 f"börsvärde) — fältet tar bara ≥ 0, så inget skrevs in.")
        return
    if ev is None:
        ccy = str((s_ or {}).get("currency") or "").strip().upper()
        if ccy and ccy not in sr.FX_TO_USD:            # talet finns, men i en valuta jobbet inte kan räkna om
            st.session_state[f"{widget_key}_msg"] = (f"Börsdata har {row.get('ticker', '?')} i {ccy}, som saknas i "
                                                     f"växelkurstabellen — skriv EV själv i MUSD.")
            return
        st.session_state[f"{widget_key}_msg"] = (f"Börsdata har ingen EV för {row.get('ticker', '?')} — "
                                                 f"skriv den själv (börsvärde + nettoskuld, MUSD).")
        return
    row["ev_musd"] = ev
    st.session_state[widget_key] = ev
    st.session_state[f"{widget_key}_msg"] = f"EV {ev:,.0f} MUSD ur Börsdata ({s_.get('asof') or 'idag'})."


def _ev_nav_section(data: dict, row: dict, durrett=None) -> bool:
    """Värderingsraden: EV mot NAV efter skatt. Utanför poängen. True vid ändring. durrett = Durrett-bolaget för
    kortets ticker: dess NAV (eller projekt-NPV) med metallpriset föreslås som ett par, och Använd fyller NAV-källan."""
    rid = row["id"]
    st.markdown(f"<div style='color:{TEXT};font-size:0.85rem;font-weight:600;margin-top:6px;'>"
                f"Värdering — EV/NAV <span style='color:{DIM};font-weight:400;'>"
                f"(utanför poängen)</span></div>", unsafe_allow_html=True)
    k_ev, k_nav, k_np = f"pr_evm_{rid}", f"pr_nav_{rid}", f"pr_navp_{rid}"
    n1, n2, n3 = st.columns(3)
    shown = {k: _nonneg(row.get(k)) for k in ("ev_musd", "nav_musd", "nav_price")}
    ev = n1.number_input("EV (MUSD)", min_value=0.0, step=10.0, value=shown["ev_musd"], key=k_ev,
                         help="Börsvärde + nettoskuld i miljoner USD. Hämtas ur Börsdata.")
    nav = n2.number_input("NAV efter skatt (MUSD)", min_value=0.0, step=10.0, value=shown["nav_musd"],
                          key=k_nav, help="Summan av gruvornas NPV efter skatt (tekniska rapporter eller "
                                          "analytikerkonsensus). Aldrig före skatt.")
    np_ = n3.number_input("Metallpris i NAV", min_value=0.0, step=10.0, value=shown["nav_price"],
                          key=k_np, help="Priset NAV:n är räknad på, i samma enhet som råvarupriset ovan.")
    ulabel = _unit_label(row)
    if ulabel:
        n3.caption(ulabel)
    src = st.text_input("NAV-källa", value=row.get("nav_source") or "", key=f"pr_navs_{rid}",
                        placeholder="t.ex. NI 43-101 2025, 5 % diskontering / analytikerkonsensus")
    b1, b2 = st.columns([1, 3])
    b1.button("Hämta EV nu", key=f"pr_evget_{rid}", on_click=_fetch_ev, args=(row, k_ev),
              help="Hämtar raden ur Börsdata nu och skriver EV i rutan (💾 sparar). Drar Börsdata-kvot. Kortets andra "
                   "förslag får samma rad.")
    if st.session_state.get(f"{k_ev}_msg"):
        b2.caption(st.session_state[f"{k_ev}_msg"])
    _suggest("producers", row, "ev_musd", "ev_musd", "EV (MUSD)", k_ev, lambda: _save(data), fmt="{:,.0f}", lo=0.0)
    if durrett is not None:
        try:
            import crosssheet
            pair = crosssheet.nav_offer(durrett, row.get("commodity"), row.get("ticker"), row.get("price"),
                                        row.get("nav_price"))
        except Exception:
            pair = None
        if pair is not None:
            _durrett_offer(data, row, "nav_musd", lambda xs: (pair.nav, pair.text), f"xs_pr_nav_{rid}", k_nav,
                           on_apply=lambda: _apply_nav_pair(data, row, pair.price, pair.source))
    _neg_note(row, {"ev_musd": "EV", "nav_musd": "NAV efter skatt", "nav_price": "metallpris i NAV"})
    changed = _write_changed(row, (("ev_musd", ev, shown["ev_musd"]), ("nav_musd", nav, shown["nav_musd"]),
                                   ("nav_price", np_, shown["nav_price"])))
    if (src or "") != (row.get("nav_source") or ""):
        row["nav_source"] = src.strip()
        changed = True
    v = ev_nav_verdict(row)
    if v is None:
        st.caption("Fyll i EV och NAV efter skatt för att se EV/NAV.")
    else:
        c = NAV_COLOR.get(v.label, DIM)
        r = ev_nav(row.get("ev_musd"), row.get("nav_musd"))
        st.markdown(
            f"<div style='border:1px solid {c}55;background:{c}0d;border-radius:8px;padding:8px 14px;"
            f"margin:6px 0;'><span style='color:{c};font-weight:700;'>{r:.2f}× NAV</span>"
            f"<span style='color:{c};font-weight:700;margin-left:12px;'>{v.label}</span>"
            f"<div style='color:{TEXT};font-size:0.8rem;margin-top:3px;'>{v.why}</div></div>",
            unsafe_allow_html=True)
    return changed


def _level(row: dict) -> bool:
    """Nivån i kortet (förr bara i ➕ Nytt bolag). Raden skrivs först när du väljer — en rad utan nivå visar –."""
    cur = _num(row.get("level"))
    cur = int(cur) if cur is not None and int(cur) in ROYALTY_LEVELS else None
    opts = ([None] if cur is None else []) + list(ROYALTY_LEVELS)
    new = st.selectbox("Nivå", opts, index=opts.index(cur), key=f"ro_lvl_{row['id']}",
                       format_func=lambda lv: "– (ej satt)" if lv is None else LEVEL_NOTE[lv],
                       help="Nivån styr säljregeln: 1 säljs i princip aldrig, 2 när GEO per aktie stagnerar två år, "
                            "3 vid uppköp eller när tesen är klar.")
    if new is not None and new != cur:
        row["level"] = new
        return True
    return False


def _offer(*args, **kw) -> bool:
    """Ett förslag med Använd (refresh_ui.offer). False utan refresh_ui."""
    try:
        import refresh_ui
    except Exception:                                   # pragma: no cover
        return False
    return refresh_ui.offer(*args, **kw)


def _offer_fields(*args, **kw) -> bool:
    """Flera fält med en Använd (refresh_ui.offer_fields). False utan refresh_ui."""
    try:
        import refresh_ui
    except Exception:                                   # pragma: no cover
        return False
    return refresh_ui.offer_fields(*args, **kw)


def _nav_currencies(price_ccy: str) -> list:
    """Valutorna NAV per aktie kan stå i: växelkurstabellens (sheets_refresh.FX_TO_USD) och kursens egen, USD och
    CAD först (NYSE och TSX)."""
    try:
        from sheets_refresh import FX_TO_USD
        codes = set(FX_TO_USD)
    except Exception:                                   # pragma: no cover
        codes = {"USD", "CAD"}
    if price_ccy:
        codes.add(price_ccy)
    first = [c for c in ("USD", "CAD") if c in codes]
    return first + sorted(codes - set(first))


def _pnav_calc(data: dict, row: dict, fresh: dict, bottom) -> Optional[float]:
    """🧮 P/NAV nu ur nattens kurs och NAV per aktie. NAV per aktie och valutan är tillfälliga rutor — de sparas
    inte. Förslaget skrivs in med Använd (P/NAV nu och rutan), bara när valutorna är samma. Returnerar det räknade
    P/NAV:t till 'Ny egen botten?', None utan förslag."""
    rid = row["id"]
    price = _nonneg(fresh.get("price"))
    with st.expander("🧮 Räkna P/NAV nu ur kursen och NAV per aktie", expanded=False):
        if not price:
            st.caption("Ingen kurs i nattens rad ännu — räknaren behöver den.")
            return None
        pccy = str(fresh.get("currency") or "").strip().upper()
        asof = str(fresh.get("asof") or "")[:10]
        st.caption(f"Kurs {_px(price)} {pccy or '(valuta okänd)'}{' · ' + asof if asof else ''}. NAV per aktie och "
                   f"valutan sparas inte — de används bara här.")
        c1, c2 = st.columns(2)
        navps = c1.number_input("NAV per aktie", min_value=0.0, step=0.5, value=None, key=f"ro_navps_{rid}",
                                help="Analytikernas eller bolagets NAV per aktie. Sparas inte.")
        nccy = c2.selectbox("NAV:ns valuta", [""] + _nav_currencies(pccy), key=f"ro_navccy_{rid}",
                            format_func=lambda c: c or "– välj", help="Måste vara kursens valuta.")
        s = pnav_suggestion(fresh, navps, nccy, bottom)
        if s.values:
            if not _offer(row, "pnav_now", s.values["pnav_now"], "P/NAV nu", key=f"ro_pnav_calc_{rid}",
                          widget_key=f"ro_pn_{rid}", on_apply=lambda: _save(data), cast=float, text=s.text):
                st.caption(f"{s.text} — står redan i rutan.")
        elif s.text:
            st.caption(s.text)
        st.caption(NAV_BASIS_NOTE)
    return s.values.get("pnav_now")


def _bottom_offer(data: dict, row: dict, pnav_now, bottom, source: str) -> None:
    """'Ny egen botten?' med Använd när P/NAV nu (räknat, annars rutans) ligger under den sparade botten. Använd
    skriver bara botten och dess ruta. Talen visas med två decimaler, fler när de annars skulle se lika ut (det
    räknade P/NAV:t är oavrundat)."""
    v = new_bottom(pnav_now, bottom)
    if v is None:
        return
    rid = row["id"]
    b = _num(bottom)
    dd = next((d for d in range(2, 7) if round(v, d) < round(b, d)), 6)
    _offer(row, "pnav_bottom", v, "P/NAV botten", key=f"ro_newbottom_{rid}", widget_key=f"ro_pb_{rid}",
           on_apply=lambda: _save(data), cast=float,
           text=f"🤖 Ny egen botten? P/NAV nu {_sv(v, dd)} ({source}) är under den sparade botten "
                f"{_sv(b, dd)}. Använd bara om NAV-basen är densamma.")


def _geo_calc(data: dict, row: dict, fresh: dict) -> None:
    """🧮 GEO per aktie nu och för tre år sedan ur total GEO (koz, tillfälliga rutor som inte sparas) och nattens
    aktieantal. Ett förslag för paret med en Använd (refresh_ui.offer_fields) — båda fälten och rutorna."""
    rid = row["id"]
    with st.expander("🧮 Räkna GEO per aktie ur total GEO och Börsdatas aktieantal", expanded=False):
        note = shares_note(fresh)
        if not note:
            st.caption("Börsdata har inget aktieantal nu och för tre år sedan för bolaget — räknaren behöver båda.")
            return
        st.caption(note)
        yr = _num(fresh.get("shares_year"))
        k1, k2 = st.columns(2)
        kn = k1.number_input(f"Total GEO {int(yr)} (koz)" if yr else "Total GEO nu (koz)", min_value=0.0,
                             step=1.0, value=None, key=f"ro_geokoz_{rid}",
                             help="Tusen uns guldekvivalenter under året — inte uns. Sparas inte.")
        k3 = k2.number_input(f"Total GEO {int(yr) - 3} (koz)" if yr else "Total GEO för 3 år sedan (koz)",
                             min_value=0.0, step=1.0, value=None, key=f"ro_geokoz3_{rid}",
                             help="Samma mått tre år tidigare, i koz. Sparas inte.")
        s = geo_pair_suggestion(kn, k3, fresh, row.get("geo_now"))
        if s.values:
            if not _offer_fields(row, s.values, s.text, key=f"ro_geo_calc_{rid}",
                                 widget_keys={"geo_now": f"ro_gn_{rid}", "geo_3y": f"ro_g3_{rid}"},
                                 on_apply=lambda: _save(data)):
                st.caption(f"{s.text} — står redan i rutorna.")
            if s.note:
                st.caption(s.note)
        elif s.text:
            st.caption(s.text)


def _royalty(data: dict) -> None:
    csv_export.download_button(
        [{**r["row"], "_discount": _r1(r["discount"]),
          "_median": _r1(r["median"]), "_geo": _r1(r["geo"]),
          "_signal": r["verdict"].label}
         for r in ranked_royalty(data[ROYALTY])],
        ROYALTY_CSV, "royalty_c", key="csv_royalty")
    st.caption(f"Köpläge kräver tre saker samtidigt: högst {DISCOUNT_MAX:g} % "
               f"över egen P/NAV-botten, EV/EBITDA under egen median, och "
               f"växande GEO per aktie.")

    with st.expander("➕ Nytt bolag", expanded=not data[ROYALTY]):
        c1, c2, c3 = st.columns([1, 2, 1])
        ticker = c1.text_input("Ticker", key="ro_new_t")
        name = c2.text_input("Bolag", key="ro_new_n")
        level = c3.selectbox("Nivå", list(ROYALTY_LEVELS), key="ro_new_l")
        if st.button("Lägg till", key="ro_new_add"):
            if ticker.strip():
                t = ticker.strip().upper()
                data[ROYALTY].append({
                    "id": _uid(), "ticker": t,
                    "name": name.strip(), "level": level, "date": _today(),
                    **_registry_identity(t, name)})     # ins_id och Börsdatas namn, så att nattjobbet tar rätt bolag
                _save(data)
                st.rerun()
            else:
                st.warning("Ticker krävs.")

    _screen_section("royalty", data, ROYALTY, lambda f: {"level": 2, **f})

    rows = ranked_royalty(data[ROYALTY])
    if not rows:
        st.caption("Inga bolag ännu. Royalty-screenern körs en gång i "
                   "Börsdata — filtret står i MER → Regler & Guider → 📚 SNABBREFERENS.")
        return

    rc = _rc()
    foc = rc.focus("royalty", [r["row"].get("ticker") for r in rows]) if rc else None
    for r in rows:
        row, vd = r["row"], r["verdict"]
        if rc and not rc.same_ticker(row.get("ticker"), foc):
            continue
        head = (f"{row.get('ticker','?')} · nivå {row.get('level','?')} · "
                f"{vd.label}")
        with st.expander(head, expanded=bool(foc)):
            top = rc.slot() if rc else None
            _identity_note(row)
            changed = _level(row)

            p1, p2, p3 = st.columns(3)
            shown = {k: _nonneg(row.get(k)) for k in ("pnav_now", "pnav_bottom", "ev_now", "ev_median",
                                                      "geo_now", "geo_3y")}
            pn = p1.number_input("P/NAV nu", min_value=0.0, step=0.05,
                                 value=shown["pnav_now"],
                                 key=f"ro_pn_{row['id']}")
            pb = p2.number_input("P/NAV historisk botten", min_value=0.0, step=0.05,
                                 value=shown["pnav_bottom"],
                                 key=f"ro_pb_{row['id']}")
            disc = discount_vs_bottom(pn, pb)
            p3.metric("Mot botten",
                      f"{disc:+.1f} %" if disc is not None else "–",
                      help=f"Köpläge kräver högst +{DISCOUNT_MAX:g} %.")
            fresh = _fresh_row(row)
            calc_pn = _pnav_calc(data, row, fresh, pb)
            _bottom_offer(data, row, calc_pn if calc_pn is not None else pn, pb,
                          "räknat ovan" if calc_pn is not None else "i rutan")

            e1, e2, e3 = st.columns(3)
            en = e1.number_input("EV/EBITDA nu", min_value=0.0, step=0.5,
                                 value=shown["ev_now"],
                                 key=f"ro_en_{row['id']}")
            em = e2.number_input("EV/EBITDA median", min_value=0.0, step=0.5,
                                 value=shown["ev_median"],
                                 key=f"ro_em_{row['id']}")
            med = vs_median(en, em)
            e3.metric("Mot median", f"{med:+.1f} %" if med is not None else "–",
                      help="Negativt = billigare än bolaget brukar vara.")
            _suggest("producers", row, "ev_now", "ev_ebitda", "EV/EBITDA nu",
                     f"ro_en_{row['id']}", lambda: _save(data), fmt="{:.1f}", lo=0.0)
            _suggest("producers", row, "ev_median", "ev_ebitda_median", "EV/EBITDA median (10 år)",
                     f"ro_em_{row['id']}", lambda: _save(data), fmt="{:.1f}")

            g1, g2, g3 = st.columns(3)
            # sex decimaler: GEO per aktie är uns per aktie (0,0021 för de stora, 0,0002 för de små)
            gn = g1.number_input("GEO/aktie nu", min_value=0.0, step=0.01,
                                 format="%.6f",
                                 value=shown["geo_now"],
                                 key=f"ro_gn_{row['id']}")
            g3y = g2.number_input("GEO/aktie för 3 år sedan", min_value=0.0,
                                  step=0.01, format="%.6f",
                                  value=shown["geo_3y"],
                                  key=f"ro_g3_{row['id']}")
            geo = geo_growth(gn, g3y)
            g3.metric("GEO-tillväxt", f"{geo:+.1f} %" if geo is not None else "–",
                      help="Per aktie. Krymper den är bolaget utspätt, inte "
                           "billigt.")
            dil = dilution_line(fresh)            # bara visning — aktieantalet ur nattens rad
            if dil:
                st.caption(dil)
            _geo_calc(data, row, fresh)

            _extract_section(data, row, "royalty",
                             {"geo_now": f"ro_gn_{row['id']}", "geo_3y": f"ro_g3_{row['id']}",
                              "pnav_now": f"ro_pn_{row['id']}"})

            _neg_note(row, {"pnav_now": "P/NAV nu", "pnav_bottom": "P/NAV botten", "ev_now": "EV/EBITDA nu",
                            "ev_median": "EV/EBITDA median", "geo_now": "GEO nu", "geo_3y": "GEO 3 år"})
            if _write_changed(row, (("pnav_now", pn, shown["pnav_now"]), ("pnav_bottom", pb, shown["pnav_bottom"]),
                                    ("ev_now", en, shown["ev_now"]), ("ev_median", em, shown["ev_median"]),
                                    ("geo_now", gn, shown["geo_now"]), ("geo_3y", g3y, shown["geo_3y"]))):
                changed = True

            vd2 = royalty_signal(row)
            c2_ = ROYALTY_COLOR.get(vd2.label, DIM)
            st.markdown(
                f"<div style='border:1px solid {c2_}55;background:{c2_}0d;"
                f"border-radius:8px;padding:10px 14px;margin:8px 0;'>"
                f"<span style='color:{c2_};font-weight:700;'>{vd2.label}</span>"
                f"<div style='color:{TEXT};font-size:0.8rem;margin-top:3px;'>"
                f"{vd2.why}</div></div>", unsafe_allow_html=True)

            _chatgpt(data, row, "royalty")
            if confirm_delete("Ta bort", key=f"ro_del_{row['id']}"):
                data[ROYALTY] = [x for x in data[ROYALTY] if x["id"] != row["id"]]
                _save(data)
                st.rerun()
            if changed:
                _save(data)
            if rc:
                rc.fill(top, "royalty", row)
