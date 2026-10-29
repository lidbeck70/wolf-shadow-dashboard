"""
review/units.py — enhetskontrollen i importen: VARNA när svaret står i en annan enhet, räkna aldrig om.

Prototypen visade faran: AISC i USD/oz mot ett kopparpris i USD/lb vände Rick Rule-domen till Passa. Därför jämförs
råvarufältens per-enhet (oz/lb/t/boe …) med råvarans prisenhet (confidence.commodities), valutan (USD/CAD/AUD …)
och skalan (miljoner/miljarder) med arkets enhet. En varning kryssar ur raden — du avgör. Rena funktioner.
"""

from __future__ import annotations

import re
from typing import Optional

from review import catalog as cat

_CCY = ("usd", "usc", "cad", "aud", "sek", "kr", "eur", "gbp", "zar", "nok", "chf", "mxn", "dkk", "jpy", "cny",
        "brl", "pen", "clp", "ars", "kes")
_SCALE_WORDS = {"m": "M", "mn": "M", "mm": "M", "mio": "M", "million": "M", "millions": "M", "miljon": "M",
                "miljoner": "M", "b": "B", "bn": "B", "billion": "B", "billions": "B", "miljard": "B",
                "miljarder": "B"}
_PER = {"oz": "oz", "ounce": "oz", "ounces": "oz", "toz": "oz", "lb": "lb", "lbs": "lb", "pound": "lb",
        "pounds": "lb", "t": "t", "tonne": "t", "tonnes": "t", "ton": "t", "tons": "t", "mt": "t", "dmt": "t",
        "kg": "kg", "g": "g", "boe": "boe", "bbl": "boe", "barrel": "boe", "fat": "boe", "mmbtu": "mmbtu",
        "ct": "ct", "dag": "dag", "day": "dag", "bu": "bu", "mbf": "mbf"}
_SIMPLE = {"%": "%", "procent": "%", "pct": "%", "percent": "%", "år": "år", "years": "år", "year": "år",
           "yrs": "år", "×": "×", "x": "×", "ggr": "×", "times": "×", "tkr": "tkr", "ksek": "tkr", "tsek": "tkr"}
# Fält som står i råvarans prisenhet (kostnaden ska stå i samma enhet som priset)
PRICE_UNITS = ("råvarans prisenhet", "samma enhet som råvarupriset")
_CONF_PRICE = {"aisc", "commodity_price", "breakeven_price", "npv_price_assumption", "c1_cost"}


def _clean(u) -> str:
    s = str(u or "").strip().lower()
    for a, b in (("us$", "usd "), ("c$", "cad "), ("a$", "aud "), ("$", "usd "), ("€", "eur "), ("£", "gbp ")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def _token(tok: str) -> tuple:
    """(valuta, skala) ur ett ord: 'musd' → ('usd', 'M'), 'cadbn' → ('cad', 'B'), 'million' → ('', 'M')."""
    if tok in _CCY:
        return tok, ""
    if tok[:1] in ("m", "b", "k") and tok[1:] in _CCY:
        return tok[1:], {"m": "M", "b": "B", "k": "K"}[tok[0]]
    for suffix in ("mn", "bn", "m", "b"):
        if tok.endswith(suffix) and tok[:-len(suffix)] in _CCY:
            return tok[:-len(suffix)], _SCALE_WORDS[suffix]
    return "", _SCALE_WORDS.get(tok, "")


def parts(u) -> tuple:
    """(valuta, per-enhet, skala) ur en enhetstext: 'US$/oz' → ('usd', 'oz', ''), 'MCAD' → ('cad', '', 'M')."""
    s = _clean(u)
    if not s:
        return "", "", ""
    head, _sep, tail = s.partition("/")
    ccy, scale = "", ""
    for tok in re.findall(r"[a-zåäöμ]+", head):
        c, sc = _token(tok)
        ccy, scale = ccy or c, scale or sc
    m = re.match(r"\s*([a-zåäöμ]+)", tail)
    per = _PER.get(m.group(1), m.group(1)) if m else ""
    return ("sek" if ccy == "kr" else ccy), per, scale


def commodity_unit(row: Optional[dict]) -> str:
    """Råvarans prisenhet ur radens råvara ('Koppar' eller 'copper' → 'USD/lb'), tom när den är okänd."""
    try:
        from confidence import commodities as com
        c = com.get((row or {}).get("commodity"))
        return c.unit if c else ""
    except Exception:
        return ""


def price_field(f: cat.SheetField) -> bool:
    """Står fältet i råvarans prisenhet (AISC, råvarupris, metallpris i NAV, CSM-pris, break-even)?"""
    if f.unit in PRICE_UNITS:
        return True
    if f.sheet in cat.CONFIDENCE_SHEETS and f.path.split(".", 1)[1] in _CONF_PRICE:
        return True
    return f.sheet == "durrett" and f.path in ("aisc", "target")


def _expected_price_unit(f: cat.SheetField, row: dict) -> str:
    if f.sheet == "durrett" and f.unit:
        return f.unit
    cu = commodity_unit(row)
    if cu:
        return cu
    if f.sheet in cat.CONFIDENCE_SHEETS:
        p = cat.get_point(row, "fields.commodity_price")
        return p.unit if p is not None and p.unit else ""
    return ""


def warnings(f: cat.SheetField, given: str, row: Optional[dict], siblings: Optional[dict] = None) -> list:
    """Varningar när enheten i svaret skiljer sig från arkets — inget räknas om. siblings = {fältnyckel: enhet}
    ur samma svar: utan känd råvara jämförs kostnaden med råvarupriset i svaret."""
    out = []
    given = str(given or "").strip()
    g_ccy, g_per, g_scale = parts(given)
    if price_field(f):
        exp_unit = _expected_price_unit(f, row or {}) or next(
            (u for k, u in (siblings or {}).items()
             if k != f.key and k.split(":", 1)[0] == f.sheet and u and parts(u)[1]), "")
        e_ccy, e_per, _s = parts(exp_unit)
        if not given:
            if exp_unit:
                out.append(f"Enhet saknas — arket räknar mot råvarupriset i {exp_unit}. Kontrollera att talet står "
                           f"i {exp_unit}.")
            return out
        if e_per and g_per and e_per != g_per:
            out.append(f"Enheten {given} skiljer sig från råvarupriset ({exp_unit}) — {g_per} mot {e_per}. "
                       f"Ingenting räknas om: rätta talet eller låt raden vara.")
        if e_ccy and g_ccy and e_ccy != g_ccy:
            out.append(f"Valutan {g_ccy.upper()} skiljer sig från råvarupriset ({exp_unit}) — räknas inte om.")
        return out
    if not given:
        return out
    e_ccy, e_per, e_scale = parts(f.unit)
    if e_ccy and g_ccy and e_ccy != g_ccy:
        out.append(f"Valutan {g_ccy.upper()} skiljer sig från arkets {f.unit} — ingenting räknas om.")
    if e_scale and g_scale and e_scale != g_scale:
        out.append(f"Skalan i {given} skiljer sig från arkets {f.unit} (miljarder/miljoner) — räknas inte om.")
    if e_per and g_per and e_per != g_per:
        out.append(f"Enheten {given} skiljer sig från arkets {f.unit} — räknas inte om.")
    e_simple, g_simple = _SIMPLE.get(_clean(f.unit)), _SIMPLE.get(_clean(given))
    if e_simple and g_simple and e_simple != g_simple:
        out.append(f"Enheten {given} skiljer sig från arkets {f.unit} — räknas inte om.")
    return out
