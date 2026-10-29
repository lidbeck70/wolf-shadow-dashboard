"""
review/importer.py — ChatGPT-svaret tillbaka in i arket: kontrollera, visa skillnaden, skriv först efter ja.

  check(answers, exp)     svaret (review.answers) mot underlaget (review.export.Export): annan ticker, okänd
                          fältnyckel och ogiltiga värden avvisas, 'Bara du' hoppas över, en enhet eller valuta som
                          skiljer sig ger en VARNING (review.units) — ingenting räknas om
  apply(changes, rows)    skriver de valda ändringarna i arkens rader: platta ark får värdet plus radens 'sources'
                          {fält: {värdet, "ChatGPT: <källa> · <datum>"}} (catalog.set_source — källan visas bara
                          medan värdet står kvar), Durrett/Wolf Asymmetry ett Datapoint med källa ('ChatGPT:
                          <källa>', källtyp ur svaret eller 'unsupported'). Tiggres katalysatorer får id och status,
                          och en tom plats svaret hoppade över tas bort
  widget_keys(…)          widgetnycklarna som ska släppas så att kortet ritas om med de nya värdena

Inget sparas härifrån — 💾 Spara sköter GitHub som vanligt. Rena funktioner, ingen Streamlit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

import lukacs
from ai import extract_prompt as xp
from confidence.data.provenance import dp
from review import catalog as cat
from review import units
from review.answers import Answer, ImportFormatError, blank, coerce, iso_date, read_file, read_text  # noqa: F401

SOURCE_TYPES = ("primary", "independent", "secondary", "mixed", "weak", "unsupported")
UNSUPPORTED = "unsupported"
CONFIDENCES = ("high", "medium", "low")
OVERWRITE = "skriver över värdet i arket — kryssa själv om det ska bytas"


@dataclass
class Change:
    """Ett värde som skiljer sig från arket och går att skriva in."""
    field: cat.SheetField
    old: Any
    new: Any
    unit: str = ""
    source: str = ""
    source_type: str = ""
    date: Optional[str] = None
    confidence: str = ""
    comment: str = ""
    warnings: list = field(default_factory=list)

    @property
    def sheet(self) -> str:
        return self.field.sheet

    @property
    def path(self) -> str:
        return self.field.path

    @property
    def key(self) -> str:
        return self.field.key

    @property
    def overwrites(self) -> bool:
        """Står det redan ett värde i arket (ditt, Börsdatas eller ett tidigare svar)?"""
        return not cat.is_empty(self.old)

    @property
    def default_use(self) -> bool:
        """Förkryssad bara med säkerhet high/medium, en källa, utan varning och i ett tomt fält — allt annat kryssar
        du själv. Ett värde som skriver över arkets är aldrig förkryssat, hur säkert svaret än är."""
        return (self.confidence in ("high", "medium") and bool(self.source) and not self.warnings
                and not self.overwrites)


@dataclass
class Result:
    changes: list = field(default_factory=list)
    issues: list = field(default_factory=list)       # [(rad, nyckel, text)] — avvisade
    ignored: list = field(default_factory=list)      # [(rad, nyckel, text)] — hoppade över med flit
    same: int = 0                                    # värden som redan stod i arket

    def signature(self) -> str:
        """Kort fingeravtryck av ändringarna — nytt svar, ny tabell (kryssen följer inte med till ett annat svar)."""
        text = "|".join(f"{c.key}={c.new!r}" for c in self.changes)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


# ── Kontrollen ───────────────────────────────────────────────────────────────
def _forms(ticker: str) -> set:
    t = str(ticker or "").strip().upper()
    if not t:
        return set()
    try:
        from thor.ticker import forms
        out = {x.upper() for x in forms(t) if x}
    except Exception:
        out = set()
    return out | {t}


def _source_type(raw: str) -> tuple:
    s = str(raw or "").strip().lower()
    s = {"primär": "primary", "oberoende": "independent", "sekundär": "secondary", "blandad": "mixed",
         "svag": "weak", "saknas": UNSUPPORTED, "ingen": UNSUPPORTED}.get(s, s)
    if not s:
        return "", ""
    return (s, "") if s in SOURCE_TYPES else ("", f"okänd källtyp ”{raw}” — lämnad tom")


def _confidence(raw: str) -> str:
    s = str(raw or "").strip().lower()
    return {"hög": "high", "medel": "medium", "låg": "low"}.get(s, s if s in CONFIDENCES else "")


def known_path(sheet: str, path: str) -> bool:
    """Finns fältet i arket i någon form — största positionen (alla kontroller), producent eller utvecklare,
    kärninnehav, Lukacs FV och fem katalysatorer? Skiljer 'visas inte just nu' från en påhittad nyckel."""
    if sheet in cat.CONFIDENCE_SHEETS:
        from confidence.config import FIELD_BY_KEY
        return path.startswith("fields.") and path.split(".", 1)[1] in FIELD_BY_KEY
    import controls as ctl
    seen: set = set()
    for kind in (ctl.PRODUCER, ctl.DEVELOPER):
        probe = {"csm_kind": kind, "is_core": True, "fv_manual": True, "nd_ebitda": 99.0, "catalysts": [{}] * 5}
        try:
            seen |= {f.path for f in cat.fields_for(sheet, probe, position_pct=99.0)}
        except Exception:
            continue
    return path in seen


def resolve(key: str, fields: dict) -> tuple:
    """(SheetField, '') för fältnyckeln, eller (None, skäl). Utan arkprefix ('unit_cost') räcker en entydig träff;
    extraktionens alias (AISC = unit_cost …) gäller bara där talet är detsamma (faktor 1 — inget räknas om)."""
    k = str(key or "").strip()
    if not k:
        return None, "fältnyckel saknas"
    if k in fields:
        return fields[k], ""
    if ":" in k:
        sheet, path = k.split(":", 1)
        here = {f.sheet for f in fields.values()}
        if sheet in cat.CONFIDENCE_SHEETS:                       # Durrett och Wolf Asymmetry delar lager
            for s in cat.CONFIDENCE_SHEETS:
                if f"{s}:{path}" in fields:
                    return fields[f"{s}:{path}"], ""
            if here & set(cat.CONFIDENCE_SHEETS):
                here.add(sheet)
        if sheet in cat.SHEETS:
            if sheet not in here:
                return None, f"hör till {cat.SHEETS[sheet].label} — läs in svaret där"
            if known_path(sheet, path):
                return None, "fältet visas inte i det här kortet just nu (t.ex. AQS/CSM under positionsgränsen)"
        return None, "okänd fältnyckel"

    def _hits(name):
        return [f for f in fields.values() if f.path == name or f.path == f"fields.{name}"]
    hits = _hits(k)
    for other, factor in xp.ALIASES.get(k, ()):
        if hits:
            break
        if factor == 1.0:
            hits = _hits(other)
    if len(hits) == 1:
        return hits[0], ""
    if len(hits) > 1:
        return None, "fältnyckeln finns i flera ark — skriv den med arkprefix (t.ex. rule:unit_cost)"
    return None, "okänd fältnyckel"


def check(answers: list, exp, writable: Optional[set] = None) -> Result:
    """Svaret mot underlaget exp (review.export.Export med fields och source_rows). writable = arken svaret får
    skriva i härifrån (None = alla i underlaget); ett annat arks rader hoppas över med en hänvisning dit."""
    res = Result()
    fields = getattr(exp, "fields", None) or {}
    rows = getattr(exp, "source_rows", None) or {}
    want = _forms(getattr(exp, "ticker", ""))
    label = {s: cat.SHEETS[s].label for s in cat.SHEETS}
    resolved, siblings = [], {}
    for a in answers or []:
        if blank(a.value):
            continue
        if a.ticker and want and not (_forms(a.ticker) & want):
            res.issues.append((a.line, a.key, f"annan ticker ({a.ticker}) — underlaget gäller {exp.ticker}"))
            continue
        f, why = resolve(a.key, fields)
        if f is None:
            (res.ignored if why.startswith(("hör till", "fältet visas inte")) else res.issues).append(
                (a.line, a.key, why))
            continue
        if f.who == cat.BARA_DU:
            res.ignored.append((a.line, f.key, "Bara du — beslutas av dig, fylls aldrig i av ChatGPT"))
            continue
        if writable is not None and f.sheet not in writable:
            res.ignored.append((a.line, f.key, f"hör till {label.get(f.sheet, f.sheet)} — läs in svaret där"))
            continue
        if rows.get(f.sheet) is None:
            res.issues.append((a.line, f.key, f"bolaget finns inte i {label.get(f.sheet, f.sheet)} än — lägg in "
                                              f"det i arket först"))
            continue
        resolved.append((a, f))
        siblings[f.key] = str(a.unit or "").strip()
    classes = {f.sheet: coerce(f, a.value)[0] for a, f in resolved if f.path == "fcf_kvalitet"}
    picked: dict = {}
    for a, f in resolved:
        new, err, unit_in_value, warn = coerce(f, a.value)
        if err:
            res.issues.append((a.line, f.key, f"{f.label}: {err}"))
            continue
        if new is None:
            continue
        row = rows.get(f.sheet) or {}
        if f.path.endswith(".target_yield"):
            # Lukacs: yielden är låst till FCF-klassens band — ett fel, inte en varning (lukacs.yield_error). Svarets
            # klass gäller om den byts i samma svar, annars kortets (arkets standard C).
            err_y = lukacs.yield_error(classes.get(f.sheet) or cat.get_value(row, "fcf_kvalitet") or "C", new)
            if err_y:
                res.issues.append((a.line, f.key, f"{f.label}: {err_y}"))
                continue
        unit = str(a.unit or "").strip() or unit_in_value
        notes = [warn] if warn else []
        if f.kind in (cat.NUMBER, cat.INT):
            notes += units.warnings(f, unit, row, siblings)
        stype, st_warn = _source_type(a.source_type)
        notes += [st_warn] if st_warn else []
        when = iso_date(a.date) if str(a.date or "").strip() else None
        if str(a.date or "").strip() and when is None:
            notes.append(f"datumet ”{a.date}” är inte ÅÅÅÅ-MM-DD — utelämnat")
        source = str(a.source or "").strip()
        if not source:
            notes.append("källa saknas")
        old = cat.get_value(row, f.path)
        if cat.same_value(old, new):              # står redan i arket
            res.same += 1
            continue
        if not cat.is_empty(old):
            notes.append(OVERWRITE)
        if f.key in picked:
            res.ignored.append((a.line, f.key, "dubbel rad — den sista gäller"))
        picked[f.key] = Change(f, old, new, unit=unit, source=source, source_type=stype, date=when,
                               confidence=_confidence(a.confidence), comment=str(a.comment or "").strip()[:1000],
                               warnings=notes)
    res.changes = list(picked.values())
    return res


# ── Skriva in ────────────────────────────────────────────────────────────────
def provenance(ch: Change) -> str:
    """Källraden i ett platt ark: 'ChatGPT: <källa> · <datum>'."""
    src = f"ChatGPT: {ch.source}" if ch.source else "ChatGPT (källa saknas)"
    return src + (f" · {ch.date}" if ch.date else "")


def point(ch: Change):
    """Datapoint för Durrett/Wolf Asymmetry-arket: ESTIMATE (bedömningar MODELLED), källtypen ur svaret eller
    'unsupported' när den saknas — då sänker den Data Quality i stället för att låtsas vara primärkälla."""
    return dp(ch.new, kind="MODELLED" if ch.field.who == cat.BEDOMNING else "ESTIMATE",
              source=f"ChatGPT: {ch.source}" if ch.source else "ChatGPT", source_type=ch.source_type or UNSUPPORTED,
              pub_date=ch.date or None, confidence=ch.confidence, unit=ch.unit or ch.field.unit,
              note=ch.comment[:300])


def _finish_catalysts(row: dict) -> None:
    """Katalysatorerna efter inskrivningen. En plats som svaret hoppade över (svaret skrev katalysator 2 i en tom
    lista) blir en tom rad utan id — den tas bort, så att listan inte får en '? / ?'-rad som arket skriver status
    på när det öppnas. En katalysator som svaret lade till får id och status Väntar, som när den läggs in i arket;
    raderna du lagt in själv rörs inte."""
    import uuid
    import tiggre
    cats = row.get("catalysts")
    if not isinstance(cats, list):
        return
    kept = []
    for c in cats:
        if not isinstance(c, dict) or not (c.get("id") or c.get("name") or c.get("date")):
            continue
        if not c.get("id"):
            c["id"] = uuid.uuid4().hex[:8]
            c.setdefault("status", tiggre.CAT_WAITING)
        kept.append(c)
    cats[:] = kept


def apply(changes: list, rows: dict) -> int:
    """Skriver ändringarna i raderna {ark: rad} (samma objekt som sessionen håller). Returnerar antalet."""
    n = 0
    for ch in changes or []:
        row = (rows or {}).get(ch.sheet)
        if not isinstance(row, dict):
            continue
        if ch.sheet in cat.CONFIDENCE_SHEETS:
            cat.set_value(row, ch.path, point(ch))
        else:
            cat.set_value(row, ch.path, ch.new)
            cat.set_source(row, ch.path, ch.new, provenance(ch))
        n += 1
    tg = (rows or {}).get("tiggre")
    if isinstance(tg, dict) and any(c.sheet == "tiggre" and c.path.startswith("catalysts.") for c in changes or []):
        _finish_catalysts(tg)
    return n


def widget_keys(keys, tokens=(), prefixes=()) -> list:
    """Widgetnycklarna som håller kortets värden: nyckeln har radens id (eller kortnyckel) som eget led
    ('pr_c_<id>', 'csm_<id>_Bear_price'), eller börjar med ett prefix ('cf_<TICKER>_'). Underlagets egna
    nycklar (rv_) rörs inte."""
    toks = [str(t) for t in tokens or () if str(t or "").strip()]
    out = []
    for k in keys or ():
        s = str(k)
        if s.startswith("rv_"):
            continue
        if any(f"_{t}_" in f"_{s}_" for t in toks) or any(s.startswith(p) for p in prefixes or ()):
            out.append(k)
    return out
