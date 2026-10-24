"""
valhall/complexes.py — Valhalls sex råvarukomplex: rutor ovanför råvarutemana.

Indelningen är EMBER:s (ember.config.THEME_TO_COMPLEX): energi, basmetaller, ädelmetaller, agri, skog, frakt —
skog och frakt är egna komplex eftersom de har egna drivare, egna nordiska bolag och egen cykel (BERSERK:s fem
lägger skog under agri; komplexen är ändå eniga om alla andra teman, testat i tests/test_ember_berserk_themes).

Varje komplex får en KOMPLEXDRIVARE: en likaviktad korg (start 100) av temadrivarnas serier, så att komplexet
kan köras genom samma cykelmotor som ett tema (valhall.cycle: fas, cykelläge, status, cykelstart, bekräftad
vändning). Bolag → komplex härleds ur registrets tema (BERSERK-vokabulär), annars ur Börsdatas bransch-id.
Det finns ingen regel här — bara gruppering och serier.
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from berserk import themes as bth
from ember.config import THEME_TO_COMPLEX
from valhall import nordic_index as ni

COMPLEXES: tuple = (("energi", "Energi"), ("basmetaller", "Basmetaller"), ("adelmetaller", "Ädelmetaller"),
                    ("agri", "Agri"), ("skog", "Skog & massa"), ("frakt", "Frakt"))
KEYS: tuple = tuple(k for k, _ in COMPLEXES)
LABEL: dict = dict(COMPLEXES)
# Nödvändighet (Rick Rule: behövt) per komplex — samma skala som segmenten/temana (0–100).
NECESSITY: dict = {"energi": 88, "basmetaller": 85, "adelmetaller": 80, "agri": 75, "skog": 68, "frakt": 74}
# Börsdatas bransch-id → komplex för bolag utan tema i registret (contrarian_alpha.necessity.BORSDATA_BRANCH_MAP):
# 1–5 olja & gas, 6 kol, 7 uran · 16 prospekt, 17 industrimetaller · 18 guld & silver · 21 skogsbolag ·
# 41 sjöfart & rederi · 60 jordbruk, 61 fiskodling.
BRANCH_TO_COMPLEX: dict = {1: "energi", 2: "energi", 3: "energi", 4: "energi", 5: "energi", 6: "energi", 7: "energi",
                           16: "basmetaller", 17: "basmetaller", 18: "adelmetaller", 21: "skog", 41: "frakt",
                           60: "agri", 61: "agri"}
_THEME_TO_COMPLEX: dict = {**THEME_TO_COMPLEX, "agri_korg": "agri"}
assert set(_THEME_TO_COMPLEX.values()) <= set(KEYS), "ett EMBER-komplex saknas i valhall.complexes"


def complex_of(theme: Optional[str]) -> Optional[str]:
    """Temats komplex (EMBER:s indelning) — None för okänt tema."""
    return _THEME_TO_COMPLEX.get(str(theme or "").strip().lower()) or None


def themes_of(key: str) -> list:
    """Komplexets teman i BERSERK:s ordning (korgarna, t.ex. agri_korg, räknas inte som tema)."""
    return [t for t in bth.THEMES if complex_of(t) == key]


def complex_for(theme: Optional[str] = None, branch_id=None) -> Optional[str]:
    """Bolagets komplex: temat först (registret/BERSERK-listan), annars Börsdatas bransch."""
    cx = complex_of(theme) if theme else None
    if cx:
        return cx
    try:
        return BRANCH_TO_COMPLEX.get(int(branch_id)) if branch_id is not None else None
    except (TypeError, ValueError):
        return None


def member_complexes(registry) -> dict:
    """{ticker: komplex} ur universum-registret (Member.theme, annars Member.branch_id)."""
    out = {}
    for t, m in (getattr(registry, "members", None) or {}).items():
        cx = complex_for(getattr(m, "theme", None), getattr(m, "branch_id", None))
        if cx:
            out[str(t).upper()] = cx
    return out


def basket(key: str, drivers: dict, extra: Optional[dict] = None) -> tuple:
    """(drivarsymboler, komplexdrivaren): likaviktad korg av komplexets temadrivare. drivers = {tema: (symbol,
    serie)} som berserk.live.load_drivers ger; extra = {tema: serie} för teman utan drivare (producentkorgar).
    Ett tema räcker (skog, frakt); saknas alla blir serien None."""
    closes, syms = {}, []
    for t in themes_of(key):
        sym, s = (drivers or {}).get(t, (None, None))
        if s is None and extra and extra.get(t) is not None:
            sym, s = "producentkorg", extra[t]
        if s is None or len(s) == 0:
            continue
        closes[t] = pd.Series(s).astype(float)
        syms.append(f"{t}:{sym}" if sym else t)
    return syms, ni.equal_weight_index(closes, min_members=1, turnover=None)


def complex_themes(rows: Iterable[dict]) -> dict:
    """{komplex: [temarad]} ur kartans segmentrader (varje tema en gång, i BERSERK:s ordning)."""
    seen: dict = {}
    for r in rows or []:
        for t in (r.get("themes") or []) if isinstance(r, dict) else []:
            if isinstance(t, dict) and t.get("theme") and t["theme"] not in seen:
                seen[t["theme"]] = t
    out = {k: [] for k in KEYS}
    for theme in bth.THEMES:
        if theme in seen and complex_of(theme):
            out[complex_of(theme)].append(seen[theme])
    return out
