"""
confidence/commodities.py — ett råvaruregister med crosswalk.

Repot har fyra råvaruvokabulärer som inte mappar till varandra (rotation:
guld/uran/…, blindspot-teman: naturgas/sallsynta/…, resurs-CSV:
oil_gas/rare_earth/…, ratio-exponeringar: gold_miner/…). Registret här är
den gemensamma nyckeln och känner till de andra namnen.

Talen följer princip 19: strategisk betydelse är seedad ur repots
befintliga nödvändighetstabeller (märkt ESTIMATE med källa). Efterfråge-
tillväxt, geopolitisk knapphet och utbudsbalans är None tills ett sourcat
värde matas in — de ger då 0 p och DATA_MISSING, aldrig ett gissat tal.
Överlagringar (användarens värden med källa/datum) merge:as via resolve().
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from confidence.data.provenance import Datapoint, dp, from_dict

_SEED_SRC = "contrarian_alpha.resource_scoring._COMMODITY_SCORES (repo-tabell)"
_SEED_DATE = "2026-09-09"


@dataclass(frozen=True)
class Commodity:
    key: str
    label: str                              # svensk etikett
    unit: str                               # prisenhet
    aliases: tuple                          # andra namn/nycklar i repot och CSV:n
    rotation_key: str = ""                  # rotation.COMMODITIES
    theme_key: str = ""                     # blindspot.theme_board
    ratio_exposure: str = ""                # alpha_regime.commodity_ratios exposure
    ember_complex: str = ""                 # ember.regime komplex
    proxies: tuple = ()                     # ETF-proxies
    strategic_significance: Datapoint = field(default_factory=Datapoint)   # 0–10
    demand_growth: Datapoint = field(default_factory=Datapoint)            # 0–5
    geopolitical_scarcity: Datapoint = field(default_factory=Datapoint)    # 0–5
    supply_balance_pct: Datapoint = field(default_factory=Datapoint)       # underskott % (neg = överskott)
    adjustments: tuple = ()                 # config.SUPPLY_ADJUSTMENTS som gäller
    # Regional knapphet — null tills sourcat (t.ex. USGS Mineral Commodity Summaries)
    supply_concentration_pct: Datapoint = field(default_factory=Datapoint)  # största producentlandets andel %
    top_supplier: str = ""                                                   # det landet
    western_share_pct: Datapoint = field(default_factory=Datapoint)          # andel ur "västliga" jurisdiktioner %
    notes: str = ""


OVERRIDE_POINTS = ("strategic_significance", "demand_growth", "geopolitical_scarcity", "supply_balance_pct",
                   "supply_concentration_pct", "western_share_pct")


def _seed(score_0_100: float) -> Datapoint:
    """Repots 0–100-nödvändighet → 0–10 strategisk betydelse, som ESTIMATE."""
    return dp(round(score_0_100 / 10.0, 1), kind="ESTIMATE", source=_SEED_SRC,
              source_type="secondary", pub_date=_SEED_DATE,
              note="Seedad ur repots nödvändighetstabell — ersätt med sourcat värde")


def _c(key, label, unit, aliases, seed, **kw) -> Commodity:
    return Commodity(key=key, label=label, unit=unit, aliases=tuple(aliases),
                     strategic_significance=_seed(seed), **kw)


REGISTRY: dict = {c.key: c for c in (
    _c("uranium", "Uran", "USD/lb U3O8", ("uran", "u3o8", "uranium"), 95.0,
       rotation_key="uran", theme_key="uran", ember_complex="agri", proxies=("URNM", "URA")),
    _c("copper", "Koppar", "USD/lb", ("koppar", "cu", "copper"), 92.0,
       rotation_key="koppar", theme_key="koppar", ratio_exposure="copper",
       ember_complex="basmetaller", proxies=("COPX", "CPER")),
    _c("lithium", "Litium", "USD/t LCE", ("litium", "li", "lithium"), 90.0,
       rotation_key="litium", ember_complex="basmetaller", proxies=("LIT",)),
    _c("rare_earth", "Sällsynta jordartsmetaller", "USD/kg NdPr", ("rare earth", "rare earths", "ree",
       "sallsynta", "critical minerals", "rare_earth"), 90.0, theme_key="sallsynta", proxies=("REMX",)),
    _c("nickel", "Nickel", "USD/t", ("ni", "nickel"), 85.0, ember_complex="basmetaller", proxies=("JJN",)),
    _c("cobalt", "Kobolt", "USD/lb", ("kobolt", "co", "cobalt"), 85.0, ember_complex="basmetaller",
       proxies=("BATT",)),
    _c("graphite", "Grafit", "USD/t", ("grafit", "graphite"), 85.0, ember_complex="basmetaller",
       proxies=("BATT",)),
    _c("tin", "Tenn", "USD/t", ("tenn", "sn", "tin"), 80.0, ember_complex="basmetaller"),
    _c("zinc", "Zink", "USD/t", ("zink", "zn", "zinc"), 75.0, rotation_key="zink",
       ember_complex="basmetaller", proxies=("XME",)),
    _c("gold", "Guld", "USD/oz", ("guld", "au", "gold"), 85.0, rotation_key="guld", theme_key="guld",
       ratio_exposure="gold_miner", ember_complex="adelmetaller", proxies=("GLD", "GDX")),
    _c("silver", "Silver", "USD/oz", ("ag", "silver"), 82.0, rotation_key="silver", theme_key="silver",
       ratio_exposure="silver", ember_complex="adelmetaller", proxies=("SLV", "SIL")),
    _c("platinum", "Platina", "USD/oz", ("platina", "pt", "pgm", "platinum"), 80.0,
       rotation_key="platina", ember_complex="adelmetaller", proxies=("PPLT",)),
    _c("palladium", "Palladium", "USD/oz", ("pd", "palladium"), 80.0, rotation_key="palladium",
       ember_complex="adelmetaller", proxies=("PALL",)),
    _c("oil_gas", "Olja & gas", "USD/boe", ("olja", "oil", "oil & gas", "oil_gas", "crude", "energy",
       "petroleum"), 80.0, rotation_key="olja", theme_key="olja", ratio_exposure="oil",
       ember_complex="energi", proxies=("XLE", "USO")),
    _c("natural_gas", "Naturgas", "USD/MMBtu", ("gas", "naturgas", "natural gas", "lng"), 78.0,
       rotation_key="gas", theme_key="naturgas", ember_complex="energi", proxies=("UNG",)),
    _c("potash", "Kali", "USD/t", ("kali", "potash"), 80.0, theme_key="agri", ember_complex="agri",
       proxies=("MOO",)),
    _c("phosphate", "Fosfat", "USD/t", ("fosfat", "phosphate"), 78.0, theme_key="agri",
       ember_complex="agri", proxies=("MOO",)),
    _c("iron_ore", "Järnmalm", "USD/t", ("jarnmalm", "järnmalm", "iron ore", "iron_ore"), 79.0,
       rotation_key="jarnmalm", ember_complex="basmetaller", proxies=("XME",)),
    _c("coal", "Kol", "USD/t", ("kol", "coal", "met coal", "thermal coal"), 55.0, rotation_key="kol",
       theme_key="kol", ember_complex="energi", proxies=("KOL",)),
    _c("aluminum", "Aluminium", "USD/t", ("aluminium", "al", "aluminum", "bauxite"), 82.0,
       ember_complex="basmetaller", proxies=("XME",)),
    _c("steel", "Stål", "USD/t", ("stål", "steel"), 80.0, ember_complex="basmetaller", proxies=("SLX",)),
    _c("diamonds", "Diamanter", "USD/ct", ("diamanter", "diamonds"), 50.0),
    _c("agri", "Jordbruk", "USD/t", ("lantbruk", "agri", "agriculture"), 70.0, theme_key="agri",
       ember_complex="agri", proxies=("DBA",)),
)}

_ALIAS: dict = {}
for _k, _c_ in REGISTRY.items():
    _ALIAS[_k] = _k
    for _a in _c_.aliases:
        _ALIAS[_a.lower()] = _k
    for _extra in (_c_.rotation_key, _c_.theme_key):
        if _extra:
            _ALIAS.setdefault(_extra.lower(), _k)


def resolve_key(name: Optional[str]) -> Optional[str]:
    """Nyckel ur valfritt namn: 'Koppar', 'copper', 'koppar', 'rare earth' → registernyckel."""
    if not name:
        return None
    s = str(name).strip().lower().replace("-", "_")
    if s in _ALIAS:
        return _ALIAS[s]
    s2 = s.replace("_", " ")
    if s2 in _ALIAS:
        return _ALIAS[s2]
    for alias, key in _ALIAS.items():
        if len(alias) >= 4 and alias in s2:
            return key
    return None


def get(name: Optional[str], overrides: Optional[dict] = None) -> Optional[Commodity]:
    """Registerposten, med användarens överlagringar (sourcade värden) på plats."""
    key = resolve_key(name)
    if key is None:
        return None
    return resolve(REGISTRY[key], (overrides or {}).get(key))


def resolve(base: Commodity, override: Optional[dict]) -> Commodity:
    """Överlagring: {"strategic_significance": {value, source, ...}, "demand_growth": {...},
    "geopolitical_scarcity": {...}, "supply_balance_pct": {...}, "adjustments": [...]}."""
    if not override:
        return base
    changes = {}
    for f in OVERRIDE_POINTS:
        if f in override and override[f] is not None:
            changes[f] = from_dict(override[f])
    if "adjustments" in override:
        changes["adjustments"] = tuple(override["adjustments"] or ())
    if "top_supplier" in override:
        changes["top_supplier"] = str(override["top_supplier"] or "")
    if "notes" in override:
        changes["notes"] = str(override["notes"] or "")
    return replace(base, **changes) if changes else base


def crosswalk(key: str) -> dict:
    """Andra modulers namn för samma råvara — för Why Now-kopplingarna."""
    c = REGISTRY.get(key)
    if c is None:
        return {}
    return {"rotation": c.rotation_key, "theme": c.theme_key, "ratio_exposure": c.ratio_exposure,
            "ember_complex": c.ember_complex, "proxies": list(c.proxies)}


def all_keys() -> list:
    return list(REGISTRY)
