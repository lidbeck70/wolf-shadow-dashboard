"""
thor/registry.py — strategiregistret: vad Thor vet om varje strategi.

Familjerna är listans filterchips (Swing · Lång · Råvara · Insider · Håvarna). Thor-rank 0–100 är en
PRESENTATIONSSKALA för sortering mellan strategier — 87 i Viking Nine och 87 i Long Screener betyder olika
saker, så strategins egen siffra står alltid bredvid. Hur ranken räknas står vid varje strategi (rank).

Playbook = nyckeln i strategy_rules.PLAYBOOKS (reglerna i dossiern). Viking Nine och BERSERK saknar egen
playbook ännu (fas 7); tills dess lånar Viking Nine Vikings. Regime = regimlayouten i dossiern (fas 4).
Paper = Valhalls papperskonto (valhall/paper.py STRATEGIES).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ui.tokens import CYAN, EMBER, GOLD, GREY, PURPLE

SWING, LANG, RAVARA, INSIDER, HAV = "swing", "lång", "råvara", "insider", "håv"
FAMILIES = ((SWING, "Swing"), (LANG, "Lång"), (RAVARA, "Råvara"), (INSIDER, "Insider"), (HAV, "Håvarna"))
FAMILY_LABEL = dict(FAMILIES)
FAMILY_COLOR = {SWING: CYAN, LANG: GOLD, RAVARA: EMBER, INSIDER: PURPLE, HAV: GREY}


@dataclass(frozen=True)
class Strategy:
    key: str
    label: str
    family: str
    source: str                    # filen/körningen kandidaterna läses ur
    rank: str                      # hur Thor-rank 0–100 räknas ur strategins egen siffra
    playbook: str = ""             # strategy_rules.PLAYBOOKS-nyckel
    regime: str = ""               # regimlayouten i dossiern
    paper: str = ""                # Valhalls papperskonto


STRATEGIES: dict = {s.key: s for s in (
    Strategy("viking_nine", "Viking Nine", SWING, "valhall_board.json (swing)", "vägd Nine 0–100",
             "viking_nine", "viking_nine", "viking"),
    Strategy("wolf", "Wolf", SWING, "arc_screeners.json (wolf)", "regimpoäng 0–125 ÷ 1,25", "wolf", "wolf", "wolf_arc"),
    Strategy("viking", "Viking", SWING, "arc_screeners.json (viking)", "V9 0–9 × 100/9", "viking", "viking"),
    Strategy("momentum", "Momentum Swing", SWING, "wolf_screener.json", "rad 1 = 100, rad 40 = 2,5",
             "momentum", "momentum", "wolf"),
    Strategy("berserk", "BERSERK", RAVARA, "berserk_scan.json", "setup S2 100 · S1 80 · S3 60, BEVAKA −20",
             "berserk", "berserk", "berserk"),
    Strategy("ember", "EMBER", RAVARA, "ember_results.json", "setup_score 0–100", "ember", "ember", "ember"),
    Strategy("quality", "Quality Contrarian", LANG, "contrarian_alpha_results_quality.json",
             "composite 0–100 (tak 100)", "quality", "alpha_quality", "quality"),
    Strategy("deep", "Deep Contrarian", LANG, "contrarian_alpha_results_deep_contrarian.json",
             "composite 0–100 (tak 100)", "contrarian", "alpha_contrarian", "deep"),
    Strategy("long", "Long Screener", LANG, "valhall_board.json (long)", "poäng % × 100", "alpha", "long_screener", "long"),
    Strategy("insider", "Insider Norden", INSIDER, "insider_scan.json", "poäng 0–10 × 10", "insider", "insider", "insider"),
    Strategy("rule", "Rick Rule", RAVARA, "valhall_board.json (råvaror) + screens.json", "arket 0–5 × 20 · Quality composite · håven utan rang",
             "rule", "sheet", "rule"),
    Strategy("durrett", "Durrett", RAVARA, "valhall_board.json (råvaror) + screens.json", "Durrett-motorns kvalitet 0–100",
             "durrett", "sheet", "durrett"),
    Strategy("tiggre", "Tiggre", RAVARA, "valhall_board.json (råvaror) + screens.json", "faktorpoäng 0–10 × 10",
             "tiggre", "sheet", "tiggre"),
    Strategy("royalty", "Royalty", RAVARA, "royalty_scan.json", "composite 0–100 (playbookens grindar + CA-pelarna)",
             "royalty", "royalty", "royalty"),
    Strategy("hav", "Håvarna", HAV, "screens.json", "ingen — en träff är en träff"),
)}
ORDER: list = list(STRATEGIES)                 # listans ordning vid lika rang


def get(key: str) -> Optional[Strategy]:
    return STRATEGIES.get(key)


def label(key: str) -> str:
    s = STRATEGIES.get(key)
    return s.label if s else str(key)


def by_family(family: str) -> list:
    return [s for s in STRATEGIES.values() if s.family == family]
