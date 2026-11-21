"""
LIDSKJALV — det som designen aldrig får röra (docs/design_lidskjalv.md §5 och §8 punkt 12).

Omgörningen ordnar, färgar och förpackar bara. Reglerna (strategy_rules*.py), de frysta Viking-filerna och
navigationens namn, nycklar och länkar är desamma före och efter varje design-PR. Kontrollsummorna och de
bokstavliga strukturerna nedan uppdateras BARA när ägaren själv ändrar en regel, ett namn eller en länk.
"""
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ui import nav                        # noqa: E402

# sha256 med radslut normaliserade till \n (samma summa på Windows-utcheckningar)
RULE_FILES = {
    "strategy_rules.py": "b4a903f6cde77b9ffd41d5eee5f12edbced916c2c4a3bb60d44ef4caaa58c0a6",
    "strategy_rules_masterguide.py": "8daa052bbe061205fbd286ce96d42d81136baf8286e27ea1f22b39c75ff83f89",
}
FROZEN_VIKING = {
    "ovtlyr/ui/layout.py": "798b7fd2207d3d568f99bdf88e33672c27ce4d1058e8b10475f5d22a2de4f65e",
    "screener_ovtlyr.py": "53b372967bb3202053107e6d374700badf390ee6bdac833fdf3cac173f854008",
    "ovtlyr_nine.py": "4ce8d79a0568d0920ea6a64e47c248067dfe49379081af6745d5d282241f6ae1",
    "viking_execution.py": "b03f0cfc9566bce9aeaa1c0952c6f06d222f610a0cfd80b036e1ac207f9fd280",
    "viking_exit.py": "d65894db63c0147246ada2a55bbe92ebb96da1f2589e2dd10459cab7c5e5fc1e",
    "viking_screen.py": "675708fdda8aeb74bd84a1f2751612b03b7aada5b9de417cab3f5b0036db8def",
    "strategies/viking.py": "058fbcebf61fff175ee128ba4019c3dcee831f9fe0081f6419fc165a280cf620",
    "ovtlyr/ui/viking_nine_regime.py": "d637bd1f0d6446d52630c6c99a428bbb4e45c04ff4a6def5882cf8bc73ecfa47",
    "ovtlyr/ui/viking_screens.py": "de6a3a970d3b879e24dd110bc8d20d62b3a39063af17b22f11adbf259bb740a3",
}


def _sha(rel: str) -> str:
    with open(os.path.join(ROOT, rel), "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


# ── Reglerna och de frysta filerna, byte för byte ────────────────────────────
def test_rule_files_are_unchanged():
    found = sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "strategy_rules*.py")))
    assert found == sorted(RULE_FILES), "en ny regelfil — lägg till dess summa efter ägarens beslut"
    for rel, digest in RULE_FILES.items():
        assert _sha(rel) == digest, f"{rel} har ändrats — regler ändras bara av ägaren"


def test_frozen_viking_files_are_unchanged():
    for rel, digest in FROZEN_VIKING.items():
        assert _sha(rel) == digest, f"{rel} är fryst (Viking) — designen rör den inte"


# ── Navigationen: namn, nycklar och länkar ───────────────────────────────────
def test_top_tabs_are_literal_and_in_decision_order():
    assert nav.TOP == (
        ("home", "🏠 I DAG"),
        ("odin", "👁 ODIN"),
        ("thor", "🔨 THOR"),
        ("review", "📒 ARKEN"),
        ("portfolio", "💼 PORTFÖLJ"),
        ("more", "⋯ MER"),
    )
    assert nav.TABS_KEY == "main_tabs"


def test_sub_navigation_strings_are_literal():
    assert nav.SUBS == {
        "odin": ["Läget", "Kartan", "Sektorrotation", "Råvaror", "Makro", "Marknad"],
        "odin/Råvaror": ["Odin's Blindspot", "🔥 EMBER", "🪓 BERSERK Regime", "Råvarurotation", "🥇 Guld"],
        "odin/Makro": ["🐺 Fiat Debasement", "Sentiment"],
        "odin/Marknad": ["Sektorflöde", "Värmekarta"],
        "review": ["🎯 Scorecard", "Rick Rule", "Royalty C", "Poängmodell", "Tiggre", "Insider",
                   "🧭 Durrett", "🐺 Wolf Asymmetry", "🚀 Råvaruhävstång"],
        "portfolio": ["Positioner", "👀 Övervakning", "🎯 Mot målet", "📊 Papperskonton", "Journal", "Veckorutin",
                      "Allokering"],
        "more": ["Larm", "⚙️ Inställningar", "Regler & Guider", "Copilot", "Motorsidor", "Egen backtest"],
        "more/Motorsidor": ["Wolf Regime", "Viking Regime", "⚔️ Viking Nine Regime", "Quality & Contrarian",
                            "Long Trend", "Market Cycle"],
        "more/Regler & Guider": ["🚀 KOM IGÅNG", "📋 HANDELSREGLER", "⚡ FUSKLAPP",
                                 "🗺 FLIKGUIDE", "📖 STRATEGIGUIDER", "🗓 ÅRSHJULET",
                                 "📚 SNABBREFERENS"],
    }


def test_widget_state_keys_are_literal():
    assert nav.STATE_KEY == {
        "odin": "sub_odin",
        "odin/Råvaror": "sub_odin_ravaror",
        "odin/Makro": "sub_odin_makro",
        "odin/Marknad": "sub_odin_marknad",
        "review": "sub_review",
        "portfolio": "sub_portfolio",
        "more": "sub_more",
        "more/Motorsidor": "sub_motor",
        "more/Regler & Guider": "rules_sub",
    }


def test_link_targets_and_their_slugs_are_literal():
    assert list(nav.TARGETS) == [
        "home", "thor", "turn", "valhall_map", "sector_rotation", "laget", "swing_regime", "market_risk",
        "rotation", "fiat", "paper", "insider", "rule", "royalty", "scoring", "tiggre", "durrett", "scorecard",
        "holdings", "watch", "journal", "alerts", "settings", "goal", "backtests",
    ]
    # ?p=-adresserna som larmen i Discord och knapparna i panelen pekar på
    assert {k: nav.target_slug(k) for k in nav.TARGETS} == {
        "home": "home", "thor": "thor", "turn": "home", "valhall_map": "odin/kartan",
        "sector_rotation": "odin/sektorrotation", "laget": "odin/laget", "swing_regime": "odin/laget",
        "market_risk": "odin/laget", "rotation": "odin/ravaror/ravarurotation", "fiat": "odin/makro/fiat-debasement",
        "paper": "portfolio/papperskonton", "insider": "review/insider", "rule": "review/rick-rule",
        "royalty": "review/royalty-c", "scoring": "review/poangmodell", "tiggre": "review/tiggre",
        "durrett": "review/durrett", "scorecard": "review/scorecard", "holdings": "portfolio/positioner",
        "watch": "portfolio/overvakning", "journal": "portfolio/journal", "alerts": "more/larm",
        "settings": "more/installningar", "goal": "portfolio/mot-malet", "backtests": "portfolio/mot-malet",
    }


def test_legacy_links_and_inner_seeds_are_unchanged():
    assert nav.LEGACY_SLUGS == {"durrett-confidence": "durrett"}
    assert nav.INNER_SEEDS == {
        "odin/ravaror/guldkvoter": ("odin_gold_tabs", "🥇 Guldkvoter"),
        "odin/makro/retail-pulse": ("odin_sentiment_tabs", "Retail"),
        "odin/marknad/retail-pulse": ("odin_sentiment_tabs", "Retail"),
        "intel/retail-pulse": ("odin_sentiment_tabs", "Retail"),
    }
    # 44 gamla sökvägar — summan över ordningen och varje par (en ändring här är ett ägarbeslut)
    blob = json.dumps(nav.LEGACY_PATHS, ensure_ascii=False).encode("utf-8")
    assert len(nav.LEGACY_PATHS) == 44
    assert hashlib.sha256(blob).hexdigest() == "38151aaa1437af2c88302f06f26c380f91a35a27249d045f33613e35a67ff906"
