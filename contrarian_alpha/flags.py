"""
flags.py — UI-flaggor per bolag för Contrarian Alpha Screener.

Fem flaggor utvärderas från ett ContrairianAlphaResult-objekt:

  🟢  VIKING_GREEN   Viking Regime (OVTLYR) är grönt för tickern
  ⚠️  VALUE_TRAP     Hat Score > 85 OCH Strength Score < 50
  📉  TRENDING_DOWN  Catalyst Score < 20 — ingen teknisk reversal i sikte
  💧  LOW_LIQUIDITY  Daglig omsättning < 500 000 USD
  📊  DATA_GAP       En eller flera datapunkter saknas; poängen är ett estimat

Flaggor returneras som list[Flag] och kan renderas i Streamlit-tabeller
via helper-funktionerna format_flags_string() och format_flags_html().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contrarian_alpha.engine import ContrairianAlphaResult

# ─── Trösklar (enkla konstanter, lätta att justera) ───────────────────────────

VALUE_TRAP_HAT_MIN      = 85.0   # Hat score ovan denna…
VALUE_TRAP_STRENGTH_MAX = 50.0   # …kombinerat med strength under denna
TRENDING_DOWN_THRESHOLD = 20.0   # Catalyst score < denna → Trending Down
LIQUIDITY_THRESHOLD_USD = 500_000.0  # Daglig omsättning < 500 k USD

# ─── Data-gap-nyckelord i flagg-strängar från sub-modulerna ─────────────────

_MISSING_KEYWORDS = ("MISSING", "NO_FUNDAMENTAL_DATA", "NO_PRICE_DATA", "PARTIAL",
                     "UNKNOWN", "SHORT")

# ─── Flag-modell ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Flag:
    key:      str    # maskinläsbar nyckel, t.ex. "VIKING_GREEN"
    emoji:    str    # visas i tabell
    label:    str    # kort etikett, t.ex. "Viking"
    detail:   str    # tooltip / expanderbar förklaring
    severity: str    # "positive" | "info" | "warning" | "danger"

    def __str__(self) -> str:
        return f"{self.emoji} {self.label}"


# ─── Flagg-definitioner ───────────────────────────────────────────────────────

FLAG_DEFINITIONS: dict[str, dict] = {
    "VIKING_GREEN": dict(
        emoji    = "🟢",
        label    = "Viking",
        severity = "positive",
        detail   = (
            "OVTLYR Viking Regime är GRÖNT: pris > EMA200, EMA50 > EMA200 "
            "och låg volatilitet. Institutionellt momentum stödjer en lång position. "
            "+5p bonus i Composite Score."
        ),
    ),
    "VALUE_TRAP": dict(
        emoji    = "⚠️",
        label    = "Value Trap?",
        severity = "danger",
        detail   = (
            f"Hat Score > {VALUE_TRAP_HAT_MIN:.0f} men Strength Score < "
            f"{VALUE_TRAP_STRENGTH_MAX:.0f}. "
            "Bolaget är hatat OCH finansiellt svagt — potentiell fallande kniv, "
            "inte en contrarian möjlighet. Granska balansräkningen manuellt."
        ),
    ),
    "TRENDING_DOWN": dict(
        emoji    = "📉",
        label    = "Trending Down",
        severity = "warning",
        detail   = (
            f"Catalyst Score < {TRENDING_DOWN_THRESHOLD:.0f}. "
            "Ingen teknisk reversalsignal: pris under SMA50, negativt lutning "
            "och/eller svag volym. Thesen är korrekt men för tidig — "
            "invänta en teknisk entry."
        ),
    ),
    "LOW_LIQUIDITY": dict(
        emoji    = "💧",
        label    = "Low Liquidity",
        severity = "warning",
        detail   = (
            f"Daglig omsättning under {LIQUIDITY_THRESHOLD_USD/1_000:.0f}k USD "
            "(pris × 20-dagars snittvolym). Hög spread-risk och svår att skala. "
            "Positionsstorleken bör begränsas."
        ),
    ),
    "DATA_GAP": dict(
        emoji    = "📊",
        label    = "Data Gap",
        severity = "info",
        detail   = (
            "En eller flera datapunkter saknas (Altman Z, short interest, "
            "analystdata eller StockTwits). Poängen för berörda komponenter "
            "är uppskattade med neutrala standardvärden. Lägre tillit."
        ),
    ),
}


# ─── Utvärdering ─────────────────────────────────────────────────────────────

def _make(key: str, **override) -> Flag:
    """Bygg ett Flag-objekt från FLAG_DEFINITIONS med valfria overrides."""
    defn = FLAG_DEFINITIONS[key].copy()
    defn.update(override)
    return Flag(key=key, **defn)


def _has_data_gap(result: ContrairianAlphaResult) -> tuple[bool, str]:
    """
    Kontrollera om någon sub-modul saknar riktig data.
    Returnerar (has_gap: bool, detail_suffix: str).
    """
    missing_sources: list[str] = []

    # Samla alla flaggor från sub-resultat
    all_sub_flags: list[str] = list(result.all_flags)
    if result.strength_result:
        all_sub_flags += result.strength_result.flags
    if result.hate_result:
        all_sub_flags += result.hate_result.flags
    if result.catalyst_result:
        all_sub_flags += result.catalyst_result.flags

    for f in all_sub_flags:
        if any(kw in f for kw in _MISSING_KEYWORDS):
            missing_sources.append(f)

    # Kontrollera Altman Z separat (påverkar Strength-score direkt)
    if result.altman_z is None:
        missing_sources.append("ALTMAN_Z_MISSING")

    # Låg hate-confidence betyder estimerade default-värden
    if result.hate_result and result.hate_result.confidence < 0.6:
        missing_sources.append("HATE_CONFIDENCE_LOW")

    if not missing_sources:
        return False, ""

    # Bygg en kortfattad list av vad som saknas (deduplicated, max 4)
    unique = list(dict.fromkeys(missing_sources))[:4]
    detail_suffix = " | Saknar: " + ", ".join(unique)
    return True, detail_suffix


def evaluate_flags(
    result: ContrairianAlphaResult,
    avg_volume_20d: float | None = None,
) -> list[Flag]:
    """
    Utvärdera alla 5 flaggor för ett enskilt instrument.

    Args:
        result:         ContrairianAlphaResult från engine.run_pipeline().
        avg_volume_20d: 20-dagars snittvolym (antal aktier). Används för
                        LOW_LIQUIDITY-beräkning. Om None används
                        result.avg_volume_20d om det finns.

    Returns:
        Sorterad lista: danger → warning → positive → info.
    """
    flags: list[Flag] = []

    # ── 🟢 VIKING_GREEN ──────────────────────────────────────────────────────
    if result.catalyst_result and result.catalyst_result.viking_regime_green:
        color = result.catalyst_result.viking_regime_color.upper()
        nine  = result.catalyst_result.ovtlyr_nine
        detail_extra = f" OVTLYR NINE: {nine}/100." if nine is not None else ""
        flags.append(_make(
            "VIKING_GREEN",
            detail=FLAG_DEFINITIONS["VIKING_GREEN"]["detail"] + detail_extra,
        ))

    # ── ⚠️ VALUE_TRAP ────────────────────────────────────────────────────────
    if (result.hat_score > VALUE_TRAP_HAT_MIN
            and result.strength_score < VALUE_TRAP_STRENGTH_MAX):
        flags.append(_make(
            "VALUE_TRAP",
            detail=(
                f"Hat {result.hat_score:.1f} > {VALUE_TRAP_HAT_MIN:.0f} "
                f"och Strength {result.strength_score:.1f} < {VALUE_TRAP_STRENGTH_MAX:.0f}. "
                + FLAG_DEFINITIONS["VALUE_TRAP"]["detail"]
            ),
        ))

    # ── 📉 TRENDING_DOWN ─────────────────────────────────────────────────────
    if result.catalyst_score < TRENDING_DOWN_THRESHOLD:
        flags.append(_make(
            "TRENDING_DOWN",
            detail=(
                f"Catalyst {result.catalyst_score:.1f} < {TRENDING_DOWN_THRESHOLD:.0f}. "
                + FLAG_DEFINITIONS["TRENDING_DOWN"]["detail"]
            ),
        ))

    # ── 💧 LOW_LIQUIDITY ─────────────────────────────────────────────────────
    vol = avg_volume_20d
    if vol is None:
        vol = getattr(result, "avg_volume_20d", None)

    if vol is not None and result.close > 0:
        daily_usd = result.close * vol
        if daily_usd < LIQUIDITY_THRESHOLD_USD:
            flags.append(_make(
                "LOW_LIQUIDITY",
                detail=(
                    f"Daglig omsättning ~{daily_usd/1_000:.0f}k USD "
                    f"(pris {result.close:.2f} × volym {vol:,.0f}). "
                    + FLAG_DEFINITIONS["LOW_LIQUIDITY"]["detail"]
                ),
            ))

    # ── 📊 DATA_GAP ──────────────────────────────────────────────────────────
    has_gap, gap_suffix = _has_data_gap(result)
    if has_gap:
        flags.append(_make(
            "DATA_GAP",
            detail=FLAG_DEFINITIONS["DATA_GAP"]["detail"] + gap_suffix,
        ))

    # Sortera: danger → warning → positive → info
    _order = {"danger": 0, "warning": 1, "positive": 2, "info": 3}
    flags.sort(key=lambda f: _order.get(f.severity, 9))

    return flags


# ─── Rendering-helpers ────────────────────────────────────────────────────────

def format_flags_string(flags: list[Flag]) -> str:
    """
    Enkel emoji-sträng för tabellceller.
    Exempel: "⚠️ 📉 📊"
    """
    return "  ".join(f.emoji for f in flags) if flags else "—"


def format_flags_labels(flags: list[Flag]) -> str:
    """
    Etikett-sträng med emoji + label, kommaseparerad.
    Exempel: "⚠️ Value Trap?, 📉 Trending Down"
    """
    return ", ".join(str(f) for f in flags) if flags else "—"


def format_flags_html(flags: list[Flag]) -> str:
    """
    HTML-badges för st.markdown() i Streamlit.

    Färgschema per severity:
      positive → grön  (#00ff88 på mörk bakgrund)
      info     → blå   (#00bfff)
      warning  → orange (#ffaa00)
      danger   → röd   (#ff3355)
    """
    if not flags:
        return "<span style='color:#555;font-size:0.85em'>—</span>"

    _colors = {
        "positive": ("#003322", "#00ff88"),   # (bg, fg)
        "info":     ("#001833", "#00bfff"),
        "warning":  ("#332200", "#ffaa00"),
        "danger":   ("#330011", "#ff3355"),
    }
    parts: list[str] = []
    for f in flags:
        bg, fg = _colors.get(f.severity, ("#111", "#ccc"))
        parts.append(
            f'<span title="{f.detail}" style="'
            f"background:{bg};color:{fg};"
            f"border:1px solid {fg}33;"
            f"border-radius:4px;padding:1px 6px;"
            f'font-size:0.78em;white-space:nowrap;margin-right:4px">'
            f"{f.emoji} {f.label}</span>"
        )
    return "".join(parts)


def flags_to_dict_list(flags: list[Flag]) -> list[dict]:
    """
    Serialisera flaggor till list[dict] — för JSON-export eller pytest.
    """
    return [
        {
            "key":      f.key,
            "emoji":    f.emoji,
            "label":    f.label,
            "severity": f.severity,
            "detail":   f.detail,
        }
        for f in flags
    ]


# ─── Convenience: utvärdera direkt från pipeline-resultat ────────────────────

def attach_flags(results: list[ContrairianAlphaResult]) -> None:
    """
    In-place: lägg till utvärderade flaggor på varje ContrairianAlphaResult
    i en pipeline-resultatlista.

    Sätter result.ui_flags (list[Flag]) — ett extra attribut utanför
    dataklassen, läggs till dynamiskt för att undvika cirkelberoende.
    """
    for r in results:
        vol = getattr(r, "avg_volume_20d", None)
        r.ui_flags = evaluate_flags(r, avg_volume_20d=vol)  # type: ignore[attr-defined]


# ─── CLI diagnostics ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from contrarian_alpha.engine import ContrairianAlphaResult
    from contrarian_alpha.catalyst import CatalystResult
    from contrarian_alpha.strength import StrengthResult
    from contrarian_alpha.hate import HateResult
    from contrarian_alpha.necessity import get_necessity_score

    def _make_result(**kw) -> ContrairianAlphaResult:
        defaults = dict(
            ticker="TST", ins_id=None, name="Test Corp", market="SE",
            sector="Materials", branch="Copper", composite_score=72.0,
            necessity_score=92.0, hat_score=55.0, strength_score=68.0,
            catalyst_score=45.0, viking_bonus_raw=100.0,
            necessity_entry=get_necessity_score(gics_sub_industry=15104025),
            close=12.50, sma50=12.10, sma200=14.00,
            high_52w=18.0, low_52w=10.5,
            altman_z=2.85, data_confidence=0.85,
            timestamp="2026-04-19T00:00:00+00:00",
        )
        defaults.update(kw)
        return ContrairianAlphaResult(**defaults)

    # Build minimal sub-results
    _hate_full = HateResult(score=55.0, confidence=1.0)
    _hate_gap  = HateResult(score=55.0, confidence=0.43,
                            flags=["ANALYST_DATA_MISSING", "SHORT_DATA_MISSING"])
    _strength_ok   = StrengthResult(score=68.0, gate_results={})
    _strength_weak = StrengthResult(score=38.0, gate_results={})
    _cat_green   = CatalystResult(score=45.0, viking_regime_green=True,
                                  viking_regime_color="green", ovtlyr_nine=74)
    _cat_red     = CatalystResult(score=12.0, viking_regime_green=False,
                                  viking_regime_color="red",
                                  flags=["SMA50_SLOPE_MISSING"])
    _cat_orange  = CatalystResult(score=30.0, viking_regime_green=False,
                                  viking_regime_color="orange")

    scenarios = [
        {
            "_desc": "Idealfall — Viking grön, inga varningar",
            "r": _make_result(
                hat_score=58.0, strength_score=68.0, catalyst_score=45.0,
                catalyst_result=_cat_green, hate_result=_hate_full,
                strength_result=_strength_ok, altman_z=2.85,
            ),
            "vol": 5_200_000,
        },
        {
            "_desc": "Value Trap — hög hat, svag balansräkning",
            "r": _make_result(
                hat_score=88.0, strength_score=34.0, catalyst_score=18.0,
                catalyst_result=_cat_red, hate_result=_hate_full,
                strength_result=_strength_weak, altman_z=1.45,
            ),
            "vol": 320_000,
        },
        {
            "_desc": "Trending Down + Data Gap",
            "r": _make_result(
                hat_score=62.0, strength_score=55.0, catalyst_score=15.0,
                catalyst_result=_cat_red, hate_result=_hate_gap,
                strength_result=_strength_ok, altman_z=None,
            ),
            "vol": 1_800_000,
        },
        {
            "_desc": "Low Liquidity + Viking orange",
            "r": _make_result(
                hat_score=52.0, strength_score=61.0, catalyst_score=35.0,
                catalyst_result=_cat_orange, hate_result=_hate_full,
                strength_result=_strength_ok, close=0.38, altman_z=2.10,
            ),
            "vol": 280_000,
        },
        {
            "_desc": "Inga flaggor (perfekt bolag)",
            "r": _make_result(
                hat_score=62.0, strength_score=75.0, catalyst_score=88.0,
                catalyst_result=_cat_green, hate_result=_hate_full,
                strength_result=_strength_ok, close=35.0, altman_z=3.20,
            ),
            "vol": 8_000_000,
        },
    ]

    print(f"\n{'─'*72}")
    print("  FLAGS.PY — DIAGNOSTIK")
    print(f"{'─'*72}")
    print(f"  {'Scenario':<40}  {'Flaggor (string)':<28}  HTML")
    print(f"  {'─'*40}  {'─'*28}  {'─'*4}")

    for s in scenarios:
        desc  = s["_desc"]
        r     = s["r"]
        vol   = s["vol"]
        flags = evaluate_flags(r, avg_volume_20d=vol)
        fs    = format_flags_string(flags)
        fl    = format_flags_labels(flags)
        html_len = len(format_flags_html(flags))
        print(f"  {desc:<40}  {fs:<28}  {html_len}b HTML")
        if flags:
            for f in flags:
                print(f"    [{f.severity:<8}]  {f.emoji} {f.label:<14}  {f.detail[:65]}…")

    print(f"\n{'─'*72}")
    print("  Rendering examples:")
    r0 = scenarios[1]["r"]
    flags0 = evaluate_flags(r0, avg_volume_20d=scenarios[1]["vol"])
    print(f"  format_flags_string : {format_flags_string(flags0)!r}")
    print(f"  format_flags_labels : {format_flags_labels(flags0)!r}")
    print(f"  flags_to_dict_list  : {[d['key'] for d in flags_to_dict_list(flags0)]}")
    print(f"{'─'*72}\n")
