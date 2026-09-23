"""
scoring.py — Poängmodellen (ersätter poangmodell_sprott_durrett.xlsx).

Samma fem faktorer 0–2 för båda strategierna, olika räknehjälp:

  Sprott  — runway = kassa / burn per år. Faktor 1 kan föreslås ur runwayen,
            men får skrivas över: siffran är ett besked om nyemission, inte
            ett omdöme om bolaget.
  Durrett — MCap/uns = börsvärde / Moz AuEq, och MCap/framtida vinst, där
            köpregeln går vid 10x.

Arkets formler:
  Sprott  F: =IF(OR(D="",E="",E=0),"",D/E)
          L: =IF(COUNT(G:K)=0,"",SUM(G:K))
          M: =IF(L="","",IF(L>=8,"Kärninnehav",IF(L>=6,"Bevakningslista","Passa")))
  Durrett F: =IF(OR(D="",E="",E=0),"",D/E)
          H: =IF(OR(D="",G="",G=0),"",D/G)

Modellen är metallagnostisk — samma fem faktorer fungerar för uran och koppar
som för guld, därav råvarufältet på varje rad.
"""

from __future__ import annotations

import streamlit as st
from dataclasses import dataclass
from datetime import date
from typing import Optional

import csv_export
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

_CACHE_KEY = "scoring_data"
STORE = "scoring"   # data/scoring.json

TEXT, DIM = "#e8e4dc", "#8a8578"
GREEN, AMBER, RED, CYAN, GOLD = "#2d8a4e", "#d4943a", "#c44545", "#00E5FF", "#c9a84c"
BG_ALT, BORDER = "#1a1f25", "#2a2a38"

SPROTT, DURRETT = "sprott", "durrett"

# ── Bedömningen (kolumn M/O) ─────────────────────────────────────────────────
CORE = "Kärninnehav"
WATCH = "Bevakningslista"
PASS = "Passa"
VERDICT_COLOR = {CORE: GREEN, WATCH: AMBER, PASS: DIM}
CORE_MIN, WATCH_MIN, MAX_SCORE = 8, 6, 10

VERDICT_ACTION = {
    CORE: "Full position enligt strategins storlek.",
    WATCH: "Köp vid dipp eller katalysator — inte i dag.",
    PASS: "Ompröva nästa kvartal.",
}

# ── De fem faktorerna (Kriterier-bladet, ordagrant) ──────────────────────────
@dataclass(frozen=True)
class Factor:
    key: str
    label: str
    two: str
    one: str
    zero: str


FACTORS: tuple[Factor, ...] = (
    Factor("balans", "1. Balansräkning & runway",
           "Nettokassa, runway > 24 mån", "Runway 12–24 mån",
           "Runway < 12 mån (emission väntar)"),
    Factor("vardering", "2. Värdering",
           "< $50/uns eller EV/EBITDA < 4", "$50–150/uns eller EV/EBITDA 4–6",
           "Dyrare"),
    Factor("tillvaxt", "3. Tillväxt & projekt",
           "Växande produktion eller bygge pågår", "PEA/PFS klar",
           "Ren prospektering"),
    Factor("agare", "4. Ägare & management",
           "Insyn > 10 % eller Sprott/Van Eck bland ägarna",
           "Visst insynsägande, ok historik",
           "Lågt insynsägande, serieutspädare"),
    Factor("havstang", "5. Hävstång mot metallpris",
           "Vinst 3x+ vid guld +50 %", "Vinst ca 2x",
           "Låg hävstång (hedgad/högkostnad)"),
)

# ── Räknehjälpen ─────────────────────────────────────────────────────────────
RUNWAY_STRONG, RUNWAY_WEAK = 2.0, 1.0      # år: > 2 -> 2p, 1–2 -> 1p
DURRETT_BUY_MAX = 10.0                     # MCap/framtida vinst under 10x
MCAP_PER_OZ_CHEAP, MCAP_PER_OZ_VERY = 100.0, 50.0

# Strategiernas hårda grindar ovanpå poängen (Masterguiden, strategy_rules):
#   Sprott  — "Runway ≥ 18 månader. Under 18 månader = stopp oavsett projekt."
#   Durrett — "Poängsätt i Poängmodellen — krav ≥ 8 OCH under 10x."
# Poängen räknas som förut; grinden avgör bara vad poängen får kallas.
SPROTT_RUNWAY_MIN_MONTHS = 18

POSITION_NOTE = {
    SPROTT: "Max 1–1,5 % per bolag (tak 1,5 %), 10–15 bolag i korgen.",
    DURRETT: "Räknas mot Durrett-ramen i allokeringsplanen (mål 8 %, tak 3 % "
             "per bolag).",
}


# ── Rena beräkningar ─────────────────────────────────────────────────────────
def _num(value, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def runway_years(cash, burn) -> Optional[float]:
    """Kassa ÷ årsburn. Arket: =IF(OR(D="",E="",E=0),"",D/E)."""
    c, b = _num(cash), _num(burn)
    if c is None or b is None or b == 0:
        return None
    return c / b


def runway_points(years) -> Optional[int]:
    """Förslag till faktor 1 ur runwayen. Överskrivbart — därav Optional."""
    y = _num(years)
    if y is None:
        return None
    if y > RUNWAY_STRONG:
        return 2
    return 1 if y >= RUNWAY_WEAK else 0


def mcap_per_oz(mcap, moz) -> Optional[float]:
    """Börsvärde per uns guldekvivalent, i dollar. MUSD ÷ Moz = $/oz."""
    m, o = _num(mcap), _num(moz)
    if m is None or o is None or o == 0:
        return None
    return m / o


def mcap_per_earnings(mcap, future_profit) -> Optional[float]:
    """Durretts köpregel: under 10x framtida vinst."""
    m, p = _num(mcap), _num(future_profit)
    if m is None or p is None or p == 0:
        return None
    return m / p


def future_profit(production, target_price, aisc) -> Optional[float]:
    """Hjälpräknaren: produktion × (målpris − AISC)."""
    prod, tp, cost = _num(production), _num(target_price), _num(aisc)
    if prod is None or tp is None or cost is None:
        return None
    return prod * (tp - cost)


def total_score(factors: dict) -> Optional[int]:
    """Summan av de fem faktorerna, eller None när ingen är satt.

    Arket: =IF(COUNT(G:K)=0,"",SUM(G:K)) — en tom rad har inte noll poäng,
    den har inget betyg alls.
    """
    scored = [f for f in FACTORS if _num(factors.get(f.key)) is not None]
    if not scored:
        return None
    return sum(max(0, min(2, int(_num(factors.get(f.key), 0) or 0)))
               for f in FACTORS)


def verdict(score: Optional[int]) -> Optional[str]:
    if score is None:
        return None
    if score >= CORE_MIN:
        return CORE
    return WATCH if score >= WATCH_MIN else PASS


def durrett_buy_ok(ratio: Optional[float]) -> bool:
    """Köpregeln: MCap/framtida vinst under 10x."""
    r = _num(ratio)
    return r is not None and r < DURRETT_BUY_MAX


def sprott_runway_ok(years) -> Optional[bool]:
    """Sprotts hårda grind: runway ≥ 18 mån. None när runwayen är okänd."""
    y = _num(years)
    if y is None:
        return None
    return y * 12.0 >= SPROTT_RUNWAY_MIN_MONTHS


def gated_verdict(key: str, row: dict) -> tuple:
    """(bedömning, orsak) — poängens bedömning efter strategins hårda grind.

    Sprott: runway under 18 mån → Passa, oavsett poäng. Durrett: Kärninnehav
    kräver att köpregeln (under 10× framtida vinst) är räknad och uppfylld;
    annars stannar raden på Bevakningslista. Orsaken är tom när grinden inte
    slog. Poängen i sig ändras aldrig — bara etiketten.
    """
    r = row or {}
    vd = verdict(total_score(r.get("factors", {})))
    if vd is None:
        return None, ""
    if key == SPROTT:
        if sprott_runway_ok(runway_years(r.get("cash"), r.get("burn"))) is False:
            return PASS, (f"Runway under {SPROTT_RUNWAY_MIN_MONTHS} mån — "
                          f"avstå oavsett poäng (Sprott-regeln).")
    elif key == DURRETT and vd == CORE:
        ratio = mcap_per_earnings(r.get("mcap"), r.get("profit"))
        if ratio is None:
            return WATCH, (f"Kärninnehav kräver köpregeln under {DURRETT_BUY_MAX:g}× "
                           f"framtida vinst — fyll i börsvärde och framtida vinst.")
        if not durrett_buy_ok(ratio):
            return WATCH, (f"{ratio:.1f}× framtida vinst — Kärninnehav kräver "
                           f"under {DURRETT_BUY_MAX:g}×.")
    return vd, ""


def ranked(rows: list, key: Optional[str] = None) -> list:
    """Kandidaterna med poäng och bedömning, bäst först.

    Med `key` (sprott/durrett) läggs strategins hårda grind på bedömningen
    och `gate` bär orsaken; utan nyckel är bedömningen ren poäng som förut.
    """
    out = []
    for r in rows or []:
        sc = total_score(r.get("factors", {}))
        vd, why = (gated_verdict(key, r) if key else (verdict(sc), ""))
        out.append({"row": r, "score": sc, "verdict": vd, "gate": why})
    out.sort(key=lambda x: (-(x["score"] if x["score"] is not None else -1),
                            (x["row"].get("ticker") or "")))
    return out


# ── Lagring ──────────────────────────────────────────────────────────────────
def _today() -> str:
    return date.today().isoformat()


def _uid() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


def _default() -> dict:
    return {SPROTT: [], DURRETT: []}


def _load() -> dict:
    """Laddas EN gång per session; därefter äger sessionen sanningen."""
    data = storage.session_load(STORE, _default(),
                                legacy_file="scoring_data.json")
    if not isinstance(data, dict):
        data = _default()
        st.session_state[STORE] = data
    for k, v in _default().items():
        data.setdefault(k, v)
    return data


def _save(data: dict) -> None:
    """Skriver till sessionen. Persistensen sker via 💾 Spara.

    Medvetet ingen nätverkstrafik här: en commit per
    tangenttryckning skulle slå i GitHubs rate limits, och en
    tyst misslyckad sådan var precis den bugg det här ersätter.
    """
    st.session_state[STORE] = data


# ── UI ───────────────────────────────────────────────────────────────────────
def render_scoring_page() -> None:
    data = _load()
    storage_ui.save_bar(STORE, "Poängmodellen")
    page_header("Poängmodell", f"Fem faktorer 0–2. {CORE_MIN}–10 = kärninnehav · "
                f"{WATCH_MIN}–7 = bevakningslista · 0–{WATCH_MIN - 1} = passa. "
                f"Durrett-delen är snabbpoängen — hela 10-stegsmetoden ligger i "
                f"GRANSKNING → 🧭 Durrett & Confidence.")

    which = st.radio("Modell", ["Sprott (optionalitet)", "Durrett (snabbpoäng)"],
                     horizontal=True, key="sc_which",
                     label_visibility="collapsed")
    key = SPROTT if which.startswith("Sprott") else DURRETT
    st.caption(POSITION_NOTE[key])

    _export(data, key)
    _new_row(data, key)
    _screen_section(data, key)
    _rows(data, key)
    _criteria()


def _extract_section(data: dict, row: dict, key: str) -> None:
    """Copilot-extraktion ur rapporten: kassa/burn (Sprott), uns/produktion/
    AISC (Durrett) — som förslag med sida och citat. Faktorerna sätter du."""
    try:
        import extract_ui
    except Exception:
        return
    rid = row["id"]
    keys = ({"cash": f"sc_cash_{rid}", "burn": f"sc_burn_{rid}"} if key == SPROTT
            else {"moz": f"sc_moz_{rid}", "prod": f"sc_prod_{rid}", "aisc": f"sc_aisc_{rid}"})
    extract_ui.render_extractor(key, row, widget_keys=keys, on_apply=lambda: _save(data))


def _screen_section(data: dict, key: str) -> None:
    """Håvens träffar (screens_scan.py: sprott/durrett) med "lägg in i arket"."""
    try:
        import screens_ui
    except Exception:
        return
    existing = {str(r.get("ticker", "")).upper() for r in data.get(key, [])}

    def _add(fields: dict) -> None:
        data[key].append({"id": _uid(), "commodity": COMMODITIES[0],
                          "date": _today(), "factors": {}, **fields})
        _save(data)

    screens_ui.render_screen_section(key, existing, _add, key_prefix=f"sc_{key}")


CSV_COMMON = [("date", "Datum"), ("ticker", "Ticker"), ("name", "Bolag"),
              ("commodity", "Råvara")]
CSV_FACTORS = [(f.key, f.label) for f in FACTORS]
CSV_TAIL = ([("_score", "Totalt (0–10)"), ("_verdict", "Bedömning")]
            + [(f.key, f"DS {f.label}") for f in ctl.DS_FIELDS]
            + [("_ds", "DS totalt"), ("_ds_band", "DS-band"),
               ("comment", "Kommentar")])
CSV_COLUMNS = {
    SPROTT: CSV_COMMON + [("cash", "Kassa (MUSD)"), ("burn", "Burn/år (MUSD)"),
                          ("_runway", "Runway (år)")] + CSV_FACTORS + CSV_TAIL,
    DURRETT: CSV_COMMON + [("mcap", "Börsvärde (MUSD)"), ("moz", "Uns (Moz AuEq)"),
                           ("_per_oz", "MCap/uns ($)"),
                           ("profit", "Framtida vinst/år (MUSD)"),
                           ("_ratio", "MCap/framtida vinst")]
             + CSV_FACTORS + CSV_TAIL,
}


def _export(data: dict, key: str) -> None:
    rows = []
    for r in ranked(data.get(key, []), key):
        row = r["row"]
        ds = ctl.ds_total(row)
        extra = {"_score": r["score"], "_verdict": r["verdict"],
                 "_ds": ds, "_ds_band": ctl.ds_band(ds),
                 **(row.get("factors") or {})}
        if key == SPROTT:
            rw = runway_years(row.get("cash"), row.get("burn"))
            extra["_runway"] = None if rw is None else round(rw, 2)
        else:
            po = mcap_per_oz(row.get("mcap"), row.get("moz"))
            ra = mcap_per_earnings(row.get("mcap"), row.get("profit"))
            extra["_per_oz"] = None if po is None else round(po, 1)
            extra["_ratio"] = None if ra is None else round(ra, 2)
        rows.append({**row, **extra})
    csv_export.download_button(rows, CSV_COLUMNS[key], f"poangmodell_{key}",
                               key=f"csv_scoring_{key}")


def _new_row(data: dict, key: str) -> None:
    with st.expander("➕ Ny kandidat", expanded=not data.get(key)):
        c1, c2, c3 = st.columns([1, 2, 1.2])
        ticker = c1.text_input("Ticker", key=f"sc_new_t_{key}")
        name = c2.text_input("Bolag", key=f"sc_new_n_{key}")
        commodity = c3.selectbox("Råvara", list(COMMODITIES),
                                 key=f"sc_new_c_{key}")
        if st.button("Lägg till", key=f"sc_new_add_{key}"):
            if ticker.strip():
                data[key].append({
                    "id": _uid(), "ticker": ticker.strip().upper(),
                    "name": name.strip(), "commodity": commodity,
                    "date": _today(), "factors": {},
                })
                _save(data)
                st.rerun()
            else:
                st.warning("Ticker krävs.")


def _rows(data: dict, key: str) -> None:
    rows = ranked(data.get(key, []), key)
    if not rows:
        st.caption("Inga kandidater ännu. Screenern körs i Börsdata — filtret "
                   "står i RULES → 📚 SNABBREFERENS.")
        return

    for r in rows:
        row, sc, vd, gate_why = r["row"], r["score"], r["verdict"], r["gate"]
        c = VERDICT_COLOR.get(vd, DIM)
        head = (f"{row.get('ticker','?')} · {row.get('commodity','')} · "
                f"{sc if sc is not None else '–'}/{MAX_SCORE} p · "
                f"{vd or 'ej poängsatt'}")
        with st.expander(head, expanded=False):
            if key == SPROTT:
                _sprott_math(data, row)
                _extract_section(data, row, SPROTT)
            else:
                _durrett_math(data, row)
                _extract_section(data, row, DURRETT)
            _factors(data, row, key)

            # DS (Masterguiden 4.0) — utspädningen är den här strategifamiljens
            # största enskilda risk, så den frågas alltid, oavsett storlek.
            rw = (runway_years(row.get("cash"), row.get("burn"))
                  if key == SPROTT else None)
            with st.expander("🧪 DS — utspädningsrisk", expanded=False):
                if controls_ui.render_ds(row, row["id"], rw):
                    _save(data)

            blocked = ctl.ds_blocks_buy(row)
            eff_c, eff_vd = (RED, "KÖP LÅST — hög DS") if blocked else (c, vd)
            note = ctl.ds_note(row) if blocked else VERDICT_ACTION.get(vd, "")
            if gate_why and not blocked:
                note = f"{gate_why} {note}".strip()
            st.markdown(
                f"<div style='border:1px solid {eff_c}55;background:{eff_c}0d;"
                f"border-radius:8px;padding:10px 14px;margin:10px 0;'>"
                f"<span style='color:{eff_c};font-weight:700;font-size:1.05rem;'>"
                f"{sc if sc is not None else '–'} / {MAX_SCORE}</span>"
                f"<span style='color:{eff_c};font-weight:700;margin-left:12px;'>"
                f"{eff_vd or 'Sätt minst en faktor'}</span>"
                f"<div style='color:{TEXT};font-size:0.8rem;margin-top:3px;'>"
                f"{note}"
                f"</div></div>", unsafe_allow_html=True)

            com = st.text_input("Kommentar", value=row.get("comment", ""),
                                key=f"sc_com_{row['id']}")
            if com != row.get("comment", ""):
                row["comment"] = com
                _save(data)
            if confirm_delete("Ta bort", key=f"sc_del_{row['id']}"):
                data[key] = [x for x in data[key] if x["id"] != row["id"]]
                _save(data)
                st.rerun()


def _sprott_math(data: dict, row: dict) -> None:
    st.markdown(f"<b style='color:{TEXT};'>Runway</b> "
                f"<span style='color:{DIM};font-size:0.78rem;'>"
                f"— kassa och burn från senaste kvartalsrapporten</span>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    cash = c1.number_input("Kassa (MUSD)", min_value=0.0, step=1.0,
                           value=_num(row.get("cash")),
                           key=f"sc_cash_{row['id']}")
    burn = c2.number_input("Burn/år (MUSD)", min_value=0.0, step=1.0,
                           value=_num(row.get("burn")),
                           key=f"sc_burn_{row['id']}")
    rw = runway_years(cash, burn)
    c3.metric("Runway", f"{rw:.1f} år" if rw is not None else "–",
              help=f"Under {SPROTT_RUNWAY_MIN_MONTHS} månader = stopp oavsett "
                   f"projekt — nyemissionen som kommer äter din uppsida.")
    # Kassa och burn ur Börsdatas rapporter (sifferuppdateringen): kassa ur
    # senaste rapporten, burn = negativt FCF rullande 12 mån. Förslag, "Använd".
    try:
        import refresh_ui
        refresh_ui.suggest("scoring", row, "cash", "cash_musd", "kassa (MUSD)",
                           f"sc_cash_{row['id']}", lambda: _save(data), fmt="{:,.1f}")
        refresh_ui.suggest("scoring", row, "burn", "burn_musd", "burn/år (MUSD)",
                           f"sc_burn_{row['id']}", lambda: _save(data), fmt="{:,.1f}")
    except Exception:
        pass
    if (storage.differs(cash, row.get("cash"))
            or storage.differs(burn, row.get("burn"))):
        row["cash"], row["burn"] = cash, burn
        _save(data)

    ok = sprott_runway_ok(rw)
    if ok is not None:
        c_ok = GREEN if ok else RED
        st.markdown(
            f"<div style='color:{c_ok};font-size:0.82rem;'>"
            f"{'✓' if ok else '✕'} Runway-grinden: {rw * 12:.0f} mån "
            f"{'≥' if ok else '<'} {SPROTT_RUNWAY_MIN_MONTHS} mån"
            f"{'' if ok else ' — avstå oavsett poäng'}</div>",
            unsafe_allow_html=True)
    pts = runway_points(rw)
    if pts is not None:
        st.caption(f"Runwayen motsvarar {pts} p på faktor 1 — förslag, inte "
                   f"facit. Nettokassa och emissionshistorik kan flytta den.")


def _durrett_math(data: dict, row: dict) -> None:
    st.markdown(f"<b style='color:{TEXT};'>Värderingen</b> "
                f"<span style='color:{DIM};font-size:0.78rem;'>"
                f"— uns ur presentationen, vinst ur hjälpräknaren</span>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    mcap = c1.number_input("Börsvärde (MUSD)", min_value=0.0, step=10.0,
                           value=_num(row.get("mcap")),
                           key=f"sc_mcap_{row['id']}")
    moz = c2.number_input("Uns (Moz AuEq)", min_value=0.0, step=0.1,
                          value=_num(row.get("moz")),
                          key=f"sc_moz_{row['id']}")
    profit = c3.number_input("Framtida vinst/år (MUSD)", min_value=0.0, step=5.0,
                             value=_num(row.get("profit")),
                             key=f"sc_profit_{row['id']}")
    if (storage.differs(mcap, row.get("mcap"))
            or storage.differs(moz, row.get("moz"))
            or storage.differs(profit, row.get("profit"))):
        row["mcap"], row["moz"], row["profit"] = mcap, moz, profit
        _save(data)
    try:
        import refresh_ui
        refresh_ui.suggest("scoring", row, "mcap", "mcap_musd", "börsvärde (MUSD)",
                           f"sc_mcap_{row['id']}", lambda: _save(data), fmt="{:,.0f}")
    except Exception:
        pass

    per_oz = mcap_per_oz(mcap, moz)
    ratio = mcap_per_earnings(mcap, profit)
    m1, m2 = st.columns(2)
    m1.metric("MCap/uns", f"${per_oz:,.0f}" if per_oz is not None else "–",
              help=f"Under ${MCAP_PER_OZ_CHEAP:g} = billigt, under "
                   f"${MCAP_PER_OZ_VERY:g} = mycket billigt.")
    m2.metric("MCap/framtida vinst",
              f"{ratio:.1f}×" if ratio is not None else "–",
              help=f"Durretts köpregel: under {DURRETT_BUY_MAX:g}×.")
    if ratio is not None:
        ok = durrett_buy_ok(ratio)
        c = GREEN if ok else RED
        st.markdown(
            f"<div style='color:{c};font-size:0.82rem;'>"
            f"{'✓' if ok else '✕'} Köpregeln: {ratio:.1f}× "
            f"{'under' if ok else 'över'} {DURRETT_BUY_MAX:g}×"
            f"{'' if ok else ' — Kärninnehav kräver under ' + format(DURRETT_BUY_MAX, 'g') + '×'}"
            f"</div>",
            unsafe_allow_html=True)
    else:
        st.caption(f"Köpregeln är inte räknad — utan MCap/framtida vinst stannar "
                   f"raden på {WATCH} även vid {CORE_MIN}+ poäng.")

    with st.expander("🧮 Hjälpräknare — framtida vinst", expanded=False):
        h1, h2, h3 = st.columns(3)
        prod = h1.number_input("Produktion (koz/år)", min_value=0.0, step=10.0,
                               value=_num(row.get("prod")),
                               key=f"sc_prod_{row['id']}")
        tp = h2.number_input("Målpris ($/oz)", min_value=0.0, step=50.0,
                             value=float(_num(row.get("target"), 3000.0) or 3000.0),
                             key=f"sc_tp_{row['id']}")
        aisc = h3.number_input("AISC ($/oz)", min_value=0.0, step=50.0,
                               value=_num(row.get("aisc")),
                               key=f"sc_aisc_{row['id']}")
        fp = future_profit(prod, tp, aisc)
        if fp is not None:
            # koz × $/oz = tusen dollar -> MUSD
            musd = fp / 1000.0
            st.caption(f"{prod:,.0f} koz × (${tp:,.0f} − ${aisc:,.0f}) = "
                       f"{musd:,.0f} MUSD per år.")
            if st.button("Använd som framtida vinst", key=f"sc_use_{row['id']}"):
                row["profit"] = round(musd, 1)
                _save(data)
                st.rerun()
        if (storage.differs(prod, row.get("prod"))
                or storage.differs(tp, row.get("target"))
                or storage.differs(aisc, row.get("aisc"))):
            row["prod"], row["target"], row["aisc"] = prod, tp, aisc
            _save(data)


def _factors(data: dict, row: dict, key: str) -> None:
    st.markdown(f"<div style='height:6px;'></div>"
                f"<b style='color:{TEXT};'>Fem faktorer (0–2)</b>",
                unsafe_allow_html=True)
    fac = row.setdefault("factors", {})
    changed = False
    cols = st.columns(5)
    opts = ["–", 0, 1, 2]
    for f, col in zip(FACTORS, cols):
        cur = _num(fac.get(f.key))
        idx = opts.index(int(cur)) if cur is not None and int(cur) in (0, 1, 2) else 0
        v = col.selectbox(f.label.split(". ", 1)[-1], opts, index=idx,
                          key=f"sc_f_{row['id']}_{f.key}",
                          help=f"2: {f.two}\n\n1: {f.one}\n\n0: {f.zero}")
        new = None if v == "–" else int(v)
        if new != (int(cur) if cur is not None else None):
            if new is None:
                fac.pop(f.key, None)
            else:
                fac[f.key] = new
            changed = True
    if changed:
        _save(data)


def _criteria() -> None:
    with st.expander("📋 Poängkriterier — 0–2 per faktor, max 10",
                     expanded=False):
        html = ("<table style='width:100%;border-collapse:collapse;'>"
                f"<tr style='border-bottom:1px solid {GOLD}33;'>"
                f"<th style='text-align:left;color:{GOLD};font-size:0.7rem;"
                f"padding:6px;'>FAKTOR</th>"
                f"<th style='text-align:left;color:{GREEN};font-size:0.7rem;"
                f"padding:6px;'>2 POÄNG</th>"
                f"<th style='text-align:left;color:{AMBER};font-size:0.7rem;"
                f"padding:6px;'>1 POÄNG</th>"
                f"<th style='text-align:left;color:{RED};font-size:0.7rem;"
                f"padding:6px;'>0 POÄNG</th></tr>")
        from html import escape as _esc
        for f in FACTORS:
            html += (f"<tr style='border-bottom:1px solid rgba(138,133,120,0.15);'>"
                     f"<td style='color:{TEXT};font-size:0.75rem;padding:6px;"
                     f"font-weight:700;'>{_esc(f.label)}</td>"
                     f"<td style='color:{DIM};font-size:0.73rem;padding:6px;'>"
                     f"{_esc(f.two)}</td>"
                     f"<td style='color:{DIM};font-size:0.73rem;padding:6px;'>"
                     f"{_esc(f.one)}</td>"
                     f"<td style='color:{DIM};font-size:0.73rem;padding:6px;'>"
                     f"{_esc(f.zero)}</td></tr>")
        st.markdown(html + "</table>", unsafe_allow_html=True)
        st.caption(f"{CORE_MIN}–10 = kärninnehav, full position · "
                   f"{WATCH_MIN}–7 = bevakningslista, köp vid dipp eller "
                   f"katalysator · 0–{WATCH_MIN - 1} = passa, ompröva nästa "
                   f"kvartal.")
