"""
review/missing.py — "Saknas för köpbeslut": fälten bakom de grindar som ännu inte går att pröva.

Härlett ur arkens egna utfall, aldrig ur egna trösklar: producer_score och asset_dying (Rick Rule),
royalty_signals tre kvoter, total_score/runway_years/mcap_per_earnings (Poängmodellen), tiggre.buy_gates,
insider.score/status/vs_cluster, controls.required_sections/ds_total/ds_blocks_buy/csm_complete, lukacs.evaluate,
scorecard.gate_state och Confidence-valideringens DATA_MISSING. Returnerar vilket fält som saknas och varför — en
grind som går att pröva men faller är inte en lucka utan ett svar, och listas inte här.

Rena funktioner, ingen Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import controls as ctl
import insider as ins
import lukacs
import producers as prod
import scorecard as sc
import scoring as sco
import tiggre as tig
from asymmetry import fields as afields
from confidence.config import FIELD_BY_KEY
from confidence.data import validation as cval
from confidence.data.models import CompanyInput
from review import catalog as cat


@dataclass(frozen=True)
class MissingItem:
    sheet: str
    path: str          # fältets sökväg i katalogen; tom när luckan inte är ett enskilt fält (t.ex. en annan rad)
    label: str
    reason: str
    gate: str


class _Out:
    """Samlar luckorna i ordning, ett fält en gång (första skälet vinner)."""

    def __init__(self, sheet: str, row: dict, labels: dict):
        self.sheet, self.row, self.labels = sheet, row, labels
        self.items: list = []
        self._seen: set = set()

    def add(self, path: str, reason: str, gate: str, label: str = "") -> None:
        key = path or f"·{label}·{reason}"
        if key in self._seen:
            return
        self._seen.add(key)
        self.items.append(MissingItem(self.sheet, path, label or self.labels.get(path, path), reason, gate))

    def blank(self, path: str) -> bool:
        return cat.is_empty(cat.get_value(self.row, path))


def _num(v) -> Optional[float]:
    return ctl._num(v)


def _need(out: _Out, keys, reason: str, gate: str, nonzero: tuple = ()) -> None:
    """Grindens indata som är tomma (eller noll där kvoten delar med dem)."""
    for k in keys:
        v = _num(cat.get_value(out.row, k))
        if v is None or (k in nonzero and v == 0):
            out.add(k, reason, gate)


# ── Kontrollerna ─────────────────────────────────────────────────────────────
def _controls(out: _Out, position_pct=None) -> None:
    sheet, row = out.sheet, out.row
    req = cat.sections(sheet, row, position_pct)
    if ctl.SEC_DS in req:
        unset = [f for f in ctl.DS_FIELDS if _num(row.get(f.key)) is None]
        why = ("DS är inte bedömd — luckor i tabellen är standardbeslutet INGEN AFFÄR."
               if ctl.ds_total(row) is None else "Delfrågan är obedömd och räknas som 0 riskpoäng.")
        for f in unset:
            out.add(f.key, why, "DS")
        if ctl.ds_blocks_buy(row):
            for k in ("fin_catalyst_text", "fin_catalyst_date"):
                if out.blank(k):
                    out.add(k, ctl.ds_note(row) or ctl.DS_BLOCK_TEXT, "DS")
    if ctl.SEC_AQS in req:
        why = (f"AQS är inte bedömd — krävs för positioner över {ctl.FULL_WORK_MIN_PCT:g} % av totalen."
               if ctl.aqs_total(row) is None else "Delfrågan är obedömd och räknas som 0 p.")
        for f in ctl.AQS_FIELDS:
            if _num(row.get(f.key)) is None:
                out.add(f.key, why, "AQS")
    if ctl.SEC_CSM in req:
        matrix = row.get("csm") if isinstance(row.get("csm"), dict) else {}
        if not ctl.csm_complete(matrix, bool(row.get("is_core"))):
            cols = (("price", "fcf_musd") if cat.csm_kind(sheet, row) == ctl.PRODUCER
                    else ("price", "nav_musd", "financing_need"))
            for s in ctl.scenarios(bool(row.get("is_core"))):
                sc_ = matrix.get(s) if isinstance(matrix.get(s), dict) else {}
                if not any(_num(sc_.get(k)) is not None for k in ("price", "fcf_musd", "nav_musd", "financing_need")):
                    for k in cols:
                        out.add(f"csm.{s}.{k}", "CSM är inte ifylld för alla scenarier.", "CSM")
    pos = cat.position(sheet, row, position_pct)
    if cat.fv_shown(sheet, row, position_pct) and lukacs.fv_required(pos, cat.SHEETS[sheet].strategy):
        _lukacs(out)


def _lukacs(out: _Out) -> None:
    """Köpgrindens steg 5 mekaniskt: fälten bakom det lukacs.evaluate inte kan räkna."""
    row, g = out.row, "Lukacs FV (köpgrindens steg 5)"
    ev = lukacs.evaluate(row)
    if ev["not_fcf_valued"]:
        return                                   # klass D: ett svar, inte en lucka
    if ev["fv_base"] is None:
        why = "Fair value för Base går inte att räkna — forward FCF, target-yield och aktieantal krävs."
        for path in ("fv.Base.forward_fcf_musd", "fv.Base.target_yield", "framtida_antal_aktier"):
            v = _num(cat.get_value(row, path))
            if v is None or (path != "fv.Base.forward_fcf_musd" and v <= 0):
                out.add(path, why, g)
    if ev["mos"] is None and _num(row.get("aktuell_kurs")) is None:
        out.add("aktuell_kurs", "Säkerhetsmarginalen kräver en kurs.", g)
    if not ev["what_must_go_right"]:
        out.add("what_must_go_right", lukacs.WMGR_MISSING + ".", g)
    if ev["delev"]["applies"] and ev["delev"]["years_ok"] is None:
        out.add("ar_till_lag_skuld", " ".join(ev["delev"]["gaps"]), g)


# ── Arken ────────────────────────────────────────────────────────────────────
def _rule(out: _Out, position_pct) -> None:
    row, g = out.row, "Rick Rule-poängen"
    if prod.producer_score(row) is None:
        _need(out, ("price", "unit_cost"), "Poängen går inte att räkna utan råvarupris och kostnad per enhet.", g,
              nonzero=("price",))
    if prod.asset_dying(row) is None:
        out.add("mine_life", "Strykregeln (döende tillgång) kan inte prövas — fyll i gruvlivslängd, eller R/P för "
                             "olja och gas.", "Strykregeln", label="Gruvlivslängd (eller R/P-kvot)")
    for k, _label, _h in prod.CHECKS:
        if row.get(k) is None:
            out.add(k, "Disciplinfrågan är obesvarad och räknas som 0 p.", g)
    if prod.ev_nav_verdict(row) is None:
        _need(out, ("ev_musd", "nav_musd"), "EV/NAV — Rules värdering bredvid poängen — går inte att räkna.",
              "EV/NAV", nonzero=("nav_musd",))
    if position_pct is None and _num(row.get("position_pct")) is None:
        out.add("position_pct", f"Positionen avgör vilka kontroller som krävs — över {ctl.FULL_WORK_MIN_PCT:g} % "
                                f"av totalen även AQS, CSM och Lukacs FV.", "Proportionalitetsregeln")
    _controls(out, position_pct)


def _royalty(out: _Out, position_pct) -> None:
    row, g = out.row, "Royalty-signalen"
    if prod.discount_vs_bottom(row.get("pnav_now"), row.get("pnav_bottom")) is None:
        _need(out, ("pnav_now", "pnav_bottom"), "Avståndet till egen P/NAV-botten går inte att räkna.", g,
              nonzero=("pnav_bottom",))
    if prod.vs_median(row.get("ev_now"), row.get("ev_median")) is None:
        _need(out, ("ev_now", "ev_median"), "EV/EBITDA mot egen median går inte att räkna.", g,
              nonzero=("ev_median",))
    if prod.geo_growth(row.get("geo_now"), row.get("geo_3y")) is None:
        _need(out, ("geo_now", "geo_3y"), "GEO-tillväxten per aktie går inte att räkna.", g, nonzero=("geo_3y",))
    _controls(out, position_pct)


def _factors(out: _Out, factors: dict, keys, total_none: bool) -> None:
    why = ("Ej poängsatt — bedömningen kräver faktorerna." if total_none
           else "Faktorn är inte satt och räknas som 0 p.")
    for k in keys:
        if _num((factors or {}).get(k)) is None:
            out.add(f"factors.{k}", why, "Fem faktorer")


def _scoring(out: _Out, position_pct) -> None:
    row = out.row
    fac = row.get("factors") if isinstance(row.get("factors"), dict) else {}
    _factors(out, fac, [f.key for f in sco.FACTORS], sco.total_score(fac) is None)
    if out.sheet == "sprott":
        if sco.runway_years(row.get("cash"), row.get("burn")) is None:
            _need(out, ("cash", "burn"), f"Sprott-grinden (runway minst {sco.SPROTT_RUNWAY_MIN_MONTHS} mån) kan inte "
                                         f"prövas.", "Runway-grinden", nonzero=("burn",))
    elif sco.mcap_per_earnings(row.get("mcap"), row.get("profit")) is None:
        _need(out, ("mcap", "profit"), f"Köpregeln (under {sco.DURRETT_BUY_MAX:g}× framtida vinst) kan inte prövas — "
                                       f"Kärninnehav kräver den.", "Köpregeln", nonzero=("profit",))
    _controls(out, position_pct)


# tiggre.buy_gates ordning (arkets): grovsållning, U/N, poäng, katalysatorer, DS, CSM
_TG_SCREEN, _TG_UN, _TG_SCORE, _TG_CATS = 0, 1, 2, 3


def _tiggre(out: _Out, position_pct) -> None:
    row = out.row
    gates = tig.buy_gates(row)
    label, passed, detail = gates[_TG_SCREEN]
    if not passed:
        screen = row.get("screen") if isinstance(row.get("screen"), dict) else {}
        for k, _l in tig.SCREEN_PHRASES:
            if screen.get(k) is None:
                out.add(f"screen.{k}", f"{label}: {detail}.", "Grovsållning")
    label, passed, detail = gates[_TG_UN]
    if not passed and tig.un_ratio(tig.upside_pct(row.get("mcap"), row.get("nav")), row.get("downside")) is None:
        for k in ("mcap", "nav", "downside"):
            v = _num(row.get(k))
            if v is None or v == 0 or (k != "downside" and v < 0):
                out.add(k, f"{label} kan inte prövas ({detail}).", "U/N-kvoten")
    label, passed, detail = gates[_TG_SCORE]
    if not passed:
        fac = row.get("factors") if isinstance(row.get("factors"), dict) else {}
        for k, _l, _h in tig.FACTORS:
            if k != "un" and _num(fac.get(k)) is None:
                out.add(f"factors.{k}", f"Faktorn är inte satt och räknas som 0 p ({label}: {detail}).",
                        "Fem faktorer")
    label, passed, detail = gates[_TG_CATS]
    if not passed:
        cats = row.get("catalysts") if isinstance(row.get("catalysts"), list) else []
        need = tig.CATALYSTS_MIN - sum(1 for c in cats if isinstance(c, dict) and c.get("name") and c.get("date"))
        i = 0
        while need > 0:
            c = cats[i] if i < len(cats) and isinstance(cats[i], dict) else {}
            if not (c.get("name") and c.get("date")):
                for part in ("name", "date"):
                    if not c.get(part):
                        out.add(f"catalysts.{i}.{part}", f"{label}: {detail}.", "Katalysatorer")
                need -= 1
            i += 1
    _controls(out, position_pct)


def _insider(out: _Out, position_pct) -> None:
    row = out.row
    if ins.score(row) is None:
        why = "Poängen kräver antal insiders, roll och belopp."
        _need(out, ("insiders", "amount"), why, "Insiderpoängen")
        if not row.get("role"):
            out.add("role", why, "Insiderpoängen")
    stat = ins.status(row)
    if stat == ins.S_RUN_GATE:
        out.add("gate", f"{ins.S_RUN_GATE} — poängen räcker, grinden är inte körd.", "Kvalitetsgrinden")
    elif stat == ins.S_WAIT_TRIGGER:
        out.add("trigger", f"{ins.S_WAIT_TRIGGER}.", "Teknisk trigger")
    if ins.vs_cluster(row) is None:
        _need(out, ("cluster_avg", "price_now"), f"Passa-regeln (mer än +{ins.CHASE_PCT:g} % mot klustersnittet) "
                                                 f"kan inte prövas.", "Passa-regeln", nonzero=("cluster_avg",))
    _controls(out, position_pct)


def _scorecard(out: _Out, entry: Optional[dict]) -> None:
    card = sc._shown(out.row, entry) if entry else dict(out.row)
    state = sc.gate_state(card, entry)
    g = "Köpgrinden"
    for gt in sc.GATES:
        if state["checks"].get(gt.key):
            continue
        if gt.key == "inga_roda_flaggor":
            for gap in state["gaps"]:
                out.add("", gap, g, label=gt.label)
        elif gt.key == "sakerhetsmarginal" and state["fv_mechanical"]:
            for gap in state["valuation_gaps"]:
                out.add("", gap, g, label=gt.label)
        elif gt.key == "position_saljregel":
            for k in ("position_pct_total", "sell_rule"):
                if out.blank(k):
                    out.add(k, "Storleken och exitregeln bestäms före köpet.", g)
            for gap in state["position_gaps"]:
                out.add("", gap, g, label=gt.label)
            if not card.get(gt.key):
                out.add(gt.key, gt.help, g)
        elif gt.key == "trigger_definierad":
            if out.blank("trigger"):
                out.add("trigger", gt.help, g)
            out.add(gt.key, gt.help, g)
        else:
            out.add(gt.key, gt.help, g)


def _company(row) -> CompanyInput:
    if isinstance(row, CompanyInput):
        return row
    d = dict(cat._as_row(row))
    d.setdefault("ticker", "")                  # en tom mall har ingen ticker än
    return CompanyInput.from_dict(d)


def _confidence(out: _Out, stage: Optional[str]) -> None:
    company = _company(out.row)
    if stage:
        company.stage = stage
    for issue in cval.validate(company):
        if issue.level != cval.MISSING:
            continue
        if issue.field in FIELD_BY_KEY:
            out.add(f"fields.{issue.field}", issue.message, "Confidence (DATA_MISSING)")
        else:
            out.add("", issue.message, "Confidence (DATA_MISSING)", label=issue.field)


def _asymmetry(out: _Out, stage: Optional[str]) -> None:
    company = _company(out.row)
    st_ = stage or company.stage
    for spec in afields.economy(st_) + afields.confidence_core(st_):
        if not company.has(spec.key):
            out.add(f"fields.{spec.key}", f"{spec.label} saknas — poängen läser fältet (DATA_MISSING).",
                    "Wolf Asymmetry")


def missing_for(sheet: str, row, entry: Optional[dict] = None, stage: Optional[str] = None,
                position_pct=None) -> list:
    """[MissingItem] för kortet: fälten bakom de grindar som inte går att pröva, i arkets ordning.

    Scorecardet: row är kortet och entry kandidatraden (scorecard.collect). position_pct är scorecardets
    position — då följer kontrollerna köpgrinden i stället för kortets egen storlek.
    """
    r = cat._as_row(row)
    strategies = tuple((entry or {}).get("strategies") or ())
    labels = {f.path: f.label for f in cat.fields_for(sheet, r, stage, position_pct, strategies)}
    out = _Out(sheet, r, labels)
    if sheet == "rule":
        _rule(out, position_pct)
    elif sheet == "royalty":
        _royalty(out, position_pct)
    elif sheet in ("sprott", "durrett"):
        _scoring(out, position_pct)
    elif sheet == "tiggre":
        _tiggre(out, position_pct)
    elif sheet == "insider":
        _insider(out, position_pct)
    elif sheet == "scorecard":
        _scorecard(out, entry)
    elif sheet == "confidence":
        _confidence(out, stage)
    elif sheet == "asymmetry":
        _asymmetry(out, stage)
    else:
        raise KeyError(f"okänt ark: {sheet}")
    return out.items
