"""
review/catalog — fältkatalogen byggs ur arkens egna konstanter.

Varje sökväg finns i sin källa (arkets modul, controls, lukacs, confidence.config), get/set går fram och tillbaka
för alla fält, inga dubbletter, och besluten (position, köpgrind, säljregel, kvalitetsgrind, trigger, Bear-
överlevnad) är "Bara du". Kontrollerna följer controls.required_sections — ändras en tröskel där följer katalogen
med, för den kopierar inga tal.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import controls as ctl  # noqa: E402
import insider as ins  # noqa: E402
import lukacs  # noqa: E402
import producers as prod  # noqa: E402
import scorecard as sc  # noqa: E402
import scoring as sco  # noqa: E402
import tiggre as tig  # noqa: E402
from asymmetry import fields as afields  # noqa: E402
from confidence import config as cfg  # noqa: E402
from review import catalog as cat  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE = {"rule": "producers.py", "royalty": "producers.py", "sprott": "scoring.py", "durrett": "scoring.py",
          "tiggre": "tiggre.py", "insider": "insider.py", "scorecard": "scorecard.py"}
DS_KEYS = {f.key for f in ctl.DS_FIELDS}
AQS_KEYS = {f.key for f in ctl.AQS_FIELDS}
# Raderna som ger flest fält: Rick Rule över gränsen med skuld (AQS, CSM, Lukacs FV, deleveraging), Tiggre med
# kärninnehav (fem scenarier) och tre katalysatorer.
FULL = {"rule": {"position_pct": 3.0, "nd_ebitda": 1.5, "fcf_kvalitet": "B", "is_core": True},
        "tiggre": {"is_core": True, "catalysts": [{"name": "FID", "date": "2027-01"}, {}, {}]},
        "royalty": {}, "sprott": {}, "durrett": {}, "insider": {}, "scorecard": {"strategy": "producenter"},
        "confidence": {"stage": "developer"}, "asymmetry": {"stage": "explorer"}}


def _src(name: str) -> str:
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def _all_fields():
    for sheet, row in FULL.items():
        for f in cat.fields_for(sheet, row, position_pct=3.0 if sheet in ("sprott", "durrett", "royalty") else None):
            yield sheet, row, f


def _control_path_ok(path: str) -> bool:
    if path in DS_KEYS or path in AQS_KEYS or path in {k for k, _l in ctl.DS_INFO_FIELDS}:
        return True
    if path in ("fin_catalyst_text", "fin_catalyst_date", "is_core", "secured_cash", "csm.bear_survival"):
        return path.split(".")[-1] in _src("controls_ui.py")
    m = re.fullmatch(r"csm\.([^.]+)\.(price|fcf_musd|nav_musd|financing_need)", path)
    if m:
        return m.group(1) in ctl.SCENARIOS_5
    m = re.fullmatch(r"fv\.([^.]+)\.(forward_fcf_musd|target_yield)", path)
    if m:
        return m.group(1) in lukacs.FV_SCENARIOS
    if path in ("fcf_kvalitet", "framtida_antal_aktier", "aktuell_kurs", "what_must_go_right", "ar_till_lag_skuld",
                "is_holding"):
        return f'"{path}"' in _src("lukacs_ui.py") or f'"{path}"' in _src("lukacs.py")
    return False


def test_every_path_exists_in_its_source():
    for sheet, _row, f in _all_fields():
        assert f.sheet == sheet and f.store == cat.SHEETS[sheet].store
        p = f.path
        if sheet in cat.CONFIDENCE_SHEETS:
            assert p.startswith("fields.") and p.split(".", 1)[1] in cfg.FIELD_BY_KEY, p
            continue
        if _control_path_ok(p):
            continue
        if p.startswith("factors."):
            keys = {x.key for x in sco.FACTORS} if sheet in ("sprott", "durrett") else {k for k, _l, _h in tig.FACTORS}
            assert p.split(".", 1)[1] in keys, (sheet, p)
        elif p.startswith("screen."):
            assert p.split(".", 1)[1] in {k for k, _l in tig.SCREEN_PHRASES}, p
        elif p.startswith("catalysts."):
            assert re.fullmatch(r"catalysts\.\d+\.(name|date)", p), p
        elif sheet == "scorecard" and p in sc.GATE_BY_KEY:
            pass
        else:                                       # arkets eget fält: nyckeln står i arkets modul
            assert f'"{p}"' in _src(MODULE[sheet]), (sheet, p)


def test_checks_factors_and_gates_come_from_the_sheets_constants():
    rule = {f.path: f for f in cat.fields_for("rule", {})}
    for k, label, help_ in prod.CHECKS:
        assert rule[k].label == label and rule[k].hint == help_ and rule[k].kind == cat.BOOL
    sprott = {f.path: f for f in cat.fields_for("sprott", {})}
    for x in sco.FACTORS:
        f = sprott[f"factors.{x.key}"]
        assert f.kind == cat.SCORE02 and x.two in f.hint and x.one in f.hint and x.zero in f.hint
    for x in ctl.DS_FIELDS:
        assert x.zero in sprott[x.key].hint and x.two in sprott[x.key].hint
    tg = [f.path for f in cat.fields_for("tiggre", {})]
    assert "factors.un" not in tg                          # U/N räknas ur kvoten, matas aldrig in
    insider = {f.path: f for f in cat.fields_for("insider", {})}
    assert insider["role"].choices == tuple(ins.ROLES) and insider["trigger"].choices == tuple(ins.TRIGGERS)
    card = [f.path for f in cat.fields_for("scorecard", {"strategy": "tiggre"}, strategies=("tiggre",))]
    assert "inga_roda_flaggor" not in card                 # sätts av kontrollerna, kan inte kryssas
    assert "sakerhetsmarginal" in card                      # manuellt kryss när Lukacs inte krävs
    mech = [f.path for f in cat.fields_for("scorecard", {"strategy": "producenter", "position_pct_total": 3.0})]
    assert "sakerhetsmarginal" not in mech                  # räknas av Lukacs FV över gränsen


def test_hints_come_from_the_extraction_specs():
    from ai import extract_prompt as xp
    rule = {f.path: f for f in cat.fields_for("rule", {})}
    want = {f.key: f.hint for f in xp.FIELDS["rule"]}
    assert rule["unit_cost"].hint == want["unit_cost"] and rule["nav_musd"].hint == want["nav_musd"]
    conf = {f.path: f for f in cat.fields_for("confidence", {"stage": "developer"})}
    assert conf["fields.npv_musd"].hint == {f.key: f.hint for f in xp.FIELDS["confidence"]}["npv_musd"]
    assert conf["fields.res_size"].hint == cfg.FIELD_BY_KEY["res_size"].hint


def test_no_duplicate_paths_and_valid_kinds_and_who():
    for sheet, row in FULL.items():
        fs = cat.fields_for(sheet, row, position_pct=3.0 if sheet in ("sprott", "durrett") else None)
        paths = [f.path for f in fs]
        assert len(paths) == len(set(paths)), sheet
        for f in fs:
            assert f.kind in cat.KINDS and f.who in cat.WHO and f.label, (sheet, f)
            assert f.key == f"{sheet}:{f.path}"


def _sample(f):
    if f.kind == cat.BOOL:
        return True
    if f.kind in (cat.SCORE02, cat.INT):
        return 1
    if f.kind == cat.CHOICE:
        return f.choices[0] if f.choices else "x"
    if f.kind in (cat.TEXT,):
        return "text"
    if f.kind == cat.MONTH:
        return "2027-03"
    if f.kind == cat.DATE:
        return "2027-03-01"
    return 12.5


def test_get_set_round_trip_for_every_field():
    for sheet, _row, f in _all_fields():
        r: dict = {}
        v = _sample(f)
        cat.set_value(r, f.path, v)
        assert cat.get_value(r, f.path) == v, (sheet, f.path, r)
        cat.set_value(r, f.path, None)                      # tömt fält = nyckeln borta (som i arket)
        assert cat.is_empty(cat.get_value(r, f.path)), (sheet, f.path)


def test_paths_reach_the_rows_the_sheets_write():
    row = {"factors": {"balans": 2}, "screen": {"fs": False}, "csm": {"Bear": {"fcf_musd": -3.0}},
           "fv": {"Base": {"target_yield": 9.0}}, "catalysts": [{"name": "FID", "date": "2027-01"}],
           "fields": {"aisc": {"value": 1450.0, "kind": "ACTUAL", "source": "MD&A", "pub_date": "2026-08-01"}}}
    assert cat.get_value(row, "factors.balans") == 2 and cat.get_value(row, "screen.fs") is False
    assert cat.get_value(row, "csm.Bear.fcf_musd") == -3.0 and cat.get_value(row, "fv.Base.target_yield") == 9.0
    assert cat.get_value(row, "catalysts.0.date") == "2027-01" and cat.get_value(row, "catalysts.3.date") is None
    assert cat.get_value(row, "fields.aisc") == 1450.0 and cat.get_value(row, "fields.npv_musd") is None
    p = cat.get_point(row, "fields.aisc")
    assert p.source == "MD&A" and p.pub_date == "2026-08-01" and cat.get_point(row, "factors.balans") is None
    cat.set_value(row, "fields.npv_musd", 250.0)            # rått värde → Datapoint utan källa, aldrig en gammal
    assert row["fields"]["npv_musd"]["value"] == 250.0 and row["fields"]["npv_musd"]["source"] == ""
    assert row["fields"]["npv_musd"]["kind"] == "ASSUMPTION"
    cat.set_value(row, "catalysts.2.name", "Byggstart")
    assert len(row["catalysts"]) == 3 and row["catalysts"][2] == {"name": "Byggstart"}
    from confidence.data.models import CompanyInput
    c = CompanyInput.from_dict({"ticker": "X", "fields": row["fields"]})
    assert cat.get_value(c, "fields.aisc") == 1450.0       # CompanyInput går också


def test_decisions_are_bara_du():
    def who(sheet, row=None, **kw):
        return {f.path: f.who for f in cat.fields_for(sheet, row or {}, **kw)}
    rule = who("rule", FULL["rule"])
    for p in ("position_pct", "is_core", "secured_cash", "csm.bear_survival", "is_holding"):
        assert rule[p] == cat.BARA_DU, p
    card = who("scorecard", {"strategy": "tiggre"}, strategies=("tiggre",))
    for p in ("strategy", "position_pct_total", "position_pct_strategy", "sell_rule"):
        assert card[p] == cat.BARA_DU, p
    assert all(card[g.key] == cat.BARA_DU for g in sc.GATES if g.key in card)
    ins_ = who("insider")
    assert ins_["gate"] == cat.BARA_DU and ins_["trigger"] == cat.BARA_DU
    assert ins_["insiders"] == cat.CHATGPT and ins_["price_now"] == cat.BORSDATA
    # poäng är bedömningar ChatGPT får föreslå, fakta med källa är ChatGPT, Börsdatas tal är Börsdata
    assert rule["ds_runway"] == cat.BEDOMNING and rule["aqs_kostnad"] == cat.BEDOMNING
    assert rule["unit_cost"] == cat.CHATGPT and rule["ev_ebitda"] == cat.BORSDATA
    conf = who("confidence", {"stage": "developer"})
    assert conf["fields.market_cap_musd"] == cat.BORSDATA and conf["fields.npv_musd"] == cat.CHATGPT
    assert conf["fields.res_size"] == cat.BEDOMNING and conf["fields.record_price_dependent"] == cat.BEDOMNING


def test_controls_follow_required_sections_like_the_sheets():
    paths = lambda sheet, row=None, **kw: {f.path for f in cat.fields_for(sheet, row or {}, **kw)}  # noqa: E731
    small = paths("rule", {"position_pct": 1.0})
    assert DS_KEYS <= small and not AQS_KEYS & small and not any(p.startswith("csm.") for p in small)
    big = paths("rule", {"position_pct": 3.0})
    assert AQS_KEYS <= big and "csm.Bear.fcf_musd" in big and "fv.Base.forward_fcf_musd" in big
    assert "csm.Bear.financing_need" not in big
    tg = paths("tiggre")                                    # arkets standard 4 %: hela paketet, utvecklare
    assert AQS_KEYS <= tg and "csm.Bear.financing_need" in tg and not any(p.startswith("fv.") for p in tg)
    assert not DS_KEYS & paths("royalty")                    # Royalty-kortet har inga kontroller …
    assert DS_KEYS <= paths("royalty", position_pct=1.0)     # … men köpgrinden kräver DS
    assert DS_KEYS <= paths("sprott") and not AQS_KEYS & paths("sprott")
    assert not DS_KEYS & paths("insider", position_pct=3.0)  # insider handlar inte om tillgången
    assert "fv.Base.forward_fcf_musd" in paths("rule", {"position_pct": 1.0, "fv_manual": True})
    five = paths("tiggre", {"is_core": True})
    assert all(f"csm.{s}.price" in five for s in ctl.SCENARIOS_5)


def test_no_threshold_is_copied(monkeypatch):
    """Flyttas gränsen i controls följer katalogen med — den läser regeln, den har ingen egen."""
    assert not AQS_KEYS & {f.path for f in cat.fields_for("rule", {"position_pct": 1.5})}
    monkeypatch.setattr(ctl, "FULL_WORK_MIN_PCT", 1.0)
    assert AQS_KEYS <= {f.path for f in cat.fields_for("rule", {"position_pct": 1.5})}
    yields = [f for f in cat.fields_for("rule", FULL["rule"]) if f.path.endswith("target_yield")]
    assert {(f.lo, f.hi) for f in yields} == {lukacs.yield_band("B")}


def test_asymmetry_is_the_fields_the_scores_read():
    for stage in cfg.STAGES:
        keys = [f.path.split(".", 1)[1] for f in cat.fields_for("asymmetry", {"stage": stage})]
        assert keys == list(dict.fromkeys(afields.all_keys(stage)))
        assert len(cat.fields_for("confidence", {"stage": stage})) == len(cfg.fields_for(stage))


def test_progress_counts_filled_fields():
    filled, total, missing = cat.progress("royalty", {"pnav_now": 0.8, "level": 2, "geo_now": ""})
    assert (filled, total) == (2, 7) and missing == ["pnav_bottom", "ev_now", "ev_median", "geo_now", "geo_3y"]
    f2, t2, _m = cat.progress("rule", {"jurisdiktion": False})      # ett besvarat Nej är ifyllt
    assert f2 == 1 and t2 == len(cat.fields_for("rule", {}))


def test_unknown_sheet_is_an_error():
    with pytest.raises(KeyError):
        cat.fields_for("finns-inte", {})
