"""
producers.py — Rick Rule + Royalty C (ur ravarurotation.xlsx).

De två granskningsarken som ligger mellan rotationen och köpet: när
rotationen sagt VAR kapitalet ska, säger de här VILKET bolag.

  Rick Rule     — Överlevarna. Guiden kallar arket "Producenter A" ("2 p i
                  Producenter A-arket"); här heter det efter sin upphovsman
                  som de andra granskningsarken gör. Dess fyra
                  granskningssteg — landrisk, kostnadsposition, ledning,
                  kapitaldisciplin — är exakt arkets fyra fält. Marginalen mot
                  kostnadskurvan plus tre disciplinfrågor: 0–5 poäng, ≥ 4
                  köpkandidat, 3 bevaka, under 3 passa.
  Royalty C     — rabatten mot egen historik, och om GEO per aktie växer.
                  En royalty som krymper per aktie är inte billig, den är
                  utspädd.

Formlerna är migrationsspecens (§6), som i sin tur är arkets:

  marginal   = (råvarupris − kostnad per enhet) / råvarupris
  poäng      = (marginal >= 40 % -> 2p, >= 25 % -> 1p) + jurisdiktion
               + kapitaldisciplin + insyn  (0/1 var)
  rabatt     = pnav_nu / pnav_botten − 1
  vs median  = pnav_nu / ev_ebitda_median-jämförelsen
  GEO-tillväxt = geo_aktie_nu / geo_aktie_3år − 1
"""

from __future__ import annotations

import streamlit as st
from dataclasses import dataclass
from datetime import date
from typing import Optional

import csv_export
import lukacs
import storage
import storage_ui
import controls as ctl
import controls_ui

try:
    from rotation import COMMODITIES as _ROT_COMMODITIES
    COMMODITIES = tuple(c.name for c in _ROT_COMMODITIES)
except Exception:                                   # pragma: no cover
    COMMODITIES = ("Guld", "Silver", "Uran", "Koppar")

_CACHE_KEY = "producers_data"
STORE = "producers"   # data/producers.json

TEXT, DIM = "#e8e4dc", "#8a8578"
GREEN, AMBER, RED, CYAN, GOLD = "#2d8a4e", "#d4943a", "#c44545", "#00E5FF", "#c9a84c"

PRODUCERS, ROYALTY = "producers", "royalty"

# Fliknamn. Lagringsnycklarna ovan står kvar — de ligger i Gist och ett byte
# där hade tappat sparade rader för att ändra en rubrik.
SHEET_RULE = "Rick Rule"
SHEET_ROYALTY = "Royalty C"
GUIDE_NAME_RULE = "Producenter A"     # guidens eget namn, behålls som spårbarhet

# ── Rick Rule — Överlevarna (arkets namn i guiden: Producenter A) ────────────────────────────────────────────────────────────
MARGIN_STRONG, MARGIN_OK = 40.0, 25.0      # procent
PROD_BUY_MIN, PROD_WATCH_MIN = 4, 3
PROD_MAX_SCORE = 5

# Den döende tillgången — Rules egen strykregel. Guiden, om screeningen:
# "stryk direkt ... bolag där låg multipel förklaras av döende tillgång
# (gruvlivslängd < 5 år, R/P < 8 år)". Och i mini-exemplet: samma 5/5-bolag
# med fyra års gruvlivslängd är en passa, för priset är lågt av ett skäl.
LIFE_MIN_YEARS = 5.0        # gruvlivslängd
RP_MIN_YEARS = 8.0          # reserver/produktion, olja och gas
DYING_ASSET = "Döende tillgång — låg multipel av ett skäl"

P_BUY = "Köpkandidat"
P_WATCH = "Bevaka"
P_PASS = "Passa"
PROD_COLOR = {P_BUY: GREEN, P_WATCH: AMBER, P_PASS: DIM}

CHECKS = (
    ("jurisdiktion", "Jurisdiktion OK",
     "Fraser-rankingens topphalva. Nedre halvan är passa oavsett siffror."),
    ("kapitaldisciplin", "Kapitaldisciplin",
     "Utdelning eller återköp pågår, ingen tillväxtcapex utanför kärnan."),
    ("insyn", "Insynsägande",
     "Ledning och styrelse äger själva, och köper snarare än säljer."),
)

# ── Royalty C ────────────────────────────────────────────────────────────────
DISCOUNT_MAX = 15.0        # högst 15 % över egen botten för köpläge
ROYALTY_LEVELS = (1, 2, 3)

R_BUY = "KÖPLÄGE"
R_NEAR = "Nära historisk botten — kolla GEO"
R_GEO_WARN = "Varning: GEO/aktie krymper"
R_NEUTRAL = "Neutral"
ROYALTY_COLOR = {R_BUY: GREEN, R_NEAR: AMBER, R_GEO_WARN: RED, R_NEUTRAL: DIM}

LEVEL_NOTE = {
    1: "Nivå 1 — de stora. Säljs i princip aldrig.",
    2: "Nivå 2 — mellanskiktet. Säljs om GEO per aktie stagnerar två år.",
    3: "Nivå 3 — de små. Säljs vid uppköp eller när tesen är klar.",
}


@dataclass(frozen=True)
class Verdict:
    label: str
    why: str


# ── Rena beräkningar ─────────────────────────────────────────────────────────
def _num(value, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def margin_pct(price, unit_cost) -> Optional[float]:
    """(råvarupris − kostnad per enhet) / råvarupris, i procent."""
    p, c = _num(price), _num(unit_cost)
    if p is None or c is None or p == 0:
        return None
    return round((p - c) / p * 100, 6)


def margin_points(margin) -> int:
    """≥ 40 % marginal = 2p, ≥ 25 % = 1p, annars 0."""
    m = _num(margin)
    if m is None:
        return 0
    if m >= MARGIN_STRONG:
        return 2
    return 1 if m >= MARGIN_OK else 0


def _years(value) -> Optional[float]:
    """Ett årsfält, eller None om det inte är ifyllt.

    Noll är inte ett svar. Fälten ritas med number_input(min_value=0.0), som
    startar på 0,0 och inte kan lämnas tomt — så en nolla betyder att frågan
    aldrig ställdes. En gruva med noll års livslängd finns inte; en sådan är
    stängd. Läser man nollan bokstavligt blir varje guldproducent en döende
    tillgång så fort någon öppnat kortet, eftersom R/P bara gäller olja och
    gas och därför alltid står kvar på noll.
    """
    v = _num(value)
    return v if v is not None and v > 0 else None


def asset_dying(row: dict) -> Optional[bool]:
    """Är den låga multipeln förklarad av att tillgången tar slut?

    None när varken gruvlivslängd eller R/P är ifylld — obesvarad fråga är
    inte ett friskintyg, men den ska heller inte döma ut bolaget här. Den
    fångas i stället av köpgrindens luck-regel om positionen är stor nog.

    Ett bolag behöver bara ett av måtten: gruvor har gruvlivslängd, olja och
    gas har R/P. Att det andra står tomt är normalfallet, inte en brist.
    """
    life = _years(row.get("mine_life"))
    rp = _years(row.get("rp_ratio"))
    if life is None and rp is None:
        return None
    if life is not None and life < LIFE_MIN_YEARS:
        return True
    if rp is not None and rp < RP_MIN_YEARS:
        return True
    return False


def producer_score(row: dict) -> Optional[int]:
    """0–5: marginalpoängen plus de tre disciplinfrågorna.

    None när marginalen inte går att räkna — utan den är poängen bara de tre
    kryssrutorna, och det är inte modellen.
    """
    m = margin_pct(row.get("price"), row.get("unit_cost"))
    if m is None:
        return None
    return margin_points(m) + sum(1 for k, _l, _h in CHECKS if row.get(k))


def producer_verdict(score: Optional[int],
                     dying: Optional[bool] = None) -> Optional[Verdict]:
    """Bedömningen. En döende tillgång överprövar poängen helt.

    Guidens mini-exempel är just den fällan: EV/EBITDA 3,8, nettokassa,
    soliditet 64 %, AISC i 25:e percentilen, Nevada, insynsägande, 6 %
    direktavkastning ger 5/5 — men med fyra års gruvlivslängd är det ändå en
    passa. Poängen mäter hur bra bolaget är på att vänta; den mäter inte om
    det finns något kvar att vänta på.
    """
    if score is None:
        return None
    if dying:
        return Verdict(P_PASS, f"{DYING_ASSET}. Poängen ({score}/"
                               f"{PROD_MAX_SCORE}) spelar ingen roll — "
                               f"gruvan tar slut före cykeln vänder.")
    if score >= PROD_BUY_MIN:
        return Verdict(P_BUY, "Marginal och disciplin på plats.")
    if score >= PROD_WATCH_MIN:
        return Verdict(P_WATCH, "Nästan — en faktor saknas.")
    return Verdict(P_PASS, "För tunn marginal eller för lös disciplin.")


def discount_vs_bottom(now, bottom) -> Optional[float]:
    """Hur många procent över sin egen P/NAV-botten bolaget handlas."""
    n, b = _num(now), _num(bottom)
    if n is None or b is None or b == 0:
        return None
    return round((n / b - 1) * 100, 6)


def vs_median(now, median) -> Optional[float]:
    """EV/EBITDA mot egen median, i procent. Negativt = billigare än vanligt."""
    n, m = _num(now), _num(median)
    if n is None or m is None or m == 0:
        return None
    return round((n / m - 1) * 100, 6)


def geo_growth(now, three_years_ago) -> Optional[float]:
    """GEO per aktie idag mot för tre år sedan, i procent."""
    n, o = _num(now), _num(three_years_ago)
    if n is None or o is None or o == 0:
        return None
    return round((n / o - 1) * 100, 6)


def royalty_signal(row: dict) -> Verdict:
    """Signalen, i specens ordning.

    Ordningen betyder något: en krympande GEO per aktie slår ut köpläget även
    när priset ser billigt ut, för det är just utspädningen som gjorde det
    billigt.
    """
    disc = discount_vs_bottom(row.get("pnav_now"), row.get("pnav_bottom"))
    med = vs_median(row.get("ev_now"), row.get("ev_median"))
    geo = geo_growth(row.get("geo_now"), row.get("geo_3y"))

    cheap = disc is not None and disc <= DISCOUNT_MAX
    if cheap and med is not None and med <= 0 and geo is not None and geo > 0:
        return Verdict(R_BUY, "Nära egen botten, under egen median, och GEO "
                              "per aktie växer.")
    if geo is not None and geo < 0:
        return Verdict(R_GEO_WARN, "GEO per aktie krymper — utspädningen äter "
                                   "tillväxten. Billigt av fel skäl.")
    if cheap:
        return Verdict(R_NEAR, "Nära historisk botten. Kolla GEO per aktie "
                               "innan du köper.")
    return Verdict(R_NEUTRAL, "Varken billig mot egen historik eller varnande.")


def ranked_producers(rows: list) -> list:
    out = []
    for r in rows or []:
        sc = producer_score(r)
        dying = asset_dying(r)
        out.append({"row": r, "score": sc,
                    "verdict": producer_verdict(sc, dying), "dying": dying,
                    "margin": margin_pct(r.get("price"), r.get("unit_cost"))})
    out.sort(key=lambda x: (-(x["score"] if x["score"] is not None else -1),
                            (x["row"].get("ticker") or "")))
    return out


def ranked_royalty(rows: list) -> list:
    order = {R_BUY: 0, R_NEAR: 1, R_NEUTRAL: 2, R_GEO_WARN: 3}
    out = []
    for r in rows or []:
        v = royalty_signal(r)
        out.append({"row": r, "verdict": v,
                    "discount": discount_vs_bottom(r.get("pnav_now"),
                                                   r.get("pnav_bottom")),
                    "median": vs_median(r.get("ev_now"), r.get("ev_median")),
                    "geo": geo_growth(r.get("geo_now"), r.get("geo_3y"))})
    out.sort(key=lambda x: (order.get(x["verdict"].label, 9),
                            (x["row"].get("ticker") or "")))
    return out


# ── Lagring ──────────────────────────────────────────────────────────────────
def _today() -> str:
    return date.today().isoformat()


def _uid() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


def _default() -> dict:
    return {PRODUCERS: [], ROYALTY: []}


def _load() -> dict:
    """Laddas EN gång per session; därefter äger sessionen sanningen."""
    data = storage.session_load(STORE, _default(),
                                legacy_file="producers_data.json")
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
def render_producers_page(sheet: Optional[str] = None) -> None:
    """Ett av de två granskningsarken.

    sheet väljer arket direkt. Utan det ritas en egen väljare, så funktionen
    fungerar både som underflik och fristående — panelen skickar in valet
    eftersom arken numera är egna underflikar under GRANSKNING.
    """
    data = _load()
    storage_ui.save_bar(STORE, "Granskningsarken")
    title = sheet if sheet in (SHEET_RULE, SHEET_ROYALTY) else "GRANSKNINGSARKEN"
    st.markdown(
        f"<div style='text-align:center;padding:10px 0 4px;'>"
        f"<h2 style='color:{CYAN};letter-spacing:0.12em;margin:0;'>"
        f"{title.upper()}</h2>"
        f"<p style='color:{DIM};font-size:0.78rem;margin:6px 0 0;'>"
        f"Rotationen säger var kapitalet ska, screenern vilka bolag som kvalar. "
        f"Det här säger vilket av dem du köper.</p></div>",
        unsafe_allow_html=True)

    which = sheet
    if which not in (SHEET_RULE, SHEET_ROYALTY):
        which = st.radio("Ark", [SHEET_RULE, SHEET_ROYALTY], horizontal=True,
                         key="pr_which", label_visibility="collapsed")
    if which == SHEET_RULE:
        _producers(data)
    else:
        _royalty(data)


PROD_CSV = [("date", "Datum"), ("ticker", "Ticker"), ("name", "Bolag"),
            ("commodity", "Råvara"), ("ev_ebitda", "EV/EBITDA"),
            ("nd_ebitda", "Nettoskuld/EBITDA"), ("unit_cost", "Kostnad/enhet"),
            ("price", "Råvarupris"), ("_margin", "Marginal %"),
            ("jurisdiktion", "Jurisdiktion"),
            ("kapitaldisciplin", "Kapitaldisciplin"), ("insyn", "Insyn"),
            ("_score", "Poäng (0–5)"), ("_verdict", "Bedömning"),
            ("position_pct", "Position % av total"),
            ("mine_life", "Gruvlivslängd (år)"), ("rp_ratio", "R/P (år)"),
            ("_dying", "Döende tillgång"),
            ("_ds", "DS"), ("_aqs", "AQS"), ("_csm_flag", "CSM röd flagga"),
            ("fcf_kvalitet", "FCF-klass"), ("_fv_base", "Fair value Base"),
            ("_mos", "Säkerhetsmarginal %"), ("_mos_band", "MoS-bedömning")]

ROYALTY_CSV = [("date", "Datum"), ("ticker", "Ticker"), ("name", "Bolag"),
               ("level", "Nivå"), ("pnav_now", "P/NAV nu"),
               ("pnav_bottom", "P/NAV botten"), ("_discount", "Mot botten %"),
               ("ev_now", "EV/EBITDA nu"), ("ev_median", "EV/EBITDA median"),
               ("_median", "Mot median %"), ("geo_now", "GEO/aktie nu"),
               ("geo_3y", "GEO/aktie 3 år"), ("_geo", "GEO-tillväxt %"),
               ("_signal", "Signal")]


def _r1(v):
    return None if v is None else round(v, 1)


def _csv_row(r: dict) -> dict:
    """En rad i Rick Rule-exporten: arket, kontrollerna och Lukacs FV."""
    row = r["row"]
    fv = lukacs.evaluate(row)
    return {**row, "_margin": _r1(r["margin"]), "_score": r["score"],
            "_verdict": r["verdict"].label if r["verdict"] else None,
            "_dying": r["dying"],
            "_ds": ctl.ds_total(row), "_aqs": ctl.aqs_total(row),
            "_csm_flag": ctl.csm_red_flag(row.get("csm_kind", ctl.PRODUCER),
                                          row.get("csm", {}),
                                          bool(row.get("secured_cash"))),
            "_fv_base": _r1(fv["fv_base"]), "_mos": _r1(fv["mos"]),
            "_mos_band": fv["mos_band"]}


def _producers(data: dict) -> None:
    st.markdown(f"<div style='color:{DIM};font-size:0.78rem;margin-bottom:4px;'>"
                f"<b style='color:{TEXT};'>{SHEET_RULE} — Överlevarna.</b> "
                f"Överlevar-screenern körs i Börsdata; de fyra "
                f"granskningsstegen — landrisk, kostnadsposition, ledning, "
                f"kapitaldisciplin — är fälten nedan. "
                f"<span style='color:{DIM};'>Guiden kallar arket "
                f"{GUIDE_NAME_RULE}.</span></div>",
                unsafe_allow_html=True)
    csv_export.download_button([_csv_row(r)
                                for r in ranked_producers(data[PRODUCERS])],
                               PROD_CSV, "rick_rule", key="csv_producers")
    st.caption(f"Marginal ≥ {MARGIN_STRONG:g} % ger 2 p, ≥ {MARGIN_OK:g} % ger "
               f"1 p. Tre disciplinfrågor ger 1 p var. "
               f"{PROD_BUY_MIN}–{PROD_MAX_SCORE} = köpkandidat · "
               f"{PROD_WATCH_MIN} = bevaka · under {PROD_WATCH_MIN} = passa.")

    with st.expander("➕ Nytt bolag", expanded=not data[PRODUCERS]):
        c1, c2, c3 = st.columns([1, 2, 1.2])
        ticker = c1.text_input("Ticker", key="pr_new_t")
        name = c2.text_input("Bolag", key="pr_new_n")
        commodity = c3.selectbox("Råvara", list(COMMODITIES), key="pr_new_c")
        if st.button("Lägg till", key="pr_new_add"):
            if ticker.strip():
                data[PRODUCERS].append({
                    "id": _uid(), "ticker": ticker.strip().upper(),
                    "name": name.strip(), "commodity": commodity,
                    "date": _today()})
                _save(data)
                st.rerun()
            else:
                st.warning("Ticker krävs.")

    rows = ranked_producers(data[PRODUCERS])
    if not rows:
        st.caption("Inga bolag ännu. Överlevar-screenern körs i Börsdata — "
                   "filtret står i RULES → 📚 SNABBREFERENS.")
        return

    for r in rows:
        row, sc, vd = r["row"], r["score"], r["verdict"]
        head = (f"{row.get('ticker','?')} · {row.get('commodity','')} · "
                f"{sc if sc is not None else '–'}/{PROD_MAX_SCORE} p · "
                f"{vd.label if vd else 'ofullständig'}")
        with st.expander(head, expanded=False):
            changed = False
            v1, v2, v3, v4 = st.columns(4)
            ev = v1.number_input("EV/EBITDA", min_value=0.0, step=0.5,
                                 value=float(_num(row.get("ev_ebitda"), 0.0) or 0.0),
                                 key=f"pr_ev_{row['id']}")
            nd = v2.number_input("Nettoskuld/EBITDA", step=0.1,
                                 value=float(_num(row.get("nd_ebitda"), 0.0) or 0.0),
                                 key=f"pr_nd_{row['id']}",
                                 help="Negativt = nettokassa.")
            cost = v3.number_input("Kostnad per enhet", min_value=0.0, step=10.0,
                                   value=float(_num(row.get("unit_cost"), 0.0) or 0.0),
                                   key=f"pr_c_{row['id']}",
                                   help="AISC eller C1 — samma enhet som priset.")
            price = v4.number_input("Råvarupris nu", min_value=0.0, step=10.0,
                                    value=float(_num(row.get("price"), 0.0) or 0.0),
                                    key=f"pr_p_{row['id']}")
            if (storage.differs(ev, row.get("ev_ebitda"), 0.0)
                    or storage.differs(nd, row.get("nd_ebitda"), 0.0)
                    or storage.differs(cost, row.get("unit_cost"), 0.0)
                    or storage.differs(price, row.get("price"), 0.0)):
                row["ev_ebitda"], row["nd_ebitda"] = ev, nd
                row["unit_cost"], row["price"] = cost, price
                changed = True

            m = r["margin"]
            l1, l2, l3 = st.columns(3)
            l1.metric("Marginal", f"{m:.1f} %" if m is not None else "–",
                      help=f"Ger {margin_points(m)} p. Kostnadskurvan är hela "
                           f"caset: de dyraste producenterna dör först i "
                           f"prisfall.")
            life = l2.number_input(
                "Gruvlivslängd (år)", min_value=0.0, step=1.0,
                value=float(_num(row.get("mine_life"), 0.0) or 0.0),
                key=f"pr_life_{row['id']}",
                help=f"Under {LIFE_MIN_YEARS:g} år = passa oavsett poäng. "
                     f"Står i presentationen.")
            rp = l3.number_input(
                "R/P-kvot (olja/gas, år)", min_value=0.0, step=1.0,
                value=float(_num(row.get("rp_ratio"), 0.0) or 0.0),
                key=f"pr_rp_{row['id']}",
                help=f"Reserver ÷ produktion. Under {RP_MIN_YEARS:g} år = "
                     f"samma sak.")
            if (storage.differs(life, row.get("mine_life"), 0.0)
                    or storage.differs(rp, row.get("rp_ratio"), 0.0)):
                row["mine_life"], row["rp_ratio"] = life, rp
                changed = True
            dying = asset_dying(row)
            if dying:
                st.error(f"{DYING_ASSET} — den låga multipeln förklaras av "
                         f"att tillgången tar slut. Passa.")
            elif dying is None:
                st.caption(f"Fyll i gruvlivslängd — eller R/P för olja och gas. "
                           f"Strykregeln (under {LIFE_MIN_YEARS:g} respektive "
                           f"{RP_MIN_YEARS:g} år) kan inte prövas utan den, och "
                           f"poängen ensam ser inte att tillgången tar slut.")

            ccols = st.columns(3)
            for (ckey, clabel, chelp), col in zip(CHECKS, ccols):
                v = col.checkbox(clabel, value=bool(row.get(ckey)),
                                 key=f"pr_ch_{row['id']}_{ckey}", help=chelp)
                if v != bool(row.get(ckey)):
                    row[ckey] = v
                    changed = True

            sc2 = producer_score(row)
            vd2 = producer_verdict(sc2, asset_dying(row))
            c2_ = PROD_COLOR.get(vd2.label if vd2 else "", DIM)
            st.markdown(
                f"<div style='border:1px solid {c2_}55;background:{c2_}0d;"
                f"border-radius:8px;padding:10px 14px;margin:8px 0;'>"
                f"<span style='color:{c2_};font-weight:700;font-size:1.05rem;'>"
                f"{sc2 if sc2 is not None else '–'} / {PROD_MAX_SCORE}</span>"
                f"<span style='color:{c2_};font-weight:700;margin-left:12px;'>"
                f"{vd2.label if vd2 else 'Fyll i pris och kostnad'}</span>"
                f"<div style='color:{TEXT};font-size:0.8rem;margin-top:3px;'>"
                f"{vd2.why if vd2 else ''}</div></div>", unsafe_allow_html=True)

            # Positionsstorleken styr vilka kontroller som krävs.
            pos = st.number_input(
                "Tänkt position (% av total)", min_value=0.0, step=0.5,
                value=float(_num(row.get("position_pct"), 0.0) or 0.0),
                key=f"pr_pos_{row['id']}",
                help=f"Över {ctl.FULL_WORK_MIN_PCT:g} % krävs AQS och CSM. "
                     f"Under räcker DS.")
            if storage.differs(pos, row.get("position_pct"), 0.0):
                row["position_pct"] = pos
                changed = True

            if controls_ui.render_all(
                    row, row["id"], position_pct=pos, strategy="producenter",
                    aqs_prefill=ctl.aqs_prefill_from_producer(row),
                    kind=ctl.PRODUCER):
                changed = True

            if st.button("Ta bort", key=f"pr_del_{row['id']}"):
                data[PRODUCERS] = [x for x in data[PRODUCERS]
                                   if x["id"] != row["id"]]
                _save(data)
                st.rerun()
            if changed:
                _save(data)


def _royalty(data: dict) -> None:
    csv_export.download_button(
        [{**r["row"], "_discount": _r1(r["discount"]),
          "_median": _r1(r["median"]), "_geo": _r1(r["geo"]),
          "_signal": r["verdict"].label}
         for r in ranked_royalty(data[ROYALTY])],
        ROYALTY_CSV, "royalty_c", key="csv_royalty")
    st.caption(f"Köpläge kräver tre saker samtidigt: högst {DISCOUNT_MAX:g} % "
               f"över egen P/NAV-botten, EV/EBITDA under egen median, och "
               f"växande GEO per aktie.")

    with st.expander("➕ Nytt bolag", expanded=not data[ROYALTY]):
        c1, c2, c3 = st.columns([1, 2, 1])
        ticker = c1.text_input("Ticker", key="ro_new_t")
        name = c2.text_input("Bolag", key="ro_new_n")
        level = c3.selectbox("Nivå", list(ROYALTY_LEVELS), key="ro_new_l")
        if st.button("Lägg till", key="ro_new_add"):
            if ticker.strip():
                data[ROYALTY].append({
                    "id": _uid(), "ticker": ticker.strip().upper(),
                    "name": name.strip(), "level": level, "date": _today()})
                _save(data)
                st.rerun()
            else:
                st.warning("Ticker krävs.")

    rows = ranked_royalty(data[ROYALTY])
    if not rows:
        st.caption("Inga bolag ännu. Royalty-screenern körs en gång i "
                   "Börsdata — filtret står i RULES → 📚 SNABBREFERENS.")
        return

    for r in rows:
        row, vd = r["row"], r["verdict"]
        head = (f"{row.get('ticker','?')} · nivå {row.get('level','?')} · "
                f"{vd.label}")
        with st.expander(head, expanded=False):
            st.caption(LEVEL_NOTE.get(int(_num(row.get("level"), 1) or 1), ""))
            changed = False

            p1, p2, p3 = st.columns(3)
            pn = p1.number_input("P/NAV nu", min_value=0.0, step=0.05,
                                 value=float(_num(row.get("pnav_now"), 0.0) or 0.0),
                                 key=f"ro_pn_{row['id']}")
            pb = p2.number_input("P/NAV historisk botten", min_value=0.0, step=0.05,
                                 value=float(_num(row.get("pnav_bottom"), 0.0) or 0.0),
                                 key=f"ro_pb_{row['id']}")
            disc = discount_vs_bottom(pn, pb)
            p3.metric("Mot botten",
                      f"{disc:+.1f} %" if disc is not None else "–",
                      help=f"Köpläge kräver högst +{DISCOUNT_MAX:g} %.")

            e1, e2, e3 = st.columns(3)
            en = e1.number_input("EV/EBITDA nu", min_value=0.0, step=0.5,
                                 value=float(_num(row.get("ev_now"), 0.0) or 0.0),
                                 key=f"ro_en_{row['id']}")
            em = e2.number_input("EV/EBITDA median", min_value=0.0, step=0.5,
                                 value=float(_num(row.get("ev_median"), 0.0) or 0.0),
                                 key=f"ro_em_{row['id']}")
            med = vs_median(en, em)
            e3.metric("Mot median", f"{med:+.1f} %" if med is not None else "–",
                      help="Negativt = billigare än bolaget brukar vara.")

            g1, g2, g3 = st.columns(3)
            gn = g1.number_input("GEO/aktie nu", min_value=0.0, step=0.01,
                                 format="%.4f",
                                 value=float(_num(row.get("geo_now"), 0.0) or 0.0),
                                 key=f"ro_gn_{row['id']}")
            g3y = g2.number_input("GEO/aktie för 3 år sedan", min_value=0.0,
                                  step=0.01, format="%.4f",
                                  value=float(_num(row.get("geo_3y"), 0.0) or 0.0),
                                  key=f"ro_g3_{row['id']}")
            geo = geo_growth(gn, g3y)
            g3.metric("GEO-tillväxt", f"{geo:+.1f} %" if geo is not None else "–",
                      help="Per aktie. Krymper den är bolaget utspätt, inte "
                           "billigt.")

            if (storage.differs(pn, row.get("pnav_now"), 0.0)
                    or storage.differs(pb, row.get("pnav_bottom"), 0.0)
                    or storage.differs(en, row.get("ev_now"), 0.0)
                    or storage.differs(em, row.get("ev_median"), 0.0)
                    or storage.differs(gn, row.get("geo_now"), 0.0)
                    or storage.differs(g3y, row.get("geo_3y"), 0.0)):
                row["pnav_now"], row["pnav_bottom"] = pn, pb
                row["ev_now"], row["ev_median"] = en, em
                row["geo_now"], row["geo_3y"] = gn, g3y
                changed = True

            vd2 = royalty_signal(row)
            c2_ = ROYALTY_COLOR.get(vd2.label, DIM)
            st.markdown(
                f"<div style='border:1px solid {c2_}55;background:{c2_}0d;"
                f"border-radius:8px;padding:10px 14px;margin:8px 0;'>"
                f"<span style='color:{c2_};font-weight:700;'>{vd2.label}</span>"
                f"<div style='color:{TEXT};font-size:0.8rem;margin-top:3px;'>"
                f"{vd2.why}</div></div>", unsafe_allow_html=True)

            if st.button("Ta bort", key=f"ro_del_{row['id']}"):
                data[ROYALTY] = [x for x in data[ROYALTY] if x["id"] != row["id"]]
                _save(data)
                st.rerun()
            if changed:
                _save(data)
