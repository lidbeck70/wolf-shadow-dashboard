"""
tabs/home.py — 🏠 I DAG: morgonbeskedet (LIDSKJALV §4.3, beslut 6 och 7).

Allt ur det som redan är sparat (tabs/home_today.load, en gång per session — och igen efter 📡, en ny dag eller
30 minuter). Sidan visar och länkar; den räknar, väljer och slår inte ihop något som motorerna inte redan gör.

  Huvudraden       🏠 I DAG · 📡 Uppdatera (home_refresh) · Morgonrundans pill "1/5 ›"
  Dagens besked    datum och hämtningstid, meningen (home_today.besked) och fyra tryckrutor med antal ur
                   home_today.counts — vändning → Kartan, KÖP → Thor, SÄLJ · MINSKA → Positioner, förfaller → Årshjulet
  Kapitalet        Lägets fyra ljus (laget.capital_row) som ETT tryckkort — ljusen slås inte ihop
  Under vecket     PR 8-ordningen: vändning → kapital → kvalar → storlek → sälj → mål, sedan larmen och
                   Morgonrundans fot. Från 1024 px två kolumner (3:2): vänster vändning, kapital, kvalar; höger sälj,
                   mål, larm. På telefonen staplas de vänster först, så ordningen håller.
  🗺 Flikguiden     zonerna ur navigationsträdet (ui/nav.HOME_ZONES), innehåll och nycklar som förut

Förut lästes tre session_state-nycklar som ingen satte, så pulsen sa alltid UNKNOWN, och "Recent alerts" var en
processlista som var tom efter varje omstart och aldrig såg de schemalagda larmen. _load_status/_next_step finns
kvar (läget ur det sparade) men ritas inte.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

import streamlit as st

from ui import nav
from ui.components import (card, head_row, kpi, light_grid, missing, note, page_header, section, source_chip, tap_card,
                           tile)
from ui.theme import section_title
from ui.tokens import (ACCENT, AMBER as AMBER_C, BG_CARD, BORDER_CARD, BORDER_FAINT, DIM, EMBER, FONT_MONO, FONT_UI,
                       GOLD as GOLD_C, GREEN, GREY, MUTED, PURPLE, RED, TEXT, regime_color, tint)

# Flikguidens zonfärger — kortens linje; rubrikerna är sektionsrubriker i MUTED (§3.5.8: cyan är aldrig en rubrik)
_ZONE_COLOR = {"odin": PURPLE, "thor": ACCENT, "review": "#c9a84c",
               "valhall": EMBER, "portfolio": GREEN, "more": GREY}
_WEEKDAYS = ("måndag", "tisdag", "onsdag", "torsdag", "fredag", "lördag", "söndag")
_MONTHS = ("januari", "februari", "mars", "april", "maj", "juni", "juli",
           "augusti", "september", "oktober", "november", "december")


def swedish_date(d: datetime) -> str:
    return f"{_WEEKDAYS[d.weekday()]} {d.day} {_MONTHS[d.month - 1]} {d.year}"


# ── Läget ur det sparade ─────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def _load_status() -> dict:
    """Ren läsning av det som redan ligger sparat. Saknas en källa blir den
    'okänd' med orsak — aldrig ett gissat läge."""
    out = {}
    try:
        from gist_storage import load_wolf_json
        d = load_wolf_json("wolf_regime.json") or {}
        out["swing"] = {"label": str(d.get("regime") or "OKÄND"),
                        "asof": str(d.get("generated") or "")[:10]}
    except Exception:
        out["swing"] = {"label": "OKÄND", "asof": ""}
    try:
        from ember.cache import load_ember_results
        e = load_ember_results() or {}
        n = len(e.get("eligible") or [])
        out["ember"] = {"label": f"{n} setup{'s' if n != 1 else ''}",
                        "asof": str(e.get("timestamp") or "")[:10]}
    except Exception:
        out["ember"] = {"label": "–", "asof": ""}
    try:
        from gist_storage import load_blob
        sc = load_blob("screens.json", None) or {}
        hits = sum(len((v or {}).get("rows") or []) for v in (sc.get("screens") or {}).values())
        new = sum(len((v or {}).get("new") or []) for v in (sc.get("screens") or {}).values())
        out["screens"] = {"label": f"{hits} träffar · {new} nya" if sc else "–",
                          "asof": str(sc.get("generated") or "")[:10]}
        sr = load_blob("sheets_refresh.json", None) or {}
        ev = sr.get("events") or []
        out["events"] = {"label": f"{len(ev)} händelser" if sr else "–",
                         "asof": str(sr.get("generated") or "")[:10],
                         "rows": ev[:8]}
    except Exception:
        out["screens"] = {"label": "–", "asof": ""}
        out["events"] = {"label": "–", "asof": "", "rows": []}
    return out


def _next_step(status: dict) -> str:
    swing = status.get("swing", {}).get("label", "OKÄND").upper()
    if swing == "RÖD":
        return "⛔ Swing-regimen är RÖD — inga nya köp, bara exits. Gå igenom innehaven mot säljreglerna."
    if swing == "GUL":
        return "🟡 GUL regim — halv positionsstorlek. Kör screenern, men var selektiv."
    if swing == "GRÖN":
        return "🟢 GRÖN regim — välj Momentum Swing i THOR och kontrollera setup A/B."
    return "🔍 Ingen regimdata ännu — kör wolf_data.py eller vänta på nästa schemalagda körning."


def _render_pulse(status: dict) -> None:
    section_title("Läget just nu", "📡")
    sw, em, sc, ev = (status.get(k, {}) for k in ("swing", "ember", "screens", "events"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Swing-regim", sw.get("label", "OKÄND"),
                        regime_color(sw.get("label")), sw.get("asof", "")), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("EMBER-setups", em.get("label", "–"), EMBER, em.get("asof", "")),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Håvarna", sc.get("label", "–"), ACCENT, sc.get("asof", "")),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Arkens händelser", ev.get("label", "–"), PURPLE, ev.get("asof", "")),
                    unsafe_allow_html=True)
    st.markdown(
        f"<div style='background:#1A1F25;border:1px solid rgba(255,255,255,0.06);"
        f"border-left:3px solid {ACCENT};border-radius:8px;padding:12px 18px;margin:12px 0 4px;"
        f"font-size:12px;color:{TEXT};'>{_next_step(status)}</div>", unsafe_allow_html=True)


# ── Zonerna ur navigationsträdet ─────────────────────────────────────────────
def _render_zones() -> None:
    for key, title, cards in nav.HOME_ZONES:
        color = _ZONE_COLOR.get(key, GREY)
        # Rubriken bär redan zonens namn ("ODIN — VAR KAPITALET SKA"): bara flikens emoji framför, inte hela etiketten
        _block(f"{nav.TOP_LABEL[key].split(' ', 1)[0]} {title}")
        cols = st.columns(len(cards))
        for col, (name, desc) in zip(cols, cards):
            with col:
                st.markdown(card(name, desc, color), unsafe_allow_html=True)
                # Kortet öppnar sin flik. Första namnet på ett kort med flera
                # ("Rick Rule · Royalty C") är målet; okänt namn → toppfliken.
                first = name.split(" · ")[0]
                segs = [key, first] if first in nav.options(key) else [key]
                target = nav.slug(segs)
                if st.button("Öppna →", key=f"home_go_{target}", width="stretch"):
                    st.session_state["nav_goto"] = target
                    st.rerun()


# ── Senaste händelserna ur arken ─────────────────────────────────────────────
def _render_recent_events(status: dict) -> None:
    section_title("Senaste händelserna ur arken", "🔔")
    rows = status.get("events", {}).get("rows") or []
    if not rows:
        st.markdown(
            f"<div style='background:#1A1F25;border:1px solid rgba(255,255,255,0.06);"
            f"border-radius:8px;padding:16px;color:{DIM};font-size:0.76rem;text-align:center;'>"
            f"Inga händelser i senaste sifferuppdateringen. Larmen skickas till Discord av "
            f"de schemalagda körningarna — se ALERTS.</div>", unsafe_allow_html=True)
        return
    for e in rows:
        st.markdown(
            f"<div style='background:#1A1F25;border:1px solid rgba(255,255,255,0.06);"
            f"border-radius:8px;padding:10px 14px;margin-bottom:6px;'>"
            f"<div style='color:{TEXT};font-size:0.8rem;font-weight:700;'>{e.get('title', '')}</div>"
            f"<div style='color:{DIM};font-size:0.74rem;'>{e.get('body', '')}</div></div>",
            unsafe_allow_html=True)


# ── 🏠 I DAG (LIDSKJALV §4.3) ────────────────────────────────────────────────
_DUE_SLUG = "more/regler-guider/arshjulet"      # 🗓 ÅRSHJULET — inget mål i nav.TARGETS, så sökvägen direkt
_TURN_SOURCES = (("Deep Contrarian", "deep", "deep"), ("Lång", "long", "long"))

# Sidans egen CSS (ui/shell.page_style, dold behållare): 8 px mellan blocken, rutorna 2 × 2 (4 i rad från 1024 px),
# kolumnerna staplade under 1024 px, vändningens glödkort och 44 px expandrar. Allt väljs på sidans egna nycklar.
_CSS = f"""
.st-key-home_page, .st-key-home_page [data-testid="stVerticalBlock"]:not([class*="st-key-tapc_"]) {{ gap: 8px; }}
.st-key-home_page [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
.home-besked .wolf-sec {{ margin-top: 8px !important; }}
.home-kap .wolf-sec {{ margin-top: 8px !important; }}
.st-key-home_page .st-key-home_tiles {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
.st-key-home_tiles > * {{ min-width: 0; width: auto !important; }}
.st-key-home_tiles .wolf-tile {{ min-height: 80px !important; }}
@media (min-width: 1024px) {{
    .st-key-home_page .st-key-home_tiles {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .home-kap .wolf-sec {{ margin-top: 20px !important; }}
}}
@media (max-width: 1023px) {{
    .st-key-home_cols [data-testid="stHorizontalBlock"] {{ flex-direction: column; flex-wrap: nowrap; gap: 8px; }}
    .st-key-home_cols [data-testid="stColumn"] {{ width: 100% !important; min-width: 100% !important;
        flex: 1 1 auto !important; }}
}}
[class*="st-key-home_turncard_"] {{ border-left: 4px solid {EMBER}; background: {tint(EMBER, .08)};
    border-radius: 12px; padding: 12px 14px; box-sizing: border-box; }}
/* Vändningens tickerknappar radbryts — även 641–1023 px, där kolumnregeln ovan annars staplar raden (en horisontell
   behållare har samma testid som kolumnerna); högre specificitet än den */
.st-key-home_page .st-key-home_cols [class*="st-key-home_turnrow_"],
[class*="st-key-home_turnrow_"] {{ flex-direction: row; flex-wrap: wrap; gap: 8px; }}
/* Listraderna ligger an (de har egna avdelare) — samma specificitet som 8 px-regeln överst, men senare */
.st-key-home_page [data-testid="stVerticalBlock"].st-key-home_kval,
.st-key-home_page [data-testid="stVerticalBlock"].st-key-home_alerts {{ gap: 0; }}
.st-key-home_page summary {{ min-height: 44px; }}
/* Huvudraden: "🏠 I DAG" är kort — rubriken behöver inte golvet på 180 px, så 📡 och pillret ryms på samma rad */
.st-key-head_home > [data-testid="stElementContainer"]:has(.wolf-head) {{ min-width: 0; }}
/* KAPITALET: chevronens luft 24 px och rutorna 10/8 px — "OMXS30 FÖRHÖJD" ryms på en rad på 390 px */
.st-key-tapc_home_laget [data-testid="stMarkdownContainer"] > div {{ padding-right: 24px !important; }}
.st-key-tapc_home_laget .wolf-tile {{ padding-left: 10px !important; padding-right: 8px !important; }}
/* Kvalraden: chip-kolumnen 84 px som åtgärdsraderna; på telefonen chipet på rubrikraden och talen över hela bredden */
.home-kval {{ display: grid; grid-template-columns: 84px minmax(0, 1fr); column-gap: 10px; row-gap: 2px;
    align-items: start; min-height: 56px; padding: 8px 0; box-sizing: border-box;
    border-bottom: 1px solid {BORDER_FAINT}; }}
.home-kval > .k-chip {{ grid-row: 1 / span 3; align-self: center; }}
@media (max-width: 640px) {{
    .home-kval {{ grid-template-columns: auto minmax(0, 1fr); }}
    .home-kval > .k-chip {{ grid-row: auto; align-self: start; padding-top: 1px; }}
    .home-kval > .k-size, .home-kval > .k-flag {{ grid-column: 1 / -1; }}
}}
/* "Hela listan i Thor →" bär sin egen pil i etiketten (oförändrad) — ingen andra › från länkraden */
.st-key-links_home .stButton > button[kind="tertiary"]::after {{ content: none; }}
"""


def _t(x) -> str:
    return escape("" if x is None else str(x), quote=False)


def _block(title: str, color: str = MUTED, count="", cls: str = "") -> None:
    """Sektionsrubriken — tunt omslag kring components.section (12 px MUTED; förut 10 px i färg)."""
    html = section(title, color, count)
    st.markdown(f'<div class="{cls}">{html}</div>' if cls else html, unsafe_allow_html=True)


def _line(text: str, color: str = TEXT, size: str = "15px") -> None:
    """En rad HTML (texten escapas av anroparen), aldrig under 12 px."""
    st.markdown(f"<div style='color:{color};font-size:{size};line-height:1.45;margin:2px 0;'>{text}</div>",
                unsafe_allow_html=True)


def _open(label: str, key: str, target: str, ticker: str = "", strategy: str = "", width: str = "stretch") -> None:
    from ui.links import go
    st.button(label, key=key, on_click=go, args=(target, ticker, strategy), width=width)


def _ht():
    from tabs import home_today as ht
    return ht


def _goto(slug: str) -> None:
    """Callback: öppna en sökväg som inte är ett mål i nav.TARGETS (Årshjulet) — wolf_panel._apply_deep_link."""
    st.session_state["nav_goto"] = slug


def _today_data() -> dict:
    """Det sparade, läst en gång per session — och igen efter 📡, när dagen bytts eller efter 30 minuter."""
    from tabs import home_today as ht
    if (st.session_state.get("_home_refresh") or "home_today" not in st.session_state
            or ht.needs_reload((st.session_state.get("home_today") or {}).get("loaded"))):
        st.session_state.pop("_home_refresh", None)
        with st.spinner("Läser det sparade …"):
            st.session_state["home_today"] = ht.load()
    return st.session_state["home_today"]


def _due_today(day):
    """Årshjulets rutiner i dag (routines.due_today) — en gång per körning; None när de inte går att läsa."""
    try:
        import routines
        return routines.due_today(day)
    except Exception:
        return None


# ── Dagens besked ────────────────────────────────────────────────────────────
def _meta(now: datetime, loaded) -> str:
    return swedish_date(now) + (f" · hämtad {loaded[11:16]}" if loaded else "")


def _besked_html(meta: str, sentence: str, absent: bool, reason: str = "") -> str:
    """Datum och hämtningstid (mono 12 px), rubriken DAGENS BESKED · ANTAL I DAG och meningen (15 px; grå när
    Gisten saknas, med orsaken en gång under)."""
    return ('<div class="home-besked">'
            f'<div style="color:{MUTED};font-family:{FONT_MONO};font-size:12px;line-height:1.3;">{_t(meta)}</div>'
            + section("Dagens besked · antal i dag")
            + f'<div style="color:{MUTED if absent else TEXT};font-size:15px;line-height:1.45;">{_t(sentence)}</div>'
            + (f'<div style="color:{MUTED};font-size:12px;line-height:1.4;margin-top:2px;">{_t(reason)}</div>'
               if reason else "")
            + "</div>")


def _tile_spec(n: dict) -> list:
    """De fyra rutorna: (nyckel, etikett, värde, färg, underrad, knappetikett, callback, args). Värdet är antalet ur
    home_today.counts; None blir Saknas-chipet (underraden är då orsaken i tooltipen). Noll står dämpat."""
    from ui.links import go

    def col(v, c):
        return c if v else MUTED

    turn, buy, due = n.get("turn"), n.get("buy"), n.get("due")
    if n.get("locked"):
        sell, sell_col, sell_sub = "—", MUTED, "🔒 Låst"
    else:
        sell = n.get("sell")
        sell_col = col(sell, RED)
        sell_sub = f"{n.get('vaksam')} VAKSAM" if sell is not None else "larmtillståndet gick inte att läsa"
    return [
        ("home_tile_turn", "⚡ Vändning", turn, col(turn, EMBER),
         f"{n.get('on_way') or 0} på väg" if turn is not None else "Valhalls karta saknas",
         "Vändningarna på Kartan →", go, ("valhall_map",)),
        ("home_tile_buy", "KÖP", buy, col(buy, GREEN),
         f"{n.get('watch')} BEVAKA" if buy is not None else "kandidaterna kunde inte hämtas",
         "KÖP i Thor →", go, ("thor",)),
        ("home_tile_sell", "Sälj · minska", sell, sell_col, sell_sub, "Säljlägena i Positioner →", go, ("holdings",)),
        ("home_tile_due", "Förfaller", due, col(due, AMBER_C),
         "rutiner i dag" if due is not None else "rutinerna gick inte att läsa",
         "Rutinerna i Årshjulet →", _goto, (_DUE_SLUG,)),
    ]


def _render_besked(d: dict, n: dict, cards: list, now: datetime) -> None:
    from tabs import home_today as ht
    sentence, absent = ht.besked(n, cards, d.get("board"))
    reason = ht.gist_reason(n.get("gist")) if absent else ""
    st.markdown(_besked_html(_meta(now, d.get("loaded")), sentence, absent, reason), unsafe_allow_html=True)
    with st.container(key="home_tiles"):
        for key, label, value, colour, sub, btn, cb, args in _tile_spec(n):
            tap_card(key, tile(label, value, colour, sub), btn, on_click=cb, args=args)


# ── Under vecket ─────────────────────────────────────────────────────────────
def _render_turns(cards: list) -> None:
    """VÄNDNING — ett glödkort per öppet köpfönster, tickerknapparna (home_turn_*) som en radbrytande rad."""
    for i, c in enumerate(cards):
        with st.container(key=f"home_turncard_{i}"):
            st.markdown(f'<div style="color:{EMBER};font-family:{FONT_UI};font-size:18px;font-weight:900;'
                        f'line-height:1.25;">{_t(_ht().turn_headline(c))}</div>'
                        f'<div style="color:{MUTED};font-size:12px;line-height:1.4;margin-top:2px;">'
                        f"{_t(c['where'])} · {_t(c.get('kind') or '')} bekräftad</div>", unsafe_allow_html=True)
            with st.container(horizontal=True, key=f"home_turnrow_{i}"):
                for kind, field, strat in _TURN_SOURCES:
                    for t in c[field]:
                        _open(f"{kind}: {t} → Thor", f"home_turn_{i}_{strat}_{t}", "thor", t, strat, width="content")
            e = c.get("etf") or {}
            if e.get("ticker"):
                _line(f"ETF: {_t(e['ticker'])} · {_t(e.get('exposure') or '')}"
                      + ("" if e.get("verified") else " · verifiera linjen hos Nordnet"), MUTED, "12px")


def _render_capital(d: dict) -> None:
    """KAPITALET — Lägets fyra ljus som ETT tryckkort (home_laget). Ljusen slås aldrig ihop."""
    from odin import laget
    from ui.links import go
    _block("Kapitalet", cls="home-kap")
    items = laget.capital_row(d.get("alloc"), d.get("regime"), d.get("risk"), d.get("board"))
    tap_card("home_laget", light_grid(items), "Läget i Odin →", on_click=go, args=("laget",))
    note(f"<span style='color:{MUTED};font-size:12px;'>Ljusen slås inte ihop.</span>")


def _kval_html(c, parts: list, title: str) -> str:
    """En kvalrad: [KÖP] TICKER [källans ord] [strategi] [NY], Storlek-radens tal i mono och GAMMAL-märket.
    title = hela Storlek-raden (size_line) som tooltip — samma tal, inget nytt."""
    from thor import registry as reg
    from ui import components as cp
    from ui import vocab
    own = "" if vocab._norm(c.status) == vocab._norm(c.status_class) else c.status
    chips = "".join(f" {x}" for x in (source_chip(own) if own else "", source_chip(reg.label(c.strategy)),
                                      source_chip("NY") if c.new else "") if x)
    sep = f'<span style="color:{MUTED};"> · </span>'
    sizes = (f'<div class="k-size" style="color:{MUTED};font-family:{FONT_MONO};font-size:13px;line-height:1.4;">'
             + sep.join(f'<span style="white-space:nowrap;">{_t(p)}</span>' for p in parts) + "</div>" if parts
             else f'<div class="k-size">{missing("nivåer saknas")}</div>')
    stale = _ht().stale_tag(c).lstrip(" ·")
    flag = (f'<div class="k-flag" style="color:{AMBER_C};font-family:{FONT_UI};font-size:12px;font-weight:600;'
            f'line-height:1.3;">{_t(stale)}</div>' if stale else "")
    return (f'<div class="home-kval" title="{escape(title, quote=True)}">'
            f'<div class="k-chip" style="min-width:0;overflow:hidden;">{cp.chip(c.status_class, source=False)}</div>'
            f'<div style="color:{TEXT};font-family:{FONT_UI};font-size:16px;font-weight:700;line-height:1.4;'
            f'min-width:0;overflow-wrap:anywhere;">{_t(c.ticker)}{chips}</div>{sizes}{flag}</div>')


def _render_kvalar(d: dict, n: dict) -> None:
    """KVALAR OCH STORLEK — antalet till höger, de fem första KÖP-raderna i Thors ordning som tryckrader
    (home_buy_<strategi>_<ticker>, etiketterna som förut) och 'Hela listan i Thor →' (home_thor)."""
    from thor import registry as reg
    from ui import settings as us
    from ui.components import link_rows
    from ui.links import go
    ht = _ht()
    thor = d.get("thor") or {}
    _n_buy, _n_watch, top = ht.qualifiers(thor.get("rows"))
    count = "" if n.get("gist") else [(f"{n.get('buy')} KÖP", GREEN), (f"{n.get('watch')} BEVAKA", AMBER_C)]
    _block("Kvalar och storlek", count=count)
    if n.get("gist"):
        st.markdown(f'<div>{missing("kandidaterna kunde inte hämtas — se Dagens besked")}</div>',
                    unsafe_allow_html=True)
    capital = us.capital() or 100000.0
    with st.container(key="home_kval"):
        for c in top:
            tap_card(f"home_buy_{c.strategy}_{c.ticker}",
                     _kval_html(c, ht.size_parts(c, capital), ht.size_line(c, capital)),
                     f"{c.ticker} → Thor ({reg.label(c.strategy)})", on_click=go, args=("thor", c.ticker, c.strategy))
    if not top and not n.get("gist"):
        _line("Inga KÖP i dag.", MUTED, "14px")
    with link_rows("home"):
        st.button("Hela listan i Thor →", key="home_thor", type="tertiary", on_click=go, args=("thor", "", ""))


_SELL_COLOR = {"SÄLJ": RED, "MINSKA": AMBER_C, "VAKSAM": AMBER_C}


def _card(body: str, rule: str = "") -> str:
    """Kortet kring en rad eller en lista (BG_CARD, radie 12, valfri 3 px linje)."""
    line = f"border-left:3px solid {rule};" if rule else ""
    return (f'<div style="background:{BG_CARD};border:1px solid {BORDER_CARD};{line}border-radius:12px;'
            f'padding:4px 14px;min-height:56px;box-sizing:border-box;display:flex;flex-direction:column;'
            f'justify-content:center;">{body}</div>')


def _render_sell(d: dict) -> None:
    """SÄLJ OCH BEVAKA — ETT tryckkort (home_holdings): innehavens SÄLJ/MINSKA/VAKSAM-rader, annars raden när inget
    ska säljas, och bakom låset bara att innehaven är låsta."""
    from portfolio import private_lock
    from ui.components import action_row
    from ui.links import go
    ht = _ht()
    _block("Sälj och bevaka")
    if not private_lock.unlocked():        # innehaven bakom lösenord (HOLDINGS_PASSWORD)
        body = (f'<div style="color:{MUTED};font-size:14px;line-height:1.45;padding:10px 0;">'
                f'🔒 Innehaven är låsta — lås upp under PORTFÖLJ → Positioner</div>')
    else:
        lines = ht.holdings_lines(d.get("alert_state"))
        body = "".join(action_row(w, _SELL_COLOR.get(w, AMBER_C), t, why) for t, w, why in lines) or \
            (f'<div style="color:{MUTED};font-size:14px;line-height:1.45;padding:10px 0;">'
             f'{_t(ht.all_hold_line(d.get("alert_state"), d.get("n_open")))}</div>')
    tap_card("home_holdings", _card(body), "Positioner →", on_click=go, args=("holdings",))


def _render_goal(d: dict) -> None:
    """MOT MÅLET — guldlinjat tryckkort med mm.today_line (home_goal)."""
    import journal_stats as js
    from portfolio import mot_malet as mm
    from trade_journal import start_capital
    from ui.links import go
    _block("Mot målet")
    line = mm.today_line(js.realized_cagr(d.get("trades") or [], start_capital()), d.get("paper"), d.get("trades"))
    body = f'<div style="color:{TEXT};font-size:15px;line-height:1.45;padding:10px 0;">{_t(line)}</div>'
    tap_card("home_goal", _card(body, GOLD_C), "Mot målet →", on_click=go, args=("goal",))


def _alert_html(r: dict) -> str:
    """En larmrad: rubriken 14 px, tiden i mono 12 px MUTED."""
    return (f'<div style="min-height:48px;padding:8px 0;box-sizing:border-box;border-bottom:1px solid {BORDER_FAINT};'
            f'display:flex;flex-direction:column;justify-content:center;">'
            f'<div style="color:{TEXT};font-family:{FONT_UI};font-size:14px;font-weight:700;line-height:1.35;'
            f'overflow-wrap:anywhere;">{_t(r.get("title", ""))}</div>'
            f'<div style="color:{MUTED};font-family:{FONT_MONO};font-size:12px;line-height:1.3;">'
            f'{_t(r.get("time", ""))}</div></div>')


def _alert_row(i: int, r: dict) -> None:
    from ui.links import go
    if r.get("target"):
        tap_card(f"home_alert_{i}", _alert_html(r), "Öppna →", on_click=go,
                 args=(r["target"], r.get("ticker") or "", r.get("strategy") or ""))
    else:
        st.markdown(_alert_html(r), unsafe_allow_html=True)


SHOW_ALERTS = 5


def _render_alert_log(d: dict) -> None:
    """SENASTE LARMEN — de fem senaste (tryckrader home_alert_<i> när larmet har ett mål), resten i 'Visa alla (n)'."""
    from alerts import history
    rows = history.latest(d.get("alert_log"), 8)
    _block("Senaste larmen", count=str(len(rows)) if rows else "")
    if not rows:
        _line("Inga sparade larm än — loggen fylls av nästa schemalagda körning som skickar ett.", MUTED, "14px")
        return
    with st.container(key="home_alerts"):
        for i, r in enumerate(rows[:SHOW_ALERTS]):
            _alert_row(i, r)
    if len(rows) > SHOW_ALERTS:
        with st.expander(f"Visa alla ({len(rows)})", expanded=False):
            for i, r in enumerate(rows[SHOW_ALERTS:], SHOW_ALERTS):
                _alert_row(i, r)


def _render_due(due) -> None:
    """I dag förfaller — Årshjulets rutiner vid namn (rutan FÖRFALLER visar antalet)."""
    if not due:
        return
    _block("I dag förfaller")
    st.markdown(" ".join(f"<span style='display:inline-block;border:1px solid {AMBER_C};border-radius:6px;"
                         f"padding:2px 8px;margin:2px;color:{AMBER_C};font-size:12px;line-height:1.4;'>"
                         f"{_t(r.title)} · {_t(r.minutes)}</span>" for r in due), unsafe_allow_html=True)


# ── Sidan ────────────────────────────────────────────────────────────────────
def tab_home() -> None:
    from portfolio import private_lock
    from tabs import home_today as ht
    from tabs.home_today import now_se
    from ui import routine
    from ui.shell import page_style
    with st.container(key="home_page"):
        page_style(_CSS, "home")
        d = _today_data()
        now = now_se()
        due = _due_today(now.date())
        n = ht.counts(d, private_lock.unlocked(), today=now, due=due)
        cards = ht.turn_cards(d.get("board"))

        with head_row("home"):
            page_header(nav.TOP_LABEL["home"])
            if st.button("📡 Uppdatera", key="home_refresh"):
                st.session_state["_home_refresh"] = True
                st.rerun()
            routine.pill("home")
        _render_besked(d, n, cards, now)

        with st.container(key="home_cols"):
            left, right = st.columns([3, 2], gap="medium")
            with left:
                _render_turns(cards)
                _render_capital(d)
                _render_kvalar(d, n)
            with right:
                _render_sell(d)
                _render_goal(d)
                _render_alert_log(d)
                _render_due(due)

        routine.footer("home")
        with st.expander("🗺 Flikguiden — var allt ligger", expanded=False):
            _render_zones()
            note(f"<span style='font-size:13px;'>Adressfältet följer fliken — kopiera URL:en (t.ex. "
                 f'<span style="font-family:{FONT_MONO};font-size:12px;white-space:nowrap;">?p=odin/laget</span>) '
                 f"för en direktlänk. "
                 f"Alla länkar står i MER → Regler & Guider → 🗺 FLIKGUIDE.</span>")
