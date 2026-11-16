"""
portfolio/broker_sync_ui.py — rutan "📥 Uppdatera från mäklaren" i PORTFÖLJ → Positioner (bakom innehavslåset).

Ladda upp Nordeas innehavslista (.xlsx) och/eller Nordnets innehavslista (.csv) — eller Nordnets transaktioner
(.csv eller .pdf). Rutan visar vad som är nytt, ändrat, borta och oförändrat mot Positioner. Inget skrivs förrän du
trycker "Uppdatera markerade" (en sparning) eller "Stäng" på en rad som saknas hos mäklaren. Att öppna sidan eller
ladda upp en fil skriver ingenting; Börsdata och Yahoo frågas bara när du trycker "Hitta tickers".
"""
from __future__ import annotations

import html

import streamlit as st

from portfolio import broker_sync as bs

BD_MAP = "bsync_bd_map"
YH_MAP = "bsync_yh_map"          # {yahoo_key: ticker} ur Yahoo-uppslaget
YH_TRIED = "bsync_yh_tried"      # innehav som redan sökts hos Yahoo (knappen visas inte igen för dem)
RESULT = "bsync_result"
PARSED = "bsync_parsed"


def _esc(text) -> str:
    return html.escape(str(text or "")).replace("$", "&#36;")


def _fmt(v) -> str:
    if v is None:
        return "–"
    from portfolio.position_hints import fmt_num
    return fmt_num(round(float(v), 4))


def _bd_lookup() -> str:
    """Börsdatas instrument (Norden + globalt) → {ISIN: ticker} i sessionen. Returnerar en varning eller ''."""
    try:
        from borsdata_api import BorsdataAPI
        api = BorsdataAPI()
        if not api.is_configured:
            st.session_state[BD_MAP] = {}
            return "Börsdata-nyckeln saknas"
        insts = list(api.get_instruments() or []) + list(api.get_global_instruments_list() or [])
        st.session_state[BD_MAP] = bs.isin_map(insts)
    except Exception as exc:
        st.session_state[BD_MAP] = {}
        return f"Börsdata svarade inte ({type(exc).__name__})"
    return ""


def _lookup(pending: list) -> None:
    """on_click: Börsdata (ISIN) en gång, sedan Yahoo (namn och börs) för innehaven som fortfarande saknar ticker —
    Nordnets innehavslista har inget ISIN för USA och Kanada, så Börsdata hittar dem inte."""
    warn = _bd_lookup() if BD_MAP not in st.session_state else ""
    bd = st.session_state.get(BD_MAP) or {}
    tried = st.session_state.setdefault(YH_TRIED, set())
    rest = [h for h in pending if not (h.isin and h.isin.upper() in bd) and bs.yahoo_key(h) not in tried]
    found = st.session_state.setdefault(YH_MAP, {})
    try:
        hits = bs.yahoo_tickers(rest)
        found.update(hits)
    except Exception as exc:
        hits = {}
        warn = (warn + " · " if warn else "") + f"Yahoo svarade inte ({type(exc).__name__})"
    tried.update(bs.yahoo_key(h) for h in rest)
    left = len({bs.yahoo_key(h) for h in rest} - set(hits))
    if warn or left:
        st.session_state[RESULT] = ("warning", " · ".join(x for x in (
            warn, f"{left} utan säker ticker — skriv den i rutan" if left else "") if x) + ".")


def _close(row_id: str, ticker: str, key: str) -> None:
    """on_click: stäng positionen (positions.close → stängda + journalen). Säljkursen är valfri."""
    import positions
    px = st.session_state.get(key)
    closed = positions.close(row_id=row_id, exit_price=px if px else None, reason="Såld enligt mäklaren")
    st.session_state[RESULT] = (("success", f"{ticker} stängd.") if closed else
                                ("warning", f"{ticker}: raden finns inte längre."))


def _apply(changes: list) -> None:
    """on_click: de markerade raderna till registret i en sparning."""
    picked = [c for c in changes if st.session_state.get(_ck(c), c.pick)]
    typed = {c.key: st.session_state.get(f"bsync_t_{c.key}", "") for c in picked if not c.ticker}
    done, skipped, err = bs.apply(picked, typed)
    msg = f"Uppdaterat: {', '.join(done)}." if done else "Inget uppdaterades."
    if skipped:
        msg += " Hoppade över: " + "; ".join(skipped) + "."
    if err:
        st.session_state[RESULT] = ("error", f"{msg} Men INTE sparat i GitHub: {err}")
    else:
        st.session_state[RESULT] = ("success" if done else "warning", msg)


def _line(c: bs.Change) -> str:
    who = f"<b>{_esc(c.ticker or '?')}</b> {_esc(c.name)}"
    accts = ", ".join(bs.BROKER_LABEL.get(a.split(':', 1)[0], a) for a in c.accounts)
    unit = f" {_esc(c.currency)}" if c.currency else ""
    if c.kind == bs.CHANGED:
        q = (f"antal {_fmt(c.old_qty)} → <b>{_fmt(c.qty)}</b>" if not bs._same(c.old_qty, c.qty, 1e-6)
             else f"antal {_fmt(c.qty)}")
        p = (f" · snittkurs {_fmt(c.old_price)} → <b>{_fmt(c.price)}</b>{unit}"
             if c.price is not None and not bs._same(c.old_price, c.price, 5e-4) else "")
        body = q + p
    elif c.kind == bs.MISSING or (c.qty is None and c.old_qty is not None):
        body = f"raden: {_fmt(c.old_qty)} st à {_fmt(c.old_price)}"
    else:
        body = f"{_fmt(c.qty)} st" + (f" à {_fmt(c.price)}{unit}" if c.price is not None else " · snittkurs saknas")
    extra = " · ".join(_esc(n) for n in c.notes)
    return (f"{who} — {body}" + (f" <span style='opacity:.7'>· {_esc(accts)}</span>" if accts else "")
            + (f"<br><span style='opacity:.7;font-size:.85em'>{extra}</span>" if extra else ""))


_MD_SPECIAL = "\\`*_{}[]()#+-.!|<>$~"


def _md(text) -> str:
    """Text i en kryssrutas etikett (markdown, ingen HTML): specialtecken skrivs bokstavligt."""
    return "".join("\\" + ch if ch in _MD_SPECIAL else ch for ch in str(text or ""))


def _label(c: bs.Change) -> str:
    """Kryssrutans etikett — samma innehåll som _line, i markdown, så rutan och texten står på samma rad även på
    mobilen (kolumner staplas under varandra där)."""
    who = f"**{_md(c.ticker or '?')}** {_md(c.name)}"
    unit = f" {_md(c.currency)}" if c.currency else ""
    if c.kind == bs.CHANGED:
        q = (f"antal {_md(_fmt(c.old_qty))} → **{_md(_fmt(c.qty))}**" if not bs._same(c.old_qty, c.qty, 1e-6)
             else f"antal {_md(_fmt(c.qty))}")
        p = (f" · snittkurs {_md(_fmt(c.old_price))} → **{_md(_fmt(c.price))}**{unit}"
             if c.price is not None and not bs._same(c.old_price, c.price, 5e-4) else "")
        body = q + p
    else:
        body = f"{_md(_fmt(c.qty))} st" + (f" à {_md(_fmt(c.price))}{unit}" if c.price is not None
                                           else " · snittkurs saknas")
    return f"{who} — {body}"


def _notes(c: bs.Change) -> str:
    accts = ", ".join(bs.BROKER_LABEL.get(a.split(':', 1)[0], a) for a in c.accounts)
    return " · ".join([accts] + list(c.notes)) if accts else " · ".join(c.notes)


def _ready(c: bs.Change) -> bool:
    """Markerad och går att spara: har en ticker, eller en skriven i rutan."""
    return bool(st.session_state.get(_ck(c), c.pick)) and bool(
        c.ticker or str(st.session_state.get(f"bsync_t_{c.key}", "") or "").strip())


def _ck(c: bs.Change) -> str:
    """Kryssrutans nyckel bär förslaget (kryssad från början, antal, kurs). Streamlit behåller en nycklad rutas värde
    även när value= ändras — ändras förslaget (en fil till, Börsdata-klicket) börjar rutan om från planens val i
    stället för att en gammal bock står kvar på ett kort som nu inte ska kryssas."""
    return f"bsync_ck_{c.key}:{int(c.pick)}:{c.qty}:{c.price}"


def _read(f) -> tuple:
    """bs.read med sessionscache per fil (namn + innehållets hash): en PDF på hundra sidor tolkas en gång, inte vid
    varje klick. Innehaven kopieras — combine och resolve skriver ISIN och ticker på dem."""
    import copy
    import hashlib
    data = f.getvalue() if hasattr(f, "getvalue") else f.read()
    key = f"{f.name}:{hashlib.sha1(data).hexdigest()}"
    cache = st.session_state.setdefault(PARSED, {})
    if key not in cache:
        cache.clear() if len(cache) > 8 else None
        cache[key] = bs.read(f.name, data)
    return copy.deepcopy(cache[key])


def _uploader():
    return st.file_uploader("Mäklarens filer", type=["xlsx", "csv", "pdf"], accept_multiple_files=True,
                            key="bsync_files")


def render_broker_sync() -> None:
    import positions
    with st.expander("📥 Uppdatera från mäklaren (Nordea ASK · Nordnet ISK)", expanded=False):
        st.caption("Nordea: Innehav → Exportera → **My holdings (.xlsx)**. Nordnet: kontots **innehavslista** "
                   "(.csv, filen heter aktier_kontonummer-…) — säkrast, antal och snittkurs läses rakt av. Nordnets "
                   "transaktioner (CSV eller PDF) går också: laddas de upp tillsammans med innehavslistan får "
                   "aktierna i USA och Kanada sitt ISIN därifrån; ensamma räknas innehaven fram ur köp och sälj och "
                   "kryssas inte i när filen kan sakna äldre köp. Antal och snittkurs uppdateras — stopp, mål, "
                   "strategi och anteckningar rörs aldrig. Inget sparas förrän du trycker på knappen.")
        files = _uploader()
        res = st.session_state.pop(RESULT, None)
        if res:
            getattr(st, res[0])(res[1])
        if not files:
            return
        reads = []
        for f in files:
            broker, hs, errors, kind = _read(f)
            for e in errors[:8]:
                st.warning(f"{f.name}: {e}")
            if len(errors) > 8:
                st.caption(f"{f.name}: {len(errors) - 8} varningar till.")
            if broker:
                st.caption(f"✓ {f.name}: {bs.BROKER_LABEL[broker]} ({kind}), {sum(1 for h in hs if h.qty)} innehav.")
            reads.append((f.name, broker, hs, kind))
        # per konto ersätter en senare fil en tidigare; Nordnets innehavslista går före transaktionerna (combine)
        holdings, uploaded, msgs = bs.combine(reads)
        for level, text in msgs:
            getattr(st, level)(text)
        if not uploaded:                               # en läst fil utan innehav = allt sålt, och ska visas
            return
        rows = positions.all_positions()
        bs.resolve(holdings, rows, st.session_state.get(BD_MAP), st.session_state.get(YH_MAP))
        tried = st.session_state.get(YH_TRIED, set())
        unsure_t = [h for h in holdings if h.qty and (not h.ticker or h.how == bs.BY_NAME)]
        pending = [h for h in unsure_t if BD_MAP not in st.session_state or bs.yahoo_key(h) not in tried]
        if pending:
            st.button(f"🔎 Hitta tickers i Börsdata och Yahoo ({len(pending)} utan säker ticker)",
                      key="bsync_bd", on_click=_lookup, args=(pending,))
        changes = bs.plan(rows, holdings, uploaded)
        actionable = [c for c in changes if c.kind in (bs.NEW, bs.CHANGED)]
        titles = {bs.NEW: "🆕 Nytt hos mäklaren — läggs till som Untagged",
                  bs.CHANGED: "✏️ Ändrat", bs.MISSING: "❌ Saknas hos mäklaren",
                  bs.INFO: "ℹ️ Ändra själv", bs.SAME: "✓ Stämmer"}
        for kind in (bs.NEW, bs.CHANGED):
            group = [c for c in changes if c.kind == kind]
            if not group:
                continue
            st.markdown(f"**{titles[kind]}** ({len(group)})")
            for c in group:
                st.checkbox(_label(c), value=c.pick, key=_ck(c))
                if not c.ticker:
                    st.text_input(f"Ticker för {c.name or c.isin}", key=f"bsync_t_{c.key}",
                                  placeholder="t.ex. HUSQ-B.ST — tomt = hoppas över")
                if _notes(c):
                    st.caption(_notes(c))
        if actionable:
            n = sum(1 for c in actionable if _ready(c))
            st.button(f"✅ Uppdatera markerade ({n})", key="bsync_apply", type="primary", disabled=n == 0,
                      on_click=_apply, args=(actionable,))
        gone = [c for c in changes if c.kind == bs.MISSING]
        if gone:
            st.markdown(f"**{titles[bs.MISSING]}** ({len(gone)})")
            for c in gone:
                a, b, d = st.columns([0.6, 0.22, 0.18], vertical_alignment="center")
                a.markdown(_line(c), unsafe_allow_html=True)
                pkey = f"bsync_px_{c.row['id']}"
                b.number_input("Säljkurs", min_value=0.0, value=None, key=pkey, placeholder="säljkurs (valfri)",
                               label_visibility="collapsed")
                d.button("Stäng", key=f"bsync_close_{c.row['id']}", on_click=_close,
                         args=(c.row["id"], c.ticker, pkey))
        for kind in (bs.INFO, bs.SAME):
            group = [c for c in changes if c.kind == kind]
            if group:                                  # ingen expander här — Streamlit tillåter inte expander i expander
                st.markdown(f"**{titles[kind]}** ({len(group)})")
                st.markdown("<br>".join(_line(c) for c in group), unsafe_allow_html=True)
        if not actionable and not gone and not any(c.kind == bs.INFO for c in changes):
            st.success("Positioner stämmer med mäklarens filer.")
