"""
lukacs_ui.py — FV-modulen som sektion i CSM-blocket.

Ligger i CSM-expandern eftersom frågorna hänger ihop: CSM säger om bolaget
överlever varje scenario, FV vad det är värt där. Muterar raden och returnerar
True om något ändrats, precis som de andra kontrollsektionerna.
"""

from __future__ import annotations

import re

import streamlit as st

import storage

import controls as ctl
import lukacs as fv

TEXT, DIM = "#e8e4dc", "#8a8578"


def _class_label(q) -> str:
    band = (f"{q.lo:g}–{q.hi:g} %" if q.lo is not None
            else "värderas ej på FCF")
    return f"{q.code} — {q.desc} Target-yield {band}"


def _refresh_row(sheet: str, key: str):
    """Sifferuppdateringens rad för bolaget (kurs, valuta, FCF, EV) — None utan blob. Yahoo-reservens rader saknar
    valuta: den läses ur tickerns suffix (VAR.OL → NOK; London i pence), som i positionernas omräkning."""
    if not sheet:
        return None
    try:
        import refresh_ui
        r = ((refresh_ui.current_blob() or {}).get("rows") or {}).get(f"{sheet}:{key}")
        if not isinstance(r, dict):
            return None
        if not r.get("currency") and r.get("source") == "yfinance" and r.get("price") is not None:
            import positions
            t = str(r.get("ticker") or "").strip().upper()
            ccy = "GBX" if t.endswith(".L") else positions.currency_for(t)
            if ccy != "USD":
                r = dict(r, currency=ccy)
        return r
    except Exception:
        return None


def bd_fcf_line(row: dict, key: str, sheet: str, ref=None) -> None:
    """'🤖 Börsdata FCF 12 mån: X MUSD' vid FCF-rutorna — bara en referens att jämföra med (senaste tolv månaderna,
    inte normaliserad), ingen knapp. Visas bara när talet säkert står i USD (lukacs.ref_fcf_musd)."""
    if ref is None:
        ref = _refresh_row(sheet, key)
    v = fv.ref_fcf_musd(ref)
    if v is None:
        return
    txt = (f"{v:,.0f}" if abs(v) >= 10 else f"{v:,.1f}").replace(",", " ").replace(".", ",").replace("-", "−")
    asof = str(ref.get("asof") or "")[:10]
    note = ""
    try:
        import refresh_ui
        note = refresh_ui.matched_note(refresh_ui.current_blob(), sheet, row, key)
    except Exception:
        pass
    st.markdown(f"<span style='color:{DIM};font-size:0.78rem;'>🤖 Börsdata FCF 12 mån: {txt} MUSD"
                f"{f' ({asof})' if asof else ''}{note} — referens, inte normaliserad.</span>",
                unsafe_allow_html=True)


SHARES_SCALE_MAX = 10.0     # framtida antal aktier mer än så många gånger från Börsdatas → troligen fel enhet


def _sv_num(v: float) -> str:
    """Tal med svenskt decimalkomma och mellanslag mellan tusental, utan onödiga decimaler: 1 234,5 · 0,0012."""
    return f"{v:,.{2 if abs(v) >= 1 else 4}f}".rstrip("0").rstrip(".").replace(",", " ").replace(".", ",")


def shares_scale_warning(typed, bd_shares, year=None, source: str = "Börsdata") -> str:
    """Varningstext när framtida antal aktier ligger mer än SHARES_SCALE_MAX× från uppdateringens aktieantal
    (miljoner) — tusental i stället för miljoner gör FV per aktie lika många gånger fel, och ett för litet tal kan
    göra steg 5 grönt. source = var aktieantalet kommer ifrån (refresh_ui.source_name: 'Börsdata', eller
    'Yahoo (ABC.V)' för Yahoo-reservens rad → 'Yahoos 135 miljoner (ABC.V)'). Bara text: ingen räkning eller grind
    ändras. Tom sträng när något tal saknas eller de ligger nära."""
    t, b = fv.num(typed), fv.num(bd_shares)
    if not t or not b or t <= 0 or b <= 0:
        return ""
    yr = f", årsrapporten {int(year)}" if fv.num(year) else ""
    who = str(source or "").strip() or "Börsdata"
    m = re.fullmatch(r"(.+?)\s*\((.+)\)", who)          # 'Yahoo (ABC.V)': genitiv på namnet, symbolen efter talet
    name, sym = (m.group(1), m.group(2)) if m else (who, "")
    ref = f"{name}s {_sv_num(b)} miljoner" + (f" ({sym}{yr})" if sym else yr)
    if t > SHARES_SCALE_MAX * b:
        k = f"{t / b:,.0f}".replace(",", " ")
        return (f"Framtida antal aktier {_sv_num(t)} är {k}× {ref}. Står talet i tusental? "
                f"Rutan är i miljoner — FV per aktie blir då {k}× för låg.")
    if t * SHARES_SCALE_MAX < b:
        k = f"{b / t:,.0f}".replace(",", " ")
        return (f"Framtida antal aktier {_sv_num(t)} är bara 1/{k} av {ref}. Står talet i miljarder? "
                f"Rutan är i miljoner — FV per aktie blir då {k}× för hög och kan göra köpgrindens steg 5 grönt.")
    return ""


def _shares_suggestion(row: dict, key: str, sheet: str, ref, shares, widget_key: str) -> None:
    """Uppdateringens aktieantal (shares_now_m, med året ur shares_year) som förslag med Använd vid 'Framtida antal
    aktier' — samma refresh_ui.suggest som kortets andra förslag — och en varning, bara text, när talet i rutan
    ligger mer än SHARES_SCALE_MAX× därifrån. Varningen säger källan: Börsdata, eller Yahoo med symbolen när
    'Hämta nu' fick Yahoo-reservens rad."""
    yr = fv.num((ref or {}).get("shares_year"))
    label = "aktier i dag (miljoner" + (f", årsrapporten {int(yr)}" if yr else "") + ")"
    src = "Börsdata"
    try:
        import refresh_ui
        src = refresh_ui.source_name(ref) or src
        refresh_ui.suggest(sheet, row, "framtida_antal_aktier", "shares_now_m", label, widget_key, row_id=key)
    except Exception:
        pass
    w = shares_scale_warning(shares, (ref or {}).get("shares_now_m"), yr, source=src)
    if w:
        st.warning(w)


def holding_hint(ticker, find) -> str:
    """'💡 Positionerna har X …' när registret har bolaget i någon av tickerns former (thor.ticker.forms) — ett
    textförslag, rutan kryssas aldrig. find = positions.find. Tom sträng utan träff. Anroparen frågar bara när
    innehaven är upplåsta."""
    t = str(ticker or "").strip()
    if not t:
        return ""
    try:
        from thor.ticker import forms
        fs = forms(t)
    except Exception:
        fs = [t.upper()]
    for f in fs:
        hit = find(f)
        if hit:
            return f"💡 Positionerna har {hit.get('ticker') or f} — kryssa om det här är ett innehav."
    return ""


def _holding_hint(row: dict) -> None:
    """holding_hint under kryssrutan. Tyst när innehaven är låsta (portfolio.private_lock) — låst avslöjar arket
    inte vad som ägs — och när registret inte går att läsa."""
    try:
        from portfolio import private_lock
        if not private_lock.unlocked():
            return
        import positions
        txt = holding_hint(row.get("ticker"), positions.find)
    except Exception:
        return
    if txt:
        st.caption(txt)


def delev_years_hint(ref) -> str:
    """Uppskattningen under 'År till låg skuld': '≈ 2,5 år (nettoskuld ÷ Börsdatas FCF 12 mån, grovt)' ur nattens
    nettoskuld och FCF 12 mån (lukacs.ref_net_debt_musd / ref_fcf_musd — samma valutaspärr), eller 'nettokassa' när
    nettoskulden är 0 eller lägre. Bara text: ingen Använd och ingen effekt på grinden — FCF bakåt är ett dåligt mått
    på den framtida amorteringen. Tom sträng när ett tal saknas, valutan är okänd eller FCF 12 mån inte är positiv."""
    if not isinstance(ref, dict) or not (ref.get("currency") or ref.get("report_currency")):
        return ""                                       # ingen valuta alls: talen kan inte sägas stå i USD
    nd = fv.ref_net_debt_musd(ref)
    if nd is None:
        return ""
    if nd <= 0:
        return "nettokassa"
    fcf = fv.ref_fcf_musd(ref)
    if fcf is None or fcf <= 0:
        return ""
    years = nd / fcf
    txt = "under 0,1" if years < 0.05 else "≈ " + f"{years:,.1f}".replace(",", " ").replace(".", ",")
    return f"{txt} år (nettoskuld ÷ Börsdatas FCF 12 mån, grovt)"


def _delev_years_line(row: dict, key: str, sheet: str, ref) -> None:
    """delev_years_hint under rutan, med datum och 'matchat:' när Börsdatas bolag inte liknar radens namn."""
    txt = delev_years_hint(ref)
    if not txt:
        return
    asof = str(ref.get("asof") or "")[:10]
    note = ""
    try:
        import refresh_ui
        note = refresh_ui.matched_note(refresh_ui.current_blob(), sheet, row, key)
    except Exception:
        pass
    st.markdown(f"<span style='color:{DIM};font-size:0.78rem;'>🤖 Börsdata{f' {asof}' if asof else ''}: {txt}"
                f"{note} — uppskattning, fylls inte i.</span>", unsafe_allow_html=True)


FX_MISS_RETRY_S = 900
_FX_MISS: dict = {}     # {'SEKUSD=X': tid för senaste misslyckade hämtning} — inte en Yahoo-fråga per kort och omkörning


def _live_fx(ccy: str):
    """Dagens växelkurs till USD (USD per enhet, datum) ur panelens kurslager (Yahoo '{CCY}USD=X'; en träff cachas
    6 h där) — None när den inte går att hämta, och då försöker vi inte igen på 15 min. Pence (GBX) = GBP/100."""
    pair = fv.fx_pair(ccy)
    if not pair:
        return None
    import time
    if time.time() - _FX_MISS.get(pair[0], 0.0) < FX_MISS_RETRY_S:
        return None
    try:
        import market_prices
        s = market_prices.close(pair[0], "1mo")
        if s is not None and not s.empty:
            rate = float(s.iloc[-1]) / pair[1]
            if rate > 0:
                return rate, str(s.index[-1])[:10]
    except Exception:
        pass
    _FX_MISS[pair[0]] = time.time()
    return None


def _use_usd(row: dict, value: float, widget_key: str) -> None:
    """on_click: kursen i USD in i raden och rutan (samma mönster som refresh_ui._apply)."""
    row["aktuell_kurs"] = value
    st.session_state[widget_key] = value


ROUGH_NEAR = 0.15   # med den grova tabellkursen: en USD-kurs i rutan inom ±15 % ersätts inte med ett klick


def _usd_price(price, cur: dict, ref: dict, row: dict, widget_key: str) -> None:
    """Uppdateringens kurs i en annan valuta än USD: förslaget omräknat till USD (forward FCF räknas i MUSD), och en
    varning när kursen i rutan ser ut att stå i den lokala valutan — då jämför säkerhetsmarginalen olika valutor.
    price = rutans värde i den här körningen (inte radens förra), så varningen följer det man just skrivit.
    Omräkningen sker med dagens växelkurs; utan den med den grova fasta tabellen, som kan ligga flera procent fel —
    då erbjuds den inte i stället för en USD-kurs man redan skrivit (felet går åt det optimistiska hållet)."""
    def _sv(v, d=None):
        d = (4 if abs(v) < 1 else 2) if d is None else d
        return f"{v:,.{d}f}".replace(",", " ").replace(".", ",")
    src = "Yahoo" if ref.get("source") == "yfinance" else "Börsdata"
    ccy, usd = cur["currency"], cur["usd"]
    now = fv.num(price)
    fx_note = ""
    if cur["fx"]:
        fx_note = (f" (1 {ccy} = {_sv(cur['fx'], 4 if cur['fx'] >= 0.01 else 6)} USD, "
                   + (f"växelkurs {cur['fx_asof'] or 'i dag'})" if cur["fx_src"] == "live" else "grov tabellkurs)"))
    if (cur["stored_is_local"] or not now) and fv.fcf_in_local_currency(row, ref) == ccy:
        # FCF i miljoner lokal valuta: en USD-kurs bredvid den (bytt eller ny i en tom ruta) gör marginalen ~10× för
        # god. Ingen knapp förrän FCF:en räknats om — men växelkursen visas, den behövs för omräkningen.
        if cur["stored_is_local"]:
            st.warning(f"Kursen {_sv(price)} ser ut att stå i {ccy} — och forward FCF ser ut att stå i miljoner {ccy} "
                       f"(se varningen under scenarierna). Räkna om båda till USD{fx_note}: FCF först, sedan kursen.")
        else:
            st.warning(f"Forward FCF ser ut att stå i miljoner {ccy} (se varningen under scenarierna). Räkna om den "
                       f"till USD{fx_note} först — sedan föreslås kursen i USD.")
        return
    if (cur["stored_is_local"] or not now) and fv.fcf_maybe_local(row, ref) == ccy:
        # Kan vara ett riktigt billigt bolag — eller FCF i miljoner {ccy}. Inget råd att ändra talen, men inget klick.
        lead = f"Kursen {_sv(price)} ser ut att stå i {ccy}. " if cur["stored_is_local"] else ""
        st.warning(f"{lead}Base-FCF:en är över 40 % av bolagets storlek men under 1,5× EBITDA — ett riktigt billigt "
                   f"bolag, eller FCF i miljoner {ccy}. Kontrollera att forward FCF står i MUSD och skriv sedan kursen "
                   f"i USD{fx_note}.")
        return
    if cur["stored_is_local"]:
        st.warning(f"Kursen {_sv(price)} ser ut att stå i {ccy} ({src}s kurs: {_sv(cur['local'])} {ccy}), men "
                   f"forward FCF räknas i USD (MUSD) — säkerhetsmarginalen jämför då olika valutor. "
                   + ("Använd kursen i USD." if usd else "Skriv kursen i USD."))
    asof = str(ref.get("asof") or "")[:10]
    head = f"<span style='color:{DIM};font-size:0.78rem;'>🤖 {src} {asof}: kurs {_sv(cur['local'])} {ccy}"
    if usd is None:                                     # ingen växelkurs alls — ingen omräkning
        if now is None:
            st.markdown(f"{head} — valutan finns varken i dagens växelkurser eller den fasta tabellen; skriv kursen "
                        f"i USD.</span>", unsafe_allow_html=True)
        return
    if usd <= 0 or (now is not None and abs(now - usd) < 1e-6):
        return
    rate = f"1 {ccy} = {_sv(cur['fx'], 4 if cur['fx'] >= 0.01 else 6)} USD"
    if cur["fx_src"] == "live":
        note = f"{rate}, växelkurs {cur['fx_asof'] or 'i dag'}"
        offer = True
    else:
        note = f"grov tabellkurs {rate} — kan avvika flera procent från dagens; skriv kursen i USD om du har den"
        offer = not (now and not cur["stored_is_local"] and abs(now / usd - 1.0) <= ROUGH_NEAR)
    if not offer:
        st.markdown(f"{head} ≈ {_sv(usd)} USD ({note})</span>", unsafe_allow_html=True)
        return
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"{head} ≈ {_sv(usd)} USD ({note})</span>", unsafe_allow_html=True)
    c2.button("Använd USD", key=f"{widget_key}_usd", on_click=_use_usd, args=(row, usd, widget_key))


def render_fv(row: dict, key: str, prefix: str = "fv", sheet: str = "") -> bool:
    """Lukacs FV-modulen för ett kandidatkort. sheet ("producers", …) gör
    att aktuell kurs kan föreslås ur sifferuppdateringen."""
    changed = False
    data = row.setdefault("fv", {})

    st.markdown(
        f"<b style='color:{TEXT};'>Lukacs FV — fair value per aktie</b> "
        f"<span style='color:{DIM};font-size:0.78rem;'>— CSM säger om bolaget "
        f"överlever scenariot, det här vad det är värt där.</span>",
        unsafe_allow_html=True)

    # ── Kvalitetsklassen, som låser yield-bandet ─────────────────────────────
    # Ingen vald klass visas som '—' (inte som C): grinden kräver ett val, och kortet räknar som grinden.
    codes = list(fv.CLASS_CODES)
    cur_code = str(row.get("fcf_kvalitet") or "").strip().upper()[:1]
    cur_code = cur_code if cur_code in codes else None
    c1, c2, c3 = st.columns([1, 1.2, 1.2])
    code = c1.selectbox(
        "FCF-kvalitet", codes, index=codes.index(cur_code) if cur_code else None,
        key=f"{prefix}_{key}_class", placeholder="—",
        format_func=lambda c: f"{c} — {fv.CLASS_BY_CODE[c].name}",
        help="\n\n".join(_class_label(q) for q in fv.QUALITY_CLASSES))
    if code is not None and code != cur_code:
        row["fcf_kvalitet"] = code
        changed = True

    q = fv.quality_class(code)
    band = fv.yield_band(code)
    locked = q is not None and band is None           # klass D — ingen klass vald är inte låst, bara ofullständig

    shares = c2.number_input(
        "Framtida antal aktier (miljoner)", min_value=0.0, step=1.0,
        value=fv.num(row.get("framtida_antal_aktier")),
        key=f"{prefix}_{key}_shares",
        help="Prognos inklusive återköp och utspädning — återköp är DS:ens "
             "spegelbild.")
    price = c3.number_input(
        "Aktuell kurs", min_value=0.0, step=0.1,
        value=fv.num(row.get("aktuell_kurs")),
        key=f"{prefix}_{key}_price",
        help="I USD — samma valuta som forward FCF (MUSD ÷ miljoner aktier = USD per aktie). Börsdatas kurs i NOK, "
             "SEK … räknas om i förslaget nedanför.")
    ref = _refresh_row(sheet, key)
    cur = fv.price_currency(price, ref, _live_fx(str((ref or {}).get("currency") or "")) if ref else None)
    usd_slot = None
    if cur is not None:     # kursen är i NOK/SEK …: förslaget i USD får sin plats här, men ritas efter scenarierna —
        usd_slot = st.container()   # det läser FCF:en, och den ska vara den här körningens (inte förra)
    elif sheet:                                         # kursen är i USD (eller valutan okänd): det vanliga förslaget
        try:
            import refresh_ui
            refresh_ui.suggest(sheet, row, "aktuell_kurs", "price", "kurs",
                               f"{prefix}_{key}_price", None, with_currency=True,
                               row_id=key)
        except Exception:
            pass
    if (storage.differs(shares, row.get("framtida_antal_aktier"))
            or storage.differs(price, row.get("aktuell_kurs"))):
        row["framtida_antal_aktier"], row["aktuell_kurs"] = shares, price
        changed = True
    if sheet:                                           # efter skrivningen: förslaget jämför med rutans tal nu
        _shares_suggestion(row, key, sheet, ref, shares, f"{prefix}_{key}_shares")

    if q is None:
        st.info(f"{fv.CLASS_MISSING} Steg 5 blir inte grönt utan den.")
    elif locked:
        st.info(f"{fv.NOT_FCF_VALUED}. Target-yield är låst — ett "
                f"extraordinärt kassaflöde ger ingen meningsfull fair value. "
                f"Byt klass om kassaflödet visar sig vara uthålligt.")
    else:
        st.caption(f"Klass {q.code}: target-yield {band[0]:g}–{band[1]:g} %. "
                   f"Bandet är låst till klassen — annars räknar man fram den "
                   f"uppsida man redan bestämt sig för.")

    # ── Scenarierna ──────────────────────────────────────────────────────────
    if sheet:
        bd_fcf_line(row, key, sheet, ref)
    for s in fv.FV_SCENARIOS:
        sc = data.setdefault(s, {})
        # Synliga korta etiketter: på mobilen staplas kolumnerna och rutorna måste säga vilket scenario de gäller
        cols = st.columns([0.6, 1.2, 1, 1, 1], vertical_alignment="bottom")
        cols[0].markdown(f"<div style='padding-bottom:8px;color:{TEXT};"
                         f"font-weight:600;'>{s}</div>", unsafe_allow_html=True)
        fcf = cols[1].number_input(
            f"Forward FCF MUSD ({s})", step=1.0,
            value=fv.num(sc.get("forward_fcf_musd")),
            key=f"{prefix}_{key}_{s}_fcf",
            help="Normaliserat råvarupris, aldrig toppår.")
        ty = cols[2].number_input(
            f"Target-yield % ({s})", min_value=0.0, step=0.5,
            value=fv.num(sc.get("target_yield")),
            key=f"{prefix}_{key}_{s}_yield",
            disabled=locked,
            help=(f"Låst till klass {q.code}: {band[0]:g}–{band[1]:g} %."
                  if band else fv.NOT_FCF_VALUED if locked else "Välj FCF-kvalitet först — klassen låser bandet."))
        vals = {"forward_fcf_musd": fcf, "target_yield": ty}
        if any(storage.differs(v, sc.get(k))
               for k, v in vals.items()):
            sc.update(vals)
            changed = True

        one = fv.fair_value_per_share(fcf, ty, shares) if not locked else None
        up = fv.upside_pct(one, price)
        cols[3].metric(f"FV/aktie ({s})", f"{one:,.2f}" if one is not None else "–")
        cols[4].metric(f"Uppsida ({s})", f"{up:+.0f} %" if up is not None else "–")

        err = fv.yield_error(code, ty)
        if err and err != fv.NOT_FCF_VALUED:
            st.error(err)

    st.caption("Kolumner: scenario · forward FCF (MUSD, normaliserat pris) · "
               "target-yield % · fair value per aktie · uppsida mot kurs.")
    if usd_slot is not None:
        with usd_slot:
            _usd_price(price, cur, ref, row, f"{prefix}_{key}_price")
    for w in fv.fcf_unit_warnings(row, ref):           # rimlighetskontroll — ändrar ingen räkning
        st.warning(w)

    # ── Sannolikheter ────────────────────────────────────────────────────────
    dev = st.checkbox(
        "Avvik från default 20/60/20", value=bool(row.get("prob_deviation")),
        key=f"{prefix}_{key}_probdev",
        help="Defaultvikterna är låsta. Avvikelser kräver skriftlig "
             "motivering — annars är en justerad sannolikhet bara en tumme "
             "på vågen.")
    if dev != bool(row.get("prob_deviation")):
        row["prob_deviation"] = dev
        changed = True

    if dev:
        stored = row.setdefault("probs", dict(fv.DEFAULT_PROBS))
        pcols = st.columns(3)
        newp = {}
        for s, col in zip(fv.FV_SCENARIOS, pcols):
            newp[s] = col.number_input(
                f"{s} %", min_value=0.0, max_value=100.0, step=5.0,
                value=float(fv.num(stored.get(s), fv.DEFAULT_PROBS[s])),
                key=f"{prefix}_{key}_p_{s}")
        if any(stored.get(s) != newp[s] for s in fv.FV_SCENARIOS):
            row["probs"] = newp
            changed = True
        mot = st.text_area(
            "Motivering till avvikelsen", value=row.get("prob_motivation", ""),
            key=f"{prefix}_{key}_probmot", height=68,
            placeholder="Varför är basfallet mindre/mer sannolikt än normalt?")
        if mot != row.get("prob_motivation", ""):
            row["prob_motivation"] = mot
            changed = True
    else:
        st.caption("Sannolikheter: Bear 20 % · Base 60 % · Bull 20 % (låsta).")

    # ── What must go right ───────────────────────────────────────────────────
    wmgr = st.text_area(
        "What must go right — Base", value=row.get("what_must_go_right", ""),
        key=f"{prefix}_{key}_wmgr", height=80,
        placeholder="Råvarupris · produktion · capex · amorteringstakt · återköp",
        help="Obligatoriskt för grönt i köpgrindens steg 5. Bryts ett "
             "antagande är det en modellförlust — inte ett väntläge.")
    if wmgr != row.get("what_must_go_right", ""):
        row["what_must_go_right"] = wmgr
        changed = True

    holding = st.checkbox(
        "Detta är ett innehav (inte bara en kandidat)",
        value=bool(row.get("is_holding")), key=f"{prefix}_{key}_holding",
        help="Slår på trimvarningen när uppsidan mot Base krympt.")
    if holding != bool(row.get("is_holding")):
        row["is_holding"] = holding
        changed = True
    if not holding:
        _holding_hint(row)

    # ── Deleveraging ─────────────────────────────────────────────────────────
    state = fv.deleveraging_state(row.get("nd_ebitda"),
                                  row.get("ar_till_lag_skuld"))
    if state["applies"]:
        st.warning(fv.DELEV_TEXT)
        yrs = st.number_input(
            "År till låg skuld", min_value=0.0, step=0.5,
            value=fv.num(row.get("ar_till_lag_skuld")),
            key=f"{prefix}_{key}_delev",
            help="Nettoskuld ÷ årlig amortering. Kravet är under "
                 f"{fv.DELEV_YEARS_MAX:g} år.")
        if storage.differs(yrs, row.get("ar_till_lag_skuld")):
            row["ar_till_lag_skuld"] = yrs
            changed = True
        if sheet:
            _delev_years_line(row, key, sheet, ref)
        for gap in fv.deleveraging_state(row.get("nd_ebitda"), yrs)["gaps"]:
            st.error(gap)

    # ── Resultatet ───────────────────────────────────────────────────────────
    ev = fv.evaluate(row)
    _summary(ev)
    return changed


def _summary(ev: dict) -> None:
    """Säkerhetsmarginal, expected value och säljregeln."""
    if ev["not_fcf_valued"]:
        return

    color = fv.MOS_COLOR.get(ev["mos_band"], DIM)
    mos = ev["mos"]
    if mos is None:
        st.caption("Säkerhetsmarginalen kräver fair value för Base och en "
                   "kurs.")
    else:
        st.markdown(
            f"<div style='border:1px solid {color}55;background:{color}0d;"
            f"border-radius:8px;padding:10px 14px;margin:8px 0;'>"
            f"<span style='color:{color};font-weight:700;font-size:1.05rem;'>"
            f"Säkerhetsmarginal {mos:.1f} %</span>"
            f"<span style='color:{color};font-weight:700;margin-left:12px;'>"
            f"{ev['mos_band']}</span></div>", unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    e1.metric("Expected value / aktie",
              f"{ev['expected_value']:,.2f}"
              if ev["expected_value"] is not None else "–",
              help="Σ(scenariovärde × sannolikhet).")
    e2.metric("EV-uppsida mot kurs",
              f"{ev['ev_upside']:+.0f} %"
              if ev["ev_upside"] is not None else "–")

    for err in ev["prob_errors"]:
        st.error(err)
    if ev["trim"]:
        st.markdown(
            f"<div style='border:1px solid {fv.ORANGE}88;"
            f"background:{fv.ORANGE}14;border-radius:8px;padding:8px 12px;"
            f"margin:6px 0;color:{fv.ORANGE};font-weight:700;'>"
            f"⚠️ {fv.TRIM_TEXT} — uppsidan mot Base är "
            f"{ev['upside_base']:.0f} %, under {fv.TRIM_UPSIDE_MIN:g} %.</div>",
            unsafe_allow_html=True)
    if mos is not None and not fv.mos_passes_gate(mos):
        st.caption(f"Köpgrindens steg 5 kräver {fv.MOS_BUY_MIN:g} % "
                   f"säkerhetsmarginal.")
    if not ev["what_must_go_right"]:
        st.caption(f"{fv.WMGR_MISSING} — steg 5 kan inte bli grönt utan det.")
