"""
portfolio/position_hints.py — förslag till positionernas fält ur regler och data som redan finns (batch 16).

  stop_hint      Stopp per strategi ur strategins egna regler: Insider (insider.stop_price), Momentum (−10 %,
                 swing_verdict.STOP_PCT), Wolf (entry − 2,5 × ATR14, strategies/wolf.py), EMA200-reglerna (Quality,
                 Alpha: nivån i dag, daterad) och 'ingen kursstopp' för arkstrategierna. En nivå på eller över
                 entry (eller utan entry) blir bara text — journalen räknar R ur stoppen.
  sector_hint    Börsdatas sektor ur sektor-id:t (registret eller Börsdatas instrument), med matchat namn.
  broker_hints   Mäklarens FIFO-snitt och antal (journal_import.fifo.OpenPosition) mot registrets rader, ett konto
                 i taget — valuta, pence i London och fel ISIN-koppling syns i texten. Står tickern på flera
                 rader eller finns den på flera konton blir talen bara text (ett kontos tal är inte radens
                 innehav). Ligger radens stopp på eller över snittkursen blir snittkursen bara text.
  rank_hint      'Ur topp 40' ur wolf_screener.json (rank ≤ 40, swing_verdict.RANK_EXIT).
  ma50_from_screener / ma50_from_closes
                 'Stängt under MA50' ur screenerns dist_ma50, annars ur en kursserie (market_prices).

Rena funktioner: inget Streamlit, inget nätverk. Sidorna ritar förslagen med refresh_ui.offer — Använd skriver,
att öppna sidan skriver ingenting. Ingen regel, tröskel eller formel ändras här; allt läses ur modulerna som äger dem.
"""
from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

# Strategitaggen (positions.STRATEGIES) → stoppregeln i strategins egna regler.
STOP_FIXED = "pct"          # Momentum: −10 % från entry
STOP_ATR = "atr"            # Wolf: entry − 2,5 × ATR14
STOP_INSIDER = "insider"    # Insider: klustersnittet × 0,85
STOP_EMA200 = "ema200"      # Quality och Alpha: under EMA200 halveras positionen — nivån rör sig, förslaget dateras
STOP_NONE = "none"          # ingen kursstopp i reglerna

STOP_RULE: dict = {
    "Momentum": STOP_FIXED, "Wolf": STOP_ATR, "Insider": STOP_INSIDER,
    "Quality": STOP_EMA200, "Alpha": STOP_EMA200,
    "Deep Contrarian": STOP_NONE, "Rule": STOP_NONE, "Royalty": STOP_NONE, "Tiggre": STOP_NONE,
    "Durrett": STOP_NONE, "Sprott": STOP_NONE,
}
EMA200_SPAN = 200           # samma EMA200 som Holdings signalkolumn (holdings._compute_long_signal)
MA50_BARS = 50              # samma MA50 som screenern (wolf_data.metrics_for: rullande medel av 50 stängningar)


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f or f in (float("inf"), float("-inf")) else f


def _pos(v) -> Optional[float]:
    f = _num(v)
    return f if f is not None and f > 0 else None


def _round(v: float) -> float:
    """En kurs som förslag: två decimaler, fyra under 1 (småbolag och pence)."""
    return round(v, 2) if abs(v) >= 1 else round(v, 4)


def fmt_num(v: float) -> str:
    """Talet i texten: decimalkomma, utan onödiga nollor."""
    return f"{v:g}".replace(".", ",")


def _day(index) -> str:
    try:
        return str(index[-1])[:10]
    except Exception:
        return ""


def _playbook_stop(strategy: str) -> str:
    try:
        import positions
        import strategy_rules
        pb = strategy_rules.PLAYBOOKS.get(positions.TAG_PLAYBOOK.get(strategy, ""))
        return pb.risk.stop if pb else ""
    except Exception:
        return ""


# ── 1. Stopp per strategi ────────────────────────────────────────────────────
def last_atr(df) -> tuple:
    """(ATR14, datum) på sista stapeln — Wolfs egen ATR (strategies/wolf._atr, atr_period). (None, '') utan
    High/Low/Close eller med för kort serie."""
    try:
        from strategies import wolf
        period = int(wolf.DEFAULT_PARAMS["atr_period"])
        if df is None or len(df) <= period or not {"High", "Low", "Close"} <= set(df.columns):
            return None, ""
        v = _num(wolf._atr(df, period).iloc[-1])
        return (v, _day(df.index)) if v and v > 0 else (None, "")
    except Exception:
        return None, ""


def ema200_level(df) -> tuple:
    """(EMA200, datum) på sista stapeln ur samma formel som signalkolumnen; (None, '') under 200 staplar."""
    try:
        if df is None or "Close" not in df or len(df) < EMA200_SPAN:
            return None, ""
        close = df["Close"].astype(float).dropna()
        if len(close) < EMA200_SPAN:
            return None, ""
        v = _num(close.ewm(span=EMA200_SPAN).mean().iloc[-1])
        return (v, _day(close.index)) if v and v > 0 else (None, "")
    except Exception:
        return None, ""


def insider_row(ticker, rows: Iterable) -> Optional[dict]:
    """Insiderbevakarens rad för tickern ('BOL.ST' och 'BOL' är samma bolag) som har en stopp — den senast upptäckta
    när bolaget har flera kluster. None utan träff."""
    import insider as ins
    import swing_verdict as sv
    want = sv.normalize_ticker(ticker)
    if not want:
        return None
    hits = [r for r in rows or [] if isinstance(r, dict) and sv.normalize_ticker(r.get("ticker", "")) == want
            and (ins.stop_price(r) or 0) > 0]
    if not hits:
        return None
    return max(hits, key=lambda r: str(r.get("found") or ""))


def _below_entry(hint: dict, entry: Optional[float]) -> dict:
    """Ett stoppförslag får Använd bara när det ligger under entry. Journalen räknar R ur positionens stopp
    (journal_bridge): en stopp på eller över entry ger fel tecken, och utan entry går det inte att se. Då visas
    nivån bara som text, med regeln och dagen."""
    value = hint.get("value")
    if value is None or (entry is not None and value < entry):
        return hint
    level = f"{hint['source']}: {hint['label']} {fmt_num(value)}" + (f" {hint['unit']}" if hint.get("unit") else "")
    why = "entry saknas" if entry is None else f"{'på' if value == entry else 'över'} entry {fmt_num(entry)}"
    return {"value": None, "text": f"{level} — {why}, sparas inte som stopp (journalens R räknas från stoppen)."}


def stop_hint(row: dict, df=None, insider_rows: Optional[list] = None) -> Optional[dict]:
    """Stoppförslaget för positionen ur strategins egen regel, eller None när regeln saknas eller underlaget fattas.

    {'value': kurs eller None, 'label', 'source', 'unit', 'note', 'basis'} — value None betyder bara text ('ingen
    kursstopp', eller en nivå på eller över entry: _below_entry). unit bär datumet för nivåer som rör sig (ATR,
    EMA200): '(2026-11-06)'. basis följer med till positionen vid Använd, så att det syns vilken regel och vilken
    dag stoppen kom ifrån."""
    row = row or {}
    strategy = str(row.get("strategy") or "")
    rule = STOP_RULE.get(strategy)
    if rule is None:
        return None
    if rule == STOP_NONE:
        pb = _playbook_stop(strategy)
        return {"value": None, "text": "Ingen kursstopp att föreslå." + (f" Regeln: {pb}" if pb else "")}
    entry = _pos(row.get("entry_price"))
    if rule == STOP_FIXED:
        import swing_verdict as sv
        if entry is None:
            return None
        pct = float(sv.STOP_PCT)                      # −10.0
        return _below_entry({"value": _round(entry * (1 + pct / 100)), "label": "stopp", "source": "Momentum-regeln",
                             "unit": "", "note": f"entry {fmt_num(entry)} − {fmt_num(abs(pct))} %",
                             "basis": f"−{fmt_num(abs(pct))} % från entry"}, entry)
    if rule == STOP_ATR:
        from strategies import wolf
        atr, day = last_atr(df)
        if entry is None or atr is None:
            return None
        mult = float(wolf.DEFAULT_PARAMS["atr_mult"])
        value = entry - mult * atr
        if value <= 0:
            return None
        return _below_entry({"value": _round(value), "label": "stopp", "source": "Wolf-regeln", "unit": f"({day})",
                             "note": f"entry {fmt_num(entry)} − {fmt_num(mult)} × ATR14 {fmt_num(round(atr, 2))}",
                             "basis": f"{fmt_num(mult)} × ATR14 {day}"}, entry)
    if rule == STOP_EMA200:
        level, day = ema200_level(df)
        if level is None:
            return None
        return _below_entry({"value": _round(level), "label": "nivå i dag", "source": "EMA200-regeln",
                             "unit": f"({day})", "note": "under EMA200 halveras positionen",
                             "basis": f"EMA200 {day}"}, entry)
    if rule == STOP_INSIDER:
        import insider as ins
        sig = insider_row(row.get("ticker"), insider_rows)
        if sig is None:
            return None
        avg = _num(sig.get("cluster_avg"))
        found = str(sig.get("found") or "")[:10]
        note = f"klustersnittet {fmt_num(avg)} × {fmt_num(ins.STOP_FRAC)}" + (f", kluster {found}" if found else "")
        return _below_entry({"value": _round(ins.stop_price(sig)), "label": "stopp", "source": "Insiderbevakaren",
                             "unit": "", "note": note, "basis": f"klustersnittet × {fmt_num(ins.STOP_FRAC)}"}, entry)
    return None


# ── 3. Sektor ────────────────────────────────────────────────────────────────
def sector_hint(sector_id, sectors: dict, bd_name: str = "", own_name: str = "") -> Optional[dict]:
    """{'value': Börsdatas sektornamn, 'note': matchat bolag} eller None utan id eller namn. Noten visar vilket bolag
    Börsdata matchade, med en varning när det inte liknar radens namn (fel ticker ger fel sektor)."""
    try:
        sid = int(sector_id)
    except (TypeError, ValueError):
        return None
    name = str((sectors or {}).get(sid) or "").strip()
    if not name:
        return None
    bd = str(bd_name or "").strip()
    note = f"Börsdata: {bd}" if bd else ""
    own = str(own_name or "").strip()
    if bd and own:
        import refresh_ui
        if not refresh_ui.name_resembles(own, bd):
            note += f" — liknar inte '{own}', kontrollera tickern"
    return {"value": name, "note": note}


# ── 2. Mäklarens tal ─────────────────────────────────────────────────────────
def broker_price(ticker: str, avg, currency: str) -> tuple:
    """(snittkurs i radens enhet eller None, text). Raden räknar i noteringsvalutan (positions.currency_for) och
    London i pence. Mäklarens valuta kan vara kontots och inte handelns (Nordnets fil har alltid SEK) — skiljer den
    sig från noteringen föreslås ingen kurs, bara antalet."""
    import positions
    a = _pos(avg)
    if a is None:
        return None, ""
    c = str(currency or "").strip()
    t = str(ticker or "").strip().upper()
    listing = positions.currency_for(t)
    if t.endswith(".L"):
        if c == "GBp" or c.upper() == "GBX":
            return a, ""
        if c.upper() == "GBP":
            pence = _round(a * 100)
            return pence, f"{fmt_num(a)} GBP = {fmt_num(pence)} pence, som raden räknar London"
        return None, f"snittkurs {fmt_num(a)} {c or '?'} enligt mäklaren — London noteras i pence, inte föreslagen"
    if c.upper() == listing:
        return a, ""
    return None, (f"snittkurs {fmt_num(a)} {c or '?'} enligt mäklaren — {t} noteras i {listing}, inte föreslagen "
                  f"(valutan kan vara kontots, inte handelns)")


def _qty(v) -> Optional[float]:
    q = _pos(v)
    if q is None:
        return None
    return int(q) if abs(q - round(q)) < 1e-6 else round(q, 6)


def _split_note(ticker: str, n: int, qty, avg, currency: str) -> str:
    """Texten när tickern står på flera rader: mäklarens tal gäller summan av raderna, inte en av dem."""
    a = _pos(avg)
    each = f" à {fmt_num(a)} {currency or '?'}" if a is not None else ""
    size = f"{fmt_num(qty)} st" if qty is not None else "antal"
    return f"{ticker} står på {n} rader i Positioner — mäklarens {size}{each} gäller summan, fördela själv"


def _accounts_note(account: str, ticker: str, n: int, qty, avg, currency: str) -> str:
    """Texten när tickern finns på flera konton men på en rad: raden är hela innehavet, kontots tal är en del av det."""
    a = _pos(avg)
    size = f"{fmt_num(qty)} st" if qty is not None else "antal okänt"
    each = f" à {fmt_num(a)} {currency or '?'}" if a is not None else ""
    return f"{account or '?'}: {size}{each} — {ticker} finns på {n} konton, fördela själv"


def _stop_note(price: float, unit: str, stop: float) -> str:
    """Texten när radens sparade stopp ligger på eller över mäklarens snittkurs: entry under stoppen ger journalen
    fel tecken på R (journal_bridge), så snittkursen erbjuds inte förrän stoppen är ändrad."""
    where = "på" if stop == price else "över"
    return (f"snittkurs {fmt_num(price)} {unit} enligt mäklaren — radens stopp {fmt_num(stop)} ligger {where} den; "
            f"ändra eller ta bort stoppen i Holdings först, annars blir journalens R fel")


def broker_hints(rows: Iterable, open_positions: Iterable) -> list:
    """En post per registerrad och mäklarposition med samma ticker: {'row', 'account', 'isin', 'name', 'qty',
    'price', 'price_note', 'warn', 'accounts', 'rows'}. Konton slås aldrig ihop — samma ticker på två konton ger två
    poster, och 'accounts' säger hur många konton tickern finns på. Talen erbjuds bara när tickern står på en rad
    och finns på ett konto: fler rader (samma bolag i två strategier) eller fler konton ger bara texten i price_note
    — ett kontos tal är inte hela radens innehav. Snittkursen erbjuds inte heller när radens sparade stopp ligger på
    eller över den (journalen räknar R ur stoppen); antalet erbjuds då som förut. warn är satt när mäklarens namn
    inte liknar radens (ISIN→ticker-kartan kan ha kopplat fel)."""
    import positions
    import refresh_ui
    rows = [r for r in rows or [] if isinstance(r, dict)]
    ops = [p for p in open_positions or [] if (getattr(p, "ticker", None) or "").strip()]
    accounts: dict = {}
    for p in ops:
        accounts.setdefault(p.ticker.strip().upper(), set()).add(str(getattr(p, "account", "") or ""))
    rows_per: dict = {}
    for r in rows:
        t = str(r.get("ticker") or "").strip().upper()
        rows_per[t] = rows_per.get(t, 0) + 1
    out = []
    for r in rows:
        t = str(r.get("ticker") or "").strip().upper()
        n_acc = len(accounts.get(t, ()))
        stop = _pos(r.get("stop"))
        for p in ops:
            if p.ticker.strip().upper() != t:
                continue
            ccy = str(getattr(p, "currency", "") or "")
            acct = str(getattr(p, "account", "") or "")
            qty = _qty(getattr(p, "qty", None))
            price, note = broker_price(t, getattr(p, "avg_price", None), ccy)
            if rows_per[t] > 1:
                price, note = None, _split_note(t, rows_per[t], qty, getattr(p, "avg_price", None), ccy)
                qty = None
            elif n_acc > 1:
                price, note = None, _accounts_note(acct, t, n_acc, qty, getattr(p, "avg_price", None), ccy)
                qty = None
            elif price is not None and stop is not None and stop >= price:
                unit = "pence" if t.endswith(".L") else positions.currency_for(t)
                price, note = None, _stop_note(price, unit, stop)
            name = str(getattr(p, "name", "") or "").strip()
            own = str(r.get("name") or "").strip()
            warn = ""
            if name and own and not refresh_ui.name_resembles(own, name):
                warn = f"mäklarens '{name}' liknar inte '{own}' — kontrollera ISIN-kopplingen"
            out.append({"row": r, "account": acct,
                        "isin": str(getattr(p, "isin", "") or ""), "name": name,
                        "qty": qty, "price": price, "price_note": note, "warn": warn,
                        "currency": ccy, "accounts": n_acc, "rows": rows_per[t]})
    return out


# ── 4. Swing: kryssen för säljregel 1 och 3 ──────────────────────────────────
def _fresh_screener(screener, today: Optional[date] = None) -> bool:
    """Screenerfilen har en topplista och är högst swing_verdict.STALE_DAYS dagar gammal."""
    import swing_verdict as sv
    if not isinstance(screener, dict) or not screener.get("top"):
        return False
    age = sv.data_age_days(screener, today)
    return age is not None and age <= sv.STALE_DAYS


def rank_hint(ticker, screener, today: Optional[date] = None) -> Optional[dict]:
    """'Ur topp 40' när tickern inte finns bland screenerns 40 (eller har rank över 40). Bara på en färsk fil — en
    gammal lista föreslår ingenting. {'asof', 'note'} eller None."""
    import swing_verdict as sv
    if not _fresh_screener(screener, today):
        return None
    row = sv.screener_row(ticker, screener)
    rank = _num((row or {}).get("rank"))
    if row is not None and (rank is None or rank <= sv.RANK_EXIT):
        return None
    asof = str(screener.get("generated") or "")[:10]
    note = (f"rank {rank:g}" if row is not None else f"finns inte bland de {sv.RANK_EXIT} i listan")
    return {"asof": asof, "note": note}


def ma50_from_screener(ticker, screener, today: Optional[date] = None) -> tuple:
    """(förslag eller None, hittad) ur screenerns dist_ma50. hittad = tickern finns i en färsk lista med dist_ma50
    — då är det svaret, och kursserien behövs inte."""
    import swing_verdict as sv
    if not _fresh_screener(screener, today):
        return None, False
    row = sv.screener_row(ticker, screener)
    dist = _num((row or {}).get("dist_ma50"))
    if dist is None:
        return None, False
    if dist >= 0:
        return None, True
    return {"asof": str(screener.get("generated") or "")[:10],
            "note": f"kurs {dist * 100:+.1f} % mot MA50".replace(".", ",").replace("-", "−")}, True


def ma50_from_closes(close) -> Optional[dict]:
    """'Stängt under MA50' ur en stängningsserie: sista stängningen under medlet av de 50 sista. None med under 50
    stängningar eller när kursen ligger över."""
    try:
        s = close.dropna().astype(float)
        if len(s) < MA50_BARS:
            return None
        ma = _num(s.rolling(MA50_BARS).mean().iloc[-1])
        px = _num(s.iloc[-1])
    except Exception:
        return None
    if ma is None or px is None or px >= ma:
        return None
    return {"asof": _day(s.index), "note": f"stängning {fmt_num(_round(px))} under MA50 {fmt_num(_round(ma))}"}
