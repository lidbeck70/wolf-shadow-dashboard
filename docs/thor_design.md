# 🔨 THOR — designen

*Status: förslag, väntar på godkännande. Inget i den befintliga panelen ändras förrän Thor är klar.*

Målet är en mycket snabbare överblick: **alla strategiers kandidater på ett ställe, ett klick till allt
panelen vet om en aktie, och en ärlig bedömning av vad som verkligen behövs i panelen efteråt.**

---

## 1. Principer

1. **Verdikt överst, bevis under.** Det mönster Viking Nine Regime, Wolf Regime och Alpha Regime redan
   har ("VAD SKA JAG GÖRA? → AVVAKTA → tre av fyra gröna") blir regeln för varje kort i Thor.
2. **Mobilen först.** Panelen används i telefonen: en kolumn, stora verdiktkort, chips, tumvänliga flikar.
3. **Motorerna är sanningen.** Thor räknar inga nya signaler. Den läser strategiernas egna utfall och
   normaliserar bara till en gemensam rangskala (märkt "Thor-rank") för sortering.
4. **Läs körningarna, räkna per aktie.** Kandidatlistan läses ur de schemalagda körningarnas filer
   (omedelbar). Dossiern räknar för EN aktie — det är billigt och går att cacha en timme.
5. **Additivt först, städa sedan.** Thor byggs som en ny toppflik utan att röra de gamla. När den är i
   drift bestämmer du vilka gamla flikar som ska bort (fas 8), inte innan.
6. **Husets stil.** Rena HTML-kortfunktioner (testbara utan Streamlit), `ui.tokens`, `ui.components`,
   BERSERK Regime/Valhall som designreferens, `storage.session_load` för arken (aldrig modulernas `_load()`).

---

## 2. Flikträdet

```
🏠 HOME · ⚡ VALHALL · 🔨 THOR · 🔱 SCREENING · 🔬 GRANSKNING · 📡 REGIME · 👁 INTELLIGENCE · …   (fas 3–7: THOR läggs till, inget annat rörs)

🔨 THOR — EN sida, inga underflikar i navigationsträdet
   1. Läget + tickersök
   2. Rullgardin "Screening": Alla strategier · Viking Nine · BERSERK · Wolf · … → listan
   3. Tryck på en aktie → analysflikarna öppnas AUTOMATISKT på samma sida, med aktien och strategin
      förifyllda: Regim · Teknik · Cykel · Råvara* · Arken · Backtest · Regler
      (* Råvara visas bara för råvarudrivna aktier). Listan fälls ihop under "◀ Kandidater (N)".
```

Flödet är alltså screening → tryck → analys → köp, utan att byta flik. Därför behövs ingen REGIME-flik:
den informationen kommer direkt efter vald screening. Djuplänk: `?p=thor&t=BOL.ST`. Valet av aktie går
via Wolf Asymmetrys kö-mönster (`thor_goto` sätts i knappens callback, läses innan widgetarna ritas).

**Inga regler ändras.** Thor visar varje strategis eget utfall (samma motorer, samma trösklar). Det enda
som är nytt är sammanställningen och en sorteringsskala.

Efter fas 8 (ditt beslut då) blir trädet:

```
🏠 HOME · ⚡ VALHALL · 🔨 THOR · 🔱 SCREENING · 📒 ARKEN · 👁 ODINS VISDOM · 💼 PORTFOLIO · 🔔 ALERTS · 📋 RULES · 🧬 STRATEGIES · 🤖 COPILOT

👁 ODINS VISDOM
├── Råvaror     Odin's Blindspot · Råvarurotation · 🥇🥈 Guld/Silver · 🥇 Guldkvoter · 🌍 EMBER Regime · 🪓 BERSERK Regime
├── Makro       🐺 Fiat Debasement · Sentiment
└── Marknad     Swing Regime · Flow Divergence · Market Cycle · 🌩️ Marknadsrisk · Heatmap · Retail Pulse

📒 ARKEN = dagens GRANSKNING, oförändrat innehåll (Rick Rule · Royalty C · Poängmodell · Tiggre · Insider · Durrett · Wolf Asymmetry · Råvaruhävstång · Scorecard)

📡 REGIME tas bort: per-aktie-sidorna (Wolf/Viking/Viking Nine/Alpha/Long Trend) ersätts av dossiern,
marknadssidorna flyttar till Odins visdom → Marknad. 🕯️ Teknisk analys ersätts av dossierns Teknik-flik.
Alla gamla länkar landar rätt via LEGACY_SLUGS.
```

---

## 3. Screeningen — rullgardinen och listan

### Överst: läget på en skärm
Fyra chips/kort: **Swing-regim** (wolf_regime.json), **Marknadsrisk** SPY/OMXS30 (market_risk_gate),
**Norden** GRÖN/GUL/RÖD (Valhall-kartans hemmamarknad), **N kandidater · M KÖP · körd datum**.
Därunder **tickersök**: ett fält → analysflikarna öppnas för vilken aktie som helst (även sådana som inte
är kandidater — "för andra ändamål").

### Rullgardinen och listan
En rullgardin **"Screening"** med *Alla strategier* överst och sedan varje strategi. *Alla* ger ett flöde
sorterat på Thor-rank; en enskild strategi ger dess egen lista i dess egen ordning och med dess egna
kolumner (Viking Nine: kategori · n/9 · vägd; Long Screener: signal · poäng %; …). 20 i taget ("visa fler").
Filterchips: *bara KÖP*, *bara Norden*, segment. Varje rad är ett kort med knappen "→" som öppnar analysen:

```
BOL.ST  Boliden                                   87
[Viking Nine] [GOLDEN TICKET] [Material GRÖN · MITTEN]
7/9 · 81 vägd · entry 312 · stop 296 · R 2,4 · Material-ETF grön          → Dossier
```

### Källor (allt ur körningarna, inga live-scanningar)

| Strategi | Fil | Thor-rank (0–100) | Status (strategins egen) |
|---|---|---|---|
| Viking Nine | valhall_board.json `swing` | `nine.weighted` | GOLDEN TICKET / READY / DEVELOPING |
| BERSERK | berserk_scan.json | setup S2 100 · S1 80 · S3 60 (BEVAKA −20) | KÖP / BEVAKA |
| Wolf (Arc, EMA/Ichimoku) | arc_screeners.json `wolf` | score ÷ 1,25 | poäng ≥ 50 |
| Viking (z-score) | arc_screeners.json `viking` | V9/9 × 100 | BUY / WATCH |
| Momentum Swing | wolf_screener.json `top` | 100 − (rank−1)·2,5 | rad 1–20 · setup A |
| EMBER | ember_results.json | `setup_score` | KÖPLÄGE / BEVAKA |
| Quality Contrarian | contrarian_alpha_results_quality.json + valhall `buys` | composite (tak 100) | BUY / WATCH · SPÄRRAD (grinden ej RÖD) |
| Deep Contrarian | contrarian_alpha_results_deep_contrarian.json | composite | ACCUMULATE-steg |
| Long Screener | valhall_board.json `long` | `score_pct` × 100 | STRONG BUY / BUY (gron och utan cykel) |
| Insider Norden | insider_scan.json | score × 10 | KÖP / Bevaka / grind |
| Rick Rule · Poängmodell · Tiggre · Durrett | arken (`storage.session_load`) | 0–5 ×20 · 0–10 ×10 · 0–10 ×10 · Durrett quality 0–100 | arkets eget verdikt |
| Håvarna (rule/sprott/durrett/tiggre/royalty) | screens.json | — (träff; "NY" sorteras först) | träff |

Datakontraktet `ThorCandidate`: `ticker (Yahoo-form), name, strategy, family, score, score_text, status,
status_class (KÖP/BEVAKA/SPÄRRAD/HÅLL/SÄLJ), segment, segment_status, cykel, theme, entry, stop, target, rr,
held, new, asof, why`. Det är CLAUDE.md:s kandidatkort (ticker · strategi · regim · sektorpoäng · setup-poäng ·
regler · entry · stop · mål · rekommendation) med "passade/fallerade regler" i dossiern.

Tickerformer normaliseras till Yahoo-form (Börsdata "ERIC B" → "ERIC-B.ST", arkens handskrivna "EQX" →
provar .ST/.OL/.CO/.HE). Ett bolag med flera aktieslag räknas en gång per strategi (`valhall.picks.company_key`).

---

## 4. Analysflikarna — öppnas automatiskt när du tryckt på en aktie

Allt panelen vet om aktien, på samma sida som listan (listan fälls ihop ovanför). Det är här analysen görs
innan köp. "Dossiern" nedan är bara ett namn på det här blocket.

### Huvudet (alltid)
- Ticker, namn, chips: marknad, segment med status (GRÖN/GUL/RÖD · cykel), råvarutema om råvarudriven,
  **INNEHAV** om aktien finns i registret, "finns i: Viking Nine · Long Screener" (dagens kandidatlistor).
- Ögonblicksbild (`market_data.snapshot`): kurs, 1 m/3 m, ATR %, avstånd EMA50/200, 52 v.
- **Strategiväljare** "Bedöm som": förvald från kandidatraden, annars gissad (nordisk likvid → Viking Nine,
  råvarudriven → BERSERK, i arken → arkets strategi).
- **VERDIKTKORTET** — strategins eget utfall i ett ord + en mening + "n av m":
  Viking Nine `GOLDEN TICKET/WAIT/NO TRADE` · Wolf `BULL/…` + entry ja/nej · BERSERK `KÖP/BEVAKA` ·
  Quality `BUY/WATCH/WAIT` · Contrarian `ACCUMULATE/HOLD/DISTRIBUTE` · Long Screener `STRONG BUY…` ·
  Momentum `KÖP-KANDIDAT/BEVAKA/HÅLL/SÄLJ` · arken `Köpkandidat/Bevaka/Passa`.
- Knappar: "Öppna i Copilot" (seedar ticker + strategi), "Öppna i arket" när bolaget finns där.

### Sidoflikarna

| Flik | Innehåll | Återanvänder |
|---|---|---|
| **Regim** | Swing-familjen: Viking Nine-layouten — marknadsbannern SPY/OMXS30 n/3, Nine-kortet Market 40/Sector 30/Stock 30 med ✓/✗, entry-beslutet (entry/stop/R/andelar), exit-kortet om innehav. BERSERK: temats drivarkort + aktiens setup. Momentum: Swing-domen (rank, ljus, setup). **Wolf**: det stora poängkortet n/125 + BULL/… + fyra mätare + 11 grindar + SL/TP. **Lång-familjen**: Alpha-layouten "VAD SKA JAG GÖRA?" + signalkorten TREND · DISCOUNT · CYCLE · QUALITY (Quality), ACCUMULATE-steg + gummibandskvoter (Contrarian), Fundamenta 0–20 · Cykel 0–3 · Teknik 0–7 + grindar (Long Screener). | `ovtlyr_nine.evaluate`, `viking_execution.evaluate_entry`, `viking_exit.evaluate_exit`, `nine_card`, `execution_card`, `exit_card`; `berserk.live.evaluate` + `regime_ui.theme_card`; `swing_verdict.verdict`; Wolf: poängen bryts ut ur `tabs/regime.py` till en ren modul (samma siffror, regressionstest), `strategies/wolf.entry_fn` för grindar/nivåer, `ui.charts.build_gauge`; `alpha_regime.engine.run_regime_analysis` + `_render_quality_mode/_render_contrarian_mode/_render_action_box`; `valhall/long_screener.score` |
| **Teknik** | Kurs med EMA och order blocks, OB-tabell, mönster, risk & momentum, avancerat — samma vy som 🕯️ Teknisk analys, för den valda aktien. | `tech_analysis.load_ohlcv` + `_render(ticker, df, src)` (redan parametriserad). Buggen vol ratio/ATR (läser fel nycklar) rättas. |
| **Cykel** | Aktiens egen psykologifas (14 faser DISBELIEF → DISBELIEF_NEW) med mätare, tidslinje och regelutfall; hemmamarknadens fas; Valhall-segmentets status/fas/cykel (TIDIG…TOPP, ⚡ cykelstart); strategins köp/håll/sälj-faser ur playbooken → PASS/MANUAL/FAIL. | `cycle.market_phase`, `market_cycle.cache`, `tabs/market_cycle._build_gauge/_build_timeline/_render_rules_breakdown`, `valhall.ui.load` + `paper.cycle_tag`, `cycle.playbook_phases/gate_from_market_phase` |
| **Råvara** *(bara råvarudrivna)* | **Råvaran** (och varifrån uppslaget kom): BERSERK-temats drivare och fas, Valhall-temats status/cykel, Blindspot-temats cykel/hat/nödvändighet/percentil, rotationsbetyget (Triple Signal) + bokens kapitel. **Hävstång**: resultathävstång (score 0–10, break-even, +20 %) och kursbeta upp/ned med graf. **Debasement**: fiat-miljön, råvaran i reala dollar och mätt i guld. **Kvoter** för temat: guld/silver, metall/guld, gummibanden. | `thor.ticker.theme` (ett uppslag: arket → `TICKER_THEME_MAP` → BERSERK `PRODUCERS` → Börsdata-bransch → Yahoo), `berserk.live.theme_states` + `theme_card`, `valhall.ui.theme_card`, `blindspot.theme_board.build_theme_board`, `cycle.cycle_state` + `commodity_book.chapter`, `commodity_leverage.engine.analyze` + `beta_chart`, `fiat_debasement.ui.real_commodity/fiat_vs_lines` + `environment.render_badge`, `gold_silver`/`gold_ratios`/`alpha_regime.commodity_ratios` |
| **Arken** | Vad arken säger om bolaget, bara läsning: Rick Rule/Royalty/Poängmodell/Tiggre/Insider → PASS/MANUAL/FAIL + ifyllda fält; Durrett-motorns 12 poäng och Confidence; Wolf Asymmetry (hävstång, MoS, break-even); Scorecardets sju kryss; insiderkluster; position i registret; journalens affärer i aktien. "Finns inte i arket → lägg in" med knapp till arket. | `review_link.review/detail_lines`, `confidence.store.get` + `engines.durrett.engine.analyze` + `asymmetry.analyze`, `scorecard.gate_state`, `insider_scan`-filen, `positions.find`, `trade_journal.load_journal` — allt via `storage.session_load` |
| **Backtest** | Strategin på aktien, lättvindigt: knapp "Kör", period 3/5/10 år → nyckeltal (affärer, träff, förväntan R, PF, max DD) + affärstabell + kursgraf med entry/exit-markörer och stoppen. Märkt med fyllnadsregeln ("nästa öppning, stop intradag"). Strategier utan per-aktie-motor (Long Screener, Contrarian Alpha, EMBER) visar Valhalls aggregerade beslut och säger det rakt ut. | `viking_backtest.backtest_ticker` (snabbläge: `risk_gate=()`, injicerad nordisk marknad), `berserk.backtest.backtest_ticker`, `swing_backtest.run` på en-kolumnspanel (märkt med förbehållen), Wolf: tunt omslag runt `wolf_shadow_backtest.Backtester` med kopierad config (INTE sidan — den skriver i delad config och skickar Discord-larm), `ovtlyr/ui/charts.build_price_chart` + markörer, `valhall.longterm.stock_rows` för Quality (grindträffar med framåtavkastning, fas 2) |
| **Regler** | Sist: strategins playbook — risk, entry/exit/mindset-regler, "så gör du steg för steg", fusklapp, fallgropar. | `strategy_rules.PLAYBOOKS`, `rules_page._rule_row/_risk_box/render_inline_rules`. Viking Nine och BERSERK saknar playbook → de skrivs (fas 7). |

Prestanda: bara den öppna sidofliken renderas (`st.tabs(on_change="rerun")` + `tab.open`, samma grepp som
toppflikarna). Allt per aktie cachas i sessionen en timme (`thor_cache[ticker]`), priser via `market_prices` (6 h).

---

## 5. Wolf och Viking Nine i Valhall

- **Viking Nine finns redan**: på Kartan (Swing-raden i Teknik och Finans, "prövas" i övriga segment) och i
  papperskontot `viking`.
- **"Wolf" i Valhall i dag är momentumlistan** (Swing Screener, wolf_screener.json) — bara i papperskontot
  `wolf`, inte på Kartan. **Arc-Wolf** (EMA/Ichimoku, sidan du gillar) körs nattligt för Norden
  (arc_screeners.json, poäng ≥ 50) men finns inte alls i Valhall.
- Förslag (PR 0, oberoende av Thor): Kartan får raden **"Wolf"** per sektorruta (Arc-Wolfs bästa i segmentet)
  och raden **"Momentum"** (topp 20 med setup A); papperskontot får `wolf_arc` (entry/exit ur
  `strategies/wolf.py`: EMA-stack, 2,5 × ATR, 2,6R/5,2R) och det gamla `wolf` döps om till "Momentum Swing"
  så att namnen slutar ljuga. Thor-listan visar alla tre.

---

## 6. Filer

```
thor/
  __init__.py
  registry.py      strategiregistret: nyckel → etikett, familj, färg, källa, rank-normalisering, playbook, dossier-familj
  candidates.py    bygger ThorCandidate-listan ur filerna + arken (ren logik, testbar)
  ticker.py        identiteten: Yahoo-form, Börsdata insId/namn/sektor/bransch, segment, hemmaindex, sektor-ETF, råvarutema med källa
  dossier.py       per aktie: ögonblicksbild, regim per familj, cykel, råvara, arken, position/journal (ren logik)
  cards.py         rena HTML-kort: verdikt, bevis, chips, kandidatrad (testbara utan Streamlit)
  backtest.py      tunna omslag → gemensam resultatform {trades, metrics, fill}
  ui.py            render_thor_page(): läget, sök, rullgardinen, listan och analysflikarna på samma sida
wolf_regime_score.py   Wolfs regimpoäng utbruten ur tabs/regime.py (samma siffror — regressionstest)
tests/test_thor_candidates.py · test_thor_ticker.py · test_thor_dossier.py · test_thor_cards.py · test_thor_backtest.py · test_thor_ui.py
```

Ändringar i befintliga filer (bara i fas 3 och 8): `ui/nav.py` (TOP, SUBS, STATE_KEY, HOME_ZONES),
`wolf_panel.py` (ett guardat block), `tabs/home.py` (zonfärg), `ovtlyr/ui/rules_page.py` (FLIKGUIDE-rader),
`tests/test_strategy_rules.py` (tillåtet huvud THOR). Fas 4: `tabs/regime.py` anropar den utbrutna poängen.

---

## 7. Faser (en PR per fas, egen branch, svensk PR-text)

| Fas | PR | Innehåll | Rör befintlig panel? |
|---|---|---|---|
| 0 | Valhall: Wolf på Kartan | Arc-Wolf + Momentum per segment, `wolf_arc`-konto, omdöpning | Ja — Valhall (du bad om det) |
| 1 | Thor-kontraktet | `thor/registry.py`, `thor/candidates.py`, tester med fixturfiler | Nej |
| 2 | Identitet + dossierdata | `thor/ticker.py`, `thor/dossier.py`, tester | Nej |
| 3 | 🔨 THOR: läget, sök, rullgardinen, listan | nav (en toppflik, inga underflikar), panelblock, `thor/ui.py`, FLIKGUIDE, tester | Bara tillägg |
| 4 | Analysflikarna: huvud · Regim · Teknik · Regler | tryck → flikarna öppnas automatiskt; verdiktkortet, tre regimlayouter, TA, playbooks; Wolf-poängen bryts ut | `tabs/regime.py` anropar ny modul (samma siffror) |
| 5 | Analysflikarna: Cykel · Råvara | 14 faser, Valhall, råvaruuppslaget, hävstång, debasement, kvoter | Nej |
| 6 | Analysflikarna: Arken · Backtest | arken (läsning), position/journal, backtestomslagen, graf med markörer | Nej |
| 7 | Playbooks Viking Nine + BERSERK | `strategy_rules`, tester | Bara tillägg |
| 8 | Städningen — efter ditt ja | ODINS VISDOM, ARKEN, REGIME bort, Teknisk analys bort, LEGACY_SLUGS, FLIKGUIDE, Home, tester | Ja — det är poängen |

Varje fas: pytest grönt, PR, "Merga NN", gren = main.

---

## 8. Kända förbehåll

- **Råvaruuppslaget** har i dag sju vokabulärer och motsägande tabeller (EMBER mot BERSERK på tolv
  tickers). Thor inför ETT uppslag med tydlig källa men ändrar inte de gamla tabellerna.
- **Thor-rank är en presentationsskala**, inte en signal: 87 i Viking Nine och 87 i Long Screener betyder
  olika saker. Därför visas alltid strategins egen siffra bredvid.
- **Backtest per aktie** finns bara för Viking Nine, BERSERK, Momentum och Wolf. Resten sägs rakt ut.
- **EMBER-filen** ligger i en egen Gist (`EMBER_GIST_ID`) — hemligheten måste finnas i Streamlit Cloud för
  att EMBER ska synas i flödet.
- **Två Wolf-poängar** finns i koden (screenern 1,5 × ATR/2R/3R mot regimsidan 2,5 × ATR/2,6R/5,2R).
  Dossiern använder regimsidans, som du gillar, och säger vilken.
- Inget live-autotrading, inga nya beräkningar i AI-lagret i v1. AI-kommentar på verdiktkortet är ett
  naturligt nästa steg (Copilotens `_ai_comment` finns redan).
