# LIDSKJALV — designspecen för Wolfpanelen

> Ägarens beslut 2026-11-21: **kör på rekommendationerna** (beslut 1–11, 13, 14, 16 = JA; 15 = senare; 17 och 18 = nej nu).
> **Beslut 12: `ovtlyr/ui/rules_page.py` är INTE fryst** — Regler & Guider får byggas om (hårda regler först, Kom igång-ordningen,
> "om 1 dag", antalet strategier ur `LEARNING_ORDER`), men varje regeltext ordagrant. Viking-filerna är fortfarande frysta.
> Skisserna: https://claude.ai/artifact/FT8fDpSPDKhmvVFVQ7Skny (privat). Specen nedan är facit för PR 13–24.

# LIDSKJALV: beslutsbryggan

**Final design spec for the Wolf panel (Nordic Arc Systems)**
Streamlit 1.59.1 · iPhone 390 px first, desktop second · same theme · no rule changes

**Where it comes from.** The base is LIDSKJALV. All three judges scored it highest, and it is the only concept checked in a 1.59.1 prototype at 390 px. Ideas taken from the other two concepts:
- **From ARC NATIVE:**
  - scroll-to-top when the route changes, and when the active tab is tapped;
  - a scan beam while Streamlit runs;
  - an amber "osparat" snackbar;
  - grey, dimmed emoji on inactive tabs;
  - scoped amber (dirty save) and red (confirm delete) exceptions to the primary colour;
  - its test guardrails.
- **From BIFROST:**
  - ink tints for small coloured text;
  - the `[theme]` palette mapping;
  - a Safari fallback for `content`;
  - a no-JS "▲ Till toppen" link;
  - the rule "the source word is a grey chip";
  - the rule "sign colour only for changes";
  - inner tabs that wrap.

**Checked against the installed environment:**
- Streamlit is 1.59.1, and every `theme.*` and `client.toolbarMode` key used below exists in it.
- `nav.TOP`, `nav.SUBS`, `nav.STATE_KEY` and `nav.TARGETS` are as quoted.
- `laget.capital_row` returns `(rubrik, ord, under, färg)`. Norden's colour comes from the majority word, so "0/9 GRÖN" is red. This is display logic and stays.
- `vocab.main(..., HOLDING)` maps WATCH → VAKSAM.

---

## 1. Namn och vision

**LIDSKJALV** är Odins högsäte, platsen där han ser alla världar. Därför passar namnet en panel med triangelögat och raden "See What the Market Can't".

Den nya Wolfpanelen blir ett högsäte i fickan. Varje sida öppnar med det beslut den finns till för, i en **Beslutsremsa**. Remsan visar motorernas egna ord i deras egna färger, sida vid sida och aldrig ihopslagna. Bevisen kommer under den, och verktygen (fält, tabeller och diagram) ett tryck längre ner.

På iPhonen krymper ramen från 464 px till 100 px. Alla sex flikar syns och följer med när du skrollar. Undernivåerna blir en rad chip som du sveper med tummen, och den valda står alltid i mitten, även när du kommer från en Discord-länk. Varje kort, rad och ljus är själva knappen.

**Morgonrundan** tar dig I DAG → Läget → Thor → 🎯 Scorecard → Positioner på fyra tryck, utan att du behöver leta.

Det är samma mörka yta, samma Arc-cyan, guld, glöd och triangelöga. Inte en enda regel, tröskel, poäng, grind, fliknamn eller länk ändras: designen ordnar, färgar och förpackar bara det som motorerna redan räknar fram.

**Vad du märker första morgonen**
- I DAG visar *Dagens besked* och de fyra kapitalljusen redan på första skärmen. Bannern på 260 px är borta från sidorna och finns kvar på inloggningen.
- Alla sex flikar 🏠 👁 🔨 📒 💼 ⋯ syns på 390 px och sitter kvar överst när du skrollar. Om du trycker på den aktiva fliken kommer du till toppen.
- En tunn cyan stråle sveper under flikraden medan panelen räknar.
- Thor-raden, kvalaren och ljusen är tryckbara i hela sin yta.
- Röd färg betyder bara SÄLJ eller fel. Streamlit-rött försvinner när du godkänner cyan som primärfärg (beslut 4).
- Om du har osparade ändringar visas en bärnstensfärgad påminnelse ovanför tummen.
- Positionerna har ett kort per innehav, och "Behöver åtgärd i dag" står överst.

---

## 2. Design principles

1. **BESLUT → BEVIS → VERKTYG.** The first block under the nav on every page is the Beslutsremsa.
   - It is built only from values the page already computes.
   - When values are known late, it uses the existing `review/card.slot()`/`fill()` placeholder pattern (an `st.empty()` at the top, filled at the end of the run).
   - Evidence comes next. Inputs, tables and charts sit in expanders or inner tabs, and only the open one renders.
2. **VISA, SLÅ ALDRIG IHOP.** A strip shows each source's own word in its own colour, or counts of existing outputs.
   - It never shows a new combined verdict.
   - Counts are labelled as counts ("ANTAL I DAG").
   - The four Läget lights stay four separate lights everywhere, including the brand-bar puls.
   - The source word (GOLDEN TICKET, EXIT, PASS) always sits as a **grey chip** beside the vocab main word and is never recoloured.
3. **RAMEN ÄR LITEN.** Chrome budget on a 390 × 844 iPhone (the fold is the 664 px Safari viewport):

   | Depth | Example | Chrome above content |
   |---|---|---|
   | 1 | I DAG, THOR | brand bar 44 + tabs 56 = **100 px** |
   | 2 | ODIN › Läget, ARKEN › Scorecard | + L2 chip rail 52 = **152 px** |
   | 3 | ODIN › Råvaror › 🥇 Guld, MER › Regler & Guider › … | + crumb 20 + L3 rail 48 = **220 px** |
   | After scrolling | any | sticky tab bar only, **56 px** |

   Today the chrome is about 464 px at depth 1 and about 1,200 px at depth 3.
4. **EN TUMME.** Every tap target is at least 44 × 44 px.
   - The whole card or row is the button: an invisible overlay with the same callback, key and label.
   - The next step of the routine sits at the top right and as a full-width button at the bottom.
   - Destructive actions sit behind `confirm_delete`.
   - Tapping the active tab or "▲ Till toppen" scrolls to the top. A new page always opens at its top.
5. **EN FÄRG = EN BETYDELSE.**
   - Cyan means interaction and selection only.
   - Gold means page titles and ARKEN.
   - GREEN/AMBER/RED mean verdicts and signs of change only. Price levels are never sign-coloured.
   - EMBER means vändning and råvaror. PURPLE means regime and ODIN accents.
   - Coloured text under 19 px uses the colour's **ink** tone (§3.1).
6. **TYPOGRAFI MED JOBB.**
   - Space Grotesk for nav, titles, verdict words, chips and buttons.
   - JetBrains Mono for every number, date and time.
   - Streamlit's Source Sans stays for long Swedish rule text.
   - The floor is 12 px. Secondary text is MUTED.
7. **SAKNAS ÄR GRÅTT OCH ÄRLIGT.** Missing data has one look: a dashed grey "Saknas" chip with the reason in its tooltip. It is never a silent 0, 50, a red 0.0 or "None". The F&G neutral-50 fallback is labelled, not changed.
8. **INGET TYST.** Loading shows the scan beam or a skeleton. A save, "Logga köp", "Bevaka" or "Registrera beslut" ends with a toast. Unsaved edits show the snackbar. A disabled button looks disabled.
9. **SAMMA REGLER, NYCKLAR OCH LÄNKAR.** Every PR is a reorder, restyle or wrapper. In addition:
   - No `st.markdown` that a test reads becomes `st.html`.
   - No inline SVG goes into `st.html`, because DOMPurify strips it.
   - No script contains the character `<`.
   - No widget changes type.

---

## 3. Design system

### 3.1 Tokens (`ui/tokens.py`): all kept, additions only

| Token | Value | Contrast on #05070A / on BG_ALT #1a1f25 | Use |
|---|---|---|---|
| APP_BG *(new name for the existing .stApp value)* | `#05070A` | – | page background, sticky tab bar |
| BG | `#0c0c12` | – | brand bar, dock-like surfaces |
| BG_CARD | `#14141e` | – | cards, chips |
| BG_ALT | `#1a1f25` | – | tiles, strips, fields |
| BORDER / BORDER_FAINT | `#2a2a38` / `rgba(255,255,255,.06)` | – | dividers |
| TEXT | `#e8e4dc` | 15.9 / 13.1 | body and values |
| NOTE *(moved from ui/components, re-exported there)* | `#c4bfb3` | 11.0 / 9.0 | subtitles, notes |
| **MUTED (new)** | `#a8a397` | 8.0 / 6.6 | labels, meta, secondary text |
| DIM | `#8a8578` | 5.5 / 4.5 | only text ≥14 px, and disabled states |
| ACCENT/CYAN | `#00E5FF` | 13.1 / 10.8 | interaction and selection |
| GOLD | `#c9a84c` | 8.8 / 7.3 | page titles, ARKEN |
| EMBER | `#FF6B3D` | 7.1 / 5.9 | vändning, råvaror |
| PURPLE | `#B400FF` | 4.2 / 3.5 | regime and ODIN accents (rules, dots) |
| GREEN | `#2d8a4e` | 4.7 / 3.8 | KÖP, positive, GRÖN |
| AMBER | `#d4943a` | 7.8 / 6.4 | BEVAKA, VAKSAM, GUL |
| RED | `#c44545` | 4.1 / 3.4 | SÄLJ, negative, error, RÖD |
| GREY | `#6b7280` | – | Saknas, off, NEJ borders |
| **GREEN_INK (new, owner decision 5)** | `#45b36b` | 7.6 / 6.3 | GREEN text under 19 px |
| **RED_INK (new, decision 5)** | `#e36464` | 6.0 / 5.0 | RED text under 19 px |
| **PURPLE_INK (new, decision 5)** | `#c35cff` | 6.1 / 5.0 | PURPLE text under 19 px |

**New helpers in `ui/tokens.py`:**
- `ink(color)` returns the ink tone for GREEN, RED and PURPLE, and every other colour unchanged. Until PR 16 it returns the base colour, so approving decision 5 is a one-line switch.
- `tint(color, a)` replaces the ad-hoc `f"{col}22"` strings.
- Scale constants: `FS_*`, `SP_*`, `R_*`, `TAP = 44`, `FIELD_H = 48`.

### 3.2 Colour roles (what each colour may and may not mean)

| Colour | May mean | Never |
|---|---|---|
| Cyan | selected tab or chip, focus ring, primary button (after decision 4), links, chevrons, scan beam | a verdict, a section label, a positive number |
| Gold | page titles, ARKEN accents, the Mot målet rule | status |
| GREEN / AMBER / RED | verdict words, light dots, signs of P/L and changes | price levels, decoration, Streamlit widgets |
| EMBER | ⚡ VÄNDNING, råvaror, EMBER | warnings |
| PURPLE | regime and ODIN rules, the aurora gradient end | text under 19 px without ink |
| GREY | Saknas, NEJ, off | "neutral reading" |

### 3.3 Type scale (px)

| Role | Font | Size / line height | Weight | Tracking, case |
|---|---|---|---|---|
| Page title (`page_header`) | Space Grotesk | `clamp(20px, 5.6vw, 26px)` / 1.15 | 700 | .06em, UPPER, `word-break:keep-all; hyphens:manual; text-wrap:balance` |
| Verdict hero word (Thor) | Space Grotesk | 32 / 1.05 | 900 | .02em |
| Beslutsremsa words | Space Grotesk | 24 / 1.15 | 800 | .02em |
| Tile value, number | JetBrains Mono | 20 / 1.2 | 600 | `tabular-nums` |
| Tile value, word | Space Grotesk | 20 (16 if over 14 characters; 15 per line when stacked) | 700 | – |
| Card title, ticker | Space Grotesk | 16 / 1.3 (17 for a holding ticker) | 700 | – |
| Body, long rule text | Source Sans (Streamlit) | 15 / 1.55 | 400 | – |
| Subtitle, note | Space Grotesk | 13 / 1.45, NOTE | 400 | max-width 60ch |
| Nav chip L2 | Space Grotesk | 14 / 1 | 600 | – |
| Nav item L3 | Space Grotesk | 14 / 1 | 500, 700 when selected | – |
| Button | Space Grotesk | 14 / 1 (13 when compact) | 600 | .06em UPPER |
| Tab word (phone) | Space Grotesk | 12 / 14 | 600 | .01em UPPER |
| Tab emoji (phone) | system emoji | 20 / 22 | – | – |
| Kicker, section label | Space Grotesk | 12 / 1.2, MUTED | 600 | .12em UPPER |
| Chip | Space Grotesk | 12 / 1, height 22 | 700 | .06em UPPER |
| Meta, time, small number | JetBrains Mono | 12–13 / 1.3, MUTED | 500 | 0 |
| Inputs | JetBrains Mono for tickers and numbers, Space Grotesk for text | **16** (prevents iOS zoom), field 48 (login 52) | 500 | – |

The floor is **12 px** everywhere. The dead Inter Tight `@import` goes. One Google Fonts import remains: Space Grotesk 400/600/700/800/900 and JetBrains Mono 400/500/600.

### 3.4 Spacing, radius, borders, motion
- **Spacing** steps are 4, 8, 12, 16, 20, 24, 32.
  - Side gutter: 16 px on phone, 24 px at 641–1023 px, 32 px at 1024 px and up.
  - Content max-width is 1280 px, centred.
  - Gaps: 8 px between list cards, 12 px between blocks.
  - Section labels: 20 px above, 8 px below.
- **Radius:** chip 6, pill 999, tile 10, field 10, button 10, card and strip 12, verdict hero 14, popover and dialog 16.
- **Borders and elevation:** cards use a 1 px `rgba(255,255,255,.08)` border and no drop shadow. Glow is reserved for the eye mark, ARC, the logo and focus.
- **Motion:**
  - Press: 120 ms `scale(.98)` on buttons, `.99` on tap cards.
  - Scan beam: 1.2 s linear sweep. Skeleton shimmer: 1.2 s.
  - `prefers-reduced-motion: reduce` turns all animation off: the beam becomes a static 2 px line and skeletons are static.

### 3.5 Components

#### 3.5.1 Shell (`ui/shell.py`, new)
- **`shell_css()`** returns ONE CSS string, generated from `nav.TOP` and `nav.STATE_KEY`. It holds:
  - the tab labels and sticky tablist;
  - the L2/L3 rails and inner tabs;
  - tap cards, head rows and `kpis_*` metrics;
  - buttons, frame padding, header handling and the beam.
- **`render_shell()`**:
  - Writes `st.container(key="wolf_shell")` holding ONE `st.markdown("<style>" + CYBERPUNK_CSS + shell_css() + "</style>" + PWA metas)` and the `st.html(nav_js(route), unsafe_allow_javascript=True)` script.
  - `.st-key-wolf_shell{display:none}` hides it. Style tags still apply under `display:none`, and the prototype measured the brand bar at y=4 instead of about 55.
- **`inject_css()`** keeps its name and delegates to the shell, so callers are unchanged.
- **Shell CSS is appended after CYBERPUNK_CSS** in the same `<style>`. Equal-specificity rules therefore win by order, and `tests/test_css_tabs.py`, which parses only CYBERPUNK_CSS, is untouched by construction.
- **`page_style(css, key)`** is a helper for page-scoped CSS: it puts one `<style>` in a hidden `st.container(key=f"pcss_{key}")`, so there are no 16 px gaps. The heatmap module uses it first.
- **Frame rules:**
  - `≤640px`: `[data-testid="stHeader"]{display:none}`, `[data-testid="stMainBlockContainer"]{padding:.25rem 1rem 5rem}`.
  - `≥641px`: `stHeader{background:transparent;height:0;min-height:0;pointer-events:none}`, `[data-testid="stToolbar"]{position:fixed;top:8px;right:12px;pointer-events:auto}`.
    - The ⋮ menu floats over the brand bar's right 48 px, which is reserved as padding-right on desktop.
    - On phone it is hidden, because it would cover MER on the sticky bar.
  - The top-level vertical block has `gap:0` (selector verified in PR 13; acceptance: brand bar to tablist ≤ 4 px).

#### 3.5.2 Brand bar (`ui/theme.render_header()`, replaces the banner)
- **Mechanics:**
  - `render_header()` keeps its call site and now *returns* an `st.empty()` slot, filled at once with the bar as known at run start.
  - `theme.fill_bar(slot)` refills it at the end of `main()`, after `storage_ui.footer()`, so the dirty count and puls are current.
  - There are two writes and no collapse.
  - It is rendered with **st.markdown**, because inline SVG renders there and not in `st.html`.
- **Size and surface:**
  - 44 px tall on phone, 52 px at ≥1024 px.
  - Background BG `#0c0c12`, bleeding to the screen edges (`margin:0 -16px; padding:0 16px`).
  - A 2 px aurora line on top: `::before`, `linear-gradient(90deg,#00E5FF,#B400FF)`.
  - A 1 px `rgba(0,229,255,.15)` hairline below.
  - `id="nas-top"` (the target of the no-JS "▲ Till toppen").
- **Left:**
  - The existing `_SVG_ICON` at 24 px, with `drop-shadow(0 0 6px rgba(0,229,255,.5))`.
  - The wordmark `NORDIC <b>ARC</b>`: Space Grotesk 700, 13 px, .22em, TEXT, with ARC in cyan and `text-shadow:0 0 12px rgba(0,229,255,.6)`.
  - At ≥1024 px, " SYSTEMS" plus the tagline "See What the Market Can't." in 12 px italic NOTE.
- **Right, in order:**
  1. Date `lör 21 nov`, JetBrains Mono 12 px MUTED. It is hidden on ≤400 px while the dirty chip shows.
  2. **Dirty chip** `● 2 osparade`, 12 px AMBER on `tint(AMBER,.14)`, radius 999, shown when `storage.dirty_stores()` is non-empty.
  3. 🔒/🔓 at 14 px from `private_lock.unlocked()`, shown only when the lock is configured.
  4. **Puls**: four 8 px dots with 4 px gaps, in the order Allokering · Swing · Marknadsrisk · Norden.
     - Each dot is coloured by its own `capital_row` colour.
     - `role="img"`, and `aria-label`/`title` list the four words separately ("Allokering NORMAL · Swing GUL · Marknadsrisk SPY LÅG · OMXS30 FÖRHÖJD · Norden 4/11 GRÖN").
     - It is built by `laget.capital_row(...)` ONLY from dicts already in `st.session_state` (the I DAG / Läget data keys read with `.get`).
     - It never calls a loader. With no data in the session it renders nothing.
     - There is no tooltip on touch, so on the phone it is a glance signal whose legend is the KAPITALET grid (same order, same colours).
- **Phone width check:** eye 24 + wordmark about 120 on the left; date 75 + puls 44 + lock 16 + gaps on the right. That is about 300 of 358 px.

#### 3.5.3 Tab bar, L1 (`st.tabs(tab_labels, on_change="rerun", key="main_tabs")`; the call line is unchanged)
- **Sticky at all widths.** The rule targets the tablist itself, not its parent:

```css
.st-key-main_tabs > div > [role="tablist"]{position:sticky;top:0;z-index:50;background:#05070A;
  box-shadow:0 1px 0 rgba(0,229,255,.15)}
```

- **Phone (≤640 px), six stacked cells, all visible.** The labels are *generated* from `nav.TOP`:

```python
for i, (_k, label) in enumerate(nav.TOP, 1):
    emo, word = label.split(" ", 1)
    t = f'.st-key-main_tabs [role="tablist"] > [role="tab"]:nth-child({i}) p'
    css += (f'{t}::before{{content:"{emo}";content:"{emo}" / "";}}'      # first line = Safari <17.4 fallback
            f'{t}::after{{content:"{word}";content:"{word}" / "";}}')
```

  - Each cell: `flex:1 1 auto; min-width:0; height:56px; padding:0 3px; border-radius:0`.
  - `p`: `font-size:0!important; text-align:center`. The real label stays in the DOM for screen readers, and the `/ ""` alt text stops it from being read twice.
  - `::before`: `display:block; font-size:20px; line-height:22px`. Inactive: `filter:grayscale(1); opacity:.7`.
  - `::after`: `display:block; font:600 12px/14px 'Space Grotesk'; letter-spacing:.01em; margin-top:3px; white-space:nowrap`. Inactive: TEXT at 70% opacity.
  - Active tab: emoji in full colour, word `#00E5FF` 700, the existing `rgba(0,229,255,.12)` cell background and the 3 px cyan indicator (the current active rules). `:active` flashes `rgba(0,229,255,.22)`.
  - Measured in the prototype at 11 px: tabs span x=16–374 and scrollWidth is 358. At 12 px the natural width is about 294 px, so the row fits down to 320 px.
- **Desktop (≥641 px):** today's inline 44 px row with 16 px text, byte-for-byte, plus sticky.
- **Scan beam:** a 2 px `linear-gradient(90deg,transparent,#00E5FF,#B400FF,transparent)` on `tablist::after`. It sweeps (`wolf-scan 1.2s linear infinite`) while `body:has([data-testid="stStatusWidgetRunningIcon"],[data-testid="stStatusWidgetRunningManIcon"])`. It sits on the sticky bar, so it is visible at any scroll depth.

#### 3.5.4 Sub-nav rails, L2 and L3: the same `st.radio` widgets, keys and labels, CSS only
- **Generation:** selectors are generated per `nav.STATE_KEY` value. Depth is the number of `/` in the path: 0 gives L2, 1 or more gives L3. The page-switch radios `[class*="st-key-rules_pb_"]`, `.st-key-sg_pick` and `.st-key-rules_ref_view` get the L2 look. React Aria DOM: `[role="radiogroup"] > label[data-testid="stRadioOption"][data-selected]`.
- **L2 chip rail** (`sub_odin`, `sub_review`, `sub_portfolio`, `sub_more`, plus the playbook radios):
  - Group: `display:flex; flex-wrap:nowrap; gap:8px; overflow-x:auto; scrollbar-width:none; scroll-snap-type:x proximity; margin:0 -16px; padding:4px 16px 8px; mask-image:linear-gradient(90deg,#000 calc(100% - 28px),transparent)`. Rail height is 52 px.
  - Chip: `min-height:44px; padding:0 16px; border-radius:999px; border:1px solid rgba(0,229,255,.25); background:#14141e; scroll-snap-align:center`. Text 14 px Space Grotesk 600 TEXT, `nowrap`.
  - Selected: `background:rgba(0,229,255,.14); border-color:#00E5FF`, text `#00E5FF`.
  - The radio circle is hidden with the exact node selector verified in `proto.py`, copied verbatim and locked by a test.
  - `:active` is `scale(.97)`.
  - At ≥1024 px: `flex-wrap:wrap; mask-image:none; margin:0`.
- **L3 underline rail** (`sub_odin_ravaror`, `sub_odin_makro`, `sub_odin_marknad`, `sub_motor`, `rules_sub`): rail 48 px; items 44 px, `padding:0 12px`, transparent background, no border, `box-shadow:inset 0 -1px 0 rgba(255,255,255,.08)`, text 14 px 500 MUTED. Selected: `box-shadow:inset 0 -2px 0 #00E5FF`, text TEXT 700.
- **Dividers:** the 9 `st.markdown("---")` after `_sub`/`_inner` in `wolf_panel.py` are removed. The rail carries its own bottom spacing. In Regler & Guider, the `<hr>` that `rules_page` draws after `rules_sub` is hidden with `.st-key-rules_sub + [data-testid="stElementContainer"]:has(hr){display:none}`.
- **Thor's `st.pills`** (`thor_filters`, `thor_markets`) get the same nowrap row, scroll and fade.

#### 3.5.5 Inner tabs, L4 (`thor_tabs`, `odin_gold_tabs`, `odin_sentiment_tabs` and page tabs)
- Scope: `.stTabs:not(.st-key-main_tabs)`. Colours are limited to CYAN, TEXT, `#05070A`, transparent and `rgba(0,229,255,…)`.
- Tabs are 44 px tall, 14 px Space Grotesk 600 MUTED, with a transparent background.
- Selected: TEXT 700 with `box-shadow:inset 0 -2px 0 #00E5FF`. The React Aria SelectionIndicator is hidden in inner tabs, so wrapping never misplaces it.
- **Phone:** `flex-wrap:wrap`, at most 2 rows, so all 8 Thor tabs are visible (2 × 44 px) with no hidden sideways scroll.
- `on_change="rerun"` and "only the open tab renders" (PR 10) are unchanged.

#### 3.5.6 Breadcrumb (`wolf_panel._crumb(path)`)
- Shown at depth 3 only. It is drawn by `wolf_panel` directly **above the L3 rail**, as the rail's title. For Regler & Guider it is drawn before `render_rules_page()`, which then draws `rules_sub`.
- Text: `ODIN › Råvaror`, `MER › Regler & Guider`. JetBrains Mono 12 px MUTED, 20 px row, uppercase top word. The leaf is omitted because the rail highlights it.
- Built from `nav.current(st.session_state)`.

#### 3.5.7 Page head (`ui/components.page_header` + new `head_row`)
- **`page_header(title, subtitle="", accent=GOLD, align="left")`** keeps its signature; only the default `align` changes to "left".
  - It is still ONE `st.markdown`: the `<h2>` holds `escape(title).upper()`, emoji included (PR 11: page title = tab name, and `test_watch_ui` reads "👀 ÖVERVAKNING").
  - Title: Space Grotesk 700, `clamp(20px,5.6vw,26px)`, .06em, `word-break:keep-all; overflow-wrap:normal; hyphens:manual; text-wrap:balance`. "RÅVARUHÄVSTÅNG" therefore fits on one line at 20 px (about 216 px) instead of breaking mid-word.
  - Subtitle: 13 px NOTE, line height 1.45, max-width 60ch.
- **`head_row(key, subtitle="", meta="")`** is a context manager.
  - It yields `st.container(horizontal=True, vertical_alignment="center", key=f"head_{key}")` for the title and actions. After the block it writes `components.note()` with meta (JetBrains Mono 12 px MUTED) and subtitle.
  - Pages on the head row call `page_header(title)` inside it and pass the subtitle to `head_row`, so `page_header(` stays at every call site.
  - CSS: `[class*="st-key-head_"]{flex-wrap:wrap; gap:8px; row-gap:4px}`. The title element is `flex:1 1 180px; min-width:0`; actions are `flex:0 0 auto`. If actions don't fit beside the title, they wrap to the next row, right-aligned.
  - Compact buttons in the head row: `min-height:44px; padding:0 12px; font-size:13px; letter-spacing:.04em`.
- **Placement rules:**
  - The save row stays ABOVE the head (existing decision).
  - One "📡 Uppdatera" per page, with its existing key, in the head row.

#### 3.5.8 Section label (`components.section(label, color=MUTED, count="")`)
- 12 px Space Grotesk 600 caps, .12em, MUTED (8:1; today's `section_title` is 2.77:1).
- An optional count on the right in JetBrains Mono (e.g. "2 KÖP · 5 BEVAKA", each word in its own ink colour).
- 1 px `rgba(0,229,255,.15)` underline with 6 px padding; margin 20 px above, 8 px below.
- `ui/theme.section_title`, `ui/css.section_title` and `tabs/home._block` become thin wrappers with the same signatures. Other helpers migrate gradually.

#### 3.5.9 Beslutsremsa and verdict hero
- **`verdict_strip(kicker, words, sentence="", meta="", rule=ACCENT)`** returns escaped HTML.
  - Container: background BG_ALT, 1 px `rgba(255,255,255,.08)` border, 4 px left rule, radius 12, padding 12/14.
  - Kicker: 12 px MUTED caps, .12em.
  - `words`: a list of `(text, colour)`, each in its own colour, Space Grotesk 800 24 px, joined by a MUTED " · ".
  - The rule is ACCENT when several words of different meaning share the strip, or the word's colour when there is exactly one source word.
  - Sentence: 15 px TEXT. Meta: 12 px mono MUTED.
  - **It never derives a word.** Callers pass existing words or counts.
- **Verdict hero:** `thor.cards.verdict_card`/`verdict_for` keep their text nodes ("VAD SKA JAG GÖRA?", ">KÖP<", the source chip, the GAMMAL KÖRNING flag). Only style attributes change:
  - Kicker 12 px MUTED; word 32 px 900 in the class colour; source chip under it.
  - Amber flag line; sentence 15 px TEXT; detail 13 px MUTED.
  - Border 1 px `tint(colour,.4)`, background `tint(colour,.06)`, 4 px left rule, radius 14.
  - The same anatomy is reused for the Läget regime card, the Scorecard READY / NO_TRADE box and the Positioner "✓ inga säljlägen" state.

#### 3.5.10 Tile, light grid, KPI strip, st.metric
- **`tile(label, value, color=ACCENT, sub="", sign=None, kind="num"|"word")`** returns HTML.
  - Background BG_ALT, radius 10, padding 10/12, a 3 px left rule in the value colour, min-height 64.
  - Label: 12 px MUTED caps, .1em.
  - Value: `num` is JetBrains Mono 600 20 px; `word` is Space Grotesk 700 20 px, 16 px when over 14 characters.
  - **Values containing " · " stack one part per line** at 15 px, each part coloured by the existing `laget._color(last token)` (a word-to-colour lookup, no new logic). "SPY LÅG · OMXS30 FÖRHÖJD" therefore takes 2 even lines, not 3 ragged ones.
  - `sign>0` gives GREEN, `sign<0` gives RED and `None` gives `color`. Used only for changes (P/L, 1 mån), never for prices.
  - Sub: 12 px MUTED, mono when it is a date or number.
  - Value text under 19 px goes through `ink()`.
- **Norden display rule** (display only, no change to the light): a word matching `^\d+/\d+ GRÖN$` renders as the mono count "4/11" plus a 12 px MUTED suffix "gröna", in the tile's existing colour. "4/11 gröna" in amber reads honestly; "GRÖN" in red does not. `laget.row_text` and `capital_row` are unchanged.
- **`light_grid(items)`** takes `capital_row` items unchanged. CSS grid: 2 columns up to 1023 px, 4 at 1024 px and up, gap 8.
- **`kpi_strip(items)`** is a one-line swipe strip of 132 × 64 tiles (Thor läge and snapshot). `thor/cards.kpi_row`/`kpi_chip` keep their API; their label floor goes from .6rem to 12 px.
- **st.metric that tests read** ("Klara för köp", "Invested", "Portfolio Total") stays `st.metric`. It moves into `st.container(horizontal=True, key="kpis_<page>")`, where scoped CSS (using `!important` against css.py's `stMetric !important`) gives the tile look:
  - Elements are `flex:1 1 calc(50% - 4px)`, 2 × 2 on phone and 4 in a row at 1024 px and up.
  - Label 12 px MUTED caps, value 20 px mono TEXT. No 28 px cyan and no `::before` stripe.

#### 3.5.11 Card and tap card
- **`components.card`** restyle: background BG_CARD, 1 px BORDER_FAINT, radius 12, padding 12/14, an optional 3 px accent rule, title 16 px 700, body 13 px NOTE.
- **`tap_card(key, html, label, on_click, args=(), kwargs=None)`**:

```python
with st.container(key=f"tapc_{key}"):
    st.markdown(html, unsafe_allow_html=True)
    st.button(label, key=key, on_click=on_click, args=args, kwargs=kwargs, width="stretch")
```

```css
[class*="st-key-tapc_"]{position:relative;gap:0}
[class*="st-key-tapc_"] > [data-testid="stElementContainer"]:has(.stButton){position:absolute;inset:0;z-index:2;width:auto!important;margin:0}
[class*="st-key-tapc_"] .stButton,[class*="st-key-tapc_"] .stButton button{width:100%;height:100%;min-height:0;opacity:0;padding:0}
[class*="st-key-tapc_"]:has(button:focus-visible){outline:2px solid #00E5FF;outline-offset:2px;border-radius:12px}
[class*="st-key-tapc_"]::after{content:"›";position:absolute;right:14px;top:50%;transform:translateY(-50%);
  color:#00E5FF;font:700 22px/1 'Space Grotesk';pointer-events:none;z-index:3}
[class*="st-key-tapc_"] [data-testid="stMarkdownContainer"] > div{padding-right:36px!important}
[class*="st-key-tapc_"]:active [data-testid="stMarkdownContainer"] > div{transform:scale(.99);background:#1a1f25!important}
```

  - The key, label, callback and args stay exactly as today: Thor keeps "Öppna →" and `thor_open_*`, and home keeps `home_buy_*`, `home_laget`, `home_holdings`, `home_goal` and `home_alert_*`.
  - The chevron comes from CSS, so `candidate_card` HTML and its text nodes are unchanged.
  - **A tap card holds no other interactive element.**

#### 3.5.12 Status chips: vocabulary mapping (`components.chip(word, context=VERDICT)` built on `vocab.pair`)

| Source words (examples) | Main word | Colour | Chip style |
|---|---|---|---|
| KÖP, BUY, GOLDEN TICKET, KLAR FÖR KÖP, Köpkandidat, KÄRNINNEHAV, ACCUMULATE n, AGERA | **KÖP** | GREEN | filled, text `#05070A` (4.7:1) |
| SÄLJ, EXIT, SELL | **SÄLJ** | RED | filled, text `#ffffff` (4.9:1) |
| REDUCE, DELVINST, PARTIAL | **MINSKA** | AMBER | outlined 1.5 px, AMBER text |
| BEVAKA, WATCH (candidate), WAIT, READY, DEVELOPING, Bevakningslista | **BEVAKA** | AMBER | outlined, AMBER text |
| WATCH (context HOLDING) | **VAKSAM** | AMBER | outlined, AMBER text with a leading "●" |
| HÅLL, HOLD | **HÅLL** | GREEN | outlined, `ink(GREEN)` text, "HÅLL ✓" |
| PASS (verdict), NO TRADE, INGEN AFFÄR, REJECT, VILA, SPÄRRAD | **NEJ** | GREY | outlined, MUTED text; SPÄRRAD keeps its strike-through |
| DATA UNAVAILABLE, DATA_GAP, DATA GAP, NO DATA, OKÄND, None | **Saknas** | GREY | dashed 1 px GREY, MUTED text, reason in `title` |
| NEUTRAL (unmapped on purpose) | — | GREY | source word as-is |
| Check context: PASS / ✓ · MANUAL / ◌ · FAIL / ✗ | OK · KOLLA · NEJ | GREEN · AMBER · RED | small chip with ✓ / ◌ / ✗ |

- **Chip size:** 22 px tall, padding 0 8 px, radius 6, 12 px 700 caps.
- **Source chip:** BG_CARD background, 1 px BORDER, MUTED text, never recoloured. Example: `[KÖP] [GOLDEN TICKET]`.
- **Lights are not chips.** GRÖN/GUL/RÖD, NORMAL/SKÄRPT/HALVERAD RISK and LÅG/FÖRHÖJD/HÖG render as an 8 px dot plus the word, coloured by `tokens.REGIME_COLOR` or `laget._color`.
- **Zone words** (EMBER PÅ/SELEKTIV/AV, BERSERK phases, RISK-ON/OFF) keep their page colours and get a vocab chip only where `vocab.main` maps them.
- **Fixes this table brings:** Thor's grey BEVAKA chip on a HÅLL row is gone, because the chip colour follows the main word. PASS reads "NEJ · PASS" in Wolf Asymmetry and "OK" in a check.

#### 3.5.13 List rows
- **`action_row(word, colour, title, detail="", meta="", chips=(), jump="")`** builds one row of at least 56 px as a grid: `[chip 76px][title + detail 1fr][meta]`.
  - Title 16 px 700 TEXT; detail 13 px MUTED; meta mono 13 px.
  - `data-jump="pos-<slug>"` makes the nav script scroll to that card. Without JS the row is still informative.
- **`link_rows(key)`** is a container; tertiary buttons inside `[class*="st-key-links_"]` get the link-row style: `width:100%; min-height:48px; text-align:left`, 15 px TEXT 600, a 1 px BORDER_FAINT divider, and a cyan `›` on the right. It is scoped, never a global tertiary restyle.
- **Alert row:** title 14 px 700, time in mono 12 px MUTED. It is a tap card when it has a target.

#### 3.5.14 Domain renderers (pure, fed by existing outputs)
- **`thor.cards.storlek_grid(sz, risk_pct)`** is a 2 × 3 grid of `tile`s read straight from the `sk.size()` dict Thor already computes, after `halved()` when the regime is GUL:

  | Row | Tiles |
  |---|---|
  | 1 | KÖP `sz['entry']` · STOPP `sz['stop']` · MÅL 3R `sz['target_3r']` |
  | 2 | RISKSTORLEK `sz['risk_shares']` st · RISK `sz['risk_amt']` kr (`risk_pct` %) · TAK `sz['cap_shares']` st (`sz['cap_amt']` kr) |

  - A `None` value shows the "Saknas" chip. There is no new arithmetic.
  - `sk.row_text()` follows as a 12 px MUTED copyable footnote (`test_design_storlek`).
- **`scorecard.gate_meter(checks, labels)`** is a 7-segment bar.
  - Each segment is 1/7 of the width, 8 px tall, with 2 px gaps.
  - `True` is GREEN, `False` is RED, anything else is hatched GREY "Saknas" (`repeating-linear-gradient(45deg,#6b7280 0 3px,transparent 3px 6px)`).
  - Under it: "5/7 kryss · saknas: …" using the existing `gate_state()['missing']` strings.
  - **No colour thresholds**: there is no "orange from 5/7".
- **`holdings._holding_card(...)`**: see the Positioner screen. It is one HTML card via `st.markdown` (so `at.markdown` still finds '12000.00' and '{:,.0f} SEK').

#### 3.5.15 Mobile table strategy (`ui/tables.py`, new)
1. **Six columns or fewer:** `st.dataframe(df, hide_index=True)`, with the first column pinned via `column_config`, and missing values shown as "—" through `Styler.format(na_rep="—")` (no more "None"). It is wrapped in `st.container(key=f"tbl_{key}")`, with a right-edge fade at ≤640 px.
2. **Seven or more columns on a decision page:** `table(df, key, primary_cols)` shows a 4–5 column subset (pinned first column) plus `st.toggle("Visa alla kolumner", key=f"tbl_{key}_all")`, and renders only one of the two. **The subset always includes every column a test reads**; if needed, add the column to the subset rather than editing the test.
3. **Row-per-entity data that drives a decision** (holdings, Inställningar's "Reglerna (låsta)"): a card list from the same rows. On desktop it is a 2–3 column card grid.
4. **Raw HTML tables** (Flikguide, Fiat, Guld, Poängmodell): wrapped in `<div class="wolf-scroll">`, which gets `overflow-x:auto`, a fade, and `table{min-width:100%}`. For pages styled by the shell only, CSS `table{display:block;overflow-x:auto;max-width:100%}`.
5. **Plotly:** `config={"displayModeBar": False}`. `st.line_chart` on decision pages becomes `ui/charts.sparkline_svg(series, colour)` rendered via `st.markdown`; it plots the same series with no new computation.

#### 3.5.16 Buttons

| Kind | Spec | When |
|---|---|---|
| Secondary (selector `.stButton > button[kind="secondary"] {` kept verbatim for test_pr6) | min-height 44, radius 10, 1 px CYAN border, background `rgba(0,229,255,.08)`, CYAN 14 px 600 caps .06em (was 3 px), padding 0 16 | default action |
| `:disabled` (new) | opacity .45, border `rgba(0,229,255,.25)`, no glow, `cursor:not-allowed` | today it looks enabled |
| `:active` | `scale(.98)` | – |
| Primary, **before decision 4** | left as today (PR 6: "primärknappar lämnas i fred"), except scoped new primaries: `.st-key-login_btn button` and `[class*="st-key-morning_next_"] button` get cyan fill with `#05070A` text, 52 / 48 px | – |
| Primary, **after decision 4** | `[theme] primaryColor="#00E5FF"`, plus CSS `button[kind="primary"]{color:#05070A}` (cyan with white text is 1.2:1) | one main action per step |
| Dirty save (after decision 4) | `[class*="st-key-save_"] button[kind="primary"]{background:#d4943a;border-color:#d4943a;color:#05070A}` | "osparat" never looks like "Lägg till" |
| Confirm delete (after decision 4) | `[class*="confirm"] button[kind="primary"]{background:#c44545;color:#fff}` (`confirm_delete` key `…__confirm`) | "Ja, ta bort" |
| Tertiary or link | inside `links_*` containers only (§3.5.13); Morgonrundan pill: tertiary, 44 px, cyan 13 px | navigation |
| Compact | inside head and save rows: 44 px, 13 px | 📡 Uppdatera, ⬇ CSV, 🔒 Lås |
| Download and form submit | `[data-testid="stDownloadButton"] button` and `[data-testid="stFormSubmitButton"] button` styled as secondary unless primary | – |

#### 3.5.17 Save row and osparat snackbar
- **`storage_ui.save_bar`** keeps its function, keys (`save_<name>`), dirty-to-primary logic, error texts and disabled state. Only the layout changes: `st.columns([1,4])` becomes `st.container(horizontal=True, vertical_alignment="center", key=f"{key}_row")`. The result is one 44 px line: `[💾 Spara]  Inga osparade ändringar · senast 16:40 UTC` (13 px MUTED, time in mono).
- **Positioner:** the lock and save bars share this row.
- **Snackbar (phone only, ≤640 px):**
  - Drawn inside the brand bar HTML on the end-of-run refill whenever `storage.dirty_stores()` is non-empty: `<div class="wolf-snack">⚠️ Osparade ändringar · Spara ↑</div>`.
  - `position:fixed; left:16px; right:16px; bottom:calc(12px + env(safe-area-inset-bottom)); height:44px; radius:12`, background `rgba(212,148,58,.16)` over a blurred `#0c0c12`, 1 px AMBER border, AMBER 14 px 600.
  - There is only ever ONE snackbar, even with two dirty stores.
  - Tapping it makes the nav script `scrollIntoView({block:"center"})` the first `[class*="st-key-save_"] button[kind="primary"]`.
  - While it shows, `body:has(.wolf-snack)` adds 72 px bottom padding to the block container and lifts `[data-testid="stToastContainer"]` by 56 px.
  - Fallback if fixed positioning misbehaves on iOS: the brand-bar dirty chip alone.
- **Toast:** after a successful save, `st.toast("Sparat", icon="💾")`, in addition to today's `st.success`.

#### 3.5.18 Confirmations
- **Deletes:** `components.confirm_delete` guards every delete: Positioner (`rm_{pkey}_{ticker}`, inside the "⚙ Edit" expander), the Veckorutin watchlist ✕ and "Rensa historiken".
- **State-locking switches:** the Veckorutin market-filter toggle moves into `st.popover("Marknadsfiltret ▾")` with a one-line warning. The toggle widget and its key are unchanged; AppTest reaches widgets in popovers, as verified.
- **No dialog** replaces any existing widget.

#### 3.5.19 Loading, empty, missing, error, toasts
- **Running:** the scan beam (§3.5.3).
- **First load:** `components.skeleton(n, h=96)` shows shimmering BG_CARD → BG_ALT cards in an `st.empty()`, only when the data is not yet in the session (I DAG `ht.load()`, Thor `load()`). It is cleared after the load. The spinner text stays.
- **Long live work:** `st.status("Räknar marknadsrisken …", expanded=True)` when the Läget toggle is on.
- **Empty:** `empty_state(text)` is BG_CARD with a 1 px dashed BORDER, 14 px MUTED, centred, plus an optional single action button.
- **Missing:** `components.missing(why="")` is the dashed "Saknas" chip. It replaces "DATA UNAVAILABLE", "DATA GAP"/"DATA_GAP", "None" and "No data available" in display code, via `vocab.main(x) == SAKNAS`. Stored values are unchanged. F&G components show "Saknas → 50 (neutral)".
- **Errors:** `st.error`/`st.warning` are kept. Repeated messages are deduplicated: the Gist read error is printed once per page as an `st.warning` "Gisten gick inte att läsa: Wolf · Viking · Viking Nine …", with the strategy names as chips.
- **Toasts:** Spara, Logga köp, 👀 Bevaka (already has one), Registrera beslut.

#### 3.5.20 Morgonrundan (`ui/routine.py`, new; navigation only)
- `STEPS = (("home","I DAG"), ("laget","Läget"), ("thor","Thor"), ("scorecard","🎯 Scorecard"), ("holdings","Positioner"))`. Every key is in `nav.TARGETS`.
- `mark(step)` writes `st.session_state["morning"] = {"date": now_se().date().isoformat(), "seen": [...]}` and resets when the date changes. It holds no data and no rule.
- **Active** means today's round has started from I DAG. On I DAG the pill and footer always show; on the other four pages they show only while the round is active, so a normal visit stays clean.
- `pill(step)`: tertiary `st.button(f"{i}/5 ›", key=f"morning_top_{step}", help=f"Morgonrundan — nästa: {label}", on_click=links.go, args=(next,))`, placed in the head row.
- `footer(step)`: dots ●●○○○ (cyan for done, MUTED for not yet), then a full-width primary `f"Nästa: {label} →"` (`morning_next_{step}`). The last step shows "✓ Morgonrundan klar · Till I DAG ↑" (`morning_next_holdings` → home).
- **Tap count:** 4 (Läget, Thor, Scorecard, Positioner), each reachable without scrolling. Today it takes about 5 taps, a sideways swipe of the tab bar and long scrolls.

#### 3.5.21 Footer (`render_footer`)
- The storage footer line, then one 12 px MUTED line: "Nordic Arc Systems © {datetime.now().year} · See What the Market Can't."
- Then "▲ Till toppen": `<a class="wolf-top" href="#nas-top">`, 44 × 44. The JS click scrolls `stMain` to the top, with the anchor as the no-JS fallback.

#### 3.5.22 Nav script (`ui/shell.nav_js(route)`)
- It is one `st.html(..., unsafe_allow_javascript=True)` inside the hidden shell.
- The route key `f"{nav slug}|{thor_ticker}"` is embedded in the text, so the element re-renders only when the route changes.
- **The script body contains no `<` character**, because DOMPurify drops a script with `<` followed by a letter; a test locks this.

```js
(function () {
  var d = document, w = window, route = "__ROUTE__", RAILS = "__RAIL_SELECTORS__";
  function sc() { return d.querySelector('[data-testid="stMain"]') || d.scrollingElement; }
  function centre(g) { var s = g.querySelector('[data-selected]'); if (!s) return;
    var a = g.getBoundingClientRect(), b = s.getBoundingClientRect();
    var off = (b.left - a.left) - (a.width - b.width) / 2;
    if (Math.abs(off) > 4) g.scrollBy({left: off, behavior: 'smooth'}); }
  function pass() { d.querySelectorAll(RAILS).forEach(centre); }
  var N = w.__wolfNav;
  if (!N) {
    N = w.__wolfNav = {route: null, busy: false};
    new MutationObserver(function () { if (N.busy) return; N.busy = true;
      w.requestAnimationFrame(function () { N.busy = false; pass(); }); })
      .observe(d.body, {subtree: true, childList: true, attributes: true, attributeFilter: ['data-selected']});
    d.addEventListener('click', function (e) {
      var t = e.target;
      if (t.closest('.st-key-main_tabs [role="tab"][aria-selected="true"], .wolf-top')) {
        e.preventDefault(); sc().scrollTo({top: 0, behavior: 'smooth'}); }
      var j = t.closest('[data-jump]'); if (j) { var el = d.getElementById(j.getAttribute('data-jump'));
        if (el) el.scrollIntoView({block: 'start', behavior: 'smooth'}); }
      if (t.closest('.wolf-snack')) { var s = d.querySelector('[class*="st-key-save_"] button[kind="primary"]');
        if (s) s.scrollIntoView({block: 'center', behavior: 'smooth'}); }
    }, true);
  }
  if (N.route !== route) { if (N.route !== null) sc().scrollTo({top: 0}); N.route = route; }
  pass();
})();
```

`RAILS` is the generated selector list for the L2/L3 and playbook `[role="radiogroup"]` elements.

---

## 4. Screens

Coordinates are for a 390 × 844 iPhone with a 664 px visible fold. Every page shares the frame: **brand bar 0–44, sticky tabs 44–100**, then a depth-2 rail at 100–152 and a depth-3 crumb plus rail at 152–220.

### 4.1 Login (`auth.render_login_gate`; logic unchanged)
**Above the fold**
- No Streamlit header. `inject_css()`/`render_shell()` now run before the gate, so the page has the panel background (`#05070A` with the faint cyan/purple radials and the 40 px grid) and the panel fonts. auth's own `<style>` keeps only `.arc-login-*` rules, with no second `@import` and no 900 weight.
- A centred column, max-width 360 px.
- y 48–196: `assets/logo_arc.*` at 148 px wide with `drop-shadow(0 0 32px rgba(0,229,255,.3))` and a 4 s breathing glow (static with reduced motion). There is no duplicate "NORDIC ARC SYSTEMS" text, because the art already holds it.
- y 208–228: "See What the Market Can't." in 14 px italic MUTED.
- y 252–384: `st.form("login_form", border=False, enter_to_submit=True)`:
  - "Åtkomstkod" label, 13 px MUTED, visible.
  - `st.text_input("Åtkomstkod", type="password", key="login_pw", placeholder="Ange åtkomstkod")`: 52 px, 16 px JetBrains Mono, cyan text on BG_ALT, radius 10, focus border `#00E5FF`.
  - `st.form_submit_button("LOGGA IN", key="login_btn", type="primary", width="stretch")`: 52 px, cyan fill, `#05070A` 700 text with .12em tracking.
  - The keyboard's "Gå" submits, so the button may sit under the keyboard.
- y 396: on a wrong code, "Fel kod — försök igen." in RED under the button (`st.error`, role=alert).
- y 424: "Inloggningen sparas i {REMEMBER_DAYS} dagar på den här enheten." in 12 px MUTED.
- Invisible: the existing localStorage/cookie bridge script, unchanged.

**Desktop:** the same column as a card: BG `#0c0c12`, 1 px `rgba(0,229,255,.15)` border, radius 16, padding 28; logo 180 px.

### 4.2 Frame, every page
1. Brand bar 44 px, scrolls away: [eye] NORDIC ARC … `lör 21 nov` · `● 2 osparade` · 🔓 · ●●●●.
2. Tab bar 56 px, sticky: 🏠 I DAG · 👁 ODIN · 🔨 THOR · 📒 ARKEN · 💼 PORTFÖLJ · ⋯ MER, emoji over the word, all six visible; the scan beam runs on its bottom edge.
3. L2 chip rail, 52 px (ODIN, ARKEN, PORTFÖLJ, MER), auto-centred on the selection, including after `?p=`.
4. At depth 3: the crumb (20 px) and the L3 underline rail (48 px).
5. Page: save row (sheets only) → head row → Beslutsremsa → evidence → tools.
6. Footer: storage line · © {year} line · "▲ Till toppen".
7. Floating, phone only, when dirty: the osparat snackbar.

**Desktop:** brand bar 52 px with SYSTEMS and the tagline, the ⋮ menu floating at its right; the tab row inline at 16 px, sticky; rails wrap; content max 1280 px, centred.

### 4.3 🏠 I DAG: Morgonbeskedet (`?p=home`)
**Above the fold**
- 0–100: frame.
- 108–152: head row: **I DAG** (gold) · "📡 Uppdatera" (`home_refresh`) · pill "1/5 ›".
- 156–174: meta in 12 px mono MUTED: "lördag 21 november · hämtad 16:52".
- 182–204: kicker (section label) **DAGENS BESKED · ANTAL I DAG**.
- 208–230: the sentence, 15 px TEXT:
  - `ht.no_turn_line` ("Ingen vändning i köpfönstret · 2 på väg"), or the first `ht.turn_headline`;
  - "Inga KÖP, SÄLJ eller vändningar i dag" when every count is 0;
  - "Saknas — Gisten gick inte att läsa" (grey) when the Gist failed, with the error printed once.
- 238–390: **four count tiles**, 2 × 2, each a `tap_card`. Counts come from the new pure `ht.counts()`, built only from `turn_cards`, `ht.qualifiers`, `ht.holdings_lines` and `routines.due_today`:

  | Tile | Value (colour) | Sub | Tap (key) → target |
  |---|---|---|---|
  | ⚡ VÄNDNING | n (EMBER) | "m på väg" | `home_tile_turn` → `valhall_map` |
  | KÖP | n (GREEN) | "k BEVAKA" | `home_tile_buy` → `thor` |
  | SÄLJ · MINSKA | n (RED) | "j VAKSAM", or "🔒 Låst" when locked | `home_tile_sell` → `holdings` |
  | FÖRFALLER | n (AMBER) | "rutiner i dag" | `home_tile_due` → `nav_goto` "more/regler-guider/arshjulet" |

- 400–616 (when no turn window is open):
  - **KAPITALET** section label, then `light_grid` 2 × 2 of `capital_row`: Allokering NORMAL "6,0 % från toppen" · Swing GUL "2026-11-20" · Marknadsrisk SPY LÅG / OMXS30 FÖRHÖJD (two lines) · Norden "4/11 gröna".
  - The whole grid is ONE `tap_card` (key `home_laget`, label "Läget i Odin →").
  - Note in 12 px MUTED: "Ljusen slås inte ihop."
- 636: the KVALAR OCH STORLEK label just shows.

**Below the fold, in PR 8 order: vändning → kapital → kvalar → storlek → sälj → mål.**
- **VÄNDNING** (only when a window is open, placed before KAPITALET):
  - An EMBER-ruled card: "⚡ VÄNDNING — Guld · dag 2 av 5" at 18 px 900.
  - Ticker buttons as a wrapping chip row, keys `home_turn_*` unchanged: "Deep Contrarian: X → Thor".
  - ETF line in 12 px: "… · verifiera linjen hos Nordnet".
- **KVALAR OCH STORLEK**, with the count "2 KÖP · 5 BEVAKA" on the right:
  - Up to 5 tap-card action rows (keys `home_buy_<strategy>_<ticker>`, labels unchanged, e.g. "VOLV-B.ST → Thor (Viking Nine)").
  - Each row: `[KÖP]` VOLV-B.ST, a `[Viking Nine]` source chip, `[NY]`, and a mono line "120 st · risk 1 500 kr · tak 22 st" from the new `ht.size_parts()`. `size_line` is unchanged.
  - "GAMMAL (körd …)" tag where it applies.
  - Then a link row "Hela listan i Thor →" (`home_thor`).
- **SÄLJ OCH BEVAKA**, one tap card (`home_holdings`, label "Positioner →"):
  - action rows `[SÄLJ]`, `[MINSKA]` or `[VAKSAM]` · ticker · reason;
  - or `ht.all_hold_line` in MUTED;
  - or "🔒 Innehaven är låsta — lås upp under PORTFÖLJ → Positioner".
- **MOT MÅLET**: one gold-ruled tap card with `mm.today_line` (`home_goal`).
- **SENASTE LARMEN**: 5 alert rows (tap cards `home_alert_i` when they have a target); the rest in "Visa alla (n)".
- **Morgonrundan footer**: ●○○○○ and "Nästa: Läget →" (`morning_next_home`).
- **🗺 Flikguiden** expander, unchanged in content and keys. The duplicated "ODIN · ODIN —" prefix is fixed.

**Desktop ≥1024 px:** the strip and tiles full width (4 tiles in a row), then `st.columns([3,2])`, left = vändning, kapital, kvalar; right = sälj, mål, larm. The columns stack left then right on the phone, so the PR 8 order holds.

### 4.4 👁 ODIN → Läget (`?p=odin/laget`)
**Above the fold**
- 0–152: frame and L2 rail, with [Läget] selected among Kartan · Sektorrotation · Råvaror · Makro · Marknad.
- 160–204: head row: **👁 LÄGET** · ONE "📡 Uppdatera" (`laget_refresh`, which now also forces `wolf_regime_ui._get_data(force=True)`) · pill "2/5 ›".
- 208–246: subtitle: "Var kapitalet ska ligga i dag: fyra ljus ur fyra källor, vart och ett med sitt eget ord."
- 256–408: **Beslutsremsa** = `light_grid` 2 × 2 of `capital_row`, with words, sub lines and colours as today.
- 414–432: note in 12 px NOTE: "Raden slår inte ihop ljusen — varje strategi läser sitt eget enligt sina regler."
- 452–474: section **SWING-LÄGET · uppdaterad 20 nov 18:05**. It replaces the white 44 px h1 "Regim" and the second 📡, through `render_wolf_regime_page(embedded=True)`; the default stays False for the motor page.
- 482–620: regime card in the verdict-hero anatomy: "● GUL — selektiv, halv storlek" at 20 px 800, with three rule lines and one bullet each (the display strips the duplicated "• "; the rule texts are unchanged).

**Below the fold**
- Evidence: three compact cards.
  - **OMXSPI VS MA200**: "3,1 % över" plus a 30-day `sparkline_svg` and the caption "1012 vs MA200 981 · 30 dagar".
  - **MARKNADSBREDD**: "48,0 %" with the existing amber bar. The caption is escaped, so it renders exactly "> 55 % friskt · 45–55 % smalnar · < 45 % bredden dör — den dör före index."
  - **KVALANDE BOLAG**: "23 av 310" plus its caption.
- "Historik" in a compact expander.
- **Marknadsrisk**: a 56 px toggle row "🌩️ Visa Marknadsrisken (räknas live, tar en stund)" (key `laget_risk`, still turned on by `&risk=1`). When on, it renders inside `st.status`.
- "Kolla en ticker": the existing input at 48 px.
- Morgonrundan footer: "Nästa: Thor →".

**Desktop:** the 4 lights in one row; the regime card and evidence in `st.columns([2,3])`.

### 4.5 🔨 THOR: kandidatlistan (`?p=thor`)
**Above the fold**
- 0–100: frame.
- 108–152: head row: **🔨 THOR** · "📡 Uppdatera" (`thor_refresh`) · ⓘ `st.popover` "Om listan", holding the existing subtitle and footnote verbatim · pill "3/5 ›".
- 156–174: meta: "hämtad 16:52 · 6 rader".
- 182–246: **läge strip** (`kpi_strip`, swipeable, same `lage_items` labels and order): Swing-regim GUL · Norden — · Globalt — · Kandidater "6 · 2 KÖP".
- When there are problems (+40 px): ONE amber line "⚠ 10 körningar gamla/saknas/fel — se Körningar per strategi nedan". The Gist error is printed once, with names as chips.
- 256–304: search row:
  - `thor_ticker`, 48 px, placeholder "Skriv en ticker — BOL.ST, ERIC B, NEM — eller tryck på en rad";
  - "⚙ Filter" `st.popover` (48 px; the label shows "⚙ Filter · 2" when something other than the default is chosen) holding **Screening** (`thor_pick`) and **Segment** (`thor_segment`). The keys are unchanged.
- 312–356: quick pills row "bara KÖP · bara Norden · bara nya" (`thor_filters`), then a "🌍 Marknad ▾" popover holding the `thor_markets` pills.
- "⚡ Skanna nu" row only when one strategy is picked (logic unchanged).
- 364–380: count eyebrow, 12 px cyan: "6 KANDIDATER · 2 KÖP · ALLA STRATEGIER".
- 388–606: **candidate rows**: `cards.candidate_card` HTML unchanged inside a `tap_card`.
  - Overlay label "Öppna →", key `thor_open_<strategy>_<ticker>`, the same `_goto`, and a cyan chevron.
  - About 105 px per row instead of about 190. Two full rows show above the fold.

**Below the fold**
- "Visa fler (n till)" (`thor_more`).
- The segment's largest companies without a signal, as grey "INGEN SIGNAL" tap rows (`thor_open_storst_*`).
- Moved below the list: "Körningar per strategi · n gamla/saknas/fel" expander (label amber when anything is stale) and Universum-registret.
- Morgonrundan footer: "Nästa: 🎯 Scorecard →".

**Desktop:** filters on one line; rows in a 2-column grid at 1280 px and up (`st.container(key="thor_rows")`).

### 4.6 🔨 THOR: analysen (`?p=thor&t=VOLV-B.ST&s=viking_nine`)
In analysis mode the list chrome (head subtitle, läge strip, freshness expander) is **not drawn above** the analysis. It stays reachable below. The verdict moves from about 1,350 px down to about 334 px.

**Above the fold**
- 0–100: frame.
- 108–156: nav row (`st.container(horizontal=True, key="thor_nav")`): "‹ Kandidater" tertiary button (`thor_back`, same `_close`) and the ticker field (`thor_ticker`) filling the rest of the row.
- 164–270: `header_card`, unchanged: "VOLV-B.ST" 24 px · Volvo B · chips (^OMX · OMXS30, Industri · GRÖN · MITTEN, INNEHAV) · "finns i dag i: [Viking Nine][KÖP][GOLDEN TICKET]".
- 278–326: "Bedöm som" (`thor_strategy`), with its label inline on the left via scoped CSS and the select filling the rest of the row.
- 334–494: **Verdict** `cards.verdict_for`, unchanged text: "VAD SKA JAG GÖRA?" · **KÖP** 32 px green · [GOLDEN TICKET] chip · "Viking Nine: GOLDEN TICKET · Nine 9/9 · viktat 92" · detail; plus the GAMMAL KÖRNING flag when relevant.
- 506–664: **STORLEK** section label, then `storlek_grid` row 1 (KÖP 271.40 · STOPP 259.00 · MÅL 3R 308.60).

**Below the fold**
- Row 2: RISKSTORLEK 120 st · RISK 1 500 kr (1,5 %) · TAK 22 st (6 000 kr). Then the `row_text` footnote and the existing notes: planning-only when the verdict isn't KÖP or the run is stale, and "Kapital 100 000 kr (standard) — fyll i ditt eget under MER → ⚙️ Inställningar."
- Action row:
  - "👀 Lägg i Övervakning (Viking Nine)" (`thor_watch_add`, toast) or "👀 Bevakas sedan … — öppna Övervakning →" (`thor_watch_open`);
  - the "📝 Logga köp" expander (`thor_buy_shares_*`, `thor_log_buy`, disabled logic unchanged, toast);
  - for sheet strategies, "Köpgrinden: öppna 🎯 Scorecard →" (`thor_to_scorecard`).
- Snapshot as a swipe `kpi_strip`: Kurs · 1 mån · 3 mån · ATR · EMA50 · EMA200 · 52 v. Only the changes (1 mån, 3 mån) are sign-coloured. With no data, "Saknas — ingen kursdata för VOLV-B.ST".
- Inner tabs `thor_tabs` (only the open one renders), wrapped onto 2 rows: Regim · Teknik · Cykel · [Råvara] · Asymmetri · Arken · Backtest · Regler. Their content (passed and failed rules, gate chips) is unchanged; bridge buttons become link rows.
- "◀ Kandidater (6)" expander (label kept), the Läge block, Universum-registret.

**Desktop ≥1024 px:** `st.columns([5,7])`. Left (decision): header, Bedöm som, verdict, storlek, actions. Right (evidence): snapshot and inner tabs.

### 4.7 📒 ARKEN → 🎯 Scorecard (`?p=review/scorecard`)
**Above the fold**
- 0–152: frame and L2 rail, [🎯 Scorecard] selected among Rick Rule · Royalty C · Poängmodell · Tiggre · Insider · 🧭 Durrett · 🐺 Wolf Asymmetry · 🚀 Råvaruhävstång, auto-centred.
- 160–204: save row: [💾 Spara] "Inga osparade ändringar · senast 16:40 UTC" (`save_scorecard`).
- 212–256: head row: **🎯 SCORECARD** · "⬇ CSV" (the existing export) · pill "4/5 ›".
- 260–298: subtitle: "Sista steget före köp. Kandidaterna läses ur de andra flikarna — här fattas bara beslutet."
- 306–396: **Beslutsremsa**: `verdict_strip("KÖPGRINDEN · ANTAL", [("0 KLARA FÖR KÖP", MUTED), ("17 kandidater", TEXT)], sentence="Luckor i tabellen = standardbeslut INGEN AFFÄR")`. The first word is GREEN when the count is above 0.
  - The two `st.metric` stay in the DOM, because `test_thor_scorecard_entry` reads "Klara för köp". They move under the strip into `kpis_scorecard`, restyled as a compact pair; on the phone they are hidden by CSS only if the owner accepts the strip as their replacement.
- 408–456: Fokus row (the `review.focus` select and "Visa alla"), unchanged.
- 466–664: card heads, 48 px each; 3 full heads plus 1 partial show above the fold.

**Card list**
- **Order:** display-only sort: ready first, then by gates passed descending, then ticker. It uses the `gate_state()` result already computed for each card.
- **Expander:** `st.expander(head, key=f"sc_card_{key}")`. A keyed expander stays open when its label changes, as verified.
- **Head:**
  - It starts with the ticker (tests split on " ·"): "AEM.TO · Agnico Eagle · Durrett · 5/7 kryss" in plain MUTED-looking markdown.
  - When ready: "LUG.ST · Lundin Gold · Rick Rule · :green[✅ KLAR FÖR KÖP]".
  - The strategy name comes from the existing display map in `scorecard.py`.
  - No threshold colours below 7/7.
- **Inside the card, FIRST** (the slot, filled last with this run's values): ONE markdown holding:
  - the `gate_meter`;
  - the verdict box: green "KLAR FÖR KÖP", or red "Luckor i tabellen = standardbeslut INGEN AFFÄR" with the missing gates;
  - the existing "1/12 fält ifyllda" and "Saknas för köpbeslut: …".
  
  One markdown keeps `test_review_focus`'s `markdown[0]` assertions ('x/y fält ifyllda', 'Saknas för köpbeslut:', `#c44545`). When ready, "Registrera beslut" (`sc_dec_<key>`) sits right under it, in the slot container.
- **Then the existing order, unchanged:**
  - Strategi · Position (% av total) · Position (% av strategidelen), the two fields side by side on one row;
  - the rule and position lines, summary chips;
  - Värdering · Trigger · Vald säljregel (+ playbook pick);
  - **Köpgrinden**: 7 checkbox rows, 48 px each, keys `sc_g_<key>_<gate>`, with the mechanical gates as today;
  - Underlag till ChatGPT.
- The old bottom verdict box is removed, because it moved up.
- After the list: the 🗓 Årsregeln expander and the Morgonrundan footer "Nästa: Positioner →".

**Desktop:** the strip and metric pair inline with the Fokus row. Inside a card, `st.columns([5,7])`: verdict and Köpgrinden on the left, underlag on the right. It stacks verdict-first on the phone.

### 4.8 📒 ARKEN → Rick Rule (`?p=review/rick-rule`; Royalty C is identical)
**Above the fold**
- 0–152: frame and rail.
- 160–204: `render_fiat_environment()` (background, no signal), wrapped in a collapsed expander "🌍 Bakgrund — fiatmiljön (styr inga beslut)". Its position is unchanged; it is 44 px instead of a full block.
- 212–256: save row. 264–308: head row **RICK RULE** · "Hämta nu" (existing).
- 316–396: **Beslutsremsa**: verdict counts from rows already scored, each in its vocab colour, e.g. "1 Köpkandidat · 2 Bevaka · 3 Passa" (GREEN / AMBER / GREY), kicker "BEDÖMNINGAR · ANTAL".
- 404–450: the scoring note "Marginal ≥ 40 % ger 2 p …", verbatim, at 13 px NOTE. It stays visible.
- 458–506: Fokus row. 514 onward: card heads.

**Cards**
- Head: "LUG.ST · Guld · 4/5 p · :green[Köpkandidat]". Only the verdict word is wrapped, the expander is keyed, and the "/5 p" marker and ticker prefix are kept.
- Top slot: "4/5 :green[Köpkandidat] — Marginal och disciplin på plats.", then the count line and "Saknas" chips for missing fields. This appears before the inputs instead of about 1,100 px down.
- Then the existing inputs, two per row on phone.

### 4.9 💼 PORTFÖLJ → Positioner (`?p=portfolio/positioner`)
**Locked** (`private_lock.gate`, logic unchanged): at 160–420, a centred card.
- "#### 🔒 Dina innehav är låsta" stays as markdown.
- The note, then `st.form(f"{key}_form", border=False)` holding the password field `holdings_pw` (52 px, 16 px) and `form_submit_button("Lås upp", key="holdings_pw_btn")`. "Gå" unlocks.
- "För många försök — vänta N sekunder." and "Fel lösenord." are unchanged. Övervakning stays outside the lock.

**Unlocked, above the fold**
- 0–152: frame and rail, [Positioner] selected among 👀 Övervakning · 🎯 Mot målet · 📊 Papperskonton · Journal · Veckorutin · Allokering.
- 160–204: ONE row (`st.container(horizontal=True)`): [💾 Spara] · [🔒 Lås innehaven] (`holdings_pw_lock`, label kept) · "Inga osparade ändringar".
- 212–256: head **POSITIONER** via `page_header` (gold, left; replaces the 3.4 rem cyan h1) · pill "5/5 ›". 260–278: "Alla strategiers positioner · säljreglerna · signalerna".
- 288–520: **Beslutsremsa "BEHÖVER ÅTGÄRD I DAG"** (a slot filled after the sections have computed their signals), as `action_row`s with `data-jump="pos-<slug>"`:
  - `[SÄLJ · EXIT] EQNR.OL · Long Trend · Below EMA200`
  - `[MINSKA · REDUCE] VOLV-B.ST · Quality · 5d below EMA200`
  - `[VAKSAM · WATCH] …`
  - amber rows from the rule monitor, e.g. "⚠ >10% position: VOLV-B.ST (39.1%)", with the ">" escaped so it renders.
  - The first three rows show, then "+n till ↓".
  - Empty: green outlined "✓ Inga säljlägen i dag — alla innehav HÅLL".
  - EQNR's EXIT moves from y≈3,500 to y≈312.
- 530–664: the 4 existing `st.metric` (labels "Total Holdings", "Invested", "Cash", "Portfolio Total" and the `{:,.0f} SEK` format kept) as a 2 × 2 grid in `kpis_holdings`.

**Below the fold**
- "Cash (SEK)" number input, full width.
- STRATEGY ALLOCATION bars (existing), every text at least 12 px.
- "📥 Uppdatera från mäklaren (Nordea ASK · Nordnet ISK)" expander.
- **Per strategy:**
  - Header row: a 4 px bar in `STRATEGY_COLORS`, "QUALITY" at 17 px, and on the right the existing summary "3 innehav · 138,980 SEK · 32.8 %".
  - "➕ Add Quality holding" expander.
  - **One `holding_card` per holding** (about 120 px, `id="pos-<slug>"`):
    - L1: **TICKER** 17 px 700 · sector · "200 st", and on the right a signal chip pair `[MINSKA] [REDUCE]`, `[SÄLJ] [EXIT]`, `[VAKSAM] [WATCH]` or `[HÅLL ✓] [HOLD ✓]`.
    - L2, mono 14 px: "271.40 (−2.6 %) · värde 54,280 · P/L +13.1 % (+6,280)". Only the day change and P/L are sign-coloured; the price is TEXT.
    - L3, 12 px MUTED: signal detail, the quality verdict or contrarian stage, tranche dots and a regime chip.
  - Under each card, a compact 44 px "⚙ Edit <ticker>" expander (label and `edit_stop_*`, `edit_target_*`, `edit_sec_*`, `save_edit_*` keys kept). It now also holds `confirm_delete("Ta bort", key=f"rm_{pkey}_{ticker}")`, replacing the one-tap ✕.
  - The orphaned column-header row is hidden ≤640 px.
- Rule monitor: `st.warning` elements kept and escaped, restyled as amber cards. Then Tranches.
- Risk · Rapporter · Korrelation · Metaller as today. Becoming inner tabs is backlog (decision 15). "FYSISKA METALLER" gets a section label instead of "---".
- Morgonrundan footer: "✓ Morgonrundan klar · Till I DAG ↑".

**Desktop:** the action strip and KPIs side by side (`st.columns([3,2])`). Each holding card's inner CSS grid shows the 6 old columns on one line, so it still reads as a table.

### 4.10 ⋯ MER → Regler & Guider (`?p=more/regler-guider/handelsregler`)
Styled from the shell only. `ovtlyr/ui/rules_page.py` is not edited until decision 12. `wolf_panel` wraps `render_rules_page()` in `st.container(key="nav_rules_page")` for scoping.

**Above the fold**
- 0–152: frame and L2 rail, [Regler & Guider] selected among Larm · ⚙️ Inställningar · Copilot · Motorsidor · Egen backtest.
- 160–180: crumb "MER › REGLER & GUIDER".
- 182–230: L3 underline rail `rules_sub`: 🚀 KOM IGÅNG · [📋 HANDELSREGLER] · ⚡ FUSKLAPP · 🗺 FLIKGUIDE · 📖 STRATEGIGUIDER · 🗓 ÅRSHJULET · 📚 SNABBREFERENS. One swipeable row, auto-centred; the following `<hr>` is hidden.
- 238–272: the page's own h2 "HANDELSREGLER", restyled within scope (`!important` only inside `.st-key-nav_rules_page`): gold, left, `clamp(20px,5.6vw,26px)`, keep-all.
- 276–314: intro, verbatim, 13 px: "Varje regel visar 📍 exakt var i panelen du hittar svaret. HÅRD REGEL = bryts aldrig, ingen bedömningsfråga."
- 322–374: playbook picker `[class*="st-key-rules_pb_"]` (15 options, easiest first) as ONE chip rail, auto-centred. It replaces 8 rows of radios and saves about 280 px.
- 382–462: playbook line (name, level badge, support badge, tagline, 📍 path), unchanged.
- 470–664: **RISKMODELL** box, word for word from `strategy_rules.py` (Risk per affär, Positionsstorlek, Antal positioner, Stop loss, Vinstmål), as a 38/62 label/value grid on the phone. It moves from about 1,320 px to about 470 px.

**Below the fold**
- "🔄 Så gör du — steg för steg", then the Entry, Exit, Mindset and Misstag rule rows with HÅRD REGEL chips, in exact text.
- Section heads restyled to 20 px gold with no mid-word wrap.
- Wide tables (Flikguide) get `overflow-x:auto`; the page no longer overflows to 868 px.
- Readability floor scoped to this page: inline sizes of 10, 11 px, .6, .65, .7 and .72 rem are raised to 12 px.
- Then "▲ Till toppen" and the footer.

**After decision 12 (separate PR, texts verbatim):**
- a "Hårda regler" card first;
- Fusklapp with its hard rules first;
- Kom igång as Börja här → Veckorutinen → the 15 playbooks as compact expanders;
- the "Fyra strategier" count computed from `len(LEARNING_ORDER)`;
- "om 1 dag" in the singular.

---

## 5. What stays exactly

- **Rules and numbers:**
  - `strategy_rules.py` and `strategy_rules_masterguide.py`, byte for byte.
  - Every threshold, score, gate, risk %, cap and count shown (for example Marginal ≥ 40 %, 20/35 % goal, MIN_CLOSED=20, 10 % position, 25 % sector, 8–10 holdings, P/NAV ≤ 15 %, Scorecard's 7 GATES and `gate_state`, EMBER SELEKTIV halving, the F&G neutral-50 fallback, which is labelled only).
  - The playbook, Fusklapp, Kom igång and HÅRD REGEL texts, word for word. Escape fixes render identical characters.
- **Names:** `nav.TOP` (🏠 I DAG · 👁 ODIN · 🔨 THOR · 📒 ARKEN · 💼 PORTFÖLJ · ⋯ MER, in order), every `nav.SUBS` string including emoji, `HOME_ZONES` text, and the strategy and model names.
- **Keys and links:** `main_tabs`, every `nav.STATE_KEY` value, `TARGETS`, `INNER_SEEDS`, `LEGACY_PATHS`, `LEGACY_SLUGS`, `?p=` slugs, `target_link` `t`/`s`/`risk=1`, `_apply_deep_link`, `_sync_url`.
  - Widget keys: `thor_*`, `sc_g_*`, `sc_dec_*`, `save_*`, `login_pw`, `login_btn`, `holdings_pw*`, `edit_*`, `home_*`, `fd_ccy`, `asym_mode`, `odin_gold_tabs`, `odin_sentiment_tabs`, `thor_tabs`.
  - Every button label that a test or a user habit relies on ("Öppna →", "📡 Uppdatera", "🔒 Lås innehaven", "◀ Kandidater (N)", "⚙ Edit <ticker>").
- **Widget types:** sub-nav stays `st.radio`; no tab, radio or toggle is replaced.
- **Code lines:** `st.tabs(tab_labels, on_change="rerun", key="main_tabs")` and the six `if _is_open(tabs["x"]):` blocks; `theme._banner_b64` (cached).
- **Auth:** password check, signed cookie, the localStorage bridge `?wolf_t=`, 30 days; `private_lock.gate` with HOLDINGS_PASSWORD; Övervakning outside the lock.
- **Frozen files:** `ovtlyr_nine.py`, `viking_execution.py`, `viking_exit.py`, `ovtlyr/ui/layout.py`, and the Viking Regime and ⚔️ Viking Nine Regime pages. They get the shell (frame and rails) and nothing else. Their own CSS leak is countered only by more specific shell selectors (`.st-key-main_tabs…`).
- **Not frozen but untouched until decision 12:** `ovtlyr/ui/rules_page.py`.
- **Prior decisions:**
  - PR 1: links point to TARGETS.
  - PR 3: `ui/vocab`.
  - PR 8: the I DAG order below the strip.
  - PR 9: six tabs in decision order.
  - PR 10: only the open inner tab renders.
  - PR 11: title = tab name = alert name, emoji included.
  - PR 12: page CSS scoped; the heatmap leak is fixed by scoping it.
  - The save row sits above the title.
  - "Bakgrund — styr inga beslut" notes stay.
  - "primärknappar lämnas i fred" holds until decision 4.
  - No autotrading; every candidate still shows passed and failed rules and the reason for its status.

---

## 6. Owner decisions needed (yes/no)

| # | Question | Rec. | If no |
|---|---|---|---|
| 1 | Phone tab bar: six stacked emoji-over-word cells (CSS-generated from `nav.TOP`), **sticky** at the top on all widths? | **JA** | Today's scrolling row, plus JS that scrolls the active tab into view |
| 2 | Replace the 260 px banner on every page with the 44/52 px brand bar? The art stays on the login. | **JA** | Banner kept and shrunk to 120 px |
| 3 | Hide Streamlit's header, Deploy and ⋮ on phones (`toolbarMode="viewer"` + CSS; ⋮ floats on desktop)? | **JA** | Header stays; chrome +60 px |
| 4 | `[theme]` cyan primary plus the panel palette (primaryColor `#00E5FF`, backgroundColor `#05070A`, red/orange/green/violet = panel tokens), with dirty Spara amber and "Ja, ta bort" red? Changes `test_css_readability.py:66`. | **JA** | Checkboxes, toggles and primary buttons stay Streamlit red; the frame is fixed by CSS only |
| 5 | Ink tints for small coloured text (GREEN_INK `#45b36b`, RED_INK `#e36464`, PURPLE_INK `#c35cff`)? Base colours stay for fills and borders. | **JA** | Small green and red text stays at 3.4–4.7:1 |
| 6 | I DAG: "Dagens besked" (labelled counts plus 4 tap tiles) above the PR 8 order? | **JA** | Page starts at VÄNDNING as today, in the new styling |
| 7 | Morgonrundan I DAG → Läget → Thor → 🎯 Scorecard → Positioner (navigation only), with Scorecard as the ARKEN step? | **JA** | No pill or footer; pages unchanged |
| 8 | ARKEN: verdict (and "Registrera beslut") at the top of each card, verdict word coloured in the head, Scorecard sorted by gates passed? | **JA** | Keyed expanders and styling only; order and placement as today |
| 9 | Thor: in analysis mode, skip the list chrome above the verdict; subtitle and footnote into ⓘ; Screening and Segment into ⚙ Filter; markets into 🌍? | **JA** | Analysis drawn below the list header as today |
| 10 | Positioner: one card per holding, Swedish signal words with the stored word as a source chip, ✕ behind a confirmation? | **JA** | Old 7-column rows, restyled |
| 11 | Läget: ONE 📡 that runs both reloads; Swing Regime's own header and button hidden when embedded? | **JA** | Two refresh buttons stay |
| 12 | Is `ovtlyr/ui/rules_page.py` **not** frozen like the Viking files? (Unlocks hard rules first, Kom igång order, "om 1 dag", the "Fyra strategier" count; texts verbatim.) | Please confirm; rec. **JA, not frozen** | Shell CSS only, forever |
| 13 | Keep Source Sans for long body and rule text (Space Grotesk for UI, JetBrains Mono for numbers), and drop the dead Inter Tight import? | **JA** | Space Grotesk everywhere (reads worse for long Swedish text) |
| 14 | Login copy "Åtkomstkod", "LOGGA IN", "Fel kod — försök igen.", "Inloggningen sparas i 30 dagar på den här enheten."? | **JA** | English copy kept, new layout |
| 15 | Later (backlog): Positioner Risk/Rapporter/Korrelation/Metaller and Larm 5→3 as `_inner` tabs; Swedish metric labels and sv-SE numbers on Positioner (`test_fix_panelen_korrekthet.py:121-123,132`)? | **JA, later** | Stays as today |
| 16 | Phone-only floating amber "osparat" snackbar (one at a time)? | **JA** | Brand-bar dirty chip only |
| 17 | Phase 2, only after a real-iPhone test: a fixed bottom dock instead of the sticky top bar? | **NEJ nu** | – |
| 18 | Aurora art as a faint background of "Dagens besked" (about 200 KB of base64 per I DAG run)? | **NEJ i v1** | – |

---

## 7. Phased PR plan (12 PRs, numbered after Design PR 12)

Every PR runs the full `pytest` and attaches 390 px and 1440 px screenshots of the touched screens.

| PR | Title | Files | What | Risk | Tests |
|---|---|---|---|---|---|
| **13** | Ramen: ett stilblock, varumärkesraden, tyst topp | `.streamlit/config.toml`, `ui/shell.py` (new), `ui/theme.py`, `ui/css.py`, `wolf_panel.py`, `tests/test_design_shell.py`, `tests/test_design_invariants.py` | See notes below | Low. Check in Safari that styles in a `display:none` container apply (verified in Chromium) | See notes below |
| **14** | Navigationen: sex flikar, sticky rad, chip-rader, rundgång | `ui/shell.py`, `wolf_panel.py`, `heatmap/heatmap_streamlit.py`, `tests/test_design_nav.py` | See notes below | Medium: CSS on Streamlit internals (`stRadioOption`, `data-selected`, `role=tablist`), locked by tests. Check sticky and `:has()` on a real iPhone; fallback is to drop the one sticky rule | See notes below |
| **15** | Byggstenarna | `ui/tokens.py`, `ui/components.py`, `ui/theme.py`, `ui/shell.py`, `ui/charts.py`, `thor/cards.py`, `storage_ui.py`, `tests/test_design_components.py` | See notes below | Low to medium: every sheet header changes look at once; review screenshots per page | See notes below |
| **16** *(decisions 4 + 5)* | Cyan och bläck | `.streamlit/config.toml`, `ui/tokens.py`, `ui/shell.py`, `ui/css.py` (comment only), `tests/test_css_readability.py` | See notes below | Low technically; it reverses a recorded choice | **Edit** `tests/test_css_readability.py:66` and its comment to the exact new dict; re-run `test_pr6_design`, `test_css_tabs` |
| **17** | Inloggningen och låset | `auth.py`, `portfolio/private_lock.py`, `wolf_panel.py`, `tests/test_design_login.py` | Login and lock forms (§4.1, §4.9); `render_shell()` runs before `render_login_gate()`; auth CSS reduced to `.arc-login-*`. Cookie, bridge, `?wolf_t=` and `check()` unchanged | Low. Verify the bridge reload once on Streamlit Cloud | Re-run `test_auth_remember`, `test_private_holdings`. New: AppTest Enter submit, wrong code text, 16 px field CSS present |
| **18** | I DAG: Dagens besked och Morgonrundan | `tabs/home.py`, `tabs/home_today.py`, `ui/routine.py` (new), `tests/test_design_home_counts.py`, `tests/test_design_morning.py` | §4.3; pure `ht.counts()` and `ht.size_parts()`; Gist error printed once; `routine.mark/pill/footer`. Data load path untouched | Low to medium | Kept: `test_design_i_dag`; `test_pr8` ('st.session_state["nav_goto"] = target' and 'Öppna →' in `tabs/home.py`); `test_pr6` (`home._next_step`). New: counts equal their sources; Saknas on a Gist failure; every routine step is in `TARGETS`; reset on a new date |
| **19** | Läget och rundans steg | `odin/laget.py`, `wolf_regime_ui.py` (`embedded=False` default), `thor/ui.py`, `scorecard.py`, `holdings.py`, `wolf_panel.py`, `tests/test_design_morning.py`, `tests/test_design_laget.py` (extend) | §4.4: one 📡, light grid, embedded regime card, single bullets, escaped '>', sparkline. Pill and footer calls on Läget, Thor, Scorecard and Positioner (Positioner after the gate) | Low | New: the caption renders a literal "> 55 % friskt"; the motor page is unchanged when `embedded=False` |
| **20** | Thor: listan och analysen beslut först | `thor/ui.py`, `thor/cards.py`, `tests/test_design_components.py` | §4.5, §4.6: tap rows, ⚙ Filter and 🌍 popovers, ⓘ, freshness below the list, errors once, analysis mode skips the list chrome, `storlek_grid`, snapshot strip, desktop columns. Draw order changes, logic does not | Medium (most-used page) | No edits expected. Run `test_thor_ui`, `test_thor_watch`, `test_thor_lage_historik_bors`, `test_thor_rattelser_verifiering`, `test_thor_segment_top`, `test_thor_granskning3`, `test_thor_scorecard_entry`, `test_design_storlek`, `test_design_vocab`. New: `storlek_grid` shows exactly the `sz` values plus `row_text` |
| **21** | ARKEN: domen överst i kortet | `review/card.py` (`fill(..., verdict_html=None)`), `scorecard.py`, `producers.py`, `wolf_panel.py` (fiat expander wrapper), `tests/test_design_arken.py` | §4.7, §4.8: keyed expanders, coloured verdict words, strip, display sort, verdict and "Registrera beslut" in the top slot, `gate_meter`. `gate_state`, GATES and scoring untouched | Medium: widget order inside cards changes | Run `test_review_focus` (labels split on " ·", `markdown[0]`), `test_scorecard*`, `test_arken_dirty_on_open`, `test_thor_scorecard_entry`, `test_thor_candidates`, `test_automatisering_batch13.py:297` (startswith "FNV · "), `batch14:449,1191`, `batch4:200`. Colour is appended only after the existing prefix; any assertion that needs a contiguous "· Köpkandidat" is listed in the PR before it is touched. New: sort uses `gate_state` only; heads start with the ticker; expanders are keyed |
| **22** | Positioner: åtgärd i dag och ett kort per innehav | `holdings.py`, `wolf_panel.py`, `tests/test_design_holding_card.py`, `tests/test_pr6_design.py` | §4.9: action-strip slot, `holding_card`, ✕ behind `confirm_delete`, one save/lock row, `kpis_holdings`, escaped '>'. Labels and number formats unchanged | Medium | **Edit** `test_pr6_design.test_deletes_sit_behind_a_confirmation`: add `"holdings.py"` to its path list. Kept: `test_fix_panelen_korrekthet:121-129` ("Invested", "Portfolio Total", '{:,.0f} SEK', ">10% position", "12000.00"), `test_automatisering_batch16:326-331`, `test_private_holdings`, `test_positions`. New: HOLDING vocab mapping plus source chip |
| **23** | Regler & Guider med bara skal-CSS | `ui/shell.py`, `wolf_panel.py` (`nav_rules_page` container and crumb), `tests/test_design_nav.py` | §4.10 above the "after decision 12" line: chip rails for `rules_pb_*`, `sg_pick`, `rules_ref_view`; underline `rules_sub`; hidden `<hr>`; scoped h2 restyle; table overflow; 12 px floor. **No edit in `ovtlyr/`** | Low (attribute selectors may miss a few nodes) | Kept: `test_pr8` (`rules_page` untouched). New: selectors cover every rules radio key; no `ovtlyr/` file in the diff |
| **24** | Tabeller på telefonen | `ui/tables.py` (new), `portfolio/watch_ui.py`, `portfolio/mot_malet.py`, `ui/settings.py` (rules as cards from `rules_rows()`), `tests/test_design_tables.py` | §3.5.15 applied to Övervakning (keep "Avstånd" in the subset), Mot målet ("—" for None) and Inställningar "Reglerna (låsta)" (one card per strategy, full rule text, no 2,276 px scroll) | Low to medium | Run `test_watch_ui`, `test_thor_watch`, the settings and goal tests; the subset includes every column a test reads. New: no "None" rendered; the toggle shows the full df |

**Notes for PR 13 (Ramen)**
- What:
  - `[client] toolbarMode="viewer"` (not "minimal": in 1.59.1 minimal draws no toolbar locally — no ⋮ and no running indicator for the PR 14 beam; viewer still has no Deploy and no developer menu). Cloud host buttons (`stToolbarActions`) hidden by CSS on desktop.
  - `render_shell()` writes one `<style>` with the PWA metas folded in, inside the hidden `wolf_shell` container.
  - The brand bar slot plus `fill_bar()` (puls from the session only, dirty chip, lock glyph, single snackbar).
  - Header hidden on ≤640 px; transparent with a floating ⋮ on desktop.
  - Frame gap 0; footer year from `datetime.now()`; "▲ Till toppen".
  - The Inter Tight import dropped; `_banner_b64` kept.
- Tests kept green: `test_pr2` (`_banner_b64`, the `st.tabs` line), `test_css_readability` (toolbarMode is a valid key; theme unchanged), `test_css_tabs` (CYBERPUNK_CSS unchanged), `test_pr6` strings.
- New tests:
  - one `<style>` per run;
  - `wolf_shell` hidden;
  - the brand bar contains the `_SVG_ICON` polygon and "NORDIC", and no base64;
  - the puls never calls a loader (monkeypatched `load()` raises);
  - the footer year is dynamic;
  - invariants: literal `nav.TOP`, `SUBS`, `STATE_KEY` and `TARGETS` keys, plus sha256 of `strategy_rules*.py` and the frozen Viking files. The hashes are updated only when the owner changes a rule.

**Notes for PR 14 (Navigationen)**
- What:
  - Generated tab labels with the Safari fallback; grey inactive emoji; sticky tablist; scan beam.
  - L2 and L3 rails generated from `STATE_KEY` depth; inner tabs wrap.
  - `nav_js(route)`; `_crumb()`; the 9 `st.markdown("---")` after `_sub`/`_inner` removed.
  - Heatmap's `_inject_css` scoped through `page_style()`.
  - No widget, key or label changes.
- Tests kept green: `test_pr8` (no literal `key="sub_`; `at.radio(key="sub_motor")`), `test_watch_ui`, `test_css_tabs`.
- New tests:
  - the generated CSS follows `nav.TOP` order;
  - each tab has the plain `content` declaration before the alt one;
  - every `STATE_KEY` value is covered, with L2 and L3 split by depth;
  - the script body has no `<`;
  - no `---` follows `_sub(`.

**Notes for PR 15 (Byggstenarna)**
- What:
  - tokens: MUTED, NOTE moved and re-exported, scales, `ink()` returning the base colour for now, `tint()`;
  - components: `page_header` restyle with the same signature and default `align="left"`, `head_row`, `section`, `verdict_strip`, `tile`, `light_grid`, `kpi_strip`, `tap_card`, `chip`, `action_row`, `link_rows`, `missing`, `skeleton`, and the `empty_state`/`card` restyle;
  - `thor/cards`: 12 px floors;
  - `save_bar`: one horizontal row, same logic;
  - buttons: `:disabled` and `:active`; the `kpis_*` metric restyle.
- Tests kept green:
  - `test_pr6` (`page_header(` call sites; `kpi`, `badge` and `verdict_box` output texts; the secondary selector string);
  - `test_watch_ui` (title in `st.markdown`);
  - `test_fix_panelen_lagring`, `test_positions`, `test_normalisering_och_plan:149,267`.
- New tests: tile sign colours, escaping, `verdict_strip` uses only the words passed in, `tap_card` keeps its key, label and callback (AppTest), the Norden display rule leaves `row_text` unchanged.
- As built (PR 15):
  - `head_row`: the title is `flex: 1 1 auto; min-width: min(180px, 100%)`, not `1 1 180px` — with 180 px the button stayed beside a long title and the title broke between its emoji and word ("🥇" over "RÅVARUHÄVSTÅNG" at 390 px); now the button wraps right-aligned instead.
  - `action_row`: the chip column is 84 px, not 76 — "● VAKSAM" measures 81 px at 12 px.
  - The scoped cyan primaries (`login_btn`, `morning_next_*`) are left to PR 17/18: the login does not load the shell yet, and both arrive after PR 16's cyan primary.
  - `ui/css.py` is not in this PR: its `.section-title` class gets the `section()` look from the shell CSS; `tabs/home._block` moves with PR 18.
  - `ui/charts.sparkline_svg(series, colour)` is added here (PR 19 uses it).

**Notes for PR 16 (Cyan och bläck)**
- Config:

```toml
[theme]
base = "dark"
primaryColor = "#00E5FF"
backgroundColor = "#05070A"
redColor = "#c44545"
orangeColor = "#d4943a"
greenColor = "#2d8a4e"
violetColor = "#B400FF"
redTextColor = "#e36464"
greenTextColor = "#45b36b"
violetTextColor = "#c35cff"
```

- CSS: `button[kind=primary]` gets `#05070A` text; dirty save is amber; confirm delete is red; the now-redundant scoped cyan rules are removed.
- `ink()` switches to the ink tints.
- Check that the dirty save still reads as "osparat" next to "⚠️ Du har osparade ändringar.".
- As built (PR 16):
  - Every `theme.*` key above exists in 1.59.1; `[browser]` and `[client]` are unchanged.
  - White on cyan measures 1.5:1 (not 1.2:1); `#05070A` on cyan is 13.1:1, and 6.4:1 on Streamlit's darker hover/focus cyan.
  - Primaries also take the secondary's shape (44 px, radius 10, Space Grotesk 14 px 600 .06em caps). A disabled primary looks like a disabled secondary, because Streamlit's own disabled primary (grey text on transparent) vanished at opacity .45.
  - Confirm delete is keyed on `[class*="__confirm"]` (the `confirm_delete` key suffix) plus `st-key-watch_del_` (Övervakning's "Ja, ta bort …"), not `[class*="confirm"]`: that would also turn the journal's "Confirm Import" (`import_confirm_btn`, not a delete) red. Hover is darker: amber `#c0842c` (dark text 6.3:1), red `#b03c3c` (white 5.9:1).
  - Widgets that Streamlit fills with the primary colour keep readable text: multiselect tags and the date picker's selected day get `#05070A` (white was 1.5:1); the number input's −/+ icon turns dark on its cyan hover; CYBERPUNK_CSS's cyan box behind the slider value is removed, because the value text is now cyan.
  - No scoped cyan primaries existed to remove: `login_btn` and `morning_next_*` arrive in PR 17/18 and inherit the theme.

**Notes for PR 19 (Läget och rundans steg)**
- As built (PR 19):
  - `render_wolf_regime_page(embedded=False)` renders byte-identically (a test fingerprints the element tree). The motor page MER › Motorsidor › Wolf Regime is `tabs/regime.tab_regime`, which never called this function, so it is unchanged by construction.
  - `embedded=True` returns the regime dict; Läget draws "Kolla en ticker" last, after the Marknadsrisk row (the §4.4 order), via `_ticker_check(d, embedded=True)`.
  - The section reads "SWING-LÄGET" with "uppdaterad 20 nov 18:05" as its mono count on the right. Evidence numbers use the existing `_sv_pct` ("3,1 %", "48,0 %"); a missing index or breadth shows the Saknas chip instead of a silent "0.0%". The sparkline is NOTE-grey (a price series: no sign colour, and cyan means interaction).
  - On Thor, Scorecard and Positioner the head row exists only while the round shows, so a normal visit is drawn as before. During the round the subtitle moves under the head row (`head_row(subtitle=…)`), so the pill stays beside the title on the phone.
  - Positioner: the pill only when unlocked (it sits in the page); the closing footer "✓ Morgonrundan klar · Till I DAG ↑" also shows under the lock card — navigation only, no holdings — so a locked Positioner can still end the round (otherwise the pill and footer stayed on Läget, Thor and Scorecard all day).
  - The Marknadsrisk row: the toggle's label fills the 56 px card, so the whole card is the tap target (§2.4). The breadth card's label is "Marknadsbredd" (§4.4); "Andel över MA200" leads its caption.
  - `ui/routine.CSS`: the footer's markdown gets `margin-bottom:0` — without it the button covered the dots on every page except I DAG.
  - Open: `links.go("thor")` keeps an open `thor_ticker`, so the round can land in Thor's analysis (no pill there). The Marknadsrisk page inside `st.status` still has labels under 12 px (`market_risk_ui.py`).

**Notes for PR 20 (Thor: listan och analysen beslut först)**
- As built (PR 20):
  - The head row is always drawn in list mode: 🔨 THOR · "📡 Uppdatera" (`thor_refresh`) · ⓘ popover (`thor_about`, label "ⓘ" with help "Om listan", chevron hidden — also on `thor_about_an` in the analysis Läge block) holding the subtitle and the footnote verbatim · the round pill. With the chevron hidden, title + 📡 + ⓘ + pill fit one 44 px row at 390 px. The meta line is `head_row(meta=…)`.
  - 📡 now uses `on_click`: the callback sets `_thor_reload`, and the reload runs at the top of the next run, before anything is drawn. The same button (same key and label) therefore also works in the analysis, where it sits in the Läge block below.
  - The Läge strip is `cards.lage_strip()` = `kpi_strip(lage_items(...))` with the same labels, order, values and colours, wrapped in `role="group"` with an `aria-label` that keeps the source words verbatim. The tile stacks "32 · 13 KÖP" into "32" / "13 KÖP" and shows "1/2 GRÖN" as "1/2 gröna"; `test_thor_ui` reads "32 · 13 KÖP" (the aria-label carries it). A "—" (no Valhall map) shows the Saknas chip. On the phone (≤ 640 px) Thor's scoped CSS keeps each sub on one line with an ellipsis (the full text is in the aria-label), shows "32 · 13 KÖP" on one line and uses 8 px vertical padding: the strip is 78 px at 390 (96 px at 1440, where nothing is trimmed). The PR 13 tile with a sub line is about 78–82 px by itself, so 64 px is not reachable without changing the component.
  - Errors once: ONE amber line (`thor-warn`). When the Gist is unreadable it holds the Gist error verbatim (`errors["gist"]`); otherwise it reads "⚠ n körningar gamla/saknas/fel — se Körningar per strategi nedan". The empty state no longer repeats the error ("Listan är tom för att körningarna inte gick att läsa …"). An empty list after a failed run names the strategy ("Körningen fallerade för Wolf — felet står under …"). `_top_section` shows only the names, as chips. The details ("Kunde inte läsa …" and the failed runs grouped by error with the strategy names as chips) moved into the "Körningar per strategi" expander, which sits below the list. Its label turns amber when anything is stale; the rule is `.st-key-thor_fresh [data-testid="stExpander"] summary p … !important`, because the shell's expander colour in `ui/css.py` is itself `!important`. Viking Nine's market notes are not errors, so they stay visible: right after the list in list mode and right after the amber line in the analysis Läge block, once per page (`tests/test_design_thor.py` locks the place).
  - Search row (`thor_search`): `thor_ticker` is 48 px with 16 px mono text and the spec placeholder (no test read the old one), next to "⚙ Filter" (`thor_filter_pop`, label "⚙ Filter · n"). Screening and Segment show their labels inside the popover; keys and options are unchanged. "🌍 Marknad" (`thor_market_pop`) also shows "· n" when markets are picked. `_ticker_field()` gives the field the same key and arguments in both modes; Streamlit identifies `text_input` by key and `max_chars`, so the value survives the move.
  - Rows are `tap_card`s in `thor_rows` (and `thor_toprows`) with the keys, label "Öppna →", `_goto` and args unchanged. `candidate_card` HTML is unchanged; only its own 4/2 px margin is removed by CSS. At 1280 px and up the rows form a 2-column grid of equal-height rows.
  - Analysis: a nav row `thor_nav` holds `thor_back` (tertiary, keeping the label "◀ Tillbaka till listan" that `test_thor_ui` asserts; the spec says "‹ Kandidater") and the ticker field. Then `st.columns([5,7])` inside `thor_an`, with `thor_left` holding the decision and `thor_right` the evidence. Scoped CSS stacks them below 1024 px.
  - `test_thor_watch` requires `_watch_button(...)` between `verdict_for(` and `_storlek(st,` in the source. So `_analysis` creates an empty STORLEK box and then the `thor_actions` box, draws 👀 into the actions box first, and `_storlek(..., actions)` fills the STORLEK box and appends "📝 Logga köp" or "Köpgrinden …" to the actions box. On screen: verdict → STORLEK → 👀 → Logga köp.
  - `storlek_grid` puts the number in the tile and the unit in its sub ("1 500" over "kr · 1.5 %", "54" over "st · 6 000 kr"), because "1 500 kr" broke in two in a 114 px tile. "halverat" in `sz['formula']` adds a "halverat" sub. On the phone, tile padding and label tracking are tighter, so RISKSTORLEK fits on one line at 390 (at 320 the label still wraps; letter-spacing is 0 below 360 px). Numbers never break inside: `white-space: nowrap` with `font-size: clamp(14px, 5.1vw, 20px)`, so "110.00" is one line at 320 (16 px) and 390 (20 px).
  - Snapshot: `kpi_strip(cards.snapshot_items(snap))`, where only 1 mån and 3 mån carry a sign. Its tiles share the Läge strip's equal-height rule (all 80 px). With no price data it shows the Saknas chip and "— ingen kursdata för X", with the error text appended unless it says the same thing.
  - "Logga köp" now also shows `st.toast` next to the existing `st.success`/`st.warning` (§2.8).
  - 12 px floor: Thor's own inline sizes are 12 px or more. Captions are 13 px but stay `st.caption`, because `test_thor_live` reads `at.caption`. The tab modules' inline sizes (0.55–0.74 rem, 9–11 px) are raised to 12 px by attribute selectors scoped to `.st-key-thor_page`. `thor/regim.py`, the other tab modules and the frozen Viking files are not edited.
  - Buttons with `help=` (👀 Lägg i Övervakning) sit one wrapper deeper than the shell's `.stButton > button`: on desktop inside `stTooltipHoverTarget`, on touch in their own wrapper. They were 40 px and unstyled. Thor gives them the secondary look in scoped CSS; a global fix belongs in `ui/shell.py`. Combobox fields and number-input steppers inside `thor_page` are at least 44 px.
  - The "⚙ Filter" and "🌍 Marknad" popovers render their body (`stPopoverBody`) in a portal outside `stMain`, outside `.st-key-thor_page` and outside the trigger keys `thor_filter_pop` / `thor_market_pop`. Rules scoped to the page or to the trigger reach only the trigger button. Only the widgets' own key classes travel into the portal, so the 44 px rules for the popover contents start with those keys: `.st-key-thor_markets` pills, the `.st-key-thor_pick` / `.st-key-thor_segment` combobox fields and their ⌄ buttons, and Segment's help icon (`stTooltipHoverTarget` button). Before this PR the same widgets sat on the page at 40 px (field), 32 × 38 (⌄) and 32 px (pills). The 16 × 16 help icon is new, because Segment's label now shows inside the popover. `tests/test_design_thor.py` checks that the rules start with the widget keys. Inside the 🌍 Marknad popover the `thor_markets` pills wrap onto two rows (a Thor-scoped override of §3.5.4's nowrap row), so all six markets show without a swipe at 390 and 320; the shell rule and the `thor_filters` rail on the page are unchanged.
  - `ui/routine.advance`: when the next step is Thor and `thor_ticker` is set, it is set to "", so the round opens Thor in list mode. Navigation only; nothing is added when the key is absent.
  - Measured (fixture: test rows, BERSERK run stale, synthetic prices; iPhone 13 / 1440 × 1000):
    - §8.3: head row top y = 112 at 390 and 116 at 1440. Analysis nav row top y = 112.
    - §8.4, analysis: verdict bottom 472 at 390 (491 live with the Gist offline; 457 and 475 at 1440).
    - §8.4, list: first card bottom 578, second 716 at 390 (about 680 with all runs fresh). Open deviation for the owner: the cards are 125–131 px (the three-line detail in `candidate_card`, whose HTML this PR keeps), not 105 (+46 px), and the two-line amber line adds 44 px. Even a 64 px strip with no amber line puts card 2 at 666. The top of card 2 (ticker, strategy and verdict chips) is above the fold.
    - §8.5: 0 visible text nodes under 12 px in list or analysis, at 390 and 1440, fixture and live.
    - §8.7: no button, tab or summary under 44 × 44 at 390; all 20 tap-card overlays cover 100 %. Popover bodies mount only when opened, so the page sweep does not see them. They were measured separately at 390 and 1440 with each popover open. ⚙ Filter: the Screening and Segment fields are 44 px tall, both ⌄ buttons 44 × 44 and the Segment help icon 44 × 44. 🌍 Marknad: all six pills are 44 px tall and 50–85 px wide.
    - §8.8: stMain scrollWidth equals clientWidth at 390 and 320.
    - §8.15: "Gisten gick inte att läsa" appears exactly once with the Gist offline, in both modes.
    - Round: I DAG pill → Läget pill → Thor, starting from an open HEXA-B.ST analysis, lands at `?p=thor` in list mode with an empty field. The head row is one 44 px row: title, 📡 at x 148, ⓘ at x 273 and "3/5 ›" at x 325.
  - Open: "DATA UNAVAILABLE" / "DATA_GAP" still appear inside the tab modules (`thor/regim.py`, `thor/backtest.py`, `thor/asymmetri.py`). `test_thor_regim` and `test_thor_backtest` assert them, so they belong to a later PR.

**Notes for PR 21 (ARKEN: domen överst i kortet)**
- As built (PR 21):
  - **Scorecard head row, always:** 🎯 page title "SCORECARD" · "⬇ CSV" (`csv_scorecard`, same label and rows; the entries are now collected before the head) · the round pill. The subtitle is verbatim under the row (`head_row(subtitle=…)`). With no candidates the row holds the title and pill only, as before (no CSV). During the round the row is one 44 px line at 390.
  - **Beslutsremsa** `KÖPGRINDEN · ANTAL`: "n KLARA FÖR KÖP" (GREEN when n > 0, else MUTED) · "n kandidater", sentence `NO_TRADE`. The two `st.metric` stay in `kpis_scorecard` as a compact pair (label and value on one line, 39 px) and stay visible on the phone.
  - **One `gate_state()` per card** feeds the strip, `st.metric`, the heads and the order. It runs on `scorecard.live_card(card, entry)`: the stored card (`_shown`) with the card's own widget values from the session (`sc_strat_`, `sc_post_`, `sc_g_*`), which are the values those widgets return later in the same run. It only reads. **Deviation:** before, the ✅ icon and "Klara för köp" lagged one run after a tick; with "n/7 kryss" in the head the lag showed on every tick (measured: head 3/7 while the meter showed 4/7). On a first draw `live_card == _shown`, so the counts equal the old `is_ready` count.
  - **Order** (`display_order`): ready first, then gates passed descending, then ticker (ties keep `collect`'s name order). Display only; the Fokus select and the CSV keep `collect`'s order.
  - **Heads. Deviation:** `test_review_focus.py:175` finds the card by `startswith("📋  BOL")` and `:223` needs "producenter" in the label. So the old icon plus two spaces stays first, then the ticker, and the strategies stay the raw keys joined by ", " as today: "📋  NEM · Newmont · producenter · 4/7 kryss", ready "✅  BOL · Boliden · producenter · :green[KLAR FÖR KÖP]". No colour below 7/7.
  - **Keyed expanders** `sc_card_<key>`, `pr_card_<id>`, `ro_card_<id>`. The key class sits on the `stLayoutWrapper` around `stExpander`. Verified in Chromium: an open card stays open when its label changes and when it moves in the order (AEM ticked from 1/7 to 6/7 moved above NEM and stayed open).
  - **Fokus suffix (review fix).** In Streamlit 1.59 a keyed expander gets a block id made from the key alone. The browser then keeps its own open/closed state and ignores later `expanded=` values. So picking a ticker in the Fokus row left that card closed, although the page reran with `expanded=True`. While a ticker is in focus, the key therefore gets the suffix `_fokus` (`sc_card_<key>_fokus`, `pr_card_<id>_fokus`, `ro_card_<id>_fokus`). That remounts the card, and the `expanded` default applies again. The `[class*="st-key-…_card_"]` CSS still matches. The child widgets keep their keys and values, because a keyed widget's id does not depend on its parent block.
    - Verified in Chromium at 390 on all three sheets with the fixture. Picking NEM (Scorecard, Rick Rule) or WPM (Royalty C) opens the card. A tick in the focused card changes its head from 4/7 to 5/7 and the card stays open. "Visa alla" shows the list with every card closed. Picking the same ticker again opens it again.
    - Scope: only a pick in the Fokus row on the same sheet was broken. A focus set from Thor or from another sheet already opened the card, because the browser drops the stored state of blocks that were not drawn in a run.
    - `tests/test_design_arken.py::test_picking_a_ticker_in_fokus_remounts_its_card_so_it_opens` locks the key change. AppTest shows only the proto, which says `expanded=True` even when the browser keeps the card closed.
  - **Card top:** `st.container(key="sc_top_<key>")` created first, holding `rc.slot()`; after `fill` the existing "Registrera beslut" (`sc_dec_<key>`, same click logic) is added to the same container, so it sits right under the verdict. `fill(..., verdict_html=…)` puts `gate_meter` + `card.verdict_box` (READY 20 px GREEN, or `NO_TRADE` 15 px red ink) + "x/y fält ifyllda" + "Saknas för köpbeslut" in ONE markdown, which stays `markdown[0]`. The old bottom box is gone; READY/NO_TRADE appear once per card.
  - **Missing gates twice, not three times (review fix).** The meter's "n/7 kryss · saknas: …" (§3.5.14) sits directly above the NO_TRADE box, so the box no longer repeats the gates as "•" lines; it carries the decision only. The existing "Saknas för köpbeslut: …" line (tests read it; it also lists field labels, capped at 8) stays. So a not-ready card lists its gates twice, as before this PR (top line + bottom box), now both in the top slot.
  - `gate_meter(checks, labels)`: 7 equal segments, 8 px high, 2 px gaps; `True` GREEN, `False` RED, anything else the grey hatch with "— Saknas" in the segment title. Text: "n/7 kryss · saknas: a, b" with the `gate_state()['missing']` strings. No thresholds.
  - Phone: Strategi takes the row; the two position fields sit side by side, bottom-aligned (the longer label wraps). Checkbox rows are 48 px over the full width, and the card body gap is 8 px.
  - **Desktop extras skipped** (strip and metrics inline with the Fokus row; `st.columns([5,7])` inside a card): they would change widget order inside the cards, which tests read. At 1440 the cards are full width.
  - **Rick Rule / Royalty C:** page container `pr_page`; head row `head_rule` / `head_royalty` with the title and the existing "⬇ CSV" (`csv_producers` / `csv_royalty`, same rows), moved up as on Scorecard. "Hämta nu" exists only per card (`pr_live_<id>`), so no page-level button was added. Without a fixed sheet (the "Ark" radio) the CSV stays under the radio.
  - **Strip** `BEDÖMNINGAR · ANTAL` from `producers.strip_words(ranked)`: counts of `r['verdict'].label` from `ranked_producers` / `ranked_royalty` ("ofullständig" when there is no score). The buy word always shows (0 in MUTED); the other words show when they occur, in the sheet's order. Rick Rule: Köpkandidat GREEN, Bevaka AMBER, Passa GREY (vocab NEJ), ofullständig MUTED. Royalty: `ROYALTY_COLOR`.
  - **Rick Rule heads:** the prefix "TICKER · råvara · n/5 p · " is unchanged; only the verdict word is wrapped (`:green[Köpkandidat]`, `:orange[Bevaka]`, `:gray[Passa]`); "ofullständig" stays plain; the " · x.xx× NAV" suffix is kept. **Royalty heads are byte-identical (deviation):** `test_automatisering_batch13.py:408,417,495,499,502` assert `_head(at).endswith(R_NEUTRAL / R_BUY)`, and a colour directive ends with "]".
  - **Top slot:** Rick Rule shows "4/5 Köpkandidat" (score in mono, word in the vocab colour; Passa has a GREY rule and a MUTED word) plus the verdict's own `why`. With no score it shows "–/5 Fyll i pris och kostnad", the old box's text. Royalty's `royalty_signal` box (label + why) moved from below the inputs to the top. Both are computed at the end of the card with this run's values; the in-flow boxes are removed.
  - The scoring note "Marginal ≥ 40 % ger 2 p …" stays verbatim as `st.caption`; captions on these pages are 13 px NOTE. Rick Rule's intro paragraph went from DIM 0.78 rem to MUTED 13 px.
  - **(Q) two inputs per row on the phone: tried and dropped.** Next to the 44 px − and + the number had about 65 px ("6000." instead of 6000.00, "12.0(").
  - **Fiat:** `st.expander("🌍 Bakgrund — fiatmiljön (styr inga beslut)", expanded=False, key="arken_fiat")` wraps the same `render_fiat_environment()` call at the same place. The condition is shared, so Wolf Asymmetry and Råvaruhävstång get the wrapper too. Its 44 px summary rule is in the shell CSS (`ui/css.py`, one rule on the unique key), so the summary is 44 px on all four pages (measured at 390).
  - **CSS** (`review/card.CSS` via `page_style(…, "arken")`, keys only):
    - 8 px page gap; compact head-row CSV.
    - Strip without its own margin; words wrap between words, never inside them.
    - **Deviation:** strip words are 20 px at ≤ 640 px. "1 KLARA FÖR KÖP · 7 kandidater" is 362 px at 24 px and 325 px fit.
    - The Fokus row (same widgets) stays on one line on the phone (132 → 72 px).
    - Number-input steppers, combobox fields and ⌄ buttons are 44 px.
    - Help icons in checkbox rows are 44 × 44. Help icons in field and metric labels are 44 × 44 with a −14 px margin, so the label rows do not grow.
    - The help-wrapped "Hämta nu" / "Hämta EV nu" get the secondary look at 44 px (they were 40 px and unstyled, as on Thor).
    - 12 px floor for inline sizes under 12 px (`screens_ui`, `extract_ui`, `storage_ui`). Scorecard's own 0.72 rem band is now 12 px MUTED.
  - **Measured** with the fixture `tests/test_design_arken.STORES`: 7 candidates, BOL ready, NEM 4/7. Rick Rule has Köpkandidat, Bevaka, Passa and ofullständig; Royalty has KÖPLÄGE and Neutral. Viewports: iPhone 13, 320 × 640, 1440 × 1000.
    - §8.3: Scorecard save row top 168, head row 220 at 390 (180 / 232 at 1440). Rick Rule and Royalty: fiat expander 168, save row 230, head row 282 at 390.
    - §8.4 Scorecard, metric pair visible: strip bottom 402; card heads 538–586, 596–644, 654–702. That is **2 full heads + 1 partial**: the third head ends 38 px below the fold.
    - §8.4 Scorecard, metric pair hidden on the phone (measured with injected CSS only): heads 499–547, 557–605, 615–663, so **3 full heads**; the fourth starts at 673.
    - §8.4 Rick Rule and Royalty: the strip is above the fold (369–460 and 369–437).
    - §8.5: 0 visible text nodes under 12 px at 390, 320 and 1440, also with a card open (the hidden tab `p`, font-size 0, is excluded).
    - §8.7 at 390: no button, summary or checkbox label under 44 px on the three pages, also with a card open. Card heads are 48 px. At 320 the only small targets are the shell's tab cells (38–44 px wide), which this PR does not touch.
    - §8.8: stMain scrollWidth equals clientWidth at 390 and 320 on all three pages.
    - After the review polish: the subtitle sits 7–8 px above the strip (was −1 to 0 px; `margin-bottom: −8px` on the `.wolf-note` / `.arken-intro` markdown), Rick Rule's intro 6 px above the scoring note (was −2 px). Scorecard heads at 390 are now 546–594, 604–652, 662–710 (strip 315–410, metric pair 418–457), so still 2 full + a sliver. At 1440 the metric value sits 12 px after its label (was ~500 px apart); the phone is unchanged.
    - After the Fokus fix (re-shot at 390, 320 and 1440): the list numbers above are unchanged. With a ticker in focus at 390, on all three sheets: the card head is 48 px, there is no text under 12 px and no target under 44 px, and scrollWidth equals clientWidth (390).
  - **Open:**
    - **Owner:** hide the Scorecard metric pair at ≤ 640 px? That gives the 3 heads of §8.4.
    - Rick Rule and Royalty heads, and the BEDÖMNINGAR strip, read the stored rows (`ranked_*` before the cards' widgets run). After an edit they are one run behind the fresh top-slot verdict until the next rerun (any tap, e.g. 💾 Spara). Measured: NEM ticked to 4/5 shows the head "Bevaka" right above the box "4/5 Köpkandidat"; the head lag existed before this PR (the box was ~1,100 px further down). Follow-up: a read-only live-row helper like `scorecard.live_card` that overlays the card's own widget values with the same write rules, feeding `ranked_producers`/`ranked_royalty` for heads and strip; the scoring functions stay untouched.
    - The first rerun after a page opens re-creates the whole tab content (`main_tabs` moves from root index 3/5 to 2/7 when the empty element before it disappears and the unsaved-changes elements appear after it). On origin/main that first tick collapses an open card; now the card stays open but the view can land away from it. Later ticks, also ones that move the card in the order, stay put. Shell fix for a later PR: constant placeholders around `st.tabs` in `wolf_panel.main` / `ui/shell.py`.
    - Help icons outside ARKEN need a shell-level rule.
    - "Registrera beslut" has no toast yet (§3.5.19).

**Notes for PR 22 (Positioner: åtgärd i dag och ett kort per innehav)**
- As built (PR 22):
  - **Save and lock on one row, CSS only.** `private_lock.gate()` draws "🔒 Lås innehaven" (`holdings_pw_lock`, label and `lock()` unchanged) before the page, and `storage_ui.save_bar` draws the save row right after it. `wolf_panel` keeps the literal `if gate():` (`test_private_holdings` and `test_design_morning` read it), so no container can be passed in. Instead `holdings.page_css()` pulls the lock into the save row's line: `.st-key-holdings_pw_lock:has(+ … > .st-key-save_holdings_row)` gets a negative bottom margin (44 px + the block gap) and sits right of 💾 Spara, and the text starts 8 px after it. Both buttons have fixed widths, Spara 92 px and Lås 148 px. At 13 px the labels measure 59 and 108 px, so a wider fallback font still fits; past that, the label ends in an ellipsis. `portfolio/private_lock.py` is not edited. Two fallbacks: if a load error is drawn between them, they stack as before. Under 390 px the two buttons share a row and the status text takes the next line (69 px at 320–389). **Review polish:** 💾 Spara is pinned to the row's top (`align-self: flex-start`) like the lock, so the two stay in line when the text wraps. After any save the text reads "Inga osparade ändringar · senast 16:40 UTC" (§3.5.17); before the polish that wrapped to 3–5 lines beside the buttons and Spara dropped 3–20 px below the lock at 360–390. Measured after it, clean and after a real save: 390 row 44 → 51 px (text 3 lines), 430 44 px, 375 and 360 69 px (text on its own line); Spara.top = Lås.top = 168 in every case.
  - **Head row, always:** `head_row("holdings", subtitle=…)` with `page_header("Positioner")` (gold, left; it replaces the 3.4 rem cyan `<h1>`) and the round pill. The pill still shows only during the round. The subtitle is verbatim.
  - **Beslutsremsa "BEHÖVER ÅTGÄRD I DAG"** (`holdings.strip_html`):
    - Sources: an `st.container()` slot under the head (in the strip column of `pos_top`) is filled after the strategy sections. The sections append each holding's existing `_signal_for_holding` dict and the rule monitor's own messages to a run-local list. Nothing is recomputed: `_signal_for_holding` runs once per holding, and a test makes every signal function raise while the strip is built.
    - **Why a container, not `st.empty()` (review fix):** an `st.empty()` slot blanks the 279–288 px strip at the start of every rerun and leaves it blank until the sections finish. Everything under it jumps up by the strip's height, then drops back. Measured at 390 with a regime load slowed to 0.5 s per holding: the strip was gone for about 1.6 s after tapping "Show Alpha Regime signals". Chromium's scroll anchoring hid it by moving `scrollTop` 920 → 632 → 920. With `overflow-anchor: none` (browsers without scroll anchoring; stable iOS Safari support is uncertain), the tapped checkbox jumped from y = 310 to y = 22. The container keeps the previous run's strip, stale, until the new one is written at the same index, so nothing moves. Re-measured after the fix, with every signal slowed to 0.4 s (a 3 s rerun): the strip stayed at 279 px and the KPIs at y = 590 throughout, and with `overflow-anchor: none` the tapped checkbox stayed at y = 310 with `scrollTop` unchanged. The steady layout is unchanged (§8 numbers below). With no holdings, Streamlit clears the stale child at the end of the run. The first open still fills late, as before. `test_strip_slot_is_a_container_so_a_rerun_keeps_the_old_strip_in_place` checks it.
    - Row order: SÄLJ (EXIT/SELL), then MINSKA (REDUCE), then VAKSAM (WATCH, `vocab.HOLDING`), in page order. The rule monitor's messages follow verbatim, as `action_row("REGEL", …)` with "QUALITY RULES" or "CONTRARIAN RULES" as detail and a jump to the rule label (`id="pos-rules-<strategy>"`).
    - **Deviation:** the monitor's one `st.error` ("🔴 n holdings — max is 10") is mirrored too, in red. Warnings are amber.
    - Escaping: the title is escaped, so "⚠ &gt;10% position" renders as ">".
    - Row anatomy: three rows show, the rest sit in a `<details>` "+n till ↓" inside the same markdown (no new widget). Each row's `data-jump` is the card id. Tapping SAND.ST scrolled its card to just under the sticky tab bar (`scroll-margin-top: 72px`). **Review polish:** the rule labels (`[id^="pos-rules-"]`) share that 72 px margin; before, a REGEL row landed the "QUALITY RULES" label under the tab bar (top 0). Now top 72 at 390.
    - Rule colour: the word's colour when the strip holds one word, otherwise ACCENT.
    - **Honest empty state (deviation):** the green "✓ Inga säljlägen i dag — alla innehav HÅLL" appears only when every holding has a HÅLL signal. If any signal is N/A, the strip says "✓ Inga säljlägen i dag" and adds a grey line: Saknas chip + "signal för n av m innehav: …". That line also appears under the rows. With no holdings, no strip is drawn.
  - **"HOLD ✓":** `vocab.main` does not know "✓", so `holdings.signal_main` strips it before calling `vocab.main(…, HOLDING)`. The source chip still shows "HOLD ✓" verbatim. `ui/vocab.py` is unchanged.
  - **KPIs:** the four `st.metric` (same labels and `{:,.0f} SEK`) sit in `kpis_holdings`, inside `st.columns([3,2])` in `pos_top`. The strip and the KPIs sit side by side from 1024 px and stack below that. The KPIs are 2 × 2 at every width, because they sit in the 2/5 column. Bug caught on the way: the columns rule (`gap: 12px`) also matched the KPI row and pushed it to one tile per row, so the selector now excludes `kpis_`. "Cash (SEK)" (`holdings_cash_input`) follows at full width. The KPI "Cash" still reads the cash at run start, so it lags one run behind an edit, as before.
  - **Allocation:** one markdown with a `section()` label "STRATEGY ALLOCATION" and grid rows. Same numbers and formats; all text 12 px mono. The number columns are fixed (84/44/24 px on the phone, 150/96/52/28 px from 641 px), so the rows line up. "DEEP CONTRARIAN" wraps onto two lines on the phone.
  - **Strategy header:** 4 px bar in `STRATEGY_COLORS`, the name at 17 px in TEXT, and on the right "3 holdings · 151,302 SEK · 63.1%". The right side reuses the allocation bars' numbers (passed in as `alloc`) and keeps the English "holding(s)" (decision 15 is later). The name is not in the strategy colour, because Viking purple and Wolf green under 19 px fail contrast; the bar carries the colour. The "➕ Add …" expander is unchanged; its label is "Add Quality holding", as today.
  - **Holding card** (`_holding_card`, name kept): ONE `st.markdown` (`holding_card_html`) with `id="pos-<ticker>-<bucket>"`, for example `pos-volv-b-st-long`. The bucket is included because the same ticker can sit in two buckets.
    - L1: ticker 17 px, sector · n st, and the chip pair.
    - L2: mono 14 px as a wrapping flex row: price (day change), "värde …", "P/L … (…)". It wraps between parts, never inside one, and needs no separators. Only the day change and P/L are sign-coloured (ink tones); 0 is MUTED (before, 0.0 % was red).
    - L3: 12 px MUTED: the signal detail, then the regime part as the old row showed it. Quality shows `chip(verdict)` + "n/m signals"; Contrarian shows the short stage in its own colour + tranche dots. An Alpha Regime error or unavailable shows the Saknas chip.
    - **Deviation:** a missing price shows the Saknas chip instead of "0.00 (+0.0%)" (§2.7). Ticker and sector are escaped.
    - Measured heights: 114–137 px at 390, 137 px at 320.
    - **Review polish:** the separator dot hangs on the word before it ("EQX&nbsp;·" "Materials&nbsp;·" "1000 st", each nowrap), so a wrapped L1 never starts a line with "·". Before, "· 100 st" / "· Health Care · 10 st" started the wrapped line on every card at 320 and on two at 390. Re-measured: 0 at 320, 390, 768 and 1440.
  - **Desktop (≥ 1024 px), done:** the card's CSS grid lays the six old columns (TICKER · PRICE (1D%) · VALUE · P&L · SIGNAL · REGIME) on one 68 px line. One header markdown per section (`pos-cols`, `display:none` inline) shows from 1024 px. The old `st.columns` header row is removed; no test read it.
  - **⚙ Edit <ticker>:**
    - Keys: the expander is now keyed `pos_edit_<bucket>_<ticker>`; the label and every widget key are unchanged.
    - Look: a light 44 px row under the card; the border shows only when the expander is open.
    - Inside: Strategy (+ Tranches), then Stopp · Mål · Sektor, the stop and sector lines (`_stop_line`, `_sector_line` unchanged), then one row `pos_act_…`: [Save] [🗑 Ta bort]. The 28 px spacer before Save is gone.
    - Delete: `confirm_delete("Ta bort", key="rm_<bucket>_<ticker>")` replaces the ✕. Its "Ja, ta bort" (`rm_…__confirm`, red) calls the same `_remove_holding` + `st.rerun()`. The popover works inside the expander (AppTest and Chromium).
    - Grouping: each holding is a `posc_<bucket>_<ticker>` container (card + Edit, 4 px apart).
  - **Rule monitor:**
    - Same thresholds and texts. The label is a `section()` with an id.
    - `st.warning`/`st.error` stay. Their text escapes ">" as `\>`. Measured before this PR: Streamlit takes the leading "⚠" as the alert icon, the rest (">10% position…") became a blockquote, and the ">" was lost. `test_fix_panelen_korrekthet` still finds ">10% position" in `w.value`.
    - Scoped CSS gives the alerts a 4 px amber (or red) left rule and radius 12.
    - Tranches unchanged.
  - **Section footer:** same numbers and strings; labels 12 px MUTED, values mono, P/L in sign ink. It used to colour the value cyan or purple by strategy.
  - **Risk · Rapporter · Korrelation:** `<hr>` replaced by `section("Risk")`, `section("Rapporter")` and `section("Korrelation")`. The cyan "PORTFOLIO CORRELATION" div is replaced by the label (no test read it); the chart's own title is unchanged.
  - **Metaller:** in `wolf_panel`, `st.markdown("---")` became `section("Fysiska metaller")`. The routine footer stays after the gate.
  - **Page CSS:** `holdings.page_css()` via `page_style(…, "holdings")`, right after the save bar; the content is in `pos_page`.
    - **Review polish:** the subtitle's markdown gets `margin-bottom: -8px` (like PR 21 on ARKEN), so it sits ~10 px above the strip instead of 2–3 px. Everything under the head moves down 8 px (numbers below).
    - 12 px floor: inline sizes from 0.6 to 0.73 rem are raised to 12 px through attribute selectors under `page_scope("holdings")`. That scope also reaches risk, earnings, the broker box and metals.
    - Captions are 13 px.
    - 44 px targets: summaries, checkboxes, number steppers, comboboxes, the popover button and help icons. "Börsdatas sektor" (a help-wrapped button) gets the secondary look at 44 px (before: 40 px, unstyled).
  - **Measured** (scratch fixture: generic rows EQX, VOLV-B.ST, AZN.L, NEM, SAND.ST, ERIC-B.ST, ABB.ST; stubbed prices and signals with one EXIT, one REDUCE, one WATCH and the rest HOLD; EQX at 69.6 % of Quality; unlocked with a test HOLDINGS_PASSWORD set only in the scratch app's environment; iPhone 13, 320 × 640, 1440 × 1000):
    - §8.3: save/lock row top y = 168 at 390, 1 row of 44 px (168–212); 180 at 1440. Head row 228 (390), 240 (1440). During the round the head row is 44 px with "5/5 ›" on the right.
    - §8.4: strip 307–586 at 390 (subtitle ends 297). The first action row (SÄLJ SAND.ST) is at 334–394; the spec's layout allows ≈312. KPIs about 598–746, below the fold. With the empty strip (all HÅLL): strip 307–375, KPIs about 387–535, all above the fold. At 320 the first row is at 357–417. At 1440: strip from 324 beside the KPIs (first row 351–407).
    - §8.5: 0 visible HTML text nodes under 12 px at 390, 320 and 1440, also with ⚙ Edit open, the 🗑 popover open, "+3 till" open and Alpha Regime on. The hidden tab `p` (font-size 0) is excluded. The Plotly charts' own SVG tick and bar labels (10–11 px, unchanged chart code) are not counted.
    - §8.7 at 390: no button, summary or checkbox label under 44 px, also with Edit and the popover open ("Ja, ta bort" 116 × 44). At 320 the only small targets are the shell's tab cells (38–44 px wide), outside this PR.
    - §8.8: stMain scrollWidth equals clientWidth at 390, 320 and 1440.
    - §8.15: "⚠ >10% position: EQX (69.6%), …" renders with the ">" in the alert and in the strip.
  - **Open:**
    - Strip rows are 60–81 px, because the 16 px title "TICKER · Strategy · reason" wraps and the source chip follows it. The strip with 3 rows and "+n till" is 279 px, so the KPIs at 390 fall below the fold.
    - The Plotly mode bar covers the titles of the sector and correlation charts at 390. This was there before; `displayModeBar: False` is §3.5.15(5).
    - Risk and Rapporter keep their own h3 ("RISK DASHBOARD", "EARNINGS CALENDAR", gold 0.7 rem subtitle, now 12 px) under the new labels. `risk_dashboard.py` and `earnings_calendar.py` are outside this PR.
    - Metaller likewise keeps its own "#### 🥇 Fysiska metaller" under the new FYSISKA METALLER label, so the title shows twice; `portfolio/metals.py` is outside this PR. Handle Risk, Rapporter and Metaller together in a later PR.
    - Later PR: the merged `review/card.slot()` and the brand-bar slot (`render_header()` → `theme.fill_bar`) are also `st.empty()`, so they blank the same way on a slow rerun. Switching them to `st.container()` is a one-token change each, but it is outside this PR.

**Backlog, owner-gated, not in the 12 PRs:**
- decision 12 content (`rules_page` hard rules first; new `tests/test_design_rules_order.py` asserting every rule text is present exactly once);
- decision 15 (inner tabs; Swedish metrics, editing `test_fix_panelen_korrekthet.py:121-123,132`);
- decision 17 (bottom dock);
- the other pages' Beslutsremsor (Kartan, EMBER, Fiat, Allokering, Veckorutin, Larm).

---

## 8. Acceptance checks

Measure with a Playwright script (`scripts/design_check.py`, not part of CI). Run it against `streamlit run wolf_panel.py` with the audit fixtures, at **390 × 844 (fold 664)** and **1440 × 1000**, on: Login, I DAG, Läget, Thor list, Thor analysis, Scorecard, Rick Rule, Positioner, Handelsregler, plus one depth-3 page (ODIN › Råvaror › 🥇 Guld). Every item must also hold on a real iPhone (Safari) before PR 14 merges.

1. **All six tabs visible at 390 px.** The tablist `scrollWidth ≤ clientWidth`; every tab is fully inside x 16–374; each tab is at least 52 px wide and at least 44 px tall. They also fit at 320 px.
2. **The tab bar sticks.** After `stMain.scrollTop = 2000`, the tablist `getBoundingClientRect().top` is between 0 and 1.
3. **The chrome is small.** The top of the first page-content element (save row or head row):
   - y ≤ 112 on I DAG and Thor;
   - y ≤ 168 on depth-2 pages;
   - y ≤ 232 on depth-3 pages.
   Brand bar bottom to tablist top is at most 4 px.
4. **The decision is above the fold (664 px).** The bottom of each of these is at y ≤ 664:
   - Dagens besked tiles and KAPITALET grid (I DAG);
   - the light grid and the regime card's first line (Läget);
   - the first two candidate cards (Thor list);
   - the verdict block (Thor analysis);
   - the Beslutsremsa and the first 3 card heads (Scorecard);
   - the strip and the first action row (Positioner);
   - the RISKMODELL box top (Handelsregler).
5. **No text under 12 px.** Every visible text node with non-empty text has a computed `font-size` ≥ 12 px. Excluded: the hidden tab `p` (font-size 0, whose pseudo labels are 12 px), Plotly SVG internals, and the frozen Viking pages.
6. **No Streamlit red.**
   - After PR 16: no visible element has a computed `color`, `background-color`, `border-color`, `fill` or `box-shadow` containing `rgb(255, 75, 75)` / `rgba(255, 75, 75`.
   - Before PR 16: the same holds inside the frame, rails, login, head rows, tap cards and Morgonrundan buttons.
7. **Tap targets.** At ≤640 px, every visible `button`, `[role=tab]`, `[data-testid=stRadioOption]`, `summary` and `a.wolf-top` has a box of at least 44 × 44 px. Tap-card overlays cover 100% of their card.
8. **No horizontal page scroll.** `document.documentElement.scrollWidth ≤ 390` on every screen, including Flikguide. Only rails, strips and tables scroll sideways, each with a fade cue.
9. **Contrast.** Text under 19 px has at least 4.5:1 against its computed background; larger bold text has at least 3:1. Spot-check MUTED and the ink tints on BG_ALT.
10. **Deep links land visibly.** For every slug in `nav.TARGETS` plus `?p=portfolio/positioner` and `?p=more/regler-guider/handelsregler`: the right tab is `aria-selected`, its selected chip is fully inside its rail's visible box within 1 s, and the page opens at `scrollTop = 0`.
11. **Morgonrundan takes 4 taps.** From I DAG, tapping only "Nästa"/pill buttons reaches Positioner in 4 taps, each target inside the fold without scrolling; the state resets on a new date.
12. **Rules unchanged.**
    - `tests/test_design_invariants.py` passes: hashes of `strategy_rules*.py` and the frozen files, plus the literal nav structures.
    - `git diff --stat` across the redesign shows no change to `strategy_rules*.py`, `ovtlyr_nine.py`, `viking_execution.py`, `viking_exit.py`, `ovtlyr/` (until decision 12), or `nav.TOP`/`SUBS`/`STATE_KEY`/`TARGETS`/`LEGACY_*`/`INNER_SEEDS`.
13. **Tests are green.** The full `pytest` passes after each PR. The only edited lines are `test_css_readability.py:66` (PR 16) and the `holdings.py` entry in `test_pr6_design.test_deletes_sit_behind_a_confirmation` (PR 22).
14. **Login.** The field computes to 16 px (no iOS zoom on focus); Enter or "Gå" submits (AppTest `at.button(key="login_btn")` and a keyboard submit in Playwright); a wrong code shows "Fel kod — försök igen."; the 30-day cookie still restores the session.
15. **Render correctness.**
    - "> 55 % friskt" and "⚠ >10% position" render with the ">";
    - no "• •";
    - no "None" in tables on the touched pages;
    - the Gist error appears exactly once per page;
    - the footer shows the current year;
    - no element contains the text "DATA_GAP" or "DATA UNAVAILABLE".
16. **Frame mechanics.**
    - exactly one shell `<style>` per run;
    - `.st-key-wolf_shell` computes to `display:none`;
    - the nav script body contains no `<`;
    - the scan beam animates only while `stStatusWidgetRunningIcon` exists;
    - at most one `.wolf-snack` is ever visible;
    - `prefers-reduced-motion` stops all animation.
17. **Desktop sanity at 1440 px.** Brand bar 52 px with the ⋮ menu not overlapping the date or puls; the tab row is today's 16 px inline style and sticky; content ≤ 1280 px centred; the Thor analysis and Scorecard cards use the two-column decision-left, evidence-right layout.