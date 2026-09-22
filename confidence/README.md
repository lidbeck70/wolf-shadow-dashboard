# Confidence score

Poängmotor för gruv-, råvaru- och strategiska mineralbolag. Två tal som
aldrig blandas: **Case Score 0–100** (hur bra är caset) och **Confidence
Score 0–100** (hur säkra är vi på bedömningen). Fliken heter
"🧭 Confidence score" under GRANSKNING. Namnet WOLF används inte — det är
Arc-regimens poäng.

## Principer

- **Hittat värde eller null.** Saknad data ger 0 p och `DATA_MISSING`,
  aldrig ett gissat tal. Confidence sänks, Case ändras inte.
- **Varje poäng går att förklara.** Varje delpoäng får en rad med värde,
  datatyp (ACTUAL/ESTIMATE/GUIDANCE/MODELLED/ASSUMPTION), källa och datum.
- **Inga magiska multiplar.** Scenarier skriver ut varje steg; config-värden
  (skatt, EV/EBITDA, P/NAV) syns som ASSUMPTION och flaggas.
- **Confidence ändrar aldrig Case Score.** Hög Case + låg Confidence ≠
  high-confidence-case; rekommendationen kräver båda.
- **Motorn rör aldrig nätverket.** Why Now-signalerna hämtas i fliken och
  skickas in som data.

## Moduler

| Modul | Gör |
|---|---|
| `config.py` | ALLA trösklar, tabeller och fältkatalogen (`FIELDS`). SPEC = ur kravspecen, VAL = eget val med motivering. |
| `data/provenance.py` | `Datapoint` med kind, källa, källtyp, datum → freshness-/source-poäng. |
| `data/models.py` | `CompanyInput`, `PillarScore`, `CaseScore`, `ConfidenceScore`. |
| `data/validation.py` | Fel (stoppar fältet) skilt från saknat (DATA_MISSING). Stage-fält som inte gäller ignoreras. |
| `commodities.py` | 23 råvaror med crosswalk till rotation, blindspot-teman, commodity_ratios, ember. Sourcade värden läggs på som överlagring. |
| `scoring/` | En modul per Case-pelare + `confidence.py` (sju delar, kill-caps efter summan). |
| `scenarios/` | Bear/Base/Bull/Super Bull, asymmetri, 5×/10×-vägar, time-to-money. |
| `why_now.py` | Why Now 0–100 ur sex signaler (som data); täckning i stället för gissning. |
| `regional.py` | Regional knapphet 0–100: jurisdiktion (repots tabell) + koncentration + västligt gap. |
| `reports.py` | `analyze()`, Thesis Killer (≥ 5 risker), rekommendation, Investment Card, rapport i Markdown/JSON. |
| `prefill.py` | Förslag ur granskningsarken (AQS, DS, Rick Rule, Tiggre). |
| `store.py` | Formen på `data/confidence.json` (bara inmatningar). |
| `ui.py` | Fliken: Analys · Indata · Råvaror · Signaler. |

## Egna val (VAL) att granska

- Economics developers 4/4/2/3/2 (NPV/CapEx, IRR, payback, break-even, capex-intensitet).
- Stress utan FS-känslighet: MODELLED-bound (0,8 × pris ≤ break-even; NPV − 0,2 × CapEx ≤ 0). Negativt utfall → Economics max 5 p.
- Underskott 0–5 % får 4 p (specen saknar raden); varje justering ±1 p.
- Balance Sheet developers: gap 4, runway 3, åtaganden 3; DS ≥ 6/8 drar 1/2 p.
- Delar som inte gäller ett stage (royalty utan AISC, explorer utan CapEx) skalas pro rata — notes säger det.
- Confidence: Resource Certainty 8/3/3/2/2/1/1; Project Maturity interpolerar milstolpar inom specens intervall; Economic Certainty 6/5/4 med lägre tak för MODELLED; Financing/Timeline/Management enligt config.
- Scenarier: pris −30/0/+30/+80 %, Bear med capex +20 %; sannolikheter 25/50/20/5; skatt 25 %, EV/EBITDA 5×, P/NAV 0,7× som synliga ASSUMPTION.
- Why Now 30/25/15/10/10/10; under 50 % täckning ingen uppskalning. Regional: jurisdiktion ensam skalas aldrig upp.
- Rekommendation: BUY CANDIDATE kräver Case ≥ 80 och Confidence ≥ 70; CRITICAL-risk → REJECT.

## TODO / kända begränsningar

- Utbudsbalans, efterfrågetillväxt, geopolitisk knapphet, koncentration och västlig andel per råvara har ingen datakälla i repot — matas in med källa under Råvaror.
- Oberoende verifiering och källkonsistens (Data Quality) är manuella 0–5.
- Royaltybolag: Resource Certainty läser samma fält som gruvbolag (portföljnivå).
- Entry/stop/target sätts inte här (v1) — köpgrind och entry ligger i strategiflikarna.
- AI-kommentar i Investment Card är tom tills AI-lagret byggs (v2).
- Ingen headless-körning eller alert-leg ännu.

## Tester

`tests/test_confidence_*.py` (65 tester) med elva syntetiska bolag i
`tests/confidence_cases.py` — talen är påhittade testdata, inte verkliga bolag.
