"""
fiat_debasement — 🐺 Fiat Debasement: penningmängd, inflation, köpkraft och
fiatvalutor (SEK, EUR, USD) mot reala tillgångar.

  sources.py   hämtar råserier (FRED, ECB, Eurostat, SCB, Riksbanken, Yahoo,
               Börsdata) med källa, serie-id, frekvens, enhet och tidpunkt
  config.py    vilka serier som används, i preferensordning (primärkälla först)
  data.py      laddning med reserver, cache, färskhetskontroll och guld/silver-skarven
  engine.py    beräkningarna: YoY, CAGR, köpkraft, Monetary Gap, fiat mot tillgång
  snapshot.py  nyckeltalen per valuta, med källa och datum för varje siffra
  probe.py     datasonden: vilka serier går faktiskt att hämta (körs i Actions)
"""
