"""
valhall — cykelkartan över hela marknaden (contrarian, Rick Rule-tänk: behövda men hatade marknader).

Grunden är Odins Blindspot och 🪓 BERSERK Regime, generaliserade från råvaruteman till elva segment
(GICS-sektorerna) och deras råvaruteman. Varje segment får en ruta: fas, cykelläge, hat och status
GRÖN/GUL/RÖD — räknat dag för dag utan facit, så att varje regel kan backtestas.

  segments.py      de elva segmenten: SPDR-drivare (signal), UCITS-kandidater (handel utanför Norden
                   hos Nordnet), Börsdatas sektorer/branscher, nödvändighet och strategins nisch
  cycle.py         cykelmotorn: fas (BERSERK) + Blindspot (tioårspercentil, 200-veckorstrend, hat)
                   → status och cykelstart, som dagliga serier
  nordic_index.py  nordiska sektorindex (likaviktade) ur Börsdatas sektorindelning
  board.py         kartan idag: en rad per segment (global = SPDR, Norden = sektorindex) och råvarutema
  swing.py         swingrutorna (styrning A): strategin per segment — Viking Nine, BERSERK eller ingen
  wolf.py          Wolf på Kartan: Arc-Wolf och momentumlistan per segment, Arc-Wolfs köpsignaler till papperet
  picks.py         kvalitets- och hävstångsköpen (Quality Contrarian, Rick Rule, Durrett, Tiggre)
  backtest.py      fas 3: förbättrar cykelstatusen swingstrategierna? (affärer märkta på signaldagen)
  paper.py         fas 4a: papperskonton per strategi, varje affär märkt med segment och cykelläge
  longterm.py      fas 4b: hjälper cykeln de långsiktiga köpen? (segment och kvalitetsaktier, punkt för punkt)
  long_screener.py fas 4c: Long Screener (cagr, oförändrad poäng) med Valhalls cykel
  long_backtest.py fas 4c: Long Screener punkt för punkt 2008–idag (tre cykelvarianter, screenerns portfölj)

Fas 5: panelen — toppfliken ⚡ VALHALL (ui.py) läser de schemalagda körningarna ur Gisten.
"""
