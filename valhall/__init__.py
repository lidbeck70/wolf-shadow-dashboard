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
  picks.py         kvalitets- och hävstångsköpen (Quality Contrarian, Rick Rule, Durrett, Tiggre)
  backtest.py      fas 3: förbättrar cykelstatusen swingstrategierna? (affärer märkta på signaldagen)
  paper.py         fas 4: papperskonton per strategi, varje affär märkt med segment och cykelläge

Motorerna först. Panelen ändras först när hela systemet är klart.
"""
