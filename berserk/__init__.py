"""
berserk — ⚔️ BERSERK: contrarian swing i hela råvarumarknaden.

Köper råvaruaktier och råvaru-ETF:er när kortsiktig panik eller en hatad
råvarucykel har tryckt ner dem — men bara när råvaran själv ger tesen stöd.
Long-only, dagsdata, köp nästa dags öppning.

  themes.py    teman (olja, koppar, guld, lax, frakt …), komplex och varje temas
               drivare i preferensordning (termin först, ETF som reserv)
  universe.py  producentbolag (Norden först, sedan Nordamerika/London) och
               råvaru-ETF:er, var och en kopplad till sitt tema
  probe.py     datasonden: vilka drivare, ETF:er och aktier går att hämta, och
               hur lång historik de har (körs i Actions: berserk-probe.yml)

Setups, backtest och skanner kommer i PR 1–2. Temaindelningen följer Ember
(ember.config) där teman är gemensamma, så att komplexen säger samma sak.
"""
