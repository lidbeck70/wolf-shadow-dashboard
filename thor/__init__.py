"""
thor — 🔨 THOR: alla strategiers kandidater på ett ställe, och allt panelen vet om en aktie.

Designen står i docs/thor_design.md. Thor räknar inga nya signaler: den läser strategiernas egna utfall ur de
schemalagda körningarnas filer (Gisten) och arken, normaliserar dem till en gemensam rangskala (Thor-rank)
för sortering, och sätter ihop analysen per aktie av de motorer som redan finns. Inga regler ändras.

  registry.py    strategiregistret: etikett, familj, källa, rang-normalisering, playbook, regimlayout, papperskonto
  candidates.py  kandidatlistan: Candidate ur varje källa, filter, sortering och sammanfattning (ren logik)
  ticker.py      identiteten: Yahoo-form, Börsdata, segment, hemmaindex, sektor-ETF, råvarutema med källa
  dossier.py     allt om en aktie: kandidatrader, ögonblicksbild, cykel med playbookens grind, arken (läsning)
"""
