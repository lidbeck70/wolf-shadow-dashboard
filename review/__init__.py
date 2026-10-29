"""
review — underlaget till köpbeslutet, ovanpå arken.

  catalog     varje inmatningsfält i arken, byggt ur arkens egna konstanter (inga kopierade trösklar)
  missing     vad som saknas för köpbeslut, härlett ur arkens egna grindar
  export      Excel/CSV per aktie och prompten till ChatGPT
  answers     ChatGPT-svaret läst (Excel, CSV, inklistrat, JSON) och varje värde tolkat i fältets typ
  units       enhetskontrollen: varna när svaret står i en annan enhet eller valuta — räkna aldrig om
  importer    svaret mot underlaget: avvisa, hoppa över, varna, visa skillnaden och skriv in det du valt
  chatgpt_ui  "📋 Underlag till ChatGPT" i arkens kort, i Scorecardet och i Thors Arken-flik (där bara läsning)
  focus       fokusläget: ett bolag i taget, Thors arkknappar öppnar arket på bolaget
  card        raden först i kortet: 'x/y fält ifyllda' och "Saknas för köpbeslut" (rubriken är oförändrad)

Arkens regler är facit. Ingenting här räknar en poäng eller en grind på nytt. ChatGPT fyller i fakta och förslag,
du granskar skillnaden och väljer vad som skrivs in i arket, och 💾 Spara sparar som vanligt — inget går till
GitHub av sig självt.
"""
