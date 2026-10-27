"""
universe/theme_members.py — de största bolagen MED kassaflöde per tema på Valhall-kartan.

Ur temaproben (universe/theme_probe.py, borsdata_probe PROBE_SECTION=themes, körd 2026-10-27): kassaflöde från
löpande verksamhet (Börsdata KPI 62) > 0, rangordnat på börsvärde (KPI 50) i USD, högst 20 per tema, en
notering per bolag (panelens om den fanns), bara marknader registret känner (Norden, USA, Kanada, London,
Australien). Teman med färre än fem bolag är tunna på riktigt — uran, litium, sällsynta jordartsmetaller,
platina/palladium, kaffe och kakao har få noterade producenter som tjänar pengar (resten prospekterar).

Listorna gör två saker: bolag utan tema i panelen får sitt tema (thor.ticker.theme_for, efter BERSERK och
EMBER — panelens egna teman ändras aldrig), och Thors temafilter visar bolag som hör till flera teman (ADM:
vete, majs, soja, socker). Inga regler ändras. Uppdatera genom att köra proben igen.
"""

from __future__ import annotations

from typing import Optional

PROBE_DATE = "2026-10-27"

THEME_MEMBERS: dict = {
    "olja": ("XOM", "CVX", "SHEL", "COP", "MPC", "VLO", "PSX", "CNQ.TO", "EQNR.OL", "SU.TO", "EOG", "SLB",
             "IMO.TO", "OXY", "CVE.TO", "FANG", "BKR"),
    "naturgas": ("LNG", "EQT", "EXE", "TOU.TO", "AR", "RRC", "SM", "CNX", "CRK", "CRGY", "PEY.TO", "GPOR",
                 "KEL.TO", "BIR.TO", "AAV.TO"),
    "naturgas_eu": ("EQNR.OL", "AKRBP.OL", "VAR.OL", "HBR.L", "DNO.OL", "ENOG.L", "BNOR.OL", "SQZ.L", "OKEA.OL",
                    "PNOR.OL"),
    "kol": ("GLEN.L", "CNR", "HCC", "ARLP", "BTU", "AMR", "NRP", "TGA.L", "HNRG", "NC"),
    "uran": ("CCJ", "BWXT", "PDN.TO", "UROY"),
    "tank": ("FRO.OL", "HAFNI.OL", "INSW", "CMBT", "TRMD-A.CO", "BWLPG.OL", "STNG", "DHT", "OET.OL", "TNK", "NAT",
             "TEN", "TK", "ASC", "PSHG"),
    "torrbulk": ("WAWI.OL", "CMBT", "SBLK", "2020.OL", "DFDS.CO", "GNK", "SB", "HSHP.OL", "PANL", "DSX", "EDRY",
                 "JIN.OL"),
    "koppar": ("BHP.L", "SCCO", "RIO.L", "FCX", "GLEN.L", "AAL.L", "ANTO.L", "TECK", "FM.TO", "LUMI.ST", "LUN.TO",
               "BOL.ST", "HBM", "CS.TO", "ERO"),
    "aluminium": ("RIO", "NHY.OL", "S32.L", "AA", "CSTM", "CENX", "GRNG.ST", "KALU"),
    "zink_nickel": ("GLEN.L", "VALE", "TECK", "LUN.TO", "BOL.ST", "HBM.TO", "ERO.TO", "TKO.TO", "OUT1V.HE", "NEXA",
                    "S.TO"),
    "stal": ("NUE", "MT", "STLD", "ATI", "RS", "PKX", "CRS", "TX", "SSAB-A.ST", "GGB", "CLF", "CMC", "OUT1V.HE",
             "ALLEI.ST"),
    "jarnmalm": ("BHP.L", "RIO.L", "VALE", "CLF", "LIF.TO", "CIA.TO", "MSB"),
    "litium": ("SQM", "ALB", "SGML"),
    "sallsynta": ("SQM", "ALB"),
    "guld": ("NEM", "AEM", "B", "WPM", "FNV", "AU", "GFI", "KGC", "FRES.L", "RGLD", "PAAS", "LUG.TO", "AGI",
             "EDV.TO"),
    "silver": ("WPM", "FRES.L", "PAAS", "CDE", "HL", "AG", "TFPM", "SSRM", "AYA.TO", "HOC.L", "FSM", "EXK", "SVM",
               "ASM"),
    "platina": ("VALT.L", "SBSW", "JMAT.L"),
    "palladium": ("VALT.L", "SBSW", "JMAT.L"),
    "godsel": ("NTR", "CF", "SQM", "YAR.OL", "CTVA", "ICL", "MOS", "UAN", "AGRO", "LXU", "FMC", "IPI"),
    "skog": ("IP", "UPM.HE", "WY", "SUZ", "SCA-B.ST", "STERV.HE", "LPX", "HOLM-B.ST", "MNDI.L", "UFPI", "RYN", "BCC",
             "BILL.ST", "SLVM", "CFP.TO", "METSB.HE", "IFP.TO", "CLW"),
    "vete": ("ADM", "BG", "CTVA", "INGR", "ANDE", "AGRO"),
    "majs": ("ADM", "BG", "DAR", "CTVA", "INGR", "ANDE", "REX", "AGRO", "GPRE", "LND", "ALTO"),
    "soja": ("ADM", "BG", "DAR", "CTVA", "INGR", "CALM", "ANDE", "AGRO", "GPRE", "LND"),
    "socker": ("ADM", "BG", "ABF.L", "TATE.L", "AGRO"),
    "kaffe": ("SBUX", "KDP", "SJM", "JVA"),
    "kakao": ("MDLZ", "HSY", "AAK.ST"),
    "lax": ("MOWI.OL", "SALM.OL", "BAKKA.OL", "LSG.OL", "AUSS.OL", "MAS.OL", "ISLAX.OL"),
}

# Bolag i flera teman får sitt huvudtema här; övriga får det första temat de står under (kartans ordning).
PRIMARY: dict = {"AGRO": "socker", "CTVA": "majs", "INGR": "majs", "CMBT": "torrbulk", "GLEN.L": "koppar",
                 "VALT.L": "platina", "JMAT.L": "platina", "SQM": "litium", "ALB": "litium"}


def members(theme: str) -> tuple:
    """Temats bolag (Yahoo-form), störst först."""
    return THEME_MEMBERS.get(str(theme or ""), ())


def themes_of(ticker) -> tuple:
    """Alla teman bolaget står under, i kartans ordning."""
    t = str(ticker or "").strip().upper()
    return tuple(th for th, ts in THEME_MEMBERS.items() if t in ts)


def theme_of(ticker) -> Optional[str]:
    """Huvudtemat för ett bolag i listorna, eller None."""
    t = str(ticker or "").strip().upper()
    if t in PRIMARY:
        return PRIMARY[t]
    hit = themes_of(t)
    return hit[0] if hit else None


def tickers() -> list:
    """Alla bolag i listorna, en gång var."""
    return list(dict.fromkeys(t for ts in THEME_MEMBERS.values() for t in ts))
