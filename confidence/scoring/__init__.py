"""
confidence/scoring — Case Score (åtta pelare) och Confidence Score (sju
delar + kill-caps). Rena funktioner: CompanyInput (+ Commodity) in,
PillarScore/CaseScore/ConfidenceScore ut. Ingen Streamlit, inga globala
tillstånd; alla trösklar i confidence.config.
"""

from confidence.scoring.case import case_score, rating_for

__all__ = ["case_score", "rating_for"]
