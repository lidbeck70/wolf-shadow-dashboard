"""
engines/durrett — Don Durretts 10-stegsmetod som fristående motor.

Indata: confidence.data.models.CompanyInput (samma bolag som Confidence
score). Utdata: DurrettAnalysis + EngineResult (engines.contract).
Rena funktioner, ingen Streamlit, inget nätverk; alla trösklar i config.
"""

from engines.durrett.config import DURRETT_CONFIG
from engines.durrett.models import (Catalyst, Classification, DurrettAnalysis, RedFlag, ScenarioCase,
                                    Score, to_jsonable)

__all__ = ["DURRETT_CONFIG", "Catalyst", "Classification", "DurrettAnalysis", "RedFlag", "ScenarioCase",
           "Score", "to_jsonable"]
