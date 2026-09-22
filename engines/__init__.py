"""
engines — analysmotorer med ett gemensamt returformat (engines.contract).

Varje motor (Durrett, senare Rick Rule, Graham, Buffett/Munger, Lynch,
Lukacs) behåller sina egna kriterier och returnerar EngineResult, så att
en framtida multi-model-vy kan läsa dem sida vid sida utan att blanda
filosofierna.
"""

from engines.contract import EngineResult, ENGINE_FIELDS

__all__ = ["EngineResult", "ENGINE_FIELDS"]
