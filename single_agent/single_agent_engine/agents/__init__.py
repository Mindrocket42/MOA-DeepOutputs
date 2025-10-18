"""
Single Agent implementations for different cognitive functions.

This module provides specialized agent implementations for the single agent workflow:
- Response Agent: Initial analysis and reasoning
- Devils Advocate Agent: Critical challenge and contrarian viewpoints
- Synthesis Agent: Integration with dissent management
- Final Agent: Authoritative decision making
"""

from .base import SingleAgent, SingleAgentGenerationError
from .openrouter import OpenRouterSingleAgent

__all__ = [
    "SingleAgent",
    "SingleAgentGenerationError",
    "OpenRouterSingleAgent"
]