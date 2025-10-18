"""
Single Agent Engine - A specialized workflow system based on MOA patterns.

This module provides a cognitive workflow system that uses specialized models
for different thinking functions: Response, Devils Advocate, Synthesis, and Final.

Key Features:
- Specialized cognitive functions for different reasoning tasks
- Layered processing for complex problem solving
- Dissent preservation and surfacing throughout workflow
- Self-review mechanisms for quality assurance
- Full tracing and observability
- Integration with existing MOA infrastructure

Workflow:
1. Response Agent: Initial analysis and reasoning
2. Self-Review: Critical self-assessment (optional)
3. Devils Advocate: Aggressive challenge and contrarian viewpoints
4. Synthesis Agent: Integration with dissent management
5. Layered Processing: Repeat for complex problems
6. Final Agent: Authoritative decision with uncertainty assessment

Usage:
    from single_agent_engine import SingleAgentWorkflow

    workflow = SingleAgentWorkflow()
    result = await workflow.run_workflow("Your prompt here")
"""

from .config import (
    SA_RESPONSE_AGENT_MODEL, SA_DEVILS_ADVOCATE_AGENT_MODEL,
    SA_SYNTHESIS_AGENT_MODEL, SA_FINAL_AGENT_MODEL,
    SA_NUM_LAYERS, SA_ENABLE_SELF_REVIEW,
    SA_OUTPUT_DIR, SA_TRACE_DIR, SA_DRY_RUN_DIR,
    SA_RESPONSE_MAX_TOKENS, SA_REVIEW_MAX_TOKENS,
    SA_DEVILS_ADVOCATE_MAX_TOKENS, SA_SYNTHESIS_MAX_TOKENS,
    SA_FINAL_MAX_TOKENS, logger
)
from .workflow import SingleAgentWorkflow
from .tracing import SingleAgentTracer
from .agents import SingleAgent, SingleAgentGenerationError, OpenRouterSingleAgent
from .main import main, run_single_agent_workflow

__version__ = "1.0.0"
__all__ = [
    # Core classes
    "SingleAgentWorkflow",
    "SingleAgentTracer",
    "SingleAgent",
    "OpenRouterSingleAgent",
    "SingleAgentGenerationError",

    # Main functions
    "main",
    "run_single_agent_workflow",

    # Configuration
    "SA_RESPONSE_AGENT_MODEL", "SA_DEVILS_ADVOCATE_AGENT_MODEL",
    "SA_SYNTHESIS_AGENT_MODEL", "SA_FINAL_AGENT_MODEL",
    "SA_NUM_LAYERS", "SA_ENABLE_SELF_REVIEW",
    "SA_OUTPUT_DIR", "SA_TRACE_DIR", "SA_DRY_RUN_DIR",
    "SA_RESPONSE_MAX_TOKENS", "SA_REVIEW_MAX_TOKENS",
    "SA_DEVILS_ADVOCATE_MAX_TOKENS", "SA_SYNTHESIS_MAX_TOKENS",
    "SA_FINAL_MAX_TOKENS", "logger"
]