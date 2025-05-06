import asyncio
import httpx
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher

from deepoutputs_engine.config import (
    AGENT1_MODEL, AGENT2_MODEL, AGENT3_MODEL, DEEP_RESEARCH_AGENT_MODEL,
    SYNTHESIS_AGENT_MODEL, DEVILS_ADVOCATE_AGENT_MODEL, FINAL_AGENT_MODEL,
    OPENROUTER_CONCURRENCY, OPENROUTER_API_KEY, API_TIMEOUT, API_RETRY_ATTEMPTS,
    API_INITIAL_BACKOFF, HTTP_REFERER, X_TITLE, INITIAL_MAX_TOKENS, AGGREGATION_MAX_TOKENS,
    SYNTHESIS_MAX_TOKENS, DEVILS_ADVOCATE_MAX_TOKENS, FINAL_MAX_TOKENS, logger
)
from deepoutputs_engine.agents.openrouter import OpenRouterAgent
from deepoutputs_engine.agents.base import AgentGenerationError
from deepoutputs_engine.utils import sanitize_for_markdown
from deepoutputs_engine.prompts import (
    build_layer_prompt, build_aggregation_prompt, build_synthesis_prompt,
    build_devils_advocate_prompt, build_final_prompt
)

try:
    from termcolor import colored
except ImportError:
    def colored(text, color):
        return text

class MixtureOfAgents:
    def __init__(self, models: List[str], num_layers: int = 2, include_deep_research: bool = True, tracer=None):
        self.include_deep_research = include_deep_research
        self.tracer = tracer
        if not models:
            raise ValueError("At least one agent model must be provided.")

        self.agents = [OpenRouterAgent(f"Agent {i+1}", model, f"Role {i+1}") for i, model in enumerate(models)]
        self.deep_research_agent = OpenRouterAgent("Deep Research Agent", DEEP_RESEARCH_AGENT_MODEL, "Deep Research Role") if self.include_deep_research else None
        self.num_layers = num_layers
        self.synthesis_agent = OpenRouterAgent("Synthesis Agent", SYNTHESIS_AGENT_MODEL, "Synthesizer Role")
        self.devils_advocate_agent = OpenRouterAgent("Devil's Advocate Agent", DEVILS_ADVOCATE_AGENT_MODEL, "Devil's Advocate Role")
        self.final_agent = OpenRouterAgent("Final Agent", FINAL_AGENT_MODEL, "Final Decision Role")

        logger.info(f"==== AGENT MODELS AFTER INITIALIZATION ====")
        for agent in self.agents:
            logger.info(f"  {agent.name} using model: {agent.model}")
        logger.info(f"  Synthesis agent using model: {self.synthesis_agent.model}")
        logger.info(f"  Devil's Advocate agent using model: {self.devils_advocate_agent.model}")
        logger.info(f"  Final agent using model: {self.final_agent.model}")
        if self.deep_research_agent:
            logger.info(f"  Deep Research Agent using model: {self.deep_research_agent.model}")
        else:
            logger.info("  Deep Research Agent disabled")
        logger.info(f"==========================================")

        self.client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": HTTP_REFERER,
                "X-Title": X_TITLE,
                "Content-Type": "application/json"
            },
            limits=httpx.Limits(max_connections=OPENROUTER_CONCURRENCY + 5, max_keepalive_connections=OPENROUTER_CONCURRENCY)
        )
        self.semaphore = asyncio.Semaphore(OPENROUTER_CONCURRENCY)
        logger.info(f"Initialized MixtureOfAgents with {len(self.agents)} agents, {self.num_layers} layers, and concurrency limit {OPENROUTER_CONCURRENCY}.")

    async def close_client(self):
        """Closes the shared httpx client."""
        await self.client.aclose()
        logger.info("HTTP client closed.")

    async def run_workflow(self, prompt: str) -> Dict[str, Any]:
        """
        Runs the full MoA workflow: initial agent calls, aggregation, synthesis, devil's advocate, final decision.
        Returns a dictionary with all outputs needed for report generation.
        """
        layer_outputs = []
        current_prompt = prompt
        agent_outputs = []

        # 1. Initial agent calls (Layer 1)
        for idx, agent in enumerate(self.agents):
            layer_prompt = build_layer_prompt(prompt, idx, config={})
            output = await agent.generate(layer_prompt, self.client, self.semaphore, INITIAL_MAX_TOKENS)
            agent_outputs.append(output)
        layer_outputs.append({"layer": 1, "agent_outputs": agent_outputs})

        # 2. Aggregation phase
        aggregation_prompt = build_aggregation_prompt(agent_outputs, config={})
        aggregation_output = await self.synthesis_agent.generate(aggregation_prompt, self.client, self.semaphore, AGGREGATION_MAX_TOKENS)
        layer_outputs.append({"layer": 2, "aggregation_output": aggregation_output})

        # 3. Synthesis phase
        synthesis_prompt = build_synthesis_prompt(aggregation_output, config={})
        synthesized_output = await self.synthesis_agent.generate(synthesis_prompt, self.client, self.semaphore, SYNTHESIS_MAX_TOKENS)
        layer_outputs.append({"layer": 3, "synthesized_output": synthesized_output})

        # 4. Devil's Advocate phase
        devils_advocate_prompt = build_devils_advocate_prompt(synthesized_output, config={})
        devils_advocate_output = await self.devils_advocate_agent.generate(devils_advocate_prompt, self.client, self.semaphore, DEVILS_ADVOCATE_MAX_TOKENS)
        layer_outputs.append({"layer": 4, "devils_advocate_output": devils_advocate_output})

        # 5. Final decision phase
        final_prompt = build_final_prompt(devils_advocate_output, config={})
        final_output = await self.final_agent.generate(final_prompt, self.client, self.semaphore, FINAL_MAX_TOKENS)
        layer_outputs.append({"layer": 5, "final_output": final_output})

        return {
            "prompt": prompt,
            "layer_outputs": layer_outputs,
            "final_output": final_output
        }