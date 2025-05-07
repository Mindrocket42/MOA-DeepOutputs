import asyncio
import httpx
import time
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
        prev_synthesis = ""
        prev_devils_advocate = ""

        for layer_idx in range(self.num_layers):
            layer_start_time = time.time()
            layer_details = {}
            
            # Log layer start
            if self.tracer:
                self.tracer.log_layer_event(
                    layer_num=layer_idx + 1,
                    event_type="Start",
                    prompt=prompt,
                    metrics={"expected_tokens": INITIAL_MAX_TOKENS, "concurrent_requests": len(self.agents)}
                )
            
            # 1. Initial agent calls for this layer
            layer_prompt = build_layer_prompt(prompt, prev_synthesis, prev_devils_advocate, layer_idx)
            layer_details["layer_prompt_details"] = layer_prompt
            
            initial_responses = []
            for agent in self.agents:
                start_time = time.time()
                output = await agent.generate(layer_prompt, self.client, self.semaphore, INITIAL_MAX_TOKENS)
                duration = time.time() - start_time
                
                # Log API call
                if self.tracer:
                    self.tracer.log_api_call(
                        model=agent.model,
                        prompt=layer_prompt,
                        response=output,
                        duration=duration,
                        tokens_used=len(output.split())  # Approximate token count
                    )
                
                initial_responses.append((agent.name, output))
            layer_details["initial_responses"] = initial_responses
            
            # 2. Aggregation phase
            aggregation_responses = []
            for agent in self.agents:
                aggregation_prompt = build_aggregation_prompt(prompt, layer_prompt, [resp[1] for resp in initial_responses])
                start_time = time.time()
                output = await agent.generate(aggregation_prompt, self.client, self.semaphore, AGGREGATION_MAX_TOKENS)
                duration = time.time() - start_time
                
                # Log API call
                if self.tracer:
                    self.tracer.log_api_call(
                        model=agent.model,
                        prompt=aggregation_prompt,
                        response=output,
                        duration=duration,
                        tokens_used=len(output.split())  # Approximate token count
                    )
                
                aggregation_responses.append((agent.name, output))
            layer_details["aggregation_responses"] = aggregation_responses
            
            # 3. Synthesis phase
            synthesis_prompt = build_synthesis_prompt(prompt, layer_prompt, [resp[1] for resp in aggregation_responses])
            start_time = time.time()
            synthesis = await self.synthesis_agent.generate(synthesis_prompt, self.client, self.semaphore, SYNTHESIS_MAX_TOKENS)
            duration = time.time() - start_time
            
            # Log API call and check for synthesis overrides
            if self.tracer:
                self.tracer.log_api_call(
                    model=self.synthesis_agent.model,
                    prompt=synthesis_prompt,
                    response=synthesis,
                    duration=duration,
                    tokens_used=len(synthesis.split())  # Approximate token count
                )
                
                # Check for synthesis overrides
                for agent_name, response in initial_responses:
                    if SequenceMatcher(None, response, synthesis).ratio() < 0.5:  # Significant difference
                        self.tracer.log_decision_point(
                            decision_type="synthesis_override",
                            context={
                                "layer": layer_idx + 1,
                                "agent": agent_name,
                                "original_response": response,
                                "synthesis": synthesis
                            },
                            outcome="synthesis_modified_agent_response"
                        )
            
            layer_details["synthesis"] = synthesis
            prev_synthesis = synthesis
            
            # 4. Devil's Advocate phase
            devils_advocate_prompt = build_devils_advocate_prompt(prompt, layer_prompt, [resp[1] for resp in aggregation_responses])
            start_time = time.time()
            devils_advocate = await self.devils_advocate_agent.generate(devils_advocate_prompt, self.client, self.semaphore, DEVILS_ADVOCATE_MAX_TOKENS)
            duration = time.time() - start_time
            
            # Log API call and check for challenges
            if self.tracer:
                self.tracer.log_api_call(
                    model=self.devils_advocate_agent.model,
                    prompt=devils_advocate_prompt,
                    response=devils_advocate,
                    duration=duration,
                    tokens_used=len(devils_advocate.split())  # Approximate token count
                )
                
                # Check for devil's advocate challenges
                if "challenge" in devils_advocate.lower() or "issue" in devils_advocate.lower():
                    self.tracer.log_decision_point(
                        decision_type="devil_advocate_challenge",
                        context={
                            "layer": layer_idx + 1,
                            "synthesis": synthesis,
                            "challenge": devils_advocate
                        },
                        outcome="challenge_identified"
                    )
            
            layer_details["devils_advocate"] = devils_advocate
            prev_devils_advocate = devils_advocate
            
            # Log layer completion
            layer_duration = time.time() - layer_start_time
            if self.tracer:
                self.tracer.log_layer_event(
                    layer_num=layer_idx + 1,
                    event_type="End",
                    prompt=prompt,
                    metrics={"duration": layer_duration}
                )
                self.tracer.performance["layer_times"].append(layer_duration)
            
            layer_details["layer_number"] = layer_idx + 1
            layer_outputs.append(layer_details)

        # 5. Final decision phase
        final_prompt = build_final_prompt(prompt, layer_outputs)
        start_time = time.time()
        final_output = await self.final_agent.generate(final_prompt, self.client, self.semaphore, FINAL_MAX_TOKENS)
        duration = time.time() - start_time
        
        # Log final API call
        if self.tracer:
            self.tracer.log_api_call(
                model=self.final_agent.model,
                prompt=final_prompt,
                response=final_output,
                duration=duration,
                tokens_used=len(final_output.split())  # Approximate token count
            )
            
            # Log resource usage
            self.tracer.log_resource_usage(
                resource_type="api_calls",
                usage={
                    "total_calls": sum(self.tracer.metrics["api_calls"].values()),
                    "calls_per_model": dict(self.tracer.metrics["api_calls"]),
                    "total_tokens": self.tracer.performance["total_tokens"]
                }
            )

        return {
            "prompt": prompt,
            "layer_outputs": layer_outputs,
            "final_output": final_output
        }