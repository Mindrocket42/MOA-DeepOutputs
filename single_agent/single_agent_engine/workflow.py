"""
Single Agent Workflow Orchestrator.

This module provides the SingleAgentWorkflow class that orchestrates the
complete single agent workflow: Response → Self-Review → Devils Advocate → Synthesis → Final.

Key features:
- Layered processing for complex problems
- Context accumulation across phases
- Dissent preservation and surfacing
- Comprehensive tracing and metrics
- Error handling and recovery
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
import httpx

from .agents import OpenRouterSingleAgent
from .prompts import (
    build_response_prompt, build_self_review_prompt,
    build_devils_advocate_prompt, build_synthesis_prompt, build_final_prompt
)
from .config import (
    SA_RESPONSE_AGENT_MODEL, SA_DEVILS_ADVOCATE_AGENT_MODEL,
    SA_SYNTHESIS_AGENT_MODEL, SA_FINAL_AGENT_MODEL,
    SA_NUM_LAYERS, SA_ENABLE_SELF_REVIEW,
    SA_RESPONSE_MAX_TOKENS, SA_REVIEW_MAX_TOKENS,
    SA_DEVILS_ADVOCATE_MAX_TOKENS, SA_SYNTHESIS_MAX_TOKENS,
    SA_FINAL_MAX_TOKENS, OPENROUTER_CONCURRENCY, logger
)
from .utils import count_tokens_approximate

try:
    from termcolor import colored
except ImportError:
    def colored(text, color):
        return text

class SingleAgentWorkflow:
    """
    Orchestrator for the single agent workflow with specialized cognitive functions.

    This class manages the complete workflow from initial response through final synthesis,
    using different models for different cognitive functions.
    """

    def __init__(self, tracer=None):
        """
        Initialize the single agent workflow.

        Args:
            tracer: Optional tracer for logging and metrics
        """
        self.tracer = tracer

        # Initialize agents for each cognitive function
        self.response_agent = OpenRouterSingleAgent(
            name="Response Agent",
            model=SA_RESPONSE_AGENT_MODEL,
            role="Deep Analysis and Reasoning",
            cognitive_function="response"
        )

        self.devils_advocate_agent = OpenRouterSingleAgent(
            name="Devils Advocate Agent",
            model=SA_DEVILS_ADVOCATE_AGENT_MODEL,
            role="Critical Challenge and Stress Testing",
            cognitive_function="devils_advocate"
        )

        self.synthesis_agent = OpenRouterSingleAgent(
            name="Synthesis Agent",
            model=SA_SYNTHESIS_AGENT_MODEL,
            role="Integration and Dissent Management",
            cognitive_function="synthesis"
        )

        self.final_agent = OpenRouterSingleAgent(
            name="Final Agent",
            model=SA_FINAL_AGENT_MODEL,
            role="Authoritative Decision Making",
            cognitive_function="final"
        )

        # Initialize shared HTTP client and semaphore
        self.client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {self.response_agent.api_key}",
                "HTTP-Referer": self.response_agent.http_referer,
                "X-Title": self.response_agent.x_title,
                "Content-Type": "application/json"
            },
            limits=httpx.Limits(max_connections=OPENROUTER_CONCURRENCY + 5, max_keepalive_connections=OPENROUTER_CONCURRENCY)
        )
        self.semaphore = asyncio.Semaphore(OPENROUTER_CONCURRENCY)

        logger.info("=== Single Agent Workflow Initialized ===")
        logger.info(f"Response Agent: {SA_RESPONSE_AGENT_MODEL}")
        logger.info(f"Devils Advocate: {SA_DEVILS_ADVOCATE_AGENT_MODEL}")
        logger.info(f"Synthesis Agent: {SA_SYNTHESIS_AGENT_MODEL}")
        logger.info(f"Final Agent: {SA_FINAL_AGENT_MODEL}")
        logger.info(f"Layers: {SA_NUM_LAYERS}, Self-Review: {SA_ENABLE_SELF_REVIEW}")
        logger.info("===========================================")

    async def close_client(self):
        """Close the shared HTTP client."""
        await self.client.aclose()
        logger.info("Single Agent Workflow HTTP client closed.")

    async def run_workflow(self, prompt: str) -> Dict[str, Any]:
        """
        Run the complete single agent workflow.

        Args:
            prompt: The user prompt to process

        Returns:
            Dictionary containing all workflow outputs and metadata
        """
        workflow_start_time = time.time()
        layer_outputs = []

        logger.info(colored(f"Starting Single Agent Workflow with {SA_NUM_LAYERS} layers", "cyan"))
        if self.tracer:
            self.tracer.log_workflow_event("Workflow Start", {"timestamp": workflow_start_time})

        try:
            # Process each layer
            prev_synthesis = ""
            prev_devils_advocate = ""

            for layer_idx in range(SA_NUM_LAYERS):
                layer_start_time = time.time()
                layer_details = {"layer_number": layer_idx + 1}

                logger.info(colored(f"Processing Layer {layer_idx + 1}/{SA_NUM_LAYERS}", "yellow"))

                if self.tracer:
                    self.tracer.log_layer_event(
                        layer_num=layer_idx + 1,
                        event_type="Start",
                        prompt=prompt,
                        metrics={"expected_tokens": SA_RESPONSE_MAX_TOKENS}
                    )

                # Phase 1: Response Agent
                response_prompt = build_response_prompt(
                    prompt, layer_idx, prev_synthesis, prev_devils_advocate
                )
                start_time = time.time()
                response = await self.response_agent.generate(
                    response_prompt, self.client, self.semaphore, SA_RESPONSE_MAX_TOKENS
                )
                duration = time.time() - start_time

                if self.tracer:
                    self.tracer.log_api_call(
                        model=self.response_agent.model,
                        prompt=response_prompt,
                        response=response,
                        duration=duration,
                        tokens_used=count_tokens_approximate(response)
                    )

                layer_details["response"] = response
                layer_details["response_prompt"] = response_prompt
                logger.info(colored(f"✓ Response Agent completed ({duration:.2f}s)", "green"))

                # Phase 2: Self-Review (optional)
                self_review = ""
                if SA_ENABLE_SELF_REVIEW:
                    review_prompt = build_self_review_prompt(prompt, response, layer_idx)
                    start_time = time.time()
                    self_review = await self.response_agent.generate(
                        review_prompt, self.client, self.semaphore, SA_REVIEW_MAX_TOKENS
                    )
                    duration = time.time() - start_time

                    if self.tracer:
                        self.tracer.log_api_call(
                            model=self.response_agent.model,
                            prompt=review_prompt,
                            response=self_review,
                            duration=duration,
                            tokens_used=count_tokens_approximate(self_review)
                        )

                    layer_details["self_review"] = self_review
                    layer_details["self_review_prompt"] = review_prompt
                    logger.info(colored(f"✓ Self-Review completed ({duration:.2f}s)", "green"))

                # Phase 3: Devils Advocate
                devils_advocate_prompt = build_devils_advocate_prompt(
                    prompt, response, self_review, layer_idx, prev_synthesis
                )
                start_time = time.time()
                devils_advocate = await self.devils_advocate_agent.generate(
                    devils_advocate_prompt, self.client, self.semaphore, SA_DEVILS_ADVOCATE_MAX_TOKENS
                )
                duration = time.time() - start_time

                if self.tracer:
                    self.tracer.log_api_call(
                        model=self.devils_advocate_agent.model,
                        prompt=devils_advocate_prompt,
                        response=devils_advocate,
                        duration=duration,
                        tokens_used=count_tokens_approximate(devils_advocate)
                    )

                layer_details["devils_advocate"] = devils_advocate
                layer_details["devils_advocate_prompt"] = devils_advocate_prompt
                prev_devils_advocate = devils_advocate
                logger.info(colored(f"✓ Devils Advocate completed ({duration:.2f}s)", "green"))

                # Phase 4: Synthesis
                synthesis_prompt = build_synthesis_prompt(
                    prompt, response, devils_advocate, self_review, layer_idx, prev_synthesis
                )
                start_time = time.time()
                synthesis = await self.synthesis_agent.generate(
                    synthesis_prompt, self.client, self.semaphore, SA_SYNTHESIS_MAX_TOKENS
                )
                duration = time.time() - start_time

                if self.tracer:
                    self.tracer.log_api_call(
                        model=self.synthesis_agent.model,
                        prompt=synthesis_prompt,
                        response=synthesis,
                        duration=duration,
                        tokens_used=count_tokens_approximate(synthesis)
                    )

                layer_details["synthesis"] = synthesis
                layer_details["synthesis_prompt"] = synthesis_prompt
                prev_synthesis = synthesis
                logger.info(colored(f"✓ Synthesis completed ({duration:.2f}s)", "green"))

                # Log layer completion
                layer_duration = time.time() - layer_start_time
                if self.tracer:
                    self.tracer.log_layer_event(
                        layer_num=layer_idx + 1,
                        event_type="End",
                        prompt=prompt,
                        metrics={"duration": layer_duration}
                    )

                layer_outputs.append(layer_details)

            # Phase 5: Final Agent
            logger.info(colored("Generating Final Answer", "yellow"))
            final_prompt = build_final_prompt(prompt, layer_outputs)
            start_time = time.time()
            final_output = await self.final_agent.generate(
                final_prompt, self.client, self.semaphore, SA_FINAL_MAX_TOKENS
            )
            duration = time.time() - start_time

            if self.tracer:
                self.tracer.log_api_call(
                    model=self.final_agent.model,
                    prompt=final_prompt,
                    response=final_output,
                    duration=duration,
                    tokens_used=count_tokens_approximate(final_output)
                )

            logger.info(colored(f"✓ Final Agent completed ({duration:.2f}s)", "green"))

            # Calculate workflow metrics
            workflow_duration = time.time() - workflow_start_time
            total_tokens = sum(
                count_tokens_approximate(layer.get("response", "")) +
                count_tokens_approximate(layer.get("self_review", "")) +
                count_tokens_approximate(layer.get("devils_advocate", "")) +
                count_tokens_approximate(layer.get("synthesis", ""))
                for layer in layer_outputs
            ) + count_tokens_approximate(final_output)

            result = {
                "prompt": prompt,
                "layer_outputs": layer_outputs,
                "final_output": final_output,
                "final_prompt": final_prompt,
                "metadata": {
                    "num_layers": SA_NUM_LAYERS,
                    "self_review_enabled": SA_ENABLE_SELF_REVIEW,
                    "workflow_duration": workflow_duration,
                    "total_tokens": total_tokens,
                    "agents_used": {
                        "response": SA_RESPONSE_AGENT_MODEL,
                        "devils_advocate": SA_DEVILS_ADVOCATE_AGENT_MODEL,
                        "synthesis": SA_SYNTHESIS_AGENT_MODEL,
                        "final": SA_FINAL_AGENT_MODEL
                    }
                }
            }

            if self.tracer:
                self.tracer.log_workflow_event("Workflow Complete", {
                    "timestamp": time.time(),
                    "duration": workflow_duration,
                    "total_tokens": total_tokens,
                    "num_layers": len(layer_outputs)
                })

            logger.info(colored(f"Single Agent Workflow completed successfully in {workflow_duration:.2f}s", "green"))
            return result

        except Exception as e:
            logger.error(colored(f"Error in Single Agent Workflow: {str(e)}", "red"))
            if self.tracer:
                self.tracer.log_workflow_event("Error", {
                    "error": str(e),
                    "traceback": str(e.__traceback__)
                })
            raise