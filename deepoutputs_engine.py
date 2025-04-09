import os
import asyncio
import httpx # Use httpx for async requests
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from termcolor import colored
from pathlib import Path
import re
from difflib import SequenceMatcher
import logging
import time # For backoff

load_dotenv()

# Configure basic console logging initially
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Get the root logger to potentially add file handlers later
logger = logging.getLogger()

# --- Configuration ---
# Agent Models (Correct as per user instruction)
AGENT1_MODEL = os.getenv("AGENT1_MODEL", "meta-llama/llama-4-maverick")
AGENT2_MODEL = os.getenv("AGENT2_MODEL", "qwen/qwq-32b")
AGENT3_MODEL = os.getenv("AGENT3_MODEL", "google/gemini-2.0-flash-001")
SYNTHESIS_AGENT_MODEL = os.getenv("SYNTHESIS_AGENT_MODEL", "google/gemini-2.0-flash-001")
DEVILS_ADVOCATE_AGENT_MODEL = os.getenv("DEVILS_ADVOCATE_AGENT_MODEL", "deepseek/deepseek-r1-distill-qwen-32b")
FINAL_AGENT_MODEL = os.getenv("FINAL_AGENT_MODEL", "google/gemini-2.5-pro-preview-03-25")

# API and Concurrency Settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

OPENROUTER_CONCURRENCY = int(os.getenv("OPENROUTER_CONCURRENCY", "5")) # Max concurrent requests
API_RETRY_ATTEMPTS = int(os.getenv("API_RETRY_ATTEMPTS", "3")) # Max retries on specific errors
API_INITIAL_BACKOFF = float(os.getenv("API_INITIAL_BACKOFF", "1.0")) # Initial delay in seconds for retry
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "120.0")) # Timeout for API calls

HTTP_REFERER = os.getenv("HTTP_REFERER", "MOA_Demo/1.0") # Recommended: Your App Name/Version
X_TITLE = os.getenv("X_TITLE", "MOA Demo") # Recommended: Your App Name

# --- Custom Exception ---
class AgentGenerationError(Exception):
    """Custom exception for agent generation failures."""
    pass

# --- Helper Functions ---
def read_prompt_from_file(file_path: str = "prompt.txt") -> str:
    """
    Reads the prompt from the specified file.
    
    Args:
        file_path: Path to the prompt file, defaults to 'prompt.txt' in the root folder.
        
    Returns:
        The prompt as a string.
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        # Log prompt reading before file handler is potentially set
        logging.info(f"Successfully read prompt from {file_path}")
        return prompt
    except FileNotFoundError:
        logging.error(f"Prompt file not found: {file_path}")
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    except Exception as e:
        logging.error(f"Error reading prompt file: {str(e)}")
        raise

def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Sanitizes a string to be suitable for use as a filename or folder name.
    Uses the first few words for relevance.
    """
    # Take the first N characters to get context
    context = text[:max_length*2] # Take more initially to get words
    words = context.split()[:5] # Take first 5 words
    if not words:
        return "untitled"
        
    base_name = "_".join(words)
    
    # Remove invalid characters
    sanitized = re.sub(r'[^\w\-_\. ]', '', base_name)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Truncate to max_length
    sanitized = sanitized[:max_length]
    # Ensure it's not empty
    if not sanitized:
        return "untitled"
    return sanitized.lower()

# --- Agent Definition ---
class Agent:
    """Base class for agents."""
    def __init__(self, name: str, model: str, role: str):
        self.name = name
        self.model = model
        self.role = role

    async def generate(self, prompt: str, client: httpx.AsyncClient, semaphore: asyncio.Semaphore, max_tokens: int) -> str:
        raise NotImplementedError

class OpenRouterAgent(Agent):
    """Agent implementation using the OpenRouter API."""
    def __init__(self, name: str, model: str, role: str):
        super().__init__(name, model, role)
        # Note: Client is no longer created here, it's passed in generate method

    async def generate(self, prompt: str, client: httpx.AsyncClient, semaphore: asyncio.Semaphore, max_tokens: int = 2000) -> str:
        """
        Generates a response using the OpenRouter API with rate limiting and retries.

        Args:
            prompt: The input prompt for the agent.
            client: The shared httpx.AsyncClient instance.
            semaphore: The asyncio.Semaphore to limit concurrency.
            max_tokens: The maximum number of tokens for the response.

        Returns:
            The generated text content.

        Raises:
            AgentGenerationError: If generation fails after retries.
        """
        async with semaphore: # Wait for semaphore before making a call
            logger.debug(f"Agent {self.name} ({self.model}) acquiring semaphore and starting generation.")
            result = None
            last_exception = None
            backoff_time = API_INITIAL_BACKOFF

            for attempt in range(API_RETRY_ATTEMPTS):
                try:
                    response = await client.post(
                        "/chat/completions",
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens
                        },
                        timeout=API_TIMEOUT
                    )

                    response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx
                    result = response.json()

                    generated_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not generated_content:
                         # Handle cases where the API returns success but empty content
                         raise AgentGenerationError(f"Agent {self.name} returned empty content.")

                    logger.debug(f"Agent {self.name} ({self.model}) generated response successfully.")
                    return generated_content

                except httpx.ReadTimeout as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1}/{API_RETRY_ATTEMPTS} - Timeout error for {self.name}: {e}. Retrying in {backoff_time:.2f} seconds...")
                except httpx.HTTPStatusError as e:
                    last_exception = e
                    if e.response.status_code == 429: # Rate limit
                        logger.warning(f"Attempt {attempt + 1}/{API_RETRY_ATTEMPTS} - Rate limit error (429) for {self.name}. Retrying in {backoff_time:.2f} seconds...")
                    elif 500 <= e.response.status_code < 600: # Server error
                         logger.warning(f"Attempt {attempt + 1}/{API_RETRY_ATTEMPTS} - Server error ({e.response.status_code}) for {self.name}. Retrying in {backoff_time:.2f} seconds...")
                    else:
                        # Non-retryable HTTP error
                        logger.error(f"HTTP Error for {self.name}: {e}")
                        raise AgentGenerationError(f"Agent {self.name} failed with non-retryable HTTP error: {e}") from e
                except Exception as e:
                    # Catch other unexpected errors during the API call or response processing
                    last_exception = e
                    logger.error(f"Unexpected error during generation for {self.name}: {e}", exc_info=True)
                    # Depending on the error, you might classify it as retryable or not
                    # For simplicity, we'll treat unexpected errors as potentially non-retryable here
                    # but log them thoroughly. If they *should* be retried, adjust logic.
                    raise AgentGenerationError(f"Agent {self.name} encountered an unexpected error: {e}") from e


                # Wait before retrying
                if attempt < API_RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(backoff_time)
                    backoff_time *= 2 # Exponential backoff
                else:
                   logger.error(f"Agent {self.name} ({self.model}) failed after {API_RETRY_ATTEMPTS} attempts.")
                   raise AgentGenerationError(f"Agent {self.name} ({self.model}) failed after {API_RETRY_ATTEMPTS} attempts.") from last_exception

            # This part should ideally not be reached if logic is correct, but as a safeguard:
            raise AgentGenerationError(f"Agent {self.name} ({self.model}) failed unexpectedly to generate a response.")

# --- Mixture of Agents Implementation ---
class MixtureOfAgents:
    def __init__(self, models: List[str], num_layers: int = 2):
        if not models:
            raise ValueError("At least one agent model must be provided.")

        self.agents = [OpenRouterAgent(f"Agent {i+1}", model, f"Role {i+1}") for i, model in enumerate(models)]
        self.num_layers = num_layers
        self.synthesis_agent = OpenRouterAgent("Synthesis Agent", SYNTHESIS_AGENT_MODEL, "Synthesizer Role")
        self.devils_advocate_agent = OpenRouterAgent("Devil's Advocate Agent", DEVILS_ADVOCATE_AGENT_MODEL, "Devil's Advocate Role")
        self.final_agent = OpenRouterAgent("Final Agent", FINAL_AGENT_MODEL, "Final Decision Role")

        # Shared HTTP client and Semaphore for rate limiting
        self.client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": HTTP_REFERER,
                "X-Title": X_TITLE,
                "Content-Type": "application/json"
            },
            # Increase default pool limits if high concurrency is expected
            limits=httpx.Limits(max_connections=OPENROUTER_CONCURRENCY + 5, max_keepalive_connections=OPENROUTER_CONCURRENCY)
        )
        self.semaphore = asyncio.Semaphore(OPENROUTER_CONCURRENCY)
        logger.info(f"Initialized MixtureOfAgents with {len(self.agents)} agents, {self.num_layers} layers, and concurrency limit {OPENROUTER_CONCURRENCY}.")

    async def close_client(self):
        """Closes the shared httpx client."""
        await self.client.aclose()
        logger.info("HTTP client closed.")

    async def generate(self, prompt: str) -> Tuple[List[Dict[str, Any]], str, Dict[str, float]]:
        """
        Runs the multi-layer MoA process.

        Returns:
            A tuple containing:
            - layer_details: List of dictionaries, each detailing a layer's execution.
            - final_response: The final generated response string.
            - utilization: Dictionary mapping agent names to utilization percentages.
        """
        layer_details = []
        all_agent_responses_for_utilization = [[] for _ in self.agents] # Store responses per agent across layers

        # current_context = "" # Removed as context is built per layer
        last_synthesis = ""
        last_devils_advocate = ""

        for i in range(self.num_layers):
            layer_num = i + 1
            logger.info(colored(f"\n* Layer {layer_num} started", "cyan"))

            layer_prompt = self.create_layer_prompt(prompt, last_synthesis, last_devils_advocate, i)
            logger.debug(f"Layer {layer_num} Prompt:\n{layer_prompt}")

            # --- Step 1: Initial Responses ---
            initial_responses = await self.run_agents_concurrently(self.agents, layer_prompt, "initial response")
            for agent_idx, response in enumerate(initial_responses):
                 if agent_idx < len(all_agent_responses_for_utilization):
                     all_agent_responses_for_utilization[agent_idx].append(response)
            logger.info(colored(f"* Layer {layer_num} initial responses received ({len(initial_responses)} agents)", "green"))

            # --- Step 2: Aggregation & Peer Review ---
            aggregation_responses = await self.run_agents_concurrently(
                self.agents,
                self.create_aggregation_prompt(prompt, layer_prompt, initial_responses),
                "aggregation"
            )
            for agent_idx, response in enumerate(aggregation_responses):
                 if agent_idx < len(all_agent_responses_for_utilization):
                     all_agent_responses_for_utilization[agent_idx].append(response)
            logger.info(colored(f"* Layer {layer_num} aggregations completed ({len(aggregation_responses)} agents)", "green"))

            # --- Step 3: Synthesis and Devil's Advocate ---
            synthesis, devils_advocate = await self.synthesize_and_critique(prompt, layer_prompt, aggregation_responses)
            last_synthesis = synthesis # Update context for next layer
            last_devils_advocate = devils_advocate # Update context for next layer
            logger.info(colored(f"* Layer {layer_num} synthesis and devil's advocate perspective generated", "green"))

            layer_details.append({
                "layer_number": layer_num,
                "layer_prompt_details": "Original Prompt" if i == 0 else "Original Prompt + Synthesis/Critique from Layer " + str(i),
                "initial_responses": list(zip([agent.name for agent in self.agents], initial_responses)),
                "aggregation_responses": list(zip([agent.name for agent in self.agents], aggregation_responses)),
                "synthesis": synthesis,
                "devils_advocate": devils_advocate,
                "synthesis_agent_name": self.synthesis_agent.name,
                "devils_advocate_agent_name": self.devils_advocate_agent.name,
            })
            logger.info(colored(f"* Layer {layer_num} completed", "cyan"))

        # --- Final Output Generation ---
        logger.info(colored("\n* Generating Final Output", "yellow"))
        final_response = await self.generate_final_output(prompt, layer_details)
        logger.info(colored("* Final Output Generated", "yellow"))

        # --- Utilization Calculation ---
        utilization = self.calculate_utilization(all_agent_responses_for_utilization, final_response)

        return layer_details, final_response, utilization

    async def run_agents_concurrently(self, agents: List[OpenRouterAgent], prompt: str, task_description: str) -> List[str]:
        """Runs multiple agents concurrently on the same prompt."""
        tasks = [agent.generate(prompt, self.client, self.semaphore) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error during {task_description} for {agents[i].name}: {result}")
                processed_results.append(f"Error: Generation failed for {agents[i].name}. Reason: {result}")
            else:
                 processed_results.append(str(result)) # Ensure result is a string
        return processed_results

    def create_layer_prompt(self, original_prompt: str, prev_synthesis: str, prev_devils_advocate: str, layer_index: int) -> str:
        """Creates the prompt for agents in a specific layer."""
        if layer_index == 0:
            return original_prompt
        else:
            # Ensure previous context is not empty before adding
            context_header = f"--- Context from Previous Layer (Layer {layer_index}): ---"
            synthesis_text = f"Synthesis:\n{prev_synthesis}\n" if prev_synthesis else ""
            critique_text = f"Devil's Advocate Critique:\n{prev_devils_advocate}\n" if prev_devils_advocate else ""
            context_block = f"{context_header}\n\n{synthesis_text}\n{critique_text}---\n\n" if synthesis_text or critique_text else ""

            return f"""Original User Prompt:
{original_prompt}

{context_block}Your Task for Layer {layer_index + 1}:
1.  Re-read the **Original User Prompt** carefully.
2.  Critically evaluate the Synthesis and Devil's Advocate Critique from the previous layer (if provided) IN THE CONTEXT of the Original User Prompt.
3.  Generate your own **independent** response to the **Original User Prompt**.
4.  Explicitly state if and how the provided context influenced your response compared to your initial thoughts. Explain your reasoning.
5.  Ensure your response is well-reasoned, addresses the core question of the Original User Prompt, and is distinct from the previous layer's output.
"""

    def create_aggregation_prompt(self, original_prompt: str, current_layer_prompt: str, initial_responses: List[str]) -> str:
        """Creates the prompt for the aggregation/peer review step."""
        response_text = "\n\n---\n\n".join([f"Response from Agent {i+1}:\n{resp}" for i, resp in enumerate(initial_responses)])

        return f"""Original User Prompt:
{original_prompt}

---
Current Layer Context/Prompt Given to Agents:
{current_layer_prompt}
---

Initial Responses Received in this Layer:
{response_text}
---

Your Task (Aggregation & Peer Review):
You are reviewing the initial responses above to the challenge posed by the 'Current Layer Context/Prompt'.

1.  **Critique All Responses:** Analyze the logic, reasoning, and factual accuracy of *all* responses provided, including potentially your own if you recognize it. Be specific in your critiques.
2.  **Identify Assumptions:** Explicitly list and challenge the key assumptions (stated or unstated) made in *each* response.
3.  **Verify (If Applicable):** If the prompt involves calculations or checkable facts, attempt to verify them. State your findings.
4.  **Explore Alternatives:** Consider alternative interpretations of the prompt or fundamentally different approaches that were missed. Could the reasoning be flawed?
5.  **Synthesize Strengths/Weaknesses:** Briefly summarize the strongest points and biggest weaknesses you observed across all responses.
6.  **Generate Improved Response:** Based *only* on your critical analysis and understanding of the Original User Prompt, provide your own improved and independent answer to the core question asked in the Original User Prompt. Do NOT simply combine the previous answers. Use them as context for critique but form your own conclusion.
7.  **Explain Your Reasoning:** Justify why your improved response is better or more accurate than the initial responses. If you agree with parts of other responses, state which and why.

Structure your output clearly, addressing each point above. Mark your final improved response clearly (e.g., "My Improved Response:").
"""

    async def synthesize_and_critique(self, original_prompt: str, current_layer_prompt: str, aggregation_responses: List[str]) -> Tuple[str, str]:
        """Generates synthesis and devil's advocate perspectives concurrently."""
        logger.info(colored("* Starting synthesis and devil's advocate generation...", "magenta"))

        synthesis_prompt = self.create_synthesis_prompt(original_prompt, current_layer_prompt, aggregation_responses)
        devils_advocate_prompt = self.create_devils_advocate_prompt(original_prompt, current_layer_prompt, aggregation_responses)

        tasks = {
            "synthesis": self.synthesis_agent.generate(synthesis_prompt, self.client, self.semaphore),
            "devils_advocate": self.devils_advocate_agent.generate(devils_advocate_prompt, self.client, self.semaphore)
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        processed_results = dict(zip(tasks.keys(), results))

        # Handle Synthesis result
        synthesis = "" # Default to empty string
        if isinstance(processed_results["synthesis"], Exception):
            logger.error(f"Error generating Synthesis: {processed_results['synthesis']}")
            synthesis = f"Error: Synthesis generation failed. Reason: {processed_results['synthesis']}"
        elif processed_results["synthesis"]:
             synthesis = processed_results["synthesis"].replace("Synthesis:", "").strip()
        else:
            logger.warning("Synthesis agent returned an empty response.")
            synthesis = "No synthesis was generated (agent returned empty response)."


        # Handle Devil's Advocate result
        devils_advocate = "" # Default to empty string
        if isinstance(processed_results["devils_advocate"], Exception):
            logger.error(f"Error generating Devil's Advocate perspective: {processed_results['devils_advocate']}")
            devils_advocate = f"Error: Devil's Advocate generation failed. Reason: {processed_results['devils_advocate']}"
        elif processed_results["devils_advocate"]:
            devils_advocate = processed_results["devils_advocate"].replace("Devil's Advocate:", "").strip()
        else:
             logger.warning("Devil's Advocate agent returned an empty response.")
             devils_advocate = "No specific Devil's Advocate perspective was provided (agent returned empty response)."

        logger.info(colored("* Synthesis and devil's advocate generation completed", "magenta"))
        return synthesis, devils_advocate


    def create_synthesis_prompt(self, original_prompt: str, current_layer_prompt: str, aggregated_responses: List[str]) -> str:
        """Creates the prompt for the synthesis agent."""
        response_text = "\n\n---\n\n".join([f"Aggregated Response from Agent {i+1}:\n{resp}" for i, resp in enumerate(aggregated_responses)])

        return f"""Original User Prompt:
{original_prompt}

---
Current Layer Context/Prompt Given to Agents:
{current_layer_prompt}
---

Aggregated/Reviewed Responses Received in this Layer:
{response_text}
---

Your Task (Synthesis Agent):
Analyze the 'Aggregated/Reviewed Responses' provided above. Your goal is to create a concise synthesis that captures the essence of the discussion and prepares for the *next* layer of analysis or the final answer.

1.  **Identify Core Insights:** What are the key conclusions, findings, or proposed solutions presented in the aggregated responses?
2.  **Highlight Key Agreements & Disagreements:** Summarize the main points where the responses concur and where they diverge significantly.
3.  **Assess Confidence/Uncertainty:** Note any areas where the responses indicate high uncertainty or conflicting information.
4.  **Identify Unresolved Issues:** What important aspects of the 'Original User Prompt' still seem unclear or require further investigation in the next layer?
5.  **Synthesize Succinctly:** Create a brief synthesis (a few paragraphs) that summarizes the current state of the analysis based on the aggregated responses. This synthesis will be part of the context for the next layer. Focus on clarity and accuracy.

Do NOT simply list the responses. Provide a coherent summary and assessment. Start your response directly with the synthesis, without introductory phrases like "Here is the synthesis:".
"""

    def create_devils_advocate_prompt(self, original_prompt: str, current_layer_prompt: str, aggregated_responses: List[str]) -> str:
        """Creates the prompt for the devil's advocate agent."""
        response_text = "\n\n---\n\n".join([f"Aggregated Response from Agent {i+1}:\n{resp}" for i, resp in enumerate(aggregated_responses)])

        return f"""Original User Prompt:
{original_prompt}

---
Current Layer Context/Prompt Given to Agents:
{current_layer_prompt}
---

Aggregated/Reviewed Responses Received in this Layer:
{response_text}
---

Your Task (Aggressive Devil's Advocate):
Your sole purpose is to rigorously challenge the prevailing conclusions and reasoning found in the 'Aggregated/Reviewed Responses'. Be critical, skeptical, and look for flaws.

1.  **Attack the Consensus:** If there's a common answer or approach, argue forcefully against it. Why might it be wrong, incomplete, or based on flawed premises?
2.  **Challenge Fundamental Assumptions:** Question the most basic assumptions made by the agents. Are they justified? What if they are wrong?
3.  **Identify Blind Spots:** What critical factors, alternative scenarios, edge cases, or potential negative consequences have *all* the responses overlooked?
4.  **Expose Logical Fallacies:** Point out any weak arguments, leaps in logic, or inconsistencies within or between the responses.
5.  **Propose Contrarian Views:** Offer at least one completely different perspective or solution, even if it seems unconventional. Explain why it *could* be valid.
6.  **Nitpick Calculations/Data (If Applicable):** If there are numbers or data, question their validity, source, or interpretation.

Your tone should be challenging and critical. Your goal is NOT to be helpful or find agreement, but to stress-test the current conclusions by finding their weakest points. Structure your critique logically. Start your response directly with the critique, without introductory phrases like "Here is the Devil's Advocate perspective:".
"""


    async def generate_final_output(self, original_prompt: str, layer_details: List[Dict[str, Any]]) -> str:
        """Generates the final response based on all layer outputs."""
        consolidated_context = ""
        for layer in layer_details:
            consolidated_context += f"--- Layer {layer['layer_number']} ---\n"
            consolidated_context += f"Synthesis:\n{layer['synthesis']}\n\n"
            consolidated_context += f"Devil's Advocate Critique:\n{layer['devils_advocate']}\n\n"

        final_prompt = f"""Original User Prompt:
{original_prompt}

---
Consolidated Context from All Layers:
{consolidated_context}
---

Your Task (Final Agent):
You must provide the definitive, final answer to the 'Original User Prompt'. Use the 'Consolidated Context' as input and analysis history, but rely ultimately on your own reasoning and interpretation of the original prompt.

1.  **Re-evaluate Prompt:** Carefully re-read and interpret the 'Original User Prompt'. What is the core question or task?
2.  **Cross-Check Context:** Review the 'Consolidated Context'. Identify the key insights, conclusions, and critiques from the layers. Note areas of agreement and disagreement.
3.  **Identify Discrepancies:** Explicitly state any major discrepancies, contradictions, or unresolved issues between the layers or between the context and the 'Original User Prompt'.
4.  **Resolve Conflicts:** Address the discrepancies you identified. Explain how you are resolving them based on your understanding of the original prompt and logical reasoning. State which arguments or pieces of information you are prioritizing and why.
5.  **Formulate Final Answer:** Provide a clear, comprehensive, and well-reasoned final answer directly addressing the 'Original User Prompt'. Structure your answer logically.
6.  **Justify Your Answer:** Briefly explain why your final answer is the most accurate and complete response, referencing how you used or discarded elements from the context. If you disagree strongly with the context, state why.
7.  **Acknowledge Uncertainty (If Any):** If ambiguity remains or multiple valid answers exist, state this clearly and explain the different possibilities or necessary assumptions.

Ensure your final output is coherent, directly answers the user's original question, and represents your best possible analysis. Start your response directly with the final answer, without introductory phrases like "Here is the final answer:".
"""

        try:
            # Try the main final agent first
            final_response = await self.final_agent.generate(final_prompt, self.client, self.semaphore, max_tokens=3000) # Allow more tokens for final
            if final_response and len(final_response.strip()) > 50: # Basic check for meaningful content
                 logger.info(f"Final response generated successfully by {self.final_agent.name}.")
                 return final_response.strip()
            else:
                 logger.warning(f"Final agent {self.final_agent.name} returned short or empty response. Trying fallback.")
                 raise AgentGenerationError("Final response was empty or too short")

        except Exception as e:
            logger.error(colored(f"Error generating final response with {self.final_agent.name}: {e}", "red"), exc_info=True)
            logger.warning(colored("Attempting fallback methods...", "yellow"))

            # Fallback 1: Try Final Agent with a simpler prompt
            try:
                logger.info("Fallback 1: Trying Final Agent with simplified prompt.")
                simplified_prompt = f"""Original User Prompt: {original_prompt}\n\nBased on extensive multi-agent analysis (including critiques), provide your own definitive and comprehensive answer to the Original User Prompt. Be clear, logical, and directly address all aspects of the original request."""
                fallback_response = await self.final_agent.generate(simplified_prompt, self.client, self.semaphore, max_tokens=2500)
                if fallback_response and len(fallback_response.strip()) > 50:
                    logger.info(f"Fallback 1 succeeded using {self.final_agent.name} with simplified prompt.")
                    return fallback_response.strip()
                else:
                    raise AgentGenerationError("Fallback 1 response was empty or too short.")
            except Exception as fallback_e1:
                logger.error(colored(f"Fallback 1 failed: {fallback_e1}", "red"))

                # Fallback 2: Try Synthesis Agent with the simplified prompt
                try:
                    logger.info(colored(f"Fallback 2: Trying Synthesis Agent ({self.synthesis_agent.name}) with simplified prompt.", "yellow"))
                    synthesis_fallback = await self.synthesis_agent.generate(simplified_prompt, self.client, self.semaphore, max_tokens=2500)
                    if synthesis_fallback and len(synthesis_fallback.strip()) > 50:
                        logger.info(f"Fallback 2 succeeded using {self.synthesis_agent.name}.")
                        return synthesis_fallback.strip()
                    else:
                        raise AgentGenerationError("Fallback 2 response was empty or too short.")
                except Exception as fallback_e2:
                    logger.error(colored(f"Fallback 2 also failed: {fallback_e2}", "red"))
                    
                    # Last Resort: Return error message with context summary
                    logger.critical("All final response generation methods failed.")
                    error_message = f"""Apologies, but the final response could not be generated due to persistent errors.

Original Prompt: "{original_prompt}"

Summary of analysis context passed to the final agent:
{consolidated_context[:1000]}...

Please try refining your prompt or Rerun the process. The specific errors have been logged."""
                    return error_message


    def calculate_utilization(self, all_agent_responses_for_utilization: List[List[str]], final_response: str) -> Dict[str, float]:
        """
        Calculates the 'utilization' of each base agent based on the similarity
        of their combined contributions (initial + aggregation per layer) to the final response.
        Note: This is a heuristic measure.
        """
        utilization = {agent.name: 0.0 for agent in self.agents}
        total_similarity = 0.0

        if not final_response or final_response.startswith("Apologies,"): # Don't calculate if final response failed
             logger.warning("Skipping utilization calculation due to failed final response.")
             return utilization

        for agent_index, agent in enumerate(self.agents):
            # Combine all responses (initial and aggregation) from this agent across all layers
            agent_combined_text = " ".join(all_agent_responses_for_utilization[agent_index])

            if not agent_combined_text.strip():
                similarity = 0.0 # Agent produced no text
            else:
                 # Use SequenceMatcher for similarity scoring
                 # isjunk=None means treat all characters as important
                 similarity = SequenceMatcher(None, agent_combined_text, final_response, autojunk=False).ratio()


            utilization[agent.name] = similarity
            total_similarity += similarity
            logger.debug(f"Agent {agent.name} combined text length: {len(agent_combined_text)}, Similarity to final: {similarity:.4f}")


        if total_similarity == 0:
            # Avoid division by zero and assign equal weight if no similarity found (or only 1 agent)
             num_agents = len(self.agents)
             if num_agents > 0:
                logger.warning("Total similarity is zero. Assigning equal utilization.")
                equal_share = 100.0 / num_agents
                return {agent.name: equal_share for agent in self.agents}
             else:
                 return {} # No agents
        else:
            # Normalize scores to percentages
            return {agent_name: (sim / total_similarity) * 100.0 for agent_name, sim in utilization.items()}


# --- Report Generation ---

def generate_markdown_report(
    prompt: str,
    layer_details: List[Dict[str, Any]],
    final_response: str,
    utilization: Dict[str, float],
    final_agent_name: str
) -> str:
    """Generates the final summary Markdown report."""
    markdown_content = f"""# MoA Response Report

## Original Prompt
> {prompt}

## Agent Utilization
{chr(10).join([f"- {agent_name}: {percentage:.2f}%" for agent_name, percentage in utilization.items()])}

*(Note: Utilization is a heuristic based on text similarity to the final output)*

## Final MoA Response
**Final Response Agent:** {final_agent_name}

```markdown
{final_response}
```
"""
    return markdown_content

def generate_detailed_markdown_report(
    prompt: str,
    layer_details: List[Dict[str, Any]],
    final_response: str,
    utilization: Dict[str, float],
    final_agent_name: str,
    synthesis_agent_name: str,
    devils_advocate_agent_name: str
) -> str:
    """Generates a detailed comprehensive Markdown report with all intermediate outputs."""
    markdown_content = f"""# MoA Detailed Response Report

## Original Prompt
> {prompt}

## Agent Utilization
{chr(10).join([f"- {agent_name}: {percentage:.2f}%" for agent_name, percentage in utilization.items()])}

*(Note: Utilization is a heuristic based on text similarity to the final output)*

## Final MoA Response
**Final Response Agent:** {final_agent_name}

```markdown
{final_response}
```

---

## Intermediate Outputs
"""

    for layer in layer_details:
        layer_num = layer["layer_number"]
        markdown_content += f"""
### Layer {layer_num}

<details>
<summary>Layer {layer_num} Details (Click to expand)</summary>

#### Layer Prompt
> {layer["layer_prompt_details"]}

#### Step 1 - Agents Initial Responses
"""
        # Ensure initial_responses is a list of tuples/lists before trying to unpack
        if layer.get("initial_responses") and isinstance(layer["initial_responses"], list):
            for item in layer["initial_responses"]:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    agent_name, response = item
                    markdown_content += f"""
##### {agent_name}
```text
{response}
```
"""
                else:
                     markdown_content += f"\n_Invalid format for initial response item: {item}_"
        else:
             markdown_content += "\n_No initial responses recorded or invalid format._"


        markdown_content += """
#### Step 2 - Agent Aggregation of All Responses
"""
        # Ensure aggregation_responses is a list of tuples/lists before trying to unpack
        if layer.get("aggregation_responses") and isinstance(layer["aggregation_responses"], list):
             for item in layer["aggregation_responses"]:
                 if isinstance(item, (list, tuple)) and len(item) == 2:
                    agent_name, response = item
                    markdown_content += f"""
##### {agent_name}
```text
{response}
```
"""
                 else:
                      markdown_content += f"\n_Invalid format for aggregation response item: {item}_"
        else:
              markdown_content += "\n_No aggregation responses recorded or invalid format._"


        markdown_content += f"""
#### Step 3 - Synthesized Aggregated Responses (Synthesis Agent: {synthesis_agent_name})

##### Synthesis
```text
{layer.get("synthesis", "N/A")}
```

##### Devil's Advocate (Agent: {devils_advocate_agent_name})
```text
{layer.get("devils_advocate", "N/A")}
```

</details>

---
"""

    markdown_content += f"""
## Information Passed to Final Response Agent

The following synthesized information from all layers, along with the original user prompt, was passed to the final response agent ({final_agent_name}). The final agent used this information to generate the final MoA response.

"""
    for layer in layer_details:
        layer_num = layer["layer_number"]
        markdown_content += f"""
### Layer {layer_num} Synthesis

```text
{layer.get("synthesis", "N/A")}
```

### Layer {layer_num} Devil's Advocate

```text
{layer.get("devils_advocate", "N/A")}
```

---
"""

    return markdown_content.strip()

async def main():
    """Main function to run the Mixture of Agents model with prompt from file."""
    file_handler = None
    moa = None
    try:
        # --- Setup ---
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Read prompt from file
        prompt = read_prompt_from_file()
        
        # Generate base name for folder/files from prompt
        base_filename = sanitize_filename(prompt)
        logger.info(f"Using base name for run: {base_filename}")
        
        # Create run-specific reports directory
        reports_base_dir = Path("reports")
        run_reports_dir = reports_base_dir / base_filename
        run_reports_dir.mkdir(parents=True, exist_ok=True)
        
        # --- Configure File Logging ---
        log_file_path = run_reports_dir / f"{base_filename}_logs_{timestamp}.log"
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler) # Add handler to the root logger
        logger.info(f"Logging configured to file: {log_file_path}")
        
        # Log initial info again, now that file handler is set
        logger.info(f"Starting MoA run with base name: {base_filename}")
        logger.info(f"Prompt read from file: {prompt[:200]}...") # Log more of the prompt

        # --- Configure Number of Layers ---
        default_num_layers = 2
        try:
            num_layers_str = os.getenv("MOA_NUM_LAYERS", str(default_num_layers))
            num_layers = int(num_layers_str)
            if num_layers < 1:
                logger.warning(f"MOA_NUM_LAYERS value '{num_layers_str}' is less than 1. Using default: {default_num_layers}")
                num_layers = default_num_layers
            else:
                logger.info(f"Using {num_layers} layers (from MOA_NUM_LAYERS environment variable).")
        except ValueError:
            logger.warning(f"Invalid MOA_NUM_LAYERS value '{num_layers_str}'. Expected an integer. Using default: {default_num_layers}")
            num_layers = default_num_layers
        
        # --- Initialize MoA ---
        moa = MixtureOfAgents(
            models=[AGENT1_MODEL, AGENT2_MODEL, AGENT3_MODEL],
            num_layers=num_layers # Use configured number of layers
        )
        
        # --- Generate Response ---
        logger.info("Starting generation process...")
        layer_details, final_response, utilization = await moa.generate(prompt)
        logger.info("Generation process completed.")
        
        # --- Generate Reports ---
        logger.info("Generating reports...")
        final_report = generate_markdown_report(
            prompt=prompt,
            layer_details=layer_details,
            final_response=final_response,
            utilization=utilization,
            final_agent_name=moa.final_agent.name
        )
        
        detailed_report = generate_detailed_markdown_report(
            prompt=prompt,
            layer_details=layer_details,
            final_response=final_response,
            utilization=utilization,
            final_agent_name=moa.final_agent.name,
            synthesis_agent_name=moa.synthesis_agent.name,
            devils_advocate_agent_name=moa.devils_advocate_agent.name
        )
        logger.info("Reports generated.")

        # --- Save Reports ---
        final_report_path = run_reports_dir / f"{base_filename}_final_report_{timestamp}.md"
        detailed_report_path = run_reports_dir / f"{base_filename}_detailed_report_{timestamp}.md"
        
        try:
            with open(final_report_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            logger.info(f"Final report saved to {final_report_path}")
        except Exception as e:
             logger.error(f"Error saving final report to {final_report_path}: {e}", exc_info=True)

        try:
            with open(detailed_report_path, "w", encoding="utf-8") as f:
                f.write(detailed_report)
            logger.info(f"Detailed report saved to {detailed_report_path}")
        except Exception as e:
            logger.error(f"Error saving detailed report to {detailed_report_path}: {e}", exc_info=True)

        # --- Output to Console ---
        print(f"\n--- MoA Run Complete ---")
        print(f"Run Folder: {run_reports_dir}")
        print(f"Logs: {log_file_path}")
        print(f"Final Report: {final_report_path}")
        print(f"Detailed Report: {detailed_report_path}")
        print(f"\nFinal Response Preview:\n{final_response[:500]}...\n") # Show preview

    except FileNotFoundError as e:
         # Specific handling for prompt file not found, already logged
         print(f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"Critical error in main function: {str(e)}", exc_info=True)
        print(f"An unexpected error occurred: {str(e)}")
    finally:
        # --- Cleanup ---
        # Close the HTTP client if it was initialized
        if moa and moa.client:
            await moa.close_client()
            logger.info("HTTP client closed.")
        
        # Remove the file handler to prevent duplicate logging if main() is called again
        if file_handler:
            logger.removeHandler(file_handler)
            file_handler.close()
            logger.info("File logging handler removed and closed.")

if __name__ == "__main__":
    asyncio.run(main())
