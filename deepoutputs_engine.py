import os
import asyncio
import httpx # Use httpx for async requests
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
try:
    from termcolor import colored
except ImportError:
    def colored(text, color):
        return text
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

# Add detailed debug logging for environment variables
logger.info("==== Environment Variable Debug ====")
for key, value in os.environ.items():
    if "MODEL" in key or "AGENT" in key:
        logger.info(f"ENV: {key}={value}")
logger.info("===================================")

# --- Configuration ---
# Agent Models (symbolic indirection)

# (Removed unused get_agent_model function for code cleanliness)

# Load models directly from environment
AGENT1_MODEL = os.getenv("AGENT1_MODEL_SYMBOLIC") or os.getenv("AGENT1_MODEL")
AGENT2_MODEL = os.getenv("AGENT2_MODEL_SYMBOLIC") or os.getenv("AGENT2_MODEL")
AGENT3_MODEL = os.getenv("AGENT3_MODEL_SYMBOLIC") or os.getenv("AGENT3_MODEL")
DEEP_RESEARCH_AGENT_MODEL = os.getenv("DEEP_RESEARCH_AGENT_MODEL_SYMBOLIC") or os.getenv("DEEP_RESEARCH_AGENT_MODEL")
SYNTHESIS_AGENT_MODEL = os.getenv("SYNTHESIS_AGENT_MODEL_SYMBOLIC") or os.getenv("SYNTHESIS_AGENT_MODEL")
DEVILS_ADVOCATE_AGENT_MODEL = os.getenv("DEVILS_ADVOCATE_AGENT_MODEL_SYMBOLIC") or os.getenv("DEVILS_ADVOCATE_AGENT_MODEL")
FINAL_AGENT_MODEL = os.getenv("FINAL_AGENT_MODEL_SYMBOLIC") or os.getenv("FINAL_AGENT_MODEL")

# API and Concurrency Settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_CONCURRENCY = int(os.getenv("OPENROUTER_CONCURRENCY", "5")) # Max concurrent requests
API_RETRY_ATTEMPTS = int(os.getenv("API_RETRY_ATTEMPTS", "3")) # Max retries on specific errors
API_INITIAL_BACKOFF = float(os.getenv("API_INITIAL_BACKOFF", "1.0")) # Initial delay in seconds for retry
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "120.0")) # Timeout for API calls

# Hardcoded values for HTTP referer and title
HTTP_REFERER = os.getenv("HTTP_REFERER", "MOA_Demo/1.0") # Recommended: Your App Name/Version
X_TITLE = os.getenv("X_TITLE", "MOA Demo") # Recommended: Your App Name

# Stage-specific max_tokens settings
INITIAL_MAX_TOKENS = None  # Let API default apply
AGGREGATION_MAX_TOKENS = None  # Let API default apply
SYNTHESIS_MAX_TOKENS = None  # Let API default apply
DEVILS_ADVOCATE_MAX_TOKENS = int(os.getenv("DEVILS_ADVOCATE_MAX_TOKENS", "16000"))
FINAL_MAX_TOKENS = int(os.getenv("FINAL_MAX_TOKENS", "6000"))

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

def sanitize_for_markdown(text: str) -> str:
    """
    Ensures that text does not break markdown formatting, especially code blocks.
    - Escapes accidental triple backticks by replacing them with a similar sequence.
    - Ensures consistent line endings.
    """
    if not isinstance(text, str):
        return str(text)
    # Replace triple backticks with a similar but safe sequence
    return text.replace('```', '``\u200b`')

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

    async def generate(self, prompt: str, client: httpx.AsyncClient, semaphore: asyncio.Semaphore, max_tokens: int = 4000) -> str:
        """
        Generates a response using the OpenRouter API with rate limiting and retries.

        Args:
            prompt: The input prompt for the agent.
            client: The shared httpx.AsyncClient instance.
            semaphore: The asyncio.Semaphore to limit concurrency.
            max_tokens: The maximum number of tokens for the response. If None, 0, or blank, let API default.

        Returns:
            The generated text content.

        Raises:
            AgentGenerationError: If generation fails after retries.
        """
        # Add extra logging for Agent 2
        if "Agent 2" in self.name:
            logger.info(f"==== AGENT2 API CALL DEBUG ====")
            logger.info(f"  Agent name: {self.name}")
            logger.info(f"  Model being sent to API: {self.model}")
            logger.info(f"============================")
            
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.role},
                {"role": "user", "content": prompt}
            ]
        }
        # Only include max_tokens if it's a positive integer
        if max_tokens is not None:
            try:
                mt_int = int(max_tokens)
            except (TypeError, ValueError):
                mt_int = None
            if mt_int and mt_int > 0:
                payload["max_tokens"] = mt_int

        backoff_time = API_INITIAL_BACKOFF
        last_exception = None
        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                async with semaphore:
                    response = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "linktr.ee/mindrocket",
                            "X-Title": "MOA-DeepOutputs",
                        },
                        timeout=API_TIMEOUT
                    )
                response.raise_for_status()
                data = response.json()
                if "choices" in data and data["choices"]:
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    raise AgentGenerationError(f"No choices returned by API for agent {self.name}.")
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt+1} failed for agent {self.name}: {e}")
                await asyncio.sleep(backoff_time)
                backoff_time *= 2 # Exponential backoff
        else:
            logger.error(f"Agent {self.name} ({self.model}) failed after {API_RETRY_ATTEMPTS} attempts.")
            raise AgentGenerationError(f"Agent {self.name} ({self.model}) failed after {API_RETRY_ATTEMPTS} attempts.") from last_exception

        # This part should ideally not be reached if logic is correct, but as a safeguard:
        raise AgentGenerationError(f"Agent {self.name} ({self.model}) failed unexpectedly to generate a response.")

# --- Mixture of Agents Implementation ---
class MixtureOfAgents:
    def __init__(self, models: List[str], num_layers: int = 2, include_deep_research: bool = True):
        """
        include_deep_research: whether to run the Deep Research Agent in layer 1.
        """
        self.include_deep_research = include_deep_research
        if not models:
            raise ValueError("At least one agent model must be provided.")

        self.agents = [OpenRouterAgent(f"Agent {i+1}", model, f"Role {i+1}") for i, model in enumerate(models)]
        # Add Deep Research Agent (Layer 1 only) if enabled
        self.deep_research_agent = OpenRouterAgent("Deep Research Agent", DEEP_RESEARCH_AGENT_MODEL, "Deep Research Role") if self.include_deep_research else None
        self.num_layers = num_layers
        self.synthesis_agent = OpenRouterAgent("Synthesis Agent", SYNTHESIS_AGENT_MODEL, "Synthesizer Role")
        self.devils_advocate_agent = OpenRouterAgent("Devil's Advocate Agent", DEVILS_ADVOCATE_AGENT_MODEL, "Devil's Advocate Role")
        self.final_agent = OpenRouterAgent("Final Agent", FINAL_AGENT_MODEL, "Final Decision Role")

        # Debug: Print what models each agent is using
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
        deep_research_responses_for_utilization = []  # Track Deep Research Agent responses if enabled

        # current_context = "" # Removed as context is built per layer
        last_synthesis = ""
        last_devils_advocate = ""

        for i in range(self.num_layers):
            # Placeholder for Deep Research Agent response in Layer 1
            deep_research_response = ""
            layer_num = i + 1
            logger.info(colored(f"\n* Layer {layer_num} started", "cyan"))

            layer_prompt = self.create_layer_prompt(prompt, last_synthesis, last_devils_advocate, i)
            logger.debug(f"Layer {layer_num} Prompt:\n{layer_prompt}")

            # --- Step 1: Initial Responses ---
            if i == 0:
                # Step 1: Initial Responses with optional Deep Research Agent
                if self.include_deep_research and self.deep_research_agent:
                    agents_to_run = self.agents + [self.deep_research_agent]
                    initial_all = await self.run_agents_concurrently(agents_to_run, layer_prompt, "initial response", INITIAL_MAX_TOKENS)
                    # Separate deep research response
                    initial_responses = initial_all[:-1]
                    deep_research_response = initial_all[-1]
                    # Record base agents' responses for utilization
                    for agent_idx, response in enumerate(initial_responses):
                        all_agent_responses_for_utilization[agent_idx].append(response)
                    # Track Deep Research Agent's response for utilization
                    deep_research_responses_for_utilization.append(deep_research_response)
                    logger.info(colored(f"* Layer {layer_num} initial responses received ({len(initial_all)} agents, including Deep Research)", "green"))
                else:
                    initial_responses = await self.run_agents_concurrently(self.agents, layer_prompt, "initial response", INITIAL_MAX_TOKENS)
                    deep_research_response = ""
                    for agent_idx, response in enumerate(initial_responses):
                        all_agent_responses_for_utilization[agent_idx].append(response)
                    logger.info(colored(f"* Layer {layer_num} initial responses received ({len(initial_responses)} agents, deep research disabled)", "green"))
            else:
                initial_responses = await self.run_agents_concurrently(self.agents, layer_prompt, "initial response", INITIAL_MAX_TOKENS)
                for agent_idx, response in enumerate(initial_responses):
                    all_agent_responses_for_utilization[agent_idx].append(response)
                logger.info(colored(f"* Layer {layer_num} initial responses received ({len(initial_responses)} agents)", "green"))

            # --- Step 2: Aggregation & Peer Review ---
            # --- Step 2: Aggregation & Peer Review ---
            # Include Deep Research output in aggregation for Layer 1 if enabled
            agg_inputs = initial_responses + ([deep_research_response] if i == 0 and self.include_deep_research else [])
            aggregation_responses = await self.run_agents_concurrently(
                self.agents,
                self.create_aggregation_prompt(prompt, layer_prompt, agg_inputs),
                "aggregation",
                AGGREGATION_MAX_TOKENS
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

            # Build names list for initial responses, include Deep Research if enabled
            init_names = [agent.name for agent in self.agents]
            init_responses = initial_responses
            if layer_num == 1 and self.include_deep_research and self.deep_research_agent:
                init_names.append(self.deep_research_agent.name)
                init_responses = initial_responses + [deep_research_response]

            layer_details.append({
                "layer_number": layer_num,
                "layer_prompt_details": "Original Prompt" if i == 0 else "Original Prompt + Synthesis/Critique from Layer " + str(i),
                "initial_responses": list(zip(init_names, init_responses)),
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
        utilization = self.calculate_utilization(
            all_agent_responses_for_utilization,
            final_response,
            deep_research_responses_for_utilization if self.include_deep_research and self.deep_research_agent else None
        )

        return layer_details, final_response, utilization

    async def run_agents_concurrently(self, agents: List[OpenRouterAgent], prompt: str, task_description: str, max_tokens: int = 2000) -> List[str]:
        """Runs multiple agents concurrently on the same prompt."""
        tasks = [agent.generate(prompt, self.client, self.semaphore, max_tokens) for agent in agents]
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
        """Creates a streamlined, LLM-friendly prompt for agents in a specific layer."""
        if layer_index == 0:
            # Add clear instructions for the first layer
            return f"""You are an expert AI agent. Your task is to answer the following user prompt as clearly and insightfully as possible, using sound reasoning and, if relevant, calculations or examples.

User Prompt:
{original_prompt}

Please provide a well-structured, direct answer. If there are ambiguities, state your assumptions."""
        else:
            context_header = f"--- Previous Layer Context (Layer {layer_index}) ---"
            synthesis_text = f"Synthesis (summary so far):\n{prev_synthesis}\n" if prev_synthesis else ""
            critique_text = f"Devil's Advocate Critique:\n{prev_devils_advocate}\n" if prev_devils_advocate else ""
            context_block = f"{context_header}\n\n{synthesis_text}{critique_text}---\n\n" if (synthesis_text or critique_text) else ""
            return f"""You are an expert AI agent. Your task is to answer the user's original prompt, taking into account the analysis and critique from the previous layer.

User Prompt:
{original_prompt}

{context_block}Instructions for Layer {layer_index + 1}:
- Carefully read the user prompt.
- Consider the synthesis and critique above (if present) as context.
- Generate your own independent answer to the user prompt, not just a rephrasing of prior outputs.
- Briefly explain if/how the previous context changed your answer, or if you disagree with it.
- Ensure your answer is clear, well-reasoned, and addresses the main question.
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
            "synthesis": self.synthesis_agent.generate(synthesis_prompt, self.client, self.semaphore, SYNTHESIS_MAX_TOKENS),
            "devils_advocate": self.devils_advocate_agent.generate(devils_advocate_prompt, self.client, self.semaphore, DEVILS_ADVOCATE_MAX_TOKENS)
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
---
Analyze the full text of the "Aggregated/Reviewed Responses" provided above and deliver one unified, richly detailed synthesis that sets the stage for the next phase of analysis. Your output should include:

1. **Core Insights:**  
   - Thoroughly extract and explain the primary conclusions, findings, and proposed solutions.  
   - Illustrate how these insights interconnect or build on each other.

2. **Consensus & Divergence:**  
   - Map out where the responses strongly agree—citing specific points or language—and where they diverge or present conflicting perspectives.  
   - For each major disagreement, briefly describe the reasoning on each side.

3. **Confidence Levels & Uncertainties:**  
   - Highlight areas where the aggregated responses express high confidence (e.g., recurring evidence, consistent recommendations).  
   - Pinpoint concepts or data points that remain ambiguous, under‑supported, or contested.

4. **Outstanding Questions & Gaps:**  
   - Identify any aspects of the Original User Prompt that still lack clarity or need deeper probing.  
   - Formulate explicit questions or objectives that the next analytical layer should address.

5. **Expansive Synthesis Narrative:**  
   - In a multi‑paragraph narrative (not limited to two or three), weave together the above elements into a coherent story of what's known, what's debated, and what's missing.  
   - Use transitions and headings (if helpful) to guide the reader through each section's logical flow.

6. **Next‑Layer Roadmap:**  
   - Conclude with a detailed outline of the next analytical steps—methodologies, data sources, or stakeholder inputs—that will resolve the open questions and drive toward a final answer.

**Do NOT** simply list bullet points. Craft a prose narrative that captures both the nuance and the breadth of the discourse, fully preparing readers for the deeper dive to come.
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
            final_response = await self.final_agent.generate(final_prompt, self.client, self.semaphore, FINAL_MAX_TOKENS) # Allow more tokens for final
            if final_response and len(final_response.strip()) > 50: # Basic check for meaningful content
                 logger.info(f"Final response generated successfully by {self.final_agent.name}.")
                 return final_response.strip()
            else:
                 logger.warning(f"Final agent {self.final_agent.name} returned short or empty response. Trying fallback.")
                 raise AgentGenerationError("Final response was empty or too short")

        except Exception as e:
            logger.error(colored(f"Error generating final response with {self.final_agent.name}: {e}", "red"), exc_info=True)
            print(f"An unexpected error occurred: {str(e)}")
            logger.warning(colored("Attempting fallback methods...", "yellow"))

            # Fallback 1: Try Final Agent with a simpler prompt
            try:
                logger.info("Fallback 1: Trying Final Agent with simplified prompt.")
                simplified_prompt = f"""Original User Prompt: {original_prompt}\n\nBased on extensive multi-agent analysis (including critiques), provide your own definitive and comprehensive answer to the Original User Prompt. Be clear, logical, and directly address all aspects of the original request."""
                fallback_response = await self.final_agent.generate(simplified_prompt, self.client, self.semaphore, FINAL_MAX_TOKENS)
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
                    synthesis_fallback = await self.synthesis_agent.generate(simplified_prompt, self.client, self.semaphore, FINAL_MAX_TOKENS)
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


    def calculate_utilization(self, all_agent_responses_for_utilization: List[List[str]], final_response: str, deep_research_responses_for_utilization: list = None) -> Dict[str, float]:
        """
        Calculates the 'utilization' of each base agent (and Deep Research Agent if present) based on the similarity
        of their combined contributions (initial + aggregation per layer) to the final response.
        Note: This is a heuristic measure.
        """
        utilization = {agent.name: 0.0 for agent in self.agents}
        if deep_research_responses_for_utilization is not None:
            utilization["Deep Research Agent"] = 0.0
        total_similarity = 0.0

        if not final_response or final_response.startswith("Apologies,"): # Don't calculate if final response failed
             logger.warning("Skipping utilization calculation due to failed final response.")
             return utilization

        # Base agents
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

        # Deep Research Agent
        if deep_research_responses_for_utilization is not None:
            dr_combined_text = " ".join(deep_research_responses_for_utilization)
            if not dr_combined_text.strip():
                dr_similarity = 0.0
            else:
                dr_similarity = SequenceMatcher(None, dr_combined_text, final_response, autojunk=False).ratio()
            utilization["Deep Research Agent"] = dr_similarity
            total_similarity += dr_similarity
            logger.debug(f"Deep Research Agent combined text length: {len(dr_combined_text)}, Similarity to final: {dr_similarity:.4f}")

        if total_similarity == 0:
            # Avoid division by zero and assign equal weight if no similarity found (or only 1 agent)
            num_agents = len(self.agents) + (1 if deep_research_responses_for_utilization is not None else 0)
            if num_agents > 0:
                logger.warning("Total similarity is zero. Assigning equal utilization.")
                equal_share = 100.0 / num_agents
                return {name: equal_share for name in utilization}
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
    final_agent_name: str,
    moa: MixtureOfAgents
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

{sanitize_for_markdown(final_response)}
"""
    return markdown_content

def generate_detailed_markdown_report(
    prompt: str,
    layer_details: List[Dict[str, Any]],
    final_response: str,
    utilization: Dict[str, float],
    final_agent_name: str,
    synthesis_agent_name: str,
    devils_advocate_agent_name: str,
    moa: MixtureOfAgents
) -> str:
    """Generates a detailed comprehensive Markdown report with all intermediate outputs."""
    # Fetch model names for special agents
    synthesis_model = getattr(moa.synthesis_agent, 'model', 'unknown')
    devils_advocate_model = getattr(moa.devils_advocate_agent, 'model', 'unknown')
    final_agent_model = getattr(moa.final_agent, 'model', 'unknown')

    markdown_content = f"""# MoA Detailed Response Report

## Original Prompt
> {prompt}

## Agent Utilization
{chr(10).join(
    [f"- {agent.name}: {utilization.get(agent.name, 0.0):.2f}%" for agent in moa.agents] +
    ([f"- Deep Research Agent: {utilization['Deep Research Agent']:.2f}%"] if 'Deep Research Agent' in utilization else []) +
    [f"- {agent_name}: {percentage:.2f}%" for agent_name, percentage in utilization.items()
     if agent_name not in [agent.name for agent in moa.agents] and agent_name != "Deep Research Agent"]
)}

*(Note: Utilization is a heuristic based on text similarity to the final output.{' Deep Research Agent utilization is included if present.' if 'Deep Research Agent' in utilization else ''})*

## Intermediate Outputs
"""

    for layer in layer_details:
        layer_num = layer["layer_number"]
        markdown_content += f"""
### Layer {layer_num}

#### Layer Prompt
> {layer["layer_prompt_details"]}

#### Step 1 - Agents Initial Responses
"""
        # Ensure initial_responses is a list of tuples/lists before trying to unpack
        if layer.get("initial_responses") and isinstance(layer["initial_responses"], list):
            for item in layer["initial_responses"]:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    agent_name, response = item
                    agent_model = next((agent.model for agent in moa.agents if agent.name == agent_name), "unknown")
                    markdown_content += f"""\n##### {agent_name} - `{agent_model}`\n\n{sanitize_for_markdown(response)}\n"""
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
                    agent_model = next((agent.model for agent in moa.agents if agent.name == agent_name), "unknown")
                    markdown_content += f"""\n##### {agent_name} - `{agent_model}`\n\n{sanitize_for_markdown(response)}\n"""
                 else:
                      markdown_content += f"\n_Invalid format for aggregation response item: {item}_"
        else:
               markdown_content += "\n_No aggregation responses recorded or invalid format._"


        markdown_content += f"""\n#### Step 3 - Synthesized Aggregated Responses (Synthesis Agent: {synthesis_agent_name} - `{synthesis_model}`)

##### Synthesis

{sanitize_for_markdown(layer.get("synthesis", "N/A"))}

##### Devil's Advocate (Agent: {devils_advocate_agent_name} - `{devils_advocate_model}`)

{sanitize_for_markdown(layer.get("devils_advocate", "N/A"))}

---
"""

    markdown_content += f"""
## Information Passed to Final Response Agent

The following synthesized information from all layers, along with the original user prompt, was passed to the final response agent ({final_agent_name} - `{final_agent_model}`). The final agent used this information to generate the final MoA response.
"""
    for layer in layer_details:
        layer_num = layer["layer_number"]
        markdown_content += f"""
### Layer {layer_num} Synthesis

{sanitize_for_markdown(layer.get("synthesis", "N/A"))}

### Layer {layer_num} Devil's Advocate

{sanitize_for_markdown(layer.get("devils_advocate", "N/A"))}

---
"""

    # Add Final MoA Response at the end
    markdown_content += f"""
## Final MoA Response
**Final Response Agent:** {final_agent_name} - `{final_agent_model}`

{sanitize_for_markdown(final_response)}
"""

    return markdown_content.strip()

def generate_detailed_markdown_report_gfm(
    prompt: str,
    layer_details: List[Dict[str, Any]],
    final_response: str,
    utilization: Dict[str, float],
    final_agent_name: str,
    synthesis_agent_name: str,
    devils_advocate_agent_name: str,
    moa: 'MixtureOfAgents'
) -> str:
    """Generates a detailed Markdown report with GitHub-flavored markdown formatting."""
    # Fetch model names for special agents
    synthesis_model = getattr(moa.synthesis_agent, 'model', 'unknown')
    devils_advocate_model = getattr(moa.devils_advocate_agent, 'model', 'unknown')
    final_agent_model = getattr(moa.final_agent, 'model', 'unknown')

    markdown_content = f"""# MoA Detailed Response Report (GitHub-Flavored)

## Original Prompt
> {prompt}

## Agent Utilization
{chr(10).join([f"- {agent_name}: {percentage:.2f}%" for agent_name, percentage in utilization.items()])}

*(Note: Utilization is a heuristic based on text similarity to the final output)*

## Intermediate Outputs
"""
    for layer in layer_details:
        layer_num = layer["layer_number"]
        markdown_content += f"""
### Layer {layer_num}

#### Layer Prompt
> {layer["layer_prompt_details"]}

#### Step 1 - Agents Initial Responses
"""
        if layer.get("initial_responses") and isinstance(layer["initial_responses"], list):
            for item in layer["initial_responses"]:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    agent_name, response = item
                    agent_model = next((agent.model for agent in moa.agents if agent.name == agent_name), "unknown")
                    markdown_content += f"""\n##### {agent_name} - `{agent_model}`\n\n{sanitize_for_markdown(response)}\n"""
                else:
                    markdown_content += f"\n_Invalid format for initial response item: {item}_"
        else:
            markdown_content += "\n_No initial responses recorded or invalid format._"
        
        markdown_content += "\n#### Step 2 - Agent Aggregation of All Responses\n"
        
        if layer.get("aggregation_responses") and isinstance(layer["aggregation_responses"], list):
            for item in layer["aggregation_responses"]:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    agent_name, response = item
                    agent_model = next((agent.model for agent in moa.agents if agent.name == agent_name), "unknown")
                    markdown_content += f"""\n##### {agent_name} - `{agent_model}`\n\n{sanitize_for_markdown(response)}\n"""
                else:
                    markdown_content += f"\n_Invalid format for aggregation response item: {item}_"
        else:
            markdown_content += "\n_No aggregation responses recorded or invalid format._"
        
        markdown_content += f"""\n#### Step 3 - Synthesized Aggregated Responses (Synthesis Agent: {synthesis_agent_name} - `{synthesis_model}`)

##### Synthesis

{sanitize_for_markdown(layer.get('synthesis', 'N/A'))}

##### Devil's Advocate (Agent: {devils_advocate_agent_name} - `{devils_advocate_model}`)

{sanitize_for_markdown(layer.get('devils_advocate', 'N/A'))}

---
"""

    markdown_content += f"""\n## Information Passed to Final Response Agent

The following synthesized information from all layers, along with the original user prompt, was passed to the final response agent ({final_agent_name} - `{final_agent_model}`). The final agent used this information to generate the final MoA response.
"""
    for layer in layer_details:
        layer_num = layer["layer_number"]
        markdown_content += f"""\n### Layer {layer_num} Synthesis

{sanitize_for_markdown(layer.get('synthesis', 'N/A'))}

### Layer {layer_num} Devil's Advocate

{sanitize_for_markdown(layer.get('devils_advocate', 'N/A'))}

---
"""

    # Add Final MoA Response at the end
    markdown_content += f"""
## Final MoA Response
**Final Response Agent:** {final_agent_name} - `{final_agent_model}`

{sanitize_for_markdown(final_response)}
"""

    return markdown_content.strip()

# In main(), after generating the detailed report, also generate the GFM version and save both
async def main():
    """Main function to run the Mixture of Agents model with prompt from file."""
    file_handler = None
    moa = None
    try:
        # --- Setup ---
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # FORCE CORRECT MODEL FOR AGENT 2
        logger.info("Forcing correct model for Agent 2 to deepseek/deepseek-chat-v3-0324")
        os.environ["AGENT2_MODEL"] = "deepseek/deepseek-chat-v3-0324"
        
        # Read prompt from file
        prompt = read_prompt_from_file()
        # Ask user whether to include Deep Research Agent responses
        include_input = input("Include deep research (Y/N)? ")
        include_deep_research = include_input.strip().lower() == 'y'
        logger.info(f"Include deep research: {include_deep_research}")
        # Disable deep research if no model configured
        if include_deep_research and not DEEP_RESEARCH_AGENT_MODEL:
            logger.warning("Deep research model not configured; disabling Deep Research Agent.")
            include_deep_research = False
        
        # Generate base name for folder/files from prompt
        base_filename = sanitize_filename(prompt)
        logger.info(f"Using base name for run: {base_filename}")
        
        # Create run-specific reports directory (now with timestamped subfolder)
        reports_base_dir = Path("reports")
        run_reports_dir = reports_base_dir / f"{base_filename}_{timestamp}"
        run_reports_dir.mkdir(parents=True, exist_ok=True)

        # --- Configure File Logging ---
        log_file_path = run_reports_dir / f"{base_filename}_logs_{timestamp}.log"
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file_path}")
        
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
        models_list = [AGENT1_MODEL, AGENT2_MODEL, AGENT3_MODEL]
        logger.info(f"Models to be used: {models_list}")
        
        # HARDCODE fix for Agent 2 if needed
        if models_list[1] == "thudm/glm-z1-rumination-32b":
            logger.warning("Detected incorrect model for Agent 2, forcing correct model")
            models_list[1] = "deepseek/deepseek-chat-v3-0324"
            logger.info(f"Updated models list: {models_list}")
            
        moa = MixtureOfAgents(
            models=models_list,
            num_layers=num_layers, # Use configured number of layers
            include_deep_research=include_deep_research
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
            final_agent_name=moa.final_agent.name,
            moa=moa
        )
        detailed_report = generate_detailed_markdown_report(
            prompt=prompt,
            layer_details=layer_details,
            final_response=final_response,
            utilization=utilization,
            final_agent_name=moa.final_agent.name,
            synthesis_agent_name=moa.synthesis_agent.name,
            devils_advocate_agent_name=moa.devils_advocate_agent.name,
            moa=moa
        )
        detailed_report_gfm = generate_detailed_markdown_report_gfm(
            prompt=prompt,
            layer_details=layer_details,
            final_response=final_response,
            utilization=utilization,
            final_agent_name=moa.final_agent.name,
            synthesis_agent_name=moa.synthesis_agent.name,
            devils_advocate_agent_name=moa.devils_advocate_agent.name,
            moa=moa
        )
        logger.info("Reports generated.")

        # --- Save Reports ---
        final_report_path = run_reports_dir / f"{base_filename}_final_report_{timestamp}.md"
        detailed_report_path = run_reports_dir / f"{base_filename}_detailed_report_{timestamp}.md"
        detailed_report_gfm_path = run_reports_dir / f"{base_filename}_detailed_report_gfm_{timestamp}.md"
        
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

        try:
            with open(detailed_report_gfm_path, "w", encoding="utf-8") as f:
                f.write(detailed_report_gfm)
            logger.info(f"GFM detailed report saved to {detailed_report_gfm_path}")
        except Exception as e:
            logger.error(f"Error saving GFM detailed report to {detailed_report_gfm_path}: {e}", exc_info=True)

        # --- Output to Console ---
        try:
            # Use utf-8 encoding for printing to handle all Unicode characters
            import sys
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass  # If reconfigure fails (older Python), fallback to default behavior
        print(f"\n--- MoA Run Complete ---")
        print(f"Run Folder: {run_reports_dir}")
        print(f"Logs: {log_file_path}")
        print(f"Final Report: {final_report_path}")
        print(f"Detailed Report: {detailed_report_path}")
        print(f"GFM Detailed Report: {detailed_report_gfm_path}")
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
