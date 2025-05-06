import asyncio
import httpx
from deepoutputs_engine.config import (
    OPENROUTER_API_KEY, API_TIMEOUT, API_RETRY_ATTEMPTS, API_INITIAL_BACKOFF
)
from deepoutputs_engine.config import logger
from deepoutputs_engine.agents.base import Agent, AgentGenerationError

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