import asyncio
import httpx
import time
from typing import Dict, Any, Optional
import os

from .base import SingleAgent, SingleAgentGenerationError
from ..config import (
    OPENROUTER_API_KEY, API_TIMEOUT, API_RETRY_ATTEMPTS,
    API_INITIAL_BACKOFF, HTTP_REFERER, X_TITLE, logger
)

try:
    from termcolor import colored
except ImportError:
    def colored(text, color):
        return text

class OpenRouterSingleAgent(SingleAgent):
    """OpenRouter-based implementation of single agents with specialized cognitive functions."""

    def __init__(self, name: str, model: str, role: str, cognitive_function: str):
        super().__init__(name, model, role, cognitive_function)

        # Load header values from environment variables
        self.api_key = OPENROUTER_API_KEY
        self.http_referer = HTTP_REFERER
        self.x_title = X_TITLE

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        # Create HTTP client
        self.client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.http_referer,
                "X-Title": self.x_title,
                "Content-Type": "application/json"
            },
            timeout=API_TIMEOUT
        )

    async def generate(self, prompt: str, client=None, semaphore=None, max_tokens: int = 4000, **kwargs) -> str:
        """
        Generate a response using OpenRouter API.

        Args:
            prompt: The prompt to respond to
            client: HTTP client (uses self.client if None)
            semaphore: Concurrency semaphore (optional)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated response string
        """
        # Use provided client or default to self.client
        http_client = client or self.client

        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
        }

        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in ["temperature", "top_p"] and not key.startswith("_"):
                payload[key] = value

        backoff = API_INITIAL_BACKOFF

        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                # Acquire semaphore if provided
                if semaphore:
                    await semaphore.acquire()

                start_time = time.time()

                logger.info(f"[{self.name}] Making API call to {self.model} (attempt {attempt + 1}/{API_RETRY_ATTEMPTS})")

                response = await http_client.post("/chat/completions", json=payload)
                response.raise_for_status()

                duration = time.time() - start_time
                data = response.json()

                if "choices" not in data or not data["choices"]:
                    raise SingleAgentGenerationError(f"No choices returned from {self.model}")

                content = data["choices"][0]["message"]["content"]
                if not content:
                    raise SingleAgentGenerationError(f"Empty content returned from {self.model}")

                # Log success
                tokens_used = len(content.split())  # Approximate token count
                logger.info(colored(
                    f"[{self.name}] API call successful - Duration: {duration:.2f}s, "
                    f"Tokens: ~{tokens_used}, Model: {self.model}",
                    "green"
                ))

                return content

            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                logger.warning(colored(f"[{self.name}] HTTP error: {error_msg}", "yellow"))

                if e.response.status_code >= 500:
                    # Server error, retry
                    if attempt < API_RETRY_ATTEMPTS - 1:
                        logger.info(colored(f"[{self.name}] Retrying in {backoff}s...", "yellow"))
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                raise SingleAgentGenerationError(f"HTTP error from {self.model}: {error_msg}")

            except httpx.RequestError as e:
                logger.warning(colored(f"[{self.name}] Request error: {str(e)}", "yellow"))

                if attempt < API_RETRY_ATTEMPTS - 1:
                    logger.info(colored(f"[{self.name}] Retrying in {backoff}s...", "yellow"))
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

                raise SingleAgentGenerationError(f"Request error from {self.model}: {str(e)}")

            except Exception as e:
                logger.error(colored(f"[{self.name}] Unexpected error: {str(e)}", "red"))
                raise SingleAgentGenerationError(f"Unexpected error from {self.model}: {str(e)}")

            finally:
                # Release semaphore if acquired
                if semaphore:
                    semaphore.release()

        raise SingleAgentGenerationError(f"All {API_RETRY_ATTEMPTS} attempts failed for {self.model}")

    async def close_client(self):
        """Close the HTTP client."""
        await self.client.aclose()
        logger.info(f"[{self.name}] HTTP client closed")