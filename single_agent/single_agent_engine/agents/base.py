class SingleAgentGenerationError(Exception):
    """Custom exception for single agent generation failures."""
    pass

class SingleAgent:
    """Base class for single agents with specialized cognitive functions."""

    def __init__(self, name: str, model: str, role: str, cognitive_function: str):
        """
        Initialize a single agent.

        Args:
            name: Human-readable name for the agent
            model: Model identifier (e.g., "anthropic/claude-3.5-sonnet")
            role: Specific role description
            cognitive_function: Cognitive function this agent performs
                              ("response", "devils_advocate", "synthesis", "final")
        """
        self.name = name
        self.model = model
        self.role = role
        self.cognitive_function = cognitive_function

    async def generate(self, prompt: str, client, semaphore, max_tokens: int, **kwargs):
        """
        Generate a response using this agent.

        Args:
            prompt: The prompt to respond to
            client: HTTP client for API calls
            semaphore: Concurrency control semaphore
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for specific agent types

        Returns:
            Generated response string

        Raises:
            SingleAgentGenerationError: If generation fails
        """
        raise NotImplementedError("Subclasses must implement generate method")

    def get_agent_info(self) -> dict:
        """Get information about this agent."""
        return {
            "name": self.name,
            "model": self.model,
            "role": self.role,
            "cognitive_function": self.cognitive_function
        }