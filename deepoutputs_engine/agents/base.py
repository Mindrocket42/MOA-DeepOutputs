class AgentGenerationError(Exception):
    """Custom exception for agent generation failures."""
    pass

class Agent:
    """Base class for agents."""
    def __init__(self, name: str, model: str, role: str):
        self.name = name
        self.model = model
        self.role = role

    async def generate(self, prompt: str, client, semaphore, max_tokens: int):
        raise NotImplementedError