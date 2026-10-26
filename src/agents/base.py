# Inspired by: https://github.com/balrog-ai/BALROG/blob/main/balrog/agents/base.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base class for agents."""

    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

class BaseCodeAgent(BaseAgent, ABC):
    """Base class for agents using dict-based observations and code as policies."""

    def __init__(self):
        pass

    @abstractmethod
    def act(self, obs: dict) -> str:
        """Generate an action based on the observation."""
        raise NotImplementedError

class BaseTextAgent(BaseAgent, ABC):
    """Base class for agents using text-based observations."""

    def __init__(self):
        pass

    @abstractmethod
    def act(self, obs: str, legal_actions: list[str]) -> str:
        """Generate an action based on the observation."""
        # TODO: For now, we assume that we are given a list of all legal actions. 
        # TODO: In the future, we will have to think about how we want to prompt for actions.
        # TODO: e.g. What should the game supply to the agent? Every turn or once per game?
        raise NotImplementedError
