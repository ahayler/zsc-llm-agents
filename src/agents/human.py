from agents.base import BaseTextAgent
from typing import Optional
from llm.request import LLMType

class HumanInputAgent(BaseTextAgent):   
    def __init__(self, max_history: Optional[int] = 16):
        super().__init__(LLMType.HUMAN, "", max_history)

    def act(self, obs, legal_actions: list[str]):
        print("Current observation:")
        print(obs)
        print("Legal actions:")
        print(legal_actions)
        while (action := input("Enter your action: ")) not in legal_actions:
            print("Invalid action. Please try again.")
        return action
