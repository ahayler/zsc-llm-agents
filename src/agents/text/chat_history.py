# DEPRECATED: Pseudo-code policies are no longer maintained. Will be removed at a later point.
from agents.base import BaseTextAgent
from llm.parser import request_and_parse_to_json
from configs.configs import ChatHistoryAgentConfig
from core import Message
from utils.history import GameHistoryDeque
from llm.request import create_client
import random

general_prompt = """
You are playing a cooperative game with another player. This game is generally turn-based. While you cannot communicate with the other player, you can see their actions and the (joined) rewards from previous turns. Your goal is to maximize your (joint) reward. 
Note that you are only playing one round of the game (one round consists of multiple turns).

Now follows a detailed description of the game and (if applicable) the history of previous turns.
"""
random_number_prompt = """
If you even need to or want to randomize, please use the following randomly generated number between 0 and 1 to help you make a truely random choice:
{random_number:.2f}
"""
action_prompt = """
You will now choose an action for the current turn. You must choose one of the following legal actions:

{legal_actions}

{random_number_str}

{cfd_str}

Respond in the following format:

REASONING:
<REASONING>

ACTION JSON:
```json
<JSON>
```

In <REASONING>, provide a short explanation of your choice of action.

In <JSON>, provide the action you are going to take in this round. The JSON should have the following format:
- "action": one of the legal actions. Make sure to use the exact string from the legal actions list or else the JSON will not be parsed correctly.

This JSON will be automatically parsed, so ensure the format is precise and the action matches one of the legal actions exactly.
"""

CFD_PROMPT = """
Ensure that the action you take is strictly compliant with the following Coordination-friendly definition (CFD):
{cfd_str}
"""

class ChatHistoryAgent(BaseTextAgent):
	"""An agent that generates actions based on an action-observation history encoded as a chat history."""

	def __init__(
			self, 
			game_description: str,
			use_langfuse: bool,
			config: ChatHistoryAgentConfig
			):
		"""Initialize the NaiveAgent with a client and prompt builder."""
		super().__init__()
		self.game_description = game_description
		self.client, self.model = create_client(config.llm_config.model, use_langfuse=use_langfuse)
		self.history = GameHistoryDeque()
		self.config = config

	def _generate_action_message(self, legal_actions: list[str]):
		legal_actions_str = "\n".join([f"{action}" for action in legal_actions])
		random_number_str = random_number_prompt.format(random_number=random.random()) if self.config.include_random_number else ""
		cfd_str = ""
		if getattr(self.config, "cfd_str", "") and self.config.cfd_str.strip():
			cfd_str = CFD_PROMPT.format(cfd_str=self.config.cfd_str.strip())

		return Message(
			role="user",
			content=action_prompt.format(
				legal_actions=legal_actions_str,
				random_number_str=random_number_str,
				cfd_str=cfd_str,
			),
		)

	def act(self, obs, legal_actions: list[str]):
		"""Generate the next action based on the observation and previous action.

		Args:
			obs (dict): The current observation in the environment.
			legal_actions (list[str], optional): The list of legal actions.

		Returns:
			str: The selected action from the LLM response.
		"""
		self.history.add_observation(obs)

		messages = self.history.get_messages(max_history=self.config.max_history, verbose_actions=self.config.verbose_actions)

		instruction = general_prompt + "\n" + self.game_description
		instruction_message = Message(role="user", content=instruction)

		action_message = self._generate_action_message(legal_actions)

		messages = [instruction_message, *messages, action_message] # The first element is the oldest message
		json_dict, response = request_and_parse_to_json(
			self.client, 
			self.model, 
			msg_history=messages, 
			max_tokens=self.config.llm_config.max_tokens, 
			temperature=self.config.llm_config.temperature,
			reasoning_effort=self.config.llm_config.reasoning_effort,
			required_fields=["action"],
		)

		final_answer = json_dict['action']

		self.history.add_action(message=response.content, action=final_answer)
		return final_answer

	def reset(self):
		"""Reset the agent."""
		self.history.reset()