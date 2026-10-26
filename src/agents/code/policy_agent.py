import copy
import re
import time
import math
import logging
import types
import ast
import collections
import concurrent.futures
from utils.containers import is_docker_container, is_root_user
from agents.base import BaseTextAgent, BaseCodeAgent
import random
from core import Message
from typing import Optional, Tuple
from configs.configs import CodePolicyAgentConfig
from utils.log_utils import get_logger

logger = get_logger(__name__)

general_prompt = """
You are playing a cooperative game with another player. This game is generally turn-based. 
While you cannot communicate with the other player, you can see your current and past observations and actions.
These include information about the game state and the actions of other players.
Note that you are only playing one round of the game (one round consists of multiple turns).

Now follows a detailed description of the game and (if applicable) the history of previous turns.
"""

action_prompt = """
You will be first provided with a detailed description of the game and a policy. Your goal is to follow/execute this policy: 
{policy}

You will now choose an action for the current turn. 
Based on the instruction from the policy, think carefully about what is your best action should be taken.
You MUST choose one of the following legal actions (exactly as they are written): {{{legal_actions}}}.
Here is a random number between 0 and 1 (ignore it, if you don't need it): {random_number:.2f}.
Otherwise the environment will not be able to interpret your move and the game fails.

**Output Specification**
In <REASONING>, provide a short explanation of your choice of action.

In <JSON>, provide the action you are going to take in this round. The JSON should have the following format:
{{
	"action": ONE of the legal actions from {{{legal_actions}}}. Make sure to use the exact string from the legal actions set.
}}

REASONING:
<REASONING>

ACTION JSON:
```json
<JSON>
```

"""



class CodePolicyAgent(BaseCodeAgent):
	"""An agent that generates actions based on observations using code as policies."""

	def __init__(self, agent_config: CodePolicyAgentConfig, allowed_modules=None):
		#Safety check (Deactivated for debugging purposes)
		if not (is_docker_container() and not is_root_user()):
			raise RuntimeError(
				f"""Policy agents execute AI generated code. You should run your code within a docker container and as a non-root user.
We currently detect the following configuration:
Running in a container: {is_docker_container()} ({'✅' if is_docker_container() else '❌'})
Running as root user: {is_root_user()} ({'✅' if not is_root_user() else '❌'})
Please refer to the README.md for more information regarding the container setup.
				"""
			)
		
		super().__init__()
		self.policy_code_str = agent_config.policy_str
		self.failed_to_interpret = False
		self.failed_to_compile = False
		self.memory = {}
		
		# Set allowed modules - default to common safe modules
		self.allowed_modules = allowed_modules or ['random', 'math', 'collections', 'itertools', 'functools', 'time', 'typing', 'numpy', 'hashlib']
		
		try:
			# Create execution environment with restricted imports
			# Handle __builtins__ being either a module or dict
			if isinstance(__builtins__, dict):
				builtins_dict = __builtins__.copy()
			else:
				builtins_dict = __builtins__.__dict__.copy()
			
			# Override __import__ with our restricted version
			builtins_dict['__import__'] = self._restricted_import

			try:
				policy_code_str_unescaped = self.policy_code_str.encode().decode('unicode_escape') # This is to handle newline characters /n in the policy code
			except Exception as e:
				logger.error(f"Failed to unescape policy code: {e} for policy: \n {self.policy_code_str}")
				policy_code_str_unescaped = self.policy_code_str
			
			globals_dict = {
				'__builtins__': builtins_dict
			}
			locals_dict = {}
			
			# Pre-import commonly needed modules/objects to make them automatically available
			try:
				from typing import Tuple
				globals_dict['Tuple'] = Tuple
				locals_dict['Tuple'] = Tuple
			except ImportError:
				pass  # typing module not available
			
			# Execute the policy code directly
			exec(policy_code_str_unescaped, globals_dict, locals_dict)
			self.act_fn = locals_dict['act']
			
			# Preserve the complete execution environment after exec
			# This captures ALL imports, variables, and objects created during execution
			self.execution_globals = globals_dict.copy()
			self.execution_locals = locals_dict.copy()
		except Exception as e:
			logger.error(f"Failed to compile policy code: {e} for policy: \n {self.policy_code_str}")
			self.failed_to_compile = True
			
	def reset(self):
		"""Reset the memory of the agent."""
		self.memory = {}

	def _restricted_import(self, name, *args, **kwargs):
		"""Custom import function that only allows specific modules."""
		if name in self.allowed_modules:
			return __import__(name, *args, **kwargs)

		# Check if it's a submodule of an allowed module
		for allowed_module in self.allowed_modules:
			if name.startswith(allowed_module + '.'):
				return __import__(name, *args, **kwargs)
		
		raise ImportError(f"Module '{name}' is not allowed. Allowed modules: {self.allowed_modules}")

	def act(self, obs: dict) -> str:
		"""Generate an action based on the observation."""
		if self.failed_to_compile:
			return None
			
		try:
			# Create a wrapper function that preserves the complete execution environment
			def wrapped_act_fn():
				# Restore the complete execution environment (all imports, variables, etc.)
				self.act_fn.__globals__.update(self.execution_globals)
				self.act_fn.__globals__.update(self.execution_locals)
				return self.act_fn(obs, self.memory)
			
			# Execute policy code with a 5-second timeout
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
				future = executor.submit(wrapped_act_fn)
				try:
					action, self.memory = future.result(timeout=5.0)
					return action
				except concurrent.futures.TimeoutError:
					logger.error(f"Policy code execution timed out (>5 seconds) for policy: \n {self.policy_code_str}")
					raise RuntimeError("Policy code execution timed out after 5 seconds")
		except Exception as e:
			logger.error(f"Failed to interpret policy code: {repr(e)} for policy: \n {self.policy_code_str}")
			logger.error(f"Observation: {obs}")
			logger.error(f"Memory: {self.memory}")
			self.failed_to_interpret = True
			return None

if __name__ == "__main__":
	config = CodePolicyAgentConfig()
	config.policy_str = """
def act(obs, memory) -> Tuple: 
    import random
    return random.choice(['test', 'a']), 'a'
	"""
	agent = CodePolicyAgent(config)
	print(agent.act({}))