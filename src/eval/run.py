from llm.request import LLMType
from envs.base import Env
from agents.base import BaseTextAgent, BaseCodeAgent, BaseAgent
from typing import Any, List, Tuple, Dict
from utils.log_utils import get_logger
from abc import ABC, abstractmethod

from core import CompactRolloutHistory

logger = get_logger(__name__)

import math

class GameRun(ABC):
    def __init__(self, env: Env, agents: List[BaseAgent]):
        self.env = env
        self.agents = agents

        self.rewards = []
        self.observation_history = {0: [], 1: []}
        self.action_history = {0: [], 1: []} 

        self._compact_rollout_history = None

    @property
    def compact_rollout_history(self) -> CompactRolloutHistory:
        if self._compact_rollout_history is None:
            raise ValueError("Please first run the game before requesting the compact rollout history.")
        return self._compact_rollout_history

    def game_run_failed(self, error_msg: str, crash_cause_idx: int):
        self.rewards.append(math.nan)
        self._compact_rollout_history = self.env.get_compact_rollout_history(ignore_not_done=True)
        self._compact_rollout_history.rewards_sum = math.nan
        self._compact_rollout_history.crashed = True
        self._compact_rollout_history.error = error_msg
        self._compact_rollout_history.crash_cause_idx = crash_cause_idx

        logger.warning(f"Game run failed: {error_msg}")
        return

    def pre_game_checks(self) -> bool:
        return True

    @abstractmethod
    def get_obs_and_act(self) -> Tuple[int, Any, List[str], str]:
        """
        Returns the current player id, the observation, the legal actions and the action.
        """
        pass

    def run(self):
        if not self.pre_game_checks():
            return
        
        # Reset the memory of the agents
        for agent in self.agents:
            agent.reset()
        
        i = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            i += 1

            current_id, obs, legal_actions, act = self.get_obs_and_act()

            self.observation_history[current_id].append(obs)
            self.action_history[current_id].append(act)

            if act is None:
                self.game_run_failed("Agent failed to act. Skipping game run.", crash_cause_idx=current_id)
                return

            try:
                act = str(act)
            except:
                self.game_run_failed("Action could not be converted to string.", crash_cause_idx=current_id)
                return
            
            if act not in legal_actions:
                self.game_run_failed(f"Agent acted with invalid action: {act}", crash_cause_idx=current_id)
                return
            else:
                reward, terminated, truncated, info = self.env.step(act)

            self.rewards.append(reward)

        # This is the history we are logging to langfuse
        self._compact_rollout_history = self.env.get_compact_rollout_history()

class TextGameRun(GameRun):
    def __init__(self, env: Env, agents: List[BaseTextAgent]):
        if not isinstance(agents, list) or not all(isinstance(agent, BaseTextAgent) for agent in agents):
            raise ValueError("agents must be a list of BaseTextAgent instances")
        
        super().__init__(env, agents)

    def get_obs_and_act(self) -> Tuple[int, Any, List[str], str]:
        current_id, obs_text, legal_actions = self.env.text_observation
        act = self.agents[current_id].act(obs_text, legal_actions)
        return current_id, obs_text, legal_actions, act

        
class CodeGameRun(GameRun):
    def __init__(self, env: Env, agents: List[BaseCodeAgent]):
        if not isinstance(agents, list) or not all(isinstance(agent, BaseCodeAgent) for agent in agents):
            raise ValueError("agents must be a list of BaseCodeAgent instances")
        
        super().__init__(env, agents)

    def check_compile_errors(self):
        for i, agent in enumerate(self.agents):
            if agent.failed_to_compile:
                self.game_run_failed(f"Compile error in agent {i}.", crash_cause_idx=i)
                return True
        return False

    def pre_game_checks(self) -> bool:
        # Check for compile errors
        if self.check_compile_errors():
            return False
        return True

    def get_obs_and_act(self) -> Tuple[int, Any, List[str], str]:
        current_id, obs_dict, legal_actions = self.env.dict_observation
        act = self.agents[current_id].act(obs_dict)
        return current_id, obs_dict, legal_actions, act

            
            


