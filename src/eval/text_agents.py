import contextvars
from langfuse import observe
from concurrent.futures import Executor
from typing import Optional, Callable, Tuple, List
from core import CompactRolloutHistory

from configs.envs import EnvConfig
from agents.base import BaseTextAgent
from configs.configs import BaseTextAgentConfig, ChatHistoryAgentConfig
from agents.text.chat_history import ChatHistoryAgent

from train.utils import get_env_from_config, get_game_description
from utils.env_impl import get_env_name_from_config
from utils.tracing import langfuse_client

from eval.run import TextGameRun
import numpy as np

from dataclasses import dataclass

from opentelemetry.trace import get_current_span
from langfuse import LangfuseOtelSpanAttributes

@dataclass
class TextAgentsRolloutResult:
    compact_rollout_histories: List[CompactRolloutHistory]
    rewards_per_run: List[float]
    avg_reward: float
    std_reward: float
    max_reward: float
    min_reward: float
    text_agent_configs: Tuple[BaseTextAgentConfig, BaseTextAgentConfig]
    langfuse_trace_id: Optional[str] = None


def init_text_agent(game_description: str, use_langfuse: bool, text_agent_config: BaseTextAgentConfig) -> BaseTextAgent:
    if isinstance(text_agent_config, ChatHistoryAgentConfig):
        return ChatHistoryAgent(game_description=game_description, use_langfuse=use_langfuse, config=text_agent_config)
    else:
        raise ValueError(f"Unknown text agent config type: {type(text_agent_config)}")

@observe(capture_output=False)
def rollout_text_agents(
    env_config: EnvConfig, 
    text_agent_configs: Tuple[BaseTextAgentConfig, BaseTextAgentConfig], 
    num_game_runs: int, 
    executor: Optional[Executor] = None, 
    langfuse_name: Optional[str] = None,
    langfuse_session_id: Optional[str] = None,
) -> TextAgentsRolloutResult:
    """
    This function is designed to be pure, so that the original policy object will not be modified 

    Inputs:
    - env_config (EnvConfig): The config for the environment to play on
    - text_agent_config_factory_tuple (Tuple[Callable[[], BaseTextAgentConfig], Callable[[], BaseTextAgentConfig]]): The agent_config factory for each of the two agents
    - num_game_runs (int): The number of game runs to play
    - langfuse_name (Optional[str]): The name of the langfuse run
    - langfuse_session_id (Optional[str]): The session id of the langfuse run

    Output:
    - compact_rollout_histories (List[CompactRolloutHistory]): The compact rollout histories for each game run
    - rewards_per_run (List[float]): The rewards for each game run
    """

    MAX_GAME_RUNS_TO_LOG = 500 # Note: If we make use of this, then we also won't get correct cost estimates on langfuse
    game_description = get_game_description(env_config)
    trace_id = langfuse_client.get_current_trace_id()
    observation_id = langfuse_client.get_current_observation_id()

    text_agent_configs = [text_agent_configs[0].model_copy(deep=True), text_agent_configs[1].model_copy(deep=True)]

    if langfuse_session_id:
        langfuse_client.update_current_trace(
            session_id=langfuse_session_id
        )

    def run_once(game_run, log_to_langfuse: bool):
        """
        Run one game_run. Records NaN if the rollout fails.
        """

        env = get_env_from_config(env_config)

        use_langfuse = True if game_run < MAX_GAME_RUNS_TO_LOG else False
        
        agents = [init_text_agent(game_description, use_langfuse, text_agent_configs[0]), init_text_agent(game_description, use_langfuse, text_agent_configs[1])]
        env.reset()
        game = TextGameRun(env, agents)

        game.run()

        compact_rollout_history = game.compact_rollout_history

        if log_to_langfuse:
            get_current_span().set_attribute(LangfuseOtelSpanAttributes.AS_ROOT, False)
            # emit to Langfuse
            _metadata = {
                "Compact Rollout History": compact_rollout_history,
            }
            langfuse_client.update_current_span(
                name=f"GameRun: {game_run}, Reward: {compact_rollout_history.rewards_sum}",
                metadata=_metadata
        )

        return compact_rollout_history, game_run

    def run_games_sequential():
        compact_rollout_histories = []
        rewards_per_run = []

        for game_run in range(num_game_runs):
            if game_run < MAX_GAME_RUNS_TO_LOG:
                compact_rollout_history, _ = observe()(run_once)(game_run, log_to_langfuse=True, langfuse_trace_id=trace_id, langfuse_parent_observation_id=observation_id)
            else:
                compact_rollout_history, _ = run_once(game_run, log_to_langfuse=False)
            compact_rollout_histories.append(compact_rollout_history)
            rewards_per_run.append(compact_rollout_history.rewards_sum)

            # TODO: Figure out what do, if the text agents crash
            # if compact_rollout_history.crashed:
            #     break

        return compact_rollout_histories, rewards_per_run
    
    if executor is None:
        compact_rollout_histories, rewards_per_run = run_games_sequential()
    else:
        compact_rollout_histories = np.zeros(num_game_runs, dtype=object)
        rewards_per_run = np.zeros(num_game_runs)

        def get_future(game_run):

            if game_run < MAX_GAME_RUNS_TO_LOG:
                return executor.submit(observe()(run_once), game_run, log_to_langfuse=True, langfuse_trace_id=trace_id, langfuse_parent_observation_id=observation_id)
            else:
                return executor.submit(run_once, game_run, log_to_langfuse=False)

        futures = [get_future(game_run) for game_run in range(num_game_runs)]

        for future in futures:
            compact_rollout_history, game_run = future.result()
            compact_rollout_histories[game_run] = compact_rollout_history
            rewards_per_run[game_run] = compact_rollout_history.rewards_sum

            # TODO: Figure out what to do, if the text agents crash
            # if compact_rollout_history.crashed:
            #     break

    
    avg_reward = float(np.mean(rewards_per_run))
    std_reward = float(np.std(rewards_per_run))
    std_error_reward = std_reward / np.sqrt(len(rewards_per_run))
    max_reward = float(max(rewards_per_run))
    min_reward = float(min(rewards_per_run))

    _metadata = {
        "Average Reward": avg_reward,
        "Min Reward": min_reward,
        "Max Reward": max_reward,
        "Std Reward": std_reward,
        "agent_0_config": text_agent_configs[0],
        "agent_1_config": text_agent_configs[1],
    }

    if not langfuse_name:
        agent_0_name = text_agent_configs[0].llm_config.model.split("/")[-1]
        agent_1_name = text_agent_configs[1].llm_config.model.split("/")[-1]
        langfuse_name = f"{get_env_name_from_config(env_config)}_{agent_0_name}_{agent_1_name}"


    langfuse_client.update_current_span(
        name=f"{langfuse_name}, Avg Reward: {avg_reward:.2f}, StdError: {std_error_reward:.2f}",
        metadata=_metadata
    )

    langfuse_trace_id = langfuse_client.get_current_trace_id()
    
    return TextAgentsRolloutResult(compact_rollout_histories, rewards_per_run, avg_reward, std_reward, max_reward, min_reward, text_agent_configs, langfuse_trace_id)