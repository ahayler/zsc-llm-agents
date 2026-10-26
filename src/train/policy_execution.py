from typing import Optional
from typing import List
from core import CodePolicy
from configs.envs import EnvConfig
from eval.run import CodeGameRun
from agents.code.policy_agent import CodePolicyAgent
from configs.configs import CodeRolloutConfig, CodePolicyAgentConfig
from langfuse import observe
from utils.tracing import langfuse_client
import math
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor

from train.utils import get_env_from_config

from utils.log_utils import get_logger

import numpy as np

logger = get_logger(__name__)

def init_code_policy_agents(game_run_config: CodeRolloutConfig, policy: CodePolicy):
    def init_agent(agent_config: CodePolicyAgentConfig) -> CodePolicyAgent:
        agent = CodePolicyAgent(agent_config=agent_config)
        return agent
    
    agent_configs = game_run_config.get_agent_configs(policy)

    agents = []
    for agent_id in range(game_run_config.num_agents):
        agents.append(init_agent(agent_configs[agent_id]))

    return agents

@observe(capture_output=False)
def run_code_policy(config: CodeRolloutConfig, env_config: EnvConfig, policy: CodePolicy, langfuse_name: Optional[str] = None) -> CodePolicy:
    """
    This function is designed to be pure, so that the original policy object will not be modified 
    Inputs: GameRunConfig; Game Environment; Policy 
    Output: Policy with Game Run Context
    """

    policy = policy.clone()	
    compact_rollout_histories = []
    rewards_per_run = []

    MAX_GAME_RUNS_TO_LOG = 20
    n_game_runs_to_log = min(config.num_game_runs, MAX_GAME_RUNS_TO_LOG)

    def run_once(game_run, log_to_langfuse: bool):
        """
        Run one game_run. Records NaN if the rollout fails.
        """

        env = get_env_from_config(env_config)
        modified_policy, policy_modifier_metadata = env.policy_modifier.modify_policy(policy)
        agents = init_code_policy_agents(config, modified_policy)
        env.reset()
        game = CodeGameRun(env, agents)

        game.run()

        compact_rollout_history = game.compact_rollout_history
        compact_rollout_history.metadata_policy_modifier = policy_modifier_metadata

        if log_to_langfuse:
            # emit to Langfuse
            _metadata = {
                "Joint Policy Description": modified_policy.joint_policy,
                "Compact Rollout History": compact_rollout_history,
            }
            langfuse_client.update_current_span(
                name=f"GameRun: {game_run}, Reward: {compact_rollout_history.rewards_sum}",
                metadata=_metadata
        )

        return compact_rollout_history
    
    for game_run in range(config.num_game_runs):
        if game_run < n_game_runs_to_log:
            compact_rollout_history = observe()(run_once)(game_run, log_to_langfuse=True)
        else:
            compact_rollout_history = run_once(game_run, log_to_langfuse=False)
        compact_rollout_histories.append(compact_rollout_history)
        rewards_per_run.append(compact_rollout_history.rewards_sum)

        if compact_rollout_history.crashed:
            break
    
    policy.avg_reward = float(np.mean(rewards_per_run))
    policy.std_reward = float(np.std(rewards_per_run))
    policy.max_reward = float(max(rewards_per_run))
    policy.min_reward = float(min(rewards_per_run))
    #avg_reward = sum(rewards_per_run)/max(1, len(rewards_per_run))

    # if math.isnan(policy.avg_reward):
    #     logger.warning(f"Avg reward of NaN found for policy: {policy.name}. Setting to None (workaround for Langfuse).")
    #     policy.avg_reward = None
    # else:
    #     policy.avg_reward = avg_reward

    _metadata = {
        "Joint Policy Descriptions": policy.joint_policy,
        "Average Reward": policy.avg_reward,
        "Min Reward": policy.min_reward,
        "Max Reward": policy.max_reward,
        "Std Reward": policy.std_reward,
    }

    policy_name = langfuse_name if langfuse_name else policy.name


    langfuse_client.update_current_span(
        name=f"Avg Reward: {policy.avg_reward:.2f}, {policy_name}",
        metadata=_metadata
    )
    
    return policy, compact_rollout_histories

def run_code_policies_sequential(policies: List[CodePolicy], code_rollout_config: CodeRolloutConfig, env_config: EnvConfig) -> List[CodePolicy]:
    """
    Run a list of joint policies in parallel and return the results.
    """
    results = [
        CodePolicy(name="init", avg_reward=math.nan)
        for _ in range(len(policies))
    ]

    for idx, policy in enumerate(policies):
        # Currently we discard the compact rollout histories here (can be changed in the future)
        results[idx] = run_code_policy(code_rollout_config, env_config, policy)[0]

    return results