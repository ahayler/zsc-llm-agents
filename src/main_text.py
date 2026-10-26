from pickle import FALSE
from configs.envs import  EnvConfig
from agents.text.chat_history import ChatHistoryAgent
from configs.configs import DEFAULT_CHAT_HISTORY_AGENT_CONFIG
from utils.env_impl import get_env_name_file_save_from_config
from eval.run import TextGameRun
from llm.request import LLMType
from configs.configs import BaseTextAgentConfig

from eval.text_agents import rollout_text_agents

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import datetime

from dataclasses import dataclass

from configs.contexts import TextSPConfig
from utils.log_utils import get_logger
from langfuse import observe
from utils.tracing import langfuse_client
from utils.metrics import paper_failure_report

logger = get_logger(__name__)

from utils.saving import save_text_xp_experiment_results, TextXPExperimentResults, std_error_array_from_rewards_array, rewards_array_from_compact_rollout_histories

def text_agents_xp(env_config: EnvConfig, agent_configs: List[BaseTextAgentConfig], num_game_runs: int, executor: ThreadPoolExecutor, experiment_name: str):

    xp_scores = np.zeros((len(agent_configs), len(agent_configs)))
    compact_rollout_histories = np.zeros((len(agent_configs), len(agent_configs)), dtype=object)

    total_evaluations = len(agent_configs) * len(agent_configs)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"Evaluating {len(agent_configs)}x{len(agent_configs)} agent pairs...", total=total_evaluations)
        
        for i in range(len(agent_configs)):
            for j in range(len(agent_configs)):
                _results = rollout_text_agents(env_config, (agent_configs[i], agent_configs[j]), num_game_runs, executor, langfuse_session_id=experiment_name)
                xp_scores[i, j] = _results.avg_reward
                compact_rollout_histories[i, j] = _results.compact_rollout_histories
                progress.advance(task)

    xp_results = TextXPExperimentResults(
        experiment_name=experiment_name,
        env_name_file_safe=get_env_name_file_save_from_config(env_config),
        agent_configs=agent_configs,
        xp_matrix=xp_scores,
        compact_rollout_histories=compact_rollout_histories,
        additional_info={
            "paper_failures": paper_failure_report(
                get_env_name_file_save_from_config(env_config), compact_rollout_histories
            ),
        },
    )

    save_text_xp_experiment_results(xp_results)

    return xp_scores, xp_results

@observe(capture_output=False, capture_input=False)
def create_xp_results_from_text_xp_config(text_sp_config: TextSPConfig, experiment_name: str, executor: ThreadPoolExecutor):
    logger.info(f"Creating SP results from text SP config:\n{text_sp_config}")

    agent_config = DEFAULT_CHAT_HISTORY_AGENT_CONFIG.model_copy(deep=True)

    if text_sp_config.include_random_number:
        agent_config.include_random_number = True
    else:
        agent_config.include_random_number = False

    # Optional CFD string for the text agent prompt.
    agent_config.cfd_str = getattr(text_sp_config, "cfd_str", "") or ""

    langfuse_client.update_current_trace(session_id=experiment_name)
    langfuse_client.update_current_span(
        name=f"{str(text_sp_config)}",
        metadata={
            "text_sp_config": text_sp_config,
        }
    )

    xp_scores, xp_results = text_agents_xp(text_sp_config.env_config, [agent_config], text_sp_config.num_game_runs, executor, experiment_name)

    return xp_scores, xp_results

@observe()
def run_llm_xp_evaluation(
    env_config: EnvConfig, 
    llms: List[LLMType], 
    num_game_runs: int, 
    executor: ThreadPoolExecutor, 
    experiment_name: str,
    reasoning_effort: str = "medium",
    max_tokens: Optional[int] = None,
    cfd_str: Optional[str] = None,
    include_random_number: bool = False,
):
    """
    Evaluates the Cross-Play (XP) matrix for agents using different LLMs.
    """
    logger.info(f"Running XP evaluation for LLMs: {llms} with reasoning_effort={reasoning_effort}, max_tokens={max_tokens}")
    
    agent_configs = []
    for model in llms:
        # Create config for this model
        config = DEFAULT_CHAT_HISTORY_AGENT_CONFIG.model_copy(deep=True)
        config.llm_config.model = model
        config.llm_config.reasoning_effort = reasoning_effort
        if max_tokens is not None:
            config.llm_config.max_tokens = max_tokens
        if cfd_str is not None:
            config.cfd_str = cfd_str
        config.include_random_number = include_random_number
        # Ensure we use a unique description for the agent so we can distinguish them in logs/results
        # BaseTextAgentConfig doesn't have a 'name' field, but get_short_name uses model name.
        agent_configs.append(config)
    
    langfuse_client.update_current_trace(session_id=experiment_name)
    
    xp_scores, xp_results = text_agents_xp(
        env_config=env_config,
        agent_configs=agent_configs,
        num_game_runs=num_game_runs,
        executor=executor,
        experiment_name=experiment_name
    )
    
    logger.info(f"XP Matrix for models {[m for m in llms]}:")
    logger.info(xp_scores)
    return xp_scores, xp_results

def main():
    import argparse
    from configs.envs import (
        CatAndDogConfig, DEFAULT_AD_HOC_CONVENTIONS_CONFIG,
        DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG,
        DEFAULT_TINY_HANABI_CONFIG, DEFAULT_TINY_HANABI_FIXED_CONFIG,
        DEFAULT_BRITTLE_BRIDGE_CONFIG_TEXT, DEFAULT_NARROW_TUNNEL_CONFIG_TEXT,
        DEFAULT_PERMUTED_LEVERS_CONFIG, DEFAULT_MATRIX_GAME_CONFIG,
        DEFAULT_HANABI_TEXT_3X3_1LIFE_CONFIG, DEFAULT_HINT_GUESS_CONFIG,
    )
    from configs.paper import (
        CORE_TEXT_EPISODES, CORE_TEXT_MODELS, CAT_DOG_XP_MODELS,
        HANABI_HINT_GUESS_MODELS, TEXT_XP_EPISODES, TEXT_CFDS,
    )

    parser = argparse.ArgumentParser(description="Paper text-agent experiments")
    parser.add_argument("--suite", choices=("core", "cat_dog_xp", "hanabi", "hint_guess"), default="core")
    parser.add_argument("--conditions", nargs="+", choices=tuple(TEXT_CFDS), default=["baseline", "handcrafted"])
    args = parser.parse_args()
    if args.suite == "core":
        env_configs = [
            CatAndDogConfig(), DEFAULT_AD_HOC_CONVENTIONS_CONFIG,
            DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG,
            DEFAULT_TINY_HANABI_CONFIG, DEFAULT_TINY_HANABI_FIXED_CONFIG,
            DEFAULT_BRITTLE_BRIDGE_CONFIG_TEXT, DEFAULT_NARROW_TUNNEL_CONFIG_TEXT,
            DEFAULT_PERMUTED_LEVERS_CONFIG, DEFAULT_MATRIX_GAME_CONFIG,
        ]
    else:
        env_configs = [{
            "cat_dog_xp": CatAndDogConfig(),
            "hanabi": DEFAULT_HANABI_TEXT_3X3_1LIFE_CONFIG,
            "hint_guess": DEFAULT_HINT_GUESS_CONFIG,
        }[args.suite]]

    with ThreadPoolExecutor(max_workers=100) as executor:
        for condition in args.conditions:
            if args.suite == "core":
                # Core scores are self-play. Learned-CFD text transfer uses Grok only.
                names = ("OR_GROK_41_FAST",) if condition == "learned" else CORE_TEXT_MODELS
                model_groups = [[LLMType[name]] for name in names]
            else:
                names = CAT_DOG_XP_MODELS if args.suite == "cat_dog_xp" else HANABI_HINT_GUESS_MODELS
                model_groups = [[LLMType[name] for name in names]]
            for env_config in env_configs:
                env_name = get_env_name_file_save_from_config(env_config)
                episodes = CORE_TEXT_EPISODES[env_name] if args.suite == "core" else TEXT_XP_EPISODES
                for models in model_groups:
                    model_tag = "+".join(model.name for model in models)
                    run_llm_xp_evaluation(
                        env_config=env_config,
                        llms=models,
                        num_game_runs=episodes,
                        executor=executor,
                        experiment_name=(
                            f"paper_text_{args.suite}_{condition}_{env_name}_{model_tag}_"
                            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                        ),
                        reasoning_effort="medium",
                        max_tokens=8192,
                        cfd_str=TEXT_CFDS[condition],
                        include_random_number=False,
                    )


if __name__ == "__main__":
    main()
