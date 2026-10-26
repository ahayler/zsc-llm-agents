#!/usr/bin/env python3
"""
Evaluate CFDs on text agents across multiple environments.

For each (CFD, environment) combination:
  1. Create text agent configs with CFD injected into the prompt
  2. Run XP evaluation (self-play for single LLM, cross-play for multiple)
  3. Save results incrementally
  4. Report SP and XP scores

Single-turn games (each player acts once): 100 episodes
Multi-turn games: 50 episodes

Usage:
    python -m meta_train.population_based.scripts.run_cfd_text_eval \
        --experiment_name my_text_eval \
        --cfd "- When the observation is ambiguous, prefer conservative actions..." \
        --envs CAT_AND_DOG BRITTLE_BRIDGE \
        --llm OR_GROK_41_FAST \
        --num_game_runs_single 100 \
        --num_game_runs_multi 50
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
from langfuse import observe

from configs.configs import (
    DEFAULT_CHAT_HISTORY_AGENT_CONFIG,
    ChatHistoryAgentConfig,
    LLMConfig,
)
from configs.envs import (
    CatAndDogConfig,
    CatAndDog2Config,
    CatAndDog3Config,
    CatAndDog4Config,
    DEFAULT_AD_HOC_CONVENTIONS_CONFIG,
    DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG,
    DEFAULT_BRITTLE_BRIDGE_CONFIG_TEXT,
    DEFAULT_NARROW_TUNNEL_CONFIG_TEXT,
    DEFAULT_TINY_HANABI_CONFIG,
    DEFAULT_TINY_HANABI_FIXED_CONFIG,
    DEFAULT_MATRIX_GAME_CONFIG,
    DEFAULT_PERMUTED_LEVERS_CONFIG,
    EnvConfig,
)
from llm.request import LLMType
from main_text import text_agents_xp
from utils.tracing import langfuse_client
from configs.paper import CORE_TEXT_EPISODES
from utils.metrics import paper_failure_report


logger = logging.getLogger(__name__)

# Environments and their configs
# Note: text agents use different configs than code agents for some envs
# (e.g. fewer max_turns for Brittle Bridge and Narrow Tunnel)
ENV_CONFIGS: Dict[str, EnvConfig] = {
    "cat_and_dog": CatAndDogConfig(),
    "ad_hoc_conventions": DEFAULT_AD_HOC_CONVENTIONS_CONFIG,
    "ad_hoc_conventions_permuted_signals": DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG,
    "tiny_hanabi": DEFAULT_TINY_HANABI_CONFIG,
    "tiny_hanabi_fixed": DEFAULT_TINY_HANABI_FIXED_CONFIG,
    "brittle_bridge": DEFAULT_BRITTLE_BRIDGE_CONFIG_TEXT,
    "narrow_tunnel": DEFAULT_NARROW_TUNNEL_CONFIG_TEXT,
    "matrix_game": DEFAULT_MATRIX_GAME_CONFIG,
    "permuted_levers": DEFAULT_PERMUTED_LEVERS_CONFIG,
}

# Single-turn games: each player acts exactly once
SINGLE_TURN_ENVS = {name for name, episodes in CORE_TEXT_EPISODES.items() if episodes == 100}


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    class ShortNameFormatter(logging.Formatter):
        def format(self, record):
            record.shortname = record.name.rsplit(".", 1)[-1]
            return super().format(record)

    formatter = ShortNameFormatter(
        "%(asctime)s - %(shortname)s - %(levelname)s - %(message)s",
        datefmt="%H:%M"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_file = output_dir / "console.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


@observe()
def run_text_eval_for_env(
    cfd: str,
    cfd_name: str,
    env_name: str,
    env_config: EnvConfig,
    llm: LLMType,
    num_game_runs: int,
    output_dir: Path,
    langfuse_session_id: str,
    executor: ThreadPoolExecutor,
    reasoning_effort: Optional[str] = None,
    max_tokens: int = 8192,
    include_random_number: bool = False,
) -> Dict:
    """Run text agent eval for one (CFD, environment) pair."""
    langfuse_client.update_current_trace(session_id=langfuse_session_id)

    save_dir = output_dir / cfd_name / env_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing results (resume support)
    results_path = save_dir / "results.json"
    if results_path.exists():
        logger.info(f"  {cfd_name}/{env_name}: Loading existing results")
        with open(results_path) as f:
            return json.load(f)

    # Save the CFD text
    with open(save_dir / "cfd.txt", "w") as f:
        f.write(cfd)

    logger.info(f"=== {cfd_name} / {env_name} ===")
    logger.info(f"  LLM: {llm.name}, Games: {num_game_runs}")
    logger.info(f"  CFD: {cfd[:80]}...")

    # Create agent config
    agent_config = DEFAULT_CHAT_HISTORY_AGENT_CONFIG.model_copy(deep=True)
    agent_config.llm_config.model = llm
    agent_config.llm_config.max_tokens = max_tokens
    agent_config.llm_config.reasoning_effort = reasoning_effort
    agent_config.include_random_number = include_random_number
    agent_config.cfd_str = cfd if cfd else ""

    start_time = time.time()

    # Run XP evaluation (1x1 matrix for single LLM = self-play)
    xp_scores, xp_results = text_agents_xp(
        env_config=env_config,
        agent_configs=[agent_config],
        num_game_runs=num_game_runs,
        executor=executor,
        experiment_name=langfuse_session_id,
    )

    elapsed = time.time() - start_time
    sp_score = float(xp_scores[0, 0])

    # Extract per-game rewards from compact rollout histories
    histories = xp_results.compact_rollout_histories[0, 0]
    rewards = [float(h.rewards_sum) for h in histories if h.rewards_sum is not None]

    result = {
        "env_name": env_name,
        "cfd_name": cfd_name,
        "llm": llm.name,
        "num_game_runs": num_game_runs,
        "sp_score": sp_score,
        "sp_std": float(np.std(rewards)) if rewards else 0.0,
        "sp_sem": float(np.std(rewards) / np.sqrt(len(rewards))) if rewards else 0.0,
        "elapsed_seconds": elapsed,
        "per_game_rewards": rewards,
        "paper_failures": paper_failure_report(env_name, xp_results.compact_rollout_histories),
    }

    # Save results
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    # Also save the XP matrix
    with open(save_dir / "xp_matrix.json", "w") as f:
        json.dump(xp_scores.tolist(), f)

    logger.info(f"  SP score: {sp_score:.2f} +/- {result['sp_sem']:.2f} ({elapsed:.0f}s)")

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CFDs on text agents across environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--cfd", type=str, required=True,
        help="CFD string to evaluate",
    )
    parser.add_argument(
        "--cfd_name", type=str, default=None,
        help="Short name for the CFD (for directory naming)",
    )
    parser.add_argument(
        "--envs", type=str, nargs="+",
        default=list(CORE_TEXT_EPISODES),
        choices=list(ENV_CONFIGS.keys()),
        help="Environments to evaluate on",
    )
    parser.add_argument(
        "--llm", type=str, default="OR_GROK_41_FAST",
        help="LLM to use for text agents",
    )
    parser.add_argument("--num_game_runs_single", type=int, default=100,
                        help="Number of game runs for single-turn games")
    parser.add_argument("--num_game_runs_multi", type=int, default=50,
                        help="Number of game runs for multi-turn games")
    parser.add_argument("--reasoning_effort", type=str, default="medium",
                        help="Reasoning effort (None to omit, or low/medium/high)")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_workers", type=int, default=64)
    random_group = parser.add_mutually_exclusive_group()
    random_group.add_argument("--include_random_number", action="store_false", dest="no_random_number",
                              help="Add a random number to the prompt (not in the paper template)")
    random_group.add_argument("--no_random_number", action="store_true", default=True,
                              help="Disable random number in prompt (default)")

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path.cwd()
    for parent in [project_root] + list(project_root.parents):
        if (parent / "pyproject.toml").exists():
            project_root = parent
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "logs" / args.experiment_name / timestamp

    setup_logging(output_dir)

    llm = LLMType[args.llm]
    cfd = args.cfd
    cfd_name = args.cfd_name or "cfd"

    logger.info("=" * 60)
    logger.info("CFD Text Agent Evaluation")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"CFD ({cfd_name}): {cfd[:100]}...")
    logger.info(f"Environments: {args.envs}")
    logger.info(f"LLM: {llm.name}")
    logger.info(f"Games: single-turn={args.num_game_runs_single}, multi-turn={args.num_game_runs_multi}")
    logger.info(f"Reasoning effort: {args.reasoning_effort}")
    logger.info("=" * 60)

    # Save config
    config = {
        "experiment_name": args.experiment_name,
        "cfd": cfd,
        "cfd_name": cfd_name,
        "envs": args.envs,
        "llm": llm.name,
        "num_game_runs_single": args.num_game_runs_single,
        "num_game_runs_multi": args.num_game_runs_multi,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "timestamp": timestamp,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    langfuse_session_id = f"{args.experiment_name}_{timestamp}"
    executor = ThreadPoolExecutor(max_workers=args.max_workers)

    all_results = {}
    for env_name in args.envs:
        env_config = ENV_CONFIGS[env_name]
        num_game_runs = (
            args.num_game_runs_single if env_name in SINGLE_TURN_ENVS
            else args.num_game_runs_multi
        )

        try:
            result = run_text_eval_for_env(
                cfd=cfd,
                cfd_name=cfd_name,
                env_name=env_name,
                env_config=env_config,
                llm=llm,
                num_game_runs=num_game_runs,
                output_dir=output_dir,
                langfuse_session_id=langfuse_session_id,
                executor=executor,
                reasoning_effort=args.reasoning_effort,
                max_tokens=args.max_tokens,
                include_random_number=not args.no_random_number,
            )
            all_results[env_name] = result
        except Exception as e:
            logger.exception(f"Failed for {env_name}: {e}")
            all_results[env_name] = {"error": str(e)}

    executor.shutdown(wait=False)

    # Save aggregate results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Environment':<35} {'SP Score':>10} {'SEM':>10} {'Games':>8}")
    logger.info("-" * 65)
    for env_name, result in all_results.items():
        if "error" in result:
            logger.info(f"{env_name:<35} ERROR: {result['error'][:30]}")
        else:
            logger.info(f"{env_name:<35} {result['sp_score']:>10.2f} {result['sp_sem']:>10.2f} {result['num_game_runs']:>8}")


if __name__ == "__main__":
    main()
