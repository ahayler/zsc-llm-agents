"""Create the starting-policy pool that Pareto meta-training samples from.

Meta-training (run_pareto_evolution.py) loads its starting policies from the saved code experiment
`coordination_failure_starting_point` for every training environment. Run this script once per
environment before meta-training. For each environment it generates joint policies with the code
agent, runs them through the policy critic with a permissive definition ("Every policy is
coordination-friendly.", so no policy is rejected), evaluates them in cross-play, and saves the results.

Example usage:
    python -m meta_train.population_based.scripts.create_starting_policies \
        --envs CAT_AND_DOG BRITTLE_BRIDGE --num_seeds 7 --num_policies 7
"""

import argparse
import logging

import wandb

from configs.contexts import DEFAULT_META_PROBLEM_CONFIGS
from configs.envs import CoordinationGame
from llm.request import LLMType
from train.pipeline import train_and_xp_eval_code
from utils.config_setup import setup_code_configs_from_xp_config
from utils.env_impl import get_env_name_file_save_from_config
from utils.saving import save_code_xp_experiment_results

logger = logging.getLogger(__name__)

STARTING_POINT_EXPERIMENT_NAME = "coordination_failure_starting_point"
STARTING_POINT_COORDINATION_FRIENDLY_DEFINITION = "Every policy is coordination-friendly."


def create_starting_policies(env: CoordinationGame, num_seeds: int, num_policies: int, llm: LLMType = None):
    xp_config = DEFAULT_META_PROBLEM_CONFIGS[env].xp_config.model_copy(deep=True)
    xp_config.num_seeds = num_seeds
    xp_config.overwrites["trainer_config.generator_config.num_policies"] = num_policies
    xp_config.overwrites["trainer_config.use_invalid_action_filtering_for_generated_policies"] = True
    if llm is not None:
        xp_config.overwrites["trainer_config.generator_config.llm_config.model"] = llm

    trainer_config, evaluator_config = setup_code_configs_from_xp_config(xp_config)
    trainer_config.use_policy_critic = True
    trainer_config.policy_critic_config.coordination_friendly_definition = STARTING_POINT_COORDINATION_FRIENDLY_DEFINITION
    trainer_config.policy_critic_config.reformulate_definition = False
    if llm is not None:
        trainer_config.policy_critic_config.llm_config.model = llm

    results = train_and_xp_eval_code(
        xp_config.num_seeds, trainer_config, evaluator_config,
        langfuse_session_id=STARTING_POINT_EXPERIMENT_NAME,
    )
    results.experiment_name = STARTING_POINT_EXPERIMENT_NAME
    results.env_name_file_safe = get_env_name_file_save_from_config(xp_config)
    results.coordination_friendly_definition = STARTING_POINT_COORDINATION_FRIENDLY_DEFINITION
    results.additional_info = {"epoch_index": 0}
    save_code_xp_experiment_results(experiment_results=results)
    n = sum(len(p) for p in results.generated_policies.values())
    logger.info(f"Saved {n} starting policies for {env.value} under '{STARTING_POINT_EXPERIMENT_NAME}'.")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", type=str, nargs="+", default=["CAT_AND_DOG", "BRITTLE_BRIDGE"],
                        help="CoordinationGame names (default: the paper's meta-training environments).")
    parser.add_argument("--num_seeds", type=int, default=7)
    parser.add_argument("--num_policies", type=int, default=7, help="Policies generated per seed.")
    parser.add_argument("--llm", type=str, default=None,
                        help="LLMType name for the generator and critic (default: the configured default model).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    wandb.init(mode="disabled")
    llm = LLMType[args.llm] if args.llm else None
    for name in args.envs:
        create_starting_policies(CoordinationGame[name], args.num_seeds, args.num_policies, llm)
    wandb.finish()


if __name__ == "__main__":
    main()
