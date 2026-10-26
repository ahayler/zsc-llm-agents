"""Code-agent training + cross-play evaluation pipeline used by meta-training and CFD evaluation.

Ported from the original main_code.py (train_and_xp_eval_code and its helpers).
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from langfuse import observe
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from configs.configs import CodeXPEvaluatorConfig, GeneticCodeTrainerConfig
from core import CodePolicy
from train.prompt_generation import DIVERSE_INSTRUCTIONS_POLICY_GENERATOR
from train.sp_tuning import EvoCodePolicyTrainer
from train.utils import get_game_description
from train.xp_eval import CodeXPEvaluator
from utils.env_impl import get_env_name_file_save_from_config
from utils.metrics import compute_xp_summary_statistics
from utils.saving import CodeXPExperimentResults
from utils.tracing import langfuse_client

logger = logging.getLogger(__name__)


@observe()
def parallel_sp_train_code(
    num_sp_policy: int,
    trainer_config: GeneticCodeTrainerConfig,
    general_instructions: Optional[str] = DIVERSE_INSTRUCTIONS_POLICY_GENERATOR,
    executor: Optional[ThreadPoolExecutor] = None,
    base_seed: Optional[int] = None,
    preset_policies_per_seed: Optional[Dict[int, List[CodePolicy]]] = None,
    invalid_action_policies: Optional[List[CodePolicy]] = None,
) -> Tuple[Dict[int, CodePolicy], Dict[int, List[CodePolicy]]]:
    """
    Generates num_sp_policy different joint policies.
    All are independently optimized based on the trainer_config (same across all "training" runs).

    If preset_policies_per_seed is provided, those seeds skip generation and use
    the preset policies as starting points for the critic.
    If invalid_action_policies is provided, they are used as partners for invalid
    action filtering instead of self-play.
    """
    if preset_policies_per_seed is None:
        preset_policies_per_seed = {}

    def train_one(
        trainer_config,
        executor,
        id: int,
        langfuse_trace_id: str,
        langfuse_parent_observation_id: str,
    ):
        """Start one SP-trainer and return the policies it learns."""
        local_config = trainer_config.model_copy(deep=True)
        seed_for_id = (base_seed + id) if base_seed is not None else None
        if seed_for_id is not None:
            local_config.generator_config.seed = seed_for_id

        trainer = EvoCodePolicyTrainer(
            local_config, executor, f"sp_trainer_{id}", general_instructions, rng_seed=seed_for_id
        )
        preset = preset_policies_per_seed.get(id)
        top_1_policy, _, generated_policy, policy_critic_histories = trainer.train(
            langfuse_trace_id=langfuse_trace_id,
            langfuse_parent_observation_id=langfuse_parent_observation_id,
            preset_policies=preset,
            invalid_action_policies=invalid_action_policies,
        )
        return id, top_1_policy, generated_policy, policy_critic_histories

    # Currently we only support top_k = 1
    assert trainer_config.top_k == 1

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        overall_task = progress.add_task(
            f"Training {num_sp_policy} policies...", total=num_sp_policy
        )

        trace_id = langfuse_client.get_current_trace_id()
        observation_id = langfuse_client.get_current_observation_id()

        def run_with_executor(exec: ThreadPoolExecutor):
            futures = [
                exec.submit(
                    train_one, trainer_config, exec, id, trace_id, observation_id
                )
                for id in range(num_sp_policy)
            ]

            # Wait for all futures and collect results
            sp_policies = {}
            generated_policies = {}
            policy_critic_histories = {}
            for future in as_completed(futures):
                id, top_1_policy, generated_policy, policy_critic_history = (
                    future.result()
                )

                if len(top_1_policy) > 0:
                    sp_policies[id] = top_1_policy[0]
                else:
                    logger.warning(
                        f"No coordination-friendly policy found for seed {id}"
                    )

                generated_policies[id] = generated_policy
                policy_critic_histories[id] = policy_critic_history
                progress.advance(overall_task)

            return sp_policies, generated_policies, policy_critic_histories

        if executor is not None:
            sp_policies, generated_policies, policy_critic_histories = run_with_executor(executor)
        else:
            with ThreadPoolExecutor(max_workers=64) as new_executor:
                sp_policies, generated_policies, policy_critic_histories = run_with_executor(new_executor)

        if not trainer_config.use_policy_critic:
            policy_critic_histories = None

    return sp_policies, generated_policies, policy_critic_histories


@observe(capture_output=False)
def xp_eval_code(
    policies: List[CodePolicy], evaluator_config: CodeXPEvaluatorConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a list of policies in N×N cross-play.
    """
    xp_evaluator = CodeXPEvaluator(evaluator_config)
    xp_scores, compact_rollout_histories = xp_evaluator.eval_grid(policies, policies)
    return xp_scores, compact_rollout_histories


@observe(capture_output=False)
def train_and_xp_eval_code(
    num_sp_policies: int,
    trainer_config: GeneticCodeTrainerConfig,
    evaluator_config: CodeXPEvaluatorConfig,
    langfuse_session_id: Optional[str] = None,
    executor: Optional[ThreadPoolExecutor] = None,
    base_seed: Optional[int] = None,
    preset_policies_per_seed: Optional[Dict[int, List[CodePolicy]]] = None,
    invalid_action_policies: Optional[List[CodePolicy]] = None,
) -> CodeXPExperimentResults:
    if langfuse_session_id is not None:
        langfuse_client.update_current_trace(session_id=langfuse_session_id)

    sp_policies, generated_policies, policy_critic_histories = parallel_sp_train_code(
        num_sp_policies, trainer_config, executor=executor, base_seed=base_seed,
        preset_policies_per_seed=preset_policies_per_seed,
        invalid_action_policies=invalid_action_policies,
    )
    sp_policies_list = [sp_policies[i] for i in sorted(sp_policies.keys())]
    xp_scores, compact_rollout_histories = xp_eval_code(
        sp_policies_list, evaluator_config,
    )

    summary_statistics = compute_xp_summary_statistics(xp_scores)

    _metadata = {
        "sp_scores_mean": summary_statistics.sp_scores_mean,
        "sp_scores_std": summary_statistics.sp_scores_std,
        "xp_scores_mean": summary_statistics.xp_scores_mean,
        "xp_scores_std": summary_statistics.xp_scores_std,
        "xp_scores": xp_scores.tolist(),
        "trainer_config": trainer_config,
        "evaluator_config": evaluator_config,
        "sp_policies": sp_policies,
    }

    langfuse_client.update_current_span(
        name=f"{get_env_name_file_save_from_config(trainer_config.env_config)}_training_and_xp_eval",
        metadata=_metadata,
    )

    trace_id = langfuse_client.get_current_trace_id()

    experiment_results = CodeXPExperimentResults(
        experiment_name=f"{get_env_name_file_save_from_config(trainer_config.env_config)}_training_and_xp_eval",
        env_name_file_safe=get_env_name_file_save_from_config(
            trainer_config.env_config
        ),
        generated_policies=generated_policies,
        policy_critic_histories=policy_critic_histories,
        selected_policies=sp_policies,
        xp_matrix=xp_scores,
        compact_rollout_histories=compact_rollout_histories,
        trace_id=trace_id,
        game_description=get_game_description(trainer_config.env_config),
    )

    return experiment_results

