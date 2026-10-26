#!/usr/bin/env python3
"""
Evaluate CFDs on code agents across multiple environments.

For each (CFD, environment, LLM) combination:
  1. Generate a policy using the CFD (via prompt injection or policy critic)
  2. Evaluate all policies in cross-play (XP matrix)
  3. Save policies + XP matrices incrementally
  4. Compute and report XP_seed scores at the end

Usage:
    python -m meta_train.population_based.scripts.run_cfd_eval \
        --experiment_name my_eval \
        --cfds "CFD text 1" "CFD text 2" \
        --envs CAT_AND_DOG BRITTLE_BRIDGE \
        --llms OR_GROK_41_FAST OR_GPT_52 \
        --num_seeds 6 \
        --num_game_runs 100 \
        --use_critic
"""

import argparse
import contextvars
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from configs.configs import (
    CodeXPEvaluatorConfig,
    DEFAULT_CODE_POLICY_AGENT_CONFIG_FACTORY,
    DEFAULT_CODE_POLICY_GENERATOR_CONFIG,
    DEFAULT_GENETIC_CODE_TRAINER_CONFIG,
    DEFAULT_POLICY_CRITIC_CONFIG,
    GeneticCodeTrainerConfig,
)
from configs.contexts import CoordinationGame, DEFAULT_XP_CONFIGS
from core import CodePolicy
from llm.request import LLMType
from langfuse import observe
from train.pipeline import train_and_xp_eval_code
from train.generate_policy import generate_policy
from train.xp_eval import CodeXPEvaluator
from utils.metrics import compute_xp_summary_statistics, paper_failure_report
from utils.env_impl import get_env_name_file_save_from_config
from utils.tracing import langfuse_client
from configs.paper import CORE_TEXT_EPISODES, CFD_TRANSFER_MODELS


logger = logging.getLogger(__name__)

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


def save_seed_result(
    save_dir: Path,
    seed_idx: int,
    xp_matrix: np.ndarray,
    policies: List[CodePolicy],
    llm_names: List[str],
):
    """Save XP matrix and policies for a single seed (incremental saving)."""
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / f"xp_matrix_{seed_idx}.json", "w") as f:
        json.dump(xp_matrix.tolist(), f)

    policies_data = []
    for policy, llm_name in zip(policies, llm_names):
        policies_data.append({
            "llm": llm_name,
            "name": policy.name,
            "joint_policy": policy.joint_policy,
        })
    with open(save_dir / f"policies_{seed_idx}.json", "w") as f:
        json.dump(policies_data, f, indent=2)


def generate_policy_for_llm(
    env_config,
    llm: LLMType,
    cfd: str,
    use_critic: bool,
    num_game_runs: int,
    langfuse_session_id: str,
    executor: Optional[ThreadPoolExecutor] = None,
    max_retries: int = 3,
) -> CodePolicy:
    """Generate a single policy for a given LLM and environment, using either
    prompt injection or the policy critic.

    - Prompt injection: CFD is embedded in the generation prompt. No critic.
    - Policy critic: policy is generated without CFD, then refined by the critic
      using the CFD as its coordination-friendly definition.
    """

    for attempt in range(max_retries):
        try:
            if use_critic:
                # Policy critic mode: generate, then refine with critic
                trainer_config = DEFAULT_GENETIC_CODE_TRAINER_CONFIG.model_copy(deep=True)
                trainer_config.env_config = env_config
                trainer_config.generator_config.llm_config.model = llm
                trainer_config.generator_config.num_policies = 1
                trainer_config.generator_config.use_cached_generations = False
                trainer_config.num_game_runs = num_game_runs
                trainer_config.top_k = 1
                trainer_config.use_policy_critic = True
                trainer_config.policy_critic_config = DEFAULT_POLICY_CRITIC_CONFIG.model_copy(deep=True)
                trainer_config.policy_critic_config.coordination_friendly_definition = cfd
                trainer_config.policy_critic_config.llm_config.model = llm

                evaluator_config = CodeXPEvaluatorConfig(
                    agent_config_factory=DEFAULT_CODE_POLICY_AGENT_CONFIG_FACTORY,
                    num_game_runs=num_game_runs,
                    num_agents=2,
                )
                evaluator_config.env_config = env_config

                results = train_and_xp_eval_code(
                    num_sp_policies=1,
                    trainer_config=trainer_config,
                    evaluator_config=evaluator_config,
                    langfuse_session_id=langfuse_session_id,
                    executor=executor,
                )
                return list(results.selected_policies.values())[0]
            else:
                # Prompt injection mode: use generate_policy() exactly as main_code2.py does
                policies = generate_policy(
                    cfd=cfd,
                    env=env_config,
                    num_policies=1,
                    llm=llm,
                    langfuse_session_id=langfuse_session_id,
                    n_generations_per_seed=1,
                    use_self_critic=False,
                    max_tokens=10000,
                )
                if not policies:
                    raise RuntimeError(f"Generator returned no policies for {llm}")
                return policies[0]
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {llm}: {e}")
            if attempt == max_retries - 1:
                raise
    raise RuntimeError("Unreachable")


def evaluate_xp_matrix(
    env_config,
    policies: List[CodePolicy],
    num_game_runs: int,
    failure_save_path: Optional[Path] = None,
) -> np.ndarray:
    """Evaluate an NxN cross-play matrix for the given policies."""
    evaluator_config = CodeXPEvaluatorConfig(
        agent_config_factory=DEFAULT_CODE_POLICY_AGENT_CONFIG_FACTORY,
        num_game_runs=num_game_runs,
        num_agents=2,
    )
    evaluator_config.env_config = env_config

    evaluator = CodeXPEvaluator(evaluator_config)
    xp_matrix, histories = evaluator.eval_grid(policies, policies, no_progress_bar=False)
    report = paper_failure_report(get_env_name_file_save_from_config(env_config), histories)
    if failure_save_path is not None and report is not None:
        failure_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_save_path, "w") as f:
            json.dump(report, f, indent=2)
    return xp_matrix


def compute_xp_llm_scores(
    xp_matrices: List[np.ndarray],
    llm_names: List[str],
    env_name: str,
) -> Dict:
    """Compute XP_llm scores from cross-LLM XP matrices.

    Each XP matrix is (n_llms x n_llms) from one seed. We average across seeds,
    then report per-LLM and overall SP/XP statistics.
    """
    if not xp_matrices:
        return {}

    n_llms = xp_matrices[0].shape[0]

    # Stack into (n_seeds, n_llms, n_llms)
    tensor = np.stack(xp_matrices, axis=0)

    # Replace NaN with environment minimum score
    min_score_map = {
        "cat_and_dog": -10, "brittle_bridge": -1000, "narrow_tunnel": -100,
        "ad_hoc_conventions": -100, "ad_hoc_conventions_permuted_signals": -100,
        "tiny_hanabi": 0, "tiny_hanabi_fixed": 0, "matrix_game": 0,
        "permuted_levers": 0,
    }
    min_score = min_score_map.get(env_name, -1000)
    tensor = np.where(np.isnan(tensor), min_score, tensor)

    # Average across seeds
    avg_matrix = np.mean(tensor, axis=0)

    # Per-LLM XP scores (off-diagonal involving this LLM)
    per_llm = {}
    for i in range(n_llms):
        mask = np.arange(n_llms) != i
        xp_vals = np.concatenate([avg_matrix[i, mask], avg_matrix[mask, i]])
        sp_vals = tensor[:, i, i]
        per_llm[llm_names[i]] = {
            "xp_llm_mean": float(np.mean(xp_vals)),
            "xp_llm_sem": float(np.std(xp_vals) / np.sqrt(len(xp_vals))),
            "sp_mean": float(np.mean(sp_vals)),
            "sp_sem": float(np.std(sp_vals) / np.sqrt(len(sp_vals))),
        }

    # Overall
    sp_overall = np.diag(avg_matrix)
    xp_overall = avg_matrix[~np.eye(n_llms, dtype=bool)]

    return {
        "avg_xp_matrix": avg_matrix.tolist(),
        "per_llm": per_llm,
        "overall": {
            "sp_mean": float(np.mean(sp_overall)),
            "sp_sem": float(np.std(sp_overall) / np.sqrt(len(sp_overall))),
            "xp_llm_mean": float(np.mean(xp_overall)),
            "xp_llm_sem": float(np.std(xp_overall) / np.sqrt(len(xp_overall))),
        },
    }


def compute_xp_seed_scores(
    env_config,
    all_policies: List[List[CodePolicy]],
    llm_names: List[str],
    env_name: str,
    num_game_runs: int,
    save_dir: Path,
) -> Dict:
    """Compute XP_seed scores by pairing policies from different seeds for the same LLM.

    For each LLM, pair seeds (0,1), (2,3), (4,5). For each pair, run a 2x2 XP grid
    and take the mean of the two off-diagonal entries as the XP_seed sample.
    Report mean and SEM of the 3 samples per LLM.

    Args:
        all_policies: List of policy lists, one per seed. Each inner list has one policy per LLM.
        llm_names: Names of the LLMs.
        env_name: Environment name (for NaN replacement).
        num_game_runs: Number of game rollouts per pair.
        save_dir: Directory to save intermediate results.
    """
    n_seeds = len(all_policies)
    n_llms = len(all_policies[0])
    n_pairs = n_seeds // 2

    if n_seeds % 2 != 0:
        logger.warning(f"Odd number of seeds ({n_seeds}), last seed will be unpaired for XP_seed")

    min_score_map = {
        "cat_and_dog": -10, "brittle_bridge": -1000, "narrow_tunnel": -100,
        "ad_hoc_conventions": -100, "ad_hoc_conventions_permuted_signals": -100,
        "tiny_hanabi": 0, "tiny_hanabi_fixed": 0, "matrix_game": 0,
        "permuted_levers": 0,
    }
    min_score = min_score_map.get(env_name, -1000)

    evaluator_config = CodeXPEvaluatorConfig(
        agent_config_factory=DEFAULT_CODE_POLICY_AGENT_CONFIG_FACTORY,
        num_game_runs=num_game_runs,
        num_agents=2,
    )
    evaluator_config.env_config = env_config
    evaluator = CodeXPEvaluator(evaluator_config)

    per_llm = {}
    all_xp_seed_samples = []

    for llm_idx in range(n_llms):
        llm_name = llm_names[llm_idx]
        samples = []

        for pair_idx in range(n_pairs):
            seed_a = 2 * pair_idx
            seed_b = 2 * pair_idx + 1

            policy_a = all_policies[seed_a][llm_idx]
            policy_b = all_policies[seed_b][llm_idx]

            # 2x2 XP grid
            xp_scores, histories = evaluator.eval_grid([policy_a, policy_b], [policy_a, policy_b], no_progress_bar=True)
            report = paper_failure_report(env_name, histories)
            if report is not None:
                with open(save_dir / f"failure_rates_seed_{llm_idx}_{pair_idx}.json", "w") as f:
                    json.dump(report, f, indent=2)
            xp_scores = np.where(np.isnan(xp_scores), min_score, xp_scores)

            # XP_seed sample = mean of off-diagonals
            xp_seed_sample = (xp_scores[0, 1] + xp_scores[1, 0]) / 2.0
            samples.append(float(xp_seed_sample))

            logger.info(f"    XP_seed {llm_name} pair ({seed_a},{seed_b}): {xp_seed_sample:.2f}")

        samples_arr = np.array(samples)
        all_xp_seed_samples.extend(samples)
        per_llm[llm_name] = {
            "xp_seed_mean": float(np.mean(samples_arr)),
            "xp_seed_sem": float(np.std(samples_arr) / np.sqrt(len(samples_arr))),
            "xp_seed_samples": samples,
        }

    all_samples_arr = np.array(all_xp_seed_samples)
    result = {
        "per_llm": per_llm,
        "overall": {
            "xp_seed_mean": float(np.mean(all_samples_arr)),
            "xp_seed_sem": float(np.std(all_samples_arr) / np.sqrt(len(all_samples_arr))),
        },
    }

    with open(save_dir / "xp_seed_scores.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


@observe()
def run_eval_for_cfd_env(
    cfd: str,
    cfd_name: str,
    env: CoordinationGame,
    llms: List[LLMType],
    num_seeds: int,
    num_game_runs: int,
    use_critic: bool,
    output_dir: Path,
    langfuse_session_id: str,
):
    """Run the full eval for one (CFD, environment) pair."""
    langfuse_client.update_current_trace(session_id=langfuse_session_id)

    env_name = env.value
    xp_config = DEFAULT_XP_CONFIGS[env]
    env_config = xp_config.env_config
    llm_names = [llm.name for llm in llms]

    save_dir = output_dir / cfd_name / env_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the CFD text
    with open(save_dir / "cfd.txt", "w") as f:
        f.write(cfd)

    logger.info(f"=== {cfd_name} / {env_name} ===")
    logger.info(f"  LLMs: {llm_names}")
    logger.info(f"  Seeds: {num_seeds}, Games/eval: {num_game_runs}")
    logger.info(f"  Mode: {'critic' if use_critic else 'prompt_injection'}")

    xp_matrices = []
    all_policies: List[List[CodePolicy]] = []

    # Load any previously saved results (resume support)
    existing_seeds = set()
    for seed_idx in range(num_seeds):
        matrix_path = save_dir / f"xp_matrix_{seed_idx}.json"
        policy_path = save_dir / f"policies_{seed_idx}.json"
        if matrix_path.exists() and policy_path.exists():
            with open(matrix_path) as f:
                xp_matrices.append(np.array(json.load(f)))
            with open(policy_path) as f:
                policies_data = json.load(f)
                policies = []
                for entry in policies_data:
                    policies.append(CodePolicy(
                        name=entry["name"],
                        joint_policy=entry["joint_policy"],
                    ))
                all_policies.append(policies)
            existing_seeds.add(seed_idx)
            logger.info(f"  Loaded existing seed {seed_idx}")

    max_attempts = num_seeds + 4  # allow up to 4 extra attempts for failures
    attempt_idx = 0
    successful_seeds = len(existing_seeds)

    while successful_seeds < num_seeds and attempt_idx < max_attempts:
        seed_idx = attempt_idx + len(existing_seeds)
        attempt_idx += 1

        # Skip already completed seeds
        if seed_idx in existing_seeds:
            continue

        logger.info(f"  Seed attempt {attempt_idx} (successful: {successful_seeds}/{num_seeds})...")
        start_time = time.time()

        try:
            # Generate one policy per LLM (in parallel, with Langfuse context propagation)
            logger.info(f"    Generating policies for {len(llms)} LLMs in parallel...")
            with ThreadPoolExecutor(max_workers=len(llms)) as gen_executor:
                futures = {}
                for llm in llms:
                    ctx = contextvars.copy_context()
                    future = gen_executor.submit(
                        ctx.run,
                        generate_policy_for_llm,
                        env_config=env_config,
                        llm=llm,
                        cfd=cfd,
                        use_critic=use_critic,
                        num_game_runs=num_game_runs,
                        langfuse_session_id=langfuse_session_id,
                    )
                    futures[llm] = future

                policies = []
                for llm in llms:
                    policy = futures[llm].result()
                    logger.info(f"    Got policy from {llm.name}")
                    policies.append(policy)

            # XP evaluation (cross-LLM)
            logger.info(f"    Evaluating {len(policies)}x{len(policies)} XP grid...")
            xp_matrix = evaluate_xp_matrix(
                env_config, policies, num_game_runs,
                failure_save_path=save_dir / f"failure_rates_{seed_idx}.json",
            )

            # Save incrementally
            save_seed_result(save_dir, seed_idx, xp_matrix, policies, llm_names)
            xp_matrices.append(xp_matrix)
            all_policies.append(policies)
            successful_seeds += 1

            elapsed = time.time() - start_time
            logger.info(f"    Seed {successful_seeds}/{num_seeds} done in {elapsed:.0f}s")
            logger.info(f"    XP matrix:\n{np.array2string(xp_matrix, precision=2)}")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"    Seed attempt failed after {elapsed:.0f}s: {e}")
            logger.info(f"    Continuing... ({successful_seeds}/{num_seeds} successful, {max_attempts - attempt_idx} attempts remaining)")

    if successful_seeds < num_seeds:
        logger.warning(f"  Only {successful_seeds}/{num_seeds} seeds completed after {max_attempts} attempts")

    # Compute XP_llm scores (cross-LLM, from existing matrices)
    xp_llm_scores = compute_xp_llm_scores(xp_matrices, llm_names, env_name)
    with open(save_dir / "xp_llm_scores.json", "w") as f:
        json.dump(xp_llm_scores, f, indent=2)

    logger.info(f"  XP_llm results for {cfd_name} / {env_name}:")
    logger.info(f"    SP mean: {xp_llm_scores['overall']['sp_mean']:.2f} +/- {xp_llm_scores['overall']['sp_sem']:.2f}")
    logger.info(f"    XP_llm mean: {xp_llm_scores['overall']['xp_llm_mean']:.2f} +/- {xp_llm_scores['overall']['xp_llm_sem']:.2f}")
    for llm_name, llm_scores in xp_llm_scores["per_llm"].items():
        logger.info(f"    {llm_name}: XP_llm={llm_scores['xp_llm_mean']:.2f}, SP={llm_scores['sp_mean']:.2f}")

    # Compute XP_seed scores (same LLM across seeds, requires additional game runs)
    logger.info(f"  Computing XP_seed scores...")
    xp_seed_scores = compute_xp_seed_scores(
        env_config=env_config,
        all_policies=all_policies,
        llm_names=llm_names,
        env_name=env_name,
        num_game_runs=num_game_runs,
        save_dir=save_dir,
    )

    logger.info(f"  XP_seed results for {cfd_name} / {env_name}:")
    logger.info(f"    XP_seed mean: {xp_seed_scores['overall']['xp_seed_mean']:.2f} +/- {xp_seed_scores['overall']['xp_seed_sem']:.2f}")
    for llm_name, llm_scores in xp_seed_scores["per_llm"].items():
        logger.info(f"    {llm_name}: XP_seed={llm_scores['xp_seed_mean']:.2f} +/- {llm_scores['xp_seed_sem']:.2f}")

    return {"xp_llm": xp_llm_scores, "xp_seed": xp_seed_scores}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CFDs on code agents across environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--cfds", type=str, nargs="+", required=True,
        help="CFD strings to evaluate. Use quotes for multi-word CFDs.",
    )
    parser.add_argument(
        "--cfd_names", type=str, nargs="+", default=None,
        help="Short names for each CFD (for directory naming). Defaults to cfd_0, cfd_1, ...",
    )
    parser.add_argument(
        "--envs", type=str, nargs="+",
        default=[name.upper() for name in CORE_TEXT_EPISODES],
        choices=[e.name for e in CoordinationGame],
        help="Environments to evaluate on",
    )
    parser.add_argument(
        "--llms", type=str, nargs="+",
        default=list(CFD_TRANSFER_MODELS),
        help="LLMs to use for policy generation",
    )
    parser.add_argument("--num_seeds", type=int, default=6,
                        help="Number of independent seeds (XP matrices) per (CFD, env)")
    parser.add_argument("--num_game_runs", type=int, default=100,
                        help="Number of game rollouts per policy pair")
    parser.add_argument("--use_critic", action="store_true",
                        help="Use policy critic instead of prompt injection")
    parser.add_argument("--no_cfd", action="store_true",
                        help="Run a no-CFD baseline (empty CFD string)")

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

    logger_instance = setup_logging(output_dir)

    # Parse CFDs
    cfds = list(args.cfds)
    if args.no_cfd:
        cfds.insert(0, "")

    cfd_names = args.cfd_names
    if cfd_names is None:
        cfd_names = []
        for i, cfd in enumerate(cfds):
            if cfd == "":
                cfd_names.append("no_cfd")
            else:
                cfd_names.append(f"cfd_{i}")
    if len(cfd_names) != len(cfds):
        raise ValueError(f"Number of cfd_names ({len(cfd_names)}) must match number of cfds ({len(cfds)})")

    envs = [CoordinationGame[e] for e in args.envs]
    llms = [LLMType[l] for l in args.llms]

    logger.info("=" * 60)
    logger.info("CFD Evaluation on Code Agents")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"CFDs: {cfd_names}")
    logger.info(f"Environments: {[e.name for e in envs]}")
    logger.info(f"LLMs: {[l.name for l in llms]}")
    logger.info(f"Seeds: {args.num_seeds}, Games: {args.num_game_runs}")
    logger.info(f"Mode: {'critic' if args.use_critic else 'prompt_injection'}")
    logger.info("=" * 60)

    # Save config
    config = {
        "experiment_name": args.experiment_name,
        "cfds": {name: cfd for name, cfd in zip(cfd_names, cfds)},
        "envs": [e.name for e in envs],
        "llms": [l.name for l in llms],
        "num_seeds": args.num_seeds,
        "num_game_runs": args.num_game_runs,
        "use_critic": args.use_critic,
        "timestamp": timestamp,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    langfuse_session_id = f"{args.experiment_name}_{timestamp}"

    all_results = {}
    for cfd, cfd_name in zip(cfds, cfd_names):
        for env in envs:
            try:
                scores = run_eval_for_cfd_env(
                    cfd=cfd,
                    cfd_name=cfd_name,
                    env=env,
                    llms=llms,
                    num_seeds=args.num_seeds,
                    num_game_runs=args.num_game_runs,
                    use_critic=args.use_critic,
                    output_dir=output_dir,
                    langfuse_session_id=langfuse_session_id,
                )
                all_results[f"{cfd_name}/{env.value}"] = scores
            except Exception as e:
                logger.exception(f"Failed for {cfd_name}/{env.value}: {e}")
                all_results[f"{cfd_name}/{env.value}"] = {"error": str(e)}

    # Save aggregate results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'CFD':<20} {'Env':<25} {'XP_llm':>10} {'XP_seed':>10} {'SP':>10}")
    logger.info("-" * 80)
    for key, scores in all_results.items():
        if "error" in scores:
            logger.info(f"{key:<45} ERROR: {scores['error'][:30]}")
        else:
            cfd_name, env_name = key.split("/")
            xp_llm = scores["xp_llm"]["overall"]["xp_llm_mean"]
            xp_seed = scores["xp_seed"]["overall"]["xp_seed_mean"]
            sp = scores["xp_llm"]["overall"]["sp_mean"]
            logger.info(f"{cfd_name:<20} {env_name:<25} {xp_llm:>10.2f} {xp_seed:>10.2f} {sp:>10.2f}")


if __name__ == "__main__":
    main()
