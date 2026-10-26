#!/usr/bin/env python3
"""
Compute XP matrices from pre-generated policies (e.g. from final_v1.tar.gz).

For a given (condition, environment):
  1. Load policies_0.json ... policies_5.json
  2. Run 4x4 XP_llm evaluations per seed (6 matrices)
  3. Run 2x2 XP_seed evaluations per (LLM, seed pair) — 3 pairs x 4 LLMs = 12 matrices
  4. Save all raw matrices for post-processing

Usage:
    python -m meta_train.population_based.scripts.run_xp_from_policies \
        --experiment_name xp_eval_cfd \
        --policies_dir data/final_v1/code/CFD/cat_and_dog \
        --env cat_and_dog \
        --num_seeds 6 \
        --llm_indices 0 1 2 3 \
        --num_game_runs 100
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from configs.configs import (
    CodeXPEvaluatorConfig,
    DEFAULT_CODE_POLICY_AGENT_CONFIG_FACTORY,
)
from configs.envs import (
    CatAndDogConfig,
    DEFAULT_AD_HOC_CONVENTIONS_CONFIG,
    DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG,
    DEFAULT_BRITTLE_BRIDGE_CONFIG_CODE,
    DEFAULT_NARROW_TUNNEL_CONFIG_CODE,
    DEFAULT_TINY_HANABI_CONFIG,
    DEFAULT_TINY_HANABI_FIXED_CONFIG,
    DEFAULT_MATRIX_GAME_CONFIG,
    DEFAULT_PERMUTED_LEVERS_CONFIG,
    EnvConfig,
)
from core import CodePolicy
from train.xp_eval import CodeXPEvaluator


logger = logging.getLogger(__name__)

ENV_CONFIGS: Dict[str, EnvConfig] = {
    "cat_and_dog": CatAndDogConfig(),
    "ad_hoc_conventions": DEFAULT_AD_HOC_CONVENTIONS_CONFIG,
    "ad_hoc_conventions_permuted_signals": DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG,
    "tiny_hanabi": DEFAULT_TINY_HANABI_CONFIG,
    "tiny_hanabi_fixed": DEFAULT_TINY_HANABI_FIXED_CONFIG,
    "brittle_bridge": DEFAULT_BRITTLE_BRIDGE_CONFIG_CODE,
    "narrow_tunnel": DEFAULT_NARROW_TUNNEL_CONFIG_CODE,
    "matrix_game": DEFAULT_MATRIX_GAME_CONFIG,
    "permuted_levers": DEFAULT_PERMUTED_LEVERS_CONFIG,
}


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


def load_policies(policies_dir: Path, seed_idx: int, llm_indices: List[int]) -> List[CodePolicy]:
    """Load policies for a given seed, selecting only the specified LLM indices."""
    path = policies_dir / f"policies_{seed_idx}.json"
    with open(path) as f:
        data = json.load(f)

    policies = []
    for idx in llm_indices:
        entry = data[idx]
        policies.append(CodePolicy(
            name=entry.get("name", "policy"),
            joint_policy=list(entry["joint_policy"]),
        ))
    return policies


def run_xp_grid(env_config: EnvConfig, policies: List[CodePolicy], num_game_runs: int) -> np.ndarray:
    """Run NxN cross-play evaluation."""
    evaluator_config = CodeXPEvaluatorConfig(
        agent_config_factory=DEFAULT_CODE_POLICY_AGENT_CONFIG_FACTORY,
        num_game_runs=num_game_runs,
        num_agents=2,
    )
    evaluator_config.env_config = env_config
    evaluator = CodeXPEvaluator(evaluator_config)
    xp_matrix, _ = evaluator.eval_grid(policies, policies, no_progress_bar=False)
    return xp_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute XP matrices from pre-generated policies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--policies_dir", type=str, required=True,
                        help="Path to directory containing policies_0.json ... policies_N.json")
    parser.add_argument("--env", type=str, required=True, choices=list(ENV_CONFIGS.keys()))
    parser.add_argument("--num_seeds", type=int, default=6)
    parser.add_argument("--llm_indices", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="Indices into the policies JSON to select LLMs (default: 0-3, excluding DeepSeek)")
    parser.add_argument("--num_game_runs", type=int, default=100)
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

    policies_dir = Path(args.policies_dir)
    env_config = ENV_CONFIGS[args.env]

    # Get LLM names from first policies file
    with open(policies_dir / "policies_0.json") as f:
        first_seed = json.load(f)
    llm_names = [first_seed[i]["llm"] for i in args.llm_indices]

    logger.info("=" * 60)
    logger.info("XP Matrix Computation from Pre-generated Policies")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Policies dir: {policies_dir}")
    logger.info(f"Environment: {args.env}")
    logger.info(f"LLMs: {llm_names}")
    logger.info(f"Seeds: {args.num_seeds}, Games: {args.num_game_runs}")
    logger.info("=" * 60)

    # Save config
    config = {
        "experiment_name": args.experiment_name,
        "policies_dir": str(policies_dir),
        "env": args.env,
        "llm_names": llm_names,
        "llm_indices": args.llm_indices,
        "num_seeds": args.num_seeds,
        "num_game_runs": args.num_game_runs,
        "timestamp": timestamp,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # =========================================================================
    # Phase 1: XP_llm matrices (NxN per seed)
    # =========================================================================
    logger.info("")
    logger.info("=== Phase 1: XP_llm matrices ===")

    xp_llm_dir = output_dir / "xp_llm"
    xp_llm_dir.mkdir(parents=True, exist_ok=True)

    all_policies: List[List[CodePolicy]] = []

    for seed_idx in range(args.num_seeds):
        matrix_path = xp_llm_dir / f"xp_matrix_{seed_idx}.json"

        # Resume support
        if matrix_path.exists():
            logger.info(f"  Seed {seed_idx}: already computed, skipping")
            all_policies.append(load_policies(policies_dir, seed_idx, args.llm_indices))
            continue

        logger.info(f"  Seed {seed_idx + 1}/{args.num_seeds}...")
        start_time = time.time()

        policies = load_policies(policies_dir, seed_idx, args.llm_indices)
        all_policies.append(policies)

        xp_matrix = run_xp_grid(env_config, policies, args.num_game_runs)

        with open(matrix_path, "w") as f:
            json.dump(xp_matrix.tolist(), f)

        elapsed = time.time() - start_time
        logger.info(f"    Done in {elapsed:.0f}s")
        logger.info(f"    XP matrix:\n{np.array2string(xp_matrix, precision=2)}")

    # =========================================================================
    # Phase 2: XP_seed matrices (2x2 per LLM per pair)
    # =========================================================================
    logger.info("")
    logger.info("=== Phase 2: XP_seed matrices ===")

    xp_seed_dir = output_dir / "xp_seed"
    xp_seed_dir.mkdir(parents=True, exist_ok=True)

    n_pairs = args.num_seeds // 2
    n_llms = len(args.llm_indices)

    for llm_idx in range(n_llms):
        llm_name = llm_names[llm_idx]
        for pair_idx in range(n_pairs):
            seed_a = 2 * pair_idx
            seed_b = 2 * pair_idx + 1

            matrix_path = xp_seed_dir / f"xp_seed_llm{llm_idx}_pair{pair_idx}.json"

            # Resume support
            if matrix_path.exists():
                logger.info(f"  {llm_name} pair ({seed_a},{seed_b}): already computed, skipping")
                continue

            logger.info(f"  {llm_name} pair ({seed_a},{seed_b})...")
            start_time = time.time()

            policy_a = all_policies[seed_a][llm_idx]
            policy_b = all_policies[seed_b][llm_idx]

            xp_matrix = run_xp_grid(env_config, [policy_a, policy_b], args.num_game_runs)

            with open(matrix_path, "w") as f:
                json.dump(xp_matrix.tolist(), f)

            xp_seed_score = (xp_matrix[0, 1] + xp_matrix[1, 0]) / 2.0
            elapsed = time.time() - start_time
            logger.info(f"    Done in {elapsed:.0f}s, XP_seed={xp_seed_score:.2f}")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("")
    logger.info("=== Summary ===")

    # Load all XP_llm matrices
    xp_llm_matrices = []
    for seed_idx in range(args.num_seeds):
        with open(xp_llm_dir / f"xp_matrix_{seed_idx}.json") as f:
            xp_llm_matrices.append(np.array(json.load(f)))

    tensor = np.stack(xp_llm_matrices, axis=0)  # (n_seeds, n_llms, n_llms)
    avg_matrix = np.mean(tensor, axis=0)

    logger.info(f"Average XP_llm matrix:\n{np.array2string(avg_matrix, precision=2)}")

    # Per-LLM SP and XP_llm
    for i, name in enumerate(llm_names):
        sp_vals = tensor[:, i, i]
        mask = np.arange(n_llms) != i
        xp_vals = np.concatenate([avg_matrix[i, mask], avg_matrix[mask, i]])
        logger.info(f"  {name}: SP={np.mean(sp_vals):.2f}+/-{np.std(sp_vals)/np.sqrt(len(sp_vals)):.2f}, "
                     f"XP_llm={np.mean(xp_vals):.2f}+/-{np.std(xp_vals)/np.sqrt(len(xp_vals)):.2f}")

    # Per-LLM XP_seed
    for llm_idx in range(n_llms):
        samples = []
        for pair_idx in range(n_pairs):
            with open(xp_seed_dir / f"xp_seed_llm{llm_idx}_pair{pair_idx}.json") as f:
                m = np.array(json.load(f))
            samples.append((m[0, 1] + m[1, 0]) / 2.0)
        samples = np.array(samples)
        logger.info(f"  {llm_names[llm_idx]}: XP_seed={np.mean(samples):.2f}+/-{np.std(samples)/np.sqrt(len(samples)):.2f}")

    logger.info("")
    logger.info(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
