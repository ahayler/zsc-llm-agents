#!/usr/bin/env python3
"""
Run the Pareto Evolution Algorithm for CFD optimization.

Example usage:
    python -m meta_train.population_based.scripts.run_pareto_evolution \
        --experiment_name my_experiment \
        --envs CAT_AND_DOG BRITTLE_BRIDGE \
        --population_size 5 \
        --num_steps 20
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from configs.contexts import CoordinationGame
from llm.request import LLMType
from meta_train.population_based.algorithms.pareto_evolution import ParetoEvolutionOptimizer
from meta_train.population_based.algorithms.binary_tournament import MUTATION_DISTRIBUTIONS


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

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to console and {log_file}")
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Pareto Evolution for CFD optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--envs", type=str, nargs="+", default=["CAT_AND_DOG", "BRITTLE_BRIDGE"],
        choices=[e.name for e in CoordinationGame],
        help="Environments to optimize CFDs on",
    )
    parser.add_argument("--population_size", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--num_seeds_cheap", type=int, default=10)
    parser.add_argument("--num_seeds_full", type=int, default=15)
    parser.add_argument("--max_bullet_points", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--optimizer_llm", type=str, default="OR_GPT54")
    parser.add_argument("--scoring_llm", type=str, default="OR_GROK_41_FAST")
    parser.add_argument("--num_policies_per_seed", type=int, default=1)
    parser.add_argument("--no_cached_generations", action="store_true")
    parser.add_argument("--sp_filtering", action="store_true")
    parser.add_argument(
        "--mutation_distribution", type=str, default="default",
        help="Named mutation operator distribution (default, with_feedback)",
    )
    parser.add_argument(
        "--initial_state_path", type=str, default=None,
        help="Path to a pareto_state.pkl file to resume from",
    )

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

    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("Pareto Evolution CFD Optimization")
    logger.info("=" * 60)
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Environments: {args.envs}")
    logger.info(f"Population size: {args.population_size}")
    logger.info(f"Number of steps: {args.num_steps}")
    logger.info(f"Num seeds (cheap): {args.num_seeds_cheap}")
    logger.info(f"Num seeds (full): {args.num_seeds_full}")
    logger.info(f"Max bullet points: {args.max_bullet_points}")
    logger.info(f"Random state: {args.random_state}")
    logger.info(f"Optimizer LLM: {args.optimizer_llm}")
    logger.info(f"Scoring LLM: {args.scoring_llm}")
    logger.info(f"Num policies per seed: {args.num_policies_per_seed}")
    logger.info(f"Use cached generations: {not args.no_cached_generations}")
    logger.info(f"SP filtering: {args.sp_filtering}")
    logger.info(f"Mutation distribution: {args.mutation_distribution}")
    logger.info(f"Initial state path: {args.initial_state_path}")
    logger.info("=" * 60)

    envs = [CoordinationGame[e] for e in args.envs]
    optimizer_llm = LLMType[args.optimizer_llm]
    scoring_llm = LLMType[args.scoring_llm]
    langfuse_session_id = f"{args.experiment_name}_{timestamp}"

    if args.mutation_distribution not in MUTATION_DISTRIBUTIONS:
        raise ValueError(f"Unknown mutation distribution: {args.mutation_distribution}. Available: {list(MUTATION_DISTRIBUTIONS.keys())}")
    mutation_operator_distribution = MUTATION_DISTRIBUTIONS[args.mutation_distribution]

    scoring_log_path = output_dir / "scoring.log"

    optimizer = ParetoEvolutionOptimizer(
        population_size=args.population_size,
        num_steps=args.num_steps,
        num_seeds_cheap=args.num_seeds_cheap,
        num_seeds_full=args.num_seeds_full,
        random_state=args.random_state,
        optimizer_llm=optimizer_llm,
        scoring_llm=scoring_llm,
        save_path=output_dir,
        max_bullet_points=args.max_bullet_points,
        use_cached_generations=not args.no_cached_generations,
        mutation_operator_distribution=mutation_operator_distribution,
        langfuse_session_id=langfuse_session_id,
        scoring_log_path=scoring_log_path,
        exclusive_scoring_log=True,
        use_sp_filtering=args.sp_filtering,
        num_policies_per_seed=args.num_policies_per_seed,
        initial_state_path=Path(args.initial_state_path) if args.initial_state_path else None,
    )

    try:
        best_cfd = optimizer.find_cfd(envs=envs)

        cfd_file = output_dir / "best_cfd.txt"
        with open(cfd_file, "w") as f:
            f.write(best_cfd)
        logger.info(f"Best CFD saved to {cfd_file}")

    except Exception as e:
        logger.exception(f"Optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()
