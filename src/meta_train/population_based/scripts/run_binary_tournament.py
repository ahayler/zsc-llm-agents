#!/usr/bin/env python3
"""
Run the Binary Tournament Evolutionary Algorithm for CFD optimization.

Example usage:
    python -m meta_train.population_based.scripts.run_binary_tournament \
        --experiment_name my_experiment \
        --env CAT_AND_DOG \
        --population_size 5 \
        --num_steps 10

Docker/cluster usage:
    The script creates a timestamped output directory with:
    - population.pkl: Saved population state after each step
    - console.log: All console output piped to file
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from configs.contexts import CoordinationGame
from llm.request import LLMType
from meta_train.population_based.algorithms.binary_tournament import BinaryTournamentOptimizer, MUTATION_DISTRIBUTIONS


def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Setup logging to both console and file.
    
    Args:
        output_dir: Directory to save console.log file.
    
    Returns:
        Configured logger.
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    class ShortNameFormatter(logging.Formatter):
        def format(self, record):
            record.shortname = record.name.rsplit(".", 1)[-1]
            return super().format(record)
    
    formatter = ShortNameFormatter(
        "%(asctime)s - %(shortname)s - %(levelname)s - %(message)s",
        datefmt="%H:%M"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    log_file = output_dir / "console.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to console and {log_file}")
    
    return logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Binary Tournament EA for CFD optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment (used for output directory)",
    )
    
    # Environment arguments
    parser.add_argument(
        "--env",
        type=str,
        default="CAT_AND_DOG",
        choices=[e.name for e in CoordinationGame],
        help="Environment to evaluate CFDs on",
    )
    
    # Algorithm parameters
    parser.add_argument(
        "--population_size",
        type=int,
        default=5,
        help="Number of genotypes in the population",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of tournament steps to run",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=7,
        help="Number of seeds per environment during evaluation",
    )
    parser.add_argument(
        "--max_bullet_points",
        type=int,
        default=5,
        help="Maximum number of bullet points in a CFD",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    # LLM arguments
    parser.add_argument(
        "--optimizer_llm",
        type=str,
        default="OR_GEMINI_PRO_3_PREVIEW",
        help="LLM for optimization/mutation operations",
    )
    parser.add_argument(
        "--scoring_llm",
        type=str,
        default="OR_GROK_41_FAST",
        help="LLM for policy generation during scoring",
    )
    
    # Caching arguments
    parser.add_argument(
        "--no_cached_generations",
        action="store_true",
        help="Disable cached policy generations (caching is enabled by default)",
    )
    parser.add_argument(
        "--sp_filtering",
        action="store_true",
        help="Enable self-play filtering during training/evaluation",
    )
    parser.add_argument(
        "--mutation_distribution",
        type=str,
        default="default",
        help="Named mutation operator distribution (default, with_feedback)",
    )
    parser.add_argument(
        "--no_judge_independently",
        action="store_true",
        help="Disable independent judging of local policies (default: critic judges each local policy independently)",
    )
    parser.add_argument(
        "--num_invalid_action_policies",
        type=int,
        default=10,
        help="Number of policies from cache for invalid action filtering (0 = self-play filtering)",
    )

    # Logging arguments
    parser.add_argument(
        "--no_separate_scoring_log",
        action="store_true",
        help="Disable separate scoring log file (by default, scoring logs go to 'scoring.log')",
    )
    parser.add_argument(
        "--scoring_log_to_console",
        action="store_true",
        help="Also print scoring logs to console/main log (by default, they only go to scoring.log)",
    )
    parser.add_argument(
        "--num_policies_per_seed",
        type=int,
        default=1,
        help="Number of policies to generate per seed",
    )
    parser.add_argument(
        "--initial_population_path",
        type=str,
        default=None,
        help="Path to a pickle file containing an initial population to load instead of generating new seeds",
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Create timestamped output directory
    # Walk up from CWD to find the project root (contains pyproject.toml)
    project_root = Path.cwd()
    for parent in [project_root] + list(project_root.parents):
        if (parent / "pyproject.toml").exists():
            project_root = parent
            break
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "logs" / args.experiment_name / timestamp
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("Binary Tournament CFD Optimization")
    logger.info("=" * 60)
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Population size: {args.population_size}")
    logger.info(f"Number of steps: {args.num_steps}")
    logger.info(f"Number of seeds: {args.num_seeds}")
    logger.info(f"Max bullet points: {args.max_bullet_points}")
    logger.info(f"Random state: {args.random_state}")
    logger.info(f"Optimizer LLM: {args.optimizer_llm}")
    logger.info(f"Scoring LLM: {args.scoring_llm}")
    logger.info(f"Use cached generations: {not args.no_cached_generations}")
    logger.info(f"SP filtering enabled: {args.sp_filtering}")
    logger.info(f"Separate scoring log: {not args.no_separate_scoring_log}")
    logger.info(f"Scoring log to console: {args.scoring_log_to_console}")
    logger.info(f"Num policies per seed: {args.num_policies_per_seed}")
    logger.info(f"Initial population path: {args.initial_population_path}")
    logger.info(f"Mutation distribution: {args.mutation_distribution}")
    logger.info(f"Judge independently: {not args.no_judge_independently}")
    logger.info(f"Num base policies: {args.num_invalid_action_policies}")
    logger.info("=" * 60)

    # Parse mutation distribution
    if args.mutation_distribution not in MUTATION_DISTRIBUTIONS:
        raise ValueError(f"Unknown mutation distribution: {args.mutation_distribution}. Available: {list(MUTATION_DISTRIBUTIONS.keys())}")
    mutation_operator_distribution = MUTATION_DISTRIBUTIONS[args.mutation_distribution]

    # Parse environment
    env = CoordinationGame[args.env]
    
    # Parse LLMs
    optimizer_llm = LLMType[args.optimizer_llm]
    scoring_llm = LLMType[args.scoring_llm]
    
    # Create langfuse session ID from experiment name and timestamp
    langfuse_session_id = f"{args.experiment_name}_{timestamp}"
    
    # Set up scoring log path (enabled by default, disabled with --no_separate_scoring_log)
    scoring_log_path = None if args.no_separate_scoring_log else output_dir / "scoring.log"
    # Exclusive by default (scoring logs only go to scoring.log), unless --scoring_log_to_console is set
    exclusive_scoring_log = not args.scoring_log_to_console
    
    # Parse initial population path
    initial_population_path = Path(args.initial_population_path) if args.initial_population_path else None
    
    # Create optimizer
    optimizer = BinaryTournamentOptimizer(
        population_size=args.population_size,
        num_steps=args.num_steps,
        num_seeds=args.num_seeds,
        random_state=args.random_state,
        optimizer_llm=optimizer_llm,
        scoring_llm=scoring_llm,
        save_path=output_dir,
        max_bullet_points=args.max_bullet_points,
        use_cached_generations=not args.no_cached_generations,
        langfuse_session_id=langfuse_session_id,
        scoring_log_path=scoring_log_path,
        exclusive_scoring_log=exclusive_scoring_log,
        use_sp_filtering=args.sp_filtering,
        initial_population_path=initial_population_path,
        num_policies_per_seed=args.num_policies_per_seed,
        mutation_operator_distribution=mutation_operator_distribution,
        judge_independently=not args.no_judge_independently,
        num_invalid_action_policies=args.num_invalid_action_policies,
    )
    
    # Run optimization
    try:
        best_cfd = optimizer.find_cfd(envs=[env])
        
        # Save final CFD to file
        cfd_file = output_dir / "best_cfd.txt"
        with open(cfd_file, "w") as f:
            f.write(best_cfd)
        logger.info(f"Best CFD saved to {cfd_file}")
        
        logger.info("=" * 60)
        logger.info("Optimization completed successfully!")
        logger.info(f"Best CFD:\n{best_cfd}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.exception(f"Optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()

