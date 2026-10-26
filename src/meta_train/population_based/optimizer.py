"""
Abstract base class for population-based CFD optimizers.
"""
import logging
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from configs.configs import CachedGenerationConfig
from configs.contexts import CoordinationGame
from llm.request import LLMType
from train.pipeline import train_and_xp_eval_code
from meta_train.population_based.population import Genotype, Population, ScoreEntry
from utils.config_setup import setup_code_configs_from_default_game_context
from core import CodePolicy
from utils.log_utils import get_logger
from utils.saving import CodeXPExperimentResults, load_most_recent_experiment_results_for_env

logger = get_logger(__name__)


# Global state for thread-safe scoped_file_logging reference counting.
# Keyed by (log_path, exclusive) to allow concurrent use of different log paths.
_scoped_logging_lock = threading.Lock()
_scoped_logging_refs: dict = {}  # log_path -> {handler, refcount, original_handlers}


@contextmanager
def scoped_file_logging(log_path: Path, exclusive: bool = False):
    """
    Thread-safe context manager that adds a file handler to the root logger.

    When multiple threads enter with the same log_path, the file handler is
    added only once (on first entry) and removed only when the last thread exits.
    This prevents duplicate log lines and broken exclusive-mode handler swaps.

    Args:
        log_path: Path to the log file.
        exclusive: If True, temporarily disable other handlers so logs only go to this file.
                   If False, logs go to both this file and existing handlers.
    """
    root_logger = logging.getLogger()
    key = str(log_path)

    with _scoped_logging_lock:
        if key in _scoped_logging_refs:
            _scoped_logging_refs[key]["refcount"] += 1
        else:
            class ShortNameFormatter(logging.Formatter):
                def format(self, record):
                    record.shortname = record.name.rsplit(".", 1)[-1]
                    return super().format(record)

            formatter = ShortNameFormatter(
                "%(asctime)s - %(shortname)s - %(levelname)s - %(message)s",
                datefmt="%H:%M"
            )

            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)

            original_handlers = []
            if exclusive:
                original_handlers = root_logger.handlers[:]
                for handler in original_handlers:
                    root_logger.removeHandler(handler)

            root_logger.addHandler(file_handler)

            _scoped_logging_refs[key] = {
                "handler": file_handler,
                "refcount": 1,
                "original_handlers": original_handlers,
                "exclusive": exclusive,
            }

    try:
        yield
    finally:
        with _scoped_logging_lock:
            entry = _scoped_logging_refs[key]
            entry["refcount"] -= 1
            if entry["refcount"] == 0:
                root_logger.removeHandler(entry["handler"])
                entry["handler"].close()

                if entry["exclusive"]:
                    for handler in entry["original_handlers"]:
                        root_logger.addHandler(handler)

                del _scoped_logging_refs[key]


class PopulationOptimizer(ABC):
    """
    Abstract base class for population-based CFD optimization.
    
    Provides common infrastructure for managing populations and scoring genotypes.
    Subclasses implement specific optimization algorithms (e.g., binary tournament).
    """
    
    def __init__(
        self,
        population_size: int,
        random_state: int = 42,
        optimizer_llm: LLMType = LLMType.OR_CLAUDE_SONNET_45,
        scoring_llm: LLMType = LLMType.OR_GROK_41_FAST,
        save_path: Optional[Path] = None,
        use_sp_filtering: bool = False,
        num_policies_per_seed: Optional[int] = None,
        judge_independently: bool = True,
        num_invalid_action_policies: Optional[int] = None,
    ):
        """
        Initialize the population optimizer.

        Args:
            population_size: Number of genotypes in the population.
            random_state: Seed for the random number generator.
            optimizer_llm: LLM to use for mutation/optimization operations.
            scoring_llm: LLM to use for policy generation during scoring.
            save_path: Optional path to save population state.
            use_sp_filtering: Enable self-play filtering during training.
            num_policies_per_seed: Number of policies to generate per seed.
                                  If None, uses the default from the trainer config.
            judge_independently: If True, the critic judges each local policy independently.
                                If False, the critic judges the joint policy as a whole.
            num_invalid_action_policies: Number of policies to load from cache for
                                        invalid action filtering. If None or 0, filtering
                                        uses self-play among generated policies.
        """
        self.population_size = population_size
        self.optimizer_llm = optimizer_llm
        self.scoring_llm = scoring_llm
        self.save_path = Path(save_path) if save_path else None
        self.rng = np.random.default_rng(random_state)
        self.population = Population()
        self.use_sp_filtering = use_sp_filtering
        self.num_policies_per_seed = num_policies_per_seed
        self.judge_independently = judge_independently
        self._invalid_action_policies: Optional[dict] = None  # env -> List[CodePolicy]
        self.num_invalid_action_policies = num_invalid_action_policies

    def _get_invalid_action_policies(self, env: CoordinationGame) -> Optional[List[CodePolicy]]:
        """Load and cache policies for invalid action filtering from the coordination_failure_starting_point cache."""
        if self.num_invalid_action_policies is None or self.num_invalid_action_policies == 0:
            return None
        if self._invalid_action_policies is None:
            self._invalid_action_policies = {}
        if env not in self._invalid_action_policies:
            results = load_most_recent_experiment_results_for_env(
                "coordination_failure_starting_point", env.value, mode="code",
            )
            all_policies = []
            for policies in results.generated_policies.values():
                all_policies.extend(policies)
            n = min(self.num_invalid_action_policies, len(all_policies))
            indices = self.rng.choice(len(all_policies), size=n, replace=False)
            self._invalid_action_policies[env] = [all_policies[i] for i in indices]
            logger.info(f"Loaded {n} invalid action filtering policies for {env.value}")
        return self._invalid_action_policies[env]

    @abstractmethod
    def find_cfd(self, envs: List[CoordinationGame]) -> str:
        """
        Run the optimization algorithm and return the best CFD found.
        
        Args:
            envs: List of environments to evaluate CFDs on.
        
        Returns:
            The best CFD string found by the optimizer.
        """
        pass
    
    def score_genotypes(
        self,
        genotypes: List[Genotype],
        envs: List[CoordinationGame],
        num_seeds: int,
        seed: Optional[int] = None,
        max_workers: int = 128,
        use_cached_generations: bool = True,
        langfuse_session_id: Optional[str] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        scoring_log_path: Optional[Path] = None,
        exclusive_scoring_log: bool = False,
    ) -> None:
        """
        Score a list of genotypes by evaluating them on the given environments.

        Uses train_and_xp_eval_code from main_code.py for full pipeline evaluation
        including policy generation, policy critic filtering, and XP evaluation.

        We provide the option to fix a seed across all genotypes to make the generated XP-matrices as comparable as possible.

        Args:
            genotypes: List of genotypes to score.
            envs: List of environments to evaluate on.
            num_seeds: Number of seeds per environment evaluation.
            seed: Optional seed that makes a lot the randomness more comparable between the different genotypes.
            In particular, it makes sure that each XP-matrix has the same cached policies in each seed (if we use caching).
            Additionally, it makes sure that the policy selection steps are as comparable as possible.
            TODO: Look again at whether it makes sense to propagate the seed up to the LLM.
            max_workers: Maximum number of parallel workers (default: 128).
            use_cached_generations: Whether to use cached policy generations (default: True).
            langfuse_session_id: Optional session ID for Langfuse tracing.
            executor: Optional ThreadPoolExecutor to use for parallel execution.
                     If not provided, a new executor with max_workers will be created.
            scoring_log_path: Optional path to write scoring logs to a separate file.
                             If provided, all logs from this function (and nested calls)
                             will be written to this file.
            exclusive_scoring_log: If True, scoring logs will ONLY go to the scoring log file
                                   (not to the main console/log). Default is False (logs go to both).
        """
        def score_single(genotype: Genotype, exec: Optional[ThreadPoolExecutor] = None) -> Tuple[Genotype, float, List[CodeXPExperimentResults]]:
            all_xp_scores = []
            results_list = []

            for env in envs:
                # Setup configs like main_code.py does
                trainer_config, evaluator_config = setup_code_configs_from_default_game_context(env)
                trainer_config.use_policy_critic = True
                trainer_config.policy_critic_config.coordination_friendly_definition = genotype.cfd
                trainer_config.sp_filtering = self.use_sp_filtering

                # Override the LLM for policy generation and policy critic
                trainer_config.generator_config.llm_config.model = self.scoring_llm
                trainer_config.policy_critic_config.llm_config.model = self.scoring_llm

                # Override number of policies per seed if specified
                if self.num_policies_per_seed is not None:
                    trainer_config.generator_config.num_policies = self.num_policies_per_seed

                # Set independent judging mode
                trainer_config.policy_critic_config.judge_independently = self.judge_independently

                # Configure cached generations if enabled
                if use_cached_generations:
                    trainer_config.generator_config.use_cached_generations = True
                    trainer_config.generator_config.cached_generations_config = CachedGenerationConfig(
                        experiment_name="coordination_failure_starting_point",
                        env_name_file_safe=env.value,
                    )

                # Load invalid action filtering policies if configured
                ia_policies = self._get_invalid_action_policies(env)

                # Run full training + XP evaluation pipeline (always N×N)
                logger.info(f"Scoring genotype on {env.value} with CFD: {genotype.cfd[:50]}...")
                experiment_results = train_and_xp_eval_code(
                    num_seeds, trainer_config, evaluator_config,
                    langfuse_session_id=langfuse_session_id,
                    executor=exec,
                    base_seed=seed,
                    invalid_action_policies=ia_policies,
                )

                xp_matrix = experiment_results.xp_matrix
                # N×N matrix: exclude diagonal (self-play)
                xp_scores = xp_matrix[~np.eye(xp_matrix.shape[0], dtype=bool)]
                all_xp_scores.extend(xp_scores.flatten())
                results_list.append(experiment_results)

            mean_score = float(np.nanmean(all_xp_scores))
            logger.info(f"Genotype scored: {mean_score:.4f}")
            return genotype, mean_score, results_list
        
        def run_scoring(exec: ThreadPoolExecutor):
            # Score genotypes in parallel using the executor
            futures = {exec.submit(score_single, g, exec): g for g in genotypes}
            for future in as_completed(futures):
                genotype, score, results = future.result()
                genotype.score_history.append(ScoreEntry(
                    experiment_results=results[0] if len(results) == 1 else results,
                    score=score,
                ))
        
        def execute_scoring():
            if executor is not None:
                # Use provided executor
                run_scoring(executor)
            else:
                # Create new executor with max_workers
                with ThreadPoolExecutor(max_workers=max_workers) as new_executor:
                    run_scoring(new_executor)
        
        # Optionally wrap scoring in scoped file logging
        if scoring_log_path is not None:
            with scoped_file_logging(scoring_log_path, exclusive=exclusive_scoring_log):
                execute_scoring()
        else:
            execute_scoring()
    
    def save_population(self) -> None:
        """Save the population to the configured save path."""
        if self.save_path:
            save_file = self.save_path / "population.pkl"
            self.population.save(save_file)
            logger.info(f"Population saved to {save_file}")

    def _log_trace_and_xp(self, genotype: Genotype) -> None:
        """Log trace ID and XP matrix for the latest score entry of a genotype."""
        if not genotype.score_history:
            return
        score_entry = genotype.score_history[-1]
        experiment_results = score_entry.experiment_results
        results_list = experiment_results if isinstance(experiment_results, list) else [experiment_results]
        for res in results_list:
            trace_id = getattr(res, "trace_id", None)
            xp_matrix = getattr(res, "xp_matrix", None)
            logger.info(
                "Trace ID for CFD (env=%s): %s",
                res.env_name_file_safe,
                trace_id if trace_id is not None else "N/A",
            )
            if xp_matrix is not None:
                logger.info(
                    "XP matrix for CFD on %s:\n%s",
                    res.env_name_file_safe,
                    np.array2string(xp_matrix, precision=4, floatmode="fixed"),
                )

