"""
Pareto-based Evolutionary Algorithm for CFD optimization.

Inspired by GEPA, uses a proper Pareto frontier over multiple environments
with cheap single-env gating before full evaluation.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import numpy as np
from langfuse import observe
from utils.tracing import langfuse_client

from configs.configs import CachedGenerationConfig, LLMConfig
from configs.contexts import CoordinationGame
from core import CodePolicy
from llm.request import LLMType
from train.pipeline import train_and_xp_eval_code
from meta_train.population_based.optimizer import scoped_file_logging
from meta_train.population_based.mutate import (
    DiverseSeedGenerator,
    SampleDistribution,
    create_mutation_operator,
)
from meta_train.population_based.algorithms.binary_tournament import (
    DEFAULT_MUTATION_OPERATOR_DISTRIBUTION,
)
from utils.config_setup import setup_code_configs_from_default_game_context
from utils.log_utils import get_logger
from utils.saving import (
    CodeXPExperimentResults,
    load_most_recent_experiment_results_for_env,
)

logger = get_logger(__name__)


@dataclass
class ParetoCandidate:
    """A candidate CFD with per-environment scores."""
    cfd: str
    env_scores: Dict[str, float] = field(default_factory=dict)
    # Per-env XP matrices from the full eval
    xp_matrices: Dict[str, np.ndarray] = field(default_factory=dict)
    parent_cfd: Optional[str] = None
    mutation_operator_name: Optional[str] = None
    step_created_at: int = 0
    is_alive: bool = True  # False if removed from Pareto frontier


@dataclass
class ParetoState:
    """Serializable state for the Pareto evolution algorithm."""
    candidates: List[ParetoCandidate] = field(default_factory=list)
    step_counter: int = 0
    history: List[dict] = field(default_factory=list)
    # Saved RNG state for resuming
    rng_state: Optional[dict] = None
    # Fixed seed used for frontier evaluations
    frontier_eval_seed: Optional[int] = None
    # Pre-sampled starting policies per env per seed (for frontier evals)
    frontier_starting_policies: Optional[Dict[str, Dict[int, list]]] = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "ParetoState":
        with open(path, 'rb') as f:
            return pickle.load(f)


def compute_pareto_frontier(
    candidates: List[ParetoCandidate],
    env_names: List[str],
) -> List[ParetoCandidate]:
    """
    Compute the Pareto frontier: iteratively remove dominated candidates.

    A candidate A dominates B iff A.env_scores[e] >= B.env_scores[e] for all envs
    and A.env_scores[e] > B.env_scores[e] for at least one env.

    Returns the list of non-dominated candidates.
    """
    alive = [c for c in candidates if c.is_alive]
    if not alive:
        return []

    non_dominated = []
    for candidate in alive:
        dominated = False
        for other in alive:
            if other is candidate:
                continue
            # Check if other dominates candidate
            all_geq = all(
                other.env_scores.get(e, float('-inf')) >= candidate.env_scores.get(e, float('-inf'))
                for e in env_names
            )
            any_gt = any(
                other.env_scores.get(e, float('-inf')) > candidate.env_scores.get(e, float('-inf'))
                for e in env_names
            )
            if all_geq and any_gt:
                dominated = True
                break
        if not dominated:
            non_dominated.append(candidate)
    return non_dominated


class ParetoEvolutionOptimizer:
    """
    Pareto-based evolutionary optimizer for CFD optimization.

    Algorithm:
    1. Initialize population with diverse seed CFDs
    2. Evaluate all seeds on all envs against fixed base policies
    3. For each step:
       a. Compute Pareto frontier
       b. Sample uniformly from frontier
       c. Mutate to create child
       d. Cheap eval: pick one random env, sample 10 base policies,
          evaluate both child and parent against them
       e. If child beats parent: full eval on all envs with all base policies
       f. If child makes Pareto frontier, add it (mark dominated candidates)
    4. Return best candidate (highest mean across envs)
    """

    def __init__(
        self,
        population_size: int = 5,
        num_steps: int = 20,
        num_seeds_cheap: int = 10,
        num_seeds_full: int = 15,
        random_state: int = 42,
        optimizer_llm: LLMType = LLMType.OR_GPT54,
        scoring_llm: LLMType = LLMType.OR_GROK_41_FAST,
        save_path: Optional[Path] = None,
        max_bullet_points: int = 5,
        use_cached_generations: bool = True,
        mutation_operator_distribution: Optional[SampleDistribution[str]] = None,
        langfuse_session_id: Optional[str] = None,
        scoring_log_path: Optional[Path] = None,
        exclusive_scoring_log: bool = False,
        use_sp_filtering: bool = False,
        num_policies_per_seed: Optional[int] = None,
        initial_state_path: Optional[Path] = None,
        max_workers: int = 128,
    ):
        self.population_size = population_size
        self.num_steps = num_steps
        self.num_seeds_cheap = num_seeds_cheap
        self.num_seeds_full = num_seeds_full
        self.optimizer_llm = optimizer_llm
        self.scoring_llm = scoring_llm
        self.save_path = Path(save_path) if save_path else None
        self.max_bullet_points = max_bullet_points
        self.use_cached_generations = use_cached_generations
        self.mutation_operator_distribution = mutation_operator_distribution or DEFAULT_MUTATION_OPERATOR_DISTRIBUTION
        self.langfuse_session_id = langfuse_session_id
        self.scoring_log_path = scoring_log_path
        self.exclusive_scoring_log = exclusive_scoring_log
        self.use_sp_filtering = use_sp_filtering
        self.num_policies_per_seed = num_policies_per_seed
        self.initial_state_path = Path(initial_state_path) if initial_state_path else None
        self.max_workers = max_workers
        self.rng = np.random.default_rng(random_state)

        self._optimizer_llm_config = LLMConfig(
            model=optimizer_llm,
            temperature=1.0,
            max_tokens=4096,
        )
        self._seed_generator = DiverseSeedGenerator(self._optimizer_llm_config)

        # Fixed seed for full (frontier) evaluations — ensures only the CFD varies
        self._frontier_eval_seed = int(self.rng.integers(0, 2**32 - 1))

        # State
        self.state = ParetoState()

        # Fixed starting policies per env per seed (used as critic input for frontier evals)
        self._frontier_starting_policies: Dict[CoordinationGame, Dict[int, List[CodePolicy]]] = {}
        # All cached policies per env (full pool to sample from)
        self._all_cached_policies: Dict[CoordinationGame, List[CodePolicy]] = {}

    def _load_starting_policies(self, envs: List[CoordinationGame]) -> None:
        """Load cached policies and pre-sample fixed starting policies for all environments."""
        for env in envs:
            if env in self._all_cached_policies:
                continue
            results = load_most_recent_experiment_results_for_env(
                "coordination_failure_starting_point", env.value, mode="code",
            )
            all_policies = []
            for policies in results.generated_policies.values():
                all_policies.extend(policies)
            self._all_cached_policies[env] = all_policies
            logger.info(f"Loaded {len(all_policies)} cached policies for {env.value}")

            # Pre-sample fixed starting policies for frontier evals (one per seed)
            self._frontier_starting_policies[env] = {}
            num_policies_per_seed = self.num_policies_per_seed or 1
            for seed_id in range(self.num_seeds_full):
                idx = self.rng.choice(len(all_policies), size=num_policies_per_seed, replace=False)
                self._frontier_starting_policies[env][seed_id] = [all_policies[i] for i in idx]
            logger.info(f"Pre-sampled fixed starting policies for {self.num_seeds_full} seeds on {env.value}")

    def _sample_fresh_starting_policies(
        self, env: CoordinationGame,
    ) -> Dict[int, List[CodePolicy]]:
        """Sample fresh (random) starting policies for cheap eval."""
        all_policies = self._all_cached_policies[env]
        num_policies_per_seed = self.num_policies_per_seed or 1
        result = {}
        for seed_id in range(self.num_seeds_cheap):
            idx = self.rng.choice(len(all_policies), size=num_policies_per_seed, replace=False)
            result[seed_id] = [all_policies[i] for i in idx]
        return result

    def _eval_cfd_on_env(
        self,
        cfd: str,
        env: CoordinationGame,
        num_seeds: int,
        preset_policies_per_seed: Optional[Dict[int, List[CodePolicy]]] = None,
        fixed_seed: Optional[int] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> Tuple[float, np.ndarray]:
        """Evaluate a CFD on one environment via N×N cross-play.

        Args:
            cfd: The CFD string to evaluate.
            env: The environment to evaluate on.
            num_seeds: Number of seeds (policies) for the N×N matrix.
            preset_policies_per_seed: If provided, use these as starting policies for the
                                     critic instead of generating/sampling from cache.
            fixed_seed: If provided, use this as base_seed for deterministic evaluation.
                       If None, draw a random seed.

        Returns:
            Tuple of (mean off-diagonal XP score, full XP matrix).
        """
        trainer_config, evaluator_config = setup_code_configs_from_default_game_context(env)
        trainer_config.use_policy_critic = True
        trainer_config.policy_critic_config.coordination_friendly_definition = cfd
        trainer_config.sp_filtering = self.use_sp_filtering
        trainer_config.generator_config.llm_config.model = self.scoring_llm
        trainer_config.policy_critic_config.llm_config.model = self.scoring_llm

        if self.num_policies_per_seed is not None:
            trainer_config.generator_config.num_policies = self.num_policies_per_seed

        if self.use_cached_generations and preset_policies_per_seed is None:
            trainer_config.generator_config.use_cached_generations = True
            trainer_config.generator_config.cached_generations_config = CachedGenerationConfig(
                experiment_name="coordination_failure_starting_point",
                env_name_file_safe=env.value,
            )

        seed = fixed_seed if fixed_seed is not None else int(self.rng.integers(0, 2**32 - 1))

        # Use a subset of cached policies for invalid action filtering
        all_cached = self._all_cached_policies.get(env)
        if all_cached and len(all_cached) > 20:
            idx = self.rng.choice(len(all_cached), size=20, replace=False)
            ia_policies = [all_cached[i] for i in idx]
        else:
            ia_policies = all_cached

        def run_eval():
            return train_and_xp_eval_code(
                num_seeds, trainer_config, evaluator_config,
                langfuse_session_id=self.langfuse_session_id,
                executor=executor,
                base_seed=seed,
                preset_policies_per_seed=preset_policies_per_seed,
                invalid_action_policies=ia_policies,
            )

        if self.scoring_log_path is not None:
            with scoped_file_logging(self.scoring_log_path, exclusive=self.exclusive_scoring_log):
                experiment_results = run_eval()
        else:
            experiment_results = run_eval()

        # N×N matrix, exclude diagonal (self-play)
        xp_matrix = experiment_results.xp_matrix
        xp_scores = xp_matrix[~np.eye(xp_matrix.shape[0], dtype=bool)]
        return float(np.nanmean(xp_scores)), xp_matrix

    def _eval_cfd_all_envs(
        self,
        cfd: str,
        envs: List[CoordinationGame],
        use_fixed_starting_policies: bool = True,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Full evaluation of a CFD on all environments via N×N cross-play (parallel across envs).

        Args:
            cfd: The CFD to evaluate.
            envs: Environments to evaluate on.
            use_fixed_starting_policies: If True, use pre-sampled fixed starting policies
                                        (for Pareto frontier comparability). If False, let
                                        the generator sample fresh ones.
            executor: Shared ThreadPoolExecutor for parallel execution.

        Returns:
            Tuple of (env_name -> mean score, env_name -> XP matrix).
        """
        def eval_one_env(env: CoordinationGame) -> Tuple[str, float, np.ndarray]:
            preset = self._frontier_starting_policies[env] if use_fixed_starting_policies else None
            seed = self._frontier_eval_seed if use_fixed_starting_policies else None
            score, xp_matrix = self._eval_cfd_on_env(cfd, env, num_seeds=self.num_seeds_full, preset_policies_per_seed=preset, fixed_seed=seed, executor=executor)
            logger.info(f"  Full eval on {env.value}: {score:.4f}")
            return env.value, score, xp_matrix

        scores = {}
        xp_matrices = {}
        for env in envs:
            env_name, score, xp_matrix = eval_one_env(env)
            scores[env_name] = score
            xp_matrices[env_name] = xp_matrix
        return scores, xp_matrices

    def _cheap_eval(
        self,
        child_cfd: str,
        parent_cfd: str,
        env: CoordinationGame,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> Tuple[float, float]:
        """
        Cheap evaluation: evaluate both child and parent with the same fresh
        (random) starting policies on one env via N×N cross-play.

        Returns (child_score, parent_score).
        """
        # Both child and parent use the SAME fresh starting policies for fair comparison
        fresh_starting = self._sample_fresh_starting_policies(env)

        child_score, _ = self._eval_cfd_on_env(
            child_cfd, env, num_seeds=self.num_seeds_cheap,
            preset_policies_per_seed=fresh_starting, executor=executor,
        )
        parent_score, _ = self._eval_cfd_on_env(
            parent_cfd, env, num_seeds=self.num_seeds_cheap,
            preset_policies_per_seed=fresh_starting, executor=executor,
        )

        return child_score, parent_score

    def _save_state(self) -> None:
        if self.save_path:
            # Save RNG state and fixed eval params for resumability
            self.state.rng_state = self.rng.bit_generator.state
            self.state.frontier_eval_seed = self._frontier_eval_seed
            # Save starting policies (convert env keys to strings for pickling)
            self.state.frontier_starting_policies = {
                env.value: policies
                for env, policies in self._frontier_starting_policies.items()
            }
            self.state.save(self.save_path / "pareto_state.pkl")
            logger.info(f"State saved to {self.save_path / 'pareto_state.pkl'}")

    def _run_evolution(
        self,
        envs: List[CoordinationGame],
        env_names: List[str],
        executor: ThreadPoolExecutor,
    ) -> None:
        """Run initialization and evolution steps using a shared executor."""
        # Resume from saved state or initialize fresh
        if self.initial_state_path is not None:
            logger.info(f"Resuming from {self.initial_state_path}...")
            self.state = ParetoState.load(self.initial_state_path)
            logger.info(f"  Loaded {len(self.state.candidates)} candidates (step counter: {self.state.step_counter})")

            # Restore RNG state
            if self.state.rng_state is not None:
                self.rng.bit_generator.state = self.state.rng_state
                logger.info("  Restored RNG state")

            # Restore frontier eval seed
            if self.state.frontier_eval_seed is not None:
                self._frontier_eval_seed = self.state.frontier_eval_seed

            # Load cached policies pool
            self._load_starting_policies(envs)

            # Restore pre-sampled frontier starting policies from state
            if self.state.frontier_starting_policies is not None:
                env_by_name = {e.value: e for e in envs}
                for env_name, policies in self.state.frontier_starting_policies.items():
                    if env_name in env_by_name:
                        self._frontier_starting_policies[env_by_name[env_name]] = policies
                logger.info("  Restored frontier starting policies")

            self._save_state()
        else:
            # Load starting policies from cache
            self._load_starting_policies(envs)

            # Phase 1: Initialize
            logger.info("Initializing population...")
            seed_cfds = self._seed_generator.generate(
                n=self.population_size, max_bullet_points=1,
            )

            for cfd in seed_cfds:
                logger.info(f"Seed CFD: {cfd}")

            for i, cfd in enumerate(seed_cfds):
                scores, xp_matrices = self._eval_cfd_all_envs(cfd, envs, executor=executor)
                logger.info(f"  Seed scored: {scores} — {cfd}")
                candidate = ParetoCandidate(
                    cfd=cfd,
                    env_scores=scores,
                    xp_matrices=xp_matrices,
                    step_created_at=self.state.step_counter + i,
                )
                self.state.candidates.append(candidate)
                self.state.step_counter += 1

            self._save_state()

        # Phase 2: Evolution steps
        for step_num in range(1, self.num_steps + 1):
            if self.langfuse_session_id:
                langfuse_client.update_current_trace(session_id=self.langfuse_session_id)

            logger.info(f"=== Pareto Evolution Step {step_num}/{self.num_steps} ===")

            # Compute Pareto frontier
            frontier = compute_pareto_frontier(self.state.candidates, env_names)
            logger.info(f"Pareto frontier size: {len(frontier)}")

            if not frontier:
                logger.warning("Empty Pareto frontier, using all alive candidates")
                frontier = [c for c in self.state.candidates if c.is_alive]

            # Sample uniformly from frontier
            idx = int(self.rng.integers(0, len(frontier)))
            parent = frontier[idx]
            logger.info(f"Selected parent:\n{parent.cfd}")
            logger.info(f"  Parent scores: {parent.env_scores}")

            # Mutate (retry if duplicate)
            from meta_train.population_based.population import Genotype
            existing_cfds = {c.cfd for c in self.state.candidates}

            max_mutation_attempts = 3
            new_cfd = None
            for attempt in range(max_mutation_attempts):
                temp_genotype = Genotype(
                    cfd=parent.cfd,
                    step_created_at=parent.step_created_at,
                )
                operator_name = self.mutation_operator_distribution.sample(self.rng)
                mutation_operator = create_mutation_operator(operator_name, self._optimizer_llm_config)
                candidate_cfd = mutation_operator.mutate(temp_genotype, self.max_bullet_points)

                if candidate_cfd not in existing_cfds:
                    new_cfd = candidate_cfd
                    break
                logger.info(f"  Mutation attempt {attempt+1} produced duplicate CFD, retrying...")

            if new_cfd is None:
                logger.warning(f"  All {max_mutation_attempts} mutation attempts produced duplicates. Skipping step.")
                self._save_state()
                continue

            logger.info(f"Mutation operator: {operator_name}")
            logger.info(f"Child CFD: {new_cfd}")

            # Cheap eval: pick one random env
            eval_env = envs[int(self.rng.integers(0, len(envs)))]
            logger.info(f"Cheap eval on {eval_env.value}...")

            child_score, parent_score = self._cheap_eval(new_cfd, parent.cfd, eval_env, executor=executor)
            logger.info(f"  Child: {child_score:.4f}, Parent: {parent_score:.4f}")

            self.state.history.append({
                "step": step_num,
                "parent_cfd": parent.cfd,
                "child_cfd": new_cfd,
                "operator": operator_name,
                "cheap_eval_env": eval_env.value,
                "child_cheap_score": child_score,
                "parent_cheap_score": parent_score,
                "passed_gate": child_score > parent_score,
            })

            if child_score <= parent_score:
                logger.info("  Child did NOT beat parent. Discarding.")
                self._save_state()
                continue

            # Full eval
            logger.info("  Child beat parent! Running full evaluation...")
            child_scores, child_xp_matrices = self._eval_cfd_all_envs(new_cfd, envs, executor=executor)

            child_candidate = ParetoCandidate(
                cfd=new_cfd,
                env_scores=child_scores,
                xp_matrices=child_xp_matrices,
                parent_cfd=parent.cfd,
                mutation_operator_name=operator_name,
                step_created_at=self.state.step_counter,
            )

            # Check if child makes the Pareto frontier
            test_candidates = [c for c in self.state.candidates if c.is_alive] + [child_candidate]
            new_frontier = compute_pareto_frontier(test_candidates, env_names)

            if child_candidate in new_frontier:
                logger.info(f"  Child ADDED to Pareto frontier! Scores: {child_scores}")
                self.state.candidates.append(child_candidate)
                self.state.step_counter += 1

                # Mark dominated candidates as dead
                for c in self.state.candidates:
                    if c.is_alive and c not in new_frontier:
                        c.is_alive = False
                        logger.info(f"  Removed dominated:\n{c.cfd}")
            else:
                logger.info(f"  Child did NOT make Pareto frontier. Discarding.")

            self.state.history[-1]["full_eval_scores"] = child_scores
            self.state.history[-1]["added_to_frontier"] = child_candidate in new_frontier

            self._save_state()

    @observe(capture_output=False)
    def find_cfd(self, envs: List[CoordinationGame]) -> str:
        """Run the Pareto evolution algorithm and return the best CFD."""
        env_names = [e.value for e in envs]

        logger.info("=" * 60)
        logger.info("Starting Pareto Evolution CFD Optimization")
        logger.info(f"  Population size: {self.population_size}")
        logger.info(f"  Number of steps: {self.num_steps}")
        logger.info(f"  Environments: {env_names}")
        logger.info(f"  Num seeds (cheap eval): {self.num_seeds_cheap}")
        logger.info(f"  Num seeds (full eval): {self.num_seeds_full}")
        logger.info(f"  Optimizer LLM: {self.optimizer_llm}")
        logger.info(f"  Scoring LLM: {self.scoring_llm}")
        logger.info(f"  Initial state path: {self.initial_state_path}")
        logger.info("=" * 60)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            self._run_evolution(envs, env_names, executor)

        # Final selection: best normalized mean across envs
        frontier = compute_pareto_frontier(self.state.candidates, env_names)
        if not frontier:
            frontier = [c for c in self.state.candidates if c.is_alive]
        if not frontier:
            frontier = self.state.candidates

        # Compute per-env min/max across the frontier for normalization
        env_min = {}
        env_max = {}
        for e in env_names:
            scores = [c.env_scores.get(e, 0.0) for c in frontier]
            env_min[e] = min(scores)
            env_max[e] = max(scores)

        def normalized_mean(candidate: ParetoCandidate) -> float:
            normed = []
            for e in env_names:
                lo, hi = env_min[e], env_max[e]
                raw = candidate.env_scores.get(e, 0.0)
                normed.append((raw - lo) / (hi - lo) if hi > lo else 0.5)
            return float(np.mean(normed))

        best = max(frontier, key=normalized_mean)

        logger.info("=" * 60)
        logger.info("Pareto Evolution Complete")
        logger.info(f"Frontier size: {len(frontier)}")
        for c in frontier:
            nm = normalized_mean(c)
            logger.info(f"  [{nm:.3f}] {c.env_scores} —\n{c.cfd}")
        logger.info(f"Best CFD (normalized mean={normalized_mean(best):.3f}): {best.cfd}")
        logger.info(f"Best scores: {best.env_scores}")
        logger.info("=" * 60)

        return best.cfd
