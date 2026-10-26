"""
Binary Tournament Evolutionary Algorithm for CFD optimization.
"""
from pathlib import Path
from typing import List, Optional

from langfuse import observe
from utils.tracing import langfuse_client

from configs.configs import LLMConfig
from configs.contexts import CoordinationGame
from llm.request import LLMType
from meta_train.population_based.mutate import (
    DiverseSeedGenerator,
    SampleDistribution,
    create_mutation_operator,
)
from meta_train.population_based.optimizer import PopulationOptimizer
from meta_train.population_based.population import (
    BinaryTournamentHistoryEntry,
    TournamentHistoryEntry,
    Genotype,
    InitializationHistoryEntry,
    Population,
)
from utils.log_utils import get_logger

logger = get_logger(__name__)


# Named mutation operator distributions
# Format: operator_name -> weight
MUTATION_DISTRIBUTIONS = {
    "default": SampleDistribution({
        "unary_add": 1,
        "unary_remove": 1,
        "unary_reformulate": 1,
        "unary_atomic": 1,
        "diverse_add": 1,
        "diverse_remove": 1,
        "diverse_reformulate": 1,
        "diverse_atomic": 1,
        "reinit": 2,
    }),
    "with_feedback": SampleDistribution({
        "unary_add": 1,
        "unary_remove": 1,
        "unary_reformulate": 1,
        "unary_atomic": 1,
        "diverse_add": 1,
        "diverse_remove": 1,
        "diverse_reformulate": 1,
        "diverse_atomic": 1,
        "feedback_add": 1,
        "feedback_remove": 1,
        "feedback_reformulate": 1,
        "feedback_atomic": 1,
        "diverse_feedback_add": 2,
        "diverse_feedback_remove": 2,
        "diverse_feedback_reformulate": 2,
        "diverse_feedback_atomic": 2,
        "reinit": 4,
    }),
    "more_reformulate": SampleDistribution({
        "unary_add": 1,
        "unary_remove": 1,
        "unary_reformulate": 2,
        "unary_atomic": 1,
        "diverse_add": 1,
        "diverse_remove": 1,
        "diverse_reformulate": 3,
        "diverse_atomic": 1,
        "reinit": 1,
        "partial_reinit": 2,
    }),
}

DEFAULT_MUTATION_OPERATOR_DISTRIBUTION = MUTATION_DISTRIBUTIONS["default"]


class BinaryTournamentOptimizer(PopulationOptimizer):
    """
    Binary Tournament Evolutionary Algorithm for CFD optimization.
    
    In each step:
    1. Sample two alive genotypes
    2. Score both genotypes
    3. Mutate the stronger one using a randomly sampled thinking style
    4. Eliminate the weaker one and add the mutated child to the population
    
    Final selection returns the genotype that has survived the longest.
    """
    
    def __init__(
        self,
        population_size: int,
        num_steps: int,
        num_seeds: int = 5,
        random_state: int = 42,
        optimizer_llm: LLMType = LLMType.OR_GEMINI_PRO_3_PREVIEW,
        scoring_llm: LLMType = LLMType.OR_GROK_41_FAST,
        save_path: Optional[Path] = None,
        max_bullet_points: int = 5,
        use_cached_generations: bool = False,
        mutation_operator_distribution: Optional[SampleDistribution[str]] = None,
        langfuse_session_id: Optional[str] = None,
        scoring_log_path: Optional[Path] = None,
        exclusive_scoring_log: bool = False,
        use_sp_filtering: bool = False,
        initial_population_path: Optional[Path] = None,
        num_policies_per_seed: Optional[int] = None,
        judge_independently: bool = True,
        num_invalid_action_policies: Optional[int] = None,
    ):
        """
        Initialize the Binary Tournament optimizer.

        Args:
            population_size: Number of genotypes in the population.
            num_steps: Number of tournament steps to run.
            num_seeds: Number of seeds per environment during evaluation.
            random_state: Seed for the random number generator.
            optimizer_llm: LLM to use for mutation operations.
            scoring_llm: LLM to use for policy generation during scoring.
            save_path: Optional path to save population state.
            max_bullet_points: Maximum number of bullet points in a CFD.
            use_cached_generations: Whether to use cached policy generations.
            mutation_operator_distribution: Distribution over mutation operator names
                (e.g., "unary_add", "unary_atomic", "feedback_reformulate").
            langfuse_session_id: Optional session ID for Langfuse tracing.
            scoring_log_path: Optional path to write scoring logs to a separate file.
                             If provided, all logs from scoring (including nested calls)
                             will be written to this file.
            exclusive_scoring_log: If True, scoring logs will ONLY go to the scoring log file
                                   (not to the main console/log). Default is False.
            use_sp_filtering: Enable self-play filtering during training/eval.
            initial_population_path: Optional path to a pickle file containing an initial
                                     population to load instead of generating new seeds.
                                     This can be different from save_path.
            num_policies_per_seed: Number of policies to generate per seed.
                                  If None, uses the default from the trainer config.
            judge_independently: If True, the critic judges each local policy independently.
                                If False, the critic judges the joint policy as a whole.
        """
        super().__init__(
            population_size=population_size,
            random_state=random_state,
            optimizer_llm=optimizer_llm,
            scoring_llm=scoring_llm,
            save_path=save_path,
            use_sp_filtering=use_sp_filtering,
            num_policies_per_seed=num_policies_per_seed,
            judge_independently=judge_independently,
            num_invalid_action_policies=num_invalid_action_policies,
        )
        self.num_steps = num_steps
        self.num_seeds = num_seeds
        self.max_bullet_points = max_bullet_points
        self.use_cached_generations = use_cached_generations
        self.mutation_operator_distribution = mutation_operator_distribution or DEFAULT_MUTATION_OPERATOR_DISTRIBUTION
        self.langfuse_session_id = langfuse_session_id
        self.scoring_log_path = scoring_log_path
        self.exclusive_scoring_log = exclusive_scoring_log
        self.initial_population_path = Path(initial_population_path) if initial_population_path else None

        # Create LLM config for optimizer
        self._optimizer_llm_config = LLMConfig(
            model=optimizer_llm,
            temperature=1.0,
            max_tokens=4096,
        )
        
        # Create seed generator
        self._seed_generator = DiverseSeedGenerator(self._optimizer_llm_config)
    
    @observe(capture_output=False)
    def _initialize(self) -> None:
        """
        Initialize the population with diverse seed CFDs or load from a pickle file.
        
        If initial_population_path is provided, loads the population from that file.
        Otherwise, uses DiverseSeedGenerator to create population_size diverse CFDs,
        wraps them in Genotype objects, and logs to population history.
        """
        if self.langfuse_session_id:
            langfuse_client.update_current_trace(session_id=self.langfuse_session_id)
        
        # Load from pickle file if provided
        if self.initial_population_path is not None:
            logger.info(f"Loading initial population from {self.initial_population_path}...")
            self.population = Population.load(self.initial_population_path)
            alive_count = len(self.population.get_alive())
            logger.info(
                f"Loaded population with {alive_count} alive genotypes "
                f"(step counter: {self.population.current_step})"
            )
            self.save_population()
            return
        
        logger.info(f"Initializing population with {self.population_size} diverse CFDs...")
        
        # Generate diverse seed CFDs
        seed_cfds = self._seed_generator.generate(
            n=self.population_size,
            max_bullet_points=1,  # Seeds start with 1 bullet point
        )
        
        # Create genotypes from seeds
        initial_genotypes = []
        current_step = self.population.current_step
        for cfd in seed_cfds:
            genotype = Genotype(
                cfd=cfd,
                step_created_at=current_step,
                mutation_operator_name="seed",
            )
            self.population.add_genotype(genotype)
            initial_genotypes.append(genotype)
            logger.info(f"Added seed genotype: {cfd}")
        
        # Log to history
        self.population.history.append(InitializationHistoryEntry(
            initial_genotypes=initial_genotypes,
            step=current_step,
        ))
        
        logger.info(f"Population initialized with {len(initial_genotypes)} genotypes")
        self.save_population()
    
    @observe(capture_output=False)
    def _step(self, envs: List[CoordinationGame], step_num: int) -> None:
        """
        Execute one tournament step.

        1. Sample three alive genotypes
        2. Score all three
        3. Eliminate the worst one
        4. Mutate a randomly chosen member of the top two to create a child
        5. Add the child to the population

        Args:
            envs: List of environments to evaluate on.
            step_num: Current step number for logging.
        """
        langfuse_client.update_current_span(name=f"step_{step_num}")

        if self.langfuse_session_id:
            langfuse_client.update_current_trace(session_id=self.langfuse_session_id)

        logger.info(f"=== Tournament Step {step_num}/{self.num_steps} ===")

        # Sample three alive genotypes
        alive = self.population.get_alive()
        tournament_size = min(3, len(alive))
        indices = self.rng.choice(len(alive), size=tournament_size, replace=False)
        contestants = [alive[i] for i in indices]

        for i, g in enumerate(contestants):
            logger.info(f"  Genotype {i+1}:\n{g.cfd}")

        # Draw a single seed for this scoring round so all contestants share RNG alignment
        scoring_seed = int(self.rng.integers(0, 2**32 - 1))

        # Score all contestants
        self.score_genotypes(
            genotypes=contestants,
            envs=envs,
            num_seeds=self.num_seeds,
            seed=scoring_seed,
            use_cached_generations=self.use_cached_generations,
            langfuse_session_id=self.langfuse_session_id,
            scoring_log_path=self.scoring_log_path,
            exclusive_scoring_log=self.exclusive_scoring_log,
        )

        # Get scores and rank (highest = best)
        scores = [g.score_history[-1].score for g in contestants]
        ranked = sorted(zip(contestants, scores), key=lambda x: x[1], reverse=True)

        for rank, (g, s) in enumerate(ranked):
            label = "BEST" if rank == 0 else ("2ND" if rank == 1 else "WORST")
            logger.info(f"  [{label}] score={s:.4f}:\n{g.cfd}")
            self._log_trace_and_xp(g)

        # The worst genotype is eliminated
        loser = ranked[-1][0]

        # Pick a parent randomly from the top two
        top_two = [g for g, _ in ranked[:2]]
        parent = top_two[int(self.rng.integers(0, len(top_two)))]
        logger.info(f"Parent selected for mutation:\n{parent.cfd}")

        # Sample mutation operator and mutate parent
        operator_name = self.mutation_operator_distribution.sample(self.rng)
        mutation_operator = create_mutation_operator(operator_name, self._optimizer_llm_config)

        logger.info(f"Mutating parent with operator: {operator_name}")
        new_cfd = mutation_operator.mutate(parent, self.max_bullet_points)

        # Create new genotype
        new_genotype = Genotype(
            cfd=new_cfd,
            step_created_at=self.population.current_step,
            parents=[parent],
            mutation_operator_name=mutation_operator.name,
        )

        logger.info(f"Created child genotype:\n{new_cfd}")

        # Eliminate loser
        self.population.remove_genotype(loser)
        logger.info(f"Eliminated worst genotype")

        # Add new genotype
        self.population.add_genotype(new_genotype)

        # Log to history
        self.population.history.append(TournamentHistoryEntry(
            evaluated_genotypes=contestants,
            scores=scores,
            parent_genotype=parent,
            new_genotype=new_genotype,
            eliminated_genotype=loser,
            mutation_operator_name=mutation_operator.name,
            step=self.population.current_step,
        ))

        # Save population
        self.save_population()
    
    @observe(capture_output=False)
    def _final_selection(self) -> Genotype:
        """
        Select the final genotype from the population.

        Chooses the alive genotype with the longest score history. If multiple
        genotypes share the same longest history length, pick the oldest
        (minimum step_created_at).
        
        Returns:
            The selected genotype.
        """
        if self.langfuse_session_id:
            langfuse_client.update_current_trace(session_id=self.langfuse_session_id)
        
        logger.info("=== Final Selection ===")
        
        alive = self.population.get_alive()
        if not alive:
            raise ValueError("No alive genotypes for final selection")
        
        longest_history = max(len(g.score_history) for g in alive)
        candidates = [g for g in alive if len(g.score_history) == longest_history]
        # TODO: Create a better tie-breaking mechanism (e.g. who has the latest scores/better mean score)
        # Currently, we pick the yougest genotype among those with the longest score history
        selected = max(candidates, key=lambda g: g.step_created_at)
        
        logger.info(f"Selected genotype (created at step {selected.step_created_at}):")
        logger.info(f"  CFD: {selected.cfd}")
        if selected.score_history:
            logger.info(f"  Latest score: {selected.score_history[-1].score:.4f}")
        
        self.save_population()
        
        return selected
    
    def find_cfd(self, envs: List[CoordinationGame]) -> str:
        """
        Run the binary tournament algorithm and return the best CFD.
        
        Orchestrates:
        1. Initialization with diverse seeds
        2. num_steps tournament steps
        3. Final selection
        
        Args:
            envs: List of environments to evaluate CFDs on.
        
        Returns:
            The best CFD string found.
        """

        logger.info("=" * 60)
        logger.info("Starting Binary Tournament CFD Optimization")
        logger.info(f"  Population size: {self.population_size}")
        logger.info(f"  Number of steps: {self.num_steps}")
        logger.info(f"  Environments: {[e.value for e in envs]}")
        logger.info(f"  Optimizer LLM: {self.optimizer_llm}")
        logger.info(f"  Scoring LLM: {self.scoring_llm}")
        logger.info(f"  SP filtering enabled: {self.use_sp_filtering}")
        logger.info(f"  Initial population path: {self.initial_population_path}")
        logger.info("=" * 60)
        
        # Phase 1: Initialization
        self._initialize()
        
        # Phase 2: Tournament steps
        for step_num in range(1, self.num_steps + 1):
            self._step(envs, step_num)
        
        # Phase 3: Final selection
        selected = self._final_selection()
        
        logger.info("=" * 60)
        logger.info("Binary Tournament Optimization Complete")
        logger.info(f"Best CFD: {selected.cfd}")
        logger.info("=" * 60)
        
        return selected.cfd

