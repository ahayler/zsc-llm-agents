from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, TypeVar, Generic, Optional, List, Tuple

import numpy as np
from langfuse import observe

from llm.parser import request_and_parse_to_json
from llm.request import create_client
from configs.configs import LLMConfig
from core import Message, CodePolicy, CompactRolloutHistory
from train.utils import code_policy_to_str
from meta_train.population_based.population import Genotype
from utils.saving import CodeXPExperimentResults


T = TypeVar('T')


class ThinkingStyle(Enum):
    """Mutation thinking styles for CFD modification."""
    ADD = "add"  # Add a new bullet point
    REMOVE = "remove"  # Remove a bullet point (simplification)
    REFORMULATE = "reformulate"  # Rewrite/modify an existing bullet point
    ATOMIC = "atomic"  # Make exactly one atomic change (add, remove, or change one bullet point)


class SampleDistribution(Generic[T]):
    """A weighted sampling utility for selecting values based on weights."""
    
    def __init__(self, weights: Dict[T, float]):
        """
        Initialize with a dictionary mapping values to weights.
        
        Args:
            weights: Dict mapping values to their sampling weights (will be normalized).
        """
        if not weights:
            raise ValueError("weights dictionary cannot be empty")
        self._values = list(weights.keys())
        self._weights = np.array(list(weights.values()), dtype=float)
        if np.any(self._weights < 0):
            raise ValueError("weights must be non-negative")
        weight_sum = self._weights.sum()
        if weight_sum == 0:
            raise ValueError("sum of weights must be positive")
        self._weights = self._weights / weight_sum  # normalize
    
    def sample(self, rng: Optional[np.random.Generator] = None) -> T:
        """
        Sample a value according to the weight distribution.
        
        Args:
            rng: Optional numpy random generator for reproducibility.
        
        Returns:
            A sampled value from the distribution.
        """
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(len(self._values), p=self._weights)
        return self._values[idx]


class MutationOperator(ABC):
    """Abstract base class for mutation operators."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this mutation operator."""
        pass
    
    @abstractmethod
    def mutate(self, genotype: Genotype, max_bullet_points: int) -> str:
        """
        Mutate a CFD and return the new CFD.
        
        Args:
            genotype: The genotype containing the CFD and score history.
            max_bullet_points: Maximum number of bullet points allowed.
        
        Returns:
            The mutated CFD string.
        """
        pass


# Shared background context for all mutation prompts
_SHARED_BACKGROUND = """
## Background

In zero-shot coordination (ZSC), independently trained policies must coordinate successfully without any shared training or communication.

**Game Structure:**
- Fully cooperative two-player games
- Each player's behavior is defined by a Python function (local policy)
- Two local policies together form a joint policy

**Training Process:**
- N independent training runs (seeds) are conducted
- In each seed, an LLM generates a policy candidate
- A **policy critic** then reviews each candidate against the CFD. The critic judges each local policy (player_0, player_1) **independently** — as if it exists in isolation, without knowledge of the other player's policy. A joint policy is only approved if both local policies individually satisfy the CFD. The critic also tries to maximize the **self-play (SP) reward** — but it has no knowledge of cross-play or other seeds.
- The CFD is the **only mechanism** through which you can influence compatibility across seeds. The critic will enforce whatever the CFD says on each local policy independently, so the CFD must be specific and unambiguous enough that independent critics produce compatible local policies.
- There is no shared training or communication between the seeds/policies.

**Evaluation - Cross-Play (XP) Score:**
- After N independent runs, we have N final policies
- Each policy is paired with every other policy (N×N matchups)
- XP score = average joint reward across all pairwise combinations

**Goal:** Evolve CFDs that maximize XP scores by encouraging policies to adopt compatible, coordination-friendly behaviors.
Priortize avoiding total coordination failures, then think about raising the XP scores in general.
"""


# Template for unary mutation prompts
_UNARY_MUTATION_TEMPLATE = """You are helping to evolve a "coordination-friendly definition" (CFD) for zero-shot coordination (ZSC).

{background}

## Current CFD

{cfd}

## Your Task

{task_instructions}

## Constraints

- The CFD must have at most {max_bullet_points} bullet points
- Each bullet point should be a single, concise sentence
{extra_constraints}

## Response Format

Respond with valid JSON:
```json
{{
    "reasoning": "{reasoning_prompt}",
    "new_cfd": "- First bullet point\\n- Second bullet point\\n..."
}}
```"""

# Thinking style specific instructions
_THINKING_STYLE_INSTRUCTIONS = {
    ThinkingStyle.ADD: {
        "task_instructions": """**ADD** a new bullet point to the CFD. The new bullet point should:
- Address an aspect of coordination not yet covered by existing points
- Be complementary to (not redundant with) existing points
- Help policies coordinate better across different game scenarios""",
        "extra_constraints": "",
        "reasoning_prompt": "Explain why you're adding this bullet point and what coordination aspect it addresses",
    },
    
    ThinkingStyle.REMOVE: {
        "task_instructions": """**REMOVE** one bullet point from the CFD. Choose to remove a bullet point that:
- Is redundant or overlaps significantly with other points
- Is too restrictive and may hurt coordination flexibility
- Is less essential than other points for achieving high XP scores""",
        "extra_constraints": "- The resulting CFD must have at least 1 bullet point. If the CFD has only 1 bullet point, reformulate it instead of removing it.",
        "reasoning_prompt": "Explain which bullet point you removed and why removing it improves the CFD",
    },
    
    ThinkingStyle.REFORMULATE: {
        "task_instructions": """**REFORMULATE or REPLACE** one bullet point in the CFD. You may either:
- Reformulate: refine the wording to be clearer, more precise, or more actionable
- Replace: swap the bullet point for an entirely different principle that better encourages coordination
- Adapt: Change some part of the bullet point while keepint the rest the same

Choose whichever approach you believe will most improve the CFD's effectiveness.""",
        "extra_constraints": "",
        "reasoning_prompt": "Explain what you changed and why it improves the CFD",
    },
    
    ThinkingStyle.ATOMIC: {
        "task_instructions": """Make exactly **ONE atomic change** to the CFD. Choose ONE of the following operations:

1. **ADD**: Add exactly one new bullet point that addresses a missing coordination aspect
2. **REMOVE**: Remove exactly one bullet point that is redundant, too restrictive, or less essential
3. **CHANGE**: Modify exactly one existing bullet point (rewording, refining, or replacing it)

Pick the operation that you believe will most improve the CFD's effectiveness for zero-shot coordination.""",
        "extra_constraints": "- You must make exactly ONE change (add one, remove one, or change one bullet point)\n- The resulting CFD must have at least 1 bullet point",
        "reasoning_prompt": "Explain which atomic operation you chose (add/remove/change), what specific change you made, and why",
    },
}


def _build_mutation_prompt(thinking_style: ThinkingStyle, cfd: str, max_bullet_points: int) -> str:
    """Build the full mutation prompt for a given thinking style."""
    style_config = _THINKING_STYLE_INSTRUCTIONS[thinking_style]
    return _UNARY_MUTATION_TEMPLATE.format(
        background=_SHARED_BACKGROUND,
        cfd=cfd,
        max_bullet_points=max_bullet_points,
        task_instructions=style_config["task_instructions"],
        extra_constraints=style_config["extra_constraints"],
        reasoning_prompt=style_config["reasoning_prompt"],
    )


class UnaryMutationOperator(MutationOperator):
    """
    A unary mutation operator that uses an LLM to mutate a CFD.
    
    Takes a thinking style at construction to determine how the CFD is mutated.
    """
    
    def __init__(self, thinking_style: ThinkingStyle, llm_config: LLMConfig):
        """
        Initialize the mutation operator.
        
        Args:
            thinking_style: The style of mutation to apply (ADD, REMOVE, REFORMULATE).
            llm_config: Configuration for the LLM to use.
        """
        self._thinking_style = thinking_style
        self._llm_config = llm_config
        self._client, self._model = create_client(llm_config.model)
    
    @property
    def name(self) -> str:
        return f"unary_{self._thinking_style.value}"
    
    @observe()
    def mutate(self, genotype: Genotype, max_bullet_points: int) -> str:
        """
        Mutate the CFD using the configured thinking style.
        
        Args:
            genotype: The genotype containing the CFD and score history.
            max_bullet_points: Maximum number of bullet points allowed.
        
        Returns:
            The mutated CFD string.
        """
        cfd = genotype.cfd
        prompt = _build_mutation_prompt(self._thinking_style, cfd, max_bullet_points)
        
        message = Message(role="user", content=prompt)
        
        json_response, _ = request_and_parse_to_json(
            client=self._client,
            model=self._model,
            msg_history=[message],
            temperature=self._llm_config.temperature,
            max_tokens=self._llm_config.max_tokens,
            reasoning_effort=self._llm_config.reasoning_effort,
            required_fields=["new_cfd"],
        )
        
        new_cfd = json_response["new_cfd"]
        
        # Validate bullet point count
        bullet_points = [line.strip() for line in new_cfd.split('\n') if line.strip().startswith('-')]
        if len(bullet_points) > max_bullet_points:
            # Truncate to max_bullet_points if necessary
            new_cfd = '\n'.join(bullet_points[:max_bullet_points])
        
        return new_cfd


# Template for diverse seed generation prompt
_DIVERSE_SEED_TEMPLATE = """You are helping to create diverse "coordination-friendly definitions" (CFDs) for zero-shot coordination (ZSC).

{background}

## Your Task

Generate {n} DIVERSE seed CFDs. Each CFD should:
- Be a single bullet point (one concise sentence)
- Encourage coordination in a unique way
- Be simple and general enough to apply to various multi-agent games
- NOT overlap significantly with other CFDs you generate

## Constraints

- Generate exactly {n} CFDs
- Each CFD is a single bullet point (one sentence)
- Maximize diversity across the CFDs

## Response Format

Respond with valid JSON:
```json
{{
    "reasoning": "Explain your strategy for creating diverse CFDs",
    "cfds": [
        "- First CFD",
        "- Second CFD",
        ...
    ]
}}
```"""


class DiverseSeedGenerator:
    """
    Generates diverse single-bullet-point CFDs to seed the population.
    
    This is a nullary operator that creates new CFDs from scratch.
    """
    
    def __init__(self, llm_config: LLMConfig):
        """
        Initialize the seed generator.
        
        Args:
            llm_config: Configuration for the LLM to use.
        """
        self._llm_config = llm_config
        self._client, self._model = create_client(llm_config.model)
    
    @property
    def name(self) -> str:
        return "DiverseSeedGenerator"
    
    def generate(self, n: int, max_bullet_points: int) -> List[str]:
        """
        Generate n diverse seed CFDs.
        
        Args:
            n: Number of diverse CFDs to generate.
            max_bullet_points: Maximum bullet points per CFD (typically 1 for seeds).
        
        Returns:
            List of CFD strings.
        """
        prompt = _DIVERSE_SEED_TEMPLATE.format(background=_SHARED_BACKGROUND, n=n)
        
        message = Message(role="user", content=prompt)
        
        json_response, _ = request_and_parse_to_json(
            client=self._client,
            model=self._model,
            msg_history=[message],
            temperature=self._llm_config.temperature,
            max_tokens=self._llm_config.max_tokens,
            reasoning_effort=self._llm_config.reasoning_effort,
            required_fields=["cfds"],
        )
        
        cfds = json_response["cfds"]
        
        # Validate and clean up CFDs
        result = []
        for cfd in cfds:
            if isinstance(cfd, str):
                # Ensure it starts with a bullet point
                cfd = cfd.strip()
                if not cfd.startswith('-'):
                    cfd = f"- {cfd}"
                
                # Check bullet point count
                bullet_points = [line.strip() for line in cfd.split('\n') if line.strip().startswith('-')]
                if len(bullet_points) <= max_bullet_points:
                    result.append(cfd)
                else:
                    # Truncate if necessary
                    result.append('\n'.join(bullet_points[:max_bullet_points]))
        
        return result[:n]  # Ensure we don't return more than requested


# ============================================================================
# Diversity-Forward Mutation Operator
# ============================================================================

_DIVERSE_MUTATION_TEMPLATE = """You are helping to evolve a "coordination-friendly definition" (CFD) for zero-shot coordination (ZSC).

{background}

## Current CFD

{cfd}

## Your Task

{task_instructions}

**IMPORTANT:** Provide {n_proposals} DIVERSE proposals for the modified CFD. Each proposal should take a meaningfully different approach to the mutation. Maximize diversity across your proposals.

Note: For some operations (e.g., removing a bullet point from a short CFD), fewer than {n_proposals} meaningfully different options may exist. In that case, return as many distinct proposals as possible — do not pad with near-duplicates.

## Constraints

- Each CFD must have at most {max_bullet_points} bullet points
- Each bullet point should be a single, concise sentence
{extra_constraints}

## Response Format

Respond with valid JSON:
```json
{{
    "reasoning": "{reasoning_prompt}",
    "proposals": [
        "- First bullet point\\n- Second bullet point\\n...",
        "- First bullet point\\n- Second bullet point\\n...",
        ...
    ]
}}
```"""


def _build_diverse_mutation_prompt(
    thinking_style: ThinkingStyle, cfd: str, max_bullet_points: int, n_proposals: int,
) -> str:
    """Build the diversity-forward mutation prompt for a given thinking style."""
    style_config = _THINKING_STYLE_INSTRUCTIONS[thinking_style]
    return _DIVERSE_MUTATION_TEMPLATE.format(
        background=_SHARED_BACKGROUND,
        cfd=cfd,
        max_bullet_points=max_bullet_points,
        n_proposals=n_proposals,
        task_instructions=style_config["task_instructions"],
        extra_constraints=style_config["extra_constraints"],
        reasoning_prompt=style_config["reasoning_prompt"],
    )


class DiverseForwardMutationOperator(MutationOperator):
    """
    A diversity-forward mutation operator that asks the LLM for multiple diverse
    proposals and randomly selects one.
    """

    def __init__(self, thinking_style: ThinkingStyle, llm_config: LLMConfig, n_proposals: int = 3):
        self._thinking_style = thinking_style
        self._llm_config = llm_config
        self._client, self._model = create_client(llm_config.model)
        self._n_proposals = n_proposals
        self._rng = np.random.default_rng()

    @property
    def name(self) -> str:
        return f"diverse_{self._thinking_style.value}"

    @observe()
    def mutate(self, genotype: Genotype, max_bullet_points: int) -> str:
        cfd = genotype.cfd
        prompt = _build_diverse_mutation_prompt(
            self._thinking_style, cfd, max_bullet_points, self._n_proposals,
        )

        message = Message(role="user", content=prompt)

        json_response, _ = request_and_parse_to_json(
            client=self._client,
            model=self._model,
            msg_history=[message],
            temperature=self._llm_config.temperature,
            max_tokens=self._llm_config.max_tokens,
            reasoning_effort=self._llm_config.reasoning_effort,
            required_fields=["proposals"],
        )

        proposals = json_response["proposals"]
        if not proposals:
            # Fallback: return original CFD unchanged
            return cfd

        # Validate and clean each proposal
        valid_proposals = []
        for proposal in proposals:
            if not isinstance(proposal, str):
                continue
            bullet_points = [line.strip() for line in proposal.split('\n') if line.strip().startswith('-')]
            if len(bullet_points) > max_bullet_points:
                proposal = '\n'.join(bullet_points[:max_bullet_points])
            valid_proposals.append(proposal)

        if not valid_proposals:
            return cfd

        # Randomly pick one
        idx = int(self._rng.integers(0, len(valid_proposals)))
        return valid_proposals[idx]


# ============================================================================
# Reinitialization Mutation Operator
# ============================================================================


class ReinitializationMutationOperator(MutationOperator):
    """
    A mutation operator that generates a fresh CFD from scratch, ignoring the
    parent genotype entirely. Uses DiverseSeedGenerator to produce multiple
    candidates and randomly picks one.
    """

    def __init__(self, llm_config: LLMConfig, n_candidates: int = 5):
        self._llm_config = llm_config
        self._seed_generator = DiverseSeedGenerator(llm_config)
        self._n_candidates = n_candidates
        self._rng = np.random.default_rng()

    @property
    def name(self) -> str:
        return "reinit"

    @observe()
    def mutate(self, genotype: Genotype, max_bullet_points: int) -> str:
        cfds = self._seed_generator.generate(n=self._n_candidates, max_bullet_points=max_bullet_points)
        if not cfds:
            return genotype.cfd
        idx = int(self._rng.integers(0, len(cfds)))
        return cfds[idx]


class PartialReinitMutationOperator(MutationOperator):
    """
    A mutation operator that replaces one random bullet point in the parent CFD
    with a freshly generated single-bullet CFD from DiverseSeedGenerator.
    If the parent has only one bullet point, this behaves like a full reinit.
    """

    def __init__(self, llm_config: LLMConfig, n_candidates: int = 5):
        self._llm_config = llm_config
        self._seed_generator = DiverseSeedGenerator(llm_config)
        self._n_candidates = n_candidates
        self._rng = np.random.default_rng()

    @property
    def name(self) -> str:
        return "partial_reinit"

    @observe()
    def mutate(self, genotype: Genotype, max_bullet_points: int) -> str:
        # Generate fresh single-bullet CFDs
        fresh_cfds = self._seed_generator.generate(n=self._n_candidates, max_bullet_points=1)
        if not fresh_cfds:
            return genotype.cfd

        # Pick one fresh bullet
        idx = int(self._rng.integers(0, len(fresh_cfds)))
        new_bullet = fresh_cfds[idx].strip()

        # Parse parent bullet points
        parent_bullets = [line.strip() for line in genotype.cfd.split('\n') if line.strip().startswith('-')]
        if len(parent_bullets) <= 1:
            # Single bullet or empty: just replace entirely
            return new_bullet

        # Replace a random bullet point
        replace_idx = int(self._rng.integers(0, len(parent_bullets)))
        parent_bullets[replace_idx] = new_bullet
        return '\n'.join(parent_bullets)


# ============================================================================
# Feedback-Based Mutation Operator
# ============================================================================

# TODO: Maybe add something to the prompt along the lines of "It is also valid to encourage completelty different policies."
# Analyzer prompt template - analyzes coordination failures to provide feedback
_ANALYZER_PROMPT_TEMPLATE = """You are analyzing a coordination failure in a zero-shot coordination (ZSC) experiment.

{background}

## Game Description

The following game was played:

{game_description}

## Current CFD (Coordination-Friendly Definition)

The following CFD was used to guide policy generation:

{cfd}

## Coordination Failure Context

Two policies were independently generated using the above CFD. 
When paired together in cross-play evaluation, they achieved a COMPARATIVELY low score.

**Policy from Seed {seed_0}:**
```python
{policy_0}
```

**Critic Reasoning for Accepting Policy from Seed {seed_0}:**
{acceptance_reasoning_0}

**Policy from Seed {seed_1}:**
```python
{policy_1}
```

**Critic Reasoning for Accepting Policy from Seed {seed_1}:**
{acceptance_reasoning_1}

**Cross-Play Pairing (Player 0 from Seed {seed_0}, Player 1 from Seed {seed_1}):**
```python
{xp_policy}
```

**Average XP Score:** {average_score}

**Sample of individual game scores:** {rewards_sample_list}

**Example of a low-scoring game rollout:**
{game_rollout}

## Your Task

- Provide GENERAL text-based feedback regarding why these two policies failed to coordinate effectively.
- DO NOT disclose the actual game dynamics or the policies and focus on general patterns.
- Keep your feedback general/abstract enough that the underlying policies or game could not be reconstructed.
- If beneficial, your analysis can connect to the current CFD.
- Only provide feedback on what went wrong and why. DO NOT provide direct suggestions on how to change the CFD.

Your analysis will be used to improve the CFD.

**Note:** It is possible that the pairing of policies did not fail to coordinate.
For example, it could also be that just one of the policies is particularly bad (i.e. even when paired with itself we would get low scores)
or that all the XP-scores are pretty good and therefore even the COMPARATIVELY lowest score is still good.
It is valid feedback to say that the pairing of policies did not fail to coordinate and you do not recommend any changes to the CFD,
or that you recommend changes to the CFD of a different kind (e.g. to filter out bad policies).

## Response Format

Respond with valid JSON (it can include additional text before or after the JSON object):
```json
{{
    "analysis": "Your detailed text-based analysis."
}}
```"""


# Feedback-based mutation prompt template
_FEEDBACK_MUTATION_TEMPLATE = """You are helping to evolve a "coordination-friendly definition" (CFD) for zero-shot coordination (ZSC).

{background}

## Current CFD

{cfd}

## Feedback from Coordination Analysis

The following feedback was generated by analyzing a policy pair that might have performed poorly in cross-play:

{analysis}

## Your Task

{task_instructions}

## Constraints

- The CFD must have at most {max_bullet_points} bullet points
- Each bullet point should be a single, concise sentence
{extra_constraints}

## Response Format

Respond with valid JSON:
```json
{{
    "reasoning": "{reasoning_prompt}",
    "new_cfd": "- First bullet point\\n- Second bullet point\\n..."
}}
```"""


def _build_feedback_mutation_prompt(
    thinking_style: ThinkingStyle,
    cfd: str,
    max_bullet_points: int,
    analysis: str,
) -> str:
    """Build the feedback-based mutation prompt for a given thinking style."""
    style_config = _THINKING_STYLE_INSTRUCTIONS[thinking_style]
    
    # Modify task instructions to incorporate feedback
    base_instructions = style_config["task_instructions"]
    feedback_instructions = f"""Based on the coordination analysis feedback above, {base_instructions.lower()}

Consider the analysis when making your changes."""
    
    return _FEEDBACK_MUTATION_TEMPLATE.format(
        background=_SHARED_BACKGROUND,
        cfd=cfd,
        analysis=analysis,
        max_bullet_points=max_bullet_points,
        task_instructions=feedback_instructions,
        extra_constraints=style_config["extra_constraints"],
        reasoning_prompt=style_config["reasoning_prompt"],
    )


def _get_min_xp_cell(xp_matrix: np.ndarray) -> Tuple[int, int]:
    """
    Find the indices of the minimum cell in the XP matrix.
    
    Args:
        xp_matrix: The cross-play score matrix (N x N).
    
    Returns:
        Tuple of (row_idx, col_idx) for the minimum cell.
    """
    min_idx = np.unravel_index(np.argmin(xp_matrix), xp_matrix.shape)
    return (int(min_idx[0]), int(min_idx[1]))


def _extract_coordination_failure_context(
    experiment_results: CodeXPExperimentResults,
    index_pair: Tuple[int, int],
) -> Tuple[CodePolicy, CodePolicy, CodePolicy, float, List[float], CompactRolloutHistory]:
    """
    Extract context for a coordination failure from experiment results.
    
    Args:
        experiment_results: The XP experiment results.
        index_pair: Tuple of (seed_0, seed_1) indices for the failing pair.
    
    Returns:
        Tuple of (policy_0, policy_1, xp_policy, average_score, rewards_sample_list, game_rollout)
    """
    xp_matrix = experiment_results.xp_matrix
    policies = experiment_results.selected_policies
    compact_rollout_histories_matrix = experiment_results.compact_rollout_histories
    
    seed_0, seed_1 = index_pair
    policy_0 = policies[seed_0]
    policy_1 = policies[seed_1]
    
    # Get rollout histories for this pair
    compact_rollout_histories = compact_rollout_histories_matrix[seed_0, seed_1]
    
    # Find the worst rollout
    worst_rollout = sorted(compact_rollout_histories, key=lambda x: x.rewards_sum)[0]
    
    # Sample rewards
    rewards = [h.rewards_sum for h in compact_rollout_histories]
    sample_size = min(20, len(rewards))
    rewards_sample_list = list(np.random.choice(rewards, size=sample_size, replace=False))
    
    # Create the XP policy (player 0 from seed_0, player 1 from seed_1)
    xp_policy = CodePolicy(
        name=f"{policy_0.name} (0) vs. {policy_1.name} (1)",
        joint_policy=[policy_0.joint_policy[0], policy_1.joint_policy[1]],
        reason=""
    )
    
    average_score = float(xp_matrix[seed_0, seed_1])
    
    return policy_0, policy_1, xp_policy, average_score, rewards_sample_list, worst_rollout


class FeedbackBasedMutationOperator(MutationOperator):
    """
    A mutation operator that uses environment feedback from the last XP experiment.
    
    This operator:
    1. Extracts the worst-performing policy pair from the genotype's score history
    2. Analyzes the coordination failure using an "Analyzer" LLM
    3. Uses the analysis feedback to guide CFD mutation
    """
    
    def __init__(self, thinking_style: ThinkingStyle, llm_config: LLMConfig):
        """
        Initialize the feedback-based mutation operator.
        
        Args:
            thinking_style: The style of mutation to apply (ADD, REMOVE, REFORMULATE).
            llm_config: Configuration for the LLM to use.
        """
        self._thinking_style = thinking_style
        self._llm_config = llm_config
        self._client, self._model = create_client(llm_config.model)
    
    @property
    def name(self) -> str:
        return f"feedback_{self._thinking_style.value}"
    
    @observe()
    def _analyze_coordination_failure(
        self,
        cfd: str,
        policy_0: CodePolicy,
        policy_1: CodePolicy,
        xp_policy: CodePolicy,
        average_score: float,
        rewards_sample_list: List[float],
        game_rollout: CompactRolloutHistory,
        seed_0: int,
        seed_1: int,
        game_description: str,
    ) -> str:
        """
        Analyze a coordination failure using the Analyzer LLM.
        
        Returns:
            The analysis text.
        """
        prompt = _ANALYZER_PROMPT_TEMPLATE.format(
            background=_SHARED_BACKGROUND,
            game_description=game_description,
            cfd=cfd,
            seed_0=seed_0,
            seed_1=seed_1,
            policy_0=code_policy_to_str(policy_0),
            acceptance_reasoning_0=policy_0.reason,
            policy_1=code_policy_to_str(policy_1),
            acceptance_reasoning_1=policy_1.reason,
            xp_policy=code_policy_to_str(xp_policy),
            average_score=average_score,
            rewards_sample_list=str([round(r, 2) for r in rewards_sample_list]),
            game_rollout=str(game_rollout),
        )
        
        message = Message(role="user", content=prompt)
        
        json_response, _ = request_and_parse_to_json(
            client=self._client,
            model=self._model,
            msg_history=[message],
            temperature=self._llm_config.temperature,
            max_tokens=self._llm_config.max_tokens,
            reasoning_effort=self._llm_config.reasoning_effort,
            required_fields=["analysis"],
        )
        
        return json_response["analysis"]
    
    @observe()
    def mutate(self, genotype: Genotype, max_bullet_points: int) -> str:
        """
        Mutate the CFD using feedback from the last XP experiment.
        
        Args:
            genotype: The genotype containing the CFD and score history.
            max_bullet_points: Maximum number of bullet points allowed.
        
        Returns:
            The mutated CFD string.
        
        Raises:
            NotImplementedError: If the score history contains multiple experiments per entry.
            ValueError: If the genotype has no score history.
        """
        # Import here to avoid circular imports
        from meta_train.population_based.population import ScoreEntry
        
        cfd = genotype.cfd
        
        # Check that we have score history
        if not genotype.score_history:
            raise ValueError("Genotype has no score history. Cannot use feedback-based mutation.")
        
        # Get the last score entry
        last_score_entry: ScoreEntry = genotype.score_history[-1]
        experiment_results = last_score_entry.experiment_results
        
        # Find the minimum cell in the XP matrix
        xp_matrix = experiment_results.xp_matrix
        min_cell = _get_min_xp_cell(xp_matrix)
        seed_0, seed_1 = min_cell
        
        # Extract coordination failure context
        policy_0, policy_1, xp_policy, average_score, rewards_sample_list, game_rollout = \
            _extract_coordination_failure_context(experiment_results, min_cell)
        
        # Get game description (fallback to env name if not available)
        game_description = experiment_results.game_description or f"Game: {experiment_results.env_name_file_safe}"
        
        # Analyze the coordination failure
        analysis = self._analyze_coordination_failure(
            cfd=cfd,
            policy_0=policy_0,
            policy_1=policy_1,
            xp_policy=xp_policy,
            average_score=average_score,
            rewards_sample_list=rewards_sample_list,
            game_rollout=game_rollout,
            seed_0=seed_0,
            seed_1=seed_1,
            game_description=game_description,
        )
        
        # Build the feedback-based mutation prompt
        prompt = _build_feedback_mutation_prompt(
            thinking_style=self._thinking_style,
            cfd=cfd,
            max_bullet_points=max_bullet_points,
            analysis=analysis,
        )
        
        message = Message(role="user", content=prompt)
        
        json_response, _ = request_and_parse_to_json(
            client=self._client,
            model=self._model,
            msg_history=[message],
            temperature=self._llm_config.temperature,
            max_tokens=self._llm_config.max_tokens,
            reasoning_effort=self._llm_config.reasoning_effort,
            required_fields=["new_cfd"],
        )

        return json_response["new_cfd"]


# ============================================================================
# Diverse Feedback Mutation Operator
# ============================================================================

_DIVERSE_FEEDBACK_MUTATION_TEMPLATE = """You are helping to evolve a "coordination-friendly definition" (CFD) for zero-shot coordination (ZSC).

{background}

## Current CFD

{cfd}

## Feedback from Coordination Analysis

The following feedback was generated by analyzing a policy pair that might have performed poorly in cross-play:

{analysis}

## Your Task

{task_instructions}

**IMPORTANT:** Provide {n_proposals} DIVERSE proposals for the modified CFD. Each proposal should take a meaningfully different approach to addressing the feedback. Maximize diversity across your proposals.

Note: For some operations (e.g., removing a bullet point from a short CFD), fewer than {n_proposals} meaningfully different options may exist. In that case, return as many distinct proposals as possible — do not pad with near-duplicates.

## Constraints

- Each CFD must have at most {max_bullet_points} bullet points
- Each bullet point should be a single, concise sentence
{extra_constraints}

## Response Format

Respond with valid JSON:
```json
{{
    "reasoning": "{reasoning_prompt}",
    "proposals": [
        "- First bullet point\\n- Second bullet point\\n...",
        "- First bullet point\\n- Second bullet point\\n...",
        ...
    ]
}}
```"""


def _build_diverse_feedback_mutation_prompt(
    thinking_style: ThinkingStyle,
    cfd: str,
    max_bullet_points: int,
    analysis: str,
    n_proposals: int,
) -> str:
    """Build the diverse feedback-based mutation prompt for a given thinking style."""
    style_config = _THINKING_STYLE_INSTRUCTIONS[thinking_style]

    base_instructions = style_config["task_instructions"]
    feedback_instructions = f"""Based on the coordination analysis feedback above, {base_instructions.lower()}

Consider the analysis when making your changes."""

    return _DIVERSE_FEEDBACK_MUTATION_TEMPLATE.format(
        background=_SHARED_BACKGROUND,
        cfd=cfd,
        analysis=analysis,
        max_bullet_points=max_bullet_points,
        n_proposals=n_proposals,
        task_instructions=feedback_instructions,
        extra_constraints=style_config["extra_constraints"],
        reasoning_prompt=style_config["reasoning_prompt"],
    )


class DiverseFeedbackMutationOperator(FeedbackBasedMutationOperator):
    """
    A feedback-based mutation operator that asks for multiple diverse proposals
    informed by the coordination failure analysis, then randomly picks one.
    """

    def __init__(self, thinking_style: ThinkingStyle, llm_config: LLMConfig, n_proposals: int = 3):
        super().__init__(thinking_style, llm_config)
        self._n_proposals = n_proposals
        self._rng = np.random.default_rng()

    @property
    def name(self) -> str:
        return f"diverse_feedback_{self._thinking_style.value}"

    @observe()
    def mutate(self, genotype: Genotype, max_bullet_points: int) -> str:
        from meta_train.population_based.population import ScoreEntry

        cfd = genotype.cfd

        if not genotype.score_history:
            raise ValueError("Genotype has no score history. Cannot use feedback-based mutation.")

        last_score_entry: ScoreEntry = genotype.score_history[-1]
        experiment_results = last_score_entry.experiment_results

        xp_matrix = experiment_results.xp_matrix
        min_cell = _get_min_xp_cell(xp_matrix)
        seed_0, seed_1 = min_cell

        policy_0, policy_1, xp_policy, average_score, rewards_sample_list, game_rollout = \
            _extract_coordination_failure_context(experiment_results, min_cell)

        game_description = experiment_results.game_description or f"Game: {experiment_results.env_name_file_safe}"

        analysis = self._analyze_coordination_failure(
            cfd=cfd,
            policy_0=policy_0,
            policy_1=policy_1,
            xp_policy=xp_policy,
            average_score=average_score,
            rewards_sample_list=rewards_sample_list,
            game_rollout=game_rollout,
            seed_0=seed_0,
            seed_1=seed_1,
            game_description=game_description,
        )

        prompt = _build_diverse_feedback_mutation_prompt(
            thinking_style=self._thinking_style,
            cfd=cfd,
            max_bullet_points=max_bullet_points,
            analysis=analysis,
            n_proposals=self._n_proposals,
        )

        message = Message(role="user", content=prompt)

        json_response, _ = request_and_parse_to_json(
            client=self._client,
            model=self._model,
            msg_history=[message],
            temperature=self._llm_config.temperature,
            max_tokens=self._llm_config.max_tokens,
            reasoning_effort=self._llm_config.reasoning_effort,
            required_fields=["proposals"],
        )

        proposals = json_response["proposals"]
        if not proposals:
            return cfd

        valid_proposals = []
        for proposal in proposals:
            if not isinstance(proposal, str):
                continue
            bullet_points = [line.strip() for line in proposal.split('\n') if line.strip().startswith('-')]
            if len(bullet_points) > max_bullet_points:
                proposal = '\n'.join(bullet_points[:max_bullet_points])
            valid_proposals.append(proposal)

        if not valid_proposals:
            return cfd

        idx = int(self._rng.integers(0, len(valid_proposals)))
        return valid_proposals[idx]


# ============================================================================
# Mutation Operator Registry
# ============================================================================

# Registry of all available mutation operator names
# Format: "prefix_thinkingstyle" where prefix is "unary", "diverse", or "feedback"
# "reinit" is a standalone operator (no thinking style)
MUTATION_OPERATOR_NAMES: List[str] = ["reinit", "partial_reinit"]
for style in ThinkingStyle:
    MUTATION_OPERATOR_NAMES.append(f"unary_{style.value}")
    MUTATION_OPERATOR_NAMES.append(f"diverse_{style.value}")
    MUTATION_OPERATOR_NAMES.append(f"feedback_{style.value}")
    MUTATION_OPERATOR_NAMES.append(f"diverse_feedback_{style.value}")


def create_mutation_operator(name: str, llm_config: LLMConfig) -> MutationOperator:
    """
    Factory function to create mutation operators by name.
    
    Args:
        name: The operator name in format "prefix_thinkingstyle" 
              (e.g., "unary_add", "feedback_reformulate", "unary_atomic").
        llm_config: Configuration for the LLM to use.
    
    Returns:
        The corresponding MutationOperator instance.
    
    Raises:
        ValueError: If the operator name is not recognized.
    """
    if name == "reinit":
        return ReinitializationMutationOperator(llm_config)

    elif name == "partial_reinit":
        return PartialReinitMutationOperator(llm_config)

    elif name.startswith("unary_"):
        style_value = name[6:]  # Remove "unary_" prefix
        try:
            style = ThinkingStyle(style_value)
        except ValueError:
            raise ValueError(f"Unknown thinking style: {style_value}")
        return UnaryMutationOperator(style, llm_config)

    elif name.startswith("diverse_feedback_"):
        style_value = name[17:]  # Remove "diverse_feedback_" prefix
        try:
            style = ThinkingStyle(style_value)
        except ValueError:
            raise ValueError(f"Unknown thinking style: {style_value}")
        return DiverseFeedbackMutationOperator(style, llm_config)

    elif name.startswith("diverse_"):
        style_value = name[8:]  # Remove "diverse_" prefix
        try:
            style = ThinkingStyle(style_value)
        except ValueError:
            raise ValueError(f"Unknown thinking style: {style_value}")
        return DiverseForwardMutationOperator(style, llm_config)

    elif name.startswith("feedback_"):
        style_value = name[9:]  # Remove "feedback_" prefix
        try:
            style = ThinkingStyle(style_value)
        except ValueError:
            raise ValueError(f"Unknown thinking style: {style_value}")
        return FeedbackBasedMutationOperator(style, llm_config)

    else:
        raise ValueError(
            f"Unknown mutation operator name: {name}. "
            f"Available operators: {MUTATION_OPERATOR_NAMES}"
        )


def get_unary_operator_names() -> List[str]:
    """Return list of all unary mutation operator names."""
    return [f"unary_{style.value}" for style in ThinkingStyle]


def get_diverse_operator_names() -> List[str]:
    """Return list of all diversity-forward mutation operator names."""
    return [f"diverse_{style.value}" for style in ThinkingStyle]


def get_feedback_operator_names() -> List[str]:
    """Return list of all feedback-based mutation operator names."""
    return [f"feedback_{style.value}" for style in ThinkingStyle]

