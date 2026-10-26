from typing import List, Optional, Literal, Union, Callable
from pydantic import BaseModel, model_validator
from llm.request import LLMType
from core import CodePolicy
from configs.contexts import XPConfig
from utils.log_utils import get_logger
import os
from dataclasses import dataclass

from configs.envs import EnvConfig

from abc import abstractmethod, ABC

POLICY_CRITIC_COORDINATION_FRIENDLY_DEFINITION = """
A policy is not coordination-friendly if it satisfies any of the following conditions:
- The policy makes use of arbitrary conventions. For this, ask the following question:
    - Is there any signalling convention that could be replaced with a different, equivalent convention?
    - Is there any role assignment that could be flipped? (This only makes sense if two players are symmetric in the game)
    - Note: Most policies make use of some convention. You need to assess whether this convention is arbitrary or not.

Otherwise, the policy is coordination-friendly. Think through whether any of the above conditions are fulfilled in the reasoning section.
"""

logger = get_logger(__name__)

class LLMConfig(BaseModel):
    model: LLMType
    temperature: float
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = "medium"

class BasePolicyAgentConfig(BaseModel):
    _policy_str: Optional[str] = ""

    @property
    def policy_str(self) -> str:
        """Get the policy string, raising an error if it's empty."""
        if not self._policy_str or self._policy_str.strip() == "":
            raise ValueError("Policy string is empty - cannot retrieve empty policy")
        return self._policy_str
    
    @policy_str.setter
    def policy_str(self, value: str):
        self._policy_str = value

class BaseTextAgentConfig(BaseModel, ABC):
    llm_config: LLMConfig

    @abstractmethod
    def get_short_name(self) -> str:
        pass


class ChatHistoryAgentConfig(BaseTextAgentConfig):
    verbose_actions: bool
    max_history: int
    include_random_number: bool = False
    # Optional "coordination-friendly definition" (CFD) string injected into the prompt.
    # Keep empty to disable.
    cfd_str: str = ""

    def get_short_name(self) -> str:
        model_name = self.llm_config.model.split("/")[-1]
        has_cfd = bool(self.cfd_str and self.cfd_str.strip())
        return f"chat_history_agent_{model_name}_reasoning_effort={self.llm_config.reasoning_effort}_verbose={self.verbose_actions}_max_hist={self.max_history}_include_random_number={self.include_random_number}_has_cfd={has_cfd}"

class CodePolicyAgentConfig(BasePolicyAgentConfig):
    pass

class GeneratorConfig(BaseModel):
    llm_config: LLMConfig
    num_policies: int
    num_players: int
    seed: Optional[int] = None

class CachedGenerationConfig(BaseModel):
    experiment_name: str
    env_name_file_safe: str

class CodePolicyGeneratorConfig(GeneratorConfig):
    use_cached_generations: Optional[bool] = False
    cached_generations_config: Optional[CachedGenerationConfig] = None
    randomly_swap_local_policies: Optional[bool] = False
    n_generations_per_seed: int = 1
    use_self_critic: bool = False

    @model_validator(mode="after")
    def validate_cached_generations_config(self):
        if self.use_cached_generations:
            if self.cached_generations_config is None:
                raise ValueError("cached_generations_config is not set, but use_cached_generations is True")
        return self
        
class AbstractRolloutConfig(BaseModel):
    _env_config: Optional[EnvConfig] = None
    num_game_runs: int
    num_agents: int

    @property
    def env_config(self) -> EnvConfig:
        if self._env_config is None:
            raise ValueError("Env config is not set")
        return self._env_config
    
    @env_config.setter
    def env_config(self, value: EnvConfig):
        self._env_config = value

class CodeRolloutConfig(AbstractRolloutConfig):
    agent_config_factory: Callable[[], CodePolicyAgentConfig]

    def get_agent_configs(self, policy: CodePolicy) -> List[CodePolicyAgentConfig]:
        if self.num_agents != len(policy.joint_policy):
            raise ValueError(f"Number of agents in rollout config ({self.num_agents}) does not match number of players in policy ({len(policy.joint_policy)})")
        
        agent_configs = [self.agent_config_factory() for _ in range(self.num_agents)]

        for agent_id in range(self.num_agents):
            agent_configs[agent_id].policy_str = policy.joint_policy[agent_id]
        
        return agent_configs

class PolicyCriticConfig(BaseModel):
    llm_config: LLMConfig
    max_iterations: int
    reformulate_definition: bool
    coordination_friendly_definition: str
    require_double_approval: bool = False
    judge_independently: bool = True

class AbstractTrainerConfig(BaseModel):
    top_k: int
    use_policy_critic: bool
    generator_config: GeneratorConfig
    policy_critic_config: PolicyCriticConfig

class CodeXPEvaluatorConfig(CodeRolloutConfig):
    pass

class GeneticCodeTrainerConfig(CodeRolloutConfig, AbstractTrainerConfig):
    invalid_action_xp_evaluator_config: CodeXPEvaluatorConfig
    use_invalid_action_filtering_for_generated_policies: bool = False
    sp_filtering: bool = True

@dataclass
class CriticOutput:
    coordination_friendly: bool
    reasoning: str
    input_policy: CodePolicy
    modified_policy: Optional[CodePolicy] = None
    modified_policy_reason: Optional[str] = None

@dataclass
class CriticHistory:
    critic_outputs: List[CriticOutput]
    found_coordination_friendly_policy: bool
    final_policy: Optional[CodePolicy] = None

# Default model for code-policy generation and the policy critic (Grok 4.1 Fast, as in the paper).
# Set ZSC_DEBUG=1 to use the cheaper GPT-5 mini instead, e.g. for quick tests.
if os.getenv("ZSC_DEBUG"):
    DEFAULT_REASONING_MODEL = LLMType.OR_GPT5_MINI
    logger.info("ZSC_DEBUG is set: using GPT-5 mini as the default reasoning model.")
else:
    DEFAULT_REASONING_MODEL = LLMType.OR_GROK_41_FAST

DEFAULT_CODE_POLICY_AGENT_CONFIG_FACTORY = lambda: CodePolicyAgentConfig(
)

DEFAULT_CHAT_HISTORY_AGENT_CONFIG = ChatHistoryAgentConfig(
    verbose_actions=True,
    llm_config=LLMConfig(
        model=LLMType.OR_GPT5_MINI,
        reasoning_effort="medium",
        temperature=0.0,
        max_tokens=100000
    ),
    max_history=10,
    include_random_number=False
)

DEFAULT_POLICY_CRITIC_CONFIG = PolicyCriticConfig(
    llm_config=LLMConfig(
        model=LLMType.OR_GROK_4_FAST,
        temperature=0.0,
        max_tokens=100000
    ),
    max_iterations=4,
    reformulate_definition=False,
    coordination_friendly_definition=POLICY_CRITIC_COORDINATION_FRIENDLY_DEFINITION
)

DEFAULT_GENERATOR_CONFIG = GeneratorConfig(
    llm_config=LLMConfig(
        model=DEFAULT_REASONING_MODEL,
        temperature=1.0,
        max_tokens=100000
    ),
    num_policies=5,
    num_players=2
)

DEFAULT_CODE_POLICY_GENERATOR_CONFIG = CodePolicyGeneratorConfig(
    llm_config=LLMConfig(
        model=DEFAULT_REASONING_MODEL,
        temperature=0.0,
        max_tokens=100000
    ),
    num_policies=1,
    num_players=2,
    seed=None,
    use_cached_generations=False,
    cached_generations_experiment_name=None,
    randomly_swap_local_policies=False,
    n_generations_per_seed=1,
    use_self_critic=False
)

DEFAULT_GENETIC_CODE_TRAINER_CONFIG = GeneticCodeTrainerConfig(
    agent_config_factory=DEFAULT_CODE_POLICY_AGENT_CONFIG_FACTORY,
    use_policy_critic=True,
    use_invalid_action_filtering_for_generated_policies=False,
    top_k=1,
    num_game_runs=10,
    num_agents=2,
    generator_config=DEFAULT_CODE_POLICY_GENERATOR_CONFIG,
    policy_critic_config=DEFAULT_POLICY_CRITIC_CONFIG,
    invalid_action_xp_evaluator_config=CodeXPEvaluatorConfig(
        agent_config_factory=DEFAULT_CODE_POLICY_AGENT_CONFIG_FACTORY,
        num_game_runs=10,
        num_agents=2,
    )
)

DEFAULT_CODE_XP_EVALUATOR_CONFIG = CodeXPEvaluatorConfig(
    agent_config_factory=DEFAULT_CODE_POLICY_AGENT_CONFIG_FACTORY,
    num_game_runs=10,
    num_agents=2,
)
