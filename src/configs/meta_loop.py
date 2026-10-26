from pandas.io.sql import pandasSQL_builder
from pydantic import BaseModel
from typing import List
from configs.configs import LLMConfig, CriticOutput
from core import CodePolicy, CompactRolloutHistory
from llm.request import LLMType

class SizeConstraints(BaseModel):
    max_num_bullet_points: int
    max_sentences_per_bullet_point: int
    max_words_per_sentence: int

class AbstractMetaProblem(BaseModel):
    coordination_friendly_definition: str

class SkippedMetaProblem(AbstractMetaProblem):
    game_description: str
    reason: str

class SkippedDueToFailureMetaProblem(SkippedMetaProblem):
    pass

class SkippedDueToSuccessMetaProblem(SkippedMetaProblem):
    pass

class CoordinationFailureMetaProblem(AbstractMetaProblem):
    game_description: str
    policy_0: CodePolicy
    policy_1: CodePolicy
    critic_acceptance_output_0: CriticOutput
    critic_acceptance_output_1: CriticOutput
    xp_policy: CodePolicy
    average_score: float
    rewards_sample_list: List[float]
    game_rollout: CompactRolloutHistory

class TooStrictMetaProblem(AbstractMetaProblem):
    game_description: str
    negative_critic_outputs: List[CriticOutput]
    sp_scores: List[float]

class LowSPMetaProblem(AbstractMetaProblem):
    game_description: str
    low_sp_seed_policy: CodePolicy
    low_sp_game_rollout: CompactRolloutHistory
    low_sp_score: float

class MetaOptimizerConfig(BaseModel):
    llm_config: LLMConfig
    use_memory_buffer: bool
    coordination_friendly_definition_size_constraints: SizeConstraints
    memory_size_constraints: SizeConstraints
    max_history: int

DEFAULT_META_OPTIMIZER_CONFIG = MetaOptimizerConfig(
    llm_config=LLMConfig(
        model=LLMType.OR_GPT5,
        temperature=1.0,
        max_tokens=100000
    ),
    use_memory_buffer=True,
    coordination_friendly_definition_size_constraints=SizeConstraints(
        max_num_bullet_points=5,
        max_sentences_per_bullet_point=1,
        max_words_per_sentence=20
    ),
    memory_size_constraints=SizeConstraints(
        max_num_bullet_points=5,
        max_sentences_per_bullet_point=1,
        max_words_per_sentence=20,
    ),
    max_history=2
)
