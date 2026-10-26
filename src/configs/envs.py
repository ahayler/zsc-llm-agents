from pydantic import BaseModel
from typing import List, Literal, Optional
from enum import Enum
from configs.policy_modifiers import BasePolicyModifierConfig, NoPolicyModifierConfig, RandomAgentsPolicyModifierConfig

class CoordinationGame(Enum):
    BRITTLE_BRIDGE = "brittle_bridge"
    BRITTLE_BRIDGE_RANDOM_AGENTS = "brittle_bridge_random_agents"
    CAT_AND_DOG = "cat_and_dog"
    CAT_AND_DOG2 = "cat_and_dog2"
    CAT_AND_DOG3 = "cat_and_dog3"
    CAT_AND_DOG4 = "cat_and_dog4"
    PERMUTED_LEVERS = "permuted_levers"
    PERMUTED_LEVERS_RANDOM_AGENTS = "permuted_levers_random_agents"
    NARROW_TUNNEL = "narrow_tunnel"
    AD_HOC_CONVENTIONS = "ad_hoc_conventions"
    AD_HOC_CONVENTIONS_PERMUTED_SIGNALS = "ad_hoc_conventions_permuted_signals"
    AD_HOC_CONVENTIONS_PERMUTED_SIGNALS2 = "ad_hoc_conventions_permuted_signals2"
    AD_HOC_CONVENTIONS_PERMUTED_SIGNALS3 = "ad_hoc_conventions_permuted_signals3"
    AD_HOC_CONVENTIONS_PERMUTED_SIGNALS4 = "ad_hoc_conventions_permuted_signals4"
    MATRIX_GAME = "matrix_game"
    TINY_HANABI = "tiny_hanabi"
    TINY_HANABI_FIXED = "tiny_hanabi_fixed"
    HINT_GUESS = "hint_guess"
    HINT_GUESS_IMPLICATURES = "hint_guess_implicatures"
    HANABI_TEXT = "hanabi_text"

class EnvConfig(BaseModel):
    policy_modifier_config: BasePolicyModifierConfig = NoPolicyModifierConfig()

class PermutedLeversConfig(EnvConfig):
    num_turns: int
    num_levers: int

class NarrowTunnelConfig(EnvConfig):
    template: List[Literal["n", "w"]]
    max_turns: int

class TigerAndTrampolineConfig(EnvConfig):
    p_tiger: float

class CatAndDogConfig(EnvConfig):
    pass

class CatAndDog2Config(EnvConfig):
    pass

class CatAndDog3Config(EnvConfig):
    pass

class CatAndDog4Config(EnvConfig):
    pass

class AdHocConventionsConfig(EnvConfig):
    lever_labels: List[str]
    signal_labels: List[str]
    num_rounds: int


class AdHocConventionsPermutedSignalsConfig(EnvConfig):
    lever_labels: List[str]
    signal_labels: List[str]
    button_labels: List[str]
    num_rounds: int

class AdHocConventionsPermutedSignals2Config(EnvConfig):
    lever_labels: List[str]
    signal_labels: List[str]
    button_labels: List[str]
    num_rounds: int

class AdHocConventionsPermutedSignals3Config(EnvConfig):
    lever_labels: List[str]
    signal_labels: List[str]
    button_labels: List[str]
    num_rounds: int

class AdHocConventionsPermutedSignals4Config(EnvConfig):
    lever_labels: List[str]
    signal_labels: List[str]
    button_labels: List[str]
    num_rounds: int

class BrittleBridgeConfig(EnvConfig):
    bridge_length: int
    bridge_break_threshold: int
    max_turns: int

class MatrixGameConfig(EnvConfig):
    template_name: Literal["TrajeDi"]
    permutation_seed: Optional[int]

class TinyHanabiConfig(EnvConfig):
    pass

class TinyHanabiFixedConfig(EnvConfig):
    pass

class HanabiTextConfig(EnvConfig):
    num_agents: int = 2
    num_colors: int = 5
    num_ranks: int = 5
    max_info_tokens: int = 8
    max_life_tokens: int = 3

class HintGuessConfig(EnvConfig):
    hand_size: int = 5
    feature_space_1: List[str] = ["1", "2", "3"]
    feature_space_2: List[str] = ["A", "B", "C"]
    permute_hands: bool = False

class HintGuessImplicaturesConfig(EnvConfig):
    """Hand-crafted 4-scenario test set from Bullard et al. (Fig. 2 / Table 1)."""
    permute_hands: bool = True


DEFAULT_PERMUTED_LEVERS_CONFIG = PermutedLeversConfig(
    num_turns=10,
    num_levers=5
)

DEFAULT_PERMUTED_LEVERS_RANDOM_AGENTS_CONFIG = PermutedLeversConfig(
    num_turns=10,
    num_levers=5,
    policy_modifier_config=RandomAgentsPolicyModifierConfig()
)

DEFAULT_NARROW_TUNNEL_CONFIG_CODE = NarrowTunnelConfig(
    template=["w", "n", "w"],
    max_turns=100
)

DEFAULT_NARROW_TUNNEL_CONFIG_TEXT = NarrowTunnelConfig(
    template=["w", "n", "w"],
    max_turns=20
)

DEFAULT_TIGER_AND_TRAMPOLINE_CONFIG = TigerAndTrampolineConfig(
    p_tiger=0.5
)

DEFAULT_BRITTLE_BRIDGE_CONFIG_CODE = BrittleBridgeConfig(
    bridge_length=5,
    bridge_break_threshold=2,
    max_turns=200,
)

DEFAULT_BRITTLE_BRIDGE_CONFIG_TEXT = BrittleBridgeConfig(
    bridge_length=5,
    bridge_break_threshold=2,
    max_turns=20,
)

DEFAULT_BRITTLE_BRIDGE_RANDOM_AGENTS_CONFIG = BrittleBridgeConfig(
    bridge_length=5,
    bridge_break_threshold=2,
    max_turns=200,
    policy_modifier_config=RandomAgentsPolicyModifierConfig()
)

DEFAULT_AD_HOC_CONVENTIONS_CONFIG = AdHocConventionsConfig(
    lever_labels=["blue", "red"],
    signal_labels=["square", "circle"],
    num_rounds=10,
)

DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG = AdHocConventionsPermutedSignalsConfig(
    lever_labels=["blue", "red"],
    signal_labels=["square", "circle"],
    button_labels=["apple", "banana"],
    num_rounds=10,
)

DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG2 = AdHocConventionsPermutedSignalsConfig(
    lever_labels=["fiction", "nonfiction"],
    signal_labels=["lightning", "thunder"],
    button_labels=["phi", "theta"],
    num_rounds=10,
)

DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG3 = AdHocConventionsPermutedSignalsConfig(
    lever_labels=["Aisle_A", "Aisle_B"],
    signal_labels=["carrot", "pumpkin"],
    button_labels=["bin_1", "bin_2"],
    num_rounds=10,
)

DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG4 = AdHocConventionsPermutedSignalsConfig(
    lever_labels=["station_1", "station_2"],
    signal_labels=["lion", "giraffe"],
    button_labels=["willow", "maple"],
    num_rounds=10,
)

DEFAULT_MATRIX_GAME_CONFIG = MatrixGameConfig(
    template_name="TrajeDi",
    permutation_seed=None,
)

DEFAULT_TINY_HANABI_CONFIG = TinyHanabiConfig()
DEFAULT_TINY_HANABI_FIXED_CONFIG = TinyHanabiFixedConfig()

DEFAULT_HANABI_TEXT_CONFIG = HanabiTextConfig()
DEFAULT_HANABI_TEXT_3X3_CONFIG = HanabiTextConfig(num_colors=3, num_ranks=3, max_info_tokens=3)
DEFAULT_HANABI_TEXT_3X3_1LIFE_CONFIG = HanabiTextConfig(num_colors=3, num_ranks=3, max_info_tokens=3, max_life_tokens=1)
DEFAULT_HANABI_TEXT_4X4_CONFIG = HanabiTextConfig(num_colors=4, num_ranks=4, max_info_tokens=5)

DEFAULT_HINT_GUESS_CONFIG = HintGuessConfig()
DEFAULT_HINT_GUESS_IMPLICATURES_CONFIG = HintGuessImplicaturesConfig()
