from envs.base import Env
from envs.ad_hoc_conventions import AdHocConventions
from envs.ad_hoc_conventions2 import AdHocConventions2
from envs.ad_hoc_conventions3 import AdHocConventions3
from envs.ad_hoc_conventions4 import AdHocConventions4
from envs.ad_hoc_conventions_permuted_signals import AdHocConventionsPermutedSignals
from envs.ad_hoc_conventions_permuted_signals2 import AdHocConventionsPermutedSignals2
from envs.ad_hoc_conventions_permuted_signals3 import AdHocConventionsPermutedSignals3
from envs.ad_hoc_conventions_permuted_signals4 import AdHocConventionsPermutedSignals4
from envs.bridge_and_buttons import BridgeAndButtons
from envs.brittle_bridge import BrittleBridge
from envs.cat_and_dog import CatAndDog
from envs.cat_and_dog2 import CatAndDog2
from envs.cat_and_dog3 import CatAndDog3
from envs.cat_and_dog4 import CatAndDog4
from envs.matrix_game import MatrixGame
from envs.narrow_tunnel import NarrowTunnel
from envs.permuted_levers import PermutedLevers
from envs.tiger_and_trampoline import TigerAndTrampoline
from envs.selection_with_semantics import SelectionWithSemantics
from envs.conventions_with_semantics import ConventionsWithSemantics
from envs.tiny_hanabi import TinyHanabi
from envs.tiny_hanabi_fixed import TinyHanabiFixed
from envs.action_features.hint_guess import HintGuess
from envs.action_features.hint_guess_implicatures import HintGuessImplicatures
from envs.hanabi.hanabi_text_env import HanabiTextEnv
# from envs.overcooked.wrapper import Overcooked
from configs.policy_modifiers import NoPolicyModifierConfig, BasePolicyModifierConfig
from typing import Optional

# Environment name to class mapping
_ENV_REGISTRY = {
    "ad_hoc_conventions": AdHocConventions,
    "ad_hoc_conventions2": AdHocConventions2,
    "ad_hoc_conventions3": AdHocConventions3,
    "ad_hoc_conventions4": AdHocConventions4,
    "ad_hoc_conventions_permuted_signals": AdHocConventionsPermutedSignals,
    "ad_hoc_conventions_permuted_signals2": AdHocConventionsPermutedSignals2,
    "ad_hoc_conventions_permuted_signals3": AdHocConventionsPermutedSignals3,
    "ad_hoc_conventions_permuted_signals4": AdHocConventionsPermutedSignals4,
    "bridge_and_buttons": BridgeAndButtons,
    "brittle_bridge": BrittleBridge,
    "cat_and_dog": CatAndDog,
    "cat_and_dog2": CatAndDog2,
    "cat_and_dog3": CatAndDog3,
    "cat_and_dog4": CatAndDog4,
    "matrix_game": MatrixGame,
    "narrow_tunnel": NarrowTunnel,
    "permuted_levers": PermutedLevers,
    "tiger_and_trampoline": TigerAndTrampoline,
    "selection_with_semantics": SelectionWithSemantics,
    "conventions_with_semantics": ConventionsWithSemantics,
    "tiny_hanabi": TinyHanabi,
    "tiny_hanabi_fixed": TinyHanabiFixed,
    "hint_guess": HintGuess,
    "hint_guess_implicatures": HintGuessImplicatures,
    "hanabi_text": HanabiTextEnv,
    # "overcooked": Overcooked,
}


def make(
    env_name: str,
    policy_modifier_config: Optional[BasePolicyModifierConfig] = None,
    **kwargs
) -> Env:
    """
    Create an environment instance by name.
    
    Args:
        env_name: Name of the environment (case-insensitive, supports underscores or camelCase)
        policy_modifier_config: Policy modifier configuration. Defaults to NoPolicyModifierConfig()
        **kwargs: Additional arguments specific to the environment class
        
    Returns:
        An instance of the requested environment
        
    Raises:
        ValueError: If the environment name is not recognized
        
    Examples:
        >>> env = make("cat_and_dog")
        >>> env = make("matrix_game", template_name="TrajeDi")
        >>> env = make("brittle_bridge", bridge_length=10)
    """
    # Normalize environment name (case-insensitive, convert camelCase to snake_case)
    env_name_lower = env_name.lower().replace("-", "_")
    
    # Try exact match first
    env_class = _ENV_REGISTRY.get(env_name_lower)
    
    # If not found, try matching class names (e.g., "CatAndDog" -> "cat_and_dog")
    if env_class is None:
        for key, cls in _ENV_REGISTRY.items():
            if cls.__name__.lower() == env_name_lower or cls.__name__ == env_name:
                env_class = cls
                break
    
    if env_class is None:
        available = ", ".join(sorted(_ENV_REGISTRY.keys()))
        raise ValueError(
            f"Unknown environment '{env_name}'. Available environments: {available}"
        )
    
    # Use default policy modifier config if not provided
    if policy_modifier_config is None:
        policy_modifier_config = NoPolicyModifierConfig()
    
    # Create environment instance
    return env_class(policy_modifier_config=policy_modifier_config, **kwargs)


__all__ = [
    "Env",
    "make",
    "AdHocConventions",
    "AdHocConventions2",
    "AdHocConventions3",
    "AdHocConventions4",
    "AdHocConventionsPermutedSignals",
    "AdHocConventionsPermutedSignals2",
    "AdHocConventionsPermutedSignals3",
    "AdHocConventionsPermutedSignals4",
    "BridgeAndButtons",
    "BrittleBridge",
    "CatAndDog",
    "CatAndDog2",
    "CatAndDog3",
    "CatAndDog4",
    "MatrixGame",
    "NarrowTunnel",
    "PermutedLevers",
    "TigerAndTrampoline",
    "SelectionWithSemantics",
    "ConventionsWithSemantics",
    "TinyHanabi",
    "TinyHanabiFixed",
    "HintGuess",
    "HintGuessImplicatures",
    "HanabiTextEnv",
    # "Overcooked",
]

