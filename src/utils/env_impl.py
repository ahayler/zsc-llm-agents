from typing import Union
from envs.base import Env
from configs.envs import (
    EnvConfig, SelectionWithSemanticsConfig, ConventionsWithSemanticsConfig, TigerAndTrampolineConfig, 
    BrittleBridgeConfig, 
    CatAndDogConfig, 
    CatAndDog2Config,
    CatAndDog3Config,
    CatAndDog4Config,
    PermutedLeversConfig,
    NarrowTunnelConfig,
    AdHocConventionsConfig,
    AdHocConventionsPermutedSignalsConfig,
    AdHocConventionsPermutedSignals2Config,
    AdHocConventionsPermutedSignals3Config,
    AdHocConventionsPermutedSignals4Config,
    MatrixGameConfig,
    TinyHanabiConfig,
    TinyHanabiFixedConfig,
    HanabiTextConfig,
    HintGuessConfig,
    HintGuessImplicaturesConfig,
)
from configs.contexts import XPConfig
from envs.brittle_bridge import BrittleBridge
from envs.cat_and_dog import CatAndDog
from envs.cat_and_dog2 import CatAndDog2
from envs.cat_and_dog3 import CatAndDog3
from envs.cat_and_dog4 import CatAndDog4
from envs.permuted_levers import PermutedLevers
from envs.narrow_tunnel import NarrowTunnel
from envs.ad_hoc_conventions import AdHocConventions
from envs.ad_hoc_conventions_permuted_signals import AdHocConventionsPermutedSignals
from envs.ad_hoc_conventions_permuted_signals2 import AdHocConventionsPermutedSignals2
from envs.ad_hoc_conventions_permuted_signals3 import AdHocConventionsPermutedSignals3
from envs.ad_hoc_conventions_permuted_signals4 import AdHocConventionsPermutedSignals4
from envs.matrix_game import MatrixGame
from envs.tiny_hanabi import TinyHanabi
from envs.tiny_hanabi_fixed import TinyHanabiFixed
from envs.action_features.hint_guess import HintGuess
from envs.action_features.hint_guess_implicatures import HintGuessImplicatures
from envs.hanabi.hanabi_text_env import HanabiTextEnv
from envs.selection_with_semantics import SelectionWithSemantics
from envs.conventions_with_semantics import ConventionsWithSemantics
from envs.tiger_and_trampoline import TigerAndTrampoline

def get_env_impl_from_config(env_config: EnvConfig) -> type[Env]:
    """
    Returns the appropriate environment implementation class based on the config type.
    
    Args:
        env_config: An environment configuration object
        
    Returns:
        The corresponding environment implementation class
        
    Raises:
        ValueError: If the config type is not recognized
    """
    if isinstance(env_config, BrittleBridgeConfig):
        return BrittleBridge
    elif isinstance(env_config, CatAndDogConfig):
        return CatAndDog
    elif isinstance(env_config, CatAndDog2Config):
        return CatAndDog2
    elif isinstance(env_config, CatAndDog3Config):
        return CatAndDog3
    elif isinstance(env_config, CatAndDog4Config):
        return CatAndDog4
    elif isinstance(env_config, PermutedLeversConfig):
        return PermutedLevers
    elif isinstance(env_config, NarrowTunnelConfig):
        return NarrowTunnel
    elif isinstance(env_config, AdHocConventionsConfig):
        return AdHocConventions
    elif isinstance(env_config, AdHocConventionsPermutedSignalsConfig):
        return AdHocConventionsPermutedSignals
    elif isinstance(env_config, AdHocConventionsPermutedSignals2Config):
        return AdHocConventionsPermutedSignals2
    elif isinstance(env_config, AdHocConventionsPermutedSignals3Config):
        return AdHocConventionsPermutedSignals3
    elif isinstance(env_config, AdHocConventionsPermutedSignals4Config):
        return AdHocConventionsPermutedSignals4
    elif isinstance(env_config, MatrixGameConfig):
        return MatrixGame
    elif isinstance(env_config, TinyHanabiConfig):
        return TinyHanabi
    elif isinstance(env_config, TinyHanabiFixedConfig):
        return TinyHanabiFixed
    elif isinstance(env_config, HanabiTextConfig):
        return HanabiTextEnv
    elif isinstance(env_config, HintGuessImplicaturesConfig):
        return HintGuessImplicatures
    elif isinstance(env_config, HintGuessConfig):
        return HintGuess
    elif isinstance(env_config, SelectionWithSemanticsConfig):
        return SelectionWithSemantics
    elif isinstance(env_config, ConventionsWithSemanticsConfig):
        return ConventionsWithSemantics
    elif isinstance(env_config, TigerAndTrampolineConfig):
        return TigerAndTrampoline
    else:
        raise ValueError(f"Unknown environment config type: {type(env_config)}")


def get_env_name_from_config(env_config: EnvConfig) -> str:
    """
    Returns the appropriate environment implementation class based on the config type.
    
    Args:
        env_config: An environment configuration object
        
    Returns:
        The corresponding environment implementation class
        
    Raises:
        ValueError: If the config type is not recognized
    """
    if isinstance(env_config, BrittleBridgeConfig):
        return "BrittleBridge"
    elif isinstance(env_config, CatAndDogConfig):
        return "CatAndDog"
    elif isinstance(env_config, PermutedLeversConfig):
        return "PermutedLevers"
    elif isinstance(env_config, NarrowTunnelConfig):
        return "NarrowTunnel"
    elif isinstance(env_config, AdHocConventionsConfig):
        return "AdHocConventions"
    elif isinstance(env_config, AdHocConventionsPermutedSignalsConfig):
        return "AdHocConventionsPermutedSignals"
    elif isinstance(env_config, MatrixGameConfig):
        return "MatrixGame"
    elif isinstance(env_config, TinyHanabiConfig):
        return "TinyHanabi"
    elif isinstance(env_config, TinyHanabiFixedConfig):
        return "TinyHanabiFixed"
    elif isinstance(env_config, HanabiTextConfig):
        suffix = f"_{env_config.num_colors}x{env_config.num_ranks}" if (env_config.num_colors != 5 or env_config.num_ranks != 5) else ""
        if env_config.max_life_tokens != 3:
            suffix += f"_{env_config.max_life_tokens}life"
        return f"Hanabi{suffix}"
    elif isinstance(env_config, HintGuessImplicaturesConfig):
        return "HintGuessImplicatures"
    elif isinstance(env_config, HintGuessConfig):
        return "HintGuess"
    elif isinstance(env_config, SelectionWithSemanticsConfig):
        return "SelectionWithSemantics"
    elif isinstance(env_config, ConventionsWithSemanticsConfig):
        return "ConventionsWithSemantics"
    elif isinstance(env_config, TigerAndTrampolineConfig):
        return "TigerAndTrampoline"
    else:
        raise ValueError(f"Unknown environment config type: {type(env_config)}")


def get_env_name_file_save_from_config(env_config: Union[EnvConfig, XPConfig]) -> str:
    """
    Returns the file-safe environment name based on the config type.
    
    If an XPConfig is provided and has env_name_file_safe set, that value is used.
    This allows numbered variants (e.g., ad_hoc_conventions_permuted_signals2, 
    ad_hoc_conventions_permuted_signals3, ad_hoc_conventions_permuted_signals4) 
    to be distinguished even though they use the same EnvConfig class.
    Otherwise, the name is derived from the EnvConfig type.
    
    Args:
        env_config: An environment configuration object (EnvConfig or XPConfig)
        
    Returns:
        The file-safe environment name string
        
    Raises:
        ValueError: If the config type is not recognized
    """
    # If it's an XPConfig with env_name_file_safe set, use that
    # This handles numbered variants like ad_hoc_conventions_permuted_signals2, 3, 4
    if isinstance(env_config, XPConfig):
        if env_config.env_name_file_safe is not None:
            return env_config.env_name_file_safe
        # Otherwise, extract the env_config and continue
        env_config = env_config.env_config
    
    # Now handle EnvConfig types (fallback for when XPConfig doesn't have env_name_file_safe set)
    if isinstance(env_config, BrittleBridgeConfig):
        return "brittle_bridge"
    elif isinstance(env_config, CatAndDogConfig):
        return "cat_and_dog"
    elif isinstance(env_config, CatAndDog2Config):
        return "cat_and_dog2"
    elif isinstance(env_config, CatAndDog3Config):
        return "cat_and_dog3"
    elif isinstance(env_config, CatAndDog4Config):
        return "cat_and_dog4"
    elif isinstance(env_config, PermutedLeversConfig):
        return "permuted_levers"
    elif isinstance(env_config, NarrowTunnelConfig):
        return "narrow_tunnel"
    elif isinstance(env_config, AdHocConventionsConfig):
        return "ad_hoc_conventions"
    elif isinstance(env_config, AdHocConventionsPermutedSignalsConfig):
        return "ad_hoc_conventions_permuted_signals"
    elif isinstance(env_config, AdHocConventionsPermutedSignals2Config):
        return "ad_hoc_conventions_permuted_signals2"
    elif isinstance(env_config, AdHocConventionsPermutedSignals3Config):
        return "ad_hoc_conventions_permuted_signals3"
    elif isinstance(env_config, AdHocConventionsPermutedSignals4Config):
        return "ad_hoc_conventions_permuted_signals4"
    elif isinstance(env_config, MatrixGameConfig):
        return "matrix_game"
    elif isinstance(env_config, TinyHanabiConfig):
        return "tiny_hanabi"
    elif isinstance(env_config, TinyHanabiFixedConfig):
        return "tiny_hanabi_fixed"
    elif isinstance(env_config, HanabiTextConfig):
        suffix = f"_{env_config.num_colors}x{env_config.num_ranks}" if (env_config.num_colors != 5 or env_config.num_ranks != 5) else ""
        if env_config.max_life_tokens != 3:
            suffix += f"_{env_config.max_life_tokens}life"
        return f"hanabi{suffix}"
    elif isinstance(env_config, HintGuessImplicaturesConfig):
        return "hint_guess_implicatures"
    elif isinstance(env_config, HintGuessConfig):
        return "hint_guess"
    elif isinstance(env_config, SelectionWithSemanticsConfig):
        return "selection_with_semantics"
    elif isinstance(env_config, ConventionsWithSemanticsConfig):
        return "conventions_with_semantics"
    elif isinstance(env_config, TigerAndTrampolineConfig):
        return "tiger_and_trampoline"
    else:
        raise ValueError(f"Unknown environment config type: {type(env_config)}")

