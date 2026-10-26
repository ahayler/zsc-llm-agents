from typing import Tuple, Any
from functools import reduce

from configs.configs import GeneticCodeTrainerConfig, CodeXPEvaluatorConfig, DEFAULT_GENETIC_CODE_TRAINER_CONFIG, DEFAULT_CODE_XP_EVALUATOR_CONFIG
from configs.contexts import DEFAULT_XP_CONFIGS, XPConfig, CoordinationGame

def apply_override(cfg, key: str, value: Any):
    keys = key.split(".")
    current_target = cfg
    # Traverse nested config objects
    for _key in keys[:-1]:
        if not hasattr(current_target, _key):
            raise ValueError(f"Config {cfg} has no attribute {_key} at path {key}")
        current_target = getattr(current_target, _key)

    if not hasattr(current_target, keys[-1]):
        raise ValueError(f"Config {cfg} has no attribute {keys[-1]} at path {key}")

    setattr(current_target, keys[-1], value)

    return cfg


# Meta-training samples starting policies at temperature 1.0 for diversity (paper, Appendix G.1);
# evaluations keep the default generator temperature of 0.
META_TRAINING_GENERATOR_TEMPERATURE = 1.0


def setup_code_configs_from_xp_config(xp_config: XPConfig) -> Tuple[GeneticCodeTrainerConfig, CodeXPEvaluatorConfig]:
    # Deep copies so per-game overrides never leak into the shared defaults.
    trainer_config = DEFAULT_GENETIC_CODE_TRAINER_CONFIG.model_copy(deep=True)
    evaluator_config = DEFAULT_CODE_XP_EVALUATOR_CONFIG.model_copy(deep=True)
    trainer_config.generator_config.llm_config.temperature = META_TRAINING_GENERATOR_TEMPERATURE

    for key, value in xp_config.overwrites.items():
        if key.startswith("trainer_config."):
            apply_override(trainer_config, key[len("trainer_config."):], value)
        elif key.startswith("evaluator_config."):
            apply_override(evaluator_config, key[len("evaluator_config."):], value)
        else:
            raise ValueError(f"Invalid key: {key}")

    evaluator_config.env_config = xp_config.env_config  
    trainer_config.env_config = xp_config.env_config

    return trainer_config, evaluator_config



def setup_code_configs_from_default_game_context(env_name: CoordinationGame) -> Tuple[GeneticCodeTrainerConfig, CodeXPEvaluatorConfig]:
    """
    Sets up the code config for the given env_name.
    """

    context = DEFAULT_XP_CONFIGS[env_name]

    return setup_code_configs_from_xp_config(context)