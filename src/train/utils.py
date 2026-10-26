from core import CodePolicy
from envs.base import Env
from configs.envs import EnvConfig
from pydantic import BaseModel
from utils.env_impl import get_env_impl_from_config

code_policy_str = """
{{
    "name": "{name}",
    "player_0": {{
        "code": "{local_policy_player_0}"
    }},
    "player_1": {{
        "code": "{local_policy_player_1}"
    }}
}}
"""

def code_policy_to_str(policy: CodePolicy) -> str:
    return code_policy_str.format(
        name=policy.name,
        local_policy_player_0=policy.joint_policy[0],
        local_policy_player_1=policy.joint_policy[1],
        
    )

def pydantic_config_to_shallow_dict(config: BaseModel) -> dict:
    shallow_dict = {k: getattr(config, k) for k in config.model_fields}
    return shallow_dict

def get_env_from_config(env_config: EnvConfig) -> Env:
    return get_env_impl_from_config(env_config)(**pydantic_config_to_shallow_dict(env_config))

def get_game_description(env_config: EnvConfig) -> str:
    return get_env_from_config(env_config).game_description
