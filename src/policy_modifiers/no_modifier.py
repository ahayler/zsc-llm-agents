from policy_modifiers.base import BasePolicyModifier
from core import CodePolicy
from configs.policy_modifiers import NoPolicyModifierConfig
from typing import Dict, Tuple

class NoPolicyModifier(BasePolicyModifier):
    def __init__(self, policy_modifier_config: NoPolicyModifierConfig):
        pass

    def modify_policy(self, policy: CodePolicy) -> Tuple[CodePolicy, Dict]:
        return policy, {}

    @property
    def policy_modifier_description(self) -> str:
        return ""