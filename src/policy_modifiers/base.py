from abc import ABC, abstractmethod
from core import CodePolicy
from typing import Dict, Tuple

class BasePolicyModifier(ABC):
    @abstractmethod
    def modify_policy(self, policy: CodePolicy) -> Tuple[CodePolicy, Dict]:
        pass

    @property
    @abstractmethod
    def policy_modifier_description(self) -> str:
        pass