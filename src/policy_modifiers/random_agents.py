from policy_modifiers.base import BasePolicyModifier
from core import CodePolicy
import numpy as np
from configs.policy_modifiers import RandomAgentsPolicyModifierConfig
from typing import Dict, Tuple

policy_modifier_description = """
The RandomAgentsPolicyModifier is applied to the policy in the following way:
For each agent, their local policy is randomly chosen among the the local policies of all agents.
**Example:**
In a two player game, this means that with equal probability, player_0 plays against player_0, player_0 plays against player_1, player_1 plays against player_0 and player_1 plays against player_1.
"""

class RandomAgentsPolicyModifier(BasePolicyModifier):
    def __init__(self, policy_modifier_config: RandomAgentsPolicyModifierConfig):
        pass

    def modify_policy(self, policy: CodePolicy) -> Tuple[CodePolicy, Dict]:
        n_agents = len(policy.joint_policy)

        agents_idx = np.random.choice(n_agents, size=n_agents, replace=True)

        modified_policy = policy.clone()
        modified_policy.joint_policy = [policy.joint_policy[idx] for idx in agents_idx]
        modified_policy.name = f"{policy.name}_random_agents={str(agents_idx)}"

        metadata = {"agent_idx_to_local_policy_idx": {str(idx): agents_idx[idx] for idx in range(n_agents)}}

        return modified_policy, metadata

    @property
    def policy_modifier_description(self) -> str:
        return policy_modifier_description