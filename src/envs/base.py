from policy_modifiers.base import BasePolicyModifier
from policy_modifiers.random_agents import RandomAgentsPolicyModifier
from policy_modifiers.no_modifier import NoPolicyModifier
from configs.policy_modifiers import BasePolicyModifierConfig, NoPolicyModifierConfig, RandomAgentsPolicyModifierConfig

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from core import CompactRolloutHistory

class Env(ABC):
    policy_modifier: BasePolicyModifier

    def __init__(self, policy_modifier_config: BasePolicyModifierConfig):
        if isinstance(policy_modifier_config, NoPolicyModifierConfig):
            self.policy_modifier = NoPolicyModifier(policy_modifier_config)
        elif isinstance(policy_modifier_config, RandomAgentsPolicyModifierConfig):
            self.policy_modifier = RandomAgentsPolicyModifier(policy_modifier_config)
        else:
            raise ValueError(f"Unsupported policy modifier config: {policy_modifier_config}")

    @property
    def game_description(self) -> str:
        """Get the game description.
        """
        if not isinstance(self.policy_modifier, NoPolicyModifier):
            game_description = self.base_game_description

            game_description += f"""
IMPORTANT: This game is played with a policy modifier. A policy modifier is applied to the policy every time a game is started and should be considered part of the game.
{self.policy_modifier.policy_modifier_description}
"""
            return game_description
        else:
            return self.base_game_description

    @property
    @abstractmethod
    def base_game_description(self) -> str:
        """Get the base game description.
        """
        pass

    @property
    @abstractmethod
    def legal_actions(self) -> List[str]:
        """Get the list of legal actions for the current player.
        """
        pass

    @property
    @abstractmethod
    def text_observation(self) -> Tuple[int, str, List[str]]:
        """Get the current observation of the environment.

        Returns:
            Tuple[int, str, List[str]]: A tuple containing:
                - current_player (int): The index of the current player
                - observation (str): The observation of the current state
                - legal_actions (List[str]): The list of legal actions for the current player
        """
        pass


    @property
    @abstractmethod
    def dict_observation(self) -> Tuple[int, Dict, List[str]]:
        """Get the current observation of the environment as a dictionary.

        Returns:
            Tuple[int, Dict, List[str]]: A tuple containing:
                - current_player (int): The index of the current player
                - observation (Dict): The observation of the current state
                - legal_actions (List[str]): The list of legal actions for the current player
        """
        pass

    @property
    @abstractmethod
    def dict_observation_description(self) -> str:
        """Get a textual description of an observation dictionary.
        """
        pass

    @property
    @abstractmethod
    def possible_actions_description(self) -> str:
        """Get a textual description of the possible actions.
        """
        pass

    @abstractmethod
    def get_compact_rollout_history(self, ignore_not_done: bool = False) -> CompactRolloutHistory:
        """Get a compact rollout history of the environment.
        """
        pass

    @property
    @abstractmethod
    def terminated(self) -> bool:
        """Check if the environment is terminated.
        """
        pass

    @property
    @abstractmethod
    def truncated(self) -> bool:
        """Check if the environment is truncated.
        """
        pass

    @property
    def done(self) -> bool:
        """Check if the environment is done.
        """
        return self.terminated or self.truncated
    
    def assert_not_done(self):
        """Assert that the environment is not done.
        """
        if self.terminated:
            raise ValueError("Environment is terminated")
        if self.truncated:
            raise ValueError("Environment is truncated")

    @abstractmethod
    def reset(self, seed: Optional[int] = None):
        """Reset the environment.

        Args:
            seed (Optional[int]): The seed to reset the environment with.
        """
        pass

    @abstractmethod
    def step(self, action: str) -> tuple[float, bool, bool, Dict]:
        """Execute one environment step.

        Takes an action (of the current player) and returns the next state observation, reward, and done flags.

        Args:
            action (str): The action to take in the environment

        Returns:
            tuple[float, bool, bool, Dict]: A tuple containing:
                - reward (float): The reward received from the action
                - terminated (bool): Whether the episode terminated naturally
                - truncated (bool): Whether the episode was artificially terminated
                - info (Dict): Additional information about the step
        """
        pass

    # @abstractmethod
    # def validate_action(self, action: str) -> Tuple[bool, str]:
    #     """Validate an action.

    #     Args:
    #         action (str): The action to validate

    #     Returns:
    #         Tuple[bool, str]: A tuple containing:
    #             - valid (bool): Whether the action is valid
    #             - error_msg (str): The error message if the action is invalid
    #     """
    #     pass

    @property
    @abstractmethod
    def current_player(self) -> int:
        """Get the index of the current player.
        """
        pass

    @property
    @abstractmethod
    def env_name(self) -> str:
        """Get the name of the environment.
        """
        pass

    @property
    @abstractmethod
    def example_text_observations(self) -> Dict[str, str]:
        """Returns a dictionary of example observations for the environment.
        """
        pass

    # TODO: Add more properties (states, actions, etc.)