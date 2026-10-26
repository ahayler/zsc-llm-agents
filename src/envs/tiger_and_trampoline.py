from envs.base import Env
from core import CompactRolloutHistory
import numpy as np
from typing import Tuple, Dict, List
from configs.policy_modifiers import BasePolicyModifierConfig

GAME_DESCRIPTION = """
In this game, Alice is standing on a balcony, while Bob stands on the ground next to a lever.
Pulling the lever will either deploy a trampoline below the balcony, or release a tiger. 
From above, Alice can observe which is the case, and then Alice decides whether to jump off the balcony (JUMP) or not (STAY). 
Bob observes Alice's choice but does not know whether the lever will deploy a trampoline or tiger, and decides to pull the lever (PULL) or not (STAY).
Alice wants to get down from the balcony, but will die from the fall unless Bob pulls the lever and it releases a trampoline. 
If Bob pulls the lever and releases a tiger, it will eat Bob.
In particular, Alice chooses first. Bob observes Alice's choice, when making his choice.

The exact (joint) payoffs are the following:

- If there is a tiger:
    - If Alice doesn't jump and Bob doesn't pull the lever: 0 (Nothing happens)
    - If Alice doesn't jump and Bob pulls the lever: -10 (Bob get's eaten)
    - If Alice jumps and Bob pulls the lever: -20 (Alices crashes and Bob is eaten)
    - If Alice jumps and Bob doesn't pull the lever: -10 (Alice crashes)
- If there is a trampoline:
    - If Alice doesn't jump and Bob doesn't pull the lever: 0 (Nothing happens)
    - If Alice doesn't jump and Bob pulls the lever: 0 (Nothing happens)
    - If Alice jumps and Bob pulls the lever: +1 (Alice lands on the trampoline)
    - If Alice jumps and Bob doesn't pull the lever: -10 (Alice crashes)

**In short:**

- If Alice lands  +1
- If Alice crashes -10
- If Bob gets eaten -10
- If Bob gets eaten and Alice crashes -20

**Role asignment:**

- player 0 is alice
- player 1 is bob
"""

dict_observation_description = """
The observation is a dictionary with the following keys:
- "current_player": int; The index of the current player (0 for Alice, 1 for Bob).
- "is_tiger": Optional[bool]; Whether the lever releases a tiger. Known to Alice, unknown to Bob (None).
- "alice_jumped": Optional[bool]; Whether Alice jumped (None before Alice acts).
- "bob_pulled_lever": Optional[bool]; Whether Bob pulled the lever (None before Bob acts).
- "p_tiger": float; The probability that the lever releases a tiger.
"""

possible_actions_description = """
The possible actions depend on the current player:
- Alice (player_0): "JUMP" or "STAY"
- Bob (player_1): "PULL" or "STAY"
"""
class TigerAndTrampoline(Env):
    def __init__(self, policy_modifier_config: BasePolicyModifierConfig, p_tiger: float = 0.5):
        super().__init__(policy_modifier_config)
        
        self.p_tiger = p_tiger

        self.is_tiger = None
        self._current_player = None
        self.alice_jumped = None
        self.bob_pulled_lever = None

        self.rewards = None
        self.actions = None
        self._terminated = False
        self._truncated = False


    def reset(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.is_tiger = np.random.binomial(1, self.p_tiger)
        self._current_player = 0
        self.alice_jumped = None
        self.bob_pulled_lever = None
        self.rewards = []
        self.actions = {0: [], 1: []}
        self._terminated = False
        self._truncated = False
        

    @property
    def legal_actions(self) -> List[str]:
        if self._current_player == 0:
            return ["JUMP", "STAY"]
        else:
            return ["PULL", "STAY"]
    
    @property
    def base_game_description(self) -> str:
        return GAME_DESCRIPTION

    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:

        if self.current_player == 0:
            # Alice's turn
            if self.is_tiger:
                obs = "You are Alice. You can see a tiger from the balcony."
            else:
                obs = "You are Alice. You can see a trampoline from the balcony."
        else:
            if self.alice_jumped:
                obs = "You are Bob. You can see that Alice jumped."
            else:
                obs = "You are Bob. You can see that Alice stayed."

        return (self.current_player, obs, self.legal_actions)
        
    def step(self, action: str)-> Tuple[float, bool, bool, Dict]:
        if action not in self.legal_actions:
            raise ValueError(f"Invalid action: {action}")

        terminated = False
        reward = 0

        if self.current_player == 0:
            # Alice's turn
            self.alice_jumped = (action == "JUMP")
            self.actions[0].append(action)
            self._current_player = 1
        else:
            # Bob's turn
            self.bob_pulled_lever = (action == "PULL")
            self.actions[1].append(action)

            if self.is_tiger:
                if self.alice_jumped:
                    reward += -10
                if self.bob_pulled_lever:
                    reward += -10
            else:
                if self.alice_jumped and not self.bob_pulled_lever:
                    reward += -10
                if self.alice_jumped and self.bob_pulled_lever:
                    reward += 1

            terminated = True
            self._terminated = True

        self.rewards.append(reward)

        return reward, self.terminated, self.truncated, {}
    
        
    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated
    
    @property
    def env_name(self) -> str:
        return "Tiger&Trampoline-v0"

    @property
    def dict_observation(self) -> Tuple[int, Dict, List[str]]:
        if self.current_player == 0:
            is_tiger = bool(self.is_tiger)
        else:
            is_tiger = None

        obs_dict = {
            "current_player": self.current_player,
            "is_tiger": is_tiger,
            "alice_jumped": self.alice_jumped,
            "bob_pulled_lever": self.bob_pulled_lever,
            "p_tiger": self.p_tiger,
        }
        return self.current_player, obs_dict, self.legal_actions

    @property
    def dict_observation_description(self) -> str:
        return dict_observation_description

    @property
    def possible_actions_description(self) -> str:
        return possible_actions_description

    def get_compact_rollout_history(self, ignore_not_done: bool = False) -> CompactRolloutHistory:
        if not self.done and not ignore_not_done:
            raise ValueError("Please finish the game before creating a compact rollout history.")

        compact_states = [
            {
                "step": 0,
                "current_player": 0,
                "alice_jumped": None,
                "bob_pulled_lever": None,
            },
            {
                "step": 1,
                "current_player": self.current_player,
                "alice_jumped": self.alice_jumped,
                "bob_pulled_lever": self.bob_pulled_lever,
            },
        ]

        hyperparams = {
            "p_tiger": self.p_tiger,
        }

        metadata = {
            "is_tiger": bool(self.is_tiger),
            "alice_action": self.actions[0][0] if self.actions[0] else None,
            "bob_action": self.actions[1][0] if self.actions[1] else None,
        }

        return CompactRolloutHistory(
            hyperparams=hyperparams,
            metadata=metadata,
            rewards_sum=sum(self.rewards),
            compact_states=compact_states,
            terminated=self.terminated,
            truncated=self.truncated,
        )

    @property
    def example_text_observations(self) -> Dict[str, str]:
        return {
            "alice_observation": "You are Alice. You can see a tiger from the balcony.",
            "bob_observation": "You are Bob. You can see that Alice jumped.",
        }
        
        


        
        