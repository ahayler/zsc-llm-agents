from envs.base import Env
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
class TigerAndTrampoline(Env):
    def __init__(self, policy_modifier_config: BasePolicyModifierConfig, p_tiger: float = 0.5):
        super().__init__(policy_modifier_config)
        
        self.p_tiger = p_tiger

        self.is_tiger = None
        self._current_player = None
        self.alice_jumped = None
        self.bob_pulled_lever = None

        self.rewards = None


    def reset(self, seed: int = None):
        if seed is not None:
            self.seed(seed)

        self.is_tiger = np.random.binomial(1, self.p_tiger)
        self._current_player = 0

        self.rewards = []
        

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
            self._current_player = 1
        else:
            # Bob's turn
            self.bob_pulled_lever = (action == "PULL")

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

        self.rewards.append(reward)

        return reward, terminated, False, {}
    
        
    @property
    def current_player(self) -> int:
        return self._current_player
    
    @property
    def env_name(self) -> str:
        return "Tiger&Trampoline-v0"
        
        


        
        