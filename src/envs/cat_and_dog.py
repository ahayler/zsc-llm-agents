from envs.base import Env
import numpy as np
from typing import Optional, Tuple, List, Literal, Dict
from core import CompactRolloutHistory
from configs.policy_modifiers import BasePolicyModifierConfig

game_description = """
You are playing a game of cat and dog with one other player. One of you is Alice and the other is Bob.
Alice is separated from Bob by a wall. The goal of the game is maximize your (joint) reward. 
On Alice's side of the wall, there is either a cat or a dog. 
Additionally, there is a light switch on Alice's side of the wall, which she can turn on to illuminate a bulb on Bob's side of the wall.

Alice goes first and can do exactly one of the following:
- bail for a (joint) reward of 1 (this ends the game immediately)
- remove the wall for a (joint) reward of -5
- turn on the light bulb (no reward)
- do nothing, i.e. the light bulb stays off (no reward)

If Alice didn't bail, it is Bob's turn. Bob observes the light status and the animal on Alice's side of the wall, if the wall was removed by Alice.
Bob can do exactly one of the following:
- bail for a (joint) reward of 0.5
- or guess the animal, i.e. either cat or dog
    - if Bob guesses correctly, he receives a (joint) reward of 10
    - if Bob guesses incorrectly, he receives a (joint) reward of -10

The players cannot communicate with each other except the options presented above (i.e. light status and removing the wall).
"""

# **Role asignment:**

# - player 0 is alice
# - player 1 is bob

observation_alice = """
You are Alice. On your side of the wall, there is a {animal}.
"""

# Whether the wall is down or up will be handeled in the code
observation_bob = """
You are Bob. The light bulb on your side of the wall is {light_status}. 
"""

dict_observation_description = """
For player_0 (Alice), the observation dict has the following format:
{{
    "animal": str # "cat" or "dog" (the animal on Alice's side of the wall)
}}

For player_1 (Bob), the observation dict has the following format:
{{
    "wall_up": bool # Whether the wall is up (True) or down/removed (False)
    "light_on": bool # Whether the light is on (True) or off (False)
    "animal": Optional[str] = None "cat" or "dog" (if the wall was removed by Alice), or None (if the wall is up)
}}
"""

legal_actions_description = """
For player_0 (Alice), the legal actions are:
- "bail"
- "remove_wall"
- "turn_on_light"
- "do_nothing"

For player_1 (Bob), the legal actions are:
- "bail"
- "guess_dog"
- "guess_cat"
"""

class CatAndDog(Env):
    def __init__(self, policy_modifier_config: BasePolicyModifierConfig):
        super().__init__(policy_modifier_config)

    def reset(self, seed: Optional[int] = None, animal: Optional[Literal["cat", "dog"]] = None):
        if seed is not None:
            np.random.seed(seed)

        self.animal = animal if animal is not None else np.random.choice(["cat", "dog"])

        self.light_on = False
        self.wall_up = True

        self.action_alice = None
        self.action_bob = None

        self._current_player = 0
        self.rewards = []
        self.actions = []
        self._terminated = False

    @property
    def example_text_observations(self) -> Dict[str, str]:
        return {
            "observation_player_0": observation_alice,
            "observation_player_1": observation_bob,
        }

    @property
    def base_game_description(self) -> str:
        return game_description
    
    @property
    def legal_actions(self) -> List[str]:
        if self._current_player == 0:
            return ["bail", "remove_wall", "turn_on_light", "do_nothing"]
        else:
            return ["bail", "guess_dog", "guess_cat"]
    
    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:
        if self._current_player == 0:
            return (0, observation_alice.format(animal=self.animal), self.legal_actions)
        else:
            obs = observation_bob.format(light_status="on" if self.light_on else "off")
            if not self.wall_up:
                obs += f"Furthermore, the wall has been removed. You can see a {self.animal} on the other side of the wall."
            else:
                obs += f"Furthermore, the wall has not been removed. You cannot see the animal on the other side of the wall."
            return (1, obs, self.legal_actions)
        
    @property
    def terminated(self) -> bool:
        return self._terminated
    
    @property
    def truncated(self) -> bool:
        return False
        
    @property
    def dict_observation(self) -> Tuple[int, Dict, List[str]]:
        if self._current_player == 0:
            return (0, {"animal": self.animal}, self.legal_actions)
        else:
            return (1, {"wall_up": self.wall_up, "light_on": self.light_on, "animal": self.animal if not self.wall_up else None}, self.legal_actions)

    def step(self, action: str) -> Tuple[float, bool, bool, dict]:
        if action not in self.legal_actions:
            raise ValueError(f"Invalid action: {action}")
        
        if self._current_player == 0:
            # Alice's turn
            self.action_alice = action
            if action == "bail":
                reward = 1
                self._terminated = True # Game ends early if Alice bails
            elif action == "remove_wall":
                self.wall_up = False
                reward = -5
            elif action == "turn_on_light":
                self.light_on = True
                reward = 0
            elif action == "do_nothing":
                reward = 0

            self._current_player = 1
        else:
            # Bob's turn
            self.action_bob = action
            if action == "bail":
                reward = 0.5
            else:
                guessed_animal = "dog" if action == "guess_dog" else "cat"
                if guessed_animal == self.animal:
                    reward = 10
                else:
                    reward = -10

            self._terminated = True

        self.rewards.append(reward)

        return reward, self._terminated, False, {}
    
    @property
    def dict_observation_description(self) -> str:
        return dict_observation_description
    
    @property
    def possible_actions_description(self) -> str:
        return legal_actions_description
    
    def get_compact_rollout_history(self, ignore_not_done: bool = False) -> CompactRolloutHistory:
        if not self.done and not ignore_not_done:
            raise ValueError("Environment is not done yet!")

        metadata = {
            "animal": self.animal,
            "light_on": self.light_on,
            "wall_up": self.wall_up,
        }

        compact_rollout_history = CompactRolloutHistory(
            metadata=metadata,
            hyperparams={},
            rewards_sum=sum(self.rewards),
            terminated=self._terminated,
            truncated=False,
            compact_states=[{
                "action_alice": self.action_alice,
                "action_bob": self.action_bob,
            }]
        )

        return compact_rollout_history
    
    @property
    def current_player(self) -> int:
        return self._current_player
    
    @property
    def env_name(self) -> str:
        return "Cat&Dog-v0"
    
if __name__ == "__main__":
    from configs.policy_modifiers import NoPolicyModifierConfig
    env = CatAndDog(NoPolicyModifierConfig())
    env.reset()
    print(env.text_observation)
    env.step("bail")
    print(env.text_observation)
    env.step("bail")
    print(env.get_compact_rollout_history())