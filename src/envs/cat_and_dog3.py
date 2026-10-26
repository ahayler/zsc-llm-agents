from envs.base import Env
import numpy as np
from typing import Optional, Tuple, List, Literal, Dict
from core import CompactRolloutHistory
from configs.policy_modifiers import BasePolicyModifierConfig

game_description = """
Across the galaxy, astronauts Alice and Bob are stranded on opposite sides of the universe, and must rendevous at a specific point in space inorder to return to Earth. There are two possible rendevous coordinates, which are either Planet "AZ-2" or Planet "LK-7". 

The rendevous location was given to Alice by HQ before the expedition, but Bob does not know it. Hence, Alice must send the rendevous location to Bob in some way.

The catch? Alice has two types of communication channels:
- "direct line" - reveals exact location of the rendezvous point, but is very slow and requires a lot of energy.
- "quantum line" - sends a subtle, instantaneous signal across light-years, which is of two types: signal "X" or signal "Y".

Alice can do exactly one of the following:
- abort the rendezvous and wait for rescue, securing a joint reward of 1.
- reveal the exact rendevous location directly to Bob, resulting in a joint reward of -5.
- send a quantum pulse to Bob, either: signal "X" or signal "Y" (no reward).

If Alice doesn't abort, it becomes Bob's turn. Based on the communication sent by Alice, Bob decides:
- abort the rendezvous and wait for rescue, taking a joint reward of 0.5.
- guess the rendevous location, which is either Planet "AZ-2" or Planet "LK-7".
    - If Bob's guess is correct, they meet safely at the rendevous location, earning a joint reward of +10.
    - If he's wrong, they miss the rendevous location, scattering themselves across the void forever for a joint penalty of -10.
"""

observation_alice = """
You are Alice. You have the rendevous location, which is {location}.
"""

observation_bob = """
You are Bob. You observe that Alice has sent a signal {signal_label} on the quantum line.
"""

dict_observation_description = """
For player_0 (Alice), the observation dict has the following format:
{{
    "location": str # "AZ-2" or "LK-7" (the rendevous location)
}}

For player_1 (Bob), the observation dict has the following format:
{{
    "direct_line": bool # Whether the direct line has been used (True) or not (False)
    "sent_signal_X": bool Whether the signal "X" has been sent (True) or not (False), or None (if no signal was sent)
    "location": Optional[str] = None # "AZ-2" or "LK-7" (the rendevous location), or None (if no signal was sent)
}}
"""

legal_actions_description = """
For player_0 (Alice), the legal actions are:
- "bail"
- "reveal_location"
- "send_signal_X"
- "send_signal_Y"

For player_1 (Bob), the legal actions are:
- "bail"
- "guess_AZ-2"
- "guess_LK-7"
"""

class CatAndDog3(Env):
    def __init__(self, policy_modifier_config: BasePolicyModifierConfig):
        super().__init__(policy_modifier_config)

    def reset(self, seed: Optional[int] = None, location: Optional[Literal["AZ-2", "LK-7"]] = None):
        if seed is not None:
            np.random.seed(seed)

        self.location = location if location is not None else np.random.choice(["AZ-2", "LK-7"])
        self.direct_line_used = False

        self.signal_X = False
        self.signal_Y = False

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
            return ["bail", "reveal_location", "send_signal_X", "send_signal_Y"]
        else:
            return ["bail", "guess_AZ-2", "guess_LK-7"]
    
    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:
        if self._current_player == 0:
            return (0, observation_alice.format(location=self.location), self.legal_actions)
        else:
            if self.direct_line_used:
                obs = f"Alice has revealed the rendevous location to you: {self.location}."
            elif self.signal_X:
                obs = observation_bob.format(signal_label="X")
                obs += " Furthermore, the direct line has not been used. You cannot see the rendevous location."
            elif self.signal_Y:
                obs = observation_bob.format(signal_label="Y")
                obs += " Furthermore, the direct line has not been used. You cannot see the rendevous location."
            else:
                obs = "You are Bob. You observe that Alice has sent no signal."
                obs += " Furthermore, the direct line has not been used. You cannot see the rendevous location."
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
            return (0, {"location": self.location}, self.legal_actions)
        else:
            return (1, {"direct_line": self.direct_line_used, "sent_signal_X": self.signal_X, "sent_signal_Y": self.signal_Y, "location": self.location if self.direct_line_used else None}, self.legal_actions)

    def step(self, action: str) -> Tuple[float, bool, bool, dict]:
        if action not in self.legal_actions:
            raise ValueError(f"Invalid action: {action}")
        
        if self._current_player == 0:
            # Alice's turn
            self.action_alice = action
            if action == "bail":
                reward = 1
                self._terminated = True # Game ends early if Alice bails
            elif action == "reveal_location":
                self.direct_line_used = True
                reward = -5
            elif action == "send_signal_X":
                reward = 0
                self.signal_X = True
            elif action == "send_signal_Y":
                reward = 0
                self.signal_Y = True

            self._current_player = 1
        else:
            # Bob's turn
            self.action_bob = action
            if action == "bail":
                reward = 0.5
            else:
                guessed_location = "AZ-2" if action == "guess_AZ-2" else "LK-7"
                if guessed_location == self.location:
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
            "location": self.location,
            "direct_line_used": self.direct_line_used,
            "signal_X": self.signal_X,
            "signal_Y": self.signal_Y,
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
        return "Cat&Dog-v1"

if __name__ == "__main__":
    from configs.policy_modifiers import NoPolicyModifierConfig
    env = CatAndDog3(NoPolicyModifierConfig())
    env.reset(seed=1)
    print(env.text_observation)
    env.step("reveal_location")
    print(env.text_observation)
    env.step("bail")
    print(env.get_compact_rollout_history())
    
