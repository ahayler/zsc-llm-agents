from envs.base import Env
import numpy as np
from typing import Optional, Tuple, List, Literal, Dict
from core import CompactRolloutHistory
from configs.policy_modifiers import BasePolicyModifierConfig

game_description = """
Alice and Bob are archaeologists on a treasure hunt in the Egyptian desert. Alice has deciphered an ancient map that reveals the location of a hidden treasure, which could be at one of two ancient temples: the "Temple of Anubis" or the "Temple of Ra". Bob does not know which temple contains the treasure, so Alice must communicate the location to him somehow.

The challenge? Alice has two ways to share this information:
- "direct reveal" - show Bob the exact location on the map, but this requires time-consuming translation and risks damaging the fragile ancient parchment (joint reward of -5).
- "ancient symbols" - send a subtle signal using one of two mystical symbols that can be seen from afar: the "Eye of Horus" or the "Scarab Beetle" (no reward).

Alice can do exactly one of the following:
- abandon the treasure hunt and return to base camp, securing a joint reward of 1 (this ends the expedition immediately).
- reveal the exact temple location directly to Bob by showing him the map, resulting in a joint reward of -5.
- send an ancient symbol to Bob, either: the "Eye of Horus" or the "Scarab Beetle" (no reward).

If Alice doesn't abandon the hunt, it becomes Bob's turn. Based on the communication from Alice, Bob decides:
- abandon the treasure hunt and return to base camp, taking a joint reward of 0.5.
- guess which temple contains the treasure, either the "Temple of Anubis" or the "Temple of Ra".
    - If Bob's guess is correct, they successfully locate the treasure and earn a joint reward of +10.
    - If he's wrong, they search the wrong temple and waste precious time, resulting in a joint penalty of -10.

The players cannot communicate with each other except through the methods described above (i.e., showing the map directly or sending ancient symbols).
"""

observation_alice = """
You are Alice. You have deciphered the ancient map, which reveals that the treasure is located at the {location}.
"""

observation_bob = """
You are Bob. You observe that Alice has sent the ancient symbol {symbol_label} as a signal.
"""

dict_observation_description = """
For player_0 (Alice), the observation dict has the following format:
{{
    "location": str # "Temple of Anubis" or "Temple of Ra" (the treasure location from the ancient map)
}}

For player_1 (Bob), the observation dict has the following format:
{{
    "map_shown": bool # Whether the map has been shown directly (True) or not (False)
    "sent_symbol_eye": bool # Whether the "Eye of Horus" symbol has been sent (True) or not (False)
    "sent_symbol_scarab": bool # Whether the "Scarab Beetle" symbol has been sent (True) or not (False)
    "location": Optional[str] = None # "Temple of Anubis" or "Temple of Ra" (the treasure location), or None (if the map was not shown)
}}
"""

legal_actions_description = """
For player_0 (Alice), the legal actions are:
- "bail"
- "reveal_location"
- "send_symbol_eye"
- "send_symbol_scarab"

For player_1 (Bob), the legal actions are:
- "bail"
- "guess_Anubis"
- "guess_Ra"
"""

class CatAndDog4(Env):
    def __init__(self, policy_modifier_config: BasePolicyModifierConfig):
        super().__init__(policy_modifier_config)

    def reset(self, seed: Optional[int] = None, location: Optional[Literal["Temple of Anubis", "Temple of Ra"]] = None):
        if seed is not None:
            np.random.seed(seed)

        self.location = location if location is not None else np.random.choice(["Temple of Anubis", "Temple of Ra"])
        self.map_shown = False

        self.symbol_eye = False
        self.symbol_scarab = False

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
            return ["bail", "reveal_location", "send_symbol_eye", "send_symbol_scarab"]
        else:
            return ["bail", "guess_Anubis", "guess_Ra"]
    
    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:
        if self._current_player == 0:
            return (0, observation_alice.format(location=self.location), self.legal_actions)
        else:
            if self.map_shown:
                obs = f"Alice has shown you the ancient map. The treasure is located at the {self.location}."
            elif self.symbol_eye:
                obs = observation_bob.format(symbol_label="Eye of Horus")
                obs += " Furthermore, the map has not been shown to you. You cannot see the treasure location."
            elif self.symbol_scarab:
                obs = observation_bob.format(symbol_label="Scarab Beetle")
                obs += " Furthermore, the map has not been shown to you. You cannot see the treasure location."
            else:
                obs = "You are Bob. You observe that Alice has sent no ancient symbol."
                obs += " Furthermore, the map has not been shown to you. You cannot see the treasure location."
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
            return (1, {"map_shown": self.map_shown, "sent_symbol_eye": self.symbol_eye, "sent_symbol_scarab": self.symbol_scarab, "location": self.location if self.map_shown else None}, self.legal_actions)

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
                self.map_shown = True
                reward = -5
            elif action == "send_symbol_eye":
                reward = 0
                self.symbol_eye = True
            elif action == "send_symbol_scarab":
                reward = 0
                self.symbol_scarab = True

            self._current_player = 1
        else:
            # Bob's turn
            self.action_bob = action
            if action == "bail":
                reward = 0.5
            else:
                guessed_location = "Temple of Anubis" if action == "guess_Anubis" else "Temple of Ra"
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
            "map_shown": self.map_shown,
            "symbol_eye": self.symbol_eye,
            "symbol_scarab": self.symbol_scarab,
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
        return "Cat&Dog-v2"

if __name__ == "__main__":
    from configs.policy_modifiers import NoPolicyModifierConfig
    env = CatAndDog4(NoPolicyModifierConfig())
    env.reset(seed=1)
    print(env.text_observation)
    env.step("send_symbol_eye")
    print(env.text_observation)
    env.step("guess_Ra")
    print(env.get_compact_rollout_history())
    
