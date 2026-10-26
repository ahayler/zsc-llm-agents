from envs.base import Env
import numpy as np
from typing import Optional, Tuple, List, Literal, Dict
from core import CompactRolloutHistory
from configs.policy_modifiers import BasePolicyModifierConfig

game_description = """
Alice and Bob are playing a guessing game. Alice can see a card in her hand, which is either a 1 or a 2. Bob cannot see the card in Alice's hand.
Additionally, Alice has access to two buttons, "A" and "B". When she presses a button, she will show a signal to Bob.
Alice can do exactly one of the following:
- bail for a (joint) reward of 1 (this ends the game immediately)
- show the card to Bob, for a (joint) reward of -5 
- press a button "A" or "B" to show a signal to Bob. (no reward)

If Alice didn't bail, it is Bob's turn. Bob observes the signal shown by Alice or the card she showed.
Bob can do exactly one of the following:
- bail for a (joint) reward of 0.5
- or guess the card, i.e. either 1 or 2
    - if Bob guesses correctly, they receive a (joint) reward of 10
    - if Bob guesses incorrectly, they receive a (joint) reward of -10

The players cannot communicate with each other except the options presented above (i.e. signal shown by Alice or card shown by Alice).
"""

observation_alice = """
You are Alice. You have a card in your hand, which is a {card_label}.
"""

observation_bob = """
You are Bob. You observe that Alice has pressed button {button_label}.
"""

dict_observation_description = """
For player_0 (Alice), the observation dict has the following format:
{{
    "card_label": str # "1" or "2" (the card in Alice's hand)
}}

For player_1 (Bob), the observation dict has the following format:
{{
    "card_shown": bool # Whether the card has been shown (True) or not (False)
    "pressed_A": bool Whether the button "A" has been pressed (True) or not (False), or None (if no button was pressed)
    "card_label": Optional[str] = None # "1" or "2" (the card in Alice's hand), or None (if no card was shown)
}}
"""

legal_actions_description = """
For player_0 (Alice), the legal actions are:
- "bail"
- "show_card"
- "press_A"
- "press_B"

For player_1 (Bob), the legal actions are:
- "bail"
- "guess_1"
- "guess_2"
"""

class CatAndDog2(Env):
    def __init__(self, policy_modifier_config: BasePolicyModifierConfig):
        super().__init__(policy_modifier_config)

    def reset(self, seed: Optional[int] = None, card_label: Optional[Literal["1", "2"]] = None):
        if seed is not None:
            np.random.seed(seed)

        self.card_label = card_label if card_label is not None else np.random.choice(["1", "2"])
        self.card_shown = False

        self.pressed_A = False
        self.pressed_B = False

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
            return ["bail", "show_card", "press_A", "press_B"]
        else:
            return ["bail", "guess_1", "guess_2"]
    
    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:
        if self._current_player == 0:
            return (0, observation_alice.format(card_label=self.card_label), self.legal_actions)
        else:
            obs = observation_bob.format(button_label="A" if self.pressed_A else "B")
            if self.card_shown:
                obs += f"Furthermore, the card has been shown. You can see a {self.card_label}."
            else:
                obs += f"Furthermore, the card has not been shown. You cannot see the card."
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
            return (0, {"card_label": self.card_label}, self.legal_actions)
        else:
            return (1, {"card_shown": self.card_shown, "pressed_A": self.pressed_A, "card_label": self.card_label if self.card_shown else None}, self.legal_actions)

    def step(self, action: str) -> Tuple[float, bool, bool, dict]:
        if action not in self.legal_actions:
            raise ValueError(f"Invalid action: {action}")
        
        if self._current_player == 0:
            # Alice's turn
            self.action_alice = action
            if action == "bail":
                reward = 1
                self._terminated = True # Game ends early if Alice bails
            elif action == "show_card":
                self.card_shown = True
                reward = -5
            elif action == "press_A":
                self.pressed_A = True
                reward = 0
            elif action == "press_B":
                self.pressed_B = True
                reward = 0

            self._current_player = 1
        else:
            # Bob's turn
            self.action_bob = action
            if action == "bail":
                reward = 0.5
            else:
                guessed_card = "1" if action == "guess_1" else "2"
                if guessed_card == self.card_label:
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
            "card_label": self.card_label,
            "pressed_A": self.pressed_A,
            "pressed_B": self.pressed_B,
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
    env = CatAndDog2(NoPolicyModifierConfig())
    env.reset()
    print(env.text_observation)
    env.step("bail")
    print(env.text_observation)
    env.step("bail")
    print(env.get_compact_rollout_history())
    
