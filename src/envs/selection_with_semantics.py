import numpy as np
from typing import Optional, List, Tuple, Dict
from core import CompactRolloutHistory
from configs.policy_modifiers import BasePolicyModifierConfig
from envs.base import Env

game_description = """
You are playing Selection with Semantics with one other player for one round.
There are {num_levers} levers labeled 1 to {num_levers}. Both players observe the same levers, but the numbering varies between players.
Each lever is associated with a payoff, which is visible to both players (but the levers might have different numbers for each player).
Additionally, one of the levers is marked with a red arrow. This red arrow is visible to both players.

The lever with the red arrow as well as eight other levers have a payoff of 1.0. One of the other levers has a payoff of 1.01.
If you both choose the same underlying lever (not the same number, but the same lever), you receive that lever's payoff for the round; otherwise you receive 0.
"""

observation = """
It is the only round of the game.
Your lever payoffs are: {lever_payoffs_text}
The red arrow points at lever {red_arrow_lever} (visible to both players).
"""

dict_observation_description = """
An observation has the following format. Each list is ordered from the first (and only) turn.
The levers are 1-indexed and all lever references are from the perspective of the current player.
{
    "current_turn": int # The current turn number (starts at 1)
    "max_turn": int # The number of total turns in the game (always 1)
    "num_levers": int # The number of levers in the game
    "red_arrow_lever": int # The lever label with the red arrow (visible to both players)
    "lever_payoffs": List[float] # Payoffs aligned to your lever labels (index 0 is lever 1)
    "your_actions": List[int] # A list of previously picked levers by you (empty before the round resolves)
    "other_actions": List[int] # A list of previously picked levers by the other player
    "rewards": List[float] # A list of rewards you have received in the last turns
}
"""

possible_actions_description = """
The possible actions for each player are (each number corresponds to a lever):
- "1"
- "2"
- "3"
- ...
- "{num_levers}"

Note that the action is a string, not an integer.
"""


class SelectionWithSemantics(Env):
    def __init__(self, policy_modifier_config: BasePolicyModifierConfig, num_levers: int = 5):
        super().__init__(policy_modifier_config)

        assert num_levers >= 2, "Number of levers must be at least 2"

        self.num_levers = num_levers

        self.current_turn = None
        self._current_player = None
        self.permutations = None
        self.inverse_permutations = None

        self.red_arrow_lever = None
        self.high_payoff_lever = None
        self.lever_payoffs = None

        self.rewards = None
        self.actions = None
        self.states = None
        self._terminated = None

    def reset(self, seed: Optional[int] = None, no_permutation: bool = False):
        if seed is not None:
            np.random.seed(seed)

        if no_permutation:
            self.permutations = [np.arange(self.num_levers) for _ in range(2)]
        else:
            self.permutations = [np.random.permutation(self.num_levers) for _ in range(2)]

        self._update_inverse_permutations()

        lever_indices = np.arange(self.num_levers)
        self.red_arrow_lever = int(np.random.choice(lever_indices))
        remaining = np.delete(lever_indices, self.red_arrow_lever)
        self.high_payoff_lever = int(np.random.choice(remaining))

        self.lever_payoffs = np.ones(self.num_levers, dtype=float)
        self.lever_payoffs[self.high_payoff_lever] = 1.01

        self.rewards = []
        self.actions = {0: [], 1: []}
        self.states = []

        self._current_player = 0
        self.current_turn = 0
        self._terminated = False

    @property
    def base_game_description(self) -> str:
        return game_description.format(num_levers=self.num_levers)

    @property
    def example_text_observations(self) -> Dict[str, str]:
        return {
            "observation_template": observation,
        }

    @property
    def legal_actions(self) -> List[str]:
        return [str(i + 1) for i in range(self.num_levers)]

    def _validate_action(self, action: str) -> Tuple[bool, str]:
        try:
            action = int(action)
        except ValueError:
            return False, "Action must be an integer"

        if action < 1 or action > self.num_levers:
            return False, f"Action must be between 1 and {self.num_levers}"

        return True, ""

    @property
    def compact_rewards(self) -> List[float]:
        return [self.rewards[2 * turn + 1] for turn in range(self.current_turn)]

    def _lever_payoffs_for_player(self, player: int) -> List[float]:
        payoffs = [0.0 for _ in range(self.num_levers)]
        for lever in range(self.num_levers):
            permuted_label = self.permutations[player][lever]
            payoffs[permuted_label] = float(self.lever_payoffs[lever])
        return payoffs

    def _red_arrow_label_for_player(self, player: int) -> int:
        return int(self.permutations[player][self.red_arrow_lever]) + 1

    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:
        self._assert_valid_game_state()

        lever_payoffs = self._lever_payoffs_for_player(self.current_player)
        lever_payoffs_text = ", ".join(
            [f"{idx + 1}: {payoff:.2f}" for idx, payoff in enumerate(lever_payoffs)]
        )
        red_arrow_lever = self._red_arrow_label_for_player(self.current_player)

        text = observation.format(
            lever_payoffs_text=lever_payoffs_text,
            red_arrow_lever=red_arrow_lever,
        )

        return self.current_player, text, self.legal_actions

    @property
    def dict_observation(self) -> Tuple[int, Dict, List[str]]:
        self._assert_valid_game_state()

        your_actions = self.actions[self.current_player][:self.current_turn]
        other_actions = self.actions[1 - self.current_player][:self.current_turn]

        your_actions_permuted = [self.permutations[self.current_player][action] + 1 for action in your_actions]
        other_actions_permuted = [self.permutations[self.current_player][action] + 1 for action in other_actions]

        dict_observation = {
            "current_turn": self.current_turn + 1,
            "max_turn": 1,
            "num_levers": self.num_levers,
            "red_arrow_lever": self._red_arrow_label_for_player(self.current_player),
            "lever_payoffs": self._lever_payoffs_for_player(self.current_player),
            "your_actions": your_actions_permuted,
            "other_actions": other_actions_permuted,
            "rewards": self.compact_rewards,
        }
        return self.current_player, dict_observation, self.legal_actions

    def _update_inverse_permutations(self):
        assert self.permutations is not None
        self.inverse_permutations = [np.argsort(perm) for perm in self.permutations]

    def llm_action_to_game_action(self, llm_action: int) -> int:
        return llm_action - 1

    def _assert_valid_game_state(self):
        assert self.current_turn is not None, "You must call reset() before calling step()."
        assert self.current_turn < 1, "You have already completed the only turn."

    @property
    def dict_observation_description(self) -> str:
        return dict_observation_description

    @property
    def possible_actions_description(self) -> str:
        return possible_actions_description.format(num_levers=self.num_levers)

    @property
    def truncated(self) -> bool:
        return False

    @property
    def terminated(self) -> bool:
        return self._terminated

    def step(self, action: str):
        self._assert_valid_game_state()

        valid, error_msg = self._validate_action(action)
        if not valid:
            raise ValueError(error_msg)

        action = self.llm_action_to_game_action(int(action))
        action_unpermuted = self.inverse_permutations[self.current_player][action]
        self.actions[self.current_player].append(action_unpermuted)
        self.states.append(self.current_turn)

        reward = 0.0
        if len(self.actions[0]) == len(self.actions[1]):
            self.current_turn += 1
            if self.actions[0][-1] == self.actions[1][-1]:
                reward = float(self.lever_payoffs[self.actions[0][-1]])

        self.rewards.append(reward)
        self._current_player = 1 - self._current_player

        if self.current_turn == 1:
            self._terminated = True

        return reward, self._terminated, False, {}

    def get_compact_rollout_history(self, ignore_not_done: bool = False) -> CompactRolloutHistory:
        if not self.done and not ignore_not_done:
            raise ValueError("Please finish the game before calling get_compact_rollout_history()!")

        compact_states = []
        for turn in range(self.current_turn):
            compact_states.append({
                "p0_action": int(self.actions[0][turn]),
                "p1_action": int(self.actions[1][turn]),
                "reward": float(self.rewards[2 * turn + 1]),
            })

        return CompactRolloutHistory(
            hyperparams={
                "num_levers": self.num_levers,
                "lever_payoffs": [float(x) for x in self.lever_payoffs],
                "red_arrow_lever": int(self.red_arrow_lever),
                "high_payoff_lever": int(self.high_payoff_lever),
            },
            metadata={
                "permutation_0": [int(x) for x in self.permutations[0]],
                "permutation_1": [int(x) for x in self.permutations[1]],
                "turn": self.current_turn,
            },
            compact_states=compact_states,
            rewards_sum=sum(self.rewards),
            terminated=self._terminated,
            truncated=False,
        )

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def env_name(self) -> str:
        return "SelectionWithSemantics-v0"
