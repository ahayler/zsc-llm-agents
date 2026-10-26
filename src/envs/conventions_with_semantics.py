import numpy as np
from typing import Optional, Tuple, Dict, List
from core import CompactRolloutHistory
from configs.policy_modifiers import BasePolicyModifierConfig
from envs.base import Env

payoff_matrix = np.array([
    [1.0, 0.0, 1.01, 0.0],
    [1.01, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.01],
    [0.0, 1.01, 0.0, 1.0],
])

row_labels = ["A", "B", "C", "D"]
column_labels = ["1", "2", "3", "4"]

game_description = """
You are playing Conventions with Semantics for one round with two players, Alice and Bob.

The payoff matrix is:
{matrix_str}

At the start of the round, one column is randomly selected as the active column.
Alice is shown the active column, but Bob is not.
Alice chooses a row signal (A, B, C, or D). Bob observes Alice's signal and chooses a column action (1, 2, 3, or 4).

If Bob chooses the active column, you receive the matrix payoff for Alice's row and Bob's column.
Otherwise, the payoff is 0.
"""

observation_alice = """
You are Alice (player_0). The active column is {active_column}.
Please pick a row signal: A, B, C, or D.
"""

observation_bob = """
You are Bob (player_1). Alice chose row signal {row_signal}.
Please pick a column action: 1, 2, 3, or 4.
"""

dict_observation_description = """
For player_0 (Alice), the observation dict has the following format:
{
    "active_column": int # The active column label (1-4)
}

For player_1 (Bob), the observation dict has the following format:
{
    "row_signal": str # The row signal chosen by Alice (A-D)
}
"""

possible_actions_description = """
The possible actions for Alice (player_0) are:
- "A"
- "B"
- "C"
- "D"

The possible actions for Bob (player_1) are:
- "1"
- "2"
- "3"
- "4"
"""


class ConventionsWithSemantics(Env):
    def __init__(self, policy_modifier_config: BasePolicyModifierConfig):
        super().__init__(policy_modifier_config)

        self._current_player = None
        self._terminated = None
        self.active_column = None
        self.action_alice = None
        self.action_bob = None
        self.rewards = None

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

        self.active_column = int(np.random.randint(0, len(column_labels)))
        self.action_alice = None
        self.action_bob = None
        self.rewards = []
        self._current_player = 0
        self._terminated = False

    @property
    def base_game_description(self) -> str:
        return game_description.format(matrix_str=np.array2string(payoff_matrix))

    @property
    def legal_actions(self) -> List[str]:
        if self.current_player == 0:
            return row_labels
        return column_labels

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def dict_observation(self) -> Tuple[int, Dict, List[str]]:
        if self.current_player == 0:
            obs = {
                "active_column": self.active_column + 1,
            }
        else:
            obs = {
                "row_signal": row_labels[self.action_alice],
            }
        return self.current_player, obs, self.legal_actions

    @property
    def dict_observation_description(self) -> str:
        return dict_observation_description

    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:
        if self.current_player == 0:
            obs = observation_alice.format(active_column=self.active_column + 1)
        else:
            obs = observation_bob.format(row_signal=row_labels[self.action_alice])
        return self.current_player, obs, self.legal_actions

    @property
    def example_text_observations(self) -> Dict[str, str]:
        return {
            "observation_player_0": observation_alice,
            "observation_player_1": observation_bob,
        }

    @property
    def possible_actions_description(self) -> str:
        return possible_actions_description

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return False

    def step(self, action: str) -> Tuple[float, bool, bool, Dict]:
        if action not in self.legal_actions:
            raise ValueError(f"Invalid action: {action}")

        if self.current_player == 0:
            self.action_alice = row_labels.index(action)
            self._current_player = 1
            reward = 0.0
        else:
            self.action_bob = column_labels.index(action)
            self._terminated = True
            if self.action_bob == self.active_column:
                reward = float(payoff_matrix[self.action_alice, self.action_bob])
            else:
                reward = 0.0

        self.rewards.append(reward)
        return reward, self.terminated, self.truncated, {}

    def get_compact_rollout_history(self, ignore_not_done: bool = False) -> CompactRolloutHistory:
        if not self.done and not ignore_not_done:
            raise ValueError("Please finish the game before calling get_compact_rollout_history()!")

        return CompactRolloutHistory(
            hyperparams={
                "matrix": payoff_matrix.tolist(),
            },
            metadata={
                "active_column": int(self.active_column),
                "action_alice": self.action_alice,
                "action_bob": self.action_bob,
            },
            rewards_sum=sum(self.rewards),
            compact_states=[],
            terminated=self._terminated,
            truncated=False,
        )

    @property
    def env_name(self) -> str:
        return "ConventionsWithSemantics-v0"
