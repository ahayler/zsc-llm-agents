from envs.base import Env
from core import CompactRolloutHistory
from typing import Literal, Optional, Tuple, Dict, List
import numpy as np
from configs.policy_modifiers import BasePolicyModifierConfig

traje_di_matrix = np.array([
    [1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
    [1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
    [0,    0,    1, 0.02,    0,    0,    0,    0,    0,    0,    0],
    [0,    0, 0.02,    1, 0.02,    0,    0,    0,    0,    0,    0],
    [0,    0,    0, 0.02,    1, 0.02,    0,    0,    0,    0,    0],
    [0,    0,    0,    0, 0.02,    1, 0.02,    0,    0,    0,    0],
    [0,    0,    0,    0,    0, 0.02,    1, 0.02,    0,    0,    0],
    [0,    0,    0,    0,    0,    0, 0.02,    1, 0.02,    0,    0],
    [0,    0,    0,    0,    0,    0,    0, 0.02,    1, 0.02,    0],
    [0,    0,    0,    0,    0,    0,    0,    0, 0.02,    1, 0.02],
    [0,    0,    0,    0,    0,    0,    0,    0,    0, 0.02,    1]
])

dict_observation_description = """
The observation is a dictionary with the following keys:
- "matrix": np.array; The game matrix (fixed; also provided in the game_description)
"""

observation_alice = """You are Alice (player_0). Please pick a row index."""
observation_bob = """You are Bob (player_1). Please pick a column index."""

game_description = """
You are playing a matrix game with two players. The game is played for one turn.
There are two players, Alice (player_0) and Bob (player_1).

The game matrix has {num_rows} rows and {num_columns} columns, each indexed starting from 0.
The exact form of the game matrix is:
{matrix_str}

The game is played as follows:
- Alice (player_0) can determine the row index and Bob (player_1) can determine the column index.
- Alice and Bob choose their actions simultaneously.
- The joint reward is the value in the game matrix at the row index of Alice's action and the column index of Bob's action.
"""

possible_actions_description = """
The possible actions for Alice (player_0) are:
{legal_column_actions_str}

The possible actions for Bob (player_1) are:
{legal_row_actions_str}
"""

class MatrixGame(Env):
    def __init__(self, policy_modifier_config: BasePolicyModifierConfig, template_name: Literal["TrajeDi"], permutation_seed: Optional[int] = None):
        super().__init__(policy_modifier_config)

        self.template_name = template_name
        self.permutation_seed = permutation_seed

        if self.template_name == "TrajeDi":
            self.matrix = traje_di_matrix
        else:
            raise ValueError(f"Template name {template_name} not supported")

        if self.permutation_seed is not None:
            # We can DETERMINISTICALLY permute the rows and columns of the matrix (This is important as every seed in XP matrix has to have the same permuted matrix)
            np.random.seed(permutation_seed)
            self.permutation = np.random.permutation(self.matrix.shape[0])
            self.matrix = self.matrix[self.permutation]
            self.permutation = np.random.permutation(self.matrix.shape[1])
            self.matrix = self.matrix[:, self.permutation]

        self._current_player = None
        self.action_alice = None
        self.action_bob = None

    def reset(self, seed: Optional[int] = None) -> None:
        self._current_player = 0
        self._terminated = False
        self.action_alice = None
        self.action_bob = None
        self.reward = None

    @property
    def truncated(self) -> bool:
        return False
    
    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def base_game_description(self) -> str:
        return game_description.format(num_rows=self.matrix.shape[0], num_columns=self.matrix.shape[1], matrix_str=np.array2string(self.matrix))

    @property
    def legal_actions(self) -> List[str]:
        return [str(i) for i in range(self.matrix.shape[self.current_player])]
    
    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def dict_observation(self) -> Tuple[int, Dict, List[str]]:
        obs_dict = {
            "matrix": self.matrix,
        }
        return self.current_player, obs_dict, self.legal_actions

    @property
    def dict_observation_description(self) -> str:
        return dict_observation_description

    @property
    def possible_actions_description(self) -> str:
        return possible_actions_description.format(legal_column_actions_str=str(list(range(self.matrix.shape[0]))), legal_row_actions_str=str(list(range(self.matrix.shape[1]))))


    def step(self, action: str) -> Tuple[int, float, bool, Dict]:
        if action not in self.legal_actions:
            raise ValueError(f"Invalid action: {action}")
        
        if self.current_player == 0:
            self.action_alice = int(action)
            self._current_player = 1
            reward = 0
        else:
            self.action_bob = int(action)
            self._terminated = True
            reward = self.matrix[self.action_alice, self.action_bob]
            self.reward = reward

        return reward, self.terminated, self.truncated, {}

    
    def get_compact_rollout_history(self, ignore_not_done: bool = False) -> CompactRolloutHistory:
        """
        Get a compact rollout history of the environment.
        """
        if not self.done and not ignore_not_done:
            raise ValueError("Please finish the game before calling get_compact_rollout_history()!")

        compact_rollout_history = CompactRolloutHistory(
            hyperparams={
                "permutation_seed": self.permutation_seed,
                "matrix": self.matrix.tolist(),
            },
            metadata={
                "action_alice": self.action_alice,
                "action_bob": self.action_bob,
            },
            rewards_sum=self.reward,
            compact_states=[],
            truncated=self.truncated,
            terminated=self.terminated,
        )
        return compact_rollout_history

    @property
    def env_name(self) -> str:
        return f"matrix_game_{self.template_name}"

    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:
        if self.current_player == 0:
            return self.current_player, observation_alice, self.legal_actions
        else:
            return self.current_player, observation_bob, self.legal_actions

    @property
    def example_text_observations(self) -> Dict[str, str]:
        return {
            "observation_alice": observation_alice,
            "observation_bob": observation_bob,
        }