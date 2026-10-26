from core import CompactRolloutHistory
from envs.base import Env
from typing import List, Literal, Tuple, Dict
from configs.policy_modifiers import BasePolicyModifierConfig

GAME_DESCRIPTION = """
We are now playing a round of Narrow Tunnel.
In this game, there are two players: left player and right player.
The left player starts on the left side of a tunnel, while the right player starts on the right side.
It is your (joint) goal to get both players to the other side of the tunnel in as few turns as possible.
In particular, the player_0 (starting on the leftmost tile also called goal_left) needs to reach goal_right and player_1 (starting on the rightmost tile also called goal_right) needs to reach goal_left.
Each turn, in which this condition is not met, you receive a reward of -1.

The board has the following structure:
{board_text}

Here 0 is the leftmost tile and {max_index} is the rightmost tile.

In each turn, you can choose to move one tile to the left (i -> i-1) or right (i -> i+1), or stay still (i -> i).
Sadly, the tunnel has narrow passages. This puts the following constraints on your movement:
- Both players can't move onto a narrow tile in the same turn
- Players can't switch tiles with each other (i.e. they try to move onto the other ones previous tile) if one of them is on a narrow tile.
If the players try to do one of the two above, nothing will happen. Note that the goal tiles are not narrow. 

A few examples where NOTHING happens because of the above constraints:
- One player is left of a narrow tile and the other player is right of a narrow tile, both try to move onto the tile
- One player is on a narrow tile and the other player is on a wide tile to right of it. If the player on the wide tile moves right, 
then nothing happens except if the player on the narrow tile moves left (this also means that two players can never be on the same narrow tile). 
Note that the player on the narrow tile can move right, if the player on the wide tile does not try to move right (i.e onto the narrow tile).
In contrast: Both players can stand on or move into a wide tile (including the goal tiles) at the same time, without any issues (i.e. the movement will be executed).

**Role asignment:**
- player_0 is always start on the leftmost tile
- player_1 is always start on the rightmost tile
"""

dict_observation_description = """
Each observation has the following format:
{{
    "goal": Literal["left", "right"] # Whether you need to reach the left or right goal tile (if left is the goal, you will have started on the rightmost tile and if the right is the goal, you will have started on the leftmost tile)
    "tunnel": List[Literal["goal_left", "wide", "narrow", "goal_right"]] # The (fixed structure) of the tunnel. 
    "current_turn": int # The current turn number (starts at 1)
    "max_turn": int # The number of total turns in the game
    "your_positions": List[int] # A list of your positions in the tunnel (from oldest to newest; i.e. at index -1 is the most recent position)
    "other_positions": List[int] # A list of the other player's positions in the tunnel (from oldest to newest; i.e. at index -1 is the most recent position)
    "your_actions": List[Literal["LEFT", "RIGHT", "STAY"]] # A list of previously chosen actions by you (will empty on the first turn)
    "other_actions": List[Literal["LEFT", "RIGHT", "STAY"]] # A list of previously chosen actions by the other player (will empty on the first turn)
}}
"""

possible_actions_description = """
The possible actions for each player are:
- "LEFT"
- "RIGHT"
- "STAY"
"""

# TODO: Think about adding equivalent version where both players think they are the left or the right player
class NarrowTunnel(Env):
    def __init__(self, policy_modifier_config: BasePolicyModifierConfig, template: List[Literal["n", "w"]], max_turns: int = 20):
        super().__init__(policy_modifier_config)
        
        for tile in template:
            if tile not in ["n", "w"]:
                raise ValueError(f"Invalid tile: {tile}")

        self.board = self._template_to_board(template)
        self.max_turns = max_turns

        self.current_action_profile = None
        self._current_player = None
        self.current_turn = None
        self.action_history = None

        self.rewards = None

    def _board_to_text(self, add_players: bool = False) -> str:
        board = self.board.copy()

        if add_players:
            board[self.pos_0] += " (left player)"
            board[self.pos_1] += " (right player)"

        return "\n".join([f"{i}: {board[i]}" for i in range(len(board))])

    def _template_to_board(self, template: List[Literal["n", "w"]]) -> List[str]:
        template = ["wide" if s == "w" else "narrow" for s in template]

        board = ["goal_left"] + template + ["goal_right"]
        return board
    
    @property
    def terminated(self) -> bool:
        return self.pos_0 == len(self.board) - 1 and self.pos_1 == 0
    
    @property
    def truncated(self) -> bool:
        return self.current_turn >= self.max_turns
    
    @property
    def is_done(self) -> bool:
        return self.terminated or self.truncated
    @property
    def pos_0(self) -> int:
        return self.player_0_positions[-1]

    @property
    def pos_1(self) -> int:
        return self.player_1_positions[-1]

    def get_action_history(self, player: int) -> List[str]:
        return [self.action_history[i][player] for i in range(len(self.action_history))]
    
    def reset(self):
        self.current_action_profile = []
        self._current_player = 0

        self.player_0_positions = [0]
        self.player_1_positions = [len(self.board) - 1]

        self.action_history = []

        self.current_turn = 0
        self.rewards = []

    def _compute_proposed_position(self, current_position: int, action: str) -> int:
        if action == "RIGHT":
            return min(current_position + 1, len(self.board) - 1)
        elif action == "LEFT":
            return max(current_position - 1, 0)
        else:
            return current_position
        
    def move_players_to(self, pos_0: int, pos_1: int):
        self.player_0_positions.append(pos_0)
        self.player_1_positions.append(pos_1)

    def handle_movement(self, action_0: str, action_1: str):
        proposed_pos_0 = self._compute_proposed_position(self.pos_0, action_0)
        proposed_pos_1 = self._compute_proposed_position(self.pos_1, action_1)

        # You both can't move onto a narrow tile
        if proposed_pos_0 == proposed_pos_1 and self.board[proposed_pos_0] == "narrow":
            self.move_players_to(self.pos_0, self.pos_1)
            return
        # You can't switch places with the other player, if one of you is on a narrow tile
        elif proposed_pos_0 == self.pos_1  and proposed_pos_1 == self.pos_0 and (self.board[self.pos_0] == "narrow" or self.board[self.pos_1] == "narrow"):
            self.move_players_to(self.pos_0, self.pos_1)
            return
        else:
            self.move_players_to(proposed_pos_0, proposed_pos_1)
            return

    @property
    def dict_observation(self) -> Tuple[int, Dict, List[str]]:
        goal = "right" if self.current_player == 0 else "left"
        tunnel = self.board
        current_turn = self.current_turn
        max_turn = self.max_turns
        if self.current_player == 0:
            your_positions = self.player_0_positions
            other_positions = self.player_1_positions
            your_actions = self.get_action_history(0)
            other_actions = self.get_action_history(1)
        else:
            your_positions = self.player_1_positions
            other_positions = self.player_0_positions
            your_actions = self.get_action_history(1)
            other_actions = self.get_action_history(0)

        dict_observation = {
            "goal": goal,
            "tunnel": tunnel,
            "current_turn": current_turn,
            "max_turn": max_turn,
            "your_positions": your_positions,
            "other_positions": other_positions,
            "your_actions": your_actions,
            "other_actions": other_actions,
        }

        return self.current_player, dict_observation, self.legal_actions
    
    def get_compact_rollout_history(self, ignore_not_done: bool = False) -> CompactRolloutHistory:
        if not self.is_done and not ignore_not_done:
            raise ValueError("Please finish the game before creating a compact rollout history.")
        
        hyperparams = {
            "layout": self.board,
            "max_turns": self.max_turns,
        }

        metadata = {
            "turn": self.current_turn,
        }

        def get_compact_state(turn: int) -> Dict:
            return {
                "turn": turn,
                "pos_0": self.player_0_positions[turn],
                "pos_1": self.player_1_positions[turn],
                "previous_action_0": self.action_history[turn - 1][0] if turn > 0 else None,
                "previous_action_1": self.action_history[turn - 1][1] if turn > 0 else None,
                "tile_0": self.board[self.player_0_positions[turn]],
                "tile_1": self.board[self.player_1_positions[turn]],
            }

        compact_state_history = [
            get_compact_state(turn) for turn in range(len(self.action_history) + 1)
        ]

        compact_rollout_history = CompactRolloutHistory(
            hyperparams=hyperparams,
            metadata=metadata,
            terminated=self.terminated,
            truncated=self.truncated,
            compact_states=compact_state_history,
            rewards_sum=sum(self.rewards),
        )

        return compact_rollout_history
    
    @property
    def example_text_observations(self) -> Dict[str, str]:
        return {
            "template": """
            You are currently at position {my_pos}: {self.board[my_pos]} and the other player is at position {other_pos}: {self.board[other_pos]}.
            In the last turn, the other player chose the action {other_act} and you chose the action {my_act}.
            """
        }

    @property
    def text_observation(self) -> Tuple[0, str, List[str]]:

        # Figure out the current board state
        if self.current_player == 0:
            obs = "You are the left player.\n"
            my_pos = self.pos_0
            other_pos = self.pos_1
        else:
            obs = "You are the right player.\n"
            my_pos = self.pos_1
            other_pos = self.pos_0

        obs += f"You are currently at position {my_pos}: {self.board[my_pos]} and the other player is at position {other_pos}: {self.board[other_pos]}.\n"

        # Figure out the last actions
        if len(self.action_history) > 0:
            if self.current_player == 0:
                my_act = self.action_history[-1][0]
                other_act = self.action_history[-1][1]
            else:
                my_act = self.action_history[-1][1]
                other_act = self.action_history[-1][0]

            obs += f"In the last turn, the other player chose the action {other_act} and you chose the action {my_act}.\n"
        else:
            obs += "This is the first turn.\n"

        return self.current_player, obs, self.legal_actions
    
    @property
    def base_game_description(self) -> str:
        return GAME_DESCRIPTION.format(board_text=self._board_to_text(), max_index=len(self.board) - 1)

    @property
    def legal_actions(self) -> List[str]:
        return ["RIGHT", "LEFT", "STAY"]

    def step(self, action: str) -> Tuple[float, bool, bool, Dict]:
        if action not in self.legal_actions:
            raise ValueError(f"Invalid action: {action}")
        
        if self.is_done:
            raise ValueError("The game is already done.")
        
        self.current_action_profile.append(action)

        reward = 0

        if len(self.current_action_profile) == 2:
            # Actual turn
            self.handle_movement(self.current_action_profile[0], self.current_action_profile[1])
            self.action_history.append(self.current_action_profile)
            self.current_action_profile = []

            self.current_turn += 1

            if not self.terminated:
                reward = -1

        self._current_player = (self._current_player + 1) % 2
        self.rewards.append(reward)

        return reward, self.terminated, self.truncated, {}
    
    @property
    def dict_observation_description(self) -> str:
        return dict_observation_description
    
    @property
    def possible_actions_description(self) -> str:
        return possible_actions_description

    @property
    def current_player(self) -> int:
        return self._current_player
    
    @property
    def env_name(self) -> str:
        return "NarrowTunnel-v0"
    
