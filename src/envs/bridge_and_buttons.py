from envs.base import Env
from typing import Tuple, List
from configs.policy_modifiers import BasePolicyModifierConfig

# Note that the button countdown is handled before the movement
GAME_DESCRIPTION = """
You are now playing a round of Bridge and Buttons. The (joint) goal of this game is for both players to reach the other side of the bridge in as little turns as possible.
The bridge is {bridge_length} units long.

Therefore we have the following states:
{states_string}

In each turn, both players choose an action simultaneously without communication.
You will be able to see each other's previous actions.
Sadly, this is not a normal bridge. The bridge will only be up and therefore safe to cross if the button countdown is larger than 0.
On each side of the bridge, there is a button (i.e. at 0: start, at {last_index}: goal). 
Each turn you will have the choice to move forward (i -> i+1), backward (i -> i-1), stand_still (i -> i) or press_button (if you are standing on a tile with a button).

After both of you have locked in your actions, the turn is proceeds in two phases:

Phase 1 (Buttons):
- If a button was pressed, the button countdown is reset to {button_duration}.
- Else the button countdown is decremented by 1 (until reaching 0).

Phase 2 (Movement):
- If the button countdown is 0 and anyone was on the bridge in the previous turn or tries to move onto the bridge, he will fall into the water and die. The game ends and you receive a joint reward of -1000.
- If the button countdown is larger than 0, you can move normally.

Each turn that passes without falling into the water, you receive a joint reward of -1. Your goal is to maximize your joint reward.
"""

class BridgeAndButtons(Env):
    def __init__(self, policy_modifier_config: BasePolicyModifierConfig, bridge_length: int, button_duration: int, max_turns: int = 30):
        super().__init__(policy_modifier_config)
        
        self.bridge_length = bridge_length
        self.board = ['start'] + ['bridge'] * self.bridge_length + ['goal']
        self.max_turns = max_turns

        self._current_player = None
        self.rewards = None
        self.button_duration = button_duration
        self.action_history = None
        self.current_action_profile = None
        self.current_turn = None

    def reset(self):
        self.pos_0 = 0
        self.pos_1 = 0

        self._current_player = 0
        self.button_countdown = 0

        self.action_history = []
        self.current_action_profile = []
        self.rewards = []
        self.current_turn = 0

    @property
    def _current_player_pos(self):
        return self.pos_0 if self._current_player == 0 else self.pos_1
    
    def board_to_text(self, add_players: bool = False):
        board = self.board.copy()
        if add_players:
            board[self.pos_0] += ' (player 0)'
            board[self.pos_1] += ' (player 1)'
        return "\n".join([f"{i}: {board[i]}" for i in range(len(board))])

    @property
    def legal_actions(self):
        if self._current_player_pos == len(self.board) - 1 or self._current_player_pos == 0:
            return ['press_button', 'forward', 'backward', 'stand_still']
        else:
            return ['forward', 'backward', 'stand_still']
        
    def _handle_actions(self, action0: str, action1: str) -> Tuple[int, bool]:
        def on_bridge(pos: int):
            return pos > 0 and pos < len(self.board) - 1
        
        def proposed_pos(pos: int, action: str):
            if action == 'forward':
                return min(pos + 1, len(self.board) - 1)
            elif action == 'backward':
                return max(pos - 1, 0)
            else:
                return pos

        # Handle button presses
        if action0 == 'press_button' or action1 == 'press_button':
            self.button_countdown = self.button_duration
        else:
            self.button_countdown = max(0, self.button_countdown - 1)
        
        # Handle movement
        pos_0_proposed = proposed_pos(self.pos_0, action0)
        pos_1_proposed = proposed_pos(self.pos_1, action1)

        # Check if anybody fell into the water
        if self.button_countdown == 0:
            if on_bridge(pos_0_proposed) or on_bridge(pos_1_proposed) or on_bridge(self.pos_0) or on_bridge(self.pos_1):
                return -1000, True
        
        self.pos_0 = pos_0_proposed
        self.pos_1 = pos_1_proposed

        # Check if both players reached the goal
        if self.pos_0 == len(self.board) - 1 and self.pos_1 == len(self.board) - 1:
            return 0, True
        
        return -1, False
        

    def step(self, action: str):
        if action not in self.legal_actions:
            raise ValueError(f"Invalid action: {action}")
        
        self.current_action_profile.append(action)

        terminated = False
        reward = 0

        if len(self.current_action_profile) == 2:
            reward, terminated = self._handle_actions(self.current_action_profile[0], self.current_action_profile[1])
            self.action_history.append(self.current_action_profile)
            self.current_action_profile = []

            self.current_turn += 1

        self._current_player = 1 - self._current_player

        truncated = self.current_turn >= self.max_turns
        self.rewards.append(reward)

        return reward, terminated, truncated, {}
    
    @property
    def text_observation(self) -> Tuple[0, str, List[str]]:

        # Figure out the current board state
        if self.current_player == 0:
            my_pos = self.pos_0
            other_pos = self.pos_1
        else:
            my_pos = self.pos_1
            other_pos = self.pos_0

        obs = f"You are currently at position {my_pos}: {self.board[my_pos]} and the other player is at position {other_pos}: {self.board[other_pos]}.\n"
        obs += f"The button countdown is currently at {self.button_countdown}.\n"

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
    def base_game_description(self):
        return GAME_DESCRIPTION.format(
            bridge_length=self.bridge_length, 
            button_duration=self.button_duration, 
            last_index=len(self.board) - 1,
            states_string=self.board_to_text()
        )
            
    @property
    def current_player(self):
        return self._current_player
    
    @property
    def env_name(self):
        return "Bridge&Buttons-v0"