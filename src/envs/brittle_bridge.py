from core import CompactRolloutHistory
from envs.base import Env
from typing import List, Tuple, Dict
from configs.policy_modifiers import BasePolicyModifierConfig

game_description = """
You are playing a game of brittle bridge with one other player. 
The goal of the game is for both players to cross the bridge in as little turns as possible.
The bridge is {bridge_length} units long. In the game there are the following positions:

- 0: start
- 1, 2, ..., {last_bridge_index}: bridge_1, bridge_2, ..., bridge_{last_bridge_index}
- {goal_index}: goal

In each turn, both players **simultaneously** choose one of the following actions:

- Move forward by 1 (new position = current position + 1)
- Move backward by 1 (new position = current position - 1)
- Stay still (new position = current position)

The players cannot move past the start or goal positions. If a player tries to do that, they will stay still.
Note that the bridge is brittle and therefore can only support one player for a prolonged period of time.
In particular, if both players can stay only on the bridge for {bridge_safe_threshold} consecutive turn(s) or else the bridge will break.
Note that it doesn't matter which positions on the bridge the players are on, only that they are on the bridge.

An example (here we assume that the bridge can only support both player for 1 consecutive turn):
The bridge won't break if: both players are on the bridge for 1 turn, 1 player is on the bridge in turn 2 and 2 players are on the bridge in turn 3.
The bridge will break if: both players are on the bridge in turn 1 and 2.

If the bridge breaks, both players will fall into the water, the game ends and you receive a joint reward of -1000.
Else in each turn (until both players have reached the goal), you receive a (joint) reward of -1.

You cannot communicate with the other player, but you can observe each other's **past** actions, but not current action.
"""

observation = """
In the last turn, you chose action {your_action} and moved from position {your_position} to position {your_new_position}.
The other player chose action {other_action} and moved from position {other_position} to position {other_new_position}.
You got a (joined) reward of {reward}.
The total reward you have received so far is {total_reward}.
"""

initial_observation = """
This is the first turn of the game. Both players are at position 0 (start).
"""

dict_observation_description = """
Note that all lists are in the order of turns (i.e. the item at index -1 is the most recent turn).
{
    "your_actions": ["stay", "forward"], # A list of your past actions
    "your_positions": [0, 0, 1], # A list of your past positions
    "other_actions": ["stay", "forward"], # A list of the other player's past actions
    "other_positions": [0, 0, 1], # A list of the other player's past positions
    "rewards": [-1, -1], # A list of your past rewards
    "total_reward": -2, # The total (joint) reward you have received so far
}
"""

possible_actions_description = """
The possible actions for both players are:
- "forward"
- "backward"
- "stay"
"""



class BrittleBridge(Env):
    def __init__(self, policy_modifier_config: BasePolicyModifierConfig, bridge_length: int, bridge_break_threshold: int = 2, max_turns: int = 30):
        super().__init__(policy_modifier_config)
        
        self.bridge_length = bridge_length
        self.bridge_break_threshold = bridge_break_threshold

        self.current_turn = None
        self._current_player = None

        # Keeps track of the player's actions and rewards in previous turns
        self.rewards = None
        self.actions = None
        self.states = None

        self.max_turns = max_turns

    def reset(self):
        self.current_turn = 0
        self._current_player = 0
        self.rewards = []
        self.actions = {0: [], 1: []}
        self.states = {0: [0], 1: [0]}

        self._terminated = False
        self._truncated = False
        
        self.turns_both_on_bridge = 0

        self.position_to_index = {
            0: "start",
            **{i: f"bridge_{i}" for i in range(1, self.bridge_length + 1)},
            self.bridge_length + 1: "goal"
        }

        self.goal_index = self.bridge_length + 1

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def example_text_observations(self) -> Dict[str, str]:
        return {
            "observation_template": observation,
        }

    @property
    def base_game_description(self) -> str:
        text = game_description.format(
            bridge_length=self.bridge_length, 
            last_bridge_index=self.bridge_length, 
            goal_index=self.bridge_length + 1,
            bridge_safe_threshold=self.bridge_break_threshold - 1
        )
        return text
    
    @property
    def legal_actions(self) -> List[str]:
        return ["forward", "backward", "stay"]
    
    def get_new_position(self, position: int, action: str) -> int:
        if action == "forward":
            return min(position + 1, self.goal_index)
        elif action == "backward":
            return max(position - 1, 0)
        else:
            return position
        
    def player_on_bridge(self, player_id: int) -> bool:
        return self.states[player_id][-1] > 0 and self.states[player_id][-1] < self.goal_index


    def step(self, action: str):
        self.assert_not_done()

        if action not in self.legal_actions:
            raise ValueError(f"Invalid action: {action}")
        self.actions[self._current_player].append(action)

        new_position = self.get_new_position(self.states[self._current_player][-1], action)
        self.states[self._current_player].append(new_position)

        # Did both players move?
        if len(self.actions[0]) == len(self.actions[1]):
            # Both players on the bridge?
            if self.player_on_bridge(0) and self.player_on_bridge(1):
                self.turns_both_on_bridge += 1
            else:
                self.turns_both_on_bridge = 0

            if self.turns_both_on_bridge >= self.bridge_break_threshold:
                reward = -1000
                self._terminated = True
            else:
                reward = -1

            # Did they both reach the goal?
            if self.states[0][-1] == self.goal_index and self.states[1][-1] == self.goal_index:
                reward = 0
                self._terminated = True

            self.current_turn += 1

            if self.current_turn >= self.max_turns:
                self._truncated = True
        else:
            # Only one player moved yet
            reward = 0

        self.rewards.append(reward)

        # Switch to the other player
        self._current_player = 1 - self._current_player

        return reward, self._terminated, self._truncated, {}
    
    def _format_position(self, position: int) -> str:
        return f"{position} ({self.position_to_index[position]})"
    
    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:
        current_player = self._current_player
        other_player = 1 - current_player


        if len(self.actions[current_player]) == 0:
            obs = initial_observation
        else:
            
            # Make sure that get the observations from the previous turn (the first player may have already moved)
            your_action = self.actions[current_player][self.current_turn - 1]
            your_position = self.states[current_player][self.current_turn - 1]
            your_new_position = self.states[current_player][self.current_turn]
            other_action = self.actions[other_player][self.current_turn - 1]
            other_position = self.states[other_player][self.current_turn - 1]
            other_new_position = self.states[other_player][self.current_turn]

            obs = observation.format(
                your_action=your_action,
                your_position=self._format_position(your_position),
                your_new_position=self._format_position(your_new_position),
                other_action=other_action,
                other_position=self._format_position(other_position),
                other_new_position=self._format_position(other_new_position),
                reward=self.rewards[-1],
                total_reward=sum(self.rewards),
            )

        return self.current_player, obs, self.legal_actions
    
    @property
    def dict_observation(self) -> Tuple[int, Dict, List[str]]:
        current_player = self._current_player

        obs_dict = {
            "your_actions": self.actions[current_player][:self.current_turn],
            "your_positions": self.states[current_player][:self.current_turn+1],
            "other_actions": self.actions[1 - current_player][:self.current_turn],
            "other_positions": self.states[1 - current_player][:self.current_turn+1],
            "rewards": self.rewards.copy(),
            "total_reward": sum(self.rewards),
        }

        return current_player, obs_dict, self.legal_actions
    
    @property
    def dict_observation_description(self) -> str:
        return dict_observation_description
    
    @property
    def possible_actions_description(self) -> str:
        return possible_actions_description
    
    def get_compact_rollout_history(self, ignore_not_done: bool = False) -> CompactRolloutHistory:
        if not self.done and not ignore_not_done:
            raise ValueError("Environment is not done yet!")
        
        def compact_state(turn: int) -> Dict:
            return {
                "turn": turn,
                "position_p0": self._format_position(self.states[0][turn]),
                "position_p1": self._format_position(self.states[1][turn]),
                "last_reward": self.rewards[(2*turn - 1)] if turn > 0 else None,
                "last_action_p0": self.actions[0][turn - 1] if turn > 0 else None,
                "last_action_p1": self.actions[1][turn - 1] if turn > 0 else None,
            }
        
        compact_states = [compact_state(turn) for turn in range(self.current_turn+1)]

        hyperparams = {
            "bridge_length": self.bridge_length,
            "bridge_break_threshold": self.bridge_break_threshold,
            "max_turns": self.max_turns,
        }

        metadata = {
            "total_turns": self.current_turn,
        }

        return CompactRolloutHistory(
            hyperparams=hyperparams,
            metadata=metadata,
            rewards_sum=sum(self.rewards),
            terminated=self.terminated,
            truncated=self.truncated,
            compact_states=compact_states,
        )
    
    @property
    def current_player(self) -> int:
        return self._current_player
    
    @property
    def env_name(self) -> str:
        return "BrittleBridge-v0"
