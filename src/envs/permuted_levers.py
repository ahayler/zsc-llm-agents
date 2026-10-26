import numpy as np
from typing import Optional, List, Tuple, Dict
from core import CompactRolloutHistory
from configs.policy_modifiers import BasePolicyModifierConfig
from envs.base import Env

game_description = """
Assume that there are {num_levers} levers labelled 1 to {num_levers}. You are playing with one other player. 
Both of you observe the same set of levers, but under a random permutation that is sampled at the start of the game and then stays fixed for the rest of the game.
For example in a game with 3 levers, your lever 1 could be lever 2 for the other player, your lever 2 is lever 3 for the other player and your lever 3 is lever 1 for the other player. While the permutation is fixed for the length of one game, it is unknown to each of you.
In each of {num_turns} turns, each of you have to choose a lever (at the same time), and if you chose the same lever you get a reward of 1, else you get a reward of 0. 
In between turns, both players can see the lever chosen by the other player, but you cannot communicate otherwise.
"""

observation = """
You are now in turn {current_turn} of {num_turns}.
In the last turn, you chose lever {your_action}, the other player chose lever {other_action}.
You got a (joined) reward of {reward}.
The total reward you have received so far is {total_reward}.
"""

dict_observation_description = """
An observation has the following format. Each list is ordered from the first turn to the last turn (i.e. at index -1 is the last action).
The levers are 1-indexed and both 'your_actions' and 'other_actions' are numbers as if observed from the perspective of the current player.
(e.g. if other_actions[-1] is 2, then the other player chose 'your' lever 2 in the last turn, in particular if your_actions[-1] == other_actions[-1], then you both chose the same lever in the last turn, i.e. rewards[-1] == 1.)
{{
    "current_turn": int # The current turn number (starts at 1)
    "max_turn": int # The number of total turns in the game
    "num_levers": int # The number of levers in the game
    "your_actions": List[int] # A list of previously picked levers by you
    "other_actions": List[int] # A list of previously picked levers by the other player
    "rewards": List[int] # A list of rewards you have received in the last turns
}}
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

# history_prompt = """
# We are now in round {current_round} of {rounds}.

# In the previous rounds, you chose the following levers:

# {rounds_info}
# """

# game_play_prompt = """
# {game_description}
# {history}

# You will now choose a lever for the current round. You must choose a number between 1 and {levers}. 
# Your choice should maximize your rewards over the coming rounds. Consider the history of the game so far to understand the other player's behavior.

# Respond in the following format:

# THOUGHT:
# <THOUGHT>

# ACTION JSON:
# ```json
# <JSON>
# ```

# In <THOUGHT>, provide a brief explanation of your reasoning behind the action you are about to take in this round. Consider the history and structure of the game.

# In <JSON>, provide the action you are going to take in this round. The JSON should have the following format:
# - "action": an integer between 1 and {levers} representing the lever you are going to choose in this round.

# This JSON will be automatically parsed, so ensure the format is precise.
# """


class PermutedLevers(Env):
    def __init__(self, policy_modifier_config: BasePolicyModifierConfig, num_turns: int, num_levers: int):
        super().__init__(policy_modifier_config)
        
        assert num_turns > 0, "Number of turns must be greater than 0"
        assert num_levers > 0, "Number of levers must be greater than 0"

        self.num_turns = num_turns
        self.num_levers = num_levers

        self.current_turn = None
        self._current_player = None
        self.permutations = None
        self.inverse_permutations = None

        # Keeps track of the player's actions and rewards in previous turns
        self.rewards = None
        self.actions = None
        self.states = None
 

    def reset(self, seed: Optional[int] = None, no_permutation: bool = False):
        self.current_turn = 0

        if seed is not None:
            np.random.seed(seed)

        # Here we sample the automorphism for each player
        if no_permutation:
            self.permutations = [np.arange(self.num_levers) for _ in range(2)]
        else:
            self.permutations = [np.random.permutation(self.num_levers) for _ in range(2)]

        self._update_inverse_permutations()

        self.rewards = []
        self.actions = {0: [], 1: []}
        self.states = []

        self._terminated = False

        self.last_lever_choice = None
        self._current_player = 0
        self.current_turn = 0

    @property
    def base_game_description(self) -> str:
        text = game_description.format(num_levers=self.num_levers, num_turns=self.num_turns)

        return text
    
    @property
    def example_text_observations(self) -> Dict[str, str]:
        return {
            "observation_template": observation,
        }
    
    @property
    def legal_actions(self) -> List[str]:
        return list([f"{str(i+1)}" for i in range(self.num_levers)])
    
    def _validate_action(self, action: str) -> Tuple[bool, str]:
        try:
            action = int(action)
        except ValueError:
            return False, "Action must be an integer"
        
        if action < 1 or action > self.num_levers:
            return False, f"Action must be between 1 and {self.num_levers}"
        
        return True, ""
    
    def get_compact_rollout_history(self, ignore_not_done: bool = False) -> CompactRolloutHistory:
        if not self.done and not ignore_not_done:
            raise ValueError("Please finish the game before calling get_compact_rollout_history()!")
        
        compact_states = []
        for turn in range(self.current_turn):
            compact_states.append({
                "p0_action": int(self.actions[0][turn]),
                "p1_action": int(self.actions[1][turn]),
                "reward": int(self.rewards[2*turn + 1]),
            })

        return CompactRolloutHistory(
            hyperparams={
                "num_levers": self.num_levers,
                "num_turns": self.num_turns,
            },
            metadata={
                "permutation_0": [int(x) for x in self.permutations[0]], # Casting ensure that we can dump to json
                "permutation_1": [int(x) for x in self.permutations[1]],
                "turn": self.current_turn,
            },
            compact_states=compact_states,
            rewards_sum=sum(self.rewards),
            terminated=self._terminated,
            truncated=False,
        )
    
    @property
    def compact_rewards(self) -> List[int]:
        return [self.rewards[2*turn+1] for turn in range(self.current_turn)]


    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:
        self._assert_valid_game_state()

        if self.current_turn == 0:
            return self.current_player, "You are in the first turn.", self.legal_actions
        
        your_action = self.actions[self.current_player][self.current_turn-1]
        other_action = self.actions[1-self.current_player][self.current_turn-1]

        # We permute with the current player's permutation and add 1 to convert to 1-indexed
        your_action_permuted = self.permutations[self.current_player][your_action] + 1
        other_action_permuted = self.permutations[self.current_player][other_action] + 1

        reward = self.rewards[2*self.current_turn-1]
        total_reward = sum(self.rewards)

        text = observation.format(
            current_turn=self.current_turn+1, 
            num_turns=self.num_turns, 
            your_action=your_action_permuted, 
            other_action=other_action_permuted, 
            reward=reward, 
            total_reward=total_reward
            )

        return self.current_player, text, self.legal_actions
    
    @property
    def dict_observation(self) -> Tuple[int, Dict, List[str]]:
        self._assert_valid_game_state()

        your_actions = self.actions[self.current_player][:self.current_turn]
        other_actions = self.actions[1-self.current_player][:self.current_turn]

        your_actions_permuted = [self.permutations[self.current_player][action] + 1 for action in your_actions]
        other_actions_permuted = [self.permutations[self.current_player][action] + 1 for action in other_actions]

        dict_observation = {
            "current_turn": self.current_turn+1,
            "max_turn": self.num_turns,
            "num_levers": self.num_levers,
            "your_actions": your_actions_permuted,
            "other_actions": other_actions_permuted,
            "rewards": self.compact_rewards,
        }
        return self.current_player, dict_observation, self.legal_actions
        

    def _update_inverse_permutations(self):
        # Dedicated function to make it easy to set permutations by hand
        assert self.permutations is not None

        self.inverse_permutations = [np.argsort(perm) for perm in self.permutations]
    
    def llm_action_to_game_action(self, llm_action: int):
        # Convert the action from the LLM to the game action
        return llm_action - 1
    
    def _assert_valid_game_state(self):
        assert self.current_turn is not None, "You must call reset() before calling step()."
        assert self.current_turn < self.num_turns, "You have already completed all turns."

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

        # We 0-index the actions to simplify dealing with the permutations
        action = self.llm_action_to_game_action(int(action))
        action_unpermuted = self.inverse_permutations[self.current_player][action]
        self.actions[self.current_player].append(action_unpermuted)
        self.states.append(self.current_turn)

        reward = 0
        # Check if both players have picked a lever in the current turn
        if len(self.actions[0]) == len(self.actions[1]):
            self.current_turn += 1
            if (self.actions[0][-1] == self.actions[1][-1]):
                reward = 1

        self.rewards.append(reward)
        self._current_player = 1 - self._current_player

        if self.current_turn == self.num_turns:
            self._terminated = True

        return reward, self._terminated, False, {}
    
    @property
    def current_player(self) -> int:
        return self._current_player
    
    @property
    def env_name(self) -> str:
        return "PermutedLevers-v0"