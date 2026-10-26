from envs.base import Env
import numpy as np
from typing import Optional, Tuple, List, Dict
from core import CompactRolloutHistory
from configs.policy_modifiers import BasePolicyModifierConfig

game_description = """
You are playing an iterated guessing game with one other player. One of you is Alice and the other is Bob.
The game is played for {num_rounds} rounds. There are {num_levers} levers, labeled {lever_labels}, 
{num_buttons} buttons labeled {button_labels} and {num_signals} signals labeled {signal_labels}.
You know that there is a fixed (across all rounds) but unknown 1:1 mapping (bijection) between the buttons and the signals 
(i.e. if Alice presses a button, exactly one signal is shown to Bob and each signal has exactly one button associated with it).
Alice does not observe which signal lights up on Bob's side when she presses a button and Bob does not observe which button Alice pressed.
Your goal is to maximize your (joint) reward at the end of the game (i.e. after all {num_rounds} rounds have been played).

Each round works as follows:
First it is Alice's turn. Alice is shown the correct lever for the current round (the correct lever for each round is chosen randomly). 
Alice can do exactly one of the following:
- bail the round for a (joint) reward of 1 (this means that the current round is skipped and the players receive a joint reward of 1)
- Alice can reval the correct lever for the current round to Bob, which costs 5 joint reward (i.e. -5 joint reward)
- Alice can press one of the {num_buttons} buttons, which will show the corresponding signal to Bob. This is free.

If Alice didn't bail, it is Bob's turn. Bob observes the signal corresponding to the button pressed by Alice or the revealed lever.
Bob can now do exactly one of the following:
- bail the round for a (joint) reward of 0.5
- pick one of the {num_levers} levers. 
    - If Bob picks the correct lever, they receive a joint reward of 10
    - If Bob picks the wrong lever, they receive a joint reward of -10

The players cannot communicate with each other except the options presented above (i.e. revealing the correct lever or pressing a button).
"""

# **Role asignment:**

# - player 0 is alice
# - player 1 is bob

observation_alice = """
You are Alice. The correct lever for the current round is {correct_lever}.
"""

# Rest will be handeled in code
observation_bob = """
You are Bob.
{signal_information}
{reval_information}
"""

dict_observation_description = """
For player_0 (Alice), the observation dict has the following format:
{{
    "correct_levers": list[str] # The list of the lever labels shown to Alice across time (at index -1 is the current round)
    "picked_levers": list[str] # The list of levers picked by Bob across time (at index -1 is the round before the current round; is empty in the first round)
    "pressed_buttons": list[str] # The list of buttons pressed by Alice across time (at index -1 is the round before the current round; is empty in the first round)
    "actions_alice": list[str] # The list of actions Alice took across time (at index -1 is the round before the current round; is empty in the first round)
    "rewards": list[float] # The list of total (joint) rewards received across time (at index -1 is the round before the current round; is empty in the first round)
}}

For player_1 (Bob), the observation dict has the following format:
{{
    "correct_levers": list[str] # The list of the lever labels shown to Alice across time (at index -1 is the round before the current round; is empty in the first round)
    "picked_levers": list[str] # The list of levers picked by Bob across time (at index -1 is the round before the current round; is empty in the first round)
    "shown_signals": list[str] # The list of signals shown by Alice to Bob across time (at index -1 is the current round, i.e. the signal alice just showed; is empty in the first round)
    "revealed_levers": list[str] # The list of levers revealed by Alice to Bob across time (at index -1 is the current round, each element is None if the lever was not revealed, otherwise it is the lever label)
    "actions_bob": list[str] # The list of actions Bob took across time (at index -1 is the round before the current round; is empty in the first round)
    "rewards": list[float] # The list of total (joint) rewards received across time (at index -1 is the round before the current round; is empty in the first round)
}}

All levers are represented by their lever labels. The following labels are possible:
{lever_labels}
- None # Bob picks no lever in a round if he or Alice bailed

All buttons are represented by their button labels. The following labels are possible:
{button_labels}
- None # The button is None if Alice doesn't press a button (bail or reveal_correct_lever)

All signals are represented by their signal labels. The following labels are possible:
{signal_labels}
- None # The signal is None if Alice doesn't press a button (bail or reveal_correct_lever)
"""

legal_actions_description = """
For player_0 (Alice), the legal actions are:
- bail
- reveal_correct_lever
{button_options}

For player_1 (Bob), the legal actions are:
- bail
{lever_options}
"""

class AdHocConventionsPermutedSignals(Env):
    def __init__(self, 
    policy_modifier_config: BasePolicyModifierConfig,
    seed: Optional[int] = None, 
    lever_labels: Optional[List[str]] = ["blue", "red"], 
    button_labels: Optional[List[str]] = ["apple", "banana"],
    signal_labels: Optional[List[str]] = ["square", "circle"],
    num_rounds: int = 10,
    ):
        super().__init__(policy_modifier_config)
        
        self.seed = seed
        self.lever_labels = lever_labels
        self.button_labels = button_labels
        self.signal_labels = signal_labels
        self.num_rounds = num_rounds
        self.button_to_signal_mapping = None

        if not len(self.lever_labels) == len(self.signal_labels):
            raise ValueError("The number of lever labels and signal labels must be the same!")
        if not len(self.button_labels) == len(self.signal_labels):
            raise ValueError("The number of button labels and signal labels must be the same!")

    def reset(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.current_round = 0

        self.button_to_signal_mapping = {button: signal for button, signal in zip(self.button_labels, np.random.permutation(self.signal_labels))}

        self.correct_levers = [np.random.choice(self.lever_labels)]
        self.shown_signals = []
        self.pressed_buttons = []
        self.picked_levers = []
        self.revealed_levers = []

        self.actions_alice = []
        self.actions_bob = []

        self._current_player = 0
        self.round_reward_buffer = 0
        self.rewards = []
        self._terminated = False

    def list_to_bullet_point_string(self, lst: List[str]) -> str:
        return "\n ".join([f"- {item}" for item in lst])

    @property
    def example_text_observations(self) -> Dict[str, str]:
        return {
            "observation_player_0": observation_alice,
            "observation_player_1": observation_bob,
        }

    @property
    def base_game_description(self) -> str:
        return game_description.format(
            num_rounds=self.num_rounds,
            num_levers=len(self.lever_labels),
            lever_labels=self.lever_labels,
            num_buttons=len(self.button_labels),
            button_labels=self.button_labels,
            num_signals=len(self.signal_labels),
            signal_labels=self.signal_labels,
        )
    
    @property
    def legal_actions(self) -> List[str]:
        if self._current_player == 0:
            return ["bail", "reveal_correct_lever"] + self.button_labels
        else:
            return ["bail"] + self.lever_labels
    
    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:
        if self._current_player == 0:
            obs = observation_alice.format(correct_lever=self.correct_levers[-1])
        else:
            if self.actions_alice[-1] == "reveal_correct_lever":
                reval_information = f"The correct lever for the current round has been revealed to you: {self.revealed_levers[-1]}."
                signal_information = ""
            else:
                reval_information = ""
                signal_information = f"Alice has shown the following signal: {self.shown_signals[-1]}."

            obs = observation_bob.format(signal_information=signal_information, reval_information=reval_information)

        if len(self.rewards) > 0:
            obs += f"\n In the previous round, you received a total reward of {self.rewards[-1]}."
        return (self._current_player, obs, self.legal_actions)
        
    @property
    def terminated(self) -> bool:
        return self._terminated
    
    @property
    def truncated(self) -> bool:
        return False
        
    # TODO: Continue here
    @property
    def dict_observation(self) -> Tuple[int, Dict, List[str]]:
        if self._current_player == 0:
            obs = {
                "correct_levers": self.correct_levers, 
                "picked_levers": self.picked_levers, 
                "pressed_buttons": self.pressed_buttons,
                "actions_alice": self.actions_alice[:self.current_round],
                "rewards": self.rewards[:self.current_round],
            }
            return (0, obs, self.legal_actions)
        else:
            obs = {
                "correct_levers": self.correct_levers[:self.current_round], 
                "picked_levers": self.picked_levers, 
                "shown_signals": self.shown_signals, 
                "revealed_levers": self.revealed_levers, 
                "actions_bob": self.actions_bob[:self.current_round],
                "rewards": self.rewards[:self.current_round],
            }
            return (1, obs, self.legal_actions)

    def step(self, action: str) -> Tuple[float, bool, bool, dict]:
        if action not in self.legal_actions:
            raise ValueError(f"Invalid action: {action}")
        
        def _next_round() -> None:
            self.current_round += 1
            self._current_player = 0
            self.rewards.append(self.round_reward_buffer)
            self.round_reward_buffer = 0

            if self.current_round < self.num_rounds:
                self.correct_levers.append(np.random.choice(self.lever_labels))
            else:
                self._terminated = True
        
        def _handle_action_alice(action: str) -> float:
            # Alice's turn
            self.actions_alice.append(action)
            if action == "bail":
                self.round_reward_buffer += 1
                self.shown_signals.append(None)
                self.pressed_buttons.append(None)
                self.revealed_levers.append(None)
                self.picked_levers.append(None)
                self.actions_bob.append(None)

                _next_round()
                reward = 1
            else:
                self._current_player = 1

                if action == "reveal_correct_lever":
                    self.shown_signals.append(None)
                    self.pressed_buttons.append(None)
                    self.revealed_levers.append(self.correct_levers[-1])
                    self.round_reward_buffer -= 5
                    reward = -5
                else:
                    self.pressed_buttons.append(action)
                    self.shown_signals.append(self.button_to_signal_mapping[action])
                    self.revealed_levers.append(None)
                    reward = 0
            return reward

        def _handle_action_bob(action: str) -> float:
            # Bob's turn
            self.actions_bob.append(action)
            if action == "bail":
                self.round_reward_buffer += 0.5
                self.picked_levers.append(None)
                reward = 0.5
            else:
                self.picked_levers.append(action)
                if action == self.correct_levers[-1]:
                    self.round_reward_buffer += 10
                    reward = 10
                else:
                    self.round_reward_buffer -= 10
                    reward = -10

            _next_round()
            return reward

        if self._current_player == 0:
            reward = _handle_action_alice(action)
        elif self._current_player == 1:
            reward = _handle_action_bob(action)

        # TODO: Remove this
        if reward is None:
            raise ValueError(f"Invalid reward: {reward}")

        return reward, self._terminated, False, {}
    
    @property
    def dict_observation_description(self) -> str:
        return dict_observation_description.format(
            lever_labels=self.list_to_bullet_point_string(self.lever_labels),
            signal_labels=self.list_to_bullet_point_string(self.signal_labels),
            button_labels=self.list_to_bullet_point_string(self.button_labels),
        )
    
    @property
    def possible_actions_description(self) -> str:
        return legal_actions_description.format(
            lever_options=self.list_to_bullet_point_string(self.lever_labels),
            button_options=self.list_to_bullet_point_string(self.button_labels),
        )
    
    def get_compact_rollout_history(self, ignore_not_done: bool = False) -> CompactRolloutHistory:
        if not self.done and not ignore_not_done:
            raise ValueError("Environment is not done yet!")

        hyperparams = {
            "lever_labels": self.lever_labels,
            "signal_labels": self.signal_labels,
            "button_labels": self.button_labels,
            "num_rounds": self.num_rounds,
        }

        metadata = {
            "button_to_signal_mapping": self.button_to_signal_mapping,
        }

        def compact_state(_round: int) -> Dict:
            return {
                "round": _round,
                "correct_lever": self.correct_levers[_round],
                "picked_lever": self.picked_levers[_round],
                "pressed_button": self.pressed_buttons[_round],
                "shown_signal": self.shown_signals[_round],
                "revealed_lever": self.revealed_levers[_round],
                "actions_alice": self.actions_alice[_round],
                "actions_bob": self.actions_bob[_round],
                "reward": self.rewards[_round], 
            }
        
        compact_states = [compact_state(_round) for _round in range(self.current_round)]

        compact_rollout_history = CompactRolloutHistory(
            hyperparams=hyperparams,
            metadata=metadata,
            rewards_sum=sum(self.rewards),
            terminated=self._terminated,
            truncated=False,
            compact_states=compact_states,
        )

        return compact_rollout_history
    
    @property
    def current_player(self) -> int:
        return self._current_player
    
    @property
    def env_name(self) -> str:
        return "AdHocConventions-v0"
    

if __name__ == "__main__":
    from configs.policy_modifiers import NoPolicyModifierConfig
    env = AdHocConventionsPermutedSignals(NoPolicyModifierConfig())
    print(env.base_game_description)
    env.reset()
    print("Mapping:", env.button_to_signal_mapping)
    while not env.terminated:
        print(env.text_observation)
        action = np.random.choice(env.legal_actions)
        print(f"Player {env.current_player} chose {action}")
        env.step(action)
    print(env.get_compact_rollout_history())