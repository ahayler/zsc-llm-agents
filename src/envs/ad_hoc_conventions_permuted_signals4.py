from envs.base import Env
import numpy as np
from typing import Optional, Tuple, List, Dict
from core import CompactRolloutHistory
from configs.policy_modifiers import BasePolicyModifierConfig

game_description = """
You are working together in a research laboratory. One of you is Alice (the lead researcher) and the other is Bob (the lab assistant).
The game consists of {num_rounds} experiment setups. There are {num_levers} possible equipment stations, labeled {lever_labels}, 
{num_buttons} protocol codes labeled {button_labels} and {num_signals} station indicators labeled {signal_labels}.
You know that there is a fixed (across all experiments) but unknown 1:1 mapping (bijection) between the protocol codes and the station indicators 
(i.e. if Alice enters a protocol code, exactly one station indicator is displayed to Bob and each station indicator has exactly one protocol code associated with it).
Alice does not observe which station indicator Bob sees when she enters a protocol code and Bob does not observe which protocol code Alice entered.
Your goal is to maximize your (joint) reward at the end of the game (i.e. after all {num_rounds} experiment setups have been completed).

Each experiment setup works as follows:
First it is Alice's turn. Alice receives the experiment protocol with the correct equipment station for the current experiment (the correct equipment station for each experiment is chosen randomly). 
Alice can do exactly one of the following:
- skip the experiment for a (joint) reward of 1 (this means that the current experiment is skipped and the players receive a joint reward of 1)
- Alice can tell Bob the exact equipment station directly, which costs 5 joint reward (i.e. -5 joint reward) because it disrupts the laboratory's protocol system and slows down research
- Alice can enter one of the {num_buttons} protocol codes, which will display the corresponding station indicator to Bob. This is free.

If Alice didn't skip, it is Bob's turn. Bob observes the station indicator corresponding to the protocol code entered by Alice or the revealed equipment station.
Bob can now do exactly one of the following:
- skip the experiment for a (joint) reward of 0.5
- go to one of the {num_levers} equipment stations. 
    - If Bob goes to the correct equipment station, they successfully set up the experiment and receive a joint reward of 10
    - If Bob goes to the wrong equipment station, they cannot set up the experiment correctly and receive a joint reward of -10

The players cannot communicate with each other except through the options presented above (i.e. revealing the exact equipment station or entering protocol codes).
"""

# **Role assignment:**
# - player 0 is alice (lead researcher)
# - player 1 is bob (lab assistant)

observation_alice = """
You are Alice, the lead researcher. The correct equipment station for the current experiment is {correct_lever}.
"""

# Rest will be handled in code
observation_bob = """
You are Bob, the lab assistant.
{signal_information}
{reval_information}
"""

dict_observation_description = """
For player_0 (Alice), the observation dict has the following format:
{{
    "correct_levers": list[str] # The list of equipment station labels shown to Alice across time (at index -1 is the current experiment)
    "picked_levers": list[str] # The list of equipment stations Bob went to across time (at index -1 is the experiment before the current experiment; is empty in the first experiment)
    "pressed_buttons": list[str] # The list of protocol codes entered by Alice across time (at index -1 is the experiment before the current experiment; is empty in the first experiment)
    "actions_alice": list[str] # The list of actions Alice took across time (at index -1 is the experiment before the current experiment; is empty in the first experiment)
    "rewards": list[float] # The list of total (joint) rewards received across time (at index -1 is the experiment before the current experiment; is empty in the first experiment)
}}

For player_1 (Bob), the observation dict has the following format:
{{
    "correct_levers": list[str] # The list of equipment station labels shown to Alice across time (at index -1 is the experiment before the current experiment; is empty in the first experiment)
    "picked_levers": list[str] # The list of equipment stations Bob went to across time (at index -1 is the experiment before the current experiment; is empty in the first experiment)
    "shown_signals": list[str] # The list of station indicators displayed to Bob across time (at index -1 is the current experiment, i.e. the station indicator Bob just saw; is empty in the first experiment)
    "revealed_levers": list[str] # The list of equipment stations revealed by Alice to Bob across time (at index -1 is the current experiment, each element is None if the equipment station was not revealed, otherwise it is the equipment station label)
    "actions_bob": list[str] # The list of actions Bob took across time (at index -1 is the experiment before the current experiment; is empty in the first experiment)
    "rewards": list[float] # The list of total (joint) rewards received across time (at index -1 is the experiment before the current experiment; is empty in the first experiment)
}}

All equipment stations are represented by their equipment station labels. The following labels are possible:
{lever_labels}
- None # Bob goes to no equipment station in an experiment if he or Alice skipped

All protocol codes are represented by their protocol code labels. The following labels are possible:
{button_labels}
- None # The protocol code is None if Alice doesn't enter a protocol code (skip or reveal_correct_station)

All station indicators are represented by their station indicator labels. The following labels are possible:
{signal_labels}
- None # The station indicator is None if Alice doesn't enter a protocol code (skip or reveal_correct_station)
"""

legal_actions_description = """
For player_0 (Alice), the legal actions are:
- bail
- reveal_correct_station
{button_options}

For player_1 (Bob), the legal actions are:
- bail
{lever_options}
"""

class AdHocConventionsPermutedSignals4(Env):
    def __init__(self, 
    policy_modifier_config: BasePolicyModifierConfig,
    seed: Optional[int] = None, 
    lever_labels: Optional[List[str]] = ["station_1", "station_2"], 
    button_labels: Optional[List[str]] = ["willow", "maple"],
    signal_labels: Optional[List[str]] = ["lion", "giraffe"],
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
            raise ValueError("The number of equipment station labels and station indicator labels must be the same!")
        if not len(self.button_labels) == len(self.signal_labels):
            raise ValueError("The number of protocol code labels and station indicator labels must be the same!")

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
            return ["bail", "reveal_correct_station"] + self.button_labels
        else:
            return ["bail"] + self.lever_labels
    
    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:
        if self._current_player == 0:
            obs = observation_alice.format(correct_lever=self.correct_levers[-1])
        else:
            if self.actions_alice[-1] == "reveal_correct_station":
                reval_information = f"The correct equipment station for the current experiment has been revealed to you: {self.revealed_levers[-1]}."
                signal_information = ""
            else:
                reval_information = ""
                signal_information = f"Alice has displayed the following station indicator: {self.shown_signals[-1]}."

            obs = observation_bob.format(signal_information=signal_information, reval_information=reval_information)

        if len(self.rewards) > 0:
            obs += f"\n In the previous experiment, you received a total reward of {self.rewards[-1]}."
        return (self._current_player, obs, self.legal_actions)
        
    @property
    def terminated(self) -> bool:
        return self._terminated
    
    @property
    def truncated(self) -> bool:
        return False
        
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

                if action == "reveal_correct_station":
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
                "correct_station": self.correct_levers[_round],
                "picked_station": self.picked_levers[_round],
                "entered_protocol_code": self.pressed_buttons[_round],
                "displayed_station_indicator": self.shown_signals[_round],
                "revealed_station": self.revealed_levers[_round],
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
        return "AdHocConventions-v3"


if __name__ == "__main__":
    from configs.policy_modifiers import NoPolicyModifierConfig
    env = AdHocConventionsPermutedSignals4(NoPolicyModifierConfig())
    print(env.base_game_description)
    env.reset()
    print("Protocol code to station indicator mapping:", env.button_to_signal_mapping)
    while not env.terminated:
        print(env.text_observation)
        action = np.random.choice(env.legal_actions)
        print(f"Player {env.current_player} chose {action}")
        env.step(action)
    print(env.get_compact_rollout_history())

