from envs.base import Env
import numpy as np
from typing import Optional, Tuple, List, Dict
from core import CompactRolloutHistory
from configs.policy_modifiers import BasePolicyModifierConfig

game_description = """
You are playing a series of spy missions with one other player. One of you is Alice (the handler) and the other is Bob (the field agent).
The game consists of {num_rounds} missions. There are {num_levers} possible mission locations, labeled {lever_labels}, and {num_signals} code words, labeled {signal_labels}.
Your goal is to maximize your (joint) reward at the end of the game (i.e. after all {num_rounds} missions have been completed).

Each mission works as follows:
First it is Alice's turn. Alice receives intelligence about the correct location for the current mission (the correct location for each mission is chosen randomly). 
Alice can do exactly one of the following:
- abort the mission for a (joint) reward of 1 (this means that the current mission is skipped and the players receive a joint reward of 1)
- Alice can reveal the correct location for the current mission directly to Bob, which costs 5 joint reward (i.e. -5 joint reward) because it risks compromising the operation
- Alice can send one of the {num_signals} code words to Bob. This is free and relatively safe.

If Alice didn't abort, it is Bob's turn. Bob observes the code word sent by Alice or the revealed location.
Bob can now do exactly one of the following:
- abort the mission for a (joint) reward of 0.5
- go to one of the {num_levers} locations. 
    - If Bob goes to the correct location, they successfully complete the mission and receive a joint reward of 10
    - If Bob goes to the wrong location, they fail the mission and receive a joint reward of -10

The players cannot communicate with each other except through the options presented above (i.e. revealing the correct location or sending code words).
"""

# **Role assignment:**
# - player 0 is alice (handler)
# - player 1 is bob (field agent)

observation_alice = """
You are Alice, the handler. The correct location for the current mission is {correct_lever}.
"""

# Rest will be handled in code
observation_bob = """
You are Bob, the field agent.
{signal_information}
{reval_information}
"""

dict_observation_description = """
For player_0 (Alice), the observation dict has the following format:
{{
    "correct_levers": list[str] # The list of location labels shown to Alice across time (at index -1 is the current mission)
    "picked_levers": list[str] # The list of locations Bob went to across time (at index -1 is the mission before the current mission; is empty in the first mission)
    "shown_signals": list[str] # The list of code words sent by Alice to Bob across time (at index -1 is the mission before the current mission; is empty in the first mission)
    "actions_alice": list[str] # The list of actions Alice took across time (at index -1 is the mission before the current mission)
    "actions_bob": list[str] # The list of actions Bob took across time (at index -1 is the mission before the current mission; action is None if Alice aborted)
    "rewards": list[float] # The list of total (joint) rewards received across time (at index -1 is the mission before the current mission; is empty in the first mission)
}}

For player_1 (Bob), the observation dict has the following format:
{{
    "correct_levers": list[str] # The list of location labels shown to Alice across time (at index -1 is the mission before the current mission; is empty in the first mission)
    "picked_levers": list[str] # The list of locations Bob went to across time (at index -1 is the mission before the current mission; is empty in the first mission)
    "shown_signals": list[str] # The list of code words sent by Alice to Bob across time (at index -1 is the current mission, i.e. the code word Alice just sent; is empty in the first mission)
    "revealed_levers": list[str] # The list of locations revealed by Alice to Bob across time (at index -1 is the current mission, each element is None if the location was not revealed, otherwise it is the location label)
    "actions_alice": list[str] # The list of actions Alice took across time (at index -1 is the mission before the current mission)
    "actions_bob": list[str] # The list of actions Bob took across time (at index -1 is the mission before the current mission; action is None if Alice aborted)
    "rewards": list[float] # The list of total (joint) rewards received across time (at index -1 is the mission before the current mission; is empty in the first mission)
}}

Note that all locations are represented by their location labels. The following labels are possible:
{lever_labels}
- None # Bob goes to no location in a mission if he or Alice aborted

All code words are represented by their code word labels. The following labels are possible:
{signal_labels}
- None # The code word is None if Alice doesn't send a code word (abort or reveal_correct_location)
"""

legal_actions_description = """
For player_0 (Alice), the legal actions are:
- bail
- reveal_correct_location
{signal_options}

For player_1 (Bob), the legal actions are:
- bail
{lever_options}
"""

class AdHocConventions2(Env):
    def __init__(
        self, 
        policy_modifier_config: BasePolicyModifierConfig, 
        seed: Optional[int] = None, 
        lever_labels: Optional[List[str]] = ["warehouse", "docks"], 
        signal_labels: Optional[List[str]] = ["eagle", "falcon"], 
        num_rounds: int = 10,
    ):
        super().__init__(policy_modifier_config)
        
        self.seed = seed
        self.lever_labels = lever_labels
        self.signal_labels = signal_labels
        self.num_rounds = num_rounds

        if not len(self.lever_labels) == len(self.signal_labels):
            raise ValueError("The number of location labels and code word labels must be the same!")

    def reset(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.current_round = 0

        self.correct_levers = [np.random.choice(self.lever_labels)]
        self.shown_signals = []
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
            num_signals=len(self.signal_labels),
            signal_labels=self.signal_labels,
        )
    
    @property
    def legal_actions(self) -> List[str]:
        if self._current_player == 0:
            return ["bail", "reveal_correct_location"] + self.signal_labels
        else:
            return ["bail"] + self.lever_labels
    
    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:
        if self._current_player == 0:
            obs = observation_alice.format(correct_lever=self.correct_levers[-1])
        else:
            if self.actions_alice[-1] == "reveal_correct_location":
                reval_information = f"The correct location for the current mission has been revealed to you: {self.revealed_levers[-1]}."
                signal_information = ""
            else:
                reval_information = ""
                signal_information = f"Alice has sent you the following code word: {self.shown_signals[-1]}."

            obs = observation_bob.format(signal_information=signal_information, reval_information=reval_information)

        if len(self.rewards) > 0:
            obs += f"\n In the previous mission, you received a total reward of {self.rewards[-1]}."
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
                "shown_signals": self.shown_signals, 
                "actions_alice": self.actions_alice[:self.current_round], 
                "actions_bob": self.actions_bob[:self.current_round],
                "rewards": self.rewards[:self.current_round],
            }
            return (0, obs, self.legal_actions)
        else:
            obs = {
                "correct_levers": self.correct_levers[:self.current_round], 
                "picked_levers": self.picked_levers, 
                "shown_signals": self.shown_signals, 
                "revealed_levers": self.revealed_levers, 
                "actions_alice": self.actions_alice[:self.current_round], 
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
                self.revealed_levers.append(None)
                self.picked_levers.append(None)
                self.actions_bob.append(None)

                _next_round()
                reward = 1
            else:
                self._current_player = 1

                if action == "reveal_correct_location":
                    self.shown_signals.append(None)
                    self.revealed_levers.append(self.correct_levers[-1])
                    self.round_reward_buffer -= 5
                    reward = -5
                else:
                    self.shown_signals.append(action)
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
        )
    
    @property
    def possible_actions_description(self) -> str:
        return legal_actions_description.format(
            lever_options=self.list_to_bullet_point_string(self.lever_labels),
            signal_options=self.list_to_bullet_point_string(self.signal_labels),
        )
    
    def get_compact_rollout_history(self, ignore_not_done: bool = False) -> CompactRolloutHistory:
        if not self.done and not ignore_not_done:
            raise ValueError("Environment is not done yet!")

        hyperparams = {
            "lever_labels": self.lever_labels,
            "signal_labels": self.signal_labels,
            "num_rounds": self.num_rounds,
        }

        def compact_state(_round: int) -> Dict:
            return {
                "round": _round,
                "correct_location": self.correct_levers[_round],
                "picked_location": self.picked_levers[_round],
                "shown_code_word": self.shown_signals[_round],
                "revealed_location": self.revealed_levers[_round],
                "actions_alice": self.actions_alice[_round],
                "actions_bob": self.actions_bob[_round],
                "reward": self.rewards[_round], 
            }
        
        compact_states = [compact_state(_round) for _round in range(self.current_round)]

        compact_rollout_history = CompactRolloutHistory(
            hyperparams=hyperparams,
            metadata={},
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
        return "AdHocConventions-v1"


if __name__ == "__main__":
    from configs.policy_modifiers import NoPolicyModifierConfig
    
    env = AdHocConventions2(NoPolicyModifierConfig())
    actions_alice = ["bail", "reveal_correct_location", "eagle", "falcon"]
    actions_bob = ["bail", "warehouse", "docks"]
    env.reset()
    
    # Run until the game is done (10 missions = up to 20 actions)
    action_count = 0
    while not env.terminated:
        print(env.text_observation)
        action = np.random.choice(env.legal_actions)
        print(f"Player {env.current_player} chose {action}")
        if env.current_player == 0:
            env.step(action)
        else:
            env.step(action)
        action_count += 1
    
    print(f"\nGame completed after {action_count} actions ({env.current_round} missions)")
    print(env.get_compact_rollout_history())