
import numpy as np
from typing import Optional, List, Tuple, Dict
from core import CompactRolloutHistory
from configs.policy_modifiers import BasePolicyModifierConfig
from envs.base import Env

game_description = """
# Game Description (Sequential, 2-Player)

- There are **two players**: player_0 acts first, and player_1 acts second.  
- Each player is randomly dealt **one card**, either **Card 1** or **Card 2**.  
- After seeing their card, each player must choose one of **three actions**: **A**, **B**, or **C**.  
- **player_0 moves first**, choosing an action without knowing player_1's card.  
- **player_1 observes** both player_0's action and their own card before choosing an action.  

---

## Payoffs

The payoff depends on both players' cards and chosen actions.

---

### Case 1: player_0 has **Card 1**, player_1 has **Card 1**

- (P1 = A, P2 = A) → **10**  
- (P1 = B, P2 = B) → **8**  
- (P1 = B, P2 = A) → **4**
- (P1 = B, P2 = C) → **4**  
- (P1 = C, P2 = A) → **10**  
- All other choices → **0**

---

### Case 2: player_0 has **Card 1**, player_1 has **Card 2**

- (P1 = A, P2 = C) → **10**  
- (P1 = B, P2 = B) → **8**  
- (P1 = B, P2 = A) → **4**  
- (P1 = B, P2 = C) → **4**  
- (P1 = C, P2 = C) → **10**  
- All other choices → **0**

---

### Case 3: player_0 has **Card 2**, player_1 has **Card 1**

- (P1 = A, P2 = C) → **10**  
- (P1 = B, P2 = B) → **8**  
- (P1 = B, P2 = A) → **4**  
- (P1 = B, P2 = C) → **4**  
- All other choices → **0**

---

### Case 4: player_0 has **Card 2**, player_1 has **Card 2**

- (P1 = A, P2 = A) → **10**  
- (P1 = B, P2 = B) → **8**  
- (P1 = B, P2 = A) → **4**  
- (P1 = B, P2 = C) → **4**  
- (P1 = C, P2 = A) → **10**  
- All other choices → **0**
"""

dict_observation_description = """
The observation dict for player_0 has the following format:
{
    "your_card": int # Your card (1 or 2)
}

The observation dict for player_1 has the following format:
{
    "your_card": int # Your card (1 or 2)
    "player_0_action": str # player_0's action
}
"""

possible_actions_description = """
The possible actions for each player are:
- "A"
- "B" 
- "C"

Note that the action is a string, not an integer.
"""


class TinyHanabi(Env):
    def __init__(self, policy_modifier_config: BasePolicyModifierConfig):
        super().__init__(policy_modifier_config)
        
        # Game state variables
        self.cards = None  # [card_p1, card_p2] where each card is 1 or 2
        self.actions = None  # [action_p1, action_p2] where actions are "A", "B", "C" or None
        self._current_player = None  # 0 for Player 1, 1 for Player 2
        self.reward = None
        
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        # Deal random cards to both players
        self.cards = [np.random.choice([1, 2]), np.random.choice([1, 2])]
        self.actions = [None, None]
        self._current_player = 0  # Player 1 starts
        self._terminated = False
        self._truncated = False
        self.reward = 0

    @property
    def base_game_description(self) -> str:
        return game_description
    
    @property
    def example_text_observations(self) -> Dict[str, str]:
        return {
            "player_0_template": "Your card: {your_card}.",
            "player_1_template": "Your card: {your_card}. player_0's action: {p0_action}",
        }
    
    @property
    def legal_actions(self) -> List[str]:
        return ["A", "B", "C"]
    
    def _validate_action(self, action: str) -> Tuple[bool, str]:
        if action not in ["A", "B", "C"]:
            return False, f"Action must be one of A, B, C, got {action}"
        return True, ""
    
    @property
    def dict_observation(self) -> Tuple[int, Dict, List[str]]:
        self._assert_valid_game_state()

        if self._current_player == 0:
            observation = {
                "your_card": self.cards[0],
            }
        else:
            observation = {
                "your_card": self.cards[1],
                "player_0_action": self.actions[0],
            }
        
        return self.current_player, observation, self.legal_actions
    
    @property
    def dict_observation_description(self) -> str:
        return dict_observation_description
    
    @property
    def possible_actions_description(self) -> str:
        return possible_actions_description
    
    def _calculate_payoff(self, p1_card: int, p2_card: int, p1_action: str, p2_action: str) -> int:
        """Calculate the payoff based on the game rules."""
        # Define payoff table
        payoffs = {
            (1, 1): {  # Case 1: Both have Card 1
                ("A", "A"): 10,
                ("B", "B"): 8,
                ("B", "A"): 4,
                ("B", "C"): 4,
                ("C", "A"): 10,
            },
            (1, 2): {  # Case 2: P1 has Card 1, P2 has Card 2
                ("A", "C"): 10,
                ("B", "B"): 8,
                ("B", "A"): 4,
                ("B", "C"): 4,
                ("C", "C"): 10,
            },
            (2, 1): {  # Case 3: P1 has Card 2, P2 has Card 1
                ("A", "C"): 10,
                ("B", "B"): 8,
                ("B", "A"): 4,
                ("B", "C"): 4,
            },
            (2, 2): {  # Case 4: Both have Card 2
                ("A", "A"): 10,
                ("B", "B"): 8,
                ("B", "A"): 4,
                ("B", "C"): 4,
                ("C", "A"): 10,
            }
        }
        
        card_combo = (p1_card, p2_card)
        action_combo = (p1_action, p2_action)
        
        return payoffs.get(card_combo, {}).get(action_combo, 0)
    
    def step(self, action: str) -> Tuple[float, bool, bool, Dict]:
        self._assert_valid_game_state()
        
        valid, error_msg = self._validate_action(action)
        if not valid:
            raise ValueError(error_msg)
        
        # Store the action
        self.actions[self._current_player] = action
        
        if self._current_player == 0:
            # Player 1 has acted, now it's Player 2's turn
            self._current_player = 1
            reward = 0  # No reward yet, game continues
        else:
            # Player 2 has acted, game is complete
            reward = self._calculate_payoff(
                self.cards[0], self.cards[1],
                self.actions[0], self.actions[1]
            )
            self.reward = reward
            self._terminated = True
        
        return float(reward), self.terminated, self.truncated, {}
    
    def _assert_valid_game_state(self):
        if self.cards is None:
            raise ValueError("You must call reset() before calling step().")
        if self._terminated:
            raise ValueError("Game is already terminated.")
        if self._current_player not in [0, 1]:
            raise ValueError("Invalid current player.")
    
    @property
    def terminated(self) -> bool:
        return self._terminated
    
    @property
    def truncated(self) -> bool:
        return self._truncated
    
    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:
        if self._current_player == 0:
            obs = f"You are player_0. Your card is: {self.cards[0]}."
        else:
            obs = f"You are player_1. Your card is: {self.cards[1]}. player_0's action: {self.actions[0]}."
        
        return self.current_player, obs, self.legal_actions
    
    @property
    def env_name(self) -> str:
        return "TinyHanabi-v0"
    
    def get_compact_rollout_history(self, ignore_not_done: bool = False) -> CompactRolloutHistory:
        if not self.done and not ignore_not_done:
            raise ValueError("Please finish the game before calling get_compact_rollout_history()!")
        
        # For TinyHanabi, we create a single compact state representing the complete game
        compact_states = []
        if self._terminated and self.actions[0] is not None and self.actions[1] is not None:
            compact_states.append({
                "p1_card": int(self.cards[0]),
                "p2_card": int(self.cards[1]),
                "p1_action": self.actions[0],
                "p2_action": self.actions[1],
                "reward": int(self.reward) if self.reward is not None else 0,
            })
        
        return CompactRolloutHistory(
            hyperparams={},
            metadata={
                "cards": [int(x) for x in self.cards] if self.cards is not None else None,
                "actions": self.actions,
            },
            compact_states=compact_states,
            rewards_sum=float(self.reward) if self.reward is not None else None,
            terminated=self.terminated,
            truncated=self.truncated,
        )