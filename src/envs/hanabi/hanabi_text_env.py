import jax
# Force CPU-only execution and disable JIT for the Hanabi simulator.
# We use JaxMARL purely as a game engine (not for training), so GPU and JIT
# compilation overhead are unnecessary and can cause version-mismatch issues.
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_disable_jit", True)

import jaxmarl
from envs.hanabi.hanabi import HanabiEnv as _HanabiEnv
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple

from core import CompactRolloutHistory
from configs.policy_modifiers import BasePolicyModifierConfig
from envs.base import Env


_COLOR_NAMES = {
    "R": "Red", "Y": "Yellow", "G": "Green", "W": "White", "B": "Blue",
}
_ALL_COLORS = ["R", "Y", "G", "W", "B"]


def _build_game_description(num_colors: int, num_ranks: int, max_info_tokens: int = 8, max_life_tokens: int = 3) -> str:
    colors = _ALL_COLORS[:num_colors]
    color_str = ", ".join(f"{c} ({_COLOR_NAMES[c]})" for c in colors)
    rank_str = ", ".join(str(r) for r in range(1, num_ranks + 1))
    max_score = num_colors * num_ranks

    # num_cards_of_rank: [3, 2, ..., 2, 1]
    cards_per_rank = [3] + [2] * (num_ranks - 2) + [1]
    deck_desc = ", ".join(
        f"three {r}s" if c == 3 else f"two {r}s" if c == 2 else f"one {r}"
        for r, c in zip(range(1, num_ranks + 1), cards_per_rank)
    )

    color_hint_examples = ", ".join(f"`H{c} to P1 relative`" for c in colors)
    rank_hint_examples = ", ".join(f"`H{r} to P1 relative`" for r in range(1, num_ranks + 1))

    return f"""# Hanabi — Cooperative Card Game

You are playing **Hanabi** with one other player. Hanabi is a cooperative card game: the two of you
work together to build {num_colors} fireworks (one per color) by playing numbered cards in ascending order.
You share a single score — your goal is to maximize it together. The maximum possible score is **{max_score}**.

## Overview

Hanabi is a fully cooperative card game. The goal is to play cards to build {num_colors} "fireworks"
(one per color) in ascending order from 1 to {num_ranks}.
The maximum possible score is **{max_score}** ({num_colors} colors × {num_ranks} ranks).

## The Catch: You Cannot See Your Own Hand

Each player holds their cards **facing outward** — you can see everyone else's cards, but not your own.
You must rely on hints from teammates to deduce what you are holding.

## Cards

- **{num_colors} colors**: {color_str}
- **Ranks**: {rank_str}
- Each color has {deck_desc}.
- Cards are shown as a color letter followed by a rank number, e.g. **{colors[0]}3** = {_COLOR_NAMES[colors[0]]} 3, **{colors[-1]}1** = {_COLOR_NAMES[colors[-1]]} 1.

## Fireworks

Each color must be played in strict order: 1 → 2 → ... → {num_ranks}.
Playing a card out of order (wrong rank for that color) is a **mistake** and costs a life token.
Completing a color (playing its {num_ranks}) refills **one information token**.

## Information Tokens

- There are **{max_info_tokens} information tokens** (the "hint budget").
- Giving a hint to another player costs **1 information token**.
- Discarding a card refills **1 information token** (up to the maximum of {max_info_tokens}).
- You cannot give a hint if there are 0 information tokens remaining.

## Life Tokens

- There are **{max_life_tokens} life token(s)**.
- Playing a card that does not legally continue a fireworks pile costs **1 life token** and the card is discarded.
- If all {max_life_tokens} life token(s) are lost, the game ends immediately with a score of **0**.

## Deck and End Condition

- Once the draw pile is exhausted, each player gets **one final turn**, then the game ends.
- The game also ends immediately if all fireworks are completed (score {max_score}) or all life tokens are lost.

## Hand Positions

Cards in hand are numbered **0 to 4** (for 5-card hands), where **0 is the oldest card** (held longest)
and **4 is the newest** (just drawn). When you play or discard card at position i, you draw a new card
which takes the next available position.

## Actions

On your turn you must take **exactly one** of the following actions:

### Discard
- `D0` through `D4`: Discard the card at hand position 0–4.
  Refills 1 information token. You draw a replacement card.

### Play
- `P0` through `P4`: Attempt to play the card at hand position 0–4 onto the fireworks.
  - If it legally continues a pile → score +1, draw a replacement card.
  - If it does not → lose 1 life token, card is discarded, draw a replacement card.
  - Playing a {num_ranks} that completes a color → score +1 AND refill 1 information token.

### Hint (Color)
- `H{{color}} to P{{n}} relative`: Tell another player which of their cards match a specific color.
  - `{{color}}` is one of {", ".join(colors)}. Examples: {color_hint_examples}.
  - `P{{n}} relative` means the player n seats to your left (P1 = player directly to your left).
  - All cards of that color in their hand are revealed; you cannot give a false hint.
  - Costs 1 information token.

### Hint (Rank)
- `H{{rank}} to P{{n}} relative`: Tell another player which of their cards match a specific rank.
  - `{{rank}}` is one of {rank_str}. Examples: {rank_hint_examples}.
  - All cards of that rank in their hand are revealed.
  - Costs 1 information token.

## Observation Format

Each turn you see:
- **Your Hand**: Your cards are hidden. You see only the hints you have received so far
  (`Hints: <color><rank>` if both known, partial otherwise) and which cards are still possible
  (`Possible: <candidates>`).
- **Other Hand(s)**: The actual cards held by other players, plus any hints they have received.
- **Board state**: Current turn, score, information tokens remaining, life tokens remaining,
  deck size, discarded cards, and fireworks progress.
- **Last action**: What the previous player did and its outcome.
- **Legal actions**: The actions you may take this turn.
"""


def _build_dict_obs_description(num_colors: int) -> str:
    colors = _ALL_COLORS[:num_colors]
    fireworks_entries = ", ".join(f'"{c}": int' for c in colors)
    return f"""The observation dict has the following format:
{{
    "player_id": int,                  # Your player index (0-based)
    "info_tokens": int,                # Information tokens remaining (0-8)
    "life_tokens": int,                # Life tokens remaining (0-3)
    "fireworks": {{                     # Progress on each color pile
        {fireworks_entries}  # 0-{num_colors} each
    }},
    "deck_size": int,                  # Cards remaining in draw pile
    "my_hand_hints": [                 # Your hand (cards hidden; hints only)
        {{
            "color_hint": str | None,  # Color revealed by a hint, or None
            "rank_hint": int | None,   # Rank revealed by a hint, or None
        }},
        ...  # one entry per card position (0 = oldest)
    ],
    "other_hands": {{                   # Other players' visible cards
        "player_1": ["{colors[0]}3", "{colors[-1]}1", ...],
        ...
    }},
    "last_action": str,                # Action string of the last action taken
    "turn": int,                       # Current turn number
    "score": int,                      # Current score
}}
"""


def _build_actions_description(num_colors: int, num_ranks: int) -> str:
    colors = _ALL_COLORS[:num_colors]
    hand_size = 5  # 2-3 player default
    discard_range = f"D0, ..., D{hand_size - 1}"
    play_range = f"P0, ..., P{hand_size - 1}"
    color_hints = ", ".join(f"H{c} to P1 relative" for c in colors)
    rank_hints = ", ".join(f"H{r} to P1 relative" for r in range(1, num_ranks + 1))
    return f"""Actions are strings. The possible actions depend on the game state (info tokens, deck size, etc.)
and are always listed in the observation under "Legal Actions".

Discard:  {discard_range}  (discard card at position 0–{hand_size - 1})
Play:     {play_range}  (play card at position 0–{hand_size - 1})
Color hints (2-player, to the one other player):
          {color_hints}
Rank hints (2-player, to the one other player):
          {rank_hints}

Hints are only legal when information tokens > 0.
Discards are always legal (even when info tokens are full; you just waste the refill).
"""


# Keep static module-level constants for the default 5×5 variant (backwards compatibility).
HANABI_GAME_DESCRIPTION = _build_game_description(5, 5)
HANABI_DICT_OBS_DESCRIPTION = _build_dict_obs_description(5)
HANABI_ACTIONS_DESCRIPTION = _build_actions_description(5, 5)


class HanabiTextEnv(Env):
    """Text environment wrapper around the JaxMARL Hanabi simulator.

    The JAX simulator is stateless (all game state lives in an immutable JAX
    dataclass). This wrapper holds the current state in Python-level attributes
    and converts it to human-readable text on each observation request.

    Args:
        policy_modifier_config: Policy modifier config (passed to Env base class).
        num_agents: Number of players (2–5). Defaults to 2.
        num_colors: Number of colors (3, 4, or 5). Defaults to 5.
        num_ranks: Number of ranks (3, 4, or 5). Defaults to 5.
    """

    def __init__(
        self,
        policy_modifier_config: BasePolicyModifierConfig,
        num_agents: int = 2,
        num_colors: int = 5,
        num_ranks: int = 5,
        max_info_tokens: int = 8,
        max_life_tokens: int = 3,
    ):
        super().__init__(policy_modifier_config)
        self.num_agents = num_agents
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.max_info_tokens = max_info_tokens
        self.max_life_tokens = max_life_tokens
        # num_cards_of_rank: standard Hanabi distribution — [3, 2, ..., 2, 1]
        num_cards_of_rank = np.array([3] + [2] * (num_ranks - 2) + [1])
        # Instantiate from the local HanabiEnv (not jaxmarl.make) so that our
        # local fixes to hanabi_game.py (terminal condition, rendering) are used.
        self._jax_env = _HanabiEnv(
            num_agents=num_agents,
            num_colors=num_colors,
            num_ranks=num_ranks,
            num_cards_of_rank=num_cards_of_rank,
            max_info_tokens=max_info_tokens,
            max_life_tokens=max_life_tokens,
        )
        # Build reverse action map: "D0" -> 0, "HR to P1 relative" -> 10, etc.
        self._action_str_to_idx: Dict[str, int] = {
            v: k for k, v in self._jax_env.action_encoding.items()
        }
        # Python-level mutable state — all reset on each reset() call.
        self._jax_state = None
        self._prev_jax_state = None
        self._last_action_int: Optional[int] = None
        self._rng_key = None
        self._seed: Optional[int] = None
        self._terminated_flag: bool = False
        self._truncated_flag: bool = False
        self._cumulative_reward: float = 0.0
        self._step_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Env interface: lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None):
        """Reset the environment to a new game.

        Args:
            seed: Optional integer seed for reproducibility.
        """
        int_seed = seed if seed is not None else int(np.random.randint(0, 2**31))
        self._seed = int_seed
        self._rng_key = jax.random.PRNGKey(int_seed)
        self._rng_key, reset_key = jax.random.split(self._rng_key)

        _, self._jax_state = self._jax_env.reset(reset_key)
        self._prev_jax_state = self._jax_state
        # Use noop action index as the "no last action" sentinel on turn 0.
        self._last_action_int = self._jax_env.num_moves - 1
        self._terminated_flag = False
        self._truncated_flag = False
        self._cumulative_reward = 0.0
        self._step_history = []

    def step(self, action: str) -> Tuple[float, bool, bool, Dict]:
        """Execute one step in the environment.

        Args:
            action: Action string for the current player (e.g. "D0", "P2",
                    "HR to P1 relative"). Must be in self.legal_actions.

        Returns:
            Tuple of (reward, terminated, truncated, info).
        """
        self.assert_not_done()

        if action not in self._action_str_to_idx:
            raise ValueError(
                f"Unknown action '{action}'. Legal actions: {self.legal_actions}"
            )

        action_int = self._action_str_to_idx[action]
        current_agent = self._jax_env.agents[self.current_player]

        # Build the actions dict: current player takes their action, others noop.
        actions_dict = {
            agent: (action_int if agent == current_agent else self._jax_env.num_moves - 1)
            for agent in self._jax_env.agents
        }

        # Split the RNG key before every step_env call (JAX requires fresh keys).
        self._rng_key, step_key = jax.random.split(self._rng_key)

        _, new_state, rewards_dict, dones_dict, _ = self._jax_env.step_env(
            step_key, self._jax_state, actions_dict
        )

        reward = float(rewards_dict["__all__"])
        done = bool(dones_dict["__all__"])

        # Record step before updating state (current_player still reflects who just acted).
        self._step_history.append({
            "player": self.current_player,
            "action": action,
            "reward": reward,
            "score": int(new_state.score),
            "turn": int(new_state.turn),
        })

        # Advance state.
        self._prev_jax_state = self._jax_state
        self._jax_state = new_state
        self._last_action_int = action_int
        self._terminated_flag = done
        self._cumulative_reward += reward

        info = {
            "score": int(new_state.score),
            "out_of_lives": bool(new_state.out_of_lives),
            "bombed": bool(new_state.bombed),
            "turn": int(new_state.turn),
        }
        return reward, self._terminated_flag, self._truncated_flag, info

    # ------------------------------------------------------------------
    # Env interface: properties
    # ------------------------------------------------------------------

    @property
    def current_player(self) -> int:
        return int(np.argmax(np.array(self._jax_state.cur_player_idx)))

    @property
    def terminated(self) -> bool:
        return self._terminated_flag

    @property
    def truncated(self) -> bool:
        return self._truncated_flag

    @property
    def env_name(self) -> str:
        return f"Hanabi-{self.num_colors}x{self.num_ranks}-{self.num_agents}p"

    # ------------------------------------------------------------------
    # Env interface: observations
    # ------------------------------------------------------------------

    @property
    def legal_actions(self) -> List[str]:
        legal_moves_dict = self._jax_env.get_legal_moves(self._jax_state)
        current_agent = self._jax_env.agents[self.current_player]
        legal_mask = np.array(legal_moves_dict[current_agent])
        return [
            self._jax_env.action_encoding[i]
            for i in np.where(legal_mask)[0]
            if self._jax_env.action_encoding[i] != 'N'
        ]

    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:
        """Return the current player's text observation.

        Calls the JAX env's get_obs_str() which produces a human-readable
        description including the masked own hand, visible other hands,
        board state, last action details, and legal actions.
        """
        player_id = self.current_player
        obs_str = self._jax_env.get_obs_str(
            new_state=self._jax_state,
            old_state=self._prev_jax_state,
            action=self._last_action_int,
            include_belief=False,
        )
        action_instructions = (
            "\n## How to respond\n\n"
            "You must choose **exactly one** action from the Legal Actions list above.\n\n"
            "Rules for your response:\n"
            "- Copy the action string **verbatim** — character for character, including spaces.\n"
            "- Hint actions contain spaces, e.g. `HR to P1 relative` or `H3 to P1 relative`. "
            "Do **not** abbreviate, paraphrase, or reformat them.\n"
            "- Play and discard actions are two characters, e.g. `P0`, `D3`.\n"
            "- Do **not** invent actions that are not in the Legal Actions list.\n"
        )

        obs_str = f"You are player_{player_id}.\n\n" + obs_str + action_instructions
        return player_id, obs_str, self.legal_actions

    @property
    def dict_observation(self) -> Tuple[int, Dict, List[str]]:
        player_id = self.current_player
        state = self._jax_state

        # Fireworks: thermometer encoding → sum per color row = pile height.
        fireworks_arr = np.array(state.fireworks)
        fireworks = {
            self._jax_env.color_map[c]: int(fireworks_arr[c].sum())
            for c in range(self._jax_env.num_colors)
        }

        # My hand: only hints are visible (own cards are hidden).
        colors_revealed = np.array(state.colors_revealed[player_id])
        ranks_revealed = np.array(state.ranks_revealed[player_id])
        my_hand_hints = []
        for card_idx in range(self._jax_env.hand_size):
            ch = colors_revealed[card_idx]
            rh = ranks_revealed[card_idx]
            my_hand_hints.append({
                "color_hint": self._jax_env.color_map[int(np.argmax(ch))] if ch.any() else None,
                "rank_hint": int(np.argmax(rh)) + 1 if rh.any() else None,
            })

        # Other players' hands: visible cards.
        other_hands = {}
        for aidx in range(self._jax_env.num_agents):
            if aidx == player_id:
                continue
            hand = np.array(state.player_hands[aidx])
            other_hands[f"player_{aidx}"] = [
                self._jax_env.card_to_string(hand[c])
                for c in range(self._jax_env.hand_size)
            ]

        obs_dict = {
            "player_id": player_id,
            "info_tokens": int(np.array(state.info_tokens).sum()),
            "life_tokens": int(np.array(state.life_tokens).sum()),
            "fireworks": fireworks,
            "deck_size": int(np.array(state.deck).sum()),
            "my_hand_hints": my_hand_hints,
            "other_hands": other_hands,
            "last_action": self._jax_env.action_encoding[self._last_action_int],
            "turn": int(state.turn),
            "score": int(state.score),
        }
        return player_id, obs_dict, self.legal_actions

    @property
    def dict_observation_description(self) -> str:
        return _build_dict_obs_description(self.num_colors)

    @property
    def possible_actions_description(self) -> str:
        return _build_actions_description(self.num_colors, self.num_ranks)

    @property
    def base_game_description(self) -> str:
        return _build_game_description(self.num_colors, self.num_ranks, self.max_info_tokens, self.max_life_tokens)

    @property
    def example_text_observations(self) -> Dict[str, str]:
        colors = _ALL_COLORS[:self.num_colors]
        possible_str = "".join(colors) + "".join(str(r) for r in range(1, self.num_ranks + 1))
        hand_size = self._jax_env.hand_size
        hand_lines = "\n".join(f"{i} Hints: , Possible: {possible_str}" for i in range(hand_size))
        other_hand_lines = "\n".join(
            f"{i} Card: {colors[i % self.num_colors]}{(i % self.num_ranks) + 1}, Hints: , Possible: {possible_str}"
            for i in range(hand_size)
        )
        discard_actions = ", ".join(f"'D{i}'" for i in range(hand_size))
        play_actions = ", ".join(f"'P{i}'" for i in range(hand_size))
        color_hint_actions = ", ".join(f"'H{c} to P1 relative'" for c in colors)
        rank_hint_actions = ", ".join(f"'H{r} to P1 relative'" for r in range(1, self.num_ranks + 1))
        legal_actions_str = f"[{discard_actions}, {play_actions}, {color_hint_actions}, {rank_hint_actions}]"
        return {
            "player_0_template": (
                f"You are player_0.\n\n"
                f"Turn: 1\n\nScore: 0\nInformation available: 8\nLives available: 3\n"
                f"Deck remaining cards: {self._jax_env.deck_size - self.num_agents * hand_size}\n"
                f"Discards: \nFireworks: \n"
                f"Your Hand:\n{hand_lines}\n"
                f"Other Hand:\n{other_hand_lines}\n"
                f"Last action: N\n"
                f"Legal Actions: {legal_actions_str}\n"
            ),
        }

    # ------------------------------------------------------------------
    # Env interface: rollout history
    # ------------------------------------------------------------------

    def get_compact_rollout_history(self, ignore_not_done: bool = False) -> CompactRolloutHistory:
        if not self.done and not ignore_not_done:
            raise ValueError(
                "Please finish the game before calling get_compact_rollout_history()!"
            )

        final_score = int(self._jax_state.score) if self._jax_state is not None else 0
        out_of_lives = bool(self._jax_state.out_of_lives) if self._jax_state is not None else False

        # compact_states: one entry per step — only (player, action) are needed to replay
        # the game from scratch given the seed; reward/score/turn are kept for analysis.
        # To replay: HanabiTextEnv(...).reset(seed=metadata["seed"]), then step through actions.
        return CompactRolloutHistory(
            hyperparams={"num_agents": self.num_agents, "num_colors": self.num_colors, "num_ranks": self.num_ranks, "max_life_tokens": self.max_life_tokens},
            metadata={
                "seed": self._seed,
                "final_score": final_score,
                "out_of_lives": out_of_lives,
                "total_turns": int(self._jax_state.turn) if self._jax_state is not None else 0,
            },
            compact_states=list(self._step_history),
            rewards_sum=self._cumulative_reward,
            terminated=self._terminated_flag,
            truncated=self._truncated_flag,
        )
