from envs.base import Env
import numpy as np
from typing import Optional, Tuple, List, Dict, Sequence
from core import CompactRolloutHistory
from configs.policy_modifiers import BasePolicyModifierConfig


game_description_template = """
You are playing the Hint-Guess game with one other player. One of you is the Hinter and the other is the Guesser.

Both players are dealt a hand of {hand_size} cards. Each card has two features: a value from {feature_space_1} and a value from {feature_space_2}. Cards are written as the concatenation of their two features (e.g. "{example_card}"). Cards are drawn independently and uniformly with replacement, so duplicates within a hand are possible. If two cards in the same hand share both features they are effectively the same card.

Both hands are public information — the Hinter and the Guesser can both see both hands. Before the game starts, one of the Guesser's cards is chosen uniformly at random to be the target card. The target's identity is revealed to the Hinter only.

The game has two rounds:
1. The Hinter picks one card from the Hinter's own hand and shows it to the Guesser as a hint.
2. The Guesser picks one card from the Guesser's own hand as its guess for the target.

Both players receive a joint reward of 1 if the features of the guessed card match the features of the target card, otherwise 0.
"""


observation_hinter_template = """
You are the Hinter. Your hand is: {hinter_view}.
The Guesser's hand is: {guesser_view}.
The target card (visible to you only) is: {target_card}.
Pick one card from YOUR hand to hint to the Guesser.
"""

observation_guesser_template = """
You are the Guesser. Your hand is: {guesser_view}.
The Hinter's hand is: {hinter_view}.
The Hinter hinted: {hint_card}.
Pick one card from YOUR hand as your guess for the target.
"""


dict_observation_description = """
For player_0 (Hinter), the observation dict has the following format:
{
    "hinter_hand": List[str]  # the Hinter's own hand, e.g. ["1A", "2B", "3C", "1B", "2A"] (order is the per-player view; may be permuted)
    "guesser_hand": List[str]  # the Guesser's hand (order is the per-player view; may be permuted)
    "target": str  # the target card, e.g. "2B" (only revealed to the Hinter)
}

For player_1 (Guesser), the observation dict has the following format:
{
    "hinter_hand": List[str]  # the Hinter's hand (order is the per-player view; may be permuted)
    "guesser_hand": List[str]  # the Guesser's own hand
    "hint": str  # the card the Hinter chose to hint, e.g. "2B"
}
"""

legal_actions_description = """
For player_0 (Hinter), the legal actions are one per UNIQUE card in the Hinter's hand:
- "hint_<card>" for each unique card in the Hinter's hand, e.g. "hint_2B"

For player_1 (Guesser), the legal actions are one per UNIQUE card in the Guesser's hand:
- "guess_<card>" for each unique card in the Guesser's hand, e.g. "guess_2B"

If the same card appears multiple times in a hand, it produces a single legal action — duplicate cards are equivalent.
"""


def _card_str(card: Tuple[str, str]) -> str:
    return f"{card[0]}{card[1]}"


def _format_hand(hand: Sequence[Tuple[str, str]]) -> str:
    return ", ".join(_card_str(c) for c in hand)


class HintGuess(Env):
    def __init__(
        self,
        policy_modifier_config: BasePolicyModifierConfig,
        hand_size: int = 5,
        feature_space_1: Sequence[str] = ("1", "2", "3"),
        feature_space_2: Sequence[str] = ("A", "B", "C"),
        permute_hands: bool = False,
    ):
        super().__init__(policy_modifier_config)
        self.hand_size = hand_size
        self.feature_space_1: Tuple[str, ...] = tuple(feature_space_1)
        self.feature_space_2: Tuple[str, ...] = tuple(feature_space_2)
        self.permute_hands = permute_hands

    def reset(
        self,
        seed: Optional[int] = None,
        hinter_hand: Optional[Sequence[Tuple[str, str]]] = None,
        guesser_hand: Optional[Sequence[Tuple[str, str]]] = None,
        target_idx: Optional[int] = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.hinter_hand: List[Tuple[str, str]] = (
            [tuple(c) for c in hinter_hand] if hinter_hand is not None else self._sample_hand()
        )
        self.guesser_hand: List[Tuple[str, str]] = (
            [tuple(c) for c in guesser_hand] if guesser_hand is not None else self._sample_hand()
        )

        if target_idx is None:
            target_idx = int(np.random.randint(len(self.guesser_hand)))
        self.target_idx: int = target_idx
        self.target_card: Tuple[str, str] = self.guesser_hand[self.target_idx]

        if self.permute_hands:
            self.hinter_view_perm = np.random.permutation(len(self.hinter_hand)).tolist()
            self.guesser_view_perm = np.random.permutation(len(self.guesser_hand)).tolist()
            self.hinter_view_perm_for_guesser = np.random.permutation(len(self.hinter_hand)).tolist()
            self.guesser_view_perm_for_hinter = np.random.permutation(len(self.guesser_hand)).tolist()
        else:
            self.hinter_view_perm = list(range(len(self.hinter_hand)))
            self.guesser_view_perm = list(range(len(self.guesser_hand)))
            self.hinter_view_perm_for_guesser = list(range(len(self.hinter_hand)))
            self.guesser_view_perm_for_hinter = list(range(len(self.guesser_hand)))

        self.hint_card: Optional[Tuple[str, str]] = None
        self.guess_card: Optional[Tuple[str, str]] = None

        self._current_player = 0
        self.rewards: List[float] = []
        self.actions: List[str] = []
        self._terminated = False

    def _sample_hand(self) -> List[Tuple[str, str]]:
        hand: List[Tuple[str, str]] = []
        for _ in range(self.hand_size):
            f1 = self.feature_space_1[int(np.random.randint(len(self.feature_space_1)))]
            f2 = self.feature_space_2[int(np.random.randint(len(self.feature_space_2)))]
            hand.append((f1, f2))
        return hand

    def _hinter_view(self) -> List[Tuple[str, str]]:
        return [self.hinter_hand[i] for i in self.hinter_view_perm]

    def _guesser_view_for_hinter(self) -> List[Tuple[str, str]]:
        return [self.guesser_hand[i] for i in self.guesser_view_perm_for_hinter]

    def _guesser_view(self) -> List[Tuple[str, str]]:
        return [self.guesser_hand[i] for i in self.guesser_view_perm]

    def _hinter_view_for_guesser(self) -> List[Tuple[str, str]]:
        return [self.hinter_hand[i] for i in self.hinter_view_perm_for_guesser]

    @property
    def base_game_description(self) -> str:
        example_card = f"{self.feature_space_1[0]}{self.feature_space_2[0]}"
        return game_description_template.format(
            hand_size=self.hand_size,
            feature_space_1="{" + ", ".join(self.feature_space_1) + "}",
            feature_space_2="{" + ", ".join(self.feature_space_2) + "}",
            example_card=example_card,
        )

    @property
    def legal_actions(self) -> List[str]:
        if self._current_player == 0:
            unique_cards = sorted({_card_str(c) for c in self.hinter_hand})
            return [f"hint_{c}" for c in unique_cards]
        else:
            unique_cards = sorted({_card_str(c) for c in self.guesser_hand})
            return [f"guess_{c}" for c in unique_cards]

    @property
    def text_observation(self) -> Tuple[int, str, List[str]]:
        if self._current_player == 0:
            obs = observation_hinter_template.format(
                hinter_view=_format_hand(self._hinter_view()),
                guesser_view=_format_hand(self._guesser_view_for_hinter()),
                target_card=_card_str(self.target_card),
            )
            return (0, obs, self.legal_actions)
        else:
            obs = observation_guesser_template.format(
                guesser_view=_format_hand(self._guesser_view()),
                hinter_view=_format_hand(self._hinter_view_for_guesser()),
                hint_card=_card_str(self.hint_card),
            )
            return (1, obs, self.legal_actions)

    @property
    def dict_observation(self) -> Tuple[int, Dict, List[str]]:
        if self._current_player == 0:
            obs = {
                "hinter_hand": [_card_str(c) for c in self._hinter_view()],
                "guesser_hand": [_card_str(c) for c in self._guesser_view_for_hinter()],
                "target": _card_str(self.target_card),
            }
            return (0, obs, self.legal_actions)
        else:
            obs = {
                "hinter_hand": [_card_str(c) for c in self._hinter_view_for_guesser()],
                "guesser_hand": [_card_str(c) for c in self._guesser_view()],
                "hint": _card_str(self.hint_card),
            }
            return (1, obs, self.legal_actions)

    @property
    def dict_observation_description(self) -> str:
        return dict_observation_description

    @property
    def possible_actions_description(self) -> str:
        return legal_actions_description

    @property
    def example_text_observations(self) -> Dict[str, str]:
        return {
            "observation_player_0": observation_hinter_template,
            "observation_player_1": observation_guesser_template,
        }

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return False

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def env_name(self) -> str:
        return f"HintGuess-N{self.hand_size}-{len(self.feature_space_1)}x{len(self.feature_space_2)}-v0"

    def _parse_card_token(self, token: str) -> Tuple[str, str]:
        for f1 in self.feature_space_1:
            if token.startswith(f1):
                f2 = token[len(f1):]
                if f2 in self.feature_space_2:
                    return (f1, f2)
        raise ValueError(f"Could not parse card token '{token}' against feature spaces {self.feature_space_1} x {self.feature_space_2}")

    def step(self, action: str) -> Tuple[float, bool, bool, Dict]:
        self.assert_not_done()
        if action not in self.legal_actions:
            raise ValueError(f"Invalid action: {action}. Legal actions: {self.legal_actions}")

        self.actions.append(action)

        if self._current_player == 0:
            card_token = action[len("hint_"):]
            self.hint_card = self._parse_card_token(card_token)
            reward = 0.0
            self._current_player = 1
        else:
            card_token = action[len("guess_"):]
            self.guess_card = self._parse_card_token(card_token)
            reward = 1.0 if self.guess_card == self.target_card else 0.0
            self._terminated = True

        self.rewards.append(reward)
        return reward, self._terminated, False, {}

    def get_compact_rollout_history(self, ignore_not_done: bool = False) -> CompactRolloutHistory:
        if not self.done and not ignore_not_done:
            raise ValueError("Environment is not done yet!")

        metadata = {
            "hinter_hand": [_card_str(c) for c in self.hinter_hand],
            "guesser_hand": [_card_str(c) for c in self.guesser_hand],
            "target_idx": self.target_idx,
            "target_card": _card_str(self.target_card),
            "hint_card": _card_str(self.hint_card) if self.hint_card is not None else None,
            "guess_card": _card_str(self.guess_card) if self.guess_card is not None else None,
            "hinter_view_perm": self.hinter_view_perm,
            "guesser_view_perm": self.guesser_view_perm,
            "hinter_view_perm_for_guesser": self.hinter_view_perm_for_guesser,
            "guesser_view_perm_for_hinter": self.guesser_view_perm_for_hinter,
        }

        hyperparams = {
            "hand_size": self.hand_size,
            "feature_space_1": list(self.feature_space_1),
            "feature_space_2": list(self.feature_space_2),
            "permute_hands": self.permute_hands,
        }

        return CompactRolloutHistory(
            metadata=metadata,
            hyperparams=hyperparams,
            rewards_sum=sum(self.rewards),
            terminated=self._terminated,
            truncated=False,
            compact_states=[{
                "hint": _card_str(self.hint_card) if self.hint_card is not None else None,
                "guess": _card_str(self.guess_card) if self.guess_card is not None else None,
            }],
        )


if __name__ == "__main__":
    from configs.policy_modifiers import NoPolicyModifierConfig
    env = HintGuess(NoPolicyModifierConfig())
    env.reset(seed=0)
    print("=== Hinter observation ===")
    print(env.text_observation)
    hint_action = env.legal_actions[0]
    print(f"Hinter plays: {hint_action}")
    env.step(hint_action)
    print("=== Guesser observation ===")
    print(env.text_observation)
    guess_action = env.legal_actions[0]
    print(f"Guesser plays: {guess_action}")
    env.step(guess_action)
    print("=== Compact rollout ===")
    print(env.get_compact_rollout_history())

    env.reset(seed=42)
    print("=== Hinter observation ===")
    print(env.text_observation)
    hint_action = env.legal_actions[0]
    print(f"Hinter plays: {hint_action}")
    env.step(hint_action)
    print("=== Guesser observation ===")
    print(env.text_observation)
    guess_action = env.legal_actions[0]
    print(f"Guesser plays: {guess_action}")
    env.step(guess_action)
    print("=== Compact rollout ===")
    print(env.get_compact_rollout_history())
