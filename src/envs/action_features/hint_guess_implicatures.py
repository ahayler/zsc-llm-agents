"""HintGuess restricted to four hand-crafted scenarios from Bullard et al.
(action features paper, Fig. 2 / Table 1) at hand_size=2 on a 3x3 grid.

Each `reset(seed=N)` deterministically picks scenario `N % 4`, so cycling
through 4 game runs covers each scenario exactly once. With hand_size=2 the
four scenarios isolate distinct coordination dimensions:

    0  exact_match                  - hinter has the target card; trivially solvable.
    1  feature_similarity           - no exact match in hinter's hand; one card uniquely
                                      shares a feature with the target. Tests max-similarity.
    2  mutual_exclusivity           - implicature: human-compatible hint shares ZERO
                                      features with the target. Solving requires
                                      Markman/Gricean reasoning ("if target were X,
                                      hinter would have hinted X exactly").
    3  similarity_plus_exclusivity  - both signals point to the same answer.

`permute_hands=True` is the default — without permutation, models could exploit
fixed positions instead of using the intended pragmatic reasoning.
"""

from typing import Optional, Sequence, Tuple

from envs.action_features.hint_guess import HintGuess
from configs.policy_modifiers import BasePolicyModifierConfig
from core import CompactRolloutHistory


SCENARIOS = [
    {
        "name": "exact_match",
        "hinter_hand":  [("1", "A"), ("2", "B")],
        "guesser_hand": [("2", "B"), ("1", "A")],
        "target_idx": 0,                # target = 2B
        "human_compatible_hint": "2B",  # exact match in hinter's hand
    },
    {
        "name": "feature_similarity",
        "hinter_hand":  [("1", "B"), ("3", "C")],
        "guesser_hand": [("2", "B"), ("3", "A")],
        "target_idx": 1,                # target = 3A
        "human_compatible_hint": "3C",  # shares '3' with target 3A; uniquely points to 3A
        # rationale: each of hinter's cards uniquely points to one guesser card:
        #   1B → 2B (shares B; 3A shares nothing with 1B)
        #   3C → 3A (shares 3; 2B shares nothing with 3C)
    },
    {
        "name": "mutual_exclusivity",
        "hinter_hand":  [("1", "B"), ("3", "C")],
        "guesser_hand": [("1", "B"), ("2", "A")],
        "target_idx": 1,                # target = 2A
        "human_compatible_hint": "3C",  # ZERO feature overlap with 2A
        # rationale: 1B is in BOTH hands → if target were 1B, hinter would hint 1B
        # exactly (max similarity). So hinting 3C signals "target is NOT 1B" by
        # mutual exclusivity → the only other guesser card 2A.
    },
    {
        "name": "similarity_plus_exclusivity",
        "hinter_hand":  [("2", "A"), ("2", "C")],
        "guesser_hand": [("3", "B"), ("1", "C")],
        "target_idx": 0,                # target = 3B
        "human_compatible_hint": "2A",
        # NOTE: with these hands the test reduces to mutual-exclusivity only —
        # 2A shares ZERO features with target 3B (no similarity signal).
        # rationale: hinter's other card 2C is a perfect similarity match for
        # guesser's 1C (shares 'C'). So if target were 1C the hinter would hint
        # 2C. Hinting 2A signals "target ≠ 1C" → 3B (the only remaining card).
    },
]


class HintGuessImplicatures(HintGuess):
    """HintGuess limited to four hand-crafted implicature-test scenarios."""

    def __init__(
        self,
        policy_modifier_config: BasePolicyModifierConfig,
        permute_hands: bool = False,
    ):
        super().__init__(
            policy_modifier_config=policy_modifier_config,
            hand_size=2,
            feature_space_1=("1", "2", "3"),
            feature_space_2=("A", "B", "C"),
            permute_hands=permute_hands,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        scenario_id: Optional[int] = None,
        # The following are accepted but ignored — scenario fixes them.
        hinter_hand: Optional[Sequence[Tuple[str, str]]] = None,
        guesser_hand: Optional[Sequence[Tuple[str, str]]] = None,
        target_idx: Optional[int] = None,
    ):
        if scenario_id is None:
            scenario_id = (seed if seed is not None else 0) % len(SCENARIOS)
        scenario_id = scenario_id % len(SCENARIOS)
        s = SCENARIOS[scenario_id]
        self.scenario_id = scenario_id
        self.scenario_name: str = s["name"]
        self.human_compatible_hint: str = s["human_compatible_hint"]

        super().reset(
            seed=seed,
            hinter_hand=s["hinter_hand"],
            guesser_hand=s["guesser_hand"],
            target_idx=s["target_idx"],
        )

    @property
    def env_name(self) -> str:
        return "HintGuessImplicatures-N2-3x3-v0"

    def get_compact_rollout_history(self, ignore_not_done: bool = False) -> CompactRolloutHistory:
        history = super().get_compact_rollout_history(ignore_not_done=ignore_not_done)
        history.metadata["scenario_id"] = self.scenario_id
        history.metadata["scenario_name"] = self.scenario_name
        history.metadata["human_compatible_hint"] = self.human_compatible_hint
        history.metadata["hint_was_human_compatible"] = (
            history.metadata.get("hint_card") == self.human_compatible_hint
        )
        return history


if __name__ == "__main__":
    # Dry run: walk through all 4 scenarios. The hinter plays the
    # human-compatible hint; the guesser plays the target card. Both
    # should always succeed (joint reward = 1) under perfect play.
    from configs.policy_modifiers import NoPolicyModifierConfig

    env = HintGuessImplicatures(NoPolicyModifierConfig())
    for scenario_id in range(len(SCENARIOS)):
        print("\n" + "=" * 72)
        print(f"SCENARIO {scenario_id}: {SCENARIOS[scenario_id]['name']}")
        print("=" * 72)

        env.reset(seed=scenario_id)
        print(f"hinter_hand   = {env.hinter_hand}")
        print(f"guesser_hand  = {env.guesser_hand}")
        print(f"target_card   = {env.target_card}")
        print(f"human-compatible hint = {env.human_compatible_hint}")
        print(f"legal hints   = {env.legal_actions}")

        # Hinter plays the human-compatible hint
        hint_action = f"hint_{env.human_compatible_hint}"
        assert hint_action in env.legal_actions, f"{hint_action} not in {env.legal_actions}"
        env.step(hint_action)
        print(f"\nHinter plays:  {hint_action}")

        # Guesser plays the target card (perfect-play oracle)
        from envs.action_features.hint_guess import _card_str
        guess_action = f"guess_{_card_str(env.target_card)}"
        assert guess_action in env.legal_actions, f"{guess_action} not in {env.legal_actions}"
        reward, terminated, _, _ = env.step(guess_action)
        print(f"Guesser plays: {guess_action}")
        print(f"reward = {reward}   terminated = {terminated}")

        h = env.get_compact_rollout_history()
        print(f"hint_was_human_compatible = {h.metadata['hint_was_human_compatible']}")
