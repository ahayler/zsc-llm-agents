"""Run on a remote node: PYTHONPATH=src python -m unittest discover -s tests."""
import unittest
from types import SimpleNamespace

import numpy as np

from utils.metrics import paper_failure_report


def histories(*cells):
    result = np.empty((1, len(cells)), dtype=object)
    for index, rewards in enumerate(cells):
        result[0, index] = [SimpleNamespace(rewards_sum=r) for r in rewards]
    return result


class PaperFailureTests(unittest.TestCase):
    def test_zero_is_a_failure_at_zero_threshold(self):
        report = paper_failure_report("cat_and_dog", histories([-10, 0, 5]))
        self.assertEqual(report["failure_rates"], [[2 / 3]])

    def test_navigation_uses_published_failure_cutoff(self):
        for env in ("brittle_bridge", "narrow_tunnel"):
            report = paper_failure_report(env, histories([-1000, -21, -20, -14, -9]))
            self.assertEqual(report["threshold"], -20)
            self.assertEqual(report["failure_rates"], [[3 / 5]])

    def test_reveal_fallback_is_not_a_convention_failure(self):
        report = paper_failure_report("ad_hoc_conventions", histories([49, 50, 100]))
        self.assertEqual(report["failure_rates"], [[1 / 3]])

    def test_unavailable_returns_are_not_silently_counted_as_success(self):
        report = paper_failure_report("tiny_hanabi", histories([None, float("nan"), 0], []))
        self.assertEqual(report["failure_rates"], [[1.0, None]])
        self.assertEqual(report["valid_episode_counts"], [[1, 0]])
        self.assertEqual(report["missing_episode_counts"], [[2, 0]])

    def test_pair_orientation_is_preserved(self):
        grid = np.empty((2, 2), dtype=object)
        for index, rewards in [((0, 0), [0]), ((0, 1), [-1]), ((1, 0), [1]), ((1, 1), [-1, 1])]:
            grid[index] = [SimpleNamespace(rewards_sum=r) for r in rewards]
        self.assertEqual(paper_failure_report("matrix_game", grid)["failure_rates"], [[1, 1], [0, 0.5]])

    def test_no_invented_threshold_for_hanabi(self):
        self.assertIsNone(paper_failure_report("hanabi_3x3_1life", histories([0, 9])))


if __name__ == "__main__":
    unittest.main()
