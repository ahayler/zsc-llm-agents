import numpy as np
import math

from dataclasses import dataclass
from configs.paper import FAILURE_THRESHOLDS


def paper_failure_report(env_name: str, histories: np.ndarray):
    """Per-pair episode failure rates; unavailable returns are reported separately.

    No threshold is published for the larger games, so return None for those.
    An episode is a failure if its return is at or below the threshold (paper rule).
    """
    if env_name not in FAILURE_THRESHOLDS:
        return None
    threshold = FAILURE_THRESHOLDS[env_name]
    rates = np.empty(histories.shape, dtype=object)
    missing = np.zeros(histories.shape, dtype=int)
    counts = np.zeros(histories.shape, dtype=int)
    for index in np.ndindex(histories.shape):
        rewards = [h.rewards_sum for h in histories[index]]
        valid = [float(r) for r in rewards if r is not None and math.isfinite(float(r))]
        missing[index] = len(rewards) - len(valid)
        counts[index] = len(valid)
        rates[index] = sum(r <= threshold for r in valid) / len(valid) if valid else None
    return {
        "threshold": threshold,
        "comparison": "return <= threshold",
        "failure_rates": rates.tolist(),
        "valid_episode_counts": counts.tolist(),
        "missing_episode_counts": missing.tolist(),
    }

@dataclass
class XPSummaryStatistics:
    sp_scores_mean: float
    sp_scores_std: float
    sp_scores_std_err: float
    xp_scores_mean: float
    xp_scores_std: float
    xp_scores_std_err: float
    sp_scores_min: float
    xp_scores_min: float

    def __str__(self):
        return f"SP scores mean: {self.sp_scores_mean:.2f}, std: {self.sp_scores_std:.2f}\nXP scores mean: {self.xp_scores_mean:.2f}, std: {self.xp_scores_std:.2f}"

def compute_xp_summary_statistics(xp_scores: np.ndarray) -> XPSummaryStatistics:
    sp_games = xp_scores.diagonal()

    if len(sp_games) == 0:
        sp_scores_mean = math.nan
        sp_scores_std = math.nan
        sp_scores_std_err = math.nan
        sp_scores_min = math.nan
    else:
        sp_scores_mean = np.mean(sp_games)
        sp_scores_std = np.std(sp_games)
        sp_scores_std_err = np.std(sp_games) / np.sqrt(len(sp_games))
        sp_scores_min = np.min(sp_games)
    
    # XP is on the off-diagonal
    xp_games = xp_scores[~np.eye(xp_scores.shape[0], dtype=bool)]

    if len(xp_games) == 0:
        xp_scores_mean = math.nan
        xp_scores_std = math.nan
        xp_scores_std_err = math.nan
        xp_scores_min = math.nan
    else:
        xp_scores_mean = np.mean(xp_games)
        xp_scores_std = np.std(xp_games)
        xp_scores_std_err = np.std(xp_games) / np.sqrt(len(xp_games))
        xp_scores_min = np.min(xp_games)    

    return XPSummaryStatistics(sp_scores_mean, sp_scores_std, sp_scores_std_err, xp_scores_mean, xp_scores_std, xp_scores_std_err, sp_scores_min, xp_scores_min)
