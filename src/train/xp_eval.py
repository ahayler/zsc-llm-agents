from configs.configs import CodeXPEvaluatorConfig
from core import CodePolicy, CompactRolloutHistory
from train.policy_execution import run_code_policy
from configs.envs import EnvConfig
from langfuse import observe
from typing import List, Tuple, Optional
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import numpy as np

class CodeXPEvaluator():
    def __init__(self, config: CodeXPEvaluatorConfig):
        self.config = config
        self.env_config: EnvConfig = config.env_config

    def eval_pair(self, policy_1: CodePolicy, policy_2: CodePolicy, num_rollouts: Optional[int] = None) -> Tuple[float, List[CompactRolloutHistory]]:
        """
        Runs an XP evaluation between a pair of policies.

        Args:
            policy_1: The first policy to evaluate.
            policy_2: The second policy to evaluate.
            num_rollouts: The number of rollouts to run for the XP evaluation. If None, the number of rollouts is set to the number of rollouts in the config.

        Returns:
            The XP score and the compact rollout histories.
        """
        config = self.config.model_copy()
        if num_rollouts is not None:
            config.num_game_runs = num_rollouts

        joint = [policy_1.joint_policy[0], policy_2.joint_policy[1]]
        xp_policy = CodePolicy(
            name=f"{policy_1.name} vs {policy_2.name}",
            joint_policy=joint
        )
        result_policy, _compact_rollout_histories = run_code_policy(config, self.env_config, xp_policy)

        return result_policy.avg_reward, _compact_rollout_histories

    def eval_grid(self, policies_rows: List[CodePolicy], policies_cols: List[CodePolicy], no_progress_bar: bool = False):
        n, m = len(policies_rows), len(policies_cols)
        xp_scores = np.zeros((n, m))
        compact_rollout_histories = np.empty((n, m), dtype=object)
        total_evals = n * m

        def inner_loop(i, j):
            xp_score, _compact_rollout_histories = self.eval_pair(policies_rows[i], policies_cols[j])
            xp_scores[i, j] = xp_score
            compact_rollout_histories[i, j] = _compact_rollout_histories

        if no_progress_bar:
            for i in range(n):
                for j in range(m):
                    inner_loop(i, j)
        else:
            with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task("[cyan]Running cross-play evaluation...", total=total_evals)
                for i in range(n):
                    for j in range(m):
                        inner_loop(i, j)
                        progress.advance(task)

        return xp_scores, compact_rollout_histories
