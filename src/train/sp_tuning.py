import time, math
from concurrent.futures import ThreadPoolExecutor
from train.critic import PolicyCritic, CriticHistory
from utils.tracing import langfuse_client
from langfuse import observe
from opentelemetry.trace import get_current_span
from langfuse import LangfuseOtelSpanAttributes

import numpy as np

from abc import ABC, abstractmethod
from configs.configs import *
from core import CodePolicy
from envs.base import Env
from typing import Optional, List, Dict

from train.prompt_generation import DIVERSE_INSTRUCTIONS_POLICY_GENERATOR
from train.policy_execution import run_code_policies_sequential

from train.policy_generator import CodePolicyGenerator

from train.xp_eval import CodeXPEvaluator

from utils.env_impl import get_env_name_file_save_from_config

from dotenv import load_dotenv
load_dotenv()

from train.utils import get_env_from_config
from utils.tracing import langfuse_client

from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from llm.request import create_client
from llm.parser import request_and_parse_to_json
from core import Message
from train.utils import code_policy_to_str
from utils.log_utils import get_logger

logger = get_logger(__name__)


_REPAIR_POLICY_PROMPT = """You are fixing a broken game policy. The policy crashed during execution with the following error.

**Policy Code:**
```python
{policy_code}
```

**Error:** {error}

**Observation at time of crash:** {observation}

**Game Description:**
{game_description}

**Policy Definition:**
{policy_definition}

Fix the bug in the policy. The fix should be minimal — only change what's necessary to prevent the crash.
Common issues include: accessing list indices out of range, missing key checks, division by zero, or incorrect assumptions about observation structure.

Respond with valid JSON:
```json
{{
    "reasoning": "Explain what caused the crash and how you fixed it",
    "fixed_policy": {{
        "name": "{policy_name}",
        "reasoning": "Fixed version",
        "player_0": {{
            "code": "<fixed python code for player_0>"
        }},
        "player_1": {{
            "code": "<fixed python code for player_1>"
        }}
    }}
}}
```"""
    

# TODO: Would need to be ported to the newest version of Langfuse v3
# @contextmanager
# def mute_langfuse():
#     langfuse_context.configure(enabled=False)
#     try:
#         yield
#     finally:
#         langfuse_context.configure(enabled=True)
    
class AbstractPolicyTrainer(ABC):
    @abstractmethod
    def train(self):
        raise NotImplementedError

class EvoCodePolicyTrainer(AbstractPolicyTrainer):
    """
    This class implements the training process for code based policies based on evolutionary algorithm.
    """
    def __init__(
        self,
        config: GeneticCodeTrainerConfig,
        executor: ThreadPoolExecutor,
        langfuse_name: Optional[str] = None,
        general_instructions: Optional[str] = DIVERSE_INSTRUCTIONS_POLICY_GENERATOR,
        rng_seed: Optional[int] = None,
    ):
        self.config: GeneticCodeTrainerConfig = config
        self.env_config: EnvConfig = config.env_config
        self.config.invalid_action_xp_evaluator_config.env_config = self.env_config
        self.executor = executor
        self.general_instructions = general_instructions
        self.generator = CodePolicyGenerator(self.config.generator_config)
        self.langfuse_name = langfuse_name if langfuse_name is not None else f"{get_env_name_file_save_from_config(self.env_config)}_training"
        self.rng = np.random.default_rng(rng_seed)

    @observe()
    def train(self, preset_policies: Optional[List[CodePolicy]] = None, invalid_action_policies: Optional[List[CodePolicy]] = None) -> tuple[Dict, List[CodePolicy], List[CodePolicy], List[CodePolicy], List[CriticHistory]]:
        """
        Args:
            preset_policies: If provided, skip generation and use these as starting policies
                            for the critic. Useful for ensuring deterministic starting points.
            invalid_action_policies: If provided, use these as partners for invalid action
                                    filtering instead of self-play among generated policies.

        Returns:
            top_k_policies: List[CodePolicy]
            trained_policies: List[CodePolicy]
            generated_policies: List[CodePolicy]
            policy_critic_histories: List[CriticHistory]

            Note that generated_policies and trained_policies are lists of the same length. Index i of the two lists corresponds to the same policy.
        """
        # Fixes a bug in langfuse, where a low level observation overwrite the root span
        get_current_span().set_attribute(LangfuseOtelSpanAttributes.AS_ROOT, False)


        # Policy Generation
        start_time = time.time()
        if preset_policies is not None:
            generated_policies = preset_policies
            logger.info(f"Using {len(generated_policies)} preset policies ({self.langfuse_name})")
        else:
            generated_policies = self.generator.generate(self.env_config)
        generation_duration = time.time() - start_time
        logger.info(f"Code policy generation completed in {generation_duration:.2f} seconds for {len(generated_policies)} policies ({self.langfuse_name})")

        # This function is mainly used to get better formatting on langfuse
        @observe()
        def filter_out_invalid_action_policies(policies: List[CodePolicy], max_repair_attempts: int = 2) -> List[CodePolicy]:
            evaluator = CodeXPEvaluator(self.config.invalid_action_xp_evaluator_config)
            partner_policies = invalid_action_policies if invalid_action_policies is not None else policies

            for repair_round in range(max_repair_attempts + 1):
                xp_scores, compact_rollout_histories = evaluator.eval_grid(policies, partner_policies, no_progress_bar=True)
                nan_indices = np.where(np.isnan(xp_scores))

                # Collect invalid policies with their error info
                invalid_info = {}  # index -> (error, observation)
                for i, j in zip(*nan_indices):
                    histories = compact_rollout_histories[i, j]
                    history = histories[0]
                    idx = history.crash_cause_idx
                    if idx is not None and idx < len(policies) and idx not in invalid_info:
                        invalid_info[idx] = (history.error, history.compact_states)

                if not invalid_info:
                    break  # No broken policies

                if repair_round >= max_repair_attempts:
                    logger.info(f"Max repair attempts reached. Filtering out {len(invalid_info)} broken policies.")
                    break

                # Attempt to repair broken policies
                logger.info(f"Repair round {repair_round + 1}: attempting to fix {len(invalid_info)} broken policies")
                env: Env = get_env_from_config(self.env_config)
                client, model = create_client(self.config.policy_critic_config.llm_config.model)

                for idx, (error, compact_states) in invalid_info.items():
                    policy = policies[idx]
                    try:
                        prompt = _REPAIR_POLICY_PROMPT.format(
                            policy_code=code_policy_to_str(policy),
                            error=error or "Unknown error",
                            observation=str(compact_states) if compact_states else "N/A",
                            game_description=env.game_description,
                            policy_definition=env.dict_observation_description,
                            policy_name=policy.name,
                        )
                        json_response, _ = request_and_parse_to_json(
                            client=client,
                            model=model,
                            msg_history=[Message(role="user", content=prompt)],
                            temperature=0,
                            max_tokens=self.config.policy_critic_config.llm_config.max_tokens,
                            reasoning_effort=self.config.policy_critic_config.llm_config.reasoning_effort,
                            required_fields=["fixed_policy"],
                        )
                        from llm.parser import parse_code_policy_from_dict
                        fixed_policy = parse_code_policy_from_dict(json_response["fixed_policy"])
                        policies[idx] = fixed_policy
                        logger.info(f"Repaired policy {idx} ({policy.name})")
                    except Exception as e:
                        logger.warning(f"Failed to repair policy {idx}: {e}")

            # Final filter: remove any still-broken policies
            invalid_indices = set(invalid_info.keys()) if invalid_info else set()
            filtered_policies = [policy for i, policy in enumerate(policies) if i not in invalid_indices]

            return filtered_policies

        # Optional pre-filtering
        if self.config.use_invalid_action_filtering_for_generated_policies:
            _n_policies_before_filtering = len(generated_policies)
            generated_policies = filter_out_invalid_action_policies(generated_policies)
            _n_policies_after_filtering = len(generated_policies)
            logger.info(f"Code policy invalid action filtering completed in {time.time() - start_time:.2f} seconds. Filtered out {_n_policies_before_filtering - _n_policies_after_filtering} of the generated policies ({self.langfuse_name}).")

        # Policy Critic
        start_time = time.time()
        if self.config.use_policy_critic:
            env: Env = get_env_from_config(self.env_config)
            policy_critic = PolicyCritic(self.config.policy_critic_config, env, self.executor)
            policy_critic_histories = policy_critic.find_coordination_friendly_policies(generated_policies, env.game_description)
            coordination_friendly_policies = [history.final_policy for history in policy_critic_histories if history.final_policy is not None]
        else:
            coordination_friendly_policies = generated_policies
            policy_critic_histories = None

        policy_critic_duration = time.time() - start_time
        logger.info(f"Code policy critic completed in {policy_critic_duration:.2f} seconds for {len(coordination_friendly_policies)} policies ({self.langfuse_name})")

        # Filtering out policies that produce invalid actions
        _n_policies_before_filtering = len(coordination_friendly_policies)
        start_time = time.time()
        coordination_friendly_policies = filter_out_invalid_action_policies(coordination_friendly_policies)
        _n_policies_after_filtering = len(coordination_friendly_policies)
        end_time = time.time()
        logger.info(f"Code policy invalid action filtering completed in {end_time - start_time:.2f} seconds. Filtered out {_n_policies_before_filtering - _n_policies_after_filtering} coordination-friendly policies ({self.langfuse_name}).")

        # SP-eval
        if self.config.sp_filtering:
            start_time = time.time()
            trained_policies = run_code_policies_sequential(coordination_friendly_policies, self.config, self.env_config)
            # shuffle deterministically using the per-trainer RNG
            shuffled_indices = self.rng.permutation(len(trained_policies))
            trained_policies = [trained_policies[i] for i in shuffled_indices]
            # filter out policies with nan avg_reward
            trained_policies = [policy for policy in trained_policies if not (math.isnan(policy.avg_reward) or policy.avg_reward is None)]
            top_k_policies: list[CodePolicy] = sorted(trained_policies, key=lambda policy: policy.avg_reward, reverse=True)[:self.config.top_k]
            sp_eval_duration = time.time() - start_time
            logger.info(f"Code policy SP-eval completed in {sp_eval_duration:.2f} seconds for {len(trained_policies)} policies ({self.langfuse_name})")
        else:
            sp_eval_duration = 0
            trained_policies = coordination_friendly_policies
            # sample deterministically without replacement
            top_k_policies = list(self.rng.choice(coordination_friendly_policies, size=self.config.top_k, replace=False))
            logger.info(f"SP-filtering disabled. Returning all {len(coordination_friendly_policies)} coordination-friendly policies in random order ({self.langfuse_name}).")

        _metadata = {
            "runtime_seconds": {
                "generation": generation_duration,
                "policy_critic": policy_critic_duration,
                "sp_eval": sp_eval_duration,
            },
            "top_k_policies": top_k_policies,
            "trained_policies": trained_policies,
            "trainer_config": self.config,
        }

        langfuse_client.update_current_span(
            name=self.langfuse_name,
            metadata=_metadata
        )

        return top_k_policies, trained_policies, generated_policies, policy_critic_histories
                

        
