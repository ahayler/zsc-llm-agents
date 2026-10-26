from configs.configs import CodePolicyGeneratorConfig, GeneratorConfig
from core import CodePolicy
from typing import List
import copy

from langfuse import observe
from train.prompt_generation import generate_code_policy_generation_prompt, generate_code_policy_self_critic_prompt
from llm.parser import request_and_parse_to_json, parse_code_policy_from_dict
from llm.request import create_client
from core import Message
from envs.base import Env
from configs.envs import EnvConfig
from utils.log_utils import get_logger
from utils.saving import load_most_recent_experiment_results_for_env
import numpy as np
from train.utils import get_env_from_config

logger = get_logger(__name__)
    
class CodePolicyGenerator:
    def __init__(self, config: CodePolicyGeneratorConfig):
        self.config = config
        self.llm_config = config.llm_config
        self.num_policies = config.num_policies
        self.num_players = config.num_players
        self.use_cached_generations = config.use_cached_generations
        self.cached_generations_config = config.cached_generations_config
        self.randomly_swap_local_policies = config.randomly_swap_local_policies
        self.seed = config.seed
        
        # New params
        self.n_generations_per_seed = getattr(config, "n_generations_per_seed", 1)
        self.use_self_critic = getattr(config, "use_self_critic", False)

        if self.use_cached_generations:
            cached_experiment_results = load_most_recent_experiment_results_for_env(self.cached_generations_config.experiment_name, self.cached_generations_config.env_name_file_safe, mode="code")

            if cached_experiment_results.generated_policies is None:
                raise ValueError(f"Cached generation experiments {self.cached_generations_config.experiment_name} did not cache any generated policies for {self.cached_generations_config.env_name_file_safe}")
            self.cached_policies_list = []
            for value in cached_experiment_results.generated_policies.values():
                self.cached_policies_list.extend(value)

            if len(self.cached_policies_list) < self.num_policies:
                raise ValueError(f"Cached generation experiments {self.cached_generations_config.experiment_name} cached {len(self.cached_policies_list)} policies, but {self.num_policies} are needed for {self.cached_generations_config.env_name_file_safe}")

        self.llm_client, self.model = create_client(self.llm_config.model)

    @observe()
    def generate_from_cached_policies(self) -> list[CodePolicy]:
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.choice(self.cached_policies_list, size=self.num_policies, replace=False).tolist()

    def randomly_swap_local_policy(self, policy: CodePolicy) -> CodePolicy:
        """
        Randomly swaps the local policies of a code policy.
        """
        if len(policy.joint_policy) != 2:
            raise ValueError(f"Random local swapping only works for exactly 2 local policies")

        if np.random.random() < 0.5:
            return policy
        
        new_policy = copy.deepcopy(policy)
        new_policy.joint_policy[0], new_policy.joint_policy[1] = new_policy.joint_policy[1], new_policy.joint_policy[0]

        if new_policy.name.endswith("_SWAPPED"):
            new_policy.name = new_policy.name[:-len("_SWAPPED")]
        else:
            new_policy.name = f"{new_policy.name}_SWAPPED"

        return new_policy

    @observe()
    def generate(self, env_config: EnvConfig, cfd: str=None) -> list[CodePolicy]:
        env = get_env_from_config(env_config)
        env_name = env.env_name

        if self.use_cached_generations:
            raw_policies = self.generate_from_cached_policies()
        else:
            raw_policies = self.generate_using_llm(env_config, cfd=cfd)
        
        if self.randomly_swap_local_policies:
            return [self.randomly_swap_local_policy(policy) for policy in raw_policies]
        else:
            return raw_policies
    
    @observe()
    def generate_using_llm(self, env_config: EnvConfig, cfd: str=None) -> list[CodePolicy]:
        env: Env = get_env_from_config(env_config)
        final_policies = []
        
        prompt_cfd = cfd
        if self.use_self_critic and prompt_cfd:
            prompt_cfd += "\n\n**Diversity:**\nGenerate diverse policies that are strictly within the CFD."

        for _ in range(self.num_policies):
            # Step 1: Generate candidates
            generation_str = generate_code_policy_generation_prompt(
                game_description=env.game_description,
                num_policies=self.n_generations_per_seed,
                num_players=self.num_players,
                dict_observation_description=env.dict_observation_description,
                possible_actions_description=env.possible_actions_description,
                cfd=prompt_cfd
            )

            json_response, _ = request_and_parse_to_json(
                msg_history=[Message(role="user", content=generation_str)],
                client=self.llm_client,
                model=self.model,
                max_tokens=self.llm_config.max_tokens,
                temperature=self.llm_config.temperature,
                max_retries=5,
                required_fields=["joint_policies"],
                reasoning_effort=self.llm_config.reasoning_effort,
            )
            
            if not json_response or "joint_policies" not in json_response:
                logger.warning("Failed to get joint_policies from LLM response")
                continue

            instructions = json_response["joint_policies"]
            candidates = []
            for instruction in instructions:
                try: 
                    policy = parse_code_policy_from_dict(instruction)
                    candidates.append(policy)
                except Exception as e:
                    logger.warning(f"Error parsing code policy: {e}, skipping...")
                    logger.warning(f"Faulty dict: {instruction}")
                    continue
            
            if not candidates:
                logger.warning("No valid policies parsed for this seed.")
                continue

            # Step 2: Select one policy
            selected_policy = None
            if len(candidates) == 1:
                selected_policy = candidates[0]
            else:
                if self.use_self_critic:
                    # Self-critic selection
                    selected_policy = self.select_best_policy_with_critic(candidates, cfd, env.game_description)
                else:
                    # Take the first one (as per instructions: "take the first seed")
                    selected_policy = candidates[0]
            
            if selected_policy:
                final_policies.append(selected_policy)

        return final_policies

    @observe()
    def select_best_policy_with_critic(self, policies: List[CodePolicy], cfd: str, game_description: str) -> CodePolicy:
        prompt = generate_code_policy_self_critic_prompt(
            policies=policies,
            cfd=cfd if cfd else "No explicit CFD provided.",
            game_description=game_description
        )
        
        json_response, _ = request_and_parse_to_json(
            msg_history=[Message(role="user", content=prompt)],
            client=self.llm_client,
            model=self.model,
            max_tokens=self.llm_config.max_tokens,
            temperature=self.llm_config.temperature,
            max_retries=3,
            required_fields=["selected_policy_id"],
            reasoning_effort=self.llm_config.reasoning_effort,
        )
        
        try:
            selected_idx = int(json_response["selected_policy_id"])
            if 0 <= selected_idx < len(policies):
                return policies[selected_idx]
            else:
                logger.warning(f"Critic returned out-of-bounds index {selected_idx} for {len(policies)} policies. Defaulting to 0.")
                return policies[0]
        except Exception as e:
            logger.warning(f"Error parsing critic response: {e}. Defaulting to 0.")
            return policies[0]
