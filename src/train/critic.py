from typing import Tuple, List, Optional, Dict, Any
import logging
from core import Message, Policy, CodePolicy
from envs.base import Env
from llm.request import create_client
from llm.parser import request_and_parse_to_json, parse_code_policy_from_dict
from configs.configs import PolicyCriticConfig
from abc import ABC, abstractmethod
from langfuse import observe, get_client
from utils.tracing import langfuse_client
from concurrent.futures import ThreadPoolExecutor, as_completed
from train.prompt_generation import (
    generate_code_policy_critic_prompt,
    reformulate_prompt,
    POLICY_CRITIC_GENERAL_INSTRUCTIONS,
    POLICY_CRITIC_GENERAL_INSTRUCTIONS_JOINT,
)
from utils.log_utils import get_logger
from configs.configs import CriticOutput, CriticHistory
from opentelemetry.trace import get_current_span
from langfuse import LangfuseOtelSpanAttributes


logger = get_logger(__name__)

class PolicyCritic:
    def __init__(self, config: PolicyCriticConfig, env: Env, executor: Optional[ThreadPoolExecutor] = None):
        self.llm_config = config.llm_config
        self.llm_client, self.llm_model = create_client(self.llm_config.model)
        self.executor = executor
        self.coordination_friendly_definition = config.coordination_friendly_definition
        self.max_iterations = config.max_iterations
        self.require_double_approval = config.require_double_approval
        self.general_instructions = (
            POLICY_CRITIC_GENERAL_INSTRUCTIONS if config.judge_independently
            else POLICY_CRITIC_GENERAL_INSTRUCTIONS_JOINT
        )

        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        
        self.reformulate_definition = config.reformulate_definition
        self.env = env

        if self.reformulate_definition:
            self.coordination_friendly_definition = reformulate_prompt(self.coordination_friendly_definition)


    @observe()
    def find_coordination_friendly_policy_step(self, policy: Policy, game_description: str) -> CriticOutput:
        
        def critic_code_policy(policy: CodePolicy, game_description: str) -> Dict[str, Any]:
            prompt = generate_code_policy_critic_prompt(
                policy=policy,
                game_description=game_description,
                coordination_friendly_definition=self.coordination_friendly_definition,
                dict_observation_description=self.env.dict_observation_description,
                possible_actions_description=self.env.possible_actions_description,
                general_instructions=self.general_instructions,
                )
            
            json_response, _ = request_and_parse_to_json(
                msg_history=[Message(role="user", content=prompt)],
                client=self.llm_client,
                model=self.llm_model,
                max_tokens=self.llm_config.max_tokens,
                temperature=0, # We want the critic to be as deterministic as possible
                max_retries=5,
                required_fields=[
                    "coordination_friendly", 
                    "reasoning",
                    "modified_policy", 
                    "modified_policy:name", 
                    "modified_policy:player_0:code", 
                    "modified_policy:player_1:code", 
                    "modified_policy:reasoning"
                ],
                reasoning_effort=self.llm_config.reasoning_effort,
            )

            return json_response
        
        if isinstance(policy, CodePolicy):
            json_response = critic_code_policy(policy, game_description)
        else:
            raise ValueError(f"Policy type {type(policy)} not supported")

        # langfuse_client.update_current_span(
        #     version=f"SequentialPolicyCritic: {policy.name}, Coordination Friendly: {json_response['coordination_friendly']}"
        # )

        if not json_response["coordination_friendly"]:
            if isinstance(policy, CodePolicy):
                modified_policy = parse_code_policy_from_dict(json_response["modified_policy"])
        else:
            modified_policy = None

        
        critic_output = CriticOutput(
            coordination_friendly=json_response["coordination_friendly"],
            reasoning=json_response["reasoning"],
            input_policy=policy,
            modified_policy=modified_policy,
            modified_policy_reason=json_response["modified_policy"]["reasoning"]
        )

        langfuse_client.update_current_span(
            name=f"SequentialPolicyCritic: {policy.name}, Coordination Friendly: {json_response['coordination_friendly']}"
        )
            
        return critic_output
    
    @observe()
    def find_coordination_friendly_policy(self, start_policy: CodePolicy, game_description: str) -> CriticHistory:
        """
        Find a coordination-friendly policy by iteratively modifying the start policy.
        
        Args:
            start_policy: The initial policy to start from.
            game_description: The game description to use.

        Returns:
            bool: Whether a coordination-friendly policy was found.
            Optional[Policy]: The coordination-friendly policy if found, otherwise None.
        """

        get_current_span().set_attribute(LangfuseOtelSpanAttributes.AS_ROOT, False)

        def update_langfuse_observation(iteration: int, is_coordination_friendly: bool, current_policy: CodePolicy):
            langfuse_client.update_current_span(
                #name=f"find_coordination_friendly_policy: {current_policy.name}, Coordination Friendly: {is_coordination_friendly}, Iteration: {iteration}"
                name=f"CoordFriendly: {is_coordination_friendly}, Iter: {iteration}, {current_policy.name}"
            )

        critic_outputs = []

        current_policy = start_policy
        for iteration in range(self.max_iterations):
            critic_output = self.find_coordination_friendly_policy_step(current_policy, game_description)
            critic_outputs.append(critic_output)
            if critic_output.coordination_friendly:
                if not self.require_double_approval:
                    break
                else:
                    # We require double approval, so we check twice if the modified policy is coordination-friendly
                    critic_output = self.find_coordination_friendly_policy_step(current_policy, game_description)
                    critic_outputs.append(critic_output)
                    if critic_output.coordination_friendly:
                        break
                    else:
                        logger.warning(f"Double approval failed for policy {current_policy.name}.")
        
            current_policy = critic_output.modified_policy

        approved = critic_outputs[-1].coordination_friendly
        if not approved:
            logger.warning(
                f"Policy critic exhausted {self.max_iterations} iterations without approving policy {start_policy.name}. "
                f"Falling back to last modified policy."
            )

        critic_history = CriticHistory(
            critic_outputs=critic_outputs,
            found_coordination_friendly_policy=approved,
            final_policy=current_policy,
        )

        update_langfuse_observation(iteration, critic_history.found_coordination_friendly_policy, current_policy)

        return critic_history
    
    @observe()
    def find_coordination_friendly_policies_sequential(self, policies: List[CodePolicy], game_description: str) -> List[CriticHistory]:
        return [self.find_coordination_friendly_policy(policy, game_description) for policy in policies]
    
    @observe()
    def find_coordination_friendly_policies_parallel(self, policies: List[CodePolicy], game_description: str, executor: ThreadPoolExecutor) -> List[CriticHistory]:
        results = [None] * len(policies)  # Initialize with None values

        # Get current Langfuse context
        trace_id = langfuse_client.get_current_trace_id()
        observation_id = langfuse_client.get_current_observation_id()

        def find_coordination_friendly_policy_index_wrapper(idx: int, policy: CodePolicy, game_description: str, langfuse_trace_id: str, langfuse_parent_observation_id: str):
            return idx, self.find_coordination_friendly_policy(
                policy, 
                game_description,
                langfuse_parent_observation_id=langfuse_parent_observation_id,
                langfuse_trace_id=langfuse_trace_id
            )

        futures = []
        for idx, policy in enumerate(policies):
            futures.append(
                executor.submit(
                    find_coordination_friendly_policy_index_wrapper,
                    idx,
                    policy,
                    game_description,
                    langfuse_trace_id=trace_id,
                    langfuse_parent_observation_id=observation_id,
                )
            )

        for fut in as_completed(futures):
            idx, result = fut.result()
            results[idx] = result

        return results
    
    @observe()
    def find_coordination_friendly_policies(self, policies: List[CodePolicy], game_description: str) -> List[CriticHistory]:
        """
        This is the main entry point to the PolicyCritic class. 
        """
        if self.executor is None:
            results = self.find_coordination_friendly_policies_sequential(policies, game_description)
        else:
            results = self.find_coordination_friendly_policies_parallel(policies, game_description, self.executor)
        
        return results