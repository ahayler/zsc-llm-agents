from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import sys

from typing import Dict, Optional, Tuple, List
#from google.genai.types import Part, Content
import numpy as np

@dataclass
class CompactRolloutHistory:
    hyperparams: Dict
    metadata: Dict
    rewards_sum: float
    terminated: bool
    truncated: bool
    compact_states: List[Dict]
    crashed: bool = False
    crash_cause_idx: Optional[int] = None
    error: Optional[str] = None
    metadata_policy_modifier: Optional[Dict] = None
    
class Message:
    """
    Represents a conversation message with role and content. 
    Class should be model agnostic as far as possible.
    Code should work with Message objects instead of dicts that might be specific to an LLM.
    """

    def __init__(self, role: str, content: str):
        self.role = role  # 'system', 'user', 'assistant'
        self.content = content  # String content of the message

    def __repr__(self):
        return f"Message(role={self.role}, content={self.content}"
    
    def to_claude_dict(self):
        return {
            "role": self.role,
            "content": [
                {
                    "type": "text",
                    "text": self.content,
                }
            ],
        }
    
    def to_openai_dict(self):
        return {
            "role": self.role,
            "content": self.content,
        }
    
    # def to_google_dict(self):
    #     return Content(
    #         role=self.role if self.role != "assistant" else "model", # It seems like the Google API expects "model" for assistant messages: https://github.com/googleapis/python-genai/blob/b46ed361d9c0845880a1447ee42dd3c0f6f7a886/google/genai/types.py#L688
    #         parts=[Part(text=self.content)]
    #     )

@dataclass
class Policy:
    name: str # required for tracing
    reason: str = ""
    max_reward: float = -sys.float_info.max
    min_reward: float = sys.float_info.max
    avg_reward: float = 0
    std_reward: float = 0

@dataclass
class CodePolicy(Policy):
    joint_policy: list[str] = field(default_factory=list) 

    def clone(self):
        return CodePolicy(
            name=self.name,
            reason=self.reason,
            joint_policy=self.joint_policy.copy(),
        )

    def __eq__(self, other):
        if not isinstance(other, CodePolicy) or not isinstance(other.joint_policy, list) or not isinstance(self.joint_policy, list):
            return False
        return len(self.joint_policy) == len(other.joint_policy) and np.all([self.joint_policy[idx] == other.joint_policy[idx] for idx in range(len(self.joint_policy))])


