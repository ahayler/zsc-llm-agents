from llm.request import create_client, get_response_from_llm, LLMType
from typing import Optional, Dict, List
from core import Message
from langfuse import observe
from core import CodePolicy
from train.utils import code_policy_to_str
from configs.configs import POLICY_CRITIC_COORDINATION_FRIENDLY_DEFINITION

## POLICY GENERATION PROMPT CODE

CODE_POLICY_GENERATION_PROMPT = """
**General Instructions:**
{general_instructions}

{cfd}

**What exactly is a policy?**
{policy_definition}

**Game Description:**
{game_description}

**Output Specification:**
{output_specification}
"""

POLICY_DEFINITION_CODE = """
Each of the two players has a (local) policy which is an action function. Such a function takes in an observation_dict and returns a legal action.
Generally, each policy function must have the following signature: def act(observation_dict: dict, memory: dict) -> tuple.
The first return value is the action (string), the second return value is the memory (dictionary).
The memory is a dictionary that can be used to store information between calls to the act function.
The memory is initialized to an empty dictionary (i.e. {{}}) at the start of the game. It is NOT initalized to None.
The memory is passed to the act function in each call.
Players do not share the memory, each player has their own memory.
You can make use of python standard libraries like random and math (but not time).

**How does the observation_dict look like? (An example, you can always expect the same general structure)**
NOTE: Only index for values using the EXACT keys (1:1 match) from the example below. Otherwise, you will not be able to access the values.
{dict_observation_description}

**What are the legal actions?**
Note that actions are encoded as strings. The returned action needs to match exactly one of the legal actions (e.g. we are case-sensitive). If an invalid action is returned the game will end immediately and both players receive a penalty.
{possible_actions_description}

Note: Each local policy should always return a valid action, even when paired with a different local policy (i.e. the actions of the other player might be different). Policies that produce invalid actions will be discarded.
"""

# DEFAULT_INSTRUCTIONS_POLICY_GENERATOR = """
# You will need to generate policies for a collaborative two-player game.

# **What is special about the game:**
# - The game is generally turn-based.
# - Agents cannot communicate with each other.
# - Agents can see their actions and the (joined) rewards from previous turns.
# - Generally, agents play each game only once (i.e. they can't remember previous games). If the game is iterated (i.e. the game is played for multiple rounds), this will be made clear in the game description. Note that each game consists of multiple turns.

# **What you need to do exactly:**
# Your goal is to generate a diverse set of policies for the game described below. 
# These policies must be encoded in concise, readable python code. More details are given below. 
# The policies you generate should achieve the highest possible joint reward (on average across many runs of the game).
# Note though that we will evaluate all the policies you generate and only select the best ones. 
# Some policies might be filtered out because they do not fit certain post-hoc criteria. 
# Therefore, you main goal should be to generate a diverse set of policies that have the possiblity of achieving high joint reward.
# In particular, you are encouraged to try to out 'risky' policies or policies that might not be as good as others, but are interesting/different (the policies you might consider best might be filtered out or not as good as you think).
# """

DEFAULT_INSTRUCTIONS_POLICY_GENERATOR = """
You will need to generate policies for a collaborative two-player game.

**What is special about the game:**
- The game is generally turn-based.
- Agents cannot communicate with each other.
- Agents can see their actions and the (joint) reward from previous turns.
- Generally, agents play each game only once (i.e. they can't remember previous games). If the game is iterated (i.e. the game is played for multiple rounds), this will be made clear in the game description. Note that each game consists of multiple turns.
- Ensure the policy is coordination-friendly.

**Instructions:**

**Policy Requirements:**
   - Encode each policy as concise, readable Python code
   - Ensure the policies have good coordination performance in the ZSC setting.

**Zero-shot Coordination (ZSC) Setting:**
   - The policy will be evaluated against independently generated policies (potentially from different LLMs) but with similar instructions (game description, etc.). 
   - Poor coordination performance often arises when policies adopt incompatible conventions or break symmetry arbitrarily.

**Ultimate goal:**
   - Maximize coordination in the ZSC setting.
"""

# Prompt used to generate diverse starting policies in meta-training and the code-CFD critic evaluation.
DIVERSE_INSTRUCTIONS_POLICY_GENERATOR = """
You will need to generate policies for a collaborative two-player game.

**What is special about the game:**
- The game is generally turn-based.
- Agents cannot communicate with each other.
- Agents can see their actions and the (joined) rewards from previous turns.
- Generally, agents play each game only once (i.e. they can't remember previous games). If the game is iterated (i.e. the game is played for multiple rounds), this will be made clear in the game description. Note that each game consists of multiple turns.

**What you need to do exactly:**
- Your goal is to generate policies that are likely to perform well in the game, both paired with themselves and with other policies.
- You should generate a diverse set of policies.
"""

CFD_TEMPLATE = """**Coordination-Friendly Definition:**
Ensure that the policy you generate is strictly compliant with the following Coordination-friendly definition (CFD):
{cfd}
"""

OUTPUT_SPECIFICATION_CODE = """
- Propose {num_policies} of sets of policies for each player, put each set under the list of "joint_policies"
- The policies will be executed in python to generate actions given the observation_dict. Make sure that you are generating valid python code or else the game ends immediately and you receive a penalty.
- There are {num_players} players in this game.
- While we use player_0, player_1, etc. to refer to the players, they are only used for constructing the json, they are not part of the game.
- The joint_policy should be formatted as json below. MAKE SURE TO FOLLOW THE FORMAT EXACTLY, as the json is parsed automatically.
{{
    "joint_policies": [
        {{
        "name": The name of the joint policy, same for all players using this policy.
        "reason": The reason why it is designed in this way.
            player_0: {{
                "code": The python code for the policy function. Should have the following signature: def act(observation_dict: dict, memory: dict) -> Tuple[str, dict]. Make sure to escape double quotes within the code.
            }},
            player_1: {{
                ...
            }},
            ... 
        }},
        ...
    ]
    
}}

**Output**
REASONING:
<REASONING>

Policy JSON:
```json
    <JSON>
```
"""

## POLICY GENERATION PROMPT
NOTES_ON_DIVERSITY = """
As stated before, there is value to diversity, because:
- (a) we will only select the best policy among all the policies you generate (and you are only rewarded for the best policy)
- (b) some good policies might be filtered out because they do not fit certain post-hoc criteria
- (c) you might not be able to perfectly predict which policies have the best average joint reward

Therefore, you should try to generate a diverse set of policies.

Some examples of criteria that you should consider to increase diversity (not at all exhaustive):
- If a policy makes use of certain role assignment/signaling conventions etc. that also could be flipped, generate another policy that filps these conventions.
- Some policies might make use of randomness, some might be rather deterministic.
- Some policies might be symmetric, some might be asymmetric.
(...)

In total, you should generate {num_policies} policies.
"""

## POLICY CRITIC PROMPT

# While creating a coordination-friendly policy is your main priority, you should also try to maximize the joint reward the modifiedpolicy achieves (on average across many runs of the game).
# Make sure to stick to the definition, i.e. only classify a policy as coordination-friendly if it satisfies the definition. 
# Policy-critic prompts (matching train/critic.py: independent vs. joint judging).
_POLICY_CRITIC_BASE_INSTRUCTIONS = """
You are a policy critic, your goal is to judge if a certain policy is 'coordination-friendly' or not, based on the definition given below.
Note that 'coordination-friendly' here is just a term that we define.
In particular, do not try to add your own interpretation of what it 'coordination-friendly' means and stick to the definition exactly.
The definition might be very short or feel incomplete. Still, only reason based on the definition and do not add your own interpretation.
It is important that you do not classify any coordination-unfriendly policy as coordination-friendly.
"""

_POLICY_CRITIC_INDEPENDENT_JUDGING = """
**IMPORTANT: Judge each local policy (player_0, player_1) INDEPENDENTLY.**
You must evaluate each player's local policy as if it exists in isolation — without knowledge of the other player's policy.
The reason is that in deployment, each local policy will be paired with an independently generated partner, NOT with the other local policy in this joint policy.
A joint policy is coordination-friendly ONLY IF both local policies are individually compliant with the coordination-friendly definition.
Do NOT approve a joint policy just because the two local policies happen to work well together — each must be compliant on its own.
"""

_POLICY_CRITIC_MODIFICATION_INSTRUCTIONS = """
In addition to determining whether a policy is coordination-friendly or not, you should also return a modified policy:
- If the policy is not coordination-friendly, you should try to create a modified policy that is coordination-friendly.
- Try to modify the policy not only to make it coordination-friendly, but also such that it returns the highest possible joint reward (on average across many runs of the game).
- If the policy is coordination-friendly, you should return the original policy as the modified policy.
"""

_POLICY_CRITIC_MODIFICATION_INSTRUCTIONS_INDEPENDENT = """
In addition to determining whether a policy is coordination-friendly or not, you should also return a modified policy:
- If the policy is not coordination-friendly, you should try to create a modified policy that is coordination-friendly.
- When modifying, ensure each local policy independently satisfies the coordination-friendly definition.
- Try to modify the policy not only to make it coordination-friendly, but also such that it returns the highest possible joint reward (on average across many runs of the game).
- If the policy is coordination-friendly, you should return the original policy as the modified policy.
"""

POLICY_CRITIC_GENERAL_INSTRUCTIONS = (
    _POLICY_CRITIC_BASE_INSTRUCTIONS
    + _POLICY_CRITIC_INDEPENDENT_JUDGING
    + _POLICY_CRITIC_MODIFICATION_INSTRUCTIONS_INDEPENDENT
)

POLICY_CRITIC_GENERAL_INSTRUCTIONS_JOINT = (
    _POLICY_CRITIC_BASE_INSTRUCTIONS
    + _POLICY_CRITIC_MODIFICATION_INSTRUCTIONS
)


CODE_POLICY_CRITIC_PROMPT = """
**General Instructions:**
{general_instructions}

**What does it mean for a policy to be coordination-friendly?**
{coordination_friendly_definition}

**Policy:**
Below you can find the policy that you will need to judge.
<START OF POLICY>
{policy}
<END OF POLICY>

**Game Description:**
The policy was generated for a game described below.

<START OF GAME DESCRIPTION>
{game_description}
<END OF GAME DESCRIPTION>

**What exactly is a policy?**
{policy_definition}

**Output Specification:**
{output_specification}
"""

CODE_POLICY_CRITIC_OUTPUT_SPECIFICATION = """
You should return a JSON object following EXACTLY the following format:

{{
    "coordination_friendly": <true or false>, # true if the original (unmodified) policy is coordination-friendly, false otherwise.
    "reasoning": <Your reasoning for why the policy is coordination-friendly or not. You can copy over your the reasoning from the REASONING section.>,
    "modified_policy": {{
            "name": <Name of the modified policy>,
            "reasoning": <Reason why it is designed in this way.>,
                player_0: {{
                    "code": <Python code for player_0>
                }},
                player_1: {{
                    "code": <Python code for player_1>
                }},
                ... 
            }}
}}

**Output**
REASONING:
<REASONING>

JSON:
```json
    <JSON>
```
"""

## REFORMULATION PROMPT

REFORMULATION_PROMPT = """
Please reformulate the following prompt. 
The information conveyed in the prompt should be the same, but you should try to word the prompt differently. 
Any elements of the form {{some_name}} must be kept the same (they will be replaced with actual values later).
We handle inserting some of the values later (game_description, JSON output_specification, etc.)
Please only return the reformulated prompt, nothing else.

Original prompt:
<PROMPT>
{prompt}
</PROMPT>
"""

def generate_code_policy_generation_prompt(
        game_description: str, 
        num_policies: int,
        num_players: int,
        dict_observation_description: str,
        possible_actions_description: str,
        general_instructions: str = DEFAULT_INSTRUCTIONS_POLICY_GENERATOR,
        cfd: str = None,
    ):
    output_specification = OUTPUT_SPECIFICATION_CODE.format(
        num_policies=num_policies,
        num_players=num_players
    )

    policy_definition = POLICY_DEFINITION_CODE.format(
        dict_observation_description=dict_observation_description,
        possible_actions_description=possible_actions_description
    )

    if cfd is not None:
        cfd = CFD_TEMPLATE.format(cfd=cfd)
    else:
        cfd = ""

    final_prompt = CODE_POLICY_GENERATION_PROMPT.format(
        general_instructions=general_instructions,
        game_description=game_description,
        policy_definition=policy_definition,
        output_specification=output_specification,
        cfd=cfd
    )
    
    return final_prompt

@observe()
def reformulate_prompt(prompt: str, model: Optional[LLMType] = LLMType.OR_GEMINI_PRO_25):
    client, model = create_client(model)
    message = Message(
        role="user",
        content=REFORMULATION_PROMPT.format(prompt=prompt)
    )

    response = get_response_from_llm(
         client,
         model,
         [message],
         temperature=1.0, # We want some diversity in the reformulations
         )
    return response.content


@observe()
def generate_code_policy_critic_prompt(
        policy: CodePolicy,
        game_description: str,
        dict_observation_description: str,
        possible_actions_description: str,
        coordination_friendly_definition: Optional[str] = POLICY_CRITIC_COORDINATION_FRIENDLY_DEFINITION,
        output_specification: Optional[str] = CODE_POLICY_CRITIC_OUTPUT_SPECIFICATION,
        general_instructions: Optional[str] = POLICY_CRITIC_GENERAL_INSTRUCTIONS,
    ):
    policy_definition = POLICY_DEFINITION_CODE.format(
        dict_observation_description=dict_observation_description,
        possible_actions_description=possible_actions_description
    )
    return CODE_POLICY_CRITIC_PROMPT.format(
        general_instructions=general_instructions,
        policy=code_policy_to_str(policy),
		policy_definition=policy_definition,
        game_description=game_description,
        coordination_friendly_definition=coordination_friendly_definition,
        output_specification=output_specification
    )

def example_observation_to_string(example_observations: Dict[str, str]) -> str:
    return "\n".join([f"{k}: {v}" for k, v in example_observations.items()])

## SELF CRITIC PROMPT

CODE_POLICY_SELF_CRITIC_PROMPT = """
**Task:**
You are an expert policy evaluator. You are given a list of candidate policies for a coordination game and a "Coordination-Friendly Definition" (CFD).
Your goal is to select the single policy from the list that best respects the CFD and is most likely to coordinate well with other agents following the same CFD.

**Coordination-Friendly Definition:**
{cfd}

**Game Description:**
{game_description}

**Candidate Policies:**
Below is the list of generated policies. Each policy has an ID.

{policies_list}

**Instructions:**
1. Analyze each policy based on how well it adheres to the CFD.
2. Select the BEST policy.
3. Return the ID of the selected policy.

**Output Specification:**
{output_specification}
"""

SELF_CRITIC_OUTPUT_SPECIFICATION = """
You should return a JSON object following EXACTLY the format below:
{{
    "reasoning": "Explanation of why this policy was chosen over others based on the CFD.",
    "selected_policy_id": <int> // The 0-based index of the selected policy
}}
"""

@observe()
def generate_code_policy_self_critic_prompt(
        policies: List[CodePolicy],
        cfd: str,
        game_description: str,
        output_specification: str = SELF_CRITIC_OUTPUT_SPECIFICATION
    ) -> str:
    
    policies_str_list = []
    for idx, policy in enumerate(policies):
        policies_str_list.append(f"Policy ID: {idx}\n{code_policy_to_str(policy)}")
    
    policies_list_str = "\n\n".join(policies_str_list)
    
    return CODE_POLICY_SELF_CRITIC_PROMPT.format(
        cfd=cfd,
        game_description=game_description,
        policies_list=policies_list_str,
        output_specification=output_specification
    )
