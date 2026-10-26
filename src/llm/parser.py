import re
import json
from typing import Optional, List, Any, Dict, Tuple, Literal
from core import Message, CodePolicy
from llm.request import LLMType, get_response_from_llm, create_client, RequestOutput
from configs.configs import LLMConfig
import time
from utils.log_utils import logger
import os

LLM_FIX_JSON_PROMPT = """
You are a helpful assistant that fixes JSON strings. The JSON string could not be parsed by a JSON parser.
Please fix the JSON string, such that it can be parsed by the parser. Do not change the structure of the JSON or the content. 
Only fix syntax errors, such that the JSON is valid. Your response should on be the valid JSON string, nothing else.
Here is a list of common problems that could cause the current parsing to fail:
- Double quotes within a string are not escaped e.g. "def act(observation_dict, memory):\n    # Always pull lever 1\n    return "1", memory" -> "def act(observation_dict, memory):\n    # Always pull lever 1\n    return \"1\", memory"

Your response will be parsed directly by the JSON parser, do not add ANY additional text or comments as the parser will immediately fail.
You should fix the following JSON string:

{broken_json_string}
"""

fix_json_llm_config = LLMConfig(
    model=LLMType.OR_GEMINI_2_FLASH,
    temperature=0.0,
    max_tokens=100000,
)

def fix_json_string_using_llm(input_string: str, llm_config: Optional[LLMConfig] = fix_json_llm_config) -> str:
    client, model = create_client(llm_config.model)
    response = get_response_from_llm(client,
     model, 
     [Message(role="user", content=LLM_FIX_JSON_PROMPT.format(broken_json_string=input_string))], 
     temperature=llm_config.temperature, 
     max_tokens=llm_config.max_tokens, 
     reasoning_effort=llm_config.reasoning_effort,
     )
    return response.content

def extract_json_between_markers(llm_output, use_llm_to_fix_json: bool = False):
    def get_json_matches(string: str): 
        # Regular expression pattern to find JSON content between ```json and ```
        json_pattern = r"```json(.*?)```"
        matches = re.findall(json_pattern, string, re.DOTALL)

        if not matches:
            # Fallback: Try to find any JSON-like content in the output
            json_pattern = r"\{.*?\}"
            matches = re.findall(json_pattern, llm_output, re.DOTALL)

            if not matches:
                logger.info("No JSON matches found in the LLM output")
        return matches

    def json_parse_with_verbose_error(json_string: str) -> Optional[Dict]:
        """Attempt to parse a JSON string, returning None if parsing fails."""
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.info(f"\n=== JSON PARSE ERROR DEBUG ===")
            logger.info(f"Error: {e}")
            logger.info(f"Position: {e.pos}, Line: {e.lineno}, Column: {e.colno}")
            
            # Show context
            context_size = 100
            start = max(0, e.pos - context_size)  
            end = min(len(json_string), e.pos + context_size)
            context = json_string[start:end]
            
            logger.info(f"Context around error:")
            logger.info("-" * 60)
            logger.info(context)
            logger.info("-" * 60)
            logger.info(" " * (e.pos - start) + "^-- Error here")
            
            return None

    def try_parse_json(json_string: str) -> Optional[Dict]:
        """Attempt to parse a JSON string, returning None if parsing fails."""
        try:
            return json.loads(json_string)
            # return json_parse_with_verbose_error(json_string)
        except json.JSONDecodeError:
            return None

    def try_clean_and_parse_json(json_string: str) -> Optional[Dict]:
        """Attempt to clean and parse a JSON string, returning None if parsing fails."""
        try:
            # Remove invalid control characters
            json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
            return json_parse_with_verbose_error(json_string_clean)
        except json.JSONDecodeError:
            return None

    def try_llm_fix_and_parse_json(json_string: str) -> Optional[Dict]:
        """Attempt to fix JSON using LLM and parse it, returning None if parsing fails."""
        if not use_llm_to_fix_json:
            return None
            
        logger.info("Attempting to fix JSON string using LLM")
        fixed_json_string = fix_json_string_using_llm(json_string)

        # Even when instructed Gemini Pro 2.5 always adds the ```json markers.
        matches = get_json_matches(fixed_json_string)
        if len(matches) == 0:
            logger.info("No JSON matches found in the fixed JSON string")
            return None

        fixed_json_string = matches[0]
        fixed_output = try_parse_json(fixed_json_string)

        if fixed_output is None:
            fixed_output = try_clean_and_parse_json(fixed_json_string)

        if fixed_output is not None:
            logger.info("Successfully fixed JSON string using LLM")
            return fixed_output
        else:
            logger.info("Failed to fix JSON string using LLM")
            return None

    matches = get_json_matches(llm_output)

    for json_string in matches:
        json_string = json_string.strip()
        
        # Try direct parsing
        if parsed_json := try_parse_json(json_string):
            return parsed_json
            
        # Try cleaning and parsing
        if parsed_json := try_clean_and_parse_json(json_string):
            return parsed_json
            
        # Try LLM fixing and parsing
        if parsed_json := try_llm_fix_and_parse_json(json_string):
            return parsed_json
        
    if os.environ.get("PARSER_LOG_LEVEL") == "DEBUG":
        head_str = llm_output[:1000]
        tail_str = llm_output[-1000:]
        print(f"""
              LLM output that could not be parsed ({len(llm_output)} characters).
              START OF LLM OUTPUT:
              ------------------------------------------------------------
              {head_str}
              ------------------------------------------------------------
              END OF LLM OUTPUT
              ------------------------------------------------------------
              {tail_str}
              """)

    return None  # No valid JSON found

def validate_policy_dict(policy_dict: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, Optional[str]]:
    """
    An explaination of required_fields:
    - splits each string by ':' and checks if this field or hirachy of fields is present in the JSON response
    - e.g. "field1" will check if "field1" is present in the JSON
    - "field1:field2" will check if "field1" is present in the JSON and if dict["field1"] has a field "field2"
    """
    missing_field = False
    missing_field_str = None
    for field_str in required_fields:
        if missing_field:
            break
        
        fields = field_str.split(":")
        current_dict = policy_dict
        for i, field in enumerate(fields):
            if not isinstance(current_dict, dict):
                missing_field = True
                missing_field_str = field_str
                break
            elif field not in current_dict:
                missing_field = True
                missing_field_str = field_str
                break
            else:
                current_dict = current_dict[field]

    return not missing_field, missing_field_str



def request_and_parse_to_json(client, 
                              model: LLMType,
                              msg_history: List[Message],
                              temperature: float,
                              max_tokens: int,
                              reasoning_effort: Optional[Literal["low", "medium", "high"]],
                              required_fields: List[str] = [], 
                              max_retries: int = 5,
                              ) -> Tuple[Dict[str, Any], RequestOutput]:
    def delay():
        time.sleep(5)
        
    for idx in range(max_retries):
        try:
            response = get_response_from_llm(client, model, msg_history, temperature=temperature, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
        except Exception as e:
            logger.warning("Attempt %d/%d: Error getting response from LLM: %s", idx, max_retries, e)
            delay()
            continue

        json_response = extract_json_between_markers(response.content, use_llm_to_fix_json=True)

        if json_response is None:
            logger.warning("Attempt %d/%d: No valid JSON was generated with model %s", idx, max_retries, model)
            delay()
            continue

        # Logic to check if the JSON response contains all required fields
        is_valid, missing_field_str = validate_policy_dict(json_response, required_fields)
        if not is_valid:
            logger.warning("Attempt %d/%d: Required field %s is missing in the JSON response with model %s", idx, max_retries, missing_field_str, model)
            delay()
            continue

        return json_response, response
    
    raise RuntimeError("Failed to generate a valid JSON response after %d attempts", max_retries)

def parse_code_policy_from_dict(policy_dict: Dict[str, Any], num_agents: Optional[int] = 2) -> CodePolicy:
    name = policy_dict["name"]
    policy = CodePolicy(name=name)
    for i in range(num_agents):
        policy.joint_policy.append(policy_dict[f"player_{i}"]["code"])
    return policy