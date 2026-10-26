# Inspired by: https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/llm.py
#import openai
import anthropic
from typing import NamedTuple, Literal, Tuple, Any, get_args

from openai.types.shared import reasoning
from llm.costs import calculate_llm_request_cost
import re
import json
from enum import Enum
import os
import threading
from google import genai
from google.genai.types import GenerationConfig
from core import Message
from typing import Optional
import logging

from utils.tracing import langfuse_client

from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)


MAX_NUM_TOKENS = 4096

class LLMType(str, Enum):
    # Anthropic models
    CLAUDE_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_HAIKU = "claude-3-5-haiku-20241022"
    # OpenAI models
    GPT4_MINI = "gpt-4o-mini-2024-07-18"
    GPT4 = "gpt-4o-2024-08-06"
    O3 = "o3-2025-04-16"
    O3_MINI_LOW = "o3-mini-2025-01-31/low"
    O3_MINI_MEDIUM = "o3-mini-2025-01-31/medium"
    O3_MINI_HIGH = "o3-mini-2025-01-31/high"

    # OpenRouter models
    # OpenAI models
    OR_GPT4 = "open-router/openai/gpt-4o"
    OR_GPT4_MINI = "open-router/openai/gpt-4o-mini"
    OR_GPT4_1_MINI = "open-router/openai/gpt-4.1-mini"
    OR_O3 = "open-router/openai/o3"
    OR_O3_MINI_HIGH = "open-router/openai/o3-mini-high"
    OR_O4_MINI_HIGH = "open-router/openai/o4-mini-high"
    OR_O4_MINI = "open-router/openai/o4-mini"
    OR_O3_MINI = "open-router/openai/o3-mini"
    OR_GPT5 = "open-router/openai/gpt-5"
    OR_GPT51 = "open-router/openai/gpt-5.1"
    OR_GPT5_MINI = "open-router/openai/gpt-5-mini"
    OR_GPT_52 = "open-router/openai/gpt-5.2"
    OR_GPT54 = "open-router/openai/gpt-5.4"
    
    # Meta
    OR_LLAMA_4_MAVERICK = "open-router/meta-llama/llama-4-maverick"

    # Anthropic models
    OR_CLAUDE_SONNET_35 = "open-router/anthropic/claude-3.5-sonnet"
    OR_CLAUDE_HAIKU_35 = "open-router/anthropic/claude-3-5-haiku"
    OR_CLAUDE_SONNET_37 = "open-router/anthropic/claude-3.7-sonnet"
    OR_CLAUDE_SONNET_4 = "open-router/anthropic/claude-sonnet-4"
    OR_CLAUDE_SONNET_45 = "open-router/anthropic/claude-sonnet-4.5"
    OR_CLAUDE_SONNET_46 = "open-router/anthropic/claude-sonnet-4.6"
    OR_CLAUDE_HAIKU_45 = "open-router/anthropic/claude-haiku-4.5"
    OR_CLAUDE_OPUS_45 = "open-router/anthropic/claude-opus-4.5"

    # Deepseek models
    OR_DEEPSEEK_V3_1 = "open-router/deepseek/deepseek-chat-v3.1"
    OR_DEEPSEEK_V3_1_TERMINUS = "open-router/deepseek/deepseek-v3.1-terminus"
    OR_DEEPSEEK_V3_2 = "open-router/deepseek/deepseek-v3.2"

    # z-ai
    OR_GLM_4_5 = "open-router/z-ai/glm-4.5"
    OR_GLM_4_6 = "open-router/z-ai/glm-4.6"
    OR_GLM_4_7 = "open-router/z-ai/glm-4.7"

    # xAI
    OR_GROK_4 = "open-router/x-ai/grok-4"
    OR_GROK_4_FAST = "open-router/x-ai/grok-4-fast"
    OR_GROK_41_FAST = "open-router/x-ai/grok-4.1-fast"

    # Perplexity models
    OR_PERPLEXITY_R1_1776 = "open-router/perplexity/r1-1776"

    # Gemini models (through OpenRouter)
    OR_GEMINI_2_FLASH = "open-router/google/gemini-2.0-flash-001"
    OR_GEMINI_FLASH_25 = "open-router/google/gemini-2.5-flash"
    OR_GEMINI_FLASH_25_PREVIEW = "open-router/google/gemini-2.5-flash-preview-05-20"
    OR_GEMINI_FLASH_THINKING = "open-router/google/gemini-2.0-flash-thinking-exp:free"
    #OR_GEMINI_FLASH_25_THINKING = "open-router/google/gemini-2.5-flash-preview:thinking"  # Endpoint doesn't exist anymore
    OR_GEMINI_PRO_25 = "open-router/google/gemini-2.5-pro"
    OR_GEMINI_PRO_3_PREVIEW = "open-router/google/gemini-3-pro-preview"
    OR_GEMINI_PRO_31_PREVIEW = "open-router/google/gemini-3.1-pro-preview"
    # Google models
    GEMINI_FLASH_THINKING = "gemini-2.0-flash-thinking-exp-01-21"

    OR_HORIZON_BETA = "open-router/openrouter/horizon-beta"

    # Qwen models
    OR_QWEN3_MAX = "open-router/qwen/qwen3-max"

    # kimi models
    OR_KIMI_K2_THINKING = "open-router/moonshotai/kimi-k2-thinking"


    # Human input
    HUMAN = "human"

    def __str__(self) -> str:
        return self.value
    
class RequestOutput(NamedTuple):
    model: LLMType
    content: str
    msg_history: list[Message]
    system_message: str

TIMEOUT_LIMIT_DICT = {
    LLMType.OR_O3: 240.0,
    LLMType.OR_O4_MINI: 240.0,
    LLMType.OR_GPT5: 240.0,
    LLMType.OR_GPT51: 240.0,
    LLMType.OR_GPT5_MINI: 180.0,
    LLMType.OR_GEMINI_FLASH_25: 30.0,
    LLMType.OR_GEMINI_PRO_25: 180.0,
    LLMType.OR_GEMINI_PRO_3_PREVIEW: 180.0,
    LLMType.OR_GPT4_1_MINI: 30.0,
    LLMType.OR_GROK_4_FAST: 240.0,
    LLMType.OR_GROK_4: 240.0,
    LLMType.OR_GROK_41_FAST: 240.0,
    LLMType.OR_CLAUDE_SONNET_45: 240.0,
}

AVAILABLE_LLMS = [model.value for model in LLMType]

def _log_timeout_warning(model: str, timeout_seconds: float, stop_event: threading.Event):
    """Helper function to log timeout warnings periodically while waiting for LLM response."""
    if not stop_event.wait(timeout_seconds):
        logger.warning(f"LLM request to {model} is taking longer than {timeout_seconds:.2f} seconds...")
        # Log additional warnings every 30 seconds after the first warning
        warning_interval = 30.0
        while not stop_event.wait(warning_interval):
            logger.warning(f"LLM request to {model} is still pending (running for more than {timeout_seconds + warning_interval:.2f} seconds)...")
            warning_interval += 30.0

#@observe(as_type="generation")
def get_response_from_llm(
        client,
        model: LLMType,
        msg_history: Optional[list[Message]] = None,
        msg: Optional[str] = None,
        system_message: Optional[str] = None,
        print_debug: bool = False,
        temperature: float = 0, #0.75
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = "medium",
        max_tokens: Optional[int] = MAX_NUM_TOKENS,
        seed: Optional[int] = 42,
        n_max_retries_empty_response: int = 3,
        default_timeout_warning_in_seconds: float = 60.0,
) -> RequestOutput:
    """Get a response from the LLM.

    Args:
        client (Any): The client to use to get the response from the LLM.
        model (LLMType): The model to use to get the response from the LLM.
        msg_history (list[Message]): The message history supplied in the call.
        msg (Optional[str]): If supplied, added as the newest message to the message history (role="user").
        system_message (Optional[str]): An optional system message.
        print_debug (bool): Whether to print the debug information.
        temperature (float): The temperature to use to get the response from the LLM.
        max_tokens (Optional[int]): The maximum number of tokens to use to get the response from the LLM.
        n_max_retries_empty_response (int): The maximum number of retries to make if the LLM returns an empty response.
        default_timeout_warning_in_seconds (float): The (default) timeout warning threshold in seconds.
    """
    assert not (msg_history is None and msg is None), "msg_history and msg cannot both be None"

    # Save all local variables for retry behavior
    args_dict = locals()

    timeout_warning_seconds = TIMEOUT_LIMIT_DICT.get(model, default_timeout_warning_in_seconds)

    if msg_history is None:
        msg_history = []

    if not system_message:
        system_message = ""

    # Add the user message to the message history
    if msg is not None:
        user_message = Message(role="user", content=msg)
        msg_history = msg_history + [user_message]

    # Start timeout warning in background thread
    stop_event = threading.Event()
    warning_thread = threading.Thread(
        target=_log_timeout_warning, 
        args=(model, timeout_warning_seconds, stop_event),
        daemon=True
    )
    warning_thread.start()

    if model.startswith("claude-"):
        logger.warning("Claude is not supported with Langfuse tracing.")

        # Convert Message objects to Claude dicts
        dict_msg_history = [message.to_claude_dict() for message in msg_history]

        langfuse_client.update_current_span(
            input=dict_msg_history,
            model=model,
            metadata={
                "system_message": system_message,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message,
                messages=dict_msg_history,
            )
        finally:
            # Stop the warning thread
            stop_event.set()

        # See docs for more details on token counts and usd cost in Langfuse
        # https://langfuse.com/docs/model-usage-and-cost
        langfuse_client.update_current_span(
            usage_details={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens
            }
        )
 
        content = response.content[0].text
    elif model in [
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    ] or model.startswith("open-router"):
                
        if model.startswith("open-router"):
            model = '/'.join(model.split('/')[1:]) # Remove the open-router/ prefix

        dict_msg_history = [message.to_openai_dict() for message in msg_history]

        try:            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    *dict_msg_history,
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                seed=seed,
                stream=False,
                reasoning_effort=reasoning_effort,
            )
        finally:
            # Stop the warning thread
            stop_event.set()

        # With this responses can be streamed. Sadly, it doesn't seem to help with the empty response issue.
        # content = ""
        # for chunk in stream:
        #     content += chunk.choices[0].delta.content

        content = response.choices[0].message.content

    elif model.startswith("o3-"):
        model, reasoning_effort = model.split("/")

        dict_msg_history = [message.to_openai_dict() for message in msg_history]
    
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    *dict_msg_history,
                ],
                # These parameters are not supported by the o3 API
                #temperature=temperature,
                #max_tokens=max_tokens,
                n=1,
                stop=None,
                seed=seed,
                # reasoning_effort=reasoning_effort,
            )
        finally:
            # Stop the warning thread
            stop_event.set()

        content = response.choices[0].message.content
    elif model.startswith("gemini-"):
        logger.warning("Gemini is not supported with Langfuse tracing.")
        if system_message != "":
            raise NotImplementedError("System messages are not yet implemented for Gemini.")

        dict_msg_history = [message.to_google_dict() for message in msg_history]

        try:
            response = client.models.generate_content(
                model=model,
                contents=dict_msg_history,

                # TODO: Langfuse observe cause a weird error when we use the GenerationConfig; This needs to be fixed/debugged at some point
                # config=GenerationConfig(
                #     temperature=temperature,
                #     max_output_tokens=MAX_NUM_TOKENS,
                #     candidate_count=1,
                # )
            )
            # response = client.models.generate_content(
            #     model=model,
            #     contents=[
            #         {
            #             "role": msg["role"],
            #             "content": msg["content"],
            #             "provider_options": {
            #                 "openrouter": {
            #                     "cache_control": {"type": "ephemeral"}
            #                 },
            #                 "google": {"cacheControl": {"type": "ephemeral"}}
            #             }
            #         }
            #         for msg in dict_msg_history
            #     ],
            #     # config=GenerationConfig(
            #     #     temperature=temperature,
            #     #     max_output_tokens=MAX_NUM_TOKENS,
            #     #     candidate_count=1,
            #     # )
            # )
        finally:
            # Stop the warning thread
            stop_event.set()

        content = response.text
    else:
        raise ValueError(f"Model {model} not supported.")
    
    # If the LLM returns an empty response, retry up to n_max_retries_empty_response times
    if content == "":
        if n_max_retries_empty_response > 0:
            args_dict["n_max_retries_empty_response"] -= 1
            logger.warning(f"{model} returned an empty response. Retrying... ({n_max_retries_empty_response} retries left)")
            return get_response_from_llm(**args_dict)
        else:
            raise ValueError(f"{model} returned an empty response.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(dict_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    output = RequestOutput(model, content, msg_history, system_message)

    return output

def create_client(model: LLMType, use_langfuse: bool = True) -> Tuple[Any, LLMType]:
    if use_langfuse:
        from langfuse.openai import openai
    else:
        import openai

    if model not in AVAILABLE_LLMS:
        raise ValueError(f"Model {model} not supported. Available models: {AVAILABLE_LLMS}")
        
    if model.startswith("claude-"):
        print(f"Using Anthropic API with model {model}.")
        return anthropic.Anthropic(), model
    elif model.startswith("gpt-") or model.startswith("o3-"):
        #print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model.startswith("open-router"):
        #print(f"Using OpenRouter API with model {model}.")
        return openai.OpenAI(base_url=f"https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")), model
    elif model.startswith("gemini-"):
        #print(f"Using Google API with model {model}.")
        return genai.Client(api_key=os.getenv("GEMINI_API_KEY")), model
    elif model == LLMType.HUMAN:
        return None, model
    else:
        raise ValueError(f"Model {model} not supported.")

