# This file is deprecated and will be removed in the future
def calculate_llm_request_cost(completion) -> float:
    """Calculate cost of OpenAI API call (GPT or Claude) based on completion.usage"""

    model = completion.model

    if "gpt" in model:
        # Pricing per 1M tokens for prompt and completion
        # Source: https://openai.com/api/pricing/
        PRICING = {
            "gpt-4o": {
                "prompt": 2.5,
                "completion": 10
            },
            "gpt-4o-mini": {
                "prompt": 0.15,
                "completion": 0.6
            },
        }

        if model.startswith("gpt-4o-mini"):
            pricing = PRICING["gpt-4o-mini"]
        elif model.startswith("gpt-4o"):
            pricing = PRICING["gpt-4o"]
        else:
            raise ValueError(f"Unknown model: {model}")
            
        usage = completion.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

    elif "claude" in model:
        PRICING = {
            "claude-3-5-sonnet": {
                "prompt": 3,
                "completion": 15
            },
            "claude-3-5-haiku": {
                "prompt": 0.8,
                "completion": 4
            },
        }

        if model.startswith("claude-3-5-sonnet"):
            pricing = PRICING["claude-3-5-sonnet"]
        elif model.startswith("claude-3-5-haiku"):
            pricing = PRICING["claude-3-5-haiku"]
        
        usage = completion.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
    

    # Calculate costs per token type
    prompt_cost = (input_tokens * pricing["prompt"]) / 1e6
    completion_cost = (output_tokens * pricing["completion"]) / 1e6
    
    total_cost = prompt_cost + completion_cost
    return round(total_cost, 6)  # Round to 6 decimal places