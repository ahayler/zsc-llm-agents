"""
Helper function to generate a policy given a coordination-friendly definition (CFD) and an environment.
"""
from typing import Union, List, Optional
from core import CodePolicy
from configs.envs import (
    EnvConfig,
    DEFAULT_SELECTION_WITH_SEMANTICS_CONFIG,
    DEFAULT_CONVENTIONS_WITH_SEMANTICS_CONFIG,
    DEFAULT_TIGER_AND_TRAMPOLINE_CONFIG,
    DEFAULT_PERMUTED_LEVERS_CONFIG,
    DEFAULT_NARROW_TUNNEL_CONFIG_CODE,
    DEFAULT_AD_HOC_CONVENTIONS_CONFIG,
    DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG,
    DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG2,
    DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG3,
    DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG4,
    DEFAULT_BRITTLE_BRIDGE_CONFIG_CODE,
    DEFAULT_MATRIX_GAME_CONFIG,
    DEFAULT_TINY_HANABI_CONFIG,
    DEFAULT_TINY_HANABI_FIXED_CONFIG,
)
from configs.envs import (
    CatAndDogConfig,
    CatAndDog2Config,
    CatAndDog3Config,
    CatAndDog4Config,
)
from configs.configs import CodePolicyGeneratorConfig, DEFAULT_CODE_POLICY_GENERATOR_CONFIG, LLMConfig
from llm.request import LLMType
from train.policy_generator import CodePolicyGenerator
from utils.tracing import langfuse_client


# Mapping from env name to default config (exported for use in other modules)
DEFAULT_ENV_CONFIGS = {
    "cat_and_dog": CatAndDogConfig(),
    "cat_and_dog2": CatAndDog2Config(),
    "cat_and_dog3": CatAndDog3Config(),
    "cat_and_dog4": CatAndDog4Config(),
    "permuted_levers": DEFAULT_PERMUTED_LEVERS_CONFIG,
    "narrow_tunnel": DEFAULT_NARROW_TUNNEL_CONFIG_CODE,
    "ad_hoc_conventions": DEFAULT_AD_HOC_CONVENTIONS_CONFIG,
    "ad_hoc_conventions_permuted_signals": DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG,
    "ad_hoc_conventions_permuted_signals2": DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG2,
    "ad_hoc_conventions_permuted_signals3": DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG3,
    "ad_hoc_conventions_permuted_signals4": DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG4,
    "brittle_bridge": DEFAULT_BRITTLE_BRIDGE_CONFIG_CODE,
    "matrix_game": DEFAULT_MATRIX_GAME_CONFIG,
    "tiny_hanabi": DEFAULT_TINY_HANABI_CONFIG,
    "tiny_hanabi_fixed": DEFAULT_TINY_HANABI_FIXED_CONFIG,
    "selection_with_semantics": DEFAULT_SELECTION_WITH_SEMANTICS_CONFIG,
    "conventions_with_semantics": DEFAULT_CONVENTIONS_WITH_SEMANTICS_CONFIG,
    "tiger_and_trampoline": DEFAULT_TIGER_AND_TRAMPOLINE_CONFIG,
}


def generate_policy(
    cfd: str,
    env: Union[str, EnvConfig],
    num_policies: int = 1,
    llm: Optional[LLMType] = None,
    generator_config: CodePolicyGeneratorConfig = None,
    langfuse_session_id: Optional[str] = None,
    n_generations_per_seed: int = 1,
    use_self_critic: bool = False,
    max_tokens: Optional[int] = None,
) -> List[CodePolicy]:
    """
    Generate one or more policies given a coordination-friendly definition (CFD) and an environment.
    
    Args:
        cfd: The coordination-friendly definition string
        env: Can be:
            - A string (environment name, e.g., "cat_and_dog") - uses default config
            - An EnvConfig instance (for custom configuration)
        num_policies: Number of policies to generate (default: 1)
        llm: Optional LLMType to use for generation. If None, uses default from config.
        generator_config: Optional CodePolicyGeneratorConfig. If None, uses default config.
            If llm is specified, it will override the model in generator_config.
        langfuse_session_id: Optional session id to group tracing under a single Langfuse trace.
        n_generations_per_seed: Number of policies to sample from the LLM at once (default 1)
        use_self_critic: If True, query the LLM again to pick the best policy. If False, take the first one. (default False)
        
    Returns:
        List of CodePolicy objects
        
    Examples:
        >>> # Using environment name (uses default config)
        >>> policies = generate_policy(
        ...     cfd="- Use clear signaling conventions\\n- Coordinate through actions",
        ...     env="cat_and_dog"
        ... )
        
        >>> # Using EnvConfig for custom configuration
        >>> from configs.envs import CatAndDogConfig
        >>> policies = generate_policy(
        ...     cfd="- Coordinate through actions",
        ...     env=CatAndDogConfig()
        ... )
        
        >>> # Generate multiple policies with specific LLM
        >>> from llm.request import LLMType
        >>> policies = generate_policy(
        ...     cfd="- Follow symmetry breaking rules",
        ...     env="matrix_game",
        ...     num_policies=3,
        ...     llm=LLMType.OR_GPT5_MINI
        ... )
        
        >>> # Using a different LLM
        >>> policies = generate_policy(
        ...     cfd="- Use clear signaling",
        ...     env="cat_and_dog",
        ...     llm=LLMType.OR_GROK_41_FAST
        ... )
    """
    if langfuse_session_id is not None:
        langfuse_client.update_current_trace(session_id=langfuse_session_id)

    # Convert env name to EnvConfig if needed
    if isinstance(env, str):
        env_name = env.lower().replace("-", "_")
        if env_name not in DEFAULT_ENV_CONFIGS:
            available = ", ".join(sorted(DEFAULT_ENV_CONFIGS.keys()))
            raise ValueError(
                f"Unknown environment '{env}'. Available environments: {available}"
            )
        env_config = DEFAULT_ENV_CONFIGS[env_name]
    elif isinstance(env, EnvConfig):
        env_config = env
    else:
        raise ValueError(f"env must be str or EnvConfig, got {type(env)}")
    
    # Create generator config if not provided
    if generator_config is None:
        # Use specified LLM or default
        if llm is not None:
            llm_config = LLMConfig(
                model=llm,
                temperature=DEFAULT_CODE_POLICY_GENERATOR_CONFIG.llm_config.temperature,
                max_tokens=max_tokens if max_tokens is not None else DEFAULT_CODE_POLICY_GENERATOR_CONFIG.llm_config.max_tokens,
                reasoning_effort=DEFAULT_CODE_POLICY_GENERATOR_CONFIG.llm_config.reasoning_effort,
            )
        else:
            llm_config = DEFAULT_CODE_POLICY_GENERATOR_CONFIG.llm_config
            if max_tokens is not None:
                llm_config = llm_config.model_copy(update={"max_tokens": max_tokens})
        
        generator_config = CodePolicyGeneratorConfig(
            llm_config=llm_config,
            num_policies=num_policies,
            num_players=2,
            seed=None,
            use_cached_generations=False,
            randomly_swap_local_policies=False,
            n_generations_per_seed=n_generations_per_seed,
            use_self_critic=use_self_critic,
        )
    else:
        # Override num_policies if specified
        if num_policies != 1:
            generator_config = generator_config.model_copy(update={"num_policies": num_policies})
        
        # Override LLM if specified
        if llm is not None:
            update_data = {"model": llm}
            if max_tokens is not None:
                update_data["max_tokens"] = max_tokens
            llm_config = generator_config.llm_config.model_copy(update=update_data)
            generator_config = generator_config.model_copy(update={"llm_config": llm_config})
        elif max_tokens is not None:
             llm_config = generator_config.llm_config.model_copy(update={"max_tokens": max_tokens})
             generator_config = generator_config.model_copy(update={"llm_config": llm_config})
            
        # Update with new args if they differ from default/config
        update_dict = {}
        if n_generations_per_seed != 1:
            update_dict["n_generations_per_seed"] = n_generations_per_seed
        if use_self_critic:
            update_dict["use_self_critic"] = use_self_critic
            
        if update_dict:
            generator_config = generator_config.model_copy(update=update_dict)
    
    # Create policy generator
    generator = CodePolicyGenerator(generator_config)
    
    # Generate policies with CFD
    policies = generator.generate(env_config, cfd=cfd)
    
    return policies
