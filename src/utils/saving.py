from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from core import CodePolicy, CompactRolloutHistory
import json
from dataclasses import asdict
from datetime import datetime
from utils.log_utils import get_logger
from configs.configs import CriticHistory, CriticOutput, BaseTextAgentConfig, ChatHistoryAgentConfig, LLMConfig

DATA_DIR = Path(__file__).parent.parent.parent/ "data"

logger = get_logger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

"""
Each experiment can span multiple environments. For each environment, it again can save results across multiple timestamps. The 'smallest' unit of observation is a CodeXPExperimentResults.
"""

@dataclass
class CodeXPExperimentResults:
    experiment_name: str
    env_name_file_safe: str
    selected_policies: Optional[Dict[int, CodePolicy]] = None
    generated_policies: Optional[Dict[int, List[CodePolicy]]] = None
    policy_critic_histories: Optional[Dict[int, List[CriticHistory]]] = None
    xp_matrix: Optional[np.ndarray] = None
    compact_rollout_histories: Optional[np.ndarray] = None
    trace_id: Optional[str] = None
    coordination_friendly_definition: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None
    game_description: Optional[str] = None

@dataclass
class TextXPExperimentResults:
    experiment_name: str
    env_name_file_safe: str
    agent_configs: List[BaseTextAgentConfig]
    xp_matrix: Optional[np.ndarray] = None
    compact_rollout_histories: Optional[np.ndarray] = None
    additional_info: Optional[Dict[str, Any]] = None

def rewards_array_from_compact_rollout_histories(compact_rollout_histories: np.ndarray) -> np.ndarray:
    n, m = compact_rollout_histories.shape
    rewards_array = np.zeros((n, m), dtype=object)
    for i in range(n):
        for j in range(m):
            rewards_array[i, j] = [history.rewards_sum for history in compact_rollout_histories[i, j]]
    return rewards_array

def std_error_array_from_rewards_array(rewards_array: np.ndarray) -> np.ndarray:
    n, m = rewards_array.shape
    std_error_array = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            std_error_array[i, j] = np.std(rewards_array[i, j], ddof=1) / np.sqrt(len(rewards_array[i, j]))

    return std_error_array

def get_current_timestamp_as_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_base_path(experiment_name: str, mode: Literal["code", "text"], subdir: Optional[str] = None) -> Path:
    if subdir is not None:
        return DATA_DIR / mode / experiment_name / subdir
    else:
        return DATA_DIR / mode / experiment_name

def get_save_path(experiment_name: str, mode: Literal["code", "text"], subdir: str):
    base_path = get_base_path(experiment_name, mode, subdir) / get_current_timestamp_as_string()

    # create directory if it doesn't exist
    base_path.mkdir(parents=True, exist_ok=True)

    return base_path

def get_all_available_envs_for_experiment(experiment_name: str, mode: Literal["code", "text"]) -> List[str]:
    base_path = get_base_path(experiment_name, mode)
    return [d.name for d in base_path.iterdir() if d.is_dir() and d.name != "meta_loop"]

def critic_history_list_to_dict_list(critic_histories: List[CriticHistory]) ->List[Dict]:
    return [asdict(critic_history) for critic_history in critic_histories]

def dict_to_critic_output(critic_output_dict: Dict) -> CriticOutput:
    critic_output_dict_copy = critic_output_dict.copy()
    input_policy_dict = critic_output_dict_copy.pop("input_policy")
    modified_policy_dict = critic_output_dict_copy.pop("modified_policy")
    critic_output_dict_copy["input_policy"] = CodePolicy(**input_policy_dict)
    if modified_policy_dict is not None:
        critic_output_dict_copy["modified_policy"] = CodePolicy(**modified_policy_dict)
    return CriticOutput(**critic_output_dict_copy)

def dict_to_critic_history(critic_history_dict: Dict) -> CriticHistory:
    critic_history_dict_copy = critic_history_dict.copy()
    critic_outputs = critic_history_dict_copy.pop("critic_outputs")
    critic_history_dict_copy["critic_outputs"] = [dict_to_critic_output(critic_output) for critic_output in critic_outputs]
    return CriticHistory(**critic_history_dict_copy)

def dict_list_to_critic_history_list(critic_history_dict_list: List[Dict]) -> List[CriticHistory]:
    return [dict_to_critic_history(critic_history_dict) for critic_history_dict in critic_history_dict_list]

def load_critic_history_list_dict_from_json(file_path: Path) -> Dict[int, List[CriticHistory]]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {int(k): dict_list_to_critic_history_list(v) for k, v in data.items()}

def policy_to_dict(policy: Union[CodePolicy]) -> Dict:
    policy_dict = asdict(policy)
    if isinstance(policy, CodePolicy):
        policy_dict["type"] = "code"
    return policy_dict

def agent_config_to_dict(agent_config: BaseTextAgentConfig) -> Dict:
    agent_config_dict = agent_config.model_dump()
    if isinstance(agent_config, ChatHistoryAgentConfig):
        agent_config_dict["type"] = "chat_history"
    else:
        raise ValueError(f"Invalid agent config type: {type(agent_config)}")
    return agent_config_dict

def policy_list_to_dict_list(policies: List[CodePolicy]) -> Dict[int, List[CodePolicy]]:
    return [policy_to_dict(p) for p in policies]

def policy_dict_to_dict(policy_dict: Dict) -> Dict[int, Dict]:
    return {k: policy_to_dict(v) for k, v in policy_dict.items()}

def save_policies_as_json(policies: List[Union[CodePolicy]], save_path: Path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump([policy_to_dict(p) for p in policies], f, indent=2, cls=NumpyEncoder)

def save_xp_matrix_as_json(xp_matrix: np.ndarray, save_path: Path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(xp_matrix.tolist(), f, indent=2, cls=NumpyEncoder)

def save_compact_rollout_histories_as_json(compact_rollout_histories: np.ndarray, save_path: Path):
    n, m = compact_rollout_histories.shape
    dict_compact_rollout_histories = np.empty((n, m), dtype=object)
    for i in range(n):
        for j in range(m):
            dict_compact_rollout_histories[i, j] = [asdict(compact_rollout_history) for compact_rollout_history in compact_rollout_histories[i, j]]
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(dict_compact_rollout_histories.tolist(), f, indent=2, cls=NumpyEncoder)

def save_text_xp_experiment_results(
        experiment_results: TextXPExperimentResults
        ):
    save_base_path = get_save_path(experiment_results.experiment_name, "text", experiment_results.env_name_file_safe)

    agent_configs = experiment_results.agent_configs
    xp_matrix = experiment_results.xp_matrix
    compact_rollout_histories = experiment_results.compact_rollout_histories
    additional_info = experiment_results.additional_info

    if agent_configs is not None:
        with open(save_base_path / "agent_configs.json", "w", encoding="utf-8") as f:
            json.dump([agent_config_to_dict(agent_config) for agent_config in agent_configs], f, indent=2, cls=NumpyEncoder)
    if xp_matrix is not None:
        save_xp_matrix_as_json(xp_matrix, save_base_path / "xp_matrix.json")
    if compact_rollout_histories is not None:
        save_compact_rollout_histories_as_json(compact_rollout_histories, save_base_path / "compact_rollout_histories.json")
    if additional_info is not None:
        try:
            with open(save_base_path / "additional_info.json", "w", encoding="utf-8") as f:
                json.dump(additional_info, f, indent=2, cls=NumpyEncoder)
        except Exception as e:
            logger.error(f"Failed to save additional info: {e}")

def save_code_xp_experiment_results(
        experiment_results: CodeXPExperimentResults
        ):
    save_base_path = get_save_path(experiment_results.experiment_name, "code", experiment_results.env_name_file_safe)

    selected_policies = experiment_results.selected_policies
    generated_policies = experiment_results.generated_policies
    xp_matrix = experiment_results.xp_matrix
    compact_rollout_histories = experiment_results.compact_rollout_histories
    trace_id = experiment_results.trace_id
    coordination_friendly_definition = experiment_results.coordination_friendly_definition
    additional_info = experiment_results.additional_info
    policy_critic_histories = experiment_results.policy_critic_histories

    if selected_policies is not None:
        with open(save_base_path / "selected_policies.json", "w", encoding="utf-8") as f:
            json.dump(policy_dict_to_dict(selected_policies), f, indent=2, cls=NumpyEncoder)
    if generated_policies is not None:
        with open(save_base_path / "generated_policies.json", "w", encoding="utf-8") as f:
            json.dump({i: policy_list_to_dict_list(policies) for i, policies in generated_policies.items()}, f, indent=2, cls=NumpyEncoder)
    if xp_matrix is not None:
        save_xp_matrix_as_json(xp_matrix, save_base_path / "xp_matrix.json")
    if compact_rollout_histories is not None:
        save_compact_rollout_histories_as_json(compact_rollout_histories, save_base_path / "compact_rollout_histories.json")
    if trace_id is not None:
        with open(save_base_path / "trace_id.txt", "w", encoding="utf-8") as f:
            f.write(trace_id)
    if coordination_friendly_definition is not None:
        with open(save_base_path / "coordination_friendly_definition.txt", "w", encoding="utf-8") as f:
            f.write(coordination_friendly_definition)
    if additional_info is not None:
        try:
            with open(save_base_path / "additional_info.json", "w", encoding="utf-8") as f:
                json.dump(additional_info, f, indent=2, cls=NumpyEncoder)
        except Exception as e:
            logger.error(f"Failed to save additional info: {e}")
    if policy_critic_histories is not None:
        with open(save_base_path / "policy_critic_histories.json", "w", encoding="utf-8") as f:
            json.dump({i: critic_history_list_to_dict_list(policy_critic_histories) for i, policy_critic_histories in policy_critic_histories.items()}, f, indent=2, cls=NumpyEncoder)
    
    game_description = experiment_results.game_description
    if game_description is not None:
        with open(save_base_path / "game_description.txt", "w", encoding="utf-8") as f:
            f.write(game_description)

def load_compact_rollout_histories_from_json(file_path: Path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Note the rollout histories can have different lengths (if some XP crashes). Therefore we can't use np.array.
    n = len(data)
    m = len(data[0])
    compact_rollout_histories = np.empty((n, m), dtype=object)
    for i in range(n):
        for j in range(m):
            compact_rollout_histories[i, j] = [CompactRolloutHistory(**compact_rollout_history) for compact_rollout_history in data[i][j]]
    return compact_rollout_histories


def dict_to_policy(policy_dict: Dict) -> Union[CodePolicy]:
    if policy_dict["type"] == "code":
        policy_dict_copy = policy_dict.copy()
        policy_dict_copy.pop("type")
        return CodePolicy(**policy_dict_copy)
    else:
        raise ValueError(f"Invalid policy type: {policy_dict['type']}")
    
def load_policy_list_from_json(file_path: Path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    policies = [dict_to_policy(entry) for entry in data]
    return policies

def load_policy_dict_from_json(file_path: Path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {int(k): dict_to_policy(v) for k, v in data.items()}

def load_policy_list_dict_from_json(file_path: Path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {int(k): [dict_to_policy(policy_dict) for policy_dict in v] for k, v in data.items()}

def load_xp_matrix_from_json(file_path: Path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data)

def load_text_agent_config_list_from_json(file_path: Path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [dict_to_agent_config(entry) for entry in data]

def dict_to_agent_config(agent_config_dict: Dict) -> BaseTextAgentConfig:
    agent_config_dict_copy = agent_config_dict.copy()
    type = agent_config_dict_copy.pop("type")
    llm_config_dict = agent_config_dict_copy.pop("llm_config")
    if type == "chat_history":
        return ChatHistoryAgentConfig(**agent_config_dict_copy, llm_config=LLMConfig(**llm_config_dict))
    else:
        raise ValueError(f"Invalid agent config type: {type}")

def save_meta_loop_artifacts(experiment_name: str, coordination_friendly_definition: str, memory: Optional[str] = None):
    save_base_path = get_save_path(experiment_name, "code", "meta_loop")
    with open(save_base_path / "coordination_friendly_definition.txt", "w", encoding="utf-8") as f:
        f.write(coordination_friendly_definition)
    if memory is not None:
        with open(save_base_path / "memory.txt", "w", encoding="utf-8") as f:
            f.write(memory)

def load_most_recent_meta_loop_artifacts(experiment_name: str) -> Tuple[str, Optional[str]]:
    """Loads the most recent coordination-friendly definition and memory buffer from the meta-loop."""
    save_base_path = get_base_path(experiment_name, "meta_loop", mode="code")
    save_path = save_base_path / get_most_recent_timestamp_at_path(save_base_path)
    with open(save_path / "coordination_friendly_definition.txt", "r", encoding="utf-8") as f:
        coordination_friendly_definition = f.read()

    memory_path = save_path / "memory.txt"
    if memory_path.exists():
        with open(memory_path, "r", encoding="utf-8") as f:
            memory = f.read()
    else:
        memory = None
    return coordination_friendly_definition, memory

def load_code_xp_experiment_results(experiment_name: str, env_name_file_safe: str, timestamp: str) -> CodeXPExperimentResults:
    """
    Loads the experiment results from the given timestamp. An experiment can have multiple observations/timestamps.

    Args:
        experiment_name: The name of the experiment.
        env_name_file_safe: The name of the environment.
        timestamp: The timestamp of the experiment results to load.

    Returns:
        The experiment results.
    """
    results_path = get_base_path(experiment_name, "code", env_name_file_safe) / timestamp

    if not results_path.exists():
        raise ValueError(f"Experiment results not found in {results_path}")
    
    if (results_path / "selected_policies.json").exists():
        selected_policies = load_policy_dict_from_json(results_path / "selected_policies.json")
    else:
        selected_policies = None

    if (results_path / "xp_matrix.json").exists():
        xp_matrix = load_xp_matrix_from_json(results_path / "xp_matrix.json")
    else:
        xp_matrix = None

    if (results_path / "generated_policies.json").exists():
        generated_policies = load_policy_list_dict_from_json(results_path / "generated_policies.json")
    else:
        generated_policies = None

    if (results_path / "trace_id.txt").exists():
        with open(results_path / "trace_id.txt", 'r') as f:
            trace_id = f.read()
    else:
        trace_id = None

    if (results_path / "compact_rollout_histories.json").exists():
        compact_rollout_histories = load_compact_rollout_histories_from_json(results_path / "compact_rollout_histories.json")
    else:
        compact_rollout_histories = None

    coordination_friendly_definition_path = results_path / "coordination_friendly_definition.txt"
    if coordination_friendly_definition_path.exists():
        with open(coordination_friendly_definition_path, 'r') as f:
            coordination_friendly_definition = f.read()
    else:
        coordination_friendly_definition = None

    additional_info_path = results_path / "additional_info.json"
    if additional_info_path.exists():
        with open(additional_info_path, 'r') as f:
            additional_info = json.load(f)
    else:
        additional_info = None

    if (results_path / "policy_critic_histories.json").exists():
        policy_critic_histories = load_critic_history_list_dict_from_json(results_path / "policy_critic_histories.json")
    else:
        policy_critic_histories = None

    if (results_path / "game_description.txt").exists():
        with open(results_path / "game_description.txt", "r", encoding="utf-8") as f:
            game_description = f.read()
    else:
        game_description = None

    return CodeXPExperimentResults(
        experiment_name=experiment_name,
        env_name_file_safe=env_name_file_safe,
        selected_policies=selected_policies,
        generated_policies=generated_policies,
        xp_matrix=xp_matrix,
        compact_rollout_histories=compact_rollout_histories,
        trace_id=trace_id,
        coordination_friendly_definition=coordination_friendly_definition,
        additional_info=additional_info,
        policy_critic_histories=policy_critic_histories,
        game_description=game_description,
    )

def load_text_xp_experiment_results(experiment_name: str, env_name_file_safe: str, timestamp: str) -> TextXPExperimentResults:
    """
    Loads the experiment results from the given timestamp. An experiment can have multiple observations/timestamps.

    Args:
        experiment_name: The name of the experiment.
        env_name_file_safe: The name of the environment.
        timestamp: The timestamp of the experiment results to load.

    Returns:
        The experiment results.
    """
    results_path = get_base_path(experiment_name, "text", env_name_file_safe) / timestamp

    if (results_path / "agent_configs.json").exists():
        agent_configs = load_text_agent_config_list_from_json(results_path / "agent_configs.json")
    else:
        agent_configs = None

    if (results_path / "xp_matrix.json").exists():
        xp_matrix = load_xp_matrix_from_json(results_path / "xp_matrix.json")
    else:
        xp_matrix = None

    if (results_path / "compact_rollout_histories.json").exists():
        compact_rollout_histories = load_compact_rollout_histories_from_json(results_path / "compact_rollout_histories.json")
    else:
        compact_rollout_histories = None

    additional_info_path = results_path / "additional_info.json"
    if additional_info_path.exists():
        with open(additional_info_path, 'r') as f:
            additional_info = json.load(f)
    else:
        additional_info = None

    return TextXPExperimentResults(
        experiment_name=experiment_name,
        env_name_file_safe=env_name_file_safe,
        agent_configs=agent_configs,
        xp_matrix=xp_matrix,
        compact_rollout_histories=compact_rollout_histories,
        additional_info=additional_info,
    )

def get_most_recent_timestamp_at_path(path: Path) -> str:
    experiment_dirs = [d for d in path.iterdir() if d.is_dir()]
    if not experiment_dirs:
        raise ValueError(f"No experiment results found in {path}")
    return max(experiment_dirs, key=lambda x: x.name).name

def load_most_recent_experiment_results_for_env(experiment_name: str, env_name_file_safe: str, mode: Literal["code", "text"]) -> CodeXPExperimentResults:
    base_path = get_base_path(experiment_name, mode, env_name_file_safe)

    most_recent_run = get_most_recent_timestamp_at_path(base_path)

    if mode == "code":
        return load_code_xp_experiment_results(experiment_name, env_name_file_safe, most_recent_run)
    elif mode == "text":
        return load_text_xp_experiment_results(experiment_name, env_name_file_safe, most_recent_run)
    else:
        raise ValueError(f"Invalid mode: {mode}")

def load_all_experiment_results_for_env(experiment_name: str, env_name_file_safe: str, mode: Literal["code", "text"]) -> List[CodeXPExperimentResults]:
    """
    Loads all experiment results for the given experiment name from most oldest to most recent.

    Args:
        experiment_name: The name of the experiment.

    Returns:
        A list of experiment results (ordered from oldest to most recent).
    """
    base_path = get_base_path(experiment_name, mode, env_name_file_safe)

    # Get all subdirectories that match the datetime format
    experiment_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    if not experiment_dirs:
        raise ValueError(f"No experiment results found in {base_path}")

    if mode == "code":
        return [load_code_xp_experiment_results(experiment_name, env_name_file_safe, d.name) for d in sorted(experiment_dirs, key=lambda x: x.name)]
    elif mode == "text":
        return [load_text_xp_experiment_results(experiment_name, env_name_file_safe, d.name) for d in sorted(experiment_dirs, key=lambda x: x.name)]
    else:
        raise ValueError(f"Invalid mode: {mode}")