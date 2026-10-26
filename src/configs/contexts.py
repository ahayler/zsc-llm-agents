from configs.envs import (
	DEFAULT_BRITTLE_BRIDGE_CONFIG_CODE, 
	DEFAULT_PERMUTED_LEVERS_CONFIG, 
	DEFAULT_NARROW_TUNNEL_CONFIG_CODE, 
	CatAndDogConfig, CatAndDog2Config, CatAndDog3Config, CatAndDog4Config,
	DEFAULT_AD_HOC_CONVENTIONS_CONFIG, 
	DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG, 
	DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG2,
	DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG3,
	DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG4,
    DEFAULT_PERMUTED_LEVERS_RANDOM_AGENTS_CONFIG,
	CoordinationGame,
	DEFAULT_MATRIX_GAME_CONFIG,
	DEFAULT_TINY_HANABI_CONFIG,
	DEFAULT_TINY_HANABI_FIXED_CONFIG,
    DEFAULT_BRITTLE_BRIDGE_RANDOM_AGENTS_CONFIG,
	EnvConfig,
)
from typing import Literal, Type, Optional, List

from envs.cat_and_dog import CatAndDog
from envs.brittle_bridge import BrittleBridge
from envs.narrow_tunnel import NarrowTunnel
from envs.permuted_levers import PermutedLevers
from envs.ad_hoc_conventions import AdHocConventions
from envs.ad_hoc_conventions_permuted_signals import AdHocConventionsPermutedSignals
from envs.matrix_game import MatrixGame
from envs.tiny_hanabi import TinyHanabi
from envs.tiny_hanabi_fixed import TinyHanabiFixed
from enum import Enum
from pydantic import BaseModel
from envs.base import Env

"""
How do EnvConfigs, EnvContexts and MetaEnvContextrelated to each other?

You can imagine the like layers of an onion with the following layers from the inside out:
- EnvConfig: Contains exactly the information needed to instantiate one Env object.
- XPConfig: Contains exactly one EnvConfig, but also a lot more information. In general, an XPConfig should provide all the information needed to create an XP-matrix.
- MetaProblemConfig: Contains exactly one XPConfig, but also additional information. In general, a MetaProblemConfig should provide all the information needed to create a meta-sample based on the XPConfig.
"""

class XPConfig(BaseModel):
	env_config: BaseModel
	overwrites: dict
	num_seeds: int = 5
	env_name_file_safe: Optional[str] = None

class TextSPConfig(BaseModel):
    include_random_number: bool
    # Optional "coordination-friendly definition" (CFD) string injected into the prompt.
    # Keep empty to disable.
    cfd_str: str = ""
    env_config: EnvConfig
    num_game_runs: int
    

class MetaProblemConfig(BaseModel):
	xp_config: XPConfig
	sp_xp_gap_max_std_deviations: Optional[float] = 1
	sufficient_score: float


DEFAULT_XP_CONFIGS: dict[CoordinationGame, XPConfig] = {
	CoordinationGame.BRITTLE_BRIDGE: XPConfig(
		env_name="BrittleBridge",
		env_name_file_safe="brittle_bridge",
		env_config=DEFAULT_BRITTLE_BRIDGE_CONFIG_CODE,
		overwrites={
			"trainer_config.num_game_runs": 100, # SP games
			"evaluator_config.num_game_runs": 100, # XP games
		},
	),
    CoordinationGame.BRITTLE_BRIDGE_RANDOM_AGENTS: XPConfig(
        env_name="BrittleBridgeRandomAgents",
        env_name_file_safe="brittle_bridge_random_agents",
        env_config=DEFAULT_BRITTLE_BRIDGE_RANDOM_AGENTS_CONFIG,
        overwrites={
            "trainer_config.num_game_runs": 100,
            "evaluator_config.num_game_runs": 100,
        }
    ),
    CoordinationGame.CAT_AND_DOG: XPConfig(
		env_name="CatAndDog",
		env_name_file_safe="cat_and_dog",
		env_config=CatAndDogConfig(),
		overwrites={
			"trainer_config.num_game_runs": 100,
			"evaluator_config.num_game_runs": 100,
		},
	),
	CoordinationGame.CAT_AND_DOG2: XPConfig(
		env_name="CatAndDog2",
		env_name_file_safe="cat_and_dog2",
		env_config=CatAndDog2Config(),
		overwrites={
			"trainer_config.num_game_runs": 100,
			"evaluator_config.num_game_runs": 100,
		}
	),
	CoordinationGame.CAT_AND_DOG3: XPConfig(
		env_name="CatAndDog3",
		env_name_file_safe="cat_and_dog3",
		env_config=CatAndDog3Config(),
		overwrites={
			"trainer_config.num_game_runs": 100,
			"evaluator_config.num_game_runs": 100,
		}
	),
	CoordinationGame.CAT_AND_DOG4: XPConfig(
		env_name="CatAndDog4",
		env_name_file_safe="cat_and_dog4",
		env_config=CatAndDog4Config(),
		overwrites={
			"trainer_config.num_game_runs": 100,
			"evaluator_config.num_game_runs": 100,
		}
	),
	CoordinationGame.PERMUTED_LEVERS: XPConfig(
		env_name="Permuted_Levers",
		env_name_file_safe="permuted_levers",
		env_config=DEFAULT_PERMUTED_LEVERS_CONFIG,
		overwrites={
			"trainer_config.num_game_runs": 100,
			"evaluator_config.num_game_runs": 100,
		}
	),
    CoordinationGame.PERMUTED_LEVERS_RANDOM_AGENTS: XPConfig(
		env_name="Permuted_Levers_Random_Agents",
		env_name_file_safe="permuted_levers_random_agents",
		env_config=DEFAULT_PERMUTED_LEVERS_RANDOM_AGENTS_CONFIG,
		overwrites={
			"trainer_config.num_game_runs": 100,
			"evaluator_config.num_game_runs": 100,
		}
	),
	CoordinationGame.NARROW_TUNNEL: XPConfig(
		env_name="NarrowTunnel",
		env_name_file_safe="narrow_tunnel",
		env_config=DEFAULT_NARROW_TUNNEL_CONFIG_CODE,
		overwrites={
			"trainer_config.num_game_runs": 100,
			"evaluator_config.num_game_runs": 100,
		}
		),
	CoordinationGame.AD_HOC_CONVENTIONS: XPConfig(
		env_name="AdHocConventions",
		env_name_file_safe="ad_hoc_conventions",
		env_config=DEFAULT_AD_HOC_CONVENTIONS_CONFIG,
		overwrites={
			"trainer_config.num_game_runs": 100,
			"evaluator_config.num_game_runs": 100,
		}
	),
	CoordinationGame.AD_HOC_CONVENTIONS_PERMUTED_SIGNALS: XPConfig(
		env_name="AdHocConventionsPermutedSignals",
		env_name_file_safe="ad_hoc_conventions_permuted_signals",
		env_config=DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG,
		overwrites={
			"trainer_config.num_game_runs": 100,
			"evaluator_config.num_game_runs": 100,
		}
	),
	CoordinationGame.AD_HOC_CONVENTIONS_PERMUTED_SIGNALS2: XPConfig(
		env_name="AdHocConventionsPermutedSignals2",
		env_name_file_safe="ad_hoc_conventions_permuted_signals2",
		env_config=DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG2,
		overwrites={
			"trainer_config.num_game_runs": 100,
			"evaluator_config.num_game_runs": 100,
		}
	),
	CoordinationGame.AD_HOC_CONVENTIONS_PERMUTED_SIGNALS3: XPConfig(
		env_name="AdHocConventionsPermutedSignals3",
		env_name_file_safe="ad_hoc_conventions_permuted_signals3",
		env_config=DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG3,
		overwrites={
			"trainer_config.num_game_runs": 100,
			"evaluator_config.num_game_runs": 100,
		}
	),
	CoordinationGame.AD_HOC_CONVENTIONS_PERMUTED_SIGNALS4: XPConfig(
		env_name="AdHocConventionsPermutedSignals4",
		env_name_file_safe="ad_hoc_conventions_permuted_signals4",
		env_config=DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG4,
		overwrites={
			"trainer_config.num_game_runs": 100,
			"evaluator_config.num_game_runs": 100,
		}
	),
	CoordinationGame.MATRIX_GAME: XPConfig(
		env_name="MatrixGame",
		env_name_file_safe="matrix_game",
		env_config=DEFAULT_MATRIX_GAME_CONFIG,
		overwrites={
			"trainer_config.num_game_runs": 100,
			"evaluator_config.num_game_runs": 100,
		}
	),
	CoordinationGame.TINY_HANABI: XPConfig(
		env_name="TinyHanabi",
		env_name_file_safe="tiny_hanabi",
		env_config=DEFAULT_TINY_HANABI_CONFIG,
		overwrites={
			"trainer_config.num_game_runs": 100,
			"evaluator_config.num_game_runs": 100,
		}
	),
	CoordinationGame.TINY_HANABI_FIXED: XPConfig(
		env_name="TinyHanabiFixed",
		env_name_file_safe="tiny_hanabi_fixed",
		env_config=DEFAULT_TINY_HANABI_FIXED_CONFIG,
		overwrites={
			"trainer_config.num_game_runs": 100,
			"evaluator_config.num_game_runs": 100,
		}
	),
}

DEFAULT_META_PROBLEM_CONFIGS = {
	CoordinationGame.PERMUTED_LEVERS: MetaProblemConfig(
		xp_config=XPConfig(
			env_name="Permuted_Levers",
			env_name_file_safe="permuted_levers",
			env_config=DEFAULT_PERMUTED_LEVERS_CONFIG,
			num_seeds=6,
			overwrites={
				"trainer_config.generator_config.num_policies": 7,
				"trainer_config.generator_config.randomly_swap_local_policies": True,
                "trainer_config.policy_critic_config.max_iterations": 3,
				"trainer_config.num_game_runs": 300,
				"evaluator_config.num_game_runs": 150,
			}
		),
		sufficient_score=5.3,
		sp_xp_gap_max_std_deviations=1.5,
	),
	CoordinationGame.CAT_AND_DOG: MetaProblemConfig(
		xp_config=XPConfig(
			env_name="CatAndDog",
			env_name_file_safe="cat_and_dog",
			env_config=CatAndDogConfig(),
			num_seeds=5,
			overwrites={
				"trainer_config.generator_config.num_policies": 5,
				"trainer_config.num_game_runs": 100,
				"evaluator_config.num_game_runs": 50,
			}
		),
		sufficient_score=5,
		sp_xp_gap_max_std_deviations=1,
	),
	CoordinationGame.CAT_AND_DOG2: MetaProblemConfig(
		xp_config=XPConfig(
			env_name="CatAndDog2",
			env_name_file_safe="cat_and_dog2",
			env_config=CatAndDog2Config(),
			num_seeds=5,
			overwrites={
				"trainer_config.generator_config.num_policies": 5,
				"trainer_config.num_game_runs": 100,
				"evaluator_config.num_game_runs": 50,
			}
		),
		sufficient_score=5,
		sp_xp_gap_max_std_deviations=1,
	),
	CoordinationGame.CAT_AND_DOG3: MetaProblemConfig(
		xp_config=XPConfig(
			env_name="CatAndDog3",
			env_name_file_safe="cat_and_dog3",
			env_config=CatAndDog3Config(),
			num_seeds=5,
			overwrites={
				"trainer_config.generator_config.num_policies": 5,
				"trainer_config.num_game_runs": 100,
				"evaluator_config.num_game_runs": 50,
			}
		),
		sufficient_score=5,
		sp_xp_gap_max_std_deviations=1,
	),
	CoordinationGame.CAT_AND_DOG4: MetaProblemConfig(
		xp_config=XPConfig(
			env_name="CatAndDog4",
			env_name_file_safe="cat_and_dog4",
			env_config=CatAndDog4Config(),
			num_seeds=5,
			overwrites={
				"trainer_config.generator_config.num_policies": 5,
				"trainer_config.num_game_runs": 100,
				"evaluator_config.num_game_runs": 50,
			}
		),
		sufficient_score=5,
		sp_xp_gap_max_std_deviations=1,
	),
	CoordinationGame.BRITTLE_BRIDGE: MetaProblemConfig(
		xp_config=XPConfig(
			env_name="BrittleBridge",
			env_name_file_safe="brittle_bridge",
			env_config=DEFAULT_BRITTLE_BRIDGE_CONFIG_CODE,
			num_seeds=5,
			overwrites={
				"trainer_config.generator_config.num_policies": 5,
				"trainer_config.generator_config.randomly_swap_local_policies": True,
				"trainer_config.num_game_runs": 100,
				"evaluator_config.num_game_runs": 100,
			},
		),
		sufficient_score=-14,
		sp_xp_gap_max_std_deviations=1,
	),
	CoordinationGame.NARROW_TUNNEL: MetaProblemConfig(
		xp_config=XPConfig(
			env_name="NarrowTunnel",
			env_name_file_safe="narrow_tunnel",
			env_config=DEFAULT_NARROW_TUNNEL_CONFIG_CODE,
			num_seeds=5,
			overwrites={
				"trainer_config.num_game_runs": 100,
				"evaluator_config.num_game_runs": 100,
			}
		),
		sufficient_score=-9,
		sp_xp_gap_max_std_deviations=1,
	),
	CoordinationGame.AD_HOC_CONVENTIONS: MetaProblemConfig(
		xp_config=XPConfig(
			env_name="AdHocConventions",
			env_name_file_safe="ad_hoc_conventions",
			env_config=DEFAULT_AD_HOC_CONVENTIONS_CONFIG,
			num_seeds=5,
			overwrites={
				"trainer_config.generator_config.num_policies": 5,
				"trainer_config.num_game_runs": 100,
				"evaluator_config.num_game_runs": 100,
			},
		),
		sufficient_score=50,
		sp_xp_gap_max_std_deviations=1,
	),
	CoordinationGame.AD_HOC_CONVENTIONS_PERMUTED_SIGNALS: MetaProblemConfig(
		xp_config=XPConfig(
			env_name="AdHocConventionsPermutedSignals",
			env_name_file_safe="ad_hoc_conventions_permuted_signals",
			env_config=DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG,
			num_seeds=5,
			overwrites={
				"trainer_config.generator_config.num_policies": 5,
				"trainer_config.num_game_runs": 100,
				"evaluator_config.num_game_runs": 100,
			},
		),
		sufficient_score=50,
		sp_xp_gap_max_std_deviations=1,
	),
	CoordinationGame.AD_HOC_CONVENTIONS_PERMUTED_SIGNALS2: MetaProblemConfig(
		xp_config=XPConfig(
			env_name="AdHocConventionsPermutedSignals2",
			env_name_file_safe="ad_hoc_conventions_permuted_signals2",
			env_config=DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG2,
			num_seeds=5,
			overwrites={
				"trainer_config.generator_config.num_policies": 5,
				"trainer_config.num_game_runs": 100,
				"evaluator_config.num_game_runs": 100,
			},
		),
		sufficient_score=50,
		sp_xp_gap_max_std_deviations=1,
	),
	CoordinationGame.AD_HOC_CONVENTIONS_PERMUTED_SIGNALS3: MetaProblemConfig(
		xp_config=XPConfig(
			env_name="AdHocConventionsPermutedSignals3",
			env_name_file_safe="ad_hoc_conventions_permuted_signals3",
			env_config=DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG3,
			num_seeds=5,
			overwrites={
				"trainer_config.generator_config.num_policies": 5,
				"trainer_config.num_game_runs": 100,
				"evaluator_config.num_game_runs": 100,
			},
		),
		sufficient_score=50,
		sp_xp_gap_max_std_deviations=1,
	),
	CoordinationGame.AD_HOC_CONVENTIONS_PERMUTED_SIGNALS4: MetaProblemConfig(
		xp_config=XPConfig(
			env_name="AdHocConventionsPermutedSignals4",
			env_name_file_safe="ad_hoc_conventions_permuted_signals4",
			env_config=DEFAULT_AD_HOC_CONVENTIONS_PERMUTED_SIGNALS_CONFIG4,
			num_seeds=5,
			overwrites={
				"trainer_config.generator_config.num_policies": 5,
				"trainer_config.num_game_runs": 100,
				"evaluator_config.num_game_runs": 100,
			},
		),
		sufficient_score=50,
		sp_xp_gap_max_std_deviations=1,
	),
    CoordinationGame.TINY_HANABI_FIXED: MetaProblemConfig(
		xp_config=XPConfig(
			env_name="TinyHanabiFixed",
			env_name_file_safe="tiny_hanabi_fixed",
			env_config=DEFAULT_TINY_HANABI_FIXED_CONFIG,
			num_seeds=5,
			overwrites={
				"trainer_config.generator_config.num_policies": 5,
				"trainer_config.num_game_runs": 100,
				"evaluator_config.num_game_runs": 100,
			},
		),
		sufficient_score=8,
		sp_xp_gap_max_std_deviations=1,
	),
    CoordinationGame.MATRIX_GAME: MetaProblemConfig(
		xp_config=XPConfig(
			env_name="MatrixGame",
			env_name_file_safe="matrix_game",
			env_config=DEFAULT_MATRIX_GAME_CONFIG,
			num_seeds=5,
			overwrites={
				"trainer_config.generator_config.num_policies": 5,
				"trainer_config.num_game_runs": 100,
				"evaluator_config.num_game_runs": 100,
			},
		),
		sufficient_score=0.5,
		sp_xp_gap_max_std_deviations=1,
	),
}

if __name__ == "__main__":
	print(DEFAULT_META_PROBLEM_CONFIGS[CoordinationGame.BRITTLE_BRIDGE])