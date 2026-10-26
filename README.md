# Zero-Shot Coordination among LLM Agents

Code for the NeurIPS 2026 paper *Zero-Shot Coordination among LLM Agents* by Adrian Hayler, Shashank Reddy, Andrei Lupu, Johannes Forkel, Bidipta Sarkar, Siheng Feng and Jakob Foerster (BOLD, University of Oxford).

The repository contains the coordination environments, text and code agents, the hand-crafted and meta-learned coordination-friendly definitions (CFDs), the Pareto meta-training loop and the policy critic.

## Installation

Code agents execute LLM-generated Python code. For safety, the code only runs policies **inside a Docker container as a non-root user** and refuses to run them otherwise. Build the image with

```
make build
```

The `make` targets below run inside this image and mount the repository.

## API keys

Create a `.env` file in the repository root:

```
OPENROUTER_API_KEY=<YOUR_OPENROUTER_API_KEY>
LANGFUSE_SECRET_KEY=<YOUR_LANGFUSE_SECRET_KEY>   # optional, for tracing
LANGFUSE_PUBLIC_KEY=<YOUR_LANGFUSE_PUBLIC_KEY>   # optional, for tracing
LANGFUSE_HOST=https://cloud.langfuse.com         # optional, for tracing
```

All models are accessed through [OpenRouter](https://openrouter.ai). The `.env` file is excluded from the Docker image and read at runtime from the mounted repository.

## Cross-play experiments

```
make run-text-xp   # Nine core games: GPT-5.2 / Grok 4.1 Fast self-play, baseline + hand-crafted CFD
make run-code-xp   # Nine core games: five-model cross-play, baseline + hand-crafted CFD
```

The default text suite uses 100 episodes for Cat & Dog, Matrix Game and both Tiny Hanabi variants, and 50 for the other five games. Code evaluations use six independent cross-play matrices with 100 episodes per pairing. Both use temperature 0. Further suites and conditions:

```sh
make run-text-xp ARGS="--conditions baseline handcrafted learned"
make run-code-xp ARGS="--conditions baseline handcrafted learned"
make run-text-xp ARGS="--suite cat_dog_xp"                           # four models, 10 episodes per cell
make run-text-xp ARGS="--suite hanabi --conditions baseline"         # 3x3 Hanabi, one life, three hints
make run-text-xp ARGS="--suite hint_guess --conditions baseline"
make run-code-xp ARGS="--suite hint_guess --conditions baseline"     # six matrices, 100 episodes per cell
make run-code-xp ARGS="--suite semantic --conditions baseline"       # semantic selection / convention games and Trampoline Tiger
```

`src/configs/paper.py` holds the model lists, CFD texts, episode counts and failure thresholds used in the paper. An episode counts as a coordination failure if its return is at or below the environment's threshold. Text results save failure rates in `additional_info.json`, code results in `failure_rates_<matrix>.json`.

## Meta-learning CFDs (Pareto evolution)

Meta-training starts from a pool of LLM-generated policies for each training environment. Create it once, then run the Pareto evolution:

```
make run-starting-policies ARGS="--envs CAT_AND_DOG BRITTLE_BRIDGE"
make run-meta-training ARGS="--experiment_name my_run"
```

The Pareto evolution algorithm searches for CFDs using a cheap single-environment gate before the full multi-environment Pareto evaluation (driver: `ParetoEvolutionOptimizer` in `src/meta_train/population_based/algorithms/pareto_evolution.py`). Common arguments (see `--help` for the full list):

| Flag | Description | Default |
| --- | --- | --- |
| `--experiment_name` | Required. Used for the output directory and Langfuse session. | — |
| `--envs` | Space-separated `CoordinationGame` names. | `CAT_AND_DOG BRITTLE_BRIDGE` |
| `--population_size` | Initial seed CFDs. | `5` |
| `--num_steps` | Evolution steps after initialisation. | `20` |
| `--num_seeds_cheap` | Seeds for the single-environment cheap gate. | `10` |
| `--num_seeds_full` | Seeds for the full multi-environment Pareto evaluation. | `15` |
| `--optimizer_llm` / `--scoring_llm` | `LLMType` names. | `OR_GPT54` / `OR_GROK_41_FAST` |
| `--initial_state_path` | Resume from a previous `pareto_state.pkl`. | — |

Output (state checkpoints, scoring log, best CFD) is written to `logs/<experiment_name>/<timestamp>/`. During meta-training, the meta-optimiser and policy generation use temperature 1.0; the policy critic uses temperature 0 and at most four iterations.

To evaluate a learned CFD against the baseline and hand-crafted CFD on the four transfer models, use `src/meta_train/population_based/scripts/run_cfd_eval.py`, with and without `--use_critic` for the two scaffolds.

## Configuration notes

- The default model for code-policy generation and the policy critic is Grok 4.1 Fast. Set `ZSC_DEBUG=1` to use the cheaper GPT-5 mini instead for quick tests.
- Brittle Bridge uses `bridge_break_threshold=2`, i.e. agents may overlap on the bridge for one turn.

## License

Apache License 2.0; see `LICENSE`. The Hanabi environment is adapted from [JaxMARL](https://github.com/FLAIROx/JaxMARL); see `NOTICE`.

## Citation

```bibtex
@inproceedings{hayler2026zeroshot,
  title     = {Zero-Shot Coordination among {LLM} Agents},
  author    = {Hayler, Adrian and Reddy, Shashank and Lupu, Andrei and Forkel, Johannes and Sarkar, Bidipta and Feng, Siheng and Foerster, Jakob},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
