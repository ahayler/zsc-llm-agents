# Include .env file if it exists
-include .env

ID = $(shell id -u)
GID = $(shell id -g)
USER = $(shell id -un)
PWD = $(shell pwd)
WANDB_API_KEY = ""

build:
	docker build --build-arg UID=$(ID) --build-arg GID=$(GID) --build-arg MYUSER=$(USER) -t ${USER}_zsc_llm -f .devcontainer/Dockerfile .

run:
	docker run -it --network host --rm -v $(PWD):/home/${USER}/ -e WANDB_API_KEY=${WANDB_API_KEY} ${USER}_zsc_llm /bin/bash

run-text-xp:
	docker run -d --rm -v $(PWD):/home/${USER}/ -w /home/${USER}/src/ -e WANDB_API_KEY=${WANDB_API_KEY} ${USER}_zsc_llm bash -c "python -u main_text.py $(ARGS)"

run-code-xp:
	docker run -d --rm -v $(PWD):/home/${USER}/ -w /home/${USER}/src/ -e WANDB_API_KEY=${WANDB_API_KEY} ${USER}_zsc_llm bash -c "python -u main_code.py $(ARGS)"

# Starting policies for meta-training (run once before run-meta-training), e.g.
#   make run-starting-policies ARGS="--envs CAT_AND_DOG BRITTLE_BRIDGE"
run-starting-policies:
	docker run -d --rm -v $(PWD):/home/${USER}/ -w /home/${USER}/src/ -e WANDB_API_KEY=${WANDB_API_KEY} ${USER}_zsc_llm bash -c "python -u -m meta_train.population_based.scripts.create_starting_policies $(ARGS)"

# Meta-training (Pareto Evolution for CFD optimization).
# Pass CLI args via ARGS=..., e.g.
#   make run-meta-training ARGS="--experiment_name my_run --envs CAT_AND_DOG --num_steps 10"
# See src/meta_train/population_based/scripts/run_pareto_evolution.py --help for the full option list.
run-meta-training:
	docker run -d --rm -v $(PWD):/home/${USER}/ -w /home/${USER}/src/ -e WANDB_API_KEY=${WANDB_API_KEY} ${USER}_zsc_llm bash -c "python -u -m meta_train.population_based.scripts.run_pareto_evolution $(ARGS)"

check-balance:
	docker run -d --rm -v $(PWD):/home/${USER}/ -w /home/${USER}/src/ -e WANDB_API_KEY=${WANDB_API_KEY} ${USER}_zsc_llm bash -c "python -u utils/openrouter.py"