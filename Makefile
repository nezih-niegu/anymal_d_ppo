COMPOSE_BASE := -f .devcontainer/docker-compose.yml
COMPOSE_GPU  := -f .devcontainer/docker-compose.yml -f .devcontainer/docker-compose.gpu.yml

.PHONY: up up-gpu down shell shell-gpu logs

## Start container (CPU — works on any machine)
up:
	docker compose $(COMPOSE_BASE) up -d

## Start container (GPU — requires NVIDIA Container Toolkit)
up-gpu:
	docker compose $(COMPOSE_GPU) up -d

## Stop and remove the container
down:
	docker compose $(COMPOSE_BASE) down

## Open a shell in the running container (CPU)
shell:
	docker exec -it anymal_d_ppo_dev bash

## Open a shell in the running container (GPU)
shell-gpu:
	docker compose $(COMPOSE_GPU) up -d
	docker exec -it anymal_d_ppo_dev bash

## Show container logs
logs:
	docker compose $(COMPOSE_BASE) logs -f
