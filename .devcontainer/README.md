# Devcontainer Environment

This folder contains the Docker configuration for a reproducible development environment used for training and development.

Overview

- Provides a non-root user `dev` and a workspace mounted at `/home/dev/ws`.
- Includes `tmux` for long-running sessions, GPU passthrough support for NVIDIA GPUs, and pre-installed dependencies required by the project (Python 3.11, PyTorch with CUDA, MuJoCo, and system libraries).

Makefile and Docker Compose

There are two Compose files in this folder:

- `docker-compose.yml` — base configuration for CPU-only environments.
- `docker-compose.gpu.yml` — override to enable NVIDIA GPU integration.

The `Makefile` in the repository root provides shortcuts that combine these files. For example, `make up-gpu` runs the equivalent of:

```bash
docker compose -f .devcontainer/docker-compose.yml -f .devcontainer/docker-compose.gpu.yml up -d
```

How to use

1) Using VS Code

- Install the Dev Containers extension.
- Open the project in a container. The default configuration (`devcontainer.json`) is for CPU users; choose the GPU configuration (`devcontainer.gpu.json`) if you need GPU passthrough.

2) Using the terminal

From the project root run:

```bash
# Start the container with GPU support
make up-gpu

# Open a shell inside the container
make shell
```

Inside the container your working directory will be `/home/dev/ws`. Files edited on the host are reflected in the container.

Starting training

Once inside the container you can start a detached tmux session and run training:

```bash
tmux new -s main
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py train
```

Notes

- The environment is designed to run offline; cloud integrations (Hugging Face, tracking) are optional and require additional environment variables and credentials.
- If you need to start the GPU container without the `Makefile`, use the `docker compose -f` command shown above.
