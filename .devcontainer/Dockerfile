# =============================================================================
# ANYmal D PPO — Dev Container Image
# =============================================================================
# A interactive development container matching the style of TE3002B_mcrChallenge.
# Features:
#   - Non-root user (dev, UID 1000) with sudo access
#   - tmux for terminal multiplexing
#   - Full project Python environment (torch with CUDA via PyPI)
#   - Offscreen MuJoCo rendering (OSMesa)
#   - GPU passthrough via NVIDIA Container Toolkit
# =============================================================================

FROM python:3.11-slim

# ---------------------------------------------------------------------------
# Non-root user setup (mirrors TE3002B devcontainer convention)
# ---------------------------------------------------------------------------
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ---------------------------------------------------------------------------
# System dependencies
#   libosmesa6 + libgl1   offscreen OpenGL for MuJoCo (MUJOCO_GL=osmesa)
#   libglib2.0-0          required by OpenGL/GLib internals
#   ffmpeg                mediapy uses it to encode MP4 videos
#   git, wget, curl       common dev tools
#   tmux                  terminal multiplexer
#   ca-certificates       HTTPS
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        git \
        wget \
        curl \
        tmux \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies — installed as root so all users share the environment
# ---------------------------------------------------------------------------

# Step 1 — PyTorch (CUDA-enabled wheel from PyPI — no download.pytorch.org needed)
RUN pip install --no-cache-dir torch

# Step 2 — Remaining project packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Switch to non-root user for interactive sessions
# ---------------------------------------------------------------------------
USER $USERNAME
WORKDIR /home/$USERNAME

# Set Python unbuffered
RUN echo 'export PYTHONUNBUFFERED=1' >> /home/$USERNAME/.bashrc

ENV SHELL=/bin/bash
