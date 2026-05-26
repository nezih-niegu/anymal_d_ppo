# =============================================================================
# ANYmal D PPO — Docker Image
# =============================================================================
# Base:      Python 3.11 slim (Debian Bookworm)
# Rendering: OSMesa (offscreen/headless OpenGL — works without a display)
#
# Build argument — PYTORCH_INDEX_URL
#   Controls which PyTorch wheel is installed. launch.sh sets this
#   automatically based on the detected hardware. You can also pass it
#   manually:
#
#   CPU only (default, ~1.5 GB image):
#     docker compose build
#
#   NVIDIA CUDA 12.x (~5 GB image):
#     PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 docker compose build
#
#   NVIDIA CUDA 11.8 (~5 GB image):
#     PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 docker compose build
#
# Note on Mac (Apple Silicon / Intel):
#   Docker on macOS runs inside a Linux VM — Metal/MPS is NOT accessible
#   inside the container. GPU acceleration on Mac requires native Python.
#   The image will always run on CPU on any Mac.
#
# Note on the training scripts:
#   Both trainers define `device = torch.device("cuda" if ...)` but never
#   call .to(device) on the model or tensors — training runs on CPU even
#   if CUDA is available. The CUDA build still matters for the build arg
#   (future fix: add .to(device) throughout the training loop).
# =============================================================================

# Build argument — injected by docker-compose.yml from the PYTORCH_INDEX_URL
# environment variable set by launch.sh. Defaults to the CPU-only wheel.
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

FROM python:3.11-slim AS base

# ---------------------------------------------------------------------------
# System dependencies
#   libosmesa6        offscreen OpenGL renderer (MUJOCO_GL=osmesa)
#   libgl1            OpenGL shared libs that MuJoCo links against
#   libglib2.0-0      required by OpenGL/GLib internals
#   ffmpeg            mediapy uses it to encode MP4 videos
#   wget + ca-certs   for any runtime downloads (HF hub, etc.)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies — two-step so the heavy torch layer is cached
# separately from the lighter packages. Changing requirements.txt only
# re-downloads the non-torch packages (much faster on rebuilds).
# ---------------------------------------------------------------------------
WORKDIR /app

# The ARG must be re-declared after FROM to be visible in RUN commands.
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Step 1 — PyTorch (separate layer so it caches independently of requirements.txt)
# The index URL is set by the build arg above — CPU by default, CUDA when
# launch.sh detects an NVIDIA GPU and passes the appropriate URL.
RUN pip install --no-cache-dir \
        torch \
        --index-url ${PYTORCH_INDEX_URL}

# Step 2 — Remaining packages (torch already satisfied, so pip skips it)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Project files
# Only copy what is actually needed at runtime. The pretrained_models/
# directory is intentionally excluded — it comes in via a volume mount
# so checkpoints are written to (and persist on) the host machine.
# ---------------------------------------------------------------------------
COPY anybotics_anymal_d/ ./anybotics_anymal_d/
COPY anymal_d/            ./anymal_d/

# Output directories (volume mounts will overlay these at runtime)
RUN mkdir -p pretrained_models/anymal_d/videos logs

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
# Force offscreen rendering — required inside a container (no display server).
ENV MUJOCO_GL=osmesa

# Matplotlib must not try to open a GUI window.
ENV MPLBACKEND=Agg

# Keep Python output unbuffered so logs appear in real time.
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Default command: polished trainer, single training run.
# Override via `docker run` or docker-compose `command:` key.
# ---------------------------------------------------------------------------
ENTRYPOINT ["python"]
CMD ["anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py", "train"]
