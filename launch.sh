#!/usr/bin/env bash
# =============================================================================
# launch.sh — Local deploy / test script for ANYmal D PPO (Docker)
#
# Automatically detects:
#   • OS and CPU architecture  → sets the correct Docker build platform
#   • NVIDIA GPU (Linux only)  → selects CUDA PyTorch + GPU compose override
#   • Apple Silicon / Intel Mac → sets linux/arm64 or linux/amd64 accordingly
#
# Note on Mac GPU:
#   Docker on macOS runs inside a Linux VM. Metal/MPS is NOT accessible
#   inside containers. For MPS-accelerated training on Mac, run native Python.
#
# Works on: macOS (Intel + Apple Silicon) and Linux (x86_64 + ARM64).
# Windows: use WSL2 with Docker Desktop, then run this script from WSL.
#
# Usage:
#   chmod +x launch.sh
#   ./launch.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colours and formatting
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✔${NC}  $*"; }
warn() { echo -e "  ${YELLOW}⚠${NC}  $*"; }
err()  { echo -e "  ${RED}✖${NC}  $*"; }
info() { echo -e "  ${CYAN}→${NC}  $*"; }
sep()  { echo -e "${DIM}────────────────────────────────────────────────────────${NC}"; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# STEP 1 — Hardware detection
# Sets (and exports) these variables:
#   PLATFORM            linux/arm64 | linux/amd64
#   PYTORCH_INDEX_URL   CPU or CUDA PyTorch wheel index
#   GPU_TYPE            none | nvidia
#   OS_LABEL            human-readable OS/arch string
#   GPU_LABEL           human-readable GPU string
#   GPU_NOTE            warning shown for Mac (no MPS in Docker)
# ---------------------------------------------------------------------------
detect_hardware() {
    local _os _arch _cuda_major _gpu_name

    _os="$(uname -s)"    # Darwin | Linux
    _arch="$(uname -m)"  # x86_64 | arm64 | aarch64

    # Defaults
    GPU_TYPE="none"
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    GPU_LABEL="CPU (no GPU)"
    GPU_NOTE=""

    # ── Platform (OS × Architecture) ─────────────────────────────────────
    case "${_os}-${_arch}" in

        Darwin-arm64)
            PLATFORM="linux/arm64"
            OS_LABEL="macOS Apple Silicon (arm64)"
            GPU_NOTE="Docker on Mac runs in a Linux VM — Metal/MPS is NOT available inside containers."$'\n'"."
            ;;

        Darwin-x86_64)
            PLATFORM="linux/amd64"
            OS_LABEL="macOS Intel (x86_64)"
            GPU_NOTE="No GPU acceleration available inside Docker on Intel Mac."
            ;;

        Linux-x86_64)
            PLATFORM="linux/amd64"
            OS_LABEL="Linux x86_64"
            ;;

        Linux-aarch64 | Linux-arm64)
            PLATFORM="linux/arm64"
            OS_LABEL="Linux ARM64"
            ;;

        *)
            PLATFORM="linux/amd64"
            OS_LABEL="${_os} ${_arch} (unknown — defaulting to amd64)"
            ;;
    esac

    # ── NVIDIA GPU (Linux only) ───────────────────────────────────────────
    if [[ "${_os}" == "Linux" ]] \
        && command -v nvidia-smi &>/dev/null \
        && nvidia-smi &>/dev/null 2>&1; then

        GPU_TYPE="nvidia"

        # GPU name (first GPU only)
        _gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
                     | head -1 || echo "unknown GPU")"

        # CUDA major version from the nvidia-smi header line
        # Example: "| NVIDIA-SMI 525.x    Driver ...    CUDA Version: 12.0 |"
        _cuda_major="$(nvidia-smi 2>/dev/null \
                       | grep -o 'CUDA Version: [0-9]*' \
                       | awk '{print $3}' || echo "")"

        case "${_cuda_major}" in
            12)  PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121" ;;
            11)  PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu118" ;;
            *)   PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
                 _cuda_major="?? (defaulting to cu121)" ;;
        esac

        GPU_LABEL="${_gpu_name}  |  CUDA ${_cuda_major}.x  →  ${PYTORCH_INDEX_URL##*/}"
    fi

    # Export so child processes (docker compose) and the build args pick them up
    export PLATFORM
    export DOCKER_DEFAULT_PLATFORM="${PLATFORM}"
    export PYTORCH_INDEX_URL
    export GPU_TYPE
}

# ---------------------------------------------------------------------------
# Print hardware summary banner
# ---------------------------------------------------------------------------
print_hardware_banner() {
    echo
    echo -e "${BOLD}${CYAN}┌─ Hardware Detection ──────────────────────────────────────┐${NC}"
    echo -e "${BOLD}${CYAN}│${NC}  OS / Arch  : ${BOLD}${OS_LABEL}${NC}"
    echo -e "${BOLD}${CYAN}│${NC}  Platform   : ${BOLD}${PLATFORM}${NC}"
    echo -e "${BOLD}${CYAN}│${NC}  Torch      : ${BOLD}${GPU_LABEL}${NC}"
    if [[ -n "${GPU_NOTE:-}" ]]; then
        # Wrap note across two lines cleanly
        echo -e "${BOLD}${CYAN}│${NC}  ${YELLOW}⚠${NC} ${DIM}${GPU_NOTE}${NC}"
    fi
    if [[ "${GPU_TYPE}" == "nvidia" ]]; then
        echo -e "${BOLD}${CYAN}│${NC}  ${DIM}Compose     : docker-compose.yml + docker-compose.gpu.yml${NC}"
    fi
    echo -e "${BOLD}${CYAN}└───────────────────────────────────────────────────────────┘${NC}"
    echo
}

# ---------------------------------------------------------------------------
# STEP 2 — Prerequisites
# ---------------------------------------------------------------------------
check_prerequisites() {
    echo -e "${BOLD}Checking prerequisites…${NC}"
    sep

    # Docker binary
    if ! command -v docker &>/dev/null; then
        err "Docker not found."
        echo "    macOS / Windows → https://docs.docker.com/desktop/"
        echo "    Linux           → https://docs.docker.com/engine/install/"
        exit 1
    fi
    ok "Docker   : $(docker --version)"

    # Docker daemon
    if ! docker info &>/dev/null; then
        err "Docker daemon is not running."
        echo "    macOS / Windows: open Docker Desktop and wait for the whale to stop animating."
        echo "    Linux:           sudo systemctl start docker"
        exit 1
    fi
    ok "Daemon   : running"

    # docker compose v2 or v1 fallback
    if docker compose version &>/dev/null 2>&1; then
        _BASE_COMPOSE="docker compose"
        ok "Compose  : $(docker compose version --short 2>/dev/null || echo v2)"
    elif command -v docker-compose &>/dev/null; then
        _BASE_COMPOSE="docker-compose"
        warn "docker-compose v1 (standalone) found. Upgrade to Docker Desktop for v2."
    else
        err "docker compose not found."
        echo "    Install Docker Desktop (includes Compose v2), or:"
        echo "    sudo apt-get install docker-compose-plugin"
        exit 1
    fi

    # NVIDIA Container Toolkit check (Linux + GPU only)
    if [[ "${GPU_TYPE}" == "nvidia" ]]; then
        if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 \
               nvidia-smi &>/dev/null 2>&1; then
            ok "NVIDIA   : Container Toolkit working"
        else
            warn "NVIDIA Container Toolkit may not be installed or configured."
            warn "GPU services will likely fail. Install guide:"
            warn "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            warn "Continuing with CPU-only compose (safe fallback)."
            GPU_TYPE="none"
            PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
            export GPU_TYPE PYTORCH_INDEX_URL
        fi
    fi

    sep
}

# ---------------------------------------------------------------------------
# Build the COMPOSE variable (base + optional GPU override)
# ---------------------------------------------------------------------------
set_compose_command() {
    if [[ "${GPU_TYPE}" == "nvidia" ]]; then
        COMPOSE="${_BASE_COMPOSE} -f docker-compose.yml -f docker-compose.gpu.yml"
        info "Using GPU compose stack (docker-compose.yml + docker-compose.gpu.yml)"
    else
        COMPOSE="${_BASE_COMPOSE}"
    fi
}

# ---------------------------------------------------------------------------
# STEP 3 — .env file
# ---------------------------------------------------------------------------
check_env_file() {
    echo -e "${BOLD}Environment configuration…${NC}"
    sep
    if [[ -f ".env" ]]; then
        ok ".env found — HF credentials will be passed to the container."
    else
        warn ".env not found → running fully offline (no HF Hub upload)."
        info "To enable cloud features:  cp .env.example .env  then fill in your tokens."
    fi
    sep
}

# ---------------------------------------------------------------------------
# STEP 4 — Build image
# ---------------------------------------------------------------------------
build_image() {
    local IMAGE_NAME="anymal-d-ppo:latest"

    echo -e "${BOLD}Docker image…${NC}"
    sep

    if docker image inspect "${IMAGE_NAME}" &>/dev/null; then
        ok "Image '${IMAGE_NAME}' already exists."
        local img_size
        img_size=$(docker image inspect "${IMAGE_NAME}" --format='{{.Size}}' \
                   | awk '{printf "%.1f GB", $1/1073741824}')
        info "Size: ${img_size}"

        echo
        read -rp "$(echo -e "  ${YELLOW}Rebuild? [y/N]: ${NC}")" _rebuild
        if [[ "$_rebuild" =~ ^[yY]$ ]]; then
            echo
            info "Building (platform=${PLATFORM}, torch=${PYTORCH_INDEX_URL##*/})…"
            $COMPOSE build
            ok "Image rebuilt."
        fi
    else
        info "Image not found — building now."
        info "Platform : ${PLATFORM}"
        info "PyTorch  : ${PYTORCH_INDEX_URL##*/}  (${PYTORCH_INDEX_URL})"
        info "First build takes ~5–10 min (downloads ~1–5 GB). Subsequent builds: ~30 sec."
        echo
        $COMPOSE build
        ok "Image built successfully."
    fi

    # Final size
    local final_size
    final_size=$(docker image inspect "${IMAGE_NAME}" --format='{{.Size}}' 2>/dev/null \
                 | awk '{printf "%.1f GB", $1/1073741824}' || echo "unknown")
    info "Image size: ${BOLD}${final_size}${NC}"
    sep
}

# ---------------------------------------------------------------------------
# Ensure output directories exist on the HOST before mounting.
# If Docker creates them they'll be owned by root, which causes permission
# issues when the host user tries to read the checkpoints.
# ---------------------------------------------------------------------------
ensure_output_dirs() {
    mkdir -p pretrained_models/anymal_d/videos logs
}

# ---------------------------------------------------------------------------
# STEP 5 — Main menu
# ---------------------------------------------------------------------------
run_menu() {
    while true; do
        echo
        echo -e "${BOLD}${CYAN}═══ ANYmal D PPO — Launch Menu ════════════════════════════${NC}"
        echo -e "  ${BOLD}1${NC}  Train           polished trainer (recommended)"
        echo -e "  ${BOLD}2${NC}  Train basic     MultivariateNormal trainer"
        echo -e "  ${BOLD}3${NC}  Sweep           hyperparameter random search"
        echo -e "  ${BOLD}4${NC}  Render videos   from polished-trainer checkpoint"
        echo -e "  ${BOLD}5${NC}  Render videos   from basic-trainer checkpoint"
        echo -e "  ${BOLD}6${NC}  Shell           bash session inside the container"
        echo -e "  ${BOLD}7${NC}  Hardware info   show detected platform / GPU"
        echo -e "  ${BOLD}8${NC}  Show outputs    list checkpoints and videos on disk"
        echo -e "  ${BOLD}9${NC}  Stop & clean    remove all stopped containers"
        echo -e "  ${BOLD}d${NC}  Dashboard       open trackio training dashboard (port 7860)"
        echo -e "  ${BOLD}q${NC}  Quit"
        echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${NC}"
        echo
        read -rp "$(echo -e "${BOLD}  Choice: ${NC}")" _choice

        case "${_choice}" in

        1)
            sep
            info "Starting polished trainer…"
            info "Outputs → pretrained_models/anymal_d/"
            info "Press Ctrl+C to stop early (checkpoint is saved each episode)."
            sep
            $COMPOSE run --rm train
            ok "Done. Check pretrained_models/anymal_d/ for checkpoints."
            ;;

        2)
            sep
            info "Starting basic trainer (MultivariateNormal)…"
            sep
            $COMPOSE run --rm train-basic
            ok "Done."
            ;;

        3)
            read -rp "$(echo -e "${BOLD}  Sweep trials [default 10]: ${NC}")" _n
            _n="${_n:-10}"
            sep
            info "Starting sweep — ${_n} trials…"
            sep
            $COMPOSE run --rm sweep \
                anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py sweep \
                --sweep-count "${_n}"
            ok "Sweep done."
            ;;

        4)
            read -rp "$(echo -e "${BOLD}  Number of videos [default 3]: ${NC}")" _n
            _n="${_n:-3}"
            sep
            info "Rendering ${_n} video(s) from polished-trainer checkpoint…"
            sep
            $COMPOSE run --rm render \
                anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py render \
                --num-videos "${_n}"
            ok "Videos saved to pretrained_models/anymal_d/videos/"
            ;;

        5)
            read -rp "$(echo -e "${BOLD}  Number of videos [default 3]: ${NC}")" _n
            _n="${_n:-3}"
            sep
            info "Rendering ${_n} video(s) from basic-trainer checkpoint…"
            sep
            $COMPOSE run --rm video \
                anymal_d/RL_PPO_ANYMAL_D_VIDEO.py \
                --num-videos "${_n}"
            ok "Videos saved to pretrained_models/anymal_d/videos/"
            ;;

        6)
            sep
            info "Opening bash shell inside the container (type 'exit' to return)."
            sep
            $COMPOSE run --rm shell
            ;;

        7)
            print_hardware_banner
            info "PYTORCH_INDEX_URL = ${PYTORCH_INDEX_URL}"
            info "DOCKER_DEFAULT_PLATFORM = ${DOCKER_DEFAULT_PLATFORM}"
            info "Compose command = ${COMPOSE}"
            if [[ "${GPU_TYPE}" == "none" ]]; then
                warn "No GPU acceleration inside Docker."
                if [[ "$(uname -s)" == "Darwin" ]]; then
                    warn "On Mac: run native Python for Metal/MPS."
                fi
            fi
            ;;

        8)
            sep
            echo -e "${BOLD}  Checkpoints${NC}"
            find pretrained_models/ -name "*.pt" -o -name "*.pth" 2>/dev/null \
                | sort \
                | while read -r f; do
                    printf "    %-10s %s\n" "[$(du -sh "$f" 2>/dev/null | cut -f1)]" "$f"
                done || true
            [[ -z "$(find pretrained_models/ \( -name '*.pt' -o -name '*.pth' \) 2>/dev/null)" ]] \
                && echo "    (none yet)"

            echo
            echo -e "${BOLD}  Videos${NC}"
            find pretrained_models/ -name "*.mp4" 2>/dev/null \
                | sort \
                | while read -r f; do
                    printf "    %-10s %s\n" "[$(du -sh "$f" 2>/dev/null | cut -f1)]" "$f"
                done || true
            [[ -z "$(find pretrained_models/ -name '*.mp4' 2>/dev/null)" ]] \
                && echo "    (none yet)"

            echo
            echo -e "${BOLD}  Logs${NC}"
            find logs/ -type f 2>/dev/null | sort || echo "    (none yet)"
            sep
            ;;

        9)
            sep
            info "Stopping and removing containers…"
            $COMPOSE down --remove-orphans
            ok "Done."
            sep
            ;;

        d | D)
            sep
            info "Starting trackio dashboard on http://localhost:7860 …"
            info "Press Ctrl+C to stop the dashboard and return to the menu."
            sep
            $COMPOSE up dashboard
            ;;

        q | Q)
            echo
            ok "Bye!"
            exit 0
            ;;

        *)
            warn "Unknown option '${_choice}'. Pick a number from the menu."
            ;;
        esac

        echo
        read -rp "$(echo -e "${DIM}  Press Enter to return to menu…${NC}")" _
    done
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
main() {
    cd "${PROJECT_DIR}"

    detect_hardware
    print_hardware_banner

    check_prerequisites
    set_compose_command
    check_env_file
    build_image
    ensure_output_dirs

    run_menu
}

main "$@"
