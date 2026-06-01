# ANYmal D PPO

Reinforcement learning locomotion for the [ANYmal D](https://www.anybotics.com/anymal/) quadruped
using Proximal Policy Optimisation (PPO) and MuJoCo.

Experiment tracking is handled by **[trackio](https://github.com/huggingface/trackio)** (a
wandb-compatible API) and optional checkpoint upload to **Hugging Face Hub** — no cloud accounts
are required; everything runs fully offline.

---

## Repository structure

```
anymal_d_ppo/
├── anymal_d/                   ← Python package (import anymal_d)
│   ├── config/                 ← Hyperparameter management
│   │   ├── config.py           ·   PPOConfig dataclass + YAML loader
│   │   ├── default_basic.yaml  ·   defaults for the basic trainer
│   │   └── default_advanced.yaml · defaults for the advanced trainer
│   ├── envs/                   ← MuJoCo environment wrappers
│   │   ├── base_env.py         ·   shared model loading, camera, lazy renderer
│   │   ├── basic_env.py        ·   37-dim obs, simple reward (basic trainer)
│   │   └── advanced_env.py     ·   35-dim obs, shaped reward, live viewer
│   ├── policies/               ← Actor-critic networks
│   │   ├── basic_agent.py      ·   MultivariateNormal + fixed external std
│   │   └── advanced_agent.py   ·   Normal + learnable log_std parameter
│   ├── training/               ← Training utilities
│   │   ├── replay_memory.py    ·   circular buffer; signals when full
│   │   ├── ppo.py              ·   train_basic / train_advanced update steps
│   │   └── sweep.py            ·   random-search sweep driver (replaces wandb.sweep)
│   ├── evaluation/             ← Evaluation and media generation
│   │   └── evaluator.py        ·   evaluate_episode() — unified renderer + plotter
│   ├── checkpoints/            ← Checkpoint I/O
│   │   └── manager.py          ·   save / load / pick_best / HF Hub upload
│   ├── tracking/               ← Experiment tracking
│   │   └── tracker.py          ·   thin trackio wrapper (init / log / finish)
│   └── utils/                  ← Shared utilities
│       └── paths.py            ·   find_project_file(), SAVE_DIR, VIDEO_DIR
│
├── scripts/                    ← Entry-point scripts (thin orchestrators)
│   ├── train_advanced.py       ·   polished trainer: learnable log_std ← start here
│   ├── train_basic.py          ·   basic trainer: MultivariateNormal
│   └── render.py               ·   render eval videos from any checkpoint
│
├── tests/                      ← Automated test suite
│   ├── test_imports.py         ·   every public module imports cleanly
│   ├── test_config.py          ·   YAML loading, overrides, round-trip
│   ├── test_env.py             ·   reset/step shape and value checks
│   └── test_smoke.py           ·   fill buffer + run one PPO update end-to-end
│
├── anybotics_anymal_d/         ← Robot model (MuJoCo MJCF, unchanged)
├── Dockerfile
├── docker-compose.yml
├── docker-compose.gpu.yml
├── launch.sh
├── pyproject.toml
└── requirements.txt
```

---

## Module guide

| Module | What it provides |
|---|---|
| `anymal_d.config` | `PPOConfig` dataclass; `from_yaml`, `from_dict`, `with_overrides` |
| `anymal_d.envs` | `BasicEnv` (37-dim), `AdvancedEnv` (35-dim, shaped reward) |
| `anymal_d.policies` | `BasicAgent` (fixed std), `AdvancedAgent` (learnable log_std) |
| `anymal_d.training` | `ReplayMemory`, `train_basic`, `train_advanced`, `run_sweep` |
| `anymal_d.evaluation` | `evaluate_episode` — renders video + reward plot, logs to tracker |
| `anymal_d.checkpoints` | `save_checkpoint`, `load_checkpoint`, `pick_best_checkpoint`, `push_to_hub` |
| `anymal_d.tracking` | `init_run`, `log`, `log_video`, `log_image`, `finish`, `is_active` |
| `anymal_d.utils` | `find_project_file`, `MODEL_XML`, `SAVE_DIR`, `VIDEO_DIR` |

---

## Installation

```bash
# Clone
git clone https://github.com/nezih-niegu/anymal_d_ppo.git
cd anymal_d_ppo

# Install dependencies + package in editable mode
pip install -r requirements.txt
pip install -e .
```

> **Headless rendering** (no display): set `MUJOCO_GL=osmesa` and install
> `libosmesa6` (Debian/Ubuntu: `sudo apt-get install libosmesa6`).

---

## Usage

### Local Python

```bash
# Advanced trainer (recommended)
python scripts/train_advanced.py train
python scripts/train_advanced.py train --live    # interactive MuJoCo viewer
python scripts/train_advanced.py sweep --sweep-count 20

# Basic trainer (MultivariateNormal)
python scripts/train_basic.py
python scripts/train_basic.py --sweep --sweep-count 20

# Render evaluation videos
python scripts/render.py                         # best advanced checkpoint
python scripts/render.py --basic                 # best basic checkpoint
python scripts/render.py --policy path/to/x_policy.pt
```

### Docker (recommended for reproducibility)

```bash
# Auto-detect hardware and launch an interactive menu
./launch.sh

# Or use docker compose directly
docker compose up train           # advanced trainer (CPU)
docker compose up train-basic
docker compose up sweep
docker compose up render
docker compose up dashboard       # trackio at http://localhost:7860
```

For GPU acceleration, merge the GPU override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up train
```

---

## Optional environment variables

| Variable | Purpose |
|---|---|
| `HF_MODEL_REPO` | `user/model-name` — enables checkpoint upload to HF Hub |
| `HF_TOKEN` | Hugging Face write token (or use `huggingface-cli login`) |
| `HF_PRIVATE` | `1` (default) = private repo, `0` = public |
| `TRACKIO_SPACE_ID` | `user/space-name` — host the dashboard on HF Spaces |

Copy `.env.example` to `.env` and fill in the values you need.

---

## Using the API

The modular structure exposes reusable components for custom experiments:

```python
from anymal_d.config import PPOConfig
from anymal_d.envs.advanced_env import AdvancedEnv, N_OBS, N_ACT
from anymal_d.policies.advanced_agent import AdvancedAgent
from anymal_d.training.replay_memory import ReplayMemory, make_advanced_dtype
from anymal_d.training.ppo import train_advanced
from anymal_d.evaluation.evaluator import evaluate_episode
from anymal_d.checkpoints.manager import save_checkpoint, load_checkpoint

# Load config from YAML, then override specific values
cfg = PPOConfig.from_yaml("anymal_d/config/default_advanced.yaml")
cfg = cfg.with_overrides({"lr": 1e-4, "ppo_epoch": 5})

# Create components
env    = AdvancedEnv(fall_threshold=cfg.fall_threshold)
policy = AdvancedAgent(N_OBS, N_ACT)
memory = ReplayMemory(cfg.replay_size, make_advanced_dtype(N_OBS, N_ACT))

# Load a checkpoint
policy = load_checkpoint("pretrained_models/anymal_d/run_1000_Reward-500_policy.pt")

# Evaluate
ep_reward, video_path = evaluate_episode(env, policy, episode=1, log_media=False)
```

---

## CI/CD pipeline

The unified workflow (`.github/workflows/ci.yml`) runs on every push and PR:

| Step | What is verified |
|---|---|
| **Format** | `black --check` over `anymal_d/`, `scripts/`, `tests/` |
| **Import tests** | Every public module imports cleanly (`test_imports.py`) |
| **Config tests** | YAML loading, dataclass round-trip, overrides (`test_config.py`) |
| **Environment tests** | `reset()` / `step()` shape and value correctness (`test_env.py`) |
| **Smoke training** | Fill a tiny replay buffer + one full PPO update — losses must be finite (`test_smoke.py`) |
| **Docker build** | Multi-platform image builds without error |

The smoke training test is deliberately lightweight (100-transition buffer, 2 PPO epochs) so CI
completes in under a minute while still exercising the full forward/backward pass for both trainer
variants.

---

## Caveats

- **The two policy variants are not interchangeable.** A checkpoint from `train_basic.py` must be
  rendered with `scripts/render.py --basic`; an advanced checkpoint does not take `--basic`.
- **ANYmal D ≠ C.** The robot geometry and mass distribution differ; reward scale and hyperparameters
  may need retuning when porting from C experiments.
- **HAA joint remap.** The front/hind hip abduction joints have asymmetric travel limits
  (`~[-0.785, 0.611] rad`). The `BasicAgent` applies `0.6 * x ± 0.1` to stay within limits.
  `AdvancedAgent` uses delta-around-nominal with `ACTION_SCALE = 0.4 rad`.
- **CPU-only PyTorch on Linux** may segfault due to a `triton` conflict. Use the Docker image or
  install PyTorch from the `--index-url https://download.pytorch.org/whl/cpu` wheel.
