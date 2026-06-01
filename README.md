# ANYmal D — PPO + Flow Matching Imitation Learning

Reinforcement learning and imitation learning for the ANYmal D quadruped robot in MuJoCo.
This repository trains a PPO baseline and a waypoint-conditioned Flow Matching policy,
then compares them using a shared modular evaluator.

---

## Repository Structure

```
anymal_d_ppo/
├── anymal_d/                        # Original PPO trainers
│   ├── RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN.py
│   └── RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py
├── anybotics_anymal_d/              # MuJoCo model (scene.xml)
├── policies/                        # Modular policy wrappers
│   ├── base.py                      # BasePolicy ABC
│   ├── ppo_multivariate.py
│   ├── ppo_normal.py
│   ├── flow_matching.py
│   └── stubs.py
├── envs/
│   └── anymal_env.py                # Shared MuJoCo environment
├── imitation/
│   ├── flow_matching.py             # Waypoint-conditioned Flow Matching model
│   ├── waypoints.py                 # Waypoint dataset loader
│   ├── dataset.py
│   └── train_flow.py
├── evaluate.py                      # Shared CLI evaluator
├── generate_dataset.py              # PPO rollout → demo dataset
├── compare_results.py               # Comparison plots and table
├── eval_smoothness.py               # Control smoothness metrics
├── eval_config.yaml                 # Centralised evaluation config
├── data/                            # Generated datasets (not tracked)
├── pretrained_models/               # Saved checkpoints (not tracked)
└── results/                         # Evaluation outputs (not tracked)
```

---

## Installation

```bash
git clone https://github.com/nezih-niegu/anymal_d_ppo.git
cd anymal_d_ppo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo apt install ffmpeg -y
export MUJOCO_GL=egl              # headless server only
```

---

## Step 1 — Train the PPO Baseline

```bash
MUJOCO_GL=egl python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py train
```

Find the best checkpoint:

```bash
ls pretrained_models/anymal_d/*policy.pt | awk -F'Reward-' '{print $2, $0}' | sort -n | tail -1 | awk '{print $2}'
```

---

## Step 2 — Generate Demonstration Dataset

```bash
MUJOCO_GL=egl python generate_dataset.py \
    --checkpoint pretrained_models/anymal_d/<checkpoint>_policy.pt \
    --num-episodes 500 \
    --out data/ppo_demos.npz
```

---

## Step 3 — Train the Flow Matching Policy

```bash
python imitation/train_flow.py \
    --dataset data/ppo_demos.npz \
    --epochs 100 \
    --batch-size 512 \
    --out pretrained_models/flow_matching/
```

---

## Step 4 — Evaluate and Compare

```bash
# PPO
MUJOCO_GL=egl python evaluate.py \
    --policy ppo_normal \
    --checkpoint pretrained_models/anymal_d/<checkpoint>_policy.pt \
    --no-deterministic --num-episodes 30

# Flow Matching
MUJOCO_GL=egl python evaluate.py \
    --policy flow_matching \
    --checkpoint pretrained_models/flow_matching/best_flow_policy.pt \
    --num-episodes 30

# Comparison plots
python compare_results.py \
    --ppo  results/eval_ppo_normal_<timestamp>.json \
    --flow results/eval_flow_matching_<timestamp>.json \
    --out  results/comparison/

# Smoothness metrics
MUJOCO_GL=egl python eval_smoothness.py \
    --policy ppo_normal \
    --checkpoint pretrained_models/anymal_d/<checkpoint>_policy.pt \
    --policy2 flow_matching \
    --checkpoint2 pretrained_models/flow_matching/best_flow_policy.pt \
    --num-episodes 10

# Videos
MUJOCO_GL=egl python evaluate.py --policy flow_matching \
    --checkpoint pretrained_models/flow_matching/best_flow_policy.pt \
    --num-episodes 3 --save-videos
```

---

## Results Summary

| Metric                  | PPO       | Flow Matching |
|-------------------------|-----------|---------------|
| Reward mean             | **1770**  | 1632          |
| Reward std              | 516       | 596           |
| Forward velocity (m/s)  | **1.271** | 1.215         |
| Fall rate               | **30%**   | 40%           |
| Action smoothness ↓     | 15.4      | **13.7**      |
| Energy proxy ↓          | 328       | **290**       |
| Jerk proxy ↓            | 26.4      | **23.6**      |

**Key finding:** PPO achieves higher reward and lower fall rate.
Flow Matching produces smoother and more energy-efficient locomotion despite
being trained purely by imitation without any reward signal.

---

## Policy Architecture

### PPO (Trainer 2)
- Shared MLP: `[Linear(35,128), Tanh] × 2`
- Actor: `[Linear(128,128), Tanh, Linear(128,12)]`
- Distribution: `Normal(mean, exp(log_std))`

### Flow Matching
- Waypoint-conditioned OT-CFM
- Input: `[x_t(12), t_emb(32), obs(35), waypoint(3)]`
- Velocity net: 4× `[Linear(256), SiLU]`
- Waypoints: xyz body position 50 steps (0.5s) ahead
- ODE: Euler, 10 steps

---

## Observation & Action Space

| | Dim | Description |
|---|---|---|
| Observation | 35 | `qpos[2:]`(17) + `qvel`(18) |
| Action | 12 | Delta; env applies `nominal + 0.4 × tanh(a)` |
| Waypoint | 3 | xyz body position 50 steps ahead |

---

## Configuration

Centralised in `eval_config.yaml`. Override with `--config path/to/config.yaml`.
