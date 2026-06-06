# PPO for ANYmal D (MuJoCo) — Hugging Face

This is the ANYmal **D** version of the ANYmal C PPO project. The three training
scripts are direct ports of the ANYmal C set, with two kinds of change only:

1. **ANYmal D.** They load the ANYmal D model (`anybotics_anymal_d/scene.xml`) and
   save to `pretrained_models/anymal_d/`. The robot's kinematics are identical
   between C and D (12 leg joints, 19-dim `qpos`, 18-dim `qvel`), so the network
   sizes, action mapping, and reward logic are unchanged. The standing pose and
   height target are read from the model's `home` keyframe, so they track the D
   XML automatically.
2. **Hugging Face.** All experiment tracking now uses the Hugging Face
  stack instead of Weights & Biases, and the scripts can upload checkpoints,
  rollout videos, reward plots, and metadata to a Hugging Face model repo.

## Layout

```
anymal_d_ppo/
├── anymal_d/                                  # the scripts
│   ├── hf_artifacts.py                            # shared Hub upload / metadata helper
│   ├── RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN.py          # MultivariateNormal trainer + sweep
│   ├── RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py# Normal/delta trainer + sweep + render
│   └── RL_PPO_ANYMAL_D_VIDEO.py                   # render videos from a checkpoint
├── anybotics_anymal_d/                        # the MuJoCo model (scene.xml, assets, …)
├── pretrained_models/anymal_d/                # checkpoints, videos, metadata
├── requirements.txt
└── README.md
```

The scripts find `anybotics_anymal_d/scene.xml` by walking up from their own
location, so you can run them from anywhere in the project.

## Install

```bash
pip install -r requirements.txt
# headless rendering (Linux): apt-get install libosmesa6 && pip install PyOpenGL
```

## Train And Export

```bash
# 1) Older trainer (MultivariateNormal policy, raw-action control)
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN.py            # single run
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN.py --sweep    # random-search sweep

# 2) Polished trainer (learnable log_std, delta-around-nominal, shaped reward)
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py train
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py train --live      # live viewer
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py sweep --sweep-count 30
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py render --num-videos 5

# 3) Make evaluation videos from the best checkpoint
python anymal_d/RL_PPO_ANYMAL_D_VIDEO.py --num-videos 5
# headless: prefix any of the above with  MUJOCO_GL=osmesa
```

Trainer **1** pairs with the video script because they share the same
`MultivariateNormal` policy class. Trainer **2** renders its own videos via its
`render` mode.

The new shared helper [anymal_d/hf_artifacts.py](anymal_d/hf_artifacts.py)
handles metadata generation and Hugging Face uploads for checkpoints, rollout
videos, and plots, so the training scripts only need to pass the artifact paths
and run metadata.

When a run crosses the save threshold, the scripts write local artifacts under
`pretrained_models/anymal_d/` and a matching metadata JSON file. If
`HF_MODEL_REPO` is set, those files are uploaded to the Hub with the following
layout:

- `checkpoints/` for saved policy and optimizer checkpoints.
- `videos/` for rollout MP4s.
- `plots/` for reward curves.
- `metadata/` for the run manifest JSON files.


## Hugging Face Upload

Everything runs **fully offline** without these. Set them to enable the cloud bits:

```bash
export HF_MODEL_REPO="your-username/anymal-d-ppo"   # enable Hub upload
export HF_TOKEN="hf_..."                            # write token (or: hf auth login)
export HF_PRIVATE=1                                 # 1=private repo (default), 0=public
export TRACKIO_SPACE_ID="your-username/anymal-d-dash"  # host the dashboard on a Space
```

To train and upload the generated artifacts in one pass:

```bash
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py train
python anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py render --num-videos 3
```

View the local dashboard any time with:

```bash
trackio show --project AIDL-PPO-ANYMAL_D
```

## Notes / caveats

- The two trainers are intentionally different policies (the C set shipped both):
  trainer 1 uses a `MultivariateNormal` with an annealed action std; trainer 2 uses
  a `Normal` with a learnable `log_std` and acts as a small delta around the
  nominal pose. Their checkpoints are **not** interchangeable.
- ANYmal D's link masses/geometry differ from C (the thigh is notably heavier),
  so reward magnitudes and the episodes needed to "solve" will differ — expect to
  re-tune rather than reuse C's numbers. The HAA action remap (`0.6*x ± 0.1`) sits
  inside D's hip limits (~[-0.785, 0.611]).
- If `loss.backward()` segfaults on a CPU-only box via a GPU torch build, remove
  the stray `triton` package (`pip uninstall triton`); torch then falls back cleanly.
```
