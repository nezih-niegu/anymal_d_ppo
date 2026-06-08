#!/usr/bin/env python3
"""
generate_dataset.py — now saves xyz body position for waypoint conditioning.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "anymal_d"))

import RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING as _trainer
import __main__

__main__.Agent = _trainer.Agent

from policies.ppo_normal import PPONormalPolicy
from envs.anymal_env import ANYmalEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--out", default="data/ppo_demos.npz")
    parser.add_argument("--min-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    policy = PPONormalPolicy(deterministic=False)
    policy.load(args.checkpoint)
    policy.set_eval_mode()
    print(f"Loaded policy from {args.checkpoint}")

    env = ANYmalEnv(
        action_convention="delta",
        obs_convention="noxy",
        reward_fn="shaped",
        fall_threshold=0.35,
    )

    all_obs, all_actions, all_xyz, all_ep_ids, all_rewards = [], [], [], [], []
    kept, discarded = 0, 0
    t0 = time.perf_counter()

    print(f"Collecting {args.num_episodes} episodes (min_steps={args.min_steps})...")
    print("-" * 60)

    for ep in range(args.num_episodes):
        seed = args.seed + ep
        obs = env.reset(seed=seed)
        policy.reset()

        ep_obs, ep_actions, ep_xyz, ep_rewards = [], [], [], []
        done = False

        while not done:
            action = policy.act(obs)
            xyz = env.data.qpos[0:3].copy()  # body position before step

            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())
            ep_xyz.append(xyz)

            obs, reward, done = env.step(action)
            ep_rewards.append(reward)

        steps = len(ep_obs)

        if steps < args.min_steps:
            discarded += 1
            if ep % 50 == 0:
                print(
                    f"  Ep {ep+1:>4}/{args.num_episodes}  steps={steps:>4}  DISCARDED"
                )
            continue

        all_obs.extend(ep_obs)
        all_actions.extend(ep_actions)
        all_xyz.extend(ep_xyz)
        all_rewards.extend(ep_rewards)
        all_ep_ids.extend([kept] * steps)
        kept += 1

        if ep % 50 == 0:
            print(
                f"  Ep {ep+1:>4}/{args.num_episodes}  steps={steps:>4}  "
                f"reward={sum(ep_rewards):.1f}  kept={kept}"
            )

    elapsed = time.perf_counter() - t0
    print("-" * 60)
    print(f"Kept {kept} episodes, discarded {discarded}")
    print(f"Total transitions: {len(all_obs)}")
    print(f"Collection time: {elapsed:.1f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        obs=np.array(all_obs, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32),
        xyz=np.array(all_xyz, dtype=np.float32),
        ep_ids=np.array(all_ep_ids, dtype=np.int32),
        rewards=np.array(all_rewards, dtype=np.float32),
    )
    print(f"Saved dataset to {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    meta = {
        "checkpoint": str(args.checkpoint),
        "num_episodes_collected": kept,
        "num_episodes_discarded": discarded,
        "num_transitions": len(all_obs),
        "obs_dim": 35,
        "act_dim": 12,
        "xyz_dim": 3,
        "waypoint_stride": 50,
        "min_steps_filter": args.min_steps,
        "seed": args.seed,
        "action_convention": "delta",
        "obs_convention": "noxy",
        "collection_time_s": round(elapsed, 1),
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
