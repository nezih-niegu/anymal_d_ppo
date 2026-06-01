#!/usr/bin/env python3
"""
eval_smoothness.py
==================
Measures control smoothness for PPO and Flow Matching policies.

Smoothness metric: mean L2 norm of consecutive action deltas per episode.
    smoothness = mean(||a_t - a_{t-1}||_2)   over all steps t

Lower = smoother, more natural movement.

Also computes:
    - energy proxy: mean(||a_t||_2^2) — proxy for total joint effort
    - jerk proxy:   mean(||a_t - 2*a_{t-1} + a_{t-2}||_2) — rate of change of acceleration

Usage
-----
python eval_smoothness.py \
    --policy ppo_normal \
    --checkpoint pretrained_models/anymal_d/brave-forest-1_5931_Reward-1805.2_policy.pt \
    --policy2 flow_matching \
    --checkpoint2 pretrained_models/flow_matching/best_flow_policy.pt \
    --num-episodes 10 \
    --out results/smoothness/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "anymal_d"))

import RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING as _trainer
import __main__
__main__.Agent = _trainer.Agent

from policies import POLICY_REGISTRY
from envs.anymal_env import ANYmalEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--policy2",     required=True)
    parser.add_argument("--checkpoint2", required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--out",          default="results/smoothness")
    return parser.parse_args()


def run_smoothness_episode(env, policy, seed):
    """Run one episode and return per-step actions."""
    policy.reset()
    obs = env.reset(seed=seed)
    actions = []
    done = False

    while not done:
        action = policy.act(obs)
        actions.append(action.copy())
        obs, _, done = env.step(action)

    actions = np.array(actions)  # (T, 12)

    # Action deltas
    deltas = np.diff(actions, axis=0)           # (T-1, 12)
    smoothness  = float(np.mean(np.linalg.norm(deltas, axis=1)))
    energy      = float(np.mean(np.sum(actions**2, axis=1)))

    # Jerk (second difference)
    if len(actions) > 2:
        jerk = np.diff(actions, n=2, axis=0)   # (T-2, 12)
        jerk_val = float(np.mean(np.linalg.norm(jerk, axis=1)))
    else:
        jerk_val = 0.0

    return {
        "smoothness":  smoothness,
        "energy":      energy,
        "jerk":        jerk_val,
        "ep_length":   len(actions),
        "action_mean": float(np.mean(np.abs(actions))),
        "action_std":  float(np.std(actions)),
    }


def evaluate_policy(name, checkpoint, num_episodes, seed):
    """Evaluate smoothness metrics for one policy."""
    policy_cls = POLICY_REGISTRY[name]
    try:
        policy = policy_cls(deterministic=False)
    except TypeError:
        policy = policy_cls()

    policy.load(checkpoint)
    policy.set_eval_mode()

    # Pick env convention based on policy
    if name == "ppo_multivariate":
        env = ANYmalEnv(action_convention="raw",   obs_convention="full")
    else:
        env = ANYmalEnv(action_convention="delta", obs_convention="noxy")

    results = []
    for ep in range(num_episodes):
        r = run_smoothness_episode(env, policy, seed=seed + ep)
        results.append(r)
        print(f"  [{name}] Ep {ep+1:>2}/{num_episodes}  "
              f"smoothness={r['smoothness']:.4f}  "
              f"energy={r['energy']:.4f}  "
              f"jerk={r['jerk']:.4f}")

    summary = {
        "policy":           name,
        "checkpoint":       checkpoint,
        "smoothness_mean":  round(float(np.mean([r["smoothness"] for r in results])), 4),
        "smoothness_std":   round(float(np.std ([r["smoothness"] for r in results])), 4),
        "energy_mean":      round(float(np.mean([r["energy"]     for r in results])), 4),
        "energy_std":       round(float(np.std ([r["energy"]     for r in results])), 4),
        "jerk_mean":        round(float(np.mean([r["jerk"]       for r in results])), 4),
        "jerk_std":         round(float(np.std ([r["jerk"]       for r in results])), 4),
        "episodes":         results,
    }
    return summary


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating {args.policy}...")
    s1 = evaluate_policy(args.policy,  args.checkpoint,  args.num_episodes, args.seed)

    print(f"\nEvaluating {args.policy2}...")
    s2 = evaluate_policy(args.policy2, args.checkpoint2, args.num_episodes, args.seed)

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print("\n" + "="*58)
    print(f"{'Metric':<25} {args.policy:>15} {args.policy2:>15}")
    print("="*58)
    for key in ["smoothness_mean", "energy_mean", "jerk_mean"]:
        label = key.replace("_mean", "").capitalize()
        print(f"{label:<25} {s1[key]:>15.4f} {s2[key]:>15.4f}")
    print("="*58)

    # Verdict
    if s1["smoothness_mean"] < s2["smoothness_mean"]:
        print(f"→ {args.policy} is SMOOTHER")
    else:
        print(f"→ {args.policy2} is SMOOTHER")

    # Save JSON
    result = {"policy1": s1, "policy2": s2}
    json_path = out_dir / "smoothness_results.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ------------------------------------------------------------------
    # Plot smoothness per episode
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Control Smoothness — PPO vs Flow Matching", fontsize=13, fontweight="bold")

    metrics_to_plot = [
        ("smoothness", "Action Delta (lower=smoother)"),
        ("energy",     "Energy Proxy (lower=less effort)"),
        ("jerk",       "Jerk Proxy (lower=smoother accel)"),
    ]

    for ax, (key, ylabel) in zip(axes, metrics_to_plot):
        v1 = [r[key] for r in s1["episodes"]]
        v2 = [r[key] for r in s2["episodes"]]
        eps = range(1, len(v1) + 1)

        ax.plot(eps, v1, "o-", color="#4C72B0", label=args.policy,  linewidth=2)
        ax.plot(eps, v2, "s-", color="#DD8452", label=args.policy2, linewidth=2)
        ax.axhline(np.mean(v1), color="#4C72B0", linestyle="--", linewidth=1)
        ax.axhline(np.mean(v2), color="#DD8452", linestyle="--", linewidth=1)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split("(")[0].strip())
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "smoothness_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
