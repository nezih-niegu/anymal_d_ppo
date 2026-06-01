#!/usr/bin/env python3
"""
compare_results.py
==================
Generates comparison plots and a summary table from evaluate.py JSON results.

Usage
-----
python compare_results.py \
    --ppo     results/eval_ppo_normal_20260531_055730.json \
    --flow    results/eval_flow_matching_20260531_091500.json \
    --out     results/comparison/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def smoothness(episodes: list[dict]) -> float:
    """Proxy: reward std across episodes (lower = more consistent)."""
    rewards = [e["episode_reward"] for e in episodes]
    return round(float(np.std(rewards)), 4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo",  required=True, help="PPO eval JSON")
    parser.add_argument("--flow", required=True, help="Flow Matching eval JSON")
    parser.add_argument("--out",  default="results/comparison")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ppo  = load(args.ppo)
    flow = load(args.flow)

    ppo_eps  = ppo["episodes"]
    flow_eps = flow["episodes"]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics = {
        "Policy":              ["PPO",                       "Flow Matching"],
        "Reward mean":         [ppo["summary"]["reward_mean"],  flow["summary"]["reward_mean"]],
        "Reward std":          [ppo["summary"]["reward_std"],   flow["summary"]["reward_std"]],
        "Fwd velocity (m/s)":  [ppo["summary"]["forward_velocity_mean"], flow["summary"]["forward_velocity_mean"]],
        "Fall rate":           [f"{ppo['summary']['fall_rate']*100:.0f}%",  f"{flow['summary']['fall_rate']*100:.0f}%"],
        "Consistency (std)":   [smoothness(ppo_eps),         smoothness(flow_eps)],
        "Ep length mean":      [ppo["summary"]["episode_length_mean"], flow["summary"]["episode_length_mean"]],
    }

    # Print table
    print("\n" + "="*52)
    print(f"{'Metric':<25} {'PPO':>12} {'Flow Matching':>12}")
    print("="*52)
    for k, v in metrics.items():
        if k == "Policy":
            continue
        print(f"{k:<25} {str(v[0]):>12} {str(v[1]):>12}")
    print("="*52)

    # Save table as JSON
    table_path = out_dir / "comparison_table.json"
    with open(table_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nTable saved to {table_path}")

    # ------------------------------------------------------------------
    # Plot 1: Episode rewards bar chart
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("PPO vs Flow Matching — Policy Comparison", fontsize=14, fontweight="bold")

    ppo_rewards  = [e["episode_reward"] for e in ppo_eps]
    flow_rewards = [e["episode_reward"] for e in flow_eps]
    ep_ids = list(range(1, max(len(ppo_rewards), len(flow_rewards)) + 1))

    ax = axes[0]
    x = np.arange(len(ppo_rewards))
    w = 0.35
    ax.bar(x - w/2, ppo_rewards,  w, label="PPO",           color="#4C72B0", alpha=0.85)
    ax.bar(x + w/2, flow_rewards[:len(ppo_rewards)], w, label="Flow Matching", color="#DD8452", alpha=0.85)
    ax.axhline(np.mean(ppo_rewards),  color="#4C72B0", linestyle="--", linewidth=1.5, label=f"PPO mean={np.mean(ppo_rewards):.0f}")
    ax.axhline(np.mean(flow_rewards), color="#DD8452", linestyle="--", linewidth=1.5, label=f"FM mean={np.mean(flow_rewards):.0f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Episode Reward")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ------------------------------------------------------------------
    # Plot 2: Forward velocity per episode
    # ------------------------------------------------------------------
    ppo_vels  = [e["forward_velocity_mean"] for e in ppo_eps]
    flow_vels = [e["forward_velocity_mean"] for e in flow_eps]

    ax = axes[1]
    ax.plot(range(1, len(ppo_vels)+1),  ppo_vels,  "o-", color="#4C72B0", label="PPO",           linewidth=2)
    ax.plot(range(1, len(flow_vels)+1), flow_vels, "s-", color="#DD8452", label="Flow Matching", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Forward Velocity (m/s)")
    ax.set_title("Forward Velocity")
    ax.legend()
    ax.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # Plot 3: Summary bar chart (reward mean + fall rate)
    # ------------------------------------------------------------------
    ax = axes[2]
    categories = ["Reward Mean", "Fall Rate (×100)", "Fwd Vel (×100)"]
    ppo_vals   = [
        ppo["summary"]["reward_mean"],
        ppo["summary"]["fall_rate"] * 100,
        ppo["summary"]["forward_velocity_mean"] * 100,
    ]
    flow_vals  = [
        flow["summary"]["reward_mean"],
        flow["summary"]["fall_rate"] * 100,
        flow["summary"]["forward_velocity_mean"] * 100,
    ]

    x = np.arange(len(categories))
    ax.bar(x - 0.2, ppo_vals,  0.4, label="PPO",           color="#4C72B0", alpha=0.85)
    ax.bar(x + 0.2, flow_vals, 0.4, label="Flow Matching", color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title("Summary Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "comparison_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")

    # ------------------------------------------------------------------
    # Plot 4: Episode length (survival)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ppo_lens  = [e["episode_length"] for e in ppo_eps]
    flow_lens = [e["episode_length"] for e in flow_eps]

    ax.plot(range(1, len(ppo_lens)+1),  ppo_lens,  "o-", color="#4C72B0", label="PPO",           linewidth=2)
    ax.plot(range(1, len(flow_lens)+1), flow_lens, "s-", color="#DD8452", label="Flow Matching", linewidth=2)
    ax.axhline(800, color="gray", linestyle="--", linewidth=1, label="Max steps (800)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length (steps)")
    ax.set_title("Episode Survival — PPO vs Flow Matching")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    survival_path = out_dir / "survival_plot.png"
    plt.savefig(survival_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Survival plot saved to {survival_path}")


if __name__ == "__main__":
    main()
