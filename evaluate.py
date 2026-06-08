#!/usr/bin/env python3
"""
evaluate.py
===========
Shared policy evaluator for the ANYmal D PPO project.

Usage examples
--------------
# Single policy, default config:
python evaluate.py --policy ppo_multivariate --checkpoint pretrained_models/anymal_d/run_1000_Reward-2500.0_policy.pt

# Override episodes and seed:
python evaluate.py --policy ppo_normal --checkpoint path/to/policy.pt --num-episodes 20 --seed 42

# Use a custom YAML config:
python evaluate.py --policy ppo_multivariate --checkpoint path/to/policy.pt --config my_eval.yaml

# CSV output:
python evaluate.py --policy ppo_normal --checkpoint path/to/policy.pt --output-format csv

# Save videos:
python evaluate.py --policy ppo_normal --checkpoint path/to/policy.pt --save-videos

Supported policy types
----------------------
  ppo_multivariate   Trainer 1 — MultivariateNormal, raw-action, 37-D obs
  ppo_normal         Trainer 2 — Normal + learnable log_std, delta actions, 35-D obs
  diffusion          (stub — raises NotImplementedError until implemented)
  flow_matching      (stub — raises NotImplementedError until implemented)
  scripted           (stub — raises NotImplementedError until implemented)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Import project modules.  evaluate.py lives at the repo root, so Python can
# find ``policies`` and ``envs`` without any sys.path manipulation as long as
# the script is run from the repo root (or the repo root is on PYTHONPATH).
# ---------------------------------------------------------------------------

try:
    from policies import POLICY_REGISTRY, BasePolicy
    from envs.anymal_env import ANYmalEnv
except ImportError as exc:
    sys.exit(
        f"Import error: {exc}\n"
        "Make sure you run evaluate.py from the repo root:\n"
        "  cd anymal_d_ppo && python evaluate.py ..."
    )

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path(__file__).parent / "eval_config.yaml"


def load_config(path: Path) -> dict:
    """Load a YAML config file.  Returns an empty dict if PyYAML is absent."""
    try:
        import yaml
    except ImportError:
        print(
            "[warn] PyYAML not installed; using built-in defaults. "
            "Install with: pip install pyyaml"
        )
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def merge_config(cfg: dict, args: argparse.Namespace) -> dict:
    """CLI args win over YAML config.  Returns a flat merged dict."""
    merged = {}
    # Env defaults from yaml
    env_cfg = cfg.get("env", {})
    merged["model_xml"] = env_cfg.get("model_xml") or None
    merged["fall_threshold"] = env_cfg.get("fall_threshold", 0.35)
    merged["reward_fn"] = env_cfg.get("reward_fn", "shaped")

    # Output defaults
    out_cfg = cfg.get("output", {})
    merged["output_format"] = out_cfg.get("format", "json")
    merged["output_dir"] = out_cfg.get("dir", "results")
    merged["output_filename"] = out_cfg.get("filename") or None
    merged["save_videos"] = out_cfg.get("save_videos", False)
    merged["video_dir"] = out_cfg.get("video_dir", "results/videos")

    # Rollout defaults
    merged["num_episodes"] = cfg.get("num_episodes", 10)
    seeds_cfg = cfg.get("seeds", 0)
    if isinstance(seeds_cfg, list):
        merged["seeds"] = seeds_cfg
    else:
        merged["seeds"] = [int(seeds_cfg)] * merged["num_episodes"]

    # CLI overrides (only if explicitly provided)
    if args.num_episodes is not None:
        merged["num_episodes"] = args.num_episodes
        # Re-generate seeds list if length changed
        if args.seed is not None:
            merged["seeds"] = [args.seed + i for i in range(merged["num_episodes"])]
        else:
            merged["seeds"] = list(range(merged["num_episodes"]))
    elif args.seed is not None:
        merged["seeds"] = [args.seed + i for i in range(merged["num_episodes"])]

    if args.output_format:
        merged["output_format"] = args.output_format
    if args.output_dir:
        merged["output_dir"] = args.output_dir
    if args.save_videos:
        merged["save_videos"] = True

    # Per-policy settings
    policy_defaults = cfg.get("policies", {}).get(args.policy, {})
    merged["obs_convention"] = policy_defaults.get("obs_convention", "noxy")
    merged["action_convention"] = policy_defaults.get("action_convention", "delta")
    merged["deterministic"] = policy_defaults.get("deterministic", True)

    # Enforce trainer-1 conventions
    if args.policy == "ppo_multivariate":
        merged["obs_convention"] = "full"
        merged["action_convention"] = "raw"

    return merged


# ---------------------------------------------------------------------------
# Per-episode rollout
# ---------------------------------------------------------------------------


def run_episode(
    env: ANYmalEnv,
    policy: BasePolicy,
    seed: int,
    render: bool = False,
) -> dict[str, Any]:
    """Run one episode and return a dict of metrics."""
    policy.reset()
    obs = env.reset(seed=seed)

    ep_reward = 0.0
    ep_length = 0
    forward_vels: list[float] = []
    fell = False

    # Waypoint tracking (for flow_matching policy)
    is_waypoint_policy = hasattr(policy, "update_xyz")
    waypoint_stride = 50
    xyz_history = []
    waypoints_total = 0
    waypoints_reached = 0
    WP_THRESHOLD = 0.3  # metres — waypoint considered reached

    while True:
        # Update xyz buffer for waypoint-conditioned policies
        if is_waypoint_policy:
            xyz = env.data.qpos[0:3].copy()
            policy.update_xyz(xyz)
            xyz_history.append(xyz.copy())

        action = policy.act(obs)
        obs, reward, done = env.step(action, render=render)

        ep_reward += reward
        ep_length += 1
        forward_vels.append(float(env.data.qvel[0]))

        # Check waypoint completion every STRIDE steps
        if is_waypoint_policy and ep_length % waypoint_stride == 0:
            current_xyz = env.data.qpos[0:3].copy()
            target_idx = max(0, len(xyz_history) - waypoint_stride)
            target_xyz = xyz_history[target_idx]
            dist = float(np.linalg.norm(current_xyz[:2] - target_xyz[:2]))
            waypoints_total += 1
            if dist < WP_THRESHOLD or current_xyz[0] > target_xyz[0]:
                waypoints_reached += 1

        height = float(env.data.qpos[2])
        if height < env.fall_threshold:
            fell = True
        if done:
            break

    wp_completion = waypoints_reached / waypoints_total if waypoints_total > 0 else None

    return {
        "seed": seed,
        "episode_reward": round(ep_reward, 4),
        "episode_length": ep_length,
        "forward_velocity_mean": round(float(np.mean(forward_vels)), 4),
        "forward_velocity_max": round(float(np.max(forward_vels)), 4),
        "fell": fell,
        "waypoint_completion": (
            round(wp_completion, 4) if wp_completion is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def make_output_path(
    output_dir: str, policy_name: str, fmt: str, filename: str | None
) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if filename:
        return Path(output_dir) / filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"eval_{policy_name}_{ts}.{fmt}"


def write_json(path: Path, results: dict) -> None:
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results written to {path}")


def write_csv(path: Path, results: dict) -> None:
    episodes = results["episodes"]
    if not episodes:
        print("[warn] No episodes to write.")
        return
    fieldnames = list(results["summary"].keys()) + list(episodes[0].keys())
    # Write summary row then episode rows
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(episodes[0].keys()))
        writer.writeheader()
        for ep in episodes:
            writer.writerow(ep)
    # Append summary as a second CSV file
    summary_path = path.with_suffix(".summary.csv")
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results["summary"].keys()))
        writer.writeheader()
        writer.writerow(results["summary"])
    print(f"Episode results  → {path}")
    print(f"Summary          → {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an ANYmal D policy using a shared environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--policy",
        required=True,
        choices=list(POLICY_REGISTRY.keys()),
        help="Policy architecture to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the policy checkpoint file.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to roll out (overrides config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed; episode i gets seed+i (overrides config).",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "csv"],
        default=None,
        help="Output format (overrides config).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for result files (overrides config).",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        default=False,
        help="Render and save one mp4 per episode.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override deterministic/stochastic inference mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load config ---------------------------------------------------------
    config_path = Path(args.config)
    cfg = load_config(config_path) if config_path.exists() else {}
    c = merge_config(cfg, args)

    # --- Build policy --------------------------------------------------------
    policy_cls = POLICY_REGISTRY[args.policy]
    # Pass deterministic kwarg only to policies that accept it
    try:
        det = (
            args.deterministic if args.deterministic is not None else c["deterministic"]
        )
        policy: BasePolicy = policy_cls(deterministic=det)
    except TypeError:
        policy = policy_cls()

    print(f"Loading {args.policy} checkpoint: {args.checkpoint}")
    policy.load(args.checkpoint)
    policy.set_eval_mode()

    # --- Build environment ---------------------------------------------------
    env_kwargs: dict[str, Any] = {
        "action_convention": c["action_convention"],
        "obs_convention": c["obs_convention"],
        "reward_fn": c["reward_fn"],
        "fall_threshold": c["fall_threshold"],
    }
    if c["model_xml"]:
        env_kwargs["model_xml"] = c["model_xml"]

    print(
        f"Building environment  (action={c['action_convention']}, "
        f"obs={c['obs_convention']}, reward={c['reward_fn']}, "
        f"fall_threshold={c['fall_threshold']})"
    )
    env = ANYmalEnv(**env_kwargs)

    # Sanity-check obs dimensions
    if policy.obs_dim > 0 and policy.obs_dim != env.obs_dim:
        print(
            f"[warn] Policy expects {policy.obs_dim}-D obs but env produces "
            f"{env.obs_dim}-D obs.  This may cause errors."
        )

    # --- Roll out ------------------------------------------------------------
    num_episodes = c["num_episodes"]
    seeds = c["seeds"]
    if len(seeds) < num_episodes:
        # Pad seeds if the list is shorter than num_episodes
        seeds = seeds + list(range(len(seeds), num_episodes))
    seeds = seeds[:num_episodes]

    save_videos = c["save_videos"]
    if save_videos:
        video_dir = Path(c["video_dir"])
        video_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {num_episodes} episodes with policy '{policy.name}' ...")
    print(f"Seeds: {seeds}")
    print("-" * 60)

    episode_results: list[dict] = []
    t0 = time.perf_counter()

    for ep_idx in range(num_episodes):
        seed = seeds[ep_idx]
        ep = run_episode(env, policy, seed=seed, render=save_videos)
        ep["episode"] = ep_idx + 1

        if save_videos and env.frames:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            vpath = video_dir / f"{args.policy}_ep{ep_idx+1}_seed{seed}_{ts}.mp4"
            try:
                env.save_video(vpath)
                ep["video_path"] = str(vpath)
                print(f"  Saved video → {vpath}")
            except Exception as exc:
                print(f"  [warn] Could not save video: {exc}")

        episode_results.append(ep)
        print(
            f"  Ep {ep_idx+1:>3}/{num_episodes}  "
            f"reward={ep['episode_reward']:>8.2f}  "
            f"steps={ep['episode_length']:>4}  "
            f"fwd_vel_mean={ep['forward_velocity_mean']:>5.3f}  "
            f"fell={'yes' if ep['fell'] else 'no '}"
        )

    elapsed = time.perf_counter() - t0

    # --- Summary statistics --------------------------------------------------
    rewards = [e["episode_reward"] for e in episode_results]
    lengths = [e["episode_length"] for e in episode_results]
    fwd_vels = [e["forward_velocity_mean"] for e in episode_results]
    fell_flags = [e["fell"] for e in episode_results]

    summary = {
        "policy": args.policy,
        "checkpoint": str(args.checkpoint),
        "num_episodes": num_episodes,
        "seeds": seeds,
        "reward_mean": round(float(np.mean(rewards)), 4),
        "reward_std": round(float(np.std(rewards)), 4),
        "reward_min": round(float(np.min(rewards)), 4),
        "reward_max": round(float(np.max(rewards)), 4),
        "episode_length_mean": round(float(np.mean(lengths)), 2),
        "forward_velocity_mean": round(float(np.mean(fwd_vels)), 4),
        "fall_rate": round(float(np.mean(fell_flags)), 4),
        "eval_wall_time_s": round(elapsed, 2),
        "env": {
            "action_convention": c["action_convention"],
            "obs_convention": c["obs_convention"],
            "reward_fn": c["reward_fn"],
            "fall_threshold": c["fall_threshold"],
        },
    }

    print("-" * 60)
    print(f"Summary:")
    print(
        f"  reward:   {summary['reward_mean']:.2f} ± {summary['reward_std']:.2f}  "
        f"(min {summary['reward_min']:.2f}, max {summary['reward_max']:.2f})"
    )
    print(f"  fwd vel:  {summary['forward_velocity_mean']:.3f} m/s (mean)")
    print(f"  fall rate:{summary['fall_rate']:.1%}")
    print(f"  wall time:{elapsed:.1f}s")

    # --- Write output --------------------------------------------------------
    output_path = make_output_path(
        c["output_dir"], args.policy, c["output_format"], c["output_filename"]
    )
    full_results = {"summary": summary, "episodes": episode_results}

    if c["output_format"] == "json":
        write_json(output_path, full_results)
    else:
        write_csv(output_path, full_results)


if __name__ == "__main__":
    main()
