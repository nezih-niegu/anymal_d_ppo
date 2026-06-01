"""
Advanced PPO trainer for ANYmal D — entry-point script.

Uses the polished variant: learnable log_std, delta-around-nominal actions,
shaped reward, and offscreen video logging.

Usage:
    python scripts/train_advanced.py train
    python scripts/train_advanced.py train --live
    python scripts/train_advanced.py sweep --sweep-count 30
    python scripts/train_advanced.py render --num-videos 5
"""

import argparse
import os
import torch

from anymal_d.config import PPOConfig
from anymal_d.envs.advanced_env import AdvancedEnv, N_OBS, N_ACT
from anymal_d.policies.advanced_agent import AdvancedAgent
from anymal_d.training.replay_memory import ReplayMemory, make_advanced_dtype
from anymal_d.training.ppo import train_advanced
from anymal_d.training.sweep import run_sweep, ADVANCED_SWEEP_CONFIG
from anymal_d.evaluation.evaluator import evaluate_episode
from anymal_d.checkpoints.manager import (
    save_checkpoint, push_to_hub, pick_best_checkpoint, load_checkpoint
)
import anymal_d.tracking.tracker as tracker
from anymal_d.utils.paths import SAVE_DIR

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "anymal_d", "config", "default_advanced.yaml",
)
TARGET_REWARD = 1800


def train_run(
    overrides: dict | None = None,
    run_name: str | None = None,
    num_episodes: int | None = None,
    live: bool = False,
) -> float:
    cfg = PPOConfig.from_yaml(_CONFIG_PATH).with_overrides(overrides)
    if num_episodes is not None:
        cfg.num_episodes = num_episodes

    run = tracker.init_run(cfg.to_dict(), run_name)
    run_name = getattr(run, "name", None) or run_name or "run"

    env = AdvancedEnv(fall_threshold=cfg.fall_threshold)
    if live:
        try:
            env.attach_viewer()
        except Exception as exc:
            print(f"Could not open viewer ({exc}). Continuing headless.")

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    policy    = AdvancedAgent(N_OBS, N_ACT)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    memory    = ReplayMemory(cfg.replay_size, make_advanced_dtype(N_OBS, N_ACT))
    hparams   = cfg.to_dict()

    running_reward = 0.0
    saving_reward  = 100.0
    print(f"Target reward: {TARGET_REWARD}")

    try:
        for i_episode in range(cfg.num_episodes):
            state, ep_reward, done = env.reset(), 0.0, False

            while not done:
                action, a_logp, _ = policy.compute_action(state)
                next_state, reward, done = env.step(action, render=False)

                if memory.store((state, action, a_logp, reward, next_state, float(done))):
                    pl, vl, ent, ratio = train_advanced(policy, optimizer, memory, hparams)
                    tracker.log({
                        "policy_loss": pl,
                        "value_loss":  vl,
                        "avg_reward":  running_reward,
                        "avg_entropy": ent,
                        "ratio":       ratio,
                        "action_std":  float(policy.log_std.detach().exp().mean()),
                    })

                state = next_state
                ep_reward += reward
                if done:
                    break

            running_reward = round(0.05 * ep_reward + 0.95 * running_reward, 2)

            if i_episode % cfg.log_interval == 0:
                print(
                    f"Episode {i_episode}\t"
                    f"Last: {ep_reward:.2f}\t"
                    f"Avg: {running_reward:.2f}"
                )

            if running_reward > saving_reward:
                saving_reward = running_reward
                p_path, o_path = save_checkpoint(
                    policy, optimizer, SAVE_DIR, run_name, i_episode, running_reward
                )
                push_to_hub(p_path)
                push_to_hub(o_path)
                print(f"Saved checkpoint → {SAVE_DIR}")
                evaluate_episode(
                    env, policy, i_episode, log_media=True, prefix="checkpoint"
                )

            if running_reward > TARGET_REWARD:
                print("Solved!")
                p_path, o_path = save_checkpoint(
                    policy, optimizer, SAVE_DIR, run_name, i_episode, running_reward
                )
                push_to_hub(p_path)
                push_to_hub(o_path)
                break

        print(f"Done. Running reward: {running_reward}")
    finally:
        env.detach_viewer()
        tracker.finish()

    return running_reward


def render_run(
    num_videos: int = 5,
    policy_path: str | None = None,
    fall_threshold: float = 0.3,
) -> None:
    if policy_path is None:
        policy_path = pick_best_checkpoint(SAVE_DIR)
    print(f"Loading: {policy_path}")
    policy = load_checkpoint(policy_path)
    env = AdvancedEnv(fall_threshold=fall_threshold)
    torch.manual_seed(0)
    for i in range(num_videos):
        ep_reward, _ = evaluate_episode(
            env, policy, i + 1, log_media=False, prefix="eval"
        )
        print(f"Video #{i + 1}  reward: {ep_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced ANYmal D PPO trainer")
    parser.add_argument(
        "mode", nargs="?", default="train",
        choices=["train", "sweep", "render"],
    )
    parser.add_argument("--live", action="store_true",
                        help="open a live MuJoCo viewer (requires a display)")
    parser.add_argument("--sweep-count",    type=int,   default=50)
    parser.add_argument("--sweep-episodes", type=int,   default=2000)
    parser.add_argument("--policy",         type=str,   default=None)
    parser.add_argument("--num-videos",     type=int,   default=5)
    parser.add_argument("--fall-threshold", type=float, default=0.3)
    args = parser.parse_args()

    try:
        if args.mode == "train":
            train_run(live=args.live)
        elif args.mode == "sweep":
            run_sweep(
                train_run, ADVANCED_SWEEP_CONFIG,
                count=args.sweep_count,
                episodes_per_trial=args.sweep_episodes,
                live=args.live,
            )
        elif args.mode == "render":
            render_run(
                num_videos=args.num_videos,
                policy_path=args.policy,
                fall_threshold=args.fall_threshold,
            )
    except KeyboardInterrupt:
        print("\nInterrupted.")
