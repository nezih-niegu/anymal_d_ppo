"""
Basic PPO trainer for ANYmal D — entry-point script.

Uses the simpler variant: MultivariateNormal with externally-annealed std,
raw qpos/qvel observation, and simple velocity+height reward.

Usage:
    python scripts/train_basic.py              # single training run
    python scripts/train_basic.py --sweep      # random-search sweep
    python scripts/train_basic.py --sweep --sweep-count 20 --sweep-episodes 2000
"""

import argparse
import os
import torch

from anymal_d.config import PPOConfig
from anymal_d.envs.basic_env import BasicEnv, N_OBS, N_ACT
from anymal_d.policies.basic_agent import BasicAgent
from anymal_d.training.replay_memory import ReplayMemory, make_basic_dtype
from anymal_d.training.ppo import train_basic
from anymal_d.training.sweep import run_sweep, BASIC_SWEEP_CONFIG
from anymal_d.evaluation.evaluator import evaluate_episode
from anymal_d.checkpoints.manager import save_checkpoint, push_to_hub
import anymal_d.tracking.tracker as tracker
from anymal_d.utils.paths import SAVE_DIR

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "anymal_d", "config", "default_basic.yaml",
)
TARGET_REWARD = 10000


def train_run(
    overrides: dict | None = None,
    run_name: str | None = None,
    num_episodes: int | None = None,
) -> float:
    cfg = PPOConfig.from_yaml(_CONFIG_PATH).with_overrides(overrides)
    if num_episodes is not None:
        cfg.num_episodes = num_episodes

    run = tracker.init_run(cfg.to_dict(), run_name)
    run_name = getattr(run, "name", None) or run_name or "run"

    env = BasicEnv()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    policy    = BasicAgent(N_OBS, N_ACT)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    memory    = ReplayMemory(cfg.replay_size, make_basic_dtype(N_OBS, N_ACT))
    hparams   = cfg.to_dict()

    std_init = cfg.std_init or 1.0
    std_min  = cfg.std_min  or 0.6
    # Linear decay applied every log_interval episodes
    std_decay = -(std_min - std_init) * cfg.log_interval / cfg.num_episodes
    action_std = std_init

    running_reward = -100.0
    saving_reward  = 0.0
    print(f"Target reward: {TARGET_REWARD}")

    try:
        for i_episode in range(cfg.num_episodes):
            state, ep_reward, done = env.reset(), 0.0, False

            while not done:
                action, a_logp, _ = policy.compute_action(state, action_std)
                next_state, reward, done = env.step(action, render=False)

                if memory.store((state, action, a_logp, reward, next_state)):
                    pl, vl, ent, ratio = train_basic(
                        policy, optimizer, memory, hparams, action_std
                    )
                    tracker.log({
                        "policy_loss": pl,
                        "value_loss":  vl,
                        "avg_reward":  running_reward,
                        "avg_entropy": ent,
                        "ratio":       ratio,
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
                    f"Avg: {running_reward:.2f}\t"
                    f"Std: {action_std:.5f}"
                )
                action_std = round(action_std - std_decay, 5)

            if running_reward > 2000 and running_reward > saving_reward:
                saving_reward = running_reward
                p_path, o_path = save_checkpoint(
                    policy, optimizer, SAVE_DIR, run_name, i_episode, running_reward
                )
                push_to_hub(p_path)
                push_to_hub(o_path)
                print(f"Saved checkpoint → {SAVE_DIR}")
                evaluate_episode(
                    env, policy, i_episode,
                    action_std=action_std, log_media=True, prefix="checkpoint"
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
        tracker.finish()

    return running_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic ANYmal D PPO trainer")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-count",    type=int, default=50)
    parser.add_argument("--sweep-episodes", type=int, default=2000)
    args = parser.parse_args()

    try:
        if args.sweep:
            run_sweep(
                train_run, BASIC_SWEEP_CONFIG,
                count=args.sweep_count,
                episodes_per_trial=args.sweep_episodes,
            )
        else:
            train_run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
