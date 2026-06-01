"""Unified episode evaluation and video/plot logging."""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anymal_d.utils.paths import VIDEO_DIR
import anymal_d.tracking.tracker as tracker

MUJOCO_STEPS: int = 5


def evaluate_episode(
    env,
    policy,
    episode: int,
    action_std: float | None = None,
    render: bool = True,
    log_media: bool = True,
    prefix: str = "eval",
    video_dir: str | None = None,
) -> tuple[float, str]:
    """Run one evaluation episode; return (total_reward, video_path).

    Compatible with both BasicAgent (pass *action_std*) and AdvancedAgent
    (*action_std* omitted — policy handles exploration internally).
    """
    state, ep_reward, done = env.reset(), 0.0, False
    counter = 0
    reward_list: list[float] = []
    cumulative_list: list[float] = []
    time_list: list[float] = []

    while not done:
        if action_std is not None:
            action, _, _ = policy.compute_action(state, action_std)
        else:
            action, _, _ = policy.compute_action(state)
        state, reward, done = env.step(action, render=render)
        reward_list.append(reward)
        time_list.append(counter * 0.002 * MUJOCO_STEPS)
        ep_reward += reward
        cumulative_list.append(ep_reward)
        counter += 1

    out_dir = video_dir or VIDEO_DIR
    os.makedirs(out_dir, exist_ok=True)
    video_path = env.close(episode, ep_reward, prefix=prefix, video_dir=out_dir)
    print(f"  wrote {video_path}")

    if log_media and tracker.is_active():
        log_key = "Video train" if prefix.startswith(("train", "checkpoint")) else "Video eval"
        tracker.log_video(video_path, key=log_key)

    plot_path = os.path.join(out_dir, f"{prefix}_reward_{episode}.png")
    plt.figure()
    plt.plot(time_list, reward_list, label="instant")
    plt.plot(time_list, cumulative_list, label="cumulative")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Reward")
    plt.title(f"Episode {episode} — {prefix}")
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

    if log_media and tracker.is_active():
        plot_key = "Reward train" if prefix.startswith(("train", "checkpoint")) else "Reward eval"
        tracker.log_image(plot_path, key=plot_key)

    return ep_reward, video_path
