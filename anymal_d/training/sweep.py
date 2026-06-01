"""Random-search hyperparameter sweep driver (replaces wandb.sweep + wandb.agent)."""

from __future__ import annotations
from typing import Callable
import numpy as np


BASIC_SWEEP_CONFIG: dict = {
    "name": "ppo_basic_sweep",
    "parameters": {
        "lr":          {"distribution": "uniform",     "min": 1e-5,  "max": 1e-4},
        "ppo_epoch":   {"distribution": "int_uniform", "min": 40,    "max": 60},
        "c2":          {"distribution": "uniform",     "min": 0.001, "max": 0.01},
        "replay_size": {"distribution": "int_uniform", "min": 6000,  "max": 10000},
        "std_init":    {"distribution": "uniform",     "min": 1.0,   "max": 1.1},
        "std_min":     {"distribution": "uniform",     "min": 0.7,   "max": 0.8},
    },
}

ADVANCED_SWEEP_CONFIG: dict = {
    "name": "ppo_advanced_sweep",
    "parameters": {
        "lr":          {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        "ppo_epoch":   {"distribution": "int_uniform",        "min": 5,    "max": 20},
        "clip_param":  {"distribution": "uniform",            "min": 0.1,  "max": 0.3},
        "c2":          {"distribution": "uniform",            "min": 0.0,  "max": 0.02},
        "replay_size": {"distribution": "int_uniform",        "min": 2048, "max": 8192},
    },
}


def sample_param(spec: dict, rng: np.random.Generator) -> float | int:
    dist = spec.get("distribution", "uniform")
    lo, hi = spec["min"], spec["max"]
    if dist == "int_uniform":
        return int(rng.integers(int(lo), int(hi) + 1))
    if dist in ("log_uniform_values", "log_uniform"):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


def run_sweep(
    train_fn: Callable,
    sweep_config: dict,
    count: int = 50,
    episodes_per_trial: int = 2000,
    seed: int = 0,
    **train_kwargs,
) -> dict:
    """Run *count* random-search trials, each calling *train_fn*.

    *train_fn* must accept keyword arguments ``overrides``, ``run_name``,
    and ``num_episodes`` (plus any extras in *train_kwargs*).
    Returns the best-trial dict.
    """
    rng = np.random.default_rng(seed)
    best: dict = {"reward": float("-inf"), "name": None, "params": None}
    print(f"Starting sweep: {count} trials × {episodes_per_trial} episodes each")

    for t in range(count):
        sampled = {k: sample_param(v, rng) for k, v in sweep_config["parameters"].items()}
        name = f"{sweep_config['name']}_trial{t:03d}"
        print(f"\n=== Trial {t + 1}/{count} :: {name} :: {sampled} ===")
        reward = train_fn(
            overrides=sampled,
            run_name=name,
            num_episodes=episodes_per_trial,
            **train_kwargs,
        )
        if reward > best["reward"]:
            best = {"reward": reward, "name": name, "params": sampled}
        print(f"Trial {name}: reward={reward}  |  best={best['reward']} ({best['name']})")

    print(f"\nSweep done. Best: {best['name']}  reward={best['reward']}\n  params={best['params']}")
    return best
