from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class PPOConfig:
    # Discount & scheduling
    gamma: float = 0.99
    log_interval: int = 50
    num_episodes: int = 15000
    # Optimizer
    lr: float = 3e-4
    # PPO clipping & coefficients
    clip_param: float = 0.2
    ppo_epoch: int = 10
    replay_size: int = 4096
    batch_size: int = 128
    c1: float = 0.5          # value-loss coefficient
    c2: float = 0.005         # entropy bonus coefficient
    max_grad_norm: float = 0.5
    # Environment
    fall_threshold: float = 0.35
    # Basic-trainer std annealing (None → not used)
    std_init: Optional[float] = None
    std_min: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PPOConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def from_yaml(cls, path: str) -> "PPOConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def with_overrides(self, overrides: dict | None) -> "PPOConfig":
        if not overrides:
            return self
        d = self.to_dict()
        d.update(overrides)
        return self.from_dict(d)
