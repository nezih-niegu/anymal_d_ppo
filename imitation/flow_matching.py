"""
imitation/flow_matching.py
==========================
Waypoint-conditioned Flow Matching policy for ANYmal D.

The network receives: [x_t (12), t_emb (32), obs (35), waypoint (3)]
and predicts the flow velocity toward the expert action.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import numpy as np


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VelocityNet(nn.Module):
    """Predicts flow velocity conditioned on obs + waypoint."""

    def __init__(
        self,
        obs_dim=35,
        act_dim=12,
        wp_dim=3,
        hidden_dim=256,
        num_layers=4,
        time_emb_dim=32,
    ):
        super().__init__()
        self.time_emb = SinusoidalTimeEmb(time_emb_dim)

        in_dim = act_dim + time_emb_dim + obs_dim + wp_dim
        layers = []
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else act_dim
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t, obs, waypoint):
        t_emb = self.time_emb(t)
        inp = torch.cat([x_t, t_emb, obs, waypoint], dim=-1)
        return self.net(inp)


class FlowMatchingPolicy(nn.Module):
    """Waypoint-conditioned CFM policy."""

    def __init__(
        self,
        obs_dim=35,
        act_dim=12,
        wp_dim=3,
        hidden_dim=256,
        num_layers=4,
        time_emb_dim=32,
        num_steps=10,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.num_steps = num_steps

        self.net = VelocityNet(
            obs_dim, act_dim, wp_dim, hidden_dim, num_layers, time_emb_dim
        )

        # Normalizer buffers
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("act_mean", torch.zeros(act_dim))
        self.register_buffer("act_std", torch.ones(act_dim))
        self.register_buffer("wp_mean", torch.zeros(wp_dim))
        self.register_buffer("wp_std", torch.ones(wp_dim))

    def loss(self, obs, x_1, waypoint):
        """CFM OT loss with waypoint conditioning."""
        B = obs.shape[0]
        device = obs.device
        x_0 = torch.randn_like(x_1)
        t = torch.rand(B, device=device)
        t_b = t.view(B, 1)
        x_t = (1.0 - t_b) * x_0 + t_b * x_1
        u_t = x_1 - x_0
        v_t = self.net(x_t, t, obs, waypoint)
        return nn.functional.mse_loss(v_t, u_t)

    @torch.no_grad()
    def sample(self, obs_raw: torch.Tensor, waypoint_raw: torch.Tensor) -> torch.Tensor:
        """Integrate ODE from noise to action given obs and waypoint."""
        squeeze = obs_raw.dim() == 1
        if squeeze:
            obs_raw = obs_raw.unsqueeze(0)
            waypoint_raw = waypoint_raw.unsqueeze(0)

        obs_norm = (obs_raw - self.obs_mean) / self.obs_std
        wp_norm = (waypoint_raw - self.wp_mean) / self.wp_std

        B = obs_norm.shape[0]
        device = obs_norm.device
        x = torch.randn(B, self.act_dim, device=device)
        dt = 1.0 / self.num_steps

        for i in range(self.num_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.net(x, t, obs_norm, wp_norm)
            x = x + dt * v

        action = x * self.act_std + self.act_mean
        return action.squeeze(0) if squeeze else action

    def set_normalizer(self, stats: dict) -> None:
        self.obs_mean.copy_(stats["obs_mean"])
        self.obs_std.copy_(stats["obs_std"])
        self.act_mean.copy_(stats["act_mean"])
        self.act_std.copy_(stats["act_std"])
        self.wp_mean.copy_(stats["wp_mean"])
        self.wp_std.copy_(stats["wp_std"])
