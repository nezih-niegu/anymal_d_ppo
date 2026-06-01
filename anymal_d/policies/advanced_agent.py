import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class AdvancedAgent(nn.Module):
    """Shared-trunk actor-critic with a state-independent learnable log_std.

    The std is a learned parameter, so it adapts during training without
    manual annealing schedules.
    """

    def __init__(self, obs_len: int, act_len: int, log_std_init: float = -0.5) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.act_len = act_len
        self.mlp = nn.Sequential(
            nn.Linear(obs_len, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, act_len),
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )
        # exp(-0.5) ≈ 0.61 — reasonable initial exploration.
        self.log_std = nn.Parameter(torch.full((act_len,), log_std_init))

    def forward(self, state: torch.Tensor):
        h = self.mlp(state)
        return self.actor(h), self.critic(h)

    def dist(self, state: torch.Tensor):
        mean, value = self(state)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), value

    @torch.no_grad()
    def compute_action(self, state: np.ndarray):
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        d, value = self.dist(state_t)
        a = d.sample()
        logp = d.log_prob(a).sum(dim=-1)
        return a.squeeze(0).numpy(), float(logp.item()), float(value.item())
