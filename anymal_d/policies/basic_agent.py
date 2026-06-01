import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class BasicAgent(nn.Module):
    """Actor-critic with fixed MultivariateNormal exploration.

    The action std is passed externally and annealed by the training loop,
    making it easy to control exploration decay without modifying the network.
    """

    def __init__(self, obs_len: int, act_len: int) -> None:
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

    def forward(self, state: torch.Tensor):
        h = self.mlp(state)
        return self.actor(h), self.critic(h)

    def compute_action(self, state: np.ndarray, action_std: float):
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self(state_t)
        probs = torch.tanh(probs)

        action_var = torch.full((self.act_len,), action_std * action_std)
        cov_mat = torch.diag(action_var).unsqueeze(0)
        dist = MultivariateNormal(probs, cov_mat)
        action = dist.sample()
        action_clamped = torch.tanh(action)

        # ANYmal D HAA joints have asymmetric travel: 0.6 * x ± 0.1 keeps
        # commands well within the hardware limits.
        action_clamped[0][0] = action_clamped[0][0] * 0.6 - 0.1
        action_clamped[0][3] = action_clamped[0][3] * 0.6 + 0.1
        action_clamped[0][6] = action_clamped[0][6] * 0.6 - 0.1
        action_clamped[0][9] = action_clamped[0][9] * 0.6 + 0.1

        return (
            action_clamped.detach().numpy(),
            dist.log_prob(action_clamped).detach().numpy(),
            state_value.detach(),
        )
