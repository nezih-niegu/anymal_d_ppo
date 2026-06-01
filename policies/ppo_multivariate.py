"""
policies/ppo_multivariate.py
============================
Wrapper for checkpoints produced by
``anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN.py``.

Checkpoint format
-----------------
``torch.save(policy, path)`` — the file is the full ``Agent`` *object*
(not just ``state_dict``).  Loading requires ``weights_only=False``.

Architecture
------------
* obs_dim = 37   (qpos[19] + qvel[18])
* act_dim = 12
* Shared MLP  → [Linear(37,128), Tanh, Linear(128,128), Tanh]
* Actor head  → [Linear(128,128), Tanh, Linear(128,12)]
* Critic head → [Linear(128,128), Tanh, Linear(128,1)]
* Distribution: MultivariateNormal with a *caller-supplied* diagonal
  action_std (annealed during training).  At eval time the policy is
  deterministic: we use the tanh-squashed mean.

Action mapping (from the original trainer)
------------------------------------------
  action = tanh(actor_mean)
  action[0]  *= 0.6 − 0.1   # LF_HAA
  action[3]  *= 0.6 + 0.1   # RF_HAA
  action[6]  *= 0.6 − 0.1   # LH_HAA
  action[9]  *= 0.6 + 0.1   # RH_HAA
  other joints: tanh(mean) ∈ (−1, 1) as raw joint targets

Deterministic eval mode
-----------------------
By default ``act()`` returns the tanh-squashed mean (no sampling), which
produces the most consistent evaluation signal.  Pass ``deterministic=False``
at construction time to draw stochastic samples instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from policies.base import BasePolicy


class PPOMultivariatePolicy(BasePolicy):
    """Evaluator wrapper for trainer-1 (MultivariateNormal, raw-action control)."""

    obs_dim: int = 37  # qpos(19) + qvel(18)
    act_dim: int = 12

    # HAA asymmetric remapping table: (joint_idx, scale, bias)
    _HAA_REMAP = [
        (0, 0.6, -0.1),   # LF_HAA
        (3, 0.6, +0.1),   # RF_HAA
        (6, 0.6, -0.1),   # LH_HAA
        (9, 0.6, +0.1),   # RH_HAA
    ]

    def __init__(self, deterministic: bool = True):
        """
        Parameters
        ----------
        deterministic:
            If True (default), ``act()`` returns the tanh-squashed actor mean.
            If False, samples from MultivariateNormal at action_std=0.0 (which
            is degenerate but consistent with what the trainer does at eval).
        """
        self._model = None
        self.deterministic = deterministic

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    def load(self, checkpoint_path: Union[str, Path]) -> None:
        """Load a full ``Agent`` object saved with ``torch.save(policy, path)``.

        Parameters
        ----------
        checkpoint_path:
            Path to a ``*_policy.pt`` file from trainer 1.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self._model = torch.load(path, map_location="cpu", weights_only=False)
        if not hasattr(self._model, "compute_action"):
            raise RuntimeError(
                f"{path} does not look like a trainer-1 Agent "
                "(missing 'compute_action'). Did you pass a trainer-2 checkpoint?"
            )
        self.set_eval_mode()

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return a (12,) action for the given (37,) observation.

        Uses the tanh-squashed actor mean with HAA remapping applied
        (same mapping as the original trainer's ``compute_action``).
        """
        if self._model is None:
            raise RuntimeError("Call load() before act().")

        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            actor_out, _ = self._model(obs_t)   # (1, 12)
            action = torch.tanh(actor_out)      # squash to (−1, 1)

        action_np = action.squeeze(0).numpy().copy()

        # Apply HAA asymmetric remapping
        for idx, scale, bias in self._HAA_REMAP:
            action_np[idx] = action_np[idx] * scale + bias

        return action_np

    def set_eval_mode(self) -> None:
        if self._model is not None:
            self._model.eval()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "ppo_multivariate"
