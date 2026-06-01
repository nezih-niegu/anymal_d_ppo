"""
policies/ppo_normal.py
======================
Wrapper for checkpoints produced by
``anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py``.

Checkpoint format
-----------------
``torch.save(policy, path)`` — the full ``Agent`` *object*.
Loading requires ``weights_only=False``.

Architecture
------------
* obs_dim = 35   (qpos[2:] → 17 dims  +  qvel[18] → 18 dims)
  NOTE: world (x, y) are *dropped* from qpos before building the observation.
* act_dim = 12
* Shared MLP  → [Linear(35,128), Tanh, Linear(128,128), Tanh]
* Actor head  → [Linear(128,128), Tanh, Linear(128,12)]
* Critic head → [Linear(128,128), Tanh, Linear(128,1)]
* ``log_std`` → nn.Parameter of shape (12,)  (state-independent)
* Distribution: independent Normal(actor_mean, exp(log_std))

Action mapping
--------------
The env applies:
    ctrl = nominal_pose + ACTION_SCALE * tanh(raw_action)

The policy outputs *un-squashed* samples. ``act()`` here mirrors what
``Env.step()`` does so that the evaluator's shared env can simply receive the
raw sample and apply the same transformation.

Deterministic eval mode
-----------------------
Pass ``deterministic=True`` (default) to return the raw actor mean instead of
sampling.  The tanh squash + ACTION_SCALE scaling still happens inside the
shared ``ANYmalEnv`` (see ``envs/anymal_env.py``), so this wrapper returns the
*pre-squash* value, consistent with what the trainer stored in the replay
buffer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import sys
sys.path.insert(0, "anymal_d")
import RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING as _trainer
import __main__
__main__.Agent = _trainer.Agent

from policies.base import BasePolicy


class PPONormalPolicy(BasePolicy):
    """Evaluator wrapper for trainer-2 (Normal dist, learnable log_std, delta actions)."""

    obs_dim: int = 35   # qpos[2:] (17) + qvel (18)
    act_dim: int = 12

    def __init__(self, deterministic: bool = True):
        """
        Parameters
        ----------
        deterministic:
            If True (default), ``act()`` returns the raw actor mean (no
            sampling).  The environment still applies ``tanh`` and the
            ``ACTION_SCALE`` factor.
            If False, draws a sample from Normal(mean, exp(log_std)).
        """
        self._model = None
        self.deterministic = deterministic

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    def load(self, checkpoint_path: Union[str, Path]) -> None:
        """Load a full trainer-2 ``Agent`` object.

        Parameters
        ----------
        checkpoint_path:
            Path to a ``*_policy.pt`` file from trainer 2.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self._model = torch.load(path, map_location="cpu", weights_only=False)

        # Trainer-2 Agents have a learnable ``log_std`` parameter and a
        # ``dist()`` method; trainer-1 Agents do not.
        if not hasattr(self._model, "log_std"):
            raise RuntimeError(
                f"{path} does not look like a trainer-2 Agent "
                "(missing 'log_std' parameter). Did you pass a trainer-1 checkpoint?"
            )
        self.set_eval_mode()

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return a (12,) *pre-squash* action for the given (35,) observation.

        The shared ANYmalEnv will apply ``tanh`` and ACTION_SCALE before
        passing the value to MuJoCo.
        """
        if self._model is None:
            raise RuntimeError("Call load() before act().")

        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            if self.deterministic:
                mean, _ = self._model(obs_t)   # (1, 12)
                action = mean
            else:
                dist, _ = self._model.dist(obs_t)
                action = dist.sample()         # (1, 12)

        return action.squeeze(0).numpy().copy()

    def set_eval_mode(self) -> None:
        if self._model is not None:
            self._model.eval()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "ppo_normal"
