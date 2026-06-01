"""
policies/base.py
================
Abstract base class for all ANYmal D policy wrappers.

Every policy, regardless of architecture or checkpoint format, must expose the
same two-method interface so ``evaluate.py`` and future tooling can treat them
interchangeably.

Interface contract
------------------
load(checkpoint_path)
    Restore weights (and any auxiliary state such as action_std) from
    ``checkpoint_path``.  Called once before the first rollout.

act(obs) -> np.ndarray
    Given a 1-D observation array, return a 1-D action array of shape (12,).
    Must be deterministic or stochastic depending on the policy's mode; the
    evaluator never calls anything other than ``act``.

Implementations must *not* modify the environment or maintain hidden env state.
All random seeding should be done by the caller before the first ``act`` call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np


class BasePolicy(ABC):
    """Abstract policy wrapper for ANYmal D evaluation."""

    # Subclasses should set these as class attributes or in __init__.
    obs_dim: int = 0   # expected observation dimensionality (for documentation)
    act_dim: int = 12  # always 12 for ANYmal D

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self, checkpoint_path: Union[str, Path]) -> None:
        """Load weights / model state from *checkpoint_path*.

        Parameters
        ----------
        checkpoint_path:
            Path to the checkpoint file (format is implementation-specific).

        Raises
        ------
        FileNotFoundError
            If *checkpoint_path* does not exist.
        RuntimeError
            If the file exists but cannot be loaded (wrong format, wrong arch,
            etc.).
        """

    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Map a single observation to an action.

        Parameters
        ----------
        obs:
            1-D float array of shape ``(obs_dim,)``.

        Returns
        -------
        np.ndarray
            1-D float array of shape ``(12,)`` — the raw joint targets or
            delta-around-nominal commands, depending on the policy type.
        """

    # ------------------------------------------------------------------
    # Optional hooks (no-ops by default)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Called at the start of every episode.

        Override in policies that maintain hidden state across steps (e.g. an
        RNN, a diffusion denoising loop, a scripted waypoint cursor).
        """

    def set_eval_mode(self) -> None:
        """Switch the underlying model to inference mode (e.g. torch eval()).

        Called once by the evaluator before rollouts begin.
        """

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable policy name (class name by default)."""
        return self.__class__.__name__

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.name}(obs_dim={self.obs_dim}, act_dim={self.act_dim})"

