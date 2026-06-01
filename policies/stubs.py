"""
policies/stubs.py
=================
Placeholder policy classes for architectures not yet implemented.

Each stub:
  * Is a valid ``BasePolicy`` subclass (passes isinstance checks).
  * Raises ``NotImplementedError`` from ``load()`` and ``act()`` with a
    clear message telling developers what to fill in.
  * Documents the expected checkpoint format and action mapping convention
    so future implementers have a specification to work from.

Stub classes
------------
DiffusionPolicy       — Score / DDPM-style diffusion over the action space
FlowMatchingPolicy    — Continuous normalizing flow / flow-matching policy
ScriptedWaypointPolicy — Hard-coded joint-angle waypoints with interpolation
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from policies.base import BasePolicy


# ---------------------------------------------------------------------------
# Diffusion Policy stub
# ---------------------------------------------------------------------------

class DiffusionPolicy(BasePolicy):
    """Stub for a diffusion-based policy (e.g. Chi et al. 2023 architecture).

    Expected checkpoint format (TBD)
    ---------------------------------
    A directory or ``.pt`` file containing:
      * ``noise_net``  — the denoising U-Net / transformer weights
      * ``normalizer`` — obs / action normalizer state

    Action mapping convention
    -------------------------
    Should output raw (12,) joint targets compatible with the trainer-2
    delta-around-nominal convention so it can share the same ``ANYmalEnv``.

    TODO
    ----
    1. Choose a denoising backbone (U-Net, transformer).
    2. Decide the number of diffusion steps and the noise schedule.
    3. Implement ``load()`` and ``act()`` in this class.
    4. Register 'diffusion' in ``evaluate.py``'s POLICY_REGISTRY.
    """

    obs_dim: int = 35   # match trainer-2 observation space
    act_dim: int = 12

    def load(self, checkpoint_path: Union[str, Path]) -> None:
        raise NotImplementedError(
            "DiffusionPolicy.load() is not yet implemented. "
            "See the docstring in policies/stubs.py for the expected format."
        )

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "DiffusionPolicy.act() is not yet implemented."
        )

    @property
    def name(self) -> str:
        return "diffusion"


# ---------------------------------------------------------------------------
# Flow Matching Policy stub
# ---------------------------------------------------------------------------

class FlowMatchingPolicy(BasePolicy):
    """Stub for a continuous-normalizing-flow / flow-matching policy.

    Expected checkpoint format (TBD)
    ---------------------------------
    A ``.pt`` file containing:
      * ``velocity_net`` — the learned velocity field
      * ``normalizer``   — obs / action normalizer state

    Action mapping convention
    -------------------------
    Same delta-around-nominal convention as trainer-2 / DiffusionPolicy.

    TODO
    ----
    1. Choose a flow-matching formulation (OT-CFM, Lipman et al., etc.).
    2. Implement an ODE integrator (fixed-step RK4 is sufficient for eval).
    3. Implement ``load()`` and ``act()`` in this class.
    4. Register 'flow_matching' in ``evaluate.py``'s POLICY_REGISTRY.
    """

    obs_dim: int = 35
    act_dim: int = 12

    def load(self, checkpoint_path: Union[str, Path]) -> None:
        raise NotImplementedError(
            "FlowMatchingPolicy.load() is not yet implemented. "
            "See the docstring in policies/stubs.py for the expected format."
        )

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "FlowMatchingPolicy.act() is not yet implemented."
        )

    @property
    def name(self) -> str:
        return "flow_matching"


# ---------------------------------------------------------------------------
# Scripted Waypoint Policy stub
# ---------------------------------------------------------------------------

class ScriptedWaypointPolicy(BasePolicy):
    """Stub for a scripted, model-free waypoint-tracking policy.

    This policy does not learn; it cycles through a fixed sequence of joint
    angle targets.  Useful as a baseline and for debugging the evaluator.

    Expected checkpoint format
    --------------------------
    A YAML or JSON file with a list of (12,) joint-angle waypoints:

        waypoints:
          - [q0, q1, ..., q11]   # step 0
          - [q0, q1, ..., q11]   # step 1
          ...
        steps_per_waypoint: 20   # env steps to hold each pose

    Action mapping convention
    -------------------------
    Outputs *absolute* joint targets (raw-action convention, trainer-1 style).
    The shared ``ANYmalEnv`` must be told which action convention to use; see
    ``envs/anymal_env.py``.

    TODO
    ----
    1. Define a standing → trot gait sequence for ANYmal D.
    2. Implement ``load()`` to parse the YAML waypoint file.
    3. Implement ``act()`` to return the current waypoint and advance the
       internal counter.
    4. Override ``reset()`` to reset the waypoint cursor to step 0.
    5. Register 'scripted' in ``evaluate.py``'s POLICY_REGISTRY.
    """

    obs_dim: int = 37   # can work with full obs; it simply ignores it
    act_dim: int = 12

    def __init__(self) -> None:
        self._waypoints: list = []
        self._steps_per_waypoint: int = 20
        self._step_counter: int = 0

    def load(self, checkpoint_path: Union[str, Path]) -> None:
        raise NotImplementedError(
            "ScriptedWaypointPolicy.load() is not yet implemented. "
            "See the docstring in policies/stubs.py for the YAML format."
        )

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "ScriptedWaypointPolicy.act() is not yet implemented."
        )

    def reset(self) -> None:
        self._step_counter = 0

    @property
    def name(self) -> str:
        return "scripted"
