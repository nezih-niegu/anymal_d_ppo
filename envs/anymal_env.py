"""
envs/anymal_env.py
==================
Shared MuJoCo environment for ANYmal D policy evaluation.

This is a *clean-room* re-implementation of the environment logic extracted
from the two trainers.  It resolves the single biggest incompatibility between
them: the two different action conventions.

Action conventions
------------------
``"raw"``   (trainer 1 / ppo_multivariate)
    The policy output is used directly as the joint-angle target.  The HAA
    asymmetric remapping is applied *inside the policy wrapper* (see
    ``policies/ppo_multivariate.py``), so the env just does:
        ctrl = action

``"delta"`` (trainer 2 / ppo_normal, diffusion, flow_matching)
    The policy output is an *un-squashed* delta.  The env applies:
        ctrl = nominal_pose + ACTION_SCALE * tanh(action)
    Matches exactly what ``RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py``
    does in ``Env.step()``.

Observation conventions
-----------------------
``"full"``  (trainer 1, scripted)  → np.concatenate([qpos (19), qvel (18)])  → 37-D
``"noxy"``  (trainer 2, diffusion, flow_matching)
                                   → np.concatenate([qpos[2:] (17), qvel (18)]) → 35-D

Both conventions are produced by the same underlying simulation; the env
returns the requested slice without re-running the simulation.

Reward
------
The evaluator always uses the *trainer-2 shaped reward* (velocity + height +
upright + alive_bonus + costs) because it is more informative for comparison.
The trainer-1 reward (``qvel[0] + qpos[2] − 0.5``) is available via the
``reward_fn="simple"`` argument for backward-compatibility checks.

Fall threshold
--------------
``fall_threshold=0.35`` matches trainer 2 (training).  Pass 0.30 to match
trainer 2's *render* mode, or 0.45 to match trainer 1.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Tuple

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Locate scene.xml robustly (same logic as the trainers)
# ---------------------------------------------------------------------------

def _find_project_file(rel_path: str) -> str:
    seeds = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    for seed in seeds:
        d = seed
        for _ in range(6):
            cand = os.path.join(d, rel_path)
            if os.path.exists(cand):
                return cand
            d = os.path.dirname(d)
    return os.path.join(".", rel_path)


DEFAULT_XML = _find_project_file(os.path.join("anybotics_anymal_d", "scene.xml"))

# Trainer-2 constant; exposed here so wrappers don't hard-code it.
ACTION_SCALE: float = 0.4
N_ACT: int = 12
MUJOCO_STEPS: int = 5
FRAMERATE: int = 60
EPISODE_DURATION: float = 8.0   # seconds
TIMESTEP: float = 0.002


ActionConvention = Literal["raw", "delta"]
ObsConvention = Literal["full", "noxy"]
RewardFn = Literal["shaped", "simple"]


class ANYmalEnv:
    """Shared evaluation environment for ANYmal D.

    Parameters
    ----------
    model_xml:
        Path to ``scene.xml``.  Defaults to the auto-located scene.
    action_convention:
        ``"raw"`` — action used directly as ctrl target (trainer-1 style).
        ``"delta"`` — ctrl = nominal + ACTION_SCALE * tanh(action) (trainer-2).
    obs_convention:
        ``"full"`` — 37-D (qpos+qvel), ``"noxy"`` — 35-D (qpos[2:]+qvel).
    reward_fn:
        ``"shaped"`` — rich trainer-2 reward (recommended for evaluation).
        ``"simple"`` — trainer-1 raw reward for backward-compatibility.
    fall_threshold:
        Body z-height below which the episode terminates as a fall.
    """

    def __init__(
        self,
        model_xml: str = DEFAULT_XML,
        action_convention: ActionConvention = "delta",
        obs_convention: ObsConvention = "noxy",
        reward_fn: RewardFn = "shaped",
        fall_threshold: float = 0.35,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model)

        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.model.opt.timestep = TIMESTEP
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.distance = 5.0

        self.action_convention: ActionConvention = action_convention
        self.obs_convention: ObsConvention = obs_convention
        self.reward_fn: RewardFn = reward_fn
        self.fall_threshold = fall_threshold

        # Read nominal pose and target height from the XML home keyframe.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.nominal_pose: np.ndarray = self.data.qpos[7 : 7 + N_ACT].copy()
        self.target_height: float = float(self.data.qpos[2])

        self._last_action: np.ndarray = np.zeros(N_ACT)
        self.frames: list = []
        self.done: bool = False

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        return 37 if self.obs_convention == "full" else 35

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the simulation to the home keyframe.

        Parameters
        ----------
        seed:
            If provided, seeds ``numpy`` for reproducibility (MuJoCo itself is
            deterministic given the same initial state).
        """
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = self.nominal_pose
        self._last_action[:] = 0.0
        self.frames.clear()
        self.done = False
        return self._obs()

    def step(
        self, action: np.ndarray, render: bool = False
    ) -> Tuple[np.ndarray, float, bool]:
        """Advance the simulation by one policy step (MUJOCO_STEPS sub-steps).

        Parameters
        ----------
        action:
            Raw policy output of shape (12,).  Interpretation depends on
            ``action_convention`` set at construction.
        render:
            If True, append an offscreen frame to ``self.frames``.

        Returns
        -------
        obs, reward, done
        """
        self.done = False
        action = np.asarray(action, dtype=np.float64).reshape(N_ACT)

        # Apply action convention
        if self.action_convention == "delta":
            bounded = np.tanh(action)
            self.data.ctrl[:] = self.nominal_pose + ACTION_SCALE * bounded
        else:  # "raw"
            bounded = action  # HAA remapping already done in PPOMultivariatePolicy
            self.data.ctrl[:] = action

        reward_acc = 0.0
        for _ in range(MUJOCO_STEPS):
            mujoco.mj_step(self.model, self.data)
            if self.reward_fn == "simple":
                reward_acc += self.data.qvel[0] + (self.data.qpos[2] - 0.5)
            if render and (len(self.frames) < self.data.time * FRAMERATE):
                self.camera.lookat = self.data.body("LH_SHANK").subtree_com
                self.renderer.update_scene(self.data, self.camera)
                self.frames.append(self.renderer.render().copy())

        if self.reward_fn == "shaped":
            reward_acc = self._shaped_reward(bounded)

        self._last_action = bounded.copy()

        # Termination conditions
        if self.data.time > EPISODE_DURATION:
            self.done = True
        height = float(self.data.qpos[2])
        if height < self.fall_threshold:
            self.done = True
            if self.reward_fn == "shaped":
                reward_acc -= 5.0           # soft fall penalty (trainer-2)
            else:
                reward_acc -= 100.0         # hard fall penalty (trainer-1)

        return self._obs(), reward_acc, self.done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        if self.obs_convention == "full":
            return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])
        else:  # "noxy"
            return np.concatenate([self.data.qpos[2:].copy(), self.data.qvel.copy()])

    def _shaped_reward(self, bounded_action: np.ndarray) -> float:
        """Trainer-2 shaped reward (computed once, after all MUJOCO_STEPS)."""
        forward_vel = float(self.data.qvel[0])
        vel_reward = 2.0 * float(np.tanh(forward_vel))

        height = float(self.data.qpos[2])
        height_reward = -2.0 * (height - self.target_height) ** 2

        qw = float(self.data.qpos[3])
        upright_reward = 0.5 * qw ** 2

        alive_bonus = 0.5
        action_cost = -0.002 * float(np.sum(bounded_action ** 2))
        smooth_cost = -0.002 * float(np.sum((bounded_action - self._last_action) ** 2))
        joint_vel_cost = -0.0001 * float(np.sum(self.data.qvel[6:] ** 2))

        return (
            vel_reward
            + height_reward
            + upright_reward
            + alive_bonus
            + action_cost
            + smooth_cost
            + joint_vel_cost
        )

    def save_video(self, path: str | Path, fps: int = FRAMERATE) -> str:
        """Write accumulated frames to an mp4 and return the path."""
        try:
            import mediapy as media
        except ImportError as exc:
            raise ImportError("Install mediapy to save videos: pip install mediapy") from exc
        media.write_video(str(path), self.frames, fps=fps)
        return str(path)
