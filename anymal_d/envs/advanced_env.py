import numpy as np
import mujoco

from anymal_d.envs.base_env import BaseEnv

MUJOCO_STEPS: int = 5
N_ACT: int = 12
# Drop world x, y (grow unbounded); keep z, quaternion, joints + all velocities.
N_OBS: int = (19 - 2) + 18   # 17 + 18 = 35
ACTION_SCALE: float = 0.4     # max delta per joint after tanh squash (radians)


class AdvancedEnv(BaseEnv):
    """Shaped ANYmal D env for the learnable-log_std PPO trainer.

    Observation: qpos[2:] ++ qvel = 35 dims (world x/y excluded).
    Action:      small delta around the nominal standing pose.
    Reward:      tanh velocity + height + upright + alive − action/smooth/joint-vel costs.
    """

    def __init__(self, fall_threshold: float = 0.35) -> None:
        super().__init__()
        self.fall_threshold = fall_threshold
        self.viewer = None

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.nominal_pose: np.ndarray = self.data.qpos[7 : 7 + N_ACT].copy()
        self.target_height: float = float(self.data.qpos[2])
        self.last_action: np.ndarray = np.zeros(N_ACT)

    # ------------------------------------------------------------------
    # Viewer (optional — requires a display)
    # ------------------------------------------------------------------

    def attach_viewer(self) -> None:
        import mujoco.viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 5
        print("Live viewer attached. Close the window to detach.")

    def detach_viewer(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

    def _sync_viewer(self) -> None:
        if self.viewer is None:
            return
        if not self.viewer.is_running():
            self.viewer = None
            return
        self.viewer.cam.lookat[:] = self.data.body("LH_SHANK").subtree_com
        self.viewer.sync()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos[2:].copy(),   # z, quat (4), 12 joints → 17
            self.data.qvel.copy(),        # 6 base vel + 12 joint vel → 18
        ])

    def reset(self) -> np.ndarray:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = self.nominal_pose
        self.last_action[:] = 0.0
        self.frames.clear()
        return self._obs()

    def step(self, raw_action: np.ndarray, render: bool = False):
        """Apply delta-around-nominal action; compute shaped reward."""
        self.done = False
        raw_action = np.asarray(raw_action, dtype=np.float64).reshape(N_ACT)
        bounded = np.tanh(raw_action)
        self.data.ctrl[:] = self.nominal_pose + ACTION_SCALE * bounded

        for _ in range(MUJOCO_STEPS):
            mujoco.mj_step(self.model, self.data)
            self._sync_viewer()
            if render:
                self._capture_frame()

        height = self.data.qpos[2]
        reward = (
            2.0 * np.tanh(self.data.qvel[0])          # forward velocity
            + -2.0 * (height - self.target_height) ** 2  # stay near home height
            + 0.5 * self.data.qpos[3] ** 2             # upright (qw)
            + 0.5                                       # alive bonus
            + -0.002 * float(np.sum(bounded ** 2))     # torque proxy
            + -0.002 * float(np.sum((bounded - self.last_action) ** 2))  # smoothness
            + -0.0001 * float(np.sum(self.data.qvel[6:] ** 2))           # joint vel
        )
        self.last_action = bounded.copy()

        if self.data.time > self.DURATION:
            self.done = True
        if height < self.fall_threshold:
            self.done = True
            reward -= 5.0

        return self._obs(), reward, self.done
