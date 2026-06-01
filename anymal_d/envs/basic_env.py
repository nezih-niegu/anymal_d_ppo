import numpy as np
import mujoco

from anymal_d.envs.base_env import BaseEnv

MUJOCO_STEPS: int = 5
N_ACT: int = 12
N_OBS: int = 37          # qpos (19) + qvel (18)
FALL_THRESHOLD: float = 0.45


class BasicEnv(BaseEnv):
    """Simple ANYmal D env for the MultivariateNormal PPO trainer.

    Observation: raw qpos (19) ++ qvel (18) = 37 dims.
    Reward:      sum of forward velocity and height bonus per sub-step.
    """

    def reset(self) -> np.ndarray:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl = np.zeros(N_ACT)
        self.frames.clear()
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def step(self, action: np.ndarray, render: bool = False):
        self.done = False
        reward = 0.0
        self.data.ctrl = action
        for _ in range(MUJOCO_STEPS):
            mujoco.mj_step(self.model, self.data)
            reward += self.data.qvel[0] + (self.data.qpos[2] - 0.5)
            if render:
                self._capture_frame()

        state = np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])
        if self.data.time > self.DURATION:
            self.done = True
        if self.data.qpos[2] < FALL_THRESHOLD:
            self.done = True
            reward -= 100.0
        return state, reward, self.done
