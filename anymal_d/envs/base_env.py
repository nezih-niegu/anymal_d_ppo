import os
import numpy as np
import mujoco
import mediapy as media

from anymal_d.utils.paths import MODEL_XML, VIDEO_DIR


class BaseEnv:
    """Shared MuJoCo setup for all ANYmal D environments."""

    FRAMERATE: int = 60
    DURATION: float = 8.0
    TIMESTEP: float = 0.002

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.model.opt.timestep = self.TIMESTEP
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.distance = 5
        self.frames: list = []
        self.done: bool = False
        self._renderer = None  # lazy — avoids requiring a GL context in headless tests

    @property
    def renderer(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model)
        return self._renderer

    def _capture_frame(self) -> None:
        if len(self.frames) < self.data.time * self.FRAMERATE:
            self.camera.lookat = self.data.body("LH_SHANK").subtree_com
            self.renderer.update_scene(self.data, self.camera)
            self.frames.append(self.renderer.render().copy())

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: np.ndarray, render: bool = False):
        raise NotImplementedError

    def close(self, episode: int, reward: float,
              prefix: str = "video", video_dir: str | None = None) -> str:
        out_dir = video_dir or VIDEO_DIR
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{prefix}_{episode}_reward_{reward:.2f}.mp4")
        media.write_video(path, self.frames, fps=self.FRAMERATE)
        return path
