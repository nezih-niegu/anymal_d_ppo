from __future__ import annotations
from pathlib import Path
from typing import Union
from collections import deque
import numpy as np
import torch
from policies.base import BasePolicy

WAYPOINT_STRIDE = 50

class FlowMatchingPolicy(BasePolicy):
    obs_dim: int = 35
    act_dim: int = 12

    def __init__(self, num_steps: int = 10):
        self._model       = None
        self.num_steps    = num_steps
        self._xyz_buffer  = deque(maxlen=WAYPOINT_STRIDE + 1)
        self._current_xyz = None

    def load(self, checkpoint_path: Union[str, Path]) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self._model = torch.load(path, map_location="cpu", weights_only=False)
        if not hasattr(self._model, "net"):
            raise RuntimeError("Not a FlowMatchingPolicy checkpoint.")
        self._model.num_steps = self.num_steps
        self.set_eval_mode()

    def update_xyz(self, xyz: np.ndarray) -> None:
        self._xyz_buffer.append(xyz.copy())
        self._current_xyz = xyz.copy()

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call load() before act().")
        if len(self._xyz_buffer) >= WAYPOINT_STRIDE:
            waypoint = self._xyz_buffer[0]
        elif self._current_xyz is not None:
            waypoint = self._current_xyz
        else:
            waypoint = np.zeros(3, dtype=np.float32)
        obs_t = torch.from_numpy(obs.astype(np.float32))
        wp_t  = torch.from_numpy(waypoint.astype(np.float32))
        action = self._model.sample(obs_t, wp_t)
        return action.numpy().copy()

    def reset(self) -> None:
        self._xyz_buffer.clear()
        self._current_xyz = None

    def set_eval_mode(self) -> None:
        if self._model is not None:
            self._model.eval()

    @property
    def name(self) -> str:
        return "flow_matching"
