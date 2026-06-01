"""
imitation/waypoints.py
======================
Extracts waypoints from the PPO demonstration dataset and provides
a conditioned dataset for waypoint-conditioned Flow Matching.

Waypoint definition
-------------------
A waypoint is the xyz body position (qpos[0:3]) at a future timestep.
For each transition at step t, the waypoint is the xyz position at
step t + WAYPOINT_STRIDE (50 steps = 0.5 seconds ahead).

If t + WAYPOINT_STRIDE exceeds the episode length, we use the last
available xyz in the episode.

Dataset output
--------------
Each sample is: (obs, action, next_waypoint)
  obs           : (35,)  — current observation
  action        : (12,)  — expert action
  next_waypoint : (3,)   — xyz body position 50 steps ahead
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

WAYPOINT_STRIDE = 50   # steps ahead for the waypoint target


class WaypointDemoDataset(Dataset):
    """Loads (obs, action, next_waypoint) from a .npz demo file with xyz."""

    def __init__(self, npz_path: str | Path, normalize: bool = True,
                 waypoint_stride: int = WAYPOINT_STRIDE):
        data = np.load(npz_path)

        if "xyz" not in data:
            raise ValueError(
                f"{npz_path} does not contain 'xyz'. "
                "Re-run generate_dataset.py to include body positions."
            )

        obs_raw     = data["obs"]      # (N, 35)
        actions_raw = data["actions"]  # (N, 12)
        xyz_raw     = data["xyz"]      # (N, 3)
        ep_ids      = data["ep_ids"]   # (N,)

        # Build next_waypoint for each transition
        next_waypoints = self._build_waypoints(xyz_raw, ep_ids, waypoint_stride)

        self.obs          = torch.from_numpy(obs_raw).float()
        self.actions      = torch.from_numpy(actions_raw).float()
        self.waypoints    = torch.from_numpy(next_waypoints).float()
        self.normalize    = normalize

        if normalize:
            self.obs_mean = self.obs.mean(0)
            self.obs_std  = self.obs.std(0).clamp(min=1e-6)
            self.act_mean = self.actions.mean(0)
            self.act_std  = self.actions.std(0).clamp(min=1e-6)
            self.wp_mean  = self.waypoints.mean(0)
            self.wp_std   = self.waypoints.std(0).clamp(min=1e-6)
        else:
            self.obs_mean = torch.zeros(obs_raw.shape[1])
            self.obs_std  = torch.ones(obs_raw.shape[1])
            self.act_mean = torch.zeros(actions_raw.shape[1])
            self.act_std  = torch.ones(actions_raw.shape[1])
            self.wp_mean  = torch.zeros(3)
            self.wp_std   = torch.ones(3)

        print(f"WaypointDataset loaded: {len(self)} transitions")
        print(f"  obs={self.obs.shape}  actions={self.actions.shape}  waypoints={self.waypoints.shape}")

    @staticmethod
    def _build_waypoints(xyz: np.ndarray, ep_ids: np.ndarray,
                         stride: int) -> np.ndarray:
        """For each step i, find xyz at step i+stride within the same episode."""
        N = len(xyz)
        next_wp = np.zeros((N, 3), dtype=np.float32)

        for i in range(N):
            target_idx = i + stride
            # Stay within the same episode
            if target_idx < N and ep_ids[target_idx] == ep_ids[i]:
                next_wp[i] = xyz[target_idx]
            else:
                # Find last step of this episode
                ep = ep_ids[i]
                mask = ep_ids == ep
                last_idx = np.where(mask)[0][-1]
                next_wp[i] = xyz[last_idx]

        return next_wp

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx):
        obs = (self.obs[idx]       - self.obs_mean) / self.obs_std
        act = (self.actions[idx]   - self.act_mean) / self.act_std
        wp  = (self.waypoints[idx] - self.wp_mean)  / self.wp_std
        return obs, act, wp

    def normalizer_state(self) -> dict:
        return {
            "obs_mean": self.obs_mean,
            "obs_std":  self.obs_std,
            "act_mean": self.act_mean,
            "act_std":  self.act_std,
            "wp_mean":  self.wp_mean,
            "wp_std":   self.wp_std,
        }


def make_waypoint_dataloader(npz_path, batch_size=512, normalize=True,
                             num_workers=2):
    dataset = WaypointDemoDataset(npz_path, normalize=normalize)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=num_workers, pin_memory=True, drop_last=True)
    return loader, dataset
