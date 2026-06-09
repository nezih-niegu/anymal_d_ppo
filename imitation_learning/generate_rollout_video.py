"""
generate_rollout_video.py
Generates a rollout video of the BC policy in MuJoCo.
"""

import os
import sys
import json
import numpy as np
import torch
import mujoco
import mediapy as media
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_scene_xml():
    seeds = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    for seed in seeds:
        d = seed
        for _ in range(6):
            cand = os.path.join(d, "anybotics_anymal_d", "scene.xml")
            if os.path.exists(cand):
                return cand
            d = os.path.dirname(d)
    raise FileNotFoundError("scene.xml not found")


def get_obs(data):
    return np.concatenate([data.qpos.copy(), data.qvel.copy()])


class BCPolicy(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, n_layers):
        super().__init__()
        layers = [torch.nn.Linear(obs_dim, hidden_dim), torch.nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()]
        layers.append(torch.nn.Linear(hidden_dim, act_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, obs):
        return torch.tanh(self.net(obs))


def generate_video(checkpoint_path, config_path, output_path, n_steps=300):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mc = cfg["model"]
    policy = BCPolicy(mc["obs_dim"], mc["act_dim"], mc["hidden_dim"], mc["n_layers"])
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    xml_path = find_scene_xml()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    frames = []

    for step in range(n_steps):
        obs = torch.tensor(get_obs(data), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = policy(obs).squeeze(0).numpy()
        data.ctrl[:] = action
        mujoco.mj_step(model, data)

        if step % 2 == 0:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    media.write_video(output_path, frames, fps=30)
    print(f"Video saved: {output_path} ({len(frames)} frames)")


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    generate_video(
        checkpoint_path="pretrained_models/anymal_d/bc/bc_final.pt",
        config_path="configs/imitation_learning.yaml",
        output_path="results/imitation_learning/bc_rollout.mp4",
        n_steps=300,
    )
