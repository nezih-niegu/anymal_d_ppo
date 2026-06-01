"""
Behavior Cloning for ANYmal D — Imitation Learning Pipeline
============================================================
Observation space : 37-dim (19 qpos + 18 qvel)
Action space      : 12-dim (joint position targets)
Method            : Behavior Cloning (supervised learning on state-action pairs)

Usage:
    python imitation_learning/bc_train.py --config configs/imitation_learning.yaml
    python imitation_learning/bc_train.py --config configs/imitation_learning.yaml --generate-demos
"""

import os
import sys
import math
import logging
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import yaml
import mujoco

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("bc_train")


def load_config(path):
    log.info(f"Loading config from {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


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


def generate_demos(cfg):
    log.info("Generating demonstration trajectories...")
    xml_path = find_scene_xml()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    n_episodes = cfg["dataset"]["n_episodes"]
    ep_steps = cfg["dataset"]["episode_steps"]
    all_obs, all_acts = [], []

    for ep in range(n_episodes):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        nominal = model.key_qpos[0][7:19].copy()
        for step in range(ep_steps):
            obs = get_obs(data)
            noise = 0.05 * math.exp(-ep / max(n_episodes, 1))
            action = np.clip(nominal + np.random.randn(12) * noise, -1.0, 1.0)
            all_obs.append(obs.astype(np.float32))
            all_acts.append(action.astype(np.float32))
            data.ctrl[:] = action
            mujoco.mj_step(model, data)
        log.info(f"  Episode {ep+1}/{n_episodes} done")

    observations = np.array(all_obs, dtype=np.float32)
    actions = np.array(all_acts, dtype=np.float32)
    log.info(f"Dataset: obs={observations.shape}, acts={actions.shape}")
    return observations, actions


def save_demos(observations, actions, cfg):
    path = cfg["dataset"]["path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, observations=observations, actions=actions)
    log.info(f"Demos saved to {path}")


def load_demos(cfg):
    path = cfg["dataset"]["path"]
    log.info(f"Loading demos from {path}")
    d = np.load(path)
    return d["observations"], d["actions"]


class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, n_layers):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, act_dim))
        self.net = nn.Sequential(*layers)
        log.info(f"BCPolicy: {obs_dim} -> {hidden_dim}x{n_layers} -> {act_dim}")

    def forward(self, obs):
        return torch.tanh(self.net(obs))


def train(cfg):
    device = torch.device(cfg["training"]["device"])
    log.info(f"Training on: {device}")

    observations, actions = load_demos(cfg)
    obs_t = torch.tensor(observations)
    act_t = torch.tensor(actions)
    dataset = TensorDataset(obs_t, act_t)
    val_size = max(1, int(len(dataset) * cfg["training"]["val_split"]))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"])

    mc = cfg["model"]
    policy = BCPolicy(
        mc["obs_dim"], mc["act_dim"], mc["hidden_dim"], mc["n_layers"]
    ).to(device)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=cfg["training"]["learning_rate"]
    )

    ckpt_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    results = {"train_loss": [], "val_loss": [], "epochs": []}
    epochs = cfg["training"]["epochs"]

    for epoch in range(1, epochs + 1):
        policy.train()
        train_losses = []
        for obs_b, act_b in train_loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            loss = F.mse_loss(policy(obs_b), act_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        policy.eval()
        val_losses = []
        with torch.no_grad():
            for obs_b, act_b in val_loader:
                val_losses.append(
                    F.mse_loss(policy(obs_b.to(device)), act_b.to(device)).item()
                )

        tl, vl = np.mean(train_losses), np.mean(val_losses)
        results["train_loss"].append(float(tl))
        results["val_loss"].append(float(vl))
        results["epochs"].append(epoch)
        log.info(f"Epoch {epoch}/{epochs} - train={tl:.5f} val={vl:.5f}")

        if epoch % cfg["training"]["checkpoint_freq"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"bc_epoch_{epoch}.pt")
            torch.save(
                {"epoch": epoch, "model_state": policy.state_dict(), "config": cfg},
                ckpt_path,
            )
            log.info(f"Checkpoint: {ckpt_path}")

    final = os.path.join(ckpt_dir, "bc_final.pt")
    torch.save(
        {"epoch": epochs, "model_state": policy.state_dict(), "config": cfg}, final
    )
    log.info(f"Final checkpoint: {final}")

    results_dir = cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return policy, results


def evaluate_bc(policy, cfg, n_episodes=5):
    log.info("Evaluating BC policy...")
    xml_path = find_scene_xml()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    device = torch.device(cfg["training"]["device"])
    policy.eval()

    ep_returns, fall_count, distances = [], [], []
    for ep in range(n_episodes):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        init_x = data.qpos[0]
        total_reward = 0.0
        fell = False
        for step in range(cfg["dataset"]["episode_steps"]):
            obs = (
                torch.tensor(get_obs(data), dtype=torch.float32).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                action = policy(obs).squeeze(0).cpu().numpy()
            data.ctrl[:] = action
            mujoco.mj_step(model, data)
            height = data.qpos[2]
            total_reward += height - 0.3
            if height < 0.2:
                fell = True
                break
        ep_returns.append(total_reward)
        fall_count.append(int(fell))
        distances.append(abs(data.qpos[0] - init_x))
        log.info(f"  Ep {ep+1}: return={total_reward:.2f} fell={fell}")

    metrics = {
        "mean_return": float(np.mean(ep_returns)),
        "std_return": float(np.std(ep_returns)),
        "fall_rate": float(np.mean(fall_count)),
        "mean_distance": float(np.mean(distances)),
    }
    log.info(f"BC Metrics: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/imitation_learning.yaml")
    parser.add_argument("--generate-demos", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.generate_demos or not os.path.exists(cfg["dataset"]["path"]):
        obs, acts = generate_demos(cfg)
        save_demos(obs, acts, cfg)

    if args.eval_only:
        mc = cfg["model"]
        policy = BCPolicy(
            mc["obs_dim"], mc["act_dim"], mc["hidden_dim"], mc["n_layers"]
        )
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        policy.load_state_dict(ckpt["model_state"])
        metrics = evaluate_bc(policy, cfg)
    else:
        policy, results = train(cfg)
        metrics = evaluate_bc(policy, cfg)

    results_dir = cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Done.")


if __name__ == "__main__":
    main()
