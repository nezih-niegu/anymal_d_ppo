#!/usr/bin/env python3
"""
imitation/train_flow.py
=======================
Training loop for waypoint-conditioned Flow Matching.

Usage
-----
python imitation/train_flow.py \
    --dataset data/ppo_demos.npz \
    --epochs 100 \
    --batch-size 512 \
    --out pretrained_models/flow_matching/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from imitation.waypoints import WaypointDemoDataset
from imitation.flow_matching import FlowMatchingPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="data/ppo_demos.npz")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int,   default=256)
    parser.add_argument("--num-layers", type=int,   default=4)
    parser.add_argument("--num-steps",  type=int,   default=10)
    parser.add_argument("--val-split",  type=float, default=0.1)
    parser.add_argument("--out",        default="pretrained_models/flow_matching")
    parser.add_argument("--seed",       type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_dataset = WaypointDemoDataset(args.dataset, normalize=True)
    n_val   = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {n_train} | Val: {n_val}")

    model = FlowMatchingPolicy(
        obs_dim=full_dataset.obs.shape[1],
        act_dim=full_dataset.actions.shape[1],
        wp_dim=full_dataset.waypoints.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_steps=args.num_steps,
    ).to(device)

    stats = full_dataset.normalizer_state()
    model.set_normalizer({k: v.to(device) for k, v in stats.items()})

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    log = {"train_loss": [], "val_loss": []}
    t0  = time.perf_counter()

    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for obs, act, wp in train_loader:
            obs, act, wp = obs.to(device), act.to(device), wp.to(device)
            loss = model.loss(obs, act, wp)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs, act, wp in val_loader:
                obs, act, wp = obs.to(device), act.to(device), wp.to(device)
                val_losses.append(model.loss(obs, act, wp).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        scheduler.step()

        log["train_loss"].append(round(train_loss, 6))
        log["val_loss"].append(round(val_loss, 6))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, out_dir / "best_flow_policy.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{args.epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"best={best_val_loss:.5f}  time={time.perf_counter()-t0:.0f}s")

    torch.save(model, out_dir / "last_flow_policy.pt")
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print("-" * 60)
    print(f"Done. Best val loss: {best_val_loss:.5f}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
