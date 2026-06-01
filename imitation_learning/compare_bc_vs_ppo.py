"""
compare_bc_vs_ppo.py
Compares Behavior Cloning policy against PPO baseline using metrics
from the previous quadruped waypoint control project.
Generates plots and a comparison table.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results/imitation_learning"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_bc_metrics():
    path = os.path.join(RESULTS_DIR, "eval_metrics.json")
    with open(path) as f:
        return json.load(f)

def load_training_results():
    path = os.path.join(RESULTS_DIR, "training_results.json")
    with open(path) as f:
        return json.load(f)

def load_ppo_baseline():
    """Use waypoint demo CSV as PPO-equivalent baseline metrics."""
    df = pd.read_csv("datasets/waypoint_walk_demo.csv")
    summary = pd.read_csv("datasets/summary_metrics.csv")
    # Use best controller (LQG had lowest resets=0)
    best = summary[summary["resets"] == summary["resets"].min()].iloc[0]
    return {
        "controller": best["controller"],
        "velocity_rmse": float(best["velocity_rmse"]),
        "xy_rmse": float(best["xy_rmse"]),
        "xy_max_error": float(best["xy_max_error"]),
        "final_xy_error": float(best["final_xy_error"]),
        "resets": int(best["resets"]),
    }, df, summary

def plot_training_curve(results):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(results["epochs"], results["train_loss"], label="Train Loss", color="blue")
    ax.plot(results["epochs"], results["val_loss"], label="Val Loss", color="orange", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Behavior Cloning — Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curve.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")

def plot_comparison(bc_metrics, ppo_baseline, summary):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Metric 1: Fall rate
    controllers = list(summary["controller"]) + ["BC (ours)"]
    fall_rates = [0.0] * len(summary) + [bc_metrics["fall_rate"]]
    colors = ["#4C72B0"] * len(summary) + ["#DD8452"]
    axes[0].bar(controllers, fall_rates, color=colors)
    axes[0].set_title("Fall Rate (lower is better)")
    axes[0].set_ylabel("Fall Rate")
    axes[0].set_ylim(0, 1)

    # Metric 2: Episode Return (BC only — PPO baseline not directly comparable)
    axes[1].bar(["BC (ours)"], [bc_metrics["mean_return"]], color="#DD8452")
    axes[1].set_title("Episode Return (BC policy)")
    axes[1].set_ylabel("Mean Return")
    axes[1].errorbar(["BC (ours)"], [bc_metrics["mean_return"]],
                     yerr=[bc_metrics["std_return"]], fmt="none", color="black", capsize=5)

    # Metric 3: XY tracking error (from waypoint controllers vs BC proxy)
    xy_rmse = list(summary["xy_rmse"])
    bc_proxy_rmse = bc_metrics["mean_distance"]
    controllers2 = list(summary["controller"]) + ["BC (ours)"]
    rmse_vals = xy_rmse + [bc_proxy_rmse]
    axes[2].bar(controllers2, rmse_vals, color=["#4C72B0"] * len(summary) + ["#DD8452"])
    axes[2].set_title("Trajectory Error / Distance (lower=better for RMSE)")
    axes[2].set_ylabel("RMSE / Distance (m)")

    plt.suptitle("BC Policy vs Waypoint Controllers Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "comparison_plot.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")

def save_comparison_table(bc_metrics, ppo_baseline, summary):
    rows = []
    for _, row in summary.iterrows():
        rows.append({
            "Method": row["controller"].upper(),
            "Type": "Optimal Control (baseline)",
            "XY RMSE (m)": round(row["xy_rmse"], 4),
            "Velocity RMSE": round(row["velocity_rmse"], 4),
            "Final XY Error (m)": round(row["final_xy_error"], 4),
            "Resets/Falls": int(row["resets"]),
            "Fall Rate": "-",
            "Mean Return": "-",
        })
    rows.append({
        "Method": "BC (Behavior Cloning)",
        "Type": "Imitation Learning (ours)",
        "XY RMSE (m)": "-",
        "Velocity RMSE": "-",
        "Final XY Error (m)": round(bc_metrics["mean_distance"], 4),
        "Resets/Falls": int(bc_metrics["fall_rate"] * 5),
        "Fall Rate": round(bc_metrics["fall_rate"], 3),
        "Mean Return": round(bc_metrics["mean_return"], 3),
    })
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    print()
    print(df.to_string(index=False))
    return df

def main():
    print("=== BC vs PPO/Optimal Control Comparison ===")
    bc_metrics = load_bc_metrics()
    results = load_training_results()
    ppo_baseline, df, summary = load_ppo_baseline()

    print(f"BC Metrics: {bc_metrics}")
    print(f"Best baseline: {ppo_baseline}")

    plot_training_curve(results)
    plot_comparison(bc_metrics, ppo_baseline, summary)
    save_comparison_table(bc_metrics, ppo_baseline, summary)
    print("Done.")

if __name__ == "__main__":
    main()
