"""Thin wrapper around trackio (wandb-compatible experiment tracker)."""

import os
import trackio as wandb

PROJECT = "AIDL-PPO-ANYMAL_D"


def init_run(config: dict, run_name: str | None = None):
    """Initialise a trackio run; return the run object."""
    if not os.environ.get("TRACKIO_SPACE_ID"):
        os.environ.pop("TRACKIO_SPACE_ID", None)
    return wandb.init(
        project=PROJECT,
        name=run_name,
        config=config,
        space_id=os.environ.get("TRACKIO_SPACE_ID"),
    )


def is_active() -> bool:
    return wandb.run is not None


def log(metrics: dict) -> None:
    wandb.log(metrics)


def log_video(path: str, key: str = "Video eval", fps: int = 4) -> None:
    try:
        wandb.log({key: wandb.Video(path, fps=fps, format="mp4")})
    except Exception as exc:
        print(f"  [trackio] video log skipped ({exc})")


def log_image(path: str, key: str = "Reward eval") -> None:
    try:
        wandb.log({key: wandb.Image(path)})
    except Exception as exc:
        print(f"  [trackio] image log skipped ({exc})")


def finish() -> None:
    wandb.finish()
