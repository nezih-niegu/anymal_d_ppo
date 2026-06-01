"""Checkpoint save / load / Hugging Face Hub helpers."""

from __future__ import annotations
import os
import re
import glob
import torch


def save_checkpoint(
    policy, optimizer, save_dir: str, run_name: str, episode: int, reward: float
) -> tuple[str, str]:
    """Save policy and optimizer; return (policy_path, optimizer_path)."""
    os.makedirs(save_dir, exist_ok=True)
    tag = f"{run_name}_{episode}_Reward-{reward}"
    policy_path = os.path.join(save_dir, f"{tag}_policy.pt")
    optim_path  = os.path.join(save_dir, f"{tag}_optimizer.pt")
    torch.save(policy, policy_path)
    torch.save(optimizer, optim_path)
    return policy_path, optim_path


def load_checkpoint(policy_path: str, map_location: str = "cpu"):
    """Load and return a policy in eval mode."""
    policy = torch.load(policy_path, map_location=map_location, weights_only=False)
    policy.eval()
    return policy


def pick_best_checkpoint(save_dir: str) -> str:
    """Return the path of the checkpoint with the highest reward in *save_dir*."""
    candidates = glob.glob(os.path.join(save_dir, "*_policy.pt"))
    candidates += glob.glob(os.path.join(save_dir, "*olicy*.pt"))
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            f"No policy checkpoints found in {save_dir}. "
            "Train first or pass --policy explicitly."
        )

    def _score(path: str) -> tuple[float, float]:
        m = re.search(r"Reward[-_]([-\d.]+)", os.path.basename(path))
        reward = float(m.group(1)) if m else float("-inf")
        return (reward, os.path.getmtime(path))

    return max(candidates, key=_score)


def push_to_hub(local_path: str, path_in_repo: str | None = None) -> str | None:
    """Upload *local_path* to a Hugging Face Hub model repo (no-op if HF_MODEL_REPO unset)."""
    repo_id = os.environ.get("HF_MODEL_REPO")
    if not repo_id:
        return None
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(
            repo_id, repo_type="model", exist_ok=True,
            private=os.environ.get("HF_PRIVATE", "1") == "1",
        )
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo or os.path.basename(local_path),
            repo_id=repo_id, repo_type="model",
        )
        url = f"https://huggingface.co/{repo_id}"
        print(f"  uploaded {os.path.basename(local_path)} → {url}")
        return url
    except Exception as exc:
        print(f"  [hub] upload skipped ({exc})")
        return None


def download_from_hub(repo_id: str, filename: str) -> str:
    """Download *filename* from a Hugging Face Hub model repo; return local path."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=repo_id, filename=filename, token=os.environ.get("HF_TOKEN")
    )
    print(f"Downloaded {filename} from {repo_id} → {path}")
    return path
