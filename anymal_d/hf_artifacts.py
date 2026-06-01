import json
import os
import re


def _resolve_repo_settings(repo_id=None, token=None, private=None):
    repo_id = repo_id or os.environ.get("HF_MODEL_REPO")
    if not repo_id:
        return None, None, None
    token = token if token is not None else os.environ.get("HF_TOKEN")
    if private is None:
        private = os.environ.get("HF_PRIVATE", "1") == "1"
    return repo_id, token, private


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-") or "artifact"


def upload_file_to_hub(
    local_path, path_in_repo=None, repo_id=None, token=None, private=None, repo_type="model"
):
    repo_id, token, private = _resolve_repo_settings(repo_id, token, private)
    if not repo_id:
        return None
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=private)
        repo_path = (path_in_repo or os.path.basename(local_path)).replace("\\", "/")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        url = f"https://huggingface.co/{repo_id}/blob/main/{repo_path}"
        print(f"  uploaded {os.path.basename(local_path)} -> {url}")
        return url
    except Exception as exc:
        print(f"  [hub] upload skipped ({exc})")
        return None


def save_metadata_json(save_dir, filename, payload):
    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, filename)
    with open(local_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return local_path


def publish_run_artifacts(
    *,
    save_dir,
    run_name,
    episode,
    running_reward,
    hparams,
    artifact_paths=None,
    metadata_extra=None,
    repo_subdirs=None,
    repo_id=None,
    token=None,
    private=None,
):
    artifact_paths = artifact_paths or {}
    repo_subdirs = repo_subdirs or {}
    metadata = {
        "run_name": run_name,
        "episode": int(episode),
        "running_reward": float(running_reward),
        "hparams": hparams,
        "artifacts": {
            kind: os.path.basename(path)
            for kind, path in artifact_paths.items()
            if path
        },
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    metadata_name = (
        f"{_safe_name(run_name)}_{int(episode)}_Reward-{float(running_reward):.2f}_metadata.json"
    )
    metadata_path = save_metadata_json(save_dir, metadata_name, metadata)
    upload_file_to_hub(
        metadata_path,
        path_in_repo=f"metadata/{os.path.basename(metadata_path)}",
        repo_id=repo_id,
        token=token,
        private=private,
    )

    for kind, local_path in artifact_paths.items():
        if not local_path:
            continue
        repo_subdir = repo_subdirs.get(kind, kind)
        upload_file_to_hub(
            local_path,
            path_in_repo=f"{repo_subdir}/{os.path.basename(local_path)}",
            repo_id=repo_id,
            token=token,
            private=private,
        )

    return metadata_path