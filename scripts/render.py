"""
Render evaluation videos from a trained ANYmal D policy.

Works with checkpoints from *either* trainer variant — pass --basic for the
MultivariateNormal checkpoint, omit for the advanced (learnable log_std) one.

Usage:
    python scripts/render.py                              # auto-pick best checkpoint (advanced)
    python scripts/render.py --policy path/to/x_policy.pt
    python scripts/render.py --basic --num-videos 3
    python scripts/render.py --hf-repo user/anymal-d-ppo --hf-file x_policy.pt
"""

import argparse
import torch

from anymal_d.envs.basic_env import BasicEnv
from anymal_d.envs.advanced_env import AdvancedEnv
from anymal_d.evaluation.evaluator import evaluate_episode
from anymal_d.checkpoints.manager import pick_best_checkpoint, load_checkpoint, download_from_hub
from anymal_d.utils.paths import SAVE_DIR


def render(
    num_videos: int = 5,
    policy_path: str | None = None,
    use_basic: bool = False,
    action_std: float = 0.858,
    fall_threshold: float = 0.3,
) -> None:
    if policy_path is None:
        policy_path = pick_best_checkpoint(SAVE_DIR)
    print(f"Loading: {policy_path}")

    policy = load_checkpoint(policy_path)
    env = BasicEnv() if use_basic else AdvancedEnv(fall_threshold=fall_threshold)
    torch.manual_seed(0)

    for i in range(num_videos):
        ep_reward, _ = evaluate_episode(
            env, policy, i + 1,
            action_std=action_std if use_basic else None,
            log_media=False, prefix="eval",
        )
        print(f"Video #{i + 1}  reward: {ep_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render ANYmal D evaluation videos")
    parser.add_argument("--num-videos",     type=int,   default=5)
    parser.add_argument("--policy",         type=str,   default=None)
    parser.add_argument("--basic",          action="store_true",
                        help="load a BasicAgent checkpoint (MultivariateNormal)")
    parser.add_argument("--std-init",       type=float, default=0.858,
                        help="action std for basic-trainer checkpoints")
    parser.add_argument("--fall-threshold", type=float, default=0.3)
    parser.add_argument("--hf-repo",        type=str,   default=None)
    parser.add_argument("--hf-file",        type=str,   default=None)
    args = parser.parse_args()

    policy_path = args.policy
    if args.hf_repo and args.hf_file:
        policy_path = download_from_hub(args.hf_repo, args.hf_file)

    render(
        num_videos=args.num_videos,
        policy_path=policy_path,
        use_basic=args.basic,
        action_std=args.std_init,
        fall_threshold=args.fall_threshold,
    )
