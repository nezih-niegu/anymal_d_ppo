"""Verify that every public module can be imported without errors."""

import pytest


def test_package_root():
    import anymal_d
    assert hasattr(anymal_d, "__version__")


def test_utils():
    from anymal_d.utils.paths import find_project_file, SAVE_DIR, VIDEO_DIR, PROJECT_ROOT
    assert callable(find_project_file)
    assert isinstance(SAVE_DIR, str)


def test_config():
    from anymal_d.config import PPOConfig
    from anymal_d.config.config import PPOConfig as PPOConfig2
    assert PPOConfig is PPOConfig2


def test_envs():
    from anymal_d.envs.basic_env import BasicEnv, N_OBS, N_ACT
    from anymal_d.envs.advanced_env import AdvancedEnv, N_OBS as AN_OBS, N_ACT as AN_ACT
    assert N_OBS == 37
    assert AN_OBS == 35
    assert N_ACT == AN_ACT == 12


def test_policies():
    from anymal_d.policies.basic_agent import BasicAgent
    from anymal_d.policies.advanced_agent import AdvancedAgent
    assert BasicAgent is not None
    assert AdvancedAgent is not None


def test_training():
    from anymal_d.training.replay_memory import ReplayMemory, make_basic_dtype, make_advanced_dtype
    from anymal_d.training.ppo import train_basic, train_advanced
    from anymal_d.training.sweep import run_sweep, BASIC_SWEEP_CONFIG, ADVANCED_SWEEP_CONFIG
    assert callable(train_basic)
    assert callable(train_advanced)
    assert "parameters" in BASIC_SWEEP_CONFIG
    assert "parameters" in ADVANCED_SWEEP_CONFIG


def test_evaluation():
    from anymal_d.evaluation.evaluator import evaluate_episode
    assert callable(evaluate_episode)


def test_checkpoints():
    from anymal_d.checkpoints.manager import (
        save_checkpoint, load_checkpoint, pick_best_checkpoint, push_to_hub
    )
    assert callable(save_checkpoint)
    assert callable(load_checkpoint)


def test_tracking():
    from anymal_d.tracking.tracker import init_run, log, log_video, log_image, finish, is_active
    assert callable(is_active)
    assert callable(log)
