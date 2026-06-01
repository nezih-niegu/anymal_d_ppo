"""Config loading and override tests."""

import os
import pytest
from anymal_d.config import PPOConfig

_HERE = os.path.dirname(os.path.dirname(__file__))
_BASIC_CFG   = os.path.join(_HERE, "anymal_d", "config", "default_basic.yaml")
_ADVANCED_CFG = os.path.join(_HERE, "anymal_d", "config", "default_advanced.yaml")


def test_defaults():
    cfg = PPOConfig()
    assert cfg.gamma == 0.99
    assert cfg.batch_size == 128
    assert cfg.num_episodes == 15000


def test_load_basic_yaml():
    cfg = PPOConfig.from_yaml(_BASIC_CFG)
    assert cfg.lr == pytest.approx(1e-5)
    assert cfg.ppo_epoch == 48
    assert cfg.std_init == pytest.approx(1.0)
    assert cfg.std_min  == pytest.approx(0.6)


def test_load_advanced_yaml():
    cfg = PPOConfig.from_yaml(_ADVANCED_CFG)
    assert cfg.lr == pytest.approx(3e-4)
    assert cfg.ppo_epoch == 10
    assert cfg.fall_threshold == pytest.approx(0.35)


def test_from_dict():
    cfg = PPOConfig.from_dict({"lr": 1e-3, "batch_size": 64})
    assert cfg.lr == pytest.approx(1e-3)
    assert cfg.batch_size == 64
    assert cfg.gamma == 0.99  # untouched default


def test_with_overrides():
    cfg = PPOConfig()
    cfg2 = cfg.with_overrides({"lr": 5e-4, "ppo_epoch": 20})
    assert cfg2.lr == pytest.approx(5e-4)
    assert cfg2.ppo_epoch == 20
    assert cfg.lr == pytest.approx(3e-4)  # original unchanged


def test_to_dict_roundtrip():
    cfg = PPOConfig(lr=2e-4, ppo_epoch=15)
    d = cfg.to_dict()
    cfg2 = PPOConfig.from_dict(d)
    assert cfg2.lr == pytest.approx(cfg.lr)
    assert cfg2.ppo_epoch == cfg.ppo_epoch


def test_unknown_keys_ignored():
    cfg = PPOConfig.from_dict({"lr": 1e-4, "unknown_future_param": 99})
    assert cfg.lr == pytest.approx(1e-4)
