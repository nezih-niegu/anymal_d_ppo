"""Environment reset / step tests — no rendering (headless-safe)."""

import numpy as np
import pytest

from anymal_d.envs.basic_env import BasicEnv, N_OBS as BASIC_N_OBS, N_ACT
from anymal_d.envs.advanced_env import AdvancedEnv, N_OBS as ADV_N_OBS


# ---------------------------------------------------------------------------
# BasicEnv
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def basic_env():
    return BasicEnv()


def test_basic_reset_shape(basic_env):
    obs = basic_env.reset()
    assert obs.shape == (BASIC_N_OBS,), f"Expected ({BASIC_N_OBS},), got {obs.shape}"


def test_basic_step_no_render(basic_env):
    basic_env.reset()
    action = np.zeros(N_ACT)
    obs, reward, done = basic_env.step(action, render=False)
    assert obs.shape == (BASIC_N_OBS,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_basic_step_multiple(basic_env):
    basic_env.reset()
    for _ in range(10):
        action = np.random.uniform(-0.5, 0.5, N_ACT)
        obs, reward, done = basic_env.step(action, render=False)
        assert obs.shape == (BASIC_N_OBS,)
        if done:
            break


# ---------------------------------------------------------------------------
# AdvancedEnv
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def advanced_env():
    return AdvancedEnv(fall_threshold=0.35)


def test_advanced_reset_shape(advanced_env):
    obs = advanced_env.reset()
    assert obs.shape == (ADV_N_OBS,), f"Expected ({ADV_N_OBS},), got {obs.shape}"


def test_advanced_step_no_render(advanced_env):
    advanced_env.reset()
    action = np.zeros(N_ACT)
    obs, reward, done = advanced_env.step(action, render=False)
    assert obs.shape == (ADV_N_OBS,)
    assert isinstance(reward, float)
    assert np.isfinite(reward)


def test_advanced_step_multiple(advanced_env):
    advanced_env.reset()
    for _ in range(10):
        action = np.random.uniform(-1.0, 1.0, N_ACT)
        obs, reward, done = advanced_env.step(action, render=False)
        assert obs.shape == (ADV_N_OBS,)
        assert np.all(np.isfinite(obs))
        if done:
            break


def test_advanced_nominal_pose(advanced_env):
    assert advanced_env.nominal_pose.shape == (N_ACT,)
    assert advanced_env.target_height > 0.3
