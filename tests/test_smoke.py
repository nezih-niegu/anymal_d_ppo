"""
Smoke tests — verify a full training step (buffer fill + PPO update) runs
end-to-end without errors. Uses a tiny replay buffer (100 transitions) so
the test completes in a few seconds.
"""

import numpy as np
import pytest
import torch

from anymal_d.envs.basic_env import BasicEnv, N_OBS as BASIC_N_OBS, N_ACT
from anymal_d.envs.advanced_env import AdvancedEnv, N_OBS as ADV_N_OBS
from anymal_d.policies.basic_agent import BasicAgent
from anymal_d.policies.advanced_agent import AdvancedAgent
from anymal_d.training.replay_memory import ReplayMemory, make_basic_dtype, make_advanced_dtype
from anymal_d.training.ppo import train_basic, train_advanced

REPLAY_SIZE = 100
BATCH_SIZE  = 32
HPARAMS_BASIC = dict(
    gamma=0.99, ppo_epoch=2, batch_size=BATCH_SIZE,
    clip_param=0.1, c1=1.0, c2=0.001,
)
HPARAMS_ADV = dict(
    gamma=0.99, ppo_epoch=2, batch_size=BATCH_SIZE,
    clip_param=0.2, c1=0.5, c2=0.005, max_grad_norm=0.5,
)


def _fill_basic_buffer():
    """Return a full BasicEnv replay buffer."""
    env    = BasicEnv()
    policy = BasicAgent(BASIC_N_OBS, N_ACT)
    mem    = ReplayMemory(REPLAY_SIZE, make_basic_dtype(BASIC_N_OBS, N_ACT))
    action_std = 0.8

    state = env.reset()
    while True:
        action, a_logp, _ = policy.compute_action(state, action_std)
        next_state, reward, done = env.step(action, render=False)
        if mem.store((state, action, a_logp, reward, next_state)):
            break
        state = env.reset() if done else next_state
    return policy, mem


def _fill_advanced_buffer():
    """Return a full AdvancedEnv replay buffer."""
    env    = AdvancedEnv()
    policy = AdvancedAgent(ADV_N_OBS, N_ACT)
    mem    = ReplayMemory(REPLAY_SIZE, make_advanced_dtype(ADV_N_OBS, N_ACT))

    state = env.reset()
    while True:
        action, a_logp, _ = policy.compute_action(state)
        next_state, reward, done = env.step(action, render=False)
        if mem.store((state, action, a_logp, reward, next_state, float(done))):
            break
        state = env.reset() if done else next_state
    return policy, mem


def test_basic_ppo_update():
    torch.manual_seed(42)
    policy, mem = _fill_basic_buffer()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    pl, vl, ent, ratio = train_basic(policy, optimizer, mem, HPARAMS_BASIC, 0.8)
    assert np.isfinite(pl),    f"policy_loss is not finite: {pl}"
    assert np.isfinite(vl),    f"value_loss is not finite: {vl}"
    assert np.isfinite(ent),   f"entropy is not finite: {ent}"
    assert np.isfinite(ratio), f"ratio is not finite: {ratio}"


def test_advanced_ppo_update():
    torch.manual_seed(42)
    policy, mem = _fill_advanced_buffer()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    pl, vl, ent, ratio = train_advanced(policy, optimizer, mem, HPARAMS_ADV)
    assert np.isfinite(pl),    f"policy_loss is not finite: {pl}"
    assert np.isfinite(vl),    f"value_loss is not finite: {vl}"
    assert np.isfinite(ent),   f"entropy is not finite: {ent}"
    assert np.isfinite(ratio), f"ratio is not finite: {ratio}"


def test_basic_agent_compute_action():
    policy = BasicAgent(BASIC_N_OBS, N_ACT)
    obs = np.zeros(BASIC_N_OBS, dtype=np.float32)
    action, logp, value = policy.compute_action(obs, 0.8)
    assert action.shape == (1, N_ACT)
    assert np.all(np.isfinite(action))


def test_advanced_agent_compute_action():
    policy = AdvancedAgent(ADV_N_OBS, N_ACT)
    obs = np.zeros(ADV_N_OBS, dtype=np.float32)
    action, logp, value = policy.compute_action(obs)
    assert action.shape == (N_ACT,)
    assert np.all(np.isfinite(action))
    assert np.isfinite(logp)
