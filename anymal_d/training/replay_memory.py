import numpy as np


def make_basic_dtype(n_obs: int = 37, n_act: int = 12) -> np.dtype:
    """Transition dtype for the basic (MultivariateNormal) trainer.

    log-prob is a vector of length n_act (one per action dimension).
    """
    return np.dtype([
        ("s",     np.float64, (n_obs,)),
        ("a",     np.float64, (n_act,)),
        ("a_logp", np.float64, (n_act,)),
        ("r",     np.float64),
        ("s_",    np.float64, (n_obs,)),
    ])


def make_advanced_dtype(n_obs: int = 35, n_act: int = 12) -> np.dtype:
    """Transition dtype for the advanced (learnable log_std) trainer.

    log-prob is a scalar (sum of independent Gaussians).
    Includes the done flag so the value target does not bootstrap past terminals.
    """
    return np.dtype([
        ("s",     np.float64, (n_obs,)),
        ("a",     np.float64, (n_act,)),
        ("a_logp", np.float64),
        ("r",     np.float64),
        ("s_",    np.float64, (n_obs,)),
        ("d",     np.float64),
    ])


class ReplayMemory:
    """Circular buffer that signals when full (time to run a PPO update)."""

    def __init__(self, capacity: int, dtype: np.dtype) -> None:
        self.buffer_capacity = capacity
        self.buffer = np.empty(capacity, dtype=dtype)
        self.counter = 0

    def store(self, transition: tuple) -> bool:
        """Store one transition; returns True when the buffer wraps (train now)."""
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        return False
