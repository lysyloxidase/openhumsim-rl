"""Tiny fallback subset of the Gymnasium API.

If Gymnasium is installed, OpenHumSim-RL uses the real package.
This module only keeps the environment runnable in a minimal NumPy-only setup.
"""

from __future__ import annotations
import numpy as np


class Env:
    metadata = {}

    def __init__(self):
        self.np_random = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

    def close(self):
        return None


class Box:
    def __init__(self, low, high, shape=None, dtype=np.float32):
        self.dtype = np.dtype(dtype)
        if shape is not None:
            self.low = np.full(shape, low, dtype=self.dtype)
            self.high = np.full(shape, high, dtype=self.dtype)
        else:
            self.low = np.asarray(low, dtype=self.dtype)
            self.high = np.asarray(high, dtype=self.dtype)
        self.shape = self.low.shape
        self._rng = np.random.default_rng()

    def sample(self):
        return self._rng.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x):
        x = np.asarray(x)
        return (
            x.shape == self.shape
            and np.all(np.isfinite(x))
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        return [seed]


class _Spaces:
    Box = Box


spaces = _Spaces()
