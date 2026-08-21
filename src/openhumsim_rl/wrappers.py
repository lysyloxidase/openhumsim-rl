from __future__ import annotations

import numpy as np

try:
    from gymnasium import spaces
except ImportError:
    from .compat import spaces

from .env import HumanHomeostasisEnv


class SymmetricActionHumanEnv(HumanHomeostasisEnv):
    """Optional [-1, 1] policy-facing action space with a neutral zero.

    OpenHumSim interventions are physically one-sided (a negative meal, saline
    bolus or drug dose is not meaningful), so a linear [-1,1] -> [0,1] map would
    make policy action 0 equal to 50% intervention. That is a poor RL neutral point.

    This wrapper instead maps:
        native = max(policy_action, 0)

    Thus 0 and negative actions are no-op for the corresponding actuator, and +1
    is maximum intervention. The negative half-space is intentionally redundant.
    The native [0,1] environment is preferred when the RL library handles one-sided
    bounded actions directly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(
            low=-np.ones_like(self.action_space.low, dtype=np.float32),
            high=np.ones_like(self.action_space.high, dtype=np.float32),
            dtype=np.float32,
        )

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        if a.shape != self.action_space.shape:
            raise ValueError(f"Action must have shape {self.action_space.shape}, got {a.shape}.")
        if not np.all(np.isfinite(a)):
            raise ValueError("Action contains NaN or infinity.")
        a = np.clip(a, -1.0, 1.0)
        native = np.maximum(a, 0.0).astype(np.float32)
        obs, reward, terminated, truncated, info = super().step(native)
        info["symmetric_policy_action"] = {
            name: float(value) for name, value in zip(info["action_names"], a)
        }
        info["symmetric_mapping"] = "native=max(policy_action,0); zero is no intervention"
        return obs, reward, terminated, truncated, info
