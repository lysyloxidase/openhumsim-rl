from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _WrapperBase = gym.Wrapper
except ImportError:
    from .compat import spaces

    class _WrapperBase:
        """Small delegation wrapper for the NumPy-only compatibility runtime."""

        def __init__(self, env):
            self.env = env
            self.action_space = env.action_space
            self.observation_space = env.observation_space

        @property
        def unwrapped(self):
            return getattr(self.env, "unwrapped", self.env)

        def render(self):
            return self.env.render()

        def close(self):
            return self.env.close()

        def __getattr__(self, name: str):
            return getattr(self.env, name)

from .env import HumanHomeostasisEnv, OBSERVABLE_REWARD_PROFILE


OBSERVATION_HISTORY_SNAPSHOT_SCHEMA = "openhumsim.observation-history-runtime.v2"


class ObservationHistoryWrapper(_WrapperBase):
    """Expose a fixed, masked history of strict benchmark observations.

    The flattened observation layout is::

        [oldest observation, ..., newest observation, valid-history mask]

    The mask has one entry per history slot in the same oldest-to-newest order.
    At reset only the newest slot is valid; preceding slots contain zeros and
    have mask value 0. The wrapper returns the environment's ``info`` mapping
    unchanged, so it cannot turn debug-only state into an oracle policy input.
    """

    def __init__(self, env, history_length: int = 4):
        if (
            isinstance(history_length, bool)
            or not isinstance(history_length, (int, np.integer))
            or int(history_length) < 1
        ):
            raise ValueError("history_length must be an integer >= 1")
        required_profiles = {
            "observation_profile": "clinical",
            "measurement_profile": "realistic",
            "info_profile": "benchmark",
            "reward_profile": OBSERVABLE_REWARD_PROFILE,
        }
        for profile_name, required_value in required_profiles.items():
            if getattr(env, profile_name, None) != required_value:
                raise ValueError(
                    "ObservationHistoryWrapper requires "
                    f"{profile_name}={required_value!r}"
                )
        if not callable(getattr(env, "to_versioned_snapshot", None)):
            raise ValueError(
                "ObservationHistoryWrapper requires a snapshot-capable base environment"
            )

        base_space = getattr(env, "observation_space", None)
        if base_space is None or len(getattr(base_space, "shape", ())) != 1:
            raise ValueError("observation_space must be a one-dimensional Box")
        base_dtype = np.dtype(getattr(base_space, "dtype", np.float32))
        if not np.issubdtype(base_dtype, np.floating):
            raise ValueError("observation_space dtype must be floating point")
        base_low = np.asarray(base_space.low, dtype=base_dtype)
        base_high = np.asarray(base_space.high, dtype=base_dtype)
        if base_low.shape != base_space.shape or base_high.shape != base_space.shape:
            raise ValueError("observation_space bounds must match its shape")
        if not np.all(np.isfinite(base_low)) or not np.all(np.isfinite(base_high)):
            raise ValueError("observation_space bounds must be finite")
        if np.any(base_low > 0.0) or np.any(base_high < 0.0):
            raise ValueError("observation_space must contain zero for masked padding")

        base_names = tuple(getattr(env, "observation_names", ()))
        if len(base_names) != int(base_space.shape[0]):
            raise ValueError(
                "env.observation_names must match the observation-space width"
            )

        super().__init__(env)
        self._history_length = int(history_length)
        self._base_observation_size = int(base_space.shape[0])
        self._base_observation_dtype = base_dtype
        self._base_observation_names = tuple(str(name) for name in base_names)
        self._base_observation_low = base_low.copy()
        self._base_observation_high = base_high.copy()

        center = getattr(env, "_obs_center", None)
        scale = getattr(env, "_obs_scale", None)
        if (center is None) != (scale is None):
            raise ValueError("observation normalization contract is incomplete")
        normalization = None
        if center is not None:
            center_array = np.asarray(center, dtype=float)
            scale_array = np.asarray(scale, dtype=float)
            expected_normalization_shape = (self._base_observation_size,)
            if (
                center_array.shape != expected_normalization_shape
                or scale_array.shape != expected_normalization_shape
                or not np.all(np.isfinite(center_array))
                or not np.all(np.isfinite(scale_array))
                or np.any(scale_array <= 0.0)
            ):
                raise ValueError("observation normalization contract is invalid")
            normalization = {
                "transform": "tanh((raw-center)/scale)",
                "centers": [float(value) for value in center_array],
                "scales": [float(value) for value in scale_array],
            }

        measurement_model = getattr(env, "measurement_model", None)
        measurement_config = (
            None
            if measurement_model is None
            else asdict(measurement_model.config)
        )
        self._base_contract = {
            "shape": [self._base_observation_size],
            "dtype": self._base_observation_dtype.name,
            "observation_names": list(self._base_observation_names),
            "low": [float(value) for value in self._base_observation_low],
            "high": [float(value) for value in self._base_observation_high],
            "profiles": {
                "observation": getattr(env, "observation_profile", None),
                "measurement": getattr(env, "measurement_profile", None),
                "info": getattr(env, "info_profile", None),
                "reward": getattr(env, "reward_profile", None),
            },
            "measurement_config": measurement_config,
            "normalization": normalization,
        }
        self._history = np.zeros(
            (self._history_length, self._base_observation_size),
            dtype=self._base_observation_dtype,
        )
        self._valid_history = np.zeros(
            self._history_length, dtype=self._base_observation_dtype
        )

        history_low = np.tile(base_low, self._history_length)
        history_high = np.tile(base_high, self._history_length)
        mask_low = np.zeros(self._history_length, dtype=self._base_observation_dtype)
        mask_high = np.ones(self._history_length, dtype=self._base_observation_dtype)
        self.observation_space = spaces.Box(
            low=np.concatenate((history_low, mask_low)),
            high=np.concatenate((history_high, mask_high)),
            dtype=self._base_observation_dtype,
        )

        history_names: list[str] = []
        mask_names: list[str] = []
        for lag in range(self._history_length - 1, -1, -1):
            label = "t" if lag == 0 else f"t-{lag}"
            history_names.extend(
                f"history[{label}]::{name}" for name in self._base_observation_names
            )
            mask_names.append(f"history_valid[{label}]")
        self.observation_names = tuple(history_names + mask_names)

    @property
    def history_length(self) -> int:
        return self._history_length

    @property
    def base_observation_size(self) -> int:
        return self._base_observation_size

    @property
    def base_observation_names(self) -> tuple[str, ...]:
        return self._base_observation_names

    @property
    def history_values_slice(self) -> slice:
        return slice(0, self._history_length * self._base_observation_size)

    @property
    def valid_history_mask_slice(self) -> slice:
        start = self._history_length * self._base_observation_size
        return slice(start, start + self._history_length)

    @property
    def latest_observation_slice(self) -> slice:
        stop = self._history_length * self._base_observation_size
        return slice(stop - self._base_observation_size, stop)

    def _coerce_observation(self, observation) -> np.ndarray:
        value = np.asarray(observation, dtype=self._base_observation_dtype)
        expected = (self._base_observation_size,)
        if value.shape != expected:
            raise ValueError(
                f"wrapped observation must have shape {expected}, got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise FloatingPointError("wrapped observation contains NaN or infinity")
        return value

    def _stacked_observation(self) -> np.ndarray:
        return np.concatenate(
            (self._history.reshape(-1), self._valid_history)
        ).astype(self._base_observation_dtype, copy=False)

    def _base_environment_sha256(self) -> str:
        return self._canonical_sha256(self.env.to_versioned_snapshot())

    @staticmethod
    def _canonical_sha256(payload: Any) -> str:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return sha256(canonical).hexdigest()

    def runtime_snapshot(self) -> dict[str, Any]:
        """Return the wrapper-owned, versioned and JSON-safe history state.

        The underlying environment has its own transactional/runtime state and is
        intentionally not duplicated here. The SHA-256 digest binds this history
        to the exact full environment snapshot, including its RNG streams. Callers
        resuming a rollout must restore the base layer before restoring this one.
        """

        payload = {
            "schema": OBSERVATION_HISTORY_SNAPSHOT_SCHEMA,
            "history_length": self._history_length,
            "base_contract": deepcopy(self._base_contract),
            "base_environment_sha256": self._base_environment_sha256(),
            "history": self._history.tolist(),
            "valid_history_mask": self._valid_history.tolist(),
        }
        payload["snapshot_sha256"] = self._canonical_sha256(payload)
        return payload

    def restore_runtime_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a wrapper snapshot only when its complete contract matches."""

        if not isinstance(snapshot, Mapping):
            raise ValueError("history runtime snapshot must be a mapping")
        expected_keys = {
            "schema",
            "history_length",
            "base_contract",
            "base_environment_sha256",
            "history",
            "snapshot_sha256",
            "valid_history_mask",
        }
        if set(snapshot) != expected_keys:
            raise ValueError("history runtime snapshot has an invalid key set")
        if snapshot["schema"] != OBSERVATION_HISTORY_SNAPSHOT_SCHEMA:
            raise ValueError("unsupported history runtime snapshot schema")
        snapshot_length = snapshot["history_length"]
        if type(snapshot_length) is not int or snapshot_length != self._history_length:
            raise ValueError("history runtime snapshot length does not match wrapper")
        if snapshot["base_contract"] != self._base_contract:
            raise ValueError("history runtime snapshot base contract does not match")
        base_environment_sha256 = snapshot["base_environment_sha256"]
        if (
            not isinstance(base_environment_sha256, str)
            or len(base_environment_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in base_environment_sha256
            )
        ):
            raise ValueError("history runtime snapshot base environment digest is invalid")
        snapshot_sha256 = snapshot["snapshot_sha256"]
        if (
            not isinstance(snapshot_sha256, str)
            or len(snapshot_sha256) != 64
            or any(character not in "0123456789abcdef" for character in snapshot_sha256)
        ):
            raise ValueError("history runtime snapshot digest is invalid")

        try:
            history = np.asarray(
                snapshot["history"], dtype=self._base_observation_dtype
            )
            valid = np.asarray(
                snapshot["valid_history_mask"],
                dtype=self._base_observation_dtype,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("history runtime snapshot is not numeric") from exc
        expected_history_shape = (
            self._history_length,
            self._base_observation_size,
        )
        if history.shape != expected_history_shape:
            raise ValueError(
                "history runtime snapshot buffer has an invalid shape"
            )
        if valid.shape != (self._history_length,):
            raise ValueError("history runtime snapshot mask has an invalid shape")
        if not np.all(np.isfinite(history)) or not np.all(np.isfinite(valid)):
            raise ValueError("history runtime snapshot must contain finite values")
        if not np.all((valid == 0.0) | (valid == 1.0)):
            raise ValueError("history runtime snapshot mask must be binary")
        if valid[-1] != 1.0:
            raise ValueError("history runtime snapshot latest observation must be valid")
        if np.any(np.diff(valid) < 0.0):
            raise ValueError(
                "history runtime snapshot valid entries must form a suffix"
            )
        if np.any(history[valid == 0.0] != 0.0):
            raise ValueError("history runtime snapshot padding must be zero")
        if np.any(history < self._base_observation_low) or np.any(
            history > self._base_observation_high
        ):
            raise ValueError("history runtime snapshot observation is out of bounds")
        binding_payload = {
            name: snapshot[name]
            for name in expected_keys
            if name != "snapshot_sha256"
        }
        try:
            expected_snapshot_sha256 = self._canonical_sha256(binding_payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("history runtime snapshot cannot be hashed") from exc
        if snapshot_sha256 != expected_snapshot_sha256:
            raise ValueError("history runtime snapshot digest does not match payload")
        if base_environment_sha256 != self._base_environment_sha256():
            raise ValueError(
                "history runtime snapshot does not match the current base environment"
            )
        current_observation = self._coerce_observation(self.env._get_obs())
        if not np.array_equal(history[-1], current_observation):
            raise ValueError(
                "history runtime snapshot latest observation does not match "
                "the current base environment"
            )

        self._history[...] = history
        self._valid_history[...] = valid

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        observation, info = self.env.reset(seed=seed, options=options)
        self._history.fill(0.0)
        self._valid_history.fill(0.0)
        self._history[-1] = self._coerce_observation(observation)
        self._valid_history[-1] = 1.0
        return self._stacked_observation(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self._history_length > 1:
            self._history[:-1] = self._history[1:]
            self._valid_history[:-1] = self._valid_history[1:]
        self._history[-1] = self._coerce_observation(observation)
        self._valid_history[-1] = 1.0
        return (
            self._stacked_observation(),
            reward,
            terminated,
            truncated,
            info,
        )


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
