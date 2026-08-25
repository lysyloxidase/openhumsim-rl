from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite
from typing import Any

import numpy as np


ENVIRONMENT_SNAPSHOT_SCHEMA = "openhumsim.environment-snapshot.v1"
_NDARRAY_MARKER = "__openhumsim_ndarray_v1__"


def require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    supplied = set(value)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        raise ValueError(
            f"{label} fields do not match schema: "
            f"missing={missing}, extra={extra}"
        )


def encode_json_value(value: Any) -> Any:
    """Encode arrays and NumPy scalars into finite JSON-native values."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not isfinite(number):
            raise ValueError("snapshot values must be finite")
        return number
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise TypeError(f"unsupported snapshot ndarray dtype {value.dtype!s}")
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise ValueError("snapshot arrays must be finite")
        return {
            _NDARRAY_MARKER: {
                "dtype": value.dtype.str,
                "shape": [int(size) for size in value.shape],
                "data": value.tolist(),
            }
        }
    if isinstance(value, Mapping):
        encoded: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("snapshot mapping keys must be strings")
            encoded[key] = encode_json_value(item)
        return encoded
    if isinstance(value, (tuple, list)):
        return [encode_json_value(item) for item in value]
    raise TypeError(f"unsupported snapshot value type {type(value).__name__}")


def decode_json_value(value: Any) -> Any:
    """Decode values produced by :func:`encode_json_value`."""
    if isinstance(value, list):
        return [decode_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if _NDARRAY_MARKER in value:
            require_exact_keys(value, {_NDARRAY_MARKER}, label="ndarray envelope")
            record = value[_NDARRAY_MARKER]
            if not isinstance(record, Mapping):
                raise TypeError("ndarray envelope must contain a mapping")
            require_exact_keys(
                record,
                {"dtype", "shape", "data"},
                label="ndarray payload",
            )
            dtype = np.dtype(record["dtype"])
            if dtype.kind not in "biuf":
                raise ValueError(f"unsupported snapshot ndarray dtype {dtype!s}")
            shape_raw = record["shape"]
            if not isinstance(shape_raw, list):
                raise TypeError("ndarray shape must be a list")
            if any(
                isinstance(size, bool) or not isinstance(size, int)
                for size in shape_raw
            ):
                raise TypeError("ndarray shape entries must be integers")
            shape = tuple(shape_raw)
            if any(size < 0 for size in shape):
                raise ValueError("ndarray shape entries must be nonnegative")
            array = np.asarray(record["data"], dtype=dtype)
            try:
                array = array.reshape(shape)
            except ValueError as exc:
                raise ValueError("ndarray data do not match declared shape") from exc
            if dtype.kind == "f" and not np.all(np.isfinite(array)):
                raise ValueError("snapshot arrays must be finite")
            return array
        return {str(key): decode_json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("snapshot values must be finite")
        return value
    raise TypeError(f"unsupported JSON snapshot value type {type(value).__name__}")


def generator_with_state(
    template: np.random.Generator,
    encoded_state: Any,
    *,
    label: str,
) -> np.random.Generator:
    """Clone a generator and validate/assign a serialized bit-generator state."""
    state = decode_json_value(encoded_state)
    if not isinstance(state, Mapping):
        raise TypeError(f"{label} RNG state must decode to a mapping")
    candidate = deepcopy(template)
    try:
        candidate.bit_generator.state = dict(state)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid or incompatible {label} RNG state") from exc
    return candidate
