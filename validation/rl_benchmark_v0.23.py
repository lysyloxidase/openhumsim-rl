"""Short, deterministic evaluation harness for observable baseline policies.

This script performs no training and loads no learned policy. It compares a
native no-op action with a deliberately simple observation-only engineering
heuristic on evaluation-only scenarios. The output is a reproducibility record,
not evidence of clinical validity, safety, or policy optimality.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from numbers import Integral
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from openhumsim_rl import (
    HumanConfig,
    HumanHomeostasisEnv,
    ObservationHistoryWrapper,
    __version__,
)
from openhumsim_rl.env import ACTION_NAMES
from openhumsim_rl.units import ACTION_SEMANTICS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "validation" / "rl_benchmark_v0.23.json"
BENCHMARK_SCHEMA = "openhumsim.rl-benchmark.v1"
DEFAULT_SEEDS = (23001, 23002)
DEFAULT_SCENARIOS = ("respiratory_acidosis", "dehydrated")
# The default realistic ABG channel is first sampled at 30 minutes and becomes
# available after its analytical delay. A 60-minute evaluation therefore lets
# a history-based policy observe and act on at least one delayed result.
DEFAULT_EPISODE_MINUTES = 60.0
DEFAULT_HISTORY_LENGTH = 4

HEURISTIC_RAW_THRESHOLDS = {
    "low_glucose_mg_dl": 75.0,
    "low_map_mmHg": 65.0,
    "low_spo2_pct": 92.0,
    "high_paco2_mmHg": 50.0,
    "moderate_paco2_mmHg": 45.0,
}
HEURISTIC_ACTION_FRACTIONS = {
    "oral_carbs": 0.20,
    "saline": 0.10,
    "oxygen": 0.25,
    "ventilation_pressure_assist": 0.20,
}
HEURISTIC_NORMALIZED_CO2_TREND_DELTA = 0.02

Policy = Callable[[np.ndarray], np.ndarray]


def canonical_json(value) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json_safe(value):
    return json.loads(canonical_json(value))


def _history_observation_contract(
    env: ObservationHistoryWrapper,
) -> dict:
    base_env = env.env
    base_space = base_env.observation_space
    contract = {
        "profile": base_env.observation_profile,
        "base_names": list(env.base_observation_names),
        "base_shape": list(base_space.shape),
        "base_dtype": np.dtype(base_space.dtype).name,
        "base_low": np.asarray(base_space.low).tolist(),
        "base_high": np.asarray(base_space.high).tolist(),
        "base_normalization": {
            "transform": "tanh((raw-center)/scale)",
            "centers": np.asarray(base_env._obs_center).tolist(),
            "scales": np.asarray(base_env._obs_scale).tolist(),
        },
        "history_length": env.history_length,
        "history_order": "oldest_to_newest",
        "padding": "zero",
        "valid_history_mask": "one scalar per history slot, appended after values",
        "stacked_names": list(env.observation_names),
        "stacked_shape": list(env.observation_space.shape),
        "stacked_dtype": np.dtype(env.observation_space.dtype).name,
        "latest_observation_slice": [
            env.latest_observation_slice.start,
            env.latest_observation_slice.stop,
        ],
        "valid_history_mask_slice": [
            env.valid_history_mask_slice.start,
            env.valid_history_mask_slice.stop,
        ],
    }
    contract["sha256"] = canonical_sha256(contract)
    return contract


def _action_contract(env: ObservationHistoryWrapper) -> dict:
    action_space = env.action_space
    contract = {
        "names": list(ACTION_NAMES),
        "shape": list(action_space.shape),
        "dtype": np.dtype(action_space.dtype).name,
        "low": np.asarray(action_space.low).tolist(),
        "high": np.asarray(action_space.high).tolist(),
        "semantics": {
            name: _json_safe(ACTION_SEMANTICS[name]) for name in ACTION_NAMES
        },
        "agent_step_min": float(env.env.config.agent_step_min),
    }
    contract["sha256"] = canonical_sha256(contract)
    return contract


def _normalized_threshold(env: ObservationHistoryWrapper, name: str, raw: float) -> float:
    index = env.base_observation_names.index(name)
    center = float(env.env._obs_center[index])
    scale = float(env.env._obs_scale[index])
    return float(np.tanh((float(raw) - center) / scale))


def make_no_op_policy(env: ObservationHistoryWrapper) -> Policy:
    shape = env.action_space.shape

    def policy(observation: np.ndarray) -> np.ndarray:
        del observation
        return np.zeros(shape, dtype=np.float32)

    return policy


def make_observable_heuristic(env: ObservationHistoryWrapper) -> Policy:
    """Build a fixed rule using only normalized public observations and history."""

    names = env.base_observation_names
    indices = {name: names.index(name) for name in (
        "sensor_glucose_mg_dl",
        "map_mmHg",
        "spo2_pct",
        "paco2_mmHg",
    )}
    action_indices = {name: ACTION_NAMES.index(name) for name in (
        "oral_carbs",
        "saline",
        "oxygen",
        "ventilation_pressure_assist",
    )}
    thresholds = {
        "low_glucose": _normalized_threshold(
            env, "sensor_glucose_mg_dl", HEURISTIC_RAW_THRESHOLDS["low_glucose_mg_dl"]
        ),
        "low_map": _normalized_threshold(
            env, "map_mmHg", HEURISTIC_RAW_THRESHOLDS["low_map_mmHg"]
        ),
        "low_spo2": _normalized_threshold(
            env, "spo2_pct", HEURISTIC_RAW_THRESHOLDS["low_spo2_pct"]
        ),
        "high_paco2": _normalized_threshold(
            env, "paco2_mmHg", HEURISTIC_RAW_THRESHOLDS["high_paco2_mmHg"]
        ),
        "moderate_paco2": _normalized_threshold(
            env,
            "paco2_mmHg",
            HEURISTIC_RAW_THRESHOLDS["moderate_paco2_mmHg"],
        ),
    }
    width = env.base_observation_size
    history_length = env.history_length
    mask_slice = env.valid_history_mask_slice

    def policy(observation: np.ndarray) -> np.ndarray:
        stacked = np.asarray(observation, dtype=np.float32)
        expected = env.observation_space.shape
        if stacked.shape != expected or not np.all(np.isfinite(stacked)):
            raise ValueError("heuristic received an invalid stacked observation")
        latest = stacked[env.latest_observation_slice]
        valid = stacked[mask_slice]
        previous = None
        if history_length > 1 and valid[-2] == 1.0:
            previous_start = (history_length - 2) * width
            previous = stacked[previous_start:previous_start + width]

        action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        if latest[indices["sensor_glucose_mg_dl"]] < thresholds["low_glucose"]:
            action[action_indices["oral_carbs"]] = HEURISTIC_ACTION_FRACTIONS[
                "oral_carbs"
            ]
        if latest[indices["map_mmHg"]] < thresholds["low_map"]:
            action[action_indices["saline"]] = HEURISTIC_ACTION_FRACTIONS["saline"]
        if latest[indices["spo2_pct"]] < thresholds["low_spo2"]:
            action[action_indices["oxygen"]] = HEURISTIC_ACTION_FRACTIONS["oxygen"]

        rising_co2 = (
            previous is not None
            and latest[indices["paco2_mmHg"]]
            > previous[indices["paco2_mmHg"]]
            + HEURISTIC_NORMALIZED_CO2_TREND_DELTA
        )
        if (
            latest[indices["paco2_mmHg"]] > thresholds["high_paco2"]
            or (
                rising_co2
                and latest[indices["paco2_mmHg"]] > thresholds["moderate_paco2"]
            )
        ):
            action[action_indices["ventilation_pressure_assist"]] = (
                HEURISTIC_ACTION_FRACTIONS["ventilation_pressure_assist"]
            )
        return action

    return policy


POLICY_FACTORIES: Mapping[str, Callable[[ObservationHistoryWrapper], Policy]] = {
    "no_op": make_no_op_policy,
    "observable_heuristic": make_observable_heuristic,
}

POLICY_DESCRIPTIONS = {
    "no_op": "All eight native actions remain zero.",
    "observable_heuristic": (
        "Fixed engineering smoke rule over public normalized glucose, MAP, SpO2, "
        "PaCO2 and the previous valid PaCO2 observation; it is not fitted or learned."
    ),
}

POLICY_CONTRACTS = {
    "no_op": {
        "kind": "deterministic_fixed_action",
        "native_action": {name: 0.0 for name in ACTION_NAMES},
    },
    "observable_heuristic": {
        "kind": "deterministic_observation_history_rule",
        "inputs": [
            "latest sensor_glucose_mg_dl",
            "latest map_mmHg",
            "latest spo2_pct",
            "latest paco2_mmHg",
            "previous paco2_mmHg when its history mask is valid",
        ],
        "raw_thresholds": HEURISTIC_RAW_THRESHOLDS,
        "normalized_co2_trend_delta": HEURISTIC_NORMALIZED_CO2_TREND_DELTA,
        "native_action_fractions": HEURISTIC_ACTION_FRACTIONS,
        "rule_combination": "independent rules set named native actions; otherwise zero",
        "fitted": False,
    },
}


def _make_environment(
    config: HumanConfig,
    scenario: str,
    history_length: int,
) -> ObservationHistoryWrapper:
    base = HumanHomeostasisEnv(
        config=config,
        scenario=scenario,
        observation_profile="clinical",
        measurement_profile="realistic",
        info_profile="benchmark",
    )
    return ObservationHistoryWrapper(base, history_length=history_length)


def _evaluate_policy(
    name: str,
    *,
    config: HumanConfig,
    scenarios: Sequence[str],
    seeds: Sequence[int],
    history_length: int,
) -> dict:
    episodes = []
    for scenario in scenarios:
        for seed in seeds:
            env = _make_environment(config, scenario, history_length)
            observation, info = env.reset(seed=int(seed))
            policy = POLICY_FACTORIES[name](env)
            total_return = 0.0
            steps = 0
            nonzero_action_steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = np.asarray(policy(observation), dtype=np.float32)
                if action.shape != env.action_space.shape or not env.action_space.contains(action):
                    raise ValueError(f"policy {name!r} produced an invalid action")
                nonzero_action_steps += int(np.any(action != 0.0))
                observation, reward, terminated, truncated, info = env.step(action)
                total_return += float(reward)
                steps += 1

            reason = info.get("termination_reason")
            if truncated and reason is None:
                reason = "episode_horizon"
            elif terminated and reason is None:
                reason = "terminated_unspecified"
            episodes.append({
                "scenario": str(scenario),
                "seed": int(seed),
                "return": float(total_return),
                "steps": int(steps),
                "duration_min": float(env.env.elapsed_minutes),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": reason,
                "nonzero_action_steps": int(nonzero_action_steps),
            })
            env.close()

    returns = np.asarray([episode["return"] for episode in episodes], dtype=float)
    reason_counts: dict[str, int] = {}
    for episode in episodes:
        reason = str(episode["termination_reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "description": POLICY_DESCRIPTIONS[name],
        "contract": _json_safe(POLICY_CONTRACTS[name]),
        "contract_sha256": canonical_sha256(POLICY_CONTRACTS[name]),
        "episodes": episodes,
        "summary": {
            "episode_count": len(episodes),
            "mean_return": float(np.mean(returns)),
            "sd_return": float(np.std(returns)),
            "termination_reason_counts": reason_counts,
        },
    }


def build_benchmark_result(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    history_length: int = DEFAULT_HISTORY_LENGTH,
    episode_minutes: float = DEFAULT_EPISODE_MINUTES,
    config: HumanConfig | None = None,
) -> dict:
    seeds = tuple(seeds)
    scenarios = tuple(scenarios)
    if any(isinstance(seed, bool) or not isinstance(seed, Integral) for seed in seeds):
        raise ValueError("seeds must contain only integers")
    if any(not isinstance(scenario, str) or not scenario for scenario in scenarios):
        raise ValueError("scenarios must contain only non-empty strings")
    seeds = tuple(int(seed) for seed in seeds)
    scenarios = tuple(str(scenario) for scenario in scenarios)
    if len(seeds) < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty sequence of unique integers")
    if len(scenarios) < 1 or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be a non-empty sequence of unique names")
    if not np.isfinite(float(episode_minutes)) or float(episode_minutes) <= 0.0:
        raise ValueError("episode_minutes must be finite and positive")

    benchmark_config = replace(
        config or HumanConfig(), episode_minutes=float(episode_minutes)
    )
    probe = _make_environment(benchmark_config, scenarios[0], history_length)
    observation_contract = _history_observation_contract(probe)
    action_contract = _action_contract(probe)
    profiles = {
        "observation": probe.env.observation_profile,
        "measurement": probe.env.measurement_profile,
        "info": probe.env.info_profile,
        "reward": probe.env.reward_profile,
    }
    measurement_config = (
        None
        if probe.env.measurement_model is None
        else _json_safe(asdict(probe.env.measurement_model.config))
    )
    rng_contract = {
        "root_seeds": list(seeds),
        "episode_root_seed": "the recorded episode seed is passed unchanged to env.reset",
        "seed_sequence": "numpy.random.SeedSequence(root_seed).spawn(2)",
        "child_stream_order": ["physiology", "measurement"],
        "child_spawn_keys": {
            "physiology": [0],
            "measurement": [1],
        },
        "bit_generator": type(probe.env._physiology_rng.bit_generator).__name__,
        "action_space_seed": "root_seed + 1",
        "seed_none_used": False,
        "policy_randomness": None,
    }
    probe.close()

    policy_results = {
        name: _evaluate_policy(
            name,
            config=benchmark_config,
            scenarios=scenarios,
            seeds=seeds,
            history_length=history_length,
        )
        for name in POLICY_FACTORIES
    }
    evaluation_contract = {
        "seeds": list(seeds),
        "held_out_scenarios": list(scenarios),
        "scenario_role": "evaluation_only_no_fitting",
        "episode_minutes": float(episode_minutes),
        "profiles": profiles,
        "measurement_config": measurement_config,
        "rng": rng_contract,
        "config": _json_safe(asdict(benchmark_config)),
        "observation_contract_sha256": observation_contract["sha256"],
        "action_contract_sha256": action_contract["sha256"],
        "policy_contracts": POLICY_CONTRACTS,
    }
    result = {
        "schema": BENCHMARK_SCHEMA,
        "openhumsim_version": __version__,
        "purpose": "deterministic evaluation harness for transparent baseline policies",
        "training": {
            "performed": False,
            "learned_policy": None,
            "training_results_reported": False,
        },
        "clinical_claims": False,
        "profiles": profiles,
        "measurement_config": measurement_config,
        "rng": rng_contract,
        "config": _json_safe(asdict(benchmark_config)),
        "observation_contract": observation_contract,
        "action_contract": action_contract,
        "evaluation": {
            "seeds": list(seeds),
            "held_out_scenarios": list(scenarios),
            "scenario_role": "evaluation_only_no_fitting",
            "episode_minutes": float(episode_minutes),
            "contract_sha256": canonical_sha256(evaluation_contract),
        },
        "policies": policy_results,
        "interpretation": (
            "Returns compare two transparent software baselines under the recorded "
            "contract. They do not establish clinical validity, safety, efficacy, "
            "optimality, or generalization beyond these seeds and scenarios."
        ),
    }
    return _json_safe(result)


def write_benchmark_result(path: str | Path, result: Mapping) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            dict(result),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def run_benchmark(
    output_path: str | Path | None = None,
    **kwargs,
) -> dict:
    result = build_benchmark_result(**kwargs)
    if output_path is not None:
        write_benchmark_result(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-length", type=int, default=DEFAULT_HISTORY_LENGTH)
    parser.add_argument("--episode-minutes", type=float, default=DEFAULT_EPISODE_MINUTES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    args = parser.parse_args()
    result = run_benchmark(
        args.output,
        seeds=args.seeds,
        scenarios=args.scenarios,
        history_length=args.history_length,
        episode_minutes=args.episode_minutes,
    )
    print(json.dumps({
        name: policy["summary"]
        for name, policy in result["policies"].items()
    }, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
