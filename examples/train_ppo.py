"""Optional Stable-Baselines3 smoke example.

Install:
    pip install -e ".[rl]"

The native one-sided [0,1] action space is used because all eight actuators are
nonnegative physical interventions and 0 is the natural no-intervention point.
The benchmark interface is deliberately strict. This feed-forward MLP is only a
smoke baseline for a POMDP; serious comparisons should use observation history
or a recurrent/belief-state policy.

The checkpoint is accompanied by a deterministic JSON manifest.  Manifest
construction deliberately has no Stable-Baselines3 dependency, so it can be
used by evaluation and provenance tooling without the optional ``rl`` extra.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Sequence

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv, __version__
from openhumsim_rl.env import CLINICAL_OBSERVATION_NAMES


CHECKPOINT_BASENAME = "openhumsim_ppo_v022_smoke"
MANIFEST_FILENAME = f"{CHECKPOINT_BASENAME}.manifest.json"
STATE_SCHEMA_VERSION = "0.22"
REWARD_PROFILE = "homeostasis_v0.21"
SCENARIO = "oral_glucose_75g"
OBSERVATION_PROFILE = "clinical"
MEASUREMENT_PROFILE = "realistic"
INFO_PROFILE = "benchmark"


def _canonical_json(value: Any) -> str:
    """Return the stable JSON representation used by manifest hashes."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def observation_names_sha256(observation_names: Sequence[str]) -> str:
    """Hash the ordered observation contract, not an unordered set of names."""

    ordered_names = list(observation_names)
    return sha256(_canonical_json(ordered_names).encode("utf-8")).hexdigest()


def build_training_manifest(
    *,
    config: HumanConfig | None = None,
    observation_names: Sequence[str] = CLINICAL_OBSERVATION_NAMES,
) -> dict[str, Any]:
    """Build reproducible checkpoint metadata without importing SB3."""

    effective_config = HumanConfig() if config is None else config
    ordered_names = list(observation_names)
    return {
        "manifest_schema_version": 1,
        "checkpoint_basename": CHECKPOINT_BASENAME,
        "checkpoint_filename": f"{CHECKPOINT_BASENAME}.zip",
        "openhumsim_version": __version__,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "reward_profile": REWARD_PROFILE,
        "scenario": SCENARIO,
        "observation_profile": OBSERVATION_PROFILE,
        "measurement_profile": MEASUREMENT_PROFILE,
        "info_profile": INFO_PROFILE,
        "observation_names": ordered_names,
        "observation_names_sha256": observation_names_sha256(ordered_names),
        "observation_names_hash_format": "sha256:canonical-json-array:utf-8",
        "config": asdict(effective_config),
    }


def write_training_manifest(
    path: str | Path = MANIFEST_FILENAME,
    *,
    config: HumanConfig | None = None,
    observation_names: Sequence[str] = CLINICAL_OBSERVATION_NAMES,
) -> Path:
    """Write a byte-for-byte deterministic training manifest."""

    output_path = Path(path)
    manifest = build_training_manifest(
        config=config,
        observation_names=observation_names,
    )
    output_path.write_text(
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    # Keep the optional dependency local: importing this module to generate or
    # inspect a manifest must work without Stable-Baselines3 installed.
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    config = HumanConfig()
    env = HumanHomeostasisEnv(
        config=config,
        scenario=SCENARIO,
        observation_profile=OBSERVATION_PROFILE,
        measurement_profile=MEASUREMENT_PROFILE,
        info_profile=INFO_PROFILE,
    )
    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=50_000)
    model.save(CHECKPOINT_BASENAME)
    write_training_manifest(config=config, observation_names=env.observation_names)


if __name__ == "__main__":
    main()
