from __future__ import annotations

from dataclasses import asdict
import builtins
import importlib.util
import json
from math import inf, nan
from pathlib import Path

import pytest

from openhumsim_rl import HumanConfig, __version__
from openhumsim_rl.env import CLINICAL_OBSERVATION_NAMES


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAINING_EXAMPLE = REPOSITORY_ROOT / "examples" / "train_ppo.py"


def _load_training_example():
    spec = importlib.util.spec_from_file_location(
        "openhumsim_train_ppo_manifest_test",
        TRAINING_EXAMPLE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_action_scales_reject_negative_and_nonfinite_values() -> None:
    action_scales = (
        "max_insulin_model_units_per_step",
        "max_carbs_g_per_step",
        "max_saline_ml_per_step",
        "max_fio2",
        "max_ventilation_support_l_min",
        "max_ventilation_pressure_assist_cmH2O",
        "max_oral_water_ml_per_step",
        "max_probe_drug_mg_per_step",
    )
    for name in action_scales:
        for invalid in (-1.0, nan, inf, -inf):
            with pytest.raises(ValueError):
                HumanConfig(**{name: invalid})

    # A disabled actuator is a valid boundary for all scales except max_fio2,
    # whose lower bound is constrained by baseline_fio2.
    for name in set(action_scales) - {"max_fio2"}:
        HumanConfig(**{name: 0.0})


def test_solver_iteration_limits_require_actual_positive_integers() -> None:
    for name in ("acid_base_max_iterations", "co2_pool_solver_max_iterations"):
        for invalid in (0, -1, 1.5, 3.0, nan, inf, -inf, True):
            with pytest.raises(ValueError, match=name):
                HumanConfig(**{name: invalid})
        HumanConfig(**{name: 1})


def test_structural_gas_parameters_are_finite_and_have_safe_signs() -> None:
    positive = (
        "pulmonary_capillary_blood_volume_ml",
        "co2_solubility_mmol_l_mmHg",
        "hemoglobin_monomer_mw_g_mmol",
        "co2_exchangeable_volume_fraction_tbw",
        "co2_gas_molar_volume_l_per_mol_stpd",
        "respiratory_tau_min",
        "gas_exchange_tau_min",
        "hemoglobin_o2_capacity_ml_g",
        "o2_standard_pco2_mmHg",
    )
    for name in positive:
        for invalid in (0.0, -1.0, nan, inf, -inf):
            with pytest.raises(ValueError, match=name):
                HumanConfig(**{name: invalid})

    nonnegative = (
        "dead_space_l",
        "pulmonary_o2_equilibration_tau_s",
        "dissolved_o2_coeff_ml_dl_mmHg",
        "oxygen_supply_transition_width_fraction",
    )
    for name in nonnegative:
        for invalid in (-1.0, nan, inf, -inf):
            with pytest.raises(ValueError, match=name):
                HumanConfig(**{name: invalid})
        HumanConfig(**{name: 0.0})

    for invalid in (-1.0, nan, inf, -inf, 760.0):
        with pytest.raises(ValueError, match="water vapor pressure"):
            HumanConfig(water_vapor_pressure_mmHg=invalid)


def test_v023_manifest_is_deterministic_exact_and_sb3_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def reject_sb3_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "stable_baselines3" or name.startswith("stable_baselines3."):
            raise AssertionError("manifest generation imported Stable-Baselines3")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_sb3_import)
    training = _load_training_example()

    config = HumanConfig(agent_step_min=2.5)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    training.write_training_manifest(first, config=config)
    training.write_training_manifest(second, config=config)

    assert first.read_bytes() == second.read_bytes()
    manifest = json.loads(first.read_text(encoding="utf-8"))
    assert manifest["checkpoint_basename"] == "openhumsim_ppo_v0232_smoke"
    assert manifest["checkpoint_filename"] == "openhumsim_ppo_v0232_smoke.zip"
    assert manifest["openhumsim_version"] == __version__ == "0.23.2"
    assert manifest["state_schema_version"] == "0.22"
    assert manifest["reward_profile"] == "observable_benchmark_v0.23"
    assert manifest["scenario"] == "oral_glucose_75g"
    assert manifest["observation_profile"] == "clinical"
    assert manifest["measurement_profile"] == "realistic"
    assert manifest["info_profile"] == "benchmark"
    assert tuple(manifest["observation_names"]) == CLINICAL_OBSERVATION_NAMES
    assert manifest["observation_names_sha256"] == training.observation_names_sha256(
        CLINICAL_OBSERVATION_NAMES
    )
    assert manifest["observation_names_hash_format"] == (
        "sha256:canonical-json-array:utf-8"
    )
    assert manifest["config"] == json.loads(json.dumps(asdict(config)))
