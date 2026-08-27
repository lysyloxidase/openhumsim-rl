from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv
from openhumsim_rl.compat import Box as CompatibilityBox
from openhumsim_rl.env import ACTION_NAMES, BENCHMARK_INFO_KEYS
from openhumsim_rl.measurement import ClinicalMeasurementConfig
from openhumsim_rl.snapshot import ENVIRONMENT_SNAPSHOT_SCHEMA


def _measurement_config() -> ClinicalMeasurementConfig:
    return ClinicalMeasurementConfig(
        monitor_dropout_probability=0.15,
        cgm_dropout_probability=0.20,
        cgm_sample_interval_min=5.0,
        abg_interval_min=10.0,
        abg_result_delay_min=7.0,
        abg_dropout_probability=0.25,
        chemistry_interval_min=10.0,
        chemistry_result_delay_min=12.0,
        chemistry_dropout_probability=0.25,
        cgm_relative_noise_sd=0.05,
        noise_multiplier=1.0,
    )


def _environment(
    *,
    config: HumanConfig | None = None,
    info_profile: str = "benchmark",
    measurement_config: ClinicalMeasurementConfig | None = None,
) -> HumanHomeostasisEnv:
    return HumanHomeostasisEnv(
        config=config
        or HumanConfig(
            agent_step_min=5.0,
            integration_step_min=0.25,
            episode_minutes=30.0,
        ),
        scenario="oral_glucose_75g",
        observation_profile="clinical",
        measurement_profile="realistic",
        measurement_config=measurement_config or _measurement_config(),
        info_profile=info_profile,
    )


def _snapshot_after_two_steps() -> tuple[HumanHomeostasisEnv, dict]:
    env = _environment()
    env.reset(seed=2307)
    first_action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    first_action[2] = 0.35
    first_action[4] = 0.10
    env.step(first_action)
    second_action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    second_action[1] = 0.08
    second_action[5] = 0.12
    env.step(second_action)
    # Move the action-space stream away from its reset state as well.
    env.action_space.sample()
    payload = json.loads(
        json.dumps(env.to_versioned_snapshot(), allow_nan=False)
    )
    return env, payload


def test_json_snapshot_restores_an_identical_stochastic_continuation():
    source, payload = _snapshot_after_two_steps()
    assert payload["schema"] == ENVIRONMENT_SNAPSHOT_SCHEMA
    assert "__openhumsim_ndarray_v1__" in json.dumps(payload)

    restored = _environment()
    restored.restore_versioned_snapshot(payload)
    assert restored.to_versioned_snapshot() == payload
    np.testing.assert_array_equal(source._get_obs(), restored._get_obs())
    assert source.state.as_dict() == restored.state.as_dict()

    # All four RNG streams resume exactly, including Gymnasium's action space.
    np.testing.assert_array_equal(
        source.action_space.sample(), restored.action_space.sample()
    )
    assert source.np_random.integers(0, 2**31) == restored.np_random.integers(
        0, 2**31
    )
    assert source._physiology_rng.normal() == restored._physiology_rng.normal()
    assert source._measurement_rng.random() == restored._measurement_rng.random()

    action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    action[0] = 0.05
    action[2] = 0.25
    action[4] = 0.15
    source_result = source.step(action)
    restored_result = restored.step(action)

    np.testing.assert_array_equal(source_result[0], restored_result[0])
    assert source_result[1:4] == restored_result[1:4]
    assert source_result[4] == restored_result[4]
    assert set(restored_result[4]) <= set(BENCHMARK_INFO_KEYS)
    assert "state" not in restored_result[4]
    assert source.state.as_dict() == restored.state.as_dict()
    assert source.to_versioned_snapshot() == restored.to_versioned_snapshot()


def test_snapshot_rejects_schema_and_contract_tampering_without_mutation():
    _, payload = _snapshot_after_two_steps()
    target = _environment()
    target.reset(seed=999)
    before = target.to_versioned_snapshot()

    mutations: list[tuple[dict, str]] = []
    bad = deepcopy(payload)
    bad["schema"] = "openhumsim.environment-snapshot.v999"
    mutations.append((bad, "schema"))
    bad = deepcopy(payload)
    bad["package_version"] = "999.0.0"
    mutations.append((bad, "package_version"))
    bad = deepcopy(payload)
    bad["state_schema_version"] = "999"
    mutations.append((bad, "state_schema_version"))
    bad = deepcopy(payload)
    bad["runtime"]["state"]["state_schema_version"] = "999"
    mutations.append((bad, "state_schema_version"))
    bad = deepcopy(payload)
    bad["environment_contract"]["observation_contract"]["names"][0] = (
        "tampered"
    )
    mutations.append((bad, "observation_contract"))
    bad = deepcopy(payload)
    bad["environment_contract"]["action_contract"]["names"][0] = "tampered"
    mutations.append((bad, "action_contract"))
    bad = deepcopy(payload)
    bad["unexpected"] = True
    mutations.append((bad, "fields do not match schema"))

    for tampered, message in mutations:
        with pytest.raises((TypeError, ValueError), match=message):
            target.restore_versioned_snapshot(tampered)
        assert target.to_versioned_snapshot() == before


def test_snapshot_rejects_target_config_measurement_and_profile_mismatches():
    _, payload = _snapshot_after_two_steps()

    changed_config = HumanConfig(
        agent_step_min=2.5,
        integration_step_min=0.25,
        episode_minutes=30.0,
    )
    with pytest.raises(ValueError, match="config"):
        _environment(config=changed_config).restore_versioned_snapshot(payload)

    changed_measurement = _measurement_config()
    changed_measurement = ClinicalMeasurementConfig(
        **{
            **changed_measurement.__dict__,
            "noise_multiplier": 2.0,
        }
    )
    with pytest.raises(ValueError, match="measurement_config"):
        _environment(
            measurement_config=changed_measurement
        ).restore_versioned_snapshot(payload)

    with pytest.raises(ValueError, match="info_profile"):
        _environment(info_profile="debug").restore_versioned_snapshot(payload)


def test_snapshot_rejects_corrupt_measurement_runtime_atomically():
    _, payload = _snapshot_after_two_steps()
    target = _environment()
    target.reset(seed=998)
    before = target.to_versioned_snapshot()
    corruptions: list[dict] = []

    bad = deepcopy(payload)
    del bad["runtime"]["measurement"]["channels"]["heart_rate_bpm"]
    corruptions.append(bad)
    bad = deepcopy(payload)
    bad["runtime"]["measurement"]["channels"]["paco2_mmHg"][
        "dropped_count"
    ] = -1
    corruptions.append(bad)
    bad = deepcopy(payload)
    pending = bad["runtime"]["measurement"]["channels"]["paco2_mmHg"][
        "pending_results"
    ]
    assert pending
    pending[0]["available_time_min"] = bad["runtime"]["elapsed_minutes"]
    corruptions.append(bad)
    bad = deepcopy(payload)
    pending = bad["runtime"]["measurement"]["channels"]["paco2_mmHg"][
        "pending_results"
    ]
    assert pending
    pending[0]["available_time_min"] -= 0.5
    corruptions.append(bad)
    bad = deepcopy(payload)
    pending = bad["runtime"]["measurement"]["channels"]["paco2_mmHg"][
        "pending_results"
    ]
    assert pending
    # This remains locally chronological and preserves the configured result
    # delay, but it is impossible because the other ABG panel members retain the
    # shared collection/result times.
    pending[0]["sample_time_min"] -= 0.5
    pending[0]["available_time_min"] -= 0.5
    corruptions.append(bad)
    bad = deepcopy(payload)
    bad["runtime"]["measurement"]["channels"]["spo2_pct"]["value"] = 101.0
    corruptions.append(bad)
    bad = deepcopy(payload)
    pending = bad["runtime"]["measurement"]["channels"]["pao2_mmHg"][
        "pending_results"
    ]
    assert pending
    pending[0]["value"] = -1.0
    corruptions.append(bad)
    bad = deepcopy(payload)
    bad["runtime"]["measurement"]["cgm_state"][
        "sensor_glucose_mg_dl"
    ] = 10_000.0
    corruptions.append(bad)
    bad = deepcopy(payload)
    bad["runtime"]["measurement"]["cgm_state"][
        "interstitial_glucose_mg_dl"
    ] = -500.0
    corruptions.append(bad)
    bad = deepcopy(payload)
    bad["runtime"]["measurement"]["cgm_next_sample_time_min"] = bad[
        "runtime"
    ]["elapsed_minutes"]
    corruptions.append(bad)
    bad = deepcopy(payload)
    bad["runtime"]["measurement"]["channels"]["heart_rate_bpm"][
        "next_sample_time_min"
    ] = 12.5
    corruptions.append(bad)
    bad = deepcopy(payload)
    bad["runtime"]["measurement"]["channels"]["heart_rate_bpm"][
        "next_sample_time_min"
    ] = 10_000.0
    corruptions.append(bad)
    bad = deepcopy(payload)
    bad["runtime"]["measurement"]["channels"]["heart_rate_bpm"][
        "delivered_count"
    ] = 0
    corruptions.append(bad)
    bad = deepcopy(payload)
    bad["runtime"]["measurement"]["channels"]["heart_rate_bpm"][
        "delivered_count"
    ] += 1
    corruptions.append(bad)
    bad = deepcopy(payload)
    bad["runtime"]["measurement"]["cgm_delivered_count"] += 1
    corruptions.append(bad)
    bad = deepcopy(payload)
    bad["runtime"]["measurement"]["cgm_next_sample_time_min"] = 10_000.0
    corruptions.append(bad)

    for corrupted in corruptions:
        with pytest.raises(ValueError, match="invalid measurement runtime"):
            target.restore_versioned_snapshot(corrupted)
        assert target.to_versioned_snapshot() == before


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("total_body_water_l", -1.0, "must be positive"),
        ("plasma_volume_l", -1.0, "must be nonnegative"),
        ("probe_total_body_mg", -1.0, "must be nonnegative"),
        ("water_lost_l", -1.0, "must be nonnegative"),
        ("cardiac_output_l_min", -1.0, "must be nonnegative"),
        ("probe_plasma_mg_l", -1.0, "must be nonnegative"),
        ("pulmonary_shunt_fraction", 1.01, r"within \[0, 1\]"),
        ("spo2_pct", 100.01, r"within \[0, 100\]"),
        ("respiratory_ventilator_mode_code", 1.5, "mode_code is invalid"),
        ("respiratory_ventilator_active", 0.5, "must be binary"),
        (
            "respiratory_ventilator_triggers_current_effort",
            1.5,
            "must be integer-valued",
        ),
    ],
)
def test_snapshot_rejects_impossible_live_state_atomically(
    field_name: str,
    invalid_value: float,
    message: str,
) -> None:
    _, payload = _snapshot_after_two_steps()
    target = _environment()
    target.reset(seed=997)
    before = target.to_versioned_snapshot()
    corrupted = deepcopy(payload)
    corrupted["runtime"]["state"]["state"][field_name] = invalid_value

    with pytest.raises(ValueError, match=message):
        target.restore_versioned_snapshot(corrupted)
    assert target.to_versioned_snapshot() == before


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("icf_volume_l", 29.0, "ECF and ICF volumes"),
        ("cv_total_blood_volume_ml", 5001.0, "cardiovascular compartments"),
        ("pulmonary_derecruited_fraction", 0.50, "recruitment fractions"),
    ],
)
def test_snapshot_rejects_broken_live_state_relationships_atomically(
    field_name: str,
    invalid_value: float,
    message: str,
) -> None:
    _, payload = _snapshot_after_two_steps()
    target = _environment()
    target.reset(seed=996)
    before = target.to_versioned_snapshot()
    corrupted = deepcopy(payload)
    corrupted["runtime"]["state"]["state"][field_name] = invalid_value

    with pytest.raises(ValueError, match=message):
        target.restore_versioned_snapshot(corrupted)
    assert target.to_versioned_snapshot() == before


def test_snapshot_rejects_broken_mass_and_concentration_identities_atomically() -> None:
    _, payload = _snapshot_after_two_steps()
    target = _environment()
    target.reset(seed=993)
    before = target.to_versioned_snapshot()
    mutations = (
        ("lactate_amount_mmol", 1.0, "lactate amount/concentration"),
        ("lactate_generated_mmol", 1.0, "lactate mass ledger"),
        ("sodium_mmol_l", 1.0, "sodium_mmol_l/ECF pool"),
        ("potassium_mmol_l", 1.0, "potassium_mmol_l/ECF pool"),
        ("chloride_mmol_l", 1.0, "chloride_mmol_l/ECF pool"),
        ("plasma_volume_l", 0.1, "blood composition volume"),
        ("hematocrit_fraction", 0.01, "hematocrit"),
        ("probe_total_body_mg", 1.0, "PBPK compartment total"),
        ("probe_mass_balance_error_mg", 1.0, "PBPK mass-balance residual"),
        ("exchangeable_co2_pool_mmol", 1.0, "CO2 mass ledger"),
        ("co2_mass_balance_error_mmol", 1.0, "CO2 mass-balance residual"),
    )

    for field_name, delta, message in mutations:
        corrupted = deepcopy(payload)
        state = corrupted["runtime"]["state"]["state"]
        state[field_name] += delta
        with pytest.raises(ValueError, match=message):
            target.restore_versioned_snapshot(corrupted)
        assert target.to_versioned_snapshot() == before


def test_snapshot_rejects_all_explicit_conservation_ledgers_atomically() -> None:
    _, payload = _snapshot_after_two_steps()
    target = _environment()
    target.reset(seed=989)
    before = target.to_versioned_snapshot()

    corruptions: list[tuple[dict, str]] = []
    bad = deepcopy(payload)
    state = bad["runtime"]["state"]["state"]
    state["total_body_water_l"] += 1.0
    state["icf_volume_l"] += 1.0
    corruptions.append((bad, "water mass ledger"))

    for pool, concentration, message in (
        ("ecf_sodium_mmol", "sodium_mmol_l", "sodium mass ledger"),
        ("ecf_chloride_mmol", "chloride_mmol_l", "chloride mass ledger"),
    ):
        bad = deepcopy(payload)
        state = bad["runtime"]["state"]["state"]
        state[pool] += 1.0
        state[concentration] = state[pool] / state["ecf_volume_l"]
        corruptions.append((bad, message))

    bad = deepcopy(payload)
    bad["runtime"]["state"]["state"]["icf_potassium_mmol"] += 1.0
    corruptions.append((bad, "potassium mass ledger"))
    bad = deepcopy(payload)
    bad["runtime"]["state"]["state"]["nonvolatile_strong_anion_mEq"] += 1.0
    corruptions.append((bad, "nonvolatile acid mass ledger"))
    bad = deepcopy(payload)
    bad["runtime"]["state"]["state"][
        "sc_insulin_mass_balance_error_model_units"
    ] += 1.0
    corruptions.append((bad, "SC insulin mass-balance residual"))
    bad = deepcopy(payload)
    bad["runtime"]["state"]["state"]["dalla_gi_mass_balance_error_mg"] += 1.0
    corruptions.append((bad, "gastrointestinal glucose mass-balance residual"))

    for corrupted, message in corruptions:
        with pytest.raises(ValueError, match=message):
            target.restore_versioned_snapshot(corrupted)
        assert target.to_versioned_snapshot() == before


@pytest.mark.parametrize("invalid_value", [True, "95.0"])
def test_snapshot_rejects_non_numeric_human_state_values_atomically(
    invalid_value: object,
) -> None:
    _, payload = _snapshot_after_two_steps()
    target = _environment()
    target.reset(seed=992)
    before = target.to_versioned_snapshot()
    corrupted = deepcopy(payload)
    corrupted["runtime"]["state"]["state"]["glucose_mg_dl"] = invalid_value

    with pytest.raises(TypeError, match="glucose_mg_dl.*real number"):
        target.restore_versioned_snapshot(corrupted)
    assert target.to_versioned_snapshot() == before


def test_snapshot_rejects_live_runtime_at_episode_horizon_atomically() -> None:
    _, payload = _snapshot_after_two_steps()
    target = _environment()
    target.reset(seed=991)
    before = target.to_versioned_snapshot()
    corrupted = deepcopy(payload)
    corrupted["runtime"]["elapsed_minutes"] = corrupted["environment_contract"][
        "config"
    ]["episode_minutes"]
    corrupted["runtime"]["needs_reset"] = False

    with pytest.raises(ValueError, match="episode horizon must require reset"):
        target.restore_versioned_snapshot(corrupted)
    assert target.to_versioned_snapshot() == before


def test_snapshot_rejects_terminal_state_marked_as_active_atomically() -> None:
    _, payload = _snapshot_after_two_steps()
    target = _environment()
    target.reset(seed=988)
    before = target.to_versioned_snapshot()
    corrupted = deepcopy(payload)
    corrupted["runtime"]["state"]["state"]["glucose_mg_dl"] = 1.0
    corrupted["runtime"]["needs_reset"] = False

    with pytest.raises(ValueError, match="active contains a terminal state"):
        target.restore_versioned_snapshot(corrupted)
    assert target.to_versioned_snapshot() == before


def test_snapshot_api_refuses_to_emit_an_impossible_live_state() -> None:
    env = _environment()
    env.reset(seed=995)
    env.state.total_body_water_l = -1.0
    with pytest.raises(ValueError, match="total_body_water_l.*positive"):
        env.to_versioned_snapshot()


@pytest.mark.parametrize(
    ("elapsed_minutes", "message"),
    [
        (-1.0, "outside the configured horizon"),
        (31.0, "outside the configured horizon"),
        (float("nan"), "outside the configured horizon"),
        (30.0, "episode horizon must require reset"),
    ],
)
def test_snapshot_api_refuses_to_emit_an_invalid_live_clock(
    elapsed_minutes: float,
    message: str,
) -> None:
    env = _environment()
    env.reset(seed=990)
    env.elapsed_minutes = elapsed_minutes
    env._needs_reset = False
    with pytest.raises(ValueError, match=message):
        env.to_versioned_snapshot()


@pytest.mark.parametrize("invalid_shape", [[1.5], [True]])
def test_snapshot_rejects_noninteger_ndarray_shape_atomically(
    invalid_shape: list[object],
) -> None:
    _, payload = _snapshot_after_two_steps()
    target = _environment()
    target.reset(seed=994)
    before = target.to_versioned_snapshot()
    corrupted = deepcopy(payload)
    corrupted["runtime"]["physiology"]["respiratory_cycle"][0]["dt"][
        "__openhumsim_ndarray_v1__"
    ]["shape"] = invalid_shape

    with pytest.raises(TypeError, match="shape entries must be integers"):
        target.restore_versioned_snapshot(corrupted)
    assert target.to_versioned_snapshot() == before


def test_snapshot_rejects_partial_breath_endpoint_inconsistent_with_state() -> None:
    _, payload = _snapshot_after_two_steps()
    target = _environment()
    target.reset(seed=987)
    before = target.to_versioned_snapshot()
    corrupted = deepcopy(payload)
    encoded_volume = corrupted["runtime"]["physiology"]["respiratory_cycle"][0][
        "volume"
    ]["__openhumsim_ndarray_v1__"]
    assert encoded_volume["data"]
    encoded_volume["data"][-1] += 1.0

    with pytest.raises(ValueError, match="volume endpoint does not match live state"):
        target.restore_versioned_snapshot(corrupted)
    assert target.to_versioned_snapshot() == before


def test_snapshot_restores_fallback_action_space_rng_without_compat_changes():
    source = HumanHomeostasisEnv(observation_profile="full")
    source.action_space = CompatibilityBox(
        low=np.zeros(len(ACTION_NAMES), dtype=np.float32),
        high=np.ones(len(ACTION_NAMES), dtype=np.float32),
        dtype=np.float32,
    )
    source.action_space.seed(2308)
    source.action_space.sample()
    payload = json.loads(json.dumps(source.to_versioned_snapshot()))

    restored = HumanHomeostasisEnv(observation_profile="full")
    restored.action_space = CompatibilityBox(
        low=np.zeros(len(ACTION_NAMES), dtype=np.float32),
        high=np.ones(len(ACTION_NAMES), dtype=np.float32),
        dtype=np.float32,
    )
    restored.restore_versioned_snapshot(payload)
    np.testing.assert_array_equal(
        source.action_space.sample(), restored.action_space.sample()
    )


def test_unreset_snapshot_preserves_reset_requirement_and_rejects_nan_state():
    realistic = HumanHomeostasisEnv()
    zero_action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    with pytest.raises(RuntimeError, match="call reset"):
        realistic.step(zero_action)
    with pytest.raises(RuntimeError, match="reset required"):
        realistic.to_versioned_snapshot()

    source = HumanHomeostasisEnv(observation_profile="full")
    payload = json.loads(
        json.dumps(source.to_versioned_snapshot(), allow_nan=False)
    )
    restored = HumanHomeostasisEnv(observation_profile="full")
    restored.restore_versioned_snapshot(payload)
    assert restored.to_versioned_snapshot() == payload
    with pytest.raises(RuntimeError, match="call reset"):
        restored.step(zero_action)

    source.state.glucose_mg_dl = float("nan")
    with pytest.raises(ValueError, match="finite"):
        source.to_versioned_snapshot()
