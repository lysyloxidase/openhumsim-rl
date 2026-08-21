from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from openhumsim_rl.calibration import reference_outputs
from openhumsim_rl.cgm import CGMObservationConfig, blood_to_cgm_trace
from openhumsim_rl.config import HumanConfig
from openhumsim_rl.env import (
    ACTION_NAMES,
    CLINICAL_OBSERVATION_NAMES,
    HumanHomeostasisEnv,
)
from openhumsim_rl.measurement import (
    ClinicalMeasurementConfig,
    ClinicalMeasurementModel,
)
from openhumsim_rl.physiology import HumanState
from openhumsim_rl.population import DEFAULT_PARAMETER_SPECS, LockedCohortManifest


ZERO_ACTION = np.zeros(len(ACTION_NAMES), dtype=np.float32)


def test_benchmark_info_uses_an_exact_nonleaking_allowlist():
    env = HumanHomeostasisEnv(info_profile="benchmark")
    _, reset_info = env.reset(seed=21)

    expected_reset = {
        "observation_names",
        "observation_profile",
        "measurement_profile",
        "info_profile",
        "action_names",
        "gymnasium_installed",
        "environment_semantics",
    }
    assert set(reset_info) == expected_reset
    forbidden = {
        "state",
        "reward_terms",
        "scenario",
        "scenario_warning",
        "action",
        "intervention",
        "measurement",
        "time_min",
        "blood_gas",
        "blood_gas_carbon",
        "oxygen_transport",
        "mass_balance",
    }
    assert forbidden.isdisjoint(reset_info)

    _, _, terminated, truncated, step_info = env.step(ZERO_ACTION)
    assert not terminated and not truncated
    assert set(step_info) == expected_reset
    assert forbidden.isdisjoint(step_info)

    def terminate(state, intervention, duration_min):
        state.glucose_mg_dl = env.config.glucose_max_terminate + 1.0
        return state

    env.model.integrate = terminate
    _, _, terminated, _, terminal_info = env.step(ZERO_ACTION)
    assert terminated
    assert set(terminal_info) == expected_reset | {"termination_reason"}
    assert terminal_info["termination_reason"] == "extreme_hyperglycemia"
    assert forbidden.isdisjoint(terminal_info)


def test_nondivisible_horizon_does_not_overshoot_or_change_bolus_contract():
    cfg = HumanConfig(agent_step_min=5.0, integration_step_min=0.25, episode_minutes=12.0)
    env = HumanHomeostasisEnv(config=cfg, observation_profile="full")
    obs, _ = env.reset(seed=22)
    time_index = env.observation_names.index("time_to_go_fraction")
    time_observations = [float(obs[time_index])]

    times = []
    for _ in range(3):
        obs, _, terminated, truncated, _ = env.step(ZERO_ACTION)
        assert not terminated
        times.append(env.elapsed_minutes)
        time_observations.append(float(obs[time_index]))
    assert times == pytest.approx([5.0, 10.0, 12.0])
    assert truncated

    full_action = np.ones(len(ACTION_NAMES), dtype=np.float32)
    assert all(a > b for a, b in zip(time_observations, time_observations[1:]))

    decoded = env._decode_action(full_action)
    assert decoded.insulin_model_units == pytest.approx(cfg.max_insulin_model_units_per_step)
    assert decoded.oral_carbs_g == pytest.approx(cfg.max_carbs_g_per_step)
    assert decoded.saline_ml == pytest.approx(cfg.max_saline_ml_per_step)
    assert decoded.oral_water_ml == pytest.approx(cfg.max_oral_water_ml_per_step)
    assert decoded.oral_probe_mg == pytest.approx(cfg.max_probe_drug_mg_per_step)

    short = HumanHomeostasisEnv(
        config=replace(cfg, episode_minutes=2.0),
        observation_profile="full",
    )
    short.reset(seed=220)
    _, _, _, short_truncated, short_info = short.step(full_action)
    assert short_truncated
    assert short_info["intervention"]["insulin_model_units"] == pytest.approx(
        cfg.max_insulin_model_units_per_step
    )
    assert short_info["intervention"]["oral_carbs_g"] == pytest.approx(
        cfg.max_carbs_g_per_step
    )

    with pytest.raises(RuntimeError, match="call reset"):
        env.step(ZERO_ACTION)

    def baseline_return(agent_step_min: float) -> float:
        candidate = HumanHomeostasisEnv(
            config=HumanConfig(
                agent_step_min=agent_step_min,
                integration_step_min=0.25,
                episode_minutes=10.0,
            ),
            observation_profile="full",
        )
        candidate.reset(seed=221)
        total = 0.0
        while True:
            _, reward, term, trunc, _ = candidate.step(ZERO_ACTION)
            total += reward
            if term or trunc:
                return total

    assert baseline_return(5.0) == pytest.approx(
        baseline_return(2.5), abs=5e-4
    )


def test_terminal_threshold_is_checked_at_integration_cadence():
    cfg = HumanConfig(agent_step_min=5.0, integration_step_min=0.25)
    env = HumanHomeostasisEnv(config=cfg, observation_profile="full")
    env.reset(seed=23)
    calls: list[float] = []

    def cross_threshold(state, intervention, duration_min):
        calls.append(float(duration_min))
        state.glucose_mg_dl = cfg.glucose_max_terminate + 1.0
        return state

    env.model.integrate = cross_threshold
    obs, reward, terminated, truncated, info = env.step(ZERO_ACTION)

    assert terminated and not truncated
    assert calls == pytest.approx([cfg.integration_step_min])
    assert env.elapsed_minutes == pytest.approx(cfg.integration_step_min)
    assert info["termination_reason"] == "extreme_hyperglycemia"
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)


def test_nonfinite_state_is_detected_before_threshold_comparisons():
    env = HumanHomeostasisEnv(observation_profile="full")
    env.reset(seed=24)
    env.state.glucose_mg_dl = float("nan")
    assert env._terminated() == (True, "numerical_failure_nonfinite_state")
    with pytest.raises(FloatingPointError, match="Observation contains"):
        env._get_obs()


def test_integration_nonfinite_failure_restores_a_finite_terminal_observation():
    env = HumanHomeostasisEnv(observation_profile="full")
    env.reset(seed=240)
    glucose_before = env.state.glucose_mg_dl
    runtime_before = env.model.runtime_snapshot()
    integrate = env.model.integrate

    def fail_nonfinite(state, intervention, duration_min):
        integrate(state, intervention, duration_min)
        state.glucose_mg_dl = float("nan")
        raise FloatingPointError("synthetic numerical failure")

    env.model.integrate = fail_nonfinite
    obs, reward, terminated, truncated, info = env.step(ZERO_ACTION)

    assert terminated and not truncated
    assert info["termination_reason"] == "numerical_failure_nonfinite_state"
    assert env.elapsed_minutes == pytest.approx(0.0)
    assert env.state.glucose_mg_dl == pytest.approx(glucose_before)
    runtime_after = env.model.runtime_snapshot()
    assert runtime_after["cardiovascular"] == runtime_before["cardiovascular"]
    before_cycle = runtime_before["respiratory_cycle"]
    after_cycle = runtime_after["respiratory_cycle"]
    assert before_cycle[2] == after_cycle[2]
    assert set(before_cycle[0]) == set(after_cycle[0])
    for name in before_cycle[0]:
        np.testing.assert_array_equal(before_cycle[0][name], after_cycle[0][name])
    assert np.all(np.isfinite(obs))
    assert reward == pytest.approx(-10.0)
    with pytest.raises(RuntimeError, match="call reset"):
        env.step(ZERO_ACTION)


def test_delayed_measurements_queue_every_observed_sample_without_backdating():
    state = HumanState()
    state.glucose_mg_dl = 100.0
    state.paco2_mmHg = 40.0
    config = ClinicalMeasurementConfig(
        monitor_dropout_probability=0.0,
        cgm_dropout_probability=0.0,
        abg_interval_min=10.0,
        abg_result_delay_min=25.0,
        cgm_relative_noise_sd=0.0,
        noise_multiplier=0.0,
    )
    model = ClinicalMeasurementModel(config)
    rng = np.random.default_rng(25)
    model.initialize(state, rng)
    assert "pulmonary_aa_gradient_mmHg" not in CLINICAL_OBSERVATION_NAMES
    assert "pulmonary_enghoff_dead_space_fraction" not in CLINICAL_OBSERVATION_NAMES
    with pytest.raises(KeyError):
        model.measurement_value("pulmonary_aa_gradient_mmHg", state)
    initial_paco2 = model.measurement_value("paco2_mmHg", state)
    for time_min, paco2 in ((10.0, 50.0), (20.0, 60.0), (30.0, 70.0)):
        state.paco2_mmHg = paco2
        model.advance(state, time_min, 10.0, rng)

    assert model.measurement_value("paco2_mmHg", state) == initial_paco2
    before = model.diagnostics()["channels"]
    assert before["paco2_mmHg"]["pending"] == 3

    state.paco2_mmHg = 80.0
    model.advance(state, 35.0, 5.0, rng)

    assert model.measurement_value("paco2_mmHg", state) == pytest.approx(50.0)
    after = model.diagnostics()["channels"]
    assert after["paco2_mmHg"]["sample_time_min"] == pytest.approx(10.0)
    assert after["paco2_mmHg"]["pending"] == 2

    with pytest.raises(ValueError, match="cannot be reconstructed"):
        model.advance(state, 100.0, 0.0, rng)


def test_cgm_seed_controls_the_first_noisy_sample():
    values = np.asarray([100.0, 110.0, 120.0, 105.0])
    config = CGMObservationConfig(relative_noise_sd=0.10)

    first = blood_to_cgm_trace(values, 5.0, config=config, seed=26)
    repeated = blood_to_cgm_trace(values, 5.0, config=config, seed=26)
    other_seed = blood_to_cgm_trace(values, 5.0, config=config, seed=27)

    np.testing.assert_array_equal(first, repeated)
    assert first[0] != other_seed[0]


def test_manifest_lock_covers_calibration_ids_validation_ids_and_split_seed():
    manifest = LockedCohortManifest.create(
        [f"SUBJ-{i:03d}" for i in range(20)],
        "external-cohort-v021",
        seed=2021,
        dataset_fingerprint="sha256:test",
    )
    assert manifest.verify_lock()
    assert not replace(
        manifest,
        calibration_subject_ids=manifest.calibration_subject_ids[::-1],
    ).verify_lock()
    assert not replace(manifest, split_seed=manifest.split_seed + 1).verify_lock()
    assert not replace(
        manifest,
        validation_subject_ids=manifest.validation_subject_ids[::-1],
    ).verify_lock()


def test_virtual_population_varies_an_active_pulmonary_parameter():
    names = {spec.name for spec in DEFAULT_PARAMETER_SPECS}
    assert "baseline_aa_gradient_mmHg" not in names
    assert "pulmonary_baseline_vq_log_sd" in names

    base = HumanConfig()
    low = reference_outputs(replace(base, pulmonary_baseline_vq_log_sd=0.08), seed=123)
    high = reference_outputs(replace(base, pulmonary_baseline_vq_log_sd=0.30), seed=123)
    assert low["pao2_mmHg"] > high["pao2_mmHg"] + 10.0


def test_measurement_configuration_rejects_invalid_time_semantics():
    with pytest.raises(ValueError, match="abg_interval_min"):
        ClinicalMeasurementModel(ClinicalMeasurementConfig(abg_interval_min=0.0))
    with pytest.raises(ValueError, match="abg_result_delay_min"):
        ClinicalMeasurementModel(ClinicalMeasurementConfig(abg_result_delay_min=-1.0))
