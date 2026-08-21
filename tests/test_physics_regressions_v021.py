from __future__ import annotations

from dataclasses import replace
from math import exp

import numpy as np

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv


ZERO = np.zeros(8, dtype=np.float32)


def _rollout(
    scenario: str,
    *,
    config: HumanConfig | None = None,
    steps: int = 2,
    seed: int = 42,
):
    env = HumanHomeostasisEnv(config=config, scenario=scenario)
    env.reset(seed=seed)
    info = None
    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(ZERO)
        assert not terminated and not truncated
    assert info is not None
    return env, info["state"]


def test_incomplete_outer_interval_does_not_become_a_breath():
    env = HumanHomeostasisEnv()
    env.reset(seed=2101)
    state = env.state
    cycle = env.model.respiratory_cycle

    # Start at a known breath boundary.  At the resting rate, 1.5 s is shorter
    # than a breath and therefore cannot define a new VT or work-per-breath.
    cycle.initialize_state(state)
    previous_vt = state.tidal_volume_l
    fragment = cycle.step(state, dt_min=0.025)

    assert state.respiratory_cycle_phase_s < 60.0 / state.respiratory_rate_bpm
    assert state.tidal_volume_l == previous_vt
    assert state.respiratory_cycle_total_work_j_breath == 0.0
    assert cycle.last_trace == {}
    assert fragment.end_expiratory_volume_above_relaxed_l == 0.0

    # Continued calls complete the same persisted breath. Diagnostics are then
    # based on approximately one physiological period, not the final fragment.
    for _ in range(3):
        cycle.step(state, dt_min=0.025)
    trace_t = cycle.last_trace["time_s"]
    assert state.respiratory_cycle_total_work_j_breath > 0.0
    assert len(trace_t) > 100
    assert trace_t[-1] > 0.90 * (60.0 / state.respiratory_rate_bpm)

    # A sudden shorter period must not silently modulo the old phase while
    # retaining old samples, which would glue two breaths into one diagnostic.
    cycle.initialize_state(state)
    state.respiratory_rate_bpm = 12.0
    cycle.step(state, dt_min=0.075)  # 4.5 s of a 5 s breath
    state.respiratory_rate_bpm = 30.0
    cycle.step(state, dt_min=0.040)  # enough for one new 2 s breath
    assert 1.8 < cycle.last_trace["time_s"][-1] < 2.1


def test_outer_integration_step_convergence_for_coupled_cardiorespiratory_state():
    base = HumanConfig()
    _, coarse = _rollout(
        "oral_glucose_75g",
        config=replace(base, integration_step_min=0.25),
    )
    _, fine = _rollout(
        "oral_glucose_75g",
        config=replace(base, integration_step_min=0.05),
    )

    assert abs(coarse["tidal_volume_l"] - fine["tidal_volume_l"]) < 0.01
    assert abs(coarse["paco2_mmHg"] - fine["paco2_mmHg"]) < 0.25
    assert abs(coarse["pao2_mmHg"] - fine["pao2_mmHg"]) < 0.25
    assert abs(coarse["map_mmHg"] - fine["map_mmHg"]) < 0.5
    assert abs(
        coarse["cardiac_output_l_min"] - fine["cardiac_output_l_min"]
    ) < 0.10


def test_hpv_fixed_point_advances_kinetics_only_once():
    env = HumanHomeostasisEnv()
    env.reset(seed=2102)
    state = env.state
    state.pulmonary_hpv_function_fraction = 1.0
    for i in range(6):
        setattr(state, f"pulmonary_hpv_tone_u{i}", 0.0)

    dt_min = 0.25
    env.model.pulmonary_exchange.estimate_arterial_oxygen(
        state,
        pco2_mmHg=80.0,
        fio2=0.15,
        exercise=0.0,
        dt_min=dt_min,
        apply=False,
    )

    tones = np.asarray(
        [getattr(state, f"pulmonary_hpv_tone_u{i}") for i in range(6)]
    )
    one_step_alpha = 1.0 - exp(
        -dt_min / env.config.pulmonary_hpv_tau_min
    )
    # Activation is bounded by one, so a single kinetic update from zero cannot
    # exceed alpha. Two updates inside the algebraic fixed point would do so in
    # this near-maximal hypoxic challenge.
    assert np.all(tones <= one_step_alpha + 1e-12)
    assert np.min(tones) > 0.95 * one_step_alpha


def test_mechanical_power_includes_dynamic_resistive_load():
    _, baseline = _rollout("baseline")
    _, obstruction = _rollout("airway_obstruction")

    assert (
        obstruction["respiratory_cycle_resistive_work_j_breath"]
        > baseline["respiratory_cycle_resistive_work_j_breath"]
    )
    assert (
        obstruction["pulmonary_mechanical_power_j_min"]
        > 1.25 * baseline["pulmonary_mechanical_power_j_min"]
    )
    expected_baseline_power = (
        baseline["respiratory_cycle_total_work_j_breath"]
        * baseline["respiratory_rate_bpm"]
    )
    assert abs(
        baseline["pulmonary_mechanical_power_j_min"]
        - expected_baseline_power
    ) < 1e-10


def test_coarse_configured_cv_step_is_stable_and_close_to_fine_reference():
    base = HumanConfig()
    _, coarse = _rollout(
        "baseline",
        config=replace(base, cv_internal_step_s=0.04),
        steps=1,
        seed=2103,
    )
    _, fine = _rollout(
        "baseline",
        config=replace(base, cv_internal_step_s=0.005),
        steps=1,
        seed=2103,
    )

    assert coarse["cv_numerical_volume_correction_ml"] == 0.0
    assert abs(coarse["cv_blood_volume_error_ml"]) < 1e-6
    assert abs(coarse["map_mmHg"] - fine["map_mmHg"]) < 0.5
    assert abs(
        coarse["cardiac_output_l_min"] - fine["cardiac_output_l_min"]
    ) < 0.10
    assert abs(
        coarse["cv_ejection_fraction"] - fine["cv_ejection_fraction"]
    ) < 0.02
