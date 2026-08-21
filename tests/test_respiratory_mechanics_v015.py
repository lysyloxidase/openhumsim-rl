from __future__ import annotations

import numpy as np

from openhumsim_rl import HumanHomeostasisEnv


ZERO = np.zeros(8, dtype=np.float32)


def rollout(scenario: str, steps: int = 6, seed: int = 42):
    env = HumanHomeostasisEnv(scenario=scenario)
    _, info = env.reset(seed=seed)
    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(ZERO)
        if terminated or truncated:
            break
    return env, info["state"]


def test_transpulmonary_pressure_identity():
    _, s = rollout("mechanical_ventilation_peep5", steps=2)
    assert abs(s["pulmonary_pressure_identity_residual_cmH2O"]) < 1e-12
    lhs = s["pulmonary_transpulmonary_pressure_end_exp_cmH2O"]
    rhs = 5.0 - s["pulmonary_pleural_pressure_end_exp_cmH2O"]
    assert abs(lhs - rhs) < 1e-10


def test_baseline_mechanics_are_finite_and_not_overdistended():
    _, s = rollout("baseline")
    assert 0.04 < s["pulmonary_respiratory_system_compliance_l_cmH2O"] < 0.10
    assert 4.0 < s["pulmonary_transpulmonary_pressure_end_exp_cmH2O"] < 6.0
    assert 7.0 < s["pulmonary_transpulmonary_pressure_end_insp_cmH2O"] < 14.0
    assert s["pulmonary_overdistension_fraction"] < 0.10
    assert s["pulmonary_recruitment_fraction"] > 0.90


def test_stiff_chest_wall_raises_airway_not_lung_driving_pressure():
    _, base = rollout("baseline")
    _, stiff = rollout("stiff_chest_wall")
    assert stiff["pulmonary_chest_wall_compliance_l_cmH2O"] < base["pulmonary_chest_wall_compliance_l_cmH2O"]
    assert stiff["pulmonary_airway_driving_pressure_cmH2O"] > base["pulmonary_airway_driving_pressure_cmH2O"] + 1.0
    assert abs(
        stiff["pulmonary_transpulmonary_driving_pressure_cmH2O"]
        - base["pulmonary_transpulmonary_driving_pressure_cmH2O"]
    ) < 0.25


def test_low_lung_compliance_raises_transpulmonary_driving_pressure():
    _, base = rollout("baseline")
    _, stiff_lung = rollout("low_lung_compliance")
    assert stiff_lung["pulmonary_lung_compliance_l_cmH2O"] < base["pulmonary_lung_compliance_l_cmH2O"]
    assert (
        stiff_lung["pulmonary_transpulmonary_driving_pressure_cmH2O"]
        > base["pulmonary_transpulmonary_driving_pressure_cmH2O"] + 2.0
    )


def test_peep_recruits_derecruited_lung():
    _, low = rollout("dependent_derecruitment")
    _, peep = rollout("recruitment_peep")
    assert low["pulmonary_recruitment_fraction"] < 0.45
    assert peep["pulmonary_recruitment_fraction"] > 0.90
    assert peep["pao2_mmHg"] > low["pao2_mmHg"] + 15.0


def test_high_peep_produces_overdistension_and_pvr_penalty():
    _, low = rollout("mechanical_ventilation_peep5")
    _, high = rollout("overdistension_peep18")
    assert high["pulmonary_overdistension_fraction"] > low["pulmonary_overdistension_fraction"] + 0.30
    assert high["pulmonary_mechanical_pvr_multiplier"] > low["pulmonary_mechanical_pvr_multiplier"] + 0.20
    assert high["pulmonary_lung_compliance_l_cmH2O"] < low["pulmonary_lung_compliance_l_cmH2O"]


def test_positive_pressure_reduces_cardiac_output():
    _, no_peep = rollout("baseline")
    _, peep12 = rollout("mechanical_ventilation_peep12")
    assert peep12["pulmonary_intrathoracic_pressure_delta_cmH2O"] > 2.0
    assert peep12["cardiac_output_l_min"] < no_peep["cardiac_output_l_min"] - 0.5


def test_positive_pressure_is_more_hazardous_when_volume_depleted():
    def custom(scenario: str, peep: float):
        env = HumanHomeostasisEnv(scenario=scenario)
        env.reset(seed=7)
        env.state.pulmonary_peep_cmH2O = peep
        env.state.pulmonary_positive_pressure_fraction = 1.0 if peep > 0 else 0.0
        info = None
        for _ in range(6):
            _, _, term, trunc, info = env.step(ZERO)
            if term or trunc:
                break
        return info["state"]

    normal = custom("baseline", 12.0)
    dry = custom("dehydrated", 12.0)
    assert dry["cardiac_output_l_min"] < normal["cardiac_output_l_min"]
    assert dry["map_mmHg"] < normal["map_mmHg"]


def test_mechanics_timestep_convergence():
    from openhumsim_rl import HumanConfig

    coarse = HumanHomeostasisEnv(
        config=HumanConfig(integration_step_min=0.25),
        scenario="mechanical_ventilation_peep12",
    )
    fine = HumanHomeostasisEnv(
        config=HumanConfig(integration_step_min=0.125),
        scenario="mechanical_ventilation_peep12",
    )
    coarse.reset(seed=19)
    fine.reset(seed=19)
    ic = ifn = None
    for _ in range(4):
        _, _, tc, trc, ic = coarse.step(ZERO)
        _, _, tf, trf, ifn = fine.step(ZERO)
        if tc or trc or tf or trf:
            break
    sc, sf = ic["state"], ifn["state"]
    assert abs(sc["pulmonary_transpulmonary_pressure_end_insp_cmH2O"] - sf["pulmonary_transpulmonary_pressure_end_insp_cmH2O"]) < 0.5
    assert abs(sc["pulmonary_overdistension_fraction"] - sf["pulmonary_overdistension_fraction"]) < 0.03
    assert abs(sc["cardiac_output_l_min"] - sf["cardiac_output_l_min"]) < 0.25
