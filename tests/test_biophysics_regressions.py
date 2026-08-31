from __future__ import annotations

from copy import deepcopy

import pytest

from openhumsim_rl import HumanConfig
from openhumsim_rl.physiology import HumanState
from openhumsim_rl.pulmonary_exchange import MultiCompartmentPulmonaryExchangeModel
from openhumsim_rl.respiratory_cycle import DynamicRespiratoryCycleModel
from openhumsim_rl.respiratory_mechanics import RespiratoryMechanicsModel


def _completed_cycle(
    *,
    positive_pressure_fraction: float,
    pressure_control_cmH2O: float = 0.0,
    action_assist_cmH2O: float = 0.0,
) -> HumanState:
    config = HumanConfig()
    state = HumanState(
        pulmonary_positive_pressure_fraction=positive_pressure_fraction,
        pulmonary_peep_cmH2O=0.0,
        respiratory_ventilator_pressure_control_cmH2O=pressure_control_cmH2O,
    )
    RespiratoryMechanicsModel(config).step(state)
    cycle = DynamicRespiratoryCycleModel(config)
    cycle.initialize_state(state)
    cycle.step(
        state,
        dt_min=0.25,
        ventilation_pressure_assist_cmH2O=action_assist_cmH2O,
    )
    return state


def test_positive_pressure_fraction_does_not_create_an_airway_pressure_source():
    spontaneous = _completed_cycle(positive_pressure_fraction=0.0)
    transmission_only = _completed_cycle(positive_pressure_fraction=1.0)

    assert transmission_only.tidal_volume_l == pytest.approx(
        spontaneous.tidal_volume_l, abs=1e-12
    )
    assert transmission_only.respiratory_cycle_peak_muscle_pressure_cmH2O == (
        pytest.approx(
            spontaneous.respiratory_cycle_peak_muscle_pressure_cmH2O,
            abs=1e-12,
        )
    )
    assert transmission_only.respiratory_cycle_peak_airway_pressure_cmH2O == 0.0
    assert transmission_only.respiratory_cycle_ventilator_work_j_breath == 0.0


def test_explicit_pressure_sources_still_drive_the_airway():
    controlled = _completed_cycle(
        positive_pressure_fraction=1.0,
        pressure_control_cmH2O=8.0,
    )
    assisted = _completed_cycle(
        positive_pressure_fraction=1.0,
        action_assist_cmH2O=4.0,
    )

    assert controlled.respiratory_cycle_peak_muscle_pressure_cmH2O == 0.0
    assert controlled.respiratory_cycle_peak_airway_pressure_cmH2O > 7.5
    assert controlled.respiratory_cycle_ventilator_work_j_breath > 0.0

    assert assisted.respiratory_cycle_peak_muscle_pressure_cmH2O > 0.0
    assert assisted.respiratory_cycle_peak_airway_pressure_cmH2O > 3.5
    assert assisted.respiratory_cycle_ventilator_work_j_breath > 0.0


@pytest.mark.parametrize("dt_min", [None, 0.0, 0.25])
def test_pulmonary_result_only_evaluation_has_no_state_side_effects(dt_min):
    state = HumanState(
        pulmonary_mean_transpulmonary_pressure_cmH2O=0.0,
        pao2_mmHg=45.0,
        spo2_pct=75.0,
    )
    before = deepcopy(vars(state))

    result = MultiCompartmentPulmonaryExchangeModel(
        HumanConfig()
    ).estimate_arterial_oxygen(
        state,
        pco2_mmHg=70.0,
        fio2=0.13,
        exercise=0.0,
        dt_min=dt_min,
        apply=False,
    )

    assert result.pao2_mmHg != state.pao2_mmHg
    assert vars(state) == before
