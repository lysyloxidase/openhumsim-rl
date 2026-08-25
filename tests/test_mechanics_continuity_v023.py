from __future__ import annotations

import numpy as np

from openhumsim_rl import HumanConfig
from openhumsim_rl.physiology import HumanPhysiology, HumanState, Intervention
from openhumsim_rl.respiratory_mechanics import RespiratoryMechanicsModel


def _single_substep(pressure_assist_cmH2O: float) -> dict[str, float]:
    config = HumanConfig()
    model = HumanPhysiology(config)
    state = model.initialize_state(HumanState())
    model._substep(
        state,
        Intervention(ventilation_pressure_assist_cmH2O=pressure_assist_cmH2O),
        config.integration_step_min,
    )
    return {
        "intrathoracic": state.pulmonary_intrathoracic_pressure_delta_cmH2O,
        "pleural_end_insp": state.pulmonary_pleural_pressure_end_insp_cmH2O,
        "cardiac_output": state.cardiac_output_l_min,
        "map": state.map_mmHg,
        "tidal_volume": state.tidal_volume_l,
        "peak_airway": state.respiratory_cycle_peak_airway_pressure_cmH2O,
    }


def test_pressure_assist_is_continuous_at_zero():
    assists = (0.0, 1e-12, 1e-6, 1e-3, 0.1)
    states = [_single_substep(assist) for assist in assists]

    for state in states:
        assert all(np.isfinite(value) for value in state.values())

    zero = states[0]
    epsilon = states[1]
    micro = states[2]
    assert abs(epsilon["intrathoracic"] - zero["intrathoracic"]) < 1e-12
    assert abs(epsilon["pleural_end_insp"] - zero["pleural_end_insp"]) < 1e-10
    assert abs(epsilon["cardiac_output"] - zero["cardiac_output"]) < 1e-10
    assert abs(epsilon["map"] - zero["map"]) < 1e-9
    assert abs(epsilon["tidal_volume"] - zero["tidal_volume"]) < 1e-10

    assert abs(micro["intrathoracic"] - zero["intrathoracic"]) < 1e-6
    assert abs(micro["pleural_end_insp"] - zero["pleural_end_insp"]) < 1e-5
    assert abs(micro["cardiac_output"] - zero["cardiac_output"]) < 1e-5
    assert abs(micro["map"] - zero["map"]) < 1e-4
    assert abs(micro["tidal_volume"] - zero["tidal_volume"]) < 1e-5

    response_limits = {
        "intrathoracic": 0.5,
        "pleural_end_insp": 1.0,
        "cardiac_output": 0.5,
        "map": 2.0,
        "tidal_volume": 0.5,
    }
    for assist, state in zip(assists, states):
        for name, max_gain in response_limits.items():
            assert abs(state[name] - zero[name]) <= max_gain * assist + 1e-10

    monotonic_directions = {
        "intrathoracic": 1.0,
        "pleural_end_insp": 1.0,
        "cardiac_output": -1.0,
        "map": -1.0,
        "tidal_volume": 1.0,
    }
    for name, direction in monotonic_directions.items():
        values = [direction * state[name] for state in states]
        assert all(
            later >= earlier - 1e-10
            for earlier, later in zip(values, values[1:])
        )

    for assist, state in zip(assists, states):
        assert abs(state["peak_airway"] - assist) < max(1e-12, 1e-9 * assist)


def test_passive_plateau_is_total_peep_plus_driving_pressure():
    state = HumanState()
    state.pulmonary_peep_cmH2O = 5.0
    state.respiratory_cycle_auto_peep_cmH2O = 3.25
    model = RespiratoryMechanicsModel(HumanConfig())

    result = model.step(state)
    expected = (
        state.pulmonary_peep_cmH2O
        + state.respiratory_cycle_auto_peep_cmH2O
        + result.airway_driving_pressure_cmH2O
    )

    assert abs(result.passive_equivalent_plateau_cmH2O - expected) < 1e-12
    assert (
        abs(state.pulmonary_passive_equivalent_plateau_pressure_cmH2O - expected)
        < 1e-12
    )
