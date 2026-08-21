from copy import deepcopy

import numpy as np
import pytest

from openhumsim_rl import HumanConfig
from openhumsim_rl.physiology import HumanState
from openhumsim_rl.pulmonary_exchange import MultiCompartmentPulmonaryExchangeModel
from openhumsim_rl.respiratory import (
    PULMONARY_RER_MAX,
    PULMONARY_RER_MIN,
    RespiratoryModel,
    alveolar_oxygen_tension_mmHg,
    effective_pulmonary_rer,
)


def test_steady_exercise_pulmonary_rer_matches_dynamic_metabolic_rq():
    config = HumanConfig()
    state = HumanState(vo2_ml_min=1_500.0, vo2_demand_ml_min=1_500.0)
    RespiratoryModel(config).update_metabolic_gas_production(state, exercise=1.0)

    # At gas-exchange steady state elimination equals oxidative production.
    state.vco2_elimination_ml_min = state.oxidative_vco2_ml_min
    rer = effective_pulmonary_rer(state, config)

    assert state.metabolic_respiratory_quotient == pytest.approx(14.0 / 15.0)
    assert rer == pytest.approx(state.metabolic_respiratory_quotient, abs=1e-12)


def test_transient_pulmonary_rer_tracks_store_flux_not_metabolic_rq():
    config = HumanConfig()
    state = HumanState(
        vo2_ml_min=250.0,
        vco2_elimination_ml_min=300.0,
        metabolic_respiratory_quotient=0.80,
    )

    rer = effective_pulmonary_rer(state, config)

    assert rer == pytest.approx(1.20)
    assert rer != state.metabolic_respiratory_quotient


@pytest.mark.parametrize(
    ("vo2", "vco2_elimination", "expected"),
    [
        (250.0, 25.0, PULMONARY_RER_MIN),
        (25.0, 250.0, PULMONARY_RER_MAX),
        (0.0, 300.0, 0.90),
        (300.0, 0.0, 0.90),
        (np.nan, 300.0, 0.90),
    ],
)
def test_pulmonary_rer_is_finite_bounded_and_falls_back_for_unreliable_flux(
    vo2, vco2_elimination, expected
):
    config = HumanConfig()
    state = HumanState(
        vo2_ml_min=vo2,
        vco2_elimination_ml_min=vco2_elimination,
        metabolic_respiratory_quotient=0.90,
    )

    rer = effective_pulmonary_rer(state, config)

    assert np.isfinite(rer)
    assert PULMONARY_RER_MIN <= rer <= PULMONARY_RER_MAX
    assert rer == pytest.approx(expected)


def test_full_alveolar_gas_equation_uses_rer_and_has_correct_fio2_limit():
    pco2 = 40.0
    room_pio2 = 0.21 * (760.0 - 47.0)
    room = float(alveolar_oxygen_tension_mmHg(
        inspired_o2_mmHg=room_pio2,
        pco2_mmHg=pco2,
        fio2=0.21,
        pulmonary_rer=1.20,
    ))
    expected_room = room_pio2 - pco2 * (0.21 + 0.79 / 1.20)
    assert room == pytest.approx(expected_room, abs=1e-12)

    # In the full equation FIO2=1 makes the PACO2 coefficient exactly one.
    pure_o2_a = float(alveolar_oxygen_tension_mmHg(
        inspired_o2_mmHg=713.0,
        pco2_mmHg=pco2,
        fio2=1.0,
        pulmonary_rer=0.60,
    ))
    pure_o2_b = float(alveolar_oxygen_tension_mmHg(
        inspired_o2_mmHg=713.0,
        pco2_mmHg=pco2,
        fio2=1.0,
        pulmonary_rer=1.80,
    ))
    assert pure_o2_a == pytest.approx(673.0, abs=1e-12)
    assert pure_o2_b == pytest.approx(pure_o2_a, abs=1e-12)


def test_multicompartment_lung_uses_effective_rer_for_regional_alveolar_o2():
    config = HumanConfig()
    model = MultiCompartmentPulmonaryExchangeModel(config)
    low_rer = HumanState(
        vo2_ml_min=250.0,
        vco2_elimination_ml_min=200.0,
        metabolic_respiratory_quotient=0.80,
    )
    high_rer = deepcopy(low_rer)
    high_rer.vco2_elimination_ml_min = 300.0

    low = model.estimate_arterial_oxygen(
        low_rer, pco2_mmHg=40.0, fio2=0.21, exercise=0.0, dt_min=0.0
    )
    high = model.estimate_arterial_oxygen(
        high_rer, pco2_mmHg=40.0, fio2=0.21, exercise=0.0, dt_min=0.0
    )

    assert low.effective_respiratory_exchange_ratio == pytest.approx(0.80)
    assert high.effective_respiratory_exchange_ratio == pytest.approx(1.20)
    assert high.mean_alveolar_pao2_mmHg > low.mean_alveolar_pao2_mmHg + 10.0
    assert np.isfinite(high.pao2_mmHg)
