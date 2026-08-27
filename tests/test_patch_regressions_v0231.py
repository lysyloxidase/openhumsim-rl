"""Regression contracts introduced by the 0.23.1 patch release."""

from dataclasses import replace

import numpy as np
import pytest

from openhumsim_rl import HumanConfig, HumanHomeostasisEnv
from openhumsim_rl.acid_base import PhysicochemicalAcidBaseModel
from openhumsim_rl.blood_gas import WholeBloodGasChemistryModel
from openhumsim_rl.physiology import HumanState
from openhumsim_rl.pulmonary_exchange import MultiCompartmentPulmonaryExchangeModel


ZERO = np.zeros(8, dtype=np.float32)


@pytest.mark.parametrize(
    "thresholds",
    [
        (-1.0, 0.0, 1.0),
        (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0),
    ],
)
def test_recruitment_requires_one_threshold_per_pulmonary_unit(thresholds):
    with pytest.raises(ValueError, match="exactly 6 values"):
        replace(
            HumanConfig(),
            pulmonary_unit_closing_pressures_cmH2O=thresholds,
        )


def test_legal_hypoxic_fio2_is_not_silently_clamped_to_015():
    config = replace(HumanConfig(), baseline_fio2=0.13)
    pulmonary = MultiCompartmentPulmonaryExchangeModel(config)

    hypoxic = pulmonary.estimate_arterial_oxygen(
        HumanState(),
        pco2_mmHg=40.0,
        fio2=0.13,
        exercise=0.0,
        dt_min=0.0,
        apply=False,
    )
    boundary = pulmonary.estimate_arterial_oxygen(
        HumanState(),
        pco2_mmHg=40.0,
        fio2=0.15,
        exercise=0.0,
        dt_min=0.0,
        apply=False,
    )

    assert hypoxic.mean_alveolar_pao2_mmHg < boundary.mean_alveolar_pao2_mmHg
    assert hypoxic.pao2_mmHg < boundary.pao2_mmHg

    acid_base = PhysicochemicalAcidBaseModel(config)
    fallback = WholeBloodGasChemistryModel(config, acid_base)
    fallback_hypoxic, _ = fallback._arterial_o2_for_pco2(
        HumanState(), 40.0, 0.13, 0.0
    )
    fallback_boundary, _ = fallback._arterial_o2_for_pco2(
        HumanState(), 40.0, 0.15, 0.0
    )
    assert fallback_hypoxic < fallback_boundary


@pytest.mark.parametrize(
    "config",
    [
        replace(HumanConfig(), carbonic_acid_pka=6.30),
        replace(HumanConfig(), co2_solubility_mmol_l_mmHg=0.0602),
    ],
)
def test_regional_ph_uses_configured_carbonate_constants(config, monkeypatch):
    state = HumanState()
    pulmonary = MultiCompartmentPulmonaryExchangeModel(config)
    original_o2_content = pulmonary._o2_content
    regional_conditions = []

    def capture_o2_content(
        po2_mmHg,
        *,
        ph,
        pco2_mmHg,
        hemoglobin_g_dl,
    ):
        regional_conditions.append((ph, pco2_mmHg))
        return original_o2_content(
            po2_mmHg,
            ph=ph,
            pco2_mmHg=pco2_mmHg,
            hemoglobin_g_dl=hemoglobin_g_dl,
        )

    monkeypatch.setattr(pulmonary, "_o2_content", capture_o2_content)
    pulmonary.estimate_arterial_oxygen(
        state,
        pco2_mmHg=40.0,
        fio2=config.baseline_fio2,
        exercise=0.0,
        dt_min=0.0,
        apply=False,
    )

    assert len(regional_conditions) == 6
    for ph, pco2_mmHg in regional_conditions:
        expected = config.carbonic_acid_pka + np.log10(
            state.bicarbonate_mmol_l
            / (config.co2_solubility_mmol_l_mmHg * pco2_mmHg)
        )
        assert ph == pytest.approx(expected, abs=1e-12)


def test_reset_rejects_terminal_initial_state_and_keeps_reset_required():
    config = replace(
        HumanConfig(),
        map_min_terminate=100.0,
        map_max_terminate=180.0,
        episode_minutes=1.0,
    )
    env = HumanHomeostasisEnv(config=config)

    with pytest.raises(
        ValueError,
        match="terminal initial state.*circulatory_failure_low_map",
    ):
        env.reset(seed=1)

    assert env._needs_reset is True
    with pytest.raises(RuntimeError, match=r"call reset\(\)"):
        env.step(ZERO)
