from __future__ import annotations

from dataclasses import fields, replace
from numbers import Real

import pytest

from openhumsim_rl import HumanConfig
from openhumsim_rl.acid_base import PhysicochemicalAcidBaseModel
from openhumsim_rl.physiology import HumanState


DEPRECATED_IGNORED_PARAMETERS = (
    "liver_glycogen_hgp_fraction",
    "baseline_rbc_volume_l",
    "renal_bicarbonate_tau_min",
    "renal_co2_compensation_gain",
    "potassium_buffer_gain_per_min",
)


def _initialized_acid_base_state() -> HumanState:
    state = HumanState()
    PhysicochemicalAcidBaseModel(HumanConfig()).initialize_state(state)
    return state


def test_acid_base_snapshot_requires_a_bracketed_charge_root() -> None:
    state = _initialized_acid_base_state()
    state.nonvolatile_strong_anion_mEq = 1e9

    with pytest.raises(FloatingPointError, match="not bracketed"):
        PhysicochemicalAcidBaseModel(HumanConfig()).snapshot_for_pco2(state, 40.0)


def test_acid_base_snapshot_requires_configured_charge_tolerance() -> None:
    state = _initialized_acid_base_state()
    config = HumanConfig(
        acid_base_charge_tolerance_mEq_l=1e-16,
        acid_base_max_iterations=1,
    )

    with pytest.raises(FloatingPointError, match="did not reach charge tolerance"):
        PhysicochemicalAcidBaseModel(config).snapshot_for_pco2(state, 40.0)


def test_acid_base_snapshot_preserves_valid_wide_pco2_candidates() -> None:
    state = _initialized_acid_base_state()
    model = PhysicochemicalAcidBaseModel(HumanConfig())

    for pco2 in (5.0, 40.0, 160.0):
        result = model.snapshot_for_pco2(state, pco2)
        assert 4.0 <= result.ph <= 10.0
        assert abs(result.charge_balance_residual_mEq_l) <= (
            model.cfg.acid_base_charge_tolerance_mEq_l
        )


@pytest.mark.parametrize("pco2", [0.0, -1.0, float("nan"), float("inf")])
def test_acid_base_snapshot_rejects_invalid_pco2(pco2: float) -> None:
    with pytest.raises(ValueError, match="paco2_mmHg"):
        PhysicochemicalAcidBaseModel(HumanConfig()).snapshot_for_pco2(
            _initialized_acid_base_state(), pco2
        )


def test_every_numeric_config_field_rejects_nan_and_infinity() -> None:
    baseline = HumanConfig()
    for definition in fields(baseline):
        current = getattr(baseline, definition.name)
        for invalid in (float("nan"), float("inf"), float("-inf")):
            if isinstance(current, tuple):
                replacement = (*current[:-1], invalid)
            else:
                assert isinstance(current, Real)
                replacement = invalid
            error_name = (
                "water vapor pressure"
                if definition.name == "water_vapor_pressure_mmHg"
                else definition.name
            )
            with pytest.raises(ValueError, match=error_name):
                replace(baseline, **{definition.name: replacement})


def test_glucagon_time_constant_must_be_strictly_positive() -> None:
    for invalid in (0.0, -1.0):
        with pytest.raises(ValueError, match="glucagon_tau_min"):
            HumanConfig(glucagon_tau_min=invalid)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"integration_step_min": 6.0}, "simulation steps"),
        ({"dead_space_l": 0.50}, "dead_space_l"),
        ({"baseline_hematocrit": 1.0}, "baseline_hematocrit"),
        ({"pulmonary_baseline_shunt_fraction": 1.1}, "shunt_fraction"),
        ({"cv_v_la0_ml": 61.0}, "compartment volumes"),
        ({"cv_v0_la_ml": 60.0}, "cv_v0_la_ml"),
        ({"cv_lv_emin": 2.30}, "cv_lv_emin"),
        (
            {"pulmonary_unit_closing_pressures_cmH2O": (-1.0, 0.0, 0.0)},
            "strictly increasing",
        ),
        (
            {"pulmonary_mechanics_peep_high_cmH2O": 18.0},
            "PEEP anchors",
        ),
        ({"carbonic_acid_pka": 7.5}, "acid-base constants"),
        ({"glucose_max_terminate": 30.0}, "glucose_min_terminate"),
        ({"pbpk_high_exposure_mg_l": 0.8}, "pbpk_target_effect_site"),
    ],
)
def test_config_rejects_structurally_inconsistent_ranges(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(HumanConfig(), **changes)


@pytest.mark.parametrize("name", DEPRECATED_IGNORED_PARAMETERS)
def test_ignored_compatibility_parameters_are_explicitly_deprecated(name: str) -> None:
    definition = next(item for item in fields(HumanConfig) if item.name == name)
    assert "ignored compatibility field" in definition.metadata["deprecated"]

    with pytest.warns(DeprecationWarning, match=rf"{name} is deprecated and ignored"):
        HumanConfig(**{name: float(definition.default) + 0.01})
