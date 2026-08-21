from __future__ import annotations

import numpy as np

from .config import HumanConfig


class WholeBodyEnergyBalanceModel:
    """Reduced whole-body oxygen-deficit and lactate amount ledger.

    This is deliberately an amount balance, not a claim that plasma lactate is
    a one-compartment disease biomarker.  The apparent distribution volume is
    tied to total body water; concentration therefore changes with both net
    lactate flux and dilution.  Basal appearance balances first-order disposal
    at the configured reference concentration.

    Lactate remains a strong anion in the Stewart-Figge calculation.  No
    additional bicarbonate-consumption term is applied here: doing so would
    count the same acid-base perturbation twice.  Likewise, lactate disposal is
    not added as a second CO2 source because oxidative VCO2 already represents
    aggregate substrate oxidation.
    """

    def __init__(self, config: HumanConfig):
        self.cfg = config

    def lactate_distribution_volume_l(self, state) -> float:
        return max(
            1e-9,
            self.cfg.lactate_distribution_volume_fraction_tbw
            * float(state.total_body_water_l),
        )

    def initialize_state(self, state):
        volume = self.lactate_distribution_volume_l(state)
        amount = max(0.0, float(state.lactate_mmol_l)) * volume
        state.lactate_distribution_volume_l = float(volume)
        state.initial_lactate_distribution_volume_l = float(volume)
        state.lactate_amount_mmol = float(amount)
        state.initial_lactate_amount_mmol = float(amount)
        state.lactate_generated_mmol = 0.0
        state.lactate_cleared_mmol = 0.0
        state.lactate_mass_balance_error_mmol = 0.0
        state.lactate_production_mmol_min = 0.0
        state.lactate_clearance_mmol_min = 0.0
        state.exercise_lactate_production_mmol_min = 0.0
        state.hypoxic_lactate_production_mmol_min = 0.0
        state.hypoxic_lactate_production_mmol_l_min = 0.0
        state.instantaneous_oxygen_deficit_ml_min = max(
            0.0, float(getattr(state, "oxygen_debt_ml_min", 0.0))
        )
        state.cumulative_oxygen_deficit_ml = 0.0
        return self.refresh_concentration(state)

    def refresh_concentration(self, state):
        volume = self.lactate_distribution_volume_l(state)
        amount = float(state.lactate_amount_mmol)
        if amount < -1e-12:
            raise FloatingPointError(
                f"negative conserved lactate amount: {amount!r} mmol"
            )
        state.lactate_distribution_volume_l = float(volume)
        state.lactate_mmol_l = float(max(0.0, amount) / volume)
        state.lactate_mass_balance_error_mmol = float(
            state.lactate_amount_mmol
            - (
                state.initial_lactate_amount_mmol
                + state.lactate_generated_mmol
                - state.lactate_cleared_mmol
            )
        )
        return state

    def step_lactate(self, state, *, exercise: float, dt_min: float):
        c = self.cfg
        dt = max(0.0, float(dt_min))
        exercise = float(np.clip(exercise, 0.0, 1.0))
        reference_volume = max(
            1e-9, float(state.initial_lactate_distribution_volume_l)
        )

        # A steady basal turnover is explicit even when concentration is stable.
        # Exercise and hypoxic concentration-rate anchors are converted to amount
        # fluxes before integration.
        basal_production = (
            c.lactate_clearance_per_min
            * c.baseline_lactate_mmol_l
            * reference_volume
        )
        exercise_concentration_rate = (
            c.exercise_lactate_production_linear_mmol_l_min * exercise
            + c.exercise_lactate_production_quadratic_mmol_l_min * exercise**2
        )
        exercise_production = exercise_concentration_rate * reference_volume

        demand = max(
            1e-9,
            float(getattr(state, "vo2_demand_ml_min", state.vo2_ml_min)),
        )
        deficit = max(0.0, float(getattr(state, "oxygen_debt_ml_min", 0.0)))
        deficit_fraction = float(np.clip(deficit / demand, 0.0, 1.0))
        hypoxic_concentration_rate = (
            c.hypoxic_lactate_gain_mmol_l_min * deficit_fraction**2
        )
        hypoxic_production = hypoxic_concentration_rate * reference_volume
        production_rate = basal_production + exercise_production + hypoxic_production

        amount_before = float(state.lactate_amount_mmol)
        if amount_before < -1e-12:
            raise FloatingPointError(
                f"negative conserved lactate amount: {amount_before!r} mmol"
            )
        amount_before = max(0.0, amount_before)
        clearance_rate = c.lactate_clearance_per_min * amount_before
        generated = production_rate * dt
        # Bound the outflow at the transfer, never by clipping the conserved pool.
        cleared = min(clearance_rate * dt, amount_before + generated)
        state.lactate_amount_mmol = float(amount_before + generated - cleared)
        state.lactate_generated_mmol += float(generated)
        state.lactate_cleared_mmol += float(cleared)
        state.lactate_production_mmol_min = float(production_rate)
        state.lactate_clearance_mmol_min = float(clearance_rate)
        state.exercise_lactate_production_mmol_min = float(exercise_production)
        state.hypoxic_lactate_production_mmol_min = float(hypoxic_production)
        state.hypoxic_lactate_production_mmol_l_min = float(
            hypoxic_concentration_rate
        )
        return self.refresh_concentration(state)

    @staticmethod
    def accumulate_oxygen_deficit(
        state, *, previous_deficit_ml_min: float, dt_min: float
    ):
        current = max(0.0, float(state.oxygen_debt_ml_min))
        previous = max(0.0, float(previous_deficit_ml_min))
        state.instantaneous_oxygen_deficit_ml_min = float(current)
        # Trapezoidal accumulation avoids treating an end-of-step rate as if it
        # had applied over the entire operator-splitting interval.
        state.cumulative_oxygen_deficit_ml += 0.5 * (previous + current) * max(
            0.0, float(dt_min)
        )
        return state
