from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import HumanConfig


@dataclass
class RenalIntervention:
    saline_ml: float = 0.0
    oral_water_ml: float = 0.0


class RenalModel:
    """Reduced-order renal, fluid, electrolyte and RAAS/ADH model.

    The model explicitly tracks extracellular water plus exchangeable Na/K pools,
    then derives concentrations from those conserved quantities. It is intended
    for RL research and systems integration, not clinical prediction.
    """

    def __init__(self, config: HumanConfig):
        self.cfg = config

    def initialize_state(self, state):
        """Initialize acute ECF/ICF osmotic partitioning."""
        state.icf_volume_l = max(
            1e-6, float(state.total_body_water_l) - float(state.ecf_volume_l)
        )
        ecf_osm = self._ecf_effective_osmoles(state)
        ecf_tonicity = ecf_osm / max(1e-9, float(state.ecf_volume_l))
        state.icf_effective_osmoles_mOsm = float(ecf_tonicity * state.icf_volume_l)
        self._update_tonicity_diagnostics(state)
        return state

    def _ecf_effective_osmoles(self, state) -> float:
        # Sodium salts dominate effective ECF osmoles. Glucose contribution is
        # derived from the conserved Dalla rapid-compartment glucose amount
        # instead of concentration*ECF volume, which would create osmoles during
        # pure water redistribution. 180.155 mg/mmol is glucose molecular mass.
        glucose_mmol = (
            max(0.0, float(state.dalla_gp_mg_kg))
            * self.cfg.body_weight_kg
            / 180.155
        )
        return float(
            2.0 * max(0.0, float(state.ecf_sodium_mmol))
            + glucose_mmol
        )

    def _update_tonicity_diagnostics(self, state):
        ecf_v = max(1e-9, float(state.ecf_volume_l))
        icf_v = max(1e-9, float(state.total_body_water_l) - ecf_v)
        state.icf_volume_l = float(icf_v)
        state.ecf_effective_tonicity_mOsm_l = float(
            self._ecf_effective_osmoles(state) / ecf_v
        )
        state.icf_effective_tonicity_mOsm_l = float(
            max(0.0, float(state.icf_effective_osmoles_mOsm)) / icf_v
        )
        return state

    def equilibrate_transcellular_water(self, state, dt_min: float | None = None):
        """Conserve TBW while moving water ECF<->ICF toward equal tonicity."""
        total = max(1e-6, float(state.total_body_water_l))
        ecf_osm = max(1e-9, self._ecf_effective_osmoles(state))
        icf_osm = max(1e-9, float(state.icf_effective_osmoles_mOsm))
        target_ecf = total * ecf_osm / (ecf_osm + icf_osm)
        target_ecf = float(np.clip(target_ecf, 0.05 * total, 0.95 * total))

        current = float(state.ecf_volume_l)
        if dt_min is None:
            fraction = 1.0
            dt_for_rate = 1.0
        else:
            dt_for_rate = max(1e-9, float(dt_min))
            tau = max(1e-6, float(self.cfg.osmotic_water_equilibration_tau_min))
            fraction = float(1.0 - np.exp(-dt_for_rate / tau))

        proposed = current + (target_ecf - current) * fraction
        new_ecf = float(np.clip(proposed, 0.05 * total, 0.95 * total))
        delta = new_ecf - current
        state.ecf_volume_l = new_ecf
        state.icf_volume_l = total - new_ecf

        plasma_fraction = (
            self.cfg.plasma_volume_baseline_l / self.cfg.ecf_volume_baseline_l
        )
        state.plasma_volume_l = max(
            0.2, float(state.plasma_volume_l) + delta * plasma_fraction
        )
        state.osmotic_water_shift_l_min = float(delta / dt_for_rate)
        self._update_derived_concentrations(state)
        self._update_tonicity_diagnostics(state)
        return state

    def _transcellular_potassium_step(self, state, exercise: float, dt: float):
        """Reduced insulin/pH/exercise-dependent ECF<->ICF K shift.

        Total exchangeable K is conserved; only compartment location changes.
        """
        c = self.cfg
        ecf_v = max(1e-9, float(state.ecf_volume_l))
        current_k = float(state.ecf_potassium_mmol) / ecf_v

        insulin_excess = max(
            0.0, float(state.insulin_uU_ml) - c.insulin_baseline_uU_ml
        )
        insulin_shift = c.potassium_insulin_shift_max_mmol_l * (
            insulin_excess
            / max(1e-9, c.potassium_insulin_half_effect_uU_ml + insulin_excess)
        )

        ph = float(state.ph_arterial)
        if ph < 7.40:
            acid_shift = c.potassium_acid_shift_mmol_l_per_0p1_ph * (
                (7.40 - ph) / 0.10
            )
        else:
            acid_shift = -c.potassium_alkali_shift_mmol_l_per_0p1_ph * (
                (ph - 7.40) / 0.10
            )
        acid_shift = float(np.clip(acid_shift, -0.8, 1.4))

        exercise_shift = (
            c.potassium_exercise_release_max_mmol_l
            * float(np.clip(exercise, 0.0, 1.0))
        )
        target = float(np.clip(
            4.2 - insulin_shift + acid_shift + exercise_shift,
            2.8, 6.8,
        ))
        state.potassium_transcellular_target_mmol_l = target

        tau = max(1e-6, c.potassium_transcellular_tau_min)
        flux = (target - current_k) * ecf_v / tau  # + = ICF -> ECF
        if flux > 0.0:
            flux = min(
                flux,
                max(0.0, state.icf_potassium_mmol - 500.0) / max(dt, 1e-9),
            )
        else:
            flux = -min(
                -flux,
                max(0.0, state.ecf_potassium_mmol - 5.0) / max(dt, 1e-9),
            )

        transfer = flux * dt
        state.ecf_potassium_mmol += transfer
        state.icf_potassium_mmol -= transfer
        state.potassium_transcellular_flux_mmol_min = float(flux)
        return state

    @staticmethod
    def pressure_autoregulation_factor(map_mmHg: float) -> float:
        """Reduced renal autoregulation curve.

        A broad near-plateau is imposed across ordinary arterial pressures;
        below ~80 mmHg filtration becomes pressure dependent. This is still a
        systems-level approximation, not a nephron/afferent-efferent model.
        """
        p = float(map_mmHg)
        if p < 45.0:
            return 0.20
        if p < 80.0:
            return 0.20 + 0.80 * (p - 45.0) / 35.0
        if p <= 140.0:
            return 1.0
        return float(min(1.20, 1.0 + 0.0025 * (p - 140.0)))

    def apply_instant_intervention(self, state, intervention: RenalIntervention):
        # Isotonic saline is represented as extracellular volume plus 154 mmol/L Na.
        saline_l = max(0.0, intervention.saline_ml) / 1000.0
        if saline_l > 0.0:
            state.total_body_water_l += saline_l
            state.ecf_volume_l += saline_l
            sodium_added = 154.0 * saline_l
            chloride_added = 154.0 * saline_l
            state.ecf_sodium_mmol += sodium_added
            state.ecf_chloride_mmol += chloride_added
            state.water_administered_l += saline_l
            state.sodium_administered_mmol += sodium_added
            state.chloride_administered_mmol += chloride_added
            # Approximate plasma fraction of ECF.
            state.plasma_volume_l += saline_l * (
                self.cfg.plasma_volume_baseline_l / self.cfg.ecf_volume_baseline_l
            )

        # Absorbed free water enters ECF first; final ECF/ICF distribution is
        # determined by tonicity rather than a hard-coded 1/3:2/3 split.
        water_l = max(0.0, intervention.oral_water_ml) / 1000.0
        if water_l > 0.0:
            state.total_body_water_l += water_l
            state.water_administered_l += water_l
            state.ecf_volume_l += water_l
            state.plasma_volume_l += water_l * (
                self.cfg.plasma_volume_baseline_l / self.cfg.ecf_volume_baseline_l
            )

        self._update_derived_concentrations(state)
        self.equilibrate_transcellular_water(state, dt_min=None)
        return state

    def step(self, state, exercise: float, dt: float):
        c = self.cfg
        exercise = float(np.clip(exercise, 0.0, 1.0))

        self._update_derived_concentrations(state)

        volume_fraction = state.plasma_volume_l / c.plasma_volume_baseline_l
        ecf_fraction = state.ecf_volume_l / c.ecf_volume_baseline_l
        perfusion_fraction = self.pressure_autoregulation_factor(state.map_mmHg)

        # Simplified plasma osmolality proxy. A constant term represents other osmoles.
        state.plasma_osmolality_mOsm_kg = (
            2.0 * state.sodium_mmol_l
            + state.glucose_mg_dl / 18.0
            + c.baseline_bun_mg_dl / 2.8
        )

        # ADH: osmolality is the main driver; hypovolemia amplifies secretion.
        adh_target = (
            1.0
            + 0.10 * (state.plasma_osmolality_mOsm_kg - 290.0)
            + 7.0 * max(0.0, 1.0 - volume_fraction)
        )
        adh_target = float(np.clip(adh_target, 0.15, 10.0))
        state.adh_relative += (
            adh_target - state.adh_relative
        ) * min(1.0, dt / c.adh_tau_min)

        # Renin -> angiotensin II -> aldosterone cascade.
        renin_target = (
            1.0
            + 4.0 * max(0.0, 1.0 - perfusion_fraction)
            + 5.0 * max(0.0, 1.0 - volume_fraction)
            + 0.04 * max(0.0, 138.0 - state.sodium_mmol_l)
        )
        renin_target = float(np.clip(renin_target, 0.15, 12.0))
        state.renin_relative += (
            renin_target - state.renin_relative
        ) * min(1.0, dt / c.renin_tau_min)

        angii_target = float(np.clip(state.renin_relative, 0.15, 12.0))
        state.angiotensin_ii_relative += (
            angii_target - state.angiotensin_ii_relative
        ) * min(1.0, dt / c.angiotensin_tau_min)

        aldosterone_target = (
            0.75 * state.angiotensin_ii_relative
            + 0.25
            + 1.6 * max(0.0, state.potassium_mmol_l - 4.2)
        )
        aldosterone_target = float(np.clip(aldosterone_target, 0.15, 12.0))
        state.aldosterone_relative += (
            aldosterone_target - state.aldosterone_relative
        ) * min(1.0, dt / c.aldosterone_tau_min)

        # GFR responds to renal reserve, perfusion and circulating volume.
        # AngII provides modest short-term support rather than unlimited rescue.
        autoregulation = perfusion_fraction * (0.75 + 0.25 * np.clip(volume_fraction, 0.4, 1.2))
        angii_support = 1.0 + 0.05 * np.clip(state.angiotensin_ii_relative - 1.0, 0.0, 4.0)
        gfr_target = (
            c.baseline_gfr_ml_min
            * state.renal_function_fraction
            * autoregulation
            * angii_support
        )
        gfr_target = float(np.clip(gfr_target, 2.0, 180.0))
        state.gfr_ml_min += (
            gfr_target - state.gfr_ml_min
        ) * min(1.0, dt / c.gfr_tau_min)
        gfr_fraction = np.clip(state.gfr_ml_min / c.baseline_gfr_ml_min, 0.02, 1.5)

        # Water excretion: ADH reduces free-water loss; volume expansion promotes it.
        volume_diuresis = 1.0 + 3.0 * max(0.0, ecf_fraction - 1.0)
        adh_antidiuresis = 1.4 / (0.4 + max(0.15, state.adh_relative))
        urine_flow_target = (
            c.baseline_urine_flow_ml_min
            * np.sqrt(gfr_fraction)
            * volume_diuresis
            * adh_antidiuresis
        )
        urine_flow_target = float(np.clip(urine_flow_target, 0.05, 12.0))
        state.urine_flow_ml_min += (
            urine_flow_target - state.urine_flow_ml_min
        ) * min(1.0, dt / c.urine_flow_tau_min)

        # Sodium excretion is reduced by RAAS/aldosterone and increased by expansion.
        natriuresis = (
            1.0
            + 4.0 * max(0.0, ecf_fraction - 1.0)
            + 0.04 * max(0.0, state.sodium_mmol_l - 140.0)
        )
        na_retention = 0.60 + 0.40 * max(0.2, state.aldosterone_relative)
        state.urine_sodium_mmol_min = float(np.clip(
            c.baseline_urine_sodium_mmol_min
            * gfr_fraction
            * natriuresis
            / na_retention,
            0.002,
            0.80,
        ))

        # Aldosterone and hyperkalemia enhance K excretion.
        k_drive = np.clip((state.potassium_mmol_l / 4.2) ** 2, 0.15, 4.0)
        state.urine_potassium_mmol_min = float(np.clip(
            c.baseline_urine_potassium_mmol_min
            * gfr_fraction
            * (0.65 + 0.35 * state.aldosterone_relative)
            * k_drive,
            0.001,
            0.50,
        ))

        # Renal acid handling is decomposed into ammonium, titratable acid and
        # urinary bicarbonate.  ``renal_acid_excretion_mmol_min`` is the signed
        # net-acid-excretion diagnostic (NH4 + TA - HCO3).  The ledgers below
        # deliberately keep its components separate: NH4 + TA remove one
        # equivalent from the nonvolatile-acid/UMA pool, whereas urinary HCO3 is
        # removed once from the exchangeable-carbon pool in ``blood_gas.py``.
        acidemia = max(0.0, 7.40 - state.ph_arterial)
        alkalemia = max(0.0, state.ph_arterial - 7.45)
        acid_drive = 1.0 + c.renal_acid_response_gain * acidemia
        nae_gross = c.baseline_net_acid_excretion_mmol_min * gfr_fraction * acid_drive
        ammonium_fraction = float(np.clip(c.baseline_ammonium_fraction_of_nae, 0.0, 1.0))
        state.urine_ammonium_mmol_min = float(max(0.0, nae_gross * ammonium_fraction))
        state.urine_titratable_acid_mmol_min = float(max(0.0, nae_gross * (1.0 - ammonium_fraction)))
        state.urine_bicarbonate_mmol_min = float(max(0.0,
            c.renal_bicarbonaturia_gain * alkalemia * gfr_fraction
        ))
        state.renal_acid_excretion_mmol_min = float(
            state.urine_ammonium_mmol_min
            + state.urine_titratable_acid_mmol_min
            - state.urine_bicarbonate_mmol_min
        )

        # Chloride is an independently conserved strong ion. In this reduced
        # model its urinary loss follows sodium. Do not add NH4-associated Cl
        # here: the alkalinizing equivalent of NH4/TA excretion is already
        # represented by removal from the UMA ledger below. Adding both would
        # raise SID through chloride loss and lower UMA for the same renal event.
        base_cl = (
            c.baseline_urine_chloride_mmol_min
            * (state.urine_sodium_mmol_min / c.baseline_urine_sodium_mmol_min)
        )
        state.urine_chloride_mmol_min = float(np.clip(
            base_cl,
            0.002,
            0.90,
        ))

        # Apply urinary and extrarenal losses to conserved pools.
        urine_l = state.urine_flow_ml_min * dt / 1000.0
        insensible_ml_min = c.basal_fluid_loss_ml_min + c.exercise_fluid_loss_ml_min * exercise
        insensible_l = insensible_ml_min * dt / 1000.0
        total_water_loss_l = urine_l + insensible_l

        # Sweat/insensible sodium and potassium losses rise with exercise.
        sweat_na_mmol = (c.baseline_sweat_na_mmol_min + 0.20 * exercise) * dt
        sweat_k_mmol = (c.baseline_sweat_k_mmol_min + 0.015 * exercise) * dt
        sweat_cl_mmol = (c.baseline_sweat_cl_mmol_min + 0.20 * exercise) * dt

        na_loss_mmol = min(
            max(0.0, state.ecf_sodium_mmol),
            state.urine_sodium_mmol_min * dt + sweat_na_mmol,
        )
        k_loss_mmol = min(
            max(0.0, state.ecf_potassium_mmol),
            state.urine_potassium_mmol_min * dt + sweat_k_mmol,
        )
        cl_loss_mmol = min(
            max(0.0, state.ecf_chloride_mmol),
            state.urine_chloride_mmol_min * dt + sweat_cl_mmol,
        )
        total_water_loss_l = min(
            total_water_loss_l, max(0.0, state.total_body_water_l)
        )

        state.ecf_sodium_mmol -= na_loss_mmol
        state.ecf_potassium_mmol -= k_loss_mmol
        state.ecf_chloride_mmol -= cl_loss_mmol
        state.total_body_water_l -= total_water_loss_l
        state.sodium_lost_mmol += na_loss_mmol
        state.potassium_lost_mmol += k_loss_mmol
        state.chloride_lost_mmol += cl_loss_mmol
        state.water_lost_l += total_water_loss_l

        # Urinary/insensible water is drawn mainly from ECF in the short-term model.
        ecf_water_loss_l = min(
            max(0.0, state.ecf_volume_l),
            0.55 * total_water_loss_l,
        )
        state.ecf_volume_l -= ecf_water_loss_l
        state.plasma_volume_l -= ecf_water_loss_l * (
            c.plasma_volume_baseline_l / c.ecf_volume_baseline_l
        )

        self.equilibrate_transcellular_water(state, dt_min=dt)
        self._transcellular_potassium_step(state, exercise=exercise, dt=dt)

        # Nonvolatile acid ledger. Endogenous acid adds a strong-anion burden;
        # NH4 + titratable-acid excretion removes that burden exactly once.
        # Urinary bicarbonate is intentionally excluded because blood_gas.py
        # subtracts it from the conserved exchangeable-carbon pool. The downstream
        # physicochemical solver then derives HCO3- and pH from PaCO2, strong ions,
        # albumin, phosphate and this burden.
        acid_generated = max(0.0, c.endogenous_acid_production_mmol_min * dt)
        state.nonvolatile_strong_anion_mEq += acid_generated
        state.nonvolatile_acid_generated_mEq += acid_generated
        removable = max(
            0.0,
            (
                state.urine_ammonium_mmol_min
                + state.urine_titratable_acid_mmol_min
            )
            * dt,
        )
        acid_removed = min(max(0.0, state.nonvolatile_strong_anion_mEq), removable)
        state.nonvolatile_strong_anion_mEq -= acid_removed
        state.nonvolatile_acid_excreted_mEq += acid_removed

        self._update_derived_concentrations(state)
        self._update_tonicity_diagnostics(state)
        return state

    @staticmethod
    def _update_derived_concentrations(state):
        # Pure derived-state update: never repair a conserved mass by clipping it.
        # Outflows are bounded at the point of transfer so mass cannot become negative.
        volume = max(1e-9, float(state.ecf_volume_l))
        state.sodium_mmol_l = float(state.ecf_sodium_mmol) / volume
        state.potassium_mmol_l = float(state.ecf_potassium_mmol) / volume
        state.chloride_mmol_l = float(state.ecf_chloride_mmol) / volume
        return state
