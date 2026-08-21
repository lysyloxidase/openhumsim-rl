from __future__ import annotations

from dataclasses import dataclass
from math import log10

import numpy as np

from .acid_base import AcidBaseDiagnostics, PhysicochemicalAcidBaseModel
from .config import HumanConfig
from .oxygen_binding import OxygenBindingModel
from .respiratory import alveolar_oxygen_tension_mmHg, effective_pulmonary_rer


@dataclass(frozen=True)
class WholeBloodCO2Snapshot:
    plasma_ph: float
    rbc_ph: float
    pco2_mmHg: float
    oxygen_saturation_fraction: float
    plasma_dissolved_co2_mmol_l: float
    plasma_bicarbonate_mmol_l: float
    plasma_carbonate_mmol_l: float
    rbc_bicarbonate_mmol_l: float
    rbc_carbonate_mmol_l: float
    carbamino_co2_mmol_l_blood: float
    total_co2_mmol_l_blood: float
    rbc_chloride_mmol_l: float
    donnan_rcl: float
    donnan_rh: float
    hemoglobin_buffer_capacity_mEq_l_pH: float
    hemoglobin_bound_proton_change_mEq_l: float


@dataclass(frozen=True)
class CarbonLedgerStart:
    """Immutable start-of-substep values for an idempotent carbon solve.

    The coupled O2/CO2 solver may need several algebraic passes because final
    oxygen delivery changes oxidative VCO2, while PaCO2 changes oxygenation and
    pulmonary CO2 elimination.  Every pass must therefore be evaluated from
    the same physical start of the interval; otherwise cumulative ledgers would
    be advanced once per numerical iteration rather than once per time step.
    """

    exchangeable_pool_mmol: float
    generated_mmol: float
    eliminated_mmol: float
    urinary_bicarbonate_loss_mmol: float
    elimination_ml_min: float


class WholeBloodGasChemistryModel:
    """Reduced whole-blood CO2 chemistry and transport model.

    An explicit rapidly exchangeable CO2 pool resolves arterial PaCO2 from
    carbon mass balance. Plasma carbonate chemistry is provided by the
    Stewart-Figge closure in ``acid_base.py``. Red-cell chemistry adds:

    - measured human Gibbs-Donnan H+/Cl- distribution relationships from
      Funder & Wieth (1966),
    - an explicit RBC bicarbonate/carbonate compartment,
    - a diagnostic estimate of hemoglobin non-carbonic buffer capacity,
    - reduced carbamino-CO2 binding with a Dash-et-al.-anchored standard
      carbamino saturation and an oxygen-linked Haldane affinity term,
    - an arteriovenous chloride-shift diagnostic with exact local chloride
      redistribution balance,
    - Fick-consistent arteriovenous CO2-content difference.

    It is deliberately not claimed to be a full reproduction of the O'Neill &
    Robbins 2017 model: Band-3 kinetics, individual Hb protonation sites,
    temperature/2,3-DPG dependence and cell-water/osmotic shifts remain reduced.
    In particular, the Hb-bound-proton estimate is not part of the plasma pH
    root, so this module must not be interpreted as whole-blood acid-base closure.
    """

    ARTERIAL_PCO2_MIN_MMHG = 5.0
    ARTERIAL_PCO2_MAX_MMHG = 150.0

    def __init__(self, config: HumanConfig, acid_base: PhysicochemicalAcidBaseModel, pulmonary_exchange=None):
        self.cfg = config
        self.acid_base = acid_base
        self.pulmonary_exchange = pulmonary_exchange
        self.oxygen_binding = OxygenBindingModel(config)

    @property
    def gas_mmol_per_ml_stpd(self) -> float:
        return 1.0 / self.cfg.co2_gas_molar_volume_l_per_mol_stpd

    def exchangeable_volume_l(self, state) -> float:
        return max(
            1.0,
            self.cfg.co2_exchangeable_volume_fraction_tbw * float(state.total_body_water_l),
        )

    def rbc_donnan_ratios(self, plasma_ph: float) -> tuple[float, float, float]:
        c = self.cfg
        rcl = float(np.clip(c.funder_rcl_intercept - c.funder_rcl_ph_slope * plasma_ph, 0.35, 1.10))
        rh = float(np.clip(c.funder_rh_intercept - c.funder_rh_ph_slope * plasma_ph, 0.35, 1.10))
        # rH = [H+]plasma/[H+]RBC, hence pHRBC = pHplasma + log10(rH).
        rbc_ph = float(plasma_ph + log10(rh))
        return rcl, rh, rbc_ph

    def hemoglobin_monomer_mmol_l_blood(self, hemoglobin_g_dl: float | None = None) -> float:
        # Hb g/dL -> g/L -> mmol monomer/L.
        hb = self.cfg.hemoglobin_g_dl if hemoglobin_g_dl is None else float(hemoglobin_g_dl)
        return (10.0 * hb) / self.cfg.hemoglobin_monomer_mw_g_mmol

    def carbamino_fraction(self, pco2_mmHg: float, oxygen_sat_fraction: float) -> float:
        c = self.cfg
        s0 = float(np.clip(c.baseline_carbamino_fraction, 1e-6, 0.95))
        # A one-site Langmuir representation is anchored to Dash et al.'s reported
        # standard-state amino-group carbamino saturation (13.1%). Deoxygenation
        # increases apparent affinity, representing the Haldane effect.
        s_ref_o2 = float(np.clip(c.carbamino_reference_o2_saturation, 0.0, 1.0))
        affinity_reference = 1.0 + c.carbamino_haldane_affinity_gain * (1.0 - s_ref_o2)
        # Choose the oxygenated-site P50 so the model reproduces the reported
        # 13.1% standard-state carbamino saturation at PCO2=40 mmHg and
        # SO2=97.2%, rather than at an unphysical exactly-100% saturation.
        p50_oxygenated = (40.0 * (1.0 - s0) / s0) * affinity_reference
        affinity_multiplier = 1.0 + c.carbamino_haldane_affinity_gain * max(
            0.0, 1.0 - float(np.clip(oxygen_sat_fraction, 0.0, 1.0))
        )
        p50 = p50_oxygenated / max(1e-9, affinity_multiplier)
        pco2 = max(0.0, float(pco2_mmHg))
        return float(np.clip(pco2 / (p50 + pco2), 0.0, 0.95))

    def snapshot(
        self,
        *,
        plasma_ph: float,
        pco2_mmHg: float,
        oxygen_sat_fraction: float,
        plasma_chloride_mmol_l: float,
        hemoglobin_g_dl: float | None = None,
        hematocrit_fraction: float | None = None,
    ) -> WholeBloodCO2Snapshot:
        c = self.cfg
        pco2 = max(1e-9, float(pco2_mmHg))
        so2 = float(np.clip(oxygen_sat_fraction, 0.0, 1.0))

        dissolved = c.co2_solubility_mmol_l_mmHg * pco2
        hco3 = dissolved * 10.0 ** (plasma_ph - c.carbonic_acid_pka)
        co3 = hco3 * 10.0 ** (plasma_ph - c.carbonate_pka2)

        rcl, rh, rbc_ph = self.rbc_donnan_ratios(plasma_ph)
        rbc_hco3 = rcl * hco3
        rbc_co3 = rbc_hco3 * 10.0 ** (rbc_ph - c.carbonate_pka2)
        rbc_cl = rcl * plasma_chloride_mmol_l

        hb = c.hemoglobin_g_dl if hemoglobin_g_dl is None else max(0.0, float(hemoglobin_g_dl))
        hct = c.baseline_hematocrit if hematocrit_fraction is None else float(np.clip(hematocrit_fraction, 0.05, 0.80))
        hb_monomer = self.hemoglobin_monomer_mmol_l_blood(hb)
        carb_fraction = self.carbamino_fraction(pco2, so2)
        carbamino = hb_monomer * carb_fraction

        plasma_water_fraction_of_blood = (1.0 - hct) * c.plasma_water_fraction
        rbc_water_fraction_of_blood = hct * c.rbc_water_fraction
        total = (
            plasma_water_fraction_of_blood * (dissolved + hco3 + co3)
            + rbc_water_fraction_of_blood * (dissolved + rbc_hco3 + rbc_co3)
            + carbamino
        )

        hb_buffer_capacity = c.hemoglobin_buffer_value_mEq_per_mmol_pH * hb_monomer
        _, _, baseline_rbc_ph = self.rbc_donnan_ratios(c.baseline_ph_arterial)
        proton_delta = hb_buffer_capacity * (baseline_rbc_ph - rbc_ph)
        # Oxygen-linked proton binding is kept explicit as a diagnostic term and
        # is not fed back into the Stewart-Figge plasma pH solve. The
        # 0.3 mol/mol coefficient corresponds to the whole-blood base-excess
        # coefficient reported for oxygen-linked Hb buffering near normal blood.
        proton_delta += 0.30 * hb_monomer * (1.0 - so2)

        return WholeBloodCO2Snapshot(
            plasma_ph=float(plasma_ph),
            rbc_ph=float(rbc_ph),
            pco2_mmHg=float(pco2),
            oxygen_saturation_fraction=so2,
            plasma_dissolved_co2_mmol_l=float(dissolved),
            plasma_bicarbonate_mmol_l=float(hco3),
            plasma_carbonate_mmol_l=float(co3),
            rbc_bicarbonate_mmol_l=float(rbc_hco3),
            rbc_carbonate_mmol_l=float(rbc_co3),
            carbamino_co2_mmol_l_blood=float(carbamino),
            total_co2_mmol_l_blood=float(total),
            rbc_chloride_mmol_l=float(rbc_cl),
            donnan_rcl=float(rcl),
            donnan_rh=float(rh),
            hemoglobin_buffer_capacity_mEq_l_pH=float(hb_buffer_capacity),
            hemoglobin_bound_proton_change_mEq_l=float(proton_delta),
        )

    @staticmethod
    def _severinghaus_saturation_fraction(po2_mmHg: float) -> float:
        po2 = max(0.0, float(po2_mmHg))
        denominator = po2**3 + 150.0 * po2
        if denominator <= 0.0:
            return 0.0
        return float(np.clip(1.0 / (1.0 + 23400.0 / denominator), 0.0, 1.0))

    def _arterial_o2_for_pco2(self, state, pco2_mmHg: float, fio2: float, exercise: float) -> tuple[float, float]:
        if self.pulmonary_exchange is not None:
            # Explicit operator splitting keeps the solve tractable: the carbon
            # pool solver holds arterial oxygenation at the current
            # pulmonary state while solving PCO2, then the six-compartment lung
            # is evaluated once after the carbon solve. This avoids nesting the
            # V/Q solver inside every PCO2 root-finding iteration.
            return float(state.pao2_mmHg), float(np.clip(state.spo2_pct / 100.0, 0.0, 1.0))
        c = self.cfg
        fio2_fraction = float(np.clip(fio2, 0.15, 1.0))
        inspired_o2 = fio2_fraction * (
            c.atmospheric_pressure_mmHg - c.water_vapor_pressure_mmHg
        )
        aa_gradient = c.baseline_aa_gradient_mmHg + 3.0 * float(np.clip(exercise, 0.0, 1.0))
        alveolar_pao2 = float(alveolar_oxygen_tension_mmHg(
            inspired_o2_mmHg=inspired_o2,
            pco2_mmHg=pco2_mmHg,
            fio2=fio2_fraction,
            pulmonary_rer=effective_pulmonary_rer(state, c),
        ))
        pao2 = max(20.0, alveolar_pao2 - aa_gradient)
        return float(pao2), self._severinghaus_saturation_fraction(pao2)

    def _content_at_pco2(
        self,
        state,
        pco2_mmHg: float,
        fio2: float,
        exercise: float,
        *,
        pao2_override_mmHg: float | None = None,
    ) -> tuple[float, AcidBaseDiagnostics, WholeBloodCO2Snapshot, float, float]:
        ab = self.acid_base.snapshot_for_pco2(state, pco2_mmHg)
        if pao2_override_mmHg is None:
            pao2, _ = self._arterial_o2_for_pco2(
                state, pco2_mmHg, fio2, exercise
            )
        else:
            pao2 = max(0.0, float(pao2_override_mmHg))
        so2 = self.oxygen_binding.saturation(pao2, ab.ph, pco2_mmHg).saturation_fraction
        wb = self.snapshot(
            plasma_ph=ab.ph,
            pco2_mmHg=pco2_mmHg,
            oxygen_sat_fraction=so2,
            plasma_chloride_mmol_l=state.chloride_mmol_l,
            hemoglobin_g_dl=float(getattr(state, "hemoglobin_g_dl", self.cfg.hemoglobin_g_dl)),
            hematocrit_fraction=float(getattr(state, "hematocrit_fraction", self.cfg.baseline_hematocrit)),
        )
        return wb.total_co2_mmol_l_blood, ab, wb, pao2, so2

    def initialize_state(self, state, *, fio2: float | None = None, exercise: float = 0.0):
        c = self.cfg
        fio2 = c.baseline_fio2 if fio2 is None else fio2
        # Preserve the scenario's current PaCO2; establish the corresponding
        # exchangeable carbon pool and ledger at t=0.
        content, ab, wb, pao2, so2 = self._content_at_pco2(
            state, state.paco2_mmHg, fio2, exercise
        )
        self.acid_base._apply_snapshot(state, ab)
        state.pao2_mmHg = float(pao2)
        state.spo2_pct = float(100.0 * so2)
        self._apply_arterial_snapshot(state, wb)
        state.exchangeable_co2_pool_mmol = float(content * self.exchangeable_volume_l(state))
        state.initial_exchangeable_co2_pool_mmol = float(state.exchangeable_co2_pool_mmol)
        state.co2_generated_mmol = 0.0
        state.co2_eliminated_mmol = 0.0
        state.co2_urinary_bicarbonate_loss_mmol = 0.0
        state.co2_mass_balance_error_mmol = 0.0
        state.co2_content_solver_residual_mmol_l = 0.0
        alveolar_dead = float(np.clip(
            getattr(state, "pulmonary_alveolar_dead_space_fraction", 0.0),
            0.0,
            0.80,
        ))
        effective_co2_va = max(
            0.0, float(state.alveolar_ventilation_l_min)
        ) * (1.0 - alveolar_dead)
        state.effective_co2_ventilation_l_min = float(effective_co2_va)
        state.vco2_elimination_ml_min = float(
            max(0.0, float(state.paco2_mmHg)) * effective_co2_va / 0.863
        )
        return state

    @staticmethod
    def capture_carbon_ledger_start(state) -> CarbonLedgerStart:
        """Capture the conserved pool and cumulative counters at interval start."""
        return CarbonLedgerStart(
            exchangeable_pool_mmol=float(state.exchangeable_co2_pool_mmol),
            generated_mmol=float(state.co2_generated_mmol),
            eliminated_mmol=float(state.co2_eliminated_mmol),
            urinary_bicarbonate_loss_mmol=float(
                state.co2_urinary_bicarbonate_loss_mmol
            ),
            elimination_ml_min=float(state.vco2_elimination_ml_min),
        )

    def _apply_arterial_snapshot(self, state, wb: WholeBloodCO2Snapshot):
        state.rbc_ph = wb.rbc_ph
        state.rbc_bicarbonate_mmol_l = wb.rbc_bicarbonate_mmol_l
        state.rbc_carbonate_mmol_l = wb.rbc_carbonate_mmol_l
        state.rbc_chloride_mmol_l = wb.rbc_chloride_mmol_l
        state.carbamino_co2_mmol_l_blood = wb.carbamino_co2_mmol_l_blood
        state.arterial_total_co2_mmol_l_blood = wb.total_co2_mmol_l_blood
        state.hemoglobin_buffer_capacity_mEq_l_pH = wb.hemoglobin_buffer_capacity_mEq_l_pH
        state.hemoglobin_bound_proton_change_mEq_l = wb.hemoglobin_bound_proton_change_mEq_l
        return state

    def _solve_carbon_content_target(
        self,
        state,
        *,
        fio2: float,
        exercise: float,
        target_content_at_pco2,
        pao2_override_mmHg: float | None = None,
    ):
        """Solve Cwb(PCO2) against a pure candidate target-content function."""
        c = self.cfg

        lo = self.ARTERIAL_PCO2_MIN_MMHG
        hi = self.ARTERIAL_PCO2_MAX_MMHG
        flo = (
            self._content_at_pco2(
                state,
                lo,
                fio2,
                exercise,
                pao2_override_mmHg=pao2_override_mmHg,
            )[0]
            - float(target_content_at_pco2(lo))
        )
        fhi = (
            self._content_at_pco2(
                state,
                hi,
                fio2,
                exercise,
                pao2_override_mmHg=pao2_override_mmHg,
            )[0]
            - float(target_content_at_pco2(hi))
        )
        if flo * fhi > 0.0:
            pco2 = lo if abs(flo) < abs(fhi) else hi
        else:
            for _ in range(c.co2_pool_solver_max_iterations):
                mid = 0.5 * (lo + hi)
                fm = (
                    self._content_at_pco2(
                        state,
                        mid,
                        fio2,
                        exercise,
                        pao2_override_mmHg=pao2_override_mmHg,
                    )[0]
                    - float(target_content_at_pco2(mid))
                )
                if abs(fm) <= c.co2_pool_solver_tolerance_mmol_l:
                    lo = hi = mid
                    break
                if flo * fm <= 0.0:
                    hi, fhi = mid, fm
                else:
                    lo, flo = mid, fm
            pco2 = 0.5 * (lo + hi)

        content, ab, wb, pao2, so2 = self._content_at_pco2(
            state,
            pco2,
            fio2,
            exercise,
            pao2_override_mmHg=pao2_override_mmHg,
        )
        target_content = float(target_content_at_pco2(pco2))
        return pco2, content, target_content, ab, wb, pao2, so2

    def _solve_current_carbon_pool(self, state, *, fio2: float, exercise: float):
        """Solve arterial chemistry for the already-stored exchangeable CO2 pool.

        This helper is deliberately free of carbon fluxes and cumulative-ledger
        updates.  In the integrated pulmonary model, ``_content_at_pco2`` holds
        the current PaO2 fixed while consistently recomputing saturation for each
        candidate pH/PCO2 pair.  That permits a post-pulmonary Haldane correction
        without advancing metabolism, ventilation, or renal bicarbonate loss a
        second time.
        """
        target_content = (
            float(state.exchangeable_co2_pool_mmol)
            / self.exchangeable_volume_l(state)
        )
        return self._solve_carbon_content_target(
            state,
            fio2=fio2,
            exercise=exercise,
            target_content_at_pco2=lambda _pco2: target_content,
            pao2_override_mmHg=float(state.pao2_mmHg),
        )

    def arterial_carbon_pool_closure_residual_mmol_l(
        self, state, fio2: float, exercise: float
    ) -> float:
        """Return final-gas whole-blood content minus stored-pool content.

        Unlike ``state.co2_content_solver_residual_mmol_l``, which records the
        residual at the most recent carbon solve, this read-only diagnostic can
        be called after pulmonary oxygenation changes PaO2/SaO2.  It therefore
        exposes any Haldane-related closure drift introduced by operator splitting.
        """
        content = self._content_at_pco2(
            state,
            float(state.paco2_mmHg),
            fio2,
            exercise,
            pao2_override_mmHg=float(state.pao2_mmHg),
        )[0]
        target_content = (
            float(state.exchangeable_co2_pool_mmol)
            / self.exchangeable_volume_l(state)
        )
        return float(content - target_content)

    def reconcile_arterial_carbon_pool(
        self, state, fio2: float, exercise: float
    ):
        """Reconcile arterial chemistry to the existing exchangeable CO2 pool.

        No CO2 is generated, ventilated, or excreted here, and none of the
        cumulative carbon ledgers are changed.  The method is intended for one
        post-pulmonary equilibrium correction after PaO2 has been updated.
        """
        pco2, content, target_content, ab, wb, pao2, so2 = (
            self._solve_current_carbon_pool(
                state, fio2=fio2, exercise=exercise
            )
        )
        return self._apply_carbon_pool_solution(
            state,
            pco2=pco2,
            content=content,
            target_content=target_content,
            ab=ab,
            wb=wb,
            pao2=pao2,
            so2=so2,
        )

    def _apply_carbon_pool_solution(
        self,
        state,
        *,
        pco2: float,
        content: float,
        target_content: float,
        ab: AcidBaseDiagnostics,
        wb: WholeBloodCO2Snapshot,
        pao2: float,
        so2: float,
    ):
        """Commit one already-solved arterial equilibrium snapshot."""
        state.paco2_mmHg = float(pco2)
        state.pao2_mmHg = float(pao2)
        state.spo2_pct = float(100.0 * so2)
        self.acid_base._apply_snapshot(state, ab)
        self._apply_arterial_snapshot(state, wb)
        state.co2_content_solver_residual_mmol_l = float(
            content - target_content
        )
        return state

    def step_arterial_carbon_balance(
        self,
        state,
        *,
        fio2: float,
        exercise: float,
        dt_min: float,
        ledger_start: CarbonLedgerStart | None = None,
        generation_average_ml_min: float | None = None,
    ):
        """Solve and commit one interval from a fixed carbon-ledger baseline.

        Passing ``ledger_start`` makes the operation idempotent: a coupled
        solver can replace a previous numerical candidate without advancing
        cumulative generation, elimination, or urinary loss a second time.
        """
        dt = float(dt_min)
        start = (
            self.capture_carbon_ledger_start(state)
            if ledger_start is None
            else ledger_start
        )

        # Metabolic production is converted from mL gas/min (STPD convention of
        # the respiratory model) to mmol/min. Pulmonary elimination follows the
        # standard alveolar-ventilation relation inverted from PaCO2 = 0.863 VCO2/VA.
        if generation_average_ml_min is None:
            generation_average_ml_min = getattr(
                state,
                "vco2_generation_interval_average_ml_min",
                state.vco2_ml_min,
            )
        generation_average_ml_min = max(
            0.0, float(generation_average_ml_min)
        )
        state.vco2_generation_interval_average_ml_min = float(
            generation_average_ml_min
        )
        generated = (
            generation_average_ml_min * self.gas_mmol_per_ml_stpd * dt
        )
        # Regional V/Q affects CO2 elimination through alveolar
        # dead-space ventilation. Alveolar ventilation already excludes
        # anatomical dead space, so only the regional wasted fraction is applied.
        alveolar_dead = float(np.clip(
            getattr(state, "pulmonary_alveolar_dead_space_fraction", 0.0), 0.0, 0.80
        ))
        effective_co2_va = max(0.0, float(state.alveolar_ventilation_l_min)) * (1.0 - alveolar_dead)
        state.effective_co2_ventilation_l_min = float(effective_co2_va)
        # This is the sole ledger for urinary HCO3 loss. Renal NH4/TA excretion
        # is handled separately in the nonvolatile-acid/UMA ledger; NAE is not
        # subtracted again from either conserved pool.
        urinary = max(0.0, float(state.urine_bicarbonate_mmol_min)) * dt

        # Elimination is trapezoidal in its stored previous endpoint and the
        # current endpoint implied by the unknown final PaCO2.  The candidate
        # pool is therefore part of the same scalar equilibrium root.  Nothing is
        # committed inside the root, so counters are advanced exactly once.
        pool_before = float(start.exchangeable_pool_mmol)
        elimination_start_ml_min = max(0.0, float(start.elimination_ml_min))
        exchangeable_volume_l = self.exchangeable_volume_l(state)

        def elimination_end_ml_min(pco2_mmHg: float) -> float:
            return max(0.0, float(pco2_mmHg) * effective_co2_va / 0.863)

        def eliminated_mmol_at(pco2_mmHg: float) -> float:
            average_elimination_ml_min = 0.5 * (
                elimination_start_ml_min
                + elimination_end_ml_min(pco2_mmHg)
            )
            return (
                average_elimination_ml_min
                * self.gas_mmol_per_ml_stpd
                * dt
            )

        def target_content_at_pco2(pco2_mmHg: float) -> float:
            candidate_pool = max(
                1e-9,
                pool_before
                + generated
                - eliminated_mmol_at(pco2_mmHg)
                - urinary,
            )
            return candidate_pool / exchangeable_volume_l

        solution = self._solve_carbon_content_target(
            state,
            fio2=fio2,
            exercise=exercise,
            target_content_at_pco2=target_content_at_pco2,
        )
        pco2, content, target_content, ab, wb, pao2, so2 = solution
        elimination_end = elimination_end_ml_min(pco2)
        eliminated = eliminated_mmol_at(pco2)
        state.exchangeable_co2_pool_mmol = max(
            1e-9,
            pool_before + generated - eliminated - urinary,
        )
        state.co2_generated_mmol = float(start.generated_mmol + generated)
        state.co2_eliminated_mmol = float(start.eliminated_mmol + eliminated)
        state.co2_urinary_bicarbonate_loss_mmol = float(
            start.urinary_bicarbonate_loss_mmol + urinary
        )
        state.vco2_elimination_ml_min = float(elimination_end)

        self._apply_carbon_pool_solution(
            state,
            pco2=pco2,
            content=content,
            target_content=target_content,
            ab=ab,
            wb=wb,
            pao2=pao2,
            so2=so2,
        )
        state.co2_mass_balance_error_mmol = float(
            state.exchangeable_co2_pool_mmol
            - (
                state.initial_exchangeable_co2_pool_mmol
                + state.co2_generated_mmol
                - state.co2_eliminated_mmol
                - state.co2_urinary_bicarbonate_loss_mmol
            )
        )
        return state

    def update_venous_diagnostics(self, state):
        c = self.cfg
        sao2 = float(np.clip(state.spo2_pct / 100.0, 0.0, 1.0))
        svo2 = float(np.clip(state.mixed_venous_o2_sat_pct / 100.0, 0.0, 1.0))
        sat_drop = max(0.0, sao2 - svo2)
        chloride_shift = min(
            c.chloride_shift_max_mmol_l,
            c.chloride_shift_gain_mmol_l_per_sat_fraction * sat_drop,
        )
        venous_plasma_cl = max(1.0, state.chloride_mmol_l - chloride_shift)

        # Fick CO2 conservation fixes the arteriovenous content increment.
        co = max(0.25, float(state.cardiac_output_l_min))
        av_delta = max(0.0, float(state.vco2_ml_min)) * self.gas_mmol_per_ml_stpd / co
        target_venous_content = float(state.arterial_total_co2_mmol_l_blood + av_delta)

        def venous_at(pco2: float):
            ab = self.acid_base.snapshot_for_pco2(
                state, pco2, chloride_override=venous_plasma_cl
            )
            wb = self.snapshot(
                plasma_ph=ab.ph,
                pco2_mmHg=pco2,
                oxygen_sat_fraction=svo2,
                plasma_chloride_mmol_l=venous_plasma_cl,
                hemoglobin_g_dl=float(getattr(state, "hemoglobin_g_dl", self.cfg.hemoglobin_g_dl)),
                hematocrit_fraction=float(getattr(state, "hematocrit_fraction", self.cfg.baseline_hematocrit)),
            )
            return wb.total_co2_mmol_l_blood, ab, wb

        lo, hi = 10.0, 160.0
        flo = venous_at(lo)[0] - target_venous_content
        fhi = venous_at(hi)[0] - target_venous_content
        if flo * fhi > 0.0:
            pv = lo if abs(flo) < abs(fhi) else hi
        else:
            for _ in range(c.co2_pool_solver_max_iterations):
                mid = 0.5 * (lo + hi)
                fm = venous_at(mid)[0] - target_venous_content
                if abs(fm) <= c.co2_pool_solver_tolerance_mmol_l:
                    lo = hi = mid
                    break
                if flo * fm <= 0.0:
                    hi, fhi = mid, fm
                else:
                    lo, flo = mid, fm
            pv = 0.5 * (lo + hi)

        venous_content, vab, vwb = venous_at(pv)

        # Chloride chemistry has two distinct effects that must not be conflated:
        # (1) Hamburger exchange transfers chloride from plasma into the RBC as
        #     bicarbonate leaves, increasing *total* RBC chloride;
        # (2) deoxygenated Hb binds more chloride, so measured *free* RBC chloride
        #     can fall from arterial to venous blood (Prange et al.).
        # We therefore conserve total local chloride using free + Hb-bound pools.
        hct = float(np.clip(getattr(state, "hematocrit_fraction", c.baseline_hematocrit), 0.05, 0.80))
        plasma_fraction = 1.0 - hct
        arterial_rbc_free_cl = float(state.rbc_chloride_mmol_l)
        total_rbc_chloride_gain = (plasma_fraction / max(1e-9, hct)) * chloride_shift
        free_cl_drop = min(
            c.rbc_free_chloride_drop_max_mmol_l,
            c.rbc_free_chloride_drop_gain_mmol_l_per_sat_fraction * sat_drop,
        )
        venous_rbc_cl = max(1.0, arterial_rbc_free_cl - free_cl_drop)
        hb_bound_chloride_gain = total_rbc_chloride_gain + free_cl_drop
        chloride_balance = (
            plasma_fraction * (venous_plasma_cl - state.chloride_mmol_l)
            + hct * (
                (venous_rbc_cl - arterial_rbc_free_cl) + hb_bound_chloride_gain
            )
        )

        # Haldane gain at the same venous PCO2/pH: compare actual deoxygenated
        # blood to an otherwise identical oxygenated blood sample.
        oxygenated_same = self.snapshot(
            plasma_ph=vab.ph,
            pco2_mmHg=pv,
            oxygen_sat_fraction=sao2,
            plasma_chloride_mmol_l=venous_plasma_cl,
            hemoglobin_g_dl=float(getattr(state, "hemoglobin_g_dl", self.cfg.hemoglobin_g_dl)),
            hematocrit_fraction=float(getattr(state, "hematocrit_fraction", self.cfg.baseline_hematocrit)),
        )

        state.mixed_venous_pco2_mmHg = float(pv)
        state.mixed_venous_ph = float(vab.ph)
        state.mixed_venous_bicarbonate_mmol_l = float(vab.bicarbonate_mmol_l)
        state.mixed_venous_total_co2_mmol_l_blood = float(venous_content)
        state.mixed_venous_plasma_chloride_mmol_l = float(venous_plasma_cl)
        state.mixed_venous_rbc_chloride_mmol_l = float(venous_rbc_cl)
        state.mixed_venous_hb_bound_chloride_gain_mmol_l_rbc = float(hb_bound_chloride_gain)
        state.chloride_shift_plasma_mmol_l = float(chloride_shift)
        state.chloride_shift_balance_residual_mmol_l_blood = float(chloride_balance)
        state.haldane_co2_content_gain_mmol_l = float(
            venous_content - oxygenated_same.total_co2_mmol_l_blood
        )
        state.co2_fick_content_residual_mmol_l = float(
            (venous_content - state.arterial_total_co2_mmol_l_blood) - av_delta
        )
        return state
