from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, asdict, fields
from typing import Any
import numpy as np

from .config import HumanConfig
from .acid_base import PhysicochemicalAcidBaseModel
from .blood_gas import WholeBloodGasChemistryModel
from .cardiovascular import ClosedLoopCardiovascularModel
from .energy_metabolism import WholeBodyEnergyBalanceModel
from .metabolism_dallaman import DallaManMealModel
from .pbpk import PBPKIntervention, ReferencePBPKModel
from .pulmonary_exchange import MultiCompartmentPulmonaryExchangeModel
from .renal import RenalIntervention, RenalModel
from .respiratory_mechanics import RespiratoryMechanicsModel
from .respiratory_cycle import DynamicRespiratoryCycleModel
from .respiratory import RespiratoryIntervention, RespiratoryModel


STATE_SCHEMA_VERSION = "0.22"


@dataclass
class HumanState:
    # Public metabolic/endocrine state
    glucose_mg_dl: float = 95.0
    insulin_uU_ml: float = 6.0
    glucagon_pg_ml: float = 60.0  # reduced-order readout; not part of Dalla Man core
    gut_carbs_g: float = 0.0
    liver_glycogen_g: float = 90.0
    muscle_glycogen_g: float = 350.0
    lactate_mmol_l: float = 1.0
    lactate_distribution_volume_l: float = 42.0
    initial_lactate_distribution_volume_l: float = 42.0
    lactate_amount_mmol: float = 42.0
    initial_lactate_amount_mmol: float = 42.0
    lactate_generated_mmol: float = 0.0
    lactate_cleared_mmol: float = 0.0
    lactate_mass_balance_error_mmol: float = 0.0
    lactate_production_mmol_min: float = 0.0
    lactate_clearance_mmol_min: float = 0.0
    exercise_lactate_production_mmol_min: float = 0.0
    hypoxic_lactate_production_mmol_min: float = 0.0
    hypoxic_lactate_production_mmol_l_min: float = 0.0

    # Endocrine extensions. SC quantities are OpenHumSim model-units, not
    # clinical insulin units.
    sc_insulin_depot1_model_units: float = 0.0
    sc_insulin_depot2_model_units: float = 0.0
    sc_insulin_absorption_model_units_min: float = 0.0
    sc_insulin_administered_model_units: float = 0.0
    sc_insulin_absorbed_model_units: float = 0.0
    sc_insulin_mass_balance_error_model_units: float = 0.0
    glucagon_counterregulatory_egp_mg_kg_min: float = 0.0
    glucagon_counterregulatory_glucose_released_mg: float = 0.0



    # Dalla Man 2007 internal states / diagnostics
    dalla_gp_mg_kg: float = 0.0
    dalla_gt_mg_kg: float = 0.0
    dalla_il_pmol_kg: float = 0.0
    dalla_ip_pmol_kg: float = 0.0
    dalla_i1_pmol_l: float = 0.0
    dalla_id_pmol_l: float = 0.0
    dalla_ipo_pmol_kg: float = 0.0
    dalla_x_pmol_l: float = 0.0
    dalla_y_pmol_kg_min: float = 0.0
    dalla_qsto1_mg: float = 0.0
    dalla_qsto2_mg: float = 0.0
    dalla_qgut_mg: float = 0.0
    dalla_meal_reference_mg: float = 0.0
    dalla_glucose_ingested_mg: float = 0.0
    dalla_glucose_gi_absorbed_mg: float = 0.0
    dalla_glucose_appeared_mg: float = 0.0
    dalla_ra_mg_kg_min: float = 0.0
    dalla_egp_mg_kg_min: float = 0.0
    dalla_u_mg_kg_min: float = 0.0
    dalla_insulin_secretion_pmol_kg_min: float = 0.0
    dalla_hepatic_extraction: float = 0.0
    dalla_gi_mass_balance_error_mg: float = 0.0
    dalla_numerical_positivity_correction: float = 0.0

    # Cardiovascular/fluid public state
    heart_rate_bpm: float = 65.0
    map_mmHg: float = 90.0
    systolic_pressure_mmHg: float = 120.0
    diastolic_pressure_mmHg: float = 75.0
    cardiac_output_l_min: float = 5.0
    stroke_volume_ml: float = 77.0
    central_venous_pressure_mmHg: float = 5.0
    pulmonary_artery_pressure_mmHg: float = 15.0
    left_ventricular_pressure_mmHg: float = 8.0
    right_ventricular_pressure_mmHg: float = 4.0
    cv_ejection_fraction: float = 0.60
    plasma_volume_l: float = 3.0
    rbc_volume_l: float = 2.0
    hematocrit_fraction: float = 0.40
    hemoglobin_g_dl: float = 14.0
    hemoglobin_mass_g: float = 700.0

    # Conserved 0D circulation compartments
    cv_la_ml: float = 60.0
    cv_lv_ml: float = 140.0
    cv_sa_ml: float = 700.0
    cv_sv_ml: float = 2500.0
    cv_ra_ml: float = 60.0
    cv_rv_ml: float = 140.0
    cv_pa_ml: float = 180.0
    cv_pv_ml: float = 1220.0
    cv_total_blood_volume_ml: float = 5000.0
    cv_blood_volume_error_ml: float = 0.0
    cv_numerical_volume_correction_ml: float = 0.0
    cv_sim_time_s: float = 0.0

    # Respiratory / acid-base
    respiratory_rate_bpm: float = 12.0
    tidal_volume_l: float = 0.50
    alveolar_ventilation_l_min: float = 4.20
    pao2_mmHg: float = 95.0
    paco2_mmHg: float = 40.0
    bicarbonate_mmol_l: float = 24.0
    ph_arterial: float = 7.40
    dissolved_co2_mmol_l: float = 1.204
    carbonate_mmol_l: float = 0.028
    total_co2_mmol_l: float = 25.232

    # Whole-blood / erythrocyte CO2 chemistry
    rbc_ph: float = 7.19
    rbc_bicarbonate_mmol_l: float = 16.0
    rbc_carbonate_mmol_l: float = 0.012
    rbc_chloride_mmol_l: float = 68.0
    carbamino_co2_mmol_l_blood: float = 1.14
    arterial_total_co2_mmol_l_blood: float = 19.9
    hemoglobin_buffer_capacity_mEq_l_pH: float = 20.0
    hemoglobin_bound_proton_change_mEq_l: float = 0.0
    mixed_venous_pco2_mmHg: float = 46.0
    mixed_venous_ph: float = 7.36
    mixed_venous_bicarbonate_mmol_l: float = 25.5
    mixed_venous_total_co2_mmol_l_blood: float = 21.7
    mixed_venous_plasma_chloride_mmol_l: float = 102.0
    mixed_venous_rbc_chloride_mmol_l: float = 69.4
    mixed_venous_hb_bound_chloride_gain_mmol_l_rbc: float = 0.0
    chloride_shift_plasma_mmol_l: float = 1.0
    chloride_shift_balance_residual_mmol_l_blood: float = 0.0
    haldane_co2_content_gain_mmol_l: float = 0.4
    co2_fick_content_residual_mmol_l: float = 0.0
    exchangeable_co2_pool_mmol: float = 210.0
    initial_exchangeable_co2_pool_mmol: float = 210.0
    co2_generated_mmol: float = 0.0
    co2_eliminated_mmol: float = 0.0
    co2_urinary_bicarbonate_loss_mmol: float = 0.0
    co2_mass_balance_error_mmol: float = 0.0
    co2_content_solver_residual_mmol_l: float = 0.0
    co2_final_gas_closure_residual_mmol_l: float = 0.0
    vco2_elimination_ml_min: float = 200.0
    effective_co2_ventilation_l_min: float = 4.2
    albumin_g_dl: float = 4.2
    phosphate_mmol_l: float = 1.0
    plasma_albumin_g: float = 126.0
    ecf_phosphate_mmol: float = 14.0
    nonvolatile_strong_anion_mEq: float = 35.0
    initial_nonvolatile_strong_anion_mEq: float = 35.0
    nonvolatile_strong_anion_mEq_l: float = 2.5
    nonvolatile_acid_generated_mEq: float = 0.0
    nonvolatile_acid_excreted_mEq: float = 0.0
    strong_ion_difference_apparent_mEq_l: float = 40.2
    strong_ion_difference_effective_mEq_l: float = 37.7
    strong_ion_gap_mEq_l: float = 2.5
    albumin_charge_mEq_l: float = 11.7
    phosphate_charge_mEq_l: float = 1.8
    charge_balance_residual_mEq_l: float = 0.0
    henderson_hasselbalch_residual: float = 0.0
    anion_gap_mEq_l: float = 13.0
    albumin_corrected_anion_gap_mEq_l: float = 13.0
    acid_base_solver_iterations: float = 0.0
    spo2_pct: float = 97.5
    vo2_ml_min: float = 250.0
    vo2_demand_ml_min: float = 250.0
    vco2_ml_min: float = 200.0
    vco2_demand_ml_min: float = 200.0
    oxidative_vco2_ml_min: float = 200.0
    vco2_generation_interval_average_ml_min: float = 200.0
    metabolic_respiratory_quotient: float = 0.80
    oxygen_debt_ml_min: float = 0.0
    instantaneous_oxygen_deficit_ml_min: float = 0.0
    cumulative_oxygen_deficit_ml: float = 0.0
    aerobic_fraction: float = 1.0
    ventilation_efficiency: float = 1.0

    # Multi-compartment pulmonary gas exchange
    pulmonary_shunt_fraction: float = 0.005
    pulmonary_vq_log_sd: float = 0.18
    pulmonary_diffusing_capacity_relative: float = 1.0
    pulmonary_mean_vq_ratio: float = 0.84
    pulmonary_low_vq_perfusion_fraction: float = 0.0
    pulmonary_high_vq_ventilation_fraction: float = 0.0
    pulmonary_capillary_transit_time_s: float = 0.75
    pulmonary_diffusion_equilibration_fraction: float = 0.98
    pulmonary_mean_alveolar_pao2_mmHg: float = 100.0
    pulmonary_aa_gradient_mmHg: float = 8.0
    pulmonary_mixed_expired_pco2_mmHg: float = 28.0
    pulmonary_alveolar_dead_space_fraction: float = 0.02
    pulmonary_enghoff_dead_space_fraction: float = 0.30

    # Dynamic regional perfusion and recruitment
    pulmonary_hpv_function_fraction: float = 1.0
    pulmonary_hpv_resistance_multiplier: float = 1.0
    pulmonary_perfusion_redistribution_index: float = 0.0
    pulmonary_hpv_diverted_flow_fraction: float = 0.0
    pulmonary_recruitment_fraction: float = 0.99
    pulmonary_derecruited_fraction: float = 0.01
    pulmonary_peep_cmH2O: float = 0.0
    pulmonary_recruitment_pressure_offset_cmH2O: float = 0.0
    pulmonary_mean_distending_pressure_cmH2O: float = 1.5
    pulmonary_effective_capillary_blood_volume_ml: float = 70.0
    pulmonary_hypoxic_perfusion_fraction: float = 0.0
    pulmonary_recruitment_u0: float = 1.0
    pulmonary_recruitment_u1: float = 1.0
    pulmonary_recruitment_u2: float = 1.0
    pulmonary_recruitment_u3: float = 1.0
    pulmonary_recruitment_u4: float = 1.0
    pulmonary_recruitment_u5: float = 1.0
    pulmonary_hpv_tone_u0: float = 0.0
    pulmonary_hpv_tone_u1: float = 0.0
    pulmonary_hpv_tone_u2: float = 0.0
    pulmonary_hpv_tone_u3: float = 0.0
    pulmonary_hpv_tone_u4: float = 0.0
    pulmonary_hpv_tone_u5: float = 0.0


    # Explicit lung/chest-wall mechanics
    pulmonary_positive_pressure_fraction: float = 0.0
    pulmonary_lung_compliance_scale: float = 1.0
    pulmonary_chest_wall_compliance_scale: float = 1.0
    pulmonary_pleural_pressure_end_exp_cmH2O: float = -5.0
    pulmonary_pleural_pressure_end_insp_cmH2O: float = -5.0
    pulmonary_transpulmonary_pressure_end_exp_cmH2O: float = 5.0
    pulmonary_transpulmonary_pressure_end_insp_cmH2O: float = 10.0
    pulmonary_mean_transpulmonary_pressure_cmH2O: float = 6.75
    pulmonary_passive_equivalent_plateau_pressure_cmH2O: float = 7.5
    pulmonary_airway_driving_pressure_cmH2O: float = 7.5
    pulmonary_transpulmonary_driving_pressure_cmH2O: float = 5.0
    pulmonary_chest_wall_driving_pressure_cmH2O: float = 2.5
    pulmonary_lung_compliance_l_cmH2O: float = 0.10
    pulmonary_chest_wall_compliance_l_cmH2O: float = 0.20
    pulmonary_respiratory_system_compliance_l_cmH2O: float = 0.0667
    pulmonary_lung_strain: float = 0.20
    pulmonary_overdistension_fraction: float = 0.02
    pulmonary_intrathoracic_pressure_delta_cmH2O: float = 0.0
    pulmonary_mechanical_pvr_multiplier: float = 1.0
    pulmonary_elastic_work_j_per_breath: float = 0.18
    pulmonary_mechanical_power_j_min: float = 2.2
    pulmonary_pressure_identity_residual_cmH2O: float = 0.0

    # Within-breath dynamic mechanics
    respiratory_drive_target_tidal_volume_l: float = 0.50
    respiratory_airway_resistance_cmH2O_s_l: float = 2.0
    respiratory_inertance_cmH2O_s2_l: float = 0.080
    respiratory_expiratory_resistance_multiplier: float = 1.15
    respiratory_expiratory_flow_limit_l_s: float = 4.0
    respiratory_inspiratory_fraction: float = 0.33
    respiratory_muscle_drive_gain: float = 1.0
    respiratory_ventilator_pressure_control_cmH2O: float = 0.0
    respiratory_cycle_rr_override_bpm: float = 0.0
    respiratory_cycle_target_vt_override_l: float = 0.0
    respiratory_cycle_volume_above_relaxed_l: float = 0.0
    respiratory_cycle_flow_l_s: float = 0.0
    respiratory_cycle_phase_s: float = 0.0
    respiratory_cycle_dynamic_hyperinflation_l: float = 0.0
    respiratory_cycle_auto_peep_cmH2O: float = 0.0
    respiratory_cycle_end_expiratory_alveolar_pressure_cmH2O: float = 0.0
    respiratory_cycle_peak_inspiratory_flow_l_s: float = 0.0
    respiratory_cycle_peak_expiratory_flow_l_s: float = 0.0
    respiratory_cycle_end_expiratory_flow_l_s: float = 0.0
    respiratory_cycle_peak_muscle_pressure_cmH2O: float = 0.0
    respiratory_cycle_peak_airway_pressure_cmH2O: float = 0.0
    respiratory_cycle_resistive_work_j_breath: float = 0.0
    respiratory_cycle_muscle_work_j_breath: float = 0.0
    respiratory_cycle_ventilator_work_j_breath: float = 0.0
    respiratory_cycle_pv_hysteresis_j_breath: float = 0.0
    respiratory_cycle_total_work_j_breath: float = 0.0
    respiratory_cycle_expiratory_flow_limited_fraction: float = 0.0
    respiratory_cycle_time_constant_s: float = 0.13
    respiratory_cycle_equation_residual_cmH2O: float = 0.0
    respiratory_cycle_flow_limiting_pressure_cmH2O: float = 0.0

    # Patient-ventilator interaction
    # mode: 0 spontaneous/legacy, 1 pressure-control, 2 pressure-support
    respiratory_ventilator_mode_code: int = 0
    respiratory_pressure_support_cmH2O: float = 8.0
    respiratory_trigger_pressure_cmH2O: float = 0.5
    respiratory_trigger_flow_l_s: float = 0.10
    respiratory_cycleoff_fraction_peak_flow: float = 0.25
    respiratory_pressure_support_rise_time_s: float = 0.15
    respiratory_pressure_support_max_ti_s: float = 3.0
    respiratory_neural_inspiratory_fraction: float = 0.30
    respiratory_leak_flow_l_s: float = 0.0
    respiratory_allow_retrigger_same_effort: float = 0.0
    respiratory_ventilator_active: float = 0.0
    respiratory_ventilator_inspiration_elapsed_s: float = 0.0
    respiratory_ventilator_peak_flow_l_s: float = 0.0
    respiratory_ventilator_refractory_remaining_s: float = 0.0
    respiratory_ventilator_triggered_current_effort: float = 0.0
    respiratory_ventilator_triggers_current_effort: float = 0.0
    respiratory_ventilator_last_trigger_delay_s: float = 0.0
    respiratory_ventilator_mean_trigger_delay_s: float = 0.0
    respiratory_ventilator_mean_cycling_delay_s: float = 0.0
    respiratory_ventilator_ineffective_trigger_fraction: float = 0.0
    respiratory_ventilator_double_trigger_fraction: float = 0.0
    respiratory_ventilator_premature_cycling_fraction: float = 0.0
    respiratory_ventilator_delayed_cycling_fraction: float = 0.0
    respiratory_ventilator_autotrigger_fraction: float = 0.0
    respiratory_ventilator_asynchrony_index_pct: float = 0.0
    respiratory_ventilator_patient_efforts_per_min: float = 0.0
    respiratory_ventilator_breaths_per_min: float = 0.0
    respiratory_ventilator_trigger_pressure_time_product_cmH2O_s: float = 0.0
    respiratory_ventilator_neural_inspiratory_time_s: float = 0.0

    # Whole-body oxygen transport diagnostics with Fick coupling
    arterial_o2_content_ml_dl: float = 18.5
    mixed_venous_o2_content_ml_dl: float = 13.5
    mixed_venous_o2_sat_pct: float = 72.0
    oxygen_delivery_ml_min: float = 925.0
    oxygen_extraction_ratio: float = 0.27
    oxygen_supply_margin_ml_min: float = 675.0

    # Renal / electrolytes / water
    total_body_water_l: float = 42.0
    ecf_volume_l: float = 14.0
    icf_volume_l: float = 28.0
    icf_effective_osmoles_mOsm: float = 7980.0
    ecf_effective_tonicity_mOsm_l: float = 285.0
    icf_effective_tonicity_mOsm_l: float = 285.0
    osmotic_water_shift_l_min: float = 0.0
    ecf_sodium_mmol: float = 1960.0
    ecf_potassium_mmol: float = 58.8
    ecf_chloride_mmol: float = 1442.0
    icf_potassium_mmol: float = 3400.0
    potassium_transcellular_flux_mmol_min: float = 0.0
    potassium_transcellular_target_mmol_l: float = 4.2
    sodium_mmol_l: float = 140.0
    potassium_mmol_l: float = 4.2
    chloride_mmol_l: float = 103.0
    plasma_osmolality_mOsm_kg: float = 294.0
    renal_function_fraction: float = 1.0
    gfr_ml_min: float = 120.0
    adh_relative: float = 1.0
    renin_relative: float = 1.0
    angiotensin_ii_relative: float = 1.0
    aldosterone_relative: float = 1.0
    urine_flow_ml_min: float = 1.0
    urine_sodium_mmol_min: float = 0.070
    urine_potassium_mmol_min: float = 0.040
    urine_chloride_mmol_min: float = 0.070
    urine_ammonium_mmol_min: float = 0.024
    urine_titratable_acid_mmol_min: float = 0.016
    urine_bicarbonate_mmol_min: float = 0.0
    renal_acid_excretion_mmol_min: float = 0.040

    # Fluid/electrolyte ledger
    initial_total_body_water_l: float = 42.0
    initial_total_exchangeable_potassium_mmol: float = 3458.8
    initial_ecf_sodium_mmol: float = 1960.0
    initial_ecf_chloride_mmol: float = 1442.0
    water_administered_l: float = 0.0
    sodium_administered_mmol: float = 0.0
    chloride_administered_mmol: float = 0.0
    water_lost_l: float = 0.0
    sodium_lost_mmol: float = 0.0
    potassium_lost_mmol: float = 0.0
    chloride_lost_mmol: float = 0.0

    # PBPK
    probe_gut_mg: float = 0.0
    probe_plasma_mg: float = 0.0
    probe_liver_mg: float = 0.0
    probe_kidney_mg: float = 0.0
    probe_muscle_mg: float = 0.0
    probe_adipose_mg: float = 0.0
    probe_rest_mg: float = 0.0
    probe_plasma_mg_l: float = 0.0
    probe_liver_mg_l: float = 0.0
    probe_kidney_mg_l: float = 0.0
    probe_effect_site_mg_l: float = 0.0
    probe_total_body_mg: float = 0.0
    probe_administered_mg: float = 0.0
    probe_eliminated_hepatic_mg: float = 0.0
    probe_eliminated_renal_mg: float = 0.0
    probe_mass_balance_error_mg: float = 0.0
    probe_hepatic_clearance_l_min: float = 0.0
    probe_renal_clearance_l_min: float = 0.0
    probe_total_tissue_flow_l_min: float = 5.0

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    def to_versioned_payload(self) -> dict[str, Any]:
        """Serialize state with an exact schema envelope.

        ``as_dict`` remains the numeric debug view used by observations and
        finite checks. Persistence must use this envelope so a pre-v0.22
        concentration-only lactate state cannot be mistaken for an amount-
        conserving state whose omitted fields silently take current defaults.
        """
        return {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "state": self.as_dict(),
        }

    @classmethod
    def from_versioned_payload(cls, payload: Mapping[str, Any]) -> "HumanState":
        """Load only an exact v0.22 state payload; reject legacy ambiguity."""
        if not isinstance(payload, Mapping):
            raise TypeError("state payload must be a mapping")
        version = payload.get("state_schema_version")
        if version != STATE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported state_schema_version "
                f"{version!r}; expected {STATE_SCHEMA_VERSION!r}. "
                "Pre-v0.22 states require an explicit migration because "
                "lactate changed from concentration-only to an amount ledger."
            )
        raw_state = payload.get("state")
        if not isinstance(raw_state, Mapping):
            raise TypeError("versioned state payload must contain a 'state' mapping")
        expected = {field.name for field in fields(cls)}
        supplied = set(raw_state)
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        if missing or extra:
            raise ValueError(
                "state payload fields do not exactly match schema 0.22: "
                f"missing={missing}, extra={extra}"
            )
        values: dict[str, float] = {}
        for name in expected:
            raw_value = raw_state[name]
            if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
                raw_value,
                (int, float, np.integer, np.floating),
            ):
                raise TypeError(f"state field {name!r} must be a real number")
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(f"state field {name!r} must be finite")
            values[name] = value
        return cls(**values)


@dataclass
class Intervention:
    insulin_model_units: float = 0.0
    oral_carbs_g: float = 0.0
    exercise_intensity: float = 0.0
    saline_ml: float = 0.0
    fio2: float = 0.21
    ventilation_pressure_assist_cmH2O: float = 0.0
    # Deprecated compatibility field. Integrated HumanPhysiology does not
    # convert this directly to alveolar ventilation.
    ventilation_support_l_min: float = 0.0
    oral_water_ml: float = 0.0
    oral_probe_mg: float = 0.0


class HumanPhysiology:
    """Coupled multi-organ model with whole-body O2 transport diagnostics."""

    def __init__(self, config: HumanConfig):
        self.cfg = config
        self.metabolism = DallaManMealModel(config)
        self.energy_metabolism = WholeBodyEnergyBalanceModel(config)
        self.cardiovascular = ClosedLoopCardiovascularModel(config)
        self.respiratory = RespiratoryModel(config)
        self.acid_base = PhysicochemicalAcidBaseModel(config)
        self.pulmonary_exchange = MultiCompartmentPulmonaryExchangeModel(config)
        self.respiratory_mechanics = RespiratoryMechanicsModel(config)
        self.respiratory_cycle = DynamicRespiratoryCycleModel(config)
        self.blood_gas = WholeBloodGasChemistryModel(
            config, self.acid_base, pulmonary_exchange=self.pulmonary_exchange
        )
        self.renal = RenalModel(config)
        self.pbpk = ReferencePBPKModel(config)

    def initialize_state(self, state: HumanState) -> HumanState:
        self.metabolism.initialize_state(state)
        self.renal.initialize_state(state)
        self.energy_metabolism.initialize_state(state)
        self._update_blood_composition(state, initialize=True)
        self.acid_base.initialize_state(state)
        self.respiratory_mechanics.step(state, dt_min=None)
        self.respiratory_cycle.initialize_state(state)
        self.respiratory.update_metabolic_gas_production(state, exercise=0.0)
        state.vco2_generation_interval_average_ml_min = float(state.vco2_ml_min)
        self.blood_gas.initialize_state(state)
        self.cardiovascular.initialize_state(state)
        self.pbpk.refresh_concentrations_with_config(state)
        return self._clip_state(state)

    def runtime_snapshot(self) -> dict[str, object]:
        """Snapshot private solver dynamics not represented in ``HumanState``."""
        return {
            "cardiovascular": self.cardiovascular.runtime_snapshot(),
            "respiratory_cycle": self.respiratory_cycle.runtime_snapshot(),
        }

    def restore_runtime_snapshot(self, snapshot: dict[str, object]) -> None:
        self.cardiovascular.restore_runtime_snapshot(snapshot["cardiovascular"])
        self.respiratory_cycle.restore_runtime_snapshot(snapshot["respiratory_cycle"])

    def _update_blood_composition(self, state: HumanState, initialize: bool = False) -> HumanState:
        """Short-horizon conserved RBC/Hb mass with plasma-volume dilution.

        The episode model assumes no erythropoiesis, hemolysis or transfusion.
        Plasma-volume changes therefore alter hematocrit and [Hb] while RBC and
        hemoglobin mass remain constant.
        """
        c = self.cfg
        if initialize or float(getattr(state, "hemoglobin_mass_g", 0.0)) <= 0.0:
            baseline_blood_l = max(1e-6, c.cv_baseline_blood_volume_ml / 1000.0)
            baseline_rbc_l = max(0.25, baseline_blood_l - c.plasma_volume_baseline_l)
            state.rbc_volume_l = float(baseline_rbc_l)
            state.hemoglobin_mass_g = float(c.hemoglobin_g_dl * baseline_blood_l * 10.0)
        blood_l = max(1e-6, float(state.plasma_volume_l) + float(state.rbc_volume_l))
        state.hematocrit_fraction = float(np.clip(state.rbc_volume_l / blood_l, 0.05, 0.80))
        state.hemoglobin_g_dl = float(max(0.0, state.hemoglobin_mass_g / (blood_l * 10.0)))
        return state

    def apply_instant_intervention(self, state: HumanState, intervention: Intervention) -> HumanState:
        self.metabolism.add_meal(state, intervention.oral_carbs_g)
        self.metabolism.add_exogenous_insulin(state, intervention.insulin_model_units)
        self.renal.apply_instant_intervention(
            state, RenalIntervention(saline_ml=intervention.saline_ml, oral_water_ml=intervention.oral_water_ml)
        )
        self.energy_metabolism.refresh_concentration(state)
        self._update_blood_composition(state)
        self.pbpk.apply_instant_intervention(state, PBPKIntervention(oral_probe_mg=intervention.oral_probe_mg))
        self.pbpk.refresh_concentrations_with_config(state)
        return self._clip_state(state)

    def integrate(self, state: HumanState, intervention: Intervention, duration_min: float) -> HumanState:
        remaining = float(duration_min)
        while remaining > 1e-9:
            dt = min(self.cfg.integration_step_min, remaining)
            state = self._substep(state, intervention, dt)
            remaining -= dt
        return state

    def _substep(self, s: HumanState, intervention: Intervention, dt: float) -> HumanState:
        c = self.cfg
        exercise = float(np.clip(intervention.exercise_intensity, 0.0, 1.0))
        vco2_at_interval_start = max(0.0, float(s.vco2_ml_min))

        # The reduced glucagon counterregulation controller is evaluated
        # before the Dalla Man step so the current hormone state can influence the
        # separately-labelled counterregulatory EGP extension.
        g = float(s.glucose_mg_dl)
        hypo_drive = 1.0 / (
            1.0 + np.exp(
                (g - c.glucagon_hypoglycemia_midpoint_mg_dl)
                / max(1e-6, c.glucagon_hypoglycemia_slope_mg_dl)
            )
        )
        glucagon_target = (
            c.glucagon_baseline_pg_ml
            + c.glucagon_hypoglycemia_max_increment_pg_ml * hypo_drive
            + 15.0 * exercise
        )
        # Hyperinsulinemia suppresses glucagon only outside clear hypoglycemia;
        # during hypoglycemia the counterregulatory drive remains dominant.
        if g > 80.0:
            glucagon_target -= 0.20 * max(0.0, s.insulin_uU_ml - 8.0)
        s.glucagon_pg_ml += (
            np.clip(glucagon_target, 20.0, 220.0) - s.glucagon_pg_ml
        ) * min(1.0, dt / c.glucagon_tau_min)

        # Published Dalla Man meal glucose-insulin core plus explicitly labelled
        # SC-insulin and hypoglycemic glucagon extensions.
        self.metabolism.step(s, exercise=exercise, dt_min=dt)

        # Lactate is integrated as an amount. Exercise contributes the glycolytic
        # source and *previous-step* unmet O2 demand contributes a
        # separately labelled hypoxic source. The explicit lag is intentional.
        previous_oxygen_deficit = max(0.0, float(s.oxygen_debt_ml_min))
        s.muscle_glycogen_g -= c.exercise_muscle_glycogen_g_min * exercise * dt
        self.energy_metabolism.step_lactate(s, exercise=exercise, dt_min=dt)

        # Kidney uses the previous-step pressure; this is explicit operator splitting.
        self.renal.step(s, exercise=exercise, dt=dt)
        # Renal/extrarenal water flux changes concentration but not lactate amount.
        self.energy_metabolism.refresh_concentration(s)
        self._update_blood_composition(s)

        # Ventilatory drive sets RR/VT. Lung/chest-wall mechanics are resolved
        # before circulation so positive intrathoracic pressure and the
        # U-shaped lung-volume contribution to PVR can feed the current CV step.
        self.respiratory.update_mechanics(
            s,
            RespiratoryIntervention(
                fio2=intervention.fio2,
                ventilation_support_l_min=0.0,
            ),
            exercise=exercise, dt=dt,
        )
        # First-pass static mechanics estimates current elastance from recruitment.
        self.respiratory_mechanics.step(s, dt_min=dt)
        # Resolve actual within-breath flow/volume from the full equation of motion.
        base_positive_pressure = float(np.clip(
            s.pulmonary_positive_pressure_fraction, 0.0, 1.0
        ))
        pressure_assist = max(
            0.0, float(intervention.ventilation_pressure_assist_cmH2O)
        )
        passive_pressure_need = max(
            1e-6, float(s.pulmonary_airway_driving_pressure_cmH2O)
        )
        action_positive_pressure_fraction = float(np.clip(
            pressure_assist / passive_pressure_need, 0.0, 1.0
        ))
        # Pressure assistance shares the inspiratory load continuously with the
        # patient's muscles.  The contribution is normalized by the current
        # passive-equivalent pressure requirement, so an infinitesimal assist
        # cannot switch the whole breath to positive-pressure hemodynamics.
        s.pulmonary_positive_pressure_fraction = float(
            base_positive_pressure
            + (1.0 - base_positive_pressure) * action_positive_pressure_fraction
        )
        # The within-breath solver weights its pressure-source amplitude by this
        # fraction.  Normalize the request so the intervention remains an actual
        # airway-pressure amplitude while the source share independently governs
        # pleural transmission and heart-lung interaction.
        cycle_pressure_assist = pressure_assist / max(
            np.finfo(float).tiny, s.pulmonary_positive_pressure_fraction
        )
        self.respiratory_cycle.step(
            s, dt_min=dt,
            ventilation_pressure_assist_cmH2O=cycle_pressure_assist,
        )
        # Recompute transpulmonary/static diagnostics from the dynamically achieved VT
        # and any intrinsic PEEP/dynamic hyperinflation.
        self.respiratory_mechanics.step(s, dt_min=dt)

        # Circulation sees current fluid volume and current intrathoracic mechanics.
        self.cardiovascular.step(s, exercise=exercise, dt_min=dt)
        # PBPK is perfusion-coupled to generated cardiac output and current GFR.
        self.pbpk.step(s, exercise=exercise, dt=dt)
        self.pbpk.refresh_concentrations_with_config(s)

        # Advance recruitment/HPV exactly once using the start-of-interval
        # PaCO2, then obtain a current oxygen-transport predictor. Subsequent
        # gas-coupling iterations use dt=0 and therefore cannot advance those
        # temporal lung states a second time.
        self.pulmonary_exchange.estimate_arterial_oxygen(
            s, pco2_mmHg=s.paco2_mmHg, fio2=intervention.fio2, exercise=exercise,
            dt_min=dt, apply=True
        )
        self.respiratory.update_oxygen_transport(s)
        self.respiratory.update_metabolic_gas_production(s, exercise=exercise)
        carbon_start = self.blood_gas.capture_carbon_ledger_start(s)
        vco2_endpoint_guess = float(s.vco2_ml_min)

        # Solve one joint fixed point for: final achieved VO2/oxidative VCO2,
        # interval CO2 generation, final PaCO2-dependent elimination, pulmonary
        # oxygenation, the Haldane content shift, and mixed-venous return.  Every
        # carbon candidate replaces the previous one from ``carbon_start``;
        # cumulative ledgers are therefore committed once, not once per
        # numerical iteration.
        closure = float("inf")
        vco2_endpoint_residual = float("inf")
        elimination_endpoint_residual = float("inf")
        pulmonary_o2_residual = float("inf")
        coupled_gas_tolerance = max(
            1e-6, 10.0 * c.co2_pool_solver_tolerance_mmol_l
        )
        for _ in range(32):
            generation_average = 0.5 * (
                vco2_at_interval_start + vco2_endpoint_guess
            )
            self.blood_gas.step_arterial_carbon_balance(
                s,
                fio2=intervention.fio2,
                exercise=exercise,
                dt_min=dt,
                ledger_start=carbon_start,
                generation_average_ml_min=generation_average,
            )

            # This call is algebraic (dt=0): it cannot progress HPV or
            # recruitment. Reconciliation changes speciation/pH/PaCO2 at the
            # candidate conserved pool but never adds buffer-derived CO2.
            self.pulmonary_exchange.estimate_arterial_oxygen(
                s,
                pco2_mmHg=s.paco2_mmHg,
                fio2=intervention.fio2,
                exercise=exercise,
                dt_min=0.0,
                apply=True,
            )
            closure = self.blood_gas.arterial_carbon_pool_closure_residual_mmol_l(
                s, fio2=intervention.fio2, exercise=exercise
            )
            if abs(closure) > c.co2_pool_solver_tolerance_mmol_l:
                self.blood_gas.reconcile_arterial_carbon_pool(
                    s, fio2=intervention.fio2, exercise=exercise
                )
            # Always evaluate the residual *after* the possible reconcile; the
            # capped path must not retain a pre-correction diagnostic.
            closure = self.blood_gas.arterial_carbon_pool_closure_residual_mmol_l(
                s, fio2=intervention.fio2, exercise=exercise
            )

            self.respiratory.update_oxygen_transport(s)
            self.respiratory.update_metabolic_gas_production(
                s, exercise=exercise
            )
            self.blood_gas.update_venous_diagnostics(s)

            final_vco2 = float(s.vco2_ml_min)
            vco2_endpoint_residual = final_vco2 - vco2_endpoint_guess
            expected_elimination = (
                max(0.0, float(s.paco2_mmHg))
                * max(0.0, float(s.effective_co2_ventilation_l_min))
                / 0.863
            )
            elimination_endpoint_residual = (
                expected_elimination - float(s.vco2_elimination_ml_min)
            )

            # A read-only lung evaluation after the venous update exposes the
            # remaining O2/Fick mismatch without mutating final gas fields.
            pulmonary_check = self.pulmonary_exchange.estimate_arterial_oxygen(
                s,
                pco2_mmHg=s.paco2_mmHg,
                fio2=intervention.fio2,
                exercise=exercise,
                dt_min=0.0,
                apply=False,
            )
            pulmonary_o2_residual = (
                float(pulmonary_check.pao2_mmHg) - float(s.pao2_mmHg)
            )

            vco2_tolerance = 1e-5 * max(1.0, abs(final_vco2))
            elimination_tolerance = 1e-5 * max(1.0, abs(expected_elimination))
            if (
                abs(closure) <= coupled_gas_tolerance
                and abs(vco2_endpoint_residual) <= vco2_tolerance
                and abs(elimination_endpoint_residual) <= elimination_tolerance
                and abs(pulmonary_o2_residual) <= 1e-4
            ):
                break
            vco2_endpoint_guess = final_vco2
        else:
            raise FloatingPointError(
                "coupled O2/CO2/VCO2 fixed point failed: "
                f"carbon={closure!r} mmol/L, "
                f"generation_endpoint={vco2_endpoint_residual!r} mL/min, "
                f"elimination_endpoint={elimination_endpoint_residual!r} mL/min, "
                f"pulmonary_O2={pulmonary_o2_residual!r} mmHg"
            )

        s.co2_final_gas_closure_residual_mmol_l = float(closure)
        if abs(closure) > coupled_gas_tolerance:
            raise FloatingPointError(
                f"coupled arterial O2/CO2 closure failed: {closure!r} mmol/L"
            )

        # The final O2/Fick and metabolic-gas state was part of the joint solve.
        self.energy_metabolism.accumulate_oxygen_deficit(
            s, previous_deficit_ml_min=previous_oxygen_deficit, dt_min=dt
        )
        # Action-driven positive pressure applies only during this integration
        # interval; scenario-defined positive pressure remains persistent.
        s.pulmonary_positive_pressure_fraction = base_positive_pressure
        return self._clip_state(s)

    @staticmethod
    def _clip_state(s: HumanState) -> HumanState:
        # Derived/non-conserved variables get bounded for numerical safety. Conserved
        # water/electrolyte/CV/PBPK masses are never upper-clipped.
        # Reject non-finite values before clipping: np.clip/max can otherwise turn
        # an overflow into an apparently plausible boundary value and hide the
        # numerical failure from the caller.
        def require_finite_state() -> None:
            for name, value in vars(s).items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    if not np.isfinite(value):
                        raise FloatingPointError(
                            f"non-finite HumanState field {name}: {value!r}"
                        )

        require_finite_state()
        s.glucose_mg_dl = float(max(0.0, s.glucose_mg_dl))
        s.insulin_uU_ml = float(max(0.0, s.insulin_uU_ml))
        s.glucagon_pg_ml = float(np.clip(s.glucagon_pg_ml, 0.0, 500.0))
        s.gut_carbs_g = float(max(0.0, s.gut_carbs_g))
        s.liver_glycogen_g = float(max(0.0, s.liver_glycogen_g))
        s.muscle_glycogen_g = float(max(0.0, s.muscle_glycogen_g))
        if s.lactate_amount_mmol < -1e-9:
            raise FloatingPointError(
                f"negative conserved lactate amount: {s.lactate_amount_mmol!r}"
            )
        s.heart_rate_bpm = float(np.clip(s.heart_rate_bpm, 20.0, 260.0))
        s.map_mmHg = float(max(0.0, s.map_mmHg))
        s.plasma_volume_l = float(max(1e-9, s.plasma_volume_l))
        s.respiratory_rate_bpm = float(np.clip(s.respiratory_rate_bpm, 2.0, 80.0))
        s.tidal_volume_l = float(np.clip(s.tidal_volume_l, 0.10, 3.0))
        s.alveolar_ventilation_l_min = float(np.clip(s.alveolar_ventilation_l_min, 0.2, 80.0))
        s.pao2_mmHg = float(np.clip(s.pao2_mmHg, 0.0, 700.0))
        s.paco2_mmHg = float(np.clip(
            s.paco2_mmHg,
            WholeBloodGasChemistryModel.ARTERIAL_PCO2_MIN_MMHG,
            WholeBloodGasChemistryModel.ARTERIAL_PCO2_MAX_MMHG,
        ))
        s.bicarbonate_mmol_l = float(max(0.0, s.bicarbonate_mmol_l))
        s.ph_arterial = float(s.ph_arterial)
        s.spo2_pct = float(np.clip(s.spo2_pct, 0.0, 100.0))
        s.vo2_ml_min = float(np.clip(s.vo2_ml_min, 0.0, 5000.0))
        s.vo2_demand_ml_min = float(np.clip(s.vo2_demand_ml_min, 0.0, 5000.0))
        s.vco2_ml_min = float(np.clip(s.vco2_ml_min, 0.0, 5000.0))
        s.vco2_demand_ml_min = float(np.clip(s.vco2_demand_ml_min, 0.0, 5000.0))
        s.oxidative_vco2_ml_min = float(np.clip(s.oxidative_vco2_ml_min, 0.0, 5000.0))
        s.vco2_generation_interval_average_ml_min = float(
            np.clip(s.vco2_generation_interval_average_ml_min, 0.0, 5000.0)
        )
        s.metabolic_respiratory_quotient = float(
            np.clip(s.metabolic_respiratory_quotient, 0.60, 1.0)
        )
        s.oxygen_debt_ml_min = float(max(0.0, s.oxygen_debt_ml_min))
        s.instantaneous_oxygen_deficit_ml_min = float(
            max(0.0, s.instantaneous_oxygen_deficit_ml_min)
        )
        s.cumulative_oxygen_deficit_ml = float(
            max(0.0, s.cumulative_oxygen_deficit_ml)
        )
        s.aerobic_fraction = float(np.clip(s.aerobic_fraction, 0.0, 1.0))
        s.hematocrit_fraction = float(np.clip(s.hematocrit_fraction, 0.05, 0.80))
        s.hemoglobin_g_dl = float(max(0.0, s.hemoglobin_g_dl))
        s.ventilation_efficiency = float(np.clip(s.ventilation_efficiency, 0.20, 1.20))
        s.arterial_o2_content_ml_dl = float(max(0.0, s.arterial_o2_content_ml_dl))
        s.mixed_venous_o2_content_ml_dl = float(max(0.0, s.mixed_venous_o2_content_ml_dl))
        s.mixed_venous_o2_sat_pct = float(np.clip(s.mixed_venous_o2_sat_pct, 0.0, 100.0))
        s.oxygen_delivery_ml_min = float(max(0.0, s.oxygen_delivery_ml_min))
        s.oxygen_extraction_ratio = float(max(0.0, s.oxygen_extraction_ratio))
        s.oxygen_supply_margin_ml_min = float(s.oxygen_supply_margin_ml_min)

        volume = max(1e-9, float(s.ecf_volume_l))
        s.sodium_mmol_l = float(s.ecf_sodium_mmol / volume)
        s.potassium_mmol_l = float(s.ecf_potassium_mmol / volume)
        s.chloride_mmol_l = float(s.ecf_chloride_mmol / volume)
        s.renal_function_fraction = float(np.clip(s.renal_function_fraction, 0.03, 1.20))
        s.gfr_ml_min = float(np.clip(s.gfr_ml_min, 1.0, 200.0))
        for attr in ("adh_relative","renin_relative","angiotensin_ii_relative","aldosterone_relative"):
            setattr(s, attr, float(np.clip(getattr(s, attr), 0.10, 15.0)))
        s.urine_flow_ml_min = float(np.clip(s.urine_flow_ml_min, 0.02, 20.0))

        for attr in ("probe_gut_mg","probe_plasma_mg","probe_liver_mg","probe_kidney_mg","probe_muscle_mg","probe_adipose_mg","probe_rest_mg","probe_plasma_mg_l","probe_liver_mg_l","probe_kidney_mg_l","probe_effect_site_mg_l","probe_total_body_mg","probe_administered_mg","probe_eliminated_hepatic_mg","probe_eliminated_renal_mg","probe_hepatic_clearance_l_min","probe_renal_clearance_l_min"):
            setattr(s, attr, float(max(0.0, getattr(s, attr))))
        require_finite_state()
        return s
