from __future__ import annotations
from dataclasses import dataclass, field, fields
from math import isclose, isfinite
from numbers import Integral, Real
from warnings import warn


@dataclass(frozen=True)
class HumanConfig:
    # Simulation clock
    agent_step_min: float = 5.0
    integration_step_min: float = 0.25
    episode_minutes: float = 12.0 * 60.0

    # Reference physiology (nominal ~70 kg adult scaffold)
    body_weight_kg: float = 70.0
    glucose_setpoint_mg_dl: float = 95.0
    insulin_baseline_uU_ml: float = 6.0
    glucagon_baseline_pg_ml: float = 60.0
    liver_glycogen_baseline_g: float = 90.0
    muscle_glycogen_baseline_g: float = 350.0
    plasma_volume_baseline_l: float = 3.0
    ecf_volume_baseline_l: float = 14.0
    total_body_water_baseline_l: float = 42.0
    baseline_bun_mg_dl: float = 14.0
    baseline_chloride_mmol_l: float = 103.0

    # Gut / glucose dynamics -- reduced-order, not a validated diabetes model
    gut_absorption_tau_min: float = 55.0
    glucose_distribution_dl: float = 440.0
    basal_hgp_mg_dl_min: float = 1.10
    basal_glucose_use_mg_dl_min: float = 0.65
    insulin_glucose_use_coeff: float = 0.030
    exercise_glucose_use_mg_dl_min: float = 1.30

    # Dalla Man 2007 meal glucose-insulin core (normal-subject parameter set)
    dalla_internal_step_min: float = 0.05
    dalla_basal_glucose_mg_dl: float = 95.0
    dalla_uU_ml_per_insulin_model_unit: float = 6.0
    # Subcutaneous insulin pharmacokinetics.
    # Two serial first-order SC depots: for an isolated bolus the absorption-rate
    # peak occurs at approximately sc_insulin_tmax_min. Units remain OpenHumSim
    # model-units, NOT clinical insulin units.
    sc_insulin_tmax_min: float = 90.0

    # Reduced glucagon counterregulation around the Dalla Man core.
    glucagon_hypoglycemia_midpoint_mg_dl: float = 70.0
    glucagon_hypoglycemia_slope_mg_dl: float = 6.0
    glucagon_hypoglycemia_max_increment_pg_ml: float = 105.0
    glucagon_counterreg_egp_max_mg_kg_min: float = 1.0
    glucagon_counterreg_glucose_gate_mg_dl: float = 82.0

    # Exercise multiplier is an OpenHumSim extension; 2007 core itself is resting/postprandial.
    dalla_exercise_vmax_gain: float = 1.25
    # UQ/calibration multipliers around the published normal parameter set.
    # 1.0 preserves the Dalla Man 2007 parameterization exactly.
    dalla_insulin_sensitivity_scale: float = 1.0
    dalla_gastric_absorption_scale: float = 1.0

    # Deprecated endocrine parameters retained for configuration compatibility.
    insulin_clearance_tau_min: float = 40.0
    insulin_glucose_secretion_coeff: float = 0.010
    insulin_low_glucose_suppression_coeff: float = 0.010
    glucagon_tau_min: float = 18.0

    # Glycogen / exercise
    exercise_muscle_glycogen_g_min: float = 0.22
    liver_glycogen_hgp_fraction: float = field(
        default=0.18,
        metadata={
            "deprecated": (
                "ignored compatibility field; hepatic glucose production is "
                "defined by the Dalla Man core and the explicit glucagon "
                "counterregulation pathway"
            )
        },
    )

    # Fluid balance
    basal_fluid_loss_ml_min: float = 0.30
    exercise_fluid_loss_ml_min: float = 2.2
    baseline_sweat_na_mmol_min: float = 0.002
    baseline_sweat_k_mmol_min: float = 0.0005
    baseline_sweat_cl_mmol_min: float = 0.002

    # Acute water/tonicity and transcellular potassium coupling.
    osmotic_water_equilibration_tau_min: float = 1.0
    potassium_transcellular_tau_min: float = 12.0
    potassium_insulin_shift_max_mmol_l: float = 0.55
    potassium_insulin_half_effect_uU_ml: float = 28.0
    potassium_acid_shift_mmol_l_per_0p1_ph: float = 0.45
    potassium_alkali_shift_mmol_l_per_0p1_ph: float = 0.30
    potassium_exercise_release_max_mmol_l: float = 0.65

    # Closed-loop 0D cardiovascular model (mL, mmHg, seconds)
    cv_internal_step_s: float = 0.02
    cv_warmup_min: float = 3.0
    cv_baseline_blood_volume_ml: float = 5000.0
    cv_baseline_cardiac_output_l_min: float = 5.0
    cv_resting_hr_bpm: float = 65.0
    cv_map_setpoint_mmHg: float = 90.0
    cv_systolic_fraction: float = 0.35
    cv_hr_tau_min: float = 0.20
    cv_exercise_hr_gain_bpm: float = 85.0
    cv_baroreflex_hr_gain: float = 0.18
    cv_baroreflex_resistance_gain: float = 0.20
    cv_angII_resistance_gain: float = 0.08
    cv_exercise_systemic_vasodilation: float = 0.52
    cv_contractility_exercise_gain: float = 1.00
    cv_exercise_venous_recruitment_ml: float = 500.0

    # Baseline compartment volumes; sum = 5000 mL
    cv_v_la0_ml: float = 60.0
    cv_v_lv0_ml: float = 140.0
    cv_v_sa0_ml: float = 700.0
    cv_v_sv0_ml: float = 2500.0
    cv_v_ra0_ml: float = 60.0
    cv_v_rv0_ml: float = 140.0
    cv_v_pa0_ml: float = 180.0
    cv_v_pv0_ml: float = 1220.0

    # Unstressed volumes / compliances
    cv_v0_la_ml: float = 20.0
    cv_v0_lv_ml: float = 10.0
    cv_v0_sa_ml: float = 565.0
    cv_v0_sv_ml: float = 2000.0
    cv_v0_ra_ml: float = 20.0
    cv_v0_rv_ml: float = 15.0
    cv_v0_pa_ml: float = 105.0
    cv_v0_pv_ml: float = 1060.0
    cv_c_la_ml_mmHg: float = 5.0
    cv_c_sa_ml_mmHg: float = 1.80
    cv_c_sv_ml_mmHg: float = 100.0
    cv_c_ra_ml_mmHg: float = 8.0
    cv_c_pa_ml_mmHg: float = 5.0
    cv_c_pv_ml_mmHg: float = 20.0

    # Ventricular elastances (mmHg/mL)
    cv_lv_emin: float = 0.06
    cv_lv_emax: float = 2.30
    cv_rv_emin: float = 0.03
    cv_rv_emax: float = 0.80

    # Hydraulic resistances (mmHg*s/mL)
    cv_r_mitral_mmHg_s_ml: float = 0.020
    cv_r_aortic_mmHg_s_ml: float = 0.020
    cv_r_tricuspid_mmHg_s_ml: float = 0.020
    cv_r_pulmonic_mmHg_s_ml: float = 0.020
    cv_r_systemic_mmHg_s_ml: float = 1.13
    cv_r_systemic_venous_mmHg_s_ml: float = 0.015
    cv_r_pulmonary_mmHg_s_ml: float = 0.085
    cv_r_pulmonary_venous_mmHg_s_ml: float = 0.015

    # Respiratory / acid-base reference parameters
    atmospheric_pressure_mmHg: float = 760.0
    water_vapor_pressure_mmHg: float = 47.0
    baseline_fio2: float = 0.21
    respiratory_quotient: float = 0.80
    dead_space_l: float = 0.15
    baseline_rr_bpm: float = 12.0
    baseline_tidal_volume_l: float = 0.50
    baseline_vo2_ml_min: float = 250.0
    baseline_vco2_ml_min: float = 200.0
    baseline_lactate_mmol_l: float = 1.0
    baseline_bicarbonate_mmol_l: float = 24.0  # configuration-compatibility reference
    baseline_ph_arterial: float = 7.40
    baseline_aa_gradient_mmHg: float = 8.0  # compatibility reference; A-a emerges from V/Q/shunt/diffusion

    # Multi-compartment pulmonary gas exchange
    pulmonary_baseline_shunt_fraction: float = 0.005
    pulmonary_baseline_vq_log_sd: float = 0.18
    pulmonary_baseline_diffusing_capacity_relative: float = 1.0
    pulmonary_capillary_blood_volume_ml: float = 70.0
    pulmonary_capillary_recruitment_gain: float = 0.10
    pulmonary_o2_equilibration_tau_s: float = 0.20
    pulmonary_diffusing_capacity_exercise_gain: float = 0.45
    pulmonary_vq_mismatch_log_sd: float = 0.45
    pulmonary_shunt_challenge_fraction: float = 0.18
    pulmonary_diffusion_limitation_relative: float = 0.30

    # Regional vascular control and recruitment/derecruitment
    # HPV is represented as a local precapillary resistance response to alveolar
    # hypoxia.  The parallel-network equivalent resistance feeds back into the
    # closed-loop pulmonary circulation on the following operator-split step.
    pulmonary_hpv_po2_half_mmHg: float = 70.0
    pulmonary_hpv_slope_mmHg: float = 8.0
    pulmonary_hpv_max_local_resistance_multiplier: float = 2.6
    pulmonary_hpv_tau_min: float = 1.5
    pulmonary_hpv_baseline_function_fraction: float = 1.0

    # Recruitment is a reduced pressure-threshold/hysteresis model.  Thresholds
    # represent progressively more dependent units; they are not CT-derived
    # patient-specific opening pressures.
    pulmonary_static_compliance_l_cmH2O: float = 0.10
    pulmonary_mean_inspiratory_pressure_fraction: float = 0.35
    pulmonary_recruitment_logistic_width_cmH2O: float = 1.25
    pulmonary_recruitment_opening_hysteresis_cmH2O: float = 1.5
    pulmonary_recruitment_open_tau_min: float = 0.35
    pulmonary_recruitment_close_tau_min: float = 1.5
    pulmonary_unit_closing_pressures_cmH2O: tuple[float, ...] = (-1.0, 0.0, 1.0, 2.0, 3.0, 4.0)
    pulmonary_min_ventilation_weight_when_closed: float = 0.005
    pulmonary_derecruitment_challenge_offset_cmH2O: float = 8.0
    pulmonary_recruitment_peep_cmH2O: float = 12.0


    # Explicit lung/chest-wall mechanics
    # Normal-order engineering anchors; not patient-specific ventilator targets.
    pulmonary_frc_l: float = 2.50
    pulmonary_lung_compliance_l_cmH2O: float = 0.10
    pulmonary_chest_wall_compliance_l_cmH2O: float = 0.20
    pulmonary_baseline_pleural_pressure_cmH2O: float = -5.0
    pulmonary_peep_pleural_transmission_fraction: float = 0.35
    pulmonary_overdistension_pl50_cmH2O: float = 20.0
    pulmonary_overdistension_slope_cmH2O: float = 2.5
    pulmonary_overdistension_compliance_loss_fraction: float = 0.55
    pulmonary_min_overdistended_compliance_fraction: float = 0.35
    pulmonary_peep_frc_gain_per_cmH2O: float = 0.025
    pulmonary_low_volume_pvr_gain: float = 0.80
    pulmonary_high_volume_pvr_gain: float = 0.70
    pulmonary_mechanics_peep_low_cmH2O: float = 5.0
    pulmonary_mechanics_peep_high_cmH2O: float = 12.0
    pulmonary_mechanics_peep_overdistension_cmH2O: float = 18.0
    pulmonary_stiff_chest_wall_scale: float = 0.50
    pulmonary_low_lung_compliance_scale: float = 0.50

    # Within-breath equation-of-motion mechanics
    respiratory_cycle_dt_s: float = 0.01
    respiratory_drive_pressure_calibration: float = 0.95
    respiratory_airway_resistance_cmH2O_s_l: float = 2.0
    respiratory_inertance_cmH2O_s2_l: float = 0.080
    respiratory_expiratory_resistance_multiplier: float = 1.15
    respiratory_expiratory_flow_limit_l_s: float = 4.0
    respiratory_inspiratory_fraction: float = 0.33
    respiratory_pressure_control_rise_fraction: float = 0.20
    respiratory_flow_nonlinearity_per_l_s: float = 0.10
    respiratory_expiratory_elastance_fraction: float = 0.90
    respiratory_external_peep_threshold_unloading_fraction: float = 0.75
    respiratory_obstruction_resistance_cmH2O_s_l: float = 12.0
    respiratory_obstruction_expiratory_resistance_multiplier: float = 1.8
    respiratory_obstruction_flow_limit_l_s: float = 0.40
    respiratory_obstruction_rr_bpm: float = 24.0
    respiratory_severe_obstruction_rr_bpm: float = 30.0
    respiratory_severe_obstruction_flow_limit_l_s: float = 0.35
    respiratory_pressure_control_cmH2O: float = 10.0

    # Patient-ventilator interaction and pressure-support controller
    respiratory_ps_pressure_support_cmH2O: float = 8.0
    respiratory_ps_trigger_pressure_cmH2O: float = 0.5
    respiratory_ps_trigger_flow_l_s: float = 0.10
    respiratory_ps_cycleoff_fraction_peak_flow: float = 0.25
    respiratory_ps_rise_time_s: float = 0.15
    respiratory_ps_max_inspiratory_time_s: float = 3.0
    respiratory_ps_refractory_time_s: float = 0.25
    respiratory_ps_asynchrony_timing_threshold_s: float = 0.40
    respiratory_ps_double_trigger_window_s: float = 0.50
    respiratory_neural_inspiratory_fraction: float = 0.30
    respiratory_ps_neural_unloading_fraction: float = 0.55
    respiratory_ps_min_patient_drive_cmH2O: float = 1.5
    respiratory_ps_high_support_cmH2O: float = 15.0
    respiratory_ps_low_cycleoff_fraction: float = 0.10
    respiratory_ps_optimized_cycleoff_fraction: float = 0.40
    respiratory_ps_premature_cycleoff_fraction: float = 0.60
    respiratory_ps_low_drive_gain: float = 0.25
    respiratory_ps_high_drive_gain: float = 1.30
    respiratory_ps_long_neural_inspiratory_fraction: float = 0.48
    respiratory_ps_external_peep_unloading_cmH2O: float = 5.0
    respiratory_ps_autotrigger_leak_flow_l_s: float = 0.16
    respiratory_ps_autotrigger_refractory_s: float = 2.5

    # Physicochemical acid-base closure
    co2_solubility_mmol_l_mmHg: float = 0.0301
    carbonic_acid_pka: float = 6.10
    baseline_albumin_g_dl: float = 4.2
    baseline_phosphate_mmol_l: float = 1.0
    acid_base_charge_tolerance_mEq_l: float = 1e-8
    acid_base_max_iterations: int = 100

    # Whole-blood CO2 / erythrocyte chemistry
    # The plasma/RBC water fractions and hematocrit are reduced-order reference
    # values used to convert compartment concentrations to per-L whole-blood content.
    baseline_hematocrit: float = 0.42
    plasma_water_fraction: float = 0.93
    rbc_water_fraction: float = 0.70
    carbonate_pka2: float = 10.33
    hemoglobin_monomer_mw_g_mmol: float = 16.114
    hemoglobin_buffer_value_mEq_per_mmol_pH: float = 2.30
    baseline_carbamino_fraction: float = 0.131
    carbamino_reference_o2_saturation: float = 0.972
    carbamino_haldane_affinity_gain: float = 2.50
    # Funder & Wieth (1966) empirical Donnan ratios: rCl=Cl_rbc/Cl_plasma,
    # rH=H_plasma/H_rbc. Validated only near physiologic human-blood conditions.
    funder_rcl_intercept: float = 3.319
    funder_rcl_ph_slope: float = 0.359
    funder_rh_intercept: float = 3.094
    funder_rh_ph_slope: float = 0.335
    # Normal arteriovenous plasma chloride shift is represented as an equilibrium
    # redistribution, not a net whole-body chloride flux.
    chloride_shift_gain_mmol_l_per_sat_fraction: float = 4.0
    chloride_shift_max_mmol_l: float = 3.0
    # Oxygen-dependent Hb chloride binding lowers *free* RBC chloride in venous
    # blood even though Hamburger exchange increases total RBC chloride. Prange
    # reports a physiologic arterial-to-venous free-Cl fall of roughly 1-3 mmol/L.
    rbc_free_chloride_drop_gain_mmol_l_per_sat_fraction: float = 5.0
    rbc_free_chloride_drop_max_mmol_l: float = 3.0
    # Lumped rapidly exchangeable CO2 capacitance. This is explicitly an effective
    # distribution volume, not total-body carbon.
    co2_exchangeable_volume_fraction_tbw: float = 0.25
    co2_gas_molar_volume_l_per_mol_stpd: float = 22.414
    co2_pool_solver_tolerance_mmol_l: float = 1e-7
    co2_pool_solver_max_iterations: int = 70

    # Respiratory response
    exercise_vo2_gain: float = 5.0
    exercise_vco2_gain: float = 6.0
    respiratory_feedback_gain_co2: float = 0.55
    respiratory_feedback_gain_acid: float = 55.0
    respiratory_tau_min: float = 1.5
    gas_exchange_tau_min: float = 0.8
    # Scenario severity parameters exposed for virtual-cohort / robustness studies.
    hypoventilation_efficiency: float = 0.68
    respiratory_acidosis_efficiency: float = 0.62
    respiratory_acidosis_initial_paco2_mmHg: float = 60.0
    # Deprecated, intentionally unused compatibility field. Lactate changes
    # Stewart SID once; applying this deprecated direct bicarbonate coefficient as
    # well would double-count the same acid-base perturbation.
    lactate_bicarbonate_coeff: float = 0.70

    # Whole-blood O2 transport / Fick coupling
    hemoglobin_g_dl: float = 14.0
    hemoglobin_o2_capacity_ml_g: float = 1.34
    dissolved_o2_coeff_ml_dl_mmHg: float = 0.0031
    # Hb-O2 affinity coupling at standard adult blood reference conditions.
    o2_standard_ph: float = 7.40
    o2_standard_pco2_mmHg: float = 40.0
    # Most CO2 affinity shift is already mediated through pH. This small residual
    # direct term prevents complete decoupling without double-counting the Bohr effect.
    o2_direct_co2_log_affinity_gain: float = 0.025
    # Whole-body supply-dependence scaffold. The extraction ceiling is an
    # engineering physiology parameter, not a patient-specific critical DO2 threshold.
    oxygen_max_extraction_fraction: float = 0.70
    oxygen_supply_transition_width_fraction: float = 0.08
    # Lactate is represented as an amount in an apparent whole-body exchange
    # space. Human 13C-lactate data place steady-state distribution near total
    # body water; the coefficient remains explicit for sensitivity analysis.
    lactate_distribution_volume_fraction_tbw: float = 1.0
    exercise_lactate_production_linear_mmol_l_min: float = 0.045
    exercise_lactate_production_quadratic_mmol_l_min: float = 0.060
    hypoxic_lactate_gain_mmol_l_min: float = 0.18
    lactate_clearance_per_min: float = 0.025
    # Dynamic hematology assumes RBC/Hb mass is conserved over short simulations.
    baseline_rbc_volume_l: float = field(
        default=2.0,
        metadata={
            "deprecated": (
                "ignored compatibility field; baseline RBC volume is derived "
                "from total blood and plasma volumes"
            )
        },
    )
    # RL action 5 drives positive airway pressure through the mechanics model.
    max_ventilation_pressure_assist_cmH2O: float = 12.0

    # Renal / electrolyte / RAAS reference parameters
    baseline_gfr_ml_min: float = 120.0
    baseline_urine_flow_ml_min: float = 1.0
    baseline_urine_sodium_mmol_min: float = 0.070
    baseline_urine_potassium_mmol_min: float = 0.040
    baseline_urine_chloride_mmol_min: float = 0.070
    baseline_net_acid_excretion_mmol_min: float = 0.040
    endogenous_acid_production_mmol_min: float = 0.040
    baseline_ammonium_fraction_of_nae: float = 0.60
    renal_acid_response_gain: float = 8.0
    renal_bicarbonaturia_gain: float = 0.20
    adh_tau_min: float = 12.0
    renin_tau_min: float = 20.0
    angiotensin_tau_min: float = 12.0
    aldosterone_tau_min: float = 45.0
    gfr_tau_min: float = 8.0
    urine_flow_tau_min: float = 8.0
    renal_bicarbonate_tau_min: float = field(
        default=2880.0,
        metadata={
            "deprecated": (
                "ignored compatibility field; renal acid handling uses explicit "
                "ammonium, titratable-acid and bicarbonate fluxes"
            )
        },
    )
    renal_co2_compensation_gain: float = field(
        default=0.35,
        metadata={
            "deprecated": (
                "ignored compatibility field; no empirical chronic CO2 "
                "compensation target is applied"
            )
        },
    )
    potassium_buffer_gain_per_min: float = field(
        default=0.25,
        metadata={
            "deprecated": (
                "ignored compatibility field; transcellular potassium uses the "
                "explicit insulin, pH and exercise target with a time constant"
            )
        },
    )

    # PBPK reference compound -- generic research probe, not a real medicine
    pbpk_internal_step_min: float = 0.05
    pbpk_absorption_ka_per_min: float = 0.035
    pbpk_fraction_unbound: float = 0.50
    pbpk_hepatic_clint_l_min: float = 0.18
    pbpk_renal_filtration_multiplier: float = 1.0
    pbpk_effect_site_tau_min: float = 20.0
    pbpk_target_effect_site_mg_l: float = 0.80
    pbpk_high_exposure_mg_l: float = 3.0

    # Effective PBPK tissue volumes (L)
    pbpk_liver_volume_l: float = 1.80
    pbpk_kidney_volume_l: float = 0.31
    pbpk_muscle_volume_l: float = 28.0
    pbpk_adipose_volume_l: float = 15.0
    pbpk_rest_volume_l: float = 20.0

    # Resting effective organ plasma/blood flows used by the reduced PBPK model (L/min)
    pbpk_liver_flow_l_min: float = 1.35
    pbpk_kidney_flow_l_min: float = 1.10
    pbpk_muscle_flow_l_min: float = 0.75
    pbpk_adipose_flow_l_min: float = 0.25
    pbpk_rest_flow_l_min: float = 1.55

    # Generic tissue:plasma partition coefficients
    pbpk_kp_liver: float = 1.6
    pbpk_kp_kidney: float = 1.3
    pbpk_kp_muscle: float = 1.0
    pbpk_kp_adipose: float = 2.5
    pbpk_kp_rest: float = 1.0

    # Action scaling
    max_insulin_model_units_per_step: float = 2.0
    max_carbs_g_per_step: float = 25.0
    max_saline_ml_per_step: float = 250.0
    max_fio2: float = 0.60
    max_ventilation_support_l_min: float = 8.0
    max_oral_water_ml_per_step: float = 300.0
    max_probe_drug_mg_per_step: float = 25.0

    # Episode limits -- simulation stop limits, not diagnostic thresholds
    glucose_min_terminate: float = 35.0
    glucose_max_terminate: float = 500.0
    map_min_terminate: float = 45.0
    map_max_terminate: float = 180.0
    spo2_min_terminate: float = 75.0
    pao2_min_terminate: float = 35.0
    paco2_max_terminate: float = 90.0
    ph_min_terminate: float = 6.90
    ph_max_terminate: float = 7.75
    sodium_min_terminate: float = 115.0
    sodium_max_terminate: float = 165.0
    potassium_min_terminate: float = 2.4
    potassium_max_terminate: float = 7.0

    def __post_init__(self) -> None:
        """Reject non-finite and structurally inconsistent configurations.

        These checks describe numerical and dimensional requirements, not
        clinical reference intervals. They fail at construction time so a bad
        value cannot be silently clipped or surface later inside a nested
        integration or root-finding loop.
        """

        definitions = fields(self)
        scalar_values: dict[str, float] = {}
        for definition in definitions:
            name = definition.name
            label = (
                "water vapor pressure"
                if name == "water_vapor_pressure_mmHg"
                else name
            )
            value = getattr(self, name)
            if isinstance(value, tuple):
                if name != "pulmonary_unit_closing_pressures_cmH2O" or not value:
                    raise ValueError(f"{name} must be a non-empty numeric tuple")
                for index, component in enumerate(value):
                    if (
                        isinstance(component, bool)
                        or not isinstance(component, Real)
                        or not isfinite(float(component))
                    ):
                        raise ValueError(
                            f"{name}[{index}] must be a finite real number"
                        )
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{label} must be a real number")
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"{label} must be finite")
            scalar_values[name] = numeric

        # Pleural pressure is the only signed scalar configuration quantity.
        # Regional closing pressures are the signed tuple checked separately.
        for name, value in scalar_values.items():
            if name != "pulmonary_baseline_pleural_pressure_cmH2O" and value < 0.0:
                label = (
                    "water vapor pressure"
                    if name == "water_vapor_pressure_mmHg"
                    else name
                )
                raise ValueError(f"{label} must be nonnegative")

        positive = {
            "agent_step_min",
            "integration_step_min",
            "episode_minutes",
            "dalla_internal_step_min",
            "cv_internal_step_s",
            "respiratory_cycle_dt_s",
            "pbpk_internal_step_min",
            "body_weight_kg",
            "plasma_volume_baseline_l",
            "ecf_volume_baseline_l",
            "total_body_water_baseline_l",
            "cv_baseline_blood_volume_ml",
            "atmospheric_pressure_mmHg",
            "respiratory_quotient",
            "baseline_rr_bpm",
            "baseline_tidal_volume_l",
            "baseline_vo2_ml_min",
            "baseline_vco2_ml_min",
            "baseline_lactate_mmol_l",
            "pulmonary_capillary_blood_volume_ml",
            "co2_solubility_mmol_l_mmHg",
            "hemoglobin_monomer_mw_g_mmol",
            "co2_exchangeable_volume_fraction_tbw",
            "co2_gas_molar_volume_l_per_mol_stpd",
            "respiratory_tau_min",
            "gas_exchange_tau_min",
            "hemoglobin_o2_capacity_ml_g",
            "o2_standard_pco2_mmHg",
            "lactate_distribution_volume_fraction_tbw",
            "lactate_clearance_per_min",
            "acid_base_charge_tolerance_mEq_l",
            "co2_pool_solver_tolerance_mmol_l",
            "glucose_setpoint_mg_dl",
            "glucagon_baseline_pg_ml",
            "glucagon_tau_min",
            "liver_glycogen_baseline_g",
            "muscle_glycogen_baseline_g",
            "gut_absorption_tau_min",
            "glucose_distribution_dl",
            "dalla_basal_glucose_mg_dl",
            "dalla_uU_ml_per_insulin_model_unit",
            "sc_insulin_tmax_min",
            "glucagon_hypoglycemia_slope_mg_dl",
            "dalla_insulin_sensitivity_scale",
            "dalla_gastric_absorption_scale",
            "osmotic_water_equilibration_tau_min",
            "potassium_transcellular_tau_min",
            "potassium_insulin_half_effect_uU_ml",
            "cv_baseline_cardiac_output_l_min",
            "cv_resting_hr_bpm",
            "cv_map_setpoint_mmHg",
            "cv_hr_tau_min",
            "cv_v_la0_ml",
            "cv_v_lv0_ml",
            "cv_v_sa0_ml",
            "cv_v_sv0_ml",
            "cv_v_ra0_ml",
            "cv_v_rv0_ml",
            "cv_v_pa0_ml",
            "cv_v_pv0_ml",
            "cv_c_la_ml_mmHg",
            "cv_c_sa_ml_mmHg",
            "cv_c_sv_ml_mmHg",
            "cv_c_ra_ml_mmHg",
            "cv_c_pa_ml_mmHg",
            "cv_c_pv_ml_mmHg",
            "cv_lv_emin",
            "cv_lv_emax",
            "cv_rv_emin",
            "cv_rv_emax",
            "cv_r_mitral_mmHg_s_ml",
            "cv_r_aortic_mmHg_s_ml",
            "cv_r_tricuspid_mmHg_s_ml",
            "cv_r_pulmonic_mmHg_s_ml",
            "cv_r_systemic_mmHg_s_ml",
            "cv_r_systemic_venous_mmHg_s_ml",
            "cv_r_pulmonary_mmHg_s_ml",
            "cv_r_pulmonary_venous_mmHg_s_ml",
            "baseline_bicarbonate_mmol_l",
            "baseline_ph_arterial",
            "pulmonary_baseline_diffusing_capacity_relative",
            "pulmonary_hpv_po2_half_mmHg",
            "pulmonary_hpv_slope_mmHg",
            "pulmonary_hpv_max_local_resistance_multiplier",
            "pulmonary_hpv_tau_min",
            "pulmonary_static_compliance_l_cmH2O",
            "pulmonary_recruitment_logistic_width_cmH2O",
            "pulmonary_recruitment_open_tau_min",
            "pulmonary_recruitment_close_tau_min",
            "pulmonary_frc_l",
            "pulmonary_lung_compliance_l_cmH2O",
            "pulmonary_chest_wall_compliance_l_cmH2O",
            "pulmonary_overdistension_slope_cmH2O",
            "respiratory_airway_resistance_cmH2O_s_l",
            "respiratory_inertance_cmH2O_s2_l",
            "respiratory_expiratory_resistance_multiplier",
            "respiratory_expiratory_flow_limit_l_s",
            "respiratory_obstruction_resistance_cmH2O_s_l",
            "respiratory_obstruction_expiratory_resistance_multiplier",
            "respiratory_obstruction_flow_limit_l_s",
            "respiratory_obstruction_rr_bpm",
            "respiratory_severe_obstruction_rr_bpm",
            "respiratory_severe_obstruction_flow_limit_l_s",
            "respiratory_ps_rise_time_s",
            "respiratory_ps_max_inspiratory_time_s",
            "carbonic_acid_pka",
            "baseline_albumin_g_dl",
            "baseline_phosphate_mmol_l",
            "plasma_water_fraction",
            "rbc_water_fraction",
            "carbonate_pka2",
            "hemoglobin_g_dl",
            "baseline_gfr_ml_min",
            "baseline_urine_flow_ml_min",
            "baseline_urine_sodium_mmol_min",
            "adh_tau_min",
            "renin_tau_min",
            "angiotensin_tau_min",
            "aldosterone_tau_min",
            "gfr_tau_min",
            "urine_flow_tau_min",
            "pbpk_absorption_ka_per_min",
            "pbpk_effect_site_tau_min",
            "pbpk_liver_volume_l",
            "pbpk_kidney_volume_l",
            "pbpk_muscle_volume_l",
            "pbpk_adipose_volume_l",
            "pbpk_rest_volume_l",
            "pbpk_liver_flow_l_min",
            "pbpk_kidney_flow_l_min",
            "pbpk_muscle_flow_l_min",
            "pbpk_adipose_flow_l_min",
            "pbpk_rest_flow_l_min",
            "pbpk_kp_liver",
            "pbpk_kp_kidney",
            "pbpk_kp_muscle",
            "pbpk_kp_adipose",
            "pbpk_kp_rest",
        }
        for name in positive:
            if scalar_values[name] <= 0.0:
                raise ValueError(f"{name} must be positive")

        if not (
            self.plasma_volume_baseline_l
            < self.ecf_volume_baseline_l
            < self.total_body_water_baseline_l
        ):
            raise ValueError(
                "baseline volumes must satisfy plasma < ECF < total body water"
            )
        if not (0.0 <= self.water_vapor_pressure_mmHg < self.atmospheric_pressure_mmHg):
            raise ValueError("water vapor pressure must be below atmospheric pressure")
        if not (0.0 < self.baseline_fio2 <= self.max_fio2 <= 1.0):
            raise ValueError("FiO2 values must satisfy 0 < baseline_fio2 <= max_fio2 <= 1")
        if not (0.0 < self.respiratory_quotient <= 1.0):
            raise ValueError("respiratory_quotient must be in (0, 1]")

        nonnegative = (
            "exercise_vo2_gain",
            "exercise_vco2_gain",
            "exercise_lactate_production_linear_mmol_l_min",
            "exercise_lactate_production_quadratic_mmol_l_min",
            "hypoxic_lactate_gain_mmol_l_min",
            "dead_space_l",
            "pulmonary_o2_equilibration_tau_s",
            "dissolved_o2_coeff_ml_dl_mmHg",
            "oxygen_supply_transition_width_fraction",
            "max_insulin_model_units_per_step",
            "max_carbs_g_per_step",
            "max_saline_ml_per_step",
            "max_ventilation_support_l_min",
            "max_ventilation_pressure_assist_cmH2O",
            "max_oral_water_ml_per_step",
            "max_probe_drug_mg_per_step",
        )
        for name in nonnegative:
            if scalar_values[name] < 0.0:
                raise ValueError(f"{name} must be nonnegative")

        closed_fractions = (
            "cv_exercise_systemic_vasodilation",
            "pulmonary_baseline_shunt_fraction",
            "pulmonary_shunt_challenge_fraction",
            "pulmonary_diffusion_limitation_relative",
            "pulmonary_hpv_baseline_function_fraction",
            "pulmonary_mean_inspiratory_pressure_fraction",
            "pulmonary_min_ventilation_weight_when_closed",
            "pulmonary_peep_pleural_transmission_fraction",
            "pulmonary_overdistension_compliance_loss_fraction",
            "pulmonary_stiff_chest_wall_scale",
            "pulmonary_low_lung_compliance_scale",
            "respiratory_pressure_control_rise_fraction",
            "respiratory_expiratory_elastance_fraction",
            "respiratory_external_peep_threshold_unloading_fraction",
            "respiratory_ps_neural_unloading_fraction",
            "baseline_carbamino_fraction",
            "carbamino_reference_o2_saturation",
            "oxygen_supply_transition_width_fraction",
            "baseline_ammonium_fraction_of_nae",
            "pbpk_fraction_unbound",
        )
        for name in closed_fractions:
            if not 0.0 <= scalar_values[name] <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        open_fractions = (
            "cv_systolic_fraction",
            "pulmonary_min_overdistended_compliance_fraction",
            "respiratory_inspiratory_fraction",
            "respiratory_ps_cycleoff_fraction_peak_flow",
            "respiratory_neural_inspiratory_fraction",
            "respiratory_ps_low_cycleoff_fraction",
            "respiratory_ps_optimized_cycleoff_fraction",
            "respiratory_ps_premature_cycleoff_fraction",
            "respiratory_ps_long_neural_inspiratory_fraction",
            "baseline_hematocrit",
            "plasma_water_fraction",
            "rbc_water_fraction",
            "hypoventilation_efficiency",
            "respiratory_acidosis_efficiency",
        )
        for name in open_fractions:
            if not 0.0 < scalar_values[name] < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if not 0.0 < self.oxygen_max_extraction_fraction <= 1.0:
            raise ValueError("oxygen_max_extraction_fraction must be in (0, 1]")

        if self.integration_step_min > self.agent_step_min:
            raise ValueError(
                "simulation steps must satisfy "
                "integration_step_min <= agent_step_min"
            )
        if not self.dead_space_l < self.baseline_tidal_volume_l:
            raise ValueError("dead_space_l must be below baseline_tidal_volume_l")

        initial_cv_volume = sum(
            getattr(self, name)
            for name in (
                "cv_v_la0_ml",
                "cv_v_lv0_ml",
                "cv_v_sa0_ml",
                "cv_v_sv0_ml",
                "cv_v_ra0_ml",
                "cv_v_rv0_ml",
                "cv_v_pa0_ml",
                "cv_v_pv0_ml",
            )
        )
        if not isclose(
            initial_cv_volume,
            self.cv_baseline_blood_volume_ml,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "initial cardiovascular compartment volumes must sum to "
                "cv_baseline_blood_volume_ml"
            )
        if not self.plasma_volume_baseline_l < self.cv_baseline_blood_volume_ml / 1000.0:
            raise ValueError(
                "plasma_volume_baseline_l must be below total baseline blood volume"
            )
        for chamber in ("la", "lv", "sa", "sv", "ra", "rv", "pa", "pv"):
            unstressed = getattr(self, f"cv_v0_{chamber}_ml")
            initial = getattr(self, f"cv_v_{chamber}0_ml")
            if not 0.0 <= unstressed < initial:
                raise ValueError(
                    f"cv_v0_{chamber}_ml must be nonnegative and below "
                    f"cv_v_{chamber}0_ml"
                )
        if not self.cv_lv_emin < self.cv_lv_emax:
            raise ValueError("cv_lv_emin must be below cv_lv_emax")
        if not self.cv_rv_emin < self.cv_rv_emax:
            raise ValueError("cv_rv_emin must be below cv_rv_emax")

        closing_pressures = self.pulmonary_unit_closing_pressures_cmH2O
        if any(
            right <= left
            for left, right in zip(closing_pressures, closing_pressures[1:])
        ):
            raise ValueError(
                "pulmonary_unit_closing_pressures_cmH2O must be strictly increasing"
            )
        if self.pulmonary_hpv_max_local_resistance_multiplier < 1.0:
            raise ValueError(
                "pulmonary_hpv_max_local_resistance_multiplier must be at least 1"
            )
        if not (
            self.pulmonary_mechanics_peep_low_cmH2O
            < self.pulmonary_mechanics_peep_high_cmH2O
            < self.pulmonary_mechanics_peep_overdistension_cmH2O
        ):
            raise ValueError(
                "pulmonary mechanics PEEP anchors must be strictly increasing"
            )
        if not (
            self.respiratory_ps_low_cycleoff_fraction
            < self.respiratory_ps_cycleoff_fraction_peak_flow
            < self.respiratory_ps_optimized_cycleoff_fraction
            < self.respiratory_ps_premature_cycleoff_fraction
        ):
            raise ValueError(
                "pressure-support cycle-off fractions must be strictly increasing"
            )
        if self.respiratory_ps_rise_time_s > self.respiratory_ps_max_inspiratory_time_s:
            raise ValueError(
                "respiratory_ps_rise_time_s must not exceed "
                "respiratory_ps_max_inspiratory_time_s"
            )

        if not self.carbonic_acid_pka < self.baseline_ph_arterial < self.carbonate_pka2:
            raise ValueError(
                "acid-base constants must satisfy carbonic_acid_pka < "
                "baseline_ph_arterial < carbonate_pka2"
            )
        if not 4.00 < self.baseline_ph_arterial < 10.00:
            raise ValueError("baseline_ph_arterial must lie inside the acid-base bracket")

        resting_rq = self.baseline_vco2_ml_min / self.baseline_vo2_ml_min
        exercising_rq = resting_rq * (
            (1.0 + self.exercise_vco2_gain)
            / (1.0 + self.exercise_vo2_gain)
        )
        if not (0.60 <= resting_rq <= 1.0 and 0.60 <= exercising_rq <= 1.0):
            raise ValueError(
                "baseline/exercise VO2 and VCO2 targets must imply metabolic RQ in [0.60, 1.0]"
            )

        for lower_name, upper_name in (
            ("glucose_min_terminate", "glucose_max_terminate"),
            ("map_min_terminate", "map_max_terminate"),
            ("ph_min_terminate", "ph_max_terminate"),
            ("sodium_min_terminate", "sodium_max_terminate"),
            ("potassium_min_terminate", "potassium_max_terminate"),
        ):
            if not getattr(self, lower_name) < getattr(self, upper_name):
                raise ValueError(f"{lower_name} must be below {upper_name}")
        if not self.glucose_min_terminate < self.glucose_setpoint_mg_dl < self.glucose_max_terminate:
            raise ValueError("glucose_setpoint_mg_dl must lie between termination limits")
        if not self.ph_min_terminate < self.baseline_ph_arterial < self.ph_max_terminate:
            raise ValueError("baseline_ph_arterial must lie between termination limits")
        if not self.pbpk_target_effect_site_mg_l < self.pbpk_high_exposure_mg_l:
            raise ValueError(
                "pbpk_target_effect_site_mg_l must be below pbpk_high_exposure_mg_l"
            )

        for name in ("acid_base_max_iterations", "co2_pool_solver_max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        for definition in definitions:
            reason = definition.metadata.get("deprecated")
            if reason is not None and getattr(self, definition.name) != definition.default:
                warn(
                    f"{definition.name} is deprecated and ignored: {reason}",
                    DeprecationWarning,
                    stacklevel=2,
                )
