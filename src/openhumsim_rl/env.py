from __future__ import annotations

from copy import deepcopy
from typing import Any
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _BaseEnv = gym.Env
    HAS_GYMNASIUM = True
except ImportError:
    from .compat import Env as _BaseEnv, spaces
    HAS_GYMNASIUM = False

from .config import HumanConfig
from .physiology import HumanPhysiology, HumanState, Intervention
from .measurement import ClinicalMeasurementConfig, ClinicalMeasurementModel


OBSERVATION_NAMES = (
    "time_to_go_fraction",
    "glucose_mg_dl", "insulin_uU_ml", "glucagon_pg_ml",
    "sc_insulin_depot1_model_units", "sc_insulin_depot2_model_units",
    "sc_insulin_absorption_model_units_min",
    "glucagon_counterregulatory_egp_mg_kg_min",
    "gut_carbs_g", "lactate_mmol_l",
    "dalla_ra_mg_kg_min", "dalla_egp_mg_kg_min", "dalla_u_mg_kg_min",
    "heart_rate_bpm", "map_mmHg", "systolic_pressure_mmHg", "diastolic_pressure_mmHg",
    "cardiac_output_l_min", "stroke_volume_ml", "central_venous_pressure_mmHg",
    "pulmonary_artery_pressure_mmHg", "cv_ejection_fraction", "plasma_volume_l",
    "hematocrit_fraction", "hemoglobin_g_dl",
    "respiratory_rate_bpm", "tidal_volume_l", "pao2_mmHg", "paco2_mmHg",
    "bicarbonate_mmol_l", "ph_arterial", "spo2_pct", "alveolar_ventilation_l_min",
    "pulmonary_shunt_fraction", "pulmonary_vq_log_sd",
    "pulmonary_diffusing_capacity_relative", "pulmonary_mean_vq_ratio",
    "pulmonary_low_vq_perfusion_fraction", "pulmonary_high_vq_ventilation_fraction",
    "pulmonary_capillary_transit_time_s", "pulmonary_diffusion_equilibration_fraction",
    "pulmonary_aa_gradient_mmHg", "pulmonary_alveolar_dead_space_fraction", "pulmonary_enghoff_dead_space_fraction",
    "pulmonary_recruitment_fraction", "pulmonary_hpv_resistance_multiplier",
    "pulmonary_perfusion_redistribution_index", "pulmonary_mean_distending_pressure_cmH2O",
    "pulmonary_effective_capillary_blood_volume_ml", "pulmonary_hypoxic_perfusion_fraction",
    "pulmonary_transpulmonary_pressure_end_exp_cmH2O",
    "pulmonary_transpulmonary_pressure_end_insp_cmH2O",
    "pulmonary_passive_equivalent_plateau_pressure_cmH2O",
    "pulmonary_airway_driving_pressure_cmH2O",
    "pulmonary_lung_compliance_l_cmH2O",
    "pulmonary_chest_wall_compliance_l_cmH2O",
    "pulmonary_respiratory_system_compliance_l_cmH2O",
    "pulmonary_lung_strain", "pulmonary_overdistension_fraction",
    "pulmonary_intrathoracic_pressure_delta_cmH2O",
    "pulmonary_mechanical_pvr_multiplier", "pulmonary_mechanical_power_j_min",
    "respiratory_airway_resistance_cmH2O_s_l",
    "respiratory_cycle_auto_peep_cmH2O",
    "respiratory_cycle_dynamic_hyperinflation_l",
    "respiratory_cycle_peak_inspiratory_flow_l_s",
    "respiratory_cycle_peak_expiratory_flow_l_s",
    "respiratory_cycle_end_expiratory_flow_l_s",
    "respiratory_cycle_peak_muscle_pressure_cmH2O",
    "respiratory_cycle_peak_airway_pressure_cmH2O",
    "respiratory_cycle_resistive_work_j_breath",
    "respiratory_cycle_muscle_work_j_breath",
    "respiratory_cycle_ventilator_work_j_breath",
    "respiratory_cycle_pv_hysteresis_j_breath",
    "respiratory_cycle_total_work_j_breath",
    "respiratory_cycle_expiratory_flow_limited_fraction",
    "respiratory_cycle_time_constant_s",
    "respiratory_ventilator_mean_trigger_delay_s",
    "respiratory_ventilator_mean_cycling_delay_s",
    "respiratory_ventilator_ineffective_trigger_fraction",
    "respiratory_ventilator_double_trigger_fraction",
    "respiratory_ventilator_premature_cycling_fraction",
    "respiratory_ventilator_delayed_cycling_fraction",
    "respiratory_ventilator_autotrigger_fraction",
    "respiratory_ventilator_asynchrony_index_pct",
    "respiratory_ventilator_patient_efforts_per_min",
    "respiratory_ventilator_breaths_per_min",
    "respiratory_ventilator_trigger_pressure_time_product_cmH2O_s",
    "respiratory_ventilator_neural_inspiratory_time_s",
    "arterial_o2_content_ml_dl", "mixed_venous_o2_content_ml_dl",
    "mixed_venous_o2_sat_pct", "oxygen_delivery_ml_min", "oxygen_extraction_ratio",
    "vo2_demand_ml_min", "vo2_ml_min", "oxygen_debt_ml_min", "aerobic_fraction",
    "sodium_mmol_l", "potassium_mmol_l", "chloride_mmol_l",
    "strong_ion_difference_apparent_mEq_l", "strong_ion_gap_mEq_l",
    "albumin_g_dl", "phosphate_mmol_l", "total_co2_mmol_l", "anion_gap_mEq_l",
    "arterial_total_co2_mmol_l_blood", "mixed_venous_pco2_mmHg",
    "mixed_venous_total_co2_mmol_l_blood", "rbc_ph",
    "rbc_bicarbonate_mmol_l", "carbamino_co2_mmol_l_blood",
    "chloride_shift_plasma_mmol_l", "haldane_co2_content_gain_mmol_l",
    "exchangeable_co2_pool_mmol", "vco2_elimination_ml_min", "effective_co2_ventilation_l_min",
    "gfr_ml_min", "urine_ammonium_mmol_min",
    "adh_relative", "renin_relative", "angiotensin_ii_relative", "aldosterone_relative",
    "urine_flow_ml_min", "plasma_osmolality_mOsm_kg", "ecf_volume_l",
    "icf_volume_l", "ecf_effective_tonicity_mOsm_l", "icf_effective_tonicity_mOsm_l",
    "osmotic_water_shift_l_min", "potassium_transcellular_flux_mmol_min",
    "potassium_transcellular_target_mmol_l",
    "renal_function_fraction", "probe_gut_mg", "probe_plasma_mg_l",
    "probe_effect_site_mg_l", "probe_total_body_mg",
)

ACTION_NAMES = (
    "insulin", "oral_carbs", "exercise", "saline", "oxygen",
    "ventilation_pressure_assist", "oral_water", "oral_probe_compound",
 )

# Reward components other than instantaneous bolus costs are treated as rates
# integrated relative to the original five-minute decision interval. This keeps
# cumulative utility comparable when ``agent_step_min`` changes.
REWARD_REFERENCE_INTERVAL_MIN = 5.0

# ``info`` is passed to benchmark policies by several evaluation helpers.  Keep
# this contract positive (an allowlist), so a debug-only diagnostic cannot
# silently become an oracle observation.  Values here are static schema/contract
# metadata, plus an optional status label after a terminal transition.  In
# particular, neither the hidden measurement-process state nor elapsed time is
# supplied out-of-band to a benchmark policy.
BENCHMARK_INFO_KEYS = (
    "observation_names",
    "observation_profile",
    "measurement_profile",
    "info_profile",
    "action_names",
    "gymnasium_installed",
    "environment_semantics",
    "termination_reason",
)

# Observables intended to approximate information obtainable from bedside
# monitoring, blood gas/laboratory sampling and ventilator telemetry. Hidden
# mechanistic states (regional V/Q, Dalla internal fluxes, HPV tone, PBPK effect
# site, exact oxygen debt, etc.) are excluded. This is the default RL profile.
CLINICAL_OBSERVATION_NAMES = (
    "time_to_go_fraction",
    "sensor_glucose_mg_dl", "lactate_mmol_l",
    "heart_rate_bpm", "map_mmHg", "systolic_pressure_mmHg", "diastolic_pressure_mmHg",
    "cardiac_output_l_min", "stroke_volume_ml", "central_venous_pressure_mmHg",
    "pulmonary_artery_pressure_mmHg", "plasma_volume_l", "hematocrit_fraction", "hemoglobin_g_dl",
    "respiratory_rate_bpm", "tidal_volume_l", "pao2_mmHg", "paco2_mmHg",
    "bicarbonate_mmol_l", "ph_arterial", "spo2_pct", "alveolar_ventilation_l_min",
    "pulmonary_passive_equivalent_plateau_pressure_cmH2O",
    "pulmonary_airway_driving_pressure_cmH2O",
    "pulmonary_respiratory_system_compliance_l_cmH2O",
    "respiratory_airway_resistance_cmH2O_s_l", "respiratory_cycle_auto_peep_cmH2O",
    "respiratory_cycle_peak_inspiratory_flow_l_s", "respiratory_cycle_peak_expiratory_flow_l_s",
    "respiratory_cycle_peak_airway_pressure_cmH2O", "respiratory_cycle_total_work_j_breath",
    "respiratory_ventilator_mean_trigger_delay_s", "respiratory_ventilator_mean_cycling_delay_s",
    "respiratory_ventilator_ineffective_trigger_fraction", "respiratory_ventilator_double_trigger_fraction",
    "respiratory_ventilator_autotrigger_fraction", "respiratory_ventilator_asynchrony_index_pct",
    "arterial_o2_content_ml_dl", "mixed_venous_o2_sat_pct", "oxygen_delivery_ml_min",
    "sodium_mmol_l", "potassium_mmol_l", "chloride_mmol_l", "albumin_g_dl",
    "phosphate_mmol_l", "anion_gap_mEq_l", "gfr_ml_min", "urine_flow_ml_min",
    "plasma_osmolality_mOsm_kg",
    "cgm_measurement_age_min", "monitor_measurement_age_min",
    "blood_gas_measurement_age_min", "chemistry_measurement_age_min",
    "hemodynamic_measurement_age_min",
)

# Smooth tanh normalization. Centers/scales are engineering normalization values,
# not diagnostic reference ranges.
_DEFAULT_CENTER_SCALE = {
    "time_to_go_fraction": (0.5, 0.5),
    # Lung/chest-wall mechanics diagnostics
    "pulmonary_transpulmonary_pressure_end_exp_cmH2O": (5.0, 8.0),
    "pulmonary_transpulmonary_pressure_end_insp_cmH2O": (10.0, 12.0),
    "pulmonary_passive_equivalent_plateau_pressure_cmH2O": (8.0, 15.0),
    "pulmonary_airway_driving_pressure_cmH2O": (7.5, 10.0),
    "pulmonary_lung_compliance_l_cmH2O": (0.10, 0.08),
    "pulmonary_chest_wall_compliance_l_cmH2O": (0.20, 0.12),
    "pulmonary_respiratory_system_compliance_l_cmH2O": (0.067, 0.05),
    "pulmonary_lung_strain": (0.20, 0.40),
    "pulmonary_overdistension_fraction": (0.02, 0.40),
    "pulmonary_intrathoracic_pressure_delta_cmH2O": (0.0, 6.0),
    "pulmonary_mechanical_pvr_multiplier": (1.0, 1.0),
    "pulmonary_mechanical_power_j_min": (2.0, 10.0),
    "respiratory_airway_resistance_cmH2O_s_l": (2.0, 8.0),
    "respiratory_cycle_auto_peep_cmH2O": (0.0, 5.0),
    "respiratory_cycle_dynamic_hyperinflation_l": (0.0, 0.5),
    "respiratory_cycle_peak_inspiratory_flow_l_s": (0.8, 1.0),
    "respiratory_cycle_peak_expiratory_flow_l_s": (0.9, 1.0),
    "respiratory_cycle_end_expiratory_flow_l_s": (0.0, 0.5),
    "respiratory_cycle_peak_muscle_pressure_cmH2O": (6.0, 10.0),
    "respiratory_cycle_peak_airway_pressure_cmH2O": (0.0, 20.0),
    "respiratory_cycle_resistive_work_j_breath": (0.1, 0.5),
    "respiratory_cycle_muscle_work_j_breath": (0.25, 0.8),
    "respiratory_cycle_ventilator_work_j_breath": (0.0, 0.8),
    "respiratory_cycle_pv_hysteresis_j_breath": (0.03, 0.2),
    "respiratory_cycle_total_work_j_breath": (0.25, 1.0),
    "respiratory_cycle_expiratory_flow_limited_fraction": (0.0, 0.5),
    "respiratory_cycle_time_constant_s": (0.15, 0.8),
    "respiratory_ventilator_mean_trigger_delay_s": (0.10, 0.30),
    "respiratory_ventilator_mean_cycling_delay_s": (0.0, 0.60),
    "respiratory_ventilator_ineffective_trigger_fraction": (0.0, 0.50),
    "respiratory_ventilator_double_trigger_fraction": (0.0, 0.30),
    "respiratory_ventilator_premature_cycling_fraction": (0.0, 0.50),
    "respiratory_ventilator_delayed_cycling_fraction": (0.0, 0.50),
    "respiratory_ventilator_autotrigger_fraction": (0.0, 0.50),
    "respiratory_ventilator_asynchrony_index_pct": (5.0, 30.0),
    "respiratory_ventilator_patient_efforts_per_min": (14.0, 15.0),
    "respiratory_ventilator_breaths_per_min": (14.0, 15.0),
    "respiratory_ventilator_trigger_pressure_time_product_cmH2O_s": (0.2, 1.0),
    "respiratory_ventilator_neural_inspiratory_time_s": (1.4, 1.0),    "hematocrit_fraction": (0.40, 0.15),
    "hemoglobin_g_dl": (14.0, 5.0),
    "vo2_demand_ml_min": (250.0, 500.0),
    "vo2_ml_min": (250.0, 500.0),
    "oxygen_debt_ml_min": (0.0, 250.0),
    "aerobic_fraction": (1.0, 0.5),
    "effective_co2_ventilation_l_min": (4.2, 4.0),
    "glucagon_pg_ml": (60.0, 80.0),
    "sc_insulin_depot1_model_units": (0.0, 2.0),
    "sc_insulin_depot2_model_units": (0.0, 2.0),
    "sc_insulin_absorption_model_units_min": (0.0, 0.03),
    "glucagon_counterregulatory_egp_mg_kg_min": (0.0, 0.8),
    "icf_volume_l": (28.0, 5.0),
    "ecf_effective_tonicity_mOsm_l": (285.0, 25.0),
    "icf_effective_tonicity_mOsm_l": (285.0, 25.0),
    "osmotic_water_shift_l_min": (0.0, 0.5),
    "potassium_transcellular_flux_mmol_min": (0.0, 1.0),
    "potassium_transcellular_target_mmol_l": (4.2, 1.5),
    "sensor_glucose_mg_dl": (100.0, 50.0),
    "cgm_measurement_age_min": (0.0, 30.0),
    "monitor_measurement_age_min": (0.0, 30.0),
    "blood_gas_measurement_age_min": (15.0, 45.0),
    "chemistry_measurement_age_min": (30.0, 90.0),
    "hemodynamic_measurement_age_min": (5.0, 30.0),

}

# Preserve the established engineering normalization for compatibility
# observation fields, with explicit scales for the mechanics variables.
_LEGACY_CENTER = np.asarray([
    100,15,25,2,1,2,2,75,90,120,75,5.0,75,5,15,0.60,3.0,
    14,0.65,95,40,24,7.40,97,5,0.005,0.18,1.0,0.85,0.05,0.05,0.75,0.98,8.0,0.02,0.30,
    0.98,1.0,0.02,1.5,70.0,0.02,
    18.5,13.5,72,925,0.27,140,4.2,103,40,2.5,4.2,1.0,25,12,
    20.0,46.0,22.0,7.20,16.0,1.2,1.0,0.5,210.0,200.0,
    120,0.024,1,1,1,1,1,294,14,1,50,1,0.8,100,
], dtype=np.float32)
_LEGACY_SCALE = np.asarray([
    50,25,50,4,3,3,4,60,40,50,40,4,60,8,20,0.30,1.0,
    20,0.7,60,25,10,0.25,10,10,0.15,0.8,0.7,1.0,0.5,0.5,0.5,0.3,40.0,0.20,0.30,
    0.30,1.0,0.30,8.0,40.0,0.30,
    8,8,35,1000,0.40,15,2,15,10,5,2,1,10,10,
    8.0,25.0,8.0,0.25,10.0,2.0,2.0,1.5,100.0,200.0,
    70,0.05,4,5,5,5,3,25,5,0.7,100,2,2,150,
], dtype=np.float32)

_legacy_map = {
    name: (_LEGACY_CENTER[i], _LEGACY_SCALE[i])
    for i, name in enumerate(
        [n for n in OBSERVATION_NAMES if n not in _DEFAULT_CENTER_SCALE]
    )
}
def _normalization_for(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray([
        _DEFAULT_CENTER_SCALE.get(name, _legacy_map[name] if name in _legacy_map else (0.0, 1.0))[0]
        for name in names
    ], dtype=np.float32)
    scale = np.asarray([
        max(1e-6, _DEFAULT_CENTER_SCALE.get(name, _legacy_map[name] if name in _legacy_map else (0.0, 1.0))[1])
        for name in names
    ], dtype=np.float32)
    return center, scale

_OBS_CENTER, _OBS_SCALE = _normalization_for(OBSERVATION_NAMES)



class HumanHomeostasisEnv(_BaseEnv):
    """Gymnasium-compatible reduced-order multi-organ physiology environment.

    The environment couples metabolic, cardiovascular, renal, acid-base and
    respiratory dynamics. Respiratory mechanisms include multi-compartment V/Q
    exchange, true shunt, finite diffusion, hypoxic pulmonary vasoconstriction,
    recruitment, lung/chest-wall mechanics, within-breath pressure-flow-volume
    dynamics and patient-ventilator interaction. Whole-blood gas chemistry
    represents plasma/RBC carbonate speciation, hemoglobin buffering, Haldane
    and carbamino effects, and a locally balanced erythrocyte chloride shift.

    This remains a research scaffold, not a patient simulator.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        config: HumanConfig | None = None,
        scenario: str = "baseline",
        render_mode: str | None = None,
        observation_profile: str = "clinical",
        measurement_profile: str | None = None,
        measurement_config: ClinicalMeasurementConfig | None = None,
        info_profile: str = "debug",
    ):
        super().__init__()
        self.config = config or HumanConfig()
        for name in ("agent_step_min", "integration_step_min", "episode_minutes"):
            value = float(getattr(self.config, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.scenario = scenario
        self.active_scenario = scenario
        self.render_mode = render_mode
        if observation_profile not in {"clinical", "full"}:
            raise ValueError("observation_profile must be 'clinical' or 'full'")
        self.observation_profile = observation_profile
        if measurement_profile is None:
            measurement_profile = "realistic" if observation_profile == "clinical" else "ideal"
        if measurement_profile not in {"realistic", "ideal"}:
            raise ValueError("measurement_profile must be 'realistic' or 'ideal'")
        if observation_profile == "full" and measurement_profile != "ideal":
            raise ValueError("full observation_profile is a ground-truth debug state and requires measurement_profile='ideal'")
        self.measurement_profile = measurement_profile
        if info_profile not in {"debug", "benchmark"}:
            raise ValueError("info_profile must be 'debug' or 'benchmark'")
        self.info_profile = info_profile
        self.observation_names = (
            CLINICAL_OBSERVATION_NAMES if observation_profile == "clinical" else OBSERVATION_NAMES
        )
        self._obs_center, self._obs_scale = _normalization_for(self.observation_names)
        self.model = HumanPhysiology(self.config)
        self.measurement_model = (
            ClinicalMeasurementModel(measurement_config)
            if observation_profile == "clinical" and measurement_profile == "realistic"
            else None
        )
        self._scenario_warning: str | None = None

        self.action_space = spaces.Box(
            low=np.zeros(len(ACTION_NAMES), dtype=np.float32),
            high=np.ones(len(ACTION_NAMES), dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.full(len(self.observation_names), -1.0, dtype=np.float32),
            high=np.full(len(self.observation_names), 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self.state = HumanState()
        self.elapsed_minutes = 0.0
        self._last_reward_terms: dict[str, float] = {}
        self._needs_reset = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        if seed is not None and hasattr(self.action_space, "seed"):
            self.action_space.seed(seed + 1)

        options = options or {}
        scenario = options.get("scenario", self.scenario)
        self.active_scenario = scenario
        self._scenario_warning = None
        jitter = lambda sd: float(self.np_random.normal(0.0, sd))

        c = self.config
        ecf_volume = c.ecf_volume_baseline_l + jitter(max(0.02, 0.006 * c.ecf_volume_baseline_l))
        sodium = 140.0 + jitter(0.7)
        potassium = 4.2 + jitter(0.08)
        chloride = c.baseline_chloride_mmol_l + jitter(0.6)
        glucose = c.dalla_basal_glucose_mg_dl + jitter(2.0)

        rr0 = max(2.0, c.baseline_rr_bpm + jitter(0.8))
        vt0 = max(0.20, c.baseline_tidal_volume_l + jitter(0.025))
        va0 = max(0.5, rr0 * max(0.05, vt0 - c.dead_space_l))
        vco20 = max(40.0, c.baseline_vco2_ml_min + jitter(4.0))
        paco20 = 863.0 * (vco20 / 1000.0) / va0
        inspired_o2 = c.baseline_fio2 * (c.atmospheric_pressure_mmHg - c.water_vapor_pressure_mmHg)
        pao20 = max(20.0, inspired_o2 - paco20 / c.respiratory_quotient - c.baseline_aa_gradient_mmHg)
        po2den = pao20**3 + 150.0 * pao20
        spo20 = 0.0 if po2den <= 0.0 else 100.0 / (1.0 + 23400.0 / po2den)
        hco30 = c.baseline_bicarbonate_mmol_l + jitter(0.4)
        ph0 = 6.10 + np.log10(max(1e-6, hco30) / max(1e-6, 0.03 * paco20))

        self.state = HumanState(
            glucose_mg_dl=glucose,
            insulin_uU_ml=max(1.0, c.insulin_baseline_uU_ml + jitter(0.5)),
            glucagon_pg_ml=max(20.0, c.glucagon_baseline_pg_ml + jitter(3.0)),
            gut_carbs_g=0.0,
            liver_glycogen_g=c.liver_glycogen_baseline_g + jitter(3.0),
            muscle_glycogen_g=c.muscle_glycogen_baseline_g + jitter(8.0),
            lactate_mmol_l=max(0.5, c.baseline_lactate_mmol_l + jitter(0.08)),
            heart_rate_bpm=c.cv_resting_hr_bpm + jitter(2.0),
            map_mmHg=c.cv_map_setpoint_mmHg + jitter(2.0),
            plasma_volume_l=c.plasma_volume_baseline_l + jitter(max(0.01, 0.01 * c.plasma_volume_baseline_l)),
            respiratory_rate_bpm=rr0,
            tidal_volume_l=vt0,
            alveolar_ventilation_l_min=va0,
            pao2_mmHg=pao20,
            paco2_mmHg=paco20,
            bicarbonate_mmol_l=hco30,
            ph_arterial=ph0,
            spo2_pct=spo20,
            vo2_ml_min=c.baseline_vo2_ml_min + jitter(5.0),
            vo2_demand_ml_min=c.baseline_vo2_ml_min,
            vco2_ml_min=vco20,
            vco2_demand_ml_min=c.baseline_vco2_ml_min,
            oxidative_vco2_ml_min=vco20,
            vco2_generation_interval_average_ml_min=vco20,
            metabolic_respiratory_quotient=(
                c.baseline_vco2_ml_min / c.baseline_vo2_ml_min
            ),
            ventilation_efficiency=1.0,
            respiratory_drive_target_tidal_volume_l=vt0,
            respiratory_airway_resistance_cmH2O_s_l=c.respiratory_airway_resistance_cmH2O_s_l,
            respiratory_inertance_cmH2O_s2_l=c.respiratory_inertance_cmH2O_s2_l,
            respiratory_expiratory_resistance_multiplier=c.respiratory_expiratory_resistance_multiplier,
            respiratory_expiratory_flow_limit_l_s=c.respiratory_expiratory_flow_limit_l_s,
            respiratory_inspiratory_fraction=c.respiratory_inspiratory_fraction,
            pulmonary_shunt_fraction=c.pulmonary_baseline_shunt_fraction,
            pulmonary_vq_log_sd=c.pulmonary_baseline_vq_log_sd,
            pulmonary_diffusing_capacity_relative=c.pulmonary_baseline_diffusing_capacity_relative,
            pulmonary_hpv_function_fraction=c.pulmonary_hpv_baseline_function_fraction,
            total_body_water_l=c.total_body_water_baseline_l + jitter(max(0.05, 0.004 * c.total_body_water_baseline_l)),
            ecf_volume_l=ecf_volume,
            ecf_sodium_mmol=sodium * ecf_volume,
            ecf_potassium_mmol=potassium * ecf_volume,
            ecf_chloride_mmol=chloride * ecf_volume,
            icf_potassium_mmol=3400.0,
            sodium_mmol_l=sodium,
            potassium_mmol_l=potassium,
            chloride_mmol_l=chloride,
            plasma_osmolality_mOsm_kg=(
                2.0 * sodium
                + glucose / 18.0
                + self.config.baseline_bun_mg_dl / 2.8
            ),
            renal_function_fraction=1.0,
            gfr_ml_min=c.baseline_gfr_ml_min + jitter(4.0),
            adh_relative=max(0.2, 1.0 + jitter(0.08)),
            renin_relative=max(0.2, 1.0 + jitter(0.08)),
            angiotensin_ii_relative=max(0.2, 1.0 + jitter(0.08)),
            aldosterone_relative=max(0.2, 1.0 + jitter(0.08)),
            urine_flow_ml_min=max(0.2, 1.0 + jitter(0.08)),
            urine_sodium_mmol_min=0.070,
            urine_potassium_mmol_min=0.040,
            urine_chloride_mmol_min=0.070,
            renal_acid_excretion_mmol_min=0.040,
        )

        # Initialize glucose and insulin at the published normal-subject
        # Dalla Man basal steady state.
        self.model.metabolism.initialize_state(self.state)
        self.model.renal.initialize_state(self.state)
        # Calibrate the baseline physicochemical charge closure before applying
        # pathology/challenge scenarios; scenarios then perturb ions, lactate or PaCO2.
        self.model.acid_base.initialize_state(self.state)
        self._apply_scenario(scenario)
        # Close acute ECF/ICF water distribution after any scenario fluid perturbation.
        self.model.renal.equilibrate_transcellular_water(self.state, dt_min=None)
        # Scenario concentration and final t=0 water volume define the initial
        # lactate amount. Episode-time source/sink counters start at zero.
        self.model.energy_metabolism.initialize_state(self.state)
        # Short-horizon RBC/Hb mass is conserved. Scenario fluid shifts must
        # update Hct/[Hb] before blood-gas/O2 initialization.
        self.model._update_blood_composition(self.state, initialize=True)
        # Initialize the conserved closed-loop circulation after scenario volume
        # changes (e.g. dehydration), then warm it to a periodic state.
        self.model.cardiovascular.initialize_state(self.state)
        # Re-solve the scenario's plasma chemistry, then seed the conserved
        # exchangeable-CO2 pool from that explicit initial blood-gas state.
        self.model.acid_base.solve(self.state)
        self.model.respiratory_mechanics.step(self.state, dt_min=None)
        self.model.respiratory_cycle.initialize_state(self.state)
        self.model.blood_gas.initialize_state(self.state, fio2=self.config.baseline_fio2)
        self.model.pulmonary_exchange.estimate_arterial_oxygen(
            self.state, pco2_mmHg=self.state.paco2_mmHg,
            fio2=self.config.baseline_fio2, exercise=0.0, apply=True
        )
        self._reconcile_initial_arterial_gases()
        self.model.respiratory.update_oxygen_transport(self.state)
        self.model.respiratory.update_metabolic_gas_production(
            self.state, exercise=0.0
        )
        self.state.vco2_generation_interval_average_ml_min = float(
            self.state.vco2_ml_min
        )
        self.model.blood_gas.update_venous_diagnostics(self.state)
        self.state.initial_total_body_water_l = float(self.state.total_body_water_l)
        self.state.initial_ecf_sodium_mmol = float(self.state.ecf_sodium_mmol)
        self.state.initial_ecf_chloride_mmol = float(self.state.ecf_chloride_mmol)
        self.state.initial_total_exchangeable_potassium_mmol = float(
            self.state.ecf_potassium_mmol + self.state.icf_potassium_mmol
        )
        self.state.initial_nonvolatile_strong_anion_mEq = float(
            self.state.nonvolatile_strong_anion_mEq
        )

        # Scenario construction defines t=0. Any fluid/electrolyte changes used to
        # construct that initial condition must not also count as episode-time
        # administrations/losses in the conservation ledger.
        self.state.water_administered_l = 0.0
        self.state.sodium_administered_mmol = 0.0
        self.state.chloride_administered_mmol = 0.0
        self.state.water_lost_l = 0.0
        self.state.sodium_lost_mmol = 0.0
        self.state.potassium_lost_mmol = 0.0
        self.state.chloride_lost_mmol = 0.0

        self.state.nonvolatile_acid_generated_mEq = 0.0
        self.state.nonvolatile_acid_excreted_mEq = 0.0
        self.model.pbpk.refresh_concentrations_with_config(self.state)
        if not self._state_is_finite():
            raise FloatingPointError("Physiology reset produced a non-finite state")
        self.elapsed_minutes = 0.0
        self._last_reward_terms = {}
        self._needs_reset = False
        if self.measurement_model is not None:
            self.measurement_model.initialize(self.state, self.np_random)

        return self._get_obs(), self._get_info(scenario=scenario)

    def _reconcile_initial_arterial_gases(self) -> None:
        """Close reset-time carbon/O2/RER equilibrium without advancing fluxes."""
        c = self.config
        fio2 = float(c.baseline_fio2)
        closure = float("inf")
        elimination_endpoint_residual = float("inf")
        pulmonary_o2_residual = float("inf")
        coupled_tolerance = max(
            1e-6, 10.0 * c.co2_pool_solver_tolerance_mmol_l
        )

        # The pulmonary RER is elimination/VO2, so the physical elimination
        # endpoint must participate in the same fixed point as PaO2, PaCO2,
        # Haldane speciation and Fick-derived VO2/venous return.  Updating the
        # endpoint only after a gas solve would change RER and invalidate that
        # supposedly final oxygen state.
        for _ in range(32):
            self.model.respiratory.update_oxygen_transport(self.state)
            self.model.respiratory.update_metabolic_gas_production(
                self.state, exercise=0.0
            )
            self.model.blood_gas.update_venous_diagnostics(self.state)

            alveolar_dead = float(np.clip(
                self.state.pulmonary_alveolar_dead_space_fraction, 0.0, 0.80
            ))
            effective_co2_va = max(
                0.0, float(self.state.alveolar_ventilation_l_min)
            ) * (1.0 - alveolar_dead)
            self.state.effective_co2_ventilation_l_min = float(effective_co2_va)
            self.state.vco2_elimination_ml_min = float(
                max(0.0, self.state.paco2_mmHg)
                * effective_co2_va
                / 0.863
            )

            self.model.pulmonary_exchange.estimate_arterial_oxygen(
                self.state,
                pco2_mmHg=self.state.paco2_mmHg,
                fio2=fio2,
                exercise=0.0,
                dt_min=0.0,
                apply=True,
            )
            closure = self.model.blood_gas.arterial_carbon_pool_closure_residual_mmol_l(
                self.state, fio2=fio2, exercise=0.0
            )
            reconciled = abs(closure) > c.co2_pool_solver_tolerance_mmol_l
            if reconciled:
                self.model.blood_gas.reconcile_arterial_carbon_pool(
                    self.state, fio2=fio2, exercise=0.0
                )
            closure = self.model.blood_gas.arterial_carbon_pool_closure_residual_mmol_l(
                self.state, fio2=fio2, exercise=0.0
            )

            # These updates alter VO2 and mixed-venous return, hence pulmonary RER
            # and the next lung evaluation.  Include them before testing the fixed
            # point instead of leaving a one-iteration reset lag.
            self.model.respiratory.update_oxygen_transport(self.state)
            self.model.respiratory.update_metabolic_gas_production(
                self.state, exercise=0.0
            )
            self.model.blood_gas.update_venous_diagnostics(self.state)

            final_alveolar_dead = float(np.clip(
                self.state.pulmonary_alveolar_dead_space_fraction, 0.0, 0.80
            ))
            final_effective_co2_va = max(
                0.0, float(self.state.alveolar_ventilation_l_min)
            ) * (1.0 - final_alveolar_dead)
            expected_elimination = (
                max(0.0, float(self.state.paco2_mmHg))
                * final_effective_co2_va
                / 0.863
            )
            elimination_endpoint_residual = (
                expected_elimination - float(self.state.vco2_elimination_ml_min)
            )

            pulmonary_check = self.model.pulmonary_exchange.estimate_arterial_oxygen(
                self.state,
                pco2_mmHg=self.state.paco2_mmHg,
                fio2=fio2,
                exercise=0.0,
                dt_min=0.0,
                apply=False,
            )
            pulmonary_o2_residual = float(
                pulmonary_check.pao2_mmHg - self.state.pao2_mmHg
            )
            elimination_tolerance = 1e-8 * max(
                1.0, abs(expected_elimination)
            )
            if (
                not reconciled
                and abs(closure) <= coupled_tolerance
                and abs(elimination_endpoint_residual) <= elimination_tolerance
                and abs(pulmonary_o2_residual) <= 1e-4
            ):
                break
        else:
            raise FloatingPointError(
                "reset arterial carbon/O2/RER fixed point failed: "
                f"carbon={closure!r} mmol/L, "
                f"elimination_endpoint={elimination_endpoint_residual!r} mL/min, "
                f"pulmonary_O2={pulmonary_o2_residual!r} mmHg"
            )

        self.state.co2_final_gas_closure_residual_mmol_l = float(closure)
        if abs(closure) > coupled_tolerance:
            raise FloatingPointError(
                f"reset arterial O2/CO2 closure failed: {closure!r} mmol/L"
            )

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("step() called after episode completion; call reset()")
        action = np.asarray(action, dtype=np.float32)
        expected = (len(ACTION_NAMES),)
        if action.shape != expected:
            raise ValueError(f"Action must have shape {expected}, got {action.shape}.")
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains NaN or infinity.")
        action = np.clip(action, 0.0, 1.0)

        remaining_episode_min = float(self.config.episode_minutes - self.elapsed_minutes)
        if remaining_episode_min <= 1e-12:
            raise RuntimeError("step() called after the episode horizon; call reset()")
        planned_duration_min = min(float(self.config.agent_step_min), remaining_episode_min)
        intervention = self._decode_action(action)

        # Observe safety thresholds at the numerical integration cadence rather
        # than only after a potentially long agent step.  This prevents a severe
        # transient from crossing a terminal boundary and recovering before the
        # policy sees it.  A last finite snapshot also guarantees a valid terminal
        # observation if a numerical substep fails.
        start_time_min = float(self.elapsed_minutes)
        remaining_step_min = planned_duration_min
        terminated = False
        termination_reason: str | None = None

        # Instantaneous interventions are part of the same transactional
        # transition as integration.  Snapshot before applying them so a partial
        # bolus/fluid/PBPK update, including private solver-state mutation, can
        # never leak out of a failed action.
        pre_action_state = deepcopy(self.state)
        pre_action_runtime = deepcopy(self.model.runtime_snapshot())
        try:
            self.model.apply_instant_intervention(self.state, intervention)
            if (
                not self._state_is_finite()
                or not self._runtime_snapshot_is_finite(
                    self.model.runtime_snapshot()
                )
            ):
                raise FloatingPointError(
                    "Instant intervention produced non-finite state or runtime"
                )
            terminated, termination_reason = self._terminated()
        except ArithmeticError:
            self.state = pre_action_state
            self.model.restore_runtime_snapshot(pre_action_runtime)
            terminated = True
            termination_reason = "numerical_failure_nonfinite_state"
        except Exception:
            # Preserve debuggability for non-numerical/programming exceptions, but
            # still leave the environment in the exact pre-action state.
            self.state = pre_action_state
            self.model.restore_runtime_snapshot(pre_action_runtime)
            raise

        instant_action_pending = not terminated
        while not terminated and remaining_step_min > 1e-12:
            dt_min = min(float(self.config.integration_step_min), remaining_step_min)
            if instant_action_pending:
                # The instant action and its first continuous evolution form one
                # atomic transition.  If that first substep fails, roll both back;
                # after one successful substep the bolus is committed and later
                # failures retain all already-completed simulated time.
                last_finite_state = pre_action_state
                last_model_runtime = pre_action_runtime
            else:
                last_finite_state = deepcopy(self.state)
                last_model_runtime = self.model.runtime_snapshot()
            try:
                self.model.integrate(self.state, intervention, duration_min=dt_min)
            except FloatingPointError:
                self.state = last_finite_state
                self.model.restore_runtime_snapshot(last_model_runtime)
                terminated = True
                termination_reason = "numerical_failure_nonfinite_state"
                break
            if not self._state_is_finite():
                self.state = last_finite_state
                self.model.restore_runtime_snapshot(last_model_runtime)
                terminated = True
                termination_reason = "numerical_failure_nonfinite_state"
                break
            self.elapsed_minutes = min(
                float(self.config.episode_minutes), self.elapsed_minutes + dt_min
            )
            remaining_step_min -= dt_min
            instant_action_pending = False
            terminated, termination_reason = self._terminated()
            if self.measurement_model is not None:
                self.measurement_model.advance(
                    self.state,
                    time_min=self.elapsed_minutes,
                    dt_min=dt_min,
                    rng=self.np_random,
                    report_cgm=bool(terminated or remaining_step_min <= 1e-12),
                )
            if terminated:
                break

        actual_duration_min = float(self.elapsed_minutes - start_time_min)

        reward, reward_terms = self._reward(action)
        # A shortened final transition contributes proportionally less running
        # utility than a full agent interval.  The terminal penalty remains an
        # event penalty and is deliberately not duration-scaled.
        duration_fraction = actual_duration_min / REWARD_REFERENCE_INTERVAL_MIN
        reward_terms = {
            name: float(value * duration_fraction)
            for name, value in reward_terms.items()
            if name != "intervention_cost"
        }
        bolus_cost, continuous_cost = self._intervention_cost_components(action)
        reward_terms["intervention_cost"] = -float(
            bolus_cost + continuous_cost * duration_fraction
        )
        reward = float(sum(reward_terms.values()))
        if terminated:
            reward_terms["terminal"] = -10.0
            reward -= 10.0
        if not np.isfinite(reward):
            raise FloatingPointError("Reward calculation produced a non-finite value")

        self._last_reward_terms = reward_terms
        truncated = self.elapsed_minutes >= self.config.episode_minutes - 1e-12
        self._needs_reset = bool(terminated or truncated)

        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._get_info(
                action=action,
                intervention=intervention,
                termination_reason=termination_reason,
            ),
        )

    def render(self):
        if self.render_mode != "ansi":
            return None
        s = self.state
        return (
            f"t={self.elapsed_minutes:6.1f} min | "
            f"G={s.glucose_mg_dl:6.1f} | BP={s.systolic_pressure_mmHg:5.1f}/{s.diastolic_pressure_mmHg:4.1f} | "
            f"CO={s.cardiac_output_l_min:4.2f} L/min | MAP={s.map_mmHg:5.1f} | "
            f"PaO2={s.pao2_mmHg:5.1f} | PaCO2={s.paco2_mmHg:5.1f} | "
            f"pH={s.ph_arterial:4.2f} | Na={s.sodium_mmol_l:5.1f} | "
            f"K={s.potassium_mmol_l:4.2f} | Cl={s.chloride_mmol_l:5.1f} | "
            f"GFR={s.gfr_ml_min:5.1f} | drug={s.probe_plasma_mg_l:5.3f} mg/L"
        )

    def _apply_scenario(self, scenario: str) -> None:
        valid = {
            "baseline",
            "fasting",
            "ogtt",  # legacy name; Dalla Man core is a mixed-meal model, not an OGTT-specific validation
            "oral_glucose_75g",
            "meal",
            "mixed_meal_reference",
            "dehydrated",
            "hypoventilation",
            "respiratory_acidosis",
            "reduced_renal_function",
            "aki",  # backwards-compatible alias; not a diagnostic AKI definition
            "hyperkalemia",
            "transient_lactic_acidosis",
            "metabolic_acidosis",  # backwards-compatible alias
            "pbpk_oral_dose",
            "pk_target",
            "saline_challenge_30ml_kg",
            "vq_mismatch",
            "hpv_disabled_vq_mismatch",
            "pulmonary_shunt",
            "diffusion_limitation",
            "dependent_derecruitment",
            "recruitment_peep",
            "mechanical_ventilation_peep5",
            "mechanical_ventilation_peep12",
            "overdistension_peep18",
            "stiff_chest_wall",
            "low_lung_compliance",
            "airway_obstruction",
            "tachypnea_airway_obstruction",
            "pressure_control_ventilation",
            "pressure_control_obstruction",
            "pressure_support_synchronous",
            "pressure_support_ineffective_trigger",
            "pressure_support_ineffective_trigger_peep",
            "pressure_support_delayed_cycling",
            "pressure_support_delayed_cycling_optimized",
            "pressure_support_premature_cycling",
            "pressure_support_double_trigger",
            "pressure_support_autotrigger_leak",
        }
        if scenario not in valid:
            raise ValueError(f"Unknown scenario {scenario!r}; choose from {sorted(valid)}.")

        if scenario in {"ogtt", "oral_glucose_75g"}:
            self.model.metabolism.add_meal(self.state, 75.0)
            if scenario == "ogtt":
                self._scenario_warning = (
                    "'ogtt' is a legacy label. The Dalla Man 2007 core was identified for mixed-meal "
                    "physiology; this 75-g bolus is an OpenHumSim challenge, not an independently "
                    "validated OGTT reproduction. Prefer 'oral_glucose_75g'."
                )
        elif scenario in {"meal", "mixed_meal_reference"}:
            self.model.metabolism.add_meal(self.state, 60.0)
        elif scenario == "dehydrated":
            self.state.total_body_water_l -= 2.0
            self.state.ecf_volume_l -= 0.75
            self.state.plasma_volume_l -= 0.16
            self._refresh_electrolytes()
        elif scenario == "fasting":
            self.state.liver_glycogen_g *= 0.85
        elif scenario == "hypoventilation":
            self.state.ventilation_efficiency = self.config.hypoventilation_efficiency
            self.state.paco2_mmHg = 52.0
            self.state.pao2_mmHg = 70.0
        elif scenario == "respiratory_acidosis":
            self.state.ventilation_efficiency = self.config.respiratory_acidosis_efficiency
            self.state.paco2_mmHg = self.config.respiratory_acidosis_initial_paco2_mmHg
            self.state.pao2_mmHg = 70.0
            # pH/HCO3- are derived later by the physicochemical acid-base solver.
        elif scenario in {"reduced_renal_function", "aki"}:
            self.state.renal_function_fraction = 0.25
            self.state.gfr_ml_min = 30.0
            if scenario == "aki":
                self._scenario_warning = (
                    "'aki' is a legacy abstract low-filtration scenario; it does not "
                    "implement KDIGO AKI diagnostic criteria. Prefer 'reduced_renal_function'."
                )
        elif scenario == "hyperkalemia":
            target_ecf_k = 6.2 * self.state.ecf_volume_l
            delta_k = target_ecf_k - self.state.ecf_potassium_mmol
            self.state.ecf_potassium_mmol = target_ecf_k
            self.state.icf_potassium_mmol = max(
                500.0, self.state.icf_potassium_mmol - delta_k
            )
            self._refresh_electrolytes()
        elif scenario in {"transient_lactic_acidosis", "metabolic_acidosis"}:
            self.state.lactate_mmol_l = 4.0
            if scenario == "metabolic_acidosis":
                self._scenario_warning = (
                    "'metabolic_acidosis' is a legacy transient lactate challenge, not a "
                    "general metabolic-acidosis disease model. Prefer 'transient_lactic_acidosis'."
                )
        elif scenario == "vq_mismatch":
            self.state.pulmonary_vq_log_sd = self.config.pulmonary_vq_mismatch_log_sd
        elif scenario == "hpv_disabled_vq_mismatch":
            self.state.pulmonary_vq_log_sd = self.config.pulmonary_vq_mismatch_log_sd
            self.state.pulmonary_hpv_function_fraction = 0.0
        elif scenario == "dependent_derecruitment":
            self.state.pulmonary_recruitment_pressure_offset_cmH2O = self.config.pulmonary_derecruitment_challenge_offset_cmH2O
        elif scenario == "recruitment_peep":
            self.state.pulmonary_recruitment_pressure_offset_cmH2O = self.config.pulmonary_derecruitment_challenge_offset_cmH2O
            self.state.pulmonary_peep_cmH2O = self.config.pulmonary_recruitment_peep_cmH2O
            self.state.pulmonary_positive_pressure_fraction = 1.0
        elif scenario == "mechanical_ventilation_peep5":
            self.state.pulmonary_peep_cmH2O = self.config.pulmonary_mechanics_peep_low_cmH2O
            self.state.pulmonary_positive_pressure_fraction = 1.0
        elif scenario == "mechanical_ventilation_peep12":
            self.state.pulmonary_peep_cmH2O = self.config.pulmonary_mechanics_peep_high_cmH2O
            self.state.pulmonary_positive_pressure_fraction = 1.0
        elif scenario == "overdistension_peep18":
            self.state.pulmonary_peep_cmH2O = self.config.pulmonary_mechanics_peep_overdistension_cmH2O
            self.state.pulmonary_positive_pressure_fraction = 1.0
        elif scenario == "stiff_chest_wall":
            self.state.pulmonary_chest_wall_compliance_scale = self.config.pulmonary_stiff_chest_wall_scale
        elif scenario == "low_lung_compliance":
            self.state.pulmonary_lung_compliance_scale = self.config.pulmonary_low_lung_compliance_scale
        elif scenario == "airway_obstruction":
            self.state.respiratory_airway_resistance_cmH2O_s_l = self.config.respiratory_obstruction_resistance_cmH2O_s_l
            self.state.respiratory_expiratory_resistance_multiplier = self.config.respiratory_obstruction_expiratory_resistance_multiplier
            self.state.respiratory_expiratory_flow_limit_l_s = self.config.respiratory_obstruction_flow_limit_l_s
        elif scenario == "tachypnea_airway_obstruction":
            self.state.respiratory_airway_resistance_cmH2O_s_l = self.config.respiratory_obstruction_resistance_cmH2O_s_l
            self.state.respiratory_expiratory_resistance_multiplier = self.config.respiratory_obstruction_expiratory_resistance_multiplier
            self.state.respiratory_expiratory_flow_limit_l_s = self.config.respiratory_obstruction_flow_limit_l_s
            self.state.respiratory_cycle_rr_override_bpm = self.config.respiratory_obstruction_rr_bpm
        elif scenario == "pressure_control_ventilation":
            self.state.pulmonary_positive_pressure_fraction = 1.0
            self.state.pulmonary_peep_cmH2O = self.config.pulmonary_mechanics_peep_low_cmH2O
            self.state.respiratory_ventilator_pressure_control_cmH2O = self.config.respiratory_pressure_control_cmH2O
        elif scenario == "pressure_control_obstruction":
            self.state.pulmonary_positive_pressure_fraction = 1.0
            self.state.pulmonary_peep_cmH2O = self.config.pulmonary_mechanics_peep_low_cmH2O
            self.state.respiratory_ventilator_pressure_control_cmH2O = self.config.respiratory_pressure_control_cmH2O
            self.state.respiratory_airway_resistance_cmH2O_s_l = self.config.respiratory_obstruction_resistance_cmH2O_s_l
            self.state.respiratory_expiratory_resistance_multiplier = self.config.respiratory_obstruction_expiratory_resistance_multiplier
            self.state.respiratory_expiratory_flow_limit_l_s = self.config.respiratory_obstruction_flow_limit_l_s
            self.state.respiratory_cycle_rr_override_bpm = self.config.respiratory_obstruction_rr_bpm
        elif scenario == "pressure_support_synchronous":
            self._configure_pressure_support()
            self.state.respiratory_pressure_support_cmH2O = 3.0
            self.state.respiratory_neural_inspiratory_fraction = 0.15
            self.state.respiratory_cycleoff_fraction_peak_flow = 0.10
        elif scenario == "pressure_support_ineffective_trigger":
            self._configure_pressure_support()
            self.state.pulmonary_peep_cmH2O = 0.0
            self.state.respiratory_airway_resistance_cmH2O_s_l = self.config.respiratory_obstruction_resistance_cmH2O_s_l
            self.state.respiratory_expiratory_resistance_multiplier = self.config.respiratory_obstruction_expiratory_resistance_multiplier
            self.state.respiratory_expiratory_flow_limit_l_s = self.config.respiratory_obstruction_flow_limit_l_s
            self.state.respiratory_cycle_rr_override_bpm = self.config.respiratory_severe_obstruction_rr_bpm
            self.state.respiratory_expiratory_flow_limit_l_s = self.config.respiratory_severe_obstruction_flow_limit_l_s
            self.state.respiratory_muscle_drive_gain = 0.10
            self.state.respiratory_trigger_pressure_cmH2O = 0.50
        elif scenario == "pressure_support_ineffective_trigger_peep":
            self._configure_pressure_support()
            self.state.pulmonary_peep_cmH2O = self.config.respiratory_ps_external_peep_unloading_cmH2O
            self.state.respiratory_airway_resistance_cmH2O_s_l = self.config.respiratory_obstruction_resistance_cmH2O_s_l
            self.state.respiratory_expiratory_resistance_multiplier = self.config.respiratory_obstruction_expiratory_resistance_multiplier
            self.state.respiratory_expiratory_flow_limit_l_s = self.config.respiratory_obstruction_flow_limit_l_s
            self.state.respiratory_cycle_rr_override_bpm = self.config.respiratory_severe_obstruction_rr_bpm
            self.state.respiratory_expiratory_flow_limit_l_s = self.config.respiratory_severe_obstruction_flow_limit_l_s
            self.state.respiratory_muscle_drive_gain = 0.10
            self.state.respiratory_trigger_pressure_cmH2O = 0.50
            self.state.respiratory_cycleoff_fraction_peak_flow = self.config.respiratory_ps_optimized_cycleoff_fraction
        elif scenario == "pressure_support_delayed_cycling":
            self._configure_pressure_support()
            self.state.respiratory_pressure_support_cmH2O = self.config.respiratory_ps_high_support_cmH2O
            self.state.respiratory_cycleoff_fraction_peak_flow = self.config.respiratory_ps_low_cycleoff_fraction
            self.state.respiratory_airway_resistance_cmH2O_s_l = self.config.respiratory_obstruction_resistance_cmH2O_s_l
            self.state.respiratory_expiratory_resistance_multiplier = self.config.respiratory_obstruction_expiratory_resistance_multiplier
            self.state.respiratory_neural_inspiratory_fraction = 0.20
        elif scenario == "pressure_support_delayed_cycling_optimized":
            self._configure_pressure_support()
            self.state.respiratory_pressure_support_cmH2O = self.config.respiratory_ps_high_support_cmH2O
            self.state.respiratory_cycleoff_fraction_peak_flow = self.config.respiratory_ps_optimized_cycleoff_fraction
            self.state.respiratory_airway_resistance_cmH2O_s_l = self.config.respiratory_obstruction_resistance_cmH2O_s_l
            self.state.respiratory_expiratory_resistance_multiplier = self.config.respiratory_obstruction_expiratory_resistance_multiplier
            self.state.respiratory_neural_inspiratory_fraction = 0.20
        elif scenario == "pressure_support_premature_cycling":
            self._configure_pressure_support()
            self.state.respiratory_cycleoff_fraction_peak_flow = self.config.respiratory_ps_premature_cycleoff_fraction
            self.state.respiratory_neural_inspiratory_fraction = self.config.respiratory_ps_long_neural_inspiratory_fraction
            self.state.respiratory_muscle_drive_gain = self.config.respiratory_ps_high_drive_gain
        elif scenario == "pressure_support_double_trigger":
            self._configure_pressure_support()
            self.state.respiratory_cycleoff_fraction_peak_flow = self.config.respiratory_ps_premature_cycleoff_fraction
            self.state.respiratory_neural_inspiratory_fraction = self.config.respiratory_ps_long_neural_inspiratory_fraction
            self.state.respiratory_muscle_drive_gain = self.config.respiratory_ps_high_drive_gain
            self.state.respiratory_trigger_pressure_cmH2O = 0.3
            self.state.respiratory_allow_retrigger_same_effort = 1.0
        elif scenario == "pressure_support_autotrigger_leak":
            self._configure_pressure_support()
            self.state.respiratory_muscle_drive_gain = 0.15
            self.state.respiratory_leak_flow_l_s = self.config.respiratory_ps_autotrigger_leak_flow_l_s
        elif scenario == "pulmonary_shunt":
            self.state.pulmonary_shunt_fraction = self.config.pulmonary_shunt_challenge_fraction
        elif scenario == "diffusion_limitation":
            self.state.pulmonary_diffusing_capacity_relative = self.config.pulmonary_diffusion_limitation_relative
        elif scenario == "saline_challenge_30ml_kg":
            # Mirrors the total crystalloid dose (10 + 20 mL/kg) used in the
            # 2025 randomized human physiology study; timing is collapsed to an
            # instantaneous challenge for this reduced-order benchmark.
            from .renal import RenalIntervention
            self.model.renal.apply_instant_intervention(
                self.state,
                RenalIntervention(saline_ml=30.0 * self.config.body_weight_kg),
            )
        elif scenario == "pbpk_oral_dose":
            self.state.probe_gut_mg = 100.0
            self.state.probe_administered_mg = 100.0
        elif scenario == "pk_target":
            pass

    def _configure_pressure_support(self):
        c = self.config
        self.state.respiratory_ventilator_mode_code = 2
        self.state.pulmonary_positive_pressure_fraction = 1.0
        self.state.pulmonary_peep_cmH2O = c.pulmonary_mechanics_peep_low_cmH2O
        self.state.respiratory_pressure_support_cmH2O = c.respiratory_ps_pressure_support_cmH2O
        self.state.respiratory_trigger_pressure_cmH2O = c.respiratory_ps_trigger_pressure_cmH2O
        self.state.respiratory_trigger_flow_l_s = c.respiratory_ps_trigger_flow_l_s
        self.state.respiratory_cycleoff_fraction_peak_flow = c.respiratory_ps_cycleoff_fraction_peak_flow
        self.state.respiratory_pressure_support_rise_time_s = c.respiratory_ps_rise_time_s
        self.state.respiratory_pressure_support_max_ti_s = c.respiratory_ps_max_inspiratory_time_s
        self.state.respiratory_neural_inspiratory_fraction = c.respiratory_neural_inspiratory_fraction

    def _refresh_electrolytes(self):
        c = self.config
        self.state.sodium_mmol_l = self.state.ecf_sodium_mmol / self.state.ecf_volume_l
        self.state.potassium_mmol_l = self.state.ecf_potassium_mmol / self.state.ecf_volume_l
        self.state.chloride_mmol_l = self.state.ecf_chloride_mmol / self.state.ecf_volume_l
        self.state.plasma_osmolality_mOsm_kg = (
            2.0 * self.state.sodium_mmol_l
            + self.state.glucose_mg_dl / 18.0
            + c.baseline_bun_mg_dl / 2.8
        )

    def _decode_action(
        self,
        action: np.ndarray,
    ) -> Intervention:
        c = self.config
        fio2 = c.baseline_fio2 + float(action[4]) * (c.max_fio2 - c.baseline_fio2)
        return Intervention(
            insulin_model_units=float(action[0] * c.max_insulin_model_units_per_step),
            oral_carbs_g=float(action[1] * c.max_carbs_g_per_step),
            exercise_intensity=float(action[2]),
            saline_ml=float(action[3] * c.max_saline_ml_per_step),
            fio2=fio2,
            ventilation_pressure_assist_cmH2O=float(
                action[5] * c.max_ventilation_pressure_assist_cmH2O
            ),
            ventilation_support_l_min=0.0,
            oral_water_ml=float(action[6] * c.max_oral_water_ml_per_step),
            oral_probe_mg=float(action[7] * c.max_probe_drug_mg_per_step),
        )

    def _get_obs(self) -> np.ndarray:
        values = []
        for name in self.observation_names:
            if name == "time_to_go_fraction":
                values.append(max(
                    0.0,
                    1.0 - self.elapsed_minutes / self.config.episode_minutes,
                ))
            elif self.measurement_model is not None:
                values.append(
                    self.measurement_model.measurement_value(name, self.state)
                )
            elif name == "sensor_glucose_mg_dl":
                values.append(float(self.state.glucose_mg_dl))
            elif name in ClinicalMeasurementModel.AGE_NAMES:
                values.append(0.0)
            else:
                values.append(float(getattr(self.state, name)))
        raw = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(raw)):
            raise FloatingPointError("Observation contains a non-finite physiological value")
        z = (raw - self._obs_center) / self._obs_scale
        obs = np.tanh(z).astype(np.float32)
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError("Observation normalization produced a non-finite value")
        return obs

    def _reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        s = self.state

        glucose_error = ((s.glucose_mg_dl - 100.0) / 45.0) ** 2
        hypo = max(0.0, 70.0 - s.glucose_mg_dl) / 20.0
        severe_hypo = max(0.0, 54.0 - s.glucose_mg_dl) / 15.0
        hyper = max(0.0, s.glucose_mg_dl - 180.0) / 100.0
        map_error = max(0.0, abs(s.map_mmHg - 90.0) - 15.0) / 35.0
        tachy = max(0.0, s.heart_rate_bpm - 130.0) / 70.0
        hypoxemia = max(0.0, 92.0 - s.spo2_pct) / 12.0
        co2_error = max(0.0, abs(s.paco2_mmHg - 40.0) - 5.0) / 25.0
        ph_error = max(0.0, abs(s.ph_arterial - 7.40) - 0.04) / 0.20
        sodium_error = max(0.0, abs(s.sodium_mmol_l - 140.0) - 4.0) / 15.0
        potassium_error = max(0.0, abs(s.potassium_mmol_l - 4.2) - 0.5) / 2.0
        low_gfr = max(0.0, 60.0 - s.gfr_ml_min) / 60.0
        excessive_o2_extraction = max(0.0, s.oxygen_extraction_ratio - 0.65) / 0.15
        oxygen_debt = max(0.0, s.oxygen_debt_ml_min) / max(100.0, s.vo2_demand_ml_min)
        asynchrony = max(0.0, s.respiratory_ventilator_asynchrony_index_pct - 10.0) / 40.0
        auto_peep = max(0.0, s.respiratory_cycle_auto_peep_cmH2O - 2.0) / 6.0
        overdistension = max(0.0, s.pulmonary_overdistension_fraction - 0.20) / 0.60
        high_strain = max(0.0, s.pulmonary_lung_strain - 0.35) / 0.65
        high_mechanical_power = max(0.0, s.pulmonary_mechanical_power_j_min - 12.0) / 15.0
        excessive_vt = max(0.0, s.tidal_volume_l - 0.75) / 0.75
        low_co = max(0.0, 3.5 - s.cardiac_output_l_min) / 2.0

        bolus_cost, continuous_cost = self._intervention_cost_components(action)
        intervention_cost = bolus_cost + continuous_cost

        in_homeostasis = (
            70.0 <= s.glucose_mg_dl <= 140.0
            and 70.0 <= s.map_mmHg <= 110.0
            and s.spo2_pct >= 94.0
            and 7.32 <= s.ph_arterial <= 7.48
            and 135.0 <= s.sodium_mmol_l <= 145.0
            and 3.5 <= s.potassium_mmol_l <= 5.0
        )

        terms: dict[str, float] = {
            "homeostasis": 0.25 if in_homeostasis else 0.0,
            "glucose_error": -0.18 * glucose_error,
            "hypoglycemia": -1.00 * hypo**2,
            "severe_hypoglycemia": -3.00 * severe_hypo**2,
            "hyperglycemia": -0.35 * hyper**2,
            "map": -0.25 * map_error**2,
            "tachycardia": -0.10 * tachy**2,
            "hypoxemia": -1.25 * hypoxemia**2,
            "co2": -0.35 * co2_error**2,
            "ph": -0.70 * ph_error**2,
            "sodium": -0.45 * sodium_error**2,
            "potassium": -0.70 * potassium_error**2,
            "renal_function": -0.25 * low_gfr**2,
            "oxygen_delivery": -0.60 * excessive_o2_extraction**2,
            "oxygen_debt": -1.20 * oxygen_debt**2,
            "ventilator_asynchrony": -0.45 * asynchrony**2,
            "auto_peep": -0.35 * auto_peep**2,
            "overdistension": -0.70 * overdistension**2,
            "lung_strain": -0.50 * high_strain**2,
            "mechanical_power": -0.25 * high_mechanical_power**2,
            "excessive_tidal_volume": -0.35 * excessive_vt**2,
            "low_cardiac_output": -0.40 * low_co**2,
            "intervention_cost": -intervention_cost,
        }

        if self.active_scenario == "pk_target":
            target = self.config.pbpk_target_effect_site_mg_l
            exposure_error = (s.probe_effect_site_mg_l - target) / max(0.1, target)
            terms["pk_target"] = -0.40 * exposure_error**2
            if abs(exposure_error) <= 0.20:
                terms["pk_in_target"] = 0.40
            if s.probe_plasma_mg_l > self.config.pbpk_high_exposure_mg_l:
                excess = (
                    s.probe_plasma_mg_l - self.config.pbpk_high_exposure_mg_l
                ) / self.config.pbpk_high_exposure_mg_l
                terms["pk_high_exposure"] = -2.0 * excess**2

        return float(sum(terms.values())), {k: float(v) for k, v in terms.items()}

    @staticmethod
    def _intervention_cost_components(action: np.ndarray) -> tuple[float, float]:
        """Separate per-decision boluses from time-held control penalties."""
        bolus = (
            0.020 * float(action[0] ** 2)
            + 0.012 * float(action[1] ** 2)
            + 0.004 * float(action[3] ** 2)
            + 0.003 * float(action[6] ** 2)
            + 0.006 * float(action[7] ** 2)
        )
        continuous = (
            0.006 * float(action[2] ** 2)
            + 0.006 * float(action[4] ** 2)
            + 0.015 * float(action[5] ** 2)
        )
        return float(bolus), float(continuous)

    def _terminated(self) -> tuple[bool, str | None]:
        c = self.config
        s = self.state
        if not self._state_is_finite():
            return True, "numerical_failure_nonfinite_state"
        checks = (
            (s.total_body_water_l <= 0.0, "invalid_nonpositive_total_body_water"),
            (s.ecf_volume_l <= 0.0, "invalid_nonpositive_ecf_volume"),
            (s.ecf_volume_l > s.total_body_water_l, "invalid_ecf_exceeds_total_body_water"),
            (s.ecf_sodium_mmol < 0.0, "invalid_negative_sodium_mass"),
            (s.ecf_potassium_mmol < 0.0, "invalid_negative_potassium_mass"),
            (s.ecf_chloride_mmol < 0.0, "invalid_negative_chloride_mass"),
            (s.lactate_amount_mmol < 0.0, "invalid_negative_lactate_amount"),
            (s.glucose_mg_dl < c.glucose_min_terminate, "severe_hypoglycemia"),
            (s.glucose_mg_dl > c.glucose_max_terminate, "extreme_hyperglycemia"),
            (s.map_mmHg < c.map_min_terminate, "circulatory_failure_low_map"),
            (s.map_mmHg > c.map_max_terminate, "extreme_hypertension"),
            (s.spo2_pct < c.spo2_min_terminate, "severe_hypoxemia"),
            (s.pao2_mmHg < c.pao2_min_terminate, "critical_low_pao2"),
            (s.paco2_mmHg > c.paco2_max_terminate, "critical_hypercapnia"),
            (s.ph_arterial < c.ph_min_terminate, "critical_acidemia"),
            (s.ph_arterial > c.ph_max_terminate, "critical_alkalemia"),
            (s.sodium_mmol_l < c.sodium_min_terminate, "critical_hyponatremia"),
            (s.sodium_mmol_l > c.sodium_max_terminate, "critical_hypernatremia"),
            (s.potassium_mmol_l < c.potassium_min_terminate, "critical_hypokalemia"),
            (s.potassium_mmol_l > c.potassium_max_terminate, "critical_hyperkalemia"),
        )
        for condition, reason in checks:
            if condition:
                return True, reason
        return False, None

    def _state_is_finite(self) -> bool:
        return all(np.isfinite(float(value)) for value in self.state.as_dict().values())

    @staticmethod
    def _runtime_snapshot_is_finite(value) -> bool:
        """Recursively validate the numeric private state used for rollback."""
        if isinstance(value, dict):
            return all(
                HumanHomeostasisEnv._runtime_snapshot_is_finite(item)
                for item in value.values()
            )
        if isinstance(value, (tuple, list)):
            return all(
                HumanHomeostasisEnv._runtime_snapshot_is_finite(item)
                for item in value
            )
        if isinstance(value, np.ndarray):
            return bool(np.all(np.isfinite(value)))
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(np.isfinite(value))
        return True

    def _get_info(
        self,
        *,
        scenario: str | None = None,
        action: np.ndarray | None = None,
        intervention: Intervention | None = None,
        termination_reason: str | None = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "time_min": float(self.elapsed_minutes),
            "state": deepcopy(self.state.as_dict()),
            "reward_terms": deepcopy(self._last_reward_terms),
            "observation_names": self.observation_names,
            "observation_profile": self.observation_profile,
            "measurement_profile": self.measurement_profile,
            "info_profile": self.info_profile,
            "measurement": (
                self.measurement_model.diagnostics()
                if self.measurement_model is not None
                else {"profile": "ideal", "ages_min": {name: 0.0 for name in ClinicalMeasurementModel.AGE_NAMES}}
            ),
            "action_names": ACTION_NAMES,
            "gymnasium_installed": HAS_GYMNASIUM,
            "environment_semantics": {
                "fully_observed_markov_state": False,
                "classification": "POMDP-style observation over a larger hidden mechanistic state",
                "clinical_validation": False,
                "finite_horizon_time_observed": True,
                "post_terminal_step_requires_reset": True,
                "state_schema_version": "0.22",
                "reward_profile": "homeostasis_v0.21",
            },
            "oxygen_transport": {
                "arterial_o2_content_ml_dl": float(self.state.arterial_o2_content_ml_dl),
                "mixed_venous_o2_content_ml_dl": float(self.state.mixed_venous_o2_content_ml_dl),
                "mixed_venous_o2_sat_pct": float(self.state.mixed_venous_o2_sat_pct),
                "oxygen_delivery_ml_min": float(self.state.oxygen_delivery_ml_min),
                "oxygen_extraction_ratio": float(self.state.oxygen_extraction_ratio),
                "oxygen_supply_margin_ml_min": float(self.state.oxygen_supply_margin_ml_min),
                "fick_residual_ml_min": float(
                    self.state.vo2_ml_min
                    - self.state.cardiac_output_l_min * 10.0
                    * (self.state.arterial_o2_content_ml_dl - self.state.mixed_venous_o2_content_ml_dl)
                ),
            },
            "mass_balance": {
                "total_body_water_l": float(self.state.total_body_water_l),
                "ecf_sodium_mmol": float(self.state.ecf_sodium_mmol),
                "ecf_potassium_mmol": float(self.state.ecf_potassium_mmol),
                "ecf_chloride_mmol": float(self.state.ecf_chloride_mmol),
                "total_exchangeable_potassium_mmol": float(
                    self.state.ecf_potassium_mmol + self.state.icf_potassium_mmol
                ),
                "probe_administered_mg": float(self.state.probe_administered_mg),
                "probe_current_body_mg": float(self.state.probe_total_body_mg),
                "probe_eliminated_mg": float(
                    self.state.probe_eliminated_hepatic_mg
                    + self.state.probe_eliminated_renal_mg
                ),
                "probe_mass_balance_error_mg": float(
                    self.state.probe_mass_balance_error_mg
                ),
                "water_mass_balance_error_l": float(
                    self.state.total_body_water_l
                    - (
                        self.state.initial_total_body_water_l
                        + self.state.water_administered_l
                        - self.state.water_lost_l
                    )
                ),
                "water_partition_residual_l": float(
                    self.state.total_body_water_l
                    - self.state.ecf_volume_l
                    - self.state.icf_volume_l
                ),
                "sc_insulin_mass_balance_error_model_units": float(
                    self.state.sc_insulin_mass_balance_error_model_units
                ),
                "sodium_mass_balance_error_mmol": float(
                    self.state.ecf_sodium_mmol
                    - (
                        self.state.initial_ecf_sodium_mmol
                        + self.state.sodium_administered_mmol
                        - self.state.sodium_lost_mmol
                    )
                ),
                "chloride_mass_balance_error_mmol": float(
                    self.state.ecf_chloride_mmol
                    - (
                        self.state.initial_ecf_chloride_mmol
                        + self.state.chloride_administered_mmol
                        - self.state.chloride_lost_mmol
                    )
                ),
                "potassium_mass_balance_error_mmol": float(
                    (self.state.ecf_potassium_mmol + self.state.icf_potassium_mmol)
                    - (
                        self.state.initial_total_exchangeable_potassium_mmol
                        - self.state.potassium_lost_mmol
                    )
                ),
                "nonvolatile_acid_mass_balance_error_mEq": float(
                    self.state.nonvolatile_strong_anion_mEq
                    - (
                        self.state.initial_nonvolatile_strong_anion_mEq
                        + self.state.nonvolatile_acid_generated_mEq
                        - self.state.nonvolatile_acid_excreted_mEq
                    )
                ),
                "exchangeable_co2_pool_mmol": float(self.state.exchangeable_co2_pool_mmol),
                "co2_generated_mmol": float(self.state.co2_generated_mmol),
                "co2_eliminated_mmol": float(self.state.co2_eliminated_mmol),
                "co2_urinary_bicarbonate_loss_mmol": float(
                    self.state.co2_urinary_bicarbonate_loss_mmol
                ),
                "co2_mass_balance_error_mmol": float(self.state.co2_mass_balance_error_mmol),
                "co2_final_gas_closure_residual_mmol_l": float(
                    self.state.co2_final_gas_closure_residual_mmol_l
                ),
                "lactate_amount_mmol": float(self.state.lactate_amount_mmol),
                "lactate_generated_mmol": float(self.state.lactate_generated_mmol),
                "lactate_cleared_mmol": float(self.state.lactate_cleared_mmol),
                "lactate_mass_balance_error_mmol": float(
                    self.state.lactate_mass_balance_error_mmol
                ),
            },
            "energy_metabolism": {
                "model": "achieved-VO2 oxidative CO2 + apparent-volume lactate amount ledger",
                "vo2_demand_ml_min": float(self.state.vo2_demand_ml_min),
                "vo2_achieved_ml_min": float(self.state.vo2_ml_min),
                "instantaneous_oxygen_deficit_ml_min": float(
                    self.state.instantaneous_oxygen_deficit_ml_min
                ),
                "cumulative_oxygen_deficit_ml": float(
                    self.state.cumulative_oxygen_deficit_ml
                ),
                "metabolic_respiratory_quotient": float(
                    self.state.metabolic_respiratory_quotient
                ),
                "oxidative_vco2_ml_min": float(self.state.oxidative_vco2_ml_min),
                "pulmonary_vco2_elimination_ml_min": float(
                    self.state.vco2_elimination_ml_min
                ),
                "lactate_distribution_volume_l": float(
                    self.state.lactate_distribution_volume_l
                ),
                "lactate_amount_mmol": float(self.state.lactate_amount_mmol),
                "lactate_concentration_mmol_l": float(self.state.lactate_mmol_l),
                "lactate_production_mmol_min": float(
                    self.state.lactate_production_mmol_min
                ),
                "lactate_clearance_mmol_min": float(
                    self.state.lactate_clearance_mmol_min
                ),
            },
            "acid_base": {
                "model": "Stewart-Figge plasma electroneutrality + explicit carbonate + whole-blood RBC CO2 coupling",
                "pH": float(self.state.ph_arterial),
                "paco2_mmHg": float(self.state.paco2_mmHg),
                "bicarbonate_mmol_l": float(self.state.bicarbonate_mmol_l),
                "carbonate_mmol_l": float(self.state.carbonate_mmol_l),
                "total_co2_mmol_l": float(self.state.total_co2_mmol_l),
                "albumin_g_dl": float(self.state.albumin_g_dl),
                "phosphate_mmol_l": float(self.state.phosphate_mmol_l),
                "sida_mEq_l": float(self.state.strong_ion_difference_apparent_mEq_l),
                "side_mEq_l": float(self.state.strong_ion_difference_effective_mEq_l),
                "strong_ion_gap_mEq_l": float(self.state.strong_ion_gap_mEq_l),
                "charge_balance_residual_mEq_l": float(self.state.charge_balance_residual_mEq_l),
                "henderson_hasselbalch_residual": float(self.state.henderson_hasselbalch_residual),
                "urine_ammonium_mmol_min": float(self.state.urine_ammonium_mmol_min),
                "urine_titratable_acid_mmol_min": float(self.state.urine_titratable_acid_mmol_min),
                "urine_bicarbonate_mmol_min": float(self.state.urine_bicarbonate_mmol_min),
            },
            "pulmonary_exchange": {
                "model": "six-compartment V/Q distribution + true shunt + finite O2 diffusion",
                "shunt_fraction": float(self.state.pulmonary_shunt_fraction),
                "vq_log_sd": float(self.state.pulmonary_vq_log_sd),
                "mean_vq_ratio": float(self.state.pulmonary_mean_vq_ratio),
                "low_vq_perfusion_fraction": float(self.state.pulmonary_low_vq_perfusion_fraction),
                "high_vq_ventilation_fraction": float(self.state.pulmonary_high_vq_ventilation_fraction),
                "capillary_transit_time_s": float(self.state.pulmonary_capillary_transit_time_s),
                "diffusion_equilibration_fraction": float(self.state.pulmonary_diffusion_equilibration_fraction),
                "aa_gradient_mmHg": float(self.state.pulmonary_aa_gradient_mmHg),
                "alveolar_dead_space_fraction": float(self.state.pulmonary_alveolar_dead_space_fraction),
                "enghoff_dead_space_fraction": float(self.state.pulmonary_enghoff_dead_space_fraction),
                "mixed_expired_pco2_mmHg": float(self.state.pulmonary_mixed_expired_pco2_mmHg),
            },
            "blood_gas_carbon": {
                "model": "reduced whole-blood CO2 species + Funder-Wieth Donnan + Haldane/carbamino + conserved exchangeable CO2",
                "arterial_total_co2_mmol_l_blood": float(self.state.arterial_total_co2_mmol_l_blood),
                "mixed_venous_total_co2_mmol_l_blood": float(self.state.mixed_venous_total_co2_mmol_l_blood),
                "mixed_venous_pco2_mmHg": float(self.state.mixed_venous_pco2_mmHg),
                "mixed_venous_ph": float(self.state.mixed_venous_ph),
                "rbc_ph": float(self.state.rbc_ph),
                "rbc_bicarbonate_mmol_l": float(self.state.rbc_bicarbonate_mmol_l),
                "rbc_chloride_mmol_l": float(self.state.rbc_chloride_mmol_l),
                "carbamino_co2_mmol_l_blood": float(self.state.carbamino_co2_mmol_l_blood),
                "hemoglobin_buffer_capacity_mEq_l_pH": float(self.state.hemoglobin_buffer_capacity_mEq_l_pH),
                "hemoglobin_bound_proton_change_mEq_l": float(self.state.hemoglobin_bound_proton_change_mEq_l),
                "chloride_shift_plasma_mmol_l": float(self.state.chloride_shift_plasma_mmol_l),
                "chloride_shift_balance_residual_mmol_l_blood": float(
                    self.state.chloride_shift_balance_residual_mmol_l_blood
                ),
                "haldane_co2_content_gain_mmol_l": float(self.state.haldane_co2_content_gain_mmol_l),
                "co2_fick_content_residual_mmol_l": float(self.state.co2_fick_content_residual_mmol_l),
                "co2_content_solver_residual_mmol_l": float(self.state.co2_content_solver_residual_mmol_l),
                "co2_final_gas_closure_residual_mmol_l": float(
                    self.state.co2_final_gas_closure_residual_mmol_l
                ),
                "vco2_elimination_ml_min": float(self.state.vco2_elimination_ml_min),
            },
            "pbpk": {
                "plasma_concentration_mg_l": float(self.state.probe_plasma_mg_l),
                "effect_site_concentration_mg_l": float(self.state.probe_effect_site_mg_l),
                "hepatic_clearance_l_min": float(self.state.probe_hepatic_clearance_l_min),
                "renal_clearance_l_min": float(self.state.probe_renal_clearance_l_min),
                "total_tissue_flow_l_min": float(self.state.probe_total_tissue_flow_l_min),
                "hepatic_eliminated_mg": float(self.state.probe_eliminated_hepatic_mg),
                "renal_eliminated_mg": float(self.state.probe_eliminated_renal_mg),
                "mass_balance_error_mg": float(self.state.probe_mass_balance_error_mg),
            },
            "metabolism": {
                "model": "Dalla Man-Rizza-Cobelli 2007 normal parameter set",
                "ra_mg_kg_min": float(self.state.dalla_ra_mg_kg_min),
                "egp_mg_kg_min": float(self.state.dalla_egp_mg_kg_min),
                "utilization_mg_kg_min": float(self.state.dalla_u_mg_kg_min),
                "insulin_secretion_pmol_kg_min": float(self.state.dalla_insulin_secretion_pmol_kg_min),
                "gi_mass_balance_error_mg": float(self.state.dalla_gi_mass_balance_error_mg),
                "numerical_positivity_correction": float(self.state.dalla_numerical_positivity_correction),
                "exercise_extension_active": bool(getattr(self.state, "dalla_u_mg_kg_min", 0.0) >= 0.0),
            },
            "cardiovascular": {
                "model": "8-compartment closed-loop 0D circulation",
                "cardiac_output_l_min": float(self.state.cardiac_output_l_min),
                "stroke_volume_ml": float(self.state.stroke_volume_ml),
                "systolic_pressure_mmHg": float(self.state.systolic_pressure_mmHg),
                "diastolic_pressure_mmHg": float(self.state.diastolic_pressure_mmHg),
                "central_venous_pressure_mmHg": float(self.state.central_venous_pressure_mmHg),
                "pulmonary_artery_pressure_mmHg": float(self.state.pulmonary_artery_pressure_mmHg),
                "ejection_fraction": float(self.state.cv_ejection_fraction),
                "total_blood_volume_ml": float(self.state.cv_total_blood_volume_ml),
                "blood_volume_error_ml": float(self.state.cv_blood_volume_error_ml),
                "numerical_volume_correction_ml": float(self.state.cv_numerical_volume_correction_ml),
            },
        }
        if scenario is not None:
            info["scenario"] = scenario
        if self._scenario_warning is not None:
            info["scenario_warning"] = self._scenario_warning
        if action is not None:
            info["action"] = {
                name: float(value) for name, value in zip(ACTION_NAMES, action)
            }
        if intervention is not None:
            info["intervention"] = {
                "insulin_model_units": float(intervention.insulin_model_units),
                "oral_carbs_g": float(intervention.oral_carbs_g),
                "exercise_intensity": float(intervention.exercise_intensity),
                "saline_ml": float(intervention.saline_ml),
                "fio2": float(intervention.fio2),
                "ventilation_pressure_assist_cmH2O": float(
                    intervention.ventilation_pressure_assist_cmH2O
                ),
                "legacy_ventilation_support_l_min": float(
                    intervention.ventilation_support_l_min
                ),
                "oral_water_ml": float(intervention.oral_water_ml),
                "oral_probe_mg": float(intervention.oral_probe_mg),
            }
        if termination_reason is not None:
            info["termination_reason"] = termination_reason
        if self.info_profile == "benchmark":
            return {
                key: info[key]
                for key in BENCHMARK_INFO_KEYS
                if key in info
            }
        return info
