from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from math import exp
import numpy as np

from .config import HumanConfig
from .oxygen_binding import OxygenBindingModel
from .respiratory import alveolar_oxygen_tension_mmHg, effective_pulmonary_rer


@dataclass(frozen=True)
class PulmonaryExchangeResult:
    pao2_mmHg: float
    sao2_fraction: float
    arterial_o2_content_ml_dl: float
    effective_respiratory_exchange_ratio: float
    mean_alveolar_pao2_mmHg: float
    aa_gradient_mmHg: float
    mean_vq_ratio: float
    low_vq_perfusion_fraction: float
    high_vq_ventilation_fraction: float
    capillary_transit_time_s: float
    diffusion_equilibration_fraction: float
    mixed_expired_pco2_mmHg: float
    alveolar_dead_space_fraction: float
    enghoff_dead_space_fraction: float
    recruitment_fraction: float
    derecruited_fraction: float
    mean_distending_pressure_cmH2O: float
    hpv_resistance_multiplier: float
    perfusion_redistribution_index: float
    hpv_diverted_flow_fraction: float
    effective_capillary_blood_volume_ml: float
    hypoxic_perfusion_fraction: float


class MultiCompartmentPulmonaryExchangeModel:
    """Six-unit V/Q model with HPV and recruitment/derecruitment.

    Content mixing and finite diffusion are coupled to dynamic regional
    perfusion and aeration:

    * local alveolar hypoxia raises precapillary resistance (HPV),
    * parallel vascular conductances redistribute perfusion between units,
    * the equivalent parallel resistance feeds the whole-lung circulation,
    * regional recruitment follows pressure thresholds with hysteresis,
    * derecruited units receive little ventilation but can remain perfused,
      creating low-V/Q / shunt-like gas exchange without an arbitrary PaO2 loss.

    This remains a reduced forward model, not a MIGET inversion, CT lung model,
    or patient-specific mechanical-ventilation simulator.
    """

    _Z = np.asarray([-1.9, -1.15, -0.38, 0.38, 1.15, 1.9], dtype=float)
    _BASE_Q = np.asarray([0.06, 0.15, 0.29, 0.29, 0.15, 0.06], dtype=float)
    _RECRUIT_ATTRS = tuple(f"pulmonary_recruitment_u{i}" for i in range(6))
    _HPV_ATTRS = tuple(f"pulmonary_hpv_tone_u{i}" for i in range(6))

    def __init__(self, config: HumanConfig):
        self.cfg = config
        self.oxygen_binding = OxygenBindingModel(config)

    @staticmethod
    def _severinghaus_sat(po2_mmHg: float) -> float:
        # Standard-curve compatibility helper. Pulmonary mixing uses
        # OxygenBindingModel with pH/PCO2-dependent affinity.
        return OxygenBindingModel.standard_severinghaus_saturation(po2_mmHg)

    def _o2_content(
        self, po2_mmHg: float, *, ph: float, pco2_mmHg: float, hemoglobin_g_dl: float
    ) -> float:
        return self.oxygen_binding.content_ml_dl(
            po2_mmHg, ph, pco2_mmHg, hemoglobin_g_dl
        )

    def _po2_from_content(
        self, content_ml_dl: float, *, ph: float, pco2_mmHg: float, hemoglobin_g_dl: float
    ) -> float:
        return self.oxygen_binding.po2_from_content(
            content_ml_dl, ph, pco2_mmHg, hemoglobin_g_dl
        )

    def _venous_po2(self, state) -> float:
        return self._po2_from_content(
            max(0.0, float(state.mixed_venous_o2_content_ml_dl)),
            ph=float(getattr(state, "mixed_venous_ph", state.ph_arterial - 0.04)),
            pco2_mmHg=float(getattr(state, "mixed_venous_pco2_mmHg", state.paco2_mmHg + 6.0)),
            hemoglobin_g_dl=float(getattr(state, "hemoglobin_g_dl", self.cfg.hemoglobin_g_dl)),
        )

    @staticmethod
    def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))

    def _mean_distending_pressure(self, state) -> float:
        # Recruitment is driven by explicit transpulmonary pressure from the
        # lung/chest-wall mechanics model rather than Paw/PEEP alone.
        value = float(getattr(state, "pulmonary_mean_transpulmonary_pressure_cmH2O", 0.0))
        if np.isfinite(value):
            return value
        c = self.cfg
        driving = float(state.tidal_volume_l) / max(1e-6, c.pulmonary_static_compliance_l_cmH2O)
        return float(state.pulmonary_peep_cmH2O) + c.pulmonary_mean_inspiratory_pressure_fraction * driving

    def _update_recruitment(self, state, dt_min: float | None) -> np.ndarray:
        c = self.cfg
        current = np.asarray([float(getattr(state, a)) for a in self._RECRUIT_ATTRS], dtype=float)
        pressure = self._mean_distending_pressure(state)
        thresholds = np.asarray(c.pulmonary_unit_closing_pressures_cmH2O, dtype=float)
        thresholds = thresholds + float(state.pulmonary_recruitment_pressure_offset_cmH2O)

        # Opening requires a somewhat higher pressure than remaining open.
        opening = thresholds + c.pulmonary_recruitment_opening_hysteresis_cmH2O
        effective_threshold = np.where(current < 0.95, opening, thresholds)
        target = self._sigmoid(
            (pressure - effective_threshold) / max(1e-6, c.pulmonary_recruitment_logistic_width_cmH2O)
        )

        if dt_min is None:
            new = np.asarray(target, dtype=float)
        else:
            tau = np.where(target > current, c.pulmonary_recruitment_open_tau_min, c.pulmonary_recruitment_close_tau_min)
            alpha = 1.0 - np.exp(-max(0.0, float(dt_min)) / np.maximum(1e-6, tau))
            new = current + alpha * (target - current)
        new = np.clip(new, 0.0, 1.0)
        for a, v in zip(self._RECRUIT_ATTRS, new):
            setattr(state, a, float(v))
        state.pulmonary_mean_distending_pressure_cmH2O = float(pressure)
        return new

    def _hpv_activation(self, local_pao2: np.ndarray, state) -> np.ndarray:
        c = self.cfg
        function = float(np.clip(state.pulmonary_hpv_function_fraction, 0.0, 1.0))
        x = (c.pulmonary_hpv_po2_half_mmHg - local_pao2) / max(1e-6, c.pulmonary_hpv_slope_mmHg)
        return function * np.asarray(self._sigmoid(x), dtype=float)

    def _update_hpv_tones(self, state, target: np.ndarray, dt_min: float | None) -> np.ndarray:
        c = self.cfg
        current = np.asarray([float(getattr(state, a)) for a in self._HPV_ATTRS], dtype=float)
        if dt_min is None:
            new = target
        else:
            alpha = 1.0 - exp(-max(0.0, float(dt_min)) / max(1e-6, c.pulmonary_hpv_tau_min))
            new = current + alpha * (target - current)
        new = np.clip(new, 0.0, 1.0)
        for a, v in zip(self._HPV_ATTRS, new):
            setattr(state, a, float(v))
        return new

    def _regional_distribution(self, state, pco2_mmHg: float, fio2: float, recruitment: np.ndarray, dt_min: float | None):
        c = self.cfg
        fio2_fraction = float(np.clip(fio2, 0.0, 1.0))
        pulmonary_rer = effective_pulmonary_rer(state, c)
        sigma = max(0.0, float(state.pulmonary_vq_log_sd))
        rel_vq = np.exp(self._Z * sigma)
        rel_vq /= max(1e-12, float(np.sum(self._BASE_Q * rel_vq)))

        # Aeration gates regional ventilation.  A tiny floor avoids a singular
        # algebraic V/Q while still making nearly closed units functionally shunt-like.
        vent_gate = c.pulmonary_min_ventilation_weight_when_closed + (1.0 - c.pulmonary_min_ventilation_weight_when_closed) * recruitment
        v_raw = self._BASE_Q * rel_vq * vent_gate
        v_frac = v_raw / max(1e-12, float(v_raw.sum()))

        co = max(0.25, float(state.cardiac_output_l_min))
        va = max(0.20, float(state.alveolar_ventilation_l_min))
        shunt = float(np.clip(state.pulmonary_shunt_fraction, 0.0, 0.80))
        perfused_flow = max(1e-6, co * (1.0 - shunt))

        tones_at_start = np.asarray(
            [float(getattr(state, a)) for a in self._HPV_ATTRS], dtype=float
        )
        tones = tones_at_start.copy()
        max_mult = c.pulmonary_hpv_max_local_resistance_multiplier

        # Two fixed-point passes are sufficient for this six-unit reduced system.
        # Each pass predicts the *same* end-of-step kinetic state from the tone at
        # the beginning of the outer step.  Advancing the stored tone on every
        # algebraic pass would apply dt twice and halve the configured HPV time
        # constant.
        target_tones = tones.copy()
        if dt_min is None:
            alpha = 1.0
        else:
            alpha = 1.0 - exp(
                -max(0.0, float(dt_min)) / max(1e-6, c.pulmonary_hpv_tau_min)
            )
        for _ in range(2):
            local_r = 1.0 + (max_mult - 1.0) * tones
            conductance = self._BASE_Q / np.maximum(local_r, 1e-6)
            q_frac = conductance / max(1e-12, float(conductance.sum()))
            local_vq = (va * v_frac) / np.maximum(1e-9, perfused_flow * q_frac)

            # Regional PACO2 from inverse V/Q, normalized to preserve the
            # ventilation-weighted candidate arterial PCO2.
            global_vq = va / perfused_flow
            inv_rel = global_vq / np.maximum(local_vq, 1e-9)
            norm = float(np.sum(v_frac * inv_rel))
            local_paco2 = float(pco2_mmHg) * inv_rel / max(1e-12, norm)
            pio2 = fio2_fraction * (
                c.atmospheric_pressure_mmHg - c.water_vapor_pressure_mmHg
            )
            local_pao2 = np.maximum(1.0, alveolar_oxygen_tension_mmHg(
                inspired_o2_mmHg=pio2,
                pco2_mmHg=local_paco2,
                fio2=fio2_fraction,
                pulmonary_rer=pulmonary_rer,
            ))
            target_tones = self._hpv_activation(local_pao2, state)
            tones = tones_at_start + alpha * (target_tones - tones_at_start)

        # Commit the temporal update exactly once, after the algebraic target has
        # converged. State still contains tones_at_start here.
        tones = self._update_hpv_tones(state, target_tones, dt_min)

        local_r = 1.0 + (max_mult - 1.0) * tones
        conductance = self._BASE_Q / np.maximum(local_r, 1e-6)
        q_frac = conductance / max(1e-12, float(conductance.sum()))
        equivalent_r_multiplier = 1.0 / max(1e-12, float(conductance.sum()))
        local_vq = (va * v_frac) / np.maximum(1e-9, perfused_flow * q_frac)
        global_vq = va / perfused_flow
        inv_rel = global_vq / np.maximum(local_vq, 1e-9)
        norm = float(np.sum(v_frac * inv_rel))
        local_paco2 = float(pco2_mmHg) * inv_rel / max(1e-12, norm)
        pio2 = fio2_fraction * (
            c.atmospheric_pressure_mmHg - c.water_vapor_pressure_mmHg
        )
        local_pao2 = np.maximum(1.0, alveolar_oxygen_tension_mmHg(
            inspired_o2_mmHg=pio2,
            pco2_mmHg=local_paco2,
            fio2=fio2_fraction,
            pulmonary_rer=pulmonary_rer,
        ))

        redistribution = 0.5 * float(np.sum(np.abs(q_frac - self._BASE_Q)))
        return q_frac, v_frac, local_vq, local_paco2, local_pao2, global_vq, equivalent_r_multiplier, redistribution

    def _diffusion_fraction(self, state, exercise: float, recruitment_fraction: float) -> tuple[float, float, float]:
        c = self.cfg
        co_ml_s = max(1e-6, float(state.cardiac_output_l_min) * 1000.0 / 60.0)
        capillary_volume = c.pulmonary_capillary_blood_volume_ml * (1.0 + c.pulmonary_capillary_recruitment_gain * exercise)
        # Derecruited tissue contributes less exchangeable capillary volume.
        capillary_volume *= 0.35 + 0.65 * float(np.clip(recruitment_fraction, 0.0, 1.0))
        transit = float(np.clip(capillary_volume / co_ml_s, 0.08, 2.0))
        dl_rel = max(0.05, float(state.pulmonary_diffusing_capacity_relative))
        dl_rel *= 1.0 + c.pulmonary_diffusing_capacity_exercise_gain * exercise
        dl_rel *= 0.30 + 0.70 * float(np.clip(recruitment_fraction, 0.0, 1.0))
        tau = c.pulmonary_o2_equilibration_tau_s / max(1e-6, dl_rel)
        equil = float(np.clip(1.0 - exp(-transit / max(1e-6, tau)), 0.0, 1.0))
        return transit, equil, float(capillary_volume)

    def estimate_arterial_oxygen(
        self,
        state,
        *,
        pco2_mmHg: float,
        fio2: float,
        exercise: float,
        dt_min: float | None = None,
        apply: bool = False,
    ) -> PulmonaryExchangeResult:
        c = self.cfg
        # A result-only evaluation must not advance recruitment/HPV kinetics or
        # alter any diagnostic on the caller's live state.  HumanState consists of
        # scalar fields, so a shallow copy is a complete isolated working state.
        working_state = state if apply else copy(state)
        exercise = float(np.clip(exercise, 0.0, 1.0))
        pulmonary_rer = effective_pulmonary_rer(working_state, c)
        recruitment = self._update_recruitment(working_state, dt_min)
        recruitment_fraction = float(np.sum(self._BASE_Q * recruitment))
        q, v_frac, local_vq, local_paco2, local_pao2, global_vq, hpv_r, redistribution = self._regional_distribution(
            working_state, pco2_mmHg, fio2, recruitment, dt_min
        )
        shunt = float(np.clip(working_state.pulmonary_shunt_fraction, 0.0, 0.80))

        pv_o2 = self._venous_po2(working_state)
        transit, equil, effective_capillary_volume = self._diffusion_fraction(
            working_state, exercise, recruitment_fraction
        )
        endcap_po2 = pv_o2 + equil * (local_pao2 - pv_o2)
        hb = float(getattr(working_state, "hemoglobin_g_dl", c.hemoglobin_g_dl))
        # Local pH follows the regional PACO2 at the current systemic bicarbonate
        # concentration. This is reduced but preserves the Bohr direction across V/Q units.
        hco3 = max(1e-6, float(working_state.bicarbonate_mmol_l))
        local_ph = c.carbonic_acid_pka + np.log10(
            hco3
            / np.maximum(
                1e-6,
                c.co2_solubility_mmol_l_mmHg * local_paco2,
            )
        )
        endcap_content = np.asarray([
            self._o2_content(float(po2), ph=float(ph), pco2_mmHg=float(pc), hemoglobin_g_dl=hb)
            for po2, ph, pc in zip(endcap_po2, local_ph, local_paco2)
        ], dtype=float)
        ventilated_cap_content = float(np.sum(q * endcap_content))
        venous_content = max(
            0.0, float(working_state.mixed_venous_o2_content_ml_dl)
        )
        arterial_content = (1.0 - shunt) * ventilated_cap_content + shunt * venous_content
        pao2 = self._po2_from_content(
            arterial_content,
            ph=float(working_state.ph_arterial),
            pco2_mmHg=float(pco2_mmHg),
            hemoglobin_g_dl=hb,
        )
        sao2 = self.oxygen_binding.saturation(
            pao2, float(working_state.ph_arterial), float(pco2_mmHg)
        ).saturation_fraction

        fio2_fraction = float(np.clip(fio2, 0.0, 1.0))
        pio2 = fio2_fraction * (
            c.atmospheric_pressure_mmHg - c.water_vapor_pressure_mmHg
        )
        ideal_global_pao2 = max(0.0, float(alveolar_oxygen_tension_mmHg(
            inspired_o2_mmHg=pio2,
            pco2_mmHg=pco2_mmHg,
            fio2=fio2_fraction,
            pulmonary_rer=pulmonary_rer,
        )))
        aa = max(0.0, ideal_global_pao2 - pao2)
        mean_alv = float(np.sum(v_frac * local_pao2))
        low_q = float(np.sum(q[local_vq < 0.5]))
        high_v = float(np.sum(v_frac[local_vq > 2.0]))

        alveolar_fraction = float(np.clip(
            (working_state.tidal_volume_l - c.dead_space_l)
            / max(1e-6, working_state.tidal_volume_l),
            0.0,
            1.0,
        ))
        high_vq_waste = float(np.sum(v_frac * np.maximum(0.0, 1.0 - 1.0 / np.maximum(local_vq, 1.0))))
        alveolar_dead_space = float(np.clip(0.25 * high_vq_waste, 0.0, 0.40))
        mixed_expired = alveolar_fraction * (1.0 - alveolar_dead_space) * float(np.sum(v_frac * local_paco2))
        enghoff = float(np.clip((float(pco2_mmHg) - mixed_expired) / max(1e-6, float(pco2_mmHg)), 0.0, 0.95))

        result = PulmonaryExchangeResult(
            pao2_mmHg=float(pao2), sao2_fraction=float(sao2),
            arterial_o2_content_ml_dl=float(arterial_content),
            effective_respiratory_exchange_ratio=float(pulmonary_rer),
            mean_alveolar_pao2_mmHg=mean_alv, aa_gradient_mmHg=float(aa),
            mean_vq_ratio=float(global_vq), low_vq_perfusion_fraction=low_q,
            high_vq_ventilation_fraction=high_v,
            capillary_transit_time_s=transit,
            diffusion_equilibration_fraction=equil,
            mixed_expired_pco2_mmHg=float(mixed_expired),
            alveolar_dead_space_fraction=alveolar_dead_space,
            enghoff_dead_space_fraction=enghoff,
            recruitment_fraction=recruitment_fraction,
            derecruited_fraction=1.0-recruitment_fraction,
            mean_distending_pressure_cmH2O=float(
                working_state.pulmonary_mean_distending_pressure_cmH2O
            ),
            hpv_resistance_multiplier=float(hpv_r),
            perfusion_redistribution_index=float(redistribution),
            hpv_diverted_flow_fraction=float(redistribution),
            effective_capillary_blood_volume_ml=effective_capillary_volume,
            hypoxic_perfusion_fraction=float(np.sum(q[local_pao2 < 60.0])),
        )
        if apply:
            self.apply_result(state, result)
        return result

    @staticmethod
    def apply_result(state, result: PulmonaryExchangeResult):
        state.pao2_mmHg = result.pao2_mmHg
        state.spo2_pct = 100.0 * result.sao2_fraction
        state.arterial_o2_content_ml_dl = result.arterial_o2_content_ml_dl
        state.pulmonary_mean_alveolar_pao2_mmHg = result.mean_alveolar_pao2_mmHg
        state.pulmonary_aa_gradient_mmHg = result.aa_gradient_mmHg
        state.pulmonary_mean_vq_ratio = result.mean_vq_ratio
        state.pulmonary_low_vq_perfusion_fraction = result.low_vq_perfusion_fraction
        state.pulmonary_high_vq_ventilation_fraction = result.high_vq_ventilation_fraction
        state.pulmonary_capillary_transit_time_s = result.capillary_transit_time_s
        state.pulmonary_diffusion_equilibration_fraction = result.diffusion_equilibration_fraction
        state.pulmonary_mixed_expired_pco2_mmHg = result.mixed_expired_pco2_mmHg
        state.pulmonary_alveolar_dead_space_fraction = result.alveolar_dead_space_fraction
        state.pulmonary_enghoff_dead_space_fraction = result.enghoff_dead_space_fraction
        state.pulmonary_recruitment_fraction = result.recruitment_fraction
        state.pulmonary_derecruited_fraction = result.derecruited_fraction
        state.pulmonary_mean_distending_pressure_cmH2O = result.mean_distending_pressure_cmH2O
        state.pulmonary_hpv_resistance_multiplier = result.hpv_resistance_multiplier
        state.pulmonary_perfusion_redistribution_index = result.perfusion_redistribution_index
        state.pulmonary_hpv_diverted_flow_fraction = result.hpv_diverted_flow_fraction
        state.pulmonary_effective_capillary_blood_volume_ml = result.effective_capillary_blood_volume_ml
        state.pulmonary_hypoxic_perfusion_fraction = result.hypoxic_perfusion_fraction
        return state
