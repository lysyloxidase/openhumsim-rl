from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
import numpy as np

from .config import HumanConfig


@dataclass(frozen=True)
class RespiratoryCycleResult:
    tidal_volume_l: float
    end_expiratory_volume_above_relaxed_l: float
    dynamic_hyperinflation_l: float
    auto_peep_cmH2O: float
    peak_inspiratory_flow_l_s: float
    peak_expiratory_flow_l_s: float
    end_expiratory_flow_l_s: float
    peak_muscle_pressure_cmH2O: float
    peak_airway_pressure_cmH2O: float
    resistive_work_j_breath: float
    muscle_work_j_breath: float
    ventilator_work_j_breath: float
    pv_hysteresis_j_breath: float
    total_mechanical_work_j_breath: float
    expiratory_flow_limited_fraction: float
    respiratory_time_constant_s: float


class DynamicRespiratoryCycleModel:
    """Within-breath single-compartment respiratory mechanics.

    The dynamic equation is

        I * dQ/dt + R(Q) * Q + E_phase * V = Paw + Pmus

    where Q is flow and V is volume above the zero-pressure relaxed volume.
    `Pmus` is represented as a positive inspiratory driving-pressure magnitude;
    the physical inspiratory muscle pressure is negative relative to atmosphere.

    A semi-implicit update is used for the resistive term, which is stable for
    the small inertance of the adult respiratory system without forcing the
    whole-body integrator onto millisecond timesteps.

    This is a reduced single-compartment forward model. It is not a validated
    ventilator, airway tree, or patient-specific respiratory-muscle model.
    """

    CMH2O_L_TO_J = 0.0980665

    def __init__(self, config: HumanConfig):
        self.cfg = config
        self.last_trace: dict[str, np.ndarray] = {}
        # Samples belonging to the currently incomplete breath.  Outer physiology
        # steps are allowed to be shorter than a respiratory period, so per-breath
        # quantities must not be reconstructed from only the latest outer step.
        self._partial_breath: dict[str, np.ndarray] = {}
        self._last_end_expiratory_volume_l = 0.0

    def runtime_snapshot(self) -> tuple[dict, dict, float]:
        """Return the private cycle state needed for transactional rollback.

        A step replaces rather than mutates arrays already held in these
        dictionaries, so shallow mapping copies are sufficient and inexpensive.
        """
        return (
            dict(self._partial_breath),
            dict(self.last_trace),
            float(self._last_end_expiratory_volume_l),
        )

    def restore_runtime_snapshot(self, snapshot: tuple[dict, dict, float]) -> None:
        self._partial_breath = dict(snapshot[0])
        self.last_trace = dict(snapshot[1])
        self._last_end_expiratory_volume_l = float(snapshot[2])

    def initialize_state(self, state) -> None:
        c = self.cfg
        self._partial_breath = {}
        self.last_trace = {}
        crs = max(0.015, float(state.pulmonary_respiratory_system_compliance_l_cmH2O))
        e = 1.0 / crs
        peep = max(0.0, float(state.pulmonary_peep_cmH2O))
        state.respiratory_cycle_volume_above_relaxed_l = peep / e
        self._last_end_expiratory_volume_l = float(peep / e)
        state.respiratory_cycle_flow_l_s = 0.0
        state.respiratory_cycle_phase_s = 0.0
        state.respiratory_cycle_dynamic_hyperinflation_l = 0.0
        state.respiratory_cycle_auto_peep_cmH2O = 0.0
        state.respiratory_cycle_end_expiratory_alveolar_pressure_cmH2O = peep
        state.respiratory_cycle_time_constant_s = (
            max(0.1, float(state.respiratory_airway_resistance_cmH2O_s_l)) * crs
        )
        state.respiratory_cycle_peak_inspiratory_flow_l_s = 0.0
        state.respiratory_cycle_peak_expiratory_flow_l_s = 0.0
        state.respiratory_cycle_end_expiratory_flow_l_s = 0.0
        state.respiratory_cycle_peak_muscle_pressure_cmH2O = 0.0
        state.respiratory_cycle_peak_airway_pressure_cmH2O = peep
        state.respiratory_cycle_resistive_work_j_breath = 0.0
        state.respiratory_cycle_muscle_work_j_breath = 0.0
        state.respiratory_cycle_ventilator_work_j_breath = 0.0
        state.respiratory_cycle_pv_hysteresis_j_breath = 0.0
        state.respiratory_cycle_total_work_j_breath = 0.0
        state.respiratory_cycle_expiratory_flow_limited_fraction = 0.0
        if int(round(float(getattr(state, "respiratory_ventilator_mode_code", 0)))) != 2:
            state.respiratory_pressure_support_cmH2O = c.respiratory_ps_pressure_support_cmH2O
            state.respiratory_trigger_pressure_cmH2O = c.respiratory_ps_trigger_pressure_cmH2O
            state.respiratory_trigger_flow_l_s = c.respiratory_ps_trigger_flow_l_s
            state.respiratory_cycleoff_fraction_peak_flow = c.respiratory_ps_cycleoff_fraction_peak_flow
            state.respiratory_pressure_support_rise_time_s = c.respiratory_ps_rise_time_s
            state.respiratory_pressure_support_max_ti_s = c.respiratory_ps_max_inspiratory_time_s
            state.respiratory_neural_inspiratory_fraction = c.respiratory_neural_inspiratory_fraction
        state.respiratory_ventilator_active = 0.0
        state.respiratory_ventilator_inspiration_elapsed_s = 0.0
        state.respiratory_ventilator_peak_flow_l_s = 0.0
        state.respiratory_ventilator_refractory_remaining_s = 0.0
        state.respiratory_ventilator_triggered_current_effort = 0.0
        state.respiratory_ventilator_triggers_current_effort = 0.0
        state.respiratory_ventilator_last_trigger_delay_s = 0.0
        state.respiratory_ventilator_mean_trigger_delay_s = 0.0
        state.respiratory_ventilator_mean_cycling_delay_s = 0.0
        state.respiratory_ventilator_ineffective_trigger_fraction = 0.0
        state.respiratory_ventilator_double_trigger_fraction = 0.0
        state.respiratory_ventilator_premature_cycling_fraction = 0.0
        state.respiratory_ventilator_delayed_cycling_fraction = 0.0
        state.respiratory_ventilator_autotrigger_fraction = 0.0
        state.respiratory_ventilator_asynchrony_index_pct = 0.0
        state.respiratory_ventilator_patient_efforts_per_min = 0.0
        state.respiratory_ventilator_breaths_per_min = 0.0
        state.respiratory_ventilator_trigger_pressure_time_product_cmH2O_s = 0.0
        state.respiratory_ventilator_neural_inspiratory_time_s = 0.0

    @staticmethod
    def _pressure_control_shape(frac_insp: float, rise_fraction: float) -> float:
        rise = max(1e-3, min(0.95, rise_fraction))
        if frac_insp < rise:
            x = frac_insp / rise
            return 0.5 * (1.0 - cos(pi * x))
        return 1.0

    def step(self, state, dt_min: float, ventilation_pressure_assist_cmH2O: float = 0.0) -> RespiratoryCycleResult:
        c = self.cfg
        dt_min = max(1e-6, float(dt_min))

        rr = float(state.respiratory_rate_bpm)
        override_rr = float(getattr(state, "respiratory_cycle_rr_override_bpm", 0.0))
        if override_rr > 0.0:
            rr = override_rr
            state.respiratory_rate_bpm = rr
        rr = float(np.clip(rr, 2.0, 60.0))

        target_vt = max(0.05, float(getattr(state, "respiratory_drive_target_tidal_volume_l", state.tidal_volume_l)))
        override_vt = float(getattr(state, "respiratory_cycle_target_vt_override_l", 0.0))
        if override_vt > 0.0:
            target_vt = override_vt

        positive_fraction = float(np.clip(state.pulmonary_positive_pressure_fraction, 0.0, 1.0))
        peep = max(0.0, float(state.pulmonary_peep_cmH2O))
        crs = max(0.015, float(state.pulmonary_respiratory_system_compliance_l_cmH2O))
        elastance = 1.0 / crs

        raw = max(0.05, float(state.respiratory_airway_resistance_cmH2O_s_l))
        inertance = max(1e-4, float(state.respiratory_inertance_cmH2O_s2_l))
        exp_r_mult = max(1.0, float(state.respiratory_expiratory_resistance_multiplier))
        flow_limit = max(0.05, float(state.respiratory_expiratory_flow_limit_l_s))

        cycle_s = 60.0 / rr
        insp_fraction = float(np.clip(state.respiratory_inspiratory_fraction, 0.15, 0.70))
        tinsp = cycle_s * insp_fraction
        qmean = target_vt / max(0.10, tinsp)

        mode_code = int(round(float(getattr(state, "respiratory_ventilator_mode_code", 0))))
        psv_mode = mode_code == 2
        neural_fraction = float(np.clip(
            getattr(state, "respiratory_neural_inspiratory_fraction", c.respiratory_neural_inspiratory_fraction),
            0.15, 0.70,
        ))
        neural_tinsp = cycle_s * neural_fraction
        state.respiratory_ventilator_neural_inspiratory_time_s = neural_tinsp

        # The neural/ventilator controller is intentionally simple: it estimates
        # the pressure required for the requested VT from current elastance and
        # a representative inspiratory flow. Actual VT then emerges from the ODE.
        pressure_need = (target_vt * elastance + raw * qmean) * c.respiratory_drive_pressure_calibration
        prior_auto = max(0.0, float(state.respiratory_cycle_auto_peep_cmH2O))
        peep_counterbalance = c.respiratory_external_peep_threshold_unloading_fraction * peep
        threshold_load = max(0.0, prior_auto - peep_counterbalance)

        configured_pc = max(0.0, float(state.respiratory_ventilator_pressure_control_cmH2O))
        action_assist = max(0.0, float(ventilation_pressure_assist_cmH2O))
        pc_amp = max(configured_pc, action_assist) if (configured_pc > 0.0 or action_assist > 0.0) else pressure_need
        pc_amp *= positive_fraction

        muscle_gain = max(0.0, float(state.respiratory_muscle_drive_gain))
        if psv_mode:
            # Short-term unloading: pressure support reduces the patient pressure
            # required for a target neural VT rather than simply adding PS on top
            # of unchanged effort. A small residual drive preserves triggering.
            ps_for_drive = max(0.0, float(getattr(state, "respiratory_pressure_support_cmH2O", c.respiratory_ps_pressure_support_cmH2O)))
            unloaded_drive = pressure_need + threshold_load - c.respiratory_ps_neural_unloading_fraction * ps_for_drive
            pmus_amp = max(c.respiratory_ps_min_patient_drive_cmH2O, unloaded_drive) * muscle_gain
            pc_amp = 0.0
        else:
            # A transient action-driven pressure assist augments a spontaneously
            # breathing patient; it must not erase Pmus merely because Paw becomes
            # positive. Only an explicitly configured pressure-control scenario is
            # treated as passive controlled ventilation in this reduced model.
            controlled_pc = configured_pc > 0.0 and positive_fraction > 0.0
            patient_fraction = 0.0 if controlled_pc else 1.0
            pmus_amp = (pressure_need + threshold_load) * patient_fraction * muscle_gain

        raw_phase_s = max(0.0, float(state.respiratory_cycle_phase_s))
        if raw_phase_s >= cycle_s:
            # A sudden increase in RR can make the old phase exceed the new
            # period before simulated time advances. This controller phase reset
            # is not a completed measured breath; discard its fragment so it
            # cannot be glued to the following cycle.
            self._partial_breath = {}
            phase_s = 0.0
        else:
            phase_s = raw_phase_s
        volume = max(0.0, float(state.respiratory_cycle_volume_above_relaxed_l))
        flow = float(state.respiratory_cycle_flow_l_s)

        # 10 ms default is enough to resolve normal human tidal breathing while
        # remaining cheap relative to whole-body simulation. Shorten only if the
        # configured inertance/time constant requires it.
        # Semi-implicit treatment of resistance keeps the stiff R/I term stable;
        # use the explicitly configured cycle timestep and verify convergence
        # against a finer setting in the scientific gate.
        dt_s = max(0.0025, float(c.respiratory_cycle_dt_s))
        total_s = dt_min * 60.0
        n = max(1, int(np.ceil(total_s / dt_s)))
        dt_s = total_s / n

        times = np.empty(n, dtype=float)
        sample_dts = np.full(n, dt_s, dtype=float)
        vols = np.empty(n, dtype=float)
        flows = np.empty(n, dtype=float)
        paws = np.empty(n, dtype=float)
        pmuss = np.empty(n, dtype=float)
        pel = np.empty(n, dtype=float)
        limited = np.zeros(n, dtype=bool)
        resistive_power = np.empty(n, dtype=float)
        eq_residual = np.empty(n, dtype=float)
        flow_limit_pressure = np.zeros(n, dtype=float)
        vent_active_trace = np.zeros(n, dtype=bool)
        neural_active_trace = np.zeros(n, dtype=bool)
        vent_support_trace = np.zeros(n, dtype=float)
        breath_end_indices: list[int] = []

        # Pressure-support state is persisted so an outer integration boundary can
        # cut through a patient or ventilator inspiration without resetting it.
        vent_active = bool(getattr(state, "respiratory_ventilator_active", 0.0) > 0.5) if psv_mode else False
        vent_elapsed = max(0.0, float(getattr(state, "respiratory_ventilator_inspiration_elapsed_s", 0.0)))
        vent_peak_flow = max(0.0, float(getattr(state, "respiratory_ventilator_peak_flow_l_s", 0.0)))
        vent_was_auto = False
        refractory = max(0.0, float(getattr(state, "respiratory_ventilator_refractory_remaining_s", 0.0)))
        triggered_current_effort = bool(getattr(state, "respiratory_ventilator_triggered_current_effort", 0.0) > 0.5)
        triggers_current_effort = int(round(float(getattr(state, "respiratory_ventilator_triggers_current_effort", 0.0))))
        prev_neural_active = (phase_s < neural_tinsp)
        effort_start_elapsed = phase_s if prev_neural_active else 0.0
        last_neural_end_abs = None
        abs_time = 0.0
        effort_count = 1 if (psv_mode and prev_neural_active) else 0
        vent_breath_count = 0
        ineffective_count = 0
        double_count = 0
        premature_count = 0
        delayed_count = 0
        auto_count = 0
        trigger_delays = []
        cycle_delays = []
        trigger_ptp = 0.0

        ps_level = max(0.0, float(getattr(state, "respiratory_pressure_support_cmH2O", c.respiratory_ps_pressure_support_cmH2O)))
        if psv_mode:
            ps_level += max(0.0, float(ventilation_pressure_assist_cmH2O))
        trigger_pressure = max(0.0, float(getattr(state, "respiratory_trigger_pressure_cmH2O", c.respiratory_ps_trigger_pressure_cmH2O)))
        trigger_flow = max(0.01, float(getattr(state, "respiratory_trigger_flow_l_s", c.respiratory_ps_trigger_flow_l_s)))
        cycleoff_fraction = float(np.clip(getattr(state, "respiratory_cycleoff_fraction_peak_flow", c.respiratory_ps_cycleoff_fraction_peak_flow), 0.05, 0.80))
        ps_rise_s = max(0.03, float(getattr(state, "respiratory_pressure_support_rise_time_s", c.respiratory_ps_rise_time_s)))
        ps_max_ti = max(0.30, float(getattr(state, "respiratory_pressure_support_max_ti_s", c.respiratory_ps_max_inspiratory_time_s)))
        leak_flow = max(0.0, float(getattr(state, "respiratory_leak_flow_l_s", 0.0)))
        allow_retrigger = bool(getattr(state, "respiratory_allow_retrigger_same_effort", 0.0) > 0.5)

        exp_elastance_fraction = float(np.clip(c.respiratory_expiratory_elastance_fraction, 0.5, 1.0))
        nonlinear_r = max(0.0, c.respiratory_flow_nonlinearity_per_l_s)
        rise_fraction = c.respiratory_pressure_control_rise_fraction

        for k in range(n):
            neural_active = phase_s < neural_tinsp
            if neural_active:
                finsp_neural = phase_s / max(1e-6, neural_tinsp)
                pmus = pmus_amp * sin(pi * finsp_neural)
            else:
                pmus = 0.0

            if psv_mode:
                # Detect neural effort transitions. Each neural inspiration is a
                # patient effort regardless of whether the ventilator triggers.
                if neural_active and not prev_neural_active:
                    effort_count += 1
                    effort_start_elapsed = 0.0
                    triggered_current_effort = False
                    triggers_current_effort = 0
                elif neural_active:
                    effort_start_elapsed += dt_s

                if (not neural_active) and prev_neural_active:
                    last_neural_end_abs = abs_time
                    if not triggered_current_effort:
                        ineffective_count += 1

                # Trigger load includes residual intrinsic PEEP that is not
                # counterbalanced by external PEEP. This creates ineffective efforts
                # mechanistically in obstructive states.
                current_auto = max(0.0, float(state.respiratory_cycle_auto_peep_cmH2O))
                threshold_load_psv = max(0.0, current_auto - c.respiratory_external_peep_threshold_unloading_fraction * peep)
                patient_trigger_signal = pmus - threshold_load_psv
                flow_trigger_signal = flow + leak_flow

                if refractory > 0.0:
                    refractory = max(0.0, refractory - dt_s)

                trigger_patient = (
                    (not vent_active) and refractory <= 0.0 and neural_active
                    and ((not triggered_current_effort) or (allow_retrigger and triggers_current_effort < 2))
                    and (patient_trigger_signal >= trigger_pressure or flow_trigger_signal >= trigger_flow)
                )
                trigger_auto = (
                    (not vent_active) and refractory <= 0.0 and (not neural_active)
                    and leak_flow >= trigger_flow
                )

                if trigger_patient or trigger_auto:
                    vent_active = True
                    vent_elapsed = 0.0
                    vent_peak_flow = max(0.0, flow)
                    vent_was_auto = bool(trigger_auto)
                    vent_breath_count += 1
                    if trigger_auto:
                        auto_count += 1
                    else:
                        delay = max(0.0, effort_start_elapsed)
                        trigger_delays.append(delay)
                        triggered_current_effort = True
                        if triggers_current_effort == 1:
                            double_count += 1
                        triggers_current_effort += 1

                if neural_active and (not vent_active) and patient_trigger_signal > 0.0:
                    trigger_ptp += max(0.0, trigger_pressure - patient_trigger_signal) * dt_s

                support = 0.0
                if vent_active:
                    support = ps_level * min(1.0, vent_elapsed / ps_rise_s)
                paw = peep + support

                # Phase mechanics follow actual patient phase rather than ventilator
                # phase: delayed cycling can therefore continue Paw into neural
                # expiration, which is essential to represent expiratory asynchrony.
                if neural_active:
                    phase_elastance = elastance
                    r_phase = raw
                else:
                    phase_elastance = elastance * exp_elastance_fraction
                    r_phase = raw * exp_r_mult
            else:
                frac = phase_s / cycle_s
                if frac < insp_fraction:
                    finsp = frac / insp_fraction
                    pmus = pmus_amp * sin(pi * finsp)
                    paw = peep + pc_amp * self._pressure_control_shape(finsp, rise_fraction)
                    phase_elastance = elastance
                    r_phase = raw
                else:
                    pmus = 0.0
                    paw = peep
                    phase_elastance = elastance * exp_elastance_fraction
                    r_phase = raw * exp_r_mult

            r_eff = r_phase * (1.0 + nonlinear_r * abs(flow))
            drive = paw + pmus

            # Semi-implicit resistance update for the complete equation of motion.
            flow_old = flow
            volume_old = volume
            flow_new = (
                flow_old + (dt_s / inertance) * (drive - phase_elastance * volume_old)
            ) / (1.0 + (dt_s / inertance) * r_eff)

            if flow_new < -flow_limit:
                flow_new = -flow_limit
                limited[k] = True

            volume_new = volume + flow_new * dt_s
            if volume_new < 0.0:
                volume_new = 0.0
                if flow_new < 0.0:
                    flow_new = 0.0

            if psv_mode and vent_active:
                vent_elapsed += dt_s
                vent_peak_flow = max(vent_peak_flow, max(0.0, flow_new))
                flow_cycle = (
                    vent_elapsed >= 0.20
                    and vent_peak_flow > 0.05
                    and flow_new <= cycleoff_fraction * vent_peak_flow
                )
                time_cycle = vent_elapsed >= ps_max_ti
                if flow_cycle or time_cycle:
                    if neural_active:
                        remaining = max(0.0, neural_tinsp - phase_s)
                        cycle_delay = -remaining
                        if remaining > c.respiratory_ps_asynchrony_timing_threshold_s:
                            premature_count += 1
                    else:
                        delay_after_neural = max(0.0, abs_time - (last_neural_end_abs if last_neural_end_abs is not None else abs_time))
                        cycle_delay = delay_after_neural
                        if delay_after_neural > c.respiratory_ps_asynchrony_timing_threshold_s:
                            delayed_count += 1
                    cycle_delays.append(cycle_delay)
                    vent_active = False
                    vent_elapsed = 0.0
                    vent_peak_flow = 0.0
                    refractory = (
                        c.respiratory_ps_autotrigger_refractory_s
                        if vent_was_auto else c.respiratory_ps_refractory_time_s
                    )
                    vent_was_auto = False

            times[k] = k * dt_s
            vols[k] = volume_new
            flows[k] = flow_new
            paws[k] = paw
            pmuss[k] = pmus
            pel[k] = phase_elastance * volume_new
            resistive_power[k] = r_eff * flow_new * flow_new
            # Numerical residual of the semi-implicit discrete equation of motion.
            raw_residual = drive - (
                inertance * (flow_new - flow_old) / dt_s
                + r_eff * flow_new
                + phase_elastance * volume_old
            )
            if limited[k]:
                # The missing pressure is the reduced-order airway-collapse/flow-
                # limitation constraint required to keep expiration at Qmax.
                flow_limit_pressure[k] = raw_residual
                eq_residual[k] = 0.0
            else:
                eq_residual[k] = raw_residual

            vent_active_trace[k] = vent_active if psv_mode else (paw > peep + 1e-6)
            neural_active_trace[k] = neural_active if psv_mode else (pmus > 1e-6)
            vent_support_trace[k] = max(0.0, paw - peep)

            volume = volume_new
            flow = flow_new
            prev_neural_active = neural_active
            phase_s += dt_s
            abs_time += dt_s
            if phase_s >= cycle_s:
                phase_s %= cycle_s
                breath_end_indices.append(k)

        # Carry an incomplete breath across outer integration boundaries.  The old
        # implementation used min(cycle_s, total_s) and therefore interpreted a
        # 1--3 s fragment as a complete tidal excursion whenever the outer step was
        # shorter than one breath.  That made VT, ventilation and blood gases depend
        # discontinuously on integration_step_min.
        current = {
            "dt": sample_dts,
            "volume": vols,
            "flow": flows,
            "paw": paws,
            "pmus": pmuss,
            "pel": pel,
            "limited": limited,
            "resistive_power": resistive_power,
            "equation_residual": eq_residual,
            "flow_limit_pressure": flow_limit_pressure,
            "ventilator_active": vent_active_trace,
            "neural_active": neural_active_trace,
            "ventilator_support": vent_support_trace,
        }

        completed: dict[str, np.ndarray] | None = None
        start = 0
        for end in breath_end_indices:
            segment = {key: value[start:end + 1] for key, value in current.items()}
            if start == 0 and self._partial_breath:
                segment = {
                    key: np.concatenate((self._partial_breath[key], value))
                    for key, value in segment.items()
                }
            completed = segment
            self._partial_breath = {}
            start = end + 1

        tail = {key: value[start:].copy() for key, value in current.items()}
        if start == 0 and self._partial_breath:
            tail = {
                key: np.concatenate((self._partial_breath[key], value))
                for key, value in tail.items()
            }
        self._partial_breath = tail

        if completed is not None:
            v = completed["volume"]
            q = completed["flow"]
            paw = completed["paw"]
            pmus = completed["pmus"]
            pe = completed["pel"]
            lim = completed["limited"]
            rp = completed["resistive_power"]
            dts = completed["dt"]

            vt_actual = float(max(0.01, np.max(v) - np.min(v)))
            min_v = float(np.min(v))
            expiratory_elastance = elastance * exp_elastance_fraction
            static_ee_volume = peep / max(1e-9, expiratory_elastance)
            dyn_hyperinflation = max(0.0, min_v - static_ee_volume)
            self._last_end_expiratory_volume_l = float(min_v)
            auto_peep = expiratory_elastance * dyn_hyperinflation
            end_exp_alv = peep + auto_peep

            peak_insp_flow = float(max(0.0, np.max(q)))
            peak_exp_flow = float(max(0.0, -np.min(q)))
            end_exp_flow = float(q[-1])

            # Integrals over one completed breath. Positive inspiratory source work;
            # resistive dissipation includes inspiration and expiration.
            insp_q = np.maximum(q, 0.0)
            resistive_work = float(np.sum(rp * dts) * self.CMH2O_L_TO_J)
            muscle_work = float(np.sum(pmus * insp_q * dts) * self.CMH2O_L_TO_J)
            ventilator_work = float(
                np.sum(np.maximum(paw - peep, 0.0) * insp_q * dts)
                * self.CMH2O_L_TO_J
            )

            # Signed P_elastic-V loop area; absolute value is dissipated hysteretic energy.
            if len(v) > 2:
                loop = float(np.sum(0.5 * (pe[1:] + pe[:-1]) * np.diff(v)))
            else:
                loop = 0.0
            hysteresis_work = abs(loop) * self.CMH2O_L_TO_J
            total_work = muscle_work + ventilator_work

            trace_time = np.cumsum(dts) - dts[0]
            self.last_trace = {
                "time_s": trace_time,
                "volume_above_relaxed_l": v.copy(),
                "flow_l_s": q.copy(),
                "airway_pressure_cmH2O": paw.copy(),
                "muscle_pressure_drive_cmH2O": pmus.copy(),
                "elastic_pressure_cmH2O": pe.copy(),
                "neural_inspiration": completed["neural_active"].astype(float).copy(),
                "ventilator_active": completed["ventilator_active"].astype(float).copy(),
                "ventilator_support_cmH2O": completed["ventilator_support"].copy(),
            }

            state.tidal_volume_l = vt_actual
            state.respiratory_cycle_dynamic_hyperinflation_l = float(dyn_hyperinflation)
            state.respiratory_cycle_auto_peep_cmH2O = float(auto_peep)
            state.respiratory_cycle_end_expiratory_alveolar_pressure_cmH2O = float(end_exp_alv)
            state.respiratory_cycle_peak_inspiratory_flow_l_s = peak_insp_flow
            state.respiratory_cycle_peak_expiratory_flow_l_s = peak_exp_flow
            state.respiratory_cycle_end_expiratory_flow_l_s = end_exp_flow
            state.respiratory_cycle_peak_muscle_pressure_cmH2O = float(np.max(pmus))
            state.respiratory_cycle_peak_airway_pressure_cmH2O = float(np.max(paw))
            state.respiratory_cycle_resistive_work_j_breath = resistive_work
            state.respiratory_cycle_muscle_work_j_breath = muscle_work
            state.respiratory_cycle_ventilator_work_j_breath = ventilator_work
            state.respiratory_cycle_pv_hysteresis_j_breath = hysteresis_work
            state.respiratory_cycle_total_work_j_breath = total_work
            state.respiratory_cycle_expiratory_flow_limited_fraction = float(np.mean(lim))
            state.respiratory_cycle_equation_residual_cmH2O = float(
                np.max(np.abs(completed["equation_residual"]))
            )
            state.respiratory_cycle_flow_limiting_pressure_cmH2O = float(
                np.max(np.abs(completed["flow_limit_pressure"]))
            )
        else:
            # Keep the most recent completed-breath diagnostics. Continuous volume,
            # flow and phase below still advance on every call.
            vt_actual = float(state.tidal_volume_l)
            min_v = float(self._last_end_expiratory_volume_l)
            dyn_hyperinflation = float(state.respiratory_cycle_dynamic_hyperinflation_l)
            auto_peep = float(state.respiratory_cycle_auto_peep_cmH2O)
            end_exp_alv = float(state.respiratory_cycle_end_expiratory_alveolar_pressure_cmH2O)
            peak_insp_flow = float(state.respiratory_cycle_peak_inspiratory_flow_l_s)
            peak_exp_flow = float(state.respiratory_cycle_peak_expiratory_flow_l_s)
            end_exp_flow = float(state.respiratory_cycle_end_expiratory_flow_l_s)
            resistive_work = float(state.respiratory_cycle_resistive_work_j_breath)
            muscle_work = float(state.respiratory_cycle_muscle_work_j_breath)
            ventilator_work = float(state.respiratory_cycle_ventilator_work_j_breath)
            hysteresis_work = float(state.respiratory_cycle_pv_hysteresis_j_breath)
            total_work = float(state.respiratory_cycle_total_work_j_breath)

        spontaneous_va = rr * max(0.05, state.tidal_volume_l - c.dead_space_l)
        # No virtual ventilation term is used. Any assistance has already altered
        # Paw -> flow -> VT through the equation of motion above.
        state.alveolar_ventilation_l_min = max(
            0.5, spontaneous_va * state.ventilation_efficiency
        )
        state.respiratory_cycle_volume_above_relaxed_l = float(volume)
        state.respiratory_cycle_flow_l_s = float(flow)
        state.respiratory_cycle_phase_s = float(phase_s)
        state.respiratory_cycle_time_constant_s = float(raw * crs)

        if psv_mode:
            expected_efforts = max(1.0, rr * dt_min)
            denominator = max(1, vent_breath_count + ineffective_count)
            total_async = ineffective_count + double_count + premature_count + delayed_count + auto_count
            state.respiratory_ventilator_active = float(vent_active)
            state.respiratory_ventilator_inspiration_elapsed_s = float(vent_elapsed)
            state.respiratory_ventilator_peak_flow_l_s = float(vent_peak_flow)
            state.respiratory_ventilator_refractory_remaining_s = float(refractory)
            state.respiratory_ventilator_triggered_current_effort = float(triggered_current_effort)
            state.respiratory_ventilator_triggers_current_effort = float(triggers_current_effort)
            state.respiratory_ventilator_last_trigger_delay_s = float(trigger_delays[-1] if trigger_delays else 0.0)
            state.respiratory_ventilator_mean_trigger_delay_s = float(np.mean(trigger_delays) if trigger_delays else 0.0)
            state.respiratory_ventilator_mean_cycling_delay_s = float(np.mean(cycle_delays) if cycle_delays else 0.0)
            state.respiratory_ventilator_ineffective_trigger_fraction = float(np.clip(ineffective_count / expected_efforts, 0.0, 1.0))
            state.respiratory_ventilator_double_trigger_fraction = float(np.clip(double_count / expected_efforts, 0.0, 1.0))
            state.respiratory_ventilator_premature_cycling_fraction = float(premature_count / max(1, vent_breath_count))
            state.respiratory_ventilator_delayed_cycling_fraction = float(delayed_count / max(1, vent_breath_count))
            state.respiratory_ventilator_autotrigger_fraction = float(auto_count / max(1, vent_breath_count))
            state.respiratory_ventilator_asynchrony_index_pct = float(min(100.0, 100.0 * total_async / denominator))
            state.respiratory_ventilator_patient_efforts_per_min = float(rr)
            state.respiratory_ventilator_breaths_per_min = float(vent_breath_count / dt_min)
            state.respiratory_ventilator_trigger_pressure_time_product_cmH2O_s = float(trigger_ptp / max(1, effort_count))
        else:
            state.respiratory_ventilator_mean_trigger_delay_s = 0.0
            state.respiratory_ventilator_mean_cycling_delay_s = 0.0
            state.respiratory_ventilator_ineffective_trigger_fraction = 0.0
            state.respiratory_ventilator_double_trigger_fraction = 0.0
            state.respiratory_ventilator_premature_cycling_fraction = 0.0
            state.respiratory_ventilator_delayed_cycling_fraction = 0.0
            state.respiratory_ventilator_autotrigger_fraction = 0.0
            state.respiratory_ventilator_asynchrony_index_pct = 0.0
            state.respiratory_ventilator_patient_efforts_per_min = rr
            state.respiratory_ventilator_breaths_per_min = rr if positive_fraction > 0 else 0.0
            state.respiratory_ventilator_trigger_pressure_time_product_cmH2O_s = 0.0

        return RespiratoryCycleResult(
            tidal_volume_l=vt_actual,
            end_expiratory_volume_above_relaxed_l=min_v,
            dynamic_hyperinflation_l=dyn_hyperinflation,
            auto_peep_cmH2O=auto_peep,
            peak_inspiratory_flow_l_s=peak_insp_flow,
            peak_expiratory_flow_l_s=peak_exp_flow,
            end_expiratory_flow_l_s=end_exp_flow,
            peak_muscle_pressure_cmH2O=float(
                state.respiratory_cycle_peak_muscle_pressure_cmH2O
            ),
            peak_airway_pressure_cmH2O=float(
                state.respiratory_cycle_peak_airway_pressure_cmH2O
            ),
            resistive_work_j_breath=resistive_work,
            muscle_work_j_breath=muscle_work,
            ventilator_work_j_breath=ventilator_work,
            pv_hysteresis_j_breath=hysteresis_work,
            total_mechanical_work_j_breath=total_work,
            expiratory_flow_limited_fraction=float(
                state.respiratory_cycle_expiratory_flow_limited_fraction
            ),
            respiratory_time_constant_s=float(raw * crs),
        )
