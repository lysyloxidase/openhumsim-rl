from __future__ import annotations

from math import exp, pi, sin
import numpy as np

from .config import HumanConfig


class ClosedLoopCardiovascularModel:
    """Eight-compartment closed-loop 0D circulation.

    The structure follows standard lumped-parameter cardiovascular modeling:
    time-varying ventricular elastance, compliant vascular compartments,
    pressure-driven resistive flows and diode-like cardiac valves.

    It is deliberately smaller than comprehensive models such as Broomé et al.
    (2013): atria are passive compliances and no valve inertia/septal/pericardial
    mechanics are included. MAP and cardiac output emerge from conserved
    compartment volumes and pressure-flow equations.
    """

    VOLUME_ATTRS = (
        "cv_la_ml", "cv_lv_ml", "cv_sa_ml", "cv_sv_ml",
        "cv_ra_ml", "cv_rv_ml", "cv_pa_ml", "cv_pv_ml",
    )

    def __init__(self, config: HumanConfig):
        self.cfg = config
        self._cardiac_phase = 0.0
        self._partial_beat = self._new_beat_accumulator()

    def runtime_snapshot(self) -> tuple[float, dict[str, float]]:
        """Return the private dynamic state needed for transactional rollback."""
        return float(self._cardiac_phase), dict(self._partial_beat)

    def restore_runtime_snapshot(
        self, snapshot: tuple[float, dict[str, float]]
    ) -> None:
        self._cardiac_phase = float(snapshot[0])
        self._partial_beat = dict(snapshot[1])

    @staticmethod
    def _new_beat_accumulator() -> dict[str, float]:
        return {
            "duration_s": 0.0,
            "sum_psa_dt": 0.0,
            "sum_pra_dt": 0.0,
            "sum_ppa_dt": 0.0,
            "sum_plv_dt": 0.0,
            "sum_prv_dt": 0.0,
            "sum_qav_dt": 0.0,
            "max_psa": -1e9,
            "min_psa": 1e9,
            "max_lv_vol": -1e9,
            "min_lv_vol": 1e9,
        }

    def initialize_state(self, s) -> None:
        c = self.cfg
        scale = self.target_blood_volume_ml(s) / c.cv_baseline_blood_volume_ml
        # Preserve chamber/vascular distribution while matching current plasma volume.
        baseline = np.asarray(
            [c.cv_v_la0_ml, c.cv_v_lv0_ml, c.cv_v_sa0_ml, c.cv_v_sv0_ml,
             c.cv_v_ra0_ml, c.cv_v_rv0_ml, c.cv_v_pa0_ml, c.cv_v_pv0_ml],
            dtype=float,
        )
        baseline *= scale
        for attr, value in zip(self.VOLUME_ATTRS, baseline):
            setattr(s, attr, float(value))
        s.cv_total_blood_volume_ml = float(sum(baseline))
        s.cv_sim_time_s = 0.0
        self._cardiac_phase = 0.0
        self._partial_beat = self._new_beat_accumulator()
        s.cardiac_output_l_min = c.cv_baseline_cardiac_output_l_min
        s.stroke_volume_ml = c.cv_baseline_cardiac_output_l_min * 1000.0 / s.heart_rate_bpm
        s.systolic_pressure_mmHg = 120.0
        s.diastolic_pressure_mmHg = 75.0
        s.central_venous_pressure_mmHg = 5.0
        s.pulmonary_artery_pressure_mmHg = 15.0
        s.left_ventricular_pressure_mmHg = 8.0
        s.right_ventricular_pressure_mmHg = 4.0
        s.cv_ejection_fraction = 0.60
        self.step(s, exercise=0.0, dt_min=c.cv_warmup_min, warmup=True)

    def target_blood_volume_ml(self, s) -> float:
        c = self.cfg
        return max(
            2500.0,
            c.cv_baseline_blood_volume_ml
            + (float(s.plasma_volume_l) - c.plasma_volume_baseline_l) * 1000.0,
        )

    @staticmethod
    def _ventricular_activation(phase: float, systolic_fraction: float) -> float:
        if phase < 0.0 or phase >= systolic_fraction:
            return 0.0
        return sin(pi * phase / systolic_fraction) ** 2

    def _pressures(self, s, phase: float, exercise: float):
        c = self.cfg
        act = self._ventricular_activation(phase, c.cv_systolic_fraction)
        e_lv = c.cv_lv_emin + (c.cv_lv_emax * (1.0 + c.cv_contractility_exercise_gain * exercise) - c.cv_lv_emin) * act
        e_rv = c.cv_rv_emin + (c.cv_rv_emax * (1.0 + 0.55 * c.cv_contractility_exercise_gain * exercise) - c.cv_rv_emin) * act

        p_la = max(0.0, (s.cv_la_ml - c.cv_v0_la_ml) / c.cv_c_la_ml_mmHg)
        p_lv = e_lv * max(0.0, s.cv_lv_ml - c.cv_v0_lv_ml)
        p_sa = max(0.0, (s.cv_sa_ml - c.cv_v0_sa_ml) / c.cv_c_sa_ml_mmHg)
        # Exercise recruits unstressed venous volume (sympathetic venoconstriction /
        # muscle-pump surrogate), raising mean systemic filling pressure and preload.
        v0_sv_eff = c.cv_v0_sv_ml - c.cv_exercise_venous_recruitment_ml * exercise
        p_sv = max(0.0, (s.cv_sv_ml - v0_sv_eff) / c.cv_c_sv_ml_mmHg)
        p_ra = max(0.0, (s.cv_ra_ml - c.cv_v0_ra_ml) / c.cv_c_ra_ml_mmHg)
        p_rv = e_rv * max(0.0, s.cv_rv_ml - c.cv_v0_rv_ml)
        p_pa = max(0.0, (s.cv_pa_ml - c.cv_v0_pa_ml) / c.cv_c_pa_ml_mmHg)
        p_pv = max(0.0, (s.cv_pv_ml - c.cv_v0_pv_ml) / c.cv_c_pv_ml_mmHg)

        # Heart-lung interaction uses transmural-like chamber pressures. Applied
        # positive intrathoracic pressure contributes to absolute intrathoracic
        # pressures used in flow gradients against systemic extrathoracic vessels.
        # Baseline spontaneous breathing has delta=0.
        cmh2o_to_mmhg = 0.735559
        p_thorax = max(
            0.0,
            float(getattr(s, "pulmonary_intrathoracic_pressure_delta_cmH2O", 0.0))
            * cmh2o_to_mmhg,
        )
        p_la += p_thorax
        p_lv += p_thorax
        p_ra += p_thorax
        p_rv += p_thorax
        p_pa += p_thorax
        p_pv += p_thorax
        return p_la, p_lv, p_sa, p_sv, p_ra, p_rv, p_pa, p_pv

    def _sync_total_volume(self, s) -> None:
        target = self.target_blood_volume_ml(s)
        current = sum(float(getattr(s, a)) for a in self.VOLUME_ATTRS)
        delta = target - current
        # Systemic veins are the dominant volume reservoir.
        s.cv_sv_ml = max(10.0, s.cv_sv_ml + delta)
        s.cv_total_blood_volume_ml = sum(float(getattr(s, a)) for a in self.VOLUME_ATTRS)

    def step(self, s, exercise: float, dt_min: float, warmup: bool = False) -> None:
        c = self.cfg
        exercise = float(np.clip(exercise, 0.0, 1.0))
        self._sync_total_volume(s)

        # Baroreflex changes HR, while MAP itself is no longer assigned by a target equation.
        hypoglycemia_drive = max(0.0, 70.0 - float(s.glucose_mg_dl))
        hr_target = (
            c.cv_resting_hr_bpm
            + c.cv_exercise_hr_gain_bpm * exercise
            + c.cv_baroreflex_hr_gain * (c.cv_map_setpoint_mmHg - float(s.map_mmHg))
            + 0.45 * hypoglycemia_drive
        )
        hr_target = float(np.clip(hr_target, 40.0, 190.0))
        # HR is advanced within the hydraulic loop. Applying its end-of-interval
        # value to the entire preceding interval makes results depend on outer dt.
        hr_dynamic = float(s.heart_rate_bpm)

        # Systemic resistance represents arteriolar tone. Exercise vasodilates;
        # angiotensin II and baroreflex increase resistance.
        ang = max(0.1, float(s.angiotensin_ii_relative))
        baro = np.clip((c.cv_map_setpoint_mmHg - float(s.map_mmHg)) / 40.0, -0.5, 1.5)
        r_sys = c.cv_r_systemic_mmHg_s_ml * (
            1.0 - c.cv_exercise_systemic_vasodilation * exercise
        ) * (1.0 + c.cv_angII_resistance_gain * (ang - 1.0)) * (1.0 + c.cv_baroreflex_resistance_gain * baro)
        r_sys = float(np.clip(r_sys, 0.25, 3.0))

        duration_s = max(0.0, dt_min) * 60.0
        if duration_s <= 0.0:
            return
        # The low-resistance valves create the fastest pressure-equalization modes.
        # For an open edge, explicit Euler has eigenvalue
        #   -(dP_i/dV_i + dP_j/dV_j) / R.
        # Cap the requested step below its stability boundary using peak ventricular
        # elastance. This keeps coarse settings (including 40 ms UQ runs)
        # conservative without changing the public configuration API.
        e_lv_peak = c.cv_lv_emax * (
            1.0 + c.cv_contractility_exercise_gain * exercise
        )
        e_rv_peak = c.cv_rv_emax * (
            1.0 + 0.55 * c.cv_contractility_exercise_gain * exercise
        )
        pressure_relaxation_rates_s = (
            (1.0 / c.cv_c_la_ml_mmHg + e_lv_peak) / c.cv_r_mitral_mmHg_s_ml,
            (e_lv_peak + 1.0 / c.cv_c_sa_ml_mmHg) / c.cv_r_aortic_mmHg_s_ml,
            (1.0 / c.cv_c_ra_ml_mmHg + e_rv_peak) / c.cv_r_tricuspid_mmHg_s_ml,
            (e_rv_peak + 1.0 / c.cv_c_pa_ml_mmHg) / c.cv_r_pulmonic_mmHg_s_ml,
            (1.0 / c.cv_c_sa_ml_mmHg + 1.0 / c.cv_c_sv_ml_mmHg)
            / max(1e-9, r_sys),
            (1.0 / c.cv_c_sv_ml_mmHg + 1.0 / c.cv_c_ra_ml_mmHg)
            / c.cv_r_systemic_venous_mmHg_s_ml,
            (1.0 / c.cv_c_pa_ml_mmHg + 1.0 / c.cv_c_pv_ml_mmHg)
            / c.cv_r_pulmonary_mmHg_s_ml,
            (1.0 / c.cv_c_pv_ml_mmHg + 1.0 / c.cv_c_la_ml_mmHg)
            / c.cv_r_pulmonary_venous_mmHg_s_ml,
        )
        fastest_rate_s = max(1e-9, *pressure_relaxation_rates_s)
        # Euler's linear limit is 2/rate; 1.25/rate leaves margin for valve
        # switching and time-varying elastance.
        stable_ds = 1.25 / fastest_rate_s
        ds = min(max(1e-5, float(c.cv_internal_step_s)), stable_ds)
        n = max(1, int(np.ceil(duration_s / ds)))
        ds = duration_s / n

        completed_beat: dict[str, float] | None = None

        for _ in range(n):
            phase = self._cardiac_phase
            p_la, p_lv, p_sa, p_sv, p_ra, p_rv, p_pa, p_pv = self._pressures(s, phase, exercise)

            q_mv = max(0.0, (p_la - p_lv) / c.cv_r_mitral_mmHg_s_ml)
            q_av = max(0.0, (p_lv - p_sa) / c.cv_r_aortic_mmHg_s_ml)
            q_sys = max(0.0, (p_sa - p_sv) / r_sys)
            q_vr = max(0.0, (p_sv - p_ra) / c.cv_r_systemic_venous_mmHg_s_ml)
            q_tv = max(0.0, (p_ra - p_rv) / c.cv_r_tricuspid_mmHg_s_ml)
            q_pv_valve = max(0.0, (p_rv - p_pa) / c.cv_r_pulmonic_mmHg_s_ml)
            pulmonary_r_multiplier = float(np.clip(
                getattr(s, "pulmonary_hpv_resistance_multiplier", 1.0)
                * getattr(s, "pulmonary_mechanical_pvr_multiplier", 1.0),
                0.5, 6.0,
            ))
            q_pulm = max(0.0, (p_pa - p_pv) / (c.cv_r_pulmonary_mmHg_s_ml * pulmonary_r_multiplier))
            q_pv_return = max(0.0, (p_pv - p_la) / c.cv_r_pulmonary_venous_mmHg_s_ml)

            # Explicit Euler with valve-like low resistances can otherwise move more
            # volume than exists in the donor compartment at coarse RL-friendly dt.
            # Donor-limited flows keep every transfer physical and exactly conserve
            # total blood volume without post-hoc mass creation.
            q_mv = min(q_mv, 0.95 * s.cv_la_ml / ds)
            q_av = min(q_av, 0.95 * s.cv_lv_ml / ds)
            q_sys = min(q_sys, 0.95 * s.cv_sa_ml / ds)
            q_vr = min(q_vr, 0.95 * s.cv_sv_ml / ds)
            q_tv = min(q_tv, 0.95 * s.cv_ra_ml / ds)
            q_pv_valve = min(q_pv_valve, 0.95 * s.cv_rv_ml / ds)
            q_pulm = min(q_pulm, 0.95 * s.cv_pa_ml / ds)
            q_pv_return = min(q_pv_return, 0.95 * s.cv_pv_ml / ds)

            # Kirchhoff volume balances. Units: mL/s * s = mL.
            s.cv_la_ml += (q_pv_return - q_mv) * ds
            s.cv_lv_ml += (q_mv - q_av) * ds
            s.cv_sa_ml += (q_av - q_sys) * ds
            s.cv_sv_ml += (q_sys - q_vr) * ds
            s.cv_ra_ml += (q_vr - q_tv) * ds
            s.cv_rv_ml += (q_tv - q_pv_valve) * ds
            s.cv_pa_ml += (q_pv_valve - q_pulm) * ds
            s.cv_pv_ml += (q_pulm - q_pv_return) * ds

            # Guard against a numerical valve step withdrawing more than a compartment.
            # With the default dt/R values this should be essentially inactive.
            for attr in self.VOLUME_ATTRS:
                if getattr(s, attr) < 1e-6:
                    correction = 1e-6 - float(getattr(s, attr))
                    setattr(s, attr, 1e-6)
                    # Keep conservation observable rather than hiding it.
                    s.cv_numerical_volume_correction_ml += correction

            beat = self._partial_beat
            beat["duration_s"] += ds
            beat["sum_psa_dt"] += p_sa * ds
            beat["sum_pra_dt"] += p_ra * ds
            beat["sum_ppa_dt"] += p_pa * ds
            beat["sum_plv_dt"] += p_lv * ds
            beat["sum_prv_dt"] += p_rv * ds
            beat["sum_qav_dt"] += q_av * ds
            beat["max_psa"] = max(beat["max_psa"], p_sa)
            beat["min_psa"] = min(beat["min_psa"], p_sa)
            beat["max_lv_vol"] = max(beat["max_lv_vol"], s.cv_lv_ml)
            beat["min_lv_vol"] = min(beat["min_lv_vol"], s.cv_lv_ml)

            # Exact first-order HR relaxation over the hydraulic substep and a
            # midpoint rate for continuous phase accumulation.
            alpha_hr_sub = 1.0 - exp(
                -(ds / 60.0) / max(1e-6, c.cv_hr_tau_min)
            )
            hr_next = hr_dynamic + (hr_target - hr_dynamic) * alpha_hr_sub
            hr_mid = 0.5 * (hr_dynamic + hr_next)
            hr_dynamic = hr_next
            phase_next = self._cardiac_phase + ds * max(1e-6, hr_mid) / 60.0
            if phase_next >= 1.0:
                completed_beat = dict(beat)
                self._partial_beat = self._new_beat_accumulator()
                phase_next %= 1.0
            self._cardiac_phase = float(phase_next)
            s.cv_sim_time_s += ds

        s.heart_rate_bpm = float(hr_dynamic)
        if completed_beat is not None:
            beat_duration = max(1e-9, completed_beat["duration_s"])
            s.map_mmHg = completed_beat["sum_psa_dt"] / beat_duration
            s.central_venous_pressure_mmHg = completed_beat["sum_pra_dt"] / beat_duration
            s.pulmonary_artery_pressure_mmHg = completed_beat["sum_ppa_dt"] / beat_duration
            s.left_ventricular_pressure_mmHg = completed_beat["sum_plv_dt"] / beat_duration
            s.right_ventricular_pressure_mmHg = completed_beat["sum_prv_dt"] / beat_duration
            s.systolic_pressure_mmHg = completed_beat["max_psa"]
            s.diastolic_pressure_mmHg = completed_beat["min_psa"]
            s.cardiac_output_l_min = (
                completed_beat["sum_qav_dt"] / beat_duration
            ) * 60.0 / 1000.0
            # Integrated aortic-valve flow is the stroke volume of this
            # completed beat. CO/instantaneous-HR is only an approximation when
            # heart rate changes during the beat.
            s.stroke_volume_ml = completed_beat["sum_qav_dt"]
            max_lv_vol = completed_beat["max_lv_vol"]
            min_lv_vol = completed_beat["min_lv_vol"]
            if max_lv_vol > 0.0:
                s.cv_ejection_fraction = float(np.clip(
                    (max_lv_vol - min_lv_vol) / max_lv_vol, 0.0, 1.0
                ))
        s.cv_total_blood_volume_ml = sum(float(getattr(s, a)) for a in self.VOLUME_ATTRS)
        s.cv_blood_volume_error_ml = s.cv_total_blood_volume_ml - self.target_blood_volume_ml(s)
