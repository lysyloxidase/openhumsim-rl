from __future__ import annotations

from dataclasses import dataclass
from math import tanh
import numpy as np

from .config import HumanConfig


@dataclass(frozen=True)
class DallaManNormalParameters:
    """Normal-subject parameter set from Dalla Man, Rizza & Cobelli (2007).

    Units follow the original paper. The model is normalized per kg body weight.
    """

    VG_dl_kg: float = 1.88
    k1_min: float = 0.065
    k2_min: float = 0.079

    VI_l_kg: float = 0.05
    m1_min: float = 0.190
    m2_min: float = 0.484
    m4_min: float = 0.194
    m5_min_kg_pmol: float = 0.0304
    m6: float = 0.6471
    HEb: float = 0.60

    kmax_min: float = 0.0558
    kmin_min: float = 0.0080
    kabs_min: float = 0.057
    kgri_min: float = 0.0558
    f: float = 0.90
    a_mg_inv: float = 0.00013
    b: float = 0.82
    c_mg_inv: float = 0.00236
    d: float = 0.010

    kp1_mg_kg_min: float = 2.70
    kp2_min: float = 0.0021
    kp3_mg_kg_min_per_pmol_l: float = 0.009
    kp4_mg_kg_min_per_pmol_kg: float = 0.0618
    ki_min: float = 0.0079

    Fcns_mg_kg_min: float = 1.0
    Vm0_mg_kg_min: float = 2.50
    Vmx_mg_kg_min_per_pmol_l: float = 0.047
    Km0_mg_kg: float = 225.59
    p2U_min: float = 0.0331

    K_pmol_kg_per_mg_dl: float = 2.30
    alpha_min: float = 0.050
    beta_pmol_kg_min_per_mg_dl: float = 0.11
    gamma_min: float = 0.5

    ke1_min: float = 0.0005
    ke2_mg_kg: float = 339.0


class DallaManMealModel:
    """Published 2007 normal-subject meal glucose-insulin model.

    Implements the model structure/equations reported in IEEE TBME 54(10),
    1740-1749 (2007): two-compartment glucose and insulin kinetics, nonlinear
    gastric emptying, endogenous glucose production, insulin-dependent and
    independent utilization, beta-cell secretion and renal glucose excretion.

    OpenHumSim adds only three interfaces around the published core:
    1) body weight from HumanConfig;
    2) renal-excretion scaling by the current simulated GFR fraction;
    3) an explicitly marked, reduced-order exercise sensitivity multiplier.

    The first two are organ couplings; the third is *not* part of the 2007
    validation and is exposed in diagnostics as an extension.
    """

    INSULIN_PMOL_L_PER_UU_ML = 6.0  # explicit model conversion convention

    def __init__(self, config: HumanConfig):
        self.cfg = config
        self.p = DallaManNormalParameters()

    def basal_values(self) -> dict[str, float]:
        p = self.p
        gb = self.cfg.dalla_basal_glucose_mg_dl

        # Table-I m5/m6 together with HEb define basal secretion exactly.
        sb = (p.m6 - p.HEb) / p.m5_min_kg_pmol
        m3b = p.HEb * p.m1_min / (1.0 - p.HEb)

        # From the basal insulin subsystem steady-state equations.
        # Il_b = ((m2 + m4)/m1) Ip_b and
        # Sb = (m1+m3)Il_b - m2 Ip_b.
        coeff = (
            (p.m1_min + m3b) * (p.m2_min + p.m4_min) / p.m1_min
            - p.m2_min
        )
        ipb = sb / coeff
        ilb = (p.m2_min + p.m4_min) / p.m1_min * ipb
        ib = ipb / p.VI_l_kg
        ipob = sb / p.gamma_min

        gpb = gb * p.VG_dl_kg
        egpb = max(
            0.0,
            p.kp1_mg_kg_min
            - p.kp2_min * gpb
            - p.kp3_mg_kg_min_per_pmol_l * ib
            - p.kp4_mg_kg_min_per_pmol_kg * ipob,
        )
        gtb = (
            p.Fcns_mg_kg_min - egpb + p.k1_min * gpb
        ) / p.k2_min

        return {
            "Gb": gb,
            "Gpb": gpb,
            "Gtb": gtb,
            "Ib": ib,
            "Ipb": ipb,
            "Ilb": ilb,
            "Sb": sb,
            "Ipob": ipob,
            "EGPb": egpb,
        }

    def initialize_state(self, state) -> None:
        b = self.basal_values()
        state.dalla_gp_mg_kg = b["Gpb"]
        state.dalla_gt_mg_kg = b["Gtb"]
        state.dalla_il_pmol_kg = b["Ilb"]
        state.dalla_ip_pmol_kg = b["Ipb"]
        state.dalla_i1_pmol_l = b["Ib"]
        state.dalla_id_pmol_l = b["Ib"]
        state.dalla_ipo_pmol_kg = b["Ipob"]
        state.dalla_x_pmol_l = 0.0
        state.dalla_y_pmol_kg_min = 0.0
        state.dalla_qsto1_mg = 0.0
        state.dalla_qsto2_mg = 0.0
        state.dalla_qgut_mg = 0.0
        state.dalla_meal_reference_mg = 0.0
        state.dalla_glucose_ingested_mg = 0.0
        state.dalla_glucose_gi_absorbed_mg = 0.0
        state.dalla_glucose_appeared_mg = 0.0
        state.sc_insulin_depot1_model_units = 0.0
        state.sc_insulin_depot2_model_units = 0.0
        state.sc_insulin_absorption_model_units_min = 0.0
        state.sc_insulin_administered_model_units = 0.0
        state.sc_insulin_absorbed_model_units = 0.0
        state.sc_insulin_mass_balance_error_model_units = 0.0
        state.glucagon_counterregulatory_egp_mg_kg_min = 0.0
        state.glucagon_counterregulatory_glucose_released_mg = 0.0
        state.dalla_ra_mg_kg_min = 0.0
        state.dalla_egp_mg_kg_min = b["EGPb"]
        state.dalla_u_mg_kg_min = b["EGPb"]
        state.dalla_insulin_secretion_pmol_kg_min = b["Sb"]
        state.dalla_hepatic_extraction = self.p.HEb
        state.glucose_mg_dl = b["Gb"]
        state.insulin_uU_ml = b["Ib"] / self.INSULIN_PMOL_L_PER_UU_ML
        state.gut_carbs_g = 0.0

    def add_meal(self, state, grams: float) -> None:
        dose_mg = max(0.0, float(grams)) * 1000.0
        if dose_mg <= 0.0:
            return
        remaining = state.dalla_qsto1_mg + state.dalla_qsto2_mg + state.dalla_qgut_mg
        if remaining < 1.0:
            state.dalla_meal_reference_mg = dose_mg
        else:
            # Approximation for overlapping boluses: treat the currently unprocessed
            # meal plus the new bolus as one gastric-emptying reference meal.
            state.dalla_meal_reference_mg = max(1.0, remaining + dose_mg)
        state.dalla_qsto1_mg += dose_mg
        state.dalla_glucose_ingested_mg += dose_mg
        self._refresh_outputs(state)

    def add_exogenous_insulin(self, state, model_units: float) -> None:
        """Deposit exogenous insulin into a two-stage SC absorption model.

        `model_units` are OpenHumSim control units, not clinical insulin units.
        The depot representation avoids nonphysiologic direct plasma injection.
        """
        units = max(0.0, float(model_units))
        if units <= 0.0:
            return
        state.sc_insulin_depot1_model_units += units
        state.sc_insulin_administered_model_units += units
        self._refresh_sc_insulin_ledger(state)

    def _refresh_sc_insulin_ledger(self, state) -> None:
        remaining = (
            max(0.0, float(state.sc_insulin_depot1_model_units))
            + max(0.0, float(state.sc_insulin_depot2_model_units))
        )
        state.sc_insulin_mass_balance_error_model_units = float(
            state.sc_insulin_administered_model_units
            - state.sc_insulin_absorbed_model_units
            - remaining
        )

    def _advance_sc_insulin(self, state, dt_min: float) -> None:
        """Advance two serial first-order SC depots analytically.

        Equal transit constants give an isolated-bolus absorption-rate peak at
        approximately `sc_insulin_tmax_min`.
        """
        dt = max(0.0, float(dt_min))
        tau = max(1e-6, float(self.cfg.sc_insulin_tmax_min))
        if dt <= 0.0:
            return

        s1 = max(0.0, float(state.sc_insulin_depot1_model_units))
        s2 = max(0.0, float(state.sc_insulin_depot2_model_units))
        k = 1.0 / tau
        e = float(np.exp(-k * dt))
        s1_new = s1 * e
        s2_new = e * (s2 + k * s1 * dt)

        absorbed = max(0.0, (s1 + s2) - (s1_new + s2_new))
        state.sc_insulin_depot1_model_units = float(s1_new)
        state.sc_insulin_depot2_model_units = float(s2_new)
        state.sc_insulin_absorbed_model_units += float(absorbed)
        state.sc_insulin_absorption_model_units_min = float(absorbed / dt)

        if absorbed > 0.0:
            delta_uU_ml = self.cfg.dalla_uU_ml_per_insulin_model_unit * absorbed
            delta_pmol_l = delta_uU_ml * self.INSULIN_PMOL_L_PER_UU_ML
            state.dalla_ip_pmol_kg += delta_pmol_l * self.p.VI_l_kg

        self._refresh_sc_insulin_ledger(state)

    def step(self, state, exercise: float, dt_min: float) -> None:
        remaining = float(dt_min)
        ds_max = self.cfg.dalla_internal_step_min
        while remaining > 1e-12:
            ds = min(ds_max, remaining)
            self._advance_sc_insulin(state, ds)
            self._rk4_substep(state, exercise=float(np.clip(exercise, 0.0, 1.0)), ds=ds)
            remaining -= ds
        self._refresh_outputs(state)

    def _state_vector(self, s) -> np.ndarray:
        return np.asarray(
            [
                s.dalla_gp_mg_kg,
                s.dalla_gt_mg_kg,
                s.dalla_il_pmol_kg,
                s.dalla_ip_pmol_kg,
                s.dalla_i1_pmol_l,
                s.dalla_id_pmol_l,
                s.dalla_ipo_pmol_kg,
                s.dalla_x_pmol_l,
                s.dalla_y_pmol_kg_min,
                s.dalla_qsto1_mg,
                s.dalla_qsto2_mg,
                s.dalla_qgut_mg,
            ],
            dtype=float,
        )

    def _set_state_vector(self, s, y: np.ndarray) -> None:
        (
            s.dalla_gp_mg_kg,
            s.dalla_gt_mg_kg,
            s.dalla_il_pmol_kg,
            s.dalla_ip_pmol_kg,
            s.dalla_i1_pmol_l,
            s.dalla_id_pmol_l,
            s.dalla_ipo_pmol_kg,
            s.dalla_x_pmol_l,
            s.dalla_y_pmol_kg_min,
            s.dalla_qsto1_mg,
            s.dalla_qsto2_mg,
            s.dalla_qgut_mg,
        ) = [float(v) for v in y]

    def _derivatives(self, s, y: np.ndarray, exercise: float):
        p = self.p
        (
            gp,
            gt,
            il,
            ip,
            i1,
            idel,
            ipo,
            x,
            ybeta,
            qsto1,
            qsto2,
            qgut,
        ) = y

        # Positive guards are only used in algebraic fluxes. Integration state is
        # allowed to expose numerical problems rather than silently clipping mass.
        gp_pos = max(0.0, gp)
        gt_pos = max(0.0, gt)
        ip_pos = max(0.0, ip)
        il_pos = max(0.0, il)
        ipo_pos = max(0.0, ipo)
        qsto1_pos = max(0.0, qsto1)
        qsto2_pos = max(0.0, qsto2)
        qgut_pos = max(0.0, qgut)

        G = gp_pos / p.VG_dl_kg
        I = ip_pos / p.VI_l_kg

        # GI subsystem, eq. (13). Table-I a,b,c,d parameterization.
        D = max(1.0, float(s.dalla_meal_reference_mg))
        qsto = qsto1_pos + qsto2_pos
        kempt = p.kmin_min + 0.5 * (p.kmax_min - p.kmin_min) * (
            tanh(p.a_mg_inv * (qsto - p.b * D))
            - tanh(p.c_mg_inv * (qsto - p.d * D))
            + 2.0
        )
        dqsto1 = -p.kgri_min * qsto1_pos
        dqsto2 = -kempt * qsto2_pos + p.kgri_min * qsto1_pos
        kabs = p.kabs_min * self.cfg.dalla_gastric_absorption_scale
        dqgut = -kabs * qgut_pos + kempt * qsto2_pos
        ra = p.f * kabs * qgut_pos / self.cfg.body_weight_kg

        # Delayed insulin signal, eq. (11).
        di1 = -p.ki_min * (i1 - I)
        did = -p.ki_min * (idel - i1)

        # EGP, eq. (10), nonnegative.
        egp_core = max(
            0.0,
            p.kp1_mg_kg_min
            - p.kp2_min * gp_pos
            - p.kp3_mg_kg_min_per_pmol_l * idel
            - p.kp4_mg_kg_min_per_pmol_kg * ipo_pos,
        )

        # Reduced counterregulatory extension.
        # The published Dalla Man normal-subject core has no explicit glucagon
        # state. We therefore add a separately reported EGP term only during
        # hypoglycemia, gated by available liver glycogen.
        gate = np.clip(
            (self.cfg.glucagon_counterreg_glucose_gate_mg_dl - G) / 20.0,
            0.0, 1.0,
        )
        glucagon_excess = max(
            0.0,
            (float(s.glucagon_pg_ml) - self.cfg.glucagon_baseline_pg_ml)
            / max(1e-9, self.cfg.glucagon_baseline_pg_ml),
        )
        hormone_effect = glucagon_excess / (0.50 + glucagon_excess) if glucagon_excess > 0 else 0.0
        glycogen_fraction = float(np.clip(
            float(s.liver_glycogen_g) / max(1e-9, self.cfg.liver_glycogen_baseline_g),
            0.0, 1.0,
        ))
        counterreg_egp = (
            self.cfg.glucagon_counterreg_egp_max_mg_kg_min
            * gate**2 * hormone_effect * glycogen_fraction
        )
        egp = egp_core + counterreg_egp

        # Utilization, eqs. (14)-(19). The exercise multiplier is an
        # OpenHumSim extension and is deliberately separated from the paper core.
        vm = p.Vm0_mg_kg_min + (
            p.Vmx_mg_kg_min_per_pmol_l
            * self.cfg.dalla_insulin_sensitivity_scale
            * max(0.0, x)
        )
        vm *= 1.0 + self.cfg.dalla_exercise_vmax_gain * exercise
        uid = vm * gt_pos / (p.Km0_mg_kg + gt_pos)
        uii = p.Fcns_mg_kg_min

        # Renal glucose excretion, eq. (27), coupled to current model GFR.
        gfr_fraction = max(0.0, float(s.gfr_ml_min)) / max(1e-9, self.cfg.baseline_gfr_ml_min)
        renal_e = (
            p.ke1_min * max(0.0, gp_pos - p.ke2_mg_kg) * gfr_fraction
            if gp_pos > p.ke2_mg_kg
            else 0.0
        )

        dgp = egp + ra - uii - renal_e - p.k1_min * gp_pos + p.k2_min * gt_pos
        dgt = -uid + p.k1_min * gp_pos - p.k2_min * gt_pos
        dG = dgp / p.VG_dl_kg

        # Beta-cell secretion, eqs. (23)-(26).
        b = self.basal_values()
        sb = b["Sb"]
        spo = ybeta + (p.K_pmol_kg_per_mg_dl * dG if dG > 0.0 else 0.0) + sb
        spo = max(0.0, spo)
        dipo = -p.gamma_min * ipo_pos + spo
        secretion = p.gamma_min * ipo_pos

        beta_drive = p.beta_pmol_kg_min_per_mg_dl * (G - b["Gb"])
        if beta_drive >= -sb:
            dybeta = -p.alpha_min * (ybeta - beta_drive)
        else:
            dybeta = -p.alpha_min * ybeta - p.alpha_min * sb

        # Time-varying hepatic extraction and insulin kinetics, eqs. (3)-(9).
        HE = float(np.clip(-p.m5_min_kg_pmol * secretion + p.m6, 0.02, 0.95))
        m3 = HE * p.m1_min / (1.0 - HE)
        dil = -(p.m1_min + m3) * il_pos + p.m2_min * ip_pos + secretion
        dip = -(p.m2_min + p.m4_min) * ip_pos + p.m1_min * il_pos

        # Remote insulin, eq. (18).
        dx = -p.p2U_min * x + p.p2U_min * (I - b["Ib"])

        deriv = np.asarray(
            [dgp, dgt, dil, dip, di1, did, dipo, dx, dybeta, dqsto1, dqsto2, dqgut],
            dtype=float,
        )
        diagnostics = {
            "G": G,
            "I": I,
            "Ra": ra,
            "EGP": egp,
            "EGP_core": egp_core,
            "EGP_counterreg": counterreg_egp,
            "U": uii + uid,
            "S": secretion,
            "HE": HE,
            "gut_absorption_mg_min": kabs * qgut_pos,
        }
        return deriv, diagnostics

    def _rk4_substep(self, s, exercise: float, ds: float) -> None:
        y0 = self._state_vector(s)
        gi_before = float(y0[9] + y0[10] + y0[11])
        k1, d1 = self._derivatives(s, y0, exercise)
        k2, d2 = self._derivatives(s, y0 + 0.5 * ds * k1, exercise)
        k3, d3 = self._derivatives(s, y0 + 0.5 * ds * k2, exercise)
        k4, d4 = self._derivatives(s, y0 + ds * k3, exercise)
        y1 = y0 + ds * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        # Positivity correction is restricted to material amounts. Track the tiny
        # correction as a diagnostic rather than hiding it.
        negative_mass_correction = 0.0
        for idx in (0, 1, 2, 3, 6, 9, 10, 11):
            if y1[idx] < 0.0:
                negative_mass_correction += -float(y1[idx])
                y1[idx] = 0.0
        s.dalla_numerical_positivity_correction += negative_mass_correction
        self._set_state_vector(s, y1)

        # The GI subsystem has only one external sink (intestinal absorption), so
        # accepted-state GI mass disappearance is the conservative absorption ledger.
        gi_after = float(y1[9] + y1[10] + y1[11])
        absorbed = max(0.0, gi_before - gi_after)
        s.dalla_glucose_gi_absorbed_mg += absorbed
        s.dalla_glucose_appeared_mg += self.p.f * absorbed

        # Refresh diagnostics from the accepted state rather than the first RK stage.
        _, d = self._derivatives(s, y1, exercise)
        s.dalla_ra_mg_kg_min = d["Ra"]
        s.dalla_egp_mg_kg_min = d["EGP"]
        s.glucagon_counterregulatory_egp_mg_kg_min = d["EGP_counterreg"]

        # Use the same RK4 quadrature for the counterregulatory glucose ledger
        # that was used to integrate plasma glucose.
        counterreg_integral_mg_kg = ds * (
            d1["EGP_counterreg"]
            + 2.0 * d2["EGP_counterreg"]
            + 2.0 * d3["EGP_counterreg"]
            + d4["EGP_counterreg"]
        ) / 6.0
        extra_glucose_mg = max(
            0.0, counterreg_integral_mg_kg * self.cfg.body_weight_kg
        )
        glycogen_used_g = min(
            max(0.0, float(s.liver_glycogen_g)),
            extra_glucose_mg / 1000.0,
        )
        s.liver_glycogen_g -= glycogen_used_g
        s.glucagon_counterregulatory_glucose_released_mg += glycogen_used_g * 1000.0
        s.dalla_u_mg_kg_min = d["U"]
        s.dalla_insulin_secretion_pmol_kg_min = d["S"]
        s.dalla_hepatic_extraction = d["HE"]

    def _refresh_outputs(self, s) -> None:
        p = self.p
        s.glucose_mg_dl = max(0.0, s.dalla_gp_mg_kg / p.VG_dl_kg)
        insulin_pmol_l = max(0.0, s.dalla_ip_pmol_kg / p.VI_l_kg)
        s.insulin_uU_ml = insulin_pmol_l / self.INSULIN_PMOL_L_PER_UU_ML
        gi_remaining = max(0.0, s.dalla_qsto1_mg + s.dalla_qsto2_mg + s.dalla_qgut_mg)
        s.gut_carbs_g = gi_remaining / 1000.0
        if gi_remaining < 1.0:
            s.dalla_meal_reference_mg = 0.0

        s.dalla_gi_mass_balance_error_mg = (
            s.dalla_glucose_ingested_mg
            - s.dalla_glucose_gi_absorbed_mg
            - gi_remaining
        )
