from __future__ import annotations

from dataclasses import dataclass
from math import exp
import numpy as np

from .config import HumanConfig


@dataclass(frozen=True)
class RespiratoryMechanicsResult:
    pleural_end_exp_cmH2O: float
    pleural_end_insp_cmH2O: float
    transpulmonary_end_exp_cmH2O: float
    transpulmonary_end_insp_cmH2O: float
    mean_transpulmonary_cmH2O: float
    passive_equivalent_plateau_cmH2O: float
    airway_driving_pressure_cmH2O: float
    transpulmonary_driving_pressure_cmH2O: float
    chest_wall_driving_pressure_cmH2O: float
    lung_compliance_l_cmH2O: float
    chest_wall_compliance_l_cmH2O: float
    respiratory_system_compliance_l_cmH2O: float
    lung_strain: float
    overdistension_fraction: float
    intrathoracic_pressure_delta_cmH2O: float
    mechanical_pvr_multiplier: float
    elastic_work_j_per_breath: float
    mechanical_power_j_min: float
    pressure_identity_residual_cmH2O: float


class RespiratoryMechanicsModel:
    """Reduced lung and chest-wall mechanics.

    The model explicitly separates:
      Paw  : airway pressure
      Ppl  : pleural-pressure surrogate
      PL   : transpulmonary pressure = Paw - Ppl

    Lung and chest-wall compliances are separate series elastic elements.
    Recruitment alters effective lung compliance; high end-inspiratory
    transpulmonary pressure causes a smooth overdistension penalty.

    For spontaneous breathing, calculated plateau/driving pressure is a
    *passive-equivalent diagnostic* and is not interpreted as measured airway
    pressure. Hemodynamic transmission is enabled only by the positive-pressure
    fraction, which is 0 in the nominal spontaneous baseline.

    This is not a ventilator controller, patient-specific esophageal-manometry
    model, or finite-element lung model.
    """

    def __init__(self, config: HumanConfig):
        self.cfg = config

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))

    def step(self, state, dt_min: float | None = None) -> RespiratoryMechanicsResult:
        c = self.cfg

        recruitment = float(np.clip(state.pulmonary_recruitment_fraction, 0.02, 1.0))
        peep = max(0.0, float(state.pulmonary_peep_cmH2O))
        positive_fraction = float(np.clip(state.pulmonary_positive_pressure_fraction, 0.0, 1.0))
        vt = max(0.05, float(state.tidal_volume_l))

        lung_scale = max(0.15, float(state.pulmonary_lung_compliance_scale))
        cw_scale = max(0.15, float(state.pulmonary_chest_wall_compliance_scale))

        # Recruitment governs how much parenchyma participates in tidal inflation.
        # A floor avoids singular compliance when the lung is nearly closed.
        recruitment_compliance_factor = 0.25 + 0.75 * recruitment
        cl_base = c.pulmonary_lung_compliance_l_cmH2O * lung_scale * recruitment_compliance_factor
        ccw = c.pulmonary_chest_wall_compliance_l_cmH2O * cw_scale

        # End-expiratory pleural pressure: applied positive airway pressure is only
        # partially transmitted to the pleural space.
        ppl_ee = (
            c.pulmonary_baseline_pleural_pressure_cmH2O
            + positive_fraction * c.pulmonary_peep_pleural_transmission_fraction * peep
        )
        intrinsic_peep = max(0.0, float(getattr(state, "respiratory_cycle_auto_peep_cmH2O", 0.0)))
        pl_ee = peep + intrinsic_peep - ppl_ee

        # First-pass elastic inflation determines end-inspiratory PL and overdistension.
        dpl0 = vt / max(1e-6, cl_base)
        pl_ei0 = pl_ee + dpl0
        overdist = self._sigmoid(
            (pl_ei0 - c.pulmonary_overdistension_pl50_cmH2O)
            / max(1e-6, c.pulmonary_overdistension_slope_cmH2O)
        )

        # Overdistension reduces incremental compliance smoothly.
        cl = cl_base * max(
            c.pulmonary_min_overdistended_compliance_fraction,
            1.0 - c.pulmonary_overdistension_compliance_loss_fraction * overdist,
        )
        dpl = vt / max(1e-6, cl)
        dpcw = vt / max(1e-6, ccw)
        pl_ei = pl_ee + dpl

        # Passive respiratory-system mechanics (series lung + chest wall).
        dpaw = dpl + dpcw
        crs = vt / max(1e-6, dpaw)
        plateau = peep + dpaw

        # During positive-pressure support, chest-wall inflation raises pleural
        # pressure; spontaneous effort is deliberately not fed into circulation.
        # At end inspiration the spontaneous component appears as a more negative
        # pleural pressure, while positive-pressure inflation raises pleural pressure.
        ppl_ei = ppl_ee - (1.0 - positive_fraction) * dpl + positive_fraction * dpcw
        intrathoracic_delta = positive_fraction * (
            c.pulmonary_peep_pleural_transmission_fraction * peep
            + c.pulmonary_mean_inspiratory_pressure_fraction * dpcw
        )

        mean_pl = pl_ee + c.pulmonary_mean_inspiratory_pressure_fraction * dpl

        frc_effective = max(
            0.5,
            c.pulmonary_frc_l
            * (0.35 + 0.65 * recruitment)
            * (1.0 + c.pulmonary_peep_frc_gain_per_cmH2O * peep),
        )
        strain = vt / frc_effective

        # Pulmonary vascular resistance is U-shaped with lung inflation in this
        # reduced model: derecruitment raises extra-alveolar resistance and
        # overdistension compresses alveolar vessels.
        pvr_mech = (
            1.0
            + c.pulmonary_low_volume_pvr_gain * (1.0 - recruitment)
            + c.pulmonary_high_volume_pvr_gain * overdist
        )

        # Elastic work remains a quasi-static diagnostic.  Once the within-breath
        # solver has completed at least one breath, mechanical power uses its
        # pressure-flow source-work integral so airway resistance and patient/
        # ventilator sharing are represented.  PEEP work is added under positive
        # pressure because the dynamic ventilator integral is defined above PEEP.
        cmh2o_l_to_j = 0.0980665
        elastic_work = 0.5 * dpaw * vt * cmh2o_l_to_j
        if positive_fraction > 0.0:
            elastic_work += positive_fraction * peep * vt * cmh2o_l_to_j
        rr = max(0.0, float(state.respiratory_rate_bpm))
        has_completed_dynamic_breath = (
            float(getattr(state, "respiratory_cycle_peak_inspiratory_flow_l_s", 0.0))
            > 1e-9
        )
        if has_completed_dynamic_breath:
            dynamic_source_work = max(
                0.0,
                float(getattr(state, "respiratory_cycle_total_work_j_breath", 0.0)),
            )
            peep_work = positive_fraction * peep * vt * cmh2o_l_to_j
            mech_power = (dynamic_source_work + peep_work) * rr
        else:
            mech_power = elastic_work * rr

        # The defining physical identity is explicit and testable.
        palv_ee = peep + intrinsic_peep
        identity = pl_ee - (palv_ee - ppl_ee)

        state.pulmonary_pleural_pressure_end_exp_cmH2O = float(ppl_ee)
        state.pulmonary_pleural_pressure_end_insp_cmH2O = float(ppl_ei)
        state.pulmonary_transpulmonary_pressure_end_exp_cmH2O = float(pl_ee)
        state.pulmonary_transpulmonary_pressure_end_insp_cmH2O = float(pl_ei)
        state.pulmonary_mean_transpulmonary_pressure_cmH2O = float(mean_pl)
        state.pulmonary_passive_equivalent_plateau_pressure_cmH2O = float(plateau)
        state.pulmonary_airway_driving_pressure_cmH2O = float(dpaw)
        state.pulmonary_transpulmonary_driving_pressure_cmH2O = float(dpl)
        state.pulmonary_chest_wall_driving_pressure_cmH2O = float(dpcw)
        state.pulmonary_lung_compliance_l_cmH2O = float(cl)
        state.pulmonary_chest_wall_compliance_l_cmH2O = float(ccw)
        state.pulmonary_respiratory_system_compliance_l_cmH2O = float(crs)
        state.pulmonary_lung_strain = float(strain)
        state.pulmonary_overdistension_fraction = float(overdist)
        state.pulmonary_intrathoracic_pressure_delta_cmH2O = float(intrathoracic_delta)
        state.pulmonary_mechanical_pvr_multiplier = float(pvr_mech)
        state.pulmonary_elastic_work_j_per_breath = float(elastic_work)
        state.pulmonary_mechanical_power_j_min = float(mech_power)
        state.pulmonary_pressure_identity_residual_cmH2O = float(identity)

        return RespiratoryMechanicsResult(
            pleural_end_exp_cmH2O=float(ppl_ee),
            pleural_end_insp_cmH2O=float(ppl_ei),
            transpulmonary_end_exp_cmH2O=float(pl_ee),
            transpulmonary_end_insp_cmH2O=float(pl_ei),
            mean_transpulmonary_cmH2O=float(mean_pl),
            passive_equivalent_plateau_cmH2O=float(plateau),
            airway_driving_pressure_cmH2O=float(dpaw),
            transpulmonary_driving_pressure_cmH2O=float(dpl),
            chest_wall_driving_pressure_cmH2O=float(dpcw),
            lung_compliance_l_cmH2O=float(cl),
            chest_wall_compliance_l_cmH2O=float(ccw),
            respiratory_system_compliance_l_cmH2O=float(crs),
            lung_strain=float(strain),
            overdistension_fraction=float(overdist),
            intrathoracic_pressure_delta_cmH2O=float(intrathoracic_delta),
            mechanical_pvr_multiplier=float(pvr_mech),
            elastic_work_j_per_breath=float(elastic_work),
            mechanical_power_j_min=float(mech_power),
            pressure_identity_residual_cmH2O=float(identity),
        )
