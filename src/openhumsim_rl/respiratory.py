from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import HumanConfig
from .oxygen_binding import OxygenBindingModel


PULMONARY_RER_MIN = 0.50
PULMONARY_RER_MAX = 2.00
PULMONARY_RER_MIN_RELIABLE_FLOW_ML_MIN = 20.0


def effective_pulmonary_rer(state, config: HumanConfig) -> float:
    """Return a bounded pulmonary RER without conflating it with cellular RQ.

    The alveolar gas equation requires the gas-exchange ratio across the lung:
    pulmonary CO2 elimination divided by pulmonary O2 uptake.  In this reduced
    model achieved whole-body VO2 is also the pulmonary O2-uptake rate because
    there is no separate O2 store.  During CO2 storage or bicarbonate-buffer
    release this RER may legitimately differ from the metabolic substrate RQ.

    Very small fluxes make their ratio ill-conditioned.  In that case the
    current metabolic RQ is the least surprising finite fallback.  The broad
    bounds retain non-steady RER values above one without allowing a near-zero
    denominator to destabilize alveolar oxygen calculations.
    """
    fallback = float(
        getattr(state, "metabolic_respiratory_quotient", config.respiratory_quotient)
    )
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(config.respiratory_quotient)
    fallback = float(np.clip(fallback, PULMONARY_RER_MIN, PULMONARY_RER_MAX))

    elimination = float(getattr(state, "vco2_elimination_ml_min", np.nan))
    uptake = float(getattr(state, "vo2_ml_min", np.nan))
    reliable = (
        np.isfinite(elimination)
        and np.isfinite(uptake)
        and elimination >= PULMONARY_RER_MIN_RELIABLE_FLOW_ML_MIN
        and uptake >= PULMONARY_RER_MIN_RELIABLE_FLOW_ML_MIN
    )
    if not reliable:
        return fallback
    return float(
        np.clip(elimination / uptake, PULMONARY_RER_MIN, PULMONARY_RER_MAX)
    )


def alveolar_oxygen_tension_mmHg(
    *, inspired_o2_mmHg: float, pco2_mmHg, fio2: float, pulmonary_rer: float
):
    """Full alveolar-gas equation using pulmonary RER, scalar or vector PCO2.

    PAO2 = PIO2 - PACO2 * (FIO2 + (1 - FIO2) / RER)

    Keeping the FIO2 term matters under supplemental oxygen: at FIO2=1 the
    coefficient of PACO2 is exactly one and must not depend on substrate RQ.
    """
    inspired = float(inspired_o2_mmHg)
    fio2_fraction = float(np.clip(fio2, 0.0, 1.0))
    rer = float(np.clip(pulmonary_rer, PULMONARY_RER_MIN, PULMONARY_RER_MAX))
    coefficient = fio2_fraction + (1.0 - fio2_fraction) / rer
    return inspired - np.asarray(pco2_mmHg, dtype=float) * coefficient


@dataclass
class RespiratoryIntervention:
    fio2: float
    ventilation_support_l_min: float


class RespiratoryModel:
    """Compact gas-exchange and acid-base subsystem.

    Responsibilities:
    - metabolic O2/CO2 demand and respiratory-drive targets,
    - provisional spontaneous alveolar ventilation before the within-breath solver,
    - pH/PCO2-dependent adult Hb-O2 binding and whole-body O2 delivery/consumption.

    Mechanical assistance is never added as virtual L/min. It is resolved by
    ``DynamicRespiratoryCycleModel`` as airway pressure -> flow -> volume.
    This remains reduced-order and is not a clinical ventilator model.
    """

    # Reduced peripheral-chemoreflex parametrization. PaO2-driven ventilation is
    # intentionally negligible above 60 mmHg and rises over a 20-mmHg scale
    # below it. These constants encode direction and a bounded acute response;
    # they are not a patient-specific controller or an apnea model.
    _HYPOXIC_THRESHOLD_MMHG = 60.0
    _HYPOXIC_SCALE_MMHG = 20.0
    _HYPOXIC_MAX_DRIVE = 2.0
    _HYPOXIC_RR_GAIN_BPM = 6.0
    _HYPOXIC_VT_GAIN_L = 0.15

    def __init__(self, config: HumanConfig):
        self.cfg = config
        self.oxygen_binding = OxygenBindingModel(config)

    @staticmethod
    def _smooth_supply_limited_consumption(
        demand_ml_min: float,
        ceiling_ml_min: float,
        transition_width_ml_min: float,
    ) -> float:
        """C1-continuous minimum of demand and the extraction ceiling.

        Outside the configured transition band this equals the hard minimum.
        Inside it, a quadratic bridge removes the non-differentiable switch
        while remaining no greater than either demand or supply ceiling.
        """
        demand = max(0.0, float(demand_ml_min))
        ceiling = max(0.0, float(ceiling_ml_min))
        width = max(0.0, float(transition_width_ml_min))
        if width <= 1e-12 or ceiling <= 0.0:
            return float(min(demand, ceiling))

        delta = demand - ceiling
        if delta <= -width:
            return float(demand)
        if delta >= width:
            return float(ceiling)

        achieved = demand - (delta + width) ** 2 / (4.0 * width)
        return float(np.clip(achieved, 0.0, min(demand, ceiling)))

    def update_oxygen_transport(self, state):
        """Derive blood O2 content and enforce whole-body supply dependence.

        Achieved consumption cannot exceed the configured whole-body extraction
        ceiling times oxygen delivery. Unmet demand is exposed as oxygen debt
        and drives lactate production on the following operator-split substep.
        """
        c = self.cfg
        hb = float(getattr(state, "hemoglobin_g_dl", c.hemoglobin_g_dl))
        sao2 = self.oxygen_binding.saturation(
            state.pao2_mmHg, state.ph_arterial, state.paco2_mmHg
        ).saturation_fraction
        state.spo2_pct = float(100.0 * sao2)
        cao2 = self.oxygen_binding.content_ml_dl(
            state.pao2_mmHg, state.ph_arterial, state.paco2_mmHg, hb
        )
        co_dl_min = max(1e-9, float(state.cardiac_output_l_min) * 10.0)
        do2 = cao2 * co_dl_min

        demand = max(0.0, float(getattr(state, "vo2_demand_ml_min", state.vo2_ml_min)))
        supply_ceiling = c.oxygen_max_extraction_fraction * do2
        transition_width = (
            c.oxygen_supply_transition_width_fraction * supply_ceiling
        )
        achieved = self._smooth_supply_limited_consumption(
            demand, supply_ceiling, transition_width
        )
        debt = max(0.0, demand - achieved)
        aerobic_fraction = 1.0 if demand <= 1e-9 else float(np.clip(achieved / demand, 0.0, 1.0))

        extraction_ml_dl = achieved / co_dl_min
        cvo2 = max(0.0, cao2 - extraction_ml_dl)
        venous_ph = float(getattr(state, "mixed_venous_ph", max(6.8, state.ph_arterial - 0.04)))
        venous_pco2 = float(getattr(state, "mixed_venous_pco2_mmHg", state.paco2_mmHg + 6.0))
        pvo2 = self.oxygen_binding.po2_from_content(cvo2, venous_ph, venous_pco2, hb)
        svo2 = 100.0 * self.oxygen_binding.saturation(pvo2, venous_ph, venous_pco2).saturation_fraction

        state.vo2_ml_min = float(achieved)
        state.oxygen_debt_ml_min = float(debt)
        state.instantaneous_oxygen_deficit_ml_min = float(debt)
        state.aerobic_fraction = float(aerobic_fraction)
        state.arterial_o2_content_ml_dl = float(cao2)
        state.mixed_venous_o2_content_ml_dl = float(cvo2)
        state.mixed_venous_o2_sat_pct = float(np.clip(svo2, 0.0, 100.0))
        state.oxygen_delivery_ml_min = float(do2)
        state.oxygen_extraction_ratio = float(achieved / max(1e-9, do2))
        # Reserve relative to usable extraction capacity, not total arterial O2
        # delivery, keeping the margin consistent with a supply-limited VO2
        # deficit.
        state.oxygen_supply_margin_ml_min = float(supply_ceiling - demand)
        return state

    def update_metabolic_gas_production(self, state, *, exercise: float):
        """Couple oxidative CO2 production to achieved, not demanded, VO2.

        ``metabolic_respiratory_quotient`` is a reduced substrate-mixture
        diagnostic derived from the configured VO2/VCO2 exercise targets. It
        is not instantaneous pulmonary RER: transient bicarbonate-store release
        is represented by CO2 elimination from the conserved carbon pool.

        The integrated physiology calls this after the current oxygen-transport
        solve.  Consequently its value is the explicit production rate used by
        the *next* operator-split interval, rather than an end-of-step rate
        retroactively applied to the interval that just elapsed.
        """
        c = self.cfg
        exercise = float(np.clip(exercise, 0.0, 1.0))
        vo2_target = c.baseline_vo2_ml_min * (1.0 + c.exercise_vo2_gain * exercise)
        vco2_target = c.baseline_vco2_ml_min * (1.0 + c.exercise_vco2_gain * exercise)
        rq = float(np.clip(vco2_target / max(1e-9, vo2_target), 0.60, 1.0))
        oxidative_vco2 = rq * max(0.0, float(state.vo2_ml_min))
        state.metabolic_respiratory_quotient = rq
        state.vco2_demand_ml_min = float(rq * max(0.0, state.vo2_demand_ml_min))
        state.oxidative_vco2_ml_min = float(oxidative_vco2)
        # Compatibility name for total oxidative metabolic CO2 generation, not
        # pulmonary elimination or a separately prescribed demand. Buffer-derived
        # expired CO2 comes out of the existing pool.
        state.vco2_ml_min = float(oxidative_vco2)
        return state

    def update_mechanics(self, state, intervention: RespiratoryIntervention, exercise: float, dt: float):
        """Update metabolic gas demand, respiratory drive and alveolar ventilation.

        PaCO2 is resolved by ``WholeBloodGasChemistryModel`` so carbon storage
        and elimination remain mass-conserving instead of using a direct
        first-order relaxation to 0.863*VCO2/VA.
        """
        c = self.cfg
        exercise = float(np.clip(exercise, 0.0, 1.0))

        vo2_target = c.baseline_vo2_ml_min * (1.0 + c.exercise_vo2_gain * exercise)
        # Exact first-order update (tau=1 min) avoids applying an Euler endpoint
        # rate over the whole interval and materially improves dt convergence of
        # the coupled VO2 -> oxidative-VCO2 carbon ledger.
        vo2_alpha = 1.0 - np.exp(-max(0.0, float(dt)) / 1.0)
        state.vo2_demand_ml_min += (
            vo2_target - state.vo2_demand_ml_min
        ) * vo2_alpha

        # Central CO2/pH feedback is signed: hypocapnia and alkalemia suppress
        # drive rather than becoming indistinguishable from normal. A bounded
        # PaO2 term adds the acute peripheral-chemoreceptor response below
        # approximately 60 mmHg. This remains a reduced awake-adult reflex; it
        # does not model sleep, sedatives, neurologic injury or apneic thresholds.
        co2_drive = float(np.clip(
            c.respiratory_feedback_gain_co2 * (state.paco2_mmHg - 40.0),
            -12.0,
            60.0,
        ))
        acid_drive = float(np.clip(
            c.respiratory_feedback_gain_acid * (7.40 - state.ph_arterial),
            -12.0,
            60.0,
        ))
        hypoxic_drive = float(np.clip(
            (self._HYPOXIC_THRESHOLD_MMHG - state.pao2_mmHg)
            / self._HYPOXIC_SCALE_MMHG,
            0.0,
            self._HYPOXIC_MAX_DRIVE,
        ))
        rr_target = (
            c.baseline_rr_bpm
            + 20.0 * exercise
            + 0.28 * co2_drive
            + 0.08 * acid_drive
            + self._HYPOXIC_RR_GAIN_BPM * hypoxic_drive
        )
        vt_target = (
            0.50
            + 0.95 * exercise
            + 0.008 * co2_drive
            + 0.004 * acid_drive
            + self._HYPOXIC_VT_GAIN_L * hypoxic_drive
        )
        rr_target = float(np.clip(rr_target, 2.0, 60.0))
        vt_target = float(np.clip(vt_target, 0.10, 2.50))

        state.respiratory_rate_bpm += (
            rr_target - state.respiratory_rate_bpm
        ) * min(1.0, dt / c.respiratory_tau_min)
        drive_vt = float(getattr(state, "respiratory_drive_target_tidal_volume_l", state.tidal_volume_l))
        drive_vt += (vt_target - drive_vt) * min(1.0, dt / c.respiratory_tau_min)
        state.respiratory_drive_target_tidal_volume_l = float(max(0.05, drive_vt))

        # Until the within-breath solver runs, use the most recent actual VT for
        # the provisional ventilation diagnostic. The within-breath solver then
        # updates it from respiratory-cycle flow and volume.
        spontaneous_va = state.respiratory_rate_bpm * max(
            0.05, state.tidal_volume_l - c.dead_space_l
        )
        effective_spontaneous_va = spontaneous_va * state.ventilation_efficiency
        # No intervention may add virtual alveolar ventilation. Mechanical
        # assistance must pass through the within-breath pressure-flow solver.
        state.alveolar_ventilation_l_min = max(0.5, effective_spontaneous_va)
        return state

    def update_oxygen_from_current_pco2(
        self, state, intervention: RespiratoryIntervention, exercise: float
    ):
        """Compatibility helper for a direct oxygen update at current PaCO2."""
        c = self.cfg
        exercise = float(np.clip(exercise, 0.0, 1.0))
        fio2 = float(np.clip(intervention.fio2, 0.15, 1.0))
        inspired_o2 = fio2 * (c.atmospheric_pressure_mmHg - c.water_vapor_pressure_mmHg)
        aa_gradient = c.baseline_aa_gradient_mmHg + 3.0 * exercise
        pulmonary_rer = effective_pulmonary_rer(state, c)
        alveolar_pao2 = float(alveolar_oxygen_tension_mmHg(
            inspired_o2_mmHg=inspired_o2,
            pco2_mmHg=state.paco2_mmHg,
            fio2=fio2,
            pulmonary_rer=pulmonary_rer,
        ))
        state.pao2_mmHg = max(
            20.0, alveolar_pao2 - aa_gradient
        )
        sat = self.oxygen_binding.saturation(
            state.pao2_mmHg, state.ph_arterial, state.paco2_mmHg
        ).saturation_fraction
        state.spo2_pct = float(np.clip(100.0 * sat, 0.0, 100.0))
        self.update_oxygen_transport(state)
        return state

    def step(self, state, intervention: RespiratoryIntervention, exercise: float, dt: float):
        """Reduced respiratory step for external compatibility.

        Integrated physiology uses ``update_mechanics`` with the carbon-conserving
        whole-blood gas module. This method supports external callers that depend
        on the relaxed-PaCO2 behavior.
        """
        self.update_mechanics(state, intervention, exercise, dt)
        c = self.cfg
        paco2_target = 863.0 * (state.vco2_ml_min / 1000.0) / max(
            0.5, state.alveolar_ventilation_l_min
        )
        paco2_target = float(np.clip(paco2_target, 15.0, 120.0))
        state.paco2_mmHg += (
            paco2_target - state.paco2_mmHg
        ) * min(1.0, dt / c.gas_exchange_tau_min)
        self.update_oxygen_from_current_pco2(state, intervention, exercise)
        return self.update_metabolic_gas_production(state, exercise=exercise)
