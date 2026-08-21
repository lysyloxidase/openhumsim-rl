from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np

from .cgm import CGMObservationConfig, CGMObservationModel, CGMObservationState


@dataclass(frozen=True)
class MeasurementChannelSpec:
    sample_interval_min: float
    result_delay_min: float = 0.0
    noise_sd: float = 0.0
    relative_noise_sd: float = 0.0
    dropout_probability: float = 0.0
    lower: float | None = None
    upper: float | None = None
    group: str = "monitor"


@dataclass(frozen=True)
class ClinicalMeasurementConfig:
    """Engineering bedside measurement model for the clinical RL profile.

    The defaults represent a plausible research/ICU-like observation process,
    not a specification for any one commercial device or hospital workflow.
    Continuous monitors are exposed at each 5-min agent step, while ABG and
    chemistry channels are sampled intermittently and become available only
    after a result delay. Dropout holds the last available value, so the policy
    sees a partially observed process rather than ground-truth state.
    """

    monitor_dropout_probability: float = 0.01
    cgm_dropout_probability: float = 0.02
    abg_interval_min: float = 30.0
    abg_result_delay_min: float = 7.0
    chemistry_interval_min: float = 60.0
    chemistry_result_delay_min: float = 12.0
    hemodynamic_interval_min: float = 15.0
    hemodynamic_result_delay_min: float = 2.0
    cgm_relative_noise_sd: float = 0.05
    cgm_lag_tau_min: float = 6.0
    noise_multiplier: float = 1.0


@dataclass
class _PendingResult:
    value: float
    sample_time_min: float
    available_time_min: float


@dataclass
class _ChannelState:
    value: float
    sample_time_min: float = 0.0
    next_sample_time_min: float = 0.0
    pending_results: deque[_PendingResult] = field(default_factory=deque)
    dropped_count: int = 0
    skipped_count: int = 0
    delivered_count: int = 1


# Clinical state channels. Glucose is handled by the separate CGM observation
# model below, because it has a physiological interstitial lag.
_BASE_CHANNELS: dict[str, MeasurementChannelSpec] = {
    # bedside/monitor channels
    "heart_rate_bpm": MeasurementChannelSpec(5.0, noise_sd=1.5, group="monitor"),
    "map_mmHg": MeasurementChannelSpec(5.0, noise_sd=2.0, group="monitor"),
    "systolic_pressure_mmHg": MeasurementChannelSpec(5.0, noise_sd=3.0, group="monitor"),
    "diastolic_pressure_mmHg": MeasurementChannelSpec(5.0, noise_sd=2.0, group="monitor"),
    "respiratory_rate_bpm": MeasurementChannelSpec(5.0, noise_sd=0.7, group="monitor"),
    "tidal_volume_l": MeasurementChannelSpec(5.0, noise_sd=0.02, lower=0.0, group="monitor"),
    "spo2_pct": MeasurementChannelSpec(5.0, noise_sd=1.8, lower=0.0, upper=100.0, group="monitor"),
    "alveolar_ventilation_l_min": MeasurementChannelSpec(5.0, relative_noise_sd=0.03, lower=0.0, group="monitor"),
    "urine_flow_ml_min": MeasurementChannelSpec(5.0, relative_noise_sd=0.08, lower=0.0, group="monitor"),

    # invasive/derived hemodynamics
    "cardiac_output_l_min": MeasurementChannelSpec(15.0, 2.0, noise_sd=0.25, lower=0.0, group="hemodynamic"),
    "stroke_volume_ml": MeasurementChannelSpec(15.0, 2.0, noise_sd=4.0, lower=0.0, group="hemodynamic"),
    "central_venous_pressure_mmHg": MeasurementChannelSpec(15.0, 2.0, noise_sd=1.0, group="hemodynamic"),
    "pulmonary_artery_pressure_mmHg": MeasurementChannelSpec(15.0, 2.0, noise_sd=1.5, group="hemodynamic"),
    "plasma_volume_l": MeasurementChannelSpec(60.0, 12.0, relative_noise_sd=0.03, lower=0.0, group="chemistry"),
    "hematocrit_fraction": MeasurementChannelSpec(60.0, 12.0, noise_sd=0.01, lower=0.0, upper=0.8, group="chemistry"),
    "hemoglobin_g_dl": MeasurementChannelSpec(60.0, 12.0, noise_sd=0.25, lower=0.0, group="chemistry"),

    # ABG / point-of-care blood gas family
    "lactate_mmol_l": MeasurementChannelSpec(30.0, 7.0, noise_sd=0.12, lower=0.0, group="abg"),
    "pao2_mmHg": MeasurementChannelSpec(30.0, 7.0, noise_sd=2.0, lower=0.0, group="abg"),
    "paco2_mmHg": MeasurementChannelSpec(30.0, 7.0, noise_sd=1.0, lower=0.0, group="abg"),
    "bicarbonate_mmol_l": MeasurementChannelSpec(30.0, 7.0, noise_sd=0.5, lower=0.0, group="abg"),
    "ph_arterial": MeasurementChannelSpec(30.0, 7.0, noise_sd=0.015, lower=6.5, upper=8.0, group="abg"),
    "arterial_o2_content_ml_dl": MeasurementChannelSpec(30.0, 7.0, noise_sd=0.3, lower=0.0, group="abg"),
    "mixed_venous_o2_sat_pct": MeasurementChannelSpec(30.0, 7.0, noise_sd=2.0, lower=0.0, upper=100.0, group="abg"),
    # chemistry / renal laboratory family
    "sodium_mmol_l": MeasurementChannelSpec(60.0, 12.0, noise_sd=0.8, group="chemistry"),
    "potassium_mmol_l": MeasurementChannelSpec(60.0, 12.0, noise_sd=0.08, group="chemistry"),
    "chloride_mmol_l": MeasurementChannelSpec(60.0, 12.0, noise_sd=0.8, group="chemistry"),
    "albumin_g_dl": MeasurementChannelSpec(60.0, 12.0, noise_sd=0.08, lower=0.0, group="chemistry"),
    "phosphate_mmol_l": MeasurementChannelSpec(60.0, 12.0, noise_sd=0.06, lower=0.0, group="chemistry"),
    "anion_gap_mEq_l": MeasurementChannelSpec(60.0, 12.0, noise_sd=0.8, group="chemistry"),
    "gfr_ml_min": MeasurementChannelSpec(60.0, 12.0, noise_sd=7.0, lower=0.0, group="chemistry"),
    "plasma_osmolality_mOsm_kg": MeasurementChannelSpec(60.0, 12.0, noise_sd=2.0, lower=0.0, group="chemistry"),
}

# Ventilator telemetry is assumed available at the decision step with small
# engineering measurement noise. These are not independently sampled labs.
_TELEMETRY_NAMES = (
    "pulmonary_passive_equivalent_plateau_pressure_cmH2O",
    "pulmonary_airway_driving_pressure_cmH2O",
    "pulmonary_respiratory_system_compliance_l_cmH2O",
    "respiratory_airway_resistance_cmH2O_s_l", "respiratory_cycle_auto_peep_cmH2O",
    "respiratory_cycle_peak_inspiratory_flow_l_s", "respiratory_cycle_peak_expiratory_flow_l_s",
    "respiratory_cycle_peak_airway_pressure_cmH2O", "respiratory_cycle_total_work_j_breath",
    "respiratory_ventilator_mean_trigger_delay_s", "respiratory_ventilator_mean_cycling_delay_s",
    "respiratory_ventilator_ineffective_trigger_fraction", "respiratory_ventilator_double_trigger_fraction",
    "respiratory_ventilator_autotrigger_fraction", "respiratory_ventilator_asynchrony_index_pct",
)

for _name in _TELEMETRY_NAMES:
    _BASE_CHANNELS.setdefault(
        _name,
        MeasurementChannelSpec(5.0, noise_sd=0.0, relative_noise_sd=0.01, group="monitor"),
    )

# Physiologic quantities sometimes clinically estimated rather than continuously
# measured. They use hemodynamic cadence to prevent ground-truth leakage.
for _name in ("oxygen_delivery_ml_min",):
    _BASE_CHANNELS.setdefault(
        _name,
        MeasurementChannelSpec(15.0, 2.0, relative_noise_sd=0.04, lower=0.0, group="hemodynamic"),
    )


class ClinicalMeasurementModel:
    """Sampling/noise/delay/missingness layer for clinical observations."""

    AGE_NAMES = (
        "cgm_measurement_age_min",
        "monitor_measurement_age_min",
        "blood_gas_measurement_age_min",
        "chemistry_measurement_age_min",
        "hemodynamic_measurement_age_min",
    )

    def __init__(self, config: ClinicalMeasurementConfig | None = None):
        self.config = config or ClinicalMeasurementConfig()
        positive = (
            "abg_interval_min",
            "chemistry_interval_min",
            "hemodynamic_interval_min",
            "cgm_lag_tau_min",
        )
        nonnegative = (
            "abg_result_delay_min",
            "chemistry_result_delay_min",
            "hemodynamic_result_delay_min",
            "cgm_relative_noise_sd",
            "noise_multiplier",
        )
        for name in positive:
            value = float(getattr(self.config, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in nonnegative:
            value = float(getattr(self.config, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("monitor_dropout_probability", "cgm_dropout_probability"):
            value = float(getattr(self.config, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        self.channels: dict[str, _ChannelState] = {}
        self.cgm_model = CGMObservationModel(
            CGMObservationConfig(
                lag_tau_min=self.config.cgm_lag_tau_min,
                relative_noise_sd=self.config.cgm_relative_noise_sd * self.config.noise_multiplier,
            )
        )
        self.cgm_state: CGMObservationState | None = None
        self.cgm_last_sample_time_min = 0.0
        self.cgm_dropped_count = 0
        self.cgm_delivered_count = 0
        self.current_time_min = 0.0

    @staticmethod
    def _truth(state, name: str) -> float:
        return float(getattr(state, name))

    def _dropout_probability(self, spec: MeasurementChannelSpec) -> float:
        if spec.group == "monitor":
            return max(spec.dropout_probability, self.config.monitor_dropout_probability)
        return spec.dropout_probability

    def _effective_spec(self, name: str, spec: MeasurementChannelSpec) -> MeasurementChannelSpec:
        if spec.group == "abg":
            return MeasurementChannelSpec(
                self.config.abg_interval_min,
                self.config.abg_result_delay_min,
                spec.noise_sd,
                spec.relative_noise_sd,
                spec.dropout_probability,
                spec.lower,
                spec.upper,
                spec.group,
            )
        if spec.group == "chemistry":
            return MeasurementChannelSpec(
                self.config.chemistry_interval_min,
                self.config.chemistry_result_delay_min,
                spec.noise_sd,
                spec.relative_noise_sd,
                spec.dropout_probability,
                spec.lower,
                spec.upper,
                spec.group,
            )
        if spec.group == "hemodynamic":
            return MeasurementChannelSpec(
                self.config.hemodynamic_interval_min,
                self.config.hemodynamic_result_delay_min,
                spec.noise_sd,
                spec.relative_noise_sd,
                spec.dropout_probability,
                spec.lower,
                spec.upper,
                spec.group,
            )
        return spec

    def _noisy(self, value: float, spec: MeasurementChannelSpec, rng: np.random.Generator) -> float:
        scale = self.config.noise_multiplier
        sd = scale * spec.noise_sd
        if spec.relative_noise_sd > 0.0:
            sd = float(np.hypot(sd, scale * spec.relative_noise_sd * max(abs(value), 1e-6)))
        out = float(value)
        if sd > 0.0:
            out += float(rng.normal(0.0, sd))
        if spec.lower is not None:
            out = max(float(spec.lower), out)
        if spec.upper is not None:
            out = min(float(spec.upper), out)
        return float(out)

    def initialize(self, state, rng: np.random.Generator) -> None:
        self.current_time_min = 0.0
        self.channels.clear()
        for name, base_spec in _BASE_CHANNELS.items():
            spec = self._effective_spec(name, base_spec)
            value = self._noisy(self._truth(state, name), spec, rng)
            self.channels[name] = _ChannelState(
                value=value,
                sample_time_min=0.0,
                next_sample_time_min=spec.sample_interval_min,
                delivered_count=1,
            )
        self.cgm_state = self.cgm_model.initialize(float(state.glucose_mg_dl), rng=rng)
        self.cgm_last_sample_time_min = 0.0
        self.cgm_dropped_count = 0
        self.cgm_delivered_count = 1

    def advance(
        self,
        state,
        time_min: float,
        dt_min: float,
        rng: np.random.Generator,
        *,
        report_cgm: bool = True,
    ) -> None:
        time_min = float(time_min)
        dt_min = float(dt_min)
        if not np.isfinite(time_min) or not np.isfinite(dt_min):
            raise ValueError("time_min and dt_min must be finite")
        if dt_min < 0.0:
            raise ValueError("dt_min must be nonnegative")
        if time_min + 1e-12 < self.current_time_min:
            raise ValueError("time_min must be monotonic")
        elapsed = time_min - self.current_time_min
        if abs(elapsed - dt_min) > 1e-9:
            raise ValueError(
                "time_min - previous time must equal dt_min; intermediate "
                "measurement truth cannot be reconstructed by backfilling"
            )
        self.current_time_min = time_min

        # The interstitial compartment evolves at each physiology substep, while
        # sensor dropout/noise is realized only once per agent transition.  This
        # lets scheduled laboratory samples use the state at their actual cadence
        # without turning the integration cadence into an artificial CGM cadence.
        assert self.cgm_state is not None
        if not report_cgm:
            alpha = 1.0 - np.exp(-dt_min / self.config.cgm_lag_tau_min) if dt_min > 0 else 0.0
            self.cgm_state.interstitial_glucose_mg_dl += alpha * (
                float(state.glucose_mg_dl) - self.cgm_state.interstitial_glucose_mg_dl
            )
        elif rng.random() >= self.config.cgm_dropout_probability:
            self.cgm_model.step(
                self.cgm_state,
                float(state.glucose_mg_dl),
                dt_min,
                rng=rng,
            )
            self.cgm_last_sample_time_min = time_min
            self.cgm_delivered_count += 1
        else:
            # Physiology still evolves internally; reporting is held last-value.
            alpha = 1.0 - np.exp(-dt_min / self.config.cgm_lag_tau_min) if dt_min > 0 else 0.0
            self.cgm_state.interstitial_glucose_mg_dl += alpha * (
                float(state.glucose_mg_dl) - self.cgm_state.interstitial_glucose_mg_dl
            )
            self.cgm_dropped_count += 1

        for name, base_spec in _BASE_CHANNELS.items():
            spec = self._effective_spec(name, base_spec)
            ch = self.channels[name]

            self._deliver_matured(ch, time_min)

            # State is known only at the end of this call. If several scheduled
            # instants were crossed, take one endpoint sample and record the
            # unresolved intermediate samples as skipped; never label current
            # truth with an earlier timestamp.
            due = 0
            while ch.next_sample_time_min <= time_min + 1e-12:
                due += 1
                ch.next_sample_time_min += spec.sample_interval_min
            if due:
                ch.skipped_count += max(0, due - 1)
                if rng.random() < self._dropout_probability(spec):
                    ch.dropped_count += 1
                else:
                    sample = self._noisy(self._truth(state, name), spec, rng)
                    sample_time = time_min
                    available = sample_time + spec.result_delay_min
                    if available <= time_min + 1e-12:
                        ch.value = sample
                        ch.sample_time_min = sample_time
                        ch.delivered_count += 1
                    else:
                        ch.pending_results.append(
                            _PendingResult(sample, sample_time, available)
                        )
            self._deliver_matured(ch, time_min)

    @staticmethod
    def _deliver_matured(ch: _ChannelState, time_min: float) -> None:
        while (
            ch.pending_results
            and ch.pending_results[0].available_time_min <= time_min + 1e-12
        ):
            result = ch.pending_results.popleft()
            ch.value = float(result.value)
            ch.sample_time_min = float(result.sample_time_min)
            ch.delivered_count += 1

    def measurement_value(self, name: str, state) -> float:
        if name == "sensor_glucose_mg_dl":
            assert self.cgm_state is not None
            return float(self.cgm_state.sensor_glucose_mg_dl)
        if name in self.AGE_NAMES:
            return float(self.group_ages().get(name, 0.0))
        ch = self.channels.get(name)
        if ch is not None:
            return float(ch.value)
        # Rare clinical observables not explicitly configured are returned from
        # state only in ideal/debug mode; realistic mode should avoid silent
        # ground-truth leakage, so reject them.
        raise KeyError(f"No realistic measurement channel configured for {name!r}")

    def group_ages(self) -> dict[str, float]:
        now = self.current_time_min
        ages: dict[str, list[float]] = {"monitor": [], "abg": [], "chemistry": [], "hemodynamic": []}
        for name, base_spec in _BASE_CHANNELS.items():
            spec = self._effective_spec(name, base_spec)
            ch = self.channels[name]
            ages.setdefault(spec.group, []).append(max(0.0, now - ch.sample_time_min))
        return {
            "cgm_measurement_age_min": max(0.0, now - self.cgm_last_sample_time_min),
            "monitor_measurement_age_min": max(ages.get("monitor") or [0.0]),
            "blood_gas_measurement_age_min": max(ages.get("abg") or [0.0]),
            "chemistry_measurement_age_min": max(ages.get("chemistry") or [0.0]),
            "hemodynamic_measurement_age_min": max(ages.get("hemodynamic") or [0.0]),
        }

    def diagnostics(self) -> dict:
        groups: dict[str, dict[str, int | float]] = {}
        channels: dict[str, dict[str, int | float | None]] = {}
        for name, base_spec in _BASE_CHANNELS.items():
            spec = self._effective_spec(name, base_spec)
            ch = self.channels[name]
            g = groups.setdefault(spec.group, {"dropped": 0, "delivered": 0})
            g["dropped"] += ch.dropped_count
            g["delivered"] += ch.delivered_count
            channels[name] = {
                "group": spec.group,
                "age_min": max(0.0, self.current_time_min - ch.sample_time_min),
                "sample_time_min": float(ch.sample_time_min),
                "next_sample_time_min": float(ch.next_sample_time_min),
                "pending": len(ch.pending_results),
                "next_result_time_min": (
                    float(ch.pending_results[0].available_time_min)
                    if ch.pending_results else None
                ),
                "dropped": int(ch.dropped_count),
                "skipped": int(ch.skipped_count),
                "delivered": int(ch.delivered_count),
            }
        groups["cgm"] = {
            "dropped": self.cgm_dropped_count,
            "delivered": self.cgm_delivered_count,
        }
        return {
            "time_min": float(self.current_time_min),
            "ages_min": self.group_ages(),
            "groups": groups,
            "channels": channels,
            "profile": "realistic",
        }
