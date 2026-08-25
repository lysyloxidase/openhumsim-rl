from __future__ import annotations

from collections import deque
from collections.abc import Mapping
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
    Bedside monitors and CGM are sampled on their own physical clocks, while
    ABG and chemistry panels are sampled intermittently and become available
    only after a result delay. Dropout holds the last available value, so the
    policy sees a partially observed process rather than ground-truth state.
    """

    monitor_dropout_probability: float = 0.01
    cgm_dropout_probability: float = 0.02
    cgm_sample_interval_min: float = 5.0
    abg_interval_min: float = 30.0
    abg_result_delay_min: float = 7.0
    abg_dropout_probability: float = 0.0
    chemistry_interval_min: float = 60.0
    chemistry_result_delay_min: float = 12.0
    chemistry_dropout_probability: float = 0.0
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
            "cgm_sample_interval_min",
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
        for name in (
            "monitor_dropout_probability",
            "cgm_dropout_probability",
            "abg_dropout_probability",
            "chemistry_dropout_probability",
        ):
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
        self.cgm_next_sample_time_min = 0.0
        self.cgm_dropped_count = 0
        self.cgm_skipped_count = 0
        self.cgm_delivered_count = 0
        self.current_time_min = 0.0

    @staticmethod
    def _truth(state, name: str) -> float:
        return float(getattr(state, name))

    def _dropout_probability(self, spec: MeasurementChannelSpec) -> float:
        if spec.group == "monitor":
            return max(spec.dropout_probability, self.config.monitor_dropout_probability)
        if spec.group == "abg":
            return max(spec.dropout_probability, self.config.abg_dropout_probability)
        if spec.group == "chemistry":
            return max(
                spec.dropout_probability,
                self.config.chemistry_dropout_probability,
            )
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
        self.cgm_next_sample_time_min = self.config.cgm_sample_interval_min
        self.cgm_dropped_count = 0
        self.cgm_skipped_count = 0
        self.cgm_delivered_count = 1

    def runtime_snapshot(self) -> dict:
        """Return a JSON-serializable copy of all mutable measurement state."""
        cgm_state = None
        if self.cgm_state is not None:
            cgm_state = {
                "interstitial_glucose_mg_dl": float(
                    self.cgm_state.interstitial_glucose_mg_dl
                ),
                "sensor_glucose_mg_dl": float(
                    self.cgm_state.sensor_glucose_mg_dl
                ),
            }
        return {
            "current_time_min": float(self.current_time_min),
            "channels": {
                name: {
                    "value": float(ch.value),
                    "sample_time_min": float(ch.sample_time_min),
                    "next_sample_time_min": float(ch.next_sample_time_min),
                    "pending_results": [
                        {
                            "value": float(result.value),
                            "sample_time_min": float(result.sample_time_min),
                            "available_time_min": float(result.available_time_min),
                        }
                        for result in ch.pending_results
                    ],
                    "dropped_count": int(ch.dropped_count),
                    "skipped_count": int(ch.skipped_count),
                    "delivered_count": int(ch.delivered_count),
                }
                for name, ch in self.channels.items()
            },
            "cgm_state": cgm_state,
            "cgm_last_sample_time_min": float(self.cgm_last_sample_time_min),
            "cgm_next_sample_time_min": float(self.cgm_next_sample_time_min),
            "cgm_dropped_count": int(self.cgm_dropped_count),
            "cgm_skipped_count": int(self.cgm_skipped_count),
            "cgm_delivered_count": int(self.cgm_delivered_count),
        }

    def restore_runtime_snapshot(self, snapshot: dict) -> None:
        """Restore a snapshot produced by :meth:`runtime_snapshot`."""
        if not isinstance(snapshot, Mapping):
            raise TypeError("measurement runtime snapshot must be a mapping")
        expected_top = {
            "current_time_min",
            "channels",
            "cgm_state",
            "cgm_last_sample_time_min",
            "cgm_next_sample_time_min",
            "cgm_dropped_count",
            "cgm_skipped_count",
            "cgm_delivered_count",
        }
        supplied_top = set(snapshot)
        if supplied_top != expected_top:
            raise ValueError(
                "measurement runtime fields do not match schema: "
                f"missing={sorted(expected_top - supplied_top)}, "
                f"extra={sorted(supplied_top - expected_top)}"
            )

        def finite_number(value, label: str) -> float:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{label} must be a finite number")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{label} must be a finite number") from exc
            if not np.isfinite(number):
                raise ValueError(f"{label} must be finite")
            return number

        def nonnegative_integer(value, label: str) -> int:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
            ):
                raise TypeError(f"{label} must be an integer")
            number = int(value)
            if number < 0:
                raise ValueError(f"{label} must be nonnegative")
            return number

        def bounded_channel_value(
            value,
            spec: MeasurementChannelSpec,
            label: str,
        ) -> float:
            number = finite_number(value, label)
            if spec.lower is not None and number < float(spec.lower):
                raise ValueError(
                    f"{label} must be at least {float(spec.lower):g}"
                )
            if spec.upper is not None and number > float(spec.upper):
                raise ValueError(
                    f"{label} must be at most {float(spec.upper):g}"
                )
            return number

        current_time = finite_number(
            snapshot["current_time_min"], "current_time_min"
        )
        if current_time < 0.0:
            raise ValueError("current_time_min must be nonnegative")
        channels_payload = snapshot["channels"]
        if not isinstance(channels_payload, Mapping):
            raise TypeError("measurement channels must be a mapping")
        expected_channels = set(_BASE_CHANNELS)
        supplied_channels = set(channels_payload)
        if supplied_channels != expected_channels:
            raise ValueError(
                "measurement channel set does not match schema: "
                f"missing={sorted(expected_channels - supplied_channels)}, "
                f"extra={sorted(supplied_channels - expected_channels)}"
            )

        expected_channel_fields = {
            "value",
            "sample_time_min",
            "next_sample_time_min",
            "pending_results",
            "dropped_count",
            "skipped_count",
            "delivered_count",
        }
        expected_pending_fields = {
            "value",
            "sample_time_min",
            "available_time_min",
        }
        restored_channels: dict[str, _ChannelState] = {}
        for name in _BASE_CHANNELS:
            spec = self._effective_spec(name, _BASE_CHANNELS[name])
            item = channels_payload[name]
            if not isinstance(item, Mapping):
                raise TypeError(f"measurement channel {name!r} must be a mapping")
            supplied_fields = set(item)
            if supplied_fields != expected_channel_fields:
                raise ValueError(
                    f"measurement channel {name!r} fields do not match schema: "
                    f"missing={sorted(expected_channel_fields - supplied_fields)}, "
                    f"extra={sorted(supplied_fields - expected_channel_fields)}"
                )
            value = bounded_channel_value(item["value"], spec, f"{name}.value")
            sample_time = finite_number(
                item["sample_time_min"], f"{name}.sample_time_min"
            )
            next_sample_time = finite_number(
                item["next_sample_time_min"], f"{name}.next_sample_time_min"
            )
            if sample_time < 0.0 or sample_time > current_time + 1e-12:
                raise ValueError(
                    f"{name}.sample_time_min must be within elapsed time"
                )
            if next_sample_time <= current_time + 1e-12:
                raise ValueError(
                    f"{name}.next_sample_time_min must be in the future"
                )
            interval = float(spec.sample_interval_min)
            schedule_tolerance = 1e-9 * max(
                1.0, current_time, next_sample_time, interval
            )
            if next_sample_time > current_time + interval + schedule_tolerance:
                raise ValueError(
                    f"{name}.next_sample_time_min skips a sampling interval"
                )
            grid_index = next_sample_time / interval
            if abs(grid_index - round(grid_index)) > 1e-9 * max(
                1.0, abs(grid_index)
            ):
                raise ValueError(
                    f"{name}.next_sample_time_min is off the sampling grid"
                )
            pending_payload = item["pending_results"]
            if not isinstance(pending_payload, list):
                raise TypeError(f"{name}.pending_results must be a list")
            pending_results: deque[_PendingResult] = deque()
            previous_sample_time = -1.0
            previous_available_time = -1.0
            for index, result in enumerate(pending_payload):
                if not isinstance(result, Mapping):
                    raise TypeError(
                        f"{name}.pending_results[{index}] must be a mapping"
                    )
                supplied_pending = set(result)
                if supplied_pending != expected_pending_fields:
                    raise ValueError(
                        f"{name}.pending_results[{index}] fields do not match schema"
                    )
                pending_value = bounded_channel_value(
                    result["value"],
                    spec,
                    f"{name}.pending_results[{index}].value",
                )
                pending_sample_time = finite_number(
                    result["sample_time_min"],
                    f"{name}.pending_results[{index}].sample_time_min",
                )
                available_time = finite_number(
                    result["available_time_min"],
                    f"{name}.pending_results[{index}].available_time_min",
                )
                expected_available_time = (
                    pending_sample_time + float(spec.result_delay_min)
                )
                delay_tolerance = 1e-9 * max(
                    1.0,
                    abs(available_time),
                    abs(expected_available_time),
                )
                if (
                    pending_sample_time < 0.0
                    or pending_sample_time > current_time + 1e-12
                    or pending_sample_time + 1e-12 < sample_time
                    or available_time < pending_sample_time
                    or available_time <= current_time + 1e-12
                    or abs(available_time - expected_available_time)
                    > delay_tolerance
                    or pending_sample_time + 1e-12 < previous_sample_time
                    or available_time + 1e-12 < previous_available_time
                ):
                    raise ValueError(
                        f"{name}.pending_results must be chronological, "
                        "sampled after the delivered result, available in the "
                        "future, and match the configured result delay"
                    )
                pending_results.append(
                    _PendingResult(
                        value=pending_value,
                        sample_time_min=pending_sample_time,
                        available_time_min=available_time,
                    )
                )
                previous_sample_time = pending_sample_time
                previous_available_time = available_time
            dropped_count = nonnegative_integer(
                item["dropped_count"], f"{name}.dropped_count"
            )
            skipped_count = nonnegative_integer(
                item["skipped_count"], f"{name}.skipped_count"
            )
            delivered_count = nonnegative_integer(
                item["delivered_count"], f"{name}.delivered_count"
            )
            if delivered_count < 1:
                raise ValueError(
                    f"{name}.delivered_count must include the initial sample"
                )
            restored_channels[str(name)] = _ChannelState(
                value=value,
                sample_time_min=sample_time,
                next_sample_time_min=next_sample_time,
                pending_results=pending_results,
                dropped_count=dropped_count,
                skipped_count=skipped_count,
                delivered_count=delivered_count,
            )
        restored_cgm = snapshot["cgm_state"]
        if not isinstance(restored_cgm, Mapping):
            raise TypeError("cgm_state must be an initialized mapping")
        expected_cgm_fields = {
            "interstitial_glucose_mg_dl",
            "sensor_glucose_mg_dl",
        }
        if set(restored_cgm) != expected_cgm_fields:
            raise ValueError("cgm_state fields do not match schema")
        interstitial_glucose = finite_number(
            restored_cgm["interstitial_glucose_mg_dl"],
            "cgm_state.interstitial_glucose_mg_dl",
        )
        sensor_glucose = finite_number(
            restored_cgm["sensor_glucose_mg_dl"],
            "cgm_state.sensor_glucose_mg_dl",
        )
        if interstitial_glucose < 0.0:
            raise ValueError(
                "cgm_state.interstitial_glucose_mg_dl must be nonnegative"
            )
        cgm_lower = float(self.cgm_model.config.lower_reportable_mg_dl)
        cgm_upper = float(self.cgm_model.config.upper_reportable_mg_dl)
        if not cgm_lower <= sensor_glucose <= cgm_upper:
            raise ValueError(
                "cgm_state.sensor_glucose_mg_dl must be within the configured "
                "reportable range"
            )
        cgm_last_sample_time = finite_number(
            snapshot["cgm_last_sample_time_min"], "cgm_last_sample_time_min"
        )
        cgm_next_sample_time = finite_number(
            snapshot["cgm_next_sample_time_min"], "cgm_next_sample_time_min"
        )
        if (
            cgm_last_sample_time < 0.0
            or cgm_last_sample_time > current_time + 1e-12
        ):
            raise ValueError("cgm_last_sample_time_min must be within elapsed time")
        if cgm_next_sample_time <= current_time + 1e-12:
            raise ValueError("cgm_next_sample_time_min must be in the future")
        cgm_interval = float(self.config.cgm_sample_interval_min)
        cgm_schedule_tolerance = 1e-9 * max(
            1.0, current_time, cgm_next_sample_time, cgm_interval
        )
        if (
            cgm_next_sample_time
            > current_time + cgm_interval + cgm_schedule_tolerance
        ):
            raise ValueError("cgm_next_sample_time_min skips a sampling interval")
        cgm_grid_index = cgm_next_sample_time / cgm_interval
        if abs(cgm_grid_index - round(cgm_grid_index)) > 1e-9 * max(
            1.0, abs(cgm_grid_index)
        ):
            raise ValueError("cgm_next_sample_time_min is off the sampling grid")
        cgm_dropped_count = nonnegative_integer(
            snapshot["cgm_dropped_count"], "cgm_dropped_count"
        )
        cgm_skipped_count = nonnegative_integer(
            snapshot["cgm_skipped_count"], "cgm_skipped_count"
        )
        cgm_delivered_count = nonnegative_integer(
            snapshot["cgm_delivered_count"], "cgm_delivered_count"
        )
        if cgm_delivered_count < 1:
            raise ValueError("cgm_delivered_count must include the initial sample")

        # Commit only after the complete snapshot has passed validation.
        self.channels = restored_channels
        self.cgm_state = CGMObservationState(
            interstitial_glucose_mg_dl=interstitial_glucose,
            sensor_glucose_mg_dl=sensor_glucose,
        )
        self.current_time_min = current_time
        self.cgm_last_sample_time_min = cgm_last_sample_time
        self.cgm_next_sample_time_min = cgm_next_sample_time
        self.cgm_dropped_count = cgm_dropped_count
        self.cgm_skipped_count = cgm_skipped_count
        self.cgm_delivered_count = cgm_delivered_count

    def advance(
        self,
        state,
        time_min: float,
        dt_min: float,
        rng: np.random.Generator,
        *,
        report_cgm: bool | None = None,
    ) -> None:
        """Advance all measurement clocks to ``time_min``.

        ``report_cgm`` is retained as a compatibility-only keyword. CGM reports
        are now governed exclusively by ``cgm_sample_interval_min`` so policy
        decision boundaries cannot change noise, dropout, or sample counts.
        """
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

        # The interstitial compartment evolves at the physiology cadence. Sensor
        # noise/dropout is realized only when the independent CGM clock is due.
        assert self.cgm_state is not None
        self.cgm_model.advance_interstitial(
            self.cgm_state,
            float(state.glucose_mg_dl),
            dt_min,
        )
        cgm_due = 0
        while self.cgm_next_sample_time_min <= time_min + 1e-12:
            cgm_due += 1
            self.cgm_next_sample_time_min += self.config.cgm_sample_interval_min
        if cgm_due:
            self.cgm_skipped_count += max(0, cgm_due - 1)
            cgm_dropout = self.config.cgm_dropout_probability
            if cgm_dropout > 0.0 and rng.random() < cgm_dropout:
                self.cgm_dropped_count += 1
            else:
                self.cgm_model.sample(self.cgm_state, rng=rng)
                # If a caller jumps across multiple scheduled instants, current
                # truth is only known at this endpoint; do not backdate it.
                self.cgm_last_sample_time_min = time_min
                self.cgm_delivered_count += 1

        effective_specs = {
            name: self._effective_spec(name, base_spec)
            for name, base_spec in _BASE_CHANNELS.items()
        }
        due_by_name: dict[str, int] = {}
        for name, spec in effective_specs.items():
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
            due_by_name[name] = due

        # ABG and chemistry are panel draws: all members share one collection
        # time and one panel-level dropout event. Channel-specific analytical
        # noise remains independent after a panel is successfully collected.
        panel_dropout: dict[str, bool] = {}
        for group in ("abg", "chemistry"):
            due_specs = [
                spec
                for name, spec in effective_specs.items()
                if spec.group == group and due_by_name[name]
            ]
            if due_specs:
                probability = max(self._dropout_probability(spec) for spec in due_specs)
                panel_dropout[group] = bool(
                    probability > 0.0 and rng.random() < probability
                )

        for name, spec in effective_specs.items():
            ch = self.channels[name]
            due = due_by_name[name]
            if due:
                ch.skipped_count += max(0, due - 1)
                if spec.group in panel_dropout:
                    dropped = panel_dropout[spec.group]
                else:
                    probability = self._dropout_probability(spec)
                    dropped = bool(
                        probability > 0.0 and rng.random() < probability
                    )
                if dropped:
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
            "skipped": self.cgm_skipped_count,
            "delivered": self.cgm_delivered_count,
            "next_sample_time_min": float(self.cgm_next_sample_time_min),
        }
        return {
            "time_min": float(self.current_time_min),
            "ages_min": self.group_ages(),
            "groups": groups,
            "channels": channels,
            "profile": "realistic",
        }
